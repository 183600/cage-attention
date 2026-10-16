"""Training and evaluation utilities."""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .data import (
    STATS_SEED,
    VAL_SEED,
    check_train_seed,
    make_batch,
    validate_batch,
)
from .metrics import (
    brier_score,
    expected_calibration_error,
    multiclass_metrics,
    pr_auc,
    roc_auc,
)


def _make_adamw(params, lr):
    """AdamW on the current device."""
    params = list(params)
    kw = dict(lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    if any(p.is_cuda for p in params):
        try:
            return torch.optim.AdamW(params, foreach=False, fused=True, **kw)
        except (TypeError, RuntimeError, ValueError):
            pass
    try:
        return torch.optim.AdamW(params, foreach=True, **kw)
    except (TypeError, RuntimeError, ValueError):
        pass
    return torch.optim.AdamW(params, **kw)


def training_curve_verdict(hist, rel_tol: float = 0.05):
    """Read the *curve*, not just the final number (over/underfit triage).

    ``hist`` is the list of dicts returned by :func:`train_model`.  The
    train and (optionally monitored) validation losses are reduced to a
    least-squares relative change over the logged window:
    ``delta = (fitted_last - fitted_first) / mean(loss)`` - scale-free and
    robust to the absolute loss level.  The verdict follows the standard
    learning-curve reading:

    * ``learning``            - train and val both fall;
    * ``overfitting``         - train falls while val *rises*: regularise,
                                add data, or early-stop (``best_val_step``
                                marks the early-stop point);
    * ``learning_gap``        - train falls, val flat: no new overfitting
                                yet but the gap widens - watch it;
    * ``plateau``             - both flat: underfitting, bad learning rate
                                or a broken data pipeline, not "converged";
    * ``diverging``           - train rises: unstable optimisation / lr too
                                high;
    * ``numerical_instability`` - a non-finite loss appeared anywhere;
    * ``insufficient_points`` - fewer than two finite points.

    Returns a dict with the verdict, the relative deltas, the best
    validation point and a human-readable ``recommendation``.
    """
    finite = [h for h in hist
              if math.isfinite(float(h.get("train_loss", float("nan"))))]
    unstable = len(finite) < len(hist) or any(
        h.get("val_loss") is not None
        and not math.isfinite(float(h["val_loss"])) for h in hist)
    out = {
        "verdict": "insufficient_points",
        "train_delta": float("nan"),
        "val_delta": float("nan"),
        "generalization_gap": float("nan"),
        "best_val_loss": float("nan"),
        "best_val_step": None,
        "n_points": len(finite),
        "recommendation": "not enough logged points to judge the curve",
    }
    if unstable:
        out["verdict"] = "numerical_instability"
        out["recommendation"] = (
            "a non-finite loss appeared; run the staged triage in "
            "diagnose_nonfinite (data -> activations -> loss -> gradients)")
    if len(finite) < 2:
        return out

    def rel_delta(points, key):
        xs = [float(p["step"]) for p in points]
        ys = [float(p[key]) for p in points]
        mean_y = sum(ys) / len(ys)
        if mean_y == 0.0:
            return 0.0
        n = len(xs)
        mx = sum(xs) / n
        my = mean_y
        denom = sum((x - mx) ** 2 for x in xs)
        if denom == 0.0:
            return 0.0
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
        return slope * (xs[-1] - xs[0]) / abs(my)

    out["train_delta"] = rel_delta(finite, "train_loss")
    val = [h for h in finite if h.get("val_loss") is not None
           and math.isfinite(float(h["val_loss"]))]
    if len(val) >= 2:
        out["val_delta"] = rel_delta(val, "val_loss")
        best = min(val, key=lambda h: h["val_loss"])
        out["best_val_loss"] = float(best["val_loss"])
        out["best_val_step"] = int(best["step"])
        common = [h for h in finite if h.get("val_loss") is not None]
        if common:
            out["generalization_gap"] = float(
                common[-1]["val_loss"] - common[-1]["train_loss"])

    td, vd = out["train_delta"], out["val_delta"]
    if unstable:
        pass
    elif td < -rel_tol:
        if math.isfinite(vd) and vd > rel_tol:
            out["verdict"] = "overfitting"
            out["recommendation"] = (
                "train loss falls while validation loss rises: add "
                "regularisation / data, or stop at best_val_step")
        elif math.isfinite(vd) and vd < -rel_tol:
            out["verdict"] = "learning"
            out["recommendation"] = (
                "both curves improve - the run is still learning; more "
                "steps or capacity may help")
        else:
            out["verdict"] = "learning_gap"
            out["recommendation"] = (
                "train improves but validation is flat: the gap widens "
                "without val regression yet; monitor or regularise")
    elif td > rel_tol:
        out["verdict"] = "diverging"
        out["recommendation"] = (
            "training loss rises: unstable optimisation - lower the "
            "learning rate or tighten gradient clipping")
    elif math.isfinite(vd) and vd < -rel_tol:
        out["verdict"] = "val_only_falling"
        out["recommendation"] = (
            "validation improves while train loss is flat - check the "
            "train loss computation/protocol before trusting it")
    elif math.isfinite(vd) and vd > rel_tol:
        out["verdict"] = "val_only_rising"
        out["recommendation"] = (
            "validation degrades while train loss is flat - check the "
            "evaluation pipeline for a distribution/protocol mismatch")
    else:
        out["verdict"] = "plateau"
        out["recommendation"] = (
            "neither curve moves: underfitting, wrong learning rate, or a "
            "data-pipeline problem - not convergence")
    return out


def diagnose_nonfinite(model, idx, tgt, meta, vocab, device="cpu",
                       grad_norm_explode: float = 100.0):
    """Staged triage for a non-finite loss (the standard debugging order).

    1. **Data**: NaN/Inf tokens or targets, structural/label violations
       (``validate_batch``);
    2. **Forward activations**: the first block whose output is non-finite,
       plus non-finite parameters and invalid calibrated temperatures
       (log(0)/div-by-zero/softmax-overflow surface here);
    3. **Loss/logits**: whether the scalar itself or the logits are bad;
    4. **Gradients**: non-finite or exploding (max abs / total norm) - the
       gradient-explosion / lr-too-large branch;
    5. **Precision**: if autocast (fp16/bfloat16) is active this is flagged,
       because low-precision overflow mimics all of the above.

    Runs its own forward/backward on the given batch and never touches
    ``param.grad`` (``torch.autograd.grad``), so it is safe to call from
    inside a training loop.  Returns a report dict; ``first_bad_stage``
    names the earliest failing stage.
    """
    report = {
        "data": "ok",
        "activations": "ok",
        "loss": "ok",
        "gradients": "ok",
        "first_bad_stage": None,
        "grad_total_norm": float("nan"),
        "grad_max_abs": float("nan"),
        "nonfinite_params": [],
        "notes": [],
        "suggestions": [],
    }

    def mark(stage, message):
        report[stage] = message
        if report["first_bad_stage"] is None:
            report["first_bad_stage"] = stage

    # ---- 1. data ---------------------------------------------------------
    try:
        validate_batch(idx, tgt, meta, vocab)
    except Exception as e:  # noqa: BLE001 - report, never raise from triage
        mark("data", f"invalid batch: {e}")
    if torch.is_tensor(idx) and idx.is_floating_point() \
            and not bool(torch.isfinite(idx).all()):
        mark("data", "idx contains NaN/Inf")
    if torch.is_tensor(tgt) and tgt.is_floating_point() \
            and not bool(torch.isfinite(tgt).all()):
        mark("data", "targets contain NaN/Inf")

    # ---- 2. forward activations -----------------------------------------
    bad_layers = []
    handles = []

    def _make_hook(layer_idx):
        def _hook(module, inputs, output):
            outs = output if isinstance(output, tuple) else (output,)
            for o in outs:
                if torch.is_tensor(o) and o.is_floating_point() and not bool(
                        torch.isfinite(o).all()):
                    bad_layers.append(layer_idx)
                    return
        return _hook

    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(_make_hook(i)))
    was_training = model.training
    model.train()
    logits = loss = None
    sup_pos = meta.get("query_pos") if isinstance(meta, dict) else None
    try:
        with torch.inference_mode(False), torch.enable_grad():
            logits, loss, _ = model(idx, tgt, sup_pos=sup_pos)
    except Exception as e:  # noqa: BLE001
        mark("activations", f"forward raised: {e}")
    finally:
        for hd in handles:
            hd.remove()
        model.train(was_training)
    if bad_layers:
        mark("activations",
             f"non-finite block output at layer(s) {sorted(set(bad_layers))} "
             f"(check log(0)/div-by-zero/softmax overflow there)")
    for name, p in model.named_parameters():
        if not bool(torch.isfinite(p).all()):
            report["nonfinite_params"].append(name)
    if report["nonfinite_params"]:
        mark("activations",
             f"non-finite parameter(s): {report['nonfinite_params'][:4]}")
    for i, blk in enumerate(model.blocks):
        tau = getattr(blk.attn, "calib_tau", None)
        if tau is not None:
            eff = float(torch.nn.functional.softplus(tau.detach())) + 0.1
            if not math.isfinite(eff) or eff <= 0.0:
                report["notes"].append(
                    f"layer {i}: effective calib temperature {eff} invalid")

    # ---- 3. loss / logits ------------------------------------------------
    loss_ok = torch.is_tensor(loss) and bool(torch.isfinite(loss))
    logits_ok = torch.is_tensor(logits) and bool(torch.isfinite(logits.float()).all())
    if not logits_ok:
        mark("loss", "logits contain NaN/Inf")
    if not loss_ok:
        val = "None" if loss is None else f"{float(loss)}"
        mark("loss", f"non-finite loss (value {val})")

    # ---- 4. gradients ----------------------------------------------------
    seen: set[int] = set()
    params, names = [], []
    for name, p in model.named_parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            params.append(p)
            names.append(name)
    if loss_ok:
        try:
            grads = torch.autograd.grad(loss, params, allow_unused=True,
                                        retain_graph=False)
            bad = [n for n, g in zip(names, grads)
                   if g is not None and not bool(torch.isfinite(g).all())]
            if bad:
                mark("gradients", f"non-finite gradient(s): {bad[:4]}")
            else:
                max_abs = max((float(g.abs().max()) for g in grads
                               if g is not None), default=0.0)
                total = math.sqrt(sum(float(g.pow(2).sum()) for g in grads
                                      if g is not None))
                report["grad_max_abs"] = max_abs
                report["grad_total_norm"] = total
                if total > grad_norm_explode:
                    mark("gradients",
                         f"gradients explode (total norm {total:.3g} > "
                         f"{grad_norm_explode:g}): lower the lr or tighten "
                         f"gradient clipping")
        except Exception as e:  # noqa: BLE001
            mark("gradients", f"backward raised: {e}")
    else:
        report["gradients"] = "skipped (loss is non-finite)"

    # ---- 5. precision ----------------------------------------------------
    autocast_on = False
    if hasattr(torch, "is_autocast_enabled"):
        autocast_on = bool(torch.is_autocast_enabled())
    if torch.cuda.is_available() and hasattr(torch, "is_autocast_cuda_enabled"):
        autocast_on = autocast_on or bool(torch.is_autocast_cuda_enabled())
    if autocast_on:
        report["notes"].append(
            "autocast (fp16/bfloat16) is active: check low-precision "
            "overflow before blaming the architecture")

    order = ("data", "activations", "loss", "gradients")
    stage = report["first_bad_stage"]
    if stage is None:
        report["suggestions"].append(
            "all stages finite on this batch: the non-finite value was "
            "transient (re-check the batch/step that produced it and any "
            "runtime precision switches)")
    elif stage == "data":
        report["suggestions"].append(
            "fix the data pipeline first: NaN/Inf or invalid labels poison "
            "every downstream stage")
    else:
        idx_stage = order.index(stage)
        if idx_stage >= order.index("activations"):
            report["suggestions"].append(
                "inspect log(0)/division and softmax inputs (max-subtract "
                "before exp) in the flagged layers")
        if idx_stage >= order.index("gradients"):
            report["suggestions"].append(
                "reduce lr / rely on gradient clipping; check for rare "
                "exploding batches")
        report["suggestions"].append(
            "if the flagged stage is only intermittently bad, suspect "
            "fp16 autocast overflow or a rare bad batch before rewriting "
            "the kernel")
    return report


def train_model(model, steps=1500, batch=64, n_pairs=16, lr=3e-4, vocab=512,
                device="cpu", log_every=250, seed=0, dense_lm=False,
                eval_every=0, val_batches=2, early_stop_patience=None,
                nonfinite="raise"):
    """Train on the recall task with optional learning-curve monitoring.

    ``seed`` governs *all* run-to-run randomness: the data generator and the
    global RNG that draws the attention/MLP dropout masks.  Without the
    latter two runs of the same config would silently differ.

    ``seed`` must not collide with a held-out evaluation stream
    (``cage.data.RESERVED_EVAL_SEEDS``); the first batch of the run is
    validated as a data-cleaning gate before any gradient step.

    Curve monitoring (``eval_every > 0``): every ``eval_every`` steps the
    model is scored on ``val_batches`` fresh batches drawn from the fixed
    validation stream (``cage.data.VAL_SEED``) - the same batches at every
    checkpoint, and never the reported test stream.  Validation forwards
    run in eval mode, so they draw *no* RNG and do not perturb the training
    trajectory.  ``hist`` entries are dicts
    ``{"step", "train_loss"[, "val_loss", "val_acc"]}``  to be fed to
    :func:`training_curve_verdict` (monitoring the curve instead of only
    the final number is what catches over/underfitting).

    ``early_stop_patience`` (opt-in, default off): stop once the monitored
    validation loss has not improved for that many consecutive evaluations.
    Off by default because the experiment contract compares fixed-budget
    runs; ``training_curve_verdict`` reports ``best_val_step`` either way.

    ``nonfinite`` (``"raise"`` | ``"skip"``): what to do when a training
    loss is not finite.  ``"raise"`` (default) runs
    :func:`diagnose_nonfinite` and aborts with the staged report - silently
    stepping an optimizer with NaN gradients corrupts the run while
    producing plausible-looking final numbers.  ``"skip"`` records the
    diagnosis in ``hist`` and skips the update instead.
    """
    check_train_seed(seed)
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    val_gen = torch.Generator(device=device).manual_seed(VAL_SEED)
    params = list(model.parameters())
    opt = _make_adamw(params, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
    hist = []
    best_val = float("inf")
    best_val_step = None
    evals_since_improve = 0
    model.train()
    for step in range(steps):
        idx, tgt, meta = make_batch(batch, n_pairs, vocab, device, dense_lm=dense_lm, gen=gen)
        if step == 0:
            validate_batch(idx, tgt, meta, vocab)
        opt.zero_grad(set_to_none=True)
        sup_pos = None if dense_lm else meta["query_pos"]
        _, loss, _ = model(idx, tgt, sup_pos=sup_pos)
        lv = loss.item()
        entry = {"step": step, "train_loss": lv}
        if not math.isfinite(lv):
            diag = diagnose_nonfinite(model, idx, tgt, meta, vocab, device=device)
            entry["diagnosis"] = diag
            print(f"  step {step:5d}  loss {lv}  <- non-finite; triage "
                  f"first_bad_stage={diag['first_bad_stage']}")
            hist.append(entry)
            opt.zero_grad(set_to_none=True)
            sched.step()
            if nonfinite == "raise":
                raise RuntimeError(
                    f"non-finite training loss {lv} at step {step}; staged "
                    f"triage first_bad_stage={diag['first_bad_stage']!r}, "
                    f"suggestions={diag['suggestions']}")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        is_log = (log_every > 0 and step % log_every == 0) or step == steps - 1
        do_val = (eval_every > 0 and (step % eval_every == 0 or step == steps - 1))
        if do_val:
            val_loss, val_acc = _validation_metrics(
                model, val_gen, n_pairs, vocab, device, batch, val_batches,
                dense_lm, validate=step == 0)
            entry["val_loss"] = val_loss
            entry["val_acc"] = val_acc
            if val_loss < best_val:
                best_val, best_val_step = val_loss, step
                evals_since_improve = 0
            else:
                evals_since_improve += 1
        if is_log or do_val:
            hist.append(entry)
        if is_log:
            line = f"  step {step:5d}  loss {lv:.4f}"
            if "val_loss" in entry:
                line += (f"  val_loss {entry['val_loss']:.4f}  "
                         f"val_acc {entry['val_acc']:.4f}")
            print(line)
        if (early_stop_patience is not None and do_val
                and evals_since_improve >= early_stop_patience):
            print(f"  early stop at step {step}: val loss has not improved "
                  f"for {evals_since_improve} evaluations "
                  f"(best {best_val:.4f} @ step {best_val_step})")
            break
    return hist


def _validation_metrics(model, val_gen, n_pairs, vocab, device, batch, batches,
                        dense_lm, validate=False):
    """Validation loss (and accuracy) on fixed, held-out validation batches.

    Uses eval mode + ``no_grad`` (no dropout, no RNG consumption), so the
    training RNG stream - and therefore the training trajectory - is
    unaffected by monitoring.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    val_gen.manual_seed(VAL_SEED)
    with torch.no_grad():
        for i in range(max(1, batches)):
            idx, tgt, meta = make_batch(batch, n_pairs, vocab, device,
                                        dense_lm=dense_lm, gen=val_gen)
            if validate and i == 0:
                validate_batch(idx, tgt, meta, vocab)
            sup_pos = None if dense_lm else meta["query_pos"]
            logits, loss, _ = model(idx, tgt, sup_pos=sup_pos)
            total_loss += float(loss)
            if not dense_lm:
                correct += int((logits.argmax(-1) == meta["answer"]).sum())
            total += idx.size(0)
    model.train(was_training)
    return total_loss / max(1, batches), (correct / total if total else float("nan"))


def estimate_forward_flops(model, seq_len: int) -> int:
    """Matmul-only FLOPs of one full-sequence forward pass at ``seq_len``.

    Counts ``2 * m * n * k`` per matmul (a MAC is two FLOPs) for q/k/v, the
    causal attention products, the output projection, both MLP projections,
    the gate projection of ``query``/``calib`` and the two per-token scale
    projections of ``seednorm``, plus the full-sequence head.  Elementwise
    work (softmax, GELU, entropy, norms) is excluded: this is a lower bound
    for *relative* cost comparison across variants, and the honest reading
    of any accuracy delta needs this denominator next to it.
    """
    cfg = model.cfg
    C, H, L, V = cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.vocab_size
    T = int(seq_len)
    per_layer = (
        2 * T * C * (3 * C)          # qkv projection
        + T * (T + 1) * C            # attention scores (q k^T, causal)
        + T * (T + 1) * C            # attention output (p v)
        + 2 * T * C * C              # output projection
        + 2 * T * C * (4 * C)        # MLP up
        + 2 * T * (4 * C) * C        # MLP down
    )
    if cfg.gate in ("query", "calib"):
        per_layer += 2 * T * C * C   # gate projection
    if cfg.norm == "seednorm":
        per_layer += 4 * T * C * C   # two dynamic scale projections
    return int(L * per_layer + 2 * T * C * V)


def measure_inference_cost(model, n_pairs, vocab, device, batch=64, warmup=3,
                           runs=10, seed=STATS_SEED):
    """Parameters / FLOPs / latency / memory for the accuracy-cost trade-off.

    Latency is the median of ``runs`` eval-mode forward passes after
    ``warmup`` (batch ``batch`` at ``n_pairs``; fixed seed so every variant
    scores the same shapes).  Memory is the CUDA peak allocation when on
    GPU; on CPU it is the profiler-attributed allocation total of one pass
    (a deterministic, same-protocol *proxy*, not a peak-RSS measurement -
    compare variants to each other, never to an absolute budget).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    idx, _, _ = make_batch(batch, n_pairs, vocab, device, gen=gen)
    was_training = model.training
    model.eval()
    times = []
    mem_mb = float("nan")
    mem_method = "unavailable"
    try:
        with torch.inference_mode():
            for _ in range(max(0, warmup)):
                model(idx)
            for _ in range(max(1, runs)):
                t0 = time.perf_counter()
                model(idx)
                times.append(time.perf_counter() - t0)
            if idx.is_cuda:
                torch.cuda.reset_peak_memory_stats()
                model(idx)
                mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                mem_method = "cuda_peak"
            else:
                try:
                    from torch.profiler import ProfilerActivity, profile
                    with profile(activities=[ProfilerActivity.CPU],
                                 profile_memory=True) as prof:
                        model(idx)
                    mem_mb = prof.key_averages().total_average(
                    ).cpu_memory_usage / (1024 * 1024)
                    mem_method = "cpu_profiler_alloc_sum"
                except Exception:  # noqa: BLE001 - measurement must not fail
                    pass
    finally:
        model.train(was_training)
    times.sort()
    median = times[len(times) // 2]
    params_total = sum(p.numel() for p in model.parameters())
    embed = getattr(model, "tok", None)
    params_non_embedding = params_total - (
        embed.weight.numel() if embed is not None else 0)
    seq_len = idx.size(1)
    return {
        "seq_len": int(seq_len),
        "batch": int(batch),
        "params_total": int(params_total),
        "params_non_embedding": int(params_non_embedding),
        "flops_per_forward": estimate_forward_flops(model, seq_len),
        "latency_ms_per_batch": median * 1e3,
        "latency_ms_per_sample": median * 1e3 / batch,
        "peak_mem_mb": mem_mb,
        "mem_method": mem_method,
    }


@torch.inference_mode()
def collect_error_cases(model, n_pairs, vocab, device, batch=128, batches=4,
                        seed=STATS_SEED, max_cases=64):
    """Bad-case mining: dump and taxonomise the samples the model gets wrong.

    Error analysis is the main source of new hypotheses, so each error is
    recorded with the evidence needed to classify *why* it failed:

    * ``confidence`` / ``margin``   - confidently wrong (systematic blind
      spot) vs low-confidence (ambiguous for the model);
    * ``attn_mass_on_key``          - mean attention mass on the correct
      key position in the last block: did the model fail to *find* the
      right key, or did it find it and then *read out* the wrong value?
    * ``pred_in_context``           - prediction is one of the visible
      values (retrieval confusion) vs an out-of-context token (readout
      failure);
    * ``naive_confusable``          - the query token appears as a later
      value, so a copy-last heuristic would fail (task-structural
      ambiguity rather than a model blind spot);
    * ``value_repeats``             - the answer value occurs more than
      once in the context (harder even for a perfect retriever);
    * ``key_pair_idx``              - position of the matching pair;
      aggregated into early/mid/late error rates (long-range retrieval).

    Labels cannot be the cause here: the task is synthetic and
    ``validate_batch`` verifies query-key uniqueness and the paired value
    on every batch (``label_noise_rate`` is 0 by construction).

    Returns ``{"cases": [...], "summary": {...}}``; cases are capped at
    ``max_cases`` but the summary aggregates every evaluated sample.
    """
    cfg = model.cfg
    C, H = cfg.d_model, cfg.n_heads
    dk = C // H
    inv_scale = dk ** -0.5
    gen = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()

    per_layer_mass: dict[int, torch.Tensor] = {}
    state: dict = {"meta": None}
    handles = []

    def _make_hook(layer_idx, null_k):
        def _hook(module, inputs):
            meta = state["meta"]
            if meta is not None:
                per_layer_mass[layer_idx] = _attention_mass_on_key(
                    inputs[0], module.qkv.weight, meta, H, inv_scale, null_k)
        return _hook

    for i, blk in enumerate(model.blocks):
        handles.append(blk.attn.register_forward_pre_hook(
            _make_hook(i, getattr(blk.attn, "null_k", None))))

    n_total = n_err = 0
    case_count = 0
    cases = []
    flags_summary = {"high": 0, "low": 0, "in_ctx": 0, "naive": 0, "repeat": 0}
    mass_err = mass_err_n = 0.0
    mass_corr = mass_corr_n = 0.0
    tertile_n = [0, 0, 0]
    tertile_err = [0, 0, 0]
    key_wrong_total = 0.0
    key_wrong_n = 0
    key_right_total = 0.0
    key_right_n = 0
    try:
        for _ in range(batches):
            idx, _, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
            per_layer_mass.clear()
            state["meta"] = meta
            logits, _, _ = model(idx)
            state["meta"] = None
            layered = torch.stack(
                [per_layer_mass[i] for i in sorted(per_layer_mass)], dim=0)
            mass_last, mass_mean = layered[-1], layered.mean(0)
            probs = logits.squeeze(1).softmax(-1)
            conf, pred = probs.max(-1)
            top2 = probs.topk(2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            answer = meta["answer"]
            wrong = pred != answer

            n_total += idx.size(0)
            n_err += int(wrong.sum())
            key_pair_idx = (meta["key_pos"] // 2).clamp(max=n_pairs - 1)
            for b in range(idx.size(0)):
                third = int(key_pair_idx[b].item() * 3 // max(1, n_pairs))
                third = min(2, third)
                tertile_n[third] += 1
                tertile_err[third] += int(wrong[b])
            m_e = mass_last[wrong]
            m_c = mass_last[~wrong]
            mass_err += float(m_e.sum())
            mass_err_n += int(m_e.numel())
            mass_corr += float(m_c.sum())
            mass_corr_n += int(m_c.numel())

            flags = _error_case_flags(idx, meta, pred, n_pairs)
            flags_summary["high"] += int((wrong & (conf >= 0.9)).sum())
            flags_summary["low"] += int((wrong & (conf < 0.5)).sum())
            flags_summary["in_ctx"] += int((wrong & flags["pred_in_context"]).sum())
            flags_summary["naive"] += int((wrong & flags["naive_confusable"]).sum())
            flags_summary["repeat"] += int((wrong & flags["value_repeats"]).sum())
            key_right = mass_last >= 0.5
            key_wrong_total += float(mass_last[wrong & ~key_right].sum())
            key_wrong_n += int((wrong & ~key_right).sum())
            key_right_total += float(mass_last[wrong & key_right].sum())
            key_right_n += int((wrong & key_right).sum())

            if case_count < max_cases:
                rows = torch.nonzero(wrong, as_tuple=False).flatten().tolist()
                for b in rows:
                    if case_count >= max_cases:
                        break
                    cases.append({
                        "context": [int(t) for t in idx[b].tolist()],
                        "query_pos": int(meta["query_pos"][b]),
                        "query_key": int(idx[b, meta["query_pos"][b]]),
                        "key_pos": int(meta["key_pos"][b]),
                        "key_pair_idx": int(key_pair_idx[b]),
                        "answer": int(answer[b]),
                        "pred": int(pred[b]),
                        "confidence": float(conf[b]),
                        "margin": float(margin[b]),
                        "attn_mass_on_key": float(mass_last[b]),
                        "attn_mass_on_key_mean_layers": float(mass_mean[b]),
                        "pred_in_context": bool(flags["pred_in_context"][b]),
                        "naive_confusable": bool(flags["naive_confusable"][b]),
                        "value_repeats": bool(flags["value_repeats"][b]),
                    })
                    case_count += 1
    finally:
        for hd in handles:
            hd.remove()
        model.train(was_training)

    def _share(count):
        return count / n_err if n_err else float("nan")

    summary = {
        "n_samples": int(n_total),
        "n_errors": int(n_err),
        "error_rate": (n_err / n_total) if n_total else float("nan"),
        "label_noise_rate": 0.0,
        "share_high_conf": _share(flags_summary["high"]),
        "share_low_conf": _share(flags_summary["low"]),
        "share_pred_in_context": _share(flags_summary["in_ctx"]),
        "share_naive_confusable": _share(flags_summary["naive"]),
        "share_value_repeats": _share(flags_summary["repeat"]),
        "attn_mass_on_key_error": (mass_err / mass_err_n) if mass_err_n else float("nan"),
        "attn_mass_on_key_correct": (mass_corr / mass_corr_n) if mass_corr_n else float("nan"),
        "error_rate_key_early": (tertile_err[0] / tertile_n[0]) if tertile_n[0] else float("nan"),
        "error_rate_key_mid": (tertile_err[1] / tertile_n[1]) if tertile_n[1] else float("nan"),
        "error_rate_key_late": (tertile_err[2] / tertile_n[2]) if tertile_n[2] else float("nan"),
        "errors_key_missed_share": ((key_wrong_n) / n_err) if n_err else float("nan"),
        "attn_mass_on_key_error_key_hit": (
            key_right_total / key_right_n) if key_right_n else float("nan"),
        "attn_mass_on_key_error_key_missed": (
            key_wrong_total / key_wrong_n) if key_wrong_n else float("nan"),
    }
    summary["attn_mass_gap"] = summary["attn_mass_on_key_correct"] - \
        summary["attn_mass_on_key_error"]
    return {"cases": cases, "summary": summary}


def _attention_mass_on_key(x, weight, meta, n_heads, inv_scale, null_k):
    """Attention mass the query puts on the *correct* key, per sample.

    Read-only: the attention input ``x`` and the shared ``qkv`` weight are
    taken from a pre-forward hook and q/k are recomputed with the *same*
    ``F.linear`` call the fixed kernel uses, so the values are identical
    to the real attention while the implementation stays untouched.  For
    ``null`` gates the learnable null slot is included in the softmax
    exactly as the kernel does, keeping the mass comparable across
    variants.
    """
    B = x.size(0)
    C = weight.size(0) // 3
    dk = C // n_heads
    q_all = F.linear(x[torch.arange(B, device=x.device), meta["query_pos"]],
                     weight)[:, :C]
    mass = torch.zeros(B, device=x.device, dtype=torch.float32)
    for b in range(B):
        qp, kp = int(meta["query_pos"][b]), int(meta["key_pos"][b])
        qb = q_all[b].view(n_heads, dk)
        kb = F.linear(x[b, :qp + 1], weight)[:, C:2 * C].view(qp + 1, n_heads, dk)
        scores = torch.einsum("hd,thd->th", qb, kb) * inv_scale
        if null_k is not None:
            s0 = (qb * null_k).sum(-1) * inv_scale
            scores = torch.cat([s0.unsqueeze(0), scores], dim=0)
            mass[b] = torch.softmax(scores, dim=0)[kp + 1].mean()
        else:
            mass[b] = torch.softmax(scores, dim=0)[kp].mean()
    return mass


def _error_case_flags(idx, meta, pred, n_pairs):
    """Per-sample structural flags used to classify an error's cause."""
    T = idx.size(1)
    pos = torch.arange(T, device=idx.device).view(1, -1)
    visible = pos < meta["query_pos"].unsqueeze(1)
    query = idx.gather(1, meta["query_pos"].unsqueeze(1)).squeeze(1)
    answer = meta["answer"]
    vals = idx[:, 1::2]
    match = (idx == query.unsqueeze(1)) & visible
    last = torch.where(match, pos, torch.full_like(pos, -1)).max(dim=1).values
    return {
        "naive_confusable": last != meta["key_pos"],
        "value_repeats": (vals == answer.unsqueeze(1)).sum(dim=1) > 1,
        "pred_in_context": (vals == pred.unsqueeze(1)).any(dim=1),
    }


@torch.inference_mode()
def eval_accuracy(model, n_pairs, vocab, device, batch=256, batches=4,
                  seed=VAL_SEED):
    """Recall accuracy at the query position."""
    gen = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for _ in range(batches):
        idx, _, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        logits, _, _ = model(idx)
        pred = logits.squeeze(1).argmax(-1)
        correct += int((pred == meta["answer"]).sum())
        total += idx.size(0)
    model.train(was_training)
    return correct / total


@torch.inference_mode()
def eval_metrics(model, n_pairs, vocab, device, batch=256, batches=4,
                 seed=VAL_SEED, topk=5):
    """Full metric panel at the query position.

    Same evaluation protocol as ``eval_accuracy`` (eval mode restored
    afterwards, independent fixed seed) extended with the explicitly
    named F1 variants, top-k accuracy, NLL and calibration metrics for the
    answer distribution.  ``seed`` defaults to the same fixed 123 as
    ``eval_accuracy`` so the two functions score identical batches;
    experiment scripts pass an explicit *validation* seed for model
    selection and a separate *test* seed for the reported numbers.

    The recall task is balanced multi-class, so ``accuracy`` is the
    primary metric; ``f1_micro`` equals it by construction and the
    macro/weighted variants guard against value-frequency skew.  See
    ``cage/metrics.py`` for the averaging conventions.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    logits_all, tgt_all = [], []
    for _ in range(batches):
        idx, _, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        logits, _, _ = model(idx)
        logits_all.append(logits.squeeze(1).float().cpu())
        tgt_all.append(meta["answer"].cpu())
    model.train(was_training)
    logits_cat = torch.cat(logits_all, dim=0)
    tgt_cat = torch.cat(tgt_all, dim=0)
    out = multiclass_metrics(logits_cat, tgt_cat, topk=topk)
    out["nll"] = float(torch.nn.functional.cross_entropy(logits_cat, tgt_cat))
    probs = logits_cat.softmax(dim=-1)
    confidence, pred = probs.max(dim=-1)
    out["ece_15bin"] = expected_calibration_error(confidence, pred == tgt_cat,
                                                  n_bins=15)
    out["brier"] = brier_score(confidence, pred == tgt_cat)
    return out


@torch.inference_mode()
def collect_stats(model, n_pairs, vocab, device, batch=128, batches=4,
                  seed=STATS_SEED):
    """Collect per-layer gate / entropy / maxprob / null-mass at query positions,
    plus false-concentration labels (high max-prob but wrong answer).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    out = {"gate": [], "entropy": [], "maxprob": [], "null_mass": [],
           "correct": [], "resid_norm": []}
    for _ in range(batches):
        idx, _, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        logits, _, layer_stats = model(idx, stat_pos=meta["query_pos"])
        pred = logits.squeeze(1).argmax(-1)
        out["correct"].append((pred == meta["answer"]).float())
        ents, maxps, nulls, gates, rns = [], [], [], [], []
        for s in layer_stats:
            ents.append(s["entropy"].mean(1))
            maxps.append(s["maxprob"].max(1)[0])
            rns.append(s["resid_norm"])
            if s["gate"] is not None:
                gates.append(s["gate"].mean(-1))
            if s["null_mass"] is not None:
                nulls.append(s["null_mass"].mean(1))
        out["entropy"].append(torch.stack(ents, 1))
        out["maxprob"].append(torch.stack(maxps, 1))
        out["resid_norm"].append(torch.stack(rns, 1))
        if gates:
            out["gate"].append(torch.stack(gates, 1))
        if nulls:
            out["null_mass"].append(torch.stack(nulls, 1))
    stats = {k: torch.cat(v).cpu() for k, v in out.items() if v}
    model.train(was_training)
    return stats


def gate_depth_ratio(stats, n_early=2):
    """Deep/early gate ratio: mean gate of the deepest layer over the mean of
    the shallowest ``n_early`` layers (E3).

    ``> 1`` means the collaborative normalisation made the deep layers more
    willing to write into the residual stream, i.e. it worked against the
    "deep layers contribute too little" symptom of Pre-LN.
    """
    g = stats.get("gate")
    if g is None or g.size(1) <= n_early:
        return float("nan")
    early = g[:, :n_early].mean()
    deep = g[:, -1].mean()
    return (deep / early).item()


def pearson(x, y):
    x = x.flatten().float()
    y = y.flatten().float()
    dx = x - x.mean()
    dy = y - y.mean()
    if (dx.abs().max() <= 1e-6 * x.abs().max().clamp_min(1.0)
            or dy.abs().max() <= 1e-6 * y.abs().max().clamp_min(1.0)):
        return torch.tensor(float("nan"))
    return (dx * dy).sum() / (dx.norm() * dy.norm())


def _fisher_mean(rs):
    """Mean of Pearson coefficients, weighted correctly through the
    Fisher z-transform; degenerate (NaN) layers are dropped.
    """
    rs = [r for r in rs if r == r]
    if not rs:
        return float("nan")
    return math.tanh(sum(math.atanh(min(max(r, -0.999999), 0.999999))
                         for r in rs) / len(rs))


def gate_stat_correlations(stats):
    """Correlations between gate value and attention entropy / max-prob /
    correctness, computed *within* each layer and averaged over layers.

    The statistics carry a sample axis and a layer axis.  Flattening both
    into one Pearson r lets the between-layer offsets (deep and shallow
    layers live at different operating points) dominate the between-sample
    variation, so the number answers "are deep gates smaller than shallow
    ones" instead of "does a given layer's gate track that same layer's
    attention statistics".  Per-layer r, aggregated with a Fisher
    z-transform, is the quantity the gate hypothesis is about.
    """
    r = {}
    g = stats.get("gate")
    if g is None:
        return r
    ent, mp, corr = stats["entropy"], stats["maxprob"], stats["correct"]
    cols = range(g.size(1))
    r["gate_vs_entropy"] = _fisher_mean(
        [pearson(g[:, i], ent[:, i]).item() for i in cols])
    r["gate_vs_maxprob"] = _fisher_mean(
        [pearson(g[:, i], mp[:, i]).item() for i in cols])
    r["gate_vs_correct"] = _fisher_mean(
        [pearson(g[:, i], corr).item() for i in cols])
    return r


def false_concentration_analysis(stats, maxprob_thresh=0.5):
    """'False concentration': max attention prob high but answer wrong.

    Computes, among high-maxprob samples, the error rate (false-concentration
    rate) and, for gated models, mean gate value on false-concentrated vs
    correct samples - a good calibrated gate should close (lower value)
    when concentration is false.
    """
    mp = stats["maxprob"].max(dim=1)[0]
    high = mp > maxprob_thresh
    corr = stats["correct"].bool()
    r = {
        "n_high_maxprob": int(high.sum()),
        "false_conc_rate": 1.0 - corr[high].float().mean().item() if high.any() else float("nan"),
        "overall_acc": corr.float().mean().item(),
    }
    g = stats.get("gate")
    if g is not None:
        gm = g.mean(1)
        fc = high & ~corr
        r["gate_on_false_conc"] = gm[fc].mean().item() if fc.any() else float("nan")
        r["gate_on_correct"] = gm[corr & high].mean().item() if (corr & high).any() else float("nan")
    return r


def error_detection_aucs(stats):
    """ROC-AUC and PR-AUC of attention statistics as *error detectors*.

    The positive class is "model answered wrong".  Scores are oriented so
    that **higher means more likely wrong** (``-maxprob``, mean entropy,
    ``-gate``), matching the mechanism hypotheses: concentration should
    track correctness, and a calibrated gate should close when it does
    not.  Errors are the rare class after convergence, so PR-AUC is the
    primary summary; ROC-AUC is reported for completeness only (with a
    huge negative population it can look optimistic while precision
    collapses; see ``cage/metrics.py``).
    """
    corr = stats["correct"].bool()
    err = ~corr
    n = int(corr.numel())
    n_err = int(err.sum())
    out = {"n": n, "n_errors": n_err,
           "error_rate": (n_err / n) if n else float("nan")}
    scores = {
        "maxprob": -stats["maxprob"].max(dim=1)[0],
        "entropy": stats["entropy"].mean(dim=1),
    }
    if stats.get("gate") is not None:
        scores["gate"] = -stats["gate"].mean(dim=1)
    for name in ("maxprob", "entropy", "gate"):
        score = scores.get(name)
        if score is None or n_err == 0 or n_err == n:
            out[f"rocauc_{name}_to_error"] = float("nan")
            out[f"prauc_{name}_to_error"] = float("nan")
            continue
        out[f"rocauc_{name}_to_error"] = roc_auc(score, err)
        out[f"prauc_{name}_to_error"] = pr_auc(score, err)
    return out
