"""Training and evaluation utilities."""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (
    STATS_SEED,
    VAL_SEED,
    batch_fingerprint,
    check_train_seed,
    make_batch,
    validate_batch,
)
from .metrics import (
    average_ranks,
    brier_score,
    expected_calibration_error,
    multiclass_metrics,
    pr_auc,
    roc_auc,
    selection_at_fraction,
    wilson_ci,
)
from .repro import seed_everything, stream_generator


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


def training_curve_verdict(hist, rel_tol: float = 0.05,
                           grad_norm_explode: float = 100.0):
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
    * ``numerical_instability`` - a non-finite loss *or* gradient norm
                                appeared anywhere;
    * ``insufficient_points`` - fewer than two finite points.

    Beyond the loss deltas the verdict reports the optimization-side curve:
    pre-clip gradient norms (``grad_norm_max``, ``grad_norm_last``,
    ``grad_norm_explode_count``) and, when validation is monitored, how far
    the final validation loss regressed past its best point
    (``val_regression_after_best``: a positive value is the loss the
    experiment contract accepted by not early-stopping).  A grid of runs
    whose curves are all ``plateau`` is underfitting/noise, not evidence
    about the mechanism under test.

    ``late_overfit`` catches the regression the global slope dilutes: a
    U-shaped validation curve can fall and rise by comparable amounts, so
    the fitted relative delta stays near zero and the verdict would read
    "learning"/"plateau" even though the run has been getting worse since
    ``best_val_step``.  It is set when the final validation loss exceeds
    its best by more than ``rel_tol`` of that best, and the recommendation
    then names the early-stop point.

    Returns a dict with the verdict, the relative deltas, the best
    validation point and a human-readable ``recommendation``.
    """
    finite = [h for h in hist
              if math.isfinite(float(h.get("train_loss", float("nan"))))]

    def _bad(v):
        return v is not None and not math.isfinite(float(v))

    unstable = len(finite) < len(hist) or any(
        _bad(h.get("val_loss")) or _bad(h.get("grad_norm")) for h in hist)
    out = {
        "verdict": "insufficient_points",
        "train_delta": float("nan"),
        "val_delta": float("nan"),
        "generalization_gap": float("nan"),
        "best_val_loss": float("nan"),
        "best_val_step": None,
        "val_regression_after_best": float("nan"),
        "late_overfit": False,
        "grad_norm_max": float("nan"),
        "grad_norm_last": float("nan"),
        "grad_norm_explode_count": 0,
        "grad_note": None,
        "n_points": len(finite),
        "recommendation": "not enough logged points to judge the curve",
    }
    if unstable:
        out["verdict"] = "numerical_instability"
        out["recommendation"] = (
            "a non-finite loss or gradient appeared; run the staged triage "
            "in diagnose_nonfinite (data -> activations -> loss -> gradients)")
    grad_norms = [float(h["grad_norm"]) for h in finite
                  if h.get("grad_norm") is not None]
    if grad_norms:
        out["grad_norm_max"] = max(grad_norms)
        out["grad_norm_last"] = grad_norms[-1]
        out["grad_norm_explode_count"] = sum(
            1 for g in grad_norms if g > grad_norm_explode)
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
        out["val_regression_after_best"] = float(
            val[-1]["val_loss"] - best["val_loss"])
        common = [h for h in finite if h.get("val_loss") is not None]
        if common:
            out["generalization_gap"] = float(
                common[-1]["val_loss"] - common[-1]["train_loss"])
        if (len(val) >= 3 and math.isfinite(out["best_val_loss"])
                and out["val_regression_after_best"]
                > rel_tol * abs(out["best_val_loss"])):
            out["late_overfit"] = True

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
    if out["late_overfit"] and out["verdict"] not in (
            "overfitting", "numerical_instability"):
        out["recommendation"] = (
            f"{out['recommendation']}; late overfitting: val loss is "
            f"{out['val_regression_after_best']:.4f} above its best "
            f"(@ step {out['best_val_step']}) - the global slope hides it, "
            f"the early-stop point is best_val_step")
    if out["grad_norm_explode_count"]:
        out["grad_note"] = (
            f"{out['grad_norm_explode_count']} logged step(s) had a pre-clip "
            f"gradient norm above {grad_norm_explode:g}: lower the learning "
            f"rate or check for rare spiking batches")
        out["recommendation"] = f"{out['recommendation']}; {out['grad_note']}"
    return out


def diagnose_nonfinite(model, idx, tgt, meta, vocab, device="cpu",
                       grad_norm_explode: float = 100.0,
                       dense_lm: bool = False):
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

    ``dense_lm`` must match the loss the training loop actually uses:
    with the default ``False`` only the query position is supervised,
    with ``True`` every position is (the full-sequence backward is a
    different graph and can surface non-finite values the single-token
    loss never touches).
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
    sup_pos = None
    if not dense_lm and isinstance(meta, dict):
        sup_pos = meta.get("query_pos")
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
    elif logits.numel():
        logit_max = float(logits.detach().abs().max())
        if logit_max > 80.0:
            report["notes"].append(
                f"max |logit| {logit_max:.1f} exceeds the fp16 range: under "
                f"autocast this saturates/overflows the softmax without a "
                f"max-subtracted stable pass")
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
    for probe in ("is_autocast_enabled", "is_autocast_cpu_enabled",
                  "is_autocast_cuda_enabled"):
        fn = getattr(torch, probe, None)
        if fn is None:
            continue
        try:
            autocast_on = autocast_on or bool(fn())
        except Exception:  # noqa: BLE001 - probe must never break the triage
            pass
    if autocast_on:
        report["notes"].append(
            "autocast (fp16/bfloat16) is active on the current device: check "
            "low-precision overflow (grad scaling, saturated logits) before "
            "blaming the architecture")

    order = ("data", "activations", "loss", "gradients")
    stage = report["first_bad_stage"]
    if stage is None:
        report["suggestions"].append(
            "all stages finite on this batch: the non-finite value was "
            "transient (re-check the batch/step that produced it and any "
            "runtime precision switches)")
    elif stage == "data":
        report["suggestions"].append(
            "address the data pipeline first: NaN/Inf or invalid labels "
            "poison every downstream stage")
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


def _permute_batch_labels(tgt: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Shuffle the supervised labels across the batch (negative-control mode).

    Every row of ``tgt`` carries its own sample's label (or ``-1``), so a
    row permutation swaps which answer each sample is trained to predict -
    the resulting training signal cannot contain the sample's true answer.
    """
    perm = torch.randperm(tgt.size(0), generator=gen, device=tgt.device)
    return tgt[perm]


def train_model(model, steps=1500, batch=64, n_pairs=16, lr=3e-4, vocab=512,
                device="cpu", log_every=250, seed=0, dense_lm=False,
                eval_every=0, val_batches=2, early_stop_patience=None,
                nonfinite="raise", grad_norm_explode=100.0,
                permute_labels=False):
    """Train on the recall task with optional learning-curve monitoring.

    ``seed`` governs *all* run-to-run randomness: ``cage.repro.seed_everything``
    seeds Python/NumPy/PyTorch (CPU + every CUDA device) for the dropout
    masks, and the data generator is a dedicated ``torch.Generator`` - so two
    runs of the same config on one machine are stable.  Without this the
    global RNG (dropout) and/or the data stream would silently differ.

    ``permute_labels`` (negative control, default off): after the first
    batch has passed the data-cleaning gate, the supervised labels of every
    batch are shuffled across the batch before the loss is computed.  A
    model trained this way cannot learn any true answer, so its accuracy on
    the *unchanged* evaluation streams must stay at chance - a clear
    above-chance result means answer information leaks through the
    pipeline.  The staged non-finite triage is fed the un-permuted batch,
    so the intentional label mismatch is never mistaken for a data
    problem.

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
    Off by default because the experiment contract compares runs at a fixed
    step count; ``training_curve_verdict`` reports ``best_val_step`` either
    way.

    ``nonfinite`` (``"raise"`` | ``"skip"``): what to do when a training
    loss *or the pre-clip gradient norm* is not finite.  ``"raise"``
    (default) runs :func:`diagnose_nonfinite` and aborts with the staged
    report - silently stepping an optimizer with NaN gradients corrupts the
    run while producing plausible-looking final numbers.  ``"skip"``
    records the diagnosis in ``hist`` and skips the update instead.

    The gradient norm returned by the fixed clipping step is logged on
    every kept entry (``entry["grad_norm"]``), so the optimization curve -
    not only the loss curve - is monitored: a finite loss can still hide
    non-finite or exploding gradients (the classic fp16-overflow pattern),
    and those are caught here *before* the update instead of being
    misattributed to the next step's activations.  Finite norms above
    ``grad_norm_explode`` are flagged (``entry["grad_explode"]``) but do
    not abort: clipping already bounds the update.
    """
    check_train_seed(seed)
    seed_everything(seed)
    gen = stream_generator(seed, device)
    val_gen = stream_generator(VAL_SEED, device)
    params = list(model.parameters())
    opt = _make_adamw(params, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
    hist = []
    best_val = float("inf")
    best_val_step = None
    evals_since_improve = 0
    printed_grad_spike = 0.0
    model.train()
    for step in range(steps):
        idx, tgt, meta = make_batch(batch, n_pairs, vocab, device, dense_lm=dense_lm, gen=gen)
        if step == 0:
            validate_batch(idx, tgt, meta, vocab)
        diag_tgt = tgt
        if permute_labels:
            tgt = _permute_batch_labels(tgt, gen)
        opt.zero_grad(set_to_none=True)
        sup_pos = None if dense_lm else meta["query_pos"]
        _, loss, _ = model(idx, tgt, sup_pos=sup_pos)
        loss.backward()
        lv = loss.item()
        entry = {"step": step, "train_loss": lv}
        if not math.isfinite(lv):
            diag = diagnose_nonfinite(model, idx, diag_tgt, meta, vocab,
                                      device=device, dense_lm=dense_lm)
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
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        gn = float(grad_norm)
        entry["grad_norm"] = gn
        if not math.isfinite(gn):
            diag = diagnose_nonfinite(model, idx, diag_tgt, meta, vocab,
                                      device=device, dense_lm=dense_lm)
            diag["clip_grad_norm"] = gn
            entry["diagnosis"] = diag
            print(f"  step {step:5d}  loss {lv:.4f}  grad_norm {gn}  <- "
                  f"non-finite gradient with finite loss; triage "
                  f"first_bad_stage={diag['first_bad_stage']}")
            hist.append(entry)
            opt.zero_grad(set_to_none=True)
            sched.step()
            if nonfinite == "raise":
                raise RuntimeError(
                    f"non-finite pre-clip gradient norm {gn} at step {step} "
                    f"with finite loss {lv}; staged triage "
                    f"first_bad_stage={diag['first_bad_stage']!r}, "
                    f"suggestions={diag['suggestions']}")
            continue
        if gn > grad_norm_explode:
            entry["grad_explode"] = True
            if gn > printed_grad_spike:
                printed_grad_spike = gn
                print(f"  step {step:5d}  pre-clip grad_norm {gn:.3g} > "
                      f"{grad_norm_explode:g} (finite, new record); clipping "
                      f"bounds the update, watch for instability")
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
            if not math.isfinite(val_loss):
                print(f"  step {step:5d}  validation loss is non-finite "
                      f"({val_loss}); the run is numerically unstable "
                      f"(see the curve verdict)")
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


def _gradient_flow(model):
    """Per-parameter backward-connectivity audit for one backward pass.

    A parameter with ``grad is None`` after ``loss.backward()`` was never
    reached by the graph - the signature of a detached branch, a
    mis-wired loss or a kernel whose custom backward forgot an input.
    Non-finite gradients are the other failure mode.  The caller runs this
    straight after backward, before clipping/zeroing, on the standard
    (non-ablated) configuration; parameters deliberately switched off in
    an ablation arm may legitimately have no gradient there.
    """
    total = with_grad = 0
    missing, nonfinite = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total += 1
        if p.grad is None:
            missing.append(name)
            continue
        with_grad += 1
        if not bool(torch.isfinite(p.grad).all()):
            nonfinite.append(name)
    return {
        "params_total": total,
        "params_with_grad": with_grad,
        "params_without_grad": missing[:8],
        "nonfinite_grad_params": nonfinite[:8],
        "all_params_connected": not missing,
        "all_grads_finite": not nonfinite,
    }


def _disable_dropout(model):
    """Turn every dropout module off after ``model.train()``.

    The sanity check must run without regularisation, whatever the
    training config says: ``model.train()`` alone would re-activate
    dropout and hide a broken gradient path behind a moving loss.  Only
    dropout is switched off (norms stay in training mode as in the real
    loop), and the settings are returned for the run record.
    """
    modules = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for m in modules:
        m.eval()
    return {
        "dropout_modules": len(modules),
        "dropout_probs": sorted({float(m.p) for m in modules}),
        "active_dropout": any(float(m.p) > 0.0 for m in modules),
        "augmentation": "none (cage.data is a synthetic generator, no "
                        "augmentation pipeline is applied)",
        "optimizer_recipe": "AdamW via train._make_adamw (D5 fixed factory)",
    }


def overfit_sanity_check(model, idx, tgt, meta, lr=3e-3, steps=300,
                         target_loss=0.1, target_acc=1.0, seed=None):
    """Can the training loop fit a single tiny batch? (gradient/loss sanity)

    The classic pre-flight check: take one fixed batch, turn regularisation
    off (dropout is force-disabled here, not assumed from the caller's
    config, and the optimizer's weight decay is zeroed on the check's
    optimizer only - the D5 factory and its hyperparameters stay unchanged
    for real runs), and train the *actual* architecture on it with the
    *actual* optimizer factory.  A correct training loop drives the loss to
    ~0 and the batch accuracy to 1.0 because the batch can be memorised; a
    loop that cannot overfit a handful of samples has broken gradients, a
    mis-wired loss or a silent dtype/precision problem, and no grid result
    above it is interpretable.

    Beyond the loss/accuracy targets the check audits the backward pass
    itself on the first step (``grad_flow``: every trainable parameter
    reached, all gradients finite), so "cannot overfit" is triaged into
    "gradients never connected" vs "optimisation is merely slow".  The
    effective regularisation is recorded (``regularization``: dropout
    force-disabled, weight decay zeroed, no augmentation), so the verdict
    documents *what* was switched off instead of assuming it.  How *fast*
    the batch is memorised is recorded too (``steps_to_target_loss`` /
    ``steps_to_target_acc``: first sampled step meeting each target), and
    the exact batch is pinned by content hash (``data_sha256``), so the
    record says which data was memorised.

    ``seed`` (optional) is recorded and, when given, re-seeds Python /
    NumPy / Torch through :func:`cage.repro.seed_everything`; the caller
    must still seed *before* building the model, because weight
    initialisation happens before this function runs.

    This is a *necessary, not sufficient* check: passing it does not prove
    the data pipeline, augmentation or label mapping are right (such
    problems survive memorisation); failing it blocks every downstream
    claim.  ``passed`` requires the targets *and* a connected, finite
    backward pass.

    Trains ``model`` in place (pass a throwaway instance) and returns the
    sampled loss/accuracy curves plus the verdict.  Uses the sparse
    (query-position) supervision the experiment contract reports.
    """
    if seed is not None:
        seed_everything(int(seed))
    model.train()
    regularization = _disable_dropout(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = _make_adamw(params, lr)
    for group in opt.param_groups:
        group["weight_decay"] = 0.0
    regularization["weight_decay"] = 0.0
    regularization["weight_decay_source"] = (
        "zeroed on the sanity-check optimizer only; the D5 AdamW factory "
        "hyperparameters are unchanged for real runs")
    sample_every = max(1, steps // 10)
    curve = []
    grad_flow = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        logits, loss, _ = model(idx, tgt, sup_pos=meta["query_pos"])
        loss.backward()
        if grad_flow is None:
            grad_flow = _gradient_flow(model)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % sample_every == 0 or step == steps - 1:
            with torch.no_grad():
                acc = float((logits.detach().argmax(-1)
                             == meta["answer"]).to(torch.float64).mean())
            curve.append({"step": step, "loss": float(loss.detach()), "acc": acc})

    def _first_sampled_at(key, threshold, better):
        for point in curve:
            if better(point[key], threshold):
                return int(point["step"])
        return None

    return {
        "steps": int(steps),
        "lr": float(lr),
        "batch": int(idx.size(0)),
        "seed": None if seed is None else int(seed),
        "data_sha256": batch_fingerprint(idx, tgt, meta),
        "loss_first": curve[0]["loss"],
        "loss_last": curve[-1]["loss"],
        "acc_first": curve[0]["acc"],
        "acc_last": curve[-1]["acc"],
        "steps_to_target_loss": _first_sampled_at(
            "loss", float(target_loss), lambda v, t: v <= t),
        "steps_to_target_acc": _first_sampled_at(
            "acc", float(target_acc), lambda v, t: v >= t),
        "curve": curve,
        "regularization": regularization,
        "grad_flow": grad_flow,
        "passed": bool(curve[-1]["loss"] <= target_loss
                       and curve[-1]["acc"] >= target_acc
                       and grad_flow["all_params_connected"]
                       and grad_flow["all_grads_finite"]),
        "target_loss": float(target_loss),
        "target_acc": float(target_acc),
    }


def _validation_metrics(model, val_gen, n_pairs, vocab, device, batch, batches,
                        dense_lm, validate=False):
    """Validation loss (and accuracy) on fixed, held-out validation batches.

    Uses eval mode + ``no_grad`` (no dropout, no RNG consumption), so the
    training RNG stream - and therefore the training trajectory - is
    unaffected by monitoring.
    """
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    correct = torch.zeros((), dtype=torch.int64, device=device)
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
            total_loss.add_(loss.detach().to(torch.float64))
            if not dense_lm:
                correct.add_((logits.argmax(-1) == meta["answer"]).sum())
            total += idx.size(0)
    model.train(was_training)
    n_batches = max(1, batches)
    return (total_loss.item() / n_batches,
            correct.item() / total if total else float("nan"))


def estimate_forward_flops(model, seq_len: int) -> int:
    """Matmul-only FLOPs of one full-sequence forward pass at ``seq_len``.

    Counts ``2 * m * n * k`` per matmul (a MAC is two FLOPs) for q/k/v, the
    causal attention products, the output projection, both MLP projections,
    the gate projection of ``query``/``calib`` and the two per-token scale
    projections of ``seednorm``, plus the full-sequence head.  Elementwise
    work (softmax, GELU, entropy, norms) is excluded, so the result is a
    lower bound on the matmul work of the pass.
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
    """Parameters / FLOPs / latency / memory of one eval-mode forward pass.

    Latency is the median of ``runs`` passes after ``warmup`` (batch
    ``batch`` at ``n_pairs``; fixed seed so every variant scores the same
    shapes).  Memory is the CUDA allocation peak on GPU; on CPU it is the
    profiler-attributed allocation total of one pass.
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


def _label_audit(idx, meta):
    """Per-sample label-integrity flags on the generated context.

    Answers "is the annotation itself wrong?" *on the very batches where
    errors were found* instead of assuming it: a label is valid when the
    query key occurs exactly once among the keys and the recorded answer
    is the value paired with it.  Mirrors the ``validate_batch``
    invariants but returns a boolean per sample so failures can be
    counted and attached to individual bad cases (``validate_batch``
    raises on the first bad batch, which is the right behaviour for a
    training gate and the wrong one for an error audit).
    """
    query = idx.gather(1, meta["query_pos"].unsqueeze(1)).squeeze(1)
    keys = idx[:, 0::2]
    match = keys.eq(query.unsqueeze(1))
    key_pos = match.to(torch.int64).argmax(dim=1).mul(2)
    paired = idx.gather(1, (key_pos + 1).unsqueeze(1)).squeeze(1)
    return match.sum(dim=1).eq(1) & paired.eq(meta["answer"])


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
    * ``label_ok`` / ``error_type`` - whether the annotation of *this*
      sample is valid (``_label_audit``) and a mutually-exclusive bucket:
      ``label_error`` > ``hard_structural`` > ``retrieval_miss``
      (attention mass < 0.5 on the correct key) > ``readout_wrong``
      (key found, wrong value read out);
    * ``key_pair_idx``              - position of the matching pair;
      aggregated into early/mid/late error rates (long-range retrieval).

    The label audit makes "labels cannot be the cause" a *measured* claim:
    ``label_noise_rate`` is computed over every evaluated sample rather
    than assumed to be 0 from the construction.

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

    n_total = n_err = n_label_err = 0
    case_count = 0
    cases = []
    flags_summary = {"high": 0, "low": 0, "in_ctx": 0, "naive": 0, "repeat": 0}
    taxonomy = {"label_error": 0, "hard_structural": 0,
                "retrieval_miss": 0, "readout_wrong": 0}
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
            label_ok = _label_audit(idx, meta)
            n_label_err += int((~label_ok).sum())
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

            ambiguous = flags["naive_confusable"] | flags["value_repeats"]
            key_missed = ~key_right
            taxonomy["label_error"] += int((wrong & ~label_ok).sum())
            taxonomy["hard_structural"] += int(
                (wrong & label_ok & ambiguous).sum())
            taxonomy["retrieval_miss"] += int(
                (wrong & label_ok & ~ambiguous & key_missed).sum())
            taxonomy["readout_wrong"] += int(
                (wrong & label_ok & ~ambiguous & ~key_missed).sum())

            if case_count < max_cases:
                rows = torch.nonzero(wrong, as_tuple=False).flatten().tolist()
                for b in rows:
                    if case_count >= max_cases:
                        break
                    if not bool(label_ok[b]):
                        etype = "label_error"
                    elif bool(ambiguous[b]):
                        etype = "hard_structural"
                    elif bool(key_missed[b]):
                        etype = "retrieval_miss"
                    else:
                        etype = "readout_wrong"
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
                        "label_ok": bool(label_ok[b]),
                        "error_type": etype,
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
        "n_label_errors": int(n_label_err),
        "label_noise_rate": (n_label_err / n_total) if n_total else float("nan"),
        "taxonomy_label_error": _share(taxonomy["label_error"]),
        "taxonomy_hard_structural": _share(taxonomy["hard_structural"]),
        "taxonomy_retrieval_miss": _share(taxonomy["retrieval_miss"]),
        "taxonomy_readout_wrong": _share(taxonomy["readout_wrong"]),
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
    B, T, _ = x.shape
    C = weight.size(0) // 3
    dk = C // n_heads
    qp = meta["query_pos"]
    kp = meta["key_pos"]
    rows = torch.arange(B, device=x.device)
    q_all = F.linear(x[rows, qp], weight[:C]).view(B, n_heads, dk)
    k_all = F.linear(x, weight[C:2 * C]).view(B, T, n_heads, dk)
    scores = torch.einsum("bhd,bthd->bth", q_all, k_all).mul_(inv_scale)
    cols = torch.arange(T, device=x.device).view(1, T, 1)
    scores.masked_fill_(cols > qp.view(B, 1, 1), float("-inf"))
    if null_k is not None:
        s0 = torch.einsum("bhd,hd->bh", q_all, null_k).mul_(inv_scale)
        scores = torch.cat([s0.unsqueeze(1), scores], dim=1)
        tapped = torch.softmax(scores, dim=1).gather(
            1, (kp + 1).view(B, 1, 1).expand(B, 1, n_heads))
    else:
        tapped = torch.softmax(scores, dim=1).gather(
            1, kp.view(B, 1, 1).expand(B, 1, n_heads))
    return tapped.squeeze(1).mean(-1)


def _error_case_flags(idx, meta, pred, n_pairs):
    """Per-sample structural flags used to classify an error's cause."""
    T = idx.size(1)
    pos = torch.arange(T, device=idx.device).view(1, -1)
    visible = pos < meta["query_pos"].unsqueeze(1)
    query = idx.gather(1, meta["query_pos"].unsqueeze(1)).squeeze(1)
    answer = meta["answer"]
    vals = idx[:, 1:2 * n_pairs:2]
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
    correct = torch.zeros((), dtype=torch.int64, device=device)
    total = 0
    for _ in range(batches):
        idx, _, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        logits, _, _ = model(idx)
        pred = logits.squeeze(1).argmax(-1)
        correct.add_((pred == meta["answer"]).sum())
        total += idx.size(0)
    model.train(was_training)
    return correct.item() / total


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

    ``accuracy_lo`` / ``accuracy_hi`` are the Wilson 95% interval for the
    accuracy over the finite evaluation stream: seed-level std quantifies
    training variance, while this interval quantifies how precisely the
    *reported* stream of ``batch * batches`` samples pins the accuracy
    down (the two are different uncertainties and both belong next to a
    headline number).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    logits_all, tgt_all = [], []
    for _ in range(batches):
        idx, _, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        logits, _, _ = model(idx)
        logits_all.append(logits.squeeze(1).float())
        tgt_all.append(meta["answer"])
    model.train(was_training)
    logits_cat = torch.cat(logits_all, dim=0).cpu()
    tgt_cat = torch.cat(tgt_all, dim=0).cpu()
    out = multiclass_metrics(logits_cat, tgt_cat, topk=topk)
    acc_ci = wilson_ci(int((logits_cat.argmax(-1) == tgt_cat).sum()),
                       int(tgt_cat.numel()))
    out["accuracy_lo"] = acc_ci["lo"]
    out["accuracy_hi"] = acc_ci["hi"]
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


def spearman(x, y):
    """Rank correlation (average ranks for ties).

    Robustness companion to ``pearson`` for H1a: the mechanism claim is a
    *monotone* association ("the gate opens when attention is diffuse"),
    which Pearson only captures as long as the relation is linear and no
    outlier dominates it.  Agreement between the two coefficients means
    the effect is not an artifact of the correlation measure.
    """
    x = x.flatten().float()
    y = y.flatten().float()
    if x.numel() == 0:
        return torch.tensor(float("nan"))
    rx = average_ranks(x)
    ry = average_ranks(y)
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    if dx.abs().max() <= 1e-12 or dy.abs().max() <= 1e-12:
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
    r["gate_vs_entropy_spearman"] = _fisher_mean(
        [spearman(g[:, i], ent[:, i]).item() for i in cols])
    r["gate_vs_maxprob_spearman"] = _fisher_mean(
        [spearman(g[:, i], mp[:, i]).item() for i in cols])
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


def false_concentration_sweep(stats, thresholds=(0.3, 0.5, 0.7, 0.9)) -> dict:
    """H1b robustness: the false-concentration split at several thresholds.

    ``maxprob_thresh=0.5`` is an a-priori choice, but the whole H1b verdict
    (gate margin = gate on correct - gate on false concentration) depends
    on where "high confidence" is cut.  Sweeping the threshold makes the
    analyst's degree of freedom explicit: a mechanism claim that only
    holds at one arbitrary cut is a property of the cut, not of the
    mechanism.  The pre-registered contrast stays at 0.5; the sweep is
    reported so cherry-picking can be ruled out.

    Returns ``{threshold: false_concentration_analysis(stats, thr)}``.
    """
    return {float(thr): false_concentration_analysis(stats, maxprob_thresh=thr)
            for thr in thresholds}


def error_detection_aucs(stats):
    """Imbalance-aware error-detection summary of attention statistics.

    The positive class is "model answered wrong".  Scores are oriented so
    that **higher means more likely wrong** (``-maxprob``, mean entropy,
    ``-gate``), matching the mechanism hypotheses: concentration should
    track correctness, and a calibrated gate should close when it does
    not.

    Errors are the rare class after convergence, so the metric *regime*
    is measured, not assumed: ``imbalance_ratio`` (negatives per
    positive), ``error_rate`` = prevalence, and ``metric_regime``
    (``extreme`` at 1:1000 or worse, ``high`` at 1:10 or worse, else
    ``balanced``).  Under ``extreme`` imbalance ROC-AUC grows with the
    huge negative population while precision can still collapse, so the
    headline is **PR-AUC, read against its chance level** - for a random
    ranker average precision equals prevalence, so ``prauc_lift_*``
    (PR-AUC / prevalence) is what says whether a score carries signal,
    and ``primary_metric`` names the headline explicitly.

    Every score also gets ``recall_at_10pct_*`` / ``lift_at_10pct_*``:
    the share of errors captured in the 10% most doubtful answers
    (ranked by that score) and the precision lift of that flagged set
    over prevalence - the operational reading of the same ranking when
    PR-AUC alone is hard to interpret (see
    ``cage.metrics.selection_at_fraction``).
    """
    corr = stats["correct"].bool()
    err = ~corr
    n = int(corr.numel())
    n_err = int(err.sum())
    n_neg = n - n_err
    prevalence = (n_err / n) if n else float("nan")
    ratio = (n_neg / n_err) if n_err else float("inf")
    if n_err == 0:
        regime = "no_errors"
    elif ratio >= 1000.0:
        regime = "extreme (>=1:1000; PR-AUC is the headline, ROC-AUC is optimistic)"
    elif ratio >= 10.0:
        regime = "high (>=1:10; report both, trust PR-AUC for rare-positive ranking)"
    else:
        regime = "balanced"
    out = {"n": n, "n_errors": n_err, "error_rate": prevalence,
           "prevalence": prevalence, "imbalance_ratio": ratio,
           "metric_regime": regime, "flag_fraction": 0.1}
    scores = {
        "maxprob": -stats["maxprob"].max(dim=1)[0],
        "entropy": stats["entropy"].mean(dim=1),
    }
    if stats.get("gate") is not None:
        scores["gate"] = -stats["gate"].mean(dim=1)
    primary = "gate" if scores.get("gate") is not None else "maxprob"
    out["primary_metric"] = f"prauc_{primary}_to_error"
    for name in ("maxprob", "entropy", "gate"):
        score = scores.get(name)
        if score is None or n_err == 0 or n_err == n:
            out[f"rocauc_{name}_to_error"] = float("nan")
            out[f"prauc_{name}_to_error"] = float("nan")
            out[f"prauc_lift_{name}_to_error"] = float("nan")
            out[f"recall_at_10pct_{name}_to_error"] = float("nan")
            out[f"lift_at_10pct_{name}_to_error"] = float("nan")
            continue
        pa = pr_auc(score, err)
        out[f"rocauc_{name}_to_error"] = roc_auc(score, err)
        out[f"prauc_{name}_to_error"] = pa
        out[f"prauc_lift_{name}_to_error"] = (
            pa / prevalence if prevalence > 0.0 else float("nan"))
        sel = selection_at_fraction(score, err, 0.1)
        out[f"recall_at_10pct_{name}_to_error"] = sel["recall"]
        out[f"lift_at_10pct_{name}_to_error"] = sel["lift"]
    return out
