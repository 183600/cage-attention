"""Training and evaluation utilities."""

from __future__ import annotations

import math

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
from .model import release_eval_scratch
from .repro import seed_everything, stream_generator


# Fixed optimisation recipe (design pin D5): a single source of truth for
# the hyperparameters the factory uses *and* for the ``training_recipe``
# block every run record stamps, so the controlled-comparison audit can
# verify optimizer equality from the records instead of assuming it.
_ADAMW_WEIGHT_DECAY = 0.01
_ADAMW_BETAS = (0.9, 0.95)
_GRAD_CLIP_MAX_NORM = 1.0
_ETA_MIN_FACTOR = 0.1


def _make_adamw(params, lr):
    """AdamW on the current device."""
    params = list(params)
    kw = dict(lr=lr, weight_decay=_ADAMW_WEIGHT_DECAY, betas=_ADAMW_BETAS)
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


def training_recipe(lr, steps=None) -> dict:
    """The complete optimisation recipe of one training run.

    Comparing method A with method B only attributes a difference to the
    module under test if *everything else* - optimizer family, weight
    decay, betas, learning-rate schedule, eta_min, gradient clipping and
    the (absent) augmentation pipeline - is held fixed.  The recipe is
    fixed centrally in ``_make_adamw`` / ``train_model``; this block is
    stamped into every run record so the controlled-variable audit can
    check that equality *from the records* rather than trusting that no
    call site silently passed a different optimizer setting.
    """
    out = {
        "optimizer": "AdamW",
        "weight_decay": _ADAMW_WEIGHT_DECAY,
        "betas": list(_ADAMW_BETAS),
        "schedule": "CosineAnnealingLR",
        "eta_min_factor": _ETA_MIN_FACTOR,
        "grad_clip_max_norm": _GRAD_CLIP_MAX_NORM,
        "augmentation": "none (synthetic generator; fixed protocol)",
        "lr": float(lr),
    }
    if steps is not None:
        out["steps"] = int(steps)
    return out


def training_curve_verdict(hist, rel_tol: float = 0.05,
                           grad_norm_explode: float = 100.0):
    """Read the *curve*, not just the final number (over/underfit triage).

    ``hist`` is the list of dicts returned by :func:`train_model`.  When a
    point carries ``train_loss_window`` (the mean over all finite update
    steps since the previous logged/evaluated point) that smoothed value
    is used for the train curve instead of the single-batch
    ``train_loss``: one batch's loss is noisy enough to flip a 5% verdict
    at small batch sizes.  The train and (optionally monitored)
    validation losses are reduced to a least-squares relative change over
    the logged window:
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

    The window is only half the reading, so the *tail* is read too:
    ``late_train_delta`` / ``late_val_delta`` refit the same relative
    delta on the last third of the logged points, which separates "was
    still improving when the step limit was reached" (plateau verdict with
    a falling tail -> more steps, not a broken pipeline) from a run whose
    final regime matches the global slope.  ``val_loss_se_last`` is the
    batch-sampling standard error of the last validation point (from
    :func:`_validation_metrics`), and ``late_overfit_significant`` marks
    whether ``val_regression_after_best`` exceeds two of those SEs - a
    surface that is noisier than the regression is not evidence of
    overfitting.  ``lr_first`` / ``lr_last`` let the reader check that the
    cosine schedule actually ran (a flat tail can be an lr that never
    annealed).

    Returns a dict with the verdict, the relative deltas, the best
    validation point and a human-readable ``recommendation``.
    ``overfitting_significant`` mirrors ``late_overfit_significant`` for
    the global verdict: it is set only when the train/val divergence is
    larger than the validation-batch noise, so a noisy flat validation
    curve cannot be sold as overfitting.
    """
    # Work on copies so the caller's history keeps the raw single-batch
    # losses; the smoothed window value only changes the verdict input when
    # the raw value is finite, so a non-finite batch loss is never hidden.
    hist = [dict(h) for h in hist]
    for h in hist:
        raw = h.get("train_loss")
        window = h.get("train_loss_window")
        if (window is not None and math.isfinite(float(window))
                and (raw is None or math.isfinite(float(raw)))):
            h["train_loss"] = float(window)
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
        "late_overfit_significant": False,
        "late_train_delta": float("nan"),
        "late_val_delta": float("nan"),
        "val_loss_se_last": float("nan"),
        "val_change_abs": float("nan"),
        "val_noise_2se": float("nan"),
        "val_movement_significant": False,
        "overfitting_significant": False,
        "lr_first": float("nan"),
        "lr_last": float("nan"),
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

    # Tail reading: the last third answers "was it still improving when the
    # step limit was reached?", which the window fit averages away.
    if len(finite) >= 3:
        k = max(2, int(math.ceil(len(finite) / 3.0)))
        out["late_train_delta"] = rel_delta(finite[-k:], "train_loss")
        if len(val) >= 3:
            # Validation is monitored less often than training, so its
            # curve is shorter: take its own last third rather than the
            # train-point count (which can swallow the whole val curve).
            k_val = max(2, int(math.ceil(len(val) / 3.0)))
            out["late_val_delta"] = rel_delta(val[-k_val:], "val_loss")
    val_se = [float(h["val_loss_se"]) for h in val
              if h.get("val_loss_se") is not None
              and math.isfinite(float(h["val_loss_se"]))]
    if val_se:
        out["val_loss_se_last"] = val_se[-1]
        if (out["late_overfit"] and out["val_loss_se_last"] > 0.0
                and math.isfinite(out["val_regression_after_best"])):
            out["late_overfit_significant"] = bool(
                out["val_regression_after_best"] > 2.0 * out["val_loss_se_last"])
    if len(val) >= 2 and val_se:
        # Is the monitored validation *movement* (first -> last) larger than
        # the batch-sampling noise of a single evaluation?  With only a
        # couple of validation batches the SE can be a sizeable fraction of
        # the loss, and a "flat" verdict read from a curve whose movement is
        # inside 2 SE says more about the monitor's resolution than about
        # the model.
        out["val_change_abs"] = float(val[-1]["val_loss"] - val[0]["val_loss"])
        out["val_noise_2se"] = float(2.0 * val_se[-1])
        out["val_movement_significant"] = bool(
            out["val_noise_2se"] > 0.0
            and abs(out["val_change_abs"]) > out["val_noise_2se"])
    lrs = [float(h["lr"]) for h in finite if h.get("lr") is not None
           and math.isfinite(float(h["lr"]))]
    if lrs:
        out["lr_first"] = lrs[0]
        out["lr_last"] = lrs[-1]

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
        if (math.isfinite(out["late_train_delta"])
                and out["late_train_delta"] < -rel_tol):
            out["recommendation"] = (
                "the window fit is flat but the last third of training is "
                "still improving: the step count may simply be too low for "
                "this recipe (raise steps before calling it underfit)")
    if (not out["val_movement_significant"]
            and math.isfinite(out["val_change_abs"])
            and out["verdict"] not in ("numerical_instability",
                                       "insufficient_points")):
        out["recommendation"] = (
            f"{out['recommendation']}; the validation movement "
            f"({out['val_change_abs']:+.4f}) is within 2x the last "
            f"validation-batch SE ({out['val_noise_2se']:.4f}), so the "
            f"val-side read has the monitor's resolution, not the model's")
    if out["late_overfit"] and out["verdict"] not in (
            "overfitting", "numerical_instability"):
        sig = ("the regression is real (larger than 2x the last validation "
               "SE)" if out["late_overfit_significant"] else
               "but the regression is within ~2x the validation-batch SE, "
               "so it may be stream noise rather than overfitting")
        out["recommendation"] = (
            f"{out['recommendation']}; late overfitting: val loss is "
            f"{out['val_regression_after_best']:.4f} above its best "
            f"(@ step {out['best_val_step']}) - the global slope hides it, "
            f"the early-stop point is best_val_step; {sig}")
    if out["grad_norm_explode_count"]:
        out["grad_note"] = (
            f"{out['grad_norm_explode_count']} logged step(s) had a pre-clip "
            f"gradient norm above {grad_norm_explode:g}: lower the learning "
            f"rate or check for rare spiking batches")
        out["recommendation"] = f"{out['recommendation']}; {out['grad_note']}"
    out["overfitting_significant"] = bool(
        out["verdict"] == "overfitting" and out["val_movement_significant"])
    return out


def diagnose_nonfinite(model, idx, tgt, meta, vocab, device="cpu",
                       grad_norm_explode: float = 100.0,
                       dense_lm: bool = False):
    """Staged triage for a non-finite loss.

    1. **Data**: NaN/Inf tokens or targets, structural/label violations
       (``validate_batch``);
    2. **Forward activations**: the first block whose output is non-finite,
       plus non-finite parameters and invalid calibrated temperatures
       (log(0)/div-by-zero/softmax-overflow surface here);
    2b. **Attention statistics**: the kernel's own entropy / max-prob /
       gate / null-mass readout is checked for finiteness and valid range,
       so a log(0) or softmax overflow that stays inside the attention
       block cannot be hidden by a finite hidden state;
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
    training_states = {module: module.training for module in model.modules()}

    # Inference tensors cannot be saved for backward, so normalise the
    # inputs to ordinary tensors before the grad-enabled forward.
    def _normal(t):
        if torch.is_tensor(t) and t.is_inference():
            with torch.inference_mode(False):
                return t.clone()
        return t

    if torch.is_tensor(idx) or torch.is_tensor(tgt):
        with torch.inference_mode(False):
            idx, tgt = _normal(idx), _normal(tgt)
    if isinstance(meta, dict):
        meta = {k: _normal(v) for k, v in meta.items()}

    report = {
        "data": "ok",
        "activations": "ok",
        "attention_stats": "ok",
        "loss": "ok",
        "gradients": "ok",
        "first_bad_stage": None,
        "grad_total_norm": float("nan"),
        "grad_max_abs": float("nan"),
        "nonfinite_params": [],
        "nonfinite_attn_stats": [],
        "notes": [],
        "suggestions": [],
    }

    def mark(stage, message):
        report[stage] = message
        if report["first_bad_stage"] is None:
            report["first_bad_stage"] = stage

    # ---- 1. data ---------------------------------------------------------
    try:
        validate_batch(idx, tgt, meta, vocab, diagnostics=False)
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
        for blk in model.blocks:
            blk.attn._capture_qk = False
            blk.attn._qk_capture = None
        for module, training in training_states.items():
            module.train(training)
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

    # ---- 2b. attention statistics (kernel-internal log/softmax) ----------
    # A non-finite entropy/max-probability can live entirely inside the
    # attention kernel and still leave the block output finite (the gate
    # multiplies by a saturated value, or the readout averages it away).
    # The public stats path reads those tensors directly, so triage checks
    # them rather than inferring kernel health from hidden states only.
    if isinstance(meta, dict) and meta.get("query_pos") is not None:
        stat_pos = meta["query_pos"]
        try:
            model.eval()
            with torch.inference_mode():
                _, _, layer_stats = model(idx, stat_pos=stat_pos)
            for layer_idx, stat in enumerate(layer_stats or ()):
                for key, value in stat.items():
                    if not torch.is_tensor(value) or not value.numel():
                        continue
                    if not bool(torch.isfinite(value).all()):
                        report["nonfinite_attn_stats"].append(
                            f"layer {layer_idx}.{key}")
                    elif key == "maxprob" and (
                            float(value.min()) < 0.0
                            or float(value.max()) > 1.0):
                        report["notes"].append(
                            f"layer {layer_idx}.maxprob outside [0, 1] "
                            f"([{float(value.min()):.3g}, "
                            f"{float(value.max()):.3g}]): softmax numerics "
                            f"are invalid")
                    elif key == "entropy" and float(value.min()) < 0.0:
                        report["notes"].append(
                            f"layer {layer_idx}.entropy is negative "
                            f"({float(value.min()):.3g}): log/div-by-zero "
                            f"inside the attention kernel")
        except Exception as e:  # noqa: BLE001 - report, never raise from triage
            mark("attention_stats", f"stats forward raised: {e}")
        finally:
            for module, training in training_states.items():
                module.train(training)
        if report["nonfinite_attn_stats"]:
            mark("attention_stats",
                 f"non-finite attention statistics at "
                 f"{report['nonfinite_attn_stats'][:6]} "
                 f"(check entropy log(0)/division and softmax overflow in "
                 f"the kernel)")

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

    order = ("data", "activations", "attention_stats", "loss", "gradients")
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
        if idx_stage >= order.index("attention_stats"):
            report["suggestions"].append(
                "the attention statistics are bad while the block output "
                "may still be finite: inspect p.clamp_min(_EPS) before the "
                "entropy log, row sums after masking, and the softmax "
                "max-subtraction inside the fixed kernel")
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
                permute_labels=False, restore_best=True,
                early_stop_info=None, fixed_batch=None,
                disable_regularization=False, grad_flow_info=None,
                regularization_info=None, batch_metrics=False,
                deterministic=False):
    """Train on the recall task with optional learning-curve monitoring.

    ``seed`` governs *all* run-to-run randomness: ``cage.repro.seed_everything``
    seeds Python/NumPy/PyTorch (CPU + every CUDA device) for the dropout
    masks, and the data generator is a dedicated ``torch.Generator`` - so two
    runs of the same config on one machine are stable.  Without this the
    global RNG (dropout) and/or the data stream would silently differ.

    ``permute_labels`` (negative control, default off): after each batch
    has passed the data-cleaning gate, its supervised labels are shuffled
    across the batch before the loss is computed.  The
    permutation draws from its own generator (seeded from ``seed``), not
    from the data generator, so a permuted run sees exactly the same
    batches as an otherwise identical non-permuted run and only the label
    channel differs - the control isolates the labels instead of also
    changing the data stream.  A model trained this way cannot learn any
    true answer, so its accuracy on the *unchanged* evaluation streams
    must stay at chance - a clear above-chance result means answer
    information leaks through the pipeline.  The staged non-finite triage
    is fed the un-permuted batch, so the intentional label mismatch is
    never mistaken for a data problem.

    ``seed`` must not collide with a held-out evaluation stream
    (``cage.data.RESERVED_EVAL_SEEDS``); every batch of the run passes the
    data-cleaning gate before its gradient step, so a generator defect
    that only appears after the first batch cannot be trained on.
    ``seed=None`` is accepted only with ``fixed_batch`` (the caller has
    already established the RNG state) and performs no re-seeding.

    Curve monitoring (``eval_every > 0``): every ``eval_every`` steps the
    model is scored on ``val_batches`` fresh batches drawn from the fixed
    validation stream (``cage.data.VAL_SEED``) - the same batches at every
    checkpoint, and never the reported test stream.  Validation forwards
    run in eval mode, so they draw *no* RNG and do not perturb the training
    trajectory.  ``hist`` entries are dicts
    ``{"step", "train_loss", "lr"[, "val_loss", "val_acc", "val_loss_se"]}``
    to be fed to :func:`training_curve_verdict` (monitoring the curve
    instead of only the final number is what catches over/underfitting;
    ``lr`` proves the cosine schedule actually annealed and ``val_loss_se``
    gives the batch-sampling noise of each monitored point).

    ``early_stop_patience`` (opt-in, default off): stop once the monitored
    validation loss has not improved for that many consecutive evaluations.
    Off by default because the experiment contract compares runs at a fixed
    step count; ``training_curve_verdict`` reports ``best_val_step`` either
    way.  ``restore_best`` (default on) reloads the state at
    ``best_val_step`` when early stopping fires - stopping on a patience
    counter but keeping the overfit last-step weights would defeat the
    purpose.  The optional ``early_stop_info`` dict is filled with
    ``enabled`` / ``stopped_early`` / ``restored_best`` / ``best_val_*`` /
    ``steps_run`` so callers can record what the run actually produced;
    ``steps_run`` counts only the iterations that reached ``opt.step()``, so
    a skipped non-finite update is not mistaken for a trained step.

    The training loss stored in each ``hist`` entry is the exact
    pre-update loss of that step (``train_loss``); the mean over all
    finite update steps since the previous logged/evaluated point is
    stored alongside as ``train_loss_window``.  A single batch's loss is
    noisy, so :func:`training_curve_verdict` prefers the window average
    when it is present: over/underfit calls are then about the trend, not
    about which batch happened to be drawn at the logging step.

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

    Diagnostics/pre-flight options (defaults leave the training loop
    unchanged):

    * ``fixed_batch=(idx, tgt, meta)`` trains every step on the *same*
      batch instead of drawing from the stream.  The data-cleaning gate
      still runs, so the tiny-batch memorisation check exercises this real
      loop rather than a parallel copy of it.
    * ``disable_regularization`` turns dropout off after ``model.train()``
      and zeroes weight decay on this optimizer only; the D5 factory and
      its hyperparameters are untouched for real runs.  The effective
      settings are copied into ``regularization_info`` when given.
    * ``grad_flow_info`` receives the first-step ``_gradient_flow`` audit
      (parameter connectivity and gradient finiteness).
    * ``batch_metrics`` adds the pre-update fixed-batch accuracy to each
      logged ``hist`` entry, which the overfit check samples as its
      learning curve.
    """
    if seed is None:
        if fixed_batch is None:
            raise ValueError(
                "seed=None is only valid with fixed_batch: a stream-drawn "
                "run must pin its data seed")
    else:
        check_train_seed(seed)
        seed_everything(seed, deterministic=deterministic)
    if early_stop_patience is not None and eval_every <= 0:
        raise ValueError(
            "early_stop_patience requires eval_every > 0: there is no "
            "validation curve to stop on otherwise")
    stream_seed = 0 if seed is None else int(seed)
    gen = stream_generator(stream_seed, device)
    # Labels are permuted from their own stream so the negative control
    # draws the same data batches as the corresponding clean run.
    perm_gen = stream_generator(stream_seed, device)
    val_gen = stream_generator(VAL_SEED, device)
    params = list(model.parameters())
    opt = _make_adamw(params, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * _ETA_MIN_FACTOR)
    hist = []
    best_val = float("inf")
    best_val_step = None
    best_state = None
    stopped_early = False
    updates_run = 0
    evals_since_improve = 0
    window_loss_sum = 0.0
    window_loss_n = 0
    printed_grad_spike = 0.0
    model.train()
    if disable_regularization:
        info = _disable_dropout(model)
        for group in opt.param_groups:
            group["weight_decay"] = 0.0
        info["weight_decay"] = 0.0
        info["weight_decay_source"] = (
            "zeroed on the diagnostic optimizer only; the D5 AdamW factory "
            "hyperparameters are unchanged for real runs")
        if regularization_info is not None:
            regularization_info.update(info)
    for step in range(steps):
        if fixed_batch is None:
            idx, tgt, meta = make_batch(batch, n_pairs, vocab, device,
                                        dense_lm=dense_lm, gen=gen)
        else:
            idx, tgt, meta = fixed_batch
        validate_batch(idx, tgt, meta, vocab, diagnostics=False)
        diag_tgt = tgt
        if permute_labels:
            tgt = _permute_batch_labels(tgt, perm_gen)
        opt.zero_grad(set_to_none=True)
        sup_pos = None if dense_lm else meta["query_pos"]
        logits, loss, _ = model(idx, tgt, sup_pos=sup_pos)
        loss.backward()
        if grad_flow_info is not None and step == 0:
            grad_flow_info.update(_gradient_flow(model))
        lv = loss.item()
        entry = {"step": step, "train_loss": lv,
                 "lr": float(opt.param_groups[0]["lr"])}
        if batch_metrics:
            with torch.no_grad():
                if dense_lm:
                    rows = torch.arange(idx.size(0), device=logits.device)
                    qlogits = logits.reshape(
                        idx.size(0), idx.size(1), -1)[rows, meta["query_pos"]]
                else:
                    qlogits = logits
                entry["batch_acc"] = float(
                    (qlogits.argmax(-1) == meta["answer"]).to(
                        torch.float64).mean())
        is_log = (log_every > 0 and step % log_every == 0) or step == steps - 1
        do_val = (eval_every > 0 and (step % eval_every == 0 or step == steps - 1))
        if not math.isfinite(lv):
            diag = diagnose_nonfinite(model, idx, diag_tgt, meta, vocab,
                                      device=device, dense_lm=dense_lm)
            entry["diagnosis"] = diag
            print(f"  step {step:5d}  loss {lv}  <- non-finite; triage "
                  f"first_bad_stage={diag['first_bad_stage']}")
            hist.append(entry)
            opt.zero_grad(set_to_none=True)
            sched.step()
            if is_log or do_val:
                if window_loss_n:
                    entry["train_loss_window"] = window_loss_sum / window_loss_n
                    window_loss_sum = 0.0
                    window_loss_n = 0
            if nonfinite == "raise":
                raise RuntimeError(
                    f"non-finite training loss {lv} at step {step}; staged "
                    f"triage first_bad_stage={diag['first_bad_stage']!r}, "
                    f"suggestions={diag['suggestions']}")
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(params, _GRAD_CLIP_MAX_NORM)
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
            if is_log or do_val:
                if window_loss_n:
                    entry["train_loss_window"] = window_loss_sum / window_loss_n
                    window_loss_sum = 0.0
                    window_loss_n = 0
            if nonfinite == "raise":
                raise RuntimeError(
                    f"non-finite pre-clip gradient norm {gn} at step {step} "
                    f"with finite loss {lv}; staged triage "
                    f"first_bad_stage={diag['first_bad_stage']!r}, "
                    f"suggestions={diag['suggestions']}")
            continue
        # Only finite steps that actually reach the optimizer update enter
        # the window (a skipped non-finite-gradient step updates nothing).
        window_loss_sum += lv
        window_loss_n += 1
        if gn > grad_norm_explode:
            entry["grad_explode"] = True
            if gn > printed_grad_spike:
                printed_grad_spike = gn
                print(f"  step {step:5d}  pre-clip grad_norm {gn:.3g} > "
                      f"{grad_norm_explode:g} (finite, new record); clipping "
                      f"bounds the update, watch for instability")
        opt.step()
        sched.step()
        updates_run += 1

        if do_val:
            val_loss, val_acc, val_batch, val_loss_se = _validation_metrics(
                model, val_gen, n_pairs, vocab, device, batch, val_batches,
                dense_lm)
            entry["val_loss"] = val_loss
            entry["val_acc"] = val_acc
            entry["val_loss_se"] = val_loss_se
            if not math.isfinite(val_loss):
                # Run the staged triage on the exact validation batch that
                # produced the value, so an eval-stream data problem can be
                # told apart from a kernel one.
                v_idx, v_tgt, v_meta = val_batch
                diag = diagnose_nonfinite(model, v_idx, v_tgt, v_meta, vocab,
                                          device=device, dense_lm=dense_lm)
                entry["val_diagnosis"] = diag
                print(f"  step {step:5d}  validation loss is non-finite "
                      f"({val_loss}); the run is numerically unstable - "
                      f"staged triage first_bad_stage="
                      f"{diag['first_bad_stage']!r} (train loss and "
                      f"pre-clip grad norm are finite, so the eval "
                      f"batch/forward is the prime suspect)")
            if val_loss < best_val:
                best_val, best_val_step = val_loss, step
                evals_since_improve = 0
                if early_stop_patience is not None and restore_best:
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()}
            else:
                evals_since_improve += 1
        if is_log or do_val:
            if window_loss_n:
                entry["train_loss_window"] = window_loss_sum / window_loss_n
                window_loss_sum = 0.0
                window_loss_n = 0
            hist.append(entry)
        if is_log:
            line = f"  step {step:5d}  loss {lv:.4f}"
            if "val_loss" in entry:
                line += (f"  val_loss {entry['val_loss']:.4f}  "
                         f"val_acc {entry['val_acc']:.4f}")
            print(line)
        if (early_stop_patience is not None and do_val
                and evals_since_improve >= early_stop_patience):
            stopped_early = True
            print(f"  early stop at step {step}: val loss has not improved "
                  f"for {evals_since_improve} evaluations "
                  f"(best {best_val:.4f} @ step {best_val_step})")
            break
    restored = False
    if stopped_early and restore_best and best_state is not None:
        model.load_state_dict(best_state)
        restored = True
        print(f"  restored weights from best validation step "
              f"{best_val_step} (val loss {best_val:.4f})")
    if early_stop_info is not None:
        early_stop_info.update({
            "enabled": early_stop_patience is not None,
            "patience": (None if early_stop_patience is None
                         else int(early_stop_patience)),
            "stopped_early": bool(stopped_early),
            "restored_best": bool(restored),
            "best_val_loss": (float(best_val) if math.isfinite(best_val)
                              else float("nan")),
            "best_val_step": best_val_step,
            "steps_run": int(updates_run),
        })
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
    dropout and perturb the loss the check reads.  Only dropout is
    switched off (norms stay in training mode as in the real loop), and
    the settings are returned for the run record.
    """
    modules = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for m in modules:
        m.eval()
    return {
        "dropout_modules": len(modules),
        "dropout_probs": sorted({float(m.p) for m in modules}),
        # Dropout was configured but is switched off for the check; report
        # the effective state, not the configured probability.
        "active_dropout": any(m.training and float(m.p) > 0.0
                              for m in modules),
        "augmentation": "none (cage.data is a synthetic generator, no "
                        "augmentation pipeline is applied)",
        "optimizer_recipe": "AdamW via train._make_adamw (D5 fixed factory)",
    }


def overfit_sanity_check(model, idx, tgt, meta, lr=3e-3, steps=300,
                         target_loss=0.1, target_acc=1.0, seed=None,
                         vocab=512, deterministic=False):
    """Can the training loop fit a single tiny batch? (gradient/loss sanity)

    The classic pre-flight check: take one fixed batch, turn regularisation
    off (dropout is force-disabled here, not assumed from the caller's
    config, and the optimizer's weight decay is zeroed on the check's
    optimizer only - the D5 factory and its hyperparameters stay unchanged
    for real runs), and train the *actual* architecture on it with the
    *actual* optimizer factory.  A correct training loop drives the loss to
    ~0 and the batch accuracy to 1.0 because the batch can be memorised; a
    loop that cannot overfit a handful of samples is invalid, and no grid
    result above it is interpretable.

    Beyond the loss/accuracy targets the check audits the backward pass
    itself on the first step (``grad_flow``: every trainable parameter
    reached, all gradients finite), so "cannot overfit" is triaged into
    "gradients never connected" vs "the optimiser configuration".  The
    effective regularisation is recorded (``regularization``: dropout
    force-disabled, weight decay zeroed, no augmentation), so the verdict
    documents *what* was switched off instead of assuming it.  The step at
    which the batch is memorised is recorded too (``steps_to_target_loss`` /
    ``steps_to_target_acc``: first sampled step meeting each target), and
    the exact batch is pinned by content hash (``data_sha256``), so the
    record says which data was memorised.

    ``loss_first``/``loss_last`` and ``acc_first``/``acc_last`` are the
    first and last points of ``curve``.  The last point is evaluated on
    the *final* weights after the training loop, not on the pre-update
    forward of the final step: the verdict must score the model the loop
    actually produced (``step == steps`` marks that post-training point).
    ``passed`` additionally requires the loss to have fallen from the
    first sample (``learned``), so a model that merely *starts* at the
    target - which would itself signal a pipeline problem - cannot pass
    without training having done anything.

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

    The check drives :func:`train_model` itself with ``fixed_batch`` and
    ``disable_regularization`` instead of re-implementing the loop: label
    wiring, the optimiser factory, clipping and the update order are then
    the same code the grids run, so a regression in the real loop cannot
    pass this pre-flight via a divergent copy.
    """
    if seed is not None:
        seed_everything(int(seed))
    sample_every = max(1, steps // 10)
    regularization = {}
    grad_flow = {}
    hist = train_model(
        model, steps=steps, batch=int(idx.size(0)), lr=lr, vocab=vocab,
        device=idx.device, log_every=sample_every,
        seed=seed, dense_lm=False,
        fixed_batch=(idx, tgt, meta), disable_regularization=True,
        grad_flow_info=grad_flow, regularization_info=regularization,
        batch_metrics=True, deterministic=deterministic)
    curve = [{"step": int(row["step"]), "loss": float(row["train_loss"]),
              "acc": float(row.get("batch_acc", float("nan")))}
             for row in hist]
    with torch.no_grad():
        final_logits, final_loss, _ = model(idx, tgt, sup_pos=meta["query_pos"])
        final_acc = float((final_logits.argmax(-1)
                           == meta["answer"]).to(torch.float64).mean())
    curve.append({"step": int(steps), "loss": float(final_loss.detach()),
                  "acc": final_acc})

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
        "learned": bool(curve[-1]["loss"] < curve[0]["loss"]),
        "passed": bool(curve[-1]["loss"] <= target_loss
                       and curve[-1]["acc"] >= target_acc
                       and curve[-1]["loss"] < curve[0]["loss"]
                       and grad_flow["all_params_connected"]
                       and grad_flow["all_grads_finite"]),
        "target_loss": float(target_loss),
        "target_acc": float(target_acc),
    }


def _validation_metrics(model, val_gen, n_pairs, vocab, device, batch, batches,
                        dense_lm):
    """Validation loss (and accuracy) on fixed, held-out validation batches.

    Uses eval mode + ``inference_mode`` (no dropout, no RNG consumption), so
    the training RNG stream - and therefore the training trajectory - is
    unaffected by monitoring.  Each module's previous training/eval flag is
    restored afterwards, so a diagnostic run with dropout force-disabled
    does not have it silently re-enabled.  Every validation batch passes the
    shared data-cleaning gate, so monitoring never reads invalid data.

    Returns ``(loss, accuracy, triage_batch, loss_se)`` where
    ``triage_batch`` is the ``(idx, targets, meta)`` of the first
    validation forward whose loss was non-finite, or the final forward when
    every loss was finite, so a non-finite validation loss can be run
    through the staged ``diagnose_nonfinite`` triage *on the exact batch
    that produced it*.  ``loss_se`` is the standard error of the per-batch
    losses (``std / sqrt(n_batches)``, sample std), i.e. the batch-sampling
    noise of the monitored loss: a validation regression smaller than a couple
    of SEs is not distinguishable from re-drawing the validation stream.
    """
    training_states = {module: module.training for module in model.modules()}
    model.eval()
    correct = torch.zeros((), dtype=torch.int64, device=device)
    total = 0
    last_batch = None
    first_bad_batch = None
    batch_losses = []
    val_gen.manual_seed(VAL_SEED)
    for i in range(max(1, batches)):
        # Batch construction stays outside inference_mode so the returned
        # triage batch remains usable by a grad-enabled triage/backward.
        idx, tgt, meta = make_batch(batch, n_pairs, vocab, device,
                                    dense_lm=dense_lm, gen=val_gen)
        validate_batch(idx, tgt, meta, vocab, diagnostics=False)
        last_batch = (idx, tgt, meta)
        sup_pos = None if dense_lm else meta["query_pos"]
        with torch.inference_mode():
            logits, loss, _ = model(idx, tgt, sup_pos=sup_pos)
            batch_losses.append(loss.detach())
            if (first_bad_batch is None
                    and not bool(torch.isfinite(loss))):
                first_bad_batch = last_batch
            if dense_lm:
                # Dense training supervises every position, so the forward
                # returns the full-sequence logits flattened row-major; the
                # recall accuracy still lives at the query position only.
                rows = torch.arange(idx.size(0), device=logits.device)
                qlogits = logits.reshape(
                    idx.size(0), idx.size(1), -1)[rows, meta["query_pos"]]
                correct.add_((qlogits.argmax(-1) == meta["answer"]).sum())
            else:
                correct.add_((logits.argmax(-1) == meta["answer"]).sum())
            total += idx.size(0)
    for module, training in training_states.items():
        module.train(training)
    n_batches = max(1, batches)
    losses = [float(x) for x in batch_losses]
    if len(losses) > 1:
        mean_loss = sum(losses) / len(losses)
        var = sum((x - mean_loss) ** 2 for x in losses) \
            / (len(losses) - 1)
        loss_se = math.sqrt(var / len(losses))
    else:
        loss_se = float("nan")
    return (sum(losses) / n_batches,
            correct.item() / total if total else float("nan"),
            first_bad_batch if first_bad_batch is not None else last_batch,
            loss_se)


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
    # Key columns only: drop the trailing SEP and query columns (see
    # ``cage.data.validate_batch``), so the SEP id can never enter the
    # key-uniqueness / answer-pairing judgment.
    keys = idx[:, :-2:2]
    match = keys.eq(query.unsqueeze(1))
    key_pos = match.to(torch.int64).argmax(dim=1).mul(2)
    paired = idx.gather(1, (key_pos + 1).unsqueeze(1)).squeeze(1)
    return match.sum(dim=1).eq(1) & paired.eq(meta["answer"])


# Mutually exclusive bad-case buckets, in priority order (see
# ``collect_stats_and_errors``).  Kept in one place so the per-type
# accumulators, the summary keys and the report tables cannot drift.
_ERROR_TYPES = ("label_error", "hard_structural", "retrieval_miss",
                "found_then_lost", "readout_wrong")


@torch.inference_mode()
def collect_stats_and_errors(model, n_pairs, vocab, device, batch=128,
                             batches=4, seed=STATS_SEED, max_cases=None):
    """One pass over the diagnostic stream producing both the per-layer
    mechanism statistics and the bad-case mining record.

    The layer statistics are read off the same forward that feeds the
    mining hooks.  ``max_cases=None`` collects statistics only (no hooks,
    no case taxonomy) and returns ``None`` for the error record.

    Statistics: per-layer gate / entropy / maxprob / null-mass at query
    positions, plus false-concentration labels (high max-prob but wrong
    answer).  Every batch passes the ``validate_batch`` cleaning gate, so
    the mechanism statistics are computed on the same validated data
    contract as training and the reported accuracy metrics.

    Error mining: each error is recorded with the evidence needed to
    classify *why* it failed:

    * ``confidence`` / ``margin``   - confidently wrong (systematic blind
      spot) vs low-confidence (ambiguous for the model);
    * ``attn_mass_on_key``          - mean attention mass on the correct
      key position in the last block: did the model fail to *find* the
      right key, or did it find it and then *read out* the wrong value?
    * ``attn_mass_on_key_by_layer`` - the same mass at *every* block, so
      a last-layer miss can be told apart from a key that was found in an
      intermediate block and lost again (the fixed readout composes
      match-then-shift, so a mid-stack hit that does not survive to the
      last block is a distinct failure mode);
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
      (attention mass < 0.5 on the correct key in *every* block) >
      ``found_then_lost`` (mass >= 0.5 in some block but < 0.5 in the
      last one) > ``readout_wrong`` (key still found in the last block,
      wrong value read out);
    * ``key_pair_idx``              - position of the matching pair;
      aggregated into early/mid/late error rates (long-range retrieval).

    The label audit makes "labels cannot be the cause" a *measured* claim:
    ``label_noise_rate`` is computed over every evaluated sample rather
    than assumed to be 0 from the construction.  Every batch additionally
    passes the full ``validate_batch`` cleaning gate, so a structurally
    invalid batch aborts the analysis instead of being taxonomised as a
    model error.

    The taxonomy is quantified, not just tabulated: each bucket carries
    its sample count and a Wilson interval over the error population
    (``taxonomy_<type>_count/_lo/_hi``), and the mean answer confidence
    and top-2 margin *within* each bucket (``conf_by_error_type`` /
    ``margin_by_error_type``).  A high-confidence ``retrieval_miss`` is a
    systematic blind spot; a low-confidence ``hard_structural`` error is
    task ambiguity; a confident ``label_error`` means the annotation is
    wrong rather than the model.  The overall ``error_rate`` and
    ``label_noise_rate`` also carry Wilson intervals, so small
    differences between variants are not read off binomial noise.

    Returns ``(stats, errors)``; cases are capped at ``max_cases`` but the
    summary aggregates every evaluated sample.  ``stats`` matches the
    ``collect_stats`` return value exactly.
    """
    cfg = model.cfg
    C, H = cfg.d_model, cfg.n_heads
    dk = C // H
    inv_scale = dk ** -0.5
    gen = torch.Generator(device=device).manual_seed(seed)
    training_states = {module: module.training for module in model.modules()}
    model.eval()

    with_errors = max_cases is not None

    # Per-batch mechanism statistics (``collect_stats``).
    stats_out = {"gate": [], "entropy": [], "maxprob": [], "null_mass": [],
                 "correct": [], "resid_norm": []}

    per_layer_mass: dict[int, torch.Tensor] = {}
    state: dict = {"meta": None}
    handles = []

    def _make_hook(layer_idx, null_k):
        def _hook(module, inputs, output):
            meta = state["meta"]
            if meta is not None:
                qk = getattr(module, "_qk_capture", None)
                if qk is not None:
                    per_layer_mass[layer_idx] = _attention_mass_from_qk(
                        qk[0], qk[1], meta, H, inv_scale, null_k)
                else:
                    per_layer_mass[layer_idx] = _attention_mass_on_key(
                        inputs[0], module.qkv.weight, meta, H, inv_scale,
                        null_k)
            module._qk_capture = None
        return _hook

    if with_errors:
        for i, blk in enumerate(model.blocks):
            blk.attn._capture_qk = True
            handles.append(blk.attn.register_forward_hook(
                _make_hook(i, getattr(blk.attn, "null_k", None))))

    # Device-side accumulators for the per-batch reductions.
    n_total = 0
    case_count = 0
    cases = []
    i64 = torch.int64
    flags_summary = {k: torch.zeros((), dtype=i64, device=device)
                     for k in ("high", "low", "in_ctx", "naive", "repeat")}
    taxonomy = {k: torch.zeros((), dtype=i64, device=device)
                for k in _ERROR_TYPES}
    # Per-error-type confidence / margin sums: the taxonomy alone says
    # *what* failed; the confidence attached to each bucket says whether
    # the failure is a confident blind spot (high confidence on a wrong
    # answer) or an ambiguous / label-noisy sample the model was unsure
    # about - the distinction that drives the next improvement.
    type_conf_sum = {k: torch.zeros((), dtype=torch.float64, device=device)
                     for k in _ERROR_TYPES}
    type_margin_sum = {k: torch.zeros((), dtype=torch.float64, device=device)
                       for k in _ERROR_TYPES}
    type_count = {k: torch.zeros((), dtype=i64, device=device)
                  for k in _ERROR_TYPES}
    n_err = torch.zeros((), dtype=i64, device=device)
    n_label_err = torch.zeros((), dtype=i64, device=device)
    mass_err = torch.zeros((), dtype=torch.float64, device=device)
    mass_err_n = torch.zeros((), dtype=i64, device=device)
    mass_corr = torch.zeros((), dtype=torch.float64, device=device)
    mass_corr_n = torch.zeros((), dtype=i64, device=device)
    layer_mass_err = None   # per-layer sums over error samples (float64)
    layer_mass_corr = None
    n_layers_seen = 0
    tertile_n = torch.zeros(3, dtype=i64, device=device)
    tertile_err = torch.zeros(3, dtype=i64, device=device)
    key_wrong_total = torch.zeros((), dtype=torch.float64, device=device)
    key_wrong_n = torch.zeros((), dtype=i64, device=device)
    key_right_total = torch.zeros((), dtype=torch.float64, device=device)
    key_right_n = torch.zeros((), dtype=i64, device=device)
    tertile_denom = max(1, n_pairs)
    try:
        for i in range(batches):
            idx, tgt, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
            validate_batch(idx, tgt, meta, vocab, diagnostics=False)
            per_layer_mass.clear()
            state["meta"] = meta
            logits, _, layer_stats = model(idx, stat_pos=meta["query_pos"])
            state["meta"] = None

            pred_stats = logits.squeeze(1).argmax(-1)
            stats_out["correct"].append((pred_stats == meta["answer"]).float())
            ents, maxps, nulls, gates, rns = [], [], [], [], []
            for s in layer_stats:
                ents.append(s["entropy"].mean(1))
                maxps.append(s["maxprob"].max(1)[0])
                rns.append(s["resid_norm"])
                if s["gate"] is not None:
                    gates.append(s["gate"].mean(-1))
                if s["null_mass"] is not None:
                    nulls.append(s["null_mass"].mean(1))
            stats_out["entropy"].append(torch.stack(ents, 1))
            stats_out["maxprob"].append(torch.stack(maxps, 1))
            stats_out["resid_norm"].append(torch.stack(rns, 1))
            if gates:
                stats_out["gate"].append(torch.stack(gates, 1))
            if nulls:
                stats_out["null_mass"].append(torch.stack(nulls, 1))
            if not with_errors:
                continue

            layered = torch.stack(
                [per_layer_mass[i] for i in sorted(per_layer_mass)], dim=0)
            mass_last, mass_mean = layered[-1], layered.mean(0)
            if layer_mass_err is None:
                layer_mass_err = torch.zeros(
                    layered.size(0), dtype=torch.float64, device=device)
                layer_mass_corr = torch.zeros_like(layer_mass_err)
                n_layers_seen = layered.size(0)
            probs = logits.squeeze(1).softmax(-1)
            conf, pred = probs.max(-1)
            top2 = probs.topk(2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            answer = meta["answer"]
            wrong = pred != answer

            n_total += idx.size(0)
            n_err += wrong.sum()
            label_ok = _label_audit(idx, meta)
            n_label_err += (~label_ok).sum()
            key_pair_idx = (meta["key_pos"] // 2).clamp(max=n_pairs - 1)
            third = (key_pair_idx * 3 // tertile_denom).clamp_(max=2).to(i64)
            tertile_n += torch.bincount(third, minlength=3)
            tertile_err += torch.bincount(third[wrong], minlength=3)
            m_e = mass_last[wrong]
            m_c = mass_last[~wrong]
            mass_err += m_e.sum().double()
            mass_err_n += m_e.numel()
            mass_corr += m_c.sum().double()
            mass_corr_n += m_c.numel()
            layer_mass_err += layered[:, wrong].sum(dim=1).double()
            layer_mass_corr += layered[:, ~wrong].sum(dim=1).double()

            flags = _error_case_flags(idx, meta, pred, n_pairs)
            flags_summary["high"] += (wrong & (conf >= 0.9)).sum()
            flags_summary["low"] += (wrong & (conf < 0.5)).sum()
            flags_summary["in_ctx"] += (wrong & flags["pred_in_context"]).sum()
            flags_summary["naive"] += (wrong & flags["naive_confusable"]).sum()
            flags_summary["repeat"] += (wrong & flags["value_repeats"]).sum()
            key_right = mass_last >= 0.5
            # A key that clears the mass threshold in *some* block but not
            # in the last one was found mid-stack and lost again; calling
            # that a plain retrieval miss would hide where retrieval broke.
            found_any = (layered >= 0.5).any(dim=0)
            key_wrong_total += mass_last[wrong & ~key_right].sum().double()
            key_wrong_n += (wrong & ~key_right).sum()
            key_right_total += mass_last[wrong & key_right].sum().double()
            key_right_n += (wrong & key_right).sum()

            ambiguous = flags["naive_confusable"] | flags["value_repeats"]
            key_missed = ~key_right
            lost = key_missed & found_any
            never_found = key_missed & ~found_any
            masks = {
                "label_error": wrong & ~label_ok,
                "hard_structural": wrong & label_ok & ambiguous,
                "retrieval_miss": wrong & label_ok & ~ambiguous & never_found,
                "found_then_lost": wrong & label_ok & ~ambiguous & lost,
                "readout_wrong": wrong & label_ok & ~ambiguous & ~key_missed,
            }
            for etype, mask in masks.items():
                taxonomy[etype] += mask.sum()
                type_count[etype] += mask.sum()
                type_conf_sum[etype] += conf[mask].sum().double()
                type_margin_sum[etype] += margin[mask].sum().double()

            if case_count < max_cases:
                rows = torch.nonzero(wrong, as_tuple=False).flatten().tolist()
                if rows:
                    # Pull the per-sample vectors to host for the case
                    # loop below.
                    conf_l = conf.tolist()
                    margin_l = margin.tolist()
                    mass_last_l = mass_last.tolist()
                    mass_mean_l = mass_mean.tolist()
                    mass_layers_l = layered.transpose(0, 1).tolist()
                    pred_l = pred.tolist()
                    answer_l = answer.tolist()
                    kp_l = key_pair_idx.tolist()
                    label_ok_l = label_ok.tolist()
                    ambiguous_l = ambiguous.tolist()
                    key_missed_l = key_missed.tolist()
                    never_found_l = never_found.tolist()
                    pic_l = flags["pred_in_context"].tolist()
                    nc_l = flags["naive_confusable"].tolist()
                    vr_l = flags["value_repeats"].tolist()
                    query_pos_l = meta["query_pos"].tolist()
                    key_pos_l = meta["key_pos"].tolist()
                    for b in rows:
                        if case_count >= max_cases:
                            break
                        if not label_ok_l[b]:
                            etype = "label_error"
                        elif ambiguous_l[b]:
                            etype = "hard_structural"
                        elif key_missed_l[b]:
                            etype = ("retrieval_miss" if never_found_l[b]
                                     else "found_then_lost")
                        else:
                            etype = "readout_wrong"
                        context = [int(t) for t in idx[b].tolist()]
                        cases.append({
                            "context": context,
                            "query_pos": int(query_pos_l[b]),
                            "query_key": context[query_pos_l[b]],
                            "key_pos": int(key_pos_l[b]),
                            "key_pair_idx": int(kp_l[b]),
                            "answer": int(answer_l[b]),
                            "pred": int(pred_l[b]),
                            "confidence": float(conf_l[b]),
                            "margin": float(margin_l[b]),
                            "attn_mass_on_key": float(mass_last_l[b]),
                            "attn_mass_on_key_mean_layers": float(mass_mean_l[b]),
                            "attn_mass_on_key_by_layer": [
                                float(v) for v in mass_layers_l[b]],
                            "pred_in_context": bool(pic_l[b]),
                            "naive_confusable": bool(nc_l[b]),
                            "value_repeats": bool(vr_l[b]),
                            "label_ok": bool(label_ok_l[b]),
                            "error_type": etype,
                        })
                        case_count += 1
    finally:
        for hd in handles:
            hd.remove()
        for blk in model.blocks:
            blk.attn._capture_qk = False
            blk.attn._qk_capture = None
        for module, training in training_states.items():
            module.train(training)

    stats = {k: torch.cat(v).cpu() for k, v in stats_out.items() if v}
    release_eval_scratch()
    if not with_errors:
        return stats, None

    n_err = int(n_err)
    n_label_err = int(n_label_err)
    flags_summary = {k: int(v) for k, v in flags_summary.items()}
    taxonomy = {k: int(v) for k, v in taxonomy.items()}
    type_count = {k: int(v) for k, v in type_count.items()}
    type_conf_mean = {
        k: (float(type_conf_sum[k]) / type_count[k]) if type_count[k]
        else float("nan") for k in _ERROR_TYPES}
    type_margin_mean = {
        k: (float(type_margin_sum[k]) / type_count[k]) if type_count[k]
        else float("nan") for k in _ERROR_TYPES}
    err_ci = wilson_ci(n_err, n_total)
    label_ci = wilson_ci(n_label_err, n_total)
    mass_err, mass_err_n = float(mass_err), int(mass_err_n)
    mass_corr, mass_corr_n = float(mass_corr), int(mass_corr_n)
    tertile_n = [int(v) for v in tertile_n.tolist()]
    tertile_err = [int(v) for v in tertile_err.tolist()]
    key_wrong_total, key_wrong_n = float(key_wrong_total), int(key_wrong_n)
    key_right_total, key_right_n = float(key_right_total), int(key_right_n)
    if layer_mass_err is not None:
        layer_mass_err = [float(v) for v in layer_mass_err.tolist()]
        layer_mass_corr = [float(v) for v in layer_mass_corr.tolist()]

    def _share(count):
        return count / n_err if n_err else float("nan")

    summary = {
        "n_samples": int(n_total),
        "n_errors": int(n_err),
        "error_rate": (n_err / n_total) if n_total else float("nan"),
        "error_rate_lo": err_ci["lo"],
        "error_rate_hi": err_ci["hi"],
        "n_label_errors": int(n_label_err),
        "label_noise_rate": (n_label_err / n_total) if n_total else float("nan"),
        "label_noise_rate_lo": label_ci["lo"],
        "label_noise_rate_hi": label_ci["hi"],
        "conf_by_error_type": dict(type_conf_mean),
        "margin_by_error_type": dict(type_margin_mean),
        "taxonomy_label_error": _share(taxonomy["label_error"]),
        "taxonomy_hard_structural": _share(taxonomy["hard_structural"]),
        "taxonomy_retrieval_miss": _share(taxonomy["retrieval_miss"]),
        "taxonomy_found_then_lost": _share(taxonomy["found_then_lost"]),
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
    for etype in _ERROR_TYPES:
        ci = wilson_ci(taxonomy[etype], n_err)
        summary[f"taxonomy_{etype}_count"] = int(taxonomy[etype])
        summary[f"taxonomy_{etype}_lo"] = ci["lo"]
        summary[f"taxonomy_{etype}_hi"] = ci["hi"]
    summary["attn_mass_gap"] = summary["attn_mass_on_key_correct"] - \
        summary["attn_mass_on_key_error"]
    if layer_mass_err is not None:
        summary["attn_mass_on_key_error_by_layer"] = [
            (v / mass_err_n) if mass_err_n else float("nan")
            for v in layer_mass_err]
        summary["attn_mass_on_key_correct_by_layer"] = [
            (v / mass_corr_n) if mass_corr_n else float("nan")
            for v in layer_mass_corr]
        summary["attn_mass_on_key_layer_gap"] = [
            c - e for c, e in zip(summary["attn_mass_on_key_correct_by_layer"],
                                  summary["attn_mass_on_key_error_by_layer"])]
        summary["n_layers_seen"] = int(n_layers_seen)
    return stats, {"cases": cases, "summary": summary}


@torch.inference_mode()
def collect_error_cases(model, n_pairs, vocab, device, batch=128, batches=4,
                        seed=STATS_SEED, max_cases=64):
    """Bad-case mining: dump and taxonomise the samples the model gets wrong.

    Single-purpose wrapper around ``collect_stats_and_errors``; returns
    ``{"cases": [...], "summary": {...}}`` with cases capped at
    ``max_cases`` while the summary aggregates every evaluated sample.
    """
    _, errors = collect_stats_and_errors(
        model, n_pairs, vocab, device, batch=batch, batches=batches, seed=seed,
        max_cases=max_cases)
    return errors


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
    from .model import _arange  # local: pinned helper, no cycle

    B, T, _ = x.shape
    C = weight.size(0) // 3
    dk = C // n_heads
    qp = meta["query_pos"]
    kp = meta["key_pos"]
    rows = _arange(B, x.device)
    q_all = F.linear(x[rows, qp], weight[:C]).view(B, n_heads, dk)
    k_all = F.linear(x, weight[C:2 * C]).view(B, T, n_heads, dk)
    k_all = k_all.transpose(1, 2)
    # Batched query-key GEMM in head-major layout (no ``bth`` transpose).
    scores = torch.matmul(q_all.unsqueeze(2),
                          k_all.transpose(-2, -1)).mul_(inv_scale)
    cols = _arange(T, x.device).view(1, 1, 1, T)
    scores.masked_fill_(cols > qp.view(B, 1, 1, 1), float("-inf"))
    if null_k is not None:
        s0 = torch.einsum("bhd,hd->bh", q_all, null_k).mul_(inv_scale)
        scores = torch.cat([s0.view(B, n_heads, 1, 1), scores], dim=-1)
        tapped = torch.softmax(scores, dim=-1).gather(
            3, (kp + 1).view(B, 1, 1, 1).expand(B, n_heads, 1, 1))
    else:
        tapped = torch.softmax(scores, dim=-1).gather(
            3, kp.view(B, 1, 1, 1).expand(B, n_heads, 1, 1))
    return tapped.mean(dim=(1, 2, 3))


def _attention_mass_from_qk(q, k, meta, n_heads, inv_scale, null_k):
    """Attention mass on the correct key from already-projected ``q`` / ``k``.

    ``q`` / ``k`` are the ``(B*h, T, dk)`` tensors the fixed kernel just used
    (``Attention._qk_capture``).  The mass is computed exactly as in
    :func:`_attention_mass_on_key`.
    """
    from .model import _arange  # local: pinned helper, no cycle

    bh, T, dk = k.shape
    B = q.size(0) // n_heads
    h = n_heads
    qp = meta["query_pos"]
    kp = meta["key_pos"]
    rows = _arange(B, q.device)
    q_all = q.view(B, h, T, dk)[rows, :, qp]
    # Match the pre-hook's head-major key layout exactly so the diagnostics
    # stay consistent with the recomputed-key path.
    k_all = k.view(B * h, T, dk)
    scores = torch.bmm(q_all.reshape(B * h, 1, dk),
                       k_all.transpose(1, 2)).view(B, h, 1, T).mul_(inv_scale)
    cols = _arange(T, q.device).view(1, 1, 1, T)
    scores.masked_fill_(cols > qp.view(B, 1, 1, 1), float("-inf"))
    if null_k is not None:
        s0 = torch.einsum("bhd,hd->bh", q_all, null_k).mul_(inv_scale)
        scores = torch.cat([s0.view(B, h, 1, 1), scores], dim=-1)
        tapped = torch.softmax(scores, dim=-1).gather(
            3, (kp + 1).view(B, 1, 1, 1).expand(B, h, 1, 1))
    else:
        tapped = torch.softmax(scores, dim=-1).gather(
            3, kp.view(B, 1, 1, 1).expand(B, h, 1, 1))
    return tapped.mean(dim=(1, 2, 3))


def _error_case_flags(idx, meta, pred, n_pairs):
    """Per-sample structural flags used to classify an error's cause."""
    from .model import _arange  # local: pinned helper, no cycle

    T = idx.size(1)
    pos = _arange(T, idx.device).view(1, -1)
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
    """Recall accuracy at the query position.

    Every batch of the stream passes the same ``validate_batch`` cleaning
    gate as training, so a reported accuracy can never be computed on
    structurally invalid or label-noisy data.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    training_states = {module: module.training for module in model.modules()}
    model.eval()
    correct = torch.zeros((), dtype=torch.int64, device=device)
    total = 0
    for i in range(batches):
        idx, tgt, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        validate_batch(idx, tgt, meta, vocab, diagnostics=False)
        logits, _, _ = model(idx)
        pred = logits.squeeze(1).argmax(-1)
        correct.add_((pred == meta["answer"]).sum())
        total += idx.size(0)
    for module, training in training_states.items():
        module.train(training)
    release_eval_scratch()
    return correct.item() / total if total else float("nan")


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
    headline number).  Every batch passes the ``validate_batch`` cleaning
    gate, so the metric panel is only computed on data that satisfies the
    same integrity/label invariants as training.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    training_states = {module: module.training for module in model.modules()}
    model.eval()
    logits_all, tgt_all = [], []
    for i in range(batches):
        idx, tgt, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        validate_batch(idx, tgt, meta, vocab, diagnostics=False)
        logits, _, _ = model(idx)
        logits_all.append(logits.squeeze(1).float())
        tgt_all.append(meta["answer"])
    for module, training in training_states.items():
        module.train(training)
    if logits_all:
        logits_cat = torch.cat(logits_all, dim=0).cpu()
        tgt_cat = torch.cat(tgt_all, dim=0).cpu()
    else:
        logits_cat = torch.empty((0, vocab))
        tgt_cat = torch.empty((0,), dtype=torch.long)
    # Reported metrics must be as traceable as training: if any evaluation
    # row carries non-finite logits, its accuracy/NLL/calibration values are
    # meaningless.  Stamp the finite fraction on every metric record and say
    # so loudly once, instead of silently averaging poisoned rows.
    finite_rows = torch.isfinite(logits_cat).all(dim=-1)
    out = multiclass_metrics(logits_cat, tgt_cat, topk=topk)
    out["finite_logit_frac"] = float(finite_rows.float().mean())
    if not bool(finite_rows.all()):
        print(f"[eval] WARNING: {int((~finite_rows).sum())}/"
              f"{finite_rows.numel()} evaluation rows have non-finite "
              f"logits; accuracy/NLL/calibration over those rows are not "
              f"interpretable (see cage.train.diagnose_nonfinite)")
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
    release_eval_scratch()
    return out


@torch.inference_mode()
def collect_stats(model, n_pairs, vocab, device, batch=128, batches=4,
                  seed=STATS_SEED):
    """Collect per-layer gate / entropy / maxprob / null-mass at query positions,
    plus false-concentration labels (high max-prob but wrong answer).

    Every batch passes the ``validate_batch`` cleaning gate, so the
    mechanism statistics are computed on the same validated data contract
    as training and the reported accuracy metrics.  Single-purpose wrapper
    around ``collect_stats_and_errors`` (statistics-only mode).
    """
    stats, _ = collect_stats_and_errors(
        model, n_pairs, vocab, device, batch=batch, batches=batches, seed=seed)
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
    if x.numel() == 0 or y.numel() != x.numel():
        return torch.tensor(float("nan"))
    dx = x - x.mean()
    dy = y - y.mean()
    # Constant-input guard relative to each vector's own magnitude.  The
    # floor keeps an all-zero input degenerate without turning every
    # small-valued vector (e.g. gates saturated near 0) into a false NaN.
    if (dx.abs().max() <= 1e-12 + 1e-6 * x.abs().max()
            or dy.abs().max() <= 1e-12 + 1e-6 * y.abs().max()):
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
    if x.numel() == 0 or y.numel() != x.numel():
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
    r_list = [pearson(g[:, i], ent[:, i]).item() for i in cols]
    s_list = [spearman(g[:, i], ent[:, i]).item() for i in cols]
    r_list_m = [pearson(g[:, i], mp[:, i]).item() for i in cols]
    s_list_m = [spearman(g[:, i], mp[:, i]).item() for i in cols]
    r_list_c = [pearson(g[:, i], corr).item() for i in cols]
    r["gate_vs_entropy"] = _fisher_mean(r_list)
    r["gate_vs_maxprob"] = _fisher_mean(r_list_m)
    r["gate_vs_correct"] = _fisher_mean(r_list_c)
    r["gate_vs_entropy_spearman"] = _fisher_mean(s_list)
    r["gate_vs_maxprob_spearman"] = _fisher_mean(s_list_m)
    # Quantify cross-layer variability (complement Fisher-mean summary)
    def _zmean_std(rs):
        vals = [x for x in rs if x == x]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        zs = [math.atanh(min(max(x, -0.999999), 0.999999)) for x in vals]
        m = sum(zs) / len(zs)
        if len(zs) < 2:
            return {"mean": math.tanh(m), "std": float("nan"), "n": len(vals)}
        var = sum((z - m) ** 2 for z in zs) / (len(zs) - 1)
        return {"mean": math.tanh(m), "std": math.sqrt(var), "n": len(vals)}
    r["gate_vs_entropy_by_layer"] = _zmean_std(r_list)
    r["gate_vs_maxprob_by_layer"] = _zmean_std(r_list_m)
    r["gate_vs_entropy_spearman_by_layer"] = _zmean_std(s_list)
    return r


def _false_concentration_at(mp, corr, gm, maxprob_thresh):
    """Threshold-specific readout over precomputed per-sample vectors."""
    high = mp > maxprob_thresh
    r = {
        "n_high_maxprob": int(high.sum()),
        "false_conc_rate": 1.0 - corr[high].float().mean().item() if high.any() else float("nan"),
        "overall_acc": corr.float().mean().item(),
    }
    if gm is not None:
        fc = high & ~corr
        r["gate_on_false_conc"] = gm[fc].mean().item() if fc.any() else float("nan")
        r["gate_on_correct"] = gm[corr & high].mean().item() if (corr & high).any() else float("nan")
    return r


def false_concentration_analysis(stats, maxprob_thresh=0.5):
    """'False concentration': max attention prob high but answer wrong.

    Computes, among high-maxprob samples, the error rate (false-concentration
    rate) and, for gated models, mean gate value on false-concentrated vs
    correct samples - a good calibrated gate should close (lower value)
    when concentration is false.
    """
    mp = stats["maxprob"].max(dim=1)[0]
    corr = stats["correct"].bool()
    g = stats.get("gate")
    gm = g.mean(1) if g is not None else None
    return _false_concentration_at(mp, corr, gm, maxprob_thresh)


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
    mp = stats["maxprob"].max(dim=1)[0]
    corr = stats["correct"].bool()
    g = stats.get("gate")
    gm = g.mean(1) if g is not None else None
    return {float(thr): _false_concentration_at(mp, corr, gm, thr)
            for thr in thresholds}


@torch.inference_mode()
def false_concentration_gate_test(stats, maxprob_thresh=0.5,
                                   n_boot: int = 10000, seed: int = 12345):
    """Bootstrap test of gate_on_correct - gate_on_false_conc for gated models.

    A calibrated gate should close (lower) on false concentration, so the
    difference ``gate_on_correct - gate_on_false_conc`` should be > 0
    and exclude 0 when the mechanism works.
    """
    from . import metrics

    mp = stats["maxprob"].max(dim=1)[0]
    corr = stats["correct"].bool()
    g = stats.get("gate")
    if g is None:
        return {"applicable": False}
    gm = g.mean(1)
    high = mp > maxprob_thresh
    fc = high & ~corr
    cc = high & corr
    if not bool(fc.any()) or not bool(cc.any()):
        return {"applicable": False, "n_fc": int(fc.sum()), "n_cc": int(cc.sum())}
    diffs = (gm[cc] - gm[fc]).float()
    ci = metrics.paired_bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
    ci["n_fc"] = int(fc.sum())
    ci["n_cc"] = int(cc.sum())
    ci["applicable"] = True
    return ci


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
    (``no_errors``, ``extreme`` at 1:1000 or worse, ``high`` at 1:10 or
    worse, else ``balanced``).  A stream with **no negatives** (every
    answer wrong, prevalence 1.0) is not "balanced": the ranking metrics
    are undefined there, so it gets its own ``all_errors`` regime instead
    of being confusable with a genuinely usable balance in the report.
    Under ``extreme`` imbalance ROC-AUC grows with the
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
    if n == 0:
        regime = "no_samples (empty stream; all metrics undefined)"
    elif n_err == 0:
        regime = "no_errors"
    elif n_neg == 0:
        # Degenerate all-wrong stream: ROC/PR-AUC are NaN (no negatives to
        # rank against) and must not be presented as a usable "balanced"
        # error-detection readout.
        regime = "all_errors (no negatives; ranking metrics undefined)"
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
    praucs = {}
    for name in ("maxprob", "entropy", "gate"):
        score = scores.get(name)
        if score is None or n_err == 0 or n_err == n:
            praucs[name] = float("nan")
            out[f"rocauc_{name}_to_error"] = float("nan")
            out[f"prauc_{name}_to_error"] = float("nan")
            out[f"prauc_lift_{name}_to_error"] = float("nan")
            out[f"recall_at_10pct_{name}_to_error"] = float("nan")
            out[f"lift_at_10pct_{name}_to_error"] = float("nan")
            continue
        pa = pr_auc(score, err)
        praucs[name] = pa
        out[f"rocauc_{name}_to_error"] = roc_auc(score, err)
        out[f"prauc_{name}_to_error"] = pa
        out[f"prauc_lift_{name}_to_error"] = (
            pa / prevalence if prevalence > 0.0 else float("nan"))
        sel = selection_at_fraction(score, err, 0.1)
        out[f"recall_at_10pct_{name}_to_error"] = sel["recall"]
        out[f"lift_at_10pct_{name}_to_error"] = sel["lift"]
    # The headline must name a readable ranking: keep the pre-registered
    # preference order (gate > maxprob > entropy) but skip any score whose
    # PR-AUC is non-finite on this stream, so ``primary_metric`` never
    # points at NaN while a finite companion exists.
    primary = next((name for name in ("gate", "maxprob", "entropy")
                    if name in scores
                    and math.isfinite(praucs[name])), None)
    out["primary_metric"] = (f"prauc_{primary}_to_error" if primary
                             else "none (no finite PR-AUC on this stream)")
    return out
