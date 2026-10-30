# cage train: recipe, curve verdict, non-finite diagnostics and gradient flow.
# Executed into the wrapper module namespace by the loader;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _finite_float(value) -> float:
    """Best-effort float for a persisted curve field.

    Curve histories are reconstructed from JSON artifacts, so an optional
    numeric field can be missing, null, a numeric string, or malformed.  A
    malformed value must abstain (NaN) rather than abort the curve report:
    the non-finite verdict is what tells the reader the history is unusable.
    A boolean is a flag, not a measurement, so it must not become 0/1.
    """
    if isinstance(value, bool):
        return float("nan")
    if torch.is_tensor(value) and (value.is_complex()
                                   or value.dtype == torch.bool):
        # A complex/count-flag tensor is not a real measurement; converting
        # it through ``float`` would drop the imaginary part or read a flag
        # as 0/1.  Both are malformed curve fields.
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return float("nan")


def _validated_curve_threshold(value, name):
    """Coerce a curve threshold to a finite non-negative number.

    ``rel_tol`` and ``grad_norm_explode`` are compared against curve
    measurements, so a string/complex/NaN threshold would otherwise escape
    as an incidental ``TypeError``/``NotImplementedError`` from the
    comparison (or silently disable the check).  A boolean is a flag, not
    a threshold.
    """
    raw = value
    if isinstance(raw, bool):
        raise ValueError(
            f"{name} must be a finite non-negative number, got {raw!r}")
    try:
        number = float(raw)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a finite non-negative number, got "
            f"{raw!r}") from exc
    if not (math.isfinite(number) and number >= 0.0):
        raise ValueError(
            f"{name} must be a finite non-negative number, got {raw!r}")
    return number


def _non_negative_integer(value, name):
    """Coerce a non-negative integral option, rejecting flags and fractions.

    ``log_every`` and ``eval_every`` use the same conversion: a boolean is a
    switch rather than an interval, and a fractional or non-finite value
    would otherwise be truncated or raise halfway through the run.
    """
    raw = value
    if isinstance(raw, bool):
        raise ValueError(
            f"{name} must be a non-negative integer, got {raw!r}")
    try:
        number = float(raw)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a non-negative integer, got {raw!r}") from exc
    if not (math.isfinite(number) and number >= 0.0
            and number == int(number)):
        raise ValueError(
            f"{name} must be a non-negative integer, got {raw!r}")
    return int(number)


def _validated_lr(value):
    """Validate a learning rate before it reaches an optimizer.

    ``torch.optim`` accepts ``inf`` (and ``0.0``) without complaint, so a
    bad rate does not fail at construction: it either freezes training or
    corrupts every parameter on the first update while the run record still
    looks like a legitimate recipe.  Both training entry points normalise
    and reject such values up front, like the other numeric options.
    """
    raw = value
    if isinstance(raw, bool):
        raise ValueError(
            f"learning rate must be a finite positive number, got {raw!r}")
    try:
        lr = float(raw)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            f"learning rate must be a finite positive number, got "
            f"{raw!r}") from exc
    if not (math.isfinite(lr) and lr > 0.0):
        raise ValueError(
            f"learning rate must be a finite positive number, got {raw!r}")
    return lr




def _step_scheduler_on_skip(sched) -> None:
    """Advance the cosine schedule for a skipped (non-finite) iteration.

    The schedule is defined over loop iterations (``T_max=steps``), so a
    skipped update must not pause it.  Before the first optimizer step
    PyTorch warns about that intentional ordering; suppress only that known
    warning instead of leaving the LR one step behind for the rest of the
    run.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Detected call of")
        sched.step()


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
        "lr": _validated_lr(lr),
    }
    if steps is not None:
        raw_steps = steps
        steps = _positive_integer(raw_steps)
        if steps is None:
            raise ValueError(
                f"steps must be a positive integer, got {raw_steps!r}")
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
    rel_tol = _validated_curve_threshold(rel_tol, "rel_tol")
    grad_norm_explode = _validated_curve_threshold(
        grad_norm_explode, "grad_norm_explode")
    # Work on copies so the caller's history keeps the raw single-batch
    # losses; the smoothed window value only changes the verdict input when
    # the raw value is finite, so a non-finite batch loss is never hidden.
    # A history restored from JSON can carry malformed optional fields; each
    # numeric field is normalised once (malformed -> NaN) so the verdict
    # abstains instead of raising from ``float("bad")`` halfway through.
    try:
        raw_hist = list(hist)
    except TypeError:
        raw_hist = []
    hist = []
    for row in raw_hist:
        if not isinstance(row, Mapping):
            continue
        h = dict(row)
        # A point without a usable step is not a curve point: ``step`` is
        # used as the regression abscissa and the best-point readout.
        if not math.isfinite(_finite_float(h.get("step"))):
            continue
        for key in ("train_loss", "train_loss_window", "val_loss",
                    "val_loss_se", "val_acc", "grad_norm", "lr"):
            if key not in h:
                continue
            if h[key] is None:
                # ``train_loss`` is mandatory for a finite point; a null
                # value is a missing loss, not a literal "not recorded".
                if key == "train_loss":
                    h[key] = float("nan")
                continue
            h[key] = _finite_float(h[key])
        hist.append(h)
    for h in hist:
        raw = h.get("train_loss")
        window = h.get("train_loss_window")
        if (window is not None and math.isfinite(float(window))
                and (raw is None or math.isfinite(float(raw)))):
            h["train_loss"] = float(window)
    finite = [h for h in hist
              if math.isfinite(_finite_float(h.get("train_loss")))]

    def _bad(v):
        return v is not None and not math.isfinite(_finite_float(v))

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
    val = [h for h in finite if h.get("val_loss") is not None
           and math.isfinite(float(h["val_loss"]))]
    gap_first = float("nan")
    gap_last = float("nan")
    if val:
        # A single validation check still identifies the best point and the
        # final regression past it (which is exactly zero).  Those readouts
        # must not depend on a second check existing: ``curve_points=1`` is
        # a supported schedule, and returning NaN there would hide the
        # early-stop point from every single-check run.
        best = min(val, key=lambda h: h["val_loss"])
        out["best_val_loss"] = float(best["val_loss"])
        out["best_val_step"] = int(best["step"])
        out["val_regression_after_best"] = float(
            val[-1]["val_loss"] - best["val_loss"])
        common = [h for h in finite if h.get("val_loss") is not None]
        if common:
            gap_first = float(
                common[0]["val_loss"] - common[0]["train_loss"])
            gap_last = float(
                common[-1]["val_loss"] - common[-1]["train_loss"])
            out["generalization_gap"] = gap_last
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
    if len(val) >= 2:
        out["val_delta"] = rel_delta(val, "val_loss")
        if (len(val) >= 2 and math.isfinite(out["best_val_loss"])
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
    last_val_se = float("nan")
    if val and val[-1].get("val_loss_se") is not None:
        candidate = float(val[-1]["val_loss_se"])
        if math.isfinite(candidate):
            last_val_se = candidate
    out["val_loss_se_last"] = last_val_se
    if (out["late_overfit"] and math.isfinite(last_val_se)
            and last_val_se > 0.0
            and math.isfinite(out["val_regression_after_best"])):
        out["late_overfit_significant"] = bool(
            out["val_regression_after_best"] > 2.0 * last_val_se)
    if len(val) >= 2 and math.isfinite(last_val_se):
        # Is the monitored validation *movement* (first -> last) larger than
        # the batch-sampling noise of a single evaluation?  With only a
        # couple of validation batches the SE can be a sizeable fraction of
        # the loss, and a "flat" verdict read from a curve whose movement is
        # inside 2 SE says more about the monitor's resolution than about
        # the model.
        out["val_change_abs"] = float(val[-1]["val_loss"] - val[0]["val_loss"])
        out["val_noise_2se"] = float(2.0 * last_val_se)
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
    # The global overfit verdict is about the train/val *gap* widening, not
    # merely about the validation curve's first-to-last movement: a run
    # whose train loss collapses while validation stays flat is diverging
    # even if the val movement itself is small.  Require the gap change to
    # clear the validation-batch noise, as documented; without a noise
    # estimate the claim abstains.
    overfitting_significant = False
    if (out["verdict"] == "overfitting"
            and math.isfinite(gap_first) and math.isfinite(gap_last)
            and math.isfinite(last_val_se) and last_val_se > 0.0):
        overfitting_significant = bool(
            (gap_last - gap_first) > 2.0 * last_val_se)
    out["overfitting_significant"] = overfitting_significant
    return out


def diagnose_nonfinite(model, idx, tgt, meta, vocab, device="cpu",
                       grad_norm_explode: float = 100.0,
                       dense_lm: bool = False,
                       train_mode: bool = True):
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

    ``train_mode`` controls the mode of the diagnostic forward.  Training
    losses are produced with dropout active, so the default ``True``
    reproduces them; a non-finite *validation* loss is produced in eval
    mode and must be diagnosed with ``train_mode=False``, otherwise dropout
    changes the graph and the triage can blame a different stage.
    """
    grad_norm_explode = _validated_curve_threshold(
        grad_norm_explode, "grad_norm_explode")
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
    if train_mode:
        model.train()
    else:
        model.eval()
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
        invalid_attn_stats = []
        try:
            model.eval()
            with torch.inference_mode():
                _, _, layer_stats = model(idx, stat_pos=stat_pos)
            for layer_idx, stat in enumerate(layer_stats or ()):
                for key, value in stat.items():
                    if not torch.is_tensor(value) or not value.numel():
                        continue
                    label = f"layer {layer_idx}.{key}"
                    if not bool(torch.isfinite(value).all()):
                        report["nonfinite_attn_stats"].append(label)
                    elif key == "maxprob" and (
                            float(value.min()) < 0.0
                            or float(value.max()) > 1.0):
                        report["notes"].append(
                            f"{label} outside [0, 1] "
                            f"([{float(value.min()):.3g}, "
                            f"{float(value.max()):.3g}]): softmax numerics "
                            f"are invalid")
                        invalid_attn_stats.append(label)
                    elif key == "entropy" and float(value.min()) < 0.0:
                        report["notes"].append(
                            f"{label} is negative "
                            f"({float(value.min()):.3g}): log/div-by-zero "
                            f"inside the attention kernel")
                        invalid_attn_stats.append(label)
                    elif key in ("gate", "null_mass") and (
                            float(value.min()) < 0.0
                            or float(value.max()) > 1.0):
                        report["notes"].append(
                            f"{label} outside [0, 1] "
                            f"([{float(value.min()):.3g}, "
                            f"{float(value.max()):.3g}]): attention "
                            f"statistic is not a probability")
                        invalid_attn_stats.append(label)
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
        elif invalid_attn_stats:
            mark("attention_stats",
                 f"out-of-range attention statistics at "
                 f"{invalid_attn_stats[:6]} (expected entropy >= 0 and "
                 f"probabilities in [0, 1]; check the key clamps and the "
                 f"softmax/entropy kernel)")

    # ---- 3. loss / logits ------------------------------------------------
    loss_ok = torch.is_tensor(loss) and bool(torch.isfinite(loss))
    # Check the panel at its native precision.  Casting to float32 first
    # turns a finite float64 logit (e.g. 1e300) into +inf and falsely
    # reports "logits contain NaN/Inf"; the fp16-range note below still
    # flags magnitudes that would overflow under autocast.
    logits_ok = torch.is_tensor(logits) and bool(torch.isfinite(logits).all())
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
    """Derange the supervised labels across the batch (negative-control mode).

    Every row of ``tgt`` carries its own sample's label (or ``-1``), so a
    row permutation swaps which answer each sample is trained to predict.
    A plain random permutation can contain fixed points, and each fixed
    point leaves that row's *true* label in the negative-control training
    signal - with a finite batch this happens often enough (about one row
    per 64) to weaken the leakage control it is meant to provide.  A
    non-zero cyclic shift is a derangement: every row receives another
    sample's labels, so no true answer is trained on.  ``batch < 2`` has
    no other row to take labels from, so it is refused instead of silently
    training the control on unpermuted labels.

    When every row has exactly one supervised column (the sparse recall
    target), only the label *values* are shifted while each row keeps its
    own supervised column.  The receiving column is the one ``train_model``
    reads, so the control still trains every row; shifting whole rows would
    move the label to the donor's column and silently drop or crash rows
    whose query columns differ.
    """
    n = int(tgt.size(0))
    if n < 2:
        raise ValueError(
            "permute_labels needs at least two rows so every sample can "
            f"receive another sample's label, got batch={n}")
    shift = int(torch.randint(1, n, (1,), generator=gen,
                              device=tgt.device).item())
    order = (torch.arange(n, device=tgt.device) + shift) % n
    # Sparse targets (the recall task) carry exactly one supervised column
    # per row (the query position).  Shifting whole rows would move that
    # label to the donor's query column, which is usually *not* the
    # receiving row's query column: ``train_model`` supervises the original
    # column, sees ``-1`` there, and either trains on a fraction of the
    # batch or (when no query positions coincide) raises "no supervised
    # position".  Permute only the label values while keeping each row's
    # own supervised position, which is the same deranged label signal
    # without dropping rows or crashing the control.
    supervised = tgt.ne(-1)
    if bool(supervised.sum(dim=1).eq(1).all()):
        columns = supervised.to(torch.int64).argmax(dim=1)
        donor = tgt.gather(1, columns.unsqueeze(1)).squeeze(1)
        permuted = tgt.new_full(tgt.shape, -1)
        permuted.scatter_(1, columns.unsqueeze(1),
                          donor.index_select(0, order).unsqueeze(1))
        return permuted
    return tgt.index_select(0, order)




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
    NumPy / Torch through :func:`cage.repro.seed_everything` with the
    caller's ``deterministic`` setting, so the pre-flight cannot silently
    switch deterministic kernels off for whatever runs afterwards; the
    caller must still seed *before* building the model, because weight
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
    raw_steps = steps
    steps = _positive_integer(raw_steps)
    if steps is None:
        raise ValueError(
            f"steps must be a positive integer, got {raw_steps!r}")
    # Validate the pass/fail targets before running any training.  Converting
    # them only when the curve is scored (after the whole loop has run) means
    # a malformed target aborts an expensive check with an incidental
    # TypeError/ValueError, while a boolean would silently become 0/1.
    target_loss = _validated_curve_threshold(target_loss, "target_loss")
    raw_target_acc = target_acc
    if isinstance(raw_target_acc, bool):
        raise ValueError(
            f"target_acc must be a finite probability in [0, 1], got "
            f"{raw_target_acc!r}")
    try:
        target_acc = float(raw_target_acc)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            f"target_acc must be a finite probability in [0, 1], got "
            f"{raw_target_acc!r}") from exc
    if not (math.isfinite(target_acc) and 0.0 <= target_acc <= 1.0):
        raise ValueError(
            f"target_acc must be a finite probability in [0, 1], got "
            f"{raw_target_acc!r}")
    if seed is not None:
        # Pass the seed through unchanged: ``seed_everything`` rejects a
        # bool/fractional seed instead of truncating it, so the pre-flight
        # cannot silently run a different stream than the caller named.
        seed_everything(seed, deterministic=deterministic)
    sample_every = max(1, steps // 10)
    regularization = {}
    grad_flow = {}
    seeding = {}
    hist = train_model(
        model, steps=steps, batch=int(idx.size(0)), lr=lr, vocab=vocab,
        device=idx.device, log_every=sample_every,
        seed=seed, dense_lm=False,
        fixed_batch=(idx, tgt, meta), disable_regularization=True,
        grad_flow_info=grad_flow, regularization_info=regularization,
        batch_metrics=True, deterministic=deterministic,
        seed_info=seeding)
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
        "seeding": seeding,
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
    try:
        model.eval()
        correct = torch.zeros((), dtype=torch.int64, device=device)
        total = 0
        last_batch = None
        first_bad_batch = None
        batch_losses = []
        val_gen.manual_seed(stream_seed_key(VAL_SEED))
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
    finally:
        for module, training in training_states.items():
            module.train(training)
