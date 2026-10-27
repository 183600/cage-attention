# ruff: noqa: F401, F821
"""Training and evaluation utilities."""

from __future__ import annotations

import base64
import math
import operator
import warnings
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (
    DEFAULT_DIFFICULTY_KEY_BINS,
    STATS_SEED,
    VAL_SEED,
    association_codes,
    association_reference_codes,
    batch_fingerprint,
    check_train_seed,
    difficulty_strata,
    stream_content_digest,
    difficulty_stratum_labels,
    make_batch,
    make_length_linked_batch,
    stream_seed_key,
    validate_batch,
)
from .metrics import (
    _bootstrap_args_ok,
    _bootstrap_seed,
    _float_or_nan,
    _positive_integer,
    average_ranks,
    auc_paired_bootstrap_ci,
    brier_score,
    cohens_d_paired,
    expected_calibration_error,
    hedges_g_paired,
    mcnemar_test,
    multiclass_metrics,
    pr_auc,
    pr_auc_ci,
    roc_auc,
    roc_auc_ci,
    selection_at_fraction,
    wilson_ci,
)
from .model import release_eval_scratch
from .repro import configure_determinism, seed_everything, stream_generator


# Fixed optimisation recipe (design pin D5): a single source of truth for
# the hyperparameters the factory uses *and* for the ``training_recipe``
# block every run record stamps, so the controlled-comparison audit can
# verify optimizer equality from the records instead of assuming it.
_ADAMW_WEIGHT_DECAY = 0.01
_ADAMW_BETAS = (0.9, 0.95)
_GRAD_CLIP_MAX_NORM = 1.0
_ETA_MIN_FACTOR = 0.1

# ---------------------------------------------------------------------------
# Implementation layout
# ---------------------------------------------------------------------------
# The implementation is split by responsibility into the fragment files
# below.  They are compiled and executed into this module's namespace in
# order, so every helper keeps its original public/private name and module
# identity for callers and tests.
_future = __import__("__future__")
_pathlib = __import__("pathlib")

_FRAGMENT_DIR = _pathlib.Path(__file__).resolve().parent
_FRAGMENTS = (
    "_train_recipe.py",
    "_train_loop.py",
    "_train_eval.py",
)
for _fragment_name in _FRAGMENTS:
    _fragment_path = _FRAGMENT_DIR / _fragment_name
    exec(compile(_fragment_path.read_text(encoding="utf-8"),
                 str(_fragment_path), "exec",
                 flags=_future.annotations.compiler_flag,
                 dont_inherit=True), globals())
del _fragment_name, _fragment_path

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

def train_model(model, steps=1500, batch=64, n_pairs=16, lr=3e-4, vocab=512,
                device="cpu", log_every=250, seed=0, dense_lm=False,
                eval_every=0, val_batches=2, early_stop_patience=None,
                nonfinite="raise", grad_norm_explode=100.0,
                permute_labels=False, restore_best=True,
                early_stop_info=None, fixed_batch=None,
                disable_regularization=False, grad_flow_info=None,
                regularization_info=None, batch_metrics=False,
                deterministic=False, seed_info=None, eval_steps=None):
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
    * ``seed_info`` receives the exact :func:`cage.repro.seed_everything`
      result applied by this run (Python/NumPy/Torch/CUDA coverage,
      ``PYTHONHASHSEED``, and the determinism kernel switches actually
      requested).  Run records stamp it beside the curve, so "the run was
      seeded" is a measured, per-run fact rather than a static policy
      claim; with ``seed=None`` it records that no re-seeding happened.
    """
    # Semantic switches must be real booleans: a truthy string such as
    # ``"False"`` would silently enable label permutation / regularisation
    # changes / extra logging and change what the run measured.
    for name, value in (("dense_lm", dense_lm),
                        ("permute_labels", permute_labels),
                        ("restore_best", restore_best),
                        ("disable_regularization", disable_regularization),
                        ("batch_metrics", batch_metrics)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a bool, got {value!r}")
    raw_steps, raw_batch = steps, batch
    steps = _positive_integer(raw_steps)
    batch = _positive_integer(raw_batch)
    if steps is None:
        raise ValueError(
            f"steps must be a positive integer, got {raw_steps!r}")
    if batch is None:
        raise ValueError(
            f"batch must be a positive integer, got {raw_batch!r}")
    lr = _validated_lr(lr)
    grad_norm_explode = _validated_curve_threshold(
        grad_norm_explode, "grad_norm_explode")
    raw_log_every = log_every
    if isinstance(raw_log_every, bool):
        raise ValueError(
            f"log_every must be a non-negative integer, got "
            f"{raw_log_every!r}")
    try:
        log_every_num = float(raw_log_every)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            f"log_every must be a non-negative integer, got "
            f"{raw_log_every!r}") from exc
    if not (math.isfinite(log_every_num) and log_every_num >= 0.0
            and log_every_num == int(log_every_num)):
        raise ValueError(
            f"log_every must be a non-negative integer, got "
            f"{raw_log_every!r}")
    log_every = int(log_every_num)
    if nonfinite not in ("raise", "skip"):
        raise ValueError(
            f"nonfinite must be 'raise' or 'skip', got {nonfinite!r}: an "
            f"unrecognised mode would silently take the skip branch")
    raw_eval_every = eval_every
    if isinstance(raw_eval_every, bool):
        raise ValueError(
            f"eval_every must be a non-negative integer, got "
            f"{raw_eval_every!r}")
    try:
        eval_every_num = float(raw_eval_every)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            f"eval_every must be a non-negative integer, got "
            f"{raw_eval_every!r}") from exc
    if not (math.isfinite(eval_every_num) and eval_every_num >= 0.0
            and eval_every_num == int(eval_every_num)):
        raise ValueError(
            f"eval_every must be a non-negative integer, got "
            f"{raw_eval_every!r}")
    eval_every = int(eval_every_num)
    # Optional explicit validation steps.  ``eval_every`` can only produce
    # a periodic grid, which cannot hit an arbitrary configured number of
    # checks for every (steps, curve_points) pair; the experiment layer
    # passes the exact evenly-spaced schedule through this argument.
    if eval_steps is not None:
        if isinstance(eval_steps, (str, bytes)) or isinstance(eval_steps, bool):
            raise ValueError(
                f"eval_steps must be an iterable of integer step indices, "
                f"got {eval_steps!r}")
        try:
            raw_eval_steps = list(eval_steps)
        except TypeError as exc:
            raise ValueError(
                f"eval_steps must be an iterable of integer step indices, "
                f"got {eval_steps!r}") from exc
        checked = set()
        for raw_step in raw_eval_steps:
            if isinstance(raw_step, bool):
                raise ValueError(
                    f"eval_steps entries must be integers, got "
                    f"{raw_step!r}")
            try:
                step_index = operator.index(raw_step)
            except TypeError as exc:
                raise ValueError(
                    f"eval_steps entries must be integers, got "
                    f"{raw_step!r}") from exc
            if step_index < 0 or step_index >= steps:
                raise ValueError(
                    f"eval_steps entries must lie in [0, {steps}), got "
                    f"{step_index!r}")
            checked.add(int(step_index))
        eval_steps = frozenset(checked) or None
    if early_stop_patience is not None:
        raw_patience = early_stop_patience
        early_stop_patience = _positive_integer(raw_patience)
        if early_stop_patience is None:
            raise ValueError(
                f"early_stop_patience must be a positive integer, got "
                f"{raw_patience!r}")
    if eval_every > 0 or eval_steps is not None:
        raw_val_batches = val_batches
        val_batches = _positive_integer(raw_val_batches)
        if val_batches is None:
            raise ValueError(
                f"val_batches must be a positive integer, got "
                f"{raw_val_batches!r}")
    if seed is None:
        if fixed_batch is None:
            raise ValueError(
                "seed=None is only valid with fixed_batch: a stream-drawn "
                "run must pin its data seed")
        # The caller owns the RNG stream (fixed batch), but the determinism
        # option still belongs to this training run: skipping
        # ``seed_everything`` must not silently leave the previous global
        # kernel mode in place while the run record claims otherwise.
        determinism = configure_determinism(deterministic)
        if seed_info is not None:
            seed_info.update(determinism)
            seed_info.update({
                "seed": None,
                "reseeded": False,
                "reason": "fixed_batch caller owns the RNG state",
            })
    else:
        check_train_seed(seed)
        applied = seed_everything(seed, deterministic=deterministic)
        if seed_info is not None:
            seed_info.update(applied)
            seed_info["reseeded"] = True
    if early_stop_patience is not None and eval_every <= 0 \
            and eval_steps is None:
        raise ValueError(
            "early_stop_patience requires eval_every > 0 or eval_steps: "
            "there is no validation curve to stop on otherwise")
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
        # ``validate_batch`` accepts any integer dtype, but the embedding
        # and cross-entropy require long targets.  Normalise only after the
        # cleaning gate has accepted the values (so a float batch still
        # fails closed instead of being truncated here).
        if idx.dtype != torch.long:
            idx = idx.to(torch.long)
        if tgt.dtype != torch.long:
            tgt = tgt.to(torch.long)
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
        do_val = bool(
            (eval_every > 0
             and (step % eval_every == 0 or step == steps - 1))
            or (eval_steps is not None and step in eval_steps))
        if not math.isfinite(lv):
            diag = diagnose_nonfinite(model, idx, diag_tgt, meta, vocab,
                                      device=device, dense_lm=dense_lm)
            entry["diagnosis"] = diag
            print(f"  step {step:5d}  loss {lv}  <- non-finite; triage "
                  f"first_bad_stage={diag['first_bad_stage']}")
            hist.append(entry)
            opt.zero_grad(set_to_none=True)
            # The loop iteration still happened even though the update is
            # skipped.  Advance the cosine schedule so T_max=steps and the
            # stamped recipe describe the same iteration axis; otherwise a
            # skipped step leaves the LR one step behind and the run never
            # reaches eta_min.
            _step_scheduler_on_skip(sched)
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
            # Same iteration-axis accounting as the non-finite-loss branch:
            # a skipped gradient update must not pause the cosine schedule.
            _step_scheduler_on_skip(sched)
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
                                          device=device, dense_lm=dense_lm,
                                          train_mode=False)
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

@torch.inference_mode()
def eval_accuracy(model, n_pairs, vocab, device, batch=256, batches=4,
                  seed=VAL_SEED):
    """Recall accuracy at the query position.

    Every batch of the stream passes the same ``validate_batch`` cleaning
    gate as training, so a reported accuracy can never be computed on
    structurally invalid or label-noisy data.
    """
    raw_batch, raw_batches = batch, batches
    batch = _positive_integer(raw_batch)
    batches = _positive_integer(raw_batches)
    if batch is None:
        raise ValueError(
            f"batch must be a positive integer, got {raw_batch!r}")
    if batches is None:
        raise ValueError(
            f"batches must be a positive integer, got {raw_batches!r}")
    gen = torch.Generator(device=device).manual_seed(stream_seed_key(seed))
    training_states = {module: module.training for module in model.modules()}
    try:
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
        return correct.item() / total if total else float("nan")
    finally:
        for module, training in training_states.items():
            module.train(training)
        release_eval_scratch()

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
