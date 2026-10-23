"""Simple baselines for the associative-recall task.

Why these baselines
-------------------
* ``rule_pair_lookup``: the task has a closed-form solution - tokens at
  even positions are keys, keys are distinct per sequence, so the query
  key pins the value that follows it.  A parameter-free rule therefore
  defines the *ceiling*.  If it does not reach ~1.0, the data
  construction itself is invalid and numbers below it cannot be attributed
  to the model.
* ``rule_naive_last_match``: the same idea without the pair-structure
  prior (the last occurrence of the query token wins).  Its gap to the
  exact rule measures the value-copy-of-key ambiguity that a naive
  string matcher would suffer.
* ``copy_last_token``: trivial floor (echo the query token).
* ``LinearBagProbe``: a one-layer linear model on bag-of-token counts,
  tuned over a deliberately wide learning-rate grid (four rates spanning
  30x) on a *validation* seed and reported on a separate *test* seed.  It
  cannot represent key-value lookup, so it should stay near chance
  (``1 / (vocab - 2)``).  A complex model earns its keep only if it
  clearly beats this tuned probe *and* approaches the rule ceiling.
  "Near chance" is checked statistically (``probe_vs_chance_test``), not
  by eyeballing: a point estimate inside "a few x chance" with a wide CI
  is inconclusive, not evidence.  A grid point whose loss goes non-finite
  is stopped, recorded (``diverged_runs``) and excluded from selection
  (its accuracy is unknown, not a legitimate "bad" tuning result), so a
  diverged run can neither fake a tuned baseline nor silently poison the
  tuned estimate.

The transformer with ``gate="none"`` is the architecture-internal
baseline that the gate variants of E1 are compared against; these
external baselines calibrate what "good" even means on this task.  On
the learned side the ladder lives in ``cage.experiments``
(``_run_depth_ladder``, hypotheses HB6/HB7): the same transformer at
each ladder depth (default 1/2/4 layers), with the learning rate tuned
per depth on a validation seed and reported on the test seed.  The
shallowest rung is the simplest *learned* model that could plausibly
solve the task, and the ladder's saturation depth is the depth against
which any depth-based motivation must be read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from .data import (
    check_eval_seed,
    check_train_seed,
    make_batch,
    stream_seed_key,
    validate_batch,
)
from .metrics import (
    _bootstrap_seed,
    _float_or_nan,
    _positive_integer,
    bootstrap_upper_p,
    paired_bootstrap_ci,
    seeds_for_power,
    wilson_ci,
)
from .repro import seed_everything, stream_generator
from .train import _make_adamw, _validated_lr

__all__ = [
    "LinearBagProbe",
    "copy_last_token_stats",
    "linear_probe_accuracy",
    "probe_vs_chance_test",
    "rule_naive_last_match_stats",
    "rule_pair_lookup_stats",
    "train_linear_probe",
    "tune_linear_probe",
]


def _answer_meta(idx, meta):
    query = idx.gather(1, meta["query_pos"].unsqueeze(1)).squeeze(1)
    return query


@torch.inference_mode()
def _eval_predictor(predict, n_pairs, vocab, device, batch, batches, seed):
    """Score a rule-based predictor on fixed streams.

    Every batch passes the shared ``validate_batch`` cleaning gate, so the
    ceiling/floor baselines are measured on the same data contract as the
    learned models.
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
    # Same canonical stream key as ``cage.repro.stream_generator`` and
    # ``stream_fingerprint``: the baseline is scored on the exact
    # realization the run record versions, for aliased seeds too.
    gen = torch.Generator(device=device).manual_seed(stream_seed_key(seed))
    correct = torch.zeros((), dtype=torch.int64, device=device)
    total = 0
    extra = {}
    for i in range(batches):
        idx, tgt, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        validate_batch(idx, tgt, meta, vocab, diagnostics=False)
        pred, stats = predict(idx, meta)
        correct.add_((pred == meta["answer"]).sum())
        total += idx.size(0)
        for key, value in stats.items():
            acc = extra.get(key)
            if acc is None:
                acc = extra[key] = torch.zeros(
                    (), dtype=torch.float64, device=device)
            acc.add_(torch.as_tensor(value, dtype=torch.float64))
    n_correct = correct.item()
    ci = wilson_ci(n_correct, total)
    out = {"accuracy": n_correct / total if total else float("nan"),
           "n": total,
           "accuracy_lo": ci["lo"], "accuracy_hi": ci["hi"]}
    out.update({key: value.item() / max(batches, 1)
                for key, value in extra.items()})
    return out


def _predict_rule_pair(idx, meta):
    query = _answer_meta(idx, meta)
    # Key columns only: drop the trailing SEP and query columns (see
    # ``cage.data.validate_batch``).
    keys = idx[:, :-2:2]
    match = keys == query.unsqueeze(1)
    unique_rate = match.sum(dim=1).eq(1).to(torch.float64).mean()
    key_pos = match.to(torch.int64).argmax(dim=1).mul(2)
    pred = idx.gather(1, (key_pos + 1).unsqueeze(1)).squeeze(1)
    return pred, {"unique_key_rate": unique_rate}


def _predict_rule_naive(idx, meta):
    T = idx.size(1)
    query = _answer_meta(idx, meta)
    pos = torch.arange(T, device=idx.device).view(1, -1)
    visible = pos < meta["query_pos"].unsqueeze(1)
    match = (idx == query.unsqueeze(1)) & visible
    last = torch.where(match, pos, pos.new_full((), -1)).max(dim=1).values
    pred = idx.gather(1, (last + 1).clamp(min=0, max=T - 1).unsqueeze(1)).squeeze(1)
    return pred, {"visible_match_rate": match.any(dim=1).to(torch.float64).mean()}


def _predict_copy_last(idx, meta):
    pred = idx[:, -1]
    return pred, {}


def rule_pair_lookup_stats(n_pairs, vocab, device="cpu", batch=256, batches=4,
                           seed=123):
    """Exact structural rule (ceiling): key at an even position, answer is
    the next token.  Also reports the unique-key rate as a data sanity
    check (it must be 1.0 for the target to be well defined)."""
    return _eval_predictor(_predict_rule_pair, n_pairs, vocab, device,
                           batch, batches, seed)


def rule_naive_last_match_stats(n_pairs, vocab, device="cpu", batch=256,
                                batches=4, seed=123):
    """Naive rule (no pair-structure prior): last visible occurrence of the
    query token wins.  The accuracy gap to the exact rule is the
    value-copy-of-key ambiguity rate."""
    return _eval_predictor(_predict_rule_naive, n_pairs, vocab, device,
                           batch, batches, seed)


def copy_last_token_stats(n_pairs, vocab, device="cpu", batch=256, batches=4,
                          seed=123):
    """Trivial floor: echo the query token."""
    return _eval_predictor(_predict_copy_last, n_pairs, vocab, device,
                           batch, batches, seed)


class LinearBagProbe(torch.nn.Module):
    """Single learned linear map on bag-of-token counts (no attention, no
    nonlinearity, no positional information)."""

    def __init__(self, vocab: int):
        super().__init__()
        self.vocab = vocab
        self.linear = torch.nn.Linear(vocab, vocab)

    def forward(self, idx):
        counts = torch.zeros(idx.size(0), self.vocab, device=idx.device,
                             dtype=self.linear.weight.dtype)
        counts.scatter_add_(1, idx, torch.ones_like(idx, dtype=counts.dtype))
        return self.linear(counts)


def train_linear_probe(n_pairs, vocab, device="cpu", steps=400, batch=64,
                       lr=3e-4, seed=0, history: list | None = None,
                       deterministic=False, seed_info: dict | None = None):
    """Fit the probe with the same optimizer/schedule recipe pinned for the
    transformer (``_make_adamw``, cosine schedule, gradient clipping).

    ``seed`` must not collide with a held-out evaluation stream; every
    batch passes the data-cleaning gate (same protocol as
    ``train_model``).  When ``history`` is given, sampled
    ``{"step", "loss"}`` points (~20 per run plus the last step) are
    appended to it: the probe is part of the HB2 negative control, so its
    training curve must be as auditable as the transformer grids'.

    If the loss (or the pre-clip gradient norm) goes non-finite, the run
    stops *before* the update with a ``"nonfinite": True`` /
    ``"nonfinite_grad": True`` history entry instead of silently stepping
    NaN weights through the rest of the schedule: a diverged grid point
    has no measured accuracy and must be excluded from tuning, not read
    as "the probe cannot do better".

    ``seed_info`` optionally receives the actual
    :func:`cage.repro.seed_everything` result applied by this run, so a
    probe record can prove which RNGs were pinned instead of citing only
    the configured seed.
    """
    check_train_seed(seed)
    raw_steps = steps
    raw_batch = batch
    steps = _positive_integer(raw_steps)
    batch = _positive_integer(raw_batch)
    if steps is None:
        raise ValueError(
            f"steps must be a positive integer, got {raw_steps!r}")
    if batch is None:
        raise ValueError(
            f"batch must be a positive integer, got {raw_batch!r}")
    lr = _validated_lr(lr)
    applied = seed_everything(seed, deterministic=deterministic)
    if seed_info is not None:
        seed_info.update(applied)
        seed_info["reseeded"] = True
    gen = stream_generator(seed, device)
    model = LinearBagProbe(vocab).to(device)
    params = list(model.parameters())
    opt = _make_adamw(params, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.1)
    sample_every = max(1, steps // 20)
    model.train()
    for step in range(steps):
        idx, targets, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        validate_batch(idx, targets, meta, vocab, diagnostics=False)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(idx), meta["answer"])
        if not bool(torch.isfinite(loss)):
            if history is not None:
                history.append({"step": int(step),
                                "loss": float(loss.detach()),
                                "nonfinite": True})
            break
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not bool(torch.isfinite(grad_norm)):
            if history is not None:
                history.append({"step": int(step),
                                "loss": float(loss.detach()),
                                "nonfinite_grad": True})
            opt.zero_grad(set_to_none=True)
            break
        opt.step()
        sched.step()
        if history is not None and (step % sample_every == 0
                                    or step == steps - 1):
            history.append({"step": int(step),
                            "loss": float(loss.detach())})
    model.eval()
    return model


@torch.inference_mode()
def linear_probe_accuracy(model, n_pairs, vocab, device="cpu", batch=256,
                          batches=4, seed=123):
    """Probe accuracy on a held-out stream, every batch cleaning-gated like
    training."""
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
    correct = torch.zeros((), dtype=torch.int64, device=device)
    total = 0
    for i in range(batches):
        idx, tgt, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        validate_batch(idx, tgt, meta, vocab, diagnostics=False)
        pred = model(idx).argmax(dim=-1)
        correct.add_((pred == meta["answer"]).sum())
        total += idx.size(0)
    return correct.item() / total if total else float("nan")


def tune_linear_probe(n_pairs, vocab, device="cpu", lr_grid=(3e-3, 1e-3, 3e-4),
                      seeds=(0,), steps=400, steps_grid=None, batch=64,
                      val_seed=123, val_batches=4, probe_batch=256,
                      deterministic=False):
    """Pick the probe learning rate *and step budget* by mean validation
    accuracy over the probe seeds.  Model selection happens on ``val_seed``
    only; the chosen recipe is then reported on a disjoint test seed by the
    caller.

    Tuning only the learning rate at a fixed step budget can understate a
    simple baseline: a linear bag-of-tokens probe may need more optimisation
    steps than the transformer's default to reach its representational
    ceiling.  The 2-D grid (lr x steps) makes "the tuned probe stays near
    chance" less attributable to an arbitrarily short budget; the selection
    audit receives the actual 2-D grid so a winner at the step boundary is
    flagged.

    Robustness: a diverged grid point (non-finite loss/gradient, see
    ``train_linear_probe``) is counted in ``diverged_runs`` and its
    accuracy treated as ``-inf`` for selection - its validation number
    measures an exploded model, not the probe's representational ceiling.
    Non-finite training losses are excluded from ``train_loss_mean``.  An
    all-diverged grid is refused with ``RuntimeError`` (mirroring
    ``tune_shared_recipe``) rather than letting ``max`` pick an arbitrary
    unmeasured row.  This matters because the grid intentionally reaches
    high learning rates so that "the tuned probe stays near chance" cannot
    be attributed to an under-tuned strawman.
    """
    def _sequence(value, name):
        """Normalise a grid/seed iterable, rejecting scalars and bytes.

        A mapping is iterable but names its *keys*, not the grid values;
        treating it as a grid silently changes which learning rates/seeds
        the probe runs.  Reject it like the Settings grid validators do.
        """
        if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
            raise ValueError(
                f"{name} must be an iterable, got {type(value).__name__}")
        try:
            return tuple(value)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be an iterable, got {type(value).__name__}"
            ) from exc

    check_eval_seed(val_seed, selection=True)
    seeds = _sequence(seeds, "seeds")
    if not seeds:
        raise ValueError("seeds must contain at least one training seed")
    if stream_seed_key(val_seed) in {stream_seed_key(s) for s in seeds}:
        raise ValueError(
            f"validation seed {val_seed} is also a probe training seed; "
            f"selection data must be disjoint from fitting data"
        )
    lr_grid = _sequence(lr_grid, "lr_grid")
    if not lr_grid:
        raise ValueError("lr_grid must contain at least one learning rate")
    lr_grid = tuple(_validated_lr(lr) for lr in lr_grid)
    if steps_grid is None:
        raw_steps = (steps,)
    else:
        # An explicitly supplied grid must be an iterable even when it is
        # empty/falsy.  The old truthiness check silently treated scalar
        # ``0``/``False``/``""`` as "no grid" and fell back to ``steps``,
        # which is a different experiment than the caller named; only an
        # empty *iterable* keeps the documented legacy fallback.
        raw_steps = _sequence(steps_grid, "steps_grid")
        if not raw_steps:
            raw_steps = (steps,)
    step_values = tuple(_positive_integer(s) for s in raw_steps)
    if not step_values or any(s is None for s in step_values):
        raise ValueError(f"steps_grid must contain positive ints, got "
                         f"{raw_steps!r}")
    step_values = tuple(int(s) for s in step_values)
    table = []
    for lr in lr_grid:
        for train_steps in step_values:
            accs = []
            diverged_accs = []
            loss_means = []
            diverged = 0
            runs = []
            for seed in seeds:
                history = []
                seeding = {}
                model = train_linear_probe(n_pairs, vocab, device,
                                           steps=train_steps,
                                           batch=batch, lr=lr, seed=seed,
                                           history=history,
                                           deterministic=deterministic,
                                           seed_info=seeding)
                run_diverged = any("nonfinite" in p or "nonfinite_grad" in p
                                   for p in history)
                if run_diverged:
                    diverged += 1
                losses = [p["loss"] for p in history
                          if math.isfinite(p.get("loss", float("nan")))]
                if losses:
                    loss_means.append(sum(losses) / len(losses))
                acc = linear_probe_accuracy(model, n_pairs, vocab, device,
                                            batch=probe_batch,
                                            batches=val_batches,
                                            seed=val_seed)
                runs.append({"seed": int(seed), "val_accuracy": acc,
                             "diverged": bool(run_diverged),
                             "seeding": seeding,
                             "history": list(history)})
                if run_diverged:
                    # The run stopped at an arbitrary step: this accuracy
                    # is recorded for transparency but is not a
                    # completed-run measurement and does not enter
                    # selection.
                    if math.isfinite(acc):
                        diverged_accs.append(acc)
                elif math.isfinite(acc):
                    accs.append(acc)
            mean = (sum(accs) / len(accs)) if accs else float("nan")
            var = (sum((a - mean) ** 2 for a in accs)
                   / (len(accs) - 1) if len(accs) > 1 else 0.0)
            # A row with even one diverged seed is not a completed-run
            # measurement.  Averaging the surviving seeds only would let a
            # high-accuracy row measured on one seed beat a fully measured
            # row; treat the row as ineligible for selection instead (the
            # documented ``-inf`` convention), while keeping the completed
            # mean/std/count in the table for transparency.
            selection_mean = float("-inf") if diverged else mean
            row = {"lr": lr, "steps": train_steps,
                   "val_accuracy_mean": selection_mean,
                   "completed_accuracy_mean": mean,
                   "val_accuracy_std": var ** 0.5, "n_seeds": len(accs),
                   "diverged_runs": diverged, "runs": runs}
            if diverged_accs:
                row["diverged_accuracy_mean"] = (sum(diverged_accs)
                                                 / len(diverged_accs))
            if loss_means:
                row["train_loss_mean"] = sum(loss_means) / len(loss_means)
            table.append(row)
    # Same all-NaN guard as ``tune_shared_recipe`` / the depth ladder:
    # ``max`` over an all-non-finite key returns an arbitrary row, which
    # would hand the caller an unmeasured rate that is then re-trained and
    # certified against by HB2 - refuse instead of fabricating a selection.
    finite_rows = [row for row in table
                   if math.isfinite(row["val_accuracy_mean"])]
    if not finite_rows:
        raise RuntimeError(
            f"linear probe tuning produced no completed finite validation "
            f"accuracy across grid lr={tuple(lr_grid)} x steps={step_values}: "
            f"every grid point either diverged on at least one training seed "
            f"or returned NaN, so no fully measured recipe can be selected "
            f"(inspect the training histories; no HB2 near-chance read from "
            f"this run is interpretable)")
    best = max(finite_rows, key=lambda row: row["val_accuracy_mean"])
    return best["lr"], int(best["steps"]), table


def probe_vs_chance_test(accuracies, chance, n_boot: int = 10000,
                         seed: int = 0, n_per_seed: int | None = None) -> dict:
    """Quantitative HB2 test: does the tuned probe stay near chance?

    ``accuracies`` holds the test-seed accuracy of the tuned probe for each
    training seed; ``chance`` is ``1 / (vocab - 2)``.  The returned CI is
    for the mean of ``accuracy - chance``, and ``hb2_supported`` is the
    pre-registered decision: the probe counts as "near chance" only when
    the 95% CI upper bound is *below* ``chance`` itself (i.e. the probe is
    confidently below 2x chance).  This is a one-sided non-inferiority
    reading of the difference test: ``p_above_chance`` is the bootstrap
    p-value for ``H0: mean margin >= +chance``, so a small p certifies the
    probe below the 2x-chance bound while ``p >= 0.5`` says the point
    estimate does not even fall below it.

    ``n_per_seed`` is the number of evaluation examples behind *each*
    seed's accuracy.  When given (the production call always gives it),
    the interval is widened to account for a second uncertainty source
    that the plain seed bootstrap treats as absent: all seeds are scored
    on the *same* sealed evaluation stream, so each replicate draws seed
    accuracies with replacement, averages them, and then makes **one**
    shared ``Binomial(n_per_seed, mean_accuracy) / n_per_seed`` scoring
    draw (measurement error, not a claim that the probe is stochastic).
    Drawing one binomial per seed instead would treat the shared stream as
    independent evidence and shrink the interval by roughly
    ``sqrt(n_seeds)`` (pseudo-replication).  The primary
    ``lo``/``hi``/``p_value``/``p_above_chance`` then carry both seed noise
    and the common-stream scoring noise.  ``sd_effective`` is the total
    per-seed SD (``sqrt(sd_seed^2 + mean(p(1-p))/n_per_seed)``) for
    display, while planning uses the SD of the mean,
    ``se_effective = sqrt(sd_seed^2/n + mean(p(1-p))/n_per_seed)``; the
    stream term does not shrink with more seeds, so
    ``seeds_for_certification`` is ``inf`` when the observed margin cannot
    clear that noise floor.  Without this correction a near-bound run can
    look certifiable purely because a small evaluation stream was scored.
    The seed-level readouts remain available as ``*_seed`` companions, and
    ``measurement_noise_accounted`` records whether the widening was
    applied.  When ``n_per_seed`` is ``None`` (legacy/smoke callers) the
    behaviour is the historical one-level bootstrap.

    The three states are explicit, because "not certified" is *not* the
    same as "leakage shown":

    * ``certified`` - CI upper bound below ``chance``;
    * ``violated`` - CI lower bound at or above ``chance`` (the probe is
      confidently at/above 2x chance: answer information leaks into the
      bag-of-token features, or a shortcut exists; the simple-baseline
      comparison must be re-examined before crediting the transformer);
    * ``inconclusive`` - the CI straddles the bound.  This is the state a
      low-power run lands in (few seeds, small probe evaluation stream),
      so the result also carries ``seeds_for_certification``: how many
      independent seeds are needed at 80% power to shrink the seed term
      below the observed margin while leaving the irreducible common-stream
      term in place (``inf`` when the observed mean is not below the bound
      or the stream-noise floor alone exceeds the margin, since more seeds
      alone cannot rescue such a point estimate).

    Returns the bootstrap CI dict for ``accuracy - chance`` plus the
    ``chance`` level, ``hb2_supported`` (certified), ``hb2_violated``,
    ``hb2_state``, ``p_above_chance`` and ``seeds_for_certification``.
    """
    if isinstance(chance, bool):
        raise ValueError(
            f"chance must be a finite probability in [0, 0.5), got "
            f"{chance!r}; a boolean is a flag, not a chance level")
    chance = _float_or_nan(chance)
    # The pre-registered HB2 bound is "below 2x chance".  For chance >= 0.5
    # that bound is at or above 1.0, so even a perfect probe would be
    # "certified"; a negative/non-finite chance is not a probability.  Fail
    # closed instead of returning a vacuous pass.
    if not (math.isfinite(chance) and 0.0 <= chance < 0.5):
        raise ValueError(
            f"chance must be a finite probability in [0, 0.5), got "
            f"{chance!r}; the 2x-chance certificate is vacuous otherwise")
    # The resampling seed is part of the reproducibility contract: reject a
    # bool/fractional value instead of letting ``int()``/torch alias it to a
    # different bootstrap stream.
    seed = _bootstrap_seed(seed)
    # A malformed accuracy sample (non-numeric container, complex value)
    # must abstain, not escape as a TypeError from a report companion.
    # ``torch.as_tensor`` also flattens a nested/tensor input consistently.
    try:
        if torch.is_tensor(accuracies):
            raw_acc = accuracies.detach().reshape(-1)
        else:
            raw_acc = torch.as_tensor(accuracies).detach().reshape(-1)
            if raw_acc.is_floating_point():
                # Keep the caller's Python float precision instead of the
                # default float32 capture; the boundary decision is a small
                # margin against chance.
                raw_acc = torch.as_tensor(
                    accuracies, dtype=torch.float64).detach().reshape(-1)
        if raw_acc.is_complex():
            raise ValueError("accuracies must be real-valued")
        raw_acc = raw_acc.to(torch.float64)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        raw_acc = torch.empty(0, dtype=torch.float64)
    accs = [float(a) for a in raw_acc if math.isfinite(float(a))]
    margins = [a - chance for a in accs]
    # ``paired_bootstrap_ci``/``bootstrap_upper_p`` degrade to NaN readouts
    # for an invalid replicate count.  Normalise once here (a float or a
    # numeric string must become the integer shape torch wants) and skip
    # the manual per-seed bootstrap below when no replicate can be drawn,
    # instead of crashing on ``torch.randint``/``torch.binomial``.
    n_boot_valid = _positive_integer(n_boot)
    bootstrap_ok = n_boot_valid is not None
    n_boot = n_boot_valid if bootstrap_ok else 0
    ci = paired_bootstrap_ci(margins, n_boot=n_boot, seed=seed)
    ci["chance"] = chance
    if n_per_seed is not None:
        raw_n_per_seed = n_per_seed
        n_per_seed = _positive_integer(raw_n_per_seed)
        if n_per_seed is None:
            raise ValueError(
                f"n_per_seed must be a positive example count, "
                f"got {raw_n_per_seed!r}")
    n = len(accs)
    sd_seed = float(ci.get("sd", float("nan")))
    mean_seed = float(ci.get("mean", float("nan")))
    lo_seed = float(ci.get("lo", float("nan")))
    hi_seed = float(ci.get("hi", float("nan")))
    p_value_seed = float(ci.get("p_value", float("nan")))
    mde80_seed = float(ci.get("mde80", float("nan")))
    p_above_seed = bootstrap_upper_p(margins, chance, n_boot=n_boot, seed=seed)
    if n_per_seed is not None and n > 0 and bootstrap_ok:
        a = torch.tensor(accs, dtype=torch.float64)
        gen = torch.Generator(device=a.device).manual_seed(seed)
        idx = torch.randint(0, n, (n_boot, n), generator=gen, device=a.device)
        draws = a[idx]
        probs = draws.clamp(0.0, 1.0)
        mean_star = probs.mean(dim=1)
        # One shared stream draw: every seed is scored on the SAME sealed
        # evaluation examples, so per-seed binomial draws would
        # pseudo-replicate the stream and shrink the interval by
        # ~sqrt(n_seeds).
        counts = torch.binomial(
            torch.full_like(mean_star, float(n_per_seed)), mean_star,
            generator=gen)
        samples = counts / float(n_per_seed) - chance
        # The point estimate is the observed seed-level mean.  ``samples``
        # is a *resampling* distribution used only for the interval and
        # p-values; reporting its Monte-Carlo mean here would randomise the
        # effect (and the certification power plan) by the bootstrap seed.
        mean = mean_seed
        lo = float(torch.quantile(samples, 0.025))
        hi = float(torch.quantile(samples, 0.975))
        p_lo = float((samples <= 0.0).to(torch.float64).mean())
        p_hi = float((samples >= 0.0).to(torch.float64).mean())
        p_two = min(1.0, 2.0 * (min(p_lo, p_hi) * n_boot + 1.0)
                    / (n_boot + 1.0))
        p_above = float((samples >= chance).to(torch.float64).mean())
        p_above = min(1.0, (p_above * n_boot + 1.0) / (n_boot + 1.0))
        # Total per-seed SD = between-seed spread + within-seed scoring
        # noise.  The scoring noise is shared by all seeds on the common
        # stream, so it enters the SD of the mean without the 1/sqrt(n)
        # reduction.
        var_binom = (float((mean_star * (1.0 - mean_star)).mean())
                     / n_per_seed if n_per_seed > 0 else float("nan"))
        sd_binomial = math.sqrt(max(var_binom, 0.0))
        sd_effective = math.sqrt(max(sd_seed, 0.0) ** 2 + max(var_binom, 0.0))
        se_effective = (math.sqrt(max(sd_seed, 0.0) ** 2 / n
                                  + max(var_binom, 0.0)) if n else float("nan"))
        ci.update({
            "mean": mean, "lo": lo, "hi": hi,
            "sd": sd_effective,
            "se_effective": se_effective,
            "half_width": (hi - lo) / 2.0,
            "mde80": 2.8015852 * se_effective,
            "p_value": p_two,
            "p_above_chance": p_above,
            "seed_level": {
                "mean": mean_seed,
                "lo": lo_seed,
                "hi": hi_seed,
                "p_value": p_value_seed,
                "p_above_chance": p_above_seed,
                "sd": sd_seed,
                "mde80": mde80_seed,
            },
            "mean_seed": mean_seed,
            "lo_seed": lo_seed,
            "hi_seed": hi_seed,
            "p_value_seed": p_value_seed,
            "p_above_chance_seed": p_above_seed,
            "sd_seed": sd_seed,
            "sd_binomial": sd_binomial,
            "sd_effective": sd_effective,
            "n_per_seed": n_per_seed,
            "measurement_noise_accounted": True,
        })
    else:
        ci.update({
            "mean_seed": mean_seed,
            "lo_seed": lo_seed,
            "hi_seed": hi_seed,
            "p_value_seed": p_value_seed,
            "p_above_chance": p_above_seed,
            "p_above_chance_seed": p_above_seed,
            "sd_seed": sd_seed,
            "sd_binomial": float("nan"),
            "sd_effective": sd_seed,
            "se_effective": (sd_seed / math.sqrt(n) if n else float("nan")),
            "n_per_seed": n_per_seed,
            "measurement_noise_accounted": False,
        })
    certified = bool(ci["n"] > 0 and ci["hi"] < chance)
    violated = bool(ci["n"] > 0 and ci["lo"] >= chance)
    ci["hb2_supported"] = certified
    ci["hb2_violated"] = violated
    if ci["n"] <= 0 or not bootstrap_ok:
        state = "untestable"
    elif certified:
        state = "certified"
    elif violated:
        state = "violated"
    else:
        state = "inconclusive"
    ci["hb2_state"] = state
    margin = chance - float(ci["mean"])
    if bootstrap_ok and ci["n"] > 0 and math.isfinite(margin) \
            and margin > 0.0:
        if n_per_seed is not None:
            # The seed term of the mean's variance shrinks as sd_seed^2/n;
            # the common-stream scoring term enters once (all seeds share
            # the evaluation examples).  Solving the target SE
            # (margin - z*sd_binomial)/z = sqrt(sd_seed^2/n) for n inflates
            # the plain seed-only estimate by sd_seed^2 / sd_seed^2/n,
            # i.e. by the current seed count.
            z = 2.8015852
            # The interval uses
            #   se_effective^2 = sd_seed^2 / n + sd_binomial^2,
            # so invert that variance model rather than subtracting the
            # stream SD from the margin linearly.
            target_var = (margin / z) ** 2
            stream_var = float(ci.get("sd_binomial", 0.0)) ** 2
            denom = target_var - stream_var
            if denom > 0.0:
                # A zero between-seed SD is a degenerate but *favorable*
                # pilot: the seed term is already zero, so no additional
                # seed variance is required (one independent seed remains
                # the minimum for a real experiment).  Returning ``inf``
                # here would say more seeds cannot certify a run that is
                # already certified.
                ci["seeds_for_certification"] = (
                    float(sd_seed ** 2 / denom) if sd_seed > 0.0 else 1.0)
            else:
                ci["seeds_for_certification"] = float("inf")
        else:
            ci["seeds_for_certification"] = float(
                seeds_for_power(ci.get("sd"), margin))
    else:
        ci["seeds_for_certification"] = float("inf")
    return ci
