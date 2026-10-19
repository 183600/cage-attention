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

import torch
import torch.nn.functional as F

from .data import check_eval_seed, check_train_seed, make_batch, validate_batch
from .metrics import bootstrap_upper_p, paired_bootstrap_ci, seeds_for_power, wilson_ci
from .repro import seed_everything, stream_generator
from .train import _make_adamw

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
    gen = torch.Generator(device=device).manual_seed(seed)
    correct = torch.zeros((), dtype=torch.int64, device=device)
    total = 0
    extra = {}
    for i in range(batches):
        idx, tgt, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        validate_batch(idx, tgt, meta, vocab)
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
    out = {"accuracy": n_correct / max(total, 1), "n": total,
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
                       lr=3e-4, seed=0, history: list | None = None):
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
    """
    check_train_seed(seed)
    seed_everything(seed)
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
        validate_batch(idx, targets, meta, vocab)
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
    gen = torch.Generator(device=device).manual_seed(seed)
    correct = torch.zeros((), dtype=torch.int64, device=device)
    total = 0
    for i in range(batches):
        idx, tgt, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        validate_batch(idx, tgt, meta, vocab)
        pred = model(idx).argmax(dim=-1)
        correct.add_((pred == meta["answer"]).sum())
        total += idx.size(0)
    return correct.item() / max(total, 1)


def tune_linear_probe(n_pairs, vocab, device="cpu", lr_grid=(3e-3, 1e-3, 3e-4),
                      seeds=(0,), steps=400, batch=64, val_seed=123,
                      val_batches=4, probe_batch=256):
    """Pick the probe learning rate by *mean validation accuracy* over the
    probe seeds.  Model selection happens on ``val_seed`` only; the chosen
    rate is then reported on a disjoint test seed by the caller.

    Robustness: a diverged grid point (non-finite loss/gradient, see
    ``train_linear_probe``) is counted in ``diverged_runs`` and its
    accuracy treated as ``-inf`` for selection - its validation number
    measures an exploded model, not the probe's representational ceiling.
    Non-finite training losses are excluded from ``train_loss_mean``.  This
    matters because the grid intentionally reaches high learning rates so
    that "the tuned probe stays near chance" cannot be attributed to an
    under-tuned strawman.
    """
    check_eval_seed(val_seed, selection=True)
    if val_seed in tuple(seeds):
        raise ValueError(
            f"validation seed {val_seed} is also a probe training seed; "
            f"selection data must be disjoint from fitting data"
        )
    table = []
    for lr in lr_grid:
        accs = []
        diverged_accs = []
        loss_means = []
        diverged = 0
        for seed in seeds:
            history = []
            model = train_linear_probe(n_pairs, vocab, device, steps=steps,
                                       batch=batch, lr=lr, seed=seed,
                                       history=history)
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
                                        batches=val_batches, seed=val_seed)
            if run_diverged:
                # The run stopped at an arbitrary step: this accuracy is
                # recorded for transparency but is not a completed-run
                # measurement and does not enter selection.
                if math.isfinite(acc):
                    diverged_accs.append(acc)
            elif math.isfinite(acc):
                accs.append(acc)
        mean = (sum(accs) / len(accs)) if accs else float("nan")
        var = (sum((a - mean) ** 2 for a in accs)
               / (len(accs) - 1) if len(accs) > 1 else 0.0)
        row = {"lr": lr, "val_accuracy_mean": mean,
               "val_accuracy_std": var ** 0.5, "n_seeds": len(accs),
               "diverged_runs": diverged}
        if diverged_accs:
            row["diverged_accuracy_mean"] = (sum(diverged_accs)
                                             / len(diverged_accs))
        if loss_means:
            row["train_loss_mean"] = sum(loss_means) / len(loss_means)
        table.append(row)
    best = max(table, key=lambda row: row["val_accuracy_mean"]
               if math.isfinite(row["val_accuracy_mean"]) else float("-inf"))
    return best["lr"], table


def probe_vs_chance_test(accuracies, chance, n_boot: int = 10000,
                         seed: int = 0) -> dict:
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
      independent seeds the observed paired SD would need at 80% power to
      certify the bound (``inf`` when the observed mean is not below the
      bound, since more seeds alone cannot rescue such a point estimate).

    Returns the bootstrap CI dict for ``accuracy - chance`` plus the
    ``chance`` level, ``hb2_supported`` (certified), ``hb2_violated``,
    ``hb2_state``, ``p_above_chance`` and ``seeds_for_certification``.
    """
    margins = [float(a) - float(chance) for a in accuracies]
    ci = paired_bootstrap_ci(margins, n_boot=n_boot, seed=seed)
    chance = float(chance)
    ci["chance"] = chance
    certified = bool(ci["n"] > 0 and ci["hi"] < chance)
    violated = bool(ci["n"] > 0 and ci["lo"] >= chance)
    ci["hb2_supported"] = certified
    ci["hb2_violated"] = violated
    if ci["n"] <= 0:
        state = "untestable"
    elif certified:
        state = "certified"
    elif violated:
        state = "violated"
    else:
        state = "inconclusive"
    ci["hb2_state"] = state
    ci["p_above_chance"] = bootstrap_upper_p(
        margins, chance, n_boot=n_boot, seed=seed)
    margin = chance - float(ci["mean"])
    ci["seeds_for_certification"] = (
        float(seeds_for_power(ci.get("sd"), margin))
        if ci["n"] > 0 and math.isfinite(margin) and margin > 0.0
        else float("inf"))
    return ci
