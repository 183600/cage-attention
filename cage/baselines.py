"""Simple baselines for the associative-recall task.

Why these baselines
-------------------
* ``rule_pair_lookup``: the task has a closed-form solution - tokens at
  even positions are keys, keys are distinct per sequence, so the query
  key pins the value that follows it.  A parameter-free rule therefore
  defines the *ceiling*.  If it does not reach ~1.0, the data
  construction itself is broken and numbers below it cannot be attributed
  to the model.
* ``rule_naive_last_match``: the same idea without the pair-structure
  prior (the last occurrence of the query token wins).  Its gap to the
  exact rule measures the value-copy-of-key ambiguity that a naive
  string matcher would suffer.
* ``copy_last_token``: trivial floor (echo the query token).
* ``LinearBagProbe``: a one-layer linear model on bag-of-token counts,
  tuned over a learning-rate grid on a *validation* seed and reported on
  a separate *test* seed.  It cannot represent key-value lookup, so it
  should stay near chance (``1 / (vocab - 2)``).  A complex model earns
  its keep only if it clearly beats this tuned probe *and* approaches
  the rule ceiling.  "Near chance" is checked statistically
  (``probe_vs_chance_test``), not by eyeballing: a point estimate inside
  "a few x chance" with a wide CI is inconclusive, not evidence.

The transformer with ``gate="none"`` is the architecture-internal
baseline that the gate variants of E1 are compared against; these
external baselines calibrate what "good" even means on this task.  On
the learned side the ladder has a third rung that lives in
``cage.experiments`` (``_run_shallow_baseline``, hypothesis HB6): the
same transformer truncated to a single layer, with its learning rate
tuned on a validation seed and reported on the test seed.  It is the
simplest *learned* model that could plausibly solve the task, so any
depth-based motivation must survive comparison against it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import check_train_seed, make_batch, validate_batch
from .metrics import paired_bootstrap_ci, wilson_ci
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
    gen = torch.Generator(device=device).manual_seed(seed)
    correct = torch.zeros((), dtype=torch.int64, device=device)
    total = 0
    extra = {}
    for _ in range(batches):
        idx, _, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
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
    keys = idx[:, 0::2]
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
                       lr=3e-4, seed=0):
    """Fit the probe with the same optimizer/schedule recipe pinned for the
    transformer (``_make_adamw``, cosine schedule, gradient clipping).

    ``seed`` must not collide with a held-out evaluation stream; the first
    batch is validated as a data-cleaning gate (same protocol as
    ``train_model``).
    """
    check_train_seed(seed)
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    model = LinearBagProbe(vocab).to(device)
    params = list(model.parameters())
    opt = _make_adamw(params, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.1)
    model.train()
    for step in range(steps):
        idx, targets, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        if step == 0:
            validate_batch(idx, targets, meta, vocab)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(idx), meta["answer"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
    model.eval()
    return model


@torch.inference_mode()
def linear_probe_accuracy(model, n_pairs, vocab, device="cpu", batch=256,
                          batches=4, seed=123):
    gen = torch.Generator(device=device).manual_seed(seed)
    correct = torch.zeros((), dtype=torch.int64, device=device)
    total = 0
    for _ in range(batches):
        idx, _, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        pred = model(idx).argmax(dim=-1)
        correct.add_((pred == meta["answer"]).sum())
        total += idx.size(0)
    return correct.item() / max(total, 1)


def tune_linear_probe(n_pairs, vocab, device="cpu", lr_grid=(3e-3, 1e-3, 3e-4),
                      seeds=(0,), steps=400, batch=64, val_seed=123,
                      val_batches=4, probe_batch=256):
    """Pick the probe learning rate by *mean validation accuracy* over the
    probe seeds.  Model selection happens on ``val_seed`` only; the chosen
    rate is then reported on a disjoint test seed by the caller."""
    if val_seed in tuple(seeds):
        raise ValueError(
            f"validation seed {val_seed} is also a probe training seed; "
            f"selection data must be disjoint from fitting data"
        )
    table = []
    for lr in lr_grid:
        accs = []
        for seed in seeds:
            model = train_linear_probe(n_pairs, vocab, device, steps=steps,
                                       batch=batch, lr=lr, seed=seed)
            accs.append(linear_probe_accuracy(model, n_pairs, vocab, device,
                                              batch=probe_batch,
                                              batches=val_batches,
                                              seed=val_seed))
        mean = sum(accs) / len(accs)
        var = sum((a - mean) ** 2 for a in accs) / len(accs)
        table.append({"lr": lr, "val_accuracy_mean": mean,
                      "val_accuracy_std": var ** 0.5, "n_seeds": len(accs)})
    best = max(table, key=lambda row: row["val_accuracy_mean"])
    return best["lr"], table


def probe_vs_chance_test(accuracies, chance, n_boot: int = 10000,
                         seed: int = 0) -> dict:
    """Quantitative HB2 test: does the tuned probe stay near chance?

    ``accuracies`` holds the test-seed accuracy of the tuned probe for each
    training seed; ``chance`` is ``1 / (vocab - 2)``.  The returned CI is
    for the mean of ``accuracy - chance``, and ``hb2_supported`` is the
    pre-registered decision: the probe counts as "near chance" only when
    the 95% CI upper bound is *below* ``chance`` itself (i.e. the probe is
    confidently below 2x chance).  A probe whose CI reaches above 2x chance
    means either answer information leaks into bag-of-token features or
    the probe is exploiting some shortcut, and the simple-baseline
    comparison must be re-examined before crediting the transformer.

    Returns the bootstrap CI dict for ``accuracy - chance`` plus the
    ``chance`` level, the decision flag ``hb2_supported``, and the p-value
    of the one-sample bootstrap test against the chance baseline.
    """
    margins = [float(a) - float(chance) for a in accuracies]
    ci = paired_bootstrap_ci(margins, n_boot=n_boot, seed=seed)
    ci["chance"] = float(chance)
    ci["hb2_supported"] = bool(ci["n"] > 0 and
                               ci["hi"] < float(chance))
    return ci
