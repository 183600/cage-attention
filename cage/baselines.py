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
  the rule ceiling.

The transformer with ``gate="none"`` is the architecture-internal
baseline that the gate variants of E1 are compared against; these
external baselines calibrate what "good" even means on this task.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import check_train_seed, make_batch, validate_batch
from .train import _make_adamw

__all__ = [
    "LinearBagProbe",
    "copy_last_token_stats",
    "linear_probe_accuracy",
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
    correct = 0
    total = 0
    extra = {}
    for _ in range(batches):
        idx, _, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        pred, stats = predict(idx, meta)
        correct += int((pred == meta["answer"]).sum())
        total += idx.size(0)
        for key, value in stats.items():
            extra[key] = extra.get(key, 0.0) + float(value)
    out = {"accuracy": correct / max(total, 1), "n": total}
    out.update({key: value / max(batches, 1) for key, value in extra.items()})
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
        x = F.one_hot(idx, self.vocab).to(self.linear.weight.dtype)
        return self.linear(x.sum(dim=1))


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
    correct = 0
    total = 0
    for _ in range(batches):
        idx, _, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        pred = model(idx).argmax(dim=-1)
        correct += int((pred == meta["answer"]).sum())
        total += idx.size(0)
    return correct / max(total, 1)


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
