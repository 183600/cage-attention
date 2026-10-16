"""Synthetic task data.

Associative-recall style LM: sequence of key-value pairs plus a query; the
model must recall the value. This makes attention quality (and false
concentration) measurable: we know the *correct* target token. Length
generalisation (E2) evaluates this same task at multiple sequence lengths
via ``length_sweep_lengths``.

Format (vocab 0..V-1):
    [k1 v1 k2 v2 ... kn vn, SEP, qi, <predict vi>]
The label for the position right after qi is vi.
"""

from __future__ import annotations

import torch

SEP_ID = 1
PAD_ID = 0

# ---------------------------------------------------------------------------
# Train / validation / test split protocol
# ---------------------------------------------------------------------------
# A sample belongs to exactly one split, decided *before* it is drawn: every
# split owns a fixed RNG stream and batches are generated only from their own
# split's stream.  Nothing is generated into a common pool that is later
# shuffled or partitioned, and no statistic is ever fitted on validation/test
# data.  The reserved seeds below identify the held-out streams; training
# entry points refuse them via ``check_train_seed`` - training with e.g.
# ``seed=2024`` would draw exactly the batches used for model selection
# (VAL_SEED), the final reported numbers (TEST_SEED) or mechanism statistics
# (STATS_SEED), i.e. train on the evaluation data itself.
#
# The synthetic task is i.i.d.: there is no temporal order and no
# per-entity/group structure, so a seed-owned stream *is* the split and the
# split-before-shuffle requirement reduces to "assign the split before
# generating".  If fitted preprocessing is ever added, it must be fit on the
# train stream only and applied transform-only to val/test.
VAL_SEED = 123      # model/recipe selection only (never reported)
TEST_SEED = 2024    # sealed: final numbers only, never used for a decision
STATS_SEED = 7      # held-out mechanism statistics (collect_stats default)
RESERVED_EVAL_SEEDS = (VAL_SEED, TEST_SEED, STATS_SEED)


def check_train_seed(seed) -> None:
    """Refuse to train on a held-out evaluation stream.

    The data is drawn on the fly from a generator seeded by ``seed``, so
    reusing a reserved evaluation seed makes training consume exactly the
    batches that belong to a held-out split.
    """
    if int(seed) in RESERVED_EVAL_SEEDS:
        raise ValueError(
            f"seed {seed} is reserved for held-out evaluation "
            f"{RESERVED_EVAL_SEEDS}; training on it would fit the model on "
            f"the validation/test data itself. Choose a different seed."
        )


def _sample_keys(batch: int, n_pairs: int, vocab: int, gen, device) -> torch.Tensor:
    """Distinct keys per sequence (sampling without replacement).

    Values may still repeat freely (a shared value never makes the target
    ambiguous, only a shared key does).  That makes ``n_pairs`` distinct keys a
    hard requirement of the task, not a nicety: a key that occurs twice gives
    the query two different values to choose from, so the "correct answer" is
    no longer well defined and the accuracy ceiling drops for reasons that have
    nothing to do with the variant under test.  ``vocab - 2`` keys are
    available (0/1 are reserved), so a request beyond that is rejected rather
    than silently downgraded to sampling with replacement.
    """
    n_avail = vocab - 2
    if n_pairs > n_avail:
        raise ValueError(
            f"n_pairs={n_pairs} needs {n_pairs} distinct keys per sequence but "
            f"vocab={vocab} only offers {n_avail} (ids 0=PAD and 1=SEP are "
            f"reserved); use vocab >= {n_pairs + 2} or n_pairs <= {n_avail}. "
            f"Sampling keys with replacement instead would make the recall "
            f"target ambiguous and cap the measured accuracy."
        )
    noise = torch.rand(batch, n_avail, generator=gen, device=device)
    return noise.topk(n_pairs, dim=1).indices.add_(2)


def make_batch(batch: int, n_pairs: int, vocab: int, device, dense_lm: bool = False, gen: torch.Generator | None = None):
    """Returns idx, targets, eval_meta.

    eval_meta: dict with positions needed for false-concentration analysis:
      query_pos  (B,) index of qi in the sequence
      key_pos    (B,) index of the matching ki
      answer     (B,) correct value token
    """
    width = 2 * n_pairs
    seq = torch.empty((batch, width + 3), device=device, dtype=torch.long)
    kv = seq[:, :width].view(batch, n_pairs, 2)
    kv[..., 0] = _sample_keys(batch, n_pairs, vocab, gen, device)
    kv[..., 1] = torch.randint(2, vocab, (batch, n_pairs), generator=gen, device=device)
    qi_idx = torch.randint(0, n_pairs, (batch,), generator=gen, device=device)
    b = torch.arange(batch, device=device)
    query = kv[b, qi_idx, 0]
    answer = kv[b, qi_idx, 1]
    sep = seq[:, width]
    sep.fill_(SEP_ID)
    seq[:, -2].copy_(query)
    seq[:, -1].copy_(answer)
    idx = seq[:, :-1]
    tgt_full = seq[:, 1:]
    key_pos = 2 * qi_idx
    query_pos = torch.full_like(qi_idx, width + 1)
    if dense_lm:
        targets = tgt_full
    else:
        targets = torch.full_like(tgt_full, -1)
        targets[b, query_pos] = answer
    meta = {"query_pos": query_pos, "key_pos": key_pos, "answer": answer}
    return idx, targets, meta


def validate_batch(idx, targets, meta, vocab) -> dict:
    """Data-cleaning gate for one freshly generated batch.

    Raises ``ValueError`` if any invariant is violated:

    * integrity: 2-D integer ids, no NaN/Inf, all token ids in ``[0, vocab)``;
    * structure: exactly one SEP token per sequence and a valid query
      position, with ``meta`` agreeing with the sequence;
    * label noise: the query key occurs exactly once among the keys and the
      recorded answer is the value paired with that key (a repeated key
      would make the correct answer ambiguous);
    * targets (when given): ids in ``[-1, vocab)`` and the supervised entry
      at the query position equal to the answer.

    Returns a small diagnostics dict (sample count, unique-key rate, number
    of exact duplicate rows inside the batch) used by the split audit.
    """
    if not torch.is_tensor(idx) or idx.dim() != 2:
        raise ValueError(f"idx must be a (B, T) tensor, got {type(idx)}")
    if idx.is_floating_point() and not bool(torch.isfinite(idx).all()):
        raise ValueError("idx contains NaN/Inf")
    if not bool((idx >= 0).all()) or not bool((idx < vocab).all()):
        raise ValueError(f"idx contains token ids outside [0, {vocab})")
    batch_size, width = idx.shape
    qp = meta["query_pos"].to(device=idx.device, dtype=torch.long)
    if qp.shape != (batch_size,) or bool(qp.lt(0).any()) or bool(qp.ge(width).any()):
        raise ValueError("query_pos is missing or out of range")
    query = idx.gather(1, qp.unsqueeze(1)).squeeze(1)
    answer = meta["answer"].to(device=idx.device, dtype=torch.long)
    if answer.shape != (batch_size,):
        raise ValueError("meta['answer'] has the wrong shape")
    keys = idx[:, 0::2]
    match = keys.eq(query.unsqueeze(1))
    if not bool(match.sum(dim=1).eq(1).all()):
        raise ValueError(
            "label noise: the query key is not unique among the keys, so "
            "the correct answer is ambiguous"
        )
    key_pos = match.to(torch.int64).argmax(dim=1).mul(2)
    paired_value = idx.gather(1, (key_pos + 1).unsqueeze(1)).squeeze(1)
    if not bool(paired_value.eq(answer).all()):
        raise ValueError("label noise: answer != value paired with the query key")
    if not bool(idx.eq(SEP_ID).sum(dim=1).eq(1).all()):
        raise ValueError("structure: expected exactly one SEP token per row")
    if "key_pos" in meta:
        kp = meta["key_pos"].to(device=idx.device, dtype=torch.long)
        if kp.shape == (batch_size,) and not bool(kp.eq(key_pos).all()):
            raise ValueError("meta['key_pos'] disagrees with the sequence")
    if targets is not None:
        if targets.shape != idx.shape:
            raise ValueError("targets must have the same shape as idx")
        if targets.is_floating_point() and not bool(torch.isfinite(targets).all()):
            raise ValueError("targets contain NaN/Inf")
        if not bool(((targets >= -1) & (targets < vocab)).all()):
            raise ValueError(f"targets contain ids outside [-1, {vocab})")
        sup = targets.gather(1, qp.unsqueeze(1)).squeeze(1)
        if not bool(sup.eq(answer).all()):
            raise ValueError("supervised target at query_pos != answer")
    rows = idx.detach().cpu().tolist()
    n_unique = len({tuple(int(t) for t in row) for row in rows})
    return {"n": int(idx.size(0)), "unique_key_rate": 1.0,
            "n_duplicate_rows": int(len(rows) - n_unique)}


def _draw_split(n_pairs, vocab, device, batch, batches, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    rows, diagnostics = [], []
    for _ in range(batches):
        idx, targets, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        diagnostics.append(validate_batch(idx, targets, meta, vocab))
        rows.append(idx.detach().cpu())
    return torch.cat(rows, dim=0), diagnostics


def _token_distribution(idx, vocab):
    counts = torch.bincount(idx.reshape(-1).cpu(), minlength=vocab).to(torch.float64)
    return counts / counts.sum().clamp_min(1.0)


def _total_variation(p, q):
    return float(0.5 * (p - q).abs().sum())


def _fingerprints(idx):
    return {tuple(int(t) for t in row) for row in idx.tolist()}


def split_leakage_audit(n_pairs, vocab, device="cpu", batch=256, batches=4,
                        train_seed=0):
    """Audit the train/val/test streams for leakage and distribution shift.

    Draws samples from the training stream and the reserved validation,
    test and statistics streams (which are never used for fitting) and
    reports:

    * exact-duplicate overlap between the training stream and each held-out
      split - a shared sample is leakage and must be 0;
    * total-variation distance between token distributions, judged against
      ``token_tv_noise_floor`` (the distance between two disjoint halves of
      the *same* training stream), so shift is calibrated by sampling noise
      rather than an arbitrary constant;
    * cleaning diagnostics from :func:`validate_batch` (label-noise and
      structural violations raise; unique-key rate and within-batch
      duplicate counts are returned).

    The audit only reads data: it fits, selects and writes nothing.
    """
    check_train_seed(train_seed)
    train, diag = _draw_split(n_pairs, vocab, device, batch, batches, train_seed)
    held_out = {name: _draw_split(n_pairs, vocab, device, batch, batches, seed)[0]
                for name, seed in (("val", VAL_SEED), ("test", TEST_SEED),
                                   ("stats", STATS_SEED))}
    p_train = _token_distribution(train, vocab)
    half = train.size(0) // 2
    floor = float("nan")
    if half:
        floor = _total_variation(_token_distribution(train[:half], vocab),
                                 _token_distribution(train[half:], vocab))
    out = {
        "train_seed": int(train_seed),
        "val_seed": VAL_SEED,
        "test_seed": TEST_SEED,
        "stats_seed": STATS_SEED,
        "n_per_split": int(batch) * int(batches),
        "token_tv_noise_floor": floor,
        "unique_key_rate": min(d["unique_key_rate"] for d in diag),
        "n_duplicate_rows_train": sum(d["n_duplicate_rows"] for d in diag),
    }
    train_fp = _fingerprints(train)
    for name, rows in held_out.items():
        out[f"duplicate_overlap_train_{name}"] = len(train_fp & _fingerprints(rows))
        out[f"token_tv_train_{name}"] = _total_variation(
            p_train, _token_distribution(rows, vocab))
    return out


def length_sweep_lengths(base: int = 16, factor: int = 2, n: int = 4,
                         max_len: int = 256):
    """``n`` geometrically spaced ``n_pairs`` values for the length sweep.

    A length ``L`` occupies ``2*L + 2`` positions (pairs + SEP + query),
    so anything that does not fit into ``max_len`` positions cannot
    be run at all and is dropped instead of producing an out-of-range
    position embedding at evaluation time.
    """
    lengths = [base * (factor ** i) for i in range(n)]
    fits = [L for L in lengths if 2 * L + 2 <= max_len]
    if not fits:
        raise ValueError(
            f"no sweep length fits max_len={max_len}: needs >= {2 * base + 2} "
            f"positions for base={base}"
        )
    return fits
