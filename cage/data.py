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

import hashlib
import math

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
# generating".  If the generator is ever replaced by a dataset with a time
# axis or with per-entity samples, the same rule applies at a higher level:
# split contiguous time blocks (past -> train, latest -> test, no shuffling
# before the split) and partition whole entity ids before drawing, so no
# entity ever appears in two splits.  The audit below covers the exact-
# overlap half of that contract (all stream pairs), and any fitted
# preprocessing must be fit on the training stream(s) only and applied
# transform-only to val/test; :func:`split_leakage_audit` follows that rule
# itself: every reference distribution and noise floor it uses is fitted on
# the training streams and only *scored* against held-out data.
VAL_SEED = 123      # model/recipe selection only (never reported)
TEST_SEED = 2024    # sealed: final numbers only, never used for a decision
STATS_SEED = 7      # held-out mechanism statistics (collect_stats default)
RESERVED_EVAL_SEEDS = (VAL_SEED, TEST_SEED, STATS_SEED)


# Tiny-batch overfit sanity check (``cage.train.overfit_sanity_check``): the
# single batch is generated from this stream and the model is *meant* to
# memorise it, so it is not an evaluation split and is deliberately not in
# ``RESERVED_EVAL_SEEDS`` (no reported metric ever reads it).
SANITY_SEED = 31337


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


def check_eval_seed(seed, selection: bool = False) -> None:
    """Refuse to read the sealed test stream while selecting a model.

    ``selection=True`` marks a call whose numbers inform a decision
    (learning-rate / recipe tuning, baseline selection, early stopping):
    such a call may read the validation stream but never the sealed test
    stream.  Evaluating candidates on the test stream and then keeping the
    winner turns the test set into an implicit validation set - the
    reported number is then optimistically biased for every future model
    that is picked the same way.  The test stream is only legitimate on the
    final, fixed model whose recipe was decided beforehand.
    """
    if selection and int(seed) == TEST_SEED:
        raise ValueError(
            f"stream seed {seed} is the sealed test set "
            f"(TEST_SEED): it may be read only for the final reported "
            f"evaluation, never for model selection. Use "
            f"VAL_SEED={VAL_SEED} for selection."
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


def batch_fingerprint(idx, tgt, meta) -> str:
    """Content hash of one batch (context, targets, answers).

    One hashing convention shared by ``stream_fingerprint`` and the
    tiny-batch overfit sanity check, so "which data did this run draw or
    memorise" always reduces to the same digest.
    """
    h = hashlib.sha256()
    for tensor in (idx, tgt, meta["answer"]):
        h.update(repr(tensor.detach().cpu().tolist()).encode("ascii"))
    return h.hexdigest()[:16]


def stream_fingerprint(n_pairs, vocab, device="cpu", batch=8, seed=TEST_SEED):
    """Content hash of the first batch of a data stream (dataset versioning).

    Two runs that agree on the fingerprint drew identical data; a changed
    fingerprint means the generation protocol changed and numbers from
    before are not comparable.  Hashing one batch is the smallest
    deterministic summary of a stream that catches every protocol change
    (key/value draw order, SEP/query layout, reserved-id handling), and it
    is recorded with every experiment result next to the code revision.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    idx, tgt, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
    return {"seed": int(seed), "batch": int(batch), "n_pairs": int(n_pairs),
            "vocab": int(vocab), "sha256": batch_fingerprint(idx, tgt, meta)}


def _draw_split(n_pairs, vocab, device, batch, batches, seed):
    """Draw one split's samples and run the cleaning gate on every batch.

    Returns the concatenated contexts, the supervised answers and the
    per-batch ``validate_batch`` diagnostics.  Nothing is fitted here; the
    caller decides which split plays the reference role.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    rows, answers, diagnostics = [], [], []
    for _ in range(batches):
        idx, targets, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        diagnostics.append(validate_batch(idx, targets, meta, vocab))
        rows.append(idx.detach().cpu())
        answers.append(meta["answer"].detach().cpu())
    return {"idx": torch.cat(rows, dim=0),
            "answer": torch.cat(answers, dim=0),
            "diagnostics": diagnostics}


# Audit fields: pooled tokens, the supervised answer distribution (the
# quantity the accuracy / loss metrics are computed on) and the key / value
# sub-populations.  All four are categorical token ids, so a distribution
# shift shows up directly as a total-variation distance.
_AUDIT_FIELDS = ("token", "answer", "key", "value")


def _field_values(split, n_pairs, field):
    idx, answer = split["idx"], split["answer"]
    if field == "token":
        return idx.reshape(-1)
    if field == "answer":
        return answer.reshape(-1)
    if field == "key":
        return idx[:, :2 * n_pairs:2].reshape(-1)
    if field == "value":
        return idx[:, 1:2 * n_pairs:2].reshape(-1)
    raise ValueError(f"unknown audit field {field!r}")


def _distribution(values, vocab):
    counts = torch.bincount(values.reshape(-1).to(torch.long).cpu(),
                            minlength=vocab).to(torch.float64)
    return counts / counts.sum().clamp_min(1.0)


def _total_variation(p, q):
    return float(0.5 * (p - q).abs().sum())


def _noise_floor(stream_values, vocab, reference):
    """Same-size sampling-noise scale of one audit field.

    With at least two training streams the floor is the mean distance of
    each stream to the pooled training distribution - exactly the operation
    the held-out comparison performs, at the same sample size.  With a
    single training stream the stream is split in half instead and the
    half-size distance is rescaled by ``sqrt(2)`` (the TV between two
    independent ``n/2`` draws is ``sqrt(2)`` times the TV between two
    ``n`` draws).  All inputs are fitted on training data only.
    """
    if len(stream_values) >= 2:
        return sum(_total_variation(_distribution(v, vocab), reference)
                   for v in stream_values) / len(stream_values)
    values = stream_values[0]
    if values.numel() < 2:
        return float("nan")
    half = values.numel() // 2
    floor = _total_variation(_distribution(values[:half], vocab),
                             _distribution(values[half:], vocab))
    return floor / math.sqrt(2.0)


def _fingerprints(idx):
    return {tuple(int(t) for t in row) for row in idx.tolist()}


def split_leakage_audit(n_pairs, vocab, device="cpu", batch=256, batches=4,
                        train_seeds=(0,)):
    """Audit every split of the protocol for leakage and distribution shift.

    Draws samples from every configured training stream and from the
    reserved validation, test and statistics streams (never used for
    fitting), then reports - with every reference statistic fitted on the
    training streams only:

    * **exact-duplicate overlap for every pair of streams** (training seeds
      among themselves, each training stream against each held-out split,
      and the held-out splits against each other).  A shared sample
      between *any* two splits is leakage; a validation/test overlap in
      particular silently couples model selection with the final report
      even though both splits are nominally "held out".  The full
      pairwise matrix is checked, not only train-vs-held-out, because the
      protocol's claim is pairwise disjointness.
    * **per-field distribution shift**: total-variation distance between
      each held-out split and the pooled training distribution of pooled
      tokens, supervised answers and key / value sub-populations, each
      judged against a same-size noise floor measured between disjoint
      training streams (:func:`_noise_floor`), so uniform sampling noise is
      not mistaken for a shift and the calibration is not an arbitrary
      constant.
    * **cleaning diagnostics** from :func:`validate_batch`, run on every
      batch of every split: NaN/Inf, out-of-range ids, structural
      violations and ambiguous label noise raise; the per-split
      duplicate-row count and unique-key rate are returned.

    The audit only *reads* data: it fits nothing on held-out streams and
    writes nothing.  A shared sample or a gap above the noise floor
    invalidates the HB3 decision and every number reported above it.
    """
    train_seeds = tuple(dict.fromkeys(int(s) for s in train_seeds))
    if not train_seeds:
        raise ValueError("train_seeds must contain at least one training seed")
    for seed in train_seeds:
        check_train_seed(seed)
    splits = {f"train:{seed}": _draw_split(n_pairs, vocab, device, batch,
                                           batches, seed)
              for seed in train_seeds}
    for name, seed in (("val", VAL_SEED), ("test", TEST_SEED),
                       ("stats", STATS_SEED)):
        splits[name] = _draw_split(n_pairs, vocab, device, batch, batches, seed)

    # Exact-duplicate overlap for every pair of streams.
    fingerprints = {name: _fingerprints(s["idx"]) for name, s in splits.items()}
    names = list(splits)
    pairwise = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairwise[f"{a}|{b}"] = len(fingerprints[a] & fingerprints[b])

    # Reference statistics are fitted on the training streams only.
    train_names = [f"train:{seed}" for seed in train_seeds]
    pooled = {field: torch.cat([_field_values(splits[name], n_pairs, field)
                                for name in train_names])
              for field in _AUDIT_FIELDS}
    reference = {field: _distribution(pooled[field], vocab)
                 for field in _AUDIT_FIELDS}
    floors = {
        field: _noise_floor([_field_values(splits[name], n_pairs, field)
                             for name in train_names], vocab, reference[field])
        for field in _AUDIT_FIELDS
    }

    out = {
        "train_seeds": list(train_seeds),
        "val_seed": VAL_SEED,
        "test_seed": TEST_SEED,
        "stats_seed": STATS_SEED,
        "n_per_split": int(batch) * int(batches),
        "audit_fields": list(_AUDIT_FIELDS),
        "tv_noise_floor": floors,
        "token_tv_noise_floor": floors["token"],
        "pairwise_duplicate_overlap": pairwise,
        "max_pairwise_duplicate_overlap": max(pairwise.values(), default=0),
    }
    per_split_dup = {name: sum(d["n_duplicate_rows"] for d in s["diagnostics"])
                     for name, s in splits.items()}
    out["n_duplicate_rows_train"] = sum(per_split_dup[n] for n in train_names)
    out["unique_key_rate"] = min(
        d["unique_key_rate"] for s in splits.values()
        for d in s["diagnostics"])
    for name in ("val", "test", "stats"):
        out[f"duplicate_overlap_train_{name}"] = pairwise[
            f"{train_names[0]}|{name}"]
        out[f"n_duplicate_rows_{name}"] = per_split_dup[name]
        for field in _AUDIT_FIELDS:
            out[f"{field}_tv_train_{name}"] = _total_variation(
                reference[field],
                _distribution(_field_values(splits[name], n_pairs, field),
                              vocab))
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
