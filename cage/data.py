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
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32,
                   torch.int64)


def _is_integer_tensor(x) -> bool:
    """Whether ``x`` is a tensor with an integer dtype."""
    return torch.is_tensor(x) and x.dtype in _INTEGER_DTYPES


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
# entity ever appears in two splits.  The audit below covers both halves of
# that contract: exact overlap for every stream pair, and a calibrated
# same-distribution test per audit field; any fitted preprocessing must be
# fit on the training stream(s) only and applied transform-only to
# val/test; :func:`split_leakage_audit` follows that rule itself: every
# reference distribution and noise floor it uses is fitted on the training
# streams and only *scored* against held-out data.
VAL_SEED = 123      # model/recipe selection only (never reported)
TEST_SEED = 2024    # sealed: final reported numbers + fixed gate rules
STATS_SEED = 7      # held-out mechanism statistics (collect_stats default)
RESERVED_EVAL_SEEDS = (VAL_SEED, TEST_SEED, STATS_SEED)


def stream_seed_key(seed) -> int:
    """The low 32 bits consumed by torch's CPU stream generators.

    ``torch.Generator.manual_seed`` keeps only these bits on CPU, so a seed
    such as ``TEST_SEED + 2**32`` draws the exact sealed stream despite
    comparing unequal to ``TEST_SEED`` as a Python integer.
    """
    return int(seed) & 0xFFFFFFFF


_RESERVED_EVAL_SEED_KEYS = frozenset(
    stream_seed_key(seed) for seed in RESERVED_EVAL_SEEDS)
_SEALED_EVAL_SEED_KEYS = frozenset(
    stream_seed_key(seed) for seed in (TEST_SEED, STATS_SEED))

# Sealing semantics, stated precisely because the short label "sealed" has
# been read as "never touched again".  Sealed means **no model,
# hyperparameter or recipe choice may read the stream**: all selection
# (learning-rate tuning, recipe freezing, baseline choice, early stopping)
# stays on VAL_SEED.  It does *not* mean the stream is evaluated once and
# only once at the very end.  The pre-registered rules fixed in the code -
# HB1's rule ceiling, HB4's label-permuted negative control and the
# HB0/HB6/HB7 competence / saturation marks - are read on the TEST stream
# and gate the grids; those rules are not tunable, and the autonomous CI
# loop re-runs ``--only baselines`` after a code change, so their verdicts
# are *re-measured* (the same fixed rule, applied again), not re-selected.
# Because no decision in that loop can see or react to the individual
# numbers (the rules and the grid are pre-registered and the gate is
# pass/fail), re-measurement cannot turn TEST into an implicit validation
# split; the bias bound is stated in the baselines verdict.


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
    if stream_seed_key(seed) in _RESERVED_EVAL_SEED_KEYS:
        raise ValueError(
            f"seed {seed} is reserved for held-out evaluation "
            f"{RESERVED_EVAL_SEEDS}; training on it would fit the model on "
            f"the validation/test data itself. Choose a different seed."
        )


def check_eval_seed(seed, selection: bool = False) -> None:
    """Refuse to read a sealed evaluation stream while selecting a model.

    ``selection=True`` marks a call whose numbers inform a decision
    (learning-rate / recipe tuning, baseline selection, early stopping):
    such a call may read the validation stream but never the sealed test
    *or statistics* stream.  Evaluating candidates on a sealed stream and
    then keeping the winner turns it into an implicit validation set - the
    reported number is then optimistically biased for every future model
    that is picked the same way.  TEST_SEED is legitimate for the final
    reported numbers and for the fixed, pre-registered gate rules (never
    for a choice); STATS_SEED only for the mechanism-statistics readout.
    """
    if selection and stream_seed_key(seed) in _SEALED_EVAL_SEED_KEYS:
        raise ValueError(
            f"stream seed {seed} is sealed (TEST_SEED={TEST_SEED} or "
            f"STATS_SEED={STATS_SEED}): it may be read only for the final "
            f"reported evaluation / pre-registered gate rules / mechanism "
            f"statistics, never for model selection. Use VAL_SEED="
            f"{VAL_SEED} for selection."
        )


# ---------------------------------------------------------------------------
# Time-order and group split primitives
# ---------------------------------------------------------------------------
# The synthetic task is i.i.d., so its split is the seed-owned stream above.
# Any future dataset with a time axis or with repeated entities must instead
# use the primitives below *before* drawing/preprocessing: contiguous time
# blocks (past -> train, newest -> test, never shuffled before the split) and
# whole groups on exactly one side.  Both are pure index/assignment helpers
# with self-tests, so the contract is executable instead of a comment.


def time_ordered_split(n_samples: int, train_frac: float = 0.7,
                       val_frac: float = 0.15) -> dict:
    """Contiguous chronological split of time-ordered samples.

    Samples are assumed to be sorted by time (row ``i`` is older than row
    ``j`` for ``i < j``).  The returned slices are contiguous blocks in that
    order: ``train`` is the oldest prefix, ``val`` the middle block and
    ``test`` the newest suffix, so no future sample can teach the model and
    no sample is shuffled across the boundary.  Every block is non-empty.

    Batch-level shuffling *inside* the training block is a separate,
    allowed operation performed by the training loop; this function only
    decides the split.
    """
    n = int(n_samples)
    if n < 3:
        raise ValueError(
            f"time_ordered_split needs at least 3 samples for three "
            f"non-empty blocks, got {n_samples!r}")
    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac!r}")
    if not (0.0 < float(val_frac) < 1.0):
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac!r}")
    if float(train_frac) + float(val_frac) >= 1.0:
        raise ValueError(
            f"train_frac + val_frac must leave a test block, got "
            f"{train_frac!r} + {val_frac!r}")
    n_val = max(1, int(n * float(val_frac)))
    n_test = max(1, n - int(n * (float(train_frac) + float(val_frac))))
    n_train = n - n_val - n_test
    if n_train < 1:
        # Extremely skewed fractions at a tiny n: keep one sample per block.
        n_train = 1
        n_val = 1
        n_test = n - 2
    return {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n),
    }


def group_disjoint_split(groups, train_frac: float = 0.7,
                         val_frac: float = 0.15, seed: int = 0) -> dict:
    """Assign **whole groups** to one split each; returns index tensors.

    ``groups`` holds one group id per sample (e.g. a user id or an object
    id).  All samples of a group land on exactly one side, so no entity is
    seen in both training and evaluation - the requirement that a random
    per-sample split violates.  Groups are shuffled with a dedicated
    ``seed`` (the assignment is random over entities, never over samples)
    and allocated by their sample mass so the requested fractions are met
    approximately; every split is guaranteed non-empty when there are at
    least three groups.

    Use :func:`assert_group_disjoint` to verify the result, and
    :func:`time_ordered_split` instead when the rows also have a time order.
    """
    if not (0.0 < float(train_frac) < 1.0) or not (
            0.0 < float(val_frac) < 1.0) or (
            float(train_frac) + float(val_frac) >= 1.0):
        raise ValueError(
            f"need 0 < train_frac, 0 < val_frac and train_frac + val_frac "
            f"< 1, got {train_frac!r}, {val_frac!r}")
    g = torch.as_tensor(groups).reshape(-1)
    n = int(g.numel())
    if n == 0:
        raise ValueError("groups must contain at least one sample")
    if g.is_floating_point() and not bool(torch.isfinite(g).all()):
        raise ValueError("group ids must be finite (no NaN/Inf)")
    uniq, inverse = torch.unique(g, return_inverse=True)
    n_groups = int(uniq.numel())
    if n_groups < 3:
        raise ValueError(
            f"group_disjoint_split needs at least 3 groups for three "
            f"non-empty splits, got {n_groups}")
    # First-appearance order keeps the allocation deterministic across runs
    # and independent of tensor hashing / dtype.
    first = torch.full((n_groups,), n, dtype=torch.long)
    first.scatter_reduce_(0, inverse,
                          torch.arange(n, dtype=torch.long), reduce="amin")
    order = torch.argsort(first)
    perm = order[torch.randperm(
        n_groups, generator=torch.Generator().manual_seed(int(seed)))]
    counts = torch.bincount(inverse, minlength=n_groups).to(torch.float64)
    cuts = (float(train_frac) * n, float(train_frac + val_frac) * n)
    assigned = {"train": [], "val": [], "test": []}
    cumulative = 0.0
    for gid in perm.tolist():
        mass = float(counts[gid])
        mid = cumulative + mass / 2.0
        if mid < cuts[0]:
            name = "train"
        elif mid < cuts[1]:
            name = "val"
        else:
            name = "test"
        assigned[name].append(gid)
        cumulative += mass
    for name in ("val", "test"):
        if not assigned[name]:
            donor = max(assigned, key=lambda k: len(assigned[k]))
            if len(assigned[donor]) <= 1:
                raise ValueError(
                    f"cannot give every split a group at n_groups={n_groups}")
            assigned[name].append(assigned[donor].pop())
    out = {}
    for name, gids in assigned.items():
        mask = torch.isin(inverse, torch.tensor(gids, dtype=torch.long))
        out[name] = torch.nonzero(mask, as_tuple=False).flatten()
    return out


def _as_indices(index, n: int) -> torch.Tensor:
    """Normalize a split index container to a ``LongTensor`` of positions."""
    if isinstance(index, slice):
        return torch.arange(*index.indices(n), dtype=torch.long)
    if isinstance(index, range):
        return torch.tensor(list(index), dtype=torch.long)
    return torch.as_tensor(index).reshape(-1).to(torch.long)


def assert_group_disjoint(groups, splits) -> None:
    """Raise if any group id occurs in more than one split."""
    g = torch.as_tensor(groups).reshape(-1)
    if g.is_floating_point() and not bool(torch.isfinite(g).all()):
        raise ValueError("group ids must be finite (no NaN/Inf)")
    seen: dict = {}
    for name in ("train", "val", "test"):
        if name not in splits:
            continue
        idx = _as_indices(splits[name], int(g.numel()))
        if idx.numel() == 0:
            continue
        for gid in torch.unique(g[idx]).tolist():
            if gid in seen and seen[gid] != name:
                raise ValueError(
                    f"group {gid!r} crosses splits {seen[gid]!r} and "
                    f"{name!r}: whole groups must stay on one side")
            seen[gid] = name


def assert_time_ordered(times, splits) -> None:
    """Raise if the split blocks are not chronological past -> future.

    The check is on the split *assignment*, not on the row order inside a
    block: training may shuffle its own batches, but no sample in ``train``
    may be newer than a validation sample, and no validation sample may be
    newer than a test sample.
    """
    t = torch.as_tensor(times).reshape(-1).to(torch.float64)
    if not bool(torch.isfinite(t).all()):
        raise ValueError("time values must be finite (no NaN/Inf): a "
                         "non-finite timestamp makes the ordering undefined")
    spans = []
    for name in ("train", "val", "test"):
        if name not in splits:
            continue
        idx = _as_indices(splits[name], int(t.numel()))
        if idx.numel() == 0:
            continue
        spans.append((float(t[idx].min()), float(t[idx].max()), name))
    for (lo_a, hi_a, a), (lo_b, hi_b, b) in zip(spans, spans[1:]):
        if hi_a > lo_b:
            raise ValueError(
                f"time leakage: {a} block ends at {hi_a} but {b} block "
                f"starts at {lo_b} (older -> newer ordering violated)")
        if lo_b < lo_a:
            raise ValueError(
                f"time leakage: {b} contains samples older than {a}")


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
        targets = tgt_full.clone()
    else:
        targets = torch.full_like(tgt_full, -1)
        targets[b, query_pos] = answer
    meta = {"query_pos": query_pos, "key_pos": key_pos, "answer": answer}
    return idx, targets, meta


def _count_duplicate_rows(idx) -> int:
    """Exact duplicate rows in ``idx`` (content comparison, no fitting).

    Used both for the per-batch cleaning diagnostics and for the
    whole-stream duplicate count in :func:`split_leakage_audit`: counting
    duplicates is a deterministic set operation, so it is safe on any
    split and needs no train-fitted statistics.  ``torch.unique(dim=0)``
    performs the same set operation on every training/evaluation batch.
    """
    if idx.numel() == 0:
        return 0
    unique_rows = torch.unique(idx.detach(), dim=0, sorted=False).size(0)
    return int(idx.size(0)) - int(unique_rows)


def validate_batch(idx, targets, meta, vocab, diagnostics: bool = True) -> dict:
    """Data-cleaning gate for one freshly generated batch.

    Raises ``ValueError`` if any invariant is violated:

    * integrity: 2-D integer ids, no NaN/Inf, all token ids in ``[0, vocab)``;
    * structure: exactly one SEP token, at the fixed separator slot, a
      query at the final position, and ``meta`` agreeing with the
      sequence;
    * label noise: the query key occurs exactly once among the keys, the
      recorded answer is the value paired with that key (a repeated key
      would make the correct answer ambiguous) and the answer id is a
      selectable token (ids 0=PAD and 1=SEP are reserved and never labels);
    * targets (when given): ids in ``[-1, vocab)``, the supervised entry
      at the query position equal to the answer, and *every* entry aligned
      with its position (``-1`` or the true next-token label).  Checking
      more than the single supervised entry catches off-by-one /
      foreign-label target misalignment: a label silently shifted onto the
      wrong position passes the one-spot check whenever the shift happens
      not to touch the query position, but fails the alignment check.

    Returns a small diagnostics dict (sample count, unique-key rate, number
    of exact duplicate rows inside the batch) used by the split audit.
    ``diagnostics=False`` skips only the duplicate-row count for callers
    that discard the result; every cleaning invariant above still runs.
    """
    if not _batch_invariants_hold(idx, targets, meta, vocab):
        _validate_batch_strict(idx, targets, meta, vocab)
    out = {"n": int(idx.size(0)), "unique_key_rate": 1.0}
    if diagnostics:
        out["n_duplicate_rows"] = _count_duplicate_rows(idx)
    return out


def _batch_invariants_hold(idx, targets, meta, vocab) -> bool:
    """``validate_batch``'s invariants as one device-side predicate.

    The same conditions as the strict checker below, accumulated on the
    device; anything that cannot be evaluated (non-tensor inputs,
    unexpected meta entries, wrong shapes, ...) reports ``False`` and is
    left to the strict checker, which raises.  The predicate never
    *accepts* a batch the strict checker would reject: every condition of
    the strict checker is mirrored here, shape-level conditions as host
    checks and value-level ones as device checks.
    """
    try:
        if not _is_integer_tensor(idx) or idx.dim() != 2:
            return False
        batch_size, width = idx.shape
        checks = [((idx >= 0) & (idx < vocab)).all()]
        if not _is_integer_tensor(meta["query_pos"]):
            return False
        qp = meta["query_pos"].to(device=idx.device, dtype=torch.long)
        if qp.shape != (batch_size,):
            return False
        checks.append(((qp >= 0) & (qp < width)).all())
        checks.append(qp.eq(width - 1).all())
        # Clamped only to keep the gathers below in range: the clamped
        # positions are simultaneously rejected by the range checks above,
        # so an invalid batch can never slip through on the clamped index.
        safe_qp = qp.unsqueeze(1).clamp(0, max(width - 1, 0))
        query = idx.gather(1, safe_qp).squeeze(1)
        if not _is_integer_tensor(meta["answer"]):
            return False
        answer = meta["answer"].to(device=idx.device, dtype=torch.long)
        answer_ok = answer.shape == (batch_size,)
        checks.append(((answer >= 2) & (answer < vocab)).all())
        # Key columns only: ``idx`` is ``[k1 v1 ... kn vn SEP query]``, so
        # the last two columns (SEP at 2n, query at 2n+1) must be excluded.
        keys = idx[:, :-2:2]
        match = keys.eq(query.unsqueeze(1))
        checks.append(match.sum(dim=1).eq(1).all())
        key_pos = match.to(torch.int64).argmax(dim=1).mul(2)
        paired_value = idx.gather(1, (key_pos + 1).unsqueeze(1)).squeeze(1)
        checks.append(paired_value.eq(answer).all())
        checks.append(idx.eq(SEP_ID).sum(dim=1).eq(1).all())
        checks.append(idx[:, -2].eq(SEP_ID).all())
        if "key_pos" in meta:
            if not _is_integer_tensor(meta["key_pos"]):
                return False
            kp = meta["key_pos"].to(device=idx.device, dtype=torch.long)
            if kp.shape != (batch_size,):
                return False
            checks.append(kp.eq(key_pos).all())
        if targets is not None:
            if not _is_integer_tensor(targets) or targets.shape != idx.shape:
                return False
            checks.append(((targets >= -1) & (targets < vocab)).all())
            sup = targets.gather(1, safe_qp).squeeze(1)
            checks.append(sup.eq(answer).all())
            # Full label alignment, not only the supervised entry: every
            # position must be unsupervised (``-1``) or carry the true
            # next-token label of that position.
            nxt = targets[:, :-1]
            cont = idx[:, 1:]
            checks.append((nxt.eq(-1) | nxt.eq(cont)).all())
            checks.append((targets[:, -1].eq(-1)
                           | targets[:, -1].eq(answer)).all())
        # One reduction over the stacked predicates instead of one host
        # decision per check (same conjunction, fewer sync points).
        return answer_ok and bool(torch.stack(checks).all())
    except Exception:  # noqa: BLE001 - any unevaluable input is strict's job
        return False


def _validate_batch_strict(idx, targets, meta, vocab) -> dict:
    """``validate_batch``'s checks, one host read-back per condition."""
    if not _is_integer_tensor(idx) or idx.dim() != 2:
        raise ValueError(
            f"idx must be a (B, T) integer tensor, got {type(idx)}")
    if not bool((idx >= 0).all()) or not bool((idx < vocab).all()):
        raise ValueError(f"idx contains token ids outside [0, {vocab})")
    batch_size, width = idx.shape
    if not _is_integer_tensor(meta["query_pos"]):
        raise ValueError("meta['query_pos'] must have an integer dtype")
    qp = meta["query_pos"].to(device=idx.device, dtype=torch.long)
    if qp.shape != (batch_size,) or bool(qp.lt(0).any()) or bool(qp.ge(width).any()):
        raise ValueError("query_pos is missing or out of range")
    if not bool(qp.eq(width - 1).all()):
        raise ValueError(
            "structure: query_pos must be the final sequence position")
    query = idx.gather(1, qp.unsqueeze(1)).squeeze(1)
    if not _is_integer_tensor(meta["answer"]):
        raise ValueError("meta['answer'] must have an integer dtype")
    answer = meta["answer"].to(device=idx.device, dtype=torch.long)
    if answer.shape != (batch_size,):
        raise ValueError("meta['answer'] has the wrong shape")
    if not bool((answer >= 2).all()) or not bool((answer < vocab).all()):
        raise ValueError(
            "label noise: answer ids must be selectable tokens in "
            f"[2, {vocab}) (0=PAD and 1=SEP are reserved and never labels)"
        )
    # Key columns only: ``idx`` is ``[k1 v1 ... kn vn SEP query]``, so the
    # last two columns (SEP at 2n, query at 2n+1) must be excluded - a
    # plain ``0::2`` stride would pick up SEP and silently rely on
    # ``query >= 2`` never matching the reserved SEP id.
    keys = idx[:, :-2:2]
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
    if not bool(idx[:, -2].eq(SEP_ID).all()):
        raise ValueError(
            "structure: SEP must occupy the fixed separator slot (T-2)")
    if "key_pos" in meta:
        if not _is_integer_tensor(meta["key_pos"]):
            raise ValueError("meta['key_pos'] must have an integer dtype")
        kp = meta["key_pos"].to(device=idx.device, dtype=torch.long)
        if kp.shape != (batch_size,):
            raise ValueError("meta['key_pos'] has the wrong shape")
        if not bool(kp.eq(key_pos).all()):
            raise ValueError("meta['key_pos'] disagrees with the sequence")
    if targets is not None:
        if not _is_integer_tensor(targets) or targets.shape != idx.shape:
            raise ValueError(
                "targets must have the same shape as idx and an integer dtype")
        if not bool(((targets >= -1) & (targets < vocab)).all()):
            raise ValueError(f"targets contain ids outside [-1, {vocab})")
        sup = targets.gather(1, qp.unsqueeze(1)).squeeze(1)
        if not bool(sup.eq(answer).all()):
            raise ValueError("supervised target at query_pos != answer")
        nxt = targets[:, :-1]
        cont = idx[:, 1:]
        if not bool((nxt.eq(-1) | nxt.eq(cont)).all()):
            raise ValueError(
                "label alignment: a target entry is neither -1 nor the "
                "true next-token continuation of its position"
            )
        if not bool((targets[:, -1].eq(-1)
                     | targets[:, -1].eq(answer)).all()):
            raise ValueError(
                "label alignment: the final target entry is neither -1 nor "
                "the sample's answer"
            )
    return {"n": int(idx.size(0)), "unique_key_rate": 1.0,
            "n_duplicate_rows": _count_duplicate_rows(idx)}


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

    The batch geometry is part of the digest, not only of the metadata:
    ``make_batch`` consumes the stream through shape-dependent draws
    (``torch.rand(batch, n_avail)`` / ``torch.randint(..., (batch,
    n_pairs))``), so the same seed at ``batch=8`` and ``batch=64`` produces
    a different first batch.  Hashing only the tensors would therefore
    report the same fingerprint for two different realizations; including
    ``(n_pairs, vocab, batch, seed)`` makes "same fingerprint" mean "same
    protocol *and* same geometry", which is the claim run records rely on.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    idx, tgt, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
    geometry = (f"n_pairs={int(n_pairs)};vocab={int(vocab)};"
                f"batch={int(batch)};seed={int(seed)};")
    digest = hashlib.sha256(
        (geometry + batch_fingerprint(idx, tgt, meta)).encode("ascii")
    ).hexdigest()[:16]
    return {"seed": int(seed), "batch": int(batch), "n_pairs": int(n_pairs),
            "vocab": int(vocab), "sha256": digest}


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


def _row_keys(idx):
    """Per-row content fingerprints (two independent 64-bit tuple hashes).

    ``hash(tuple(ints))`` is deterministic across processes because integer
    hashing is not salted, so the same row gets the same key in every audit
    run.  The second, position-weighted hash makes an accidental collision
    of the first effectively impossible for the <1e6 rows an audit sees;
    a collision would be reported as overlap (fail closed), never as
    disjointness.
    """
    out = []
    for row in idx.tolist():
        ints = [int(t) for t in row]
        forward = tuple(ints)
        weighted = tuple((i + 1) * t for i, t in enumerate(ints))
        out.append((hash(forward), hash(weighted)))
    return out


def _draw_stream_keys(seed, n_pairs, vocab, device, batch, units):
    """Replay one stream exactly as its consumer reads it and hash the rows.

    ``units`` is the number of batches the consumer draws (training steps
    or evaluation batches) at its own batch size, in generator order.  Every
    batch passes the shared cleaning gate, so a stream that would be
    rejected during training/evaluation aborts the audit instead of being
    hashed.
    """
    gen = torch.Generator(device=device).manual_seed(int(seed))
    keys, n_rows = set(), 0
    for _ in range(int(units)):
        idx, tgt, meta = make_batch(int(batch), n_pairs, vocab, device,
                                    gen=gen)
        validate_batch(idx, tgt, meta, vocab, diagnostics=False)
        batch_keys = _row_keys(idx)
        keys.update(batch_keys)
        n_rows += len(batch_keys)
    return keys, n_rows


def actual_stream_overlap_audit(n_pairs, vocab, device="cpu",
                                train_specs=None, heldout_specs=None):
    """Exact content overlap of the streams a run *actually* consumes.

    ``split_leakage_audit`` checks the first ``batch * batches`` samples of
    every seed at one fixed geometry.  Training, however, reads
    ``settings.steps * settings.batch`` samples, and ``make_batch`` draws
    through shape-dependent random calls, so the audited prefix is not the
    training realization.  This audit replays each specified stream with
    exactly the geometry its consumer uses:

    * ``train_specs`` maps a stream name to ``{"seed", "batch", "steps"}``;
    * ``heldout_specs`` maps a stream name to ``{"seed", "batch",
      "batches"}`` (e.g. final eval, validation monitoring, stats).

    A spec may carry ``"group"`` to name the split it belongs to (training
    specs default to ``"train"``, held-out specs to the spec name).  Two
    realizations of the *same* split (e.g. the validation stream at the
    monitoring geometry and at the final-eval geometry) may legitimately
    share samples; only overlaps between *different* groups - train vs
    held-out, validation vs test, test vs statistics - are leakage.  The
    full pairwise matrix is still reported, but ``max_overlap`` counts
    cross-group pairs only, so re-reading one split at two geometries
    cannot fail the gate.

    It returns per-stream row/duplicate counts and the pairwise overlap
    matrix over row-content fingerprints.  Any cross-group overlap is
    leakage; the caller decides how to gate on it.
    """
    train_specs = dict(train_specs or {})
    heldout_specs = dict(heldout_specs or {})
    if not train_specs or not heldout_specs:
        raise ValueError(
            "actual_stream_overlap_audit needs at least one training and "
            "one held-out stream spec")
    key_sets, streams, groups = {}, {}, {}
    all_specs = ([(name, spec, "train")
                  for name, spec in train_specs.items()]
                 + [(name, spec, "heldout")
                    for name, spec in heldout_specs.items()])
    for name, spec, kind in all_specs:
        unit_key = "steps" if kind == "train" else "batches"
        if spec.get(unit_key) is None:
            raise ValueError(
                f"{kind} spec {name!r} needs {unit_key!r}")
        keys, n_rows = _draw_stream_keys(
            spec["seed"], n_pairs, vocab, device, spec["batch"],
            spec[unit_key])
        key_sets[name] = keys
        groups[name] = str(spec.get(
            "group", "train" if kind == "train" else name))
        streams[name] = {
            "kind": kind, "group": groups[name],
            "seed": int(spec["seed"]), "batch": int(spec["batch"]),
            "units": int(spec[unit_key]),
            "n_rows": int(n_rows), "n_unique_rows": int(len(keys)),
            "n_duplicate_rows": int(n_rows - len(keys)),
        }

    names = list(key_sets)
    pairwise = {}
    cross_group = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = len(key_sets[a] & key_sets[b])
            pairwise[f"{a}|{b}"] = overlap
            if groups[a] != groups[b]:
                cross_group.append(overlap)
    train_names = list(train_specs)
    heldout_names = list(heldout_specs)
    train_vs_heldout = [
        pairwise[f"{a}|{b}"] for a in train_names for b in heldout_names]
    heldout_vs_heldout = [
        pairwise[f"{a}|{b}"] for i, a in enumerate(heldout_names)
        for b in heldout_names[i + 1:]]
    return {
        "train_specs": {k: dict(v) for k, v in train_specs.items()},
        "heldout_specs": {k: dict(v) for k, v in heldout_specs.items()},
        "streams": streams,
        "pairwise_overlap": pairwise,
        "train_vs_heldout_max": max(train_vs_heldout, default=0),
        "heldout_vs_heldout_max": max(heldout_vs_heldout, default=0),
        "max_overlap": max(cross_group, default=0),
        "n_rows_total": int(sum(s["n_rows"] for s in streams.values())),
    }


# Audit fields: pooled tokens, the supervised answer distribution (the
# quantity the accuracy / loss metrics are computed on), the key / value
# sub-populations, and the *structural difficulty* covariates that decide
# whether accuracy is comparable across splits: where the matching key sits
# (retrieval distance), whether the answer value repeats in the context, and
# whether a naive last-match rule would be misled by the query token
# appearing later as a value.  Marginal token frequencies can match perfectly
# while one split contains systematically later keys or more ambiguous
# samples; such a shift changes the ceiling/accuracy of a split and would
# otherwise be invisible to the audit.  All fields are categorical, so a
# distribution shift shows up directly as a total-variation distance and a
# calibrated chi-square.
_AUDIT_FIELDS = ("token", "answer", "key", "value", "key_pos",
                 "value_repeats", "naive_confusable")
_AUDIT_DERIVED_FIELDS = ("key_pos", "value_repeats", "naive_confusable")

# Calibration of the distribution-shift check.  ``_AUDIT_SIM`` Monte Carlo
# replicates set the p-value resolution to ``1 / (_AUDIT_SIM + 1)`` (500
# replicates resolve the 0.0024 rank-one threshold of a 21-test
# Benjamini-Hochberg family at q=0.05); the fixed seed keeps the audit
# deterministic; ``_AUDIT_SMOOTH`` is the Haldane-Anscombe add-half count
# that keeps unobserved classes testable instead of producing an infinite
# statistic.
_AUDIT_SIM = 500
_AUDIT_SIM_SEED = 90210
_AUDIT_SMOOTH = 0.5


def _difficulty_features(split, n_pairs):
    """Per-sample structural difficulty covariates of one drawn split.

    Derived from the validated ``[k1 v1 ... kn vn SEP query]`` layout (the
    query is the last column of ``idx``; the answer is stored separately),
    so no fitted statistic is involved:

    * ``key_pos`` - pair index of the matching key (retrieval distance);
    * ``value_repeats`` - the answer value occurs more than once among the
      values, so a perfect key lookup does not uniquely determine it;
    * ``naive_confusable`` - the query token occurs after the matching key
      as a value, so the naive "last visible match wins" rule differs from
      the exact pair lookup.

    Mirrors ``cage.train._error_case_flags`` so the audit measures exactly
    the difficulty factors used to explain errors downstream.
    """
    idx = split["idx"]
    answer = split["answer"]
    query = idx[:, -1]
    keys = idx[:, :2 * n_pairs:2]
    match = keys.eq(query.unsqueeze(1))
    key_pos = match.to(torch.int64).argmax(dim=1)
    values = idx[:, 1:2 * n_pairs:2]
    value_repeats = (values.eq(answer.unsqueeze(1)).sum(dim=1)
                     > 1).to(torch.int64)
    positions = torch.arange(idx.size(1)).view(1, -1)
    visible = positions < (idx.size(1) - 1)
    last_match = torch.where(idx.eq(query.unsqueeze(1)) & visible,
                             positions,
                             torch.full_like(positions, -1)).max(dim=1).values
    naive_confusable = last_match.ne(key_pos.mul(2)).to(torch.int64)
    return {"key_pos": key_pos, "value_repeats": value_repeats,
            "naive_confusable": naive_confusable}


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
    if field in _AUDIT_DERIVED_FIELDS:
        return _difficulty_features(split, n_pairs)[field]
    raise ValueError(f"unknown audit field {field!r}")


def _field_categories(field, vocab, n_pairs):
    """Number of possible categories for a categorical audit field."""
    if field == "key_pos":
        return max(int(n_pairs), 1)
    if field in ("value_repeats", "naive_confusable"):
        return 2
    return int(vocab)


def _counts(values, n_categories):
    return torch.bincount(values.reshape(-1).to(torch.long).cpu(),
                          minlength=n_categories).to(torch.int64)


def _distribution_from_counts(counts):
    counts = counts.to(torch.float64)
    return counts / counts.sum().clamp_min(1.0)


def _distribution(values, n_categories):
    return _distribution_from_counts(_counts(values, n_categories))


def _total_variation(p, q):
    return float(0.5 * (p - q).abs().sum())


def _noise_floor(stream_values, n_categories):
    """Same-size sampling-noise scale of one audit field.

    With at least two training streams the floor is the mean distance of
    each stream to the pooled distribution of the *other* streams
    (leave-one-out) - exactly the operation the held-out comparison
    performs, at the same sample size.  Including the stream in its own
    reference would understate the floor by ``(m-1)/m``, which would make
    the tolerance tighter than the advertised margin.  With a single
    training stream the stream is split in half instead and the half-size
    distance is rescaled by ``sqrt(2)`` (the TV between two independent
    ``n/2`` draws is ``sqrt(2)`` times the TV between two ``n`` draws).
    All inputs are fitted on training data only.
    """
    if len(stream_values) >= 2:
        counts = [_counts(values, n_categories) for values in stream_values]
        dists = [_distribution_from_counts(count) for count in counts]
        total = counts[0].clone()
        for count in counts[1:]:
            total.add_(count)
        floors = []
        for j in range(len(counts)):
            # Leave-one-out reference from the cached counts instead of
            # re-binning the concatenation of all other streams.
            floors.append(
                _total_variation(dists[j],
                                 _distribution_from_counts(total - counts[j])))
        return sum(floors) / len(floors)
    values = stream_values[0]
    if values.numel() < 2:
        return float("nan")
    half = values.numel() // 2
    lo = _counts(values[:half], n_categories)
    hi = _counts(values[half:], n_categories)
    floor = _total_variation(_distribution_from_counts(lo),
                             _distribution_from_counts(hi))
    return floor / math.sqrt(2.0)


def _distribution_shift_p(observed, reference, n_sim=_AUDIT_SIM,
                          seed=_AUDIT_SIM_SEED):
    """Monte Carlo p-value for "held-out and training share a distribution".

    ``observed`` / ``reference`` are categorical count vectors at the
    held-out and pooled-training sample sizes, fitted on training data
    only.  The statistic is the Pearson chi-square of the observed counts
    against their expectation under the reference distribution,

    ``X^2 = sum_i (o_i - n_h * p_i)^2 / (n_h * p_i),  p_i = (r_i + c) /
    (N_r + c * K)``

    with the Haldane-Anscombe correction ``c = _AUDIT_SMOOTH`` so classes
    unseen in the reference stay testable (a class with ``r_i = 0`` that
    nevertheless appears held-out then contributes a large finite term
    instead of an infinite one).  The null distribution of ``X^2`` is not
    read off a chi-square table - the reference itself is a finite sample -
    but re-derived by a parametric bootstrap that redraws **both** count
    vectors from the smoothed reference probabilities, reference at
    training size and held-out at held-out size, and recomputes the
    statistic with the simulated reference (Poisson draws are a standard
    stand-in for multinomial counts whose conditioning correction is
    ``O(1/N)`` and negligible against the statistic's sampling variation
    at these sizes).  The returned p-value is
    ``(1 + #{X^2_sim >= X^2_obs}) / (n_sim + 1)``, deterministic in the
    audit seed.

    This replaces a "total variation <= 3 x noise floor" tolerance because
    that point rule goes blind exactly where the audit matters most: the
    supervised answer spreads ~1e3 samples over ``vocab - 2`` classes, so
    even two *same-source* draws already have TV ~ 0.3, three times the
    floor approaches the maximum TV of 1.0, and a held-out split drawn
    from half the class range (TV ~ 0.5) passes unnoticed.  The chi-square
    statistic senses mass moving between classes at the same sample sizes
    and the bootstrap quantiles it against the real null (verified in
    ``python3 -m cage.data``: same-source draws are not flagged, a
    half-range split is).  Returns ``(nan, nan)`` for empty inputs.
    """
    obs = observed.detach().cpu().to(torch.float64)
    ref = reference.detach().cpu().to(torch.float64)
    n_ref = float(ref.sum())
    n_h = float(obs.sum())
    if n_ref <= 0.0 or n_h <= 0.0:
        return float("nan"), float("nan")
    k = ref.numel()
    probs = (ref + _AUDIT_SMOOTH) / (n_ref + _AUDIT_SMOOTH * k)
    expected = n_h * probs
    chi2 = float((((obs - expected) ** 2) / expected).sum())

    gen = torch.Generator().manual_seed(seed)
    ref_sim = torch.poisson((n_ref * probs).expand(n_sim, k).contiguous(),
                            generator=gen)
    h_sim = torch.poisson((n_h * probs).expand(n_sim, k).contiguous(),
                          generator=gen)
    probs_sim = (ref_sim + _AUDIT_SMOOTH) / (
        ref_sim.sum(1, keepdim=True) + _AUDIT_SMOOTH * k)
    expected_sim = h_sim.sum(1, keepdim=True) * probs_sim
    chi2_sim = (((h_sim - expected_sim) ** 2) / expected_sim).sum(1)
    p_value = float((1.0 + (chi2_sim >= chi2).to(torch.float64).sum())
                    / (n_sim + 1.0))
    return p_value, chi2


# Outlier/tail audit.  The cleaning gate rejects invalid, label-noisy and
# duplicated samples; the calibrated marginal tests above detect a shift in
# each field separately.  Neither catches an *atypical sample* or a heavier
# difficulty tail in a held-out split, so a train-fitted typicality score is
# added here.  Every probability and the tail threshold are fitted on the
# training streams only; held-out splits are only scored (transform-only).
_OUTLIER_FIELDS = ("answer", "key_pos", "value_repeats", "naive_confusable")
_OUTLIER_QUANTILE = 0.99
# Conservative tail-rate bound: flag a held-out split only when its atypical
# rate exceeds both a multiple of the training rate and an absolute floor.
# At the default ~1% training rate this needs >5% of a held-out split to be
# in the training tail - far beyond same-distribution sampling noise (a
# Binomial(1024, 0.01) reaching 5% has probability ~1e-30), so the rule has
# essentially no false-positive rate while still catching a real tail shift.
_OUTLIER_RATE_MULTIPLIER = 5.0
_OUTLIER_RATE_FLOOR = 0.05

# Per-sample fields used by the training-stationarity diagnostic: each has
# exactly one value per sample, so "first half of the stream" maps onto
# whole samples (unlike pooled token fields, which are per-token).
_STATIONARITY_FIELDS = ("answer", "key_pos", "value_repeats",
                        "naive_confusable")


def _typicality_scores(split_values, reference) -> torch.Tensor:
    """Per-sample train-fitted surprisal over the structural audit fields.

    ``split_values`` maps each :data:`_OUTLIER_FIELDS` name to that split's
    per-sample values; ``reference`` holds Haldane-smoothed training
    category probabilities.  The generator draws the fields independently,
    so the joint negative log-likelihood is the sum of the per-field
    surprisals.  A category unseen in training keeps a finite (smoothed)
    probability, so the score is always finite; no held-out statistic enters
    the reference.
    """
    total = None
    for field in _OUTLIER_FIELDS:
        values = split_values[field].reshape(-1).to(torch.long)
        surprisal = -reference[field][values].log()
        total = surprisal if total is None else total + surprisal
    return total


def _outlier_audit(split_values, train_names, names, reference) -> dict:
    """Train-fitted atypical-sample rate per split.

    The threshold is the training ``_OUTLIER_QUANTILE`` of the typicality
    score (fit on the pooled training streams).  A split's outlier rate is
    the share of samples beyond that threshold.  Held-out splits are flagged
    only under the conservative rate rule documented above, so a genuine
    same-distribution split cannot trip the gate while a split whose
    difficulty tail has moved can.
    """
    def scores(name):
        return _typicality_scores(
            {field: split_values[(name, field)] for field in _OUTLIER_FIELDS},
            reference)

    train_scores = torch.cat([scores(name) for name in train_names])
    threshold = torch.quantile(train_scores, _OUTLIER_QUANTILE)
    flagged = {}
    for name in names:
        s = scores(name)
        flagged[name] = int((s > threshold).sum())
    rate = {name: flagged[name] / max(int(
        split_values[(name, _OUTLIER_FIELDS[0])].numel()), 1)
        for name in names}
    n_train = sum(int(split_values[(name, _OUTLIER_FIELDS[0])].numel())
                  for name in train_names)
    train_rate = (sum(flagged[name] for name in train_names)
                  / max(n_train, 1))
    bound = max(_OUTLIER_RATE_MULTIPLIER * train_rate, _OUTLIER_RATE_FLOOR)
    return {
        "fields": list(_OUTLIER_FIELDS),
        "quantile": float(_OUTLIER_QUANTILE),
        "threshold_train": float(threshold),
        "rate_train": float(train_rate),
        "rate_by_split": rate,
        "flagged_by_split": flagged,
        "rate_bound": float(bound),
        "shift_by_split": {
            name: bool(rate[name] > bound)
            for name in names if name not in train_names},
    }


def _support_coverage(pooled_counts, split_values, heldout_names, categories,
                      fields) -> dict:
    """Train-fitted category support and per-split unseen-value rates.

    For each field, the pooled training counts define which selectable
    categories were ever observed (ids 0=PAD and 1=SEP are never selectable
    content).  A held-out value whose category is absent from that support
    cannot be predicted (an answer) or meaningfully embedded (an input
    token), so its rate is a transparent bound on achievable accuracy.
    The support comes from training only; held-out values are looked up,
    never used to define it.
    """
    out = {}
    for field in fields:
        counts = pooled_counts[field]
        seen = counts > 0
        seen[:min(2, int(seen.numel()))] = False
        n_categories = max(int(categories[field]) - 2, 1)
        entry = {
            "n_categories": n_categories,
            "n_seen_train": int(seen.sum()),
            "fraction_seen_train": int(seen.sum()) / n_categories,
            "unseen_by_split": {},
        }
        for name in heldout_names:
            values = split_values[(name, field)].reshape(-1).to(torch.long)
            values = values[values >= 2]
            unseen = ~seen[values]
            entry["unseen_by_split"][name] = {
                "n": int(unseen.sum()),
                "rate": (float(unseen.to(torch.float64).mean())
                         if unseen.numel() else 0.0),
            }
        out[field] = entry
    return out


def _train_stationarity(split_values, splits, train_names, categories) -> dict:
    """First-half vs second-half drift inside each training stream.

    The held-out comparison asks whether evaluation data matches training;
    it cannot ask whether *training itself* is stationary.  If an early
    training block differs from a late one, a single held-out split cannot
    represent the future even with zero overlap.  Both halves come from the
    training stream, so this fits and scores on training data only.
    """
    out = {}
    for name in train_names:
        n_rows = int(splits[name]["idx"].size(0))
        half = n_rows // 2
        if half < 2:
            continue
        for field in _STATIONARITY_FIELDS:
            values = split_values[(name, field)].reshape(-1)
            if values.numel() != n_rows:
                continue
            early = _counts(values[:half], categories[field])
            late = _counts(values[half:], categories[field])
            p_value, chi2 = _distribution_shift_p(late, early)
            tv = _total_variation(_distribution_from_counts(late),
                                  _distribution_from_counts(early))
            out[f"{name}:{field}"] = {"p": p_value, "chi2": chi2, "tv": tv}
    return out


# Calibration of the answer-uniformity test (``answer_uniformity_test``):
# the sample size enters the test, so only the simulation count and its
# fixed seed are constants here.
_UNIFORMITY_SIM = 500
_UNIFORMITY_SIM_SEED = 8675309


def answer_uniformity_test(counts, n_sim=_UNIFORMITY_SIM,
                           seed=_UNIFORMITY_SIM_SEED) -> dict:
    """Monte Carlo goodness-of-fit of answer counts against *uniform*.

    Why this is a *decision*, not a diagnostic: the primary metric of the
    recall task is plain accuracy, and that choice is only meaningful
    because every value class is (by construction) equally likely - under
    a majority class accuracy degrades into a class-prior metric and the
    explicit F1 variants / PR-AUC companions in ``cage.metrics`` would
    become the headline.  The claim "the task is balanced multi-class"
    must therefore be *tested* on the drawn stream, not inferred from the
    generator description.

    ``counts`` is a 1-D count vector over the selectable classes (ids
    ``2..vocab-1`` - PAD and SEP are never answers; the caller slices
    accordingly).  The statistic is the Pearson chi-square against the
    uniform expectation ``n / K``.  The null is calibrated by a
    parametric bootstrap that redraws counts as independent
    ``Poisson(n / K)`` variates - the standard Poissonization of
    multinomial counts, whose conditioning correction is ``O(1/n)`` and
    negligible at these sizes (same convention as
    :func:`_distribution_shift_p`) - at the observed sample size, so the
    small expected count per class (about 10 at the default audit size,
    where the asymptotic chi-square table is not reliable) does not have
    to be trusted.

    Returns ``{"n", "n_classes", "expected_per_class", "chi2", "df",
    "p_value", "uniform_supported"}`` with
    ``p = (1 + #{chi2_sim >= chi2_obs}) / (n_sim + 1)`` and
    ``uniform_supported = p > 0.05``.  A *passing* test bounds gross
    imbalance at the audit sample size; it is not an equivalence proof
    for arbitrarily small skew (use the reported per-class shares for
    the descriptive bound).  Degenerate input (no samples, fewer than
    two classes) returns NaN fields and ``uniform_supported=False``.
    """
    c = torch.as_tensor(counts).detach().reshape(-1).to(torch.float64)
    n = float(c.sum())
    k = c.numel()
    out = {"n": int(n), "n_classes": int(k),
           "expected_per_class": float(n / k) if k else float("nan"),
           "chi2": float("nan"), "df": int(max(k - 1, 0)),
           "p_value": float("nan"), "uniform_supported": False}
    if k < 2 or n <= 0.0:
        return out
    expected = n / k
    chi2 = float((((c - expected) ** 2) / expected).sum())
    gen = torch.Generator().manual_seed(seed)
    sim = torch.poisson(torch.full((n_sim, k), expected), generator=gen)
    chi2_sim = (((sim - expected) ** 2) / expected).sum(1)
    p_value = float((1.0 + (chi2_sim >= chi2).to(torch.float64).sum())
                    / (n_sim + 1.0))
    out.update({"chi2": chi2, "p_value": p_value,
                "uniform_supported": bool(p_value > 0.05)})
    return out


def _fingerprints(idx):
    # ``tolist`` already yields Python ints; no per-token conversion needed.
    return {tuple(row) for row in idx.tolist()}


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
    * **per-field distribution shift**: for pooled tokens, supervised
      answers, key / value sub-populations and the structural difficulty
      covariates (key position, repeated answer values, naive-confusable
      queries), the total-variation distance between each held-out split
      and the pooled training distribution is reported against a same-size
      noise floor measured between disjoint training streams
      (:func:`_noise_floor`), *and* the same-distribution null is tested
      with a Monte Carlo-calibrated Pearson chi-square
      (:func:`_distribution_shift_p`) whose null redraws both count vectors
      at their audit sizes - the TV/floor pair is descriptive, the
      calibrated p-value is what keeps a sparse field (a few hundred
      answer classes at ~1e3 samples) from hiding a real shift behind an
      unattainable tolerance.  The difficulty fields matter because equal
      marginals do not imply equal task difficulty: a held-out split with
      systematically later keys or more ambiguous values has a different
      achievable accuracy, so an accuracy contrast against it would
      attribute a data difference to the mechanism.
    * **category-support coverage**: an answer class absent from the
      training streams is unpredictable and an input token absent from them
      is unembedable; for the answer, token, key and value fields the train
      support and each held-out split's unseen-value rate are reported as a
      bound on achievable accuracy.
    * **atypical-sample / difficulty-tail audit**: a per-sample typicality
      score (the train-fitted surprisal over the answer, key position,
      repeated-value and naive-confusable fields) is thresholded at the
      training tail quantile, and the share of samples beyond that threshold
      is reported for every split.  The reference and the threshold are fit
      on the training streams only.  A held-out split whose *tail* moved can
      pass every marginal test; it is flagged under a conservative rate
      bound (:data:`_OUTLIER_RATE_MULTIPLIER`, :data:`_OUTLIER_RATE_FLOOR`).
    * **training-stream stationarity**: the same calibrated shift test run
      between the first and second halves of each training stream, so
      non-stationary training is visible before a held-out split is trusted
      to represent the future.  This uses training data only.
    * **cleaning diagnostics** from :func:`validate_batch`, run on every
      batch of every split: NaN/Inf, out-of-range ids, structural
      violations and ambiguous/misaligned label noise raise; a whole-split
      exact duplicate-row count and the unique-key rate are returned, plus
      the subset of duplicates that occur *within* one batch (a stream-level
      count above the within-batch count means the same sample was drawn
      in two different batches of the same stream; overlaps *between*
      streams are the leakage signal above).

    The audit only *reads* data: it fits nothing on held-out streams and
    writes nothing.  A shared sample or a gap above the noise floor
    invalidates the HB3 decision and every number reported above it.
    """
    unique = {}
    for seed in train_seeds:
        unique.setdefault(stream_seed_key(seed), int(seed))
    train_seeds = tuple(unique.values())
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

    # Reference statistics are fitted on the training streams only.  The
    # structural difficulty covariates are derived once per split and the
    # per-field values / counts are reused by the floor, shift and TV
    # readouts instead of being recomputed for every threshold/test.
    train_names = [f"train:{seed}" for seed in train_seeds]
    categories = {field: _field_categories(field, vocab, n_pairs)
                  for field in _AUDIT_FIELDS}
    features = {name: _difficulty_features(split, n_pairs)
                for name, split in splits.items()}

    def values(name, field):
        if field in _AUDIT_DERIVED_FIELDS:
            return features[name][field]
        return _field_values(splits[name], n_pairs, field)

    split_values = {(name, field): values(name, field)
                    for name in names for field in _AUDIT_FIELDS}
    pooled = {field: torch.cat([split_values[(name, field)]
                                for name in train_names])
              for field in _AUDIT_FIELDS}
    pooled_counts = {field: _counts(pooled[field], categories[field])
                     for field in _AUDIT_FIELDS}
    reference = {field: _distribution_from_counts(pooled_counts[field])
                 for field in _AUDIT_FIELDS}
    floors = {
        field: _noise_floor([split_values[(name, field)]
                             for name in train_names], categories[field])
        for field in _AUDIT_FIELDS
    }
    observed_counts = {
        (name, field): _counts(split_values[(name, field)], categories[field])
        for name in ("val", "test", "stats") for field in _AUDIT_FIELDS}
    shift_p = {field: {} for field in _AUDIT_FIELDS}
    shift_chi2 = {field: {} for field in _AUDIT_FIELDS}
    for field in _AUDIT_FIELDS:
        for name in ("val", "test", "stats"):
            p_value, chi2 = _distribution_shift_p(
                observed_counts[(name, field)], pooled_counts[field])
            shift_p[field][name] = p_value
            shift_chi2[field][name] = chi2

    # Train-fitted atypical-sample threshold and train-only stationarity.
    # The reference is Haldane-smoothed so a category absent from training
    # still yields a finite surprisal instead of an infinite score.
    smoothed_reference = {
        field: (pooled_counts[field].to(torch.float64) + _AUDIT_SMOOTH)
        / (pooled_counts[field].sum()
           + _AUDIT_SMOOTH * categories[field])
        for field in _AUDIT_FIELDS}
    outliers = _outlier_audit(split_values, train_names, names,
                              smoothed_reference)
    stationarity = _train_stationarity(split_values, splits, train_names,
                                       categories)

    # Data-sufficiency disclosure: a category that never occurs in the
    # training streams is unpredictable (an answer class) or unembedable (an
    # input token), so report the train support and the share of each
    # held-out split's values outside it.  The support is fitted on train
    # counts only and held-out values are merely looked up
    # (transform-only); the rate bounds the accuracy any model can reach,
    # which the marginal shift tests do not report.
    support_coverage = _support_coverage(
        pooled_counts, split_values, ("val", "test", "stats"), categories,
        ("answer", "token", "key", "value"))

    # Descriptive difficulty profile per split: the calibrated tests decide
    # *whether* a shift exists, these shares say *what* differs (mean pair
    # index of the matching key, repeated-answer rate, naive-confusable
    # rate), all computed without any held-out-fitted statistic.
    difficulty = {}
    for name, split in splits.items():
        feat = features[name]
        n = max(int(split["idx"].size(0)), 1)
        difficulty[name] = {
            "mean_key_pos": float(feat["key_pos"].to(torch.float64)
                                  .mean()) if split["idx"].size(0) else
            float("nan"),
            "value_repeat_rate": float(feat["value_repeats"].sum()) / n,
            "naive_confusable_rate": (
                float(feat["naive_confusable"].sum()) / n),
        }

    out = {
        "train_seeds": list(train_seeds),
        "val_seed": VAL_SEED,
        "test_seed": TEST_SEED,
        "stats_seed": STATS_SEED,
        "n_per_split": int(batch) * int(batches),
        "audit_fields": list(_AUDIT_FIELDS),
        "audit_field_categories": categories,
        "tv_noise_floor": floors,
        "token_tv_noise_floor": floors["token"],
        "shift_p_value": shift_p,
        "shift_chi2": shift_chi2,
        "shift_test": {"n_sim": _AUDIT_SIM, "seed": _AUDIT_SIM_SEED,
                       "smoothing": _AUDIT_SMOOTH},
        "difficulty_by_split": difficulty,
        "outlier_audit": outliers,
        "support_coverage": support_coverage,
        "train_stationarity": stationarity,
        "train_stationarity_min_p": min(
            (v["p"] for v in stationarity.values()), default=1.0),
        "pairwise_duplicate_overlap": pairwise,
        "max_pairwise_duplicate_overlap": max(pairwise.values(), default=0),
    }
    per_split_dup = {name: sum(d["n_duplicate_rows"] for d in s["diagnostics"])
                     for name, s in splits.items()}
    stream_dup = {name: _count_duplicate_rows(s["idx"])
                  for name, s in splits.items()}
    out["n_duplicate_rows_train"] = sum(stream_dup[n] for n in train_names)
    out["n_duplicate_rows_within_batch_train"] = sum(
        per_split_dup[n] for n in train_names)
    out["unique_key_rate"] = min(
        d["unique_key_rate"] for s in splits.values()
        for d in s["diagnostics"])
    for name in ("val", "test", "stats"):
        out[f"duplicate_overlap_train_{name}"] = pairwise[
            f"{train_names[0]}|{name}"]
        out[f"n_duplicate_rows_{name}"] = stream_dup[name]
        out[f"n_duplicate_rows_within_batch_{name}"] = per_split_dup[name]
        for field in _AUDIT_FIELDS:
            out[f"{field}_tv_train_{name}"] = _total_variation(
                reference[field],
                _distribution_from_counts(observed_counts[(name, field)]))
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


def _self_test() -> int:
    """Deterministic self-checks of the split-audit machinery.

    Pins the two properties the distribution-shift decision rests on:
    a same-source held-out draw must not be flagged (calibration), and a
    held-out split drawn from half the class range must be (power) even
    though its total-variation gap stays below the historical
    ``3 x noise floor`` tolerance - the blindness that motivated the
    Monte Carlo-calibrated chi-square.  Also pins the duplicate-row
    accounting used by the split audit, including duplicates that only
    appear when two batches are concatenated.  Run with
    ``python3 -m cage.data``.
    """
    failures = []

    def check(name: str, cond: bool) -> None:
        if not cond:
            failures.append(name)

    vocab, classes = 512, 510
    n_h, n_ref = 1024, 5120

    def draw_counts(k, n, seed):
        gen = torch.Generator().manual_seed(seed)
        return torch.bincount(torch.randint(0, k, (n,), generator=gen),
                              minlength=vocab).to(torch.int64)

    def _tv_between_counts(a, b):
        pa = a.to(torch.float64) / a.sum()
        pb = b.to(torch.float64) / b.sum()
        return 0.5 * float((pa - pb).abs().sum())

    reference = draw_counts(classes, n_ref, 11)
    same_source = draw_counts(classes, n_h, 12)
    p_same, _ = _distribution_shift_p(same_source, reference, n_sim=400,
                                      seed=90210)
    check("same_source_not_flagged", p_same > 0.05)

    half_range = draw_counts(classes // 2, n_h, 13)
    p_shift, chi2_shift = _distribution_shift_p(half_range, reference,
                                                n_sim=400, seed=90210)
    check("half_range_shift_detected", p_shift <= 1.0 / 401.0)
    check("shift_chi2_large", chi2_shift > 900.0)

    floor = _tv_between_counts(same_source, draw_counts(classes, n_h, 14))
    tv_shift = _tv_between_counts(half_range, reference)
    check("old_tv_tolerance_would_pass", tv_shift < 3.0 * floor)

    # Answer-uniformity decision: the balance claim behind "accuracy is the
    # headline metric" must survive a uniform draw and reject a
    # half-range draw at the default audit sample size (5 seeds x 4
    # batches x 256 ~ 5120 samples, ~10 per class).
    n_audit = 5 * 4 * 256
    uni_counts = draw_counts(classes, n_audit, 31)[:classes]
    uni = answer_uniformity_test(uni_counts, n_sim=400, seed=90210)
    check("uniform_draw_supported", uni["uniform_supported"])
    check("uniform_chi2_finite", math.isfinite(uni["chi2"]))
    skewed_counts = draw_counts(classes // 2, n_audit, 32)[:classes]
    skewed = answer_uniformity_test(skewed_counts, n_sim=400, seed=90210)
    check("skewed_draw_rejected", not skewed["uniform_supported"])
    check("skewed_p_below_floor", skewed["p_value"] <= 1.0 / 401.0)
    degenerate = answer_uniformity_test(torch.zeros(1, dtype=torch.int64))
    check("degenerate_input_not_supported",
          not degenerate["uniform_supported"]
          and math.isnan(degenerate["p_value"]))

    # Structural difficulty covariates: hand-built sequences pin where the
    # matching key sits, whether the answer value repeats, and whether the
    # naive last-match rule is misled by the query token appearing later
    # as a value.
    feat_split = {
        "idx": torch.tensor([[5, 10, 7, 7, 9, 12, SEP_ID, 7],
                             [3, 20, 4, 20, 5, 9, SEP_ID, 4]]),
        "answer": torch.tensor([7, 20]),
    }
    feat = _difficulty_features(feat_split, 3)
    check("difficulty_key_pos",
          feat["key_pos"].tolist() == [1, 1])
    check("difficulty_value_repeats",
          feat["value_repeats"].tolist() == [0, 1])
    check("difficulty_naive_confusable",
          feat["naive_confusable"].tolist() == [1, 0])

    # A shift in retrieval distance must be detected by the same calibrated
    # test used for the token marginals, even when only the *position*
    # distribution changes.
    gen = torch.Generator().manual_seed(41)
    ref_kp = torch.bincount(torch.randint(0, 16, (4096,), generator=gen),
                            minlength=16)
    obs_kp = torch.bincount(torch.randint(0, 8, (1024,), generator=gen),
                            minlength=16)
    p_kp, _ = _distribution_shift_p(obs_kp, ref_kp, n_sim=400, seed=90210)
    check("key_pos_shift_detected", p_kp <= 1.0 / 401.0)
    same_kp = torch.bincount(torch.randint(0, 16, (1024,), generator=gen),
                             minlength=16)
    p_kp_same, _ = _distribution_shift_p(same_kp, ref_kp, n_sim=400,
                                         seed=90210)
    check("key_pos_same_source_not_flagged", p_kp_same > 0.05)

    # Time-order split contract: contiguous, chronological blocks with no
    # future sample in train, and a crossing assignment must be rejected.
    times = torch.arange(100, dtype=torch.float64)
    t_split = time_ordered_split(100)
    assert_time_ordered(times, t_split)
    check("time_split_contiguous_chronological",
          t_split["train"] == slice(0, 70)
          and t_split["val"] == slice(70, 85)
          and t_split["test"] == slice(85, 100))
    try:
        assert_time_ordered(
            times, {"train": slice(0, 60), "val": slice(50, 80),
                    "test": slice(80, 100)})
        check("time_leakage_rejected", False)
    except ValueError:
        check("time_leakage_rejected", True)

    # Group split contract: whole groups on one side, full coverage, and a
    # per-sample crossing assignment must be rejected.
    groups = torch.arange(200, dtype=torch.long) // 4
    g_split = group_disjoint_split(groups, seed=3)
    assert_group_disjoint(groups, g_split)
    check("group_split_covers_every_sample",
          sum(int(v.numel()) for v in g_split.values()) == 200)
    check("group_split_nonempty",
          all(int(v.numel()) > 0 for v in g_split.values()))
    g_split_again = group_disjoint_split(groups, seed=3)
    check("group_split_deterministic",
          all(torch.equal(g_split[k], g_split_again[k]) for k in g_split))
    try:
        assert_group_disjoint(
            groups, {"train": torch.arange(0, 100),
                     "val": torch.arange(100, 150),
                     "test": torch.arange(150, 200)})
        check("group_crossing_rejected", False)
    except ValueError:
        check("group_crossing_rejected", True)
    try:
        assert_time_ordered(torch.tensor([0.0, float("nan"), 2.0]),
                            time_ordered_split(3))
        check("time_nonfinite_rejected", False)
    except ValueError:
        check("time_nonfinite_rejected", True)
    try:
        group_disjoint_split(torch.tensor([0.0, 1.0, float("nan"), 2.0, 3.0]))
        check("group_nonfinite_rejected", False)
    except ValueError:
        check("group_nonfinite_rejected", True)

    # Outlier/tail audit: a same-distribution held-out draw must not be
    # flagged, while a held-out draw pushed into the rare tail of the same
    # train-fitted reference must be.
    rare_probs = {
        "answer": torch.tensor(
            [0.5, 0.2, 0.1, 0.07, 0.05, 0.04, 0.03, 0.01],
            dtype=torch.float64),
        "key_pos": torch.tensor([0.7, 0.2, 0.08, 0.02], dtype=torch.float64),
        "value_repeats": torch.tensor([0.9, 0.1], dtype=torch.float64),
        "naive_confusable": torch.tensor([0.8, 0.2], dtype=torch.float64),
    }

    def draw_rare_split(seed, n=2048):
        gen = torch.Generator().manual_seed(seed)
        return {field: torch.multinomial(p, n, replacement=True,
                                         generator=gen)
                for field, p in rare_probs.items()}

    odd_values = {}
    for i, name in enumerate(("train:0", "train:1")):
        for field, vals in draw_rare_split(50 + i).items():
            odd_values[(name, field)] = vals
    for seed, name in ((60, "val"), (61, "test")):
        for field, vals in draw_rare_split(seed).items():
            odd_values[(name, field)] = vals
    tail = draw_rare_split(62)
    tail["answer"] = torch.full((2048,), 7, dtype=torch.long)
    tail["key_pos"] = torch.full((2048,), 3, dtype=torch.long)
    tail["value_repeats"] = torch.ones(2048, dtype=torch.long)
    tail["naive_confusable"] = torch.ones(2048, dtype=torch.long)
    for field, vals in tail.items():
        odd_values[("stats", field)] = vals
    odd_names = ["train:0", "train:1", "val", "test", "stats"]
    odd = _outlier_audit(odd_values, ["train:0", "train:1"], odd_names,
                         rare_probs)
    check("outlier_threshold_is_train_tail", odd["rate_train"] <= 0.05)
    check("outlier_same_source_not_flagged",
          not any(odd["shift_by_split"][n] for n in ("val", "test")))
    check("outlier_tail_shift_detected", odd["shift_by_split"]["stats"])
    check("outlier_tail_rate_near_one",
          odd["rate_by_split"]["stats"] > 0.9)

    # Actual-stream replay: same seed and geometry is caught as overlap,
    # two different seeds are disjoint, and re-reading one split at two
    # geometries is same-group and must not fail the leakage gate.
    train_spec = {"seed": 0, "batch": 8, "steps": 3, "group": "train"}
    spec_overlap = actual_stream_overlap_audit(
        4, 64, "cpu",
        train_specs={"train:0": train_spec},
        heldout_specs={
            "val": {"seed": VAL_SEED, "batch": 8, "batches": 2,
                    "group": "val"},
            "test": {"seed": TEST_SEED, "batch": 8, "batches": 2,
                     "group": "test"},
        })
    check("actual_overlap_zero_between_seeds",
          spec_overlap["max_overlap"] == 0)
    same_stream = actual_stream_overlap_audit(
        4, 64, "cpu",
        train_specs={"train:0": {"seed": 5, "batch": 8, "steps": 2,
                                 "group": "train"}},
        heldout_specs={"leaky": {"seed": 5, "batch": 8, "batches": 2,
                                 "group": "val"}})
    check("actual_overlap_detects_same_stream",
          same_stream["max_overlap"]
          == same_stream["streams"]["train:0"]["n_rows"] > 0)
    same_split = actual_stream_overlap_audit(
        4, 64, "cpu",
        train_specs={"train:0": train_spec},
        heldout_specs={
            "val_monitor": {"seed": VAL_SEED, "batch": 8, "batches": 2,
                            "group": "val"},
            "val_final": {"seed": VAL_SEED, "batch": 8, "batches": 3,
                          "group": "val"},
        })
    same_pair = same_split["pairwise_overlap"]["val_monitor|val_final"]
    check("same_group_prefix_overlap_observed", same_pair > 0)
    check("same_group_overlap_not_leakage",
          same_split["max_overlap"] == 0 and same_pair > 0)

    # Dataset-versioning fingerprint must depend on the batch geometry the
    # stream is realized with, not only on the seed.
    fp_a = stream_fingerprint(4, 64, "cpu", batch=4, seed=77)
    fp_b = stream_fingerprint(4, 64, "cpu", batch=8, seed=77)
    check("fingerprint_geometry_sensitive",
          fp_a["sha256"] != fp_b["sha256"])
    check("fingerprint_repeatable",
          fp_a == stream_fingerprint(4, 64, "cpu", batch=4, seed=77))

    # Sealed-stream guard: selection may read the validation stream only.
    # Both the test and the statistics streams are held out, so using
    # either for a choice would turn it into an implicit validation set;
    # reading them *without* selection (final report / pre-registered gate
    # rules / mechanism statistics) stays allowed.
    check_eval_seed(VAL_SEED, selection=True)
    for bad in (TEST_SEED, STATS_SEED):
        try:
            check_eval_seed(bad, selection=True)
            check(f"selection_refused_on_stream_{bad}", False)
        except ValueError:
            check(f"selection_refused_on_stream_{bad}", True)
        check_eval_seed(bad)

    # Stream-level duplicate accounting: rows repeated across *different*
    # batches of one stream must be counted, not only within one batch.
    gen = torch.Generator().manual_seed(21)
    batch_a, _, _ = make_batch(8, 4, 64, "cpu", gen=gen)
    batch_b, _, _ = make_batch(8, 4, 64, "cpu", gen=gen)
    check("clean_batch_has_no_duplicates",
          _count_duplicate_rows(batch_a) == 0)
    check("stream_duplicates_counted",
          _count_duplicate_rows(torch.cat([batch_a, batch_a[:3], batch_b])) == 3)

    # Label-alignment and reserved-id gates: freshly drawn sparse and dense
    # batches pass the cleaning gate; a target that is neither ``-1`` nor
    # the true continuation is rejected (off-by-one / foreign-label
    # misalignment), and a reserved answer id (PAD/SEP) is rejected even
    # when the paired value and the supervised entry agree with it.
    gen = torch.Generator().manual_seed(22)
    sparse_idx, sparse_tgt, sparse_meta = make_batch(8, 4, 64, "cpu", gen=gen)
    check("validate_accepts_sparse_targets",
          validate_batch(sparse_idx, sparse_tgt, sparse_meta, 64)["n"] == 8)
    dense_idx, dense_tgt, dense_meta = make_batch(
        8, 4, 64, "cpu", dense_lm=True, gen=torch.Generator().manual_seed(23))
    check("validate_accepts_dense_targets",
          validate_batch(dense_idx, dense_tgt, dense_meta, 64)["n"] == 8)

    misaligned = sparse_tgt.clone()
    misaligned[:, 0] = (sparse_idx[:, 1] + 1) % 64
    try:
        validate_batch(sparse_idx, misaligned, sparse_meta, 64)
        check("validate_rejects_misaligned_target", False)
    except ValueError:
        check("validate_rejects_misaligned_target", True)

    rows = torch.arange(8)
    reserved_idx = sparse_idx.clone()
    reserved_idx[rows, sparse_meta["key_pos"] + 1] = 0
    reserved_meta = {"query_pos": sparse_meta["query_pos"],
                     "key_pos": sparse_meta["key_pos"],
                     "answer": torch.zeros(8, dtype=torch.long)}
    reserved_tgt = torch.full_like(sparse_tgt, -1)
    reserved_tgt[rows, reserved_meta["query_pos"]] = 0
    try:
        validate_batch(reserved_idx, reserved_tgt, reserved_meta, 64)
        check("validate_rejects_reserved_answer", False)
    except ValueError:
        check("validate_rejects_reserved_answer", True)

    # The predicate that guards every training batch must agree with the
    # strict checker exactly: it may not accept a batch the strict checker
    # rejects (that would silently disable the cleaning gate), and it may
    # not reject a valid one.
    def gate_agrees(i_, t_, m_, vocab_):
        held = _batch_invariants_hold(i_, t_, m_, vocab_)
        try:
            _validate_batch_strict(i_, t_, m_, vocab_)
            strict = True
        except Exception:  # noqa: BLE001 - any raise is a rejection
            strict = False
        return held == strict

    nan_idx = sparse_idx.clone().float()
    nan_idx[0, 0] = float("nan")
    bad_qp = sparse_meta["query_pos"] + 1
    meta_qp = {"query_pos": bad_qp, "key_pos": sparse_meta["key_pos"],
               "answer": sparse_meta["answer"]}
    meta_kp = {"query_pos": sparse_meta["query_pos"],
               "key_pos": sparse_meta["key_pos"] + 2,
               "answer": sparse_meta["answer"]}
    meta_qp_at_key = {"query_pos": sparse_meta["key_pos"],
                      "key_pos": sparse_meta["key_pos"],
                      "answer": sparse_meta["answer"]}
    soft_tgt = sparse_tgt.clone()
    soft_tgt[0, 0] = -2
    final_tgt = sparse_tgt.clone()
    final_tgt[:, -1] = sparse_meta["answer"] + 1
    dup_key = sparse_idx.clone()
    dup_key[:, 2] = sparse_idx[:, 0]
    for name, (i_, t_, m_) in {
            "valid_sparse": (sparse_idx, sparse_tgt, sparse_meta),
            "valid_dense": (dense_idx, dense_tgt, dense_meta),
            "misaligned_target": (sparse_idx, misaligned, sparse_meta),
            "reserved_answer": (reserved_idx, reserved_tgt, reserved_meta),
            "idx_out_of_range": (sparse_idx.clone().fill_(64), sparse_tgt,
                                 sparse_meta),
            "idx_not_2d": (sparse_idx.reshape(-1), sparse_tgt, sparse_meta),
            "idx_with_nan": (nan_idx, sparse_tgt, sparse_meta),
            "query_pos_out_of_range": (sparse_idx, sparse_tgt, meta_qp),
            "query_pos_not_final": (sparse_idx, None, meta_qp_at_key),
            "key_pos_disagrees": (sparse_idx, sparse_tgt, meta_kp),
            "key_not_unique": (dup_key, sparse_tgt, sparse_meta),
            "target_shape_wrong": (sparse_idx, sparse_tgt.reshape(-1),
                                   sparse_meta),
            "target_out_of_range": (sparse_idx, soft_tgt, sparse_meta),
            "final_target_wrong": (sparse_idx, final_tgt, sparse_meta),
    }.items():
        check(f"invariants_agree_{name}", gate_agrees(i_, t_, m_, 64))

    # End-to-end wiring of the train-fitted sections: a small deterministic
    # same-distribution audit must expose the atypical-tail, coverage and
    # stationarity readouts and must not trip the tail gate.
    e2e = split_leakage_audit(8, 32, "cpu", batch=32, batches=8,
                              train_seeds=(0,))
    check("audit_e2e_outlier_fields",
          e2e["outlier_audit"]["fields"] == list(_OUTLIER_FIELDS))
    check("audit_e2e_no_tail_shift",
          not any(e2e["outlier_audit"]["shift_by_split"].values()))
    check("audit_e2e_stationarity_present",
          len(e2e["train_stationarity"]) == len(_STATIONARITY_FIELDS))
    cov = e2e["support_coverage"]
    check("audit_e2e_coverage_sane",
          all(0 <= cov[field]["n_seen_train"] <= cov[field]["n_categories"]
              and all(0.0 <= cov[field]["unseen_by_split"][name]["rate"] <= 1.0
                      for name in ("val", "test", "stats"))
              for field in ("answer", "token", "key", "value")))
    check("audit_e2e_outlier_rates_sane",
          all(0.0 <= v <= 1.0
              for v in e2e["outlier_audit"]["rate_by_split"].values()))

    if failures:
        print("data self-test FAILED: " + ", ".join(failures))
        return 1
    print(f"data self-test: all checks passed "
          f"(same-source p={p_same:.3f}, half-range p={p_shift:.4f}, "
          f"chi2={chi2_shift:.1f}, tv={tv_shift:.3f} vs 3x floor="
          f"{3.0 * floor:.3f}; uniformity p={uni['p_value']:.3f} "
          f"chi2={uni['chi2']:.1f}, skewed p={skewed['p_value']:.4f}; "
          f"key-pos shift p={p_kp:.4f} vs same-source "
          f"p={p_kp_same:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
