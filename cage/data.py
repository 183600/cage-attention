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
import operator
from collections.abc import Mapping

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

    ``seed`` must be integral: silently truncating a fractional value would
    make it draw a *different* stream than the one the caller named (and
    would let e.g. ``0.5`` alias the protected seed ``0``).
    """
    if isinstance(seed, bool):
        raise ValueError(f"stream seed must be an integer, got {seed!r}")
    try:
        value = operator.index(seed)
    except TypeError as exc:
        raise ValueError(
            f"stream seed must be an integer, got {seed!r}") from exc
    return int(value) & 0xFFFFFFFF


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
    if isinstance(n_samples, bool):
        raise ValueError(
            f"n_samples must be an integer, got {n_samples!r}")
    try:
        n = operator.index(n_samples)
    except TypeError as exc:
        raise ValueError(
            f"n_samples must be an integer, got {n_samples!r}") from exc
    if n < 3:
        raise ValueError(
            f"time_ordered_split needs at least 3 samples for three "
            f"non-empty blocks, got {n_samples!r}")
    try:
        train_frac = float(train_frac)
        val_frac = float(val_frac)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"train_frac and val_frac must be numbers in (0, 1), got "
            f"{train_frac!r} and {val_frac!r}") from exc
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac!r}")
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac!r}")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac must leave a test block, got "
            f"{train_frac!r} + {val_frac!r}")
    n_val = max(1, int(n * val_frac))
    n_test = max(1, n - int(n * (train_frac + val_frac)))
    n_train = n - n_val - n_test
    if n_train < 1:
        # Extremely skewed fractions at a tiny n: keep one sample per block.
        n_train = 1
        n_val = 1
        n_test = n - 2
    out = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n),
    }
    assert_split_partition(n, out)
    return out


def _group_codes(groups) -> torch.Tensor:
    """Encode group ids as stable integer codes.

    Real group ids are often strings (user ids, document ids, object names),
    which ``torch.as_tensor`` cannot consume.  Map any hashable id to a dense
    integer code in first-appearance order; tensor inputs are kept as-is so
    numeric ids retain their exact values.  Non-finite numeric ids and
    unhashable ids are rejected rather than silently coerced.
    """
    if isinstance(groups, torch.Tensor):
        if groups.dim() != 1:
            raise ValueError(
                "groups must be one-dimensional (one id per sample), got "
                f"shape {tuple(groups.shape)}; a 2-D matrix would be "
                f"flattened into the wrong number of samples")
        g = groups.reshape(-1)
        if g.is_complex():
            raise ValueError("group ids must not be complex")
        if g.is_floating_point() and not bool(torch.isfinite(g).all()):
            raise ValueError("group ids must be finite (no NaN/Inf)")
        return g
    try:
        iterator = iter(groups)
    except TypeError as exc:
        raise ValueError(
            f"groups must be an iterable of ids, got "
            f"{type(groups).__name__}") from exc
    mapping, codes = {}, []
    for value in iterator:
        if isinstance(value, complex):
            # The tensor branch rejects complex ids; the generic branch must
            # not silently turn ``1+2j`` into the real id ``1`` either.
            raise ValueError("group ids must be real, not complex")
        try:
            if value in mapping:
                codes.append(mapping[value])
            else:
                mapping[value] = len(mapping)
                codes.append(mapping[value])
        except TypeError as exc:
            raise ValueError(
                f"group ids must be hashable, got "
                f"{type(value).__name__}") from exc
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("group ids must be finite (no NaN/Inf)")
    return torch.tensor(codes, dtype=torch.long)


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
    try:
        train_frac = float(train_frac)
        val_frac = float(val_frac)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"need 0 < train_frac, 0 < val_frac and train_frac + val_frac "
            f"< 1, got {train_frac!r}, {val_frac!r}") from exc
    if not (0.0 < train_frac < 1.0) or not (0.0 < val_frac < 1.0) or (
            train_frac + val_frac >= 1.0):
        raise ValueError(
            f"need 0 < train_frac, 0 < val_frac and train_frac + val_frac "
            f"< 1, got {train_frac!r}, {val_frac!r}")
    g = _group_codes(groups)
    n = int(g.numel())
    if n == 0:
        raise ValueError("groups must contain at least one sample")
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
    # Canonicalise exactly like the data-stream generators: a split assigned
    # under an aliased seed must be the same split, never a different one.
    perm = order[torch.randperm(
        n_groups,
        generator=torch.Generator().manual_seed(stream_seed_key(seed)))]
    counts = torch.bincount(inverse, minlength=n_groups).to(torch.float64)
    cuts = (train_frac * n, (train_frac + val_frac) * n)
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
    targets = {
        "train": train_frac * n,
        "val": val_frac * n,
        "test": (1.0 - train_frac - val_frac) * n,
    }

    def _masses():
        return {name: sum(float(counts[gid]) for gid in gids)
                for name, gids in assigned.items()}

    def _objective(masses=None):
        masses = _masses() if masses is None else masses
        return sum((masses[name] - targets[name]) ** 2 for name in targets)

    # The midpoint pass can leave a split empty when one group dominates the
    # cumulative mass.  Filling it with an arbitrary last group (the old
    # behaviour) can move the *largest* group out of ``train`` and destroy
    # the requested mass allocation.  Donate instead the group whose move
    # yields the smallest squared deviation from the three target masses, so
    # the non-empty guarantee does not come at the cost of the split shape.
    for name in ("train", "val", "test"):
        if assigned[name]:
            continue
        best = None
        for donor in assigned:
            if len(assigned[donor]) <= 1:
                continue
            for gid in list(assigned[donor]):
                masses = _masses()
                masses[donor] -= float(counts[gid])
                masses[name] += float(counts[gid])
                candidate = (_objective(masses), donor, int(gid))
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            raise ValueError(
                f"cannot give every split a group at n_groups={n_groups}")
        _, donor, gid = best
        assigned[donor].remove(gid)
        assigned[name].append(gid)

    # The midpoint pass (and even the non-empty repair above) can leave a
    # whole-group assignment far from the requested mass split when one
    # group dominates: e.g. 54/17/19 at 70/15/15 may put the 54 group in
    # ``val``.  Improve the assignment by best-improvement single moves and
    # pairwise swaps against the same squared-deviation objective.  Moves
    # never empty a split; swaps preserve every split's non-emptiness.  The
    # search is bounded, deterministic, and skipped for very large group
    # counts where the quadratic swap scan would dominate (moves still run).
    split_names = ("train", "val", "test")

    def _improve_once() -> bool:
        current = _objective()
        base_masses = _masses()
        best = None
        for donor in split_names:
            if len(assigned[donor]) <= 1:
                continue
            for gid in assigned[donor]:
                mass = float(counts[gid])
                for target in split_names:
                    if target == donor:
                        continue
                    masses = dict(base_masses)
                    masses[donor] -= mass
                    masses[target] += mass
                    candidate = (_objective(masses), 0, donor, target,
                                 int(gid), None)
                    if best is None or candidate < best:
                        best = candidate
        if n_groups <= 128:
            for i, left in enumerate(split_names):
                for right in split_names[i + 1:]:
                    for gid_left in assigned[left]:
                        ml = float(counts[gid_left])
                        for gid_right in assigned[right]:
                            mr = float(counts[gid_right])
                            masses = dict(base_masses)
                            masses[left] += mr - ml
                            masses[right] += ml - mr
                            candidate = (_objective(masses), 1, left, right,
                                         int(gid_left), int(gid_right))
                            if best is None or candidate < best:
                                best = candidate
        if best is None or best[0] >= current - 1e-12:
            return False
        _, kind, left, right, gid_left, gid_right = best
        if kind == 0:
            assigned[left].remove(gid_left)
            assigned[right].append(gid_left)
        else:
            assigned[left].remove(gid_left)
            assigned[right].remove(gid_right)
            assigned[left].append(gid_right)
            assigned[right].append(gid_left)
        return True

    for _ in range(32):
        if not _improve_once():
            break
    out = {}
    for name, gids in assigned.items():
        mask = torch.isin(inverse, torch.tensor(gids, dtype=torch.long))
        out[name] = torch.nonzero(mask, as_tuple=False).flatten()
    assert_split_partition(n, out)
    assert_group_disjoint(g, out)
    return out


def _as_indices(index, n: int) -> torch.Tensor:
    """Normalize a split index container to a ``LongTensor`` of positions.

    Boolean masks are interpreted as masks of the sample axis, *not* cast to
    0/1 positions (the old behaviour silently audited sample 0/1 for every
    ``True`` entry).  Floating-point and complex index arrays are rejected
    instead of being truncated: a split must name whole samples.  The
    returned positions are checked to lie in ``[0, n)``; negative indices
    would otherwise wrap and a wrapped index could hide a crossing sample.
    Duplicate detection is left to :func:`assert_split_partition`, because
    a repeated index is only invalid in the context of a split assignment.
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"sample count must be non-negative, got {n}")
    if isinstance(index, slice):
        idx = torch.arange(*index.indices(n), dtype=torch.long)
    elif isinstance(index, range):
        idx = torch.tensor(list(index), dtype=torch.long)
    else:
        try:
            t = torch.as_tensor(index)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                f"split indices must be integers or a boolean mask, got "
                f"{type(index).__name__}") from exc
        if t.dtype == torch.bool:
            m = t.reshape(-1)
            if int(m.numel()) != n:
                raise ValueError(
                    f"boolean split mask must have length {n}, got "
                    f"{int(m.numel())}")
            idx = torch.nonzero(m, as_tuple=False).flatten().to(torch.long)
        else:
            # ``torch.as_tensor([])`` is float32, but an empty split is a
            # legitimate assignment (the docstring above allows it).  Only
            # non-empty float arrays are a truncation hazard.
            if t.numel() == 0:
                return torch.empty(0, dtype=torch.long)
            if t.is_floating_point() or t.is_complex():
                raise ValueError(
                    "split indices must be integers or a boolean mask, got "
                    f"dtype {t.dtype}")
            idx = t.reshape(-1).to(torch.long)
    if idx.numel() and (bool((idx < 0).any()) or bool((idx >= n).any())):
        raise ValueError(
            f"split indices must lie in [0, {n}), got range "
            f"[{int(idx.min())}, {int(idx.max())}]")
    return idx


def _split_container(splits):
    """The split assignment must be a name -> indices mapping.

    A list/tuple of pairs, a bare index container or a string is iterable
    but cannot name the blocks the audit reasons about; translating it to
    the documented ValueError avoids an incidental ``AttributeError`` from
    the ``.items()`` calls below.
    """
    if not isinstance(splits, Mapping):
        raise ValueError(
            "splits must be a mapping of split name to indices, got "
            f"{type(splits).__name__}")
    return splits


def assert_split_partition(n_samples: int, splits, require_all: bool = True) -> None:
    """Raise unless the split indices are disjoint (and optionally complete).

    Disjointness is the sample-level counterpart of the group/time
    contract: if the same row appears in two splits, that row is both
    trained on and evaluated, regardless of how the split was produced.
    ``require_all=True`` additionally requires every sample to be assigned
    exactly once, so a typo in a custom split cannot silently drop samples
    from the evaluated population.  Empty splits are allowed here; the
    split constructors enforce non-emptiness themselves.
    """
    if isinstance(n_samples, bool):
        raise ValueError("n_samples must be an integer, not bool")
    try:
        n = operator.index(n_samples)
    except TypeError as exc:
        raise ValueError(
            f"n_samples must be an integer, got {n_samples!r}") from exc
    if n < 1:
        raise ValueError(f"n_samples must be positive, got {n_samples!r}")
    splits = _split_container(splits)
    seen = torch.zeros(n, dtype=torch.bool)
    for name, indices in splits.items():
        idx = _as_indices(indices, n)
        if idx.numel() == 0:
            continue
        if idx.numel() != idx.unique().numel():
            raise ValueError(
                f"split {name!r} contains repeated row indices; an index "
                f"split is a set and duplicates would silently over-weight "
                f"those rows")
        if bool(seen[idx].any()):
            raise ValueError(
                f"split {name!r} repeats a sample already assigned to "
                f"another split: splits must be disjoint")
        seen[idx] = True
    if require_all and not bool(seen.all()):
        missing = int((~seen).sum())
        raise ValueError(
            f"split assignment leaves {missing} of {n} samples unassigned")


def assert_group_disjoint(groups, splits) -> None:
    """Raise if any group id occurs in more than one split."""
    g = _group_codes(groups)
    assert_split_partition(int(g.numel()), splits, require_all=False)
    seen: dict = {}
    for name, indices in splits.items():
        idx = _as_indices(indices, int(g.numel()))
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
    try:
        raw = torch.as_tensor(times)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            "time values must be numeric timestamps, got a non-numeric "
            "container") from exc
    if raw.dim() != 1:
        raise ValueError(
            "time values must be one-dimensional (one timestamp per "
            f"sample), got shape {tuple(raw.shape)}; flattening a matrix "
            f"would audit the wrong samples")
    raw = raw.reshape(-1)
    if raw.is_complex() or raw.dtype == torch.bool:
        raise ValueError(
            f"time values must be real numeric timestamps, got dtype "
            f"{raw.dtype}")
    # Keep integer timestamps in their integer dtype: nanosecond Unix
    # timestamps exceed 2**53, so casting int64 to float64 collapses
    # distinct instants and can silently accept a chronological violation.
    # Floating timestamps are widened for the comparison as before.
    t = raw.to(torch.float64) if raw.is_floating_point() else raw
    if not bool(torch.isfinite(t).all()):
        raise ValueError("time values must be finite (no NaN/Inf): a "
                         "non-finite timestamp makes the ordering undefined")
    splits = _split_container(splits)
    unknown = [name for name in splits
               if name not in ("train", "val", "test")]
    if unknown:
        # Silently skipping an unrecognised name would let a misspelled
        # ``valid``/``holdout`` block escape the chronology check entirely
        # while still looking like a valid split container.
        raise ValueError(
            f"unknown split name(s) {unknown!r}: time-ordered splits must "
            f"use the canonical train/val/test names so every block is "
            f"checked in past -> future order")
    assert_split_partition(int(t.numel()), splits, require_all=False)
    spans = []
    for name in ("train", "val", "test"):
        if name not in splits:
            continue
        idx = _as_indices(splits[name], int(t.numel()))
        if idx.numel() == 0:
            continue
        # ``item()`` keeps integer timestamps exact (a Python int); the
        # comparison below never routes them through float64.
        spans.append((t[idx].min().item(), t[idx].max().item(), name))
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


def check_data_geometry(n_pairs, vocab, max_len=None) -> None:
    """Refuse a data configuration that cannot be realized.

    The generator draws ``n_pairs`` distinct keys without replacement from
    the ``vocab - 2`` selectable ids (0=PAD and 1=SEP are reserved), so
    ``1 <= n_pairs <= vocab - 2`` is required.  The sequence is
    ``[k1 v1 ... kn vn SEP query]`` and needs ``2*n_pairs + 2`` positions.
    Checking this at construction turns a shape-dependent ``topk`` failure
    into an explicit configuration error instead of aborting mid-run.
    """
    def _geometry_int(value, name):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, not bool")
        try:
            return operator.index(value)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be an integer, got {value!r}") from exc

    n_pairs = _geometry_int(n_pairs, "n_pairs")
    vocab = _geometry_int(vocab, "vocab")
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    if vocab < 4:
        raise ValueError(
            f"vocab must be >= 4 (two reserved ids plus at least two "
            f"selectable ids), got {vocab}")
    if n_pairs > vocab - 2:
        raise ValueError(
            f"n_pairs={n_pairs} needs {n_pairs} distinct key ids but only "
            f"{vocab - 2} selectable ids exist (vocab={vocab})")
    if max_len is not None:
        max_len = _geometry_int(max_len, "max_len")
        needed = 2 * n_pairs + 2
        if max_len < needed:
            raise ValueError(
                f"max_len={max_len} cannot hold the sequence: "
                f"2*n_pairs + 2 = {needed} positions are required")


def make_batch(batch: int, n_pairs: int, vocab: int, device, dense_lm: bool = False, gen: torch.Generator | None = None):
    """Returns idx, targets, eval_meta.

    eval_meta: dict with positions needed for false-concentration analysis:
      query_pos  (B,) index of qi in the sequence
      key_pos    (B,) index of the matching ki
      answer     (B,) correct value token
    """
    canonical = {}
    for name, value in (("batch", batch), ("n_pairs", n_pairs),
                        ("vocab", vocab)):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        try:
            canonical[name] = operator.index(value)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be an integer, got {value!r}") from exc
    batch, n_pairs, vocab = (canonical["batch"], canonical["n_pairs"],
                             canonical["vocab"])
    if batch < 0:
        raise ValueError(f"batch must be non-negative, got {batch}")
    check_data_geometry(n_pairs, vocab)
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


def association_codes(idx, meta, vocab) -> dict:
    """Per-row exact ``(query key, answer value)`` association of a batch.

    The split audit's exact-row fingerprints answer "did the generator draw
    the same *sequence* twice?".  In a synthetic task that question is
    almost always false, because the surrounding pairs differ, yet the same
    *key -> value association* can still recur across the training and
    held-out streams.  A model can memorise that association instead of
    reading the value from the context, so this helper exposes the unit the
    exposure audit needs: the query key, its paired value and a compact
    integer code ``key * vocab + value``.

    The batch is checked against the same task contract as the cleaning
    gate: ``meta`` must carry ``key_pos`` (an even column index into the
    query's key/value pair) and ``answer``; the final query token must equal
    the key at ``key_pos`` and the value at ``key_pos + 1`` must equal the
    recorded answer.  A malformed or mismatched batch raises ``ValueError``
    instead of contributing a fabricated association to the audit.
    """
    vocab = _batch_vocab_int(vocab)
    if vocab * vocab > (1 << 63) - 1:
        raise ValueError(
            "vocab is too large for an int64 association code "
            f"(vocab={vocab}); the key*value encoding would overflow")
    if not _is_integer_tensor(idx) or idx.dim() != 2:
        raise ValueError(
            f"idx must be a (batch, T) integer tensor, got {type(idx)}")
    if not isinstance(meta, Mapping) or "key_pos" not in meta \
            or "answer" not in meta:
        raise ValueError(
            "association codes need meta with 'key_pos' and 'answer'")
    batch, width = idx.shape
    key_pos = meta["key_pos"]
    answer = meta["answer"]
    if not _is_integer_tensor(key_pos) or not _is_integer_tensor(answer):
        raise ValueError(
            "meta['key_pos'] and meta['answer'] must have an integer dtype")
    key_pos = key_pos.to(device=idx.device, dtype=torch.long).reshape(-1)
    answer = answer.to(device=idx.device, dtype=torch.long).reshape(-1)
    if key_pos.shape != (batch,) or answer.shape != (batch,):
        raise ValueError(
            "meta['key_pos'] and meta['answer'] must carry one entry per row")
    if batch:
        if bool((key_pos < 0).any()) or bool((key_pos >= width - 1).any()):
            raise ValueError(
                "meta['key_pos'] is outside the key/value columns")
        if bool(key_pos.remainder(2).ne(0).any()):
            raise ValueError("meta['key_pos'] must be an even key position")
        if bool((answer < 2).any()) or bool((answer >= vocab).any()):
            raise ValueError(
                f"meta['answer'] must be a selectable id in [2, {vocab})")
    key = idx.gather(1, key_pos.unsqueeze(1)).squeeze(1)
    value = idx.gather(1, (key_pos + 1).unsqueeze(1)).squeeze(1)
    if batch and not bool(key.eq(idx[:, -1]).all()):
        raise ValueError(
            "the final query token does not match meta['key_pos']; the "
            "association row would describe a different pair")
    if batch and not bool(value.eq(answer).all()):
        raise ValueError(
            "meta['answer'] does not match the value at meta['key_pos']+1")
    if batch and (bool((key < 2).any()) or bool((key >= vocab).any())):
        raise ValueError(
            f"the paired key must be a selectable id in [2, {vocab})")
    code = key.mul(vocab).add(value)
    return {"key": key, "value": value, "code": code}


def association_reference_codes(codes) -> frozenset:
    """Normalize a training-stream association-code reference.

    ``eval_metrics`` scores each evaluated item against the exact pairs a
    training stream realized.  The reference arrives from the audit as a
    Python set, but it may cross a process boundary as another iterable or
    as JSON-style integral floats.  Accept only non-negative integers (bool
    and fractional values are rejected rather than silently coerced to a
    different pair code) and return an immutable set so the readout cannot
    be mutated between batches.
    """
    if isinstance(codes, (str, bytes)) or isinstance(codes, Mapping):
        raise ValueError(
            "association reference must be an iterable of integer codes, "
            f"got {type(codes).__name__}")
    try:
        iterator = iter(codes)
    except TypeError as exc:
        raise ValueError(
            "association reference must be an iterable of integer codes, "
            f"got {type(codes).__name__}") from exc
    out = set()
    for code in iterator:
        if isinstance(code, bool):
            raise ValueError(
                f"association codes must be integers, got {code!r}")
        if isinstance(code, float):
            # JSON round-trips may spell an integral code as 2.0; accept
            # those but never truncate a genuinely fractional one.
            if not math.isfinite(code) or not code.is_integer():
                raise ValueError(
                    f"association codes must be integers, got {code!r}")
            code = int(code)
        try:
            value = operator.index(code)
        except TypeError as exc:
            raise ValueError(
                f"association codes must be integers, got {code!r}") from exc
        if value < 0:
            raise ValueError(
                f"association codes must be non-negative, got {code!r}")
        out.add(int(value))
    return frozenset(out)


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


def _batch_vocab_int(vocab) -> int:
    """Validate and canonicalise the ``vocab`` scalar of a batch gate.

    A float vocab would make the ``idx < vocab`` comparison accept a batch
    whose ids are not drawn from a discrete vocabulary, and a bool is not a
    vocabulary size.  The generator and ``check_data_geometry`` already
    refuse both, so the cleaning gate must not be looser than the producer
    it is checking.
    """
    if isinstance(vocab, bool):
        raise ValueError(f"vocab must be an integer, not bool")
    try:
        value = operator.index(vocab)
    except TypeError as exc:
        raise ValueError(f"vocab must be an integer, got {vocab!r}") from exc
    if value < 1:
        raise ValueError(f"vocab must be >= 1, got {value}")
    return int(value)


def validate_batch(idx, targets, meta, vocab, diagnostics: bool = True) -> dict:
    """Data-cleaning gate for one freshly generated batch.

    Raises ``ValueError`` if any invariant is violated:

    * integrity: 2-D integer ids, no NaN/Inf, all token ids in ``[0, vocab)``;
    * structure: exactly one SEP token, at the fixed separator slot, a
      query at the final position, and ``meta`` agreeing with the
      sequence;
    * label noise: all keys are distinct (the generator samples them
      without replacement), the query key occurs exactly once among the
      keys, the recorded answer is the value paired with that key (a
      repeated key would make the correct answer ambiguous) and the answer
      id is a selectable token (ids 0=PAD and 1=SEP are reserved and never
      labels);
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
    vocab = _batch_vocab_int(vocab)
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
        # The task layout is ``[k1 v1 ... kn vn SEP query]``: an odd width
        # cannot pair every key with a value, and width < 4 cannot hold one
        # pair plus SEP/query.  Reject before the key/value gathers can
        # read a "value" that is really the next key or the SEP slot.
        if width < 4 or width % 2:
            return False
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
        # 0=PAD and 1=SEP are reserved: SEP appears only at the fixed
        # separator slot, so every other position (keys, values, query) must
        # be a selectable content id in [2, vocab).  Without this a PAD
        # query/key would pass the label checks and enter training/eval as a
        # token the generator never emits.
        checks.append((idx[:, :-2] >= 2).all())
        checks.append((idx[:, -1] >= 2).all())
        # Key columns only: ``idx`` is ``[k1 v1 ... kn vn SEP query]``, so
        # the last two columns (SEP at 2n, query at 2n+1) must be excluded.
        keys = idx[:, :-2:2]
        # Every key must be distinct, not just the queried one: the task
        # samples keys without replacement, and a repeated non-query key
        # means the batch violates the generator contract even when the
        # queried answer happens to stay unique.
        if keys.size(1) > 1:
            ordered = keys.sort(dim=1).values
            checks.append((ordered[:, 1:] != ordered[:, :-1]).all())
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
    if width < 4 or width % 2:
        raise ValueError(
            "structure: the sequence must be [k v]*n + SEP + query, so its "
            f"width must be even and at least 4, got {width}")
    # ``validate_batch`` promises ValueError for every invariant violation.
    # A missing / non-mapping metadata object used to escape as KeyError or
    # TypeError from the raw subscript below, which is neither documented nor
    # catchable by callers that handle the cleaning gate's ValueError.
    if not isinstance(meta, Mapping):
        raise ValueError(
            f"meta must be a mapping with 'query_pos' and 'answer', got "
            f"{type(meta).__name__}")
    missing = [key for key in ("query_pos", "answer") if key not in meta]
    if missing:
        raise ValueError(
            f"meta is missing required key(s) {missing}; got keys "
            f"{sorted(map(str, meta))}")
    if not _is_integer_tensor(meta["query_pos"]):
        raise ValueError("meta['query_pos'] must have an integer dtype")
    qp = meta["query_pos"].to(device=idx.device, dtype=torch.long)
    if qp.shape != (batch_size,) or bool(qp.lt(0).any()) or bool(qp.ge(width).any()):
        raise ValueError("query_pos is missing or out of range")
    if not bool(qp.eq(width - 1).all()):
        raise ValueError(
            "structure: query_pos must be the final sequence position")
    # 0=PAD and 1=SEP are reserved: SEP appears only at the fixed separator
    # slot, so every key/value position and the query must be a selectable
    # content id in [2, vocab).  The generator never emits PAD; accepting it
    # here would let a malformed batch bypass the cleaning gate and be
    # trained/evaluated as if it were valid task data.
    if not bool((idx[:, :-2] >= 2).all()) or not bool((idx[:, -1] >= 2).all()):
        raise ValueError(
            "structure: ids 0=PAD and 1=SEP are reserved; all key/value "
            f"positions and the query must be selectable ids in [2, {vocab})"
        )
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
    if keys.size(1) > 1:
        ordered = keys.sort(dim=1).values
        if not bool((ordered[:, 1:] != ordered[:, :-1]).all()):
            raise ValueError(
                "structure: keys must be distinct within a sequence (the "
                "generator samples them without replacement); a repeated "
                "key violates the task contract"
            )
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


def _metadata_signature(meta) -> str:
    """Stable, content-level representation of one batch's metadata.

    ``batch_fingerprint`` deliberately hashes only the context, targets and
    answers (the tensors model fitting depends on).  The evaluation-stream
    digest also needs to catch a corrupted ``query_pos`` / ``key_pos`` or
    an extra metadata field, so this helper serialises *all* metadata
    entries in sorted-key order.  Tensors are converted to plain Python
    values first; non-tensor entries are included via ``repr``.
    """
    if not isinstance(meta, Mapping):
        raise ValueError(
            f"batch meta must be a mapping, got {type(meta).__name__}")
    parts = []
    for key in sorted(meta, key=str):
        value = meta[key]
        if torch.is_tensor(value):
            parts.append((str(key), value.detach().cpu().tolist()))
        else:
            parts.append((str(key), repr(value)))
    return repr(parts)


def stream_content_digest(batches) -> dict:
    """Order-sensitive digest of a *complete* realized data stream.

    ``stream_fingerprint`` versions a stream by its first batch, which is
    the right cheap summary for the generator protocol.  Evaluation and
    training, however, consume a whole sequence of batches, and a change
    that only affects a later batch (an accidental re-seed, a different
    ``eval_batches`` loop, or a corrupted metadata field) would leave that
    prefix digest unchanged.  This helper hashes every batch in consumption
    order, including its shape and the *full* metadata signature, so two
    runs can verify that their per-item correctness masks (``McNemar`` /
    fold companions) were computed on byte-identical data.

    ``batches`` is an iterable of ``(idx, targets, meta)`` triples, exactly
    as returned by :func:`make_batch`.  Each batch must be a 2-D integer
    context with an equally shaped integer target tensor and a mapping
    metadata object carrying a 1-D ``answer`` tensor; optional position
    tensors are length-checked too.  Invalid input raises ``ValueError``
    rather than versioning an unvalidated draw.  An empty iterable has a
    well-defined digest (of nothing) with ``n_batches == 0``; callers that
    require data check that count before using the digest.
    """
    h = hashlib.sha256()
    n_batches = 0
    n_rows = 0
    try:
        stream = iter(batches)
    except TypeError as exc:
        raise ValueError(
            "batches must be an iterable of (idx, targets, meta) triples, "
            f"got {type(batches).__name__}") from exc
    for batch in stream:
        if not isinstance(batch, (tuple, list)) or len(batch) != 3:
            raise ValueError(
                "each stream batch must be an (idx, targets, meta) triple")
        idx, tgt, meta = batch
        if not _is_integer_tensor(idx) or idx.dim() != 2:
            raise ValueError(
                "stream batch idx must be a 2-D integer tensor")
        batch_size, _ = idx.shape
        if not _is_integer_tensor(tgt) or tgt.shape != idx.shape:
            raise ValueError(
                "stream batch targets must be an integer tensor with the "
                "same shape as idx")
        if not isinstance(meta, Mapping) or "answer" not in meta:
            raise ValueError(
                "stream batch meta must be a mapping with an 'answer' entry")
        answer = meta["answer"]
        if not _is_integer_tensor(answer) or answer.shape != (batch_size,):
            raise ValueError(
                "stream batch answer must be a 1-D integer tensor with one "
                "entry per row")
        for name in ("query_pos", "key_pos"):
            if name in meta:
                value = meta[name]
                if not _is_integer_tensor(value) or value.shape != (batch_size,):
                    raise ValueError(
                        f"stream batch {name} must be a 1-D integer tensor "
                        f"with one entry per row")
        shape = (int(batch_size), int(idx.size(1)))
        h.update(
            (f"{n_batches}:{shape}:{batch_fingerprint(idx, tgt, meta)}:"
             f"{_metadata_signature(meta)};").encode("utf-8"))
        n_batches += 1
        n_rows += int(batch_size)
    return {"sha256": h.hexdigest()[:16], "n_batches": int(n_batches),
            "n_rows": int(n_rows)}


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

    ``seed`` must be an integer (``stream_seed_key`` still canonicalises the
    low 32 bits, so an aliased integer seed hashes identically); a
    fractional seed would otherwise be silently truncated to a different
    stream before hashing.
    """
    raw_seed = seed
    if isinstance(raw_seed, bool):
        raise ValueError(f"seed must be an integer, got {raw_seed!r}")
    try:
        seed = operator.index(raw_seed)
    except TypeError as exc:
        raise ValueError(
            f"seed must be an integer, got {raw_seed!r}") from exc
    key = stream_seed_key(seed)
    batch = _positive_audit_count(batch, "batch")
    check_data_geometry(n_pairs, vocab)
    gen = torch.Generator(device=device).manual_seed(key)
    idx, tgt, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
    # A dataset-version hash must summarise a *valid* realization: hashing
    # an empty (batch=0) or structurally invalid draw would version a
    # dataset the training/evaluation cleaning gate would reject.
    validate_batch(idx, tgt, meta, vocab, diagnostics=False)
    geometry = (f"n_pairs={int(n_pairs)};vocab={int(vocab)};"
                f"batch={int(batch)};seed={key};")
    digest = hashlib.sha256(
        (geometry + batch_fingerprint(idx, tgt, meta)).encode("ascii")
    ).hexdigest()[:16]
    return {"seed": int(seed), "stream_key": key, "batch": int(batch),
            "n_pairs": int(n_pairs), "vocab": int(vocab), "sha256": digest}


def _positive_audit_count(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be an integer, got {value!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _draw_split(n_pairs, vocab, device, batch, batches, seed):
    """Draw one split's samples and run the cleaning gate on every batch.

    Returns the concatenated contexts, the supervised answers and the
    per-batch ``validate_batch`` diagnostics.  Nothing is fitted here; the
    caller decides which split plays the reference role.
    """
    batch = _positive_audit_count(batch, "batch")
    batches = _positive_audit_count(batches, "batches")
    gen = torch.Generator(device=device).manual_seed(stream_seed_key(seed))
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

    Returns ``(payload, n_rows)`` where ``payload`` holds the exact-row
    fingerprints (``row_keys``), the per-item query associations in draw
    order (``associations``), their unique set (``association_set``) and the
    set of query keys seen in the stream (``key_set``).  The association
    lists are what lets the audit report memorisation exposure, not only
    duplicate *sequences*.
    """
    gen = torch.Generator(device=device).manual_seed(stream_seed_key(seed))
    batch = _positive_audit_count(batch, "batch")
    units = _positive_audit_count(units, "units")
    keys, n_rows = set(), 0
    associations, association_set, key_set = [], set(), set()
    for _ in range(units):
        idx, tgt, meta = make_batch(batch, n_pairs, vocab, device,
                                    gen=gen)
        validate_batch(idx, tgt, meta, vocab, diagnostics=False)
        batch_keys = _row_keys(idx)
        keys.update(batch_keys)
        codes = association_codes(
            idx, meta, vocab)["code"].tolist()
        for code in codes:
            code = int(code)
            associations.append(code)
            association_set.add(code)
            key_set.add(code // int(vocab))
        n_rows += len(batch_keys)
    return {
        "row_keys": keys,
        "associations": associations,
        "association_set": association_set,
        "key_set": key_set,
    }, n_rows


def _audit_stream_name(name, kind):
    """Reject names the flat pair-key format cannot represent unambiguously."""
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"{kind} stream names must be non-empty strings, got {name!r}")
    if "|" in name:
        raise ValueError(
            f"{kind} stream name {name!r} may not contain '|': the audit "
            "encodes pair keys as '<name>|<name>', so a delimiter in a "
            "name could make two different pairs share one key and hide "
            "a cross-split overlap")
    return name


def actual_stream_overlap_audit(n_pairs, vocab, device="cpu",
                                train_specs=None, heldout_specs=None,
                                include_train_association_codes=False):
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

    Content overlap alone is not sufficient for this task: a split is a
    seed-owned stream, and ``make_batch``'s draws are shape-dependent, so
    the *same* seed read at two different batch sizes can produce two
    disjoint realizations.  Reusing a training seed as a held-out stream
    at a different geometry would therefore show zero duplicate rows while
    still training and evaluating on the same underlying stream.  The
    audit separately compares the canonical 32-bit seed keys of every
    stream pair and reports ``cross_group_seed_collisions``; any such
    collision across groups is leakage even when the content matrices are
    empty.

    **Association exposure.** Exact-row duplicates are the wrong leakage
    unit for a synthetic sequence task: two draws almost never repeat the
    whole sequence, but the same query ``(key, value)`` association can
    recur across splits, and a model can memorise it.  The replay therefore
    also records every item's association code (see
    :func:`association_codes`).  For each non-training group it reports the
    item-weighted share whose exact pair was observed in a training stream
    (``pair_seen_train_rate``) and the share whose query key was observed
    with any value (``key_seen_train_rate``).  This is a *disclosure*, not
    a leakage verdict: a shared vocabulary makes some association overlap
    expected, while only cross-group *row* overlap and seed reuse are hard
    failures.  Pass ``include_train_association_codes=True`` to also return
    the training code set (``train_association_codes``, a Python set that
    is not JSON-serializable) for exposure-stratified evaluation; callers
    must pop it before persisting the audit.

    It returns per-stream row/duplicate counts, the pairwise overlap matrix
    over row-content fingerprints, and the association-exposure readout.
    Any cross-group row overlap or seed-key collision is leakage; the
    caller decides how to gate on it.
    """
    def _spec_mapping(value, label):
        """Normalise a stream-spec container, rejecting non-mappings.

        ``bytes``/strings are iterable but are not spec mappings (a byte
        string would otherwise fail deep inside ``dict()`` with an
        incidental TypeError), and a bare integer is not iterable at all.
        The audit's input contract is a mapping, so translate every such
        malformed container into a ValueError like the stream-name and
        seed checks do.
        """
        if value is None:
            return {}
        if isinstance(value, (str, bytes)):
            raise ValueError(
                f"{label} must be a mapping of stream name to spec, got "
                f"{type(value).__name__}")
        try:
            return dict(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} must be a mapping of stream name to spec, got "
                f"{type(value).__name__}") from exc

    train_specs = _spec_mapping(train_specs, "train_specs")
    heldout_specs = _spec_mapping(heldout_specs, "heldout_specs")
    if not train_specs or not heldout_specs:
        raise ValueError(
            "actual_stream_overlap_audit needs at least one training and "
            "one held-out stream spec")
    for name in train_specs:
        _audit_stream_name(name, "training")
    for name in heldout_specs:
        _audit_stream_name(name, "held-out")
    shared_names = sorted(set(train_specs) & set(heldout_specs), key=str)
    if shared_names:
        raise ValueError(
            "stream names must be unique across train_specs and "
            f"heldout_specs; duplicated: {shared_names}")
    key_sets, streams, groups, seed_keys = {}, {}, {}, {}
    associations, association_sets, query_key_sets = {}, {}, {}
    all_specs = ([(name, spec, "train")
                  for name, spec in train_specs.items()]
                 + [(name, spec, "heldout")
                    for name, spec in heldout_specs.items()])
    for name, spec, kind in all_specs:
        unit_key = "steps" if kind == "train" else "batches"
        if not isinstance(spec, Mapping):
            raise ValueError(
                f"{kind} spec {name!r} must be a mapping with 'seed', "
                f"'batch' and {unit_key!r}, got {type(spec).__name__}")
        if spec.get("seed") is None:
            raise ValueError(f"{kind} spec {name!r} needs 'seed'")
        if spec.get("batch") is None:
            raise ValueError(f"{kind} spec {name!r} needs 'batch'")
        if spec.get(unit_key) is None:
            raise ValueError(
                f"{kind} spec {name!r} needs {unit_key!r}")
        raw_seed = spec["seed"]
        if isinstance(raw_seed, bool):
            raise ValueError(
                f"{kind} spec {name!r} seed must be an integer, "
                f"got {raw_seed!r}")
        try:
            seed = operator.index(raw_seed)
        except TypeError as exc:
            raise ValueError(
                f"{kind} spec {name!r} seed must be an integer, "
                f"got {raw_seed!r}") from exc
        batch = _positive_audit_count(
            spec["batch"], f"{kind} spec {name!r} batch")
        units = _positive_audit_count(
            spec[unit_key], f"{kind} spec {name!r} {unit_key}")
        seed_key = stream_seed_key(seed)
        replay, n_rows = _draw_stream_keys(
            seed, n_pairs, vocab, device, batch, units)
        keys = replay["row_keys"]
        associations[name] = replay["associations"]
        association_sets[name] = replay["association_set"]
        query_key_sets[name] = replay["key_set"]
        key_sets[name] = keys
        seed_keys[name] = seed_key
        groups[name] = str(spec.get(
            "group", "train" if kind == "train" else name))
        streams[name] = {
            "kind": kind, "group": groups[name],
            "seed": seed, "stream_key": seed_key,
            "batch": batch, "units": units,
            "n_rows": int(n_rows), "n_unique_rows": int(len(keys)),
            "n_duplicate_rows": int(n_rows - len(keys)),
            "n_unique_associations": int(len(association_sets[name])),
        }

    names = list(key_sets)
    pairwise = {}
    pairwise_seed_collision = {}
    cross_group = []
    seed_collisions = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = len(key_sets[a] & key_sets[b])
            pair = f"{a}|{b}"
            pairwise[pair] = overlap
            collision = seed_keys[a] == seed_keys[b]
            pairwise_seed_collision[pair] = collision
            if groups[a] != groups[b]:
                cross_group.append(overlap)
                if collision:
                    seed_collisions.append(pair)
    train_names = list(train_specs)
    heldout_names = list(heldout_specs)
    train_vs_heldout = [
        pairwise[f"{a}|{b}"] for a in train_names for b in heldout_names]
    heldout_vs_heldout = [
        pairwise[f"{a}|{b}"] for i, a in enumerate(heldout_names)
        for b in heldout_names[i + 1:]]

    # Association exposure: the training groups' exact query pairs form the
    # reference; held-out items are only *scored* against it (nothing is
    # fitted on held-out data).  Item-weighted rates are reported because
    # the question is "what share of the evaluated items could be answered
    # from a memorised association?", not "how many distinct pair types
    # recur".  A split that is itself in a training group (e.g. the
    # validation stream re-read at a second geometry) is not held-out and
    # is left out of this readout.
    train_codes, train_keys_seen = set(), set()
    for name in names:
        if groups[name] == "train":
            train_codes.update(association_sets[name])
            train_keys_seen.update(query_key_sets[name])
    vocab_int = int(vocab)
    exposure = {}
    for name in names:
        if groups[name] == "train":
            continue
        codes = associations[name]
        n = len(codes)
        pair_seen = sum(1 for code in codes if code in train_codes)
        key_seen = sum(
            1 for code in codes if code // vocab_int in train_keys_seen)
        unique = set(codes)
        pair_seen_unique = len(unique & train_codes)
        exposure[name] = {
            "group": groups[name],
            "n_items": int(n),
            "n_unique_pairs": int(len(unique)),
            "n_unique_keys": int(len({code // vocab_int for code in codes})),
            "pair_seen_train": int(pair_seen),
            "pair_seen_train_rate": (pair_seen / n if n else float("nan")),
            "key_seen_train": int(key_seen),
            "key_seen_train_rate": (key_seen / n if n else float("nan")),
            # Seen key, unseen exact pair: the item's key occurred in
            # training but not with this value, so a memorised key -> value
            # lookup cannot directly supply the answer.  This is the cohort
            # where in-context retrieval is required.
            "pair_unseen_key_seen": int(key_seen - pair_seen),
            "pair_unseen_key_seen_rate": (
                (key_seen - pair_seen) / n if n else float("nan")),
            "pair_seen_unique": int(pair_seen_unique),
            "pair_seen_unique_rate": (
                pair_seen_unique / len(unique) if unique else float("nan")),
        }
    train_summary = {
        "n_items": int(sum(len(associations[name]) for name in names
                          if groups[name] == "train")),
        "n_unique_pairs": int(len(train_codes)),
        "n_unique_keys": int(len(train_keys_seen)),
        "n_possible_pairs": int((vocab_int - 2) ** 2),
        "n_possible_keys": int(vocab_int - 2),
        "pair_space_coverage": (
            len(train_codes) / ((vocab_int - 2) ** 2)
            if vocab_int > 2 else float("nan")),
    }
    max_exposure = max(
        (row["pair_seen_train_rate"] for row in exposure.values()
         if math.isfinite(row["pair_seen_train_rate"])), default=float("nan"))
    out = {
        "train_specs": {k: dict(v) for k, v in train_specs.items()},
        "heldout_specs": {k: dict(v) for k, v in heldout_specs.items()},
        "streams": streams,
        "pairwise_overlap": pairwise,
        "pairwise_seed_collision": pairwise_seed_collision,
        "cross_group_seed_collisions": seed_collisions,
        "train_vs_heldout_max": max(train_vs_heldout, default=0),
        "heldout_vs_heldout_max": max(heldout_vs_heldout, default=0),
        "max_overlap": max(cross_group, default=0),
        "n_rows_total": int(sum(s["n_rows"] for s in streams.values())),
        "train_association_summary": train_summary,
        "association_exposure": exposure,
        "max_heldout_pair_exposure": float(max_exposure),
        "association_exposure_note": (
            "item-weighted share of held-out query (key, value) pairs whose "
            "exact association was observed in a training-group stream. "
            "This is a memorisation-opportunity disclosure, not a leakage "
            "verdict: exact-pair exposure is expected with a shared "
            "vocabulary, while cross-group whole-row overlap and seed reuse "
            "remain hard failures.  Read clean and counterfactual accuracy "
            "separately on the exposure cohorts before attributing an "
            "accuracy gain to in-context retrieval."),
    }
    if include_train_association_codes:
        # Not JSON-serializable on purpose: the caller must pop it before
        # persisting the audit (the experiment layer does so when it hands
        # the set to exposure-stratified evaluation).
        out["train_association_codes"] = train_codes
    return out


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
                 "value_repeats", "naive_confusable", "difficulty_joint")
_AUDIT_DERIVED_FIELDS = ("key_pos", "value_repeats", "naive_confusable")

def _audit_seed(value) -> int:
    """Validate a Monte Carlo audit seed instead of silently truncating it.

    ``torch.Generator.manual_seed`` raises an incidental ``RuntimeError`` for
    bool/fractional seeds, and an ``int(seed)`` coercion would alias e.g.
    ``0.5`` to stream 0.  Audit p-values are part of the reproducibility
    contract, so every non-integral seed is translated into ``ValueError``
    before a replicate is drawn.
    """
    if isinstance(value, bool):
        raise ValueError(f"audit seed must be an integer, got {value!r}")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(
            f"audit seed must be an integer, got {value!r}") from exc


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


# Ambiguity strata used by :func:`difficulty_strata`.  The encoding is a
# two-bit code: bit 0 = naive-confusable, bit 1 = repeated answer value, so
# every id in [0, 4) names one combination and the id is stable across
# audits/records.
_AMBIGUITY_LABELS = ("none", "naive_confusable", "value_repeats", "both")

# Relative retrieval-position bins used by the default difficulty stratum.
# Three bins keep every cell populated at the audit/evaluation sizes while
# still separating early / middle / late retrieval, and the same id means
# the same *relative* position at every sequence length.
DEFAULT_DIFFICULTY_KEY_BINS = 3


def _difficulty_int(value, name):
    """Reject bool/fractional metadata instead of silently truncating it."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(
            f"{name} must be an integer, got {value!r}") from exc


def difficulty_features(idx, answer, n_pairs):
    """Per-sample structural difficulty covariates of one batch.

    Derived from the validated ``[k1 v1 ... kn vn SEP query]`` layout (the
    query is the last column of ``idx``; the answer is stored separately),
    so no fitted statistic is involved:

    * ``key_pos`` - pair index of the matching key (retrieval distance);
    * ``value_repeats`` - the answer value occurs more than once among the
      values, so a perfect key lookup does not uniquely determine it;
    * ``naive_confusable`` - the query token occurs after the matching key
      as a value, so the naive "last visible match wins" rule differs from
      the exact pair lookup.

    These are exactly the task-structural factors that make a length sweep
    non-comparable: with more pairs per sequence the chance that a value
    repeats or that the query token appears later as a value grows, so raw
    accuracy at a longer length mixes a length effect with a data-
    difficulty effect.  :func:`difficulty_strata` turns the covariates into
    a small categorical stratum that can be used for post-stratification
    (:func:`cage.metrics.standardized_proportion`).

    Mirrors ``cage.train._error_case_flags`` so the audit measures exactly
    the difficulty factors used to explain errors downstream.  The batch
    geometry is checked instead of inferred from slicing: a narrower
    ``idx`` would otherwise silently pair the wrong columns and label a
    malformed batch as easy/hard.
    """
    n_pairs = _difficulty_int(n_pairs, "n_pairs")
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    if not _is_integer_tensor(idx) or idx.dim() != 2:
        raise ValueError(
            "idx must be a (batch, 2*n_pairs + 2) integer tensor, got "
            f"{type(idx)}")
    if not _is_integer_tensor(answer) or answer.dim() != 1:
        raise ValueError("answer must be a 1-D integer tensor")
    batch, width = idx.shape
    expected = 2 * n_pairs + 2
    if width != expected:
        raise ValueError(
            f"idx width {width} does not match the [k v]*{n_pairs} + SEP + "
            f"query layout ({expected} positions)")
    if int(answer.numel()) != int(batch):
        raise ValueError(
            f"answer has {int(answer.numel())} entries for {int(batch)} "
            f"sequences")
    query = idx[:, -1]
    # ``meta`` tensors carry no values to read back (they are a shape-only
    # device), so they skip the value-level checks while preserving
    # placement.
    readable = idx.device.type != "meta"
    keys = idx[:, :2 * n_pairs:2]
    if readable and int(idx.size(0)):
        # These are the same structural/label invariants the cleaning gate
        # enforces.  Without them this helper would happily describe a batch
        # (reserved SEP/PAD ids in content slots, duplicate keys, a misplaced
        # SEP) that ``validate_batch`` rejects, and could count the SEP slot
        # as a "later value" of a reserved query token.
        if not bool((idx[:, :-2] >= 2).all()) \
                or not bool((idx[:, -1] >= 2).all()):
            raise ValueError(
                "ids 0=PAD and 1=SEP are reserved; all key/value positions "
                "and the query must be selectable ids >= 2")
        if not bool(idx.eq(SEP_ID).sum(dim=1).eq(1).all()) \
                or not bool(idx[:, -2].eq(SEP_ID).all()):
            raise ValueError(
                "structure: expected exactly one SEP token at the fixed "
                "separator slot (T-2)")
        if keys.size(1) > 1:
            ordered = keys.sort(dim=1).values
            if not bool((ordered[:, 1:] != ordered[:, :-1]).all()):
                raise ValueError(
                    "structure: keys must be distinct within a sequence")
    match = keys.eq(query.unsqueeze(1))
    # A batch with no matching key (or a duplicated key) has no defined
    # retrieval position; ``argmax`` would silently label it as an easy
    # early-key sample.  Refuse it instead, matching the geometry checks.
    if readable and int(idx.size(0)) and not bool(
            match.sum(dim=1).eq(1).all()):
        raise ValueError(
            "idx must contain exactly one key equal to the final query "
            "token in every sequence")
    key_pos = match.to(torch.int64).argmax(dim=1)
    values = idx[:, 1:2 * n_pairs:2]
    # ``answer`` is normally already on ``idx``'s device, but the helper is
    # public: align it rather than failing a valid CPU-metadata/CUDA-batch
    # call (or, worse, broadcasting the wrong side).
    answer = answer.to(device=idx.device)
    paired_value = values.gather(1, key_pos.unsqueeze(1)).squeeze(1)
    if readable and int(idx.size(0)) and not bool(
            paired_value.eq(answer).all()):
        raise ValueError(
            "answer does not match the value paired with the query key; "
            "difficulty features would describe a different label")
    value_repeats = (values.eq(answer.unsqueeze(1)).sum(dim=1)
                     > 1).to(torch.int64)
    # Position indices must live on the batch's device: the comparison
    # below is boolean-ANDed with a tensor derived from ``idx``.
    positions = torch.arange(idx.size(1), device=idx.device).view(1, -1)
    visible = positions < (idx.size(1) - 1)
    last_match = torch.where(idx.eq(query.unsqueeze(1)) & visible,
                             positions,
                             torch.full_like(positions, -1)).max(dim=1).values
    naive_confusable = last_match.ne(key_pos.mul(2)).to(torch.int64)
    return {"key_pos": key_pos, "value_repeats": value_repeats,
            "naive_confusable": naive_confusable}


def difficulty_stratum_labels(n_key_bins=DEFAULT_DIFFICULTY_KEY_BINS):
    """JSON-safe labels for :func:`difficulty_strata` ids.

    Ids are ``key_bin * 4 + ambiguity_code``; the accompanying bounds are
    in *pair-index* space (``key_bin`` is a relative retrieval-position
    tercile, so the same id is comparable across lengths even though the
    absolute distance differs).
    """
    n_key_bins = _difficulty_int(n_key_bins, "n_key_bins")
    if n_key_bins < 1:
        raise ValueError(f"n_key_bins must be >= 1, got {n_key_bins}")
    return [
        {"id": bin_idx * 4 + ambiguity,
         "key_bin": bin_idx,
         "ambiguity": _AMBIGUITY_LABELS[ambiguity]}
        for bin_idx in range(n_key_bins)
        for ambiguity in range(4)
    ]


def difficulty_strata(idx, answer, n_pairs,
                      n_key_bins=DEFAULT_DIFFICULTY_KEY_BINS):
    """Map each sample to a joint retrieval-position x ambiguity stratum.

    ``n_key_bins`` relative retrieval-position bins are crossed with the
    four ambiguity states of :data:`_AMBIGUITY_LABELS`, giving
    ``4 * n_key_bins`` strata.  A stratum is a *pre-treatment* covariate:
    it is fixed by the generated sequence, not by a model, so conditioning
    on it cannot remove the effect of the mechanism under test.  That is
    what makes it valid for post-stratifying accuracy across lengths:
    ``standardized_proportion(correct, n, reference_weights)`` reweights
    each length's per-stratum accuracy to the same reference mix instead
    of comparing compositions that differ by construction.

    Returns ``{"stratum", "key_bin", "ambiguity", "labels",
    "key_bin_bounds"}``.  ``stratum``/``key_bin``/``ambiguity`` are
    per-sample tensors; ``labels`` and ``key_bin_bounds`` describe the
    encoding for records (all JSON-safe).
    """
    n_key_bins = _difficulty_int(n_key_bins, "n_key_bins")
    if n_key_bins < 1:
        raise ValueError(f"n_key_bins must be >= 1, got {n_key_bins}")
    n_pairs = _difficulty_int(n_pairs, "n_pairs")
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    features = difficulty_features(idx, answer, n_pairs)
    key_bin = (features["key_pos"] * n_key_bins // n_pairs).clamp(
        max=n_key_bins - 1).to(torch.long)
    # bit 0 = naive-confusable, bit 1 = repeated value.  Ordering repeat
    # above naive matches the historical ``difficulty_joint`` encoding at
    # n_key_bins == n_pairs, so the two audit views cannot drift apart.
    ambiguity = (features["value_repeats"] * 2
                 + features["naive_confusable"]).to(torch.long)
    stratum = key_bin * 4 + ambiguity
    bounds = [[math.ceil(bin_idx * n_pairs / n_key_bins),
               math.ceil((bin_idx + 1) * n_pairs / n_key_bins)]
              for bin_idx in range(n_key_bins)]
    return {
        "stratum": stratum,
        "key_bin": key_bin,
        "ambiguity": ambiguity,
        "labels": difficulty_stratum_labels(n_key_bins),
        "key_bin_bounds": bounds,
        "n_key_bins": n_key_bins,
        "n_strata": 4 * n_key_bins,
    }


def _difficulty_features(split, n_pairs):
    """Backward-compatible wrapper around :func:`difficulty_features`."""
    return difficulty_features(split["idx"], split["answer"], n_pairs)


def _joint_difficulty_values(features):
    """Encode ``(key_pos, value_repeats, naive_confusable)`` as one category.

    The binary ambiguity flags are the two least-significant digits, so the
    encoding is dense (``0 .. 4*n_pairs-1``) and no joint cell is dropped.
    """
    return (((features["key_pos"].to(torch.long) * 2
              + features["value_repeats"].to(torch.long)) * 2)
            + features["naive_confusable"].to(torch.long))


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
    if field == "difficulty_joint":
        return _joint_difficulty_values(_difficulty_features(split, n_pairs))
    if field in _AUDIT_DERIVED_FIELDS:
        return _difficulty_features(split, n_pairs)[field]
    raise ValueError(f"unknown audit field {field!r}")


def _field_categories(field, vocab, n_pairs):
    """Number of possible categories for a categorical audit field."""
    if field == "key_pos":
        return max(int(n_pairs), 1)
    if field == "difficulty_joint":
        return max(int(n_pairs), 1) * 4
    if field in ("value_repeats", "naive_confusable"):
        return 2
    return int(vocab)


def _counts(values, n_categories):
    n_categories = int(n_categories)
    if n_categories < 0:
        raise ValueError(
            f"n_categories must be non-negative, got {n_categories}")
    try:
        raw = torch.as_tensor(values)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            "audit values must be a real numeric vector, got "
            f"{type(values).__name__}") from exc
    raw = raw.reshape(-1)
    # ``Tensor.to(long)`` silently discards the imaginary part (with only a
    # warning), which would fabricate a count the caller never supplied.
    if raw.is_complex():
        raise ValueError("audit values must be real integers")
    # A boolean is a flag, not a category id: casting it to long would
    # silently turn ``[True, False]`` into the categories 1 and 0 and
    # fabricate counts the caller never supplied.
    if raw.dtype == torch.bool:
        raise ValueError("audit values must be integers, not booleans")
    if raw.is_floating_point() and not bool(raw.eq(raw.round()).all()):
        raise ValueError("audit values must be integers")
    flat = raw.to(torch.long).cpu()
    if flat.numel() and (bool((flat < 0).any())
                         or bool((flat >= n_categories).any())):
        raise ValueError(
            f"audit values must lie in [0, {n_categories}), got range "
            f"[{int(flat.min())}, {int(flat.max())}]")
    return torch.bincount(flat, minlength=n_categories).to(torch.int64)


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
    if not stream_values:
        # No training stream to fit a reference floor from; the caller
        # should treat the floor as undefined rather than crash on an
        # empty list.
        return float("nan")
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
    def _as_count_vector(values, name):
        try:
            raw = torch.as_tensor(values)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError(
                f"{name} counts must be a real numeric vector, got "
                f"{type(values).__name__}") from exc
        if raw.is_complex():
            raise ValueError(f"{name} counts must be real integers")
        if raw.dtype == torch.bool:
            raise ValueError(
                f"{name} counts must be integers, not booleans")
        try:
            return raw.detach().cpu().reshape(-1).to(torch.float64)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError(
                f"{name} counts must be a real numeric vector") from exc

    obs = _as_count_vector(observed, "observed")
    ref = _as_count_vector(reference, "reference")
    if obs.numel() != ref.numel():
        raise ValueError(
            f"observed and reference count vectors must have the same "
            f"length, got {obs.numel()} and {ref.numel()}")
    for name, counts in (("observed", obs), ("reference", ref)):
        if not bool(torch.isfinite(counts).all()):
            raise ValueError(f"{name} counts contain NaN/Inf")
        if bool((counts < 0).any()):
            raise ValueError(f"{name} counts must be non-negative")
        if not bool(counts.eq(counts.round()).all()):
            raise ValueError(f"{name} counts must be integers")
    n_ref = float(ref.sum())
    n_h = float(obs.sum())
    if n_ref <= 0.0 or n_h <= 0.0:
        return float("nan"), float("nan")
    if isinstance(n_sim, bool):
        return float("nan"), float("nan")
    try:
        n_sim = operator.index(n_sim)
    except TypeError:
        return float("nan"), float("nan")
    if n_sim < 1:
        return float("nan"), float("nan")
    k = ref.numel()
    probs = (ref + _AUDIT_SMOOTH) / (n_ref + _AUDIT_SMOOTH * k)
    expected = n_h * probs
    chi2 = float((((obs - expected) ** 2) / expected).sum())

    gen = torch.Generator().manual_seed(_audit_seed(seed))
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

    For each content field, the pooled training counts define which
    selectable categories were ever observed (ids 0=PAD and 1=SEP are never
    selectable content).  A held-out value whose category is absent from
    that support cannot be predicted (an answer) or meaningfully embedded
    (an input token), so its rate is a transparent bound on achievable
    accuracy.  Derived structural fields (key position, repeated value,
    naive-confusable) use their full category range: their zero/one values
    are measurements, not reserved ids.  The support comes from training
    only; held-out values are looked up, never used to define it.
    """
    out = {}
    for field in fields:
        counts = pooled_counts[field]
        seen = counts > 0
        content = field not in _AUDIT_DERIVED_FIELDS
        if content:
            seen[:min(2, int(seen.numel()))] = False
            n_categories = max(int(categories[field]) - 2, 1)
        else:
            n_categories = max(int(categories[field]), 1)
        entry = {
            "n_categories": n_categories,
            "n_seen_train": int(seen.sum()),
            "fraction_seen_train": int(seen.sum()) / n_categories,
            "unseen_by_split": {},
        }
        for name in heldout_names:
            values = split_values[(name, field)].reshape(-1).to(torch.long)
            if content:
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
    empty = {"n": 0, "n_classes": 0, "expected_per_class": float("nan"),
             "chi2": float("nan"), "df": 0, "p_value": float("nan"),
             "uniform_supported": False}
    try:
        raw = torch.as_tensor(counts).detach().reshape(-1)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        # A non-numeric container is not a count vector.  Like the other
        # reporting companions, abstain instead of letting ``as_tensor``
        # escape as a TypeError/RuntimeError from an audit readout.
        return empty
    if raw.dtype == torch.bool:
        raise ValueError("answer counts must be integers, not booleans")
    if raw.is_complex():
        raise ValueError("answer counts must be real integers")
    c = raw.detach().cpu().to(torch.float64)
    if not bool(torch.isfinite(c).all()):
        raise ValueError("answer counts contain NaN/Inf")
    if bool((c < 0).any()):
        raise ValueError("answer counts must be non-negative")
    if not bool(c.eq(c.round()).all()):
        raise ValueError("answer counts must be integers")
    n = float(c.sum())
    k = c.numel()
    out = {"n": int(n), "n_classes": int(k),
           "expected_per_class": float(n / k) if k else float("nan"),
           "chi2": float("nan"), "df": int(max(k - 1, 0)),
           "p_value": float("nan"), "uniform_supported": False}
    if k < 2 or n <= 0.0:
        return out
    if isinstance(n_sim, bool):
        return out
    try:
        n_sim_int = operator.index(n_sim)
    except TypeError:
        return out
    if n_sim_int < 1:
        return out
    n_sim = n_sim_int
    expected = n / k
    chi2 = float((((c - expected) ** 2) / expected).sum())
    gen = torch.Generator().manual_seed(_audit_seed(seed))
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
      queries) and their joint ``(key_pos, value_repeats, naive_confusable)``
      cell, the total-variation distance between each held-out split
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
    check_data_geometry(n_pairs, vocab)
    batch = _positive_audit_count(batch, "batch")
    batches = _positive_audit_count(batches, "batches")
    if batch * batches < 2:
        raise ValueError(
            "split_leakage_audit needs at least two samples per split to "
            f"estimate a sampling-noise floor, got batch={batch} x "
            f"batches={batches}=1; the floor would be NaN and the TV/floor "
            f"check would silently pass")
    if isinstance(train_seeds, (str, bytes)) or isinstance(
            train_seeds, Mapping):
        raise ValueError(
            "train_seeds must be an iterable of integer seeds, got "
            f"{type(train_seeds).__name__}")
    try:
        train_seed_iter = iter(train_seeds)
    except TypeError as exc:
        raise ValueError(
            "train_seeds must be an iterable of integer seeds, got "
            f"{type(train_seeds).__name__}") from exc
    unique = {}
    for seed in train_seed_iter:
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
    # A held-out row can match any training stream, not just the first
    # seed's stream; count the overlap against the union so a multi-seed
    # protocol cannot hide leakage behind seed order.
    train_fingerprints = set()
    for train_name in train_names:
        train_fingerprints.update(fingerprints[train_name])
    for name in ("val", "test", "stats"):
        out[f"duplicate_overlap_train_{name}"] = len(
            fingerprints[name] & train_fingerprints)
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

    A sweep needs at least two fitted lengths: with a single point the
    base-to-longest drop and entropy drift are identically zero, so H2a
    and H2b would read a degenerate "no degradation" as evidence.  Raise
    rather than return such a one-point grid so the caller cannot train a
    full E2 grid that answers nothing.
    """
    values = {}
    for name, value in (("base", base), ("factor", factor), ("n", n),
                        ("max_len", max_len)):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, not bool")
        try:
            values[name] = operator.index(value)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be an integer, got {value!r}") from exc
    base, factor, n, max_len = (values[name] for name in
                               ("base", "factor", "n", "max_len"))
    if base < 1:
        raise ValueError(f"base must be >= 1, got {base}")
    if factor <= 1:
        raise ValueError(
            f"factor must be > 1 so the sweep contains distinct lengths, "
            f"got {factor}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if max_len < 1:
        raise ValueError(f"max_len must be >= 1, got {max_len}")
    lengths = [base * (factor ** i) for i in range(n)]
    fits = [L for L in lengths if 2 * L + 2 <= max_len]
    if len(fits) < 2:
        raise ValueError(
            f"length sweep needs at least two fitted lengths, got {fits} "
            f"under max_len={max_len} (base={base}, factor={factor}, n={n}); "
            f"raise max_len or reduce base/n"
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
    malformed = answer_uniformity_test(object())
    check("malformed_answer_counts_abstain",
          not malformed["uniform_supported"]
          and math.isnan(malformed["p_value"]))
    try:
        answer_uniformity_test(torch.tensor([True, False]))
        check("boolean_answer_counts_rejected", False)
    except ValueError:
        check("boolean_answer_counts_rejected", True)
    for bad_counts in (torch.tensor([-1, 2]),
                       torch.tensor([float("nan"), 1.0]),
                       torch.tensor([0.5, 1.5])):
        try:
            answer_uniformity_test(bad_counts)
            check("invalid_answer_counts_rejected", False)
        except ValueError:
            check("invalid_answer_counts_rejected", True)
    try:
        _distribution_shift_p(torch.tensor([1.0, 2.0]),
                              torch.tensor([1.0, 2.0, 3.0]))
        check("mismatched_shift_counts_rejected", False)
    except ValueError:
        check("mismatched_shift_counts_rejected", True)
    try:
        _distribution_shift_p(torch.tensor([0.5, 1.5]),
                              torch.tensor([1.0, 2.0]))
        check("fractional_shift_counts_rejected", False)
    except ValueError:
        check("fractional_shift_counts_rejected", True)
    try:
        _counts(torch.tensor([0, 3]), 3)
        check("out_of_range_audit_value_rejected", False)
    except ValueError:
        check("out_of_range_audit_value_rejected", True)
    try:
        _counts(torch.tensor([0.5, 1.5]), 3)
        check("fractional_audit_value_rejected", False)
    except ValueError:
        check("fractional_audit_value_rejected", True)

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
    check("difficulty_joint_encoding",
          _joint_difficulty_values(feat).tolist() == [5, 6])
    # Difficulty strata are the *pre-treatment* covariate used to
    # post-stratify accuracy across lengths.  With one bin per pair the
    # stratum must reproduce the historical joint encoding exactly; with
    # three relative bins the same samples collapse into the middle bin.
    strata_exact = difficulty_strata(
        feat_split["idx"], feat_split["answer"], 3, n_key_bins=3)
    check("difficulty_strata_matches_joint",
          strata_exact["stratum"].tolist() == [5, 6])
    check("difficulty_strata_labels_complete",
          len(strata_exact["labels"]) == 12
          and [row["id"] for row in strata_exact["labels"]]
          == list(range(12)))
    check("difficulty_strata_bounds_cover_pairs",
          strata_exact["key_bin_bounds"][0][0] == 0
          and strata_exact["key_bin_bounds"][-1][1] == 3)
    # A malformed width must be refused rather than silently slicing the
    # wrong columns into an "easy" label.
    try:
        difficulty_features(feat_split["idx"][:, :-1],
                            feat_split["answer"], 3)
        check("difficulty_strata_geometry_rejected", False)
    except ValueError:
        check("difficulty_strata_geometry_rejected", True)

    # Joint-difficulty shift: the 2x2 marginals are identical in both
    # samples, so the separate field tests are blind by construction; the
    # joint test must still detect the reversed dependence.  A same-source
    # draw must not be flagged.
    joint_ref = torch.tensor([300, 200, 200, 300], dtype=torch.int64)
    joint_obs = torch.tensor([200, 300, 300, 200], dtype=torch.int64)

    def _joint_marginals(c):
        return (torch.stack([c[0] + c[1], c[2] + c[3]]),
                torch.stack([c[0] + c[2], c[1] + c[3]]))

    ref_rows, ref_cols = _joint_marginals(joint_ref)
    obs_rows, obs_cols = _joint_marginals(joint_obs)
    check("joint_shift_marginals_equal",
          torch.equal(ref_rows, obs_rows) and torch.equal(ref_cols, obs_cols))
    p_joint, chi2_joint = _distribution_shift_p(
        joint_obs, joint_ref, n_sim=400, seed=90210)
    check("joint_shift_detected", p_joint <= 1.0 / 401.0)
    check("joint_shift_chi2_large", chi2_joint > 100.0)
    jgen = torch.Generator().manual_seed(43)
    joint_same = torch.bincount(
        torch.multinomial(joint_ref.to(torch.float64) / joint_ref.sum(),
                          1000, replacement=True, generator=jgen),
        minlength=4).to(torch.int64)
    p_joint_same, _ = _distribution_shift_p(
        joint_same, joint_ref, n_sim=400, seed=90210)
    check("joint_same_source_not_flagged", p_joint_same > 0.05)

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
    # A misspelled split name used to be skipped by the chronology loop, so
    # an interleaved ``valid`` block could pass while only train/test were
    # compared.  Unknown names must be refused instead.
    try:
        assert_time_ordered(
            times, {"train": slice(0, 60), "valid": slice(50, 80),
                    "test": slice(80, 100)})
        check("time_unknown_split_name_rejected", False)
    except ValueError:
        check("time_unknown_split_name_rejected", True)
    # Integer timestamps above 2**53 must be compared exactly.  Casting
    # them to float64 collapses adjacent nanosecond instants and would let
    # a training row newer than a validation row pass the chronology gate.
    ns_times = torch.tensor([1_700_000_000_000_000_001,
                             1_700_000_000_000_000_000], dtype=torch.int64)
    try:
        assert_time_ordered(ns_times, {"train": [0], "val": [1],
                                       "test": []})
        check("time_int64_precision_leakage_rejected", False)
    except ValueError:
        check("time_int64_precision_leakage_rejected", True)

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
    # A dominant group can make the midpoint pass leave a split empty.  The
    # fallback must donate a group that improves the mass allocation, not
    # the arbitrary last group (which could be the dominant one and leave
    # train with only a single small group).
    imbalanced = torch.arange(3).repeat_interleave(
        torch.tensor([10, 10, 80]))
    imbalanced_split = group_disjoint_split(imbalanced, seed=1)
    check("group_split_fallback_preserves_train_mass",
          int(imbalanced_split["train"].numel()) == 80)
    assert_group_disjoint(imbalanced, imbalanced_split)
    # Non-empty is not enough: the mass rebalance must also avoid putting
    # the dominant group in val/test when the midpoint pass picked a poor
    # permutation (the old fallback produced 10/80/10 here).
    mixed = torch.arange(3).repeat_interleave(torch.tensor([54, 17, 19]))
    mixed_split = group_disjoint_split(mixed, seed=1)
    check("group_split_mass_rebalanced",
          int(mixed_split["train"].numel()) == 54)
    assert_group_disjoint(mixed, mixed_split)
    g_split_alias = group_disjoint_split(groups, seed=3 + 2 ** 32)
    check("group_split_seed_alias_canonical",
          all(torch.equal(g_split[k], g_split_alias[k]) for k in g_split))
    try:
        assert_group_disjoint(
            groups, {"train": torch.arange(0, 100),
                     "val": torch.arange(100, 150),
                     "test": torch.arange(150, 200)})
        check("group_crossing_rejected", False)
    except ValueError:
        check("group_crossing_rejected", True)

    # String group ids (user/document names) must be usable; the encoder
    # maps them to dense codes and the assertion audits every named split.
    str_groups = [f"user{i // 2}" for i in range(20)]
    str_split = group_disjoint_split(str_groups, seed=5)
    assert_group_disjoint(str_groups, str_split)
    check("string_group_split_covers",
          sum(len(str_split[name]) for name in ("train", "val", "test"))
          == len(str_groups))
    try:
        assert_group_disjoint(
            ["alice", "alice", "bob"],
            {"train": [0], "holdout": [1, 2]})
        check("group_crossing_arbitrary_name_rejected", False)
    except ValueError:
        check("group_crossing_arbitrary_name_rejected", True)
    try:
        group_disjoint_split([["unhashable"], ["group"], ["ids"]])
        check("unhashable_group_rejected", False)
    except ValueError:
        check("unhashable_group_rejected", True)
    try:
        group_disjoint_split(3)
        check("scalar_group_rejected", False)
    except ValueError:
        check("scalar_group_rejected", True)

    try:
        assert_time_ordered(torch.tensor([0.0, float("nan"), 2.0]),
                            time_ordered_split(3))
        check("time_nonfinite_rejected", False)
    except ValueError:
        check("time_nonfinite_rejected", True)
    for bad_times in (["2020-01-01", "2020-01-02"],
                      torch.tensor([True, False]),
                      torch.tensor([0.0 + 0.0j, 1.0 + 0.0j])):
        try:
            assert_time_ordered(bad_times, {"train": [0], "test": [1]})
            check("time_non_numeric_rejected", False)
        except ValueError:
            check("time_non_numeric_rejected", True)

    # Boolean split masks must select the masked rows, not be cast to 0/1
    # positions; duplicated or missing sample assignments must be rejected
    # instead of silently auditing the wrong data.
    bool_train = torch.tensor([False, False, True, True])
    bool_test = ~bool_train
    try:
        assert_group_disjoint(
            torch.tensor([0, 0, 1, 1]),
            {"train": bool_train, "test": bool_test})
        check("bool_mask_split_read_correctly", True)
    except ValueError:
        check("bool_mask_split_read_correctly", False)
    try:
        assert_split_partition(
            4, {"train": [0, 1], "val": [1, 2], "test": [3]})
        check("split_duplicate_rejected", False)
    except ValueError:
        check("split_duplicate_rejected", True)
    try:
        assert_split_partition(4, {"train": [0, 1], "val": [2]})
        check("split_missing_sample_rejected", False)
    except ValueError:
        check("split_missing_sample_rejected", True)
    try:
        assert_split_partition(4, {"train": torch.tensor([0.5, 1.0])})
        check("split_float_index_rejected", False)
    except ValueError:
        check("split_float_index_rejected", True)
    try:
        assert_split_partition(
            4, {"train": [0, 0, 1], "val": [2], "test": [3]})
        check("within_split_duplicate_rejected", False)
    except ValueError:
        check("within_split_duplicate_rejected", True)
    try:
        assert_split_partition(
            4, {"train": [0, 1], "valid": [1, 2]})
        check("arbitrary_split_name_audited", False)
    except ValueError:
        check("arbitrary_split_name_audited", True)
    for bad_n in (True, 3.5, 0):
        try:
            assert_split_partition(bad_n, {"train": [0]})
            check("split_bad_n_rejected", False)
        except ValueError:
            check("split_bad_n_rejected", True)
    # An empty split container is a legitimate assignment (the function
    # documents empty splits as allowed), including ``[]`` whose inferred
    # dtype is float32.  It must not be rejected as a floating index array.
    for empty in ([], (), torch.tensor([])):
        try:
            assert_split_partition(4, {"train": [0, 1, 2, 3],
                                       "test": empty})
            check("empty_split_container_allowed", True)
        except ValueError:
            check("empty_split_container_allowed", False)
    try:
        time_ordered_split(10.5)
        check("time_split_fractional_n_rejected", False)
    except ValueError:
        check("time_split_fractional_n_rejected", True)
    try:
        group_disjoint_split(torch.tensor([0.0, 1.0, float("nan"), 2.0, 3.0]))
        check("group_nonfinite_rejected", False)
    except ValueError:
        check("group_nonfinite_rejected", True)

    # Data-geometry contract: too many keys for the selectable ids, or a
    # position budget that cannot hold the sequence, must fail explicitly.
    check("geometry_valid_accepted",
          check_data_geometry(3, 5, max_len=8) is None)
    try:
        check_data_geometry(5, 6)
        check("geometry_too_many_keys_rejected", False)
    except ValueError:
        check("geometry_too_many_keys_rejected", True)
    try:
        check_data_geometry(16, 512, max_len=33)
        check("geometry_max_len_rejected", False)
    except ValueError:
        check("geometry_max_len_rejected", True)
    for geo in ((3.5, 64), (3, 64.0), (3, 64, 7.5)):
        try:
            check_data_geometry(*geo)
            check(f"geometry_fractional_rejected_{geo}", False)
        except ValueError:
            check(f"geometry_fractional_rejected_{geo}", True)
    for args in ((2, 0, 64), (2, -1, 64), (2, 4.0, 64), (2, 4, -1)):
        try:
            make_batch(*args, device="cpu")
            check(f"batch_geometry_rejected_{args}", False)
        except ValueError:
            check(f"batch_geometry_rejected_{args}", True)

    # Length-sweep geometry: a one-point "sweep" would make the base-to-
    # longest drop and entropy drift identically zero, turning H2b's
    # equivalence test into a tautology.  The helper must reject it.
    check("length_sweep_default",
          length_sweep_lengths(16, 2, 4, 256) == [16, 32, 64])
    try:
        length_sweep_lengths(16, 2, 4, 34)
        check("length_sweep_single_point_rejected", False)
    except ValueError:
        check("length_sweep_single_point_rejected", True)
    zero_sim = answer_uniformity_test(torch.tensor([5, 5]), n_sim=0)
    check("uniformity_zero_sim_no_claim",
          math.isnan(zero_sim["p_value"])
          and not zero_sim["uniform_supported"])
    for args in ((0, 2, 4, 256), (16, 1, 4, 256), (16, 0, 4, 256),
                 (16, 2, 0, 256)):
        try:
            length_sweep_lengths(*args)
            check(f"length_sweep_invalid_rejected_{args}", False)
        except ValueError:
            check(f"length_sweep_invalid_rejected_{args}", True)

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
    # A split is a seed-owned stream, but make_batch's draws are
    # shape-dependent: the same seed at a different batch size produces a
    # disjoint realization.  Reusing a training seed as a held-out stream
    # at another geometry therefore has zero content overlap while still
    # training and evaluating on the same stream.  The canonical seed key
    # must flag the cross-group collision on its own.
    seed_collision = actual_stream_overlap_audit(
        4, 64, "cpu",
        train_specs={"train:0": {"seed": 6, "batch": 8, "steps": 2,
                                 "group": "train"}},
        heldout_specs={"val": {"seed": 6, "batch": 9, "batches": 2,
                               "group": "val"}})
    check("seed_collision_has_no_content_overlap",
          seed_collision["max_overlap"] == 0
          and seed_collision["pairwise_overlap"]["train:0|val"] == 0)
    check("cross_group_seed_collision_detected",
          seed_collision["cross_group_seed_collisions"] == ["train:0|val"])
    alias_collision = actual_stream_overlap_audit(
        4, 64, "cpu",
        train_specs={"train:0": {"seed": 6, "batch": 8, "steps": 2,
                                 "group": "train"}},
        heldout_specs={"test": {"seed": 6 + 2 ** 32, "batch": 9,
                                "batches": 2, "group": "test"}})
    check("seed_alias_collision_detected",
          alias_collision["cross_group_seed_collisions"] == ["train:0|test"])
    same_group_seed = actual_stream_overlap_audit(
        4, 64, "cpu",
        train_specs={"train:0": {"seed": 6, "batch": 8, "steps": 2,
                                 "group": "train"}},
        heldout_specs={"train_geom2": {"seed": 6, "batch": 9, "batches": 2,
                                       "group": "train"}})
    check("same_group_seed_reuse_not_leakage",
          same_group_seed["cross_group_seed_collisions"] == [])
    # A shared name would overwrite one stream in the replay map and make
    # the train-vs-held-out pairwise lookup raise ``KeyError``; names must
    # be rejected explicitly while the caller can still fix the spec.
    try:
        actual_stream_overlap_audit(
            4, 64, "cpu",
            train_specs={"shared": {"seed": 0, "batch": 8, "steps": 1}},
            heldout_specs={"shared": {"seed": 1, "batch": 8, "batches": 1}})
        check("duplicate_stream_name_rejected", False)
    except ValueError:
        check("duplicate_stream_name_rejected", True)
    # Pair keys are flattened as "<name>|<name>"; a delimiter inside a name
    # can make two different pairs share one key, so reject it explicitly.
    try:
        actual_stream_overlap_audit(
            4, 64, "cpu",
            train_specs={"train|0": {"seed": 0, "batch": 8, "steps": 1}},
            heldout_specs={"val": {"seed": 1, "batch": 8, "batches": 1}})
        check("stream_name_delimiter_rejected", False)
    except ValueError:
        check("stream_name_delimiter_rejected", True)

    for label, kwargs in (
            ("split_zero_batches",
             dict(batch=8, batches=0)),
            ("split_zero_batch",
             dict(batch=0, batches=1)),
    ):
        try:
            split_leakage_audit(4, 64, "cpu", train_seeds=(0,), **kwargs)
            check(f"{label}_rejected", False)
        except ValueError:
            check(f"{label}_rejected", True)
    for label, train_spec, heldout_spec in (
            ("actual_zero_steps",
             {"seed": 0, "batch": 8, "steps": 0},
             {"seed": 1, "batch": 8, "batches": 1}),
            ("actual_zero_batches",
             {"seed": 0, "batch": 8, "steps": 1},
             {"seed": 1, "batch": 8, "batches": 0}),
            ("actual_zero_batch",
             {"seed": 0, "batch": 0, "steps": 1},
             {"seed": 1, "batch": 8, "batches": 1}),
    ):
        try:
            actual_stream_overlap_audit(
                4, 64, "cpu",
                train_specs={"train:0": train_spec},
                heldout_specs={"val": heldout_spec})
            check(f"{label}_rejected", False)
        except ValueError:
            check(f"{label}_rejected", True)

    for label, train_spec in (
            ("actual_nonmapping_spec", None),
            ("actual_missing_seed",
             {"batch": 8, "steps": 1}),
            ("actual_null_seed",
             {"seed": None, "batch": 8, "steps": 1}),
            ("actual_missing_batch",
             {"seed": 0, "steps": 1}),
            ("actual_fractional_seed",
             {"seed": 0.5, "batch": 8, "steps": 1}),
            ("actual_fractional_batch",
             {"seed": 0, "batch": 8.5, "steps": 1}),
    ):
        try:
            actual_stream_overlap_audit(
                4, 64, "cpu",
                train_specs={"train:0": train_spec},
                heldout_specs={"val": {"seed": 1, "batch": 8, "batches": 1}})
            check(f"{label}_rejected", False)
        except ValueError:
            check(f"{label}_rejected", True)

    # Dataset-versioning fingerprint must depend on the batch geometry the
    # stream is realized with, not only on the seed.
    fp_a = stream_fingerprint(4, 64, "cpu", batch=4, seed=77)
    fp_b = stream_fingerprint(4, 64, "cpu", batch=8, seed=77)
    check("fingerprint_geometry_sensitive",
          fp_a["sha256"] != fp_b["sha256"])
    check("fingerprint_repeatable",
          fp_a == stream_fingerprint(4, 64, "cpu", batch=4, seed=77))
    fp_alias = stream_fingerprint(4, 64, "cpu", batch=4,
                                  seed=77 + 2 ** 32)
    check("fingerprint_seed_alias_canonical",
          fp_alias["sha256"] == fp_a["sha256"]
          and fp_alias["stream_key"] == fp_a["stream_key"])
    # A non-integral seed must not be truncated into a different stream
    # before the digest is computed.
    for bad_seed in (77.5, 77.0, "77", True, None):
        try:
            stream_fingerprint(4, 64, "cpu", batch=4, seed=bad_seed)
            check(f"fingerprint_bad_seed_rejected_{bad_seed!r}", False)
        except ValueError:
            check(f"fingerprint_bad_seed_rejected_{bad_seed!r}", True)
    # Versioning must not hash an empty or unrealizable draw: such a digest
    # would certify a dataset the cleaning gate rejects.
    for bad_batch in (0, -1, True, 4.0, "4"):
        try:
            stream_fingerprint(4, 64, "cpu", batch=bad_batch, seed=77)
            check(f"fingerprint_bad_batch_rejected_{bad_batch!r}", False)
        except ValueError:
            check(f"fingerprint_bad_batch_rejected_{bad_batch!r}", True)
    try:
        stream_fingerprint(5, 6, "cpu", batch=4, seed=77)
        check("fingerprint_impossible_geometry_rejected", False)
    except ValueError:
        check("fingerprint_impossible_geometry_rejected", True)

    # A full-stream digest must cover every consumption batch, not only the
    # first one, and must be stable for the same generator sequence.
    def _digest_batches(seed):
        gen = torch.Generator().manual_seed(seed)
        return [make_batch(4, 3, 32, "cpu", gen=gen) for _ in range(2)]

    digest_a = stream_content_digest(_digest_batches(123))
    digest_b = stream_content_digest(_digest_batches(123))
    check("stream_digest_repeatable", digest_a == digest_b)
    check("stream_digest_batch_count",
          digest_a["n_batches"] == 2 and digest_a["n_rows"] == 8)
    batches_rev = list(reversed(_digest_batches(123)))
    check("stream_digest_order_sensitive",
          stream_content_digest(batches_rev)["sha256"]
          != digest_a["sha256"])
    tampered = [(idx.clone(), tgt.clone(), dict(meta))
                for idx, tgt, meta in _digest_batches(123)]
    tampered[-1][1][0, 0] = (tampered[-1][1][0, 0] + 1) % 30
    check("stream_digest_covers_later_batches",
          stream_content_digest(tampered)["sha256"] != digest_a["sha256"])
    try:
        stream_content_digest([(_digest_batches(123)[0][0].reshape(-1),) +
                               _digest_batches(123)[0][1:]])
        check("stream_digest_bad_shape_rejected", False)
    except ValueError:
        check("stream_digest_bad_shape_rejected", True)

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

    # The cleaning gate's contract is ValueError for *every* malformed batch.
    # A missing or non-mapping meta object used to escape as KeyError /
    # TypeError from the strict checker's raw subscript, so callers that
    # handle the documented ValueError did not catch it.
    for label, bad_meta in (
            ("missing_query_pos", {"answer": sparse_meta["answer"]}),
            ("missing_answer", {"query_pos": sparse_meta["query_pos"]}),
            ("not_a_mapping", None)):
        try:
            validate_batch(sparse_idx, sparse_tgt, bad_meta, 64)
            check(f"validate_rejects_{label}", False)
        except ValueError:
            check(f"validate_rejects_{label}", True)

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

    # PAD is equally reserved on the input side: a PAD query/key or a PAD
    # value is not produced by the generator and must not pass the cleaning
    # gate merely because the answer pairing still happens to agree.
    pad_query_idx = sparse_idx.clone()
    pad_query_idx[rows, sparse_meta["query_pos"]] = 0
    pad_query_idx[rows, sparse_meta["key_pos"]] = 0
    try:
        validate_batch(pad_query_idx, sparse_tgt, sparse_meta, 64)
        check("validate_rejects_pad_query_key", False)
    except ValueError:
        check("validate_rejects_pad_query_key", True)

    pad_value_idx = sparse_idx.clone()
    pad_value_col = torch.where(sparse_meta["key_pos"] != 0,
                                torch.ones_like(rows), torch.full_like(rows, 3))
    pad_value_idx[rows, pad_value_col] = 0
    try:
        validate_batch(pad_value_idx, sparse_tgt, sparse_meta, 64)
        check("validate_rejects_pad_value", False)
    except ValueError:
        check("validate_rejects_pad_value", True)

    # A repeated non-query key violates the generator's without-replacement
    # contract even though the queried key is still unique; the cleaning gate
    # must reject it (the old check only looked at the query key).
    dup_nonquery = torch.tensor(
        [[2, 10, 3, 11, 3, 12, 4, 13, SEP_ID, 4]], dtype=torch.long)
    dup_nonquery_meta = {
        "query_pos": torch.tensor([9], dtype=torch.long),
        "key_pos": torch.tensor([6], dtype=torch.long),
        "answer": torch.tensor([13], dtype=torch.long),
    }
    try:
        validate_batch(dup_nonquery, None, dup_nonquery_meta, 32)
        check("validate_rejects_duplicate_nonquery_key", False)
    except ValueError:
        check("validate_rejects_duplicate_nonquery_key", True)

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
            "pad_query_key": (pad_query_idx, sparse_tgt, sparse_meta),
            "pad_value": (pad_value_idx, sparse_tgt, sparse_meta),
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
    check("audit_e2e_joint_field_present",
          "difficulty_joint" in e2e["audit_fields"]
          and set(e2e["shift_p_value"]["difficulty_joint"])
          == {"val", "test", "stats"})
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
