# cage data: streams, split protocol and batch construction.
# Executed into the wrapper module namespace by the loader;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
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
    if n > _MAX_TORCH_INDEX:
        raise ValueError(
            f"n_samples={n} exceeds torch's signed 64-bit index range")
    try:
        train_frac = float(train_frac)
        val_frac = float(val_frac)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
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
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
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
    first = torch.full((n_groups,), n, dtype=torch.long, device=g.device)
    first.scatter_reduce_(0, inverse,
                          torch.arange(n, dtype=torch.long, device=g.device),
                          reduce="amin")
    order = torch.argsort(first)
    # Canonicalise exactly like the data-stream generators: a split assigned
    # under an aliased seed must be the same split, never a different one.
    perm = order[torch.randperm(
        n_groups, device=g.device,
        generator=torch.Generator(device=g.device).manual_seed(
            stream_seed_key(seed)))]
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
        mask = torch.isin(
            inverse,
            torch.tensor(gids, dtype=torch.long, device=g.device))
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
        start, stop, step = index.indices(n)
        if (step > 0 and start >= stop) or (step < 0 and start <= stop):
            idx = torch.empty(0, dtype=torch.long)
        else:
            idx = torch.arange(start, stop, step, dtype=torch.long)
    elif isinstance(index, range):
        idx = torch.tensor(list(index), dtype=torch.long)
    else:
        try:
            t = torch.as_tensor(index)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError(
                f"split indices must be integers or a boolean mask, got "
                f"{type(index).__name__}") from exc
        if t.dtype == torch.bool:
            if t.dim() != 1:
                raise ValueError(
                    "boolean split masks must be one-dimensional (one flag "
                    f"per sample), got shape {tuple(t.shape)}; flattening a "
                    f"matrix would select the wrong samples")
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
            if t.dim() > 1:
                raise ValueError(
                    "split indices must be one-dimensional (one index per "
                    f"sample), got shape {tuple(t.shape)}; flattening a "
                    f"matrix would select the wrong samples")
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


_DENSE_PARTITION_LIMIT = 10_000_000


def _assert_sparse_partition(n: int, splits, require_all: bool) -> None:
    intervals = []
    points = {}
    for name, indices in splits.items():
        if isinstance(indices, slice):
            start, stop, step = indices.indices(n)
            if step == 1:
                if start < stop:
                    intervals.append((start, stop, name))
                continue
        elif isinstance(indices, range) and indices.step == 1:
            start, stop = indices.start, indices.stop
            if start < stop:
                if start < 0 or stop > n:
                    raise ValueError(
                        f"split indices must lie in [0, {n}), got range "
                        f"[{start}, {stop - 1}]")
                intervals.append((start, stop, name))
            continue
        idx = _as_indices(indices, n)
        if idx.numel() == 0:
            continue
        if idx.numel() != idx.unique().numel():
            raise ValueError(
                f"split {name!r} contains repeated row indices; an index "
                f"split is a set and duplicates would silently over-weight "
                f"those rows")
        for value in idx.tolist():
            value = int(value)
            if value in points:
                raise ValueError(
                    f"split {name!r} repeats a sample already assigned to "
                    f"another split: splits must be disjoint")
            points[value] = name
    intervals.sort(key=lambda item: (item[0], item[1]))
    merged = []
    covered = 0
    for start, stop, name in intervals:
        if merged and start < merged[-1][1]:
            raise ValueError(
                f"split {name!r} repeats a sample already assigned to "
                f"another split: splits must be disjoint")
        if merged and start == merged[-1][1]:
            covered += stop - start
            merged[-1] = (merged[-1][0], stop)
        else:
            covered += stop - start
            merged.append((start, stop))
    starts = [start for start, _ in merged]
    for value, name in points.items():
        i = bisect.bisect_right(starts, value) - 1
        if i >= 0 and value < merged[i][1]:
            raise ValueError(
                f"split {name!r} repeats a sample already assigned to "
                f"another split: splits must be disjoint")
    if require_all and covered + len(points) != n:
        missing = n - covered - len(points)
        raise ValueError(
            f"split assignment leaves {missing} of {n} samples unassigned")


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
    if n > _MAX_TORCH_INDEX:
        raise ValueError(
            f"n_samples={n} exceeds torch's signed 64-bit index range")
    splits = _split_container(splits)
    if n > _DENSE_PARTITION_LIMIT:
        _assert_sparse_partition(n, splits, require_all)
        return
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
    if raw.is_floating_point() and not torch.is_tensor(times):
        # ``torch.as_tensor`` captures a Python float list as float32 by
        # default.  That rounds e.g. 16777216.0 and 16777217.0 to the same
        # value, silently accepting a chronological violation.  Rebuild the
        # original container directly at float64; integer containers stay in
        # their exact integer dtype.
        try:
            raw = torch.as_tensor(times, dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError(
                "time values must be numeric timestamps, got a container "
                "that cannot be widened to float64") from exc
        if raw.dim() != 1:
            raise ValueError(
                "time values must be one-dimensional (one timestamp per "
                f"sample), got shape {tuple(raw.shape)}; flattening a matrix "
                f"would audit the wrong samples")
        raw = raw.reshape(-1)
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
    if n_pairs > _MAX_TORCH_INDEX or vocab > _MAX_TORCH_INDEX:
        raise ValueError(
            f"n_pairs={n_pairs} and vocab={vocab} must fit torch's signed "
            f"64-bit index range")
    tensor_width = 2 * n_pairs + 3
    if tensor_width > _MAX_TORCH_INDEX:
        raise ValueError(
            f"n_pairs={n_pairs} needs a per-row tensor width of "
            f"{tensor_width}, exceeding torch's signed 64-bit index range")
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
    # Token ids may be any integer dtype, but the association code ranges
    # over [0, vocab*vocab); arithmetic in int16/int32 would wrap and feed
    # a negative/colliding code to the exposure audit.  The returned code is
    # the documented exact integer, so widen before multiplying.
    key = key.to(torch.long)
    value = value.to(torch.long)
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
    if value > _MAX_TORCH_INDEX:
        raise ValueError(
            f"vocab={value} exceeds torch's signed 64-bit index range")
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
    n_rows = int(idx.size(0))
    # A zero-row batch has no keys, so "unique" is undefined rather than a
    # perfect 1.0 score.
    out = {"n": n_rows,
           "unique_key_rate": 1.0 if n_rows else float("nan")}
    if diagnostics:
        out["n_duplicate_rows"] = _count_duplicate_rows(idx)
    return out


def make_length_linked_batch(idx, meta, n_pairs, target_pairs, vocab,
                             gen: torch.Generator | None = None):
    """Extend a validated recall batch with fresh distractors, paired by item.

    The natural length sweep generates an independent stream at every
    ``n_pairs``, so a base-length item and a long-length item are different
    samples.  Comparing raw accuracy across those streams therefore mixes
    the length effect with a changing item/difficulty composition (later
    retrieval positions, more repeated values, more naive-confusable
    queries).  Post-stratification can reweight the observed strata, but it
    cannot recover the counterfactual "what would this *same* item score at
    a longer length?".

    This helper constructs that counterfactual without changing the task
    distribution.  Starting from a validated ``[k1 v1 ... kn vn SEP q]``
    batch, it copies the first ``n_pairs`` pairs verbatim and appends
    ``target_pairs - n_pairs`` fresh pairs before SEP/query:

    * the queried key, its paired answer and its pair index are unchanged,
      so every row keeps its exact ``(query key, answer)`` association;
    * the added keys are drawn uniformly *without replacement* from the
      ids not already used as keys (exactly the conditional draw of the
      natural generator at the longer length);
    * the added values are drawn uniformly from the selectable ids, so
      repeated values occur with the same probability as in natural data.

    Conditional on the base rows, the result is distributed like a natural
    ``target_pairs`` sample whose first ``n_pairs`` pairs are fixed.  It is
    therefore a within-item length manipulation, not a different task.
    Callers can chain the calls to build a nested sweep (the longer batch
    contains the shorter one as a prefix).  Targets are sparse (``-1``
    except at the query position), the same convention as ``make_batch``'s
    default.  The returned batch passes the same :func:`validate_batch`
    cleaning gate as every generated batch.

    ``meta`` must be the base batch's metadata (``query_pos``, ``answer``
    and optionally ``key_pos``).  ``target_pairs`` must be at least
    ``n_pairs``; the geometry is checked before any draw so an impossible
    extension raises ``ValueError`` instead of failing in ``topk``.
    """
    n_pairs = _positive_audit_count(n_pairs, "n_pairs")
    target_pairs = _positive_audit_count(target_pairs, "target_pairs")
    vocab = _batch_vocab_int(vocab)
    if target_pairs < n_pairs:
        raise ValueError(
            f"target_pairs={target_pairs} is shorter than the source batch "
            f"({n_pairs} pairs); length extension can only add pairs")
    check_data_geometry(target_pairs, vocab)
    if not _is_integer_tensor(idx) or idx.dim() != 2:
        raise ValueError(
            f"idx must be a (batch, 2*n_pairs + 2) integer tensor, got "
            f"{type(idx)}")
    # The model and target contract use long token ids.  Accepting another
    # integer dtype but then assigning long draws/targets into it would
    # either raise a dtype mismatch or silently wrap a wider vocabulary.
    idx = idx.to(torch.long)
    batch, width = idx.shape
    if width != 2 * n_pairs + 2:
        raise ValueError(
            f"idx width {width} does not match n_pairs={n_pairs} "
            f"(expected {2 * n_pairs + 2} positions)")
    validate_batch(idx, None, meta, vocab, diagnostics=False)
    answer = meta["answer"].to(device=idx.device, dtype=torch.long).reshape(-1)
    if answer.shape != (batch,):
        raise ValueError(
            "meta['answer'] must carry one entry per row of idx")
    keys = idx[:, :2 * n_pairs:2]
    query = idx[:, -1]
    match = keys.eq(query.unsqueeze(1))
    key_pos = match.to(torch.long).argmax(dim=1).mul(2)

    extra = target_pairs - n_pairs
    out = torch.empty((batch, 2 * target_pairs + 2), dtype=idx.dtype,
                      device=idx.device)
    out[:, :2 * n_pairs].copy_(idx[:, :2 * n_pairs])
    if extra:
        n_avail = vocab - 2
        scores = torch.rand((batch, n_avail), generator=gen,
                            device=idx.device)
        # Reserved/used keys keep an -inf score, so ``topk`` samples the
        # remaining ids without replacement, exactly like ``_sample_keys``.
        scores.scatter_(1, keys.to(torch.long) - 2, float("-inf"))
        new_keys = scores.topk(extra, dim=1).indices.add(2)
        new_values = torch.randint(2, vocab, (batch, extra), generator=gen,
                                   device=idx.device)
        out[:, 2 * n_pairs:2 * target_pairs:2].copy_(
            new_keys.to(dtype=idx.dtype))
        out[:, 2 * n_pairs + 1:2 * target_pairs:2].copy_(
            new_values.to(dtype=idx.dtype))
    out[:, 2 * target_pairs].fill_(SEP_ID)
    out[:, 2 * target_pairs + 1].copy_(query)

    targets = torch.full_like(out, -1)
    rows = torch.arange(batch, device=idx.device)
    targets[rows, 2 * target_pairs + 1] = answer
    out_meta = {
        "query_pos": torch.full((batch,), 2 * target_pairs + 1,
                                dtype=torch.long, device=idx.device),
        "key_pos": key_pos.to(device=idx.device, dtype=torch.long),
        "answer": answer,
    }
    validate_batch(out, targets, out_meta, vocab, diagnostics=False)
    return out, targets, out_meta


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
    n_rows = int(idx.size(0))
    return {"n": n_rows,
            "unique_key_rate": 1.0 if n_rows else float("nan"),
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
    if value > _MAX_TORCH_INDEX:
        raise ValueError(
            f"{name}={value} exceeds torch's signed 64-bit index range")
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
