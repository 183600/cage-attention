# cage/experiments/_design.py - Design-time helpers: run ordering, difficulty stamps and coverage.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _finite(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError, TypeError):
        # A JSON integer can be arbitrarily large; converting it to float
        # raises OverflowError.  That value is not a usable finite statistic,
        # so report it as non-finite instead of aborting the audit.
        return False


def _validated_reference_lr(value):
    """Return ``(lr, None)`` or ``(None, reason)`` for a reference rate.

    The sensitivity and per-variant companions accept the frozen reference
    lr from a caller.  A complex tensor (or a JSON-sized integer) makes
    ``float()`` raise ``RuntimeError``/``OverflowError``; a companion should
    abstain with a reason instead of leaking a conversion error.
    """
    if isinstance(value, bool):
        return None, "reference lr must be a number, not a bool"
    try:
        number = float(value)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return None, "reference lr must be a finite positive number"
    if not math.isfinite(number) or number <= 0.0:
        return None, "non-positive reference lr"
    return number, None


def _depth_step_candidates(settings: Settings) -> tuple:
    """Step budgets searched by the depth ladder and its capacity control.

    The search is expressed as multipliers of ``settings.steps`` so the
    command-line ``--steps`` override shifts the whole grid.  The default
    budget is always included even when the multiplier list is empty or
    omits 1.0.
    """
    multipliers = tuple(settings.depth_step_multipliers or (1.0,))
    raw_steps = settings.steps
    base_steps = _positive_integer(raw_steps)
    if base_steps is None:
        raise ValueError(
            f"steps must be a positive integer before the depth ladder can "
            f"build its step grid, got {raw_steps!r}")
    steps = {int(base_steps)}
    for multiplier in multipliers:
        if not _finite(multiplier) or float(multiplier) <= 0.0:
            raise ValueError(
                f"depth_step_multipliers must be finite positive numbers, "
                f"got {multiplier!r}")
        try:
            scaled = base_steps * float(multiplier)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"depth_step_multipliers must keep steps * multiplier "
                f"finite, got {multiplier!r} at steps={base_steps}") from exc
        if not math.isfinite(scaled):
            raise ValueError(
                f"depth_step_multipliers must keep steps * multiplier "
                f"finite, got {multiplier!r} at steps={base_steps}")
        steps.add(max(1, int(round(scaled))))
    return tuple(sorted(steps))


def _declared_seed(value):
    """The exact integer declared as a training seed, or ``None``.

    Sorting/auditing a pooled record must not convert an arbitrarily large
    JSON integer through ``float`` (which can overflow) or reject a valid
    integer merely because it is outside the float64 exact range.
    """
    if isinstance(value, bool):
        return None
    try:
        return int(operator.index(value))
    except TypeError:
        return None


def _seed_order(seed):
    """Deterministic ordering for seed labels of mixed / missing types."""
    exact = _declared_seed(seed)
    if exact is not None:
        return (0, exact, "")
    if isinstance(seed, float) and math.isfinite(seed):
        return (0, seed, "")
    return (1, 0.0, str(seed))


def _blocked_run_order(cells, seeds, schedule_seed, shuffle=True):
    """Randomised complete-block schedule for a grid.

    Each training seed is one block: every declared cell is run exactly
    once inside it.  Running all seeds of one arm consecutively (the
    historical loop order) aliases the arm with any slow drift of the
    machine - thermal throttling, a competing process, a driver change -
    so an apparent treatment effect can be an order effect.  A complete
    block removes that alias: every arm experiences every block position
    on average, and the recorded schedule makes the realised order part of
    the experiment rather than an accident.

    The permutation is deterministic from ``(schedule_seed, block index)``
    and never touches the global RNG, so reruns reproduce the same order
    without perturbing data or dropout streams.  ``shuffle=False`` keeps
    the cell-major order (useful for a controlled legacy replay) but is
    still returned as explicit ``(cell, seed, block, position)`` rows.
    """
    cells = list(cells)
    seeds = list(seeds)
    if not cells or not seeds:
        return []
    out = []
    for block, seed in enumerate(seeds):
        indices = list(range(len(cells)))
        if shuffle:
            # ``random.Random`` with an integer seed is local and
            # deterministic; deriving it from the registered schedule seed
            # and the block keeps the schedule auditable.
            key = ((stream_seed_key(schedule_seed) << 20)
                   ^ (int(block) * 0x9E3779B1) ^ len(cells))
            random.Random(key).shuffle(indices)
        for position, index in enumerate(indices):
            out.append((cells[index], seed, block, position))
    return out


def _run_order_audit(order, cells, seeds, randomized=True):
    """Check that a schedule is a complete, non-degenerate block design.

    ``complete`` means every block contains every declared cell exactly
    once; without that the randomisation cannot balance the treatment
    arms against time.  ``always_first`` / ``always_last`` catch a
    schedule that is technically complete but systematically puts one arm
    at an extreme position in every block.  With only a few blocks, a
    genuinely random schedule can put one arm at an extreme in every block
    by chance, so the extreme-position warning is only raised when the
    family-wise probability under random complete blocks
    (``n_cells * (1/n_cells)**n_blocks``) is below 0.05; a deliberately
    unshuffled schedule is always flagged.  The audit is stored next to
    the run records so the realised order can be checked without rerunning
    the grid.
    """
    container_errors = []
    try:
        cells = list(cells)
    except TypeError:
        cells = []
        container_errors.append({
            "cell": None, "block": None, "position": None,
            "reason": "declared cells are not iterable"})
    try:
        seeds = list(seeds)
    except TypeError:
        seeds = []
        container_errors.append({
            "cell": None, "block": None, "position": None,
            "reason": "declared seeds are not iterable"})
    try:
        order = list(order)
    except TypeError:
        order = None
    cell_keys = [_coverage_cell_key(c) for c in cells]
    expected = set(cell_keys)
    cell_counts: dict[str, int] = {}
    for key in cell_keys:
        cell_counts[key] = cell_counts.get(key, 0) + 1
    duplicate_cells = sorted(
        key for key, count in cell_counts.items() if count > 1)
    n_cells = len(cells)
    n_blocks = len(seeds)
    invalid_seeds = [
        {"block": block, "seed": repr(seed),
         "reason": "declared training seed is not an integer"}
        for block, seed in enumerate(seeds)
        if _declared_seed(seed) is None
    ]
    per_block: dict[int, dict[str, int]] = {}
    positions: dict[str, list[int]] = {key: [] for key in expected}
    per_block_positions: dict[int, list[int]] = {}
    block_seeds: dict[int, list] = {}
    unexpected = []
    seed_mismatches = []
    unexpected_seeds = []
    invalid_indices = list(container_errors)

    def _order_index(value):
        if isinstance(value, bool):
            return None
        try:
            return operator.index(value)
        except TypeError:
            return None

    if order is None:
        # The audit is a reporting helper: a non-iterable schedule is an
        # invalid schedule, not a reason to leak a TypeError and lose the
        # rest of the audit evidence.
        invalid_indices.append({
            "cell": None, "block": None, "position": None,
            "reason": "run-order schedule is not iterable"})
        order = []
    for index, row in enumerate(order):
        if not isinstance(row, (tuple, list)) or len(row) != 4:
            # A short/scalar row used to raise ``ValueError: not enough
            # values to unpack`` from this loop, aborting the audit before
            # it could report that the schedule is incomplete.
            invalid_indices.append({
                "order_index": index, "cell": None, "block": None,
                "position": None,
                "reason": "schedule row is not a (cell, seed, block, "
                          "position) tuple"})
            continue
        cell, order_seed, block, position = row
        key = _coverage_cell_key(cell)
        raw_block, raw_position = block, position
        block, position = _order_index(block), _order_index(position)
        if block is None or position is None:
            invalid_indices.append({
                "order_index": index, "cell": key,
                "block": repr(raw_block),
                "position": repr(raw_position),
                "reason": "block and position must be integers"})
            continue
        # The seed is an identity, not a numeric value: ``True == 1`` and
        # ``1.0 == 1`` would otherwise certify a schedule labelled with a
        # non-integer seed as the planned integer block.  Normalise through
        # the same strict check used for the declared seeds.
        exact_seed = _declared_seed(order_seed)
        if exact_seed is None:
            invalid_indices.append({
                "order_index": index, "cell": key,
                "seed": repr(order_seed),
                "block": repr(raw_block),
                "position": repr(raw_position),
                "reason": "schedule seed must be an integer"})
            continue
        order_seed = exact_seed
        per_block.setdefault(block, {}).setdefault(key, 0)
        per_block[block][key] += 1
        per_block_positions.setdefault(block, []).append(position)
        block_seeds.setdefault(block, []).append(order_seed)
        # The seed is part of the realised design, not metadata: a block
        # labelled with an undeclared seed (or with another block's seed)
        # means the recorded complete-block schedule is not the one that
        # ran, so treatment and time are no longer balanced by block.
        if 0 <= block < n_blocks:
            expected_seed = seeds[block]
            if order_seed != expected_seed:
                seed_mismatches.append({
                    "block": block, "cell": key, "seed": order_seed,
                    "expected": expected_seed})
        else:
            seed_mismatches.append({
                "block": block, "cell": key, "seed": order_seed,
                "expected": None})
        if order_seed not in seeds:
            unexpected_seeds.append({
                "block": block, "cell": key, "seed": order_seed})
        if key not in expected:
            # A cell that was never declared is an unplanned run: it makes
            # the schedule incomplete even if every declared cell is also
            # present exactly once.
            unexpected.append({"block": block, "cell": key,
                               "position": position})
            continue
        positions[key].append(position)
    missing = []
    duplicated = []
    for block, counts in sorted(per_block.items()):
        for key in sorted(expected):
            if counts.get(key, 0) == 0:
                missing.append({"block": block, "cell": key})
            elif counts.get(key, 0) > 1:
                duplicated.append({"block": block, "cell": key,
                                   "count": counts[key]})
    # A complete block design also needs each cell at a distinct slot
    # 0..n_cells-1 inside its block.  Without this check a schedule that
    # puts every cell at position 0 (or out of range) would be certified
    # as complete and balanced.
    invalid_positions = []
    for block, slots in sorted(per_block_positions.items()):
        seen_slots = set()
        for slot in slots:
            if not 0 <= slot < n_cells:
                invalid_positions.append({
                    "block": block, "position": slot,
                    "reason": f"outside [0, {n_cells})"})
            elif slot in seen_slots:
                invalid_positions.append({
                    "block": block, "position": slot,
                    "reason": "duplicate position within the block"})
            seen_slots.add(slot)
    missing_seeds = [
        {"block": block, "seed": seed}
        for block, seed in enumerate(seeds)
        if block not in per_block or seed not in block_seeds.get(block, [])
    ]
    declared_streams: dict[int, list] = {}
    for seed in seeds:
        exact = _declared_seed(seed)
        if exact is not None:
            declared_streams.setdefault(stream_seed_key(exact),
                                        []).append(seed)
    duplicate_seeds = sorted(
        {seed for group in declared_streams.values() if len(group) > 1
         for seed in group},
        key=_seed_order)
    complete = (bool(cells) and bool(seeds)
                and not duplicate_cells
                and set(per_block) == set(range(n_blocks))
                and not missing and not duplicated and not unexpected
                and not invalid_positions and not invalid_indices
                and not seed_mismatches and not unexpected_seeds
                and not missing_seeds and not duplicate_seeds
                and not invalid_seeds)
    always_first = sorted(
        key for key, vals in positions.items()
        if len(vals) == n_blocks and vals and all(v == 0 for v in vals))
    always_last = sorted(
        key for key, vals in positions.items()
        if len(vals) == n_blocks and vals and all(
            v == n_cells - 1 for v in vals))
    if n_cells > 1 and n_blocks > 0:
        extreme_p = min(1.0, n_cells * (1.0 / n_cells) ** n_blocks)
    else:
        extreme_p = 1.0
    # A systematic (unshuffled) schedule is always a real order confound;
    # a random one is only surprising when the family-wise chance of an
    # extreme position is below 5%.
    flag_extremes = (not randomized) or extreme_p < 0.05
    mean_position = {
        key: (sum(vals) / len(vals) if vals else float("nan"))
        for key, vals in sorted(positions.items())}
    finite_means = [v for v in mean_position.values() if _finite(v)]
    return {
        "cells": [_coverage_cell_key(c) for c in cells],
        "seeds": [_declared_seed(s) if _declared_seed(s) is not None
                  else repr(s) for s in seeds],
        "n_runs": len(order),
        "complete": bool(complete),
        "missing": missing,
        "duplicated": duplicated,
        "unexpected": unexpected,
        "invalid_positions": invalid_positions,
        "invalid_indices": invalid_indices,
        "invalid_seeds": invalid_seeds,
        "duplicate_cells": duplicate_cells,
        "seed_mismatches": seed_mismatches,
        "unexpected_seeds": unexpected_seeds,
        "missing_seeds": missing_seeds,
        "duplicate_seeds": duplicate_seeds,
        "always_first": always_first,
        "always_last": always_last,
        "extreme_p_value": extreme_p,
        "extreme_flag_threshold": 0.05,
        "extreme_check_triggered": bool(flag_extremes),
        "mean_position": mean_position,
        "mean_position_spread": (
            max(finite_means) - min(finite_means)
            if finite_means else float("nan")),
        "balanced": bool(
            complete and not (
                flag_extremes and (always_first or always_last))),
    }


def _write_run_order(settings: Settings, out_dir: pathlib.Path, tag: str,
                     order, cells, seeds) -> dict:
    """Persist the realised complete-block schedule before training."""
    # Materialise once: the audit below consumes iterables, and the payload
    # must describe the same schedule/cells/seeds it certified.  A
    # non-iterable container is reported by the audit as invalid rather
    # than aborting the write with an incidental TypeError.
    try:
        order = list(order)
    except TypeError:
        order = []
    try:
        cells = list(cells)
    except TypeError:
        cells = []
    try:
        seeds = list(seeds)
    except TypeError:
        seeds = []
    audit = _run_order_audit(
        order, cells, seeds, randomized=settings.randomize_run_order)
    order_rows = []
    for row in order:
        if not isinstance(row, (tuple, list)) or len(row) != 4:
            order_rows.append({
                "cell": None, "seed": None, "block": None, "position": None,
                "malformed": repr(row)})
            continue
        cell, seed, block, position = row
        order_rows.append({
            "cell": _coverage_cell_key(cell),
            "seed": (_declared_seed(seed) if _declared_seed(seed) is not None
                     else repr(seed)),
            "block": (int(block) if _declared_seed(block) is not None
                      else repr(block)),
            "position": (int(position)
                         if _declared_seed(position) is not None
                         else repr(position))})
    payload = {
        "experiment": tag,
        "schedule_seed": int(settings.run_order_seed),
        "randomized": bool(settings.randomize_run_order),
        "order": order_rows,
        **audit,
        **_stamp(settings),
    }
    _write_json(out_dir / f"{tag}_run_order.json", payload)
    if not audit["complete"]:
        print(f"[{tag}] WARNING: run-order schedule is not a complete block "
              f"design: missing={audit['missing']} "
              f"duplicated={audit['duplicated']}; treatment arms are not "
              f"balanced against run time")
    elif audit["extreme_check_triggered"] and (
            audit["always_first"] or audit["always_last"]):
        print(f"[{tag}] WARNING: run-order schedule puts "
              f"{audit['always_first'] or audit['always_last']} at the same "
              f"extreme position in every block "
              f"(family-wise chance {audit['extreme_p_value']:.4f})")
    else:
        print(f"[{tag}] run order: complete randomised blocks over "
              f"{len(list(seeds))} seed(s) x {len(list(cells))} cell(s); "
              f"mean-position spread="
              f"{audit['mean_position_spread']:.2f}")
    return payload


def _realized_order_audit(runs, order, cells, seeds, cell_of) -> dict:
    """Compare the realised result rows with the frozen block schedule.

    Writing a randomised schedule is not the same as running it.  An
    interrupted-and-resumed grid, a manual retry, or a loop that iterates
    its own cell list instead of the schedule can leave a complete-looking
    record file whose treatment arms are aliased with time.  The result
    rows already carry ``run_block`` / ``run_position``, so the schedule can
    be verified after the fact without rerunning anything.

    ``matches_schedule`` is true only when the frozen schedule is itself a
    complete block design and every planned (cell, seed, block, position)
    tuple appears exactly once with no unplanned tuple or malformed index.
    Missing rows are reported as an incomplete run, not silently ignored:
    the caller decides whether that is a hard failure or just a partial
    grid.
    """
    invalid_order = []
    invalid = []
    try:
        cells = list(cells)
    except TypeError:
        cells = []
        invalid_order.append({
            "order_index": None,
            "reason": "declared cells are not iterable"})
    try:
        seeds = list(seeds)
    except TypeError:
        seeds = []
        invalid_order.append({
            "order_index": None,
            "reason": "declared seeds are not iterable"})
    try:
        order = list(order)
    except TypeError:
        order = []
        invalid_order.append({
            "order_index": None,
            "reason": "run-order schedule is not iterable"})
    try:
        runs = list(runs)
    except TypeError:
        runs = []
        invalid.append({
            "run_index": None,
            "reason": "realised run records are not iterable"})

    def _integer(value):
        if isinstance(value, bool):
            return None
        try:
            return operator.index(value)
        except TypeError:
            return None

    declared_cells = {_coverage_cell_key(cell) for cell in cells}
    declared_streams: dict[int, list] = {}
    invalid_seeds = []
    for seed in seeds:
        exact = _integer(seed)
        if exact is None:
            invalid_seeds.append({
                "seed": repr(seed),
                "reason": "declared training seed is not an integer"})
            continue
        declared_streams.setdefault(stream_seed_key(exact),
                                    []).append(exact)
    declared_seeds = {seed for group in declared_streams.values()
                      for seed in group}
    duplicate_seeds = sorted(
        {seed for group in declared_streams.values() if len(group) > 1
         for seed in group},
        key=_seed_order)
    expected = []
    seen_order = {}
    duplicate_order = []
    for index, row in enumerate(order):
        if not isinstance(row, (tuple, list)) or len(row) != 4:
            invalid_order.append({
                "order_index": index,
                "reason": "schedule row is not a 4-tuple",
            })
            continue
        cell, seed, block, position = row
        seed_i = _integer(seed)
        block_i = _integer(block)
        position_i = _integer(position)
        if seed_i is None or block_i is None or position_i is None:
            invalid_order.append({
                "order_index": index,
                "reason": "non-integral seed/block/position in schedule",
            })
            continue
        key = (_coverage_cell_key(cell), seed_i, block_i, position_i)
        if key[0] not in declared_cells:
            invalid_order.append({
                "order_index": index, "cell": key[0],
                "reason": "schedule cell was not declared",
            })
            continue
        if key[1] not in declared_seeds:
            invalid_order.append({
                "order_index": index, "seed": key[1],
                "reason": "schedule seed was not declared",
            })
            continue
        if key in seen_order:
            # A duplicate in the frozen schedule would otherwise collapse
            # through ``expected_set`` and certify a run row as if every
            # planned tuple had appeared exactly once.  Record the schedule
            # duplicate explicitly so the audit cannot pass.
            duplicate_order.append({"order_index": index, "cell": key[0], "seed": key[1], "block": key[2], "position": key[3]})
            continue
        seen_order[key] = index
        expected.append(key)
    expected_set = set(expected)
    # Matching a malformed frozen schedule is not evidence of a valid
    # complete-block design.  In particular, a schedule that swaps the seeds
    # attached to two blocks (or repeats a within-block position, omits a
    # block, or labels a block outside ``range(len(seeds))``) can still have
    # one unique tuple per declared cell, so the set comparison below would
    # certify it.  Reuse the schedule audit's structural invariants.
    schedule_complete = bool(
        _run_order_audit(order, cells, seeds, randomized=True)["complete"])

    observed = []
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            invalid.append({"run_index": index,
                            "reason": "record is not a JSON object"})
            continue
        try:
            cell = cell_of(run)
        except Exception as exc:  # noqa: BLE001 - report malformed records
            invalid.append({"run_index": index,
                            "reason": f"cell extractor raised: {exc}"})
            continue
        seed = _integer(run.get("seed"))
        block = _integer(run.get("run_block"))
        position = _integer(run.get("run_position"))
        if cell is None or seed is None or block is None or position is None:
            invalid.append({
                "run_index": index, "cell": _coverage_cell_key(cell),
                "reason": "missing or non-integral cell/seed/block/position",
                "seed": repr(run.get("seed")),
                "block": repr(run.get("run_block")),
                "position": repr(run.get("run_position")),
            })
            continue
        observed.append((_coverage_cell_key(cell), seed, block, position))

    counts = {}
    for key in observed:
        counts[key] = counts.get(key, 0) + 1
    observed_set = set(observed)
    missing = sorted(expected_set - observed_set, key=repr)
    unexpected = sorted(observed_set - expected_set, key=repr)
    duplicates = sorted(
        (key for key, count in counts.items() if count > 1), key=repr)

    def _row(key):
        return {"cell": key[0], "seed": key[1],
                "block": key[2], "position": key[3]}

    matches = bool(
        expected and schedule_complete
        and not missing and not unexpected and not duplicates
        and not invalid and not invalid_order and not duplicate_order
        and not invalid_seeds and not duplicate_seeds)
    return {
        "cells": [_coverage_cell_key(c) for c in cells],
        "seeds": [s if _integer(s) is not None else repr(s) for s in seeds],
        "n_expected": len(expected),
        "n_observed": len(observed),
        "schedule_complete": schedule_complete,
        "matches_schedule": matches,
        "missing": [_row(key) for key in missing],
        "unexpected": [_row(key) for key in unexpected],
        "duplicates": [_row(key) for key in duplicates],
        "duplicate_order": duplicate_order,
        "invalid_seeds": invalid_seeds,
        "duplicate_seeds": duplicate_seeds,
        "invalid": invalid,
        "invalid_order": invalid_order,
    }


def _report_realized_run_order(out_dir: pathlib.Path, tag: str, runs,
                               order, cells, seeds, cell_of) -> dict:
    """Persist and warn about the realised schedule audit."""
    audit = _realized_order_audit(runs, order, cells, seeds, cell_of)
    payload = {"experiment": tag, **audit}
    _write_json(pathlib.Path(out_dir) / f"{tag}_run_order_realized.json",
                payload)
    if not audit["matches_schedule"]:
        print(f"[{tag}] WARNING: realised run order deviates from the "
              f"frozen complete-block schedule "
              f"(missing={len(audit['missing'])}, "
              f"unexpected={len(audit['unexpected'])}, "
              f"duplicates={len(audit['duplicates'])}, "
              f"invalid={len(audit['invalid'])}, "
              f"invalid_schedule={len(audit['invalid_order'])}); "
              f"treatment arms may be aliased with run time")
    else:
        print(f"[{tag}] realised run order matches the frozen complete-block "
              f"schedule ({audit['n_observed']} runs)")
    return payload


def _mean_std(values):
    try:
        iterator = iter(values)
    except TypeError:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    vals = [float(v) for v in iterator if _finite(v)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = sum(vals) / len(vals)
    if not math.isfinite(mean):
        # The sample mean of finite values can still overflow an intermediate
        # sum (e.g. two 1e308 observations whose true mean is 1e308).  Scale
        # each term before summing; this cannot overflow.
        try:
            mean = math.fsum(v / len(vals) for v in vals)
        except (OverflowError, ValueError):
            mean = float("nan")
    if len(vals) > 1:
        try:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        except OverflowError:
            var = float("inf")
        if (not math.isfinite(var)
                or (var == 0.0 and any(v != mean for v in vals))):
            # Same scaling issue for the squared deviations, in either
            # direction: factor out the largest deviation so a true finite
            # SD remains representable when the raw squares overflow or
            # underflow.
            scale = max(abs(v - mean) for v in vals)
            if scale > 0.0 and math.isfinite(scale):
                scaled_var = (sum(((v - mean) / scale) ** 2 for v in vals)
                              / (len(vals) - 1))
                std = scale * math.sqrt(scaled_var)
            else:
                # The deviation itself can overflow even when the true SD
                # is representable (one huge outlier among many modest
                # values).  Rescale the original values instead, which keeps
                # every deviation bounded and recovers the finite SD.
                base = max(abs(v) for v in vals)
                if base > 0.0 and math.isfinite(base):
                    scaled = [v / base for v in vals]
                    scaled_mean = math.fsum(scaled) / len(vals)
                    scaled_var = (
                        math.fsum((v - scaled_mean) ** 2 for v in scaled)
                        / (len(vals) - 1))
                    std = base * math.sqrt(scaled_var)
                else:
                    std = float("nan")
        else:
            std = math.sqrt(var)
    else:
        std = 0.0
    return {"mean": mean, "std": std, "n": len(vals)}


def _linear_slope(xs, ys):
    """Least-squares slope of ``ys`` on ``xs`` (NaN when not identifiable).

    Used for the E2 degradation-shape readout: fitting accuracy on
    ``log(T)`` across the whole sweep tests "degrades more slowly" with all
    the measured lengths instead of only the endpoint, and is scale-free
    with respect to the absolute accuracy level.  Non-finite pairs are
    dropped; fewer than two points or a constant x returns NaN rather than
    an arbitrary number.
    """
    try:
        xs = list(xs)
        ys = list(ys)
    except TypeError:
        return float("nan")
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)
             if _finite(x) and _finite(y)]
    if len(pairs) < 2:
        return float("nan")
    n = len(pairs)
    # Translate to the first point before centring so a large common offset
    # does not destroy the differences (``1e308`` and its next float differ
    # by only ~2e-16 in relative terms).  If the raw differences overflow
    # (opposite-sign extremes), normalise both axes first and translate
    # those bounded values instead.
    x0, y0 = pairs[0]
    dx = [x - x0 for x, _ in pairs]
    dy = [y - y0 for _, y in pairs]
    x_base = y_base = 1.0
    if not (all(math.isfinite(value) for value in dx)
            and all(math.isfinite(value) for value in dy)):
        x_base = max(abs(x) for x, _ in pairs)
        y_base = max(abs(y) for _, y in pairs)
        if x_base == 0.0:
            return float("nan")
        xs_norm = [x / x_base for x, _ in pairs]
        ys_norm = ([0.0] * n if y_base == 0.0
                   else [y / y_base for _, y in pairs])
        dx = [x - xs_norm[0] for x in xs_norm]
        dy = [y - ys_norm[0] for y in ys_norm]
    x_scale = max(abs(value) for value in dx)
    y_scale = max(abs(value) for value in dy)
    if x_scale == 0.0:
        return float("nan")
    if y_scale == 0.0:
        return 0.0
    dx_norm = [value / x_scale for value in dx]
    dy_norm = [value / y_scale for value in dy]
    # Translating to the first pair removes the common offset; least squares
    # still needs the *mean* of the translated coordinates, otherwise the
    # fitted line is forced through the first observation.  Centring the
    # bounded (normalised) values keeps the sums representable even when the
    # raw offsets are extreme.
    x_mean = math.fsum(dx_norm) / n
    y_mean = math.fsum(dy_norm) / n
    dx_centered = [value - x_mean for value in dx_norm]
    dy_centered = [value - y_mean for value in dy_norm]
    denom = math.fsum(value * value for value in dx_centered)
    if denom <= 0.0:
        return float("nan")
    slope_norm = math.fsum(
        x * y for x, y in zip(dx_centered, dy_centered)) / denom
    if slope_norm == 0.0:
        return 0.0
    if not math.isfinite(slope_norm):
        return math.copysign(float("inf"), slope_norm)
    m1, e1 = math.frexp(slope_norm)
    m2, e2 = math.frexp(y_scale)
    m2b, e2b = math.frexp(y_base)
    m3, e3 = math.frexp(x_scale)
    m3b, e3b = math.frexp(x_base)
    mantissa = (m1 * m2 * m2b) / (m3 * m3b)
    try:
        return math.ldexp(
            mantissa, e1 + e2 + e2b - e3 - e3b)
    except OverflowError:
        return math.copysign(float("inf"), slope_norm)


def _difficulty_count(value) -> int | None:
    """One non-negative integer count from a recorded stratum row.

    ``bool`` and fractional values are rejected rather than truncated: a
    count that is not a whole number cannot describe generated samples.
    JSON round-trips may turn an integer into ``3.0``, so integral floats
    are accepted.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) and value == int(value):
        return int(value) if value >= 0 else None
    return None


def _length_entry(mapping, length):
    """Read a per-length record keyed by either an int or its JSON string.

    ``json.dumps`` turns the integer ``n_pairs`` keys of
    ``metrics_by_length`` / ``accuracy_by_length`` into strings, so a
    record that has been round-tripped through JSON would otherwise look
    empty to a companion reading the in-memory integer key.  Prefer the
    integer key when both exist (the in-memory representation) and fall
    back to the canonical string form.
    """
    if not isinstance(mapping, Mapping):
        return None
    if length in mapping:
        return mapping[length]
    return mapping.get(str(length))


def _length_mapping(mapping, length):
    """Per-length record as a mapping, or ``None`` when malformed.

    ``_length_entry`` already resolves the int/JSON-string key.  This helper
    additionally rejects a value that is not a mapping: reporting companions
    must abstain on a corrupted per-length row instead of raising
    ``AttributeError`` halfway through a results table.
    """
    entry = _length_entry(mapping, length)
    return entry if isinstance(entry, Mapping) else None


def _positive_lengths(lengths) -> list[int]:
    """Materialise a length sweep, dropping entries that are not positive ints."""
    out = []
    for value in _materialize_runs(lengths):
        length = _positive_integer(value)
        if length is not None:
            out.append(length)
    return out


def _difficulty_vectors(difficulty):
    """Extract ``(totals, corrects)`` from an ``eval_metrics`` difficulty block.

    Returns two equal-length integer lists **indexed by stratum id**, so a
    record whose ``strata`` rows are shuffled, or that only stores the
    observed strata, still lines its counts up with the reference mix.  A
    missing or malformed block yields empty lists so a companion can
    abstain instead of inventing a composition.  Legacy blocks without
    ``id`` entries fall back to row order (their only available encoding).
    """
    if not isinstance(difficulty, dict):
        return [], []
    rows = difficulty.get("strata") or []
    if not isinstance(rows, (list, tuple)) or not rows:
        return [], []
    rows = list(rows)
    if not all(isinstance(row, dict) for row in rows):
        return [], []
    has_id = ["id" in row for row in rows]
    if any(has_id) and not all(has_id):
        return [], []
    if all(has_id):
        ids = []
        for row in rows:
            # Use the same JSON-round-trip rule as the counts: an integral
            # float such as ``2.0`` is a valid serialisation of stratum id
            # 2, while bool/fractional/negative values are not.
            sid = _difficulty_count(row["id"])
            if sid is None:
                return [], []
            ids.append(sid)
        if len(set(ids)) != len(ids):
            return [], []
        # ``eval_metrics`` also records the declared encoding size.  Use it
        # so a compact record that stores only observed strata still keeps
        # its ids aligned with a full reference encoding (the missing ids
        # are genuine zero-count strata, not a different encoding).
        size = max(ids) + 1
        declared = (difficulty or {}).get("n_strata")
        if declared is not None:
            # The declared encoding size follows the same JSON convention as
            # the counts: an integral float (e.g. ``12.0``) is a valid
            # round-trip of an integer, while a bool/fractional/negative
            # value is malformed and must not be silently ignored.
            declared = _difficulty_count(declared)
            if declared is None or declared < size:
                return [], []
            size = declared
        if size > _MAX_DIFFICULTY_STRATA:
            return [], []
        totals = [0] * size
        correct = [0] * size
        for sid, row in zip(ids, rows):
            if "n" not in row or "correct" not in row:
                # A partially recorded row cannot be completed by assuming
                # the missing count is zero: that would silently fabricate a
                # perfectly failed (or empty) stratum and bias the
                # standardized rate.
                return [], []
            total = _difficulty_count(row["n"])
            hit = _difficulty_count(row["correct"])
            if total is None or hit is None or hit > total:
                return [], []
            totals[sid] += total
            correct[sid] += hit
        return totals, correct
    totals, correct = [], []
    for row in rows:
        if "n" not in row or "correct" not in row:
            return [], []
        total = _difficulty_count(row["n"])
        hit = _difficulty_count(row["correct"])
        if total is None or hit is None or hit > total:
            return [], []
        totals.append(total)
        correct.append(hit)
    return totals, correct


def _stratum_mix_tv(reference, other):
    """Total-variation distance between two difficulty-stratum mixes.

    ``0`` means the two evaluation streams have exactly the same
    retrieval-position x ambiguity composition; a positive value is the
    share of probability mass that moved.  This is the direct measure of
    the confound the standardized readout adjusts for, so it is reported
    beside every standardized contrast.
    """
    ref_totals, _ = _difficulty_vectors(reference)
    other_totals, _ = _difficulty_vectors(other)
    if (not ref_totals or len(ref_totals) != len(other_totals)
            or sum(ref_totals) <= 0 or sum(other_totals) <= 0):
        return float("nan")
    ref_sum, other_sum = sum(ref_totals), sum(other_totals)
    return 0.5 * sum(abs(a / ref_sum - b / other_sum)
                     for a, b in zip(ref_totals, other_totals))


def _stamp_difficulty_standardization(rec, base_length, lengths):
    """Add composition-standardized accuracy fields to one run record.

    ``rec["metrics_by_length"][L]["difficulty"]`` carries the per-stratum
    counts from ``eval_metrics``.  Reweighting each length to the
    base-length stratum mix is a pure function of those counts, so it is a
    separate helper (and is unit-tested) rather than an untested inline
    block in the E2 loop.

    The fields are exploratory companions: the pre-registered H2 decisions
    keep reading the raw ``accuracy_*`` / ``accuracy_drop`` values.
    """
    if not isinstance(rec, Mapping):
        # Not a run record at all; there is nothing to stamp.
        return rec
    lengths = _positive_lengths(lengths)
    # A legacy/malformed record may lack the per-length table (or carry a
    # non-dict placeholder).  The companion then abstains with NaN fields
    # instead of raising in the middle of the E2 loop.
    metrics_by_length = rec.get("metrics_by_length")
    if not isinstance(metrics_by_length, dict):
        metrics_by_length = {}

    def _block(length):
        entry = _length_mapping(metrics_by_length, length)
        return entry.get("difficulty") if entry is not None else None

    def _bucket(name):
        bucket = rec.get(name)
        if not isinstance(bucket, dict):
            bucket = {}
            rec[name] = bucket
        return bucket

    base_difficulty = _block(base_length)
    reference_totals, _ = _difficulty_vectors(base_difficulty)
    base_std = float("nan")
    if not lengths:
        # No sweep: still record the (undefined) summary fields rather than
        # indexing ``lengths[-1]`` and aborting the caller.
        for name in ("difficulty_standardized_accuracy_by_length",
                     "difficulty_standardized_coverage_by_length",
                     "difficulty_standardized_effective_n_by_length",
                     "difficulty_standardized_min_stratum_n_by_length",
                     "difficulty_mix_tv_by_length"):
            _bucket(name)
        rec["difficulty_standardized_accuracy_at_base"] = base_std
        for key in ("difficulty_standardized_accuracy_at_longest",
                    "difficulty_standardized_accuracy_drop",
                    "difficulty_standardized_accuracy_slope_log_len",
                    "difficulty_standardized_coverage_at_longest",
                    "difficulty_standardized_effective_n_at_longest",
                    "difficulty_mix_tv_at_longest"):
            rec[key] = float("nan")
        rec["difficulty_standardized_min_stratum_n_at_longest"] = 0
        return rec
    for length in lengths:
        block = _block(length)
        totals, correct = _difficulty_vectors(block)
        standardized = standardized_proportion(correct, totals,
                                               reference_totals)
        _bucket("difficulty_standardized_accuracy_by_length")[length] = (
            standardized["rate"])
        _bucket("difficulty_standardized_coverage_by_length")[length] = (
            standardized["coverage"])
        _bucket("difficulty_standardized_effective_n_by_length")[length] = (
            standardized["effective_n"])
        _bucket("difficulty_standardized_min_stratum_n_by_length")[length] = (
            standardized["min_stratum_n"])
        _bucket("difficulty_mix_tv_by_length")[length] = (
            _stratum_mix_tv(base_difficulty, block))
        if length == base_length:
            base_std = standardized["rate"]
    long_std = rec.get("difficulty_standardized_accuracy_by_length", {}).get(
        lengths[-1], float("nan"))
    rec["difficulty_standardized_accuracy_at_base"] = base_std
    rec["difficulty_standardized_accuracy_at_longest"] = long_std
    rec["difficulty_standardized_accuracy_drop"] = base_std - long_std
    rec["difficulty_standardized_accuracy_slope_log_len"] = _linear_slope(
        [math.log(max(2 * L + 2, 2)) for L in lengths],
        [rec["difficulty_standardized_accuracy_by_length"].get(L)
         for L in lengths])
    rec["difficulty_standardized_coverage_at_longest"] = rec.get(
        "difficulty_standardized_coverage_by_length", {}).get(
            lengths[-1], float("nan"))
    rec["difficulty_standardized_effective_n_at_longest"] = rec.get(
        "difficulty_standardized_effective_n_by_length", {}).get(
            lengths[-1], float("nan"))
    rec["difficulty_standardized_min_stratum_n_at_longest"] = rec.get(
        "difficulty_standardized_min_stratum_n_by_length", {}).get(
            lengths[-1], 0)
    rec["difficulty_mix_tv_at_longest"] = rec.get(
        "difficulty_mix_tv_by_length", {}).get(lengths[-1], float("nan"))
    return rec


def _stamp_linked_length_extension(rec, model, settings, lengths, eval_seed):
    """Add the within-item length-extension companion to one E2 record.

    The ordinary sweep generates a new stream at every length, so a raw
    accuracy drop mixes length with a changed item/retrieval-position
    composition.  This
    helper evaluates each *same* base item after appending fresh distractors
    (:func:`cage.train.eval_linked_lengths`) and stamps the paired
    accuracy / retention / McNemar fields.  It is isolated from the E2 loop
    so the exact record contract is unit-tested and a malformed sweep can
    abstain instead of aborting the runner.

    The companion is exploratory: the pre-registered H2 decision still
    reads the raw ``accuracy_*`` fields.
    """
    if not isinstance(rec, Mapping):
        # No run record to stamp; do not fabricate a companion payload.
        return None
    lengths = _positive_lengths(lengths)
    if not lengths:
        rec["linked_length_extension"] = None
        rec["linked_accuracy_by_length"] = {}
        rec["linked_accuracy_at_base"] = float("nan")
        rec["linked_accuracy_at_longest"] = float("nan")
        rec["linked_accuracy_drop"] = float("nan")
        rec["linked_retention_at_longest"] = None
        rec["linked_rescue_at_longest"] = None
        rec["linked_net_change_at_longest"] = None
        rec["linked_mcnemar_p_at_longest"] = float("nan")
        rec["linked_composition_effect_at_longest"] = float("nan")
        rec["linked_base_matches_raw"] = False
        return None
    linked = eval_linked_lengths(
        model, settings.n_pairs, settings.vocab, settings.device,
        lengths=lengths, batch=settings.eval_batch,
        batches=settings.eval_batches, seed=eval_seed)
    longest = lengths[-1]
    base = settings.n_pairs
    rec["linked_length_extension"] = linked
    rec["linked_accuracy_by_length"] = linked["accuracy_by_length"]
    rec["linked_accuracy_at_base"] = linked["accuracy_by_length"].get(base)
    rec["linked_accuracy_at_longest"] = linked["accuracy_by_length"].get(
        longest)
    rec["linked_accuracy_drop"] = (
        rec["linked_accuracy_at_base"] - rec["linked_accuracy_at_longest"]
        if _finite(rec["linked_accuracy_at_base"])
        and _finite(rec["linked_accuracy_at_longest"]) else float("nan"))
    rec["linked_retention_at_longest"] = linked["retention_by_length"].get(
        longest)
    rec["linked_rescue_at_longest"] = linked["rescue_by_length"].get(longest)
    rec["linked_net_change_at_longest"] = linked["net_change_by_length"].get(
        longest)
    rec["linked_mcnemar_p_at_longest"] = linked["mcnemar_by_length"].get(
        longest, {}).get("p_exact")
    raw_base = _length_entry(rec.get("accuracy_by_length"), base)
    raw_long = _length_entry(rec.get("accuracy_by_length"), longest)
    raw_drop = (raw_base - raw_long
                if _finite(raw_base) and _finite(raw_long)
                else float("nan"))
    rec["linked_composition_effect_at_longest"] = (
        raw_drop - rec["linked_accuracy_drop"]
        if _finite(raw_drop) and _finite(rec["linked_accuracy_drop"])
        else float("nan"))
    rec["linked_base_matches_raw"] = bool(
        _finite(rec["linked_accuracy_at_base"]) and _finite(raw_base)
        and abs(rec["linked_accuracy_at_base"] - raw_base) <= 1e-12)
    if not rec["linked_base_matches_raw"]:
        print(f"[e2] WARNING: linked base accuracy "
              f"{rec['linked_accuracy_at_base']!r} does not reproduce the "
              f"raw base accuracy {raw_base!r}; the paired length companion "
              f"is not comparable to the raw sweep")
    return linked


def _stamp_calib_term_knockout(rec, model, settings, lengths, eval_seed):
    """Add the within-model calibration-term knockout to one run record.

    The retrained E2 term ablations change both the architecture and the
    optimisation trajectory.  This stamp holds the trained full-calibration
    weights fixed and zeroes ``calib_a`` / ``calib_c`` at evaluation time
    on a paired stream, isolating how much of the *learned* correction is
    used.  The helper is isolated from the runner so the record contract
    can be unit-tested and malformed sweeps abstain instead of aborting.

    Exploratory companion: the pre-registered H2 decisions still read the
    retrained-arm contrasts.
    """
    if not isinstance(rec, Mapping):
        return None
    lengths = list(dict.fromkeys(_positive_lengths(lengths)))
    if not lengths:
        rec["calib_term_knockout"] = {
            "applicable": False,
            "reason": "empty length list",
            "lengths": [],
        }
        return None
    terms = ("length", "concentration", "both")
    by_length = {
        str(length): calib_term_knockout_eval(
            model, length, settings.vocab, settings.device,
            batch=settings.eval_batch, batches=settings.eval_batches,
            seed=eval_seed, terms=terms)
        for length in lengths
    }
    base_length = lengths[0]
    longest = lengths[-1]

    def _entry(length):
        return _length_mapping(by_length, length) or {}

    def _term_field(length, term, field):
        row = ((_entry(length).get("by_term") or {}).get(term) or {})
        value = row.get(field)
        return float(value) if _finite(value) else float("nan")

    def _baseline(length):
        value = _entry(length).get("baseline")
        return float(value) if _finite(value) else float("nan")

    for term in terms:
        for label, length in (("base", base_length),
                              ("longest", longest)):
            rec[f"calib_knockout_{term}_accuracy_at_{label}"] = _term_field(
                length, term, "accuracy")
            rec[f"calib_knockout_{term}_delta_at_{label}"] = _term_field(
                length, term, "delta")
    for label, length in (("base", base_length), ("longest", longest)):
        rec[f"calib_knockout_baseline_at_{label}"] = _baseline(length)
    raw_base = _length_entry(rec.get("accuracy_by_length"), base_length)
    eval_base = _baseline(base_length)
    matches = bool(
        _finite(raw_base) and _finite(eval_base)
        and abs(float(eval_base) - float(raw_base)) <= 1e-12)
    rec["calib_knockout_base_matches_raw"] = matches
    if _finite(raw_base) and not matches:
        print(f"[e2] WARNING: calibration-knockout baseline {eval_base!r} "
              f"does not reproduce the raw base accuracy {raw_base!r}; the "
              f"within-model probe is not on the same stream")
    rec["calib_term_knockout"] = {
        "applicable": any(entry.get("applicable")
                          for entry in by_length.values()),
        "lengths": [int(length) for length in lengths],
        "base_length": int(base_length),
        "longest": int(longest),
        "by_length": by_length,
        "base_matches_raw": matches,
        "terms": list(terms),
        "note": ("within-model coefficient knockout on the same sealed eval "
                 "stream; exploratory complement to the retrained term "
                 "ablations, not a pre-registered H2 decision row"),
    }
    return rec["calib_term_knockout"]


def _materialize_runs(runs):
    """Materialize a run container; a non-iterable one has no records."""
    try:
        return list(runs)
    except TypeError:
        return []


def _aggregate_runs(runs, skip=("seed", "lr", "params", "eval_seed",
                                "stats_seed")):
    records = [r for r in _materialize_runs(runs) if isinstance(r, Mapping)]
    keys = sorted({k for r in records for k, v in r.items()
                   if _finite(v) and k not in skip})
    return {k: _mean_std([r[k] for r in records if k in r]) for k in keys}


# Recipe fields that the controlled-comparison contract requires to be
# *identical* on both sides of every contrast (see ``_controlled_audit``):
# optimizer-relevant learning rate, schedule length, batch geometry, data
# sizes and every evaluation-stream size.  ``config`` is the resolved
# Settings snapshot stored with each run record.  The per-run evaluation
# stream *seeds* are not Settings fields; the run records stamp them and
# ``_recipe_view`` adds them to the audited view, so "same stream size on
# a different stream" is caught too.
_AUDIT_RECIPE_KEYS = (
    "device", "d_model", "n_heads", "n_layers", "max_len", "vocab", "batch",
    "n_pairs", "steps", "eval_batch", "eval_batches", "stats_batch",
    "stats_batches", "val_batches", "curve_points", "fdr_q", "dropout",
    "deterministic",
)

# Record fields that describe the module under test.  They are the only
# dimensions a contrast may vary; every other recorded design setting has to
# match across the two sides.
_AUDIT_FACTOR_KEYS = ("gate", "norm", "layer_scale", "calib_variant")

# The resolved-``Cfg`` keys a contrast is allowed to vary.  When both sides
# carry the ``model_cfg`` the model was actually built from, every *other*
# key must match: a difference there is an undeclared confound no matter
# what the labels say.
_AUDIT_MODEL_FACTOR_KEYS = ("gate", "norm", "layer_scale",
                            "calib_use_len", "calib_use_ent",
                            "calib_shuffle_ent", "calib_ent_norm")

# Ground truth of the calib gate's additive terms, keyed by the resolved
# ``(calib_use_len, calib_use_ent, calib_shuffle_ent)`` flags of the model
# that trained: the cell of the 2x2 "length term x concentration term"
# design a run is in, plus the E1 active-control (shuffled entropy) cell.
# Cross-checking the declared ``calib_variant`` label against the flags the
# model was actually built with catches a mislabelled or mis-wired arm.
_CALIB_FACTORIAL = {
    (True, True, False): "full",
    (True, False, False): "calib_len_only",
    (False, True, False): "calib_ent_only",
    (False, False, False): "calib_no_terms",
    (True, True, True): "calib_shuf_ent",
}

_CALIB_FACTORIAL_CELLS = ("full", "calib_len_only", "calib_ent_only",
                          "calib_no_terms")

# The two axes of the calibration-term 2x2.  A main effect averages the
# factor's effect over the two levels of the other term, so the "on"/"off"
# cells and the two within-level pairwise audits are both part of the
# contract.  ``model_key`` is the resolved Cfg flag that must be the only
# model-config difference in each pair.
_FACTORIAL_MAIN_EFFECTS = {
    "length": {
        "on": ("full", "calib_len_only"),
        "off": ("calib_ent_only", "calib_no_terms"),
        # Within each concentration level, the pair that differs only in
        # the length flag: full-vs-ent_only (entropy on) and
        # len_only-vs-no_terms (entropy off).
        "pairs": (("full", "calib_ent_only"),
                  ("calib_len_only", "calib_no_terms")),
        "model_key": "calib_use_len",
    },
    "concentration": {
        "on": ("full", "calib_ent_only"),
        "off": ("calib_len_only", "calib_no_terms"),
        # Within each length level, the pair that differs only in the
        # concentration flag: full-vs-len_only and ent_only-vs-no_terms.
        "pairs": (("full", "calib_len_only"),
                  ("calib_ent_only", "calib_no_terms")),
        "model_key": "calib_use_ent",
    },
}

_CALIB_PLACEBO_CELL = "calib_shuf_ent"

# E2 normalisation-ablation cell (H2g): the full calib architecture and both
# additive terms, but the concentration term reads the raw row-mean entropy
# instead of the registered H/log(T).  It is a one-factor ablation of the
# normalisation itself (same parameters, same length term, same kernel), kept
# outside the 2x2 term factorisation because at a fixed length the two forms
# differ only by a constant absorbed into ``calib_c`` - it is identifiable
# only across the length sweep.
_CALIB_RAW_ENT_CELL = "calib_raw_ent"

# Factor-vector expansion of a declared ``calib_variant`` cell label.  New
# records carry the resolved ``model_cfg`` (the ground truth), but pooled or
# legacy records may carry only the label; expanding the label here keeps the
# single-factor audit meaningful for them too.  ``calib_variant`` names a
# *cell*, not one factor: e.g. ``calib_no_terms`` turns off both additive
# terms, while ``calib_raw_ent`` differs from ``calib`` only in the entropy
# normalisation.
_CALIB_CELL_FACTORS = {
    cell: {"calib_use_len": bool(flags[0]),
           "calib_use_ent": bool(flags[1]),
           "calib_shuffle_ent": bool(flags[2]),
           "calib_ent_norm": True}
    for flags, cell in _CALIB_FACTORIAL.items()
}
_CALIB_CELL_FACTORS[_CALIB_RAW_ENT_CELL] = {
    "calib_use_len": True, "calib_use_ent": True,
    "calib_shuffle_ent": False, "calib_ent_norm": False}


def _calib_cell(run):
    """The calibration-term cell a run actually trained (None when N/A).

    Uses the resolved ``model_cfg`` when present (ground truth); falls back
    to the declared label for records that predate the stamp.  A shuffle
    flag combined with anything but the full term set is not a declared
    cell: it returns a non-matching name so the label audit fails loudly
    instead of crediting a mis-wired placebo.  Likewise, the raw-entropy
    normalisation ablation is a declared cell only when both additive terms
    are on and the entropy is not shuffled (the flags that define H2g's
    single-factor arm).
    """
    if not isinstance(run, Mapping):
        # A malformed pooled row cannot name a trained cell; reporting it
        # here instead of raising lets the coverage audit flag the missing
        # cell and continue with the remaining evidence.
        return None
    cfg = run.get("model_cfg")
    if cfg is not None:
        if not isinstance(cfg, Mapping):
            return "model_cfg_malformed"
        if cfg.get("gate") != "calib":
            return None
        # The resolved model config is JSON provenance: these flags are
        # booleans, and a string such as ``"False"`` is truthy.  Coercing
        # it with ``bool`` would silently read a malformed record as the
        # opposite arm and could certify the contrast.  Fail closed with a
        # non-cell sentinel so the label cross-check below reports it.
        raw_flags = (cfg.get("calib_use_len", True),
                     cfg.get("calib_use_ent", True),
                     cfg.get("calib_shuffle_ent", False))
        raw_ent_norm = cfg.get("calib_ent_norm", True)
        if any(not isinstance(value, bool) for value in raw_flags) \
                or not isinstance(raw_ent_norm, bool):
            return "model_cfg_malformed"
        flags = raw_flags
        if not raw_ent_norm:
            return (_CALIB_RAW_ENT_CELL
                    if flags == (True, True, False)
                    else "calib_raw_ent_undeclared")
        cell = _CALIB_FACTORIAL.get(flags)
        if cell is None and flags[2]:
            return "calib_shuf_ent_undeclared"
        return cell
    label = run.get("calib_variant")
    if label is None or label == "":
        return None
    if not isinstance(label, str):
        # Labels are consumed as set/dict keys by the coverage and audit
        # helpers; a list/dict label would raise TypeError and abort the
        # whole contrast instead of failing closed.
        return "calib_variant_malformed"
    return label


def _coverage_cell_key(cell) -> str:
    """JSON-safe key for a design cell (tuples become ``a|b|c``)."""
    if isinstance(cell, tuple):
        return "|".join(str(part) for part in cell)
    return str(cell)


def _cell_seed_balance(runs, cell_of, expected, planned_seeds):
    """Seeds per design cell vs the planned training-seed set.

    Presence-only coverage can call a factorial complete while one cell
    silently lost seeds: cell membership would then correlate with seed
    availability, so that cell's mean is read on a different sample than
    its neighbours' - a confound no per-contrast recipe audit sees if the
    affected contrast is not pre-registered (e.g. an E3 cell that only
    enters exploratory reads).  Returns the per-cell seed payload and the
    list of unbalanced cell keys.
    """
    try:
        planned = (sorted({int(s) for s in planned_seeds if
                           isinstance(s, int) and not isinstance(s, bool)})
                   if planned_seeds else [])
    except TypeError:
        # A non-iterable planned set is not a seed plan; the resulting
        # empty plan makes every cell unbalanced (fail closed) rather than
        # crashing the coverage report.
        planned = []
    seen: dict[str, set] = {}
    for run in _materialize_runs(runs):
        # Coverage is evidence about real runs: a non-mapping row, a cell
        # extractor that raises, or a malformed seed cannot certify that a
        # planned cell/seed was trained.  Skip it so the missing seed stays
        # visible as unbalanced instead of being silently coerced.
        if not isinstance(run, Mapping):
            continue
        try:
            cell = cell_of(run)
        except Exception:  # noqa: BLE001 - malformed rows are not evidence
            continue
        seed = run.get("seed")
        if cell is None or isinstance(seed, bool) \
                or not isinstance(seed, int):
            continue
        seen.setdefault(_coverage_cell_key(cell), set()).add(int(seed))
    balance = {}
    unbalanced = []
    for cell in expected:
        key = _coverage_cell_key(cell)
        seeds = sorted(seen.get(key, set()))
        ok = bool(planned) and seeds == planned
        balance[key] = {"seeds": seeds, "planned": planned, "balanced": ok}
        if not ok:
            unbalanced.append(key)
    return balance, unbalanced


def _report_ablation_coverage(out_dir, tag, runs, cells=_CALIB_FACTORIAL_CELLS,
                              planned_seeds=None, design=None):
    """Verify a declared factorial ablation ran every cell of its design.

    A "each component is doing work" claim over a 2x2 (or leave-one-out)
    design is only supported if every cell was actually trained; a missing
    cell silently leaves one effect unmeasured, and a cell that trained
    only part of the planned seed set is read on a different sample than
    the others (see ``_cell_seed_balance``).  The check reads the cell each
    run trained (resolved flags, not labels), prints a warning listing the
    missing or unbalanced cells, and stores
    ``<tag>_ablation_coverage.json`` next to the run records.
    """
    runs = _materialize_runs(runs)
    cells = _materialize_runs(cells)
    present = sorted({c for c in (_calib_cell(r) for r in runs) if c in cells})
    missing = [c for c in cells if c not in present]
    balance, unbalanced = _cell_seed_balance(runs, _calib_cell, cells,
                                             planned_seeds)
    payload = {
        "tag": tag,
        "design": design or "calib_use_len x calib_use_ent (2x2)",
        "expected_cells": list(cells),
        "present_cells": present,
        "missing_cells": missing,
        "complete": bool(cells) and not missing,
        "cell_seed_balance": balance,
        "unbalanced_cells": unbalanced,
        "seed_balanced": bool(cells) and not unbalanced,
        "n_runs": len(runs),
    }
    if not cells:
        print(f"[{tag}] WARNING: calibration-term ablation coverage has no "
              f"declared cells; coverage cannot be certified")
    elif missing:
        print(f"[{tag}] WARNING: calibration-term ablation coverage "
              f"incomplete: {len(present)}/{len(cells)} cells trained "
              f"(missing {', '.join(missing)}); the corresponding term "
              f"effects are not measured")
    else:
        print(f"[{tag}] ablation coverage: {len(cells)}/{len(cells)} "
              f"calibration-term cells trained ({', '.join(present)})")
    if unbalanced:
        print(f"[{tag}] WARNING: ablation cell(s) {', '.join(unbalanced)} "
              f"did not train the full planned seed set "
              f"{list(planned_seeds or ())}: their means are read on a "
              f"different sample than the other cells'; the factorial is "
              f"unbalanced and those cell effects are confounded with seed "
              f"availability")
    _write_json(out_dir / f"{tag}_ablation_coverage.json", payload)
    return payload


def _report_design_coverage(out_dir, tag, runs, design, extract, expected,
                            planned_seeds=None):
    """Verify a declared factorial design trained every declared cell.

    Generalises the ablation-coverage check to whole experiment grids
    (E1's four gate arms, E2's eight arms, E3's 2x4x2 grid): an effect is
    only attributable if every cell it contrasts was actually trained, and
    a missing arm leaves that effect unmeasured - or worse, invites a
    claim from the cells that happen to have run.  The extractor reads the
    resolved per-run factors (not labels); missing cells and cells that
    trained only part of the planned seed set are printed as warnings and
    stored in ``<tag>_<design>_coverage.json``.
    """
    runs = _materialize_runs(runs)
    expected = _materialize_runs(expected)
    observed = set()
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        try:
            value = extract(run)
        except Exception:  # noqa: BLE001 - malformed rows are not evidence
            continue
        if value is None:
            continue
        try:
            observed.add(value)
        except TypeError:
            # An unhashable "cell" is not a declared design cell; skipping
            # it leaves the expected cell missing instead of aborting the
            # coverage report.
            continue
    present = [c for c in expected if any(c == o for o in observed)]
    missing = [c for c in expected if not any(c == p for p in present)]
    balance, unbalanced = _cell_seed_balance(runs, extract, expected,
                                             planned_seeds)
    payload = {
        "tag": tag,
        "design": design,
        "expected_cells": [list(c) if isinstance(c, tuple) else c
                           for c in expected],
        "present_cells": [list(c) if isinstance(c, tuple) else c
                          for c in present],
        "missing_cells": [list(c) if isinstance(c, tuple) else c
                          for c in missing],
        "complete": bool(expected) and not missing,
        "cell_seed_balance": balance,
        "unbalanced_cells": unbalanced,
        "seed_balanced": bool(expected) and not unbalanced,
        "n_runs": len(runs),
    }
    if not expected:
        print(f"[{tag}] WARNING: {design} coverage has no declared cells; "
              f"coverage cannot be certified")
    elif missing:
        print(f"[{tag}] WARNING: {design} coverage incomplete: "
              f"{len(present)}/{len(expected)} cells trained "
              f"(missing {missing}); those effects are not measured")
    else:
        print(f"[{tag}] design coverage: {len(expected)}/{len(expected)} "
              f"{design} cells trained")
    if unbalanced:
        print(f"[{tag}] WARNING: {design} cell(s) {', '.join(unbalanced)} "
              f"did not train the full planned seed set "
              f"{list(planned_seeds or ())}: the factorial is unbalanced "
              f"and those cell effects are confounded with seed "
              f"availability")
    _write_json(out_dir / f"{tag}_{design}_coverage.json", payload)
    return payload
