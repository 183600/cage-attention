# cage/experiments/_synthesis.py - Factorial effects, hypothesis synthesis and curve/selection diagnostics.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _counterfactual_companion(out_dir, tag, by_arm, lengths=None,
                              reference="none"):
    """Report the data-level minimal-pair retrieval probes per arm.

    ``eval_metrics`` edits each already-realized evaluation batch three ways
    (queried value swapped, a distractor value swapped, the queried pair
    relocated) and records how the model's answer responds.  This companion
    aggregates those rates across training seeds and, when a reference arm is
    present, reports the paired contrast in ``cf_value_follow_rate`` (the
    share of clean-correct rows whose answer follows the edited value).

    ``reference`` may be a fixed label (``"none"`` for E1/E2) or a callable
    mapping each arm label to its matched control (E3 uses the same
    ``norm`` / ``layer_scale`` cell with ``gate="query"``); a callable keeps
    the contrast within-cell instead of comparing against one global arm.

    The probes are *data-processing* manipulation checks: they reuse the
    finite stream, invent no new distribution, and are deterministic.  They
    stay outside the FDR family and cannot change a pre-registered verdict;
    their job is to prevent "accuracy improved" from being read as "the model
    retrieves the in-context value" when the model is actually memorising or
    position-copying.
    """
    def arm_label(label):
        if isinstance(label, str):
            return label
        try:
            iterator = iter(label)
        except TypeError:
            # An arm label is display metadata; a scalar label (e.g. an
            # integer gate id) is still printable instead of aborting the
            # whole companion.
            return str(label)
        parts = []
        for part in iterator:
            if part is None:
                parts.append("off")
            elif isinstance(part, float):
                parts.append(f"{part:g}")
            else:
                parts.append(str(part))
        return "|".join(parts)

    if lengths is not None:
        lengths = _positive_lengths(lengths)
    if not isinstance(by_arm, Mapping):
        # A malformed arm container names no cells; report an empty
        # companion instead of failing with an AttributeError.
        by_arm = {}

    def value(run, metric, length):
        if length is None:
            return run.get(metric)
        block = _length_mapping(run.get("metrics_by_length"), length) or {}
        return block.get(metric)

    arm_map = {arm_label(label): (label, _mapping_runs(runs))
               for label, runs in by_arm.items()}
    arms = {}
    for label, (_, runs) in arm_map.items():
        if lengths is None:
            entry = {"n_runs": len(runs)}
            for metric in _COUNTERFACTUAL_METRICS:
                entry[metric] = _mean_std(
                    [value(r, metric, None) for r in runs
                     if _finite(value(r, metric, None))])
            entry["clean_accuracy"] = _mean_std(
                [r.get("accuracy") for r in runs
                 if _finite(r.get("accuracy"))])
        else:
            entry = {"n_runs": len(runs), "by_length": {}}
            for length in lengths:
                entry["by_length"][str(length)] = {
                    metric: _mean_std(
                        [value(r, metric, length) for r in runs
                         if _finite(value(r, metric, length))])
                    for metric in _COUNTERFACTUAL_METRICS}
                entry["by_length"][str(length)]["clean_accuracy"] = _mean_std(
                    [value(r, "accuracy", length) for r in runs
                     if _finite(value(r, "accuracy", length))])
        arms[label] = entry

    contrasts = {}
    if callable(reference):
        ref_for = {}
        for label, (raw, _) in arm_map.items():
            target = arm_label(reference(raw))
            if target in arm_map and target != label:
                ref_for[label] = target
    elif reference is not None:
        ref_label = arm_label(reference)
        ref_for = ({label: ref_label for label in arm_map
                    if label != ref_label} if ref_label in arm_map else {})
    else:
        ref_for = {}
    for label, ref_label in ref_for.items():
        _, runs = arm_map[label]
        _, ref_runs = arm_map[ref_label]
        rule = f"{label} - {ref_label}"
        if lengths is None:
            row = {}
            for metric in ("cf_value_accuracy",
                           "cf_distractor_accuracy",
                           "cf_relocation_accuracy",
                           "cf_value_follow_rate",
                           "cf_distractor_unchanged_rate",
                           "cf_relocation_unchanged_rate"):
                ci = _paired_diff(runs, ref_runs, metric)
                row[metric] = {
                    "mean": ci.get("mean"), "lo": ci.get("lo"),
                    "hi": ci.get("hi"), "p_value": ci.get("p_value"),
                    "n": ci.get("n"),
                    "seeds_used": ci.get("seeds_used"),
                }
            contrasts[rule] = row
        else:
            by_length = {}
            for length in lengths:
                tagged_a = [
                    dict(r, **{
                        metric: value(r, metric, length)
                        for metric in _COUNTERFACTUAL_METRICS})
                    for r in runs]
                tagged_b = [
                    dict(r, **{
                        metric: value(r, metric, length)
                        for metric in _COUNTERFACTUAL_METRICS})
                    for r in ref_runs]
                ci = _paired_diff(
                    tagged_a, tagged_b, "cf_value_accuracy")
                by_length[str(length)] = {
                    "mean": ci.get("mean"), "lo": ci.get("lo"),
                    "hi": ci.get("hi"), "p_value": ci.get("p_value"),
                    "n": ci.get("n"),
                    "seeds_used": ci.get("seeds_used"),
                }
            contrasts[rule] = by_length

    payload = {
        "experiment": tag,
        "reference": ("per-arm control mapping" if callable(reference)
                      else reference),
        "lengths": None if lengths is None else [int(L) for L in lengths],
        "arms": arms,
        "contrasts": contrasts,
        "note": (
            "data-level minimal-pair probes. Cross-arm contrasts use the "
            "all-item cf_value_accuracy / cf_*_accuracy, which keep a fixed "
            "denominator; value_swap follow_rate and the *_unchanged_rate "
            "fields condition on model behaviour and are descriptive only. "
            "Outside the FDR family."),
    }
    _write_json(out_dir / f"{tag}_counterfactual.json", payload)
    print(f"\n[{tag}] counterfactual retrieval probes (data-level minimal "
          f"edits; means over training seeds, outside the FDR family):")
    print(f"{'arm':>22} {'clean_acc':>11} {'value_cf_acc':>13} "
          f"{'value_follow':>13} {'distractor_acc':>15} "
          f"{'relocation_acc':>15}")
    for label in arm_map:
        entry = arms[label]
        if lengths is None:
            cells = entry
        elif not lengths:
            cells = {}
        else:
            # Console summary uses the training length (first sweep point).
            cells = entry["by_length"][str(lengths[0])]
        def fmt(metric):
            value_ = (cells.get(metric) or {}).get("mean")
            return f"{value_:>13.4f}" if _finite(value_) else f"{'n/a':>13}"
        print(f"{label:>22} {fmt('clean_accuracy'):>11} "
              f"{fmt('cf_value_accuracy')} "
              f"{fmt('cf_value_follow_rate'):>13} "
              f"{fmt('cf_distractor_accuracy'):>15} "
              f"{fmt('cf_relocation_accuracy'):>15}")
    if contrasts and lengths is None:
        for rule, row in contrasts.items():
            ci = row["cf_value_accuracy"]
            print(f"      {rule} on value-swap accuracy: {_fmt_ci(ci)}")
    return payload


def _factorial_main_effect(cells, metric, factor):
    """Averaged main effect of one factor in the complete calibration 2x2.

    ``cells`` maps the four ``_CALIB_FACTORIAL_CELLS`` to their run lists.
    The contrast is the factor-on mean minus the factor-off mean, where each
    arm averages over both levels of the other factor: for ``length`` this
    is ``mean(full, calib_len_only) - mean(calib_ent_only, calib_no_terms)``.
    Unlike a leave-one-out contrast (which is conditional on the other term
    being on), this estimates the term's average contribution across the
    full factorial, so a component that only helps in one configuration is
    not credited as a general effect.

    The paired sample is the seeds where *all four* cells carry a finite
    value (the same seed must appear in every cell); the effective seed set
    and every dropped seed are recorded, including runs with no training
    seed (they cannot be paired).  The audit checks both within-level
    pairwise contrasts (the two edges that vary only ``factor``), requires
    the resolved model config to differ only in that factor's flag, and
    verifies one global recipe and one seed set across all four cells - a
    main effect averaged over a cell with a different optimizer or epoch
    step count would not be a factorial effect.
    """
    if factor not in _FACTORIAL_MAIN_EFFECTS:
        raise ValueError(
            f"unknown factorial factor {factor!r}; expected one of "
            f"{sorted(_FACTORIAL_MAIN_EFFECTS)}")
    spec = _FACTORIAL_MAIN_EFFECTS[factor]
    required = _CALIB_FACTORIAL_CELLS
    if not isinstance(cells, Mapping):
        raise ValueError(
            "factorial main effect needs a mapping of cell name to runs, "
            f"got {type(cells).__name__}")
    # Materialise each cell exactly once: a generator cell must not be
    # consumed by the missing-run scan and then look empty to the pairing
    # logic.
    cells = {c: _materialize_runs(cells.get(c)) for c in required}
    missing = [c for c in required if not cells.get(c)]
    if missing:
        raise ValueError(
            f"{factor} main effect needs all 2x2 cells; missing {missing}")
    missing_runs = [r for c in required for r in cells[c]
                    if _paired_seed(r) is None]
    by_cell = {}
    for c in required:
        keyed = {}
        for r in cells[c]:
            seed = _paired_seed(r)
            if seed is not None:
                keyed[seed] = r
        by_cell[c] = keyed
    planned = sorted(set().union(*(set(m) for m in by_cell.values())),
                     key=_seed_order)
    diffs, used, dropped, unpaired, undefined = [], [], [], [], []
    for seed in planned:
        rs = [by_cell[c].get(seed) for c in required]
        if any(r is None for r in rs):
            dropped.append(seed)
            unpaired.append(seed)
            continue
        vals = [r.get(metric) for r in rs]
        finite = [_finite(v) for v in vals]
        if not all(finite):
            dropped.append(seed)
            if any(finite):
                unpaired.append(seed)
            else:
                undefined.append(seed)
            continue
        used.append(seed)
        values = {c: float(by_cell[c][seed][metric]) for c in required}
        diffs.append(
            sum(values[c] for c in spec["on"]) / len(spec["on"])
            - sum(values[c] for c in spec["off"]) / len(spec["off"]))
    ci = paired_bootstrap_ci(diffs)
    on_vals = [float(by_cell[c][s][metric])
               for s in used for c in spec["on"]]
    off_vals = [float(by_cell[c][s][metric])
                for s in used for c in spec["off"]]
    ci["arms"] = {"a": _mean_std(on_vals), "b": _mean_std(off_vals)}
    ci["arms_note"] = (f"pooled {factor}-on vs {factor}-off cells on the "
                       f"paired seeds")
    ci["baseline_mean"] = ci["arms"]["b"]["mean"]
    ci["diffs"] = list(diffs)
    ci["curve_invalid_seeds"] = _curve_invalid_seeds_for(
        used, tuple(by_cell[c] for c in required))
    ci["seeds_used"] = sorted(used, key=_seed_order)
    ci["seeds_dropped"] = (sorted(dropped, key=_seed_order)
                           + [None] * len(missing_runs))
    ci["seeds_unpaired"] = (sorted(unpaired, key=_seed_order)
                            + [None] * len(missing_runs))
    ci["seeds_undefined"] = sorted(undefined, key=_seed_order)

    mismatches, factors = [], {}
    for left, right in spec["pairs"]:
        clean_left = [r for r in cells[left] if _paired_seed(r) is not None]
        clean_right = [r for r in cells[right] if _paired_seed(r) is not None]
        audit = _controlled_audit(clean_left, clean_right)
        if len(clean_left) != len(cells[left]) \
                or len(clean_right) != len(cells[right]):
            audit["mismatches"].append(
                f"{left} vs {right}: run(s) without a usable training "
                f"seed cannot be paired")
            audit["controlled"] = False
        mismatches.extend(audit.get("mismatches", []))
        factors.update(audit.get("varying_factors", {}))
        diff_keys = set((audit.get("model_factor_diffs") or {}).keys())
        if diff_keys:
            if diff_keys != {spec["model_key"]}:
                mismatches.append(
                    f"{left} vs {right}: expected only "
                    f"{spec['model_key']} to differ in the resolved model "
                    f"config, got {sorted(diff_keys)}")
        elif spec["model_key"] not in (audit.get("varying_factors") or {}):
            mismatches.append(
                f"{left} vs {right}: no audited {spec['model_key']} "
                f"difference between the cells")
    seed_sets = {c: frozenset(by_cell[c]) for c in required}
    if len(set(seed_sets.values())) != 1:
        mismatches.append(
            "2x2 cells do not share one seed set: "
            + ", ".join(f"{c}={sorted(s, key=_seed_order)}"
                        for c, s in seed_sets.items()))
    views = [(c, j, _recipe_view(r))
             for c in required for j, r in enumerate(cells[c])
             if isinstance(r, Mapping)]
    if views:
        first = views[0][2]
        for cell, j, view in views[1:]:
            diff = sorted(k for k in set(first) | set(view)
                          if view.get(k) != first.get(k))
            if diff:
                mismatches.append(
                    f"factorial recipe differs in cell {cell} run {j}: "
                    f"{diff}")
    ci["controlled"] = {"controlled": not mismatches,
                        "mismatches": mismatches,
                        "varying_factors": factors}
    return ci


def _grid_level_main_effect(metric, level_cells, baseline_cells):
    """Averaged main effect of one factor level, collapsed over controls.

    E3's grid is gate x norm x LayerScale, but the only registered norm
    contrast was ``seednorm - pre`` *within* one gate and LayerScale cell.
    An average-own-level effect (each norm level vs the ``pre`` baseline,
    collapsed over gate and LayerScale) is what answers "does this
    normalisation level contribute at all", not merely "does it differ in
    one control cell".  ``level_cells`` is the list of run lists carrying
    the level to test (one per control combination, e.g. for each gate and
    LayerScale), ``baseline_cells`` the matching ``pre``-level run lists in
    the same order.

    The paired sample is the seeds where *every* one of the level and
    baseline cells carries a finite value; the effect is the level mean
    minus the baseline mean within each seed, so both gate and LayerScale
    are averaged inside the paired difference (repeated-measures pairing:
    every control combination contributes exactly one level run and one
    baseline run per seed).  Effective seed set and dropped seeds are
    recorded (including runs with no training seed), and each cell pair is
    audited with the same
    controlled-variable contract as a marginal contrast: the two cells must
    vary only the ``norm`` factor and share one recipe.  A main effect
    averaged over a cell with a different optimizer or with a non-norm
    config difference would not be a factorial effect.
    """
    try:
        level_cells = list(level_cells)
    except TypeError as exc:
        raise ValueError(
            "grid-level main effect needs an iterable of level cell run "
            f"lists, got {type(level_cells).__name__}") from exc
    try:
        baseline_cells = list(baseline_cells)
    except TypeError as exc:
        raise ValueError(
            "grid-level main effect needs an iterable of baseline cell run "
            f"lists, got {type(baseline_cells).__name__}") from exc
    level_cells = [_materialize_runs(rs) for rs in level_cells]
    baseline_cells = [_materialize_runs(rs) for rs in baseline_cells]
    n = len(level_cells)
    if n == 0 or n != len(baseline_cells):
        raise ValueError(
            "grid-level main effect needs equally many level and baseline "
            f"cells (got {n} vs {len(baseline_cells)})")
    missing_runs = [r for rs in level_cells + baseline_cells for r in rs
                    if _paired_seed(r) is None]

    def _keyed(rs):
        out = {}
        for r in rs:
            seed = _paired_seed(r)
            if seed is not None:
                out[seed] = r
        return out

    maps = [_keyed(rs) for rs in level_cells]
    base_maps = [_keyed(rs) for rs in baseline_cells]
    planned = sorted(set().union(*(set(m) for m in maps + base_maps)),
                     key=_seed_order)
    diffs, used, dropped, unpaired, undefined = [], [], [], [], []
    for seed in planned:
        level_vals, base_vals = [], []
        present = True
        finite_any = False
        for m in maps:
            r = m.get(seed)
            if r is None:
                present = False
                break
            v = r.get(metric)
            level_vals.append(v)
            finite_any = finite_any or _finite(v)
        for m in base_maps:
            r = m.get(seed)
            if r is None:
                present = False
                break
            v = r.get(metric)
            base_vals.append(v)
            finite_any = finite_any or _finite(v)
        if not present:
            dropped.append(seed)
            unpaired.append(seed)
            continue
        if not all(_finite(v) for v in level_vals + base_vals):
            dropped.append(seed)
            (unpaired if finite_any else undefined).append(seed)
            continue
        used.append(seed)
        diffs.append(sum(float(v) for v in level_vals) / len(level_vals)
                     - sum(float(v) for v in base_vals) / len(base_vals))
    ci = paired_bootstrap_ci(diffs)
    level_pool = [float(m[s][metric]) for s in used for m in maps]
    base_pool = [float(m[s][metric]) for s in used for m in base_maps]
    ci["arms"] = {"a": _mean_std(level_pool), "b": _mean_std(base_pool)}
    ci["arms_note"] = ("level vs pre pooled over gate x LayerScale, on the "
                       "paired seeds")
    ci["baseline_mean"] = ci["arms"]["b"]["mean"]
    ci["diffs"] = list(diffs)
    ci["curve_invalid_seeds"] = _curve_invalid_seeds_for(
        used, tuple(maps + base_maps))
    ci["seeds_used"] = sorted(used, key=_seed_order)
    ci["seeds_dropped"] = (sorted(dropped, key=_seed_order)
                           + [None] * len(missing_runs))
    ci["seeds_unpaired"] = (sorted(unpaired, key=_seed_order)
                            + [None] * len(missing_runs))
    ci["seeds_undefined"] = sorted(undefined, key=_seed_order)

    mismatches, factors = [], {}
    for level_rs, base_rs in zip(level_cells, baseline_cells):
        clean_level = [r for r in level_rs if _paired_seed(r) is not None]
        clean_base = [r for r in base_rs if _paired_seed(r) is not None]
        audit = _controlled_audit(clean_level, clean_base)
        if len(clean_level) != len(level_rs) \
                or len(clean_base) != len(base_rs):
            audit["mismatches"].append(
                "norm level pair contains run(s) without a usable "
                "training seed")
            audit["controlled"] = False
        mismatches.extend(audit.get("mismatches", []))
        factors.update(audit.get("varying_factors", {}))
        diff_keys = set((audit.get("model_factor_diffs") or {}).keys())
        if diff_keys and diff_keys != {"norm"}:
            mismatches.append(
                f"norm level pair varies {sorted(diff_keys)} rather than "
                f"only the norm factor")
        elif not diff_keys and "norm" not in (
                audit.get("varying_factors") or {}):
            mismatches.append(
                "norm level pair shows no audited norm difference")
    seed_sets = {i: frozenset(m) for i, m in enumerate(maps + base_maps)}
    if len(set(seed_sets.values())) != 1:
        mismatches.append("grid cells do not share one seed set")
    views = [(i, j, _recipe_view(r))
             for i, rs in enumerate(level_cells + baseline_cells)
             for j, r in enumerate(rs) if isinstance(r, Mapping)]
    if views:
        first = views[0][2]
        for cell, j, view in views[1:]:
            diff = sorted(k for k in set(first) | set(view)
                          if view.get(k) != first.get(k))
            if diff:
                mismatches.append(
                    f"grid-level recipe differs in cell {cell} run {j}: "
                    f"{diff}")
    ci["controlled"] = {"controlled": not mismatches,
                        "mismatches": mismatches,
                        "varying_factors": factors}
    return ci


def _std_effect(ci):
    """Standardized paired effect (mean / paired-difference SD).

    ``rel_effect`` (effect / control-side mean) answers "how large is this
    against the baseline level"; Cohen's dz answers "how large is this
    against the seed-to-seed noise", which is the scale the paired design
    actually resolves.  Both are reported because a gain can be
    economically tiny but noise-free (large dz, small rel) or vice versa.
    """
    sd = ci.get("sd")
    mean = ci.get("mean")
    if not (_finite(sd) and _finite(mean)) or float(sd) <= 0.0:
        return float("nan")
    return float(mean) / float(sd)


def small_n_flags(ci) -> str:
    """Short flags for the small-sample robustness companions of a contrast.

    Empty when the pre-registered decision and its jackknife / BCa
    companions agree; otherwise names *which* companion dissents, so the
    console line shows at a glance that a supported contrast needs a
    second look.  See ``cage.metrics`` for the definitions.
    """
    flags = []
    n = ci.get("n") or 0
    if ci.get("test") == "equivalence":
        if n >= 2 and ci.get("loo_within_bound") is False:
            flags.append("LOO outside bound")
    elif ci.get("test") == "noninferiority":
        if n >= 2 and ci.get("loo_above_bound") is False:
            flags.append("LOO below bound")
    elif ci.get("loo_sign_flip"):
        flags.append("LOO sign flip")
    if ci.get("bca_agrees") is False:
        flags.append("BCa differs")
    return " ".join(flags)


def _fmt_ci(ci) -> str:
    if ci["n"] == 0 or not math.isfinite(ci["mean"]):
        return "n/a"
    p = ci.get("p_value", float("nan"))
    p_s = f"p={p:.3f} " if math.isfinite(p) else ""
    exact_p = ci.get("sign_flip_p", float("nan"))
    if _finite(exact_p):
        p_s += f"exact_p={float(exact_p):.3f} "
    if ci.get("test") == "equivalence":
        # Null-prediction contrast: a *small* TOST p and a CI inside the
        # bound both support the prediction, so the verdict wording is the
        # mirror image of the difference tests below.
        bound = ci.get("bound", float("nan"))
        if ci.get("equivalent"):
            verdict = f"CI inside +-{bound:g} (prediction holds)"
        elif ci["lo"] > bound or ci["hi"] < -bound:
            verdict = f"CI outside +-{bound:g} (prediction refuted)"
        else:
            verdict = f"CI not inside +-{bound:g} (inconclusive)"
        return (f"{ci['mean']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}] "
                f"n={ci['n']} {p_s}{verdict}")
    if ci.get("test") == "noninferiority":
        # One-sided null prediction: only the lower end matters, and the
        # threshold is the *negative* of the registered positive bound.
        bound = ci.get("bound", float("nan"))
        if ci.get("noninferior"):
            verdict = f"CI lower bound above -{bound:g} (prediction holds)"
        elif _finite(ci.get("hi")) and ci["hi"] < -bound:
            verdict = f"CI entirely below -{bound:g} (prediction refuted)"
        else:
            verdict = f"CI not above -{bound:g} (inconclusive)"
        return (f"{ci['mean']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}] "
                f"n={ci['n']} {p_s}{verdict}")
    supported = (ci["lo"] > 0.0) or (ci["hi"] < 0.0)
    mde = ci.get("mde80", float("nan"))
    if supported:
        verdict = "CI excludes 0"
    elif math.isfinite(mde) and abs(ci["mean"]) < mde:
        verdict = (f"inconclusive: |effect| < MDE80={mde:.4f} "
                   f"(design cannot resolve it)")
    else:
        verdict = "CI includes 0"
    return (f"{ci['mean']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}] "
            f"n={ci['n']} {p_s}{verdict}")


def _row_is_refutation(row) -> bool:
    """Is this row a pre-registered prediction *refuted* by its own test?

    Mirrors the decision branch in :meth:`ContrastLog.finalize`: a
    directional row that is FDR-significant in the opposite direction, or a
    null-prediction row whose whole interval lies beyond the registered
    bound, is a refutation.  Claim-ineligible or confounded rows are not read
    as a refutation (they are uninformative, not contradictory).  A null
    row's TOST p-value is a p-value *for* equivalence and is ~1 when the
    interval is far outside the bound, so the CI rule - not that p-value -
    decides a null refutation; using the TOST p here would make a large
    violation look "non-significant".
    """
    if row.get("controlled") is False or not row.get("claim_eligible", True):
        return False
    test = row.get("test")
    if test in ("equivalence", "noninferiority"):
        bound = (row.get("equivalence_bound") if test == "equivalence"
                 else row.get("noninferiority_bound"))
        if bound is None or not _finite(bound):
            return False
        if test == "equivalence":
            return bool(_finite(row.get("lo")) and _finite(row.get("hi"))
                        and (row["lo"] > float(bound)
                             or row["hi"] < -float(bound)))
        return bool(_finite(row.get("hi")) and row["hi"] < -float(bound))
    return bool(row.get("fdr_reject") and row.get("direction_reversed"))


def _hypothesis_evidence_synthesis(rows, hypotheses) -> dict:
    """Fold a family of contrast rows into a per-hypothesis claim status.

    The pre-registered hypotheses are *conjunctions* ("A and B and C"), but
    the decision table is row-level.  Reading "H2a supported" off a single
    surviving row would turn a conjunction into a disjunction and overstate
    the evidence.  This companion groups the rows by hypothesis and reports
    how many of its registered prediction rows were confirmed, refuted or
    left inconclusive:

    * ``supported`` - every registered directional/null-prediction row
      supports the hypothesis;
    * ``partial`` - at least one row supports it, but another registered row
      is inconclusive (or has no decision yet);
    * ``refuted`` - at least one registered row is a refutation, whatever
      the other rows say;
    * ``inconclusive`` - no row supports and none refutes;
    * ``undirected`` - the hypothesis has only two-sided/descriptive rows
      (no registered sign or bound to conjoin); their row-level readout
      stands, but it is not a confirmed signed conjunction;
    * ``no_contrast`` - nothing was run for the hypothesis.

    Only rows with a registered direction (``+``/``-``) or a registered
    null-prediction test (``equivalence``/``noninferiority``) enter the
    conjunction; completeness/robustness companions without a registered
    direction stay descriptive.  The status is a *reading companion*: it
    does not change ``supports_hypothesis`` or the FDR family.  A
    ``partial`` or ``refuted`` status is what the report warns about so a
    multi-row hypothesis is not credited on one surviving row.
    """
    # This is a report companion: a pooled/legacy result row that is not a
    # mapping (or a non-iterable container) cannot name a hypothesis and must
    # not abort the synthesis half-way through finalize().
    rows = _mapping_runs(rows)
    if isinstance(hypotheses, str):
        hypotheses = (hypotheses,)
    hypotheses = _materialize_runs(hypotheses)
    out = {}
    for hyp in hypotheses:
        hrows = [r for r in rows if r.get("hypothesis") == hyp]
        required = [r for r in hrows
                    if r.get("direction") in ("+", "-")
                    or r.get("test") in ("equivalence", "noninferiority")]
        confirmed = [r for r in required if r.get("supports_hypothesis")]
        refuted = [r for r in required if _row_is_refutation(r)]
        confirmed_ids = {id(r) for r in confirmed}
        refuted_ids = {id(r) for r in refuted}
        inconclusive = [r for r in required
                        if id(r) not in confirmed_ids
                        and id(r) not in refuted_ids]
        required_ids = {id(r) for r in required}
        descriptive = [r for r in hrows if id(r) not in required_ids]
        descriptive_support = [r for r in descriptive
                               if r.get("supports_hypothesis")]
        if not hrows:
            status = "no_contrast"
        elif not required:
            # A hypothesis with only descriptive/two-sided rows has no
            # conjunction to confirm, whether or not those rows happen to
            # be significant.  Calling a non-supporting one
            # "no_directional_prediction" would invent a status outside the
            # documented set and drop it from the payload's undirected
            # count.
            status = "undirected"
        elif refuted:
            status = "refuted"
        elif len(confirmed) == len(required):
            status = "supported"
        elif confirmed:
            status = "partial"
        else:
            status = "inconclusive"
        rep_uncertain = [r for r in confirmed + descriptive_support
                         if r.get("replication_uncertain")]
        provisional = [r for r in confirmed
                       if r.get("sesoi_point_estimate_only")]

        def _label(r):
            return f"{r.get('rule')} ({r.get('metric')})"

        out[hyp] = {
            "status": status,
            "n_rows": len(hrows),
            "n_required": len(required),
            "n_confirmed": len(confirmed),
            "n_refuted": len(refuted),
            "n_inconclusive": len(inconclusive),
            "n_descriptive": len(descriptive),
            "n_descriptive_supporting": len(descriptive_support),
            "n_replication_uncertain": len(rep_uncertain),
            "n_practically_provisional": len(provisional),
            "refuted_contrasts": [_label(r) for r in refuted],
            "unconfirmed_contrasts": [_label(r) for r in inconclusive],
            "replication_uncertain_contrasts": [_label(r)
                                                for r in rep_uncertain],
            "practically_provisional_contrasts": [_label(r)
                                                  for r in provisional],
        }
    return out


# Metrics that are raw accuracy (or an accuracy difference) and therefore
# inherit the held-out answer-balance premise.  ``f1_macro``,
# ``recall_macro`` and the PR-AUC family are explicitly excluded: they are
# the companions that remain interpretable under a skewed stream.
_ACCURACY_METRICS = {
    "accuracy", "accuracy_at_base", "accuracy_at_longest", "accuracy_drop",
    # The E2 whole-sweep slope is fitted to accuracy, so the same balance
    # premise applies; omitting it would let the slope rows survive an
    # answer-skew gate that invalidates the endpoint/drop rows.
    "accuracy_slope_log_len", "norm_accuracy",
}

CURVE_INVALID_VERDICTS = (
    "plateau", "diverging", "numerical_instability", "insufficient_points")


def _curve_valid_runs(runs):
    return [r for r in _materialize_runs(runs)
            if isinstance(r, Mapping)
            and (r.get("curve_verdict") is None
                 or (isinstance(r.get("curve_verdict"), str)
                     and r["curve_verdict"] not in CURVE_INVALID_VERDICTS))]


def _curve_invalid_seeds_for(used, run_maps):
    """Seeds whose paired contrast is contaminated by a curve-invalid run.

    A contrast seed is only usable for the sensitivity readout when *every*
    side/cell entering that seed's difference has a claim-valid curve.  The
    returned list is stored with the contrast so the decision layer can
    recompute the effect without those seeds.
    """
    bad = set()
    run_maps = _materialize_runs(run_maps)
    for seed in _materialize_runs(used):
        for runs in run_maps:
            if not isinstance(runs, Mapping):
                continue
            run = runs.get(seed)
            verdict = (run.get("curve_verdict")
                       if isinstance(run, Mapping) else None)
            if run is not None and (
                    (isinstance(verdict, str)
                     and verdict in CURVE_INVALID_VERDICTS)
                    or (verdict is not None
                        and not isinstance(verdict, str))):
                bad.add(seed)
                break
    return sorted(bad, key=_seed_order)


def _curve_health(runs):
    """Verdict counts and the number of curve-invalid runs in ``runs``.

    A curve-invalid run never learned (``plateau``), diverged, or hit
    non-finite values: the accuracies and mechanism statistics measured on
    it cannot support *any* claim about the variant - an underfit run
    produces "no difference" for reasons unrelated to the hypothesis (and
    a test set cannot measure the value of a mechanism the optimiser never
    engaged).  ``overfitting`` stays claim-valid under the fixed-step
    contract but is surfaced through ``late_overfit`` /
    ``val_regression_after_best`` instead.
    """
    counts = {}
    for r in _materialize_runs(runs):
        if not isinstance(r, Mapping):
            continue
        verdict = r.get("curve_verdict")
        if verdict is not None and not isinstance(verdict, str):
            verdict = "malformed"
        counts[verdict] = counts.get(verdict, 0) + 1
    bad = (sum(counts.get(v, 0) for v in CURVE_INVALID_VERDICTS)
           + counts.get("malformed", 0))
    return counts, bad


def _curve_warn(*labeled_runs) -> str:
    """Contrast annotation: which side has curve-invalid runs, and how many.

    Contrasts are printed regardless (they are the pre-registered readout),
    but a supported CI that rests on curves which never learned is not
    evidence about the mechanism, so the pair is flagged at the point of
    interpretation.
    """
    parts = []
    for item in _materialize_runs(labeled_runs):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        label, runs = item
        runs = _materialize_runs(runs)
        counts, bad = _curve_health(runs)
        if not bad:
            continue
        detail_parts = [f"{v}*{counts[v]}"
                        for v in CURVE_INVALID_VERDICTS if counts.get(v)]
        if counts.get("malformed"):
            detail_parts.append(f"malformed*{counts['malformed']}")
        detail = ",".join(detail_parts)
        parts.append(f"{label}: {bad}/{len(runs)} curve-invalid ({detail})")
    return "WARN " + "; ".join(parts) if parts else ""


def _selection_reliability(rows, metric="val_accuracy_mean",
                           std_key="val_accuracy_std",
                           n_key="n_seeds", curve_key="curve_verdicts",
                           diverged_key="diverged_runs", grid=None,
                           grid_keys=None, full_rows=None) -> dict:
    """Audit an argmax hyperparameter selection for winner's curse.

    Every tuned baseline here selects a value by taking the argmax of a
    validation column over a grid, then freezes it for the headline
    comparisons.  An argmax with no uncertainty attached is a point
    selection that can be entirely noise: the tuned baseline must beat
    the untuned floor and be significantly beaten by the model under
    test, so if the winning value sits inside the seed-to-seed spread of
    the runner-up the "tuned" baseline is a strawman in the *conservative*
    direction (it under-reports the baseline) and the contrast loses
    power - exactly the failure mode the design contract exists to
    prevent.  This helper records, next to the chosen value:

    - ``margin`` = best - runner-up on the selection metric;
    - ``margin_se`` an upper-bound gap standard error,
      ``sqrt(std_best^2 / n_best + std_second^2 / n_second)``.  The grid
      rows come from the *same* seeds (paired), so the conservative
      unpaired combination is used deliberately: it cannot claim more
      resolution than a paired test would;
    - ``within_noise`` = ``margin < margin_se`` or ``margin <= 0`` (an
      exact tie is never resolved): the winner is not resolved beyond
      seed noise, and small follow-up differences built on this value
      inherit that ambiguity;
    - ``curve_invalid_best``: how many of the best row's runs are
      curve-invalid (``plateau``/``diverging``/...): an argmax chosen
      among runs that never learned measures the optimizer, not the
      baseline.  Only present when the rows carry ``curve_verdicts``;
    - ``diverged_runs_best``: how many of the best row's runs diverged
      and were excluded from its statistic by the tuner (e.g.
      ``tune_linear_probe``): a mean over the surviving runs understates
      the uncertainty.  Only present when the rows carry
      ``diverged_runs``;
    - ``best_at_grid_boundary``: the winning index is the first or last
      grid point, so the optimum may lie outside the searched range and
      the true baseline could be stronger.  Only present when ``grid``
      is given.  ``full_rows`` optionally supplies the complete
      pre-filter table: axis boundaries are then read from the searched
      grid, not from the curve-valid subset, so excluding an endpoint row
      cannot make an interior winner look resolved;
    - ``reliable``: ``True`` only when the selection resolves the value
      (no boundary, no within-noise margin, and no curve-invalid runs in
      the winning row); ``None`` when the audit cannot be computed at
      all (missing/constant metric).  ``None`` is an open audit item,
      not a pass.

    ``rows`` is the table (list of dicts) the tuner searched, ordered as
    the grid; the selection metric is read with ``_finite`` so a
    missing statistic can never be read as 0.0.
    """
    rows = _materialize_runs(rows)
    if grid is not None:
        grid = _materialize_runs(grid)
    if grid_keys is not None:
        grid_keys = _materialize_runs(grid_keys)
    metric_rows = [(i, r) for i, r in enumerate(rows)
                   if isinstance(r, Mapping) and _finite(r.get(metric))]
    out = {"metric": metric, "margin": float("nan"),
           "margin_se": float("nan"), "within_noise": None,
           "curve_invalid_best": None, "diverged_runs_best": None,
           "best_at_grid_boundary": None, "reliable": None}
    if len(metric_rows) < 2:
        return out
    ranked = sorted(metric_rows, key=lambda t: -float(t[1][metric]))
    (i_best, r_best), (i_second, r_second) = ranked[0], ranked[1]
    margin = float(r_best[metric]) - float(r_second[metric])

    def _se(row):
        # A single-seed row carries a convention-0 std, not a noise
        # estimate; reading it as 0 resolution would be exactly the
        # overconfidence this audit exists to catch.  Return the standard
        # error itself: squaring a large finite std overflows, and squaring
        # a tiny one underflows below the representable range even when its
        # square root (the actual SE) is representable.
        std = row.get(std_key)
        n = _positive_integer(row.get(n_key))
        if (_finite(std) and float(std) >= 0.0
                and n is not None and n >= 2):
            return float(std) / math.sqrt(float(n))
        return float("nan")

    se_best, se_second = _se(r_best), _se(r_second)
    margin_se = math.hypot(se_best, se_second) \
        if math.isfinite(se_best) and math.isfinite(se_second) \
        else float("nan")
    within_noise = None
    if margin <= 0.0:
        # An exact tie has no winner even when no seed-variance estimate is
        # available (e.g. one run per row): ``margin < margin_se`` would
        # call that resolved (0 < 0 is false) and stamp the arbitrary argmax
        # as reliable.
        within_noise = True
    elif math.isfinite(margin_se):
        within_noise = bool(margin < margin_se)

    curve_invalid_best = None
    curve_invalid_unknown = False
    if curve_key in r_best:
        verdicts = r_best.get(curve_key)
        if isinstance(verdicts, (list, tuple)):
            if any(not isinstance(v, str) for v in verdicts):
                curve_invalid_unknown = True
            curve_invalid_best = sum(
                1 for v in verdicts
                if isinstance(v, str) and v in CURVE_INVALID_VERDICTS)
        else:
            curve_invalid_unknown = True
    diverged_best = None
    diverged_unknown = False
    if diverged_key in r_best:
        diverged_best = _difficulty_count(r_best.get(diverged_key))
        diverged_unknown = diverged_best is None
    boundary = None
    boundary_rows = (rows if full_rows is None
                     else _materialize_runs(full_rows))
    if (full_rows is None and grid is not None
            and len(grid) == len(rows)):
        boundary = bool(i_best in (0, len(rows) - 1))
    # A multi-dimensional grid (e.g. lr x steps) has a boundary on *each*
    # axis: an optimum at the smallest or largest learning rate is just as
    # unresolved as one at the shortest step budget, even when the flattened
    # row index is interior.
    axis_boundaries = {}
    if grid_keys:
        for key in grid_keys:
            vals = sorted({float(r[key]) for r in boundary_rows
                           if isinstance(r, Mapping) and _finite(r.get(key))})
            if len(vals) >= 2 and _finite(r_best.get(key)):
                axis_boundaries[str(key)] = bool(
                    float(r_best[key]) in (vals[0], vals[-1]))
        if axis_boundaries:
            boundary = bool(boundary or any(axis_boundaries.values()))
    reliable = None
    if within_noise is not None:
        reliable = not within_noise
        if boundary is not None:
            reliable = reliable and not boundary
        if curve_invalid_best is not None:
            reliable = reliable and curve_invalid_best == 0
        if diverged_best is not None:
            reliable = reliable and diverged_best == 0
        if diverged_unknown or curve_invalid_unknown:
            reliable = False
    elif boundary or curve_invalid_best or diverged_best:
        # The noise-based decision can be unavailable (e.g. a one-seed
        # winner), but a grid-boundary winner, or a winning row that
        # contains diverged or curve-invalid runs, is demonstrably not a
        # resolved measurement - it must not be left as the neutral ``None``
        # state that callers read as "open".
        reliable = False
    if diverged_unknown or curve_invalid_unknown:
        reliable = False
    out.update({"best_index": int(i_best), "second_index": int(i_second),
                "best": float(r_best[metric]),
                "runner_up": float(r_second[metric]), "margin": margin,
                "margin_se": margin_se, "within_noise": within_noise,
                "curve_invalid_best": curve_invalid_best,
                "diverged_runs_best": diverged_best,
                "diverged_runs_unknown": bool(diverged_unknown),
                "curve_verdicts_unknown": bool(curve_invalid_unknown),
                "best_at_grid_boundary": boundary,
                "axis_boundaries": axis_boundaries or None,
                "reliable": reliable})
    return out


def _curve_valid_selection_rows(table):
    """Split a tuning table into rows eligible for selection.

    A tuner must not freeze a recipe because its winning runs were
    curve-invalid (plateau/diverging/instability): those accuracies measure
    an optimizer failure, not the learning rate's merit.  Rows with no
    curve-invalid run are preferred; when every finite row is invalid the
    tuner falls back to all finite rows (and says so) rather than refusing
    to select at all.  Returns a dict with the eligible ``rows``, the
    ``n_invalid`` count among finite rows, the ``n_excluded`` count actually
    removed, and the selection ``pool`` label.
    """
    table = _materialize_runs(table)
    finite = [row for row in table
              if isinstance(row, Mapping)
              and _finite(row.get("val_accuracy_mean"))]
    valid = [row for row in finite if not row.get("curve_invalid_runs")]
    if valid:
        return {"rows": valid, "n_finite": len(finite),
                "n_invalid": len(finite) - len(valid),
                "n_excluded": len(finite) - len(valid),
                "pool": "curve-valid"}
    return {"rows": finite, "n_finite": len(finite),
            "n_invalid": len(finite), "n_excluded": 0,
            "pool": "all-finite (no curve-valid row)"}


def _selection_warn(label, selection) -> str:
    """Human-readable annotation for an unreliable selection, or ""."""
    if not selection:
        return ""
    if (selection.get("reliable") is not False
            and not selection.get("best_at_grid_boundary")
            and not selection.get("within_noise")
            and not selection.get("diverged_runs_best")
            and not selection.get("curve_invalid_best")):
        return ""
    reasons = []
    if selection.get("best_at_grid_boundary"):
        axes = selection.get("axis_boundaries") or {}
        at = [k for k, v in axes.items() if v]
        suffix = f" ({', '.join(at)})" if at else ""
        reasons.append(f"optimum at grid boundary{suffix} - range may "
                       f"exclude it")
    if selection.get("within_noise"):
        reasons.append(f"margin {selection['margin']:.4f} < "
                       f"seed-gap SE {selection['margin_se']:.4f}")
    if selection.get("diverged_runs_best"):
        reasons.append(f"{selection['diverged_runs_best']} diverged runs in "
                       f"winning row (statistic over survivors only)")
    if selection.get("curve_invalid_best"):
        reasons.append(f"{selection['curve_invalid_best']} curve-invalid "
                       f"runs in winning row (underfit/broken, not the "
                       f"baseline)")
    return f"WARN {label}: selection not resolved ({'; '.join(reasons)})"
