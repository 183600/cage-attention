# cage/experiments/_single_factor.py - Design-time single-factor contract.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _cell_factor_vector(gate, **overrides) -> dict:
    """The resolved factor vector of one experimental cell.

    A controlled contrast may vary exactly one of these keys; every other
    key of the resolved ``Cfg`` belongs to the shared recipe.  The vector is
    built from the *same* ``Cfg`` the model is built from, so the design
    certificate describes the models that will actually train instead of a
    hand-written copy of their switches.
    """
    cfg = Cfg(gate=gate, **overrides)
    return {key: getattr(cfg, key) for key in _AUDIT_MODEL_FACTOR_KEYS}


def _pair_factor_diff(cells, arm_a, arm_b) -> dict:
    """The resolved model factors that differ between two cells."""
    vector_a = cells.get(arm_a)
    vector_b = cells.get(arm_b)
    if not isinstance(vector_a, Mapping) \
            or not isinstance(vector_b, Mapping):
        missing = arm_a if not isinstance(vector_a, Mapping) else arm_b
        raise ValueError(
            f"arm {missing!r} is not a cell of the design")
    return {
        name: {"a": vector_a.get(name), "b": vector_b.get(name)}
        for name in sorted(set(vector_a) | set(vector_b))
        if vector_a.get(name) != vector_b.get(name)}


def _single_factor_design_audit(tag, cells, pairs,
                                exploratory_pairs=()) -> dict:
    """Certify that every declared controlled contrast varies one factor.

    ``cells`` maps a cell label to its resolved factor vector, ``pairs`` is
    the ordered list of ``(arm_a, arm_b)`` contrasts the experiment declares
    as controlled one-factor comparisons, and ``exploratory_pairs`` lists
    the contrasts the experiment prints *outside* the decision log.  The
    audit is the *design-time* counterpart of the per-row controlled-variable
    audit: the row audit can only see a multi-factor manipulation after the
    grid has trained (and then refuses the row), while this audit refuses
    to start the grid at all.

    The exploratory contrasts are recorded with the factors they vary, but
    they are not certified: they are the deliberately uncontrolled readings
    (a placebo arm against an unrelated control), they are printed outside
    the FDR family, and the certificate makes that visible instead of
    leaving it implicit.

    The certificate also lists the cells that take part in no declared
    one-factor contrast.  Those are the factorial corners and the
    benchmark-only arms: they are read through registered aggregates or
    through the explicitly non-controlled companions, never through a
    one-factor decision row.
    """
    if not isinstance(cells, Mapping):
        raise ValueError(
            "the single-factor design audit needs a mapping of cell label "
            f"to factor vector, got {type(cells).__name__}")
    try:
        pairs = list(pairs)
    except TypeError as exc:
        raise ValueError(
            "the single-factor design audit needs an iterable of "
            f"(arm_a, arm_b) pairs, got {type(pairs).__name__}") from exc
    try:
        exploratory_pairs = list(exploratory_pairs)
    except TypeError as exc:
        raise ValueError(
            "the exploratory contrasts must be given as an iterable of "
            f"(arm_a, arm_b) pairs, got {type(exploratory_pairs).__name__}"
        ) from exc
    rows = []
    violations = []
    declared = set()
    for item in pairs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                "a declared contrast must be an (arm_a, arm_b) pair, got "
                f"{item!r}")
        arm_a, arm_b = str(item[0]), str(item[1])
        key = (arm_a, arm_b)
        if key in declared:
            violations.append({
                "arm_a": arm_a, "arm_b": arm_b,
                "reason": "the pair is declared twice"})
            continue
        declared.add(key)
        try:
            diffs = _pair_factor_diff(cells, arm_a, arm_b)
        except ValueError as exc:
            violations.append({
                "arm_a": arm_a, "arm_b": arm_b, "reason": str(exc)})
            continue
        rows.append({
            "arm_a": arm_a, "arm_b": arm_b,
            "varying_factors": diffs,
            "n_varying_factors": len(diffs),
            "single_factor": len(diffs) == 1,
            "controlled": True,
        })
        if len(diffs) != 1:
            violations.append({
                "arm_a": arm_a, "arm_b": arm_b,
                "reason": (f"the contrast varies {len(diffs)} model factor(s) "
                           f"{sorted(diffs)} instead of exactly one")})
    exploratory_rows = []
    for item in exploratory_pairs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                "an exploratory contrast must be an (arm_a, arm_b) pair, "
                f"got {item!r}")
        arm_a, arm_b = str(item[0]), str(item[1])
        try:
            diffs = _pair_factor_diff(cells, arm_a, arm_b)
        except ValueError as exc:
            raise ValueError(
                f"[{tag}] exploratory contrast {arm_a} vs {arm_b}: "
                f"{exc}") from exc
        exploratory_rows.append({
            "arm_a": arm_a, "arm_b": arm_b,
            "varying_factors": diffs,
            "n_varying_factors": len(diffs),
            "single_factor": len(diffs) == 1,
            "controlled": False,
        })
    used = {row["arm_a"] for row in rows} | {row["arm_b"] for row in rows}
    return {
        "experiment": str(tag),
        "n_cells": len(cells),
        "n_pairs": len(rows),
        "pairs": rows,
        "exploratory_pairs": exploratory_rows,
        "cells_without_one_factor_contrast": sorted(
            str(cell) for cell in cells if str(cell) not in used),
        "single_factor": not violations,
        "violations": violations,
        "note": ("design-time certificate of the single-factor contract: "
                 "every declared controlled contrast varies exactly one "
                 "resolved model factor, so the grid may only produce "
                 "one-factor decision rows.  The factorial corners and the "
                 "benchmark-only arms take part in no one-factor contrast "
                 "and are read through registered aggregates or explicitly "
                 "non-controlled companions; the contrasts listed as "
                 "exploratory are printed outside the decision log and are "
                 "not certified."),
    }


# The controlled one-factor contrasts each experiment declares.  The order
# of a pair is the order of the difference (``arm_a - arm_b``), so the
# declaration also fixes the sign of the registered effect.
_E1_SINGLE_FACTOR_PAIRS = (
    ("calib", "none"),
    ("calib", "query"),
    ("query", "none"),
    ("null", "none"),
    ("calib", "calib_len_only"),
    ("calib", "calib_ent_only"),
    ("calib", _CALIB_PLACEBO_CELL),
    # The position-conflation control is declared in the order of its
    # registered difference (position-neutral arm minus the registered arm),
    # because the pre-registered H1h prediction is a one-sided
    # non-inferiority test in exactly that direction: the position-neutral
    # arm must not be materially *worse* than the registered arm.
    (_CALIB_POS_NORM_CELL, "calib"),
)

# The placebo arm against the query gate changes the gate *and* the
# shuffle switch, so it is not a one-factor contrast; it is printed outside
# the decision log as an exploratory capacity check.
_E1_EXPLORATORY_PAIRS = (
    (_CALIB_PLACEBO_CELL, "query"),
)

_E2_SINGLE_FACTOR_PAIRS = (
    ("calib", "none"),
    ("calib", "query"),
    ("calib", _CALIB_RAW_ENT_CELL),
    ("calib", "calib_ent_only"),
    ("calib", "calib_len_only"),
)

_E3_LAYER_SCALES = (None, 1e-2)


def _layer_scale_label(layer_scale) -> str:
    """The display label of a LayerScale level (``off`` when disabled)."""
    return "off" if layer_scale is None else f"{layer_scale:g}"


def _e3_cell_label(gate, norm, layer_scale) -> str:
    """The stable label of one E3 grid cell."""
    return f"{gate}|{norm}|{_layer_scale_label(layer_scale)}"


def _e3_single_factor_pairs() -> tuple:
    """The controlled one-factor contrasts of the E3 grid.

    Three families, each holding one factor fixed: the dynamic
    normalisation against pre-LN inside one gate and LayerScale cell, the
    LayerScale level against the same cell without it, and the calibrated
    gate against the query gate inside one normalisation and LayerScale
    cell.  The grid-level main effects and the co-design interaction are
    not pairwise contrasts and declare themselves as aggregates.
    """
    pairs = []
    for gate in ("query", "calib"):
        for layer_scale in _E3_LAYER_SCALES:
            pairs.append((_e3_cell_label(gate, "seednorm", layer_scale),
                          _e3_cell_label(gate, "pre", layer_scale)))
        for norm in NORMS:
            pairs.append((_e3_cell_label(gate, norm, 1e-2),
                          _e3_cell_label(gate, norm, None)))
    for norm in NORMS:
        for layer_scale in _E3_LAYER_SCALES:
            pairs.append((_e3_cell_label("calib", norm, layer_scale),
                          _e3_cell_label("query", norm, layer_scale)))
    return tuple(pairs)


def _e1_single_factor_design() -> dict:
    """The E1 grid and its declared controlled contrasts."""
    plan = _e1_cell_plan()
    cells = {}
    for cell in plan["cells"]:
        cells[cell] = _cell_factor_vector(
            plan["gate_of"][cell], **plan["overrides"].get(cell, {}))
    return {"cells": cells, "pairs": _E1_SINGLE_FACTOR_PAIRS,
            "exploratory_pairs": _E1_EXPLORATORY_PAIRS}


def _e2_single_factor_design() -> dict:
    """The E2 grid and its declared controlled contrasts."""
    cells = {}
    for label, gate, _record_extra, overrides in _E2_VARIANTS:
        cells[label] = _cell_factor_vector(gate, **overrides)
    return {"cells": cells, "pairs": _E2_SINGLE_FACTOR_PAIRS,
            "exploratory_pairs": ()}


def _e3_single_factor_design() -> dict:
    """The E3 grid and its declared controlled contrasts."""
    cells = {}
    for gate in ("query", "calib"):
        for norm in NORMS:
            for layer_scale in _E3_LAYER_SCALES:
                cells[_e3_cell_label(gate, norm, layer_scale)] = (
                    _cell_factor_vector(gate, norm=norm,
                                        layer_scale=layer_scale))
    return {"cells": cells, "pairs": _e3_single_factor_pairs(),
            "exploratory_pairs": ()}


# The certificates of the designs that are currently running, keyed by
# experiment tag.  The rows of a running grid may only contrast the pairs
# its own certificate declares.
_SINGLE_FACTOR_DESIGNS: dict[str, dict] = {}


def _enforce_single_factor_design(tag, design, out_dir=None) -> dict:
    """Refuse to start a grid whose declared contrasts are not one-factor.

    The certificate is computed from the resolved cell configurations
    *before* the first training run and written next to the design plan, so
    the preparation record shows which factor every planned contrast
    manipulates.  A violation stops the run: a multi-factor manipulation
    cannot be read as a one-factor claim, so training it would only spend
    compute on rows that may not support their hypothesis.
    """
    if not isinstance(design, Mapping):
        raise ValueError(
            "a single-factor design is a mapping with the cells, the "
            "controlled pairs and (optionally) the exploratory pairs, got "
            f"{type(design).__name__}")
    cells = design.get("cells") or {}
    pairs = design.get("pairs") or ()
    if not cells:
        raise ValueError(
            f"[{tag}] the single-factor design declares no cell: a grid "
            "without cells cannot be certified")
    if not pairs:
        raise ValueError(
            f"[{tag}] the single-factor design declares no controlled "
            "contrast: a grid whose contrasts are not declared cannot be "
            "certified")
    audit = _single_factor_design_audit(
        tag, cells, pairs, design.get("exploratory_pairs") or ())
    if not audit["single_factor"]:
        details = "; ".join(
            f"{item['arm_a']} vs {item['arm_b']}: {item['reason']}"
            for item in audit["violations"])
        raise ValueError(
            f"[{tag}] single-factor design contract violated: {details}.  "
            "A controlled contrast must vary exactly one model factor, so "
            "the grid is not started.")
    _SINGLE_FACTOR_DESIGNS[str(tag)] = audit
    if out_dir is not None:
        _write_json(pathlib.Path(out_dir) /
                    f"{tag}_single_factor_design.json", audit)
    print(f"[{tag}] single-factor design: {audit['n_pairs']} declared "
          f"controlled contrast(s) over {audit['n_cells']} cell(s); every "
          f"one varies exactly one resolved model factor")
    return audit


def _design_pair(tag, arm_a, arm_b, runs_a, runs_b) -> tuple:
    """The two arms of a *declared* one-factor contrast.

    The certificate names every controlled contrast the experiment may
    execute.  Routing the decision rows through this helper keeps the
    executed contrasts and the certified design the same set: a row that
    compares an undeclared pair is a design/execution mismatch, and it
    stops the run instead of silently producing a contrast the certificate
    does not describe.
    """
    audit = _SINGLE_FACTOR_DESIGNS.get(str(tag))
    if audit is None:
        raise ValueError(
            f"[{tag}] no single-factor design certificate is registered: "
            "the experiment has to certify its design before it contrasts "
            "anything")
    declared = {(row["arm_a"], row["arm_b"])
                for row in audit.get("pairs", [])}
    key = (str(arm_a), str(arm_b))
    if key not in declared:
        raise ValueError(
            f"[{tag}] contrast {key[0]} vs {key[1]} is not a declared "
            f"one-factor contrast; declared pairs: {sorted(declared)}")
    return runs_a, runs_b
