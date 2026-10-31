# cage/experiments/_contrasts.py - Contrast construction: controlled audits and paired statistics.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _training_record(settings: Settings, lr: float,
                     steps: int | None = None) -> dict:
    """The shared optimisation recipe block stamped into every run record."""
    return {"training_recipe": training_recipe(
        lr, settings.steps if steps is None else int(steps))}


def _controlled_audit(runs_a, runs_b):
    """Machine-check the controlled-variable contract of a contrast.

    Comparing method A with method B only attributes a difference to the
    module under test if *everything else* is held fixed.  This audit
    verifies, from the stored run records:

    1. **Paired seeds**: both sides were run on the same seed set, so the
       paired bootstrap differences each seed with itself (fewer / different
       seeds silently shrink the sample and break the pairing).  A seed
       repeated within one side is refused too: it is one run counted
       twice, which inflates the paired n and narrows the CI (new grids
       cannot get here - ``Settings`` / ``_run_context`` validate the seed
       set - but pooled or legacy records can);
    2. **Within-side recipe**: every run on a side used the same learning
       rate, step count, batch geometry, data size and evaluation-stream
       size and seed, the same *actual* ``steps_run`` / early-stop state,
       and - when the record carries the stamped ``training_recipe`` - the
       same optimizer, weight decay, betas, schedule, eta_min, gradient
       clipping and augmentation pipeline (no per-variant optimizer or
       schedule drift, and no run that silently trained for fewer steps);
    3. **Across-side recipe**: the two recipes are identical (the same
       fields as above), so a "gain" that is really a changed optimizer or
       learning-rate schedule is refused instead of reported as a win;
    4. **Single varying factor**: exactly one declared model factor (gate /
       norm / layer_scale / calibration-term or normalisation flag) differs,
       so the contrast is both non-degenerate and attributable.  A row that
       flips two factors at once (e.g. full calib vs the arm with *both*
       additive terms removed) is refused here: its effect is the sum of two
       manipulations, and it belongs in a registered factorial main effect
       or interaction rather than in a single-factor decision row.  Records
       that predate the resolved ``model_cfg`` stamp fall back to the
       declared factor labels; a legacy ``calib_variant`` cell label is only
       counted after it expands to a known flag vector, so an unknown/empty
       label is refused instead of being treated as one opaque factor;
    5. **Trained-model ground truth**: when both sides carry the resolved
       ``model_cfg``, every non-factor config key must match across the
       sides (an undeclared architectural difference is a confound even
       with matching recipe), and the declared labels must agree with the
       flags the model was actually built with - in particular a
       ``calib_variant`` label must name the (calib_use_len, calib_use_ent)
       cell that trained, so a mis-wired ablation arm cannot pass as the
       intended condition.

    The actual parameter count each side trained is recorded as well.  A
    differing count is legitimate across gates (``calib`` carries the
    calibration terms, ``none`` does not), so it is not a mismatch by
    itself - but where the design's *defining claim* is matched capacity
    it is enforced: the H1f active control differs from full ``calib``
    only in ``calib_shuffle_ent`` and is registered as the "identical
    calib architecture, parameter count, computation and optimiser"
    placebo, so a capacity difference there would silently confound the
    specificity test and is refused.  A side that trained more than one
    parameter count is refused as a build inconsistency whatever the
    design.

    The optimizer factory, cosine schedule, clipping and gradient settings
    are fixed in the single training entry point (``train.train_model``) and
    are therefore not per-run fields; ``lr`` plus ``steps`` pin the schedule
    that factory is combined with.  Returns an audit dict; a contrast whose
    audit fails is not read as supporting its hypothesis.

    Records with a malformed ``config`` / ``model_cfg`` object (for example
    a string where a mapping was stamped) are refused as a control
    mismatch instead of raising an ``AttributeError`` halfway through the
    audit: a provenance field that cannot be read is not evidence that the
    two sides shared one recipe.
    """
    def _audit_row(run):
        return run if isinstance(run, Mapping) else {}

    runs_a = _materialize_runs(runs_a)
    runs_b = _materialize_runs(runs_b)
    malformed_a = sum(1 for r in runs_a if not isinstance(r, Mapping))
    malformed_b = sum(1 for r in runs_b if not isinstance(r, Mapping))
    runs_a = [_audit_row(r) for r in runs_a]
    runs_b = [_audit_row(r) for r in runs_b]

    def _seed_key(seed):
        if seed is None:
            return (1, 0.0, "")
        exact = _declared_seed(seed)
        if exact is not None:
            return (0, exact, "")
        if isinstance(seed, float) and math.isfinite(seed):
            return (0, seed, "")
        return (0, float("inf"), str(seed))

    def _stream_key(seed):
        """Canonical data-stream key used to detect aliased seeds.

        ``0`` and ``2**32`` draw the same CPU stream (see
        ``cage.data.stream_seed_key``), so a pooled record carrying both is
        one independent run counted twice even though the raw seed values
        differ.  Non-numeric legacy seeds fall back to their string form.
        """
        if seed is None:
            return None
        if isinstance(seed, bool):
            return ("raw", repr(seed))
        try:
            return ("stream", stream_seed_key(seed))
        except (TypeError, ValueError, OverflowError):
            return ("raw", str(seed))

    audit = {
        "controlled": True,
        "seeds_paired": False,
        "n_a": len(runs_a),
        "n_b": len(runs_b),
        "seeds_a": sorted((r.get("seed") for r in runs_a), key=_seed_key),
        "seeds_b": sorted((r.get("seed") for r in runs_b), key=_seed_key),
        "varying_factors": {},
        "model_factor_diffs": {},
        "n_varying_factors": None,
        "single_factor": None,
        "mismatches": [],
    }
    for side, count in (("a", malformed_a), ("b", malformed_b)):
        if count:
            audit["mismatches"].append(
                f"side {side} has {count} non-mapping run record(s): the "
                f"controlled-variable contract cannot be verified")
    audit["seeds_paired"] = audit["seeds_a"] == audit["seeds_b"]
    if not audit["seeds_paired"]:
        audit["mismatches"].append(
            f"seed sets differ: a={audit['seeds_a']} b={audit['seeds_b']} "
            f"(contrast must be paired over identical seeds)")
    for side, seeds in (("a", audit["seeds_a"]), ("b", audit["seeds_b"])):
        missing = sum(1 for s in seeds if s is None)
        if missing:
            audit["mismatches"].append(
                f"side {side} has {missing} run(s) without a training seed: "
                f"the contrast cannot be paired")
        malformed_seeds = [
            s for s in seeds
            if s is not None and (
                isinstance(s, bool)
                or not isinstance(s, (int, str))
                or (isinstance(s, str) and not s))]
        if malformed_seeds:
            audit["mismatches"].append(
                f"side {side} has malformed training seed(s) "
                f"{[repr(s) for s in malformed_seeds]}: the contrast cannot "
                f"pair runs by an independent seed")
        stream_counts = {}
        for s in seeds:
            if s is None:
                continue
            key = _stream_key(s)
            stream_counts[key] = stream_counts.get(key, 0) + 1
        repeat_keys = {key for key, count in stream_counts.items()
                       if count > 1}
        # Deduplicate by the canonical stream key, not by the raw seed:
        # a malformed/unhashable seed (e.g. a list) would make the raw set
        # comprehension raise instead of being reported as a repeat.
        repeats, seen_repeats = [], set()
        for s in seeds:
            if s is None:
                continue
            key = _stream_key(s)
            if key in repeat_keys and key not in seen_repeats:
                seen_repeats.add(key)
                repeats.append(s)
        repeats = sorted(repeats, key=_seed_key)
        if repeats:
            audit["mismatches"].append(
                f"side {side} repeats training seed/stream(s) {repeats}: a "
                f"repeated seed (including a 32-bit stream alias such as "
                f"0 vs 2**32) is one run counted twice, so the paired n "
                f"would be inflated and the difference is not independent")
    # A controlled side must be one fixed condition, not a pooled mix of
    # cells.  The resolved ``model_cfg`` gets this check below; legacy
    # records that carry only declared labels need it too, otherwise a side
    # holding two different cell labels can still pass the cross-side
    # single-factor count (e.g. a={full, len_only} vs b={full} varies only
    # the concentration flag as a *set*, although side a is not one arm).
    def _label_value(run, key):
        # Missing/null/empty labels all mean "not declared".  They must
        # still be one of the side's values: a side that contains both a
        # labelled run and an unlabelled run has no single declared
        # condition, even when the other side has only one label (so the
        # key never enters the cross-side ``varying_factors`` map).
        if key not in run:
            return "<unset>"
        value = run.get(key)
        if value is None:
            return "<unset>"
        if isinstance(value, str):
            return "<unset>" if not value else value
        return str(value)

    side_pooled = False
    for side, runs in (("a", runs_a), ("b", runs_b)):
        for key in _AUDIT_FACTOR_KEYS:
            values = sorted({_label_value(r, key) for r in runs})
            if len(values) > 1:
                side_pooled = True
                audit["mismatches"].append(
                    f"side {side} carries multiple {key} labels {values}: "
                    f"the side pools different conditions, so it is not one "
                    f"fixed contrast arm")
    # Capacity: the parameter count each side actually trained (recorded for
    # every contrast; the placebo-specific enforcement happens below, where
    # the resolved model configs have been compared).
    malformed_params = [
        r for r in runs_a + runs_b
        if "params" in r and _positive_integer(r.get("params")) is None]
    if malformed_params:
        audit["mismatches"].append(
            f"{len(malformed_params)} run(s) carry a malformed parameter "
            f"count; capacity cannot be verified")
    params_a = sorted({value for value in
                       (_positive_integer(r.get("params")) for r in runs_a)
                       if value is not None})
    params_b = sorted({value for value in
                       (_positive_integer(r.get("params")) for r in runs_b)
                       if value is not None})
    audit["params"] = {"a": params_a, "b": params_b}
    for side, counts in (("a", params_a), ("b", params_b)):
        if len(counts) > 1:
            audit["mismatches"].append(
                f"side {side} trained {len(counts)} distinct parameter "
                f"counts {counts}: the side is not one fixed model")
    if len(params_a) == 1 and len(params_b) == 1:
        audit["capacity"] = {
            "a": params_a[0], "b": params_b[0],
            "delta_b_minus_a": params_b[0] - params_a[0],
            "matched": params_a[0] == params_b[0],
        }
    for side, runs in (("a", runs_a), ("b", runs_b)):
        malformed_configs = [
            r for r in runs
            if r.get("config") is not None
            and not isinstance(r.get("config"), Mapping)]
        if malformed_configs:
            audit["mismatches"].append(
                f"side {side} has {len(malformed_configs)} run(s) with a "
                f"malformed config object (expected a mapping): the training "
                f"recipe cannot be verified")
        malformed_recipes = [
            r for r in runs
            if r.get("training_recipe") is not None
            and not isinstance(r.get("training_recipe"), Mapping)]
        if malformed_recipes:
            audit["mismatches"].append(
                f"side {side} has {len(malformed_recipes)} run(s) with a "
                f"malformed training_recipe object (expected a mapping): "
                f"the optimisation recipe cannot be verified")
        views = [_recipe_view(r) for r in runs]
        if not views:
            audit["mismatches"].append(f"side {side} has no runs")
            continue
        first = views[0]
        for j, other in enumerate(views[1:], start=1):
            for key in sorted(set(first) | set(other)):
                if other.get(key) != first.get(key):
                    audit["mismatches"].append(
                        f"side {side} recipe differs between runs 0 and {j}: "
                        f"{key}={first.get(key)!r} vs {other.get(key)!r}")
    if runs_a and runs_b:
        va, vb = _recipe_view(runs_a[0]), _recipe_view(runs_b[0])
        for key in sorted(set(va) | set(vb)):
            if vb.get(key) != va.get(key):
                audit["mismatches"].append(
                    f"recipe differs across sides: {key}={va.get(key)!r} vs "
                    f"{vb.get(key)!r}")
        for key in _AUDIT_FACTOR_KEYS:
            in_a = any(key in r for r in runs_a)
            in_b = any(key in r for r in runs_b)
            if not (in_a and in_b):
                continue
            set_a = {str(r.get(key)) for r in runs_a}
            set_b = {str(r.get(key)) for r in runs_b}
            if set_a != set_b:
                audit["varying_factors"][key] = {
                    "a": sorted(set_a), "b": sorted(set_b)}
        # Ground truth of the trained model: check the resolved configs
        # directly rather than the hand-attached labels.
        cfgs_a = [r.get("model_cfg") for r in runs_a]
        cfgs_b = [r.get("model_cfg") for r in runs_b]
        malformed_cfgs = [
            c for c in cfgs_a + cfgs_b
            if c is not None and not isinstance(c, Mapping)]
        if malformed_cfgs:
            audit["mismatches"].append(
                f"malformed model_cfg object(s) (expected mappings): "
                f"{len(malformed_cfgs)} run(s) cannot be checked against the "
                f"trained configuration")
        # The top-level design labels are what a reader sees in the table;
        # they must name the model that actually trained.  Check this for
        # *every* run that carries a resolved config, not only when both
        # sides do: a mixed-provenance pair (new record versus a legacy one)
        # would otherwise let a mislabelled new run hide a multi-factor
        # manipulation behind its declared label.  Checking only
        # ``calib_variant`` would let a run labelled gate="none" carry a
        # query/calib config, making an identical-config or mislabelled
        # contrast look controlled.
        for side, side_runs in (("a", runs_a), ("b", runs_b)):
            for r in side_runs:
                cfg = r.get("model_cfg")
                if not isinstance(cfg, Mapping):
                    continue
                for key in ("gate", "norm", "layer_scale"):
                    if key not in r:
                        continue
                    label = r.get(key)
                    trained = cfg.get(key)
                    if label != trained:
                        audit["mismatches"].append(
                            f"side {side}: {key} label {label!r} does "
                            f"not match the trained model_cfg "
                            f"({trained!r})")
                label = r.get("calib_variant")
                if not label:
                    continue
                if cfg.get("gate") != "calib":
                    audit["mismatches"].append(
                        f"side {side}: calib_variant={label!r} on a "
                        f"gate={cfg.get('gate')!r} run")
                    continue
                cell = _calib_cell(r)
                if label != cell:
                    audit["mismatches"].append(
                        f"side {side}: calib_variant label {label!r} "
                        f"does not match the trained flags "
                        f"(cell {cell!r})")
        if all(isinstance(c, Mapping) for c in cfgs_a + cfgs_b):
            for side, cfgs in (("a", cfgs_a), ("b", cfgs_b)):
                for j, other in enumerate(cfgs[1:], start=1):
                    if other != cfgs[0]:
                        audit["mismatches"].append(
                            f"side {side} model_cfg differs between runs 0 "
                            f"and {j}: {cfgs[0]} vs {other}")
            for key in sorted(set(cfgs_a[0]) | set(cfgs_b[0])):
                va, vb = cfgs_a[0].get(key), cfgs_b[0].get(key)
                if va == vb:
                    continue
                if key in _AUDIT_MODEL_FACTOR_KEYS:
                    audit["model_factor_diffs"][key] = {"a": va, "b": vb}
                else:
                    audit["mismatches"].append(
                        f"undeclared model-config difference across sides: "
                        f"{key}={va!r} vs {vb!r}")
            audit["model_cfg"] = {"a": cfgs_a[0], "b": cfgs_b[0]}
            labels_a = {r.get("calib_variant") for r in runs_a
                        if isinstance(r.get("calib_variant"), str)
                        and r.get("calib_variant")}
            labels_b = {r.get("calib_variant") for r in runs_b
                        if isinstance(r.get("calib_variant"), str)
                        and r.get("calib_variant")}
            cells_a = {c for c in (_calib_cell(r) for r in runs_a) if c}
            cells_b = {c for c in (_calib_cell(r) for r in runs_b) if c}
            if (labels_a or labels_b) and labels_a != labels_b \
                    and cells_a == cells_b:
                audit["mismatches"].append(
                    f"declared ablation labels differ "
                    f"({sorted(labels_a)} vs {sorted(labels_b)}) but the "
                    f"trained calibration-term cells are identical "
                    f"({sorted(cells_a)}): the contrast does not actually "
                    f"vary the mechanism")
            if audit["model_factor_diffs"]:
                audit["varying_factors"].update(audit["model_factor_diffs"])
            # H1f's active control is registered as the identical calib
            # architecture / parameter count / computation, differing only
            # in the shuffled entropy signal.  When the resolved configs
            # say that is the *only* difference, matched capacity is a
            # defining property of the contrast, so a differing count is a
            # confound (capacity, not entropy information) and the audit
            # refuses the contrast instead of letting the specificity test
            # read as if the placebo were capacity-matched.
            capacity = audit.get("capacity")
            if capacity and not capacity["matched"] \
                    and set(audit["model_factor_diffs"]) == \
                    {"calib_shuffle_ent"}:
                audit["mismatches"].append(
                    f"placebo capacity mismatch: the only resolved "
                    f"model-config difference is calib_shuffle_ent, whose "
                    f"defining claim is the identical calib architecture "
                    f"and parameter count, but the sides trained "
                    f"{capacity['a']} vs {capacity['b']} parameters - the "
                    f"capacity would be confounded with the shuffled "
                    f"entropy signal")
        # Single-factor contract (see point 4 above).  The resolved config
        # is ground truth when both sides carry it; otherwise the declared
        # labels are the only evidence available.  ``calib_variant`` is a
        # *cell* label, not an independent factor: counting it alongside the
        # resolved flags would make an ordinary one-flag ablation look like
        # a two-factor contrast, so only the resolved diffs are counted when
        # the model config is present.
        unresolved_cell = False
        if audit.get("model_cfg") is not None:
            factor_keys = set(audit["model_factor_diffs"])
        else:
            malformed_labels = [
                r for r in runs_a + runs_b
                if r.get("calib_variant") is not None
                and not isinstance(r.get("calib_variant"), str)]
            if malformed_labels:
                audit["mismatches"].append(
                    "legacy record(s) carry a non-string calib_variant "
                    "label; the ablation cannot be identified")
            factor_keys = set(audit["varying_factors"])
            # Expand a label-only ``calib_variant`` difference into the
            # factor vector it names, so a legacy record cannot pass a
            # two-term manipulation (e.g. full vs both-terms-off) as one
            # factor merely because only the cell label was stamped.
            if "calib_variant" in factor_keys:
                fa = [_CALIB_CELL_FACTORS.get(r.get("calib_variant"))
                      if isinstance(r.get("calib_variant"), str) else None
                      for r in runs_a]
                fb = [_CALIB_CELL_FACTORS.get(r.get("calib_variant"))
                      if isinstance(r.get("calib_variant"), str) else None
                      for r in runs_b]
                if all(x is not None for x in fa + fb):
                    factor_keys.discard("calib_variant")
                    audit["varying_factors"].pop("calib_variant", None)
                    for key in _CALIB_ABLATION_FLAGS:
                        va = {x[key] for x in fa}
                        vb = {x[key] for x in fb}
                        if va != vb:
                            factor_keys.add(key)
                            audit["varying_factors"][key] = {
                                "a": sorted(va), "b": sorted(vb)}
                else:
                    # A ``calib_variant`` label names a *cell* (a whole
                    # flag vector), not one independent factor.  When at
                    # least one side's label is unknown/empty/non-string,
                    # the vector cannot be recovered, so counting the
                    # label as a single factor would certify a contrast
                    # whose manipulation is actually unverified.  Fail
                    # closed instead: a legacy record may only pass as
                    # single-factor when its cell can be expanded.
                    unresolved_cell = True
                    labels = sorted({
                        str(r.get("calib_variant"))
                        for r in runs_a + runs_b
                        if r.get("calib_variant") is not None})
                    audit["mismatches"].append(
                        "single-factor contract: legacy calib_variant "
                        f"label(s) {labels} do not all name a known "
                        "calibration cell, so the manipulated flag(s) "
                        "cannot be identified; a cell label is not one "
                        "independent factor and the contrast is not "
                        "certified as single-factor")
        audit["n_varying_factors"] = len(factor_keys)
        audit["single_factor"] = (len(factor_keys) == 1
                                  and not unresolved_cell and not side_pooled)
        if not factor_keys:
            if audit.get("model_cfg") is not None:
                # A stale declared label difference does not make an
                # identical-config pair a controlled contrast.  This check
                # is separate from the label-only fallback because
                # ``varying_factors`` still contains the declared labels
                # even when the resolved configs are identical.
                audit["mismatches"].append(
                    "the resolved model config is identical on both sides "
                    "(no model factor differs), so the declared label "
                    "difference does not identify a controlled manipulation")
            else:
                audit["mismatches"].append(
                    "no declared model factor (gate/norm/layer_scale/"
                    "calib_variant) differs across the sides: degenerate or "
                    "mislabeled contrast")
        elif len(factor_keys) > 1:
            audit["mismatches"].append(
                f"single-factor contract: the contrast varies "
                f"{len(factor_keys)} model factor(s) "
                f"{sorted(factor_keys)}; a controlled contrast must vary "
                f"exactly one factor, so a multi-factor manipulation has to "
                f"be read through a registered factorial main effect or "
                f"interaction instead")
    audit["controlled"] = not audit["mismatches"]
    return audit


_STREAM_DIGEST_HEX = frozenset("0123456789abcdefABCDEF")


def _paired_stream_test(by_a, by_b, used, metric, seed_mean):
    """Exact paired test on the shared finite evaluation stream.

    Training seeds are independent, but the evaluation items are not: every
    seed is scored on the same sealed stream.  The seed-level bootstrap
    answers "would this effect replicate over training runs on this stream",
    not "is the stream large enough to resolve it".  When both sides store
    the compact per-item correctness mask from ``eval_metrics``, this helper
    runs McNemar's exact test on the concatenated items and reports the
    item-level effect, so a positive seed-level mean whose item-level sign
    disagrees is caught instead of being printed as a win.

    That pooled exact p is only descriptive: concatenating seeds repeats the
    same sealed items, so the pooled item table violates McNemar's
    independence assumption and its p is anti-conservative.  The per-seed
    discordant counts, the seed-level sign test over the independent training
    seeds, and the consistency of the per-seed item effects are therefore
    reported alongside it; only the seed-level summaries may corroborate a
    claim.  The pooled p is retained for continuity and explicitly labelled
    descriptive.

    Only ``accuracy`` has a stored mask.  Any missing mask, differing item
    count, or malformed record makes the companion not applicable (it is a
    diagnostic, never a reason to fabricate a test).
    """
    out = {"applicable": False, "reason": None, "n": 0,
           "n_seeds": 0, "b": None, "c": None, "p_exact": None,
           "item_mean_diff": None, "direction_agrees": None,
           "pooled_descriptive": True, "per_seed": [],
           "seed_sign_positive": 0, "seed_sign_negative": 0,
           "seed_sign_zero": 0, "seed_sign_test_p": None,
           "seed_sign_consistent_with_mean": None,
           "per_seed_direction_agreement": None,
           "stream_digest": None, "digest_verified": False,
           "crossed_bootstrap": {"valid": False, "reason": "not computed"},
           "fold_audit": {"valid": False, "applicable": False,
                          "reason": "not computed"}}
    if metric != "accuracy":
        out["reason"] = "only accuracy stores per-item correctness"
        return out
    used = _materialize_runs(used)
    if not isinstance(by_a, Mapping) or not isinstance(by_b, Mapping):
        out["reason"] = (
            "paired run records must be mappings keyed by training seed")
        return out
    masks_a, masks_b, ns = [], [], []
    digests_a, digests_b = set(), set()
    digest_pairs_present = 0
    for seed in used:
        ra, rb = by_a.get(seed), by_b.get(seed)
        if ra is None or rb is None:
            out["reason"] = f"missing run for seed {seed}"
            return out
        if not isinstance(ra, Mapping) or not isinstance(rb, Mapping):
            out["reason"] = (
                f"malformed run record for seed {seed}: expected a mapping, "
                f"got {type(ra).__name__} / {type(rb).__name__}")
            return out
        ea, eb = ra.get("correct_bitmask"), rb.get("correct_bitmask")
        na = _positive_integer(ra.get("n", ra.get("eval_n")))
        nb = _positive_integer(rb.get("n", rb.get("eval_n")))
        if not isinstance(ea, str) or not isinstance(eb, str):
            out["reason"] = "per-item correctness not recorded on both sides"
            return out
        if na is None or nb is None:
            out["reason"] = "evaluation item counts are not positive integers"
            return out
        if na != nb:
            out["reason"] = "evaluation item counts differ"
            return out
        n = na
        try:
            ma = decode_correct_mask(ea, n)
            mb = decode_correct_mask(eb, n)
        except (TypeError, ValueError) as exc:
            out["reason"] = f"invalid correctness mask: {exc}"
            return out
        da = ra.get("stream_digest")
        db = rb.get("stream_digest")
        for side, value in (("a", da), ("b", db)):
            if value is None:
                continue
            if not isinstance(value, str):
                out["reason"] = (
                    f"evaluation-stream digest on side {side} is not a "
                    f"string")
                return out
            # ``stream_content_digest`` only ever emits hexadecimal
            # SHA-256 prefixes.  A blank or arbitrary string is a
            # malformed stamp, not evidence that the two item streams
            # match: accepting it would let two unversioned records pass
            # as "verified".
            if not value or any(ch not in _STREAM_DIGEST_HEX for ch in value):
                out["reason"] = (
                    f"evaluation-stream digest on side {side} is not a "
                    f"hexadecimal content digest")
                return out
        if da is not None and db is not None:
            # Both sides stamped the digest: only this pair can verify that
            # the masks describe the same realized items.
            digest_pairs_present += 1
            digests_a.add(da)
            digests_b.add(db)
            if da != db:
                out["reason"] = (
                    "evaluation-stream content digests differ between the "
                    "paired sides; the item masks were not computed on the "
                    "same data")
                return out
        elif da is not None or db is not None:
            # Legacy/mixed records: one side did not stamp the digest, so
            # item-by-item pairing cannot be verified.  Keep the companion
            # available but disclose the weaker evidence; the digest still
            # identifies the one side that has it.
            if da is not None:
                digests_a.add(da)
            if db is not None:
                digests_b.add(db)
        masks_a.append(ma)
        masks_b.append(mb)
        ns.append(n)
    if not masks_a:
        out["reason"] = "no paired seeds"
        return out
    if len(set(ns)) != 1:
        # Every training seed is scored on the same sealed stream, so one
        # seed with a different item count is a protocol mismatch.  Pooling
        # the masks would still concatenate, but the per-seed and pooled
        # discordant tables would describe different samples.
        out["reason"] = (
            f"evaluation item counts differ across paired seeds: "
            f"{sorted(set(ns))}")
        return out
    # A content digest identifies the actual realization behind each mask;
    # identical item counts are not enough to pair items if a record was
    # produced by a different stream/geometry.  A disagreement *within* one
    # side is also a protocol violation: the paired seeds must all be
    # scored on one common stream.
    if len(digests_a) > 1 or len(digests_b) > 1 or (
            digests_a and digests_b and digests_a != digests_b):
        out["reason"] = (
            "evaluation-stream content digests are not identical across "
            "the paired runs")
        return out
    stream_digest = (next(iter(digests_a | digests_b))
                     if digests_a or digests_b else None)
    digest_verified = bool(
        digest_pairs_present == len(used) and stream_digest is not None)
    ca = torch.cat(masks_a)
    cb = torch.cat(masks_b)
    if ca.numel() != cb.numel() or ca.numel() == 0:
        out["reason"] = "no evaluation items"
        return out
    per_seed = []
    for seed, ma, mb, n in zip(used, masks_a, masks_b, ns):
        mc_seed = mcnemar_test(ma, mb)
        diff_seed = float(ma.float().mean() - mb.float().mean())
        per_seed.append({
            "seed": seed, "n": int(n),
            "b": int(mc_seed["b"]), "c": int(mc_seed["c"]),
            "item_mean_diff": diff_seed,
            "p_exact": mc_seed["p_exact"],
            "signed_direction": (1 if diff_seed > 0.0 else
                                 -1 if diff_seed < 0.0 else 0),
        })
    mc = mcnemar_test(ca, cb)
    item_mean = float(ca.float().mean() - cb.float().mean())
    if _finite(seed_mean) and _finite(item_mean):
        agrees = ((seed_mean > 0.0 and item_mean > 0.0)
                  or (seed_mean < 0.0 and item_mean < 0.0)
                  or (seed_mean == 0.0 and item_mean == 0.0))
    else:
        agrees = None
    directions = [row["signed_direction"] for row in per_seed]
    nonzero = [d for d in directions if d != 0]
    pos = sum(1 for d in directions if d > 0)
    neg = sum(1 for d in directions if d < 0)
    zero = sum(1 for d in directions if d == 0)
    overall_dir = (1 if item_mean > 0.0 else -1 if item_mean < 0.0 else 0)
    if overall_dir and nonzero:
        agreement = sum(1 for d in nonzero if d == overall_dir) / len(nonzero)
        consistent = all(d == overall_dir for d in nonzero)
    else:
        agreement = None
        consistent = None
    if nonzero:
        k = max(pos, neg)
        tail = sum(math.comb(len(nonzero), i)
                   for i in range(k, len(nonzero) + 1))
        sign_p = min(1.0, 2.0 * tail / (2 ** len(nonzero)))
    else:
        sign_p = None
    fold_audit = evaluation_fold_audit(
        torch.stack(masks_a), torch.stack(masks_b), n_folds=8)
    crossed = crossed_cluster_bootstrap_ci(
        torch.stack(masks_a), torch.stack(masks_b), n_boot=2000)
    out.update({
        "applicable": True, "n": int(ca.numel()),
        "n_seeds": len(masks_a), "n_per_seed": sorted(set(ns)),
        "stream_digest": stream_digest,
        "digest_verified": digest_verified,
        "b": mc["b"], "c": mc["c"],
        "p_exact": mc["p_exact"], "p_chi2_cc": mc["p_chi2_cc"],
        "item_mean_diff": item_mean, "direction_agrees": agrees,
        "pooled_descriptive": True,
        "pooled_note": (
            "the pooled McNemar p concatenates every seed's evaluation of "
            "the same sealed items, so repeated item evaluations are not "
            "independent and this p must not be read as a confirmation or "
            "a claim gate; use the per-seed counts and the independent-seed "
            "sign test"),
        "per_seed": per_seed,
        "seed_sign_positive": int(pos), "seed_sign_negative": int(neg),
        "seed_sign_zero": int(zero), "seed_sign_test_p": sign_p,
        "seed_sign_consistent_with_mean": consistent,
        "per_seed_direction_agreement": agreement,
        "fold_audit": fold_audit,
        "crossed_bootstrap": crossed,
    })
    return out


def _mapping_runs(runs):
    """Materialize run records, dropping rows that are not mappings."""
    return [run for run in _materialize_runs(runs)
            if isinstance(run, Mapping)]


def _paired_seed(run):
    """The usable training seed of a stored run record, or ``None``.

    A pooled/legacy row may be malformed (not a mapping, a bool, a
    fractional value, or an unhashable container).  ``_paired_diff`` used
    these values directly as dict keys, so a list seed raised
    ``TypeError: unhashable type`` and a non-mapping row raised
    ``AttributeError`` before the contrast could abstain.  Only plain
    integers (the contract produced by ``Settings``) and legacy string
    labels are usable pairing keys; everything else is an unpaired row.
    """
    if not isinstance(run, Mapping):
        return None
    seed = run.get("seed")
    if isinstance(seed, bool):
        return None
    if isinstance(seed, int):
        return int(seed)
    if isinstance(seed, str) and seed:
        return seed
    return None


def _paired_diff(runs_a, runs_b, metric):
    """Paired contrast ``a - b`` plus both arms' levels.

    ``baseline_mean`` (the mean of ``metric`` on side ``b``) is attached so
    the report can print the effect as a relative change: a contrast can be
    statistically significant with zero variance yet practically
    negligible, and the relative effect is what exposes that.  The full
    per-arm mean +- std (and the seed count each mean is read on) travel
    with the contrast too, so the decision table answers the pre-registered
    "did the effect exceed the seed-to-seed spread?" question directly
    instead of only through ``dz``/MDE80 - both arms are reported on the
    paired sample (the seeds where both sides carry a finite value), so the
    printed levels, the effect and ``rel_effect`` can never describe
    different seed sets.  ``arms_all`` keeps the full-run levels for
    transparency, but only the paired ``arms`` feed the decision table.

    The controlled-variable audit (``_controlled_audit``) is attached as
    ``controlled``: a difference between recipes (optimizer settings, step
    count, data geometry, evaluation streams, unpaired seeds) would
    confound the contrast, so the claim is refused downstream instead of
    being reported as a win.

    Seed accounting: only seeds where both sides carry a finite value for
    this metric enter the paired difference.  ``seeds_used`` records them;
    ``seeds_dropped`` records the rest, split into ``seeds_unpaired``
    (value present on only one side - a pairing failure) and
    ``seeds_undefined`` (metric undefined on both sides, e.g. no
    false-concentration cases), so a shrunken sample is never silently
    read as if it were the full seed set.
    """
    runs_a = _materialize_runs(runs_a)
    runs_b = _materialize_runs(runs_b)
    missing_a = [r for r in runs_a if _paired_seed(r) is None]
    missing_b = [r for r in runs_b if _paired_seed(r) is None]
    by_a = {}
    for r in runs_a:
        seed = _paired_seed(r)
        if seed is not None:
            by_a[seed] = r
    by_seed = {}
    for r in runs_b:
        seed = _paired_seed(r)
        if seed is not None:
            by_seed[seed] = r
    diffs = []
    used, dropped, unpaired, undefined = [], [], [], []
    # Iterate the *union* of the two seed sets: a seed present only on one
    # side is a pairing failure and must be recorded as dropped/unpaired,
    # not silently absent from the accounting just because it happens to
    # be missing on side a.
    planned = sorted(set(by_a) | set(by_seed), key=_seed_order)
    for seed in planned:
        ra, rb = by_a.get(seed), by_seed.get(seed)
        va = None if ra is None else ra.get(metric)
        vb = None if rb is None else rb.get(metric)
        finite_a, finite_b = _finite(va), _finite(vb)
        if ra is not None and rb is not None and finite_a and finite_b:
            used.append(seed)
            diffs.append(float(va) - float(vb))
        elif ra is None or rb is None or finite_a != finite_b:
            dropped.append(seed)
            unpaired.append(seed)
        else:
            # Metric undefined on both sides (e.g. no false-concentration
            # cases at this scale): not a pairing failure, but it still
            # shrinks the effective sample and must be visible.
            dropped.append(seed)
            undefined.append(seed)
    if missing_a or missing_b:
        n_missing = len(missing_a) + len(missing_b)
        dropped.extend([None] * n_missing)
        unpaired.extend([None] * n_missing)
    ci = paired_bootstrap_ci(diffs)
    # Levels are reported on exactly the seeds that entered the paired
    # difference.  Using all runs here would let a metric-undefined or
    # unpaired seed contribute to an arm mean but not to the effect (or
    # vice versa), making the printed levels and the contrast describe
    # different samples.  The full-run levels remain available as
    # ``arms_all`` for inspection.
    if used:
        side_a = _mean_std([by_a[s][metric] for s in used])
        base = _mean_std([by_seed[s][metric] for s in used])
    else:
        side_a = _mean_std([])
        base = _mean_std([])
    ci["arms"] = {"a": side_a, "b": base}

    def all_levels(runs):
        return _mean_std([r[metric] for r in runs
                          if isinstance(r, Mapping) and metric in r])

    ci["arms_all"] = {"a": all_levels(runs_a), "b": all_levels(runs_b)}
    ci["arms_note"] = "paired seeds used by the effect"
    ci["baseline_mean"] = base["mean"]
    # The shared finite evaluation stream is a second source of noise on
    # top of training-seed variance: every seed is scored on the same
    # items.
    # Run the item-level paired test (McNemar) whenever both sides
    # recorded their per-item correctness.  The pooled p is descriptive
    # only: concatenating seeds repeats the same sealed items, so it cannot
    # replace the seed-level decision.  Per-seed discordant counts and an
    # independent-seed sign test travel alongside it, and the pooled item
    # sign still catches a gain that is only an artifact of the stream.
    ci["stream_test"] = _paired_stream_test(
        by_a, by_seed, used, metric, ci.get("mean"))
    # Annotate the crossed (seed x item) bootstrap with the seed-level
    # decision it is meant to audit.  The pre-registered CI conditions on
    # the sealed stream; the crossed CI also resamples its items, so a
    # verdict that survives the first but not the second is conditional on
    # this particular evaluation stream rather than a property of the task.
    crossed = ci["stream_test"].get("crossed_bootstrap") or {}
    if crossed.get("valid"):
        seed_lo, seed_hi = ci.get("lo"), ci.get("hi")
        if _finite(seed_lo) and _finite(seed_hi):
            seed_excludes = bool(seed_lo > 0.0 or seed_hi < 0.0)
            crossed_excludes = bool(crossed.get("ci_excludes_zero"))
            crossed["seed_ci_lo"] = float(seed_lo)
            crossed["seed_ci_hi"] = float(seed_hi)
            crossed["seed_ci_excludes_zero"] = seed_excludes
            crossed["crossed_ci_excludes_zero"] = crossed_excludes
            crossed["decision_agrees"] = bool(
                seed_excludes == crossed_excludes)
            crossed["stream_limited"] = bool(
                seed_excludes and not crossed_excludes)
    # Raw per-seed differences: lets an equivalence companion reuse the
    # identical pairing instead of re-deriving it (and drifting from it).
    ci["diffs"] = list(diffs)
    ci["curve_invalid_seeds"] = _curve_invalid_seeds_for(
        used, (by_a, by_seed))
    # The paired sample is only the seeds where *both* sides carry a
    # finite value for this metric: recording which seeds actually entered
    # makes a silently shrunken sample visible (and the finalize layer
    # refuses claims below the pre-registered minimum seed floor).
    ci["seeds_used"] = sorted(used, key=_seed_order)
    ci["seeds_dropped"] = sorted(dropped, key=_seed_order)
    ci["seeds_unpaired"] = sorted(unpaired, key=_seed_order)
    ci["seeds_undefined"] = sorted(undefined, key=_seed_order)
    audit_a = [r for r in runs_a if _paired_seed(r) is not None]
    audit_b = [r for r in runs_b if _paired_seed(r) is not None]
    audit = _controlled_audit(audit_a, audit_b)
    if missing_a or missing_b:
        audit["mismatches"].append(
            f"{len(missing_a) + len(missing_b)} run(s) lack a usable "
            f"(hashable, non-boolean) training seed and cannot be paired")
        audit["controlled"] = False
    ci["controlled"] = audit
    return ci


def _paired_benchmark(runs_a, runs_b, metric, sesoi, reference,
                      *, comparison="benchmark", treatment=None):
    """Paired tuned-baseline benchmark (not a controlled mechanism contrast).

    The whole point of a complexity comparison is that the two sides differ
    in architecture and each was tuned independently, so the controlled-
    recipe audit does not apply and the row must never be routed through
    :class:`ContrastLog` (which would refuse it as confounded).  The paired
    seed differences, CI and SESOI are still the right effect-size readout:
    ``complexity_justified`` means the arm is both practically above and
    statistically above the tuned reference, while ``below_reference`` means
    the reference is confidently better.  Curve-invalid runs on either side
    are excluded before pairing (they cannot support a depth claim), and the
    excluded seeds are recorded under ``curve_invalid_excluded``.  The result
    carries an explicit ``comparison`` tag and a summary of the (expected)
    recipe mismatches, never the full audit, so it cannot be mistaken for a
    controlled contrast.

    ``comparison``/``treatment`` let a caller distinguish the other legitimate
    benchmark family: the E1 per-variant fairness account, where the arms
    keep their architecture but each carries its *own* val-selected lr (and
    calib_tau_init).  The audit still fails by design, so the label must say
    which treatment was varied; the pre-registered same-recipe rows remain a
    separate accounting.
    """
    runs_a = _materialize_runs(runs_a)
    runs_b = _materialize_runs(runs_b)
    excluded = {}
    for side, runs in (("a", runs_a), ("b", runs_b)):
        bad = [r.get("seed") for r in runs
               if isinstance(r, Mapping)
               and r.get("curve_verdict") in CURVE_INVALID_VERDICTS]
        if bad:
            excluded[side] = sorted(bad, key=_seed_order)
    ci = _paired_diff(_curve_valid_runs(runs_a), _curve_valid_runs(runs_b),
                      metric)
    ci["curve_invalid_excluded"] = excluded
    audit = ci.get("controlled") or {}
    ci["comparison"] = str(comparison)
    ci["treatment"] = (str(treatment) if treatment else
                       "architecture/capacity and the val-tuned recipe")
    ci["reference"] = str(reference)
    ci["sesoi"] = float(sesoi)
    # The audit is *expected* to fail (the point is a different architecture
    # and/or a separately tuned recipe), so the row is marked as
    # intentionally non-controlled instead of carrying a wall of recipe
    # mismatches into the summary; only their count is kept for traceability.
    ci["controlled"] = None
    ci["controlled_audit"] = {
        "applicable": False,
        "reason": (f"{ci['comparison']} deliberately varies "
                   f"{ci['treatment']}"),
        "observed_mismatch_count": len(audit.get("mismatches") or []),
    }
    mean, lo, hi = ci.get("mean"), ci.get("lo"), ci.get("hi")
    ci["complexity_justified"] = bool(
        _finite(mean) and _finite(lo)
        and float(mean) >= float(sesoi) and float(lo) > 0.0)
    ci["below_reference"] = bool(_finite(hi) and float(hi) < 0.0)
    # "Indistinguishable" is an equivalence-style read: it requires a
    # measured paired effect whose CI lies inside the pre-registered
    # practical bound.  A wide interval that merely includes 0 is
    # inconclusive, not evidence of no difference; and when every seed was
    # excluded as curve-invalid (or no seed could be paired) there is no
    # evidence at all.
    ci["unmeasured"] = not (ci.get("seeds_used") and _finite(mean))
    ci["indistinguishable"] = bool(
        not ci["unmeasured"] and _finite(lo) and _finite(hi)
        and math.isfinite(float(sesoi))
        and float(lo) > -float(sesoi) and float(hi) < float(sesoi))
    ci["inconclusive"] = bool(
        not ci["unmeasured"] and not ci["complexity_justified"]
        and not ci["below_reference"] and not ci["indistinguishable"])
    return ci


def _one_sample_contrast(runs, metric, bound=None):
    """One-sample contrast against zero, with curve-invalid seed metadata.

    Several pre-registered readouts (the null-arm error margin, the entropy
    and gate drifts) are one-sample claims: the per-seed value is compared
    with 0, not with another arm.  They still need the same seed accounting
    and curve-invalid sensitivity as a paired contrast - a drift computed
    on a run whose curve never learned cannot support a mechanism claim -
    so this helper attaches ``diffs``/``seeds_used``/``curve_invalid_seeds``
    in the form :meth:`ContrastLog.finalize` already consumes.  ``bound``
    selects the bootstrap-TOST equivalence test instead of the directional
    CI (the return value then carries ``test='equivalence'`` from
    :func:`equivalence_test`).  Runs without a training seed cannot be
    independent units; they are dropped and recorded like an unpaired seed.
    """
    runs = _materialize_runs(runs)
    missing = [r for r in runs if _paired_seed(r) is None]
    by_seed = {}
    for r in runs:
        seed = _paired_seed(r)
        if seed is not None:
            by_seed[seed] = r
    used, values = [], []
    for seed in sorted(by_seed, key=_seed_order):
        value = by_seed[seed].get(metric)
        if _finite(value):
            used.append(seed)
            values.append(float(value))
    ci = (equivalence_test(values, bound) if bound is not None
          else paired_bootstrap_ci(values))
    dropped = sorted(set(by_seed) - set(used), key=_seed_order)
    ci["diffs"] = list(values)
    ci["seeds_used"] = used
    ci["seeds_dropped"] = dropped + [None] * len(missing)
    ci["seeds_unpaired"] = [None] * len(missing)
    ci["seeds_undefined"] = dropped
    ci["curve_invalid_seeds"] = _curve_invalid_seeds_for(used, (by_seed,))

    def _stream_key(seed):
        if isinstance(seed, str):
            return ("raw", seed)
        try:
            return ("stream", stream_seed_key(seed))
        except (TypeError, ValueError, OverflowError):
            return ("raw", str(seed))

    counts = {}
    for seed in by_seed:
        key = _stream_key(seed)
        counts[key] = counts.get(key, 0) + 1
    aliases = sorted(
        {seed for seed in by_seed if counts[_stream_key(seed)] > 1},
        key=_seed_order)
    if aliases:
        # A one-sample claim has one unit per independent stream.  Two raw
        # seeds that alias the same 32-bit stream (0 vs 2**32) are one run
        # counted twice, exactly as the paired audit treats them.
        ci["seed_aliases"] = aliases
        ci["controlled"] = {
            "controlled": False,
            "mismatches": [
                f"one-sample contrast repeats training stream(s) "
                f"{aliases}: aliased seeds are not independent units"],
            "varying_factors": {},
        }
    return ci


def _interaction_manipulation(audit: dict) -> dict:
    """The factor a controlled contrast actually varies, in comparable form.

    ``_controlled_audit`` reports two descriptions: ``varying_factors``
    from the declared labels (always present) and ``model_factor_diffs``
    from the resolved ``model_cfg`` (present when the records carry it).
    The resolved one is ground truth for *which* factor a contrast varies,
    but the label description mixes genuine factors with the
    ``calib_variant`` cell label: two legitimate interaction marginals like
    full-vs-ent_only and len_only-vs-no_terms manipulate the same
    ``calib_use_len`` flag while carrying different cell labels, so
    comparing the label sets would refuse a valid factorial contrast.
    Prefer the resolved diffs; fall back to the labels only for records
    that predate ``model_cfg``.
    """
    if not isinstance(audit, Mapping):
        return {}
    resolved = audit.get("model_factor_diffs") or {}
    if resolved:
        return {key: value for key, value in resolved.items()}
    if "model_cfg" in audit:
        # The resolved configs are available and say no factor differs;
        # stale declared labels must not resurrect a manipulation.
        return {}
    return dict(audit.get("varying_factors") or {})


_CLASS_METRIC_COMPANIONS = (
    ("recall_macro",
     "balanced accuracy: unweighted mean per-class recall, target-present "
     "classes"),
    ("f1_macro",
     "macro F1: unweighted mean per-class F1, target-present classes "
     "(micro F1 equals accuracy)"),
    ("f1_weighted",
     "support-weighted F1, target-present classes"),
)


def _class_frequency_companions(rules, title):
    """Re-read accuracy contrasts on class-frequency-robust metrics.

    Accuracy is only prior-invariant while the answer stream is balanced.
    The E1/E3 runners audit that premise and gate claims on it, but the
    *decision* rows are still accuracy-only: a stable accuracy gain can
    hide a macro-F1 loss when one answer class absorbs the errors.  These
    companions keep the identical seed pairing and hierarchy (seed-level
    paired bootstrap of ``a - b``) and explicitly label the averaging
    convention for each metric.  They are exploratory robustness readouts,
    not additional pre-registered decisions, so they are reported outside
    the FDR family and never overwrite the registered verdict.
    """
    rows = []
    print(f"\n[{title}] class-frequency robustness companions "
          f"(exploratory, outside the FDR family; seed-paired a - b):")
    print(f"  {'contrast':<34} {'metric':<15} {'effect [95% CI]':>29} "
          f"{'n':>3}  {'vs accuracy':>18}")
    for item in _materialize_runs(rules):
        if not isinstance(item, (tuple, list)) or len(item) != 4:
            # A malformed companion rule is not a contrast; skip it so one
            # bad record cannot abort the rest of the report.
            continue
        label, runs_a, runs_b, hypothesis = item
        label = str(label)
        # Reuse one materialized sample for the accuracy row and all three
        # macro companions: a generator would otherwise be exhausted by the
        # first contrast and the remaining rows would silently read n=0.
        runs_a = _mapping_runs(runs_a)
        runs_b = _mapping_runs(runs_b)
        acc = _paired_diff(runs_a, runs_b, "accuracy")
        for metric, convention in _CLASS_METRIC_COMPANIONS:
            ci = _paired_diff(runs_a, runs_b, metric)
            sign_disagrees = None
            uncertainty_disagrees = None
            if (_finite(acc.get("mean")) and _finite(ci.get("mean"))
                    and acc["mean"] != 0.0 and ci["mean"] != 0.0):
                sign_disagrees = ((acc["mean"] > 0.0)
                                  != (ci["mean"] > 0.0))
            if ci.get("n") and acc.get("n"):
                acc_sig = bool(_finite(acc.get("lo"))
                               and _finite(acc.get("hi"))
                               and (acc["lo"] > 0.0 or acc["hi"] < 0.0))
                met_sig = bool(_finite(ci.get("lo"))
                               and _finite(ci.get("hi"))
                               and (ci["lo"] > 0.0 or ci["hi"] < 0.0))
                uncertainty_disagrees = (acc_sig != met_sig)
            if sign_disagrees:
                note = "sign flips"
            elif uncertainty_disagrees:
                note = "CI decisive differs"
            else:
                note = "consistent"
            effect = (f"{ci['mean']:+.4f} [{ci['lo']:+.4f}, "
                      f"{ci['hi']:+.4f}]"
                      if (_finite(ci.get("mean")) and _finite(ci.get("lo"))
                          and _finite(ci.get("hi"))) else "n/a")
            print(f"  {label:<34} {metric:<15} {effect:>29} "
                  f"{ci.get('n', 0):>3}  {note:>18}")
            rows.append({
                "contrast": label,
                "hypothesis": hypothesis,
                "metric": metric,
                "convention": convention,
                "effect": ci.get("mean"),
                "lo": ci.get("lo"),
                "hi": ci.get("hi"),
                "n": ci.get("n", 0),
                "seeds_used": ci.get("seeds_used", []),
                "baseline_mean": ci.get("baseline_mean"),
                "accuracy_mean": acc.get("mean"),
                "accuracy_lo": acc.get("lo"),
                "accuracy_hi": acc.get("hi"),
                "accuracy_n": acc.get("n", 0),
                "sign_disagrees_with_accuracy": sign_disagrees,
                "uncertainty_disagrees_with_accuracy": (
                    uncertainty_disagrees),
            })
    return {
        "title": title,
        "outside_fdr_family": True,
        "note": ("same seed pairing and paired bootstrap as the registered "
                 "accuracy row; macro metrics exclude answer classes with "
                 "zero target support, and f1_weighted weights by support. "
                 "These rows are robustness companions, not extra "
                 "pre-registered decisions."),
        "rows": rows,
    }


def _paired_did(runs_a1, runs_b1, runs_a2, runs_b2, metric):
    """Paired difference-in-differences ``(a1 - b1) - (a2 - b2)``.

    Tests an *interaction* claim - e.g. "the calibrated gate's advantage
    over the query-only gate is specific to the dynamic-normalisation arm"
    - rather than a marginal effect.  Both differences are taken on the
    same training seed, so between-seed variance (the dominant term at
    small seed counts) cancels exactly as in a paired contrast.

    The controlled audit is the conjunction of the four marginal audits:
    within each arm the recipe is fixed and only the intended factor
    varies, and all four cells share one recipe and one seed set.  Both
    marginal contrasts must vary the *same* resolved factor with the same
    level sets, and both cross-context edges (a1 vs a2, b1 vs b2) must
    vary the same context factor, so the DiD cannot difference two
    different manipulations; the marginal and context factors must also be
    different, otherwise one factor varies on both axes and the row is not
    a factorial interaction (see :func:`_interaction_manipulation` for why
    the resolved ``model_cfg`` diff, not the cell label, is compared).  A
    contrast whose audit fails is refused downstream exactly like a
    marginal one, and the effective paired seed set is recorded.  ``arms``
    carries each marginal difference (a = a1-b1, b = a2-b2) with its mean
    +- std over the interaction sample, so the row shows the levels it
    differences; ``baseline_mean`` is the control marginal (b = a2-b2),
    which is what the interaction modifies.  Runs without a training seed
    cannot be paired; they are dropped and recorded as unpaired.
    """
    def _by_seed(rs):
        out = {}
        for r in rs:
            seed = _paired_seed(r)
            if seed is not None:
                out[seed] = r
        return out

    runs_a1, runs_b1, runs_a2, runs_b2 = (
        _materialize_runs(rs)
        for rs in (runs_a1, runs_b1, runs_a2, runs_b2))
    missing = [r for rs in (runs_a1, runs_b1, runs_a2, runs_b2)
               for r in rs if _paired_seed(r) is None]
    m1a, m1b, m2a, m2b = (_by_seed(rs)
                          for rs in (runs_a1, runs_b1, runs_a2, runs_b2))
    planned = sorted(set(m1a) | set(m1b) | set(m2a) | set(m2b),
                     key=_seed_order)
    diffs, used, dropped, unpaired, undefined = [], [], [], [], []
    for seed in planned:
        cells = [m.get(seed) for m in (m1a, m1b, m2a, m2b)]
        if any(c is None for c in cells):
            dropped.append(seed)
            unpaired.append(seed)
            continue
        vals = [c.get(metric) for c in cells]
        finite = [_finite(v) for v in vals]
        if all(finite):
            used.append(seed)
            diffs.append((float(vals[0]) - float(vals[1]))
                         - (float(vals[2]) - float(vals[3])))
        elif not any(finite):
            dropped.append(seed)
            undefined.append(seed)
        else:
            dropped.append(seed)
            unpaired.append(seed)
    ci = paired_bootstrap_ci(diffs)
    # The two arms of an interaction are the two *marginal differences*
    # (a1-b1 and a2-b2), each summarised with mean +- std over the same
    # four-cell paired seeds the interaction itself uses.  Restricting to
    # ``used`` (not ``planned``) keeps the printed marginal levels on the
    # interaction sample: a seed missing one cell must not contribute to
    # an arm but not to the DiD.
    arms = {}
    for name, (ra, rb) in (("a", (runs_a1, runs_b1)), ("b", (runs_a2, runs_b2))):
        ma, mb = _by_seed(ra), _by_seed(rb)
        ds = [float(ma[s][metric]) - float(mb[s][metric]) for s in used
              if s in ma and s in mb
              and _finite(ma[s].get(metric)) and _finite(mb[s].get(metric))]
        arms[name] = _mean_std(ds)
    ci["arms"] = arms
    ci["arms_note"] = ("marginal differences (a1-b1 vs a2-b2) on the "
                       "interaction sample")
    # The interaction modifies the second marginal effect (a2-b2), so
    # ``rel_effect`` must be relative to that control marginal - not to
    # the difference between the two b-levels, which is a different
    # quantity and can be near zero even when the interaction is real.
    ci["baseline_mean"] = arms["b"]["mean"]
    ci["diffs"] = list(diffs)
    ci["curve_invalid_seeds"] = _curve_invalid_seeds_for(
        used, (m1a, m1b, m2a, m2b))
    ci["seeds_used"] = sorted(used, key=_seed_order)
    ci["seeds_dropped"] = (sorted(dropped, key=_seed_order)
                           + [None] * len(missing))
    ci["seeds_unpaired"] = (sorted(unpaired, key=_seed_order)
                            + [None] * len(missing))
    ci["seeds_undefined"] = sorted(undefined, key=_seed_order)
    audits = []
    for left, right in ((runs_a1, runs_b1), (runs_a2, runs_b2),
                        (runs_a1, runs_a2), (runs_b1, runs_b2)):
        clean_left = [r for r in left if _paired_seed(r) is not None]
        clean_right = [r for r in right if _paired_seed(r) is not None]
        audit = _controlled_audit(clean_left, clean_right)
        if len(clean_left) != len(left) or len(clean_right) != len(right):
            audit["mismatches"].append(
                "interaction edge contains run(s) without a usable "
                "(hashable, non-boolean) training seed")
            audit["controlled"] = False
        audits.append(audit)
    mismatches = [msg for audit in audits
                  for msg in audit.get("mismatches", [])]
    # A difference-in-differences is only an interaction of *the same*
    # manipulation measured under two contexts: both marginal contrasts
    # must vary the same declared factor with the same level sets (e.g.
    # calib - query under seednorm and under pre).  Otherwise the DiD
    # mixes two different manipulations and cannot be read as one
    # interaction.
    marginal_factors = [_interaction_manipulation(a) for a in audits[:2]]
    if marginal_factors[0] != marginal_factors[1]:
        mismatches.append(
            "interaction marginals vary different factor(s): "
            f"{sorted(marginal_factors[0])} vs "
            f"{sorted(marginal_factors[1])}")
    # The cross-context edges (a1 vs a2 and b1 vs b2) are what the
    # interaction conditions on: both must vary the *same* context factor
    # with the same level sets, otherwise the DiD differences two different
    # context manipulations.  A marginal factor reappearing on the context
    # axis is refused too: then the design varies one factor on both axes
    # and the "interaction" is not a factorial contrast.
    context_factors = [_interaction_manipulation(a) for a in audits[2:]]
    if context_factors[0] != context_factors[1]:
        mismatches.append(
            "interaction contexts vary different factor(s): "
            f"{sorted(context_factors[0])} vs "
            f"{sorted(context_factors[1])}")
    overlap = sorted(set(marginal_factors[0]) & set(context_factors[0]))
    if overlap:
        mismatches.append(
            f"interaction marginal and context edges vary the same "
            f"factor(s) {overlap}: a factorial interaction needs two "
            f"different factors")
    factors = {}
    for audit in audits:
        factors.update(audit.get("varying_factors", {}))
    ci["controlled"] = {"controlled": not mismatches,
                        "mismatches": mismatches,
                        "varying_factors": factors}
    return ci


def _paired_equivalence(runs_a, runs_b, metric, bound, mode="equivalence"):
    """Paired bootstrap-TOST (or non-inferiority) contrast with full audit.

    A null prediction ("arm A stays close to arm B", or "arm A is not
    materially worse than arm B") still needs the same seed pairing and
    recipe audit as a directional contrast: equivalence computed on
    mispaired or recipe-confounded arms would certify the wrong thing.
    ``mode`` selects the two-sided equivalence test or the one-sided
    non-inferiority test.  Reuses :func:`_paired_diff`'s exact pairing and
    copies its audit/arm/seed metadata onto the test result, so the printed
    row and the decision table carry the same sample the effect was computed
    on.
    """
    paired = _paired_diff(runs_a, runs_b, metric)
    diffs = paired.get("diffs") or []
    ci = (noninferiority_test(diffs, bound) if mode == "noninferior"
          else equivalence_test(diffs, bound))
    for key in ("arms", "arms_all", "arms_note", "baseline_mean",
                "diffs", "curve_invalid_seeds", "seeds_used",
                "seeds_dropped", "seeds_unpaired", "seeds_undefined",
                "stream_test", "controlled"):
        if key in paired:
            ci[key] = paired[key]
    return ci


_COUNTERFACTUAL_METRICS = (
    "cf_value_follow_rate",
    "cf_value_stubborn_rate",
    "cf_value_accuracy",
    "cf_value_flip_rate",
    "cf_distractor_unchanged_rate",
    "cf_distractor_accuracy",
    "cf_relocation_unchanged_rate",
    "cf_relocation_accuracy",
    "cf_n",
    "cf_n_original_correct",
    "cf_n_distractor",
    "cf_n_relocation",
)
