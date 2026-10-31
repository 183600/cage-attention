# cage/experiments/_contrastlog_finalize.py - ContrastLog.finalize: multiplicity control and reporting.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _fcr_decision_holds(row, lo, hi) -> bool:
    """Whether an FCR-adjusted interval still supports the row's prediction.

    The ordinary decision rule is unchanged; this only re-reads the same
    directional/equivalence/NI predicate at the wider selection-adjusted
    bounds.  NaN bounds or a malformed row return ``False`` so an
    unavailable interval can never be counted as support.
    """
    if not (_finite(lo) and _finite(hi)):
        return False
    test = row.get("test")
    if test == "equivalence":
        bound = row.get("equivalence_bound")
        return bool(_finite(bound) and lo > -float(bound)
                    and hi < float(bound))
    if test == "noninferiority":
        bound = row.get("noninferiority_bound")
        return bool(_finite(bound) and lo > -float(bound))
    direction = row.get("direction")
    if direction == "+":
        return bool(lo > 0.0)
    if direction == "-":
        return bool(hi < 0.0)
    return bool(lo > 0.0 or hi < 0.0)


def _contrastlog_finalize(self, out_dir: pathlib.Path, q: float = 0.05) -> dict:
    # TOST p-values of equivalence rows enter the same BH family as the
    # directional p-values: both answer "is this contrast worth a claim
    # at FDR q?" and splitting the family would loosen the control.
    adj = benjamini_hochberg([r["p_value"] for r in self.rows], q=q)
    for row, a in zip(self.rows, adj):
        row["p_adj"] = a["p_adj"]
        # Keep the raw BH selection event separate from the claim gate:
        # the FCR interval must condition on every BH-selected contrast,
        # including one later refused because its design audit failed.
        row["bh_selected"] = bool(a["reject"])
        # A confounded contrast cannot support its hypothesis even if
        # its p-value is small: refuse the claim at the decision point.
        row["fdr_reject"] = bool(a["reject"] and row.get("controlled") is not False)
        row["confounded"] = row.get("controlled") is False
        base = row.get("baseline_mean")
        row["rel_effect"] = (
            row["mean"] / abs(base)
            if base is not None and math.isfinite(base) and base != 0.0
            else float("nan"))
        sesoi = row.get("sesoi")
        mean = row.get("mean")
        finite_effect = mean is not None and math.isfinite(mean)
        if row["test"] == "equivalence":
            # Null prediction: the practical gate is the CI-within-
            # bound rule, *not* |effect| >= SESOI (which would reward
            # exactly the drift the prediction says should not occur).
            row["practically_relevant"] = bool(row.get("equivalent"))
        elif row["test"] == "noninferiority":
            # One-sided null prediction ("the arm is not materially
            # worse"): support is the CI's lower bound staying above the
            # negative bound, so a loss beyond it is a refutation rather
            # than "no significant difference".
            row["practically_relevant"] = bool(row.get("noninferior"))
        else:
            # Directional pre-registration: a prediction like
            # "null_mass_error_margin > 0" is only confirmed by an
            # effect in *that* direction.  A significant effect of the
            # opposite sign refutes the hypothesis, so the sign is
            # enforced here (before the SESOI magnitude gate) and such
            # a row can never count as support.
            direction = row.get("direction")
            row["sign_ok"] = (None if direction is None
                              or not finite_effect
                              else bool(mean > 0.0 if direction == "+"
                                        else mean < 0.0))
            row["direction_reversed"] = bool(
                direction is not None and finite_effect
                and not row["sign_ok"])
            row["practically_relevant"] = bool(
                not row["direction_reversed"]
                and (sesoi is None
                     or (finite_effect and math.isfinite(float(sesoi))
                         and abs(mean) >= float(sesoi))))
        # Pre-registered seed floor: below ``min_seeds`` independent
        # paired seeds a "supported" verdict would be noise dressed as
        # evidence; such rows cannot support a claim at the decision
        # point whatever their p-value.  HB0's competence prerequisite
        # gates the same way: no claim from an underfit grid.
        row["competence_passed"] = (
            None if self.competence is None
            else bool(self.competence.get("passed")))
        coverage_failures = []
        for payload in self.coverage:
            if payload.get("complete") is False:
                coverage_failures.append(
                    f"{payload.get('tag', '?')}/{payload.get('design', 'design')} "
                    f"missing cells {payload.get('missing_cells')}")
            if payload.get("seed_balanced") is False:
                coverage_failures.append(
                    f"{payload.get('tag', '?')}/{payload.get('design', 'design')} "
                    f"seed-unbalanced cells {payload.get('unbalanced_cells')}")
        row["coverage_gate_failed"] = bool(coverage_failures)
        row["coverage_failures"] = coverage_failures
        row["seed_floor_failed"] = bool(
            (row.get("n") or 0) < self.min_seeds)
        row["claim_eligible"] = bool(
            (row.get("n") or 0) >= self.min_seeds
            and row["competence_passed"] is not False
            and not coverage_failures)
        # Curve-invalid sensitivity: a supported effect that rests on
        # seeds whose run never learned (plateau/diverging/...) is not
        # evidence about the mechanism.  Drop those seeds and recompute
        # the same effect; if the support disappears (or the surviving
        # sample falls below the pre-registered seed floor), the row is
        # marked ineligible.  This is a sensitivity gate, not a second
        # decision rule: rows without curve-invalid seeds are unchanged.
        bad_seeds = set(row.get("curve_invalid_seeds") or [])
        row["curve_invalid_n"] = len(bad_seeds)
        row["curve_valid_n"] = row.get("n")
        row["curve_invalid_removes_support"] = False
        if bad_seeds:
            paired = list(zip(row.get("seeds_used") or [],
                              row.get("diffs") or []))
            valid_diffs = [d for seed, d in paired
                           if seed not in bad_seeds]
            row["curve_valid_n"] = len(valid_diffs)
            if row["test"] == "equivalence":
                sens = equivalence_test(
                    valid_diffs, row.get("equivalence_bound"))
                sens_ok = bool(sens.get("equivalent"))
            elif row["test"] == "noninferiority":
                sens = noninferiority_test(
                    valid_diffs, row.get("noninferiority_bound"))
                sens_ok = bool(sens.get("noninferior"))
            else:
                sens = paired_bootstrap_ci(valid_diffs)
                direction = row.get("direction")
                if direction == "+":
                    sign_ok = bool(sens["mean"] > 0.0 and sens["lo"] > 0.0)
                elif direction == "-":
                    sign_ok = bool(sens["mean"] < 0.0 and sens["hi"] < 0.0)
                else:
                    sign_ok = bool(sens["lo"] > 0.0 or sens["hi"] < 0.0)
                row_sesoi = row.get("sesoi")
                magnitude_ok = (
                    row_sesoi is None
                    or (math.isfinite(sens["mean"])
                        and abs(sens["mean"]) >= float(row_sesoi)))
                sens_ok = bool(sign_ok and magnitude_ok)
            row["curve_valid_mean"] = sens.get("mean")
            row["curve_valid_lo"] = sens.get("lo")
            row["curve_valid_hi"] = sens.get("hi")
            row["curve_valid_p"] = sens.get("p_value")
            # The replication forecast must use the same seed sample as
            # the sensitivity decision, not the contaminated one: a run
            # that never learned is no more evidence about a future
            # replication than it is about the current effect.
            rep_mode = ("equivalence" if row["test"] == "equivalence"
                        else "noninferiority"
                        if row["test"] == "noninferiority"
                        else "difference")
            rep_bound = (row.get("equivalence_bound")
                         if row["test"] == "equivalence"
                         else row.get("noninferiority_bound")
                         if row["test"] == "noninferiority"
                         else None)
            row["replication_curve_valid"] = replication_interval(
                valid_diffs, direction=row.get("direction"),
                mode=rep_mode, bound=rep_bound)
            if (row["claim_eligible"] and row["fdr_reject"]
                    and row["practically_relevant"]
                    and (len(valid_diffs) < self.min_seeds
                         or not sens_ok)):
                row["claim_eligible"] = False
                row["curve_invalid_removes_support"] = True
        # The seed-level effect and the item-level paired effect must
        # at least share a sign.  Opposite signs mean the two
        # summaries are not measuring the same contrast (e.g. unequal
        # per-seed item counts or a generated mismatch), so the row
        # cannot support a directional claim.
        if (row["claim_eligible"]
                and row.get("stream_direction_agrees") is False):
            row["claim_eligible"] = False
            row["stream_sign_conflict"] = True
        # Accuracy is the pre-registered headline only under the
        # measured balance premise.  If the held-out stream failed the
        # gross-imbalance audit, the same accuracy contrast is partly a
        # class-prior contrast: keep it in the table for transparency
        # but refuse the claim (macro-F1 / PR-AUC companions can still
        # carry the mechanism reading).
        if (not self.balance_ok and row["claim_eligible"]
                and row["metric"] in _ACCURACY_METRICS):
            row["claim_eligible"] = False
            row["accuracy_balance_failed"] = True
        row["supports_hypothesis"] = bool(
            row["fdr_reject"] and row["practically_relevant"]
            and row["claim_eligible"])
        # Replication companion: a supported effect whose 95%
        # same-size-repeat interval still includes zero (or crosses a
        # null-prediction bound) is not expected to reproduce its sign
        # at 95% in a fresh run of the same size.  This annotates, and
        # never changes, the pre-registered decision above.
        rep = (row.get("replication_curve_valid")
               or row.get("replication") or {})
        row["replication_effective"] = rep
        row["replication_uncertain"] = bool(
            row["supports_hypothesis"] and rep.get("valid")
            and not rep.get("replicates"))
        row["replication_prob"] = rep.get("prob_replicates")
        # Small-sample robustness: does the *decision* survive when any
        # single seed is dropped (jackknife), and does BCa reach the
        # same decision as the pre-registered percentile interval?
        # These are annotations, not decision rules: the verdict above
        # stays exactly as pre-registered, but a supported verdict that
        # is this fragile is flagged below.
        claim = row["supports_hypothesis"] and (row.get("n") or 0) >= 2
        if row["test"] == "equivalence":
            row["loo_flips_decision"] = bool(
                claim and row.get("loo_within_bound") is False)
        elif row["test"] == "noninferiority":
            row["loo_flips_decision"] = bool(
                claim and row.get("loo_above_bound") is False)
        else:
            row["loo_flips_decision"] = bool(
                claim and row.get("loo_sign_flip"))
        row["bca_disagrees"] = row.get("bca_agrees") is False
        # A-priori power planning for the pre-registered SESOI at the
        # observed paired-difference SD: how many seeds would this
        # contrast need to resolve an effect it cares about?  (Adjust
        # the design *before* the next run, not after.)  For an
        # equivalence row the target is the bound: how many seeds
        # until a CI of this width could fit inside +-SESOI.
        if sesoi is None:
            row["n_seeds_for_sesoi"] = float("inf")
        else:
            # ``_required_seeds`` names the one-sided mode
            # ``"noninferior"`` (the registry's spelling), while the row
            # stores the test name ``"noninferiority"``.  Passing the
            # latter straight through used to make the helper reject the
            # mode and report every non-inferiority power plan as
            # unattainable (``inf``).
            mode = ("equivalence" if row["test"] == "equivalence"
                    else "noninferior"
                    if row["test"] == "noninferiority"
                    else "superiority")
            required = _required_seeds(
                mode, row.get("sd"), sesoi, q, 0.8)
            row["n_seeds_for_sesoi"] = (
                float(required) if _finite(required)
                else float("inf"))
    # Minimum-effect (SESOI) companion.  The pre-registered practicality
    # gate above compares the *point estimate* with the SESOI, so a row
    # whose 95% CI still spans a negligible effect can be labelled
    # practically relevant on the point estimate alone.  This parallel
    # family asks the interval-level question - does the whole CI clear
    # the SESOI in the registered direction? - with the bootstrap
    # minimum-effect test.  It does not change ``supports_hypothesis``
    # (that rule is pre-registered); a supported row that fails it is
    # instead reported as "SESOI point-estimate only", so a claim is not
    # read as practically established when the data are also compatible
    # with an effect below the practical floor.
    for row in self.rows:
        direction = row.get("direction")
        sesoi = row.get("sesoi")
        row["minimum_effect_p"] = float("nan")
        row["minimum_effect_established"] = False
        row["minimum_effect_loo"] = None
        row["minimum_effect_bca_agrees"] = None
        if (row.get("test") == "difference"
                and direction in ("+", "-")
                and sesoi is not None and _finite(sesoi)
                and float(sesoi) > 0.0
                and len(row.get("diffs") or []) >= 2):
            met = minimum_effect_test(
                row.get("diffs"), float(sesoi), direction=direction)
            row["minimum_effect_p"] = met.get("p_value")
            row["minimum_effect_lo"] = met.get("lo")
            row["minimum_effect_hi"] = met.get("hi")
            row["minimum_effect_loo"] = met.get("loo_beyond_sesoi")
            row["minimum_effect_bca_agrees"] = met.get("bca_agrees")
            row["minimum_effect_established"] = bool(
                met.get("beyond_sesoi"))
    met_adj = benjamini_hochberg(
        [r.get("minimum_effect_p") for r in self.rows], q=q)
    for row, met_a in zip(self.rows, met_adj):
        row["minimum_effect_p_adj"] = met_a["p_adj"]
        row["minimum_effect_fdr_reject"] = bool(
            met_a["reject"]
            and row.get("controlled") is not False
            and row.get("claim_eligible", True))
        row["minimum_effect_established"] = bool(
            row["minimum_effect_fdr_reject"]
            and row.get("minimum_effect_established"))
        row["sesoi_point_estimate_only"] = bool(
            row.get("test") == "difference"
            and row.get("direction") in ("+", "-")
            and row.get("sesoi") is not None
            and row.get("supports_hypothesis")
            and row.get("claim_eligible", True)
            and not row["minimum_effect_established"])
    # Hypothesis-level evidence synthesis.  The rows above are the
    # decision unit (the pre-registered FDR family), but the hypotheses
    # themselves are conjunctions; a single surviving row must not be
    # read as "the hypothesis is confirmed".  This companion groups the
    # rows and reports supported / partial / refuted / inconclusive, so
    # the report cannot silently turn "A and B and C" into "any of
    # A, B, C".  It is a reading companion and changes no decision.
    cited_hypotheses = sorted({
        r.get("hypothesis") for r in self.rows if r.get("hypothesis")})
    synthesis_order = list(self.hypotheses) + [
        h for h in cited_hypotheses if h not in self.hypotheses]
    synthesis = _hypothesis_evidence_synthesis(self.rows,
                                               synthesis_order, q=q)
    synth_supported = [h for h, v in synthesis.items()
                       if v["status"] == "supported"]
    synth_partial = [h for h, v in synthesis.items()
                     if v["status"] == "partial"]
    synth_refuted = [h for h, v in synthesis.items()
                     if v["status"] == "refuted"]
    synth_inconclusive = [h for h, v in synthesis.items()
                          if v["status"] == "inconclusive"]
    synth_undirected = [h for h, v in synthesis.items()
                        if v["status"] == "undirected"]
    synth_no_contrast = [h for h, v in synthesis.items()
                         if v["status"] == "no_contrast"]
    synth_rep_uncertain = [
        h for h, v in synthesis.items()
        if v["n_replication_uncertain"] and v["status"] in (
            "supported", "partial", "undirected")]
    synth_provisional = [
        h for h, v in synthesis.items()
        if v["n_practically_provisional"]]
    synth_partial_registry = [
        h for h, v in synthesis.items()
        if v["status"] == "supported"
        and v.get("registry_complete") is False]
    synth_not_family_significant = [
        h for h, v in synthesis.items()
        if v["status"] == "supported"
        and not v.get("conjunction_fdr_reject")]
    synth_claim_level = [
        h for h, v in synthesis.items()
        if v.get("claim_level_supported")]
    synth_dependence_sensitive = [
        h for h, v in synthesis.items()
        if v.get("claim_level_supported")
        and not (v.get("conjunction_by_reject")
                 and v.get("conjunction_holm_reject"))]
    # Parallel exact-test family: the same BH correction applied to the
    # paired sign-flip p-values.  This is a companion, not a replacement
    # for the pre-registered bootstrap decision; it answers "could this
    # design have produced this p at all?".  It runs after every row has
    # its final claim-eligibility flags, so the two verdicts are
    # compared on the same eligible sample.
    exact_adj = benjamini_hochberg(
        [r.get("sign_flip_p") for r in self.rows], q=q)
    for row, exact_a in zip(self.rows, exact_adj):
        row["sign_flip_p_adj"] = exact_a["p_adj"]
        row["sign_flip_fdr_reject"] = bool(
            exact_a["reject"]
            and row.get("controlled") is not False
            and row.get("claim_eligible", True))
        row["sign_flip_supports"] = bool(
            row["sign_flip_fdr_reject"]
            and row.get("practically_relevant")
            and row.get("claim_eligible", True))
        row["sign_flip_disagrees"] = bool(
            _finite(row.get("p_adj"))
            and _finite(row.get("sign_flip_p_adj"))
            and bool(row["supports_hypothesis"])
            != bool(row["sign_flip_supports"]))
        row["sign_flip_cannot_reject"] = bool(
            _finite(row.get("sign_flip_min_p"))
            and float(row["sign_flip_min_p"]) > q)
    testable = [r for r in self.rows if math.isfinite(r["p_adj"])]
    # A contrast is unfalsifiable by the exact test when even its
    # smallest attainable p is above ``q``.  The per-row BH threshold is
    # at most ``q`` (a row with p <= q can reject when it is compared at
    # the most permissive rank), so the family size does not raise the
    # floor for a particular row; using q / n_testable here labelled
    # rejectable rows as unfalsifiable whenever another exact p was
    # smaller.
    for exact_row in testable:
        exact_row["sign_flip_cannot_reject"] = bool(
            _finite(exact_row.get("sign_flip_min_p"))
            and float(exact_row["sign_flip_min_p"]) > q)
    surviving = [r for r in testable if r["fdr_reject"]]
    supporting = [r for r in testable if r["supports_hypothesis"]]
    # Selection-adjusted (false-coverage-rate) companion.  The BH family
    # answers "which contrasts may be read as discoveries?"; it does not
    # repair the *interval* reported for those selected contrasts.  A
    # selected set is enriched for large observed effects, so its ordinary
    # 95% intervals under-cover the selected parameters.  The FCR interval
    # uses the Benjamini-Yekutieli level ``1 - q * R / m`` (and the
    # harmonic-factor variant under arbitrary dependence), so a supported
    # row whose claim disappears once selection is accounted for is named
    # instead of being presented as robust.  This is a companion: the
    # pre-registered decision above is unchanged.
    selected = [r for r in testable if r.get("bh_selected")]
    n_selected = len(selected)
    n_tests = len(testable)
    selected_ids = {id(r) for r in selected}
    fcr_sensitive = []
    fcr_by_sensitive = []
    fcr_unavailable = []
    fcr_by_unavailable = []
    empty_fcr = {
        "valid": False, "reason": "not selected by BH",
        "n": 0, "n_boot": 0, "q": q, "n_selected": n_selected,
        "n_tests": n_tests, "dependence": "prds",
        "harmonic_factor": float("nan"), "adjusted_alpha": float("nan"),
        "confidence_level": float("nan"), "mean": float("nan"),
        "lo": float("nan"), "hi": float("nan"),
        "half_width": float("nan"), "excludes_zero": False,
    }
    for row in self.rows:
        is_selected = id(row) in selected_ids
        row["fcr_selected"] = bool(is_selected)
        row["fcr_n_selected"] = n_selected
        row["fcr_n_tests"] = n_tests
        fcr = dict(empty_fcr)
        fcr_by = dict(empty_fcr)
        fcr_by["dependence"] = "arbitrary"
        if is_selected:
            fcr = false_coverage_ci(
                row.get("diffs"), n_selected, n_tests, q=q,
                dependence="prds")
            fcr_by = false_coverage_ci(
                row.get("diffs"), n_selected, n_tests, q=q,
                dependence="arbitrary")
        row["fcr"] = fcr
        row["fcr_by"] = fcr_by
        row["fcr_lo"] = fcr.get("lo")
        row["fcr_hi"] = fcr.get("hi")
        row["fcr_adjusted_alpha"] = fcr.get("adjusted_alpha")
        row["fcr_confidence_level"] = fcr.get("confidence_level")
        row["fcr_decision_holds"] = bool(
            is_selected and _fcr_decision_holds(
                row, fcr.get("lo"), fcr.get("hi")))
        row["fcr_supports"] = bool(
            row["fcr_decision_holds"]
            and row.get("practically_relevant")
            and row.get("claim_eligible", True)
            and row.get("controlled") is not False)
        row["fcr_disagrees"] = bool(
            is_selected and bool(row.get("supports_hypothesis"))
            != row["fcr_supports"])
        row["fcr_by_lo"] = fcr_by.get("lo")
        row["fcr_by_hi"] = fcr_by.get("hi")
        row["fcr_by_adjusted_alpha"] = fcr_by.get("adjusted_alpha")
        row["fcr_by_confidence_level"] = fcr_by.get("confidence_level")
        row["fcr_by_decision_holds"] = bool(
            is_selected and _fcr_decision_holds(
                row, fcr_by.get("lo"), fcr_by.get("hi")))
        row["fcr_by_supports"] = bool(
            row["fcr_by_decision_holds"]
            and row.get("practically_relevant")
            and row.get("claim_eligible", True)
            and row.get("controlled") is not False)
        row["fcr_by_disagrees"] = bool(
            is_selected and bool(row.get("supports_hypothesis"))
            != row["fcr_by_supports"])
        if is_selected and row.get("supports_hypothesis"):
            if fcr.get("valid"):
                if not row["fcr_supports"]:
                    fcr_sensitive.append(row)
            else:
                fcr_unavailable.append(row)
            if fcr_by.get("valid"):
                if not row["fcr_by_supports"]:
                    fcr_by_sensitive.append(row)
            else:
                fcr_by_unavailable.append(row)
    # Multiplicity-model sensitivity companions.  BH's FDR guarantee
    # assumes independent or positively dependent tests, but contrasts
    # in one family share training seeds and a sealed evaluation
    # stream, so that premise is not established.  BY stays valid under
    # arbitrary dependence; Holm controls the family-wise error rate.
    # Both are reported next to the pre-registered BH decision, which is
    # left unchanged, so a claim that survives only under the
    # independence assumption is visible rather than assumed.
    p_family = [r.get("p_value") for r in self.rows]
    by_adj = benjamini_yekutieli(p_family, q=q)
    holm_adj = holm_bonferroni(p_family, q=q)
    for row, by_a, holm_a in zip(self.rows, by_adj, holm_adj):
        row["p_adj_by"] = by_a["p_adj"]
        row["by_fdr_reject"] = bool(
            by_a["reject"] and row.get("controlled") is not False)
        row["by_supports"] = bool(
            row["by_fdr_reject"]
            and row.get("practically_relevant")
            and row.get("claim_eligible", True))
        row["p_adj_holm"] = holm_a["p_adj"]
        row["holm_fw_reject"] = bool(
            holm_a["reject"] and row.get("controlled") is not False)
        row["holm_supports"] = bool(
            row["holm_fw_reject"]
            and row.get("practically_relevant")
            and row.get("claim_eligible", True))
        row["by_disagrees"] = bool(
            _finite(row.get("p_adj")) and _finite(row.get("p_adj_by"))
            and bool(row.get("supports_hypothesis"))
            != bool(row["by_supports"]))
        row["holm_disagrees"] = bool(
            _finite(row.get("p_adj")) and _finite(row.get("p_adj_holm"))
            and bool(row.get("supports_hypothesis"))
            != bool(row["holm_supports"]))
    family_simes = simes_test(p_family, alpha=q)
    by_supporting = [r for r in testable if r.get("by_supports")]
    holm_supporting = [r for r in testable if r.get("holm_supports")]
    by_conflicts = [r for r in testable if r.get("by_disagrees")]
    holm_conflicts = [r for r in testable if r.get("holm_disagrees")]
    dependence_weaker = [
        r for r in supporting
        if not (r.get("by_supports") and r.get("holm_supports"))]
    met_established = [
        r for r in supporting
        if r.get("claim_eligible", True)
        and r.get("minimum_effect_established")]
    sesoi_point_only = [
        r for r in supporting if r.get("sesoi_point_estimate_only")]
    replication_uncertain = [
        r for r in supporting if r.get("replication_uncertain")]
    # Standardized-effect intervals are a magnitude companion: a raw
    # effect can be supported while its paired-SD scale is too poorly
    # estimated (or the bootstrap too degenerate) to standardize.  The
    # pre-registered decision is unchanged; the row is flagged so the
    # missing effect-size precision is visible.
    effect_size_unavailable = [
        r for r in supporting if r.get("effect_size_valid") is False]
    loo_flips = [r for r in testable if r.get("loo_flips_decision")]
    bca_differs = [r for r in testable if r.get("bca_disagrees")]
    sign_flip_supporting = [
        r for r in testable if r.get("sign_flip_supports")]
    sign_flip_conflicts = [
        r for r in testable if r.get("sign_flip_disagrees")]
    sign_flip_unfalsifiable = [
        r for r in testable if r.get("sign_flip_cannot_reject")]
    sign_flip_below_floor = [
        r for r in testable
        if _finite(r.get("p_value"))
        and _finite(r.get("sign_flip_min_p"))
        and float(r["p_value"]) < float(r["sign_flip_min_p"])]
    low_resolution = [r for r in testable if r["n"] <= 5]
    # A refutation is a decision claim like any other: it needs the
    # pre-registered seed floor and an unconfounded contrast (the same
    # precedence as the printed verdict), otherwise the row stays a raw
    # sign observation without a claim.
    reversed_effects = [
        r for r in surviving
        if r.get("direction_reversed") and r.get("claim_eligible", True)
        and not r.get("confounded")]
    below_sesoi = [r for r in surviving
                   if not r["practically_relevant"]
                   and not r.get("direction_reversed")
                   and r["test"] not in ("equivalence",
                                         "noninferiority")]
    equiv_rows = [r for r in testable if r["test"] == "equivalence"]
    equivalent = [r for r in equiv_rows if r.get("equivalent")]
    equiv_borderline = [r for r in surviving
                        if r["test"] == "equivalence"
                        and not r.get("equivalent")]
    noninf_rows = [r for r in testable if r["test"] == "noninferiority"]
    noninferior = [r for r in noninf_rows if r.get("noninferior")]
    noninf_borderline = [r for r in surviving
                         if r["test"] == "noninferiority"
                         and not r.get("noninferior")]
    confounded = [r for r in testable if r.get("confounded")]
    controlled_pass = [r for r in self.rows if r.get("controlled") is True]
    audited = [r for r in self.rows if r.get("controlled") is not None]
    print(f"\n[{self.tag}] multiplicity control (Benjamini-Hochberg FDR, "
          f"q={q:g}): {len(surviving)}/{len(testable)} testable "
          f"contrasts survive, {len(supporting)} support a "
          f"pre-registered decision "
          f"(uncorrected p -> adjusted p; only 'SURVIVES'/'EQUIVALENT' "
          f"supports a pre-registered decision; rel = effect relative "
          f"to the control-side mean, dz = standardized effect (mean / "
          f"paired SD), loo = jackknife mean range - read rel/dz for "
          f"practical significance; a statistically significant effect "
          f"below its SESOI is practically negligible and does NOT "
          f"support the hypothesis; a null prediction is supported only "
          f"when the 95% CI lies inside its equivalence bound; a "
          f"positive verdict that flips when one seed is dropped, or "
          f"dissents from the BCa interval, is flagged after the "
          f"table):")
    for r in testable:
        rel_s = (f", rel={r['rel_effect']:+.2%}"
                 if math.isfinite(r["rel_effect"]) else "")
        sesoi = r.get("sesoi")
        equiv = r["test"] == "equivalence"
        noninf = r["test"] == "noninferiority"
        null_test = equiv or noninf
        sesoi_s = (f"{'bound' if equiv else ('NI bound' if noninf else 'SESOI')}"
                   f"={sesoi:g}" if sesoi is not None else "")
        if r.get("confounded"):
            verdict = "REFUSED (confounded contrast)"
        elif r.get("competence_passed") is False:
            verdict = ("INELIGIBLE (HB0 competence prerequisite failed: "
                       "no arm above chance, so the contrast cannot be "
                       "read as a mechanism statement)")
        elif r.get("accuracy_balance_failed"):
            verdict = ("INELIGIBLE (held-out answer-balance premise "
                       "failed: accuracy is a class-prior reading on "
                       "this stream; the macro-F1/PR-AUC companions, "
                       "not accuracy, carry the claim)")
        elif r.get("stream_sign_conflict"):
            verdict = ("INELIGIBLE (seed-level effect sign disagrees "
                       "with the item-level paired sign on the shared "
                       "evaluation stream: the two summaries are not "
                       "measuring the same contrast)")
        elif r.get("curve_invalid_removes_support"):
            verdict = ("INELIGIBLE (supported effect disappears after "
                       "dropping curve-invalid seeds: the claim rests "
                       "on runs that did not learn)")
        elif r.get("coverage_gate_failed"):
            failures = "; ".join(r.get("coverage_failures") or [])
            verdict = ("INELIGIBLE (design coverage gate failed: "
                       f"{failures}: the contrasted design was not "
                       "fully measured)")
        elif not r.get("claim_eligible", True):
            verdict = (f"INELIGIBLE (n={r['n']} < pre-registered "
                       f"minimum {self.min_seeds} seeds: too few "
                       f"independent runs for a claim)")
        elif equiv and r.get("equivalent") and r["fdr_reject"]:
            verdict = "EQUIVALENT (CI inside bound, null prediction holds)"
        elif noninf and r.get("noninferior") and r["fdr_reject"]:
            verdict = ("NON-INFERIOR (CI lower bound above the negative "
                       "bound: no material loss, null prediction holds)")
        elif null_test and sesoi is not None and (
                (equiv and (r["lo"] > float(sesoi)
                            or r["hi"] < -float(sesoi)))
                or (noninf and _finite(r.get("hi"))
                    and r["hi"] < -float(sesoi))):
            verdict = ("REFUTED (CI entirely beyond the "
                       f"{'equivalence' if equiv else 'non-inferiority'} "
                       "bound, null prediction fails)")
        elif not r["fdr_reject"]:
            verdict = ("not significant (TOST)" if equiv
                       else ("not significant (one-sided NI)" if noninf
                             else "not significant"))
        elif equiv:
            verdict = ("TOST-significant but CI not fully inside bound "
                       "(borderline, not read as support)")
        elif noninf:
            verdict = ("NI false positive-guard: CI lower bound not above "
                       "the negative bound (borderline, not read as "
                       "support)")
        elif r.get("direction_reversed"):
            verdict = ("REFUTED (FDR-significant effect opposite to the "
                       "pre-registered direction, no support)")
        elif not r["practically_relevant"]:
            verdict = ("FDR-significant but |effect| < SESOI "
                       "(practically negligible)")
        else:
            verdict = "SURVIVES"
        need = r.get("n_seeds_for_sesoi", float("inf"))
        need_s = ""
        if math.isfinite(need) and r["n"] > 0 and need > r["n"]:
            need_s = (f", needs ~{math.ceil(need)} seeds for "
                      f"{'the bound' if null_test else 'SESOI'} at 80% "
                      f"power (has {r['n']})")
        dz = r.get("dz", float("nan"))
        dz_lo = r.get("cohens_d_lo", float("nan"))
        dz_hi = r.get("cohens_d_hi", float("nan"))
        if math.isfinite(dz):
            dz_s = f", dz={dz:+.2f}"
            if _finite(dz_lo) and _finite(dz_hi):
                dz_s += f" [{dz_lo:+.2f}, {dz_hi:+.2f}]"
        else:
            dz_s = ""
        loo_lo, loo_hi = r.get("loo_mean_min"), r.get("loo_mean_max")
        loo_s = (f", loo=[{loo_lo:+.4f}, {loo_hi:+.4f}]"
                 if _finite(loo_lo) and _finite(loo_hi) else "")
        exact_p = r.get("sign_flip_p")
        exact_s = (f", exact p={float(exact_p):.4f}"
                   f" (min {float(r['sign_flip_min_p']):.4f})"
                   if _finite(exact_p) and _finite(r.get("sign_flip_min_p"))
                   else "")
        dep_s = (f", BY p_adj={float(r['p_adj_by']):.4f}, "
                 f"Holm p_adj={float(r['p_adj_holm']):.4f}"
                 if _finite(r.get("p_adj_by"))
                 and _finite(r.get("p_adj_holm")) else "")
        sesoi_note = ("  [SESOI point-estimate only: 95% CI does not "
                      "clear the practical bound]"
                      if r.get("sesoi_point_estimate_only") else "")
        print(f"  [{r['hypothesis']}] {r['rule']} ({r['metric']}): "
              f"p={r['p_value']:.4f} -> p_adj={r['p_adj']:.4f} "
              f"{exact_s}{dep_s} {verdict}{sesoi_note} "
              f"(effect={r['mean']:+.4f}{rel_s}{dz_s}, "
              f"MDE80={r['mde80']:.4f}{', ' if sesoi_s else ''}"
              f"{sesoi_s}, seeds={r['n']}, "
              f"sign-consistency={r['sign_consistency']:.2f}"
              f"{loo_s}{need_s})")
        a_mean, a_std = r.get("a_mean"), r.get("a_std")
        b_mean, b_std = r.get("b_mean"), r.get("b_std")
        if _finite(a_mean) and _finite(b_mean):
            sd = r.get("sd")
            if _finite(sd) and sd > 0.0 and _finite(r["mean"]):
                spread = ("|effect| > paired sd (gain exceeds the paired "
                          "seed-to-seed spread)"
                          if abs(r["mean"]) > sd else
                          "|effect| <= paired sd (gain is within the "
                          "paired seed-to-seed spread)")
                var_read = f"paired sd={sd:.4f}; {spread}"
            elif _finite(sd):
                var_read = ("paired diffs identical across seeds "
                            "(paired sd=0)")
            else:
                var_read = "paired sd n/a"
            std_a = f"{a_std:.4f}" if _finite(a_std) else "n/a"
            std_b = f"{b_std:.4f}" if _finite(b_std) else "n/a"
            note = r.get("arms_note")
            label = f"arms ({note})" if note else "arms (mean±std)"
            print(f"      {label}: a={a_mean:.4f}±{std_a} "
                  f"(n={r.get('a_n')}) vs b={b_mean:.4f}±{std_b} "
                  f"(n={r.get('b_n')}); {var_read}")
    if audited:
        print(f"[{self.tag}] controlled-variable audit: "
              f"{len(controlled_pass)}/{len(audited)} contrasts pass "
              f"(paired seeds + identical training recipe across sides + "
              f"a declared model factor varies)")
    if confounded:
        print(f"[{self.tag}] WARNING: {len(confounded)} contrast(s) "
              f"failed the controlled-variable audit and were refused "
              f"regardless of their p-values")
    if reversed_effects:
        names = ", ".join(f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                          for r in reversed_effects)
        print(f"[{self.tag}] WARNING: {len(reversed_effects)} "
              f"FDR-significant contrast(s) point in the opposite "
              f"direction of their pre-registered prediction ({names}): "
              f"each is read as a refutation, not as support, and never "
              f"enters the supporting set")
    if below_sesoi:
        underpowered = [r for r in below_sesoi
                        if math.isfinite(r.get("n_seeds_for_sesoi", 0.0))
                        and r["n_seeds_for_sesoi"] > r["n"]]
        print(f"[{self.tag}] WARNING: {len(below_sesoi)} FDR-significant "
              f"contrast(s) are smaller than their pre-registered SESOI: "
              f"statistically detectable but practically negligible; "
              f"{len(underpowered)} of them would need more seeds "
              f"(listed per row above) to resolve the SESOI - do not "
              f"read them as supporting the mechanism")
    if sesoi_point_only:
        names = ", ".join(
            f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
            for r in sesoi_point_only)
        print(f"[{self.tag}] WARNING: {len(sesoi_point_only)}/"
              f"{len(supporting)} supported contrast(s) clear their SESOI "
              f"on the point estimate but NOT at the interval level (the "
              f"95% CI still includes effects below the practical "
              f"floor): {names}; report them as practically provisional "
              f"(minimum-effect test), not as established practical "
              f"effects, even though the pre-registered decision rule "
              f"still reads them as SURVIVES")
    if met_established:
        print(f"[{self.tag}] minimum-effect (SESOI) companion: "
              f"{len(met_established)}/{len(supporting)} supported "
              f"contrast(s) also have their whole 95% CI above the "
              f"registered practical bound (their practical size is "
              f"established, not just their non-zero sign)")
    if replication_uncertain:
        names = ", ".join(
            f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
            for r in replication_uncertain)
        print(f"[{self.tag}] WARNING: {len(replication_uncertain)}/"
              f"{len(supporting)} supported contrast(s) have a 95% "
              f"same-size replication interval that still crosses zero "
              f"or their null-prediction bound: {names}; the average "
              f"effect is supported, but a fresh run of the same size "
              f"is NOT expected to reproduce its sign/bound at 95% - "
              f"report these as replication-uncertain, not as robust "
              f"effects")
    if effect_size_unavailable:
        names = ", ".join(
            f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
            for r in effect_size_unavailable)
        print(f"[{self.tag}] note: {len(effect_size_unavailable)}/"
              f"{len(supporting)} supported contrast(s) have no "
              f"estimable standardized-effect CI (zero/non-finite "
              f"paired SD or too few usable bootstrap resamples): "
              f"{names}; report their raw-unit effect without a "
              f"standardized magnitude")
    print(f"[{self.tag}] hypothesis-level synthesis (companion): "
          f"{len(synth_supported)} fully supported, "
          f"{len(synth_partial)} partial, {len(synth_refuted)} refuted, "
          f"{len(synth_inconclusive)} inconclusive, "
          f"{len(synth_undirected)} undirected-only, "
          f"{len(synth_no_contrast)} with no contrast; a hypothesis is "
          f"'fully supported' only when every registered directional/"
          f"null-prediction row supports it (row-level SURVIVES alone "
          f"is not a confirmed conjunction)")
    if synth_partial:
        names = ", ".join(synth_partial)
        print(f"[{self.tag}] WARNING: {len(synth_partial)} hypothesis(es) "
              f"have only PARTIAL evidence: {names}; at least one "
              f"registered prediction row is inconclusive, so the "
              f"hypothesis as stated (a conjunction) is not confirmed")
    if synth_refuted:
        names = ", ".join(synth_refuted)
        print(f"[{self.tag}] WARNING: {len(synth_refuted)} hypothesis(es) "
              f"are REFUTED by at least one registered prediction row: "
              f"{names}; supporting rows in the same hypothesis cannot "
              f"overturn the refutation")
    if synth_rep_uncertain:
        names = ", ".join(synth_rep_uncertain)
        print(f"[{self.tag}] WARNING: {len(synth_rep_uncertain)} "
              f"hypothesis(es) rest on supported effects that are "
              f"replication-uncertain at the same size: {names}; the "
              f"direction is supported by the current run, but the "
              f"replication interval says a repeat could reverse it")
    if synth_provisional:
        names = ", ".join(synth_provisional)
        print(f"[{self.tag}] note: {len(synth_provisional)} "
              f"hypothesis(es) contain supported rows whose practical "
              f"size is only point-estimate level ({names}); their "
              f"practical significance remains provisional even where "
              f"the conjunction is otherwise supported)")
    if synth_partial_registry:
        names = ", ".join(synth_partial_registry)
        print(f"[{self.tag}] WARNING: {len(synth_partial_registry)} "
              f"hypothesis(es) are supported on the executed rows but "
              f"have at least one registered decision metric with no "
              f"contrast at all ({names}); report them as partial-registry "
              f"support, not as a complete pre-registered conjunction")
    if synth_not_family_significant:
        names = ", ".join(synth_not_family_significant)
        print(f"[{self.tag}] WARNING: {len(synth_not_family_significant)} "
              f"hypothesis(es) have every executed row supported, but the "
              f"hypothesis-level intersection-union test does not survive "
              f"the hypothesis family correction ({names}); treat them as "
              f"not claim-level significant")
    if synth_claim_level:
        print(f"[{self.tag}] hypothesis-level multiplicity companion: "
              f"{len(synth_claim_level)}/{len(synthesis)} "
              f"hypothesis(es) pass the conjunction test, complete "
              f"registered-metric coverage and the hypothesis-family FDR "
              f"screen; the row-level pre-registered decision remains the "
              f"decision unit")
    if synth_dependence_sensitive:
        names = ", ".join(synth_dependence_sensitive)
        print(f"[{self.tag}] WARNING: {len(synth_dependence_sensitive)} "
              f"claim-level hypothesis(es) fail at least one "
              f"dependence-robust companion ({names}); their conjunction "
              f"support is multiplicity-model sensitive")
    if equiv_rows:
        print(f"[{self.tag}] equivalence (null-prediction) tests: "
              f"{len(equivalent)}/{len(equiv_rows)} have their 95% CI "
              f"inside the pre-registered bound")
    if equiv_borderline:
        print(f"[{self.tag}] WARNING: {len(equiv_borderline)} "
              f"equivalence contrast(s) pass the TOST FDR gate but their "
              f"CI is not fully inside the bound (borderline); they do "
              f"NOT count as confirming the null prediction")
    if noninf_rows:
        print(f"[{self.tag}] non-inferiority (one-sided null-prediction) "
              f"tests: {len(noninferior)}/{len(noninf_rows)} have their "
              f"95% CI lower bound above the negative pre-registered "
              f"bound")
    if noninf_borderline:
        print(f"[{self.tag}] WARNING: {len(noninf_borderline)} "
              f"non-inferiority contrast(s) pass the one-sided FDR gate "
              f"but their CI lower bound is not above the negative bound "
              f"(borderline); they do NOT count as confirming the null "
              f"prediction")
    competence_failed = [r for r in testable
                         if r.get("competence_passed") is False]
    if self.competence is not None and not self.competence.get("passed"):
        print(f"[{self.tag}] WARNING: HB0 competence prerequisite "
              f"FAILED: no trained arm has a seed-clustered 95% lower "
              f"bound above chance {self.competence.get('chance', float('nan')):.4g} "
              f"(best: {self.competence.get('best_arm') or 'none'}), so "
              f"the grid did not demonstrably solve the task at this "
              f"step count: all {len(competence_failed)} contrast(s) are "
              f"marked claim-ineligible and may not be read as mechanism "
              f"statements; raise the step count and "
              f"re-run before interpreting any contrast")
    stream_conflict = [r for r in testable
                       if r.get("stream_sign_conflict")]
    if stream_conflict:
        names = ", ".join(
            f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
            for r in stream_conflict)
        print(f"[{self.tag}] WARNING: {len(stream_conflict)} "
              f"contrast(s) have a seed-level effect whose sign "
              f"disagrees with the item-level paired effect on the "
              f"shared evaluation stream ({names}): marked "
              f"claim-ineligible")
    stream_uncorroborated = [
        r for r in supporting
        if r.get("stream_direction_agrees") is not None
        and r.get("stream_seed_sign_consistent") is False]
    stream_fold_fragile = [
        r for r in supporting if r.get("stream_fold_fragile")]
    stream_crossed_limited = [
        r for r in supporting if r.get("stream_crossed_stream_limited")]
    if stream_uncorroborated:
        print(f"[{self.tag}] WARNING: {len(stream_uncorroborated)} "
              f"supported accuracy contrast(s) have per-seed item-level "
              f"effects that do not all share the seed-level sign: the "
              f"pooled McNemar p reuses the same sealed items across "
              f"seeds and is descriptive only, so it cannot corroborate "
              f"the claim; report the per-seed counts next to it")
    if stream_fold_fragile:
        names = ", ".join(
            f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
            for r in stream_fold_fragile)
        print(f"[{self.tag}] WARNING: {len(stream_fold_fragile)} "
              f"supported accuracy contrast(s) have an item-level "
              f"effect that is fragile to one evaluation-stream fold "
              f"(a leave-one-fold-out sign flip or one fold carrying "
              f"more than half of the absolute effect): {names}; the "
              f"seed-level verdict stands, but the data-level "
              f"generalisation of the effect needs a wider or "
              f"independent evaluation stream")
    if stream_crossed_limited:
        names = ", ".join(
            f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
            for r in stream_crossed_limited)
        print(f"[{self.tag}] WARNING: {len(stream_crossed_limited)} "
              f"supported accuracy contrast(s) have a seed-level 95% CI "
              f"that excludes 0 but a seed x evaluation-item bootstrap "
              f"CI that includes 0 ({names}): the effect is conditional "
              f"on this realized evaluation stream, so report it as "
              f"stream-limited and replicate on a wider or independent "
              f"stream before claiming it generalises to the task")
    curve_invalid_gated = [r for r in testable
                           if r.get("curve_invalid_removes_support")]
    balance_gated = [r for r in testable
                     if r.get("accuracy_balance_failed")]
    if balance_gated:
        print(f"[{self.tag}] WARNING: {len(balance_gated)} accuracy "
              f"contrast(s) are marked claim-ineligible because the "
              f"held-out answer-balance audit failed; accuracy is a "
              f"class-prior reading on that stream, so only the "
              f"macro/PR-AUC companions can support a mechanism claim")
    low_seed = [r for r in testable
                if r.get("seed_floor_failed")
                and r.get("competence_passed") is not False
                and not r.get("curve_invalid_removes_support")
                and not r.get("coverage_gate_failed")
                and not r.get("accuracy_balance_failed")
                and not r.get("stream_sign_conflict")]
    if low_seed:
        print(f"[{self.tag}] WARNING: {len(low_seed)}/{len(testable)} "
              f"testable contrast(s) rest on fewer than "
              f"{self.min_seeds} independent seeds and are marked "
              f"claim-ineligible: the pre-registered minimum for a "
              f"seed-level claim is {self.min_seeds} seeds, so their "
              f"p-values cannot support a decision (replicated runs "
              f"with >=3 seeds needed)")
    coverage_failed = [r for r in testable
                       if r.get("coverage_gate_failed")]
    if coverage_failed:
        print(f"[{self.tag}] WARNING: {len(coverage_failed)}/"
              f"{len(testable)} testable contrast(s) are marked "
              f"claim-ineligible because the design coverage gate "
              f"failed (missing or seed-unbalanced cells): the effect "
              f"was not measured on the full planned grid")
    if curve_invalid_gated:
        names = ", ".join(
            f"{r['hypothesis']}/{r['rule']}" for r in curve_invalid_gated)
        print(f"[{self.tag}] WARNING: {len(curve_invalid_gated)} "
              f"supported contrast(s) lose support when curve-invalid "
              f"seeds are dropped ({names}): they are marked "
              f"claim-ineligible, because a difference produced by runs "
              f"that never learned is not measured evidence for the "
              f"mechanism")
    coverage_loss = [r for r in self.rows if r.get("seeds_dropped")]
    unpaired_rows = [r for r in coverage_loss if r.get("seeds_unpaired")]
    undefined_rows = [r for r in coverage_loss if r.get("seeds_undefined")]
    if coverage_loss:
        weakened = [r for r in coverage_loss
                    if (r.get("n") or 0) < self.min_seeds]
        parts = []
        if unpaired_rows:
            parts.append(f"{len(unpaired_rows)} with a value present on "
                         f"only one side (pairing failure)")
        if undefined_rows:
            parts.append(f"{len(undefined_rows)} where the metric is "
                         f"undefined on both sides (e.g. no "
                         f"false-concentration cases)")
        print(f"[{self.tag}] WARNING: {len(coverage_loss)} contrast(s) "
              f"could not use the full planned seed set: "
              f"{'; '.join(parts) or 'seed coverage loss'}; "
              f"{len(weakened)} fell below the {self.min_seeds}-seed "
              f"claim floor; seeds_used/seeds_dropped are recorded per "
              f"row for the audit")
    if loo_flips:
        names = ", ".join(f"{r['hypothesis']}/{r['rule']}"
                          for r in loo_flips)
        print(f"[{self.tag}] WARNING: {len(loo_flips)} positive "
              f"decision(s) flip when any single seed is dropped "
              f"(jackknife): {names}; the effect rests on one "
              f"influential seed, so replicate at more seeds before "
              f"reading it as a mechanism")
    if bca_differs:
        names = ", ".join(f"{r['hypothesis']}/{r['rule']}"
                          for r in bca_differs)
        print(f"[{self.tag}] WARNING: {len(bca_differs)} testable "
              f"contrast(s) where the bias-corrected (BCa) interval "
              f"reaches the opposite decision from the pre-registered "
              f"percentile interval: {names}; at these seed counts the "
              f"readout is sensitive to the interval construction, so "
              f"treat either verdict as provisional until more seeds "
              f"settle the disagreement")
    if low_resolution:
        print(f"[{self.tag}] note: {len(low_resolution)}/{len(testable)} "
              f"testable contrast(s) rest on <=5 seeds; an exact "
              f"two-sided sign test cannot reach p<0.05 below n=6, so "
              f"direction reads rest on the bootstrap's moment "
              f"assumptions - the exact sign-flip p/floor, jackknife and "
              f"BCa annotations above show the exposure")
    if sign_flip_supporting or sign_flip_conflicts or sign_flip_below_floor:
        below = (f"; {len(sign_flip_below_floor)} bootstrap p-value(s) "
                 f"fall below the exact test's attainable floor, so the "
                 f"bootstrap number overstates the design's resolution"
                 if sign_flip_below_floor else "")
        print(f"[{self.tag}] exact sign-flip family (companion, not the "
              f"pre-registered bootstrap decision): "
              f"{len(sign_flip_supporting)}/{len(testable)} testable "
              f"contrast(s) support their prediction under the exact "
              f"test; {len(sign_flip_conflicts)} disagree with the "
              f"bootstrap verdict{below}")
    if sign_flip_conflicts:
        shown = sign_flip_conflicts[:5]
        names = ", ".join(
            f"{r['hypothesis']}/{r['rule']} ({r['metric']}: "
            f"bootstrap={'yes' if r.get('supports_hypothesis') else 'no'}"
            f", exact={'yes' if r.get('sign_flip_supports') else 'no'})"
            for r in shown)
        if len(sign_flip_conflicts) > len(shown):
            names += f", ... (+{len(sign_flip_conflicts) - len(shown)})"
        print(f"[{self.tag}] WARNING: {len(sign_flip_conflicts)} "
              f"contrast(s) change verdict between the pre-registered "
              f"bootstrap rule and the exact sign-flip companion "
              f"({names}): treat the bootstrap-only verdict as "
              f"provisional and replicate at more seeds")
    if sign_flip_unfalsifiable:
        shown = sign_flip_unfalsifiable[:5]
        names = ", ".join(
            f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
            for r in shown)
        if len(sign_flip_unfalsifiable) > len(shown):
            names += (f", ... (+{len(sign_flip_unfalsifiable) - len(shown)}"
                      f" more)")
        print(f"[{self.tag}] WARNING: {len(sign_flip_unfalsifiable)} "
              f"contrast(s) cannot be rejected by their exact sign-flip "
              f"test even in principle after BH across this family at "
              f"q={q:g} (min attainable p > q; "
              f"{names}): their direction is not falsifiable with the "
              f"current seed count, so add seeds rather than reading the "
              f"bootstrap p as confirmation")
    if testable:
        simes_s = (f"global Simes p={float(family_simes['p']):.4f}"
                   if family_simes.get("valid") else
                   "global Simes p=n/a")
        print(f"[{self.tag}] dependence-robust multiplicity companions "
              f"(pre-registered BH decision unchanged): {simes_s}; "
              f"BY-arbitrary-dependence {len(by_supporting)}/"
              f"{len(testable)} supports, Holm-FWER "
              f"{len(holm_supporting)}/{len(testable)} supports; "
              f"{len(by_conflicts)} BY and {len(holm_conflicts)} Holm "
              f"verdict(s) disagree with BH")
    if dependence_weaker:
        shown = dependence_weaker[:5]
        names = ", ".join(
            f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
            for r in shown)
        if len(dependence_weaker) > len(shown):
            names += (f", ... (+{len(dependence_weaker) - len(shown)}"
                      f" more)")
        print(f"[{self.tag}] WARNING: {len(dependence_weaker)}/"
              f"{len(supporting)} BH-supported contrast(s) do NOT "
              f"survive both dependence-robust companions ({names}): "
              f"the family reuses seeds and a shared evaluation stream, "
              f"so the BH-only support is conditional on an "
              f"independence/PRDS premise; report these as "
              f"multiplicity-model sensitive, not as robust claims")
    if selected:
        fcr_level = (1.0 - q * n_selected / n_tests
                     if n_tests else float("nan"))
        print(f"[{self.tag}] selection-adjusted (FCR) companion: "
              f"{n_selected}/{n_tests} BH-selected contrast(s); the "
              f"PRDS interval level is {fcr_level:.4f} "
              f"(1 - q*R/m); {len(fcr_sensitive)} supported claim(s) "
              f"lose their interval decision after selection adjustment "
              f"and {len(fcr_by_sensitive)} lose it under the "
              f"arbitrary-dependence (BY) variant")
    if fcr_unavailable or fcr_by_unavailable:
        print(f"[{self.tag}] note: {len(fcr_unavailable)} selected "
              f"contrast(s) have no PRDS FCR interval and "
              f"{len(fcr_by_unavailable)} have no arbitrary-dependence "
              f"FCR interval (too few paired seeds or a degenerate "
              f"bootstrap); their ordinary CI is not selection-adjusted")
    if fcr_sensitive:
        names = ", ".join(
            f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
            for r in fcr_sensitive)
        print(f"[{self.tag}] WARNING: {len(fcr_sensitive)}/"
              f"{len(supporting)} supported contrast(s) do NOT keep "
              f"their interval decision after false-coverage-rate "
              f"adjustment for the BH-selected set ({names}): their "
              f"ordinary 95% CI is conditional on selection; report "
              f"them as selection-sensitive, not as robust effects")
    if fcr_by_sensitive:
        names = ", ".join(
            f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
            for r in fcr_by_sensitive)
        print(f"[{self.tag}] WARNING: {len(fcr_by_sensitive)}/"
              f"{len(supporting)} supported contrast(s) lose the "
              f"interval decision under the arbitrary-dependence FCR "
              f"variant ({names}); the selected-set interval needs the "
              f"PRDS premise to clear zero")
    registered = list(self.hypotheses)
    covered = sorted({r["hypothesis"] for r in self.rows})
    uncovered = [h for h in registered if h not in covered]
    unknown = [h for h in covered if h not in HYPOTHESES]
    misplaced = [h for h in covered
                 if h in HYPOTHESES
                 and HYPOTHESES[h].get("experiment") != self.tag]
    if registered:
        hit = len([h for h in registered if h in covered])
        print(f"[{self.tag}] pre-registration coverage: {hit}/"
              f"{len(registered)} registered hypotheses have at least "
              f"one contrast in this family")
    if uncovered:
        print(f"[{self.tag}] WARNING: pre-registered hypothesis(es) "
              f"{', '.join(uncovered)} have NO contrast in this family; "
              f"their pre-registered decision cannot be evaluated")
    if unknown:
        print(f"[{self.tag}] WARNING: contrast row(s) cite unknown "
              f"hypothesis id(s) {', '.join(unknown)} (not in the "
              f"hashed registry)")
    if misplaced:
        for h in misplaced:
            print(f"[{self.tag}] WARNING: hypothesis {h} is registered "
                  f"to experiment {HYPOTHESES[h].get('experiment')!r}, "
                  f"not {self.tag!r}; check the mapping")
    payload = {"tag": self.tag, "q": q, "n_contrasts": len(self.rows),
               "n_testable": len(testable),
               "n_surviving": len(surviving),
               "n_supporting": len(supporting),
               "n_below_sesoi": len(below_sesoi),
               "n_minimum_effect_established": len(met_established),
               "n_sesoi_point_estimate_only": len(sesoi_point_only),
               "sesoi_point_estimate_only_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in sesoi_point_only],
               "n_replication_uncertain": len(replication_uncertain),
               "replication_uncertain_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in replication_uncertain],
               "n_effect_size_ci_unavailable": len(
                   effect_size_unavailable),
               "effect_size_ci_unavailable_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in effect_size_unavailable],
               "hypothesis_synthesis": synthesis,
               "n_hypotheses_supported": len(synth_supported),
               "n_hypotheses_partial": len(synth_partial),
               "n_hypotheses_refuted": len(synth_refuted),
               "n_hypotheses_inconclusive": len(synth_inconclusive),
               "n_hypotheses_undirected": len(synth_undirected),
               "n_hypotheses_no_contrast": len(synth_no_contrast),
               "n_hypotheses_replication_uncertain":
                   len(synth_rep_uncertain),
               "n_hypotheses_partial_registry":
                   len(synth_partial_registry),
               "partial_registry_hypotheses": synth_partial_registry,
               "n_hypotheses_not_family_significant":
                   len(synth_not_family_significant),
               "not_family_significant_hypotheses":
                   synth_not_family_significant,
               "n_hypotheses_claim_level_supported":
                   len(synth_claim_level),
               "claim_level_supported_hypotheses": synth_claim_level,
               "n_hypotheses_dependence_sensitive":
                   len(synth_dependence_sensitive),
               "dependence_sensitive_hypotheses":
                   synth_dependence_sensitive,
               "partial_hypotheses": synth_partial,
               "refuted_hypotheses": synth_refuted,
               "replication_uncertain_hypotheses": synth_rep_uncertain,
               "n_direction_reversed": len(reversed_effects),
               "n_equivalence": len(equiv_rows),
               "n_equivalent": len(equivalent),
               "n_equiv_borderline": len(equiv_borderline),
               "n_noninferiority": len(noninf_rows),
               "n_noninferior": len(noninferior),
               "n_noninf_borderline": len(noninf_borderline),
               "n_confounded": len(confounded),
               "n_audited": len(audited),
               "n_control_passed": len(controlled_pass),
               "n_low_seed": len(low_seed),
               "n_curve_invalid_gated": len(curve_invalid_gated),
               "n_claim_ineligible_competence": len(competence_failed),
               "n_claim_ineligible": len(
                   [r for r in testable
                    if not r.get("claim_eligible", True)]),
               "competence": self.competence,
               "n_seed_coverage_loss": len(coverage_loss),
               "n_seed_unpaired": len(unpaired_rows),
               "n_seed_undefined": len(undefined_rows),
               "min_seeds_for_claim": self.min_seeds,
               "n_loo_flip_decision": len(loo_flips),
               "n_bca_disagree": len(bca_differs),
               "n_low_resolution": len(low_resolution),
               "n_stream_fold_fragile": len(stream_fold_fragile),
               "n_stream_crossed_stream_limited":
                   len(stream_crossed_limited),
               "n_sign_flip_supporting": len(sign_flip_supporting),
               "n_sign_flip_disagree": len(sign_flip_conflicts),
               "n_sign_flip_unfalsifiable":
                   len(sign_flip_unfalsifiable),
               "n_bootstrap_below_exact_floor":
                   len(sign_flip_below_floor),
               "simes_family_p": family_simes.get("p"),
               "simes_family_reject": bool(
                   family_simes.get("reject")),
               "n_by_supporting": len(by_supporting),
               "n_by_disagree": len(by_conflicts),
               "n_holm_supporting": len(holm_supporting),
               "n_holm_disagree": len(holm_conflicts),
               "n_dependence_weaker": len(dependence_weaker),
               "n_fcr_selected": n_selected,
               "n_fcr_tests": n_tests,
               "n_fcr_decision_sensitive": len(fcr_sensitive),
               "n_fcr_by_decision_sensitive": len(fcr_by_sensitive),
               "n_fcr_unavailable": len(fcr_unavailable),
               "n_fcr_by_unavailable": len(fcr_by_unavailable),
               "fcr_decision_sensitive_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in fcr_sensitive],
               "fcr_by_decision_sensitive_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in fcr_by_sensitive],
               "direction_reversed_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in reversed_effects],
               "curve_invalid_gated_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in curve_invalid_gated],
               "stream_fold_fragile_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in stream_fold_fragile],
               "stream_crossed_stream_limited_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in stream_crossed_limited],
               "seed_coverage_loss_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in coverage_loss],
               "loo_flip_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in loo_flips],
               "bca_disagree_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in bca_differs],
               "sign_flip_disagree_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in sign_flip_conflicts],
               "sign_flip_unfalsifiable_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in sign_flip_unfalsifiable],
               "bootstrap_below_exact_floor_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in sign_flip_below_floor],
               "by_disagree_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in by_conflicts],
               "holm_disagree_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in holm_conflicts],
               "dependence_weaker_contrasts": [
                   f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                   for r in dependence_weaker],
               "registered_hypotheses": registered,
               "covered_hypotheses": covered,
               "uncovered_hypotheses": uncovered,
               "unknown_hypotheses": unknown,
               "misplaced_hypotheses": misplaced,
               "rows": self.rows}
    _write_json(out_dir / f"{self.tag}_contrasts.json", payload)
    return payload

ContrastLog.finalize = _contrastlog_finalize
