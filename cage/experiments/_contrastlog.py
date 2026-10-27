# cage/experiments/_contrastlog.py - ContrastLog: family accumulation and per-contrast printing.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
class ContrastLog:
    """One experiment's pre-registered contrast family, FDR-controlled.

    Every contrast is printed as soon as it is computed (the readout is
    unchanged), and :meth:`finalize` applies Benjamini-Hochberg control
    across the whole family.  The family is the unit of inference: with
    ~16 contrasts per experiment, an uncorrected per-contrast 95% readout
    would be expected to produce about one spurious "supported" claim when
    every null is true.  The finalize report prints each contrast's
    bootstrap p-value, adjusted p, effect size, MDE80 and seed
    sign-consistency, and writes the table to ``<tag>_contrasts.json`` so
    the corrected readout is auditable next to the run records.

    Every contrast built by ``_paired_diff`` also carries its
    controlled-variable audit: a contrast whose two sides differ in
    optimizer/recipe settings, whose evaluation streams differ (size *or*
    seed), whose seeds are not paired, or that does not vary a declared
    model factor is *not* read as supported, whatever its
    p-value - the difference would not be attributable to the module under
    test.  The audit verdict is printed inline and stored per row, and so
    is each arm's mean +- std and seed count, so the decision table itself
    answers "did the effect exceed the seed-to-seed spread?" for every
    metric - including the E3 mechanism readouts that have no per-arm
    table of their own.

    The decision layer also separates statistical from *practical*
    significance: the hypothesis's pre-registered SESOI (looked up in
    ``HYPOTHESES``, so it is part of the hashed pre-registration) is
    compared with the effect, ``seeds_for_power`` plans how many seeds the
    SESOI would need at the observed variance, and an FDR-significant
    effect below its SESOI is reported as practically negligible instead
    of "supported".

    The pre-registered *direction* of the prediction (the ``direction``
    sign registered per metric in ``HYPOTHESES``, e.g. H1c's
    "null_mass_error_margin > 0") is enforced at the decision point as
    well: an FDR-significant effect of the opposite sign refutes the
    hypothesis, is labelled as such, and can never enter the supporting
    set.  The SESOI gate bounds the *magnitude* of the required effect;
    it is not a licence to count a reversed effect as support.

    Contrasts whose pre-registered prediction is *null/equivalence*
    ("this quantity stays inside +-bound") must be added through
    :meth:`print_equivalence`, which runs the bootstrap TOST from
    ``cage.metrics`` with the bound registered under ``sesoi_test`` and
    reads support as CI-within-bound - never as "CI includes 0".  Its
    TOST p-value joins the same BH family as the directional p-values.

    Because rows in one family share training seeds and a sealed
    evaluation stream, the independence/PRDS premise behind BH is not
    established.  :meth:`finalize` therefore also reports Benjamini-
    Yekutieli (FDR under arbitrary dependence) and Holm (family-wise
    error) adjusted decisions, plus a Simes omnibus p-value.  These are
    sensitivity companions: the pre-registered BH verdict and the
    ``supports_hypothesis`` flag are never changed, but a BH-supported
    claim that fails either dependence-robust model is named in the
    console warning and in ``<tag>_contrasts.json`` as multiplicity-model
    sensitive.

    ``hypotheses`` lists the pre-registered ids this family is supposed to
    evaluate; :meth:`finalize` audits coverage (registered but absent, or
    rows citing unknown ids) and records it in the payload, so a
    pre-registered decision that no contrast executes is flagged instead
    of silently missing.

    Small-sample robustness: at the seed counts used here the percentile
    bootstrap is the first-order approximation in the chain, so every row
    also stores the jackknife (leave-one-out) mean range, its
    sign/bound-flip flag and the BCa interval verdict (see
    ``cage.metrics``).  :meth:`finalize` counts the contrasts whose
    *decision* flips when any single seed is dropped and those where BCa
    disagrees with the percentile readout, prints them as warnings, and
    writes the counts into the payload - the pre-registered decision rule
    itself is untouched, but a claim carried by one influential seed is
    no longer invisible.  Every row additionally stores an exact paired
    sign-flip p-value for its registered alternative and the minimum p the
    seed count can attain; :meth:`finalize` runs a parallel BH family over
    those exact p-values and flags disagreements, bootstrap p-values below
    the exact floor, and contrasts that cannot be rejected by an exact
    test even in principle.  This is a validity audit, not a second
    decision rule.

    Seed floor: a contrast may only support a claim when at least
    ``min_seeds`` (default 3, pre-registered in ``Settings``) independent
    training seeds actually entered the paired difference.  A 0.2% effect
    next to a 0.5% seed-level std - or a two-seed "win" - is not evidence,
    so such rows are marked ``claim_eligible=False`` at the decision
    point and read as INELIGIBLE regardless of their p-value.  Rows also
    record ``seeds_used``/``seeds_dropped`` (split into ``seeds_unpaired``
    and ``seeds_undefined``) so a sample that silently shrank - a value
    present on one side only, or a metric undefined for a seed - is
    visible instead of feeding an unpaired comparison.

    Two further companions sit on top of the row decisions and change none
    of them.  Each row stores a same-size **replication interval** for a
    future study's mean and its probability of keeping the registered sign
    (``cage.metrics.replication_interval``); a supported row whose repeat
    interval still crosses zero is marked ``replication_uncertain``.
    ``finalize`` also groups the rows by hypothesis and reports whether the
    pre-registered *conjunction* is ``supported``, ``partial``,
    ``refuted``, ``inconclusive``, ``undirected`` or ``no_contrast`` - one
    surviving row is not a confirmed multi-row hypothesis.  Every row
    additionally carries a paired-bootstrap interval for Cohen's dz and
    Hedges' g (``cage.metrics.paired_effect_size_ci``): the raw CI says how
    large the effect is in metric units, while the standardized interval
    shows how well the seed-to-seed scale supporting that magnitude read is
    itself estimated.  All three are reading aids; the pre-registered row
    rule and FDR family are unchanged.
    """

    def __init__(self, tag: str, hypotheses=(), min_seeds: int = 3,
                 competence: dict | None = None, coverage=(),
                 balance_ok: bool = True):
        self.tag = tag
        # A string/bytes/mapping is iterable but does not name hypothesis
        # ids; a scalar is not iterable at all.  Reject the container
        # rather than turning bytes into integer hypothesis ids.
        if isinstance(hypotheses, (str, bytes)) \
                or isinstance(hypotheses, Mapping):
            raise ValueError(
                "hypotheses must be an iterable of hypothesis ids, got "
                f"{type(hypotheses).__name__}")
        try:
            self.hypotheses = tuple(hypotheses)
        except TypeError as exc:
            raise ValueError(
                "hypotheses must be an iterable of hypothesis ids, got "
                f"{type(hypotheses).__name__}") from exc
        if competence is not None and not isinstance(competence, Mapping):
            raise ValueError(
                "competence must be a mapping or None, got "
                f"{type(competence).__name__}")
        # The seed floor is a pre-registered decision rule.  Coercing a
        # bool/fractional/zero value with ``int`` would silently lower it
        # (``True`` -> 1, ``0`` -> 0) and make underpowered rows eligible.
        floor = _positive_integer(min_seeds)
        if floor is None:
            raise ValueError(
                f"min_seeds must be a positive integer, got {min_seeds!r}")
        self.min_seeds = int(floor)
        self.rows: list[dict] = []
        # Held-out answer-balance premise for accuracy-based claims.  The
        # gross-imbalance audit is not merely descriptive: if the stream is
        # not uniform, an accuracy contrast is partly a class-prior contrast
        # and cannot support a mechanism claim.  ``True``/``None`` (not
        # audited) leave the decision untouched; only an explicit failure
        # gates the accuracy rows.  Macro-F1/balanced-accuracy and PR-AUC
        # rows are unaffected.
        # Fail closed on a malformed mark: a stored ``"False"`` (or any
        # other non-bool) is not evidence that the balance audit passed.
        # ``None`` means the older verdict predates the audit and leaves the
        # decision untouched.
        if balance_ok is None or isinstance(balance_ok, bool):
            self.balance_ok = True if balance_ok is None else balance_ok
        else:
            self.balance_ok = False
        # HB0: the family's competence prerequisite (at least one arm above
        # chance).  When it fails, no contrast in the family may support a
        # claim - an underfit grid's differences describe the step count, not
        # the mechanism.  ``None`` means "not assessed" and does not gate.
        self.competence = competence
        # Factorial/design coverage payloads (``_report_ablation_coverage``
        # / ``_report_design_coverage``).  A missing cell or a cell that
        # trained only part of the planned seed set means the corresponding
        # effect was never measured; a contrast from such a design cannot
        # support a claim even when its own p-value is small.  A single
        # payload and an iterable of payloads are both accepted.
        if coverage is None:
            coverage = ()
        elif isinstance(coverage, Mapping):
            coverage = (coverage,)
        elif isinstance(coverage, (str, bytes)):
            raise ValueError(
                "coverage must be a mapping or an iterable of mappings, "
                f"got {type(coverage).__name__}")
        try:
            self.coverage = tuple(coverage)
        except TypeError as exc:
            raise ValueError(
                "coverage must be a mapping or an iterable of mappings, "
                f"got {type(coverage).__name__}") from exc
        malformed = [type(payload).__name__ for payload in self.coverage
                     if not isinstance(payload, Mapping)]
        if malformed:
            raise ValueError(
                "coverage entries must be mappings, got "
                f"{sorted(set(malformed))}")

    def print(self, rule, hyp, metric, ci, warn="",
              direction_key=None) -> None:
        """Register one computed contrast in the family.

        ``metric`` is the display/registry key used for the SESOI lookup.
        ``direction_key`` optionally names a *different* registry key for
        the pre-registered sign, which is needed when one metric carries
        both a directional main-effect row and a two-sided companion row
        (e.g. H2e's concentration main effect vs. its interaction): the
        companion must not inherit the main effect's sign.  An explicit
        ``direction_key`` that is not registered is rejected instead of
        silently producing a two-sided row.
        """
        test = ci.get("test")
        equiv = test == "equivalence"
        noninf = test == "noninferiority"
        null_test = equiv or noninf
        audit = ci.get("controlled")
        sesoi = (HYPOTHESES.get(hyp, {}).get("sesoi") or {}).get(metric)
        if null_test:
            # A null-prediction row's decision bound is the CI's registered
            # bound, which can live under a separate registry key when the
            # displayed metric is shared with directional rows (H2c's
            # reverse check).  ``sesoi`` above is keyed by the display
            # metric, so using it here would print and apply the
            # directional SESOI as if it were the bound.
            bound = ci.get("bound")
            if bound is not None:
                sesoi = bound
        # Pre-registered sign of the predicted effect for this metric
        # ("+" / "-", looked up in HYPOTHESES so it is part of the hashed
        # pre-registration).  ``None`` means the prediction is not
        # directional (e.g. H3b's "the readout shifts"), in which case a
        # non-zero effect of either sign is admissible.
        direction_map = (HYPOTHESES.get(hyp, {}).get("direction") or {})
        if direction_key is not None and direction_key not in direction_map:
            raise ValueError(
                f"{hyp} has no pre-registered direction under "
                f"direction_key={direction_key!r}; registered keys: "
                f"{sorted(direction_map)}")
        direction = direction_map.get(
            metric if direction_key is None else direction_key)
        # Replication companion: what a same-size repeat would plausibly
        # show, not just the precision of the current mean.  Computed here
        # because the registered direction (and the null-prediction bound)
        # live in the registry lookup above; finalize() only turns the
        # interval into the "uncertain" flag once the row's decision is
        # known, so the pre-registered rule is never changed by it.
        rep_mode = ("equivalence" if equiv else "noninferiority" if noninf
                    else "difference")
        rep = replication_interval(
            ci.get("diffs"),
            direction=None if null_test else direction,
            mode=rep_mode,
            bound=ci.get("bound") if null_test else None)
        arms = ci.get("arms") or {}
        arm_a, arm_b = arms.get("a") or {}, arms.get("b") or {}
        # Full-run levels are stored alongside the paired levels so a reader
        # can see exactly how much the sample restriction moved an arm mean
        # (the decision table itself always reads ``arms``).
        arms_all = ci.get("arms_all") or {}
        all_a, all_b = arms_all.get("a") or {}, arms_all.get("b") or {}
        # Exact paired sign-flip companion.  The bootstrap p-value remains
        # the pre-registered decision; the exact p is read in the direction
        # the hypothesis registered (or as a TOST for null predictions), so
        # a bootstrap "win" that is below the design's exact resolution is
        # visible instead of being silently trusted.
        sf_greater = ci.get("sign_flip_p_greater")
        sf_less = ci.get("sign_flip_p_less")
        sf_two = ci.get("sign_flip_p_two_sided")
        if null_test:
            sf_p = ci.get("sign_flip_p")
            sf_min = ci.get("sign_flip_min_p")
            sf_alt = ("equivalence" if equiv else "noninferiority")
        elif direction == "+":
            sf_p, sf_min, sf_alt = (sf_greater,
                                    ci.get("sign_flip_min_p_greater"),
                                    "greater")
        elif direction == "-":
            sf_p, sf_min, sf_alt = (sf_less,
                                    ci.get("sign_flip_min_p_less"), "less")
        else:
            sf_p, sf_min, sf_alt = (sf_two,
                                    ci.get("sign_flip_min_p_two_sided"),
                                    "two-sided")
        self.rows.append({
            "rule": rule, "hypothesis": hyp, "metric": metric,
            "mean": ci.get("mean"), "lo": ci.get("lo"), "hi": ci.get("hi"),
            "n": ci.get("n"), "p_value": ci.get("p_value"),
            "mde80": ci.get("mde80"), "sd": ci.get("sd"),
            "dz": _std_effect(ci),
            # Standardized-effect uncertainty from the paired bootstrap.
            # ``dz`` remains the point readout; these bounds say how well
            # the seed-to-seed scale itself is estimated, which a raw-unit
            # CI cannot show.  They are companions, not decision rules.
            "cohens_d": ci.get("cohens_d"),
            "cohens_d_lo": ci.get("cohens_d_lo"),
            "cohens_d_hi": ci.get("cohens_d_hi"),
            "hedges_g": ci.get("hedges_g"),
            "hedges_g_lo": ci.get("hedges_g_lo"),
            "hedges_g_hi": ci.get("hedges_g_hi"),
            "effect_size_valid": bool(ci.get("effect_size_valid")),
            "effect_size_n_boot_valid": ci.get("effect_size_n_boot_valid"),
            "effect_size_reason": ci.get("effect_size_reason"),
            "a_mean": arm_a.get("mean"), "a_std": arm_a.get("std"),
            "a_n": arm_a.get("n"),
            "b_mean": arm_b.get("mean"), "b_std": arm_b.get("std"),
            "b_n": arm_b.get("n"),
            "a_all_mean": all_a.get("mean"), "a_all_n": all_a.get("n"),
            "b_all_mean": all_b.get("mean"), "b_all_n": all_b.get("n"),
            "arms_note": ci.get("arms_note"),
            "sesoi": sesoi,
            # An equivalence prediction has no directional alternative:
            # whichever sign the small drift takes, the claim is "inside
            # the bound".  Keeping the directional registry value here
            # would let a later reader misread the row as a one-sided
            # superiority test.
            "direction": None if null_test else direction,
            "test": test if test in ("equivalence", "noninferiority")
                    else "difference",
            "equivalence_bound": ci.get("bound") if null_test else None,
            "noninferiority_bound": (ci.get("bound") if noninf else None),
            "equivalent": ci.get("equivalent") if equiv else None,
            "noninferior": ci.get("noninferior") if noninf else None,
            "sign_consistency": ci.get("sign_consistency"),
            # Exact paired sign-flip companion (see cage.metrics).  The
            # stored p is the one for this row's registered alternative;
            # the three alternatives stay available for audit.
            "sign_flip_p": sf_p,
            "sign_flip_alternative": sf_alt,
            "sign_flip_p_greater": sf_greater,
            "sign_flip_p_less": sf_less,
            "sign_flip_p_two_sided": sf_two,
            "sign_flip_method": ci.get("sign_flip_method"),
            "sign_flip_min_p": sf_min,
            "loo_mean_min": ci.get("loo_mean_min"),
            "loo_mean_max": ci.get("loo_mean_max"),
            "loo_sign_flip": ci.get("loo_sign_flip"),
            "loo_within_bound": ci.get("loo_within_bound"),
            "loo_above_bound": ci.get("loo_above_bound"),
            "lo_bca": ci.get("lo_bca"),
            "hi_bca": ci.get("hi_bca"),
            "equivalent_bca": ci.get("equivalent_bca") if equiv else None,
            "noninferior_bca": (ci.get("noninferior_bca") if noninf
                                else None),
            "bca_agrees": ci.get("bca_agrees"),
            "baseline_mean": ci.get("baseline_mean"),
            "controlled": None if audit is None else audit["controlled"],
            # Expose the single-factor verdict next to the boolean so a
            # persisted contrast row answers "was this contrast allowed to
            # vary exactly one factor?" without re-running the audit.  None
            # for benchmarks and for aggregate factorial/interaction rows,
            # which are not single-factor pairwise contrasts by design.
            "single_factor": (None if audit is None
                              else audit.get("single_factor")),
            "n_varying_factors": (None if audit is None
                                  else audit.get("n_varying_factors")),
            "varying_factors": ({} if audit is None
                                else audit.get("varying_factors")),
            "control_mismatches": ([] if audit is None
                                   else audit.get("mismatches", [])),
            # The parameter counts the two sides actually trained (with the
            # matched flag): a gate contrast may legitimately differ, but
            # the reader must see the capacity delta rather than infer that
            # "only the factor varies" also means "same size".
            "capacity": (None if audit is None
                         else audit.get("capacity")),
            "seeds_used": ci.get("seeds_used"),
            "seeds_dropped": ci.get("seeds_dropped"),
            "seeds_unpaired": ci.get("seeds_unpaired"),
            "seeds_undefined": ci.get("seeds_undefined"),
            # Per-seed paired differences and the seeds contaminated by a
            # curve-invalid run: finalize() drops the latter and checks
            # whether the supported effect survives (sensitivity, not a
            # replacement for the pre-registered decision rule).
            "diffs": list(ci.get("diffs") or []),
            # Same-size replication readout (``cage.metrics``): the t-based
            # interval for a future study mean and the probability that the
            # repeat keeps the registered sign.  ``replication_uncertain``
            # is filled by finalize(), after the row's decision is known.
            "replication": rep,
            "replication_lo": rep.get("lo"),
            "replication_hi": rep.get("hi"),
            "replication_lo_single": rep.get("lo_single"),
            "replication_hi_single": rep.get("hi_single"),
            "replication_prob": rep.get("prob_replicates"),
            "replication_replicates": rep.get("replicates"),
            "replication_uncertain": False,
            "replication_reason": rep.get("reason"),
            "curve_invalid_seeds": list(ci.get("curve_invalid_seeds") or []),
            # Item-level companion on the shared finite evaluation stream
            # (McNemar).  ``stream_sign_conflict`` is a hard gate below:
            # a seed-level sign that disagrees with the sign over the
            # actual items is not measuring the same effect.
            "stream_test": ci.get("stream_test"),
            "stream_test_p": (ci.get("stream_test") or {}).get("p_exact"),
            "stream_pooled_descriptive": (
                ci.get("stream_test") or {}).get("pooled_descriptive"),
            "stream_per_seed": (ci.get("stream_test") or {}).get("per_seed"),
            "stream_seed_sign_test_p": (
                ci.get("stream_test") or {}).get("seed_sign_test_p"),
            "stream_seed_sign_consistent": (
                ci.get("stream_test") or {}).get(
                    "seed_sign_consistent_with_mean"),
            "stream_seed_sign_positive": (
                ci.get("stream_test") or {}).get("seed_sign_positive"),
            "stream_seed_sign_negative": (
                ci.get("stream_test") or {}).get("seed_sign_negative"),
            "stream_direction_agrees":
                (ci.get("stream_test") or {}).get("direction_agrees"),
            "stream_digest": (ci.get("stream_test") or {}).get(
                "stream_digest"),
            "stream_digest_verified": (ci.get("stream_test") or {}).get(
                "digest_verified"),
            # Data-level fold jackknife over the shared evaluation items.
            # This is a sensitivity companion, not a decision rule: the
            # pre-registered seed-level verdict is left unchanged.
            "stream_fold_audit": (ci.get("stream_test") or {}).get(
                "fold_audit"),
            "stream_fold_fragile": bool(
                ((ci.get("stream_test") or {}).get("fold_audit") or {}).get(
                    "fragile")),
            "stream_fold_loo_sign_flip": bool(
                ((ci.get("stream_test") or {}).get("fold_audit") or {}).get(
                    "loo_sign_flip")),
            "stream_fold_dominant_share": (
                (ci.get("stream_test") or {}).get("fold_audit") or {}).get(
                    "dominant_fold_share"),
            # Crossed (training seed x evaluation item) bootstrap: does the
            # seed-level verdict survive resampling the finite evaluation
            # stream as well?  A companion, not a second decision rule.
            "stream_crossed_bootstrap": (
                (ci.get("stream_test") or {}).get("crossed_bootstrap")),
            "stream_crossed_lo": (
                (ci.get("stream_test") or {}).get(
                    "crossed_bootstrap") or {}).get("lo"),
            "stream_crossed_hi": (
                (ci.get("stream_test") or {}).get(
                    "crossed_bootstrap") or {}).get("hi"),
            "stream_crossed_stream_limited": bool(
                ((ci.get("stream_test") or {}).get(
                    "crossed_bootstrap") or {}).get("stream_limited")),
            "stream_crossed_decision_agrees": (
                (ci.get("stream_test") or {}).get(
                    "crossed_bootstrap") or {}).get("decision_agrees"),
            "stream_sign_conflict": False,
        })
        line = f"  [{hyp}] {rule} ({metric}): {_fmt_ci(ci)}"
        if audit is not None:
            factors = ",".join(sorted(audit.get("varying_factors", {}))) or "n/a"
            line += (f"  [control: {'ok' if audit['controlled'] else 'FAILED'}"
                     f", varies={factors}]")
            cap = audit.get("capacity")
            if cap and not cap["matched"]:
                line += (f"  [capacity: {cap['a']} vs {cap['b']} params, "
                         f"delta={cap['delta_b_minus_a']:+d}]")
        if sesoi is not None:
            if equiv:
                line += f"  [equivalence bound={sesoi:g}]"
            elif noninf:
                line += f"  [non-inferiority bound={sesoi:g}]"
            else:
                line += f"  [SESOI={sesoi:g}]"
        if not null_test and direction is not None:
            line += (f"  [prediction: effect "
                     f"{'>' if direction == '+' else '<'} 0]")
        dropped = ci.get("seeds_dropped") or []
        if dropped:
            why = []
            if ci.get("seeds_unpaired"):
                why.append(f"{len(ci['seeds_unpaired'])} unpaired")
            if ci.get("seeds_undefined"):
                why.append(f"{len(ci['seeds_undefined'])} metric-undefined")
            line += (f"  [seed coverage: {len(dropped)} of "
                     f"{len(dropped) + len(ci.get('seeds_used') or [])} "
                     f"planned seed(s) dropped ({', '.join(why) or 'n/a'}); "
                     f"effective n={ci.get('n')}]")
        stream = ci.get("stream_test") or {}
        if stream.get("applicable"):
            per_seed = stream.get("per_seed") or []
            counts = ", ".join(
                f"{s['seed']}:{s['b']}/{s['c']}" for s in per_seed)
            sign_p = stream.get("seed_sign_test_p")
            sign_p_s = f"{sign_p:.4g}" if _finite(sign_p) else "n/a"
            line += (f"  [finite-stream McNemar (pooled, descriptive only: "
                     f"repeated sealed items): b={stream['b']}, "
                     f"c={stream['c']}, p={stream['p_exact']:.4g}, "
                     f"n_items={stream['n']}; per-seed discordant b/c="
                     f"[{counts}]; seed signs +/-/0={stream['seed_sign_positive']}/"
                     f"{stream['seed_sign_negative']}/{stream['seed_sign_zero']}, "
                     f"independent-seed sign p={sign_p_s}]")
        elif stream.get("reason"):
            line += f"  [finite-stream test n/a: {stream['reason']}]"
        fold = stream.get("fold_audit") or {}
        if fold.get("valid"):
            loo = "yes" if fold.get("loo_sign_flip") else "no"
            fragile = ("  [FRAGILE: one stream slice drives the effect]"
                       if fold.get("fragile") else "")
            line += (
                f"  [stream folds: k={fold['n_folds']}, "
                f"effect={fold['overall_effect']:+.4f}, "
                f"range=[{fold['fold_effect_min']:+.4f}, "
                f"{fold['fold_effect_max']:+.4f}], "
                f"LOO sign flip={loo}, dominant fold share="
                f"{fold['dominant_fold_share']:.2f}; "
                f"per-seed fold sign flips="
                f"{fold['n_unit_fold_sign_flips']}; "
                f"digest verified={stream.get('digest_verified')}]"
                f"{fragile}")
        elif fold.get("reason") and fold.get("reason") != "not computed":
            line += (f"  [stream-fold companion n/a: {fold['reason']}]")
        crossed = stream.get("crossed_bootstrap") or {}
        if crossed.get("valid"):
            def _crossed_ci(lo, hi):
                if _finite(lo) and _finite(hi):
                    return f"[{float(lo):+.4f}, {float(hi):+.4f}]"
                return "[n/a]"
            share = crossed.get("variance_share_seed")
            share_s = f"{share:.2f}" if _finite(share) else "n/a"
            limited = ("  [STREAM-LIMITED: seed-level CI excludes 0 but "
                       "the seed x item CI includes 0]"
                       if crossed.get("stream_limited") else "")
            line += (
                f"  [seed x item bootstrap: 95% CI="
                f"{_crossed_ci(crossed.get('lo'), crossed.get('hi'))} "
                f"(seed-only "
                f"{_crossed_ci(crossed.get('seed_lo'), crossed.get('seed_hi'))}"
                f", item-only "
                f"{_crossed_ci(crossed.get('item_lo'), crossed.get('item_hi'))}"
                f", seed variance share={share_s})]"
                f"{limited}")
        elif crossed.get("reason") and crossed.get("reason") != "not computed":
            line += (f"  [seed x item bootstrap n/a: {crossed['reason']}]")
        if _finite(sf_p):
            floor = (f", floor={float(sf_min):.4g}"
                     if _finite(sf_min) else "")
            line += (f"  [exact sign-flip {sf_alt} p={float(sf_p):.4g}"
                     f"{floor}]")
        if rep.get("valid"):
            if null_test:
                rep_note = ("inside bound" if rep.get("replicates")
                            else "NOT inside bound")
                line += (f"  [same-size replication 95% interval="
                         f"[{rep['lo']:+.4f}, {rep['hi']:+.4f}], "
                         f"{rep_note}]")
            elif direction in ("+", "-"):
                line += (f"  [same-size replication 95% interval="
                         f"[{rep['lo']:+.4f}, {rep['hi']:+.4f}], "
                         f"P(repeat {'>' if direction == '+' else '<'} 0)="
                         f"{rep['prob_replicates']:.2f}]")
            else:
                line += (f"  [same-size replication 95% interval="
                         f"[{rep['lo']:+.4f}, {rep['hi']:+.4f}], "
                         f"P(same sign)={rep['prob_replicates']:.2f}]")
        flags = small_n_flags(ci)
        if flags:
            line += f"  [{flags}]"
        if warn:
            line += f"  [{warn}]"
        print(line)
        if audit is not None and not audit["controlled"]:
            for mismatch in audit["mismatches"]:
                print(f"      CONTROL WARNING: {mismatch}")

    def print_equivalence(self, rule, hyp, metric, diffs, warn="",
                          bound_metric=None) -> None:
        """Add a pre-registered *null-prediction* contrast (bootstrap TOST).

        ``bound`` is read from the hypothesis's registered SESOI and the
        metric must be declared as an equivalence test under
        ``sesoi_test``; otherwise the call raises, because running a null
        prediction through the difference machinery would misread "CI
        includes 0" (which may merely mean "underpowered") as support.

        ``bound_metric`` selects a separately registered bound key when
        the displayed metric is also used by directional contrasts (e.g.
        H2c's reverse check displays ``accuracy_at_longest`` but registers
        its null prediction under ``calib_minus_len_only_...``).
        """
        key = metric if bound_metric is None else bound_metric
        entry = HYPOTHESES.get(hyp, {})
        bound = (entry.get("sesoi") or {}).get(key)
        mode = (entry.get("sesoi_test") or {}).get(key)
        if mode not in ("equivalence", "noninferior") or bound is None:
            raise ValueError(
                f"{hyp}/{key} has no pre-registered equivalence or "
                f"non-inferiority bound in HYPOTHESES; register it under "
                f"'sesoi_test' first")
        ci = (noninferiority_test(diffs, bound) if mode == "noninferior"
              else equivalence_test(diffs, bound))
        # Keep the paired sample on the row: the replication companion is
        # computed inside ``print`` from ``ci["diffs"]``, and a null
        # prediction registered through this helper would otherwise always
        # report "no paired differences supplied".
        ci["diffs"] = list(diffs)
        self.print_equivalence_ci(rule, hyp, metric, ci, warn=warn)

    def print_equivalence_ci(self, rule, hyp, metric, ci, warn="") -> None:
        """Register an already-computed equivalence/non-inferiority CI.

        The two-arm reverse check needs the same paired-seed accounting and
        controlled-variable audit as a directional contrast.  Building the
        CI through ``_paired_equivalence`` and registering it here keeps the
        TOST (or one-sided non-inferiority) decision rule while refusing to
        print a null claim whose arms were not actually paired.
        """
        if ci.get("test") not in ("equivalence", "noninferiority"):
            raise ValueError(
                f"{hyp}/{metric}: print_equivalence_ci needs a CI from "
                f"equivalence_test or noninferiority_test "
                f"(got test={ci.get('test')!r})")
        self.print(rule, hyp, metric, ci, warn=warn)
