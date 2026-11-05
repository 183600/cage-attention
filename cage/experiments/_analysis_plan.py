# cage/experiments/_analysis_plan.py - the study-level analysis plan.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
"""Pre-registered, study-level analysis plan.

The three experiment families (E1/E2/E3) each control their own
Benjamini-Hochberg family at ``q``.  Those guarantees do not compose:
three families at ``q=0.05`` do not make the union of their discoveries an
FDR-0.05 set, and the study's headline claims are re-tested in more than one
family (the calibrated gate is compared with the query gate in E1 *and* E2,
and the placebo control lives in E1 only).  A reader who takes "supported in
its own family" at face value therefore reads a study-level error rate that
was never controlled.

This module closes that gap at *preparation* time:

* ``_PRIMARY_CONTRASTS`` registers the small set of contrasts that carry
  the study's confirmatory claims, and ``_STUDY_CLAIMS`` registers how
  those legs combine into the study's headline claims.  Both are checked
  against the hypothesis registry, so a leg cannot name a hypothesis,
  metric, direction or SESOI that the registry does not contain.
* ``_publish_analysis_plan`` writes the registry as a content-hashed
  artifact (``analysis_plan.json``) *before* the first grid trains and
  freezes it once any outcome row exists: revising the primary set after
  seeing the data is a post-hoc redesign and raises ``DesignPlanError``.
* ``_primary_contrast_power`` translates the pre-registered SESOI into the
  seed count the *pooled* primary family needs.  The per-experiment design
  plan certifies each family separately; the study-level family is larger,
  so a design that clears every per-family budget can still be unable to
  resolve the claim it is supposed to deliver.
* ``_study_level_adjudication`` reads the finalized contrast files and
  reports, for every registered claim, the intersection-union (IUT)
  verdict, the pooled-family verdict and the adherence of the run to the
  plan.  A study whose declared experiments are missing is reported as
  *incomplete* instead of being summarised as if it were the whole study.

None of this rewrites a per-experiment decision: the source families keep
their own pre-registered verdicts, and the study-level reading is reported
next to them.
"""

_ANALYSIS_PLAN_SCHEMA_VERSION = 1

# The experiment families that carry the study's confirmatory claims.
_STUDY_EXPERIMENTS = ("e1", "e2", "e3")

# The pre-registered primary contrasts.  Every entry names one row that the
# runners actually emit, identified by the triple (experiment, hypothesis,
# metric, rule) that the contrast log records for it.  The legs are the
# study's confirmatory evidence; everything else the runners compute stays
# in its own experiment family as a secondary or exploratory row.
#
# ``arms`` names the grid cells the leg contrasts, so the registration is
# self-describing: a reader (and the design pre-flight) can check that the
# cells a leg needs are actually trained by its family, instead of having
# to parse the human-readable ``rule`` string.  ``kind`` says how the arms
# combine: ``one_factor`` contrasts two cells, ``interaction`` contrasts
# the difference of two differences over four cells.
_PRIMARY_CONTRASTS = (
    {
        "id": "P1",
        "experiment": "e1",
        "hypothesis": "H1a",
        "metric": "abs_gate_vs_entropy",
        "rule": "calib - query",
        "arms": ("calib", "query"),
        "kind": "one_factor",
        "claim": "C1",
        "text": "the calibrated gate is associated with the attention "
                "entropy far more strongly than the query-only gate",
    },
    {
        "id": "P2",
        "experiment": "e1",
        "hypothesis": "H1a",
        "metric": "abs_gate_vs_maxprob",
        "rule": "calib - query",
        "arms": ("calib", "query"),
        "kind": "one_factor",
        "claim": "C1",
        "text": "the same holds for the max-probability concentration "
                "readout",
    },
    {
        "id": "P3",
        "experiment": "e1",
        "hypothesis": "H1f",
        "metric": "abs_gate_vs_entropy",
        "rule": "calib - calib_shuf_ent",
        "arms": ("calib", "calib_shuf_ent"),
        "kind": "one_factor",
        "claim": "C1",
        "text": "the association collapses when the gate reads a "
                "position-shuffled entropy (active placebo)",
    },
    {
        "id": "P4",
        "experiment": "e1",
        "hypothesis": "H1b",
        "metric": "gate_margin",
        "rule": "calib - query",
        "arms": ("calib", "query"),
        "kind": "one_factor",
        "claim": "C2",
        "text": "the calibrated gate closes more on false concentration than "
                "the query-only gate",
    },
    {
        "id": "P5",
        "experiment": "e1",
        "hypothesis": "H1f",
        "metric": "accuracy",
        "rule": "calib - calib_shuf_ent",
        "arms": ("calib", "calib_shuf_ent"),
        "kind": "one_factor",
        "claim": "C2",
        "text": "the placebo arm does not reproduce the calibrated gate's "
                "accuracy advantage",
    },
    {
        "id": "P6",
        "experiment": "e2",
        "hypothesis": "H2d",
        "metric": "accuracy_at_longest",
        "rule": "calib - query",
        "arms": ("calib", "query"),
        "kind": "one_factor",
        "claim": "C3",
        "text": "at the longest swept length the calibrated gate is more "
                "accurate than the query-only gate",
    },
    {
        "id": "P7",
        "experiment": "e2",
        "hypothesis": "H2d",
        "metric": "accuracy_drop",
        "rule": "calib - query",
        "arms": ("calib", "query"),
        "kind": "one_factor",
        "claim": "C3",
        "text": "the calibrated gate degrades less across the length sweep "
                "than the query-only gate",
    },
    {
        "id": "P8",
        "experiment": "e3",
        "hypothesis": "H3c",
        "metric": "accuracy",
        "rule": "seednorm x (calib - query) - pre x (calib - query)",
        "arms": ("calib|seednorm|off", "query|seednorm|off",
                 "calib|pre|off", "query|pre|off"),
        "kind": "interaction",
        "claim": "C4",
        "text": "the calibrated gate's advantage is specific to the dynamic "
                "normalisation arm (co-design interaction)",
    },
    {
        "id": "P9",
        "experiment": "e3",
        "hypothesis": "H3c",
        "metric": "gate_depth_ratio",
        "rule": "seednorm x (calib - query) - pre x (calib - query)",
        "arms": ("calib|seednorm|off", "query|seednorm|off",
                 "calib|pre|off", "query|pre|off"),
        "kind": "interaction",
        "claim": "C4",
        "text": "the same interaction on the deep/early gate ratio",
    },
    {
        "id": "P10",
        "experiment": "e1",
        "hypothesis": "H1h",
        "metric": "accuracy",
        "rule": "calib_pos_norm - calib",
        # The cell label is written literally (like the other arms here):
        # this module is loaded before the design module that owns the
        # constant, and the preparation-time reachability and single-factor
        # certificates resolve both labels against the authoritative E1
        # cell plan, so a renamed cell is caught before the grid starts.
        "arms": ("calib_pos_norm", "calib"),
        "kind": "one_factor",
        "claim": "C1",
        "text": "the calibrated gate keeps its behaviour when the position "
                "information that the global log(T) normalisation mixes into "
                "it is removed (the position-neutral arm is not materially "
                "worse than the registered arm; a one-sided "
                "non-inferiority test, not a superiority claim)",
    },
)

# The number of arms each kind of leg contrasts.  A one-factor leg contrasts
# two cells; an interaction leg contrasts the difference of two differences,
# so it needs four cells.
_PRIMARY_CONTRAST_ARITY = {"one_factor": 2, "interaction": 4}

# The study's headline claims.  A claim is a conjunction of its legs: it is
# supported only when every leg is, which is the intersection-union (IUT)
# reading and needs no multiplicity correction *across* claims (rejecting a
# conjunction requires rejecting every one of its legs).
_STUDY_CLAIMS = (
    {
        "id": "C1",
        "question": "Does the calibrated gate really read the attention "
                    "distribution, rather than merely having an extra "
                    "data-dependent input?",
        "legs": ("P1", "P2", "P3", "P10"),
    },
    {
        "id": "C2",
        "question": "Does reading the distribution change what the gate "
                    "does - suppressing writes on false concentration - "
                    "instead of only moving the attention statistics?",
        "legs": ("P4", "P5"),
    },
    {
        "id": "C3",
        "question": "Does the calibration buy length robustness beyond the "
                    "query-only gate?",
        "legs": ("P6", "P7"),
    },
    {
        "id": "C4",
        "question": "Do the calibrated gate and the dynamic normalisation "
                    "work together rather than additively?",
        "legs": ("P8", "P9"),
    },
)

_ANALYSIS_PLAN_FINGERPRINT_KEYS = (
    "schema_version", "experiments", "primary_contrasts", "claims",
    "fdr_q", "min_seeds_for_claim", "planned_seeds", "hypotheses_sha256",
)

_ANALYSIS_PLAN_REGISTRY_KEYS = (
    "schema_version", "experiments", "primary_contrasts", "claims",
)

# The per-leg fields that enter the content hash.  ``arms`` and ``kind``
# belong here: they decide which grid cells a leg needs, so changing them
# changes the design even when the rule label and the metric stay the same.
_ANALYSIS_PLAN_LEG_KEYS = (
    "id", "experiment", "hypothesis", "metric", "rule", "arms", "kind",
    "claim",
)


def _analysis_plan_legs(legs=None):
    """The registered primary contrasts as a fresh list of dictionaries."""
    source = _PRIMARY_CONTRASTS if legs is None else legs
    out = []
    for leg in source:
        out.append(dict(leg) if isinstance(leg, Mapping) else leg)
    return out


def _analysis_plan_claims(claims=None):
    """The registered study claims as a fresh list of dictionaries."""
    source = _STUDY_CLAIMS if claims is None else claims
    out = []
    for claim in source:
        out.append(dict(claim) if isinstance(claim, Mapping) else claim)
    return out


def _analysis_plan_audit(legs=None, claims=None):
    """Check the analysis plan against the hypothesis registry.

    The registry is the only source of truth for a hypothesis' experiment,
    SESOI, direction and null-prediction mode.  A plan entry that names a
    hypothesis the registry does not contain, a metric without a registered
    SESOI, a metric with neither a registered direction nor a registered
    null-prediction test, or a claim whose legs do not resolve, is a broken
    pre-registration: it would adjudicate a comparison nobody registered.
    The audit reports the problems instead of raising, so the caller can
    decide whether to refuse the run.
    """
    issues = []
    legs = _analysis_plan_legs(legs)
    claims = _analysis_plan_claims(claims)
    leg_ids = []
    for leg in legs:
        if not isinstance(leg, Mapping):
            issues.append(f"primary contrast {leg!r} is not a mapping")
            continue
        lid = leg.get("id")
        if not isinstance(lid, str) or not lid.strip():
            issues.append(f"primary contrast {leg!r} has no usable id")
            continue
        if lid in leg_ids:
            issues.append(f"primary contrast id {lid!r} is used twice")
        leg_ids.append(lid)
        experiment = leg.get("experiment")
        if experiment not in _STUDY_EXPERIMENTS:
            issues.append(
                f"{lid}: experiment {experiment!r} is not one of the study "
                f"families {list(_STUDY_EXPERIMENTS)}")
        hypothesis = leg.get("hypothesis")
        entry = HYPOTHESES.get(hypothesis) if isinstance(hypothesis, str) \
            else None
        if entry is None:
            issues.append(
                f"{lid}: hypothesis {hypothesis!r} is not in the registered "
                f"hypothesis registry")
            continue
        if entry.get("experiment") != experiment:
            issues.append(
                f"{lid}: hypothesis {hypothesis!r} belongs to experiment "
                f"{entry.get('experiment')!r}, not {experiment!r}")
        metric = leg.get("metric")
        sesoi = entry.get("sesoi") or {}
        if metric not in sesoi:
            issues.append(
                f"{lid}: metric {metric!r} has no registered SESOI under "
                f"{hypothesis!r}")
        directions = entry.get("direction") or {}
        null_tests = entry.get("sesoi_test") or {}
        if metric not in directions and metric not in null_tests:
            issues.append(
                f"{lid}: metric {metric!r} has neither a registered "
                f"direction nor a registered null-prediction test under "
                f"{hypothesis!r}")
        rule = leg.get("rule")
        if not isinstance(rule, str) or not rule.strip():
            issues.append(f"{lid}: the contrast label (rule) is missing")
        kind = leg.get("kind")
        if kind not in _PRIMARY_CONTRAST_ARITY:
            issues.append(
                f"{lid}: contrast kind {kind!r} is not one of "
                f"{sorted(_PRIMARY_CONTRAST_ARITY)}")
        arms = leg.get("arms")
        if isinstance(arms, (str, bytes)) or not isinstance(
                arms, (list, tuple)):
            issues.append(
                f"{lid}: arms must be a list of grid-cell labels")
        else:
            labels = list(arms)
            if not labels:
                issues.append(f"{lid}: a contrast with no arms adjudicates "
                              f"nothing")
            elif any(not isinstance(label, str) or not label.strip()
                     for label in labels):
                issues.append(f"{lid}: every arm label must be a non-empty "
                              f"string")
            elif len(set(labels)) != len(labels):
                issues.append(f"{lid}: the same arm is listed twice")
            elif kind in _PRIMARY_CONTRAST_ARITY and \
                    len(labels) != _PRIMARY_CONTRAST_ARITY[kind]:
                issues.append(
                    f"{lid}: a {kind!r} contrast contrasts "
                    f"{_PRIMARY_CONTRAST_ARITY[kind]} cells, but "
                    f"{len(labels)} were named")
    claim_ids = []
    used_legs = []
    for claim in claims:
        if not isinstance(claim, Mapping):
            issues.append(f"study claim {claim!r} is not a mapping")
            continue
        cid = claim.get("id")
        if not isinstance(cid, str) or not cid.strip():
            issues.append(f"study claim {claim!r} has no usable id")
            continue
        if cid in claim_ids:
            issues.append(f"study claim id {cid!r} is used twice")
        claim_ids.append(cid)
        claim_legs = claim.get("legs")
        if isinstance(claim_legs, (str, bytes)) or not isinstance(
                claim_legs, (list, tuple)):
            issues.append(f"{cid}: legs must be a list of primary contrast ids")
            continue
        if not claim_legs:
            issues.append(f"{cid}: a study claim without legs adjudicates "
                          f"nothing")
        for leg_id in claim_legs:
            if leg_id not in leg_ids:
                issues.append(
                    f"{cid}: leg {leg_id!r} is not a registered primary "
                    f"contrast")
            elif leg_id in used_legs:
                issues.append(
                    f"{cid}: leg {leg_id!r} is already used by another claim")
            else:
                used_legs.append(leg_id)
    for leg in legs:
        if not isinstance(leg, Mapping):
            continue
        lid = leg.get("id")
        if isinstance(lid, str) and lid not in used_legs:
            issues.append(
                f"{lid}: primary contrast belongs to no study claim, so it "
                f"would enter the pooled family without a claim to support")
    return {
        "ok": not issues,
        "issues": issues,
        "n_legs": len(legs),
        "n_claims": len(claims),
    }


def _analysis_plan_payload(settings):
    """The full analysis-plan payload, including its content hashes."""
    audit = _analysis_plan_audit()
    payload = {
        "schema_version": _ANALYSIS_PLAN_SCHEMA_VERSION,
        "description": "Pre-registered study-level analysis plan: the "
                       "confirmatory contrasts, the claims they form and the "
                       "multiplicity level under which they are adjudicated.",
        "experiments": list(_STUDY_EXPERIMENTS),
        "primary_contrasts": _analysis_plan_legs(),
        "claims": _analysis_plan_claims(),
        "fdr_q": float(settings.fdr_q),
        "min_seeds_for_claim": int(settings.min_seeds_for_claim),
        "planned_seeds": [int(seed) for seed in settings.seeds],
        "hypotheses_sha256": _hypotheses_sha(),
        "registry_consistent": bool(audit["ok"]),
        "registry_issues": list(audit["issues"]),
    }
    payload["analysis_plan_registry_sha256"] = \
        _analysis_plan_registry_fingerprint(payload)
    payload["analysis_plan_sha256"] = _analysis_plan_fingerprint(payload)
    return payload


def _analysis_plan_registry_payload():
    """The settings-independent part of the plan: what will be analysed.

    The registry hash covers only the registered contrasts and the claims
    they form.  It is the part that a later reader can compare against the
    code that produced a result, independently of the seed count or the
    multiplicity level a particular invocation used.
    """
    return {
        "schema_version": _ANALYSIS_PLAN_SCHEMA_VERSION,
        "experiments": list(_STUDY_EXPERIMENTS),
        "primary_contrasts": _analysis_plan_legs(),
        "claims": _analysis_plan_claims(),
    }


def _analysis_plan_registry_fingerprint(payload):
    """Content hash of the registered contrasts and claims alone."""
    if not isinstance(payload, Mapping):
        return None
    missing = [key for key in _ANALYSIS_PLAN_REGISTRY_KEYS
               if key not in payload]
    if missing:
        return None
    contrasts = payload.get("primary_contrasts")
    claims = payload.get("claims")
    experiments = payload.get("experiments")
    for container in (contrasts, claims, experiments):
        if isinstance(container, (str, bytes, Mapping)):
            return None
        try:
            list(container)
        except TypeError:
            return None
    design = {
        "schema_version": payload.get("schema_version"),
        "experiments": list(experiments),
        "primary_contrasts": [
            {str(key): leg.get(key) for key in _ANALYSIS_PLAN_LEG_KEYS}
            if isinstance(leg, Mapping) else leg
            for leg in contrasts],
        "claims": [
            {str(key): claim.get(key) for key in ("id", "legs")}
            if isinstance(claim, Mapping) else claim
            for claim in claims],
    }
    try:
        encoded = json.dumps(design, ensure_ascii=False, sort_keys=True,
                             default=str)
    except (TypeError, ValueError, OverflowError):
        return None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _analysis_plan_fingerprint(payload):
    """Content hash of the *design-relevant* part of the analysis plan.

    Invocation provenance (run id, environment, timestamps) changes on every
    invocation and must not change the pre-registration, so only the
    registered contrasts, the claims they form, the multiplicity level, the
    claim floor, the planned seeds and the hypothesis-registry hash enter
    the hash.  A malformed or partial payload returns ``None``: callers must
    read that as "cannot certify this plan", never as "unchanged".
    """
    if not isinstance(payload, Mapping):
        return None
    missing = [key for key in _ANALYSIS_PLAN_FINGERPRINT_KEYS
               if key not in payload]
    if missing:
        return None
    contrasts = payload.get("primary_contrasts")
    claims = payload.get("claims")
    seeds = payload.get("planned_seeds")
    experiments = payload.get("experiments")
    for container in (contrasts, claims, seeds, experiments):
        if container is None:
            return None
        if isinstance(container, (str, bytes, Mapping)):
            return None
        try:
            list(container)
        except TypeError:
            return None
    design = {
        "schema_version": payload.get("schema_version"),
        "experiments": list(experiments),
        "primary_contrasts": [
            {str(key): leg.get(key) for key in _ANALYSIS_PLAN_LEG_KEYS}
            if isinstance(leg, Mapping) else leg
            for leg in contrasts],
        "claims": [
            {str(key): claim.get(key) for key in ("id", "legs")}
            if isinstance(claim, Mapping) else claim
            for claim in claims],
        "fdr_q": payload.get("fdr_q"),
        "min_seeds_for_claim": payload.get("min_seeds_for_claim"),
        "planned_seeds": list(seeds),
        "hypotheses_sha256": payload.get("hypotheses_sha256"),
    }
    try:
        encoded = json.dumps(design, ensure_ascii=False, sort_keys=True,
                             default=str)
    except (TypeError, ValueError, OverflowError):
        return None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _analysis_plan_outcomes_exist(out_dir) -> bool:
    """Whether *any* study family already has outcome rows on disk.

    The freeze is study-wide, not experiment-wide: the primary set is a
    single study-level registration, so E1 may not be analysed under one
    primary set and E2 under another.  A missing or unreadable outcome path
    counts as evidence, so a plan published next to it cannot pretend to be
    prospective.
    """
    root = pathlib.Path(out_dir)
    for tag in _STUDY_EXPERIMENTS:
        for suffix in ("_runs.jsonl", "_contrasts.json"):
            path = root / f"{tag}{suffix}"
            try:
                if not path.exists() and not path.is_symlink():
                    continue
                if not path.is_file():
                    return True
                if path.stat().st_size > 0:
                    return True
            except OSError:
                return True
    return False


def _record_analysis_plan_event(settings, out_dir, event: str,
                                plan_sha256: str,
                                supersedes: str | None = None) -> None:
    """Append one analysis-plan event to an auditable history."""
    record = {
        "event": event,
        "analysis_plan_sha256": plan_sha256,
        "supersedes_analysis_plan_sha256": supersedes,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **_stamp(settings),
    }
    _write_result(pathlib.Path(out_dir) / "analysis_plan_history.jsonl",
                  record)


def _publish_analysis_plan(settings, out_dir) -> dict:
    """Publish the study-level analysis plan, freezing it after outcomes.

    Before any outcome row exists the plan may still be amended - that is
    what preparation is - and every amendment is kept in
    ``analysis_plan_history.jsonl``.  Once any family has written rows, the
    registered primary set is frozen: changing it then would be choosing
    the confirmatory contrasts after seeing the data, so it raises
    ``DesignPlanError`` instead of overwriting the artifact that was
    supposed to pre-register the study.  Re-publishing an identical plan is
    allowed.
    """
    audit = _analysis_plan_audit()
    if not audit["ok"]:
        raise ValueError(
            "the pre-registered analysis plan is inconsistent with the "
            "hypothesis registry: " + "; ".join(audit["issues"]))
    payload = _analysis_plan_payload(settings)
    new_hash = payload.get("analysis_plan_sha256")
    if not isinstance(new_hash, str) or not new_hash:
        raise DesignPlanError(
            "refusing to publish an analysis plan with no readable content "
            "hash")
    path = pathlib.Path(out_dir) / "analysis_plan.json"
    if path.exists() and not path.is_file():
        raise DesignPlanError(
            f"the existing analysis-plan path {path} is not a regular file, "
            f"so it cannot be verified or replaced; restore the registered "
            f"plan or use a fresh --out directory")
    old = None
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            old = None
    outcomes = _analysis_plan_outcomes_exist(out_dir)
    old_hash = _analysis_plan_fingerprint(old) if isinstance(old, Mapping) \
        else None
    if outcomes and not path.exists():
        raise DesignPlanError(
            "outcome rows already exist but no registered analysis plan "
            "does; a plan written now would be post-hoc, so restore the "
            "original plan or start the study in a fresh --out directory")
    if path.exists() and outcomes:
        if old_hash is None:
            raise DesignPlanError(
                "an unreadable existing analysis plan cannot be verified "
                "after outcome rows were written; restore the registered "
                "plan or use a fresh --out directory")
        if old_hash != new_hash:
            raise DesignPlanError(
                f"refusing to revise the registered analysis plan "
                f"{old_hash} -> {new_hash} after outcome rows exist; "
                f"changing the confirmatory contrasts, their multiplicity "
                f"level, the claim floor or the planned seed count after the "
                f"data were seen is a post-hoc redesign, so use a fresh "
                f"--out directory and report it as a separate study")
    if path.exists() and old_hash is not None and old_hash != new_hash:
        payload["supersedes_analysis_plan_sha256"] = old_hash
        _record_analysis_plan_event(settings, out_dir, "amended", new_hash,
                                    old_hash)
    elif not path.exists():
        _record_analysis_plan_event(settings, out_dir, "created", new_hash)
    elif old_hash is None:
        _record_analysis_plan_event(settings, out_dir, "replaced_malformed",
                                    new_hash)
    _write_json(path, payload)
    return payload


def _directional_p(row):
    """One-sided p-value of a contrast row in its registered direction.

    The contrast log stores the two-sided bootstrap p-value and the
    separately registered direction.  For a directional prediction the
    one-sided p-value is half the two-sided one when the observed effect
    points the predicted way and ``1 - p/2`` when it points the other way;
    an undirected prediction keeps the two-sided value.  This is what the
    intersection-union test of a claim needs: every leg is tested against
    its own registered alternative, not against the pooled family's
    threshold.
    """
    p_value = row.get("p_value")
    if not (_finite(p_value) and 0.0 <= float(p_value) <= 1.0):
        return None
    p_value = float(p_value)
    direction = row.get("direction")
    if direction not in ("+", "-"):
        return p_value
    mean = row.get("mean")
    if not (_finite(mean) and float(mean) != 0.0):
        return None
    half = p_value / 2.0
    if direction == "+":
        return half if float(mean) > 0.0 else 1.0 - half
    return half if float(mean) < 0.0 else 1.0 - half


def _primary_contrast_power(settings, verdict, legs=None):
    """Prospective power of the pooled primary family.

    The per-experiment design plan certifies each family on its own: it
    asks whether *that* family can resolve the SESOI.  The study-level
    family is the pooled set of primary contrasts, so the family size is
    larger and the threshold a single true effect must beat is lower (the
    worst case of Benjamini-Hochberg is ``q / m``).  A design can therefore
    clear every per-family budget and still be unable to deliver the
    study-level claim; this block says so before the grid spends compute.

    Two thresholds are reported per leg:

    * ``claim_level`` uses the un-inflated ``q`` - the level at which an
      intersection-union test of a claim rejects one of its legs;
    * ``pooled_family`` uses ``q / m`` - the worst-case level at which the
      pooled Benjamini-Hochberg family can reject its smallest p-value.

    Only accuracy-scale legs have an empirical pilot variance; the other
    legs report ``no_pilot`` instead of a fabricated seed count, exactly
    like the per-experiment plan.
    """
    legs = _analysis_plan_legs(legs)
    audit = _analysis_plan_audit(legs=legs)
    n = len(settings.seeds)
    q = float(settings.fdr_q)
    alpha = float(settings.design_alpha)
    target = float(settings.power_target)
    paired_sd, paired_source, _ = _pilot_paired_accuracy_sd(verdict)
    m = max(len(legs), 1)
    alpha_pooled = q / m
    rows = []
    for leg in legs:
        hypothesis = leg.get("hypothesis")
        metric = leg.get("metric")
        entry = HYPOTHESES.get(hypothesis) or {}
        sesoi = (entry.get("sesoi") or {}).get(metric)
        accuracy_metric = "accuracy" in str(metric)
        sd = paired_sd if accuracy_metric else float("nan")
        row = {
            "id": leg.get("id"),
            "experiment": leg.get("experiment"),
            "hypothesis": hypothesis,
            "metric": metric,
            "sesoi": sesoi,
            "pilot_paired_sd": (float(sd) if accuracy_metric
                                and _finite(sd) else None),
            "pilot_paired_source": paired_source if accuracy_metric else None,
        }
        for label, level in (("claim_level", q), ("pooled_family",
                                                  alpha_pooled)):
            z = _power_z("superiority", level, target)
            max_sd = (float(sesoi) * math.sqrt(max(n, 1)) / z
                      if (_finite(sesoi) and _finite(z) and z > 0.0)
                      else None)
            required = None
            power = None
            unattainable = False
            if accuracy_metric and _finite(sd) and float(sd) > 0.0:
                required = _required_seeds("superiority", sd, sesoi, level,
                                           target)
                if required is not None and not _finite(required):
                    # A SESOI so small that the required seed count is not
                    # representable is unattainable, not a huge integer.
                    # It is *known* that no planned n can resolve it, so the
                    # leg is reported as underpowered rather than as
                    # "unknown"; the distinction from a missing pilot is
                    # kept in the explicit flag.
                    required = None
                    unattainable = True
                power = _power_at_effect("superiority", sesoi, sd, n, level)
            row[f"alpha_{label}"] = level
            row[f"z_{label}"] = z if _finite(z) else None
            row[f"max_sd_for_power_{label}"] = max_sd
            row[f"required_seeds_{label}"] = required
            row[f"required_seeds_unattainable_{label}"] = unattainable
            row[f"power_at_planned_{label}"] = power
            row[f"powered_{label}"] = (
                None if (required is None and not unattainable)
                else bool(required is not None
                          and n >= math.ceil(float(required))))
        rows.append(row)
    claim_rows = []
    for claim in _analysis_plan_claims():
        legs_of_claim = [row for row in rows
                         if row.get("id") in (claim.get("legs") or ())]
        powered = [row for row in legs_of_claim
                   if row.get("powered_claim_level") is not None]
        unknown = [row for row in legs_of_claim
                   if row.get("powered_claim_level") is None]
        required = [row.get("required_seeds_claim_level")
                    for row in legs_of_claim
                    if row.get("required_seeds_claim_level") is not None]
        powers = [row.get("power_at_planned_claim_level")
                  for row in legs_of_claim
                  if row.get("power_at_planned_claim_level") is not None]
        # A leg whose requirement is unattainable binds the claim harder
        # than any finite seed count does, so it is named as the binding
        # leg even though it carries no finite requirement.
        unattainable = [row for row in legs_of_claim
                        if row.get("required_seeds_unattainable_claim_level")]
        claim_rows.append({
            "id": claim.get("id"),
            "legs": list(claim.get("legs") or ()),
            "n_legs": len(legs_of_claim),
            "n_legs_without_pilot": len(unknown),
            "n_legs_unattainable": len(unattainable),
            "required_seeds": (max(required) if required else None),
            "binding_leg": (
                unattainable[0]["id"] if unattainable
                else (max(
                    (row for row in legs_of_claim
                     if row.get("required_seeds_claim_level") is not None),
                    key=lambda row: row["required_seeds_claim_level"])["id"]
                    if required else None)),
            "min_leg_power_at_planned": (min(powers) if powers else None),
            "powered": (None if not powered else
                        bool(all(row.get("powered_claim_level")
                                 for row in powered))),
        })
    underpowered = [row["id"] for row in rows
                    if row.get("powered_claim_level") is False]
    unpiloted = [row["id"] for row in rows
                 if row.get("powered_claim_level") is None]
    return {
        "registry_consistent": bool(audit["ok"]),
        "registry_issues": list(audit["issues"]),
        "n_primary_contrasts": len(rows),
        "n_claims": len(claim_rows),
        "fdr_q": q,
        "alpha_claim_level": q,
        "alpha_pooled_family_worst_case": alpha_pooled,
        "planned_n": n,
        "legs": rows,
        "claims": claim_rows,
        "underpowered_claim_level_legs": underpowered,
        "legs_without_pilot": unpiloted,
        "all_legs_powered_at_claim_level": (
            None if unpiloted else not underpowered),
        "note": "A leg without an empirical pilot variance reports "
                "no_pilot instead of a seed count; the pooled threshold "
                "q/m is the worst-case Benjamini-Hochberg level for the "
                "single best primary contrast.",
    }


def _study_level_adjudication(out_dir, q: float = 0.05) -> dict:
    """Adjudicate the study-level claims against the finalized families.

    The function reads the finalized ``<tag>_contrasts.json`` files, pairs
    them with the registered primary contrasts and reports three things:

    * the **claim verdict** per registered claim, using the
      intersection-union reading: a conjunction is supported only when
      every one of its legs is, and the claim's global p-value is the
      largest leg p-value in the registered direction;
    * the **pooled-family verdict**, a Benjamini-Hochberg screen over the
      pooled primary contrasts with the by/Holm dependence-robust
      companions, and the count of claims that were supported inside their
      own experiment family but not in the pooled family;
    * the **adherence** of the evidence to the plan: missing experiment
      families, missing or ambiguous primary contrasts, and an analysis
      plan whose content hash no longer matches the registry.

    A study whose declared families are missing is reported as incomplete:
    a partial set of families must not be summarised as if it were the
    whole study.
    """
    try:
        q = float(q)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"q must be a finite number in (0, 1), got {q!r}") from exc
    if not (math.isfinite(q) and 0.0 < q < 1.0):
        raise ValueError(f"q must be a finite number in (0, 1), got {q!r}")
    root = pathlib.Path(out_dir)
    legs = _analysis_plan_legs()
    claims = _analysis_plan_claims()
    audit = _analysis_plan_audit(legs=legs, claims=claims)
    sources = {}
    matched = {}
    missing_experiments = []
    invalid_sources = []
    for tag in _STUDY_EXPERIMENTS:
        path = root / f"{tag}_contrasts.json"
        if not path.is_file():
            sources[str(tag)] = {"present": False}
            missing_experiments.append(str(tag))
            continue
        try:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            sources[str(tag)] = {
                "present": True, "valid": False,
                "error": f"{type(exc).__name__}: {exc}"}
            invalid_sources.append(str(tag))
            continue
        rows = payload.get("rows") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            sources[str(tag)] = {
                "present": True, "valid": False,
                "error": "payload has no 'rows' list"}
            invalid_sources.append(str(tag))
            continue
        recorded_tag = payload.get("tag")
        if recorded_tag is not None and str(recorded_tag) != str(tag):
            sources[str(tag)] = {
                "present": True, "valid": False,
                "error": (f"payload tag {recorded_tag!r} does not match "
                          f"file tag {str(tag)!r}")}
            invalid_sources.append(str(tag))
            continue
        sources[str(tag)] = {
            "present": True, "valid": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "q": payload.get("q"),
            "n_rows": len(rows),
        }
        for leg in legs:
            if leg.get("experiment") != tag:
                continue
            hits = [
                row for row in rows
                if isinstance(row, Mapping)
                and row.get("hypothesis") == leg.get("hypothesis")
                and row.get("metric") == leg.get("metric")
                and row.get("rule") == leg.get("rule")]
            matched.setdefault(leg.get("id"), []).extend(hits)
    leg_reports = []
    for leg in legs:
        hits = matched.get(leg.get("id")) or []
        report = {
            "id": leg.get("id"),
            "experiment": leg.get("experiment"),
            "hypothesis": leg.get("hypothesis"),
            "metric": leg.get("metric"),
            "rule": leg.get("rule"),
            "claim": leg.get("claim"),
            "text": leg.get("text"),
        }
        if not hits:
            report["status"] = "missing"
            report["measured"] = False
        elif len(hits) > 1:
            report["status"] = "ambiguous"
            report["measured"] = False
            report["n_matching_rows"] = len(hits)
        else:
            row = hits[0]
            report["status"] = "measured"
            report["measured"] = True
            report["n"] = row.get("n")
            report["mean"] = row.get("mean")
            report["lo"] = row.get("lo")
            report["hi"] = row.get("hi")
            report["p_value"] = row.get("p_value")
            report["p_adj_in_family"] = row.get("p_adj")
            report["direction"] = row.get("direction")
            report["sesoi"] = row.get("sesoi")
            report["supports_in_family"] = bool(row.get("supports_hypothesis"))
            report["fdr_reject_in_family"] = bool(row.get("fdr_reject"))
            report["practically_relevant"] = bool(
                row.get("practically_relevant"))
            report["claim_eligible"] = bool(row.get("claim_eligible"))
            report["controlled"] = row.get("controlled")
            report["confounded"] = bool(row.get("confounded"))
            report["direction_reversed"] = bool(row.get("direction_reversed"))
            one_sided = _directional_p(row)
            report["p_one_sided"] = one_sided
            eligible = bool(row.get("controlled") is not False
                            and row.get("claim_eligible", True))
            report["leg_eligible"] = eligible
            report["leg_rejects_claim_level"] = bool(
                one_sided is not None and float(one_sided) <= q
                and row.get("practically_relevant") and eligible)
            # A row that is *not* eligible for a claim - too few paired
            # seeds, a curve-invalid arm, an unbalanced evaluation stream,
            # an uncontrolled comparison - cannot decide the claim in
            # either direction.  A significant effect in the opposite
            # direction is only a refutation when the leg was allowed to
            # support the claim in the first place; this is the same
            # convention the per-hypothesis synthesis uses.
            report["leg_refutes_claim"] = bool(
                report.get("direction_reversed")
                and report.get("fdr_reject_in_family") and eligible)
        leg_reports.append(report)
    testable = [report for report in leg_reports
                if report.get("measured")
                and _finite(report.get("p_value"))]
    p_values = [float(report["p_value"]) for report in testable]
    bh = benjamini_hochberg(p_values, q=q) if p_values else []
    by = benjamini_yekutieli(p_values, q=q) if p_values else []
    holm = holm_bonferroni(p_values, q=q) if p_values else []
    for report, bh_row, by_row, holm_row in zip(testable, bh, by, holm):
        report["p_adj_pooled_family"] = bh_row["p_adj"]
        report["pooled_fdr_reject"] = bool(bh_row["reject"])
        report["pooled_by_p_adj"] = by_row["p_adj"]
        report["pooled_by_reject"] = bool(by_row["reject"])
        report["pooled_holm_p_adj"] = holm_row["p_adj"]
        report["pooled_holm_reject"] = bool(holm_row["reject"])
        report["supports_pooled_family"] = bool(
            report.get("supports_in_family") and bh_row["reject"])
        report["family_inflated"] = bool(
            report.get("supports_in_family") and not bh_row["reject"])
    claim_reports = []
    for claim in claims:
        leg_ids = list(claim.get("legs") or ())
        leg_rows = [report for report in leg_reports
                    if report.get("id") in leg_ids]
        measured = [report for report in leg_rows if report.get("measured")]
        claim_report = {
            "id": claim.get("id"),
            "question": claim.get("question"),
            "legs": leg_ids,
            "n_legs": len(leg_ids),
            "n_legs_measured": len(measured),
            "n_legs_supporting_in_family": sum(
                1 for report in leg_rows
                if report.get("supports_in_family")),
            "n_legs_rejecting_claim_level": sum(
                1 for report in leg_rows
                if report.get("leg_rejects_claim_level")),
            "n_legs_supporting_pooled_family": sum(
                1 for report in leg_rows
                if report.get("supports_pooled_family")),
            "direction_reversed_legs": [
                report.get("id") for report in leg_rows
                if report.get("leg_refutes_claim")],
            "direction_reversed_but_ineligible_legs": [
                report.get("id") for report in leg_rows
                if report.get("direction_reversed")
                and report.get("fdr_reject_in_family")
                and not report.get("leg_eligible")],
        }
        one_sided = [report.get("p_one_sided") for report in leg_rows
                     if report.get("p_one_sided") is not None]
        claim_report["global_p_one_sided"] = (
            max(one_sided) if len(one_sided) == len(leg_ids) else None)
        if len(measured) < len(leg_ids):
            claim_report["status"] = "not_adjudicable"
            claim_report["reason"] = (
                "one or more legs of the claim were not measured")
        elif all(report.get("supports_in_family") for report in leg_rows):
            claim_report["status"] = "supported"
        elif any(report.get("supports_in_family") for report in leg_rows):
            claim_report["status"] = "partial"
        elif claim_report["direction_reversed_legs"]:
            claim_report["status"] = "refuted"
        else:
            claim_report["status"] = "inconclusive"
        claim_report["supports_all_legs_claim_level"] = bool(
            claim_report["status"] != "not_adjudicable"
            and claim_report["n_legs_rejecting_claim_level"] == len(leg_ids))
        # The second reading: the same intersection-union logic, but with
        # every leg judged inside the *pooled* primary family instead of
        # inside its own experiment family.  The per-family verdicts are
        # not wrong - they are the pre-registered decisions of their own
        # families - but a claim whose legs each survived their own FDR
        # family has not necessarily survived the study-level family, and
        # the study-level family is the one that controls the error rate
        # of the study's headline claims.
        pooled_support = [bool(report.get("supports_pooled_family"))
                          for report in leg_rows]
        if len(measured) < len(leg_ids):
            claim_report["status_pooled_family"] = "not_adjudicable"
        elif all(pooled_support):
            claim_report["status_pooled_family"] = "supported"
        elif any(pooled_support):
            claim_report["status_pooled_family"] = "partial"
        else:
            claim_report["status_pooled_family"] = "inconclusive"
        claim_report["family_inflated"] = bool(
            claim_report["status"] == "supported"
            and claim_report["status_pooled_family"] != "supported")
        claim_reports.append(claim_report)
    plan_path = root / "analysis_plan.json"
    plan_state = {"present": False}
    if plan_path.is_file():
        try:
            stored = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            stored = None
            plan_state = {"present": True, "valid": False,
                          "error": f"{type(exc).__name__}: {exc}"}
        else:
            stored_hash = _analysis_plan_registry_fingerprint(stored) \
                if isinstance(stored, Mapping) else None
            current_hash = _analysis_plan_registry_fingerprint(
                _analysis_plan_registry_payload())
            plan_state = {
                "present": True,
                "valid": stored_hash is not None,
                "stored_registry_sha256": stored_hash,
                "current_registry_sha256": current_hash,
                "stored_sha256": (
                    _analysis_plan_fingerprint(stored)
                    if isinstance(stored, Mapping) else None),
                "matches_registry": bool(
                    stored_hash is not None and current_hash is not None
                    and stored_hash == current_hash),
            }
    missing_legs = [report["id"] for report in leg_reports
                    if report.get("status") == "missing"]
    ambiguous_legs = [report["id"] for report in leg_reports
                      if report.get("status") == "ambiguous"]
    deviations = []
    if missing_experiments:
        deviations.append(
            f"study families not present: {', '.join(missing_experiments)}")
    if invalid_sources:
        deviations.append(
            f"study families present but unusable: "
            f"{', '.join(invalid_sources)}")
    if missing_legs:
        deviations.append(
            f"registered primary contrasts without a row: "
            f"{', '.join(missing_legs)}")
    if ambiguous_legs:
        deviations.append(
            f"registered primary contrasts matched by more than one row: "
            f"{', '.join(ambiguous_legs)}")
    if plan_state.get("present") and plan_state.get("matches_registry") \
            is False:
        deviations.append(
            "the registered analysis plan does not match the current "
            "registry (the confirmatory contrasts changed after the run)")
    if not audit["ok"]:
        deviations.append(
            "the analysis plan is inconsistent with the hypothesis registry: "
            + "; ".join(audit["issues"]))
    complete = not missing_experiments and not invalid_sources
    statuses = [claim_report.get("status") for claim_report in claim_reports]
    pooled_statuses = [claim_report.get("status_pooled_family")
                       for claim_report in claim_reports]
    if not complete:
        study_status = "incomplete"
    elif statuses and all(status == "supported" for status in statuses):
        study_status = "supported"
    elif any(status == "supported" for status in statuses):
        study_status = "partial"
    elif any(status == "refuted" for status in statuses):
        study_status = "refuted"
    else:
        study_status = "inconclusive"
    if not complete:
        study_status_pooled = "incomplete"
    elif pooled_statuses and all(status == "supported"
                                 for status in pooled_statuses):
        study_status_pooled = "supported"
    elif any(status == "supported" for status in pooled_statuses):
        study_status_pooled = "partial"
    else:
        study_status_pooled = "inconclusive"
    payload = {
        "q": q,
        "schema_version": _ANALYSIS_PLAN_SCHEMA_VERSION,
        "experiments": list(_STUDY_EXPERIMENTS),
        "sources": sources,
        "missing_experiments": missing_experiments,
        "invalid_sources": invalid_sources,
        "registry_consistent": bool(audit["ok"]),
        "registry_issues": list(audit["issues"]),
        "analysis_plan": plan_state,
        "n_primary_contrasts": len(leg_reports),
        "n_measured_primary_contrasts": len(testable),
        "legs": leg_reports,
        "claims": claim_reports,
        "n_claims_supported": sum(
            1 for claim_report in claim_reports
            if claim_report.get("status") == "supported"),
        "n_claims_partial": sum(
            1 for claim_report in claim_reports
            if claim_report.get("status") == "partial"),
        "n_claims_refuted": sum(
            1 for claim_report in claim_reports
            if claim_report.get("status") == "refuted"),
        "n_claims_not_adjudicable": sum(
            1 for claim_report in claim_reports
            if claim_report.get("status") == "not_adjudicable"),
        "n_family_inflated_legs": sum(
            1 for report in leg_reports if report.get("family_inflated")),
        "family_inflated_legs": [
            report.get("id") for report in leg_reports
            if report.get("family_inflated")],
        "n_claims_supported_pooled_family": sum(
            1 for claim_report in claim_reports
            if claim_report.get("status_pooled_family") == "supported"),
        "family_inflated_claims": [
            claim_report.get("id") for claim_report in claim_reports
            if claim_report.get("family_inflated")],
        "deviations": deviations,
        "adherence_ok": not deviations,
        "study_complete": complete,
        "study_status": study_status,
        "study_status_pooled_family": study_status_pooled,
        "note": "Study-level companion over the pre-registered primary "
                "contrasts. Per-experiment decisions are unchanged; a leg "
                "supported inside its own family but not inside the pooled "
                "family is reported as family-inflated, and a claim is "
                "supported only when every one of its legs is.  "
                "study_status reads the per-family verdicts, "
                "study_status_pooled_family reads the same claims inside "
                "the pooled primary family.",
    }
    _write_json(root / "study_level_adjudication.json", payload)
    print(f"\n[study] pre-registered analysis plan ({len(leg_reports)} "
          f"primary contrast(s), {len(claim_reports)} claim(s), q={q:g}): "
          f"{payload['n_claims_supported']}/{len(claim_reports)} claim(s) "
          f"supported, {payload['n_claims_partial']} partial, "
          f"{payload['n_claims_refuted']} refuted, "
          f"{payload['n_claims_not_adjudicable']} not adjudicable; study "
          f"status: {study_status}")
    if missing_experiments:
        print(f"[study] WARNING: study famil(y/ies) "
              f"{', '.join(missing_experiments)} are missing; the study is "
              f"INCOMPLETE and must not be reported as a whole-study result")
    if deviations:
        print(f"[study] WARNING: {len(deviations)} deviation(s) from the "
              f"registered analysis plan: {'; '.join(deviations)}")
    if payload["n_family_inflated_legs"]:
        print(f"[study] WARNING: {payload['n_family_inflated_legs']} "
              f"primary contrast(s) are supported inside their own family "
              f"but not inside the pooled study family "
              f"({', '.join(payload['family_inflated_legs'])}); report them "
              f"as study-level uncertain")
    if payload["family_inflated_claims"]:
        print(f"[study] WARNING: {len(payload['family_inflated_claims'])} "
              f"claim(s) ({', '.join(payload['family_inflated_claims'])}) "
              f"are supported inside their own families but not inside the "
              f"pooled study family; the study-level reading of these "
              f"claims is '{study_status_pooled}'")
    return payload
