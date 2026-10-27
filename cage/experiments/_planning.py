# cage/experiments/_planning.py - Power/SESOI planning, design plans and recipe views.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _exact_alternatives_for(metric: str, mode: str, hypotheses) -> list:
    """Registered sign-flip alternatives behind one planned metric.

    A metric row can merge several hypotheses (same metric/bound/test
    mode), and the hypotheses need not read that metric in the same
    direction: H2e registers a directional concentration main effect and a
    two-sided interaction on the same accuracy readout.  Planning only the
    pooled main-effect sign would therefore miss a registered two-sided
    companion whose exact-test floor is higher.  For every contributing
    hypothesis this helper reads its own ``direction`` registry:

    * a null prediction (equivalence/non-inferiority) is read through
      one-sided boundary tests, so its floor is the one-sided floor;
    * an explicit ``+``/``-`` direction is one-sided, while an explicit
      ``None`` (or a metric with no registered direction) is two-sided;
    * direction keys that name a companion of this display metric
      (``interaction_accuracy``, ``query_none_accuracy``,
      ``postmix_norm_...``) contribute their own alternative, so an
      undirected companion cannot hide behind a directional main effect.

    The union is deliberately conservative: the design must clear the
    largest floor among the registered readings of that metric.
    """
    alternatives = set()
    for hyp in hypotheses:
        spec = HYPOTHESES.get(hyp) or {}
        direction = spec.get("direction") or {}
        if mode in ("equivalence", "noninferior"):
            # TOST / non-inferiority use one-sided boundary tests even when
            # the display metric also carries a directional main effect.
            alternatives.add("greater")
        elif metric in direction:
            value = direction.get(metric)
            alternatives.add("two-sided" if value is None
                             else "greater" if value == "+"
                             else "less" if value == "-"
                             else "two-sided")
        else:
            # The hypothesis registers this metric (it is in the SESOI
            # table) but no sign for it: the row is an undirected readout.
            alternatives.add("two-sided")
        for key, value in direction.items():
            if not isinstance(key, str) or key == metric:
                continue
            if not key.endswith(f"_{metric}"):
                continue
            alternatives.add("two-sided" if value is None
                             else "greater" if value == "+"
                             else "less" if value == "-"
                             else "two-sided")
    return sorted(alternatives)


def _registered_sesoi_metrics(experiment: str) -> list:
    """Pre-registered numeric SESOIs for one experiment, de-duplicated.

    The planning table is derived from ``HYPOTHESES`` (the same hashed
    registry that carries the decision rules), not from a second hard-coded
    list: adding a registered effect automatically adds it to the design
    plan, and the plan cannot silently drift from the pre-registration.
    ``mode`` records whether the registered prediction is a superiority /
    directional test or a null (equivalence / non-inferiority) claim, since
    the latter needs a different power calculation and must not be read as
    an underpowered superiority test.  ``exact_alternatives`` lists every
    registered sign-flip reading of the metric (see
    :func:`_exact_alternatives_for`), so the prospective plan can check the
    exact-test resolution of the directional and two-sided rows together.
    """
    rows: dict[tuple, dict] = {}
    for hyp_id, spec in HYPOTHESES.items():
        if (spec.get("experiment") or "") != experiment:
            continue
        test_modes = spec.get("sesoi_test") or {}
        for metric, bound in (spec.get("sesoi") or {}).items():
            if not _finite(bound) or float(bound) <= 0.0:
                continue
            mode = test_modes.get(metric)
            if mode not in ("equivalence", "noninferior"):
                mode = "superiority"
            # Key by mode as well: the same display metric can carry a
            # directional claim in one hypothesis and a non-inferiority /
            # equivalence claim in another.  Collapsing them would plan one
            # of the two with the wrong z.
            key = (metric, float(bound), str(mode))
            row = rows.setdefault(key, {
                "metric": metric,
                "sesoi": float(bound),
                "mode": str(mode),
                "hypotheses": set(),
            })
            row["hypotheses"].add(hyp_id)
    out = []
    for key in sorted(rows):
        row = rows[key]
        row["hypotheses"] = sorted(row["hypotheses"])
        row["exact_alternatives"] = _exact_alternatives_for(
            row["metric"], row["mode"], row["hypotheses"])
        out.append(row)
    return out


def _pilot_mapping(value) -> dict:
    """A nested pilot payload, or an empty mapping when malformed."""
    return value if isinstance(value, dict) else {}


def _pilot_seed_accuracies(row) -> list[float]:
    """Usable per-seed accuracies from one pilot row, never raising."""
    if not isinstance(row, dict):
        return []
    by_seed = row.get("accuracy_by_seed")
    if not isinstance(by_seed, dict):
        return []
    return [float(v) for v in by_seed.values() if _finite(v)]


def _pilot_accuracy_sd(verdict) -> tuple[float, str | None]:
    """Seed-level accuracy SD from the baseline pilot (largest rung).

    The power of every accuracy contrast depends on the between-seed
    spread, which is unknown before the grid.  The baseline depth ladder
    already trains standard transformers at several depths on the same
    sealed stream, so its per-rung seed spread is an empirical, pre-grid
    pilot.  The **largest** rung SD is used deliberately: an optimistic
    pilot would certify an underpowered design, while the conservative
    maximum only asks for more seeds than strictly needed.

    Returns ``(nan, None)`` when the verdict carries no usable pilot;
    callers then report a ``no_pilot`` status instead of inventing a
    variance.
    """
    if not isinstance(verdict, dict):
        return float("nan"), None
    candidates: list[tuple[float, str]] = []
    ladder = _pilot_mapping(verdict.get("depth_ladder"))
    for depth, row in _pilot_mapping(ladder.get("by_depth")).items():
        if not isinstance(row, dict) or row.get("unmeasured"):
            continue
        values = _pilot_seed_accuracies(row)
        if len(values) >= 2:
            sd = _mean_std(values)["std"]
            if _finite(sd) and float(sd) >= 0.0:
                candidates.append((float(sd), f"depth_ladder.d{depth}"))
            continue
        sd = row.get("accuracy_std")
        if _finite(sd) and float(sd) >= 0.0:
            candidates.append((float(sd), f"depth_ladder.d{depth}.std"))
    shallow = _pilot_mapping(verdict.get("shallow_none"))
    shallow_values = _pilot_seed_accuracies(shallow)
    if len(shallow_values) >= 2:
        sd = _mean_std(shallow_values)["std"]
        if _finite(sd) and float(sd) >= 0.0:
            candidates.append((float(sd), "shallow_none"))
    elif _finite(shallow.get("accuracy_std")) \
            and float(shallow["accuracy_std"]) >= 0.0:
        candidates.append((float(shallow["accuracy_std"]),
                           "shallow_none.std"))
    if not candidates:
        return float("nan"), None
    sd, source = max(candidates, key=lambda item: item[0])
    return float(sd), source


def _pilot_accuracy_level(verdict) -> tuple[float, str | None]:
    """Pilot accuracy level used for the finite-stream precision floor.

    The prospective design plan also has to say how much of the SESOI is
    already consumed by scoring a *finite* evaluation stream.  That floor
    depends on the accuracy level (the binomial variance ``p(1-p)``), so
    this helper pools every per-seed pilot accuracy carried by the
    baseline verdict.  Pooling rungs is deliberately crude: the quantity
    only sets the worst-case binomial floor, and using all measured seeds
    avoids tuning the planning input to one lucky rung.

    Returns ``(nan, None)`` when the verdict carries no usable accuracy.
    """
    if not isinstance(verdict, dict):
        return float("nan"), None
    values: list[float] = []
    ladder = _pilot_mapping(verdict.get("depth_ladder"))
    for row in _pilot_mapping(ladder.get("by_depth")).values():
        if not isinstance(row, dict) or row.get("unmeasured"):
            continue
        row_values = _pilot_seed_accuracies(row)
        values.extend(row_values)
        if not row_values and _finite(row.get("accuracy_mean")):
            values.append(float(row["accuracy_mean"]))
    shallow = _pilot_mapping(verdict.get("shallow_none"))
    shallow_values = _pilot_seed_accuracies(shallow)
    values.extend(shallow_values)
    if not shallow_values and _finite(shallow.get("accuracy_mean")):
        values.append(float(shallow["accuracy_mean"]))
    if not values:
        return float("nan"), None
    mean = _mean_std(values)["mean"]
    if not _finite(mean):
        return float("nan"), None
    return float(mean), f"pooled {len(values)} pilot seed accuracies"


def _pilot_paired_accuracy_sd(verdict) -> tuple[float, str | None, list]:
    """Paired-difference seed SD for the prospective power plan.

    The registered accuracy contrasts are *paired* over training seeds, so
    their power depends on ``Var(A - B)``, not on the between-seed SD of
    one arm.  Using the single-arm SD (``_pilot_accuracy_sd``) silently
    assumes the two arms' seed effects are perfectly correlated, which
    makes the plan optimistic exactly when it matters.  This helper takes
    the largest available paired-variance estimate:

    * every adjacent-depth contrast in the baseline ladder is paired over
      the same seeds, and its percentile bootstrap already reports the SD
      of those paired differences (``adjacent_contrasts[*]["sd"]``); this
      is the only *empirical paired* pilot in the artifact;
    * ``sqrt(2) * max_rung_sd`` is the paired SD under independent arm
      effects (``Var(A-B) = Var(A) + Var(B)`` with equal arm variances),
      and is the fallback when no adjacent contrast is available.

    Taking the maximum is the same conservative choice as
    ``_pilot_accuracy_sd`` (largest rung): a pilot that happens to show
    small paired variation must not certify an underpowered design.  The
    returned components are recorded in the plan so the inflation is
    auditable rather than buried in one number.

    Returns ``(nan, None, [])`` when no usable pilot exists.
    """
    single, single_source = _pilot_accuracy_sd(verdict)
    components: list[dict] = []
    if _finite(single) and float(single) >= 0.0:
        components.append({
            "value": float(single),
            "source": single_source,
            "kind": "single_arm",
            "note": ("between-seed SD of one arm; a lower bound on the "
                     "paired SD only when the two arms are correlated"),
        })
        components.append({
            "value": math.sqrt(2.0) * float(single),
            "source": f"sqrt(2) * {single_source}",
            "kind": "independent_arms",
            "note": ("paired SD under independent arm effects, "
                     "Var(A-B)=Var(A)+Var(B) with equal arm variances"),
        })
    ladder = (_pilot_mapping(verdict.get("depth_ladder"))
              if isinstance(verdict, dict) else {})
    for name, row in _pilot_mapping(
            ladder.get("adjacent_contrasts")).items():
        if not isinstance(row, dict):
            continue
        n = _positive_integer(row.get("n"))
        if n is None or n < 2 or not _finite(row.get("sd")):
            continue
        sd = float(row["sd"])
        if sd < 0.0:
            continue
        components.append({
            "value": sd,
            "source": f"depth_ladder.adjacent[{name}]",
            "kind": "adjacent_paired",
            "note": ("observed SD of the same-seed adjacent-depth "
                     "difference; empirical paired-variance pilot"),
        })
    usable = [c for c in components if _finite(c["value"])
              and float(c["value"]) >= 0.0]
    if not usable:
        return float("nan"), None, []
    chosen = max(usable, key=lambda c: float(c["value"]))
    return (float(chosen["value"]), chosen["source"], components)


def _power_z(mode: str, alpha: float, power: float) -> float:
    """Multiplier ``z`` in ``n = (z * sd / bound)**2`` for a test mode.

    * ``superiority``: two-sided difference test at ``alpha``;
    * ``noninferior``: one-sided non-inferiority test at ``alpha``;
    * ``equivalence``: two one-sided tests (TOST) at ``alpha`` each, with
      the target power evaluated at a true difference of zero.

    The non-inferiority/equivalence modes exist because several registered
    accuracy claims are null/reverse checks, not superiority claims: planning
    them with the two-sided superiority formula would overstate their power.
    """
    if mode not in ("superiority", "noninferior", "equivalence") \
            or not (_finite(alpha) and _finite(power)) \
            or not (0.0 < float(alpha) < 1.0
                    and 0.0 < float(power) < 1.0):
        # An unknown test mode or an out-of-range planning probability is
        # not a design: return an unusable multiplier rather than silently
        # falling back to the two-sided superiority formula.
        return float("nan")
    nd = NormalDist()
    alpha = float(alpha)
    power = float(power)
    if mode == "equivalence":
        return nd.inv_cdf(1.0 - alpha) + nd.inv_cdf((1.0 + power) / 2.0)
    if mode == "noninferior":
        return nd.inv_cdf(1.0 - alpha) + nd.inv_cdf(power)
    return nd.inv_cdf(1.0 - alpha / 2.0) + nd.inv_cdf(power)


def _required_seeds(mode: str, sd, bound, alpha: float, power: float):
    """Normal-approximation seed count for one registered test mode."""
    if mode not in ("superiority", "noninferior", "equivalence"):
        return None
    if not (_finite(sd) and _finite(bound) and float(sd) > 0.0
            and float(bound) > 0.0):
        return None
    z = _power_z(mode, alpha, power)
    if not (_finite(z) and float(z) > 0.0):
        return None
    try:
        return float((z * float(sd) / float(bound)) ** 2)
    except OverflowError:
        # A finite but vanishingly small SESOI needs an unrepresentable
        # number of seeds; report "unattainable" (inf) rather than letting
        # the design plan crash on an OverflowError.
        return float("inf")


def _power_at_effect(mode: str, effect, sd, n: int, alpha: float,
                     bound=None) -> float:
    """Normal power at ``effect`` under the registered test mode.

    ``effect`` is the true difference used for the calculation.  For
    non-inferiority/equivalence rows the conservative planning point is a
    true difference of zero; for superiority rows it is the SESOI.  The
    null/reverse modes need the SESOI as their comparison ``bound``:
    non-inferiority rejects when the lower confidence limit clears
    ``-bound`` and TOST when the whole interval lies inside ``+-bound``.
    """
    if mode not in ("superiority", "noninferior", "equivalence"):
        return float("nan")
    n = _positive_integer(n)
    if n is None:
        # A bool or fractional unit count is a flag/label, not a number of
        # independent seeds; truncating it would silently plan a different
        # experiment than the caller named.
        return float("nan")
    needed = [_finite(effect), _finite(sd), _finite(alpha)]
    if mode in ("equivalence", "noninferior"):
        needed.append(_finite(bound))
    if not (all(needed) and float(sd) > 0.0
            and 0.0 < float(alpha) < 1.0):
        return float("nan")
    if mode in ("equivalence", "noninferior") and float(bound) <= 0.0:
        # A null-prediction power calculation is defined only for a
        # positive equivalence/non-inferiority bound.  A zero or negative
        # bound would make the "CI inside the bound" event impossible (or
        # invert it) while still returning an apparently usable number.
        return float("nan")
    se = float(sd) / math.sqrt(n)
    if se <= 0.0:
        return float("nan")
    nd = NormalDist()
    alpha = float(alpha)
    cutoff = nd.inv_cdf(1.0 - alpha)
    if mode == "equivalence":
        upper = (float(bound) - float(effect)) / se
        lower = (-float(bound) - float(effect)) / se
        power = nd.cdf(upper - cutoff) - nd.cdf(lower + cutoff)
    elif mode == "noninferior":
        power = nd.cdf((float(bound) + float(effect)) / se - cutoff)
    else:
        cutoff = nd.inv_cdf(1.0 - alpha / 2.0)
        shifted = float(effect) / se
        power = 1.0 - nd.cdf(cutoff - shifted) + nd.cdf(-cutoff - shifted)
    return min(1.0, max(0.0, float(power)))


class DesignPlanError(RuntimeError):
    """A pre-registered design plan would be revised after data collection."""


_DESIGN_PLAN_FINGERPRINT_KEYS = (
    "experiment", "planned_n", "planned_seeds", "alpha", "power_target",
    "fdr_q", "min_seeds_for_claim", "planned_eval_items",
    "hypotheses_sha256",
)


def _design_plan_fingerprint(payload, *,
                             include_fdr_q: bool = True) -> str | None:
    """Content hash of the *design-relevant* part of a plan.

    The plan file also carries invocation provenance (run id, environment,
    timestamps), which legitimately changes on every invocation.  The
    pre-registration is the planned sample, the registered SESOIs/modes,
    the multiplicity level and the claim floor, so only those fields enter
    this hash.  ``fdr_q`` joins them because the exact sign-flip resolution
    and the final supported/refuted readout are defined at that q: changing
    it after outcomes would be an adaptive goalpost move.  Plans written
    before ``fdr_q`` was stamped remain hashable (the field is treated as
    absent for them), and ``include_fdr_q=False`` lets the publisher
    compare such a plan with the current one on its original fields before
    accepting the migration.  A missing or malformed plan returns
    ``None``; callers must treat that as "cannot certify this design", not
    as "unchanged".
    """
    if not isinstance(payload, Mapping):
        return None
    # A partial or empty object is not a registered design: hashing the
    # handful of fields it happens to contain would let a malformed plan
    # (or the empty dict) masquerade as a valid pre-registration.  fdr_q is
    # optional for legacy plans (see the docstring), and this view omits it
    # entirely when comparing such a plan with the current design.
    keys = tuple(key for key in _DESIGN_PLAN_FINGERPRINT_KEYS
                 if include_fdr_q or key != "fdr_q")
    missing = [key for key in keys
               if key not in payload and key != "fdr_q"]
    if missing or not payload:
        return None
    raw_rows = payload.get("rows")
    if raw_rows is None:
        raw_rows = []
    if isinstance(raw_rows, (str, bytes, Mapping)):
        return None
    try:
        raw_rows = list(raw_rows)
    except TypeError:
        return None
    rows = []
    for row in raw_rows:
        if not isinstance(row, Mapping):
            return None
        hypotheses = row.get("hypotheses")
        if hypotheses is None:
            hypotheses = []
        if not isinstance(hypotheses, list):
            return None
        rows.append({
            "metric": row.get("metric"),
            "sesoi": row.get("sesoi"),
            "mode": row.get("mode"),
            "hypotheses": [str(h) for h in hypotheses],
        })
    seeds = payload.get("planned_seeds")
    if seeds is None:
        seeds = []
    if isinstance(seeds, (str, bytes, Mapping)):
        return None
    try:
        seeds = list(seeds)
    except TypeError:
        return None
    design = {
        key: payload.get(key) for key in keys
        if key != "rows"
        and (key != "fdr_q" or "fdr_q" in payload)
    }
    design["planned_seeds"] = seeds
    design["rows"] = rows
    try:
        encoded = json.dumps(design, ensure_ascii=False, sort_keys=True,
                             default=str)
    except (TypeError, ValueError, OverflowError):
        return None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _design_plan_outcomes_exist(out_dir: pathlib.Path, tag: str) -> bool:
    """Whether result evidence for this experiment is already on disk.

    The plan is prospective only while no outcome row exists.  A missing or
    unreadable outcome file is treated as "no outcome"; a non-empty run or
    contrast file is decisive, so a second invocation with a different
    design cannot silently turn a post-hoc analysis into a "planned" one.
    """
    for suffix in ("_runs.jsonl", "_contrasts.json"):
        path = pathlib.Path(out_dir) / f"{tag}{suffix}"
        try:
            if not path.exists() and not path.is_symlink():
                continue
            if not path.is_file():
                # The outcome path exists but is not a readable regular
                # file (e.g. a directory or a broken link).  Treat that as
                # evidence we cannot audit rather than silently planning
                # over it; publishing a "prospective" plan next to an
                # unreadable outcome path would be post-hoc by construction.
                return True
            if path.stat().st_size > 0:
                return True
        except OSError:
            # A path we cannot inspect may be evidence; fail closed.
            return True
    return False


def _record_design_plan_event(settings: Settings, out_dir: pathlib.Path,
                              tag: str, event: str, design_sha256: str,
                              supersedes: str | None = None) -> None:
    """Append one prep-registration event to an auditable history."""
    record = {
        "experiment": tag,
        "event": event,
        "design_sha256": design_sha256,
        "supersedes_design_sha256": supersedes,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **_stamp(settings),
    }
    _write_result(pathlib.Path(out_dir)
                  / f"{tag}_design_plan_history.jsonl", record)


def _publish_design_plan(settings: Settings, out_dir: pathlib.Path,
                         tag: str, payload: dict) -> dict:
    """Publish a prospective design plan, refusing post-outcome revisions.

    Before the first outcome row, a plan may be revised (preparation is
    still preparation); the revision is recorded in a history file and the
    superseded hash is stamped on the new plan.  Once a run or contrast file
    for this experiment is non-empty, the design hash is frozen: changing
    the planned seeds, SESOIs, test modes, FDR level or claim floor now
    would be an adaptive redesign, so it raises :class:`DesignPlanError`
    instead of silently overwriting the artifact that was supposed to
    pre-register the study.  Re-running an identical design is allowed.
    A plan written before ``fdr_q`` entered the fingerprint is accepted if
    all of its old fields match; the migration adds the current q to the
    artifact and records that it was registered after outcomes existed.
    """
    payload = dict(payload)
    path = pathlib.Path(out_dir) / f"{tag}_design_plan.json"
    new_hash = _design_plan_fingerprint(payload)
    if new_hash is None:
        raise DesignPlanError(
            f"{tag}: refusing to publish a design plan with no readable "
            f"design fingerprint")
    payload["design_sha256"] = new_hash
    if path.exists() and not path.is_file():
        # A directory/socket/FIFO at the plan path cannot be a registered
        # plan and cannot be atomically replaced with ``os.replace`` either.
        # Fail closed instead of leaking IsADirectoryError/OSError after the
        # design id has been attached to Settings.
        raise DesignPlanError(
            f"{tag}: the existing design-plan path {path} is not a regular "
            f"file, so it cannot be verified or replaced; restore the "
            f"registered plan or use a fresh --out directory")
    old = None
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            old = None
    outcomes = _design_plan_outcomes_exist(out_dir, tag)
    old_hash = _design_plan_fingerprint(old) if old is not None else None
    # Plans written before fdr_q was fingerprintable have no field to
    # compare against the current q.  Compare the old plan and the current
    # payload on their original fields; if those match, the only change is
    # the metadata upgrade, not a redesign, so an already-running study can
    # be rechecked.  The rewritten plan records the current q explicitly.
    legacy_hash = None
    if old is not None and "fdr_q" not in old:
        old_legacy = _design_plan_fingerprint(old, include_fdr_q=False)
        new_legacy = _design_plan_fingerprint(
            payload, include_fdr_q=False)
        if old_legacy is not None and old_legacy == new_legacy:
            legacy_hash = old_legacy
    if outcomes and not path.exists():
        raise DesignPlanError(
            f"{tag}: outcome rows already exist but no registered design "
            f"plan does; a plan written now would be post-hoc, so restore "
            f"the original plan or start the study in a fresh --out "
            f"directory")
    if path.exists() and outcomes:
        if old_hash is None:
            raise DesignPlanError(
                f"{tag}: an unreadable existing design plan cannot be "
                f"verified after outcome rows were written; restore the "
                f"registered plan or use a fresh --out directory")
        if old_hash != new_hash and legacy_hash is None:
            raise DesignPlanError(
                f"{tag}: refusing to revise the registered design "
                f"{old_hash} -> {new_hash} after outcome rows exist; a "
                f"changed sample size/SESOI/test mode is a post-hoc "
                f"redesign, so use a fresh --out directory and report it "
                f"as a separate study")
    if legacy_hash is not None and outcomes:
        print(f"[{tag}] WARNING: the registered design plan predates fdr_q; "
              f"q={float(settings.fdr_q):g} is being added to the frozen "
              f"record now, so it was not part of the original "
              f"pre-registration")
    # Link every subsequent run record (and the manifest) to the accepted
    # pre-registration.  Do this only after the freeze checks pass: a
    # refused revision must not leave a rejected design id on the Settings
    # object.  The context cache may have been built before the plan
    # existed, so invalidate it rather than leaving records that cannot be
    # joined back to their design.
    settings._design_sha256 = new_hash
    settings._run_context_cache = None
    settings._run_context_cache_key = None
    if path.exists() and old_hash is not None and old_hash != new_hash:
        # For a legacy plan the recomputed hash uses the new ``fdr_q: None``
        # sentinel, which never appeared in its history; point the event at
        # the id that was actually registered instead.
        superseded = old.get("design_sha256") if legacy_hash is not None \
            else old_hash
        if not isinstance(superseded, str):
            superseded = old_hash
        payload["supersedes_design_sha256"] = superseded
        _record_design_plan_event(
            settings, out_dir, tag, "amended", new_hash, superseded)
        if legacy_hash is not None:
            payload["fdr_q_registered_after_outcomes"] = bool(outcomes)
    elif not path.exists():
        _record_design_plan_event(
            settings, out_dir, tag, "created", new_hash)
    elif old_hash is None:
        _record_design_plan_event(
            settings, out_dir, tag, "replaced_malformed", new_hash)
    _write_json(path, payload)
    return payload


def _design_sensitivity_plan(settings: Settings, verdict, out_dir,
                             tag: str) -> dict:
    """Write the prospective sensitivity budget before the grid trains.

    The project already reports a *post-hoc* MDE80 next to every contrast.
    That is too late to change the design: if the planned seed count cannot
    resolve the pre-registered SESOI, the grid spends compute to produce an
    "inconclusive" that could have been predicted.  This plan translates
    each registered SESOI into the largest seed-level SD the design can
    tolerate at the configured alpha/power and, where an empirical pilot
    exists (accuracy from the baseline ladder), into the required seed
    count and the power at the planned N.  The artifact is written before
    the first grid run, so it is part of the preparation record rather than
    a rationalisation after the fact.

    It does **not** change the pre-registered decision rule: an
    underpowered plan is a warning that a null read is uninformative, not a
    licence to drop the contrast.  Power is also not claim eligibility: a
    design below ``min_seeds_for_claim`` is marked ineligible even when its
    sensitivity budget is met.  One further preparation criterion is
    independent of power: the exact paired sign-flip test can only report
    p-values down to a floor fixed by the seed count and the registered
    alternative.  A two-sided claim at five seeds (floor 0.0625) cannot be
    confirmed at q=0.05 even if the bootstrap CI is tight, so each metric
    row also reports the exact floor and the seed count required to clear
    it.  The published artifact is frozen once outcome rows exist (see
    :func:`_publish_design_plan`), so a later invocation cannot silently
    rewrite the study it is analysing.
    """
    # Re-validate at the boundary: the CLI and tests mutate Settings
    # attributes after construction, and a malformed seed set or claim floor
    # must not silently size the study (or crash halfway through the plan).
    settings.seeds = _validate_training_seeds(settings.seeds)
    n = len(settings.seeds)
    floor = _positive_integer(settings.min_seeds_for_claim)
    if floor is None:
        raise ValueError(
            f"min_seeds_for_claim must be a positive integer, got "
            f"{settings.min_seeds_for_claim!r}")
    meets_floor = n >= floor
    try:
        alpha = float(settings.design_alpha)
        target = float(settings.power_target)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            "design_alpha and power_target must be finite numbers strictly "
            f"between 0 and 1, got {settings.design_alpha!r} and "
            f"{settings.power_target!r}") from exc
    for name, value in (("design_alpha", alpha),
                        ("power_target", target)):
        if not (_finite(value) and 0.0 < value < 1.0):
            raise ValueError(
                f"{name} must be a finite number strictly between 0 and 1, "
                f"got {value!r}")
    # The FDR level is part of the decision rule, not just an analysis
    # switch: the exact sign-flip floor is compared with this q, and a
    # change of q after outcomes would move the goalposts.  Re-validate it
    # here because callers may mutate Settings after construction.
    try:
        fdr_q = float(settings.fdr_q)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            f"fdr_q must be a finite number strictly between 0 and 1, got "
            f"{settings.fdr_q!r}") from exc
    if not (_finite(fdr_q) and 0.0 < fdr_q < 1.0):
        raise ValueError(
            f"fdr_q must be a finite number strictly between 0 and 1, got "
            f"{settings.fdr_q!r}")
    # Two different pilot quantities answer two different questions:
    #   * the single-arm SD is the legacy, optimistic scale (it assumes
    #     the two arms' seed effects are perfectly correlated);
    #   * the paired SD is the scale of the registered paired contrast
    #     and is what the power calculation below must use.
    pilot_sd, pilot_source = _pilot_accuracy_sd(verdict)
    paired_sd, paired_source, paired_components = \
        _pilot_paired_accuracy_sd(verdict)
    pilot_acc, pilot_acc_source = _pilot_accuracy_level(verdict)
    planned_items = max(1, int(settings.eval_batch)
                        * int(settings.eval_batches))
    # Worst-case finite-stream floor: if the two arms' per-item errors were
    # independent draws, the paired difference on one item would have
    # variance p(1-p) + p(1-p).  This is deliberately a *worst case*: the
    # realized paired item variance is smaller when the arms share item
    # difficulty, and the post-hoc seed x item bootstrap measures it.  It
    # is reported as a planning companion, not added to the seed-level
    # variance (the seed-level differences already contain the realized
    # finite-stream noise of this stream).
    if _finite(pilot_acc):
        p_level = min(max(float(pilot_acc), 0.0), 1.0)
        stream_item_var = 2.0 * p_level * (1.0 - p_level)
        stream_item_sd = math.sqrt(stream_item_var / planned_items)
    else:
        p_level = float("nan")
        stream_item_var = float("nan")
        stream_item_sd = float("nan")
    rows = []
    for item in _registered_sesoi_metrics(tag):
        metric = item["metric"]
        bound = float(item["sesoi"])
        mode = item["mode"]
        z = _power_z(mode, alpha, target)
        max_sd = (bound * math.sqrt(max(n, 1)) / z
                  if _finite(z) and z > 0.0 else float("nan"))
        row = {
            "metric": metric,
            "sesoi": bound,
            "mode": mode,
            "hypotheses": item["hypotheses"],
            "z_for_mode": z,
            "max_sd_for_power": max_sd,
            "min_seeds_for_claim": floor,
            "claim_floor_met": meets_floor,
        }
        # Exact paired sign-flip resolution.  Power alone can certify a
        # design that the exact confirmation test cannot reject: at n=5 a
        # two-sided exact p is at best 0.0625, above the usual q=0.05, so
        # an undirected contrast is unfalsifiable no matter how large its
        # effect.  Plan every registered alternative of this
        # metric (see ``_exact_alternatives_for``), not only the pooled
        # main-effect sign, and report the required seed count before the
        # grid spends compute.
        exact_resolution = {}
        for alternative in item.get("exact_alternatives") or ["two-sided"]:
            resolution = sign_flip_resolution(n, alternative)
            required_exact = sign_flip_min_units(alternative, fdr_q)
            exact_resolution[alternative] = {
                "min_attainable_p": float(
                    resolution["min_attainable_p"]),
                "method": resolution["method"],
                "resolved_at_planned_n": bool(
                    resolution["valid"]
                    and float(resolution["min_attainable_p"]) <= fdr_q),
                "min_seeds": required_exact,
            }
        exact_unfalsifiable = sorted(
            alternative for alternative, info
            in exact_resolution.items()
            if not info["resolved_at_planned_n"])
        exact_required = [info["min_seeds"]
                          for info in exact_resolution.values()]
        exact_required_seeds = (
            max(exact_required)
            if exact_required and all(value is not None
                                      for value in exact_required)
            else None)
        row.update({
            "exact_alternatives": sorted(exact_resolution),
            "exact_resolution": exact_resolution,
            "exact_resolution_met": not exact_unfalsifiable,
            "exact_unfalsifiable_alternatives": exact_unfalsifiable,
            "exact_required_seeds": exact_required_seeds,
            "exact_seed_shortfall": (
                max(0, int(exact_required_seeds) - n)
                if exact_required_seeds is not None else None),
        })
        # The pilot is an accuracy-scale quantity; applying it to a metric
        # with different units would be a fabricated variance.
        accuracy_metric = "accuracy" in metric
        sd_single = pilot_sd if accuracy_metric else float("nan")
        sd = paired_sd if accuracy_metric else float("nan")
        source = pilot_source if accuracy_metric else None
        row.update({
            # ``pilot_sd`` stays the single-arm field for compatibility with
            # earlier design plans; the *decision-relevant* field is
            # ``pilot_paired_sd`` and is what the power calculation uses.
            "pilot_sd": float(sd_single),
            "pilot_source": source,
            "pilot_sd_single_arm": float(sd_single),
            "pilot_paired_sd": float(sd),
            "pilot_paired_source": paired_source if accuracy_metric else None,
            "pilot_paired_components": (
                list(paired_components) if accuracy_metric else []),
            "pilot_paired_inflation": (
                float(sd) / float(sd_single)
                if _finite(sd) and _finite(sd_single)
                and float(sd_single) > 0.0 else float("nan")),
            "stream_items": int(planned_items) if accuracy_metric else None,
            "stream_item_variance_floor": (
                float(stream_item_var) if accuracy_metric
                else float("nan")),
            "stream_item_sd_floor": (
                float(stream_item_sd) if accuracy_metric
                else float("nan")),
            "stream_items_for_sesoi": None,
            "stream_limited_worst_case": None,
        })
        # A finite evaluation stream imposes a precision floor that more
        # training seeds cannot remove.  Report the item count that would
        # keep the worst-case independent-arm binomial SE at or below the
        # SESOI's planning resolution.  This is a companion: the realized
        # paired item variance is measured post hoc by the crossed
        # bootstrap, and the pre-registered seed-level decision is
        # unchanged.
        if accuracy_metric and _finite(stream_item_var) \
                and _finite(z) and z > 0.0:
            target_var = (bound / z) ** 2
            if target_var > 0.0:
                items_needed = stream_item_var / target_var
                try:
                    items_for_sesoi = int(math.ceil(items_needed))
                except (OverflowError, ValueError):
                    # A SESOI so small that the required stream is not
                    # representable is unattainable, not a huge integer.
                    items_for_sesoi = None
                row["stream_items_for_sesoi"] = items_for_sesoi
                row["stream_limited_worst_case"] = bool(
                    _finite(items_needed) and items_needed > planned_items)
        if not _finite(sd):
            row.update({"required_seeds": None, "power_at_planned": None,
                        "powered": None, "status": "no_pilot"})
        elif float(sd) <= 0.0:
            row.update({"required_seeds": None, "power_at_planned": None,
                        "powered": None,
                        "status": "pilot_degenerate"})
        else:
            required = _required_seeds(mode, sd, bound, alpha, target)
            if _finite(required):
                required = int(math.ceil(required))
            else:
                required = None
            required_single = _required_seeds(
                mode, sd_single, bound, alpha, target)
            if _finite(required_single):
                required_single = int(math.ceil(required_single))
            else:
                required_single = None
            # The conservative planning point for a directional test is the
            # SESOI itself; for a null/reverse check it is a true
            # difference of zero.
            true_effect = 0.0 if mode in ("equivalence", "noninferior") \
                else bound
            power_at = _power_at_effect(
                mode, true_effect, sd, n, alpha, bound=bound)
            power_single = _power_at_effect(
                mode, true_effect, sd_single, n, alpha, bound=bound)
            powered = None if required is None else bool(required <= n)
            row.update({
                "required_seeds": required,
                "required_seeds_single_arm": required_single,
                "power_at_planned": power_at,
                "power_at_planned_single_arm": power_single,
                "powered": powered,
                "status": ("powered" if powered
                           else "underpowered" if powered is False
                           else "unpowerable"),
            })
        rows.append(row)
    accuracy_rows = [r for r in rows if "accuracy" in r["metric"]
                     and r["status"] in ("powered", "underpowered",
                                         "unpowerable", "pilot_degenerate")]
    underpowered = sorted({r["metric"] for r in accuracy_rows
                           if r["status"] == "underpowered"})
    unpowerable = sorted({r["metric"] for r in accuracy_rows
                          if r["status"] == "unpowerable"})
    stream_limited = sorted({r["metric"] for r in accuracy_rows
                             if r.get("stream_limited_worst_case")})
    exact_unfalsifiable = sorted(
        {r["metric"] for r in rows if not r["exact_resolution_met"]})
    exact_unfalsifiable_hypotheses = sorted(
        {hyp for r in rows if not r["exact_resolution_met"]
         for hyp in (r.get("hypotheses") or [])})
    accuracy_exact_unfalsifiable = sorted(
        {r["metric"] for r in accuracy_rows
         if not r["exact_resolution_met"]})
    exact_shortfall = [
        int(r["exact_seed_shortfall"]) for r in rows
        if r.get("exact_seed_shortfall") is not None]
    if not accuracy_rows:
        all_powered = None
    else:
        # ``unpowerable`` is not a powered plan: including it only in the
        # status list while treating "no underpowered rows" as success
        # would certify a design whose SESOI resolution is not even
        # representable.
        all_powered = all(r["status"] == "powered" for r in accuracy_rows)
    if not accuracy_rows:
        all_accuracy_exact_met = None
    else:
        all_accuracy_exact_met = all(
            r["exact_resolution_met"] for r in accuracy_rows)
    all_metric_exact_met = (
        all(r["exact_resolution_met"] for r in rows) if rows else None)
    # Power and claim eligibility are different questions.  A design can
    # clear its sensitivity budget at n=5 yet still be below the
    # pre-registered independent-seed floor; the contrast layer would mark
    # every row INELIGIBLE, so preparation must say that now rather than
    # let the run spend compute on a claim it cannot make.  Quick configs
    # (n=2) deliberately fail this check.
    seed_floor_blocked = ([] if meets_floor else
                          sorted({str(r["metric"]) for r in rows}))
    if not meets_floor:
        # The seed floor blocks a claim even when no accuracy pilot exists:
        # this is a property of the planned sample, not of the pilot.
        all_claim_seed_eligible = False
        all_claims_ready = False
    elif not accuracy_rows:
        all_claim_seed_eligible = None
        all_claims_ready = None
    else:
        all_claim_seed_eligible = True
        all_claims_ready = bool(all_powered)
    if not meets_floor:
        all_claims_confirmable = False
    elif not accuracy_rows:
        all_claims_confirmable = None
    else:
        # A claim needs statistical power *and* an exact test that can
        # actually reject at this n; either condition alone is not a
        # confirmable design.
        all_claims_confirmable = bool(all_powered
                                      and all_accuracy_exact_met)
    payload = {
        "experiment": tag,
        "planned_n": n,
        "planned_seeds": [int(s) for s in settings.seeds],
        "min_seeds_for_claim": floor,
        "planned_n_meets_claim_floor": meets_floor,
        "claim_floor_shortfall": max(0, floor - n),
        "claim_floor_blocked_metrics": seed_floor_blocked,
        "all_accuracy_claims_seed_eligible": all_claim_seed_eligible,
        "all_accuracy_claims_ready": all_claims_ready,
        "all_accuracy_claims_confirmable": all_claims_confirmable,
        "alpha": alpha,
        "power_target": target,
        # FDR level is part of the hashed design (see
        # ``_design_plan_fingerprint``), because the exact-test floor is
        # compared with it and changing q after outcomes would be an
        # adaptive decision-rule change.
        "fdr_q": fdr_q,
        "min_seeds_for_exact_one_sided": sign_flip_min_units(
            "greater", fdr_q),
        "min_seeds_for_exact_two_sided": sign_flip_min_units(
            "two-sided", fdr_q),
        "exact_unfalsifiable_metrics": exact_unfalsifiable,
        "exact_unfalsifiable_hypotheses": exact_unfalsifiable_hypotheses,
        "accuracy_exact_unfalsifiable": accuracy_exact_unfalsifiable,
        "all_accuracy_exact_resolutions_met": all_accuracy_exact_met,
        "all_metric_exact_resolutions_met": all_metric_exact_met,
        "max_exact_seed_shortfall": (
            max(exact_shortfall) if exact_shortfall else 0),
        # Legacy single-arm field (kept for readers of older plans) and the
        # paired field actually used by the power calculation.
        "pilot_accuracy_sd": float(pilot_sd),
        "pilot_accuracy_source": pilot_source,
        "pilot_paired_accuracy_sd": float(paired_sd),
        "pilot_paired_source": paired_source,
        "pilot_paired_components": paired_components,
        "pilot_accuracy_level": float(p_level),
        "pilot_accuracy_level_source": pilot_acc_source,
        "planned_eval_items": int(planned_items),
        "stream_item_variance_floor": float(stream_item_var),
        "stream_item_sd_floor": float(stream_item_sd),
        "paired_sd_formula": (
            "max(observed SD of same-seed adjacent-depth differences, "
            "sqrt(2) * largest single-arm rung SD); the single-arm SD alone "
            "assumes perfectly correlated arm effects and is optimistic"),
        "max_sd_formula": (
            "SESOI * sqrt(planned_n) / z(mode, alpha, power); "
            "z is two-sided for superiority, one-sided for noninferiority, "
            "and TOST for equivalence"),
        "rows": rows,
        "accuracy_underpowered": underpowered,
        "accuracy_unpowerable": unpowerable,
        "accuracy_stream_limited_worst_case": stream_limited,
        "all_accuracy_plans_powered": all_powered,
        "note": (
            "Prospective sensitivity budget from the pre-registered SESOIs "
            "and the baseline-ladder pilot variance.  Accuracy contrasts "
            "are paired over seeds, so their power is planned on the paired "
            "difference SD (largest empirical adjacent-depth paired SD or "
            "the independent-arm sqrt(2) bound), not on the single-arm SD; "
            "using the latter would certify an underpowered design whenever "
            "the two arms are not perfectly correlated.  A metric with no "
            "pilot (non-accuracy mechanism readout) reports only the "
            "largest SD the planned N can tolerate; it is not certified.  "
            "The finite-stream item floor is a separate worst-case "
            "companion: it assumes independent arm errors, and more seeds "
            "cannot remove it, so a stream-limited warning calls for a "
            "wider evaluation stream (the post-hoc seed x item bootstrap "
            "measures the realized paired item variance).  The exact "
            "sign-flip resolution is planned per registered alternative: "
            "at n=5 a two-sided exact p-value cannot fall below 0.0625, so "
            "an undirected row is unfalsifiable at q=0.05 however large its "
            "effect; the plan reports the required seed count rather than "
            "discovering this after the grid.  An underpowered row means a "
            "null result is uninformative at this N, not that the effect is "
            "absent."),
        "hypotheses_sha256": _hypotheses_sha(),
        **_stamp(settings),
    }
    payload = _publish_design_plan(settings, out_dir, tag, payload)
    print(f"\n[{tag}] prospective sensitivity plan (written before the grid; "
          f"alpha={alpha:g} per registered test mode, power={target:g}, "
          f"n={n} seeds)")
    if _finite(paired_sd):
        single_s = f"{pilot_sd:.4f}" if _finite(pilot_sd) else "n/a"
        print(f"  accuracy pilot: single-arm SD={single_s} "
              f"({pilot_source or 'n/a'}); paired SD={paired_sd:.4f} "
              f"({paired_source or 'n/a'})")
        print("  the power plan uses the paired SD: the single-arm SD "
              "assumes perfectly correlated arm effects and would certify "
              "an underpowered design when the arms are independent")
    else:
        print("  accuracy pilot unavailable: the design plan reports the "
              "tolerable SD but cannot certify power; run `--only "
              "baselines` first")
    if _finite(stream_item_sd):
        print(f"  finite-stream floor (worst case, independent arm errors): "
              f"items={planned_items}, p={p_level:.3f}, SE="
              f"{stream_item_sd:.4f}")
    if not meets_floor:
        print(f"  WARNING: planned n={n} is below the pre-registered "
              f"min_seeds_for_claim={floor}; every contrast will be "
              f"INELIGIBLE for a claim at this sample size "
              f"(shortfall={max(0, floor - n)} seed(s)).  Use this plan "
              f"for a pipeline check, not for reported evidence.")
    print(f"  {'metric':>34} {'mode':>12} {'SESOI':>8} {'maxSD':>8} "
          f"{'armSD':>8} {'pairSD':>8} {'need_n':>7} {'power':>7} "
          f"{'status':>16}")
    for row in rows:
        required = row["required_seeds"]
        need = "-" if required is None else str(required)
        power = row["power_at_planned"]
        power_s = f"{power:.2f}" if _finite(power) else "-"
        arm_sd = (f"{row['pilot_sd']:.3f}"
                  if _finite(row["pilot_sd"]) else "-")
        pair_sd = (f"{row['pilot_paired_sd']:.3f}"
                   if _finite(row["pilot_paired_sd"]) else "-")
        print(f"  {row['metric']:>34} {row['mode']:>12} "
              f"{row['sesoi']:>8.4g} "
              f"{row['max_sd_for_power']:>8.4g} {arm_sd:>8} "
              f"{pair_sd:>8} {need:>7} {power_s:>7} {row['status']:>16}")
    if underpowered or unpowerable:
        names = ", ".join(underpowered + unpowerable)
        print(f"  WARNING: planned n={n} does not clear the paired-variance "
              f"requirement for {names}: add seeds or pre-register a larger "
              f"SESOI before claiming a null; the grid still runs under the "
              f"pre-registered rule")
    elif accuracy_rows:
        print("  accuracy contrasts: the planned seed count clears the "
              "paired-variance requirement at the registered SESOI")
    if stream_limited:
        print(f"  WARNING: the planned evaluation stream "
              f"({planned_items} items) is below the worst-case "
              f"independent-arm item requirement for "
              f"{', '.join(stream_limited)}; more training seeds cannot "
              f"remove this finite-stream floor, so widen the evaluation "
              f"stream (eval_batch x eval_batches) if the realized "
              f"seed x item bootstrap also shows item variance dominating")
    one_sided_floor = sign_flip_resolution(n, "greater")[
        "min_attainable_p"]
    two_sided_floor = sign_flip_resolution(n, "two-sided")[
        "min_attainable_p"]
    print(f"  exact sign-flip floor at n={n} (q={fdr_q:g}): "
          f"one-sided={one_sided_floor:.5g}, "
          f"two-sided={two_sided_floor:.5g}")
    if exact_unfalsifiable:
        details = []
        for metric in exact_unfalsifiable:
            entries = [r for r in rows if r["metric"] == metric]
            alternatives = sorted({
                alternative for r in entries
                for alternative in r["exact_unfalsifiable_alternatives"]})
            required = [r.get("exact_required_seeds") for r in entries]
            required_s = ("/".join(str(v) for v in sorted(set(required))
                                   if v is not None) or "n/a")
            details.append(f"{metric} ({'/'.join(alternatives)}; "
                           f"need>={required_s})")
        print(f"  WARNING: at n={n} the exact paired sign-flip test cannot "
              f"reach q={fdr_q:g} for {', '.join(details)}; the bootstrap "
              f"may appear significant while the design cannot confirm it. "
              f"Add seeds before claiming these rows (or register a "
              f"direction where the hypothesis is directional)")
    elif rows:
        print("  exact sign-flip resolution: every registered alternative "
              "is falsifiable at the planned seed count")
    return payload


def _recipe_view(run):
    """The controlled part of a run's configuration (a flat dict).

    Besides the resolved ``Settings`` snapshot this includes the stamped
    ``training_recipe`` block (optimizer family, weight decay, betas,
    schedule, eta_min, gradient clipping, augmentation) when the record
    carries it, so a contrast checks the *actual* optimisation and
    augmentation recipe - not only the learning rate - for equality across
    the two sides.  The stamped evaluation-stream seeds enter the view as
    well: equal stream *sizes* do not prove equal streams, and a metric
    read from a different stream is an evaluation-protocol difference the
    audit must refuse exactly like a recipe difference.  The stamped
    ``data_fingerprints`` (the content hash of every training/validation/
    test/statistics stream) enter the view for the same reason one level
    deeper: the same *seeds* can still generate a different stream if the
    batch geometry or generator path changed, and then the two sides did
    not train or read the same data at all.  Finally, the *executed* step
    count / early-stop state is included: the intended step count in
    ``training_recipe.steps`` does not prove both sides actually ran it.
    """
    cfg = run.get("config")
    if not isinstance(cfg, Mapping):
        # ``_controlled_audit`` records this as a mismatch; return an empty
        # view so the audit still reports every other difference instead of
        # raising on the malformed object.
        cfg = {}
    view = {k: cfg.get(k) for k in _AUDIT_RECIPE_KEYS}
    view["lr"] = run.get("lr")
    view["eval_seed"] = run.get("eval_seed")
    view["stats_seed"] = run.get("stats_seed")
    # The intended step count lives in ``training_recipe.steps``; these
    # stamped fields are what the run *actually* executed.  Auditing them
    # keeps "same epochs" true even if early stopping is ever enabled, and
    # makes a short/diverged run visible as a confound rather than as a
    # silent extra degree of freedom.
    view["steps_run"] = run.get("steps_run")
    view["stopped_early"] = run.get("stopped_early")
    view["restored_best"] = run.get("restored_best")
    # The data-stream fingerprints ride with the run context; a record that
    # predates the stamp compares as ``None == None`` and is skipped, while
    # stamped records now prove "same seed, same stream" - per training
    # seed (the ``train`` sub-map is keyed by seed, so a stray Settings
    # whose stream geometry changed cannot masquerade as the paired run).
    view["data_fingerprints"] = run.get("data_fingerprints")
    recipe = run.get("training_recipe")
    if isinstance(recipe, dict):
        for key, val in recipe.items():
            view[f"training_recipe.{key}"] = val
    return view
