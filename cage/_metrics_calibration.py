# cage metrics: prospective calibration of the pre-registered decision rule.
# Executed into the wrapper module namespace by the loader;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541

_CALIBRATION_RULES = ("bootstrap", "sign_flip")
_CALIBRATION_DIRECTIONS = ("greater", "less", "two-sided")
_CALIBRATION_DEFAULT_FAMILY_SIZES = (1, 2, 4, 8, 16, 32)
_CALIBRATION_CHUNK_ROWS = 4
_CALIBRATION_CACHE: dict = {}

_CALIBRATION_ASSUMPTIONS = (
    "the paired seed differences are symmetric around zero under the null "
    "(the null the exact sign-flip test itself assumes);",
    "the null spread is the pilot paired-difference SD of the accuracy "
    "ladder, so the calibration lives on the same scale as the power "
    "plan;",
    "contrasts inside one family reuse the same training seeds and one "
    "sealed evaluation stream, so their per-seed noise is positively "
    "correlated (the correlation parameter);",
    "the family is m difference rows read through the pre-registered "
    "decision chain: p-value -> Benjamini-Hochberg at q -> registered "
    "direction -> |effect| >= SESOI -> seed floor;",
    "the bootstrap p-value is the same estimator as paired_bootstrap_ci "
    "(two-sided percentile bootstrap with +1 smoothing) but draws its "
    "resamples in one batch, so only Monte-Carlo noise separates it from "
    "the analysis layer;",
    "the exact sign-flip companion uses the row's registered alternative "
    "(one-sided for a directional row, two-sided otherwise), while the "
    "bootstrap p-value is two-sided for every row, exactly as in the "
    "analysis layer.",
)


def _calibration_direction(value) -> str:
    """Normalise a registered direction into the calibration's vocabulary."""
    text = str(value).strip().lower().replace("_", "-")
    if text in ("greater", "up", "+", "positive", "superior"):
        return "greater"
    if text in ("less", "down", "-", "negative", "inferior"):
        return "less"
    return "two-sided"


def _calibration_generator(seed: int) -> torch.Generator:
    """A CPU generator for the calibration's own resampling streams."""
    return torch.Generator(device="cpu").manual_seed(_bootstrap_seed(seed))


def _calibration_values(n_units: int, n_rows: int, n_replicates: int,
                        sd: float, correlation: float, effect,
                        generator: torch.Generator) -> torch.Tensor:
    """Paired-difference samples for one calibration scenario.

    The rows of a contrast family share training seeds and one sealed
    evaluation stream, so their per-seed noise is positively correlated:
    a common seed-level factor is shared by every row of a replicate and
    an idiosyncratic factor is drawn per row.  ``correlation`` is the
    share of the variance carried by the common factor, i.e. the
    correlation of the row means inside one replicate.  The target row
    (row 0) receives ``effect`` as a constant shift, which is the
    alternative scenario; the null scenario leaves every row centred.
    """
    common = torch.randn((n_replicates, n_units), generator=generator,
                         dtype=torch.float64)
    specific = torch.randn((n_replicates, n_rows, n_units),
                           generator=generator, dtype=torch.float64)
    weight_common = math.sqrt(correlation)
    weight_specific = math.sqrt(1.0 - correlation)
    values = sd * (weight_common * common.unsqueeze(1)
                   + weight_specific * specific)
    if effect is not None and float(effect) != 0.0:
        values[:, 0, :] = values[:, 0, :] + float(effect)
    return values


def _calibration_bootstrap_p(values: torch.Tensor, n_boot: int,
                             generator: torch.Generator) -> torch.Tensor:
    """Batched mirror of ``paired_bootstrap_ci``'s two-sided p-value.

    The estimator is copied exactly - resample ``n_boot`` means with
    replacement, count the fraction on either side of zero and report
    ``2 * (min(p_lo, p_hi) * n_boot + 1) / (n_boot + 1)`` capped at one -
    so the calibration measures the decision layer the grids actually
    use.  Only the resampling stream differs: the analysis layer draws
    one ``(n_boot, n)`` index matrix per contrast, the calibration draws
    one batched matrix for the whole family, which is what makes a
    family-level operating-characteristic study affordable.
    """
    replicates, n_rows, n_units = values.shape
    out = torch.empty((replicates, n_rows), dtype=torch.float64)
    for start in range(0, n_rows, _CALIBRATION_CHUNK_ROWS):
        stop = min(n_rows, start + _CALIBRATION_CHUNK_ROWS)
        block = values[:, start:stop, :]
        width = stop - start
        idx = torch.randint(0, n_units,
                            (replicates, width, n_boot, n_units),
                            generator=generator, dtype=torch.int64)
        expanded = block.unsqueeze(2).expand(replicates, width, n_boot, n_units)
        means = torch.gather(expanded, 3, idx).mean(dim=3)
        p_lo = (means <= 0.0).to(torch.float64).mean(dim=2)
        p_hi = (means >= 0.0).to(torch.float64).mean(dim=2)
        out[:, start:stop] = torch.clamp(
            2.0 * (torch.minimum(p_lo, p_hi) * n_boot + 1.0)
            / (n_boot + 1.0), max=1.0)
    return out


def _calibration_sign_flip_p(values: torch.Tensor, n_random: int,
                             generator: torch.Generator,
                             max_exact: int) -> dict:
    """Batched mirror of ``sign_flip_test``'s three alternative p-values.

    Below ``max_exact`` units every sign assignment is enumerated (the
    exact test); above it the sign assignments are sampled and the
    counts are smoothed with ``+1`` as in the analysis layer.  The
    counting convention is the analysis layer's: ``greater`` counts
    assignments at or above the observed mean, ``less`` counts
    assignments at or below it and ``two-sided`` counts assignments
    whose absolute mean reaches the observed absolute mean, each with a
    scale-relative tie tolerance.
    """
    replicates, n_rows, n_units = values.shape
    keys = ("greater", "less", "two-sided")
    out = {key: torch.empty((replicates, n_rows), dtype=torch.float64)
           for key in keys}
    observed = values.mean(dim=2)
    if n_units <= max_exact:
        total = 1 << n_units
        assignments = torch.arange(total, dtype=torch.int64)
        shifts = torch.arange(n_units, dtype=torch.int64)
        bits = (assignments.unsqueeze(1) >> shifts) & 1
        signs = bits.to(torch.float64) * 2.0 - 1.0
        for start in range(0, n_rows, _CALIBRATION_CHUNK_ROWS):
            stop = min(n_rows, start + _CALIBRATION_CHUNK_ROWS)
            block = values[:, start:stop, :]
            obs = observed[:, start:stop]
            means = torch.einsum("an,rcn->arc", signs, block) / float(n_units)
            scale = torch.maximum(obs.abs(), block.abs().amax(dim=2))
            tol = (1e-12 * scale).unsqueeze(0)
            out["greater"][:, start:stop] = (
                (means >= (obs - tol)).sum(dim=0).to(torch.float64)
                / float(total))
            out["less"][:, start:stop] = (
                (means <= (obs + tol)).sum(dim=0).to(torch.float64)
                / float(total))
            out["two-sided"][:, start:stop] = (
                (means.abs() >= (obs.abs() - tol)).sum(dim=0).to(torch.float64)
                / float(total))
    else:
        signs = torch.randint(0, 2, (n_random, n_units),
                              generator=generator, dtype=torch.int64)
        signs = signs.to(torch.float64) * 2.0 - 1.0
        for start in range(0, n_rows, _CALIBRATION_CHUNK_ROWS):
            stop = min(n_rows, start + _CALIBRATION_CHUNK_ROWS)
            block = values[:, start:stop, :]
            obs = observed[:, start:stop]
            means = torch.einsum("an,rcn->arc", signs, block) / float(n_units)
            scale = torch.maximum(obs.abs(), block.abs().amax(dim=2))
            tol = (1e-12 * scale).unsqueeze(0)
            out["greater"][:, start:stop] = (
                (means >= (obs - tol)).sum(dim=0).to(torch.float64) + 1.0
            ) / float(n_random + 1)
            out["less"][:, start:stop] = (
                (means <= (obs + tol)).sum(dim=0).to(torch.float64) + 1.0
            ) / float(n_random + 1)
            out["two-sided"][:, start:stop] = (
                (means.abs() >= (obs.abs() - tol)).sum(dim=0).to(torch.float64)
                + 1.0) / float(n_random + 1)
    return out


def _calibration_bh_reject(p_values: torch.Tensor, q: float) -> torch.Tensor:
    """Batched mirror of ``benjamini_hochberg``'s reject decisions.

    The step-up adjusted p-values are reproduced exactly: sort the
    family, scale each p-value by ``m / rank``, take the running minimum
    from the largest rank downwards and cap at one; a row is rejected
    when its adjusted p-value is at most ``q``.  Ties are broken by the
    row's position, as in the analysis layer.
    """
    replicates, m = p_values.shape
    order = torch.argsort(p_values, dim=1, stable=True)
    ordered = torch.gather(p_values, 1, order)
    ranks = torch.arange(1, m + 1, dtype=torch.float64).unsqueeze(0)
    ratios = ordered * float(m) / ranks
    suffix = torch.flip(
        torch.cummin(torch.flip(ratios, dims=[1]), dim=1).values, dims=[1])
    adjusted = torch.clamp(suffix, max=1.0)
    reject_ordered = adjusted <= float(q)
    reject = torch.zeros_like(reject_ordered)
    reject.scatter_(1, order, reject_ordered)
    return reject


def _calibration_claims(values: torch.Tensor, reject: torch.Tensor,
                        direction: str, sesoi, min_seeds: int,
                        n_units: int) -> torch.Tensor:
    """Rows that satisfy the pre-registered *claim* gate.

    A row supports a pre-registered decision when it survives the BH
    family, points in the registered direction, is practically large
    enough (``|effect| >= SESOI``) and comes from enough independent
    seeds.  This mirrors ``ContrastLog.finalize``: the sign gate and the
    magnitude gate are applied after the multiplicity correction, and
    the seed floor refuses a claim outright.
    """
    mean = values.mean(dim=2)
    if direction == "greater":
        sign_ok = mean > 0.0
    elif direction == "less":
        sign_ok = mean < 0.0
    else:
        sign_ok = torch.ones_like(mean, dtype=torch.bool)
    if sesoi is None:
        magnitude_ok = torch.ones_like(mean, dtype=torch.bool)
    else:
        magnitude_ok = mean.abs() >= float(sesoi)
    eligible = torch.full_like(mean, bool(n_units >= int(min_seeds)),
                               dtype=torch.bool)
    return reject & sign_ok & magnitude_ok & eligible


def _calibration_family_report(reject: torch.Tensor, claims: torch.Tensor,
                                target, m: int) -> dict:
    """Operating characteristics of one family size inside one scenario.

    ``reject`` and ``claims`` are per-row boolean matrices of shape
    ``(replicates, family)``; ``target`` marks the rows that carry a true
    effect (``None`` when the scenario has no true effect, e.g. the null
    scenario or an alternative scenario without a shifted row).  The
    false-discovery rate is the usual ``E[V] / E[max(1, R)]`` with ``V``
    the false discoveries and ``R`` the total discoveries, so a family
    that never rejects contributes zero rather than an undefined ratio.
    """
    selected = reject[:, :m]
    supported = claims[:, :m]
    if target is None:
        # No row carries a true effect: every selection and every claim
        # is a false discovery.
        true_rows = torch.zeros_like(selected)
    else:
        true_rows = target[:, :m]
    false_selected = selected & ~true_rows
    false_supported = supported & ~true_rows
    n_selected = selected.sum(dim=1).to(torch.float64)
    n_false_selected = false_selected.sum(dim=1).to(torch.float64)
    n_supported = supported.sum(dim=1).to(torch.float64)
    n_false_supported = false_supported.sum(dim=1).to(torch.float64)
    denom_selected = torch.clamp(n_selected, min=1.0)
    denom_supported = torch.clamp(n_supported, min=1.0)
    target_claim = None
    if target is not None:
        # The power of the claim gate on the row that carries the effect.
        # Without a target row there is nothing to report.
        target_claim = float(supported[:, 0].to(torch.float64).mean())
    return {
        "family_size": int(m),
        "rejections_mean": float(n_selected.mean()),
        "any_rejection_rate": float((n_selected > 0).to(torch.float64).mean()),
        "rejection_fdr": float((n_false_selected.sum() / denom_selected.sum())),
        "claims_mean": float(n_supported.mean()),
        "any_claim_rate": float((n_supported > 0).to(torch.float64).mean()),
        "claim_fdr": float((n_false_supported.sum() / denom_supported.sum())),
        "expected_false_claims": float(n_false_supported.mean()),
        "target_claim_rate": target_claim,
    }


def _calibration_scenario(values: torch.Tensor, rule: str, q: float,
                          direction: str, sesoi, min_seeds: int,
                          family_sizes, n_boot: int, n_random: int,
                          max_exact: int, seed: int,
                          target) -> dict:
    """Operating characteristics of one scenario (null or alternative).

    ``target`` marks the rows that carry a true effect; it is ``None``
    when the scenario has none (the null scenario, or an alternative
    scenario whose row was not shifted), and then every rejection and
    every claim is a false discovery.
    """
    replicates, n_rows, n_units = values.shape
    generator = _calibration_generator(seed)
    if rule == "bootstrap":
        p_values = _calibration_bootstrap_p(values, n_boot, generator)
    else:
        flips = _calibration_sign_flip_p(values, n_random, generator, max_exact)
        p_values = flips[direction]
    reject = _calibration_bh_reject(p_values, q)
    claims = _calibration_claims(values, reject, direction, sesoi, min_seeds,
                                 n_units)
    return {
        "per_row_rejection_rate": float(reject.to(torch.float64).mean()),
        "per_row_claim_rate": float(claims.to(torch.float64).mean()),
        "families": {
            str(m): _calibration_family_report(reject, claims, target, m)
            for m in family_sizes
        },
    }


def decision_rule_calibration(
        n_units: int,
        *,
        family_sizes=_CALIBRATION_DEFAULT_FAMILY_SIZES,
        sd: float = 1.0,
        sesoi=None,
        effect=None,
        q: float = 0.05,
        direction: str = "greater",
        n_replicates: int = 200,
        n_boot: int = 2000,
        n_random: int = 2000,
        correlation: float = 0.5,
        seed: int = 0,
        rules=_CALIBRATION_RULES,
        min_seeds: int = 1,
        max_exact: int = 12) -> dict:
    """Operating characteristics of the pre-registered decision rule.

    The design plan answers "how many seeds does the registered SESOI
    need?" with a normal approximation that treats one contrast at a
    time.  The grids, however, decide with a chain: a percentile
    bootstrap p-value, a Benjamini-Hochberg family at ``q``, a
    registered direction, a practical ``SESOI`` gate and a seed floor.
    None of those five steps is covered by the per-contrast power
    formula, and at the handful of seeds this project can afford the
    percentile bootstrap p-value is not even nominal: its null
    rejection rate at five seeds is several times the level it is
    compared against.  This helper measures the chain itself.

    It simulates the registered pipeline on paired seed differences
    under a symmetric null at the pilot SD (and again with one row
    shifted by ``effect``) and reports, per rule and per family size:
    the per-row null rejection rate, the family false-discovery rate of
    the multiplicity step, the false-discovery rate of the full claim
    gate, and the power of the claim gate at a true effect of
    ``effect``.  The ``bootstrap`` rule is the pre-registered decision
    and the ``sign_flip`` rule is the exact companion the analysis layer
    already reports; calibrating both makes the gap between them a
    design-time number instead of a post-hoc surprise.

    Malformed arguments abstain with ``valid=False`` and a reason
    rather than raising, matching the rest of the inference layer.
    """
    out = {
        "valid": False,
        "reason": "",
        "n_units": None,
        "family_sizes": [],
        "sd": float("nan"),
        "sesoi": None,
        "effect": None,
        "q": float("nan"),
        "direction": direction,
        "correlation": float("nan"),
        "n_replicates": None,
        "n_boot": None,
        "n_random": None,
        "min_seeds": None,
        "max_exact": None,
        "seed": None,
        "rules": {},
        "assumptions": list(_CALIBRATION_ASSUMPTIONS),
    }
    n = _positive_integer(n_units)
    if n is None:
        out["reason"] = "n_units must be a positive integer"
        return out
    replicates = _positive_integer(n_replicates)
    if replicates is None:
        out["reason"] = "n_replicates must be a positive integer"
        return out
    boots = _positive_integer(n_boot)
    if boots is None:
        out["reason"] = "n_boot must be a positive integer"
        return out
    randoms = _positive_integer(n_random)
    if randoms is None:
        out["reason"] = "n_random must be a positive integer"
        return out
    exact = _positive_integer(max_exact)
    if exact is None:
        out["reason"] = "max_exact must be a positive integer"
        return out
    floor = _positive_integer(min_seeds)
    if floor is None:
        out["reason"] = "min_seeds must be a positive integer"
        return out
    try:
        sd_value = float(sd)
        q_value = float(q)
        rho = float(correlation)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        out["reason"] = "sd, q and correlation must be finite numbers"
        return out
    if not (math.isfinite(sd_value) and sd_value > 0.0):
        out["reason"] = "sd must be a finite positive number"
        return out
    if not (math.isfinite(q_value) and 0.0 < q_value < 1.0):
        out["reason"] = "q must be a finite number strictly between 0 and 1"
        return out
    if not (math.isfinite(rho) and 0.0 <= rho <= 1.0):
        out["reason"] = "correlation must be a finite number in [0, 1]"
        return out
    sesoi_value = None
    if sesoi is not None:
        try:
            sesoi_value = float(sesoi)
        except (TypeError, ValueError, RuntimeError, OverflowError):
            out["reason"] = "sesoi must be a finite non-negative number"
            return out
        if not (math.isfinite(sesoi_value) and sesoi_value >= 0.0):
            out["reason"] = "sesoi must be a finite non-negative number"
            return out
    effect_value = None
    if effect is not None:
        try:
            effect_value = float(effect)
        except (TypeError, ValueError, RuntimeError, OverflowError):
            out["reason"] = "effect must be a finite number"
            return out
        if not math.isfinite(effect_value):
            out["reason"] = "effect must be a finite number"
            return out
    alt = _calibration_direction(direction)
    sizes = []
    if isinstance(family_sizes, (str, bytes)) or family_sizes is None:
        out["reason"] = "family_sizes must be a sequence of positive integers"
        return out
    try:
        requested = list(family_sizes)
    except TypeError:
        out["reason"] = "family_sizes must be a sequence of positive integers"
        return out
    for size in requested:
        parsed = _positive_integer(size)
        if parsed is None:
            out["reason"] = "family_sizes must be positive integers"
            return out
        sizes.append(int(parsed))
    if not sizes:
        out["reason"] = "family_sizes must not be empty"
        return out
    sizes = sorted(set(sizes))
    selected_rules = []
    if isinstance(rules, (str, bytes)) or rules is None:
        out["reason"] = "rules must be a sequence of rule names"
        return out
    try:
        requested_rules = list(rules)
    except TypeError:
        out["reason"] = "rules must be a sequence of rule names"
        return out
    for rule in requested_rules:
        name = str(rule).strip().lower()
        if name not in _CALIBRATION_RULES:
            out["reason"] = f"unknown calibration rule {rule!r}"
            return out
        selected_rules.append(name)
    if not selected_rules:
        out["reason"] = "rules must not be empty"
        return out
    selected_rules = sorted(set(selected_rules))
    try:
        seed_int = int(operator.index(seed))
    except TypeError:
        out["reason"] = "seed must be an integer"
        return out
    key = (n, tuple(sizes), sd_value, sesoi_value, effect_value, q_value, alt,
           replicates, boots, randoms, rho, seed_int, tuple(selected_rules),
           floor, exact)
    cached = _CALIBRATION_CACHE.get(key)
    if cached is not None:
        return cached
    max_rows = max(sizes)
    generator = _calibration_generator(seed_int)
    null_values = _calibration_values(n, max_rows, replicates, sd_value, rho,
                                      None, generator)
    alternative_values = _calibration_values(n, max_rows, replicates, sd_value,
                                              rho, effect_value, generator)
    # The null scenario has no true effect at all, and the alternative
    # scenario only has one when its first row was actually shifted.  A
    # row that carries no effect must not be booked as a true discovery:
    # that would hide its claims from the false-discovery count and
    # understate the family FDR.
    target = None
    if effect_value is not None and float(effect_value) != 0.0:
        target = torch.zeros((replicates, max_rows), dtype=torch.bool)
        target[:, 0] = True
    rules_out = {}
    for rule in selected_rules:
        rules_out[rule] = {
            "null": _calibration_scenario(
                null_values, rule, q_value, alt, sesoi_value, floor, sizes,
                boots, randoms, exact, seed_int, None),
            "alternative": _calibration_scenario(
                alternative_values, rule, q_value, alt, sesoi_value, floor,
                sizes, boots, randoms, exact, seed_int, target),
        }
    out.update({
        "valid": True,
        "n_units": n,
        "family_sizes": sizes,
        "sd": sd_value,
        "sesoi": sesoi_value,
        "effect": effect_value,
        "q": q_value,
        "direction": alt,
        "correlation": rho,
        "n_replicates": replicates,
        "n_boot": boots,
        "n_random": randoms,
        "min_seeds": floor,
        "max_exact": exact,
        "seed": seed_int,
        "rules": rules_out,
    })
    _CALIBRATION_CACHE[key] = out
    return out


def decision_rule_calibration_ladder(n_ladder, **kwargs) -> dict:
    """Calibrate the decision rule at several seed counts.

    A single seed count answers "what does this design do?", not "what
    would this design have to look like?".  The ladder runs the same
    calibration at a list of seed counts so preparation can state the
    seed count at which the registered pipeline actually honours its
    nominal error control and its power target - a number the
    normal-approximation power plan cannot produce, because the
    percentile bootstrap p-value's small-sample behaviour and the
    family-wise multiplicity step both enter the calibration and
    neither enters that formula.
    """
    if isinstance(n_ladder, (str, bytes)) or n_ladder is None:
        return {"valid": False, "reason": "n_ladder must be a sequence",
                "scenarios": {}}
    try:
        requested = list(n_ladder)
    except TypeError:
        return {"valid": False, "reason": "n_ladder must be a sequence",
                "scenarios": {}}
    scenarios = {}
    for value in requested:
        n = _positive_integer(value)
        if n is None:
            return {"valid": False,
                    "reason": "n_ladder must contain positive integers",
                    "scenarios": {}}
        scenario = decision_rule_calibration(n, **kwargs)
        scenarios[str(n)] = scenario
        if not scenario["valid"]:
            return {"valid": False, "reason": scenario["reason"],
                    "scenarios": scenarios}
    return {"valid": True, "reason": "", "scenarios": scenarios}
