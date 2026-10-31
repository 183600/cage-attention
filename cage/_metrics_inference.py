# cage metrics: bootstrap inference, equivalence and multiplicity control.
# Executed into the wrapper module namespace by the loader;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def paired_bootstrap_ci(diffs, n_boot: int = 10000, seed: int = 0,
                        alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for the mean of *paired* differences.

    ``diffs`` holds one value per independent unit (e.g. the same training
    seed under two configurations).  Differencing removes the between-unit
    variance that a two-sample test would fold in, which matters with the
    small seed counts used here.  NaNs should be filtered by the caller;
    if a non-finite difference remains, the readout abstains.

    Besides the CI the result carries the quantities needed to interpret
    *why* it looks the way it does:

    * ``p_value`` - two-sided bootstrap p-value
      ``2 * min(P(mean* <= 0), P(mean* >= 0))`` with ``+1`` smoothing
      (the fraction of resampled means on the wrong side of zero);
    * ``half_width`` / ``sd`` - CI half-width and paired-difference SD;
    * ``mde80`` - the minimal detectable effect at 80% power,
      ``(z_0.975 + z_0.8) * sd / sqrt(n)``.  An observed ``|mean|`` below
      it means the design could not have resolved an effect this small
      even if it were real, so "CI includes 0" must be read as
      *inconclusive*, not as evidence of absence;
    * ``cohens_d`` / ``hedges_g`` and their ``*_lo``/``*_hi`` bounds - the
      paired standardized effect and its percentile bootstrap interval
      (see :func:`paired_effect_size_ci`).  The interval quantifies
      uncertainty in the seed-to-seed scale, which the raw-unit CI cannot;
      it is descriptive and does not alter the pre-registered decision;
    * ``sign_consistency`` - fraction of non-zero paired differences that
      share the sign of the mean (robustness to a single seed driving the
      mean);
    * ``loo_mean_min`` / ``loo_mean_max`` / ``loo_sign_flip`` - jackknife
      companions: the range of the mean when each seed is left out in
      turn, and whether removing any one seed flips the direction of the
      effect.  A supported contrast with ``loo_sign_flip`` is carried by
      a single influential seed, whatever the CI says;
    * ``lo_bca`` / ``hi_bca`` / ``bca_agrees`` - the bias-corrected
      accelerated interval and whether it reaches the same
      excludes-0 decision as the pre-registered percentile interval
      (``None`` below n=4, where the jackknife acceleration is not
      estimable, and for degenerate bootstrap distributions);
    * ``sign_flip_p_greater`` / ``sign_flip_p_less`` /
      ``sign_flip_p_two_sided`` - exact (or Monte-Carlo above
      ``max_exact`` units) paired sign-flip p-values, with the design's
      ``sign_flip_min_p_*`` resolution floor.  These are validity
      companions, not a replacement for the pre-registered bootstrap
      decision; a bootstrap p below the exact floor is evidence that the
      interval is using moment assumptions the seed count cannot support.
    """
    alpha = _float_or_nan(alpha)
    d = _flatten_float(diffs)
    signflip = _sign_flip_companions(d)
    if d.numel() == 0:
        return {"mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n": 0, "sd": float("nan"),
                "half_width": float("nan"), "mde80": float("nan"),
                "p_value": float("nan"), "sign_consistency": float("nan"),
                "alpha": alpha,
                "loo_mean_min": float("nan"), "loo_mean_max": float("nan"),
                "loo_sign_flip": False, "lo_bca": float("nan"),
                "hi_bca": float("nan"), "bca_agrees": None,
                "bootstrap_below_sign_flip_floor": False,
                **_effect_size_abstention(alpha),
                **signflip}
    n = d.numel()
    if not bool(torch.isfinite(d).all()) or not _finite_moments(d):
        return {"mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n": int(n), "sd": float("nan"),
                "half_width": float("nan"), "mde80": float("nan"),
                "p_value": float("nan"), "sign_consistency": float("nan"),
                "alpha": alpha,
                "loo_mean_min": float("nan"), "loo_mean_max": float("nan"),
                "loo_sign_flip": False, "lo_bca": float("nan"),
                "hi_bca": float("nan"), "bca_agrees": None,
                "bootstrap_below_sign_flip_floor": False,
                **_effect_size_abstention(alpha),
                **signflip}
    if not _bootstrap_args_ok(n_boot, alpha):
        return {"mean": float(d.mean()), "lo": float("nan"),
                "hi": float("nan"), "n": int(n),
                "sd": (float(d.std(unbiased=True)) if n > 1
                       else float("nan")),
                "half_width": float("nan"), "mde80": float("nan"),
                "p_value": float("nan"), "sign_consistency": float("nan"),
                "alpha": alpha,
                "loo_mean_min": float("nan"), "loo_mean_max": float("nan"),
                "loo_sign_flip": False, "lo_bca": float("nan"),
                "hi_bca": float("nan"), "bca_agrees": None,
                "bootstrap_below_sign_flip_floor": False,
                **_effect_size_abstention(alpha),
                **signflip}
    n_boot = _positive_integer(n_boot)
    means = _bootstrap_means(d, n_boot, seed)
    if not bool(torch.isfinite(means).all()):
        # A resample overshot the float64 range even though the observed
        # values were finite: quantiles/p-values from this distribution are
        # not representable, so abstain rather than report a spurious
        # near-zero p-value.
        return {"mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n": int(n), "sd": float("nan"),
                "half_width": float("nan"), "mde80": float("nan"),
                "p_value": float("nan"), "sign_consistency": float("nan"),
                "alpha": alpha,
                "loo_mean_min": float("nan"), "loo_mean_max": float("nan"),
                "loo_sign_flip": False, "lo_bca": float("nan"),
                "hi_bca": float("nan"), "bca_agrees": None,
                "bootstrap_below_sign_flip_floor": False,
                **_effect_size_abstention(alpha),
                **signflip}
    out = _bootstrap_summary(d, means, alpha)
    out.update(signflip)
    p_lo = float((means <= 0.0).to(torch.float64).mean())
    p_hi = float((means >= 0.0).to(torch.float64).mean())
    out["p_value"] = (min(
        1.0, 2.0 * (min(p_lo, p_hi) * n_boot + 1.0) / (n_boot + 1.0))
        if n > 1 else float("nan"))
    floor = out.get("sign_flip_min_p_two_sided", float("nan"))
    out["bootstrap_below_sign_flip_floor"] = bool(
        math.isfinite(out["p_value"]) and math.isfinite(float(floor))
        and float(out["p_value"]) < float(floor))
    nonzero = d != 0.0
    if int(nonzero.sum()) == 0 or out["mean"] == 0.0:
        out["sign_consistency"] = float("nan")
    else:
        direction = 1.0 if out["mean"] > 0.0 else -1.0
        out["sign_consistency"] = float(
            (d[nonzero] * direction > 0.0).to(torch.float64).mean())
    loo = _jackknife_means(d)
    out["loo_mean_min"] = float(loo.min()) if loo.numel() else float("nan")
    out["loo_mean_max"] = float(loo.max()) if loo.numel() else float("nan")
    # A leave-one-out mean of exactly zero has no direction to flip; only a
    # strictly opposite sign counts (the same convention as the fold audit's
    # ``_sign`` helper).  Counting zero would flag a degenerate jackknife
    # replication as a reversed effect.
    out["loo_sign_flip"] = bool(
        loo.numel() > 0 and out["mean"] != 0.0
        and bool((((loo > 0.0) != (out["mean"] > 0.0))
                  & (loo != 0.0)).any()))
    bca = _bca_bounds(d, means, alpha)
    if bca is None:
        out["lo_bca"] = out["hi_bca"] = float("nan")
        out["bca_agrees"] = None
    else:
        out["lo_bca"], out["hi_bca"] = bca
        percentile_excludes = out["lo"] > 0.0 or out["hi"] < 0.0
        bca_excludes = out["lo_bca"] > 0.0 or out["hi_bca"] < 0.0
        out["bca_agrees"] = bool(percentile_excludes == bca_excludes)
    out.update(paired_effect_size_ci(
        d, n_boot=n_boot, seed=seed, alpha=alpha))
    return out


def equivalence_test(diffs, bound, n_boot: int = 10000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """Bootstrap equivalence (TOST) test for a *null prediction*.

    For hypotheses that predict a quantity stays put ("the H/log(T)-
    normalised entropy drift is length-stable"), the interesting result is
    not "the effect differs from 0" (which a sufficiently powered design
    always shows for any nonzero drift) but "the effect is small enough to
    be physically irrelevant": the pre-registered bound ``bound`` is the
    smallest effect that would refute the prediction.  ``diffs`` holds the
    paired per-seed values as in :func:`paired_bootstrap_ci`.

    Two one-sided bootstrap tests are combined (TOST): H0 is
    ``|mean| >= bound`` and ``p_value`` is
    ``max(P(mean* <= -bound), P(mean* >= +bound))`` with ``+1`` smoothing.
    ``equivalent`` is the CI-based decision rule (the 95% percentile CI of
    the mean lies strictly inside ``(-bound, +bound)``), which is the
    quantity pre-registered in the hypothesis registry; the TOST p-value
    feeds the same Benjamini-Hochberg family as the directional p-values,
    so null predictions and difference claims are controlled together.

    A CI that merely *includes* 0 does not establish equivalence: with
    seed-level noise wider than the bound, "not significant" is
    inconclusive, and the returned dict carries the usual
    ``lo``/``hi``/``sd``/``mde80`` fields so the report can say which.
    Returns NaN fields and ``equivalent=False`` for an empty input, a
    non-finite difference, or a non-positive/non-finite bound.

    The same small-sample robustness companions as in
    :func:`paired_bootstrap_ci` are attached: ``loo_within_bound`` (every
    leave-one-out mean still lies inside the bound - a single seed cannot
    create the apparent stability) and the BCa interval
    (``lo_bca``/``hi_bca``/``equivalent_bca``/``bca_agrees``), so an
    equivalence claim that is an artifact of one lucky seed or of
    percentile-interval bias is flagged at the decision point.  The exact
    sign-flip TOST companion (``sign_flip_p`` / ``sign_flip_p_lower`` /
    ``sign_flip_p_upper``) reports the same two boundary tests with a
    finite-sample randomization distribution.
    """
    bound = _float_or_nan(bound)
    alpha = _float_or_nan(alpha)
    d = _flatten_float(diffs)
    signflip = _sign_flip_tost_companions(d, bound, "equivalence")
    out = {
        "mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
        "n": 0, "sd": float("nan"), "half_width": float("nan"),
        "mde80": float("nan"), "p_value": float("nan"),
        "sign_consistency": float("nan"), "alpha": alpha,
        "test": "equivalence", "bound": bound, "equivalent": False,
        "loo_mean_min": float("nan"), "loo_mean_max": float("nan"),
        "loo_within_bound": False, "lo_bca": float("nan"),
        "hi_bca": float("nan"), "equivalent_bca": None,
        "bca_agrees": None,
        **_effect_size_abstention(alpha),
        **signflip,
    }
    if d.numel() < 2 or not bool(torch.isfinite(d).all()) \
            or not _finite_moments(d) \
            or not math.isfinite(bound) or bound <= 0.0 \
            or not _bootstrap_args_ok(n_boot, alpha):
        if d.numel() and not bool(torch.isfinite(d).all()):
            out["n"] = int(d.numel())
        if d.numel() == 1 and bool(torch.isfinite(d).all()) \
                and math.isfinite(bound) and bound > 0.0:
            # Report the observed point but not a decision: one paired unit
            # cannot establish equivalence (the exact sign-flip companion
            # already has p=0.5 in this case).
            out["mean"] = float(d.mean())
            out["n"] = 1
        return out
    n_boot = _positive_integer(n_boot)
    means = _bootstrap_means(d, n_boot, seed)
    if not bool(torch.isfinite(means).all()):
        return out
    out.update(_bootstrap_summary(d, means, alpha))
    p_below = float((means <= -bound).to(torch.float64).mean())
    p_above = float((means >= bound).to(torch.float64).mean())
    out["p_value"] = min(
        1.0, (max(p_below, p_above) * n_boot + 1.0) / (n_boot + 1.0))
    out["equivalent"] = bool(out["lo"] > -bound and out["hi"] < bound)
    loo = _jackknife_means(d)
    out["loo_mean_min"] = float(loo.min()) if loo.numel() else float("nan")
    out["loo_mean_max"] = float(loo.max()) if loo.numel() else float("nan")
    out["loo_within_bound"] = bool(
        loo.numel() > 0 and bool(loo.abs().max() < bound))
    bca = _bca_bounds(d, means, alpha)
    if bca is None:
        out["lo_bca"] = out["hi_bca"] = float("nan")
        out["equivalent_bca"] = None
        out["bca_agrees"] = None
    else:
        out["lo_bca"], out["hi_bca"] = bca
        out["equivalent_bca"] = bool(
            out["lo_bca"] > -bound and out["hi_bca"] < bound)
        out["bca_agrees"] = bool(out["equivalent"] == out["equivalent_bca"])
    out.update(paired_effect_size_ci(
        d, n_boot=n_boot, seed=seed, alpha=alpha))
    return out


def noninferiority_test(diffs, bound, n_boot: int = 10000, seed: int = 0,
                        alpha: float = 0.05) -> dict:
    """Bootstrap one-sided non-inferiority test for a *null prediction*.

    A component-completeness claim such as "adding the null slot does not
    cost accuracy relative to the no-gate arm" is neither a superiority
    test (the effect is expected to be ~0) nor a symmetric equivalence test
    (only *worse* than the bound matters, not *better*).  The pre-registered
    ``bound`` (> 0) is the smallest material loss: H0 is
    ``mean(diffs) <= -bound`` and the one-sided p-value is the bootstrap
    mass at or below ``-bound`` with the same ``+1`` smoothing as the other
    bootstrap p-values.  The decision rule, pre-registered in the
    hypothesis registry, is the CI counterpart: ``noninferior`` is True
    when the 95% percentile CI's *lower* bound lies strictly above
    ``-bound`` (the whole plausible effect is at most a sub-bound loss).
    A CI whose lower bound falls below ``-bound`` is a refutation (material
    loss), not "inconclusive" - the difference from the two-sided
    equivalence test, which needs both ends inside the bound.

    ``diffs`` holds the paired per-seed values as in
    :func:`paired_bootstrap_ci`; the usual ``lo``/``hi``/``sd``/``mde80``
    and small-sample (jackknife/BCa) companions are attached.  Returns NaN
    fields and ``noninferior=False`` for an empty input, a non-finite
    difference, or a non-positive/non-finite bound.
    """
    bound = _float_or_nan(bound)
    alpha = _float_or_nan(alpha)
    d = _flatten_float(diffs)
    signflip = _sign_flip_tost_companions(d, bound, "noninferiority")
    out = {
        "mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
        "n": 0, "sd": float("nan"), "half_width": float("nan"),
        "mde80": float("nan"), "p_value": float("nan"),
        "sign_consistency": float("nan"), "alpha": alpha,
        "test": "noninferiority", "bound": bound, "noninferior": False,
        "loo_mean_min": float("nan"), "loo_mean_max": float("nan"),
        "loo_above_bound": False, "lo_bca": float("nan"),
        "hi_bca": float("nan"), "noninferior_bca": None,
        "bca_agrees": None,
        **_effect_size_abstention(alpha),
        **signflip,
    }
    if d.numel() < 2 or not bool(torch.isfinite(d).all()) \
            or not _finite_moments(d) \
            or not math.isfinite(bound) or bound <= 0.0 \
            or not _bootstrap_args_ok(n_boot, alpha):
        if d.numel() and not bool(torch.isfinite(d).all()):
            out["n"] = int(d.numel())
        if d.numel() == 1 and bool(torch.isfinite(d).all()) \
                and math.isfinite(bound) and bound > 0.0:
            out["mean"] = float(d.mean())
            out["n"] = 1
        return out
    n_boot = _positive_integer(n_boot)
    means = _bootstrap_means(d, n_boot, seed)
    if not bool(torch.isfinite(means).all()):
        return out
    out.update(_bootstrap_summary(d, means, alpha))
    p_below = float((means <= -bound).to(torch.float64).mean())
    out["p_value"] = min(1.0, (p_below * n_boot + 1.0) / (n_boot + 1.0))
    out["noninferior"] = bool(out["lo"] > -bound)
    loo = _jackknife_means(d)
    out["loo_mean_min"] = float(loo.min()) if loo.numel() else float("nan")
    out["loo_mean_max"] = float(loo.max()) if loo.numel() else float("nan")
    out["loo_above_bound"] = bool(
        loo.numel() > 0 and bool(loo.min() > -bound))
    bca = _bca_bounds(d, means, alpha)
    if bca is None:
        out["lo_bca"] = out["hi_bca"] = float("nan")
        out["noninferior_bca"] = None
        out["bca_agrees"] = None
    else:
        out["lo_bca"], out["hi_bca"] = bca
        out["noninferior_bca"] = bool(out["lo_bca"] > -bound)
        out["bca_agrees"] = bool(out["noninferior"] == out["noninferior_bca"])
    out.update(paired_effect_size_ci(
        d, n_boot=n_boot, seed=seed, alpha=alpha))
    return out


def bootstrap_upper_p(diffs, bound, n_boot: int = 10000, seed: int = 0) -> float:
    """One-sided bootstrap p-value for ``H0: mean(diffs) >= bound``.

    The decision companion of the pre-registered "certified below a bound"
    rules (e.g. HB2's near-chance gate): a small p means the resampled
    means sit clearly below the bound, so the bound is rejected as an upper
    limit on the effect.  The same ``+1`` smoothing as the two-sided
    :func:`paired_bootstrap_ci` p-value is used, so the two readouts are
    on the same scale.  Returns NaN for an empty sample, a non-finite
    difference, or a non-finite bound; ``p >= 0.5`` is expected whenever
    the point estimate does not even fall below the bound (no amount of
    resampling certifies it).
    """
    d = _flatten_float(diffs)
    bound = _float_or_nan(bound)
    if d.numel() == 0 or not bool(torch.isfinite(d).all()) \
            or not _finite_moments(d) \
            or not math.isfinite(bound) \
            or not _bootstrap_args_ok(n_boot, 0.05):
        return float("nan")
    n_boot = _positive_integer(n_boot)
    means = _bootstrap_means(d, n_boot, seed)
    if not bool(torch.isfinite(means).all()):
        return float("nan")
    p = float((means >= bound).to(torch.float64).mean())
    return min(1.0, (p * n_boot + 1.0) / (n_boot + 1.0))


def minimum_effect_test(diffs, bound, direction: str = "+",
                        n_boot: int = 10000, seed: int = 0,
                        alpha: float = 0.05) -> dict:
    """Bootstrap minimum-effect (SESOI) test for a directional claim.

    A statistically significant effect is not automatically a
    *practically* relevant one.  The experiment registry therefore
    pre-registers a smallest effect size of interest (SESOI) per metric,
    but a gate that compares the **point estimate** with the SESOI still
    calls an effect practically relevant when its 95% CI spans both the
    SESOI and zero.  The minimum-effect test closes that gap: for a
    prediction "the effect is positive and at least ``bound``", H0 is
    ``mean(diffs) <= +bound`` (``direction='+'``) or
    ``mean(diffs) >= -bound`` (``direction='-'``), and support requires
    the whole 95% percentile CI to clear the bound in the registered
    direction (``beyond_sesoi``).  ``p_value`` is the bootstrap mass on
    the wrong side of the bound with the same ``+1`` smoothing as the
    other bootstrap p-values, so it can join a Benjamini-Hochberg family.

    This is the one-sided counterpart of :func:`equivalence_test`: TOST
    asks whether the effect is small enough to be negligible, this asks
    whether it is large enough to matter.  It is deliberately *not* a
    two-sided "|effect| >= bound" test, because the registry only makes
    directional SESOI claims; an undirected prediction has no minimum
    effect to establish (pass ``direction`` from the registry, never
    infer it from the observed sign).

    ``diffs`` holds the paired per-seed values as in
    :func:`paired_bootstrap_ci`.  The usual ``lo``/``hi``/``sd``/``mde80``
    and small-sample companions are attached: ``loo_beyond_sesoi`` (every
    leave-one-seed-out mean still clears the bound) and the BCa interval
    (``beyond_sesoi_bca``/``bca_agrees``).  Returns NaN fields and
    ``beyond_sesoi=False`` for an empty sample, fewer than two paired
    units, a non-finite difference, a non-positive/non-finite bound, an
    unknown direction, or invalid bootstrap arguments - a design that
    cannot estimate uncertainty must not certify a practical effect.
    """
    bound = _float_or_nan(bound)
    alpha = _float_or_nan(alpha)
    d = _flatten_float(diffs)
    out = {
        "mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
        "n": 0, "sd": float("nan"), "half_width": float("nan"),
        "mde80": float("nan"), "p_value": float("nan"),
        "sign_consistency": float("nan"), "alpha": alpha,
        "test": "minimum_effect", "bound": bound,
        "direction": direction, "beyond_sesoi": False,
        "loo_mean_min": float("nan"), "loo_mean_max": float("nan"),
        "loo_beyond_sesoi": False, "lo_bca": float("nan"),
        "hi_bca": float("nan"), "beyond_sesoi_bca": None,
        "bca_agrees": None,
    }
    if direction not in ("+", "-") or d.numel() < 2 \
            or not bool(torch.isfinite(d).all()) \
            or not _finite_moments(d) \
            or not math.isfinite(bound) or bound <= 0.0 \
            or not _bootstrap_args_ok(n_boot, alpha):
        if d.numel() and not bool(torch.isfinite(d).all()):
            out["n"] = int(d.numel())
        if direction in ("+", "-") and d.numel() == 1 \
                and bool(torch.isfinite(d).all()) \
                and math.isfinite(bound) and bound > 0.0:
            # Report the observed point but not a decision: one paired
            # unit cannot estimate the interval the claim rests on.
            out["mean"] = float(d.mean())
            out["n"] = 1
        return out
    n_boot = _positive_integer(n_boot)
    means = _bootstrap_means(d, n_boot, seed)
    if not bool(torch.isfinite(means).all()):
        return out
    out.update(_bootstrap_summary(d, means, alpha))
    if direction == "+":
        p_wrong_side = float((means <= bound).to(torch.float64).mean())
        out["beyond_sesoi"] = bool(out["lo"] > bound)
    else:
        p_wrong_side = float((means >= -bound).to(torch.float64).mean())
        out["beyond_sesoi"] = bool(out["hi"] < -bound)
    out["p_value"] = min(
        1.0, (p_wrong_side * n_boot + 1.0) / (n_boot + 1.0))
    nonzero = d != 0.0
    if int(nonzero.sum()) == 0 or out["mean"] == 0.0:
        out["sign_consistency"] = float("nan")
    else:
        sign = 1.0 if out["mean"] > 0.0 else -1.0
        out["sign_consistency"] = float(
            (d[nonzero] * sign > 0.0).to(torch.float64).mean())
    loo = _jackknife_means(d)
    out["loo_mean_min"] = float(loo.min()) if loo.numel() else float("nan")
    out["loo_mean_max"] = float(loo.max()) if loo.numel() else float("nan")
    if loo.numel():
        out["loo_beyond_sesoi"] = bool(
            bool((loo > bound).all()) if direction == "+"
            else bool((loo < -bound).all()))
    bca = _bca_bounds(d, means, alpha)
    if bca is None:
        out["lo_bca"] = out["hi_bca"] = float("nan")
        out["beyond_sesoi_bca"] = None
        out["bca_agrees"] = None
    else:
        out["lo_bca"], out["hi_bca"] = bca
        out["beyond_sesoi_bca"] = bool(
            out["lo_bca"] > bound if direction == "+"
            else out["hi_bca"] < -bound)
        out["bca_agrees"] = bool(
            out["beyond_sesoi"] == out["beyond_sesoi_bca"])
    return out


# ---------------------------------------------------------------------------
# Small-sample replication readout (Student-t, no SciPy dependency)
# ---------------------------------------------------------------------------
def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularised incomplete beta function.

    Numerical Recipes' ``betacf``; the convergence/underflow guards keep a
    legitimate tail evaluation finite instead of dividing by zero.
    """
    maxit, eps, fpmin = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)`` for ``a, b > 0`` and ``0 <= x <= 1``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_cdf(t: float, df: float) -> float:
    """CDF of Student's t with ``df`` degrees of freedom."""
    if math.isnan(t):
        return float("nan")
    if not math.isfinite(t):
        return 0.0 if t < 0.0 else 1.0
    if not (math.isfinite(df) and df > 0.0):
        return float("nan")
    x = df / (df + t * t)
    tail = 0.5 * _regularized_incomplete_beta(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0.0 else tail


def _student_t_sf(t: float, df: float) -> float:
    """Survival function of Student's t (the upper tail)."""
    if math.isnan(t):
        return float("nan")
    if not math.isfinite(t):
        return 0.0 if t > 0.0 else 1.0
    if not (math.isfinite(df) and df > 0.0):
        return float("nan")
    x = df / (df + t * t)
    tail = 0.5 * _regularized_incomplete_beta(df / 2.0, 0.5, x)
    return tail if t > 0.0 else 1.0 - tail


def _student_t_ppf(p: float, df: float) -> float:
    """Quantile of Student's t (bisection on the monotone survival).

    ``p`` is restricted to ``(0, 1)``.  Searching the *upper* tail directly
    avoids forming ``1 - p`` for a tiny lower-tail probability: that
    subtraction rounds to 1 and made a valid tiny alpha return NaN (or an
    inaccurate quantile) instead of the finite large-``t`` critical value.
    The very-large-``df`` branch avoids spending iterations on a
    distribution that is normal to machine precision.
    """
    if not (0.0 < p < 1.0) or not (math.isfinite(df) and df > 0.0):
        return float("nan")
    if p == 0.5:
        return 0.0
    if df > 1e6:
        return _normal_quantile(p)
    sign = -1.0 if p < 0.5 else 1.0
    target = p if p < 0.5 else 1.0 - p
    lo, hi = 0.0, 1.0
    while _student_t_sf(hi, df) > target:
        # Heavy tails (df <= 2) need a quantile far beyond 1e12 for tiny
        # but valid tail probabilities; keep doubling until the float64
        # range itself is exhausted.
        hi *= 2.0
        if not math.isfinite(hi):
            return sign * float("inf")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _student_t_sf(mid, df) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-14 * max(1.0, hi):
            break
    return sign * 0.5 * (lo + hi)


def replication_interval(diffs, direction=None, mode: str = "difference",
                         bound=None, alpha: float = 0.05) -> dict:
    """How large would the effect be in a *replication*, not just on average?

    A paired bootstrap CI answers "is the mean seed-level effect nonzero?".
    It does **not** answer the result-reading question a reviewer asks next:
    "if this experiment were repeated with the same number of training
    seeds, how large would the effect plausibly be, and would it keep its
    sign?".  With five seeds a small but significant mean can coexist with
    a replicate that is flat or reversed, because the CI shrinks with
    ``sqrt(n)`` while a new study's mean carries its own sampling variance.

    This function returns the Student-t **replication interval** for the
    mean of a future study of the same size,
    ``mean +- t_{1-alpha/2, n-1} * sd * sqrt(2/n)``.  The ``sqrt(2/n)``
    (instead of the CI's ``sqrt(1/n)``) is the variance of
    ``mean_new - mean_old`` when both studies have ``n`` seeds and the same
    seed-level SD, so the interval describes a same-size repeat rather than
    the precision of the current mean.  ``lo_single``/``hi_single`` give
    the wider prediction interval for one *individual* future seed,
    ``mean +- t * sd * sqrt(1 + 1/n)``, as a secondary readout.

    ``prob_replicates`` is the t-distribution probability that a same-size
    repeat's mean lies in the pre-registered direction (``+``: above 0,
    ``-``: below 0).  For a two-sided/`undirected` prediction it is the
    probability that the repeat keeps the observed sign, which is
    descriptive rather than a claim.  ``replicates`` is the interval-level
    decision companion: the whole future-study interval must clear zero in
    the registered direction.  For null predictions the same idea is read
    against the registered bound: ``mode='equivalence'`` requires the
    interval inside ``+-bound``, ``mode='noninferiority'`` requires its
    lower end above ``-bound``.  A future *single* seed inside a tight
    equivalence bound is usually unattainable even when the mean is
    compliant, so the same-size-repeat interval is the relevant one.

    The readout assumes seed-level effects are independent and
    approximately normal with a common SD; at these seed counts that is an
    explicit modelling assumption, not a distribution-free guarantee.  It
    is a **companion**, never a replacement for the pre-registered
    decision rule: a supported row whose replication interval includes
    zero is reported as *replication-uncertain*, not as unsupported.
    Returns ``valid=False`` (and NaN fields) for fewer than three paired
    seeds, non-finite differences, an invalid direction/mode/bound or an
    invalid ``alpha`` - a design that cannot estimate the seed SD cannot
    forecast a replication.
    """
    bound_is_bool = isinstance(bound, bool)
    alpha = _float_or_nan(alpha)
    bound = _float_or_nan(bound)
    # ``None`` means the caller did not pass the paired differences (e.g. a
    # display-only CI dict); it is not an empty sample that can be resampled.
    bad_input = False
    if diffs is None:
        d = torch.empty(0, dtype=torch.float64)
    else:
        try:
            d = _flatten_float(diffs)
        except (TypeError, ValueError, RuntimeError, OverflowError):
            d = torch.empty(0, dtype=torch.float64)
            bad_input = True
    out = {
        "valid": False,
        "reason": None,
        "n": int(d.numel()),
        "df": float("nan"),
        "mean": float("nan"),
        "sd": float("nan"),
        "alpha": alpha,
        "t_crit": float("nan"),
        "lo": float("nan"),
        "hi": float("nan"),
        "lo_single": float("nan"),
        "hi_single": float("nan"),
        "prob_replicates": float("nan"),
        "prob_single_same_sign": float("nan"),
        "replicates": False,
        "mode": mode,
        "direction": direction,
        "bound": bound,
        "assumptions": ("independent seed effects with a common SD; "
                        "t-based normal approximation"),
    }
    if bound_is_bool:
        # A boolean is a flag, not an equivalence bound; ``True`` would
        # otherwise be silently read as 1.0.
        out["reason"] = "bound must be a number, not a bool"
        return out
    if direction not in (None, "+", "-"):
        out["reason"] = f"unknown direction {direction!r}"
        return out
    if mode not in ("difference", "equivalence", "noninferiority"):
        out["reason"] = f"unknown mode {mode!r}"
        return out
    if mode != "difference" and not (
            math.isfinite(bound) and bound > 0.0):
        out["reason"] = "null-prediction mode needs a finite positive bound"
        return out
    if bad_input:
        out["reason"] = "paired differences are not a real numeric sample"
        return out
    if d.numel() == 0:
        out["reason"] = "no paired differences supplied"
        return out
    if d.numel() == 1 and bool(torch.isfinite(d).all()):
        out["mean"] = float(d.mean())
        out["reason"] = "fewer than 3 paired seeds: seed SD not estimable"
        return out
    if d.numel() < 3:
        out["reason"] = "fewer than 3 paired seeds: seed SD not estimable"
        return out
    if not bool(torch.isfinite(d).all()):
        out["reason"] = "non-finite paired difference"
        return out
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        out["reason"] = "alpha must lie strictly between 0 and 1"
        return out
    n = int(d.numel())
    df = float(n - 1)
    mean = float(d.mean())
    sd = float(d.std(unbiased=True))
    t_crit = -_student_t_ppf(alpha / 2.0, df)
    out.update({"valid": True, "n": n, "df": df, "mean": mean, "sd": sd,
                "t_crit": t_crit})
    if not (math.isfinite(t_crit) and math.isfinite(sd)
            and math.isfinite(mean)):
        # Finite differences can still overflow their sum/SD.  Do not
        # report an infinite replication mean next to NaN bounds.
        out["mean"] = mean if math.isfinite(mean) else float("nan")
        out["sd"] = sd if math.isfinite(sd) else float("nan")
        out["reason"] = "t quantile, seed SD or sample mean is not finite"
        out["valid"] = False
        return out
    # A future same-size study's mean, and one individual future seed.
    rep_half = t_crit * sd * math.sqrt(2.0 / n)
    single_half = t_crit * sd * math.sqrt(1.0 + 1.0 / n)
    out["lo"], out["hi"] = mean - rep_half, mean + rep_half
    out["lo_single"] = mean - single_half
    out["hi_single"] = mean + single_half

    def _prob(direction_sign, variance_factor):
        if sd == 0.0:
            if mean == 0.0:
                return 0.5
            return 1.0 if (mean > 0.0) == (direction_sign > 0) else 0.0
        scale = sd * math.sqrt(variance_factor)
        if direction_sign > 0:
            return _student_t_cdf(mean / scale, df)
        return _student_t_cdf(-mean / scale, df)

    if direction == "+":
        out["prob_replicates"] = _prob(1.0, 2.0 / n)
        out["prob_single_same_sign"] = _prob(1.0, 1.0 + 1.0 / n)
    elif direction == "-":
        out["prob_replicates"] = _prob(-1.0, 2.0 / n)
        out["prob_single_same_sign"] = _prob(-1.0, 1.0 + 1.0 / n)
    else:
        # Undirected prediction: report the chance the repeat keeps the
        # observed sign (descriptive; no direction was registered).
        if mean == 0.0:
            out["prob_replicates"] = 0.5
            out["prob_single_same_sign"] = 0.5
        else:
            keep = _prob(1.0 if mean > 0.0 else -1.0, 2.0 / n)
            out["prob_replicates"] = keep
            out["prob_single_same_sign"] = _prob(
                1.0 if mean > 0.0 else -1.0, 1.0 + 1.0 / n)
    if mode == "equivalence":
        out["replicates"] = bool(out["lo"] > -bound and out["hi"] < bound)
    elif mode == "noninferiority":
        out["replicates"] = bool(out["lo"] > -bound)
    elif direction == "+":
        out["replicates"] = bool(out["lo"] > 0.0)
    elif direction == "-":
        out["replicates"] = bool(out["hi"] < 0.0)
    else:
        out["replicates"] = bool(out["lo"] > 0.0 or out["hi"] < 0.0)
    return out


def seeds_for_power(sd, effect, alpha: float = 0.05,
                    power: float = 0.8) -> float:
    """Paired-design seed count for ``power`` at a given effect size.

    Planning counterpart of the post-hoc ``mde80``: with the observed
    paired-difference standard deviation ``sd``, the normal approximation
    ``n = ((z_{1-alpha/2} + z_power) * sd / |effect|)^2`` gives the number
    of independent units (training seeds) a contrast needs to reach the
    requested power.  Use it *before* running the grid to size the design
    for the pre-registered SESOI, or on a pilot estimate of ``sd``; a
    contrast that "includes 0" at fewer seeds than this is underpowered
    rather than negative.  Returns ``inf`` when ``sd`` is not positive or
    ``effect`` is zero/non-finite, and the raw (non-integer) count
    otherwise - callers round up.
    """
    if any(isinstance(value, bool)
           for value in (sd, effect, alpha, power)):
        # Booleans are flags, not effect sizes / variances / probabilities.
        return float("inf")
    try:
        sd = float(sd)
        effect = float(effect)
        alpha = float(alpha)
        power = float(power)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return float("inf")
    if not (math.isfinite(sd) and math.isfinite(effect)
            and math.isfinite(alpha) and math.isfinite(power)) \
            or sd <= 0.0 or effect == 0.0 \
            or not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        return float("inf")
    try:
        z = (-_normal_quantile(alpha / 2.0)
             + _normal_quantile(power))
    except (StatisticsError, ValueError):
        return float("inf")
    try:
        return float((z * sd / abs(effect)) ** 2)
    except OverflowError:
        # A finite but practically zero effect needs more seeds than a float
        # can represent.  The planning contract is "inf = unattainable", not
        # an OverflowError escaping into a results report.
        return float("inf")


def _validated_p_family(p_values, q):
    """Materialize a p-value family and validate its error level.

    Shared by the multiplicity procedures so a malformed container, a
    boolean level, or a p-value outside ``[0, 1]`` is handled the same way
    everywhere: the family is still reported, but the malformed entries
    stay untestable and an invalid level rejects nothing.
    """
    if isinstance(p_values, (str, bytes)) or isinstance(p_values, Mapping):
        raise ValueError(
            "p_values must be an iterable of p-values, got "
            f"{type(p_values).__name__}")
    try:
        p_values = list(p_values)
    except TypeError as exc:
        raise ValueError(
            "p_values must be an iterable of p-values, got "
            f"{type(p_values).__name__}") from exc
    out = [{"p": float("nan"), "p_adj": float("nan"), "reject": False}
           for _ in p_values]
    q = _float_or_nan(q)
    q_ok = math.isfinite(q) and 0.0 < q < 1.0
    finite = []
    for i, p in enumerate(p_values):
        if p is None:
            continue
        fp = _float_or_nan(p)
        if math.isfinite(fp) and 0.0 <= fp <= 1.0:
            finite.append((i, fp))
    return out, finite, q, q_ok


def _step_up_adjusted(finite, scale: float = 1.0) -> dict:
    """Monotone step-up adjusted p-values for already-validated entries."""
    m = len(finite)
    order = sorted(finite, key=lambda t: (t[1], t[0]))
    adjusted: dict[int, float] = {}
    prev = 1.0
    for rank in range(m, 0, -1):
        i, p = order[rank - 1]
        value = min(prev, p * m * scale / rank, 1.0)
        adjusted[i] = value
        prev = value
    return adjusted


def benjamini_hochberg(p_values, q: float = 0.05) -> list:
    """Benjamini-Hochberg step-up FDR control over one contrast family.

    ``p_values`` is a sequence of per-contrast p-values (NaN = not
    testable, e.g. a degenerate metric).  Returns one dict per input with
    ``p``, ``p_adj`` (monotone step-up adjusted p) and ``reject``
    (``p_adj <= q``).  Applying BH to the whole family *before* reading
    any contrast as supported is what keeps a grid of ~16 pre-registered
    contrasts from accumulating ~1 spurious "supported" claim under the
    null.
    """
    # Materialize once: a one-shot iterable must not be consumed by the
    # output scaffold and then silently read as an all-NaN family.  A bare
    # string/bytes/mapping is iterable but names characters/keys/byte
    # values, not p-values; a scalar has no family at all.  Reject these
    # malformed containers instead of manufacturing a testable family.
    out, finite, q, q_ok = _validated_p_family(p_values, q)
    m = len(finite)
    if not m:
        return out
    adjusted = _step_up_adjusted(finite)
    for i, p in finite:
        out[i] = {"p": p, "p_adj": adjusted[i],
                  "reject": bool(q_ok and adjusted[i] <= q)}
    return out


def benjamini_yekutieli(p_values, q: float = 0.05) -> list:
    """Benjamini-Yekutieli FDR control under arbitrary dependence.

    BH's FDR guarantee assumes independent or positively dependent tests.
    These contrast families reuse the same training seeds and one sealed
    evaluation stream, so that premise is not established.  BY multiplies
    each BH adjusted p-value by the harmonic factor
    ``c(m) = sum(1/i, i=1..m)``, which restores FDR control at ``q`` under
    arbitrary dependence at the cost of conservativeness.

    Returns the same per-row shape as ``benjamini_hochberg``.  This is a
    reporting companion: the pre-registered BH decision is unchanged.
    """
    out, finite, q, q_ok = _validated_p_family(p_values, q)
    m = len(finite)
    if not m:
        return out
    factor = sum(1.0 / i for i in range(1, m + 1))
    adjusted = _step_up_adjusted(finite, factor)
    for i, p in finite:
        out[i] = {"p": p, "p_adj": adjusted[i],
                  "reject": bool(q_ok and adjusted[i] <= q)}
    return out


def holm_bonferroni(p_values, q: float = 0.05) -> list:
    """Holm-Bonferroni step-down family-wise error control.

    Unlike FDR procedures, Holm bounds the probability of *any* false
    rejection in the family at ``q``.  It is valid under arbitrary
    dependence and uniformly more powerful than plain Bonferroni.  Reported
    next to the pre-registered BH decision as a multiplicity-model
    sensitivity companion.
    """
    out, finite, q, q_ok = _validated_p_family(p_values, q)
    m = len(finite)
    if not m:
        return out
    order = sorted(finite, key=lambda t: (t[1], t[0]))
    adjusted: dict[int, float] = {}
    prev = 0.0
    for rank, (i, p) in enumerate(order, start=1):
        value = min(1.0, max(prev, (m - rank + 1) * p))
        adjusted[i] = value
        prev = value
    for i, p in finite:
        out[i] = {"p": p, "p_adj": adjusted[i],
                  "reject": bool(q_ok and adjusted[i] <= q)}
    return out


def simes_test(p_values, alpha: float = 0.05) -> dict:
    """Simes global test of the family-wide null.

    Answers "does this family contain any signal at all?" with
    ``min_i m * p_(i) / i`` before any per-contrast claim is read.  Under
    the global null the Simes p-value is valid when the tests are
    independent or positively dependent; it is a descriptive omnibus
    companion, not a substitute for the per-row adjusted decisions.
    """
    _, finite, alpha, alpha_ok = _validated_p_family(p_values, alpha)
    m = len(finite)
    if not m:
        return {"p": float("nan"), "m": 0, "reject": False, "valid": False}
    order = sorted(finite, key=lambda t: (t[1], t[0]))
    p_simes = min(1.0, min(p * m / rank
                           for rank, (_, p) in enumerate(order, start=1)))
    return {"p": p_simes, "m": m,
            "reject": bool(alpha_ok and p_simes <= alpha),
            "valid": bool(alpha_ok)}


def false_coverage_ci(diffs, n_selected, n_tests, q: float = 0.05,
                      n_boot: int = 10000, seed: int = 0,
                      dependence: str = "prds") -> dict:
    """Selection-adjusted interval for a contrast selected by BH.

    The per-contrast 95% interval is the right uncertainty statement for a
    *pre-specified* comparison, but a results table reports only the
    contrasts that survived a multiplicity procedure.  Conditional on that
    selection the ordinary interval no longer covers the selected effect
    at its nominal rate: the reported set is biased towards large observed
    effects (the winner's-curse/selection problem).  Benjamini and
    Yekutieli's false-coverage-rate (FCR) adjustment repairs the statement
    for the *selected set*: when ``R`` of ``m`` contrasts are selected at
    FDR level ``q``, each selected contrast is reported with the
    ``1 - q * R / m`` interval (for PRDS/independence).  The selected set
    then has expected false coverage proportion at most ``q``, exactly the
    error rate the BH decision already targets.

    ``dependence='arbitrary'`` divides the adjusted level by the
    Benjamini-Yekutieli harmonic factor ``H_m = sum(1/i, i=1..m)``, giving
    an FCR statement that does not rely on positive dependence - the same
    companion logic as the BH vs BY multiplicity readouts.  The interval is
    therefore wider than the pre-registered percentile interval whenever
    ``R < m``; it equals the ordinary interval only when every tested
    contrast is selected and ``dependence='prds'``.

    ``diffs`` holds the same paired per-seed differences used by
    :func:`paired_bootstrap_ci`; the returned percentile interval uses the
    same bootstrap distribution.  This is a **reporting companion**: it
    does not change ``supports_hypothesis`` or the pre-registered BH
    decision.  A supported row whose FCR interval no longer clears zero is
    flagged as selection-sensitive, not silently presented as robust.
    Returns ``valid=False`` with NaN fields for an empty/non-finite
    sample, fewer than two paired units, ``n_selected`` outside
    ``1..n_tests``, an invalid error level, bootstrap arguments or
    dependence model, or an adjusted level outside ``(0, 1)``.
    """
    q_value = _float_or_nan(q)
    dep = str(dependence).strip().lower().replace("-", "_")
    if dep in ("prds", "independent", "positive_dependence"):
        dep = "prds"
    elif dep in ("arbitrary", "by", "any"):
        dep = "arbitrary"
    out = {
        "valid": False,
        "reason": None,
        "n": 0,
        "n_boot": 0,
        "q": q_value,
        "n_selected": None,
        "n_tests": None,
        "dependence": dependence,
        "harmonic_factor": float("nan"),
        "adjusted_alpha": float("nan"),
        "confidence_level": float("nan"),
        "mean": float("nan"),
        "lo": float("nan"),
        "hi": float("nan"),
        "half_width": float("nan"),
        "excludes_zero": False,
    }
    if dep not in ("prds", "arbitrary"):
        out["reason"] = f"unknown dependence model {dependence!r}"
        return out
    out["dependence"] = dep
    n_sel = _positive_integer(n_selected)
    n_tst = _positive_integer(n_tests)
    out["n_selected"] = n_sel
    out["n_tests"] = n_tst
    if n_sel is None:
        out["reason"] = "n_selected must be a positive integer"
        return out
    if n_tst is None:
        out["reason"] = "n_tests must be a positive integer"
        return out
    if n_sel > n_tst:
        out["reason"] = "n_selected cannot exceed n_tests"
        return out
    if not (math.isfinite(q_value) and 0.0 < q_value < 1.0):
        out["reason"] = "q must lie strictly between 0 and 1"
        return out
    n_boot_int = _positive_integer(n_boot)
    if n_boot_int is None:
        out["reason"] = "n_boot must be a positive integer"
        return out
    try:
        seed_int = _bootstrap_seed(seed)
    except ValueError as exc:
        out["reason"] = str(exc)
        return out
    try:
        d = _flatten_float(diffs)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        out["reason"] = "paired differences are not a real numeric sample"
        return out
    out["n"] = int(d.numel())
    if d.numel() < 2:
        out["reason"] = "fewer than two paired units"
        return out
    if not bool(torch.isfinite(d).all()) or not _finite_moments(d):
        out["reason"] = "non-finite paired difference"
        return out
    # ``H_m`` is the BY harmonic factor.  It is only needed for the
    # arbitrary-dependence variant; the exact sum is used for realistic
    # family sizes, with the asymptotic form above that to keep a huge
    # (but representable) n_tests from turning the report into a long loop.
    if dep == "arbitrary":
        if n_tst <= 100000:
            factor = math.fsum(1.0 / i for i in range(1, n_tst + 1))
        else:
            factor = math.log(n_tst) + 0.5772156649015329 + 0.5 / n_tst
        adjusted_alpha = q_value * n_sel / (n_tst * factor)
    else:
        factor = 1.0
        adjusted_alpha = q_value * n_sel / n_tst
    out["harmonic_factor"] = float(factor)
    out["adjusted_alpha"] = float(adjusted_alpha)
    if not (math.isfinite(adjusted_alpha)
            and 0.0 < adjusted_alpha < 1.0):
        out["reason"] = (
            "adjusted alpha leaves no finite interval "
            f"(q * R / m = {adjusted_alpha:g})")
        return out
    out["confidence_level"] = float(1.0 - adjusted_alpha)
    n_boot_int = int(n_boot_int)
    out["n_boot"] = n_boot_int
    means = _bootstrap_means(d, n_boot_int, seed_int)
    if not bool(torch.isfinite(means).all()):
        out["reason"] = "bootstrap means overflow float64"
        return out
    lo = float(torch.quantile(means, adjusted_alpha / 2.0))
    hi = float(torch.quantile(means, 1.0 - adjusted_alpha / 2.0))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        out["reason"] = "adjusted interval is not finite"
        return out
    out.update({
        "valid": True,
        "mean": float(d.mean()),
        "lo": lo,
        "hi": hi,
        "half_width": (hi - lo) / 2.0,
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    })
    return out


def wilson_ci(successes: int, total: int, z: float = 1.959964) -> dict:
    """Wilson score interval for a binomial proportion.

    Used for the parameter-free baselines, whose accuracy is a count over
    a finite evaluation stream: reporting ``accuracy`` without this
    interval would hide that a rule can only be checked to +-binomial
    noise at the evaluated sample size.
    """
    # Counts are integer samples, not booleans; ``float(True)`` would
    # silently read a flag as one success.
    if isinstance(successes, bool) or isinstance(total, bool):
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0}
    try:
        exact_successes = int(operator.index(successes))
    except (TypeError, ValueError, OverflowError):
        exact_successes = None
    try:
        exact_total = int(operator.index(total))
    except (TypeError, ValueError, OverflowError):
        exact_total = None
    exact_counts = exact_successes is not None and exact_total is not None
    if exact_counts:
        # Exact integer counts stay exact (and need no finite check): floats
        # cannot represent counts above 2**53, and some valid Python ints are
        # even beyond the float range entirely.
        if exact_total <= 0:
            return {"p": float("nan"), "lo": float("nan"),
                    "hi": float("nan"), "n": 0}
        n_out = exact_total
        try:
            total_f = float(exact_total)
        except OverflowError:
            total_f = None
    else:
        try:
            successes_f = float(successes)
            total_f = float(total)
        except (TypeError, ValueError, RuntimeError, OverflowError):
            return {"p": float("nan"), "lo": float("nan"),
                    "hi": float("nan"), "n": 0}
        if (not (math.isfinite(successes_f) and math.isfinite(total_f))
                or total_f <= 0):
            return {"p": float("nan"), "lo": float("nan"),
                    "hi": float("nan"), "n": 0}
        n_out = int(total_f)
    # The multiplier must be a finite positive real; NaN would make the
    # interval silently collapse to the point estimate, and a boolean is
    # a flag rather than a z quantile.
    z = _float_or_nan(z)
    if not (math.isfinite(z) and z > 0.0):
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": n_out}
    # Counts must be whole samples; accepting e.g. 0.5/2 would report a
    # binomial interval for a quantity that is not a proportion of trials.
    # Two exact integers above 2**53 can round to the same float64, so the
    # ordering check must use the integer values when both are available.
    if exact_counts:
        if exact_successes < 0 or exact_successes > exact_total:
            return {"p": float("nan"), "lo": float("nan"),
                    "hi": float("nan"), "n": exact_total}
        p = exact_successes / exact_total
    else:
        if successes_f != round(successes_f) or total_f != round(total_f) \
                or successes_f < 0 or successes_f > total_f:
            return {"p": float("nan"), "lo": float("nan"),
                    "hi": float("nan"), "n": n_out}
        p = successes_f / total_f
    z2 = z * z
    if total_f is not None:
        denom = 1.0 + z2 / total_f
        center = (p + z2 / (2.0 * total_f)) / denom
        half = (z * math.sqrt(p * (1.0 - p) / total_f
                              + z2 / (4.0 * total_f * total_f)) / denom)
    else:
        # ``total_f`` overflowed, so evaluate the interval in a scaled form.
        # ``sqrt(total)`` is still representable for counts up to roughly
        # 10**616; beyond that every correction underflows to zero and the
        # interval collapses to the point estimate, exactly as float64 does.
        try:
            sqrt_total = float(math.isqrt(exact_total))
        except OverflowError:
            sqrt_total = 0.0
        if sqrt_total == 0.0 or not math.isfinite(sqrt_total):
            denom, center, half = 1.0, p, 0.0
        else:
            inv = 1.0 / sqrt_total
            inv2 = inv * inv
            denom = 1.0 + z2 * inv2
            center = (p + (z2 / 2.0) * inv2) / denom
            base = math.sqrt(max(0.0, p * (1.0 - p)))
            half = z * math.hypot(base * inv,
                                  (z / 2.0) * inv2) / denom
    lo = min(p, max(0.0, center - half))
    hi = max(p, min(1.0, center + half))
    return {"p": p, "lo": lo, "hi": hi, "n": n_out}


def standardized_proportion(successes, totals, reference_weights=None) -> dict:
    """Post-stratified (directly standardized) rate over fixed strata.

    ``successes[i] / totals[i]`` is the observed rate inside stratum ``i``
    and ``reference_weights[i]`` is that stratum's mass in a *common*
    reference population.  The returned ``rate`` is

    ``sum_i w_i * p_i / sum_i w_i``  (over strata with ``totals[i] > 0``)

    i.e. the accuracy the arm would have if every length/condition had the
    reference composition instead of its own.  This is the standard
    post-stratification estimator for a *pre-treatment* covariate: the
    stratum is fixed by the generated sample (retrieval distance /
    structural ambiguity), never by the model, so reweighting cannot
    remove the mechanism effect being measured - it only removes the
    composition difference that otherwise confounds a length sweep.

    ``reference_weights=None`` uses ``totals`` itself, which reproduces the
    pooled (unstratified) rate; passing the base-length stratum counts
    makes a longer-length accuracy comparable to the base length.  Strata
    with no target observations cannot be estimated; they are dropped and
    ``coverage`` reports the fraction of reference mass that remains, so a
    comparison across a sparse stratum is never silently extrapolated.

    ``se``/``lo``/``hi`` use the design-based variance of a post-stratified
    proportion, ``sum_i W_i^2 p_i (1-p_i) / n_i`` with ``W_i`` the
    normalized reference weights, and a normal 95% interval.  It is a
    *within-stream* approximation used for the exploratory companion; the
    seed-level paired bootstrap remains the decision layer.

    Counts must be one-dimensional non-negative integers; malformed input
    returns a ``valid=False`` payload with NaN fields instead of raising,
    matching ``wilson_ci``'s contract so a reporting companion can abstain.
    """
    def _as_vector(values):
        if torch.is_tensor(values):
            if values.dim() > 1:
                return None, "counts must be one-dimensional"
        elif isinstance(values, (list, tuple)) and any(
                isinstance(item, (list, tuple)) for item in values):
            return None, "counts must be one-dimensional"
        exact = _exact_integer_sequence(values)
        if exact is not None:
            if not exact:
                return None, "empty counts"
            try:
                out = torch.tensor([float(value) for value in exact],
                                   dtype=torch.float64)
            except (TypeError, ValueError, RuntimeError, OverflowError):
                return None, "counts must be finite"
            if not bool(torch.isfinite(out).all()):
                return None, "counts must be finite"
            return out, ""
        try:
            raw = torch.as_tensor(values)
        except (TypeError, ValueError, RuntimeError, OverflowError):
            return None, "values must be numeric"
        if raw.dtype == torch.bool or raw.is_complex():
            return None, "values must be real numeric counts"
        if raw.numel() == 0:
            return None, "empty counts"
        try:
            if torch.is_tensor(values):
                out = values.detach().reshape(-1).to(device="cpu",
                                                     dtype=torch.float64)
            else:
                # Python float lists are captured as float32 by
                # ``as_tensor``; converting *after* that would round exact
                # integral counts above 2**24.  Ask for float64 directly.
                out = torch.as_tensor(values, dtype=torch.float64).detach(
                ).reshape(-1)
                out = out.to(device="cpu")
        except (TypeError, ValueError, RuntimeError, OverflowError):
            return None, "values must be numeric counts"
        if not bool(torch.isfinite(out).all()):
            return None, "counts must be finite"
        return out, ""

    s, why = _as_vector(successes)
    n, why_n = _as_vector(totals)
    first_error = why or why_n
    n_strata = 0 if s is None else int(s.numel())
    out = {
        "rate": float("nan"), "se": float("nan"),
        "lo": float("nan"), "hi": float("nan"),
        "coverage": 0.0, "n_strata": n_strata, "n_used": 0,
        "total_success": 0, "total_n": 0, "pooled_rate": float("nan"),
        "effective_n": float("nan"), "min_stratum_n": 0,
        "strata": [], "valid": False, "reason": "",
    }
    def _exact_fallback(s_exact, n_exact, w_exact):
        weights = list(n_exact if w_exact is None else w_exact)
        if any(value < 0 for value in s_exact):
            out["reason"] = "successes must be non-negative"
            return out
        if any(value < 0 for value in n_exact):
            out["reason"] = "totals must be non-negative"
            return out
        if any(value > limit for value, limit in zip(s_exact, n_exact)):
            out["reason"] = "successes must not exceed totals"
            return out
        if any(value < 0 for value in weights):
            out["reason"] = "reference_weights must be non-negative"
            return out
        weight_total_exact = sum(weights)
        if (isinstance(weight_total_exact, float)
                and not math.isfinite(weight_total_exact)):
            out["reason"] = "reference weights overflow when summed"
            return out
        if weight_total_exact <= 0:
            out["reason"] = "reference weights sum to zero"
            return out
        usable = [(s_i, n_i, w_i)
                  for s_i, n_i, w_i in zip(s_exact, n_exact, weights)
                  if n_i > 0 and w_i > 0]
        out["n_strata"] = len(s_exact)
        out["n_used"] = len(usable)
        if not usable:
            out["reason"] = (
                "no stratum is observed in both target and reference")
            return out
        weight_sum_used = sum(w_i for _, _, w_i in usable)
        out["coverage"] = (sum(w_i for _, _, w_i in usable)
                           / weight_total_exact)
        normalized = [w_i / weight_sum_used for _, _, w_i in usable]
        rates = [s_i / n_i for s_i, n_i, _ in usable]
        rate = (math.fsum(w * p for w, p in zip(normalized, rates))
                / math.fsum(normalized))
        variance = math.fsum(
            (w * w) * p * (1.0 - p) * (1 / n_i)
            for w, p, (_, n_i, _) in zip(normalized, rates, usable))
        precision = math.fsum(
            (w * w) * (1 / n_i)
            for w, (_, n_i, _) in zip(normalized, usable))
        try:
            effective_n = (1.0 / precision if precision > 0.0
                           else float("inf"))
        except OverflowError:
            effective_n = float("inf")
        se = math.sqrt(max(variance, 0.0))
        total_success = sum(s_i for s_i, _, _ in usable)
        total_n = sum(n_i for _, n_i, _ in usable)
        z = 1.959964
        out.update({
            "rate": float(rate), "se": float(se),
            "lo": max(0.0, rate - z * se),
            "hi": min(1.0, rate + z * se),
            "total_success": total_success,
            "total_n": total_n,
            "pooled_rate": (total_success / total_n
                            if total_n > 0 else float("nan")),
            "effective_n": effective_n,
            "min_stratum_n": min(n_i for _, n_i, _ in usable),
            "strata": [
                {"id": i,
                 "successes": s_i, "total": n_i,
                 "rate": (s_i / n_i if n_i > 0 else float("nan")),
                 "reference_weight": (float(w_i) if w_i <= _FLOAT_MAX
                                      else float("inf")),
                 "used": n_i > 0 and w_i > 0}
                for i, (s_i, n_i, w_i) in enumerate(
                    zip(s_exact, n_exact, weights))
            ],
            "valid": True, "reason": "",
        })
        return out

    if first_error:
        if first_error == "counts must be finite":
            s_exact = _exact_integer_sequence(successes)
            n_exact = _exact_integer_sequence(totals)
            if (s_exact is not None and n_exact is not None
                    and len(s_exact) == len(n_exact)):
                if reference_weights is None:
                    return _exact_fallback(s_exact, n_exact, None)
                w_values, why_w = _as_vector(reference_weights)
                if not why_w:
                    if w_values.numel() != len(s_exact):
                        out["reason"] = (
                            "reference_weights must have one entry per "
                            "stratum")
                        return out
                    w_exact = _exact_integer_sequence(reference_weights)
                    if w_exact is not None \
                            and len(w_exact) == len(s_exact):
                        return _exact_fallback(s_exact, n_exact, w_exact)
                    return _exact_fallback(
                        s_exact, n_exact,
                        [float(value) for value in w_values.tolist()])
                if why_w == "counts must be finite":
                    w_exact = _exact_integer_sequence(reference_weights)
                    if w_exact is not None \
                            and len(w_exact) == len(s_exact):
                        return _exact_fallback(s_exact, n_exact, w_exact)
                out["reason"] = f"reference_weights: {why_w}"
                return out
        out["reason"] = first_error
        return out
    if n is None or s.numel() != n.numel():
        out["reason"] = "successes and totals must have the same length"
        return out
    s_exact = _exact_integer_sequence(successes)
    n_exact = _exact_integer_sequence(totals)
    exact_counts = (s_exact is not None and n_exact is not None
                    and len(s_exact) == s.numel()
                    and len(n_exact) == n.numel())
    if exact_counts:
        for name, counts in (("successes", s_exact), ("totals", n_exact)):
            if any(value < 0 for value in counts):
                out["reason"] = f"{name} must be non-negative"
                return out
        if any(value > limit for value, limit in zip(s_exact, n_exact)):
            out["reason"] = "successes must not exceed totals"
            return out
        # Exact integer inputs must not be routed through float64 merely
        # because each count fits in a float: counts above 2**53 round to a
        # different value, and two finite counts near the float maximum
        # overflow when summed (the default reference weights).  The exact
        # fallback already existed for counts beyond the float range; use it
        # for all exact integer inputs, with exact weights when available.
        if reference_weights is None:
            return _exact_fallback(s_exact, n_exact, None)
    else:
        for name, values in (("successes", s), ("totals", n)):
            if bool((values < 0).any()):
                out["reason"] = f"{name} must be non-negative"
                return out
            if not bool(values.eq(values.round()).all()):
                out["reason"] = f"{name} must be integers"
                return out
            if name == "successes" and not bool((values <= n).all()):
                out["reason"] = "successes must not exceed totals"
                return out
    if reference_weights is None:
        w = n.clone()
    else:
        w, why_w = _as_vector(reference_weights)
        if why_w:
            if why_w == "counts must be finite" and s_exact is not None \
                    and n_exact is not None:
                w_exact = _exact_integer_sequence(reference_weights)
                if (w_exact is not None
                        and len(s_exact) == len(n_exact)
                        and len(w_exact) == len(s_exact)):
                    return _exact_fallback(s_exact, n_exact, w_exact)
            out["reason"] = f"reference_weights: {why_w}"
            return out
        if w.numel() != s.numel():
            out["reason"] = (
                "reference_weights must have one entry per stratum")
            return out
        if bool((w < 0).any()):
            out["reason"] = "reference_weights must be non-negative"
            return out
        if exact_counts:
            # The float path above validated the container shape/ordering;
            # compute with the exact integers instead of their rounded
            # float64 copies, whose sum can overflow even when each weight is
            # individually finite.
            w_exact = _exact_integer_sequence(reference_weights)
            if w_exact is not None and len(w_exact) == len(s_exact):
                return _exact_fallback(s_exact, n_exact, w_exact)
    weight_total = float(w.sum())
    if not math.isfinite(weight_total):
        # Normalizing by an infinite total would make every weight zero and
        # then ``1 / 0`` in the effective-n readout; all inputs were finite,
        # so abstain instead of leaking a ZeroDivisionError.
        out["reason"] = "reference weights overflow when summed"
        return out
    if weight_total <= 0.0:
        out["reason"] = "reference weights sum to zero"
        return out
    usable = (n > 0) & (w > 0)
    out["coverage"] = float(w[usable].sum() / w.sum())
    out["n_used"] = int(usable.sum())
    if not bool(usable.any()):
        out["reason"] = "no stratum is observed in both target and reference"
        return out
    p = torch.where(n > 0, s / torch.clamp(n, min=1.0),
                    torch.zeros_like(s))
    w_used = w[usable]
    weight_sum = w_used.sum()
    if not (math.isfinite(float(weight_sum)) and float(weight_sum) > 0.0):
        out["reason"] = "reference weights overflow when summed"
        return out
    w_norm = w_used / weight_sum
    counts_used = torch.clamp(n[usable], min=1.0)
    rate = (w_used * p[usable]).sum() / weight_sum
    variance = ((w_norm ** 2) * p[usable]
                * (1.0 - p[usable]) / counts_used).sum()
    # Kish-style effective sample size of the weighted stratum mean: a
    # reference bin represented by one or two base samples can dominate a
    # standardized comparison, so the companion must expose its precision
    # instead of printing a four-decimal rate as if it were well measured.
    effective_n = 1.0 / float(((w_norm ** 2) / counts_used).sum())
    se = float(torch.sqrt(torch.clamp(variance, min=0.0)))
    z = 1.959964
    used_success = s[usable].sum()
    used_total = n[usable].sum()
    if not (bool(torch.isfinite(used_success))
            and bool(torch.isfinite(used_total))):
        out["reason"] = "pooled success/total counts overflow"
        return out
    usable_flags = usable.tolist()
    if exact_counts:
        total_success = sum(value for value, used in zip(s_exact,
                                                         usable_flags) if used)
        total_n = sum(value for value, used in zip(n_exact,
                                                   usable_flags) if used)
        min_stratum_n = min(value for value, used in zip(n_exact,
                                                         usable_flags) if used)
    else:
        total_success = int(used_success.item())
        total_n = int(used_total.item())
        min_stratum_n = int(counts_used.min().item())
    rate_f = float(rate)
    lo = max(0.0, rate_f - z * se)
    hi = min(1.0, rate_f + z * se)
    out.update({
        "rate": rate_f, "se": se, "lo": lo, "hi": hi,
        "total_success": total_success,
        "total_n": total_n,
        "pooled_rate": (total_success / total_n
                        if total_n > 0 else float("nan")),
        "effective_n": float(effective_n),
        "min_stratum_n": min_stratum_n,
        "strata": [
            {"id": i,
             "successes": (s_exact[i] if exact_counts else int(s[i].item())),
             "total": (n_exact[i] if exact_counts else int(n[i].item())),
             "rate": (float(p[i]) if n[i] > 0 else float("nan")),
             "reference_weight": float(w[i]),
             "used": bool(usable[i])}
            for i in range(int(s.numel()))
        ],
        "valid": True, "reason": "",
    })
    return out
