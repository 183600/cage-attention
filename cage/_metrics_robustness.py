# cage metrics: clustered/standardized proportions, fold audits and sign-flip tests.
# Executed into the wrapper module namespace by the loader;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def clustered_proportion_ci(successes, n_per_unit, n_boot: int = 10000,
                            seed: int = 0, alpha: float = 0.05,
                            lower_alpha: float | None = None) -> dict:
    """Percentile interval for a mean of per-unit proportions.

    ``successes[i] / n_per_unit[i]`` is one training seed's accuracy on the
    sealed evaluation stream.  Seeds are independent units, but every seed
    is scored on the *same* stream, so pooling the counts treats repeated
    evaluations of identical examples as independent Bernoulli trials
    (pseudo-replication): the pooled interval keeps shrinking with the
    number of seeds even when their accuracies are identical.  Here seeds
    are resampled (between-seed noise) and each replicate draws one *shared*
    binomial measurement from the resampled mean accuracy on a stream of
    ``min(n_per_unit)`` examples (finite-stream noise).  The shared draw does
    not shrink with the number of seeds, so a five-seed grid cannot buy five
    times the evaluation precision.  ``pooled_wilson`` is kept as a
    descriptive companion only; decisions use the clustered bounds.

    ``lower_alpha`` optionally sets the lower-tail probability directly
    (e.g. ``0.05 / n_arms`` for a Bonferroni family-wise lower bound); the
    upper bound stays at ``1 - alpha / 2``.  Degenerate inputs return a
    ``valid=False`` payload instead of raising, so callers can abstain.
    """
    alpha = _float_or_nan(alpha)
    lo_q = alpha / 2.0 if lower_alpha is None else _float_or_nan(lower_alpha)
    hi_q = 1.0 - alpha / 2.0
    pooled = wilson_ci(0, 0)
    empty = {
        "mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
        "n": 0, "n_examples": 0,
        "sd_between": float("nan"), "se_between": float("nan"),
        "se_stream": float("nan"), "se": float("nan"),
        "pooled_wilson": pooled, "alpha": alpha,
        "lower_quantile": lo_q, "upper_quantile": hi_q,
        "valid": False, "reason": "",
    }
    def _count_vector(values):
        # Probe first: casting a complex tensor straight to float64 drops
        # its imaginary part (warning only), and a boolean is a flag rather
        # than a count of successes/examples - both would fabricate data.
        if torch.is_tensor(values):
            if values.dim() > 1:
                raise ValueError("counts must be one-dimensional")
        elif isinstance(values, (list, tuple)) and any(
                isinstance(item, (list, tuple)) for item in values):
            raise ValueError("counts must be one-dimensional")
        exact = _exact_integer_sequence(values)
        if exact is not None:
            if not exact:
                return torch.empty(0, dtype=torch.float64)
            try:
                return torch.tensor(
                    [float(value) for value in exact], dtype=torch.float64)
            except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                raise ValueError("counts must be finite") from exc
        try:
            probe = torch.as_tensor(values)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError("counts must be numeric") from exc
        if probe.is_complex():
            raise ValueError("counts must be real, not complex")
        if probe.dtype == torch.bool:
            raise ValueError("counts must be integers, not booleans")
        try:
            return torch.as_tensor(values, dtype=torch.float64).reshape(-1)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError("counts must be numeric") from exc

    try:
        s = _count_vector(successes)
        n = _count_vector(n_per_unit)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        empty["reason"] = "successes and unit lengths must be numeric counts"
        return empty
    s_exact = _exact_integer_sequence(successes)
    n_exact = _exact_integer_sequence(n_per_unit)
    if n.numel() == 1 and s.numel() > 1:
        n = n.expand_as(s).clone()
        if n_exact is not None and len(n_exact) == 1:
            n_exact = n_exact * int(s.numel())
    if s.numel() == 1 and n.numel() > 1:
        s = s.expand_as(n).clone()
        if s_exact is not None and len(s_exact) == 1:
            s_exact = s_exact * int(n.numel())
    if s.numel() != n.numel() or s.numel() == 0:
        empty["reason"] = "empty or mismatched inputs"
        return empty
    exact_counts = (s_exact is not None and n_exact is not None
                    and len(s_exact) == s.numel()
                    and len(n_exact) == n.numel())
    if exact_counts:
        if any(value <= 0 for value in n_exact):
            empty["reason"] = "non-finite or non-positive-length unit(s)"
            return empty
        if any(value < 0 or value > limit
               for value, limit in zip(s_exact, n_exact)):
            empty["reason"] = "success count outside [0, n]"
            return empty
    else:
        if not bool((torch.isfinite(s) & torch.isfinite(n) & (n > 0)).all()):
            empty["reason"] = "non-finite or non-positive-length unit(s)"
            return empty
        if not bool(((s >= 0) & (s <= n)).all()):
            empty["reason"] = "success count outside [0, n]"
            return empty
        # These are counts: ``successes[i]`` errors out of ``n_per_unit[i]``
        # examples.  Fractional values would make the shared binomial draw
        # meaningless (``torch.binomial`` takes an integer trial count) and
        # ``wilson_ci`` would silently truncate the pooled totals.
        if not bool((s.eq(s.round()) & n.eq(n.round())).all()):
            empty["reason"] = "success counts and unit lengths must be integers"
            return empty
    if not (math.isfinite(lo_q) and math.isfinite(hi_q)
            and 0.0 <= lo_q < hi_q <= 1.0):
        empty["reason"] = "quantile levels must lie in [0, 1]"
        return empty
    n_boot_int = _positive_integer(n_boot)
    if n_boot_int is None:
        empty["reason"] = "n_boot must be a positive integer"
        return empty
    if exact_counts:
        # Individually finite counts can overflow their float64 sum (e.g.
        # two counts near the float maximum).  The pooled totals and Wilson
        # companion accept exact Python integers, and the clustered
        # interval only uses per-unit ratios, so keep them exact instead of
        # abstaining on an otherwise valid estimation stream.
        total_success = sum(s_exact)
        total_n = sum(n_exact)
    else:
        if not (bool(torch.isfinite(s.sum()))
                and bool(torch.isfinite(n.sum()))):
            # Each count is finite but their pooled totals can still overflow
            # float64; the integer readout below would then raise on ``inf``.
            empty["reason"] = "pooled success/total counts overflow"
            return empty
        total_success = int(round(float(s.sum().item())))
        total_n = int(round(float(n.sum().item())))
    p = s / n
    units = int(s.numel())
    mean_acc = float(p.mean())
    sd_between = float(p.std(unbiased=True)) if units > 1 else 0.0
    se_between = sd_between / math.sqrt(units)
    n_stream = float(n.min())
    # A plug-in Binomial(n, 1) has zero variance, so a perfect (or
    # perfectly failed) stream would be reported as infinitely precise.
    # Keep half a pseudo-observation of each outcome in the draw
    # probability (Haldane/Jeffreys-style boundary smoothing): interior
    # accuracies are untouched, while the endpoints retain the finite
    # stream's resolution and the CI stays inside [0, 1].
    # Beta(0.5, 0.5) posterior predictive: after n successes out of n,
    # (n + 0.5) / (n + 1), i.e. move an endpoint only as far as half a
    # pseudo-observation justifies.  Interior accuracies are untouched.
    boundary = 0.5 / (n_stream + 1.0)
    p_stream = (boundary if mean_acc <= 0.0 else
                1.0 - boundary if mean_acc >= 1.0 else mean_acc)
    se_stream = math.sqrt(max(p_stream * (1.0 - p_stream) / n_stream, 0.0))
    device = p.device
    gen = torch.Generator(device=device).manual_seed(
        _bootstrap_seed(seed))
    idx = torch.randint(0, units, (n_boot_int, units), generator=gen,
                        device=device)
    mean_star = p[idx].mean(dim=1)
    draw_p = torch.where(mean_star <= 0.0, boundary,
                         torch.where(mean_star >= 1.0, 1.0 - boundary,
                                     mean_star))
    # One shared finite-stream draw per replicate, not one per seed: the
    # evaluation examples are common to every training seed.
    stream = torch.binomial(
        torch.full_like(draw_p, n_stream), draw_p, generator=gen
    ) / n_stream
    lo = float(torch.quantile(stream, lo_q).item())
    hi = float(torch.quantile(stream, hi_q).item())
    return {
        "mean": mean_acc, "lo": lo, "hi": hi,
        "n": units, "n_examples": total_n,
        "sd_between": sd_between, "se_between": se_between,
        "se_stream": se_stream,
        "se": math.sqrt(se_between ** 2 + se_stream ** 2),
        "pooled_wilson": wilson_ci(total_success, total_n),
        "alpha": alpha, "lower_quantile": lo_q, "upper_quantile": hi_q,
        "valid": True, "reason": "",
    }


def crossed_cluster_bootstrap_ci(correct_a, correct_b, n_boot: int = 4000,
                                 seed: int = 0, alpha: float = 0.05) -> dict:
    """Two-way (training seed x evaluation item) bootstrap for a paired effect.

    The pre-registered seed-level interval resamples *training seeds* while
    holding the sealed evaluation stream fixed.  It therefore estimates
    "would this contrast reproduce over new training runs **on this stream**",
    not "would it reproduce over new training runs **and** new evaluation
    items".  Those are different questions, and at the small seed counts used
    here the second one can be materially less certain: a difference carried
    by a handful of evaluation items is a property of the realized stream,
    not of the task population.

    ``correct_a`` / ``correct_b`` are boolean/0-1 arrays with shape
    ``(n_units, n_items)`` (rows are training seeds, columns are the shared
    evaluation items in consumption order); a 1-D array is read as one unit.
    Each bootstrap replicate resamples **both** axes with replacement - seeds
    to capture training variance, items to capture finite-stream variance -
    and averages the paired difference over the resampled seed x item grid.
    The returned percentile interval is a *data-level generalisation*
    companion: it is reported next to the pre-registered seed-level decision,
    never substituted for it.

    The result also carries the one-axis bootstrap intervals (``seed_lo`` /
    ``seed_hi`` keep all items, ``item_lo`` / ``item_hi`` keep all seeds) and
    the corresponding bootstrap SDs, so a reader can see which axis dominates
    the uncertainty.  ``variance_share_seed`` is the descriptive share
    ``sd_seed^2 / (sd_seed^2 + sd_item^2)`` (NaN when both are zero); it is
    not an ANOVA decomposition because the crossed bootstrap also contains
    the seed x item interaction.

    Both axes need at least two levels for their variance to be estimable;
    malformed, non-binary, or degenerate input returns ``valid=False`` with a
    reason instead of fabricating an interval, matching the other reporting
    companions.  The routine is deterministic for a given ``seed``.
    """
    alpha = _float_or_nan(alpha)
    empty = {
        "mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
        "n_units": 0, "n_items": 0, "n_boot": 0, "alpha": alpha,
        "seed_lo": float("nan"), "seed_hi": float("nan"),
        "item_lo": float("nan"), "item_hi": float("nan"),
        "sd_seed": float("nan"), "sd_item": float("nan"),
        "sd_total": float("nan"), "variance_share_seed": float("nan"),
        "dominant_axis": None, "ci_excludes_zero": False,
        "valid": False, "reason": "",
    }

    def _binary_matrix(values, label):
        try:
            raw = _exact_numeric_tensor(values)
        except (TypeError, ValueError, RuntimeError, OverflowError):
            return None, f"{label} must be numeric"
        if raw.is_complex():
            return None, f"{label} must be real-valued"
        try:
            raw = raw.to(torch.float64)
        except (TypeError, ValueError, RuntimeError, OverflowError):
            return None, f"{label} must be numeric"
        if raw.dim() == 1:
            raw = raw.reshape(1, -1)
        if raw.dim() != 2:
            return None, f"{label} must be 1-D or 2-D"
        if raw.numel() == 0:
            return None, f"{label} is empty"
        if not bool(torch.isfinite(raw).all()):
            return None, f"{label} must be finite"
        if not bool(((raw == 0.0) | (raw == 1.0)).all()):
            return None, f"{label} must contain only 0/1 correctness values"
        return raw, ""

    a, why_a = _binary_matrix(correct_a, "correct_a")
    b, why_b = _binary_matrix(correct_b, "correct_b")
    if why_a or why_b:
        empty["reason"] = why_a or why_b
        return empty
    if a.shape != b.shape:
        empty["reason"] = (
            "correct_a and correct_b must have the same (seed, item) shape")
        return empty
    if b.device != a.device:
        b = b.to(a.device)
    n_units, n_items = int(a.shape[0]), int(a.shape[1])
    empty["n_units"], empty["n_items"] = n_units, n_items
    if n_units < 2:
        empty["reason"] = (
            "at least two training seeds are needed to estimate the "
            "between-seed variance")
        return empty
    if n_items < 2:
        empty["reason"] = (
            "at least two evaluation items are needed to estimate the "
            "between-item variance")
        return empty
    n_boot_int = _positive_integer(n_boot)
    if n_boot_int is None:
        empty["reason"] = "n_boot must be a positive integer"
        return empty
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        empty["reason"] = "alpha must lie strictly between 0 and 1"
        return empty

    d = a - b
    mean = float(d.mean())
    device = d.device
    gen = torch.Generator(device=device).manual_seed(
        _bootstrap_seed(seed))
    # Bound the advanced-index temporary: chunk x n_units x n_items stays
    # near one million elements even when the evaluation stream is large.
    per_draw = max(1, n_units * n_items)
    chunk = max(1, min(512, 1_000_000 // per_draw))
    crossed_parts, seed_parts, item_parts = [], [], []
    remaining = n_boot_int
    while remaining > 0:
        size = min(chunk, remaining)
        seed_idx = torch.randint(0, n_units, (size, n_units),
                                 generator=gen, device=device)
        item_idx = torch.randint(0, n_items, (size, n_items),
                                 generator=gen, device=device)
        crossed = d[seed_idx[:, :, None], item_idx[:, None, :]].mean(
            dim=(1, 2))
        seed_only = d[seed_idx].mean(dim=(1, 2))
        item_only = d[:, item_idx].mean(dim=(0, 2))
        crossed_parts.append(crossed)
        seed_parts.append(seed_only)
        item_parts.append(item_only)
        remaining -= size
    crossed = torch.cat(crossed_parts)
    seed_dist = torch.cat(seed_parts)
    item_dist = torch.cat(item_parts)
    lo = float(torch.quantile(crossed, alpha / 2.0).item())
    hi = float(torch.quantile(crossed, 1.0 - alpha / 2.0).item())
    seed_lo = float(torch.quantile(seed_dist, alpha / 2.0).item())
    seed_hi = float(torch.quantile(seed_dist, 1.0 - alpha / 2.0).item())
    item_lo = float(torch.quantile(item_dist, alpha / 2.0).item())
    item_hi = float(torch.quantile(item_dist, 1.0 - alpha / 2.0).item())
    sd_seed = float(seed_dist.std(unbiased=True)) if n_boot_int > 1 else 0.0
    sd_item = float(item_dist.std(unbiased=True)) if n_boot_int > 1 else 0.0
    sd_total = (float(crossed.std(unbiased=True)) if n_boot_int > 1 else 0.0)
    denom = sd_seed ** 2 + sd_item ** 2
    if denom > 0.0:
        share = sd_seed ** 2 / denom
        axis = "seed" if sd_seed > sd_item else (
            "item" if sd_item > sd_seed else "tie")
    else:
        share = float("nan")
        axis = "none"
    return {
        "mean": mean, "lo": lo, "hi": hi,
        "n_units": n_units, "n_items": n_items,
        "n_boot": n_boot_int, "alpha": alpha,
        "seed_lo": seed_lo, "seed_hi": seed_hi,
        "item_lo": item_lo, "item_hi": item_hi,
        "sd_seed": sd_seed, "sd_item": sd_item, "sd_total": sd_total,
        "variance_share_seed": share, "dominant_axis": axis,
        "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "valid": True, "reason": "",
    }


def evaluation_fold_audit(correct_a, correct_b, n_folds=8) -> dict:
    """Fold jackknife over the finite evaluation stream.

    The independent units of the experiment are training seeds, but every
    seed is scored on the *same* finite stream of evaluation items.  The
    seed-level bootstrap answers "would the contrast replicate over new
    training runs on this stream"; it cannot ask whether the measured item
    difference is stable over the *items*.  This companion partitions the
    evaluation items into ``n_folds`` contiguous blocks (pseudo-batches),
    recomputes the paired effect in every block, and leaves one block out at
    a time.  A sign flip under either operation means the item-level effect
    is concentrated in one slice of the stream, so the reported difference
    is not a property of the evaluation population as a whole.

    ``correct_a`` / ``correct_b`` are boolean/0-1 arrays with shape
    ``(n_items,)`` for one seed or ``(n_units, n_items)`` when several
    training seeds share the same items (rows are seeds, columns are the
    common items).  The item ordering must be the common consumption order;
    contiguous folds then approximate the generator's batches.  ``n_folds``
    is capped at ``n_items // 2`` so every fold is non-empty, and at least
    two folds are required.

    The return value carries the per-fold effects and sizes, the
    leave-one-fold-out effects, the sign of the overall item effect, the
    share of the absolute effect carried by the most influential fold, and
    per-seed fold-sign-flip counts.  ``fragile`` is True when dropping one
    fold reverses the sign of the pooled effect or when a single fold
    carries more than half of the absolute effect (with at least three
    folds).  This is a *data-level sensitivity companion*: it is reported
    next to the pre-registered seed-level decision and does not replace it.
    Malformed or empty input returns ``valid=False`` with a reason instead
    of raising, matching the other reporting companions.
    """
    out = {
        "valid": False, "reason": "",
        "n_units": 0, "n_items": 0, "n_folds": 0, "requested_folds": 0,
        "overall_effect": float("nan"), "b": 0, "c": 0,
        "fold_sizes": [], "fold_effects": [],
        "fold_effect_mean": float("nan"), "fold_effect_sd": float("nan"),
        "fold_effect_min": float("nan"), "fold_effect_max": float("nan"),
        "fold_sign_consistency": float("nan"),
        "fold_all_same_sign": False,
        "loo_effects": [], "loo_effect_min": float("nan"),
        "loo_effect_max": float("nan"), "loo_sign_flip": False,
        "dominant_fold_share": float("nan"),
        "unit_effects": [], "unit_fold_sign_flips": [],
        "unit_fold_sign_consistency": [],
        "n_unit_fold_sign_flips": 0,
        "fragile": False,
    }
    try:
        a = _exact_numeric_tensor(correct_a)
        b = _exact_numeric_tensor(correct_b)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        out["reason"] = "correctness inputs must be numeric arrays"
        return out
    if a.numel() == 0 or b.numel() == 0:
        out["reason"] = "empty correctness arrays"
        return out
    if a.shape != b.shape:
        out["reason"] = "correctness arrays must have the same shape"
        return out
    if a.dim() == 1:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    if a.dim() != 2:
        out["reason"] = "correctness arrays must be 1-D or 2-D"
        return out
    for name, values in (("correct_a", a), ("correct_b", b)):
        if values.is_complex():
            out["reason"] = f"{name} must be real booleans/0-1"
            return out
        if values.is_floating_point() and (
                not bool(torch.isfinite(values).all())
                or not bool(values.eq(values.round()).all())):
            out["reason"] = f"{name} must be integral booleans/0-1"
            return out
        if not bool(((values == 0) | (values == 1)).all()):
            out["reason"] = f"{name} must contain only 0/1 (or booleans)"
            return out
    a = a.to(torch.bool)
    if b.device != a.device:
        b = b.to(a.device)
    b = b.to(torch.bool)
    n_units, n_items = int(a.size(0)), int(a.size(1))
    if n_items < 4:
        out["reason"] = "need at least four evaluation items for folds"
        out["n_units"] = n_units
        out["n_items"] = n_items
        return out
    if isinstance(n_folds, bool):
        out["reason"] = "n_folds must be an integer, not bool"
        return out
    try:
        requested = operator.index(n_folds)
    except TypeError:
        out["reason"] = "n_folds must be an integer"
        return out
    if requested < 2:
        out["reason"] = "n_folds must be at least two"
        return out
    k = min(int(requested), n_items // 2)
    if k < 2:
        out["reason"] = "not enough items for two non-empty folds"
        return out
    folds = torch.tensor_split(torch.arange(n_items), k)
    if any(int(fold.numel()) == 0 for fold in folds):
        out["reason"] = "fold construction produced an empty fold"
        return out
    diff = a.to(torch.float64) - b.to(torch.float64)
    total = diff.sum()
    # ``diff`` is (n_units, n_items): the item-level effect pools every
    # seed's evaluation of the common items, so the denominator is
    # n_units * n_items, not n_items alone.
    overall = float(total / (n_units * n_items))
    unit_effects = diff.mean(dim=1)
    eps = 1e-12

    def _sign(value: float) -> int:
        return 1 if value > eps else (-1 if value < -eps else 0)

    overall_sign = _sign(overall)
    fold_effects, fold_sizes, loo_effects = [], [], []
    for fold in folds:
        size = int(fold.numel())
        fold_effect = float(diff[:, fold].mean())
        fold_effects.append(fold_effect)
        fold_sizes.append(size)
        loo_effects.append(
            float((total - diff[:, fold].sum())
                  / (n_units * (n_items - size))))
    fold_effect_t = torch.tensor(fold_effects, dtype=torch.float64)
    if overall_sign == 0:
        consistency = float("nan")
    else:
        nonzero = [f for f in fold_effects if abs(f) > eps]
        consistency = (sum(1 for f in fold_effects
                           if _sign(f) == overall_sign) / len(nonzero)
                       if nonzero else float("nan"))
    all_same = bool(
        overall_sign != 0 and all(_sign(f) in (0, overall_sign)
                                  for f in fold_effects))
    loo_sign_flip = bool(
        overall_sign != 0
        and any(_sign(value) not in (0, overall_sign) for value in loo_effects))
    contributions = [size * abs(effect)
                     for size, effect in zip(fold_sizes, fold_effects)]
    contribution_sum = sum(contributions)
    dominant = (max(contributions) / contribution_sum
                if contribution_sum > eps else float("nan"))
    unit_fold_flips, unit_consistency = [], []
    for unit in range(n_units):
        unit_sign = _sign(float(unit_effects[unit]))
        fold_signs = []
        for fold in folds:
            effect = float(diff[unit, fold].mean())
            if abs(effect) > eps:
                fold_signs.append(_sign(effect))
        if not fold_signs:
            unit_consistency.append(float("nan"))
            continue
        has_positive = any(sign > 0 for sign in fold_signs)
        has_negative = any(sign < 0 for sign in fold_signs)
        if unit_sign == 0:
            # The unit's mean effect cancels exactly, so there is no
            # overall direction to measure consistency against.  The fold
            # effects can still point in opposite directions, and the
            # per-seed flip count is specifically about that disagreement;
            # dropping it here silently hid the most inconsistent seeds.
            unit_consistency.append(float("nan"))
            if has_positive and has_negative:
                unit_fold_flips.append(int(unit))
            continue
        matching = sum(1 for sign in fold_signs if sign == unit_sign)
        unit_consistency.append(matching / len(fold_signs))
        if has_positive and has_negative:
            unit_fold_flips.append(int(unit))
    fragile = bool(
        overall_sign != 0
        and (loo_sign_flip
             or (k >= 3 and math.isfinite(dominant) and dominant > 0.5)))
    out.update({
        "valid": True, "reason": "",
        "n_units": n_units, "n_items": n_items, "n_folds": k,
        "requested_folds": int(requested),
        "overall_effect": overall,
        "b": int((~a & b).sum()),
        "c": int((a & ~b).sum()),
        "fold_sizes": fold_sizes, "fold_effects": fold_effects,
        "fold_effect_mean": float(fold_effect_t.mean()),
        "fold_effect_sd": (float(fold_effect_t.std(unbiased=True))
                           if k > 1 else float("nan")),
        "fold_effect_min": float(fold_effect_t.min()),
        "fold_effect_max": float(fold_effect_t.max()),
        "fold_sign_consistency": consistency,
        "fold_all_same_sign": all_same,
        "loo_effects": loo_effects,
        "loo_effect_min": min(loo_effects),
        "loo_effect_max": max(loo_effects),
        "loo_sign_flip": loo_sign_flip,
        "dominant_fold_share": dominant,
        "unit_effects": [float(v) for v in unit_effects],
        "unit_fold_sign_flips": unit_fold_flips,
        "unit_fold_sign_consistency": unit_consistency,
        "n_unit_fold_sign_flips": len(unit_fold_flips),
        "fragile": fragile,
    })
    return out


def sign_flip_test(diffs, alternative: str = "two-sided",
                   n_random: int = 10000, seed: int = 0,
                   max_exact: int = 16) -> dict:
    """Exact / Monte-Carlo paired sign-flip randomization test.

    Under the null that the paired differences are symmetric around zero,
    each difference's sign is exchangeable.  Enumerating all ``2**n`` sign
    assignments gives a finite-sample exact p-value; above ``max_exact``
    units the test samples sign assignments instead and reports its Monte
    Carlo resolution.  ``max_exact`` is capped at 20 (≈1M assignments) so
    the companion cannot accidentally become a denial of service.

    This is the small-sample validity companion to
    :func:`paired_bootstrap_ci`.  At the five seeds used by the grids the
    percentile bootstrap's p-value can be several orders of magnitude
    below the smallest p-value the design can produce (all five differences
    in the predicted direction gives an exact one-sided p of
    ``1/2**5 = 0.03125`` and an exact two-sided p of ``0.0625``).  The
    bootstrap interval remains the pre-registered decision rule; this test
    makes the resolution limit explicit instead of letting a moment
    approximation claim precision the paired design does not have.

    ``alternative`` is ``"two-sided"``, ``"greater"`` or ``"less"``.  The
    returned ``p_values`` dict always carries all three alternatives so a
    pre-registered direction can be read without re-running the test.
    Returns ``valid=False`` with NaN fields for empty/non-finite input or
    invalid arguments; it never raises on malformed data.
    """
    alt = str(alternative).strip().lower().replace("_", "-")
    if alt == "two sided":
        alt = "two-sided"
    try:
        d = (_flatten_float(diffs) if diffs is not None
             else _flatten_float([]))
    except (TypeError, ValueError, RuntimeError, OverflowError):
        # The documented contract is an abstention, not an exception, for
        # malformed input.  A non-numeric sequence (e.g. ``["a"]``) cannot
        # be cast to a paired-difference sample, so return the NaN payload
        # with a reason instead of letting torch's conversion escape.
        d = _flatten_float([])
        malformed = True
    else:
        malformed = False
    n = int(d.numel())
    out = {
        "p_value": float("nan"),
        "p_values": {"greater": float("nan"), "less": float("nan"),
                     "two-sided": float("nan")},
        "min_attainable_p": float("nan"),
        "min_attainable_p_values": {"greater": float("nan"),
                                    "less": float("nan"),
                                    "two-sided": float("nan")},
        "n": n, "observed": float("nan"), "alternative": alternative,
        "method": "none", "n_permutations": 0, "valid": False,
        "reason": "",
    }
    if alt not in ("two-sided", "greater", "less"):
        out["reason"] = f"unknown alternative {alternative!r}"
        return out
    if malformed:
        out["reason"] = "paired differences must be numeric"
        return out
    if n == 0:
        out["reason"] = "empty paired sample"
        return out
    if not bool(torch.isfinite(d).all()):
        out["reason"] = "paired differences must be finite"
        return out
    n_random_int = _positive_integer(n_random)
    if n_random_int is None:
        out["reason"] = "n_random must be a positive integer"
        return out
    max_exact_int = _positive_integer(max_exact)
    if max_exact_int is None:
        out["reason"] = "max_exact must be a positive integer"
        return out
    if max_exact_int > 20:
        # 2**20 sign assignments is already ≈1M rows; a larger exact
        # threshold would turn a statistics companion into an accidental
        # denial of service.  The documented contract caps the threshold,
        # so a caller asking for more gets the 20-unit cap (and the
        # Monte-Carlo fallback above it) instead of an abstention.
        max_exact_int = 20
    if isinstance(seed, bool):
        out["reason"] = "seed must be an integer, not bool"
        return out
    try:
        # Match the other resampling entry points: an integer-like object
        # with ``__index__`` names the same exact stream, while floats and
        # strings stay invalid instead of being truncated or parsed.
        seed_int = int(operator.index(seed))
    except TypeError:
        out["reason"] = "seed must be an integer"
        return out
    if not _TORCH_SEED_MIN <= seed_int <= _TORCH_SEED_MAX:
        out["reason"] = "seed must fit in torch's 64-bit seed range"
        return out

    obs = float(d.mean())
    if not math.isfinite(obs):
        # Finite observations can still overflow their sum/mean (e.g.
        # several 1e308 differences).  The sign-flip distribution is then
        # not representable and a p-value from it would be fabricated.
        out["reason"] = "paired differences overflow the sample mean"
        return out
    abs_obs = abs(obs)
    # Roundoff in the enumerated means scales with the largest difference,
    # not with an absolute floor: a sample of tiny (but valid) effects such
    # as [1e-13] * 5 must still reach the exact 1/2**n floor instead of
    # having every assignment counted as a tie.  Keep the tolerance
    # relative to the data scale (and to the observed mean after
    # cancellation).
    scale = max(abs_obs, float(d.abs().max()))
    tol = 1e-12 * scale
    counts = {"greater": 0, "less": 0, "two-sided": 0}
    overflow = False
    if n <= max_exact_int:
        total = 1 << n
        chunk = min(total, 65536)
        shifts = torch.arange(n, dtype=torch.int64, device=d.device)
        for start in range(0, total, chunk):
            stop = min(total, start + chunk)
            idx = torch.arange(start, stop, dtype=torch.int64,
                               device=d.device)
            bits = (idx.unsqueeze(1) >> shifts) & 1
            signs = bits.to(dtype=d.dtype) * 2.0 - 1.0
            means = (signs * d).sum(dim=1) / float(n)
            if not bool(torch.isfinite(means).all()):
                overflow = True
                break
            counts["greater"] += int((means >= obs - tol).sum())
            counts["less"] += int((means <= obs + tol).sum())
            counts["two-sided"] += int((means.abs() >= abs_obs - tol).sum())
        if overflow:
            out["reason"] = "sign-flip means overflow float64"
            return out
        p_values = {key: counts[key] / float(total) for key in counts}
        # Two-sided tails contain the all-plus and all-minus assignments, so
        # their exact floor is twice the one-sided floor (capped at 1).
        min_values = {"greater": 1.0 / total, "less": 1.0 / total,
                      "two-sided": min(1.0, 2.0 / total)}
        method, n_perm = "exact", total
    else:
        # Use the full user seed.  Masking to 32 bits silently aliased a
        # different requested stream (e.g. 0 and 2**32) to the same test.
        gen = torch.Generator(device=d.device).manual_seed(seed_int)
        chunk = min(n_random_int, 20000)
        done = 0
        while done < n_random_int:
            size = min(chunk, n_random_int - done)
            signs = torch.randint(0, 2, (size, n),
                                  generator=gen, device=d.device)
            signs = signs.to(dtype=d.dtype) * 2.0 - 1.0
            means = (signs * d).sum(dim=1) / float(n)
            if not bool(torch.isfinite(means).all()):
                out["reason"] = "sign-flip means overflow float64"
                return out
            counts["greater"] += int((means >= obs - tol).sum())
            counts["less"] += int((means <= obs + tol).sum())
            counts["two-sided"] += int((means.abs() >= abs_obs - tol).sum())
            done += size
        # +1 smoothing makes the Monte-Carlo p strictly positive and keeps
        # the test honest at the sampled resolution.
        p_values = {key: (counts[key] + 1.0) / (n_random_int + 1.0)
                    for key in counts}
        min_values = {key: 1.0 / (n_random_int + 1.0) for key in counts}
        method, n_perm = "monte-carlo", n_random_int
    out.update({
        "p_value": p_values[alt],
        "p_values": p_values,
        "min_attainable_p": min_values[alt],
        "min_attainable_p_values": min_values,
        "observed": obs,
        "alternative": alt,
        "method": method,
        "n_permutations": n_perm,
        "valid": True,
        "reason": "",
    })
    return out


def sign_flip_resolution(n, alternative: str = "two-sided", *,
                         n_random: int = 10000,
                         max_exact: int = 16) -> dict:
    """Smallest p-value the paired sign-flip test can attain at ``n`` units.

    The analysis layer reports this floor next to every contrast, but a
    design problem is cheaper to catch *before* training.  This helper
    mirrors :func:`sign_flip_test` exactly: for ``n <= max_exact`` the
    exact enumeration floor is ``1 / 2**n`` for a one-sided alternative and
    ``min(1, 2 / 2**n)`` for a two-sided one; above that threshold the
    Monte-Carlo fallback cannot report a p-value below ``1 / (n_random+1)``
    regardless of ``n``.  ``max_exact`` is capped at 20 like the test
    itself.

    Returns a payload with ``min_attainable_p`` and ``method``, or
    ``valid=False`` (NaN floor) for a non-positive/fractional ``n``, an
    unknown alternative, or invalid resampling arguments.
    """
    alt = str(alternative).strip().lower().replace("_", "-")
    if alt == "two sided":
        alt = "two-sided"
    out = {
        "n": None,
        "alternative": alternative,
        "method": "none",
        "min_attainable_p": float("nan"),
        "valid": False,
        "reason": "",
    }
    if alt not in ("two-sided", "greater", "less"):
        out["reason"] = f"unknown alternative {alternative!r}"
        return out
    n_int = _positive_integer(n)
    if n_int is None:
        out["reason"] = "n must be a positive integer"
        return out
    n_random_int = _positive_integer(n_random)
    if n_random_int is None:
        out["reason"] = "n_random must be a positive integer"
        return out
    max_exact_int = _positive_integer(max_exact)
    if max_exact_int is None:
        out["reason"] = "max_exact must be a positive integer"
        return out
    max_exact_int = min(max_exact_int, 20)
    out["n"] = n_int
    if n_int <= max_exact_int:
        total = 1 << n_int
        floor = (min(1.0, 2.0 / total) if alt == "two-sided"
                 else 1.0 / total)
        out["method"] = "exact"
    else:
        floor = 1.0 / (n_random_int + 1.0)
        out["method"] = "monte-carlo"
    out["min_attainable_p"] = float(floor)
    out["valid"] = True
    return out


def sign_flip_min_units(alternative: str = "two-sided", q: float = 0.05,
                        *, n_random: int = 10000,
                        max_exact: int = 16):
    """Smallest paired-unit count whose sign-flip floor clears ``q``.

    Returns ``None`` when no attainable count clears the threshold (for
    example a very small ``q`` below the Monte-Carlo fallback floor), and
    for malformed arguments.  This is the design-planning counterpart of
    :func:`sign_flip_resolution`: a two-sided claim at ``q=0.05`` needs six
    seeds because ``2 / 2**5 = 0.0625``, while a directional claim can be
    decided with five (``1 / 2**5 = 0.03125``).
    """
    q_value = _float_or_nan(q)
    if not (math.isfinite(q_value) and 0.0 < float(q_value) < 1.0):
        return None
    max_exact_int = _positive_integer(max_exact)
    if max_exact_int is None:
        return None
    max_exact_int = min(max_exact_int, 20)
    for n in range(1, max_exact_int + 2):
        resolution = sign_flip_resolution(
            n, alternative, n_random=n_random, max_exact=max_exact_int)
        if not resolution["valid"]:
            return None
        if float(resolution["min_attainable_p"]) <= float(q_value):
            return n
    return None


def _sign_flip_companions(diffs, n_random: int = 10000,
                          seed: int = 0, max_exact: int = 16) -> dict:
    """All three exact sign-flip p-values in the shape contrasts store."""
    result = sign_flip_test(diffs, "two-sided", n_random=n_random,
                            seed=seed, max_exact=max_exact)
    pv = result["p_values"]
    mins = result["min_attainable_p_values"]
    return {
        # Convenience aliases for callers that do not know the registered
        # direction: the default readout is the two-sided test.
        "sign_flip_p": pv["two-sided"],
        "sign_flip_min_p": mins["two-sided"],
        "sign_flip_p_greater": pv["greater"],
        "sign_flip_p_less": pv["less"],
        "sign_flip_p_two_sided": pv["two-sided"],
        "sign_flip_method": result["method"],
        "sign_flip_n_permutations": result["n_permutations"],
        "sign_flip_min_p_greater": mins["greater"],
        "sign_flip_min_p_less": mins["less"],
        "sign_flip_min_p_two_sided": mins["two-sided"],
        "sign_flip_valid": result["valid"],
    }


def _sign_flip_tost_companions(diffs, bound, mode: str) -> dict:
    """Exact sign-flip counterpart of the bootstrap TOST / NI decision.

    For equivalence, the two one-sided boundaries are tested by shifting
    the paired differences to the boundary (``d + bound`` for
    ``H0: mean <= -bound`` and ``d - bound`` for ``H0: mean >= +bound``)
    and applying the one-sided sign-flip test; TOST takes the maximum.
    Non-inferiority only needs the lower boundary.  This is a companion:
    the pre-registered decision remains the bootstrap CI rule.
    """
    empty = {
        "sign_flip_p": float("nan"),
        "sign_flip_p_lower": float("nan"),
        "sign_flip_p_upper": float("nan"),
        "sign_flip_method": "none",
        "sign_flip_min_p": float("nan"),
        "sign_flip_valid": False,
    }
    d = _flatten_float(diffs)
    bound = _float_or_nan(bound)
    if d.numel() == 0 or not bool(torch.isfinite(d).all()) \
            or not math.isfinite(bound) or bound <= 0.0:
        return empty
    lower = sign_flip_test(d + bound, "greater")
    if mode == "equivalence":
        upper = sign_flip_test(d - bound, "less")
        if not (lower["valid"] and upper["valid"]):
            return empty
        return {
            "sign_flip_p": max(lower["p_value"], upper["p_value"]),
            "sign_flip_p_lower": lower["p_value"],
            "sign_flip_p_upper": upper["p_value"],
            "sign_flip_method": lower["method"],
            "sign_flip_min_p": max(lower["min_attainable_p"],
                                   upper["min_attainable_p"]),
            "sign_flip_valid": True,
        }
    if not lower["valid"]:
        return empty
    return {
        "sign_flip_p": lower["p_value"],
        "sign_flip_p_lower": lower["p_value"],
        "sign_flip_p_upper": float("nan"),
        "sign_flip_method": lower["method"],
        "sign_flip_min_p": lower["min_attainable_p"],
        "sign_flip_valid": True,
    }
