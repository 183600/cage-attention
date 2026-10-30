# cage metrics: ranking, calibration and effect-size primitives.
# Executed into the wrapper module namespace by the loader;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _flatten_float(x) -> torch.Tensor:
    # ``dtype`` must be given to ``as_tensor``: a Python list would otherwise
    # be captured as float32 (torch's default) and the later upcast would
    # only convert already-truncated values, inventing ties in the
    # rank-based metrics and rounding every seed-level statistic.
    # A complex dtype can be *silently* cast to float64 by torch, dropping
    # the imaginary part (it only emits a warning).  A complex paired
    # difference is not a real effect: accepting its real projection would
    # fabricate a statistic the caller never supplied.  Reject it before the
    # cast, like the audit-count/split helpers do.  Tensor inputs are checked
    # directly; Python containers are converted with the target dtype (the
    # ``dtype`` argument also prevents the initial float32 capture described
    # above, and a complex container raises on that cast).
    if torch.is_tensor(x):
        if x.is_complex():
            # An empty sample makes every caller take its documented
            # abstention/NaN branch instead of silently scoring the real
            # projection.
            return torch.empty(0, dtype=torch.float64)
        return x.detach().to(dtype=torch.float64).reshape(-1)
    try:
        probe = torch.as_tensor(x)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            f"metric inputs must be real numeric values, got "
            f"{type(x).__name__}") from exc
    if probe.is_complex():
        return torch.empty(0, dtype=torch.float64)
    try:
        return torch.as_tensor(x, dtype=torch.float64).detach().reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"metric inputs must be real numeric values, got "
            f"{type(x).__name__}") from exc


def _exact_numeric_tensor(values) -> torch.Tensor:
    """A numeric tensor without the default float32 capture.

    Python float containers become float32 through a bare ``as_tensor``;
    that rounds e.g. ``1.0000000001`` to a valid-looking 0/1 label and large
    integral counts to a different value.  Integer containers keep their
    exact integer dtype; floating containers are rebuilt at float64.
    """
    if torch.is_tensor(values):
        return values.detach()
    try:
        probe = torch.as_tensor(values)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return torch.as_tensor(values, dtype=torch.float64)
    if probe.is_floating_point():
        return torch.as_tensor(values, dtype=torch.float64)
    return probe


def _binary_labels(values, device=None):
    """Flatten boolean/0-1 labels, or return ``None`` for anything else.

    The ranking and McNemar helpers used to call ``.bool()`` unconditionally:
    ``-1``/``0.5``/``2`` then silently became a positive label, fabricating
    a class the caller never named.  Only true booleans and integral 0/1
    values are accepted (``evaluation_fold_audit`` already enforces the same
    contract); malformed input makes the companion abstain instead.
    """
    try:
        raw = torch.as_tensor(values)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return None
    if raw.is_complex():
        return None
    if raw.dtype == torch.bool:
        out = raw.detach().reshape(-1)
    else:
        try:
            if torch.is_tensor(values):
                numeric = raw.detach().reshape(-1).to(torch.float64)
            else:
                # A Python float list is captured as float32 by default;
                # 0.9999999999 would then round to 1.0 and be accepted as a
                # binary label.  Rebuild the original container at float64.
                numeric = torch.as_tensor(
                    values, dtype=torch.float64).detach().reshape(-1)
        except (TypeError, ValueError, RuntimeError, OverflowError):
            return None
        if not bool(torch.isfinite(numeric).all()):
            return None
        if not bool(((numeric == 0.0) | (numeric == 1.0)).all()):
            return None
        out = numeric.eq(1.0)
    if device is not None:
        out = out.to(device)
    return out


_MAX_TORCH_INDEX = int(torch.iinfo(torch.long).max)
_MAX_EXACT_FLOAT_INT = 1 << 53


def _positive_integer(value):
    if isinstance(value, bool):
        return None
    # Preserve exact integers instead of routing them through float: a
    # Python/NumPy integer above 2**53 would otherwise be silently rounded
    # to a *different* count (e.g. 2**53 + 1 -> 2**53).  Counts size ranges,
    # bootstrap loops and ``bincount(minlength=...)``, so anything outside
    # torch's signed 64-bit index range cannot be honoured: reject it here
    # rather than letting torch fail with an overflow or attempting an
    # impossible allocation.
    try:
        exact = operator.index(value)
    except TypeError:
        exact = None
    if exact is None and isinstance(value, str):
        try:
            exact = int(value.strip(), 10)
        except (TypeError, ValueError):
            exact = None
    if exact is not None:
        if exact < 1 or exact > _MAX_TORCH_INDEX:
            return None
        return int(exact)
    try:
        number = float(value)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return None
    if not (math.isfinite(number) and 1.0 <= number <= _MAX_TORCH_INDEX
            and number == int(number)):
        return None
    return int(number)


def _exact_integer_sequence(values):
    """Flatten integer input to exact Python ints, or return ``None``.

    Counts above 2**53 cannot round-trip through float64.  Integer tensors
    and Python integer containers are therefore read exactly here; floating
    or mixed containers are left to the existing float64 validation path.
    """
    if torch.is_tensor(values):
        if (values.dtype == torch.bool or values.is_complex()
                or values.is_floating_point()):
            return None
        try:
            flat = values.detach().reshape(-1).cpu().tolist()
        except (TypeError, ValueError, RuntimeError, OverflowError):
            return None
        return flat if isinstance(flat, list) else [flat]
    if isinstance(values, bool):
        return None
    try:
        return [operator.index(values)]
    except (TypeError, ValueError, OverflowError):
        pass
    if not isinstance(values, (list, tuple)):
        return None
    pending = list(reversed(values))
    exact = []
    while pending:
        item = pending.pop()
        if isinstance(item, (list, tuple)):
            pending.extend(reversed(item))
            continue
        if isinstance(item, bool):
            return None
        try:
            exact.append(operator.index(item))
        except (TypeError, ValueError, OverflowError):
            return None
    return exact


_TORCH_SEED_MIN = -(1 << 63)
_TORCH_SEED_MAX = (1 << 64) - 1


def _bootstrap_seed(seed) -> int:
    """Validate a resampling seed instead of silently truncating it.

    ``int(True)``/``int(0.5)`` would alias a boolean or fractional request to
    a different stream, and ``torch.Generator.manual_seed`` otherwise raises
    an incidental ``RuntimeError``.  Bootstrap seeds are part of the
    reproducibility contract, so translate every non-integral value into the
    documented ``ValueError`` before drawing any replicate.

    ``torch.Generator.manual_seed`` stores seeds in a 64-bit field and raises
    an incidental ``ValueError`` (``Overflow when unpacking long long``) for
    integers outside it.  Reject those here as well so a JSON seed such as
    ``10**400`` cannot leak torch's implementation detail from any bootstrap
    entry point.
    """
    if isinstance(seed, bool):
        raise ValueError(f"bootstrap seed must be an integer, got {seed!r}")
    try:
        seed_int = int(operator.index(seed))
    except TypeError as exc:
        raise ValueError(
            f"bootstrap seed must be an integer, got {seed!r}") from exc
    if not _TORCH_SEED_MIN <= seed_int <= _TORCH_SEED_MAX:
        raise ValueError(
            f"bootstrap seed must fit in torch's 64-bit seed range "
            f"[{_TORCH_SEED_MIN}, {_TORCH_SEED_MAX}], got {seed!r}")
    return seed_int


def _float_or_nan(value) -> float:
    """Coerce a scalar argument to float; NaN when it is not a number.

    ``_bootstrap_args_ok`` deliberately accepts numeric strings and rejects
    anything it cannot convert, but the abstention branches echo the raw
    argument (``float(alpha)``) and the valid path divides it
    (``alpha / 2.0``).  Without one normalisation the guard's "unusable
    level" verdict is an exception instead of a NaN readout, so every
    entry point normalises first and lets the guard reject the NaN.
    """
    if isinstance(value, bool):
        # ``float(True)`` is 1.0, which would turn a boolean "bound" or
        # "fraction" into a legitimate-looking numeric threshold.  Booleans
        # are semantic flags, not effect sizes or probabilities.
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return float("nan")


def average_ranks(scores: torch.Tensor) -> torch.Tensor:
    """1-based average ranks; tied values share the mean of their ranks."""
    s = _flatten_float(scores)
    n = s.numel()
    if n == 0:
        return s
    order = torch.argsort(s)
    s_sorted = s[order]
    pos = torch.arange(n, dtype=torch.float64, device=s.device)
    new_group = torch.ones(n, dtype=torch.bool, device=s.device)
    new_group[1:] = s_sorted[1:] != s_sorted[:-1]
    group = torch.cumsum(new_group.to(torch.int64), dim=0) - 1
    counts = torch.bincount(group).to(torch.float64)
    sums = torch.zeros(counts.numel(), dtype=torch.float64,
                       device=s.device).index_add_(0, group, pos)
    ranks_sorted = (sums / counts)[group] + 1.0
    ranks = torch.empty(n, dtype=torch.float64, device=s.device)
    ranks[order] = ranks_sorted
    return ranks


def roc_auc(scores, positives) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U, ties via average ranks).

    ``positives`` is a boolean tensor; ``scores`` need not be probabilities.
    Returns ``nan`` when only one class is present.
    """
    s = _flatten_float(scores)
    y = _binary_labels(positives, s.device)
    if y is None or y.numel() != s.numel() \
            or not bool(torch.isfinite(s).all()):
        return float("nan")
    n_pos = int(y.sum())
    n_neg = y.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = average_ranks(s)
    auc = (r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pr_auc(scores, positives) -> float:
    """Average precision (area under the precision-recall curve).

    Computed as ``sum_n (R_n - R_{n-1}) * P_n`` over the descending-score
    ranking.  Equal scores are grouped and consume one precision/recall
    point, so the result is the tie-corrected average precision and does
    not depend on the arbitrary sort order within a tie.
    """
    s = _flatten_float(scores)
    y = _binary_labels(positives, s.device)
    if y is None or y.numel() != s.numel() \
            or not bool(torch.isfinite(s).all()):
        return float("nan")
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = torch.argsort(s, descending=True)
    s_sorted = s[order]
    y = y[order].to(torch.float64)
    tp = torch.cumsum(y, dim=0)
    new_group = torch.ones(y.numel(), dtype=torch.bool, device=s.device)
    new_group[1:] = s_sorted[1:] != s_sorted[:-1]
    starts = torch.nonzero(new_group, as_tuple=False).reshape(-1)
    ends = torch.cat((
        starts[1:] - 1,
        torch.full((1,), y.numel() - 1, dtype=starts.dtype,
                   device=starts.device),
    ))
    tp_end = tp[ends]
    precision = tp_end / (ends + 1).to(torch.float64)
    prev_tp = torch.cat((tp_end.new_zeros(1), tp_end[:-1]))
    return float(((tp_end - prev_tp) / n_pos * precision).sum())


def selection_at_fraction(scores, positives, frac: float = 0.1) -> dict:
    """Error-capture statistics when the top ``frac`` scores are flagged.

    ``scores`` are oriented so that **higher means more likely positive**
    (an error); the top ``ceil(frac * n)`` rows are "flagged".  Returns
    ``recall`` (share of all positives that are flagged), ``precision``
    (share of flagged rows that are positive) and ``lift`` (precision
    divided by prevalence), plus the flag count and the score threshold.

    Under extreme imbalance this is the interpretable companion to PR-AUC:
    it answers the operational question ("if we review the most doubtful
    5-10% of answers, how many errors do we catch and how much of the
    reviewed set is actually wrong?") instead of asking the reader to
    calibrate a raw average-precision number against its prevalence
    baseline.
    """
    s = _flatten_float(scores)
    y = _binary_labels(positives, s.device)
    n = s.numel()
    if y is None or n == 0 or y.numel() != n \
            or not bool(torch.isfinite(s).all()):
        return {"frac": float("nan"), "k": 0, "recall": float("nan"),
                "precision": float("nan"), "lift": float("nan"),
                "threshold": float("nan")}
    raw_frac = frac
    frac = _float_or_nan(frac)
    # ``frac`` is a fraction of the evaluated rows.  Clamping an
    # out-of-range value would silently answer a different question (e.g.
    # -0.1 -> flag nothing, 1.5 -> flag everything), so abstain just like a
    # malformed value.  A NaN already follows this branch.
    if not (math.isfinite(frac) and 0.0 <= frac <= 1.0):
        return {"frac": float("nan"), "k": 0, "recall": float("nan"),
                "precision": float("nan"), "lift": float("nan"),
                "threshold": float("nan")}
    # ``frac * n`` can land a few ulps above an exact integer (0.07 * 100 is
    # 7.000000000000001 in IEEE-754), which would make ``ceil`` overcount by
    # one.  Snap only a product that is already within rounding distance of
    # a positive integer; rounding the whole product to 9 decimals first
    # would turn a genuinely positive sub-5e-10 product into 0 and contradict
    # the documented ``ceil(frac * n)``.
    product = frac * n
    nearest = round(product)
    # Snap only floating-point noise around an integer (a few ulps), not a
    # genuine excess as large as 1e-9: ``0.2000000001 * 5`` is
    # 1.0000000005 and the documented ``ceil`` must flag two rows.
    tolerance = (4.0 * math.ulp(product)
                 if math.isfinite(product) and product > 0.0 else 0.0)
    # A float32 ``0.1`` reaches this function as 0.10000000149..., so its
    # product is a few ulps above 1.0 *in float64* even though the caller
    # supplied what it regards as exactly one tenth.  The product's error is
    # bounded by ``n * spacing(frac)``, where the spacing is that of the
    # *input* dtype (``finfo.eps`` is the spacing at 1.0).  Only widen the
    # tolerance when the caller actually handed in a low-precision tensor;
    # a Python float such as 0.2000000001 keeps the strict product test.
    if (torch.is_tensor(raw_frac) and raw_frac.numel() == 1
            and raw_frac.is_floating_point()):
        try:
            eps = torch.finfo(raw_frac.dtype).eps
        except (TypeError, ValueError):
            eps = None
        if eps is not None and math.isfinite(product) and frac > 0.0:
            _, exponent = math.frexp(frac)
            spacing = math.ldexp(eps, exponent - 1)
            tolerance = max(tolerance, n * spacing)
    if nearest >= 1 and abs(product - nearest) <= tolerance:
        k = int(nearest)
    else:
        k = int(math.ceil(product))
    if k <= 0:
        return {"frac": 0.0, "k": 0, "recall": float("nan"),
                "precision": float("nan"), "lift": float("nan"),
                "threshold": float("nan")}
    order = torch.argsort(s, descending=True)
    flagged = torch.zeros(n, dtype=torch.bool, device=s.device)
    flagged[order[:k]] = True
    n_pos = int(y.sum())
    tp = int((flagged & y).sum())
    prevalence = n_pos / n
    recall = tp / n_pos if n_pos else float("nan")
    precision = tp / k
    lift = precision / prevalence if prevalence > 0.0 else float("nan")
    return {"frac": k / n, "k": k, "recall": recall,
            "precision": precision, "lift": lift,
            "threshold": float(s[order[k - 1]])}


def roc_auc_ci(scores, positives, n_boot: int = 10000, seed: int = 0,
               alpha: float = 0.05) -> dict:
    """Bootstrap confidence interval for ROC-AUC over rows.

    Rows are resampled with replacement (percentile bootstrap). Returns
    NaN for degenerate cases where only one class is present.
    """
    alpha = _float_or_nan(alpha)
    s = _flatten_float(scores)
    y = _binary_labels(positives, s.device)
    if y is None or y.numel() != s.numel() or y.numel() == 0 \
            or not bool(torch.isfinite(s).all()):
        return {"auc": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0 if y is None else int(y.numel())}
    n_pos = int(y.sum())
    n_neg = y.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        auc = roc_auc(s, y)
        return {"auc": auc, "lo": float("nan"), "hi": float("nan"),
                "n": int(y.numel())}
    auc = roc_auc(s, y)
    if not _bootstrap_args_ok(n_boot, alpha):
        return {"auc": auc, "lo": float("nan"), "hi": float("nan"),
                "n": int(y.numel())}
    n_boot = _positive_integer(n_boot)
    gen = torch.Generator(device=s.device).manual_seed(
        _bootstrap_seed(seed))
    boots = []
    for _ in range(n_boot):
        idx = torch.randint(0, s.numel(), (s.numel(),), generator=gen,
                            device=s.device)
        sb = s[idx]
        yb = y[idx]
        nb = int(yb.sum())
        nn = yb.numel() - nb
        if nb == 0 or nn == 0:
            continue
        boots.append(roc_auc(sb, yb))
    if not boots:
        return {"auc": auc, "lo": float("nan"), "hi": float("nan"),
                "n": int(y.numel())}
    bt = torch.tensor(boots, dtype=torch.float64)
    lo = float(torch.quantile(bt, alpha / 2.0))
    hi = float(torch.quantile(bt, 1.0 - alpha / 2.0))
    return {"auc": auc, "lo": lo, "hi": hi, "n": int(y.numel())}


def pr_auc_ci(scores, positives, n_boot: int = 10000, seed: int = 0,
              alpha: float = 0.05) -> dict:
    """Bootstrap confidence interval for PR-AUC (average precision)."""
    alpha = _float_or_nan(alpha)
    s = _flatten_float(scores)
    y = _binary_labels(positives, s.device)
    if y is None or y.numel() != s.numel() or y.numel() == 0 \
            or not bool(torch.isfinite(s).all()):
        return {"auc": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0 if y is None else int(y.numel())}
    n_pos = int(y.sum())
    if n_pos == 0:
        auc = pr_auc(s, y)
        return {"auc": auc, "lo": float("nan"), "hi": float("nan"),
                "n": int(y.numel())}
    auc = pr_auc(s, y)
    if not _bootstrap_args_ok(n_boot, alpha):
        return {"auc": auc, "lo": float("nan"), "hi": float("nan"),
                "n": int(y.numel())}
    n_boot = _positive_integer(n_boot)
    gen = torch.Generator(device=s.device).manual_seed(
        _bootstrap_seed(seed))
    boots = []
    for _ in range(n_boot):
        idx = torch.randint(0, s.numel(), (s.numel(),), generator=gen,
                            device=s.device)
        sb = s[idx]
        yb = y[idx]
        npp = int(yb.sum())
        if npp == 0:
            continue
        boots.append(pr_auc(sb, yb))
    if not boots:
        return {"auc": auc, "lo": float("nan"), "hi": float("nan"),
                "n": int(y.numel())}
    bt = torch.tensor(boots, dtype=torch.float64)
    lo = float(torch.quantile(bt, alpha / 2.0))
    hi = float(torch.quantile(bt, 1.0 - alpha / 2.0))
    return {"auc": auc, "lo": lo, "hi": hi, "n": int(y.numel())}


def auc_paired_bootstrap_ci(scores_a, scores_b, positives, n_boot: int = 10000,
                            seed: int = 0, alpha: float = 0.05,
                            metric: str = "roc") -> dict:
    """Paired bootstrap CI for difference of ROC/PR-AUC on same items."""
    alpha = _float_or_nan(alpha)
    sa = _flatten_float(scores_a)
    sb = _flatten_float(scores_b)
    # The three row-aligned inputs must live on one device: callers can
    # legitimately pass CUDA scores with a CPU boolean mask (or vice versa),
    # and the bootstrap indexes all three with one device-local index.
    y = _binary_labels(positives, sa.device)
    sb = sb.to(sa.device)
    if y is None or sa.numel() != sb.numel() or sa.numel() != y.numel() \
            or sa.numel() == 0 or not bool(torch.isfinite(sa).all()) \
            or not bool(torch.isfinite(sb).all()):
        return {"mean_diff": float("nan"), "lo": float("nan"),
                "hi": float("nan"),
                "n": 0 if y is None else int(sa.numel())}
    if not isinstance(metric, str):
        raise ValueError(
            f"metric must be 'roc' or 'pr', got {metric!r}")
    metric = metric.lower()
    if metric not in ("roc", "pr"):
        raise ValueError(
            f"metric must be 'roc' or 'pr', got {metric!r}: an unknown "
            f"name would silently select PR-AUC")
    def _auc(sv):
        return roc_auc(sv, y) if metric == "roc" else pr_auc(sv, y)
    a0 = _auc(sa)
    b0 = _auc(sb)
    point = a0 - b0 if (math.isfinite(a0) and math.isfinite(b0)) \
        else float("nan")
    if not _bootstrap_args_ok(n_boot, alpha):
        return {"mean_diff": point, "lo": float("nan"),
                "hi": float("nan"), "n": int(sa.numel())}
    n_boot = _positive_integer(n_boot)
    gen = torch.Generator(device=sa.device).manual_seed(
        _bootstrap_seed(seed))
    diffs = []
    n = sa.numel()
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=gen, device=sa.device)
        sbi = sa[idx]; sbj = sb[idx]; yb = y[idx]
        if metric == "roc":
            npo = int(yb.sum()); nno = yb.numel() - npo
            if npo == 0 or nno == 0:
                continue
        else:
            if int(yb.sum()) == 0:
                continue
        da = roc_auc(sbi, yb) if metric == "roc" else pr_auc(sbi, yb)
        db = roc_auc(sbj, yb) if metric == "roc" else pr_auc(sbj, yb)
        if math.isfinite(da) and math.isfinite(db):
            diffs.append(da - db)
    if not diffs:
        return {"mean_diff": point, "lo": float("nan"),
                "hi": float("nan"), "n": n}
    dt = torch.tensor(diffs, dtype=torch.float64)
    lo = float(torch.quantile(dt, alpha / 2.0))
    hi = float(torch.quantile(dt, 1.0 - alpha / 2.0))
    md = point if math.isfinite(point) else float(dt.mean())
    return {"mean_diff": float(md), "lo": lo, "hi": hi, "n": n}


def cohens_d_paired(diffs) -> float:
    """Cohen's d for paired differences: mean(diff)/sd(diff)."""
    d = _flatten_float(diffs)
    if d.numel() == 0:
        return float("nan")
    scale = float(d.abs().max())
    if not (math.isfinite(scale) and scale > 0.0):
        return float("nan")
    d = d / scale
    if d.numel() == 1:
        sd = float(d.std(unbiased=False))
    else:
        sd = float(d.std(unbiased=True))
    if sd == 0.0 or not math.isfinite(sd):
        return float("nan")
    return float(d.mean()) / sd


def _hedges_j(n: int) -> float:
    """Small-sample correction factor for Hedges' g at ``n`` pairs."""
    if n < 2:
        return float("nan")
    # For df=1 (n=2 paired differences) the exact gamma ratio has a pole
    # in the denominator, so the limit J=0 is used instead of evaluating
    # math.gamma(0).  ``lgamma`` keeps the remaining degrees of freedom
    # finite across the whole range.
    df = n - 1
    if df <= 1:
        return 0.0
    log_j = (math.lgamma(df / 2.0) - 0.5 * math.log(df / 2.0)
             - math.lgamma((df - 1.0) / 2.0))
    return math.exp(log_j)


def hedges_g_paired(diffs) -> float:
    """Hedges' g for paired differences (small-sample bias correction)."""
    d = _flatten_float(diffs)
    if d.numel() == 0:
        return float("nan")
    d_c = cohens_d_paired(d)
    if not math.isfinite(d_c):
        return float("nan")
    n = d.numel()
    if n < 2:
        return float(d_c)
    return float(d_c) * _hedges_j(n)


def _effect_size_abstention(alpha: float = float("nan")) -> dict:
    """Schema for a standardized-effect readout that has no usable CI."""
    return {
        "cohens_d": float("nan"),
        "hedges_g": float("nan"),
        "cohens_d_lo": float("nan"),
        "cohens_d_hi": float("nan"),
        "hedges_g_lo": float("nan"),
        "hedges_g_hi": float("nan"),
        "effect_size_alpha": alpha,
        "effect_size_n_boot": 0,
        "effect_size_n_boot_valid": 0,
        "effect_size_valid": False,
        "effect_size_reason": None,
    }


def paired_effect_size_ci(diffs, n_boot: int = 10000, seed: int = 0,
                          alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for paired Cohen's d and Hedges' g.

    A contrast's raw mean and its standardized effect answer different
    questions: the raw CI says how large the gain is in the metric's own
    units, while ``d = mean / paired SD`` says how large it is relative to
    the seed-to-seed noise the paired design resolves.  The existing
    ``cohens_d_paired`` / ``hedges_g_paired`` helpers reported only a point
    estimate, so a small seed count could make an effect look precise in
    standardized units even though the scale itself was estimated from the
    same handful of pairs.

    This companion resamples the paired differences with replacement,
    recomputes ``d`` on every resample, and returns the percentile interval
    (``hedges_g_lo``/``hedges_g_hi`` use the fixed-n Hedges correction).
    It is a *description of effect-size uncertainty*, not a second decision
    rule: the pre-registered inference remains the raw paired bootstrap.
    Resamples whose paired SD is zero or non-finite are excluded; if fewer
    than 90% of resamples are usable (and at least 100 replicates were
    requested), the interval abstains instead of reporting a distribution
    conditioned on unusually variable pairs or on too few order statistics.

    Returns NaN bounds and ``effect_size_valid=False`` with a ``reason`` for
    empty/non-finite input, fewer than three pairs (the Hedges correction
    is degenerate at n=2), a zero/non-finite paired SD, or an invalid
    bootstrap configuration.  ``effect_size_n_boot_valid`` records how many
    resamples entered the interval so a near-degenerate sample is visible.
    """
    alpha = _float_or_nan(alpha)
    out = _effect_size_abstention(alpha)
    out["n"] = 0
    try:
        d = _flatten_float(diffs)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        out["effect_size_reason"] = (
            "paired differences are not a real numeric sample")
        return out
    n = int(d.numel())
    out["n"] = n
    if n == 0:
        out["effect_size_reason"] = "no paired differences supplied"
        return out
    if not bool(torch.isfinite(d).all()):
        out["effect_size_reason"] = "paired differences are not finite"
        return out
    d_point = cohens_d_paired(d)
    g_point = hedges_g_paired(d)
    out["cohens_d"] = d_point
    out["hedges_g"] = g_point
    if n < 3:
        out["effect_size_reason"] = (
            "fewer than 3 paired units: the standardized effect and its "
            "Hedges correction are not estimable")
        return out
    if not (math.isfinite(d_point) and math.isfinite(g_point)):
        out["effect_size_reason"] = (
            "paired SD is zero or the standardized effect is not finite")
        return out
    if not _bootstrap_args_ok(n_boot, alpha):
        out["effect_size_reason"] = (
            "bootstrap needs a positive replicate count and an alpha in "
            "(0, 1)")
        return out
    n_boot = _positive_integer(n_boot)
    # Validate the reproducibility contract before any size shortcut: an
    # invalid seed must always be an error, not be masked by an abstention.
    boot_seed = _bootstrap_seed(seed)
    out["effect_size_n_boot"] = int(n_boot)
    if n_boot < 100:
        out["effect_size_reason"] = (
            "fewer than 100 bootstrap resamples: a percentile "
            "standardized-effect interval would be too granular")
        return out
    # The eventual resample matrix is ``n_boot x n``.  A caller may request a
    # huge replicate count to test the count plumbing; this companion must
    # abstain instead of trying to materialize an impossible matrix.
    if n_boot * n > 5_000_000:
        out["effect_size_reason"] = (
            "bootstrap grid is too large to materialize for the "
            "standardized-effect companion")
        return out
    scale = float(d.abs().max())
    if not (math.isfinite(scale) and scale > 0.0):
        out["effect_size_reason"] = (
            "paired differences have no finite scale")
        return out
    scaled = d / scale
    gen = torch.Generator(device=d.device).manual_seed(boot_seed)
    idx = torch.randint(0, n, (n_boot, n), generator=gen, device=d.device)
    samples = scaled[idx]
    means = samples.mean(dim=1)
    sds = samples.std(dim=1, unbiased=True)
    usable = torch.isfinite(means) & torch.isfinite(sds) & (sds > 0.0)
    n_valid = int(usable.sum())
    out["effect_size_n_boot_valid"] = n_valid
    if n_valid < int(math.ceil(0.9 * n_boot)):
        out["effect_size_reason"] = (
            "too few bootstrap resamples have a positive paired SD: the "
            "standardized-effect interval is not estimable")
        return out
    d_boot = means[usable] / sds[usable]
    if not bool(torch.isfinite(d_boot).all()):
        out["effect_size_reason"] = (
            "a bootstrap standardized effect is not finite")
        return out
    j = _hedges_j(n)
    if not (math.isfinite(j) and j > 0.0):
        out["effect_size_reason"] = (
            "the Hedges small-sample correction is not positive")
        return out
    d_lo = float(torch.quantile(d_boot, alpha / 2.0))
    d_hi = float(torch.quantile(d_boot, 1.0 - alpha / 2.0))
    if not (math.isfinite(d_lo) and math.isfinite(d_hi) and d_lo <= d_hi):
        out["effect_size_reason"] = (
            "the bootstrap standardized-effect quantiles are not finite")
        return out
    out.update({
        "cohens_d_lo": d_lo,
        "cohens_d_hi": d_hi,
        "hedges_g_lo": d_lo * j,
        "hedges_g_hi": d_hi * j,
        "effect_size_valid": True,
        "effect_size_reason": None,
    })
    return out


def mcnemar_test(correct_a, correct_b) -> dict:
    """McNemar's test for paired binary classification outcomes.

    ``correct_a`` and ``correct_b`` are boolean-like arrays of length n
    indicating correctness of two models on the *same* test items. Returns
    the discordant counts (b: a wrong,b right; c: a right,b wrong), the
    exact binomial p-value (mid-p not required, use standard exact for
    small b+c) and the continuity-corrected chi-square approximation.
    """
    ca = _binary_labels(correct_a)
    cb = _binary_labels(correct_b)
    if ca is None or cb is None or ca.numel() != cb.numel() \
            or ca.numel() == 0:
        return {"n": 0 if ca is None else int(ca.numel()),
                "b": 0, "c": 0, "chi2_cc": float("nan"),
                "p_chi2_cc": float("nan"), "p_exact": float("nan")}
    if cb.device != ca.device:
        cb = cb.to(ca.device)
    b = int((~ca & cb).sum())
    c = int((ca & ~cb).sum())
    n_dis = b + c
    # continuity-corrected chi-square
    if n_dis == 0:
        chi2_cc = 0.0
    else:
        chi2_cc = (abs(b - c) - 1.0) ** 2 / n_dis if (abs(b - c) - 1.0) >= 0 else 0.0
    # exact binomial: test b/(b+c)=0.5
    from math import comb
    if n_dis == 0:
        p_exact = 1.0
    else:
        k = min(b, c)
        # Keep the denominator an exact integer: ``2.0 ** n_dis`` overflows
        # to inf for n_dis > 1023 (and then raises), while Python's big-int
        # true division scales the ratio correctly (including underflow to
        # 0.0 for very small p-values).
        p_exact = (2 * sum(comb(n_dis, i) for i in range(0, k + 1))) \
            / (2 ** n_dis)
        if p_exact > 1.0:
            p_exact = 1.0
    # simple chi2 cc p-value approximation (no scipy) - use normal approx not ideal
    # just report values; experiments can interpret
    try:
        from math import erfc, sqrt
        # chi2_1 approx tail ~ erfc(sqrt(x/2))
        p_chi2_cc = erfc(sqrt(chi2_cc / 2.0))
    except Exception:
        p_chi2_cc = float("nan")
    return {"n": int(ca.numel()), "b": b, "c": c, "chi2_cc": chi2_cc,
            "p_chi2_cc": p_chi2_cc, "p_exact": p_exact}


def multiclass_metrics(logits, targets, topk: int = 5) -> dict:
    """Accuracy / top-k accuracy / explicitly named F1 variants / NLL.

    ``logits`` is ``(n, num_classes)`` and ``targets`` ``(n,)`` token ids.
    For single-label multi-class prediction ``f1_micro`` is identically
    equal to accuracy; both are returned so the identity is auditable.
    ``f1_macro`` / ``precision_macro`` / ``recall_macro`` average over the
    classes that occur in ``targets`` (classes with zero support have
    undefined recall and are excluded); predictions of never-observed
    classes still lower the recall of the true classes, so they are not
    silently dropped.

    ``f1_macro_all_labels`` / ``precision_macro_all_labels`` /
    ``recall_macro_all_labels`` are explicit companions that average over
    the union of target-present and predicted classes, i.e. the usual
    sklearn macro convention.  They differ from the pinned ``*_macro``
    values exactly when the argmax predicts a class absent from the
    targets: that class then contributes a zero-recall/zero-F1 term instead
    of being excluded.  The existing definitions are left unchanged because
    the design notes pin them; the companions only make the convention
    visible.

    ``topk`` must be a positive integer; values above the number of classes
    are clamped to that count.
    """
    raw_topk = topk
    topk = _positive_integer(raw_topk)
    if topk is None:
        raise ValueError(
            f"topk must be a positive integer, got {raw_topk!r}")
    keys = ["accuracy", "f1_micro", "f1_macro", "f1_weighted",
            "precision_macro", "recall_macro"]
    # Keep the abstention payload schema identical to the scored payload:
    # callers read the label-union macro companions unconditionally, so a
    # malformed panel must expose them as NaN rather than omit the keys.
    companion_keys = ["f1_macro_all_labels", "precision_macro_all_labels",
                      "recall_macro_all_labels"]
    keys.extend(companion_keys)

    def _abstain(k=None):
        out = {"n": 0, "n_classes_present": 0, "nll": float("nan")}
        out.update({key: float("nan") for key in keys})
        out[f"top{topk if k is None else min(topk, k)}_accuracy"] = float("nan")
        return out

    # A malformed input container is not a logit panel; abstain instead of
    # letting ``as_tensor``/tuple-unpacking escape as a TypeError from a
    # reporting metric.
    try:
        if torch.is_tensor(logits):
            logits = logits.detach()
            if logits.is_complex() or logits.dtype == torch.bool:
                return _abstain()
        else:
            probe = torch.as_tensor(logits)
            if probe.is_complex() or probe.dtype == torch.bool:
                return _abstain()
            # Python float lists would otherwise be captured as float32,
            # rounding distinct logits into the same value and changing
            # argmax/NLL.  Convert the original container at float64.
            logits = torch.as_tensor(logits, dtype=torch.float64).detach()
        if torch.is_tensor(targets):
            raw_targets = targets.detach().reshape(-1)
            if raw_targets.is_complex() or raw_targets.dtype == torch.bool:
                return _abstain()
        else:
            probe = torch.as_tensor(targets)
            if probe.is_complex() or probe.dtype == torch.bool:
                return _abstain()
            raw_targets = torch.as_tensor(
                targets, dtype=torch.float64).detach().reshape(-1)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return _abstain()
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if logits.dim() != 2:
        return _abstain()
    n, k = logits.shape
    # A non-finite logit has no ordering or likelihood: ``argmax`` would
    # still emit a class and can report perfect accuracy from an all-NaN
    # panel, while the NLL is NaN.  Abstain rather than publish an accuracy
    # the model never produced.
    if logits.is_floating_point() \
            and not bool(torch.isfinite(logits).all()):
        return _abstain(k)
    # A target is a class *index*, so it must already be an integer.  A
    # float/ complex value cast with ``.long()`` would silently truncate
    # (1.9 -> 1, -0.5 -> 0), fabricating a label the caller never named;
    # integral JSON-style floats (1.0) are accepted, everything else
    # abstains.
    # A boolean is a correctness flag, not a class index: casting it to
    # long would silently turn ``[True, False]`` into labels 1 and 0.
    bad_targets = bool(raw_targets.is_complex()
                       or raw_targets.dtype == torch.bool)
    if not bad_targets and raw_targets.is_floating_point():
        bad_targets = bool(
            not torch.isfinite(raw_targets).all()
            or not raw_targets.eq(raw_targets.round()).all())
    if bad_targets:
        return _abstain(k)
    targets = raw_targets.to(torch.long)
    if (n == 0 or targets.numel() != n
            or bool((targets < 0).any()) or bool((targets >= k).any())):
        return _abstain(k)
    # Row alignment does not imply device alignment: callers may hand in
    # CUDA logits with a CPU target/mask (or vice versa), and both top-k
    # comparison and cross-entropy require one device.
    if targets.device != logits.device:
        targets = targets.to(logits.device)

    pred = logits.argmax(dim=-1)
    correct = pred == targets
    acc = float(correct.to(torch.float64).mean())
    topk_used = min(topk, k)
    top = logits.topk(topk_used, dim=-1).indices
    topk_acc = float((top == targets.unsqueeze(1)).any(dim=1)
                     .to(torch.float64).mean())

    tgt_cpu = targets.cpu()
    pred_cpu = pred.cpu()
    tgt_counts = torch.bincount(tgt_cpu, minlength=k).to(torch.float64)
    pred_counts = torch.bincount(pred_cpu, minlength=k).to(torch.float64)
    tp = torch.bincount(pred_cpu[correct.cpu()], minlength=k).to(torch.float64)
    fp = pred_counts - tp
    fn = tgt_counts - tp

    present = tgt_counts > 0
    has_pred = (tp + fp) > 0
    has_support = (tp + fn) > 0
    precision = torch.where(has_pred, tp / (tp + fp).clamp_min(1.0),
                            torch.zeros_like(tp))
    recall = torch.where(has_support, tp / (tp + fn).clamp_min(1.0),
                         torch.zeros_like(tp))
    f1 = torch.where(precision + recall > 0,
                     2 * precision * recall / (precision + recall).clamp_min(1e-12),
                     torch.zeros_like(tp))

    f1_macro = float(f1[present].mean()) if bool(present.any()) else float("nan")
    precision_macro = (float(precision[present].mean())
                       if bool(present.any()) else float("nan"))
    recall_macro = (float(recall[present].mean())
                    if bool(present.any()) else float("nan"))
    all_labels = present | has_pred
    if bool(all_labels.any()):
        f1_macro_all_labels = float(f1[all_labels].mean())
        precision_macro_all_labels = float(precision[all_labels].mean())
        recall_macro_all_labels = float(recall[all_labels].mean())
    else:
        f1_macro_all_labels = float("nan")
        precision_macro_all_labels = float("nan")
        recall_macro_all_labels = float("nan")
    support = tgt_counts.sum()
    f1_weighted = float((f1 * (tgt_counts / support.clamp_min(1.0))).sum())
    tp_t, fp_t, fn_t = tp.sum(), fp.sum(), fn.sum()
    f1_micro = float(2 * tp_t / (2 * tp_t + fp_t + fn_t).clamp_min(1.0))
    # Do not force float32 here: a float64 logit panel (and extreme but
    # finite logits) must be scored at the precision it was recorded in.
    # ``cross_entropy`` needs a floating dtype, so only truly integer input
    # is promoted.
    nll_logits = logits if logits.is_floating_point() else logits.float()
    nll = float(torch.nn.functional.cross_entropy(nll_logits, targets))

    return {
        "n": int(n),
        "n_classes_present": int(present.sum()),
        "nll": nll,
        "accuracy": acc,
        f"top{topk_used}_accuracy": topk_acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        # Label-union macro companions: predicted-only classes contribute
        # their zero recall/F1 instead of being excluded from the average.
        "f1_macro_all_labels": f1_macro_all_labels,
        "precision_macro_all_labels": precision_macro_all_labels,
        "recall_macro_all_labels": recall_macro_all_labels,
    }


def expected_calibration_error(probs, correct, n_bins: int = 15) -> float:
    """ECE: support-weighted gap between confidence and accuracy per bin.

    ``probs`` are the top-1 softmax probabilities and ``correct`` the
    matching boolean correctness labels; ``n_bins`` equal-width confidence
    bins over [0, 1].
    """
    # A malformed container is not a calibration sample; abstain instead
    # of letting ``as_tensor`` escape as a TypeError from a reporting
    # metric (the same contract the value-level checks below follow).
    try:
        # ``_flatten_float`` asks for float64 directly, so a Python float
        # list is not first rounded through float32.  A complex input is
        # returned as an empty sample (calibration abstains).
        p = _flatten_float(probs)
        c = _flatten_float(correct).to(p.device)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return float("nan")
    n = p.numel()
    if n == 0 or c.numel() != n:
        return float("nan")
    n_bins = _positive_integer(n_bins)
    # A bin index must be exactly representable in the float64 arithmetic
    # that computes it; beyond 2**53 the index grid is coarser than the
    # requested count and no honest ECE can be reported.  (Counts outside
    # torch's int64 range are already rejected by ``_positive_integer``.)
    if (n_bins is None or n_bins > _MAX_EXACT_FLOAT_INT
            or not bool(torch.isfinite(p).all())
            or not bool(torch.isfinite(c).all())):
        return float("nan")
    # Correctness is a binary outcome; a fractional/out-of-range value is
    # not a count of correct rows and would silently distort the bin
    # accuracy (and the implied calibration error).
    if not bool(((c == 0.0) | (c == 1.0)).all()):
        return float("nan")
    # A probability outside [0, 1] is not a confidence.  Clamping it would
    # silently turn e.g. 2.0 into 1.0 and report perfect calibration for an
    # impossible input, so abstain as the non-finite branch already does.
    if not bool(((p >= 0.0) & (p <= 1.0)).all()):
        return float("nan")
    bin_id = (p * n_bins).long().clamp(max=n_bins - 1)
    # Only the bins that actually contain a sample contribute to ECE.  Use
    # their observed labels as compact group ids instead of asking
    # ``bincount`` to allocate one slot per requested bin: n_bins can be
    # large even when the evaluated panel has only a handful of rows.
    observed, inverse = torch.unique(bin_id, return_inverse=True)
    n_observed = int(observed.numel())
    counts = torch.bincount(inverse, minlength=n_observed).to(torch.float64)
    conf = torch.bincount(inverse, weights=p, minlength=n_observed)
    acc = torch.bincount(inverse, weights=c, minlength=n_observed)
    nz = counts > 0
    gap = (acc[nz] / counts[nz] - conf[nz] / counts[nz]).abs()
    return float(((counts[nz] / n) * gap).sum())


def brier_score(probs, correct) -> float:
    """Brier score of the top-1 answer correctness prediction."""
    # As in ECE, malformed containers abstain rather than raising from a
    # reporting companion.
    try:
        p = _flatten_float(probs)
        c = _flatten_float(correct).to(p.device)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return float("nan")
    if p.numel() == 0 or c.numel() != p.numel() \
            or not bool(torch.isfinite(p).all()) \
            or not bool(torch.isfinite(c).all()):
        return float("nan")
    # As with ECE, the target is a binary correctness outcome; accepting
    # e.g. 0.5 or 2.0 would report a Brier score for a different question.
    if not bool(((c == 0.0) | (c == 1.0)).all()):
        return float("nan")
    # As with ECE, a probability outside [0, 1] is not a confidence and
    # must not be scored as if it were one.
    if not bool(((p >= 0.0) & (p <= 1.0)).all()):
        return float("nan")
    return float(((p - c) ** 2).mean())


def brier_decomposition(probs, correct, n_bins: int = 15) -> dict:
    """Murphy decomposition of the top-1 correctness Brier score.

    ``brier_score`` says *how wrong* the confidence forecasts are; the
    decomposition says *why*: for the bin-constant forecast ``f_k`` and the
    observed correctness rate ``o_k`` in confidence bin ``k``,

        Brier_binned = reliability - resolution + uncertainty
        reliability  = sum_k (n_k / n) * (f_k - o_k)^2
        resolution   = sum_k (n_k / n) * (o_k - o_bar)^2
        uncertainty  = o_bar * (1 - o_bar)

    Reliability is the calibration penalty (forecast vs observed rate),
    resolution is the information gained over the base rate (how much the
    bins separate correct from wrong rows), and uncertainty is the Brier
    score of always predicting the base rate.  A low Brier can therefore
    mean "well calibrated" or merely "easy stream"; ``brier_skill_score``
    ``1 - Brier / uncertainty`` normalises for that and is NaN when the
    stream is degenerate (all correct or all wrong).

    The decomposition is exact for the *binned* forecasts.  The raw
    (unbinned) Brier is also returned, and ``within_bin`` is their
    difference: it is the part of the score that depends on the spread of
    forecasts inside a bin, not a variance component.  The function follows
    the reporting contract of :func:`expected_calibration_error` and
    :func:`brier_score`: malformed, non-finite, non-binary or out-of-range
    inputs abstain with ``valid=False`` instead of raising.
    """
    invalid = {
        "brier": float("nan"), "brier_binned": float("nan"),
        "within_bin": float("nan"), "reliability": float("nan"),
        "resolution": float("nan"), "uncertainty": float("nan"),
        "brier_skill_score": float("nan"), "n": 0, "n_bins": 0,
        "n_observed_bins": 0, "valid": False, "reason": "",
    }
    try:
        p = _flatten_float(probs)
        c = _flatten_float(correct).to(p.device)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        invalid["reason"] = "probs/correct must be real numeric values"
        return invalid
    n = p.numel()
    if n == 0 or c.numel() != n:
        invalid["reason"] = "probs and correct must be non-empty and aligned"
        invalid["n"] = int(n)
        return invalid
    n_bins = _positive_integer(n_bins)
    if (n_bins is None or n_bins > _MAX_EXACT_FLOAT_INT):
        invalid["reason"] = "n_bins must be a positive integer"
        invalid["n"] = int(n)
        return invalid
    invalid["n_bins"] = int(n_bins)
    if not bool(torch.isfinite(p).all()) \
            or not bool(torch.isfinite(c).all()):
        invalid["reason"] = "probs and correct must be finite"
        return invalid
    if not bool(((c == 0.0) | (c == 1.0)).all()):
        invalid["reason"] = "correct must contain only 0/1 outcomes"
        return invalid
    if not bool(((p >= 0.0) & (p <= 1.0)).all()):
        invalid["reason"] = "probs must lie in [0, 1]"
        return invalid

    brier = float(((p - c) ** 2).mean())
    base = float(c.mean())
    uncertainty = base * (1.0 - base)
    bin_id = (p * n_bins).long().clamp(max=n_bins - 1)
    observed, inverse = torch.unique(bin_id, return_inverse=True)
    n_observed = int(observed.numel())
    counts = torch.bincount(inverse, minlength=n_observed).to(torch.float64)
    conf = torch.bincount(inverse, weights=p, minlength=n_observed)
    acc = torch.bincount(inverse, weights=c, minlength=n_observed)
    forecast = conf / counts
    observed_rate = acc / counts
    weights = counts / n
    reliability = float((weights * (forecast - observed_rate) ** 2).sum())
    resolution = float(
        (weights * (observed_rate - base) ** 2).sum())
    binned = reliability - resolution + uncertainty
    skill = (1.0 - brier / uncertainty
             if math.isfinite(uncertainty) and uncertainty > 1e-12
             else float("nan"))
    return {
        "brier": brier, "brier_binned": binned,
        "within_bin": brier - binned,
        "reliability": reliability, "resolution": resolution,
        "uncertainty": uncertainty, "brier_skill_score": skill,
        "n": int(n), "n_bins": int(n_bins),
        "n_observed_bins": n_observed, "valid": True, "reason": "",
    }


def _bootstrap_means(d: torch.Tensor, n_boot: int,
                     seed: int) -> torch.Tensor:
    """Bootstrap distribution of the mean of the paired values ``d``."""
    n = d.numel()
    gen = torch.Generator(device=d.device).manual_seed(
        _bootstrap_seed(seed))
    idx = torch.randint(0, n, (n_boot, n), generator=gen, device=d.device)
    return d[idx].mean(dim=1)


def _finite_moments(d: torch.Tensor) -> bool:
    """Whether the sample mean and SD are representable in float64.

    Finite inputs can still overflow their sum (e.g. several ``1e308``
    differences), and a finite mean with a non-finite SD cannot produce a
    meaningful interval either.  Callers use this to abstain instead of
    reporting an infinite effect with a spuriously small bootstrap p-value.
    """
    try:
        if not math.isfinite(float(d.mean())):
            return False
        if d.numel() > 1 and not math.isfinite(float(d.std(unbiased=True))):
            return False
    except (RuntimeError, ValueError, OverflowError):
        return False
    return True


def _bootstrap_summary(d: torch.Tensor, means: torch.Tensor,
                       alpha: float) -> dict:
    """Percentile CI and dispersion of a bootstrap-mean distribution."""
    lo = float(torch.quantile(means, alpha / 2.0))
    hi = float(torch.quantile(means, 1.0 - alpha / 2.0))
    n = d.numel()
    # The sample SD (and therefore the MDE) is not estimable from a single
    # paired unit.  Reporting 0.0 there would make the design look
    # infinitely powered: ``mde80 == 0`` claims every effect is resolvable.
    sd = float(d.std(unbiased=True)) if n > 1 else float("nan")
    return {
        "mean": float(d.mean()),
        "lo": lo,
        "hi": hi,
        "n": int(n),
        "sd": sd,
        "half_width": (hi - lo) / 2.0,
        "mde80": (2.8015852 * sd / math.sqrt(n)
                  if n > 1 and math.isfinite(sd) else float("nan")),
        "alpha": float(alpha),
    }


def _jackknife_means(d: torch.Tensor) -> torch.Tensor:
    """Leave-one-out means of ``d`` (one jackknife replication per unit).

    For ``n`` paired differences the replication ``theta_(i)`` is the mean
    with unit ``i`` removed.  Its spread answers "would the conclusion
    survive dropping the most influential seed?", which the plain
    bootstrap cannot: a single outlier seed shifts the resampled
    distribution but is re-drawn in most resamples.  Returns an empty
    tensor for ``n < 2``.
    """
    n = d.numel()
    if n < 2:
        return d.new_empty(0)
    return (d.sum() - d) / (n - 1)


def _normal_quantile(p: float) -> float:
    """Standard-normal quantile without ``1 - p`` cancellation.

    ``NormalDist.inv_cdf(1 - alpha/2)`` rounds to exactly 1 for a
    tiny-but-valid alpha and raises ``StatisticsError``.  Using the
    symmetry ``z_{1-p} = -z_p`` keeps the upper tail computable whenever the
    lower-tail probability is representable.
    """
    nd = NormalDist()
    if p > 0.5:
        return -nd.inv_cdf(1.0 - p)
    return nd.inv_cdf(p)


def _bca_bounds(d: torch.Tensor, means: torch.Tensor,
                alpha: float):
    """Bias-corrected and accelerated (BCa) bootstrap interval for the mean.

    Efron's BCa end-points correct the percentile interval for median
    bias (``z0``, read off the bootstrap distribution) and for scale /
    skewness error (``a``, the jackknife acceleration) - the regime the
    seed-level contrasts live in: a handful of paired differences with no
    symmetry guarantee.  BCa is second-order accurate where the
    percentile interval is first-order, so the two disagreeing is
    diagnostic; the percentile interval stays the pre-registered primary
    readout either way.

    Returns ``(lo, hi)`` or ``None`` when the correction is undefined
    (n < 4: the acceleration from three or fewer jackknife points is not
    estimable and the correction would only add noise; degenerate bootstrap
    distribution; or an adjusted percentile outside ``(0, 1)``); callers
    then record BCa as unavailable rather than fabricating a bound.
    """
    n = d.numel()
    if n < 4:
        return None
    nd = NormalDist()
    obs = float(d.mean())
    b = int(means.numel())
    n_less = int((means < obs).sum())
    if n_less == 0:
        z0 = -6.0
    elif n_less == b:
        z0 = 6.0
    else:
        z0 = nd.inv_cdf(n_less / b)
    loo = _jackknife_means(d)
    # Standard BCa acceleration: the third central moment of
    # ``theta_bar - theta_(i)`` over the leave-one-out estimates, where
    # ``theta_bar`` is their mean (equal to the observed mean for the
    # jackknife).  The sign matters - using the LOO-minus-mean direction
    # flips ``a`` and therefore the corrected endpoints.
    dev = loo.mean() - loo
    try:
        den = 6.0 * float(dev.pow(2).sum()) ** 1.5
        num = float(dev.pow(3).sum())
    except OverflowError:
        # The jackknife acceleration is not representable for an
        # extreme-scale sample; BCa is a companion, so abstain instead of
        # aborting the whole contrast report.
        return None
    if not (math.isfinite(den) and math.isfinite(num)):
        # Either sum overflowed to infinity; the ratio would be NaN and the
        # adjusted quantiles are then undefined.
        return None
    a = (num / den) if den > 0.0 else 0.0

    def _adjusted(z: float):
        denom = 1.0 - a * (z0 + z)
        if abs(denom) < 1e-9:
            return None
        return float(nd.cdf(z0 + (z0 + z) / denom))

    try:
        z_lo = _normal_quantile(alpha / 2.0)
        z_hi = -z_lo
    except (StatisticsError, ValueError):
        return None
    a_lo = _adjusted(z_lo)
    a_hi = _adjusted(z_hi)
    if a_lo is None or a_hi is None:
        return None
    a_lo = min(max(a_lo, 1e-6), 1.0 - 1e-6)
    a_hi = min(max(a_hi, 1e-6), 1.0 - 1e-6)
    if a_lo >= a_hi:
        return None
    lo = float(torch.quantile(means, a_lo))
    hi = float(torch.quantile(means, a_hi))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return (lo, hi)


def _bootstrap_args_ok(n_boot, alpha) -> bool:
    """Whether a bootstrap has at least one replicate and a usable alpha."""
    try:
        a = float(alpha)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return False
    return (_positive_integer(n_boot) is not None
            and math.isfinite(a) and 0.0 < a < 1.0)
