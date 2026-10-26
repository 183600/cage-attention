"""Torch-only evaluation metrics for the associative-recall experiments.

Metric choices (and why)
------------------------
* The recall task is **single-label multi-class**: each evaluated row
  predicts exactly one token out of ``vocab``.  Value targets are drawn
  uniformly, so the class distribution is balanced by construction and
  plain accuracy is not dominated by a majority class; accuracy stays the
  primary metric.  ``multiclass_metrics`` nevertheless reports F1 with
  the averaging convention spelled out, because "F1" alone is ambiguous:
  ``f1_micro == accuracy`` for single-label classification, ``f1_macro``
  weights every *present* class equally (rare values matter as much as
  frequent ones) and is averaged over the classes that occur in the
  targets, and ``f1_weighted`` weights each class by its support.  Because
  a macro average over target-present classes excludes classes that were
  only *predicted*, ``*_macro_all_labels`` companions average over the
  union of true and predicted labels (the conventional sklearn macro
  convention), so a false-positive class contributes its zero recall
  instead of disappearing from the mean.
* Error-detection diagnostics are **binary and imbalanced** (after
  convergence, wrong answers are rare).  For scores that are supposed to
  flag errors we report both ROC-AUC and PR-AUC and treat PR-AUC as the
  headline: with a large negative population the false-positive rate
  grows so slowly that ROC-AUC can look optimistic while precision
  collapses (the extreme-imbalance regime).
* Calibration is the research object here ("calibrated" gating), so
  expected calibration error and the Brier score of the answer
  distribution are first-class metrics rather than extras.
* Comparisons across configurations are **paired over training seeds**;
  ``paired_bootstrap_ci`` quantifies the uncertainty of the mean
  difference without assuming normality (and without SciPy).  The same
  routine returns a bootstrap p-value, the design's minimal detectable
  effect at 80% power (``mde80``) and the sign consistency across seeds,
  because a CI that merely *includes* 0 is not evidence of "no effect"
  when the experiment cannot resolve the effect in the first place.
* The seed-level unit count is small (five by default), which is exactly
  the regime where the percentile bootstrap is weakest: its coverage is
  only first-order accurate and one influential seed can carry the mean.
  Every contrast therefore also reports two small-sample robustness
  companions, computed in the same call: the **jackknife**
  (leave-one-seed-out) mean range with a sign-flip / bound-flip flag, and
  the **bias-corrected accelerated (BCa)** interval, which corrects for
  median bias and skewness.  The pre-registered decision rule remains the
  percentile CI - switching it after seeing results would move the
  goalposts and invalidate the registry hash - but the companions are
  reported next to it, so a supported claim carried by a single seed, or
  reversed by the BCa correction, is flagged instead of silently trusted.
* Some pre-registered predictions are *null/equivalence* claims (a
  quantity is expected to stay within a bound, not to move).  Those are
  tested with ``equivalence_test`` (bootstrap TOST): support requires the
  95% CI of the effect to lie **inside** ``+-bound``, and the TOST
  p-value (H0: ``|effect| >= bound``) enters the same FDR family as the
  directional p-values - "not significant" on a difference test is not
  the same as "equivalent within the bound" and must never be read as
  confirmation of a null prediction.
* Each experiment interprets a family of contrasts, so
  ``benjamini_hochberg`` controls the false-discovery rate across that
  family; an uncorrected per-contrast 95% readout would be expected to
  produce ~1 false "supported" claim per experiment made of all-null
  hypotheses.  The contrasts are not independent (they share training
  seeds and one evaluation stream), so ``benjamini_yekutieli`` (valid under
  arbitrary dependence) and ``holm_bonferroni`` (family-wise error) are
  reported as multiplicity-model companions next to the pre-registered BH
  decision.  ``simes_test`` gives the corresponding family-wide omnibus
  p-value, and a BH-supported claim that fails the dependence-robust
  companions is flagged rather than presented as assumption-free.
* Statistical significance is not practical significance: every
  hypothesis pre-registers a **SESOI** (smallest effect size of interest)
  per metric, and ``seeds_for_power`` is the *planning* counterpart of the
  post-hoc ``mde80`` - it says how many seeds a contrast of practical
  interest would need at 80% power, so the design can be sized before it
  is run instead of discovered to be underpowered afterwards.  The SESOI
  itself is read through ``minimum_effect_test``: a row counts as
  practically established only when the whole 95% CI clears the bound in
  the pre-registered direction, not when the point estimate happens to
  exceed it while the interval still spans a negligible effect.
* For rare positive classes (error detection after convergence), the
  ranking summaries are paired with ``selection_at_fraction``: the
  operational question "if the most doubtful fraction of answers is
  flagged/reviewed, how many errors are caught (recall) and how much
  better than prevalence is the flagged set (lift)?" is interpretable
  where a raw PR-AUC number is not.
* At the default five training seeds the percentile bootstrap is a
  *large-sample* approximation: its p-value can fall far below the
  smallest p-value a paired design can actually produce.  Every contrast
  therefore also carries an **exact paired sign-flip randomization test**
  (``sign_flip_test``), enumerated over all ``2**n`` sign assignments for
  small n and Monte-Carlo sampled above the exact threshold.  The
  pre-registered bootstrap rule is unchanged; the exact test is a
  validity companion that exposes when a "significant" bootstrap result
  is below the design's own resolution (e.g. two-sided p < 0.05 is
  impossible with n <= 5).
* The seed-level interval conditions on the sealed evaluation stream.
  ``crossed_cluster_bootstrap_ci`` resamples training seeds **and**
  evaluation items, reporting the interval that also reflects finite-stream
  sampling; a seed-level win whose crossed interval includes 0 is flagged
  as *stream-limited* rather than generalised to the task population.
* A confidence interval for the current mean is not a forecast for a
  **replication**: ``replication_interval`` returns the t-based interval
  for a same-size future study's mean (``sqrt(2/n)``, because a new study's
  mean carries its own sampling variance) and the probability that the
  repeat keeps the registered sign.  A supported but noisy effect whose
  replication interval crosses zero is reported as *replication-uncertain*
  rather than "robust"; this is the results-analysis counterpart of the
  design/power layers, not a replacement for the pre-registered decision.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from statistics import NormalDist, StatisticsError

import torch

__all__ = [
    "average_ranks",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "holm_bonferroni",
    "simes_test",
    "roc_auc",
    "pr_auc",
    "roc_auc_ci",
    "pr_auc_ci",
    "auc_paired_bootstrap_ci",
    "multiclass_metrics",
    "expected_calibration_error",
    "brier_score",
    "paired_bootstrap_ci",
    "sign_flip_test",
    "equivalence_test",
    "noninferiority_test",
    "minimum_effect_test",
    "replication_interval",
    "bootstrap_upper_p",
    "clustered_proportion_ci",
    "crossed_cluster_bootstrap_ci",
    "evaluation_fold_audit",
    "selection_at_fraction",
    "seeds_for_power",
    "standardized_proportion",
    "wilson_ci",
    "cohens_d_paired",
    "hedges_g_paired",
    "mcnemar_test",
]


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


def _positive_integer(value):
    if isinstance(value, bool):
        return None
    # Preserve exact integers instead of routing them through float: a
    # Python/NumPy integer above 2**53 would otherwise be silently rounded
    # to a *different* count (e.g. 2**53 + 1 -> 2**53).  Keep the historical
    # fail-closed behavior for integers too large to be represented as a
    # finite float, so a JSON value such as 10**400 cannot size a loop.
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
        if exact < 1:
            return None
        try:
            float(exact)
        except (OverflowError, ValueError):
            return None
        return int(exact)
    try:
        number = float(value)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return None
    if not (math.isfinite(number) and number >= 1.0
            and number == int(number)):
        return None
    return int(number)


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
    if d.numel() == 1:
        sd = float(d.std(unbiased=False))
    else:
        sd = float(d.std(unbiased=True))
    if sd == 0.0 or not math.isfinite(sd):
        return float("nan")
    return float(d.mean()) / sd


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
    # J correction factor.  For df=1 (n=2 paired differences) the exact
    # gamma ratio has a pole in the denominator, so the limit J=0 is used
    # instead of evaluating math.gamma(0).  ``lgamma`` keeps the remaining
    # degrees of freedom finite across the whole range.
    df = n - 1
    if df <= 1:
        return 0.0
    log_j = (math.lgamma(df / 2.0) - 0.5 * math.log(df / 2.0)
             - math.lgamma((df - 1.0) / 2.0))
    return float(d_c) * math.exp(log_j)


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
    if n_bins is None or not bool(torch.isfinite(p).all()) \
            or not bool(torch.isfinite(c).all()):
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
    counts = torch.bincount(bin_id, minlength=n_bins).to(torch.float64)
    conf = torch.bincount(bin_id, weights=p, minlength=n_bins)
    acc = torch.bincount(bin_id, weights=c, minlength=n_bins)
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
                "cohens_d": float("nan"), "hedges_g": float("nan"),
                "bootstrap_below_sign_flip_floor": False,
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
                "cohens_d": float("nan"), "hedges_g": float("nan"),
                "bootstrap_below_sign_flip_floor": False,
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
                "cohens_d": float("nan"), "hedges_g": float("nan"),
                "bootstrap_below_sign_flip_floor": False,
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
                "cohens_d": float("nan"), "hedges_g": float("nan"),
                "bootstrap_below_sign_flip_floor": False,
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
    out["cohens_d"] = cohens_d_paired(d)
    out["hedges_g"] = hedges_g_paired(d)
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
    if not math.isfinite(t):
        return 0.0 if t < 0.0 else 1.0
    if not (math.isfinite(df) and df > 0.0):
        return float("nan")
    x = df / (df + t * t)
    tail = 0.5 * _regularized_incomplete_beta(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0.0 else tail


def _student_t_ppf(p: float, df: float) -> float:
    """Quantile of Student's t (bisection on the monotone CDF).

    ``p`` is restricted to ``(0, 1)``; the symmetry of the t distribution
    lets the search run on the upper half only.  The very-large-``df``
    branch avoids spending iterations on a distribution that is normal to
    machine precision.
    """
    if not (0.0 < p < 1.0) or not (math.isfinite(df) and df > 0.0):
        return float("nan")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -_student_t_ppf(1.0 - p, df)
    if df > 1e6:
        return NormalDist().inv_cdf(p)
    lo, hi = 0.0, 1.0
    while _student_t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:
            return float("inf")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-14 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


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
        exact_total = int(operator.index(total))
    except (TypeError, ValueError, OverflowError):
        exact_total = None
    try:
        successes = float(successes)
        total = float(total)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0}
    if not (math.isfinite(successes) and math.isfinite(total)) or total <= 0:
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0}
    # The multiplier must be a finite positive real; NaN would make the
    # interval silently collapse to the point estimate, and a boolean is
    # a flag rather than a z quantile.
    z = _float_or_nan(z)
    if not (math.isfinite(z) and z > 0.0):
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": int(total)}
    # Counts must be whole samples; accepting e.g. 0.5/2 would report a
    # binomial interval for a quantity that is not a proportion of trials.
    if successes != round(successes) or total != round(total) \
            or successes < 0 or successes > total:
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": int(total)}
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denom
    lo = min(p, max(0.0, center - half))
    hi = max(p, min(1.0, center + half))
    return {"p": p, "lo": lo, "hi": hi,
            "n": int(exact_total) if exact_total is not None else int(total)}


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

    Counts must be non-negative integers; malformed input returns a
    ``valid=False`` payload with NaN fields instead of raising, matching
    ``wilson_ci``'s contract so a reporting companion can abstain.
    """
    def _as_vector(values):
        try:
            raw = torch.as_tensor(values)
        except (TypeError, ValueError, RuntimeError):
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
    if first_error:
        out["reason"] = first_error
        return out
    if n is None or s.numel() != n.numel():
        out["reason"] = "successes and totals must have the same length"
        return out
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
            out["reason"] = f"reference_weights: {why_w}"
            return out
        if w.numel() != s.numel():
            out["reason"] = (
                "reference_weights must have one entry per stratum")
            return out
        if bool((w < 0).any()):
            out["reason"] = "reference_weights must be non-negative"
            return out
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
    rate_f = float(rate)
    lo = max(0.0, rate_f - z * se)
    hi = min(1.0, rate_f + z * se)
    out.update({
        "rate": rate_f, "se": se, "lo": lo, "hi": hi,
        "total_success": int(used_success.item()),
        "total_n": int(used_total.item()),
        "pooled_rate": (float(used_success / used_total)
                        if float(used_total) > 0.0 else float("nan")),
        "effective_n": float(effective_n),
        "min_stratum_n": int(counts_used.min().item()),
        "strata": [
            {"id": i,
             "successes": int(s[i].item()), "total": int(n[i].item()),
             "rate": (float(p[i]) if n[i] > 0 else float("nan")),
             "reference_weight": float(w[i]),
             "used": bool(usable[i])}
            for i in range(int(s.numel()))
        ],
        "valid": True, "reason": "",
    })
    return out


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
    if n.numel() == 1 and s.numel() > 1:
        n = n.expand_as(s).clone()
    if s.numel() == 1 and n.numel() > 1:
        s = s.expand_as(n).clone()
    if s.numel() != n.numel() or s.numel() == 0:
        empty["reason"] = "empty or mismatched inputs"
        return empty
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
    if not (bool(torch.isfinite(s.sum())) and bool(torch.isfinite(n.sum()))):
        # Each count is finite but their pooled totals can still overflow
        # float64; the integer readout below would then raise on ``inf``.
        empty["reason"] = "pooled success/total counts overflow"
        return empty
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
    total_success = int(round(float(s.sum().item())))
    total_n = int(round(float(n.sum().item())))
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
    tol = 1e-12 * max(1.0, abs_obs)
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


def _self_test() -> int:
    """Deterministic self-checks of the inference machinery.

    The statistics are the decision layer of every experiment, so their
    behaviour is pinned here on synthetic inputs with known answers (no
    SciPy, no dataset needed).  Run with ``python3 -m cage.metrics``.
    """
    failures = []

    def check(name: str, cond: bool) -> None:
        if not cond:
            failures.append(name)

    # Well-separated positive effects: CI excludes 0, no single seed needed.
    ci = paired_bootstrap_ci([1.0, 1.1, 0.9, 1.2, 1.0], n_boot=2000, seed=0)
    check("positive_ci_excludes_0", ci["lo"] > 0.0)
    check("positive_loo_stable", ci["loo_sign_flip"] is False)
    check("positive_bca_available", math.isfinite(ci["lo_bca"]))
    check("positive_bca_agrees", ci["bca_agrees"] is True)

    # Exact paired sign-flip: all five differences positive has an exact
    # one-sided p of 1/32 and a two-sided p of 2/32.  The bootstrap p must
    # not be read as if it had that resolution - exposing the gap is the
    # point of the companion.
    sf = sign_flip_test([1.0, 1.0, 1.0, 1.0, 1.0])
    check("sign_flip_exact", sf["valid"] and sf["method"] == "exact")
    check("sign_flip_greater_floor",
          abs(sf["p_values"]["greater"] - 1.0 / 32.0) < 1e-12)
    check("sign_flip_less_one",
          abs(sf["p_values"]["less"] - 1.0) < 1e-12)
    check("sign_flip_two_sided_floor",
          abs(sf["p_values"]["two-sided"] - 2.0 / 32.0) < 1e-12)
    check("sign_flip_below_floor",
          ci["sign_flip_p_two_sided"] > ci["p_value"])
    four = sign_flip_test([1.0, 1.0, 1.0, 1.0, -1.0])
    check("sign_flip_four_of_five",
          abs(four["p_values"]["greater"] - 6.0 / 32.0) < 1e-12)
    check("sign_flip_symmetric_two_sided",
          sign_flip_test([1.0] * 4)["p_values"]["two-sided"] == 2.0 / 16.0)
    mc = sign_flip_test(list(range(1, 21)), n_random=2000, max_exact=10)
    mc2 = sign_flip_test(list(range(1, 21)), n_random=2000, max_exact=10)
    check("sign_flip_monte_carlo_deterministic",
          mc["valid"] and mc["method"] == "monte-carlo"
          and mc["p_value"] == mc2["p_value"])
    check("sign_flip_monte_carlo_resolution",
          math.isfinite(mc["min_attainable_p"])
          and mc["min_attainable_p"] > 0.0)
    check("sign_flip_empty_no_claim", not sign_flip_test([])["valid"])
    check("sign_flip_nonfinite_no_claim",
          not sign_flip_test([1.0, float("nan")])["valid"])
    check("sign_flip_bad_alt_no_claim",
          not sign_flip_test([1.0, 2.0], alternative="up")["valid"])
    check("sign_flip_bad_boot_no_claim",
          not sign_flip_test([1.0, 2.0], n_random=0)["valid"])
    check("sign_flip_bad_exact_threshold_no_claim",
          not sign_flip_test([1.0, 2.0], max_exact=0)["valid"])
    check("sign_flip_bool_seed_no_claim",
          not sign_flip_test([1.0, 2.0], seed=True)["valid"])

    # BCa needs at least four points: with fewer the jackknife acceleration
    # is not estimable, so the audit must abstain (not fabricate bounds).
    tiny = paired_bootstrap_ci([1.0, 0.9, 1.1], n_boot=500, seed=0)
    check("bca_needs_four_points", tiny["bca_agrees"] is None)

    # Skewed sample: the percentile CI excludes 0 but the BCa interval
    # (standard jackknife acceleration) includes it.  The audit must detect
    # the gap rather than silently trusting the first-order interval.
    skewed = paired_bootstrap_ci([1.521, 2.38, -0.028, 1.926, -0.531],
                                 n_boot=2000, seed=0)
    check("bca_disagreement_detected",
          skewed["lo"] > 0.0 > skewed["lo_bca"]
          and skewed["bca_agrees"] is False)

    # The acceleration must be built from ``theta_bar - theta_(i)``.  This
    # sample was chosen so the percentile interval includes 0 while the
    # correctly signed BCa interval excludes it; reversing the acceleration
    # sign produces the opposite (wrong) bound.
    standard_sign = paired_bootstrap_ci(
        [0.343, -0.674, 0.721, 0.929, 3.138], n_boot=2000, seed=0)
    check("bca_standard_acceleration_sign",
          standard_sign["lo"] < 0.0 < standard_sign["lo_bca"]
          and standard_sign["bca_agrees"] is False)

    # One dominant seed with opposite-sign remainder: jackknife must flag it.
    ci2 = paired_bootstrap_ci([5.0, -1.0, -1.0, -1.0, -1.0],
                              n_boot=2000, seed=0)
    check("loo_flags_influence", ci2["loo_sign_flip"] is True)
    check("loo_range_ordered", ci2["loo_mean_min"] <= ci2["loo_mean_max"])
    check("loo_min_is_negative", ci2["loo_mean_min"] < 0.0)
    # Sign consistency is the share of non-zero differences agreeing with
    # the mean's direction, not the size of the larger vote: here a single
    # dominant seed carries the positive mean against the majority.
    check("sign_consistency_ignores_vote_share",
          abs(ci2["sign_consistency"] - 0.2) < 1e-12)
    zeros = paired_bootstrap_ci([2.0, 1.0, -0.5, 0.0, 0.0], n_boot=200)
    check("sign_consistency_excludes_zeros",
          abs(zeros["sign_consistency"] - 2.0 / 3.0) < 1e-12)
    tie = paired_bootstrap_ci([1.0, -1.0], n_boot=200)
    check("sign_consistency_undefined_at_zero_mean",
          tie["sign_consistency"] != tie["sign_consistency"])

    # Equivalence: bounded noise inside the bound, a clear shift outside.
    eq = equivalence_test([0.1, -0.05, 0.02, 0.0, 0.08], 0.5,
                          n_boot=2000, seed=0)
    check("equiv_inside", eq["equivalent"] is True)
    check("equiv_loo_within", eq["loo_within_bound"] is True)
    # Equivalence claim whose BCa interval pokes out of the bound: the
    # companion must flag the disagreement with the percentile verdict.
    eq_dis = equivalence_test([0.0456, 0.0213, -0.0476, 0.0053, -0.0723],
                              0.05, n_boot=2000, seed=0)
    check("equiv_bca_disagreement",
          eq_dis["equivalent"] is True and eq_dis["equivalent_bca"] is False)
    neq = equivalence_test([2.0, 2.1, 1.9, 2.0, 2.05], 1.0,
                           n_boot=2000, seed=0)
    check("equiv_outside", neq["equivalent"] is False)
    check("equiv_outside_loo", neq["loo_within_bound"] is False)

    # Non-inferiority: the returned bound stays a positive magnitude (the
    # decision threshold is its negation), so downstream consumers can pass
    # it straight back in.  A clearly worse arm must be refuted.
    ni = noninferiority_test([0.1, -0.05, 0.02, 0.0, 0.08], 0.5,
                             n_boot=2000, seed=0)
    check("noninf_inside", ni["noninferior"] is True)
    check("noninf_bound_positive", ni["bound"] == 0.5)
    check("noninf_loo_above", ni["loo_above_bound"] is True)
    ni_bad = noninferiority_test([-2.0, -2.1, -1.9, -2.0, -2.05], 1.0,
                                 n_boot=2000, seed=0)
    check("noninf_outside", ni_bad["noninferior"] is False)
    check("noninf_outside_loo", ni_bad["loo_above_bound"] is False)
    check("equiv_has_sign_flip", eq["sign_flip_valid"] is True)
    check("noninf_has_sign_flip", ni["sign_flip_valid"] is True)
    check("sign_flip_tost_bounded",
          0.0 <= eq["sign_flip_p"] <= 1.0
          and 0.0 <= ni["sign_flip_p"] <= 1.0)

    # Minimum-effect (SESOI) test: a point estimate above the bound is not
    # enough; the interval must clear it.  A tight sample far above the
    # bound establishes the practical effect, while a sample whose mean
    # exceeds the bound but whose interval straddles it does not (the
    # point-estimate-only failure mode this companion exists to expose).
    met_clear = minimum_effect_test([0.60, 0.65, 0.62, 0.58, 0.61], 0.5,
                                    direction="+", n_boot=2000, seed=0)
    check("met_clear_established",
          met_clear["beyond_sesoi"] is True
          and met_clear["loo_beyond_sesoi"] is True
          and met_clear["mean"] > met_clear["bound"])
    met_spans = minimum_effect_test([1.0, 0.1, 0.2, 0.0, 0.1], 0.2,
                                    direction="+", n_boot=2000, seed=0)
    check("met_point_estimate_only_not_established",
          abs(met_spans["mean"] - 0.28) < 1e-12
          and met_spans["mean"] > met_spans["bound"]
          and met_spans["lo"] <= met_spans["bound"]
          and met_spans["beyond_sesoi"] is False)
    met_neg = minimum_effect_test([-0.60, -0.65, -0.62, -0.58, -0.61], 0.5,
                                  direction="-", n_boot=2000, seed=0)
    check("met_negative_direction_established",
          met_neg["beyond_sesoi"] is True
          and met_neg["hi"] < -met_neg["bound"])
    met_neg_wrong = minimum_effect_test([0.60, 0.65, 0.62, 0.58, 0.61], 0.5,
                                        direction="-", n_boot=2000, seed=0)
    check("met_wrong_direction_not_established",
          met_neg_wrong["beyond_sesoi"] is False)
    check("met_invalid_bound_no_claim",
          minimum_effect_test([1.0, 2.0], 0.0)["beyond_sesoi"] is False
          and math.isnan(minimum_effect_test([1.0, 2.0], 0.0)["p_value"]))
    check("met_invalid_direction_no_claim",
          minimum_effect_test([1.0, 2.0], 0.5,
                              direction="up")["beyond_sesoi"] is False)
    check("met_single_unit_no_claim",
          minimum_effect_test([1.0], 0.5)["n"] == 1
          and minimum_effect_test([1.0], 0.5)["beyond_sesoi"] is False)
    check("met_nonfinite_no_claim",
          minimum_effect_test([1.0, float("nan")], 0.5)["beyond_sesoi"]
          is False)
    check("met_zero_boot_no_claim",
          minimum_effect_test([1.0, 2.0, 3.0], 0.5,
                              n_boot=0)["beyond_sesoi"] is False)

    # Replication interval: a same-size future study, not the CI of the
    # current mean.  The t quantile is pinned against the standard table
    # (df=4 -> 2.776445...); a tight effect replicates, a noisy one does
    # not even though its mean is positive, and the single-seed interval is
    # wider than the same-size-study interval.
    rep_strong = replication_interval([0.20, 0.19, 0.21, 0.18, 0.22],
                                      direction="+")
    check("rep_t_quantile",
          rep_strong["valid"]
          and abs(rep_strong["t_crit"] - 2.7764451051977987) < 1e-12)
    check("rep_tight_replicates",
          rep_strong["replicates"] is True and rep_strong["lo"] > 0.0
          and rep_strong["prob_replicates"] > 0.99)
    check("rep_single_wider",
          (rep_strong["hi_single"] - rep_strong["lo_single"])
          > (rep_strong["hi"] - rep_strong["lo"]))
    rep_noisy = replication_interval([0.02, 0.03, 0.04, 0.05, 0.15],
                                     direction="+")
    check("rep_noisy_mean_positive_but_not_replicated",
          rep_noisy["mean"] > 0.0 and rep_noisy["replicates"] is False
          and rep_noisy["lo"] < 0.0 < rep_noisy["hi"]
          and rep_noisy["prob_replicates"] < 0.975)
    rep_neg = replication_interval([-0.20, -0.19, -0.21, -0.18, -0.22],
                                   direction="-")
    check("rep_negative_direction", rep_neg["replicates"] is True
          and rep_neg["hi"] < 0.0)
    rep_eq = replication_interval([0.01, -0.01, 0.02, -0.02, 0.0],
                                  mode="equivalence", bound=0.05)
    check("rep_equivalence_replicates", rep_eq["replicates"] is True)
    rep_shift = replication_interval([0.20, 0.19, 0.21, 0.18, 0.22],
                                     mode="equivalence", bound=0.05)
    check("rep_equivalence_shift_not_replicated",
          rep_shift["replicates"] is False)
    rep_ni = replication_interval([0.01, -0.01, 0.02, -0.02, 0.0],
                                  mode="noninferiority", bound=0.05)
    check("rep_noninferiority_replicates", rep_ni["replicates"] is True)
    check("rep_small_n_abstains",
          replication_interval([0.1, 0.2])["valid"] is False
          and replication_interval([0.1, 0.2, 0.3])["valid"] is True)
    check("rep_invalid_inputs_abstain",
          replication_interval([0.1, float("nan"), 0.2])["valid"] is False
          and replication_interval([0.1, 0.2, 0.3],
                                   alpha=2.0)["valid"] is False
          and replication_interval([0.1, 0.2, 0.3],
                                   direction="up")["valid"] is False
          and replication_interval([0.1, 0.2, 0.3],
                                   mode="equivalence",
                                   bound=0.0)["valid"] is False
          and replication_interval([0.1, 0.2, 0.3],
                                   mode="equivalence",
                                   bound=True)["valid"] is False
          and replication_interval([0.1, 0.2, 0.3],
                                   mode="up")["valid"] is False)

    # Degenerate inputs must not raise and must not claim anything.
    empty = paired_bootstrap_ci([])
    check("empty_no_claim", empty["n"] == 0 and empty["bca_agrees"] is None)
    bad_bound = equivalence_test([1.0, 2.0], 0.0)
    check("invalid_bound", bad_bound["equivalent"] is False)
    zero_boot = paired_bootstrap_ci([1.0, 2.0, 3.0], n_boot=0)
    check("paired_zero_boot_no_claim",
          zero_boot["n"] == 3 and math.isnan(zero_boot["lo"]))
    check("equiv_zero_boot_no_claim",
          equivalence_test([1.0, 2.0], 0.5, n_boot=0)["equivalent"] is False)
    check("noninf_zero_boot_no_claim",
          noninferiority_test([1.0, 2.0], 0.5,
                              n_boot=0)["noninferior"] is False)
    check("upper_p_zero_boot_no_claim",
          math.isnan(bootstrap_upper_p([1.0, 2.0], 0.5, n_boot=0)))
    check("upper_p_nonfinite_diff_no_claim",
          math.isnan(bootstrap_upper_p([float("nan")], 0.5, n_boot=50))
          and math.isnan(bootstrap_upper_p([float("inf")], 0.5, n_boot=50)))
    check("paired_bad_alpha_no_claim",
          math.isnan(paired_bootstrap_ci([1.0, 2.0, 3.0], alpha=2.0)["lo"]))
    check("paired_nonfinite_diff_no_claim",
          math.isnan(paired_bootstrap_ci([float("nan")], n_boot=50)["lo"])
          and math.isnan(paired_bootstrap_ci([float("inf")],
                                             n_boot=50)["p_value"]))
    check("equiv_nonfinite_diff_no_claim",
          equivalence_test([float("nan")], 0.5,
                           n_boot=50)["equivalent"] is False
          and math.isnan(equivalence_test([float("nan")], 0.5,
                                          n_boot=50)["p_value"]))
    check("noninf_nonfinite_diff_no_claim",
          noninferiority_test([float("nan")], 0.5,
                              n_boot=50)["noninferior"] is False
          and math.isnan(noninferiority_test([float("nan")], 0.5,
                                             n_boot=50)["p_value"]))

    # Mismatched rows must abstain rather than broadcast one side or raise
    # an incidental indexing error.
    check("mismatched_ranking_no_claim",
          math.isnan(roc_auc([1.0, 2.0, 3.0], [False, True]))
          and math.isnan(pr_auc([1.0, 2.0, 3.0], [False, True])))
    nan_scores = [1.0, float("nan"), 3.0]
    check("nonfinite_ranking_no_claim",
          math.isnan(roc_auc(nan_scores, [False, True, True]))
          and math.isnan(pr_auc(nan_scores, [False, True, True])))
    nan_sel = selection_at_fraction(nan_scores, [False, True, True])
    check("nonfinite_selection_no_claim",
          nan_sel["k"] == 0 and math.isnan(nan_sel["precision"]))
    nan_ci = roc_auc_ci(nan_scores, [False, True, True], n_boot=20)
    check("nonfinite_ranking_ci_no_claim", math.isnan(nan_ci["auc"]))
    nan_pair = auc_paired_bootstrap_ci(
        [1.0, float("inf"), 3.0], [1.0, 2.0, 3.0],
        [False, True, True], n_boot=20)
    check("nonfinite_paired_ranking_no_claim",
          math.isnan(nan_pair["mean_diff"]))
    for bad_boot in (0, -1, 2.5, float("nan"), "x"):
        roc_bad = roc_auc_ci([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1],
                             n_boot=bad_boot)
        pr_bad = pr_auc_ci([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1],
                           n_boot=bad_boot)
        check(f"ranking_ci_bad_boot_{bad_boot}_no_claim",
              math.isfinite(roc_bad["auc"])
              and math.isnan(roc_bad["lo"])
              and math.isfinite(pr_bad["auc"])
              and math.isnan(pr_bad["lo"]))
    paired_bad = auc_paired_bootstrap_ci(
        [0.1, 0.2, 0.8, 0.9], [0.2, 0.1, 0.7, 0.8], [0, 0, 1, 1],
        n_boot=float("nan"))
    check("paired_ranking_bad_boot_no_claim",
          math.isnan(paired_bad["lo"]))
    sel = selection_at_fraction([1.0, 2.0, 3.0], [False, True])
    check("mismatched_selection_no_claim",
          sel["k"] == 0 and math.isnan(sel["precision"]))
    nan_frac = selection_at_fraction([1.0, 2.0, 3.0], [False, True],
                                     frac=float("nan"))
    check("nan_selection_fraction_no_claim",
          nan_frac["k"] == 0 and math.isnan(nan_frac["precision"]))
    check("mismatched_calibration_no_claim",
          math.isnan(expected_calibration_error([0.1, 0.2], [True]))
          and math.isnan(brier_score([0.1, 0.2], [True])))
    mm_bad = multiclass_metrics(torch.randn(3, 5), torch.tensor([0]))
    check("mismatched_multiclass_no_claim",
          mm_bad["n"] == 0 and math.isnan(mm_bad["accuracy"]))
    mm_good = multiclass_metrics(torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                                 torch.tensor([1, 0]))
    check("multiclass_nll_reported", math.isfinite(mm_good["nll"]))
    mm_clamped = multiclass_metrics(
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]), torch.tensor([1, 0]),
        topk=99)
    check("multiclass_topk_clamped_to_classes",
          mm_clamped["top2_accuracy"] == 1.0)
    bad_topk_rejected = True
    for bad_topk in (0, -1, 2.5, None, True):
        try:
            multiclass_metrics(torch.randn(2, 4), torch.tensor([0, 1]),
                               topk=bad_topk)
        except ValueError:
            continue
        bad_topk_rejected = False
    check("multiclass_invalid_topk_rejected", bad_topk_rejected)
    # Row 0 predicts target 0 correctly; row 1 predicts class 2 while the
    # target is class 1.  The pinned target-present macro-F1 ignores class 2
    # (mean of 2/3 and 0), while the label-union companion charges it as a
    # zero-recall/zero-F1 class (mean of 2/3, 0, 0).
    mm_union = multiclass_metrics(
        torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]]),
        torch.tensor([0, 1]))
    check("multiclass_macro_union_is_not_target_present_macro",
          abs(mm_union["f1_macro"] - 0.5) < 1e-9
          and abs(mm_union["f1_macro_all_labels"] - 1.0 / 3.0) < 1e-9
          and abs(mm_union["recall_macro_all_labels"] - 1.0 / 3.0) < 1e-9)
    check("multiclass_union_equals_present_when_no_predicted_only",
          abs(mm_good["f1_macro"] - mm_good["f1_macro_all_labels"]) < 1e-12)
    mm_range = multiclass_metrics(torch.randn(2, 4), torch.tensor([0, 4]))
    check("multiclass_target_range_no_claim",
          mm_range["n"] == 0 and math.isnan(mm_range["accuracy"]))
    mm_neg = multiclass_metrics(torch.randn(2, 4), torch.tensor([-1, 0]))
    check("multiclass_negative_target_no_claim",
          mm_neg["n"] == 0 and math.isnan(mm_neg["accuracy"]))
    mm_bool = multiclass_metrics(torch.randn(2, 4),
                                 torch.tensor([True, False]))
    check("multiclass_boolean_target_no_claim",
          mm_bool["n"] == 0 and math.isnan(mm_bool["accuracy"]))
    mm_complex = multiclass_metrics(
        torch.tensor([[1 + 2j, 0j], [0j, 1 + 1j]]),
        torch.tensor([0, 1]))
    check("multiclass_complex_logits_no_claim",
          mm_complex["n"] == 0 and math.isnan(mm_complex["accuracy"]))
    check("brier_nonfinite_no_claim",
          math.isnan(brier_score([float("nan")], [True]))
          and math.isnan(brier_score([float("inf")], [True])))
    check("ece_invalid_bins_no_claim",
          math.isnan(expected_calibration_error([0.5], [True], n_bins=0))
          and math.isnan(expected_calibration_error([0.5], [True], n_bins=-2))
          and math.isnan(expected_calibration_error([0.5], [True], n_bins=15.5)))
    check("ece_nonfinite_prob_no_claim",
          math.isnan(expected_calibration_error([float("nan")], [True]))
          and math.isnan(expected_calibration_error([float("inf")], [True]))
          and math.isnan(expected_calibration_error([float("-inf")], [True])))

    # FDR: monotone adjusted p, small p-values survive, large ones do not,
    # NaN stays untestable.
    adj = benjamini_hochberg([0.001, 0.004, 0.2, 0.8, float("nan")], q=0.05)
    check("bh_reject_small", adj[0]["reject"] and adj[1]["reject"])
    check("bh_keep_large", not adj[2]["reject"] and not adj[3]["reject"])
    check("bh_nan_untestable", adj[4]["p_adj"] != adj[4]["p_adj"])
    gen_adj = benjamini_hochberg(
        (p for p in [0.001, 0.004, 0.2, 0.8, float("nan")]), q=0.05)
    check("bh_generator_input",
          [r["reject"] for r in gen_adj] == [r["reject"] for r in adj])

    # Dependence-robust companions.  BH rejects all four p-values at 0.05,
    # while Holm's step-down and BY's harmonic factor (c(4)=2.0833) keep
    # only the smallest.  The family is deliberately chosen so the three
    # multiplicity models disagree on the same data.
    dep_p = [0.001, 0.02, 0.03, 0.04, float("nan")]
    dep_bh = benjamini_hochberg(dep_p, q=0.05)
    dep_by = benjamini_yekutieli(dep_p, q=0.05)
    dep_holm = holm_bonferroni(dep_p, q=0.05)
    c4 = 1.0 + 1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0
    check("by_harmonic_factor_pinned",
          abs(dep_by[0]["p_adj"] - 0.004 * c4) < 1e-12)
    check("by_is_never_less_conservative",
          all(b["p_adj"] >= a["p_adj"]
              for a, b in zip(dep_bh[:4], dep_by[:4])))
    check("by_can_reject_fewer_than_bh",
          all(r["reject"] for r in dep_bh[:4])
          and sum(r["reject"] for r in dep_by[:4]) == 1
          and dep_by[0]["reject"])
    check("holm_adjusted_pinned",
          abs(dep_holm[0]["p_adj"] - 0.004) < 1e-12
          and abs(dep_holm[1]["p_adj"] - 0.06) < 1e-12)
    check("holm_step_down_rejects_only_first",
          dep_holm[0]["reject"]
          and not any(r["reject"] for r in dep_holm[1:4]))
    check("by_nan_untestable",
          dep_by[4]["p_adj"] != dep_by[4]["p_adj"]
          and not dep_by[4]["reject"])
    check("by_invalid_q_no_rejection",
          not any(r["reject"]
                  for r in benjamini_yekutieli(dep_p, q=True)))
    check("holm_invalid_q_no_rejection",
          not any(r["reject"]
                  for r in holm_bonferroni(dep_p, q=float("nan"))))
    simes = simes_test(dep_p, alpha=0.05)
    check("simes_global_p_pinned",
          simes["valid"] and simes["reject"]
          and abs(simes["p"] - 0.004) < 1e-12)
    simes_null = simes_test([0.5, 0.6], alpha=0.05)
    check("simes_global_null_not_rejected",
          simes_null["valid"] and not simes_null["reject"]
          and abs(simes_null["p"] - 0.6) < 1e-12)
    check("simes_empty_no_claim",
          not simes_test([])["valid"]
          and simes_test([float("nan")])["m"] == 0)
    check("simes_invalid_alpha_no_claim",
          not simes_test(dep_p, alpha=True)["valid"])

    # Wilson interval brackets the point estimate, including exact endpoints.
    w = wilson_ci(60, 100)
    check("wilson_brackets", w["lo"] <= w["p"] <= w["hi"])
    check("wilson_upper_exact", wilson_ci(4, 4)["hi"] == 1.0)
    check("wilson_lower_exact", wilson_ci(0, 4)["lo"] == 0.0)

    # The exact McNemar tail must stay finite past the point where a float
    # power of two overflows (n_dis > 1023).
    n_dis = 1100
    half = n_dis // 2
    mc_a = [False] * half + [True] * half
    mc_b = [True] * half + [False] * half
    mc = mcnemar_test(mc_a, mc_b)
    check("mcnemar_large_sample_exact_finite",
          mc["b"] == half and mc["c"] == half
          and math.isfinite(mc["p_exact"]) and 0.0 < mc["p_exact"] <= 1.0)
    check("wilson_invalid_counts_no_claim",
          math.isnan(wilson_ci(5, 4)["p"])
          and math.isnan(wilson_ci(-1, 4)["p"]))
    check("wilson_fractional_counts_no_claim",
          math.isnan(wilson_ci(0.5, 2)["p"])
          and math.isnan(wilson_ci(1, 2.5)["p"])
          and math.isnan(wilson_ci(float("nan"), 2)["p"]))

    # Seed-clustered proportion: identical seed accuracies evaluated on one
    # shared stream must retain the stream's measurement noise; pooling the
    # same examples as independent trials is pseudo-replication.
    five = clustered_proportion_ci([50] * 5, [100] * 5, n_boot=4000, seed=0)
    check("cluster_brackets", five["lo"] <= five["mean"] <= five["hi"])
    pooled_width = (five["pooled_wilson"]["hi"]
                    - five["pooled_wilson"]["lo"])
    check("cluster_shared_stream_wider",
          five["hi"] - five["lo"] > pooled_width)
    again = clustered_proportion_ci([50] * 5, [100] * 5, n_boot=4000, seed=0)
    check("cluster_deterministic", again["lo"] == five["lo"])
    spread = clustered_proportion_ci([20, 80, 50, 50, 50], [100] * 5,
                                     n_boot=4000, seed=0)
    check("cluster_between_seed_widens",
          spread["hi"] - spread["lo"] > five["hi"] - five["lo"])
    one = clustered_proportion_ci([50], [100], n_boot=4000, seed=0)
    check("cluster_single_unit", one["lo"] <= one["mean"] <= one["hi"])
    family = clustered_proportion_ci([50] * 5, [100] * 5, n_boot=4000,
                                     seed=0, lower_alpha=0.00625)
    check("cluster_family_lower_more_conservative", family["lo"] < five["lo"])
    empty_cluster = clustered_proportion_ci([], [])
    check("cluster_empty_no_claim",
          empty_cluster["n"] == 0 and not empty_cluster["valid"])
    bad_cluster = clustered_proportion_ci([2, 1], [1, 1])
    check("cluster_bad_counts_no_claim", not bad_cluster["valid"])
    mixed_cluster = clustered_proportion_ci([50, float("nan")], [100, 100])
    check("cluster_partial_invalid_no_claim", not mixed_cluster["valid"])
    fractional_cluster = clustered_proportion_ci([0.5, 0.5], [1, 1])
    check("cluster_fractional_counts_no_claim",
          not fractional_cluster["valid"])
    zero_boot_cluster = clustered_proportion_ci([50] * 5, [100] * 5,
                                               n_boot=0)
    check("cluster_zero_boot_no_claim", not zero_boot_cluster["valid"])
    fractional_boot_cluster = clustered_proportion_ci(
        [50] * 5, [100] * 5, n_boot=2.5)
    check("cluster_fractional_boot_no_claim",
          not fractional_boot_cluster["valid"])
    bad_alpha_cluster = clustered_proportion_ci([50] * 5, [100] * 5,
                                               lower_alpha=2.0)
    check("cluster_bad_quantile_no_claim", not bad_alpha_cluster["valid"])

    # Evaluation-stream fold audit: a uniform item effect across all folds
    # must not be called fragile, while a concentrated effect that flips
    # sign under leave-one-fold-out must be.
    fold_uniform = evaluation_fold_audit([1] * 16, [0] * 16, n_folds=4)
    check("fold_uniform_not_fragile",
          fold_uniform["valid"] and not fold_uniform["fragile"]
          and fold_uniform["overall_effect"] == 1.0)
    a_fold = [1, 1, 1, 1] + [0] * 12
    b_fold = [0, 0, 0, 0, 1, 1, 1] + [0] * 9
    fold_concentrated = evaluation_fold_audit(a_fold, b_fold, n_folds=4)
    check("fold_concentrated_flagged",
          fold_concentrated["valid"] and fold_concentrated["fragile"]
          and fold_concentrated["loo_sign_flip"])
    fold_multi = evaluation_fold_audit(
        [[1] * 16, a_fold],
        [[0] * 16, b_fold], n_folds=4)
    check("fold_multi_unit_pooled_mean",
          fold_multi["valid"]
          and abs(fold_multi["overall_effect"] - 17.0 / 32.0) < 1e-12
          and fold_multi["unit_fold_sign_flips"] == [1])
    check("fold_malformed_no_claim",
          not evaluation_fold_audit([1, 0, 1], [1, 0, 0])["valid"]
          and not evaluation_fold_audit([1] * 8, [0] * 8,
                                        n_folds=1)["valid"])

    # Crossed (seed x item) bootstrap: a uniform effect must stay exactly
    # 1, while an effect carried by a single evaluation item must widen to
    # include 0 even though every seed has the same positive mean.
    crossed_uniform = crossed_cluster_bootstrap_ci(
        [[1] * 8, [1] * 8], [[0] * 8, [0] * 8], n_boot=500, seed=0)
    check("crossed_uniform_exact",
          crossed_uniform["valid"] and crossed_uniform["mean"] == 1.0
          and crossed_uniform["lo"] == 1.0
          and crossed_uniform["hi"] == 1.0
          and crossed_uniform["sd_total"] == 0.0
          and crossed_uniform["ci_excludes_zero"] is True)
    crossed_concentrated = crossed_cluster_bootstrap_ci(
        [[1] + [0] * 7] * 4, [[0] * 8] * 4, n_boot=1000, seed=0)
    check("crossed_item_sampling_widens",
          crossed_concentrated["valid"]
          and crossed_concentrated["seed_lo"] > 0.0
          and crossed_concentrated["item_lo"] <= 0.0
          and crossed_concentrated["lo"] <= 0.0
          and crossed_concentrated["dominant_axis"] == "item")
    crossed_again = crossed_cluster_bootstrap_ci(
        [[1] + [0] * 7] * 4, [[0] * 8] * 4, n_boot=1000, seed=0)
    check("crossed_deterministic",
          crossed_again["lo"] == crossed_concentrated["lo"]
          and crossed_again["hi"] == crossed_concentrated["hi"])
    check("crossed_malformed_no_claim",
          not crossed_cluster_bootstrap_ci([1, 0, 1, 0],
                                            [0, 0, 1, 0])["valid"]
          and not crossed_cluster_bootstrap_ci([[1, 0]], [[0, 0]])["valid"]
          and not crossed_cluster_bootstrap_ci([[1, 0], [0, 1]],
                                               [[0, 1], [0, 0.5]])["valid"]
          and not crossed_cluster_bootstrap_ci([[1, 0], [1, 0]],
                                               [[0, 1]])["valid"]
          and not crossed_cluster_bootstrap_ci([[1, 0], [1, 0]],
                                               [[0, 1], [0, 1]],
                                               n_boot=0)["valid"]
          and not crossed_cluster_bootstrap_ci([[1, 0], [1, 0]],
                                               [[0, 1], [0, 1]],
                                               alpha=2.0)["valid"])

    # Post-stratification: reweighting to a common reference mix must
    # reproduce the pooled rate with self-weights, apply the reference
    # weights exactly, and drop a stratum with no target observations
    # while reporting how much reference mass was lost.
    std_pooled = standardized_proportion([9, 1], [10, 10])
    check("standardized_self_weights_pooled",
          abs(std_pooled["rate"] - 0.5) < 1e-12
          and abs(std_pooled["pooled_rate"] - 0.5) < 1e-12
          and std_pooled["coverage"] == 1.0)
    std_ref = standardized_proportion([9, 1], [10, 10], [90, 10])
    check("standardized_reference_weights",
          abs(std_ref["rate"] - 0.82) < 1e-12
          and std_ref["lo"] <= std_ref["rate"] <= std_ref["hi"])
    std_missing = standardized_proportion([9, 0], [10, 0], [100, 100])
    check("standardized_missing_stratum_renormalized",
          abs(std_missing["rate"] - 0.9) < 1e-12
          and std_missing["coverage"] == 0.5
          and std_missing["n_used"] == 1
          and std_missing["effective_n"] == 10.0
          and std_missing["min_stratum_n"] == 10)
    check("standardized_effective_n_disclosed",
          abs(std_pooled["effective_n"] - 20.0) < 1e-12)
    check("standardized_empty_no_claim",
          not standardized_proportion([], [])["valid"])
    check("standardized_all_zero_target_no_claim",
          not standardized_proportion([0, 0], [0, 0])["valid"])
    check("standardized_mismatched_no_claim",
          not standardized_proportion([1], [1, 1])["valid"])
    check("standardized_fractional_no_claim",
          not standardized_proportion([0.5, 1], [1, 1])["valid"])
    check("standardized_success_overflow_no_claim",
          not standardized_proportion([2, 0], [1, 1])["valid"])
    check("standardized_negative_no_claim",
          not standardized_proportion([-1, 1], [1, 1])["valid"])
    check("standardized_bad_reference_no_claim",
          not standardized_proportion([1, 1], [2, 2], [-1, 1])["valid"]
          and not standardized_proportion([1, 1], [2, 2], [0, 0])["valid"]
          and not standardized_proportion([1, 1], [2, 2], [1])["valid"])

    for label, kwargs in (
            ("alpha", {"sd": 1.0, "effect": 0.5, "alpha": 1.5}),
            ("power", {"sd": 1.0, "effect": 0.5, "power": -0.1}),
            ("sd", {"sd": "x", "effect": 0.5}),
            ("effect", {"sd": 1.0, "effect": "x"}),
    ):
        check(f"power_plan_bad_{label}_no_claim",
              math.isinf(seeds_for_power(**kwargs)))

    if failures:
        print("metrics self-test FAILED: " + ", ".join(failures))
        return 1
    print("metrics self-test: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
