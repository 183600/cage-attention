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
  targets, and ``f1_weighted`` weights each class by its support.
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
  hypotheses.
* Statistical significance is not practical significance: every
  hypothesis pre-registers a **SESOI** (smallest effect size of interest)
  per metric, and ``seeds_for_power`` is the *planning* counterpart of the
  post-hoc ``mde80`` - it says how many seeds a contrast of practical
  interest would need at 80% power, so the design can be sized before it
  is run instead of discovered to be underpowered afterwards.
* For rare positive classes (error detection after convergence), the
  ranking summaries are paired with ``selection_at_fraction``: the
  operational question "if the most doubtful fraction of answers is
  flagged/reviewed, how many errors are caught (recall) and how much
  better than prevalence is the flagged set (lift)?" is interpretable
  where a raw PR-AUC number is not.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import torch

__all__ = [
    "average_ranks",
    "benjamini_hochberg",
    "roc_auc",
    "pr_auc",
    "multiclass_metrics",
    "expected_calibration_error",
    "brier_score",
    "paired_bootstrap_ci",
    "equivalence_test",
    "bootstrap_upper_p",
    "selection_at_fraction",
    "seeds_for_power",
    "wilson_ci",
]


def _flatten_float(x) -> torch.Tensor:
    return torch.as_tensor(x).detach().reshape(-1).to(torch.float64)


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
    y = torch.as_tensor(positives).reshape(-1).to(s.device).bool()
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
    y = torch.as_tensor(positives).reshape(-1).to(s.device).bool()
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
    y = torch.as_tensor(positives).reshape(-1).to(s.device).bool()
    n = s.numel()
    if n == 0:
        return {"frac": float("nan"), "k": 0, "recall": float("nan"),
                "precision": float("nan"), "lift": float("nan"),
                "threshold": float("nan")}
    frac = min(max(float(frac), 0.0), 1.0)
    k = int(math.ceil(frac * n))
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
    """
    logits = torch.as_tensor(logits).detach()
    targets = torch.as_tensor(targets).detach().reshape(-1).long()
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    n, k = logits.shape
    keys = ["accuracy", "f1_micro", "f1_macro", "f1_weighted",
            "precision_macro", "recall_macro"]
    if n == 0:
        out = {"n": 0, "n_classes_present": 0}
        out.update({key: float("nan") for key in keys})
        out[f"top{min(int(topk), k)}_accuracy"] = float("nan")
        return out

    pred = logits.argmax(dim=-1)
    correct = pred == targets
    acc = float(correct.to(torch.float64).mean())
    topk_used = min(int(topk), k)
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
    support = tgt_counts.sum()
    f1_weighted = float((f1 * (tgt_counts / support.clamp_min(1.0))).sum())
    tp_t, fp_t, fn_t = tp.sum(), fp.sum(), fn.sum()
    f1_micro = float(2 * tp_t / (2 * tp_t + fp_t + fn_t).clamp_min(1.0))

    return {
        "n": int(n),
        "n_classes_present": int(present.sum()),
        "accuracy": acc,
        f"top{topk_used}_accuracy": topk_acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
    }


def expected_calibration_error(probs, correct, n_bins: int = 15) -> float:
    """ECE: support-weighted gap between confidence and accuracy per bin.

    ``probs`` are the top-1 softmax probabilities and ``correct`` the
    matching boolean correctness labels; ``n_bins`` equal-width confidence
    bins over [0, 1].
    """
    p = torch.as_tensor(probs).detach().reshape(-1).float().clamp(0.0, 1.0)
    c = torch.as_tensor(correct).detach().reshape(-1).to(p.device).float()
    n = p.numel()
    if n == 0:
        return float("nan")
    bin_id = (p * n_bins).long().clamp(max=n_bins - 1)
    counts = torch.bincount(bin_id, minlength=n_bins).float()
    conf = torch.bincount(bin_id, weights=p, minlength=n_bins)
    acc = torch.bincount(bin_id, weights=c, minlength=n_bins)
    nz = counts > 0
    gap = (acc[nz] / counts[nz] - conf[nz] / counts[nz]).abs()
    return float(((counts[nz] / n) * gap).sum())


def brier_score(probs, correct) -> float:
    """Brier score of the top-1 answer correctness prediction."""
    p = torch.as_tensor(probs).detach().reshape(-1).float()
    c = torch.as_tensor(correct).detach().reshape(-1).to(p.device).float()
    if p.numel() == 0:
        return float("nan")
    return float(((p - c) ** 2).mean())


def _bootstrap_means(d: torch.Tensor, n_boot: int,
                     seed: int) -> torch.Tensor:
    """Bootstrap distribution of the mean of the paired values ``d``."""
    n = d.numel()
    gen = torch.Generator(device=d.device).manual_seed(seed)
    idx = torch.randint(0, n, (n_boot, n), generator=gen, device=d.device)
    return d[idx].mean(dim=1)


def _bootstrap_summary(d: torch.Tensor, means: torch.Tensor,
                       alpha: float) -> dict:
    """Percentile CI and dispersion of a bootstrap-mean distribution."""
    lo = float(torch.quantile(means, alpha / 2.0))
    hi = float(torch.quantile(means, 1.0 - alpha / 2.0))
    n = d.numel()
    sd = float(d.std(unbiased=True)) if n > 1 else 0.0
    return {
        "mean": float(d.mean()),
        "lo": lo,
        "hi": hi,
        "n": int(n),
        "sd": sd,
        "half_width": (hi - lo) / 2.0,
        "mde80": 2.8015852 * sd / math.sqrt(n) if n else float("nan"),
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
    dev = loo.mean() - loo
    den = 6.0 * float(dev.pow(2).sum()) ** 1.5
    a = (float(dev.pow(3).sum()) / den) if den > 0.0 else 0.0

    def _adjusted(z: float):
        denom = 1.0 - a * (z0 + z)
        if abs(denom) < 1e-9:
            return None
        return float(nd.cdf(z0 + (z0 + z) / denom))

    a_lo = _adjusted(nd.inv_cdf(alpha / 2.0))
    a_hi = _adjusted(nd.inv_cdf(1.0 - alpha / 2.0))
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


def paired_bootstrap_ci(diffs, n_boot: int = 10000, seed: int = 0,
                        alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for the mean of *paired* differences.

    ``diffs`` holds one value per independent unit (e.g. the same training
    seed under two configurations).  Differencing removes the between-unit
    variance that a two-sample test would fold in, which matters with the
    small seed counts used here.  NaNs must be filtered by the caller.

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
      estimable, and for degenerate bootstrap distributions).
    """
    d = _flatten_float(diffs)
    if d.numel() == 0:
        return {"mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n": 0, "sd": float("nan"),
                "half_width": float("nan"), "mde80": float("nan"),
                "p_value": float("nan"), "sign_consistency": float("nan"),
                "alpha": float(alpha),
                "loo_mean_min": float("nan"), "loo_mean_max": float("nan"),
                "loo_sign_flip": False, "lo_bca": float("nan"),
                "hi_bca": float("nan"), "bca_agrees": None}
    n = d.numel()
    means = _bootstrap_means(d, n_boot, seed)
    out = _bootstrap_summary(d, means, alpha)
    p_lo = float((means <= 0.0).to(torch.float64).mean())
    p_hi = float((means >= 0.0).to(torch.float64).mean())
    out["p_value"] = min(
        1.0, 2.0 * (min(p_lo, p_hi) * n_boot + 1.0) / (n_boot + 1.0))
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
    out["loo_sign_flip"] = bool(
        loo.numel() > 0 and out["mean"] != 0.0
        and bool(((loo > 0.0) != (out["mean"] > 0.0)).any()))
    bca = _bca_bounds(d, means, alpha)
    if bca is None:
        out["lo_bca"] = out["hi_bca"] = float("nan")
        out["bca_agrees"] = None
    else:
        out["lo_bca"], out["hi_bca"] = bca
        percentile_excludes = out["lo"] > 0.0 or out["hi"] < 0.0
        bca_excludes = out["lo_bca"] > 0.0 or out["hi_bca"] < 0.0
        out["bca_agrees"] = bool(percentile_excludes == bca_excludes)
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
    Returns NaN fields and ``equivalent=False`` for an empty input or a
    non-positive/non-finite bound.

    The same small-sample robustness companions as in
    :func:`paired_bootstrap_ci` are attached: ``loo_within_bound`` (every
    leave-one-out mean still lies inside the bound - a single seed cannot
    create the apparent stability) and the BCa interval
    (``lo_bca``/``hi_bca``/``equivalent_bca``/``bca_agrees``), so an
    equivalence claim that is an artifact of one lucky seed or of
    percentile-interval bias is flagged at the decision point.
    """
    bound = float(bound)
    d = _flatten_float(diffs)
    out = {
        "mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
        "n": 0, "sd": float("nan"), "half_width": float("nan"),
        "mde80": float("nan"), "p_value": float("nan"),
        "sign_consistency": float("nan"), "alpha": float(alpha),
        "test": "equivalence", "bound": bound, "equivalent": False,
        "loo_mean_min": float("nan"), "loo_mean_max": float("nan"),
        "loo_within_bound": False, "lo_bca": float("nan"),
        "hi_bca": float("nan"), "equivalent_bca": None,
        "bca_agrees": None,
    }
    if d.numel() == 0 or not math.isfinite(bound) or bound <= 0.0:
        return out
    means = _bootstrap_means(d, n_boot, seed)
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


def bootstrap_upper_p(diffs, bound, n_boot: int = 10000, seed: int = 0) -> float:
    """One-sided bootstrap p-value for ``H0: mean(diffs) >= bound``.

    The decision companion of the pre-registered "certified below a bound"
    rules (e.g. HB2's near-chance gate): a small p means the resampled
    means sit clearly below the bound, so the bound is rejected as an upper
    limit on the effect.  The same ``+1`` smoothing as the two-sided
    :func:`paired_bootstrap_ci` p-value is used, so the two readouts are
    on the same scale.  Returns NaN for an empty sample or a non-finite
    bound; ``p >= 0.5`` is expected whenever the point estimate does not
    even fall below the bound (no amount of resampling certifies it).
    """
    d = _flatten_float(diffs)
    bound = float(bound)
    if d.numel() == 0 or not math.isfinite(bound):
        return float("nan")
    means = _bootstrap_means(d, n_boot, seed)
    p = float((means >= bound).to(torch.float64).mean())
    return min(1.0, (p * n_boot + 1.0) / (n_boot + 1.0))


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
    if effect is None or sd is None:
        return float("inf")
    sd = float(sd)
    effect = float(effect)
    if not math.isfinite(sd) or not math.isfinite(effect) \
            or sd <= 0.0 or effect == 0.0:
        return float("inf")
    nd = NormalDist()
    z = nd.inv_cdf(1.0 - alpha / 2.0) + nd.inv_cdf(power)
    return (z * sd / abs(effect)) ** 2


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
    out = [{"p": float("nan"), "p_adj": float("nan"), "reject": False}
           for _ in p_values]
    finite = []
    for i, p in enumerate(p_values):
        if p is None:
            continue
        fp = float(p)
        if math.isfinite(fp):
            finite.append((i, min(max(fp, 0.0), 1.0)))
    m = len(finite)
    if not m:
        return out
    order = sorted(finite, key=lambda t: (t[1], t[0]))
    adjusted: dict[int, float] = {}
    prev = 1.0
    for rank in range(m, 0, -1):
        i, p = order[rank - 1]
        value = min(prev, p * m / rank)
        adjusted[i] = value
        prev = value
    for i, p in finite:
        out[i] = {"p": p, "p_adj": adjusted[i], "reject": adjusted[i] <= q}
    return out


def wilson_ci(successes: int, total: int, z: float = 1.959964) -> dict:
    """Wilson score interval for a binomial proportion.

    Used for the parameter-free baselines, whose accuracy is a count over
    a finite evaluation stream: reporting ``accuracy`` without this
    interval would hide that a rule can only be checked to +-binomial
    noise at the evaluated sample size.
    """
    if total <= 0:
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0}
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denom
    lo = min(p, max(0.0, center - half))
    hi = max(p, min(1.0, center + half))
    return {"p": p, "lo": lo, "hi": hi, "n": total}


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

    # BCa needs at least four points: with fewer the jackknife acceleration
    # is not estimable, so the audit must abstain (not fabricate bounds).
    tiny = paired_bootstrap_ci([1.0, 0.9, 1.1], n_boot=500, seed=0)
    check("bca_needs_four_points", tiny["bca_agrees"] is None)

    # Skewed sample: percentile CI includes 0 but BCa excludes it (one
    # dominant seed pulls the mean).  The audit must detect the gap.
    skewed = paired_bootstrap_ci([0.343, -0.674, 0.721, 0.929, 3.138],
                                 n_boot=2000, seed=0)
    check("bca_disagreement_detected",
          skewed["lo"] < 0.0 < skewed["lo_bca"]
          and skewed["bca_agrees"] is False)

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
    eq_dis = equivalence_test([-0.021, 0.0207, -0.0123, -0.0028, 0.0456],
                              0.03, n_boot=2000, seed=0)
    check("equiv_bca_disagreement",
          eq_dis["equivalent"] is True and eq_dis["equivalent_bca"] is False)
    neq = equivalence_test([2.0, 2.1, 1.9, 2.0, 2.05], 1.0,
                           n_boot=2000, seed=0)
    check("equiv_outside", neq["equivalent"] is False)
    check("equiv_outside_loo", neq["loo_within_bound"] is False)

    # Degenerate inputs must not raise and must not claim anything.
    empty = paired_bootstrap_ci([])
    check("empty_no_claim", empty["n"] == 0 and empty["bca_agrees"] is None)
    bad_bound = equivalence_test([1.0, 2.0], 0.0)
    check("invalid_bound", bad_bound["equivalent"] is False)

    # FDR: monotone adjusted p, small p-values survive, large ones do not,
    # NaN stays untestable.
    adj = benjamini_hochberg([0.001, 0.004, 0.2, 0.8, float("nan")], q=0.05)
    check("bh_reject_small", adj[0]["reject"] and adj[1]["reject"])
    check("bh_keep_large", not adj[2]["reject"] and not adj[3]["reject"])
    check("bh_nan_untestable", adj[4]["p_adj"] != adj[4]["p_adj"])

    # Wilson interval brackets the point estimate, including exact endpoints.
    w = wilson_ci(60, 100)
    check("wilson_brackets", w["lo"] <= w["p"] <= w["hi"])
    check("wilson_upper_exact", wilson_ci(4, 4)["hi"] == 1.0)
    check("wilson_lower_exact", wilson_ci(0, 4)["lo"] == 0.0)

    if failures:
        print("metrics self-test FAILED: " + ", ".join(failures))
        return 1
    print("metrics self-test: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
