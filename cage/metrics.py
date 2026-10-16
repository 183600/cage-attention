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
* Each experiment interprets a family of contrasts, so
  ``benjamini_hochberg`` controls the false-discovery rate across that
  family; an uncorrected per-contrast 95% readout would be expected to
  produce ~1 false "supported" claim per experiment made of all-null
  hypotheses.
"""

from __future__ import annotations

import math

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
    ranking, i.e. the usual average precision.  Ties between a positive
    and a negative score may be split by the arbitrary sort order, which
    makes the result a lower bound on the tie-corrected value; on
    continuous scores (entropy, gate values) this is immaterial.
    """
    s = _flatten_float(scores)
    y = torch.as_tensor(positives).reshape(-1).to(s.device).bool()
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = torch.argsort(s, descending=True)
    y = y[order].to(torch.float64)
    tp = torch.cumsum(y, dim=0)
    precision = tp / torch.arange(1, y.numel() + 1, dtype=torch.float64,
                                  device=s.device)
    return float((precision * y).sum() / n_pos)


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
      mean).
    """
    d = _flatten_float(diffs)
    n = d.numel()
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n": 0, "sd": float("nan"),
                "half_width": float("nan"), "mde80": float("nan"),
                "p_value": float("nan"), "sign_consistency": float("nan"),
                "alpha": float(alpha)}
    gen = torch.Generator(device=d.device).manual_seed(seed)
    idx = torch.randint(0, n, (n_boot, n), generator=gen, device=d.device)
    means = d[idx].mean(dim=1)
    lo = float(torch.quantile(means, alpha / 2.0))
    hi = float(torch.quantile(means, 1.0 - alpha / 2.0))
    p_lo = float((means <= 0.0).to(torch.float64).mean())
    p_hi = float((means >= 0.0).to(torch.float64).mean())
    p_two = min(1.0, 2.0 * (min(p_lo, p_hi) * n_boot + 1.0) / (n_boot + 1.0))
    sd = float(d.std(unbiased=True)) if n > 1 else 0.0
    pos = float((d > 0.0).to(torch.float64).mean())
    neg = float((d < 0.0).to(torch.float64).mean())
    sign_consistency = max(pos, neg) if pos + neg > 0.0 else float("nan")
    return {
        "mean": float(d.mean()),
        "lo": lo,
        "hi": hi,
        "n": int(n),
        "sd": sd,
        "half_width": (hi - lo) / 2.0,
        "mde80": 2.8015852 * sd / math.sqrt(n),
        "p_value": p_two,
        "sign_consistency": sign_consistency,
        "alpha": float(alpha),
    }


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
    return {"p": p, "lo": max(0.0, center - half),
            "hi": min(1.0, center + half), "n": total}
