# ruff: noqa: F401, F821, F822
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
* Raw-unit intervals do not say how well the *seed-to-seed scale* is
  estimated.  Every difference contrast therefore also reports a paired
  bootstrap interval for **Cohen's dz and Hedges' g**
  (``paired_effect_size_ci``), next to their point estimates.  The
  standardized interval is descriptive: it can make a magnitude read look
  precise even when the paired SD rests on only a handful of seeds, and
  abstains when the bootstrap scale is too often zero or non-finite to
  support one.
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
  impossible with n <= 5).  ``sign_flip_resolution`` and
  ``sign_flip_min_units`` expose the same floor *before* data collection,
  so the prospective design plan can require enough seeds for the
  registered alternative instead of discovering an unfalsifiable
  two-sided claim afterwards.
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
import sys
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
    "sign_flip_resolution",
    "sign_flip_min_units",
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
    "paired_effect_size_ci",
    "mcnemar_test",
]

_FLOAT_MAX = sys.float_info.max



# ---------------------------------------------------------------------------
# Implementation layout
# ---------------------------------------------------------------------------
# The implementation is split by responsibility into the fragment files
# below.  They are compiled and executed into this module's namespace in
# order, so every helper keeps its original public/private name and module
# identity for callers and tests.
_future = __import__("__future__")
_pathlib = __import__("pathlib")

_FRAGMENT_DIR = _pathlib.Path(__file__).resolve().parent
_FRAGMENTS = (
    "_metrics_ranking.py",
    "_metrics_inference.py",
    "_metrics_robustness.py",
)
for _fragment_name in _FRAGMENTS:
    _fragment_path = _FRAGMENT_DIR / _fragment_name
    exec(compile(_fragment_path.read_text(encoding="utf-8"),
                 str(_fragment_path), "exec",
                 flags=_future.annotations.compiler_flag,
                 dont_inherit=True), globals())
del _fragment_name, _fragment_path

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
