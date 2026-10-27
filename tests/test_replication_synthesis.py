"""Replication forecasting and hypothesis-level evidence synthesis.

These tests pin the two results-analysis companions added on top of the
per-contrast decision table:

* ``replication_interval`` - a same-size future study's effect is not the
  current mean's confidence interval; a significant mean can still be
  expected to reverse in a fresh run.
* ``ContrastLog.finalize``'s hypothesis synthesis - a pre-registered
  hypothesis is a conjunction, so one surviving row must not be read as
  "the hypothesis is confirmed".

Both are companions: they annotate the pre-registered FDR decision and
never replace it.
"""

from __future__ import annotations

import contextlib
import io
import math
import pathlib
import tempfile
import unittest

from cage.experiments import ContrastLog, _hypothesis_evidence_synthesis
from cage.metrics import (
    equivalence_test,
    paired_bootstrap_ci,
    replication_interval,
)


def _ci(diffs):
    ci = paired_bootstrap_ci(diffs)
    # Real contrasts carry their paired differences; finalize uses them for
    # the replication companion.
    ci["diffs"] = list(diffs)
    ci["controlled"] = {
        "controlled": True,
        "mismatches": [],
        "varying_factors": {"gate": {"a": ["calib"], "b": ["query"]}},
        "capacity": {"matched": True, "a": 10, "b": 10,
                     "delta_b_minus_a": 0},
    }
    return ci


def _finalize(tag, hypotheses, rows):
    log = ContrastLog(tag, hypotheses=hypotheses, min_seeds=3)
    with contextlib.redirect_stdout(io.StringIO()):
        for rule, hyp, metric, diffs in rows:
            log.print(rule, hyp, metric, _ci(diffs))
    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            payload = log.finalize(pathlib.Path(tmp), q=0.05)
    return log, payload, captured.getvalue()


class EffectSizeIntervalAuditTest(unittest.TestCase):
    def test_finalize_exposes_the_standardized_effect_interval(self):
        diffs = [0.20, 0.19, 0.21, 0.18, 0.22]
        _log, payload, captured = _finalize(
            "e1", ("H1a",), [("calib - none", "H1a", "accuracy", diffs)])
        row = payload["rows"][0]
        self.assertTrue(row["effect_size_valid"])
        self.assertLessEqual(row["cohens_d_lo"], row["cohens_d"])
        self.assertLessEqual(row["cohens_d"], row["cohens_d_hi"])
        self.assertLessEqual(row["hedges_g_lo"], row["hedges_g"])
        self.assertLessEqual(row["hedges_g"], row["hedges_g_hi"])
        self.assertIn("dz=", captured)
        self.assertEqual(payload["n_effect_size_ci_unavailable"], 0)


class ReplicationIntervalTest(unittest.TestCase):
    def test_t_quantile_and_interval_are_pinned(self):
        rep = replication_interval([0.20, 0.19, 0.21, 0.18, 0.22],
                                   direction="+")
        self.assertTrue(rep["valid"])
        self.assertEqual(rep["n"], 5)
        # Standard t table: t_{0.975, 4} = 2.776445...
        self.assertAlmostEqual(rep["t_crit"], 2.7764451051977987, places=12)
        # The same-size-study interval uses sqrt(2/n), the single-seed
        # prediction interval uses sqrt(1 + 1/n), so the latter is wider.
        self.assertGreater(rep["hi_single"] - rep["lo_single"],
                           rep["hi"] - rep["lo"])
        self.assertGreater(rep["lo"], 0.0)
        self.assertTrue(rep["replicates"])
        self.assertGreater(rep["prob_replicates"], 0.99)

    def test_noisy_positive_mean_is_not_expected_to_replicate(self):
        # Mean 0.058, but the between-seed spread is large enough that a
        # same-size repeat's interval crosses zero.  The current mean is
        # positive; the forecast is not.
        rep = replication_interval([0.02, 0.03, 0.04, 0.05, 0.15],
                                   direction="+")
        self.assertGreater(rep["mean"], 0.0)
        self.assertLess(rep["lo"], 0.0)
        self.assertGreater(rep["hi"], 0.0)
        self.assertFalse(rep["replicates"])
        self.assertLess(rep["prob_replicates"], 0.975)

    def test_negative_direction_requires_the_upper_bound_below_zero(self):
        rep = replication_interval([-0.20, -0.19, -0.21, -0.18, -0.22],
                                   direction="-")
        self.assertTrue(rep["replicates"])
        self.assertLess(rep["hi"], 0.0)
        # A positive sample cannot forecast a negative replication.
        wrong = replication_interval([0.20, 0.19, 0.21, 0.18, 0.22],
                                     direction="-")
        self.assertFalse(wrong["replicates"])

    def test_undirected_prediction_reports_sign_persistence(self):
        rep = replication_interval([0.20, 0.19, 0.21, 0.18, 0.22])
        self.assertTrue(rep["valid"])
        self.assertEqual(rep["replicates"], True)
        self.assertGreater(rep["prob_replicates"], 0.99)

    def test_null_prediction_modes_read_the_bound(self):
        tight = [0.01, -0.01, 0.02, -0.02, 0.0]
        eq = replication_interval(tight, mode="equivalence", bound=0.05)
        self.assertTrue(eq["replicates"])
        shift = replication_interval([0.20, 0.19, 0.21, 0.18, 0.22],
                                     mode="equivalence", bound=0.05)
        self.assertFalse(shift["replicates"])
        ni = replication_interval(tight, mode="noninferiority", bound=0.05)
        self.assertTrue(ni["replicates"])

    def test_invalid_inputs_abstain(self):
        for kwargs in (
            {"diffs": [0.1, 0.2]},  # n < 3: seed SD not estimable
            {"diffs": None},
            {"diffs": [0.1, float("nan"), 0.2]},
            {"diffs": [0.1, 0.2, 0.3], "alpha": 2.0},
            {"diffs": [0.1, 0.2, 0.3], "direction": "up"},
            {"diffs": [0.1, 0.2, 0.3], "mode": "up"},
            {"diffs": [0.1, 0.2, 0.3], "mode": "equivalence", "bound": 0.0},
            {"diffs": [0.1, 0.2, 0.3], "mode": "equivalence", "bound": True},
        ):
            with self.subTest(kwargs=kwargs):
                rep = replication_interval(kwargs.pop("diffs"), **kwargs)
                self.assertFalse(rep["valid"])
                self.assertFalse(rep["replicates"])
                self.assertTrue(math.isnan(rep["lo"]))
                self.assertTrue(rep["reason"])


class ContrastReplicationAuditTest(unittest.TestCase):
    def test_supported_effect_can_be_replication_uncertain(self):
        _, payload, _ = _finalize(
            "e2", ("H2a",),
            [("calib-none", "H2a", "accuracy_at_longest",
              [0.02, 0.03, 0.04, 0.05, 0.16])])
        row = payload["rows"][0]
        self.assertTrue(row["supports_hypothesis"])
        self.assertFalse(row["replication_replicates"])
        self.assertTrue(row["replication_uncertain"])
        self.assertEqual(payload["n_replication_uncertain"], 1)
        self.assertIn("H2a/calib-none", payload[
            "replication_uncertain_contrasts"][0])

    def test_tight_effect_is_not_replication_uncertain(self):
        _, payload, _ = _finalize(
            "e2", ("H2a",),
            [("calib-none", "H2a", "accuracy_at_longest",
              [0.05, 0.04, 0.06, 0.05, 0.05])])
        row = payload["rows"][0]
        self.assertTrue(row["supports_hypothesis"])
        self.assertTrue(row["replication_replicates"])
        self.assertFalse(row["replication_uncertain"])
        self.assertEqual(payload["n_replication_uncertain"], 0)

    def test_unsupported_effect_is_not_labelled_uncertain(self):
        _, payload, _ = _finalize(
            "e2", ("H2a",),
            [("calib-none", "H2a", "accuracy_at_longest",
              [0.001, -0.001, 0.002, -0.001, 0.0])])
        row = payload["rows"][0]
        self.assertFalse(row["supports_hypothesis"])
        self.assertFalse(row["replication_uncertain"])

    def test_null_prediction_replication_uses_the_bound(self):
        ci = equivalence_test([0.01, -0.01, 0.02, -0.02, 0.0], 0.05)
        ci["diffs"] = [0.01, -0.01, 0.02, -0.02, 0.0]
        ci["controlled"] = {
            "controlled": True, "mismatches": [],
            "varying_factors": {}, "capacity": {
                "matched": True, "a": 10, "b": 10, "delta_b_minus_a": 0}}
        log = ContrastLog("e2", hypotheses=("H2b",), min_seeds=3)
        with contextlib.redirect_stdout(io.StringIO()):
            log.print_equivalence_ci("drift", "H2b", "entropy_norm_drift", ci)
        row = log.rows[0]
        self.assertEqual(row["test"], "equivalence")
        self.assertTrue(row["replication_replicates"])
        self.assertTrue(row["replication"]["valid"])

    def test_curve_invalid_seed_is_removed_from_the_forecast(self):
        # A replicon forecast must use the same healthy-seed sample as the
        # curve sensitivity gate: a run that never learned cannot inform a
        # future replication.
        diffs = [0.05, 0.04, 0.06, 0.05, 0.20]
        ci = _ci(diffs)
        ci["seeds_used"] = [0, 1, 2, 3, 4]
        ci["curve_invalid_seeds"] = [4]
        log = ContrastLog("e2", hypotheses=("H2a",), min_seeds=3)
        with contextlib.redirect_stdout(io.StringIO()):
            log.print("calib-none", "H2a", "accuracy_at_longest", ci)
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = log.finalize(pathlib.Path(tmp), q=0.05)
        row = payload["rows"][0]
        self.assertEqual(row["replication"]["n"], 5)
        self.assertTrue(row["replication_curve_valid"]["valid"])
        self.assertEqual(row["replication_effective"]["n"], 4)
        self.assertEqual(row["replication_prob"],
                         row["replication_curve_valid"]["prob_replicates"])


class HypothesisSynthesisTest(unittest.TestCase):
    def test_malformed_rows_and_containers_do_not_abort_synthesis(self):
        # A pooled/legacy row can be a scalar or a non-mapping object.  It is
        # not evidence for any hypothesis, so the companion must skip it
        # instead of raising AttributeError from ``.get``.
        row = {"hypothesis": "H2a", "direction": "+",
               "supports_hypothesis": True}
        out = _hypothesis_evidence_synthesis(
            [None, "not-a-row", 3, row], ["H2a"])
        self.assertEqual(out["H2a"]["n_rows"], 1)
        self.assertEqual(out["H2a"]["n_confirmed"], 1)
        self.assertEqual(out["H2a"]["status"], "supported")
        empty = _hypothesis_evidence_synthesis(None, ["H2a"])
        self.assertEqual(empty["H2a"]["status"], "no_contrast")
        self.assertEqual(empty["H2a"]["n_rows"], 0)

    def test_full_conjunction_is_supported(self):
        _, payload, _ = _finalize(
            "e2", ("H2a",),
            [("longest", "H2a", "accuracy_at_longest",
              [0.05, 0.04, 0.06, 0.05, 0.05]),
             ("drop", "H2a", "accuracy_drop",
              [-0.05, -0.04, -0.06, -0.05, -0.05]),
             ("slope", "H2a", "accuracy_slope_log_len",
              [0.05, 0.04, 0.06, 0.05, 0.05])])
        synth = payload["hypothesis_synthesis"]["H2a"]
        self.assertEqual(synth["status"], "supported")
        self.assertEqual(synth["n_required"], 3)
        self.assertEqual(synth["n_confirmed"], 3)
        self.assertEqual(payload["n_hypotheses_supported"], 1)
        self.assertEqual(payload["n_hypotheses_partial"], 0)

    def test_one_surviving_row_is_not_a_confirmed_hypothesis(self):
        _, payload, captured = _finalize(
            "e2", ("H2a",),
            [("longest", "H2a", "accuracy_at_longest",
              [0.05, 0.04, 0.06, 0.05, 0.05]),
             ("drop", "H2a", "accuracy_drop",
              [-0.05, -0.04, -0.06, -0.05, -0.05]),
             ("slope", "H2a", "accuracy_slope_log_len",
              [0.001, -0.001, 0.002, -0.001, 0.0])])
        synth = payload["hypothesis_synthesis"]["H2a"]
        self.assertEqual(synth["status"], "partial")
        self.assertEqual(synth["n_confirmed"], 2)
        self.assertEqual(synth["n_required"], 3)
        self.assertEqual(synth["n_inconclusive"], 1)
        self.assertTrue(synth["unconfirmed_contrasts"])
        self.assertEqual(payload["n_hypotheses_partial"], 1)
        self.assertIn("PARTIAL evidence", captured)
        self.assertIn("H2a", captured)

    def test_a_refutation_dominates_supporting_rows(self):
        _, payload, captured = _finalize(
            "e2", ("H2a",),
            [("longest", "H2a", "accuracy_at_longest",
              [0.05, 0.04, 0.06, 0.05, 0.05]),
             ("drop", "H2a", "accuracy_drop",
              [0.05, 0.04, 0.06, 0.05, 0.05])])
        synth = payload["hypothesis_synthesis"]["H2a"]
        self.assertEqual(synth["status"], "refuted")
        self.assertEqual(synth["n_refuted"], 1)
        self.assertTrue(synth["refuted_contrasts"])
        self.assertEqual(payload["n_hypotheses_refuted"], 1)
        self.assertIn("REFUTED", captured)

    def test_null_prediction_refutation_is_counted(self):
        ci = equivalence_test([0.20, 0.19, 0.21, 0.18, 0.22], 0.05)
        ci["diffs"] = [0.20, 0.19, 0.21, 0.18, 0.22]
        ci["controlled"] = {
            "controlled": True, "mismatches": [],
            "varying_factors": {}, "capacity": {
                "matched": True, "a": 10, "b": 10, "delta_b_minus_a": 0}}
        log = ContrastLog("e2", hypotheses=("H2b",), min_seeds=3)
        with contextlib.redirect_stdout(io.StringIO()):
            log.print_equivalence_ci("drift", "H2b", "entropy_norm_drift", ci)
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = log.finalize(pathlib.Path(tmp), q=0.05)
        synth = payload["hypothesis_synthesis"]["H2b"]
        self.assertEqual(synth["status"], "refuted")
        self.assertEqual(synth["n_refuted"], 1)

    def test_undirected_hypothesis_is_not_reported_as_confirmed(self):
        _, payload, _ = _finalize(
            "e3", ("H3b",),
            [("layerscale", "H3b", "resid_norm_last_over_first",
              [0.20, 0.19, 0.21, 0.18, 0.22])])
        synth = payload["hypothesis_synthesis"]["H3b"]
        self.assertEqual(synth["status"], "undirected")
        self.assertEqual(synth["n_required"], 0)
        self.assertEqual(synth["n_descriptive_supporting"], 1)
        self.assertEqual(payload["n_hypotheses_undirected"], 1)
        self.assertEqual(payload["n_hypotheses_supported"], 0)

    def test_non_significant_undirected_row_is_still_undirected(self):
        # A descriptive/two-sided row has no registered conjunction to
        # confirm.  Whether it happens to exclude zero or not, the status is
        # "undirected" (the documented category), not an uncounted fifth
        # state.
        _, payload, _ = _finalize(
            "e3", ("H3b",),
            [("layerscale", "H3b", "resid_norm_last_over_first",
              [-0.01, 0.01, -0.02, 0.02, 0.0])])
        synth = payload["hypothesis_synthesis"]["H3b"]
        self.assertEqual(synth["status"], "undirected")
        self.assertEqual(synth["n_required"], 0)
        self.assertEqual(synth["n_descriptive_supporting"], 0)
        self.assertEqual(payload["n_hypotheses_undirected"], 1)

    def test_unrun_hypothesis_is_flagged_as_no_contrast(self):
        _, payload, _ = _finalize("e2", ("H2g",), [])
        synth = payload["hypothesis_synthesis"]["H2g"]
        self.assertEqual(synth["status"], "no_contrast")
        self.assertEqual(synth["n_rows"], 0)
        self.assertEqual(payload["n_hypotheses_no_contrast"], 1)

    def test_print_equivalence_keeps_the_paired_sample(self):
        diffs = [0.01, -0.01, 0.02, -0.02, 0.0]
        log = ContrastLog("e2", hypotheses=("H2b",), min_seeds=3)
        with contextlib.redirect_stdout(io.StringIO()):
            log.print_equivalence("drift", "H2b", "entropy_norm_drift",
                                  diffs)
        row = log.rows[0]
        self.assertEqual(row["diffs"], diffs)
        self.assertTrue(row["replication"]["valid"])
        self.assertEqual(row["replication"]["n"], len(diffs))

    def test_replication_uncertainty_is_aggregated_per_hypothesis(self):
        _, payload, _ = _finalize(
            "e2", ("H2a",),
            [("longest", "H2a", "accuracy_at_longest",
              [0.02, 0.03, 0.04, 0.05, 0.16])])
        synth = payload["hypothesis_synthesis"]["H2a"]
        self.assertEqual(synth["status"], "supported")
        self.assertEqual(synth["n_replication_uncertain"], 1)
        self.assertEqual(payload["n_hypotheses_replication_uncertain"], 1)
        self.assertIn("H2a", payload[
            "replication_uncertain_hypotheses"])


if __name__ == "__main__":
    unittest.main()
