"""Minimum-effect (SESOI) testing for practically meaningful results.

A pre-registered SESOI is only useful if it is read at the interval level:
an effect whose point estimate clears the bound while its 95% CI still
spans a negligible effect is not established as practically relevant.
These tests pin the bootstrap minimum-effect test, its small-sample
companions, and the contrast-level audit that keeps the pre-registered
decision rule separate from the interval-level practical claim.
"""

from __future__ import annotations

import contextlib
import io
import math
import pathlib
import tempfile
import unittest

from cage.experiments import (
    ContrastLog,
    _fmt_ci,
    _row_is_refutation,
)
from cage.metrics import (
    equivalence_test,
    minimum_effect_test,
    noninferiority_test,
    paired_bootstrap_ci,
)


class MinimumEffectMetricTest(unittest.TestCase):
    def test_tight_effect_above_bound_is_established(self):
        values = [0.20, 0.19, 0.21, 0.18, 0.22]
        result = minimum_effect_test(values, 0.05, direction="+")
        self.assertTrue(result["beyond_sesoi"])
        self.assertGreater(result["lo"], result["bound"])
        self.assertTrue(result["loo_beyond_sesoi"])
        self.assertTrue(result["bca_agrees"])
        self.assertLess(result["p_value"], 0.05)

    def test_point_estimate_above_bound_is_not_established_when_ci_spans_it(self):
        # Mean 0.058 > SESOI 0.05, but the 95% CI reaches below the bound.
        # This is exactly the reading that a point-estimate SESOI gate
        # cannot distinguish from a practically established effect.
        values = [0.02, 0.03, 0.04, 0.05, 0.15]
        result = minimum_effect_test(values, 0.05, direction="+")
        self.assertGreater(result["mean"], result["bound"])
        self.assertLessEqual(result["lo"], result["bound"])
        self.assertFalse(result["beyond_sesoi"])
        self.assertGreater(result["p_value"], 0.05)

    def test_negative_direction_requires_the_upper_bound_to_clear(self):
        values = [-0.20, -0.19, -0.21, -0.18, -0.22]
        result = minimum_effect_test(values, 0.05, direction="-")
        self.assertTrue(result["beyond_sesoi"])
        self.assertLess(result["hi"], -result["bound"])
        # A positive sample cannot establish a negative minimum effect.
        wrong = minimum_effect_test([0.20, 0.19, 0.21, 0.18, 0.22], 0.05,
                                    direction="-")
        self.assertFalse(wrong["beyond_sesoi"])

    def test_result_is_deterministic(self):
        values = [0.02, 0.03, 0.04, 0.05, 0.15]
        first = minimum_effect_test(values, 0.05, direction="+")
        second = minimum_effect_test(values, 0.05, direction="+")
        self.assertEqual(first["p_value"], second["p_value"])
        self.assertEqual(first["lo"], second["lo"])

    def test_malformed_inputs_abstain(self):
        cases = [
            ([], 0.05, "+"),
            ([0.1], 0.05, "+"),
            ([0.1, float("nan")], 0.05, "+"),
            ([0.1, float("inf")], 0.05, "+"),
            ([0.1, 0.2], 0.0, "+"),
            ([0.1, 0.2], -0.1, "+"),
            ([0.1, 0.2], float("nan"), "+"),
            ([0.1, 0.2], 0.05, "up"),
            ([0.1, 0.2], 0.05, None),
        ]
        for values, bound, direction in cases:
            with self.subTest(values=values, bound=bound,
                              direction=direction):
                result = minimum_effect_test(values, bound,
                                             direction=direction)
                self.assertFalse(result["beyond_sesoi"])
                self.assertTrue(math.isnan(result["p_value"]))
        self.assertFalse(
            minimum_effect_test([0.1, 0.2], 0.05, n_boot=0)["beyond_sesoi"])
        self.assertFalse(
            minimum_effect_test([0.1, 0.2], 0.05,
                                alpha=2.0)["beyond_sesoi"])
        for test in (equivalence_test, noninferiority_test,
                     minimum_effect_test):
            with self.subTest(test=test.__name__):
                out = test([0.1, float("nan")], 0.5)
                self.assertEqual(out["n"], 2)
                self.assertFalse(out.get("equivalent", False))

    def test_single_unit_reports_point_without_a_decision(self):
        result = minimum_effect_test([0.5], 0.05)
        self.assertEqual(result["n"], 1)
        self.assertFalse(result["beyond_sesoi"])
        self.assertIsNone(result["bca_agrees"])

    def test_single_unit_with_invalid_bound_abstains(self):
        # A one-unit point may be reported only for a usable positive
        # bound; a malformed bound must not leak the point through the
        # early-return path that normally abstains with NaN fields.
        tests = (
            ("equivalence", equivalence_test),
            ("noninferiority", noninferiority_test),
            ("minimum_effect", minimum_effect_test),
        )
        for name, test in tests:
            for bound in (0.0, -0.1, float("nan"), float("inf")):
                with self.subTest(test=name, bound=bound):
                    result = test([0.5], bound)
                    self.assertEqual(result["n"], 0)
                    self.assertTrue(math.isnan(result["mean"]))


class ContrastMinimumEffectAuditTest(unittest.TestCase):
    @staticmethod
    def _ci(diffs):
        ci = paired_bootstrap_ci(diffs)
        # Real contrasts carry their paired differences from _paired_diff /
        # _one_sample_contrast; finalize needs them for the companion.
        ci["diffs"] = list(diffs)
        ci["controlled"] = {
            "controlled": True,
            "mismatches": [],
            "varying_factors": {"gate": {"a": ["calib"], "b": ["query"]}},
            "capacity": {"matched": True, "a": 10, "b": 10,
                         "delta_b_minus_a": 0},
        }
        return ci

    def _finalize(self, tag, hypothesis, metric, diffs):
        log = ContrastLog(tag, hypotheses=(hypothesis,), min_seeds=3)
        with contextlib.redirect_stdout(io.StringIO()):
            log.print("step", hypothesis, metric, self._ci(diffs))
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                return log.finalize(pathlib.Path(tmp), q=0.05)

    def test_noninferiority_row_gets_a_real_power_plan(self):
        # The row stores the test name "noninferiority", while the power
        # helper names the one-sided mode "noninferior".  Passing the test
        # name through used to make every non-inferiority row report
        # n_seeds_for_sesoi=inf regardless of its seed-level SD.
        diffs = [-0.005, -0.002, 0.001, -0.003, 0.0]
        ci = noninferiority_test(diffs, 0.05)
        ci["diffs"] = list(diffs)
        ci["controlled"] = {
            "controlled": True,
            "mismatches": [],
            "varying_factors": {"norm": {"a": ["seednorm"], "b": ["pre"]}},
            "capacity": {"matched": False, "a": 10, "b": 8,
                         "delta_b_minus_a": -2},
        }
        log = ContrastLog("e3", hypotheses=("H3a",), min_seeds=3)
        with contextlib.redirect_stdout(io.StringIO()):
            log.print("step", "H3a", "accuracy", ci)
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = log.finalize(pathlib.Path(tmp), q=0.05)
        row = payload["rows"][0]
        self.assertEqual(row["test"], "noninferiority")
        self.assertTrue(math.isfinite(row["n_seeds_for_sesoi"]),
                        row["n_seeds_for_sesoi"])
        self.assertGreater(row["n_seeds_for_sesoi"], 0.0)

    def test_point_estimate_only_row_is_flagged_without_changing_verdict(self):
        payload = self._finalize(
            "e1", "H1a", "abs_gate_vs_entropy",
            [0.02, 0.03, 0.04, 0.05, 0.15])
        row = payload["rows"][0]
        # The pre-registered decision (point estimate >= SESOI plus FDR)
        # still supports the contrast ...
        self.assertTrue(row["supports_hypothesis"])
        # ... but its practical size is not established at the interval
        # level, and the report says so.
        self.assertFalse(row["minimum_effect_established"])
        self.assertTrue(row["sesoi_point_estimate_only"])
        self.assertFalse(row["minimum_effect_fdr_reject"])
        self.assertEqual(payload["n_minimum_effect_established"], 0)
        self.assertEqual(payload["n_sesoi_point_estimate_only"], 1)
        self.assertIn("H1a", payload[
            "sesoi_point_estimate_only_contrasts"][0])

    def test_interval_level_practical_effect_is_established(self):
        payload = self._finalize(
            "e1", "H1a", "abs_gate_vs_entropy",
            [0.20, 0.19, 0.21, 0.18, 0.22])
        row = payload["rows"][0]
        self.assertTrue(row["supports_hypothesis"])
        self.assertTrue(row["minimum_effect_established"])
        self.assertTrue(row["minimum_effect_fdr_reject"])
        self.assertFalse(row["sesoi_point_estimate_only"])
        self.assertEqual(payload["n_minimum_effect_established"], 1)
        self.assertEqual(payload["n_sesoi_point_estimate_only"], 0)

    def test_negative_direction_row_is_audited_with_its_own_sign(self):
        payload = self._finalize(
            "e2", "H2a", "accuracy_drop",
            [-0.005, -0.02, -0.03, -0.003, -0.01])
        row = payload["rows"][0]
        self.assertEqual(row["direction"], "-")
        self.assertTrue(row["supports_hypothesis"])
        self.assertTrue(row["sesoi_point_estimate_only"])
        self.assertFalse(row["minimum_effect_established"])

    def test_undirected_prediction_has_no_minimum_effect(self):
        # H3b registers no sign for this metric; inferring the direction
        # from the observed effect would turn a two-sided readout into a
        # one-sided claim, so the companion must abstain.
        payload = self._finalize(
            "e3", "H3b", "resid_norm_last_over_first",
            [0.20, 0.19, 0.21, 0.18, 0.22])
        row = payload["rows"][0]
        self.assertIsNone(row["direction"])
        self.assertTrue(math.isnan(row["minimum_effect_p"]))
        self.assertFalse(row["sesoi_point_estimate_only"])
        self.assertFalse(row["minimum_effect_established"])

    def test_noninferiority_straddling_bound_is_inconclusive_not_refuted(self):
        # lo below the negative bound alone is the non-inferiority test's
        # non-rejection, not evidence that the whole interval is materially
        # worse.  Only hi < -bound is a refutation.
        diffs = [-0.05, 0.0, 0.02, -0.01, 0.01]
        ci = noninferiority_test(diffs, 0.01)
        self.assertLess(ci["lo"], -ci["bound"])
        self.assertGreater(ci["hi"], -ci["bound"])
        self.assertFalse(ci["noninferior"])
        self.assertIn("inconclusive", _fmt_ci(ci))
        self.assertNotIn("refuted", _fmt_ci(ci))
        row = {
            "test": "noninferiority", "noninferiority_bound": 0.01,
            "lo": ci["lo"], "hi": ci["hi"], "controlled": True,
            "claim_eligible": True,
        }
        self.assertFalse(_row_is_refutation(row))
        ci["diffs"] = list(diffs)
        ci["controlled"] = {
            "controlled": True, "mismatches": [],
            "varying_factors": {"gate": {"a": ["calib"], "b": ["none"]}},
            "capacity": {"matched": True, "a": 10, "b": 10,
                         "delta_b_minus_a": 0},
        }
        log = ContrastLog("e1", hypotheses=("H1c",), min_seeds=3)
        with contextlib.redirect_stdout(io.StringIO()):
            log.print("calib-none", "H1c", "accuracy", ci)
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = log.finalize(pathlib.Path(tmp), q=0.05)
        self.assertEqual(
            payload["hypothesis_synthesis"]["H1c"]["status"],
            "inconclusive")
        # A genuinely material loss (the whole CI below -bound) remains a
        # refutation.
        worse = noninferiority_test([-0.20, -0.19, -0.21, -0.18, -0.22],
                                    0.01)
        self.assertTrue(_row_is_refutation({
            "test": "noninferiority", "noninferiority_bound": 0.01,
            "lo": worse["lo"], "hi": worse["hi"], "controlled": True,
            "claim_eligible": True,
        }))


if __name__ == "__main__":
    unittest.main()
