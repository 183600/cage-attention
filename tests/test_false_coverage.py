"""Selection-adjusted (false-coverage-rate) intervals for BH discoveries."""

from __future__ import annotations

import contextlib
import io
import pathlib
import tempfile
import unittest

from cage import experiments
from cage.metrics import false_coverage_ci, paired_bootstrap_ci


class FalseCoverageIntervalTest(unittest.TestCase):
    def test_selection_adjustment_widens_the_interval(self):
        diffs = [0.10, 0.12, 0.11, 0.09, 0.13]
        raw = paired_bootstrap_ci(diffs, n_boot=4000)
        adjusted = false_coverage_ci(
            diffs, n_selected=1, n_tests=10, q=0.05, n_boot=4000)
        self.assertTrue(adjusted["valid"])
        self.assertAlmostEqual(adjusted["adjusted_alpha"], 0.005)
        self.assertAlmostEqual(adjusted["confidence_level"], 0.995)
        self.assertLess(adjusted["lo"], raw["lo"])
        self.assertGreater(adjusted["hi"], raw["hi"])
        self.assertTrue(adjusted["excludes_zero"])

    def test_all_selected_matches_the_ordinary_interval(self):
        diffs = [0.10, 0.12, 0.11, 0.09, 0.13]
        raw = paired_bootstrap_ci(diffs, n_boot=4000)
        adjusted = false_coverage_ci(
            diffs, n_selected=5, n_tests=5, q=0.05, n_boot=4000)
        self.assertTrue(adjusted["valid"])
        self.assertAlmostEqual(adjusted["adjusted_alpha"], 0.05)
        self.assertAlmostEqual(adjusted["lo"], raw["lo"])
        self.assertAlmostEqual(adjusted["hi"], raw["hi"])

    def test_arbitrary_dependence_applies_the_harmonic_factor(self):
        diffs = [0.10, 0.12, 0.11, 0.09, 0.13]
        prds = false_coverage_ci(
            diffs, n_selected=2, n_tests=10, q=0.05, n_boot=20000)
        by = false_coverage_ci(
            diffs, n_selected=2, n_tests=10, q=0.05, n_boot=20000,
            dependence="arbitrary")
        harmonic = sum(1.0 / i for i in range(1, 11))
        self.assertTrue(by["valid"])
        self.assertAlmostEqual(by["harmonic_factor"], harmonic)
        self.assertAlmostEqual(
            by["adjusted_alpha"],
            0.05 * 2 / (10 * harmonic))
        self.assertLess(by["adjusted_alpha"], prds["adjusted_alpha"])
        self.assertLess(by["lo"], prds["lo"])
        self.assertGreater(by["hi"], prds["hi"])

    def test_invalid_inputs_abstain_instead_of_claiming_coverage(self):
        diffs = [0.10, 0.12, 0.11, 0.09, 0.13]
        cases = (
            (([], 1, 10), {}),
            (([0.1], 1, 10), {}),
            ((diffs, 0, 10), {}),
            ((diffs, 11, 10), {}),
            ((diffs, 1, 10), {"q": 0.0}),
            ((diffs, 1, 10), {"n_boot": 0}),
            ((diffs, 1, 10), {"seed": True}),
            ((diffs, 1, 10), {"dependence": "unknown"}),
            (([float("nan"), 0.1], 1, 10), {}),
            ((diffs, 1, 10), {"q": 1.0}),
        )
        for args, kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs):
                out = false_coverage_ci(*args, **kwargs)
                self.assertFalse(out["valid"])
                self.assertTrue(out["reason"])


class FcrDecisionTest(unittest.TestCase):
    def test_directional_bounds_use_the_registered_sign(self):
        positive = {"test": "difference", "direction": "+"}
        negative = {"test": "difference", "direction": "-"}
        self.assertTrue(experiments._fcr_decision_holds(positive, 0.01, 0.2))
        self.assertFalse(experiments._fcr_decision_holds(positive, -0.1, 0.2))
        self.assertTrue(experiments._fcr_decision_holds(negative, -0.2, -0.01))
        self.assertFalse(experiments._fcr_decision_holds(negative, -0.2, 0.1))

    def test_null_predictions_use_their_bound(self):
        equivalence = {"test": "equivalence", "equivalence_bound": 0.05}
        noninferior = {
            "test": "noninferiority", "noninferiority_bound": 0.05}
        self.assertTrue(
            experiments._fcr_decision_holds(equivalence, -0.04, 0.04))
        self.assertFalse(
            experiments._fcr_decision_holds(equivalence, -0.06, 0.04))
        self.assertTrue(
            experiments._fcr_decision_holds(noninferior, -0.04, 0.20))
        self.assertFalse(
            experiments._fcr_decision_holds(noninferior, -0.06, 0.20))

    def test_unavailable_bounds_never_support(self):
        row = {"test": "difference", "direction": "+"}
        self.assertFalse(experiments._fcr_decision_holds(
            row, float("nan"), float("nan")))


class ContrastLogFcrTest(unittest.TestCase):
    @staticmethod
    def _ci(mean, diffs, p_value):
        n = len(diffs)
        return {
            "mean": mean,
            "lo": mean - 0.01,
            "hi": mean + 0.01,
            "n": n,
            "p_value": p_value,
            "sd": 0.02,
            "mde80": 0.03,
            "diffs": diffs,
            "sign_consistency": 1.0,
            "controlled": None,
            "arms": {
                "a": {"mean": 0.50, "std": 0.10, "n": n},
                "b": {"mean": 0.45, "std": 0.10, "n": n},
            },
            "arms_all": {
                "a": {"mean": 0.50, "n": n},
                "b": {"mean": 0.45, "n": n},
            },
            "seeds_used": list(range(n)),
            "seeds_dropped": [],
            "seeds_unpaired": [],
            "seeds_undefined": [],
        }

    def test_finalize_flags_a_selection_sensitive_claim(self):
        log = experiments.ContrastLog("e1", hypotheses=("H1b",))
        # The mean clears the registered SESOI and the p-value survives BH,
        # but the paired differences are noisy enough that the 99.5% FCR
        # interval still includes zero.  That is exactly the row the
        # selection-adjusted companion must expose.
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                log.print("calib - query", "H1b", "gate_margin",
                          self._ci(0.025,
                                   [-0.01, 0.08, 0.0, 0.03, 0.025], 0.001))
                for index in range(9):
                    log.print(f"null-{index}", "H1b", "gate_margin",
                              self._ci(0.0, [0.0] * 5, 0.9))
                payload = log.finalize(pathlib.Path(tmp), q=0.05)
        row = payload["rows"][0]
        self.assertTrue(row["supports_hypothesis"])
        self.assertTrue(row["fcr_selected"])
        self.assertAlmostEqual(row["fcr_adjusted_alpha"], 0.005)
        self.assertFalse(row["fcr_decision_holds"])
        self.assertFalse(row["fcr_supports"])
        self.assertEqual(payload["n_fcr_selected"], 1)
        self.assertEqual(payload["n_fcr_tests"], 10)
        self.assertEqual(payload["n_fcr_decision_sensitive"], 1)
        self.assertEqual(
            payload["fcr_decision_sensitive_contrasts"],
            ["H1b/calib - query (gate_margin)"])

    def test_fcr_selection_uses_the_raw_bh_set(self):
        log = experiments.ContrastLog("e1", hypotheses=("H1b",))
        confounded = self._ci(
            0.025, [-0.01, 0.08, 0.0, 0.03, 0.025], 0.0005)
        confounded["controlled"] = {
            "controlled": False,
            "mismatches": ["test confound"],
            "single_factor": False,
            "n_varying_factors": 2,
            "varying_factors": {},
            "capacity": None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                log.print("clean", "H1b", "gate_margin",
                          self._ci(0.025,
                                   [-0.01, 0.08, 0.0, 0.03, 0.025], 0.001))
                log.print("confounded", "H1b", "gate_margin", confounded)
                for index in range(8):
                    log.print(f"null-{index}", "H1b", "gate_margin",
                              self._ci(0.0, [0.0] * 5, 0.9))
                payload = log.finalize(pathlib.Path(tmp), q=0.05)
        rows = {row["rule"]: row for row in payload["rows"]}
        self.assertTrue(rows["confounded"]["bh_selected"])
        self.assertFalse(rows["confounded"]["fdr_reject"])
        self.assertTrue(rows["confounded"]["fcr_selected"])
        self.assertFalse(rows["confounded"]["fcr_supports"])
        self.assertEqual(payload["n_fcr_selected"], 2)


if __name__ == "__main__":
    unittest.main()
