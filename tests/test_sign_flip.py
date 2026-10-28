"""Exact small-sample inference for paired seed contrasts.

The grids run with five training seeds.  At that size a percentile
bootstrap p-value can be far smaller than anything the paired design can
actually produce (all five differences in the predicted direction is an
exact one-sided p of 1/32).  These tests pin the exact sign-flip test, its
resolution floor, its Monte-Carlo fallback, and the contrast-level audit
that keeps the pre-registered bootstrap decision separate from the exact
companion.
"""

from __future__ import annotations

import contextlib
import io
import math
import pathlib
import tempfile
import unittest

from cage.experiments import ContrastLog
from cage.metrics import (
    equivalence_test,
    noninferiority_test,
    paired_bootstrap_ci,
    sign_flip_min_units,
    sign_flip_resolution,
    sign_flip_test,
)


class ExactSignFlipTest(unittest.TestCase):
    def test_five_positive_differences_hit_the_exact_floor(self):
        result = sign_flip_test([1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertTrue(result["valid"])
        self.assertEqual(result["method"], "exact")
        self.assertAlmostEqual(result["p_values"]["greater"], 1 / 32)
        self.assertAlmostEqual(result["p_values"]["two-sided"], 2 / 32)
        self.assertAlmostEqual(result["p_values"]["less"], 1.0)
        self.assertAlmostEqual(result["min_attainable_p_values"]["greater"],
                               1 / 32)
        self.assertAlmostEqual(result["min_attainable_p_values"]["two-sided"],
                               2 / 32)

    def test_tolerance_is_relative_to_the_data_scale(self):
        # Tiny but non-zero effects are still a perfectly consistent sign
        # pattern.  An absolute 1e-12 tie tolerance used to count every
        # assignment as a tie here, returning p=1 instead of the exact
        # floor.
        reference = sign_flip_test([1.0] * 5)["p_values"]
        for scale in (1e-10, 1e-12, 1e-13, 1e-15):
            with self.subTest(scale=scale):
                result = sign_flip_test([scale] * 5)
                self.assertTrue(result["valid"], result["reason"])
                self.assertAlmostEqual(result["p_values"]["greater"],
                                       reference["greater"])
                self.assertAlmostEqual(result["p_values"]["less"],
                                       reference["less"])
                self.assertAlmostEqual(result["p_values"]["two-sided"],
                                       reference["two-sided"])
                self.assertAlmostEqual(
                    result["min_attainable_p_values"]["greater"], 1 / 32)

    def test_partial_sign_consistency_is_not_significant(self):
        # Four positive and one negative: six of 32 sign assignments are at
        # least this extreme, so p = 6/32 = 0.1875.  A bootstrap on the same
        # sample can look much more significant than this.
        result = sign_flip_test([1.0, 1.0, 1.0, 1.0, -1.0])
        self.assertAlmostEqual(result["p_values"]["greater"], 6 / 32)
        self.assertAlmostEqual(result["p_values"]["two-sided"], 12 / 32)

    def test_two_sided_resolution_depends_on_n(self):
        self.assertEqual(sign_flip_test([1.0] * 4)["p_values"]["two-sided"],
                         2 / 16)
        self.assertEqual(sign_flip_test([1.0] * 5)["p_values"]["two-sided"],
                         2 / 32)
        self.assertEqual(sign_flip_test([1.0] * 6)["p_values"]["two-sided"],
                         2 / 64)

    def test_monte_carlo_fallback_is_reproducible_and_bounded(self):
        values = [float(i) for i in range(1, 21)]
        first = sign_flip_test(values, n_random=2000, max_exact=10,
                               seed=7)
        second = sign_flip_test(values, n_random=2000, max_exact=10,
                                seed=7)
        self.assertEqual(first["method"], "monte-carlo")
        self.assertEqual(first["p_value"], second["p_value"])
        self.assertTrue(0.0 < first["p_value"] <= 1.0)
        self.assertGreater(first["min_attainable_p"], 0.0)

    def test_integer_like_seed_names_the_same_stream(self):
        class IndexSeed:
            def __index__(self):
                return 7

        values = [float(i) for i in range(1, 21)]
        plain = sign_flip_test(values, n_random=500, max_exact=10, seed=7)
        wrapped = sign_flip_test(values, n_random=500, max_exact=10,
                                 seed=IndexSeed())
        self.assertTrue(wrapped["valid"], wrapped["reason"])
        self.assertEqual(wrapped["p_value"], plain["p_value"])

    def test_malformed_inputs_abstain(self):
        cases = [
            ([], {}),
            ([1.0, float("nan")], {}),
            ([1.0, float("inf")], {}),
            ([1.0, 2.0], {"alternative": "up"}),
            ([1.0, 2.0], {"n_random": 0}),
            ([1.0, 2.0], {"max_exact": 0}),
            ([1.0, 2.0], {"seed": True}),
            ([1.0, 2.0], {"seed": 1.5}),
            ([1.0, 2.0], {"seed": 1.0}),
            # The Monte-Carlo path feeds the seed to torch's generator;
            # out-of-range integers must abstain like other bad seeds
            # rather than leaking torch's 64-bit overflow ValueError.
            ([1.0] * 20, {"n_random": 100, "max_exact": 10,
                          "seed": 2 ** 64}),
            ([1.0] * 20, {"n_random": 100, "max_exact": 10,
                          "seed": -(2 ** 63) - 1}),
            ([1.0] * 20, {"n_random": 100, "max_exact": 10,
                          "seed": 10 ** 400}),
            (["not-a-number"], {}),
            ([object()], {}),
        ]
        for values, kwargs in cases:
            with self.subTest(values=values, kwargs=kwargs):
                result = sign_flip_test(values, **kwargs)
                self.assertFalse(result["valid"])
                self.assertTrue(math.isnan(result["p_value"]))


class ExactResolutionPlanningTest(unittest.TestCase):
    """The design plan needs the same floor the analysis will report."""

    def test_exact_and_monte_carlo_floors_match_the_test(self):
        one = sign_flip_resolution(5, "greater")
        two = sign_flip_resolution(5, "two-sided")
        self.assertTrue(one["valid"])
        self.assertEqual(one["method"], "exact")
        self.assertAlmostEqual(one["min_attainable_p"], 1 / 32)
        self.assertAlmostEqual(two["min_attainable_p"], 2 / 32)
        # Above max_exact the companion samples signs, so the floor is the
        # Monte-Carlo resolution rather than 2**-n.
        sampled = sign_flip_resolution(17, "two-sided", n_random=999)
        self.assertEqual(sampled["method"], "monte-carlo")
        self.assertAlmostEqual(sampled["min_attainable_p"], 1 / 1000)

    def test_minimum_units_depend_on_the_registered_alternative(self):
        self.assertEqual(sign_flip_min_units("greater", 0.05), 5)
        self.assertEqual(sign_flip_min_units("two-sided", 0.05), 6)
        self.assertEqual(sign_flip_min_units("greater", 0.01), 7)
        self.assertEqual(sign_flip_min_units("two-sided", 0.01), 8)
        # No seed count can clear a q below the Monte-Carlo fallback floor.
        self.assertIsNone(sign_flip_min_units("two-sided", 1e-6,
                                              n_random=1000))

    def test_malformed_resolution_arguments_abstain(self):
        for n in (0, -1, 1.5, True, None):
            with self.subTest(n=n):
                self.assertFalse(sign_flip_resolution(n)["valid"])
        for alternative in ("up", "", None):
            with self.subTest(alternative=alternative):
                self.assertFalse(
                    sign_flip_resolution(5, alternative)["valid"])
        for q in (0.0, 1.0, -0.1, float("nan"), float("inf"), True):
            with self.subTest(q=q):
                self.assertIsNone(sign_flip_min_units("two-sided", q))


class BootstrapCompanionTest(unittest.TestCase):
    def test_bootstrap_result_carries_exact_companions(self):
        ci = paired_bootstrap_ci([1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(ci["sign_flip_p_greater"], 1 / 32)
        self.assertAlmostEqual(ci["sign_flip_p_two_sided"], 2 / 32)
        self.assertEqual(ci["sign_flip_method"], "exact")
        # The bootstrap p is below the exact floor: this is precisely why
        # the exact companion must travel with the result.
        self.assertLess(ci["p_value"], ci["sign_flip_min_p_two_sided"])
        self.assertTrue(ci["bootstrap_below_sign_flip_floor"])
        self.assertFalse(paired_bootstrap_ci([])[
            "bootstrap_below_sign_flip_floor"])

    def test_null_prediction_tests_carry_an_exact_tost(self):
        eq = equivalence_test([0.1, -0.05, 0.02, 0.0, 0.08], 0.5)
        self.assertTrue(eq["sign_flip_valid"])
        self.assertTrue(0.0 <= eq["sign_flip_p"] <= 1.0)
        ni = noninferiority_test([0.1, -0.05, 0.02, 0.0, 0.08], 0.5)
        self.assertTrue(ni["sign_flip_valid"])
        self.assertTrue(0.0 <= ni["sign_flip_p"] <= 1.0)


class ContrastSeedFloorValidationTest(unittest.TestCase):
    def test_invalid_min_seeds_is_rejected(self):
        for value in (0, -1, True, 2.5, None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ContrastLog("e1", hypotheses=("H1a",),
                                min_seeds=value)

    def test_integral_float_seed_floor_is_canonicalised(self):
        log = ContrastLog("e1", hypotheses=("H1a",), min_seeds=3.0)
        self.assertEqual(log.min_seeds, 3)
        self.assertIsInstance(log.min_seeds, int)

    def test_malformed_family_containers_are_rejected(self):
        for kwargs in ({"hypotheses": 5}, {"hypotheses": b"\x01"},
                       {"hypotheses": {"H1a": 1}},
                       {"hypotheses": ("H1a",), "competence": []},
                       {"hypotheses": ("H1a",), "coverage": "x"},
                       {"hypotheses": ("H1a",), "coverage": [1]}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ContrastLog("e1", **kwargs)


class ContrastExactAuditTest(unittest.TestCase):
    @staticmethod
    def _ci(diffs):
        ci = paired_bootstrap_ci(diffs)
        ci["controlled"] = {
            "controlled": True,
            "mismatches": [],
            "varying_factors": {"gate": {"a": ["calib"], "b": ["query"]}},
            "capacity": {"matched": True, "a": 10, "b": 10,
                         "delta_b_minus_a": 0},
        }
        return ci

    def test_exact_verdict_is_reported_next_to_the_bootstrap_verdict(self):
        # 0.1875 exact one-sided p: the bootstrap can be significant, the
        # exact design cannot corroborate it.  The row must say so instead
        # of silently presenting the bootstrap p as the last word.
        diffs = [0.20, 0.18, 0.19, 0.17, -0.01]
        log = ContrastLog("e1", hypotheses=("H1a",), min_seeds=3)
        with contextlib.redirect_stdout(io.StringIO()):
            log.print("calib - query", "H1a", "abs_gate_vs_entropy",
                      self._ci(diffs))
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = log.finalize(pathlib.Path(tmp), q=0.05)
        row = payload["rows"][0]
        self.assertTrue(row["supports_hypothesis"])
        self.assertFalse(row["sign_flip_supports"])
        self.assertTrue(row["sign_flip_disagrees"])
        self.assertEqual(payload["n_sign_flip_disagree"], 1)
        self.assertIn("H1a", payload["sign_flip_disagree_contrasts"][0])
        self.assertIn("bootstrap_below_exact_floor_contrasts", payload)

    def test_exact_floor_is_not_inflated_by_the_family_size(self):
        # With two exact p-values at the same attainable floor, BH compares
        # the larger one at rank 2 where its adjusted p equals p itself
        # (<= q), so it is not unfalsifiable.  Scaling the floor by the
        # family size (q/2 here) wrongly flagged both rows and contradicted
        # their own ``sign_flip_supports`` verdicts.
        log = ContrastLog("e3", hypotheses=("H3b",), min_seeds=3)
        ci_a = self._ci([1.0] * 6)
        ci_b = self._ci([1.0] * 6)
        for ci, p in ((ci_a, 0.001), (ci_b, 0.03125)):
            for key in ("sign_flip_p", "sign_flip_p_greater",
                        "sign_flip_p_less", "sign_flip_p_two_sided"):
                ci[key] = p
            for key in ("sign_flip_min_p", "sign_flip_min_p_greater",
                        "sign_flip_min_p_less",
                        "sign_flip_min_p_two_sided"):
                ci[key] = 0.03125
        ci_a["p_value"] = 0.001
        ci_b["p_value"] = 0.03
        with contextlib.redirect_stdout(io.StringIO()):
            log.print("row A", "H3b", "resid_norm_last_over_first", ci_a)
            log.print("row B", "H3b", "resid_norm_last_over_first", ci_b)
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = log.finalize(pathlib.Path(tmp), q=0.05)
        rows = payload["rows"]
        self.assertTrue(all(row["sign_flip_supports"] for row in rows))
        self.assertFalse(any(row["sign_flip_cannot_reject"] for row in rows))
        self.assertEqual(payload["n_sign_flip_unfalsifiable"], 0)

    def test_two_sided_claim_is_flagged_unfalsifiable_at_five_seeds(self):
        # H3b registers no direction, so the exact alternative is
        # two-sided; its floor at n=5 is 0.0625 > q=0.05.
        log = ContrastLog("e3", hypotheses=("H3b",), min_seeds=3)
        with contextlib.redirect_stdout(io.StringIO()):
            log.print("LayerScale effect", "H3b",
                      "resid_norm_last_over_first",
                      self._ci([0.2, 0.21, 0.19, 0.22, 0.18]))
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = log.finalize(pathlib.Path(tmp), q=0.05)
        row = payload["rows"][0]
        self.assertAlmostEqual(row["sign_flip_min_p"], 2 / 32)
        self.assertTrue(row["sign_flip_cannot_reject"])
        self.assertEqual(payload["n_sign_flip_unfalsifiable"], 1)

    def test_stream_limited_crossed_bootstrap_is_surfaced(self):
        # The seed-level CI excludes 0, but the crossed seed x item
        # bootstrap includes it: the row must keep its pre-registered
        # verdict, record the companion, and count the limitation.
        log = ContrastLog("e1", hypotheses=("H1a",), min_seeds=3)
        ci = self._ci([0.20, 0.18, 0.19, 0.17, 0.21])
        ci["stream_test"] = {
            "crossed_bootstrap": {
                "valid": True, "lo": -0.005, "hi": 0.25,
                "seed_ci_excludes_zero": True,
                "crossed_ci_excludes_zero": False,
                "decision_agrees": False, "stream_limited": True,
            },
        }
        with contextlib.redirect_stdout(io.StringIO()):
            log.print("calib - query", "H1a", "abs_gate_vs_entropy", ci)
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = log.finalize(pathlib.Path(tmp), q=0.05)
        row = payload["rows"][0]
        self.assertTrue(row["supports_hypothesis"])
        self.assertTrue(row["stream_crossed_stream_limited"])
        self.assertFalse(row["stream_crossed_decision_agrees"])
        self.assertEqual(payload["n_stream_crossed_stream_limited"], 1)
        self.assertIn("H1a", payload[
            "stream_crossed_stream_limited_contrasts"][0])


if __name__ == "__main__":
    unittest.main()
