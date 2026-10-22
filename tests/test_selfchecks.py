"""Run the package's deterministic scientific self-checks under unittest."""

from __future__ import annotations

import contextlib
import io
import math
import unittest

from cage import data, metrics


class MetricInputValidationTest(unittest.TestCase):
    def test_non_binary_correctness_abstains(self):
        self.assertTrue(math.isnan(
            metrics.expected_calibration_error([0.5], [0.5])))
        self.assertTrue(math.isnan(
            metrics.expected_calibration_error([0.5], [2.0])))
        self.assertTrue(math.isnan(metrics.brier_score([0.5], [0.5])))
        self.assertTrue(math.isnan(metrics.brier_score([0.5], [-1.0])))

    def test_fractional_multiclass_targets_abstain(self):
        import torch

        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        for targets in ([0.5, 1.0], [1.9, 0.0], [-0.5, 0.0],
                        [float("nan"), 0.0]):
            with self.subTest(targets=targets):
                out = metrics.multiclass_metrics(logits, targets)
                self.assertEqual(out["n"], 0)
                self.assertTrue(math.isnan(out["accuracy"]))
        # Integral floating-point labels are a valid JSON round-trip of the
        # token ids and must still be scored.
        out = metrics.multiclass_metrics(logits, [1.0, 0.0])
        self.assertEqual(out["n"], 2)
        self.assertTrue(math.isfinite(out["accuracy"]))

    def test_complex_metric_samples_are_rejected_not_truncated(self):
        import torch

        # ``Tensor.to(float64)`` silently discards the imaginary part with
        # only a warning; a complex paired difference must not be read as
        # its real projection.
        with self.assertRaises(ValueError):
            metrics.average_ranks(torch.tensor([1.0 + 2.0j, 3.0 + 4.0j]))
        with self.assertRaises(ValueError):
            metrics.paired_bootstrap_ci(torch.tensor([0.1 + 0.1j, 0.2 + 0j]))
        # Companions that document an abstention instead of raising still
        # abstain rather than silently truncating.
        self.assertFalse(metrics.sign_flip_test(
            torch.tensor([0.1 + 0.1j, 0.2 + 0j]))["valid"])
        self.assertFalse(metrics.replication_interval(
            torch.tensor([0.1 + 0.1j, 0.2 + 0j, 0.3 + 0j]))["valid"])

    def test_malformed_cluster_inputs_abstain(self):
        for successes, totals in ((["not-a-count"], [1]),
                                  ([1], ["not-a-count"]),
                                  ({"not": "a-count"}, [1])):
            with self.subTest(successes=successes, totals=totals):
                out = metrics.clustered_proportion_ci(successes, totals)
                self.assertFalse(out["valid"])
                self.assertTrue(out["reason"])


class ProjectSelfCheckTest(unittest.TestCase):
    def test_metrics_self_check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            failures = metrics._self_test()
        self.assertEqual(failures, 0, "cage.metrics self-test failed")

    def test_data_self_check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            failures = data._self_test()
        self.assertEqual(failures, 0, "cage.data self-test failed")


if __name__ == "__main__":
    unittest.main()
