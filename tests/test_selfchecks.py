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
