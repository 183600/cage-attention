"""Validation-check scheduling: ``curve_points`` must be honoured exactly."""

from __future__ import annotations

import contextlib
import io
import unittest

from cage.experiments import Settings, _train_and_eval
from cage.train import training_curve_verdict


class CurveVerdictSingleCheckTest(unittest.TestCase):
    def test_one_validation_check_still_reports_the_best_point(self):
        hist = [
            {"step": 0, "train_loss": 1.0, "lr": 1e-3,
             "val_loss": 2.0, "val_loss_se": 0.1},
            {"step": 1, "train_loss": 0.8, "lr": 1e-4},
        ]
        out = training_curve_verdict(hist)
        self.assertEqual(out["best_val_loss"], 2.0)
        self.assertEqual(out["best_val_step"], 0)
        self.assertEqual(out["val_regression_after_best"], 0.0)
        self.assertAlmostEqual(out["generalization_gap"], 1.0)

    def test_single_point_history_reports_best_before_abstaining(self):
        out = training_curve_verdict(
            [{"step": 0, "train_loss": 1.0, "val_loss": 2.0, "lr": 1e-3}])
        self.assertEqual(out["verdict"], "insufficient_points")
        self.assertEqual(out["best_val_loss"], 2.0)
        self.assertEqual(out["best_val_step"], 0)
        self.assertEqual(out["val_regression_after_best"], 0.0)


class CurvePointsTest(unittest.TestCase):
    def _curve(self, steps, points):
        settings = Settings.quick()
        settings.steps = steps
        settings.curve_points = points
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, _, curve, _ = _train_and_eval(
                settings, 0, 1e-3, eval_seed=123, gate="none")
        return curve["curve_steps"]

    def test_configured_count_is_exact_at_small_step_budgets(self):
        self.assertEqual(len(self._curve(12, 10)), 10)
        self.assertEqual(len(self._curve(5, 3)), 3)

    def test_single_point_is_the_final_check(self):
        self.assertEqual(self._curve(5, 1), [4])

    def test_more_points_than_steps_is_capped_at_one_per_step(self):
        self.assertEqual(self._curve(4, 10), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
