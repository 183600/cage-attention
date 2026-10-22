"""Validation-check scheduling: ``curve_points`` must be honoured exactly."""

from __future__ import annotations

import contextlib
import io
import unittest

from cage.experiments import Settings, _train_and_eval


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
