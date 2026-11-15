"""E2 length-sweep geometry contract.

The E2 grid's *length axis* is a design parameter, not an implementation
detail: the pre-registered hypotheses read an accuracy drop and a slope
across the sweep, and both are only meaningful over the range that was
actually measured.  These tests pin

* what ``cage.data.length_sweep_plan`` reports (requested grid, realized
  grid, dropped lengths and why),
* that the realized grid is never silently shorter than the registered
  minimum (three lengths: without an interior point the degradation slope
  is the endpoint difference rescaled and cannot detect a mid-sweep
  reversal),
* that the E2 runner and the evaluation-balance audit resolve the *same*
  sweep, and
* that the machine-readable pre-registration block in
  ``EXPERIMENT_PROTOCOL.md`` still matches what the code executes.
"""

from __future__ import annotations

import pathlib
import re
import unittest

from cage.data import length_sweep_lengths, length_sweep_plan
# Reproducibility: Set random seeds
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass

PROTOCOL = (pathlib.Path(__file__).resolve().parents[1]
            / "EXPERIMENT_PROTOCOL.md")


class LengthSweepPlanTest(unittest.TestCase):
    def test_plan_reports_requested_and_realized_grids(self):
        plan = length_sweep_plan(base=16, factor=2, n=4, max_len=256)
        self.assertEqual(plan["requested_lengths"], [16, 32, 64, 128])
        self.assertEqual(plan["realized_lengths"], [16, 32, 64])
        self.assertEqual(plan["requested_count"], 4)
        self.assertFalse(plan["requested_grid_complete"])
        self.assertTrue(plan["truncated"])
        self.assertEqual(plan["n_dropped"], 1)

    def test_complete_requested_grid_is_flagged_complete(self):
        plan = length_sweep_plan(base=16, factor=2, n=4, max_len=512)
        self.assertTrue(plan["requested_grid_complete"])
        self.assertEqual(plan["requested_count"], 4)
        self.assertEqual(plan["n_dropped"], 0)

    def test_dropped_length_states_the_geometry_that_excluded_it(self):
        plan = length_sweep_plan(base=16, factor=2, n=4, max_len=256)
        self.assertEqual(len(plan["dropped"]), 1)
        row = plan["dropped"][0]
        self.assertEqual(row["length"], 128)
        self.assertEqual(row["required_positions"], 258)
        self.assertEqual(row["max_len"], 256)
        self.assertIn("258", row["reason"])

    def test_grid_that_fits_is_not_truncated(self):
        plan = length_sweep_plan(base=16, factor=2, n=4, max_len=512)
        self.assertEqual(plan["realized_lengths"], [16, 32, 64, 128])
        self.assertEqual(plan["dropped"], [])
        self.assertFalse(plan["truncated"])
        self.assertEqual(plan["n_dropped"], 0)
        self.assertIsNone(plan["min_max_len_for_next_length"])

    def test_coverage_reports_the_measured_extrapolation_range(self):
        plan = length_sweep_plan(base=16, factor=2, n=4, max_len=256)
        coverage = plan["coverage"]
        self.assertEqual(coverage["base_pairs"], 16)
        self.assertEqual(coverage["longest_pairs"], 64)
        self.assertEqual(coverage["base_positions"], 34)
        self.assertEqual(coverage["longest_positions"], 130)
        self.assertAlmostEqual(coverage["pair_ratio"], 4.0)
        self.assertAlmostEqual(coverage["position_ratio"], 130 / 34)

    def test_two_point_grid_is_flagged_as_slope_degenerate(self):
        # 2 pairs -> 6 positions, 4 pairs -> 10, 8 pairs -> 18 > 16.
        plan = length_sweep_plan(base=2, factor=2, n=3, max_len=16)
        self.assertEqual(plan["realized_lengths"], [2, 4])
        self.assertTrue(plan["slope_degenerate"])

    def test_three_point_grid_is_not_slope_degenerate(self):
        # 2 pairs -> 6 positions, 4 pairs -> 10, 8 pairs -> 18 <= 20.
        plan = length_sweep_plan(base=2, factor=2, n=4, max_len=20)
        self.assertEqual(plan["realized_lengths"], [2, 4, 8])
        self.assertFalse(plan["slope_degenerate"])

    def test_min_lengths_refuses_a_grid_without_an_interior_point(self):
        with self.assertRaises(ValueError) as ctx:
            length_sweep_plan(base=2, factor=2, n=4, max_len=16,
                              min_lengths=3)
        self.assertIn("at least 3 fitted lengths", str(ctx.exception))

    def test_min_lengths_two_keeps_the_historical_contract(self):
        self.assertEqual(
            length_sweep_plan(base=2, factor=2, n=4, max_len=16,
                              min_lengths=2)["realized_lengths"], [2, 4])

    def test_lengths_helper_is_the_realized_view_of_the_plan(self):
        self.assertEqual(length_sweep_lengths(16, 2, 4, 256), [16, 32, 64])
        self.assertEqual(
            length_sweep_lengths(base=16, factor=2, n=4, max_len=256),
            length_sweep_plan(base=16, factor=2, n=4,
                              max_len=256)["realized_lengths"])

    def test_historical_error_message_is_preserved(self):
        with self.assertRaises(ValueError) as ctx:
            length_sweep_lengths(16, 2, 4, 34)
        self.assertIn("at least two fitted lengths", str(ctx.exception))

    def test_invalid_geometry_is_rejected(self):
        for args in ((0, 2, 4, 256), (16, 1, 4, 256), (16, 0, 4, 256),
                     (16, 2, 0, 256)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    length_sweep_plan(*args)

    def test_bool_geometry_is_rejected(self):
        for kwargs in ({"base": True}, {"factor": True}, {"n": True},
                       {"max_len": True}, {"min_lengths": True}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    length_sweep_plan(**kwargs)

    def test_huge_request_does_not_build_an_unbounded_grid(self):
        plan = length_sweep_plan(base=1, factor=2, n=10 ** 9, max_len=16)
        self.assertEqual(plan["realized_lengths"], [1, 2, 4])
        self.assertEqual(plan["n_dropped"], 10 ** 9 - 3)
        # The full requested grid is not representable, so the field is
        # None instead of a fabricated huge number.
        self.assertIsNone(plan["min_max_len_for_requested"])
        self.assertEqual(plan["min_max_len_for_next_length"], 18)

    def test_plan_is_json_safe(self):
        import json

        plan = length_sweep_plan(base=16, factor=2, n=4, max_len=256)
        text = json.dumps(plan, allow_nan=False)
        self.assertIn("realized_lengths", text)


class SettingsSweepContractTest(unittest.TestCase):
    def test_defaults_match_the_registered_sweep(self):
        from cage.experiments import Settings

        settings = Settings()
        self.assertEqual(settings.e2_sweep_lengths, 4)
        self.assertEqual(settings.e2_sweep_factor, 2)
        self.assertEqual(settings.e2_sweep_min_lengths, 3)

    def test_factor_must_exceed_one(self):
        from cage.experiments import Settings

        with self.assertRaises(ValueError):
            Settings(e2_sweep_factor=1)

    def test_min_lengths_cannot_exceed_requested_lengths(self):
        from cage.experiments import Settings

        with self.assertRaises(ValueError):
            Settings(e2_sweep_lengths=2, e2_sweep_min_lengths=3)

    def test_shared_helper_matches_the_data_level_plan(self):
        from cage.experiments import Settings, _e2_length_sweep_plan

        settings = Settings()
        plan = _e2_length_sweep_plan(settings)
        self.assertEqual(
            plan["realized_lengths"],
            length_sweep_plan(base=settings.n_pairs,
                              factor=settings.e2_sweep_factor,
                              n=settings.e2_sweep_lengths,
                              max_len=settings.max_len,
                              min_lengths=settings.e2_sweep_min_lengths
                              )["realized_lengths"])

    def test_shared_helper_refuses_a_degenerate_geometry(self):
        from cage.experiments import Settings, _e2_length_sweep_plan

        # 2 pairs -> 6 positions, 4 pairs -> 10, 8 pairs -> 18; a
        # max_len of 12 therefore realizes only two lengths.
        settings = Settings(n_pairs=2, vocab=64, max_len=12)
        with self.assertRaises(ValueError):
            _e2_length_sweep_plan(settings)

    def test_quick_configuration_realizes_three_lengths(self):
        from cage.experiments import Settings, _e2_length_sweep_plan

        plan = _e2_length_sweep_plan(Settings.quick())
        self.assertEqual(len(plan["realized_lengths"]), 3)
        self.assertFalse(plan["slope_degenerate"])


class ProtocolConsistencyTest(unittest.TestCase):
    """The pre-registered sweep must match what the code executes."""

    def _contract(self):
        text = PROTOCOL.read_text(encoding="utf-8")
        match = re.search(
            r"<!-- e2-sweep-contract\s*(?P<body>.*?)-->", text, re.DOTALL)
        self.assertIsNotNone(
            match, "EXPERIMENT_PROTOCOL.md lost its e2-sweep-contract block")
        values = {}
        for line in match.group("body").splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, _, raw = line.partition("=")
            values[key.strip()] = raw.strip()
        return values

    def test_registered_block_matches_the_executed_sweep(self):
        from cage.experiments import Settings, _e2_length_sweep_plan

        values = self._contract()
        settings = Settings()
        self.assertEqual(int(values["base"]), settings.n_pairs)
        self.assertEqual(int(values["factor"]), settings.e2_sweep_factor)
        self.assertEqual(int(values["n"]), settings.e2_sweep_lengths)
        self.assertEqual(int(values["max_len"]), settings.max_len)
        self.assertEqual(int(values["min_lengths"]),
                         settings.e2_sweep_min_lengths)
        plan = _e2_length_sweep_plan(settings)
        registered = [int(x) for x in values["realized"].split(",")]
        self.assertEqual(registered, plan["realized_lengths"])

    def test_registered_block_never_names_a_length_that_cannot_run(self):
        values = self._contract()
        max_len = int(values["max_len"])
        for raw in values["realized"].split(","):
            length = int(raw)
            self.assertLessEqual(2 * length + 2, max_len)

    def test_protocol_states_the_realized_grid_not_the_requested_one(self):
        text = PROTOCOL.read_text(encoding="utf-8")
        self.assertIn("**实际执行网格**", text)
        # The stale claim that the default geometry runs four lengths must
        # not come back.
        self.assertNotIn("→ `[16, 32, 64, 128]`", text)


if __name__ == "__main__":
    unittest.main()
