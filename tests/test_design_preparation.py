"""Design-preparation tests: blocked run order and the prospective power plan.

Both helpers exist to catch a design problem *before* the grid spends
compute: an arm-major schedule aliases treatment with machine drift, and a
seed count chosen without a pilot variance can make every later null read
uninformative.  The tests pin the schedule's complete-block property and the
power plan's arithmetic/status handling.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import pathlib
import tempfile
import unittest

from cage.experiments import (
    Settings,
    _calib_cell,
    _controlled_audit,
    _blocked_run_order,
    _design_sensitivity_plan,
    _pilot_accuracy_level,
    _pilot_accuracy_sd,
    _pilot_paired_accuracy_sd,
    _power_at_effect,
    _run_order_audit,
    _verdict_seed_floor,
)


class BlockedRunOrderTest(unittest.TestCase):
    CELLS = ("none", "query", "calib", "null")
    SEEDS = (0, 1, 2)

    def test_every_block_contains_every_cell_once(self):
        order = _blocked_run_order(self.CELLS, self.SEEDS, 11)
        self.assertEqual(len(order), len(self.CELLS) * len(self.SEEDS))
        for block in range(len(self.SEEDS)):
            rows = [row for row in order if row[2] == block]
            self.assertEqual({row[0] for row in rows}, set(self.CELLS))
            self.assertEqual(len(rows), len(self.CELLS))
            self.assertEqual(sorted(row[3] for row in rows),
                             list(range(len(self.CELLS))))

    def test_schedule_is_deterministic_and_seed_dependent(self):
        first = _blocked_run_order(self.CELLS, self.SEEDS, 11)
        second = _blocked_run_order(self.CELLS, self.SEEDS, 11)
        self.assertEqual(first, second)
        third = _blocked_run_order(self.CELLS, self.SEEDS, 12)
        self.assertNotEqual(first, third)

    def test_shuffle_false_keeps_cell_major_order(self):
        order = _blocked_run_order(self.CELLS, (0, 1), 11, shuffle=False)
        self.assertEqual([(c, s) for c, s, _, _ in order],
                         [(c, s) for s in (0, 1) for c in self.CELLS])

    def test_audit_flags_incomplete_and_extreme_schedules(self):
        complete = _blocked_run_order(self.CELLS, self.SEEDS, 11)
        audit = _run_order_audit(complete, self.CELLS, self.SEEDS)
        self.assertTrue(audit["complete"])
        self.assertFalse(audit["always_first"] and audit["always_last"])
        # Cell-major order is complete but always puts the first cell at
        # position 0: the audit must notice the systematic extreme.
        legacy = _blocked_run_order(self.CELLS, self.SEEDS, 11,
                                    shuffle=False)
        legacy_audit = _run_order_audit(
            legacy, self.CELLS, self.SEEDS, randomized=False)
        self.assertTrue(legacy_audit["complete"])
        self.assertEqual(legacy_audit["always_first"], ["none"])
        self.assertFalse(legacy_audit["balanced"])
        # A missing row is a real schedule bug, not a style issue.
        broken = [row for row in complete if not (
            row[0] == "calib" and row[1] == 0)]
        broken_audit = _run_order_audit(broken, self.CELLS, self.SEEDS)
        self.assertFalse(broken_audit["complete"])
        self.assertTrue(broken_audit["missing"])

    def test_audit_flags_unplanned_cells_and_invalid_slots(self):
        complete = _blocked_run_order(self.CELLS, self.SEEDS, 11)
        extra = list(complete) + [("unplanned", 0, 0, len(self.CELLS))]
        extra_audit = _run_order_audit(extra, self.CELLS, self.SEEDS)
        self.assertFalse(extra_audit["complete"])
        self.assertEqual(extra_audit["unexpected"][0]["cell"], "unplanned")
        # Every declared cell present, but all at slot 0: the block design
        # is broken even though the presence counts look complete.
        bad_slots = [(cell, seed, block, 0)
                     for cell, seed, block, _ in complete]
        slot_audit = _run_order_audit(bad_slots, self.CELLS, self.SEEDS)
        self.assertFalse(slot_audit["complete"])
        self.assertTrue(slot_audit["invalid_positions"])
        self.assertFalse(slot_audit["balanced"])

    def test_audit_rejects_undeclared_or_misblocked_seeds(self):
        complete = _blocked_run_order(self.CELLS, self.SEEDS, 11)
        # The cell/block/position structure is still complete, but every row
        # claims a seed that was never declared: the schedule is not the one
        # the audit was asked to certify.
        bad_seed = [(cell, 99, block, pos)
                    for cell, _seed, block, pos in complete]
        audit = _run_order_audit(bad_seed, self.CELLS, self.SEEDS)
        self.assertFalse(audit["complete"])
        self.assertFalse(audit["balanced"])
        self.assertTrue(audit["seed_mismatches"])
        self.assertTrue(audit["unexpected_seeds"])
        # Using the wrong declared seed for a block is also a mismatch even
        # though that seed appears in the declared set.
        swapped = [(cell, self.SEEDS[(block + 1) % len(self.SEEDS)],
                    block, pos)
                   for cell, _seed, block, pos in complete]
        self.assertFalse(
            _run_order_audit(swapped, self.CELLS, self.SEEDS)["complete"])
        # Two blocks labelled with the same seed are not independent blocks
        # even when every cell appears once in each.
        duplicate_seed = _blocked_run_order(
            self.CELLS, (0, 0), 11)
        self.assertFalse(
            _run_order_audit(duplicate_seed, self.CELLS, (0, 0))["complete"])
        # A fractional block/position would otherwise be silently truncated
        # to an integer and could still look like a valid complete block.
        fractional = [(cell, seed, float(block), pos)
                      for cell, seed, block, pos in complete]
        fractional[0] = (fractional[0][0], fractional[0][1], 0.5,
                         fractional[0][3])
        audit = _run_order_audit(fractional, self.CELLS, self.SEEDS)
        self.assertFalse(audit["complete"])
        self.assertTrue(audit["invalid_indices"])

    def test_two_block_random_schedule_is_not_over_flagged(self):
        # With only two blocks, a random schedule can put one arm first in
        # both by chance; the family-wise check must not call that a design
        # failure (p = 2*(1/2)^2 = 0.5).
        cells = ("a", "b")
        order = [("a", 0, 0, 0), ("b", 0, 0, 1),
                 ("a", 1, 1, 0), ("b", 1, 1, 1)]
        audit = _run_order_audit(order, cells, (0, 1), randomized=True)
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["always_first"], ["a"])
        self.assertGreaterEqual(audit["extreme_p_value"], 0.05)
        self.assertTrue(audit["balanced"])


class ControlledAuditProvenanceTest(unittest.TestCase):
    CFG = {
        "vocab_size": 64, "d_model": 32, "n_heads": 4, "n_layers": 2,
        "max_len": 64, "dropout": 0.0, "gate": "calib", "norm": "pre",
        "layer_scale": None, "calib_use_len": True, "calib_use_ent": True,
        "calib_shuffle_ent": False, "calib_ent_norm": True,
    }

    def _run(self, **cfg_overrides):
        cfg = dict(self.CFG)
        cfg.update(cfg_overrides)
        return {
            "seed": 0, "config": {"d_model": 32}, "lr": 1e-3,
            "eval_seed": 2024, "stats_seed": 7, "steps_run": 10,
            "stopped_early": False, "restored_best": False,
            "data_fingerprints": None, "training_recipe": {},
            "params": 100, "gate": "calib", "norm": "pre",
            "layer_scale": None, "calib_variant": "full",
            "model_cfg": cfg,
        }

    def test_malformed_calib_flag_does_not_certify_the_audit(self):
        # ``"False"`` is truthy; coercing it would read the record as the
        # full arm and let the contrast pass as controlled.  The malformed
        # flag must surface as a control mismatch instead.
        good = self._run()
        bad = self._run(calib_use_len="False")
        self.assertEqual(_calib_cell(bad), "model_cfg_malformed")
        audit = _controlled_audit([good], [bad])
        self.assertFalse(audit["controlled"])
        self.assertTrue(audit["mismatches"])

    def test_non_boolean_entropy_normalisation_flag_fails_closed(self):
        for raw in ("False", 0, None):
            with self.subTest(raw=raw):
                run = self._run(calib_ent_norm=raw)
                self.assertEqual(_calib_cell(run), "model_cfg_malformed")

    def test_malformed_training_recipe_cannot_pass_the_recipe_audit(self):
        good = self._run()
        bad = self._run()
        bad["training_recipe"] = "not-a-mapping"
        audit = _controlled_audit([good], [bad])
        self.assertFalse(audit["controlled"])
        self.assertTrue(any("training_recipe" in msg
                            for msg in audit["mismatches"]))


class PilotSdTest(unittest.TestCase):
    def test_largest_rung_sd_is_used(self):
        verdict = {
            "depth_ladder": {"by_depth": {
                "1": {"accuracy_by_seed": {"0": 0.80, "1": 0.82},
                      "unmeasured": False},
                "2": {"accuracy_by_seed": {"0": 0.90, "1": 0.94},
                      "unmeasured": False},
            }},
        }
        sd, source = _pilot_accuracy_sd(verdict)
        self.assertAlmostEqual(sd, 0.028284271247461902, places=12)
        self.assertEqual(source, "depth_ladder.d2")

    def test_missing_or_unmeasured_pilot_abstains(self):
        self.assertTrue(math.isnan(_pilot_accuracy_sd({})[0]))
        self.assertTrue(math.isnan(_pilot_accuracy_sd(None)[0]))
        verdict = {"depth_ladder": {"by_depth": {
            "1": {"accuracy_by_seed": {"0": 0.8, "1": 0.9},
                  "unmeasured": True},
        }}}
        self.assertTrue(math.isnan(_pilot_accuracy_sd(verdict)[0]))


class PairedPilotSdTest(unittest.TestCase):
    """The power plan must use the paired-difference variance."""

    @staticmethod
    def _verdict(arm_sd=0.01, adjacent_sd=None):
        # Two points give the requested sample SD: values are the mean
        # plus/minus sd/sqrt(2).
        delta = arm_sd / math.sqrt(2.0)
        payload = {
            "depth_ladder": {"by_depth": {
                "2": {"accuracy_by_seed": {"0": 0.9 - delta,
                                            "1": 0.9 + delta},
                      "unmeasured": False},
            }},
        }
        if adjacent_sd is not None:
            payload["depth_ladder"]["adjacent_contrasts"] = {
                "1->2": {"n": 2, "sd": adjacent_sd},
            }
        return payload

    def test_independent_arm_bound_is_the_fallback(self):
        sd, source, components = _pilot_paired_accuracy_sd(
            self._verdict(arm_sd=0.02))
        self.assertAlmostEqual(sd, math.sqrt(2.0) * 0.02, places=12)
        self.assertIn("sqrt(2)", source)
        kinds = {c["kind"] for c in components}
        self.assertIn("single_arm", kinds)
        self.assertIn("independent_arms", kinds)

    def test_largest_empirical_adjacent_paired_sd_wins(self):
        sd, source, _ = _pilot_paired_accuracy_sd(
            self._verdict(arm_sd=0.01, adjacent_sd=0.08))
        self.assertAlmostEqual(sd, 0.08, places=12)
        self.assertIn("adjacent", source)

    def test_abstains_without_a_pilot(self):
        sd, source, components = _pilot_paired_accuracy_sd({})
        self.assertTrue(math.isnan(sd))
        self.assertIsNone(source)
        self.assertEqual(components, [])

    def test_accuracy_level_pools_pilot_seeds(self):
        level, source = _pilot_accuracy_level(self._verdict(arm_sd=0.01))
        self.assertAlmostEqual(level, 0.9, places=12)
        self.assertIn("2 pilot", source)


class DesignPlanTest(unittest.TestCase):
    @staticmethod
    def _verdict(sd=0.001):
        # Two seeds with this sample SD; the helper reads the per-seed
        # values, so the pilot is an observed spread, not a hard-coded field.
        return {"depth_ladder": {"by_depth": {
            "2": {"accuracy_by_seed": {"0": 0.9, "1": 0.9 + sd},
                  "unmeasured": False},
        }}}

    def test_plan_is_written_before_training_and_flags_underpower(self):
        settings = Settings.quick()  # n=2 seeds
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _design_sensitivity_plan(
                    settings, self._verdict(sd=0.05), out, "e1")
            path = out / "e1_design_plan.json"
            self.assertTrue(path.exists())
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored["experiment"], "e1")
            self.assertEqual(stored["planned_n"], 2)
            self.assertIn("rows", stored)
            self.assertIn("accuracy", stored["accuracy_underpowered"])
            self.assertIs(stored["all_accuracy_plans_powered"], False)
            self.assertEqual(payload["accuracy_underpowered"],
                             stored["accuracy_underpowered"])

    def test_plan_passes_when_pilot_sd_is_small(self):
        settings = Settings.quick()
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _design_sensitivity_plan(
                    settings, self._verdict(sd=0.0001), pathlib.Path(tmp),
                    "e2")
            self.assertIs(payload["all_accuracy_plans_powered"], True)

    def test_non_accuracy_rows_report_no_pilot_not_a_fabricated_sd(self):
        settings = Settings.quick()
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _design_sensitivity_plan(
                    settings, self._verdict(), pathlib.Path(tmp), "e3")
            rows = {row["metric"]: row for row in payload["rows"]}
            self.assertEqual(rows["gate_depth_ratio"]["status"], "no_pilot")
            self.assertTrue(math.isnan(
                rows["gate_depth_ratio"]["pilot_sd"]))
            self.assertIsNone(rows["gate_depth_ratio"]["required_seeds"])

    def test_power_helpers_agree_on_direction(self):
        # More seeds or a larger SESOI can only increase power; a null
        # (non-inferiority) plan needs the bound passed explicitly.
        low = _power_at_effect("superiority", 0.01, 0.01, 5, 0.05)
        high = _power_at_effect("superiority", 0.01, 0.01, 50, 0.05)
        self.assertGreater(high, low)
        noninf = _power_at_effect(
            "noninferior", 0.0, 0.01, 50, 0.05, bound=0.01)
        self.assertGreater(noninf, 0.5)
        equiv = _power_at_effect(
            "equivalence", 0.0, 0.01, 50, 0.05, bound=0.01)
        self.assertGreater(equiv, 0.5)

    def test_plan_uses_paired_sd_not_the_single_arm_sd(self):
        # Single-arm pilot looks tight (SD~7e-5) but the observed paired
        # adjacent-depth difference has SD=0.2: the plan must follow the
        # paired variance and flag the design, not certify it from the
        # optimistic single-arm scale.
        settings = Settings.quick()
        verdict = PairedPilotSdTest._verdict(
            arm_sd=0.0001, adjacent_sd=0.2)
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _design_sensitivity_plan(
                    settings, verdict, pathlib.Path(tmp), "e1")
        self.assertAlmostEqual(payload["pilot_paired_accuracy_sd"], 0.2,
                               places=12)
        row = next(r for r in payload["rows"]
                   if r["metric"] == "accuracy")
        self.assertAlmostEqual(row["pilot_paired_sd"], 0.2, places=12)
        self.assertLess(row["required_seeds_single_arm"],
                        row["required_seeds"])
        self.assertEqual(row["status"], "underpowered")
        self.assertIn("accuracy", payload["accuracy_underpowered"])

    def test_finite_eval_stream_floor_is_reported_as_a_companion(self):
        # A 128-item stream at p~0.9 cannot resolve a 0.01 accuracy SESOI
        # under independent arm errors; more seeds cannot remove that
        # floor, so the plan must expose it instead of claiming power.
        settings = Settings.quick()  # eval 64 x 2 = 128 items
        verdict = PairedPilotSdTest._verdict(
            arm_sd=0.0001, adjacent_sd=0.0001)
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _design_sensitivity_plan(
                    settings, verdict, pathlib.Path(tmp), "e1")
        row = next(r for r in payload["rows"]
                   if r["metric"] == "accuracy")
        self.assertEqual(row["stream_items"], 128)
        self.assertGreater(row["stream_items_for_sesoi"], 128)
        self.assertIs(row["stream_limited_worst_case"], True)
        self.assertIn("accuracy",
                      payload["accuracy_stream_limited_worst_case"])
        self.assertGreater(payload["stream_item_sd_floor"], 0.0)

    def test_malformed_seed_floor_fails_closed(self):
        settings = Settings(seeds=(0, 1, 2))
        # An explicit malformed per-mark count is not treated as absent:
        # it drops only that mark's floor to zero and makes it ineligible.
        audit = _verdict_seed_floor(
            {"seed_floor": {"n_seeds": 3, "hb2_n_seeds": None}}, settings)
        self.assertEqual(audit["hb2_n_seeds"], 0)
        self.assertFalse(audit["hb2_claim_eligible"])
        self.assertEqual(audit["hb4_n_seeds"], 3)
        self.assertTrue(audit["hb4_claim_eligible"])
        # A malformed total corrupts every fallback count.
        audit = _verdict_seed_floor(
            {"seed_floor": {"n_seeds": None}}, settings)
        self.assertEqual(audit["n_seeds"], 0)
        self.assertFalse(any(audit[f"{key}_claim_eligible"]
                             for key in ("hb2", "hb4", "hb6", "hb7")))
        # Non-bool eligibility flags fail closed instead of being truthy.
        audit = _verdict_seed_floor(
            {"seed_floor": {"n_seeds": 3,
                            "hb2_claim_eligible": "False"}}, settings)
        self.assertFalse(audit["hb2_claim_eligible"])

    def test_exact_selection_tie_is_unresolved_without_an_se(self):
        from cage.experiments import _selection_reliability

        rows = [{"val_accuracy_mean": 0.5, "val_accuracy_std": 0.0,
                 "n_seeds": 1} for _ in range(3)]
        audit = _selection_reliability(rows)
        self.assertIs(audit["within_noise"], True)
        self.assertIs(audit["reliable"], False)

    def test_settings_reject_invalid_design_fields(self):
        with self.assertRaises(ValueError):
            Settings(design_alpha=0.0)
        with self.assertRaises(ValueError):
            Settings(power_target=1.0)
        with self.assertRaises(ValueError):
            Settings(randomize_run_order="yes")
        with self.assertRaises(ValueError):
            Settings(run_order_seed=True)

    def test_settings_reject_invalid_numeric_configuration(self):
        for field, values in {
            "batch": (0, -1, 1.5, True, "8"),
            "steps": (0, -1, 1.5, True),
            "eval_batch": (0, 1.5, True),
            "eval_batches": (0, 1.5, True),
            "stats_batch": (0, 1.5, True),
            "stats_batches": (0, 1.5, True),
            "val_batches": (0, 1.5, True),
            "probe_batch": (0, 1.5, True),
            "probe_steps": (0, 1.5, True),
            "probe_eval_batch": (0, 1.5, True),
            "probe_eval_batches": (0, 1.5, True),
            "overfit_steps": (0, 1.5, True),
            "overfit_batch": (0, 1.5, True),
            "curve_points": (-1, 1.5, True),
            "d_model": (0, 1.5, True),
            "n_heads": (0, 1.5, True),
            "n_layers": (0, 1.5, True),
            "dropout": (-0.1, 1.1, float("nan"), True),
            "negative_control_tol": (-0.1, float("nan"), True),
            "overfit_lr": (0.0, -1.0, float("nan"), True),
            "lr": (0.0, -1.0, float("nan"), True),
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        Settings(**{field: value})
        with self.assertRaises(ValueError):
            Settings(d_model=10, n_heads=4)
        for values in ((), (0,), (True,), (1.5,)):
            with self.subTest(depth_ladder=values):
                with self.assertRaises(ValueError):
                    Settings(depth_ladder=values)
        for values in ((0.0,), (-1.0,), (float("nan"),), (True,)):
            with self.subTest(depth_step_multipliers=values):
                with self.assertRaises(ValueError):
                    Settings(depth_step_multipliers=values)
        # Empty optional grids keep the documented fallback behaviour.
        self.assertEqual(Settings(probe_steps_grid=()).probe_steps_grid, ())
        self.assertEqual(
            Settings(depth_step_multipliers=()).depth_step_multipliers, ())


if __name__ == "__main__":
    unittest.main()
