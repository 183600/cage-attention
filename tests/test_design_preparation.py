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
from unittest import mock

from cage import experiments
from cage.experiments import (
    DesignPlanError,
    Settings,
    _calib_cell,
    _controlled_audit,
    _blocked_run_order,
    _design_plan_fingerprint,
    _design_plan_outcomes_exist,
    _design_sensitivity_plan,
    _pilot_accuracy_level,
    _pilot_accuracy_sd,
    _pilot_paired_accuracy_sd,
    _power_at_effect,
    _power_z,
    _realized_order_audit,
    _report_realized_run_order,
    _required_seeds,
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

    def test_malformed_schedule_rows_are_reported_not_raised(self):
        complete = _blocked_run_order(self.CELLS, self.SEEDS, 11)
        for broken in (list(complete) + [("none", 0, 0)],
                       list(complete) + [None],
                       list(complete) + [object()]):
            with self.subTest(row=broken[-1]):
                audit = _run_order_audit(broken, self.CELLS, self.SEEDS)
                self.assertFalse(audit["complete"])
                self.assertTrue(audit["invalid_indices"])

    def test_non_iterable_schedule_is_not_certified(self):
        audit = _run_order_audit(None, self.CELLS, self.SEEDS)
        self.assertFalse(audit["complete"])
        self.assertTrue(audit["invalid_indices"])
        self.assertEqual(audit["n_runs"], 0)

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


class ReferenceLrValidationTest(unittest.TestCase):
    def test_boolean_reference_lr_is_not_accepted_as_one(self):
        from cage.experiments import (
            _e1_lr_sensitivity,
            _e1_per_variant_tuning,
            _e2_per_variant_tuning,
        )
        settings = Settings.quick()
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                e1 = _e1_per_variant_tuning(settings, out, True)
                e2 = _e2_per_variant_tuning(settings, out, True, {}, [3])
                lr = _e1_lr_sensitivity(settings, out, True)
        for payload in (e1, e2, lr):
            self.assertFalse(payload["available"])
            self.assertIn("bool", payload["reason"])


class FiniteScalarTest(unittest.TestCase):
    def test_overflowing_and_boolean_values_are_not_finite_statistics(self):
        from cage.experiments import _finite

        self.assertFalse(_finite(10 ** 400))
        self.assertFalse(_finite(True))
        self.assertFalse(_finite(None))
        self.assertTrue(_finite(0))
        self.assertTrue(_finite(-1.25))


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

    def test_unpowerable_accuracy_row_is_not_reported_as_powered(self):
        settings = Settings.quick()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(
                    experiments, "_required_seeds",
                    return_value=float("inf")):
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _design_sensitivity_plan(
                    settings, self._verdict(sd=0.05), pathlib.Path(tmp),
                    "e1")
        row = next(r for r in payload["rows"] if r["metric"] == "accuracy")
        self.assertEqual(row["status"], "unpowerable")
        self.assertIn("accuracy", payload["accuracy_unpowerable"])
        self.assertIs(payload["all_accuracy_plans_powered"], False)

    def test_degenerate_pilot_is_not_reported_as_fully_powered(self):
        settings = Settings.quick()
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _design_sensitivity_plan(
                    settings, self._verdict(sd=0.0), pathlib.Path(tmp),
                    "e1")
        self.assertIs(payload["all_accuracy_plans_powered"], False)
        statuses = {row["metric"]: row["status"] for row in payload["rows"]}
        self.assertEqual(statuses["accuracy"], "pilot_degenerate")

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

    def test_plan_rejects_malformed_planning_probabilities(self):
        for field in ("design_alpha", "power_target"):
            settings = Settings.quick()
            setattr(settings, field, object())
            with tempfile.TemporaryDirectory() as tmp:
                with self.subTest(field=field):
                    with self.assertRaises(ValueError):
                        _design_sensitivity_plan(
                            settings, {}, pathlib.Path(tmp), "e1")

    def test_settings_reject_invalid_design_fields(self):
        with self.assertRaises(ValueError):
            Settings(design_alpha=0.0)
        with self.assertRaises(ValueError):
            Settings(power_target=1.0)
        with self.assertRaises(ValueError):
            Settings(randomize_run_order="yes")
        with self.assertRaises(ValueError):
            Settings(run_order_seed=True)

    def test_settings_reject_byte_encoded_tuple_fields(self):
        # Bytes are iterable, so a byte string would otherwise be read as
        # its integer byte values (e.g. b"\x01\x02" -> depth ladder 1/2).
        for field in ("depth_ladder", "probe_steps_grid", "lr_grid",
                      "probe_lr_grid", "calib_tau_grid",
                      "depth_step_multipliers"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    Settings(**{field: b"\x01\x02"})
        self.assertEqual(Settings(lr_grid=[1e-3]).lr_grid, (1e-3,))

    def test_settings_reject_overflowing_grid_values(self):
        for field in ("lr_grid", "probe_lr_grid", "calib_tau_grid"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    Settings(**{field: (10 ** 400,)})

    def test_settings_reject_non_sequence_or_byte_seeds(self):
        # ``bytes`` is iterable and would otherwise silently turn into its
        # integer byte values; a bare int/None is not a seed sequence.
        for bad in (5, None, object(), b"\x00\x01", {"seed": 0}):
            with self.subTest(seeds=bad):
                with self.assertRaises(ValueError):
                    Settings(seeds=bad)
        # A normal iterable is still canonicalised to a tuple of ints.
        self.assertEqual(Settings(seeds=range(3)).seeds, (0, 1, 2))

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
            "max_error_cases": (-1, 1.5, True, "8"),
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
        # The probe's lr grid is used verbatim by ``tune_linear_probe``;
        # an empty/invalid grid fails only after the baseline runs have
        # been written, so it must be rejected at construction like the
        # transformer grids.
        for values in ((), (0.0,), (-1.0,), (float("nan"),), (True,),
                       ("1e-3",)):
            with self.subTest(probe_lr_grid=values):
                with self.assertRaises(ValueError):
                    Settings(probe_lr_grid=values)
        # Zero error cases is the documented "collect no cases" value.
        self.assertEqual(Settings(max_error_cases=0).max_error_cases, 0)
        # Empty optional grids keep the documented fallback behaviour.
        self.assertEqual(Settings(probe_steps_grid=()).probe_steps_grid, ())
        self.assertEqual(
            Settings(depth_step_multipliers=()).depth_step_multipliers, ())


class ClaimFloorPlanTest(unittest.TestCase):
    """Power and claim eligibility must be separate preparation readouts."""

    @staticmethod
    def _verdict(sd=0.001):
        return {"depth_ladder": {"by_depth": {
            "2": {"accuracy_by_seed": {"0": 0.9, "1": 0.9 + sd},
                  "unmeasured": False},
        }}}

    def test_below_floor_plan_is_powered_but_not_claim_eligible(self):
        settings = Settings.quick()  # two seeds, claim floor is three
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                payload = _design_sensitivity_plan(
                    settings, self._verdict(sd=0.0001), out, "e1")
            history = out / "e1_design_plan_history.jsonl"
            self.assertTrue(history.exists())
            events = [json.loads(line) for line in
                      history.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[0]["event"], "created")
        self.assertIs(payload["all_accuracy_plans_powered"], True)
        self.assertEqual(payload["planned_n"], 2)
        self.assertEqual(payload["min_seeds_for_claim"], 3)
        self.assertIs(payload["planned_n_meets_claim_floor"], False)
        self.assertEqual(payload["claim_floor_shortfall"], 1)
        self.assertIs(payload["all_accuracy_claims_seed_eligible"], False)
        self.assertIs(payload["all_accuracy_claims_ready"], False)
        self.assertIn("accuracy", payload["claim_floor_blocked_metrics"])
        for row in payload["rows"]:
            self.assertIs(row["claim_floor_met"], False)
        self.assertIn("below the pre-registered min_seeds_for_claim",
                      printed.getvalue())
        self.assertRegex(payload["design_sha256"], r"^[0-9a-f]{16}$")

    def test_no_pilot_does_not_hide_an_unmet_claim_floor(self):
        settings = Settings.quick()
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _design_sensitivity_plan(
                    settings, {}, pathlib.Path(tmp), "e3")
        self.assertIs(payload["all_accuracy_plans_powered"], None)
        self.assertIs(payload["all_accuracy_claims_seed_eligible"], False)
        self.assertIs(payload["all_accuracy_claims_ready"], False)

    def test_plan_at_or_above_floor_is_claim_eligible(self):
        settings = Settings(seeds=(0, 1, 2))
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _design_sensitivity_plan(
                    settings, self._verdict(sd=0.0001),
                    pathlib.Path(tmp), "e2")
        self.assertIs(payload["planned_n_meets_claim_floor"], True)
        self.assertEqual(payload["claim_floor_shortfall"], 0)
        self.assertEqual(payload["claim_floor_blocked_metrics"], [])
        self.assertIs(payload["all_accuracy_claims_seed_eligible"], True)
        self.assertIs(payload["all_accuracy_claims_ready"], True)


class DesignPlanFreezeTest(unittest.TestCase):
    """A design may be revised while preparing, never after outcome rows."""

    @staticmethod
    def _verdict():
        return {"depth_ladder": {"by_depth": {
            "2": {"accuracy_by_seed": {"0": 0.9, "1": 0.905},
                  "unmeasured": False},
        }}}

    def test_before_outcomes_revision_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            first = Settings.quick()
            with contextlib.redirect_stdout(io.StringIO()):
                plan_a = _design_sensitivity_plan(
                    first, self._verdict(), out, "e1")
            second = Settings.quick()
            second.seeds = (0, 1, 2)
            self.assertFalse(_design_plan_outcomes_exist(out, "e1"))
            with contextlib.redirect_stdout(io.StringIO()):
                plan_b = _design_sensitivity_plan(
                    second, self._verdict(), out, "e1")
            self.assertNotEqual(plan_a["design_sha256"],
                                plan_b["design_sha256"])
            self.assertEqual(plan_b["supersedes_design_sha256"],
                             plan_a["design_sha256"])
            events = [json.loads(line) for line in
                      (out / "e1_design_plan_history.jsonl")
                      .read_text(encoding="utf-8").splitlines()]
            self.assertEqual([e["event"] for e in events],
                             ["created", "amended"])

    def test_unreadable_outcome_path_is_treated_as_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            (out / "e1_runs.jsonl").mkdir()
            self.assertTrue(_design_plan_outcomes_exist(out, "e1"))

    def test_after_outcomes_a_changed_design_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            settings = Settings.quick()
            with contextlib.redirect_stdout(io.StringIO()):
                plan = _design_sensitivity_plan(
                    settings, self._verdict(), out, "e1")
            (out / "e1_runs.jsonl").write_text(
                json.dumps({"value": 1}) + "\n", encoding="utf-8")
            changed = Settings.quick()
            changed.seeds = (0, 1, 2)
            with self.assertRaises(DesignPlanError):
                _design_sensitivity_plan(
                    changed, self._verdict(), out, "e1")
            # The refused revision must not have rewritten the registered
            # artifact: the prospective hash is still the original one.
            stored = json.loads(
                (out / "e1_design_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["design_sha256"],
                             plan["design_sha256"])

    def test_after_outcomes_an_identical_design_may_be_rechecked(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            settings = Settings.quick()
            with contextlib.redirect_stdout(io.StringIO()):
                plan = _design_sensitivity_plan(
                    settings, self._verdict(), out, "e1")
            (out / "e1_runs.jsonl").write_text("{}\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                again = _design_sensitivity_plan(
                    settings, self._verdict(), out, "e1")
            self.assertEqual(again["design_sha256"],
                             plan["design_sha256"])
            self.assertNotIn("supersedes_design_sha256", again)

    def test_after_outcomes_a_malformed_plan_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            (out / "e1_runs.jsonl").write_text("{}\n", encoding="utf-8")
            (out / "e1_design_plan.json").write_text(
                "{not json", encoding="utf-8")
            with self.assertRaises(DesignPlanError):
                _design_sensitivity_plan(
                    Settings.quick(), self._verdict(), out, "e1")

    def test_outcomes_without_a_plan_cannot_get_a_post_hoc_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            (out / "e1_runs.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(DesignPlanError):
                _design_sensitivity_plan(
                    Settings.quick(), self._verdict(), out, "e1")

    def test_before_outcomes_a_malformed_plan_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            (out / "e1_design_plan.json").write_text(
                "{not json", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _design_sensitivity_plan(
                    Settings.quick(), self._verdict(), out, "e1")
            self.assertRegex(payload["design_sha256"], r"^[0-9a-f]{16}$")
            stored = json.loads(
                (out / "e1_design_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["design_sha256"],
                             payload["design_sha256"])

    def test_fingerprint_ignores_invocation_provenance(self):
        base = {"experiment": "e1", "planned_n": 2,
                "planned_seeds": [0, 1], "alpha": 0.05,
                "power_target": 0.8, "min_seeds_for_claim": 3,
                "planned_eval_items": 128, "hypotheses_sha256": "abc",
                "rows": [{"metric": "accuracy", "sesoi": 0.01,
                          "mode": "superiority", "hypotheses": ["H1a"]}],
                "run_id": "run-a", "created_utc": "yesterday"}
        changed = dict(base, run_id="run-b", created_utc="today")
        self.assertEqual(_design_plan_fingerprint(base),
                         _design_plan_fingerprint(changed))
        changed["planned_n"] = 3
        self.assertNotEqual(_design_plan_fingerprint(base),
                            _design_plan_fingerprint(changed))
        self.assertIsNone(_design_plan_fingerprint("not a plan"))


class RealizedRunOrderAuditTest(unittest.TestCase):
    CELLS = ("none", "query")
    SEEDS = (0, 1)

    def _records(self, order):
        return [{"cell": cell, "seed": seed, "run_block": block,
                 "run_position": position}
                for cell, seed, block, position in order]

    def test_matching_records_are_certified(self):
        order = _blocked_run_order(self.CELLS, self.SEEDS, 7,
                                   shuffle=False)
        audit = _realized_order_audit(
            self._records(order), order, self.CELLS, self.SEEDS,
            lambda run: run["cell"])
        self.assertIs(audit["matches_schedule"], True)
        self.assertEqual(audit["n_observed"], len(order))
        self.assertEqual(audit["missing"], [])

    def test_missing_reordered_and_duplicate_rows_are_flagged(self):
        order = _blocked_run_order(self.CELLS, self.SEEDS, 7,
                                   shuffle=False)
        records = self._records(order)
        missing = _realized_order_audit(
            records[:-1], order, self.CELLS, self.SEEDS,
            lambda run: run["cell"])
        self.assertIs(missing["matches_schedule"], False)
        self.assertEqual(len(missing["missing"]), 1)

        reordered = self._records(order)
        reordered[0] = dict(reordered[0], run_position=99)
        changed = _realized_order_audit(
            reordered, order, self.CELLS, self.SEEDS,
            lambda run: run["cell"])
        self.assertIs(changed["matches_schedule"], False)
        self.assertTrue(changed["missing"])
        self.assertTrue(changed["unexpected"])

        duplicated = self._records(order) + [self._records(order)[0]]
        dup = _realized_order_audit(
            duplicated, order, self.CELLS, self.SEEDS,
            lambda run: run["cell"])
        self.assertIs(dup["matches_schedule"], False)
        self.assertEqual(len(dup["duplicates"]), 1)

    def test_boolean_and_fractional_indices_are_invalid(self):
        order = _blocked_run_order(self.CELLS, self.SEEDS, 7,
                                   shuffle=False)
        records = self._records(order)
        records[0] = dict(records[0], run_block=True)
        records[1] = dict(records[1], run_position=1.5)
        audit = _realized_order_audit(
            records, order, self.CELLS, self.SEEDS,
            lambda run: run["cell"])
        self.assertIs(audit["matches_schedule"], False)
        self.assertEqual(len(audit["invalid"]), 2)

    def test_malformed_schedule_rows_are_not_certified(self):
        order = _blocked_run_order(self.CELLS, self.SEEDS, 7,
                                   shuffle=False)
        broken = list(order)
        broken[0] = ("none", True, 0, 0)
        broken[1] = ("query", 1, 1)
        audit = _realized_order_audit(
            self._records(order), broken, self.CELLS, self.SEEDS,
            lambda run: run["cell"])
        self.assertIs(audit["matches_schedule"], False)
        self.assertEqual(len(audit["invalid_order"]), 2)

    def test_duplicate_schedule_tuple_is_not_certified(self):
        order = _blocked_run_order(self.CELLS, self.SEEDS, 7,
                                   shuffle=False)
        duplicated_order = list(order) + [order[0]]
        audit = _realized_order_audit(
            self._records(order), duplicated_order, self.CELLS,
            self.SEEDS, lambda run: run["cell"])
        self.assertIs(audit["matches_schedule"], False)
        self.assertEqual(len(audit["duplicate_order"]), 1)
        self.assertTrue(audit["duplicate_order"][0].get("cell"))

    def test_schedule_outside_declared_cells_or_seeds_is_not_certified(self):
        order = _blocked_run_order(self.CELLS, self.SEEDS, 7,
                                   shuffle=False)
        for cell, seed in (("unplanned", self.SEEDS[0]),
                           (self.CELLS[0], 99)):
            with self.subTest(cell=cell, seed=seed):
                broken = list(order)
                broken[0] = (cell, seed, 0, 0)
                audit = _realized_order_audit(
                    self._records(order), broken, self.CELLS,
                    self.SEEDS, lambda run: run["cell"])
                self.assertIs(audit["matches_schedule"], False)
                self.assertTrue(audit["invalid_order"])

    def test_report_writes_the_audit_next_to_the_runs(self):
        order = _blocked_run_order(self.CELLS, self.SEEDS, 7,
                                   shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                payload = _report_realized_run_order(
                    out, "e1", self._records(order), order,
                    self.CELLS, self.SEEDS, lambda run: run["cell"])
            stored = json.loads(
                (out / "e1_run_order_realized.json").read_text(
                    encoding="utf-8"))
        self.assertIs(payload["matches_schedule"], True)
        self.assertEqual(stored["matches_schedule"], True)
        self.assertIn("matches the frozen", printed.getvalue())


class PowerArgumentValidationTest(unittest.TestCase):
    def test_boolean_or_fractional_seed_count_is_not_power_planned(self):
        for bad_n in (True, 2.5, 0, -1):
            with self.subTest(n=bad_n):
                self.assertTrue(math.isnan(_power_at_effect(
                    "superiority", 0.01, 0.01, bad_n, 0.05)))

    def test_non_positive_null_bound_is_not_power_planned(self):
        for mode in ("equivalence", "noninferior"):
            for bound in (0.0, -0.1):
                with self.subTest(mode=mode, bound=bound):
                    self.assertTrue(math.isnan(_power_at_effect(
                        mode, 0.0, 0.01, 10, 0.05, bound=bound)))

    def test_unknown_mode_or_probability_abstains(self):
        self.assertTrue(math.isnan(_power_z("bogus", 0.05, 0.8)))
        self.assertTrue(math.isnan(_power_z("superiority", True, 0.8)))
        self.assertIsNone(_required_seeds(
            "bogus", 0.1, 0.01, 0.05, 0.8))


if __name__ == "__main__":
    unittest.main()
