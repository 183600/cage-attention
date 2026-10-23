"""E1 comparison/ablation design: shared schedule and complete factorial.

The E1 ablation arms used to be trained in a second grid after the four
gate variants.  A slow machine drift between the two invocations would then
be aliased with the ablation factor, defeating the complete-block
randomisation the project relies on.  These tests pin the single shared
schedule and the interaction row that completes the calibration-term 2x2.
"""

from __future__ import annotations

import contextlib
import io
import unittest

from cage.experiments import (
    GATES,
    HYPOTHESES,
    Settings,
    _E1_ABLATION_LABELS,
    _blocked_run_order,
    _competence_check,
    _e1_cell_of,
    _e1_cell_plan,
    _model_cfg,
    _paired_benchmark,
    _paired_did,
    _paired_diff,
    _one_sample_contrast,
    _print_conf_by_error_type,
    _print_structural_decomposition,
    _run_order_audit,
)


class E1SharedScheduleTest(unittest.TestCase):
    def test_plan_covers_every_gate_and_ablation_cell_once(self):
        plan = _e1_cell_plan()
        expected = tuple(GATES) + _E1_ABLATION_LABELS
        self.assertEqual(plan["cells"], expected)
        self.assertEqual(len(set(plan["cells"])), len(expected))
        for cell in plan["cells"]:
            self.assertIn(cell, plan["gate_of"])
        for gate in GATES:
            self.assertEqual(plan["gate_of"][gate], gate)
            self.assertEqual(plan["overrides"].get(gate), None)
        for label in _E1_ABLATION_LABELS:
            self.assertEqual(plan["gate_of"][label], "calib")
            self.assertIn(label, plan["overrides"])
        self.assertEqual(plan["variant_of"]["calib"], "full")

    def test_main_and_ablation_cells_share_every_seed_block(self):
        # The point of the change: full calib and the ablated/placebo arms
        # appear in the *same* block, so a time-varying machine state cannot
        # be aliased with the ablation factor across two separate grids.
        plan = _e1_cell_plan()
        seeds = (0, 1, 2)
        order = _blocked_run_order(plan["cells"], seeds, 20240517)
        audit = _run_order_audit(order, plan["cells"], seeds)
        self.assertTrue(audit["complete"])
        by_block = {}
        for cell, seed, block, _position in order:
            by_block.setdefault(block, set()).add(cell)
        for block, cells in by_block.items():
            self.assertEqual(cells, set(plan["cells"]))
            self.assertIn("calib", cells)
            for label in _E1_ABLATION_LABELS:
                self.assertIn(label, cells)

    def test_cell_recovery_from_new_and_legacy_records(self):
        self.assertEqual(_e1_cell_of({"e1_cell": "calib_len_only"}),
                         "calib_len_only")
        self.assertEqual(_e1_cell_of({"gate": "calib",
                                     "calib_variant": "full"}), "calib")
        self.assertEqual(_e1_cell_of({"gate": "calib",
                                     "calib_variant": "calib_ent_only"}),
                         "calib_ent_only")
        self.assertEqual(_e1_cell_of({"gate": "query"}), "query")
        self.assertIsNone(_e1_cell_of("not-a-record"))
        self.assertIsNone(_e1_cell_of({"gate": None}))


class E1FactorialInteractionTest(unittest.TestCase):
    CELLS = ("full", "calib_len_only", "calib_ent_only", "calib_no_terms")
    SEEDS = (0, 1)

    def _records(self, values, flags, variant):
        settings = Settings.quick()
        cfg = _model_cfg(settings, gate="calib",
                         calib_use_len=flags[0],
                         calib_use_ent=flags[1])
        return [
            {
                "seed": seed,
                "accuracy": value,
                "gate": "calib",
                "calib_variant": variant,
                "model_cfg": dict(cfg),
                "config": {},
                "lr": 1e-3,
                "eval_seed": 123,
                "stats_seed": 7,
            }
            for seed, value in zip(self.SEEDS, values)
        ]

    def _cell(self, values, flags, variant):
        return self._records(values, flags, variant)

    def test_interaction_is_the_difference_of_conditional_effects(self):
        cells = {
            "full": self._cell((0.8, 0.7), (True, True), "full"),
            "calib_len_only": self._cell(
                (0.6, 0.5), (True, False), "calib_len_only"),
            "calib_ent_only": self._cell(
                (0.5, 0.55), (False, True), "calib_ent_only"),
            "calib_no_terms": self._cell(
                (0.4, 0.45), (False, False), "calib_no_terms"),
        }
        ci = _paired_did(
            cells["full"], cells["calib_len_only"],
            cells["calib_ent_only"], cells["calib_no_terms"], "accuracy")
        self.assertEqual(ci["n"], 2)
        self.assertAlmostEqual(ci["mean"], 0.1)
        self.assertTrue(ci["controlled"]["controlled"])
        # The two marginal differences the interaction contrasts must be
        # the conditional concentration effects, not the length edge.
        self.assertAlmostEqual(ci["arms"]["a"]["mean"], 0.2)
        self.assertAlmostEqual(ci["arms"]["b"]["mean"], 0.1)
        self.assertEqual(ci["seeds_used"], list(self.SEEDS))
        self.assertEqual(ci["seeds_dropped"], [])

    def test_contrast_helpers_materialize_generators_once(self):
        cells = {
            "full": self._cell((0.8, 0.7), (True, True), "full"),
            "calib_len_only": self._cell(
                (0.6, 0.5), (True, False), "calib_len_only"),
            "calib_ent_only": self._cell(
                (0.5, 0.55), (False, True), "calib_ent_only"),
            "calib_no_terms": self._cell(
                (0.4, 0.45), (False, False), "calib_no_terms"),
        }
        did = _paired_did(
            (r for r in cells["full"]),
            (r for r in cells["calib_len_only"]),
            (r for r in cells["calib_ent_only"]),
            (r for r in cells["calib_no_terms"]), "accuracy")
        self.assertEqual(did["n"], 2)
        self.assertAlmostEqual(did["mean"], 0.1)
        diff = _paired_diff(
            (r for r in cells["full"]),
            (r for r in cells["calib_len_only"]), "accuracy")
        self.assertEqual(diff["seeds_used"], list(self.SEEDS))
        self.assertAlmostEqual(diff["mean"], 0.2)
        one = _one_sample_contrast(
            (r for r in cells["full"]), "accuracy")
        self.assertEqual(one["seeds_used"], list(self.SEEDS))
        self.assertAlmostEqual(one["mean"], 0.75)
        bench = _paired_benchmark(
            (r for r in cells["full"]),
            (r for r in cells["calib_len_only"]), "accuracy", 0.05, "full")
        self.assertEqual(bench["seeds_used"], list(self.SEEDS))

    def test_interaction_row_is_registered_as_two_sided(self):
        registry = HYPOTHESES["H1d"]
        for metric in ("accuracy", "gate_margin", "abs_gate_vs_entropy"):
            key = f"interaction_{metric}"
            self.assertIn(key, registry["direction"])
            self.assertIsNone(registry["direction"][key])
            self.assertIn(metric, registry["sesoi"])

    def test_miswired_cell_is_refused_not_averaged(self):
        # A label that claims the full cell but carries the ablated flags
        # must fail the controlled audit; the interaction row may not be
        # read as if it were a valid 2x2 contrast.
        full = self._cell((0.8, 0.7), (True, True), "full")
        mislabelled = self._cell(
            (0.6, 0.5), (True, False), "full")
        ci = _paired_did(
            full, mislabelled,
            self._cell((0.5, 0.55), (False, True), "calib_ent_only"),
            self._cell((0.4, 0.45), (False, False), "calib_no_terms"),
            "accuracy")
        self.assertFalse(ci["controlled"]["controlled"])
        self.assertTrue(ci["controlled"]["mismatches"])


class ReportMalformedRunTest(unittest.TestCase):
    """Malformed rows are uninformative; reports must not abort on them."""

    def test_competence_ignores_non_mapping_and_unreadable_rows(self):
        good = {"accuracy": 1.0, "n": 100}
        out = _competence_check(
            [("arm", [None, good, {"accuracy": "x", "n": 100}])],
            lambda r: (int(round(r["accuracy"] * r["n"])), int(r["n"])),
            0.2)
        self.assertTrue(out["passed"])
        self.assertEqual(out["arms"]["arm"]["correct"], 100)
        self.assertEqual(out["arms"]["arm"]["n"], 100)

    def test_competence_skips_unreadable_counts(self):
        good = {"accuracy": 1.0, "n": 100}
        bad_count = {"accuracy": 1.0, "n": "not-a-count"}
        bad_shape = {"accuracy": 1.0, "n": 100, "bad_shape": True}

        def count(run):
            if run.get("bad_shape"):
                return 1
            return (int(round(run["accuracy"] * run["n"])), int(run["n"]))

        out = _competence_check(
            [("arm", [bad_count, bad_shape, good])], count, 0.2)
        self.assertTrue(out["passed"])
        self.assertEqual(out["arms"]["arm"]["correct"], 100)
        self.assertEqual(out["arms"]["arm"]["n"], 100)

    def test_competence_abstains_when_runs_are_not_iterable(self):
        out = _competence_check(
            [("arm", None)], lambda r: (0, 1), 0.2)
        self.assertFalse(out["passed"])
        self.assertEqual(out["arms"]["arm"]["n"], 0)

    def test_error_tables_skip_non_mapping_rows(self):
        conf = {"err_conf_by_error_type": {"retrieval_miss": 0.9}}
        structural = {"err_taxonomy_hard_structural_naive_only": 0.2}
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            _print_conf_by_error_type([("arm", [None, conf])])
            _print_structural_decomposition([("arm", [None, structural])])
        self.assertIn("arm", buf.getvalue())
        self.assertIn("arm", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
