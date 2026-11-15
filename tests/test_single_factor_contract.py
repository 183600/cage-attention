"""The single-factor contract: one manipulated factor, certified and reported.

A controlled contrast may differ in exactly one factor, and the report must
say so in a machine-checkable way.  The tests below pin the three parts of
that contract:

* the audit separates the *independent factors* (``varying_factors``) from
  the *declared label differences* (``declared_label_diffs``), so a cell
  label such as ``calib_variant`` - which names a whole flag vector - can
  never be counted as a second factor;
* the decision layer refuses a controlled row that carries no single-factor
  certificate (fail closed), while the explicitly declared aggregate rows
  (factorial main effect, grid-level main effect, interaction) stay
  readable;
* the persisted row and the printed line report the verdict next to the
  factors, so no reader has to re-run the audit.
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import tempfile
import unittest

from cage import experiments
from cage.experiments import ContrastLog, _controlled_audit, paired_bootstrap_ci
# Ensure reproducibility
import random as _rnd
_rnd.seed(42)
try:
    import numpy as _np
    _np.random.seed(42)
except ImportError:
    pass



class SingleFactorAuditTest(unittest.TestCase):
    CFG = {
        "vocab_size": 64, "d_model": 32, "n_heads": 4, "n_layers": 2,
        "max_len": 64, "dropout": 0.0, "gate": "calib", "norm": "pre",
        "layer_scale": None, "calib_use_len": True, "calib_use_ent": True,
        "calib_shuffle_ent": False, "calib_ent_norm": True,
        "calib_pos_norm": False,
    }

    def _run(self, cell="full", **cfg_overrides):
        cfg = dict(self.CFG)
        cfg.update(cfg_overrides)
        return {
            "seed": 0, "config": {"d_model": 32}, "lr": 1e-3,
            "eval_seed": 2024, "stats_seed": 7, "steps_run": 10,
            "stopped_early": False, "restored_best": False,
            "data_fingerprints": None, "training_recipe": {},
            "params": 100, "gate": "calib", "norm": "pre",
            "layer_scale": None, "calib_variant": cell,
            "model_cfg": cfg,
        }

    def _legacy(self, cell):
        run = self._run(cell)
        run.pop("model_cfg")
        return run

    def test_cell_label_is_not_counted_as_an_independent_factor(self):
        # "full" vs "calib_ent_only" manipulates exactly one switch (the
        # length term), but the two arms carry different cell labels.  A
        # consumer that counted the label would see two factors; the
        # contract requires one.
        audit = _controlled_audit(
            [self._run("full")],
            [self._run("calib_ent_only", calib_use_len=False)])
        self.assertTrue(audit["controlled"], audit["mismatches"])
        self.assertTrue(audit["single_factor"])
        self.assertEqual(audit["n_varying_factors"], 1)
        self.assertEqual(set(audit["varying_factors"]), {"calib_use_len"})
        self.assertNotIn("calib_variant", audit["varying_factors"])
        # The label difference is still reported, just not as a factor.
        self.assertIn("calib_variant", audit["declared_label_diffs"])

    def test_legacy_cell_label_expands_into_the_factor_vector(self):
        # Legacy records carry only the cell label.  The label is expanded
        # into the flag vector it names, so the one-factor ablation still
        # certifies and the label itself stays out of the factor list.
        audit = _controlled_audit(
            [self._legacy("full")], [self._legacy("calib_ent_only")])
        self.assertTrue(audit["controlled"], audit["mismatches"])
        self.assertTrue(audit["single_factor"])
        self.assertEqual(audit["n_varying_factors"], 1)
        self.assertEqual(set(audit["varying_factors"]), {"calib_use_len"})
        self.assertIn("calib_variant", audit["declared_label_diffs"])

    def test_unresolvable_cell_leaves_the_factor_count_undetermined(self):
        # An unknown cell label cannot be expanded, so the number of
        # manipulated factors is unknown - not zero - and the contrast is
        # not certified as single-factor.
        audit = _controlled_audit(
            [self._legacy("full")], [self._legacy("mystery")])
        self.assertFalse(audit["controlled"])
        self.assertFalse(audit["single_factor"])
        self.assertIsNone(audit["n_varying_factors"])

    def test_two_term_legacy_ablation_still_counts_two_factors(self):
        audit = _controlled_audit(
            [self._legacy("full")], [self._legacy("calib_no_terms")])
        self.assertFalse(audit["controlled"])
        self.assertFalse(audit["single_factor"])
        self.assertEqual(audit["n_varying_factors"], 2)
        self.assertEqual(set(audit["varying_factors"]),
                         {"calib_use_len", "calib_use_ent"})


class SingleFactorDecisionTest(unittest.TestCase):
    @staticmethod
    def _ci(diffs, audit):
        ci = paired_bootstrap_ci(diffs)
        ci["diffs"] = list(diffs)
        ci["controlled"] = audit
        return ci

    def _audit(self, **overrides):
        audit = {
            "controlled": True,
            "mismatches": [],
            "varying_factors": {"gate": {"a": ["calib"], "b": ["query"]}},
            "single_factor": True,
            "n_varying_factors": 1,
        }
        audit.update(overrides)
        return audit

    def _finalize(self, ci):
        log = ContrastLog("e1", hypotheses=("H1a",), min_seeds=3)
        with contextlib.redirect_stdout(io.StringIO()):
            log.print("calib - query", "H1a", "abs_gate_vs_entropy", ci)
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = log.finalize(pathlib.Path(tmp), q=0.05)
        return payload

    def test_certified_single_factor_row_supports_its_hypothesis(self):
        payload = self._finalize(self._ci([0.2, 0.19, 0.21, 0.18, 0.2],
                                          self._audit()))
        row = payload["rows"][0]
        self.assertTrue(row["supports_hypothesis"])
        self.assertFalse(row["single_factor_violation"])
        self.assertEqual(row["single_factor"], True)
        self.assertEqual(row["n_varying_factors"], 1)

    def test_uncertified_controlled_row_cannot_support_a_hypothesis(self):
        # A row that claims to be controlled but carries no single-factor
        # certificate is refused at the decision point: without the
        # certificate it may not be read as a one-factor contrast.
        payload = self._finalize(
            self._ci([0.2, 0.19, 0.21, 0.18, 0.2],
                     self._audit(single_factor=None, n_varying_factors=None)))
        row = payload["rows"][0]
        self.assertTrue(row["single_factor_violation"])
        self.assertFalse(row["claim_eligible"])
        self.assertFalse(row["supports_hypothesis"])

    def test_declared_aggregate_row_is_not_refused(self):
        # A factorial main effect is not a one-factor pairwise contrast,
        # but it declares itself as an aggregate, so it stays readable.
        payload = self._finalize(
            self._ci([0.2, 0.19, 0.21, 0.18, 0.2],
                     self._audit(single_factor=None,
                                 n_varying_factors=None,
                                 aggregate="factorial main effect")))
        row = payload["rows"][0]
        self.assertFalse(row["single_factor_violation"])
        self.assertTrue(row["claim_eligible"])
        self.assertEqual(row["aggregate"], "factorial main effect")

    def test_uncertified_row_is_reported_in_the_summary(self):
        log = ContrastLog("e1", hypotheses=("H1a",), min_seeds=3)
        ci = self._ci([0.2, 0.19, 0.21, 0.18, 0.2],
                      self._audit(single_factor=None,
                                  n_varying_factors=None))
        with contextlib.redirect_stdout(io.StringIO()):
            log.print("calib - query", "H1a", "abs_gate_vs_entropy", ci)
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                payload = log.finalize(pathlib.Path(tmp), q=0.05)
        self.assertIn("single-factor certificate", buffer.getvalue())
        self.assertEqual(payload["n_single_factor_violations"], 1)
        self.assertEqual(payload["n_single_factor"], 0)
        self.assertEqual(payload["n_aggregate_rows"], 0)
        self.assertEqual(payload["single_factor_violation_contrasts"],
                         ["H1a/calib - query (abs_gate_vs_entropy)"])

    def test_persisted_row_keeps_the_verdict_and_the_label_diffs(self):
        log = ContrastLog("e1", hypotheses=("H1a",), min_seeds=3)
        ci = self._ci([0.2, 0.19, 0.21, 0.18, 0.2],
                      self._audit(declared_label_diffs={
                          "calib_variant": {"a": ["full"],
                                            "b": ["calib_ent_only"]}}))
        with contextlib.redirect_stdout(io.StringIO()):
            log.print("calib - query", "H1a", "abs_gate_vs_entropy", ci)
        row = log.rows[0]
        self.assertEqual(row["single_factor"], True)
        self.assertEqual(row["n_varying_factors"], 1)
        self.assertEqual(set(row["varying_factors"]), {"gate"})
        self.assertIn("calib_variant", row["declared_label_diffs"])

    def test_printed_line_states_the_single_factor_verdict(self):
        log = ContrastLog("e1", hypotheses=("H1a",), min_seeds=3)
        ci = self._ci([0.2, 0.19, 0.21, 0.18, 0.2], self._audit())
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            log.print("calib - query", "H1a", "abs_gate_vs_entropy", ci)
        printed = buffer.getvalue()
        self.assertIn("single-factor=yes", printed)
        self.assertIn("varies=gate", printed)


class AggregateRowDeclarationTest(unittest.TestCase):
    def test_factorial_main_effect_declares_itself_as_an_aggregate(self):
        cells = {}
        for name, flags in (
                ("full", {"calib_use_len": True, "calib_use_ent": True}),
                ("calib_len_only", {"calib_use_len": True,
                                    "calib_use_ent": False}),
                ("calib_ent_only", {"calib_use_len": False,
                                    "calib_use_ent": True}),
                ("calib_no_terms", {"calib_use_len": False,
                                    "calib_use_ent": False})):
            cells[name] = [{
                "seed": seed, "accuracy": 0.5 + 0.01 * seed
                + (0.02 if flags["calib_use_ent"] else 0.0),
                "config": {"d_model": 32}, "lr": 1e-3, "eval_seed": 2024,
                "stats_seed": 7, "steps_run": 10, "stopped_early": False,
                "restored_best": False, "data_fingerprints": None,
                "training_recipe": {}, "params": 100,
                "gate": "calib", "norm": "pre", "layer_scale": None,
                "calib_variant": name,
                "model_cfg": {
                    "vocab_size": 64, "d_model": 32, "n_heads": 4,
                    "n_layers": 2, "max_len": 64, "dropout": 0.0,
                    "gate": "calib", "norm": "pre", "layer_scale": None,
                    "calib_shuffle_ent": False, "calib_ent_norm": True,
                    "calib_pos_norm": False, **flags},
            } for seed in range(3)]
        ci = experiments._factorial_main_effect(cells, "accuracy",
                                                "concentration")
        audit = ci["controlled"]
        self.assertIsNone(audit["single_factor"])
        self.assertIsNone(audit["n_varying_factors"])
        self.assertEqual(audit["aggregate"], "factorial main effect")

    def test_interaction_row_declares_itself_as_an_aggregate(self):
        runs = {
            "calib": [{"seed": s, "accuracy": 0.5 + 0.01 * s,
                       "config": {"d_model": 32}, "lr": 1e-3,
                       "eval_seed": 2024, "stats_seed": 7, "steps_run": 10,
                       "stopped_early": False, "restored_best": False,
                       "data_fingerprints": None, "training_recipe": {},
                       "params": 100, "gate": "calib", "norm": "pre",
                       "layer_scale": None, "calib_variant": "full"}
                      for s in range(3)],
            "query": [{"seed": s, "accuracy": 0.4 + 0.01 * s,
                       "config": {"d_model": 32}, "lr": 1e-3,
                       "eval_seed": 2024, "stats_seed": 7, "steps_run": 10,
                       "stopped_early": False, "restored_best": False,
                       "data_fingerprints": None, "training_recipe": {},
                       "params": 100, "gate": "query", "norm": "pre",
                       "layer_scale": None} for s in range(3)],
        }
        ci = experiments._paired_did(runs["calib"], runs["query"],
                                     runs["calib"], runs["query"], "accuracy")
        audit = ci["controlled"]
        self.assertIsNone(audit["single_factor"])
        self.assertEqual(audit["aggregate"], "interaction")


if __name__ == "__main__":
    unittest.main()
