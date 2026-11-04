"""The design-time single-factor contract: the grid may only start when every
declared controlled contrast varies exactly one resolved model factor.

The row-level audit (``tests/test_single_factor_contract.py``) can only see
a multi-factor manipulation *after* the grid has trained, and then it
refuses the row.  These tests pin the earlier, design-time half of the
same contract:

* each experiment declares its controlled contrasts *and* the resolved cell
  configurations they are read from, and the declaration is certified to
  vary exactly one model factor per contrast;
* a design whose declared contrast varies more than one factor stops the
  run before the first training step, so no compute is spent on a contrast
  that could not support its hypothesis;
* the two-factor factorial corner (``calib_no_terms``) can never be turned
  into a decision row: it is part of the grid, but it is read through the
  registered aggregates only;
* the decision rows are routed through the design, so an executed contrast
  and the certified design cannot drift apart;
* the contrasts that are deliberately *not* controlled (the placebo against
  the query gate) are recorded as uncontrolled instead of being silently
  mixed into the controlled family.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import pathlib
import tempfile
import unittest

from cage import experiments


def _runner_source(name: str) -> str:
    path = pathlib.Path(experiments.__file__).parent / name
    return path.read_text(encoding="utf-8")


def _contrast_calls(tree: ast.AST, names) -> list:
    """Every call of one of ``names`` in the parsed source."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in names:
            found.append(node)
    return found


class SingleFactorDesignCertificateTest(unittest.TestCase):
    """The declared designs of E1, E2 and E3 are one-factor designs."""

    def _audit(self, design, tag):
        return experiments._single_factor_design_audit(
            tag, design["cells"], design["pairs"],
            design.get("exploratory_pairs") or ())

    def test_e1_declared_contrasts_vary_exactly_one_factor(self):
        design = experiments._e1_single_factor_design()
        audit = self._audit(design, "e1")
        self.assertTrue(audit["single_factor"], audit["violations"])
        factors = {(row["arm_a"], row["arm_b"]):
                   sorted(row["varying_factors"])
                   for row in audit["pairs"]}
        self.assertEqual(factors[("calib", "none")], ["gate"])
        self.assertEqual(factors[("calib", "query")], ["gate"])
        self.assertEqual(factors[("query", "none")], ["gate"])
        self.assertEqual(factors[("null", "none")], ["gate"])
        self.assertEqual(factors[("calib", "calib_len_only")],
                         ["calib_use_ent"])
        self.assertEqual(factors[("calib", "calib_ent_only")],
                         ["calib_use_len"])
        self.assertEqual(factors[("calib", "calib_shuf_ent")],
                         ["calib_shuffle_ent"])

    def test_e2_declared_contrasts_vary_exactly_one_factor(self):
        design = experiments._e2_single_factor_design()
        audit = self._audit(design, "e2")
        self.assertTrue(audit["single_factor"], audit["violations"])
        factors = {(row["arm_a"], row["arm_b"]):
                   sorted(row["varying_factors"])
                   for row in audit["pairs"]}
        self.assertEqual(factors[("calib", "none")], ["gate"])
        self.assertEqual(factors[("calib", "query")], ["gate"])
        self.assertEqual(factors[("calib", "calib_raw_ent")],
                         ["calib_ent_norm"])
        self.assertEqual(factors[("calib", "calib_ent_only")],
                         ["calib_use_len"])
        self.assertEqual(factors[("calib", "calib_len_only")],
                         ["calib_use_ent"])

    def test_e3_declared_contrasts_vary_exactly_one_factor(self):
        design = experiments._e3_single_factor_design()
        audit = self._audit(design, "e3")
        self.assertTrue(audit["single_factor"], audit["violations"])
        self.assertEqual(audit["n_pairs"], 20)
        for row in audit["pairs"]:
            self.assertEqual(row["n_varying_factors"], 1, row)
            self.assertIn(sorted(row["varying_factors"])[0],
                          {"norm", "layer_scale", "gate"})
        self.assertEqual(audit["cells_without_one_factor_contrast"], [])

    def test_every_design_covers_its_whole_grid(self):
        # A cell that no declared contrast uses is either a factorial corner
        # (read through the registered aggregates) or a benchmark-only arm;
        # the certificate names it instead of leaving it invisible.
        for tag, builder, expected in (
                ("e1", experiments._e1_single_factor_design,
                 ["calib_no_terms"]),
                ("e2", experiments._e2_single_factor_design,
                 ["calib_no_terms", "null"]),
                ("e3", experiments._e3_single_factor_design, [])):
            with self.subTest(experiment=tag):
                audit = self._audit(builder(), tag)
                self.assertEqual(
                    audit["cells_without_one_factor_contrast"], expected)

    def test_the_two_factor_corner_is_not_a_declared_contrast(self):
        # ``calib_no_terms`` switches both calibration terms off, so it is a
        # two-factor manipulation: it may not enter the controlled family.
        design = experiments._e1_single_factor_design()
        self.assertIn("calib_no_terms", design["cells"])
        self.assertNotIn(("calib", "calib_no_terms"), design["pairs"])
        broken = {"cells": design["cells"],
                  "pairs": [("calib", "calib_no_terms")]}
        with self.assertRaises(ValueError) as ctx:
            experiments._enforce_single_factor_design("e9", broken)
        self.assertIn("calib_no_terms", str(ctx.exception))
        self.assertIn("2 model factor", str(ctx.exception))

    def test_the_placebo_contrast_is_recorded_as_uncontrolled(self):
        # The placebo arm changes the gate *and* the shuffle switch, so it is
        # not a one-factor contrast; it is printed outside the decision log
        # and the certificate says so instead of hiding it.
        design = experiments._e1_single_factor_design()
        audit = self._audit(design, "e1")
        self.assertEqual(len(audit["exploratory_pairs"]), 1)
        row = audit["exploratory_pairs"][0]
        self.assertEqual((row["arm_a"], row["arm_b"]),
                         ("calib_shuf_ent", "query"))
        self.assertFalse(row["controlled"])
        self.assertEqual(row["n_varying_factors"], 2)
        self.assertEqual(sorted(row["varying_factors"]),
                         ["calib_shuffle_ent", "gate"])


class SingleFactorDesignEnforcementTest(unittest.TestCase):
    """A design that is not one-factor stops the grid before training."""

    def setUp(self):
        self._designs = dict(experiments._SINGLE_FACTOR_DESIGNS)

    def tearDown(self):
        experiments._SINGLE_FACTOR_DESIGNS.clear()
        experiments._SINGLE_FACTOR_DESIGNS.update(self._designs)

    def _two_factor_design(self):
        return {
            "cells": {
                "full": experiments._cell_factor_vector("calib"),
                "no_terms": experiments._cell_factor_vector(
                    "calib", calib_use_len=False, calib_use_ent=False),
            },
            "pairs": [("full", "no_terms")],
        }

    def test_a_multi_factor_contrast_refuses_to_start_the_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            with self.assertRaises(ValueError) as ctx:
                experiments._enforce_single_factor_design(
                    "e9", self._two_factor_design(), out)
            self.assertIn("single-factor design contract violated",
                          str(ctx.exception))
            self.assertFalse(
                (out / "e9_single_factor_design.json").exists())
        self.assertNotIn("e9", experiments._SINGLE_FACTOR_DESIGNS)

    def test_an_unknown_arm_refuses_to_start_the_grid(self):
        design = {"cells": {"full": experiments._cell_factor_vector("calib")},
                  "pairs": [("full", "missing")]}
        with self.assertRaises(ValueError):
            experiments._enforce_single_factor_design("e9", design)

    def test_a_malformed_design_is_refused(self):
        for design in ({}, {"cells": [], "pairs": [("a", "b")]},
                       {"cells": {"a": {}}, "pairs": [("a",)]}):
            with self.subTest(design=design):
                with self.assertRaises(ValueError):
                    experiments._enforce_single_factor_design("e9", design)

    def test_the_certificate_is_written_before_the_grid_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                audit = experiments._enforce_single_factor_design(
                    "e1", experiments._e1_single_factor_design(), out)
            payload = json.loads(
                (out / "e1_single_factor_design.json").read_text(
                    encoding="utf-8"))
        self.assertTrue(payload["single_factor"])
        self.assertEqual(payload["n_pairs"], audit["n_pairs"])
        self.assertEqual(payload["experiment"], "e1")
        for row in payload["pairs"]:
            self.assertTrue(row["single_factor"], row)
            self.assertEqual(row["n_varying_factors"], 1)


class SingleFactorDesignGuardTest(unittest.TestCase):
    """The executed contrasts and the certified design cannot drift apart."""

    def setUp(self):
        self._designs = dict(experiments._SINGLE_FACTOR_DESIGNS)

    def tearDown(self):
        experiments._SINGLE_FACTOR_DESIGNS.clear()
        experiments._SINGLE_FACTOR_DESIGNS.update(self._designs)

    def _certify(self, tag):
        builder = {"e1": experiments._e1_single_factor_design,
                   "e2": experiments._e2_single_factor_design,
                   "e3": experiments._e3_single_factor_design}[tag]
        with contextlib.redirect_stdout(io.StringIO()):
            return experiments._enforce_single_factor_design(tag, builder())

    def test_a_declared_pair_is_accepted(self):
        self._certify("e1")
        runs_a, runs_b = [1], [2]
        self.assertEqual(
            experiments._design_pair("e1", "calib", "none", runs_a, runs_b),
            (runs_a, runs_b))

    def test_an_undeclared_pair_is_refused(self):
        self._certify("e1")
        with self.assertRaises(ValueError) as ctx:
            experiments._design_pair("e1", "calib", "calib_no_terms", [], [])
        self.assertIn("not a declared one-factor contrast", str(ctx.exception))

    def test_a_contrast_without_a_certificate_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            experiments._design_pair("e9", "calib", "none", [], [])
        self.assertIn("no single-factor design certificate",
                      str(ctx.exception))


class SingleFactorDesignWiringTest(unittest.TestCase):
    """The runners certify their design and read their rows from it."""

    RUNNERS = (("_e1_runner.py", "run_e1", {"_e1_run", "_train_and_eval"}),
               ("_e2_runner.py", "run_e2", {"_train_and_eval"}),
               ("_e3_cli.py", "run_e3", {"_train_and_eval"}))

    def _function(self, name: str, function: str):
        tree = ast.parse(_runner_source(name))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function:
                return node
        self.fail(f"{name} has no {function} function")

    def test_the_runners_certify_their_design_before_the_first_training_run(self):
        for name, function, training in self.RUNNERS:
            with self.subTest(runner=name):
                body = self._function(name, function)
                enforce = _contrast_calls(
                    body, {"_enforce_single_factor_design"})
                train = _contrast_calls(body, training)
                self.assertTrue(enforce, name)
                self.assertTrue(train, name)
                self.assertLess(min(call.lineno for call in enforce),
                                min(call.lineno for call in train))

    def test_every_decision_row_reads_a_certified_design_pair(self):
        # A decision row is a pre-registered contrast that may support a
        # hypothesis, so it has to be one of the declared one-factor pairs.
        # The aggregate rows (factorial main effect, grid-level main effect,
        # interaction) are not pairwise contrasts and declare themselves.
        for name, _function, _training in self.RUNNERS:
            with self.subTest(runner=name):
                tree = ast.parse(_runner_source(name))
                rows = [
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("print", "print_equivalence_ci")
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "clog"]
                self.assertTrue(rows, name)
                checked = 0
                for row in rows:
                    for call in _contrast_calls(
                            row, {"_paired_diff", "_paired_equivalence"}):
                        checked += 1
                        self.assertTrue(
                            isinstance(call.args[0], ast.Starred),
                            f"{name}: a decision row compares arms without "
                            f"reading them from the certified design")
                self.assertTrue(checked, name)


if __name__ == "__main__":
    unittest.main()
