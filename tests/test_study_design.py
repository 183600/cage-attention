"""Study-level design preparation: the analysis plan, its pre-flight and cost.

The per-experiment machinery certifies one family at a time.  The study
reads across three families, and three questions exist only at that level:

* is the pre-registered confirmatory set consistent with the hypothesis
  registry, and is every leg *reachable* - does its family actually train
  the cells the leg contrasts?
* does the study's own multiplicity level leave the headline claims
  resolvable at the planned seed count?
* what will the study cost, so the compute request is not a hand-written
  number?

The tests pin the answers, because each of them is a property that a later
refactor can silently break: an unregistered leg, an arm that no longer
exists in a grid, a plan that can be revised after the data were seen, and
a study verdict assembled from a partial set of families.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from cage import experiments
from cage.experiments import (
    DesignPlanError,
    HYPOTHESES,
    Settings,
    _analysis_plan_audit,
    _analysis_plan_fingerprint,
    _analysis_plan_legs,
    _analysis_plan_registry_fingerprint,
    _family_cell_labels,
    _primary_contrast_reachability,
    _primary_contrast_single_factor,
    _publish_analysis_plan,
    _study_cost_plan,
    _study_design_preflight,
    _study_level_adjudication,
)
from cage.efficiency import analytic_matmul_flops
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


def _leg(**overrides):
    """A minimal registered leg, with the fields the audit reads."""
    leg = {
        "id": "PX",
        "experiment": "e1",
        "hypothesis": "H1a",
        "metric": "abs_gate_vs_entropy",
        "rule": "calib - query",
        "arms": ("calib", "query"),
        "kind": "one_factor",
        "claim": "C1",
        "text": "test leg",
    }
    leg.update(overrides)
    return leg


def _contrast_row(hypothesis, metric, rule, mean=0.2, p_value=0.001,
                  supports=True, fdr_reject=True, practically_relevant=True,
                  claim_eligible=True, controlled=True, direction="+",
                  direction_reversed=False, n=5):
    """One finalized contrast row, shaped like ``<tag>_contrasts.json``."""
    return {
        "hypothesis": hypothesis,
        "metric": metric,
        "rule": rule,
        "n": n,
        "mean": mean,
        "lo": mean - 0.05,
        "hi": mean + 0.05,
        "p_value": p_value,
        "p_adj": p_value,
        "direction": direction,
        "sesoi": 0.01,
        "supports_hypothesis": bool(supports),
        "fdr_reject": bool(fdr_reject),
        "practically_relevant": bool(practically_relevant),
        "claim_eligible": bool(claim_eligible),
        "controlled": controlled,
        "confounded": False,
        "direction_reversed": bool(direction_reversed),
    }


def _write_contrasts(root, tag, rows):
    path = pathlib.Path(root) / f"{tag}_contrasts.json"
    path.write_text(json.dumps({"tag": tag, "q": 0.05, "rows": rows}),
                    encoding="utf-8")
    return path


class AnalysisPlanRegistryTest(unittest.TestCase):
    """The registered confirmatory set must agree with the registry."""

    def test_registry_is_consistent_with_the_hypothesis_registry(self):
        audit = _analysis_plan_audit()
        self.assertTrue(audit["ok"], audit["issues"])
        self.assertEqual(audit["issues"], [])

    def test_every_leg_names_a_hypothesis_of_its_own_experiment(self):
        for leg in _analysis_plan_legs():
            entry = HYPOTHESES[leg["hypothesis"]]
            self.assertEqual(entry["experiment"], leg["experiment"])
            self.assertIn(leg["metric"], entry["sesoi"])

    def test_every_leg_names_arms_that_its_family_trains(self):
        report = _primary_contrast_reachability()
        self.assertTrue(report["ok"], report["unreachable_legs"])
        self.assertEqual(report["unreachable_legs"], [])

    def test_family_cell_labels_come_from_the_cell_plans(self):
        self.assertEqual(len(_family_cell_labels("e1")), 9)
        self.assertEqual(len(_family_cell_labels("e2")), 8)
        self.assertEqual(len(_family_cell_labels("e3")), 16)
        self.assertEqual(_family_cell_labels("nope"), ())

    def test_an_arm_outside_the_grid_is_reported_unreachable(self):
        legs = [_leg(arms=("calib", "not_a_cell"))]
        report = _primary_contrast_reachability(legs)
        self.assertFalse(report["ok"])
        self.assertEqual(report["unreachable_legs"], ["PX"])
        self.assertEqual(report["legs"][0]["missing_arms"], ["not_a_cell"])

    def test_audit_rejects_a_wrong_number_of_arms(self):
        legs = [_leg(arms=("calib", "query", "none"))]
        issues = _analysis_plan_audit(legs=legs)["issues"]
        self.assertTrue(any("contrasts 2 cells" in issue for issue in issues),
                        issues)

    def test_audit_rejects_a_missing_kind(self):
        legs = [_leg(kind="diff_of_diffs")]
        issues = _analysis_plan_audit(legs=legs)["issues"]
        self.assertTrue(any("contrast kind" in issue for issue in issues),
                        issues)

    def test_audit_rejects_a_repeated_arm(self):
        legs = [_leg(arms=("calib", "calib"))]
        issues = _analysis_plan_audit(legs=legs)["issues"]
        self.assertTrue(any("listed twice" in issue for issue in issues),
                        issues)

    def test_audit_rejects_an_unknown_hypothesis(self):
        legs = [_leg(hypothesis="H9z")]
        issues = _analysis_plan_audit(legs=legs)["issues"]
        self.assertTrue(any("not in the registered" in issue
                            for issue in issues), issues)

    def test_registry_hash_covers_the_arms(self):
        payload = {
            "schema_version": 1,
            "experiments": ["e1"],
            "primary_contrasts": [_leg()],
            "claims": [{"id": "C1", "legs": ("PX",)}],
        }
        before = _analysis_plan_registry_fingerprint(payload)
        changed = json.loads(json.dumps(payload))
        changed["primary_contrasts"][0]["arms"] = ["calib", "none"]
        after = _analysis_plan_registry_fingerprint(changed)
        self.assertIsNotNone(before)
        self.assertNotEqual(before, after)

    def test_fingerprint_is_none_for_a_malformed_payload(self):
        self.assertIsNone(_analysis_plan_fingerprint({"schema_version": 1}))
        self.assertIsNone(_analysis_plan_registry_fingerprint(None))


class PrimaryContrastSingleFactorTest(unittest.TestCase):
    """The study-level confirmatory set is certified against the designs.

    The per-family certificate only guards the pairs a family declares.
    The registered primary contrasts are a second registration, so a leg
    that drifts - a two-factor corner promoted to a one-factor claim, an
    interaction that differences two different manipulations - must be
    refused before the grid trains, not read afterwards as the claim it is
    not.
    """

    def test_every_registered_primary_contrast_certifies_as_designed(self):
        report = _primary_contrast_single_factor()
        self.assertTrue(report["ok"], report["violating_legs"])
        self.assertEqual(report["violating_legs"], [])
        self.assertEqual(report["n_legs"], 10)
        for row in report["legs"]:
            self.assertTrue(row["certified"], row)
            self.assertEqual(row["issues"], [])
            if row["kind"] == "one_factor":
                self.assertTrue(row["single_factor"], row)
                self.assertEqual(len(row["varying_factors"]["marginal"]), 1)

    def test_interaction_leg_is_a_factorial_difference_of_two_edges(self):
        report = _primary_contrast_single_factor()
        interactions = [row for row in report["legs"]
                        if row["kind"] == "interaction"]
        self.assertEqual(len(interactions), 2)
        for row in interactions:
            self.assertEqual(row["varying_factors"]["marginal a"],
                             ["gate"])
            self.assertEqual(row["varying_factors"]["marginal b"],
                             ["gate"])
            self.assertEqual(row["varying_factors"]["context a"], ["norm"])
            self.assertEqual(row["varying_factors"]["context b"], ["norm"])

    def test_a_two_factor_one_factor_leg_is_refused(self):
        legs = [_leg(arms=("calib", "calib_no_terms"))]
        report = _primary_contrast_single_factor(legs)
        self.assertFalse(report["ok"])
        self.assertEqual(report["violating_legs"], ["PX"])
        issues = " ".join(report["legs"][0]["issues"])
        self.assertIn("2 model factor", issues)
        self.assertIn("not a declared controlled contrast", issues)

    def test_an_undeclared_single_factor_leg_is_refused(self):
        # A reversed pair varies exactly one factor but is not the declared
        # oriented contrast of the family, so its registered sign cannot be
        # trusted.
        legs = [_leg(arms=("none", "calib"))]
        report = _primary_contrast_single_factor(legs)
        self.assertFalse(report["ok"])
        self.assertIn("not a declared controlled contrast",
                      " ".join(report["legs"][0]["issues"]))

    def test_an_interaction_with_different_marginals_is_refused(self):
        legs = [_leg(
            experiment="e3", hypothesis="H3c", metric="accuracy",
            rule="bad interaction", kind="interaction",
            arms=("calib|seednorm|off", "query|pre|off",
                  "calib|pre|off", "query|seednorm|off"))]
        report = _primary_contrast_single_factor(legs)
        self.assertFalse(report["ok"])
        issues = " ".join(report["legs"][0]["issues"])
        self.assertIn("2 model factor", issues)

    def test_an_interaction_with_an_unresolvable_axis_is_refused(self):
        # The two "context" edges collapse onto the same cell, so the
        # difference-of-differences is not a factorial contrast and the
        # repeated arm is not a declared controlled pair.
        legs = [_leg(
            experiment="e3", hypothesis="H3c", metric="accuracy",
            rule="bad interaction", kind="interaction",
            arms=("calib|seednorm|off", "query|seednorm|off",
                  "calib|seednorm|off", "query|seednorm|off"))]
        report = _primary_contrast_single_factor(legs)
        self.assertFalse(report["ok"])
        issues = " ".join(report["legs"][0]["issues"])
        self.assertIn("not a declared controlled contrast", issues)

    def test_unknown_family_leaves_the_leg_uncertified(self):
        legs = [_leg(experiment="e9", arms=("a", "b"))]
        report = _primary_contrast_single_factor(legs)
        self.assertFalse(report["ok"])
        self.assertFalse(report["legs"][0]["certified"])


class StudyDesignPreflightTest(unittest.TestCase):
    """The pre-flight certifies the design and prices it."""

    def test_preflight_writes_the_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _study_design_preflight(Settings.quick(),
                                                  pathlib.Path(tmp))
            path = pathlib.Path(tmp) / "study_design_preflight.json"
            self.assertTrue(path.exists())
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(stored["registry_consistent"])
            self.assertTrue(stored["reachable"])
            self.assertEqual(len(stored["primary_contrasts"]), 10)
            self.assertIn("compute", stored)
            self.assertEqual(payload["unreachable_legs"], [])

    def test_preflight_refuses_an_unreachable_leg(self):
        broken = [_leg(arms=("calib", "not_a_cell"))]
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(experiments, "_PRIMARY_CONTRASTS", broken):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(DesignPlanError) as caught:
                    _study_design_preflight(Settings.quick(),
                                            pathlib.Path(tmp))
        self.assertIn("not_a_cell", str(caught.exception))
        self.assertFalse((pathlib.Path(tmp) /
                          "study_design_preflight.json").exists())

    def test_preflight_refuses_a_registry_inconsistency(self):
        broken = [_leg(hypothesis="H9z")]
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(experiments, "_PRIMARY_CONTRASTS", broken):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(DesignPlanError):
                    _study_design_preflight(Settings.quick(),
                                            pathlib.Path(tmp))

    def test_preflight_refuses_a_multi_factor_primary_contrast(self):
        broken = [_leg(arms=("calib", "calib_no_terms"))]
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(experiments, "_PRIMARY_CONTRASTS", broken):
            out = pathlib.Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(DesignPlanError) as caught:
                    _study_design_preflight(Settings.quick(), out)
            self.assertIn("2 model factor", str(caught.exception))
            self.assertFalse((out / "study_design_preflight.json").exists())

    def test_preflight_records_the_single_factor_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _study_design_preflight(Settings.quick(),
                                                  pathlib.Path(tmp))
        self.assertTrue(payload["single_factor_certified"])
        self.assertEqual(len(payload["single_factor_legs"]), 10)
        for row in payload["single_factor_legs"]:
            self.assertTrue(row["certified"], row)

    def test_cost_plan_counts_the_grid_cells_of_every_family(self):
        settings = Settings.quick()
        plan = _study_cost_plan(settings)
        n = len(settings.seeds)
        self.assertEqual(plan["families"]["e1"]["cells"], 9)
        self.assertEqual(plan["families"]["e2"]["cells"], 8)
        self.assertEqual(plan["families"]["e3"]["cells"], 16)
        grid = {
            tag: next(item["runs"] for item in fam["exact_runs"]
                      if item["name"] == "primary grid")
            for tag, fam in plan["families"].items()}
        self.assertEqual(grid["e1"], 9 * n)
        self.assertEqual(grid["e2"], 8 * n)
        self.assertEqual(grid["e3"], 16 * n)

    def test_cost_plan_scales_with_the_seed_count(self):
        one = _study_cost_plan(Settings(seeds=(0,)))
        two = _study_cost_plan(Settings(seeds=(0, 1)))
        self.assertEqual(two["total_exact_runs"], 2 * one["total_exact_runs"])
        self.assertAlmostEqual(two["total_training_tflops"],
                               2 * one["total_training_tflops"], places=6)

    def test_cost_plan_brackets_the_recipe_selection_runs(self):
        plan = _study_cost_plan(Settings.quick())
        self.assertLessEqual(plan["total_runs_min"], plan["total_exact_runs"])
        self.assertLessEqual(plan["total_exact_runs"], plan["total_runs_max"])
        for tag, family in plan["families"].items():
            self.assertLessEqual(family["runs_min"], family["exact_runs_total"])
            self.assertGreaterEqual(family["runs_max"],
                                    family["exact_runs_total"])

    def test_cost_plan_flops_match_the_analytic_model(self):
        settings = Settings.quick()
        plan = _study_cost_plan(settings)
        family = plan["families"]["e1"]
        entry = next(cell for cell in family["cells_detail"]
                     if cell["gate"] == "calib")
        expected = analytic_matmul_flops(
            experiments.Cfg(vocab_size=settings.vocab,
                            d_model=settings.d_model,
                            n_heads=settings.n_heads,
                            n_layers=settings.n_layers,
                            max_len=settings.max_len,
                            gate="calib", norm="pre"),
            plan["train_seq_len"], batch=1, head_tokens=1)
        self.assertEqual(entry["forward_flops_per_sequence"], expected)
        self.assertEqual(
            entry["training_flops_per_run"],
            expected * settings.steps * settings.batch * 3)

    def test_cost_plan_counts_one_run_per_cell_and_seed(self):
        settings = Settings.quick()
        plan = _study_cost_plan(settings)
        for family in plan["families"].values():
            for cell in family["cells_detail"]:
                self.assertEqual(cell["runs"],
                                 cell["cells"] * plan["n_seeds"])


class AnalysisPlanFreezeTest(unittest.TestCase):
    """The confirmatory set is frozen once outcomes exist."""

    def _publish(self, out_dir, settings=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return _publish_analysis_plan(settings or Settings.quick(),
                                          out_dir)

    def test_publishing_records_a_created_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._publish(pathlib.Path(tmp))
            history = pathlib.Path(tmp) / "analysis_plan_history.jsonl"
            self.assertTrue(history.exists())
            events = [json.loads(line) for line in
                      history.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([e["event"] for e in events], ["created"])

    def test_republishing_an_identical_plan_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self._publish(pathlib.Path(tmp))
            second = self._publish(pathlib.Path(tmp))
            self.assertEqual(first["analysis_plan_sha256"],
                             second["analysis_plan_sha256"])

    def test_an_amendment_before_outcomes_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._publish(pathlib.Path(tmp))
            amended = Settings.quick()
            amended.min_seeds_for_claim = 4
            payload = self._publish(pathlib.Path(tmp), amended)
            self.assertIsNotNone(payload.get("supersedes_analysis_plan_sha256"))
            history = pathlib.Path(tmp) / "analysis_plan_history.jsonl"
            events = [json.loads(line) for line in
                      history.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([e["event"] for e in events],
                             ["created", "amended"])

    def test_a_revision_after_outcomes_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            self._publish(out)
            (out / "e1_runs.jsonl").write_text("{}\n", encoding="utf-8")
            changed = Settings.quick()
            changed.min_seeds_for_claim = 4
            with self.assertRaises(DesignPlanError) as caught:
                self._publish(out, changed)
            self.assertIn("post-hoc redesign", str(caught.exception))

    def test_a_first_plan_after_outcomes_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            (out / "e2_runs.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(DesignPlanError) as caught:
                self._publish(out)
            self.assertIn("post-hoc", str(caught.exception))


class StudyLevelAdjudicationTest(unittest.TestCase):
    """The study verdict is assembled over the registered primary set."""

    def _legs(self):
        return {leg["id"]: leg for leg in _analysis_plan_legs()}

    def _rows(self, **overrides):
        rows = []
        for leg_id, leg in self._legs().items():
            kwargs = dict(overrides.get(leg_id, {}))
            rows.append(_contrast_row(leg["hypothesis"], leg["metric"],
                                      leg["rule"], **kwargs))
        return rows

    def _adjudicate(self, root, **overrides):
        rows = self._rows(**overrides)
        by_tag = {}
        for row in rows:
            leg = next(leg for leg in self._legs().values()
                       if leg["hypothesis"] == row["hypothesis"]
                       and leg["metric"] == row["metric"]
                       and leg["rule"] == row["rule"])
            by_tag.setdefault(leg["experiment"], []).append(row)
        for tag, tag_rows in by_tag.items():
            _write_contrasts(root, tag, tag_rows)
        with contextlib.redirect_stdout(io.StringIO()):
            return _study_level_adjudication(root, q=0.05)

    def test_a_complete_study_is_adjudicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._adjudicate(tmp)
            self.assertTrue(payload["study_complete"])
            self.assertEqual(payload["missing_experiments"], [])
            self.assertEqual(payload["study_status"], "supported")
            self.assertEqual(payload["n_claims_supported"], 4)
            self.assertEqual(payload["study_status_pooled_family"],
                             "supported")

    def test_a_missing_family_is_reported_as_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._rows()
            e1 = [row for row in rows
                  if self._legs()["P1"]["hypothesis"] == row["hypothesis"]
                  and self._legs()["P1"]["metric"] == row["metric"]
                  and self._legs()["P1"]["rule"] == row["rule"]]
            _write_contrasts(tmp, "e1", e1)
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _study_level_adjudication(tmp, q=0.05)
            self.assertFalse(payload["study_complete"])
            self.assertEqual(payload["study_status"], "incomplete")
            self.assertIn("e2", payload["missing_experiments"])
            self.assertFalse(payload["adherence_ok"])

    def test_a_claim_needs_every_leg(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._adjudicate(tmp, P2={"supports": False,
                                                "fdr_reject": False})
            claim = next(c for c in payload["claims"] if c["id"] == "C1")
            self.assertEqual(claim["status"], "partial")
            self.assertEqual(claim["n_legs_supporting_in_family"], 3)

    def test_an_ineligible_reversed_leg_does_not_refute_a_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._adjudicate(
                tmp, P2={"supports": False, "fdr_reject": True,
                        "direction_reversed": True, "claim_eligible": False})
            claim = next(c for c in payload["claims"] if c["id"] == "C1")
            self.assertNotEqual(claim["status"], "refuted")
            self.assertEqual(claim["direction_reversed_legs"], [])
            self.assertEqual(claim["direction_reversed_but_ineligible_legs"],
                             ["P2"])

    def test_an_eligible_reversed_leg_refutes_a_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._adjudicate(
                tmp, P1={"supports": False, "fdr_reject": False},
                P2={"supports": False, "fdr_reject": True,
                    "direction_reversed": True, "claim_eligible": True},
                P3={"supports": False, "fdr_reject": False},
                P10={"supports": False, "fdr_reject": False})
            claim = next(c for c in payload["claims"] if c["id"] == "C1")
            self.assertEqual(claim["status"], "refuted")
            self.assertEqual(claim["direction_reversed_legs"], ["P2"])

    def test_a_partially_supported_claim_is_partial_not_refuted(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._adjudicate(
                tmp, P2={"supports": False, "fdr_reject": True,
                        "direction_reversed": True, "claim_eligible": True})
            claim = next(c for c in payload["claims"] if c["id"] == "C1")
            self.assertEqual(claim["status"], "partial")
            self.assertEqual(claim["direction_reversed_legs"], ["P2"])

    def test_a_leg_missing_from_the_family_is_a_deviation(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._rows()
            kept = [row for row in rows
                    if not (row["hypothesis"] == "H1f"
                            and row["metric"] == "accuracy")]
            _write_contrasts(tmp, "e1", kept)
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _study_level_adjudication(tmp, q=0.05)
            leg = next(item for item in payload["legs"] if item["id"] == "P5")
            self.assertEqual(leg["status"], "missing")
            self.assertFalse(leg["measured"])
            self.assertFalse(payload["adherence_ok"])

    def test_an_ambiguous_leg_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._rows()
            duplicate = [row for row in rows
                         if row["hypothesis"] == "H1a"
                         and row["metric"] == "abs_gate_vs_entropy"]
            _write_contrasts(tmp, "e1", rows + duplicate)
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _study_level_adjudication(tmp, q=0.05)
            leg = next(item for item in payload["legs"] if item["id"] == "P1")
            self.assertEqual(leg["status"], "ambiguous")
            self.assertEqual(leg["n_matching_rows"], 2)

    def test_a_leg_supported_in_family_but_not_pooled_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Ten legs, all significant on their own; the pooled family
            # sees ten p-values, so only the strongest survives q/m.
            rows = []
            for index, leg in enumerate(self._legs().values()):
                rows.append(_contrast_row(
                    leg["hypothesis"], leg["metric"], leg["rule"],
                    p_value=0.01 * (index + 1)))
            by_tag = {}
            for row in rows:
                leg = next(leg for leg in self._legs().values()
                           if leg["hypothesis"] == row["hypothesis"]
                           and leg["metric"] == row["metric"]
                           and leg["rule"] == row["rule"])
                by_tag.setdefault(leg["experiment"], []).append(row)
            for tag, tag_rows in by_tag.items():
                _write_contrasts(tmp, tag, tag_rows)
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _study_level_adjudication(tmp, q=0.05)
            self.assertTrue(payload["n_family_inflated_legs"] > 0)
            self.assertTrue(payload["family_inflated_claims"])
            self.assertNotEqual(payload["study_status_pooled_family"],
                                "supported")

    def test_the_adjudication_is_written_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._adjudicate(tmp)
            path = pathlib.Path(tmp) / "study_level_adjudication.json"
            self.assertTrue(path.exists())
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored["q"], 0.05)
            self.assertEqual(len(stored["legs"]), 10)
            self.assertEqual(len(stored["claims"]), 4)


class StudyRunnerWiringTest(unittest.TestCase):
    """Every family registers the plan and refreshes the study verdict.

    The bug this guards against is a wiring one: a study-level plan that
    only E1 publishes means an E2-first run cannot register its own study,
    and a study verdict that is only written after E1 means a full run
    leaves a partial-study artifact behind.
    """

    def test_every_runner_registers_the_study_plan(self):
        for name in ("run_e1", "run_e2", "run_e3"):
            source = inspect.getsource(getattr(experiments, name))
            self.assertIn("_publish_analysis_plan", source, name)
            self.assertIn("_study_design_preflight", source, name)

    def test_every_runner_refreshes_the_study_verdict(self):
        for name in ("run_e1", "run_e2", "run_e3"):
            source = inspect.getsource(getattr(experiments, name))
            self.assertIn("_study_level_adjudication", source, name)

    def test_run_all_writes_the_verdict_after_every_family(self):
        source = inspect.getsource(experiments.run_all)
        self.assertIn("_study_level_adjudication", source)
        self.assertLess(source.index("run_e3("),
                        source.index("_study_level_adjudication"))

    def test_the_analysis_plan_fragment_is_loaded(self):
        for name in ("_publish_analysis_plan", "_study_design_preflight",
                     "_study_level_adjudication", "_analysis_plan_audit",
                     "_primary_contrast_reachability",
                     "_primary_contrast_single_factor", "_study_cost_plan"):
            self.assertTrue(callable(getattr(experiments, name)), name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
