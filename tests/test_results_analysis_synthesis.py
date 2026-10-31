"""Registered-metric coverage, conjunction tests and project-level synthesis."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import pathlib
import tempfile
import unittest

from cage.experiments import (
    ContrastLog,
    _hypothesis_conjunction_p,
    _hypothesis_evidence_synthesis,
    _hypothesis_registry_coverage,
    _project_evidence_synthesis,
    _registered_decision_keys,
    main,
)


def _row(hyp, metric, p_value, supports=True):
    return {
        "hypothesis": hyp,
        "rule": f"{hyp}-{metric}",
        "metric": metric,
        "direction": "+",
        "test": "difference",
        "p_value": p_value,
        "supports_hypothesis": supports,
        "claim_eligible": True,
        "controlled": True,
    }


def _h2a_rows(p_value, metrics=("accuracy_at_longest", "accuracy_drop",
                                "accuracy_slope_log_len")):
    return [_row("H2a", metric, p_value) for metric in metrics]


def _write_family(root, tag, rows, q=0.05):
    path = pathlib.Path(root) / f"{tag}_contrasts.json"
    raw = json.dumps({"tag": tag, "q": q, "rows": rows}).encode("utf-8")
    path.write_bytes(raw)
    return path, raw


class RegistryCoverageTest(unittest.TestCase):
    def test_h3b_secondary_readout_is_context_not_a_required_decision(self):
        self.assertEqual(_registered_decision_keys("H3b"),
                         ["resid_norm_last_over_first"])
        coverage = _hypothesis_registry_coverage(
            [{"hypothesis": "H3b",
              "registry_keys": ["resid_norm_last_over_first"]}], "H3b")
        self.assertTrue(coverage["complete"])
        self.assertEqual(coverage["missing"], [])
        self.assertNotIn("gate_depth_ratio", coverage["registered"])

    def test_missing_registered_metric_is_visible(self):
        coverage = _hypothesis_registry_coverage(
            [{"hypothesis": "H2a",
              "registry_keys": ["accuracy_at_longest"]}], "H2a")
        self.assertFalse(coverage["complete"])
        self.assertIn("accuracy_drop", coverage["missing"])
        self.assertIn("accuracy_slope_log_len", coverage["missing"])

    def test_unknown_hypothesis_is_not_called_complete(self):
        coverage = _hypothesis_registry_coverage(
            [{"hypothesis": "HX", "registry_keys": ["accuracy"]}], "HX")
        self.assertFalse(coverage["known"])
        self.assertIsNone(coverage["complete"])

    def test_null_bound_key_does_not_cover_the_display_decision_metric(self):
        log = ContrastLog("e2", hypotheses=("H2c",), min_seeds=3)
        with contextlib.redirect_stdout(io.StringIO()):
            log.print_equivalence(
                "reverse", "H2c", "accuracy_at_longest",
                [0.0, 0.001, -0.001, 0.002, -0.002],
                bound_metric="calib_minus_len_only_accuracy_at_longest")
        row = log.rows[0]
        self.assertEqual(
            row["registry_keys"],
            ["calib_minus_len_only_accuracy_at_longest"])
        coverage = _hypothesis_registry_coverage([row], "H2c")
        self.assertIn("accuracy_at_longest", coverage["missing"])
        self.assertIn("accuracy_drop", coverage["missing"])


class ConjunctionTest(unittest.TestCase):
    def test_intersection_union_p_is_the_maximum_component_p(self):
        rows = [_row("H2a", "accuracy_at_longest", 0.01),
                _row("H2a", "accuracy_drop", 0.03)]
        result = _hypothesis_conjunction_p(rows, "H2a")
        self.assertTrue(result["complete"])
        self.assertEqual(result["n_required"], 2)
        self.assertAlmostEqual(result["p"], 0.03)

    def test_non_finite_component_leaves_the_conjunction_undecided(self):
        rows = [_row("H2a", "accuracy_at_longest", 0.01),
                _row("H2a", "accuracy_drop", float("nan"))]
        result = _hypothesis_conjunction_p(rows, "H2a")
        self.assertFalse(result["complete"])
        self.assertIsNone(result["p"])
        self.assertEqual(result["n_finite"], 1)

    def test_descriptive_rows_do_not_enter_the_conjunction(self):
        descriptive = _row("H2a", "accuracy_drop", 0.50)
        descriptive["direction"] = None
        rows = [_row("H2a", "accuracy_at_longest", 0.02), descriptive]
        result = _hypothesis_conjunction_p(rows, "H2a")
        self.assertTrue(result["complete"])
        self.assertEqual(result["n_required"], 1)
        self.assertAlmostEqual(result["p"], 0.02)


class HypothesisMultiplicityTest(unittest.TestCase):
    def test_complete_conjunction_passes_the_hypothesis_family(self):
        synthesis = _hypothesis_evidence_synthesis(
            _h2a_rows(0.01), ("H2a",), q=0.05)
        entry = synthesis["H2a"]
        self.assertEqual(entry["status"], "supported")
        self.assertTrue(entry["registry_complete"])
        self.assertTrue(entry["conjunction_fdr_reject"])
        self.assertTrue(entry["claim_level_supported"])
        self.assertEqual(entry["claim_level_status"], "supported")

    def test_family_q_can_withhold_claim_level_support(self):
        synthesis = _hypothesis_evidence_synthesis(
            _h2a_rows(0.03), ("H2a",), q=0.01)
        entry = synthesis["H2a"]
        self.assertEqual(entry["status"], "supported")
        self.assertFalse(entry["conjunction_fdr_reject"])
        self.assertFalse(entry["claim_level_supported"])
        self.assertEqual(entry["claim_level_status"],
                         "not_family_significant")

    def test_partial_registry_withholds_claim_level_support(self):
        synthesis = _hypothesis_evidence_synthesis(
            _h2a_rows(0.001, metrics=("accuracy_at_longest",)),
            ("H2a",), q=0.05)
        entry = synthesis["H2a"]
        self.assertEqual(entry["status"], "supported")
        self.assertFalse(entry["registry_complete"])
        self.assertFalse(entry["claim_level_supported"])
        self.assertEqual(entry["claim_level_status"], "partial_registry")

    def test_h3b_context_metric_does_not_make_coverage_partial(self):
        synthesis = _hypothesis_evidence_synthesis(
            [{"hypothesis": "H3b", "metric": "resid_norm_last_over_first",
              "test": "difference", "p_value": 0.01,
              "supports_hypothesis": True}], ("H3b",), q=0.05)
        entry = synthesis["H3b"]
        self.assertTrue(entry["registry_complete"])
        self.assertEqual(entry["status"], "undirected")


class ProjectEvidenceSynthesisTest(unittest.TestCase):
    def test_pooled_family_and_source_hashes_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            strong, strong_raw = _write_family(
                tmp, "e1",
                [_row("H1a", "accuracy", 0.0001)])
            weak, weak_raw = _write_family(
                tmp, "e2",
                [_row("H2a", "accuracy_at_longest", 0.50)])
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _project_evidence_synthesis(tmp)
            self.assertIsNotNone(payload)
            self.assertEqual(payload["n_rows"], 2)
            self.assertEqual(payload["n_family_supporting"], 2)
            self.assertEqual(payload["n_project_supporting"], 1)
            self.assertEqual(payload["n_family_support_lost_globally"], 1)
            self.assertEqual(
                payload["family_support_lost_globally_contrasts"],
                ["e2/H2a/H2a-accuracy_at_longest (accuracy_at_longest)"])
            self.assertEqual(
                payload["sources"]["e1"]["sha256"],
                hashlib.sha256(strong_raw).hexdigest())
            self.assertEqual(payload["sources"]["e2"]["bytes"],
                             len(weak_raw))
            self.assertEqual(payload["sources"]["e3"], {"present": False})
            self.assertTrue(strong.is_file())
            self.assertTrue(weak.is_file())
            self.assertTrue(
                (pathlib.Path(tmp) / "project_evidence.json").is_file())

    def test_mislabeled_source_is_excluded_from_the_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "e1_contrasts.json").write_text(json.dumps({
                "tag": "e2", "q": 0.05,
                "rows": [_row("H1a", "accuracy", 0.001)]}))
            _write_family(tmp, "e2",
                          [_row("H2a", "accuracy_at_longest", 0.01)])
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _project_evidence_synthesis(tmp)
            self.assertEqual(payload["invalid_sources"], ["e1"])
            self.assertFalse(payload["sources"]["e1"]["valid"])
            self.assertIn("does not match file tag",
                          payload["sources"]["e1"]["error"])
            self.assertEqual(payload["n_rows"], 1)
            self.assertEqual(payload["sources"]["e2"]["q_matches_request"],
                             True)

    def test_mixed_source_q_levels_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_family(tmp, "e1",
                          [_row("H1a", "accuracy", 0.001)], q=0.10)
            _write_family(tmp, "e2",
                          [_row("H2a", "accuracy_at_longest", 0.01)],
                          q=0.05)
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                payload = _project_evidence_synthesis(tmp)
            self.assertEqual(payload["source_q_mismatches"], ["e1"])
            self.assertTrue(payload["mixed_source_q"])
            self.assertIn("mixed q levels", captured.getvalue())

    def test_project_hypothesis_status_reads_project_level_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_family(tmp, "e2", _h2a_rows(0.06), q=0.10)
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _project_evidence_synthesis(tmp)
            entry = payload["hypothesis_synthesis"]["H2a"]
            self.assertEqual(entry["support_basis"], "project_supports")
            self.assertEqual(entry["refutation_basis"],
                             "project_fdr_reject")
            self.assertEqual(entry["n_source_family_confirmed"], 3)
            self.assertEqual(entry["n_confirmed"], 0)
            self.assertEqual(entry["status"], "inconclusive")
            self.assertFalse(entry["claim_level_supported"])
            self.assertEqual(payload["n_hypotheses_supported"], 0)

    def test_no_usable_family_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_family(tmp, "e1", [])
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                payload = _project_evidence_synthesis(tmp)
            self.assertIsNone(payload)
            self.assertIn("nothing to synthesize", captured.getvalue())

    def test_invalid_q_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _project_evidence_synthesis(tmp, q=1.0)

    def test_cli_synthesis_writes_and_verifies_an_artifact_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_family(tmp, "e2",
                          [_row("H2a", "accuracy_at_longest", 0.01)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(["--synthesize-results", "--out", tmp,
                          "--fdr-q", "0.05"]), 0)
            index = json.loads(
                (pathlib.Path(tmp) / "artifact_index.json").read_text())
            files = {entry["file"] for entry in index["artifacts"]}
            self.assertIn("e2_contrasts.json", files)
            self.assertIn("project_evidence.json", files)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(["--verify-artifacts", "--out", tmp]), 0)


if __name__ == "__main__":
    unittest.main()
