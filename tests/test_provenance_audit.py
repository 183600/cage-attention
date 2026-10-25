"""Semantic provenance-audit tests (``--verify-provenance``).

Hash verification answers "did the bytes change?".  These tests cover the
next question: is every row really joinable to the setup/environment it
declares, and does one ``run_id`` name only one protocol?
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import pathlib
import tempfile
import unittest

from cage import experiments
from cage.experiments import Settings


class ProvenanceAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_settings = Settings(device="cpu")
        cls.base = experiments._run_context(base_settings)
        other_settings = Settings(device="cpu", d_model=128)
        cls.other = experiments._run_context(other_settings)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _record(self, context=None, **updates):
        record = copy.deepcopy(context or self.base)
        record.update(updates)
        return record

    def _write_json(self, name: str, payload) -> pathlib.Path:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _write_jsonl(self, name: str, records) -> pathlib.Path:
        path = self.root / name
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8")
        return path

    def _audit(self, require_homogeneous=False):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = experiments._audit_provenance(
                self.root, require_homogeneous=require_homogeneous)
        return code, out.getvalue()

    # ------------------------------------------------------------------
    # happy path
    # ------------------------------------------------------------------
    def test_clean_result_file_passes(self):
        self._write_json("environment.json", self.base)
        self._write_jsonl("e1_runs.jsonl", [self._record()])
        code, text = self._audit()
        self.assertEqual(code, 0, text)
        self.assertIn("PASS", text)

    def test_compact_stamp_matches_its_run_anchor(self):
        self._write_json("environment.json", self.base)
        self._write_jsonl("e1_runs.jsonl", [self._record()])
        stamp = {
            key: self.base[key] for key in (
                "run_id", "schema_version", "setup_sha256", "git_sha",
                "source_tree_sha256")
        }
        self._write_jsonl("e1_error_cases.jsonl", [stamp])
        code, text = self._audit()
        self.assertEqual(code, 0, text)
        self.assertNotIn("no full-context anchor", text)

    def test_unknown_compact_stamp_is_a_warning_not_a_failure(self):
        self._write_json("environment.json", self.base)
        self._write_jsonl("e1_runs.jsonl", [self._record()])
        stamp = {
            key: self.base[key] for key in (
                "run_id", "schema_version", "setup_sha256", "git_sha",
                "source_tree_sha256")
        }
        stamp["run_id"] = "unseen-run"
        self._write_jsonl("e1_error_cases.jsonl", [stamp])
        code, text = self._audit()
        self.assertEqual(code, 0, text)
        self.assertIn("no full-context anchor", text)

    # ------------------------------------------------------------------
    # self-consistency
    # ------------------------------------------------------------------
    def test_tampered_config_does_not_match_its_setup_id(self):
        record = self._record()
        record["config"] = dict(record["config"], steps=1)
        self._write_jsonl("e1_runs.jsonl", [record])
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("does not match the hash", text)

    def test_tampered_environment_does_not_match_its_id(self):
        record = self._record()
        record["environment"] = dict(record["environment"], torch="0.0.0")
        self._write_jsonl("e1_runs.jsonl", [record])
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("does not match the hash of the environment", text)

    def test_missing_provenance_is_rejected(self):
        self._write_jsonl("e1_runs.jsonl", [{"run_id": "x"}])
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("missing provenance keys", text)

    def test_malformed_json_line_is_rejected(self):
        (self.root / "e1_runs.jsonl").write_text("{not json\n",
                                                 encoding="utf-8")
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("invalid JSON", text)

    def test_malformed_source_hash_is_rejected(self):
        record = self._record(source_tree_sha256="not-a-hash")
        self._write_jsonl("e1_runs.jsonl", [record])
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("source_tree_sha256 must be a 16-character", text)

    def test_malformed_full_context_types_are_rejected(self):
        for key, value, expected in (
                ("config", [], "config must be a JSON object"),
                ("data_fingerprints", "not-an-object",
                 "data_fingerprints must be a JSON object"),
                ("hypotheses_sha256", "not-a-content-id",
                 "hypotheses_sha256 must be")):
            with self.subTest(key=key):
                record = self._record(**{key: value})
                self._write_jsonl("e1_runs.jsonl", [record])
                code, text = self._audit()
                self.assertEqual(code, 1, text)
                self.assertIn(expected, text)
                # Each subtest must start from a clean directory.
                for path in self.root.iterdir():
                    path.unlink()

    def test_tampered_environment_document_is_rejected(self):
        self._write_jsonl("e1_runs.jsonl", [self._record()])
        broken = copy.deepcopy(self.base)
        broken["environment"]["torch"] = "0.0.0"
        self._write_json("environment.json", broken)
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("environment_sha256", text)

    # ------------------------------------------------------------------
    # run-level consistency and homogeneity
    # ------------------------------------------------------------------
    def test_one_run_id_cannot_name_two_setups(self):
        first = self._record(run_id="same-run")
        second = self._record(self.other, run_id="same-run")
        self._write_jsonl("e1_runs.jsonl", [first])
        self._write_jsonl("e2_runs.jsonl", [second])
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("claimed by 2 different setups", text)

    def test_mixed_file_warns_by_default_and_fails_when_required(self):
        self._write_json("environment.json", self.base)
        self._write_jsonl("e1_runs.jsonl", [self._record(),
                                            self._record(self.other,
                                                         run_id="other-run")])
        code, text = self._audit()
        self.assertEqual(code, 0, text)
        self.assertIn("mixes 2 setups", text)
        strict_code, strict_text = self._audit(require_homogeneous=True)
        self.assertEqual(strict_code, 1, strict_text)
        self.assertIn("mixes 2 setups", strict_text)

    def test_no_result_records_returns_two(self):
        (self.root / "environment.json").write_text(
            json.dumps(self.base), encoding="utf-8")
        code, text = self._audit()
        self.assertEqual(code, 2, text)

    def test_malformed_anchor_without_results_still_fails(self):
        broken = copy.deepcopy(self.base)
        broken["environment"]["torch"] = "0.0.0"
        self._write_json("environment.json", broken)
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("environment_sha256", text)

    def test_cli_route_writes_nothing(self):
        self._write_json("environment.json", self.base)
        self._write_jsonl("e1_runs.jsonl", [self._record()])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = experiments.main(
                ["--verify-provenance", "--out", str(self.root)])
        self.assertEqual(code, 0, out.getvalue())
        self.assertFalse((self.root / "run_index.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
