"""Read-only source-version audit tests (``--check-source``).

Artifact verification answers "did the bytes change?" and provenance audit
answers "are the rows internally consistent?".  These tests cover the
remaining question: were the archived numbers produced by the package
source that is checked out right now?
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


class SourceCheckTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.context = experiments._run_context(Settings(device="cpu"))

    def tearDown(self):
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
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

    def _record(self, **updates):
        record = copy.deepcopy(self.context)
        record.update(updates)
        return record

    def _check(self, root=None):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = experiments._check_source(root or self.root)
        return code, out.getvalue()

    # ------------------------------------------------------------------
    # happy path
    # ------------------------------------------------------------------
    def test_matching_environment_and_rows_pass(self):
        self._write_json("environment.json", self._record())
        self._write_jsonl("e1_runs.jsonl", [self._record()])
        code, text = self._check()
        self.assertEqual(code, 0, text)
        self.assertIn("PASS", text)
        self.assertIn(experiments._source_tree_sha256(), text)
        self.assertIn("2 code-provenance record(s)", text)

    def test_manifest_and_nested_provenance_pass(self):
        self._write_json("e1_manifest.json", self._record())
        self._write_json("efficiency.json", {
            "_provenance": {
                "source_tree_sha256": experiments._source_tree_sha256(),
            },
        })
        code, text = self._check()
        self.assertEqual(code, 0, text)
        self.assertIn("PASS", text)

    # ------------------------------------------------------------------
    # stale code
    # ------------------------------------------------------------------
    def test_mismatch_reports_file_line_and_run(self):
        stale = self._record(
            source_tree_sha256="0" * 16, run_id="run-old")
        self._write_jsonl("e1_runs.jsonl", [stale])
        code, text = self._check()
        self.assertEqual(code, 1, text)
        self.assertIn("FAIL", text)
        self.assertIn("e1_runs.jsonl:1", text)
        self.assertIn("run-old", text)
        self.assertIn("0" * 16, text)

    def test_mixed_versions_fail(self):
        self._write_jsonl("e1_runs.jsonl", [
            self._record(),
            self._record(source_tree_sha256="1" * 16, run_id="run-stale"),
        ])
        code, text = self._check()
        self.assertEqual(code, 1, text)
        self.assertIn("2 distinct source version(s)", text)
        self.assertIn("(current)", text)
        self.assertIn("(other)", text)

    def test_records_inside_top_level_arrays_are_checked(self):
        stale = self._record(
            source_tree_sha256="2" * 16, run_id="run-in-array")
        self._write_json("pooled.json", [self._record(), stale])
        code, text = self._check()
        self.assertEqual(code, 1, text)
        self.assertIn("pooled.json", text)
        self.assertIn("run-in-array", text)
        self.assertIn("2 distinct source version(s)", text)

        (self.root / "pooled.json").unlink()
        self._write_jsonl("e1_runs.jsonl", [[self._record(), stale]])
        code, text = self._check()
        self.assertEqual(code, 1, text)
        self.assertIn("run-in-array", text)

    def test_records_nested_under_mapping_keys_are_checked(self):
        stale = self._record(
            source_tree_sha256="4" * 16, run_id="run-nested-key")
        self._write_json("tuning.json", {
            "runs": [self._record(), stale],
        })
        code, text = self._check()
        self.assertEqual(code, 1, text)
        self.assertIn("tuning.json", text)
        self.assertIn("run-nested-key", text)
        self.assertIn("2 distinct source version(s)", text)

    def test_nested_provenance_is_counted_once(self):
        self._write_json("efficiency.json", {
            "_provenance": self._record(),
        })
        code, text = self._check()
        self.assertEqual(code, 0, text)
        self.assertIn("scanned 1 code-provenance record(s)", text)
        self.assertIn("1 record(s)", text)

    def test_missing_hash_on_run_record_fails(self):
        record = self._record()
        del record["source_tree_sha256"]
        self._write_jsonl("e1_runs.jsonl", [record])
        code, text = self._check()
        self.assertEqual(code, 1, text)
        self.assertIn("no source_tree_sha256", text)

    def test_missing_hash_in_nested_provenance_fails(self):
        self._write_json("efficiency.json", {
            "_provenance": {
                "run_id": "run-nested",
                "setup_sha256": "3" * 16,
            },
        })
        code, text = self._check()
        self.assertEqual(code, 1, text)
        self.assertIn("no source_tree_sha256", text)

    def test_malformed_hash_fails(self):
        self._write_jsonl("e1_runs.jsonl", [
            self._record(source_tree_sha256="not-a-hash"),
        ])
        code, text = self._check()
        self.assertEqual(code, 1, text)
        self.assertIn("malformed", text)

    def test_malformed_json_fails(self):
        (self.root / "e1_runs.jsonl").write_text(
            "this is not json\n", encoding="utf-8")
        code, text = self._check()
        self.assertEqual(code, 1, text)
        self.assertIn("invalid JSON", text)

    # ------------------------------------------------------------------
    # nothing to check
    # ------------------------------------------------------------------
    def test_empty_directory_returns_two(self):
        code, text = self._check()
        self.assertEqual(code, 2, text)
        self.assertIn("no archived code-provenance records", text)

    def test_missing_directory_returns_two(self):
        code, text = self._check(self.root / "missing")
        self.assertEqual(code, 2, text)
        self.assertIn("no results directory", text)

    def test_unrelated_json_is_ignored(self):
        self._write_json("notes.json", {"note": "not a result record"})
        code, text = self._check()
        self.assertEqual(code, 2, text)
        self.assertIn("no archived code-provenance records", text)

    # ------------------------------------------------------------------
    # CLI contract
    # ------------------------------------------------------------------
    def test_cli_check_writes_nothing(self):
        self._write_json("environment.json", self._record())
        before = {path.relative_to(self.root)
                  for path in self.root.rglob("*")}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = experiments.main(
                ["--check-source", "--out", str(self.root)])
        after = {path.relative_to(self.root)
                 for path in self.root.rglob("*")}
        self.assertEqual(code, 0, out.getvalue())
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
