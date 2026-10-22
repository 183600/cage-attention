"""Integrity tests for the cumulative results artifact index."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from cage import experiments


class ArtifactIndexTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name).resolve()
        self._saved_artifacts = experiments._RUN_ARTIFACTS
        experiments._RUN_ARTIFACTS = set()
        self._context = mock.patch.object(
            experiments, "_run_context",
            return_value={"run_id": "test-run", "schema_version": 1})
        self._context.start()

    def tearDown(self):
        self._context.stop()
        experiments._RUN_ARTIFACTS = self._saved_artifacts
        self._tmp.cleanup()

    def _write(self, relative: str, payload: str) -> pathlib.Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        experiments._register_artifact(path)
        return path

    def _index(self) -> dict:
        return json.loads(
            (self.root / "artifact_index.json").read_text(encoding="utf-8"))

    def _verify(self) -> int:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = experiments._verify_artifacts(self.root)
        self.stdout = out.getvalue()
        return code

    def test_index_is_cumulative_across_invocations(self):
        self._write("first.json", '{"a": 1}\n')
        experiments._write_artifact_index(
            self.root, None, "first", completed=True, elapsed_seconds=0.1)
        self.assertEqual([e["file"] for e in self._index()["artifacts"]],
                         ["first.json"])

        # A second invocation writes a different file only.  The first
        # file must still be covered by the index.
        experiments._RUN_ARTIFACTS = set()
        self._write("second.json", '{"b": 2}\n')
        experiments._write_artifact_index(
            self.root, None, "second", completed=True, elapsed_seconds=0.2)
        payload = self._index()
        self.assertEqual([e["file"] for e in payload["artifacts"]],
                         ["first.json", "second.json"])
        self.assertEqual(payload["artifact_count"], 2)
        self.assertEqual(payload["current_artifact_count"], 1)
        self.assertEqual(payload["updated_entries"], ["second.json"])
        self.assertEqual(self._verify(), 0)

    def test_rewritten_file_supersedes_its_entry(self):
        path = self._write("log.jsonl", "one\n")
        experiments._write_artifact_index(
            self.root, None, "first", completed=True, elapsed_seconds=0.0)
        with path.open("a", encoding="utf-8") as fh:
            fh.write("two\n")
        experiments._RUN_ARTIFACTS = {path.resolve()}
        experiments._write_artifact_index(
            self.root, None, "second", completed=True, elapsed_seconds=0.0)
        entries = self._index()["artifacts"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["records"], 2)
        self.assertEqual(self._verify(), 0)

    def test_tampering_with_an_earlier_file_is_detected(self):
        first = self._write("first.json", '{"a": 1}\n')
        experiments._write_artifact_index(
            self.root, None, "first", completed=True, elapsed_seconds=0.0)
        experiments._RUN_ARTIFACTS = set()
        self._write("second.json", '{"b": 2}\n')
        experiments._write_artifact_index(
            self.root, None, "second", completed=True, elapsed_seconds=0.0)
        first.write_text('{"a": 999}\n', encoding="utf-8")
        self.assertEqual(self._verify(), 1)
        self.assertIn("first.json", self.stdout)

    def test_missing_earlier_file_is_detected(self):
        first = self._write("first.json", '{"a": 1}\n')
        experiments._write_artifact_index(
            self.root, None, "first", completed=True, elapsed_seconds=0.0)
        first.unlink()
        self.assertEqual(self._verify(), 1)
        self.assertIn("missing", self.stdout)

    def test_nested_artifact_is_covered(self):
        self._write("nested/report.json", '{"ok": true}\n')
        experiments._write_artifact_index(
            self.root, None, "nested", completed=True, elapsed_seconds=0.0)
        self.assertEqual([e["file"] for e in self._index()["artifacts"]],
                         ["nested/report.json"])
        self.assertEqual(self._verify(), 0)

    def test_malformed_previous_index_is_reported(self):
        (self.root / "artifact_index.json").write_text(
            "{not json", encoding="utf-8")
        self._write("fresh.json", '{"ok": true}\n')
        payload = experiments._write_artifact_index(
            self.root, None, "fresh", completed=True, elapsed_seconds=0.0)
        self.assertIn("previous_index_error", payload)
        self.assertEqual([e["file"] for e in payload["artifacts"]],
                         ["fresh.json"])

    def test_malformed_previous_entry_is_not_trusted(self):
        (self.root / "artifact_index.json").write_text(
            json.dumps({"artifacts": [{"file": "old.json", "bytes": 1}]}),
            encoding="utf-8")
        self._write("fresh.json", '{"ok": true}\n')
        payload = experiments._write_artifact_index(
            self.root, None, "fresh", completed=True, elapsed_seconds=0.0)
        self.assertIn("previous_index_error", payload)
        self.assertEqual([e["file"] for e in payload["artifacts"]],
                         ["fresh.json"])
        self._write("a.json", '{"a": 1}\n')
        experiments._write_artifact_index(
            self.root, None, "first", completed=True, elapsed_seconds=0.0)
        payload = self._index()
        payload["artifacts"].append(dict(payload["artifacts"][0]))
        (self.root / "artifact_index.json").write_text(
            json.dumps(payload), encoding="utf-8")
        self.assertEqual(self._verify(), 1)
        self.assertIn("duplicate", self.stdout)

    def test_path_outside_results_is_rejected(self):
        index = {
            "run_id": "test-run",
            "artifacts": [{
                "file": "../escape.json",
                "bytes": 2,
                "sha256": "0" * 64,
            }],
        }
        (self.root / "artifact_index.json").write_text(
            json.dumps(index), encoding="utf-8")
        self.assertEqual(self._verify(), 1)
        self.assertIn("outside results directory", self.stdout)

    def test_no_index_returns_two(self):
        self.assertEqual(self._verify(), 2)


if __name__ == "__main__":
    unittest.main()
