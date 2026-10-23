"""Append-only integrity tests for cumulative JSONL result files.

The artifact index certifies what was on disk when an invocation ended.  A
later invocation must not extend a file that no longer matches that
certificate, otherwise the next index write would silently re-certify a
tampered history.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from cage import experiments


class ArtifactAppendGuardTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name).resolve()
        self._saved = experiments._RUN_ARTIFACTS
        experiments._RUN_ARTIFACTS = set()
        self._context = mock.patch.object(
            experiments, "_run_context",
            return_value={"run_id": "test-run", "schema_version": 1})
        self._context.start()

    def tearDown(self):
        self._context.stop()
        experiments._RUN_ARTIFACTS = self._saved
        self._tmp.cleanup()

    @staticmethod
    def _record(value=1):
        return {
            "run_id": "test-run", "schema_version": 1,
            "setup_sha256": "setup", "git_sha": "deadbeef",
            "source_tree_sha256": "source", "value": value,
        }

    def _certify(self, path):
        experiments._register_artifact(path)
        experiments._write_artifact_index(
            self.root, None, "test", completed=True, elapsed_seconds=0.0)
        # Simulate a fresh invocation: this process no longer remembers the
        # file, so the pre-append certificate is the only source of truth.
        experiments._RUN_ARTIFACTS = set()

    def _verify(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return experiments._verify_artifacts(self.root)

    def test_valid_append_is_allowed_and_recertified(self):
        path = self.root / "runs.jsonl"
        experiments._write_result(path, self._record(1))
        self._certify(path)
        experiments._write_result(path, self._record(2))
        self.assertEqual(path.read_text(encoding="utf-8").count("\n"), 2)
        experiments._write_artifact_index(
            self.root, None, "test", completed=True, elapsed_seconds=0.0)
        self.assertEqual(self._verify(), 0)

    def test_tampered_file_is_refused(self):
        path = self.root / "runs.jsonl"
        experiments._write_result(path, self._record(1))
        self._certify(path)
        path.write_text('{"value": 999}\n', encoding="utf-8")
        with self.assertRaises(experiments.TrackingError):
            experiments._write_result(path, self._record(2))
        # The refused append must not have changed the tampered file either;
        # the caller has to restore or re-run verification explicitly.
        self.assertEqual(path.read_text(encoding="utf-8"),
                         '{"value": 999}\n')

    def test_truncated_file_is_refused(self):
        path = self.root / "runs.jsonl"
        experiments._write_result(path, self._record(1))
        self._certify(path)
        path.write_text("", encoding="utf-8")
        with self.assertRaises(experiments.TrackingError):
            experiments._write_result(path, self._record(2))

    def test_deleted_file_is_not_silently_recreated(self):
        path = self.root / "runs.jsonl"
        experiments._write_result(path, self._record(1))
        self._certify(path)
        path.unlink()
        with self.assertRaises(experiments.TrackingError):
            experiments._write_result(path, self._record(2))
        self.assertFalse(path.exists())

    def test_malformed_index_refuses_append(self):
        path = self.root / "runs.jsonl"
        path.write_text("one\n", encoding="utf-8")
        (self.root / "artifact_index.json").write_text(
            "{not json", encoding="utf-8")
        with self.assertRaises(experiments.TrackingError):
            experiments._write_result(path, self._record(2))

    def test_untracked_legacy_file_is_allowed(self):
        path = self.root / "runs.jsonl"
        path.write_text("legacy\n", encoding="utf-8")
        (self.root / "artifact_index.json").write_text(
            json.dumps({"artifacts": []}), encoding="utf-8")
        experiments._write_result(path, self._record(2))
        self.assertEqual(path.read_text(encoding="utf-8").count("\n"), 2)

    def test_torn_tail_without_newline_is_refused(self):
        path = self.root / "runs.jsonl"
        path.write_text('{"partial": ', encoding="utf-8")
        with self.assertRaises(experiments.TrackingError) as ctx:
            experiments._write_result(path, self._record(2))
        self.assertIn("newline", str(ctx.exception))
        self.assertEqual(path.read_text(encoding="utf-8"), '{"partial": ')

    def test_complete_line_without_trailing_newline_is_refused(self):
        path = self.root / "runs.jsonl"
        path.write_text('{"complete": true}', encoding="utf-8")
        with self.assertRaises(experiments.TrackingError) as ctx:
            experiments._write_result(path, self._record(2))
        self.assertIn("newline", str(ctx.exception))
        self.assertEqual(path.read_text(encoding="utf-8"),
                         '{"complete": true}')

    def test_trailing_blank_lines_are_compatible(self):
        path = self.root / "runs.jsonl"
        path.write_text('{"complete": true}\n\n', encoding="utf-8")
        experiments._write_result(path, self._record(2))
        text = path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith('{"complete": true}\n\n{'))
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(text.count('"run_id": "test-run"'), 1)

    def test_nested_artifact_uses_nearest_index(self):
        path = self.root / "nested" / "runs.jsonl"
        experiments._write_result(path, self._record(1))
        self._certify(path)
        experiments._write_artifact_index(
            self.root, None, "test", completed=True, elapsed_seconds=0.0)
        path.write_text("tampered\n", encoding="utf-8")
        with self.assertRaises(experiments.TrackingError):
            experiments._write_result(path, self._record(2))


if __name__ == "__main__":
    unittest.main()
