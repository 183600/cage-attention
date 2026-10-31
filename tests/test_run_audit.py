"""Evidence-completeness tests (``--verify-runs``).

The artifact index proves bytes have not changed and the provenance audit
proves rows are joinable to their setup.  These tests cover the remaining
evidence question: did every invocation actually finish, and were all rows
produced under one critical software/numerical environment?
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile
import unittest

from cage import experiments


class RunAuditTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_json(self, name: str, payload) -> pathlib.Path:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _write_jsonl(self, name: str, rows) -> pathlib.Path:
        path = self.root / name
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8")
        return path

    def _event(self, run_id, event, **updates):
        return {"run_id": run_id, "event": event, "created_utc": "now",
                **updates}

    def _history(self, run_id, compatible, **updates):
        return {"run_id": run_id, "compatible": compatible,
                "identical": compatible, "differences": [], **updates}

    def _audit(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = experiments._verify_runs(self.root)
        return code, out.getvalue()

    def test_complete_run_with_compatible_history_passes(self):
        self._write_json("artifact_index.json",
                         {"completed": True, "run_id": "r1"})
        self._write_jsonl("run_index.jsonl", [
            self._event("r1", "started"),
            self._event("r1", "completed"),
        ])
        self._write_jsonl("environment_history.jsonl", [
            self._history("r1", None),
            self._history("r2", True),
        ])
        code, text = self._audit()
        self.assertEqual(code, 0, text)
        self.assertIn("1 completed run", text)
        self.assertIn("2 history row", text)

    def test_started_without_terminal_event_fails(self):
        self._write_json("artifact_index.json",
                         {"completed": True, "run_id": "r1"})
        self._write_jsonl("run_index.jsonl", [self._event("r1", "started")])
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("no completed/aborted/failed terminal event", text)
        self.assertIn("r1", text)

    def test_aborted_and_failed_runs_are_not_citable(self):
        for event in ("aborted", "failed"):
            with self.subTest(event=event):
                self._write_jsonl("run_index.jsonl", [
                    self._event("r1", "started"),
                    self._event("r1", event, reason="boom"),
                ])
                code, text = self._audit()
                self.assertEqual(code, 1, text)
                self.assertIn(event, text)
                self.assertIn("boom", text)
                self.assertIn("cannot be cited", text)

    def test_terminal_event_without_started_event_fails(self):
        self._write_jsonl("run_index.jsonl", [
            self._event("r1", "completed"),
        ])
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("without a started event", text)

    def test_terminal_event_before_started_fails(self):
        self._write_jsonl("run_index.jsonl", [
            self._event("r1", "completed"),
            self._event("r1", "started"),
        ])
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("precedes its started event", text)

    def test_duplicate_lifecycle_events_fail(self):
        self._write_jsonl("run_index.jsonl", [
            self._event("r1", "started"),
            self._event("r1", "started"),
            self._event("r1", "completed"),
            self._event("r1", "completed"),
        ])
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("2 started events", text)
        self.assertIn("2 terminal events", text)

    def test_critical_environment_drift_fails(self):
        self._write_json("artifact_index.json", {"completed": True})
        self._write_jsonl("run_index.jsonl", [
            self._event("r1", "started"),
            self._event("r1", "completed"),
        ])
        self._write_jsonl("environment_history.jsonl", [
            self._history(
                "r1", False,
                differences=[{"field": "torch", "severity": "critical"}],
                reason="torch changed"),
        ])
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("non-compatible environment", text)
        self.assertIn("torch changed", text)

    def test_advisory_environment_drift_is_a_warning(self):
        self._write_json("artifact_index.json", {"completed": True})
        self._write_jsonl("run_index.jsonl", [
            self._event("r1", "started"),
            self._event("r1", "completed"),
        ])
        self._write_jsonl("environment_history.jsonl", [
            self._history(
                "r1", True, identical=False,
                differences=[{"field": "platform", "severity": "advisory"}],
                reason="critical fields match"),
        ])
        code, text = self._audit()
        self.assertEqual(code, 0, text)
        self.assertIn("advisory environment drift", text)

    def test_malformed_lifecycle_and_history_rows_fail_closed(self):
        (self.root / "run_index.jsonl").write_text(
            "{not json\n[]\n", encoding="utf-8")
        (self.root / "environment_history.jsonl").write_text(
            "{\"compatible\": \"yes\"}\n", encoding="utf-8")
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("run_index.jsonl:1", text)
        self.assertIn("run_index.jsonl:2", text)
        self.assertIn("compatible must be true, false or null", text)

    def test_last_artifact_index_must_be_completed(self):
        self._write_json("artifact_index.json",
                         {"completed": False, "run_id": "r1"})
        self._write_jsonl("run_index.jsonl", [
            self._event("r1", "started"),
            self._event("r1", "completed"),
        ])
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("completed=False", text)

    def test_history_requires_an_explicit_boolean_verdict(self):
        self._write_jsonl("run_index.jsonl", [
            self._event("r1", "started"),
            self._event("r1", "completed"),
        ])
        self._write_jsonl("environment_history.jsonl", [
            {"run_id": "r1", "differences": []},
            {"run_id": "r1", "compatible": 1, "differences": []},
        ])
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("missing compatible verdict", text)
        self.assertIn("compatible must be true, false or null", text)

    def test_history_critical_difference_cannot_claim_compatibility(self):
        self._write_jsonl("run_index.jsonl", [
            self._event("r1", "started"),
            self._event("r1", "completed"),
        ])
        self._write_jsonl("environment_history.jsonl", [
            self._history(
                "r1", True,
                differences=[{"field": "torch", "severity": "critical"}]),
        ])
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("compatible=true contradicts", text)
        self.assertIn("torch", text)

    def test_history_requires_run_id_and_difference_list(self):
        self._write_jsonl("run_index.jsonl", [
            self._event("r1", "started"),
            self._event("r1", "completed"),
        ])
        self._write_jsonl("environment_history.jsonl", [
            {"compatible": True, "differences": []},
            {"run_id": "r1", "compatible": True, "differences": "none"},
        ])
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("run_id must be a non-empty string", text)
        self.assertIn("differences must be a list", text)

    def test_artifact_index_run_must_have_completed_lifecycle(self):
        self._write_json("artifact_index.json",
                         {"completed": True, "run_id": "r2"})
        self._write_jsonl("run_index.jsonl", [
            self._event("r1", "started"),
            self._event("r1", "completed"),
        ])
        self._write_jsonl("environment_history.jsonl", [
            self._history("r1", True),
        ])
        code, text = self._audit()
        self.assertEqual(code, 1, text)
        self.assertIn("r2 has no completed lifecycle event", text)

    def test_missing_run_log_returns_two(self):
        code, text = self._audit()
        self.assertEqual(code, 2, text)
        self.assertIn("no run lifecycle log", text)

    def test_empty_run_log_returns_two(self):
        (self.root / "run_index.jsonl").write_text("", encoding="utf-8")
        code, text = self._audit()
        self.assertEqual(code, 2, text)
        self.assertIn("is empty", text)

    def test_missing_history_is_reported_but_does_not_hide_completion(self):
        self._write_jsonl("run_index.jsonl", [
            self._event("r1", "started"),
            self._event("r1", "completed"),
        ])
        code, text = self._audit()
        self.assertEqual(code, 0, text)
        self.assertIn("predates the environment-compatibility contract", text)

    def test_cli_check_is_read_only(self):
        self._write_json("artifact_index.json", {"completed": True})
        self._write_jsonl("run_index.jsonl", [
            self._event("r1", "started"),
            self._event("r1", "completed"),
        ])
        self._write_jsonl("environment_history.jsonl", [
            self._history("r1", True),
        ])
        before = {path.relative_to(self.root)
                  for path in self.root.rglob("*")}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = experiments.main(
                ["--verify-runs", "--out", str(self.root)])
        after = {path.relative_to(self.root)
                 for path in self.root.rglob("*")}
        self.assertEqual(code, 0, out.getvalue())
        self.assertEqual(before, after)
        self.assertIn("1 completed run", out.getvalue())

    def test_cli_rejects_combination_with_replay_and_synthesis(self):
        for argv in (
            ["--verify-runs", "--replay-manifest", "manifest.json"],
            ["--verify-runs", "--synthesize-results"],
        ):
            with self.subTest(argv=argv):
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    code = experiments.main(
                        [*argv, "--out", str(self.root)])
                self.assertEqual(code, 2, out.getvalue())
                self.assertIn("cannot be combined", out.getvalue())


if __name__ == "__main__":
    unittest.main()
