"""Environment-contract tests: fingerprint, drift diagnosis and history.

The reproducibility layer already pins code, config, seeds and data.  These
tests cover the remaining piece: the *software environment* a result was
produced in, and the fact that appending to a results directory under a
different environment must be visible instead of silently pooled.
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
from cage.experiments import Settings


class EnvironmentFingerprintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = experiments._environment_record(
            Settings(device="cpu"), git_dirty=False)

    def test_inventory_and_declared_dependencies_are_pinned(self):
        for key in ("python_full_version", "package_inventory",
                    "dependency_manifest", "environment_sha256"):
            self.assertIn(key, self.record)
        # torch is a declared runtime dependency, so it must be in both the
        # installed inventory and the version-specific fields.
        self.assertIn("torch", self.record["package_inventory"])
        self.assertGreaterEqual(len(self.record["package_inventory"]), 1)
        self.assertIn("pyproject.toml", self.record["dependency_manifest"])
        for entry in self.record["dependency_manifest"].values():
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertGreaterEqual(entry["bytes"], 0)

    def test_fingerprint_is_stable_and_covers_versions(self):
        first = experiments._environment_fingerprint(self.record)
        self.assertEqual(first, self.record["environment_sha256"])
        self.assertEqual(len(first), 16)
        changed = dict(self.record, torch="0.0.0")
        self.assertNotEqual(first,
                            experiments._environment_fingerprint(changed))

    def test_fingerprint_ignores_launch_metadata(self):
        changed = dict(
            self.record, argv=["python3", "other.py"], cwd="/elsewhere",
            repo_root="/another/checkout", git_dirty=True,
            pythonhashseed="12345")
        self.assertEqual(self.record["environment_sha256"],
                         experiments._environment_fingerprint(changed))


class EnvironmentDriftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = experiments._environment_record(
            Settings(device="cpu"), git_dirty=False)

    def test_same_environment_is_identical(self):
        drift = experiments._environment_drift(self.record, dict(self.record))
        self.assertTrue(drift["compatible"])
        self.assertTrue(drift["identical"])
        self.assertEqual(drift["differences"], [])

    def test_critical_version_drift_is_incompatible(self):
        changed = dict(self.record, torch="0.0.0")
        drift = experiments._environment_drift(self.record, changed)
        self.assertFalse(drift["compatible"])
        self.assertFalse(drift["identical"])
        self.assertIn("torch", [d["field"] for d in
                                drift["critical_differences"]])

    def test_advisory_hardware_drift_is_reported_but_compatible(self):
        changed = dict(self.record, platform="other-platform")
        drift = experiments._environment_drift(self.record, changed)
        self.assertTrue(drift["compatible"])
        self.assertFalse(drift["identical"])
        self.assertEqual([d["field"] for d in drift["advisory_differences"]],
                         ["platform"])
        self.assertEqual(drift["critical_differences"], [])

    def test_missing_critical_field_fails_closed(self):
        changed = dict(self.record)
        changed.pop("package_inventory")
        drift = experiments._environment_drift(self.record, changed)
        self.assertFalse(drift["compatible"])
        self.assertEqual(drift["critical_differences"][0]["kind"],
                         "missing_current")

    def test_nested_mapping_difference_is_summarised(self):
        inventory = dict(self.record["package_inventory"])
        inventory["extra-package"] = "1.0"
        changed = dict(self.record, package_inventory=inventory)
        drift = experiments._environment_drift(self.record, changed)
        self.assertFalse(drift["compatible"])
        detail = drift["critical_differences"][0]
        self.assertIn("extra-package", detail["extra_keys"])
        self.assertNotIn("recorded", detail)

    def test_fingerprint_schema_mismatch_is_critical(self):
        changed = dict(self.record, environment_sha256="0" * 16)
        drift = experiments._environment_drift(self.record, changed)
        self.assertFalse(drift["compatible"])
        self.assertEqual(drift["critical_differences"][0]["kind"],
                         "fingerprint_schema")

    def test_malformed_fingerprint_fails_closed(self):
        for bad in ("", "not-a-hash", 123, "g" * 16):
            with self.subTest(bad=bad):
                drift = experiments._environment_drift(
                    dict(self.record, environment_sha256=bad), self.record)
                self.assertFalse(drift["compatible"])
                self.assertEqual(drift["critical_differences"][0]["kind"],
                                 "malformed")

    def test_malformed_record_is_not_an_empty_match(self):
        for bad in (None, {}, "not-a-record", []):
            with self.subTest(bad=bad):
                drift = experiments._environment_drift(bad, self.record)
                self.assertFalse(drift["compatible"])
                self.assertFalse(drift["identical"])
                self.assertIn("reason", drift)


class EnvironmentHistoryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self._saved = experiments._RUN_ARTIFACTS
        experiments._RUN_ARTIFACTS = set()
        self.env = experiments._environment_record(
            Settings(device="cpu"), git_dirty=False)

    def tearDown(self):
        experiments._RUN_ARTIFACTS = self._saved
        self._tmp.cleanup()

    def _context(self, run_id="r1", environment=None):
        return {
            "run_id": run_id,
            "schema_version": 1,
            "setup_sha256": "setup-" + run_id,
            "git_sha": "deadbeef",
            "source_tree_sha256": "source",
            "environment": dict(environment or self.env),
        }

    def _rows(self):
        path = self.root / "environment_history.jsonl"
        return [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines()]

    def test_first_run_then_identical_run_are_recorded(self):
        first = experiments._record_environment_history(
            self.root, self._context("r1"))
        self.assertIsNone(first["compatible"])
        (self.root / "environment.json").write_text(
            json.dumps(self._context("r1")), encoding="utf-8")
        second = experiments._record_environment_history(
            self.root, self._context("r2"))
        self.assertTrue(second["compatible"])
        self.assertTrue(second["identical"])
        rows = self._rows()
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[0]["compatible"])
        self.assertTrue(rows[1]["compatible"])
        self.assertEqual(rows[1]["previous_environment_sha256"],
                         self.env["environment_sha256"])
        self.assertEqual(rows[1]["current_environment_sha256"],
                         self.env["environment_sha256"])

    def test_critical_drift_is_recorded(self):
        (self.root / "environment.json").write_text(
            json.dumps(self._context("r1", dict(self.env, torch="0.0"))),
            encoding="utf-8")
        drift = experiments._record_environment_history(
            self.root, self._context("r2"))
        self.assertFalse(drift["compatible"])
        row = self._rows()[-1]
        self.assertFalse(row["compatible"])
        self.assertEqual(row["differences"][0]["field"], "torch")

    def test_malformed_previous_record_is_visible(self):
        (self.root / "environment.json").write_text("{not json",
                                                    encoding="utf-8")
        drift = experiments._record_environment_history(
            self.root, self._context("r2"))
        self.assertFalse(drift["compatible"])
        self.assertIn("unreadable", drift["reason"])
        self.assertIn("unreadable", self._rows()[-1]["reason"])

    def test_history_is_append_only_against_tampering(self):
        experiments._record_environment_history(self.root, self._context("r1"))
        history = self.root / "environment_history.jsonl"
        experiments._write_artifact_index(
            self.root, None, "history", completed=True, elapsed_seconds=0.0)
        experiments._RUN_ARTIFACTS = set()
        history.write_text('{"tampered": true}\n', encoding="utf-8")
        with self.assertRaises(experiments.TrackingError):
            experiments._record_environment_history(
                self.root, self._context("r2"))


class EnvironmentCheckTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.env = experiments._environment_record(
            Settings(device="cpu"), git_dirty=False)
        self.context = {
            "run_id": "r1", "schema_version": 1,
            "setup_sha256": "setup", "git_sha": "deadbeef",
            "source_tree_sha256": "source", "environment": self.env,
        }

    def tearDown(self):
        self._tmp.cleanup()

    def _check(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = experiments._check_environment(self.root)
        return code, out.getvalue()

    def test_compatible_record_passes(self):
        (self.root / "environment.json").write_text(
            json.dumps(self.context), encoding="utf-8")
        code, text = self._check()
        self.assertEqual(code, 0, text)
        self.assertIn("PASS", text)

    def test_critical_drift_fails(self):
        self.context["environment"] = dict(self.env, torch="0.0")
        (self.root / "environment.json").write_text(
            json.dumps(self.context), encoding="utf-8")
        code, text = self._check()
        self.assertEqual(code, 1, text)
        self.assertIn("torch", text)

    def test_missing_or_malformed_record_is_an_error(self):
        code, _ = self._check()
        self.assertEqual(code, 2)
        (self.root / "environment.json").write_text("{", encoding="utf-8")
        code, _ = self._check()
        self.assertEqual(code, 2)

    def test_manifest_is_accepted_as_a_fallback(self):
        (self.root / "e1_manifest.json").write_text(
            json.dumps(self.context), encoding="utf-8")
        code, text = self._check()
        self.assertEqual(code, 0, text)

    def test_main_routes_the_check_without_training(self):
        (self.root / "environment.json").write_text(
            json.dumps(self.context), encoding="utf-8")
        with mock.patch.object(experiments, "_write_result") as write:
            with contextlib.redirect_stdout(io.StringIO()):
                code = experiments.main(
                    ["--check-environment", "--out", str(self.root)])
        self.assertEqual(code, 0)
        write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
