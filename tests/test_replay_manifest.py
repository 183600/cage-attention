"""Executable-manifest replay tests.

A run manifest is only useful for reproducibility when it can be turned back
into the exact frozen recipe and re-run under the recorded code, hypotheses
and data streams.  These tests cover that contract and the CLI guard rails
that keep a replay from silently becoming a different experiment.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from cage import experiments
from cage.experiments import Settings


class SourceFingerprintPathTest(unittest.TestCase):
    def _tree(self, root: pathlib.Path, layout) -> str:
        for relative, text in layout:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return experiments._source_tree_sha256(root)

    def test_same_basenames_in_different_directories_are_distinct(self):
        with tempfile.TemporaryDirectory() as first_dir, \
                tempfile.TemporaryDirectory() as second_dir:
            first = self._tree(
                pathlib.Path(first_dir),
                (("a/mod.py", "A"), ("b/helper.py", "B")))
            second = self._tree(
                pathlib.Path(second_dir),
                (("c/mod.py", "A"), ("d/helper.py", "B")))
        # The old fingerprint hashed only basenames + contents, so moving
        # both modules to different subpackages produced the same id.
        self.assertNotEqual(first, second)

    def test_production_fingerprint_is_cached(self):
        first = experiments._source_tree_sha256()
        second = experiments._source_tree_sha256()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)


class ReplayManifestTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self._saved_artifacts = experiments._RUN_ARTIFACTS
        experiments._RUN_ARTIFACTS = set()
        self.settings = Settings.quick("cpu")
        self.manifest = experiments._write_manifest(
            self.settings, self.source, "sanity")
        self.path = self.source / "sanity_manifest.json"

    def tearDown(self):
        experiments._RUN_ARTIFACTS = self._saved_artifacts
        self._tmp.cleanup()

    def _mutated(self, mutate) -> pathlib.Path:
        payload = json.loads(json.dumps(self.manifest))
        mutate(payload)
        path = self.source / "mutated_manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_round_trip_recovers_the_frozen_recipe(self):
        settings, manifest, drift, digest = (
            experiments._load_replay_manifest(self.path))
        self.assertEqual(manifest["experiment"], "sanity")
        self.assertEqual(settings.seeds, self.settings.seeds)
        self.assertEqual(settings.steps, self.settings.steps)
        self.assertEqual(settings.n_layers, self.settings.n_layers)
        self.assertTrue(drift["compatible"])
        self.assertTrue(drift["identical"])
        self.assertEqual(len(digest), 64)

    def test_source_mismatch_is_refused(self):
        path = self._mutated(
            lambda payload: payload.__setitem__(
                "source_tree_sha256", "0" * 16))
        with self.assertRaises(experiments.ReplayManifestError):
            experiments._load_replay_manifest(path)

    def test_hypothesis_registry_mismatch_is_refused(self):
        path = self._mutated(
            lambda payload: payload.__setitem__(
                "hypotheses_sha256", "0" * 16))
        with self.assertRaises(experiments.ReplayManifestError):
            experiments._load_replay_manifest(path)

    def test_data_stream_mismatch_is_refused(self):
        def mutate(payload):
            fingerprints = dict(payload["data_fingerprints"])
            fingerprints["val"] = "0" * 16
            payload["data_fingerprints"] = fingerprints

        with self.assertRaises(experiments.ReplayManifestError):
            experiments._load_replay_manifest(self._mutated(mutate))

    def test_partial_or_extended_config_is_refused(self):
        def missing(payload):
            config = dict(payload["config"])
            config.pop("steps")
            payload["config"] = config

        def unknown(payload):
            config = dict(payload["config"])
            config["not_a_setting"] = 1
            payload["config"] = config

        for mutate in (missing, unknown):
            with self.subTest(mutate=mutate.__name__):
                with self.assertRaises(experiments.ReplayManifestError):
                    experiments._load_replay_manifest(self._mutated(mutate))

    def test_unknown_experiment_is_refused(self):
        path = self._mutated(
            lambda payload: payload.__setitem__("experiment", "e9"))
        with self.assertRaises(experiments.ReplayManifestError):
            experiments._load_replay_manifest(path)

    def test_edited_recipe_is_refused_by_its_own_setup_id(self):
        def mutate(payload):
            config = dict(payload["config"])
            config["steps"] = int(config["steps"]) + 1
            payload["config"] = config

        path = self._mutated(mutate)
        with self.assertRaises(experiments.ReplayManifestError) as raised:
            experiments._load_replay_manifest(path)
        self.assertIn("setup_sha256", str(raised.exception))

    def test_environment_drift_requires_an_explicit_override(self):
        def mutate(payload):
            environment = dict(payload["environment"])
            environment["torch"] = "0.0.0"
            payload["environment"] = environment

        path = self._mutated(mutate)
        with self.assertRaises(experiments.ReplayManifestError):
            experiments._load_replay_manifest(path)
        _, _, drift, _ = experiments._load_replay_manifest(
            path, allow_environment_drift=True)
        self.assertFalse(drift["compatible"])
        self.assertIn("torch", [d["field"] for d in
                                drift["critical_differences"]])

    def test_missing_or_malformed_manifest_is_refused(self):
        missing = self.source / "does-not-exist.json"
        malformed = self.source / "malformed.json"
        malformed.write_text("{not json", encoding="utf-8")
        not_object = self.source / "array.json"
        not_object.write_text("[]", encoding="utf-8")
        for path in (missing, malformed, not_object):
            with self.subTest(path=path.name):
                with self.assertRaises(experiments.ReplayManifestError):
                    experiments._load_replay_manifest(path)


class ReplayCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self._saved_artifacts = experiments._RUN_ARTIFACTS
        experiments._RUN_ARTIFACTS = set()
        self.settings = Settings.quick("cpu")
        self.manifest = experiments._write_manifest(
            self.settings, self.source, "sanity")
        self.path = self.source / "sanity_manifest.json"

    def tearDown(self):
        experiments._RUN_ARTIFACTS = self._saved_artifacts
        self._tmp.cleanup()

    def _main(self, argv):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = experiments.main(argv)
        return code, out.getvalue()

    def test_replay_runs_the_manifest_experiment_in_a_new_directory(self):
        out = self.root / "replay"
        with mock.patch.object(experiments, "run_sanity") as run:
            code, text = self._main([
                "--replay-manifest", str(self.path), "--out", str(out)])
        self.assertEqual(code, 0, text)
        run.assert_called_once()
        called_settings, called_out = run.call_args.args
        self.assertEqual(called_settings.seeds, self.settings.seeds)
        self.assertEqual(called_settings.steps, self.settings.steps)
        self.assertEqual(pathlib.Path(called_out), out)
        replay = json.loads((out / "replay.json").read_text(
            encoding="utf-8"))
        environment = json.loads((out / "environment.json").read_text(
            encoding="utf-8"))
        self.assertEqual(replay["source_run_id"], self.manifest["run_id"])
        self.assertEqual(replay["replay_run_id"], environment["run_id"])
        self.assertEqual(replay["manifest_sha256"],
                         hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertFalse(replay["environment_override"])
        self.assertTrue((out / "artifact_index.json").is_file())

    def test_environment_drift_override_is_recorded_in_replay_json(self):
        payload = json.loads(json.dumps(self.manifest))
        payload["environment"]["torch"] = "0.0.0"
        path = self.source / "drifted.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        out = self.root / "replay"
        with mock.patch.object(experiments, "run_sanity"):
            code, text = self._main([
                "--replay-manifest", str(path),
                "--allow-environment-drift", "--out", str(out)])
        self.assertEqual(code, 0, text)
        replay = json.loads((out / "replay.json").read_text(
            encoding="utf-8"))
        self.assertTrue(replay["environment_override"])
        self.assertFalse(replay["environment_compatible"])

    def test_replay_refuses_to_append_to_the_source_directory(self):
        code, text = self._main([
            "--replay-manifest", str(self.path), "--out", str(self.source)])
        self.assertEqual(code, 2)
        self.assertIn("must differ", text)

    def test_replay_refuses_recipe_overrides(self):
        for flag, value in (("--seeds", "3"), ("--steps", "7"),
                            ("--lr", "0.001"), ("--fdr-q", "0.1")):
            with self.subTest(flag=flag):
                code, text = self._main([
                    "--replay-manifest", str(self.path), flag, value,
                    "--out", str(self.root / f"out{flag[2:]}")])
                self.assertEqual(code, 2)
                self.assertIn("cannot be overridden", text)

    def test_replay_cannot_be_combined_with_a_verify_action(self):
        code, text = self._main([
            "--replay-manifest", str(self.path), "--verify-artifacts",
            "--out", str(self.root / "replay")])
        self.assertEqual(code, 2)
        self.assertIn("cannot be combined", text)

    def test_environment_override_requires_replay(self):
        code, text = self._main([
            "--allow-environment-drift", "--out", str(self.root / "out")])
        self.assertEqual(code, 2)
        self.assertIn("only applies", text)

    def test_default_output_names_the_source_run(self):
        previous = os.getcwd()
        work = self.root / "work"
        work.mkdir()
        try:
            os.chdir(work)
            with mock.patch.object(experiments, "run_sanity"):
                code, text = self._main([
                    "--replay-manifest", str(self.path)])
        finally:
            os.chdir(previous)
        self.assertEqual(code, 0, text)
        expected = work / f"results_replay_{self.manifest['run_id']}"
        self.assertTrue((expected / "replay.json").is_file())

    def test_unsafe_source_run_id_cannot_escape_the_default_output(self):
        payload = json.loads(json.dumps(self.manifest))
        payload["run_id"] = "../../escape"
        path = self.source / "unsafe.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        previous = os.getcwd()
        work = self.root / "work"
        work.mkdir()
        try:
            os.chdir(work)
            with mock.patch.object(experiments, "run_sanity"):
                code, text = self._main([
                    "--replay-manifest", str(path)])
        finally:
            os.chdir(previous)
        self.assertEqual(code, 0, text)
        self.assertTrue(
            (work / "results_replay_unknown" / "replay.json").is_file())


if __name__ == "__main__":
    unittest.main()
