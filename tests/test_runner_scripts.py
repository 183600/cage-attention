"""Regression tests for the top-level runner scripts.

``run_experiment.py`` and ``capture_environment.py`` are the entry points an
operator uses to produce the evidence directory, so a silent no-op in
either one is a provenance bug, not a cosmetic one:

* the baseline gate read a verdict field that does not exist, so *every*
  run aborted with "Baseline gates failed" even when HB1/HB3/HB4/HB5 passed;
* ``--replay`` was documented but never forwarded, so the runner silently
  produced a fresh run instead of replaying the frozen manifest;
* the environment capture defaulted to the *parent* of the checkout, so the
  recorded git sha was unreadable, the tree always looked dirty and the
  dependency manifest was empty;
* the source-tree stamp hashed the current working directory instead of the
  repository.
"""

import argparse
import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import capture_environment  # noqa: E402
import run_experiment  # noqa: E402


def _passing_verdict() -> dict:
    """A verdict whose blocking baseline gates all passed."""
    return {
        "hb1_supported": True,
        "hb2_state": "certified",
        "hb3_supported": True,
        "hb4_state": "supported",
        "hb4_supported": True,
        "hb4_violated": False,
        "hb5_supported": True,
        "hb6_supported": True,
        "hb7_depth_required": False,
    }


def _runner(out_dir, **overrides):
    args = argparse.Namespace(
        out=str(out_dir), seeds=3, experiment="e1", deterministic=False,
        fdr_q=0.05, quick=True, verify=False, skip_gate=True,
        skip_tests=True, skip_baselines=False, replay=None)
    for key, value in overrides.items():
        setattr(args, key, value)
    return run_experiment.ExperimentRunner(args)


class CaptureEnvironmentTest(unittest.TestCase):
    """The environment record must describe this checkout."""

    def test_default_root_is_the_repository(self):
        self.assertEqual(pathlib.Path(capture_environment._default_root()),
                         ROOT.resolve())

    def test_capture_records_the_real_git_state_and_dependencies(self):
        env = capture_environment.capture_environment()
        self.assertEqual(pathlib.Path(env["repo_root"]).resolve(),
                         ROOT.resolve())
        # The standalone capture must produce the *same* record as the
        # experiment runner: one environment implementation, one
        # fingerprint.  A second, divergent implementation made the two
        # ``environment_sha256`` values incomparable.
        from cage.experiments import Settings
        from cage import experiments
        canonical = experiments._environment_record(
            Settings(device="cpu"), git_dirty=None)
        self.assertEqual(env["environment_sha256"],
                         canonical["environment_sha256"])
        self.assertEqual(len(experiments._git_sha()), 40)
        for rel in ("pyproject.toml", "requirements-ci.txt"):
            self.assertIn(rel, env["dependency_manifest"])

    def test_capture_refuses_a_foreign_root(self):
        with self.assertRaises(ValueError):
            capture_environment.capture_environment(root="/tmp")

    def test_compare_reports_the_same_verdict_as_the_package_check(self):
        env = capture_environment.capture_environment()
        drift = capture_environment.compare_environments(env, dict(env))
        self.assertTrue(drift["compatible"])
        self.assertTrue(drift["identical"])

class SourceTreeStampTest(unittest.TestCase):
    """The provenance stamps must describe the repository, not the CWD."""

    def test_source_tree_hash_ignores_the_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(pathlib.Path(tmp))
            here = runner._get_source_tree_sha256()
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                elsewhere = runner._get_source_tree_sha256()
            finally:
                os.chdir(cwd)
            self.assertEqual(here, elsewhere)
            self.assertEqual(len(here), 16)

    def test_git_sha_is_read_from_the_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(pathlib.Path(tmp))
            sha = runner._get_git_sha()
            self.assertNotEqual(sha, "unknown")
            expected = subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(sha, expected)

class BaselineVerdictTest(unittest.TestCase):
    """``run_baselines`` must accept a passing verdict and reject a
    failing one, using the real per-gate marks."""

    def test_passing_verdict_has_no_failures(self):
        self.assertEqual(
            run_experiment.baseline_verdict_failures(_passing_verdict()), [])

    def test_failing_gates_are_reported(self):
        verdict = _passing_verdict()
        verdict["hb3_supported"] = False
        verdict["hb4_state"] = "violated"
        failures = run_experiment.baseline_verdict_failures(verdict)
        self.assertTrue(any("hb3_supported" in f for f in failures))
        self.assertTrue(any("HB4" in f for f in failures))

    def test_unknown_hb4_state_fails_closed(self):
        verdict = _passing_verdict()
        verdict["hb4_state"] = "banana"
        self.assertTrue(
            run_experiment.baseline_verdict_failures(verdict))

    def test_inconclusive_hb4_does_not_block(self):
        verdict = _passing_verdict()
        verdict["hb4_state"] = "inconclusive"
        verdict["hb4_supported"] = False
        self.assertEqual(
            run_experiment.baseline_verdict_failures(verdict), [])

    def test_non_dict_verdict_fails_closed(self):
        self.assertTrue(run_experiment.baseline_verdict_failures([]))

    def test_run_baselines_accepts_passing_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            (out / "baselines_verdict.json").write_text(
                json.dumps(_passing_verdict()), encoding="utf-8")
            runner = _runner(out)
            calls = []

            def fake_run_command(cmd, capture=True, check=True, env=None):
                calls.append(cmd)
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            runner.run_command = fake_run_command
            self.assertTrue(runner.run_baselines())
            self.assertEqual(len(calls), 1)

    def test_run_baselines_rejects_failing_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            verdict = _passing_verdict()
            verdict["hb5_supported"] = False
            (out / "baselines_verdict.json").write_text(
                json.dumps(verdict), encoding="utf-8")
            runner = _runner(out)
            runner.run_command = lambda *a, **k: types.SimpleNamespace(
                returncode=0, stdout="", stderr="")
            self.assertFalse(runner.run_baselines())

    def test_run_baselines_rejects_missing_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(pathlib.Path(tmp))
            runner.run_command = lambda *a, **k: types.SimpleNamespace(
                returncode=0, stdout="", stderr="")
            self.assertFalse(runner.run_baselines())

class ReplayOptionTest(unittest.TestCase):
    """``--replay`` must reach the experiment CLI."""

    def test_replay_is_forwarded_to_the_experiment_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = pathlib.Path(tmp) / "e1_manifest.json"
            manifest.write_text(json.dumps({"experiment": "e1"}),
                                encoding="utf-8")
            out = pathlib.Path(tmp) / "replay_out"
            runner = _runner(out, replay=str(manifest))
            calls = []

            def fake_run_command(cmd, capture=True, check=True, env=None):
                calls.append(cmd)
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            runner.run_command = fake_run_command
            self.assertTrue(runner.run_replay())
            self.assertEqual(len(calls), 1)
            cmd = calls[0]
            self.assertIn("--replay-manifest", cmd)
            self.assertIn(str(manifest), cmd)
            self.assertIn(str(out), cmd)
            # The manifest freezes the recipe: the runner's own recipe
            # switches must not be forwarded (the experiment CLI rejects
            # them in replay mode).
            for forbidden in ("--seeds", "--quick", "--fdr-q", "--deterministic"):
                self.assertNotIn(forbidden, cmd)
            self.assertEqual(runner.replayed_experiment, "e1")

    def test_replay_mode_skips_the_baseline_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = pathlib.Path(tmp) / "e1_manifest.json"
            manifest.write_text(json.dumps({"experiment": "e1"}),
                                encoding="utf-8")
            out = pathlib.Path(tmp) / "replay_out"
            runner = _runner(out, replay=str(manifest))
            seen = []
            runner.capture_environment = lambda: {}
            runner.run_dependency_check = lambda: True
            runner.run_gate_checks = lambda: True
            runner.run_unit_tests = lambda: True
            runner.run_baselines = lambda: seen.append("baselines") or True
            runner.run_replay = lambda: seen.append("replay") or True
            runner.run_experiment = lambda exp: seen.append(exp) or True
            runner.run_verification = lambda: True
            runner.generate_summary_report = lambda: {}
            runner.create_replication_package = lambda: pathlib.Path(tmp)
            runner.run_command = lambda *a, **k: types.SimpleNamespace(
                returncode=0, stdout="", stderr="")
            runner.run()
            self.assertNotIn("baselines", seen)
            self.assertIn("replay", seen)

    def test_missing_replay_manifest_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(pathlib.Path(tmp) / "out",
                             replay=str(pathlib.Path(tmp) / "nope.json"))
            runner.run_command = lambda *a, **k: types.SimpleNamespace(
                returncode=0, stdout="", stderr="")
            self.assertFalse(runner.run_replay())



class SkipBaselinesTest(unittest.TestCase):
    """``--skip-baselines`` must be honoured for every experiment choice."""

    def _run(self, tmp, experiment, skip_baselines):
        runner = _runner(pathlib.Path(tmp), experiment=experiment,
                         skip_baselines=skip_baselines)
        seen = []
        runner.capture_environment = lambda: {}
        runner.run_dependency_check = lambda: True
        runner.run_gate_checks = lambda: True
        runner.run_unit_tests = lambda: True
        runner.run_baselines = lambda: seen.append("baselines") or True
        runner.run_replay = lambda: seen.append("replay") or True
        runner.run_experiment = lambda exp: seen.append(exp) or True
        runner.run_verification = lambda: True
        runner.generate_summary_report = lambda: {}
        runner.create_replication_package = lambda: pathlib.Path(tmp)
        runner.run_command = lambda *a, **k: types.SimpleNamespace(
            returncode=0, stdout="", stderr="")
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run()
        return seen

    def test_skip_baselines_is_honoured_for_every_experiment(self):
        for experiment in ("all", "baselines", "e1", "e2", "e3"):
            with tempfile.TemporaryDirectory() as tmp:
                seen = self._run(tmp, experiment, True)
                self.assertNotIn("baselines", seen, experiment)

    def test_baselines_run_when_they_are_not_skipped(self):
        for experiment in ("all", "baselines", "e1"):
            with tempfile.TemporaryDirectory() as tmp:
                seen = self._run(tmp, experiment, False)
                self.assertIn("baselines", seen, experiment)


class SummaryReportTest(unittest.TestCase):
    """The summary report must carry the hypothesis-level synthesis."""

    def test_synthesis_is_read_from_the_contrast_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            (out / "e1_manifest.json").write_text(
                json.dumps({"experiment": "e1"}), encoding="utf-8")
            (out / "e1_contrasts.json").write_text(
                json.dumps({"hypothesis_synthesis": {
                    "H1a": {"claim_level_status": "supported"}}}),
                encoding="utf-8")
            runner = _runner(out)
            with contextlib.redirect_stdout(io.StringIO()):
                report = runner.generate_summary_report()
            self.assertIn("e1", report["experiments"])
            self.assertEqual(
                report["experiments"]["e1"]["synthesis"]["H1a"][
                    "claim_level_status"], "supported")

    def test_a_dedicated_synthesis_file_still_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            (out / "e1_manifest.json").write_text(
                json.dumps({"experiment": "e1"}), encoding="utf-8")
            (out / "e1_contrasts.json").write_text(
                json.dumps({"hypothesis_synthesis": {"H1a": "from-log"}}),
                encoding="utf-8")
            (out / "e1_synthesis.json").write_text(
                json.dumps({"H1a": "from-file"}), encoding="utf-8")
            runner = _runner(out)
            with contextlib.redirect_stdout(io.StringIO()):
                report = runner.generate_summary_report()
            self.assertEqual(report["experiments"]["e1"]["synthesis"]["H1a"],
                             "from-file")


class WorkingDirectoryTest(unittest.TestCase):
    """The runner must work when it is started outside the repository.

    ``run_experiment.py`` is documented as ``python3 /path/to/repo/
    run_experiment.py ...``, but it invoked its helpers by relative path,
    so the pre-flight environment capture died with "can't open file
    '/tmp/capture_environment.py'" before anything was recorded.
    """

    def test_subprocesses_run_in_the_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(pathlib.Path(tmp))
            with mock.patch.object(run_experiment.subprocess, "run") as run:
                run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr="")
                with contextlib.redirect_stdout(io.StringIO()):
                    runner.run_command([sys.executable, "-c", "pass"])
            kwargs = run.call_args.kwargs
            self.assertEqual(pathlib.Path(kwargs["cwd"]).resolve(),
                             ROOT.resolve())
            entries = kwargs["env"]["PYTHONPATH"].split(os.pathsep)
            self.assertIn(str(ROOT.resolve()), entries)

    def test_helper_scripts_are_addressed_by_repository_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            runner = _runner(out)
            calls = []

            def fake_run_command(cmd, capture=True, check=True, env=None):
                calls.append(cmd)
                if cmd[1].endswith("capture_environment.py"):
                    target = cmd[cmd.index("--output") + 1]
                    pathlib.Path(target).write_text(
                        json.dumps({"environment_sha256": "x"}),
                        encoding="utf-8")
                return types.SimpleNamespace(returncode=0, stdout="",
                                             stderr="")

            runner.run_command = fake_run_command
            with contextlib.redirect_stdout(io.StringIO()):
                runner.capture_environment()
                runner.run_dependency_check()
                runner.run_gate_checks()
                runner.run_unit_tests()
            # The helper scripts are addressed by absolute path, whatever
            # the caller's working directory is.  The calls are located by
            # what they run, not by position: the pre-flight steps may grow.
            scripts = [pathlib.Path(call[1]).resolve() for call in calls
                       if len(call) > 1 and call[1].endswith(".py")]
            self.assertIn(ROOT.resolve() / "capture_environment.py", scripts)
            self.assertIn(ROOT.resolve() / "ci_gate.py", scripts)
            self.assertTrue(any(str(ROOT.resolve() / "tests") in call
                                for call in calls))


if __name__ == "__main__":
    unittest.main()
