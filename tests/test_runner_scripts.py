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
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest

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
        self.assertEqual(pathlib.Path(env["git"]["repo_root"]).resolve(),
                         ROOT.resolve())
        self.assertNotEqual(env["git"]["git_sha"], "unavailable")
        self.assertEqual(len(env["git"]["git_sha"]), 40)
        for rel in ("pyproject.toml", "requirements-ci.txt", "README.md",
                    "docs/design_notes.md"):
            self.assertIn(rel, env["dependency_manifest"])

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



if __name__ == "__main__":
    unittest.main()
