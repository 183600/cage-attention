"""Provenance-record tests: the environment block must stay rebuildable."""

from __future__ import annotations

import unittest
from unittest import mock

from cage import experiments
from cage.experiments import Settings, _environment_record


class EnvironmentRecordTest(unittest.TestCase):
    def test_environment_pins_numerical_and_build_identity(self):
        record = _environment_record(Settings(device="cpu"), git_dirty=False)
        for key in (
            "python", "python_implementation", "python_compiler",
            "torch", "torch_git_version", "platform", "machine",
            "os_release", "cpu_count", "cpu_model", "torch_num_threads",
            "torch_num_interop_threads", "deterministic_algorithms",
            "cublas_workspace_config", "tf32", "numeric_env",
            "pythonhashseed", "hash_randomization", "argv", "cwd",
            "repo_root", "git_dirty",
        ):
            self.assertIn(key, record)
        self.assertFalse(record["git_dirty"])
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "CUBLAS_WORKSPACE_CONFIG"):
            self.assertIn(key, record["numeric_env"])


class GitProvenanceTest(unittest.TestCase):
    def test_git_probes_are_anchored_at_the_package_root(self):
        completed = mock.Mock(stdout="abc123\n", returncode=0)
        with mock.patch.object(experiments.subprocess, "run",
                               return_value=completed) as run:
            self.assertEqual(experiments._git_sha(), "abc123")
            self.assertEqual(run.call_args.kwargs["cwd"],
                             str(experiments._REPO_ROOT))
        with mock.patch.object(
                experiments.subprocess, "run",
                return_value=mock.Mock(stdout="")) as run:
            self.assertFalse(experiments._git_dirty())
            self.assertEqual(run.call_args.kwargs["cwd"],
                             str(experiments._REPO_ROOT))

    def test_git_dirty_fails_closed(self):
        with mock.patch.object(experiments.subprocess, "run",
                               side_effect=OSError("git missing")):
            self.assertEqual(experiments._git_sha(), "unknown")
            self.assertTrue(experiments._git_dirty())


if __name__ == "__main__":
    unittest.main()
