"""CLI argument validation: a malformed argument must be refused cleanly.

The experiment CLI validates its recipe switches before it touches the
results directory (``--steps``, ``--d-model``, ``--lr``, ...).  An
unparsable ``--device`` used to escape as a raw torch ``RuntimeError``
from deep inside the first training step, so the operator saw a
traceback instead of the usual refusal.
"""

import contextlib
import io
import unittest

from cage import experiments
# Reproducibility: Set random seeds
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass


class CliArgumentValidationTest(unittest.TestCase):

    def _run(self, argv):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = experiments.main(argv)
        return code, out.getvalue()

    def test_invalid_device_is_refused(self):
        code, text = self._run(
            ["--device", "not-a-device", "--only", "sanity", "--quick"])
        self.assertEqual(code, 2, text)
        self.assertIn("--device", text)

    def test_valid_device_is_accepted(self):
        # The refusal must not fire for a device string torch accepts: the
        # run still fails later (the manifest does not exist), but for a
        # different reason.
        code, text = self._run(
            ["--device", "cpu", "--replay-manifest", "/nonexistent/manifest"])
        self.assertEqual(code, 2, text)
        self.assertNotIn("--device must be", text)
        self.assertIn("REPLAY REJECTED", text)

    def test_other_recipe_switches_still_refuse_cleanly(self):
        for argv, needle in (
            (["--steps", "0"], "--steps"),
            (["--d-model", "0"], "--d-model"),
            (["--n-layers", "0"], "--n-layers"),
            (["--lr", "-1.0"], "--lr"),
            (["--seeds", "0"], "--seeds"),
            (["--fdr-q", "1.5"], "--fdr-q"),
        ):
            with self.subTest(argv=argv):
                code, text = self._run(["--only", "sanity", "--quick", *argv])
                self.assertEqual(code, 2, text)
                self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
