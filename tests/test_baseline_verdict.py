"""The baseline-validity gate must fail closed on malformed verdict marks."""

from __future__ import annotations

import contextlib
import io
import pathlib
import tempfile
import unittest
from unittest import mock

from cage import experiments
from cage.experiments import (
    BaselineVerdictError,
    Settings,
    _enforce_baseline_verdict,
)


class BaselineVerdictGateTest(unittest.TestCase):
    def _run(self, verdict):
        settings = Settings.quick()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(
                    experiments, "_load_baseline_verdict",
                    return_value=verdict), \
                mock.patch.object(
                    experiments, "_verdict_provenance_note",
                    return_value=None), \
                mock.patch.object(
                    experiments, "_run_context", return_value={}), \
                contextlib.redirect_stdout(io.StringIO()):
            return _enforce_baseline_verdict(
                settings, pathlib.Path(tmp), "e1")

    def test_truthy_string_is_not_a_passing_blocking_gate(self):
        verdict = {
            "hb1_supported": "False", "hb3_supported": True,
            "hb5_supported": True, "hb4_state": "inconclusive",
            "hb4_supported": False,
        }
        with self.assertRaises(BaselineVerdictError):
            self._run(verdict)

    def test_supported_hb4_state_requires_the_boolean_mark(self):
        verdict = {
            "hb1_supported": True, "hb3_supported": True,
            "hb5_supported": True, "hb4_state": "supported",
            "hb4_supported": "True",
        }
        with self.assertRaises(BaselineVerdictError):
            self._run(verdict)

    def test_well_formed_verdict_passes(self):
        verdict = {
            "hb1_supported": True, "hb3_supported": True,
            "hb5_supported": True, "hb4_state": "inconclusive",
            "hb4_supported": False,
        }
        loaded = self._run(verdict)
        self.assertIs(loaded, verdict)


if __name__ == "__main__":
    unittest.main()
