"""Single-writer lock tests for the results directory."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from cage import experiments
from cage.runlock import LOCK_FILENAME, ResultsLock, ResultsLockError


class ResultsLockTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_second_writer_is_refused_until_release(self):
        first = ResultsLock(self.root)
        first.acquire()
        self.assertEqual(first.mode, "exclusive")
        second = ResultsLock(self.root)
        with self.assertRaises(ResultsLockError):
            second.acquire()
        first.release()
        # The lock file remains on disk for diagnostics, but the advisory
        # lock is gone, so a later invocation can proceed.
        self.assertTrue((self.root / LOCK_FILENAME).exists())
        second.acquire()
        second.release()

    def test_different_directories_do_not_block_each_other(self):
        other = pathlib.Path(self._tmp.name) / "other"
        first = ResultsLock(self.root)
        second = ResultsLock(other)
        try:
            first.acquire()
            second.acquire()
        finally:
            first.release()
            second.release()

    def test_context_manager_releases_on_exception(self):
        with self.assertRaises(RuntimeError):
            with ResultsLock(self.root):
                raise RuntimeError("boom")
        with ResultsLock(self.root):
            pass

    def test_main_refuses_to_start_when_locked(self):
        lock = ResultsLock(self.root)
        lock.acquire()
        try:
            with mock.patch.object(experiments, "run_sanity") as sanity:
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    code = experiments.main(
                        ["--only", "sanity", "--quick",
                         "--out", str(self.root)])
            self.assertEqual(code, 4)
            self.assertIn("already locked", out.getvalue())
            sanity.assert_not_called()
            # The refused invocation must not start the evidence chain.
            self.assertFalse((self.root / "run_index.jsonl").exists())
            self.assertFalse((self.root / "environment.json").exists())
        finally:
            lock.release()

    def test_lock_mode_is_recorded_on_start(self):
        # A real (tiny) run is unnecessary: the started event carries the
        # mode.  Patch the expensive pre-flight so the lifecycle path runs.
        ran = {}

        def fake_sanity(settings, out_dir):
            ran["out_dir"] = out_dir

        with mock.patch.object(experiments, "run_sanity", fake_sanity):
            with contextlib.redirect_stdout(io.StringIO()):
                code = experiments.main(
                    ["--only", "sanity", "--quick",
                     "--out", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("out_dir", ran)
        rows = [
            json.loads(line)
            for line in (self.root / "run_index.jsonl").read_text(
                encoding="utf-8").splitlines()
        ]
        started = [row for row in rows if row.get("event") == "started"]
        self.assertEqual(len(started), 1)
        self.assertIn(started[0].get("lock_mode"), {"exclusive",
                                                    "unavailable"})


if __name__ == "__main__":
    unittest.main()
