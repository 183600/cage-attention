"""Single-writer lock for a results directory.

The results directory is an append-only evidence store: JSONL result rows
are appended, the cumulative ``artifact_index.json`` is rewritten at the end
of every invocation, and the environment history is a chain.  Those
guarantees assume **one writer at a time**.  Two concurrent invocations
would interleave rows, race on ``environment.json`` and let the last index
write overwrite entries the other invocation just certified.

This module provides a small advisory lock around the directory.  It is an
operational guard, not a cryptographic one: it prevents two cooperating
processes (the normal case: a user starts ``make sanity`` in one terminal
while another command is still running) from corrupting the evidence chain.

On platforms without ``fcntl`` the lock degrades to ``mode="unavailable"``
and the runner reports that fact rather than pretending the directory is
protected.  The lock file is a dotfile and is deliberately *not* registered
as a result artifact: it carries no scientific evidence, only the identity
of the last holder for diagnostics.  Leaving it behind after a crash is
harmless because the OS releases the advisory lock with the process.
"""

from __future__ import annotations

import os
import pathlib
import platform
import time

try:  # POSIX advisory locks; Windows runners simply report "unavailable".
    import fcntl
except ImportError:  # pragma: no cover - exercised on non-POSIX platforms
    fcntl = None


LOCK_FILENAME = ".results.lock"
LOCK_EXIT_CODE = 4


class ResultsLockError(RuntimeError):
    """Another process currently owns the results directory."""


class ResultsLock:
    """Exclusive, non-blocking advisory lock for one results directory.

    ``acquire`` fails fast instead of waiting: a long-running grid can hold
    the directory for hours, and silently queueing a second invocation
    behind it would hide the conflict.  The caller decides whether to abort
    (the CLI does, with :data:`LOCK_EXIT_CODE`) or retry.
    """

    def __init__(self, out_dir):
        self.directory = pathlib.Path(out_dir)
        self.path = self.directory / LOCK_FILENAME
        self.mode = "unavailable"
        self._handle = None

    def _holder_text(self) -> str:
        """Last-holder description from the lock file, if readable."""
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return " ".join(text.split())

    def _write_holder(self) -> None:
        """Record who holds the lock (diagnostics only)."""
        if self._handle is None:
            return
        holder = (
            f"pid={os.getpid()} host={platform.node() or 'unknown'} "
            f"acquired_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
        )
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(holder + "\n")
        self._handle.flush()

    def acquire(self) -> "ResultsLock":
        """Take the lock or raise :class:`ResultsLockError`."""
        self.directory.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        if fcntl is None:
            # No advisory locking primitive: keep the handle for a stable
            # API but tell the caller the guarantee is unavailable.
            self._handle = handle
            self.mode = "unavailable"
            return self
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            holder = self._holder_text()
            handle.close()
            detail = f" (holder: {holder})" if holder else ""
            raise ResultsLockError(
                f"results directory {self.directory.resolve()} is already "
                f"locked by another process{detail}; concurrent writers "
                f"would interleave JSONL records and could drop artifact-index "
                f"entries, so this invocation does not start"
            ) from exc
        except BaseException:  # pragma: no cover - defensive close
            handle.close()
            raise
        self._handle = handle
        self.mode = "exclusive"
        try:
            self._write_holder()
        except BaseException:  # pragma: no cover - never leak the lock
            self.release()
            raise
        return self

    def release(self) -> None:
        """Release the lock; safe to call more than once."""
        handle, self._handle = self._handle, None
        if handle is None:
            return
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:  # pragma: no cover - closing releases it anyway
                pass
        try:
            handle.close()
        except OSError:  # pragma: no cover - nothing useful to retry
            pass

    def __enter__(self) -> "ResultsLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
