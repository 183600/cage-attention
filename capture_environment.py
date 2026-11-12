#!/usr/bin/env python3
"""Capture the software environment of this checkout, or compare one.

This script is a thin command-line wrapper around the *canonical*
environment implementation in ``cage.experiments``
(``_environment_record`` / ``_environment_drift``).  It used to carry a
second, independent implementation of the same idea, and the two drifted
apart:

* the standalone record used a different schema and a different fingerprint
  algorithm, so its ``environment_sha256`` was not comparable with the
  ``environment_sha256`` of a results directory - a reader could capture an
  environment, compare it with the runner's record and get a verdict about
  two unrelated objects;
* it classified the *whole* installed package inventory as a critical
  difference, so any unrelated package made an environment look
  incompatible, while a missing or wrong torch was only one field among
  many.

Both facts now live in one place.  The record written here is the same
environment block the experiment runner stores in
``results/environment.json`` (inside the run context), so
``capture_environment.py --compare results/environment.json`` and
``python3 -m cage.experiments --check-environment --out results`` answer
the same question with the same answer - and both accept either shape (the
bare block or the whole context).

The record deliberately contains no timestamp: the environment fingerprint
has to be stable across invocations, otherwise every capture would look
like a critical drift to the next one.

Usage:
    python3 capture_environment.py [--output ENVIRONMENT.json]
    python3 capture_environment.py --compare ENVIRONMENT.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cage import experiments  # noqa: E402
from cage.experiments import Settings  # noqa: E402


def _default_root() -> pathlib.Path:
    """The repository root: the directory this script lives in.

    ``capture_environment.py`` sits at the top level of the checkout, so the
    repository root is its own directory.  Taking the *parent* of that
    directory would point the record at whatever surrounds the checkout:
    the git sha would be unreadable ("unavailable"), the tree would always
    look dirty, and the dependency manifest would be empty.
    """
    return ROOT


def _settings_from_record(record) -> Settings:
    """Rebuild the settings an archived record was captured under.

    The device and the determinism switch are part of the numerical
    recipe, so the comparison must rebuild the current record under the
    *recorded* switch instead of this process's defaults.
    """
    device = "cpu"
    deterministic = True
    if isinstance(record, dict):
        if isinstance(record.get("device"), str):
            device = record["device"]
        if isinstance(record.get("deterministic_requested"), bool):
            deterministic = bool(record["deterministic_requested"])
    return Settings(device=device, deterministic=deterministic)


def capture_environment(root=None) -> dict:
    """Return the canonical environment record of this checkout.

    ``root`` is accepted for backwards compatibility and must be this
    checkout: the record describes the code that will run, so it is always
    built from the package's own repository root.
    """
    if root is not None and pathlib.Path(root).resolve() != _default_root():
        raise ValueError(
            f"the environment record describes this checkout "
            f"({_default_root()}); it cannot be captured for {root!r}")
    return experiments._environment_record(Settings(device="cpu"),
                                           git_dirty=None)


def compare_environments(reference, current) -> dict:
    """Compare two environment records, returning the drift diagnosis."""
    return experiments._environment_drift(reference, current)


def _describe(drift: dict) -> str:
    """One readable line per recorded difference."""
    lines = []
    for difference in drift.get("differences", []):
        lines.append("    " + experiments._describe_environment_difference(
            difference))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture or compare the CAGE-Attention environment")
    parser.add_argument("--output", "-o",
                        help="Output JSON file path "
                             "(default: <repo>/environment_capture.json)")
    parser.add_argument("--compare", "-c", metavar="FILE",
                        help="Compare the recorded environment in FILE with "
                             "the current process")
    args = parser.parse_args(argv)

    if args.compare:
        path = pathlib.Path(args.compare)
        recorded, _payload, error = experiments._read_environment_file(path)
        if error is not None:
            print(f"[env-capture] cannot read {path}: {error}")
            return 2
        if recorded is None:
            print(f"[env-capture] {path} carries no environment block")
            return 2
        try:
            current = experiments._environment_record(
                _settings_from_record(recorded), git_dirty=None)
        except Exception as exc:  # noqa: BLE001 - a bad record is an error
            print(f"[env-capture] cannot build the current environment "
                  f"record: {type(exc).__name__}: {exc}")
            return 2
        drift = compare_environments(recorded, current)
        print(f"[env-capture] recorded={path} "
              f"recorded_env={drift.get('recorded_environment_sha256')} "
              f"current_env={drift.get('current_environment_sha256')}")
        detail = _describe(drift)
        if detail:
            print(detail)
        print(f"[env-capture] {drift.get('reason')}")
        if drift.get("compatible"):
            print("[env-capture] PASS: the critical software/numerical "
                  "fields match")
            return 0
        print("[env-capture] FAIL: the recorded environment was produced "
              "under a different software/numerical environment")
        return 1

    env = capture_environment()
    output = pathlib.Path(args.output) if args.output else (
        _default_root() / "environment_capture.json")
    output.write_text(json.dumps(env, indent=2, ensure_ascii=False,
                                 sort_keys=True), encoding="utf-8")
    print(f"Environment captured to {output}")
    print(f"Environment fingerprint: {env['environment_sha256']}")
    print(f"Repository: {env.get('repo_root')} "
          f"(git_sha={experiments._git_sha()}, dirty={env.get('git_dirty')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
