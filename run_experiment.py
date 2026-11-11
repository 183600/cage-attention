#!/usr/bin/env python3
"""Automated experiment execution with full provenance tracking.

This script orchestrates the complete experiment pipeline with:
- Pre-flight environment capture and validation
- Baseline gates (HB0-HB7) enforcement
- Experiment execution with randomised complete-block design
- Post-run verification (artifacts, provenance, runs, environment, source)
- Replication package generation
- Summary report generation

Usage:
    python3 run_experiment.py --experiment e1 --seeds 5 --out results
    python3 run_experiment.py --experiment all --seeds 5 --out results --verify
    python3 run_experiment.py --replay results/e1_manifest.json --out results_replay
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any
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


# The repository this runner belongs to.  Provenance stamps (git sha, source
# tree hash) must describe *this* checkout, not whatever directory the
# runner happened to be started from.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent


# The baseline gates whose failure invalidates every grid result.  These
# are the same marks ``cage.experiments._gates._enforce_baseline_verdict``
# blocks on (HB1 task construction, HB3 split disjointness, HB5 training
# loop); HB2 and HB6/HB7 are stamped but do not block the grid, and HB4 is
# three-state (only a *violated* negative control blocks).
_BLOCKING_BASELINE_GATES = ("hb1_supported", "hb3_supported",
                            "hb5_supported")


def baseline_verdict_failures(verdict: Any) -> list[str]:
    """The blocking baseline gates that did not pass in ``verdict``.

    ``baselines_verdict.json`` has no single "overall" field: it carries one
    mark per pre-registered gate (HB1..HB7).  Reading a non-existent
    aggregate key would report every *passing* baseline run as a failure and
    abort the whole pipeline, so the marks are evaluated here with the
    pre-registered semantics of the canonical gate in
    ``cage.experiments._gates``:

    * HB1 / HB3 / HB5 block when they are not certified (a literal
      ``True`` - a missing or hand-edited truthy string is not a pass);
    * HB4 is three-state: only a *violated* control (or an unusable /
      malformed state) blocks, while an inconclusive interval is stamped
      rather than read as leakage.  The state and its ``hb4_supported`` /
      ``hb4_violated`` booleans must also agree, mirroring
      ``_gates._enforce_baseline_verdict`` so this pre-check and the
      experiment CLI's own gate cannot disagree about the same verdict.
    """
    if not isinstance(verdict, dict):
        return ["baselines_verdict.json is not a verdict record"]
    failures = [key for key in _BLOCKING_BASELINE_GATES
                if verdict.get(key) is not True]
    state = verdict.get("hb4_state")
    if state is None:
        # Legacy verdict without the three-state field: the supported
        # boolean is the only signal available.
        if verdict.get("hb4_supported") is not True:
            failures.append("HB4 (missing/not certified)")
    elif state == "violated":
        failures.append("HB4 (violated: control interval above tolerance)")
    elif state == "untestable":
        failures.append("HB4 (untestable)")
    elif state not in ("supported", "inconclusive"):
        # An unknown state must fail closed: treating it as a pass would
        # let a hand-edited verdict bypass the negative-control gate.
        failures.append(f"HB4 (unrecognised state {state!r})")
    elif state == "supported" and verdict.get("hb4_supported") is not True:
        failures.append(
            "HB4 (state says supported but the supported boolean is not true)")
    # The state and its two booleans must agree, exactly as the canonical
    # ``_gates._enforce_baseline_verdict`` requires: a hand-edited verdict
    # could otherwise pair an inconclusive state with ``hb4_violated=True``
    # (or a supported state with a truthy but non-boolean mark) and pass this
    # pre-check while the experiment CLI later refuses to train.
    for key in ("hb4_supported", "hb4_violated"):
        value = verdict.get(key)
        if value is not None and not isinstance(value, bool):
            failures.append(
                f"HB4 ({key} malformed: expected a boolean, got {value!r})")
    if state is None and verdict.get("hb4_violated") is True:
        failures.append(
            "HB4 (legacy verdict has no state but hb4_violated is true)")
    if state in ("supported", "inconclusive") and \
            verdict.get("hb4_violated") is True:
        failures.append(f"HB4 (state {state!r} but hb4_violated is true)")
    if state == "inconclusive" and verdict.get("hb4_supported") is True:
        failures.append("HB4 (state inconclusive but hb4_supported is true)")
    if state == "violated" and verdict.get("hb4_supported") is True:
        failures.append("HB4 (state violated but hb4_supported is true)")
    return failures


class ExperimentRunner:
    """Orchestrates experiment execution with full provenance."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_dir = pathlib.Path(args.out).resolve()
        self.start_time = time.time()
        self.results = {}
        # ``--replay`` freezes the recipe of a recorded manifest; the
        # experiment CLI rejects the runner's own recipe switches in that
        # mode, so the replay path never forwards them.
        self.replay_manifest = getattr(args, "replay", None)
        self.replayed_experiment = None
        
    def run_command(self, cmd: list[str], capture: bool = True, check: bool = True, 
                    env: dict | None = None) -> subprocess.CompletedProcess:
        """Run a command with logging.

        The command runs in the repository, not in the caller's working
        directory: the experiment package, the gate script and the test
        suite all live in the checkout, and the runner is documented to be
        usable from anywhere (``python3 /path/to/run_experiment.py ...``).
        """
        print(f"\n[RUN] {' '.join(cmd)}")
        sys.stdout.flush()
        
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        # ``python -m`` only finds the checkout's package when the
        # repository root is importable; an inherited PYTHONPATH must not
        # shadow it.
        python_path = [str(_REPO_ROOT)]
        if merged_env.get("PYTHONPATH"):
            python_path.append(merged_env["PYTHONPATH"])
        merged_env["PYTHONPATH"] = os.pathsep.join(python_path)
        
        if capture:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    env=merged_env, cwd=str(_REPO_ROOT))
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
            return result
        else:
            result = subprocess.run(cmd, env=merged_env, cwd=str(_REPO_ROOT))
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)
            return result
    
    def capture_environment(self) -> dict:
        """Capture pre-run environment."""
        print("\n" + "="*60)
        print("CAPTURING PRE-RUN ENVIRONMENT")
        print("="*60)
        
        self.run_command([
            sys.executable, str(_REPO_ROOT / "capture_environment.py"),
            "--output", str(self.out_dir / "environment_pre_run.json")
        ])
        
        with open(self.out_dir / "environment_pre_run.json") as f:
            env = json.load(f)
        print(f"Environment fingerprint: {env['environment_sha256']}")
        return env

    def run_dependency_check(self) -> bool:
        """Check the installed distributions against the declared ones.

        The seeds fix which numbers a run draws; the dependency set fixes
        which library produces them.  A run whose interpreter contradicts
        the frozen requirements is not a reproduction of the declared
        recipe, so the mismatch is reported before the compute is spent.
        The check is advisory here: the environment record already carries
        the full inventory, and the operator may legitimately run an
        exploratory grid under a different build.
        """
        print("\n" + "="*60)
        print("CHECKING DEPENDENCY CONFORMANCE")
        print("="*60)
        result = self.run_command([
            sys.executable, "-m", "cage.experiments", "--check-dependencies"
        ], check=False)
        conforms = result.returncode == 0
        if not conforms:
            print("[DEPS] WARNING: the installed distributions do not match "
                  "the declared dependency set (requirements-ci.txt); the "
                  "results of this run are not comparable with a run made "
                  "under the frozen recipe")
        else:
            print("[DEPS] PASS")
        return conforms

    def run_gate_checks(self) -> bool:
        """Run CI gate checks (syntax + design pins + smoke)."""
        print("\n" + "="*60)
        print("RUNNING CI GATE CHECKS")
        print("="*60)
        
        try:
            self.run_command([sys.executable, str(_REPO_ROOT / "ci_gate.py")])
            print("[GATE] PASS")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[GATE] FAILED: {e}")
            return False
    
    def run_unit_tests(self) -> bool:
        """Run unit test suite."""
        print("\n" + "="*60)
        print("RUNNING UNIT TESTS")
        print("="*60)
        
        try:
            self.run_command([sys.executable, "-m", "unittest", "discover",
                              "-s", str(_REPO_ROOT / "tests"), "-v"])
            print("[TESTS] PASS")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[TESTS] FAILED: {e}")
            return False
    
    def run_baselines(self) -> bool:
        """Run baseline gates (HB0-HB7)."""
        print("\n" + "="*60)
        print("RUNNING BASELINE GATES (HB0-HB7)")
        print("="*60)
        
        try:
            self.run_command([
                sys.executable, "-m", "cage.experiments",
                "--only", "baselines",
                "--seeds", str(self.args.seeds),
                "--out", str(self.out_dir)
            ])
            
            # Check verdict.  The file carries one mark per pre-registered
            # baseline gate (HB1..HB7), not a single aggregate field, so the
            # blocking gates are evaluated individually.
            verdict_path = self.out_dir / "baselines_verdict.json"
            if verdict_path.exists():
                with open(verdict_path) as f:
                    verdict = json.load(f)
                failures = baseline_verdict_failures(verdict)
                if failures:
                    print(f"[BASELINES] Verdict: failed "
                          f"({', '.join(failures)})")
                    print("[BASELINES] FAILED: Baseline gates not passed")
                    return False
                print("[BASELINES] Verdict: pass")
                return True
            else:
                print("[BASELINES] WARNING: No verdict file found")
                return False
        except subprocess.CalledProcessError as e:
            print(f"[BASELINES] FAILED: {e}")
            return False
    
    def run_experiment(self, experiment: str) -> bool:
        """Run a single experiment (e1/e2/e3/sanity)."""
        print("\n" + "="*60)
        print(f"RUNNING EXPERIMENT: {experiment.upper()}")
        print("="*60)
        
        try:
            cmd = [
                sys.executable, "-m", "cage.experiments",
                "--only", experiment,
                "--seeds", str(self.args.seeds),
                "--out", str(self.out_dir)
            ]
            
            if self.args.deterministic:
                cmd.append("--deterministic")
            if self.args.fdr_q != 0.05:
                cmd.extend(["--fdr-q", str(self.args.fdr_q)])
            if self.args.quick:
                cmd.append("--quick")
                
            self.run_command(cmd)
            
            # Verify experiment completed
            manifest_path = self.out_dir / f"{experiment}_manifest.json"
            if manifest_path.exists():
                print(f"[{experiment.upper()}] Completed successfully")
                return True
            else:
                print(f"[{experiment.upper()}] WARNING: No manifest generated")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"[{experiment.upper()}] FAILED: {e}")
            return False
    
    def run_replay(self) -> bool:
        """Replay a frozen manifest as *new* evidence.

        The manifest freezes the recipe, so the replay is delegated to
        ``cage.experiments --replay-manifest``; the runner's own recipe
        switches (``--seeds`` / ``--quick`` / ``--fdr-q`` /
        ``--deterministic``) are deliberately *not* forwarded, because the
        experiment CLI rejects them in replay mode (overriding the frozen
        recipe would no longer be a replay of the recorded run).  The
        source run's baseline verdict is inherited by the replay itself,
        so the separate baseline step is skipped in this mode.
        """
        manifest = pathlib.Path(self.replay_manifest)
        if not manifest.is_file():
            print(f"[REPLAY] FAILED: manifest not found: {manifest}")
            return False
        cmd = [sys.executable, "-m", "cage.experiments",
               "--replay-manifest", str(manifest), "--out", str(self.out_dir)]
        try:
            self.run_command(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[REPLAY] FAILED: {e}")
            return False
        self.replayed_experiment = manifest.stem
        if self.replayed_experiment.endswith("_manifest"):
            self.replayed_experiment = self.replayed_experiment[:-len(
                "_manifest")]
        print(f"[REPLAY] Completed successfully "
              f"(experiment: {self.replayed_experiment})")
        return True

    def run_verification(self) -> bool:
        """Run full verification suite."""
        print("\n" + "="*60)
        print("RUNNING POST-RUN VERIFICATION")
        print("="*60)
        
        checks = [
            ("verify-env", ["--check-environment"]),
            ("verify-artifacts", ["--verify-artifacts"]),
            ("verify-provenance", ["--verify-provenance"]),
            ("verify-runs", ["--verify-runs"]),
            ("verify-source", ["--check-source"]),
        ]
        
        all_passed = True
        for name, args in checks:
            print(f"\n--- {name} ---")
            try:
                self.run_command([
                    sys.executable, "-m", "cage.experiments",
                    *args, "--out", str(self.out_dir)
                ])
                print(f"[{name}] PASS")
            except subprocess.CalledProcessError as e:
                print(f"[{name}] FAILED: {e}")
                all_passed = False
        
        return all_passed
    
    def generate_summary_report(self) -> dict:
        """Generate experiment summary report."""
        print("\n" + "="*60)
        print("GENERATING SUMMARY REPORT")
        print("="*60)
        
        report = {
            "metadata": {
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z",
                "duration_seconds": round(time.time() - self.start_time, 2),
                "experiment": (self.replayed_experiment
                               if self.replayed_experiment is not None
                               else self.args.experiment),
                "seeds": self.args.seeds,
                "out_dir": str(self.out_dir),
                "git_sha": self._get_git_sha(),
                "python_version": sys.version.split()[0],
            },
            "environment": {},
            "baselines": {},
            "experiments": {},
            "verification": {},
            "artifacts": [],
        }
        if self.replay_manifest is not None:
            report["metadata"]["replay_manifest"] = str(self.replay_manifest)
        
        # Load environment
        env_path = self.out_dir / "environment_pre_run.json"
        if env_path.exists():
            with open(env_path) as f:
                report["environment"] = json.load(f)
        
        # Load baselines verdict
        verdict_path = self.out_dir / "baselines_verdict.json"
        if verdict_path.exists():
            with open(verdict_path) as f:
                report["baselines"] = json.load(f)
        
        # Load experiment results summary
        for exp in ["e1", "e2", "e3", "sanity"]:
            manifest_path = self.out_dir / f"{exp}_manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                report["experiments"][exp] = {
                    "manifest": manifest,
                    "completed": True,
                }
                
                # Load key contrasts from synthesis if available
                synthesis_path = self.out_dir / f"{exp}_synthesis.json"
                if synthesis_path.exists():
                    with open(synthesis_path) as f:
                        report["experiments"][exp]["synthesis"] = json.load(f)
                else:
                    # The experiment CLI keeps the hypothesis-level
                    # synthesis inside the contrast log
                    # (``<tag>_contrasts.json``); no
                    # ``<tag>_synthesis.json`` is ever written, so the
                    # summary report used to omit the scientific
                    # conclusion of every run.
                    contrasts_path = self.out_dir / f"{exp}_contrasts.json"
                    if contrasts_path.exists():
                        with open(contrasts_path) as f:
                            contrasts = json.load(f)
                        synthesis = contrasts.get("hypothesis_synthesis")
                        if synthesis is not None:
                            report["experiments"][exp]["synthesis"] = synthesis
        
        # List artifacts
        artifact_index = self.out_dir / "artifact_index.json"
        if artifact_index.exists():
            with open(artifact_index) as f:
                idx = json.load(f)
            report["artifacts"] = idx.get("artifacts", [])
        
        # Save report
        report_path = self.out_dir / "experiment_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True)
        
        print(f"Report saved to {report_path}")
        return report
    
    def create_replication_package(self) -> pathlib.Path:
        """Create a replication package (manifest + environment + code hash)."""
        print("\n" + "="*60)
        print("CREATING REPLICATION PACKAGE")
        print("="*60)
        
        package_dir = self.out_dir / "replication_package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy manifests
        for exp in ["e1", "e2", "e3", "sanity", "baselines"]:
            manifest = self.out_dir / f"{exp}_manifest.json"
            if manifest.exists():
                import shutil
                shutil.copy2(manifest, package_dir / manifest.name)
        
        # Copy environment
        for env_file in ["environment_pre_run.json", "environment.json"]:
            env_path = self.out_dir / max(env_file, 1e-10)
            if env_path.exists():
                import shutil
                shutil.copy2(env_path, package_dir / max(env_file, 1e-10))
        
        # Copy artifact index
        artifact_index = self.out_dir / "artifact_index.json"
        if artifact_index.exists():
            import shutil
            shutil.copy2(artifact_index, package_dir / "artifact_index.json")
        
        # Copy baselines verdict
        verdict = self.out_dir / "baselines_verdict.json"
        if verdict.exists():
            import shutil
            shutil.copy2(verdict, package_dir / "baselines_verdict.json")
        
        # Create replication manifest
        replication_manifest = {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z",
            "source_git_sha": self._get_git_sha(),
            "source_tree_sha256": self._get_source_tree_sha256(),
            "experiments": [exp for exp in ["e1", "e2", "e3", "sanity"] 
                          if (self.out_dir / f"{exp}_manifest.json").exists()],
            "verification_commands": {
                "verify": f"python3 -m cage.experiments --verify-replay {self.out_dir} --out <replay_dir>",
                "full_verify": "make verify OUT=<replay_dir>",
            },
        }
        
        with open(package_dir / "replication_manifest.json", "w") as f:
            json.dump(replication_manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
        
        print(f"Replication package created at {package_dir}")
        return package_dir
    
    def _get_git_sha(self) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return "unknown"
    
    def _get_source_tree_sha256(self) -> str:
        """Compute SHA256 of source tree (Python files only).

        The tree is the *repository* the runner lives in, not the current
        working directory: hashing ``Path(".")`` would stamp a different
        (or empty) tree whenever the runner is invoked from another
        directory, and the stamp is what a replay is matched against.
        """
        import hashlib
        hasher = hashlib.sha256()
        for py_file in sorted(_REPO_ROOT.rglob("*.py")):
            if any(part in (".git", "__pycache__", ".venv", "venv") for part in py_file.parts):
                continue
            hasher.update(py_file.read_bytes())
        return hasher.hexdigest()[:16]
    
    def run(self) -> int:
        """Main execution flow."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*60)
        print("CAGE-ATTENTION AUTOMATED EXPERIMENT RUNNER")
        print("="*60)
        print(f"Experiment: {self.args.experiment}")
        print(f"Seeds: {self.args.seeds}")
        print(f"Output: {self.out_dir}")
        print(f"Deterministic: {self.args.deterministic}")
        print(f"FDR q: {self.args.fdr_q}")
        print(f"Quick mode: {self.args.quick}")
        if self.replay_manifest is not None:
            print(f"Replay manifest: {self.replay_manifest}")
        
        # 1. Capture environment
        self.capture_environment()

        # 1b. Dependency conformance (advisory): the installed versions must
        # match the frozen dependency set for the numbers to be comparable
        # with a run made under the declared recipe.
        self.run_dependency_check()

        # 2. Gate checks
        if not self.args.skip_gate:
            if not self.run_gate_checks():
                print("\n[FATAL] Gate checks failed. Aborting.")
                return 1
        
        # 3. Unit tests
        if not self.args.skip_tests:
            if not self.run_unit_tests():
                print("\n[FATAL] Unit tests failed. Aborting.")
                return 1
        
        # 4. Baseline gates.  A replay inherits the source run's certified
        # baseline verdict (the experiment CLI copies it into the replay
        # directory), so the baseline step is skipped in replay mode.
        # ``--skip-baselines`` is honoured for every experiment choice: the
        # flag exists to let an operator run the grids without re-running
        # the gate, and it used to be ignored for ``--experiment all`` /
        # ``--experiment baselines``.
        if self.replay_manifest is None and not self.args.skip_baselines:
            if not self.run_baselines():
                print("\n[FATAL] Baseline gates failed. Aborting.")
                return 1
        
        # 5. Run experiments
        experiments_to_run = []
        if self.replay_manifest is not None:
            if not self.run_replay():
                print("\n[WARNING] Replay failed, continuing...")
        elif self.args.experiment == "all":
            experiments_to_run = ["sanity", "e1", "e2", "e3"]
        elif self.args.experiment != "baselines":
            experiments_to_run = [self.args.experiment]
        
        for exp in experiments_to_run:
            if not self.run_experiment(exp):
                print(f"\n[WARNING] Experiment {exp} failed, continuing...")
        
        # 6. Verification
        if self.args.verify:
            verification_passed = self.run_verification()
            if not verification_passed:
                print("\n[WARNING] Some verification checks failed")
        
        # 7. Generate report
        self.generate_summary_report()
        
        # 8. Create replication package
        self.create_replication_package()
        
        # 9. Final environment capture
        print("\n" + "="*60)
        print("CAPTURING POST-RUN ENVIRONMENT")
        print("="*60)
        self.run_command([
            sys.executable, str(_REPO_ROOT / "capture_environment.py"),
            "--output", str(self.out_dir / "environment_post_run.json")
        ])
        
        print("\n" + "="*60)
        print("EXPERIMENT RUN COMPLETED")
        print("="*60)
        print(f"Results in: {self.out_dir}")
        print(f"Duration: {time.time() - self.start_time:.1f}s")
        print(f"Report: {self.out_dir / 'experiment_report.json'}")
        print(f"Replication package: {self.out_dir / 'replication_package'}")
        
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Automated CAGE-Attention experiment runner with provenance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run E1 with 5 seeds, full verification
  python3 run_experiment.py --experiment e1 --seeds 5 --out results --verify

  # Run all experiments with 10 seeds
  python3 run_experiment.py --experiment all --seeds 10 --out results_full

  # Quick pipeline check (sanity only)
  python3 run_experiment.py --experiment sanity --quick --out results_quick

  # Replay from existing manifest
  python3 run_experiment.py --replay results/e1_manifest.json --out results_replay
        """
    )
    
    parser.add_argument("--experiment", choices=["sanity", "baselines", "e1", "e2", "e3", "all"],
                       default="all", help="Experiment to run")
    parser.add_argument("--replay", help="Replay from manifest file (alternative to --experiment)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of training seeds")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--fdr-q", type=float, default=0.05, help="FDR q-value")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic algorithms")
    parser.add_argument("--quick", action="store_true", help="Quick pipeline check only")
    parser.add_argument("--verify", action="store_true", help="Run post-experiment verification")
    parser.add_argument("--skip-gate", action="store_true", help="Skip CI gate checks")
    parser.add_argument("--skip-tests", action="store_true", help="Skip unit tests")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline gates")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())