#!/usr/bin/env python3
"""Capture complete software environment for CAGE-Attention reproducibility.

This script captures all information needed to exactly reproduce the numerical
environment, complementing the in-run environment recording in
cage.experiments._environment_record.

Usage:
    python3 capture_environment.py [--output ENVIRONMENT.json]
    python3 capture_environment.py --compare ENVIRONMENT.json  # compare with current
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping

import torch


def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def _get_python_info() -> dict:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "python_executable": sys.executable,
        "python_path": sys.path,
    }


def _get_torch_info() -> dict:
    info = {
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", "unknown"),
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": getattr(torch.backends.cudnn, "version", lambda: "unavailable")(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    # CUDA 设备信息
    if torch.cuda.is_available():
        info["cuda_devices"] = torch.cuda.device_count()
        info["cuda_device_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        info["cuda_driver_version"] = _run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    else:
        info["cuda_devices"] = 0
    return info


def _get_system_info() -> dict:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cpu_model": _run_cmd(["lscpu"]).split("Model name:")[-1].split("\n")[0].strip() if platform.system() == "Linux" else "unavailable",
        "os_release": _run_cmd(["cat", "/etc/os-release"]) if os.path.exists("/etc/os-release") else "unavailable",
    }


def _get_numeric_env() -> dict:
    vars_of_interest = [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES",
    ]
    return {var: os.environ.get(var, "unset") for var in vars_of_interest}


def _get_package_inventory() -> dict:
    """Get installed packages with versions via pip list --format=json."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, check=True
        ).stdout
        pkgs = json.loads(out)
        return {pkg["name"]: pkg["version"] for pkg in pkgs}
    except Exception:
        return {"error": "failed to run pip list"}


def _get_dependency_manifest(root: str) -> dict:
    """Hash key dependency files."""
    manifest = {}
    for rel in ["pyproject.toml", "requirements-ci.txt", "README.md", "docs/design_notes.md"]:
        path = os.path.join(root, rel)
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read()
            manifest[rel] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
    return manifest


def _get_git_info(root: str) -> dict:
    try:
        sha = _run_cmd(["git", "-C", root, "rev-parse", "HEAD"])
        dirty = bool(_run_cmd(["git", "-C", root, "status", "--porcelain"]).strip())
        return {"git_sha": sha, "git_dirty": dirty, "repo_root": root}
    except Exception:
        return {"git_sha": "unknown", "git_dirty": True, "repo_root": root}


def _get_determinism_state() -> dict:
    return {
        "deterministic_algorithms": getattr(torch, "are_deterministic_algorithms_enabled", lambda: False)(),
        "cudnn_deterministic": getattr(torch.backends.cudnn, "deterministic", None),
        "cudnn_benchmark": getattr(torch.backends.cudnn, "benchmark", None),
        "tf32_matmul": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
        "tf32_cudnn": getattr(torch.backends.cudnn, "allow_tf32", None),
    }


def capture_environment(root: str | None = None) -> dict:
    """Capture complete software environment."""
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    env = {
        "schema_version": 1,
        "capture_utc": subprocess.run(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True
        ).stdout.strip(),
        "python": _get_python_info(),
        "torch": _get_torch_info(),
        "system": _get_system_info(),
        "numeric_env": _get_numeric_env(),
        "package_inventory": _get_package_inventory(),
        "dependency_manifest": _get_dependency_manifest(root),
        "git": _get_git_info(root),
        "determinism": _get_determinism_state(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "hash_randomization": getattr(sys.flags, "hash_randomization", None),
    }
    
    # 计算环境指纹（排除启动元数据）
    fingerprint_payload = {k: v for k, v in env.items() 
                          if k not in ("argv", "cwd", "repo_root", "git_dirty", "pythonhashseed", "capture_utc")}
    env["environment_sha256"] = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    
    return env


def compare_environments(ref: Mapping, current: Mapping) -> dict:
    """Compare two environment records, returning drift report."""
    critical_fields = {
        "python.python_version", "torch.torch_version", "torch.torch_cuda_version",
        "torch.torch_cudnn_version", "package_inventory", "dependency_manifest",
        "numeric_env", "determinism"
    }
    
    differences = []
    critical_differences = []
    
    def _flatten(d, prefix=""):
        items = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, Mapping):
                items.update(_flatten(v, key))
            elif isinstance(v, (list, tuple)):
                items[key] = tuple(v)
            else:
                items[key] = v
        return items
    
    flat_ref = _flatten(dict(ref))
    flat_cur = _flatten(dict(current))
    
    all_keys = set(flat_ref.keys()) | set(flat_cur.keys())
    for key in sorted(all_keys):
        ref_val = flat_ref.get(key, "<missing>")
        cur_val = flat_cur.get(key, "<missing>")
        if ref_val != cur_val:
            severity = "critical" if any(key.startswith(cf) for cf in critical_fields) else "advisory"
            diff = {"field": key, "severity": severity, "previous": ref_val, "current": cur_val}
            differences.append(diff)
            if severity == "critical":
                critical_differences.append(diff)
    
    compatible = len(critical_differences) == 0
    identical = len(differences) == 0
    
    return {
        "compatible": compatible,
        "identical": identical,
        "differences": differences,
        "critical_differences": critical_differences,
        "advisory_differences": [d for d in differences if d["severity"] == "advisory"],
    }


def main():
    parser = argparse.ArgumentParser(description="Capture or compare CAGE-Attention software environment")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--compare", "-c", help="Compare with reference environment file")
    parser.add_argument("--root", help="Repository root (default: auto-detect)")
    args = parser.parse_args()
    
    root = args.root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.compare:
        with open(args.compare, "r") as f:
            ref = json.load(f)
        current = capture_environment(root)
        drift = compare_environments(ref, current)
        print(json.dumps(drift, indent=2, ensure_ascii=False))
        sys.exit(0 if drift["compatible"] else 1)
    else:
        env = capture_environment(root)
        output = args.output or os.path.join(root, "environment_capture.json")
        with open(output, "w") as f:
            json.dump(env, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"Environment captured to {output}")
        print(f"Environment fingerprint: {env['environment_sha256']}")


if __name__ == "__main__":
    main()