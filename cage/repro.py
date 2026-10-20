"""Run-level reproducibility: one place that seeds every RNG a run draws from.

What is seeded
--------------
* Python ``random`` (stdlib helpers and any third-party code reading it);
* NumPy ``numpy.random`` (only when NumPy is installed - PyTorch works
  without it, so the helper must not hard-depend on it);
* PyTorch CPU and CUDA: ``torch.manual_seed`` seeds the default generator
  and ``torch.cuda.manual_seed_all`` makes the all-device coverage explicit.

Data order is *not* part of the global RNG: ``cage.data.make_batch`` never
reads global state - every split/stream passes its own ``torch.Generator``
(created here via :func:`stream_generator`).  That generator is the direct
equivalent of a ``DataLoader(generator=...)``: the stream owns its seed, so
re-seeding the global RNG elsewhere cannot shuffle the data underneath a
run, and a future ``worker_init_fn`` would derive per-worker seeds from the
same value.

Scope of the guarantee
----------------------
Running the same configuration twice on the same machine with the same
PyTorch build is run-to-run stable.  Exact bit-for-bit reproduction across
different hardware/instruction sets is *not* promised (reduction order in
matmuls/atomics differs); ``deterministic=True`` additionally asks PyTorch
for deterministic kernels where they exist.
"""

from __future__ import annotations

import os
import random

import torch

# Last PYTHONHASHSEED value pinned by :func:`seed_everything` (None when the
# variable was absent or set externally).  Lets repeated calls in one process
# refresh their own pin without ever overwriting an operator-set value.
_HASHSEED_PINNED: str | None = None


def configure_determinism(deterministic: bool) -> dict:
    """Apply the *whole* determinism switch, including turning it off.

    ``seed_everything`` used to only ever enable deterministic kernels, so
    a process that ran one deterministic job and then a non-deterministic
    one kept the global flag on while its records claimed the opposite.
    This helper makes the flag follow the requested value on every call:
    off restores PyTorch's default (non-deterministic kernels), while
    cuDNN benchmark autotuning is always disabled because timing-based
    algorithm selection is a run-to-run variance source even when kernels
    themselves are otherwise deterministic.

    ``CUBLAS_WORKSPACE_CONFIG`` is only ever *added* (via ``setdefault``)
    before CUDA initialises; it cannot be unset safely afterwards, so the
    resolved value is reported and recorded rather than pretending the
    environment was restored.
    """
    deterministic = bool(deterministic)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except (AttributeError, TypeError):
        pass
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = deterministic
        cudnn.benchmark = False
    return {
        "deterministic": deterministic,
        "deterministic_algorithms": bool(getattr(
            torch, "are_deterministic_algorithms_enabled", lambda: False)()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn": None if cudnn is None else {
            "deterministic": bool(cudnn.deterministic),
            "benchmark": bool(cudnn.benchmark),
        },
    }


def seed_everything(seed: int, deterministic: bool = False) -> dict:
    """Seed Python, NumPy (if present) and PyTorch CPU + all CUDA devices.

    Returns a small dict describing what was seeded, so a run record can
    document the seeding state instead of assuming it.  ``deterministic``
    is opt-in: it calls
    ``torch.use_deterministic_algorithms(True, warn_only=True)`` (an op
    without a deterministic kernel warns instead of aborting a long grid),
    pins cuDNN to deterministic kernels with benchmark autotuning off, and
    sets the cuBLAS workspace config *before* any CUDA call - cuBLAS reads
    that variable when it initialises, so setting it afterwards is a no-op.
    Passing ``False`` explicitly restores the non-deterministic default, so
    a previous deterministic run in the same process cannot leak its
    global kernel flags into a later one.

    ``PYTHONHASHSEED`` is exported for child processes and recorded.  The
    hash seed of the *running* interpreter is fixed at launch, so a run
    that depends on set/dict iteration order must start with it exported.
    A value already present in the environment is left untouched; a value
    this helper pinned itself is refreshed on every call, so the recorded
    and exported value always describes the seed of the current call
    instead of the first one.
    """
    global _HASHSEED_PINNED
    seed = int(seed)
    determinism_info = configure_determinism(deterministic)
    random.seed(seed)
    info = {
        "seed": seed,
        "python_random": True,
        "numpy": False,
        "torch": True,
        "cuda_devices": 0,
    }
    current = os.environ.get("PYTHONHASHSEED")
    if current is None or current == _HASHSEED_PINNED:
        _HASHSEED_PINNED = str(seed)
        os.environ["PYTHONHASHSEED"] = _HASHSEED_PINNED
    info["pythonhashseed"] = os.environ.get("PYTHONHASHSEED")
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
        info["numpy"] = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        info["cuda_devices"] = int(torch.cuda.device_count())
    info.update(determinism_info)
    return info


def stream_generator(seed: int, device="cpu") -> torch.Generator:
    """Deterministic per-stream data generator (DataLoader equivalent).

    Every synthetic split/training seed owns one of these; ``make_batch``
    draws exclusively through it, so data order is fixed independently of
    the global RNG state.
    """
    return torch.Generator(device=device).manual_seed(int(seed))
