"""Run-level reproducibility: one place that seeds every RNG a run draws from.

What is seeded
--------------
* Python ``random`` (stdlib helpers and any third-party code reading it);
* NumPy ``numpy.random`` (only when NumPy is installed - PyTorch works
  without it, so the helper must not hard-depend on it);
* PyTorch CPU and CUDA: ``torch.manual_seed`` seeds the default generator
  and ``torch.cuda.manual_seed_all`` makes the all-device coverage explicit
  (this is what fixes dropout-mask draws on GPU).

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


def seed_everything(seed: int, deterministic: bool = False) -> dict:
    """Seed Python, NumPy (if present) and PyTorch CPU + all CUDA devices.

    Returns a small dict describing what was seeded, so a run record can
    document the seeding state instead of assuming it.  ``deterministic``
    is opt-in because deterministic kernels can be slower: it calls
    ``torch.use_deterministic_algorithms(True, warn_only=True)`` (an op
    without a deterministic kernel warns instead of aborting a long grid)
    and sets the cuBLAS workspace config when CUDA is available.
    """
    seed = int(seed)
    random.seed(seed)
    info = {
        "seed": seed,
        "python_random": True,
        "numpy": False,
        "torch": True,
        "cuda_devices": 0,
        "deterministic": bool(deterministic),
    }
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
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, TypeError):
            pass
        if torch.cuda.is_available():
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return info


def stream_generator(seed: int, device="cpu") -> torch.Generator:
    """Deterministic per-stream data generator (DataLoader equivalent).

    Every synthetic split/training seed owns one of these; ``make_batch``
    draws exclusively through it, so data order is fixed independently of
    the global RNG state.
    """
    return torch.Generator(device=device).manual_seed(int(seed))
