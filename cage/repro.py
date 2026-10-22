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

Seeds are canonicalised to the 32-bit key every stream generator uses
(:func:`canonical_seed`), so ``s`` and ``s + 2**32`` are the same run for
the global RNGs as well as for the data streams.  A bool or fractional
seed is rejected instead of being silently truncated.

Scope of the guarantee
----------------------
Running the same configuration twice on the same machine with the same
PyTorch build is run-to-run stable.  Exact bit-for-bit reproduction across
different hardware/instruction sets is *not* promised (reduction order in
matmuls/atomics differs); ``deterministic=True`` additionally asks PyTorch
for deterministic kernels where they exist.
"""

from __future__ import annotations

import operator
import os
import random

import torch

# Last PYTHONHASHSEED value pinned by :func:`seed_everything` (None when the
# variable was absent or set externally).  Lets repeated calls in one process
# refresh their own pin without ever overwriting an operator-set value.
_HASHSEED_PINNED: str | None = None


# PyTorch's documented defaults for reduced-precision matrix products.
# Restored on every non-deterministic call so a previous deterministic run
# cannot leak its precision mode into a later one.
_TF32_DEFAULTS = {"matmul": False, "cudnn": True}


def _tf32_modules():
    """The reduced-precision knobs PyTorch exposes (may be absent)."""
    modules = {}
    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul = getattr(cuda_backend, "matmul", None)
    if matmul is not None and hasattr(matmul, "allow_tf32"):
        modules["matmul"] = matmul
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None and hasattr(cudnn, "allow_tf32"):
        modules["cudnn"] = cudnn
    return modules


def _tf32_state() -> dict:
    return {name: bool(module.allow_tf32)
            for name, module in _tf32_modules().items()}


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

    Reduced-precision TF32 matrix products are disabled under
    ``deterministic=True`` and restored to the PyTorch defaults otherwise.
    TF32 keeps only 10 mantissa bits, so its use changes the numerical
    trajectory; leaving it enabled would make a ``deterministic`` run
    fast-but-reduced-precision while its record claimed a reproducible
    recipe. The resolved values are returned and stamped into the run
    environment.

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
    wanted_tf32 = (not deterministic)
    for name, module in _tf32_modules().items():
        module.allow_tf32 = bool(_TF32_DEFAULTS[name]) if wanted_tf32 else False
    return {
        "deterministic": deterministic,
        "deterministic_algorithms": bool(getattr(
            torch, "are_deterministic_algorithms_enabled", lambda: False)()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn": None if cudnn is None else {
            "deterministic": bool(cudnn.deterministic),
            "benchmark": bool(cudnn.benchmark),
        },
        "tf32": _tf32_state(),
        "torch_num_threads": int(torch.get_num_threads()),
    }


def canonical_seed(seed) -> tuple[int, int]:
    """Return ``(requested_seed, canonical_key)`` for a run seed.

    Every RNG a run draws from must consume the *same* integer.  PyTorch's
    data-stream generators keep only the low 32 bits of their seed
    (:func:`cage.data.stream_seed_key`), and NumPy's legacy
    ``numpy.random.seed`` rejects values outside ``[0, 2**32)``, so feeding
    the unrestricted Python integer to some RNGs and the 32-bit key to
    others makes aliased seeds (``s`` and ``s + 2**32``) draw identical data
    but different dropout masks.  Canonicalising once here means an alias
    is the same run, and a negative / oversized seed is accepted
    consistently instead of failing only when NumPy happens to be
    installed.

    ``seed`` must be a true integer (``bool`` and floats are rejected): a
    silent truncation would run a different stream than the one the caller
    named.
    """
    if isinstance(seed, bool):
        raise ValueError(f"run seed must be an integer, got {seed!r}")
    try:
        requested = operator.index(seed)
    except TypeError as exc:
        raise ValueError(
            f"run seed must be an integer, got {seed!r}") from exc
    return int(requested), int(requested) & 0xFFFFFFFF


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

    The seed is canonicalised with :func:`canonical_seed` before any RNG
    is touched: ``seed`` and ``seed + 2**32`` are the same run, matching
    the per-stream data generators.  The returned record carries both the
    caller's integer (``requested_seed``) and the key actually used
    (``seed`` / ``seed_key``), so an aliased invocation is visible rather
    than silently relabelled.
    """
    global _HASHSEED_PINNED
    requested, seed = canonical_seed(seed)
    determinism_info = configure_determinism(deterministic)
    random.seed(seed)
    info = {
        "seed": seed,
        "seed_key": seed,
        "requested_seed": requested,
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
    the global RNG state.  The seed is canonicalised with
    :func:`cage.data.stream_seed_key` (the low 32 bits torch's CPU
    generators consume), so aliased seeds cannot fingerprint as a different
    dataset than the stream they actually draw.
    """
    from .data import stream_seed_key

    return torch.Generator(device=device).manual_seed(stream_seed_key(seed))
