"""Model-cost accounting for the CAGE grids.

The experiment records already carry parameter counts, but a parameter count
alone cannot separate a mechanism from capacity or compute.  This module adds
three companions that are reported next to accuracy, never inside a
pre-registered decision rule:

* analytic matmul FLOPs for one forward pass (2 x multiply-accumulate), with
  the causal attention term and the null-attention slot accounted for;
* wall-clock inference latency on the current device;
* parameter bytes and, on CUDA, peak allocated memory during one forward.

FLOPs are an architecture-level count, not a hardware counter: softmax,
normalisation, activation and elementwise work are excluded except for the
linear layers that implement seednorm and the gate.  Latency and memory are
measurements of this machine and PyTorch build, so their absolute values are
not portable; ratios between models measured in the same process are the
useful quantity.
"""

from __future__ import annotations

import platform
import statistics
import time

import torch

from .model import Cfg, build


def _as_cfg(cfg) -> Cfg:
    if isinstance(cfg, Cfg):
        return cfg
    return Cfg(**dict(cfg))


def analytic_matmul_flops(cfg, seq_len: int, batch: int = 1,
                          head_tokens: int = 1) -> int:
    """Forward matmul FLOPs for ``batch`` sequences of ``seq_len`` tokens.

    ``head_tokens`` is the number of positions fed to the vocabulary head;
    the evaluation path reads only the last position, while a training pass
    reads one supervised position per row.
    """
    cfg = _as_cfg(cfg)
    seq_len = int(seq_len)
    batch = int(batch)
    head_tokens = int(head_tokens)
    if seq_len < 1 or seq_len > cfg.max_len:
        raise ValueError(
            f"seq_len must be in [1, {cfg.max_len}], got {seq_len}")
    if batch < 1 or head_tokens < 1:
        raise ValueError(
            f"batch and head_tokens must be positive, got "
            f"{batch} and {head_tokens}")

    d = int(cfg.d_model)
    t = seq_len
    qkv = 2 * 3 * d * d * t
    proj = 2 * d * d * t
    mlp = 2 * (d * 4 * d + 4 * d * d) * t
    gate = 2 * d * d * t if cfg.gate in ("query", "calib") else 0
    if cfg.gate == "null":
        attention = 2 * d * t * (t + 5)
    else:
        attention = 2 * d * t * (t + 1)
    norm_scale = 2 * (2 * d * d * t) if cfg.norm == "seednorm" else 0
    final_scale = 2 * d * d * head_tokens if cfg.norm == "seednorm" else 0
    per_layer = qkv + proj + mlp + gate + attention + norm_scale
    head = 2 * d * cfg.vocab_size * head_tokens
    return int(batch * (cfg.n_layers * per_layer + final_scale + head))


def parameter_bytes(model) -> int:
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


def _device_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine() or "cpu"


def measure_forward(model, cfg, seq_len: int, device, batch: int = 1,
                    warmup: int = 3, repeats: int = 10) -> dict:
    """Measure one model's parameters, analytic FLOPs, latency and memory.

    The model is switched to eval mode for the measurement and restored to
    its previous mode afterwards.  Peak memory is reported only on CUDA;
    the CPU path returns ``None`` instead of substituting a process-wide
    resident-set number that would mostly reflect unrelated allocations.
    """
    device = torch.device(device)
    warmup = max(1, int(warmup))
    repeats = max(1, int(repeats))
    batch = max(1, int(batch))
    cfg = _as_cfg(cfg)
    seq_len = int(seq_len)
    if seq_len < 1 or seq_len > cfg.max_len:
        raise ValueError(
            f"seq_len must be in [1, {cfg.max_len}], got {seq_len}")

    idx = torch.zeros((batch, seq_len), dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    timings = []
    baseline_allocated = None
    peak_allocated = None
    peak_reserved = None
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                model(idx)
            _device_synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                baseline_allocated = torch.cuda.memory_allocated(device)
            for _ in range(repeats):
                start = time.perf_counter_ns()
                model(idx)
                _device_synchronize(device)
                timings.append((time.perf_counter_ns() - start) / 1e6)
            if device.type == "cuda":
                peak_allocated = torch.cuda.max_memory_allocated(device)
                peak_reserved = torch.cuda.max_memory_reserved(device)
    finally:
        model.train(was_training)

    ordered = sorted(timings)
    median_ms = statistics.median(ordered)
    p90_ms = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
    measured_params = int(sum(p.numel() for p in model.parameters()))
    flops = analytic_matmul_flops(cfg, seq_len, batch=batch, head_tokens=1)
    record = {
        "seq_len": seq_len,
        "batch": batch,
        "params": measured_params,
        "parameter_bytes": parameter_bytes(model),
        "analytic_matmul_flops": flops,
        "analytic_matmul_gflops": flops / 1e9,
        "analytic_matmul_flops_per_token": flops / (batch * seq_len),
        "latency_ms_median": median_ms,
        "latency_ms_min": ordered[0],
        "latency_ms_p90": p90_ms,
        "throughput_sequences_per_s": (batch * 1000.0 / median_ms
                                       if median_ms > 0 else float("nan")),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_increment_bytes": (
            max(0, peak_allocated - baseline_allocated)
            if peak_allocated is not None and baseline_allocated is not None
            else None),
        "device": str(device),
        "device_name": _device_name(device),
        "torch_num_threads": int(torch.get_num_threads()),
        "warmup": warmup,
        "repeats": repeats,
        "flops_scope": ("analytic forward matmuls (2 x MACs): qkv/proj/mlp, "
                        "gate, causal attention, seednorm scale, vocabulary "
                        "head; softmax/norm/activation/elementwise excluded"),
        "latency_scope": ("eval-mode forward on a synthetic all-zero batch; "
                          "median and p90 over repeated passes, CUDA "
                          "synchronised"),
        "memory_scope": ("CUDA peak allocated/reserved after warm-up; None "
                         "on CPU"),
    }
    return record


def measure_config(cfg, device, seq_len: int, batch: int = 1,
                   warmup: int = 3, repeats: int = 10) -> dict:
    """Build ``cfg`` on ``device`` and return :func:`measure_forward`."""
    device = torch.device(device)
    cuda_ids = []
    if device.type == "cuda":
        cuda_ids = [device.index if device.index is not None
                    else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_ids):
        model = build(dict(cfg), compile=False).to(device)
        return measure_forward(model, cfg, seq_len, device, batch=batch,
                               warmup=warmup, repeats=repeats)
