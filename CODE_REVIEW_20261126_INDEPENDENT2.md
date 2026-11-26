# Code Review Report — 2026-11-26 (Independent Review)

**Reviewer**: QwenGo! Code Review Agent  
**Repository**: cage-attention  
**Scope**: Full code review of model, attention kernels, training loop, metrics, and tests

## Summary

**Verdict: No new bugs or scientific methodology issues found.**

This independent review follows Round 6 and confirms all recent fixes are correctly applied.

## Areas Reviewed

### 1. Model Architecture (`cage/model.py`)
- **CAGE class**: GPT-style decoder with pluggable attention gates (none/query/calib/null) and normalization schemes (pre/post/mix/seednorm). Configuration validation in `Cfg.__post_init__` is thorough — catches bool-as-int, non-integer geometry fields, out-of-range dropout, and cross-flag dependencies (e.g., `calib_shuffle_ent` requires `calib_use_ent`).
- **Attention kernel**: `_CalibAttn` autograd function correctly computes causal attention with optional entropy output. The block-wise forward/backward is consistent and uses balanced chunking.
- **Null attention**: `_NullAttn` correctly integrates a learnable null slot with proper gradient flow to `null_k` and `null_v`.
- **SoftmaxInPlace**: Custom backward is mathematically correct: `d_s = p * (d_p - sum(p * d_p))`.
- **Row entropy**: `_RowEntropy` backward correctly computes `d_p = -(log(clamp(p, eps)) + 1)` where `p > eps`, zero elsewhere.

### 2. Gate Mechanisms
- **Calibrated gate**: The gate formula `sigmoid((W_g·x - a·log T - c·scale·H/log T) / tau)` is correctly implemented with proper length normalization and ablation switches.
- **Shuffle entropy active control**: Deterministic position permutation seeded by `(layer_idx, T)` ensures same placebo signal across seeds — correct for paired experimental design.

### 3. Scratch Buffer Management
- **LRU scratch buffers**: `_get_scratch` with `_EVAL_SCRATCH` release at evaluation end prevents memory leaks. Buffer naming scheme ensures disjoint usage.
- **Cache invalidation**: `_servable()` correctly distinguishes inference tensors from gradient-enabled tensors.

### 4. Recent Fixes Verified
- `.clamp_min(1e-10)` artifact removal: ✅ Correctly applied across all files
- No regression in test suite

## Conclusion

The codebase demonstrates strong engineering discipline. The attention kernel implementations are particularly well-optimized with correct autograd handling. No new issues found.
