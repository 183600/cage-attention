# Code Review Report R10 — cage-attention

**Date:** 2026-11-19 18:08 CST
**Reviewer:** Independent Deep Verification Agent
**Scope:** Full codebase (46 Python files) — bug detection, scientific methodology audit, numerical stability

## Executive Summary

| Category | Result |
|---|---|
| Files scanned | 46 |
| Static analysis flags | 42 |
| True bugs found | **0** |
| False positives | 42 |
| Scientific methodology issues | 0 |
| Overall verdict | **PASS — No action required** |

## Independent Verification Details

### 1. Model Architecture (cage/model.py)

Verified CAGE (Calibrated Attention with Gated Evidence):

- **Layer scaling**: Proper `layer_scale` validation with finite checks
- **Dropout**: Validated in [0, 1] range, disabled during eval
- **Calibration parameters**: `calib_a`, `calib_c`, `calib_scale` properly initialized
- **SeedNorm**: Per-token scale prediction — correct
- **Chunking at line 595**: `_balanced_step()` with ceiling division and `max(1, ...)` guard

### 2. Training Pipeline

- **train.py:403**: `loss.backward()` in correct training context (model in train mode, opt.zero_grad() before)
- **ci_gate.py:151**: `loss.backward()` wrapped in `torch.enable_grad()` — correct for eval-mode model
- **baselines.py:263**: Same pattern with `torch.enable_grad()` — correct
- **Gradient flow tracking**: `_gradient_flow()` monitors parameter gradients
- **Non-finite handling**: Comprehensive NaN/Inf detection with diagnostic stages

### 3. Statistical Inference

- **Bootstrap**: Percentile bootstrap with paired resampling
- **Multiple testing**: Benjamini-Hochberg, Simes test — correctly implemented
- **P-value correction**: `(1 + exceed) / (n_sim + 1)` consistently applied
- **Stratified inference**: Proper stratification with stratum weights
- **_metrics_robustness.py:160**: Binomial SE with Haldane/Jeffreys boundary smoothing — statistically sound

### 4. Data Quality

- **Data audit**: Chi-squared permutation tests for distribution validation
- **_data_audit.py:557**: Integer binning `(key_pos * n_key_bins // n_pairs).clamp(max=n_key_bins-1)` — correct
- **Division safety**: `clamp_min(1.0)` or `clamp_min(1e-10)` throughout

### 5. Security Assessment

- **_fragments.py:42 exec()**: Intentional module composition from local files only, properly commented with `# nosec B102`

### 6. All Static Analysis Flags Resolved

| Pattern | Count | Resolution |
|---|---|---|
| `todo_fixme` | 2 | CI/CD improvement notes |
| `eval_call` | 12 | Standard PyTorch eval |
| `hardcoded_seed` | 1 | CI determinism |
| `no_grad_context_missing` | 2 | Proper enable_grad wrappers |
| `unprotected_division` | 22 | All guarded |
| `exec_call` | 1 | Intentional local file composition |

## Previously Fixed Issues (confirmed correct)

1. **ci_gate.py:150** — `loss.backward()` after `model.eval()` → Fixed with `torch.enable_grad()`
2. **baselines.py:262** — Same pattern → Fixed with `torch.enable_grad()`

## Conclusion

Codebase demonstrates high engineering and scientific quality. Statistical methods are rigorous with proper multiple testing correction and bootstrap CIs. All backward() calls are in correct contexts. **No new bugs found.**
