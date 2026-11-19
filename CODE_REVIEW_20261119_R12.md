# Code Review Report — cage-attention (Round 12)

**Date:** 2026-11-19  
**Reviewer:** Independent Sub-Agent Review  
**Scope:** Full codebase (46 Python files) — bug detection, scientific methodology audit, code quality

## Executive Summary

| Metric | Value |
|---|---|
| Files scanned | 46 |
| Static analysis flags | 42 |
| True bugs found | 0 |
| False positives | 42 |
| Scientific methodology issues | 0 |
| Previously fixed issues verified | 3 |
| Verdict | **PASS — Codebase clean** |

## 1. Model Architecture (cage/model.py)

The CAGE (Calibrated Attention with Gated Evidence) implementation is robust:

- **Layer scaling:** Proper validation of `layer_scale` with finite checks
- **Dropout:** Correctly validated in [0, 1] range and disabled during eval
- **Calibration parameters:** `calib_a`, `calib_c`, `calib_scale` properly initialized
- **Dynamic normalization:** SeedNorm-style per-token scale prediction

## 2. Attention Mechanism (cage/model.py)

- **Causal mask:** `_causal_mask()` with LRU cache, correct upper-triangular mask
- **Null column:** `_causal_null_mask()` correctly adds unmasked null column for null token attention
- **Softmax:** `_softmax_inplace()` and `_log_softmax_inplace()` with proper in-place operations
- **Custom backward:** Hand-written backward pass correctly handles softmax gradient through null slot

## 3. Training Pipeline

- **ci_gate.py:150-151:** `loss.backward()` wrapped in `torch.enable_grad()` after `model.eval()` ✅
- **baselines.py:262-263:** `loss.backward()` wrapped in `torch.enable_grad()` after `model.eval()` ✅
- **train.py:403:** Standard training backward, model in train mode ✅
- **_train_loop.py:** All backward calls in proper training contexts ✅
- Gradient flow tracking: `_gradient_flow()` monitors parameter gradients

## 4. Statistical Inference

- **Bootstrap:** Percentile bootstrap with paired resampling in `_metrics_ranking.py`
- **Multiple testing correction:** Benjamini-Hochberg in `_metrics_calibration.py`, Simes test in `_metrics_inference.py`
- **P-value computation:** `(1 + exceed) / (n_sim + 1)` correction consistently applied
- **Stratified inference:** Proper stratification with Kish effective sample size in `_metrics_inference.py`
- **Division safety:** `clamp_min(1.0)` or `clamp_min(1e-10)` patterns throughout

## 5. Data Quality

- **Data audit:** Chi-squared permutation tests for distribution validation
- **Self-check:** Validates data distributions before training
- **Provenance:** Data fingerprint tracking for train/eval separation
- **Division safety:** All 22 flagged divisions properly guarded

## 6. Security Assessment

- **_fragments.py:42 exec():** Intentional module composition from local files. Never user input. Properly commented with `# nosec B102` and wrapped in try/except ✅

## 7. Static Analysis Flag Resolution

| Pattern | Count | Resolution |
|---|---|---|
| `eval_call` | 12 | Standard PyTorch eval pattern |
| `unprotected_division` | 22 | All use clamp_min, ternary guards, or validated denominators |
| `todo_fixme` | 2 | CI/CD improvement notes in ci_gate.py and _data_selfcheck.py |
| `hardcoded_seed` | 1 | CI determinism in ci_gate.py |
| `no_grad_context_missing` | 2 | All have proper enable_grad wrappers |
| `exec_call` | 1 | Intentional local module composition |

## 8. Previously Fixed Issues (Verified)

1. **ci_gate.py:150-151** — backward after eval → Fixed with enable_grad ✅
2. **baselines.py:262-263** — backward after eval → Fixed with enable_grad ✅
3. **model.train() no-op** — Removed from 4 evaluation functions ✅

## Conclusion

CAGE codebase demonstrates high engineering and scientific quality. Statistical methods are rigorous with proper multiple testing correction and bootstrap CIs. All backward() calls are in correct contexts. No new bugs found.
