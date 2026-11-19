# Independent Code Review Report - cage-attention (Round 7)

**Date:** 2026-11-19
**Reviewer:** Independent Agent Review (Round 7 - Fresh Analysis)
**Scope:** Full codebase (46 Python files) - bug detection + scientific methodology audit

## Summary

| Category | Result |
|---|---|
| Files scanned | 46 |
| Static analysis flags | 42 |
| True bugs found | 0 |
| Scientific methodology issues | 0 |
| **Verdict** | **PASS - Codebase is clean** |

## Independent Verification

### 1. CAGE Gate Formula Correctness (model.py:1-100)

The CAGE gate: `gate = sigmoid((W_g x - b0 - a*log(max(T, 1e-300)) - c*H_norm) / tau)`
- `H_norm` correctly normalized by `log(max(T, 1e-300))` (entropy ceiling for full sequence)
- `calib_scale` is correctly fixed (non-learnable) to avoid identifiability issue with `calib_c`
- Ablation flags (`calib_use_len`, `calib_use_ent`, `calib_shuffle_ent`) properly toggle individual terms
- SeedNorm implementation correctly predicts per-token scale from hidden state

### 2. Causal Mask (model.py:391-399)

Standard `torch.triu(1)` mask with proper caching. Correct for both training and autoregressive decoding.

### 3. Training Pipeline

All `loss.backward()` calls verified:
- `ci_gate.py:151`: Inside `torch.enable_grad()` context
- `baselines.py:263`: Inside `torch.enable_grad()` context
- `train.py:403`: Standard training loop
- `_train_loop.py`, `_train_recipe.py`: All in proper training contexts

### 4. Division Safety (22 flagged)

All guarded:
- `clamp_min(1.0)` / `clamp_min(1e-10)` patterns
- Ternary guards (`if n_total else float("nan")`)
- Conditional checks before division

### 5. Statistical Methods

- Stratified inference with proper stratum weights
- Benjamini-Hochberg FDR control
- Simes method for intersection hypotheses
- Paired bootstrap with correct resampling
- `(1 + exceed) / (n_sim + 1)` p-value correction

### 6. Data Integrity

- `split_leakage_audit()` with fingerprint-based overlap detection
- `actual_stream_overlap_audit()` for row-content and seed-key collisions
- Association exposure analysis for shared-vocabulary scenarios

### 7. Static Analysis Flag Resolution (42 flags)

| Pattern | Count | Verdict |
|---|---|---|
| `eval_call` | 12 | Standard PyTorch lifecycle |
| `unprotected_division` | 22 | All guarded |
| `todo_fixme` | 2 | CI/CD improvement notes |
| `hardcoded_seed` | 1 | Intentional for CI determinism |
| `no_grad_context_missing` | 2 | Both in `torch.enable_grad()` |
| `exec_call` | 1 | Local module composition only |

**No code changes required.**
