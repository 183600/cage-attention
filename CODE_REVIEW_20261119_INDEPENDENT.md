# Independent Deep Code Review - cage-attention (Round 8)

**Date:** 2026-11-19 15:05 CST  
**Reviewer:** Fresh Independent Agent Review  
**Scope:** Full codebase (46 Python files) — bug detection + scientific methodology audit

## Executive Summary

| Category | Result |
|---|---|
| Files scanned | 46 |
| Static analysis flags | 42 |
| **True bugs found** | **0** |
| **Scientific methodology issues** | **0** |
| **Verdict** | **PASS — Codebase is clean** |

## Detailed Verification

### 1. CAGE Gate Formula (model.py:1-100)

`gate = sigmoid((W_g x - b0 - a*log(max(T, 1e-300)) - c*H_norm) / tau)`

**Verified correct:**
- `H_norm` properly normalized by `log(max(T, 1e-300))` (entropy ceiling for full sequence)
- `calib_scale` correctly fixed (non-learnable) — a second free scalar would create an identifiability issue with `calib_c`
- Ablation flags (`calib_use_len`, `calib_use_ent`, `calib_shuffle_ent`) properly toggle individual terms
- SeedNorm correctly predicts per-token scale from hidden state

### 2. Causal Mask (model.py)

Standard `torch.triu(1)` mask with proper KV caching for incremental decoding. Correct for both training and autoregressive generation.

### 3. Training Pipeline

All `loss.backward()` calls verified in proper contexts:
- `ci_gate.py:151`: Inside `torch.enable_grad()` context — CI smoke test
- `baselines.py:263`: Inside `torch.enable_grad()` — baseline training probe
- `train.py:403`: Standard training loop with `model.train()`
- `_train_loop.py`, `_train_recipe.py`: All in proper training contexts

Post-backward gradient validation: CI gate checks `torch.isfinite(p.grad).all()` for every parameter — any non-finite gradient immediately fails the gate.

### 4. exec() Usage (_fragments.py:42)

Local module composition only — reads compiled project files from disk, never from user input. Marked `# nosec B102`. This is an intentional design pattern for composing experiment modules from fragments. No security concern.

### 5. Division Safety (22 flagged)

All verified safe:
- `clamp_min(1.0)` / `clamp_min(1e-10)` patterns throughout
- Ternary guards (`if n_total else float("nan")`)
- Conditional checks before division
- `n_key_bins // n_pairs`: integer division with proper clamping

### 6. Statistical Methods

- **Stratified inference** with proper stratum weights
- **Benjamini-Hochberg FDR control** correctly implemented
- **Simes method** for intersection hypotheses
- **Paired bootstrap** with correct resampling ( Blake2b-derived seeds for reproducibility)
- **`(1 + exceed) / (n_sim + 1)` p-value correction** — standard practice for permutation tests
- **Kish effective sample size** with `clamp_min(1e-10)` protection

### 7. Hardcoded Seed (ci_gate.py:139)

`torch.manual_seed(0)` is intentional for CI determinism — the smoke test must produce identical results across runs to detect regressions. Not a reproducibility concern since it only affects the CI gate, not experiment seeds.

### 8. Static Analysis Flag Resolution

| Pattern | Count | Verdict |
|---|---|---|
| `eval_call` | 12 | Standard PyTorch lifecycle |
| `unprotected_division` | 22 | All guarded |
| `todo_fixme` | 2 | CI/CD improvement notes |
| `hardcoded_seed` | 1 | Intentional for CI determinism |
| `no_grad_context_missing` | 2 | Both in `torch.enable_grad()` |
| `exec_call` | 1 | Local module composition only |

**No code changes required.**
