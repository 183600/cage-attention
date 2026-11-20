# Independent Code Review Report — cage-attention
**Date:** 2026-11-20 13:07 CST
**Reviewer:** Independent Sub-Agent Review (V3)
**Scope:** Full codebase — 46 Python files — bug detection + research methodology audit

## Executive Summary

| Metric | Value |
|--------|-------|
| Status | **PASS** |
| Real bugs found | 0 |
| False positives from static analysis | 39 |
| Code changes required | 0 |
| Code quality rating | Excellent |

## Analysis Performed

### 1. Training Loop
- Gradient accumulation: `loss / grad_accum` scaling before backward, `zero_grad(set_to_none=True)` before accumulation. Correct ordering.
- Non-finite loss detection with history logging and ordered triage.
- Gradient clipping applied after backward, before optimizer step.
- `model.train()` / `model.eval()` mode switching correctly placed.

### 2. Statistical Inference
- **Paired bootstrap CI**: Percentile bootstrap with +1 smoothing for two-sided p-values: `2 * (min(p_lo, p_hi) * n_boot + 1) / (n_boot + 1)`. Standard formula.
- **Kish effective sample size**: `1 / Σ(w_i²/n_i)` for weighted stratified estimates. Properly handles small strata with `clamp_min(1e-10)`.
- **Stratified inference**: Reference weights normalized. Variance formula `Σ W_i² p_i(1-p_i)/n_i` is standard for stratified proportion estimation.
- **Sign-flip test**: Exact (or Monte-Carlo) companion to bootstrap. Properly assumes symmetry under null.
- **Multiplicity control**: Benjamini-Hochberg and Holm corrections correctly implemented.

### 3. Data Integrity
- `_data_selfcheck.py`: Validates stratification geometry. Joint-difficulty shift test properly designed.
- Batch validation with cleaning gates.
- `_data_audit.py`: Key-position binning with guaranteed positive divisor.

### 4. Model Implementation
- `ci_gate.py`: CI smoke test with hardcoded seed (intentional for reproducibility). Forward/backward validation across all gate variants.
- `_fragments.py`: `exec()` for modular experiment composition from local project files only. `# nosec B102` annotation with justification.

### 5. Static Analysis False Positives (39 total)
All 39 flagged issues verified as false positives:
- `eval_call` (12): Correctly placed in evaluation contexts
- `unprotected_division` (20): All guarded by conditionals, clamp, or prior validation
- `no_grad_context_missing` (3): Inside explicit `enable_grad()` or training loop
- `hardcoded_seed` (1): Intentional for CI reproducibility
- `exec_call` (1): Intentional design for module composition
- `todo_fixme` (2): Known limitations, not bugs

## Conclusion
The cage-attention codebase is well-engineered with robust statistical methodology. No bugs or methodology issues found. No code changes required.
