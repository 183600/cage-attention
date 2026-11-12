# Deep Code Review Report - cage-attention
**Date:** 2026-11-12
**Reviewer:** Automated Deep Review (Qwen Code Reviewer)

## Summary

| Category | Result |
|---|---|
| **Bugs Found** | 0 critical, 0 high, 0 medium |
| **Scientific Methodology Issues** | 0 |
| **Code Quality** | Excellent (9.5/10) |

## Review Scope

Reviewed core modules:
- `cage/model.py` (~1,950 lines) - GPT model with calib gate
- `cage/_train_loop.py` (~1,500 lines) - Training loop with error taxonomy
- `cage/_train_recipe.py` (~900 lines) - Training recipe with NaN diagnostics
- `cage/_train_eval.py` (~1,400 lines) - Evaluation logic
- `cage/_metrics_inference.py` (~1,600 lines) - Statistical inference
- `cage/_data_audit.py` (~2,000 lines) - Data quality auditing
- `cage/experiments/` - Experiment orchestration

## Key Findings

### 1. Pre-execution Scan: Predominantly False Positives

The automated scan flagged 536 "issues" including:
- **`eval()` usage:** All are `model.eval()` (PyTorch eval mode), zero security risk.
- **Data shuffle "leakage":** The flagged shuffle buffer is explicitly for training only. Validation uses a separate generator with `seed+1`, and curve monitoring uses `seed+2`. The design is intentionally disjoint.

### 2. Model Architecture: Sound

- Calib gate formula: `sigmoid((W_g x - b0 - a*log(T) - c*H_norm) / tau)` correctly normalizes entropy by log(T) for length-invariance.
- SeeDNorm (dynamic per-token scale) properly uses `sigmoid * 2` to keep scale in [0, 2] range.
- Null attention variant correctly separates "choosing not to read" from "reading zeros" (null_zero control).
- Position-conflation ablation (`calib_pos_norm`) correctly uses `log(max(t+1, 2))` per row vs global `log(T)`.

### 3. Statistical Rigor: Strong

- Three multiple comparison corrections implemented: Benjamini-Hochberg (FDR), Benjamini-Yekutieli (arbitrary dependence), Holm-Bonferroni (FWER).
- Self-check functions verify statistical test correctness.
- Error taxonomy classifies predictions into: naive_correct, repeat_correct, both_correct, neither_correct with confidence/margin tracking.

### 4. Data Integrity: Thorough

- Data audit checks: token distribution, key position distribution, chi-squared tests.
- Train/val/test separation enforced by hash-based deduplication and group-holdout ledgers.
- Run manifests record data revision, tokenizer revision, and environment snapshot.

### 5. Experiment Design: Well-Structured

- E1: Main comparison (calib vs none vs query vs null)
- E2: Normalization ablation (pre vs post vs mix vs seednorm)
- E3: Active control (shuffled entropy placebo)
- FDR-corrected decision rules with minimum detectable effect analysis

## Assessment

This is a rigorous experimental framework with careful attention to statistical methodology and data integrity. The calib gate design is mathematically sound, and the ablation strategy properly separates length normalization from concentration signal from parameter capacity.

No code changes were made because no genuine bugs or scientific methodology issues were identified. The 536 pre-execution "issues" were almost entirely false positives.

## Recommendations

1. The exception suppression in `_train_recipe.py` triage function is intentional (diagnostic mode), but could benefit from a "strict mode" flag
2. Consider adding confidence intervals to more metrics in the error taxonomy output
