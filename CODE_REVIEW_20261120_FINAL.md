# Code Review Report - cage-attention
**Date:** 2026-11-20  
**Reviewer:** Automated Deep Review Agent  
**Scope:** Full codebase (46 Python files) + Research Methodology

## Executive Summary

After comprehensive static analysis and deep methodology review, **no bugs or research methodology issues requiring code changes were found**. The codebase demonstrates high-quality implementation with proper statistical methodology and thorough self-checking infrastructure.

## Static Analysis Results

- **Files scanned:** 46
- **Issues flagged by static analysis:** 39
- **True bugs found:** 0
- **All flagged issues classified as false positives**

## Detailed Analysis

### 1. eval_call pattern (12 instances) — FALSE POSITIVE
All `model.eval()` / `m.eval()` calls are correctly placed in evaluation contexts:
- `ci_gate.py:155` — After smoke test training, before validation
- `_train_eval.py:283` — In evaluation function
- `_train_loop.py:132,709` — Before validation phases
- `_train_recipe.py:611,657,909,1097` — Before evaluation steps
- `baselines.py:279` — After linear probe training, before evaluation
- `efficiency.py:145` — Before throughput measurement
- `train.py:567` — After training completion

### 2. unprotected_division pattern (20 instances) — FALSE POSITIVE
All divisions have proper guards:
- `_data_audit.py:557` — Integer division with `n_pairs` guaranteed > 0
- `_data_audit.py:698` — `counts.sum().clamp_min(1.0)`
- `_metrics_inference.py:1572` — `counts_used.clamp_min(1e-10)` in Kish denominator
- `_metrics_ranking.py:245` — `counts.clamp_min(1.0)`
- `_train_eval.py:190` — `if n_items else float("nan")`
- `_train_loop.py` — Various clamped/guarded divisions
- `baselines.py:632` — `if n_per_seed > 0 else float("nan")`
- `model.py:595` — Ceiling division with `n_chunks >= 2` guaranteed
- `experiments/_training.py:423,508` — `if total_counts > 0` / `if counts.sum().item() > 0`

### 3. no_grad_context_missing pattern (3 instances) — FALSE POSITIVE
- `ci_gate.py:151` — Inside explicit `torch.enable_grad()` context
- `baselines.py:263` — Inside explicit `torch.enable_grad()` context
- `train.py:403` — In training loop with `model.train()` active, no outer `no_grad` context

### 4. hardcoded_seed pattern (1 instance) — NOT A BUG
- `ci_gate.py:139` — `torch.manual_seed(0)` in CI smoke test; hardcoded seed is intentional for reproducibility of the gate check

### 5. exec_call pattern (1 instance) — INTENTIONAL
- `_fragments.py:42` — `exec()` with `# nosec B102` comment; intentional design for composing module fragments from local project files only

## Research Methodology Review

### Statistical Inference
- **Paired bootstrap CI:** Correctly implements percentile bootstrap with +1 smoothing for two-sided p-values
- **Kish-style effective sample size:** Properly computed as `1 / Σ(w_i²/n_i)` for weighted stratified estimates
- **Stratified inference:** Reference weights normalized correctly; variance formula `Σ W_i² p_i(1-p_i)/n_i` is standard

### Calibration
- Pre-registered decision rules properly calibrated with family-wise error control
- Benjamini-Hochberg and Holm corrections correctly implemented
- Sign-flip test properly assumes symmetry under null

### Training Pipeline
- Proper `zero_grad(set_to_none=True)` before each step
- Non-finite loss detection with history logging
- Gradient clipping after backward pass

### Data Integrity
- `_data_selfcheck.py` correctly validates stratification geometry
- Joint-difficulty shift test properly designed (same marginals, reversed dependence)
- Batch validation with cleaning gates

## Conclusion

The cage-attention codebase is well-engineered with robust statistical methodology and comprehensive self-checking. All static analysis flags are false positives. No code changes are required.
