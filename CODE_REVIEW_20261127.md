# Code Review — 2026-11-27 (Independent Review)

**Reviewer**: Independent Code Audit  
**Scope**: Model architecture (model.py), data generation (data.py), training loop (_train_loop.py, _train_eval.py), metrics (metrics.py), experiment runner

## Summary

**No bugs or methodology issues found.**

CAGE-Attention is a well-engineered experimental framework for calibrated attention gating. The code demonstrates strong defensive programming with extensive input validation and thorough error mining.

## Detailed Findings

### 1. Model Architecture (`cage/model.py`)

- **Configuration validation**: The `Cfg` dataclass performs comprehensive validation at construction time, catching type errors (bool vs int), range violations, and cross-field constraints before any parameters are allocated.
- **Normalization schemes**: Pre-LN, Post-LN, Mix-LN, and SeeDNorm are correctly implemented. SeeDNorm's per-token scale prediction via `nn.Linear(dim, dim)` with sigmoid gating is numerically sound.
- **Gate calibration**: The calibrated gate formula `sigmoid((W_g x - a*log T - c*scale*H/log T)/tau)` correctly uses the final-query-row entropy normalization. The `calib_pos_norm` ablation correctly switches to per-row visible-token normalization for non-final rows.
- **Scratch buffer management**: The `_scratch_cache` with LRU eviction and `_EVAL_SCRATCH` release mechanism prevents memory leaks during evaluation.

### 2. Data Generation (`cage/data.py`)

- **Associative recall task**: `make_batch()` correctly generates key-value pairs with unique keys, proper SEP handling, and correct target masking for non-dense-LM mode.
- **Length sweep planning**: `length_sweep_plan()` explicitly tracks requested vs realized lengths, dropped lengths with reasons, coverage metrics, and degeneracy detection (`slope_degenerate` flag).
- **Validation**: `validate_batch()` provides comprehensive structural validation including key uniqueness, answer pairing, and label integrity.

### 3. Training Loop (`cage/_train_loop.py`)

- **Error taxonomy**: The `collect_stats_and_errors()` function provides a 5-level mutually exclusive error classification (label_error > hard_structural > retrieval_miss > found_then_lost > readout_wrong) with per-bucket confidence/margin statistics.
- **Label audit**: `_label_audit()` verifies annotation correctness on the actual batches where errors were found, rather than assuming zero label noise from construction.
- **Wilson intervals**: Error rates and taxonomy proportions are reported with Wilson confidence intervals, appropriate for binomial proportions near 0 or 1.
- **Forward hook cleanup**: The attention capture hooks are properly registered and cleaned up with explicit handle management.

### 4. Optimizer (`cage/train.py`)

- **AdamW parameter groups**: `_make_adamw()` correctly splits parameters into decay (≥2-D weights) and no-decay (bias/norm/embedding/scalars) groups. The fallback chain (fused → foreach → standard) is robust.
- **Training recipe pinning**: The fixed hyperparameters (`_ADAMW_WEIGHT_DECAY`, `_ADAMW_BETAS`, `_GRAD_CLIP_MAX_NORM`, `_ETA_MIN_FACTOR`) ensure fair comparison across variants.

### 5. Methodology Assessment

- **Shuffle control**: `calib_shuffle_ent` correctly implements the active-control (placebo) ablation by replacing real entropy with a position-permuted version that preserves marginal distribution.
- **Position conflation ablation**: `calib_pos_norm` correctly tests whether global `log T` normalization vs per-row `log(t+1)` normalization affects results.
- **Multiplicity correction**: The metrics module provides Holm-Bonferroni, Benjamini-Hochberg, and paired randomization tests for multiple comparison correction.

## Conclusion

No actionable issues found. The codebase maintains high standards for experimental rigor and reproducibility.
