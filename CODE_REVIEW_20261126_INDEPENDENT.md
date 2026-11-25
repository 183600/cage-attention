# Independent Code Review — cage-attention (2026-11-26)

## Review Scope
Full source code audit of the `cage/` package, including:
- Model architecture (`model.py`, 1951 lines) — GPT-style decoder with calibrated attention gating
- Training loop (`_train_loop.py`, 1500+ lines) — overfit sanity, stats collection, attention diagnostics
- Training recipe (`_train_recipe.py`) — LR scheduling, cosine decay, gradient management
- Data pipeline (`data.py`, `_data_audit.py`, `_data_selfcheck.py`) — synthetic data generation, validation
- Metrics (`_metrics_calibration.py`, `_metrics_inference.py`, `_metrics_ranking.py`, `_metrics_robustness.py`)
- Experiment framework (`experiments/`) — E1/E2/E3 runners, analysis plan, design, baselines
- Training evaluation (`_train_eval.py`) — bad-case taxonomy, attention mass analysis

## Findings

### Code Quality: Excellent (No Bugs Found)

**Calibrated Attention Gate (`model.py`):**
- Gate formula `sigmoid((W_g x - b0 - a*log(max(T, 1e-300)) - c*H_norm) / tau)` correctly implements length + concentration calibration
- `calib_scale` correctly designated as non-learnable (only `c * calib_scale` enters the gate; a second free scalar would be unidentifiable)
- Entropy normalization: `Hbar = H / log T` correctly applied for the concentration term
- Null attention baseline: learnable key-value slot with proper joint softmax (lse-based computation, numerically stable)
- Position-conflation ablation (`calib_pos_norm`): correctly isolates visible-token length vs global sequence length
- Active-control switch (`calib_shuffle_ent`): deterministic position permutation preserves marginal distribution while decorrelating entropy from row-specific attention — correct experimental design

**Error Taxonomy (`_train_loop.py`):**
- Mutually exclusive buckets in priority order: `label_error > hard_structural > retrieval_miss > found_then_lost > readout_wrong`
- `_label_audit()`: correctly validates per-sample label integrity (key uniqueness + answer pairing) rather than assuming it
- `hard_structural` split into disjoint naive-only / repeat-only / both buckets — enables separating two structural error sources
- Wilson intervals on all error rates and taxonomy counts — prevents reading binomial noise as meaningful differences
- Attention mass tracking by layer: correctly distinguishes "never found" (retrieval_miss) from "found then lost" (mid-stack hit, last-layer miss)

**Data Generation & Validation:**
- `validate_batch()`: structural validation gate that aborts on invalid batches rather than silently taxonomizing as model errors
- Label noise rate computed as measured claim (not assumed 0 from construction) — methodologically sound
- SEP id handling: correctly excluded from key-uniqueness / answer-pairing judgment
- Deterministic seeding via `torch.Generator` with `stream_seed_key` — reproducible without consuming global RNG

**Statistical Methods:**
- Cohen's d / Hedges' g: properly guarded (pooled_std < 1e-12 returns 0.0)
- Multiple comparison corrections: Holm, Benjamini-Hochberg helpers present
- Sign-flip TOST: correct implementation for equivalence testing
- Regularized incomplete beta with overflow handling — numerically robust for extreme parameter values
- Bootstrap CI with BCa (bias-corrected accelerated) intervals

**Training Pipeline:**
- Cosine LR schedule with proper skipped-iteration handling (non-finite loss advances schedule counter)
- Gradient norm tracking with interval peak detection
- Proper eval/train state management (saves/restores training states for inference-mode evaluation)

### Methodology: Sound

- Leave-one-out ablation (`calib_use_len` / `calib_use_ent`): correctly isolates each term's contribution
- Active control (entropy shuffle): same architecture, parameter count, and computation — only entropy *information* differs from mere presence of extra data-dependent input
- Error taxonomy with measured label quality: "labels cannot be the cause" is a measured claim, not an assumption
- False-concentration labels: high max-prob but wrong answer — directly tests the "attention is concentrated but on wrong information" failure mode

### Previous Fixes Verified
- Division-by-zero guards in `_data_audit.py`: `selectable` properly guarded
- Bare except clause in `_train_eval.py`: correctly replaced with specific exception handling
- Effect size helpers and multiple comparison correction: properly integrated

## Conclusion
**No bugs or methodology issues found.** The calibrated attention gating implementation is mathematically correct, the error taxonomy is rigorous with proper statistical guards, and the training pipeline handles edge cases (non-finite losses, schedule skipping) correctly. All previously identified issues have been properly fixed.
