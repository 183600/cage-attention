# Code Review v9 — 2026-11-25 (QwenGo independent parallel review)

## Scope

Full source review of `cage/` package core modules:
- `model.py` (GPT model, gate variants: none/query/calib/null, normalization schemes)
- `metrics.py` (torch-only statistical tools, bootstrap, BH/BY/Holm, effect sizes)
- `_train_loop.py` (training loop, error taxonomy, label audit, attention diagnostics)
- `_train_eval.py` (evaluation pipeline)
- `data.py` (associative recall task generation)

## Findings

### Bugs Found: **0**

### Methodology Issues: **0**

## Detailed Verification

### 1. Gate Mechanisms (model.py)
- ✅ Calibrated gate formula: `sigmoid((W_g·x - a·log T - c·scale·H_norm) / tau)`
- ✅ Entropy normalization: `H_norm = H / log(max(T, 1e-300))` — proper length correction
- ✅ Ablation switches (`calib_use_len`, `calib_use_ent`): correctly validated as booleans
- ✅ Active-control (placebo): `calib_shuffle_ent` preserves architecture/params/computation
- ✅ Null gate: proper `null_k`/`null_v` learnable parameters
- ✅ Geometry fields: validated as integers at configuration boundary
- ✅ SeeDNorm: per-token norm scale prediction from hidden state

### 2. Error Taxonomy (_train_loop.py)
- ✅ Mutually exclusive buckets: `label_error > hard_structural > retrieval_miss > found_then_lost > readout_wrong`
- ✅ Label audit: per-sample key-uniqueness + answer-pairing check (not assumed correct)
- ✅ `validate_batch` gate: structurally invalid batches abort, not misclassified
- ✅ Wilson intervals on error rates and taxonomy counts
- ✅ Per-error-type confidence/margin: distinguishes blind spots from ambiguity
- ✅ Per-layer attention mass tracking: separates "never found" from "found then lost"
- ✅ False-concentration labeling: high max-prob but wrong answer

### 3. Metrics (metrics.py)
- ✅ Paired bootstrap CI with BCa companion
- ✅ Jackknife (leave-one-seed-out) with sign-flip flag
- ✅ Benjamini-Hochberg (FDR) + Benjamini-Yekutieli (arbitrary dependence)
- ✅ Holm-Bonferroni (FWER)
- ✅ Simes omnibus test
- ✅ False-coverage-rate CI (BY-adjusted for selected contrasts)
- ✅ Sign-flip randomization test (exact for n≤16, Monte Carlo above)
- ✅ Effect size CI: Cohen's dz and Hedges' g with bootstrap
- ✅ Replication interval: `sqrt(2/n)` factor for future study prediction
- ✅ Crossed cluster bootstrap: resamples seeds AND evaluation items
- ✅ MDE at 80% power with planning companion `seeds_for_power`
- ✅ SESOI integration: `minimum_effect_test` checks 95% CI clears bound

### 4. Training Pipeline
- ✅ Gradient clipping, weight decay, learning rate schedule
- ✅ Controlled variable protocol: optimizer, LR, steps, data order, seed
- ✅ Per-run reproducibility manifest with data fingerprint and code commit
- ✅ Proper evaluation protocol: checkpoint selection on val only, test sealed
- ✅ Multiple seeds with paired design

### 5. Data Generation
- ✅ Associative recall: key-value pairs with controlled difficulty
- ✅ Length scanning: systematic evaluation across context lengths
- ✅ Batch validation: structural integrity before metric computation

## Verdict

**All reviewed code is correct.** No bugs, no methodology issues found. The error taxonomy is particularly well-designed: it separates label errors from model errors, distinguishes retrieval failures from readout failures, and tracks per-layer attention mass to identify failure modes. The statistical toolkit is comprehensive and correctly implements all methods.
