# Code Review Report - cage-attention
## Pass 6: Independent Deep Source Code Audit

**Date:** 2026-11-14 05:00 UTC
**Reviewer:** QwenGo Independent Review System (Pass 6)
**Scope:** Full independent source code audit — focus on attention computation,
custom autograd functions, training loop, and data pipeline integrity.

---

## Summary

### Verdict: CODE QUALITY CONFIRMED — No new issues found

This pass performed direct source-code reading of all core modules, with
particular focus on:
- `cage/model.py` — `_CalibAttn` autograd function, `_NullAttn`, `_SoftmaxInPlace`,
  `_RowEntropy`, `Cfg` dataclass validation, gate computation
- `cage/train.py` — training loop with gradient accumulation, early stopping,
  non-finite handling, label permutation negative control
- `cage/_data_task.py` — train/val/test split protocol, seed management
- Previous review verification (Passes 1-5: all clean)

---

## Detailed Checks Performed

### 1. _CalibAttn Autograd Function (model.py:684-850)
**Status:** ✅ CORRECT

- **Forward pass:** Uses pre-allocated scratch buffers for in-place operations,
  correctly separating gradient-required and no-gradient paths. The `ybt`
  layout path is restricted to no-gradient (enforced by explicit check).
- **Backward pass:** Recomputes softmax probabilities from saved `qkv` input
  (not from saved softmax output) — this is numerically correct and memory
  efficient. The softmax gradient `_softmax_grad_from_p` correctly implements
  `d_s = p * (d_p - sum(d_p * p))` with optional entropy gradient injection.
- **Block processing:** Multi-block backward correctly accumulates `d_v` and
  `d_k` using `add_` for prefix slices, with the first block using `out=`
  (overwrite) and subsequent blocks using `add_` (accumulate).
- **Entropy gradient:** The `log(max(p, eps)) + 1` term for entropy gradient
  correctly handles masked entries (`p == 0`) via the `p > eps` mask, and
  `eps` clamping prevents `log(0)`.

### 2. _NullAttn (model.py:940+)
**Status:** ✅ CORRECT

- Null key logit computation properly joins null slot into softmax via
  max-stable numerics: `max_val = max(real_max, null_score)`, then
  `exp(s - max_val)` for both real and null entries.
- The null slot's attention weight `p_null` is correctly subtracted from
  the output scaling, ensuring the attention output is `p_real @ v`
  (not `(p_real + p_null) @ v`).

### 3. Configuration Validation (Cfg dataclass)
**Status:** ✅ ROBUST

- Boolean coercion prevention: explicit `isinstance(x, bool)` checks prevent
  `"False"` (truthy string) from being accepted as a boolean.
- Integer overflow protection via `operator.index()` and `_MAX_TORCH_INDEX`.
- Ablation switches validated as semantic booleans.
- All gate/norm combinations properly validated at construction time.

### 4. Training Loop (train.py:101+)
**Status:** ✅ CORRECT

- **Gradient accumulation:** Not used in single-step training; loss computed
  per batch without scaling (correct for single-batch updates).
- **Non-finite handling:** Pre-clip gradient norm checked for finiteness
  before optimizer step. NaN/Inf losses can either raise or skip based on
  `nonfinite` parameter.
- **Label permutation:** Dedicated generator seeded from `seed` (not from
  data generator), ensuring permuted runs see identical batches.
- **Early stopping:** `steps_run` counts only iterations reaching
  `opt.step()`, so skipped non-finite updates aren't miscounted.
- **Eval mode restoration:** `model.eval()` / `model.train()` properly
  saved and restored in `finally` blocks.

### 5. Data Split Protocol (_data_task.py)
**Status:** ✅ NO LEAKAGE

- Fixed split-by-seed design: each split owns a dedicated RNG stream.
- Reserved eval seeds (VAL_SEED=123, TEST_SEED=2024, STATS_SEED=7)
  with `check_train_seed` preventing training on eval streams.
- Seed key truncation to 32 bits documented and handled via
  `stream_seed_key()` to prevent aliasing.
- Split-before-generation ensures no post-hoc partitioning.

## Conclusion

The cage-attention codebase demonstrates production-quality implementation
with careful attention to autograd correctness, numerical stability, and
research methodology rigor. The custom attention kernels correctly handle
both forward and backward passes with proper memory management. No bugs or
methodology errors were found. No code changes were necessary.
