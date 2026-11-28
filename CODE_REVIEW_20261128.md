# Code Review — 2026-11-28

## Summary
Independent review of cage-attention core modules (model.py, experiments/).

## Findings

### No Critical Bugs Found

The CAGE codebase implements calibrated attention gating with entropy-based concentration and length correction. Key findings:

1. **Calibrated gate formula**: `gate = sigmoid((W_g x - a*log T - c*scale*H/log T) / tau)` correctly implemented. The `tau` parameterization via `softplus + 0.1` ensures positivity.

2. **Custom autograd functions**: `_CalibAttn` and `_NullAttn` correctly implement forward/backward with block-wise processing. The entropy gradient `_softmax_grad_from_p` correctly handles the `-(p*log p)` backward via `log(max(p, eps)) + 1` with proper sub-eps masking.

3. **Scratch buffer management**: Eval scratch properly released via `release_eval_scratch()`. Buffer keys include device and dtype for correct isolation.

4. **Ablation switches**: `calib_use_len`, `calib_use_ent`, `calib_shuffle_ent`, `calib_ent_norm`, `calib_pos_norm` are properly validated in `Cfg.__post_init__` with clear mutual-dependency checks.

### Minor Observations

1. **Null attention backward**: Uses staging buffer `null_bwd_stg` for multi-block case. The first-block/later-block logic correctly handles `d_v`/`d_k` accumulation.

2. **Position extrapolation**: `set_pos_extrapolation` provides "learned"/"fold"/"zero" modes — sound methodology for separating position-vector effects from attention behavior at longer ranges.

3. **SeedNorm**: Per-token scale predicted from hidden state with `sigmoid * 2` — correctly keeps scale in (0, 2) range.

## Methodology Assessment

- **Active control (placebo)**: `calib_shuffle_ent` uses deterministic position permutation seeded by `(layer_idx, T)` — preserves marginal distribution while destroying row-level association. Sound experimental control.
- **Length normalization ablation**: `calib_ent_norm` and `calib_pos_norm` allow testing whether `log T` normalization is what makes concentration signal length-stable.
- **Position conflation**: `calib_pos_norm=True` uses per-row `log(t+1)` instead of global `log T` — correctly identifies the confound on non-final rows.

## Verdict
✅ No code changes required. Codebase is well-designed and correctly implemented.
