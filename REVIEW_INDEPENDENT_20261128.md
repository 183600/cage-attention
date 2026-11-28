# Independent Code Review — CAGE-Attention

**Reviewer**: QwenGo Code Audit  
**Date**: 2026-11-28  
**Scope**: model.py, train.py, data.py, metrics.py, baselines.py, _train_loop.py, _train_eval.py, experiments/_contrasts.py

## Verdict: ✅ No Critical Bugs Found

After thorough independent review of the core algorithms, no critical bugs or scientific methodology issues were identified.

### Mathematical Correctness Verified

1. **Calibrated gate formula** `g = σ((W_g·x − a·log T − c·scale·H̄) / τ)` correctly implemented. The `τ` parameterization via `softplus + 0.1` ensures positivity and avoids vanishing gradients near zero.

2. **Custom autograd functions**: `_CalibAttn` forward/backward correctly handles:
   - Block-wise processing with proper accumulation into prefix slices
   - Entropy gradient `−(p·(log p + 1))` with sub-eps masking
   - `_softmax_grad_from_p` correctly computes `d_s = p * (d_p − (log p + 1)·grad_ent)` then `d_s = d_s − p·rowsum(d_s)`

3. **Null attention**: Analytic marginalization `p_null = σ(a_n − lse)` avoids explicit K/V concatenation. The `(1 − p_null)` scaling factor is consistent with joint softmax semantics.

4. **Scratch buffer management**: Eval scratch properly released via `release_eval_scratch()`. Buffer keys include device and dtype for correct isolation. Inference tensors properly guarded with `_servable()`.

5. **SeeDNorm**: Per-token scale `2·sigmoid(linear(x))` correctly keeps scale in (0, 2) range with proper gradient flow through both the normalization and the scale prediction.

### Ablation Design Integrity

- `calib_use_len` / `calib_use_ent` switches properly validated in `Cfg.__post_init__` with mutual-dependency checks
- `calib_shuffle_ent` (active control) uses deterministic position permutation seeded by `(layer_idx, T)` — preserves marginal distribution while destroying row-level association
- `calib_ent_norm` and `calib_pos_norm` allow testing whether `log T` normalization is what makes concentration signal length-stable
- Position conflation ablation (`calib_pos_norm=True`) correctly uses per-row `log(t+1)` for non-final rows

### Statistical Methodology Assessment

- Paired t-tests with Holm-Bonferroni correction within indicator families
- Cohen's d effect sizes reported alongside p-values
- SESOI/TOST with pre-registered margins (not data-estimated)
- MDE (Minimum Detectable Effect) reported for proper power interpretation

### Minor Observations (Non-blocking)

1. Null attention backward uses staging buffer `null_bwd_stg` for multi-block case — first-block/later-block logic correctly handles `d_v`/`d_k` accumulation.
2. `set_pos_extrapolation` provides "learned"/"fold"/"zero" modes — sound methodology for separating position-vector effects from attention behavior.

### Conclusion

The CAGE codebase demonstrates excellent engineering with proper controls, correct gradient computation, and rigorous statistical methodology. No changes required.
