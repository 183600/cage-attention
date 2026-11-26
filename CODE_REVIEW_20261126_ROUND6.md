## Code Review 2026-11-26 (Sixth Round)

### Summary

Round 6 deep code review of the cage-attention repository. Focused on:

1. **Causal masking correctness**: Verified `_causal_mask` and
   `_causal_null_mask` implementations.
2. **Training loop integrity**: Checked gradient accumulation, loss
   computation, optimizer step ordering.
3. **Metrics calibration**: Reviewed calibration metrics, ranking
   metrics, and robustness metrics.
4. **Data audit pipeline**: Verified data self-checks and audit
   protocols.
5. **Experiment runner**: Checked experiment configuration, gates,
   and contrast logging.

### Findings

**No bugs found.** All automated warnings were verified as false
positives:

- **`model.eval()` without restore** (train_recipe.py:1099,
  baselines.py:281, train.py:568): All follow the save-eval-restore
  pattern with `was_training` flag.
- **Softmax dim** (_train_loop.py:255): `.softmax(-1)` correctly
  passes dim as positional argument.
- **Collection mutation** (_training.py:768): `dict.pop()` in
  comprehension context, not mutation during iteration.

### Methodology Assessment

- **Causal mask**: `_causal_mask(T)` creates upper-triangular `-inf`
  mask via `m.triu_(1)`, correctly blocking future positions.
- **Null attention slot**: `_causal_null_mask(T)` shape `(T, T+1)`
  with column 0 (null) unmasked and columns 1..T using standard
  causal mask. Correct.
- **SeedNorm**: Dynamic per-token normalization with proper scratch
  buffer management. No gradient leakage between batch elements.
- **Training stationarity**: Diagnostic correctly splits train data
  into halves and tests for distribution shift.

### Conclusion

The cage-attention codebase is well-engineered with proper causal
attention, null slot handling, and comprehensive metrics. No changes
required.
