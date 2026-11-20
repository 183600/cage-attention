# Independent Code Audit — 2026-11-21 (v2)

## Scope
Full review of cage-attention repository: static analysis triage, statistical methodology validation, numerical stability audit, and scientific methodology review.

## Static Analysis Triage
No high-priority issues flagged. 77 low-priority style/minor issues reviewed — no actionable bugs.

## Numerical Stability
- ✅ All division operations protected via `clamp_min()` or conditional checks
- ✅ Chi-squared computation uses `expected.clamp_min(1e-10)` to prevent division by zero
- ✅ Gradient norm computation verified correct (prior fix in _train_recipe.py confirmed)
- ✅ No unprotected log/exp/sqrt calls found in critical paths

## Statistical Methods Validation
- ✅ `wilson_ci`: Correct Wilson score interval implementation with proper edge case handling (bool input guard, zero-total guard, integer overflow protection)
- ✅ Parametric bootstrap for distribution shift tests uses proper resampling
- ✅ P-value computation uses `(1 + exceed) / (n_sim + 1)` correction (never returns exactly 0)
- ✅ All statistical comparisons account for practical significance vs statistical significance

## Scientific Methodology
- ✅ Data leakage audit (`split_leakage_audit`) comprehensively checks cross-group row overlap and seed collisions
- ✅ Association exposure analysis properly distinguishes leakage from memorisation opportunities
- ✅ Training recipe properly separates train/val/test evaluation protocols
- ✅ Model state management (train/eval mode) properly handled
- ✅ Difficulty-stratified analysis with proper feature engineering

## Conclusion
No new bugs found. All prior fixes verified correct. Code quality is high with thorough defensive programming.
