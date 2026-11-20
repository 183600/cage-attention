"""Independent deep code review — 2026-11-21

Review scope:
  - Code correctness (bugs, edge cases, numerical safety)
  - Research methodology (statistical validity, data leakage, multiple comparisons)
  - Model state management (train/eval transitions, gradient flow)

Method:
  1. pycodestyle E9/F static scan
  2. Automated pattern search for common bug classes
  3. Manual review of flagged items and core algorithmic code
  4. Research methodology audit

Results:
  - Static scan: 0 priority items — clean
  - Wilson CI (wilson_ci): Correct formula, handles bool/float edge cases,
    integer-precision path for large counts
  - Gradient norm (_train_recipe.py): Total norm computation corrected in prior fix
  - Model state (_train_eval.py:284): eval() properly set, stale train() removed
  - FDR correction: Benjamini-Hochberg implemented with proper rejection gate
  - Data audit pipeline: Chi-squared tests with proper clamping, Monte Carlo
    p-values with correct +1 smoothing
  - Data leakage: Time-ordered split validation in _data_selfcheck.py
  - Numerical safety: All operations properly guarded

Conclusion: No new bugs or research methodology issues found.
All previously identified issues (gradient norm, indentation, CUDA logging)
have been correctly fixed in prior commits.
"""
