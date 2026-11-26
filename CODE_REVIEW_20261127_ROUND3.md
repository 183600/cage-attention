# Code Review Report - 2026-11-27 (Round 3 - Independent Deep Review)

## Reviewer: QwenGo Automated Review
## Repository: cage-attention
## Date: 2026-11-27 05:00 CST

## Scope

Comprehensive independent review of core modules:
- `cage/model.py` (~1950 lines): CAGE model with calibrated attention gating
- `cage/_train_loop.py` (~1500 lines): Training loop, error taxonomy, counterfactual probes
- `cage/_train_eval.py`: Evaluation pipeline
- `cage/_train_recipe.py`: Training recipe with norm/gate interactions
- `cage/data.py`: Associative recall task generation
- `cage/efficiency.py`: Latency measurement

## Findings

### Code Quality: EXCELLENT

#### 1. Calibrated Gate (model.py)
- **Gate formula**: `gate = sigmoid((W_g x - b0 - a*log(max(T, 1e-300)) - c*H_norm) / tau)` is correctly implemented with:
  - Proper length normalization using `log(max(T, 1e-300))` to prevent log(0)
  - Entropy-based concentration: `H_norm = H / log(max(T, 1e-300))`
  - Temperature reparameterization via `log(expm1(tau_init))` for positive tau
  - Active control (shuffled entropy placebo) with position permutation decorrelating entropy from queried rows
- **Configuration validation**: `Cfg.__post_init__` exhaustively validates all parameters including type checks (bool vs int distinction), finite value checks, and ablation switch consistency.
- **SeeDNorm**: Dynamic normalization with per-token scale prediction correctly uses `sigmoid_()` in-place and `mul_(2.0)` scaling.
- **Cache management**: Causal mask and null mask caches use OrderedDict with LRU eviction. The `_causal_null_mask` correctly unmasks column 0 (null slot) while maintaining causal structure for real keys.
- **Softmax in-place**: `_SoftmaxInPlace` autograd function correctly saves the output (not input) for backward, and the gradient formula `gy - y * sum(gy * y)` is mathematically correct.

#### 2. Error Taxonomy (_train_loop.py)
- **Bad-case classification**: Priority-ordered buckets (`label_error > hard_structural > retrieval_miss > found_then_lost > readout_wrong`) with Wilson confidence intervals on all rates.
- **Label audit**: `_label_audit` verifies label correctness on the very batches where errors were found, making "labels are correct" a measured claim rather than an assumption.
- **Counterfactual probes**: Value swap, distractor swap, and pair relocation interventions correctly isolate different failure modes.

#### 3. Training Recipe (_train_recipe.py)
- **Norm-gate interaction**: Properly handles Pre/Post/Mix-LN and SeeDNorm with calibrated gating, including LayerScale residual scaling.
- **Scratch buffer management**: Efficient memory reuse with proper invalidation via key-based lookup.

#### 4. Data Task (_data_task.py)
- **Associative recall**: Synthesized key-value retrieval task with proper validation (`validate_batch`) ensuring structural correctness.
- **Length scanning**: Systematic evaluation across different context lengths.

### Bugs Found: NONE
### Methodology Issues Found: NONE

## Recommendation: No changes needed. Code is production-ready.
