# CAGE-Attention: Calibrated Gated Attention × Normalization Co-design

[中文说明](README.md) | **English**

A controlled experimental codebase for studying gated attention. It addresses
two reviewer concerns:

1. Are gated attention mechanisms merely minor variants of existing gates?
2. When deep residual streams contribute too little, how should gating and
   normalization be co-designed?

Core idea: **the gate should not depend only on the query (Gated-Attention
style); it should also be calibrated by the concentration (entropy) of the
attention distribution and by sequence length.** The gate is co-designed with
Pre/Post/Mix-LN, SeeDNorm-style dynamic normalization and LayerScale so that
deep residual streams can write useful content.

The design boundary and experiment contract live in
[docs/design_notes.md](docs/design_notes.md). Decisions marked 【固定】 keep the
numerical path consistent across implementations and commits; code changes
must respect them.

## Layout

| Path | Purpose |
|---|---|
| `cage/model.py` | CAGE model: 4 gate variants × 4 norm schemes × optional LayerScale |
| `cage/data.py` | Associative-recall synthetic task and length sweep |
| `cage/train.py` | Training, evaluation, gate statistics and false-concentration analysis |
| `cage/metrics.py` | Torch-only statistics and effect-size utilities |
| `cage/baselines.py` | Rule, linear-probe and depth-ladder baselines |
| `cage/experiments/` | E1/E2/E3 runners, result records and provenance checks |
| `tests/` | Standard-library `unittest` suite |
| `ci_gate.py` | Syntax, design-pin and smoke-test gate |
| `docs/design_notes.md` | Fixed design decisions and experiment contract |

## Environment and installation

Python 3.10+ and PyTorch are required. Install the runtime dependencies:

```bash
python3 -m pip install -e .
```

To reproduce the pinned CI dependency set only:

```bash
python3 -m pip install -r requirements-ci.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

## Quick start

Run the quality gate and tests:

```bash
make check          # ci_gate.py + unittest
make gate           # design pins and smoke tests only
make test           # unittest only
```

The gate can also be invoked directly:

```bash
python3 ci_gate.py
python3 -m unittest discover -s tests
```

Check the complete pipeline with a tiny configuration:

```bash
python3 -m cage.experiments --only sanity --quick --out results_quick
```

Then run the experiment you need:

```bash
python3 -m cage.experiments --only e1 --seeds 5 --out results
python3 -m cage.experiments --only e2 --seeds 5 --out results
python3 -m cage.experiments --only e3 --seeds 5 --out results
```

`--quick` is a pipeline check only; conclusions should come from the default
(non-quick) settings with the complete `--out` directory retained. Outputs
land in `results/` by default, which is git-ignored.

## Use from Python

```python
from cage.model import build
from cage.train import collect_stats, eval_accuracy, train_model

model = build({
    "vocab_size": 64,
    "d_model": 32,
    "n_heads": 4,
    "n_layers": 2,
    "max_len": 64,
    "gate": "calib",
})

hist = train_model(model, steps=3, batch=4, n_pairs=6, vocab=64,
                   log_every=1)
accuracy = eval_accuracy(model, 6, 64, "cpu", batch=4, batches=1)
stats = collect_stats(model, 6, 64, "cpu", batch=4, batches=1)
print(hist[-1], accuracy, sorted(stats))
```

The `seed` passed to `train_model` governs both the data stream and dropout
randomness; evaluation and statistics use fixed seeds by default, so identical
configurations can be compared directly.

## E1: the four strictly compared variants

| gate | Description |
|---|---|
| `none`  | Standard attention (no gating) |
| `query` | Pure query-dependent gating (Gated-Attention style), `σ(W_g x)` |
| `calib` | Our length- and concentration-calibrated gate, `σ((W_g x − a·log T − c·H̄)/τ)`, where `H̄` is attention entropy normalized by `log T` |
| `null`  | Explicit null attention / sparse-selection baseline: learnable null KV slots compete with real tokens |

The `calib` ablation switches reproduce the full gate at their defaults; they
exist only for the single-factor E1/E2 contrasts:

| Option | Default | Effect |
|---|---|---|
| `calib_use_len` | `True` | Use the length term `a·log T` |
| `calib_use_ent` | `True` | Use the concentration term `c·H̄` |
| `calib_ent_norm` | `True` | Normalize concentration by `log T`; `False` reads raw row-mean entropy |
| `calib_pos_norm` | `False` | Position-conflation ablation: divisor and length term use each row's visible-token count `log(max(t+1,2))`; strictly identical on the final query row |
| `calib_shuffle_ent` | `False` | Active control: permute only the entropy read by the gate, keeping the same signal distribution |

## Experiments

- **E1 gate comparison**: associative-recall accuracy; for `query` / `calib`,
  per-layer Pearson correlation of the gate with attention entropy and
  max-prob (Fisher-z averaged); whether the gate closes under “false
  concentration” (high max-prob but wrong answer), plus gate deciles on
  both splits (the cost of wrongly closing correct answers); `null`
  additionally reports `null_mass` and a full-distribution false-
  concentration readout. E1 records also carry the learned calibration
  coefficients (`calib_coefficients`, the same readout E2 stamps).
- **E2 length generalization**: train at a fixed length and evaluate on a
  length sweep spaced by factors of 2. The calibrated gate explicitly corrects
  the `log T` entropy growth and should degrade more slowly. Accuracy at each
  length plus entropy, max-prob and gate drift are reported, together with a
  position-embedding extrapolation control (`pos_extrapolation_control`, see
  Methodological boundaries below).
- **E3 normalization/activation co-design**: `query` / `calib` × `pre` /
  `post` / `mix` / `seednorm` × LayerScale (none / `1e-2`). Per-layer residual
  norms and `gate_depth_ratio` (mean deep gate / mean shallow gate), plus the
  cross-scheme-comparable per-layer relative writes (`attn_write_ratio` /
  `mlp_write_ratio`) and the per-layer knockout probe (`layer_knockout_eval`,
  see Methodological boundaries below).

## Key metrics

| Metric | Reading |
|---|---|
| `gate_vs_entropy` / `gate_vs_maxprob` | Whether the gate genuinely uses attention-distribution information; `query` should not correlate strongly with entropy, while `calib` should |
| `false_conc_rate` | Error rate among high-max-prob samples (the `null` variant additionally reports the full-distribution `false_conc_rate_full` and `mean_null_mass_on_high`) |
| `gate_on_false_conc` vs `gate_on_correct` | Whether the gate closes more often on false concentration (gate deciles `*_q10/q50/q90` are reported for both splits) |
| `gate_depth_ratio` | Above 1 means deep-layer gates are more open than shallow ones (a gate-value ratio, not layer importance; importance evidence comes from `layer_knockout_eval` and the per-layer write ratios) |
| `null_mass` | Share of attention weight assigned to the null slot in the `null` variant (the null KV is learnable, not a fixed empty slot) |

## Methodological boundaries

This section states the evidence boundaries and known limitations of each
claim, so results are not over-read relative to the controlled design.

1. **Correlation is not mechanism.** The `calib` gate formula contains the
   entropy term, so part of the gate-entropy correlation is built in. The
   evidence chain for “calibrated gating detects errors” is the ablation and
   active-control arms (H1d single-term ablations, H1f shuffled-entropy
   placebo with enforced capacity matching) plus the counterfactual probes -
   not the raw correlation.
2. **Positional semantics of the entropy normalisation.** The registered
   gate divides every row by the full-sequence `log T`, although causal row
   `t` sees only `t+1` keys (entropy ceiling `log(t+1)`). This task supervises
   and measures only the final query row, where the visible-key count equals
   `T` and the global divisor is exact; the forward-time behaviour on
   non-final rows does mix in position, and can be ablated with
   `calib_pos_norm=True` (strictly identical on the final row).
3. **E2 position-encoding confound and its control.** With learned absolute
   position embeddings trained at a fixed length, every sweep position beyond
   the trained length carries an untrained vector, so a length-sweep drop
   mixes attention/gate behaviour with position extrapolation. E2 now
   additionally evaluates each extrapolated length with `fold` (positions
   folded back into the trained range) and `zero` (extrapolated position
   vectors zeroed) on the same sealed stream
   (`pos_extrapolation_control`, exploratory companion): if folding recovers
   most of the drop, the degradation is mostly position extrapolation rather
   than the mechanism under test.
4. **Parameter identifiability.** Only the product `calib_c × calib_scale`
   enters the gate, so `calib_scale` is a fixed hyperparameter (a buffer, not
   a learnable parameter) and `calib_c` is *the* learnable concentration
   coefficient (`c_effective = calib_c × calib_scale`). Both E1 and E2
   records stamp the learned coefficients (`a_mean` / `c_effective_mean` ...,
   the H2f manipulation check).
5. **Initial gate-openness matching.** The `query` gate bias initialises at
   2 (openness ≈0.88), `calib` at 6: after subtracting `a·log T + c·H̄` at
   the default training length (T=34) both arms start at a comparable
   openness (≈0.8–0.9). The per-layer parameter delta (3 extra scalars for
   calib) is displayed in the contrast log.
6. **Conditional readouts for the `null` variant.** The entropy / max-prob
   reported for `null` describe the real-token distribution conditional on
   not choosing null (comparable with the other variants); `null_mass` is
   reported alongside, and `false_conc_rate_full` gives the null-slot-aware
   full-distribution version, so “writes almost nothing, and wrongly” is not
   misread as “confidently wrong”.
7. **Residual norms are not comparable across normalisation schemes.**
   Post-norm `resid_norm` is measured after LayerNorm (scale-pinned),
   pre/seednorm on the raw residual stream (depth-growing). For
   cross-scheme “deep writes” comparisons use `attn_write_ratio` /
   `mlp_write_ratio` (each branch relative to the stream it writes into)
   and `layer_knockout_eval` (replace one block with identity, measure the
   accuracy cost - interventional evidence). The LayerScale attention/MLP
   branches are separate parameters (ls1/ls2) and their learned values can
   be read out separately.
8. **Novelty boundary.** Query-dependent gating belongs to the Gated
   Attention family (*Gated Attention for Large Language Models:
   Non-linearity, Sparsity, and Attention-Sink-Free*, Qiu et al., 2025,
   arXiv:2505.06708, NeurIPS 2025 - head-specific sigmoid gating, sparse
   gating and long-context extrapolation). This project’s novelty claim is
   limited to the entropy × length calibration terms and must be
   established by the H1d/H1f/H2g ablations; “gating works” or “gating
   helps long context” is not a novelty claim.
9. **Gradient verification of the hand-written kernels.** The four custom
   autograd kernels (`_SoftmaxInPlace` / `_RowEntropy` / `_CalibAttn` /
   `_NullAttn`) are verified in `tests/test_gradcheck.py` by float64
   gradcheck, reference-implementation parity, the multi-block query
   decomposition, and train-path/eval-path agreement.

## Reproducibility and result checks

- `make check` runs `ci_gate.py` (syntax, design pins, smoke) and then the
  standard-library `unittest` suite under `tests/`; the tests need neither
  pytest nor numpy. `tests/test_gradcheck.py` adds double-precision
  gradcheck and reference parity for the hand-written autograd kernels.
- Every RNG and data stream uses the same 32-bit canonical seed: `s` and
  `s + 2**32` denote the same run, while bool or fractional seeds are rejected
  instead of being silently truncated.
- Every result record carries its code revision or source hash, the resolved
  configuration, data-stream fingerprints, and an environment snapshot
  including thread counts and determinism switches.
- `artifact_index.json` is cumulative across invocations. The following
  commands verify a historical results directory independently:

```bash
python3 -m cage.experiments --verify-artifacts --out results
python3 -m cage.experiments --check-environment --out results
python3 -m cage.experiments --verify-provenance --out results
python3 -m cage.experiments --verify-runs --out results
python3 -m cage.experiments --check-source --out results
```

`--verify-runs` rejects a results directory that contains an unfinished
(aborted/failed/interrupted) invocation or a history of critical environment
drift, so a cited evidence directory contains only completed runs produced
under one numerical environment. `make verify` runs the environment, artifact,
provenance, run-completeness and source checks in sequence.

## Design constraints

Code changes must respect the 【固定】 decisions in
[docs/design_notes.md](docs/design_notes.md). Run `python3 ci_gate.py` before
submitting any change; changes touching the experiment pipeline should also
run the full test suite.
