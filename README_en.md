# CAGE-Attention: Calibrated Gated Attention × Normalization/Activation Co-design

An experimental codebase addressing two reviewer concerns: "are sparse/gated
attention just minor variants of existing gating?" and "insufficient
contribution of deep layers". Core idea: **the gate should not depend only on
the query (Gated-Attention style) but also be calibrated by the concentration
(entropy) of the attention distribution and the sequence length**, co-designed
with Pre/Post/Mix-LN, SeeDNorm-style dynamic normalization and LayerScale so
that deep residual streams are "willing to write useful content".

[中文说明](README.md) | **English**

## Structure

```
cage/
  model.py   # CAGE model: 4 gate variants × 4 norm schemes × optional LayerScale
  data.py    # associative-recall synthetic task with measurable "correct answers"
  train.py   # training, evaluation, gate–entropy/max-prob correlation, false-concentration analysis
```

## Strictly compared variants (E1)

| gate | Description |
|---|---|
| `none`  | Standard attention (no gating) |
| `query` | Pure query-dependent gating (Gated-Attention style), `σ(W_g x)` |
| `calib` | **ours: length + concentration calibrated gate**, `σ((W_g x − a·log T − c·H̄)/τ)`, where H̄ is the attention entropy normalized by log T |
| `null`  | Explicit null attention / sparse-selection baseline (learnable null KV slots compete with real tokens) |

## Experiments

- **E1 gate comparison**: associative-recall accuracy, Pearson correlation of
  gate value vs attention entropy / max-prob, and whether the gate closes on
  "false concentration" (high max-prob but wrong answer) — the calibrated
  gate should give lower gate values under false concentration.
- **E2 length generalization**: fixed training length, evaluated on a length
  sweep with 2× spacing; the calibrated gate explicitly corrects the log-T
  entropy growth and should degrade more slowly; entropy/max-prob/gate drift
  at long lengths is reported.
- **E3 normalization/activation co-design**: `query`/`calib` ×
  `pre`/`post`/`mix`/`seednorm` × LayerScale (none/1e-2), reporting per-layer
  residual-norm growth and the **deep/early gate ratio** (mean deep gate /
  mean shallow gate), measuring whether deep layers become more willing to
  "write useful content".

## Key metric interpretation

- `gate_vs_entropy` / `gate_vs_maxprob`: whether the gate genuinely uses
  attention-distribution information (a `query` gate should not correlate
  strongly with entropy; `calib` should correlate significantly).
- `false_conc_rate`: error rate among high-max-prob samples; read together
  with `gate_on_false_conc` vs `gate_on_correct` to assess the gate's
  "self-doubt" capability.
- `deep_gate_ratio > 1`: after co-designed normalization, deep layers write
  more than shallow ones, indicating the Pre-LN deep-layer
  under-contribution problem is mitigated.

## Reproducibility and engineering checks

- `make check` runs `ci_gate.py` (syntax + design pins + smoke) and then the
  standard-library `unittest` suite under `tests/` (seed canonicalisation,
  data contract, environment snapshot and results-directory integrity). The
  tests need neither pytest nor numpy.
- Every RNG and data stream consumes the same 32-bit canonical seed, so `s`
  and `s + 2**32` are the same run; a bool or fractional seed is rejected
  instead of being silently truncated.
- Each result record carries its code revision/source hash, resolved config,
  data-stream fingerprints and an environment snapshot (thread counts and
  determinism switches included). `artifact_index.json` is cumulative across
  invocations, and
  `python3 -m cage.experiments --verify-artifacts --out <dir>` checks that
  historical result files have not been truncated or edited.
  `--check-environment` compares the recorded software/numerical
  environment, while `--check-source` compares every archived
  `source_tree_sha256` with the current `cage/*.py` tree; `make verify`
  runs the environment, artifact, provenance and source checks in sequence.
