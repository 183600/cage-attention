"""Hypothesis-driven experiment runners for E1/E2/E3.

Design principles
-----------------
1. **Hypotheses first.**  Every config is annotated with the hypothesis
   it tests (``HYPOTHESES``) and a pre-registered decision rule.  The
   runner prints the hypotheses before executing and stores their ids
   with every result record, so the E3 grid is not a blind
   hyperparameter sweep: each contrast answers a mechanism question.
2. **Tuned baselines and sanity checks before complex models.**  The
   complexity ladder is explicit and every rung is tuned: the rule
   system (exact ceiling), the trivial floor, a learning-rate-tuned
   linear probe, and a learning-rate-tuned **single-layer standard
   transformer** (HB6, the simplest learned model) all run before any
   deep grid.  A shared training recipe is tuned on the plain ``none``
   baseline (validation seed), then frozen for all gate variants so the
   E1/E3 comparisons stay controlled (each side of every pre-registered
   contrast carries a machine-checked controlled-variable audit).  The
   baselines are checked *statistically* before the grids run: HB1/HB2/
   HB3 verdicts are written to ``baselines_verdict.json`` (rule ceiling
   with a Wilson CI, a bootstrap near-chance test for the tuned probe,
   and the split audit), and the learned path gets two more pre-flight
   gates: HB5 (tiny-batch overfit: the loop must be able to memorise a
   handful of samples) and HB4 (negative control: the same architecture
   trained on permuted labels must stay at chance, which rules out
   answer leakage).  E1 re-reads the verdict, the rule ceiling and the
   shallow baseline as positive controls and reports a paired
   deep-vs-shallow complexity contrast, so a gate variant is only
   credited for what it does beyond tuned simple baselines on a task the
   simple rule can actually solve.
3. **Honest evaluation.**  Tuning uses the validation seed only.  Final
   numbers are reported on a separate test seed that is never used for
   selection.  Every key claim is reported as mean +- std over
   independent training seeds together with a paired bootstrap 95% CI
   of the difference (``cage/metrics.py``); a claim is only read as
   supported when that CI excludes 0 *and* the contrast survives FDR
   control over its family.  The training curve is monitored
   too: every contrast is annotated when either side contains
   curve-invalid runs (plateau / diverging / numerical instability),
   since an underfit run produces "no difference" for reasons unrelated
   to the hypothesis, and a recipe selected between such curves is
   flagged when it is frozen.
4. **Split hygiene.**  Train/val/test streams are disjoint by
   construction (``cage.data``): the split is assigned before samples are
   drawn, reserved evaluation seeds are refused for training, and HB3
   audits duplicate overlap and distribution shift before any model runs.
5. **Multiplicity, power and practical significance.**  Each experiment
   interprets a *family* of contrasts, so the family is read through
   Benjamini-Hochberg FDR control (``--fdr-q``, default 0.05).  Every
   contrast reports its bootstrap p-value, the fraction of seeds
   agreeing on the sign, and the design's minimal detectable effect at
   80% power (MDE80); "CI includes 0" with ``|effect| < MDE80`` is
   always reported as *inconclusive*, not as evidence of absence.  Every
   hypothesis additionally pre-registers a **SESOI** (smallest effect
   size of interest) per metric (stored in the hashed registry):
   statistical significance below the SESOI is reported as
   *practically negligible* rather than "supported", and the finalize
   report plans the required seed count at the observed variance
   (``cage.metrics.seeds_for_power``), so a contrast is only read as
   supporting its hypothesis when it is both FDR-significant and
   practically relevant - a complex model that beats a strawman by a
   statistically detectable sliver has not earned its complexity.  This
   matters at the 5-seed scale used here: an uncorrected 95% readout over
   ~16 null contrasts would produce about one spurious "supported" claim
   per experiment.  At that same scale the percentile bootstrap is the
   first-order approximation in the chain, so every contrast additionally
   carries a jackknife (leave-one-seed-out) diagnostic and a bias-corrected
   accelerated (BCa) interval, and the report flags any positive verdict
   whose decision flips when a single seed is dropped or dissents from the
   BCa readout (see ``cage.metrics``).  The companions audit the
   pre-registered decision rule; they never replace it.

Outputs land in ``results/`` (git-ignored); nothing is written into the
tracked tree.

Usage
-----
    python3 -m cage.experiments --only sanity               # HB5 gate
    python3 -m cage.experiments --only baselines
    python3 -m cage.experiments --only e1 --seeds 5
    python3 -m cage.experiments --only all --quick          # pipeline check only
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import pathlib
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field

import torch

from .baselines import (
    copy_last_token_stats,
    linear_probe_accuracy,
    probe_vs_chance_test,
    rule_naive_last_match_stats,
    rule_pair_lookup_stats,
    train_linear_probe,
    tune_linear_probe,
)
from .data import (
    RESERVED_EVAL_SEEDS,
    SANITY_SEED,
    STATS_SEED,
    TEST_SEED,
    VAL_SEED,
    batch_fingerprint,
    length_sweep_lengths,
    make_batch,
    split_leakage_audit,
    stream_fingerprint,
    validate_batch,
)
from .metrics import (
    benjamini_hochberg,
    equivalence_test,
    paired_bootstrap_ci,
    seeds_for_power,
    wilson_ci,
)
from .model import GATES, NORMS, build
from .repro import seed_everything, stream_generator
from .train import (
    collect_error_cases,
    collect_stats,
    error_detection_aucs,
    eval_metrics,
    false_concentration_analysis,
    false_concentration_sweep,
    gate_depth_ratio,
    gate_stat_correlations,
    measure_inference_cost,
    overfit_sanity_check,
    train_model,
    training_curve_verdict,
)

# Seeds with fixed roles (independent of every training seed) are defined in
# ``cage.data`` so that every entry point shares one source of truth:
#   VAL_SEED   - model selection only (learning-rate tuning, baseline checks)
#   TEST_SEED  - the reported numbers; never used for any decision
#   STATS_SEED - mechanism statistics (matches the collect_stats default)

HYPO_IDS = ("HB1", "HB2", "HB3", "HB4", "HB5", "HB6", "H1a", "H1b", "H1c",
            "H1d", "H2a", "H2b", "H2c", "H3a", "H3b")

# Every hypothesis entry may carry ``"sesoi"``: the smallest effect size of
# interest per metric (same metric keys as the pre-registered contrasts).
# A contrast is only read as supporting its hypothesis when it is both
# FDR-significant *and* at least the SESOI: at the seed counts used here an
# effect of 0.1 accuracy points can be statistically detectable while being
# scientifically meaningless, and "significant but tiny" must not be
# reported as "the mechanism works".  SESOI units follow the metric: accuracy
# and error-margin shares in probability points, correlations/ratios in
# their own scale (see each entry).  The registry is content-hashed into
# every result record, so the SESOI is part of the pre-registration.
#
# A hypothesis whose prediction is a *null/equivalence* claim (a quantity is
# expected to stay inside a bound, e.g. H2b's length-stable H/log(T)) lists
# the metric under ``"sesoi_test"`` with value ``"equivalence"``: the SESOI
# then acts as the equivalence bound and the contrast is supported only when
# the 95% CI of the effect lies inside +-bound (bootstrap TOST,
# ``cage.metrics.equivalence_test``).  "Not significant" on a difference
# test is *not* evidence for a null prediction, so the two decision rules
# must not be conflated.
HYPOTHESES = {
    "HB1": {
        "experiment": "baselines",
        "question": "Is the recall task well posed with a known solution?",
        "hypothesis": "The pair structure admits an exact parameter-free "
                      "lookup rule because keys are distinct per sequence.",
        "prediction": "rule accuracy ~ 1.0 (ceiling) and unique_key_rate = 1.0",
        "decision": "rule accuracy < 0.99 (point estimate; the Wilson 95% "
                    "CI is reported to show the precision of the finite "
                    "evaluation stream) indicates a broken task/data "
                    "construction and invalidates everything above it",
        "sesoi": {"accuracy": 0.01},
    },
    "HB2": {
        "experiment": "baselines",
        "question": "Does the task require content-based lookup rather than "
                    "a simple linear decision?",
        "hypothesis": "A linear model on bag-of-token counts cannot express "
                      "key-value lookup; its accuracy stays near chance "
                      "1/(vocab-2) even after tuning.",
        "prediction": "the 95% bootstrap CI of (tuned probe accuracy - chance) "
                      "stays below +chance, i.e. the probe is confidently "
                      "below 2x chance",
        "decision": "The quantitative near-chance test "
                    "(cage.baselines.probe_vs_chance_test) must be supported; "
                    "a probe whose CI reaches above 2x chance would mean "
                    "answer information leaks into the bag-of-tokens "
                    "features (or a shortcut exists) and invalidates the "
                    "simple-baseline comparison",
    },
    "HB3": {
        "experiment": "baselines",
        "question": "Are the train / validation / test / stats streams "
                    "disjoint (no data leakage)?",
        "hypothesis": "The split is assigned before generation and each split "
                      "owns a disjoint RNG stream, so no sample is shared "
                      "between splits and no fitted statistic is computed on "
                      "held-out data.",
        "prediction": "exact-duplicate overlap between the training stream "
                      "and each held-out stream is 0; token total-variation "
                      "distance stays within the same-stream noise floor",
        "decision": "a shared sample, or a distribution gap above the noise "
                    "floor, invalidates every reported number",
    },
    "HB4": {
        "experiment": "baselines",
        "question": "Does the pipeline leak the answer into the training "
                    "signal (negative control)?",
        "hypothesis": "With the supervised answers permuted across the batch "
                      "during training, no true answer can be learned, so "
                      "accuracy on the unchanged test stream must stay at "
                      "chance.",
        "prediction": "the upper bound of the 95% bootstrap CI of "
                      "(control accuracy - chance) stays below the "
                      "pre-registered absolute tolerance "
                      "(Settings.negative_control_tol, default 0.02); the "
                      "pooled Wilson interval is reported for the "
                      "finite-stream uncertainty",
        "decision": "a control clearly above chance means answer "
                    "information leaks through the data/pipeline (or a "
                    "shortcut exists) and every accuracy reported above it "
                    "is invalid",
    },
    "HB5": {
        "experiment": "baselines",
        "question": "Can the training loop fit anything at all "
                    "(gradient/loss sanity)?",
        "hypothesis": "A small model trained on one fixed tiny batch with "
                      "dropout off should drive the training loss to ~0 and "
                      "the batch accuracy to 1.0; if it cannot, gradients, "
                      "the loss or the precision path are broken.",
        "prediction": "final train loss <= 0.1 and train accuracy = 1.0",
        "decision": "failure invalidates the training loop and blocks the "
                    "interpretation of every grid result (necessary, not "
                    "sufficient: data/label problems can survive "
                    "memorisation)",
    },
    "HB6": {
        "experiment": "baselines",
        "question": "Does the simplest learned model (one attention layer) "
                    "already solve the recall task, making depth (and the "
                    "E3 deep-contribution premise) unnecessary?",
        "hypothesis": "The recall readout needs match-then-shift "
                      "composition (find the key that equals the query, "
                      "then read the token one position later), which a "
                      "single attention layer cannot express, so the tuned "
                      "1-layer transformer should stay clearly below the "
                      "parameter-free rule ceiling.",
        "prediction": "the Wilson 95% upper bound of the tuned 1-layer "
                      "standard-transformer accuracy on the test stream "
                      "lies below the rule-ceiling point estimate",
        "decision": "the baseline is a complexity control, not a pipeline "
                    "validity gate: if the shallow upper bound reaches the "
                    "ceiling, every depth-based motivation (E3) must be "
                    "annotated that a 1-layer model suffices at this scale; "
                    "if it stays below, depth is measurably required and "
                    "E1's 8-layer gate contrasts are informative for it",
        "sesoi": {"accuracy": 0.01},
    },
    "H1a": {
        "experiment": "e1",
        "question": "Does the calibrated gate use attention concentration / "
                    "length information beyond the query?",
        "hypothesis": "The query-only gate has no access to the attention "
                      "distribution, so |gate_vs_entropy| is clearly larger "
                      "for calib than for query.",
        "prediction": "|gate_vs_entropy|(calib) - |gate_vs_entropy|(query) > 0; "
                      "the same direction holds for the rank (Spearman) "
                      "correlation as a robustness check",
        "decision": "paired bootstrap 95% CI of the seed differences "
                    "excludes 0 after Benjamini-Hochberg FDR control across "
                    "the E1 contrasts (q=0.05)",
        "sesoi": {"accuracy": 0.01, "abs_gate_vs_entropy": 0.05},
    },
    "H1b": {
        "experiment": "e1",
        "question": "Does the calibrated gate 'doubt' itself on false "
                    "concentration?",
        "hypothesis": "On samples with high max-prob but wrong answers the "
                      "calibrated gate closes more than on high-max-prob "
                      "correct samples, more so than the query gate.",
        "prediction": "gate_margin = gate_on_correct - gate_on_false_conc "
                      "is larger for calib than for query, and the sign is "
                      "stable when the high-maxprob threshold is swept "
                      "(0.3/0.5/0.7/0.9)",
        "decision": "paired bootstrap 95% CI excludes 0 after "
                    "Benjamini-Hochberg FDR control across the E1 contrasts "
                    "(q=0.05); the pre-registered threshold 0.5 is the "
                    "decision point, the sweep only reports robustness",
        "sesoi": {"gate_margin": 0.02},
    },
    "H1c": {
        "experiment": "e1",
        "question": "Does the explicit null slot suppress writes when the "
                    "model is wrong?",
        "hypothesis": "null_mass is higher on errors than on correct "
                      "answers (the null slot absorbs the write).",
        "prediction": "null_mass_error_margin > 0",
        "decision": "paired bootstrap 95% CI excludes 0 after "
                    "Benjamini-Hochberg FDR control across the E1 contrasts "
                    "(q=0.05)",
        "sesoi": {"null_mass_error_margin": 0.02},
    },
    "H1d": {
        "experiment": "e1",
        "question": "Does each calibration term carry its own contribution "
                    "(no dead component)?",
        "hypothesis": "The calib gate claims two additive terms (length "
                      "-a*log T and concentration -c*Hbar).  Within the "
                      "identical calib architecture, removing the "
                      "concentration term must dismantle the gate/entropy "
                      "association, and removing a term that does work must "
                      "show up in accuracy or in the false-concentration "
                      "margin - a component removable at no cost is not "
                      "doing work.",
        "prediction": "|gate_vs_entropy| collapses toward the query gate in "
                      "the no-concentration arms and stays clearly positive "
                      "in full calib; accuracy and gate_margin of full calib "
                      "are at least as high as in the ablated arms "
                      "(the fixed-length main effect of the length term is "
                      "expected to be small because at a fixed T it is a "
                      "constant absorbed by the gate bias; H2c tests it "
                      "where it matters, across lengths)",
        "decision": "paired bootstrap 95% CIs of accuracy(calib - ablated "
                    "arm) and gate_margin(calib - ablated arm) exclude 0 "
                    "after Benjamini-Hochberg FDR control across the E1 "
                    "family (q=0.05); the mechanism check validates that "
                    "each ablation removed the intended pathway and is "
                    "reported alongside, not as a decision criterion",
        "sesoi": {"accuracy": 0.01, "gate_margin": 0.02,
                  "abs_gate_vs_entropy": 0.05},
    },
    "H2a": {
        "experiment": "e2",
        "question": "Does length calibration slow the degradation beyond "
                    "the training length?",
        "hypothesis": "Because the gate subtracts a*log(T) and the "
                      "log(T)-normalised entropy, calib degrades less than "
                      "none when evaluated at 2x-spaced lengths.",
        "prediction": "accuracy_drop(calib) < accuracy_drop(none) and "
                      "accuracy@longest(calib) > accuracy@longest(none)",
        "decision": "paired bootstrap 95% CI of the differences excludes 0 "
                    "after Benjamini-Hochberg FDR control across the E2 "
                    "contrasts (q=0.05)",
        "sesoi": {"accuracy_at_longest": 0.01, "accuracy_drop": 0.01},
    },
    "H2b": {
        "experiment": "e2",
        "question": "Do the attention statistics drift with length as the "
                    "calibration assumes?",
        "hypothesis": "Raw entropy grows with log(T), while the quantity the "
                      "gate normalises by, H/log(T), is length-stable.",
        "prediction": "raw entropy drift (longest - base) > 0 (directional) "
                      "and the normalised entropy drift H/log(T) lies inside "
                      "the pre-registered equivalence bound, with the "
                      "calibrated gate drift inside its bound too",
        "decision": "paired bootstrap 95% CIs, FDR-controlled across the E2 "
                    "contrasts (q=0.05): the raw entropy drift CI excludes 0 "
                    "with a positive mean (difference test); the H/log(T) "
                    "drift and the gate drift are *equivalence* (TOST) tests "
                    "- each is supported only when its CI lies inside "
                    "+-SESOI, so a CI that merely includes 0 while spanning "
                    "a drift larger than the bound is 'not established', "
                    "not length-stability evidence",
        "sesoi": {"entropy_drift": 0.1, "entropy_norm_drift": 0.05,
                  "gate_drift": 0.02},
        # Metrics whose pre-registered prediction is a *null/equivalence*
        # claim: the SESOI above is then the bound the 95% CI must lie
        # inside (metrics.equivalence_test), not a minimum meaningful
        # effect.
        "sesoi_test": {"entropy_norm_drift": "equivalence",
                       "gate_drift": "equivalence"},
    },
    "H2c": {
        "experiment": "e2",
        "question": "Is the length term (not the concentration term) the "
                    "specific cause of calib's length robustness?",
        "hypothesis": "The advantage of calib over none beyond the training "
                      "length comes from subtracting a*log(T): removing the "
                      "length term (concentration-only arm) must lose the "
                      "advantage, while removing the concentration term "
                      "(length-only arm) must keep it.",
        "prediction": "accuracy_at_longest(calib) > accuracy_at_longest("
                      "calib_ent_only) and accuracy_drop(calib) < "
                      "accuracy_drop(calib_ent_only); calib_len_only stays "
                      "close to the full calib arm",
        "decision": "paired bootstrap 95% CI of calib - calib_ent_only on "
                    "accuracy_at_longest excludes 0 after "
                    "Benjamini-Hochberg FDR control across the E2 contrasts "
                    "(q=0.05); the calib_len_only arm is the reverse check "
                    "and is read together with it",
        "sesoi": {"accuracy_at_longest": 0.01, "accuracy_drop": 0.01},
    },
    "H3a": {
        "experiment": "e3",
        "question": "Does dynamic normalisation make deep layers write more "
                    "into the residual stream?",
        "hypothesis": "seednorm (per-token dynamic scale) raises the "
                      "deep/early gate ratio relative to pre-LN.",
        "prediction": "gate_depth_ratio(seednorm) > gate_depth_ratio(pre) "
                      "and deep residual norms grow more",
        "decision": "paired bootstrap 95% CI excludes 0 after "
                    "Benjamini-Hochberg FDR control across the E3 contrasts "
                    "(q=0.05)",
        "sesoi": {"gate_depth_ratio": 0.05, "accuracy": 0.01},
    },
    "H3b": {
        "experiment": "e3",
        "question": "Does LayerScale change how strongly deep layers write?",
        "hypothesis": "LayerScale (1e-2) damps the initial write strength "
                      "and thereby changes the residual-norm profile and "
                      "the gate depth ratio.",
        "prediction": "the primary readout "
                      "resid_norm_last_over_first shifts between LayerScale "
                      "off and on; gate_depth_ratio is a secondary readout",
        "decision": "paired bootstrap 95% CI of the primary readout "
                    "excludes 0 after Benjamini-Hochberg FDR control across "
                    "the E3 contrasts (q=0.05); the secondary readout is "
                    "reported for context and is not a decision criterion",
        "sesoi": {"resid_norm_last_over_first": 0.05,
                  "gate_depth_ratio": 0.05},
    },
}


@dataclass
class Settings:
    device: str = "cpu"
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 8
    max_len: int = 256
    vocab: int = 512
    batch: int = 64
    n_pairs: int = 16
    steps: int = 1500
    seeds: tuple = (0, 1, 2, 3, 4)
    eval_batch: int = 256
    eval_batches: int = 4
    stats_batch: int = 128
    stats_batches: int = 4
    probe_steps: int = 400
    probe_batch: int = 64
    dropout: float = 0.0        # model regularisation (recorded in the recipe)
    deterministic: bool = False  # opt-in deterministic torch kernels
    lr: float | None = None
    lr_grid: tuple = (1e-3, 3e-4, 1e-4)
    curve_points: int = 10      # validation checks per run (0 disables)
    val_batches: int = 2        # validation batches per check (fixed stream)
    cost_warmup: int = 2
    cost_runs: int = 7
    max_error_cases: int = 64   # per gate x seed, dumped to disk
    fdr_q: float = 0.05         # BH false-discovery rate for contrast families
    overfit_steps: int = 300    # HB5 tiny-batch overfit check
    overfit_batch: int = 8
    overfit_lr: float = 3e-3
    negative_control_tol: float = 0.02  # HB4 pre-registered tolerance
    out_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path("results"))

    @classmethod
    def quick(cls, device: str = "cpu") -> "Settings":
        """Tiny pipeline-check configuration (never for reported numbers)."""
        return cls(
            device=device, d_model=48, n_heads=4, n_layers=4, max_len=64,
            vocab=64, batch=16, n_pairs=6, steps=12, seeds=(0, 1),
            eval_batch=64, eval_batches=2, stats_batch=32, stats_batches=2,
            probe_steps=40, probe_batch=16, overfit_steps=100,
        )


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _mean_std(values):
    vals = [float(v) for v in values if _finite(v)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    else:
        std = 0.0
    return {"mean": mean, "std": std, "n": len(vals)}


def _aggregate_runs(runs, skip=("seed", "lr", "params")):
    keys = sorted({k for r in runs for k, v in r.items()
                   if _finite(v) and k not in skip})
    return {k: _mean_std([r[k] for r in runs if k in r]) for k in keys}


# Recipe fields that the controlled-comparison contract requires to be
# *identical* on both sides of every contrast (see ``_controlled_audit``):
# optimizer-relevant learning rate, schedule length, batch geometry, data
# sizes and every evaluation-stream size.  ``config`` is the resolved
# Settings snapshot stored with each run record.
_AUDIT_RECIPE_KEYS = (
    "device", "d_model", "n_heads", "n_layers", "max_len", "vocab", "batch",
    "n_pairs", "steps", "eval_batch", "eval_batches", "stats_batch",
    "stats_batches", "val_batches", "curve_points", "fdr_q", "dropout",
    "deterministic",
)

# Record fields that describe the module under test.  They are the only
# dimensions a contrast may vary; every other recorded design setting has to
# match across the two sides.
_AUDIT_FACTOR_KEYS = ("gate", "norm", "layer_scale", "calib_variant")

# The resolved-``Cfg`` keys a contrast is allowed to vary.  When both sides
# carry the ``model_cfg`` the model was actually built from, every *other*
# key must match: a difference there is an undeclared confound no matter
# what the labels say.
_AUDIT_MODEL_FACTOR_KEYS = ("gate", "norm", "layer_scale",
                            "calib_use_len", "calib_use_ent")

# Ground truth of the calib gate's additive terms, keyed by the resolved
# ``(calib_use_len, calib_use_ent)`` flags of the model that trained: the
# cell of the 2x2 "length term x concentration term" design a run is in.
# Cross-checking the declared ``calib_variant`` label against the flags the
# model was actually built with catches a mislabelled or mis-wired arm.
_CALIB_FACTORIAL = {
    (True, True): "full",
    (True, False): "calib_len_only",
    (False, True): "calib_ent_only",
    (False, False): "calib_no_terms",
}

_CALIB_FACTORIAL_CELLS = ("full", "calib_len_only", "calib_ent_only",
                          "calib_no_terms")


def _calib_cell(run):
    """The calibration-term cell a run actually trained (None when N/A).

    Uses the resolved ``model_cfg`` when present (ground truth); falls back
    to the declared label for records that predate the stamp.
    """
    cfg = run.get("model_cfg")
    if cfg is not None:
        if cfg.get("gate") != "calib":
            return None
        return _CALIB_FACTORIAL.get(
            (bool(cfg.get("calib_use_len", True)),
             bool(cfg.get("calib_use_ent", True))))
    label = run.get("calib_variant")
    return label or None


def _report_ablation_coverage(out_dir, tag, runs, cells=_CALIB_FACTORIAL_CELLS):
    """Verify a declared factorial ablation ran every cell of its design.

    A "each component is doing work" claim over a 2x2 (or leave-one-out)
    design is only supported if every cell was actually trained; a missing
    cell silently leaves one effect unmeasured.  The check reads the cell
    each run trained (resolved flags, not labels), prints a warning listing
    the missing cells, and stores ``<tag>_ablation_coverage.json`` next to
    the run records.
    """
    present = sorted({c for c in (_calib_cell(r) for r in runs) if c in cells})
    missing = [c for c in cells if c not in present]
    payload = {
        "tag": tag,
        "design": "calib_use_len x calib_use_ent (2x2)",
        "expected_cells": list(cells),
        "present_cells": present,
        "missing_cells": missing,
        "complete": not missing,
        "n_runs": len(runs),
    }
    if missing:
        print(f"[{tag}] WARNING: calibration-term ablation coverage "
              f"incomplete: {len(present)}/{len(cells)} cells trained "
              f"(missing {', '.join(missing)}); the corresponding term "
              f"effects are not measured")
    else:
        print(f"[{tag}] ablation coverage: {len(cells)}/{len(cells)} "
              f"calibration-term cells trained ({', '.join(present)})")
    _write_json(out_dir / f"{tag}_ablation_coverage.json", payload)
    return payload


def _recipe_view(run):
    """The controlled part of a run's configuration (a flat dict)."""
    cfg = run.get("config") or {}
    view = {k: cfg.get(k) for k in _AUDIT_RECIPE_KEYS}
    view["lr"] = run.get("lr")
    return view


def _controlled_audit(runs_a, runs_b):
    """Machine-check the controlled-variable contract of a contrast.

    Comparing method A with method B only attributes a difference to the
    module under test if *everything else* is held fixed.  This audit
    verifies, from the stored run records:

    1. **Paired seeds**: both sides were run on the same seed set, so the
       paired bootstrap differences each seed with itself (fewer / different
       seeds silently shrink the sample and break the pairing);
    2. **Within-side recipe**: every run on a side used the same learning
       rate, step count, batch geometry, data size and evaluation-stream
       size (no per-variant optimizer or schedule drift);
    3. **Across-side recipe**: the two recipes are identical;
    4. **Varying factor**: at least one declared model factor (gate / norm /
       layer_scale / calibration-ablation arm) actually differs, so the
       contrast is not degenerate;
    5. **Trained-model ground truth**: when both sides carry the resolved
       ``model_cfg``, every non-factor config key must match across the
       sides (an undeclared architectural difference is a confound even
       with matching recipe), and the declared labels must agree with the
       flags the model was actually built with - in particular a
       ``calib_variant`` label must name the (calib_use_len, calib_use_ent)
       cell that trained, so a mis-wired ablation arm cannot pass as the
       intended condition.

    The optimizer factory, cosine schedule, clipping and gradient settings
    are fixed in the single training entry point (``train.train_model``) and
    are therefore not per-run fields; ``lr`` plus ``steps`` pin the schedule
    that factory is combined with.  Returns an audit dict; a contrast whose
    audit fails is not read as supporting its hypothesis.
    """
    audit = {
        "controlled": True,
        "seeds_paired": False,
        "n_a": len(runs_a),
        "n_b": len(runs_b),
        "seeds_a": sorted(r.get("seed") for r in runs_a),
        "seeds_b": sorted(r.get("seed") for r in runs_b),
        "varying_factors": {},
        "model_factor_diffs": {},
        "mismatches": [],
    }
    audit["seeds_paired"] = audit["seeds_a"] == audit["seeds_b"]
    if not audit["seeds_paired"]:
        audit["mismatches"].append(
            f"seed sets differ: a={audit['seeds_a']} b={audit['seeds_b']} "
            f"(contrast must be paired over identical seeds)")
    for side, runs in (("a", runs_a), ("b", runs_b)):
        views = [_recipe_view(r) for r in runs]
        if not views:
            audit["mismatches"].append(f"side {side} has no runs")
            continue
        first = views[0]
        for j, other in enumerate(views[1:], start=1):
            for key, val in first.items():
                if other.get(key) != val:
                    audit["mismatches"].append(
                        f"side {side} recipe differs between runs 0 and {j}: "
                        f"{key}={val!r} vs {other.get(key)!r}")
    if runs_a and runs_b:
        va, vb = _recipe_view(runs_a[0]), _recipe_view(runs_b[0])
        for key, val in va.items():
            if vb.get(key) != val:
                audit["mismatches"].append(
                    f"recipe differs across sides: {key}={val!r} vs "
                    f"{vb.get(key)!r}")
        for key in _AUDIT_FACTOR_KEYS:
            in_a = any(key in r for r in runs_a)
            in_b = any(key in r for r in runs_b)
            if not (in_a and in_b):
                continue
            set_a = {str(r.get(key)) for r in runs_a}
            set_b = {str(r.get(key)) for r in runs_b}
            if set_a != set_b:
                audit["varying_factors"][key] = {
                    "a": sorted(set_a), "b": sorted(set_b)}
        # Ground truth of the trained model: check the resolved configs
        # directly rather than the hand-attached labels.
        cfgs_a = [r.get("model_cfg") for r in runs_a]
        cfgs_b = [r.get("model_cfg") for r in runs_b]
        if all(c is not None for c in cfgs_a + cfgs_b):
            for side, cfgs in (("a", cfgs_a), ("b", cfgs_b)):
                for j, other in enumerate(cfgs[1:], start=1):
                    if other != cfgs[0]:
                        audit["mismatches"].append(
                            f"side {side} model_cfg differs between runs 0 "
                            f"and {j}: {cfgs[0]} vs {other}")
            for key in sorted(set(cfgs_a[0]) | set(cfgs_b[0])):
                va, vb = cfgs_a[0].get(key), cfgs_b[0].get(key)
                if va == vb:
                    continue
                if key in _AUDIT_MODEL_FACTOR_KEYS:
                    audit["model_factor_diffs"][key] = {"a": va, "b": vb}
                else:
                    audit["mismatches"].append(
                        f"undeclared model-config difference across sides: "
                        f"{key}={va!r} vs {vb!r}")
            audit["model_cfg"] = {"a": cfgs_a[0], "b": cfgs_b[0]}
            for side, side_runs in (("a", runs_a), ("b", runs_b)):
                for r in side_runs:
                    label = r.get("calib_variant")
                    if not label:
                        continue
                    if (r.get("model_cfg") or {}).get("gate") != "calib":
                        audit["mismatches"].append(
                            f"side {side}: calib_variant={label!r} on a "
                            f"gate={(r.get('model_cfg') or {}).get('gate')!r} "
                            f"run")
                        continue
                    cell = _calib_cell(r)
                    if label != cell:
                        audit["mismatches"].append(
                            f"side {side}: calib_variant label {label!r} "
                            f"does not match the trained flags "
                            f"(cell {cell!r})")
            labels_a = {r.get("calib_variant") for r in runs_a
                        if r.get("calib_variant")}
            labels_b = {r.get("calib_variant") for r in runs_b
                        if r.get("calib_variant")}
            cells_a = {c for c in (_calib_cell(r) for r in runs_a) if c}
            cells_b = {c for c in (_calib_cell(r) for r in runs_b) if c}
            if (labels_a or labels_b) and labels_a != labels_b \
                    and cells_a == cells_b:
                audit["mismatches"].append(
                    f"declared ablation labels differ "
                    f"({sorted(labels_a)} vs {sorted(labels_b)}) but the "
                    f"trained calibration-term cells are identical "
                    f"({sorted(cells_a)}): the contrast does not actually "
                    f"vary the mechanism")
            if audit["model_factor_diffs"]:
                audit["varying_factors"].update(audit["model_factor_diffs"])
        if not audit["varying_factors"]:
            audit["mismatches"].append(
                "no declared model factor (gate/norm/layer_scale/"
                "calib_variant) differs across the sides: degenerate or "
                "mislabeled contrast")
    audit["controlled"] = not audit["mismatches"]
    return audit


def _paired_diff(runs_a, runs_b, metric):
    """Paired contrast ``a - b`` plus the control-side level.

    ``baseline_mean`` (the mean of ``metric`` on side ``b``) is attached so
    the report can print the effect as a relative change: a contrast can be
    statistically significant with zero variance yet practically
    negligible, and the relative effect is what exposes that.

    The controlled-variable audit (``_controlled_audit``) is attached as
    ``controlled``: a difference between recipes (optimizer settings, step
    count, data geometry, unpaired seeds) would confound the contrast, so
    the claim is refused downstream instead of being reported as a win.
    """
    by_seed = {r["seed"]: r for r in runs_b}
    diffs = []
    for ra in runs_a:
        rb = by_seed.get(ra["seed"])
        if rb is None or metric not in ra or metric not in rb:
            continue
        if _finite(ra[metric]) and _finite(rb[metric]):
            diffs.append(float(ra[metric]) - float(rb[metric]))
    ci = paired_bootstrap_ci(diffs)
    base = _mean_std([r[metric] for r in runs_b if metric in r])
    ci["baseline_mean"] = base["mean"]
    ci["controlled"] = _controlled_audit(runs_a, runs_b)
    return ci


def _std_effect(ci):
    """Standardized paired effect (mean / paired-difference SD).

    ``rel_effect`` (effect / control-side mean) answers "how large is this
    against the baseline level"; Cohen's dz answers "how large is this
    against the seed-to-seed noise", which is the scale the paired design
    actually resolves.  Both are reported because a gain can be
    economically tiny but noise-free (large dz, small rel) or vice versa.
    """
    sd = ci.get("sd")
    mean = ci.get("mean")
    if not (_finite(sd) and _finite(mean)) or float(sd) <= 0.0:
        return float("nan")
    return float(mean) / float(sd)


def small_n_flags(ci) -> str:
    """Short flags for the small-sample robustness companions of a contrast.

    Empty when the pre-registered decision and its jackknife / BCa
    companions agree; otherwise names *which* companion dissents, so the
    console line shows at a glance that a supported contrast needs a
    second look.  See ``cage.metrics`` for the definitions.
    """
    flags = []
    n = ci.get("n") or 0
    if ci.get("test") == "equivalence":
        if n >= 2 and ci.get("loo_within_bound") is False:
            flags.append("LOO outside bound")
    elif ci.get("loo_sign_flip"):
        flags.append("LOO sign flip")
    if ci.get("bca_agrees") is False:
        flags.append("BCa differs")
    return " ".join(flags)


def _fmt_ci(ci) -> str:
    if ci["n"] == 0 or not math.isfinite(ci["mean"]):
        return "n/a"
    p = ci.get("p_value", float("nan"))
    p_s = f"p={p:.3f} " if math.isfinite(p) else ""
    if ci.get("test") == "equivalence":
        # Null-prediction contrast: a *small* TOST p and a CI inside the
        # bound both support the prediction, so the verdict wording is the
        # mirror image of the difference tests below.
        bound = ci.get("bound", float("nan"))
        if ci.get("equivalent"):
            verdict = f"CI inside +-{bound:g} (prediction holds)"
        elif ci["lo"] > bound or ci["hi"] < -bound:
            verdict = f"CI outside +-{bound:g} (prediction refuted)"
        else:
            verdict = f"CI not inside +-{bound:g} (inconclusive)"
        return (f"{ci['mean']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}] "
                f"n={ci['n']} {p_s}{verdict}")
    supported = (ci["lo"] > 0.0) or (ci["hi"] < 0.0)
    mde = ci.get("mde80", float("nan"))
    if supported:
        verdict = "CI excludes 0"
    elif math.isfinite(mde) and abs(ci["mean"]) < mde:
        verdict = (f"inconclusive: |effect| < MDE80={mde:.4f} "
                   f"(design cannot resolve it)")
    else:
        verdict = "CI includes 0"
    return (f"{ci['mean']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}] "
            f"n={ci['n']} {p_s}{verdict}")


CURVE_INVALID_VERDICTS = (
    "plateau", "diverging", "numerical_instability", "insufficient_points")


def _curve_health(runs):
    """Verdict counts and the number of curve-invalid runs in ``runs``.

    A curve-invalid run never learned (``plateau``), diverged, or hit
    non-finite values: the accuracies and mechanism statistics measured on
    it cannot support *any* claim about the variant - an underfit run
    produces "no difference" for reasons unrelated to the hypothesis (and
    a test set cannot measure the value of a mechanism the optimiser never
    engaged).  ``overfitting`` stays claim-valid under the fixed-step
    contract but is surfaced through ``late_overfit`` /
    ``val_regression_after_best`` instead.
    """
    counts = {}
    for r in runs:
        verdict = r.get("curve_verdict")
        counts[verdict] = counts.get(verdict, 0) + 1
    bad = sum(counts.get(v, 0) for v in CURVE_INVALID_VERDICTS)
    return counts, bad


def _curve_warn(*labeled_runs) -> str:
    """Contrast annotation: which side has curve-invalid runs, and how many.

    Contrasts are printed regardless (they are the pre-registered readout),
    but a supported CI that rests on curves which never learned is not
    evidence about the mechanism, so the pair is flagged at the point of
    interpretation.
    """
    parts = []
    for label, runs in labeled_runs:
        counts, bad = _curve_health(runs)
        if not bad:
            continue
        detail = ",".join(f"{v}*{counts[v]}" for v in CURVE_INVALID_VERDICTS
                          if counts.get(v))
        parts.append(f"{label}: {bad}/{len(runs)} curve-invalid ({detail})")
    return "WARN " + "; ".join(parts) if parts else ""


class ContrastLog:
    """One experiment's pre-registered contrast family, FDR-controlled.

    Every contrast is printed as soon as it is computed (the readout is
    unchanged), and :meth:`finalize` applies Benjamini-Hochberg control
    across the whole family.  The family is the unit of inference: with
    ~16 contrasts per experiment, an uncorrected per-contrast 95% readout
    would be expected to produce about one spurious "supported" claim when
    every null is true.  The finalize report prints each contrast's
    bootstrap p-value, adjusted p, effect size, MDE80 and seed
    sign-consistency, and writes the table to ``<tag>_contrasts.json`` so
    the corrected readout is auditable next to the run records.

    Every contrast built by ``_paired_diff`` also carries its
    controlled-variable audit: a contrast whose two sides differ in
    optimizer/recipe settings, whose seeds are not paired, or that does not
    vary a declared model factor is *not* read as supported, whatever its
    p-value - the difference would not be attributable to the module under
    test.  The audit verdict is printed inline and stored per row.

    The decision layer also separates statistical from *practical*
    significance: the hypothesis's pre-registered SESOI (looked up in
    ``HYPOTHESES``, so it is part of the hashed pre-registration) is
    compared with the effect, ``seeds_for_power`` plans how many seeds the
    SESOI would need at the observed variance, and an FDR-significant
    effect below its SESOI is reported as practically negligible instead
    of "supported".

    Contrasts whose pre-registered prediction is *null/equivalence*
    ("this quantity stays inside +-bound") must be added through
    :meth:`print_equivalence`, which runs the bootstrap TOST from
    ``cage.metrics`` with the bound registered under ``sesoi_test`` and
    reads support as CI-within-bound - never as "CI includes 0".  Its
    TOST p-value joins the same BH family as the directional p-values.

    ``hypotheses`` lists the pre-registered ids this family is supposed to
    evaluate; :meth:`finalize` audits coverage (registered but absent, or
    rows citing unknown ids) and records it in the payload, so a
    pre-registered decision that no contrast executes is flagged instead
    of silently missing.

    Small-sample robustness: at the seed counts used here the percentile
    bootstrap is the first-order approximation in the chain, so every row
    also stores the jackknife (leave-one-out) mean range, its
    sign/bound-flip flag and the BCa interval verdict (see
    ``cage.metrics``).  :meth:`finalize` counts the contrasts whose
    *decision* flips when any single seed is dropped and those where BCa
    disagrees with the percentile readout, prints them as warnings, and
    writes the counts into the payload - the pre-registered decision rule
    itself is untouched, but a claim carried by one influential seed is
    no longer invisible.
    """

    def __init__(self, tag: str, hypotheses=()):
        self.tag = tag
        self.hypotheses = tuple(hypotheses)
        self.rows: list[dict] = []

    def print(self, rule, hyp, metric, ci, warn="") -> None:
        equiv = ci.get("test") == "equivalence"
        audit = ci.get("controlled")
        sesoi = (HYPOTHESES.get(hyp, {}).get("sesoi") or {}).get(metric)
        self.rows.append({
            "rule": rule, "hypothesis": hyp, "metric": metric,
            "mean": ci.get("mean"), "lo": ci.get("lo"), "hi": ci.get("hi"),
            "n": ci.get("n"), "p_value": ci.get("p_value"),
            "mde80": ci.get("mde80"), "sd": ci.get("sd"),
            "dz": _std_effect(ci),
            "sesoi": sesoi,
            "test": "equivalence" if equiv else "difference",
            "equivalence_bound": ci.get("bound") if equiv else None,
            "equivalent": ci.get("equivalent") if equiv else None,
            "sign_consistency": ci.get("sign_consistency"),
            "loo_mean_min": ci.get("loo_mean_min"),
            "loo_mean_max": ci.get("loo_mean_max"),
            "loo_sign_flip": ci.get("loo_sign_flip"),
            "loo_within_bound": ci.get("loo_within_bound"),
            "lo_bca": ci.get("lo_bca"),
            "hi_bca": ci.get("hi_bca"),
            "equivalent_bca": ci.get("equivalent_bca") if equiv else None,
            "bca_agrees": ci.get("bca_agrees"),
            "baseline_mean": ci.get("baseline_mean"),
            "controlled": None if audit is None else audit["controlled"],
            "varying_factors": ({} if audit is None
                                else audit.get("varying_factors")),
            "control_mismatches": ([] if audit is None
                                   else audit.get("mismatches", [])),
        })
        line = f"  [{hyp}] {rule} ({metric}): {_fmt_ci(ci)}"
        if audit is not None:
            factors = ",".join(sorted(audit.get("varying_factors", {}))) or "n/a"
            line += (f"  [control: {'ok' if audit['controlled'] else 'FAILED'}"
                     f", varies={factors}]")
        if sesoi is not None:
            line += (f"  [equivalence bound={sesoi:g}]" if equiv
                     else f"  [SESOI={sesoi:g}]")
        flags = small_n_flags(ci)
        if flags:
            line += f"  [{flags}]"
        if warn:
            line += f"  [{warn}]"
        print(line)
        if audit is not None and not audit["controlled"]:
            for mismatch in audit["mismatches"]:
                print(f"      CONTROL WARNING: {mismatch}")

    def print_equivalence(self, rule, hyp, metric, diffs, warn="") -> None:
        """Add a pre-registered *null-prediction* contrast (bootstrap TOST).

        ``bound`` is read from the hypothesis's registered SESOI and the
        metric must be declared as an equivalence test under
        ``sesoi_test``; otherwise the call raises, because running a null
        prediction through the difference machinery would misread "CI
        includes 0" (which may merely mean "underpowered") as support.
        """
        entry = HYPOTHESES.get(hyp, {})
        bound = (entry.get("sesoi") or {}).get(metric)
        if (entry.get("sesoi_test") or {}).get(metric) != "equivalence" \
                or bound is None:
            raise ValueError(
                f"{hyp}/{metric} has no pre-registered equivalence bound "
                f"in HYPOTHESES; register it under 'sesoi_test' first")
        self.print(rule, hyp, metric, equivalence_test(diffs, bound),
                   warn=warn)

    def finalize(self, out_dir: pathlib.Path, q: float = 0.05) -> dict:
        # TOST p-values of equivalence rows enter the same BH family as the
        # directional p-values: both answer "is this contrast worth a claim
        # at FDR q?" and splitting the family would loosen the control.
        adj = benjamini_hochberg([r["p_value"] for r in self.rows], q=q)
        for row, a in zip(self.rows, adj):
            row["p_adj"] = a["p_adj"]
            # A confounded contrast cannot support its hypothesis even if
            # its p-value is small: refuse the claim at the decision point.
            row["fdr_reject"] = bool(a["reject"] and row.get("controlled") is not False)
            row["confounded"] = row.get("controlled") is False
            base = row.get("baseline_mean")
            row["rel_effect"] = (
                row["mean"] / abs(base)
                if base is not None and math.isfinite(base) and base != 0.0
                else float("nan"))
            sesoi = row.get("sesoi")
            mean = row.get("mean")
            finite_effect = mean is not None and math.isfinite(mean)
            if row["test"] == "equivalence":
                # Null prediction: the practical gate is the CI-within-
                # bound rule, *not* |effect| >= SESOI (which would reward
                # exactly the drift the prediction says should not occur).
                row["practically_relevant"] = bool(row.get("equivalent"))
            else:
                row["practically_relevant"] = bool(
                    sesoi is None
                    or (finite_effect and math.isfinite(float(sesoi))
                        and abs(mean) >= float(sesoi)))
            row["supports_hypothesis"] = bool(
                row["fdr_reject"] and row["practically_relevant"])
            # Small-sample robustness: does the *decision* survive when any
            # single seed is dropped (jackknife), and does BCa reach the
            # same decision as the pre-registered percentile interval?
            # These are annotations, not decision rules: the verdict above
            # stays exactly as pre-registered, but a supported verdict that
            # is this fragile is flagged below.
            claim = row["supports_hypothesis"] and (row.get("n") or 0) >= 2
            if row["test"] == "equivalence":
                row["loo_flips_decision"] = bool(
                    claim and row.get("loo_within_bound") is False)
            else:
                row["loo_flips_decision"] = bool(
                    claim and row.get("loo_sign_flip"))
            row["bca_disagrees"] = row.get("bca_agrees") is False
            # A-priori power planning for the pre-registered SESOI at the
            # observed paired-difference SD: how many seeds would this
            # contrast need to resolve an effect it cares about?  (Adjust
            # the design *before* the next run, not after.)  For an
            # equivalence row the target is the bound: how many seeds
            # until a CI of this width could fit inside +-SESOI.
            row["n_seeds_for_sesoi"] = (
                float("inf") if sesoi is None
                else seeds_for_power(row.get("sd"), sesoi))
        testable = [r for r in self.rows if math.isfinite(r["p_adj"])]
        surviving = [r for r in testable if r["fdr_reject"]]
        supporting = [r for r in testable if r["supports_hypothesis"]]
        loo_flips = [r for r in testable if r.get("loo_flips_decision")]
        bca_differs = [r for r in testable if r.get("bca_disagrees")]
        low_resolution = [r for r in testable if r["n"] <= 5]
        below_sesoi = [r for r in surviving
                       if not r["practically_relevant"]
                       and r["test"] != "equivalence"]
        equiv_rows = [r for r in testable if r["test"] == "equivalence"]
        equivalent = [r for r in equiv_rows if r.get("equivalent")]
        equiv_borderline = [r for r in surviving
                            if r["test"] == "equivalence"
                            and not r.get("equivalent")]
        confounded = [r for r in testable if r.get("confounded")]
        controlled_pass = [r for r in self.rows if r.get("controlled") is True]
        audited = [r for r in self.rows if r.get("controlled") is not None]
        print(f"\n[{self.tag}] multiplicity control (Benjamini-Hochberg FDR, "
              f"q={q:g}): {len(surviving)}/{len(testable)} testable "
              f"contrasts survive, {len(supporting)} support a "
              f"pre-registered decision "
              f"(uncorrected p -> adjusted p; only 'SURVIVES'/'EQUIVALENT' "
              f"supports a pre-registered decision; rel = effect relative "
              f"to the control-side mean, dz = standardized effect (mean / "
              f"paired SD), loo = jackknife mean range - read rel/dz for "
              f"practical significance; a statistically significant effect "
              f"below its SESOI is practically negligible and does NOT "
              f"support the hypothesis; a null prediction is supported only "
              f"when the 95% CI lies inside its equivalence bound; a "
              f"positive verdict that flips when one seed is dropped, or "
              f"dissents from the BCa interval, is flagged after the "
              f"table):")
        for r in testable:
            rel_s = (f", rel={r['rel_effect']:+.2%}"
                     if math.isfinite(r["rel_effect"]) else "")
            sesoi = r.get("sesoi")
            equiv = r["test"] == "equivalence"
            sesoi_s = (f"{'bound' if equiv else 'SESOI'}={sesoi:g}"
                       if sesoi is not None else "")
            if r.get("confounded"):
                verdict = "REFUSED (confounded contrast)"
            elif equiv and r.get("equivalent") and r["fdr_reject"]:
                verdict = "EQUIVALENT (CI inside bound, null prediction holds)"
            elif equiv and sesoi is not None and (
                    r["lo"] > float(sesoi) or r["hi"] < -float(sesoi)):
                verdict = ("REFUTED (CI entirely beyond the equivalence "
                           "bound, null prediction fails)")
            elif not r["fdr_reject"]:
                verdict = ("not significant (TOST)" if equiv
                           else "not significant")
            elif equiv:
                verdict = ("TOST-significant but CI not fully inside bound "
                           "(borderline, not read as support)")
            elif not r["practically_relevant"]:
                verdict = ("FDR-significant but |effect| < SESOI "
                           "(practically negligible)")
            else:
                verdict = "SURVIVES"
            need = r.get("n_seeds_for_sesoi", float("inf"))
            need_s = ""
            if math.isfinite(need) and r["n"] > 0 and need > r["n"]:
                need_s = (f", needs ~{math.ceil(need)} seeds for "
                          f"{'the bound' if equiv else 'SESOI'} at 80% "
                          f"power (has {r['n']})")
            dz = r.get("dz", float("nan"))
            dz_s = f", dz={dz:+.2f}" if math.isfinite(dz) else ""
            loo_lo, loo_hi = r.get("loo_mean_min"), r.get("loo_mean_max")
            loo_s = (f", loo=[{loo_lo:+.4f}, {loo_hi:+.4f}]"
                     if _finite(loo_lo) and _finite(loo_hi) else "")
            print(f"  [{r['hypothesis']}] {r['rule']} ({r['metric']}): "
                  f"p={r['p_value']:.4f} -> p_adj={r['p_adj']:.4f} "
                  f"{verdict} "
                  f"(effect={r['mean']:+.4f}{rel_s}{dz_s}, "
                  f"MDE80={r['mde80']:.4f}{', ' if sesoi_s else ''}"
                  f"{sesoi_s}, seeds={r['n']}, "
                  f"sign-consistency={r['sign_consistency']:.2f}"
                  f"{loo_s}{need_s})")
        if audited:
            print(f"[{self.tag}] controlled-variable audit: "
                  f"{len(controlled_pass)}/{len(audited)} contrasts pass "
                  f"(paired seeds + identical training recipe across sides + "
                  f"a declared model factor varies)")
        if confounded:
            print(f"[{self.tag}] WARNING: {len(confounded)} contrast(s) "
                  f"failed the controlled-variable audit and were refused "
                  f"regardless of their p-values")
        if below_sesoi:
            underpowered = [r for r in below_sesoi
                            if math.isfinite(r.get("n_seeds_for_sesoi", 0.0))
                            and r["n_seeds_for_sesoi"] > r["n"]]
            print(f"[{self.tag}] WARNING: {len(below_sesoi)} FDR-significant "
                  f"contrast(s) are smaller than their pre-registered SESOI: "
                  f"statistically detectable but practically negligible; "
                  f"{len(underpowered)} of them would need more seeds "
                  f"(listed per row above) to resolve the SESOI - do not "
                  f"read them as supporting the mechanism")
        if equiv_rows:
            print(f"[{self.tag}] equivalence (null-prediction) tests: "
                  f"{len(equivalent)}/{len(equiv_rows)} have their 95% CI "
                  f"inside the pre-registered bound")
        if equiv_borderline:
            print(f"[{self.tag}] WARNING: {len(equiv_borderline)} "
                  f"equivalence contrast(s) pass the TOST FDR gate but their "
                  f"CI is not fully inside the bound (borderline); they do "
                  f"NOT count as confirming the null prediction")
        low_seed = [r for r in testable if r["n"] < 3]
        if low_seed:
            print(f"[{self.tag}] WARNING: {len(low_seed)}/{len(testable)} "
                  f"testable contrast(s) rest on fewer than 3 independent "
                  f"seeds; the recommended minimum for a seed-level claim "
                  f"is 3-5, so treat them as provisional")
        if loo_flips:
            names = ", ".join(f"{r['hypothesis']}/{r['rule']}"
                              for r in loo_flips)
            print(f"[{self.tag}] WARNING: {len(loo_flips)} positive "
                  f"decision(s) flip when any single seed is dropped "
                  f"(jackknife): {names}; the effect rests on one "
                  f"influential seed, so replicate at more seeds before "
                  f"reading it as a mechanism")
        if bca_differs:
            names = ", ".join(f"{r['hypothesis']}/{r['rule']}"
                              for r in bca_differs)
            print(f"[{self.tag}] WARNING: {len(bca_differs)} testable "
                  f"contrast(s) where the bias-corrected (BCa) interval "
                  f"reaches the opposite decision from the pre-registered "
                  f"percentile interval: {names}; at these seed counts the "
                  f"readout is sensitive to the interval construction, so "
                  f"treat either verdict as provisional until more seeds "
                  f"settle the disagreement")
        if low_resolution:
            print(f"[{self.tag}] note: {len(low_resolution)}/{len(testable)} "
                  f"testable contrast(s) rest on <=5 seeds; an exact "
                  f"two-sided sign test cannot reach p<0.05 below n=6, so "
                  f"direction reads rest on the bootstrap's moment "
                  f"assumptions - the jackknife/BCa annotations above show "
                  f"the exposure")
        registered = list(self.hypotheses)
        covered = sorted({r["hypothesis"] for r in self.rows})
        uncovered = [h for h in registered if h not in covered]
        unknown = [h for h in covered if h not in HYPOTHESES]
        misplaced = [h for h in covered
                     if h in HYPOTHESES
                     and HYPOTHESES[h].get("experiment") != self.tag]
        if registered:
            hit = len([h for h in registered if h in covered])
            print(f"[{self.tag}] pre-registration coverage: {hit}/"
                  f"{len(registered)} registered hypotheses have at least "
                  f"one contrast in this family")
        if uncovered:
            print(f"[{self.tag}] WARNING: pre-registered hypothesis(es) "
                  f"{', '.join(uncovered)} have NO contrast in this family; "
                  f"their pre-registered decision cannot be evaluated")
        if unknown:
            print(f"[{self.tag}] WARNING: contrast row(s) cite unknown "
                  f"hypothesis id(s) {', '.join(unknown)} (not in the "
                  f"hashed registry)")
        if misplaced:
            for h in misplaced:
                print(f"[{self.tag}] WARNING: hypothesis {h} is registered "
                      f"to experiment {HYPOTHESES[h].get('experiment')!r}, "
                      f"not {self.tag!r}; check the mapping")
        payload = {"tag": self.tag, "q": q, "n_contrasts": len(self.rows),
                   "n_testable": len(testable),
                   "n_surviving": len(surviving),
                   "n_supporting": len(supporting),
                   "n_below_sesoi": len(below_sesoi),
                   "n_equivalence": len(equiv_rows),
                   "n_equivalent": len(equivalent),
                   "n_equiv_borderline": len(equiv_borderline),
                   "n_confounded": len(confounded),
                   "n_audited": len(audited),
                   "n_control_passed": len(controlled_pass),
                   "n_low_seed": len(low_seed),
                   "n_loo_flip_decision": len(loo_flips),
                   "n_bca_disagree": len(bca_differs),
                   "n_low_resolution": len(low_resolution),
                   "loo_flip_contrasts": [
                       f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                       for r in loo_flips],
                   "bca_disagree_contrasts": [
                       f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                       for r in bca_differs],
                   "registered_hypotheses": registered,
                   "covered_hypotheses": covered,
                   "uncovered_hypotheses": uncovered,
                   "unknown_hypotheses": unknown,
                   "misplaced_hypotheses": misplaced,
                   "rows": self.rows}
        _write_json(out_dir / f"{self.tag}_contrasts.json", payload)
        return payload


def _format_case(case) -> str:
    """One-line, human-readable bad case for the console."""
    ctx = case["context"]
    n_pairs = max(0, (len(ctx) - 2) // 2)
    pairs = " ".join(f"{ctx[2 * i]}:{ctx[2 * i + 1]}" for i in range(n_pairs))
    flags = []
    if case["confidence"] >= 0.9:
        flags.append("high_conf")
    if case["confidence"] < 0.5:
        flags.append("low_conf")
    if case["pred_in_context"]:
        flags.append("pred_in_context")
    if case["naive_confusable"]:
        flags.append("naive_confusable")
    if case["value_repeats"]:
        flags.append("value_repeats")
    return (f"pair#{case['key_pair_idx']} key={case['query_key']} "
            f"ans={case['answer']} pred={case['pred']} "
            f"conf={case['confidence']:.2f} "
            f"mass_key={case['attn_mass_on_key']:.2f} "
            f"type={case['error_type']} "
            f"flags=[{','.join(flags) or 'plain'}] pairs=[{pairs}]")


def _announce(hyp_ids):
    for hid in hyp_ids:
        h = HYPOTHESES[hid]
        print(f"[hypothesis {hid}] {h['question']}")
        print(f"    prediction: {h['prediction']}")
        print(f"    decision:   {h['decision']}")
        sesoi = h.get("sesoi") or {}
        equiv = h.get("sesoi_test") or {}
        directional = {k: v for k, v in sorted(sesoi.items())
                       if k not in equiv}
        if directional:
            detail = ", ".join(f"{k}>={v:g}"
                               for k, v in directional.items())
            print(f"    SESOI (smallest effect size of interest): {detail} "
                  f"- statistical significance below this is reported as "
                  f"practically negligible, not as support")
        for k in sorted(equiv):
            if k not in sesoi:
                continue
            print(f"    equivalence bound (null prediction, bootstrap TOST): "
                  f"|{k}| <= {sesoi[k]:g} - supported only when the 95% CI "
                  f"of the effect lies inside this bound; a CI that merely "
                  f"includes 0 is not enough")


def _write_result(path: pathlib.Path, record: dict) -> None:
    """Append one JSONL run record, stamped with its creation time.

    The timestamp is injected here rather than by each runner, so every
    record on disk carries "when" next to the ``git_sha`` / ``setup_sha256``
    provenance it already stores inline.
    """
    record.setdefault("created_utc",
                      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               sort_keys=True), encoding="utf-8")


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    """Whether the working tree differs from the recorded revision.

    A result produced from a dirty tree cannot be rebuilt from its
    ``git_sha`` alone, so every run record says so explicitly instead of
    silently attributing the numbers to the committed revision.
    """
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except Exception:
        return True


def _hypotheses_sha() -> str:
    """Content hash of the pre-registered hypothesis registry.

    Records carry the hash so a result can be matched to the exact
    decision rules in force when it was produced: if the registry is
    edited mid-analysis, old and new records can no longer be silently
    pooled under "the same" hypotheses.
    """
    payload = json.dumps(HYPOTHESES, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _environment_record(settings: Settings) -> dict:
    """Software/hardware provenance for reproducibility.

    Beyond versions the record pins the numerical-behaviour switches a
    rerun must match: CPU thread count (parallel reductions change the
    summation order), cuDNN determinism/benchmark, whether deterministic
    algorithms are on, the cuBLAS workspace config and the NumPy version
    when installed.
    """
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": settings.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "torch_num_threads": int(torch.get_num_threads()),
        "deterministic_algorithms": bool(getattr(
            torch, "are_deterministic_algorithms_enabled", lambda: False)()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "git_dirty": _git_dirty(),
    }
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        env["cudnn_deterministic"] = bool(cudnn.deterministic)
        env["cudnn_benchmark"] = bool(cudnn.benchmark)
        try:
            env["cudnn_version"] = cudnn.version()
        except Exception:  # noqa: BLE001 - probe must not fail a record
            env["cudnn_version"] = None
    try:
        import numpy as np
        env["numpy"] = np.__version__
    except ImportError:
        env["numpy"] = None
    if settings.device.startswith("cuda") and torch.cuda.is_available():
        env["cuda_device"] = torch.cuda.get_device_name(0)
    return env


_SEEDING_POLICY = {
    "every_training_run": (
        "cage.repro.seed_everything(seed): python random + numpy (when "
        "installed) + torch CPU and all CUDA devices; deterministic kernels "
        "only when Settings.deterministic=True"),
    "data_streams": (
        "each split and training seed owns a dedicated torch.Generator "
        "(DataLoader generator= equivalent; make_batch never reads the "
        "global RNG), so data order cannot be perturbed by other seeding"),
    "eval_stream_seeds": {"val": VAL_SEED, "test": TEST_SEED,
                          "stats": STATS_SEED},
    "reserved_eval_seeds": list(RESERVED_EVAL_SEEDS),
    "scope": ("run-to-run stable on the same machine/build; bit-exact "
              "reproduction across different hardware or kernel libraries "
              "is not promised"),
}


def _run_context(settings: Settings) -> dict:
    """Self-contained provenance block attached to every result record.

    Code revision, hypothesis-registry hash, resolved configuration (every
    hyperparameter and seed actually used), software environment, the
    seeding policy and the data-stream fingerprints travel *with* each
    record, so a result file is traceable without relying on defaults that
    may have changed since.  The ``setup_sha256`` hash over exactly
    {revision, hypotheses, config, data fingerprints} gives every record a
    compact id for "same code + same recipe + same data": records carrying
    different ids were not produced by the same setup and cannot be pooled
    silently.  Cached on the Settings object: the environment probe (and
    git call) runs once per process.
    """
    ctx = getattr(settings, "_run_context_cache", None)
    if ctx is not None:
        return ctx
    config = {name: getattr(settings, name)
              for name in Settings.__dataclass_fields__ if name != "out_dir"}
    fingerprints = {
        name: stream_fingerprint(settings.n_pairs, settings.vocab,
                                 settings.device, seed=seed)
        for name, seed in (("val", VAL_SEED), ("test", TEST_SEED),
                           ("stats", STATS_SEED))
    }
    fingerprints["train"] = {
        str(seed): stream_fingerprint(settings.n_pairs, settings.vocab,
                                      settings.device, seed=seed)
        for seed in settings.seeds
    }
    payload = json.dumps({
        "git_sha": _git_sha(),
        "hypotheses_sha256": _hypotheses_sha(),
        "config": config,
        "data_fingerprints": fingerprints,
    }, ensure_ascii=False, sort_keys=True, default=str)
    ctx = {
        "git_sha": _git_sha(),
        "hypotheses_sha256": _hypotheses_sha(),
        "setup_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        "environment": _environment_record(settings),
        "config": config,
        "seeding": dict(_SEEDING_POLICY),
        "data_fingerprints": fingerprints,
    }
    settings._run_context_cache = ctx
    return ctx


def _write_manifest(settings: Settings, out_dir: pathlib.Path, tag: str) -> dict:
    """Per-experiment provenance manifest, written before any training.

    One ``<tag>_manifest.json`` per experiment pins the code revision
    (plus dirty flag), the pre-registered hypotheses and their hash, every
    hyperparameter and seed, the data-stream fingerprints (including each
    training stream) and the software environment.  Run records repeat
    this context inline; the manifest is the single artifact that makes a
    results directory traceable back to a reproducible setup without
    joining files by hand.
    """
    manifest = {
        "experiment": tag,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **_run_context(settings),
    }
    _write_json(out_dir / f"{tag}_manifest.json", manifest)
    return manifest


def _load_baseline_verdict(out_dir: pathlib.Path):
    """The HB1-HB5 verdicts written by ``run_baselines``, if present.

    E1/E2/E3 read the learned model against the simple baselines, so a
    missing verdict file is surfaced as a note instead of silently
    treating the baseline check as passed.  A verdict file written by an
    older version lacks the HB4/HB5 keys; those are reported as ``n/a``
    (not ``FAILED``) so the right response is to re-run
    ``--only baselines``, not to read the missing checks as failures.
    """
    path = out_dir / "baselines_verdict.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _model_cfg(settings: Settings, **overrides) -> dict:
    cfg = {
        "vocab_size": settings.vocab,
        "d_model": settings.d_model,
        "n_heads": settings.n_heads,
        "n_layers": settings.n_layers,
        "max_len": settings.max_len,
        "dropout": settings.dropout,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Seed-determinism self-check: the "one seed pins one run" contract
# ---------------------------------------------------------------------------
# ``cage.repro`` documents that same-seed runs are stable on one machine,
# but a claim in a docstring is not a measurement.  The audit below runs the
# *actual* entry points twice and compares bytes: data streams, weight
# initialisation and a short training trajectory.  A mismatch means some
# randomness is not under the seed (or the seed is ignored), in which case
# every seed-level statistic in a grid is only apparently repeatable.
_REPRO_AUDIT_BASE_SEED = 4242
_REPRO_AUDIT_STEPS = 5
_REPRO_AUDIT_BATCH = 8

_REPRO_AUDIT_KEYS = (
    ("data_same_seed_identical", "same seed draws byte-identical batches"),
    ("data_diff_seed_differs", "a different seed draws different data"),
    ("data_draw_leaves_global_rng_untouched",
     "data draws do not consume the global RNG (stream-owns-its-state)"),
    ("init_same_seed_identical", "same seed builds byte-identical parameters"),
    ("init_diff_seed_differs", "a different seed builds different parameters"),
    ("trajectory_same_seed_identical",
     "same seed replays the exact training-loss trajectory"),
    ("trajectory_after_global_rng_noise_identical",
     "global-RNG noise before a run does not change the trajectory "
     "(entry re-seeding covers it)"),
    ("trajectory_diff_seed_differs", "a different seed changes the trajectory"),
)


class ReproducibilityError(RuntimeError):
    """The seed-determinism audit failed; run results would not be reproducible."""


def _model_state_hash(model) -> str:
    """Content hash over every parameter tensor (process-order independent)."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(repr(p.detach().float().cpu().tolist()).encode("ascii"))
    return h.hexdigest()[:16]


def _probe_seed(settings: Settings, start: int) -> int:
    """First seed >= ``start`` that is neither reserved nor a grid seed."""
    seeds = {int(s) for s in settings.seeds}
    s = int(start)
    while s in RESERVED_EVAL_SEEDS or s in seeds:
        s += 1
    return s


def _reproducibility_audit(settings: Settings) -> dict:
    """Machine-check that one seed pins one run on the current machine.

    Everything is compared through the *real* entry points
    (``cage.data.make_batch`` via per-stream generators,
    ``cage.model.build``, ``cage.train.train_model``) on throwaway probe
    seeds that collide with neither the reserved evaluation streams nor the
    configured training seeds:

    * **data**: two fresh generators with the same seed produce the same
      batch fingerprint; a different seed produces a different one;
    * **RNG isolation**: drawing data leaves the global Python and Torch
      RNG state untouched (the DataLoader ``generator=`` equivalent);
    * **initialisation**: building the model twice under the same seed
      yields byte-identical parameters, a different seed yields different
      ones;
    * **trajectory**: a short training run replays the exact loss sequence
      under the same seed, is unaffected by global-RNG noise left by
      anything that ran before it (the entry re-seeds), and changes under a
      different seed.

    Scope follows ``cage.repro``: same machine and PyTorch build, not
    bit-exact reproduction across hardware/instruction sets.  Cached on
    ``Settings`` so one process (``--only all``) runs it once.
    """
    cached = getattr(settings, "_repro_audit_cache", None)
    if cached is not None:
        return cached
    device = settings.device
    seed = _probe_seed(settings, _REPRO_AUDIT_BASE_SEED)
    other = _probe_seed(settings, seed + 1)
    cfg = _model_cfg(settings, gate="calib", d_model=48, n_layers=2)

    def _draw(s):
        gen = stream_generator(s, device)
        return make_batch(settings.overfit_batch, settings.n_pairs,
                          settings.vocab, device, gen=gen)

    fp_same_a = batch_fingerprint(*_draw(seed))
    fp_same_b = batch_fingerprint(*_draw(seed))
    fp_other = batch_fingerprint(*_draw(other))

    torch_state = torch.get_rng_state()
    py_state = random.getstate()
    cuda_states = (torch.cuda.get_rng_state_all()
                   if device.startswith("cuda") and torch.cuda.is_available()
                   else None)
    _draw(seed)
    rng_untouched = bool(torch.equal(torch.get_rng_state(), torch_state))
    rng_untouched = rng_untouched and random.getstate() == py_state
    if cuda_states is not None:
        rng_untouched = rng_untouched and all(
            torch.equal(a, b) for a, b in
            zip(torch.cuda.get_rng_state_all(), cuda_states))

    seed_everything(seed)
    init_same_a = _model_state_hash(build(cfg, compile=False))
    seed_everything(seed)
    init_same_b = _model_state_hash(build(cfg, compile=False))
    seed_everything(other)
    init_other = _model_state_hash(build(cfg, compile=False))

    def _trajectory(s, rng_noise=False):
        if rng_noise:
            torch.manual_seed(s + 777)
            random.seed(s + 777)
        with contextlib.redirect_stdout(io.StringIO()):
            seed_everything(s)
            model = build(cfg, compile=False)
            hist = train_model(model, steps=_REPRO_AUDIT_STEPS,
                               batch=_REPRO_AUDIT_BATCH,
                               n_pairs=settings.n_pairs, lr=1e-3,
                               vocab=settings.vocab, device=device,
                               log_every=1, seed=s)
        return [float(h["train_loss"]) for h in hist]

    traj_a = _trajectory(seed)
    traj_replay = _trajectory(seed)
    traj_noisy = _trajectory(seed, rng_noise=True)
    traj_other = _trajectory(other)

    audit = {
        "probe_seed": int(seed),
        "other_seed": int(other),
        "device": device,
        "steps": _REPRO_AUDIT_STEPS,
        "batch": _REPRO_AUDIT_BATCH,
        "data_same_seed_sha256": fp_same_a,
        "data_diff_seed_sha256": fp_other,
        "init_same_seed_sha256": init_same_a,
        "trajectory_losses": traj_a,
        "data_same_seed_identical": bool(fp_same_a == fp_same_b),
        "data_diff_seed_differs": bool(fp_same_a != fp_other),
        "data_draw_leaves_global_rng_untouched": rng_untouched,
        "init_same_seed_identical": bool(init_same_a == init_same_b),
        "init_diff_seed_differs": bool(init_same_a != init_other),
        "trajectory_same_seed_identical": bool(traj_a == traj_replay),
        "trajectory_after_global_rng_noise_identical": bool(traj_a == traj_noisy),
        "trajectory_diff_seed_differs": bool(traj_a != traj_other),
        "deterministic_algorithms": bool(getattr(
            torch, "are_deterministic_algorithms_enabled", lambda: False)()),
        "scope": ("same machine + same PyTorch build; bit-exact reproduction "
                  "across different hardware is not claimed"),
    }
    audit["passed"] = all(bool(audit[key]) for key, _ in _REPRO_AUDIT_KEYS)
    settings._repro_audit_cache = audit
    return audit


def _enforce_reproducibility(settings: Settings, out_dir: pathlib.Path,
                             tag: str) -> dict:
    """Refuse to run a grid whose same-seed reruns are not repeatable.

    Stores the audit under ``<tag>_repro_audit.json`` and aborts with
    :class:`ReproducibilityError` when any check fails: a grid whose
    seed-level statistics are not reproducible cannot underpin any claim,
    so it is not started (same policy as the HB5 overfit gate).
    """
    audit = _reproducibility_audit(settings)
    _write_json(out_dir / f"{tag}_repro_audit.json",
                {**audit, **_run_context(settings)})
    print(f"[{tag}] seed-determinism audit (probe seeds "
          f"{audit['probe_seed']}/{audit['other_seed']}, "
          f"{audit['steps']}-step trajectories):")
    for key, text in _REPRO_AUDIT_KEYS:
        mark = "ok" if audit[key] else "FAILED"
        print(f"    {mark:>6}: {text}")
    if not audit["passed"]:
        failed = [key for key, _ in _REPRO_AUDIT_KEYS if not audit[key]]
        _write_json(out_dir / f"{tag}_aborted.json",
                    {"experiment": tag,
                     "reason": "seed-determinism audit failed",
                     "failed_checks": failed,
                     "repro_audit": audit,
                     **_run_context(settings)})
        raise ReproducibilityError(
            f"{tag}: seed-determinism audit failed for {failed}; same-seed "
            f"reruns are not stable on this machine, so every seed-level "
            f"statistic produced by the grid would be unrepeatable")
    return audit


def _curve_record(hist):
    """Per-run learning-curve summary (monitor the curve, not just the end).

    Stores both the reduced verdict and the raw series (steps, train loss,
    validation loss, pre-clip gradient norm), so a run that is later read
    as overfitting/plateauing can be re-examined without retraining.
    """
    verdict = training_curve_verdict(hist)
    train = [h for h in hist if _finite(h.get("train_loss"))]
    val = [h for h in hist if h.get("val_loss") is not None
           and _finite(h.get("val_loss"))]
    return {
        "train_loss_first": train[0]["train_loss"] if train else float("nan"),
        "train_loss_last": train[-1]["train_loss"] if train else float("nan"),
        "val_loss_first": val[0]["val_loss"] if val else float("nan"),
        "val_loss_last": val[-1]["val_loss"] if val else float("nan"),
        "train_delta": verdict["train_delta"],
        "val_delta": verdict["val_delta"],
        "generalization_gap": verdict["generalization_gap"],
        "best_val_loss": verdict["best_val_loss"],
        "best_val_step": verdict["best_val_step"],
        "val_regression_after_best": verdict["val_regression_after_best"],
        "late_overfit": verdict["late_overfit"],
        "grad_norm_max": verdict["grad_norm_max"],
        "grad_norm_explode_count": verdict["grad_norm_explode_count"],
        "curve_steps": [int(h["step"]) for h in train],
        "curve_train_loss": [float(h["train_loss"]) for h in train],
        "curve_val_loss": [None if h.get("val_loss") is None
                           else float(h["val_loss"]) for h in train],
        "curve_grad_norm": [None if h.get("grad_norm") is None
                            else float(h["grad_norm"]) for h in train],
        "curve_verdict": verdict["verdict"],
        "curve_recommendation": verdict["recommendation"],
    }


def _cost_record(settings: Settings, model):
    return measure_inference_cost(
        model, settings.n_pairs, settings.vocab, settings.device,
        batch=min(settings.eval_batch, 128), warmup=settings.cost_warmup,
        runs=settings.cost_runs)


def _train_and_eval(settings: Settings, seed: int, lr: float,
                    eval_seed: int = TEST_SEED, permute_labels: bool = False,
                    **model_overrides):
    """Train + score one run through the single shared entry point.

    Returns ``(model, params, metrics, curve, model_cfg)`` where
    ``model_cfg`` is the exact resolved configuration the model was built
    from - every ``Cfg`` field including defaults, so an unset ablation flag
    cannot be mistaken for an off switch.  Run records keep it so the
    controlled-variable audit can check the *trained* configuration
    directly (ground truth) instead of trusting the hand-attached labels.
    """
    if eval_seed in tuple(settings.seeds):
        raise ValueError(
            f"eval_seed={eval_seed} is also a configured training seed; "
            f"evaluation data must come from a stream that never trained "
            f"the model"
        )
    seed_everything(seed, deterministic=settings.deterministic)
    model = build(_model_cfg(settings, **model_overrides), compile=False)
    model_cfg = asdict(model.cfg)
    params = sum(p.numel() for p in model.parameters())
    eval_every = (max(1, settings.steps // settings.curve_points)
                  if settings.curve_points > 0 else 0)
    hist = train_model(model, steps=settings.steps, batch=settings.batch,
                       n_pairs=settings.n_pairs, lr=lr, vocab=settings.vocab,
                       device=settings.device, log_every=0, seed=seed,
                       eval_every=eval_every, val_batches=settings.val_batches,
                       permute_labels=permute_labels)
    metrics = eval_metrics(model, settings.n_pairs, settings.vocab,
                           settings.device, batch=settings.eval_batch,
                           batches=settings.eval_batches, seed=eval_seed)
    return model, params, metrics, _curve_record(hist), model_cfg


def tune_shared_recipe(settings: Settings):
    """Tune the learning rate on the plain ``none`` baseline.

    Selection uses the validation seed only (up to two training seeds);
    the winning rate is then frozen for *every* variant, so a gate variant
    can never win via a luckier optimizer setting - the E1/E3 comparisons
    remain controlled.

    The learning-curve verdict of every tuning run is stored with the
    table: if the selected rate's runs are all curve-invalid (plateau /
    diverging / instability) the selection was made between underfit or
    broken runs, and the frozen recipe cannot be trusted to expose the
    mechanism under test - ``_resolve_lr`` prints that warning.
    """
    tune_seeds = settings.seeds[:max(1, min(2, len(settings.seeds)))]
    table = []
    for lr in settings.lr_grid:
        accs, verdicts = [], []
        for seed in tune_seeds:
            _, _, metrics, curve, _ = _train_and_eval(
                settings, seed, lr, eval_seed=VAL_SEED, gate="none")
            accs.append(metrics["accuracy"])
            verdicts.append(curve["curve_verdict"])
        mean = sum(accs) / len(accs)
        var = (sum((a - mean) ** 2 for a in accs) / len(accs)) if accs else 0.0
        table.append({"lr": lr, "val_accuracy_mean": mean,
                      "val_accuracy_std": var ** 0.5, "n_seeds": len(accs),
                      "curve_verdicts": verdicts})
    best = max(table, key=lambda row: row["val_accuracy_mean"])
    return best["lr"], table


def _resolve_lr(settings: Settings, out_dir: pathlib.Path, tag: str):
    """Return the shared recipe lr, tuning at most once per Settings object
    so E1/E2/E3 in one ``--only all`` run train under an identical recipe
    (the tuning runs only once)."""
    if settings.lr is not None:
        return settings.lr
    cached = getattr(settings, "_tuned_recipe", None)
    if cached is None:
        cached = tune_shared_recipe(settings)
        settings._tuned_recipe = cached
        print(f"[{tag}] shared recipe tuned on gate=none (val seed {VAL_SEED}): "
              f"lr={cached[0]:g}")
        for row in cached[1]:
            verdicts = {}
            for v in row.get("curve_verdicts", ()):
                verdicts[v] = verdicts.get(v, 0) + 1
            print(f"    lr={row['lr']:<8g} "
                  f"val_acc={row['val_accuracy_mean']:.4f} "
                  f"+- {row['val_accuracy_std']:.4f}  curves={verdicts}")
        chosen = next(r for r in cached[1] if r["lr"] == cached[0])
        invalid = [v for v in chosen.get("curve_verdicts", ())
                   if v in CURVE_INVALID_VERDICTS]
        if invalid:
            print(f"[{tag}] WARNING: the selected lr={cached[0]:g} has "
                  f"{len(invalid)}/{len(chosen['curve_verdicts'])} "
                  f"curve-invalid tuning run(s) ({sorted(set(invalid))}): the "
                  f"selection is between underfit/broken curves, so the frozen "
                  f"recipe may make every variant look equally flat")
    else:
        print(f"[{tag}] reusing shared tuned recipe lr={cached[0]:g}")
    lr, table = cached
    _write_json(out_dir / f"{tag}_lr_tuning.json",
                {"val_seed": VAL_SEED, "table": table, "chosen_lr": lr})
    return lr


def _answer_balance(settings: Settings, batch=256, batches=4, seed=VAL_SEED):
    gen = torch.Generator(device=settings.device).manual_seed(seed)
    counts = torch.zeros(settings.vocab, dtype=torch.float64)
    for _ in range(batches):
        _, _, meta = make_batch(batch, settings.n_pairs, settings.vocab,
                                settings.device, gen=gen)
        counts += torch.bincount(meta["answer"].cpu(), minlength=settings.vocab)
    present = counts > 0
    shares = counts[present] / counts.sum()
    return {
        "n_samples": int(counts.sum()),
        "n_classes_possible": settings.vocab - 2,
        "n_classes_present": int(present.sum()),
        "min_class_share": float(shares.min()),
        "max_class_share": float(shares.max()),
        "expected_uniform_share": 1.0 / (settings.vocab - 2),
    }


def _audit_data_splits(settings: Settings, out_dir: pathlib.Path) -> dict:
    """HB3: verify the train/val/test streams are disjoint and IID before
    any model is trained.

    The audit only reads data (it selects nothing), but a shared sample or
    a distribution gap between train and the held-out splits invalidates
    every number reported afterwards, so the outcome is printed and stored
    next to the baselines.  Distribution shift is judged against the
    same-stream sampling-noise floor instead of an arbitrary constant.
    """
    train_seed = settings.seeds[0] if settings.seeds else 0
    audit = split_leakage_audit(
        settings.n_pairs, settings.vocab, settings.device,
        batch=settings.eval_batch, batches=settings.eval_batches,
        train_seed=train_seed)
    floor = audit["token_tv_noise_floor"]
    tol = max(3.0 * floor, 0.02) if math.isfinite(floor) else float("inf")
    overlaps = {name: audit[f"duplicate_overlap_train_{name}"]
                for name in ("val", "test", "stats")}
    tvs = {name: audit[f"token_tv_train_{name}"]
           for name in ("val", "test", "stats")}
    overlap_ok = all(v == 0 for v in overlaps.values())
    shift_ok = all(v <= tol for v in tvs.values())
    audit["tv_tolerance"] = tol
    audit["hb3_supported"] = bool(overlap_ok and shift_ok)
    detail = "; ".join(
        f"{name}: overlap={overlaps[name]} tv={tvs[name]:.5f}"
        for name in ("val", "test", "stats"))
    print(f"[baselines] split audit (HB3, train seed {audit['train_seed']}): "
          f"{detail} (same-stream noise floor {floor:.5f}, tol {tol:.5f}); "
          f"unique_key_rate={audit['unique_key_rate']:.4f}")
    if not audit["hb3_supported"]:
        print("[baselines] WARNING: HB3 split/disjointness audit failed; "
              "all reported numbers are suspect")
    _write_json(out_dir / "baselines_split_audit.json", audit)
    return audit


def _run_shallow_baseline(settings: Settings, out_dir: pathlib.Path,
                          ceiling: dict | None) -> dict:
    """HB6 complexity control: the tuned single-layer standard transformer.

    The simplest *learned* model on the baseline ladder.  It goes through
    the same entry point as the deep grids (``_train_and_eval``): same
    optimizer factory, cosine schedule, batch geometry and evaluation
    streams, with the learning rate tuned on the validation seed (up to
    two seeds per grid point) and the chosen rate reported on the disjoint
    test seed.  Depth is the only architectural difference from the
    ``none`` baseline of E1, so the gap to the rule ceiling measures how
    much depth the recall readout actually requires, and a complex model
    is only worth crediting for what it does beyond *this* tuned baseline
    (E1 pairs the 8-layer ``none`` runs against it as a complexity
    reference).
    """
    tune_seeds = settings.seeds[:max(1, min(2, len(settings.seeds)))]
    table = []
    for lr in settings.lr_grid:
        accs = []
        for seed in tune_seeds:
            _, _, metrics, _, _ = _train_and_eval(
                settings, seed, lr, eval_seed=VAL_SEED, gate="none",
                n_layers=1)
            accs.append(metrics["accuracy"])
        mean = sum(accs) / len(accs)
        std = ((sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5
               if len(accs) > 1 else 0.0)
        table.append({"lr": lr, "val_accuracy_mean": mean,
                      "val_accuracy_std": std, "n_seeds": len(accs)})
    best = max(table, key=lambda row: row["val_accuracy_mean"])
    runs = []
    for seed in settings.seeds:
        _, params, metrics, curve, _ = _train_and_eval(
            settings, seed, best["lr"], eval_seed=TEST_SEED, gate="none",
            n_layers=1)
        n = int(metrics["n"])
        runs.append({
            "seed": seed, "accuracy": metrics["accuracy"],
            "accuracy_lo": metrics["accuracy_lo"],
            "accuracy_hi": metrics["accuracy_hi"], "n": n,
            "correct": int(round(metrics["accuracy"] * n)),
            "params": params, "curve_verdict": curve["curve_verdict"],
        })
        _write_result(out_dir / "baselines_runs.jsonl",
                      {"experiment": "baselines", "hypotheses": ["HB6"],
                       "name": "shallow_none_baseline", "lr": best["lr"],
                       "training_seed": seed, "accuracy": metrics["accuracy"],
                       "accuracy_lo": metrics["accuracy_lo"],
                       "accuracy_hi": metrics["accuracy_hi"], "n": n,
                       **_run_context(settings), **curve})
    pooled = wilson_ci(sum(r["correct"] for r in runs),
                       sum(r["n"] for r in runs))
    summary = _mean_std([r["accuracy"] for r in runs])
    ceiling_acc = float(ceiling["accuracy"]) if ceiling else float("nan")
    hb6_supported = bool(
        math.isfinite(ceiling_acc) and math.isfinite(pooled["hi"])
        and pooled["hi"] < ceiling_acc)
    payload = {
        "name": "shallow_none", "n_layers": 1,
        "val_seed": VAL_SEED, "test_seed": TEST_SEED,
        "chosen_lr": best["lr"], "tuning_table": table,
        "accuracy_mean": summary["mean"], "accuracy_std": summary["std"],
        "accuracy_by_seed": {str(r["seed"]): r["accuracy"] for r in runs},
        "pooled_wilson": pooled, "ceiling": ceiling_acc,
        "ceiling_gap": (ceiling_acc - summary["mean"]
                        if math.isfinite(ceiling_acc) else float("nan")),
        "hb6_supported": hb6_supported, "runs": runs,
    }
    print(f"[baselines] HB6 complexity control: tuned 1-layer standard "
          f"transformer (lr={best['lr']:g} on val seed {VAL_SEED}, test seed "
          f"{TEST_SEED}): accuracy={summary['mean']:.4f} +- "
          f"{summary['std']:.4f} (pooled Wilson "
          f"[{pooled['lo']:.4f}, {pooled['hi']:.4f}], n={pooled['n']}); "
          f"rule ceiling={ceiling_acc:.4f}, gap="
          f"{payload['ceiling_gap']:.4f}; "
          f"{'SUPPORTED (a single layer is not enough: depth is measurably required)' if hb6_supported else 'FLAG - the shallow baseline reaches the rule ceiling within its CI: annotate depth-based (E3) motivations'}")
    _write_json(out_dir / "baselines_shallow_none.json", payload)
    _write_json(out_dir / "baselines_shallow_tuning.json",
                {"val_seed": VAL_SEED, "table": table,
                 "chosen_lr": best["lr"]})
    return payload


class OverfitSanityError(RuntimeError):
    """The tiny-batch overfit pre-flight failed; grid training is refused."""


# Tiny-batch overfit configurations of the HB5 pre-flight: every gate
# variant plus the E3 architecture paths (norm placement, dynamic
# normalisation, LayerScale).  A broken backward on a normalisation or
# write-strength path would survive a gate-only check yet make every E3
# number uninterpretable, so those paths get the same memorisation test.
_SANITY_CONFIGS = (
    ("gate=none", {"gate": "none"}),
    ("gate=query", {"gate": "query"}),
    ("gate=calib", {"gate": "calib"}),
    ("gate=null", {"gate": "null"}),
    ("gate=calib,norm=post", {"gate": "calib", "norm": "post"}),
    ("gate=calib,norm=mix", {"gate": "calib", "norm": "mix"}),
    ("gate=calib,norm=seednorm", {"gate": "calib", "norm": "seednorm"}),
    ("gate=calib,layer_scale=1e-2", {"gate": "calib", "layer_scale": 1e-2}),
)
# The plain baseline arm whose row supplies the shared batch/step/lr fields.
_SANITY_REFERENCE = "gate=none"


def _overfit_sanity_by_config(settings: Settings) -> dict:
    """Tiny-batch memorisation check for every trained path (cached).

    One fixed batch from ``SANITY_SEED`` is pushed through the tiny 2-layer
    model of each configuration with dropout off and weight decay zeroed, so
    every custom kernel's backward path *and* every normalisation /
    LayerScale path the grids train is exercised - a passing ``gate=none``
    check alone would not prove the calib/null gates or the
    seednorm/Post-LN/LayerScale paths backprop.  Cached on ``Settings``:
    E1/E2/E3 in one ``--only all`` run share a single execution.
    """
    cached = getattr(settings, "_overfit_sanity_cache", None)
    if cached is not None:
        return cached
    gen = stream_generator(SANITY_SEED, settings.device)
    sidx, stgt, smeta = make_batch(settings.overfit_batch, settings.n_pairs,
                                   settings.vocab, settings.device, gen=gen)
    validate_batch(sidx, stgt, smeta, settings.vocab)
    by_config = {}
    for label, overrides in _SANITY_CONFIGS:
        seed_everything(SANITY_SEED)
        tiny = build(_model_cfg(settings, d_model=48, n_layers=2, **overrides),
                     compile=False)
        by_config[label] = overfit_sanity_check(
            tiny, sidx, stgt, smeta, lr=settings.overfit_lr,
            steps=settings.overfit_steps, seed=SANITY_SEED)
    settings._overfit_sanity_cache = by_config
    return by_config


def _enforce_overfit_gate(settings: Settings, out_dir: pathlib.Path,
                          tag: str) -> dict:
    """Refuse to start a grid when the loop cannot memorise a tiny batch.

    Runs the check on the *current* code revision (a stale
    ``baselines_verdict.json`` cannot satisfy it), stores the per-config
    evidence under ``<tag>_overfit_sanity.json`` and aborts with
    :class:`OverfitSanityError` when any configuration fails: an invalid
    training loop would make every number a grid produces unreliable, so
    the grid is not started.
    """
    by_config = _overfit_sanity_by_config(settings)
    passed = all(row["passed"] for row in by_config.values())
    sanity = by_config[_SANITY_REFERENCE]
    summary = {
        "passed": passed,
        "batch": int(sanity["batch"]),
        "steps": int(sanity["steps"]),
        "lr": float(sanity["lr"]),
        "seed": sanity["seed"],
        "by_config": {
            label: {
                "passed": row["passed"],
                "loss_first": row["loss_first"],
                "loss_last": row["loss_last"],
                "acc_first": row["acc_first"],
                "acc_last": row["acc_last"],
                "steps_to_target_loss": row["steps_to_target_loss"],
                "steps_to_target_acc": row["steps_to_target_acc"],
                "data_sha256": row["data_sha256"],
                "params_connected": (
                    f"{row['grad_flow']['params_with_grad']}/"
                    f"{row['grad_flow']['params_total']}"),
                "all_grads_finite": row["grad_flow"]["all_grads_finite"],
                "active_dropout": row["regularization"]["active_dropout"],
                "weight_decay": row["regularization"]["weight_decay"],
            }
            for label, row in by_config.items()
        },
    }
    _write_json(out_dir / f"{tag}_overfit_sanity.json",
                {**summary, **_run_context(settings)})
    print(f"[{tag}] overfit sanity gate (HB5; tiny batch of "
          f"{summary['batch']} samples, {summary['steps']} steps, lr="
          f"{summary['lr']:g}, seed {summary['seed']}, data "
          f"{sanity['data_sha256']}; dropout off, weight decay zeroed, no "
          f"augmentation):")
    for label, row in summary["by_config"].items():
        loss_s = (f"{row['loss_first']:.4f} -> {row['loss_last']:.4f}"
                  if _finite(row["loss_first"]) and _finite(row["loss_last"])
                  else "non-finite")
        print(f"    {label:>28}: loss {loss_s}, batch acc "
              f"{row['acc_last']:.3f}, steps to loss/acc target "
              f"{row['steps_to_target_loss']}/{row['steps_to_target_acc']}, "
              f"grads {row['params_connected']} "
              f"finite={row['all_grads_finite']}; "
              f"{'passed' if row['passed'] else 'FAILED'}")
    if not passed:
        failed = [label for label, row in summary["by_config"].items()
                  if not row["passed"]]
        _write_json(out_dir / f"{tag}_aborted.json",
                    {"experiment": tag,
                     "reason": "HB5 tiny-batch overfit gate failed",
                     "failed_configs": failed,
                     "overfit_sanity": summary,
                     **_run_context(settings)})
        raise OverfitSanityError(
            f"{tag}: tiny-batch overfit sanity failed for {failed}; "
            f"address the training loop before running the grid (this "
            f"check is necessary, not sufficient: data/label problems can "
            f"still survive memorisation)")
    return summary


def run_baselines(settings: Settings, out_dir: pathlib.Path) -> None:
    _announce(("HB1", "HB2", "HB3", "HB4", "HB5", "HB6"))
    _write_manifest(settings, out_dir, "baselines")
    repro_audit = _enforce_reproducibility(settings, out_dir, "baselines")
    record = {"experiment": "baselines",
              "hypotheses": ["HB1", "HB2", "HB3", "HB4", "HB5", "HB6"],
              "seed": TEST_SEED, **_run_context(settings)}

    audit = _audit_data_splits(settings, out_dir)

    balance = _answer_balance(settings)
    print(f"[baselines] answer-value balance over {balance['n_samples']} "
          f"samples: {balance['n_classes_present']} classes present, share "
          f"min={balance['min_class_share']:.5f} max={balance['max_class_share']:.5f} "
          f"(uniform={balance['expected_uniform_share']:.5f})")
    _write_json(out_dir / "baselines_balance.json", balance)

    # HB5 first: a broken training loop invalidates every learned number
    # below it (including the negative control).  The tiny batch is run
    # through *every* trained path: each gate variant (so the calib/null
    # parameters' backward paths are exercised - a passing gate=none check
    # alone would not prove the calib gate backprops) plus the E3
    # normalisation/LayerScale paths.  Dropout is forced off inside the
    # check and per-parameter gradient connectivity/finiteness is audited
    # on the first backward.  The same shared helper is the hard pre-flight
    # gate of the E1/E2/E3 runners, so the baseline report and the grids
    # can never disagree about it.
    sanity_by_config = _overfit_sanity_by_config(settings)
    sanity = sanity_by_config[_SANITY_REFERENCE]
    hb5_supported = all(row["passed"] for row in sanity_by_config.values())
    print(f"[baselines] HB5 tiny-batch overfit sanity (batch="
          f"{sanity['batch']}, steps={sanity['steps']}, lr={sanity['lr']:g}; "
          f"dropout forced off, no augmentation, gradient connectivity "
          f"audited):")
    for label, row in sanity_by_config.items():
        flow = row["grad_flow"]
        print(f"    {label:>28}: loss {row['loss_first']:.4f} -> "
              f"{row['loss_last']:.4f}, batch acc={row['acc_last']:.3f}, "
              f"grads connected={flow['params_with_grad']}/"
              f"{flow['params_total']}, finite={flow['all_grads_finite']}; "
              f"{'passed' if row['passed'] else 'FAILED'}")
    print(f"  {'SUPPORTED (every trained path can memorise a tiny batch)' if hb5_supported else 'FAILED - gradients/loss/precision are broken in at least one trained path, do not trust any grid result'}")
    overfit_payload = {**sanity, "passed": hb5_supported,
                       "by_config": sanity_by_config}
    _write_json(out_dir / "baselines_overfit_sanity.json", overfit_payload)
    for label, overrides in _SANITY_CONFIGS:
        _write_result(out_dir / "baselines_runs.jsonl",
                      {**record, "name": f"overfit_sanity_{label}",
                       "config": label,
                       "gate": overrides.get("gate"),
                       **sanity_by_config[label]})

    hb1_supported = None
    rule_stats = None
    for name, fn in (
        ("rule_pair_lookup", rule_pair_lookup_stats),
        ("rule_naive_last_match", rule_naive_last_match_stats),
        ("copy_last_token", copy_last_token_stats),
    ):
        stats = fn(settings.n_pairs, settings.vocab, settings.device,
                   batch=settings.eval_batch, batches=settings.eval_batches,
                   seed=TEST_SEED)
        print(f"[baselines] {name}: accuracy={stats['accuracy']:.4f} "
              f"[{stats['accuracy_lo']:.4f}, {stats['accuracy_hi']:.4f}] "
              f"(Wilson 95%, n={stats['n']})")
        _write_result(out_dir / "baselines_runs.jsonl",
                      {**record, "name": name, **stats})
        if name == "rule_pair_lookup":
            rule_stats = stats
            hb1_supported = bool(stats["accuracy"] >= 0.99
                                 and stats["unique_key_rate"] == 1.0)
    if not hb1_supported:
        print("[baselines] WARNING: HB1 failed - the parameter-free rule does "
              "not reach the ceiling, so the task/data construction is broken "
              "and every number reported above is invalid")

    best_lr, table = tune_linear_probe(
        settings.n_pairs, settings.vocab, settings.device,
        lr_grid=settings.lr_grid, seeds=settings.seeds[:max(1, len(settings.seeds))],
        steps=settings.probe_steps, batch=settings.probe_batch,
        val_seed=VAL_SEED, val_batches=settings.eval_batches,
        probe_batch=settings.eval_batch)
    _write_json(out_dir / "baselines_probe_tuning.json",
                {"val_seed": VAL_SEED, "table": table, "chosen_lr": best_lr})
    print(f"[baselines] linear probe tuned on val seed {VAL_SEED}: lr={best_lr:g}")
    probe_accs = []
    for seed in settings.seeds:
        probe_curve = []
        probe = train_linear_probe(settings.n_pairs, settings.vocab,
                                   settings.device, steps=settings.probe_steps,
                                   batch=settings.probe_batch, lr=best_lr,
                                   seed=seed, history=probe_curve)
        acc = linear_probe_accuracy(probe, settings.n_pairs, settings.vocab,
                                    settings.device, batch=settings.eval_batch,
                                    batches=settings.eval_batches, seed=TEST_SEED)
        probe_accs.append(acc)
        _write_result(out_dir / "baselines_runs.jsonl",
                      {**record, "name": "linear_bag_probe", "lr": best_lr,
                       "training_seed": seed, "accuracy": acc,
                       "train_loss_first": probe_curve[0]["loss"],
                       "train_loss_last": probe_curve[-1]["loss"],
                       "probe_curve": probe_curve,
                       "n": settings.eval_batch * settings.eval_batches})
    summary = _mean_std(probe_accs)
    chance = 1.0 / (settings.vocab - 2)
    hb2 = probe_vs_chance_test(probe_accs, chance,
                               seed=settings.seeds[0] if settings.seeds else 0)
    print(f"[baselines] tuned linear probe (test seed {TEST_SEED}): "
          f"accuracy={summary['mean']:.4f} +- {summary['std']:.4f} "
          f"(chance={chance:.4f}, n_seeds={summary['n']})")
    print(f"[baselines] HB2 near-chance test: accuracy - chance = "
          f"{hb2['mean']:+.4f} [{hb2['lo']:+.4f}, {hb2['hi']:+.4f}] "
          f"(p={hb2['p_value']:.3f}, MDE80={hb2['mde80']:.4f}); "
          f"{'SUPPORTED (probe confidently <2x chance)' if hb2['hb2_supported'] else 'NOT SUPPORTED - inspect the simple-baseline comparison'}")

    # HB6 complexity control: the simplest learned model must be tried --
    # and tuned -- before the deep grids can claim that depth/gating is
    # needed.  Runs before the negative control so a shallow-model finding
    # (single layer already solves the task) is on record next to it.
    shallow = _run_shallow_baseline(settings, out_dir, rule_stats)

    # HB4 negative control: same architecture and shared recipe as E1, but
    # the supervised answers are permuted across the batch during training.
    # Chance-level test accuracy is the expected outcome; above-chance means
    # answer information leaks into the pipeline.
    lr_shared = _resolve_lr(settings, out_dir, "baselines")
    ctrl_runs = []
    for seed in settings.seeds:
        _, _, metrics, curve, _ = _train_and_eval(
            settings, seed, lr_shared, gate="none", permute_labels=True)
        n = int(metrics["n"])
        ctrl_runs.append({
            "seed": seed, "accuracy": metrics["accuracy"],
            "accuracy_lo": metrics["accuracy_lo"],
            "accuracy_hi": metrics["accuracy_hi"], "n": n,
            "correct": int(round(metrics["accuracy"] * n)),
            "train_loss_last": curve["train_loss_last"],
            "curve_verdict": curve["curve_verdict"],
        })
        _write_result(out_dir / "baselines_runs.jsonl",
                      {**record, "name": "negative_control_permuted_labels",
                       "lr": lr_shared, "training_seed": seed,
                       "accuracy": metrics["accuracy"],
                       "accuracy_lo": metrics["accuracy_lo"],
                       "accuracy_hi": metrics["accuracy_hi"], "n": n,
                       **curve})
    margins = [r["accuracy"] - chance for r in ctrl_runs]
    hb4 = paired_bootstrap_ci(margins)
    pooled = wilson_ci(sum(r["correct"] for r in ctrl_runs),
                       sum(r["n"] for r in ctrl_runs))
    ctrl_summary = _mean_std([r["accuracy"] for r in ctrl_runs])
    hb4_supported = bool(hb4["n"] > 0
                         and hb4["hi"] < settings.negative_control_tol)
    print(f"[baselines] HB4 negative control (labels permuted in training, "
          f"shared recipe lr={lr_shared:g}, test seed {TEST_SEED}): "
          f"accuracy={ctrl_summary['mean']:.4f} +- {ctrl_summary['std']:.4f} "
          f"over {ctrl_summary['n']} seeds; accuracy - chance = "
          f"{hb4['mean']:+.4f} [{hb4['lo']:+.4f}, {hb4['hi']:+.4f}] "
          f"(pooled Wilson [{pooled['lo']:.4f}, {pooled['hi']:.4f}], "
          f"tol={settings.negative_control_tol:g}); "
          f"{'SUPPORTED (no label leakage detected)' if hb4_supported else 'FAILED - the control beats chance: answer information leaks into the pipeline'}")
    _write_json(out_dir / "baselines_negative_control.json", {
        "lr": lr_shared, "chance": chance,
        "tolerance": settings.negative_control_tol,
        "accuracy_mean": ctrl_summary["mean"],
        "accuracy_std": ctrl_summary["std"],
        "margin": hb4, "pooled_wilson": pooled, "runs": ctrl_runs,
        "hb4_supported": hb4_supported,
    })

    verdict = {
        "train_seed": settings.seeds[0] if settings.seeds else None,
        "test_seed": TEST_SEED,
        "val_seed": VAL_SEED,
        "chance": chance,
        "repro_audit_passed": bool(repro_audit["passed"]),
        "repro_audit": {
            "probe_seed": repro_audit["probe_seed"],
            "other_seed": repro_audit["other_seed"],
            "steps": repro_audit["steps"],
            "checks": {key: bool(repro_audit[key])
                       for key, _ in _REPRO_AUDIT_KEYS},
        },
        "hb1_supported": bool(hb1_supported),
        "hb2_supported": bool(hb2["hb2_supported"]),
        "hb3_supported": bool(audit.get("hb3_supported")),
        "hb4_supported": hb4_supported,
        "hb5_supported": bool(hb5_supported),
        "hb6_supported": bool(shallow["hb6_supported"]),
        "probe_accuracy_mean": summary["mean"],
        "probe_accuracy_std": summary["std"],
        "shallow_none": {
            "n_layers": 1, "lr": shallow["chosen_lr"],
            "tuning_table": shallow["tuning_table"],
            "accuracy_mean": shallow["accuracy_mean"],
            "accuracy_std": shallow["accuracy_std"],
            "accuracy_by_seed": shallow["accuracy_by_seed"],
            "pooled_wilson": shallow["pooled_wilson"],
            "ceiling": shallow["ceiling"],
            "ceiling_gap": shallow["ceiling_gap"],
        },
        "hb2_accuracy_minus_chance": {
            "mean": hb2["mean"], "lo": hb2["lo"], "hi": hb2["hi"],
            "p_value": hb2["p_value"], "mde80": hb2["mde80"],
            "n": hb2["n"],
        },
        "hb4_negative_control": {
            "lr": lr_shared, "tolerance": settings.negative_control_tol,
            "accuracy_mean": ctrl_summary["mean"],
            "accuracy_std": ctrl_summary["std"],
            "margin": hb4, "pooled_wilson": pooled,
        },
        "hb5_overfit_sanity": {
            "passed": bool(hb5_supported),
            "loss_first": sanity["loss_first"],
            "loss_last": sanity["loss_last"],
            "acc_last": sanity["acc_last"],
            "steps": sanity["steps"], "lr": sanity["lr"],
            "batch": sanity["batch"],
            "by_config": {
                label: {
                    "passed": row["passed"],
                    "loss_last": row["loss_last"],
                    "acc_last": row["acc_last"],
                    "params_connected": (
                        f"{row['grad_flow']['params_with_grad']}/"
                        f"{row['grad_flow']['params_total']}"),
                    "all_grads_finite": row["grad_flow"]["all_grads_finite"],
                    "active_dropout": row["regularization"]["active_dropout"],
                }
                for label, row in sanity_by_config.items()
            },
        },
    }
    _write_json(out_dir / "baselines_verdict.json", verdict)
    print(f"[baselines] verdict: HB1={'ok' if verdict['hb1_supported'] else 'FAILED'}"
          f" HB2={'ok' if verdict['hb2_supported'] else 'FAILED'}"
          f" HB3={'ok' if verdict['hb3_supported'] else 'FAILED'}"
          f" HB4={'ok' if verdict['hb4_supported'] else 'FAILED'}"
          f" HB5={'ok' if verdict['hb5_supported'] else 'FAILED'}"
          f" HB6={'ok' if verdict['hb6_supported'] else 'FLAG'} "
          f"(stored in baselines_verdict.json; a failed baseline check "
          f"invalidates the corresponding claims above it, HB6 only flags "
          f"depth-based motivations)")


# E1 calibration-term ablation arms: the full factorisation of the two
# additive terms inside the identical ``calib`` gate.  The full cell (both
# terms on) is the ``calib`` condition already present in the GATES grid, so
# these three arms complete the 2x2 "length term x concentration term"
# design.  All arms share the frozen recipe and the evaluation streams with
# the main grid - only the two gate-term flags differ.
_E1_CALIB_ABLATION = (
    ("calib_len_only", {"calib_use_len": True, "calib_use_ent": False}),
    ("calib_ent_only", {"calib_use_len": False, "calib_use_ent": True}),
    ("calib_no_terms", {"calib_use_len": False, "calib_use_ent": False}),
)


def _e1_run(settings: Settings, seed: int, lr: float, gate: str,
            record_extra: dict, **model_overrides):
    """Train + score one E1 condition and build its full run record.

    Every condition goes through the identical entry point (frozen recipe,
    fixed test/stats streams) so the only difference between conditions is
    ``model_overrides``; ``record_extra`` carries the ablation label that
    the controlled-variable audit compares, and the record stamps the
    *resolved* ``model_cfg`` the model was actually built from so the audit
    can check the trained configuration instead of the label alone.
    Returns ``(record, errors)``.
    """
    model, params, metrics, curve, model_cfg = _train_and_eval(
        settings, seed, lr, gate=gate, **model_overrides)
    stats = collect_stats(model, settings.n_pairs, settings.vocab,
                          settings.device, batch=settings.stats_batch,
                          batches=settings.stats_batches, seed=STATS_SEED)
    corr = gate_stat_correlations(stats)
    fc = false_concentration_analysis(stats)
    fc_sweep = false_concentration_sweep(stats)
    ed = error_detection_aucs(stats)
    errors = collect_error_cases(
        model, settings.n_pairs, settings.vocab, settings.device,
        batch=settings.stats_batch, batches=settings.stats_batches,
        seed=STATS_SEED, max_cases=settings.max_error_cases)
    rec = {
        "experiment": "e1", "hypotheses": ["H1a", "H1b", "H1c", "H1d"],
        "gate": gate, "seed": seed, "lr": lr,
        "params": params, "model_cfg": model_cfg, **metrics, **corr, **fc, **ed,
        **{f"err_{k}": v for k, v in errors["summary"].items()},
        **_cost_record(settings, model), **curve, **record_extra,
        **_run_context(settings),
    }
    if _finite(rec.get("gate_vs_entropy")):
        rec["abs_gate_vs_entropy"] = abs(rec["gate_vs_entropy"])
    if _finite(fc.get("gate_on_correct")) and \
            _finite(fc.get("gate_on_false_conc")):
        rec["gate_margin"] = fc["gate_on_correct"] - fc["gate_on_false_conc"]
    rec["false_conc_thresholds"] = sorted(fc_sweep)
    for thr, row in fc_sweep.items():
        if _finite(row.get("gate_on_correct")) and \
                _finite(row.get("gate_on_false_conc")):
            rec[f"gate_margin_thr{thr:g}"] = (
                row["gate_on_correct"] - row["gate_on_false_conc"])
    rec["false_conc_sweep"] = fc_sweep
    if gate == "null":
        rec["null_mass_error_margin"] = _null_mass_margin(stats)
    return rec, errors


def run_e1(settings: Settings, out_dir: pathlib.Path) -> None:
    _announce(("H1a", "H1b", "H1c", "H1d"))
    _write_manifest(settings, out_dir, "e1")
    _enforce_reproducibility(settings, out_dir, "e1")
    _enforce_overfit_gate(settings, out_dir, "e1")
    lr = _resolve_lr(settings, out_dir, "e1")
    runs = []
    examples = {}
    for gate in GATES:
        for seed in settings.seeds:
            # The full calib cell is labelled too, so the audit can see that
            # the ablation contrasts vary exactly the calibration-term flag.
            extra = {"calib_variant": "full"} if gate == "calib" else {}
            rec, errors = _e1_run(settings, seed, lr, gate, extra)
            if gate not in examples or \
                    errors["summary"]["n_errors"] > examples[gate][0]:
                ranked = sorted(errors["cases"],
                                key=lambda c: -float(c["confidence"]))
                examples[gate] = (errors["summary"]["n_errors"], ranked[:3])
            for case in errors["cases"]:
                _write_result(out_dir / "e1_error_cases.jsonl",
                              {"experiment": "e1", "gate": gate, "seed": seed,
                               "lr": lr, **case})
            _write_result(out_dir / "e1_runs.jsonl", rec)
            runs.append(rec)
    ablation_runs = {}
    for label, overrides in _E1_CALIB_ABLATION:
        for seed in settings.seeds:
            rec, errors = _e1_run(settings, seed, lr, "calib",
                                  {"calib_variant": label}, **overrides)
            for case in errors["cases"]:
                _write_result(out_dir / "e1_error_cases.jsonl",
                              {"experiment": "e1", "gate": "calib",
                               "ablation": label, "seed": seed, "lr": lr,
                               **case})
            _write_result(out_dir / "e1_runs.jsonl", rec)
            ablation_runs.setdefault(label, []).append(rec)
    by_gate = {g: [r for r in runs if r["gate"] == g] for g in GATES}
    table = {g: _aggregate_runs(rs) for g, rs in by_gate.items()}
    _report_ablation_coverage(
        out_dir, "e1",
        runs + [r for rs in ablation_runs.values() for r in rs])

    print(f"\n[e1] test seed {TEST_SEED}, stats seed {STATS_SEED}, "
          f"lr={lr:g}, steps={settings.steps}")
    header = f"{'gate':>6} {'params':>8} {'accuracy':>16} {'f1_macro':>16} " \
             f"{'top5':>16} {'abs_r_ent':>16} {'pr_auc_gate':>16}"
    print(header)
    for gate in GATES:
        t = table[gate]
        def cell(key):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return f"{m['mean']:.4f}±{m['std']:.4f}"
        print(f"{gate:>6} {int(by_gate[gate][0]['params']):>8} "
              f"{cell('accuracy'):>16} {cell('f1_macro'):>16} "
              f"{cell('top5_accuracy'):>16} {cell('abs_gate_vs_entropy'):>16} "
              f"{cell('prauc_gate_to_error'):>16}")

    # Metric regime per gate: error detection is a rare-class problem once
    # the model converges, so the measured imbalance is reported next to
    # each score's ranking power.  PR-AUC is read against its chance level
    # (lift = PR-AUC / prevalence; a random ranker has lift 1.0) and the
    # operational reading (errors caught in the 10% most doubtful answers)
    # sits next to ROC-AUC, which stays optimistic under imbalance.
    print("\n[e1] error-detection regime (positive = wrong answer): lift = "
          "PR-AUC / prevalence (1.0 = random ranking); recall@10% = share of "
          "errors captured in the 10% lowest-gate answers")
    print(f"{'gate':>6} {'primary':>8} {'err_rate':>9} {'imbalance':>10} "
          f"{'regime':>10} {'prauc':>8} {'lift':>8} {'recall@10%':>11} "
          f"{'rocauc':>8}")
    for gate in GATES:
        t = table[gate]
        primary = str(by_gate[gate][0].get("primary_metric", ""))
        if primary.startswith("prauc_"):
            score = primary.split("_")[1]
        elif _finite(by_gate[gate][0].get("prauc_gate_to_error")):
            score = "gate"
        else:
            score = "maxprob"

        def ecell(key):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return f"{m['mean']:.4f}"

        err_rate = t.get("error_rate", {}).get("mean", float("nan"))
        ratio = t.get("imbalance_ratio", {}).get("mean", float("nan"))
        if not math.isfinite(err_rate) or err_rate <= 0.0:
            regime = "no_errors"
        elif ratio >= 1000.0:
            regime = "extreme"
        elif ratio >= 10.0:
            regime = "high"
        else:
            regime = "balanced"
        print(f"{gate:>6} {score:>8} {ecell('error_rate'):>9} "
              f"{ecell('imbalance_ratio'):>10} {regime:>10} "
              f"{ecell(f'prauc_{score}_to_error'):>8} "
              f"{ecell(f'prauc_lift_{score}_to_error'):>8} "
              f"{ecell(f'recall_at_10pct_{score}_to_error'):>11} "
              f"{ecell(f'rocauc_{score}_to_error'):>8}")

    abl_table = {label: _aggregate_runs(rs)
                 for label, rs in ablation_runs.items()}
    abl_table["full"] = table["calib"]
    abl_index = {"full": by_gate["calib"], **ablation_runs}
    print(f"\n[e1] calibration-term ablation (H1d): full 2x2 over the two "
          f"additive calib terms (length -a*log T, concentration -c*Hbar), "
          f"all inside the identical architecture with the frozen recipe and "
          f"the same seeds.  'abs_r_ent' must collapse when the concentration "
          f"term is removed (validating the ablation); 'full' is the "
          f"main-grid calib cell:")
    print(f"{'arm':>16} {'params':>8} {'accuracy':>17} {'abs_r_ent':>17} "
          f"{'gate_margin':>17}")
    for label in ("full", "calib_len_only", "calib_ent_only",
                  "calib_no_terms"):
        t = abl_table[label]

        def acell(key):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return f"{m['mean']:.4f}±{m['std']:.4f}"

        print(f"{label:>16} {int(abl_index[label][0]['params']):>8} "
              f"{acell('accuracy'):>17} {acell('abs_gate_vs_entropy'):>17} "
              f"{acell('gate_margin'):>17}")

    print(f"\n[e1] positive control: the parameter-free structural rule is "
          f"the task ceiling, so every learned accuracy must be read as a "
          f"gap to it (a gate far below the ceiling means the mechanism "
          f"statistics describe an underfit solution, not the mechanism):")
    ceiling = rule_pair_lookup_stats(
        settings.n_pairs, settings.vocab, settings.device,
        batch=settings.eval_batch, batches=settings.eval_batches,
        seed=TEST_SEED)
    print(f"  rule ceiling = {ceiling['accuracy']:.4f} "
          f"[{ceiling['accuracy_lo']:.4f}, {ceiling['accuracy_hi']:.4f}] "
          f"(Wilson 95%, n={ceiling['n']}, unique_key_rate="
          f"{ceiling['unique_key_rate']:.3f})")
    print(f"{'gate':>6} {'accuracy':>10} {'ceiling_gap':>12}")
    for gate in GATES:
        agg = table[gate].get("accuracy", {})
        acc = agg.get("mean", float("nan"))
        gap = (ceiling["accuracy"] - acc) if math.isfinite(acc) else float("nan")
        print(f"{gate:>6} {acc:>10.4f} {gap:>12.4f}")
    verdict = _load_baseline_verdict(out_dir)
    if verdict is None:
        print("[e1] note: no baselines_verdict.json in the output dir - run "
              "`--only baselines` first so the learned models can be read "
              "against the tuned simple baselines (probe, rules, shallow "
              "transformer) before crediting any gate variant")
    else:
        probe = verdict.get("probe_accuracy_mean", float("nan"))

        def _mark(key):
            value = verdict.get(key)
            return "n/a" if value is None else ("ok" if value else "FAILED")

        marks = "/".join(_mark(k) for k in ("hb1_supported", "hb2_supported",
                                            "hb3_supported", "hb4_supported",
                                            "hb5_supported", "hb6_supported"))
        print(f"[e1] simple-baseline check (from --only baselines): "
              f"HB1/HB2/HB3/HB4/HB5/HB6={marks}, tuned linear probe test "
              f"accuracy={probe:.4f} vs chance "
              f"{verdict.get('chance', float('nan')):.4f} - the transformer is "
              f"only credited for what it does beyond this tuned baseline and "
              f"the rule ceiling (HB4 = no label leakage, HB5 = loop can "
              f"memorise a tiny batch, HB6 = depth is measurably required)")
        shallow = verdict.get("shallow_none") or {}
        shallow_by_seed = {}
        for key, value in (shallow.get("accuracy_by_seed") or {}).items():
            try:
                shallow_by_seed[int(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if shallow_by_seed:
            print(f"[e1] complexity control (HB6): tuned 1-layer standard "
                  f"transformer (lr={shallow.get('lr', float('nan')):g}) test "
                  f"accuracy={shallow.get('accuracy_mean', float('nan')):.4f}, "
                  f"rule ceiling={shallow.get('ceiling', float('nan')):.4f}, "
                  f"gap={shallow.get('ceiling_gap', float('nan')):.4f}")
            # Paired complexity reference: same seeds, same test stream,
            # each side optimised with its own val-tuned learning rate
            # (that is the point of a tuned-baseline comparison), so the
            # strict controlled-recipe audit does not apply and this line
            # is deliberately kept outside the FDR family.
            diffs = [r["accuracy"] - shallow_by_seed[r["seed"]]
                     for r in by_gate["none"] if r["seed"] in shallow_by_seed]
            if len(diffs) >= 2:
                ci = paired_bootstrap_ci(diffs)
                print(f"      paired deep(none, 8 layers) - shallow(1 layer) "
                      f"accuracy: {_fmt_ci(ci)} (each side val-tuned; a "
                      f"supported gap means the depth of the E1/E3 grids is "
                      f"measurably required, a gap including 0 means a gate "
                      f"gain must be beyond what one layer already reaches)")

    print("\n[e1] H1a robustness: rank (Spearman) gate-entropy correlation "
          "alongside Pearson (a monotone effect must survive the change of "
          "association measure):")
    print(f"{'gate':>6} {'r_entropy':>10} {'rho_entropy':>12} "
          f"{'r_maxprob':>10} {'rho_maxprob':>12}")
    for gate in GATES:
        t = table[gate]
        def rcell(key):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return float("nan")
            return m["mean"]
        print(f"{gate:>6} {rcell('gate_vs_entropy'):>10.4f} "
              f"{rcell('gate_vs_entropy_spearman'):>12.4f} "
              f"{rcell('gate_vs_maxprob'):>10.4f} "
              f"{rcell('gate_vs_maxprob_spearman'):>12.4f}")

    print("\n[e1] H1b robustness: gate margin (correct - false-conc) across "
          "the high-maxprob threshold sweep (pre-registered decision point "
          "is thr=0.5; a sign flip at other cutoffs means the margin is a "
          "property of the cutoff, not the mechanism):")
    thresholds = sorted({float(t) for r in runs
                         for t in r.get("false_conc_thresholds", ())})
    print(f"{'gate':>6}" + "".join(f"{f'thr={t:g}':>12}" for t in thresholds))
    for gate in GATES:
        cells = []
        for thr in thresholds:
            agg = table[gate].get(f"gate_margin_thr{thr:g}", {})
            m = agg.get("mean", float("nan"))
            cells.append("n/a" if not math.isfinite(m) else f"{m:+.4f}")
        print(f"{gate:>6}" + "".join(f"{c:>12}" for c in cells))

    print("\n[e1] pre-registered contrasts (calib vs query/none):")
    clog = ContrastLog("e1", hypotheses=("H1a", "H1b", "H1c", "H1d"))
    clog.print("calib - none", "H1a", "accuracy",
               _paired_diff(by_gate["calib"], by_gate["none"], "accuracy"),
               warn=_curve_warn(("calib", by_gate["calib"]),
                                ("none", by_gate["none"])))
    # The query gate is the closest tuned simpler model (same parameter
    # count, same architecture, gate on the query only): calibration must
    # earn its keep against *this* baseline, not only against `none`
    # (which lacks a gate altogether).  Pre-registered so it is read
    # through the same FDR + SESOI decision layer as the other contrasts.
    clog.print("calib - query", "H1a", "accuracy",
               _paired_diff(by_gate["calib"], by_gate["query"], "accuracy"),
               warn=_curve_warn(("calib", by_gate["calib"]),
                                ("query", by_gate["query"])))
    clog.print("calib - query", "H1a", "abs_gate_vs_entropy",
               _paired_diff(by_gate["calib"], by_gate["query"],
                            "abs_gate_vs_entropy"),
               warn=_curve_warn(("calib", by_gate["calib"]),
                                ("query", by_gate["query"])))
    clog.print("calib - query", "H1b", "gate_margin",
               _paired_diff(by_gate["calib"], by_gate["query"],
                            "gate_margin"),
               warn=_curve_warn(("calib", by_gate["calib"]),
                                ("query", by_gate["query"])))
    null_margins = [r["null_mass_error_margin"] for r in by_gate["null"]
                    if _finite(r.get("null_mass_error_margin"))]
    clog.print("null margin vs 0", "H1c", "null_mass_error_margin",
               paired_bootstrap_ci(null_margins),
               warn=_curve_warn(("null", by_gate["null"])))
    print(f"\n[e1] H1d calibration-term ablation contrasts (removing a term "
          f"from the full gate; only the calib_variant factor varies, the "
          f"recipe stays frozen - every contrast carries its "
          f"controlled-variable audit):")
    for label in ("calib_no_terms", "calib_len_only", "calib_ent_only"):
        rs = ablation_runs[label]
        warn = _curve_warn(("calib", by_gate["calib"]), (label, rs))
        clog.print(f"calib - {label}", "H1d", "accuracy",
                   _paired_diff(by_gate["calib"], rs, "accuracy"), warn=warn)
        clog.print(f"calib - {label}", "H1d", "gate_margin",
                   _paired_diff(by_gate["calib"], rs, "gate_margin"),
                   warn=warn)
    for label in ("calib_len_only", "calib_no_terms"):
        rs = ablation_runs[label]
        clog.print(f"calib - {label}", "H1d", "abs_gate_vs_entropy",
                   _paired_diff(by_gate["calib"], rs, "abs_gate_vs_entropy"),
                   warn=_curve_warn(("calib", by_gate["calib"]), (label, rs)))
    clog.finalize(out_dir, q=settings.fdr_q)

    print(f"\n[e1] training curves (val monitored on fixed stream seed "
          f"{VAL_SEED}; verdict -> recommendation; gradmax = max pre-clip "
          f"gradient norm; vreg = val loss regression past its best point; "
          f"invalid = plateau/diverging/instability runs, whose claims cannot "
          f"be read as evidence; late = U-shaped val curves where the global "
          f"slope hides the regression):")
    for gate in GATES:
        verdicts, invalid = _curve_health(by_gate[gate])
        late = sum(1 for r in by_gate[gate] if r.get("late_overfit"))
        gap = _mean_std([r["generalization_gap"] for r in by_gate[gate]])
        gmax = _mean_std([r["grad_norm_max"] for r in by_gate[gate]])
        vreg = _mean_std([r["val_regression_after_best"] for r in by_gate[gate]])
        first = by_gate[gate][0]
        gmax_s = f"{gmax['mean']:.2f}" if math.isfinite(gmax["mean"]) else "n/a"
        vreg_s = f"{vreg['mean']:+.4f}" if math.isfinite(vreg["mean"]) else "n/a"
        print(f"{gate:>6} train {first['train_loss_first']:.3f}->"
              f"{first['train_loss_last']:.3f}  val "
              f"{first['val_loss_first']:.3f}->{first['val_loss_last']:.3f}  "
              f"gap={gap['mean']:.4f}  gradmax={gmax_s}  vreg={vreg_s}  "
              f"invalid={invalid}/{len(by_gate[gate])}  late={late}  "
              f"verdicts={verdicts}")
        for label, count in verdicts.items():
            rec = next(r["curve_recommendation"] for r in by_gate[gate]
                       if r["curve_verdict"] == label)
            print(f"       [{label} x{count}] {rec}")

    print(f"\n[e1] inference cost (batch="
          f"{min(settings.eval_batch, 128)}, seq_len={by_gate['none'][0]['seq_len']};"
          f" latency = median of {settings.cost_runs} runs; cpu memory is a"
          f" profiler alloc-sum proxy, not peak RSS):")
    print(f"{'gate':>6} {'params':>10} {'flops/M':>9} {'ms/sample':>10} "
          f"{'mem/MB':>9}")
    for gate in GATES:
        t = table[gate]

        def cell(key, fmt="{:.3f}"):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return fmt.format(m["mean"])

        flops = t.get("flops_per_forward", {}).get("mean", float("nan")) / 1e6
        flops_s = f"{flops:.2f}" if math.isfinite(flops) else "n/a"
        print(f"{gate:>6} {cell('params_total', '{:.0f}'):>10} "
              f"{flops_s:>9} "
              f"{cell('latency_ms_per_sample'):>10} "
              f"{cell('peak_mem_mb', '{:.2f}'):>9}")

    def _ratio(a, b, key):
        ma = table[a].get(key, {}).get("mean")
        mb = table[b].get(key, {}).get("mean")
        if not ma or not mb or not math.isfinite(ma) or not math.isfinite(mb) \
                or mb == 0:
            return float("nan")
        return ma / mb

    print(f"[e1] accuracy delta vs relative cost, each gate vs none "
          f"(exploratory and uncorrected - the pre-registered family above "
          f"is FDR-controlled; a gain is only read as supported when the CI "
          f"excludes 0, and a supported gain still has to be read against "
          f"the multipliers):")
    for gate in GATES:
        if gate == "none":
            continue
        acc_ci = _paired_diff(by_gate[gate], by_gate["none"], "accuracy")
        warn = _curve_warn((gate, by_gate[gate]), ("none", by_gate["none"]))
        print(f"  {gate:>6}: {_fmt_ci(acc_ci)} | "
              f"params x{_ratio(gate, 'none', 'params_total'):.3f}"
              f"  flops x{_ratio(gate, 'none', 'flops_per_forward'):.3f}"
              f"  latency x{_ratio(gate, 'none', 'latency_ms_per_sample'):.3f}"
              f"  mem x{_ratio(gate, 'none', 'peak_mem_mb'):.3f}"
              + (f"  [{warn}]" if warn else ""))

    print(f"\n[e1] bad cases (stats seed {STATS_SEED}; train batch: "
          f"{settings.stats_batch}x{settings.stats_batches}; raw cases in"
          f" e1_error_cases.jsonl):")
    print(f"{'gate':>6} {'err_rate':>9} {'high_conf':>10} {'key_miss':>9} "
          f"{'in_ctx':>7} {'naive':>6} {'mass_key_err':>13} {'mass_key_ok':>12}")
    for gate in GATES:
        t = table[gate]
        def ecell(key, fmt="{:.3f}"):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return fmt.format(m["mean"])
        print(f"{gate:>6} {ecell('err_error_rate'):>9} "
              f"{ecell('err_share_high_conf'):>10} "
              f"{ecell('err_errors_key_missed_share'):>9} "
              f"{ecell('err_share_pred_in_context'):>7} "
              f"{ecell('err_share_naive_confusable'):>6} "
              f"{ecell('err_attn_mass_on_key_error'):>13} "
              f"{ecell('err_attn_mass_on_key_correct'):>12}")

    print(f"\n[e1] error taxonomy (share of errors; priority: label error > "
          f"structural ambiguity (naive_confusable/value_repeats) > retrieval "
          f"miss (attention mass < 0.5 on the correct key) > wrong readout; "
          f"label_noise_rate is *measured* on these batches):")
    print(f"{'gate':>6} {'label':>8} {'struct':>8} {'retrieval':>10} "
          f"{'readout':>8} {'label_noise':>12}")
    for gate in GATES:
        t = table[gate]
        def tcell(key):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return f"{m['mean']:.3f}"
        print(f"{gate:>6} {tcell('err_taxonomy_label_error'):>8} "
              f"{tcell('err_taxonomy_hard_structural'):>8} "
              f"{tcell('err_taxonomy_retrieval_miss'):>10} "
              f"{tcell('err_taxonomy_readout_wrong'):>8} "
              f"{tcell('err_label_noise_rate'):>12}")
    print("  most confident wrong samples per gate (systematic blind spots "
          "are the most informative):")
    for gate in GATES:
        for case in examples.get(gate, (0, []))[1]:
            print(f"  [{gate}] {_format_case(case)}")
    _write_json(out_dir / "e1_summary.json", table)


def _null_mass_margin(stats):
    nm = stats.get("null_mass")
    if nm is None:
        return float("nan")
    corr = stats["correct"].bool()
    if bool(corr.all()) or bool((~corr).all()):
        return float("nan")
    mass = nm.mean(dim=1)
    return float(mass[~corr].mean() - mass[corr].mean())


# E2 length-generalisation arms: the four gate variants plus the complete
# 2x2 over the calib gate's two additive terms (the full cell is the
# ``calib`` arm itself), so the arm that isolates any H2a/H2b effect (the
# length term, H2c) is measured on the same frozen recipe and the same
# length sweep inside a full-factorial design - no term combination is left
# unmeasured.  The calib arms carry ``calib_variant`` labels next to
# ``gate="calib"``, so the controlled-variable audit attributes a contrast
# to the term flag rather than to the gate.
_E2_VARIANTS = (
    ("none", "none", {}, {}),
    ("query", "query", {}, {}),
    ("calib", "calib", {"calib_variant": "full"}, {}),
    ("calib_len_only", "calib", {"calib_variant": "calib_len_only"},
     {"calib_use_len": True, "calib_use_ent": False}),
    ("calib_ent_only", "calib", {"calib_variant": "calib_ent_only"},
     {"calib_use_len": False, "calib_use_ent": True}),
    ("calib_no_terms", "calib", {"calib_variant": "calib_no_terms"},
     {"calib_use_len": False, "calib_use_ent": False}),
    ("null", "null", {}, {}),
)


def run_e2(settings: Settings, out_dir: pathlib.Path) -> None:
    _announce(("H2a", "H2b", "H2c"))
    _write_manifest(settings, out_dir, "e2")
    _enforce_reproducibility(settings, out_dir, "e2")
    _enforce_overfit_gate(settings, out_dir, "e2")
    lr = _resolve_lr(settings, out_dir, "e2")
    lengths = length_sweep_lengths(base=settings.n_pairs, factor=2, n=4,
                                   max_len=settings.max_len)
    print(f"[e2] training at n_pairs={settings.n_pairs}, sweep over {lengths} "
          f"(test seed {TEST_SEED}, stats seed {STATS_SEED})")
    runs = []
    examples = {}
    for label, gate, record_extra, model_overrides in _E2_VARIANTS:
        for seed in settings.seeds:
            model, params, base_metrics, curve, model_cfg = _train_and_eval(
                settings, seed, lr, gate=gate, **model_overrides)
            rec = {"experiment": "e2", "hypotheses": ["H2a", "H2b", "H2c"],
                   "gate": gate, "variant": label, "seed": seed, "lr": lr,
                   "params": params, "model_cfg": model_cfg,
                   "accuracy_by_length": {},
                   "entropy_by_length": {}, "entropy_norm_by_length": {},
                   "maxprob_by_length": {}, "gate_by_length": {},
                   **record_extra,
                   **_cost_record(settings, model), **curve,
                   **_run_context(settings)}
            for length in lengths:
                metrics = (base_metrics if length == settings.n_pairs else
                           eval_metrics(model, length, settings.vocab,
                                        settings.device,
                                        batch=settings.eval_batch,
                                        batches=settings.eval_batches,
                                        seed=TEST_SEED))
                stats = collect_stats(model, length, settings.vocab,
                                      settings.device, batch=settings.stats_batch,
                                      batches=settings.stats_batches,
                                      seed=STATS_SEED)
                rec["accuracy_by_length"][length] = metrics["accuracy"]
                rec["entropy_by_length"][length] = float(stats["entropy"].mean())
                rec["entropy_norm_by_length"][length] = (
                    rec["entropy_by_length"][length]
                    / math.log(max(2 * length + 2, 2)))
                rec["maxprob_by_length"][length] = float(stats["maxprob"].mean())
                if stats.get("gate") is not None:
                    rec["gate_by_length"][length] = float(stats["gate"].mean())
            base_acc = rec["accuracy_by_length"][settings.n_pairs]
            long_acc = rec["accuracy_by_length"][lengths[-1]]
            rec["accuracy_at_base"] = base_acc
            rec["accuracy_at_longest"] = long_acc
            rec["accuracy_drop"] = base_acc - long_acc
            rec["entropy_drift"] = (rec["entropy_by_length"][lengths[-1]]
                                    - rec["entropy_by_length"][settings.n_pairs])
            rec["entropy_norm_drift"] = (
                rec["entropy_norm_by_length"][lengths[-1]]
                - rec["entropy_norm_by_length"][settings.n_pairs])
            errors = collect_error_cases(
                model, lengths[-1], settings.vocab, settings.device,
                batch=settings.stats_batch, batches=settings.stats_batches,
                seed=STATS_SEED, max_cases=settings.max_error_cases)
            rec.update({f"err_{k}": v for k, v in errors["summary"].items()})
            for case in errors["cases"]:
                _write_result(out_dir / "e2_error_cases.jsonl",
                              {"experiment": "e2", "gate": gate,
                               "variant": label, "seed": seed,
                               "length": lengths[-1], "lr": lr, **case})
            if label not in examples or \
                    errors["summary"]["n_errors"] > examples[label][0]:
                ranked = sorted(errors["cases"],
                                key=lambda c: -float(c["confidence"]))
                examples[label] = (errors["summary"]["n_errors"], ranked[:3])
            _write_result(out_dir / "e2_runs.jsonl", rec)
            runs.append(rec)

    by_variant = {label: [r for r in runs if r["variant"] == label]
                  for label, *_ in _E2_VARIANTS}
    table = {label: _aggregate_runs(rs) for label, rs in by_variant.items()}
    _report_ablation_coverage(out_dir, "e2", runs)
    print(f"\n[e2] accuracy by n_pairs (mean +- std over seeds; the "
          f"calib_len_only / calib_ent_only / calib_no_terms arms are the "
          f"H2c calibration-term ablations, completing the 2x2 with calib):")
    line = f"{'arm':>16}" + "".join(f"{L:>12}" for L in lengths) + \
           f"{'drop':>10} {'entropy_drift':>14}"
    print(line)
    for label, *_ in _E2_VARIANTS:
        cells = []
        for length in lengths:
            agg = _mean_std([r["accuracy_by_length"][length]
                             for r in by_variant[label]])
            cells.append(f"{agg['mean']:.3f}±{agg['std']:.3f}")
        drop = _mean_std([r["accuracy_drop"] for r in by_variant[label]])
        drift = _mean_std([r["entropy_drift"] for r in by_variant[label]])
        print(f"{label:>16}" + "".join(f"{c:>12}" for c in cells)
              + f"{drop['mean']:>10.3f} {drift['mean']:>14.3f}")

    print(f"\n[e2] inference cost at the training length (same protocol as "
          f"e1; check whether the length benefit costs latency; curve_ok = "
          f"curve-valid runs / all runs):")
    print(f"{'arm':>16} {'params':>10} {'flops/M':>9} {'ms/sample':>10} "
          f"{'mem/MB':>9} {'curve_ok':>8} {'curve':>10}")
    for label, *_ in _E2_VARIANTS:
        t = table[label]

        def cell(key, fmt="{:.3f}"):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return fmt.format(m["mean"])

        flops = t.get("flops_per_forward", {}).get("mean", float("nan")) / 1e6
        flops_s = f"{flops:.2f}" if math.isfinite(flops) else "n/a"
        _, invalid = _curve_health(by_variant[label])
        verdicts = sorted({r["curve_verdict"] for r in by_variant[label]})
        print(f"{label:>16} {cell('params_total', '{:.0f}'):>10} "
              f"{flops_s:>9} "
              f"{cell('latency_ms_per_sample'):>10} "
              f"{cell('peak_mem_mb', '{:.2f}'):>9} "
              f"{invalid}/{len(by_variant[label])} "
              f"{'/'.join(verdicts):>10}")

    print(f"\n[e2] bad cases at the longest length (n_pairs={lengths[-1]}; "
          f"stats seed {STATS_SEED}, batch {settings.stats_batch}x"
          f"{settings.stats_batches}; raw cases in e2_error_cases.jsonl).  "
          f"retrieval = attention mass < 0.5 on the correct key (the model "
          f"never found it); readout = key found but the wrong value read "
          f"out - the split that tells apart 'cannot find' from 'cannot "
          f"use':")
    print(f"{'arm':>16} {'err_rate':>9} {'label':>7} {'struct':>7} "
          f"{'retrieval':>10} {'readout':>8} {'key_miss':>9} {'mass_key_err':>13}")
    for label, *_ in _E2_VARIANTS:
        t = table[label]

        def ecell(key, fmt="{:.3f}"):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return fmt.format(m["mean"])

        print(f"{label:>16} {ecell('err_error_rate'):>9} "
              f"{ecell('err_taxonomy_label_error'):>7} "
              f"{ecell('err_taxonomy_hard_structural'):>7} "
              f"{ecell('err_taxonomy_retrieval_miss'):>10} "
              f"{ecell('err_taxonomy_readout_wrong'):>8} "
              f"{ecell('err_errors_key_missed_share'):>9} "
              f"{ecell('err_attn_mass_on_key_error'):>13}")
    print("  most confident wrong samples per arm at the longest length:")
    for label, *_ in _E2_VARIANTS:
        for case in examples.get(label, (0, []))[1]:
            print(f"  [{label}] {_format_case(case)}")

    print("\n[e2] pre-registered contrasts:")
    clog = ContrastLog("e2", hypotheses=("H2a", "H2b", "H2c"))
    calib_runs = by_variant["calib"]
    calib_warn = _curve_warn(("calib", calib_runs))
    clog.print("calib - none", "H2a", "accuracy_at_longest",
               _paired_diff(by_variant["calib"], by_variant["none"],
                            "accuracy_at_longest"),
               warn=_curve_warn(("calib", by_variant["calib"]),
                                ("none", by_variant["none"])))
    clog.print("calib - none", "H2a", "accuracy_drop",
               _paired_diff(by_variant["calib"], by_variant["none"],
                            "accuracy_drop"),
               warn=_curve_warn(("calib", by_variant["calib"]),
                                ("none", by_variant["none"])))
    clog.print("calib raw entropy drift vs 0", "H2b", "entropy_drift",
               paired_bootstrap_ci([r["entropy_drift"] for r in calib_runs]),
               warn=calib_warn)
    # The H/log(T) drift and the gate drift are *null predictions*: the
    # decision rule is CI-within-bound (bootstrap TOST), not "CI excludes
    # 0" - support means "indistinguishable from length-stable within the
    # pre-registered bound", and a large drift must refute, not support,
    # the prediction.
    clog.print_equivalence("calib normalised entropy drift vs 0", "H2b",
                           "entropy_norm_drift",
                           [r["entropy_norm_drift"] for r in calib_runs],
                           warn=calib_warn)
    if calib_runs[0]["gate_by_length"]:
        gate_drifts = [r["gate_by_length"][lengths[-1]]
                       - r["gate_by_length"][settings.n_pairs]
                       for r in calib_runs]
        clog.print_equivalence("calib gate drift vs 0", "H2b", "gate_drift",
                               gate_drifts, warn=calib_warn)
    print(f"\n[e2] H2c length-term ablation contrasts (each ablation arm "
          f"removes additive terms from the full gate at identical recipe; "
          f"only the calibration-term flags vary, so the full 2x2 is "
          f"measured across the length sweep):")
    for arm in ("calib_ent_only", "calib_len_only", "calib_no_terms"):
        rs = by_variant[arm]
        warn = _curve_warn(("calib", calib_runs), (arm, rs))
        clog.print(f"calib - {arm}", "H2c", "accuracy_at_longest",
                   _paired_diff(calib_runs, rs, "accuracy_at_longest"),
                   warn=warn)
    for arm in ("calib_ent_only", "calib_no_terms"):
        clog.print(f"calib - {arm}", "H2c", "accuracy_drop",
                   _paired_diff(calib_runs, by_variant[arm], "accuracy_drop"),
                   warn=_curve_warn(("calib", calib_runs),
                                    (arm, by_variant[arm])))
    clog.finalize(out_dir, q=settings.fdr_q)
    _write_json(out_dir / "e2_summary.json",
                {label: {str(L): _mean_std([r["accuracy_by_length"][L]
                                            for r in by_variant[label]])
                         for L in lengths} for label, *_ in _E2_VARIANTS})


def run_e3(settings: Settings, out_dir: pathlib.Path) -> None:
    _announce(("H3a", "H3b"))
    _write_manifest(settings, out_dir, "e3")
    _enforce_reproducibility(settings, out_dir, "e3")
    _enforce_overfit_gate(settings, out_dir, "e3")
    lr = _resolve_lr(settings, out_dir, "e3")
    print(f"[e3] grid gate x norm x layer_scale at lr={lr:g}, "
          f"test seed {TEST_SEED}, stats seed {STATS_SEED}")
    runs = []
    for gate in ("query", "calib"):
        for norm in NORMS:
            for layer_scale in (None, 1e-2):
                for seed in settings.seeds:
                    model, params, metrics, curve, model_cfg = _train_and_eval(
                        settings, seed, lr, gate=gate, norm=norm,
                        layer_scale=layer_scale)
                    stats = collect_stats(model, settings.n_pairs,
                                          settings.vocab, settings.device,
                                          batch=settings.stats_batch,
                                          batches=settings.stats_batches,
                                          seed=STATS_SEED)
                    resid = stats["resid_norm"].mean(dim=0)
                    rec = {
                        "experiment": "e3", "hypotheses": ["H3a", "H3b"],
                        "gate": gate, "norm": norm,
                        "layer_scale": layer_scale, "seed": seed, "lr": lr,
                        "params": params, "model_cfg": model_cfg,
                        "accuracy": metrics["accuracy"],
                        "f1_macro": metrics["f1_macro"],
                        "gate_depth_ratio": gate_depth_ratio(stats),
                        "resid_norm_by_layer": resid.tolist(),
                        "resid_norm_last_over_first": float(
                            resid[-1] / resid[0].clamp_min(1e-12)),
                        **_cost_record(settings, model), **curve,
                        **_run_context(settings),
                    }
                    _write_result(out_dir / "e3_runs.jsonl", rec)
                    runs.append(rec)

    print(f"\n[e3] {'gate':>5} {'norm':>9} {'ls':>5} {'accuracy':>16} "
          f"{'gate_depth_ratio':>18} {'resid_growth':>14} "
          f"{'params':>9} {'ms/sample':>10} {'mem/MB':>8} {'curve_ok':>9} {'curve':>10}")
    index = {}
    for gate in ("query", "calib"):
        for norm in NORMS:
            for layer_scale in (None, 1e-2):
                rs = [r for r in runs if r["gate"] == gate
                      and r["norm"] == norm
                      and r["layer_scale"] == layer_scale]
                index[(gate, norm, layer_scale)] = rs
                acc = _mean_std([r["accuracy"] for r in rs])
                ratio = _mean_std([r["gate_depth_ratio"] for r in rs])
                growth = _mean_std([r["resid_norm_last_over_first"] for r in rs])
                params = _mean_std([r["params_total"] for r in rs])
                lat = _mean_std([r["latency_ms_per_sample"] for r in rs])
                mem = _mean_std([r["peak_mem_mb"] for r in rs])
                verdicts = sorted({r["curve_verdict"] for r in rs})
                _, invalid = _curve_health(rs)
                ls_label = "off" if layer_scale is None else f"{layer_scale:g}"
                print(f"{gate:>5} {norm:>9} {ls_label:>5} "
                      f"{acc['mean']:>8.4f}±{acc['std']:<7.4f} "
                      f"{ratio['mean']:>9.4f}±{ratio['std']:<8.4f} "
                      f"{growth['mean']:>7.3f}±{growth['std']:<6.3f} "
                      f"{params['mean']:>9.0f} {lat['mean']:>10.3f} "
                      f"{mem['mean']:>8.2f} {invalid}/{len(rs)} "
                      f"{'/'.join(verdicts):>10}")

    print("\n[e3] seednorm vs pre cost (exploratory and uncorrected: the "
          "pre-registered family below is FDR-controlled; accuracy delta vs "
          "relative cost):")
    for gate in ("query", "calib"):
        acc_ci = _paired_diff(index[(gate, "seednorm", None)],
                              index[(gate, "pre", None)], "accuracy")
        a = _aggregate_runs(index[(gate, "seednorm", None)])
        b = _aggregate_runs(index[(gate, "pre", None)])
        ratios = []
        for key, label in (("params_total", "params"),
                           ("flops_per_forward", "flops"),
                           ("latency_ms_per_sample", "latency")):
            ma = a.get(key, {}).get("mean")
            mb = b.get(key, {}).get("mean")
            r = (ma / mb) if ma and mb else float("nan")
            ratios.append(f"{label} x{r:.3f}" if math.isfinite(r)
                          else f"{label} n/a")
        warn = _curve_warn(
            (f"{gate}/seednorm", index[(gate, "seednorm", None)]),
            (f"{gate}/pre", index[(gate, "pre", None)]))
        print(f"  [{gate}] {_fmt_ci(acc_ci)} | {'  '.join(ratios)}"
              + (f"  [{warn}]" if warn else ""))

    print("\n[e3] pre-registered contrasts (H3b: "
          "resid_norm_last_over_first is the primary readout; "
          "gate_depth_ratio is secondary):")
    clog = ContrastLog("e3", hypotheses=("H3a", "H3b"))
    for gate in ("query", "calib"):
        for layer_scale in (None, 1e-2):
            a = index[(gate, "seednorm", layer_scale)]
            b = index[(gate, "pre", layer_scale)]
            clog.print(f"{gate}: seednorm - pre (ls={layer_scale})",
                       "H3a", "gate_depth_ratio",
                       _paired_diff(a, b, "gate_depth_ratio"),
                       warn=_curve_warn((f"{gate}/seednorm", a),
                                        (f"{gate}/pre", b)))
    for gate in ("query", "calib"):
        for norm in NORMS:
            a = index[(gate, norm, 1e-2)]
            b = index[(gate, norm, None)]
            clog.print(f"{gate}/{norm}: ls=1e-2 - off", "H3b",
                       "resid_norm_last_over_first",
                       _paired_diff(a, b, "resid_norm_last_over_first"),
                       warn=_curve_warn((f"{gate}/{norm}/ls", a),
                                        (f"{gate}/{norm}/off", b)))
    for norm in NORMS:
        a = index[("calib", norm, None)]
        b = index[("query", norm, None)]
        clog.print(f"{norm}: calib - query", "H3a", "accuracy",
                   _paired_diff(a, b, "accuracy"),
                   warn=_curve_warn((f"calib/{norm}", a),
                                    (f"query/{norm}", b)))
    clog.finalize(out_dir, q=settings.fdr_q)
    _write_json(out_dir / "e3_summary.json",
                {f"{g}|{n}|{ls}": _aggregate_runs(rs)
                 for (g, n, ls), rs in index.items()})


def run_sanity(settings: Settings, out_dir: pathlib.Path) -> None:
    """Standalone pre-flight: seed-determinism audit + HB5 overfit gate.

    The E1/E2/E3 runners invoke the identical gates themselves, so a clean
    E1/E2/E3 run implies a passed pre-flight (``--only sanity``).
    """
    _announce(("HB5",))
    _write_manifest(settings, out_dir, "sanity")
    _enforce_reproducibility(settings, out_dir, "sanity")
    _enforce_overfit_gate(settings, out_dir, "sanity")


def run_all(settings: Settings, out_dir: pathlib.Path) -> None:
    run_baselines(settings, out_dir)
    run_e1(settings, out_dir)
    run_e2(settings, out_dir)
    run_e3(settings, out_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Hypothesis-driven runners for E1/E2/E3 "
                    "(see cage/experiments.py for the registry).")
    parser.add_argument("--only", default="baselines",
                        choices=["baselines", "sanity", "e1", "e2", "e3", "all"])
    parser.add_argument("--quick", action="store_true",
                        help="tiny pipeline-check configuration")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, default=None,
                        help="number of training seeds (0..n-1)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None,
                        help="skip recipe tuning and use this lr directly")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--fdr-q", type=float, default=0.05,
                        help="Benjamini-Hochberg FDR level for the "
                             "pre-registered contrast families")
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)

    settings = Settings.quick(args.device) if args.quick else Settings(
        device=args.device)
    if args.seeds is not None:
        requested = tuple(range(args.seeds))
        dropped = tuple(s for s in requested if s in RESERVED_EVAL_SEEDS)
        settings.seeds = tuple(s for s in requested
                               if s not in RESERVED_EVAL_SEEDS)
        if dropped:
            print(f"[runner] WARNING: dropping training seeds {dropped}: "
                  f"they are reserved held-out evaluation streams "
                  f"{RESERVED_EVAL_SEEDS}; training on them would fit the "
                  f"model on validation/test data")
        if not settings.seeds:
            print("[runner] no training seed left after excluding reserved "
                  "evaluation streams; aborting")
            return 2
    if args.steps is not None:
        settings.steps = args.steps
    if args.lr is not None:
        settings.lr = args.lr
    if args.d_model is not None:
        settings.d_model = args.d_model
    if args.n_layers is not None:
        settings.n_layers = args.n_layers
    settings.fdr_q = args.fdr_q
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "hypotheses.json", HYPOTHESES)
    _write_json(out_dir / "environment.json", _run_context(settings))

    print(f"[runner] only={args.only} quick={args.quick} "
          f"seeds={settings.seeds} steps={settings.steps} "
          f"lr={settings.lr if settings.lr is not None else 'tuned'} "
          f"fdr_q={settings.fdr_q:g} "
          f"device={settings.device} out={out_dir}")
    try:
        if args.only == "baselines":
            run_baselines(settings, out_dir)
        elif args.only == "sanity":
            run_sanity(settings, out_dir)
        elif args.only == "e1":
            run_e1(settings, out_dir)
        elif args.only == "e2":
            run_e2(settings, out_dir)
        elif args.only == "e3":
            run_e3(settings, out_dir)
        else:
            run_all(settings, out_dir)
    except (OverfitSanityError, ReproducibilityError) as e:
        print(f"[runner] ABORTED before grid training: {e}")
        return 3
    print(f"\n[runner] done; records in {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
