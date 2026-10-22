"""Hypothesis-driven experiment runners for E1/E2/E3.

Design principles
-----------------
1. **Hypotheses first.**  Every config is annotated with the hypothesis
   it tests (``HYPOTHESES``) and a pre-registered decision rule.  The
   runner prints the hypotheses before executing and stores their ids
   with every result record, so the E3 grid is not a blind
   hyperparameter sweep: each contrast answers a mechanism question.  E2
   additionally reads the trained calibration coefficients (and their
   implied length/concentration contributions at the base and longest
   lengths), and H2f turns the mechanism claim into a one-sample test that
   the learned length and concentration coefficients are positive - so the
   claim that calib *explicitly* corrects ``log T`` entropy growth is
   checked on the learned gate rather than inferred from endpoint accuracy
   alone.  H2g then ablates the normalisation itself: a raw-entropy arm
   with the same architecture and parameters sits in the same length
   sweep, so "H/log T is stable" (a property of the statistic) is not
   confused with "the model needs the normalisation" (a property of the
   trained mechanism).
2. **Tuned baselines and sanity checks before complex models.**  The
   complexity ladder is explicit and every rung is tuned: the rule
   system (exact ceiling), the trivial floor, a wide-grid-tuned linear
   probe, and a learning-rate-tuned **depth ladder of standard
   transformers** (HB6/HB7: 1/2/4 layers, each rung tuned on the
   validation seed) all run before any deep grid, so both the simplest
   learned model and the depth at which the task saturates are measured
   rather than assumed.  A shared training recipe is tuned on the plain ``none``
    baseline (validation seed), then frozen for all gate variants so the
    E1/E3 comparisons stay controlled (each side of every pre-registered
    contrast carries a machine-checked controlled-variable audit).  Because
    that frozen point was selected on ``none`` alone, E1 and the primary E2
    arms additionally run a **separate per-variant fairness accounting**:
    every arm gets its own validation-selected lr (and calib its own
    ``calib_tau_init``), and the resulting calib-vs-baseline paired
    benchmark is reported *next to* the controlled columns, never pooled
    with them.  E2 selects at the training length and reads the endpoint
    metrics at the longest sweep length, so selection never sees the
    extrapolation length.  The primary pre-registered conclusion is still
    the same-recipe family; the tuned account answers whether it survives
    when the baselines also get their best shot.  Every
    run record stamps the full optimisation recipe it trained under
    (optimizer family, weight decay, betas, schedule, eta_min, gradient
    clipping, augmentation - from the single source of truth in
    ``cage.train.training_recipe``, beside the already-audited learning
    rate and step count), so "the gain is just a different optimizer,
    schedule or evaluation stream" is refused by the audit instead of
    being reported.  The
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
       statistically detectable sliver has not earned its complexity.  Each
    directional prediction additionally pre-registers its *sign* per
    metric: an FDR-significant effect in the opposite direction is a
    refutation of the hypothesis and can never be read as support,
    whatever its magnitude.  This
   matters at the 5-seed scale used here: an uncorrected 95% readout over
   ~16 null contrasts would produce about one spurious "supported" claim
   per experiment.  At that same scale the percentile bootstrap is the
   first-order approximation in the chain, so every contrast additionally
   carries a jackknife (leave-one-seed-out) diagnostic and a bias-corrected
   accelerated (BCa) interval, and the report flags any positive verdict
   whose decision flips when a single seed is dropped or dissents from the
    BCa readout (see ``cage.metrics``).  The companions audit the
    pre-registered decision rule; they never replace it.  A contrast also
    needs at least ``min_seeds_for_claim`` independent paired seeds
    (default 3, pre-registered in ``Settings``) before any verdict may
    count as a claim; rows below the floor are marked INELIGIBLE, and the
    seeds that actually entered each paired difference are recorded
    (``seeds_used``/``seeds_dropped``) so a sample that silently shrank is
    visible.  Every factorial design (E1's four gate arms, E2's eight
    arms, E3's 2x4x2 grid) is additionally checked for cell coverage, so
    an incomplete ablation cannot silently leave an effect unmeasured.
    The E2 design includes a one-factor normalisation ablation
    (``calib_raw_ent``, H2g) next to the two-term 2x2: H2b shows that
    ``H/log T`` is length-stable, and H2g asks whether the trained gate
    actually needs that normalisation rather than merely whether the
    statistic is stable.

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
import sys
import time
import uuid
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
    actual_stream_overlap_audit,
    answer_uniformity_test,
    batch_fingerprint,
    check_data_geometry,
    check_eval_seed,
    check_train_seed,
    length_sweep_lengths,
    make_batch,
    split_leakage_audit,
    stream_fingerprint,
    stream_seed_key,
    validate_batch,
)
from .efficiency import measure_config
from .metrics import (
    _positive_integer,
    benjamini_hochberg,
    mcnemar_test,
    clustered_proportion_ci,
    equivalence_test,
    noninferiority_test,
    paired_bootstrap_ci,
    seeds_for_power,
    wilson_ci,
)
from .model import Cfg, GATES, NORMS, build
from .repro import configure_determinism, seed_everything, stream_generator
from .train import (
    _ERROR_TYPES,
    decode_correct_mask,
    collect_stats,
    collect_stats_and_errors,
    error_detection_aucs,
    eval_accuracy,
    eval_metrics,
    false_concentration_analysis,
    false_concentration_by_layer,
    false_concentration_gate_test,
    false_concentration_sweep,
    gate_depth_ratio,
    gate_stat_correlations,
    null_mass_gate_test,
    overfit_sanity_check,
    train_model,
    training_curve_verdict,
    training_recipe,
)

# Seeds with fixed roles (independent of every training seed) are defined in
# ``cage.data`` so that every entry point shares one source of truth:
#   VAL_SEED   - model selection only (learning-rate tuning, baseline checks)
#   TEST_SEED  - the final reported numbers *and* the pre-registered gate
#                rules fixed in this module (HB1 rule ceiling, HB4 permuted-
#                label control, HB0/HB6/HB7 competence/saturation marks);
#                never read by any model/hyperparameter/recipe choice
#   STATS_SEED - mechanism statistics (matches the collect_stats default)
# Seal = no *selection* reads these streams; the fixed rules above are
# re-measured on TEST when the CI loop re-runs `--only baselines`, which is
# not re-selection.  See the sealing-note paragraph in ``cage.data``.

HYPO_IDS = ("HB0", "HB1", "HB2", "HB3", "HB4", "HB5", "HB6", "HB7",
            "H1a", "H1b",
            "H1c", "H1d", "H1e", "H1f", "H1g", "H2a", "H2b", "H2c", "H2d",
            "H2e", "H2f", "H2g", "H3a", "H3b", "H3c")

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
    "HB0": {
        "experiment": "all",
        "question": "Is a mechanism claim readable at all - did the models "
                    "under test actually learn the task at this step count?",
        "hypothesis": "The depth/mechanism contrasts are only informative "
                      "about the mechanism when the trained arms are in a "
                      "competent regime; an underfit grid produces "
                      "'no difference' or 'depth needed' readings for "
                      "reasons unrelated to the mechanism.",
        "prediction": "at least one trained arm (a ladder rung in "
                      "baselines, or a grid arm in E1/E2/E3) has a "
                      "seed-clustered one-sided 95% family-wise lower bound "
                      "(Bonferroni over arms) above chance "
                      "1/(vocab - 2) on the sealed test stream",
        "decision": "the competence prerequisite is a precondition, not a "
                    "mechanism test: when no arm clears chance, every "
                    "mechanism contrast in that family is marked "
                    "claim-ineligible (no claim may be read from an "
                    "underfit grid) and the depth requirement (HB7) is "
                    "'not determinable' rather than 'required'.  The lower "
                    "bound resamples training seeds and keeps the shared "
                    "evaluation-stream measurement error, so re-scoring more "
                    "seeds on the same test stream cannot buy precision",
    },
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
                      "below 2x chance; the CI resamples seeds and adds one "
                      "shared Binomial(n_per_seed) scoring draw per "
                      "replication on the common evaluation stream (not one "
                      "draw per seed), so the bound is certified against "
                      "measurement noise without pseudo-replicating the "
                      "stream; the one-sided bootstrap p-value for H0: "
                      "margin >= +chance certifies this at adequate power",
        "decision": "The quantitative near-chance test "
                    "(cage.baselines.probe_vs_chance_test) must be "
                    "*certified* (CI upper bound < +chance); a CI that "
                    "merely straddles the bound is 'inconclusive' - not "
                    "certified, possibly underpowered - and the reported "
                    "seeds_for_certification (planned on seed spread plus "
                    "the non-shrinking common-stream noise floor) says "
                    "whether more seeds/the probe stream can settle it; only "
                    "a CI "
                    "entirely at or above +chance (state 'violated') is "
                    "evidence that answer information leaks into the "
                    "bag-of-tokens features (or a shortcut exists), which "
                    "invalidates the simple-baseline comparison",
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
                      "and each held-out stream is 0; no audit field shows a "
                      "distribution gap beyond same-stream sampling noise "
                      "(Monte Carlo-calibrated chi-square, Benjamini-"
                      "Hochberg-adjusted over the field x split family)",
        "decision": "a shared sample, or an adjusted distribution-shift "
                    "p-value below the pre-registered FDR, invalidates every "
                    "reported number",
    },
    "HB4": {
        "experiment": "baselines",
        "question": "Does the pipeline leak the answer into the training "
                    "signal (negative control)?",
        "hypothesis": "With the supervised answers permuted across the batch "
                      "during training, no true answer can be learned, so "
                      "accuracy on the unchanged test stream must stay at "
                      "chance.",
        "prediction": "the upper bound of the seed-clustered 95% bootstrap "
                      "CI of (control accuracy - chance) stays below the "
                      "pre-registered absolute tolerance "
                      "(Settings.negative_control_tol, default 0.02); the "
                      "clustered interval resamples training seeds and keeps "
                      "the shared finite-stream measurement error, so the "
                      "control cannot pass by scoring a lucky small test "
                      "stream (the seed-only and pooled Wilson intervals are "
                      "descriptive companions)",
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
        "prediction": "the tuned 1-layer standard transformer is competent "
                      "(seed-clustered 95% lower bound above chance) and its "
                      "seed-clustered 95% *upper* bound on the test stream "
                      "lies at least the pre-registered SESOI (0.01 "
                      "accuracy) below the rule-ceiling point estimate - "
                      "i.e. the plausible gap is practically meaningful, "
                      "not merely statistically non-zero",
        "decision": "the baseline is a complexity control, not a pipeline "
                    "validity gate.  The rung is read as *practically at the "
                    "ceiling* when its seed-clustered 95% *lower* bound "
                    "lies within SESOI of the ceiling (one-sided equivalence: the "
                    "whole interval sits inside the practically-saturated "
                    "band), in which case every depth-based motivation (E3) "
                    "must be annotated that a 1-layer model suffices at this "
                    "scale.  By contrast, 'depth measurably required' needs "
                    "the seed-clustered 95% *upper* bound to stay at least "
                    "SESOI below the ceiling: a statistically detectable but "
                    "practically negligible gap does not earn the complexity. "
                    "The bare 'CI includes the ceiling point estimate' "
                    "criterion is absence of evidence, so it is recorded as a "
                    "companion (hb6_supported_ci) and never decides.  A rung "
                    "at chance (HB0 competence prerequisite failed) shows "
                    "only that this step count did not solve the task - it is "
                    "not evidence that depth is required",
        "sesoi": {"accuracy": 0.01},
    },
    "HB7": {
        "experiment": "baselines",
        "question": "At which depth does the tuned standard transformer "
                    "saturate on the recall task, and is the 8-layer grid "
                    "depth-justified at this scale?",
        "hypothesis": "Depth helps the match-then-shift readout, but it "
                      "saturates at some small depth: once a rung's accuracy "
                      "is no longer distinguishable from the "
                      "parameter-free ceiling, deeper grids (E1/E3 at 8 "
                      "layers) can only test what depth unlocks, not whether "
                      "depth is needed.",
        "prediction": "the ladder's saturation depth d* - the smallest "
                      "*competent* depth whose seed-clustered 95% lower "
                      "bound lies within the pre-registered SESOI (0.01 accuracy) "
                      "of the rule-ceiling point estimate (one-sided "
                      "equivalence: the whole interval is inside the "
                      "practically-saturated band) - is reported (None when "
                      "no ladder rung is equivalent-within-SESOI); adjacent-"
                      "depth paired bootstrap CIs plus bootstrap-TOST "
                      "equivalence checks against the same SESOI show where "
                      "the gain stops being practically meaningful; the depth "
                      "demand is only readable when at least one rung is "
                      "competent (HB0: seed-clustered 95% lower bound above "
                      "chance)",
        "decision": "pre-registered readout: depth beyond two layers is "
                    "'measurably required' only when d* is None or d* > 2 "
                    "*and* at least one rung is above chance; for d* <= 2 "
                    "every depth-based motivation (E3's deep-contribution "
                    "premise, the 8-layer E1 setting) must carry the "
                    "annotation that a 2-layer standard transformer already "
                    "saturates the task at this scale; when no rung clears "
                    "chance the ladder did not solve the task at this "
                    "step count, so the depth demand is 'not determinable' "
                    "(never 'required') and E3's depth framing is "
                    "unverified.  Saturation is an equivalence claim with the "
                    "registered SESOI bound, never the bare 'CI reaches the "
                    "ceiling' (absence of evidence) reading",
        "sesoi": {"accuracy": 0.01},
    },
    "H1a": {
        "experiment": "e1",
        "question": "Does the calibrated gate use attention concentration / "
                    "length information beyond the query?",
        "hypothesis": "The query-only gate has no access to the attention "
                      "distribution, so |gate_vs_entropy| and "
                      "|gate_vs_maxprob| are both clearly larger for calib "
                      "than for query.",
        "prediction": "|gate_vs_entropy|(calib) - |gate_vs_entropy|(query) > 0 "
                      "and |gate_vs_maxprob|(calib) - "
                      "|gate_vs_maxprob|(query) > 0; the same direction holds "
                      "for the rank (Spearman) correlations as a robustness "
                      "check.  The registered "
                      "accuracy companion (calib - none and calib - query) "
                      "is a one-sided non-inferiority check: the calibrated "
                      "gate must not lose accuracy beyond the registered "
                      "SESOI to its controls - a loss larger than that bound "
                      "refutes the mechanism, while a gain is not required",
        "decision": "paired bootstrap 95% CI of the seed differences "
                    "excludes 0 after Benjamini-Hochberg FDR control across "
                    "the E1 contrasts (q=0.05); the accuracy companions are "
                    "one-sided non-inferiority tests against the registered "
                    "SESOI (lower bound above -SESOI), not directional "
                    "superiority claims",
        # Both distribution readouts are in the pre-registered family: the
        # claim is that calib reads the attention distribution while query
        # does not, and max-prob is the other observable summary of that
        # distribution.  Registering only entropy would leave half of the
        # stated evidence exploratory.
        "sesoi": {"accuracy": 0.01, "abs_gate_vs_entropy": 0.05,
                  "abs_gate_vs_maxprob": 0.05},
        # The one-sided accuracy companion covers the registered calib
        # contrasts (calib - none, calib - query) and is registered as
        # non-inferiority under ``sesoi_test``, not as a direction.  The
        # query-vs-none completeness companion measures whether *any* gate
        # moves accuracy and is also not a signed claim, so it registers its
        # own undirected key.
        "direction": {"abs_gate_vs_entropy": "+",
                      "abs_gate_vs_maxprob": "+",
                      "query_none_accuracy": None},
        "sesoi_test": {"accuracy": "noninferior"},
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
        "direction": {"gate_margin": "+"},
    },
    "H1c": {
        "experiment": "e1",
        "question": "Does the explicit null slot suppress writes when the "
                    "model is wrong, and does the null arm stay at least as "
                    "accurate as the no-gate baseline?",
        "hypothesis": "null_mass is higher on errors than on correct "
                      "answers (the null slot absorbs the write).  Because "
                      "the null arm claims a dedicated slot is a benign, "
                      "useful addition (not an accuracy trade), its accuracy "
                      "must not fall below the no-gate ``none`` arm - a "
                      "significant accuracy loss would mean the null slot "
                      "buys its suppression by breaking recall.",
        "prediction": "null_mass_error_margin > 0 (and the same margin "
                      "remains positive on the high-maxprob false-"
                      "concentration subset); and the null arm's "
                      "accuracy relative to ``none`` stays inside the "
                      "pre-registered non-inferiority bound (null - none "
                      ">= -SESOI, i.e. the 95% CI of null - none lies above "
                      "the negative bound rather than materially below "
                      "``none``)",
        "decision": "paired bootstrap 95% CI excludes 0 for the margin after "
                    "Benjamini-Hochberg FDR control across the E1 contrasts "
                    "(q=0.05); the null-vs-none accuracy row is a *null* "
                    "reverse check, read as a one-sided non-inferiority test: "
                    "the lower bound of the CI of null - none must stay above "
                    "the registered bound (-accuracy SESOI), so a confidence "
                    "interval entirely below it refutes the claim rather than "
                    "being read as a null result",
        "sesoi": {"null_mass_error_margin": 0.02,
                  "null_mass_false_conc_margin": 0.02, "accuracy": 0.01},
        "direction": {"null_mass_error_margin": "+",
                      "null_mass_false_conc_margin": "+"},
        # The accuracy companion is a null/non-inferiority prediction: its
        # bound is the accuracy SESOI above, tested one-sided (CI above the
        # negative bound), not a directional superiority claim.
        "sesoi_test": {"accuracy": "noninferior"},
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
                      "margin - a component removable without effect is not "
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
                    "for each *single-term* ablated arm after "
                    "Benjamini-Hochberg FDR control across the E1 family "
                    "(q=0.05); the joint both-terms-off cell is not a "
                    "single-factor row and enters only the registered 2x2 "
                    "factorial main effect.  The mechanism check "
                    "validates that each ablation removed the intended "
                    "pathway and is reported alongside, not as a decision "
                    "criterion",
        "sesoi": {"accuracy": 0.01, "gate_margin": 0.02,
                  "abs_gate_vs_entropy": 0.05},
        "direction": {"accuracy": "+", "gate_margin": "+",
                      "abs_gate_vs_entropy": "+"},
    },
    "H1e": {
        "experiment": "e1",
        "question": "Does the calibrated gate rank the model's own errors "
                    "above the query-only gate's ranking (the operational, "
                    "imbalance-aware reading of 'the gate doubts itself')?",
        "hypothesis": "Errors are the rare class after convergence, so the "
                      "mechanism's operational value is a rare-positive "
                      "ranking claim: low calibrated-gate values should "
                      "concentrate on wrong answers more than the query "
                      "gate's values do.  Under this imbalance the metric "
                      "choice matters: PR-AUC is the headline (ROC-AUC stays "
                      "optimistic as the negative population grows), and the "
                      "recall of the most doubtful 10% is its operational "
                      "companion.",
        "prediction": "prauc_lift_gate_to_error(calib) - "
                      "prauc_lift_gate_to_error(query) > 0 and "
                      "recall_at_10pct_gate_to_error(calib) - "
                      "recall_at_10pct_gate_to_error(query) > 0 (lift = "
                      "PR-AUC / prevalence, 1.0 = random ranking; ROC-AUC is "
                      "reported as the optimistic companion, not the "
                      "decision metric)",
        "decision": "paired bootstrap 95% CI of the pre-registered "
                    "imbalance-aware readout excludes 0 after "
                    "Benjamini-Hochberg FDR control across the E1 contrasts "
                    "(q=0.05); seeds whose stats stream contains no errors "
                    "leave the metric undefined and are recorded as such, "
                    "never imputed",
        "sesoi": {"prauc_lift_gate_to_error": 0.1,
                  "recall_at_10pct_gate_to_error": 0.02},
        "direction": {"prauc_lift_gate_to_error": "+",
                      "recall_at_10pct_gate_to_error": "+"},
    },
    "H1f": {
        "experiment": "e1",
        "question": "Is the calibrated gate's association with the attention "
                    "distribution carried by the real entropy *information*, "
                    "or only by the presence of an extra data-dependent "
                    "input (active placebo control)?",
        "hypothesis": "An active-control arm with the identical calib "
                      "architecture, parameter count, computation and "
                      "optimiser, but whose gate reads a fixed "
                      "position-shuffled copy of the attention entropy (same "
                      "values and marginal distribution, decorrelated from "
                      "each queried row's own attention), must lose the "
                      "gate/entropy association and must not reproduce the "
                      "full gate's advantage over the query-only gate.  The "
                      "leave-one-out ablations remove the term *and* its "
                      "parameter, so they cannot separate 'the gate reads "
                      "the concentration information' from 'an extra "
                      "data-dependent input helps'; this control can.",
        "prediction": "|gate_vs_entropy|(calib) - |gate_vs_entropy|("
                      "calib_shuf_ent) > 0 (the shuffled signal contains no "
                      "information about the read row, so the association "
                      "must collapse), and accuracy and gate_margin of full "
                      "calib exceed the placebo arm's - a placebo that "
                      "matches full calib would refute the calibration "
                      "story and show the effect is capacity, not "
                      "attention-distribution information",
        "decision": "paired bootstrap 95% CIs of the three calib - placebo "
                    "differences exclude 0 in the pre-registered directions "
                    "after Benjamini-Hochberg FDR control across the E1 "
                    "family (q=0.05).  The mechanism leg validates that the "
                    "placebo actually severed the signal pathway; the "
                    "accuracy/gate_margin legs are the specificity test - a "
                    "reversed or null read there means the calibration "
                    "benefit is not attributable to the entropy information",
        "sesoi": {"accuracy": 0.01, "gate_margin": 0.02,
                  "abs_gate_vs_entropy": 0.05},
        "direction": {"accuracy": "+", "gate_margin": "+",
                      "abs_gate_vs_entropy": "+"},
    },
    "H1g": {
        "experiment": "e1",
        "question": "Is the variant difference in attention-distribution use "
                    "visible at the correlation level itself: calib's gate "
                    "is associated with entropy / max-prob, while the "
                    "query-only gate is not strongly associated?",
        "hypothesis": "H1a tests that calib's |r| is larger than query's.  "
                      "A difference of two near-zero associations would not "
                      "show that calib reads the distribution, and "
                      "'query not significant' would not show that it does "
                      "not.  This correlation-level claim makes both halves "
                      "falsifiable: calib's one-sample |r| must clear a "
                      "practical floor, and query's must lie inside a "
                      "pre-registered near-zero band.",
        "prediction": "calib |r|(gate, entropy) and |r|(gate, max-prob) > 0 "
                      "by at least the SESOI; query's two |r| values lie "
                      "inside the equivalence bound 0.10",
        "decision": "one-sample bootstrap 95% CIs through the same "
                    "Benjamini-Hochberg E1 family (q=0.05): calib rows are "
                    "directional (CI excludes 0 and effect >= SESOI), query "
                    "rows are bootstrap-TOST equivalence tests (CI inside "
                    "+-bound).  A query CI that merely includes 0 while "
                    "spanning the bound is 'not established', not evidence "
                    "of no association",
        "sesoi": {"calib_abs_gate_vs_entropy": 0.05,
                  "calib_abs_gate_vs_maxprob": 0.05,
                  "query_abs_gate_vs_entropy": 0.10,
                  "query_abs_gate_vs_maxprob": 0.10},
        "direction": {"calib_abs_gate_vs_entropy": "+",
                      "calib_abs_gate_vs_maxprob": "+"},
        "sesoi_test": {"query_abs_gate_vs_entropy": "equivalence",
                       "query_abs_gate_vs_maxprob": "equivalence"},
    },
    "H2a": {
        "experiment": "e2",
        "question": "Does length calibration slow the degradation beyond "
                    "the training length?",
        "hypothesis": "Because the gate subtracts a*log(T) and the "
                      "log(T)-normalised entropy, calib degrades less than "
                      "none when evaluated at 2x-spaced lengths.",
        "prediction": "accuracy_drop(calib) < accuracy_drop(none) and "
                      "accuracy@longest(calib) > accuracy@longest(none); "
                      "the fitted accuracy slope on log(T) across the whole "
                      "sweep is also less negative for calib (endpoint-only "
                      "reads can hide a mid-sweep reversal)",
        "decision": "paired bootstrap 95% CI of the differences excludes 0 "
                    "after Benjamini-Hochberg FDR control across the E2 "
                    "contrasts (q=0.05)",
        "sesoi": {"accuracy_at_longest": 0.01, "accuracy_drop": 0.01,
                  "accuracy_slope_log_len": 0.01},
        "direction": {"accuracy_at_longest": "+", "accuracy_drop": "-",
                      "accuracy_slope_log_len": "+"},
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
        "direction": {"entropy_drift": "+"},
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
                      "close to the full calib arm (within the registered "
                      "bound on both accuracy_at_longest and accuracy_drop)",
        "decision": "paired bootstrap 95% CI of calib - calib_ent_only on "
                    "accuracy_at_longest excludes 0 after "
                    "Benjamini-Hochberg FDR control across the E2 contrasts "
                    "(q=0.05).  The calib_len_only arm is a *null* reverse "
                    "check, not a superiority row: its 95% CI for calib - "
                    "calib_len_only must lie inside +-SESOI on both "
                    "accuracy_at_longest and accuracy_drop (bootstrap TOST, "
                    "same FDR family); a non-significant difference is not "
                    "enough to call the term dispensable",
        "sesoi": {"accuracy_at_longest": 0.01, "accuracy_drop": 0.01,
                  # Separately registered bounds for the reverse check:
                  # the displayed metrics are shared with the directional
                  # calib - calib_ent_only rows, so the null prediction
                  # needs its own keys in the registry.
                  "calib_minus_len_only_accuracy_at_longest": 0.01,
                  "calib_minus_len_only_accuracy_drop": 0.01},
        "direction": {"accuracy_at_longest": "+", "accuracy_drop": "-"},
        "sesoi_test": {
            "calib_minus_len_only_accuracy_at_longest": "equivalence",
            "calib_minus_len_only_accuracy_drop": "equivalence",
        },
    },
    "H2d": {
        "experiment": "e2",
        "question": "Does calib's length robustness exceed the closest "
                    "simpler gate (query), not only the no-gate baseline?",
        "hypothesis": "The length-robustness claim is only about the "
                      "calibration terms if the calibrated gate degrades "
                      "less than the query-only gate, which shares the same "
                      "attention and gating architecture but lacks the "
                      "length and concentration terms.  A gain over "
                      "``none`` alone could be produced by the presence of "
                      "any learned gate.",
        "prediction": "accuracy_at_longest(calib) > "
                      "accuracy_at_longest(query) and "
                      "accuracy_drop(calib) < accuracy_drop(query), with "
                      "the same sign on the whole-sweep accuracy slope on "
                      "log(T)",
        "decision": "paired bootstrap 95% CI excludes 0 in the "
                    "pre-registered directions after Benjamini-Hochberg FDR "
                    "control across the E2 contrasts (q=0.05)",
        "sesoi": {"accuracy_at_longest": 0.01, "accuracy_drop": 0.01,
                  "accuracy_slope_log_len": 0.01},
        "direction": {"accuracy_at_longest": "+", "accuracy_drop": "-",
                      "accuracy_slope_log_len": "+"},
    },
    "H2e": {
        "experiment": "e2",
        "question": "Does the concentration term carry an average "
                    "contribution across the length sweep, and is the two-"
                    "term design additive (full 2x2 over the length sweep)?",
        "hypothesis": "With all four calibration-term cells trained on the "
                      "same length sweep, the concentration term's effect "
                      "averaged over both length-term levels must be "
                      "measured as well as the length term's (H2c); the "
                      "length x concentration interaction is the "
                      "two-sided companion that checks the terms are not "
                      "merely interchangeable.",
        "prediction": "concentration main effect on accuracy_at_longest > 0 "
                      "and on accuracy_drop < 0; the interaction "
                      "(length effect under concentration on minus under "
                      "concentration off) is reported as a two-sided "
                      "readout, since the design tests additivity rather "
                      "than a signed interaction",
        "decision": "paired bootstrap 95% CIs through the same "
                    "Benjamini-Hochberg family as the H2c rows (q=0.05): "
                    "the concentration main effect carries a registered "
                    "direction and SESOI; the interaction is two-sided "
                    "(no registered direction) but must still exceed its "
                    "SESOI to be practically meaningful",
        "sesoi": {"accuracy_at_longest": 0.01, "accuracy_drop": 0.01},
        # Direction is keyed separately from the display metric so the
        # two-sided interaction rows (same metrics) do not inherit the
        # directional prediction of the main-effect rows.
        "direction": {"concentration_accuracy_at_longest": "+",
                      "concentration_accuracy_drop": "-"},
    },
    "H2f": {
        "experiment": "e2",
        "question": "Does the full calibrated gate actually learn the "
                    "explicit corrections its mechanism claim names?",
        "hypothesis": "The calibrated gate is defined as "
                      "sigmoid((W_g x - a*log T - c*scale*Hbar)/tau).  The "
                      "mechanism claim is therefore an explicit correction "
                      "with a>0 (subtract the log-length entropy growth) and "
                      "c*scale>0 (subtract the normalised concentration).  "
                      "A full-calib arm whose learned a or c*scale is "
                      "non-positive is not implementing that correction, "
                      "whatever its endpoint accuracy.",
        "prediction": "one-sample across training seeds: a_mean > 0 and "
                      "c_effective_mean > 0; a gate that passes the "
                      "endpoint accuracy test with a non-positive "
                      "coefficient is a capacity effect, not the claimed "
                      "explicit calibration",
        "decision": "one-sample bootstrap 95% CI excludes 0 in the "
                    "registered direction after Benjamini-Hochberg FDR "
                    "control across the E2 contrasts (q=0.05).  This is a "
                    "mechanism-manipulation check: failure annotates H2a/H2b "
                    "(and the E2 coefficients table) as not showing the "
                    "named explicit correction, even when an endpoint "
                    "accuracy contrast passes",
        # Coefficients are in pre-sigmoid gate units; 0.01 is a deliberately
        # small practical floor so a numerically non-zero but negligible
        # coefficient is not sold as an implemented mechanism.
        "sesoi": {"a_mean": 0.01, "c_effective_mean": 0.01},
        "direction": {"a_mean": "+", "c_effective_mean": "+"},
    },
    "H2g": {
        "experiment": "e2",
        "question": "Is the log-T normalisation of the concentration term "
                    "load-bearing, or would raw entropy do just as well "
                    "across lengths?",
        "hypothesis": "The calibrated gate claims a length-stable "
                      "concentration signal: Hbar = H / log T.  H2b measures "
                      "that H/log T is stable, but a stable *statistic* does "
                      "not show that the model needs the normalisation.  An "
                      "arm with the identical calib architecture, parameter "
                      "count, computation and length term, but reading raw "
                      "row-mean entropy (Cfg.calib_ent_norm=False), therefore "
                      "isolates the normalisation.  At the training length "
                      "the two parameterisations differ by a constant factor "
                      "that the learned c can absorb, so the informative "
                      "divergence is at the extrapolation lengths, where H "
                      "keeps growing with log T and an unnormalised term "
                      "over-corrects relative to the registered gate.",
        "prediction": "accuracy_at_longest(calib) > "
                      "accuracy_at_longest(calib_raw_ent) and "
                      "accuracy_drop(calib) < accuracy_drop(calib_raw_ent); "
                      "the full arm's concentration coefficient was already "
                      "shown positive by H2f, so this is a comparison between "
                      "two implemented corrections, not between a term and "
                      "no term",
        "decision": "paired bootstrap 95% CI of calib - calib_raw_ent "
                    "excludes 0 in the pre-registered directions after "
                    "Benjamini-Hochberg FDR control across the E2 contrasts "
                    "(q=0.05); a CI including 0 while spanning the SESOI is "
                    "'inconclusive', a confidently positive effect below the "
                    "SESOI is 'practically negligible', and an FDR-significant "
                    "effect in the opposite direction refutes the "
                    "normalisation claim rather than being read as a null "
                    "result",
        "sesoi": {"accuracy_at_longest": 0.01, "accuracy_drop": 0.01},
        "direction": {"accuracy_at_longest": "+", "accuracy_drop": "-"},
    },
    "H3a": {
        "experiment": "e3",
        "question": "Does dynamic normalisation make deep layers write more "
                    "into the residual stream?",
        "hypothesis": "seednorm (per-token dynamic scale) raises the "
                      "deep/early gate ratio relative to pre-LN.",
        "prediction": "gate_depth_ratio(seednorm) > gate_depth_ratio(pre) "
                      "and the deep/early residual-norm ratio "
                      "resid_norm_last_over_first is larger under seednorm "
                      "than under pre-LN; the registered "
                      "accuracy companion (calib - query within each norm "
                      "and LayerScale level) is a one-sided non-inferiority "
                      "check: the calibrated gate must not lose accuracy "
                      "beyond the registered SESOI to the query-only gate - "
                      "a loss larger than that bound refutes the co-design "
                      "story, while a gain is not required",
        "decision": "paired bootstrap 95% CI excludes 0 after "
                    "Benjamini-Hochberg FDR control across the E3 contrasts "
                    "(q=0.05); the accuracy companion is a one-sided "
                    "non-inferiority test against the registered SESOI "
                    "(lower bound above -SESOI), not a directional "
                    "superiority claim",
        "sesoi": {"gate_depth_ratio": 0.05,
                  "resid_norm_last_over_first": 0.05,
                  "accuracy": 0.01,
                  "norm_gate_depth_ratio": 0.05,
                  "norm_resid_norm_last_over_first": 0.05,
                  "norm_accuracy": 0.01},
        # The average-own-level main effects (each norm level vs pre-LN,
        # collapsed over gate and LayerScale) reuse the H3a mechanism
        # direction for seednorm; post/mix are two-sided companions that
        # measure whether a normalisation level moves the readout at all.
        # The companions register their own (undirected) keys so they cannot
        # inherit seednorm's sign through the shared display metric.  The
        # accuracy companion is non-inferiority under ``sesoi_test``, not a
        # directional superiority claim, so it has no direction entry.
        "direction": {"gate_depth_ratio": "+",
                      "resid_norm_last_over_first": "+",
                      "norm_gate_depth_ratio": "+",
                      "norm_resid_norm_last_over_first": "+",
                      "postmix_norm_gate_depth_ratio": None,
                      "postmix_norm_resid_norm_last_over_first": None},
        "sesoi_test": {"accuracy": "noninferior"},
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
    "H3c": {
        "experiment": "e3",
        "question": "Is the calibrated gate's benefit specific to the "
                    "dynamic (seednorm) normalisation - a co-design "
                    "interaction - rather than uniform across norms?",
        "hypothesis": "The calib-over-query accuracy advantage is larger "
                      "under seednorm than under pre-LN (positive "
                      "interaction), because the learned calibration terms "
                      "read the same per-token dynamic scale; the "
                      "deep/early gate ratio shows the same interaction.",
        "prediction": "difference-in-differences "
                      "[acc(calib|seednorm) - acc(query|seednorm)] - "
                      "[acc(calib|pre) - acc(query|pre)] > 0 at LayerScale "
                      "off, with the same sign for gate_depth_ratio - and "
                      "the marginal calib-vs-query contrasts must remain "
                      "controlled for the interaction to be attributable",
        "decision": "paired bootstrap 95% CI of the per-seed "
                    "difference-in-differences excludes 0 after "
                    "Benjamini-Hochberg FDR control across the E3 contrasts "
                    "(q=0.05)",
        "sesoi": {"accuracy": 0.01, "gate_depth_ratio": 0.05},
        "direction": {"accuracy": "+", "gate_depth_ratio": "+"},
    },
}


def _validate_training_seeds(seeds):
    """Validate the configured training-seed set; return it as an int tuple.

    A training-seed set must be non-empty, integral, duplicate-free and
    disjoint from the reserved held-out evaluation streams:

    * **duplicates** break the premises the seed-level statistics rest on.
      A paired contrast pairs runs by seed (``_paired_diff``); a seed that
      appears twice on a side is counted twice in ``diffs``, so ``n`` is
      inflated and the bootstrap CI too narrow, and the "independent
      seeds" behind a Mean +- Std are not independent at all;
    * **reserved seeds** (``VAL_SEED`` / ``TEST_SEED`` / ``STATS_SEED``)
      would fit the model on the validation/test/statistics streams, so
      the reported numbers stop being held-out.

    Comparisons use :func:`cage.data.stream_seed_key`, the 32-bit key the
    CPU stream generators actually consume: ``0`` and ``2**32`` are the
    same stream and count as duplicates, and a reserved-seed alias is
    refused exactly like the reserved seed itself.

    Checked both at ``Settings`` construction and from ``_run_context``
    (which every runner reaches before training), so a post-construction
    attribute assignment - the CLI applies ``--seeds`` that way - cannot
    bypass it either.
    """
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError(
            "seeds must be a non-empty sequence of training seeds")
    if any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise ValueError(
            f"training seeds must be plain integers, got {seeds!r}")
    seeds = tuple(int(s) for s in seeds)
    seen, duplicates = {}, []
    for s in seeds:
        key = stream_seed_key(s)
        if key in seen:
            duplicates.extend((seen[key], s))
        else:
            seen[key] = s
    if duplicates:
        raise ValueError(
            f"duplicate training seeds/streams {sorted(set(duplicates))} in "
            f"{list(seeds)}: a repeated seed is the same run twice, so paired "
            f"contrasts would double-count it (inflated n, too-narrow CI) and "
            f"the seed-level Mean +- Std would not average independent runs")
    reserved = sorted(
        s for s in seeds
        if any(stream_seed_key(s) == stream_seed_key(r)
               for r in RESERVED_EVAL_SEEDS))
    if reserved:
        raise ValueError(
            f"training seeds {reserved} are reserved held-out evaluation "
            f"streams {RESERVED_EVAL_SEEDS}: training on them would fit the "
            f"model on validation/test data")
    return seeds


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
    # The simple-baseline controls are tuned harder than the mechanism grids
    # on purpose: "the complex model beats a tuned simple baseline" is only
    # credible when the simple baseline got its best shot.  The probe grid
    # spans 30x (a diverged high-lr point is recorded and excluded, not
    # read as "cannot do better"); the depth ladder re-tunes the learning
    # rate and step budget per depth so the shallow rungs are not
    # handicapped by inheriting an under-trained recipe.
    #
    # The probe evaluation stream is deliberately wider than the grid
    # evaluation stream: HB2 certifies "the tuned probe stays below
    # 2x chance", and with a narrow stream the per-seed binomial noise
    # (sd ~ sqrt(p(1-p)/n)) can exceed the chance-level margin, making the
    # certificate fail for power reasons on a clean pipeline.  512x16
    # samples shrink that noise well below the margin; the one-sided
    # p-value and the seeds_for_certification power plan are reported next
    # to the decision so residual underpower is visible either way.
    probe_eval_batch: int = 512
    probe_eval_batches: int = 16
    probe_lr_grid: tuple = (1e-2, 3e-3, 1e-3, 3e-4)
    # The simple baseline is tuned over (lr, steps), not lr alone: a fixed
    # 400-step budget can make a linear probe look weaker than it is.  An
    # empty tuple keeps the legacy single-budget behaviour for callers that
    # pass ``probe_steps`` explicitly.
    probe_steps_grid: tuple = (200, 400, 800)
    depth_ladder: tuple = (1, 2, 4)
    # The depth ladder tunes (lr, steps) per rung.  A fixed step budget can
    # make a shallow rung look weaker merely because its best learning rate
    # was selected at an under-trained budget; the multipliers are relative
    # to settings.steps so a --steps override propagates.  The multi-axis
    # boundary audit flags a winner at the 2x edge instead of silently
    # treating it as resolved.
    depth_step_multipliers: tuple = (1.0, 2.0)
    dropout: float = 0.0        # model regularisation (recorded in the recipe)
    # Deterministic kernels are on by default: fixed seeds alone do not make
    # CUDA runs repeatable when an op may pick a non-deterministic algorithm.
    # The official grids therefore run under the stricter mode unless the
    # caller explicitly opts out (--no-deterministic), and the resolved value
    # is stamped into every record.
    deterministic: bool = True
    lr: float | None = None
    lr_grid: tuple = (1e-3, 3e-4, 1e-4)
    # Second, deliberately separate accounting of baseline fairness: the
    # controlled grid above freezes ONE recipe (tuned on gate=none) for every
    # arm, which makes the gate contrast clean but can handicap a baseline
    # whose best lr/temperature differs.  This companion gives every arm its
    # own validation-selected recipe (lr, and calib_tau_init for calib) and
    # then re-pairs the arms on the sealed test stream.  It is an exploratory
    # benchmark, never a replacement for the pre-registered controlled
    # family: a calib gain that survives both readings is the robust claim.
    per_variant_tuning: bool = True
    # Selection seeds for the per-variant tuned account.  Default to the
    # pre-registered claim floor: the shared/controlled recipe already
    # refuses to freeze a point on fewer seeds than a claim would need, and
    # the tuned baseline comparison should not be decided by a smaller
    # (noisier) argmax than that.  ``min(per_variant_tune_seeds,
    # len(seeds))`` is used, so quick configs still work.
    per_variant_tune_seeds: int = 3
    calib_tau_grid: tuple = (0.5, 0.9, 1.5)
    curve_points: int = 10      # validation checks per run (0 disables)
    val_batches: int = 2        # validation batches per check (fixed stream)
    max_error_cases: int = 64   # per gate x seed, dumped to disk
    fdr_q: float = 0.05         # BH false-discovery rate for contrast families
    # Pre-registered floor on the number of independent training seeds a
    # contrast needs before it may support a claim: a 0.2% effect next to a
    # 0.5% seed-level std is not evidence, and a two-seed "win" is noise.
    min_seeds_for_claim: int = 3
    overfit_steps: int = 300    # HB5 tiny-batch overfit check
    overfit_batch: int = 8
    overfit_lr: float = 3e-3
    negative_control_tol: float = 0.02  # HB4 pre-registered tolerance
    out_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path("results"))

    def __post_init__(self) -> None:
        # Fail at construction on an invalid training-seed set instead of
        # letting a duplicated reserve-dataset seed flow into a grid; the
        # runner entry point re-checks (see ``_validate_training_seeds``)
        # because ``--seeds`` is applied to the attribute afterwards.
        self.seeds = _validate_training_seeds(self.seeds)
        check_data_geometry(self.n_pairs, self.vocab, self.max_len)
        raw_q = self.fdr_q
        try:
            q = float(raw_q)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"fdr_q must be a finite number strictly between 0 and 1, "
                f"got {raw_q!r}") from exc
        if not (math.isfinite(q) and 0.0 < q < 1.0):
            raise ValueError(
                f"fdr_q must be a finite number strictly between 0 and 1, "
                f"got {raw_q!r}")
        self.fdr_q = q
        floor = _positive_integer(self.min_seeds_for_claim)
        if floor is None:
            raise ValueError(
                f"min_seeds_for_claim must be a positive integer, got "
                f"{self.min_seeds_for_claim!r}")
        self.min_seeds_for_claim = int(floor)
        if not isinstance(self.per_variant_tuning, bool):
            raise ValueError(
                f"per_variant_tuning must be a bool, got "
                f"{self.per_variant_tuning!r}")
        if _positive_integer(self.per_variant_tune_seeds) is None:
            raise ValueError(
                f"per_variant_tune_seeds must be a positive integer, got "
                f"{self.per_variant_tune_seeds!r}")
        for name in ("lr_grid", "calib_tau_grid"):
            values = tuple(getattr(self, name))
            if not values or any(
                    not isinstance(v, (int, float)) or isinstance(v, bool)
                    or not math.isfinite(float(v)) or float(v) <= 0.0
                    for v in values):
                raise ValueError(
                    f"{name} must be a non-empty tuple of finite positive "
                    f"numbers, got {values!r}")

    @classmethod
    def quick(cls, device: str = "cpu") -> "Settings":
        """Tiny pipeline-check configuration (never for reported numbers)."""
        return cls(
            device=device, d_model=48, n_heads=4, n_layers=4, max_len=64,
            vocab=64, batch=16, n_pairs=6, steps=12, seeds=(0, 1),
            eval_batch=64, eval_batches=2, stats_batch=32, stats_batches=2,
            probe_steps=40, probe_steps_grid=(20, 40, 80),
            probe_batch=16, probe_eval_batch=64,
            probe_eval_batches=2, overfit_steps=100,
        )


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _depth_step_candidates(settings: Settings) -> tuple:
    """Step budgets searched by the depth ladder and its capacity control.

    The search is expressed as multipliers of ``settings.steps`` so the
    command-line ``--steps`` override shifts the whole grid.  The default
    budget is always included even when the multiplier list is empty or
    omits 1.0.
    """
    multipliers = tuple(settings.depth_step_multipliers or (1.0,))
    raw_steps = settings.steps
    base_steps = _positive_integer(raw_steps)
    if base_steps is None:
        raise ValueError(
            f"steps must be a positive integer before the depth ladder can "
            f"build its step grid, got {raw_steps!r}")
    steps = {int(base_steps)}
    for multiplier in multipliers:
        if not _finite(multiplier) or float(multiplier) <= 0.0:
            raise ValueError(
                f"depth_step_multipliers must be finite positive numbers, "
                f"got {multiplier!r}")
        steps.add(max(1, int(round(base_steps * float(multiplier)))))
    return tuple(sorted(steps))


def _seed_order(seed):
    """Deterministic ordering for seed labels of mixed / missing types."""
    if isinstance(seed, (int, float)) and not isinstance(seed, bool):
        return (0, float(seed), "")
    return (1, 0.0, str(seed))


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


def _linear_slope(xs, ys):
    """Least-squares slope of ``ys`` on ``xs`` (NaN when not identifiable).

    Used for the E2 degradation-shape readout: fitting accuracy on
    ``log(T)`` across the whole sweep tests "degrades more slowly" with all
    the measured lengths instead of only the endpoint, and is scale-free
    with respect to the absolute accuracy level.  Non-finite pairs are
    dropped; fewer than two points or a constant x returns NaN rather than
    an arbitrary number.
    """
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)
             if _finite(x) and _finite(y)]
    if len(pairs) < 2:
        return float("nan")
    n = len(pairs)
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    denom = sum((x - mx) ** 2 for x, _ in pairs)
    if denom <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / denom


def _aggregate_runs(runs, skip=("seed", "lr", "params", "eval_seed",
                                "stats_seed")):
    keys = sorted({k for r in runs for k, v in r.items()
                   if _finite(v) and k not in skip})
    return {k: _mean_std([r[k] for r in runs if k in r]) for k in keys}


# Recipe fields that the controlled-comparison contract requires to be
# *identical* on both sides of every contrast (see ``_controlled_audit``):
# optimizer-relevant learning rate, schedule length, batch geometry, data
# sizes and every evaluation-stream size.  ``config`` is the resolved
# Settings snapshot stored with each run record.  The per-run evaluation
# stream *seeds* are not Settings fields; the run records stamp them and
# ``_recipe_view`` adds them to the audited view, so "same stream size on
# a different stream" is caught too.
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
                            "calib_use_len", "calib_use_ent",
                            "calib_shuffle_ent", "calib_ent_norm")

# Ground truth of the calib gate's additive terms, keyed by the resolved
# ``(calib_use_len, calib_use_ent, calib_shuffle_ent)`` flags of the model
# that trained: the cell of the 2x2 "length term x concentration term"
# design a run is in, plus the E1 active-control (shuffled entropy) cell.
# Cross-checking the declared ``calib_variant`` label against the flags the
# model was actually built with catches a mislabelled or mis-wired arm.
_CALIB_FACTORIAL = {
    (True, True, False): "full",
    (True, False, False): "calib_len_only",
    (False, True, False): "calib_ent_only",
    (False, False, False): "calib_no_terms",
    (True, True, True): "calib_shuf_ent",
}

_CALIB_FACTORIAL_CELLS = ("full", "calib_len_only", "calib_ent_only",
                          "calib_no_terms")

# The two axes of the calibration-term 2x2.  A main effect averages the
# factor's effect over the two levels of the other term, so the "on"/"off"
# cells and the two within-level pairwise audits are both part of the
# contract.  ``model_key`` is the resolved Cfg flag that must be the only
# model-config difference in each pair.
_FACTORIAL_MAIN_EFFECTS = {
    "length": {
        "on": ("full", "calib_len_only"),
        "off": ("calib_ent_only", "calib_no_terms"),
        # Within each concentration level, the pair that differs only in
        # the length flag: full-vs-ent_only (entropy on) and
        # len_only-vs-no_terms (entropy off).
        "pairs": (("full", "calib_ent_only"),
                  ("calib_len_only", "calib_no_terms")),
        "model_key": "calib_use_len",
    },
    "concentration": {
        "on": ("full", "calib_ent_only"),
        "off": ("calib_len_only", "calib_no_terms"),
        # Within each length level, the pair that differs only in the
        # concentration flag: full-vs-len_only and ent_only-vs-no_terms.
        "pairs": (("full", "calib_len_only"),
                  ("calib_ent_only", "calib_no_terms")),
        "model_key": "calib_use_ent",
    },
}

_CALIB_PLACEBO_CELL = "calib_shuf_ent"

# E2 normalisation-ablation cell (H2g): the full calib architecture and both
# additive terms, but the concentration term reads the raw row-mean entropy
# instead of the registered H/log(T).  It is a one-factor ablation of the
# normalisation itself (same parameters, same length term, same kernel), kept
# outside the 2x2 term factorisation because at a fixed length the two forms
# differ only by a constant absorbed into ``calib_c`` - it is identifiable
# only across the length sweep.
_CALIB_RAW_ENT_CELL = "calib_raw_ent"

# Factor-vector expansion of a declared ``calib_variant`` cell label.  New
# records carry the resolved ``model_cfg`` (the ground truth), but pooled or
# legacy records may carry only the label; expanding the label here keeps the
# single-factor audit meaningful for them too.  ``calib_variant`` names a
# *cell*, not one factor: e.g. ``calib_no_terms`` turns off both additive
# terms, while ``calib_raw_ent`` differs from ``calib`` only in the entropy
# normalisation.
_CALIB_CELL_FACTORS = {
    cell: {"calib_use_len": bool(flags[0]),
           "calib_use_ent": bool(flags[1]),
           "calib_shuffle_ent": bool(flags[2]),
           "calib_ent_norm": True}
    for flags, cell in _CALIB_FACTORIAL.items()
}
_CALIB_CELL_FACTORS[_CALIB_RAW_ENT_CELL] = {
    "calib_use_len": True, "calib_use_ent": True,
    "calib_shuffle_ent": False, "calib_ent_norm": False}


def _calib_cell(run):
    """The calibration-term cell a run actually trained (None when N/A).

    Uses the resolved ``model_cfg`` when present (ground truth); falls back
    to the declared label for records that predate the stamp.  A shuffle
    flag combined with anything but the full term set is not a declared
    cell: it returns a non-matching name so the label audit fails loudly
    instead of crediting a mis-wired placebo.  Likewise, the raw-entropy
    normalisation ablation is a declared cell only when both additive terms
    are on and the entropy is not shuffled (the flags that define H2g's
    single-factor arm).
    """
    cfg = run.get("model_cfg")
    if cfg is not None:
        if cfg.get("gate") != "calib":
            return None
        flags = (bool(cfg.get("calib_use_len", True)),
                 bool(cfg.get("calib_use_ent", True)),
                 bool(cfg.get("calib_shuffle_ent", False)))
        if not bool(cfg.get("calib_ent_norm", True)):
            return (_CALIB_RAW_ENT_CELL
                    if flags == (True, True, False)
                    else "calib_raw_ent_undeclared")
        cell = _CALIB_FACTORIAL.get(flags)
        if cell is None and flags[2]:
            return "calib_shuf_ent_undeclared"
        return cell
    label = run.get("calib_variant")
    return label or None


def _coverage_cell_key(cell) -> str:
    """JSON-safe key for a design cell (tuples become ``a|b|c``)."""
    if isinstance(cell, tuple):
        return "|".join(str(part) for part in cell)
    return str(cell)


def _cell_seed_balance(runs, cell_of, expected, planned_seeds):
    """Seeds per design cell vs the planned training-seed set.

    Presence-only coverage can call a factorial complete while one cell
    silently lost seeds: cell membership would then correlate with seed
    availability, so that cell's mean is read on a different sample than
    its neighbours' - a confound no per-contrast recipe audit sees if the
    affected contrast is not pre-registered (e.g. an E3 cell that only
    enters exploratory reads).  Returns the per-cell seed payload and the
    list of unbalanced cell keys.
    """
    planned = sorted({int(s) for s in planned_seeds}) if planned_seeds else []
    seen: dict[str, set] = {}
    for run in runs:
        cell = cell_of(run)
        seed = run.get("seed")
        if cell is None or seed is None:
            continue
        seen.setdefault(_coverage_cell_key(cell), set()).add(int(seed))
    balance = {}
    unbalanced = []
    for cell in expected:
        key = _coverage_cell_key(cell)
        seeds = sorted(seen.get(key, set()))
        ok = bool(planned) and seeds == planned
        balance[key] = {"seeds": seeds, "planned": planned, "balanced": ok}
        if not ok:
            unbalanced.append(key)
    return balance, unbalanced


def _report_ablation_coverage(out_dir, tag, runs, cells=_CALIB_FACTORIAL_CELLS,
                              planned_seeds=None, design=None):
    """Verify a declared factorial ablation ran every cell of its design.

    A "each component is doing work" claim over a 2x2 (or leave-one-out)
    design is only supported if every cell was actually trained; a missing
    cell silently leaves one effect unmeasured, and a cell that trained
    only part of the planned seed set is read on a different sample than
    the others (see ``_cell_seed_balance``).  The check reads the cell each
    run trained (resolved flags, not labels), prints a warning listing the
    missing or unbalanced cells, and stores
    ``<tag>_ablation_coverage.json`` next to the run records.
    """
    present = sorted({c for c in (_calib_cell(r) for r in runs) if c in cells})
    missing = [c for c in cells if c not in present]
    balance, unbalanced = _cell_seed_balance(runs, _calib_cell, cells,
                                             planned_seeds)
    payload = {
        "tag": tag,
        "design": design or "calib_use_len x calib_use_ent (2x2)",
        "expected_cells": list(cells),
        "present_cells": present,
        "missing_cells": missing,
        "complete": not missing,
        "cell_seed_balance": balance,
        "unbalanced_cells": unbalanced,
        "seed_balanced": not unbalanced,
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
    if unbalanced:
        print(f"[{tag}] WARNING: ablation cell(s) {', '.join(unbalanced)} "
              f"did not train the full planned seed set "
              f"{list(planned_seeds or ())}: their means are read on a "
              f"different sample than the other cells'; the factorial is "
              f"unbalanced and those cell effects are confounded with seed "
              f"availability")
    _write_json(out_dir / f"{tag}_ablation_coverage.json", payload)
    return payload


def _report_design_coverage(out_dir, tag, runs, design, extract, expected,
                            planned_seeds=None):
    """Verify a declared factorial design trained every declared cell.

    Generalises the ablation-coverage check to whole experiment grids
    (E1's four gate arms, E2's eight arms, E3's 2x4x2 grid): an effect is
    only attributable if every cell it contrasts was actually trained, and
    a missing arm leaves that effect unmeasured - or worse, invites a
    claim from the cells that happen to have run.  The extractor reads the
    resolved per-run factors (not labels); missing cells and cells that
    trained only part of the planned seed set are printed as warnings and
    stored in ``<tag>_<design>_coverage.json``.
    """
    observed = {extract(r) for r in runs if extract(r) is not None}
    present = [c for c in expected if any(c == o for o in observed)]
    missing = [c for c in expected if not any(c == p for p in present)]
    balance, unbalanced = _cell_seed_balance(runs, extract, expected,
                                             planned_seeds)
    payload = {
        "tag": tag,
        "design": design,
        "expected_cells": [list(c) if isinstance(c, tuple) else c
                           for c in expected],
        "present_cells": [list(c) if isinstance(c, tuple) else c
                          for c in present],
        "missing_cells": [list(c) if isinstance(c, tuple) else c
                          for c in missing],
        "complete": not missing,
        "cell_seed_balance": balance,
        "unbalanced_cells": unbalanced,
        "seed_balanced": not unbalanced,
        "n_runs": len(runs),
    }
    if missing:
        print(f"[{tag}] WARNING: {design} coverage incomplete: "
              f"{len(present)}/{len(expected)} cells trained "
              f"(missing {missing}); those effects are not measured")
    else:
        print(f"[{tag}] design coverage: {len(expected)}/{len(expected)} "
              f"{design} cells trained")
    if unbalanced:
        print(f"[{tag}] WARNING: {design} cell(s) {', '.join(unbalanced)} "
              f"did not train the full planned seed set "
              f"{list(planned_seeds or ())}: the factorial is unbalanced "
              f"and those cell effects are confounded with seed "
              f"availability")
    _write_json(out_dir / f"{tag}_{design}_coverage.json", payload)
    return payload


def _recipe_view(run):
    """The controlled part of a run's configuration (a flat dict).

    Besides the resolved ``Settings`` snapshot this includes the stamped
    ``training_recipe`` block (optimizer family, weight decay, betas,
    schedule, eta_min, gradient clipping, augmentation) when the record
    carries it, so a contrast checks the *actual* optimisation and
    augmentation recipe - not only the learning rate - for equality across
    the two sides.  The stamped evaluation-stream seeds enter the view as
    well: equal stream *sizes* do not prove equal streams, and a metric
    read from a different stream is an evaluation-protocol difference the
    audit must refuse exactly like a recipe difference.  The stamped
    ``data_fingerprints`` (the content hash of every training/validation/
    test/statistics stream) enter the view for the same reason one level
    deeper: the same *seeds* can still generate a different stream if the
    batch geometry or generator path changed, and then the two sides did
    not train or read the same data at all.  Finally, the *executed* step
    count / early-stop state is included: the intended step count in
    ``training_recipe.steps`` does not prove both sides actually ran it.
    """
    cfg = run.get("config") or {}
    view = {k: cfg.get(k) for k in _AUDIT_RECIPE_KEYS}
    view["lr"] = run.get("lr")
    view["eval_seed"] = run.get("eval_seed")
    view["stats_seed"] = run.get("stats_seed")
    # The intended step count lives in ``training_recipe.steps``; these
    # stamped fields are what the run *actually* executed.  Auditing them
    # keeps "same epochs" true even if early stopping is ever enabled, and
    # makes a short/diverged run visible as a confound rather than as a
    # silent extra degree of freedom.
    view["steps_run"] = run.get("steps_run")
    view["stopped_early"] = run.get("stopped_early")
    view["restored_best"] = run.get("restored_best")
    # The data-stream fingerprints ride with the run context; a record that
    # predates the stamp compares as ``None == None`` and is skipped, while
    # stamped records now prove "same seed, same stream" - per training
    # seed (the ``train`` sub-map is keyed by seed, so a stray Settings
    # whose stream geometry changed cannot masquerade as the paired run).
    view["data_fingerprints"] = run.get("data_fingerprints")
    recipe = run.get("training_recipe")
    if isinstance(recipe, dict):
        for key, val in recipe.items():
            view[f"training_recipe.{key}"] = val
    return view


def _training_record(settings: Settings, lr: float,
                     steps: int | None = None) -> dict:
    """The shared optimisation recipe block stamped into every run record."""
    return {"training_recipe": training_recipe(
        lr, settings.steps if steps is None else int(steps))}


def _controlled_audit(runs_a, runs_b):
    """Machine-check the controlled-variable contract of a contrast.

    Comparing method A with method B only attributes a difference to the
    module under test if *everything else* is held fixed.  This audit
    verifies, from the stored run records:

    1. **Paired seeds**: both sides were run on the same seed set, so the
       paired bootstrap differences each seed with itself (fewer / different
       seeds silently shrink the sample and break the pairing).  A seed
       repeated within one side is refused too: it is one run counted
       twice, which inflates the paired n and narrows the CI (new grids
       cannot get here - ``Settings`` / ``_run_context`` validate the seed
       set - but pooled or legacy records can);
    2. **Within-side recipe**: every run on a side used the same learning
       rate, step count, batch geometry, data size and evaluation-stream
       size and seed, the same *actual* ``steps_run`` / early-stop state,
       and - when the record carries the stamped ``training_recipe`` - the
       same optimizer, weight decay, betas, schedule, eta_min, gradient
       clipping and augmentation pipeline (no per-variant optimizer or
       schedule drift, and no run that silently trained for fewer steps);
    3. **Across-side recipe**: the two recipes are identical (the same
       fields as above), so a "gain" that is really a changed optimizer or
       learning-rate schedule is refused instead of reported as a win;
    4. **Single varying factor**: exactly one declared model factor (gate /
       norm / layer_scale / calibration-term or normalisation flag) differs,
       so the contrast is both non-degenerate and attributable.  A row that
       flips two factors at once (e.g. full calib vs the arm with *both*
       additive terms removed) is refused here: its effect is the sum of two
       manipulations, and it belongs in a registered factorial main effect
       or interaction rather than in a single-factor decision row.  Records
       that predate the resolved ``model_cfg`` stamp fall back to the
       declared factor labels;
    5. **Trained-model ground truth**: when both sides carry the resolved
       ``model_cfg``, every non-factor config key must match across the
       sides (an undeclared architectural difference is a confound even
       with matching recipe), and the declared labels must agree with the
       flags the model was actually built with - in particular a
       ``calib_variant`` label must name the (calib_use_len, calib_use_ent)
       cell that trained, so a mis-wired ablation arm cannot pass as the
       intended condition.

    The actual parameter count each side trained is recorded as well.  A
    differing count is legitimate across gates (``calib`` carries the
    calibration terms, ``none`` does not), so it is not a mismatch by
    itself - but where the design's *defining claim* is matched capacity
    it is enforced: the H1f active control differs from full ``calib``
    only in ``calib_shuffle_ent`` and is registered as the "identical
    calib architecture, parameter count, computation and optimiser"
    placebo, so a capacity difference there would silently confound the
    specificity test and is refused.  A side that trained more than one
    parameter count is refused as a build inconsistency whatever the
    design.

    The optimizer factory, cosine schedule, clipping and gradient settings
    are fixed in the single training entry point (``train.train_model``) and
    are therefore not per-run fields; ``lr`` plus ``steps`` pin the schedule
    that factory is combined with.  Returns an audit dict; a contrast whose
    audit fails is not read as supporting its hypothesis.
    """
    def _seed_key(seed):
        if seed is None:
            return (1, 0.0, "")
        if isinstance(seed, (int, float)) and not isinstance(seed, bool):
            return (0, float(seed), "")
        return (0, float("inf"), str(seed))

    def _stream_key(seed):
        """Canonical data-stream key used to detect aliased seeds.

        ``0`` and ``2**32`` draw the same CPU stream (see
        ``cage.data.stream_seed_key``), so a pooled record carrying both is
        one independent run counted twice even though the raw seed values
        differ.  Non-numeric legacy seeds fall back to their string form.
        """
        if seed is None:
            return None
        if isinstance(seed, bool):
            return ("raw", repr(seed))
        try:
            return ("stream", stream_seed_key(seed))
        except (TypeError, ValueError, OverflowError):
            return ("raw", str(seed))

    audit = {
        "controlled": True,
        "seeds_paired": False,
        "n_a": len(runs_a),
        "n_b": len(runs_b),
        "seeds_a": sorted((r.get("seed") for r in runs_a), key=_seed_key),
        "seeds_b": sorted((r.get("seed") for r in runs_b), key=_seed_key),
        "varying_factors": {},
        "model_factor_diffs": {},
        "n_varying_factors": None,
        "single_factor": None,
        "mismatches": [],
    }
    audit["seeds_paired"] = audit["seeds_a"] == audit["seeds_b"]
    if not audit["seeds_paired"]:
        audit["mismatches"].append(
            f"seed sets differ: a={audit['seeds_a']} b={audit['seeds_b']} "
            f"(contrast must be paired over identical seeds)")
    for side, seeds in (("a", audit["seeds_a"]), ("b", audit["seeds_b"])):
        missing = sum(1 for s in seeds if s is None)
        if missing:
            audit["mismatches"].append(
                f"side {side} has {missing} run(s) without a training seed: "
                f"the contrast cannot be paired")
        stream_counts = {}
        for s in seeds:
            if s is None:
                continue
            key = _stream_key(s)
            stream_counts[key] = stream_counts.get(key, 0) + 1
        repeats = sorted({s for s in seeds
                          if s is not None
                          and stream_counts[_stream_key(s)] > 1},
                         key=_seed_key)
        if repeats:
            audit["mismatches"].append(
                f"side {side} repeats training seed/stream(s) {repeats}: a "
                f"repeated seed (including a 32-bit stream alias such as "
                f"0 vs 2**32) is one run counted twice, so the paired n "
                f"would be inflated and the difference is not independent")
    # Capacity: the parameter count each side actually trained (recorded for
    # every contrast; the placebo-specific enforcement happens below, where
    # the resolved model configs have been compared).
    params_a = sorted({int(r["params"]) for r in runs_a
                       if _finite(r.get("params"))})
    params_b = sorted({int(r["params"]) for r in runs_b
                       if _finite(r.get("params"))})
    audit["params"] = {"a": params_a, "b": params_b}
    for side, counts in (("a", params_a), ("b", params_b)):
        if len(counts) > 1:
            audit["mismatches"].append(
                f"side {side} trained {len(counts)} distinct parameter "
                f"counts {counts}: the side is not one fixed model")
    if len(params_a) == 1 and len(params_b) == 1:
        audit["capacity"] = {
            "a": params_a[0], "b": params_b[0],
            "delta_b_minus_a": params_b[0] - params_a[0],
            "matched": params_a[0] == params_b[0],
        }
    for side, runs in (("a", runs_a), ("b", runs_b)):
        views = [_recipe_view(r) for r in runs]
        if not views:
            audit["mismatches"].append(f"side {side} has no runs")
            continue
        first = views[0]
        for j, other in enumerate(views[1:], start=1):
            for key in sorted(set(first) | set(other)):
                if other.get(key) != first.get(key):
                    audit["mismatches"].append(
                        f"side {side} recipe differs between runs 0 and {j}: "
                        f"{key}={first.get(key)!r} vs {other.get(key)!r}")
    if runs_a and runs_b:
        va, vb = _recipe_view(runs_a[0]), _recipe_view(runs_b[0])
        for key in sorted(set(va) | set(vb)):
            if vb.get(key) != va.get(key):
                audit["mismatches"].append(
                    f"recipe differs across sides: {key}={va.get(key)!r} vs "
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
                    cfg = r.get("model_cfg") or {}
                    # The top-level design labels are what a reader sees in
                    # the table; they must name the model that actually
                    # trained.  Checking only ``calib_variant`` would let a
                    # run labelled gate="none" carry a query/calib config,
                    # making an identical-config or mislabelled contrast
                    # look controlled.
                    for key in ("gate", "norm", "layer_scale"):
                        if key not in r:
                            continue
                        label = r.get(key)
                        trained = cfg.get(key)
                        if label != trained:
                            audit["mismatches"].append(
                                f"side {side}: {key} label {label!r} does "
                                f"not match the trained model_cfg "
                                f"({trained!r})")
                    label = r.get("calib_variant")
                    if not label:
                        continue
                    if cfg.get("gate") != "calib":
                        audit["mismatches"].append(
                            f"side {side}: calib_variant={label!r} on a "
                            f"gate={cfg.get('gate')!r} run")
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
            # H1f's active control is registered as the identical calib
            # architecture / parameter count / computation, differing only
            # in the shuffled entropy signal.  When the resolved configs
            # say that is the *only* difference, matched capacity is a
            # defining property of the contrast, so a differing count is a
            # confound (capacity, not entropy information) and the audit
            # refuses the contrast instead of letting the specificity test
            # read as if the placebo were capacity-matched.
            capacity = audit.get("capacity")
            if capacity and not capacity["matched"] \
                    and set(audit["model_factor_diffs"]) == \
                    {"calib_shuffle_ent"}:
                audit["mismatches"].append(
                    f"placebo capacity mismatch: the only resolved "
                    f"model-config difference is calib_shuffle_ent, whose "
                    f"defining claim is the identical calib architecture "
                    f"and parameter count, but the sides trained "
                    f"{capacity['a']} vs {capacity['b']} parameters - the "
                    f"capacity would be confounded with the shuffled "
                    f"entropy signal")
        # Single-factor contract (see point 4 above).  The resolved config
        # is ground truth when both sides carry it; otherwise the declared
        # labels are the only evidence available.  ``calib_variant`` is a
        # *cell* label, not an independent factor: counting it alongside the
        # resolved flags would make an ordinary one-flag ablation look like
        # a two-factor contrast, so only the resolved diffs are counted when
        # the model config is present.
        if audit.get("model_cfg") is not None:
            factor_keys = set(audit["model_factor_diffs"])
        else:
            factor_keys = set(audit["varying_factors"])
            # Expand a label-only ``calib_variant`` difference into the
            # factor vector it names, so a legacy record cannot pass a
            # two-term manipulation (e.g. full vs both-terms-off) as one
            # factor merely because only the cell label was stamped.
            if "calib_variant" in factor_keys:
                fa = [_CALIB_CELL_FACTORS.get(r.get("calib_variant"))
                      for r in runs_a]
                fb = [_CALIB_CELL_FACTORS.get(r.get("calib_variant"))
                      for r in runs_b]
                if all(x is not None for x in fa + fb):
                    factor_keys.discard("calib_variant")
                    audit["varying_factors"].pop("calib_variant", None)
                    for key in ("calib_use_len", "calib_use_ent",
                                "calib_shuffle_ent", "calib_ent_norm"):
                        va = {x[key] for x in fa}
                        vb = {x[key] for x in fb}
                        if va != vb:
                            factor_keys.add(key)
                            audit["varying_factors"][key] = {
                                "a": sorted(va), "b": sorted(vb)}
        audit["n_varying_factors"] = len(factor_keys)
        audit["single_factor"] = len(factor_keys) == 1
        if not factor_keys:
            if audit.get("model_cfg") is not None:
                # A stale declared label difference does not make an
                # identical-config pair a controlled contrast.  This check
                # is separate from the label-only fallback because
                # ``varying_factors`` still contains the declared labels
                # even when the resolved configs are identical.
                audit["mismatches"].append(
                    "the resolved model config is identical on both sides "
                    "(no model factor differs), so the declared label "
                    "difference does not identify a controlled manipulation")
            else:
                audit["mismatches"].append(
                    "no declared model factor (gate/norm/layer_scale/"
                    "calib_variant) differs across the sides: degenerate or "
                    "mislabeled contrast")
        elif len(factor_keys) > 1:
            audit["mismatches"].append(
                f"single-factor contract: the contrast varies "
                f"{len(factor_keys)} model factor(s) "
                f"{sorted(factor_keys)}; a controlled contrast must vary "
                f"exactly one factor, so a multi-factor manipulation has to "
                f"be read through a registered factorial main effect or "
                f"interaction instead")
    audit["controlled"] = not audit["mismatches"]
    return audit


def _paired_stream_test(by_a, by_b, used, metric, seed_mean):
    """Exact paired test on the shared finite evaluation stream.

    Training seeds are independent, but the evaluation items are not: every
    seed is scored on the same sealed stream.  The seed-level bootstrap
    answers "would this effect replicate over training runs on this stream",
    not "is the stream large enough to resolve it".  When both sides store
    the compact per-item correctness mask from ``eval_metrics``, this helper
    runs McNemar's exact test on the concatenated items and reports the
    item-level effect, so a positive seed-level mean whose item-level sign
    disagrees is caught instead of being printed as a win.

    That pooled exact p is only descriptive: concatenating seeds repeats the
    same sealed items, so the pooled item table violates McNemar's
    independence assumption and its p is anti-conservative.  The per-seed
    discordant counts, the seed-level sign test over the independent training
    seeds, and the consistency of the per-seed item effects are therefore
    reported alongside it; only the seed-level summaries may corroborate a
    claim.  The pooled p is retained for continuity and explicitly labelled
    descriptive.

    Only ``accuracy`` has a stored mask.  Any missing mask, differing item
    count, or malformed record makes the companion not applicable (it is a
    diagnostic, never a reason to fabricate a test).
    """
    out = {"applicable": False, "reason": None, "n": 0,
           "n_seeds": 0, "b": None, "c": None, "p_exact": None,
           "item_mean_diff": None, "direction_agrees": None,
           "pooled_descriptive": True, "per_seed": [],
           "seed_sign_positive": 0, "seed_sign_negative": 0,
           "seed_sign_zero": 0, "seed_sign_test_p": None,
           "seed_sign_consistent_with_mean": None,
           "per_seed_direction_agreement": None}
    if metric != "accuracy":
        out["reason"] = "only accuracy stores per-item correctness"
        return out
    masks_a, masks_b, ns = [], [], []
    for seed in used:
        ra, rb = by_a.get(seed), by_b.get(seed)
        if ra is None or rb is None:
            out["reason"] = f"missing run for seed {seed}"
            return out
        ea, eb = ra.get("correct_bitmask"), rb.get("correct_bitmask")
        na = _positive_integer(ra.get("n", ra.get("eval_n")))
        nb = _positive_integer(rb.get("n", rb.get("eval_n")))
        if not isinstance(ea, str) or not isinstance(eb, str):
            out["reason"] = "per-item correctness not recorded on both sides"
            return out
        if na is None or nb is None:
            out["reason"] = "evaluation item counts are not positive integers"
            return out
        if na != nb:
            out["reason"] = "evaluation item counts differ"
            return out
        n = na
        try:
            ma = decode_correct_mask(ea, n)
            mb = decode_correct_mask(eb, n)
        except (TypeError, ValueError) as exc:
            out["reason"] = f"invalid correctness mask: {exc}"
            return out
        masks_a.append(ma)
        masks_b.append(mb)
        ns.append(n)
    if not masks_a:
        out["reason"] = "no paired seeds"
        return out
    if len(set(ns)) != 1:
        # Every training seed is scored on the same sealed stream, so one
        # seed with a different item count is a protocol mismatch.  Pooling
        # the masks would still concatenate, but the per-seed and pooled
        # discordant tables would describe different samples.
        out["reason"] = (
            f"evaluation item counts differ across paired seeds: "
            f"{sorted(set(ns))}")
        return out
    ca = torch.cat(masks_a)
    cb = torch.cat(masks_b)
    if ca.numel() != cb.numel() or ca.numel() == 0:
        out["reason"] = "no evaluation items"
        return out
    per_seed = []
    for seed, ma, mb, n in zip(used, masks_a, masks_b, ns):
        mc_seed = mcnemar_test(ma, mb)
        diff_seed = float(ma.float().mean() - mb.float().mean())
        per_seed.append({
            "seed": seed, "n": int(n),
            "b": int(mc_seed["b"]), "c": int(mc_seed["c"]),
            "item_mean_diff": diff_seed,
            "p_exact": mc_seed["p_exact"],
            "signed_direction": (1 if diff_seed > 0.0 else
                                 -1 if diff_seed < 0.0 else 0),
        })
    mc = mcnemar_test(ca, cb)
    item_mean = float(ca.float().mean() - cb.float().mean())
    if _finite(seed_mean) and _finite(item_mean):
        agrees = ((seed_mean > 0.0 and item_mean > 0.0)
                  or (seed_mean < 0.0 and item_mean < 0.0)
                  or (seed_mean == 0.0 and item_mean == 0.0))
    else:
        agrees = None
    directions = [row["signed_direction"] for row in per_seed]
    nonzero = [d for d in directions if d != 0]
    pos = sum(1 for d in directions if d > 0)
    neg = sum(1 for d in directions if d < 0)
    zero = sum(1 for d in directions if d == 0)
    overall_dir = (1 if item_mean > 0.0 else -1 if item_mean < 0.0 else 0)
    if overall_dir and nonzero:
        agreement = sum(1 for d in nonzero if d == overall_dir) / len(nonzero)
        consistent = all(d == overall_dir for d in nonzero)
    else:
        agreement = None
        consistent = None
    if nonzero:
        k = max(pos, neg)
        tail = sum(math.comb(len(nonzero), i)
                   for i in range(k, len(nonzero) + 1))
        sign_p = min(1.0, 2.0 * tail / (2 ** len(nonzero)))
    else:
        sign_p = None
    out.update({
        "applicable": True, "n": int(ca.numel()),
        "n_seeds": len(masks_a), "n_per_seed": sorted(set(ns)),
        "b": mc["b"], "c": mc["c"],
        "p_exact": mc["p_exact"], "p_chi2_cc": mc["p_chi2_cc"],
        "item_mean_diff": item_mean, "direction_agrees": agrees,
        "pooled_descriptive": True,
        "pooled_note": (
            "the pooled McNemar p concatenates every seed's evaluation of "
            "the same sealed items, so repeated item evaluations are not "
            "independent and this p must not be read as a confirmation or "
            "a claim gate; use the per-seed counts and the independent-seed "
            "sign test"),
        "per_seed": per_seed,
        "seed_sign_positive": int(pos), "seed_sign_negative": int(neg),
        "seed_sign_zero": int(zero), "seed_sign_test_p": sign_p,
        "seed_sign_consistent_with_mean": consistent,
        "per_seed_direction_agreement": agreement,
    })
    return out


def _paired_diff(runs_a, runs_b, metric):
    """Paired contrast ``a - b`` plus both arms' levels.

    ``baseline_mean`` (the mean of ``metric`` on side ``b``) is attached so
    the report can print the effect as a relative change: a contrast can be
    statistically significant with zero variance yet practically
    negligible, and the relative effect is what exposes that.  The full
    per-arm mean +- std (and the seed count each mean is read on) travel
    with the contrast too, so the decision table answers the pre-registered
    "did the effect exceed the seed-to-seed spread?" question directly
    instead of only through ``dz``/MDE80 - both arms are reported on the
    paired sample (the seeds where both sides carry a finite value), so the
    printed levels, the effect and ``rel_effect`` can never describe
    different seed sets.  ``arms_all`` keeps the full-run levels for
    transparency, but only the paired ``arms`` feed the decision table.

    The controlled-variable audit (``_controlled_audit``) is attached as
    ``controlled``: a difference between recipes (optimizer settings, step
    count, data geometry, evaluation streams, unpaired seeds) would
    confound the contrast, so the claim is refused downstream instead of
    being reported as a win.

    Seed accounting: only seeds where both sides carry a finite value for
    this metric enter the paired difference.  ``seeds_used`` records them;
    ``seeds_dropped`` records the rest, split into ``seeds_unpaired``
    (value present on only one side - a pairing failure) and
    ``seeds_undefined`` (metric undefined on both sides, e.g. no
    false-concentration cases), so a shrunken sample is never silently
    read as if it were the full seed set.
    """
    missing_a = [r for r in runs_a if r.get("seed") is None]
    missing_b = [r for r in runs_b if r.get("seed") is None]
    by_a = {r["seed"]: r for r in runs_a if r.get("seed") is not None}
    by_seed = {r["seed"]: r for r in runs_b if r.get("seed") is not None}
    diffs = []
    used, dropped, unpaired, undefined = [], [], [], []
    # Iterate the *union* of the two seed sets: a seed present only on one
    # side is a pairing failure and must be recorded as dropped/unpaired,
    # not silently absent from the accounting just because it happens to
    # be missing on side a.
    planned = sorted(set(by_a) | set(by_seed), key=_seed_order)
    for seed in planned:
        ra, rb = by_a.get(seed), by_seed.get(seed)
        va = None if ra is None else ra.get(metric)
        vb = None if rb is None else rb.get(metric)
        finite_a, finite_b = _finite(va), _finite(vb)
        if ra is not None and rb is not None and finite_a and finite_b:
            used.append(seed)
            diffs.append(float(va) - float(vb))
        elif ra is None or rb is None or finite_a != finite_b:
            dropped.append(seed)
            unpaired.append(seed)
        else:
            # Metric undefined on both sides (e.g. no false-concentration
            # cases at this scale): not a pairing failure, but it still
            # shrinks the effective sample and must be visible.
            dropped.append(seed)
            undefined.append(seed)
    if missing_a or missing_b:
        n_missing = len(missing_a) + len(missing_b)
        dropped.extend([None] * n_missing)
        unpaired.extend([None] * n_missing)
    ci = paired_bootstrap_ci(diffs)
    # Levels are reported on exactly the seeds that entered the paired
    # difference.  Using all runs here would let a metric-undefined or
    # unpaired seed contribute to an arm mean but not to the effect (or
    # vice versa), making the printed levels and the contrast describe
    # different samples.  The full-run levels remain available as
    # ``arms_all`` for inspection.
    if used:
        side_a = _mean_std([by_a[s][metric] for s in used])
        base = _mean_std([by_seed[s][metric] for s in used])
    else:
        side_a = _mean_std([])
        base = _mean_std([])
    ci["arms"] = {"a": side_a, "b": base}
    ci["arms_all"] = {
        "a": _mean_std([r[metric] for r in runs_a if metric in r]),
        "b": _mean_std([r[metric] for r in runs_b if metric in r]),
    }
    ci["arms_note"] = "paired seeds used by the effect"
    ci["baseline_mean"] = base["mean"]
    # The shared finite evaluation stream is a second source of noise on
    # top of training-seed variance: every seed is scored on the same
    # items.
    # Run the item-level paired test (McNemar) whenever both sides
    # recorded their per-item correctness.  The pooled p is descriptive
    # only: concatenating seeds repeats the same sealed items, so it cannot
    # replace the seed-level decision.  Per-seed discordant counts and an
    # independent-seed sign test travel alongside it, and the pooled item
    # sign still catches a gain that is only an artifact of the stream.
    ci["stream_test"] = _paired_stream_test(
        by_a, by_seed, used, metric, ci.get("mean"))
    # Raw per-seed differences: lets an equivalence companion reuse the
    # identical pairing instead of re-deriving it (and drifting from it).
    ci["diffs"] = list(diffs)
    ci["curve_invalid_seeds"] = _curve_invalid_seeds_for(
        used, (by_a, by_seed))
    # The paired sample is only the seeds where *both* sides carry a
    # finite value for this metric: recording which seeds actually entered
    # makes a silently shrunken sample visible (and the finalize layer
    # refuses claims below the pre-registered minimum seed floor).
    ci["seeds_used"] = sorted(used, key=_seed_order)
    ci["seeds_dropped"] = sorted(dropped, key=_seed_order)
    ci["seeds_unpaired"] = sorted(unpaired, key=_seed_order)
    ci["seeds_undefined"] = sorted(undefined, key=_seed_order)
    ci["controlled"] = _controlled_audit(runs_a, runs_b)
    return ci


def _paired_benchmark(runs_a, runs_b, metric, sesoi, reference,
                      *, comparison="benchmark", treatment=None):
    """Paired tuned-baseline benchmark (not a controlled mechanism contrast).

    The whole point of a complexity comparison is that the two sides differ
    in architecture and each was tuned independently, so the controlled-
    recipe audit does not apply and the row must never be routed through
    :class:`ContrastLog` (which would refuse it as confounded).  The paired
    seed differences, CI and SESOI are still the right effect-size readout:
    ``complexity_justified`` means the arm is both practically above and
    statistically above the tuned reference, while ``below_reference`` means
    the reference is confidently better.  Curve-invalid runs on either side
    are excluded before pairing (they cannot support a depth claim), and the
    excluded seeds are recorded under ``curve_invalid_excluded``.  The result
    carries an explicit ``comparison`` tag and a summary of the (expected)
    recipe mismatches, never the full audit, so it cannot be mistaken for a
    controlled contrast.

    ``comparison``/``treatment`` let a caller distinguish the other legitimate
    benchmark family: the E1 per-variant fairness account, where the arms
    keep their architecture but each carries its *own* val-selected lr (and
    calib_tau_init).  The audit still fails by design, so the label must say
    which treatment was varied; the pre-registered same-recipe rows remain a
    separate accounting.
    """
    excluded = {}
    for side, runs in (("a", runs_a), ("b", runs_b)):
        bad = [r.get("seed") for r in runs
               if r.get("curve_verdict") in CURVE_INVALID_VERDICTS]
        if bad:
            excluded[side] = sorted(bad, key=_seed_order)
    ci = _paired_diff(_curve_valid_runs(runs_a), _curve_valid_runs(runs_b),
                      metric)
    ci["curve_invalid_excluded"] = excluded
    audit = ci.get("controlled") or {}
    ci["comparison"] = str(comparison)
    ci["treatment"] = (str(treatment) if treatment else
                       "architecture/capacity and the val-tuned recipe")
    ci["reference"] = str(reference)
    ci["sesoi"] = float(sesoi)
    # The audit is *expected* to fail (the point is a different architecture
    # and/or a separately tuned recipe), so the row is marked as
    # intentionally non-controlled instead of carrying a wall of recipe
    # mismatches into the summary; only their count is kept for traceability.
    ci["controlled"] = None
    ci["controlled_audit"] = {
        "applicable": False,
        "reason": (f"{ci['comparison']} deliberately varies "
                   f"{ci['treatment']}"),
        "observed_mismatch_count": len(audit.get("mismatches") or []),
    }
    mean, lo, hi = ci.get("mean"), ci.get("lo"), ci.get("hi")
    ci["complexity_justified"] = bool(
        _finite(mean) and _finite(lo)
        and float(mean) >= float(sesoi) and float(lo) > 0.0)
    ci["below_reference"] = bool(_finite(hi) and float(hi) < 0.0)
    # "Indistinguishable" is an equivalence-style read: it requires a
    # measured paired effect whose CI lies inside the pre-registered
    # practical bound.  A wide interval that merely includes 0 is
    # inconclusive, not evidence of no difference; and when every seed was
    # excluded as curve-invalid (or no seed could be paired) there is no
    # evidence at all.
    ci["unmeasured"] = not (ci.get("seeds_used") and _finite(mean))
    ci["indistinguishable"] = bool(
        not ci["unmeasured"] and _finite(lo) and _finite(hi)
        and math.isfinite(float(sesoi))
        and float(lo) > -float(sesoi) and float(hi) < float(sesoi))
    ci["inconclusive"] = bool(
        not ci["unmeasured"] and not ci["complexity_justified"]
        and not ci["below_reference"] and not ci["indistinguishable"])
    return ci


def _one_sample_contrast(runs, metric, bound=None):
    """One-sample contrast against zero, with curve-invalid seed metadata.

    Several pre-registered readouts (the null-arm error margin, the entropy
    and gate drifts) are one-sample claims: the per-seed value is compared
    with 0, not with another arm.  They still need the same seed accounting
    and curve-invalid sensitivity as a paired contrast - a drift computed
    on a run whose curve never learned cannot support a mechanism claim -
    so this helper attaches ``diffs``/``seeds_used``/``curve_invalid_seeds``
    in the form :meth:`ContrastLog.finalize` already consumes.  ``bound``
    selects the bootstrap-TOST equivalence test instead of the directional
    CI (the return value then carries ``test='equivalence'`` from
    :func:`equivalence_test`).  Runs without a training seed cannot be
    independent units; they are dropped and recorded like an unpaired seed.
    """
    missing = [r for r in runs if r.get("seed") is None]
    by_seed = {r["seed"]: r for r in runs if r.get("seed") is not None}
    used, values = [], []
    for seed in sorted(by_seed, key=_seed_order):
        value = by_seed[seed].get(metric)
        if _finite(value):
            used.append(seed)
            values.append(float(value))
    ci = (equivalence_test(values, bound) if bound is not None
          else paired_bootstrap_ci(values))
    dropped = sorted(set(by_seed) - set(used), key=_seed_order)
    ci["diffs"] = list(values)
    ci["seeds_used"] = used
    ci["seeds_dropped"] = dropped + [None] * len(missing)
    ci["seeds_unpaired"] = [None] * len(missing)
    ci["seeds_undefined"] = dropped
    ci["curve_invalid_seeds"] = _curve_invalid_seeds_for(used, (by_seed,))
    return ci


def _interaction_manipulation(audit: dict) -> dict:
    """The factor a controlled contrast actually varies, in comparable form.

    ``_controlled_audit`` reports two descriptions: ``varying_factors``
    from the declared labels (always present) and ``model_factor_diffs``
    from the resolved ``model_cfg`` (present when the records carry it).
    The resolved one is ground truth for *which* factor a contrast varies,
    but the label description mixes genuine factors with the
    ``calib_variant`` cell label: two legitimate interaction marginals like
    full-vs-ent_only and len_only-vs-no_terms manipulate the same
    ``calib_use_len`` flag while carrying different cell labels, so
    comparing the label sets would refuse a valid factorial contrast.
    Prefer the resolved diffs; fall back to the labels only for records
    that predate ``model_cfg``.
    """
    resolved = audit.get("model_factor_diffs") or {}
    if resolved:
        return {key: value for key, value in resolved.items()}
    return dict(audit.get("varying_factors") or {})


_CLASS_METRIC_COMPANIONS = (
    ("recall_macro",
     "balanced accuracy: unweighted mean per-class recall, target-present "
     "classes"),
    ("f1_macro",
     "macro F1: unweighted mean per-class F1, target-present classes "
     "(micro F1 equals accuracy)"),
    ("f1_weighted",
     "support-weighted F1, target-present classes"),
)


def _class_frequency_companions(rules, title):
    """Re-read accuracy contrasts on class-frequency-robust metrics.

    Accuracy is only prior-invariant while the answer stream is balanced.
    The E1/E3 runners audit that premise and gate claims on it, but the
    *decision* rows are still accuracy-only: a stable accuracy gain can
    hide a macro-F1 loss when one answer class absorbs the errors.  These
    companions keep the identical seed pairing and hierarchy (seed-level
    paired bootstrap of ``a - b``) and explicitly label the averaging
    convention for each metric.  They are exploratory robustness readouts,
    not additional pre-registered decisions, so they are reported outside
    the FDR family and never overwrite the registered verdict.
    """
    rows = []
    print(f"\n[{title}] class-frequency robustness companions "
          f"(exploratory, outside the FDR family; seed-paired a - b):")
    print(f"  {'contrast':<34} {'metric':<15} {'effect [95% CI]':>29} "
          f"{'n':>3}  {'vs accuracy':>18}")
    for label, runs_a, runs_b, hypothesis in rules:
        acc = _paired_diff(runs_a, runs_b, "accuracy")
        for metric, convention in _CLASS_METRIC_COMPANIONS:
            ci = _paired_diff(runs_a, runs_b, metric)
            sign_disagrees = None
            uncertainty_disagrees = None
            if (_finite(acc.get("mean")) and _finite(ci.get("mean"))
                    and acc["mean"] != 0.0 and ci["mean"] != 0.0):
                sign_disagrees = ((acc["mean"] > 0.0)
                                  != (ci["mean"] > 0.0))
            if ci.get("n") and acc.get("n"):
                acc_sig = bool(_finite(acc.get("lo"))
                               and _finite(acc.get("hi"))
                               and (acc["lo"] > 0.0 or acc["hi"] < 0.0))
                met_sig = bool(_finite(ci.get("lo"))
                               and _finite(ci.get("hi"))
                               and (ci["lo"] > 0.0 or ci["hi"] < 0.0))
                uncertainty_disagrees = (acc_sig != met_sig)
            if sign_disagrees:
                note = "sign flips"
            elif uncertainty_disagrees:
                note = "CI decisive differs"
            else:
                note = "consistent"
            effect = (f"{ci['mean']:+.4f} [{ci['lo']:+.4f}, "
                      f"{ci['hi']:+.4f}]"
                      if (_finite(ci.get("mean")) and _finite(ci.get("lo"))
                          and _finite(ci.get("hi"))) else "n/a")
            print(f"  {label:<34} {metric:<15} {effect:>29} "
                  f"{ci.get('n', 0):>3}  {note:>18}")
            rows.append({
                "contrast": label,
                "hypothesis": hypothesis,
                "metric": metric,
                "convention": convention,
                "effect": ci.get("mean"),
                "lo": ci.get("lo"),
                "hi": ci.get("hi"),
                "n": ci.get("n", 0),
                "seeds_used": ci.get("seeds_used", []),
                "baseline_mean": ci.get("baseline_mean"),
                "accuracy_mean": acc.get("mean"),
                "accuracy_lo": acc.get("lo"),
                "accuracy_hi": acc.get("hi"),
                "accuracy_n": acc.get("n", 0),
                "sign_disagrees_with_accuracy": sign_disagrees,
                "uncertainty_disagrees_with_accuracy": (
                    uncertainty_disagrees),
            })
    return {
        "title": title,
        "outside_fdr_family": True,
        "note": ("same seed pairing and paired bootstrap as the registered "
                 "accuracy row; macro metrics exclude answer classes with "
                 "zero target support, and f1_weighted weights by support. "
                 "These rows are robustness companions, not extra "
                 "pre-registered decisions."),
        "rows": rows,
    }


def _paired_did(runs_a1, runs_b1, runs_a2, runs_b2, metric):
    """Paired difference-in-differences ``(a1 - b1) - (a2 - b2)``.

    Tests an *interaction* claim - e.g. "the calibrated gate's advantage
    over the query-only gate is specific to the dynamic-normalisation arm"
    - rather than a marginal effect.  Both differences are taken on the
    same training seed, so between-seed variance (the dominant term at
    small seed counts) cancels exactly as in a paired contrast.

    The controlled audit is the conjunction of the four marginal audits:
    within each arm the recipe is fixed and only the intended factor
    varies, and all four cells share one recipe and one seed set.  Both
    marginal contrasts must vary the *same* resolved factor with the same
    level sets, and both cross-context edges (a1 vs a2, b1 vs b2) must
    vary the same context factor, so the DiD cannot difference two
    different manipulations; the marginal and context factors must also be
    different, otherwise one factor varies on both axes and the row is not
    a factorial interaction (see :func:`_interaction_manipulation` for why
    the resolved ``model_cfg`` diff, not the cell label, is compared).  A
    contrast whose audit fails is refused downstream exactly like a
    marginal one, and the effective paired seed set is recorded.  ``arms``
    carries each marginal difference (a = a1-b1, b = a2-b2) with its mean
    +- std over the interaction sample, so the row shows the levels it
    differences; ``baseline_mean`` is the control marginal (b = a2-b2),
    which is what the interaction modifies.  Runs without a training seed
    cannot be paired; they are dropped and recorded as unpaired.
    """
    def _by_seed(rs):
        return {r["seed"]: r for r in rs if r.get("seed") is not None}

    missing = [r for rs in (runs_a1, runs_b1, runs_a2, runs_b2)
               for r in rs if r.get("seed") is None]
    m1a, m1b, m2a, m2b = (_by_seed(rs)
                          for rs in (runs_a1, runs_b1, runs_a2, runs_b2))
    planned = sorted(set(m1a) | set(m1b) | set(m2a) | set(m2b),
                     key=_seed_order)
    diffs, used, dropped, unpaired, undefined = [], [], [], [], []
    for seed in planned:
        cells = [m.get(seed) for m in (m1a, m1b, m2a, m2b)]
        if any(c is None for c in cells):
            dropped.append(seed)
            unpaired.append(seed)
            continue
        vals = [c.get(metric) for c in cells]
        finite = [_finite(v) for v in vals]
        if all(finite):
            used.append(seed)
            diffs.append((float(vals[0]) - float(vals[1]))
                         - (float(vals[2]) - float(vals[3])))
        elif not any(finite):
            dropped.append(seed)
            undefined.append(seed)
        else:
            dropped.append(seed)
            unpaired.append(seed)
    ci = paired_bootstrap_ci(diffs)
    # The two arms of an interaction are the two *marginal differences*
    # (a1-b1 and a2-b2), each summarised with mean +- std over the same
    # four-cell paired seeds the interaction itself uses.  Restricting to
    # ``used`` (not ``planned``) keeps the printed marginal levels on the
    # interaction sample: a seed missing one cell must not contribute to
    # an arm but not to the DiD.
    arms = {}
    for name, (ra, rb) in (("a", (runs_a1, runs_b1)), ("b", (runs_a2, runs_b2))):
        ma, mb = _by_seed(ra), _by_seed(rb)
        ds = [float(ma[s][metric]) - float(mb[s][metric]) for s in used
              if s in ma and s in mb
              and _finite(ma[s].get(metric)) and _finite(mb[s].get(metric))]
        arms[name] = _mean_std(ds)
    ci["arms"] = arms
    ci["arms_note"] = ("marginal differences (a1-b1 vs a2-b2) on the "
                       "interaction sample")
    # The interaction modifies the second marginal effect (a2-b2), so
    # ``rel_effect`` must be relative to that control marginal - not to
    # the difference between the two b-levels, which is a different
    # quantity and can be near zero even when the interaction is real.
    ci["baseline_mean"] = arms["b"]["mean"]
    ci["diffs"] = list(diffs)
    ci["curve_invalid_seeds"] = _curve_invalid_seeds_for(
        used, (m1a, m1b, m2a, m2b))
    ci["seeds_used"] = sorted(used, key=_seed_order)
    ci["seeds_dropped"] = (sorted(dropped, key=_seed_order)
                           + [None] * len(missing))
    ci["seeds_unpaired"] = (sorted(unpaired, key=_seed_order)
                            + [None] * len(missing))
    ci["seeds_undefined"] = sorted(undefined, key=_seed_order)
    audits = [_controlled_audit(runs_a1, runs_b1),
              _controlled_audit(runs_a2, runs_b2),
              _controlled_audit(runs_a1, runs_a2),
              _controlled_audit(runs_b1, runs_b2)]
    mismatches = [msg for audit in audits
                  for msg in audit.get("mismatches", [])]
    # A difference-in-differences is only an interaction of *the same*
    # manipulation measured under two contexts: both marginal contrasts
    # must vary the same declared factor with the same level sets (e.g.
    # calib - query under seednorm and under pre).  Otherwise the DiD
    # mixes two different manipulations and cannot be read as one
    # interaction.
    marginal_factors = [_interaction_manipulation(a) for a in audits[:2]]
    if marginal_factors[0] != marginal_factors[1]:
        mismatches.append(
            "interaction marginals vary different factor(s): "
            f"{sorted(marginal_factors[0])} vs "
            f"{sorted(marginal_factors[1])}")
    # The cross-context edges (a1 vs a2 and b1 vs b2) are what the
    # interaction conditions on: both must vary the *same* context factor
    # with the same level sets, otherwise the DiD differences two different
    # context manipulations.  A marginal factor reappearing on the context
    # axis is refused too: then the design varies one factor on both axes
    # and the "interaction" is not a factorial contrast.
    context_factors = [_interaction_manipulation(a) for a in audits[2:]]
    if context_factors[0] != context_factors[1]:
        mismatches.append(
            "interaction contexts vary different factor(s): "
            f"{sorted(context_factors[0])} vs "
            f"{sorted(context_factors[1])}")
    overlap = sorted(set(marginal_factors[0]) & set(context_factors[0]))
    if overlap:
        mismatches.append(
            f"interaction marginal and context edges vary the same "
            f"factor(s) {overlap}: a factorial interaction needs two "
            f"different factors")
    factors = {}
    for audit in audits:
        factors.update(audit.get("varying_factors", {}))
    ci["controlled"] = {"controlled": not mismatches,
                        "mismatches": mismatches,
                        "varying_factors": factors}
    return ci


def _paired_equivalence(runs_a, runs_b, metric, bound, mode="equivalence"):
    """Paired bootstrap-TOST (or non-inferiority) contrast with full audit.

    A null prediction ("arm A stays close to arm B", or "arm A is not
    materially worse than arm B") still needs the same seed pairing and
    recipe audit as a directional contrast: equivalence computed on
    mispaired or recipe-confounded arms would certify the wrong thing.
    ``mode`` selects the two-sided equivalence test or the one-sided
    non-inferiority test.  Reuses :func:`_paired_diff`'s exact pairing and
    copies its audit/arm/seed metadata onto the test result, so the printed
    row and the decision table carry the same sample the effect was computed
    on.
    """
    paired = _paired_diff(runs_a, runs_b, metric)
    diffs = paired.get("diffs") or []
    ci = (noninferiority_test(diffs, bound) if mode == "noninferior"
          else equivalence_test(diffs, bound))
    for key in ("arms", "arms_all", "arms_note", "baseline_mean",
                "diffs", "curve_invalid_seeds", "seeds_used",
                "seeds_dropped", "seeds_unpaired", "seeds_undefined",
                "stream_test", "controlled"):
        if key in paired:
            ci[key] = paired[key]
    return ci


def _factorial_main_effect(cells, metric, factor):
    """Averaged main effect of one factor in the complete calibration 2x2.

    ``cells`` maps the four ``_CALIB_FACTORIAL_CELLS`` to their run lists.
    The contrast is the factor-on mean minus the factor-off mean, where each
    arm averages over both levels of the other factor: for ``length`` this
    is ``mean(full, calib_len_only) - mean(calib_ent_only, calib_no_terms)``.
    Unlike a leave-one-out contrast (which is conditional on the other term
    being on), this estimates the term's average contribution across the
    full factorial, so a component that only helps in one configuration is
    not credited as a general effect.

    The paired sample is the seeds where *all four* cells carry a finite
    value (the same seed must appear in every cell); the effective seed set
    and every dropped seed are recorded, including runs with no training
    seed (they cannot be paired).  The audit checks both within-level
    pairwise contrasts (the two edges that vary only ``factor``), requires
    the resolved model config to differ only in that factor's flag, and
    verifies one global recipe and one seed set across all four cells - a
    main effect averaged over a cell with a different optimizer or epoch
    step count would not be a factorial effect.
    """
    spec = _FACTORIAL_MAIN_EFFECTS[factor]
    required = _CALIB_FACTORIAL_CELLS
    missing = [c for c in required if not cells.get(c)]
    if missing:
        raise ValueError(
            f"{factor} main effect needs all 2x2 cells; missing {missing}")
    missing_runs = [r for c in required for r in cells[c]
                    if r.get("seed") is None]
    by_cell = {c: {r["seed"]: r for r in cells[c]
                   if r.get("seed") is not None} for c in required}
    planned = sorted(set().union(*(set(m) for m in by_cell.values())),
                     key=_seed_order)
    diffs, used, dropped, unpaired, undefined = [], [], [], [], []
    for seed in planned:
        rs = [by_cell[c].get(seed) for c in required]
        if any(r is None for r in rs):
            dropped.append(seed)
            unpaired.append(seed)
            continue
        vals = [r.get(metric) for r in rs]
        finite = [_finite(v) for v in vals]
        if not all(finite):
            dropped.append(seed)
            if any(finite):
                unpaired.append(seed)
            else:
                undefined.append(seed)
            continue
        used.append(seed)
        values = {c: float(by_cell[c][seed][metric]) for c in required}
        diffs.append(
            sum(values[c] for c in spec["on"]) / len(spec["on"])
            - sum(values[c] for c in spec["off"]) / len(spec["off"]))
    ci = paired_bootstrap_ci(diffs)
    on_vals = [float(by_cell[c][s][metric])
               for s in used for c in spec["on"]]
    off_vals = [float(by_cell[c][s][metric])
                for s in used for c in spec["off"]]
    ci["arms"] = {"a": _mean_std(on_vals), "b": _mean_std(off_vals)}
    ci["arms_note"] = (f"pooled {factor}-on vs {factor}-off cells on the "
                       f"paired seeds")
    ci["baseline_mean"] = ci["arms"]["b"]["mean"]
    ci["diffs"] = list(diffs)
    ci["curve_invalid_seeds"] = _curve_invalid_seeds_for(
        used, tuple(by_cell[c] for c in required))
    ci["seeds_used"] = sorted(used, key=_seed_order)
    ci["seeds_dropped"] = (sorted(dropped, key=_seed_order)
                           + [None] * len(missing_runs))
    ci["seeds_unpaired"] = (sorted(unpaired, key=_seed_order)
                            + [None] * len(missing_runs))
    ci["seeds_undefined"] = sorted(undefined, key=_seed_order)

    mismatches, factors = [], {}
    for left, right in spec["pairs"]:
        audit = _controlled_audit(cells[left], cells[right])
        mismatches.extend(audit.get("mismatches", []))
        factors.update(audit.get("varying_factors", {}))
        diff_keys = set((audit.get("model_factor_diffs") or {}).keys())
        if diff_keys:
            if diff_keys != {spec["model_key"]}:
                mismatches.append(
                    f"{left} vs {right}: expected only "
                    f"{spec['model_key']} to differ in the resolved model "
                    f"config, got {sorted(diff_keys)}")
        elif spec["model_key"] not in (audit.get("varying_factors") or {}):
            mismatches.append(
                f"{left} vs {right}: no audited {spec['model_key']} "
                f"difference between the cells")
    seed_sets = {c: frozenset(by_cell[c]) for c in required}
    if len(set(seed_sets.values())) != 1:
        mismatches.append(
            "2x2 cells do not share one seed set: "
            + ", ".join(f"{c}={sorted(s, key=_seed_order)}"
                        for c, s in seed_sets.items()))
    views = [(c, j, _recipe_view(r))
             for c in required for j, r in enumerate(cells[c])]
    if views:
        first = views[0][2]
        for cell, j, view in views[1:]:
            diff = sorted(k for k in set(first) | set(view)
                          if view.get(k) != first.get(k))
            if diff:
                mismatches.append(
                    f"factorial recipe differs in cell {cell} run {j}: "
                    f"{diff}")
    ci["controlled"] = {"controlled": not mismatches,
                        "mismatches": mismatches,
                        "varying_factors": factors}
    return ci


def _grid_level_main_effect(metric, level_cells, baseline_cells):
    """Averaged main effect of one factor level, collapsed over controls.

    E3's grid is gate x norm x LayerScale, but the only registered norm
    contrast was ``seednorm - pre`` *within* one gate and LayerScale cell.
    An average-own-level effect (each norm level vs the ``pre`` baseline,
    collapsed over gate and LayerScale) is what answers "does this
    normalisation level contribute at all", not merely "does it differ in
    one control cell".  ``level_cells`` is the list of run lists carrying
    the level to test (one per control combination, e.g. for each gate and
    LayerScale), ``baseline_cells`` the matching ``pre``-level run lists in
    the same order.

    The paired sample is the seeds where *every* one of the level and
    baseline cells carries a finite value; the effect is the level mean
    minus the baseline mean within each seed, so both gate and LayerScale
    are averaged inside the paired difference (repeated-measures pairing:
    every control combination contributes exactly one level run and one
    baseline run per seed).  Effective seed set and dropped seeds are
    recorded (including runs with no training seed), and each cell pair is
    audited with the same
    controlled-variable contract as a marginal contrast: the two cells must
    vary only the ``norm`` factor and share one recipe.  A main effect
    averaged over a cell with a different optimizer or with a non-norm
    config difference would not be a factorial effect.
    """
    n = len(level_cells)
    if n == 0 or n != len(baseline_cells):
        raise ValueError(
            "grid-level main effect needs equally many level and baseline "
            f"cells (got {n} vs {len(baseline_cells)})")
    missing_runs = [r for rs in level_cells + baseline_cells for r in rs
                    if r.get("seed") is None]
    maps = [({r["seed"]: r for r in rs if r.get("seed") is not None})
            for rs in level_cells]
    base_maps = [({r["seed"]: r for r in rs if r.get("seed") is not None})
                 for rs in baseline_cells]
    planned = sorted(set().union(*(set(m) for m in maps + base_maps)),
                     key=_seed_order)
    diffs, used, dropped, unpaired, undefined = [], [], [], [], []
    for seed in planned:
        level_vals, base_vals = [], []
        present = True
        finite_any = False
        for m in maps:
            r = m.get(seed)
            if r is None:
                present = False
                break
            v = r.get(metric)
            level_vals.append(v)
            finite_any = finite_any or _finite(v)
        for m in base_maps:
            r = m.get(seed)
            if r is None:
                present = False
                break
            v = r.get(metric)
            base_vals.append(v)
            finite_any = finite_any or _finite(v)
        if not present:
            dropped.append(seed)
            unpaired.append(seed)
            continue
        if not all(_finite(v) for v in level_vals + base_vals):
            dropped.append(seed)
            (unpaired if finite_any else undefined).append(seed)
            continue
        used.append(seed)
        diffs.append(sum(float(v) for v in level_vals) / len(level_vals)
                     - sum(float(v) for v in base_vals) / len(base_vals))
    ci = paired_bootstrap_ci(diffs)
    level_pool = [float(m[s][metric]) for s in used for m in maps]
    base_pool = [float(m[s][metric]) for s in used for m in base_maps]
    ci["arms"] = {"a": _mean_std(level_pool), "b": _mean_std(base_pool)}
    ci["arms_note"] = ("level vs pre pooled over gate x LayerScale, on the "
                       "paired seeds")
    ci["baseline_mean"] = ci["arms"]["b"]["mean"]
    ci["diffs"] = list(diffs)
    ci["curve_invalid_seeds"] = _curve_invalid_seeds_for(
        used, tuple(maps + base_maps))
    ci["seeds_used"] = sorted(used, key=_seed_order)
    ci["seeds_dropped"] = (sorted(dropped, key=_seed_order)
                           + [None] * len(missing_runs))
    ci["seeds_unpaired"] = (sorted(unpaired, key=_seed_order)
                            + [None] * len(missing_runs))
    ci["seeds_undefined"] = sorted(undefined, key=_seed_order)

    mismatches, factors = [], {}
    for level_rs, base_rs in zip(level_cells, baseline_cells):
        audit = _controlled_audit(level_rs, base_rs)
        mismatches.extend(audit.get("mismatches", []))
        factors.update(audit.get("varying_factors", {}))
        diff_keys = set((audit.get("model_factor_diffs") or {}).keys())
        if diff_keys and diff_keys != {"norm"}:
            mismatches.append(
                f"norm level pair varies {sorted(diff_keys)} rather than "
                f"only the norm factor")
        elif not diff_keys and "norm" not in (
                audit.get("varying_factors") or {}):
            mismatches.append(
                "norm level pair shows no audited norm difference")
    seed_sets = {i: frozenset(m) for i, m in enumerate(maps + base_maps)}
    if len(set(seed_sets.values())) != 1:
        mismatches.append("grid cells do not share one seed set")
    views = [(i, j, _recipe_view(r))
             for i, rs in enumerate(level_cells + baseline_cells)
             for j, r in enumerate(rs)]
    if views:
        first = views[0][2]
        for cell, j, view in views[1:]:
            diff = sorted(k for k in set(first) | set(view)
                          if view.get(k) != first.get(k))
            if diff:
                mismatches.append(
                    f"grid-level recipe differs in cell {cell} run {j}: "
                    f"{diff}")
    ci["controlled"] = {"controlled": not mismatches,
                        "mismatches": mismatches,
                        "varying_factors": factors}
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
    elif ci.get("test") == "noninferiority":
        if n >= 2 and ci.get("loo_above_bound") is False:
            flags.append("LOO below bound")
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
    if ci.get("test") == "noninferiority":
        # One-sided null prediction: only the lower end matters, and the
        # threshold is the *negative* of the registered positive bound.
        bound = ci.get("bound", float("nan"))
        if ci.get("noninferior"):
            verdict = f"CI lower bound above -{bound:g} (prediction holds)"
        elif ci["lo"] <= -bound:
            verdict = f"CI lower bound below -{bound:g} (prediction refuted)"
        else:
            verdict = f"CI not above -{bound:g} (inconclusive)"
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


# Metrics that are raw accuracy (or an accuracy difference) and therefore
# inherit the held-out answer-balance premise.  ``f1_macro``,
# ``recall_macro`` and the PR-AUC family are explicitly excluded: they are
# the companions that remain interpretable under a skewed stream.
_ACCURACY_METRICS = {
    "accuracy", "accuracy_at_base", "accuracy_at_longest", "accuracy_drop",
    # The E2 whole-sweep slope is fitted to accuracy, so the same balance
    # premise applies; omitting it would let the slope rows survive an
    # answer-skew gate that invalidates the endpoint/drop rows.
    "accuracy_slope_log_len", "norm_accuracy",
}

CURVE_INVALID_VERDICTS = (
    "plateau", "diverging", "numerical_instability", "insufficient_points")


def _curve_valid_runs(runs):
    return [r for r in runs
            if r.get("curve_verdict") not in CURVE_INVALID_VERDICTS]


def _curve_invalid_seeds_for(used, run_maps):
    """Seeds whose paired contrast is contaminated by a curve-invalid run.

    A contrast seed is only usable for the sensitivity readout when *every*
    side/cell entering that seed's difference has a claim-valid curve.  The
    returned list is stored with the contrast so the decision layer can
    recompute the effect without those seeds.
    """
    bad = set()
    for seed in used:
        for runs in run_maps:
            run = runs.get(seed)
            if run is not None and run.get("curve_verdict") in \
                    CURVE_INVALID_VERDICTS:
                bad.add(seed)
                break
    return sorted(bad, key=_seed_order)


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


def _selection_reliability(rows, metric="val_accuracy_mean",
                           std_key="val_accuracy_std",
                           n_key="n_seeds", curve_key="curve_verdicts",
                           diverged_key="diverged_runs", grid=None,
                           grid_keys=None) -> dict:
    """Audit an argmax hyperparameter selection for winner's curse.

    Every tuned baseline here selects a value by taking the argmax of a
    validation column over a grid, then freezes it for the headline
    comparisons.  An argmax with no uncertainty attached is a point
    selection that can be entirely noise: the tuned baseline must beat
    the untuned floor and be significantly beaten by the model under
    test, so if the winning value sits inside the seed-to-seed spread of
    the runner-up the "tuned" baseline is a strawman in the *conservative*
    direction (it under-reports the baseline) and the contrast loses
    power - exactly the failure mode the design contract exists to
    prevent.  This helper records, next to the chosen value:

    - ``margin`` = best - runner-up on the selection metric;
    - ``margin_se`` an upper-bound gap standard error,
      ``sqrt(std_best^2 / n_best + std_second^2 / n_second)``.  The grid
      rows come from the *same* seeds (paired), so the conservative
      unpaired combination is used deliberately: it cannot claim more
      resolution than a paired test would;
    - ``within_noise`` = ``margin < margin_se`` or ``margin <= 0`` (an
      exact tie is never resolved): the winner is not resolved beyond
      seed noise, and small follow-up differences built on this value
      inherit that ambiguity;
    - ``curve_invalid_best``: how many of the best row's runs are
      curve-invalid (``plateau``/``diverging``/...): an argmax chosen
      among runs that never learned measures the optimizer, not the
      baseline.  Only present when the rows carry ``curve_verdicts``;
    - ``diverged_runs_best``: how many of the best row's runs diverged
      and were excluded from its statistic by the tuner (e.g.
      ``tune_linear_probe``): a mean over the surviving runs understates
      the uncertainty.  Only present when the rows carry
      ``diverged_runs``;
    - ``best_at_grid_boundary``: the winning index is the first or last
      grid point, so the optimum may lie outside the searched range and
      the true baseline could be stronger.  Only present when ``grid``
      is given;
    - ``reliable``: ``True`` only when the selection resolves the value
      (no boundary, no within-noise margin, and no curve-invalid runs in
      the winning row); ``None`` when the audit cannot be computed at
      all (missing/constant metric).  ``None`` is an open audit item,
      not a pass.

    ``rows`` is the table (list of dicts) the tuner searched, ordered as
    the grid; the selection metric is read with ``_finite`` so a
    missing statistic can never be read as 0.0.
    """
    metric_rows = [(i, r) for i, r in enumerate(rows)
                   if _finite(r.get(metric))]
    out = {"metric": metric, "margin": float("nan"),
           "margin_se": float("nan"), "within_noise": None,
           "curve_invalid_best": None, "diverged_runs_best": None,
           "best_at_grid_boundary": None, "reliable": None}
    if len(metric_rows) < 2:
        return out
    ranked = sorted(metric_rows, key=lambda t: -float(t[1][metric]))
    (i_best, r_best), (i_second, r_second) = ranked[0], ranked[1]
    margin = float(r_best[metric]) - float(r_second[metric])

    def _se(row):
        # A single-seed row carries a convention-0 std, not a noise
        # estimate; reading it as 0 resolution would be exactly the
        # overconfidence this audit exists to catch.
        std = row.get(std_key)
        n = row.get(n_key)
        if _finite(std) and _finite(n) and float(n) >= 2.0:
            return (max(float(std), 0.0) ** 2) / float(n)
        return float("nan")

    var_best, var_second = _se(r_best), _se(r_second)
    margin_se = math.sqrt(var_best + var_second) \
        if math.isfinite(var_best) and math.isfinite(var_second) \
        else float("nan")
    within_noise = None
    if math.isfinite(margin_se):
        # An exact tie has no winner even when both rows happen to have
        # zero seed variance; ``margin < margin_se`` would call that
        # resolved (0 < 0 is false) and stamp the arbitrary argmax as
        # reliable.
        within_noise = bool(margin <= 0.0 or margin < margin_se)

    curve_invalid_best = None
    verdicts = r_best.get(curve_key)
    if isinstance(verdicts, (list, tuple)):
        curve_invalid_best = sum(1 for v in verdicts
                                 if v in CURVE_INVALID_VERDICTS)
    diverged_best = None
    if _finite(r_best.get(diverged_key)):
        diverged_best = int(r_best[diverged_key])
    boundary = None
    if grid is not None and len(grid) == len(rows):
        boundary = bool(i_best in (0, len(rows) - 1))
    # A multi-dimensional grid (e.g. lr x steps) has a boundary on *each*
    # axis: an optimum at the smallest or largest learning rate is just as
    # unresolved as one at the shortest step budget, even when the flattened
    # row index is interior.
    axis_boundaries = {}
    if grid_keys:
        for key in grid_keys:
            vals = sorted({float(r[key]) for r in rows
                           if _finite(r.get(key))})
            if len(vals) >= 2 and _finite(r_best.get(key)):
                axis_boundaries[str(key)] = bool(
                    float(r_best[key]) in (vals[0], vals[-1]))
        if axis_boundaries:
            boundary = bool(boundary or any(axis_boundaries.values()))
    reliable = None
    if within_noise is not None:
        reliable = not within_noise
        if boundary is not None:
            reliable = reliable and not boundary
        if curve_invalid_best is not None:
            reliable = reliable and curve_invalid_best == 0
        if diverged_best is not None:
            reliable = reliable and diverged_best == 0
    elif boundary or curve_invalid_best or diverged_best:
        # The noise-based decision can be unavailable (e.g. a one-seed
        # winner), but a grid-boundary winner, or a winning row that
        # contains diverged or curve-invalid runs, is demonstrably not a
        # resolved measurement - it must not be left as the neutral ``None``
        # state that callers read as "open".
        reliable = False
    out.update({"best_index": int(i_best), "second_index": int(i_second),
                "best": float(r_best[metric]),
                "runner_up": float(r_second[metric]), "margin": margin,
                "margin_se": margin_se, "within_noise": within_noise,
                "curve_invalid_best": curve_invalid_best,
                "diverged_runs_best": diverged_best,
                "best_at_grid_boundary": boundary,
                "axis_boundaries": axis_boundaries or None,
                "reliable": reliable})
    return out


def _curve_valid_selection_rows(table):
    """Split a tuning table into rows eligible for selection.

    A tuner must not freeze a recipe because its winning runs were
    curve-invalid (plateau/diverging/instability): those accuracies measure
    an optimizer failure, not the learning rate's merit.  Rows with no
    curve-invalid run are preferred; when every finite row is invalid the
    tuner falls back to all finite rows (and says so) rather than refusing
    to select at all.  Returns a dict with the eligible ``rows``, the
    ``n_invalid`` count among finite rows, the ``n_excluded`` count actually
    removed, and the selection ``pool`` label.
    """
    finite = [row for row in table if _finite(row.get("val_accuracy_mean"))]
    valid = [row for row in finite if not row.get("curve_invalid_runs")]
    if valid:
        return {"rows": valid, "n_finite": len(finite),
                "n_invalid": len(finite) - len(valid),
                "n_excluded": len(finite) - len(valid),
                "pool": "curve-valid"}
    return {"rows": finite, "n_finite": len(finite),
            "n_invalid": len(finite), "n_excluded": 0,
            "pool": "all-finite (no curve-valid row)"}


def _selection_warn(label, selection) -> str:
    """Human-readable annotation for an unreliable selection, or ""."""
    if not selection:
        return ""
    if (selection.get("reliable") is not False
            and not selection.get("best_at_grid_boundary")
            and not selection.get("within_noise")
            and not selection.get("diverged_runs_best")
            and not selection.get("curve_invalid_best")):
        return ""
    reasons = []
    if selection.get("best_at_grid_boundary"):
        axes = selection.get("axis_boundaries") or {}
        at = [k for k, v in axes.items() if v]
        suffix = f" ({', '.join(at)})" if at else ""
        reasons.append(f"optimum at grid boundary{suffix} - range may "
                       f"exclude it")
    if selection.get("within_noise"):
        reasons.append(f"margin {selection['margin']:.4f} < "
                       f"seed-gap SE {selection['margin_se']:.4f}")
    if selection.get("diverged_runs_best"):
        reasons.append(f"{selection['diverged_runs_best']} diverged runs in "
                       f"winning row (statistic over survivors only)")
    if selection.get("curve_invalid_best"):
        reasons.append(f"{selection['curve_invalid_best']} curve-invalid "
                       f"runs in winning row (underfit/broken, not the "
                       f"baseline)")
    return f"WARN {label}: selection not resolved ({'; '.join(reasons)})"


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
    optimizer/recipe settings, whose evaluation streams differ (size *or*
    seed), whose seeds are not paired, or that does not vary a declared
    model factor is *not* read as supported, whatever its
    p-value - the difference would not be attributable to the module under
    test.  The audit verdict is printed inline and stored per row, and so
    is each arm's mean +- std and seed count, so the decision table itself
    answers "did the effect exceed the seed-to-seed spread?" for every
    metric - including the E3 mechanism readouts that have no per-arm
    table of their own.

    The decision layer also separates statistical from *practical*
    significance: the hypothesis's pre-registered SESOI (looked up in
    ``HYPOTHESES``, so it is part of the hashed pre-registration) is
    compared with the effect, ``seeds_for_power`` plans how many seeds the
    SESOI would need at the observed variance, and an FDR-significant
    effect below its SESOI is reported as practically negligible instead
    of "supported".

    The pre-registered *direction* of the prediction (the ``direction``
    sign registered per metric in ``HYPOTHESES``, e.g. H1c's
    "null_mass_error_margin > 0") is enforced at the decision point as
    well: an FDR-significant effect of the opposite sign refutes the
    hypothesis, is labelled as such, and can never enter the supporting
    set.  The SESOI gate bounds the *magnitude* of the required effect;
    it is not a licence to count a reversed effect as support.

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

    Seed floor: a contrast may only support a claim when at least
    ``min_seeds`` (default 3, pre-registered in ``Settings``) independent
    training seeds actually entered the paired difference.  A 0.2% effect
    next to a 0.5% seed-level std - or a two-seed "win" - is not evidence,
    so such rows are marked ``claim_eligible=False`` at the decision
    point and read as INELIGIBLE regardless of their p-value.  Rows also
    record ``seeds_used``/``seeds_dropped`` (split into ``seeds_unpaired``
    and ``seeds_undefined``) so a sample that silently shrank - a value
    present on one side only, or a metric undefined for a seed - is
    visible instead of feeding an unpaired comparison.
    """

    def __init__(self, tag: str, hypotheses=(), min_seeds: int = 3,
                 competence: dict | None = None, coverage=(),
                 balance_ok: bool = True):
        self.tag = tag
        self.hypotheses = tuple(hypotheses)
        self.min_seeds = int(min_seeds)
        self.rows: list[dict] = []
        # Held-out answer-balance premise for accuracy-based claims.  The
        # gross-imbalance audit is not merely descriptive: if the stream is
        # not uniform, an accuracy contrast is partly a class-prior contrast
        # and cannot support a mechanism claim.  ``True``/``None`` (not
        # audited) leave the decision untouched; only an explicit failure
        # gates the accuracy rows.  Macro-F1/balanced-accuracy and PR-AUC
        # rows are unaffected.
        self.balance_ok = balance_ok is not False
        # HB0: the family's competence prerequisite (at least one arm above
        # chance).  When it fails, no contrast in the family may support a
        # claim - an underfit grid's differences describe the step count, not
        # the mechanism.  ``None`` means "not assessed" and does not gate.
        self.competence = competence
        # Factorial/design coverage payloads (``_report_ablation_coverage``
        # / ``_report_design_coverage``).  A missing cell or a cell that
        # trained only part of the planned seed set means the corresponding
        # effect was never measured; a contrast from such a design cannot
        # support a claim even when its own p-value is small.  A single
        # payload and an iterable of payloads are both accepted.
        if coverage is None:
            coverage = ()
        elif isinstance(coverage, dict):
            coverage = (coverage,)
        self.coverage = tuple(coverage)

    def print(self, rule, hyp, metric, ci, warn="",
              direction_key=None) -> None:
        """Register one computed contrast in the family.

        ``metric`` is the display/registry key used for the SESOI lookup.
        ``direction_key`` optionally names a *different* registry key for
        the pre-registered sign, which is needed when one metric carries
        both a directional main-effect row and a two-sided companion row
        (e.g. H2e's concentration main effect vs. its interaction): the
        companion must not inherit the main effect's sign.  An explicit
        ``direction_key`` that is not registered is rejected instead of
        silently producing a two-sided row.
        """
        test = ci.get("test")
        equiv = test == "equivalence"
        noninf = test == "noninferiority"
        null_test = equiv or noninf
        audit = ci.get("controlled")
        sesoi = (HYPOTHESES.get(hyp, {}).get("sesoi") or {}).get(metric)
        if null_test:
            # A null-prediction row's decision bound is the CI's registered
            # bound, which can live under a separate registry key when the
            # displayed metric is shared with directional rows (H2c's
            # reverse check).  ``sesoi`` above is keyed by the display
            # metric, so using it here would print and apply the
            # directional SESOI as if it were the bound.
            bound = ci.get("bound")
            if bound is not None:
                sesoi = bound
        # Pre-registered sign of the predicted effect for this metric
        # ("+" / "-", looked up in HYPOTHESES so it is part of the hashed
        # pre-registration).  ``None`` means the prediction is not
        # directional (e.g. H3b's "the readout shifts"), in which case a
        # non-zero effect of either sign is admissible.
        direction_map = (HYPOTHESES.get(hyp, {}).get("direction") or {})
        if direction_key is not None and direction_key not in direction_map:
            raise ValueError(
                f"{hyp} has no pre-registered direction under "
                f"direction_key={direction_key!r}; registered keys: "
                f"{sorted(direction_map)}")
        direction = direction_map.get(
            metric if direction_key is None else direction_key)
        arms = ci.get("arms") or {}
        arm_a, arm_b = arms.get("a") or {}, arms.get("b") or {}
        # Full-run levels are stored alongside the paired levels so a reader
        # can see exactly how much the sample restriction moved an arm mean
        # (the decision table itself always reads ``arms``).
        arms_all = ci.get("arms_all") or {}
        all_a, all_b = arms_all.get("a") or {}, arms_all.get("b") or {}
        self.rows.append({
            "rule": rule, "hypothesis": hyp, "metric": metric,
            "mean": ci.get("mean"), "lo": ci.get("lo"), "hi": ci.get("hi"),
            "n": ci.get("n"), "p_value": ci.get("p_value"),
            "mde80": ci.get("mde80"), "sd": ci.get("sd"),
            "dz": _std_effect(ci),
            "a_mean": arm_a.get("mean"), "a_std": arm_a.get("std"),
            "a_n": arm_a.get("n"),
            "b_mean": arm_b.get("mean"), "b_std": arm_b.get("std"),
            "b_n": arm_b.get("n"),
            "a_all_mean": all_a.get("mean"), "a_all_n": all_a.get("n"),
            "b_all_mean": all_b.get("mean"), "b_all_n": all_b.get("n"),
            "arms_note": ci.get("arms_note"),
            "sesoi": sesoi,
            # An equivalence prediction has no directional alternative:
            # whichever sign the small drift takes, the claim is "inside
            # the bound".  Keeping the directional registry value here
            # would let a later reader misread the row as a one-sided
            # superiority test.
            "direction": None if null_test else direction,
            "test": test if test in ("equivalence", "noninferiority")
                    else "difference",
            "equivalence_bound": ci.get("bound") if null_test else None,
            "noninferiority_bound": (ci.get("bound") if noninf else None),
            "equivalent": ci.get("equivalent") if equiv else None,
            "noninferior": ci.get("noninferior") if noninf else None,
            "sign_consistency": ci.get("sign_consistency"),
            "loo_mean_min": ci.get("loo_mean_min"),
            "loo_mean_max": ci.get("loo_mean_max"),
            "loo_sign_flip": ci.get("loo_sign_flip"),
            "loo_within_bound": ci.get("loo_within_bound"),
            "loo_above_bound": ci.get("loo_above_bound"),
            "lo_bca": ci.get("lo_bca"),
            "hi_bca": ci.get("hi_bca"),
            "equivalent_bca": ci.get("equivalent_bca") if equiv else None,
            "noninferior_bca": (ci.get("noninferior_bca") if noninf
                                else None),
            "bca_agrees": ci.get("bca_agrees"),
            "baseline_mean": ci.get("baseline_mean"),
            "controlled": None if audit is None else audit["controlled"],
            "varying_factors": ({} if audit is None
                                else audit.get("varying_factors")),
            "control_mismatches": ([] if audit is None
                                   else audit.get("mismatches", [])),
            # The parameter counts the two sides actually trained (with the
            # matched flag): a gate contrast may legitimately differ, but
            # the reader must see the capacity delta rather than infer that
            # "only the factor varies" also means "same size".
            "capacity": (None if audit is None
                         else audit.get("capacity")),
            "seeds_used": ci.get("seeds_used"),
            "seeds_dropped": ci.get("seeds_dropped"),
            "seeds_unpaired": ci.get("seeds_unpaired"),
            "seeds_undefined": ci.get("seeds_undefined"),
            # Per-seed paired differences and the seeds contaminated by a
            # curve-invalid run: finalize() drops the latter and checks
            # whether the supported effect survives (sensitivity, not a
            # replacement for the pre-registered decision rule).
            "diffs": list(ci.get("diffs") or []),
            "curve_invalid_seeds": list(ci.get("curve_invalid_seeds") or []),
            # Item-level companion on the shared finite evaluation stream
            # (McNemar).  ``stream_sign_conflict`` is a hard gate below:
            # a seed-level sign that disagrees with the sign over the
            # actual items is not measuring the same effect.
            "stream_test": ci.get("stream_test"),
            "stream_test_p": (ci.get("stream_test") or {}).get("p_exact"),
            "stream_pooled_descriptive": (
                ci.get("stream_test") or {}).get("pooled_descriptive"),
            "stream_per_seed": (ci.get("stream_test") or {}).get("per_seed"),
            "stream_seed_sign_test_p": (
                ci.get("stream_test") or {}).get("seed_sign_test_p"),
            "stream_seed_sign_consistent": (
                ci.get("stream_test") or {}).get(
                    "seed_sign_consistent_with_mean"),
            "stream_seed_sign_positive": (
                ci.get("stream_test") or {}).get("seed_sign_positive"),
            "stream_seed_sign_negative": (
                ci.get("stream_test") or {}).get("seed_sign_negative"),
            "stream_direction_agrees":
                (ci.get("stream_test") or {}).get("direction_agrees"),
            "stream_sign_conflict": False,
        })
        line = f"  [{hyp}] {rule} ({metric}): {_fmt_ci(ci)}"
        if audit is not None:
            factors = ",".join(sorted(audit.get("varying_factors", {}))) or "n/a"
            line += (f"  [control: {'ok' if audit['controlled'] else 'FAILED'}"
                     f", varies={factors}]")
            cap = audit.get("capacity")
            if cap and not cap["matched"]:
                line += (f"  [capacity: {cap['a']} vs {cap['b']} params, "
                         f"delta={cap['delta_b_minus_a']:+d}]")
        if sesoi is not None:
            if equiv:
                line += f"  [equivalence bound={sesoi:g}]"
            elif noninf:
                line += f"  [non-inferiority bound={sesoi:g}]"
            else:
                line += f"  [SESOI={sesoi:g}]"
        if not null_test and direction is not None:
            line += (f"  [prediction: effect "
                     f"{'>' if direction == '+' else '<'} 0]")
        dropped = ci.get("seeds_dropped") or []
        if dropped:
            why = []
            if ci.get("seeds_unpaired"):
                why.append(f"{len(ci['seeds_unpaired'])} unpaired")
            if ci.get("seeds_undefined"):
                why.append(f"{len(ci['seeds_undefined'])} metric-undefined")
            line += (f"  [seed coverage: {len(dropped)} of "
                     f"{len(dropped) + len(ci.get('seeds_used') or [])} "
                     f"planned seed(s) dropped ({', '.join(why) or 'n/a'}); "
                     f"effective n={ci.get('n')}]")
        stream = ci.get("stream_test") or {}
        if stream.get("applicable"):
            per_seed = stream.get("per_seed") or []
            counts = ", ".join(
                f"{s['seed']}:{s['b']}/{s['c']}" for s in per_seed)
            sign_p = stream.get("seed_sign_test_p")
            sign_p_s = f"{sign_p:.4g}" if _finite(sign_p) else "n/a"
            line += (f"  [finite-stream McNemar (pooled, descriptive only: "
                     f"repeated sealed items): b={stream['b']}, "
                     f"c={stream['c']}, p={stream['p_exact']:.4g}, "
                     f"n_items={stream['n']}; per-seed discordant b/c="
                     f"[{counts}]; seed signs +/-/0={stream['seed_sign_positive']}/"
                     f"{stream['seed_sign_negative']}/{stream['seed_sign_zero']}, "
                     f"independent-seed sign p={sign_p_s}]")
        elif stream.get("reason"):
            line += f"  [finite-stream test n/a: {stream['reason']}]"
        flags = small_n_flags(ci)
        if flags:
            line += f"  [{flags}]"
        if warn:
            line += f"  [{warn}]"
        print(line)
        if audit is not None and not audit["controlled"]:
            for mismatch in audit["mismatches"]:
                print(f"      CONTROL WARNING: {mismatch}")

    def print_equivalence(self, rule, hyp, metric, diffs, warn="",
                          bound_metric=None) -> None:
        """Add a pre-registered *null-prediction* contrast (bootstrap TOST).

        ``bound`` is read from the hypothesis's registered SESOI and the
        metric must be declared as an equivalence test under
        ``sesoi_test``; otherwise the call raises, because running a null
        prediction through the difference machinery would misread "CI
        includes 0" (which may merely mean "underpowered") as support.

        ``bound_metric`` selects a separately registered bound key when
        the displayed metric is also used by directional contrasts (e.g.
        H2c's reverse check displays ``accuracy_at_longest`` but registers
        its null prediction under ``calib_minus_len_only_...``).
        """
        key = metric if bound_metric is None else bound_metric
        entry = HYPOTHESES.get(hyp, {})
        bound = (entry.get("sesoi") or {}).get(key)
        mode = (entry.get("sesoi_test") or {}).get(key)
        if mode not in ("equivalence", "noninferior") or bound is None:
            raise ValueError(
                f"{hyp}/{key} has no pre-registered equivalence or "
                f"non-inferiority bound in HYPOTHESES; register it under "
                f"'sesoi_test' first")
        ci = (noninferiority_test(diffs, bound) if mode == "noninferior"
              else equivalence_test(diffs, bound))
        self.print_equivalence_ci(rule, hyp, metric, ci, warn=warn)

    def print_equivalence_ci(self, rule, hyp, metric, ci, warn="") -> None:
        """Register an already-computed equivalence/non-inferiority CI.

        The two-arm reverse check needs the same paired-seed accounting and
        controlled-variable audit as a directional contrast.  Building the
        CI through ``_paired_equivalence`` and registering it here keeps the
        TOST (or one-sided non-inferiority) decision rule while refusing to
        print a null claim whose arms were not actually paired.
        """
        if ci.get("test") not in ("equivalence", "noninferiority"):
            raise ValueError(
                f"{hyp}/{metric}: print_equivalence_ci needs a CI from "
                f"equivalence_test or noninferiority_test "
                f"(got test={ci.get('test')!r})")
        self.print(rule, hyp, metric, ci, warn=warn)

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
            elif row["test"] == "noninferiority":
                # One-sided null prediction ("the arm is not materially
                # worse"): support is the CI's lower bound staying above the
                # negative bound, so a loss beyond it is a refutation rather
                # than "no significant difference".
                row["practically_relevant"] = bool(row.get("noninferior"))
            else:
                # Directional pre-registration: a prediction like
                # "null_mass_error_margin > 0" is only confirmed by an
                # effect in *that* direction.  A significant effect of the
                # opposite sign refutes the hypothesis, so the sign is
                # enforced here (before the SESOI magnitude gate) and such
                # a row can never count as support.
                direction = row.get("direction")
                row["sign_ok"] = (None if direction is None
                                  or not finite_effect
                                  else bool(mean > 0.0 if direction == "+"
                                            else mean < 0.0))
                row["direction_reversed"] = bool(
                    direction is not None and finite_effect
                    and not row["sign_ok"])
                row["practically_relevant"] = bool(
                    not row["direction_reversed"]
                    and (sesoi is None
                         or (finite_effect and math.isfinite(float(sesoi))
                             and abs(mean) >= float(sesoi))))
            # Pre-registered seed floor: below ``min_seeds`` independent
            # paired seeds a "supported" verdict would be noise dressed as
            # evidence; such rows cannot support a claim at the decision
            # point whatever their p-value.  HB0's competence prerequisite
            # gates the same way: no claim from an underfit grid.
            row["competence_passed"] = (
                None if self.competence is None
                else bool(self.competence.get("passed")))
            coverage_failures = []
            for payload in self.coverage:
                if payload.get("complete") is False:
                    coverage_failures.append(
                        f"{payload.get('tag', '?')}/{payload.get('design', 'design')} "
                        f"missing cells {payload.get('missing_cells')}")
                if payload.get("seed_balanced") is False:
                    coverage_failures.append(
                        f"{payload.get('tag', '?')}/{payload.get('design', 'design')} "
                        f"seed-unbalanced cells {payload.get('unbalanced_cells')}")
            row["coverage_gate_failed"] = bool(coverage_failures)
            row["coverage_failures"] = coverage_failures
            row["seed_floor_failed"] = bool(
                (row.get("n") or 0) < self.min_seeds)
            row["claim_eligible"] = bool(
                (row.get("n") or 0) >= self.min_seeds
                and row["competence_passed"] is not False
                and not coverage_failures)
            # Curve-invalid sensitivity: a supported effect that rests on
            # seeds whose run never learned (plateau/diverging/...) is not
            # evidence about the mechanism.  Drop those seeds and recompute
            # the same effect; if the support disappears (or the surviving
            # sample falls below the pre-registered seed floor), the row is
            # marked ineligible.  This is a sensitivity gate, not a second
            # decision rule: rows without curve-invalid seeds are unchanged.
            bad_seeds = set(row.get("curve_invalid_seeds") or [])
            row["curve_invalid_n"] = len(bad_seeds)
            row["curve_valid_n"] = row.get("n")
            row["curve_invalid_removes_support"] = False
            if bad_seeds:
                paired = list(zip(row.get("seeds_used") or [],
                                  row.get("diffs") or []))
                valid_diffs = [d for seed, d in paired
                               if seed not in bad_seeds]
                row["curve_valid_n"] = len(valid_diffs)
                if row["test"] == "equivalence":
                    sens = equivalence_test(
                        valid_diffs, row.get("equivalence_bound"))
                    sens_ok = bool(sens.get("equivalent"))
                elif row["test"] == "noninferiority":
                    sens = noninferiority_test(
                        valid_diffs, row.get("noninferiority_bound"))
                    sens_ok = bool(sens.get("noninferior"))
                else:
                    sens = paired_bootstrap_ci(valid_diffs)
                    direction = row.get("direction")
                    if direction == "+":
                        sign_ok = bool(sens["mean"] > 0.0 and sens["lo"] > 0.0)
                    elif direction == "-":
                        sign_ok = bool(sens["mean"] < 0.0 and sens["hi"] < 0.0)
                    else:
                        sign_ok = bool(sens["lo"] > 0.0 or sens["hi"] < 0.0)
                    row_sesoi = row.get("sesoi")
                    magnitude_ok = (
                        row_sesoi is None
                        or (math.isfinite(sens["mean"])
                            and abs(sens["mean"]) >= float(row_sesoi)))
                    sens_ok = bool(sign_ok and magnitude_ok)
                row["curve_valid_mean"] = sens.get("mean")
                row["curve_valid_lo"] = sens.get("lo")
                row["curve_valid_hi"] = sens.get("hi")
                row["curve_valid_p"] = sens.get("p_value")
                if (row["claim_eligible"] and row["fdr_reject"]
                        and row["practically_relevant"]
                        and (len(valid_diffs) < self.min_seeds
                             or not sens_ok)):
                    row["claim_eligible"] = False
                    row["curve_invalid_removes_support"] = True
            # The seed-level effect and the item-level paired effect must
            # at least share a sign.  Opposite signs mean the two
            # summaries are not measuring the same contrast (e.g. unequal
            # per-seed item counts or a generated mismatch), so the row
            # cannot support a directional claim.
            if (row["claim_eligible"]
                    and row.get("stream_direction_agrees") is False):
                row["claim_eligible"] = False
                row["stream_sign_conflict"] = True
            # Accuracy is the pre-registered headline only under the
            # measured balance premise.  If the held-out stream failed the
            # gross-imbalance audit, the same accuracy contrast is partly a
            # class-prior contrast: keep it in the table for transparency
            # but refuse the claim (macro-F1 / PR-AUC companions can still
            # carry the mechanism reading).
            if (not self.balance_ok and row["claim_eligible"]
                    and row["metric"] in _ACCURACY_METRICS):
                row["claim_eligible"] = False
                row["accuracy_balance_failed"] = True
            row["supports_hypothesis"] = bool(
                row["fdr_reject"] and row["practically_relevant"]
                and row["claim_eligible"])
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
            elif row["test"] == "noninferiority":
                row["loo_flips_decision"] = bool(
                    claim and row.get("loo_above_bound") is False)
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
        # A refutation is a decision claim like any other: it needs the
        # pre-registered seed floor and an unconfounded contrast (the same
        # precedence as the printed verdict), otherwise the row stays a raw
        # sign observation without a claim.
        reversed_effects = [
            r for r in surviving
            if r.get("direction_reversed") and r.get("claim_eligible", True)
            and not r.get("confounded")]
        below_sesoi = [r for r in surviving
                       if not r["practically_relevant"]
                       and not r.get("direction_reversed")
                       and r["test"] not in ("equivalence",
                                             "noninferiority")]
        equiv_rows = [r for r in testable if r["test"] == "equivalence"]
        equivalent = [r for r in equiv_rows if r.get("equivalent")]
        equiv_borderline = [r for r in surviving
                            if r["test"] == "equivalence"
                            and not r.get("equivalent")]
        noninf_rows = [r for r in testable if r["test"] == "noninferiority"]
        noninferior = [r for r in noninf_rows if r.get("noninferior")]
        noninf_borderline = [r for r in surviving
                             if r["test"] == "noninferiority"
                             and not r.get("noninferior")]
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
            noninf = r["test"] == "noninferiority"
            null_test = equiv or noninf
            sesoi_s = (f"{'bound' if equiv else ('NI bound' if noninf else 'SESOI')}"
                       f"={sesoi:g}" if sesoi is not None else "")
            if r.get("confounded"):
                verdict = "REFUSED (confounded contrast)"
            elif r.get("competence_passed") is False:
                verdict = ("INELIGIBLE (HB0 competence prerequisite failed: "
                           "no arm above chance, so the contrast cannot be "
                           "read as a mechanism statement)")
            elif r.get("accuracy_balance_failed"):
                verdict = ("INELIGIBLE (held-out answer-balance premise "
                           "failed: accuracy is a class-prior reading on "
                           "this stream; the macro-F1/PR-AUC companions, "
                           "not accuracy, carry the claim)")
            elif r.get("stream_sign_conflict"):
                verdict = ("INELIGIBLE (seed-level effect sign disagrees "
                           "with the item-level paired sign on the shared "
                           "evaluation stream: the two summaries are not "
                           "measuring the same contrast)")
            elif r.get("curve_invalid_removes_support"):
                verdict = ("INELIGIBLE (supported effect disappears after "
                           "dropping curve-invalid seeds: the claim rests "
                           "on runs that did not learn)")
            elif r.get("coverage_gate_failed"):
                failures = "; ".join(r.get("coverage_failures") or [])
                verdict = ("INELIGIBLE (design coverage gate failed: "
                           f"{failures}: the contrasted design was not "
                           "fully measured)")
            elif not r.get("claim_eligible", True):
                verdict = (f"INELIGIBLE (n={r['n']} < pre-registered "
                           f"minimum {self.min_seeds} seeds: too few "
                           f"independent runs for a claim)")
            elif equiv and r.get("equivalent") and r["fdr_reject"]:
                verdict = "EQUIVALENT (CI inside bound, null prediction holds)"
            elif noninf and r.get("noninferior") and r["fdr_reject"]:
                verdict = ("NON-INFERIOR (CI lower bound above the negative "
                           "bound: no material loss, null prediction holds)")
            elif null_test and sesoi is not None and (
                    (equiv and (r["lo"] > float(sesoi)
                                or r["hi"] < -float(sesoi)))
                    or (noninf and r["lo"] <= -float(sesoi))):
                verdict = ("REFUTED (CI entirely beyond the "
                           f"{'equivalence' if equiv else 'non-inferiority'} "
                           "bound, null prediction fails)")
            elif not r["fdr_reject"]:
                verdict = ("not significant (TOST)" if equiv
                           else ("not significant (one-sided NI)" if noninf
                                 else "not significant"))
            elif equiv:
                verdict = ("TOST-significant but CI not fully inside bound "
                           "(borderline, not read as support)")
            elif noninf:
                verdict = ("NI false positive-guard: CI lower bound not above "
                           "the negative bound (borderline, not read as "
                           "support)")
            elif r.get("direction_reversed"):
                verdict = ("REFUTED (FDR-significant effect opposite to the "
                           "pre-registered direction, no support)")
            elif not r["practically_relevant"]:
                verdict = ("FDR-significant but |effect| < SESOI "
                           "(practically negligible)")
            else:
                verdict = "SURVIVES"
            need = r.get("n_seeds_for_sesoi", float("inf"))
            need_s = ""
            if math.isfinite(need) and r["n"] > 0 and need > r["n"]:
                need_s = (f", needs ~{math.ceil(need)} seeds for "
                          f"{'the bound' if null_test else 'SESOI'} at 80% "
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
            a_mean, a_std = r.get("a_mean"), r.get("a_std")
            b_mean, b_std = r.get("b_mean"), r.get("b_std")
            if _finite(a_mean) and _finite(b_mean):
                sd = r.get("sd")
                if _finite(sd) and sd > 0.0 and _finite(r["mean"]):
                    spread = ("|effect| > paired sd (gain exceeds the paired "
                              "seed-to-seed spread)"
                              if abs(r["mean"]) > sd else
                              "|effect| <= paired sd (gain is within the "
                              "paired seed-to-seed spread)")
                    var_read = f"paired sd={sd:.4f}; {spread}"
                elif _finite(sd):
                    var_read = ("paired diffs identical across seeds "
                                "(paired sd=0)")
                else:
                    var_read = "paired sd n/a"
                std_a = f"{a_std:.4f}" if _finite(a_std) else "n/a"
                std_b = f"{b_std:.4f}" if _finite(b_std) else "n/a"
                note = r.get("arms_note")
                label = f"arms ({note})" if note else "arms (mean±std)"
                print(f"      {label}: a={a_mean:.4f}±{std_a} "
                      f"(n={r.get('a_n')}) vs b={b_mean:.4f}±{std_b} "
                      f"(n={r.get('b_n')}); {var_read}")
        if audited:
            print(f"[{self.tag}] controlled-variable audit: "
                  f"{len(controlled_pass)}/{len(audited)} contrasts pass "
                  f"(paired seeds + identical training recipe across sides + "
                  f"a declared model factor varies)")
        if confounded:
            print(f"[{self.tag}] WARNING: {len(confounded)} contrast(s) "
                  f"failed the controlled-variable audit and were refused "
                  f"regardless of their p-values")
        if reversed_effects:
            names = ", ".join(f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                              for r in reversed_effects)
            print(f"[{self.tag}] WARNING: {len(reversed_effects)} "
                  f"FDR-significant contrast(s) point in the opposite "
                  f"direction of their pre-registered prediction ({names}): "
                  f"each is read as a refutation, not as support, and never "
                  f"enters the supporting set")
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
        if noninf_rows:
            print(f"[{self.tag}] non-inferiority (one-sided null-prediction) "
                  f"tests: {len(noninferior)}/{len(noninf_rows)} have their "
                  f"95% CI lower bound above the negative pre-registered "
                  f"bound")
        if noninf_borderline:
            print(f"[{self.tag}] WARNING: {len(noninf_borderline)} "
                  f"non-inferiority contrast(s) pass the one-sided FDR gate "
                  f"but their CI lower bound is not above the negative bound "
                  f"(borderline); they do NOT count as confirming the null "
                  f"prediction")
        competence_failed = [r for r in testable
                             if r.get("competence_passed") is False]
        if self.competence is not None and not self.competence.get("passed"):
            print(f"[{self.tag}] WARNING: HB0 competence prerequisite "
                  f"FAILED: no trained arm has a seed-clustered 95% lower "
                  f"bound above chance {self.competence.get('chance', float('nan')):.4g} "
                  f"(best: {self.competence.get('best_arm') or 'none'}), so "
                  f"the grid did not demonstrably solve the task at this "
                  f"step count: all {len(competence_failed)} contrast(s) are "
                  f"marked claim-ineligible and may not be read as mechanism "
                  f"statements; raise the step count and "
                  f"re-run before interpreting any contrast")
        stream_conflict = [r for r in testable
                           if r.get("stream_sign_conflict")]
        if stream_conflict:
            names = ", ".join(
                f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                for r in stream_conflict)
            print(f"[{self.tag}] WARNING: {len(stream_conflict)} "
                  f"contrast(s) have a seed-level effect whose sign "
                  f"disagrees with the item-level paired effect on the "
                  f"shared evaluation stream ({names}): marked "
                  f"claim-ineligible")
        stream_uncorroborated = [
            r for r in supporting
            if r.get("stream_direction_agrees") is not None
            and r.get("stream_seed_sign_consistent") is False]
        if stream_uncorroborated:
            print(f"[{self.tag}] WARNING: {len(stream_uncorroborated)} "
                  f"supported accuracy contrast(s) have per-seed item-level "
                  f"effects that do not all share the seed-level sign: the "
                  f"pooled McNemar p reuses the same sealed items across "
                  f"seeds and is descriptive only, so it cannot corroborate "
                  f"the claim; report the per-seed counts next to it")
        curve_invalid_gated = [r for r in testable
                               if r.get("curve_invalid_removes_support")]
        balance_gated = [r for r in testable
                         if r.get("accuracy_balance_failed")]
        if balance_gated:
            print(f"[{self.tag}] WARNING: {len(balance_gated)} accuracy "
                  f"contrast(s) are marked claim-ineligible because the "
                  f"held-out answer-balance audit failed; accuracy is a "
                  f"class-prior reading on that stream, so only the "
                  f"macro/PR-AUC companions can support a mechanism claim")
        low_seed = [r for r in testable
                    if r.get("seed_floor_failed")
                    and r.get("competence_passed") is not False
                    and not r.get("curve_invalid_removes_support")
                    and not r.get("coverage_gate_failed")
                    and not r.get("accuracy_balance_failed")
                    and not r.get("stream_sign_conflict")]
        if low_seed:
            print(f"[{self.tag}] WARNING: {len(low_seed)}/{len(testable)} "
                  f"testable contrast(s) rest on fewer than "
                  f"{self.min_seeds} independent seeds and are marked "
                  f"claim-ineligible: the pre-registered minimum for a "
                  f"seed-level claim is {self.min_seeds} seeds, so their "
                  f"p-values cannot support a decision (replicated runs "
                  f"with >=3 seeds needed)")
        coverage_failed = [r for r in testable
                           if r.get("coverage_gate_failed")]
        if coverage_failed:
            print(f"[{self.tag}] WARNING: {len(coverage_failed)}/"
                  f"{len(testable)} testable contrast(s) are marked "
                  f"claim-ineligible because the design coverage gate "
                  f"failed (missing or seed-unbalanced cells): the effect "
                  f"was not measured on the full planned grid")
        if curve_invalid_gated:
            names = ", ".join(
                f"{r['hypothesis']}/{r['rule']}" for r in curve_invalid_gated)
            print(f"[{self.tag}] WARNING: {len(curve_invalid_gated)} "
                  f"supported contrast(s) lose support when curve-invalid "
                  f"seeds are dropped ({names}): they are marked "
                  f"claim-ineligible, because a difference produced by runs "
                  f"that never learned is not measured evidence for the "
                  f"mechanism")
        coverage_loss = [r for r in self.rows if r.get("seeds_dropped")]
        unpaired_rows = [r for r in coverage_loss if r.get("seeds_unpaired")]
        undefined_rows = [r for r in coverage_loss if r.get("seeds_undefined")]
        if coverage_loss:
            weakened = [r for r in coverage_loss
                        if (r.get("n") or 0) < self.min_seeds]
            parts = []
            if unpaired_rows:
                parts.append(f"{len(unpaired_rows)} with a value present on "
                             f"only one side (pairing failure)")
            if undefined_rows:
                parts.append(f"{len(undefined_rows)} where the metric is "
                             f"undefined on both sides (e.g. no "
                             f"false-concentration cases)")
            print(f"[{self.tag}] WARNING: {len(coverage_loss)} contrast(s) "
                  f"could not use the full planned seed set: "
                  f"{'; '.join(parts) or 'seed coverage loss'}; "
                  f"{len(weakened)} fell below the {self.min_seeds}-seed "
                  f"claim floor; seeds_used/seeds_dropped are recorded per "
                  f"row for the audit")
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
                   "n_direction_reversed": len(reversed_effects),
                   "n_equivalence": len(equiv_rows),
                   "n_equivalent": len(equivalent),
                   "n_equiv_borderline": len(equiv_borderline),
                   "n_noninferiority": len(noninf_rows),
                   "n_noninferior": len(noninferior),
                   "n_noninf_borderline": len(noninf_borderline),
                   "n_confounded": len(confounded),
                   "n_audited": len(audited),
                   "n_control_passed": len(controlled_pass),
                   "n_low_seed": len(low_seed),
                   "n_curve_invalid_gated": len(curve_invalid_gated),
                   "n_claim_ineligible_competence": len(competence_failed),
                   "n_claim_ineligible": len(
                       [r for r in testable
                        if not r.get("claim_eligible", True)]),
                   "competence": self.competence,
                   "n_seed_coverage_loss": len(coverage_loss),
                   "n_seed_unpaired": len(unpaired_rows),
                   "n_seed_undefined": len(undefined_rows),
                   "min_seeds_for_claim": self.min_seeds,
                   "n_loo_flip_decision": len(loo_flips),
                   "n_bca_disagree": len(bca_differs),
                   "n_low_resolution": len(low_resolution),
                   "direction_reversed_contrasts": [
                       f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                       for r in reversed_effects],
                   "curve_invalid_gated_contrasts": [
                       f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                       for r in curve_invalid_gated],
                   "seed_coverage_loss_contrasts": [
                       f"{r['hypothesis']}/{r['rule']} ({r['metric']})"
                       for r in coverage_loss],
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
    by_layer = case.get("attn_mass_on_key_by_layer") or []
    profile = ""
    if by_layer:
        profile = " mass_by_layer=[" + ",".join(
            f"{v:.2f}" for v in by_layer) + "]"
    return (f"pair#{case['key_pair_idx']} key={case['query_key']} "
            f"ans={case['answer']} pred={case['pred']} "
            f"conf={case['confidence']:.2f} "
            f"mass_key={case['attn_mass_on_key']:.2f}{profile} "
            f"type={case['error_type']} "
            f"flags=[{','.join(flags) or 'plain'}] pairs=[{pairs}]")


def _print_conf_by_error_type(named_runs, title="[e1]"):
    """Mean confidence inside each error bucket, per arm / grid cell.

    The bucket shares say what failed; this table says whether the model
    was confident while failing.  A high-confidence ``retrieval_miss`` is a
    systematic blind spot worth a mechanism change, while a low-confidence
    ``hard_structural`` error is ambiguity the task imposes and no
    architecture fix will remove.
    """
    print(f"\n{title} mean answer confidence inside each error bucket "
          f"(high = confident blind spot; low = ambiguous sample); 'n/a' "
          f"where the bucket is empty:")
    width = max(16, max(len(t) for t in _ERROR_TYPES) + 1)
    print(f"{'arm':>{width}}" + "".join(f"{t:>{width}}" for t in _ERROR_TYPES))
    for label, rs in named_runs:
        cells = []
        for etype in _ERROR_TYPES:
            vals = [
                float((r.get("err_conf_by_error_type") or {}).get(etype))
                for r in rs
                if _finite((r.get("err_conf_by_error_type") or {}).get(etype))]
            cells.append(f"{sum(vals) / len(vals):.3f}"
                         if vals else "n/a")
        print(f"{label:>{width}}" + "".join(f"{c:>{width}}" for c in cells))


def _print_structural_decomposition(named_runs, title="[e1]"):
    """Disjoint decomposition of the ``hard_structural`` error bucket.

    ``naive_confusable`` (the query token also appears as a later value)
    and ``value_repeats`` (the answer value occurs more than once) are
    overlapping task properties; the primary taxonomy keeps their union as
    ``hard_structural``.  This companion reports the disjoint naive-only /
    repeat-only / both shares so one source cannot hide behind the other.
    """
    print(f"\n{title} hard-structural decomposition (share of errors; "
          f"disjoint components of the unchanged ``hard_structural`` "
          f"bucket):")
    print(f"{'arm':>23} {'naive_only':>12} {'repeat_only':>12} "
          f"{'both':>12}")
    for label, rs in named_runs:
        cells = []
        for key in ("naive_only", "repeat_only", "both"):
            agg = _mean_std([
                r.get(f"err_taxonomy_hard_structural_{key}") for r in rs])
            cells.append(f"{agg['mean']:.3f}" if _finite(agg["mean"])
                         else "n/a")
        print(f"{str(label):>23} " + " ".join(
            f"{cell:>12}" for cell in cells))


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
        direction = h.get("direction") or {}
        if directional:
            terms = []
            for k, v in directional.items():
                sign = direction.get(k)
                if sign == "+":
                    terms.append(f"{k} >= +{v:g}")
                elif sign == "-":
                    terms.append(f"{k} <= -{v:g}")
                else:
                    terms.append(f"|{k}| >= {v:g} (undirected)")
            print(f"    SESOI (smallest effect size of interest): "
                  f"{', '.join(terms)} - statistical significance below "
                  f"this is reported as practically negligible, not as "
                  f"support; a significant effect opposite to a registered "
                  f"direction is a refutation, whatever its size")
        for k in sorted(equiv):
            if k not in sesoi:
                continue
            if equiv[k] == "noninferior":
                print(f"    non-inferiority bound (one-sided null "
                      f"prediction, bootstrap): {k} >= -{sesoi[k]:g} - "
                      f"supported only when the 95% CI's lower bound lies "
                      f"above the negative bound; a CI entirely below it is "
                      f"a refutation, not an inconclusive result")
            else:
                print(f"    equivalence bound (null prediction, bootstrap "
                      f"TOST): |{k}| <= {sesoi[k]:g} - supported only when "
                      f"the 95% CI of the effect lies inside this bound; a "
                      f"CI that merely includes 0 is not enough")


def _run_event(settings: Settings, experiment: str, event: str, **details) -> None:
    """Append one lifecycle row to ``run_index.jsonl``.

    A results directory may accumulate many invocations (baselines, sanity,
    E1..E3, retries).  The index records started/completed/aborted/failed
    per invocation with the same ``run_id`` as its manifest and records, so
    a half-finished grid can be told apart from a complete one; metadata
    write failures are warnings rather than experiment failures.
    """
    try:
        try:
            ctx = _run_context(settings)
        except Exception as e:  # noqa: BLE001 - keep the failure event
            ctx = _fallback_context(e)
        record = {
            "run_id": ctx["run_id"],
            "schema_version": ctx["schema_version"],
            "run_started_utc": ctx["run_started_utc"],
            "experiment": experiment,
            "event": event,
            "git_sha": ctx["git_sha"],
            "git_dirty": ctx["git_dirty"],
            "source_tree_sha256": ctx["source_tree_sha256"],
            "setup_sha256": ctx["setup_sha256"],
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **details,
        }
        path = pathlib.Path(settings.out_dir) / "run_index.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Append directly rather than through ``_write_result``: a fallback
        # context has no setup hash, and this lifecycle row must still be
        # written when the environment probe itself failed.
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False,
                                sort_keys=True) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:  # noqa: PERF203 - durability is best-effort
                pass
        _register_artifact(path)
    except Exception as e:  # noqa: BLE001 - tracking must not kill a run
        print(f"[runner] WARNING: could not write run_index event "
              f"{event}: {e}")


_RECORD_PROVENANCE_KEYS = (
    "run_id", "schema_version", "setup_sha256", "git_sha",
    "source_tree_sha256",
)


class TrackingError(RuntimeError):
    """A result row cannot be joined to its run manifest / reproducible setup."""


def _validate_result_record(record) -> None:
    """Refuse to persist a row without the provenance needed to trace it.

    Every result row must be joinable to the run manifest that describes the
    code revision, resolved config, seeds and data fingerprints behind it.
    The runners pass ``**_run_context`` (full rows) or ``**_stamp`` (auxiliary
    error-case rows); a future call site that forgets either would otherwise
    append an untraceable number to the results file, silently violating the
    "every run records code/config/seed/data/curve/result" contract.  The
    check is deliberately limited to the join keys, so auxiliary rows with
    different payloads remain valid.
    """
    if not isinstance(record, dict):
        raise TrackingError(
            f"result record must be a dict, got {type(record).__name__}")
    missing = [key for key in _RECORD_PROVENANCE_KEYS
               if record.get(key) in (None, "")]
    if missing:
        raise TrackingError(
            f"result record is missing provenance {missing}; stamp it with "
            f"**_run_context(settings) or **_stamp(settings) so it can be "
            f"joined to its manifest")
    try:
        json.dumps(record, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TrackingError(
            f"result record is not JSON-serializable: {exc}") from exc


def _write_result(path: pathlib.Path, record: dict) -> None:
    """Append one JSONL run record, stamped with its creation time.

    The timestamp is injected here rather than by each runner, so every
    record on disk carries "when" next to the ``git_sha`` / ``setup_sha256``
    provenance it already stores inline.  The write is refused (instead of
    appending an orphan row) when the provenance join keys are absent.
    """
    _validate_result_record(record)
    record.setdefault("created_utc",
                      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:  # noqa: PERF203 - durability is best-effort
            pass
    _register_artifact(path)


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    """Write ``text`` to ``path`` without exposing a half-written file.

    Result and index files are evidence: a crash or full disk in the middle
    of a plain ``write_text`` leaves a truncated JSON document that looks
    like a corrupted run.  Writing a sibling temporary file and renaming it
    over the target makes each artifact appear atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:  # noqa: PERF203 - durability is best-effort
                pass
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def _write_json(path: pathlib.Path, payload) -> None:
    _atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2,
                         sort_keys=True))
    _register_artifact(path)


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _git_sha() -> str:
    """Commit hash of the repository that contains this package.

    The probe is anchored at the package root instead of the process CWD:
    a run launched from another checkout (or from a subdirectory that is
    itself inside a different repository) would otherwise stamp a
    ``git_sha`` that does not identify the code that produced the record.
    """
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=str(_REPO_ROOT),
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    """Whether the working tree differs from the recorded revision.

    A result produced from a dirty tree cannot be rebuilt from its
    ``git_sha`` alone, so every run record says so explicitly instead of
    silently attributing the numbers to the committed revision.  The
    status is read from the package's repository root for the same reason
    as :func:`_git_sha`; a non-repository location fails closed (dirty).
    """
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             cwd=str(_REPO_ROOT),
                             capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except Exception:
        return True


_SOURCE_TREE_SHA: str | None = None
_RUN_ID: str | None = None
_RUN_STARTED_UTC: str | None = None
_RECORD_SCHEMA = 1
_RUN_ARTIFACTS: set[pathlib.Path] = set()


def _register_artifact(path) -> None:
    """Remember a file written by this invocation for the artifact index."""
    try:
        _RUN_ARTIFACTS.add(pathlib.Path(path).resolve())
    except Exception:  # noqa: BLE001 - tracking must not kill a run
        pass


def _fallback_context(error: Exception) -> dict:
    """Minimal provenance when the full context itself failed.

    Setup problems (e.g. asking for CUDA on a CPU-only build) can make
    ``_run_context`` raise before any run event is written.  Recording a
    minimal block keeps the failure traceable instead of losing it just
    because the environment probe failed too.
    """
    run_id, run_started_utc = _run_identity()
    return {
        "run_id": run_id,
        "schema_version": _RECORD_SCHEMA,
        "run_started_utc": run_started_utc,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "source_tree_sha256": _source_tree_sha256(),
        "setup_sha256": None,
        "context_error": f"{type(error).__name__}: {error}",
    }


def _read_artifact_entries(index_path: pathlib.Path) -> tuple[dict, str | None]:
    """Read the entries of an existing index, or return no entries + why.

    A malformed index is not silently treated as empty: the new index keeps
    the ``previous_index_error`` so a reader knows that artifacts from
    earlier invocations are no longer covered.  The previous list is
    dropped as a whole rather than partially trusted - a partially readable
    integrity list is not a certificate.
    """
    if not index_path.exists():
        return {}, None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("index must be a JSON object")
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("artifacts must be a list")
        entries = {}
        for entry in artifacts:
            if not isinstance(entry, dict):
                raise ValueError("every artifact entry must be an object")
            name = entry.get("file")
            if not isinstance(name, str) or not name:
                raise ValueError("every artifact entry needs a non-empty file")
            if name in entries:
                # A duplicated entry is not a partially trustworthy list:
                # silently keeping the last copy would let the next index
                # write launder a state that ``_verify_artifacts`` rejects.
                raise ValueError(
                    f"duplicate artifact entry for file {name!r}")
            entries[name] = entry
        return entries, None
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def _write_artifact_index(out_dir: pathlib.Path, settings: Settings,
                          experiment: str, completed: bool,
                          elapsed_seconds: float) -> dict:
    """Hash every artifact written so far into one cumulative index file.

    A results directory is append-only across invocations and can be
    truncated or edited after the fact.  The index records the size, line
    count and SHA-256 of each file written by *this* run **and keeps the
    entries of earlier runs that this invocation did not rewrite**, so
    ``--verify-artifacts`` certifies the whole directory rather than only
    its last invocation.  A rewritten/append-appended file replaces its own
    entry (the current hash always wins).  The index itself is written last
    and is not listed.

    Entries are stored with paths relative to ``out_dir`` so artifacts in
    nested directories are covered too.  A previous index that cannot be
    parsed is reported in ``previous_index_error`` instead of being
    mistaken for an empty directory.
    """
    root = pathlib.Path(out_dir).resolve()
    previous, previous_error = _read_artifact_entries(
        root / "artifact_index.json")
    current = {}
    for path in sorted(_RUN_ARTIFACTS):
        try:
            relative = path.resolve().relative_to(root)
        except (OSError, RuntimeError, ValueError):
            continue
        if not path.is_file():
            continue
        data = path.read_bytes()
        entry = {
            "file": relative.as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        if path.suffix == ".jsonl":
            entry["records"] = data.count(b"\n")
        current[entry["file"]] = entry
    merged = {**previous, **current}
    entries = [merged[name] for name in sorted(merged)]
    try:
        context = _run_context(settings)
    except Exception as e:  # noqa: BLE001 - an index must still be written
        context = _fallback_context(e)
    payload = {
        "experiment": experiment,
        "completed": bool(completed),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "artifact_count": len(entries),
        "current_artifact_count": len(current),
        "updated_entries": sorted(current),
        "artifacts": entries,
    }
    if previous_error is not None:
        payload["previous_index_error"] = previous_error
    payload.update(context)
    _atomic_write_text(
        root / "artifact_index.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def _verify_artifacts(out_dir) -> int:
    """Check a results directory against its cumulative artifact index.

    Exit 0 when every listed file exists with the recorded size and hash,
    1 when any file is missing/changed or an entry is malformed, 2 when no
    usable index exists.  Because the index is cumulative, a file from an
    earlier invocation is checked too; a missing entry therefore means the
    results directory was truncated after that invocation.
    """
    root = pathlib.Path(out_dir).resolve()
    index_path = root / "artifact_index.json"
    if not index_path.exists():
        print(f"[verify] no artifact index at {index_path}; nothing to check")
        return 2
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("index must be a JSON object")
        artifacts = payload["artifacts"]
        if not isinstance(artifacts, list):
            raise ValueError("artifacts must be a list")
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as e:
        print(f"[verify] unreadable artifact index {index_path}: {e}")
        return 2
    failures = []
    seen = set()
    for entry in artifacts:
        if not isinstance(entry, dict):
            failures.append(f"<malformed entry>: not an object")
            continue
        name = entry.get("file")
        if not isinstance(name, str) or not name:
            failures.append("<malformed entry>: missing non-empty file")
            continue
        if name in seen:
            failures.append(f"{name}: duplicate entry in index")
            continue
        seen.add(name)
        try:
            path = (root / name).resolve()
            path.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            failures.append(f"{name}: outside results directory")
            continue
        if not path.is_file():
            failures.append(f"{name}: missing")
            continue
        if isinstance(entry.get("bytes"), bool) \
                or not isinstance(entry.get("bytes"), int) \
                or entry.get("bytes") < 0:
            failures.append(f"{name}: invalid byte count")
            continue
        digest_expected = entry.get("sha256")
        if not isinstance(digest_expected, str) \
                or len(digest_expected) != 64 \
                or any(c not in "0123456789abcdef" for c in digest_expected):
            failures.append(f"{name}: invalid sha256")
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            failures.append(f"{name}: unreadable ({exc})")
            continue
        if len(data) != entry["bytes"]:
            failures.append(f"{name}: size {len(data)} != "
                            f"{entry['bytes']}")
        digest = hashlib.sha256(data).hexdigest()
        if digest != digest_expected:
            failures.append(f"{name}: sha256 {digest[:12]} != "
                            f"{digest_expected[:12]}")
        lines = data.count(b"\n")
        if path.suffix == ".jsonl":
            expected_lines = entry.get("records")
            if isinstance(expected_lines, bool) \
                    or not isinstance(expected_lines, int) \
                    or expected_lines < 0:
                failures.append(f"{name}: invalid line count")
            elif lines != expected_lines:
                failures.append(f"{name}: line count {lines} != "
                                f"{expected_lines}")
    if failures:
        print(f"[verify] FAILED ({len(failures)} issue(s)) for run "
              f"{payload.get('run_id')} in {root}:")
        for failure in failures:
            print(f"    {failure}")
        return 1
    print(f"[verify] ok: {len(artifacts)} artifact(s) match run "
          f"{payload.get('run_id')} (completed={payload.get('completed')}, "
          f"schema={payload.get('schema_version')}, "
          f"current={payload.get('current_artifact_count', 'n/a')})")
    return 0


def _source_tree_sha256() -> str:
    """Content hash of the training package (cage/*.py).

    ``git_sha`` identifies the committed revision, but a run from a dirty
    tree is not rebuildable from that revision alone and the dirty flag does
    not say *what* code ran.  Hashing the actual source files gives every
    record a content-addressed code version that survives uncommitted
    edits, so two runs can be compared (and a stale baseline verdict can be
    refused) even when both report the same ``git_sha``.
    """
    global _SOURCE_TREE_SHA
    if _SOURCE_TREE_SHA is None:
        h = hashlib.sha256()
        for path in sorted(pathlib.Path(__file__).resolve().parent.glob("*.py")):
            h.update(path.name.encode("utf-8") + b"\0")
            h.update(path.read_bytes())
            h.update(b"\0")
        _SOURCE_TREE_SHA = h.hexdigest()[:16]
    return _SOURCE_TREE_SHA


def _run_identity() -> tuple[str, str]:
    """Per-process run id and start timestamp (excluded from setup hash).

    One invocation can write many JSONL records; the run id groups them and
    links the compact provenance stamp on error-case rows back to the full
    manifest, so stale rows from an earlier invocation are never mistaken
    for part of the current one.
    """
    global _RUN_ID, _RUN_STARTED_UTC
    if _RUN_ID is None:
        _RUN_ID = (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                   + "-" + uuid.uuid4().hex[:8])
        _RUN_STARTED_UTC = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return _RUN_ID, _RUN_STARTED_UTC


def _hypotheses_sha() -> str:
    """Content hash of the pre-registered hypothesis registry.

    Records carry the hash so a result can be matched to the exact
    decision rules in force when it was produced: if the registry is
    edited mid-analysis, old and new records can no longer be silently
    pooled under "the same" hypotheses.
    """
    payload = json.dumps(HYPOTHESES, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _environment_record(settings: Settings, git_dirty: bool | None = None) -> dict:
    """Software/hardware provenance for reproducibility.

    Beyond versions the record pins the numerical-behaviour switches a
    rerun must match: CPU thread count and the BLAS/OpenMP environment
    overrides that determine it (parallel reductions change the summation
    order), the CPU model and OS release, cuDNN determinism/benchmark,
    whether deterministic algorithms are on, the cuBLAS workspace config,
    the torch build identity/backends and the NumPy version when installed.
    The determinism switch is applied from ``Settings.deterministic``
    *before* it is read, so the recorded state is the one this run will
    train under even when the context is built before the first training
    call (a previous run in the same process may have left the global flag
    in the other position).
    """
    determinism = configure_determinism(settings.deterministic)
    numeric_env = {}
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "CUBLAS_WORKSPACE_CONFIG"):
        numeric_env[name] = os.environ.get(name)
    env = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "torch": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "torch_debug": bool(getattr(torch.version, "debug", False)),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "os_release": platform.release(),
        "device": settings.device,
        "cpu_count": os.cpu_count(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "torch_num_threads": determinism["torch_num_threads"],
        "deterministic_requested": bool(settings.deterministic),
        "deterministic_algorithms": determinism["deterministic_algorithms"],
        "cublas_workspace_config": determinism["cublas_workspace_config"],
        "tf32": determinism["tf32"],
        # Thread-count overrides are read by the BLAS/OpenMP runtimes before
        # torch is imported, so they are part of the numerical recipe even
        # though ``torch_num_threads`` records the resulting value.
        "numeric_env": numeric_env,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "hash_randomization": bool(sys.flags.hash_randomization),
        # How the run was launched is part of the recipe: the same config
        # can mean different code paths through CLI flags, and a result is
        # only rebuildable when its invocation is recorded.
        "argv": [str(arg) for arg in sys.argv],
        "cwd": os.getcwd(),
        "repo_root": str(_REPO_ROOT),
        "git_dirty": (_git_dirty() if git_dirty is None else bool(git_dirty)),
    }
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    env["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        env.setdefault("cpu_model", None)
    for name in ("mkldnn", "openmp"):
        backend = getattr(torch.backends, name, None)
        probe = getattr(backend, "is_available", None)
        try:
            env[f"torch_{name}"] = bool(probe()) if probe is not None else None
        except Exception:  # noqa: BLE001 - probe must not fail a record
            env[f"torch_{name}"] = None
    try:
        env["torch_num_interop_threads"] = int(torch.get_num_interop_threads())
    except Exception:  # noqa: BLE001 - older/limited builds may not expose it
        env["torch_num_interop_threads"] = None
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
    if torch.cuda.is_available():
        try:
            count = int(torch.cuda.device_count())
            devices = []
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "index": i,
                    "name": props.name,
                    "capability": f"{props.major}.{props.minor}",
                    "total_memory_bytes": int(props.total_memory),
                })
            env["cuda_device_count"] = count
            env["cuda_devices"] = devices
        except Exception as exc:  # noqa: BLE001 - probe must not fail a record
            env["cuda_devices_error"] = f"{type(exc).__name__}: {exc}"
    return env


_SEEDING_POLICY = {
    "every_training_run": (
        "cage.repro.seed_everything(seed): python random + numpy (when "
        "installed) + torch CPU and all CUDA devices; deterministic kernels "
        "are requested by default (Settings.deterministic=True)"),
    "seed_canonicalisation": (
        "every RNG (Python, NumPy, torch, CUDA) and every data-stream "
        "generator consumes the same low-32-bit seed key, so s and "
        "s + 2**32 are the same run; bool/fractional seeds are rejected "
        "instead of silently truncated"),
    "determinism_switch": (
        "Settings.deterministic is applied by every seeding call, including "
        "train_model; it defaults to True and can be opted out via "
        "--no-deterministic; False explicitly restores non-deterministic "
        "kernels so a prior deterministic run cannot leak into a later one"),
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


def _config_cache_key(settings: Settings) -> str:
    """Content key for per-``Settings`` caches.

    The runners mutate ``Settings`` in place (``--seeds``, CLI overrides,
    verdict marks), so object identity alone must never decide whether a
    cached audit is still valid: a mutated config would otherwise reuse
    results computed for a different recipe and silently certify the wrong
    setup.  Every cache below stores this key beside its value.
    """
    return json.dumps(
        {name: getattr(settings, name)
         for name in Settings.__dataclass_fields__ if name != "out_dir"},
        ensure_ascii=False, sort_keys=True, default=str)


def _run_context(settings: Settings) -> dict:
    """Self-contained provenance block attached to every result record.

    Code revision, hypothesis-registry hash, resolved configuration (every
    hyperparameter and seed actually used), software environment, the
    seeding policy and the data-stream fingerprints travel *with* each
    record, so a result file is traceable without relying on defaults that
    may have changed since.  The ``setup_sha256`` hash over exactly
    {revision, source-tree hash, hypotheses, config, data fingerprints}
    gives every record a compact id for "same code + same recipe + same
    data": records carrying different ids were not produced by the same
    setup and cannot be pooled silently.  ``run_id`` groups all records of
    one invocation.  Cached on the Settings object and invalidated when the
    resolved config changes, so a mutated Settings cannot reuse a stale
    provenance block.

    The configured training-seed set is re-validated on every call (cheap,
    before the cache lookup): every official entry point builds the run
    context before training, so a seed set mutated after construction -
    the runner applies ``--seeds`` that way - is still refused when it is
    not duplicate-free / disjoint from the reserved evaluation streams.
    """
    # Keep the validated/normalised tuple: a caller may have assigned a
    # one-shot iterable after construction, and validating it would consume
    # the iterator before the config/fingerprint loops below read it.
    settings.seeds = _validate_training_seeds(settings.seeds)
    config = {name: getattr(settings, name)
              for name in Settings.__dataclass_fields__ if name != "out_dir"}
    cache_key = _config_cache_key(settings)
    ctx = getattr(settings, "_run_context_cache", None)
    if ctx is not None and getattr(settings, "_run_context_cache_key", None) \
            == cache_key:
        return ctx
    # Fingerprint each stream at the geometry its consumer actually uses:
    # make_batch's draws are shape-dependent, so one seed at two batch
    # sizes is two realizations.  Training reads settings.batch; validation
    # monitoring reads settings.batch too (same fixed VAL stream), final
    # val/test read eval_batch, mechanism statistics read stats_batch, the
    # linear probe trains at probe_batch and selects/scores at
    # probe_eval_batch, and the overfit sanity check reads overfit_batch.
    fingerprints = {
        "val": stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.eval_batch, seed=VAL_SEED),
        "val_monitor": stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.batch, seed=VAL_SEED),
        "test": stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.eval_batch, seed=TEST_SEED),
        "stats": stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.stats_batch, seed=STATS_SEED),
        "probe_val": stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.probe_eval_batch, seed=VAL_SEED),
        "probe_test": stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.probe_eval_batch, seed=TEST_SEED),
        "overfit_sanity": stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.overfit_batch, seed=SANITY_SEED),
    }
    fingerprints["train"] = {
        str(seed): stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.batch, seed=seed)
        for seed in settings.seeds
    }
    fingerprints["probe_train"] = {
        str(seed): stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.probe_batch, seed=seed)
        for seed in settings.seeds
    }
    git_sha = _git_sha()
    git_dirty = _git_dirty()
    source_tree = _source_tree_sha256()
    run_id, run_started_utc = _run_identity()
    payload = json.dumps({
        "git_sha": git_sha,
        "source_tree_sha256": source_tree,
        "hypotheses_sha256": _hypotheses_sha(),
        "config": config,
        "data_fingerprints": fingerprints,
    }, ensure_ascii=False, sort_keys=True, default=str)
    ctx = {
        "run_id": run_id,
        "schema_version": _RECORD_SCHEMA,
        "run_started_utc": run_started_utc,
        "git_sha": git_sha,
        "git_dirty": bool(git_dirty),
        "source_tree_sha256": source_tree,
        "hypotheses_sha256": _hypotheses_sha(),
        "setup_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        "environment": _environment_record(settings, git_dirty),
        "config": config,
        "seeding": dict(_SEEDING_POLICY),
        "data_fingerprints": fingerprints,
        # Validity-gate summary stamped by ``_enforce_baseline_verdict``
        # (None for the baselines runner itself, which is the producer).
        # Deliberately excluded from the setup hash above: the verdict
        # describes whether this protocol passed its gates, not which
        # protocol it is.
        "baseline_validity": getattr(settings, "_baseline_validity", None),
    }
    settings._run_context_cache = ctx
    settings._run_context_cache_key = cache_key
    return ctx


def _stamp(settings: Settings) -> dict:
    """Compact provenance stamp for auxiliary/error-case records.

    Full provenance lives in the run manifest; the stamp carries just the
    ids needed to join a raw case row back to it without repeating the
    environment and every hyperparameter on each row.
    """
    ctx = _run_context(settings)
    return {key: ctx[key] for key in
            ("run_id", "schema_version", "setup_sha256", "git_sha",
             "source_tree_sha256")}


def _protocol_sha256(ctx: dict) -> str:
    """Hash over exactly the resolved config and data-stream fingerprints.

    Excludes the code revision, hypothesis registry and environment on
    purpose: those do not change *which data* a check was run on.  The
    source-tree hash is checked separately by
    :func:`_verdict_provenance_note`, because the gate *implementation*
    must also match.  Any config or stream-fingerprint change invalidates
    the stored HB marks.
    """
    payload = json.dumps({"config": ctx["config"],
                          "data_fingerprints": ctx["data_fingerprints"]},
                         ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _verdict_provenance_note(verdict: dict, settings: Settings) -> str | None:
    """Explain why a stored baselines verdict may not apply to this run.

    A verdict written under a different config/data protocol (or before
    the stamp existed) cannot annotate the current grids: its HB1/HB2/HB4/
    HB5/HB6/HB7 marks were measured on different streams or a different
    recipe.  The source-tree hash is checked separately: the gates are
    implemented in this package, so a verdict produced by older code is not
    evidence about the current code even when config and data match (a
    dirty tree with the same ``git_sha`` makes that distinction necessary).
    """
    stored = verdict.get("protocol_sha256")
    if stored is None:
        return ("baselines verdict predates protocol stamping; re-run "
                "`--only baselines` so its HB marks describe the current "
                "data/recipe")
    current = _protocol_sha256(_run_context(settings))
    if stored != current:
        return (f"baselines verdict protocol {stored} != current {current}; "
                f"it was produced under a different config/data protocol, so "
                f"its HB marks do not describe this run - re-run "
                f"`--only baselines`")
    stored_source = verdict.get("source_tree_sha256")
    current_source = _source_tree_sha256()
    if stored_source is None:
        return ("baselines verdict predates source-tree stamping; re-run "
                "`--only baselines` so its HB marks were produced by the "
                "current gate code")
    if stored_source != current_source:
        return (f"baselines verdict source {stored_source} != current "
                f"{current_source}; it was produced by older gate code, so "
                f"its HB marks do not certify this run - re-run "
                f"`--only baselines`")
    return None


def _write_manifest(settings: Settings, out_dir: pathlib.Path, tag: str) -> dict:
    """Per-experiment provenance manifest, written before any training.

    One ``<tag>_manifest.json`` per experiment pins the invocation run id,
    the code revision (plus dirty flag and content hash), the
    pre-registered hypotheses and their hash, every hyperparameter and
    seed, the data-stream fingerprints (including each training stream) and
    the software environment.  Run records repeat this context inline; the
    manifest is the single artifact that makes a results directory
    traceable back to a reproducible setup without joining files by hand.
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
    older version lacks the HB4/HB5 keys or the protocol/source stamps; the
    marks are reported as ``n/a`` (not ``FAILED``) and a provenance warning
    asks for a re-run, so neither a missing check nor a stale one is
    mistaken for a pass.
    """
    path = out_dir / "baselines_verdict.json"
    if not path.exists():
        return None
    try:
        verdict = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # A syntactically valid JSON array/scalar is not a verdict record; treat
    # it like a corrupt/missing file so the caller raises the documented
    # BaselineVerdictError instead of failing on ``.get``.
    return verdict if isinstance(verdict, dict) else None


class BaselineVerdictError(RuntimeError):
    """A baseline validity gate failed (or its verdict is missing/stale).

    HB1 (task construction), HB3 (split disjointness), HB4 (label-leak
    negative control) and HB5 (training loop) pre-register that their
    failure invalidates every result above them.  The E1/E2/E3 runners
    enforce that semantics by refusing to train on a protocol whose
    verdict is absent, stale, or failed - the same hard-gate policy as
    the reproducibility and overfit pre-flights.
    """


# Validity gates whose failure invalidates every grid result (the
# pre-registered "invalidate everything above it" semantics of HB1/HB3,
# plus the loop gate).  HB2 is deliberately not in this set: its
# registered semantics invalidate the *simple-baseline comparison*, so its
# state is stamped and displayed (certified/inconclusive/violated) instead
# of blocking the grid.  HB4 is also special-cased below: only a
# *violated* negative control blocks, while an inconclusive one is
# stamped (with a finite stream, "not shown" is not "leakage shown").
_BLOCKING_VERDICT_GATES = ("hb1_supported", "hb3_supported",
                           "hb5_supported")


def _verdict_seed_floor(verdict: dict, settings: Settings) -> dict:
    """The pre-registered seed floor as applied to the verdict's marks.

    ``min_seeds_for_claim`` (Settings, hashed) says a seed-level claim
    needs at least that many independent training seeds.  The baselines
    verdict applies the same floor to HB2/HB4/HB6/HB7: a supported mark
    resting on fewer seeds is *provisional* (displayed and stamped as
    such, never silently certified), matching the decision layer of the
    contrast families where a 2-seed "win" is marked INELIGIBLE.
    """
    floor = int(settings.min_seeds_for_claim)
    stored = verdict.get("seed_floor") or {}
    n_seeds = stored.get("n_seeds")
    if n_seeds is None:
        n_seeds = len(settings.seeds)

    def eligible(key: str, n) -> bool:
        if f"{key}_claim_eligible" in stored:
            return bool(stored[f"{key}_claim_eligible"])
        return bool(int(n) >= floor)

    out = {"min_seeds_for_claim": floor, "n_seeds": int(n_seeds)}
    for key, n in (("hb2", stored.get("hb2_n_seeds", n_seeds)),
                   ("hb4", stored.get("hb4_n_seeds", n_seeds)),
                   ("hb6", stored.get("hb6_n_seeds", n_seeds)),
                   ("hb7", stored.get("hb7_n_seeds", n_seeds))):
        out[f"{key}_n_seeds"] = int(n)
        out[f"{key}_claim_eligible"] = eligible(key, n)
    return out


def _enforce_baseline_verdict(settings: Settings, out_dir: pathlib.Path,
                              tag: str) -> dict:
    """Read and enforce ``baselines_verdict.json`` before a grid trains.

    The pre-registered semantics of HB1/HB3/HB5 are "failure invalidates
    everything above it", and HB4 blocks only in its *violated* state, so
    the E1/E2/E3 runners must not keep training and writing result files
    when those gates fail.  This gate:

    * refuses a missing verdict (``--only baselines`` has to be certified
      first) and a stale one (protocol hash or source-tree hash mismatch:
      the marks were measured on a different config/data protocol or by
      older gate code);
    * refuses when any blocking gate failed;
    * stamps the verdict's marks (HB1..HB7 + seed-floor eligibility) into
      ``settings._baseline_validity`` so every subsequent manifest and
      run record carries the validity status inline and a reader of the
      JSON alone can see which gates certified the numbers;
    * warns (without blocking) when a supported mark is only
      *provisional* because it rests on fewer than ``min_seeds_for_claim``
      seeds - the same floor the contrast families enforce.

    Returns the loaded verdict for downstream annotations (E1's marks,
    E3's HB7 context).
    """
    verdict = _load_baseline_verdict(out_dir)
    if verdict is None:
        raise BaselineVerdictError(
            f"{tag}: no baselines_verdict.json in {out_dir}; the HB1/HB3/"
            f"HB4/HB5 validity gates have not been certified for this "
            f"protocol, so no mechanism claim may be produced - run "
            f"`--only baselines` first")
    stale = _verdict_provenance_note(verdict, settings)
    if stale:
        raise BaselineVerdictError(
            f"{tag}: {stale}; refusing to train on an uncertified protocol")
    failed = [key for key in _BLOCKING_VERDICT_GATES
              if not verdict.get(key)]
    # HB4 is a three-state validity gate: only a *violated* control (the
    # whole interval above tolerance) invalidates the grid.  An
    # inconclusive interval must not be read as leakage - with a finite
    # test stream it can be underpowered - and is stamped for the reader.
    hb4_state = verdict.get("hb4_state")
    if hb4_state is None:
        if verdict.get("hb4_supported") is not True:
            failed.append("HB4 (missing/not certified)")
    elif hb4_state == "violated":
        failed.append("HB4 (violated: control interval above tolerance)")
    elif hb4_state == "untestable":
        failed.append("HB4 (untestable)")
    elif hb4_state not in ("supported", "inconclusive"):
        # A malformed/unknown state must fail closed: treating it as a
        # pass would let a hand-edited verdict bypass the negative-control
        # gate entirely.
        failed.append(f"HB4 (unrecognised state {hb4_state!r})")
    if failed:
        raise BaselineVerdictError(
            f"{tag}: baseline validity gate(s) {', '.join(failed)} failed "
            f"in baselines_verdict.json; their pre-registered semantics "
            f"invalidate every grid result above them, so the grid is not "
            f"started (fix the task/data/loop and re-run `--only "
            f"baselines`)")
    floor = _verdict_seed_floor(verdict, settings)
    marks = {
        "source": "baselines_verdict.json",
        "protocol_sha256": verdict.get("protocol_sha256"),
        "hb1_supported": verdict.get("hb1_supported"),
        "hb2_supported": verdict.get("hb2_supported"),
        "hb2_state": verdict.get("hb2_state"),
        "hb3_supported": verdict.get("hb3_supported"),
        "hb4_supported": verdict.get("hb4_supported"),
        "hb4_violated": verdict.get("hb4_violated"),
        "hb4_state": verdict.get("hb4_state"),
        "hb5_supported": verdict.get("hb5_supported"),
        "hb6_supported": verdict.get("hb6_supported"),
        "hb7_depth_required": verdict.get("hb7_depth_required"),
        "saturation_depth": verdict.get("saturation_depth"),
        "ladder_competent": verdict.get("ladder_competent"),
        # Held-out-stream balance stamp: ``None`` on an older verdict that
        # predates the audit, explicit bool otherwise.
        "eval_answer_balance_all_uniform": (
            verdict.get("eval_answer_balance") or {}).get("all_uniform"),
        "seed_floor": floor,
    }
    settings._baseline_validity = marks
    # Force the cached provenance block to be rebuilt so every manifest and
    # run record written from here on carries the marks inline.
    settings._run_context_cache = None
    _run_context(settings)
    provisional = [key for key in ("hb2", "hb4", "hb6", "hb7")
                   if verdict.get(f"{key}_supported"
                                  if key != "hb7" else "hb7_depth_required")
                   and not floor.get(f"{key}_claim_eligible", True)]
    if provisional:
        print(f"[{tag}] WARNING: baseline mark(s) "
              f"{', '.join(provisional)} are supported only provisionally: "
              f"they rest on {floor['n_seeds']} seed(s), below the "
              f"pre-registered claim floor {floor['min_seeds_for_claim']} "
              f"(Settings.min_seeds_for_claim); they are stamped and "
              f"displayed as provisional, and no seed-level claim may be "
              f"based on them")
    if marks["eval_answer_balance_all_uniform"] is False:
        bad = [label for label, row in
               ((verdict.get("eval_answer_balance") or {}).get("streams")
                or {}).items()
               if not row["uniformity"]["uniform_supported"]]
        print(f"[{tag}] WARNING: held-out answer balance failed on stream(s) "
              f"{', '.join(bad) or 'unknown'}: accuracy-based readings on "
              f"those streams are class-prior readings, so the F1/PR-AUC "
              f"companions carry the headline and the balance premise is "
              f"stamped as failed in the baseline validity marks")
    e2_balance = ((verdict.get("eval_answer_balance") or {})
                  .get("e2_length_sweep") or {})
    if e2_balance.get("endpoint_uniform") is False:
        bad_lengths = [
            str(length) for length in (e2_balance.get("endpoint_lengths")
                                       or [])
            if not ((e2_balance.get("streams") or {}).get(str(length)) or {})
            .get("uniformity", {}).get("uniform_supported")]
        print(f"[{tag}] WARNING: E2 length-sweep endpoint balance failed at "
              f"length(s) {', '.join(bad_lengths) or 'unknown'}: accuracy at "
              f"the base/longest endpoints is a class-prior reading, so the "
              f"length-generalisation accuracy contrasts cannot support a "
              f"claim")
    return verdict


def _competence_check(named_runs, correct_and_n, chance, what="arm"):
    """HB0 pre-registered competence prerequisite for a contrast family.

    A mechanism contrast (does the gate read entropy? does depth help?)
    is only interpretable when at least one trained condition demonstrably
    learned the task: an all-at-chance grid turns "no difference" into a
    statement about the step count, not the mechanism, and makes HB7 read
    "depth required" from pure underfit.  This is the one premise the
    registry lacked (HB0), so it is checked *before* any claim.

    Every run in an arm is one training seed evaluated on the *same*
    sealed test stream.  Pooling the seeds' correct counts through a Wilson
    interval treats those repeated evaluations as independent examples
    (pseudo-replication) and shrinks the interval by roughly
    ``sqrt(n_seeds)``.  We instead use the two-level (seed-clustered)
    bootstrap in :func:`clustered_proportion_ci`, which resamples seeds and
    keeps a shared finite-stream measurement error.  Because HB0 is a union
    claim over arms, the lower quantile is Bonferroni-adjusted by the number
    of arms before reading "at least one arm cleared chance"; the pooled
    Wilson interval is retained as a descriptive companion only.

    ``named_runs`` is an iterable of ``(name, runs)``; ``correct_and_n``
    maps one run to ``(n_correct, n_total)``.  Returns the payload written
    to ``<tag>_competence.json`` and handed to :class:`ContrastLog`, which
    marks every contrast ``claim_eligible=False`` when it fails.
    """
    raw = []
    for name, runs in named_runs:
        counts = []
        sizes = []
        for run in runs:
            pair = correct_and_n(run)
            if pair is None:
                continue
            n_correct, n_total = pair
            if n_correct is None or n_total is None or not int(n_total):
                continue
            counts.append(int(n_correct))
            sizes.append(int(n_total))
        raw.append((str(name), counts, sizes, sum(sizes)))
    n_arms = sum(1 for _, _, _, total in raw if total > 0)
    alpha_per_arm = 0.05 / max(n_arms, 1)
    arms = {}
    best = None
    for name, counts, sizes, total in raw:
        pooled = wilson_ci(sum(counts), total)
        cluster = clustered_proportion_ci(
            counts, sizes, lower_alpha=alpha_per_arm)
        above = bool(
            cluster["valid"] and _finite(cluster["lo"])
            and cluster["lo"] > float(chance))
        arms[name] = {
            "correct": sum(counts), "n": total,
            "accuracy": cluster["mean"] if cluster["valid"] else pooled["p"],
            "cluster_lo": cluster["lo"], "cluster_hi": cluster["hi"],
            "wilson_lo": pooled["lo"], "wilson_hi": pooled["hi"],
            "pooled_accuracy": pooled["p"], "above_chance": above,
            "n_units": cluster["n"], "se": cluster["se"],
            "valid": cluster["valid"], "reason": cluster["reason"],
        }
        if above and (best is None
                      or arms[name]["accuracy"] > arms[best]["accuracy"]):
            best = name
    return {
        "check": (f"at least one {what} has a seed-clustered one-sided "
                  f"95% family-wise lower bound above chance "
                  f"(Bonferroni over {n_arms} arms)"),
        "method": ("two-level bootstrap: resample training seeds, shared "
                   "binomial draw on the common evaluation stream; lower "
                   "quantile Bonferroni-adjusted across arms"),
        "chance": float(chance),
        "family_alpha": 0.05,
        "n_arms": n_arms,
        "alpha_per_arm": alpha_per_arm,
        "passed": best is not None,
        "best_arm": best,
        "arms": arms,
    }


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


_EFFICIENCY_WARMUP = 5
_EFFICIENCY_REPEATS = 30


def _efficiency_seq_len(settings: Settings) -> int:
    return 2 * int(settings.n_pairs) + 2


def _efficiency_companion(settings: Settings, out_dir: pathlib.Path,
                          experiment: str, cells, reference=None) -> dict:
    """Measure the inference-cost companion for one experiment's cells.

    ``cells`` is ``[(label, runs), ...]``.  The architecture is rebuilt from
    each cell's recorded ``model_cfg`` rather than from the runner's current
    settings, so the cost row describes the model that actually trained.  A
    measurement failure is recorded as an unavailable row instead of
    aborting an experiment whose real results are already on disk.  When a
    ``reference`` cell is given, ``d_acc`` is the paired mean accuracy
    difference against that cell; cost ratios and the accuracy difference
    are read together, never as a new decision rule.
    """
    seq_len = _efficiency_seq_len(settings)
    measured = {}
    entries = {}
    runs_by_label = {str(label): runs for label, runs in cells}
    ref_runs = (runs_by_label.get(str(reference))
                if reference is not None else None)
    valid_ref_runs = _curve_valid_runs(ref_runs or [])
    ref_accuracy = {}
    for run in valid_ref_runs:
        seed = run.get("seed")
        if seed is not None and _finite(run.get("accuracy")):
            ref_accuracy[seed] = float(run["accuracy"])
    all_ref_accuracy = {}
    for run in ref_runs or ():
        seed = run.get("seed")
        if seed is not None and _finite(run.get("accuracy")):
            all_ref_accuracy[seed] = float(run["accuracy"])
    for label, runs in cells:
        label = str(label)
        cfg = next((r.get("model_cfg") for r in runs
                    if isinstance(r.get("model_cfg"), dict)), None)
        if cfg is None:
            entries[label] = {"available": False,
                              "reason": "runs carry no model_cfg"}
            continue
        valid_runs = _curve_valid_runs(runs)
        acc = _mean_std([r.get("accuracy") for r in runs])
        valid_acc = _mean_std([r.get("accuracy") for r in valid_runs])
        recorded = sorted({int(r["params"]) for r in runs
                           if _finite(r.get("params"))})
        key = json.dumps(cfg, sort_keys=True)
        try:
            if key not in measured:
                measured[key] = measure_config(
                    cfg, settings.device, seq_len, batch=1,
                    warmup=_EFFICIENCY_WARMUP,
                    repeats=_EFFICIENCY_REPEATS)
            rec = dict(measured[key])
        except Exception as exc:  # noqa: BLE001 - report, do not kill run
            entries[label] = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            continue
        deltas = [
            float(run["accuracy"]) - ref_accuracy[run["seed"]]
            for run in valid_runs
            if run.get("seed") in ref_accuracy
            and _finite(run.get("accuracy"))]
        all_deltas = [
            float(run["accuracy"]) - all_ref_accuracy[run["seed"]]
            for run in runs
            if run.get("seed") in all_ref_accuracy
            and _finite(run.get("accuracy"))]
        rec.update({
            "accuracy_mean": acc["mean"],
            "accuracy_std": acc["std"],
            "accuracy_n": acc["n"],
            "accuracy_mean_curve_valid": valid_acc["mean"],
            "accuracy_std_curve_valid": valid_acc["std"],
            "accuracy_n_curve_valid": valid_acc["n"],
            "accuracy_delta_vs_reference": (
                sum(deltas) / len(deltas) if deltas else float("nan")),
            "accuracy_delta_n": len(deltas),
            "accuracy_delta_all": (
                sum(all_deltas) / len(all_deltas)
                if all_deltas else float("nan")),
            "accuracy_delta_n_all": len(all_deltas),
            "recorded_params": recorded[0] if len(recorded) == 1 else None,
            "params_match_run": (len(recorded) == 1
                                 and recorded[0] == rec["params"]),
        })
        entries[label] = rec

    print(f"\n[{experiment}] inference cost (batch=1, seq_len={seq_len}; "
          f"FLOPs are analytic forward matmuls; latency is the eval-mode "
          f"median of {_EFFICIENCY_REPEATS} passes on {settings.device}; "
          f"d_acc uses curve-valid paired seeds (acc is all runs); ratios "
          f"and cost are companions, not decision rules):")
    print(f"{'cell':>23} {'params':>9} {'param_x':>8} {'GFLOP':>8} "
          f"{'gflop_x':>8} {'lat_ms':>8} {'lat_x':>7} {'d_acc':>8} "
          f"{'acc':>7}")
    ref = entries.get(str(reference)) if reference is not None else None
    for label, _ in cells:
        label = str(label)
        rec = entries[label]
        if not rec.get("available", True):
            print(f"{label:>23}  unavailable: {rec.get('reason')}")
            continue
        param_x = (rec["params"] / ref["params"]
                   if ref and ref.get("available", True) and ref["params"]
                   else float("nan"))
        flop_x = (rec["analytic_matmul_flops"] / ref["analytic_matmul_flops"]
                  if ref and ref.get("available", True)
                  and ref["analytic_matmul_flops"] else float("nan"))
        lat_x = (rec["latency_ms_median"] / ref["latency_ms_median"]
                 if ref and ref.get("available", True)
                 and ref["latency_ms_median"] else float("nan"))

        def ratio(value):
            return f"{value:.3f}" if _finite(value) else "-"

        delta = rec.get("accuracy_delta_vs_reference")
        delta_s = f"{delta:+.4f}" if _finite(delta) else "-"
        print(f"{label:>23} {rec['params']:>9} {ratio(param_x):>8} "
              f"{rec['analytic_matmul_gflops']:>8.4f} {ratio(flop_x):>8} "
              f"{rec['latency_ms_median']:>8.3f} {ratio(lat_x):>7} "
              f"{delta_s:>8} {rec['accuracy_mean']:>7.4f}")

    payload = {
        "experiment": experiment,
        "seq_len": seq_len,
        "reference": None if reference is None else str(reference),
        "comparison": "efficiency_companion",
        "note": ("cost/accuracy companion: analytic matmul FLOPs plus "
                 "on-device latency and peak memory, with a paired "
                 "accuracy difference against the reference cell; it never "
                 "changes a pre-registered decision rule"),
        "cells": entries,
        "_provenance": _stamp(settings),
    }
    _write_json(out_dir / f"{experiment}_efficiency.json", payload)
    return payload


def _efficiency_pair(target, reference) -> dict:
    """Compare two measured cost records without re-measuring either side."""
    if not isinstance(target, dict) or not isinstance(reference, dict):
        return {"available": False, "reason": "missing cost record"}
    if not target.get("available", True) \
            or not reference.get("available", True):
        return {"available": False, "reason": "cost measurement unavailable"}
    fields = ("params", "analytic_matmul_flops", "latency_ms_median",
              "peak_increment_bytes")
    target_view = {key: target.get(key) for key in fields}
    reference_view = {key: reference.get(key) for key in fields}
    ratios = {}
    for key in fields:
        a, b = target_view[key], reference_view[key]
        if _finite(a) and _finite(b) and float(b) != 0.0:
            ratios[f"{key}_ratio_target_over_reference"] = float(a) / float(b)
    return {
        "available": True,
        "target": target_view,
        "reference": reference_view,
        "ratios": ratios,
    }


def _complexity_cost_companion(settings: Settings, out_dir: pathlib.Path,
                               verdict: dict, by_gate: dict,
                               deep_cost=None) -> dict:
    """Report the cost of the deep model against its tuned shallow rungs."""
    shallow = verdict.get("shallow_none") or {}
    matched = verdict.get("matched_shallow") or {}
    cells = {}
    deep_runs = by_gate.get("none", [])
    if isinstance(deep_cost, dict) and deep_cost.get("available", True):
        deep = dict(deep_cost)
        valid_deep = _curve_valid_runs(deep_runs)
        acc = _mean_std([r.get("accuracy") for r in valid_deep])
        all_acc = _mean_std([r.get("accuracy") for r in deep_runs])
        deep["accuracy_mean"] = acc["mean"]
        deep["accuracy_std"] = acc["std"]
        deep["accuracy_n"] = acc["n"]
        deep["accuracy_mean_all"] = all_acc["mean"]
        deep["accuracy_n_all"] = all_acc["n"]
        deep["role"] = "deep none"
        cells["deep_none"] = deep

    shallow_depth = int(shallow.get("n_layers") or 0)
    if shallow_depth > 0 and shallow.get("accuracy_by_seed"):
        try:
            rec = measure_config(
                _model_cfg(settings, gate="none", n_layers=shallow_depth),
                settings.device, _efficiency_seq_len(settings), batch=1,
                warmup=_EFFICIENCY_WARMUP, repeats=_EFFICIENCY_REPEATS)
            rec["accuracy_mean"] = shallow.get("accuracy_mean")
            rec["accuracy_std"] = shallow.get("accuracy_std")
            rec["role"] = f"tuned shallow ({shallow_depth} layer)"
            cells["tuned_shallow"] = rec
        except Exception as exc:  # noqa: BLE001 - cost is a companion
            cells["tuned_shallow"] = {
                "available": False,
                "role": f"tuned shallow ({shallow_depth} layer)",
                "reason": f"{type(exc).__name__}: {exc}"}
    elif shallow_depth > 0:
        cells["tuned_shallow"] = {
            "available": False,
            "role": f"tuned shallow ({shallow_depth} layer)",
            "reason": "reference rung is unmeasured (no accuracy_by_seed)"}

    wide_d = matched.get("d_model")
    if wide_d and matched.get("accuracy_by_seed"):
        try:
            rec = measure_config(
                _model_cfg(settings, gate="none", n_layers=1,
                           d_model=int(wide_d)),
                settings.device, _efficiency_seq_len(settings), batch=1,
                warmup=_EFFICIENCY_WARMUP, repeats=_EFFICIENCY_REPEATS)
            matched_acc = []
            for value in matched["accuracy_by_seed"].values():
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if _finite(value):
                    matched_acc.append(value)
            acc = _mean_std(matched_acc)
            rec["accuracy_mean"] = acc["mean"]
            rec["accuracy_std"] = acc["std"]
            rec["role"] = "capacity-matched shallow"
            cells["matched_shallow"] = rec
        except Exception as exc:  # noqa: BLE001 - cost is a companion
            cells["matched_shallow"] = {
                "available": False,
                "role": "capacity-matched shallow",
                "reason": f"{type(exc).__name__}: {exc}"}
    elif wide_d:
        cells["matched_shallow"] = {
            "available": False,
            "role": "capacity-matched shallow",
            "reason": "reference rung is unmeasured (no accuracy_by_seed)"}

    deep = cells.get("deep_none") or {}
    deep_by_seed = {}
    for run in _curve_valid_runs(deep_runs):
        seed = run.get("seed")
        if seed is not None and _finite(run.get("accuracy")):
            deep_by_seed[seed] = float(run["accuracy"])
    references = {
        "tuned_shallow": shallow.get("accuracy_by_seed"),
        "matched_shallow": matched.get("accuracy_by_seed"),
    }
    for key, rec in cells.items():
        if key == "deep_none" or not rec.get("available", True):
            continue
        rec["cost_ratio_deep_over_reference"] = _efficiency_pair(
            deep, rec)
        ref_by_seed = {}
        for seed, value in (references.get(key) or {}).items():
            try:
                seed = int(seed)
                value = float(value)
            except (TypeError, ValueError):
                continue
            if _finite(value):
                ref_by_seed[seed] = value
        deltas = [deep_by_seed[seed] - value
                  for seed, value in ref_by_seed.items()
                  if seed in deep_by_seed]
        rec["accuracy_delta_vs_deep"] = (
            sum(deltas) / len(deltas) if deltas else float("nan"))
        rec["accuracy_delta_n"] = len(deltas)

    print(f"\n[e1] complexity cost (batch=1, seq_len="
          f"{_efficiency_seq_len(settings)}; cost ratios are deep none / "
          f"reference; d_acc is deep minus reference, and both acc and "
          f"d_acc use curve-valid seeds (all-run deep mean stored as "
          f"accuracy_mean_all); companions only):")
    print(f"{'arm':>28} {'params':>9} {'GFLOP':>8} {'lat_ms':>8} "
          f"{'acc':>7} {'d_acc':>8} {'param_x':>8} {'gflop_x':>8} "
          f"{'lat_x':>8}")
    for key in ("deep_none", "tuned_shallow", "matched_shallow"):
        rec = cells.get(key)
        if not rec or not rec.get("available", True):
            if rec:
                print(f"{str(rec.get('role', key)):>28}  unavailable: "
                      f"{rec.get('reason')}")
            continue
        ratio = rec.get("cost_ratio_deep_over_reference") or {}
        ratios = ratio.get("ratios") or {}
        def _x(name):
            value = ratios.get(f"{name}_ratio_target_over_reference")
            return f"{value:.3f}" if _finite(value) else "-"
        delta = rec.get("accuracy_delta_vs_deep")
        delta_s = f"{delta:+.4f}" if _finite(delta) else "-"
        print(f"{str(rec.get('role', key)):>28} {rec['params']:>9} "
              f"{rec['analytic_matmul_gflops']:>8.4f} "
              f"{rec['latency_ms_median']:>8.3f} "
              f"{rec.get('accuracy_mean', float('nan')):>7.4f} "
              f"{delta_s:>8} "
              f"{_x('params'):>8} {_x('analytic_matmul_flops'):>8} "
              f"{_x('latency_ms_median'):>8}")

    payload = {
        "experiment": "e1",
        "seq_len": _efficiency_seq_len(settings),
        "comparison": "complexity_cost_companion",
        "note": ("cost of the deep ungated model against its tuned and "
                 "capacity-matched one-layer references; ratios are deep / "
                 "reference; never a decision rule"),
        "cells": cells,
        "_provenance": _stamp(settings),
    }
    _write_json(out_dir / "e1_complexity_cost.json", payload)
    return payload


def _length_efficiency_companion(settings: Settings, out_dir: pathlib.Path,
                                 experiment: str, cells, lengths,
                                 reference=None) -> dict:
    """Measure how cost grows with the E2 sweep length for every arm."""
    lengths = [int(length) for length in lengths]
    runs_by_label = {str(label): runs for label, runs in cells}
    ref_runs = (runs_by_label.get(str(reference))
                if reference is not None else None)
    ref_accuracy = {}
    for run in _curve_valid_runs(ref_runs or []):
        seed = run.get("seed")
        if seed is not None:
            ref_accuracy[seed] = run.get("accuracy_by_length") or {}
    all_ref_accuracy = {}
    for run in ref_runs or ():
        seed = run.get("seed")
        if seed is not None:
            all_ref_accuracy[seed] = run.get("accuracy_by_length") or {}
    measured = {}
    entries = {}
    for label, runs in cells:
        label = str(label)
        cfg = next((r.get("model_cfg") for r in runs
                    if isinstance(r.get("model_cfg"), dict)), None)
        if cfg is None:
            entries[label] = {"available": False,
                              "reason": "runs carry no model_cfg"}
            continue
        valid_runs = _curve_valid_runs(runs)
        by_length = {}
        for length in lengths:
            seq_len = 2 * length + 2
            key = (json.dumps(cfg, sort_keys=True), seq_len)
            try:
                if key not in measured:
                    measured[key] = measure_config(
                        cfg, settings.device, seq_len, batch=1,
                        warmup=_EFFICIENCY_WARMUP,
                        repeats=_EFFICIENCY_REPEATS)
                rec = dict(measured[key])
            except Exception as exc:  # noqa: BLE001 - report, do not kill run
                by_length[str(length)] = {
                    "available": False,
                    "reason": f"{type(exc).__name__}: {exc}"}
                continue
            acc = _mean_std([r["accuracy_by_length"][length]
                             for r in runs
                             if _finite((r.get("accuracy_by_length")
                                         or {}).get(length))])
            valid_acc = _mean_std([r["accuracy_by_length"][length]
                                   for r in valid_runs
                                   if _finite((r.get("accuracy_by_length")
                                               or {}).get(length))])
            rec["accuracy_mean"] = acc["mean"]
            rec["accuracy_std"] = acc["std"]
            rec["accuracy_n"] = acc["n"]
            rec["accuracy_mean_curve_valid"] = valid_acc["mean"]
            rec["accuracy_std_curve_valid"] = valid_acc["std"]
            rec["accuracy_n_curve_valid"] = valid_acc["n"]
            by_length[str(length)] = rec
        if ref_accuracy or all_ref_accuracy:
            for length in lengths:
                rec = by_length.get(str(length)) or {}
                if not rec.get("available", True):
                    continue
                deltas = []
                for run in valid_runs:
                    seed = run.get("seed")
                    refs = ref_accuracy.get(seed) or {}
                    truth = (run.get("accuracy_by_length") or {}).get(length)
                    other = refs.get(length)
                    if _finite(truth) and _finite(other):
                        deltas.append(float(truth) - float(other))
                all_deltas = []
                for run in runs:
                    seed = run.get("seed")
                    refs = all_ref_accuracy.get(seed) or {}
                    truth = (run.get("accuracy_by_length") or {}).get(length)
                    other = refs.get(length)
                    if _finite(truth) and _finite(other):
                        all_deltas.append(float(truth) - float(other))
                rec["accuracy_delta_vs_reference"] = (
                    sum(deltas) / len(deltas) if deltas else float("nan"))
                rec["accuracy_delta_n"] = len(deltas)
                rec["accuracy_delta_all"] = (
                    sum(all_deltas) / len(all_deltas)
                    if all_deltas else float("nan"))
                rec["accuracy_delta_n_all"] = len(all_deltas)
        entries[label] = {"by_length": by_length}

    base, longest = lengths[0], lengths[-1]
    print(f"\n[{experiment}] inference cost scaling (batch=1; FLOPs are "
          f"analytic forward matmuls; latency is the eval-mode median of "
          f"{_EFFICIENCY_REPEATS} passes on {settings.device}; d_acc uses "
          f"curve-valid paired seeds (acc is all runs); cost is a "
          f"companion, not a decision rule):")
    print(f"{'arm':>16} {'GFLOP@' + str(base):>11} "
          f"{'GFLOP@' + str(longest):>11} {'gflop_x':>8} "
          f"{'lat@' + str(base):>10} {'lat@' + str(longest):>10} "
          f"{'lat_x':>7} {'acc_long':>9} {'d_acc':>8}")
    for label, _ in cells:
        label = str(label)
        entry = entries[label]
        if not entry.get("available", True):
            print(f"{label:>16}  unavailable: {entry.get('reason')}")
            continue
        base_rec = entry["by_length"].get(str(base)) or {}
        long_rec = entry["by_length"].get(str(longest)) or {}
        if not base_rec.get("available", True) \
                or not long_rec.get("available", True):
            print(f"{label:>16}  unavailable at a sweep length")
            continue
        flop_x = (long_rec["analytic_matmul_flops"]
                  / base_rec["analytic_matmul_flops"]
                  if base_rec["analytic_matmul_flops"] else float("nan"))
        lat_x = (long_rec["latency_ms_median"]
                 / base_rec["latency_ms_median"]
                 if base_rec["latency_ms_median"] else float("nan"))
        delta = long_rec.get("accuracy_delta_vs_reference")
        delta_s = f"{delta:+.4f}" if _finite(delta) else "-"
        print(f"{label:>16} "
              f"{base_rec['analytic_matmul_gflops']:>11.4f} "
              f"{long_rec['analytic_matmul_gflops']:>11.4f} "
              f"{flop_x:>8.2f} "
              f"{base_rec['latency_ms_median']:>10.3f} "
              f"{long_rec['latency_ms_median']:>10.3f} "
              f"{lat_x:>7.2f} {long_rec['accuracy_mean']:>9.4f} "
              f"{delta_s:>8}")
    payload = {
        "experiment": experiment,
        "lengths": lengths,
        "seq_lengths": [2 * length + 2 for length in lengths],
        "reference": None if reference is None else str(reference),
        "comparison": "length_efficiency_companion",
        "note": ("cost scaling across the E2 sweep: analytic matmul FLOPs "
                 "and on-device latency at each length; d_acc is the paired "
                 "accuracy difference vs the reference at the longest "
                 "length; never a decision rule"),
        "cells": entries,
        "_provenance": _stamp(settings),
    }
    _write_json(out_dir / f"{experiment}_efficiency.json", payload)
    return payload


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

_CROSS_PROCESS_REPLAY_SCRIPT = r"""
import contextlib
import io
import json
import sys

from cage.data import batch_fingerprint, make_batch
from cage.model import build
from cage.repro import seed_everything, stream_generator
from cage.train import train_model

spec = json.loads(sys.argv[1])
seed = int(spec["seed"])
device = spec["device"]
deterministic = bool(spec["deterministic"])
seed_everything(seed, deterministic=deterministic)
gen = stream_generator(seed, device)
idx, tgt, meta = make_batch(int(spec["batch"]), int(spec["n_pairs"]),
                            int(spec["vocab"]), device, gen=gen)
fingerprint = batch_fingerprint(idx, tgt, meta)
seed_everything(seed, deterministic=deterministic)
model = build(dict(spec["model_cfg"]), compile=False)
with contextlib.redirect_stdout(io.StringIO()):
    hist = train_model(model, steps=int(spec["steps"]),
                       batch=int(spec["batch"]),
                       n_pairs=int(spec["n_pairs"]), lr=float(spec["lr"]),
                       vocab=int(spec["vocab"]), device=device,
                       log_every=1, seed=seed,
                       deterministic=deterministic)
print(json.dumps({"data": fingerprint,
                  "losses": [float(row["train_loss"]) for row in hist],
                  "deterministic": deterministic}))
"""

_REPRO_AUDIT_KEYS = (
    ("data_same_seed_identical", "same seed draws byte-identical batches"),
    ("data_diff_seed_differs", "a different seed draws different data"),
    ("data_seed_alias_same_stream",
     "aliased seeds with the same canonical 32-bit stream key draw the "
     "identical realization (fingerprint)"),
    ("eval_seed_alias_identical",
     "aliased evaluation seeds read the same held-out batches and replay "
     "identical metrics"),
    ("data_draw_leaves_global_rng_untouched",
     "data draws do not consume the global RNG (stream-owns-its-state)"),
    ("eval_leaves_global_rng_untouched",
     "accuracy/metrics/statistics evaluation does not consume the global "
     "RNG (its fixed evaluation streams own their generators)"),
    ("eval_restores_module_modes",
     "evaluation restores every module's train/eval flag (dropout stays "
     "off afterwards)"),
    ("eval_metrics_same_seed_identical",
     "same model + same evaluation seed replays identical metrics"),
    ("rng_same_seed_identical",
     "every RNG seed_everything pins (Python, Torch CPU[, NumPy][, CUDA]) "
     "replays exactly from the same seed"),
    ("rng_diff_seed_differs", "each seeded RNG responds to a different seed"),
    ("init_same_seed_identical", "same seed builds byte-identical parameters"),
    ("init_diff_seed_differs", "a different seed builds different parameters"),
    ("trajectory_same_seed_identical",
     "same seed replays the exact training-loss trajectory"),
    ("trajectory_after_global_rng_noise_identical",
     "global-RNG noise before a run does not change the trajectory "
     "(entry re-seeding covers it)"),
    ("trajectory_diff_seed_differs", "a different seed changes the trajectory"),
    ("trajectory_dropout_same_seed_identical",
     "same seed replays the dropout trajectory (the global-RNG-consuming "
     "training path, which a dropout=0 run never exercises)"),
    ("trajectory_dropout_diff_seed_differs",
     "a different seed changes the dropout trajectory"),
    ("cross_process_data_identical",
     "a fresh interpreter draws the same data batch for the same seed"),
    ("cross_process_trajectory_identical",
     "a fresh interpreter replays the training trajectory (same "
     "seed/config/device)"),
    ("determinism_flag_follows_request",
     "the deterministic-kernel switch follows the requested setting (on "
     "enables it, off restores the default instead of inheriting a "
     "previous deterministic run)"),
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
    seeds = {stream_seed_key(s) for s in settings.seeds}
    reserved = {stream_seed_key(s) for s in RESERVED_EVAL_SEEDS}
    s = int(start)
    while stream_seed_key(s) in reserved or stream_seed_key(s) in seeds:
        s += 1
    return s


def _cross_process_replay(settings: Settings, seed: int, cfg: dict,
                          data_fingerprint: str, losses: list[float]) -> dict:
    """Replay the probe run in a fresh interpreter and compare results.

    In-process repetition cannot catch state that *survives* a replay:
    module-level caches, hash-randomisation order, or a seeding step that
    only works because the first run happened to leave the RNG at the right
    place.  A fresh process with its own ``PYTHONHASHSEED`` (derived from
    the probe seed) exercises the actual launch path.  The child imports
    only the public data/model/train entry points and prints its batch
    fingerprint plus loss sequence as the last stdout line; warnings from
    torch stay on stderr.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    spec = {
        "seed": int(seed),
        "batch": _REPRO_AUDIT_BATCH,
        "n_pairs": int(settings.n_pairs),
        "vocab": int(settings.vocab),
        "steps": _REPRO_AUDIT_STEPS,
        "device": settings.device,
        "deterministic": bool(settings.deterministic),
        "model_cfg": cfg,
        "lr": 1e-3,
    }
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (str(root) if not pythonpath
                         else str(root) + os.pathsep + pythonpath)
    env["PYTHONHASHSEED"] = str(stream_seed_key(seed))
    base = {
        "started": False,
        "error": None,
        "pythonhashseed": env["PYTHONHASHSEED"],
        "data_identical": False,
        "trajectory_identical": False,
        "max_abs_diff": float("inf"),
        "child_losses": [],
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CROSS_PROCESS_REPLAY_SCRIPT,
             json.dumps(spec)],
            capture_output=True, text=True, env=env, cwd=str(root),
            timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base
    lines = proc.stdout.strip().splitlines()
    if proc.returncode != 0 or not lines:
        detail = (proc.stderr.strip() or "no stdout").splitlines()
        base["error"] = (f"child exit {proc.returncode}: "
                         + " | ".join(detail[-3:]))
        return base
    try:
        child = json.loads(lines[-1])
        child_losses = [float(v) for v in child["losses"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        base["error"] = f"unparseable child output: {type(exc).__name__}: {exc}"
        return base
    same_length = len(child_losses) == len(losses)
    diffs = [abs(a - b) for a, b in zip(losses, child_losses)]
    max_diff = max(diffs) if diffs else float("inf")
    base.update({
        "started": True,
        "child_data": child.get("data"),
        "child_deterministic": bool(child.get("deterministic")),
        "child_losses": child_losses,
        "data_identical": child.get("data") == data_fingerprint,
        "trajectory_identical": bool(
            same_length and max_diff <= 1e-9),
        "max_abs_diff": max_diff,
    })
    return base


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
    * **RNG pinning**: one draw from every generator ``seed_everything``
      claims to seed - Python ``random``, Torch CPU, NumPy (when installed)
      and CUDA (when active) - replays under the same seed and changes under
      a different one.  This is measured directly because a dropout=0
      trajectory never reads the global Torch RNG, so it cannot catch a
      forgotten ``torch.manual_seed`` on its own;
    * **initialisation**: building the model twice under the same seed
      yields byte-identical parameters, a different seed yields different
      ones;
    * **trajectory**: a short training run replays the exact loss sequence
      under the same seed, is unaffected by global-RNG noise left by
      anything that ran before it (the entry re-seeds), and changes under a
      different seed.  The same replay check is repeated with dropout on,
      which *does* consume the global RNG: only that run verifies the
      dropout-mask stream is seeded and not shared with the data stream;
    * **cross-process replay**: a fresh interpreter (separate
      ``PYTHONHASHSEED``, its own module state) re-draws the probe batch
      and re-runs the short trajectory through the public entry points.
      This catches seeding that only works because in-process state was
      already warm, which repeated calls in one process cannot expose.

    Scope follows ``cage.repro``: same machine and PyTorch build, not
    bit-exact reproduction across hardware/instruction sets.  Cached on
    ``Settings`` so one process (``--only all``) runs it once.
    """
    cache_key = _config_cache_key(settings)
    cached = getattr(settings, "_repro_audit_cache", None)
    if cached is not None and getattr(
            settings, "_repro_audit_cache_key", None) == cache_key:
        return cached
    device = settings.device
    seed = _probe_seed(settings, _REPRO_AUDIT_BASE_SEED)
    other = _probe_seed(settings, seed + 1)
    cfg = _model_cfg(settings, gate="calib", d_model=48, n_layers=2)
    cfg_drop = _model_cfg(settings, gate="calib", d_model=48, n_layers=2,
                          dropout=0.1)
    try:
        import numpy as np
    except ImportError:
        np = None

    def _rng_draws(s):
        """One draw from each RNG ``seed_everything`` claims to pin.

        The trajectory checks below run with dropout=0, which never reads
        the global Torch RNG, so they cannot fail if ``torch.manual_seed``
        were forgotten.  Drawing directly from every seeded generator
        turns "Python/NumPy/Torch/CUDA are seeded" from a docstring claim
        into a measured one.
        """
        seed_everything(s)
        draws = {"python_random": random.random(),
                 "torch_cpu": float(torch.rand(()))}
        if np is not None:
            draws["numpy"] = float(np.random.random())
        # Probe every CUDA device whenever the build exposes them, even for
        # a CPU run: ``seed_everything`` calls ``manual_seed_all``, so
        # "all devices" is the claim that must be measured rather than
        # assumed from a single-device draw.
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                try:
                    with torch.cuda.device(index):
                        draws[f"torch_cuda_{index}"] = float(
                            torch.rand((), device=f"cuda:{index}"))
                except Exception:  # noqa: BLE001 - a broken device probe must
                    pass          # not hide the checks that still ran
        return draws

    def _draw(s):
        gen = stream_generator(s, device)
        return make_batch(_REPRO_AUDIT_BATCH, settings.n_pairs,
                          settings.vocab, device, gen=gen)

    fp_same_a = batch_fingerprint(*_draw(seed))
    fp_same_b = batch_fingerprint(*_draw(seed))
    fp_other = batch_fingerprint(*_draw(other))
    alias_seed = seed + 2 ** 32
    fp_alias = batch_fingerprint(*_draw(alias_seed))

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

    rng_a = _rng_draws(seed)
    rng_b = _rng_draws(seed)
    rng_other = _rng_draws(other)

    det_on = seed_everything(seed, deterministic=True)
    det_off = seed_everything(seed, deterministic=False)
    determinism_toggle = bool(
        det_on["deterministic_algorithms"]
        and (det_on["cudnn"] is None
             or det_on["cudnn"]["deterministic"])
        and not det_off["deterministic_algorithms"]
        and (det_off["cudnn"] is None
             or not det_off["cudnn"]["deterministic"]))

    seed_everything(seed)
    init_same_a = _model_state_hash(build(cfg, compile=False))
    seed_everything(seed)
    init_same_b = _model_state_hash(build(cfg, compile=False))
    seed_everything(other)
    init_other = _model_state_hash(build(cfg, compile=False))

    def _same_metrics(a, b):
        if a.keys() != b.keys():
            return False
        for key in a:
            va, vb = a[key], b[key]
            if isinstance(va, float) and isinstance(vb, float):
                if math.isnan(va) and math.isnan(vb):
                    continue
            if va != vb:
                return False
        return True

    def _eval_isolation():
        """Evaluation must neither consume the global RNG nor change modes.

        Training calls validation statistics between optimizer steps.  If
        an evaluation helper quietly drew from the global generator (or
        left dropout enabled), the remaining training trajectory would
        depend on *when* the evaluation ran - a reproducibility hole the
        data-stream check alone cannot see.
        """
        seed_everything(seed, deterministic=settings.deterministic)
        model = build(cfg, compile=False)
        model.train()
        modes = {id(m): bool(m.training) for m in model.modules()}
        torch_before = torch.get_rng_state()
        py_before = random.getstate()
        np_before = (np.random.get_state() if np is not None else None)
        cuda_before = (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None)
        kwargs = dict(batch=_REPRO_AUDIT_BATCH, batches=1, seed=seed)
        metrics_a = eval_metrics(model, settings.n_pairs, settings.vocab,
                                 device, **kwargs)
        metrics_b = eval_metrics(model, settings.n_pairs, settings.vocab,
                                 device, **kwargs)
        alias_kwargs = dict(kwargs, seed=alias_seed)
        metrics_alias = eval_metrics(
            model, settings.n_pairs, settings.vocab, device, **alias_kwargs)
        eval_accuracy(model, settings.n_pairs, settings.vocab, device,
                      **kwargs)
        collect_stats(model, settings.n_pairs, settings.vocab, device,
                      **kwargs)
        rng_ok = bool(torch.equal(torch.get_rng_state(), torch_before))
        rng_ok = rng_ok and random.getstate() == py_before
        if np_before is not None:
            now = np.random.get_state()
            rng_ok = rng_ok and bool(
                now[0] == np_before[0]
                and np.array_equal(now[1], np_before[1])
                and now[2:] == np_before[2:])
        if cuda_before is not None:
            rng_ok = rng_ok and all(
                torch.equal(a, b) for a, b in
                zip(torch.cuda.get_rng_state_all(), cuda_before))
        modes_ok = all(bool(m.training) == modes[id(m)]
                       for m in model.modules())
        return (rng_ok, modes_ok, _same_metrics(metrics_a, metrics_b),
                _same_metrics(metrics_a, metrics_alias))

    (eval_rng_ok, eval_modes_ok, eval_replay_ok,
     eval_alias_ok) = _eval_isolation()

    def _trajectory(s, rng_noise=False, model_cfg=None):
        with contextlib.redirect_stdout(io.StringIO()):
            # Initialise the model under the controlled seed, then perturb
            # the *global* RNG immediately before the training entry point.
            # The entry must re-seed (``train_model`` does), otherwise the
            # perturbation would reach the dropout masks and change the
            # trajectory.  Seeding after the perturbation here would make
            # the check vacuous: it would erase the noise itself.
            seed_everything(s, deterministic=settings.deterministic)
            model = build(model_cfg if model_cfg is not None else cfg,
                          compile=False)
            if rng_noise:
                torch.manual_seed(s + 777)
                random.seed(s + 777)
            hist = train_model(model, steps=_REPRO_AUDIT_STEPS,
                               batch=_REPRO_AUDIT_BATCH,
                               n_pairs=settings.n_pairs, lr=1e-3,
                               vocab=settings.vocab, device=device,
                               log_every=1, seed=s,
                               deterministic=settings.deterministic)
        return [float(h["train_loss"]) for h in hist]

    traj_a = _trajectory(seed)
    traj_replay = _trajectory(seed)
    traj_other = _trajectory(other)
    traj_drop_a = _trajectory(seed, model_cfg=cfg_drop)
    traj_drop_replay = _trajectory(seed, model_cfg=cfg_drop)
    traj_drop_other = _trajectory(other, model_cfg=cfg_drop)
    # The dropout trajectory is the one that actually consumes the global
    # RNG, so the "noise before the entry is erased by the entry's own
    # re-seeding" claim must be checked on it (a dropout=0 trajectory is
    # deterministically independent of the global RNG either way).
    traj_drop_noisy = _trajectory(seed, rng_noise=True, model_cfg=cfg_drop)

    cross = _cross_process_replay(settings, seed, cfg, fp_same_a, traj_a)

    audit = {
        "probe_seed": int(seed),
        "other_seed": int(other),
        "device": device,
        "steps": _REPRO_AUDIT_STEPS,
        "batch": _REPRO_AUDIT_BATCH,
        "data_same_seed_sha256": fp_same_a,
        "data_diff_seed_sha256": fp_other,
        "data_alias_seed_sha256": fp_alias,
        "eval_seed_alias_identical": bool(eval_alias_ok),
        "init_same_seed_sha256": init_same_a,
        "trajectory_losses": traj_a,
        "rngs_probed": sorted(rng_a),
        "rng_draws_same_seed": rng_a,
        "dropout": float(cfg_drop["dropout"]),
        "trajectory_dropout_losses": traj_drop_a,
        "trajectory_dropout_noise_losses": traj_drop_noisy,
        "data_same_seed_identical": bool(fp_same_a == fp_same_b),
        "data_diff_seed_differs": bool(fp_same_a != fp_other),
        "data_seed_alias_same_stream": bool(fp_same_a == fp_alias),
        "data_draw_leaves_global_rng_untouched": rng_untouched,
        "eval_leaves_global_rng_untouched": bool(eval_rng_ok),
        "eval_restores_module_modes": bool(eval_modes_ok),
        "eval_metrics_same_seed_identical": bool(eval_replay_ok),
        "rng_same_seed_identical": bool(rng_a == rng_b),
        "rng_diff_seed_differs": bool(rng_a != rng_other),
        "init_same_seed_identical": bool(init_same_a == init_same_b),
        "init_diff_seed_differs": bool(init_same_a != init_other),
        "trajectory_same_seed_identical": bool(traj_a == traj_replay),
        "trajectory_after_global_rng_noise_identical": bool(
            traj_drop_a == traj_drop_noisy),
        "trajectory_diff_seed_differs": bool(traj_a != traj_other),
        "trajectory_dropout_same_seed_identical": bool(
            traj_drop_a == traj_drop_replay),
        "trajectory_dropout_diff_seed_differs": bool(
            traj_drop_a != traj_drop_other),
        "cross_process_data_identical": bool(cross["data_identical"]),
        "cross_process_trajectory_identical": bool(
            cross["trajectory_identical"]),
        "cross_process_max_abs_diff": cross["max_abs_diff"],
        "cross_process_pythonhashseed": cross["pythonhashseed"],
        "cross_process_error": cross["error"],
        "cross_process": cross,
        "determinism_flag_follows_request": determinism_toggle,
        "deterministic_on": det_on,
        "deterministic_off": det_off,
        "deterministic_algorithms": bool(getattr(
            torch, "are_deterministic_algorithms_enabled", lambda: False)()),
        "scope": ("same machine + same PyTorch build; bit-exact reproduction "
                  "across different hardware is not claimed"),
    }
    audit["passed"] = all(bool(audit[key]) for key, _ in _REPRO_AUDIT_KEYS)
    settings._repro_audit_cache = audit
    settings._repro_audit_cache_key = cache_key
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


def _steps_run_from_hist(hist):
    """Best-effort optimizer-step count when ``early_stop_info`` is absent."""
    if not hist:
        return 0
    skipped = sum(1 for h in hist
                  if isinstance(h.get("diagnosis"), dict)
                  or not _finite(h.get("train_loss")))
    return max(0, max(int(h["step"]) for h in hist) + 1 - skipped)


def _curve_record(hist, early_stop_info=None, seeding=None):
    """Per-run learning-curve summary (monitor the curve, not just the end).

    Stores both the reduced verdict and the raw series (steps, train loss,
    validation loss, validation accuracy, pre-clip gradient norm), so a run
    that is later read as overfitting/plateauing can be re-examined without
    retraining.
    ``steps_run`` records the number of optimizer steps actually executed
    (from ``train_model``'s ``early_stop_info``, or the logged history as a
    fallback) and ``stopped_early``/``restored_best`` record whether the
    fixed epoch step count was cut short.  The controlled audit reads these:
    ``training_recipe.steps`` is the *intended* step count, so without this
    stamp a future early-stop could compare runs trained for different
    numbers of steps while every recipe field still matched.

    The staged non-finite triage (``diagnose_nonfinite``) is persisted too:
    the training loop prints it at the step it fires, but a console line is
    not part of the record, so a later reader of ``curve_verdict =
    numerical_instability`` could not tell whether the data, the
    activations, the loss or the gradients were blamed.  Each report is
    stored with its step under ``diagnoses`` / ``val_diagnoses`` (the
    validation-side triage runs on the exact batch that produced the bad
    value), and the steps are listed under ``nonfinite_steps``.

    ``seeding`` is the actual :func:`cage.repro.seed_everything` result
    returned for this run (Python/NumPy/Torch/CUDA coverage, pinned
    ``PYTHONHASHSEED``, deterministic-kernel/TF32 state).  It rides with
    the curve so a record proves which RNGs this run actually pinned
    instead of only citing the static policy.
    """
    verdict = training_curve_verdict(hist)
    train = [h for h in hist if _finite(h.get("train_loss"))]
    val = [h for h in hist if h.get("val_loss") is not None
           and _finite(h.get("val_loss"))]
    val_acc = [h for h in hist if h.get("val_acc") is not None
               and _finite(h.get("val_acc"))]

    def _diagnoses(key):
        return [{"step": int(h["step"]), **h[key]} for h in hist
                if isinstance(h.get(key), dict)]

    diagnoses = _diagnoses("diagnosis")
    val_diagnoses = _diagnoses("val_diagnosis")
    nonfinite_steps = sorted({int(h["step"]) for h in hist
                              if isinstance(h.get("diagnosis"), dict)
                              or isinstance(h.get("val_diagnosis"), dict)
                              or not _finite(h.get("train_loss"))
                              or (h.get("val_loss") is not None
                                  and not _finite(h.get("val_loss")))
                              or (h.get("grad_norm") is not None
                                  and not _finite(h.get("grad_norm")))})
    return {
        "train_loss_first": train[0]["train_loss"] if train else float("nan"),
        "train_loss_last": train[-1]["train_loss"] if train else float("nan"),
        "val_loss_first": val[0]["val_loss"] if val else float("nan"),
        "val_loss_last": val[-1]["val_loss"] if val else float("nan"),
        "val_acc_first": val_acc[0]["val_acc"] if val_acc else float("nan"),
        "val_acc_last": val_acc[-1]["val_acc"] if val_acc else float("nan"),
        "best_val_acc": max(float(h["val_acc"]) for h in val_acc)
        if val_acc else float("nan"),
        "train_delta": verdict["train_delta"],
        "val_delta": verdict["val_delta"],
        "generalization_gap": verdict["generalization_gap"],
        "best_val_loss": verdict["best_val_loss"],
        "best_val_step": verdict["best_val_step"],
        "val_regression_after_best": verdict["val_regression_after_best"],
        "late_overfit": verdict["late_overfit"],
        "late_overfit_significant": verdict["late_overfit_significant"],
        "overfitting_significant": verdict["overfitting_significant"],
        "late_train_delta": verdict["late_train_delta"],
        "late_val_delta": verdict["late_val_delta"],
        "val_loss_se_last": verdict["val_loss_se_last"],
        "val_change_abs": verdict["val_change_abs"],
        "val_noise_2se": verdict["val_noise_2se"],
        "val_movement_significant": verdict["val_movement_significant"],
        "nonfinite_steps": nonfinite_steps,
        "diagnoses": diagnoses,
        "val_diagnoses": val_diagnoses,
        "lr_first": verdict["lr_first"],
        "lr_last": verdict["lr_last"],
        "grad_norm_max": verdict["grad_norm_max"],
        "grad_norm_explode_count": verdict["grad_norm_explode_count"],
        "curve_steps": [int(h["step"]) for h in train],
        "curve_train_loss": [float(h["train_loss"]) for h in train],
        "curve_train_loss_window": [
            None if h.get("train_loss_window") is None
            else float(h["train_loss_window"]) for h in train],
        "curve_val_loss": [None if h.get("val_loss") is None
                           else float(h["val_loss"]) for h in train],
        "curve_val_loss_se": [None if h.get("val_loss_se") is None
                              else float(h["val_loss_se"]) for h in train],
        "curve_val_acc": [None if h.get("val_acc") is None
                          else float(h["val_acc"]) for h in train],
        "curve_grad_norm": [None if h.get("grad_norm") is None
                            else float(h["grad_norm"]) for h in train],
        "curve_lr": [None if h.get("lr") is None else float(h["lr"])
                     for h in train],
        "curve_verdict": verdict["verdict"],
        "curve_recommendation": verdict["recommendation"],
        "steps_run": int(
            (early_stop_info or {}).get("steps_run")
            if (early_stop_info or {}).get("steps_run") is not None
            else _steps_run_from_hist(hist)),
        "early_stop_enabled": bool(
            (early_stop_info or {}).get("enabled", False)),
        "stopped_early": bool(
            (early_stop_info or {}).get("stopped_early", False)),
        "restored_best": bool(
            (early_stop_info or {}).get("restored_best", False)),
        "seeding": dict(seeding or {}),
    }


def _train_and_eval(settings: Settings, seed: int, lr: float, eval_seed: int,
                    permute_labels: bool = False, selection: bool = False,
                    steps: int | None = None, **model_overrides):
    """Train + score one run through the single shared entry point.

    ``eval_seed`` is mandatory so every call site states which stream it
    reads; a defaulted sealed test stream would let a future tuning call
    silently fit decisions to the final numbers.  ``selection=True`` marks
    a call whose metrics inform a choice (recipe/baseline tuning) and is
    refused on the sealed test stream by ``check_eval_seed``.

    ``steps`` optionally overrides the shared recipe's step budget for
    tuned baselines (the depth ladder and its capacity control); ``None``
    keeps ``settings.steps`` exactly as before, so every controlled grid
    still trains one frozen recipe.

    Returns ``(model, params, metrics, curve, model_cfg)`` where
    ``model_cfg`` is the exact resolved configuration the model was built
    from - every ``Cfg`` field including defaults, so an unset ablation flag
    cannot be mistaken for an off switch.  Run records keep it so the
    controlled-variable audit can check the *trained* configuration
    directly (ground truth) instead of trusting the hand-attached labels.
    """
    check_eval_seed(eval_seed, selection=selection)
    if stream_seed_key(eval_seed) in {
            stream_seed_key(s) for s in settings.seeds}:
        raise ValueError(
            f"eval_seed={eval_seed} is also a configured training seed; "
            f"evaluation data must come from a stream that never trained "
            f"the model"
        )
    raw_train_steps = settings.steps if steps is None else steps
    train_steps = _positive_integer(raw_train_steps)
    if train_steps is None:
        raise ValueError(
            f"steps must be a positive integer, got {raw_train_steps!r}")
    raw_curve_points = settings.curve_points
    if isinstance(raw_curve_points, bool):
        raise ValueError(
            f"curve_points must be a non-negative integer, got "
            f"{raw_curve_points!r}")
    try:
        curve_points_num = float(raw_curve_points)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"curve_points must be a non-negative integer, got "
            f"{raw_curve_points!r}") from exc
    if not (math.isfinite(curve_points_num) and curve_points_num >= 0.0
            and curve_points_num == int(curve_points_num)):
        raise ValueError(
            f"curve_points must be a non-negative integer, got "
            f"{raw_curve_points!r}")
    curve_points = int(curve_points_num)
    seed_everything(seed, deterministic=settings.deterministic)
    model = build(_model_cfg(settings, **model_overrides), compile=False)
    model_cfg = asdict(model.cfg)
    params = sum(p.numel() for p in model.parameters())
    eval_every = (max(1, train_steps // curve_points)
                  if curve_points > 0 else 0)
    early_stop_info = {}
    seeding = {}
    hist = train_model(model, steps=train_steps, batch=settings.batch,
                       n_pairs=settings.n_pairs, lr=lr, vocab=settings.vocab,
                       device=settings.device, log_every=0, seed=seed,
                       eval_every=eval_every, val_batches=settings.val_batches,
                       permute_labels=permute_labels,
                       early_stop_info=early_stop_info,
                       deterministic=settings.deterministic,
                       seed_info=seeding)
    metrics = eval_metrics(model, settings.n_pairs, settings.vocab,
                           settings.device, batch=settings.eval_batch,
                           batches=settings.eval_batches, seed=eval_seed)
    return model, params, metrics, _curve_record(
        hist, early_stop_info, seeding), model_cfg


def tune_shared_recipe(settings: Settings):
    """Tune the learning rate on the plain ``none`` baseline.

    Selection uses the validation seed only, over the pre-registered
    ``min_seeds_for_claim`` floor's worth of training seeds (the grid never
    freezes a recipe on fewer seeds than a claim would need - a two-seed
    argmax is inside the seed-to-seed noise); the winning rate is then
    frozen for *every* variant, so a gate variant can never win via a
    luckier optimizer setting - the E1/E3 comparisons remain controlled.

    The learning-curve verdict of every tuning run is stored with the
    table: if the selected rate's runs are all curve-invalid (plateau /
    diverging / instability) the selection was made between underfit or
    broken runs, and the frozen recipe cannot be trusted to expose the
    mechanism under test - ``_resolve_lr`` prints that warning.  Each row
    also carries the per-seed validation accuracy and full curve record, so
    the selection runs are traceable experiments themselves rather than
    only an aggregate mean/verdict.

    Returns ``(best_lr, table, selection)`` where ``selection`` is the
    :func:`_selection_reliability` audit of the argmax: even at the claim
    floor the winning rate can sit inside the seed-to-seed spread of a
    neighbour, and that point selection must be recorded rather than
    printed as if the grid had resolved the recipe.
    """
    tune_seeds = settings.seeds[:max(1, min(settings.min_seeds_for_claim,
                                             len(settings.seeds)))]
    table = []
    for lr in settings.lr_grid:
        accs, verdicts, runs = [], [], []
        for seed in tune_seeds:
            _, params, metrics, curve, _ = _train_and_eval(
                settings, seed, lr, eval_seed=VAL_SEED, selection=True,
                gate="none")
            accs.append(metrics["accuracy"])
            verdicts.append(curve["curve_verdict"])
            runs.append({"seed": int(seed), "params": int(params),
                         "val_accuracy": metrics["accuracy"],
                         "eval_seed": int(VAL_SEED),
                         "curve": curve})
        mean = sum(accs) / len(accs)
        var = (sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)
               if len(accs) > 1 else 0.0)
        table.append({"lr": lr, "val_accuracy_mean": mean,
                      "val_accuracy_std": var ** 0.5, "n_seeds": len(accs),
                      "curve_verdicts": verdicts,
                      "curve_invalid_runs": sum(
                          v in CURVE_INVALID_VERDICTS for v in verdicts),
                      "runs": runs})
    # Same all-NaN guard as the depth ladder: ``max`` over an all-NaN key
    # returns an arbitrary row, which would freeze an unmeasured recipe for
    # every controlled contrast - refuse instead of fabricating a selection.
    finite_rows = [row for row in table
                   if _finite(row["val_accuracy_mean"])]
    if not finite_rows:
        raise RuntimeError(
            f"shared recipe tuning produced no finite validation accuracy "
            f"across lr grid {tuple(settings.lr_grid)}: every tuning run "
            f"diverged or returned NaN, so no learning rate can be frozen "
            f"for the controlled grids (inspect the run records; no "
            f"downstream contrast from this run is interpretable)")
    pick = _curve_valid_selection_rows(table)
    if pick["n_excluded"]:
        print(f"[baselines] shared recipe selection excluded "
              f"{pick['n_excluded']} of {pick['n_finite']} finite lr row(s) "
              f"with curve-invalid runs; selecting among curve-valid rows "
              f"only")
    elif pick["n_invalid"]:
        print(f"[baselines] WARNING: every finite lr row has curve-invalid "
              f"runs; falling back to all {pick['n_finite']} finite row(s) - "
              f"the frozen recipe is not validated by a healthy curve")
    eligible = pick["rows"]
    best = max(eligible, key=lambda row: row["val_accuracy_mean"])
    selection = _selection_reliability(
        eligible, grid=[row["lr"] for row in eligible])
    selection["pool"] = pick["pool"]
    selection["excluded_curve_invalid_rows"] = pick["n_excluded"]
    selection["curve_invalid_rows"] = pick["n_invalid"]
    return best["lr"], table, selection


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
    else:
        print(f"[{tag}] reusing shared tuned recipe lr={cached[0]:g}")
    lr, table, selection = cached
    warn = _selection_warn(f"shared recipe selection lr={lr:g}", selection)
    if warn:
        print(f"[{tag}] {warn}: the frozen recipe may make every variant "
              f"look equally flat")
    _write_json(out_dir / f"{tag}_lr_tuning.json",
                {"val_seed": VAL_SEED, "table": table, "chosen_lr": lr,
                 "selection": selection})
    return lr


def _answer_balance(settings: Settings, batch=256, batches=4, seeds=None):
    """Class-balance diagnostic for the answer values, fitted on the
    training streams only.

    Method contract: every fitted statistic is computed on the data the
    model is trained on, never on a held-out split.  The held-out answer
    distributions are instead *scored against* this training reference by
    the HB3 split audit (per-field chi-square shift tests), so
    validation/test/stats data is transform-only here.  The complementary
    held-out *uniformity* check (does the evaluation stream itself satisfy
    the accuracy premise?) is the separate scoring-only audit in
    :func:`_eval_answer_balance`; no training statistic is fitted there
    either.  ``seeds`` defaults
    to the configured training seeds (``(0,)`` when none are configured,
    matching the training convention elsewhere); each seed is refused if
    it is a reserved evaluation seed.

    The returned ``uniformity`` block is the *decision* behind using
    accuracy as the headline metric: a Monte Carlo-calibrated chi-square
    test of the drawn answer counts against the uniform expectation over
    ids ``2..vocab-1`` (PAD/SEP are never answers).  Shares alone cannot
    separate sampling noise from a real shift at ~10 expected samples per
    class; the test can (see ``cage.data.answer_uniformity_test``).  When
    it fails, the verdict layer flags the balance premise and the
    PR-AUC/F1 companions - not accuracy - carry the claims.  Every drawn
    batch passes the shared ``validate_batch`` cleaning gate first, so the
    balance decision is computed on the same validated data contract as
    training and the reported metrics.
    """
    if seeds is None:
        seeds = tuple(settings.seeds) if settings.seeds else (0,)
    counts = torch.zeros(settings.vocab, dtype=torch.float64)
    per_seed = {}
    for seed in seeds:
        check_train_seed(seed)
        gen = stream_generator(seed, settings.device)
        n_seed = 0
        for _ in range(batches):
            idx, targets, meta = make_batch(batch, settings.n_pairs,
                                            settings.vocab,
                                            settings.device, gen=gen)
            validate_batch(idx, targets, meta, settings.vocab, diagnostics=False)
            counts += torch.bincount(meta["answer"].cpu(),
                                     minlength=settings.vocab)
            n_seed += meta["answer"].numel()
        per_seed[int(seed)] = int(n_seed)
    present = counts > 0
    shares = counts[present] / counts.sum()
    uniformity = answer_uniformity_test(counts[2:])
    return {
        "train_seeds": [int(s) for s in seeds],
        "batches_per_seed": int(batches),
        "samples_per_seed": per_seed,
        "n_samples": int(counts.sum()),
        "n_classes_possible": settings.vocab - 2,
        "n_classes_present": int(present.sum()),
        "min_class_share": (float(shares.min()) if shares.numel()
                            else float("nan")),
        "max_class_share": (float(shares.max()) if shares.numel()
                            else float("nan")),
        "expected_uniform_share": 1.0 / (settings.vocab - 2),
        "uniformity": uniformity,
        "power_note": (
            "gross-imbalance test only: a pass at "
            f"{uniformity['expected_per_class']:.2f} expected samples/class "
            "does not certify the absence of a small or moderate skew"),
    }


# The held-out streams whose class balance backs the accuracy readout.
# VAL/TEST are consumed at ``eval_batch x eval_batches`` by ``eval_metrics``;
# STATS is consumed at ``stats_batch x stats_batches`` by
# ``collect_stats``.  The linear probe is scored on its own, deliberately
# wider geometry (``probe_eval_batch x probe_eval_batches``) and its HB2
# near-chance certificate is an accuracy claim, so the probe VAL/TEST
# streams are audited at that geometry too - the eval-geometry prefix is a
# different realization and cannot certify them.  The audit reproduces each
# consumer's exact geometry and generator seed, so the tested sample is the
# sample the models are scored on - a wider audit draw would test a
# different prefix and could pass while the reported stream is skewed.
# E2's longer sweep lengths are audited separately in
# ``_eval_answer_balance`` because the stream realization depends on length.
_EVAL_BALANCE_STREAMS = (
    ("val", VAL_SEED, "eval"),
    ("test", TEST_SEED, "eval"),
    ("stats", STATS_SEED, "stats"),
    ("probe_val", VAL_SEED, "probe"),
    ("probe_test", TEST_SEED, "probe"),
)


def _eval_answer_balance(settings: Settings) -> dict:
    """Class-balance audit for the *evaluation* answer streams.

    ``_answer_balance`` tests the training streams because the train-time
    class prior is what the model sees.  Accuracy is also the headline on
    the held-out streams, and that choice is only justified when those
    streams are uniform too: a skewed TEST stream turns every reported
    accuracy (and the E1/E2/E3 accuracy contrasts) into a class-prior
    metric.  The HB3 split audit compares each held-out marginal to the
    *training* reference, which is a different question - a uniformly
    skewed training stream and an equally skewed test stream can pass the
    shift test while both violate the accuracy premise.

    The audit is scoring-only: it draws from the fixed evaluation seeds and
    validates each batch through the shared cleaning gate, but fits nothing
    and selects nothing.  The Monte Carlo calibration and the finite-stream
    caveat are the same as ``_answer_balance``; the per-stream
    ``expected_per_class`` is reported so a small quick-mode draw is not
    mistaken for a well-powered balance certificate.

    E2 scores accuracy at every length in its sweep, so the TEST stream is
    also audited at each sweep length (the stream realization depends on
    the length).  The pre-registered E2 readouts use only the base and
    longest endpoints; the serialized per-length rows make a mid-sweep
    imbalance visible without letting an exploratory length gate the
    endpoint decision.
    """
    def _audit_stream(seed, length, batch, batches):
        check_eval_seed(seed)
        gen = stream_generator(seed, settings.device)
        counts = torch.zeros(settings.vocab, dtype=torch.float64)
        for _ in range(max(0, int(batches))):
            idx, targets, meta = make_batch(batch, length,
                                            settings.vocab,
                                            settings.device, gen=gen)
            validate_batch(idx, targets, meta, settings.vocab,
                           diagnostics=False)
            counts += torch.bincount(meta["answer"].cpu(),
                                     minlength=settings.vocab)
        present = counts > 0
        shares = counts[present] / counts.sum() if counts.sum() > 0 else \
            counts[present]
        uniformity = answer_uniformity_test(counts[2:])
        return {
            "stream_seed": int(seed),
            "seed_key": stream_seed_key(seed),
            "length": int(length),
            "batch": int(batch),
            "batches": int(batches),
            "n": int(counts.sum()),
            "n_classes_possible": settings.vocab - 2,
            "n_classes_present": int(present.sum()),
            "min_class_share": (float(shares.min()) if shares.numel()
                                else float("nan")),
            "max_class_share": (float(shares.max()) if shares.numel()
                                else float("nan")),
            "expected_uniform_share": 1.0 / (settings.vocab - 2),
            "uniformity": uniformity,
            "power_note": (
                "gross-imbalance test only: a pass at "
                f"{uniformity['expected_per_class']:.2f} expected "
                "samples/class does not certify the absence of a small or "
                "moderate skew"),
        }

    streams = {}
    for label, seed, geometry in _EVAL_BALANCE_STREAMS:
        if geometry == "stats":
            batch, batches = settings.stats_batch, settings.stats_batches
        elif geometry == "probe":
            batch, batches = (settings.probe_eval_batch,
                              settings.probe_eval_batches)
        else:
            batch, batches = settings.eval_batch, settings.eval_batches
        streams[label] = _audit_stream(seed, settings.n_pairs, batch, batches)
    all_uniform = all(row["uniformity"]["uniform_supported"]
                      for row in streams.values())

    try:
        e2_lengths = length_sweep_lengths(
            base=settings.n_pairs, factor=2, n=4,
            max_len=settings.max_len)
    except ValueError as exc:
        e2_sweep = {
            "lengths": [], "streams": {}, "endpoint_lengths": [],
            "endpoint_uniform": None, "all_uniform": None,
            "error": str(exc),
        }
    else:
        e2_streams = {}
        for length in e2_lengths:
            e2_streams[str(length)] = (
                dict(streams["test"]) if length == settings.n_pairs else
                _audit_stream(TEST_SEED, length, settings.eval_batch,
                              settings.eval_batches))
        endpoints = [settings.n_pairs, e2_lengths[-1]]
        e2_sweep = {
            "lengths": [int(L) for L in e2_lengths],
            "streams": e2_streams,
            "endpoint_lengths": [int(L) for L in endpoints],
            "endpoint_uniform": bool(all(
                e2_streams[str(L)]["uniformity"]["uniform_supported"]
                for L in endpoints)),
            "all_uniform": bool(all(
                row["uniformity"]["uniform_supported"]
                for row in e2_streams.values())),
        }
    return {
        "streams": streams,
        "all_uniform": bool(all_uniform),
        "e2_length_sweep": e2_sweep,
        "geometry": {
            "eval_batch": int(settings.eval_batch),
            "eval_batches": int(settings.eval_batches),
            "stats_batch": int(settings.stats_batch),
            "stats_batches": int(settings.stats_batches),
            "probe_eval_batch": int(settings.probe_eval_batch),
            "probe_eval_batches": int(settings.probe_eval_batches),
        },
        "note": ("uniformity is a gross-imbalance test at the reported "
                 "stream size, not an equivalence proof; expected_per_class "
                 "is printed so a low-power draw is not read as a "
                 "certificate"),
    }


class SplitAuditError(RuntimeError):
    """The HB3 split/disjointness audit failed; no grid may run on this data."""


# Minimum samples per stream for the calibrated shift tests.  The audit is
# not the evaluation stream: reusing a tiny quick-mode eval geometry (e.g.
# 64 x 2 = 128 rows, ~21 per key position) leaves the tests noisy and
# underpowered, so a legitimate random draw can land on the wrong side of
# the decision boundary.  Drawing at least this many rows at the eval batch
# size keeps the audit's decisions stable without touching the evaluation
# data (the audit only reads additional samples from the same split
# streams).
_SPLIT_AUDIT_MIN_SAMPLES = 1024


def _audit_data_splits(settings: Settings, out_dir: pathlib.Path,
                       tag: str = "baselines", enforce: bool = False) -> dict:
    """HB3: verify every stream pair is disjoint and field-wise IID before
    any model is trained.

    The audit only reads data (it selects nothing), but a shared sample
    between *any* two streams - validation vs test included - or a
    distribution gap between the training set and a held-out split
    invalidates every number reported afterwards, so the outcome is printed
    and stored under ``<tag>_split_audit.json``.  Beyond the
    ``split_leakage_audit`` prefix comparison, ``actual_stream_overlap_audit``
    replays every stream at the geometry its consumer actually reads
    (training at ``batch`` x ``steps``, eval at ``eval_batch`` x
    ``eval_batches``, ...) and the gate also refuses any cross-split content
    overlap found there: the prefix check alone cannot cover the full
    training stream, which ``make_batch``'s shape-dependent draws make a
    different realization from the eval-geometry prefix.  Every configured training
    seed enters the audit as its own stream and the exact-overlap check
    covers all stream pairs.  Each audit field's shift is decided by a Monte
    Carlo-calibrated Pearson chi-square against the pooled training
    distribution (see ``cage.data.split_leakage_audit``), Benjamini-
    Hochberg-adjusted over the field x split family at ``settings.fdr_q``:
    a total-variation readout against ``3 x noise floor`` cannot fail for
    sparse fields (a few hundred answer classes at ~1e3 samples) because
    the tolerance approaches the maximum possible TV, so the calibrated
    test - not an arbitrary constant - carries the decision.  The
    TV/floor pair stays in the report as a descriptive companion.  The
    replay also compares canonical stream seed keys: a split is a
    seed-owned stream, so the same seed reused by two groups is leakage
    even when shape-dependent draws leave their row fingerprints disjoint.

    A second, train-fitted gate covers what the marginal tests cannot: an
    *atypical-sample* tail.  ``split_leakage_audit`` scores every sample by
    its training-fitted surprisal over the structural audit fields and
    thresholds at the training tail quantile; a held-out split whose
    atypical rate exceeds five times the training rate (and an absolute 5%
    floor) fails HB3 even when every marginal passes.  Separately, a
    train-only first-half/second-half comparison reports training-stream
    non-stationarity (BH-adjusted within its own family); it warns rather
    than blocks, since drift is a property to disclose.

    The audit is deterministic for one ``Settings`` (fixed split seeds and
    a fixed Monte Carlo seed), so the computation runs once per process
    and every experiment that depends on the protocol writes its
    own copy.  ``enforce=True`` (baselines, E1/E2/E3 and the standalone
    pre-flight) turns a failed audit into a hard
    :class:`SplitAuditError`: a protocol with a shared sample or a
    held-out distribution shift cannot support any grid result, so the
    grid is not started (same policy as the reproducibility and overfit
    gates).
    """
    train_seeds = tuple(settings.seeds) or (0,)
    cache_key = _config_cache_key(settings)
    audit = getattr(settings, "_split_audit_cache", None)
    fresh = (audit is None
             or getattr(settings, "_split_audit_cache_key", None) != cache_key)
    if fresh:
        audit_batches = max(
            int(settings.eval_batches),
            math.ceil(_SPLIT_AUDIT_MIN_SAMPLES
                      / max(int(settings.eval_batch), 1)))
        audit = split_leakage_audit(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.eval_batch, batches=audit_batches,
            train_seeds=train_seeds)
        fields = audit["audit_fields"]
        family = [(field, name) for field in fields
                  for name in ("val", "test", "stats")]
        bh = benjamini_hochberg(
            [audit["shift_p_value"][field][name] for field, name in family],
            q=settings.fdr_q)
        shift_tests = {
            f"{field}_train_{name}": {
                "p": item["p"], "p_adj": item["p_adj"],
                "reject": item["reject"],
                "chi2": audit["shift_chi2"][field][name],
            }
            for (field, name), item in zip(family, bh)
        }
        audit["shift_p_value_bh"] = shift_tests
        # Prefix overlap is necessary but not sufficient: training reads
        # at least settings.steps * settings.batch samples at its own
        # geometry, which is a different realization from the
        # eval-geometry prefix.  The depth ladder and its capacity control
        # reuse the same seed-owned training stream for up to the largest
        # _depth_step_candidates budget, so replay that full prefix; a
        # longer in-order prefix subsumes every shorter training budget.
        # Replay the exact streams each consumer reads (training, val
        # monitoring, final val/test, stats, probe train/val/test) and
        # refuse any content overlap or cross-group seed reuse between
        # different splits.  Realizations of the same split (val monitoring
        # vs final val, probe val vs val) must remain allowed.
        training_steps = max(_depth_step_candidates(settings))
        actual = actual_stream_overlap_audit(
            settings.n_pairs, settings.vocab, settings.device,
            train_specs={
                f"train:{seed}": {
                    "seed": int(seed), "batch": settings.batch,
                    "steps": training_steps, "group": "train",
                }
                for seed in train_seeds
            } | {
                f"probe_train:{seed}": {
                    "seed": int(seed), "batch": settings.probe_batch,
                    "steps": max(settings.probe_steps_grid
                                 or (settings.probe_steps,)),
                    "group": "train",
                }
                for seed in train_seeds
            } | {
                "overfit_sanity": {
                    "seed": SANITY_SEED, "batch": settings.overfit_batch,
                    "steps": 1, "group": "train",
                },
            },
            heldout_specs={
                "val_monitor": {
                    "seed": VAL_SEED, "batch": settings.batch,
                    "batches": settings.val_batches, "group": "val",
                },
                "val": {
                    "seed": VAL_SEED, "batch": settings.eval_batch,
                    "batches": settings.eval_batches, "group": "val",
                },
                # The linear probe selects lr x steps on the probe-geometry
                # validation stream (``tune_linear_probe``), which is a
                # different realization from the eval-geometry ``val``
                # stream.  Replaying only ``probe_test`` would leave the
                # stream that actually drives a selection unaudited.
                "probe_val": {
                    "seed": VAL_SEED, "batch": settings.probe_eval_batch,
                    "batches": settings.probe_eval_batches, "group": "val",
                },
                "test": {
                    "seed": TEST_SEED, "batch": settings.eval_batch,
                    "batches": settings.eval_batches, "group": "test",
                },
                "stats": {
                    "seed": STATS_SEED, "batch": settings.stats_batch,
                    "batches": settings.stats_batches, "group": "stats",
                },
                "probe_test": {
                    "seed": TEST_SEED, "batch": settings.probe_eval_batch,
                    "batches": settings.probe_eval_batches,
                    "group": "test",
                },
            })
        audit["actual_stream_overlap"] = actual
        audit["max_actual_stream_overlap"] = actual["max_overlap"]
        audit["actual_stream_seed_collisions"] = actual[
            "cross_group_seed_collisions"]
        # Training-stationarity diagnostic (train-only halves): reported and
        # BH-adjusted across its own family, but it warns rather than blocks,
        # because a drifting training stream is a property to disclose, not
        # a leakage failure of the split.
        stationarity = audit.get("train_stationarity", {})
        st_keys = sorted(stationarity)
        st_bh = benjamini_hochberg(
            [stationarity[key]["p"] for key in st_keys],
            q=settings.fdr_q) if st_keys else []
        audit["train_stationarity_bh"] = {
            key: {"p": item["p"], "p_adj": item["p_adj"],
                  "reject": item["reject"],
                  "tv": stationarity[key]["tv"]}
            for key, item in zip(st_keys, st_bh)
        }
        # Statistical rejection alone would fire on ~5% of same-distribution
        # audits (a p-value family at q=0.05 must reject something that
        # often).  Require the descriptive effect size as well: the
        # first/second-half TV gap must clear the train-only noise floor by
        # the historical 3x margin.  Drift is then disclosed only when it is
        # both significant and materially larger than stream-to-stream noise.
        audit["train_drift_suspected"] = bool(any(
            item["reject"]
            and item.get("tv", 0.0) > 3.0 * audit["tv_noise_floor"].get(
                key.rsplit(":", 1)[-1], 0.0)
            for key, item in audit["train_stationarity_bh"].items()))
        outlier = audit.get("outlier_audit", {})
        outlier_shift = outlier.get("shift_by_split", {})
        audit["outlier_shift_detected"] = bool(any(outlier_shift.values()))
        audit["hb3_supported"] = bool(
            audit["max_pairwise_duplicate_overlap"] == 0
            and actual["max_overlap"] == 0
            and not actual["cross_group_seed_collisions"]
            and not any(item["reject"] for item in shift_tests.values())
            # A held-out split whose difficulty *tail* has moved passes the
            # marginal tests; the conservative train-fitted tail-rate bound
            # catches it before it can masquerade as a mechanism effect.
            and not audit["outlier_shift_detected"])
        settings._split_audit_cache = audit
        settings._split_audit_cache_key = cache_key

    fields = audit["audit_fields"]
    floors = audit["tv_noise_floor"]
    shift_tests = audit["shift_p_value_bh"]
    rejected = [key for key, item in shift_tests.items() if item["reject"]]
    if fresh:
        detail = "; ".join(
            f"{name}: tv_max={max(audit[f'{field}_tv_train_{name}']
                                  for field in fields):.5f}"
            for name in ("val", "test", "stats"))
        floor_txt = ", ".join(f"{field}={floors[field]:.5f}" for field in fields)
        adj_txt = ", ".join(
            f"{field}_train_{name}={shift_tests[f'{field}_train_{name}']['p_adj']:.3f}"
            for field in fields for name in ("val", "test", "stats"))
        dup_txt = ", ".join(
            f"{name}={audit[f'n_duplicate_rows_{name}']}"
            for name in ("val", "test", "stats"))
        outlier = audit.get("outlier_audit", {})
        outlier_txt = ", ".join(
            f"{name}={outlier['rate_by_split'][name]:.4f}"
            for name in ("val", "test", "stats")) if outlier else "n/a"
        coverage = audit.get("support_coverage", {})
        cov_txt = ", ".join(
            f"{field}: train_support="
            f"{coverage[field]['fraction_seen_train']:.3f} "
            + "/".join(
                f"{name}={coverage[field]['unseen_by_split'][name]['rate']:.4f}"
                for name in ("val", "test", "stats"))
            for field in ("answer", "token", "key", "value")
            if field in coverage) if coverage else "n/a"
        actual = audit["actual_stream_overlap"]
        print(f"[{tag}] split audit (HB3, train seeds {list(train_seeds)}): "
              f"{detail}; pairwise overlap max="
              f"{audit['max_pairwise_duplicate_overlap']} over "
              f"{len(audit['pairwise_duplicate_overlap'])} stream pairs; "
              f"actual stream replay={actual['n_rows_total']} rows, "
              f"cross-split overlap max={actual['max_overlap']}, "
              f"cross-group seed collisions="
              f"{actual['cross_group_seed_collisions'] or 'none'}; "
              f"TV noise floor [{floor_txt}]; calibrated shift test "
              f"(BH q={settings.fdr_q:g}) adjusted p: [{adj_txt}]; "
              f"atypical-sample rate train="
              f"{outlier.get('rate_train', float('nan')):.4f} "
              f"(bound={outlier.get('rate_bound', float('nan')):.4f}) "
              f"[{outlier_txt}]; train stationarity min p="
              f"{audit.get('train_stationarity_min_p', 1.0):.3f}; "
              f"unseen-value rates [{cov_txt}]; "
              f"stream duplicates train="
              f"{audit['n_duplicate_rows_train']} ({dup_txt}); "
              f"unique_key_rate={audit['unique_key_rate']:.4f}")
    else:
        print(f"[{tag}] split audit (HB3, train seeds {list(train_seeds)}): "
              f"deterministic result reused; hb3_supported="
              f"{audit['hb3_supported']}")
    if rejected:
        print(f"[{tag}] WARNING: calibrated shift test rejects "
              f"{', '.join(rejected)}: held-out counts differ from the "
              f"pooled training distribution beyond same-stream sampling "
              f"noise")
    if audit.get("outlier_shift_detected"):
        flagged = [f"{name}={audit['outlier_audit']['rate_by_split'][name]:.4f}"
                   for name, bad in
                   audit["outlier_audit"]["shift_by_split"].items() if bad]
        print(f"[{tag}] WARNING: atypical-sample tail rate exceeds the "
              f"train-fitted bound (bound="
              f"{audit['outlier_audit']['rate_bound']:.4f}): "
              f"{', '.join(flagged)}; the held-out difficulty tail differs "
              f"from training")
    if audit.get("train_drift_suspected"):
        drift = [key for key, item in
                 audit.get("train_stationarity_bh", {}).items()
                 if item["reject"]
                 and item.get("tv", 0.0) > 3.0 * audit["tv_noise_floor"].get(
                     key.rsplit(":", 1)[-1], 0.0)]
        print(f"[{tag}] WARNING: training-stream non-stationarity suspected "
              f"({', '.join(drift)}): first and second halves of the "
              f"training stream differ beyond sampling noise, so a single "
              f"held-out split may not represent the future")
    if not audit["hb3_supported"]:
        print(f"[{tag}] WARNING: HB3 split/disjointness audit failed; "
              f"all reported numbers are suspect")
    _write_json(out_dir / f"{tag}_split_audit.json", audit)
    if enforce and not audit["hb3_supported"]:
        _write_json(out_dir / f"{tag}_aborted.json", {
            "experiment": tag,
            "reason": "HB3 split/disjointness audit failed",
            "rejected_shift_tests": rejected,
            "outlier_shift_detected": audit.get("outlier_shift_detected"),
            "train_drift_suspected": audit.get("train_drift_suspected"),
            "max_pairwise_duplicate_overlap":
                audit["max_pairwise_duplicate_overlap"],
            "max_actual_stream_overlap":
                audit.get("max_actual_stream_overlap"),
            "cross_group_seed_collisions":
                audit.get("actual_stream_seed_collisions"),
            "split_audit": audit,
            **_run_context(settings),
        })
        raise SplitAuditError(
            f"{tag}: HB3 failed (max pairwise duplicate overlap="
            f"{audit['max_pairwise_duplicate_overlap']}, max actual-stream "
            f"overlap={audit.get('max_actual_stream_overlap')}, cross-group "
            f"seed collisions="
            f"{audit.get('actual_stream_seed_collisions') or 'none'}, "
            f"rejected shift tests={rejected or 'none'}); a data protocol "
            f"with leakage or a held-out distribution shift cannot support "
            f"any grid result")
    return audit


def _run_depth_ladder(settings: Settings, out_dir: pathlib.Path,
                      ceiling: dict | None) -> dict:
    """HB6/HB7 complexity ladder: the tuned standard transformer per depth.

    The simplest *learned* model on the baseline ladder is the 1-layer
    transformer (HB6), but a single shallow point cannot say where depth
    stops mattering.  This runner tunes the learning rate *and step
    budget* of the standard ``gate="none"`` transformer separately at every
    ladder depth (default 1/2/4 layers) through the same entry point as the
    deep grids - same optimizer factory, cosine schedule, batch geometry
    and evaluation streams - on the validation seed (over the
    pre-registered ``min_seeds_for_claim`` floor's worth of training seeds,
    so no rung's recipe is frozen on a two-seed argmax), and reports the
    chosen recipe on the disjoint test seed.  The step grid is expressed as
    multipliers of ``settings.steps``; a rung whose optimum sits at either
    axis boundary is flagged by the selection audit as unresolved.  Depth is the only architectural difference from
    E1's ``none`` arm, so:

    * each rung's gap to the parameter-free rule ceiling measures how much
      depth the recall readout actually requires - but only rungs whose
      seed-clustered 95% *lower* bound clears chance count as evidence:
      the HB0 competence premise separates "needs more depth" from "this
      step count did not learn the task", and with no competent rung the depth
      demand is reported as *not determinable* instead of "required";
    * the saturation depth ``d*`` (smallest competent rung whose
      seed-clustered 95% *lower* bound lies within the pre-registered SESOI
      of the ceiling estimate, i.e. a one-sided equivalence statement) is the
      reference the deep grids must justify themselves against (HB7): a
      gain a 2-layer standard transformer already matches is not evidence
      about depth.  The bare "the CI touches the ceiling" criterion is an
      *absence-of-evidence* reading and is kept only as the companion
      ``reaches_ceiling_ci``; the same SESOI is the minimum plausible gap
      for "depth measurably required" (``ceiling - clustered_hi >= SESOI``),
      so a statistically detectable but practically negligible gap cannot
      earn the complexity;
    * a complex model is only credited for what it does beyond the tuned
      shallow baseline.

    Each depth's argmax is audited by
    :func:`_selection_reliability` (margin vs runner-up in seed units,
    winning curve verdicts, grid-boundary position) and stored in the
    payload's ``selection`` block: a rung whose tuned rate is not
    resolved is a rung whose accuracy may understate the depth being
    credited, and the warning says so at selection time.

    Returns ``{"depths": {depth: payload}, "ladder": [...],
    "saturation_depth": int|None, "hb7_depth_required": bool|None,
    "shallowest": depth}``; the shallow rung reuses the legacy names
    (``shallow_none_baseline`` records, ``baselines_shallow_none.json``)
    that E1's complexity readout reads from the verdict.
    """
    raw_ladder = tuple(settings.depth_ladder)
    depths = tuple(_positive_integer(d) for d in raw_ladder)
    if not depths or any(d is None for d in depths):
        raise ValueError(
            f"depth_ladder must be non-empty positive ints, got "
            f"{raw_ladder!r}")
    ladder = tuple(sorted(set(depths)))
    tune_seeds = settings.seeds[:max(1, min(settings.min_seeds_for_claim,
                                             len(settings.seeds)))]
    ceiling_acc = float(ceiling["accuracy"]) if ceiling else float("nan")
    ceiling_known = math.isfinite(ceiling_acc)
    chance = 1.0 / (settings.vocab - 2)
    # HB0 is a union claim over the ladder rungs ("at least one rung is
    # competent"), so the per-rung lower quantile is Bonferroni-adjusted by
    # the number of rungs exactly as ``_competence_check`` does for the
    # E1/E2/E3 arm families.  Without the adjustment the ladder could call
    # a rung competent on an uncorrected one-sided bound while the
    # registered family-wise criterion would not.
    competence_alpha = 0.05 / max(len(ladder), 1)
    # Pre-registered practical-significance bound for the depth readouts:
    # HB6/HB7 both register ``sesoi: {accuracy: 0.01}``, so the same bound
    # decides "saturated" (one-sided equivalence within the band) and
    # "depth measurably required" (plausible gap at least the band).
    sesoi = float((HYPOTHESES["HB6"].get("sesoi") or {}).get("accuracy", 0.01))
    if float((HYPOTHESES["HB7"].get("sesoi") or {}).get("accuracy", sesoi)) \
            != sesoi:
        raise ValueError(
            "HB6/HB7 must share one registered accuracy SESOI for the "
            "depth ladder's saturation rule")
    depths: dict[int, dict] = {}
    tuning_tables: dict[int, list] = {}
    step_candidates = _depth_step_candidates(settings)
    for depth in ladder:
        table = []
        for train_steps in step_candidates:
            for lr in settings.lr_grid:
                accs, verdicts, tuning_runs = [], [], []
                for seed in tune_seeds:
                    _, params, metrics, curve, _ = _train_and_eval(
                        settings, seed, lr, eval_seed=VAL_SEED,
                        selection=True, steps=train_steps, gate="none",
                        n_layers=depth)
                    accs.append(metrics["accuracy"])
                    verdicts.append(curve["curve_verdict"])
                    tuning_runs.append({
                        "seed": int(seed), "params": int(params),
                        "val_accuracy": metrics["accuracy"],
                        "eval_seed": int(VAL_SEED), "curve": curve})
                mean = sum(accs) / len(accs)
                std = ((sum((a - mean) ** 2 for a in accs)
                        / (len(accs) - 1)) ** 0.5 if len(accs) > 1 else 0.0)
                table.append({"lr": lr, "steps": int(train_steps),
                              "val_accuracy_mean": mean,
                              "val_accuracy_std": std,
                              "n_seeds": len(accs),
                              "curve_verdicts": verdicts,
                              "curve_invalid_runs": sum(
                                  v in CURVE_INVALID_VERDICTS
                                  for v in verdicts),
                              "runs": tuning_runs})
        # An all-non-finite tuning table (every run diverged / produced no
        # finite validation accuracy) must not be silently read as an argmax:
        # NaN comparisons make ``max`` return an arbitrary row, which would
        # freeze an unmeasured learning rate and manufacture a rung.
        finite_rows = [row for row in table
                       if _finite(row["val_accuracy_mean"])]
        # Audit the full table for the unmeasured branch; the selected
        # (curve-valid) pool is audited again below.  Both axes matter: a
        # winner at the 2x step edge is as unresolved as one at the lr edge.
        selection = _selection_reliability(
            table, grid=[(row["lr"], row["steps"]) for row in table],
            grid_keys=("lr", "steps"))
        if not finite_rows:
            print(f"[baselines] WARNING: depth={depth} tuning table has no "
                  f"finite validation accuracy (all {len(table)} "
                  f"lr x steps candidates non-finite); the rung is recorded "
                  f"as unmeasured and cannot support HB6/HB7 or the "
                  f"saturation depth")
            depths[depth] = {
                "name": f"shallow_none_d{depth}", "n_layers": depth,
                "val_seed": VAL_SEED, "test_seed": TEST_SEED,
                "chosen_lr": None, "chosen_steps": None,
                "tuning_table": table,
                "selection": selection,
                "accuracy_mean": float("nan"),
                "accuracy_std": float("nan"), "accuracy_by_seed": {},
                "pooled_wilson": {"p": float("nan"), "lo": float("nan"),
                                  "hi": float("nan"), "n": 0},
                "cluster_ci": clustered_proportion_ci([], []),
                "ceiling": ceiling_acc, "chance": chance,
                "sesoi_accuracy": sesoi,
                "above_chance": False, "ceiling_gap": float("nan"),
                "ceiling_gap_lower": float("nan"),
                "reaches_ceiling": False, "reaches_ceiling_ci": False,
                "hb6_supported": False, "hb6_supported_ci": False,
                "unmeasured": True, "runs": [],
            }
            tuning_tables[depth] = table
            _write_json(out_dir / f"baselines_depth{depth}.json",
                        depths[depth])
            continue
        pick = _curve_valid_selection_rows(table)
        if pick["n_excluded"]:
            print(f"[baselines] depth={depth} selection excluded "
                  f"{pick['n_excluded']} of {pick['n_finite']} finite "
                  f"lr x steps row(s) with curve-invalid runs; selecting "
                  f"among curve-valid rows only")
        elif pick["n_invalid"]:
            print(f"[baselines] WARNING: depth={depth} has no curve-valid "
                  f"finite lr x steps row; falling back to all "
                  f"{pick['n_finite']} finite row(s) - the rung's recipe is "
                  f"not validated by a healthy curve")
        eligible = pick["rows"]
        selection = _selection_reliability(
            eligible, grid=[(row["lr"], row["steps"]) for row in eligible],
            grid_keys=("lr", "steps"))
        selection["pool"] = pick["pool"]
        selection["excluded_curve_invalid_rows"] = pick["n_excluded"]
        selection["curve_invalid_rows"] = pick["n_invalid"]
        best = max(eligible, key=lambda row: row["val_accuracy_mean"])
        warn = _selection_warn(
            f"depth={depth} lr x steps selection", selection)
        if warn:
            print(f"[baselines] {warn}: this rung's tuned accuracy may "
                  f"understate the depth it is credited with")
        runs = []
        for seed in settings.seeds:
            model, params, metrics, curve, _ = _train_and_eval(
                settings, seed, best["lr"], eval_seed=TEST_SEED, gate="none",
                n_layers=depth, steps=best["steps"])
            n = int(metrics["n"])
            runs.append({
                "seed": seed, "accuracy": metrics["accuracy"],
                "accuracy_lo": metrics["accuracy_lo"],
                "accuracy_hi": metrics["accuracy_hi"], "n": n,
                "correct": int(round(metrics["accuracy"] * n)),
                "params": params, "curve_verdict": curve["curve_verdict"],
            })
            name = ("shallow_none_baseline" if depth == ladder[0]
                    else f"shallow_none_baseline_d{depth}")
            _write_result(out_dir / "baselines_runs.jsonl",
                          {"experiment": "baselines",
                           "hypotheses": ["HB6", "HB7"], "name": name,
                           "n_layers": depth, "lr": best["lr"],
                           "steps": int(best["steps"]),
                           "training_seed": seed, "eval_seed": TEST_SEED,
                           "accuracy": metrics["accuracy"],
                           "accuracy_lo": metrics["accuracy_lo"],
                           "accuracy_hi": metrics["accuracy_hi"], "n": n,
                           "params": params,
                           **_training_record(settings, best["lr"],
                                              best["steps"]),
                           **_run_context(settings), **curve})
        # Curve-invalid final runs (plateau/diverging/instability) cannot
        # support a depth claim: their accuracies measure an optimisation
        # failure, not the depth's capacity.  The rung's decision statistics
        # therefore read only the curve-valid seeds; every run stays in the
        # record so the exclusion is visible.
        curve_invalid_seeds = sorted(
            int(r["seed"]) for r in runs
            if r.get("curve_verdict") in CURVE_INVALID_VERDICTS)
        valid_runs = [r for r in runs if r["seed"] not in curve_invalid_seeds]
        if curve_invalid_seeds:
            print(f"[baselines] WARNING: depth={depth} final runs "
                  f"{curve_invalid_seeds} are curve-invalid and are excluded "
                  f"from the rung's competence/saturation statistics")
        if valid_runs:
            pooled = wilson_ci(sum(r["correct"] for r in valid_runs),
                               sum(r["n"] for r in valid_runs))
            # Seeds share one sealed evaluation stream, so the decision
            # interval must resample seeds and keep a shared finite-stream
            # error; pooling the counts as independent examples would
            # pseudo-replicate.
            cluster = clustered_proportion_ci(
                [r["correct"] for r in valid_runs],
                [r["n"] for r in valid_runs],
                lower_alpha=competence_alpha)
            summary = _mean_std([r["accuracy"] for r in valid_runs])
        else:
            pooled = {"p": float("nan"), "lo": float("nan"),
                      "hi": float("nan"), "n": 0}
            cluster = clustered_proportion_ci([], [])
            summary = _mean_std([])
        # A rung whose final runs are all curve-invalid has no accuracy
        # estimate at all.  It is unmeasured, not "at chance": treating it
        # as measured would let HB7 claim depth is required while a rung
        # that could still saturate was silently discarded.
        unmeasured = not bool(valid_runs)
        # HB0 competence premise for HB6/HB7: "below the ceiling" is only
        # evidence that depth is required when the rung demonstrably
        # learned the task.  A rung at chance is an underfit/step-count
        # statement, not a depth statement, so it can neither support HB6
        # ("depth measurably required") nor feed the saturation depth.
        above_chance = bool(cluster["valid"] and math.isfinite(cluster["lo"])
                            and cluster["lo"] > chance)
        # The registered SESOI turns both depth readouts from
        # absence-of-evidence into practical claims:
        #   * reaches_ceiling (saturation): one-sided equivalence - even the
        #     pool's *lower* bound lies within SESOI of the ceiling, so the
        #     rung is practically at the ceiling;
        #   * hb6_supported ("depth measurably required"): the pool's
        #     *upper* bound still sits at least SESOI below the ceiling, so
        #     the gap is practically meaningful at the optimistic end, not
        #     merely a statistically detectable sliver.
        # The historical bare-CI readings are kept as companions only.
        within_sesoi = bool(
            ceiling_known and above_chance and math.isfinite(cluster["lo"])
            and ceiling_acc - cluster["lo"] <= sesoi)
        reaches_ceiling_ci = bool(
            ceiling_known and above_chance and math.isfinite(cluster["hi"])
            and cluster["hi"] >= ceiling_acc)
        hb6_supported = bool(
            ceiling_known and above_chance and math.isfinite(cluster["hi"])
            and ceiling_acc - cluster["hi"] >= sesoi)
        hb6_supported_ci = bool(
            ceiling_known and above_chance and math.isfinite(cluster["hi"])
            and cluster["hi"] < ceiling_acc)
        ceiling_gap = (ceiling_acc - summary["mean"]
                       if ceiling_known else float("nan"))
        ceiling_gap_lower = (ceiling_acc - cluster["hi"]
                             if math.isfinite(cluster["hi"])
                             and ceiling_known else float("nan"))
        payload = {
            "name": f"shallow_none_d{depth}", "n_layers": depth,
            "val_seed": VAL_SEED, "test_seed": TEST_SEED,
            "chosen_lr": best["lr"], "chosen_steps": int(best["steps"]),
            "tuning_table": table, "selection": selection,
            "accuracy_mean": summary["mean"], "accuracy_std": summary["std"],
            # Per-seed accuracies behind ``accuracy_mean``: curve-invalid
            # runs are excluded exactly as they are from the mean/std and
            # from the rung's other decision statistics.  Keeping them here
            # would let E1 pair its valid deep runs against shallow runs that
            # the ladder itself marked as optimisation failures (the raw
            # runs remain available under ``runs`` for transparency).
            "accuracy_by_seed": {str(r["seed"]): r["accuracy"]
                                 for r in valid_runs},
            "curve_invalid_seeds": curve_invalid_seeds,
            "curve_valid_seeds": [int(r["seed"]) for r in valid_runs],
            "n_curve_invalid": len(curve_invalid_seeds),
            "pooled_wilson": pooled, "cluster_ci": cluster,
            "ceiling": ceiling_acc,
            "chance": chance, "above_chance": above_chance,
            "competence_alpha": competence_alpha,
            "sesoi_accuracy": sesoi,
            "ceiling_gap": ceiling_gap,
            # Smallest plausible gap to the ceiling (upper-bound side): the
            # SESOI gate for "depth measurably required" reads this.
            "ceiling_gap_lower": ceiling_gap_lower,
            "reaches_ceiling": within_sesoi,
            "reaches_ceiling_ci": reaches_ceiling_ci,
            "hb6_supported": hb6_supported,
            "hb6_supported_ci": hb6_supported_ci,
            "unmeasured": unmeasured,
            "runs": runs,
        }
        depths[depth] = payload
        tuning_tables[depth] = table
        if unmeasured:
            status = (f"unmeasured (all {len(runs)} final runs are "
                      f"curve-invalid: {curve_invalid_seeds}); this rung "
                      f"cannot support HB6/HB7 or the saturation depth")
        elif within_sesoi:
            status = (f"within SESOI {sesoi:g} of the ceiling (saturated: "
                      f"even the seed-clustered lower bound is inside the "
                      f"practical band, reach={ceiling_gap:+.4f})")
        elif not above_chance:
            status = (f"at chance (seed-clustered lower bound "
                      f"{cluster['lo']:.4f} <= chance {chance:.4f}): this "
                      f"rung did not learn the task at this step count, so it "
                      f"is not depth evidence")
        elif reaches_ceiling_ci:
            status = (f"CI touches the ceiling but the seed-clustered lower "
                      f"bound is more than SESOI={sesoi:g} below it "
                      f"(reach={ceiling_gap:+.4f}): saturation not "
                      f"established")
        else:
            status = "below the ceiling"
        print(f"[baselines] HB6/HB7 depth={depth} tuned standard "
              f"transformer (lr={best['lr']:g}, steps={int(best['steps'])} "
              f"on val seed {VAL_SEED}, test seed {TEST_SEED}): "
              f"accuracy={summary['mean']:.4f} +- "
              f"{summary['std']:.4f} (seed-clustered "
              f"[{cluster['lo']:.4f}, {cluster['hi']:.4f}], "
              f"n={cluster['n']} seeds/{cluster['n_examples']} examples; "
              f"pooled Wilson companion "
              f"[{pooled['lo']:.4f}, {pooled['hi']:.4f}]); "
              f"chance={chance:.4f}; rule ceiling={ceiling_acc:.4f}, gap="
              f"{ceiling_gap:.4f} (lower-bound gap="
              f"{ceiling_gap_lower:.4f}); params={int(runs[0]['params'])}; "
              f"{status}")
        _write_json(out_dir / f"baselines_depth{depth}.json", payload)

    ladder_competent = any(depths[d]["above_chance"] for d in ladder)
    unmeasured_rungs = [d for d in ladder if depths[d].get("unmeasured")]
    saturation_depth = next((d for d in ladder if depths[d]["reaches_ceiling"]),
                            None)
    saturation_depth_ci = next(
        (d for d in ladder if depths[d]["reaches_ceiling_ci"]), None)
    # HB7's competence premise: when no rung clears chance the ladder did
    # not solve the task at this step count, so the depth demand is "not
    # determinable" - never "required" (a None saturation depth alone must
    # not be read as "needs more depth").
    if not ceiling_known or not ladder_competent:
        hb7_depth_required = None
    elif saturation_depth is None:
        # An unmeasured rung could still saturate; without it the absence
        # of a saturation point cannot be read as "depth required".
        hb7_depth_required = None if unmeasured_rungs else True
    elif saturation_depth <= 2:
        # A *measured* saturation at depth <= 2 settles the registered
        # question ("is depth beyond two layers required?") even if a
        # shallower rung was unmeasured: d* can only move lower, so it
        # cannot become > 2.
        hb7_depth_required = False
    elif any(d < saturation_depth for d in unmeasured_rungs):
        # The measured saturation point is deeper than 2, but an
        # unmeasured *shallower* rung could itself have saturated within
        # SESOI.  In that case d* could be <= 2, so "depth beyond two
        # layers is required" is not determinable.  Unmeasured rungs
        # deeper than the measured saturation point cannot lower d*.
        hb7_depth_required = None
    else:
        hb7_depth_required = True
    adjacent = {}
    for a, b in zip(ladder, ladder[1:]):
        valid_a = [r for r in depths[a]["runs"]
                   if r.get("seed") is not None
                   and r.get("curve_verdict") not in CURVE_INVALID_VERDICTS]
        by_seed_b = {r["seed"]: r for r in depths[b]["runs"]
                     if r.get("seed") is not None
                     and r.get("curve_verdict") not in CURVE_INVALID_VERDICTS}
        # Positive mean = the deeper rung gains accuracy over the shallower
        # one, matching the ``a->b`` arrow and the "paired gain" reading
        # below (the TOST is symmetric, so only the reported sign moves).
        diffs = [by_seed_b[r["seed"]]["accuracy"] - r["accuracy"]
                 for r in valid_a if r["seed"] in by_seed_b]
        if len(diffs) < 2:
            # One paired seed gives a zero-width percentile interval and a
            # degenerate TOST (equivalence_test([0.0], bound) returns
            # equivalent=True), which is not a seed-clustered claim.
            adjacent[f"{a}->{b}"] = {
                "n": len(diffs), "sesoi_accuracy": sesoi,
                "equivalent": False, "unmeasured": True,
            }
            reason = ("no paired runs (unmeasured rung)" if not diffs else
                      "only one paired run; at least two independent pairs "
                      "are needed for the bootstrap/TOST read")
            print(f"[baselines] HB7 adjacent-depth contrast {a}->{b} layers: "
                  f"{reason}; skipped")
            continue
        ci = paired_bootstrap_ci(diffs)
        # The paired gain gets the same practical-significance reading as
        # the point rungs: a gain whose paired interval sits entirely
        # within +-SESOI is no longer *practically* meaningful even if its
        # point estimate is positive, matching the saturation rule.
        eq = equivalence_test(diffs, sesoi)
        adjacent[f"{a}->{b}"] = {
            **ci,
            "sesoi_accuracy": sesoi,
            "within_sesoi": bool(eq["equivalent"]),
            "tost_p_value": eq["p_value"],
            "tost_n": eq["n"],
        }
        print(f"[baselines] HB7 adjacent-depth contrast {a}->{b} layers "
              f"(paired over seeds; each rung independently val-tuned over "
              f"lr x steps, so this is a performance-envelope contrast, not "
              f"a controlled single-recipe depth effect): {_fmt_ci(ci)}; "
              f"companion TOST vs SESOI={sesoi:g}: "
              f"{eq['mean']:+.4f} [{eq['lo']:+.4f}, {eq['hi']:+.4f}] "
              f"n={eq['n']} p={eq['p_value']:.3f} "
              f"({'within SESOI' if eq['equivalent'] else 'beyond SESOI'})")
    shallowest = ladder[0]
    ladder_payload = {
        "name": "depth_ladder_none",
        "val_seed": VAL_SEED, "test_seed": TEST_SEED,
        "ladder": list(ladder), "ceiling": ceiling_acc,
        "sesoi_accuracy": sesoi,
        "saturation_rule": ("smallest competent rung whose seed-clustered "
                            "95% lower bound lies within SESOI of the rule "
                            "ceiling (one-sided equivalence); the bare "
                            "'interval reaches the ceiling' companion is "
                            "reaches_ceiling_ci only"),
        "depth_contrast_note": (
            "each rung is independently val-tuned over lr x steps, so "
            "adjacent-depth contrasts describe achievable performance "
            "envelopes rather than a controlled single-recipe depth effect; "
            "the selection payload records both axes and boundary flags"),
        "depths": {str(d): depths[d] for d in ladder},
        "tuning_tables": {str(d): tuning_tables[d] for d in ladder},
        "adjacent_contrasts": adjacent,
        "saturation_depth": saturation_depth,
        "saturation_depth_ci": saturation_depth_ci,
        "hb7_depth_required": hb7_depth_required,
        "chance": chance,
        "competence": {
            "criterion": ("at least one rung's seed-clustered one-sided 95% "
                          "family-wise lower bound above chance "
                          "(Bonferroni over the ladder rungs)"),
            "chance": chance,
            "alpha_per_rung": competence_alpha,
            "ladder_competent": ladder_competent,
            "above_chance_by_depth": {str(d): depths[d]["above_chance"]
                                      for d in ladder},
        },
    }
    sat_txt = (saturation_depth if saturation_depth is not None
               else "none (no rung reaches the ceiling)")
    if saturation_depth is None and saturation_depth_ci is not None:
        sat_txt = (f"none within SESOI {sesoi:g} (depth "
                   f"{saturation_depth_ci} only touches the ceiling - "
                   f"absence of evidence, not saturation)")
    if hb7_depth_required is None:
        if unmeasured_rungs:
            depth_txt = (f"not determinable - depth(s) {unmeasured_rungs} "
                         f"are unmeasured, so the ladder cannot establish "
                         f"whether depth is required")
        elif not ladder_competent:
            depth_txt = ("not determinable - no rung is above chance, so "
                         "the ladder did not solve the task at this step "
                         "count (raise the step count before reading depth)")
        elif not ceiling_known:
            depth_txt = "n/a (rule ceiling unavailable)"
        else:
            depth_txt = ("not determinable - an unmeasured rung could still "
                         "saturate, so the ladder cannot establish whether "
                         "depth is required")
    elif hb7_depth_required:
        depth_txt = "measurably required"
    else:
        depth_txt = ("NOT measurably required - annotate every depth-based "
                     "motivation")
    print(f"[baselines] HB7 saturation: depth ladder {list(ladder)} -> "
          f"saturation_depth={sat_txt}; depth beyond two layers is "
          f"{depth_txt}")
    # Legacy single-depth artifacts: E1 reads the shallow rung
    # (``shallow_none``) and its tuning table from the verdict.
    _write_json(out_dir / "baselines_shallow_none.json", depths[shallowest])
    _write_json(out_dir / "baselines_shallow_tuning.json",
                {"val_seed": VAL_SEED, "table": tuning_tables[shallowest],
                 "chosen_lr": depths[shallowest]["chosen_lr"],
                 "chosen_steps": depths[shallowest]["chosen_steps"],
                 "selection": depths[shallowest]["selection"]})
    _write_json(out_dir / "baselines_depth_ladder.json", ladder_payload)
    return {**ladder_payload, "depths": depths, "shallowest": shallowest}


def _matched_shallow_control(settings: Settings, out_dir: pathlib.Path,
                             ladder: dict) -> dict:
    """Train a parameter-matched 1-layer control for the depth ladder.

    The depth-only ladder holds width fixed, so a deep-vs-shallow gain
    confounds depth with parameter count.  This control trains a 1-layer
    ``gate="none"`` transformer widened until its parameter count is as
    close as possible to the deepest ladder rung, tunes its learning rate
    and step budget on the same validation seed/grid, and reports it on the
    sealed test stream.  It is a *capacity* control, not a mechanism contrast: depth and
    width both differ from the deep rung, so it answers only "could a
    shallow model with comparable capacity reach the deep model's accuracy?"
    (a no/yes answer changes how the depth framing is read, not the gate
    contrast itself).

    The width is chosen by binary search over ``d_model`` multiples of
    ``n_heads``; the payload records both parameter counts and the relative
    mismatch, and marks the control ``capacity_matched=False`` when no width
    in the search range reaches the target.  All selection uses VAL_SEED;
    the final runs use TEST_SEED exactly like the ladder rungs.
    """
    target_depth = int(ladder["ladder"][-1])
    target_payload = (ladder["depths"].get(target_depth)
                      or ladder["depths"].get(str(target_depth)))
    target_runs = target_payload.get("runs") or []
    target_params = next((int(r["params"]) for r in target_runs
                          if _finite(r.get("params"))), None)
    base = {
        "name": "matched_shallow_none",
        "control_for_depth": target_depth,
        "target_params": target_params,
        "n_layers": 1,
        "val_seed": VAL_SEED, "test_seed": TEST_SEED,
        "chance": ladder.get("chance"),
        "ceiling": ladder.get("ceiling"),
        "sesoi_accuracy": ladder.get("sesoi_accuracy"),
        "comparison": "capacity_control",
        "note": ("parameter-matched width control: depth and width both "
                 "differ from the deep rung, so this is not a depth or "
                 "mechanism contrast; it only tests whether comparable "
                 "capacity at one layer closes the gap"),
    }
    if target_params is None:
        base.update({"d_model": None, "params": None,
                     "capacity_matched": False, "unmeasured": True,
                     "reason": "target rung has no recorded parameter count",
                     "runs": []})
        _write_json(out_dir / "baselines_matched_shallow.json", base)
        return base

    step = int(settings.n_heads)

    def _params_for(d_model):
        model = build(_model_cfg(settings, gate="none", n_layers=1,
                                 d_model=int(d_model)), compile=False)
        return int(sum(p.numel() for p in model.parameters()))

    # Binary search the smallest multiple of n_heads whose parameter count
    # reaches the target; also keep the previous multiple so the closest
    # (rather than only the first overshoot) can be selected.
    lo_m, hi_m = 1, max(1, int(settings.d_model) // step)
    while _params_for(hi_m * step) < target_params and hi_m < 8192 // step:
        hi_m *= 2
    while lo_m < hi_m:
        mid = (lo_m + hi_m) // 2
        if _params_for(mid * step) < target_params:
            lo_m = mid + 1
        else:
            hi_m = mid
    candidates = sorted({max(step, (lo_m - 1) * step), lo_m * step,
                         max(step, (lo_m + 1) * step)})
    wide_d = min(candidates,
                 key=lambda d: (abs(_params_for(d) - target_params),
                                _params_for(d) < target_params, d))
    wide_params = _params_for(wide_d)
    matched = bool(abs(wide_params - target_params)
                   <= 0.05 * max(target_params, 1))
    base.update({"d_model": int(wide_d), "params": int(wide_params),
                 "capacity_matched": matched,
                 "params_delta": int(wide_params - target_params),
                 "params_delta_frac": ((wide_params - target_params)
                                       / max(target_params, 1))})
    if not matched:
        print(f"[baselines] WARNING: matched-shallow control width d_model="
              f"{wide_d} gives {wide_params} params vs target "
              f"{target_params} (>5% mismatch); the control is recorded "
              f"but capacity_matched=False")

    tune_seeds = settings.seeds[:max(1, min(settings.min_seeds_for_claim,
                                             len(settings.seeds)))]
    step_candidates = _depth_step_candidates(settings)
    table = []
    for train_steps in step_candidates:
        for lr in settings.lr_grid:
            accs, verdicts, tuning_runs = [], [], []
            for seed in tune_seeds:
                _, params, metrics, curve, _ = _train_and_eval(
                    settings, seed, lr, eval_seed=VAL_SEED, selection=True,
                    steps=train_steps, gate="none", n_layers=1,
                    d_model=wide_d)
                accs.append(metrics["accuracy"])
                verdicts.append(curve["curve_verdict"])
                tuning_runs.append({
                    "seed": int(seed), "params": int(params),
                    "val_accuracy": metrics["accuracy"],
                    "eval_seed": int(VAL_SEED), "curve": curve})
            mean = sum(accs) / len(accs)
            std = ((sum((a - mean) ** 2 for a in accs)
                    / (len(accs) - 1)) ** 0.5 if len(accs) > 1 else 0.0)
            table.append({"lr": lr, "steps": int(train_steps),
                          "val_accuracy_mean": mean,
                          "val_accuracy_std": std, "n_seeds": len(accs),
                          "curve_verdicts": verdicts,
                          "curve_invalid_runs": sum(
                              v in CURVE_INVALID_VERDICTS
                              for v in verdicts),
                          "runs": tuning_runs})
    finite = [row for row in table if _finite(row["val_accuracy_mean"])]
    base.update({"tuning_table": table})
    if not finite:
        base.update({"chosen_lr": None, "chosen_steps": None,
                     "unmeasured": True, "above_chance": False, "runs": [],
                     "reason": "all tuning runs non-finite"})
        _write_json(out_dir / "baselines_matched_shallow.json", base)
        return base
    pick = _curve_valid_selection_rows(table)
    eligible = pick["rows"]
    selection = _selection_reliability(
        eligible, grid=[(row["lr"], row["steps"]) for row in eligible],
        grid_keys=("lr", "steps"))
    selection["pool"] = pick["pool"]
    best = max(eligible, key=lambda row: row["val_accuracy_mean"])
    warn = _selection_warn(
        "matched-shallow lr x steps selection", selection)
    if warn:
        print(f"[baselines] {warn}: the matched-capacity control may "
              f"understate one-layer capacity")
    runs = []
    for seed in settings.seeds:
        _, params, metrics, curve, _ = _train_and_eval(
            settings, seed, best["lr"], eval_seed=TEST_SEED, gate="none",
            n_layers=1, d_model=wide_d, steps=best["steps"])
        n = int(metrics["n"])
        runs.append({
            "seed": seed, "accuracy": metrics["accuracy"],
            "accuracy_lo": metrics["accuracy_lo"],
            "accuracy_hi": metrics["accuracy_hi"], "n": n,
            "correct": int(round(metrics["accuracy"] * n)),
            "correct_bitmask": metrics["correct_bitmask"],
            "params": params, "curve_verdict": curve["curve_verdict"],
        })
        _write_result(out_dir / "baselines_runs.jsonl",
                      {"experiment": "baselines", "hypotheses": ["HB6",
                                                                 "HB7"],
                       "name": "matched_shallow_none", "n_layers": 1,
                       "d_model": wide_d, "lr": best["lr"],
                       "steps": int(best["steps"]),
                       "training_seed": seed, "eval_seed": TEST_SEED,
                       "accuracy": metrics["accuracy"],
                       "accuracy_lo": metrics["accuracy_lo"],
                       "accuracy_hi": metrics["accuracy_hi"], "n": n,
                       "params": params, "capacity_matched": matched,
                       **_training_record(settings, best["lr"],
                                          best["steps"]),
                       **_run_context(settings), **curve})
    valid = [r for r in runs
             if r.get("curve_verdict") not in CURVE_INVALID_VERDICTS]
    cluster = clustered_proportion_ci(
        [r["correct"] for r in valid], [r["n"] for r in valid])
    summary = _mean_std([r["accuracy"] for r in valid])
    above = bool(cluster["valid"] and math.isfinite(cluster["lo"])
                 and cluster["lo"] > float(ladder.get("chance") or 0.0))
    ceiling = float(ladder.get("ceiling", float("nan")))
    base.update({
        "chosen_lr": best["lr"], "chosen_steps": int(best["steps"]),
        "selection": selection,
        "accuracy_mean": summary["mean"],
        "accuracy_std": summary["std"],
        "accuracy_by_seed": {str(r["seed"]): r["accuracy"] for r in valid},
        "correct_bitmask_by_seed": {
            str(r["seed"]): r["correct_bitmask"] for r in valid},
        "eval_n_by_seed": {str(r["seed"]): int(r["n"]) for r in valid},
        "cluster_ci": cluster, "above_chance": above,
        "unmeasured": not bool(valid),
        "ceiling_gap": (ceiling - summary["mean"]
                        if math.isfinite(ceiling) else float("nan")),
        "runs": runs,
    })
    print(f"[baselines] matched-capacity shallow control (1 layer, "
          f"d_model={wide_d}, {wide_params} params vs target "
          f"{target_params} for depth {target_depth}, "
          f"capacity_matched={matched}; lr={best['lr']:g}, "
          f"steps={int(best['steps'])} on val seed {VAL_SEED}, test seed "
          f"{TEST_SEED}): accuracy="
          f"{summary['mean']:.4f} +- {summary['std']:.4f} "
          f"(seed-clustered [{cluster['lo']:.4f}, {cluster['hi']:.4f}], "
          f"chance={float(ladder.get('chance') or 0.0):.4f}, n="
          f"{cluster['n']} seeds/{cluster['n_examples']} examples); "
          f"{'above chance' if above else 'NOT above chance'}")
    _write_json(out_dir / "baselines_matched_shallow.json", base)
    return base


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


def _sanity_configs(settings: Settings) -> tuple:
    """Per-path HB5 configurations, including the configured model depth.

    The static matrix isolates gate/normalisation/LayerScale paths on a
    2-layer miniature so the pre-flight stays cheap.  A backward problem
    that only appears once activations traverse the configured depth would
    still pass that matrix while every grid run at ``settings.n_layers``
    fails, so the two architectures the grids actually depend on (the
    default Pre-LN arm and the E3 seednorm arm, both with the calibrated
    gate) are repeated at the real depth.
    """
    configs = list(_SANITY_CONFIGS)
    depth = int(settings.n_layers)
    if depth > 2:
        for norm in ("pre", "seednorm"):
            configs.append((
                f"gate=calib,norm={norm},n_layers={depth}",
                {"gate": "calib", "norm": norm, "n_layers": depth}))
    return tuple(configs)


def _overfit_sanity_by_config(settings: Settings) -> dict:
    """Tiny-batch memorisation check for every trained path (cached).

    One fixed batch from ``SANITY_SEED`` is pushed through the model of each
    configuration with dropout off and weight decay zeroed, so every custom
    kernel's backward path *and* every normalisation / LayerScale path the
    grids train is exercised - a passing ``gate=none`` check alone would not
    prove the calib/null gates or the seednorm/Post-LN/LayerScale paths
    backprop.  Beyond the cheap 2-layer matrix, ``_sanity_configs`` repeats
    the default and seednorm arms at ``settings.n_layers`` so depth-specific
    gradient/loss failures are caught before a grid starts.  Cached on
    ``Settings``: E1/E2/E3 in one ``--only all`` run share a single
    execution.
    """
    cache_key = _config_cache_key(settings)
    cached = getattr(settings, "_overfit_sanity_cache", None)
    if cached is not None and getattr(
            settings, "_overfit_sanity_cache_key", None) == cache_key:
        return cached
    # The sanity batch is trained on, so it must never alias a held-out
    # stream even though it is a single non-reported batch.
    check_train_seed(SANITY_SEED)
    gen = stream_generator(SANITY_SEED, settings.device)
    sidx, stgt, smeta = make_batch(settings.overfit_batch, settings.n_pairs,
                                   settings.vocab, settings.device, gen=gen)
    validate_batch(sidx, stgt, smeta, settings.vocab, diagnostics=False)
    by_config = {}
    # The miniature width is 48 whenever that is a valid head multiple, but
    # an arbitrary ``n_heads`` (e.g. 5 or 7) must not make the HB5 pre-flight
    # itself unbuildable: round up to the next multiple of the head count.
    heads = max(1, int(settings.n_heads))
    mini_d_model = 48
    if mini_d_model % heads:
        mini_d_model = heads * ((mini_d_model + heads - 1) // heads)
    for label, overrides in _sanity_configs(settings):
        seed_everything(SANITY_SEED, deterministic=settings.deterministic)
        tiny = build(_model_cfg(
            settings, **{"d_model": mini_d_model, "n_layers": 2, **overrides}),
            compile=False)
        row = overfit_sanity_check(
            tiny, sidx, stgt, smeta, lr=settings.overfit_lr,
            steps=settings.overfit_steps, seed=SANITY_SEED,
            vocab=settings.vocab, deterministic=settings.deterministic)
        row["model_cfg"] = asdict(tiny.cfg)
        row["params"] = int(sum(p.numel() for p in tiny.parameters()))
        row["stream_seed"] = SANITY_SEED
        row["n_pairs"] = int(settings.n_pairs)
        row["vocab"] = int(settings.vocab)
        row["device"] = settings.device
        row["deterministic"] = bool(settings.deterministic)
        by_config[label] = row
    settings._overfit_sanity_cache = by_config
    settings._overfit_sanity_cache_key = cache_key
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
        "stream_seed": sanity["stream_seed"],
        "n_pairs": sanity["n_pairs"],
        "vocab": sanity["vocab"],
        "device": sanity["device"],
        "deterministic": sanity["deterministic"],
        "by_config": {
            label: {
                "passed": row["passed"],
                "learned": row["learned"],
                "loss_first": row["loss_first"],
                "loss_last": row["loss_last"],
                "acc_first": row["acc_first"],
                "acc_last": row["acc_last"],
                "steps_to_target_loss": row["steps_to_target_loss"],
                "steps_to_target_acc": row["steps_to_target_acc"],
                "target_loss": row["target_loss"],
                "target_acc": row["target_acc"],
                "curve": row["curve"],
                "seeding": dict(row.get("seeding") or {}),
                "data_sha256": row["data_sha256"],
                "params": row["params"],
                "model_cfg": row["model_cfg"],
                "params_connected": (
                    f"{row['grad_flow']['params_with_grad']}/"
                    f"{row['grad_flow']['params_total']}"),
                "all_grads_finite": row["grad_flow"]["all_grads_finite"],
                "active_dropout": row["regularization"]["active_dropout"],
                "dropout_probs": row["regularization"]["dropout_probs"],
                "weight_decay": row["regularization"]["weight_decay"],
                "augmentation": row["regularization"]["augmentation"],
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
    _announce(("HB0", "HB1", "HB2", "HB3", "HB4", "HB5", "HB6", "HB7"))
    _write_manifest(settings, out_dir, "baselines")
    audit = _audit_data_splits(settings, out_dir, "baselines", enforce=True)
    repro_audit = _enforce_reproducibility(settings, out_dir, "baselines")
    record = {"experiment": "baselines",
              "hypotheses": ["HB1", "HB2", "HB3", "HB4", "HB5", "HB6",
                             "HB7"],
              **_run_context(settings)}

    balance = _answer_balance(settings)
    uni = balance["uniformity"]
    print(f"[baselines] answer-value balance over {balance['n_samples']} "
          f"samples from training streams {balance['train_seeds']}: "
          f"{balance['n_classes_present']} classes present, share "
          f"min={balance['min_class_share']:.5f} max={balance['max_class_share']:.5f} "
          f"(uniform={balance['expected_uniform_share']:.5f}); "
          f"uniformity chi2={uni['chi2']:.1f} (df={uni['df']}, "
          f"E/class={uni['expected_per_class']:.1f}) p={uni['p_value']:.4f} -> "
          f"{'uniform supported' if uni['uniform_supported'] else 'NOT uniform'}")
    if not uni["uniform_supported"]:
        print("[baselines] WARNING: the drawn answers are not uniform at the "
              "audit sample size: accuracy degrades toward a class-prior "
              "metric, so the F1/PR-AUC companions (not accuracy) carry the "
              "headline claims and the balance premise is flagged in "
              "baselines_verdict.json")
    _write_json(out_dir / "baselines_balance.json", balance)

    # The same accuracy premise has to hold on the streams where accuracy is
    # reported, not only on the training draws.  This audit reproduces each
    # consumer's exact geometry and stamps the result in the verdict so
    # E1/E2/E3 can warn even when run later from a stored verdict.
    eval_balance = _eval_answer_balance(settings)
    _write_json(out_dir / "baselines_eval_balance.json", eval_balance)
    for label, row in eval_balance["streams"].items():
        uni_row = row["uniformity"]
        print(f"[baselines] evaluation answer balance ({label} stream, "
              f"seed {row['stream_seed']}, {row['batch']}x{row['batches']}="
              f"{row['n']} samples): {row['n_classes_present']} classes "
              f"present, share min={row['min_class_share']:.5f} "
              f"max={row['max_class_share']:.5f} "
              f"(E/class={uni_row['expected_per_class']:.2f}); "
              f"uniformity p={uni_row['p_value']:.4f} -> "
              f"{'uniform supported' if uni_row['uniform_supported'] else 'NOT uniform'}")
    if not eval_balance["all_uniform"]:
        bad = [label for label, row in eval_balance["streams"].items()
               if not row["uniformity"]["uniform_supported"]]
        print("[baselines] WARNING: answer values are not uniform on held-out "
              f"stream(s) {', '.join(bad)} at the reported geometry: the "
              "accuracy headline and the accuracy-based contrasts are "
              "class-prior readings on those streams; read the F1/PR-AUC "
              "companions and treat the balance premise as failed in "
              "baselines_verdict.json")
    else:
        print("[baselines] evaluation answer balance: all held-out streams "
              "pass the gross-imbalance test at the reported geometry "
              "(finite-stream test; see baselines_eval_balance.json for "
              "expected samples/class)")
    e2_balance = eval_balance.get("e2_length_sweep") or {}
    if e2_balance.get("lengths"):
        cells = []
        for length in e2_balance["lengths"]:
            row = (e2_balance.get("streams") or {}).get(str(length)) or {}
            ok = (row.get("uniformity") or {}).get("uniform_supported")
            cells.append(f"L={length}:{'uniform' if ok else 'NOT uniform'}")
        print("[baselines] E2 length-sweep answer balance (TEST stream at "
              "each sweep length): " + ", ".join(cells))
        if e2_balance.get("endpoint_uniform") is False:
            bad_lengths = [
                str(length) for length in e2_balance["endpoint_lengths"]
                if not ((e2_balance.get("streams") or {})
                        .get(str(length)) or {})
                .get("uniformity", {}).get("uniform_supported")]
            print("[baselines] WARNING: E2 length-sweep endpoint balance "
                  f"failed at length(s) {', '.join(bad_lengths)}: the "
                  "length-generalisation accuracy read is a class-prior "
                  "reading on that endpoint")
    elif e2_balance.get("error"):
        print("[baselines] E2 length-sweep answer balance: not audited "
              f"({e2_balance['error']})")

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
    for label, overrides in _sanity_configs(settings):
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

    probe_steps_grid = tuple(settings.probe_steps_grid
                             or (settings.probe_steps,))
    best_lr, best_steps, table = tune_linear_probe(
        settings.n_pairs, settings.vocab, settings.device,
        lr_grid=settings.probe_lr_grid,
        seeds=settings.seeds[:max(1, len(settings.seeds))],
        steps=settings.probe_steps, steps_grid=probe_steps_grid,
        batch=settings.probe_batch,
        val_seed=VAL_SEED, val_batches=settings.probe_eval_batches,
        probe_batch=settings.probe_eval_batch,
        deterministic=settings.deterministic)
    probe_selection = _selection_reliability(
        table, grid=[(row.get("lr"), row.get("steps")) for row in table],
        grid_keys=("lr", "steps"))
    _write_json(out_dir / "baselines_probe_tuning.json",
                {"val_seed": VAL_SEED, "table": table, "chosen_lr": best_lr,
                 "chosen_steps": best_steps,
                 "lr_grid": list(settings.probe_lr_grid),
                 "steps_grid": list(probe_steps_grid),
                 "selection": probe_selection})
    probe_warn = _selection_warn(
        f"linear probe selection lr={best_lr:g}, steps={best_steps}",
        probe_selection)
    if probe_warn:
        print(f"[baselines] {probe_warn}: HB2's near-chance bound may be an "
              f"under-tuned strawman")
    n_diverged = int(sum(row.get("diverged_runs", 0) for row in table))
    print(f"[baselines] linear probe tuned on val seed {VAL_SEED} over "
          f"lr x steps grid {settings.probe_lr_grid} x {probe_steps_grid}: "
          f"lr={best_lr:g}, steps={best_steps}"
          + ("" if not n_diverged else
             f" ({n_diverged} diverged grid runs excluded from selection, "
             f"see baselines_probe_tuning.json)"))
    probe_accs = []
    probe_diverged = []
    for seed in settings.seeds:
        probe_curve = []
        probe_seeding = {}
        probe = train_linear_probe(settings.n_pairs, settings.vocab,
                                   settings.device, steps=best_steps,
                                   batch=settings.probe_batch, lr=best_lr,
                                   seed=seed, history=probe_curve,
                                   deterministic=settings.deterministic,
                                   seed_info=probe_seeding)
        acc = linear_probe_accuracy(
            probe, settings.n_pairs, settings.vocab, settings.device,
            batch=settings.probe_eval_batch,
            batches=settings.probe_eval_batches, seed=TEST_SEED)
        diverged = bool(any("nonfinite" in p or "nonfinite_grad" in p
                            for p in probe_curve))
        # A diverged run stopped at an arbitrary step, so its accuracy is
        # not a completed-run measurement: keep it in the raw record for
        # transparency but exclude it from the HB2 mean/CI, exactly like
        # ``tune_linear_probe`` excludes diverged grid runs from selection.
        if diverged:
            probe_diverged.append({"seed": seed, "accuracy": acc})
        else:
            probe_accs.append(acc)
        _write_result(out_dir / "baselines_runs.jsonl",
                      {**record, "name": "linear_bag_probe", "lr": best_lr,
                       "steps": best_steps,
                       "training_seed": seed, "eval_seed": TEST_SEED,
                       "accuracy": acc,
                       "diverged": diverged,
                       "train_loss_first": (
                           probe_curve[0]["loss"] if probe_curve
                           else float("nan")),
                       "train_loss_last": (
                           probe_curve[-1]["loss"] if probe_curve
                           else float("nan")),
                       "probe_curve": probe_curve,
                       "seeding": probe_seeding,
                       "n": settings.probe_eval_batch
                            * settings.probe_eval_batches})
    if probe_diverged:
        print(f"[baselines] WARNING: {len(probe_diverged)} final probe "
              f"run(s) diverged at lr={best_lr:g} and are excluded from "
              f"HB2: seeds "
              f"{[row['seed'] for row in probe_diverged]}")
    if not probe_accs:
        raise RuntimeError(
            f"all {len(settings.seeds)} final linear-probe runs diverged "
            f"at the selected lr={best_lr:g}: no completed test-seed "
            f"measurement exists, so HB2 cannot be certified (inspect "
            f"baselines_runs.jsonl and the probe training curves)")
    summary = _mean_std(probe_accs)
    chance = 1.0 / (settings.vocab - 2)
    probe_n_per_seed = settings.probe_eval_batch * settings.probe_eval_batches
    hb2 = probe_vs_chance_test(
        probe_accs, chance,
        seed=settings.seeds[0] if settings.seeds else 0,
        n_per_seed=probe_n_per_seed)
    n_seeds = len(settings.seeds)
    floor = int(settings.min_seeds_for_claim)
    print(f"[baselines] tuned linear probe (test seed {TEST_SEED}): "
          f"accuracy={summary['mean']:.4f} +- {summary['std']:.4f} "
          f"(chance={chance:.4f}, n_seeds={summary['n']}, eval n="
          f"{settings.probe_eval_batch * settings.probe_eval_batches} per "
          f"seed)")
    print(f"[baselines] HB2 near-chance test: accuracy - chance = "
          f"{hb2['mean']:+.4f} [{hb2['lo']:+.4f}, {hb2['hi']:+.4f}] "
          f"(seed-level [{hb2['lo_seed']:+.4f}, {hb2['hi_seed']:+.4f}]; "
          f"seed resampling + one shared Binomial scoring draw per replicate "
          f"on the common stream, n_per_seed={hb2['n_per_seed']}; one-sided "
          f"p(H0: margin >= +chance)={hb2['p_above_chance']:.3f}, "
          f"MDE80={hb2['mde80']:.4f}, sd_seed={hb2['sd_seed']:.4f}, "
          f"sd_scoring={hb2['sd_binomial']:.4f}, sd_per_seed="
          f"{hb2['sd_effective']:.4f}, se_mean={hb2['se_effective']:.4f})")
    if hb2["hb2_state"] == "certified":
        print("  state: CERTIFIED - the tuned probe stays confidently below "
              "the 2x-chance bound")
    elif hb2["hb2_state"] == "violated":
        print("  state: VIOLATED - the probe is confidently at/above 2x "
              "chance: answer information leaks into the bag-of-tokens "
              "features (or a shortcut exists), so the simple-baseline "
              "comparison cannot be credited before re-examining the "
              "pipeline")
    else:
        if math.isfinite(hb2["seeds_for_certification"]):
            need_s = (f"needs ~"
                      f"{math.ceil(hb2['seeds_for_certification'])} "
                      f"independent seeds at the observed seed spread plus "
                      f"the common-stream noise floor for 80% power at this "
                      f"margin")
        else:
            need_s = ("the observed margin cannot clear the irreducible "
                      "common-stream noise floor (or the point estimate is "
                      "not below the bound), so more seeds alone cannot "
                      "certify it; 'not certified' is an open question about "
                      "the probe, not evidence about the transformer")
        print("  state: INCONCLUSIVE - near-chance NOT certified (the CI "
              "straddles the +chance bound); this is not evidence of "
              f"leakage: {need_s}; or widen the probe evaluation stream "
              "(Settings.probe_eval_batch/probe_eval_batches) and re-run")
    if hb2["hb2_supported"] and int(hb2["n"]) < floor:
        print(f"  NOTE: HB2 is provisional: certified on {int(hb2['n'])} "
              f"completed seed(s) < the pre-registered claim floor {floor} "
              f"(Settings.min_seeds_for_claim)")

    # HB6/HB7 complexity ladder: the simplest learned model must be tried --
    # and tuned -- before the deep grids can claim that depth/gating is
    # needed, and a single shallow point cannot say *where* depth stops
    # mattering.  Runs before the negative control so a shallow-model
    # finding (few layers already saturate the task) is on record next to it.
    ladder = _run_depth_ladder(settings, out_dir, rule_stats)
    ladder["matched_shallow"] = _matched_shallow_control(
        settings, out_dir, ladder)
    shallow = ladder["depths"][ladder["shallowest"]]

    # HB4 negative control: same architecture and shared recipe as E1, but
    # the supervised answers are permuted across the batch during training.
    # Chance-level test accuracy is the expected outcome; above-chance means
    # answer information leaks into the pipeline.
    lr_shared = _resolve_lr(settings, out_dir, "baselines")
    ctrl_runs = []
    for seed in settings.seeds:
        _, _, metrics, curve, _ = _train_and_eval(
            settings, seed, lr_shared, eval_seed=TEST_SEED, gate="none",
            permute_labels=True)
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
                       "eval_seed": TEST_SEED,
                       "accuracy": metrics["accuracy"],
                       "accuracy_lo": metrics["accuracy_lo"],
                       "accuracy_hi": metrics["accuracy_hi"], "n": n,
                       **_training_record(settings, lr_shared),
                       **curve})
    margins = [r["accuracy"] - chance for r in ctrl_runs]
    hb4 = paired_bootstrap_ci(margins)
    pooled = wilson_ci(sum(r["correct"] for r in ctrl_runs),
                       sum(r["n"] for r in ctrl_runs))
    ctrl_cluster = clustered_proportion_ci(
        [r["correct"] for r in ctrl_runs], [r["n"] for r in ctrl_runs])
    ctrl_summary = _mean_std([r["accuracy"] for r in ctrl_runs])
    # The control's per-seed accuracies are measured on the same sealed
    # stream, so the seed-only bootstrap upper bound can be too narrow.
    # The decision uses the seed-clustered (shared-stream) upper bound;
    # the historical seed-only interval remains a companion.
    hb4_measure = {
        "mean": ctrl_cluster["mean"] - chance,
        "lo": ctrl_cluster["lo"] - chance,
        "hi": ctrl_cluster["hi"] - chance,
        "n": ctrl_cluster["n"], "n_examples": ctrl_cluster["n_examples"],
        "se": ctrl_cluster["se"],
    }
    hb4_supported = bool(
        ctrl_cluster["valid"] and ctrl_cluster["n"] > 0
        and hb4_measure["hi"] < settings.negative_control_tol)
    hb4_violated = bool(
        ctrl_cluster["valid"] and ctrl_cluster["n"] > 0
        and hb4_measure["lo"] >= settings.negative_control_tol)
    if not ctrl_cluster["valid"] or ctrl_cluster["n"] == 0:
        hb4_state = "untestable"
    elif hb4_supported:
        hb4_state = "supported"
    elif hb4_violated:
        hb4_state = "violated"
    else:
        hb4_state = "inconclusive"
    hb4_claim_eligible = bool(hb4["n"] >= floor)
    hb4_note = ("" if not hb4_supported or hb4_claim_eligible else
                f" [provisional: {hb4['n']} seed(s) < claim floor {floor}]")
    hb4_status = {
        "supported": "SUPPORTED (no label leakage detected)",
        "violated": ("VIOLATED - the control interval is entirely above "
                     "tolerance: answer information leaks into the pipeline"),
        "inconclusive": ("INCONCLUSIVE - the interval straddles tolerance "
                         "(not evidence of leakage; raise the seed count or "
                         "widen the control stream)"),
        "untestable": "untestable",
    }[hb4_state]
    print(f"[baselines] HB4 negative control (labels permuted in training, "
          f"shared recipe lr={lr_shared:g}, test seed {TEST_SEED}): "
          f"accuracy={ctrl_summary['mean']:.4f} +- {ctrl_summary['std']:.4f} "
          f"over {ctrl_summary['n']} seeds; accuracy - chance = "
          f"{hb4_measure['mean']:+.4f} "
          f"[{hb4_measure['lo']:+.4f}, {hb4_measure['hi']:+.4f}] "
          f"(seed-clustered, n={ctrl_cluster['n']} seeds/"
          f"{ctrl_cluster['n_examples']} examples; seed-only companion "
          f"{hb4['mean']:+.4f} [{hb4['lo']:+.4f}, {hb4['hi']:+.4f}]; "
          f"pooled Wilson companion "
          f"[{pooled['lo']:.4f}, {pooled['hi']:.4f}], "
          f"tol={settings.negative_control_tol:g}); "
          f"{hb4_status}{hb4_note}")
    _write_json(out_dir / "baselines_negative_control.json", {
        "lr": lr_shared, "chance": chance,
        "tolerance": settings.negative_control_tol,
        "accuracy_mean": ctrl_summary["mean"],
        "accuracy_std": ctrl_summary["std"],
        "margin": hb4_measure, "margin_seed_only": hb4,
        "cluster_ci": ctrl_cluster, "pooled_wilson": pooled,
        "runs": ctrl_runs, "hb4_supported": hb4_supported,
        "hb4_violated": hb4_violated, "hb4_state": hb4_state,
    })

    hb6_n_seeds = len(shallow.get("curve_valid_seeds") or [])
    hb7_n_seeds = min(
        (len(ladder["depths"][d].get("curve_valid_seeds") or [])
         for d in ladder["ladder"]), default=0)
    verdict = {
        "protocol_sha256": _protocol_sha256(_run_context(settings)),
        "git_sha": _run_context(settings)["git_sha"],
        "git_dirty": _run_context(settings)["git_dirty"],
        "source_tree_sha256": _run_context(settings)["source_tree_sha256"],
        "setup_sha256": _run_context(settings)["setup_sha256"],
        "run_id": _run_context(settings)["run_id"],
        "train_seed": settings.seeds[0] if settings.seeds else None,
        "test_seed": TEST_SEED,
        "val_seed": VAL_SEED,
        # Seal semantics, written into the verdict so a reader of the JSON
        # alone cannot mistake "sealed" for "evaluated once at the very
        # end": TEST_SEED is read by the final reported numbers *and* by
        # the pre-registered, fixed-in-code validity gates (HB1 rule
        # ceiling, HB4 permuted-label control, HB0/HB6/HB7 competence and
        # saturation marks).  No model/hyperparameter/recipe choice ever
        # reads it - all selection is VAL_SEED-only - and the CI loop's
        # re-measurement of the same fixed rules after a code change is
        # not re-selection (rules and grid are pre-registered, the gate is
        # pass/fail), so the stream cannot become an implicit validation
        # split.
        "sealing_note": (
            "sealed = no model/hyperparameter/recipe choice may read the "
            "stream; the fixed pre-registered gate rules and the final "
            "report may. Re-running `--only baselines` re-measures those "
            "fixed rules on TEST (verdicts are recomputed, not reselected)."
        ),
        "chance": chance,
        "repro_audit_passed": bool(repro_audit["passed"]),
        "repro_audit": {
            "probe_seed": repro_audit["probe_seed"],
            "other_seed": repro_audit["other_seed"],
            "steps": repro_audit["steps"],
            "checks": {key: bool(repro_audit[key])
                       for key, _ in _REPRO_AUDIT_KEYS},
        },
        "min_seeds_for_claim": floor,
        "n_seeds": n_seeds,
        "hb1_supported": bool(hb1_supported),
        "hb2_supported": bool(hb2["hb2_supported"]),
        "hb2_state": hb2["hb2_state"],
        "hb3_supported": bool(audit.get("hb3_supported")),
        "hb4_supported": hb4_supported,
        "hb4_violated": hb4_violated,
        "hb4_state": hb4_state,
        "hb5_supported": bool(hb5_supported),
        "hb6_supported": bool(shallow["hb6_supported"]),
        "hb6_supported_ci": bool(shallow["hb6_supported_ci"]),
        "hb6_competent": bool(shallow["above_chance"]),
        "hb7_depth_required": ladder["hb7_depth_required"],
        "ladder_competent": bool(ladder["competence"]["ladder_competent"]),
        "saturation_depth": ladder["saturation_depth"],
        "saturation_depth_ci": ladder["saturation_depth_ci"],
        # Pre-registered seed floor (Settings.min_seeds_for_claim) applied
        # to the seed-level marks: a supported HB2/HB4/HB6/HB7 resting on
        # fewer seeds is provisional and must be displayed/stamped as such.
        "seed_floor": {
            "min_seeds_for_claim": floor,
            "n_seeds": n_seeds,
            "hb2_n_seeds": int(hb2["n"]),
            "hb2_claim_eligible": bool(hb2["n"] >= floor),
            "hb4_n_seeds": int(hb4["n"]),
            "hb4_claim_eligible": hb4_claim_eligible,
            "hb6_n_seeds": hb6_n_seeds,
            "hb6_claim_eligible": bool(hb6_n_seeds >= floor),
            "hb7_n_seeds": hb7_n_seeds,
            "hb7_claim_eligible": bool(hb7_n_seeds >= floor),
        },
        "answer_balance": balance,
        # Held-out-stream balance: the accuracy premise applies where the
        # numbers are reported, not only where the model trains.  Stamped
        # separately so a reader cannot confuse it with the training audit.
        "eval_answer_balance": eval_balance,
        "depth_ladder": {
            "ladder": ladder["ladder"],
            "val_seed": VAL_SEED,
            "test_seed": TEST_SEED,
            "chance": chance,
            "sesoi_accuracy": ladder["sesoi_accuracy"],
            "saturation_rule": ladder["saturation_rule"],
            "saturation_depth_ci": ladder["saturation_depth_ci"],
            "competence": ladder["competence"],
            "by_depth": {
                str(d): {
                    "lr": payload["chosen_lr"],
                    "steps": payload["chosen_steps"],
                    "tuning_table": payload["tuning_table"],
                    "selection": payload["selection"],
                    "accuracy_mean": payload["accuracy_mean"],
                    "accuracy_std": payload["accuracy_std"],
                    "accuracy_by_seed": payload["accuracy_by_seed"],
                    "pooled_wilson": payload["pooled_wilson"],
                    "cluster_ci": payload["cluster_ci"],
                    "above_chance": payload["above_chance"],
                    "unmeasured": payload["unmeasured"],
                    "sesoi_accuracy": payload["sesoi_accuracy"],
                    "ceiling_gap": payload["ceiling_gap"],
                    "ceiling_gap_lower": payload["ceiling_gap_lower"],
                    "reaches_ceiling": payload["reaches_ceiling"],
                    "reaches_ceiling_ci": payload["reaches_ceiling_ci"],
                    "hb6_supported": payload["hb6_supported"],
                    "hb6_supported_ci": payload["hb6_supported_ci"],
                }
                for d, payload in ladder["depths"].items()
            },
            "adjacent_contrasts": ladder["adjacent_contrasts"],
        },
        "probe_accuracy_mean": summary["mean"],
        "probe_accuracy_std": summary["std"],
        "probe_tuning": {
            "lr_grid": list(settings.probe_lr_grid),
            "steps_grid": list(probe_steps_grid),
            "chosen_steps": best_steps,
            "diverged_runs": n_diverged,
            "final_diverged_seeds": [row["seed"] for row in probe_diverged],
            "final_diverged_accuracy": {
                str(row["seed"]): row["accuracy"] for row in probe_diverged},
            "selection": probe_selection,
            "by_lr": table,
        },
        "matched_shallow": ladder.get("matched_shallow"),
        "shallow_none": {
            "n_layers": ladder["shallowest"], "lr": shallow["chosen_lr"],
            "steps": shallow.get("chosen_steps"),
            "tuning_table": shallow["tuning_table"],
            "selection": shallow["selection"],
            "accuracy_mean": shallow["accuracy_mean"],
            "accuracy_std": shallow["accuracy_std"],
            "accuracy_by_seed": shallow["accuracy_by_seed"],
            "pooled_wilson": shallow["pooled_wilson"],
            "cluster_ci": shallow["cluster_ci"],
            "chance": chance,
            "above_chance": shallow["above_chance"],
            "unmeasured": shallow["unmeasured"],
            "ceiling": shallow["ceiling"],
            "sesoi_accuracy": shallow["sesoi_accuracy"],
            "ceiling_gap": shallow["ceiling_gap"],
            "ceiling_gap_lower": shallow["ceiling_gap_lower"],
            "reaches_ceiling": shallow["reaches_ceiling"],
            "reaches_ceiling_ci": shallow["reaches_ceiling_ci"],
            "hb6_supported": shallow["hb6_supported"],
            "hb6_supported_ci": shallow["hb6_supported_ci"],
        },
        "hb2_accuracy_minus_chance": {
            "mean": hb2["mean"], "lo": hb2["lo"], "hi": hb2["hi"],
            "p_value": hb2["p_value"],
            "p_above_chance": hb2["p_above_chance"],
            "mde80": hb2["mde80"], "sd": hb2["sd"],
            "n": hb2["n"],
            "state": hb2["hb2_state"],
            "seeds_for_certification": hb2["seeds_for_certification"],
            "seed_level": hb2["seed_level"],
            "sd_seed": hb2["sd_seed"],
            "sd_binomial": hb2["sd_binomial"],
            "sd_effective": hb2["sd_effective"],
            "se_effective": hb2["se_effective"],
            "n_per_seed": hb2["n_per_seed"],
            "measurement_noise_accounted": hb2["measurement_noise_accounted"],
        },
        "hb4_negative_control": {
            "lr": lr_shared, "tolerance": settings.negative_control_tol,
            "accuracy_mean": ctrl_summary["mean"],
            "accuracy_std": ctrl_summary["std"],
            "margin": hb4_measure, "margin_seed_only": hb4,
            "pooled_wilson": pooled, "cluster_ci": ctrl_cluster,
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
                    "learned": row["learned"],
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
    hb2_txt = {
        "certified": "ok (certified)",
        "violated": "VIOLATED - probe at/above 2x chance",
        "inconclusive": "INCONCLUSIVE (near-chance not certified)",
        "untestable": "untestable",
    }[verdict["hb2_state"]]
    if verdict["hb7_depth_required"] is None:
        hb7_txt = ("not determinable (no competent rung)"
                   if not verdict["ladder_competent"] else "n/a")
    elif verdict["hb7_depth_required"]:
        hb7_txt = "required"
    else:
        hb7_txt = (f"NOT required (saturated within SESOI "
                   f"{ladder['sesoi_accuracy']:g} at depth "
                   f"{verdict['saturation_depth']})")
    provisional = [k for k in ("hb2", "hb4", "hb6")
                   if verdict[f"{k}_supported"] is True
                   and not verdict["seed_floor"][f"{k}_claim_eligible"]]
    if (verdict["hb7_depth_required"] is True
            and not verdict["seed_floor"]["hb7_claim_eligible"]):
        provisional.append("hb7")
    provisional_txt = (
        f"; PROVISIONAL: {' '.join(provisional).upper()} rest on fewer than "
        f"{floor} seeds - raise Settings.seeds or downgrade the claim"
        if provisional else "")
    hb4_verdict_txt = {
        "supported": "ok", "violated": "VIOLATED",
        "inconclusive": "INCONCLUSIVE", "untestable": "untestable",
    }.get(verdict.get("hb4_state"),
          "ok" if verdict["hb4_supported"] else "FAILED")
    print(f"[baselines] verdict: "
          f"HB1={'ok' if verdict['hb1_supported'] else 'FAILED'}"
          f" HB2={hb2_txt}"
          f" HB3={'ok' if verdict['hb3_supported'] else 'FAILED'}"
          f" HB4={hb4_verdict_txt}"
          f" HB5={'ok' if verdict['hb5_supported'] else 'FAILED'}"
          f" HB6={'ok' if verdict['hb6_supported'] else 'FLAG'}"
          f" HB7={hb7_txt}"
          f"{provisional_txt} "
          f"(stored in baselines_verdict.json; a failed baseline check "
          f"invalidates the corresponding claims above it, HB6/HB7 only flag "
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

# E1 active-control (placebo) arm for the concentration signal (H1f): the
# identical calib architecture, parameter count, computation and recipe,
# but the entropy driving the gate is a fixed position permutation of the
# real entropy (Cfg.calib_shuffle_ent).  The treatment therefore has the
# same marginal distribution as the real signal and differs from full
# calib *only* in the information linking a queried row to its own
# attention concentration - which the leave-one-out ablations (removing the
# term and its parameter) cannot isolate.
_E1_CALIB_PLACEBO = (
    ("calib_shuf_ent", {"calib_use_len": True, "calib_use_ent": True,
                        "calib_shuffle_ent": True}),
)


# Learning-rate sensitivity points for the primary gate contrast, expressed
# as multiplicative factors of the frozen shared lr.  The controlled audit
# proves both sides of every contrast use the *same* recipe, but a single
# point cannot show that the effect is not an artefact of that particular
# recipe.  These points keep the recipe identical across arms *within* each
# point and vary only the shared lr between points, so a sign that survives
# the sweep is not an optimisation-winner artefact; the rows are exploratory
# companions and never enter the FDR decision family.
_E1_LR_SENSITIVITY_FACTORS = (0.5, 2.0)


def _e1_per_variant_tuning(settings: Settings, out_dir: pathlib.Path,
                           reference_lr: float,
                           reference_runs: dict | None = None) -> dict:
    """Give every E1 arm its own validation-selected recipe.

    The pre-registered E1 family freezes *one* recipe (tuned on ``none``) for
    all four arms.  That is the correct controlled contrast, but it is not
    the only fairness question a baseline is entitled to: if ``query`` or
    ``null`` happens to prefer a different learning rate, the frozen recipe
    can make a baseline look weaker than it is, and the same is true for
    calib if its ``calib_tau_init`` was never tuned.  This companion
    therefore selects, per arm and on ``VAL_SEED`` only:

    * an lr from the shared ``lr_grid`` plus the frozen reference point; and
    * for calib, a ``calib_tau_init`` from ``calib_tau_grid`` at the chosen
      lr (sequential selection; every run is logged).

    The selected recipes are then re-trained on the full training-seed set
    and scored on the sealed test stream, and ``calib - baseline`` is paired
    by seed.  It is deliberately **not** routed through ``ContrastLog``: the
    arms differ in their optimiser/temperature, so the controlled-variable
    audit must fail and the row is an exploratory benchmark.  The
    pre-registered same-recipe family remains the primary conclusion; this
    payload answers the separate question "does calib still win when the
    baselines also get their own reasonable recipe?".

    The reference recipe is reused when it wins an arm's selection, so a
    baseline that is already optimal under the frozen point costs no extra
    final runs.  Every tuning run (validation only) and every final run is
    written to disk, so the selection is reproducible rather than a printed
    argmax.
    """
    if not settings.per_variant_tuning:
        return {"available": False,
                "reason": "disabled by Settings.per_variant_tuning"}
    reference_lr = float(reference_lr)
    if not math.isfinite(reference_lr) or reference_lr <= 0.0:
        return {"available": False, "reason": "non-positive reference lr"}
    tune_seeds = list(settings.seeds[:max(
        1, min(int(settings.per_variant_tune_seeds), len(settings.seeds)))])
    claim_floor = min(int(settings.min_seeds_for_claim), len(settings.seeds))
    if len(tune_seeds) < claim_floor:
        print(f"[e1] WARNING: per-variant tuning selects on "
              f"{len(tune_seeds)} seed(s), below the pre-registered claim "
              f"floor {claim_floor}; the tuned recipes are noisier than the "
              f"controlled ones and the two accounts must still be read "
              f"separately")
    candidate_lrs = sorted({reference_lr,
                            *(float(v) for v in settings.lr_grid)})
    default_tau = float(Cfg().calib_tau_init)
    candidate_taus = sorted({default_tau,
                             *(float(v) for v in settings.calib_tau_grid)})
    by_arm, tuning_rows = {}, []

    def _tune_run(gate, seed, lr, tau, phase):
        overrides = ({"calib_tau_init": tau} if gate == "calib" else {})
        _, params, metrics, curve, model_cfg = _train_and_eval(
            settings, seed, lr, eval_seed=VAL_SEED, selection=True,
            gate=gate, **overrides)
        rec = {
            "experiment": "e1_per_variant_tuning",
            "hypotheses": ["H1a"],
            "phase": phase, "gate": gate, "seed": int(seed), "lr": float(lr),
            "calib_tau_init": (float(tau) if gate == "calib" else None),
            "eval_seed": VAL_SEED, "params": int(params),
            "val_accuracy": float(metrics["accuracy"]),
            "model_cfg": model_cfg,
            "curve": curve,
            **_training_record(settings, lr),
            **_run_context(settings),
            # ``_curve_record`` holds the actual seed_everything result for
            # this run; promote it over the static policy copy stamped by
            # the run context (the nested curve stays for the full record).
            "seeding": curve["seeding"],
        }
        _write_result(out_dir / "e1_per_variant_tuning_runs.jsonl", rec)
        tuning_rows.append(rec)
        return rec

    def _pick(rows, key, label, preferred=None):
        pick = _curve_valid_selection_rows(rows)
        eligible = pick["rows"]
        if not eligible:
            return None, {
                "label": label, "pool": pick["pool"],
                "n_finite": pick["n_finite"], "n_invalid": pick["n_invalid"],
                "reason": "no finite validation row",
            }
        # Break accuracy ties toward the frozen reference point: an
        # arbitrary argmax on equal validation accuracy would select a
        # different recipe for no measured reason and forgo reference-run
        # reuse.  The controlled point is the conservative choice.
        best = max(eligible, key=lambda row: (
            row["val_accuracy_mean"],
            preferred is not None and row[key] == preferred))
        selection = _selection_reliability(
            eligible, grid=[row[key] for row in eligible])
        selection["pool"] = pick["pool"]
        selection["n_finite"] = pick["n_finite"]
        selection["n_invalid"] = pick["n_invalid"]
        selection["excluded_curve_invalid_rows"] = pick["n_excluded"]
        return best, selection

    for gate in GATES:
        lr_rows = []
        for lr in candidate_lrs:
            runs = [_tune_run(gate, seed, lr, default_tau, "lr")
                    for seed in tune_seeds]
            accs = [r["val_accuracy"] for r in runs]
            mean = sum(accs) / len(accs) if accs else float("nan")
            var = (sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)
                   if len(accs) > 1 else 0.0)
            verdicts = [r["curve"]["curve_verdict"] for r in runs]
            lr_rows.append({
                "lr": float(lr), "calib_tau_init": default_tau,
                "val_accuracy_mean": mean,
                "val_accuracy_std": math.sqrt(var) if math.isfinite(var)
                else float("nan"),
                "n_seeds": len(accs), "curve_verdicts": verdicts,
                "curve_invalid_runs": sum(
                    v in CURVE_INVALID_VERDICTS for v in verdicts),
                "runs": runs,
            })
        best_lr_row, lr_selection = _pick(
            lr_rows, "lr", f"{gate}:lr", preferred=reference_lr)
        fallback = None
        if best_lr_row is None:
            final_lr = reference_lr
            fallback = "no finite lr row; reused frozen reference lr"
        else:
            final_lr = float(best_lr_row["lr"])
        if gate != "calib":
            final_tau = None
            tau_selection = {"applicable": False}
            validation = best_lr_row
        else:
            tau_rows = []
            for tau in candidate_taus:
                if (tau == default_tau and best_lr_row is not None
                        and float(best_lr_row["lr"]) == final_lr):
                    runs = list(best_lr_row["runs"])
                else:
                    runs = [_tune_run(gate, seed, final_lr, tau, "tau")
                            for seed in tune_seeds]
                accs = [r["val_accuracy"] for r in runs]
                mean = sum(accs) / len(accs) if accs else float("nan")
                var = (sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)
                       if len(accs) > 1 else 0.0)
                verdicts = [r["curve"]["curve_verdict"] for r in runs]
                tau_rows.append({
                    "lr": final_lr, "calib_tau_init": float(tau),
                    "val_accuracy_mean": mean,
                    "val_accuracy_std": math.sqrt(var) if math.isfinite(var)
                    else float("nan"),
                    "n_seeds": len(accs), "curve_verdicts": verdicts,
                    "curve_invalid_runs": sum(
                        v in CURVE_INVALID_VERDICTS for v in verdicts),
                    "runs": runs,
                })
            best_tau_row, tau_selection = _pick(
                tau_rows, "calib_tau_init", "calib:tau",
                preferred=default_tau)
            if best_tau_row is None:
                final_tau = default_tau
                fallback = ((fallback + "; ") if fallback else "") + \
                    "no finite tau row; reused default calib_tau_init"
                validation = best_lr_row
            else:
                final_tau = float(best_tau_row["calib_tau_init"])
                validation = best_tau_row
            tau_selection["rows"] = [
                {k: v for k, v in row.items() if k != "runs"}
                for row in tau_rows]
        reuse = (final_lr == reference_lr
                 and (gate != "calib" or final_tau == default_tau))
        reused_reference = bool(reuse and reference_runs
                                and reference_runs.get(gate))
        if reused_reference:
            final_runs = list(reference_runs[gate])
        else:
            final_runs = []
            for seed in settings.seeds:
                extra = {
                    "per_variant_tuned": True,
                    "tuned_lr": final_lr,
                    "tuned_calib_tau_init": final_tau,
                }
                overrides = ({"calib_tau_init": final_tau}
                             if gate == "calib" else {})
                if gate == "calib":
                    extra["calib_variant"] = "full"
                rec, _ = _e1_run(settings, seed, final_lr, gate, extra,
                                 **overrides)
                _write_result(out_dir / "e1_per_variant_tuned_runs.jsonl",
                              rec)
                final_runs.append(rec)
        by_arm[gate] = {
            "recipe": {"lr": final_lr, "calib_tau_init": final_tau},
            "reused_reference_runs": reused_reference,
            "fallback": fallback,
            "validation": (None if validation is None else {
                "accuracy_mean": validation["val_accuracy_mean"],
                "accuracy_std": validation["val_accuracy_std"],
                "n_seeds": validation["n_seeds"],
                "curve_invalid_runs": validation["curve_invalid_runs"],
            }),
            "lr_selection": lr_selection,
            "tau_selection": tau_selection,
            "lr_grid": [{
                k: v for k, v in row.items() if k != "runs"
            } for row in lr_rows],
            "test": {
                # The paired contrasts below drop curve-invalid runs, so the
                # per-arm levels printed next to them must use the same seed
                # set; the all-run mean remains as a separate companion.
                "n_runs": len(_curve_valid_runs(final_runs)),
                "n_runs_all": len(final_runs),
                **{key: _mean_std([r.get(key)
                                   for r in _curve_valid_runs(final_runs)])
                   for key in ("accuracy", "abs_gate_vs_entropy",
                               "abs_gate_vs_maxprob", "gate_margin",
                               "false_conc_rate", "null_mass_mean")},
                "all": {key: _mean_std([r.get(key) for r in final_runs])
                        for key in ("accuracy", "abs_gate_vs_entropy",
                                    "abs_gate_vs_maxprob", "gate_margin",
                                    "false_conc_rate", "null_mass_mean")},
            },
            "curve_invalid_runs": sum(
                r.get("curve_verdict") in CURVE_INVALID_VERDICTS
                for r in final_runs),
            "runs": final_runs,
        }

    accuracy_sesoi = float(HYPOTHESES["H1a"]["sesoi"]["accuracy"])
    mechanism_sesoi = float(HYPOTHESES["H1a"]["sesoi"]["abs_gate_vs_entropy"])
    maxprob_sesoi = float(
        HYPOTHESES["H1a"]["sesoi"]["abs_gate_vs_maxprob"])
    margin_sesoi = float(HYPOTHESES["H1b"]["sesoi"]["gate_margin"])
    contrasts = {}

    def _state(ci):
        if ci.get("unmeasured"):
            return "unmeasured"
        if ci["complexity_justified"]:
            return "above tuned baseline by SESOI"
        if ci["below_reference"]:
            return "tuned baseline better"
        if ci.get("indistinguishable"):
            return "indistinguishable"
        # A wide CI that contains 0 but reaches outside the SESOI is
        # inconclusive, not evidence that the arms are indistinguishable.
        return "inconclusive"

    for baseline in ("none", "query", "null"):
        if baseline == "calib" or baseline not in by_arm:
            continue
        ref = (f"tuned {baseline} (lr="
               f"{by_arm[baseline]['recipe']['lr']:g})")
        row = {"accuracy": _paired_benchmark(
            by_arm["calib"]["runs"], by_arm[baseline]["runs"], "accuracy",
            accuracy_sesoi, ref,
            comparison="per_variant_tuned_benchmark",
            treatment=("each arm's own val-selected lr and calib_tau_init "
                       "for calib"))}
        row["accuracy"]["state"] = _state(row["accuracy"])
        if baseline == "query":
            for metric, sesoi in (
                ("abs_gate_vs_entropy", mechanism_sesoi),
                ("abs_gate_vs_maxprob", maxprob_sesoi),
                ("gate_margin", margin_sesoi),
            ):
                ci = _paired_benchmark(
                    by_arm["calib"]["runs"], by_arm[baseline]["runs"], metric,
                    sesoi, ref, comparison="per_variant_tuned_benchmark",
                    treatment=("each arm's own val-selected lr and "
                               "calib_tau_init for calib"))
                ci["state"] = _state(ci)
                row[metric] = ci
        contrasts[f"calib - {baseline}"] = row

    tuned_states = {name: row["accuracy"]["state"]
                    for name, row in contrasts.items()}
    unmeasured = [name for name, state in tuned_states.items()
                  if state == "unmeasured"]
    above = [name for name, state in tuned_states.items()
             if state == "above tuned baseline by SESOI"]
    below = [name for name, state in tuned_states.items()
             if state == "tuned baseline better"]
    if unmeasured:
        readout = ("not determinable for "
                   + ", ".join(unmeasured)
                   + ": at least one side has no curve-valid paired seed")
    elif below:
        readout = ("NOT robust to per-variant tuning: "
                   + ", ".join(below)
                   + " is confidently better than calib")
    elif len(above) == len(tuned_states) and tuned_states:
        readout = ("robust: calib is above every tuned baseline by the "
                   "pre-registered accuracy SESOI")
    else:
        readout = ("not established: calib is not above every tuned "
                   "baseline by the accuracy SESOI")
    calib_advantage = {
        "by_baseline": tuned_states,
        "all_baselines_beaten_by_sesoi": bool(
            tuned_states and len(above) == len(tuned_states)),
        "any_tuned_baseline_better": bool(below),
        "unmeasured": unmeasured,
        "readout": readout,
    }

    controlled = {}
    if reference_runs:
        for baseline in ("none", "query", "null"):
            if baseline not in reference_runs:
                continue
            controlled[f"calib - {baseline}"] = {
                metric: {
                    "mean": ci.get("mean"), "lo": ci.get("lo"),
                    "hi": ci.get("hi"), "n": ci.get("n"),
                }
                for metric, ci in (
                    ("accuracy", _paired_diff(
                        reference_runs["calib"], reference_runs[baseline],
                        "accuracy")),
                    ("abs_gate_vs_entropy", _paired_diff(
                        reference_runs["calib"], reference_runs[baseline],
                        "abs_gate_vs_entropy")),
                    ("abs_gate_vs_maxprob", _paired_diff(
                        reference_runs["calib"], reference_runs[baseline],
                        "abs_gate_vs_maxprob")),
                    ("gate_margin", _paired_diff(
                        reference_runs["calib"], reference_runs[baseline],
                        "gate_margin")),
                )
            }

    print(f"\n[e1] per-variant tuning (separate fairness accounting, "
          f"exploratory and outside the FDR family): each arm's own recipe "
          f"is selected on validation seed {VAL_SEED} over "
          f"lr={candidate_lrs} (calib also over calib_tau_init="
          f"{candidate_taus}); selection seeds="
          f"{tune_seeds}.  The controlled rows above freeze one recipe for "
          f"all arms - this table asks whether calib still wins when the "
          f"none/query/null baselines also get their own best shot:")
    def _sel_flag(selection):
        if not selection or selection.get("applicable") is False:
            return "n/a"
        if selection.get("reliable") is True:
            return "ok"
        if selection.get("reliable") is False:
            reasons = []
            if selection.get("within_noise"):
                reasons.append("noise")
            if selection.get("best_at_grid_boundary"):
                reasons.append("boundary")
            if selection.get("curve_invalid_best"):
                reasons.append("curve")
            return "/".join(reasons) or "unreliable"
        return "n/a"

    print(f"{'arm':>6} {'lr':>10} {'tau':>7} {'val_acc':>17} "
          f"{'val_n':>6} {'curve_invalid':>13} {'test_acc':>17} "
          f"{'reused':>7} {'lr_sel':>12} {'tau_sel':>12}")
    for gate in GATES:
        row = by_arm[gate]
        recipe = row["recipe"]
        val = row["validation"] or {}
        test = row["test"]
        tau = ("n/a" if recipe["calib_tau_init"] is None
               else f"{recipe['calib_tau_init']:g}")
        val_s = (f"{val.get('accuracy_mean', float('nan')):.4f}±"
                 f"{val.get('accuracy_std', float('nan')):.4f}"
                 if _finite(val.get("accuracy_mean")) else "n/a")
        test_acc = test.get("accuracy", {})
        test_s = (f"{test_acc.get('mean', float('nan')):.4f}±"
                  f"{test_acc.get('std', float('nan')):.4f}"
                  if _finite(test_acc.get("mean")) else "n/a")
        print(f"{gate:>6} {recipe['lr']:>10g} {tau:>7} {val_s:>17} "
              f"{int(val.get('n_seeds', 0)):>6} "
              f"{row['curve_invalid_runs']:>13} {test_s:>17} "
              f"{'yes' if row['reused_reference_runs'] else 'no':>7} "
              f"{_sel_flag(row.get('lr_selection')):>12} "
              f"{_sel_flag(row.get('tau_selection')):>12}")
        if row.get("fallback"):
            print(f"       fallback: {row['fallback']}")
    print("  (lr_sel/tau_sel = selection reliability audit: 'noise' = "
          "winner inside the seed spread of the runner-up, 'boundary' = "
          "winner at the end of the grid, 'curve' = winning runs had an "
          "invalid training curve; a tuned baseline with an unreliable "
          "winner is under-tuned in the conservative direction for the "
          "gate claim)")
    print(f"\n[e1] two accounting conventions side by side (accuracy and "
          f"mechanism readouts; controlled = same frozen recipe, tuned = "
          f"each arm's own validation-selected recipe).  The pre-registered "
          f"FDR family is the controlled column; the tuned column is the "
          f"fairness robustness check and must not be averaged with it:")
    print(f"{'contrast':>18} {'metric':>22} {'controlled':>28} "
          f"{'tuned':>28} {'tuned state':>34}")
    for name, row in contrasts.items():
        for metric in ("accuracy", "abs_gate_vs_entropy",
                       "abs_gate_vs_maxprob", "gate_margin"):
            if metric not in row:
                continue
            tuned = row[metric]
            ctl = (controlled.get(name) or {}).get(metric) or {}
            ctl_s = (f"{ctl.get('mean', float('nan')):+.4f} "
                     f"[{ctl.get('lo', float('nan')):+.4f},"
                     f"{ctl.get('hi', float('nan')):+.4f}]"
                     if _finite(ctl.get("mean")) else "n/a")
            print(f"{name:>18} {metric:>22} {ctl_s:>28} "
                  f"{_fmt_ci(tuned):>28} {tuned.get('state', ''):>34}")
    print(f"[e1] per-variant tuned accuracy conclusion: {readout}")
    payload = {
        "available": True,
        "selection_seed": VAL_SEED,
        "selection_rule": "max mean validation accuracy among curve-valid rows",
        "tune_seeds": tune_seeds,
        "candidate_lrs": candidate_lrs,
        "calib_tau_grid": candidate_taus,
        "by_arm": {gate: {
            k: v for k, v in row.items() if k != "runs"
        } for gate, row in by_arm.items()},
        "contrasts": contrasts,
        "controlled_contrasts": controlled,
        "calib_advantage": calib_advantage,
        "comparison": "per_variant_tuned_benchmark",
        "note": ("exploratory fairness companion, not a controlled mechanism "
                 "contrast and not part of the pre-registered FDR family: "
                 "each arm's lr (and calib's calib_tau_init) is selected on "
                 "the validation seed, so the controlled-variable audit does "
                 "not apply; read it next to the same-recipe controlled "
                 "family, never instead of it"),
    }
    _write_json(out_dir / "e1_per_variant_tuning.json", payload)
    return payload


def _e1_lr_sensitivity(settings: Settings, out_dir: pathlib.Path,
                       reference_lr: float,
                       reference_runs: dict | None = None) -> dict:
    """Re-run the primary gate arms at alternative shared learning rates.

    For each factor in ``_E1_LR_SENSITIVITY_FACTORS`` the same seeds and
    streams train the control baselines ``none`` / ``query`` / ``null`` and
    the ``calib`` arm under one shared lr, and every A-vs-B contrast is
    re-paired within that point.  ``null`` is a full control arm, not a
    companion: its frozen-lr point must be re-measured like the other two,
    otherwise "calib - null" could be recipe-specific without the
    sensitivity table saying so.  The comparison is
    therefore still controlled at each lr (only the gate factor differs),
    while the *between-point* shift shows whether the calib-over-control
    effect is robust to the frozen-recipe choice or exists only at the
    selected point.  The report is explicitly exploratory: the
    pre-registered E1 family and its FDR decision are untouched.

    When ``reference_runs`` is supplied (the already-trained pre-registered
    ``by_gate`` runs), the frozen-lr point is included by reuse rather than
    retraining, so the stability read covers the decision point itself at no
    extra cost.  A stability statement that omitted the selected point could
    otherwise be stable at every alternative while reversing at the point
    that actually decides the claim.
    """
    reference_lr = float(reference_lr)
    if not math.isfinite(reference_lr) or reference_lr <= 0.0:
        return {"available": False, "reason": "non-positive reference lr"}
    alternative_lrs = sorted({
        reference_lr * float(factor)
        for factor in _E1_LR_SENSITIVITY_FACTORS
        if math.isfinite(reference_lr * float(factor))
        and reference_lr * float(factor) > 0.0})
    lrs = sorted({reference_lr, *alternative_lrs})
    by_cell = {}
    if reference_runs:
        for gate in ("none", "query", "calib", "null"):
            reused = reference_runs.get(gate)
            if reused:
                by_cell[(reference_lr, gate)] = list(reused)
    for alt_lr in lrs:
        for gate in ("none", "query", "calib", "null"):
            if (alt_lr, gate) in by_cell:
                continue
            for seed in settings.seeds:
                extra = {
                    "lr_sensitivity": True,
                    "reference_lr": reference_lr,
                    "sensitivity_factor": alt_lr / reference_lr,
                }
                if gate == "calib":
                    extra["calib_variant"] = "full"
                rec, _ = _e1_run(settings, seed, alt_lr, gate, extra)
                _write_result(out_dir / "e1_lr_sensitivity_runs.jsonl", rec)
                by_cell.setdefault((alt_lr, gate), []).append(rec)

    print(f"\n[e1] learning-rate sensitivity (exploratory, outside the FDR "
          f"family): the frozen reference lr={reference_lr:g} point reuses "
          f"the pre-registered runs, and all four arms (including the null "
          f"control baseline) are re-trained at "
          f"{', '.join(f'{v:g}' for v in alternative_lrs)}; at each lr both "
          f"sides of a contrast still share the identical recipe, steps and "
          f"streams.  A sign that flips or a CI that only excludes 0 at the "
          f"selected point means the effect is recipe-specific, not a "
          f"mechanism conclusion:")
    print(f"{'contrast':>24} {'metric':>24} {'lr':>10} "
          f"{'paired_diff':>28} {'control':>9}")
    comparisons = (
        ("calib - query", "calib", "query", "accuracy"),
        ("calib - query", "calib", "query", "abs_gate_vs_entropy"),
        ("calib - query", "calib", "query", "abs_gate_vs_maxprob"),
        ("calib - query", "calib", "query", "gate_margin"),
        ("calib - none", "calib", "none", "accuracy"),
        # null is the third control baseline: re-measure the frozen-recipe
        # contrast at every shared lr instead of only in the per-variant
        # tuned companion.
        ("calib - null", "calib", "null", "accuracy"),
        ("null - none", "null", "none", "accuracy"),
    )
    rows = []
    for rule, a_gate, b_gate, metric in comparisons:
        stability = []
        for alt_lr in lrs:
            runs_a = by_cell.get((alt_lr, a_gate)) or []
            runs_b = by_cell.get((alt_lr, b_gate)) or []
            ci = _paired_diff(runs_a, runs_b, metric)
            audit = ci.get("controlled") or {}
            supported = bool(
                _finite(ci.get("lo")) and _finite(ci.get("hi"))
                and (ci["lo"] > 0.0 or ci["hi"] < 0.0))
            measured = bool(int(ci.get("n") or 0) > 0
                            and _finite(ci.get("mean")))
            row = {
                "rule": rule, "metric": metric, "lr": float(alt_lr),
                "reference_lr": reference_lr,
                "factor": alt_lr / reference_lr,
                "is_reference_point": bool(alt_lr == reference_lr),
                "mean": ci.get("mean"), "lo": ci.get("lo"),
                "hi": ci.get("hi"), "p_value": ci.get("p_value"),
                "n": ci.get("n"), "seeds_used": ci.get("seeds_used"),
                "seeds_dropped": ci.get("seeds_dropped"),
                "curve_invalid_seeds": ci.get("curve_invalid_seeds"),
                "controlled": audit.get("controlled"),
                "control_mismatches": audit.get("mismatches"),
                "capacity": audit.get("capacity"),
                "measured": measured,
                "ci_excludes_zero": supported,
            }
            rows.append(row)
            if measured:
                stability.append(float(ci["mean"]))
            print(f"{rule:>24} {metric:>24} {alt_lr:>10g} "
                  f"{_fmt_ci(ci):>28} "
                  f"{'ok' if audit.get('controlled') else 'FAILED':>9}")
        if stability:
            has_pos = any(v > 0.0 for v in stability)
            has_neg = any(v < 0.0 for v in stability)
            has_zero = any(v == 0.0 for v in stability)
            sign_flip = has_pos and has_neg
            same_sign = (not sign_flip) and (has_pos or has_neg) \
                and not has_zero
            measured_rows = [
                row for row in rows
                if row["rule"] == rule and row["metric"] == metric
                and row.get("measured")]
            all_exclude = all(
                row["ci_excludes_zero"] for row in measured_rows)
            any_exclude = any(
                row["ci_excludes_zero"] for row in measured_rows)
            if sign_flip:
                verdict = ("SIGN FLIP across learning rates: the effect is "
                           "recipe-specific")
            elif has_zero and (has_pos or has_neg):
                verdict = ("point estimate is exactly zero at some lrs: the "
                           "effect is not stable across the recipe")
            elif not same_sign:
                verdict = ("point estimate is zero at every lr: the design "
                           "resolves no directional effect")
            elif all_exclude:
                verdict = "stable: same-signed CI excluding 0 at every lr"
            elif not any_exclude:
                verdict = ("same-signed point estimates but no lr resolves "
                           "the effect")
            else:
                verdict = ("same-signed across lrs, but CI excludes 0 at "
                           "only some points")
            print(f"      -> {rule} ({metric}) stability: {verdict} "
                  f"[means {', '.join(f'{v:+.4f}' for v in stability)}]")
            for row in rows:
                if row["rule"] == rule and row["metric"] == metric:
                    row["stability"] = verdict
                    row["sign_stable"] = bool(same_sign)
                    row["ci_excludes_zero_all_lrs"] = bool(all_exclude)
    payload = {
        "reference_lr": reference_lr,
        "factors": list(_E1_LR_SENSITIVITY_FACTORS),
        "alternative_lrs": alternative_lrs,
        "lrs": lrs,
        "reference_point_reused": bool(reference_runs),
        "comparison": "learning_rate_sensitivity",
        "note": ("exploratory robustness companion; each lr point is a "
                 "controlled within-point pair, but the family decision "
                 "remains the pre-registered one at the frozen lr"),
        "rows": rows,
    }
    _write_json(out_dir / "e1_lr_sensitivity.json", payload)
    return payload


def _e1_run(settings: Settings, seed: int, lr: float, gate: str,
            record_extra: dict, **model_overrides):
    """Train + score one E1 condition and build its full run record.

    Every condition goes through the identical entry point (frozen recipe,
    fixed test/stats streams) so the only difference between conditions is
    ``model_overrides``; ``record_extra`` carries the ablation label that
    the controlled-variable audit compares, and the record stamps the
    *resolved* ``model_cfg`` the model was actually built from so the audit
    can check the trained configuration instead of the label alone.  The
    evaluation-stream seeds are stamped too (from the same locals used for
    the calls), so the audit verifies both sides read the same streams.
    Returns ``(record, errors)``.
    """
    eval_seed, stats_seed = TEST_SEED, STATS_SEED
    model, params, metrics, curve, model_cfg = _train_and_eval(
        settings, seed, lr, eval_seed=eval_seed, gate=gate, **model_overrides)
    # One diagnostic pass feeds both the layer statistics and the error
    # mining.
    stats, errors = collect_stats_and_errors(
        model, settings.n_pairs, settings.vocab, settings.device,
        batch=settings.stats_batch, batches=settings.stats_batches,
        seed=stats_seed, max_cases=settings.max_error_cases)
    corr = gate_stat_correlations(stats)
    fc_sweep = false_concentration_sweep(stats)
    # The sweep already contains the pre-registered 0.5 cut.
    fc = (fc_sweep[0.5] if 0.5 in fc_sweep
          else false_concentration_analysis(stats))
    # Same readout with the layer axis kept aligned (each layer's own
    # max-prob against that layer's own gate/null mass).  The aggregate
    # ``fc`` above remains the pre-registered H1b/H1c contrast; this
    # companion checks that the "close on false concentration" effect is
    # not an artifact of mixing a max-over-layers confidence with a
    # mean-over-layers gate.
    fc_layers = false_concentration_by_layer(stats)
    ed = error_detection_aucs(stats)
    # Finite-stats-stream uncertainty on gate_on_correct - gate_on_false_conc
    # (the independent two-sample bootstrap over the disjoint row groups).
    # The seed-level gate_margin above is a different uncertainty: it asks
    # whether the effect survives training-seed variance, not whether the
    # finite diagnostic stream resolves the levels within one run.
    fc_gate_boot = (false_concentration_gate_test(stats)
                    if gate in ("query", "calib") else None)
    rec = {
        "experiment": "e1",
        "hypotheses": ["H1a", "H1b", "H1c", "H1d", "H1e", "H1f",
                        "H1g"],
        "gate": gate, "seed": seed, "lr": lr,
        "eval_seed": eval_seed, "stats_seed": stats_seed,
        "params": params, "model_cfg": model_cfg,
        **metrics, **corr, **fc,
        # ed's "n" is the stats-stream sample count; it must not overwrite
        # metrics' "n" (the eval stream the accuracy/Wilson CI come from).
        # The stats count stays recorded as err_n_samples below.
        **{k: v for k, v in ed.items() if k != "n"},
        **{f"err_{k}": v for k, v in errors["summary"].items()},
        **_training_record(settings, lr),
        **_run_context(settings),
        **curve, **record_extra,
    }
    if _finite(rec.get("gate_vs_entropy")):
        rec["abs_gate_vs_entropy"] = abs(rec["gate_vs_entropy"])
    if _finite(rec.get("gate_vs_maxprob")):
        rec["abs_gate_vs_maxprob"] = abs(rec["gate_vs_maxprob"])
    # H1g reads the correlation-level claim per arm with separate metric
    # keys, so a calib one-sample row and a query equivalence row cannot
    # inherit each other's direction/bound through the shared display key.
    if gate == "calib":
        if _finite(rec.get("abs_gate_vs_entropy")):
            rec["calib_abs_gate_vs_entropy"] = rec["abs_gate_vs_entropy"]
        if _finite(rec.get("abs_gate_vs_maxprob")):
            rec["calib_abs_gate_vs_maxprob"] = rec["abs_gate_vs_maxprob"]
    elif gate == "query":
        if _finite(rec.get("abs_gate_vs_entropy")):
            rec["query_abs_gate_vs_entropy"] = rec["abs_gate_vs_entropy"]
        if _finite(rec.get("abs_gate_vs_maxprob")):
            rec["query_abs_gate_vs_maxprob"] = rec["abs_gate_vs_maxprob"]
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
    rec["false_conc_by_layer"] = fc_layers
    # Scalar layer-profile summaries for the aggregate table / JSON.  The
    # list itself stays in the record so a layer-level sign flip or a
    # single-layer association is visible instead of being averaged away.
    _layer_rates = [row["false_conc_rate"] for row in fc_layers
                    if _finite(row.get("false_conc_rate"))]
    if _layer_rates:
        rec["false_conc_layer_rate_mean"] = (
            sum(_layer_rates) / len(_layer_rates))
    if gate in ("query", "calib"):
        _margins = [row["gate_margin"] for row in fc_layers
                    if _finite(row.get("gate_margin"))]
        if _margins:
            rec["false_conc_layer_margin_mean"] = (
                sum(_margins) / len(_margins))
            rec["false_conc_layer_margin_min"] = min(_margins)
            rec["false_conc_layer_margin_positive_layers"] = sum(
                1 for value in _margins if value > 0.0)
            rec["false_conc_layer_n"] = len(_margins)
            rec["false_conc_layer_sign_flip"] = float(
                any(value < 0.0 for value in _margins))
    if gate == "null":
        _null_margins = [row["null_mass_error_margin"] for row in fc_layers
                         if _finite(row.get("null_mass_error_margin"))]
        if _null_margins:
            rec["null_mass_layer_margin_mean"] = (
                sum(_null_margins) / len(_null_margins))
            rec["null_mass_layer_margin_min"] = min(_null_margins)
            rec["null_mass_layer_margin_positive_layers"] = sum(
                1 for value in _null_margins if value > 0.0)
            rec["null_mass_layer_n"] = len(_null_margins)
            rec["null_mass_layer_sign_flip"] = float(
                any(value < 0.0 for value in _null_margins))
    if fc_gate_boot is not None:
        rec["false_conc_gate_boot"] = fc_gate_boot
    if gate == "null":
        rec["null_mass_error_margin"] = _null_mass_margin(stats)
        # Within-stream uncertainty companions to the seed-level H1c
        # contrast: all errors vs all correct, and the false-concentration
        # subset (the same high-max-prob cut used for the gate margin).
        rec["null_mass_all_boot"] = null_mass_gate_test(stats)
        rec["null_mass_false_conc_boot"] = null_mass_gate_test(
            stats, 0.5)
        # Report the null slot's absolute occupancy too, not only the
        # error-minus-correct margin: a slot that is uniformly open (or
        # uniformly closed) has a near-zero margin but no suppression
        # interpretation, so the level is needed next to the contrast.
        nm = stats.get("null_mass")
        if nm is not None:
            mass = nm.mean(dim=1)
            corr = stats["correct"].bool()
            rec["null_mass_mean"] = float(mass.mean())
            if bool((~corr).any()):
                rec["null_mass_error"] = float(mass[~corr].mean())
            if bool(corr.any()):
                rec["null_mass_correct"] = float(mass[corr].mean())
            # False-concentration subset: the H1c margin on the same
            # high-maxprob rows used by H1b, so the null arm's suppression
            # claim is testable where the gate claim is tested instead of
            # only on the all-rows average (which mixes confident and
            # unconfident errors).
            high = stats["maxprob"].max(dim=1)[0] > 0.5
            fc = high & ~corr
            cc = high & corr
            if bool(fc.any()) and bool(cc.any()):
                rec["null_mass_false_conc_error"] = float(mass[fc].mean())
                rec["null_mass_false_conc_correct"] = float(mass[cc].mean())
                rec["null_mass_false_conc_margin"] = float(
                    mass[fc].mean() - mass[cc].mean())
    return rec, errors


def run_e1(settings: Settings, out_dir: pathlib.Path) -> None:
    _announce(("H1a", "H1b", "H1c", "H1d", "H1e", "H1f", "H1g"))
    # HB1/HB3/HB4/HB5 hard gate: the baselines verdict must exist, be
    # current for this exact protocol and pass, before any grid trains
    # (their pre-registered semantics invalidate everything above them;
    # the marks are stamped into every record via _run_context).
    verdict = _enforce_baseline_verdict(settings, out_dir, "e1")
    _write_manifest(settings, out_dir, "e1")
    _audit_data_splits(settings, out_dir, "e1", enforce=True)
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
                              {"experiment": "e1", **_stamp(settings),
                               "gate": gate, "seed": seed,
                               "lr": lr, **case})
            _write_result(out_dir / "e1_runs.jsonl", rec)
            runs.append(rec)
    ablation_runs = {}
    for label, overrides in _E1_CALIB_ABLATION + _E1_CALIB_PLACEBO:
        for seed in settings.seeds:
            rec, errors = _e1_run(settings, seed, lr, "calib",
                                  {"calib_variant": label}, **overrides)
            for case in errors["cases"]:
                _write_result(out_dir / "e1_error_cases.jsonl",
                              {"experiment": "e1", **_stamp(settings),
                               "gate": "calib", "ablation": label,
                               "seed": seed, "lr": lr, **case})
            _write_result(out_dir / "e1_runs.jsonl", rec)
            ablation_runs.setdefault(label, []).append(rec)
    by_gate = {g: [r for r in runs if r["gate"] == g] for g in GATES}
    table = {g: _aggregate_runs(rs) for g, rs in by_gate.items()}
    ablation_coverage = _report_ablation_coverage(
        out_dir, "e1",
        runs + [r for rs in ablation_runs.values() for r in rs],
        cells=_CALIB_FACTORIAL_CELLS + (_CALIB_PLACEBO_CELL,),
        planned_seeds=settings.seeds,
        design=("calib_use_len x calib_use_ent (2x2) + shuffled-entropy "
                "active control"))
    design_coverage = _report_design_coverage(
        out_dir, "e1", runs, "gate",
        lambda r: r.get("gate"), tuple(GATES),
        planned_seeds=settings.seeds)

    print(f"\n[e1] test seed {TEST_SEED}, stats seed {STATS_SEED}, "
          f"lr={lr:g}, steps={settings.steps}")
    header = f"{'gate':>6} {'params':>8} {'accuracy':>16} {'bal_acc':>16} " \
             f"{'f1_macro':>16} {'top5':>16} {'abs_r_ent':>16} " \
             f"{'pr_auc_gate':>16}"
    print(header)
    print("  (bal_acc = macro recall, equal weight per target class: a gap "
          "from accuracy means the reading depends on the answer-class "
          "distribution)")
    for gate in GATES:
        t = table[gate]
        def cell(key):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return f"{m['mean']:.4f}±{m['std']:.4f}"
        print(f"{gate:>6} {int(by_gate[gate][0]['params']):>8} "
              f"{cell('accuracy'):>16} {cell('recall_macro'):>16} "
              f"{cell('f1_macro'):>16} {cell('top5_accuracy'):>16} "
              f"{cell('abs_gate_vs_entropy'):>16} "
              f"{cell('prauc_gate_to_error'):>16}")

    # Metric regime per gate: error detection is a rare-class problem once
    # the model converges, so the measured imbalance is reported next to
    # each score's ranking power.  PR-AUC is read against its chance level
    # (lift = PR-AUC / prevalence; a random ranker has lift 1.0) and the
    # operational reading (errors caught in the 10% most doubtful answers)
    # sits next to ROC-AUC, which stays optimistic under imbalance.
    print("\n[e1] error-detection regime (positive = wrong answer): lift = "
          "PR-AUC / prevalence (1.0 = random ranking); recall@10% = share of "
          "errors captured in the 10% lowest-gate answers; mean±std over "
          "seeds (these are H1e's headline metrics, whose decision rows carry "
          "the FDR-controlled CI below):")
    print(f"{'gate':>6} {'primary':>8} {'err_rate':>17} {'imbalance':>17} "
          f"{'regime':>10} {'prauc':>17} {'lift':>17} {'recall@10%':>17} "
          f"{'rocauc':>17}")
    for gate in GATES:
        t = table[gate]
        primary = str(by_gate[gate][0].get("primary_metric", ""))
        if primary.startswith("prauc_"):
            score = primary.split("_")[1]
        else:
            def _finite_prauc(name):
                m = t.get(f"prauc_{name}_to_error")
                return bool(m and math.isfinite(m["mean"]))
            score = next((name for name in ("gate", "maxprob", "entropy")
                          if _finite_prauc(name)), "maxprob")

        def ecell(key):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return f"{m['mean']:.4f}±{m['std']:.4f}"

        err_rate = t.get("error_rate", {}).get("mean", float("nan"))
        ratio = t.get("imbalance_ratio", {}).get("mean", float("nan"))
        # Degenerate streams get their own labels (mirroring
        # train.error_detection_aucs' metric_regime): an all-wrong stream
        # has no negatives, so the ranking metrics are undefined and the
        # regime must not read as a usable "balanced".
        if not math.isfinite(err_rate) or err_rate <= 0.0:
            regime = "no_errors"
        elif err_rate >= 1.0:
            regime = "all_errors"
        elif not math.isfinite(ratio):
            regime = "n/a"
        elif ratio >= 1000.0:
            regime = "extreme"
        elif ratio >= 10.0:
            regime = "high"
        else:
            regime = "balanced"
        print(f"{gate:>6} {score:>8} {ecell('error_rate'):>17} "
              f"{ecell('imbalance_ratio'):>17} {regime:>10} "
              f"{ecell(f'prauc_{score}_to_error'):>17} "
              f"{ecell(f'prauc_lift_{score}_to_error'):>17} "
              f"{ecell(f'recall_at_10pct_{score}_to_error'):>17} "
              f"{ecell(f'rocauc_{score}_to_error'):>17}")

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

    placebo_runs = ablation_runs[_CALIB_PLACEBO_CELL]
    placebo_table = _aggregate_runs(placebo_runs)
    print(f"\n[e1] H1f active control (placebo): the shuffled-entropy arm "
          f"keeps the identical architecture, parameter count, computation "
          f"and recipe; only the link between a queried row and its own "
          f"attention concentration is severed.  'query' is printed as the "
          f"capacity reference: if the placebo tracks full calib, the "
          f"effect is capacity, not attention-distribution information; if "
          f"it collapses toward 'query', the entropy content is doing the "
          f"work.")
    print(f"{'arm':>16} {'params':>8} {'accuracy':>17} {'abs_r_ent':>17} "
          f"{'gate_margin':>17}")
    for label, cell_table in (("full", abl_table["full"]),
                              (_CALIB_PLACEBO_CELL, placebo_table),
                              ("query", table["query"])):
        def pcell(key):
            m = cell_table.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return f"{m['mean']:.4f}±{m['std']:.4f}"

        n_params = int(abl_index[label][0]["params"]) if label in abl_index \
            else int(by_gate["query"][0]["params"])
        print(f"{label:>16} {n_params:>8} "
              f"{pcell('accuracy'):>17} {pcell('abs_gate_vs_entropy'):>17} "
              f"{pcell('gate_margin'):>17}")

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
    # The ceiling gap is an effect size like any other: it is printed with
    # the seed-level spread of the accuracy it is derived from (the rule
    # ceiling itself is a fixed parameter-free reference with its own Wilson
    # interval above), so a gap smaller than the seed noise cannot be read
    # as a real distance from the ceiling.
    print(f"{'gate':>6} {'accuracy':>17} {'ceiling_gap':>17}")
    for gate in GATES:
        agg = table[gate].get("accuracy", {})
        acc = agg.get("mean", float("nan"))
        acc_std = agg.get("std", float("nan"))
        gap = (ceiling["accuracy"] - acc) if math.isfinite(acc) else float("nan")
        acc_s = (f"{acc:.4f}±{acc_std:.4f}" if _finite(acc)
                 else "n/a")
        gap_s = (f"{gap:.4f}±{acc_std:.4f}" if _finite(gap) else "n/a")
        print(f"{gate:>6} {acc_s:>17} {gap_s:>17}")
    probe = verdict.get("probe_accuracy_mean", float("nan"))
    probe_std = verdict.get("probe_accuracy_std", float("nan"))
    probe_s = (f"{probe:.4f}±{probe_std:.4f}" if _finite(probe)
               else f"{probe:.4f}")
    floor = verdict.get("seed_floor") or {}

    def _mark(key, claim_key=None):
        value = verdict.get(key)
        if value is None:
            return "n/a"
        if not value:
            return "FAILED"
        if claim_key and not floor.get(f"{claim_key}_claim_eligible", True):
            return "ok (provisional: below seed floor)"
        return "ok"

    hb4_state = verdict.get("hb4_state")
    if hb4_state == "inconclusive":
        hb4_mark = "INCONCLUSIVE"
    elif hb4_state == "violated":
        hb4_mark = "VIOLATED"
    elif hb4_state == "untestable":
        hb4_mark = "untestable"
    else:
        hb4_mark = _mark("hb4_supported", "hb4")
    hb2_state = verdict.get("hb2_state")
    hb2_mark = {
        "violated": "VIOLATED",
        "inconclusive": "INCONCLUSIVE",
        "untestable": "untestable",
    }.get(hb2_state, _mark("hb2_supported", "hb2"))
    hb6_mark = ("n/a" if verdict.get("hb6_supported") is None
                else _mark("hb6_supported", "hb6")
                if verdict.get("hb6_supported") else "FLAG")
    marks = "/".join(
        [_mark("hb1_supported"),
         hb2_mark,
         _mark("hb3_supported"),
         hb4_mark,
         _mark("hb5_supported"),
         hb6_mark])
    hb4_note = ""
    if hb4_state == "inconclusive":
        hb4_note = ("; HB4 INCONCLUSIVE - the negative-control interval "
                    "straddles tolerance, so neither leakage nor its absence "
                    "is certified (not a failure; raise the seed count or "
                    "widen the control stream)")
    elif hb4_state == "violated":
        hb4_note = ("; HB4 VIOLATED - the permutation control is above "
                    "tolerance: answer information leaks into the pipeline")
    if hb2_state in (None, "certified"):
        hb2_note = ""
    elif hb2_state == "violated":
        hb2_note = (f"; HB2 VIOLATED - the probe is at/above 2x chance: the "
                    f"simple-baseline comparison must be re-examined before "
                    f"crediting any gate variant")
    else:
        hb2_note = (f"; HB2 state={hb2_state} - near-chance not certified "
                    f"(underpower/wide CI, not evidence of leakage)")
    print(f"[e1] simple-baseline check (from --only baselines, enforced "
          f"before training): HB1/HB2/HB3/HB4/HB5/HB6={marks}, tuned linear "
          f"probe test accuracy={probe_s} (mean±std over probe seeds) vs "
          f"chance "
          f"{verdict.get('chance', float('nan')):.4f}{hb2_note}{hb4_note} - the "
          f"transformer is only credited for what it does beyond this tuned "
          f"baseline and the rule ceiling (HB4 = no label leakage, HB5 = "
          f"loop can memorise a tiny batch, HB6 = the shallowest ladder rung "
          f"is below the ceiling)")
    shallow = verdict.get("shallow_none") or {}
    shallow_by_seed = {}
    for key, value in (shallow.get("accuracy_by_seed") or {}).items():
        try:
            shallow_by_seed[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    complexity_benchmark = None
    matched_benchmark = None
    if shallow_by_seed:
        shallow_depth = int(shallow.get("n_layers", 1))
        reference_competent = bool(
            shallow.get("above_chance", True)
            and not shallow.get("unmeasured", False))
        hb6_prov = ("" if not verdict.get("hb6_supported")
                    or floor.get("hb6_claim_eligible", True)
                    else " [provisional: below the pre-registered seed floor]")
        if not reference_competent:
            hb6_prov += (" [HB6 not read as depth evidence: the rung is not "
                         "above chance, so this step count did not learn the "
                         "task]")
        if (shallow.get("reaches_ceiling_ci")
                and not shallow.get("reaches_ceiling")):
            hb6_prov += (" [companion: the 95% CI touches the ceiling "
                         "(absence of evidence); saturation requires the "
                         "seed-clustered lower bound within SESOI]")
        print(f"[e1] complexity control (HB6): tuned {shallow_depth}-layer "
              f"standard transformer "
              f"(lr={shallow.get('lr', float('nan')):g}, "
              f"steps={shallow.get('steps')}) test "
              f"accuracy={shallow.get('accuracy_mean', float('nan')):.4f}"
              f"±{shallow.get('accuracy_std', float('nan')):.4f}, "
              f"rule ceiling={shallow.get('ceiling', float('nan')):.4f}, "
              f"gap={shallow.get('ceiling_gap', float('nan')):.4f} "
              f"(lower-bound gap="
              f"{shallow.get('ceiling_gap_lower', float('nan')):.4f}, "
              f"SESOI={shallow.get('sesoi_accuracy', float('nan')):g})"
              f"{hb6_prov}")
        # Paired complexity reference: same seeds, same test stream,
        # each side optimised with its own val-tuned learning rate and
        # step budget (that is the point of a tuned-baseline comparison),
        # so the
        # strict controlled-recipe audit does not apply and this line
        # is deliberately kept outside the FDR family.  The pre-registered
        # seed floor still applies to how the line is *labelled*: below
        # ``min_seeds_for_claim`` paired seeds a CI is a provisional
        # observation, never a "depth is required" result.
        deep_runs = [r for r in _curve_valid_runs(by_gate["none"])
                     if r["seed"] in shallow_by_seed]
        diffs = [r["accuracy"] - shallow_by_seed[r["seed"]]
                 for r in deep_runs]
        excluded_deep = sorted(
            [r["seed"] for r in by_gate["none"]
             if r.get("curve_verdict") in CURVE_INVALID_VERDICTS],
            key=_seed_order)
        floor_note = ""
        if excluded_deep:
            floor_note += ("  [curve-invalid deep run(s) excluded: "
                           f"{', '.join(str(s) for s in excluded_deep)}]")
        if len(diffs) >= 2:
            ci = paired_bootstrap_ci(diffs)
            deep_layers = int(
                (by_gate["none"][0].get("model_cfg") or {})
                .get("n_layers", settings.n_layers))
            claim_floor_ok = len(diffs) >= settings.min_seeds_for_claim
            if not claim_floor_ok:
                floor_note = (f"  [provisional: n={len(diffs)} paired seeds "
                              f"< the pre-registered claim floor "
                              f"{settings.min_seeds_for_claim}: too few "
                              f"runs for a depth claim]" + floor_note)
            if not reference_competent:
                readout = ("the reference rung is not above chance, so "
                           "this comparison cannot support a depth claim")
            elif not claim_floor_ok:
                readout = ("a gap excluding 0 is a provisional observation, "
                           "not a depth claim")
            else:
                readout = ("a supported gap means the depth of the E1/E3 "
                           "grids is measurably required, a gap including 0 "
                           "means a gate gain must be beyond what one layer "
                           "already reaches")
            print(f"      paired deep(none, "
                  f"{deep_layers} layers) - "
                  f"shallow({shallow_depth} layer) "
                  f"accuracy: {_fmt_ci(ci)} (each side val-tuned; "
                  f"{readout})"
                  f"{floor_note}")
        # Per-arm complexity benchmark: the depth-only reference above says
        # whether 8 layers beat 1 layer, but it does not say whether any
        # *mechanism* arm earns its capacity over the tuned simple model.
        # Each gate is compared on the same seeds and sealed test stream,
        # with both sides val-tuned.  This is deliberately not a controlled
        # mechanism contrast (architecture and lr are the treatment), so it
        # stays outside the FDR family; the pre-registered SESOI and seed
        # floor still decide how the row may be labelled.
        benchmark_sesoi = float(
            (HYPOTHESES["HB6"].get("sesoi") or {}).get("accuracy", 0.01))
        shallow_runs = [{"seed": seed, "accuracy": shallow_by_seed[seed]}
                        for seed in sorted(shallow_by_seed, key=_seed_order)]
        reference_name = (f"tuned {shallow_depth}-layer standard "
                          f"transformer (lr={shallow.get('lr', float('nan')):g}, "
                          f"steps={shallow.get('steps')})")
        print(f"\n[e1] complexity benchmark vs the tuned shallow rung "
              f"(paired over seeds; SESOI={benchmark_sesoi:g} accuracy; "
              f"each arm carries its own val-tuned lr and step budget, so "
              f"this is a benchmark, not a controlled mechanism contrast):")
        if not reference_competent:
            print(f"      reference rung is not above chance: rows are "
                  f"diagnostics only and cannot justify complexity")
        print(f"{'arm':>6} {'arm_acc':>17} {'shallow_acc':>17} "
              f"{'paired_diff':>28} {'state':>18}")
        benchmark_by_arm = {}
        for gate in GATES:
            bench = _paired_benchmark(by_gate[gate], shallow_runs, "accuracy",
                                      benchmark_sesoi, reference_name)
            arm_mean = (bench.get("arms") or {}).get("a", {}).get(
                "mean", float("nan"))
            ref_mean = (bench.get("arms") or {}).get("b", {}).get(
                "mean", float("nan"))
            prov_parts = []
            if int(bench.get("n") or 0) < settings.min_seeds_for_claim:
                prov_parts.append(
                    f"[provisional: n={int(bench.get('n') or 0)} "
                    f"< claim floor {settings.min_seeds_for_claim}]")
            if not reference_competent:
                prov_parts.append(
                    "[reference not above chance: not a complexity claim]")
            prov = "".join(f" {part}" for part in prov_parts)
            claim_eligible = (int(bench.get("n") or 0)
                              >= settings.min_seeds_for_claim
                              and reference_competent)
            if bench["complexity_justified"]:
                if not reference_competent:
                    state = "beats (ref not competent)"
                else:
                    state = "beats" if claim_eligible else "beats (provisional)"
            elif bench["below_reference"]:
                state = "below"
            elif bench.get("unmeasured"):
                state = "unmeasured (no curve-valid paired seeds)"
            elif bench.get("indistinguishable"):
                state = "indistinguishable"
            else:
                state = "inconclusive (CI wider than SESOI)"
            print(f"{gate:>6} "
                  f"{arm_mean:>17.4f} {ref_mean:>17.4f} "
                  f"{_fmt_ci(bench):>28} {state:>18}{prov}")
            benchmark_by_arm[gate] = {
                "arm_accuracy": arm_mean,
                "reference_accuracy": ref_mean,
                "mean": bench.get("mean"),
                "lo": bench.get("lo"),
                "hi": bench.get("hi"),
                "p_value": bench.get("p_value"),
                "n": bench.get("n"),
                "seeds_used": bench.get("seeds_used"),
                "seeds_dropped": bench.get("seeds_dropped"),
                "curve_invalid_excluded": bench.get("curve_invalid_excluded"),
                "sesoi": benchmark_sesoi,
                "state": state,
                "reference_competent": reference_competent,
                "claim_eligible": claim_eligible,
                "complexity_justified": bench["complexity_justified"],
                "below_reference": bench["below_reference"],
                "indistinguishable": bench["indistinguishable"],
                "inconclusive": bench["inconclusive"],
                "controlled": False,
                "observed_recipe_mismatches": (
                    (bench.get("controlled_audit") or {}).get(
                        "observed_mismatch_count", 0)),
            }
        above_sesoi = [g for g, row in benchmark_by_arm.items()
                       if row["complexity_justified"]]
        # A win may be statistically and practically above the reference and
        # still rest on fewer paired seeds than the pre-registered claim
        # floor.  Those rows are displayed as ``beats (provisional)`` and
        # must not enter the summary's *justified* lists: that would turn a
        # provisional observation into a claim by storage.
        justified = [g for g in above_sesoi
                     if benchmark_by_arm[g]["claim_eligible"]]
        reference_incompetent = [
            g for g in above_sesoi
            if not benchmark_by_arm[g]["reference_competent"]]
        provisional = [g for g in above_sesoi
                       if not benchmark_by_arm[g]["claim_eligible"]
                       and benchmark_by_arm[g]["reference_competent"]]
        # ``none`` is the ungated deep baseline, not a gating arm: it can
        # justify the depth of the grid but says nothing about the gate's
        # capacity.  Keep the two readings separate so a depth win is not
        # printed as a gate win.
        gating_justified = [g for g in justified if g != "none"]
        # Capacity-matched control: the depth-only benchmark above holds
        # width fixed, so its deep-vs-shallow gain is depth + parameters.
        # The baselines runner trains a 1-layer model widened to the deep
        # rung's parameter count; comparing the deep ungated arm to that
        # control is the capacity reading (still not a mechanism contrast:
        # depth and width both differ).
        matched = verdict.get("matched_shallow") or {}
        matched_runs = []
        matched_by_seed = matched.get("accuracy_by_seed") or {}
        matched_masks = matched.get("correct_bitmask_by_seed") or {}
        matched_ns = matched.get("eval_n_by_seed") or {}
        for key, acc in matched_by_seed.items():
            try:
                seed = int(key)
                n = int(matched_ns.get(str(seed), 0))
            except (TypeError, ValueError):
                continue
            mask = matched_masks.get(str(seed))
            if n <= 0 or not isinstance(mask, str):
                continue
            matched_runs.append({"seed": seed, "accuracy": float(acc),
                                 "n": n, "correct_bitmask": mask})
        if (matched_runs and matched.get("above_chance")
                and not matched.get("unmeasured", False)):
            matched_ref_name = (
                f"capacity-matched 1-layer standard transformer "
                f"(d_model={matched.get('d_model')}, "
                f"{matched.get('params')} params; target "
                f"{matched.get('target_params')}; "
                f"matched={matched.get('capacity_matched')})")
            bench = _paired_benchmark(
                by_gate["none"], matched_runs, "accuracy",
                benchmark_sesoi, matched_ref_name)
            capacity_ok = bool(matched.get("capacity_matched"))
            floor_ok = (int(bench.get("n") or 0)
                        >= settings.min_seeds_for_claim)
            claim_ok = capacity_ok and floor_ok
            if bench["complexity_justified"]:
                if not capacity_ok:
                    state = ("deep none beats capacity-unmatched shallow "
                             "control")
                elif not floor_ok:
                    state = "deep none beats matched capacity (provisional)"
                else:
                    state = "deep none beats matched capacity"
            elif bench["below_reference"]:
                state = ("deep none below capacity-unmatched shallow control"
                         if not capacity_ok else
                         "deep none below matched capacity")
            elif bench.get("unmeasured"):
                state = "unmeasured (no curve-valid paired seeds)"
            elif not capacity_ok:
                state = ("indistinguishable from capacity-unmatched shallow "
                         "control" if bench.get("indistinguishable") else
                         "inconclusive vs capacity-unmatched shallow control")
            elif bench.get("indistinguishable"):
                state = "indistinguishable from matched capacity"
            else:
                state = "inconclusive vs matched capacity (CI wider than SESOI)"
            print(f"\n[e1] capacity-matched shallow control (target = depth "
                  f"{matched.get('control_for_depth')} rung's "
                  f"{matched.get('target_params')} params; 1 layer, "
                  f"d_model={matched.get('d_model')}, "
                  f"{matched.get('params')} params, "
                  f"matched={matched.get('capacity_matched')}):")
            print(f"      deep(none) - matched shallow accuracy: "
                  f"{_fmt_ci(bench)}; {state}"
                  + ("" if floor_ok else
                     f" [provisional: n={bench.get('n')} < claim floor "
                     f"{settings.min_seeds_for_claim}]"))
            matched_benchmark = {
                "reference": matched_ref_name,
                "control_for_depth": matched.get("control_for_depth"),
                "target_params": matched.get("target_params"),
                "params": matched.get("params"),
                "d_model": matched.get("d_model"),
                "capacity_matched": capacity_ok,
                "reference_accuracy": (bench.get("arms") or {}).get(
                    "b", {}).get("mean"),
                "deep_accuracy": (bench.get("arms") or {}).get(
                    "a", {}).get("mean"),
                "mean": bench.get("mean"), "lo": bench.get("lo"),
                "hi": bench.get("hi"), "n": bench.get("n"),
                "seeds_used": bench.get("seeds_used"),
                "seeds_dropped": bench.get("seeds_dropped"),
                "curve_invalid_excluded": bench.get(
                    "curve_invalid_excluded"),
                "state": state, "claim_eligible": claim_ok,
                "complexity_justified": bench["complexity_justified"],
                "below_reference": bench["below_reference"],
                "indistinguishable": bench["indistinguishable"],
                "inconclusive": bench["inconclusive"],
                "unmeasured": bench["unmeasured"],
                "controlled": False,
                "observed_recipe_mismatches": (
                    (bench.get("controlled_audit") or {}).get(
                        "observed_mismatch_count", 0)),
                "comparison": "capacity_benchmark",
                "note": ("depth and width both differ from the deep rung; "
                         "this tests whether comparable one-layer capacity "
                         "closes the gap, not whether depth itself does"),
            }
        else:
            reason = matched.get("reason") or (
                "control not above chance" if matched_runs
                else "no matched-shallow control in the verdict")
            print(f"\n[e1] capacity-matched shallow control: not usable "
                  f"({reason}); depth claims remain confounded with "
                  f"parameter count")
            matched_benchmark = {"available": False, "reason": reason}
        complexity_benchmark = {
            "reference": reference_name,
            "reference_depth": shallow_depth,
            "reference_lr": shallow.get("lr"),
            "reference_accuracy_mean": shallow.get("accuracy_mean"),
            "reference_competent": reference_competent,
            "sesoi_accuracy": benchmark_sesoi,
            "by_arm": benchmark_by_arm,
            "arms_above_sesoi": above_sesoi,
            "arms_justified": justified,
            "provisional_arms_above_sesoi": provisional,
            "reference_incompetent_arms_above_sesoi": reference_incompetent,
            "gating_arms_justified": gating_justified,
            "provisional_gating_arms_above_sesoi": [
                g for g in provisional if g != "none"],
            "deep_none_justified": "none" in justified,
            "deep_none_provisional": "none" in provisional,
            "deep_none_reference_incompetent": "none" in reference_incompetent,
            "capacity_matched_reference": matched_benchmark,
            "comparison": "benchmark",
            "note": ("paired benchmark, not a controlled mechanism "
                     "contrast: architecture/capacity and the val-tuned "
                     "learning rate are the treatment; a non-justified arm "
                     "may be reported as a mechanism probe at fixed depth "
                     "but not as complexity justified over the tuned "
                     "simple model"),
        }
        if gating_justified:
            print(f"      gating arm(s) beating the tuned shallow reference "
                  f"by SESOI: {', '.join(gating_justified)}")
        elif "none" in justified:
            print(f"      the ungated deep baseline (none) beats the tuned "
                  f"shallow reference by SESOI, but no gated arm does: "
                  f"depth is justified, the gate's capacity is not")
        elif provisional:
            print(f"      provisional only: arm(s) {', '.join(provisional)} "
                  f"exceed the tuned shallow reference by SESOI but rest on "
                  f"fewer paired seeds than the claim floor "
                  f"({settings.min_seeds_for_claim}); not recorded as "
                  f"complexity justified")
        elif reference_incompetent:
            print(f"      reference not competent: arm(s) "
                  f"{', '.join(reference_incompetent)} exceed the tuned "
                  f"shallow reference by SESOI, but the reference rung is "
                  f"not above chance, so none is recorded as complexity "
                  f"justified")
        else:
            print(f"      WARNING: no E1 arm beats the tuned shallow "
                  f"reference by the pre-registered SESOI="
                  f"{benchmark_sesoi:g}; report these as mechanism probes at "
                  f"the fixed 8-layer depth, not as evidence that the "
                  f"gated architecture is worth its complexity at this scale")
    ladder = verdict.get("depth_ladder") or {}
    by_depth = ladder.get("by_depth") or {}
    if by_depth:
        rungs = "/".join(
            f"{d}L {float(row.get('accuracy_mean', float('nan'))):.4f}"
            f"±{float(row.get('accuracy_std', float('nan'))):.4f}"
            for d, row in sorted(by_depth.items(), key=lambda kv: int(kv[0])))
        sat = verdict.get("saturation_depth")
        req = verdict.get("hb7_depth_required")
        if req is None:
            if verdict.get("ladder_competent") is False:
                sat_txt = ("not determinable: no ladder rung is above "
                           "chance, so the ladder did not solve the task at "
                           "this step count - depth is unverified either way "
                           "(raise the step count and re-run `--only baselines`)")
            else:
                sat_txt = "unavailable (rule ceiling unknown)"
        elif req:
            sat_txt = ("no rung up to "
                       f"{max(int(d) for d in by_depth)} layers is within "
                       f"SESOI={ladder.get('sesoi_accuracy', float('nan')):g} "
                       "of the ceiling: depth is measurably required")
            if (sat is None
                    and verdict.get("saturation_depth_ci") is not None):
                sat_txt += (" (companion: depth "
                            f"{verdict['saturation_depth_ci']} only touches "
                            f"the ceiling by CI - absence of evidence, not "
                            f"saturation)")
        else:
            sat_txt = (f"saturated within SESOI="
                       f"{ladder.get('sesoi_accuracy', float('nan')):g} at "
                       f"{sat} layers: depth beyond that is not measurably "
                       f"required here")
        if req is True and not floor.get("hb7_claim_eligible", True):
            sat_txt += " [provisional: below the pre-registered seed floor]"
        print(f"[e1] complexity ladder (HB7, each rung val-tuned, "
              f"test accuracy): {rungs}; {sat_txt}")
        for pair, ci in (ladder.get("adjacent_contrasts") or {}).items():
            print(f"      adjacent-depth {pair} accuracy: {_fmt_ci(ci)}")

    print("\n[e1] H1b false-concentration decomposition (pre-registered "
          "threshold maxprob>0.5): 'false_conc_rate' is the error rate "
          "*within* the confident set, and a calibrated gate must close on "
          "those rows relative to its value on confident correct rows.  "
          "The margin alone is not enough: the two conditional levels and "
          "the size of the confident set are printed so a margin driven by "
          "a near-empty false-concentration group is visible; mean±std over "
          "seeds, 'n_high' is the confident-sample count per stats pass):")
    print(f"{'gate':>6} {'false_conc_rate[95%CI]':>27} "
          f"{'gate@false_conc':>17} {'gate@correct':>17} {'margin':>17} "
          f"{'n_high':>10} {'null_mass':>17} {'boot>0':>7}")
    for gate in GATES:
        t = table[gate]

        def fcell(key, fmt="{:.4f}"):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return fmt.format(m["mean"])

        rate = fcell('false_conc_rate')
        if rate != "n/a":
            lo = t.get("false_conc_rate_lo", {}).get("mean", float("nan"))
            hi = t.get("false_conc_rate_hi", {}).get("mean", float("nan"))
            if _finite(lo) and _finite(hi):
                rate += f"[{lo:.3f},{hi:.3f}]"
        boots = [r.get("false_conc_gate_boot") for r in by_gate[gate]]
        boots = [b for b in boots if b and b.get("applicable")]
        boot_ok = sum(1 for b in boots
                      if _finite(b.get("lo")) and float(b["lo"]) > 0.0)
        boot_s = f"{boot_ok}/{len(boots)}" if boots else "n/a"
        print(f"{gate:>6} {rate:>27} "
              f"{fcell('gate_on_false_conc'):>17} "
              f"{fcell('gate_on_correct'):>17} "
              f"{fcell('gate_margin'):>17} "
              f"{fcell('n_high_maxprob', '{:.1f}'):>10} "
              f"{fcell('null_mass_mean'):>17} {boot_s:>7}")
    print("  (false_conc_rate[95%CI] = Wilson interval on the confident rows "
          "of one stats stream; boot>0 = seeds whose within-stream "
          "independent two-sample bootstrap CI for gate_on_correct - "
          "gate_on_false_conc lies entirely above 0 - a different "
          "uncertainty from the seed-level margin)")

    # H1c's stream-level companion: the null arm has no gate to bootstrap,
    # so its null-mass error margin gets the same independent two-sample
    # treatment as the gated arms' gate margin.  This separates "the
    # diagnostic stream cannot resolve the two conditional levels" from
    # "the seed-level effect is absent".
    null_boot_ok = 0
    null_boot_n = 0
    null_fc_boot_ok = 0
    null_fc_boot_n = 0
    for r in by_gate["null"]:
        for key, counter in (("null_mass_all_boot", "all"),
                             ("null_mass_false_conc_boot", "fc")):
            boot = r.get(key) or {}
            if not boot.get("applicable"):
                continue
            if counter == "all":
                null_boot_n += 1
                null_boot_ok += int(_finite(boot.get("lo"))
                                    and float(boot["lo"]) > 0.0)
            else:
                null_fc_boot_n += 1
                null_fc_boot_ok += int(_finite(boot.get("lo"))
                                       and float(boot["lo"]) > 0.0)
    print(f"[e1] H1c null-slot within-stream check: independent two-sample "
          f"bootstrap of null_mass(error) - null_mass(correct) on each "
          f"seed's stats stream.  CI entirely above 0 (all errors): "
          f"{null_boot_ok}/{null_boot_n} seeds; on the high-maxprob subset: "
          f"{null_fc_boot_ok}/{null_fc_boot_n} seeds.  The seed-level H1c "
          f"contrast above remains the decision; this says whether the "
          f"finite diagnostic stream resolves the margin within a run.")

    # Keep the layer axis aligned: the aggregate table above uses a
    # max-over-layers confidence and a mean-over-layers gate, which is not
    # the within-layer quantity the gate actually computes from.  This
    # profile checks that the false-concentration closure (or the null-slot
    # error margin) is present in the individual layers and not created by
    # the layer collapse.
    print("\n[e1] H1b/H1c per-layer false-concentration profile (mean over "
          "seeds): each layer's own max-prob forms the confident set and that "
          "layer's own gate / null mass is read on it.  'margin' = "
          "gate@correct - gate@false_conc; 'null_margin' = null@error - "
          "null@correct.  The aggregate rows above remain the pre-registered "
          "H1b/H1c decision; a positive mean margin carried by one layer, or "
          "a layer with the opposite sign, is visible here:")
    print(f"{'arm':>6} {'layer':>6} {'false_conc_rate':>16} "
          f"{'gate@fc':>10} {'gate@ok':>10} {'margin':>10} "
          f"{'null@fc':>10} {'null@ok':>10} {'null_margin':>12}")
    layer_fc = {}

    def _layer_mean(rows, layer, key):
        vals = []
        for row_list in rows:
            if not isinstance(row_list, list) or layer >= len(row_list):
                continue
            value = row_list[layer].get(key)
            if _finite(value):
                vals.append(float(value))
        return (sum(vals) / len(vals)) if vals else float("nan"), len(vals)

    for gate in ("query", "calib", "null"):
        rows_by_seed = [r.get("false_conc_by_layer") for r in by_gate[gate]
                        if isinstance(r.get("false_conc_by_layer"), list)
                        and r["false_conc_by_layer"]]
        if not rows_by_seed:
            continue
        n_layers = max(len(row_list) for row_list in rows_by_seed)
        layer_fc[gate] = []
        for layer in range(n_layers):
            profile = {}
            for key in ("false_conc_rate", "gate_on_false_conc",
                        "gate_on_correct", "gate_margin", "null_mass_error",
                        "null_mass_correct", "null_mass_error_margin"):
                profile[key], profile[f"{key}_n_seeds"] = _layer_mean(
                    rows_by_seed, layer, key)
            profile["layer"] = layer
            layer_fc[gate].append(profile)

            def _cell(key, fmt="{:+.3f}"):
                value = profile[key]
                return fmt.format(value) if _finite(value) else "n/a"

            print(f"{gate:>6} {layer:>6} "
                  f"{_cell('false_conc_rate', '{:.3f}'):>16} "
                  f"{_cell('gate_on_false_conc', '{:.3f}'):>10} "
                  f"{_cell('gate_on_correct', '{:.3f}'):>10} "
                  f"{_cell('gate_margin'):>10} "
                  f"{_cell('null_mass_error', '{:.3f}'):>10} "
                  f"{_cell('null_mass_correct', '{:.3f}'):>10} "
                  f"{_cell('null_mass_error_margin'):>12}")
    if layer_fc:
        table["_false_concentration_by_layer"] = layer_fc

    print("\n[e1] H1a robustness: rank (Spearman) gate-entropy correlation "
          "alongside Pearson (a monotone effect must survive the change of "
          "association measure; mean±std over seeds - a difference smaller "
          "than the seed spread is not a robustness conclusion).  Both r "
          "columns are the Fisher-z mean of the *within-layer* Pearson "
          "coefficients, so between-layer operating-point offsets cannot "
          "create the association:")
    print(f"{'gate':>6} {'r_entropy':>17} {'rho_entropy':>19} "
          f"{'r_maxprob':>17} {'rho_maxprob':>19}")
    for gate in GATES:
        t = table[gate]

        def rcell(key):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return f"{m['mean']:.4f}±{m['std']:.4f}"

        print(f"{gate:>6} {rcell('gate_vs_entropy'):>17} "
              f"{rcell('gate_vs_entropy_spearman'):>19} "
              f"{rcell('gate_vs_maxprob'):>17} "
              f"{rcell('gate_vs_maxprob_spearman'):>19}")

    # The Fisher-z mean is a summary of *per-layer* associations; print the
    # layer profile for the gated arms so a single-layer association or a
    # depth sign flip is not hidden by the average.
    print("\n[e1] per-layer Pearson r (mean over seeds) behind the Fisher-z "
          "summary; 'L-i' are layer indices, 'per-layer std' is the "
          "between-seed spread at the layer level, not the Fisher-z spread:")
    layer_keys = []
    for gate in ("query", "calib"):
        for r in by_gate[gate]:
            values = r.get("gate_vs_entropy_per_layer")
            if isinstance(values, list) and values:
                layer_keys = list(range(len(values)))
                break
        if layer_keys:
            break
    if layer_keys:
        print(f"{'gate':>6} {'metric':>14}"
              + "".join(f"{'L' + str(i):>9}" for i in layer_keys))
        for gate in ("query", "calib"):
            for metric, label in (
                ("gate_vs_entropy_per_layer", "r_entropy"),
                ("gate_vs_maxprob_per_layer", "r_maxprob"),
            ):
                rows = [r[metric] for r in by_gate[gate]
                        if isinstance(r.get(metric), list) and r[metric]]
                if not rows:
                    continue
                width = min(len(v) for v in rows)
                cells = []
                for i in range(width):
                    vals = [float(v[i]) for v in rows
                            if i < len(v) and _finite(v[i])]
                    cells.append(
                        f"{sum(vals) / len(vals):+.3f}" if vals else "n/a")
                print(f"{gate:>6} {label:>14}"
                      + "".join(f"{c:>9}" for c in cells))

    # A significant calib-vs-query *difference* in |r| (the pre-registered
    # H1a row) does not by itself show that calib's gate is correlated with
    # its own attention statistics: both arms could be near zero.  These
    # one-sample CIs and the paired max-prob difference make the mechanism
    # leg readable at the correlation level.  They are exploratory (the
    # FDR-controlled decision remains H1a's calib-vs-query |r| row).
    corr_level = {}
    print("\n[e1] H1a correlation-level check (exploratory, outside the FDR "
          "family): |r|=0 is no association.  Calib's gate should be "
          "clearly associated with its own entropy/max-prob; the query-only "
          "gate has no access to that signal, so its |r| should stay weaker.  "
          "A calib |r| CI including 0 means the mechanism path is not "
          "established even if the calib-minus-query difference is positive:")
    print(f"{'readout':>40} {'paired/one-sample diff':>30} {'n':>5}")
    for metric, label in (("abs_gate_vs_entropy", "|r| entropy"),
                          ("abs_gate_vs_maxprob", "|r| maxprob")):
        for gate in ("query", "calib"):
            ci = _one_sample_contrast(by_gate[gate], metric)
            corr_level[f"{gate}:{metric}_vs_0"] = {
                "mean": ci.get("mean"), "lo": ci.get("lo"),
                "hi": ci.get("hi"), "n": ci.get("n"),
                "seeds_used": ci.get("seeds_used"),
                "seeds_dropped": ci.get("seeds_dropped"),
            }
            print(f"{gate + ' ' + label + ' vs 0':>40} {_fmt_ci(ci):>30} "
                  f"{int(ci.get('n') or 0):>5}")
        ci = _paired_diff(by_gate["calib"], by_gate["query"], metric)
        corr_level[f"calib-query:{metric}"] = {
            "mean": ci.get("mean"), "lo": ci.get("lo"), "hi": ci.get("hi"),
            "n": ci.get("n"), "seeds_used": ci.get("seeds_used"),
            "seeds_dropped": ci.get("seeds_dropped"),
            "controlled": (ci.get("controlled") or {}).get("controlled"),
        }
        print(f"{'calib - query ' + label:>40} {_fmt_ci(ci):>30} "
              f"{int(ci.get('n') or 0):>5}")
    table["_correlation_level_check"] = corr_level

    print("\n[e1] H1b robustness: gate margin (correct - false-conc) across "
          "the high-maxprob threshold sweep (pre-registered decision point "
          "is thr=0.5; a sign flip at other cutoffs means the margin is a "
          "property of the cutoff, not the mechanism; mean±std over seeds):")
    thresholds = sorted({float(t) for r in runs
                         for t in r.get("false_conc_thresholds", ())})
    print(f"{'gate':>6}" + "".join(f"{f'thr={t:g}':>17}" for t in thresholds))
    for gate in GATES:
        cells = []
        for thr in thresholds:
            agg = table[gate].get(f"gate_margin_thr{thr:g}", {})
            m = agg.get("mean", float("nan"))
            cells.append("n/a" if not math.isfinite(m)
                         else f"{m:+.4f}±{agg.get('std', float('nan')):.4f}")
        print(f"{gate:>6}" + "".join(f"{c:>17}" for c in cells))

    # HB0 competence prerequisite: "does the gate read entropy?" is only a
    # mechanism question when at least one trained condition demonstrably
    # learned the task at this step count (seed-clustered lower bound above
    # chance).  Written to e1_competence.json and handed to the ContrastLog,
    # which then marks every contrast claim-ineligible if it fails.
    competence = _competence_check(
        [(f"gate:{g}", by_gate[g]) for g in GATES]
        + [(f"ablation:{label}", rs)
           for label, rs in ablation_runs.items()],
        lambda r: (int(round(r["accuracy"] * r["n"])), int(r["n"])),
        1.0 / (settings.vocab - 2), what="E1 condition")
    _write_json(out_dir / "e1_competence.json",
                {**competence, **_run_context(settings)})
    if competence["passed"]:
        print(f"[e1] HB0 competence: ok (best arm {competence['best_arm']} "
              f"has seed-clustered 95% lower bound above chance "
              f"{competence['chance']:.4f})")
    else:
        print(f"[e1] HB0 competence: FAILED - no condition's seed-clustered "
              f"95% lower bound is above chance {competence['chance']:.4f}; "
              f"the grid did not demonstrably learn the task at this step count, "
              f"so no contrast below may be read as a mechanism statement")
    print("\n[e1] pre-registered contrasts (calib vs query/none):")
    clog = ContrastLog("e1",
                       hypotheses=("H1a", "H1b", "H1c", "H1d", "H1e", "H1f",
                                      "H1g"),
                       min_seeds=settings.min_seeds_for_claim,
                       competence=competence,
                       coverage=(ablation_coverage, design_coverage),
                       balance_ok=((verdict.get("eval_answer_balance") or {})
                                   .get("all_uniform")))
    clog.print_equivalence_ci(
        "calib - none", "H1a", "accuracy",
        _paired_equivalence(by_gate["calib"], by_gate["none"], "accuracy",
                            HYPOTHESES["H1a"]["sesoi"]["accuracy"],
                            mode="noninferior"),
        warn=_curve_warn(("calib", by_gate["calib"]),
                         ("none", by_gate["none"])))
    # The query gate is the closest tuned simpler model (same parameter
    # count, same architecture, gate on the query only): calibration must
    # not lose materially to *this* baseline, not only to `none` (which
    # lacks a gate altogether).  Pre-registered as one-sided
    # non-inferiority so a small loss inside the SESOI is not misread as a
    # directional refutation of H1a.
    clog.print_equivalence_ci(
        "calib - query", "H1a", "accuracy",
        _paired_equivalence(by_gate["calib"], by_gate["query"], "accuracy",
                            HYPOTHESES["H1a"]["sesoi"]["accuracy"],
                            mode="noninferior"),
        warn=_curve_warn(("calib", by_gate["calib"]),
                         ("query", by_gate["query"])))
    clog.print("calib - query", "H1a", "abs_gate_vs_entropy",
               _paired_diff(by_gate["calib"], by_gate["query"],
                            "abs_gate_vs_entropy"),
               warn=_curve_warn(("calib", by_gate["calib"]),
                                ("query", by_gate["query"])))
    # Same registered contrast on the other observable attention summary:
    # calib's gate must use the distribution, query's cannot, and max-prob
    # is the concentration readout the H1b false-concentration argument is
    # built on.  Leaving it exploratory would make half of the stated
    # evidence unaccounted for in the FDR family.
    clog.print("calib - query", "H1a", "abs_gate_vs_maxprob",
               _paired_diff(by_gate["calib"], by_gate["query"],
                            "abs_gate_vs_maxprob"),
               warn=_curve_warn(("calib", by_gate["calib"]),
                                ("query", by_gate["query"])))
    # H1g: the correlation-level halves of the same variant difference.
    # A calib-vs-query difference of two near-zero associations would not
    # show that calib reads the distribution; "query not significant" would
    # not show that it does not.  Calib rows are one-sample directional
    # tests, query rows are pre-registered equivalence (TOST) tests.
    for metric, label in (
        ("calib_abs_gate_vs_entropy", "entropy"),
        ("calib_abs_gate_vs_maxprob", "max-prob"),
    ):
        clog.print(f"calib |r| {label} vs 0", "H1g", metric,
                   _one_sample_contrast(by_gate["calib"], metric),
                   warn=_curve_warn(("calib", by_gate["calib"])))
    for metric, label in (
        ("query_abs_gate_vs_entropy", "entropy"),
        ("query_abs_gate_vs_maxprob", "max-prob"),
    ):
        bound = float(HYPOTHESES["H1g"]["sesoi"][metric])
        clog.print_equivalence_ci(
            f"query |r| {label} within {bound:g}", "H1g", metric,
            _one_sample_contrast(by_gate["query"], metric, bound=bound),
            warn=_curve_warn(("query", by_gate["query"])))
    clog.print("calib - query", "H1b", "gate_margin",
               _paired_diff(by_gate["calib"], by_gate["query"],
                            "gate_margin"),
               warn=_curve_warn(("calib", by_gate["calib"]),
                                ("query", by_gate["query"])))
    # H1e: the operational, imbalance-aware reading of the same gate
    # values.  Errors are the rare class after convergence, so the claim
    # "the calibrated gate concentrates doubt on its own errors" is a
    # rare-positive ranking claim, read through PR-AUC (reported as lift
    # over prevalence; 1.0 = random) and the recall of the most doubtful
    # 10%.  ROC-AUC travels in the tables as the optimistic companion but
    # is deliberately not a decision metric here.  Seeds whose stats
    # stream contains no errors leave the metric undefined: _paired_diff
    # records and drops them (never imputes them as 0).  The contrast is
    # pre-registered as calib vs query only: with gate="none" there is no
    # gate to rank by, so a calib-vs-none row would be undefined on one
    # side by construction and misread as a pairing failure.
    for metric, hyp in (
        ("prauc_lift_gate_to_error", "H1e"),
        ("recall_at_10pct_gate_to_error", "H1e"),
    ):
        clog.print("calib - query", hyp, metric,
                   _paired_diff(by_gate["calib"], by_gate["query"], metric),
                   warn=_curve_warn(("calib", by_gate["calib"]),
                                    ("query", by_gate["query"])))
    clog.print("null margin vs 0", "H1c", "null_mass_error_margin",
               _one_sample_contrast(by_gate["null"],
                                    "null_mass_error_margin"),
               warn=_curve_warn(("null", by_gate["null"])))
    # The same H1c claim on the H1b false-concentration subset: a null slot
    # that absorbs writes only on unconfident errors but not on confident
    # false concentration would not support the suppression reading at the
    # operating point the gate mechanism is evaluated.
    clog.print("null false-conc margin vs 0", "H1c",
               "null_mass_false_conc_margin",
               _one_sample_contrast(by_gate["null"],
                                    "null_mass_false_conc_margin"),
               warn=_curve_warn(("null", by_gate["null"])))
    # H1c completeness companion: the null arm's *accuracy* relative to the
    # no-gate baseline.  The within-arm margin above only shows the slot
    # absorbs the write; a slot that also costs recall would make the null
    # arm a bad component whatever the margin says.  This is a one-sided
    # non-inferiority check (the registered null prediction "adding the null
    # slot is not materially worse than no gate"), so it is routed through
    # the same FDR family with its own pre-registered bound rather than left
    # to the exploratory loop.
    clog.print_equivalence_ci(
        "null - none", "H1c", "accuracy",
        _paired_equivalence(by_gate["null"], by_gate["none"], "accuracy",
                            HYPOTHESES["H1c"]["sesoi"]["accuracy"],
                            mode="noninferior"),
        warn=_curve_warn(("null", by_gate["null"]), ("none", by_gate["none"])))
    # H1a completeness companion: the query-only gate vs the no-gate
    # baseline.  The registered `calib` contrasts credit calibration against
    # the query and against `none`, but the query arm's own effect was only
    # ever in the uncorrected exploratory loop; registering it gives the
    # "does having a gate at all help" leg a pre-registered, FDR-controlled
    # row (two-sided: the companion measures whether the simplest gate moves
    # accuracy, not a signed claim - it routes through its own undirected
    # registry key so a significant loss is not read as refuting H1a's
    # calib-vs-controls prediction).
    clog.print("query - none", "H1a", "accuracy",
               _paired_diff(by_gate["query"], by_gate["none"], "accuracy"),
               warn=_curve_warn(("query", by_gate["query"]),
                                ("none", by_gate["none"])),
               direction_key="query_none_accuracy")
    print(f"\n[e1] H1d calibration-term ablation contrasts (each row removes "
          f"exactly one term from the full gate - calib_len_only drops the "
          f"concentration term, calib_ent_only drops the length term.  Only "
          f"that one resolved term flag varies, the recipe stays frozen, and "
          f"every contrast carries its controlled-variable audit.  The "
          f"calib_no_terms cell disables *both* terms at once, so it is a "
          f"two-factor manipulation, not a leave-one-out arm: it is read "
          f"only through the registered 2x2 factorial main effect below "
          f"and is never given a single-term decision row):")
    for label in ("calib_len_only", "calib_ent_only"):
        rs = ablation_runs[label]
        warn = _curve_warn(("calib", by_gate["calib"]), (label, rs))
        clog.print(f"calib - {label}", "H1d", "accuracy",
                   _paired_diff(by_gate["calib"], rs, "accuracy"), warn=warn)
        clog.print(f"calib - {label}", "H1d", "gate_margin",
                   _paired_diff(by_gate["calib"], rs, "gate_margin"),
                   warn=warn)
    # Only the arm that actually removes the concentration term can test
    # the mechanism leg ("the association collapses when the entropy term
    # is removed").  ``calib_ent_only`` keeps the concentration term on and
    # removes the *length* term, so its |r| is not a leave-one-out
    # manipulation of the pathway this row measures; routing it through the
    # same directional H1d decision would attribute a length-term change to
    # the concentration term.  The complete 2x2 is still read through the
    # factorial main effect below.
    for label in ("calib_len_only",):
        rs = ablation_runs[label]
        clog.print(f"calib - {label}", "H1d", "abs_gate_vs_entropy",
                   _paired_diff(by_gate["calib"], rs, "abs_gate_vs_entropy"),
                   warn=_curve_warn(("calib", by_gate["calib"]), (label, rs)))
    # Full-factorial companion to the conditional leave-one-out rows: the
    # concentration term's effect averaged over both levels of the length
    # term.  The length main effect at a *fixed* training length is a
    # constant absorbed by the gate bias (the hypothesis says so), so it is
    # tested where it varies (H2c), not here.  These rows go through the
    # same FDR family as the LOO contrasts.
    factorial_cells = {"full": by_gate["calib"]}
    for cell in _CALIB_FACTORIAL_CELLS:
        if cell != "full":
            factorial_cells[cell] = ablation_runs[cell]
    factorial_warn = _curve_warn(
        *[(cell, factorial_cells[cell]) for cell in _CALIB_FACTORIAL_CELLS])
    print(f"\n[e1] H1d factorial main effect (concentration term): the "
          f"average of full-vs-len_only and ent_only-vs-no_terms, i.e. the "
          f"term's contribution averaged over both length-term levels (the "
          f"LOO rows above are conditional on the other term being on):")
    for metric in ("accuracy", "gate_margin", "abs_gate_vs_entropy"):
        ci = _factorial_main_effect(factorial_cells, metric, "concentration")
        clog.print("concentration main effect (2x2)", "H1d", metric, ci,
                   warn=factorial_warn)
    print(f"\n[e1] H1f active-control (placebo) contrasts: full calib vs "
          f"the shuffled-entropy arm (identical architecture/parameters/"
          f"recipe; only the signal's link to the read row changes).  A "
          f"supported accuracy/gate_margin read is the specificity test - "
          f"the entropy *information* does the work; the mechanism leg "
          f"validates that the placebo actually severed the pathway:")
    placebo_warn = _curve_warn(("calib", by_gate["calib"]),
                               (_CALIB_PLACEBO_CELL, placebo_runs))
    for metric in ("abs_gate_vs_entropy", "accuracy", "gate_margin"):
        clog.print(f"calib - {_CALIB_PLACEBO_CELL}", "H1f", metric,
                   _paired_diff(by_gate["calib"], placebo_runs, metric),
                   warn=placebo_warn)
    for metric in ("abs_gate_vs_entropy", "accuracy", "gate_margin"):
        ci = _paired_diff(placebo_runs, by_gate["query"], metric)
        print(f"      exploratory (outside the FDR family): "
              f"{_CALIB_PLACEBO_CELL} - query ({metric}): {_fmt_ci(ci)} - "
              f"a CI including 0 means the placebo's extra capacity alone "
              f"buys no advantage over the query gate")
    clog.finalize(out_dir, q=settings.fdr_q)
    metric_robustness = _class_frequency_companions(
        [("calib - none", by_gate["calib"], by_gate["none"], "H1a"),
         ("calib - query", by_gate["calib"], by_gate["query"], "H1a"),
         ("query - none", by_gate["query"], by_gate["none"], "H1a"),
         (f"calib - {_CALIB_PLACEBO_CELL}", by_gate["calib"],
          placebo_runs, "H1f")],
        "e1")
    lr_sensitivity = _e1_lr_sensitivity(settings, out_dir, lr, by_gate)
    per_variant_tuning = _e1_per_variant_tuning(
        settings, out_dir, lr, by_gate)

    print(f"\n[e1] training curves (val monitored on fixed stream seed "
          f"{VAL_SEED}; verdict -> recommendation; gradmax = max pre-clip "
          f"gradient norm; vreg = val loss regression past its best point; "
          f"late = U-shaped val curves where the global slope hides the "
          f"regression; overfit_sig = overfitting verdicts that also exceed "
          f"the validation-noise tolerance (window-averaged train loss, see "
          f"train_loss_window in e1_runs.jsonl); tail_train/tail_val = "
          f"relative change over the last "
          f"third (a falling tail under a plateau verdict means the step "
          f"count, not the pipeline, is the limit); vse = last validation "
          f"point's "
          f"batch-sampling SE (vreg within ~2x vse is stream noise); "
          f"lr = first->last logged learning rate):")
    for gate in GATES:
        verdicts, invalid = _curve_health(by_gate[gate])
        late = sum(1 for r in by_gate[gate] if r.get("late_overfit"))
        late_sig = sum(1 for r in by_gate[gate]
                       if r.get("late_overfit_significant"))
        overfit_sig = sum(1 for r in by_gate[gate]
                          if r.get("overfitting_significant"))
        gap = _mean_std([r["generalization_gap"] for r in by_gate[gate]])
        gmax = _mean_std([r["grad_norm_max"] for r in by_gate[gate]])
        vreg = _mean_std([r["val_regression_after_best"] for r in by_gate[gate]])
        ttail = _mean_std([r.get("late_train_delta") for r in by_gate[gate]])
        vtail = _mean_std([r.get("late_val_delta") for r in by_gate[gate]])
        vse = _mean_std([r.get("val_loss_se_last") for r in by_gate[gate]])
        first = by_gate[gate][0]
        gmax_s = f"{gmax['mean']:.2f}" if math.isfinite(gmax["mean"]) else "n/a"
        vreg_s = f"{vreg['mean']:+.4f}" if math.isfinite(vreg["mean"]) else "n/a"
        ttail_s = (f"{ttail['mean']:+.3f}" if math.isfinite(ttail["mean"])
                   else "n/a")
        vtail_s = (f"{vtail['mean']:+.3f}" if math.isfinite(vtail["mean"])
                   else "n/a")
        vse_s = f"{vse['mean']:.4f}" if math.isfinite(vse["mean"]) else "n/a"
        lr_first = first.get("lr_first", float("nan"))
        lr_last = first.get("lr_last", float("nan"))
        lr_s = (f"{lr_first:g}->{lr_last:g}"
                if _finite(lr_first) and _finite(lr_last) else "n/a")
        print(f"{gate:>6} train {first['train_loss_first']:.3f}->"
              f"{first['train_loss_last']:.3f}  val "
              f"{first['val_loss_first']:.3f}->{first['val_loss_last']:.3f}  "
              f"gap={gap['mean']:.4f}  gradmax={gmax_s}  vreg={vreg_s}  "
              f"tail_train={ttail_s}  tail_val={vtail_s}  vse={vse_s}  "
              f"lr={lr_s}  invalid={invalid}/{len(by_gate[gate])}  "
              f"late={late} (sig={late_sig})  overfit_sig={overfit_sig}  "
              f"verdicts={verdicts}")
        for label, count in verdicts.items():
            rec = next(r["curve_recommendation"] for r in by_gate[gate]
                       if r["curve_verdict"] == label)
            print(f"       [{label} x{count}] {rec}")

    print(f"[e1] accuracy delta vs none, each gate (exploratory and "
          f"uncorrected - the pre-registered family above is FDR-controlled; "
          f"a gain is only read as supported when the CI excludes 0):")
    for gate in GATES:
        if gate == "none":
            continue
        acc_ci = _paired_diff(by_gate[gate], by_gate["none"], "accuracy")
        warn = _curve_warn((gate, by_gate[gate]), ("none", by_gate["none"]))
        print(f"  {gate:>6}: {_fmt_ci(acc_ci)}"
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
          f"miss (attention mass < 0.5 on the correct key in every block) > "
          f"found_then_lost (mass >= 0.5 in some block, < 0.5 in the last) > "
          f"wrong readout; label_noise_rate is *measured* on these batches):")
    print(f"{'gate':>6} {'label':>8} {'struct':>8} {'retrieval':>10} "
          f"{'lost':>8} {'readout':>8} {'label_noise':>12}")
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
              f"{tcell('err_taxonomy_found_then_lost'):>8} "
              f"{tcell('err_taxonomy_readout_wrong'):>8} "
              f"{tcell('err_label_noise_rate'):>12}")
    _print_structural_decomposition(
        [(gate, by_gate[gate]) for gate in GATES], title="[e1]")
    print(f"\n[e1] attention mass on the correct key by layer (mean over "
          f"stats samples; the last-layer miss vs the per-layer profile tells "
          f"'never found' apart from 'found mid-stack, lost again'):")
    def _layer_means(gate, key):
        rows = [r[key] for r in by_gate[gate]
                if isinstance(r.get(key), list) and r[key]]
        if not rows:
            return []
        n = min(len(v) for v in rows)
        return [sum(float(v[i]) for v in rows) / len(rows) for i in range(n)]

    layer_keys = []
    for gate in GATES:
        means = _layer_means(gate, "err_attn_mass_on_key_error_by_layer")
        if means:
            layer_keys = list(range(len(means)))
            break
    if layer_keys:
        print(f"{'gate':>6}" + "".join(f"{'L' + str(i):>8}" for i in layer_keys)
              + f"{'last-first':>11}")
        for gate in GATES:
            means = _layer_means(gate, "err_attn_mass_on_key_error_by_layer")
            if not means:
                continue
            cells = "".join(
                f"{v:>8.3f}" if math.isfinite(v) else f"{'n/a':>8}"
                for v in means)
            first, last = means[0], means[-1]
            drift = (last - first) if (math.isfinite(first)
                                       and math.isfinite(last)) else float("nan")
            drift_s = f"{drift:+.3f}" if math.isfinite(drift) else "n/a"
            print(f"{gate:>6}{cells}{drift_s:>11}")
    _print_conf_by_error_type([(g, by_gate[g]) for g in GATES], title="[e1]")
    print("  most confident wrong samples per gate (systematic blind spots "
          "are the most informative):")
    for gate in GATES:
        for case in examples.get(gate, (0, []))[1]:
            print(f"  [{gate}] {_format_case(case)}")
    table["_efficiency"] = _efficiency_companion(
        settings, out_dir, "e1", [(gate, by_gate[gate]) for gate in GATES],
        reference="none")
    table["_complexity_cost"] = _complexity_cost_companion(
        settings, out_dir, verdict, by_gate,
        deep_cost=table["_efficiency"]["cells"].get("none"))
    table["_provenance"] = _stamp(settings)
    if complexity_benchmark is not None:
        table["_complexity_benchmark"] = complexity_benchmark
    table["_class_frequency_companions"] = metric_robustness
    table["_lr_sensitivity"] = lr_sensitivity
    table["_per_variant_tuning"] = per_variant_tuning
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


def _calibration_coefficients(model) -> dict:
    """Read the learned calibration coefficients of a trained model.

    H2b's mechanism claim is that the calibrated gate *explicitly* corrects
    the ``log T`` growth of attention entropy.  The gate is
    ``sigmoid((W_g x - a*log T - c*scale*Hbar)/tau)``, so the direct
    manipulation check is the sign and size of the learned ``a`` (length
    term) and effective ``c*scale`` (concentration term) - not another grid
    point.  A full-calib run whose learned length coefficient is negative
    would be evidence that the length correction is being used in the
    opposite direction, whatever the end accuracy; an ablated arm that
    reports a non-zero ``a`` would mean the ablation did not actually
    remove the pathway.

    This is a readout, not a new decision rule: the pre-registered H2a/H2c
    contrasts remain the decisions, and the coefficients explain *how* the
    trained gate implements (or fails to implement) the mechanism.  Returns
    ``{"applicable": False}`` for non-calib gates and a per-layer list plus
    aggregate means/sign counts for calib variants.
    """
    rows = []
    for layer_idx, block in enumerate(getattr(model, "blocks", ())):
        attn = block.attn
        tau = getattr(attn, "calib_tau", None)
        if tau is None:
            continue
        scale_p = getattr(attn, "calib_scale", None)
        scale = (float(scale_p.detach()) if scale_p is not None
                 else float("nan"))
        a_p = getattr(attn, "calib_a", None)
        c_p = getattr(attn, "calib_c", None)
        a = float(a_p.detach()) if a_p is not None else float("nan")
        c = float(c_p.detach()) if c_p is not None else float("nan")
        effective_tau = float(torch.nn.functional.softplus(tau.detach())) + 0.1
        ent_norm = bool(getattr(attn, "calib_ent_norm", True))
        rows.append({
            "layer": int(layer_idx),
            "a": a, "c": c, "scale": scale,
            "c_effective": (c * scale if math.isfinite(c)
                            and math.isfinite(scale) else float("nan")),
            "tau_effective": effective_tau,
            # The concentration coefficient's units depend on the signal it
            # multiplies: H/log T (registered gate) or raw H (H2g arm).
            # Recorded so a c*scale comparison across arms is not silently
            # read as if the two coefficients had the same scale.
            "ent_norm": ent_norm,
        })
    if not rows:
        return {"applicable": False}

    def _mean(key):
        vals = [row[key] for row in rows if _finite(row.get(key))]
        return sum(vals) / len(vals) if vals else float("nan")

    def _positive(key):
        vals = [row[key] for row in rows if _finite(row.get(key))]
        return sum(1 for v in vals if v > 0.0)

    return {
        "applicable": True,
        "layers": rows,
        "n_layers": len(rows),
        "a_mean": _mean("a"),
        "c_mean": _mean("c"),
        "c_effective_mean": _mean("c_effective"),
        "scale_mean": _mean("scale"),
        "tau_effective_mean": _mean("tau_effective"),
        "a_positive_layers": _positive("a"),
        "c_effective_positive_layers": _positive("c_effective"),
        "a_layers": sum(1 for row in rows if _finite(row.get("a"))),
        "c_layers": sum(1 for row in rows if _finite(row.get("c"))),
        "ent_norm_layers": sum(1 for row in rows if row.get("ent_norm")),
        "concentration_signal": (
            "H/logT" if all(row.get("ent_norm") for row in rows)
            else "H" if not any(row.get("ent_norm") for row in rows)
            else "mixed"),
    }


# E2 length-generalisation arms: the four gate variants plus the complete
# 2x2 over the calib gate's two additive terms (the full cell is the
# ``calib`` arm itself), so the arm that isolates any H2a/H2b effect (the
# length term, H2c) is measured on the same frozen recipe and the same
# length sweep inside a full-factorial design - no term combination is left
# unmeasured.  ``calib_raw_ent`` adds the one-factor normalisation ablation
# (H2g): both terms on, identical architecture, but the concentration term
# reads raw entropy instead of H/log(T).  The calib arms carry
# ``calib_variant`` labels next to ``gate="calib"``, so the
# controlled-variable audit attributes a contrast to the term/normalisation
# flag rather than to the gate.
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
    ("calib_raw_ent", "calib", {"calib_variant": _CALIB_RAW_ENT_CELL},
     {"calib_use_len": True, "calib_use_ent": True,
      "calib_ent_norm": False}),
    ("null", "null", {}, {}),
)


def _e2_per_length_contrasts(settings: Settings, out_dir: pathlib.Path,
                             by_variant: dict, lengths: list) -> dict:
    """Paired calib-vs-control CIs at *every* sweep length.

    The pre-registered E2 decision reads only the longest length (plus the
    base-to-longest drop), which is the right endpoint for the length- and
    concentration-term hypotheses.  It can still hide a mid-sweep reversal:
    an arm that is worse at an intermediate length but recovers by the
    longest one would leave the endpoint verdict unchanged while
    undermining the claim that calibration helps as length grows.  This
    companion re-pairs the same seeds and streams at each length, reports
    the paired CI, and flags interior CIs whose sign is opposite to the
    endpoint effect (the ``worst_length`` is also recorded, since an
    endpoint that is not the worst point overstates robustness).  The rows
    are exploratory: they do not enter the FDR family and cannot change
    the pre-registered endpoint decision.
    """
    print(f"\n[e2] per-length paired contrasts (exploratory, outside the "
          f"FDR family): the endpoint decision above uses L={lengths[-1]} "
          f"only; these rows ask whether an earlier sweep length reverses "
          f"or exaggerates that effect on the same seeds/streams.  An "
          f"interior reversal is diagnostic, not a new decision rule.")
    print(f"{'contrast':>18} {'length':>7} {'arm_a':>9} {'control':>9} "
          f"{'paired_diff':>28} {'seeds':>6} {'control':>9}")
    payload = {"lengths": list(lengths), "contrasts": {},
               "note": ("exploratory per-length companion; endpoint rows "
                        "above remain the pre-registered decision and the "
                        "FDR family is unchanged")}
    for rule, a_label, b_label in (
        ("calib - none", "calib", "none"),
        ("calib - query", "calib", "query"),
        ("calib - calib_raw_ent", "calib", _CALIB_RAW_ENT_CELL),
    ):
        runs_a = by_variant[a_label]
        runs_b = by_variant[b_label]
        rule_rows = []
        for length in lengths:
            def _at_length(r):
                metrics = (r.get("metrics_by_length") or {}).get(length) or {}
                return dict(
                    r, accuracy=r["accuracy_by_length"][length],
                    correct_bitmask=metrics.get("correct_bitmask"),
                    n=(r.get("eval_n_by_length") or {}).get(length))
            tagged_a = [_at_length(r) for r in runs_a]
            tagged_b = [_at_length(r) for r in runs_b]
            ci = _paired_diff(tagged_a, tagged_b, "accuracy")
            audit = ci.get("controlled") or {}
            signed = 0
            if (_finite(ci.get("lo")) and _finite(ci.get("hi"))
                    and (ci["lo"] > 0.0 or ci["hi"] < 0.0)
                    and _finite(ci.get("mean"))):
                signed = 1 if ci["mean"] > 0.0 else -1
            row = {
                "rule": rule, "length": int(length),
                "arm_a": a_label, "control": b_label,
                "a_mean": (ci.get("arms") or {}).get("a", {}).get("mean"),
                "b_mean": (ci.get("arms") or {}).get("b", {}).get("mean"),
                "mean": ci.get("mean"), "lo": ci.get("lo"),
                "hi": ci.get("hi"), "p_value": ci.get("p_value"),
                "n": ci.get("n"), "sesoi": HYPOTHESES["H2a"]["sesoi"]["accuracy_at_longest"],
                "seeds_used": ci.get("seeds_used"),
                "seeds_dropped": ci.get("seeds_dropped"),
                "curve_invalid_seeds": ci.get("curve_invalid_seeds"),
                "controlled": audit.get("controlled"),
                "ci_excludes_zero": bool(signed),
                "signed_direction": signed,
                "stream_test": ci.get("stream_test"),
            }
            rule_rows.append(row)
            print(f"{rule:>18} {length:>7} "
                  f"{row['a_mean'] if _finite(row['a_mean']) else float('nan'):>9.4f} "
                  f"{row['b_mean'] if _finite(row['b_mean']) else float('nan'):>9.4f} "
                  f"{_fmt_ci(ci):>28} {ci.get('n', 0):>6} "
                  f"{'ok' if audit.get('controlled') else 'FAILED':>9}")
        endpoint = rule_rows[-1]
        direct = _paired_diff(runs_a, runs_b, "accuracy_at_longest")
        recompute = {
            "mean": direct.get("mean"), "lo": direct.get("lo"),
            "hi": direct.get("hi"), "n": direct.get("n"),
            "seeds_used": direct.get("seeds_used"),
        }

        def _same_number(x, y):
            if x is None or y is None:
                return x is None and y is None
            try:
                xf, yf = float(x), float(y)
            except (TypeError, ValueError, OverflowError):
                return False
            if math.isnan(xf) or math.isnan(yf):
                return math.isnan(xf) and math.isnan(yf)
            if math.isinf(xf) or math.isinf(yf):
                return xf == yf
            return abs(xf - yf) <= 1e-12

        endpoint_matches = bool(
            _same_number(endpoint["mean"], recompute["mean"])
            and _same_number(endpoint["lo"], recompute["lo"])
            and _same_number(endpoint["hi"], recompute["hi"])
            and endpoint["n"] == recompute["n"]
            and endpoint["seeds_used"] == recompute["seeds_used"])
        if not endpoint_matches:
            print(f"      -> {rule}: WARNING the per-length longest-row does "
                  f"not reproduce the pre-registered accuracy_at_longest "
                  f"contrast (per-length {_fmt_ci(endpoint)} vs direct "
                  f"{_fmt_ci(direct)}); treat the per-length alignment as "
                  f"broken")
        endpoint_sign = endpoint["signed_direction"]
        interior = rule_rows[1:-1]
        reversals = [row["length"] for row in interior
                     if endpoint_sign and row["signed_direction"]
                     and row["signed_direction"] != endpoint_sign]
        finite_means = [row for row in rule_rows if _finite(row["mean"])]
        worst = (min(finite_means, key=lambda row: row["mean"])
                 if finite_means else None)
        best = (max(finite_means, key=lambda row: row["mean"])
                if finite_means else None)

        def _same_mean(row):
            # A tie for the most extreme point still includes the endpoint:
            # comparing only the arg-min/arg-max *index* would report
            # ``endpoint_is_worst=False`` when another length happens to be
            # returned first by ``min`` despite an equal mean.
            return bool(row is not None
                        and _same_number(endpoint["mean"], row["mean"]))

        verdict = {
            "endpoint_length": int(lengths[-1]),
            "endpoint_mean": endpoint["mean"],
            "endpoint_signed_direction": endpoint_sign,
            "interior_reversal_lengths": reversals,
            "interior_reversal": bool(reversals),
            "endpoint_is_worst": _same_mean(worst),
            "endpoint_is_best": _same_mean(best),
            "worst_length": None if worst is None else worst["length"],
            "worst_mean": None if worst is None else worst["mean"],
            "best_length": None if best is None else best["length"],
            "best_mean": None if best is None else best["mean"],
            "endpoint_recompute": recompute,
            "endpoint_matches_registered": endpoint_matches,
        }
        if reversals:
            print(f"      -> {rule}: INTERIOR REVERSAL at L={reversals} "
                  f"(endpoint sign {endpoint_sign:+d}); the endpoint "
                  f"decision is unchanged, but length robustness is not "
                  f"monotone")
        elif endpoint_sign and worst is not None and worst["length"] != lengths[-1]:
            direction = ("benefit" if endpoint_sign > 0 else "loss")
            relation = ("overstates" if endpoint_sign > 0 else "understates")
            print(f"      -> {rule}: no sign reversal, but the endpoint "
                  f"{relation} the {direction} (most extreme L="
                  f"{worst['length']}, {worst['mean']:+.4f} vs endpoint "
                  f"{endpoint['mean']:+.4f}); the endpoint decision is "
                  f"unchanged but is not the conservative summary")
        elif endpoint_sign:
            print(f"      -> {rule}: no interior reversal and the endpoint "
                  f"is the worst point")
        else:
            print(f"      -> {rule}: endpoint effect is not confidently "
                  f"signed, so no reversal conclusion is drawn")
        payload["contrasts"][rule] = {"rows": rule_rows,
                                      "summary": verdict}
    _write_json(out_dir / "e2_length_contrasts.json", payload)
    return payload


def _e2_per_variant_tuning(settings: Settings, out_dir: pathlib.Path,
                           reference_lr: float, reference_runs: dict,
                           lengths: list) -> dict:
    """Give every primary E2 arm its own validation-selected recipe.

    The pre-registered E2 family freezes one recipe (tuned on ``none`` at the
    training length) for every arm.  That is the correct controlled contrast,
    but it leaves the same fairness question as E1: if ``query`` or ``null``
    prefers a different learning rate (or ``calib`` a different temperature),
    the frozen recipe can make a control look weaker than it is.  This
    companion selects, per primary arm and on ``VAL_SEED`` only:

    * an lr from the shared ``lr_grid`` plus the frozen reference point;
    * for calib, a ``calib_tau_init`` at the selected lr.

    Selection is read at the *training* length (the same length the shared
    recipe was tuned on); the selected recipe is then trained on the full
    seed set and scored on the sealed test stream at the training and longest
    sweep lengths.  It is deliberately not routed through ``ContrastLog``:
    the arms differ in lr/temperature, so the controlled-variable audit must
    fail and the row is an exploratory benchmark.  The pre-registered
    same-recipe H2a/H2d family remains the primary conclusion; this payload
    answers "does calib still degrade less when the baselines also get their
    own best shot?".

    Only the four primary arms are tuned (the calibration-term ablations are
    mechanism decompositions, not baselines).  Reference runs are reused when
    the selected recipe is the frozen one, so a baseline already optimal under
    the controlled point costs no extra final runs.
    """
    if not settings.per_variant_tuning:
        return {"available": False,
                "reason": "disabled by Settings.per_variant_tuning"}
    reference_lr = float(reference_lr)
    if not math.isfinite(reference_lr) or reference_lr <= 0.0:
        return {"available": False, "reason": "non-positive reference lr"}
    primary = ("none", "query", "calib", "null")
    tune_seeds = list(settings.seeds[:max(
        1, min(int(settings.per_variant_tune_seeds), len(settings.seeds)))])
    claim_floor = min(int(settings.min_seeds_for_claim), len(settings.seeds))
    if len(tune_seeds) < claim_floor:
        print(f"[e2] WARNING: per-variant tuning selects on "
              f"{len(tune_seeds)} seed(s), below the pre-registered claim "
              f"floor {claim_floor}; the tuned recipes are noisier than the "
              f"controlled ones and the two accounts must still be read "
              f"separately")
    candidate_lrs = sorted({reference_lr,
                            *(float(v) for v in settings.lr_grid)})
    default_tau = float(Cfg().calib_tau_init)
    candidate_taus = sorted({default_tau,
                             *(float(v) for v in settings.calib_tau_grid)})
    by_arm = {}
    tuning_rows = []

    def _tune_run(gate, seed, lr, tau, phase):
        overrides = ({"calib_tau_init": tau} if gate == "calib" else {})
        _, params, metrics, curve, model_cfg = _train_and_eval(
            settings, seed, lr, eval_seed=VAL_SEED, selection=True,
            gate=gate, **overrides)
        rec = {
            "experiment": "e2_per_variant_tuning",
            "hypotheses": ["H2a", "H2d"],
            "phase": phase, "variant": gate, "gate": gate,
            "seed": int(seed), "lr": float(lr),
            "calib_tau_init": (float(tau) if gate == "calib" else None),
            "eval_seed": VAL_SEED, "params": int(params),
            "val_accuracy": float(metrics["accuracy"]),
            "model_cfg": model_cfg,
            **_training_record(settings, lr),
            **_run_context(settings),
            **curve,
        }
        _write_result(out_dir / "e2_per_variant_tuning_runs.jsonl", rec)
        tuning_rows.append(rec)
        return rec

    def _pick(rows, key, label, preferred=None):
        pick = _curve_valid_selection_rows(rows)
        eligible = pick["rows"]
        if not eligible:
            return None, {
                "label": label, "pool": pick["pool"],
                "n_finite": pick["n_finite"], "n_invalid": pick["n_invalid"],
                "reason": "no finite validation row",
            }
        # Break accuracy ties toward the frozen reference point: an
        # arbitrary argmax on equal validation accuracy would select a
        # different recipe for no measured reason and forgo reference-run
        # reuse.  The controlled point is the conservative choice.
        best = max(eligible, key=lambda row: (
            row["val_accuracy_mean"],
            preferred is not None and row[key] == preferred))
        selection = _selection_reliability(
            eligible, grid=[row[key] for row in eligible])
        selection["pool"] = pick["pool"]
        selection["n_finite"] = pick["n_finite"]
        selection["n_invalid"] = pick["n_invalid"]
        selection["excluded_curve_invalid_rows"] = pick["n_excluded"]
        return best, selection

    for gate in primary:
        lr_rows = []
        for lr in candidate_lrs:
            runs = [_tune_run(gate, seed, lr, default_tau, "lr")
                    for seed in tune_seeds]
            accs = [r["val_accuracy"] for r in runs]
            mean = sum(accs) / len(accs) if accs else float("nan")
            var = (sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)
                   if len(accs) > 1 else 0.0)
            verdicts = [r["curve_verdict"] for r in runs]
            lr_rows.append({
                "lr": float(lr), "calib_tau_init": default_tau,
                "val_accuracy_mean": mean,
                "val_accuracy_std": math.sqrt(var) if math.isfinite(var)
                else float("nan"),
                "n_seeds": len(accs), "curve_verdicts": verdicts,
                "curve_invalid_runs": sum(
                    v in CURVE_INVALID_VERDICTS for v in verdicts),
                "runs": runs,
            })
        best_lr_row, lr_selection = _pick(
            lr_rows, "lr", f"{gate}:lr", preferred=reference_lr)
        fallback = None
        if best_lr_row is None:
            final_lr = reference_lr
            fallback = "no finite lr row; reused frozen reference lr"
        else:
            final_lr = float(best_lr_row["lr"])
        if gate != "calib":
            final_tau = None
            tau_selection = {"applicable": False}
            validation = best_lr_row
        else:
            tau_rows = []
            for tau in candidate_taus:
                if (tau == default_tau and best_lr_row is not None
                        and float(best_lr_row["lr"]) == final_lr):
                    runs = list(best_lr_row["runs"])
                else:
                    runs = [_tune_run(gate, seed, final_lr, tau, "tau")
                            for seed in tune_seeds]
                accs = [r["val_accuracy"] for r in runs]
                mean = sum(accs) / len(accs) if accs else float("nan")
                var = (sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)
                       if len(accs) > 1 else 0.0)
                verdicts = [r["curve_verdict"] for r in runs]
                tau_rows.append({
                    "lr": final_lr, "calib_tau_init": float(tau),
                    "val_accuracy_mean": mean,
                    "val_accuracy_std": math.sqrt(var) if math.isfinite(var)
                    else float("nan"),
                    "n_seeds": len(accs), "curve_verdicts": verdicts,
                    "curve_invalid_runs": sum(
                        v in CURVE_INVALID_VERDICTS for v in verdicts),
                    "runs": runs,
                })
            best_tau_row, tau_selection = _pick(
                tau_rows, "calib_tau_init", "calib:tau",
                preferred=default_tau)
            if best_tau_row is None:
                final_tau = default_tau
                fallback = ((fallback + "; ") if fallback else "") + \
                    "no finite tau row; reused default calib_tau_init"
                validation = best_lr_row
            else:
                final_tau = float(best_tau_row["calib_tau_init"])
                validation = best_tau_row
            tau_selection["rows"] = [
                {k: v for k, v in row.items() if k != "runs"}
                for row in tau_rows]
        reuse = (final_lr == reference_lr
                 and (gate != "calib" or final_tau == default_tau))
        reused_reference = bool(reuse and reference_runs
                                and reference_runs.get(gate))
        if reused_reference:
            final_runs = list(reference_runs[gate])
        else:
            final_runs = []
            for seed in settings.seeds:
                overrides = ({"calib_tau_init": final_tau}
                             if gate == "calib" else {})
                model, params, base_metrics, curve, model_cfg = \
                    _train_and_eval(
                        settings, seed, final_lr, eval_seed=TEST_SEED,
                        gate=gate, **overrides)
                long_metrics = eval_metrics(
                    model, lengths[-1], settings.vocab, settings.device,
                    batch=settings.eval_batch,
                    batches=settings.eval_batches, seed=TEST_SEED)
                rec = {
                    "experiment": "e2_per_variant_tuning",
                    "hypotheses": ["H2a", "H2d"],
                    "phase": "final", "variant": gate, "gate": gate,
                    "seed": int(seed), "lr": float(final_lr),
                    "calib_tau_init": (float(final_tau)
                                       if gate == "calib" else None),
                    "eval_seed": TEST_SEED, "params": int(params),
                    "eval_n": int(base_metrics["n"]),
                    "eval_n_longest": int(long_metrics["n"]),
                    "accuracy_at_base": float(base_metrics["accuracy"]),
                    "accuracy_at_longest": float(long_metrics["accuracy"]),
                    "accuracy_drop": float(base_metrics["accuracy"]
                                           - long_metrics["accuracy"]),
                    "model_cfg": model_cfg,
                    **_training_record(settings, final_lr),
                    **_run_context(settings),
                    **curve,
                }
                _write_result(
                    out_dir / "e2_per_variant_tuned_runs.jsonl", rec)
                final_runs.append(rec)
        by_arm[gate] = {
            "recipe": {"lr": final_lr, "calib_tau_init": final_tau},
            "reused_reference_runs": reused_reference,
            "fallback": fallback,
            "validation": (None if validation is None else {
                "accuracy_mean": validation["val_accuracy_mean"],
                "accuracy_std": validation["val_accuracy_std"],
                "n_seeds": validation["n_seeds"],
                "curve_invalid_runs": validation["curve_invalid_runs"],
            }),
            "lr_selection": lr_selection,
            "tau_selection": tau_selection,
            "lr_grid": [{k: v for k, v in row.items() if k != "runs"}
                        for row in lr_rows],
            "curve_invalid_runs": sum(
                r.get("curve_verdict") in CURVE_INVALID_VERDICTS
                for r in final_runs),
            "runs": final_runs,
        }

    longest_sesoi = float(HYPOTHESES["H2a"]["sesoi"]["accuracy_at_longest"])
    drop_sesoi = float(HYPOTHESES["H2a"]["sesoi"]["accuracy_drop"])

    def _tuned_state(ci, sesoi, direction):
        if ci.get("unmeasured"):
            return "unmeasured"
        lo, hi = ci.get("lo"), ci.get("hi")
        if direction == "+":
            if ci.get("complexity_justified"):
                return "calib above tuned baseline by SESOI"
            if ci.get("below_reference"):
                return "tuned baseline better"
        else:
            if _finite(hi) and float(hi) < -float(sesoi):
                return "calib drops less by SESOI"
            if _finite(lo) and float(lo) > float(sesoi):
                return "tuned baseline drops less"
        if ci.get("indistinguishable"):
            return "indistinguishable"
        return "inconclusive"

    contrasts = {}
    for baseline in ("none", "query", "null"):
        ref = (f"tuned {baseline} (lr={by_arm[baseline]['recipe']['lr']:g}"
               + ("" if by_arm[baseline]['recipe']['calib_tau_init'] is None
                  else f", tau={by_arm[baseline]['recipe']['calib_tau_init']:g}")
               + ")")
        row = {}
        for metric, sesoi, direction in (
            ("accuracy_at_longest", longest_sesoi, "+"),
            ("accuracy_drop", drop_sesoi, "-"),
        ):
            ci = _paired_benchmark(
                by_arm["calib"]["runs"], by_arm[baseline]["runs"], metric,
                sesoi, ref,
                comparison="e2_per_variant_tuned_benchmark",
                treatment=("each arm's own val-selected lr and calib_tau_init "
                           "for calib; selection at the training length"))
            ci["state"] = _tuned_state(ci, sesoi, direction)
            row[metric] = ci
        contrasts[f"calib - {baseline}"] = row

    def _is_good(row, metric):
        state = (row.get(metric) or {}).get("state")
        return state in ("calib above tuned baseline by SESOI",
                         "calib drops less by SESOI")

    def _is_ref_better(row, metric):
        state = (row.get(metric) or {}).get("state")
        return state in ("tuned baseline better",
                         "tuned baseline drops less")

    unmeasured = [name for name, row in contrasts.items()
                  if any(ci.get("unmeasured") for ci in row.values())]
    baseline_better = [name for name, row in contrasts.items()
                       if any(_is_ref_better(row, metric)
                              for metric in ("accuracy_at_longest",
                                             "accuracy_drop"))]
    robust = [name for name, row in contrasts.items()
              if all(_is_good(row, metric)
                     for metric in ("accuracy_at_longest", "accuracy_drop"))]
    if unmeasured:
        readout = ("not determinable for " + ", ".join(unmeasured)
                   + ": at least one side has no curve-valid paired seed")
    elif baseline_better:
        readout = ("NOT robust to per-variant tuning: "
                   + ", ".join(baseline_better)
                   + " is confidently better on at least one endpoint metric")
    elif len(robust) == len(contrasts) and contrasts:
        readout = ("robust: calib beats every tuned baseline on both the "
                   "longest-length level and the base-to-long drop by the "
                   "pre-registered SESOI")
    else:
        readout = ("not established: calib is not better than every tuned "
                   "baseline by the pre-registered SESOI on both endpoint "
                   "metrics")

    print(f"\n[e2] per-variant tuning (separate fairness accounting, "
          f"exploratory and outside the FDR family): each primary arm's own "
          f"recipe is selected on validation seed {VAL_SEED} at the training "
          f"length over lr={candidate_lrs} (calib also over "
          f"calib_tau_init={candidate_taus}); selection seeds={tune_seeds}.  "
          f"The controlled rows above freeze one recipe for all arms - this "
          f"table asks whether calib still degrades less when none/query/null "
          f"also get their own best shot:")
    print(f"{'arm':>6} {'lr':>10} {'tau':>7} {'val_acc':>17} {'val_n':>6} "
          f"{'curve_invalid':>13} {'test_at_long':>17} {'test_drop':>17} "
          f"{'reused':>7}")
    for gate in primary:
        row = by_arm[gate]
        recipe = row["recipe"]
        val = row["validation"] or {}
        tau = ("n/a" if recipe["calib_tau_init"] is None
               else f"{recipe['calib_tau_init']:g}")

        def _mcell(key):
            agg = _mean_std([r.get(key) for r in
                             _curve_valid_runs(row["runs"])])
            return (f"{agg['mean']:.4f}±{agg['std']:.4f}"
                    if _finite(agg["mean"]) else "n/a")

        val_s = (f"{val.get('accuracy_mean', float('nan')):.4f}±"
                 f"{val.get('accuracy_std', float('nan')):.4f}"
                 if _finite(val.get("accuracy_mean")) else "n/a")
        print(f"{gate:>6} {recipe['lr']:>10g} {tau:>7} {val_s:>17} "
              f"{int(val.get('n_seeds', 0)):>6} "
              f"{row['curve_invalid_runs']:>13} "
              f"{_mcell('accuracy_at_longest'):>17} "
              f"{_mcell('accuracy_drop'):>17} "
              f"{'yes' if row['reused_reference_runs'] else 'no':>7}")
        if row.get("fallback"):
            print(f"       fallback: {row['fallback']}")
    print(f"\n[e2] two accounting conventions side by side (endpoint "
          f"accuracy metrics; controlled = same frozen recipe, tuned = each "
          f"arm's own validation-selected recipe).  The pre-registered FDR "
          f"family is the controlled column; the tuned column is the "
          f"fairness robustness check and must not be averaged with it:")
    print(f"{'contrast':>18} {'metric':>22} {'controlled':>28} "
          f"{'tuned':>28} {'tuned state':>38}")
    controlled_columns = {}
    for name, row in contrasts.items():
        baseline = name.split(" - ", 1)[1]
        for metric in ("accuracy_at_longest", "accuracy_drop"):
            tuned = row.get(metric) or {}
            controlled = _paired_diff(
                (reference_runs or {}).get("calib") or [],
                (reference_runs or {}).get(baseline) or [], metric)
            controlled_columns.setdefault(name, {})[metric] = {
                "mean": controlled.get("mean"), "lo": controlled.get("lo"),
                "hi": controlled.get("hi"), "n": controlled.get("n"),
            }
            ctl_s = (f"{controlled.get('mean', float('nan')):+.4f} "
                     f"[{controlled.get('lo', float('nan')):+.4f},"
                     f"{controlled.get('hi', float('nan')):+.4f}]"
                     if _finite(controlled.get("mean")) else "n/a")
            print(f"{name:>18} {metric:>22} {ctl_s:>28} "
                  f"{_fmt_ci(tuned):>28} {tuned.get('state', ''):>38}")
    print(f"[e2] per-variant tuned endpoint conclusion: {readout}")
    payload = {
        "available": True,
        "selection_seed": VAL_SEED,
        "selection_length": int(settings.n_pairs),
        "selection_rule": "max mean validation accuracy among curve-valid rows",
        "tune_seeds": tune_seeds,
        "candidate_lrs": candidate_lrs,
        "calib_tau_grid": candidate_taus,
        "by_arm": {gate: {k: v for k, v in row.items() if k != "runs"}
                   for gate, row in by_arm.items()},
        "contrasts": contrasts,
        "controlled_contrasts": controlled_columns,
        "robust_baselines": robust,
        "baselines_better": baseline_better,
        "unmeasured": unmeasured,
        "readout": readout,
        "comparison": "e2_per_variant_tuned_benchmark",
        "note": ("exploratory fairness companion, not a controlled mechanism "
                 "contrast and not part of the pre-registered FDR family: "
                 "each arm's lr (and calib's calib_tau_init) is selected on "
                 "the validation seed at the training length, then the "
                 "selected recipe is evaluated at the longest sweep length; "
                 "read it next to the same-recipe controlled family, never "
                 "instead of it"),
    }
    _write_json(out_dir / "e2_per_variant_tuning.json", payload)
    return payload


def run_e2(settings: Settings, out_dir: pathlib.Path) -> None:
    _announce(("H2a", "H2b", "H2c", "H2d", "H2e", "H2f", "H2g"))
    # HB1/HB3/HB4/HB5 hard gate (see run_e1).
    verdict = _enforce_baseline_verdict(settings, out_dir, "e2")
    _write_manifest(settings, out_dir, "e2")
    _audit_data_splits(settings, out_dir, "e2", enforce=True)
    _enforce_reproducibility(settings, out_dir, "e2")
    _enforce_overfit_gate(settings, out_dir, "e2")
    lr = _resolve_lr(settings, out_dir, "e2")
    lengths = length_sweep_lengths(base=settings.n_pairs, factor=2, n=4,
                                   max_len=settings.max_len)
    # Per-length absolute reference: the parameter-free structural rule is
    # the task ceiling at *every* length.  Without it a length-related
    # accuracy drop cannot be attributed to the mechanism under test rather
    # than to a changing task.  Keys are drawn without replacement (see
    # data._sample_keys), so the rule is exact at every length;
    # unique_key_rate is still reported alongside as a data-contract check
    # (it must stay 1.0) rather than as a length-dependent ceiling caveat.
    rule_ceiling_by_length = {}
    for length in lengths:
        rule_ceiling_by_length[length] = rule_pair_lookup_stats(
            length, settings.vocab, settings.device,
            batch=settings.eval_batch, batches=settings.eval_batches,
            seed=TEST_SEED)
    _write_json(out_dir / "e2_rule_ceiling.json", {
        "experiment": "e2", "test_seed": TEST_SEED,
        "by_length": {str(L): stats
                      for L, stats in rule_ceiling_by_length.items()},
        "_provenance": _stamp(settings),
    })
    print(f"[e2] training at n_pairs={settings.n_pairs}, sweep over {lengths} "
          f"(test seed {TEST_SEED}, stats seed {STATS_SEED})")
    print(f"[e2] per-length rule ceiling (parameter-free structural rule; "
          f"exact only where keys are unique): "
          + ", ".join(f"L={L}: acc={rule_ceiling_by_length[L]['accuracy']:.4f} "
                      f"unique={rule_ceiling_by_length[L]['unique_key_rate']:.3f}"
                      for L in lengths))
    eval_seed, stats_seed = TEST_SEED, STATS_SEED
    runs = []
    examples = {}
    for label, gate, record_extra, model_overrides in _E2_VARIANTS:
        for seed in settings.seeds:
            model, params, base_metrics, curve, model_cfg = _train_and_eval(
                settings, seed, lr, eval_seed=eval_seed, gate=gate,
                **model_overrides)
            rec = {"experiment": "e2",
                   "hypotheses": ["H2a", "H2b", "H2c", "H2d", "H2e", "H2f",
                                   "H2g"],
                   "gate": gate, "variant": label, "seed": seed, "lr": lr,
                   "eval_seed": eval_seed, "stats_seed": stats_seed,
                   "params": params, "model_cfg": model_cfg,
                   "eval_n": int(base_metrics["n"]),
                   **{k: v for k, v in base_metrics.items() if k != "n"},
                   "metrics_by_length": {},
                   "accuracy_by_length": {}, "eval_n_by_length": {},
                   "balanced_accuracy_by_length": {},
                   "f1_macro_by_length": {}, "f1_weighted_by_length": {},
                   "entropy_by_length": {}, "entropy_norm_by_length": {},
                   "maxprob_by_length": {}, "gate_by_length": {},
                   "gate_vs_entropy_by_length": {},
                   "gate_vs_maxprob_by_length": {},
                   **_run_context(settings),
                   **record_extra,
                   **_training_record(settings, lr),
                   **curve}
            errors = None
            for length in lengths:
                metrics = (base_metrics if length == settings.n_pairs else
                           eval_metrics(model, length, settings.vocab,
                                        settings.device,
                                        batch=settings.eval_batch,
                                        batches=settings.eval_batches,
                                        seed=eval_seed))
                want_errors = length == lengths[-1]
                # Only the last length also collects the error-mining
                # record.
                stats, last_errors = collect_stats_and_errors(
                    model, length, settings.vocab, settings.device,
                    batch=settings.stats_batch,
                    batches=settings.stats_batches, seed=stats_seed,
                    max_cases=(settings.max_error_cases if want_errors
                               else None))
                if want_errors:
                    errors = last_errors
                rec["accuracy_by_length"][length] = metrics["accuracy"]
                rec["balanced_accuracy_by_length"][length] = (
                    metrics["recall_macro"])
                rec["f1_macro_by_length"][length] = metrics["f1_macro"]
                rec["f1_weighted_by_length"][length] = (
                    metrics["f1_weighted"])
                rec["eval_n_by_length"][length] = int(metrics["n"])
                rec["metrics_by_length"][length] = {
                    k: v for k, v in metrics.items() if k != "n"}
                rec["entropy_by_length"][length] = float(stats["entropy"].mean())
                rec["entropy_norm_by_length"][length] = (
                    rec["entropy_by_length"][length]
                    / math.log(max(2 * length + 2, 2)))
                rec["maxprob_by_length"][length] = float(stats["maxprob"].mean())
                if stats.get("gate") is not None:
                    rec["gate_by_length"][length] = float(stats["gate"].mean())
                # The gate hypothesis is about tracking the attention
                # distribution, not only about mean entropy drift.  Reuse
                # the statistics already collected at this length (no new
                # runs) to ask whether the within-layer association is
                # present and stable across the sweep, exactly as E1 reads
                # it at the training length.
                corr_by_len = gate_stat_correlations(stats)
                if corr_by_len:
                    rec["gate_vs_entropy_by_length"][length] = (
                        corr_by_len.get("gate_vs_entropy", float("nan")))
                    rec["gate_vs_maxprob_by_length"][length] = (
                        corr_by_len.get("gate_vs_maxprob", float("nan")))
            # Mechanism manipulation check for H2b (no new run/grid point):
            # the trained length and concentration coefficients say whether
            # the gate actually implements the explicit log-T / entropy
            # correction the hypothesis names.
            coefficients = _calibration_coefficients(model)
            rec["calib_coefficients"] = coefficients
            rec["concentration_signal"] = coefficients.get(
                "concentration_signal")
            # Top-level scalar coefficients: H2f is a one-sample contrast
            # across training seeds, so the per-run values have to be
            # readable by ``_one_sample_contrast`` (the nested per-layer
            # table stays for the manipulation breakdown).
            for coeff_key in ("a_mean", "c_mean", "c_effective_mean",
                              "scale_mean", "tau_effective_mean"):
                value = coefficients.get(coeff_key)
                if _finite(value):
                    rec[coeff_key] = float(value)
            base_acc = rec["accuracy_by_length"][settings.n_pairs]
            long_acc = rec["accuracy_by_length"][lengths[-1]]
            rec["accuracy_at_base"] = base_acc
            rec["accuracy_at_longest"] = long_acc
            rec["accuracy_drop"] = base_acc - long_acc
            # Degradation *shape*, not only the endpoint: fit accuracy on
            # log(T) across the measured sweep.  A mid-sweep reversal can
            # leave the base-to-longest drop unchanged while contradicting
            # "degrades more slowly"; this readout uses data already
            # collected, no extra grid point.
            rec["accuracy_slope_log_len"] = _linear_slope(
                [math.log(max(2 * L + 2, 2)) for L in lengths],
                [rec["accuracy_by_length"][L] for L in lengths])
            base_bal = rec["balanced_accuracy_by_length"][settings.n_pairs]
            long_bal = rec["balanced_accuracy_by_length"][lengths[-1]]
            rec["balanced_accuracy_at_base"] = base_bal
            rec["balanced_accuracy_at_longest"] = long_bal
            rec["balanced_accuracy_drop"] = base_bal - long_bal
            rec["rule_ceiling_by_length"] = {
                L: rule_ceiling_by_length[L]["accuracy"] for L in lengths}
            rec["rule_unique_key_rate_by_length"] = {
                L: rule_ceiling_by_length[L]["unique_key_rate"]
                for L in lengths}
            rec["ceiling_gap_by_length"] = {
                L: rule_ceiling_by_length[L]["accuracy"]
                - rec["accuracy_by_length"][L] for L in lengths}
            rec["ceiling_gap_at_longest"] = rec["ceiling_gap_by_length"][lengths[-1]]
            rec["entropy_drift"] = (rec["entropy_by_length"][lengths[-1]]
                                    - rec["entropy_by_length"][settings.n_pairs])
            rec["entropy_norm_drift"] = (
                rec["entropy_norm_by_length"][lengths[-1]]
                - rec["entropy_norm_by_length"][settings.n_pairs])
            rec.update({f"err_{k}": v for k, v in errors["summary"].items()})
            for case in errors["cases"]:
                _write_result(out_dir / "e2_error_cases.jsonl",
                              {"experiment": "e2", **_stamp(settings),
                               "gate": gate, "variant": label, "seed": seed,
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
    ablation_coverage = _report_ablation_coverage(
        out_dir, "e2", runs,
        cells=_CALIB_FACTORIAL_CELLS + (_CALIB_RAW_ENT_CELL,),
        planned_seeds=settings.seeds,
        design=("calib_use_len x calib_use_ent (2x2) + raw-entropy "
                "normalisation ablation (H2g)"))
    design_coverage = _report_design_coverage(
        out_dir, "e2", runs, "variant",
        lambda r: r.get("variant"),
        tuple(label for label, *_ in _E2_VARIANTS),
        planned_seeds=settings.seeds)
    print(f"\n[e2] accuracy by n_pairs (mean +- std over seeds; the "
          f"calib_len_only / calib_ent_only / calib_no_terms arms complete "
          f"the H2c calibration-term 2x2 with calib - H2c's directional rows "
          f"use only the single-term ablations, while calib_no_terms (both "
          f"terms off) enters the factorial main effects/interaction; "
          f"calib_raw_ent is the H2g normalisation ablation - identical "
          f"architecture/parameters, raw entropy instead of H/log T; "
          f"gap@long = mean gap to the per-length rule ceiling at the "
          f"longest length, the absolute reference for H2a's drop):")
    # ``drop`` / ``gap@long`` / ``entropy_drift`` are the pre-registered H2a
    # readouts, so they carry the same mean +- std treatment as the per-length
    # cells: an effect size printed without its seed-to-seed spread invites a
    # 0.2%-effect-vs-0.5%-std reading that the contrast layer would refuse.
    line = f"{'arm':>16}" + "".join(f"{L:>12}" for L in lengths) + \
           f"{'drop':>17} {'gap@long':>17} {'entropy_drift':>17}"
    print(line)
    for label, *_ in _E2_VARIANTS:
        cells = []
        for length in lengths:
            agg = _mean_std([r["accuracy_by_length"][length]
                             for r in by_variant[label]])
            cells.append(f"{agg['mean']:.3f}±{agg['std']:.3f}")
        drop = _mean_std([r["accuracy_drop"] for r in by_variant[label]])
        gap = _mean_std([r["ceiling_gap_at_longest"]
                         for r in by_variant[label]])
        drift = _mean_std([r["entropy_drift"] for r in by_variant[label]])

        def dcell(agg):
            if not _finite(agg["mean"]):
                return "n/a"
            return f"{agg['mean']:.3f}±{agg['std']:.3f}"

        print(f"{label:>16}" + "".join(f"{c:>12}" for c in cells)
              + f"{dcell(drop):>17} {dcell(gap):>17} "
              + f"{dcell(drift):>17}")
    ceil_cells = [f"{rule_ceiling_by_length[L]['accuracy']:.3f}"
                  for L in lengths]
    print(f"{'rule ceiling':>16}" + "".join(f"{c:>12}" for c in ceil_cells)
          + f"{'-':>17} {'(gap@long)':>17} {'-':>17}")

    print(f"\n[e2] whole-sweep degradation slope (least-squares accuracy per "
          f"unit log(T) over all {len(lengths)} measured lengths, "
          f"mean±std over seeds; more negative = faster degradation).  This "
          f"is the shape readout behind H2a/H2d's 'degrades more slowly' "
          f"claim: the endpoint drop alone can hide a mid-sweep reversal:")
    print(f"{'arm':>16} {'slope/logT':>17} {'seeds_slope<0':>14}")
    for label, *_ in _E2_VARIANTS:
        agg = _mean_std([
            r.get("accuracy_slope_log_len") for r in by_variant[label]])
        neg = sum(1 for r in by_variant[label]
                  if _finite(r.get("accuracy_slope_log_len"))
                  and float(r["accuracy_slope_log_len"]) < 0.0)
        slope_s = (f"{agg['mean']:+.4f}±{agg['std']:.4f}"
                   if _finite(agg["mean"]) else "n/a")
        print(f"{label:>16} {slope_s:>17} {neg:>14}")

    # Balanced accuracy (macro recall) is the distribution-robust companion
    # to the headline accuracy at every sweep length.  The values agree
    # under a uniform stream; a divergence means the length reading depends
    # on class frequency and must be reported as such.
    print(f"\n[e2] balanced accuracy (macro recall) and macro-F1 by n_pairs "
          f"(mean +- std over seeds; companions to the accuracy table - "
          f"divergence at a length means that length's reading depends on "
          f"answer-class frequency; macro-F1 averages F1 over the classes "
          f"present in the target, f1_micro would equal accuracy):")
    print(f"{'arm':>16} {'metric':>8}" + "".join(f"{L:>12}" for L in lengths)
          + f"{'drop':>17}")
    for label, *_ in _E2_VARIANTS:
        for metric, key in (("bal_acc", "balanced_accuracy_by_length"),
                            ("f1_macro", "f1_macro_by_length")):
            cells = []
            for length in lengths:
                agg = _mean_std([r[key][length]
                                 for r in by_variant[label]])
                cells.append(f"{agg['mean']:.3f}±{agg['std']:.3f}")
            drop = _mean_std(
                [r[key][settings.n_pairs] - r[key][lengths[-1]]
                 for r in by_variant[label]])
            drop_s = (f"{drop['mean']:.3f}±{drop['std']:.3f}"
                      if _finite(drop["mean"]) else "n/a")
            print(f"{label:>16} {metric:>8}"
                  + "".join(f"{c:>12}" for c in cells)
                  + f"{drop_s:>17}")

    print(f"\n[e2] learned calibration coefficients (mechanism manipulation "
          f"check for H2b; read from the trained gates, no extra grid point). "
          f"The gate computes sigmoid((W_g x - a*log T - c*scale*Hbar)/tau): "
          f"a>0 is the explicit length correction, c*scale>0 is the "
          f"concentration correction.  'a+'/'c+' count layers with a positive "
          f"coefficient; an ablated arm that should not have a term must "
          f"report n/a there.  Mean over seeds and layers, ± seed std:")
    print(f"{'arm':>16} {'signal':>10} {'a':>16} {'c_eff':>16} "
          f"{'tau_eff':>16} {'a+':>8} {'c+':>8} {'layers':>8}")
    for label, *_ in _E2_VARIANTS:
        if not label.startswith("calib"):
            continue
        rs = by_variant[label]
        coeffs = [r.get("calib_coefficients") or {} for r in rs]
        coeffs = [c for c in coeffs if c.get("applicable")]

        def ccell(key, layers_key=None):
            if not coeffs:
                return "n/a"
            if layers_key:
                vals = [c.get(layers_key) for c in coeffs]
                denom_key = ("a_layers" if layers_key.startswith("a_")
                             else "c_layers")
                total = sum(c.get(denom_key, 0) for c in coeffs)
                return f"{sum(vals)}/{total}"
            agg = _mean_std([c.get(key) for c in coeffs])
            if not _finite(agg["mean"]):
                return "n/a"
            return f"{agg['mean']:+.4f}±{agg['std']:.4f}"

        print(f"{label:>16} "
              f"{(coeffs[0].get('concentration_signal') if coeffs else 'n/a'):>10} "
              f"{ccell('a_mean'):>16} "
              f"{ccell('c_effective_mean'):>16} "
              f"{ccell('tau_effective_mean'):>16} "
              f"{ccell(None, 'a_positive_layers'):>8} "
              f"{ccell(None, 'c_effective_positive_layers'):>8} "
              f"{(coeffs[0].get('n_layers') if coeffs else 0):>8}")
    print("  (H2c manipulation check: only the arms with the length term "
          "on should show a finite 'a'; only the arms with the concentration "
          "term on should show a finite 'c_eff'.  A full-calib run with "
          "a<=0 or c_eff<=0 is not implementing the correction H2a/H2b "
          "name, whatever its endpoint accuracy.  'signal' names what c_eff "
          "multiplies: H/logT for the registered gate, H for the H2g "
          "raw-entropy ablation - the two c_eff values are therefore not in "
          "the same units and must not be compared directly.)")

    print(f"\n[e2] gate/attention-distribution association across the length "
          f"sweep (within-layer Pearson, Fisher-z mean over layers, mean±std "
          f"over seeds; computed from the same per-length statistics, no new "
          f"runs).  The length-calibration claim predicts that the calib "
          f"gate keeps reading entropy / max-prob at longer lengths; a "
          f"query-only gate has no such pathway, so its association should "
          f"stay weaker and flatter.  'drift' = longest - training length "
          f"(positive = association strengthens with length):")
    print(f"{'arm':>16} {'metric':>12}" + "".join(f"{L:>17}" for L in lengths)
          + f"{'drift':>17}")
    gate_corr_by_length = {}
    for label, *_ in _E2_VARIANTS:
        if not label.startswith("calib") and label != "query":
            continue
        for metric in ("gate_vs_entropy_by_length",
                       "gate_vs_maxprob_by_length"):
            cells = []
            per_length_agg = {}
            for length in lengths:
                agg = _mean_std([
                    (r.get(metric) or {}).get(length)
                    for r in by_variant[label]])
                per_length_agg[str(length)] = agg
                cells.append(f"{agg['mean']:+.3f}±{agg['std']:.3f}"
                             if _finite(agg["mean"]) else "n/a")
            drift = _mean_std([
                ((r.get(metric) or {}).get(lengths[-1])
                 - (r.get(metric) or {}).get(settings.n_pairs))
                for r in by_variant[label]])
            drift_s = (f"{drift['mean']:+.3f}±{drift['std']:.3f}"
                       if _finite(drift["mean"]) else "n/a")
            label_metric = ("r_entropy" if metric.startswith("gate_vs_entropy")
                            else "r_maxprob")
            print(f"{label:>16} {label_metric:>12}"
                  + "".join(f"{c:>17}" for c in cells)
                  + f"{drift_s:>17}")
            gate_corr_by_length[f"{label}:{label_metric}"] = {
                "by_length": per_length_agg, "drift": drift,
            }
    print("  (this is a mechanism readout, not a second decision rule: "
          "H2a/H2b/H2f above carry the pre-registered decisions; the table "
          "checks that the calibrated arm's distribution pathway is actually "
          "used at the extrapolated lengths rather than only at the training "
          "length.)")

    # Decompose the trained gate into the two additive calibration terms at
    # the training and longest lengths.  This is derived from data already
    # recorded (learned a / c_eff and the per-length normalised entropy), so
    # it adds no runs or grid points: it asks *which* term moves with length,
    # the explicit mechanism H2b/H2c is about.
    print(f"\n[e2] implied calibration-term contributions at the base and "
          f"longest sweep lengths (a*log T and c_eff*S, using the trained "
          f"coefficients; mean±std over seeds).  S is the concentration "
          f"signal the arm actually reads: H/log T for the registered gate, "
          f"raw H for the H2g raw-entropy arm.  A length term that is on "
          f"should show a non-zero drift a*(log T_long - log T_base); a "
          f"stable concentration term drifts only with its own signal:")
    print(f"{'arm':>16} {'length@base':>17} {'length@long':>17} "
          f"{'length_drift':>17} {'conc@base':>17} {'conc@long':>17} "
          f"{'conc_drift':>17}")
    term_contributions = {}
    for label, *_ in _E2_VARIANTS:
        if not label.startswith("calib"):
            continue
        per_run = []
        for r in by_variant[label]:
            coeff = r.get("calib_coefficients") or {}
            if not coeff.get("applicable"):
                continue
            a = coeff.get("a_mean", float("nan"))
            c_eff = coeff.get("c_effective_mean", float("nan"))
            base_L, long_L = settings.n_pairs, lengths[-1]
            log_base = math.log(max(2 * base_L + 2, 2))
            log_long = math.log(max(2 * long_L + 2, 2))
            # The H2g arm reads raw row-mean entropy, not H/log T.  Using
            # the normalised series for it would understate the trained
            # concentration contribution by log T (and describe a gate that
            # was not the one trained), so select the signal per arm from
            # the resolved coefficient readout.
            if coeff.get("concentration_signal") == "H":
                h_base = r["entropy_by_length"][base_L]
                h_long = r["entropy_by_length"][long_L]
            else:
                h_base = r["entropy_norm_by_length"][base_L]
                h_long = r["entropy_norm_by_length"][long_L]
            per_run.append({
                "length_base": a * log_base if _finite(a) else float("nan"),
                "length_long": a * log_long if _finite(a) else float("nan"),
                "length_drift": (a * (log_long - log_base)
                                 if _finite(a) else float("nan")),
                "conc_base": (c_eff * h_base if _finite(c_eff)
                              else float("nan")),
                "conc_long": (c_eff * h_long if _finite(c_eff)
                              else float("nan")),
                "conc_drift": (c_eff * (h_long - h_base) if _finite(c_eff)
                               else float("nan")),
            })
        keys = ("length_base", "length_long", "length_drift",
                "conc_base", "conc_long", "conc_drift")
        agg = {key: _mean_std([row[key] for row in per_run])
               for key in keys}
        term_contributions[label] = agg

        def tcell(key):
            m = agg[key]
            if not math.isfinite(m["mean"]):
                return "n/a"
            return f"{m['mean']:+.4f}±{m['std']:.4f}"

        print(f"{label:>16} {tcell('length_base'):>17} "
              f"{tcell('length_long'):>17} {tcell('length_drift'):>17} "
              f"{tcell('conc_base'):>17} {tcell('conc_long'):>17} "
              f"{tcell('conc_drift'):>17}")

    print(f"\n[e2] training-curve health (curve_ok = curve-valid runs / "
          f"all runs; late = U-shaped val curves; tail_train = relative "
          f"change over the last third - falling under a plateau verdict "
          f"means the step count is the limit; vse = last validation "
          f"point's batch-sampling SE):")
    print(f"{'arm':>16} {'curve_ok':>8} {'late':>5} {'tail_train':>11} "
          f"{'vse':>8} {'curve':>10}")
    for label, *_ in _E2_VARIANTS:
        _, invalid = _curve_health(by_variant[label])
        verdicts = sorted({r["curve_verdict"] for r in by_variant[label]})
        late = sum(1 for r in by_variant[label] if r.get("late_overfit"))
        tail = _mean_std([r.get("late_train_delta")
                          for r in by_variant[label]])
        vse = _mean_std([r.get("val_loss_se_last")
                         for r in by_variant[label]])
        tail_s = (f"{tail['mean']:+.3f}" if _finite(tail["mean"]) else "n/a")
        vse_s = f"{vse['mean']:.4f}" if _finite(vse["mean"]) else "n/a"
        print(f"{label:>16} "
              f"{len(by_variant[label]) - invalid}/{len(by_variant[label])} "
              f"{late:>5} {tail_s:>11} {vse_s:>8} "
              f"{'/'.join(verdicts):>10}")

    print(f"\n[e2] bad cases at the longest length (n_pairs={lengths[-1]}; "
          f"stats seed {STATS_SEED}, batch {settings.stats_batch}x"
          f"{settings.stats_batches}; raw cases in e2_error_cases.jsonl).  "
          f"retrieval = attention mass < 0.5 on the correct key in every "
          f"block (the model never found it); lost = found mid-stack but "
          f"not in the last block; readout = key found in the last block "
          f"but the wrong value read out - the split that tells apart "
          f"'cannot find' from 'found then lost' from 'cannot use':")
    print(f"{'arm':>16} {'err_rate':>9} {'label':>7} {'struct':>7} "
          f"{'retrieval':>10} {'lost':>7} {'readout':>8} {'key_miss':>9} "
          f"{'mass_key_err':>13}")
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
              f"{ecell('err_taxonomy_found_then_lost'):>7} "
              f"{ecell('err_taxonomy_readout_wrong'):>8} "
              f"{ecell('err_errors_key_missed_share'):>9} "
              f"{ecell('err_attn_mass_on_key_error'):>13}")
    _print_structural_decomposition(
        [(label, by_variant[label]) for label, *_ in _E2_VARIANTS],
        title="[e2]")
    _print_conf_by_error_type([(label, by_variant[label])
                               for label, *_ in _E2_VARIANTS], title="[e2]")
    print("  most confident wrong samples per arm at the longest length:")
    for label, *_ in _E2_VARIANTS:
        for case in examples.get(label, (0, []))[1]:
            print(f"  [{label}] {_format_case(case)}")

    # HB0 competence prerequisite: a length-generalisation contrast is only
    # readable as a mechanism statement when at least one arm demonstrably
    # learned the task at the training length.
    competence = _competence_check(
        [(label, rs) for label, rs in by_variant.items()],
        lambda r: (int(round(r["accuracy_at_base"] * r["eval_n"])),
                   int(r["eval_n"])),
        1.0 / (settings.vocab - 2), what="E2 arm (training length)")
    _write_json(out_dir / "e2_competence.json",
                {**competence, **_run_context(settings)})
    if competence["passed"]:
        print(f"[e2] HB0 competence: ok (best arm {competence['best_arm']} "
              f"has seed-clustered 95% lower bound above chance "
              f"{competence['chance']:.4f} at the training length)")
    else:
        print(f"[e2] HB0 competence: FAILED - no arm's seed-clustered 95% "
              f"lower bound is above chance {competence['chance']:.4f} at "
              f"the training length; the grid did not demonstrably learn the "
              f"task at this step count, so no contrast below may be read as a "
              f"mechanism statement")
    eval_balance = verdict.get("eval_answer_balance") or {}
    e2_balance = eval_balance.get("e2_length_sweep") or {}
    balance_ok = e2_balance.get("endpoint_uniform")
    if balance_ok is None:
        balance_ok = eval_balance.get("all_uniform")
    print("\n[e2] pre-registered contrasts:")
    clog = ContrastLog("e2", hypotheses=("H2a", "H2b", "H2c", "H2d", "H2e",
                                       "H2f", "H2g"),
                       min_seeds=settings.min_seeds_for_claim,
                       competence=competence,
                       coverage=(ablation_coverage, design_coverage),
                       balance_ok=balance_ok)
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
    clog.print("calib - none", "H2a", "accuracy_slope_log_len",
               _paired_diff(by_variant["calib"], by_variant["none"],
                            "accuracy_slope_log_len"),
               warn=_curve_warn(("calib", by_variant["calib"]),
                                ("none", by_variant["none"])))
    # H2d: the same length-robustness read against the closest simpler
    # gate.  ``query`` shares the attention/gating architecture and differs
    # only in the calibration terms, so a calib-vs-query gain is the one
    # that can be attributed to length/concentration calibration rather
    # than to "having any learned gate".
    clog.print("calib - query", "H2d", "accuracy_at_longest",
               _paired_diff(by_variant["calib"], by_variant["query"],
                            "accuracy_at_longest"),
               warn=_curve_warn(("calib", by_variant["calib"]),
                                ("query", by_variant["query"])))
    clog.print("calib - query", "H2d", "accuracy_drop",
               _paired_diff(by_variant["calib"], by_variant["query"],
                            "accuracy_drop"),
               warn=_curve_warn(("calib", by_variant["calib"]),
                                ("query", by_variant["query"])))
    clog.print("calib - query", "H2d", "accuracy_slope_log_len",
               _paired_diff(by_variant["calib"], by_variant["query"],
                            "accuracy_slope_log_len"),
               warn=_curve_warn(("calib", by_variant["calib"]),
                                ("query", by_variant["query"])))
    clog.print("calib raw entropy drift vs 0", "H2b", "entropy_drift",
               _one_sample_contrast(calib_runs, "entropy_drift"),
               warn=calib_warn)
    # The H/log(T) drift and the gate drift are *null predictions*: the
    # decision rule is CI-within-bound (bootstrap TOST), not "CI excludes
    # 0" - support means "indistinguishable from length-stable within the
    # pre-registered bound", and a large drift must refute, not support,
    # the prediction.
    clog.print_equivalence_ci(
        "calib normalised entropy drift vs 0", "H2b", "entropy_norm_drift",
        _one_sample_contrast(
            calib_runs, "entropy_norm_drift",
            bound=HYPOTHESES["H2b"]["sesoi"]["entropy_norm_drift"]),
        warn=calib_warn)
    if calib_runs[0]["gate_by_length"]:
        missing_gate = [r for r in calib_runs if r.get("seed") is None]
        gate_by_seed = {r["seed"]: r for r in calib_runs
                        if r.get("seed") is not None}
        gate_used, gate_drifts = [], []
        for r in calib_runs:
            if r.get("seed") is None:
                continue
            value = (r["gate_by_length"][lengths[-1]]
                     - r["gate_by_length"][settings.n_pairs])
            if _finite(value):
                gate_used.append(r["seed"])
                gate_drifts.append(float(value))
        gate_ci = equivalence_test(
            gate_drifts, HYPOTHESES["H2b"]["sesoi"]["gate_drift"])
        gate_undefined = sorted(set(gate_by_seed) - set(gate_used),
                                key=_seed_order)
        gate_ci["diffs"] = list(gate_drifts)
        gate_ci["seeds_used"] = gate_used
        gate_ci["seeds_dropped"] = (gate_undefined
                                    + [None] * len(missing_gate))
        gate_ci["seeds_unpaired"] = [None] * len(missing_gate)
        gate_ci["seeds_undefined"] = gate_undefined
        gate_ci["curve_invalid_seeds"] = _curve_invalid_seeds_for(
            gate_used, (gate_by_seed,))
        clog.print_equivalence_ci("calib gate drift vs 0", "H2b",
                                  "gate_drift", gate_ci, warn=calib_warn)
    # H2f: the mechanism is an *explicit* additive correction, so the
    # trained coefficients themselves must point the right way.  This is a
    # one-sample test across seeds (no new grid point): a full-calib arm
    # with a<=0 or c*scale<=0 cannot be said to implement the length /
    # concentration correction, whatever the endpoint accuracy says.
    print(f"\n[e2] H2f explicit-correction manipulation check (one-sample "
          f"across training seeds): the gate is "
          f"sigmoid((W_g x - a*log T - c*scale*Hbar)/tau), so a>0 and "
          f"c*scale>0 are the learned coefficients that implement the named "
          f"correction.  A full calib arm with a non-positive coefficient is "
          f"a capacity/accuracy effect, not the claimed explicit "
          f"calibration:")
    for metric, label in (
        ("a_mean", "calib length coefficient a"),
        ("c_effective_mean", "calib concentration coefficient c*scale"),
    ):
        clog.print(f"{label} vs 0", "H2f", metric,
                   _one_sample_contrast(calib_runs, metric),
                   warn=calib_warn)
    # H2g: ablate the log-T normalisation of the concentration term.  The
    # raw-entropy arm keeps the same architecture, parameters, length term
    # and kernel; only Hbar = H/log T becomes H.  At the training length a
    # constant factor is absorbed by calib_c, so the endpoint/drop contrast
    # is where the normalisation can show up.
    raw_runs = by_variant[_CALIB_RAW_ENT_CELL]
    raw_warn = _curve_warn(("calib", calib_runs),
                           (_CALIB_RAW_ENT_CELL, raw_runs))
    full_norm = {(r.get("model_cfg") or {}).get("calib_ent_norm")
                 for r in calib_runs}
    raw_norm = {(r.get("model_cfg") or {}).get("calib_ent_norm")
                for r in raw_runs}
    print(f"\n[e2] H2g log-T normalisation ablation (single factor: both "
          f"additive terms on, identical architecture/parameter count/"
          f"computation, but the concentration term reads raw entropy "
          f"instead of Hbar=H/log T).  Resolved flags: full calib "
          f"calib_ent_norm={sorted(full_norm, key=str)}, "
          f"{_CALIB_RAW_ENT_CELL} calib_ent_norm={sorted(raw_norm, key=str)}. "
          f"A gain for the normalised arm at the longest length / a smaller "
          f"drop is evidence that the normalisation - not merely the "
          f"presence of a concentration term - carries length robustness:")
    if full_norm != {True} or raw_norm != {False}:
        print(f"      WARNING: the resolved normalisation flags do not match "
              f"the declared arms (full={full_norm}, raw={raw_norm}); the "
              f"H2g contrasts are mis-wired and must not be read")
    # Descriptive manipulation check: H2g compares two *implemented*
    # corrections, so the raw arm must still use its concentration term
    # (otherwise the contrast would silently become full-vs-no-term).  This
    # is a readout, not a second decision rule; H2f owns the full arm's
    # coefficient test.
    raw_c = _mean_std([r.get("c_effective_mean") for r in raw_runs])
    raw_c_s = (f"{raw_c['mean']:+.4f}±{raw_c['std']:.4f}"
               if _finite(raw_c["mean"]) else "n/a")
    print(f"      descriptive manipulation check: {_CALIB_RAW_ENT_CELL} "
          f"c*scale={raw_c_s} (coefficient on raw H; a non-positive value "
          f"would mean this arm did not use its concentration term, in which "
          f"case the contrast is not a normalisation-only comparison)")
    for metric in ("accuracy_at_longest", "accuracy_drop"):
        clog.print("calib - calib_raw_ent", "H2g", metric,
                   _paired_diff(calib_runs, raw_runs, metric),
                   warn=raw_warn)
    print(f"\n[e2] H2c length-term ablation contrast (the single-factor "
          f"arm removes only the length term from the full gate at "
          f"identical recipe, leaving the concentration term on; the "
          f"calib_no_terms cell disables both terms at once, so it is a "
          f"two-factor manipulation and is read only through the registered "
          f"2x2 main effects/interaction below, not as a length-term "
          f"decision row):")
    for arm in ("calib_ent_only",):
        rs = by_variant[arm]
        warn = _curve_warn(("calib", calib_runs), (arm, rs))
        clog.print(f"calib - {arm}", "H2c", "accuracy_at_longest",
                   _paired_diff(calib_runs, rs, "accuracy_at_longest"),
                   warn=warn)
        clog.print(f"calib - {arm}", "H2c", "accuracy_drop",
                   _paired_diff(calib_runs, rs, "accuracy_drop"),
                   warn=warn)
    # Reverse check: H2c predicts the length-only arm *stays close* to the
    # full arm.  That is a null prediction, so a non-significant
    # superiority test would not support it; use the pre-registered TOST
    # bound on both the long-length level and the drop.  The contrast keeps
    # the same pairing and controlled audit as the directional rows.
    reverse_warn = _curve_warn(
        ("calib", calib_runs), ("calib_len_only", by_variant["calib_len_only"]))
    for metric, bound_key in (
        ("accuracy_at_longest", "calib_minus_len_only_accuracy_at_longest"),
        ("accuracy_drop", "calib_minus_len_only_accuracy_drop"),
    ):
        ci = _paired_equivalence(
            calib_runs, by_variant["calib_len_only"], metric,
            HYPOTHESES["H2c"]["sesoi"][bound_key])
        clog.print_equivalence_ci(
            "calib - calib_len_only (reverse check)", "H2c", metric, ci,
            warn=reverse_warn)
    # Full-factorial companion: the length term's effect averaged over both
    # concentration-term levels, on the long-length level and on the
    # base-to-long drop.  The LOO rows above are conditional on the
    # concentration term being on; this row answers whether the length term
    # carries an average contribution in the complete 2x2.
    e2_factorial_cells = {"full": calib_runs}
    for cell in _CALIB_FACTORIAL_CELLS:
        if cell != "full":
            e2_factorial_cells[cell] = by_variant[cell]
    factorial_warn = _curve_warn(
        *[(cell, e2_factorial_cells[cell])
          for cell in _CALIB_FACTORIAL_CELLS])
    print(f"\n[e2] H2c factorial main effect (length term): the average of "
          f"full-vs-ent_only and len_only-vs-no_terms, i.e. the length "
          f"term's contribution averaged over both concentration-term "
          f"levels:")
    for metric in ("accuracy_at_longest", "accuracy_drop"):
        ci = _factorial_main_effect(e2_factorial_cells, metric, "length")
        clog.print("length main effect (2x2)", "H2c", metric, ci,
                   warn=factorial_warn)
    print(f"\n[e2] H2e factorial companions: the concentration term's main "
          f"effect (average of full-vs-len_only and ent_only-vs-no_terms, "
          f"i.e. averaged over both length-term levels) and the two-term "
          f"interaction (length effect under concentration on minus under "
          f"concentration off).  Together with the length main effect these "
          f"read the complete 2x2, so neither term nor their additivity is "
          f"left unmeasured:")
    for metric in ("accuracy_at_longest", "accuracy_drop"):
        ci = _factorial_main_effect(e2_factorial_cells, metric,
                                    "concentration")
        clog.print("concentration main effect (2x2)", "H2e", metric, ci,
                   direction_key=f"concentration_{metric}",
                   warn=factorial_warn)
    for metric in ("accuracy_at_longest", "accuracy_drop"):
        ci = _paired_did(
            e2_factorial_cells["full"], e2_factorial_cells["calib_ent_only"],
            e2_factorial_cells["calib_len_only"],
            e2_factorial_cells["calib_no_terms"], metric)
        clog.print("length x concentration interaction (2x2)", "H2e", metric,
                   ci, warn=factorial_warn)
    e2_contrast_payload = clog.finalize(out_dir, q=settings.fdr_q)
    h2f_rows = [row for row in (e2_contrast_payload.get("rows") or [])
                if row.get("hypothesis") == "H2f"]
    h2f_evaluable = bool(h2f_rows) and all(
        row.get("claim_eligible", True) for row in h2f_rows)
    h2f_supported = bool(h2f_rows) and all(
        row.get("supports_hypothesis") for row in h2f_rows)
    if not h2f_supported:
        if not h2f_evaluable:
            print(f"[e2] NOTE: H2f is not evaluable on this run (the "
                  f"coefficient contrasts are claim-ineligible: HB0 "
                  f"competence, seed floor, coverage or curve filtering), "
                  f"so neither a positive nor a non-positive coefficient "
                  f"read is a mechanism conclusion here.")
        else:
            print(f"[e2] WARNING: H2f not supported - the full calib arm did "
                  f"not show both learned corrections (a>0 and c*scale>0) "
                  f"with FDR-significant, practically-sized coefficients.  "
                  f"The H2a/H2b endpoint and entropy-drift rows remain their "
                  f"own tests, but the 'explicit correction' mechanism "
                  f"reading is not established by the trained coefficients.")
    learned_correction_check = {
        "hypothesis": "H2f",
        "supported": h2f_supported,
        "evaluable": h2f_evaluable,
        "rows": [{
            "metric": row.get("metric"), "mean": row.get("mean"),
            "lo": row.get("lo"), "hi": row.get("hi"),
            "p_value": row.get("p_value"), "p_adj": row.get("p_adj"),
            "sesoi": row.get("sesoi"), "n": row.get("n"),
            "seeds_used": row.get("seeds_used"),
            "supports_hypothesis": row.get("supports_hypothesis"),
            "verdict": ("supported" if row.get("supports_hypothesis")
                        else "not supported"),
        } for row in h2f_rows],
        "by_arm": {
            label: {
                "concentration_signal": next(
                    (r.get("concentration_signal")
                     for r in by_variant[label]
                     if r.get("concentration_signal")), None),
                **{
                    key: _mean_std([r.get(key) for r in by_variant[label]])
                    for key in ("a_mean", "c_effective_mean",
                                "tau_effective_mean")
                },
            }
            for label, *_ in _E2_VARIANTS if label.startswith("calib")
        },
        "note": ("mechanism-manipulation check on the trained gate: a "
                 "positive a is the explicit log-T correction and a positive "
                 "c*scale is the concentration correction; H2f is tested in "
                 "the same E2 FDR family and is not replaced by endpoint "
                 "accuracy"),
    }
    length_contrasts = _e2_per_length_contrasts(
        settings, out_dir, by_variant, lengths)
    endpoint_companions = _e2_endpoint_class_companions(by_variant, lengths)
    per_variant_tuning = _e2_per_variant_tuning(
        settings, out_dir, lr, by_variant, lengths)
    length_efficiency = _length_efficiency_companion(
        settings, out_dir, "e2",
        [(label, by_variant[label]) for label, *_ in _E2_VARIANTS],
        lengths, reference="none")
    e2_summary = {
        label: {str(L): _mean_std([r["accuracy_by_length"][L]
                                   for r in by_variant[label]])
                for L in lengths}
        for label, *_ in _E2_VARIANTS}
    e2_summary["_accuracy_slope_log_len"] = {
        label: _mean_std([r.get("accuracy_slope_log_len")
                          for r in by_variant[label]])
        for label, *_ in _E2_VARIANTS}
    e2_summary["_balanced_accuracy"] = {
        label: {str(L): _mean_std(
            [r["balanced_accuracy_by_length"][L] for r in by_variant[label]])
            for L in lengths}
        for label, *_ in _E2_VARIANTS}
    e2_summary["_f1_macro"] = {
        label: {str(L): _mean_std(
            [r["f1_macro_by_length"][L] for r in by_variant[label]])
            for L in lengths}
        for label, *_ in _E2_VARIANTS}
    e2_summary["_f1_weighted"] = {
        label: {str(L): _mean_std(
            [r["f1_weighted_by_length"][L] for r in by_variant[label]])
            for L in lengths}
        for label, *_ in _E2_VARIANTS}
    e2_summary["_per_length_contrasts"] = length_contrasts
    e2_summary["_calibration_coefficients"] = {
        label: {
            key: _mean_std([
                (r.get("calib_coefficients") or {}).get(key)
                for r in by_variant[label]
                if (r.get("calib_coefficients") or {}).get("applicable")])
            for key in ("a_mean", "c_mean", "c_effective_mean",
                        "scale_mean", "tau_effective_mean",
                        "a_positive_layers", "c_effective_positive_layers")
        }
        for label, *_ in _E2_VARIANTS
        if label.startswith("calib")
    }
    e2_summary["_calibration_term_contributions"] = term_contributions
    e2_summary["_learned_correction_check"] = learned_correction_check
    e2_summary["_gate_stat_correlations_by_length"] = gate_corr_by_length
    e2_summary["_per_variant_tuning"] = per_variant_tuning
    e2_summary["_efficiency"] = length_efficiency
    e2_summary["_class_frequency_companions"] = endpoint_companions
    e2_summary["_provenance"] = _stamp(settings)
    _write_json(out_dir / "e2_summary.json", e2_summary)


def _e2_endpoint_class_companions(by_variant: dict,
                                  lengths: list) -> dict:
    """Class-frequency robustness companions for the E2 endpoint decision.

    The registered E2 rows read accuracy at the longest sweep length (and
    the base-to-longest drop); the per-length tables already carry macro-F1
    and balanced accuracy, but without a paired uncertainty read.  This
    re-pairs the same seeds at the endpoint and reports the same
    class-frequency-robust metrics as the E1/E3 companions, so an endpoint
    accuracy gain that reverses on macro-F1 is visible without adding a new
    pre-registered decision.
    """
    endpoint = int(lengths[-1])

    def tagged(runs):
        out = []
        for run in runs:
            metrics = ((run.get("metrics_by_length") or {})
                       .get(endpoint) or {})
            out.append({
                **run,
                "accuracy": metrics.get(
                    "accuracy", (run.get("accuracy_by_length") or {})
                    .get(endpoint)),
                "recall_macro": metrics.get("recall_macro"),
                "f1_macro": metrics.get("f1_macro"),
                "f1_weighted": metrics.get("f1_weighted"),
                "correct_bitmask": metrics.get("correct_bitmask"),
                "n": (run.get("eval_n_by_length") or {}).get(endpoint),
            })
        return out

    return _class_frequency_companions(
        [("calib - none", tagged(by_variant["calib"]),
          tagged(by_variant["none"]), "H2a"),
         ("calib - query", tagged(by_variant["calib"]),
          tagged(by_variant["query"]), "H2d")],
        f"e2 (endpoint L={endpoint})")


def _e3_capacity_matched_pre_reference(settings: Settings,
                                       out_dir: pathlib.Path, gate: str,
                                       layer_scale: float | None,
                                       target_params: int, lr: float) -> dict:
    """Train a capacity-matched Pre-LN reference for one E3 seednorm cell.

    ``seednorm`` adds two ``d_model x d_model`` projections per layer while
    ``pre`` does not, so the registered ``seednorm - pre`` readout confounds
    the normalisation mechanism with roughly 1.1M extra parameters at the
    default width.  This control keeps the same depth, gate, LayerScale,
    frozen optimiser/lr, steps, data streams and training seeds, but widens
    the Pre-LN model until its parameter count is as close as possible to
    the seednorm cell's.  It is a *capacity benchmark*, not a mechanism
    contrast (width and normalisation both differ): if the seednorm gain
    disappears against this reference, the gain is not attributable to the
    dynamic normalisation alone.

    The width search mirrors ``_matched_shallow_control`` (binary search
    over multiples of ``n_heads``, closest of the neighbouring candidates),
    and the payload records the target/actual parameter counts plus the
    relative mismatch.  The reference deliberately reuses E3's frozen lr
    rather than tuning its own, so the comparison changes only capacity and
    normalisation - the exact confounding question being audited.
    """
    step = max(1, int(settings.n_heads))
    ls_label = "off" if layer_scale is None else f"{layer_scale:g}"

    def _params_for(d_model):
        model = build(_model_cfg(settings, gate=gate, norm="pre",
                                 layer_scale=layer_scale,
                                 d_model=int(d_model)), compile=False)
        return int(sum(p.numel() for p in model.parameters()))

    lo_m, hi_m = 1, max(1, int(settings.d_model) // step)
    while _params_for(hi_m * step) < target_params and hi_m < 8192 // step:
        hi_m *= 2
    while lo_m < hi_m:
        mid = (lo_m + hi_m) // 2
        if _params_for(mid * step) < target_params:
            lo_m = mid + 1
        else:
            hi_m = mid
    candidates = sorted({max(step, (lo_m - 1) * step), lo_m * step,
                         max(step, (lo_m + 1) * step)})
    wide_d = min(candidates,
                 key=lambda d: (abs(_params_for(d) - target_params),
                                _params_for(d) < target_params, d))
    wide_params = _params_for(wide_d)
    matched = bool(abs(wide_params - target_params)
                   <= 0.05 * max(target_params, 1))
    label = f"{gate}|seednorm|{ls_label}"
    reference = (f"capacity-matched pre-LN ({gate}, ls={ls_label}, "
                 f"d_model={wide_d}, {wide_params} params; "
                 f"seednorm target {target_params})")
    runs = []
    for seed in settings.seeds:
        model, params, metrics, curve, model_cfg = _train_and_eval(
            settings, seed, lr, eval_seed=TEST_SEED, gate=gate,
            norm="pre", layer_scale=layer_scale, d_model=wide_d)
        stats, errors = collect_stats_and_errors(
            model, settings.n_pairs, settings.vocab, settings.device,
            batch=settings.stats_batch, batches=settings.stats_batches,
            seed=STATS_SEED, max_cases=0)
        resid = stats["resid_norm"].mean(dim=0)
        rec = {
            "experiment": "e3_capacity_control",
            "hypotheses": ["H3a"],
            "gate": gate, "norm": "pre", "layer_scale": layer_scale,
            "seed": seed, "lr": lr, "eval_seed": TEST_SEED,
            "stats_seed": STATS_SEED, "params": params,
            "model_cfg": model_cfg, "eval_n": int(metrics["n"]),
            "capacity_control_for": label,
            "target_params": int(target_params),
            "capacity_matched": matched,
            "reference": reference,
            **{k: v for k, v in metrics.items() if k != "n"},
            "gate_depth_ratio": gate_depth_ratio(stats),
            "resid_norm_by_layer": resid.tolist(),
            "resid_norm_last_over_first": float(
                resid[-1] / resid[0].clamp_min(1e-12)),
            **{f"err_{k}": v for k, v in errors["summary"].items()},
            **_training_record(settings, lr),
            **_run_context(settings),
            **curve,
        }
        _write_result(out_dir / "e3_capacity_matched_runs.jsonl", rec)
        runs.append(rec)
    try:
        efficiency = measure_config(
            runs[-1]["model_cfg"], settings.device,
            _efficiency_seq_len(settings), batch=1,
            warmup=_EFFICIENCY_WARMUP, repeats=_EFFICIENCY_REPEATS)
        efficiency["recorded_params"] = int(wide_params)
        efficiency["params_match_run"] = (
            int(efficiency["params"]) == int(wide_params))
    except Exception as exc:  # noqa: BLE001 - cost is a companion
        efficiency = {"available": False,
                      "reason": f"{type(exc).__name__}: {exc}"}
    payload = {
        "name": f"capacity_matched_pre_{gate}_{ls_label}",
        "gate": gate, "norm": "pre", "layer_scale": layer_scale,
        "reference_for": label,
        "n_layers": int(settings.n_layers),
        "d_model": int(wide_d),
        "params": int(wide_params),
        "target_params": int(target_params),
        "params_delta": int(wide_params - target_params),
        "params_delta_frac": ((wide_params - target_params)
                              / max(target_params, 1)),
        "capacity_matched": matched,
        "efficiency": efficiency,
        "chosen_lr": float(lr),
        "comparison": "capacity_benchmark",
        "note": ("same depth/gate/LayerScale/recipe/seeds, width chosen to "
                 "match the seednorm cell's parameter count; width and "
                 "normalisation both differ, so this is not a mechanism "
                 "contrast"),
        "runs": runs,
    }
    _write_json(out_dir / f"e3_capacity_matched_{gate}_{ls_label}.json",
                payload)
    return payload


def _e3_capacity_benchmark(settings: Settings, out_dir: pathlib.Path,
                           index: dict, lr: float,
                           efficiency: dict | None = None) -> dict:
    """Compare each seednorm cell with its capacity-matched Pre-LN control.

    The registered ``seednorm - pre`` contrasts hold the model width fixed,
    so their effect includes the seednorm parameters.  This benchmark asks
    the stricter question: does the seednorm cell still beat a Pre-LN model
    of the *same parameter count* under the identical frozen recipe?  The
    metric directions follow H3a (higher gate-depth ratio / residual growth
    / accuracy).  A win against a reference that did not itself learn the
    task is not evidence for the readout, and neither is a ``win`` by a
    target cell that never learned: competence is checked on *both* sides
    per cell and carried as ``target_competent`` / ``reference_competent``
    (the reference premise mirrors the E1 complexity benchmark).  It is
    reported outside the FDR family and never changes the pre-registered
    decision.
    """
    out = {}
    sesoi = HYPOTHESES["H3a"].get("sesoi") or {}
    metric_sesoi = {
        "accuracy": float(sesoi.get("accuracy", 0.01)),
        "gate_depth_ratio": float(sesoi.get("gate_depth_ratio", 0.05)),
        "resid_norm_last_over_first": float(
            sesoi.get("resid_norm_last_over_first", 0.05)),
    }
    print(f"\n[e3] capacity-matched Pre-LN benchmark (H3a robustness): "
          f"seednorm adds parameters that pre-LN does not have, so the "
          f"same-width readout above can confound normalisation with "
          f"capacity.  Here pre-LN is widened to the seednorm cell's "
          f"parameter count and trained with the same frozen lr, steps, "
          f"data streams and seeds.  This is a capacity benchmark, not a "
          f"controlled mechanism contrast; it is outside the FDR family.")
    print(f"{'cell':>23} {'metric':>26} {'seednorm':>10} "
          f"{'matched_pre':>12} {'paired_diff':>28} {'state':>26}")
    for gate in ("query", "calib"):
        for layer_scale in (None, 1e-2):
            ls_label = "off" if layer_scale is None else f"{layer_scale:g}"
            target_runs = index.get((gate, "seednorm", layer_scale)) or []
            if not target_runs:
                out[f"{gate}|{ls_label}"] = {
                    "available": False,
                    "reason": "seednorm cell not trained"}
                continue
            target_params = next(
                (int(r["params"]) for r in target_runs
                 if _finite(r.get("params"))), None)
            if target_params is None:
                out[f"{gate}|{ls_label}"] = {
                    "available": False,
                    "reason": "seednorm cell has no recorded parameter count"}
                continue
            ref = _e3_capacity_matched_pre_reference(
                settings, out_dir, gate, layer_scale, target_params, lr)
            target_eff = ((efficiency or {}).get("cells") or {}).get(
                f"{gate}|seednorm|{ls_label}")
            cost = _efficiency_pair(target_eff, ref.get("efficiency"))
            reference_competence = _competence_check(
                [(ref["name"], _curve_valid_runs(ref["runs"]))],
                lambda r: (int(round(r["accuracy"] * r["eval_n"])),
                           int(r["eval_n"])),
                1.0 / (settings.vocab - 2),
                what="capacity-matched reference")
            reference_competent = bool(reference_competence["passed"])
            target_competence = _competence_check(
                [(f"{gate}|seednorm|{ls_label}",
                  _curve_valid_runs(target_runs))],
                lambda r: (int(round(r["accuracy"] * r["eval_n"])),
                           int(r["eval_n"])),
                1.0 / (settings.vocab - 2),
                what="capacity-benchmark target")
            target_competent = bool(target_competence["passed"])
            cell = {
                "reference": ref["name"],
                "d_model": ref["d_model"],
                "params": ref["params"],
                "target_params": ref["target_params"],
                "capacity_matched": ref["capacity_matched"],
                "efficiency": cost,
                "target_competent": target_competent,
                "target_competence": target_competence,
                "reference_competent": reference_competent,
                "reference_competence": reference_competence,
                "by_metric": {},
            }
            for metric, sesoi_value in metric_sesoi.items():
                bench = _paired_benchmark(
                    target_runs, ref["runs"], metric, sesoi_value,
                    ref["name"])
                capacity_ok = bool(ref["capacity_matched"])
                floor_ok = (int(bench.get("n") or 0)
                            >= settings.min_seeds_for_claim)
                claim_ok = (capacity_ok and floor_ok
                            and target_competent and reference_competent)
                if bench["complexity_justified"]:
                    if not capacity_ok:
                        state = "seednorm beats pre-LN (capacity not matched)"
                    elif not target_competent:
                        state = ("seednorm beats pre-LN (target not "
                                 "competent)")
                    elif not reference_competent:
                        state = ("seednorm beats pre-LN (reference not "
                                 "competent)")
                    elif not floor_ok:
                        state = "seednorm beats matched capacity (provisional)"
                    else:
                        state = "seednorm beats matched capacity"
                elif bench["below_reference"]:
                    if not capacity_ok:
                        state = "seednorm below pre-LN (capacity not matched)"
                    elif not target_competent:
                        state = ("seednorm below pre-LN (target not "
                                 "competent)")
                    elif not reference_competent:
                        state = ("seednorm below pre-LN (reference not "
                                 "competent)")
                    else:
                        state = "seednorm below matched capacity"
                elif bench.get("unmeasured"):
                    state = "unmeasured (no curve-valid paired seeds)"
                elif not capacity_ok:
                    state = ("indistinguishable (capacity not matched)"
                             if bench.get("indistinguishable") else
                             "inconclusive (capacity not matched)")
                elif not target_competent:
                    state = "inconclusive (target not competent)"
                elif not reference_competent:
                    state = "inconclusive (reference not competent)"
                elif bench.get("indistinguishable"):
                    state = "indistinguishable"
                else:
                    state = "inconclusive (CI wider than SESOI)"
                # The floor applies to every directional/equivalence state,
                # not only to a win: a 1-2 seed "below" or "indistinguishable"
                # reading is a provisional observation, not a settled result.
                if (not floor_ok and not state.startswith("unmeasured")
                        and "provisional" not in state):
                    state += (f" (provisional: n={int(bench.get('n') or 0)} < "
                              f"claim floor {settings.min_seeds_for_claim})")
                cell["by_metric"][metric] = {
                    "seednorm_mean": (bench.get("arms") or {}).get(
                        "a", {}).get("mean"),
                    "reference_mean": (bench.get("arms") or {}).get(
                        "b", {}).get("mean"),
                    "mean": bench.get("mean"), "lo": bench.get("lo"),
                    "hi": bench.get("hi"), "p_value": bench.get("p_value"),
                    "n": bench.get("n"),
                    "seeds_used": bench.get("seeds_used"),
                    "seeds_dropped": bench.get("seeds_dropped"),
                    "curve_invalid_excluded": bench.get(
                        "curve_invalid_excluded"),
                    "sesoi": sesoi_value,
                    "state": state,
                    "capacity_matched": capacity_ok,
                    "target_competent": target_competent,
                    "reference_competent": reference_competent,
                    "claim_eligible": claim_ok,
                    "complexity_justified": bench["complexity_justified"],
                    "below_reference": bench["below_reference"],
                    "indistinguishable": bench["indistinguishable"],
                    "inconclusive": bench["inconclusive"],
                    "unmeasured": bench["unmeasured"],
                }
                a = cell["by_metric"][metric]["seednorm_mean"]
                b = cell["by_metric"][metric]["reference_mean"]
                print(f"{gate + '/' + ls_label:>23} {metric:>26} "
                      f"{a if _finite(a) else float('nan'):>10.4f} "
                      f"{b if _finite(b) else float('nan'):>12.4f} "
                      f"{_fmt_ci(bench):>28} {state:>26}")
            if cost.get("available"):
                ratios = cost.get("ratios") or {}

                def _ratio_text(key):
                    value = ratios.get(
                        f"{key}_ratio_target_over_reference")
                    return f"{value:.3f}" if _finite(value) else "-"

                peak = _ratio_text("peak_increment_bytes")
                print(f"      cost target/reference: params "
                      f"{_ratio_text('params')}x, FLOPs "
                      f"{_ratio_text('analytic_matmul_flops')}x, latency "
                      f"{_ratio_text('latency_ms_median')}x"
                      + (f", peak-memory {peak}x" if peak != "-" else ""))
            out[f"{gate}|{ls_label}"] = cell
    return out


def run_e3(settings: Settings, out_dir: pathlib.Path) -> None:
    _announce(("H3a", "H3b", "H3c"))
    # HB1/HB3/HB4/HB5 hard gate (see run_e1).
    verdict = _enforce_baseline_verdict(settings, out_dir, "e3")
    _write_manifest(settings, out_dir, "e3")
    _audit_data_splits(settings, out_dir, "e3", enforce=True)
    _enforce_reproducibility(settings, out_dir, "e3")
    _enforce_overfit_gate(settings, out_dir, "e3")
    lr = _resolve_lr(settings, out_dir, "e3")
    # HB7 pre-registers an annotation on every depth-based motivation: the
    # deep-write readouts of H3a/H3b are only informative about depth if the
    # task actually needs the grid's depth.  The baselines verdict carries
    # the tuned ladder's saturation point and its HB0 competence premise, so
    # the annotation is automatic rather than left to the reader.
    required = verdict.get("hb7_depth_required")
    saturation = verdict.get("saturation_depth")
    if required is None:
        if verdict.get("ladder_competent") is False:
            print("[e3] HB7 NOT DETERMINABLE: no ladder rung (1/2/4 layers) "
                  "had a seed-clustered lower bound above chance, so the "
                  "ladder did not solve the task at this step count - the "
                  "depth-based framing of H3a/H3b is unverified (raise the "
                  "step count and re-run `--only baselines`)")
        else:
            print("[e3] HB7 depth context unavailable (rule ceiling unknown): "
                  "the depth-based framing of H3a/H3b is unverified")
    elif not required:
        sat_sesoi = (verdict.get("depth_ladder") or {}).get(
            "sesoi_accuracy", float("nan"))
        print(f"[e3] HB7 FLAG: a tuned standard transformer is practically "
              f"at the rule ceiling (saturated within SESOI={sat_sesoi:g}) "
              f"at depth {saturation}, so the deep-write "
              f"readouts (H3a/H3b) must not be presented as evidence "
              f"that depth is needed at this scale")
    else:
        hb7_prov = (" [provisional: below the pre-registered seed floor]"
                    if not (verdict.get("seed_floor") or {}).get(
                        "hb7_claim_eligible", True) else "")
        print("[e3] HB7: depth beyond two layers is measurably required"
              f"{hb7_prov}, so the deep-write readouts are informative about "
              "depth")
    print(f"[e3] grid gate x norm x layer_scale at lr={lr:g}, "
          f"test seed {TEST_SEED}, stats seed {STATS_SEED}")
    runs = []
    examples = {}
    eval_seed, stats_seed = TEST_SEED, STATS_SEED
    for gate in ("query", "calib"):
        for norm in NORMS:
            for layer_scale in (None, 1e-2):
                for seed in settings.seeds:
                    model, params, metrics, curve, model_cfg = _train_and_eval(
                        settings, seed, lr, eval_seed=eval_seed, gate=gate,
                        norm=norm, layer_scale=layer_scale)
                    # The same diagnostic pass feeds the mechanism
                    # statistics and the bad-case mining, so the E3 grid
                    # gets the same error attribution (label error vs
                    # structural ambiguity vs retrieval miss vs readout)
                    # as E1/E2 instead of residual-norm readouts alone.
                    stats, errors = collect_stats_and_errors(
                        model, settings.n_pairs, settings.vocab,
                        settings.device, batch=settings.stats_batch,
                        batches=settings.stats_batches, seed=stats_seed,
                        max_cases=settings.max_error_cases)
                    resid = stats["resid_norm"].mean(dim=0)
                    cell = (gate, norm, layer_scale)
                    if cell not in examples or \
                            errors["summary"]["n_errors"] > examples[cell][0]:
                        ranked = sorted(errors["cases"],
                                        key=lambda c: -float(c["confidence"]))
                        examples[cell] = (errors["summary"]["n_errors"],
                                          ranked[:1])
                    for case in errors["cases"]:
                        _write_result(out_dir / "e3_error_cases.jsonl",
                                      {"experiment": "e3", **_stamp(settings),
                                       "gate": gate, "norm": norm,
                                       "layer_scale": layer_scale,
                                       "seed": seed, "lr": lr, **case})
                    rec = {
                        "experiment": "e3",
                        "hypotheses": ["H3a", "H3b", "H3c"],
                        "gate": gate, "norm": norm,
                        "layer_scale": layer_scale, "seed": seed, "lr": lr,
                        "eval_seed": eval_seed, "stats_seed": stats_seed,
                        "params": params, "model_cfg": model_cfg,
                        "eval_n": int(metrics["n"]),
                        **{k: v for k, v in metrics.items() if k != "n"},
                        "gate_depth_ratio": gate_depth_ratio(stats),
                        "resid_norm_by_layer": resid.tolist(),
                        "resid_norm_last_over_first": float(
                            resid[-1] / resid[0].clamp_min(1e-12)),
                        **{f"err_{k}": v
                           for k, v in errors["summary"].items()},
                        **_training_record(settings, lr),
                        **_run_context(settings),
                        **curve,
                    }
                    _write_result(out_dir / "e3_runs.jsonl", rec)
                    runs.append(rec)

    print(f"\n[e3] {'gate':>5} {'norm':>9} {'ls':>5} {'accuracy':>16} "
          f"{'bal_acc':>16} "
          f"{'gate_depth_ratio':>18} {'resid_growth':>14} "
          f"{'params':>9} "
          f"{'curve_ok':>9} {'late':>8} {'ofit':>5} {'tail':>7} "
          f"{'vse':>7} {'curve':>10}")
    print("  (late = U-shaped val curves (sig = also beyond validation "
          "noise); ofit = significant overfitting verdicts; tail = last-third "
          "relative train-loss change; vse = last val point's batch-sampling "
          "SE; curve_ok = curve-valid runs / all runs)")
    index = {}
    for gate in ("query", "calib"):
        for norm in NORMS:
            for layer_scale in (None, 1e-2):
                rs = [r for r in runs if r["gate"] == gate
                      and r["norm"] == norm
                      and r["layer_scale"] == layer_scale]
                index[(gate, norm, layer_scale)] = rs
                acc = _mean_std([r["accuracy"] for r in rs])
                bal = _mean_std([r["recall_macro"] for r in rs])
                ratio = _mean_std([r["gate_depth_ratio"] for r in rs])
                growth = _mean_std([r["resid_norm_last_over_first"] for r in rs])
                params = _mean_std([r["params"] for r in rs])
                tail = _mean_std([r.get("late_train_delta") for r in rs])
                vse = _mean_std([r.get("val_loss_se_last") for r in rs])
                late = sum(1 for r in rs if r.get("late_overfit"))
                late_sig = sum(1 for r in rs
                               if r.get("late_overfit_significant"))
                overfit_sig = sum(1 for r in rs
                                  if r.get("overfitting_significant"))
                verdicts = sorted({r["curve_verdict"] for r in rs})
                _, invalid = _curve_health(rs)
                ls_label = "off" if layer_scale is None else f"{layer_scale:g}"
                tail_s = (f"{tail['mean']:+.3f}" if _finite(tail["mean"])
                          else "n/a")
                vse_s = f"{vse['mean']:.4f}" if _finite(vse["mean"]) else "n/a"
                print(f"{gate:>5} {norm:>9} {ls_label:>5} "
                      f"{acc['mean']:>8.4f}±{acc['std']:<7.4f} "
                      f"{bal['mean']:>8.4f}±{bal['std']:<7.4f} "
                      f"{ratio['mean']:>9.4f}±{ratio['std']:<8.4f} "
                      f"{growth['mean']:>7.3f}±{growth['std']:<6.3f} "
                      f"{params['mean']:>9.0f} "
                      f"{len(rs) - invalid}/{len(rs)} "
                      f"{late:>3}({late_sig}) {overfit_sig:>5} "
                      f"{tail_s:>7} {vse_s:>7} "
                      f"{'/'.join(verdicts):>10}")

    efficiency_cells = []
    for (gate, norm, layer_scale), rs in index.items():
        ls_label = "off" if layer_scale is None else f"{layer_scale:g}"
        efficiency_cells.append((f"{gate}|{norm}|{ls_label}", rs))
    efficiency = _efficiency_companion(
        settings, out_dir, "e3", efficiency_cells)
    capacity_benchmark = _e3_capacity_benchmark(
        settings, out_dir, index, lr, efficiency=efficiency)

    # Same bad-case attribution as E1/E2 (raw records in
    # e3_error_cases.jsonl): the residual/gate readouts of H3a/H3b say
    # *where* the write happens, the error taxonomy says *what* failed
    # there (never found the key / found then lost it / read the wrong
    # value / label or structural ambiguity), which is what a follow-up
    # mechanism change has to act on.
    print(f"\n[e3] bad cases per grid cell (stats seed {STATS_SEED}, batch "
          f"{settings.stats_batch}x{settings.stats_batches}; raw cases in "
          f"e3_error_cases.jsonl):")
    print(f"{'gate':>5} {'norm':>9} {'ls':>5} {'err_rate':>9} "
          f"{'label':>7} {'struct':>7} {'retrieval':>10} {'lost':>7} "
          f"{'readout':>8} {'key_miss':>9} {'high_conf':>10}")
    for cell, rs in index.items():
        gate, norm, layer_scale = cell

        def ecell(key, fmt="{:.3f}"):
            m = _mean_std([r.get(key) for r in rs])
            return "n/a" if not _finite(m["mean"]) else fmt.format(m["mean"])

        ls_label = "off" if layer_scale is None else f"{layer_scale:g}"
        print(f"{gate:>5} {norm:>9} {ls_label:>5} "
              f"{ecell('err_error_rate'):>9} "
              f"{ecell('err_taxonomy_label_error'):>7} "
              f"{ecell('err_taxonomy_hard_structural'):>7} "
              f"{ecell('err_taxonomy_retrieval_miss'):>10} "
              f"{ecell('err_taxonomy_found_then_lost'):>7} "
              f"{ecell('err_taxonomy_readout_wrong'):>8} "
              f"{ecell('err_errors_key_missed_share'):>9} "
              f"{ecell('err_share_high_conf'):>10}")
    structural_cells = []
    for (gate, norm, layer_scale), rs in index.items():
        ls_label = "off" if layer_scale is None else f"{layer_scale:g}"
        structural_cells.append((f"{gate}/{norm}/ls={ls_label}", rs))
    _print_structural_decomposition(structural_cells, title="[e3]")
    print("  most confident wrong sample per cell:")
    for cell, (_n_err_cell, ranked) in examples.items():
        if not ranked:
            continue
        gate, norm, layer_scale = cell
        ls_label = "off" if layer_scale is None else f"{layer_scale:g}"
        print(f"  [{gate}/{norm}/ls={ls_label}] {_format_case(ranked[0])}")

    design_coverage = _report_design_coverage(
        out_dir, "e3", runs, "grid",
        lambda r: (r.get("gate"), r.get("norm"), r.get("layer_scale")),
        tuple((gate, norm, ls)
              for gate in ("query", "calib")
              for norm in NORMS
              for ls in (None, 1e-2)),
        planned_seeds=settings.seeds)

    # HB0 competence prerequisite: the deep-write readouts under test are
    # only interpretable when at least one grid cell demonstrably learned
    # the task at this step count.
    competence = _competence_check(
        [(f"{g}/{n}/{('off' if ls is None else f'{ls:g}')}", rs)
         for (g, n, ls), rs in index.items()],
        lambda r: (int(round(r["accuracy"] * r["eval_n"])),
                   int(r["eval_n"])),
        1.0 / (settings.vocab - 2), what="E3 grid cell")
    _write_json(out_dir / "e3_competence.json",
                {**competence, **_run_context(settings)})
    if competence["passed"]:
        print(f"[e3] HB0 competence: ok (best cell "
              f"{competence['best_arm']} has seed-clustered 95% lower "
              f"bound above chance {competence['chance']:.4f})")
    else:
        print(f"[e3] HB0 competence: FAILED - no grid cell's seed-clustered "
              f"95% lower bound is above chance {competence['chance']:.4f}; "
              f"the grid did not demonstrably learn the task at this step count, "
              f"so no contrast below may be read as a mechanism statement")
    print("\n[e3] pre-registered contrasts (H3b: "
          "resid_norm_last_over_first is the primary readout; "
          "gate_depth_ratio is secondary; H3c: "
          "calib-vs-query interaction with the dynamic-normalisation "
          "arm):")
    clog = ContrastLog("e3", hypotheses=("H3a", "H3b", "H3c"),
                       min_seeds=settings.min_seeds_for_claim,
                       competence=competence,
                       coverage=design_coverage,
                       balance_ok=((verdict.get("eval_answer_balance") or {})
                                   .get("all_uniform")))
    for gate in ("query", "calib"):
        for layer_scale in (None, 1e-2):
            a = index[(gate, "seednorm", layer_scale)]
            b = index[(gate, "pre", layer_scale)]
            ls_label = "off" if layer_scale is None else f"{layer_scale:g}"
            for metric in ("gate_depth_ratio",
                           "resid_norm_last_over_first"):
                clog.print(f"{gate}: seednorm - pre (ls={ls_label})",
                           "H3a", metric,
                           _paired_diff(a, b, metric),
                           warn=_curve_warn((f"{gate}/seednorm", a),
                                            (f"{gate}/pre", b)))
    # H3a completeness: the average-own-level norm main effects, collapsed
    # over gate and LayerScale.  The within-cell seednorm-vs-pre rows above
    # are conditional on one control cell; without the collapsed readout a
    # level whose contribution is real under one gate but absent under the
    # other is never averaged, and `post`/`mix` (8 trained cells) would enter
    # no contrast at all.  Each level is differenced against `pre` inside
    # each gate x LayerScale control and then averaged (repeated-measures
    # pairing), so the gate/scale variance cancels exactly as in the paired
    # marginal rows.  The seednorm rows carry H3a's registered mechanism
    # direction via the norm_* registry keys; post/mix are two-sided
    # companions (accuracy is two-sided for all levels).
    print(f"\n[e3] H3a norm-level main effects (each norm vs pre-LN, "
          f"collapsed over gate x LayerScale; H3a's mechanism readout is the "
          f"registered seednorm row, post/mix are two-sided completeness "
          f"companions):")
    controls = [(g, ls) for g in ("query", "calib") for ls in (None, 1e-2)]
    for norm in NORMS:
        if norm == "pre":
            continue
        level_cells = [index[(g, norm, ls)] for g, ls in controls]
        base_cells = [index[(g, "pre", ls)] for g, ls in controls]
        warn = _curve_warn(
            *[(f"{g}/{norm}/ls={('off' if ls is None else f'{ls:g}')}", rs)
              for (g, ls), rs in zip(controls, level_cells)],
            *[(f"{g}/pre/ls={('off' if ls is None else f'{ls:g}')}", rs)
              for (g, ls), rs in zip(controls, base_cells)])
        # Only seednorm carries H3a's registered "+" mechanism sign; post/mix
        # route through their own undirected registry keys so a significant
        # negative main effect is a two-sided finding, not an H3a refutation
        # the hypothesis never predicted.
        comp = "" if norm == "seednorm" else "postmix_"
        clog.print(f"{norm} - pre (norm main effect, gate_depth_ratio)",
                   "H3a", "norm_gate_depth_ratio",
                   _grid_level_main_effect("gate_depth_ratio",
                                           level_cells, base_cells),
                   warn=warn,
                   direction_key=f"{comp}norm_gate_depth_ratio")
        clog.print(f"{norm} - pre (norm main effect, "
                   f"resid_norm_last_over_first)",
                   "H3a", "norm_resid_norm_last_over_first",
                   _grid_level_main_effect("resid_norm_last_over_first",
                                           level_cells, base_cells),
                   warn=warn,
                   direction_key=f"{comp}norm_resid_norm_last_over_first")
        clog.print(f"{norm} - pre (norm main effect, accuracy)",
                   "H3a", "norm_accuracy",
                   _grid_level_main_effect("accuracy",
                                           level_cells, base_cells),
                   warn=warn)
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
        for layer_scale in (None, 1e-2):
            a = index[("calib", norm, layer_scale)]
            b = index[("query", norm, layer_scale)]
            ls_label = "off" if layer_scale is None else f"{layer_scale:g}"
            clog.print_equivalence_ci(
                f"{norm}/ls={ls_label}: calib - query", "H3a", "accuracy",
                _paired_equivalence(a, b, "accuracy",
                                    HYPOTHESES["H3a"]["sesoi"]["accuracy"],
                                    mode="noninferior"),
                warn=_curve_warn((f"calib/{norm}/ls={ls_label}", a),
                                 (f"query/{norm}/ls={ls_label}", b)))
    # H3c: the co-design interaction.  The marginal calib-vs-query
    # differences under seednorm and under pre-LN are differenced on the
    # same training seed, so the (large) between-seed variance cancels and
    # only the interaction remains; a positive DiD says the calibration
    # benefit is specific to the dynamic-normalisation arm, i.e. the two
    # components work together rather than additively.
    for metric in ("accuracy", "gate_depth_ratio"):
        cells = (index[("calib", "seednorm", None)],
                 index[("query", "seednorm", None)],
                 index[("calib", "pre", None)],
                 index[("query", "pre", None)])
        clog.print(
            "seednorm x (calib - query) - pre x (calib - query)",
            "H3c", metric,
            _paired_did(*cells, metric),
            warn=_curve_warn(("calib/seednorm", cells[0]),
                             ("query/seednorm", cells[1]),
                             ("calib/pre", cells[2]),
                             ("query/pre", cells[3])))
    # H3c robustness companion: the same co-design interaction at LayerScale
    # on.  H3c is pre-registered "at LayerScale off"; if the interaction is
    # a property of the co-design it should not flip sign when LayerScale
    # turns on.  Reported as a companion (not a second decision rule) so a
    # sign flip is visible instead of being read off a single control cell.
    print(f"\n[e3] H3c robustness companion: the same co-design interaction at "
          f"LayerScale=1e-2 (the pre-registered row is at LayerScale off):")
    for metric in ("accuracy", "gate_depth_ratio"):
        cells = (index[("calib", "seednorm", 1e-2)],
                 index[("query", "seednorm", 1e-2)],
                 index[("calib", "pre", 1e-2)],
                 index[("query", "pre", 1e-2)])
        ci = _paired_did(*cells, metric)
        print(f"      ls=1e-2: seednorm x (calib - query) - pre x "
              f"(calib - query) ({metric}): {_fmt_ci(ci)}")
    clog.finalize(out_dir, q=settings.fdr_q)
    robustness_rules = []
    for norm in NORMS:
        for layer_scale in (None, 1e-2):
            ls_label = "off" if layer_scale is None else f"{layer_scale:g}"
            robustness_rules.append((
                f"{norm}/ls={ls_label}: calib - query",
                index[("calib", norm, layer_scale)],
                index[("query", norm, layer_scale)], "H3a"))
    for gate in ("query", "calib"):
        for layer_scale in (None, 1e-2):
            ls_label = "off" if layer_scale is None else f"{layer_scale:g}"
            robustness_rules.append((
                f"{gate}: seednorm - pre (ls={ls_label})",
                index[(gate, "seednorm", layer_scale)],
                index[(gate, "pre", layer_scale)], "H3a"))
    metric_robustness = _class_frequency_companions(
        robustness_rules, "e3")
    e3_summary = {f"{g}|{n}|{ls}": _aggregate_runs(rs)
                  for (g, n, ls), rs in index.items()}
    e3_summary["_capacity_matched_pre"] = capacity_benchmark
    e3_summary["_efficiency"] = efficiency
    e3_summary["_class_frequency_companions"] = metric_robustness
    e3_summary["_provenance"] = _stamp(settings)
    _write_json(out_dir / "e3_summary.json", e3_summary)


def run_sanity(settings: Settings, out_dir: pathlib.Path) -> None:
    """Standalone pre-flight: HB3 split audit + seed-determinism + HB5 gate.

    The E1/E2/E3 runners invoke the identical gates themselves, so a clean
    E1/E2/E3 run implies a passed pre-flight (``--only sanity``).
    """
    _announce(("HB3", "HB5"))
    _write_manifest(settings, out_dir, "sanity")
    _audit_data_splits(settings, out_dir, "sanity", enforce=True)
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
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="force deterministic torch kernels on/off "
                             "(default: on; --no-deterministic opts out and "
                             "is recorded in the run provenance)")
    parser.add_argument("--per-variant-tuning",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="tune each E1/E2 primary arm's own lr (and "
                             "calib's calib_tau_init) on validation data as "
                             "a separate fairness benchmark (default: on; "
                             "the same-recipe controlled families are "
                             "unchanged)")
    parser.add_argument("--out", default="results")
    parser.add_argument("--verify-artifacts", action="store_true",
                        help="check files in --out against artifact_index.json "
                             "and exit")
    args = parser.parse_args(argv)
    if args.verify_artifacts:
        return _verify_artifacts(args.out)

    settings = Settings.quick(args.device) if args.quick else Settings(
        device=args.device)
    if args.deterministic is not None:
        settings.deterministic = bool(args.deterministic)
    if args.per_variant_tuning is not None:
        settings.per_variant_tuning = bool(args.per_variant_tuning)
    if args.seeds is not None:
        if args.seeds < 1:
            print("[runner] --seeds must be a positive integer; aborting")
            return 2
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
    if settings.steps < 1:
        print("[runner] --steps must be a positive integer; aborting")
        return 2
    if settings.lr is not None and (
            not math.isfinite(settings.lr) or settings.lr <= 0.0):
        print("[runner] --lr must be a finite positive number; aborting")
        return 2
    if not 0.0 < settings.fdr_q < 1.0:
        print("[runner] --fdr-q must lie strictly between 0 and 1; aborting")
        return 2
    if settings.d_model < 1 or settings.d_model % settings.n_heads:
        print(f"[runner] --d-model must be a positive multiple of n_heads="
              f"{settings.n_heads}; aborting")
        return 2
    if settings.n_layers < 1:
        print("[runner] --n-layers must be a positive integer; aborting")
        return 2
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings.out_dir = out_dir
    _RUN_ARTIFACTS.clear()
    started = time.monotonic()
    status = "failed"
    try:
        _write_json(out_dir / "hypotheses.json", HYPOTHESES)
        _run_event(settings, args.only, "started", quick=bool(args.quick),
                   seeds=list(settings.seeds), steps=settings.steps,
                   lr=settings.lr, device=settings.device,
                   fdr_q=settings.fdr_q)
        _write_json(out_dir / "environment.json", _run_context(settings))
        print(f"[runner] only={args.only} quick={args.quick} "
              f"seeds={settings.seeds} steps={settings.steps} "
              f"lr={settings.lr if settings.lr is not None else 'tuned'} "
              f"fdr_q={settings.fdr_q:g} "
              f"device={settings.device} out={out_dir}")
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
    except (OverfitSanityError, ReproducibilityError,
            SplitAuditError, BaselineVerdictError) as e:
        status = "aborted"
        print(f"[runner] ABORTED before grid training: {e}")
        _run_event(settings, args.only, "aborted",
                   error=type(e).__name__, reason=str(e),
                   elapsed_seconds=round(time.monotonic() - started, 3))
        return 3
    except Exception as e:
        _run_event(settings, args.only, "failed",
                   error=type(e).__name__, reason=str(e),
                   elapsed_seconds=round(time.monotonic() - started, 3))
        raise
    else:
        status = "completed"
        _run_event(settings, args.only, "completed",
                   elapsed_seconds=round(time.monotonic() - started, 3))
        print(f"\n[runner] done; records in {out_dir}/")
        return 0
    finally:
        try:
            _write_artifact_index(out_dir, settings, args.only,
                                  completed=(status == "completed"),
                                  elapsed_seconds=time.monotonic() - started)
        except Exception as e:  # noqa: BLE001 - index failure is a warning
            print(f"[runner] WARNING: could not write artifact index: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
