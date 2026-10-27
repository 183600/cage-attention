# ruff: noqa: F401
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
    BCa readout (see ``cage.metrics``).  Because the percentile bootstrap
    p-value can also fall below what a paired design can produce at this
    n, every contrast carries an **exact paired sign-flip** p-value and
    its attainable floor (``sign_flip_*``), with a parallel BH-corrected
    exact family; a bootstrap-only verdict is reported as provisional
    rather than as confirmation.  The companions audit the
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
6. **Design preparation before data collection.**  Every grid is written a
    ``<tag>_design_plan.json`` *before* its first run: the pre-registered
    SESOI per metric is translated into the largest seed-level SD the
    planned N can tolerate (and, for accuracy, into the required seed count
    from the baseline-ladder pilot variance).  Because accuracy contrasts
    are paired over seeds, that pilot is the *paired-difference* variance -
    the largest observed adjacent-depth paired SD or the independent-arm
    ``sqrt(2)`` bound - not the single-arm SD, which would assume perfectly
    correlated arms and certify an underpowered design.  The plan also
    reports the finite-evaluation-stream binomial floor separately: more
    training seeds cannot remove it, so a stream-limited warning asks for a
    wider evaluation stream.  An underpowered plan is
    flagged at preparation time instead of being discovered as an
    "inconclusive" after the compute is spent.  The plan separately reports
    whether the planned N clears ``Settings.min_seeds_for_claim``: power at
    n=2 is not a claim licence when the registered floor is 3, and a quick
    pipeline check must not be mistaken for evidence.  The plan is frozen
    once the first outcome row exists (same design may be rechecked;
    changed seeds/SESOIs/test modes/FDR level raise ``DesignPlanError``),
    with every pre-outcome revision kept in
    ``<tag>_design_plan_history.jsonl``.  The plan also translates the
    registered direction of each metric into the exact paired sign-flip
    test's attainable floor: a two-sided contrast at five seeds has a
    minimum exact p of 0.0625, so it cannot be confirmed at q=0.05 however
    large the effect, and the required seed count is reported before the
    grid runs instead of being discovered when the bootstrap looks
    significant.  The
    runs themselves are scheduled as randomised complete blocks (one block
    per training seed, every cell once per block, recorded in
    ``<tag>_run_order.json``), so a slow machine drift cannot be aliased
    with one treatment arm; E1's calibration-term ablations and its
    active-control placebo are part of the *same* schedule as the four gate
    variants, so an ablation-vs-full contrast is not two grids run at
    different times.  The schedule is deterministic from
    ``Settings.run_order_seed`` and never perturbs the data/dropout RNG
    streams.  After the grid, ``<tag>_run_order_realized.json`` checks the
    rows' recorded block/position against that frozen schedule: a resumed
    or reordered loop is caught instead of being certified from the plan
    alone.
7. **Replication and claim-level synthesis.**  Two readings sit on top of
   the per-contrast decision without changing it.  First, every row carries
   a t-based **same-size replication interval** for a future study's mean
   (``sqrt(2/n)``, not the CI's ``sqrt(1/n)``) and the probability that a
   repeat keeps the registered sign, so a significant but noisy effect is
   labelled *replication-uncertain* instead of "robust".  Second, the
   pre-registered hypotheses are conjunctions ("A and B and C"); a single
   surviving row must not be read as "the hypothesis is confirmed".  The
   report groups rows per hypothesis and marks it ``supported`` only when
   every registered directional/null-prediction row supports it, with
   ``partial``/``refuted``/``undirected``/``no_contrast`` as explicit
   outcomes.  Both are companions: row-level ``supports_hypothesis`` and
   the BH family are untouched.  See ``docs/results_analysis.md``.

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
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
import hashlib
import io
import json
import math
import operator
import os
import pathlib
import platform
import random
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from statistics import NormalDist

import torch

from ..baselines import (
    copy_last_token_stats,
    linear_probe_accuracy,
    probe_vs_chance_test,
    rule_naive_last_match_stats,
    rule_pair_lookup_stats,
    train_linear_probe,
    tune_linear_probe,
)
from ..data import (
    _MAX_DIFFICULTY_STRATA,
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
from ..efficiency import measure_config
from ..metrics import (
    _positive_integer,
    benjamini_hochberg,
    benjamini_yekutieli,
    holm_bonferroni,
    mcnemar_test,
    simes_test,
    clustered_proportion_ci,
    crossed_cluster_bootstrap_ci,
    equivalence_test,
    evaluation_fold_audit,
    minimum_effect_test,
    noninferiority_test,
    paired_bootstrap_ci,
    replication_interval,
    sign_flip_min_units,
    sign_flip_resolution,
    standardized_proportion,
    wilson_ci,
)
from ..model import Cfg, GATES, NORMS, build
from ..repro import configure_determinism, seed_everything, stream_generator
from ..runlock import LOCK_EXIT_CODE, ResultsLock, ResultsLockError
from ..train import (
    _ERROR_TYPES,
    _stats_correct_vector,
    _stats_layer_matrix,
    decode_correct_mask,
    collect_stats,
    collect_stats_and_errors,
    error_detection_aucs,
    eval_accuracy,
    eval_linked_lengths,
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

# ---------------------------------------------------------------------------
# Implementation layout
# ---------------------------------------------------------------------------
# The runner was historically one 19k-line module.  It is now split by
# responsibility under ``cage/experiments/``.  The fragments are compiled and
# executed into this package's namespace (in dependency order) so that every
# helper keeps working as ``cage.experiments.<name>``, including tests and
# tooling that monkeypatch module attributes such as ``_run_context`` or
# ``_RUN_ARTIFACTS``.  Behaviour is byte-for-byte the pre-split implementation.
import __future__ as _future

_FRAGMENT_DIR = pathlib.Path(__file__).resolve().parent
_FRAGMENTS = (
    "_config.py",
    "_design.py",
    "_planning.py",
    "_contrasts.py",
    "_synthesis.py",
    "_contrastlog.py",
    "_contrastlog_finalize.py",
    "_artifacts.py",
    "_environment.py",
    "_gates.py",
    "_training.py",
    "_audits.py",
    "_baselines.py",
    "_e1.py",
    "_e1_runner.py",
    "_e2.py",
    "_e2_runner.py",
    "_e3_cli.py",
)
for _fragment_name in _FRAGMENTS:
    _fragment_path = _FRAGMENT_DIR / _fragment_name
    exec(compile(_fragment_path.read_text(encoding="utf-8"),
                 str(_fragment_path), "exec",
                 flags=_future.annotations.compiler_flag,
                 dont_inherit=True), globals())
del _fragment_name, _fragment_path
