"""Hypothesis-driven experiment runners for E1/E2/E3.

Design principles
-----------------
1. **Hypotheses first.**  Every config is annotated with the hypothesis
   it tests (``HYPOTHESES``) and a pre-registered decision rule.  The
   runner prints the hypotheses before executing and stores their ids
   with every result record, so the E3 grid is not a blind
   hyperparameter sweep: each contrast answers a mechanism question.
2. **Tuned baselines before complex models.**  The rule system (exact
   ceiling), the trivial floor and a learning-rate-tuned linear probe
   run before any transformer grid.  A shared training recipe is tuned
   on the plain ``none`` baseline (validation seed), then frozen for all
   gate variants so the E1/E3 comparisons stay controlled.  The
   baselines are checked *statistically* before the grids run: HB1/HB2/
   HB3 verdicts are written to ``baselines_verdict.json`` (rule ceiling
   with a Wilson CI, a bootstrap near-chance test for the tuned probe,
   and the split audit), and E1 re-reads that verdict and the rule
   ceiling as a positive control, so a gate variant is only credited for
   what it does beyond a tuned simple baseline on a task the simple rule
   can actually solve.
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
5. **Multiplicity and power.**  Each experiment interprets a *family* of
   contrasts, so the family is read through Benjamini-Hochberg FDR
   control (``--fdr-q``, default 0.05).  Every contrast reports its
   bootstrap p-value, the fraction of seeds agreeing on the sign, and
   the design's minimal detectable effect at 80% power (MDE80); "CI
   includes 0" with ``|effect| < MDE80`` is always reported as
   *inconclusive*, not as evidence of absence.  This matters at the
   5-seed scale used here: an uncorrected 95% readout over ~16 null
   contrasts would produce about one spurious "supported" claim per
   experiment.

Outputs land in ``results/`` (git-ignored); nothing is written into the
tracked tree.

Usage
-----
    python3 -m cage.experiments --only baselines
    python3 -m cage.experiments --only e1 --seeds 5
    python3 -m cage.experiments --only all --quick          # pipeline check only
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
from dataclasses import dataclass, field

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
    STATS_SEED,
    TEST_SEED,
    VAL_SEED,
    length_sweep_lengths,
    make_batch,
    split_leakage_audit,
)
from .metrics import benjamini_hochberg, paired_bootstrap_ci
from .model import GATES, NORMS, build
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
    train_model,
    training_curve_verdict,
)

# Seeds with fixed roles (independent of every training seed) are defined in
# ``cage.data`` so that every entry point shares one source of truth:
#   VAL_SEED   - model selection only (learning-rate tuning, baseline checks)
#   TEST_SEED  - the reported numbers; never used for any decision
#   STATS_SEED - mechanism statistics (matches the collect_stats default)

HYPO_IDS = ("HB1", "HB2", "HB3", "H1a", "H1b", "H1c", "H2a", "H2b", "H3a",
            "H3b")

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
    },
    "H2b": {
        "experiment": "e2",
        "question": "Do the attention statistics drift with length as the "
                    "calibration assumes?",
        "hypothesis": "Raw entropy grows with log(T), while the quantity the "
                      "gate normalises by, H/log(T), is length-stable.",
        "prediction": "raw entropy drift (longest - base) > 0 and "
                      "normalised entropy drift H/log(T) ~ 0, with the "
                      "calibrated gate drift small",
        "decision": "paired bootstrap 95% CIs, FDR-controlled across the E2 "
                    "contrasts (q=0.05): raw entropy drift CI excludes 0 "
                    "with a positive mean, and the H/log(T) drift CI "
                    "includes 0",
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
    lr: float | None = None
    lr_grid: tuple = (1e-3, 3e-4, 1e-4)
    curve_points: int = 10      # validation checks per run (0 disables)
    val_batches: int = 2        # validation batches per check (fixed stream)
    cost_warmup: int = 2
    cost_runs: int = 7
    max_error_cases: int = 64   # per gate x seed, dumped to disk
    fdr_q: float = 0.05         # BH false-discovery rate for contrast families
    out_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path("results"))

    @classmethod
    def quick(cls, device: str = "cpu") -> "Settings":
        """Tiny pipeline-check configuration (never for reported numbers)."""
        return cls(
            device=device, d_model=48, n_heads=4, n_layers=4, max_len=64,
            vocab=64, batch=16, n_pairs=6, steps=12, seeds=(0, 1),
            eval_batch=64, eval_batches=2, stats_batch=32, stats_batches=2,
            probe_steps=40, probe_batch=16,
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


def _paired_diff(runs_a, runs_b, metric):
    """Paired contrast ``a - b`` plus the control-side level.

    ``baseline_mean`` (the mean of ``metric`` on side ``b``) is attached so
    the report can print the effect as a relative change: a contrast can be
    statistically significant with zero variance yet practically
    negligible, and the relative effect is what exposes that.
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
    return ci


def _fmt_ci(ci) -> str:
    if ci["n"] == 0 or not math.isfinite(ci["mean"]):
        return "n/a"
    supported = (ci["lo"] > 0.0) or (ci["hi"] < 0.0)
    mde = ci.get("mde80", float("nan"))
    if supported:
        verdict = "CI excludes 0"
    elif math.isfinite(mde) and abs(ci["mean"]) < mde:
        verdict = (f"inconclusive: |effect| < MDE80={mde:.4f} "
                   f"(design cannot resolve it)")
    else:
        verdict = "CI includes 0"
    p = ci.get("p_value", float("nan"))
    p_s = f"p={p:.3f} " if math.isfinite(p) else ""
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
    """

    def __init__(self, tag: str):
        self.tag = tag
        self.rows: list[dict] = []

    def print(self, rule, hyp, metric, ci, warn="") -> None:
        self.rows.append({
            "rule": rule, "hypothesis": hyp, "metric": metric,
            "mean": ci.get("mean"), "lo": ci.get("lo"), "hi": ci.get("hi"),
            "n": ci.get("n"), "p_value": ci.get("p_value"),
            "mde80": ci.get("mde80"),
            "sign_consistency": ci.get("sign_consistency"),
            "baseline_mean": ci.get("baseline_mean"),
        })
        line = f"  [{hyp}] {rule} ({metric}): {_fmt_ci(ci)}"
        if warn:
            line += f"  [{warn}]"
        print(line)

    def finalize(self, out_dir: pathlib.Path, q: float = 0.05) -> dict:
        adj = benjamini_hochberg([r["p_value"] for r in self.rows], q=q)
        for row, a in zip(self.rows, adj):
            row["p_adj"] = a["p_adj"]
            row["fdr_reject"] = a["reject"]
            base = row.get("baseline_mean")
            row["rel_effect"] = (
                row["mean"] / abs(base)
                if base is not None and math.isfinite(base) and base != 0.0
                else float("nan"))
        testable = [r for r in self.rows if math.isfinite(r["p_adj"])]
        surviving = [r for r in testable if r["fdr_reject"]]
        print(f"\n[{self.tag}] multiplicity control (Benjamini-Hochberg FDR, "
              f"q={q:g}): {len(surviving)}/{len(testable)} testable "
              f"contrasts survive (uncorrected p -> adjusted p; only "
              f"'SURVIVES' supports a pre-registered decision; rel = effect "
              f"relative to the control-side mean, read it for practical "
              f"significance):")
        for r in testable:
            rel_s = (f", rel={r['rel_effect']:+.2%}"
                     if math.isfinite(r["rel_effect"]) else "")
            print(f"  [{r['hypothesis']}] {r['rule']} ({r['metric']}): "
                  f"p={r['p_value']:.4f} -> p_adj={r['p_adj']:.4f} "
                  f"{'SURVIVES' if r['fdr_reject'] else 'not significant'} "
                  f"(effect={r['mean']:+.4f}{rel_s}, "
                  f"MDE80={r['mde80']:.4f}, seeds={r['n']}, "
                  f"sign-consistency={r['sign_consistency']:.2f})")
        payload = {"tag": self.tag, "q": q, "n_contrasts": len(self.rows),
                   "n_testable": len(testable),
                   "n_surviving": len(surviving), "rows": self.rows}
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


def _write_result(path: pathlib.Path, record: dict) -> None:
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


def _load_baseline_verdict(out_dir: pathlib.Path):
    """The HB1/HB2/HB3 verdicts written by ``run_baselines``, if present.

    E1/E2/E3 read the learned model against the simple baselines, so a
    missing verdict file is surfaced as a note instead of silently
    treating the baseline check as passed.
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
    }
    cfg.update(overrides)
    return cfg


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
                    eval_seed: int = TEST_SEED, **model_overrides):
    if eval_seed in tuple(settings.seeds):
        raise ValueError(
            f"eval_seed={eval_seed} is also a configured training seed; "
            f"evaluation data must come from a stream that never trained "
            f"the model"
        )
    torch.manual_seed(seed)
    model = build(_model_cfg(settings, **model_overrides), compile=False)
    params = sum(p.numel() for p in model.parameters())
    eval_every = (max(1, settings.steps // settings.curve_points)
                  if settings.curve_points > 0 else 0)
    hist = train_model(model, steps=settings.steps, batch=settings.batch,
                       n_pairs=settings.n_pairs, lr=lr, vocab=settings.vocab,
                       device=settings.device, log_every=0, seed=seed,
                       eval_every=eval_every, val_batches=settings.val_batches)
    metrics = eval_metrics(model, settings.n_pairs, settings.vocab,
                           settings.device, batch=settings.eval_batch,
                           batches=settings.eval_batches, seed=eval_seed)
    return model, params, metrics, _curve_record(hist)


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
            _, _, metrics, curve = _train_and_eval(
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


def run_baselines(settings: Settings, out_dir: pathlib.Path) -> None:
    _announce(("HB1", "HB2", "HB3"))
    sha = _git_sha()
    record = {"experiment": "baselines", "hypotheses": ["HB1", "HB2", "HB3"],
              "git_sha": sha, "seed": TEST_SEED}

    audit = _audit_data_splits(settings, out_dir)

    balance = _answer_balance(settings)
    print(f"[baselines] answer-value balance over {balance['n_samples']} "
          f"samples: {balance['n_classes_present']} classes present, share "
          f"min={balance['min_class_share']:.5f} max={balance['max_class_share']:.5f} "
          f"(uniform={balance['expected_uniform_share']:.5f})")
    _write_json(out_dir / "baselines_balance.json", balance)

    hb1_supported = None
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
        probe = train_linear_probe(settings.n_pairs, settings.vocab,
                                   settings.device, steps=settings.probe_steps,
                                   batch=settings.probe_batch, lr=best_lr,
                                   seed=seed)
        acc = linear_probe_accuracy(probe, settings.n_pairs, settings.vocab,
                                    settings.device, batch=settings.eval_batch,
                                    batches=settings.eval_batches, seed=TEST_SEED)
        probe_accs.append(acc)
        _write_result(out_dir / "baselines_runs.jsonl",
                      {**record, "name": "linear_bag_probe", "lr": best_lr,
                       "training_seed": seed, "accuracy": acc,
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
    verdict = {
        "train_seed": settings.seeds[0] if settings.seeds else None,
        "test_seed": TEST_SEED,
        "val_seed": VAL_SEED,
        "chance": chance,
        "hb1_supported": bool(hb1_supported),
        "hb2_supported": bool(hb2["hb2_supported"]),
        "hb3_supported": bool(audit.get("hb3_supported")),
        "probe_accuracy_mean": summary["mean"],
        "probe_accuracy_std": summary["std"],
        "hb2_accuracy_minus_chance": {
            "mean": hb2["mean"], "lo": hb2["lo"], "hi": hb2["hi"],
            "p_value": hb2["p_value"], "mde80": hb2["mde80"],
            "n": hb2["n"],
        },
    }
    _write_json(out_dir / "baselines_verdict.json", verdict)
    print(f"[baselines] verdict: HB1={'ok' if verdict['hb1_supported'] else 'FAILED'}"
          f" HB2={'ok' if verdict['hb2_supported'] else 'FAILED'}"
          f" HB3={'ok' if verdict['hb3_supported'] else 'FAILED'} "
          f"(stored in baselines_verdict.json; a failed baseline check "
          f"invalidates the corresponding claims above it)")


def run_e1(settings: Settings, out_dir: pathlib.Path) -> None:
    _announce(("H1a", "H1b", "H1c"))
    lr = _resolve_lr(settings, out_dir, "e1")
    sha = _git_sha()
    runs = []
    examples = {}
    for gate in GATES:
        for seed in settings.seeds:
            model, params, metrics, curve = _train_and_eval(settings, seed, lr,
                                                            gate=gate)
            stats = collect_stats(model, settings.n_pairs, settings.vocab,
                                  settings.device, batch=settings.stats_batch,
                                  batches=settings.stats_batches,
                                  seed=STATS_SEED)
            corr = gate_stat_correlations(stats)
            fc = false_concentration_analysis(stats)
            fc_sweep = false_concentration_sweep(stats)
            ed = error_detection_aucs(stats)
            errors = collect_error_cases(
                model, settings.n_pairs, settings.vocab, settings.device,
                batch=settings.stats_batch, batches=settings.stats_batches,
                seed=STATS_SEED, max_cases=settings.max_error_cases)
            if gate not in examples or \
                    errors["summary"]["n_errors"] > examples[gate][0]:
                ranked = sorted(errors["cases"],
                                key=lambda c: -float(c["confidence"]))
                examples[gate] = (errors["summary"]["n_errors"], ranked[:3])
            for case in errors["cases"]:
                _write_result(out_dir / "e1_error_cases.jsonl",
                              {"experiment": "e1", "gate": gate, "seed": seed,
                               "lr": lr, **case})
            rec = {
                "experiment": "e1", "hypotheses": ["H1a", "H1b", "H1c"],
                "git_sha": sha, "gate": gate, "seed": seed, "lr": lr,
                "params": params, **metrics, **corr, **fc, **ed,
                **{f"err_{k}": v for k, v in errors["summary"].items()},
                **_cost_record(settings, model), **curve,
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
            _write_result(out_dir / "e1_runs.jsonl", rec)
            runs.append(rec)
    by_gate = {g: [r for r in runs if r["gate"] == g] for g in GATES}
    table = {g: _aggregate_runs(rs) for g, rs in by_gate.items()}

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
              "against the tuned simple baselines (probe, rules) before "
              "crediting any gate variant")
    else:
        probe = verdict.get("probe_accuracy_mean", float("nan"))
        marks = "/".join("ok" if verdict.get(k) else "FAILED"
                         for k in ("hb1_supported", "hb2_supported",
                                   "hb3_supported"))
        print(f"[e1] simple-baseline check (from --only baselines): "
              f"HB1/HB2/HB3={marks}, tuned linear probe test accuracy="
              f"{probe:.4f} vs chance {verdict.get('chance', float('nan')):.4f} "
              f"- the transformer is only credited for what it does beyond "
              f"this tuned baseline and the rule ceiling")

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
    clog = ContrastLog("e1")
    clog.print("calib - none", "H1a", "accuracy",
               _paired_diff(by_gate["calib"], by_gate["none"], "accuracy"),
               warn=_curve_warn(("calib", by_gate["calib"]),
                                ("none", by_gate["none"])))
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


def run_e2(settings: Settings, out_dir: pathlib.Path) -> None:
    _announce(("H2a", "H2b"))
    lr = _resolve_lr(settings, out_dir, "e2")
    lengths = length_sweep_lengths(base=settings.n_pairs, factor=2, n=4,
                                   max_len=settings.max_len)
    print(f"[e2] training at n_pairs={settings.n_pairs}, sweep over {lengths} "
          f"(test seed {TEST_SEED}, stats seed {STATS_SEED})")
    sha = _git_sha()
    runs = []
    examples = {}
    for gate in GATES:
        for seed in settings.seeds:
            model, params, _, curve = _train_and_eval(settings, seed, lr,
                                                      gate=gate)
            rec = {"experiment": "e2", "hypotheses": ["H2a", "H2b"],
                   "git_sha": sha, "gate": gate, "seed": seed, "lr": lr,
                   "params": params, "accuracy_by_length": {},
                   "entropy_by_length": {}, "entropy_norm_by_length": {},
                   "maxprob_by_length": {}, "gate_by_length": {},
                   **_cost_record(settings, model), **curve}
            for length in lengths:
                metrics = eval_metrics(model, length, settings.vocab,
                                       settings.device, batch=settings.eval_batch,
                                       batches=settings.eval_batches,
                                       seed=TEST_SEED)
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
                              {"experiment": "e2", "gate": gate, "seed": seed,
                               "length": lengths[-1], "lr": lr, **case})
            if gate not in examples or \
                    errors["summary"]["n_errors"] > examples[gate][0]:
                ranked = sorted(errors["cases"],
                                key=lambda c: -float(c["confidence"]))
                examples[gate] = (errors["summary"]["n_errors"], ranked[:3])
            _write_result(out_dir / "e2_runs.jsonl", rec)
            runs.append(rec)

    by_gate = {g: [r for r in runs if r["gate"] == g] for g in GATES}
    print(f"\n[e2] accuracy by n_pairs (mean +- std over seeds):")
    line = f"{'gate':>6}" + "".join(f"{L:>12}" for L in lengths) + \
           f"{'drop':>10} {'entropy_drift':>14}"
    print(line)
    for gate in GATES:
        cells = []
        for length in lengths:
            agg = _mean_std([r["accuracy_by_length"][length]
                             for r in by_gate[gate]])
            cells.append(f"{agg['mean']:.3f}±{agg['std']:.3f}")
        drop = _mean_std([r["accuracy_drop"] for r in by_gate[gate]])
        drift = _mean_std([r["entropy_drift"] for r in by_gate[gate]])
        print(f"{gate:>6}" + "".join(f"{c:>12}" for c in cells)
              + f"{drop['mean']:>10.3f} {drift['mean']:>14.3f}")

    print(f"\n[e2] inference cost at the training length (same protocol as "
          f"e1; check whether the length benefit costs latency; curve_ok = "
          f"curve-valid runs / all runs):")
    print(f"{'gate':>6} {'params':>10} {'flops/M':>9} {'ms/sample':>10} "
          f"{'mem/MB':>9} {'curve_ok':>8} {'curve':>10}")
    for gate in GATES:
        t = _aggregate_runs(by_gate[gate])

        def cell(key, fmt="{:.3f}"):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return fmt.format(m["mean"])

        flops = t.get("flops_per_forward", {}).get("mean", float("nan")) / 1e6
        flops_s = f"{flops:.2f}" if math.isfinite(flops) else "n/a"
        _, invalid = _curve_health(by_gate[gate])
        verdicts = sorted({r["curve_verdict"] for r in by_gate[gate]})
        print(f"{gate:>6} {cell('params_total', '{:.0f}'):>10} "
              f"{flops_s:>9} "
              f"{cell('latency_ms_per_sample'):>10} "
              f"{cell('peak_mem_mb', '{:.2f}'):>9} "
              f"{invalid}/{len(by_gate[gate])} {'/'.join(verdicts):>10}")

    print(f"\n[e2] bad cases at the longest length (n_pairs={lengths[-1]}; "
          f"stats seed {STATS_SEED}, batch {settings.stats_batch}x"
          f"{settings.stats_batches}; raw cases in e2_error_cases.jsonl).  "
          f"retrieval = attention mass < 0.5 on the correct key (the model "
          f"never found it); readout = key found but the wrong value read "
          f"out - the split that tells apart 'cannot find' from 'cannot "
          f"use':")
    print(f"{'gate':>6} {'err_rate':>9} {'label':>7} {'struct':>7} "
          f"{'retrieval':>10} {'readout':>8} {'key_miss':>9} {'mass_key_err':>13}")
    for gate in GATES:
        t = _aggregate_runs(by_gate[gate])

        def ecell(key, fmt="{:.3f}"):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return fmt.format(m["mean"])

        print(f"{gate:>6} {ecell('err_error_rate'):>9} "
              f"{ecell('err_taxonomy_label_error'):>7} "
              f"{ecell('err_taxonomy_hard_structural'):>7} "
              f"{ecell('err_taxonomy_retrieval_miss'):>10} "
              f"{ecell('err_taxonomy_readout_wrong'):>8} "
              f"{ecell('err_errors_key_missed_share'):>9} "
              f"{ecell('err_attn_mass_on_key_error'):>13}")
    print("  most confident wrong samples per gate at the longest length:")
    for gate in GATES:
        for case in examples.get(gate, (0, []))[1]:
            print(f"  [{gate}] {_format_case(case)}")

    print("\n[e2] pre-registered contrasts:")
    clog = ContrastLog("e2")
    calib_runs = by_gate["calib"]
    calib_warn = _curve_warn(("calib", calib_runs))
    clog.print("calib - none", "H2a", "accuracy_at_longest",
               _paired_diff(by_gate["calib"], by_gate["none"],
                            "accuracy_at_longest"),
               warn=_curve_warn(("calib", by_gate["calib"]),
                                ("none", by_gate["none"])))
    clog.print("calib - none", "H2a", "accuracy_drop",
               _paired_diff(by_gate["calib"], by_gate["none"],
                            "accuracy_drop"),
               warn=_curve_warn(("calib", by_gate["calib"]),
                                ("none", by_gate["none"])))
    clog.print("calib raw entropy drift vs 0", "H2b", "entropy_drift",
               paired_bootstrap_ci([r["entropy_drift"] for r in calib_runs]),
               warn=calib_warn)
    clog.print("calib normalised entropy drift vs 0", "H2b",
               "entropy_norm_drift",
               paired_bootstrap_ci([r["entropy_norm_drift"]
                                    for r in calib_runs]),
               warn=calib_warn)
    if calib_runs[0]["gate_by_length"]:
        gate_drifts = [r["gate_by_length"][lengths[-1]]
                       - r["gate_by_length"][settings.n_pairs]
                       for r in calib_runs]
        clog.print("calib gate drift vs 0", "H2b", "gate_drift",
                   paired_bootstrap_ci(gate_drifts), warn=calib_warn)
    clog.finalize(out_dir, q=settings.fdr_q)
    _write_json(out_dir / "e2_summary.json",
                {g: {str(L): _mean_std([r["accuracy_by_length"][L]
                                        for r in by_gate[g]])
                     for L in lengths} for g in GATES})


def run_e3(settings: Settings, out_dir: pathlib.Path) -> None:
    _announce(("H3a", "H3b"))
    lr = _resolve_lr(settings, out_dir, "e3")
    print(f"[e3] grid gate x norm x layer_scale at lr={lr:g}, "
          f"test seed {TEST_SEED}, stats seed {STATS_SEED}")
    sha = _git_sha()
    runs = []
    for gate in ("query", "calib"):
        for norm in NORMS:
            for layer_scale in (None, 1e-2):
                for seed in settings.seeds:
                    model, params, metrics, curve = _train_and_eval(
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
                        "git_sha": sha, "gate": gate, "norm": norm,
                        "layer_scale": layer_scale, "seed": seed, "lr": lr,
                        "params": params, "accuracy": metrics["accuracy"],
                        "f1_macro": metrics["f1_macro"],
                        "gate_depth_ratio": gate_depth_ratio(stats),
                        "resid_norm_by_layer": resid.tolist(),
                        "resid_norm_last_over_first": float(
                            resid[-1] / resid[0].clamp_min(1e-12)),
                        **_cost_record(settings, model), **curve,
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
    clog = ContrastLog("e3")
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
                        choices=["baselines", "e1", "e2", "e3", "all"])
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

    print(f"[runner] only={args.only} quick={args.quick} "
          f"seeds={settings.seeds} steps={settings.steps} "
          f"lr={settings.lr if settings.lr is not None else 'tuned'} "
          f"fdr_q={settings.fdr_q:g} "
          f"device={settings.device} out={out_dir}")
    if args.only == "baselines":
        run_baselines(settings, out_dir)
    elif args.only == "e1":
        run_e1(settings, out_dir)
    elif args.only == "e2":
        run_e2(settings, out_dir)
    elif args.only == "e3":
        run_e3(settings, out_dir)
    else:
        run_all(settings, out_dir)
    print(f"\n[runner] done; records in {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
