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
   gate variants so the E1/E3 comparisons stay controlled.
3. **Honest evaluation.**  Tuning uses the validation seed only.  Final
   numbers are reported on a separate test seed that is never used for
   selection.  Every key claim is reported as mean +- std over
   independent training seeds together with a paired bootstrap 95% CI
   of the difference (``cage/metrics.py``); a claim is only read as
   supported when that CI excludes 0.

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
    rule_naive_last_match_stats,
    rule_pair_lookup_stats,
    train_linear_probe,
    tune_linear_probe,
)
from .data import length_sweep_lengths, make_batch
from .metrics import paired_bootstrap_ci
from .model import GATES, NORMS, build
from .train import (
    collect_stats,
    error_detection_aucs,
    eval_metrics,
    false_concentration_analysis,
    gate_depth_ratio,
    gate_stat_correlations,
    train_model,
)

# Seeds with fixed roles (independent of every training seed):
#   VAL_SEED   - model selection only (learning-rate tuning, baseline checks)
#   TEST_SEED  - the reported numbers; never used for any decision
#   STATS_SEED - mechanism statistics (matches the collect_stats default)
VAL_SEED = 123
TEST_SEED = 2024
STATS_SEED = 7

HYPO_IDS = ("HB1", "HB2", "H1a", "H1b", "H1c", "H2a", "H2b", "H3a", "H3b")

HYPOTHESES = {
    "HB1": {
        "experiment": "baselines",
        "question": "Is the recall task well posed with a known solution?",
        "hypothesis": "The pair structure admits an exact parameter-free "
                      "lookup rule because keys are distinct per sequence.",
        "prediction": "rule accuracy ~ 1.0 (ceiling) and unique_key_rate = 1.0",
        "decision": "rule accuracy < 0.99 indicates a broken task/data "
                    "construction and invalidates everything above it",
    },
    "HB2": {
        "experiment": "baselines",
        "question": "Does the task require content-based lookup rather than "
                    "a simple linear decision?",
        "hypothesis": "A linear model on bag-of-token counts cannot express "
                      "key-value lookup; its accuracy stays near chance "
                      "1/(vocab-2) even after tuning.",
        "prediction": "tuned linear probe accuracy <= a few x chance",
        "decision": "a clearly above-chance probe would mean answer "
                    "information leaks into the bag-of-tokens features",
    },
    "H1a": {
        "experiment": "e1",
        "question": "Does the calibrated gate use attention concentration / "
                    "length information beyond the query?",
        "hypothesis": "The query-only gate has no access to the attention "
                      "distribution, so |gate_vs_entropy| is clearly larger "
                      "for calib than for query.",
        "prediction": "|gate_vs_entropy|(calib) - |gate_vs_entropy|(query) > 0",
        "decision": "paired bootstrap 95% CI of the seed differences "
                    "excludes 0",
    },
    "H1b": {
        "experiment": "e1",
        "question": "Does the calibrated gate 'doubt' itself on false "
                    "concentration?",
        "hypothesis": "On samples with high max-prob but wrong answers the "
                      "calibrated gate closes more than on high-max-prob "
                      "correct samples, more so than the query gate.",
        "prediction": "gate_margin = gate_on_correct - gate_on_false_conc "
                      "is larger for calib than for query",
        "decision": "paired bootstrap 95% CI excludes 0",
    },
    "H1c": {
        "experiment": "e1",
        "question": "Does the explicit null slot suppress writes when the "
                    "model is wrong?",
        "hypothesis": "null_mass is higher on errors than on correct "
                      "answers (the null slot absorbs the write).",
        "prediction": "null_mass_error_margin > 0",
        "decision": "paired bootstrap 95% CI excludes 0",
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
        "decision": "paired bootstrap 95% CI of the differences excludes 0",
    },
    "H2b": {
        "experiment": "e2",
        "question": "Do the attention statistics drift with length as the "
                    "calibration assumes?",
        "hypothesis": "Raw entropy grows with log(T) while max-prob and the "
                      "calibrated gate stay comparably flat.",
        "prediction": "entropy@longest > entropy@base; gate drift smaller "
                      "in magnitude than entropy drift",
        "decision": "sign-consistency across seeds",
    },
    "H3a": {
        "experiment": "e3",
        "question": "Does dynamic normalisation make deep layers write more "
                    "into the residual stream?",
        "hypothesis": "seednorm (per-token dynamic scale) raises the "
                      "deep/early gate ratio relative to pre-LN.",
        "prediction": "gate_depth_ratio(seednorm) > gate_depth_ratio(pre) "
                      "and deep residual norms grow more",
        "decision": "paired bootstrap 95% CI excludes 0",
    },
    "H3b": {
        "experiment": "e3",
        "question": "Does LayerScale change how strongly deep layers write?",
        "hypothesis": "LayerScale (1e-2) damps the initial write strength "
                      "and thereby changes the residual-norm profile and "
                      "the gate depth ratio.",
        "prediction": "residual-norm growth and/or gate_depth_ratio shift "
                      "between LayerScale off and on",
        "decision": "paired bootstrap 95% CI excludes 0",
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
    by_seed = {r["seed"]: r for r in runs_b}
    diffs = []
    for ra in runs_a:
        rb = by_seed.get(ra["seed"])
        if rb is None or metric not in ra or metric not in rb:
            continue
        if _finite(ra[metric]) and _finite(rb[metric]):
            diffs.append(float(ra[metric]) - float(rb[metric]))
    return paired_bootstrap_ci(diffs)


def _fmt_ci(ci) -> str:
    if ci["n"] == 0 or not math.isfinite(ci["mean"]):
        return "n/a"
    supported = (ci["lo"] > 0.0) or (ci["hi"] < 0.0)
    return (f"{ci['mean']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}] "
            f"n={ci['n']} {'CI excludes 0' if supported else 'CI includes 0'}")


def _print_contrast(rule, hyp, metric, ci):
    print(f"  [{hyp}] {rule} ({metric}): {_fmt_ci(ci)}")


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


def _train_and_eval(settings: Settings, seed: int, lr: float, **model_overrides):
    torch.manual_seed(seed)
    model = build(_model_cfg(settings, **model_overrides), compile=False)
    params = sum(p.numel() for p in model.parameters())
    train_model(model, steps=settings.steps, batch=settings.batch,
                n_pairs=settings.n_pairs, lr=lr, vocab=settings.vocab,
                device=settings.device, log_every=0, seed=seed)
    metrics = eval_metrics(model, settings.n_pairs, settings.vocab,
                           settings.device, batch=settings.eval_batch,
                           batches=settings.eval_batches, seed=TEST_SEED)
    return model, params, metrics


def tune_shared_recipe(settings: Settings):
    """Tune the learning rate on the plain ``none`` baseline.

    Selection uses the validation seed only (up to two training seeds);
    the winning rate is then frozen for *every* variant, so a gate variant
    can never win via a luckier optimizer setting - the E1/E3 comparisons
    remain controlled.
    """
    tune_seeds = settings.seeds[:max(1, min(2, len(settings.seeds)))]
    table = []
    for lr in settings.lr_grid:
        accs = []
        for seed in tune_seeds:
            _, _, metrics = _train_and_eval(settings, seed, lr, gate="none")
            accs.append(metrics["accuracy"])
        mean = sum(accs) / len(accs)
        var = (sum((a - mean) ** 2 for a in accs) / len(accs)) if accs else 0.0
        table.append({"lr": lr, "val_accuracy_mean": mean,
                      "val_accuracy_std": var ** 0.5, "n_seeds": len(accs)})
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
            print(f"    lr={row['lr']:<8g} "
                  f"val_acc={row['val_accuracy_mean']:.4f} "
                  f"+- {row['val_accuracy_std']:.4f}")
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


def run_baselines(settings: Settings, out_dir: pathlib.Path) -> None:
    _announce(("HB1", "HB2"))
    sha = _git_sha()
    record = {"experiment": "baselines", "hypotheses": ["HB1", "HB2"],
              "git_sha": sha, "seed": TEST_SEED}

    balance = _answer_balance(settings)
    print(f"[baselines] answer-value balance over {balance['n_samples']} "
          f"samples: {balance['n_classes_present']} classes present, share "
          f"min={balance['min_class_share']:.5f} max={balance['max_class_share']:.5f} "
          f"(uniform={balance['expected_uniform_share']:.5f})")
    _write_json(out_dir / "baselines_balance.json", balance)

    for name, fn in (
        ("rule_pair_lookup", rule_pair_lookup_stats),
        ("rule_naive_last_match", rule_naive_last_match_stats),
        ("copy_last_token", copy_last_token_stats),
    ):
        stats = fn(settings.n_pairs, settings.vocab, settings.device,
                   batch=settings.eval_batch, batches=settings.eval_batches,
                   seed=TEST_SEED)
        print(f"[baselines] {name}: accuracy={stats['accuracy']:.4f} "
              f"(n={stats['n']})")
        _write_result(out_dir / "baselines_runs.jsonl",
                      {**record, "name": name, **stats})

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
    print(f"[baselines] tuned linear probe (test seed {TEST_SEED}): "
          f"accuracy={summary['mean']:.4f} +- {summary['std']:.4f} "
          f"(chance={chance:.4f}, n_seeds={summary['n']})")


def run_e1(settings: Settings, out_dir: pathlib.Path) -> None:
    _announce(("H1a", "H1b", "H1c"))
    lr = _resolve_lr(settings, out_dir, "e1")
    sha = _git_sha()
    runs = []
    for gate in GATES:
        for seed in settings.seeds:
            model, params, metrics = _train_and_eval(settings, seed, lr,
                                                     gate=gate)
            stats = collect_stats(model, settings.n_pairs, settings.vocab,
                                  settings.device, batch=settings.stats_batch,
                                  batches=settings.stats_batches,
                                  seed=STATS_SEED)
            corr = gate_stat_correlations(stats)
            fc = false_concentration_analysis(stats)
            ed = error_detection_aucs(stats)
            rec = {
                "experiment": "e1", "hypotheses": ["H1a", "H1b", "H1c"],
                "git_sha": sha, "gate": gate, "seed": seed, "lr": lr,
                "params": params, **metrics, **corr, **fc, **ed,
            }
            if _finite(rec.get("gate_vs_entropy")):
                rec["abs_gate_vs_entropy"] = abs(rec["gate_vs_entropy"])
            if _finite(fc.get("gate_on_correct")) and \
                    _finite(fc.get("gate_on_false_conc")):
                rec["gate_margin"] = fc["gate_on_correct"] - fc["gate_on_false_conc"]
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

    print("\n[e1] pre-registered contrasts (calib vs query/none):")
    _print_contrast("calib - none", "H1a", "accuracy",
                    _paired_diff(by_gate["calib"], by_gate["none"], "accuracy"))
    _print_contrast("calib - query", "H1a", "abs_gate_vs_entropy",
                    _paired_diff(by_gate["calib"], by_gate["query"],
                                 "abs_gate_vs_entropy"))
    _print_contrast("calib - query", "H1b", "gate_margin",
                    _paired_diff(by_gate["calib"], by_gate["query"],
                                 "gate_margin"))
    null_margins = [r["null_mass_error_margin"] for r in by_gate["null"]
                    if _finite(r.get("null_mass_error_margin"))]
    _print_contrast("null margin vs 0", "H1c", "null_mass_error_margin",
                    paired_bootstrap_ci(null_margins))
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
    for gate in GATES:
        for seed in settings.seeds:
            model, params, _ = _train_and_eval(settings, seed, lr, gate=gate)
            rec = {"experiment": "e2", "hypotheses": ["H2a", "H2b"],
                   "git_sha": sha, "gate": gate, "seed": seed, "lr": lr,
                   "params": params, "accuracy_by_length": {},
                   "entropy_by_length": {}, "maxprob_by_length": {},
                   "gate_by_length": {}}
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

    print("\n[e2] pre-registered contrasts:")
    _print_contrast("calib - none", "H2a", "accuracy_at_longest",
                    _paired_diff(by_gate["calib"], by_gate["none"],
                                 "accuracy_at_longest"))
    _print_contrast("calib - none", "H2a", "accuracy_drop",
                    _paired_diff(by_gate["calib"], by_gate["none"],
                                 "accuracy_drop"))
    calib_runs = by_gate["calib"]
    _print_contrast("calib entropy drift vs 0", "H2b", "entropy_drift",
                    paired_bootstrap_ci([r["entropy_drift"]
                                         for r in calib_runs]))
    if calib_runs[0]["gate_by_length"]:
        gate_drifts = [r["gate_by_length"][lengths[-1]]
                       - r["gate_by_length"][settings.n_pairs]
                       for r in calib_runs]
        _print_contrast("calib gate drift vs 0", "H2b", "gate_drift",
                        paired_bootstrap_ci(gate_drifts))
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
                    model, params, metrics = _train_and_eval(
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
                    }
                    _write_result(out_dir / "e3_runs.jsonl", rec)
                    runs.append(rec)

    print(f"\n[e3] {'gate':>5} {'norm':>9} {'ls':>5} {'accuracy':>16} "
          f"{'gate_depth_ratio':>18} {'resid_growth':>14}")
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
                ls_label = "off" if layer_scale is None else f"{layer_scale:g}"
                print(f"{gate:>5} {norm:>9} {ls_label:>5} "
                      f"{acc['mean']:>8.4f}±{acc['std']:<7.4f} "
                      f"{ratio['mean']:>9.4f}±{ratio['std']:<8.4f} "
                      f"{growth['mean']:>7.3f}±{growth['std']:<6.3f}")

    print("\n[e3] pre-registered contrasts:")
    for gate in ("query", "calib"):
        for layer_scale in (None, 1e-2):
            a = index[(gate, "seednorm", layer_scale)]
            b = index[(gate, "pre", layer_scale)]
            _print_contrast(f"{gate}: seednorm - pre (ls={layer_scale})",
                            "H3a", "gate_depth_ratio",
                            _paired_diff(a, b, "gate_depth_ratio"))
    for gate in ("query", "calib"):
        for norm in NORMS:
            a = index[(gate, norm, 1e-2)]
            b = index[(gate, norm, None)]
            _print_contrast(f"{gate}/{norm}: ls=1e-2 - off", "H3b",
                            "resid_norm_last_over_first",
                            _paired_diff(a, b, "resid_norm_last_over_first"))
    for norm in NORMS:
        a = index[("calib", norm, None)]
        b = index[("query", norm, None)]
        _print_contrast(f"{norm}: calib - query", "H3a", "accuracy",
                        _paired_diff(a, b, "accuracy"))
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
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)

    settings = Settings.quick(args.device) if args.quick else Settings(
        device=args.device)
    if args.seeds is not None:
        settings.seeds = tuple(range(args.seeds))
    if args.steps is not None:
        settings.steps = args.steps
    if args.lr is not None:
        settings.lr = args.lr
    if args.d_model is not None:
        settings.d_model = args.d_model
    if args.n_layers is not None:
        settings.n_layers = args.n_layers
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "hypotheses.json", HYPOTHESES)

    print(f"[runner] only={args.only} quick={args.quick} "
          f"seeds={settings.seeds} steps={settings.steps} "
          f"lr={settings.lr if settings.lr is not None else 'tuned'} "
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
