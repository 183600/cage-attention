# cage/experiments/_e1.py - E1 cell plan, per-variant tuning, lr sensitivity and single-run helper.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
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

_E1_ABLATION_LABELS = tuple(
    label for label, _ in _E1_CALIB_ABLATION + _E1_CALIB_PLACEBO)


def _e1_cell_plan() -> dict:
    """The E1 cells that enter one shared complete-block schedule.

    E1 has two conceptually different groups of conditions: the four gate
    variants (the pre-registered E1 family) and the calibration-term
    ablation / active-control arms that complete the 2x2 and the placebo.
    Scheduling them as two separate grids would make every ablation
    contrast compare a full-calib run from the first grid with an ablated
    run from the second: any slow machine drift between the two invocations
    would then be aliased with the ablation factor, exactly the confound the
    complete-block randomisation exists to remove.  They therefore share
    one seed block and one frozen schedule.

    ``e1_cell`` is the schedule label.  The full-calib cell keeps the gate
    label ``"calib"`` (its result-file row still carries
    ``calib_variant="full"``), and every ablation/placebo cell is labelled
    by its ``calib_variant`` value.  ``overrides`` maps each non-gate cell
    to the resolved ``Cfg`` flags it varies, and ``gate_of`` says which gate
    implementation that cell builds, so the runner dispatches from one
    single source of truth instead of reconstructing the mapping inline.
    """
    cells = tuple(GATES) + _E1_ABLATION_LABELS
    overrides = {
        label: dict(flags)
        for label, flags in _E1_CALIB_ABLATION + _E1_CALIB_PLACEBO}
    gate_of = {gate: gate for gate in GATES}
    gate_of.update({label: "calib" for label in _E1_ABLATION_LABELS})
    variant_of = {"calib": "full"}
    variant_of.update({label: label for label in _E1_ABLATION_LABELS})
    return {
        "cells": cells,
        "overrides": overrides,
        "gate_of": gate_of,
        "variant_of": variant_of,
    }


def _e1_cell_of(run) -> str | None:
    """Recover an E1 record's schedule cell.

    New records stamp ``e1_cell`` directly.  Legacy records predate the
    shared schedule, so fall back to the resolved labels: ``calib`` with
    ``calib_variant="full"`` is the main-grid cell, an ablation label is its
    own cell, and every other gate is its own cell.
    """
    if not isinstance(run, Mapping):
        return None
    cell = run.get("e1_cell")
    if isinstance(cell, str) and cell:
        return cell
    variant = run.get("calib_variant")
    if variant == "full":
        return "calib"
    if isinstance(variant, str) and variant:
        return variant
    gate = run.get("gate")
    return gate if isinstance(gate, str) and gate else None


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
    if not isinstance(reference_runs, Mapping):
        reference_runs = {}
    reference_lr, reason = _validated_reference_lr(reference_lr)
    if reason is not None:
        return {"available": False, "reason": reason}
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
            eligible, grid_keys=(key,), full_rows=rows)
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
    if not isinstance(reference_runs, Mapping):
        reference_runs = {}
    reference_lr, reason = _validated_reference_lr(reference_lr)
    if reason is not None:
        return {"available": False, "reason": reason}
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
    # Learned calibration coefficients, stamped with the same readout E2
    # uses (``_calibration_coefficients``): at the fixed training length
    # the mechanism claim is about the gate's coefficients (a > 0,
    # c_effective > 0), so the E1 record carries them too instead of
    # leaving the manipulation check E2-only.  Readout, not a new decision
    # rule - the pre-registered H1 contrasts are unchanged.
    coefficients = _calibration_coefficients(model)
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
    rec["calib_coefficients"] = coefficients
    for coeff_key in ("a_mean", "c_mean", "c_effective_mean",
                      "scale_mean", "tau_effective_mean"):
        value = coefficients.get(coeff_key)
        if _finite(value):
            rec[coeff_key] = float(value)
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
