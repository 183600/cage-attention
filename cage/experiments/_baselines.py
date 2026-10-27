# cage/experiments/_baselines.py - Baseline experiment runner.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
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
                       # ``config`` is the full resolved-Settings context
                       # merged in by ``record``; overwriting it with a
                       # label made the row's own setup hash unverifiable.
                       # Keep the arm label in its own field instead.
                       "sanity_config": label,
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
        table, grid_keys=("lr", "steps"), full_rows=table)
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
        unmeasured = [
            d for d, row in ((verdict.get("depth_ladder") or {})
                             .get("by_depth") or {}).items()
            if row.get("unmeasured")]
        if not verdict["ladder_competent"]:
            hb7_txt = "not determinable (no competent rung)"
        elif unmeasured:
            hb7_txt = ("not determinable (unmeasured rung(s) "
                       f"{', '.join(str(d) for d in unmeasured)})")
        else:
            hb7_txt = "not determinable (rule ceiling unknown)"
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
