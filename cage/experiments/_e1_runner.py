# cage/experiments/_e1_runner.py - E1 runner and report.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def run_e1(settings: Settings, out_dir: pathlib.Path) -> None:
    _announce(("H1a", "H1b", "H1c", "H1d", "H1e", "H1f", "H1g"))
    # HB1/HB3/HB4/HB5 hard gate: the baselines verdict must exist, be
    # current for this exact protocol and pass, before any grid trains
    # (their pre-registered semantics invalidate everything above them;
    # the marks are stamped into every record via _run_context).
    verdict = _enforce_baseline_verdict(settings, out_dir, "e1")
    _design_sensitivity_plan(settings, verdict, out_dir, "e1")
    _write_manifest(settings, out_dir, "e1")
    _audit_data_splits(settings, out_dir, "e1", enforce=True)
    _enforce_reproducibility(settings, out_dir, "e1")
    _enforce_overfit_gate(settings, out_dir, "e1")
    lr = _resolve_lr(settings, out_dir, "e1")
    runs = []
    examples = {}
    ablation_runs = {}
    cell_plan = _e1_cell_plan()
    # All E1 cells - the four gate variants and the calibration-term
    # ablation/active-control arms - are co-randomised in the same complete
    # blocks.  Running the ablation arms in a second grid would alias the
    # full-vs-ablated contrasts with whatever changed on the machine between
    # the two invocations; one schedule keeps the whole 2x2 and the placebo
    # comparable to the full calib cell block by block.
    e1_cells = cell_plan["cells"]
    e1_order = _blocked_run_order(
        e1_cells, settings.seeds, settings.run_order_seed,
        shuffle=settings.randomize_run_order)
    _write_run_order(settings, out_dir, "e1", e1_order, e1_cells,
                     settings.seeds)
    for cell, seed, block, position in e1_order:
        gate = cell_plan["gate_of"][cell]
        extra = {"e1_cell": cell, "run_block": block,
                 "run_position": position}
        variant = cell_plan["variant_of"].get(cell)
        if variant is not None:
            # The full calib cell is labelled too, so the audit can see that
            # the ablation contrasts vary exactly the calibration-term flag.
            extra["calib_variant"] = variant
        if cell in cell_plan["overrides"]:
            label = cell
            rec, errors = _e1_run(
                settings, seed, lr, gate, extra,
                **cell_plan["overrides"][label])
            for case in errors["cases"]:
                _write_result(out_dir / "e1_error_cases.jsonl",
                              {"experiment": "e1", **_stamp(settings),
                               "gate": gate, "ablation": label,
                               "seed": seed, "lr": lr, **case})
            _write_result(out_dir / "e1_runs.jsonl", rec)
            ablation_runs.setdefault(label, []).append(rec)
            continue
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
    ablation_flat = [r for rs in ablation_runs.values() for r in rs]
    _report_realized_run_order(
        out_dir, "e1", runs + ablation_flat, e1_order, e1_cells,
        settings.seeds, _e1_cell_of)
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
        except (TypeError, ValueError, RuntimeError, OverflowError):
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
        matched_digests = matched.get("stream_digest_by_seed") or {}
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
            matched_runs.append({
                "seed": seed, "accuracy": float(acc), "n": n,
                "correct_bitmask": mask,
                "stream_digest": matched_digests.get(str(seed)),
            })
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
            unmeasured = [d for d, row in by_depth.items()
                          if row.get("unmeasured")]
            if verdict.get("ladder_competent") is False:
                sat_txt = ("not determinable: no ladder rung is above "
                           "chance, so the ladder did not solve the task at "
                           "this step count - depth is unverified either way "
                           "(raise the step count and re-run `--only baselines`)")
            elif unmeasured:
                sat_txt = ("not determinable: rung(s) "
                           f"{', '.join(str(d) for d in unmeasured)} were "
                           "not measured, so the ladder cannot establish "
                           "whether depth is required")
            else:
                sat_txt = "not determinable (rule ceiling unknown)"
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

    print("\n[e1] H1a confound-controlled check (exploratory, outside the "
          "FDR family): partial r removes max-prob linearly within each "
          "layer, so the gate-entropy path is not credited merely because "
          "both track attention concentration; the correct/error columns "
          "are descriptive subgroup correlations (correctness is an "
          "outcome, not a pre-treatment covariate).  Mean±std over seeds; "
          "n_L = layers with a finite partial estimate:")
    print(f"{'gate':>6} {'r_entropy':>19} {'partial r|maxprob':>21} "
          f"{'n_L':>5} {'r_ent|correct':>19} {'n_correct':>10} "
          f"{'r_ent|error':>19} {'n_error':>9}")
    for gate in GATES:
        t = table[gate]

        def mcell(key):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return f"{m['mean']:.4f}±{m['std']:.4f}"

        def ncell(key):
            m = t.get(key)
            if not m or not math.isfinite(m["mean"]):
                return "n/a"
            return f"{int(round(m['mean']))}"

        print(f"{gate:>6} {mcell('gate_vs_entropy'):>19} "
              f"{mcell('gate_vs_entropy_partial_maxprob'):>21} "
              f"{ncell('gate_vs_entropy_partial_maxprob_layers'):>5} "
              f"{mcell('gate_vs_entropy_within_correct'):>19} "
              f"{ncell('n_correct'):>10} "
              f"{mcell('gate_vs_entropy_within_error'):>19} "
              f"{ncell('n_error'):>9}")

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

    # Calibration is the research object ("calibrated" gating), yet ECE and
    # the Brier score were recorded per run and never analysed.  Report the
    # per-arm levels (with bin-count sensitivity, so a single bin width
    # cannot carry the readout) and the paired calib-minus-control
    # differences.  Lower ECE/Brier is better; higher Brier skill is better.
    # This is exploratory: the pre-registered H1 decisions remain the FDR
    # family above, and a calibration difference is not substituted for the
    # accuracy/mechanism rows.
    calibration = {}
    print("\n[e1] calibration check (exploratory, outside the FDR family): "
          "ECE is reported at three bin counts; Rel/Res are the Murphy "
          "reliability/resolution terms of the binned Brier score (lower "
          "reliability is better, higher resolution is better); BSS = "
          "1 - Brier/uncertainty, so BSS>0 beats the base-rate forecast.  "
          "Mean±std over seeds:")
    print(f"{'arm':>6} {'ECE10':>15} {'ECE15':>15} {'ECE20':>15} "
          f"{'Brier':>15} {'Rel':>15} {'Res':>15} {'BSS':>15}")
    for gate in GATES:
        cells = []
        for metric in ("ece_10bin", "ece_15bin", "ece_20bin", "brier",
                       "brier_reliability", "brier_resolution",
                       "brier_skill_score"):
            agg = _mean_std([r.get(metric) for r in by_gate[gate]])
            calibration[f"{gate}:{metric}"] = agg
            cells.append("n/a" if not _finite(agg["mean"])
                         else f"{agg['mean']:.3f}±{agg['std']:.3f}")
        print(f"{gate:>6}" + "".join(f"{c:>15}" for c in cells))
    print("  paired calibration differences (calib - baseline; a negative "
          "ECE/Brier difference is better, a positive BSS difference is "
          "better):")
    print(f"{'contrast':>34} {'readout':>10} {'mean [95% CI]':>30} {'n':>5}")
    for baseline in ("none", "query", "null"):
        for metric, label in (("ece_15bin", "ECE15"),
                              ("brier", "Brier"),
                              ("brier_skill_score", "BSS")):
            ci = _paired_diff(by_gate["calib"], by_gate[baseline], metric)
            calibration[f"calib-{baseline}:{metric}"] = {
                "mean": ci.get("mean"), "lo": ci.get("lo"),
                "hi": ci.get("hi"), "n": ci.get("n"),
                "seeds_used": ci.get("seeds_used"),
                "seeds_dropped": ci.get("seeds_dropped"),
                "controlled": (ci.get("controlled") or {}).get(
                    "controlled"),
            }
            print(f"{'calib - ' + baseline:>34} {label:>10} "
                  f"{_fmt_ci(ci):>30} {int(ci.get('n') or 0):>5}")
    table["_calibration_check"] = calibration

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
    # A main effect alone cannot distinguish an additive two-term gate from
    # one whose terms substitute for each other: the average of the
    # concentration effect can be positive while its size depends on whether
    # the length term is on.  The interaction is the missing row of the
    # complete 2x2 (the E2 design already reads it as H2e), so it is
    # registered here as a two-sided additivity check with its own
    # undirected registry keys.  It is not a directional claim: either sign
    # is admissible, but a practically negligible interaction supports the
    # additive reading while a large one says the "each term carries its own
    # contribution" story must be stated conditionally.
    print(f"\n[e1] H1d factorial interaction (length x concentration, "
          f"two-sided additivity check): (full - len_only) - "
          f"(ent_only - no_terms), i.e. the concentration effect with the "
          f"length term on minus the concentration effect with it off.  The "
          f"term main effects above are averages over this interaction; a "
          f"large interaction means the terms are not interchangeable and "
          f"each component's contribution must be reported conditional on "
          f"the other:")
    for metric in ("accuracy", "gate_margin", "abs_gate_vs_entropy"):
        ci = _paired_did(
            factorial_cells["full"], factorial_cells["calib_len_only"],
            factorial_cells["calib_ent_only"],
            factorial_cells["calib_no_terms"], metric)
        clog.print("length x concentration interaction (2x2)", "H1d", metric,
                   ci, direction_key=f"interaction_{metric}",
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
    table["_counterfactual"] = _counterfactual_companion(
        out_dir, "e1", by_gate, reference="none")
    table["_provenance"] = _stamp(settings)
    if complexity_benchmark is not None:
        table["_complexity_benchmark"] = complexity_benchmark
    table["_class_frequency_companions"] = metric_robustness
    table["_lr_sensitivity"] = lr_sensitivity
    table["_per_variant_tuning"] = per_variant_tuning
    _write_json(out_dir / "e1_summary.json", table)
