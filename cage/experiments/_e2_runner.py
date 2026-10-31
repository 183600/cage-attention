# cage/experiments/_e2_runner.py - E2 runner and report.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _stamp_position_extrapolation_control(rec, model, settings, lengths,
                                          eval_seed):
    """Eval-only control separating position-vector extrapolation from
    length effects (exploratory companion; the H2 decisions read the raw
    sweep).

    The model uses learned absolute position embeddings and E2 trains at a
    fixed length, so every sweep position beyond the trained sequence
    length carries an embedding that was never updated by training.  A
    length-related accuracy drop therefore mixes (a) attention/gate
    behaviour at longer ranges with (b) untrained position vectors.  This
    control re-evaluates the same sealed stream with the extrapolated
    positions folded back into the trained range (``fold``) or removed
    (``zero``): if folding recovers most of the drop, the degradation is
    mostly (b) - a position-encoding confound - rather than the mechanism
    under test.
    """
    trained_len = 2 * settings.n_pairs + 2
    by_length = {}
    for length in lengths:
        seq_len = 2 * length + 2
        if seq_len <= trained_len:
            continue
        learned = rec["accuracy_by_length"].get(length)
        entry = {"seq_len": seq_len, "learned": learned}
        for mode in ("fold", "zero"):
            model.set_pos_extrapolation(mode, trained_len)
            try:
                entry[mode] = eval_accuracy(
                    model, length, settings.vocab, settings.device,
                    batch=settings.eval_batch,
                    batches=settings.eval_batches, seed=eval_seed)
            finally:
                model.set_pos_extrapolation("learned", None)
            entry[f"{mode}_minus_learned"] = (
                entry[mode] - learned
                if _finite(learned) and _finite(entry[mode])
                else float("nan"))
        by_length[length] = entry
    if not by_length:
        return
    longest = max(by_length)
    rec["pos_extrapolation_control"] = {
        "trained_seq_len": trained_len,
        "by_length": {str(k): v for k, v in by_length.items()},
        "longest": longest,
        "fold_recovery_at_longest": by_length[longest].get(
            "fold_minus_learned"),
        "zero_recovery_at_longest": by_length[longest].get(
            "zero_minus_learned"),
        "modes": ("learned", "fold", "zero"),
        "note": ("exploratory companion: same sealed eval stream, "
                 "eval-time position handling only; the pre-registered "
                 "H2 decision rows remain the raw sweep readouts"),
    }


def run_e2(settings: Settings, out_dir: pathlib.Path) -> None:
    _announce(("H2a", "H2b", "H2c", "H2d", "H2e", "H2f", "H2g"))
    # HB1/HB3/HB4/HB5 hard gate (see run_e1).
    verdict = _enforce_baseline_verdict(settings, out_dir, "e2")
    _design_sensitivity_plan(settings, verdict, out_dir, "e2")
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
    cells = tuple(label for label, *_ in _E2_VARIANTS)
    by_label = {label: (label, gate, record_extra, model_overrides)
                for label, gate, record_extra, model_overrides
                in _E2_VARIANTS}
    order = _blocked_run_order(
        cells, settings.seeds, settings.run_order_seed,
        shuffle=settings.randomize_run_order)
    _write_run_order(settings, out_dir, "e2", order, cells,
                     settings.seeds)
    for label, seed, block, position in order:
        label, gate, record_extra, model_overrides = by_label[label]
        record_extra = {**record_extra, "run_block": block,
                        "run_position": position}
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
               "ece_by_length": {}, "brier_by_length": {},
               "brier_skill_by_length": {},
               "difficulty_standardized_accuracy_by_length": {},
               "difficulty_standardized_coverage_by_length": {},
               "difficulty_standardized_effective_n_by_length": {},
               "difficulty_standardized_min_stratum_n_by_length": {},
               "difficulty_mix_tv_by_length": {},
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
                                    seed=eval_seed,
                                    train_association_codes=getattr(
                                        settings,
                                        "_train_association_codes", None),
                                    train_association_counts=getattr(
                                        settings,
                                        "_train_association_counts", None)))
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
            # Calibration drift is a separate failure mode from accuracy
            # drift: a model can keep its accuracy at long lengths while
            # becoming systematically overconfident.  Read the same
            # eval_metrics fields the base length already carries.
            rec["ece_by_length"][length] = metrics.get(
                "ece_15bin", float("nan"))
            rec["brier_by_length"][length] = metrics.get(
                "brier", float("nan"))
            rec["brier_skill_by_length"][length] = metrics.get(
                "brier_skill_score", float("nan"))
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
        # Within-item length extension.  The independent-stream sweep above
        # changes both length and the item/retrieval-position composition;
        # this paired probe keeps
        # each base item's query/answer/pair index and only appends fresh
        # distractors (nested across the sweep).  It is the counterfactual
        # post-stratification can only approximate, and it uses data already
        # drawn from the sealed base stream.  Exploratory companion: H2's
        # pre-registered decision still reads the raw sweep.
        _stamp_linked_length_extension(rec, model, settings, lengths, eval_seed)
        # Difficulty-composition control.  A length sweep is not a pure
        # length manipulation: more pairs per sequence also raise the
        # chance that the answer value repeats and that the query token
        # appears later as a value, so raw accuracy at a longer length
        # mixes the length effect with a harder data mix.  Re-weight
        # every length's per-stratum accuracy to the *base-length*
        # stratum mix (post-stratification on a pre-treatment
        # covariate).  This is an exploratory robustness companion;
        # the pre-registered H2 decision rows remain the raw readouts.
        _stamp_difficulty_standardization(
            rec, settings.n_pairs, lengths)
        # Position-embedding extrapolation control: untrained position
        # vectors beyond the training length are a confound of the raw
        # length sweep; the fold/zero re-evaluations quantify how much of
        # the drop they explain (exploratory companion, same stream).
        _stamp_position_extrapolation_control(
            rec, model, settings, lengths, eval_seed)
        # Mechanism manipulation check for H2b (no new run/grid point):
        # the trained length and concentration coefficients say whether
        # the gate actually implements the explicit log-T / entropy
        # correction the hypothesis names.
        coefficients = _calibration_coefficients(model)
        rec["calib_coefficients"] = coefficients
        rec["concentration_signal"] = coefficients.get(
            "concentration_signal")
        if label in ("calib", _CALIB_RAW_ENT_CELL):
            _stamp_calib_term_knockout(
                rec, model, settings, [settings.n_pairs, lengths[-1]],
                eval_seed)
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
    _report_realized_run_order(
        out_dir, "e2", runs, order, cells, settings.seeds,
        lambda run: run.get("variant"))

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

    print(f"\n[e2] within-item linked length extension (nested: each base "
          f"item keeps its query/answer/pair index and only gains "
          f"distractors; retention=P(correct@L|correct@base), rescue="
          f"P(correct@L|wrong@base); exploratory companion, the H2 decision "
          f"still reads the raw sweep):")
    print(f"{'arm':>16} {'raw_drop':>17} {'linked_drop':>17} "
          f"{'retain@long':>14} {'rescue@long':>14} {'net@long':>17} "
          f"{'raw-linked':>15} {'p<.05':>7}")

    def _fmt_cell(agg, signed=False):
        if not _finite(agg["mean"]):
            return "n/a"
        width = "+.3f" if signed else ".3f"
        return f"{format(agg['mean'], width)}±{agg['std']:.3f}"

    for label, *_ in _E2_VARIANTS:
        rs = by_variant[label]
        raw_drop = _mean_std([r.get("accuracy_drop") for r in rs])
        linked_drop = _mean_std([r.get("linked_accuracy_drop") for r in rs])
        retain = _mean_std([r.get("linked_retention_at_longest") for r in rs])
        rescue = _mean_std([r.get("linked_rescue_at_longest") for r in rs])
        net = _mean_std([r.get("linked_net_change_at_longest") for r in rs])
        composition = _mean_std(
            [r.get("linked_composition_effect_at_longest") for r in rs])
        sig = sum(
            1 for r in rs
            if _finite(r.get("linked_mcnemar_p_at_longest"))
            and float(r["linked_mcnemar_p_at_longest"]) < 0.05)
        print(f"{label:>16} {_fmt_cell(raw_drop):>17} "
              f"{_fmt_cell(linked_drop):>17} {_fmt_cell(retain):>14} "
              f"{_fmt_cell(rescue):>14} {_fmt_cell(net, True):>17} "
              f"{_fmt_cell(composition, True):>15} "
              f"{sig:>3}/{len(rs):<3}")
    linked_reproduced = sum(
        1 for rs in by_variant.values() for r in rs
        if r.get("linked_base_matches_raw"))
    print(f"  linked base reproduces the raw base accuracy in "
          f"{linked_reproduced}/{sum(len(rs) for rs in by_variant.values())} "
          f"runs; a linked drop smaller than raw means composition inflated "
          f"the raw length loss, a linked drop larger than raw means an "
          f"easier composition masked a within-item length loss, and a drop "
          f"present in both is a length effect.")

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

    # Accuracy drift is not the only length-generalisation failure mode: a
    # model can keep its accuracy while its confidence becomes
    # systematically wrong.  Read the calibration fields eval_metrics
    # already computes at every sweep length (no new runs).  Exploratory
    # companion; H2a/H2b remain the decisions.
    print(f"\n[e2] calibration drift across the length sweep (ECE15 and "
          f"Brier; lower is better, BSS>0 beats the base-rate forecast; "
          f"mean±std over seeds).  'ECE drift' = longest - training length "
          f"(positive = more miscalibrated at the extrapolated length):")
    print(f"{'arm':>16} {'ECE@base':>15} {'ECE@long':>15} "
          f"{'ECE drift':>15} {'Brier@base':>15} {'Brier@long':>15} "
          f"{'BSS@long':>15}")
    calibration_by_length = {}
    for label, *_ in _E2_VARIANTS:
        rows = by_variant[label]

        def _length_value(r, metric, length):
            values = r.get(metric)
            if not isinstance(values, Mapping):
                return float("nan")
            return values.get(length)

        def _at(metric, length):
            return _mean_std([_length_value(r, metric, length)
                              for r in rows])

        def _cell(agg):
            if not _finite(agg["mean"]):
                return "n/a"
            return f"{agg['mean']:.3f}±{agg['std']:.3f}"

        def _ece_drift(r):
            base = _length_value(r, "ece_by_length", settings.n_pairs)
            long = _length_value(r, "ece_by_length", lengths[-1])
            if not (_finite(base) and _finite(long)):
                return float("nan")
            return float(long) - float(base)

        ece_base = _at("ece_by_length", settings.n_pairs)
        ece_long = _at("ece_by_length", lengths[-1])
        ece_drift = _mean_std([_ece_drift(r) for r in rows])
        brier_base = _at("brier_by_length", settings.n_pairs)
        brier_long = _at("brier_by_length", lengths[-1])
        bss_long = _at("brier_skill_by_length", lengths[-1])
        print(f"{label:>16} {_cell(ece_base):>15} {_cell(ece_long):>15} "
              f"{_cell(ece_drift):>15} {_cell(brier_base):>15} "
              f"{_cell(brier_long):>15} {_cell(bss_long):>15}")
        calibration_by_length[label] = {
            "ece_base": ece_base, "ece_long": ece_long,
            "ece_drift": ece_drift, "brier_base": brier_base,
            "brier_long": brier_long, "bss_long": bss_long,
        }
    print("  (exploratory: a length-stable accuracy curve with a positive "
          "ECE drift means the arm becomes overconfident even where it still "
          "answers correctly; the H2 decision rows above are unchanged.)")

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
            warn=reverse_warn, registry_key=bound_key)
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
    difficulty_companions = _e2_difficulty_companions(by_variant, lengths)
    term_knockout = _e2_term_knockout_companion(by_variant, lengths)
    _write_json(out_dir / "e2_term_knockout.json", term_knockout)
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
    e2_summary["_calibration_by_length"] = calibration_by_length
    e2_summary["_per_variant_tuning"] = per_variant_tuning
    e2_summary["_efficiency"] = length_efficiency
    e2_summary["_class_frequency_companions"] = endpoint_companions
    e2_summary["_difficulty_companions"] = difficulty_companions
    e2_summary["_term_knockout"] = term_knockout
    e2_summary["_difficulty_standardized"] = {
        label: {
            str(L): _mean_std([
                _length_entry(
                    r.get("difficulty_standardized_accuracy_by_length"), L)
                for r in by_variant[label]])
            for L in lengths}
        for label, *_ in _E2_VARIANTS}
    e2_summary["_difficulty_standardized_coverage"] = {
        label: {
            str(L): _mean_std([
                _length_entry(
                    r.get("difficulty_standardized_coverage_by_length"), L)
                for r in by_variant[label]])
            for L in lengths}
        for label, *_ in _E2_VARIANTS}
    e2_summary["_difficulty_mix_tv"] = {
        label: {
            str(L): _mean_std([
                _length_entry(r.get("difficulty_mix_tv_by_length"), L)
                for r in by_variant[label]])
            for L in lengths}
        for label, *_ in _E2_VARIANTS}
    e2_summary["_linked_length_extension"] = {
        label: {
            "accuracy_by_length": {
                str(L): _mean_std([
                    _length_entry(r.get("linked_accuracy_by_length"), L)
                    for r in by_variant[label]])
                for L in lengths},
            "raw_accuracy_drop": _mean_std(
                [r.get("accuracy_drop") for r in by_variant[label]]),
            "linked_accuracy_drop": _mean_std(
                [r.get("linked_accuracy_drop") for r in by_variant[label]]),
            "retention_at_longest": _mean_std(
                [r.get("linked_retention_at_longest")
                 for r in by_variant[label]]),
            "rescue_at_longest": _mean_std(
                [r.get("linked_rescue_at_longest")
                 for r in by_variant[label]]),
            "net_change_at_longest": _mean_std(
                [r.get("linked_net_change_at_longest")
                 for r in by_variant[label]]),
            "composition_effect_at_longest": _mean_std(
                [r.get("linked_composition_effect_at_longest")
                 for r in by_variant[label]]),
            # A p-value is not an effect size: never average it across
            # seeds.  The per-seed values stay in the run records; the
            # summary only counts how many seeds' paired McNemar test
            # rejects at 0.05.
            "mcnemar_significant_seeds_at_longest": sum(
                1 for r in by_variant[label]
                if _finite(r.get("linked_mcnemar_p_at_longest"))
                and float(r["linked_mcnemar_p_at_longest"]) < 0.05),
            "mcnemar_seeds_at_longest": sum(
                1 for r in by_variant[label]
                if _finite(r.get("linked_mcnemar_p_at_longest"))),
            "base_matches_raw": all(
                bool(r.get("linked_base_matches_raw"))
                for r in by_variant[label]),
        }
        for label, *_ in _E2_VARIANTS}
    e2_summary["_counterfactual"] = _counterfactual_companion(
        out_dir, "e2", by_variant, lengths=lengths, reference="none")
    e2_summary["_provenance"] = _stamp(settings)
    _write_json(out_dir / "e2_summary.json", e2_summary)
