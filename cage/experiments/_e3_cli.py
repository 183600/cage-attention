# cage/experiments/_e3_cli.py - E3 runner, sanity/all entry points and CLI.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _e2_difficulty_companions(by_variant: dict, lengths: list) -> dict:
    """Difficulty-composition robustness companion for the E2 endpoint.

    The length sweep grows ``n_pairs`` with the sequence length, so the
    *pre-treatment* difficulty mix changes: a longer sequence is more likely
    to contain a repeated answer value or a query token that also appears
    later as a value.  A raw accuracy drop therefore mixes the length effect
    with a harder data mix.  This companion re-weights every arm's
    per-stratum accuracy at the endpoint to the base-length stratum mix
    (:func:`cage.metrics.standardized_proportion`) and re-pairs the same
    seeds.  It is exploratory - the pre-registered H2 decision rows remain
    the raw readouts - but a sign reversal here means the raw endpoint
    should not be attributed to length calibration.
    """
    lengths = _positive_lengths(lengths)
    if not lengths:
        # Reporting an empty sweep has no endpoint to standardize; abstain
        # instead of indexing an empty list (the same rule the per-record
        # stamp follows).
        return {
            "reference_length": None,
            "endpoint_length": None,
            "by_arm": {},
            "contrasts": {},
            "note": ("empty length sweep: no endpoint to standardize; "
                     "companion abstained"),
        }
    source = by_variant if isinstance(by_variant, Mapping) else {}
    by_variant = {label: _mapping_runs(source.get(label, []))
                  for label, *_ in _E2_VARIANTS}
    base_length = int(lengths[0])
    endpoint = int(lengths[-1])

    def tagged(runs):
        rows = []
        for run in runs:
            # ``accuracy_at_longest`` is the endpoint point estimate, but
            # the item-level companions must also read the *endpoint*
            # masks/digest/count.  The top-level fields on a run describe
            # the base length (the training/evaluation length), so leaving
            # them in place paired base-length McNemar/fold/crossed
            # bootstrap results with endpoint accuracy.
            metrics = _length_mapping(
                run.get("metrics_by_length"), endpoint) or {}
            rows.append({
                **run,
                "accuracy": run.get("accuracy_at_longest"),
                "difficulty_standardized_accuracy": run.get(
                    "difficulty_standardized_accuracy_at_longest"),
                "difficulty_standardized_accuracy_drop": run.get(
                    "difficulty_standardized_accuracy_drop"),
                "correct_bitmask": metrics.get("correct_bitmask"),
                "stream_digest": metrics.get("stream_digest"),
                "n": _length_entry(run.get("eval_n_by_length"), endpoint),
            })
        return rows

    def _sign(ci):
        mean = ci.get("mean")
        if not _finite(mean) or float(mean) == 0.0:
            return 0
        return 1 if float(mean) > 0.0 else -1

    def _sign_label(raw_ci, std_ci):
        raw, std = _sign(raw_ci), _sign(std_ci)
        if raw == 0 or std == 0:
            return "undetermined"
        return "consistent" if raw == std else "REVERSED"

    print(f"\n[e2] difficulty-composition robustness companion (exploratory, "
          f"outside the FDR family): reweighting each length's per-stratum "
          f"accuracy to the L={base_length} retrieval-position x ambiguity "
          f"mix.  More pairs at a longer length means more repeated values "
          f"and naive-confusable queries, so raw accuracy at L={endpoint} "
          f"mixes length with a harder data mix; 'std' removes that mix "
          f"difference.  The pre-registered H2 rows stay the raw readouts.")
    print(f"{'arm':>16} {'raw@long':>18} {'std@long':>18} "
          f"{'raw_drop':>18} {'std_drop':>18} {'cov@long':>9} "
          f"{'effN@long':>10} {'mixTV@long':>11}")
    by_arm = {}
    for label, *_ in _E2_VARIANTS:
        runs = by_variant[label]
        raw_acc = _mean_std([r.get("accuracy_at_longest") for r in runs])
        std_acc = _mean_std([
            r.get("difficulty_standardized_accuracy_at_longest")
            for r in runs])
        raw_drop = _mean_std([r.get("accuracy_drop") for r in runs])
        std_drop = _mean_std([
            r.get("difficulty_standardized_accuracy_drop") for r in runs])
        coverage = _mean_std([
            r.get("difficulty_standardized_coverage_at_longest")
            for r in runs])
        effective_n = _mean_std([
            r.get("difficulty_standardized_effective_n_at_longest")
            for r in runs])
        mix_tv = _mean_std([
            r.get("difficulty_mix_tv_at_longest") for r in runs])
        by_arm[label] = {
            "raw_accuracy_at_longest": raw_acc,
            "difficulty_standardized_accuracy_at_longest": std_acc,
            "raw_accuracy_drop": raw_drop,
            "difficulty_standardized_accuracy_drop": std_drop,
            "difficulty_standardized_coverage_at_longest": coverage,
            "difficulty_standardized_effective_n_at_longest": effective_n,
            "difficulty_mix_tv_at_longest": mix_tv,
        }

        def _cell(agg):
            if not _finite(agg.get("mean")):
                return "n/a"
            return f"{agg['mean']:+.4f}±{agg['std']:.4f}"

        print(f"{label:>16} {_cell(raw_acc):>18} {_cell(std_acc):>18} "
              f"{_cell(raw_drop):>18} {_cell(std_drop):>18} "
              f"{_cell(coverage):>9} {_cell(effective_n):>10} "
              f"{_cell(mix_tv):>11}")
    print(f"\n[e2] difficulty-standardized endpoint contrasts (paired over "
          f"training seeds; same protocol as the accuracy companions):")
    print(f"{'contrast':>18} {'raw acc':>28} {'std acc':>28} "
          f"{'raw drop':>28} {'std drop':>28} {'signs':>15}")
    contrasts = {}
    for rule, a_label, b_label in (
            ("calib - none", "calib", "none"),
            ("calib - query", "calib", "query")):
        tagged_a = tagged(by_variant[a_label])
        tagged_b = tagged(by_variant[b_label])
        raw_acc = _paired_diff(tagged_a, tagged_b, "accuracy")
        std_acc = _paired_diff(tagged_a, tagged_b,
                               "difficulty_standardized_accuracy")
        raw_drop = _paired_diff(tagged_a, tagged_b, "accuracy_drop")
        std_drop = _paired_diff(tagged_a, tagged_b,
                                "difficulty_standardized_accuracy_drop")
        acc_signs = _sign_label(raw_acc, std_acc)
        drop_signs = _sign_label(raw_drop, std_drop)
        contrasts[rule] = {
            "raw_accuracy_at_longest": raw_acc,
            "difficulty_standardized_accuracy_at_longest": std_acc,
            "raw_accuracy_drop": raw_drop,
            "difficulty_standardized_accuracy_drop": std_drop,
            "accuracy_sign_agreement": acc_signs,
            "drop_sign_agreement": drop_signs,
            # An undetermined endpoint (no finite paired effect) is not
            # agreement: calling it "consistent" would turn an unmeasured
            # contrast into a robustness pass.
            "raw_and_standardized_agree": bool(
                acc_signs == "consistent" and drop_signs == "consistent"),
        }
        print(f"{rule:>18} {_fmt_ci(raw_acc):>28} {_fmt_ci(std_acc):>28} "
              f"{_fmt_ci(raw_drop):>28} {_fmt_ci(std_drop):>28} "
              f"{(acc_signs + '/' + drop_signs):>15}")
        if acc_signs == "REVERSED" or drop_signs == "REVERSED":
            print(f"      -> {rule}: WARNING the raw and "
                  f"difficulty-standardized endpoint effects do not agree "
                  f"in sign (accuracy: {acc_signs}, drop: {drop_signs}); "
                  f"the raw H2 reading is confounded by the changing "
                  f"difficulty mix and must not be attributed to length "
                  f"calibration alone")
    return {
        "reference_length": base_length,
        "endpoint_length": endpoint,
        "by_arm": by_arm,
        "contrasts": contrasts,
        "note": ("exploratory post-stratification companion: per-stratum "
                 "accuracy at the endpoint reweighted to the base-length "
                 "retrieval-position x ambiguity mix; the pre-registered "
                 "H2 rows and FDR family are unchanged"),
    }


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
    lengths = _positive_lengths(lengths)
    if not lengths:
        return _class_frequency_companions(
            [], "e2 (empty length sweep; no endpoint)")
    source = by_variant if isinstance(by_variant, Mapping) else {}
    by_variant = {label: _mapping_runs(source.get(label, []))
                  for label, *_ in _E2_VARIANTS}
    endpoint = int(lengths[-1])

    def tagged(runs):
        out = []
        for run in runs:
            metrics = _length_mapping(
                run.get("metrics_by_length"), endpoint) or {}
            out.append({
                **run,
                "accuracy": metrics.get(
                    "accuracy", _length_entry(
                        run.get("accuracy_by_length"), endpoint)),
                "recall_macro": metrics.get("recall_macro"),
                "f1_macro": metrics.get("f1_macro"),
                "f1_weighted": metrics.get("f1_weighted"),
                "correct_bitmask": metrics.get("correct_bitmask"),
                "stream_digest": metrics.get("stream_digest"),
                "n": _length_entry(run.get("eval_n_by_length"), endpoint),
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
        # Cross-scheme write companions: resid_norm is read after the
        # block's final LayerNorm in post-norm blocks but on the raw
        # residual stream in pre/seednorm blocks, so its level is not
        # comparable across normalisation schemes.  The relative branch
        # writes are (see Block._stamp_write_ratio).
        awr = stats.get("attn_write_ratio")
        mwr = stats.get("mlp_write_ratio")
        awr_mean = None if awr is None else awr.mean(dim=0)
        mwr_mean = None if mwr is None else mwr.mean(dim=0)
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
            "attn_write_ratio_by_layer": (None if awr_mean is None
                                          else awr_mean.tolist()),
            "mlp_write_ratio_by_layer": (None if mwr_mean is None
                                         else mwr_mean.tolist()),
            "mlp_write_ratio_last_over_first": (
                float("nan") if mwr_mean is None else float(
                    mwr_mean[-1] / mwr_mean[0].clamp_min(1e-12))),
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
            recorded = [_positive_integer(r.get("params"))
                        for r in target_runs if isinstance(r, Mapping)
                        and "params" in r]
            target_params = (recorded[0] if recorded
                             and all(value is not None for value in recorded)
                             and len(set(recorded)) == 1 else None)
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
    _design_sensitivity_plan(settings, verdict, out_dir, "e3")
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
            unmeasured = [
                d for d, row in ((verdict.get("depth_ladder") or {})
                                 .get("by_depth") or {}).items()
                if row.get("unmeasured")]
            if unmeasured:
                print("[e3] HB7 NOT DETERMINABLE: rung(s) "
                      f"{', '.join(str(d) for d in unmeasured)} were not "
                      "measured, so the ladder cannot establish whether "
                      "depth is required; the depth-based framing of "
                      "H3a/H3b is unverified")
            else:
                print("[e3] HB7 depth context unavailable (rule ceiling "
                      "unknown): the depth-based framing of H3a/H3b is "
                      "unverified")
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
    cells = tuple((gate, norm, layer_scale)
                  for gate in ("query", "calib")
                  for norm in NORMS
                  for layer_scale in (None, 1e-2))
    order = _blocked_run_order(
        cells, settings.seeds, settings.run_order_seed,
        shuffle=settings.randomize_run_order)
    _write_run_order(settings, out_dir, "e3", order, cells,
                     settings.seeds)
    for (gate, norm, layer_scale), seed, block, position in order:
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
        # Cross-scheme write companions (see the capacity-control site for
        # the rationale): the relative branch writes are comparable between
        # pre/post/mix/seednorm blocks, unlike the raw resid_norm levels.
        awr = stats.get("attn_write_ratio")
        mwr = stats.get("mlp_write_ratio")
        awr_mean = None if awr is None else awr.mean(dim=0)
        mwr_mean = None if mwr is None else mwr.mean(dim=0)
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
            "run_block": block, "run_position": position,
            "eval_seed": eval_seed, "stats_seed": stats_seed,
            "params": params, "model_cfg": model_cfg,
            "eval_n": int(metrics["n"]),
            **{k: v for k, v in metrics.items() if k != "n"},
            "gate_depth_ratio": gate_depth_ratio(stats),
            "resid_norm_by_layer": resid.tolist(),
            "resid_norm_last_over_first": float(
                resid[-1] / resid[0].clamp_min(1e-12)),
            "attn_write_ratio_by_layer": (None if awr_mean is None
                                          else awr_mean.tolist()),
            "mlp_write_ratio_by_layer": (None if mwr_mean is None
                                         else mwr_mean.tolist()),
            "mlp_write_ratio_last_over_first": (
                float("nan") if mwr_mean is None else float(
                    mwr_mean[-1] / mwr_mean[0].clamp_min(1e-12))),
            **{f"err_{k}": v
               for k, v in errors["summary"].items()},
            **_training_record(settings, lr),
            **_run_context(settings),
            **curve,
        }
        _write_result(out_dir / "e3_runs.jsonl", rec)
        runs.append(rec)
    _report_realized_run_order(
        out_dir, "e3", runs, order, cells, settings.seeds,
        lambda run: (run.get("gate"), run.get("norm"),
                     run.get("layer_scale")))

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
    e3_summary = {
        f"{g}|{n}|{'off' if ls is None else f'{ls:g}'}":
            _aggregate_runs(rs)
        for (g, n, ls), rs in index.items()}
    e3_summary["_capacity_matched_pre"] = capacity_benchmark
    e3_summary["_efficiency"] = efficiency
    e3_summary["_class_frequency_companions"] = metric_robustness
    e3_summary["_counterfactual"] = _counterfactual_companion(
        out_dir, "e3", index,
        reference=lambda cell: ("query", cell[1], cell[2]))
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
                    "(see cage/experiments/ for the registry).")
    parser.add_argument("--only", default=None,
                        choices=["baselines", "sanity", "e1", "e2", "e3", "all"],
                        help="experiment to run (default: baselines)")
    parser.add_argument("--quick", action="store_true",
                        help="tiny pipeline-check configuration")
    parser.add_argument("--device", default=None,
                        help="torch device (default: cpu)")
    parser.add_argument("--seeds", type=int, default=None,
                        help="number of training seeds (0..n-1)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None,
                        help="skip recipe tuning and use this lr directly")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--fdr-q", type=float, default=None,
                        help="Benjamini-Hochberg FDR level for the "
                             "pre-registered contrast families "
                             "(default: 0.05)")
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
    parser.add_argument("--out", default=None,
                        help="results directory (default: results)")
    parser.add_argument("--replay-manifest", default=None, metavar="PATH",
                        help="re-run the frozen recipe recorded in a run "
                             "manifest; the source tree, hypothesis registry "
                             "and data-stream fingerprints must match, and "
                             "--out must be a new directory")
    parser.add_argument("--allow-environment-drift", action="store_true",
                        help="with --replay-manifest: permit a replay under "
                             "a different critical software/numerical "
                             "environment and record the drift instead of "
                             "refusing")
    parser.add_argument("--verify-artifacts", action="store_true",
                        help="check files in --out against artifact_index.json "
                             "and exit")
    parser.add_argument("--check-environment", action="store_true",
                        help="compare --out's recorded environment with this "
                             "process and exit (0-compatible, 1-drift, "
                             "2-no usable record); writes nothing")
    parser.add_argument("--check-source", action="store_true",
                        help="compare every archived code hash in --out with "
                             "this checkout and exit (0-match, 1-mismatch or "
                             "malformed, 2-no code-provenance records); "
                             "writes nothing")
    parser.add_argument("--verify-provenance", action="store_true",
                        help="recompute every result row's setup/environment "
                             "hashes and check that one run_id never names "
                             "two protocols; exit 0-consistent, 1-defect, "
                             "2-no result records; writes nothing")
    parser.add_argument("--require-homogeneous", action="store_true",
                        help="with --verify-provenance: treat a result file "
                             "that mixes setups/environments as an error "
                             "instead of a warning")
    parser.add_argument("--strict-environment", action="store_true",
                        help="abort a run when the recorded environment in "
                             "--out differs from the current one in a "
                             "critical (software/numerical) field; the "
                             "default is to record and warn")
    args = parser.parse_args(argv)
    verify_flags = (args.verify_artifacts or args.check_environment
                    or args.check_source or args.verify_provenance)
    if args.replay_manifest is not None and verify_flags:
        print("[runner] --replay-manifest cannot be combined with a "
              "read-only --verify/--check action; aborting")
        return 2
    if args.allow_environment_drift and args.replay_manifest is None:
        print("[runner] --allow-environment-drift only applies to "
              "--replay-manifest; aborting")
        return 2
    verify_out = args.out or "results"
    if args.verify_artifacts:
        return _verify_artifacts(verify_out)
    if args.check_environment:
        return _check_environment(verify_out)
    if args.check_source:
        return _check_source(verify_out)
    if args.verify_provenance:
        return _audit_provenance(
            verify_out, require_homogeneous=bool(args.require_homogeneous))

    replay = None
    if args.replay_manifest is not None:
        conflicts = []
        if args.quick:
            conflicts.append("--quick")
        for name, flag in (
                ("only", "--only"), ("device", "--device"),
                ("seeds", "--seeds"), ("steps", "--steps"), ("lr", "--lr"),
                ("d_model", "--d-model"), ("n_layers", "--n-layers"),
                ("fdr_q", "--fdr-q"),
                ("deterministic", "--deterministic/--no-deterministic"),
                ("per_variant_tuning",
                 "--per-variant-tuning/--no-per-variant-tuning")):
            if getattr(args, name) is not None:
                conflicts.append(flag)
        if conflicts:
            print(f"[runner] REPLAY REJECTED: the manifest freezes the "
                  f"recipe, so {', '.join(conflicts)} cannot be overridden; "
                  f"only --out and --allow-environment-drift are accepted")
            return 2
        try:
            settings, manifest, manifest_drift, manifest_sha = (
                _load_replay_manifest(
                    args.replay_manifest,
                    allow_environment_drift=bool(
                        args.allow_environment_drift)))
        except ReplayManifestError as exc:
            print(f"[runner] REPLAY REJECTED: {exc}")
            return 2
        except Exception as exc:  # noqa: BLE001 - a failed probe is a refusal
            print(f"[runner] REPLAY REJECTED: cannot verify manifest "
                  f"({type(exc).__name__}: {exc})")
            return 2
        experiment = manifest["experiment"]
        source_run = manifest.get("run_id")
        safe_run = (source_run
                    if isinstance(source_run, str) and source_run
                    and all(ch.isalnum() or ch in "._-"
                            for ch in source_run)
                    else "unknown")
        out_dir = pathlib.Path(
            args.out if args.out is not None
            else f"results_replay_{safe_run}")
        manifest_dir = pathlib.Path(args.replay_manifest).resolve().parent
        if out_dir.resolve() == manifest_dir:
            print("[runner] REPLAY REJECTED: --out must differ from the "
                  "source results directory; a replay is new evidence, not "
                  "an append to the run it is reproducing")
            return 2
        replay = {
            "path": str(pathlib.Path(args.replay_manifest).resolve()),
            "manifest_sha256": manifest_sha,
            "source_run_id": source_run,
            "source_setup_sha256": manifest.get("setup_sha256"),
            "source_tree_sha256": manifest.get("source_tree_sha256"),
            "environment_compatible": manifest_drift.get("compatible"),
            "environment_identical": manifest_drift.get("identical"),
            "environment_override": bool(args.allow_environment_drift),
        }
    else:
        settings = Settings.quick(args.device or "cpu") if args.quick else (
            Settings(device=args.device or "cpu"))
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
                print("[runner] no training seed left after excluding "
                      "reserved evaluation streams; aborting")
                return 2
        if args.steps is not None:
            settings.steps = args.steps
        if args.lr is not None:
            settings.lr = args.lr
        if args.d_model is not None:
            settings.d_model = args.d_model
        if args.n_layers is not None:
            settings.n_layers = args.n_layers
        if args.fdr_q is not None:
            settings.fdr_q = args.fdr_q
        if settings.steps < 1:
            print("[runner] --steps must be a positive integer; aborting")
            return 2
        if settings.lr is not None and (
                not math.isfinite(settings.lr) or settings.lr <= 0.0):
            print("[runner] --lr must be a finite positive number; aborting")
            return 2
        if not 0.0 < settings.fdr_q < 1.0:
            print("[runner] --fdr-q must lie strictly between 0 and 1; "
                  "aborting")
            return 2
        if settings.d_model < 1 or settings.d_model % settings.n_heads:
            print(f"[runner] --d-model must be a positive multiple of n_heads="
                  f"{settings.n_heads}; aborting")
            return 2
        if settings.n_layers < 1:
            print("[runner] --n-layers must be a positive integer; aborting")
            return 2
        experiment = args.only or "baselines"
        out_dir = pathlib.Path(args.out or "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    settings.out_dir = out_dir
    # One writer per results directory: the append-only JSONL files, the
    # environment chain and the cumulative artifact index are all
    # read-modify-write state.  A second concurrent invocation would
    # interleave rows and could drop the other's index entries, so fail fast
    # with a dedicated exit code instead of corrupting the evidence store.
    lock = ResultsLock(out_dir)
    try:
        lock.acquire()
    except ResultsLockError as exc:
        print(f"[runner] ABORTED: {exc}")
        return LOCK_EXIT_CODE
    except OSError as exc:
        lock.release()
        print(f"[runner] ABORTED: cannot lock {out_dir} "
              f"({type(exc).__name__}: {exc})")
        return LOCK_EXIT_CODE
    if lock.mode == "unavailable":
        print("[runner] WARNING: advisory file locking is unavailable on "
              "this platform; concurrent writers are not prevented")
    _RUN_ARTIFACTS.clear()
    started = time.monotonic()
    # Fail closed before writing new evidence: appending fresh rows to a
    # directory whose earlier rows no longer match the cumulative
    # certificate would mix trustworthy and untrustworthy data under one
    # index.  The end-of-run index writer must never be the first place a
    # tampered history is noticed.
    index_path = out_dir / "artifact_index.json"
    if index_path.exists():
        verified = _verify_artifacts(out_dir)
        if verified != 0:
            print(f"[runner] ABORTED: existing results in {out_dir} do not "
                  f"match their artifact_index.json; refusing to append new "
                  f"evidence to a broken chain of custody. Re-run "
                  f"--verify-artifacts, restore the certified files, or "
                  f"choose a fresh --out directory")
            lock.release()
            return ARTIFACT_INTEGRITY_EXIT_CODE
    status = "failed"
    try:
        # Read the previous environment before writing the new one, then
        # record the compatibility verdict in the cumulative history.  The
        # environment probe is also the one used for the manifest and every
        # record, so all copies agree.
        context = _run_context(settings)
        drift = _record_environment_history(out_dir, context)
        critical_fields = [d.get("field") for d in
                           drift.get("critical_differences", [])]
        advisory_fields = [d.get("field") for d in
                           drift.get("advisory_differences", [])]
        if drift.get("compatible") is False:
            print(f"[runner] WARNING: environment drift vs {out_dir}: "
                  f"{drift.get('reason')}")
            if critical_fields:
                print(f"[runner]   critical fields: "
                      f"{', '.join(str(f) for f in critical_fields)}")
            if args.strict_environment:
                status = "aborted"
                print("[runner] ABORTED: --strict-environment requested and "
                      "the recorded environment is not compatible")
                _run_event(settings, experiment, "aborted",
                           error="EnvironmentMismatch",
                           reason=drift.get("reason"),
                           critical_fields=critical_fields,
                           elapsed_seconds=round(
                               time.monotonic() - started, 3))
                return 3
        elif drift.get("compatible") is True and advisory_fields:
            print(f"[runner] note: advisory hardware drift vs {out_dir} "
                  f"({', '.join(str(f) for f in advisory_fields)}); "
                  f"critical numerical fields match")
        elif drift.get("compatible") is None:
            print(f"[runner] environment history: {drift.get('reason')}")
        _write_json(out_dir / "hypotheses.json", HYPOTHESES)
        replay_event = {}
        if replay is not None:
            replay_event = {
                "replay_of": replay["source_run_id"],
                "replay_manifest": replay["path"],
                "replay_manifest_sha256": replay["manifest_sha256"],
                "replay_environment_override": replay[
                    "environment_override"],
            }
        _run_event(settings, experiment, "started", quick=bool(args.quick),
                   seeds=list(settings.seeds), steps=settings.steps,
                   lr=settings.lr, device=settings.device,
                   fdr_q=settings.fdr_q,
                   lock_mode=lock.mode,
                   environment_compatible=drift.get("compatible"),
                   environment_identical=drift.get("identical"),
                   environment_critical_fields=critical_fields,
                   environment_advisory_fields=advisory_fields,
                   **replay_event)
        _write_json(out_dir / "environment.json", context)
        if replay is not None:
            _write_json(out_dir / "replay.json", {
                **replay,
                "schema_version": _RECORD_SCHEMA,
                "replay_run_id": context["run_id"],
                "created_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        replay_note = (f" replay_of={replay['source_run_id']}"
                       if replay is not None else "")
        print(f"[runner] only={experiment} quick={bool(args.quick)} "
              f"seeds={settings.seeds} steps={settings.steps} "
              f"lr={settings.lr if settings.lr is not None else 'tuned'} "
              f"fdr_q={settings.fdr_q:g} "
              f"device={settings.device} out={out_dir}{replay_note}")
        if experiment == "baselines":
            run_baselines(settings, out_dir)
        elif experiment == "sanity":
            run_sanity(settings, out_dir)
        elif experiment == "e1":
            run_e1(settings, out_dir)
        elif experiment == "e2":
            run_e2(settings, out_dir)
        elif experiment == "e3":
            run_e3(settings, out_dir)
        else:
            run_all(settings, out_dir)
    except (DesignPlanError, OverfitSanityError, ReproducibilityError,
            SplitAuditError, BaselineVerdictError) as e:
        status = "aborted"
        print(f"[runner] ABORTED before grid training: {e}")
        _run_event(settings, experiment, "aborted",
                   error=type(e).__name__, reason=str(e),
                   elapsed_seconds=round(time.monotonic() - started, 3))
        return 3
    except Exception as e:
        _run_event(settings, experiment, "failed",
                   error=type(e).__name__, reason=str(e),
                   elapsed_seconds=round(time.monotonic() - started, 3))
        raise
    else:
        status = "completed"
        _run_event(settings, experiment, "completed",
                   elapsed_seconds=round(time.monotonic() - started, 3))
        print(f"\n[runner] done; records in {out_dir}/")
        return 0
    finally:
        try:
            _write_artifact_index(out_dir, settings, experiment,
                                  completed=(status == "completed"),
                                  elapsed_seconds=time.monotonic() - started)
        except Exception as e:  # noqa: BLE001 - index failure is a warning
            print(f"[runner] WARNING: could not write artifact index: {e}")
        finally:
            lock.release()
