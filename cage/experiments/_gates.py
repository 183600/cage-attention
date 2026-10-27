# cage/experiments/_gates.py - Baseline verdict gate, competence checks, efficiency and reproducibility audits.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
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
    # Require the literal JSON boolean True: a hand-edited string such as
    # ``"False"`` is truthy and must not pass a validity gate.
    failed = [key for key in _BLOCKING_VERDICT_GATES
              if verdict.get(key) is not True]
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
    elif hb4_state == "supported" and verdict.get("hb4_supported") is not True:
        failed.append(
            "HB4 (state says supported but the supported boolean is not true)")
    # The state and its two booleans must agree.  A hand-edited verdict
    # could otherwise pair an inconclusive state with ``hb4_violated=True``
    # (or a supported state with a truthy but non-boolean mark) and pass the
    # gate while its own fields say the control failed.
    for key in ("hb4_supported", "hb4_violated"):
        value = verdict.get(key)
        if value is not None and not isinstance(value, bool):
            failed.append(
                f"HB4 ({key} malformed: expected a boolean, got {value!r})")
    if hb4_state is None and verdict.get("hb4_violated") is True:
        failed.append(
            "HB4 (legacy verdict has no state but hb4_violated is true)")
    if hb4_state in ("supported", "inconclusive") and \
            verdict.get("hb4_violated") is True:
        failed.append(
            f"HB4 (state {hb4_state!r} but hb4_violated is true)")
    if hb4_state == "inconclusive" and \
            verdict.get("hb4_supported") is True:
        failed.append("HB4 (state inconclusive but hb4_supported is true)")
    if hb4_state == "violated" and verdict.get("hb4_supported") is True:
        failed.append("HB4 (state violated but hb4_supported is true)")
    # The held-out balance block is nested and its marks gate accuracy
    # claims.  A malformed block must fail closed here rather than crash the
    # runners later (``.get`` on a string/list) or be read as "not audited".
    # Absence of the whole block remains the documented legacy case (None).
    balance = verdict.get("eval_answer_balance")
    balance_all_uniform = None
    balance_streams = {}
    e2_balance = {}
    if balance is not None:
        if not isinstance(balance, Mapping):
            failed.append(
                "eval_answer_balance (malformed: expected an object)")
        else:
            mark = balance.get("all_uniform")
            if mark is not None and not isinstance(mark, bool):
                failed.append(
                    "eval_answer_balance.all_uniform (malformed: expected a "
                    "boolean)")
            else:
                balance_all_uniform = mark
            streams = balance.get("streams")
            if streams is not None and not isinstance(streams, Mapping):
                failed.append(
                    "eval_answer_balance.streams (malformed: expected an "
                    "object)")
            elif isinstance(streams, Mapping):
                balance_streams = streams
                for label, row in streams.items():
                    uniformity = (row.get("uniformity")
                                  if isinstance(row, Mapping) else None)
                    if not isinstance(uniformity, Mapping) or \
                            not isinstance(
                                uniformity.get("uniform_supported"), bool):
                        failed.append(
                            f"eval_answer_balance.streams[{label!r}] "
                            f"(malformed: missing boolean uniformity mark)")
            e2 = balance.get("e2_length_sweep")
            if e2 is not None:
                if not isinstance(e2, Mapping):
                    failed.append(
                        "eval_answer_balance.e2_length_sweep (malformed: "
                        "expected an object)")
                else:
                    e2_balance = e2
                    endpoint_mark = e2.get("endpoint_uniform")
                    if endpoint_mark is not None and not isinstance(
                            endpoint_mark, bool):
                        failed.append(
                            "eval_answer_balance.e2_length_sweep."
                            "endpoint_uniform (malformed: expected a boolean)")
                    e2_streams = e2.get("streams")
                    if e2_streams is not None and not isinstance(
                            e2_streams, Mapping):
                        failed.append(
                            "eval_answer_balance.e2_length_sweep.streams "
                            "(malformed: expected an object)")
                    elif isinstance(e2_streams, Mapping):
                        for label, row in e2_streams.items():
                            uniformity = (
                                row.get("uniformity")
                                if isinstance(row, Mapping) else None)
                            if not isinstance(uniformity, Mapping) or \
                                    not isinstance(
                                        uniformity.get("uniform_supported"),
                                        bool):
                                failed.append(
                                    "eval_answer_balance.e2_length_sweep."
                                    f"streams[{label!r}] (malformed: "
                                    "missing boolean uniformity mark)")
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
        # predates the audit, explicit bool otherwise.  A malformed mark was
        # rejected above, so this is either a real bool or the legacy None.
        "eval_answer_balance_all_uniform": balance_all_uniform,
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
        bad = [label for label, row in balance_streams.items()
               if isinstance(row, Mapping)
               and (row.get("uniformity") or {}).get(
                   "uniform_supported") is False]
        print(f"[{tag}] WARNING: held-out answer balance failed on stream(s) "
              f"{', '.join(bad) or 'unknown'}: accuracy-based readings on "
              f"those streams are class-prior readings, so the F1/PR-AUC "
              f"companions carry the headline and the balance premise is "
              f"stamped as failed in the baseline validity marks")
    if e2_balance.get("endpoint_uniform") is False:
        raw_endpoints = e2_balance.get("endpoint_lengths")
        endpoints = (list(raw_endpoints)
                     if isinstance(raw_endpoints, (list, tuple)) else [])
        e2_streams = e2_balance.get("streams")
        if not isinstance(e2_streams, Mapping):
            e2_streams = {}
        bad_lengths = [
            str(length) for length in endpoints
            if (((e2_streams.get(str(length)) or {})
                 .get("uniformity") or {})
                .get("uniform_supported")) is not True]
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
    for item in _materialize_runs(named_runs):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        name, runs = item
        counts = []
        sizes = []
        for run in _mapping_runs(runs):
            try:
                pair = correct_and_n(run)
                if pair is None:
                    continue
                n_correct, n_total = pair
                n_correct = _difficulty_count(n_correct)
                n_total = _difficulty_count(n_total)
                if (n_correct is None or n_total is None
                        or n_total <= 0 or n_correct > n_total):
                    # Fractional/bool/negative counts are not a measured
                    # binomial arm; truncating them would fabricate a
                    # clustered CI.
                    continue
                counts.append(n_correct)
                sizes.append(n_total)
            except Exception:  # noqa: BLE001 - malformed rows are not evidence
                continue
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
    clean_cells = []
    for item in _materialize_runs(cells):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            clean_cells.append((item[0], _mapping_runs(item[1])))
    cells = clean_cells
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
        raw_params = [_positive_integer(r.get("params"))
                      for r in runs if "params" in r]
        malformed_params = any(value is None for value in raw_params)
        recorded = sorted({value for value in raw_params
                           if value is not None})
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
            "recorded_params": (recorded[0]
                                if len(recorded) == 1
                                and not malformed_params else None),
            "recorded_params_malformed": bool(malformed_params),
            "params_match_run": (not malformed_params
                                 and len(recorded) == 1
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
    if not isinstance(verdict, Mapping):
        verdict = {}
    shallow = verdict.get("shallow_none") or {}
    if not isinstance(shallow, Mapping):
        shallow = {}
    matched = verdict.get("matched_shallow") or {}
    if not isinstance(matched, Mapping):
        matched = {}
    cells = {}
    if not isinstance(by_gate, Mapping):
        by_gate = {}
    deep_runs = _materialize_runs(by_gate.get("none", []))
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
                except (TypeError, ValueError, RuntimeError, OverflowError):
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
            except (TypeError, ValueError, RuntimeError, OverflowError):
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
    lengths = _positive_lengths(lengths)
    if not lengths:
        payload = {
            "experiment": experiment,
            "lengths": [],
            "seq_lengths": [],
            "reference": None if reference is None else str(reference),
            "comparison": "length_efficiency_companion",
            "note": ("empty length sweep: no base/longest cost point to "
                     "measure; companion abstained"),
            "cells": {},
            "_provenance": _stamp(settings),
        }
        _write_json(out_dir / f"{experiment}_efficiency.json", payload)
        print(f"[{experiment}] inference cost scaling: empty sweep; no "
              f"base/longest point to measure")
        return payload
    clean_cells = []
    for item in _materialize_runs(cells):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            clean_cells.append((item[0], _mapping_runs(item[1])))
    cells = clean_cells
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
            acc = _mean_std([
                _length_entry(r.get("accuracy_by_length"), length)
                for r in runs
                if _finite(_length_entry(r.get("accuracy_by_length"),
                                         length))])
            valid_acc = _mean_std([
                _length_entry(r.get("accuracy_by_length"), length)
                for r in valid_runs
                if _finite(_length_entry(r.get("accuracy_by_length"),
                                         length))])
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
                    truth = _length_entry(run.get("accuracy_by_length"),
                                          length)
                    other = _length_entry(refs, length)
                    if _finite(truth) and _finite(other):
                        deltas.append(float(truth) - float(other))
                all_deltas = []
                for run in runs:
                    seed = run.get("seed")
                    refs = all_ref_accuracy.get(seed) or {}
                    truth = _length_entry(run.get("accuracy_by_length"),
                                          length)
                    other = _length_entry(refs, length)
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
    root = pathlib.Path(__file__).resolve().parent.parent.parent
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
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError,
            json.JSONDecodeError) as exc:
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

    def _same_metric_value(va, vb):
        """Structural equality that treats NaN as a stable abstention.

        Metric records carry nested blocks (e.g. the exposure cohorts) as
        well as top-level floats.  A NaN inside such a block is the
        documented "undefined at this sample size" value, so two replay
        runs must compare equal there; plain ``==`` on a dict containing
        NaN is always false and would make a deterministic evaluation look
        irreproducible.
        """
        if isinstance(va, float) and isinstance(vb, float):
            if math.isnan(va) and math.isnan(vb):
                return True
        if isinstance(va, Mapping) and isinstance(vb, Mapping):
            return (va.keys() == vb.keys()
                    and all(_same_metric_value(va[key], vb[key])
                            for key in va))
        if isinstance(va, (list, tuple)) and isinstance(vb, (list, tuple)):
            return (len(va) == len(vb)
                    and all(_same_metric_value(x, y)
                            for x, y in zip(va, vb)))
        return va == vb

    def _same_metrics(a, b):
        if a.keys() != b.keys():
            return False
        return all(_same_metric_value(a[key], b[key]) for key in a)

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
