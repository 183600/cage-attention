# cage/experiments/_e2.py - E2 helpers: calibration coefficients, per-length contrasts and tuning.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _null_mass_margin(stats):
    """Error-minus-correct null-slot mass, or NaN when it is undefined.

    The run record may be assembled from a restored or hand-edited stats
    payload, so a missing/mismatched/non-finite matrix or malformed
    correctness flags must abstain instead of escaping as an incidental
    ``AttributeError`` / ``IndexError`` from the reporting layer.
    """
    nm = _stats_layer_matrix(stats, "null_mass")
    if nm is None:
        return float("nan")
    corr = _stats_correct_vector(stats, nm.size(0))
    if corr is None or bool(corr.all()) or bool((~corr).all()):
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
    lengths = _positive_lengths(lengths)
    if not lengths:
        payload = {
            "lengths": [],
            "contrasts": {},
            "note": ("empty length sweep: no endpoint or interior length "
                     "to read; companion abstained"),
        }
        _write_json(out_dir / "e2_length_contrasts.json", payload)
        print("[e2] per-length paired contrasts: empty sweep; no endpoint "
              "to read")
        return payload
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
    by_variant = ({label: _mapping_runs(runs)
                   for label, runs in by_variant.items()}
                  if isinstance(by_variant, Mapping) else {})
    for rule, a_label, b_label in (
        ("calib - none", "calib", "none"),
        ("calib - query", "calib", "query"),
        ("calib - calib_raw_ent", "calib", _CALIB_RAW_ENT_CELL),
    ):
        runs_a = _mapping_runs(by_variant.get(a_label, []))
        runs_b = _mapping_runs(by_variant.get(b_label, []))
        rule_rows = []
        for length in lengths:
            def _at_length(r):
                metrics = _length_mapping(
                    r.get("metrics_by_length"), length) or {}
                return dict(
                    r,
                    accuracy=_length_entry(r.get("accuracy_by_length"),
                                           length),
                    correct_bitmask=metrics.get("correct_bitmask"),
                    stream_digest=metrics.get("stream_digest"),
                    n=_length_entry(r.get("eval_n_by_length"), length))
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
            except (TypeError, ValueError, RuntimeError, OverflowError):
                return False
            if math.isnan(xf) or math.isnan(yf):
                return math.isnan(xf) and math.isnan(yf)
            if math.isinf(xf) or math.isinf(yf):
                return xf == yf
            return abs(xf - yf) <= 1e-12

        endpoint_matches = bool(
            (endpoint.get("n") or 0) > 0
            and _same_number(endpoint["mean"], recompute["mean"])
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
    if not isinstance(reference_runs, Mapping):
        reference_runs = {}
    if not settings.per_variant_tuning:
        return {"available": False,
                "reason": "disabled by Settings.per_variant_tuning"}
    lengths = _positive_lengths(lengths)
    if not lengths:
        return {"available": False,
                "reason": ("empty length sweep: no endpoint at which to "
                           "score the tuned recipe")}
    reference_lr, reason = _validated_reference_lr(reference_lr)
    if reason is not None:
        return {"available": False, "reason": reason}
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
            eligible, grid_keys=(key,), full_rows=rows)
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
                    batches=settings.eval_batches, seed=TEST_SEED,
                    train_association_codes=getattr(
                        settings, "_train_association_codes", None),
                    train_association_counts=getattr(
                        settings, "_train_association_counts", None))
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
