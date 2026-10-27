# cage/experiments/_audits.py - Data split audit, depth ladder, shallow control and overfit gates.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _run_depth_ladder(settings: Settings, out_dir: pathlib.Path,
                      ceiling: dict | None) -> dict:
    """HB6/HB7 complexity ladder: the tuned standard transformer per depth.

    The simplest *learned* model on the baseline ladder is the 1-layer
    transformer (HB6), but a single shallow point cannot say where depth
    stops mattering.  This runner tunes the learning rate *and step
    budget* of the standard ``gate="none"`` transformer separately at every
    ladder depth (default 1/2/4 layers) through the same entry point as the
    deep grids - same optimizer factory, cosine schedule, batch geometry
    and evaluation streams - on the validation seed (over the
    pre-registered ``min_seeds_for_claim`` floor's worth of training seeds,
    so no rung's recipe is frozen on a two-seed argmax), and reports the
    chosen recipe on the disjoint test seed.  The step grid is expressed as
    multipliers of ``settings.steps``; a rung whose optimum sits at either
    axis boundary is flagged by the selection audit as unresolved.  Depth is the only architectural difference from
    E1's ``none`` arm, so:

    * each rung's gap to the parameter-free rule ceiling measures how much
      depth the recall readout actually requires - but only rungs whose
      seed-clustered 95% *lower* bound clears chance count as evidence:
      the HB0 competence premise separates "needs more depth" from "this
      step count did not learn the task", and with no competent rung the depth
      demand is reported as *not determinable* instead of "required";
    * the saturation depth ``d*`` (smallest competent rung whose
      seed-clustered 95% *lower* bound lies within the pre-registered SESOI
      of the ceiling estimate, i.e. a one-sided equivalence statement) is the
      reference the deep grids must justify themselves against (HB7): a
      gain a 2-layer standard transformer already matches is not evidence
      about depth.  The bare "the CI touches the ceiling" criterion is an
      *absence-of-evidence* reading and is kept only as the companion
      ``reaches_ceiling_ci``; the same SESOI is the minimum plausible gap
      for "depth measurably required" (``ceiling - clustered_hi >= SESOI``),
      so a statistically detectable but practically negligible gap cannot
      earn the complexity;
    * a complex model is only credited for what it does beyond the tuned
      shallow baseline.

    Each depth's argmax is audited by
    :func:`_selection_reliability` (margin vs runner-up in seed units,
    winning curve verdicts, grid-boundary position) and stored in the
    payload's ``selection`` block: a rung whose tuned rate is not
    resolved is a rung whose accuracy may understate the depth being
    credited, and the warning says so at selection time.

    Returns ``{"depths": {depth: payload}, "ladder": [...],
    "saturation_depth": int|None, "hb7_depth_required": bool|None,
    "shallowest": depth}``; the shallow rung reuses the legacy names
    (``shallow_none_baseline`` records, ``baselines_shallow_none.json``)
    that E1's complexity readout reads from the verdict.
    """
    raw_ladder = tuple(settings.depth_ladder)
    depths = tuple(_positive_integer(d) for d in raw_ladder)
    if not depths or any(d is None for d in depths):
        raise ValueError(
            f"depth_ladder must be non-empty positive ints, got "
            f"{raw_ladder!r}")
    ladder = tuple(sorted(set(depths)))
    tune_seeds = settings.seeds[:max(1, min(settings.min_seeds_for_claim,
                                             len(settings.seeds)))]
    ceiling_acc = float(ceiling["accuracy"]) if ceiling else float("nan")
    ceiling_known = math.isfinite(ceiling_acc)
    chance = 1.0 / (settings.vocab - 2)
    # HB0 is a union claim over the ladder rungs ("at least one rung is
    # competent"), so the per-rung lower quantile is Bonferroni-adjusted by
    # the number of rungs exactly as ``_competence_check`` does for the
    # E1/E2/E3 arm families.  Without the adjustment the ladder could call
    # a rung competent on an uncorrected one-sided bound while the
    # registered family-wise criterion would not.
    competence_alpha = 0.05 / max(len(ladder), 1)
    # Pre-registered practical-significance bound for the depth readouts:
    # HB6/HB7 both register ``sesoi: {accuracy: 0.01}``, so the same bound
    # decides "saturated" (one-sided equivalence within the band) and
    # "depth measurably required" (plausible gap at least the band).
    sesoi = float((HYPOTHESES["HB6"].get("sesoi") or {}).get("accuracy", 0.01))
    if float((HYPOTHESES["HB7"].get("sesoi") or {}).get("accuracy", sesoi)) \
            != sesoi:
        raise ValueError(
            "HB6/HB7 must share one registered accuracy SESOI for the "
            "depth ladder's saturation rule")
    depths: dict[int, dict] = {}
    tuning_tables: dict[int, list] = {}
    step_candidates = _depth_step_candidates(settings)
    for depth in ladder:
        table = []
        for train_steps in step_candidates:
            for lr in settings.lr_grid:
                accs, verdicts, tuning_runs = [], [], []
                for seed in tune_seeds:
                    _, params, metrics, curve, _ = _train_and_eval(
                        settings, seed, lr, eval_seed=VAL_SEED,
                        selection=True, steps=train_steps, gate="none",
                        n_layers=depth)
                    accs.append(metrics["accuracy"])
                    verdicts.append(curve["curve_verdict"])
                    tuning_runs.append({
                        "seed": int(seed), "params": int(params),
                        "val_accuracy": metrics["accuracy"],
                        "eval_seed": int(VAL_SEED), "curve": curve})
                mean = sum(accs) / len(accs)
                std = ((sum((a - mean) ** 2 for a in accs)
                        / (len(accs) - 1)) ** 0.5 if len(accs) > 1 else 0.0)
                table.append({"lr": lr, "steps": int(train_steps),
                              "val_accuracy_mean": mean,
                              "val_accuracy_std": std,
                              "n_seeds": len(accs),
                              "curve_verdicts": verdicts,
                              "curve_invalid_runs": sum(
                                  v in CURVE_INVALID_VERDICTS
                                  for v in verdicts),
                              "runs": tuning_runs})
        # An all-non-finite tuning table (every run diverged / produced no
        # finite validation accuracy) must not be silently read as an argmax:
        # NaN comparisons make ``max`` return an arbitrary row, which would
        # freeze an unmeasured learning rate and manufacture a rung.
        finite_rows = [row for row in table
                       if _finite(row["val_accuracy_mean"])]
        # Audit the full table for the unmeasured branch; the selected
        # (curve-valid) pool is audited again below.  Both axes matter: a
        # winner at the 2x step edge is as unresolved as one at the lr edge.
        selection = _selection_reliability(
            table, grid_keys=("lr", "steps"), full_rows=table)
        if not finite_rows:
            print(f"[baselines] WARNING: depth={depth} tuning table has no "
                  f"finite validation accuracy (all {len(table)} "
                  f"lr x steps candidates non-finite); the rung is recorded "
                  f"as unmeasured and cannot support HB6/HB7 or the "
                  f"saturation depth")
            depths[depth] = {
                "name": f"shallow_none_d{depth}", "n_layers": depth,
                "val_seed": VAL_SEED, "test_seed": TEST_SEED,
                "chosen_lr": None, "chosen_steps": None,
                "tuning_table": table,
                "selection": selection,
                "accuracy_mean": float("nan"),
                "accuracy_std": float("nan"), "accuracy_by_seed": {},
                "pooled_wilson": {"p": float("nan"), "lo": float("nan"),
                                  "hi": float("nan"), "n": 0},
                "cluster_ci": clustered_proportion_ci([], []),
                "ceiling": ceiling_acc, "chance": chance,
                "sesoi_accuracy": sesoi,
                "above_chance": False, "ceiling_gap": float("nan"),
                "ceiling_gap_lower": float("nan"),
                "reaches_ceiling": False, "reaches_ceiling_ci": False,
                "hb6_supported": False, "hb6_supported_ci": False,
                "unmeasured": True, "runs": [],
            }
            tuning_tables[depth] = table
            _write_json(out_dir / f"baselines_depth{depth}.json",
                        depths[depth])
            continue
        pick = _curve_valid_selection_rows(table)
        if pick["n_excluded"]:
            print(f"[baselines] depth={depth} selection excluded "
                  f"{pick['n_excluded']} of {pick['n_finite']} finite "
                  f"lr x steps row(s) with curve-invalid runs; selecting "
                  f"among curve-valid rows only")
        elif pick["n_invalid"]:
            print(f"[baselines] WARNING: depth={depth} has no curve-valid "
                  f"finite lr x steps row; falling back to all "
                  f"{pick['n_finite']} finite row(s) - the rung's recipe is "
                  f"not validated by a healthy curve")
        eligible = pick["rows"]
        selection = _selection_reliability(
            eligible, grid_keys=("lr", "steps"), full_rows=table)
        selection["pool"] = pick["pool"]
        selection["excluded_curve_invalid_rows"] = pick["n_excluded"]
        selection["curve_invalid_rows"] = pick["n_invalid"]
        best = max(eligible, key=lambda row: row["val_accuracy_mean"])
        warn = _selection_warn(
            f"depth={depth} lr x steps selection", selection)
        if warn:
            print(f"[baselines] {warn}: this rung's tuned accuracy may "
                  f"understate the depth it is credited with")
        runs = []
        for seed in settings.seeds:
            model, params, metrics, curve, _ = _train_and_eval(
                settings, seed, best["lr"], eval_seed=TEST_SEED, gate="none",
                n_layers=depth, steps=best["steps"])
            n = int(metrics["n"])
            runs.append({
                "seed": seed, "accuracy": metrics["accuracy"],
                "accuracy_lo": metrics["accuracy_lo"],
                "accuracy_hi": metrics["accuracy_hi"], "n": n,
                "correct": int(round(metrics["accuracy"] * n)),
                "params": params, "curve_verdict": curve["curve_verdict"],
            })
            name = ("shallow_none_baseline" if depth == ladder[0]
                    else f"shallow_none_baseline_d{depth}")
            _write_result(out_dir / "baselines_runs.jsonl",
                          {"experiment": "baselines",
                           "hypotheses": ["HB6", "HB7"], "name": name,
                           "n_layers": depth, "lr": best["lr"],
                           "steps": int(best["steps"]),
                           "training_seed": seed, "eval_seed": TEST_SEED,
                           "accuracy": metrics["accuracy"],
                           "accuracy_lo": metrics["accuracy_lo"],
                           "accuracy_hi": metrics["accuracy_hi"], "n": n,
                           "params": params,
                           **_training_record(settings, best["lr"],
                                              best["steps"]),
                           **_run_context(settings), **curve})
        # Curve-invalid final runs (plateau/diverging/instability) cannot
        # support a depth claim: their accuracies measure an optimisation
        # failure, not the depth's capacity.  The rung's decision statistics
        # therefore read only the curve-valid seeds; every run stays in the
        # record so the exclusion is visible.
        curve_invalid_seeds = sorted(
            int(r["seed"]) for r in runs
            if r.get("curve_verdict") in CURVE_INVALID_VERDICTS)
        valid_runs = [r for r in runs if r["seed"] not in curve_invalid_seeds]
        if curve_invalid_seeds:
            print(f"[baselines] WARNING: depth={depth} final runs "
                  f"{curve_invalid_seeds} are curve-invalid and are excluded "
                  f"from the rung's competence/saturation statistics")
        if valid_runs:
            pooled = wilson_ci(sum(r["correct"] for r in valid_runs),
                               sum(r["n"] for r in valid_runs))
            # Seeds share one sealed evaluation stream, so the decision
            # interval must resample seeds and keep a shared finite-stream
            # error; pooling the counts as independent examples would
            # pseudo-replicate.
            cluster = clustered_proportion_ci(
                [r["correct"] for r in valid_runs],
                [r["n"] for r in valid_runs],
                lower_alpha=competence_alpha)
            summary = _mean_std([r["accuracy"] for r in valid_runs])
        else:
            pooled = {"p": float("nan"), "lo": float("nan"),
                      "hi": float("nan"), "n": 0}
            cluster = clustered_proportion_ci([], [])
            summary = _mean_std([])
        # A rung whose final runs are all curve-invalid has no accuracy
        # estimate at all.  It is unmeasured, not "at chance": treating it
        # as measured would let HB7 claim depth is required while a rung
        # that could still saturate was silently discarded.
        unmeasured = not bool(valid_runs)
        # HB0 competence premise for HB6/HB7: "below the ceiling" is only
        # evidence that depth is required when the rung demonstrably
        # learned the task.  A rung at chance is an underfit/step-count
        # statement, not a depth statement, so it can neither support HB6
        # ("depth measurably required") nor feed the saturation depth.
        above_chance = bool(cluster["valid"] and math.isfinite(cluster["lo"])
                            and cluster["lo"] > chance)
        # The registered SESOI turns both depth readouts from
        # absence-of-evidence into practical claims:
        #   * reaches_ceiling (saturation): one-sided equivalence - even the
        #     pool's *lower* bound lies within SESOI of the ceiling, so the
        #     rung is practically at the ceiling;
        #   * hb6_supported ("depth measurably required"): the pool's
        #     *upper* bound still sits at least SESOI below the ceiling, so
        #     the gap is practically meaningful at the optimistic end, not
        #     merely a statistically detectable sliver.
        # The historical bare-CI readings are kept as companions only.
        within_sesoi = bool(
            ceiling_known and above_chance and math.isfinite(cluster["lo"])
            and ceiling_acc - cluster["lo"] <= sesoi)
        reaches_ceiling_ci = bool(
            ceiling_known and above_chance and math.isfinite(cluster["hi"])
            and cluster["hi"] >= ceiling_acc)
        hb6_supported = bool(
            ceiling_known and above_chance and math.isfinite(cluster["hi"])
            and ceiling_acc - cluster["hi"] >= sesoi)
        hb6_supported_ci = bool(
            ceiling_known and above_chance and math.isfinite(cluster["hi"])
            and cluster["hi"] < ceiling_acc)
        ceiling_gap = (ceiling_acc - summary["mean"]
                       if ceiling_known else float("nan"))
        ceiling_gap_lower = (ceiling_acc - cluster["hi"]
                             if math.isfinite(cluster["hi"])
                             and ceiling_known else float("nan"))
        payload = {
            "name": f"shallow_none_d{depth}", "n_layers": depth,
            "val_seed": VAL_SEED, "test_seed": TEST_SEED,
            "chosen_lr": best["lr"], "chosen_steps": int(best["steps"]),
            "tuning_table": table, "selection": selection,
            "accuracy_mean": summary["mean"], "accuracy_std": summary["std"],
            # Per-seed accuracies behind ``accuracy_mean``: curve-invalid
            # runs are excluded exactly as they are from the mean/std and
            # from the rung's other decision statistics.  Keeping them here
            # would let E1 pair its valid deep runs against shallow runs that
            # the ladder itself marked as optimisation failures (the raw
            # runs remain available under ``runs`` for transparency).
            "accuracy_by_seed": {str(r["seed"]): r["accuracy"]
                                 for r in valid_runs},
            "curve_invalid_seeds": curve_invalid_seeds,
            "curve_valid_seeds": [int(r["seed"]) for r in valid_runs],
            "n_curve_invalid": len(curve_invalid_seeds),
            "pooled_wilson": pooled, "cluster_ci": cluster,
            "ceiling": ceiling_acc,
            "chance": chance, "above_chance": above_chance,
            "competence_alpha": competence_alpha,
            "sesoi_accuracy": sesoi,
            "ceiling_gap": ceiling_gap,
            # Smallest plausible gap to the ceiling (upper-bound side): the
            # SESOI gate for "depth measurably required" reads this.
            "ceiling_gap_lower": ceiling_gap_lower,
            "reaches_ceiling": within_sesoi,
            "reaches_ceiling_ci": reaches_ceiling_ci,
            "hb6_supported": hb6_supported,
            "hb6_supported_ci": hb6_supported_ci,
            "unmeasured": unmeasured,
            "runs": runs,
        }
        depths[depth] = payload
        tuning_tables[depth] = table
        if unmeasured:
            status = (f"unmeasured (all {len(runs)} final runs are "
                      f"curve-invalid: {curve_invalid_seeds}); this rung "
                      f"cannot support HB6/HB7 or the saturation depth")
        elif within_sesoi:
            status = (f"within SESOI {sesoi:g} of the ceiling (saturated: "
                      f"even the seed-clustered lower bound is inside the "
                      f"practical band, reach={ceiling_gap:+.4f})")
        elif not above_chance:
            status = (f"at chance (seed-clustered lower bound "
                      f"{cluster['lo']:.4f} <= chance {chance:.4f}): this "
                      f"rung did not learn the task at this step count, so it "
                      f"is not depth evidence")
        elif reaches_ceiling_ci:
            status = (f"CI touches the ceiling but the seed-clustered lower "
                      f"bound is more than SESOI={sesoi:g} below it "
                      f"(reach={ceiling_gap:+.4f}): saturation not "
                      f"established")
        else:
            status = "below the ceiling"
        print(f"[baselines] HB6/HB7 depth={depth} tuned standard "
              f"transformer (lr={best['lr']:g}, steps={int(best['steps'])} "
              f"on val seed {VAL_SEED}, test seed {TEST_SEED}): "
              f"accuracy={summary['mean']:.4f} +- "
              f"{summary['std']:.4f} (seed-clustered "
              f"[{cluster['lo']:.4f}, {cluster['hi']:.4f}], "
              f"n={cluster['n']} seeds/{cluster['n_examples']} examples; "
              f"pooled Wilson companion "
              f"[{pooled['lo']:.4f}, {pooled['hi']:.4f}]); "
              f"chance={chance:.4f}; rule ceiling={ceiling_acc:.4f}, gap="
              f"{ceiling_gap:.4f} (lower-bound gap="
              f"{ceiling_gap_lower:.4f}); params={int(runs[0]['params'])}; "
              f"{status}")
        _write_json(out_dir / f"baselines_depth{depth}.json", payload)

    ladder_competent = any(depths[d]["above_chance"] for d in ladder)
    unmeasured_rungs = [d for d in ladder if depths[d].get("unmeasured")]
    saturation_depth = next((d for d in ladder if depths[d]["reaches_ceiling"]),
                            None)
    saturation_depth_ci = next(
        (d for d in ladder if depths[d]["reaches_ceiling_ci"]), None)
    # HB7's competence premise: when no rung clears chance the ladder did
    # not solve the task at this step count, so the depth demand is "not
    # determinable" - never "required" (a None saturation depth alone must
    # not be read as "needs more depth").
    if not ceiling_known or not ladder_competent:
        hb7_depth_required = None
    elif saturation_depth is None:
        # An unmeasured rung could still saturate; without it the absence
        # of a saturation point cannot be read as "depth required".
        hb7_depth_required = None if unmeasured_rungs else True
    elif saturation_depth <= 2:
        # A *measured* saturation at depth <= 2 settles the registered
        # question ("is depth beyond two layers required?") even if a
        # shallower rung was unmeasured: d* can only move lower, so it
        # cannot become > 2.
        hb7_depth_required = False
    elif any(d < saturation_depth for d in unmeasured_rungs):
        # The measured saturation point is deeper than 2, but an
        # unmeasured *shallower* rung could itself have saturated within
        # SESOI.  In that case d* could be <= 2, so "depth beyond two
        # layers is required" is not determinable.  Unmeasured rungs
        # deeper than the measured saturation point cannot lower d*.
        hb7_depth_required = None
    else:
        hb7_depth_required = True
    adjacent = {}
    for a, b in zip(ladder, ladder[1:]):
        valid_a = [r for r in depths[a]["runs"]
                   if r.get("seed") is not None
                   and r.get("curve_verdict") not in CURVE_INVALID_VERDICTS]
        by_seed_b = {r["seed"]: r for r in depths[b]["runs"]
                     if r.get("seed") is not None
                     and r.get("curve_verdict") not in CURVE_INVALID_VERDICTS}
        # Positive mean = the deeper rung gains accuracy over the shallower
        # one, matching the ``a->b`` arrow and the "paired gain" reading
        # below (the TOST is symmetric, so only the reported sign moves).
        diffs = [by_seed_b[r["seed"]]["accuracy"] - r["accuracy"]
                 for r in valid_a if r["seed"] in by_seed_b]
        if len(diffs) < 2:
            # One paired seed gives a zero-width percentile interval and a
            # degenerate TOST (equivalence_test([0.0], bound) returns
            # equivalent=True), which is not a seed-clustered claim.
            adjacent[f"{a}->{b}"] = {
                "n": len(diffs), "sesoi_accuracy": sesoi,
                "equivalent": False, "unmeasured": True,
            }
            reason = ("no paired runs (unmeasured rung)" if not diffs else
                      "only one paired run; at least two independent pairs "
                      "are needed for the bootstrap/TOST read")
            print(f"[baselines] HB7 adjacent-depth contrast {a}->{b} layers: "
                  f"{reason}; skipped")
            continue
        ci = paired_bootstrap_ci(diffs)
        # The paired gain gets the same practical-significance reading as
        # the point rungs: a gain whose paired interval sits entirely
        # within +-SESOI is no longer *practically* meaningful even if its
        # point estimate is positive, matching the saturation rule.
        eq = equivalence_test(diffs, sesoi)
        adjacent[f"{a}->{b}"] = {
            **ci,
            "sesoi_accuracy": sesoi,
            "within_sesoi": bool(eq["equivalent"]),
            "tost_p_value": eq["p_value"],
            "tost_n": eq["n"],
        }
        print(f"[baselines] HB7 adjacent-depth contrast {a}->{b} layers "
              f"(paired over seeds; each rung independently val-tuned over "
              f"lr x steps, so this is a performance-envelope contrast, not "
              f"a controlled single-recipe depth effect): {_fmt_ci(ci)}; "
              f"companion TOST vs SESOI={sesoi:g}: "
              f"{eq['mean']:+.4f} [{eq['lo']:+.4f}, {eq['hi']:+.4f}] "
              f"n={eq['n']} p={eq['p_value']:.3f} "
              f"({'within SESOI' if eq['equivalent'] else 'beyond SESOI'})")
    shallowest = ladder[0]
    ladder_payload = {
        "name": "depth_ladder_none",
        "val_seed": VAL_SEED, "test_seed": TEST_SEED,
        "ladder": list(ladder), "ceiling": ceiling_acc,
        "sesoi_accuracy": sesoi,
        "saturation_rule": ("smallest competent rung whose seed-clustered "
                            "95% lower bound lies within SESOI of the rule "
                            "ceiling (one-sided equivalence); the bare "
                            "'interval reaches the ceiling' companion is "
                            "reaches_ceiling_ci only"),
        "depth_contrast_note": (
            "each rung is independently val-tuned over lr x steps, so "
            "adjacent-depth contrasts describe achievable performance "
            "envelopes rather than a controlled single-recipe depth effect; "
            "the selection payload records both axes and boundary flags"),
        "depths": {str(d): depths[d] for d in ladder},
        "tuning_tables": {str(d): tuning_tables[d] for d in ladder},
        "adjacent_contrasts": adjacent,
        "saturation_depth": saturation_depth,
        "saturation_depth_ci": saturation_depth_ci,
        "hb7_depth_required": hb7_depth_required,
        "chance": chance,
        "competence": {
            "criterion": ("at least one rung's seed-clustered one-sided 95% "
                          "family-wise lower bound above chance "
                          "(Bonferroni over the ladder rungs)"),
            "chance": chance,
            "alpha_per_rung": competence_alpha,
            "ladder_competent": ladder_competent,
            "above_chance_by_depth": {str(d): depths[d]["above_chance"]
                                      for d in ladder},
        },
    }
    sat_txt = (saturation_depth if saturation_depth is not None
               else "none (no rung reaches the ceiling)")
    if saturation_depth is None and saturation_depth_ci is not None:
        sat_txt = (f"none within SESOI {sesoi:g} (depth "
                   f"{saturation_depth_ci} only touches the ceiling - "
                   f"absence of evidence, not saturation)")
    if hb7_depth_required is None:
        if unmeasured_rungs:
            depth_txt = (f"not determinable - depth(s) {unmeasured_rungs} "
                         f"are unmeasured, so the ladder cannot establish "
                         f"whether depth is required")
        elif not ladder_competent:
            depth_txt = ("not determinable - no rung is above chance, so "
                         "the ladder did not solve the task at this step "
                         "count (raise the step count before reading depth)")
        elif not ceiling_known:
            depth_txt = "n/a (rule ceiling unavailable)"
        else:
            depth_txt = ("not determinable - an unmeasured rung could still "
                         "saturate, so the ladder cannot establish whether "
                         "depth is required")
    elif hb7_depth_required:
        depth_txt = "measurably required"
    else:
        depth_txt = ("NOT measurably required - annotate every depth-based "
                     "motivation")
    print(f"[baselines] HB7 saturation: depth ladder {list(ladder)} -> "
          f"saturation_depth={sat_txt}; depth beyond two layers is "
          f"{depth_txt}")
    # Legacy single-depth artifacts: E1 reads the shallow rung
    # (``shallow_none``) and its tuning table from the verdict.
    _write_json(out_dir / "baselines_shallow_none.json", depths[shallowest])
    _write_json(out_dir / "baselines_shallow_tuning.json",
                {"val_seed": VAL_SEED, "table": tuning_tables[shallowest],
                 "chosen_lr": depths[shallowest]["chosen_lr"],
                 "chosen_steps": depths[shallowest]["chosen_steps"],
                 "selection": depths[shallowest]["selection"]})
    _write_json(out_dir / "baselines_depth_ladder.json", ladder_payload)
    return {**ladder_payload, "depths": depths, "shallowest": shallowest}


def _matched_shallow_control(settings: Settings, out_dir: pathlib.Path,
                             ladder: dict) -> dict:
    """Train a parameter-matched 1-layer control for the depth ladder.

    The depth-only ladder holds width fixed, so a deep-vs-shallow gain
    confounds depth with parameter count.  This control trains a 1-layer
    ``gate="none"`` transformer widened until its parameter count is as
    close as possible to the deepest ladder rung, tunes its learning rate
    and step budget on the same validation seed/grid, and reports it on the
    sealed test stream.  It is a *capacity* control, not a mechanism contrast: depth and
    width both differ from the deep rung, so it answers only "could a
    shallow model with comparable capacity reach the deep model's accuracy?"
    (a no/yes answer changes how the depth framing is read, not the gate
    contrast itself).

    The width is chosen by binary search over ``d_model`` multiples of
    ``n_heads``; the payload records both parameter counts and the relative
    mismatch, and marks the control ``capacity_matched=False`` when no width
    in the search range reaches the target.  All selection uses VAL_SEED;
    the final runs use TEST_SEED exactly like the ladder rungs.
    """
    target_depth = int(ladder["ladder"][-1])
    target_payload = (ladder["depths"].get(target_depth)
                      or ladder["depths"].get(str(target_depth)))
    target_runs = target_payload.get("runs") or []
    recorded = [_positive_integer(r.get("params"))
                for r in target_runs if isinstance(r, Mapping)
                and "params" in r]
    target_params = (recorded[0] if recorded
                     and all(value is not None for value in recorded)
                     and len(set(recorded)) == 1 else None)
    base = {
        "name": "matched_shallow_none",
        "control_for_depth": target_depth,
        "target_params": target_params,
        "n_layers": 1,
        "val_seed": VAL_SEED, "test_seed": TEST_SEED,
        "chance": ladder.get("chance"),
        "ceiling": ladder.get("ceiling"),
        "sesoi_accuracy": ladder.get("sesoi_accuracy"),
        "comparison": "capacity_control",
        "note": ("parameter-matched width control: depth and width both "
                 "differ from the deep rung, so this is not a depth or "
                 "mechanism contrast; it only tests whether comparable "
                 "capacity at one layer closes the gap"),
    }
    if target_params is None:
        base.update({"d_model": None, "params": None,
                     "capacity_matched": False, "unmeasured": True,
                     "reason": "target rung has no recorded parameter count",
                     "runs": []})
        _write_json(out_dir / "baselines_matched_shallow.json", base)
        return base

    step = int(settings.n_heads)

    def _params_for(d_model):
        model = build(_model_cfg(settings, gate="none", n_layers=1,
                                 d_model=int(d_model)), compile=False)
        return int(sum(p.numel() for p in model.parameters()))

    # Binary search the smallest multiple of n_heads whose parameter count
    # reaches the target; also keep the previous multiple so the closest
    # (rather than only the first overshoot) can be selected.
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
    base.update({"d_model": int(wide_d), "params": int(wide_params),
                 "capacity_matched": matched,
                 "params_delta": int(wide_params - target_params),
                 "params_delta_frac": ((wide_params - target_params)
                                       / max(target_params, 1))})
    if not matched:
        print(f"[baselines] WARNING: matched-shallow control width d_model="
              f"{wide_d} gives {wide_params} params vs target "
              f"{target_params} (>5% mismatch); the control is recorded "
              f"but capacity_matched=False")

    tune_seeds = settings.seeds[:max(1, min(settings.min_seeds_for_claim,
                                             len(settings.seeds)))]
    step_candidates = _depth_step_candidates(settings)
    table = []
    for train_steps in step_candidates:
        for lr in settings.lr_grid:
            accs, verdicts, tuning_runs = [], [], []
            for seed in tune_seeds:
                _, params, metrics, curve, _ = _train_and_eval(
                    settings, seed, lr, eval_seed=VAL_SEED, selection=True,
                    steps=train_steps, gate="none", n_layers=1,
                    d_model=wide_d)
                accs.append(metrics["accuracy"])
                verdicts.append(curve["curve_verdict"])
                tuning_runs.append({
                    "seed": int(seed), "params": int(params),
                    "val_accuracy": metrics["accuracy"],
                    "eval_seed": int(VAL_SEED), "curve": curve})
            mean = sum(accs) / len(accs)
            std = ((sum((a - mean) ** 2 for a in accs)
                    / (len(accs) - 1)) ** 0.5 if len(accs) > 1 else 0.0)
            table.append({"lr": lr, "steps": int(train_steps),
                          "val_accuracy_mean": mean,
                          "val_accuracy_std": std, "n_seeds": len(accs),
                          "curve_verdicts": verdicts,
                          "curve_invalid_runs": sum(
                              v in CURVE_INVALID_VERDICTS
                              for v in verdicts),
                          "runs": tuning_runs})
    finite = [row for row in table if _finite(row["val_accuracy_mean"])]
    base.update({"tuning_table": table})
    if not finite:
        base.update({"chosen_lr": None, "chosen_steps": None,
                     "unmeasured": True, "above_chance": False, "runs": [],
                     "reason": "all tuning runs non-finite"})
        _write_json(out_dir / "baselines_matched_shallow.json", base)
        return base
    pick = _curve_valid_selection_rows(table)
    eligible = pick["rows"]
    selection = _selection_reliability(
        eligible, grid_keys=("lr", "steps"), full_rows=table)
    selection["pool"] = pick["pool"]
    best = max(eligible, key=lambda row: row["val_accuracy_mean"])
    warn = _selection_warn(
        "matched-shallow lr x steps selection", selection)
    if warn:
        print(f"[baselines] {warn}: the matched-capacity control may "
              f"understate one-layer capacity")
    runs = []
    for seed in settings.seeds:
        _, params, metrics, curve, _ = _train_and_eval(
            settings, seed, best["lr"], eval_seed=TEST_SEED, gate="none",
            n_layers=1, d_model=wide_d, steps=best["steps"])
        n = int(metrics["n"])
        runs.append({
            "seed": seed, "accuracy": metrics["accuracy"],
            "accuracy_lo": metrics["accuracy_lo"],
            "accuracy_hi": metrics["accuracy_hi"], "n": n,
            "correct": int(round(metrics["accuracy"] * n)),
            "correct_bitmask": metrics["correct_bitmask"],
            "stream_digest": metrics.get("stream_digest"),
            "params": params, "curve_verdict": curve["curve_verdict"],
        })
        _write_result(out_dir / "baselines_runs.jsonl",
                      {"experiment": "baselines", "hypotheses": ["HB6",
                                                                 "HB7"],
                       "name": "matched_shallow_none", "n_layers": 1,
                       "d_model": wide_d, "lr": best["lr"],
                       "steps": int(best["steps"]),
                       "training_seed": seed, "eval_seed": TEST_SEED,
                       "accuracy": metrics["accuracy"],
                       "accuracy_lo": metrics["accuracy_lo"],
                       "accuracy_hi": metrics["accuracy_hi"], "n": n,
                       "params": params, "capacity_matched": matched,
                       **_training_record(settings, best["lr"],
                                          best["steps"]),
                       **_run_context(settings), **curve})
    valid = [r for r in runs
             if r.get("curve_verdict") not in CURVE_INVALID_VERDICTS]
    cluster = clustered_proportion_ci(
        [r["correct"] for r in valid], [r["n"] for r in valid])
    summary = _mean_std([r["accuracy"] for r in valid])
    above = bool(cluster["valid"] and math.isfinite(cluster["lo"])
                 and cluster["lo"] > float(ladder.get("chance") or 0.0))
    ceiling = float(ladder.get("ceiling", float("nan")))
    base.update({
        "chosen_lr": best["lr"], "chosen_steps": int(best["steps"]),
        "selection": selection,
        "accuracy_mean": summary["mean"],
        "accuracy_std": summary["std"],
        "accuracy_by_seed": {str(r["seed"]): r["accuracy"] for r in valid},
        "correct_bitmask_by_seed": {
            str(r["seed"]): r["correct_bitmask"] for r in valid},
        "stream_digest_by_seed": {
            str(r["seed"]): r.get("stream_digest") for r in valid},
        "eval_n_by_seed": {str(r["seed"]): int(r["n"]) for r in valid},
        "cluster_ci": cluster, "above_chance": above,
        "unmeasured": not bool(valid),
        "ceiling_gap": (ceiling - summary["mean"]
                        if math.isfinite(ceiling) else float("nan")),
        "runs": runs,
    })
    print(f"[baselines] matched-capacity shallow control (1 layer, "
          f"d_model={wide_d}, {wide_params} params vs target "
          f"{target_params} for depth {target_depth}, "
          f"capacity_matched={matched}; lr={best['lr']:g}, "
          f"steps={int(best['steps'])} on val seed {VAL_SEED}, test seed "
          f"{TEST_SEED}): accuracy="
          f"{summary['mean']:.4f} +- {summary['std']:.4f} "
          f"(seed-clustered [{cluster['lo']:.4f}, {cluster['hi']:.4f}], "
          f"chance={float(ladder.get('chance') or 0.0):.4f}, n="
          f"{cluster['n']} seeds/{cluster['n_examples']} examples); "
          f"{'above chance' if above else 'NOT above chance'}")
    _write_json(out_dir / "baselines_matched_shallow.json", base)
    return base


class OverfitSanityError(RuntimeError):
    """The tiny-batch overfit pre-flight failed; grid training is refused."""


# Tiny-batch overfit configurations of the HB5 pre-flight: every gate
# variant plus the E3 architecture paths (norm placement, dynamic
# normalisation, LayerScale).  A broken backward on a normalisation or
# write-strength path would survive a gate-only check yet make every E3
# number uninterpretable, so those paths get the same memorisation test.
_SANITY_CONFIGS = (
    ("gate=none", {"gate": "none"}),
    ("gate=query", {"gate": "query"}),
    ("gate=calib", {"gate": "calib"}),
    ("gate=null", {"gate": "null"}),
    ("gate=calib,norm=post", {"gate": "calib", "norm": "post"}),
    ("gate=calib,norm=mix", {"gate": "calib", "norm": "mix"}),
    ("gate=calib,norm=seednorm", {"gate": "calib", "norm": "seednorm"}),
    ("gate=calib,layer_scale=1e-2", {"gate": "calib", "layer_scale": 1e-2}),
)
# The plain baseline arm whose row supplies the shared batch/step/lr fields.
_SANITY_REFERENCE = "gate=none"


def _sanity_configs(settings: Settings) -> tuple:
    """Per-path HB5 configurations, including the configured model depth.

    The static matrix isolates gate/normalisation/LayerScale paths on a
    2-layer miniature so the pre-flight stays cheap.  A backward problem
    that only appears once activations traverse the configured depth would
    still pass that matrix while every grid run at ``settings.n_layers``
    fails, so the two architectures the grids actually depend on (the
    default Pre-LN arm and the E3 seednorm arm, both with the calibrated
    gate) are repeated at the real depth.
    """
    configs = list(_SANITY_CONFIGS)
    depth = int(settings.n_layers)
    if depth > 2:
        for norm in ("pre", "seednorm"):
            configs.append((
                f"gate=calib,norm={norm},n_layers={depth}",
                {"gate": "calib", "norm": norm, "n_layers": depth}))
    return tuple(configs)


def _overfit_sanity_by_config(settings: Settings) -> dict:
    """Tiny-batch memorisation check for every trained path (cached).

    One fixed batch from ``SANITY_SEED`` is pushed through the model of each
    configuration with dropout off and weight decay zeroed, so every custom
    kernel's backward path *and* every normalisation / LayerScale path the
    grids train is exercised - a passing ``gate=none`` check alone would not
    prove the calib/null gates or the seednorm/Post-LN/LayerScale paths
    backprop.  Beyond the cheap 2-layer matrix, ``_sanity_configs`` repeats
    the default and seednorm arms at ``settings.n_layers`` so depth-specific
    gradient/loss failures are caught before a grid starts.  Cached on
    ``Settings``: E1/E2/E3 in one ``--only all`` run share a single
    execution.
    """
    cache_key = _config_cache_key(settings)
    cached = getattr(settings, "_overfit_sanity_cache", None)
    if cached is not None and getattr(
            settings, "_overfit_sanity_cache_key", None) == cache_key:
        return cached
    # The sanity batch is trained on, so it must never alias a held-out
    # stream even though it is a single non-reported batch.
    check_train_seed(SANITY_SEED)
    gen = stream_generator(SANITY_SEED, settings.device)
    sidx, stgt, smeta = make_batch(settings.overfit_batch, settings.n_pairs,
                                   settings.vocab, settings.device, gen=gen)
    validate_batch(sidx, stgt, smeta, settings.vocab, diagnostics=False)
    by_config = {}
    # The miniature width is 48 whenever that is a valid head multiple, but
    # an arbitrary ``n_heads`` (e.g. 5 or 7) must not make the HB5 pre-flight
    # itself unbuildable: round up to the next multiple of the head count.
    heads = max(1, int(settings.n_heads))
    mini_d_model = 48
    if mini_d_model % heads:
        mini_d_model = heads * ((mini_d_model + heads - 1) // heads)
    for label, overrides in _sanity_configs(settings):
        seed_everything(SANITY_SEED, deterministic=settings.deterministic)
        tiny = build(_model_cfg(
            settings, **{"d_model": mini_d_model, "n_layers": 2, **overrides}),
            compile=False)
        row = overfit_sanity_check(
            tiny, sidx, stgt, smeta, lr=settings.overfit_lr,
            steps=settings.overfit_steps, seed=SANITY_SEED,
            vocab=settings.vocab, deterministic=settings.deterministic)
        row["model_cfg"] = asdict(tiny.cfg)
        row["params"] = int(sum(p.numel() for p in tiny.parameters()))
        row["stream_seed"] = SANITY_SEED
        row["n_pairs"] = int(settings.n_pairs)
        row["vocab"] = int(settings.vocab)
        row["device"] = settings.device
        row["deterministic"] = bool(settings.deterministic)
        by_config[label] = row
    settings._overfit_sanity_cache = by_config
    settings._overfit_sanity_cache_key = cache_key
    return by_config


def _enforce_overfit_gate(settings: Settings, out_dir: pathlib.Path,
                          tag: str) -> dict:
    """Refuse to start a grid when the loop cannot memorise a tiny batch.

    Runs the check on the *current* code revision (a stale
    ``baselines_verdict.json`` cannot satisfy it), stores the per-config
    evidence under ``<tag>_overfit_sanity.json`` and aborts with
    :class:`OverfitSanityError` when any configuration fails: an invalid
    training loop would make every number a grid produces unreliable, so
    the grid is not started.
    """
    by_config = _overfit_sanity_by_config(settings)
    passed = all(row["passed"] for row in by_config.values())
    sanity = by_config[_SANITY_REFERENCE]
    summary = {
        "passed": passed,
        "batch": int(sanity["batch"]),
        "steps": int(sanity["steps"]),
        "lr": float(sanity["lr"]),
        "seed": sanity["seed"],
        "stream_seed": sanity["stream_seed"],
        "n_pairs": sanity["n_pairs"],
        "vocab": sanity["vocab"],
        "device": sanity["device"],
        "deterministic": sanity["deterministic"],
        "by_config": {
            label: {
                "passed": row["passed"],
                "learned": row["learned"],
                "loss_first": row["loss_first"],
                "loss_last": row["loss_last"],
                "acc_first": row["acc_first"],
                "acc_last": row["acc_last"],
                "steps_to_target_loss": row["steps_to_target_loss"],
                "steps_to_target_acc": row["steps_to_target_acc"],
                "target_loss": row["target_loss"],
                "target_acc": row["target_acc"],
                "curve": row["curve"],
                "seeding": dict(row.get("seeding") or {}),
                "data_sha256": row["data_sha256"],
                "params": row["params"],
                "model_cfg": row["model_cfg"],
                "params_connected": (
                    f"{row['grad_flow']['params_with_grad']}/"
                    f"{row['grad_flow']['params_total']}"),
                "all_grads_finite": row["grad_flow"]["all_grads_finite"],
                "active_dropout": row["regularization"]["active_dropout"],
                "dropout_probs": row["regularization"]["dropout_probs"],
                "weight_decay": row["regularization"]["weight_decay"],
                "augmentation": row["regularization"]["augmentation"],
            }
            for label, row in by_config.items()
        },
    }
    _write_json(out_dir / f"{tag}_overfit_sanity.json",
                {**summary, **_run_context(settings)})
    print(f"[{tag}] overfit sanity gate (HB5; tiny batch of "
          f"{summary['batch']} samples, {summary['steps']} steps, lr="
          f"{summary['lr']:g}, seed {summary['seed']}, data "
          f"{sanity['data_sha256']}; dropout off, weight decay zeroed, no "
          f"augmentation):")
    for label, row in summary["by_config"].items():
        loss_s = (f"{row['loss_first']:.4f} -> {row['loss_last']:.4f}"
                  if _finite(row["loss_first"]) and _finite(row["loss_last"])
                  else "non-finite")
        print(f"    {label:>28}: loss {loss_s}, batch acc "
              f"{row['acc_last']:.3f}, steps to loss/acc target "
              f"{row['steps_to_target_loss']}/{row['steps_to_target_acc']}, "
              f"grads {row['params_connected']} "
              f"finite={row['all_grads_finite']}; "
              f"{'passed' if row['passed'] else 'FAILED'}")
    if not passed:
        failed = [label for label, row in summary["by_config"].items()
                  if not row["passed"]]
        _write_json(out_dir / f"{tag}_aborted.json",
                    {"experiment": tag,
                     "reason": "HB5 tiny-batch overfit gate failed",
                     "failed_configs": failed,
                     "overfit_sanity": summary,
                     **_run_context(settings)})
        raise OverfitSanityError(
            f"{tag}: tiny-batch overfit sanity failed for {failed}; "
            f"address the training loop before running the grid (this "
            f"check is necessary, not sufficient: data/label problems can "
            f"still survive memorisation)")
    return summary
