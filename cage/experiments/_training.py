# cage/experiments/_training.py - Curve records, training/evaluation helpers and recipe resolution.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _steps_run_from_hist(hist):
    """Best-effort optimizer-step count when ``early_stop_info`` is absent."""
    rows = []
    for row in _mapping_runs(hist):
        step = _difficulty_count(row.get("step"))
        if step is not None:
            rows.append((step, row))
    if not rows:
        return 0
    skipped = sum(
        1 for _, row in rows
        if isinstance(row.get("diagnosis"), dict)
        or not _finite(row.get("train_loss")))
    return max(0, max(step for step, _ in rows) + 1 - skipped)


def _curve_optional_float(value):
    """Optional numeric curve field: None stays None, malformed -> None.

    A stored curve is JSON provenance, so a hand-edited/corrupted optional
    field must make that series point abstain rather than abort the whole
    run record.  ``None`` is the existing sentinel for "not recorded" in the
    curve series, so a malformed value uses the same representation.  A
    boolean is a flag rather than a measurement, and a non-finite value
    cannot be plotted or summarized as a curve point; both abstain.
    """
    if value is None or isinstance(value, bool):
        return None
    if torch.is_tensor(value) and (value.is_complex()
                                   or value.dtype == torch.bool):
        # ``float`` on a complex tensor either drops the imaginary part or
        # raises RuntimeError when it is non-zero; neither is a curve
        # measurement.  A boolean tensor is a flag, not a value.
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _curve_record(hist, early_stop_info=None, seeding=None):
    """Per-run learning-curve summary (monitor the curve, not just the end).

    Stores both the reduced verdict and the raw series (steps, train loss,
    validation loss, validation accuracy, pre-clip gradient norm), so a run
    that is later read as overfitting/plateauing can be re-examined without
    retraining.
    ``steps_run`` records the number of optimizer steps actually executed
    (from ``train_model``'s ``early_stop_info``, or the logged history as a
    fallback) and ``stopped_early``/``restored_best`` record whether the
    fixed epoch step count was cut short.  The controlled audit reads these:
    ``training_recipe.steps`` is the *intended* step count, so without this
    stamp a future early-stop could compare runs trained for different
    numbers of steps while every recipe field still matched.

    The staged non-finite triage (``diagnose_nonfinite``) is persisted too:
    the training loop prints it at the step it fires, but a console line is
    not part of the record, so a later reader of ``curve_verdict =
    numerical_instability`` could not tell whether the data, the
    activations, the loss or the gradients were blamed.  Each report is
    stored with its step under ``diagnoses`` / ``val_diagnoses`` (the
    validation-side triage runs on the exact batch that produced the bad
    value), and the steps are listed under ``nonfinite_steps``.

    ``seeding`` is the actual :func:`cage.repro.seed_everything` result
    returned for this run (Python/NumPy/Torch/CUDA coverage, pinned
    ``PYTHONHASHSEED``, deterministic-kernel/TF32 state).  It rides with
    the curve so a record proves which RNGs this run actually pinned
    instead of only citing the static policy.
    """
    # Materialize once and drop rows that cannot carry a curve point: a
    # generator would be consumed by the verdict and then look empty to the
    # step count, while a scalar row used to raise AttributeError/KeyError.
    hist = [row for row in _mapping_runs(hist)
            if _difficulty_count(row.get("step")) is not None]
    early_stop_info = (early_stop_info
                       if isinstance(early_stop_info, Mapping) else {})
    seeding = seeding if isinstance(seeding, Mapping) else {}
    verdict = training_curve_verdict(hist)
    train = [h for h in hist if _finite(h.get("train_loss"))]
    val = [h for h in hist if h.get("val_loss") is not None
           and _finite(h.get("val_loss"))]
    val_acc = [h for h in hist if h.get("val_acc") is not None
               and _finite(h.get("val_acc"))]

    def _diagnoses(key):
        return [{"step": int(h["step"]), **h[key]} for h in hist
                if isinstance(h.get(key), dict)]

    diagnoses = _diagnoses("diagnosis")
    val_diagnoses = _diagnoses("val_diagnosis")
    nonfinite_steps = sorted({int(h["step"]) for h in hist
                              if isinstance(h.get("diagnosis"), dict)
                              or isinstance(h.get("val_diagnosis"), dict)
                              or not _finite(h.get("train_loss"))
                              or (h.get("val_loss") is not None
                                  and not _finite(h.get("val_loss")))
                              or (h.get("grad_norm") is not None
                                  and not _finite(h.get("grad_norm")))})
    return {
        "train_loss_first": train[0]["train_loss"] if train else float("nan"),
        "train_loss_last": train[-1]["train_loss"] if train else float("nan"),
        "val_loss_first": val[0]["val_loss"] if val else float("nan"),
        "val_loss_last": val[-1]["val_loss"] if val else float("nan"),
        "val_acc_first": val_acc[0]["val_acc"] if val_acc else float("nan"),
        "val_acc_last": val_acc[-1]["val_acc"] if val_acc else float("nan"),
        "best_val_acc": max(float(h["val_acc"]) for h in val_acc)
        if val_acc else float("nan"),
        "train_delta": verdict["train_delta"],
        "val_delta": verdict["val_delta"],
        "generalization_gap": verdict["generalization_gap"],
        "best_val_loss": verdict["best_val_loss"],
        "best_val_step": verdict["best_val_step"],
        "val_regression_after_best": verdict["val_regression_after_best"],
        "late_overfit": verdict["late_overfit"],
        "late_overfit_significant": verdict["late_overfit_significant"],
        "overfitting_significant": verdict["overfitting_significant"],
        "late_train_delta": verdict["late_train_delta"],
        "late_val_delta": verdict["late_val_delta"],
        "val_loss_se_last": verdict["val_loss_se_last"],
        "val_change_abs": verdict["val_change_abs"],
        "val_noise_2se": verdict["val_noise_2se"],
        "val_movement_significant": verdict["val_movement_significant"],
        "nonfinite_steps": nonfinite_steps,
        "diagnoses": diagnoses,
        "val_diagnoses": val_diagnoses,
        "lr_first": verdict["lr_first"],
        "lr_last": verdict["lr_last"],
        "grad_norm_max": verdict["grad_norm_max"],
        "grad_norm_explode_count": verdict["grad_norm_explode_count"],
        "curve_steps": [int(h["step"]) for h in train],
        "curve_train_loss": [float(h["train_loss"]) for h in train],
        "curve_train_loss_window": [
            _curve_optional_float(h.get("train_loss_window"))
            for h in train],
        "curve_val_loss": [_curve_optional_float(h.get("val_loss"))
                           for h in train],
        "curve_val_loss_se": [_curve_optional_float(h.get("val_loss_se"))
                              for h in train],
        "curve_val_acc": [_curve_optional_float(h.get("val_acc"))
                          for h in train],
        "curve_grad_norm": [_curve_optional_float(h.get("grad_norm"))
                            for h in train],
        "curve_lr": [_curve_optional_float(h.get("lr")) for h in train],
        "curve_verdict": verdict["verdict"],
        "curve_recommendation": verdict["recommendation"],
        "steps_run": int(
            _difficulty_count(early_stop_info.get("steps_run"))
            if _difficulty_count(early_stop_info.get("steps_run")) is not None
            else _steps_run_from_hist(hist)),
        "early_stop_enabled": bool(
            early_stop_info.get("enabled", False)),
        "stopped_early": bool(
            early_stop_info.get("stopped_early", False)),
        "restored_best": bool(
            early_stop_info.get("restored_best", False)),
        "seeding": dict(seeding),
    }


def _train_and_eval(settings: Settings, seed: int, lr: float, eval_seed: int,
                    permute_labels: bool = False, selection: bool = False,
                    steps: int | None = None, **model_overrides):
    """Train + score one run through the single shared entry point.

    ``eval_seed`` is mandatory so every call site states which stream it
    reads; a defaulted sealed test stream would let a future tuning call
    silently fit decisions to the final numbers.  ``selection=True`` marks
    a call whose metrics inform a choice (recipe/baseline tuning) and is
    refused on the sealed test stream by ``check_eval_seed``.

    ``steps`` optionally overrides the shared recipe's step budget for
    tuned baselines (the depth ladder and its capacity control); ``None``
    keeps ``settings.steps`` exactly as before, so every controlled grid
    still trains one frozen recipe.

    Returns ``(model, params, metrics, curve, model_cfg)`` where
    ``model_cfg`` is the exact resolved configuration the model was built
    from - every ``Cfg`` field including defaults, so an unset ablation flag
    cannot be mistaken for an off switch.  Run records keep it so the
    controlled-variable audit can check the *trained* configuration
    directly (ground truth) instead of trusting the hand-attached labels.
    """
    check_eval_seed(eval_seed, selection=selection)
    if stream_seed_key(eval_seed) in {
            stream_seed_key(s) for s in settings.seeds}:
        raise ValueError(
            f"eval_seed={eval_seed} is also a configured training seed; "
            f"evaluation data must come from a stream that never trained "
            f"the model"
        )
    raw_train_steps = settings.steps if steps is None else steps
    train_steps = _positive_integer(raw_train_steps)
    if train_steps is None:
        raise ValueError(
            f"steps must be a positive integer, got {raw_train_steps!r}")
    raw_curve_points = settings.curve_points
    if isinstance(raw_curve_points, bool):
        raise ValueError(
            f"curve_points must be a non-negative integer, got "
            f"{raw_curve_points!r}")
    try:
        curve_points_num = float(raw_curve_points)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            f"curve_points must be a non-negative integer, got "
            f"{raw_curve_points!r}") from exc
    if not (math.isfinite(curve_points_num) and curve_points_num >= 0.0
            and curve_points_num == int(curve_points_num)):
        raise ValueError(
            f"curve_points must be a non-negative integer, got "
            f"{raw_curve_points!r}")
    curve_points = int(curve_points_num)
    seed_everything(seed, deterministic=settings.deterministic)
    model = build(_model_cfg(settings, **model_overrides), compile=False)
    model_cfg = asdict(model.cfg)
    params = sum(p.numel() for p in model.parameters())
    if curve_points <= 0:
        eval_every, eval_steps = 0, None
    else:
        # ``curve_points`` is the number of checks per run (Settings).  An
        # arithmetic ``eval_every`` cannot hit that count for every
        # (steps, curve_points) pair (e.g. 12 steps with 10 points), so
        # pass the exact evenly-spaced schedule.  A run cannot have more
        # checks than iterations; cap there instead of duplicating steps.
        points = min(curve_points, train_steps)
        if points == 1:
            eval_steps = {train_steps - 1}
        else:
            eval_steps = {
                int(round(i * (train_steps - 1) / (points - 1)))
                for i in range(points)}
        eval_every = 0
    early_stop_info = {}
    seeding = {}
    hist = train_model(model, steps=train_steps, batch=settings.batch,
                       n_pairs=settings.n_pairs, lr=lr, vocab=settings.vocab,
                       device=settings.device, log_every=0, seed=seed,
                       eval_every=eval_every, eval_steps=eval_steps,
                       val_batches=settings.val_batches,
                       permute_labels=permute_labels,
                       early_stop_info=early_stop_info,
                       deterministic=settings.deterministic,
                       seed_info=seeding)
    metrics = eval_metrics(model, settings.n_pairs, settings.vocab,
                           settings.device, batch=settings.eval_batch,
                           batches=settings.eval_batches, seed=eval_seed,
                           train_association_codes=getattr(
                               settings, "_train_association_codes", None))
    return model, params, metrics, _curve_record(
        hist, early_stop_info, seeding), model_cfg


def tune_shared_recipe(settings: Settings):
    """Tune the learning rate on the plain ``none`` baseline.

    Selection uses the validation seed only, over the pre-registered
    ``min_seeds_for_claim`` floor's worth of training seeds (the grid never
    freezes a recipe on fewer seeds than a claim would need - a two-seed
    argmax is inside the seed-to-seed noise); the winning rate is then
    frozen for *every* variant, so a gate variant can never win via a
    luckier optimizer setting - the E1/E3 comparisons remain controlled.

    The learning-curve verdict of every tuning run is stored with the
    table: if the selected rate's runs are all curve-invalid (plateau /
    diverging / instability) the selection was made between underfit or
    broken runs, and the frozen recipe cannot be trusted to expose the
    mechanism under test - ``_resolve_lr`` prints that warning.  Each row
    also carries the per-seed validation accuracy and full curve record, so
    the selection runs are traceable experiments themselves rather than
    only an aggregate mean/verdict.

    Returns ``(best_lr, table, selection)`` where ``selection`` is the
    :func:`_selection_reliability` audit of the argmax: even at the claim
    floor the winning rate can sit inside the seed-to-seed spread of a
    neighbour, and that point selection must be recorded rather than
    printed as if the grid had resolved the recipe.
    """
    tune_seeds = settings.seeds[:max(1, min(settings.min_seeds_for_claim,
                                             len(settings.seeds)))]
    table = []
    for lr in settings.lr_grid:
        accs, verdicts, runs = [], [], []
        for seed in tune_seeds:
            _, params, metrics, curve, _ = _train_and_eval(
                settings, seed, lr, eval_seed=VAL_SEED, selection=True,
                gate="none")
            accs.append(metrics["accuracy"])
            verdicts.append(curve["curve_verdict"])
            runs.append({"seed": int(seed), "params": int(params),
                         "val_accuracy": metrics["accuracy"],
                         "eval_seed": int(VAL_SEED),
                         "curve": curve})
        mean = sum(accs) / len(accs)
        var = (sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)
               if len(accs) > 1 else 0.0)
        table.append({"lr": lr, "val_accuracy_mean": mean,
                      "val_accuracy_std": var ** 0.5, "n_seeds": len(accs),
                      "curve_verdicts": verdicts,
                      "curve_invalid_runs": sum(
                          v in CURVE_INVALID_VERDICTS for v in verdicts),
                      "runs": runs})
    # Same all-NaN guard as the depth ladder: ``max`` over an all-NaN key
    # returns an arbitrary row, which would freeze an unmeasured recipe for
    # every controlled contrast - refuse instead of fabricating a selection.
    finite_rows = [row for row in table
                   if _finite(row["val_accuracy_mean"])]
    if not finite_rows:
        raise RuntimeError(
            f"shared recipe tuning produced no finite validation accuracy "
            f"across lr grid {tuple(settings.lr_grid)}: every tuning run "
            f"diverged or returned NaN, so no learning rate can be frozen "
            f"for the controlled grids (inspect the run records; no "
            f"downstream contrast from this run is interpretable)")
    pick = _curve_valid_selection_rows(table)
    if pick["n_excluded"]:
        print(f"[baselines] shared recipe selection excluded "
              f"{pick['n_excluded']} of {pick['n_finite']} finite lr row(s) "
              f"with curve-invalid runs; selecting among curve-valid rows "
              f"only")
    elif pick["n_invalid"]:
        print(f"[baselines] WARNING: every finite lr row has curve-invalid "
              f"runs; falling back to all {pick['n_finite']} finite row(s) - "
              f"the frozen recipe is not validated by a healthy curve")
    eligible = pick["rows"]
    best = max(eligible, key=lambda row: row["val_accuracy_mean"])
    selection = _selection_reliability(
        eligible, grid_keys=("lr",), full_rows=table)
    selection["pool"] = pick["pool"]
    selection["excluded_curve_invalid_rows"] = pick["n_excluded"]
    selection["curve_invalid_rows"] = pick["n_invalid"]
    return best["lr"], table, selection


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
    else:
        print(f"[{tag}] reusing shared tuned recipe lr={cached[0]:g}")
    lr, table, selection = cached
    warn = _selection_warn(f"shared recipe selection lr={lr:g}", selection)
    if warn:
        print(f"[{tag}] {warn}: the frozen recipe may make every variant "
              f"look equally flat")
    _write_json(out_dir / f"{tag}_lr_tuning.json",
                {"val_seed": VAL_SEED, "table": table, "chosen_lr": lr,
                 "selection": selection})
    return lr


def _answer_balance(settings: Settings, batch=256, batches=4, seeds=None):
    """Class-balance diagnostic for the answer values, fitted on the
    training streams only.

    Method contract: every fitted statistic is computed on the data the
    model is trained on, never on a held-out split.  The held-out answer
    distributions are instead *scored against* this training reference by
    the HB3 split audit (per-field chi-square shift tests), so
    validation/test/stats data is transform-only here.  The complementary
    held-out *uniformity* check (does the evaluation stream itself satisfy
    the accuracy premise?) is the separate scoring-only audit in
    :func:`_eval_answer_balance`; no training statistic is fitted there
    either.  ``seeds`` defaults
    to the configured training seeds (``(0,)`` when none are configured,
    matching the training convention elsewhere); each seed is refused if
    it is a reserved evaluation seed.

    The returned ``uniformity`` block is the *decision* behind using
    accuracy as the headline metric: a Monte Carlo-calibrated chi-square
    test of the drawn answer counts against the uniform expectation over
    ids ``2..vocab-1`` (PAD/SEP are never answers).  Shares alone cannot
    separate sampling noise from a real shift at ~10 expected samples per
    class; the test can (see ``cage.data.answer_uniformity_test``).  When
    it fails, the verdict layer flags the balance premise and the
    PR-AUC/F1 companions - not accuracy - carry the claims.  Every drawn
    batch passes the shared ``validate_batch`` cleaning gate first, so the
    balance decision is computed on the same validated data contract as
    training and the reported metrics.
    """
    if seeds is None:
        seeds = tuple(settings.seeds) if settings.seeds else (0,)
    counts = torch.zeros(settings.vocab, dtype=torch.float64)
    per_seed = {}
    for seed in seeds:
        check_train_seed(seed)
        gen = stream_generator(seed, settings.device)
        n_seed = 0
        for _ in range(batches):
            idx, targets, meta = make_batch(batch, settings.n_pairs,
                                            settings.vocab,
                                            settings.device, gen=gen)
            validate_batch(idx, targets, meta, settings.vocab, diagnostics=False)
            counts += torch.bincount(meta["answer"].cpu(),
                                     minlength=settings.vocab)
            n_seed += meta["answer"].numel()
        per_seed[int(seed)] = int(n_seed)
    present = counts > 0
    shares = counts[present] / counts.sum()
    uniformity = answer_uniformity_test(counts[2:])
    return {
        "train_seeds": [int(s) for s in seeds],
        "batches_per_seed": int(batches),
        "samples_per_seed": per_seed,
        "n_samples": int(counts.sum()),
        "n_classes_possible": settings.vocab - 2,
        "n_classes_present": int(present.sum()),
        "min_class_share": (float(shares.min()) if shares.numel()
                            else float("nan")),
        "max_class_share": (float(shares.max()) if shares.numel()
                            else float("nan")),
        "expected_uniform_share": 1.0 / (settings.vocab - 2),
        "uniformity": uniformity,
        "power_note": (
            "gross-imbalance test only: a pass at "
            f"{uniformity['expected_per_class']:.2f} expected samples/class "
            "does not certify the absence of a small or moderate skew"),
    }


# The held-out streams whose class balance backs the accuracy readout.
# VAL/TEST are consumed at ``eval_batch x eval_batches`` by ``eval_metrics``;
# STATS is consumed at ``stats_batch x stats_batches`` by
# ``collect_stats``.  The linear probe is scored on its own, deliberately
# wider geometry (``probe_eval_batch x probe_eval_batches``) and its HB2
# near-chance certificate is an accuracy claim, so the probe VAL/TEST
# streams are audited at that geometry too - the eval-geometry prefix is a
# different realization and cannot certify them.  The audit reproduces each
# consumer's exact geometry and generator seed, so the tested sample is the
# sample the models are scored on - a wider audit draw would test a
# different prefix and could pass while the reported stream is skewed.
# E2's longer sweep lengths are audited separately in
# ``_eval_answer_balance`` because the stream realization depends on length.
_EVAL_BALANCE_STREAMS = (
    ("val", VAL_SEED, "eval"),
    ("test", TEST_SEED, "eval"),
    ("stats", STATS_SEED, "stats"),
    ("probe_val", VAL_SEED, "probe"),
    ("probe_test", TEST_SEED, "probe"),
)


def _eval_answer_balance(settings: Settings) -> dict:
    """Class-balance audit for the *evaluation* answer streams.

    ``_answer_balance`` tests the training streams because the train-time
    class prior is what the model sees.  Accuracy is also the headline on
    the held-out streams, and that choice is only justified when those
    streams are uniform too: a skewed TEST stream turns every reported
    accuracy (and the E1/E2/E3 accuracy contrasts) into a class-prior
    metric.  The HB3 split audit compares each held-out marginal to the
    *training* reference, which is a different question - a uniformly
    skewed training stream and an equally skewed test stream can pass the
    shift test while both violate the accuracy premise.

    The audit is scoring-only: it draws from the fixed evaluation seeds and
    validates each batch through the shared cleaning gate, but fits nothing
    and selects nothing.  The Monte Carlo calibration and the finite-stream
    caveat are the same as ``_answer_balance``; the per-stream
    ``expected_per_class`` is reported so a small quick-mode draw is not
    mistaken for a well-powered balance certificate.

    E2 scores accuracy at every length in its sweep, so the TEST stream is
    also audited at each sweep length (the stream realization depends on
    the length).  The pre-registered E2 readouts use only the base and
    longest endpoints; the serialized per-length rows make a mid-sweep
    imbalance visible without letting an exploratory length gate the
    endpoint decision.
    """
    def _audit_stream(seed, length, batch, batches):
        check_eval_seed(seed)
        gen = stream_generator(seed, settings.device)
        counts = torch.zeros(settings.vocab, dtype=torch.float64)
        for _ in range(max(0, int(batches))):
            idx, targets, meta = make_batch(batch, length,
                                            settings.vocab,
                                            settings.device, gen=gen)
            validate_batch(idx, targets, meta, settings.vocab,
                           diagnostics=False)
            counts += torch.bincount(meta["answer"].cpu(),
                                     minlength=settings.vocab)
        present = counts > 0
        shares = counts[present] / counts.sum() if counts.sum() > 0 else \
            counts[present]
        uniformity = answer_uniformity_test(counts[2:])
        return {
            "stream_seed": int(seed),
            "seed_key": stream_seed_key(seed),
            "length": int(length),
            "batch": int(batch),
            "batches": int(batches),
            "n": int(counts.sum()),
            "n_classes_possible": settings.vocab - 2,
            "n_classes_present": int(present.sum()),
            "min_class_share": (float(shares.min()) if shares.numel()
                                else float("nan")),
            "max_class_share": (float(shares.max()) if shares.numel()
                                else float("nan")),
            "expected_uniform_share": 1.0 / (settings.vocab - 2),
            "uniformity": uniformity,
            "power_note": (
                "gross-imbalance test only: a pass at "
                f"{uniformity['expected_per_class']:.2f} expected "
                "samples/class does not certify the absence of a small or "
                "moderate skew"),
        }

    streams = {}
    for label, seed, geometry in _EVAL_BALANCE_STREAMS:
        if geometry == "stats":
            batch, batches = settings.stats_batch, settings.stats_batches
        elif geometry == "probe":
            batch, batches = (settings.probe_eval_batch,
                              settings.probe_eval_batches)
        else:
            batch, batches = settings.eval_batch, settings.eval_batches
        streams[label] = _audit_stream(seed, settings.n_pairs, batch, batches)
    all_uniform = all(row["uniformity"]["uniform_supported"]
                      for row in streams.values())

    try:
        e2_lengths = length_sweep_lengths(
            base=settings.n_pairs, factor=2, n=4,
            max_len=settings.max_len)
    except ValueError as exc:
        e2_sweep = {
            "lengths": [], "streams": {}, "endpoint_lengths": [],
            "endpoint_uniform": None, "all_uniform": None,
            "error": str(exc),
        }
    else:
        e2_streams = {}
        for length in e2_lengths:
            e2_streams[str(length)] = (
                dict(streams["test"]) if length == settings.n_pairs else
                _audit_stream(TEST_SEED, length, settings.eval_batch,
                              settings.eval_batches))
        endpoints = [settings.n_pairs, e2_lengths[-1]]
        e2_sweep = {
            "lengths": [int(L) for L in e2_lengths],
            "streams": e2_streams,
            "endpoint_lengths": [int(L) for L in endpoints],
            "endpoint_uniform": bool(all(
                e2_streams[str(L)]["uniformity"]["uniform_supported"]
                for L in endpoints)),
            "all_uniform": bool(all(
                row["uniformity"]["uniform_supported"]
                for row in e2_streams.values())),
        }
    return {
        "streams": streams,
        "all_uniform": bool(all_uniform),
        "e2_length_sweep": e2_sweep,
        "geometry": {
            "eval_batch": int(settings.eval_batch),
            "eval_batches": int(settings.eval_batches),
            "stats_batch": int(settings.stats_batch),
            "stats_batches": int(settings.stats_batches),
            "probe_eval_batch": int(settings.probe_eval_batch),
            "probe_eval_batches": int(settings.probe_eval_batches),
        },
        "note": ("uniformity is a gross-imbalance test at the reported "
                 "stream size, not an equivalence proof; expected_per_class "
                 "is printed so a low-power draw is not read as a "
                 "certificate"),
    }


class SplitAuditError(RuntimeError):
    """The HB3 split/disjointness audit failed; no grid may run on this data."""


# Minimum samples per stream for the calibrated shift tests.  The audit is
# not the evaluation stream: reusing a tiny quick-mode eval geometry (e.g.
# 64 x 2 = 128 rows, ~21 per key position) leaves the tests noisy and
# underpowered, so a legitimate random draw can land on the wrong side of
# the decision boundary.  Drawing at least this many rows at the eval batch
# size keeps the audit's decisions stable without touching the evaluation
# data (the audit only reads additional samples from the same split
# streams).
_SPLIT_AUDIT_MIN_SAMPLES = 1024


def _audit_data_splits(settings: Settings, out_dir: pathlib.Path,
                       tag: str = "baselines", enforce: bool = False) -> dict:
    """HB3: verify every stream pair is disjoint and field-wise IID before
    any model is trained.

    The audit only reads data (it selects nothing), but a shared sample
    between *any* two streams - validation vs test included - or a
    distribution gap between the training set and a held-out split
    invalidates every number reported afterwards, so the outcome is printed
    and stored under ``<tag>_split_audit.json``.  Beyond the
    ``split_leakage_audit`` prefix comparison, ``actual_stream_overlap_audit``
    replays every stream at the geometry its consumer actually reads
    (training at ``batch`` x ``steps``, eval at ``eval_batch`` x
    ``eval_batches``, ...) and the gate also refuses any cross-split content
    overlap found there: the prefix check alone cannot cover the full
    training stream, which ``make_batch``'s shape-dependent draws make a
    different realization from the eval-geometry prefix.  Every configured training
    seed enters the audit as its own stream and the exact-overlap check
    covers all stream pairs.  Each audit field's shift is decided by a Monte
    Carlo-calibrated Pearson chi-square against the pooled training
    distribution (see ``cage.data.split_leakage_audit``), Benjamini-
    Hochberg-adjusted over the field x split family at ``settings.fdr_q``:
    a total-variation readout against ``3 x noise floor`` cannot fail for
    sparse fields (a few hundred answer classes at ~1e3 samples) because
    the tolerance approaches the maximum possible TV, so the calibrated
    test - not an arbitrary constant - carries the decision.  The
    TV/floor pair stays in the report as a descriptive companion.  The
    replay also compares canonical stream seed keys: a split is a
    seed-owned stream, so the same seed reused by two groups is leakage
    even when shape-dependent draws leave their row fingerprints disjoint.

    A second, train-fitted gate covers what the marginal tests cannot: an
    *atypical-sample* tail.  ``split_leakage_audit`` scores every sample by
    its training-fitted surprisal over the structural audit fields and
    thresholds at the training tail quantile; a held-out split whose
    atypical rate exceeds five times the training rate (and an absolute 5%
    floor) fails HB3 even when every marginal passes.  Separately, a
    train-only first-half/second-half comparison reports training-stream
    non-stationarity (BH-adjusted within its own family); it warns rather
    than blocks, since drift is a property to disclose.

    The audit is deterministic for one ``Settings`` (fixed split seeds and
    a fixed Monte Carlo seed), so the computation runs once per process
    and every experiment that depends on the protocol writes its
    own copy.  ``enforce=True`` (baselines, E1/E2/E3 and the standalone
    pre-flight) turns a failed audit into a hard
    :class:`SplitAuditError`: a protocol with a shared sample or a
    held-out distribution shift cannot support any grid result, so the
    grid is not started (same policy as the reproducibility and overfit
    gates).
    """
    train_seeds = tuple(settings.seeds) or (0,)
    cache_key = _config_cache_key(settings)
    audit = getattr(settings, "_split_audit_cache", None)
    fresh = (audit is None
             or getattr(settings, "_split_audit_cache_key", None) != cache_key)
    if fresh:
        audit_batches = max(
            int(settings.eval_batches),
            math.ceil(_SPLIT_AUDIT_MIN_SAMPLES
                      / max(int(settings.eval_batch), 1)))
        audit = split_leakage_audit(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.eval_batch, batches=audit_batches,
            train_seeds=train_seeds)
        fields = audit["audit_fields"]
        family = [(field, name) for field in fields
                  for name in ("val", "test", "stats")]
        bh = benjamini_hochberg(
            [audit["shift_p_value"][field][name] for field, name in family],
            q=settings.fdr_q)
        shift_tests = {
            f"{field}_train_{name}": {
                "p": item["p"], "p_adj": item["p_adj"],
                "reject": item["reject"],
                "chi2": audit["shift_chi2"][field][name],
            }
            for (field, name), item in zip(family, bh)
        }
        audit["shift_p_value_bh"] = shift_tests
        # Prefix overlap is necessary but not sufficient: training reads
        # at least settings.steps * settings.batch samples at its own
        # geometry, which is a different realization from the
        # eval-geometry prefix.  The depth ladder and its capacity control
        # reuse the same seed-owned training stream for up to the largest
        # _depth_step_candidates budget, so replay that full prefix; a
        # longer in-order prefix subsumes every shorter training budget.
        # Replay the exact streams each consumer reads (training, val
        # monitoring, final val/test, stats, probe train/val/test) and
        # refuse any content overlap or cross-group seed reuse between
        # different splits.  Realizations of the same split (val monitoring
        # vs final val, probe val vs val) must remain allowed.
        training_steps = max(_depth_step_candidates(settings))
        actual = actual_stream_overlap_audit(
            settings.n_pairs, settings.vocab, settings.device,
            train_specs={
                f"train:{seed}": {
                    "seed": int(seed), "batch": settings.batch,
                    "steps": training_steps, "group": "train",
                }
                for seed in train_seeds
            } | {
                f"probe_train:{seed}": {
                    "seed": int(seed), "batch": settings.probe_batch,
                    "steps": max(settings.probe_steps_grid
                                 or (settings.probe_steps,)),
                    "group": "train",
                }
                for seed in train_seeds
            } | {
                "overfit_sanity": {
                    "seed": SANITY_SEED, "batch": settings.overfit_batch,
                    "steps": 1, "group": "train",
                },
            },
            heldout_specs={
                "val_monitor": {
                    "seed": VAL_SEED, "batch": settings.batch,
                    "batches": settings.val_batches, "group": "val",
                },
                "val": {
                    "seed": VAL_SEED, "batch": settings.eval_batch,
                    "batches": settings.eval_batches, "group": "val",
                },
                # The linear probe selects lr x steps on the probe-geometry
                # validation stream (``tune_linear_probe``), which is a
                # different realization from the eval-geometry ``val``
                # stream.  Replaying only ``probe_test`` would leave the
                # stream that actually drives a selection unaudited.
                "probe_val": {
                    "seed": VAL_SEED, "batch": settings.probe_eval_batch,
                    "batches": settings.probe_eval_batches, "group": "val",
                },
                "test": {
                    "seed": TEST_SEED, "batch": settings.eval_batch,
                    "batches": settings.eval_batches, "group": "test",
                },
                "stats": {
                    "seed": STATS_SEED, "batch": settings.stats_batch,
                    "batches": settings.stats_batches, "group": "stats",
                },
                "probe_test": {
                    "seed": TEST_SEED, "batch": settings.probe_eval_batch,
                    "batches": settings.probe_eval_batches,
                    "group": "test",
                },
            },
            include_train_association_codes=True)
        # The code set is not JSON-serializable and can be large; keep it on
        # the in-process Settings so exposure-stratified evaluation can use
        # it, and persist only the aggregate exposure readout below.
        settings._train_association_codes = frozenset(
            actual.pop("train_association_codes", ()))
        audit["actual_stream_overlap"] = actual
        audit["max_actual_stream_overlap"] = actual["max_overlap"]
        audit["actual_stream_seed_collisions"] = actual[
            "cross_group_seed_collisions"]
        audit["train_association_summary"] = actual[
            "train_association_summary"]
        audit["association_exposure"] = actual["association_exposure"]
        audit["max_heldout_pair_exposure"] = actual[
            "max_heldout_pair_exposure"]
        # Training-stationarity diagnostic (train-only halves): reported and
        # BH-adjusted across its own family, but it warns rather than blocks,
        # because a drifting training stream is a property to disclose, not
        # a leakage failure of the split.
        stationarity = audit.get("train_stationarity", {})
        st_keys = sorted(stationarity)
        st_bh = benjamini_hochberg(
            [stationarity[key]["p"] for key in st_keys],
            q=settings.fdr_q) if st_keys else []
        audit["train_stationarity_bh"] = {
            key: {"p": item["p"], "p_adj": item["p_adj"],
                  "reject": item["reject"],
                  "tv": stationarity[key]["tv"]}
            for key, item in zip(st_keys, st_bh)
        }
        # Statistical rejection alone would fire on ~5% of same-distribution
        # audits (a p-value family at q=0.05 must reject something that
        # often).  Require the descriptive effect size as well: the
        # first/second-half TV gap must clear the train-only noise floor by
        # the historical 3x margin.  Drift is then disclosed only when it is
        # both significant and materially larger than stream-to-stream noise.
        audit["train_drift_suspected"] = bool(any(
            item["reject"]
            and item.get("tv", 0.0) > 3.0 * audit["tv_noise_floor"].get(
                key.rsplit(":", 1)[-1], 0.0)
            for key, item in audit["train_stationarity_bh"].items()))
        outlier = audit.get("outlier_audit", {})
        outlier_shift = outlier.get("shift_by_split", {})
        audit["outlier_shift_detected"] = bool(any(outlier_shift.values()))
        audit["hb3_supported"] = bool(
            audit["max_pairwise_duplicate_overlap"] == 0
            and actual["max_overlap"] == 0
            and not actual["cross_group_seed_collisions"]
            and not any(item["reject"] for item in shift_tests.values())
            # A held-out split whose difficulty *tail* has moved passes the
            # marginal tests; the conservative train-fitted tail-rate bound
            # catches it before it can masquerade as a mechanism effect.
            and not audit["outlier_shift_detected"])
        settings._split_audit_cache = audit
        settings._split_audit_cache_key = cache_key

    fields = audit["audit_fields"]
    floors = audit["tv_noise_floor"]
    shift_tests = audit["shift_p_value_bh"]
    rejected = [key for key, item in shift_tests.items() if item["reject"]]
    if fresh:
        def _tv_max(name):
            return max(audit[f"{field}_tv_train_{name}"] for field in fields)

        detail = "; ".join(
            f"{name}: tv_max={_tv_max(name):.5f}"
            for name in ("val", "test", "stats"))
        floor_txt = ", ".join(f"{field}={floors[field]:.5f}" for field in fields)
        adj_txt = ", ".join(
            f"{field}_train_{name}={shift_tests[f'{field}_train_{name}']['p_adj']:.3f}"
            for field in fields for name in ("val", "test", "stats"))
        dup_txt = ", ".join(
            f"{name}={audit[f'n_duplicate_rows_{name}']}"
            for name in ("val", "test", "stats"))
        outlier = audit.get("outlier_audit", {})
        outlier_txt = ", ".join(
            f"{name}={outlier['rate_by_split'][name]:.4f}"
            for name in ("val", "test", "stats")) if outlier else "n/a"
        coverage = audit.get("support_coverage", {})
        cov_txt = ", ".join(
            f"{field}: train_support="
            f"{coverage[field]['fraction_seen_train']:.3f} "
            + "/".join(
                f"{name}={coverage[field]['unseen_by_split'][name]['rate']:.4f}"
                for name in ("val", "test", "stats"))
            for field in ("answer", "token", "key", "value")
            if field in coverage) if coverage else "n/a"
        actual = audit["actual_stream_overlap"]
        print(f"[{tag}] split audit (HB3, train seeds {list(train_seeds)}): "
              f"{detail}; pairwise overlap max="
              f"{audit['max_pairwise_duplicate_overlap']} over "
              f"{len(audit['pairwise_duplicate_overlap'])} stream pairs; "
              f"actual stream replay={actual['n_rows_total']} rows, "
              f"cross-split overlap max={actual['max_overlap']}, "
              f"cross-group seed collisions="
              f"{actual['cross_group_seed_collisions'] or 'none'}; "
              f"TV noise floor [{floor_txt}]; calibrated shift test "
              f"(BH q={settings.fdr_q:g}) adjusted p: [{adj_txt}]; "
              f"atypical-sample rate train="
              f"{outlier.get('rate_train', float('nan')):.4f} "
              f"(bound={outlier.get('rate_bound', float('nan')):.4f}) "
              f"[{outlier_txt}]; train stationarity min p="
              f"{audit.get('train_stationarity_min_p', 1.0):.3f}; "
              f"unseen-value rates [{cov_txt}]; "
              f"stream duplicates train="
              f"{audit['n_duplicate_rows_train']} ({dup_txt}); "
              f"unique_key_rate={audit['unique_key_rate']:.4f}")
    else:
        print(f"[{tag}] split audit (HB3, train seeds {list(train_seeds)}): "
              f"deterministic result reused; hb3_supported="
              f"{audit['hb3_supported']}")
    exposure = audit.get("association_exposure") or {}
    if exposure:
        train_assoc = audit.get("train_association_summary") or {}
        cells = ", ".join(
            f"{name}={row['pair_seen_train_rate']:.4f} "
            f"(key_seen={row['key_seen_train_rate']:.4f})"
            for name, row in sorted(exposure.items()))
        print(f"[{tag}] association exposure (exact query key-value pair "
              f"seen in a training stream): {cells}; training pair-space "
              f"coverage={train_assoc.get('pair_space_coverage', float('nan')):.4f} "
              f"({train_assoc.get('n_unique_pairs', 0)}/"
              f"{train_assoc.get('n_possible_pairs', 0)} possible pairs). "
              f"This is a memorisation-opportunity disclosure, not a "
              f"leakage verdict; exposure-stratified accuracy is stamped "
              f"on every metric record.")
    if rejected:
        print(f"[{tag}] WARNING: calibrated shift test rejects "
              f"{', '.join(rejected)}: held-out counts differ from the "
              f"pooled training distribution beyond same-stream sampling "
              f"noise")
    if audit.get("outlier_shift_detected"):
        flagged = [f"{name}={audit['outlier_audit']['rate_by_split'][name]:.4f}"
                   for name, bad in
                   audit["outlier_audit"]["shift_by_split"].items() if bad]
        print(f"[{tag}] WARNING: atypical-sample tail rate exceeds the "
              f"train-fitted bound (bound="
              f"{audit['outlier_audit']['rate_bound']:.4f}): "
              f"{', '.join(flagged)}; the held-out difficulty tail differs "
              f"from training")
    if audit.get("train_drift_suspected"):
        drift = [key for key, item in
                 audit.get("train_stationarity_bh", {}).items()
                 if item["reject"]
                 and item.get("tv", 0.0) > 3.0 * audit["tv_noise_floor"].get(
                     key.rsplit(":", 1)[-1], 0.0)]
        print(f"[{tag}] WARNING: training-stream non-stationarity suspected "
              f"({', '.join(drift)}): first and second halves of the "
              f"training stream differ beyond sampling noise, so a single "
              f"held-out split may not represent the future")
    if not audit["hb3_supported"]:
        print(f"[{tag}] WARNING: HB3 split/disjointness audit failed; "
              f"all reported numbers are suspect")
    _write_json(out_dir / f"{tag}_split_audit.json", audit)
    if enforce and not audit["hb3_supported"]:
        _write_json(out_dir / f"{tag}_aborted.json", {
            "experiment": tag,
            "reason": "HB3 split/disjointness audit failed",
            "rejected_shift_tests": rejected,
            "outlier_shift_detected": audit.get("outlier_shift_detected"),
            "train_drift_suspected": audit.get("train_drift_suspected"),
            "max_pairwise_duplicate_overlap":
                audit["max_pairwise_duplicate_overlap"],
            "max_actual_stream_overlap":
                audit.get("max_actual_stream_overlap"),
            "cross_group_seed_collisions":
                audit.get("actual_stream_seed_collisions"),
            "split_audit": audit,
            **_run_context(settings),
        })
        raise SplitAuditError(
            f"{tag}: HB3 failed (max pairwise duplicate overlap="
            f"{audit['max_pairwise_duplicate_overlap']}, max actual-stream "
            f"overlap={audit.get('max_actual_stream_overlap')}, cross-group "
            f"seed collisions="
            f"{audit.get('actual_stream_seed_collisions') or 'none'}, "
            f"rejected shift tests={rejected or 'none'}); a data protocol "
            f"with leakage or a held-out distribution shift cannot support "
            f"any grid result")
    return audit
