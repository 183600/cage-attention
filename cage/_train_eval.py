# cage train: evaluation, counterfactual/association and error-detection helpers.
# Executed into the wrapper module namespace by the loader;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _association_exposure_summary(totals) -> dict:
    """Rates and flat record fields for the exposure-stratified readout."""
    raw_totals = totals if isinstance(totals, Mapping) else {}
    fields = tuple(_blank_exposure_cohort())

    def _parse_cohort(entry):
        if not isinstance(entry, Mapping):
            return None
        parsed = {}
        for name in fields:
            count = _nonnegative_count(entry.get(name))
            if count is None:
                return None
            parsed[name] = count
        n = parsed["n"]
        if not (
            parsed["n_correct"] <= n
            and parsed["n_value_swap"] <= n
            and parsed["n_value_correct"] <= parsed["n_value_swap"]
            and parsed["n_value_follow"] <= min(
                parsed["n_correct"], parsed["n_value_correct"])
            # Stubborn rows only exist where the value-swap probe ran, and
            # a stubborn row keeps the clean answer while a value-correct
            # row emits the (different) swapped answer, so the two groups
            # are disjoint inside the swappable rows.
            and parsed["n_value_stubborn"] <= parsed["n_value_swap"]
            and (parsed["n_value_stubborn"] + parsed["n_value_correct"]
                 <= parsed["n_value_swap"])
            and parsed["n_value_stubborn"] <= parsed["n_correct"]
            # Follow and stubborn are disjoint clean-correct subsets.
            and (parsed["n_value_follow"] + parsed["n_value_stubborn"]
                 <= parsed["n_correct"])
            # Inclusion-exclusion: clean-correct and value-swap-correct
            # intersect exactly on the follow rows.
            and (parsed["n_correct"] + parsed["n_value_correct"]
                 - parsed["n_value_follow"] <= n)
            and parsed["n_distractor"] <= n
            and parsed["n_distractor_correct"] <= parsed["n_distractor"]
            and parsed["n_relocation"] <= n
            and parsed["n_relocation_correct"] <= parsed["n_relocation"]
        ):
            return None
        return parsed

    cohorts = {}
    for name in ("seen", "unseen"):
        if name in raw_totals:
            cohorts[name] = _parse_cohort(raw_totals[name])
        else:
            # A cohort with no rows is a genuine zero-row stratum, not a
            # malformed row: it abstains through NaN rates.
            cohorts[name] = {field: 0 for field in fields}
    if any(entry is None for entry in cohorts.values()):
        # One malformed cohort would make the seen-minus-unseen gap
        # incomparable, so abstain for both instead of publishing a number
        # built from fabricated zeros.
        cohorts = {name: {field: 0 for field in fields}
                   for name in ("seen", "unseen")}

    def rate(num, den):
        # Exact integer counts above 2**53 (or even above the float range)
        # must not be routed through float separately: Python's int/int
        # division scales first and still returns the representable ratio.
        return num / den if den else float("nan")

    def cohort(name):
        entry = cohorts[name]
        n = entry["n"]
        n_correct = entry["n_correct"]
        n_swap = entry["n_value_swap"]
        n_distractor = entry["n_distractor"]
        n_relocation = entry["n_relocation"]
        return {
            "n": n,
            "n_correct": n_correct,
            "accuracy": rate(n_correct, n),
            "n_value_swappable": n_swap,
            "n_value_correct": entry["n_value_correct"],
            "cf_value_accuracy": rate(entry["n_value_correct"], n_swap),
            "n_value_follow": entry["n_value_follow"],
            "cf_value_follow_rate": rate(entry["n_value_follow"], n_correct),
            "n_value_stubborn": entry["n_value_stubborn"],
            "cf_value_stubborn_rate": rate(
                entry["n_value_stubborn"], n_correct),
            "n_distractor": n_distractor,
            "distractor_accuracy": rate(
                entry["n_distractor_correct"], n_distractor),
            "n_relocation": n_relocation,
            "relocation_accuracy": rate(
                entry["n_relocation_correct"], n_relocation),
        }

    seen, unseen = cohort("seen"), cohort("unseen")
    n_items = seen["n"] + unseen["n"]
    gap = (seen["accuracy"] - unseen["accuracy"]
           if math.isfinite(seen["accuracy"])
           and math.isfinite(unseen["accuracy"]) else float("nan"))
    out = {
        "definition": (
            "exposure cohort = whether the exact evaluated (query key, "
            "answer value) association was realized by a training stream"),
        "reference": "training-stream association replay",
        "n_items": n_items,
        "n_seen": seen["n"],
        "n_unseen": unseen["n"],
        "seen_rate": rate(seen["n"], n_items),
        "seen": seen,
        "unseen": unseen,
        "accuracy_gap_seen_minus_unseen": gap,
        "note": (
            "descriptive pre-treatment stratification, not a causal "
            "contrast: items whose association occurred in training can in "
            "principle be answered by memorisation, items whose exact pair "
            "did not must be resolved from context.  Read the clean and "
            "counterfactual accuracy together; a large seen-minus-unseen "
            "gap with high clean but low value-swap accuracy is evidence "
            "that clean accuracy is not purely in-context retrieval."),
    }
    out["flat"] = {
        "assoc_exposure_n": int(n_items),
        "assoc_exposure_n_seen": int(seen["n"]),
        "assoc_exposure_n_unseen": int(unseen["n"]),
        "assoc_exposure_seen_rate": out["seen_rate"],
        "assoc_exposure_accuracy_seen": seen["accuracy"],
        "assoc_exposure_accuracy_unseen": unseen["accuracy"],
        "assoc_exposure_accuracy_gap": gap,
        "assoc_exposure_cf_value_accuracy_seen": seen["cf_value_accuracy"],
        "assoc_exposure_cf_value_accuracy_unseen": unseen["cf_value_accuracy"],
        "assoc_exposure_cf_value_follow_rate_seen":
            seen["cf_value_follow_rate"],
        "assoc_exposure_cf_value_follow_rate_unseen":
            unseen["cf_value_follow_rate"],
    }
    return out


def _association_exposure_dose_summary(totals) -> dict:
    """Rates for the fixed training-exposure dose bins.

    Counts are kept exact until this final step, so an empty bin abstains
    with NaN instead of contributing a fabricated zero.  The table is a
    dose-response companion to the binary seen/unseen block: it asks
    whether accuracy changes as the same exact association is replayed
    more often in the training streams.
    """
    entries = totals if isinstance(totals, (list, tuple)) else ()
    rows = []
    n_items = n_correct = exposure_total = max_exposure = 0
    for label, entry in zip(_EXPOSURE_DOSE_LABELS, entries):
        n = int(entry["n"])
        correct = int(entry["n_correct"])
        n_items += n
        n_correct += correct
        exposure_total += int(entry["exposure_total"])
        max_exposure = max(max_exposure, int(entry["max_exposure"]))
        rows.append({
            "bin": label,
            "n": n,
            "n_correct": correct,
            "accuracy": correct / n if n else float("nan"),
            "mean_train_count": (
                int(entry["exposure_total"]) / n if n else float("nan")),
            "max_train_count": int(entry["max_exposure"]),
        })
    seen_n = sum(row["n"] for row in rows[1:])
    seen_correct = sum(row["n_correct"] for row in rows[1:])
    unseen = rows[0]
    seen_accuracy = seen_correct / seen_n if seen_n else float("nan")
    gap = (seen_accuracy - unseen["accuracy"]
           if seen_n and unseen["n"] else float("nan"))
    return {
        "definition": (
            "training-exposure dose = number of times the exact evaluated "
            "(query key, answer value) association was drawn by a "
            "training-group stream"),
        "bins": rows,
        "n_items": int(n_items),
        "n_seen": int(seen_n),
        "seen_rate": seen_n / n_items if n_items else float("nan"),
        "seen_accuracy": seen_accuracy,
        "seen_minus_unseen_accuracy": gap,
        "mean_train_count": (
            exposure_total / n_items if n_items else float("nan")),
        "max_train_count": int(max_exposure),
        "note": (
            "pre-treatment descriptive dose-response, not a causal "
            "contrast: repeated training exposure is not randomised, so "
            "read the per-bin accuracy next to the counterfactual "
            "value-swap rates before attributing an accuracy gain to "
            "memorisation."),
    }


@torch.inference_mode()
def eval_metrics(model, n_pairs, vocab, device, batch=256, batches=4,
                 seed=VAL_SEED, topk=5, train_association_codes=None,
                 train_association_counts=None):
    """Full metric panel at the query position.

    Same evaluation protocol as ``eval_accuracy`` (eval mode restored
    afterwards, independent fixed seed) extended with the explicitly
    named F1 variants, top-k accuracy, NLL and calibration metrics for the
    answer distribution.  ``seed`` defaults to the same fixed 123 as
    ``eval_accuracy`` so the two functions score identical batches;
    experiment scripts pass an explicit *validation* seed for model
    selection and a separate *test* seed for the reported numbers.

    The recall task is balanced multi-class, so ``accuracy`` is the
    primary metric; ``f1_micro`` equals it by construction and the
    macro/weighted variants guard against value-frequency skew.  See
    ``cage/metrics.py`` for the averaging conventions.

    ``accuracy_lo`` / ``accuracy_hi`` are the Wilson 95% interval for the
    accuracy over the finite evaluation stream: seed-level std quantifies
    training variance, while this interval quantifies how precisely the
    *reported* stream of ``batch * batches`` samples pins the accuracy
    down (the two are different uncertainties and both belong next to a
    headline number).  Every batch passes the ``validate_batch`` cleaning
    gate, so the metric panel is only computed on data that satisfies the
    same integrity/label invariants as training.

    The same pass also runs the deterministic minimal-pair probes
    (:func:`_counterfactual_edits`): swap the queried value, swap a
    distractor value, and relocate the queried pair.  The probe rates
    separate in-context retrieval from memorisation, fixed-position copying
    and global edit sensitivity without drawing a second evaluation stream;
    see ``out['counterfactual']`` and the flat ``cf_*`` scalars.

    ``train_association_codes`` optionally supplies the exact ``(key,
    value)`` association codes realized by the *training* streams (from
    :func:`cage.data.actual_stream_overlap_audit`).  When present, every
    item is classified as "seen" (its exact association occurred in
    training, so memorisation could answer it) or "unseen" (it could not),
    and clean / value-swap accuracy is reported per cohort under
    ``out['association_exposure']``.  The split is pre-treatment: it depends
    only on the training data, never on the model's answer, so it can expose
    an accuracy gain that is concentrated on memorisable pairs instead of
    being attributed to in-context retrieval.

    ``train_association_counts`` optionally supplies the training replay
    count for every association code (from the same audit).  When present,
    ``out['association_exposure_dose']`` bins items by that count (0, 1,
    2, 3-4, 5-9, 10+) and reports accuracy per bin.  This separates a
    single-exposure effect from a dose-response: if clean accuracy keeps
    rising with repeated training exposure, the binary seen/unseen
    contrast is pooling materially different memorisation opportunities.
    Passing counts alone also enables the binary exposure block, because
    its reference set is the mapping's key set.  If both references are
    given they must describe the same exact association set.
    """
    batch, batches = _validated_batch_counts(batch, batches)
    gen = torch.Generator(device=device).manual_seed(stream_seed_key(seed))
    # Keep the stratum encoding independent of the current length so the
    # same id means the same relative position across the whole sweep (the
    # cross-length standardization relies on that).  With fewer pairs than
    # the default tertile count some bins are simply unrealizable; report
    # them separately rather than changing the encoding or dividing the
    # coverage by cells that can never be observed.
    n_key_bins = DEFAULT_DIFFICULTY_KEY_BINS
    n_realizable_key_bins = min(n_key_bins, max(1, int(n_pairs)))
    reference_codes = (None if train_association_codes is None else
                       association_reference_codes(train_association_codes))
    reference_counts = (None if train_association_counts is None else
                        association_reference_counts(
                            train_association_counts))
    if reference_counts is not None:
        count_codes = frozenset(reference_counts)
        if reference_codes is None:
            reference_codes = count_codes
        elif reference_codes != count_codes:
            raise ValueError(
                "train_association_codes and train_association_counts "
                "describe different exact association sets")
    training_states = {module: module.training for module in model.modules()}
    try:
        model.eval()
        logits_all, tgt_all, strata_all = [], [], []
        digest_batches = []
        cf_totals = None
        exposure_totals = {}
        exposure_dose_totals = (
            _blank_exposure_dose() if reference_counts is not None else None)
        for i in range(batches):
            idx, tgt, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
            validate_batch(idx, tgt, meta, vocab, diagnostics=False)
            # Keep the validated realization so the metric record can
            # version the *whole* evaluation stream, not only its first
            # batch.  The tensors are tiny next to the logits panel.
            digest_batches.append((idx, tgt, meta))
            logits, _, _ = model(idx)
            panel = logits.squeeze(1)
            if panel.dtype != torch.float64:
                panel = panel.float()
            logits_all.append(panel)
            tgt_all.append(meta["answer"])
            strata_all.append(difficulty_strata(
                idx, meta["answer"], n_pairs,
                n_key_bins=n_key_bins)["stratum"].cpu())
            # Minimal-pair data edits on the *validated* realization.  The
            # edited contexts go through the same cleaning gate, so a probe
            # can never compare predictions on malformed data.
            edits = _counterfactual_edits(idx, meta, n_pairs, vocab)
            pred = logits.squeeze(1).argmax(-1)
            swap_pred = distractor_pred = relocation_pred = None
            if edits is not None:
                validate_batch(edits["value_swap"], None,
                               edits["value_swap_meta"], vocab,
                               diagnostics=False)
                swap_logits, _, _ = model(edits["value_swap"])
                swap_pred = swap_logits.squeeze(1).argmax(-1)
                if edits["distractor_swap"] is not None:
                    validate_batch(edits["distractor_swap"], None,
                                   edits["distractor_swap_meta"], vocab,
                                   diagnostics=False)
                    distractor_logits, _, _ = model(edits["distractor_swap"])
                    distractor_pred = distractor_logits.squeeze(1).argmax(-1)
                if edits["relocation"] is not None:
                    validate_batch(edits["relocation"], None,
                                   edits["relocation_meta"], vocab,
                                   diagnostics=False)
                    relocation_logits, _, _ = model(edits["relocation"])
                    relocation_pred = relocation_logits.squeeze(1).argmax(-1)
                counts = _counterfactual_counts(
                    pred, meta["answer"], swap_pred,
                    edits["value_swap_answer"], distractor_pred,
                    relocation_pred)
                if cf_totals is None:
                    cf_totals = counts
                else:
                    for key in cf_totals:
                        cf_totals[key] += counts[key]
            if reference_codes is not None:
                _association_exposure_accumulate(
                    exposure_totals, idx, meta, vocab, reference_codes,
                    pred, meta["answer"], swap_pred=swap_pred,
                    swap_answer=(None if edits is None
                                 else edits["value_swap_answer"]),
                    distractor_pred=distractor_pred,
                    relocation_pred=relocation_pred)
            if reference_counts is not None:
                _association_exposure_dose_accumulate(
                    exposure_dose_totals, idx, meta, vocab,
                    reference_counts, pred, meta["answer"])
        if logits_all:
            logits_cat = torch.cat(logits_all, dim=0).cpu()
            tgt_cat = torch.cat(tgt_all, dim=0).cpu()
            strata_cat = torch.cat(strata_all, dim=0).cpu()
        else:
            logits_cat = torch.empty((0, vocab))
            tgt_cat = torch.empty((0,), dtype=torch.long)
            strata_cat = torch.empty((0,), dtype=torch.long)
        # Reported metrics must be as traceable as training: if any evaluation
        # row carries non-finite logits, its accuracy/NLL/calibration values are
        # meaningless.  Stamp the finite fraction on every metric record and say
        # so loudly once, instead of silently averaging poisoned rows.
        finite_rows = torch.isfinite(logits_cat).all(dim=-1)
        out = multiclass_metrics(logits_cat, tgt_cat, topk=topk)
        out["finite_logit_frac"] = float(finite_rows.float().mean())
        if not bool(finite_rows.all()):
            print(f"[eval] WARNING: {int((~finite_rows).sum())}/"
                  f"{finite_rows.numel()} evaluation rows have non-finite "
                  f"logits; accuracy/NLL/calibration over those rows are not "
                  f"interpretable (see cage.train.diagnose_nonfinite)")
        acc_ci = wilson_ci(int((logits_cat.argmax(-1) == tgt_cat).sum()),
                           int(tgt_cat.numel()))
        out["accuracy_lo"] = acc_ci["lo"]
        out["accuracy_hi"] = acc_ci["hi"]
        out["nll"] = float(torch.nn.functional.cross_entropy(logits_cat, tgt_cat))
        probs = logits_cat.softmax(dim=-1)
        confidence, pred = probs.max(dim=-1)
        # Calibration is the research object, so the headline ECE is
        # accompanied by a bin-count sensitivity readout (a single bin width
        # can manufacture or hide a gap) and by the Murphy decomposition of
        # the Brier score.  The decomposition separates a reliability change
        # from a resolution/easy-stream change, which the raw Brier scalar
        # cannot: an arm that is less accurate but better calibrated must not
        # be read as uniformly worse.
        correct_rows_cal = pred == tgt_cat
        out["ece_10bin"] = expected_calibration_error(
            confidence, correct_rows_cal, n_bins=10)
        out["ece_15bin"] = expected_calibration_error(
            confidence, correct_rows_cal, n_bins=15)
        out["ece_20bin"] = expected_calibration_error(
            confidence, correct_rows_cal, n_bins=20)
        out["brier"] = brier_score(confidence, correct_rows_cal)
        brier_parts = brier_decomposition(
            confidence, correct_rows_cal, n_bins=15)
        out["brier_decomposition"] = brier_parts
        out["brier_reliability"] = brier_parts["reliability"]
        out["brier_resolution"] = brier_parts["resolution"]
        out["brier_uncertainty"] = brier_parts["uncertainty"]
        out["brier_skill_score"] = brier_parts["brier_skill_score"]
        # Item-level correctness in a compact, JSON-safe form: the shared
        # finite stream is a real source of uncertainty for paired contrasts
        # (see ``cage.experiments._paired_diff``).
        out["correct_bitmask"] = encode_correct_mask(pred == tgt_cat)
        # Content digest of the exact evaluation realization the mask was
        # computed on.  The item-level McNemar / fold companions compare
        # correctness *by position*; without a digest, two runs that
        # evaluated different data at the same item count would be paired
        # as if they had scored identical items.
        stream_digest = stream_content_digest(digest_batches)
        out["stream_digest"] = stream_digest["sha256"]
        out["stream_n_batches"] = stream_digest["n_batches"]
        out["stream_n_rows"] = stream_digest["n_rows"]
        # Per-stratum recall by *pre-treatment* difficulty: retrieval
        # position (relative tertile) x structural ambiguity.  A length
        # sweep changes the ambiguity mix by construction (more pairs ->
        # more repeated values, naive-confusable queries and answer-as-key
        # copy-source targets), so the raw accuracy drop conflates length
        # with data difficulty.  Recording the joint counts lets the
        # experiment layer post-stratify to a common reference mix
        # (``cage.metrics.standardized_proportion``) without a second model
        # pass.
        n_strata = 8 * n_key_bins
        n_realizable_strata = 8 * n_realizable_key_bins
        correct_rows = pred == tgt_cat
        strata_counts = torch.bincount(strata_cat, minlength=n_strata)
        strata_correct = torch.bincount(
            strata_cat[correct_rows], minlength=n_strata)
        strata_rows = []
        for sid, label in enumerate(difficulty_stratum_labels(n_key_bins)):
            n_s = int(strata_counts[sid].item())
            k_s = int(strata_correct[sid].item())
            strata_rows.append({
                "id": int(sid),
                "key_bin": int(label["key_bin"]),
                "ambiguity": label["ambiguity"],
                "n": n_s, "correct": k_s,
                # ``None`` (not NaN) for an unobserved stratum: records are
                # compared for exact reproducibility, and NaN != NaN would
                # make an otherwise identical evaluation look different.
                "accuracy": (k_s / n_s) if n_s else None,
            })
        observed = sum(1 for row in strata_rows if row["n"] > 0)
        out["difficulty"] = {
            "n_pairs": int(n_pairs),
            "n_key_bins": int(n_key_bins),
            "n_realizable_key_bins": int(n_realizable_key_bins),
            "n_strata": int(n_strata),
            "n_realizable_strata": int(n_realizable_strata),
            "observed_strata": int(observed),
            "coverage_of_possible": (
                observed / n_realizable_strata
                if n_realizable_strata else float("nan")),
            "strata": strata_rows,
            "note": ("pre-treatment retrieval-position x ambiguity strata; "
                     "post-stratify across lengths with the base-length "
                     "counts instead of comparing raw compositions"),
        }
        cf_summary = _counterfactual_summary(cf_totals or {})
        out["counterfactual"] = cf_summary
        out.update(cf_summary["flat"])
        if reference_codes is not None:
            exposure = _association_exposure_summary(exposure_totals)
            out["association_exposure"] = exposure
            out.update(exposure["flat"])
        if reference_counts is not None:
            out["association_exposure_dose"] = (
                _association_exposure_dose_summary(exposure_dose_totals))
        return out
    finally:
        for module, training in training_states.items():
            module.train(training)
        release_eval_scratch()


def _stats_layer_matrix(stats, key, n_rows=None):
    """A finite real ``(samples, layers)`` statistic matrix, or ``None``.

    Diagnostic reports are often built from a stats payload that may have
    been restored from JSON or assembled by hand.  A missing key, a boolean
    or integer matrix (which has no real mean), a 1-D tensor without a layer
    axis, a sample-axis mismatch, or a non-finite value must make the
    companion abstain instead of escaping as an ``AttributeError`` /
    ``IndexError`` / scalar fabrication from the reporting layer.
    """
    if not isinstance(stats, Mapping):
        return None
    value = stats.get(key)
    if (not torch.is_tensor(value) or value.dim() != 2
            or value.size(1) < 1 or not value.is_floating_point()
            or (n_rows is not None and value.size(0) != n_rows)
            or not bool(torch.isfinite(value).all())):
        return None
    return value


def _stats_correct_vector(stats, n_rows=None):
    """The per-sample 0/1 correctness flags, or ``None`` if malformed.

    Correctness is a binary flag, not an arbitrary numeric score: a complex
    or fractional/non-finite value would be silently reinterpreted by
    ``.bool()`` and fabricate which rows were answered correctly.
    """
    if not isinstance(stats, Mapping):
        return None
    value = stats.get("correct")
    if (not torch.is_tensor(value) or value.dim() != 1
            or value.is_complex()
            or (n_rows is not None and value.size(0) != n_rows)):
        return None
    if value.dtype != torch.bool:
        if value.is_floating_point() \
                and not bool(torch.isfinite(value).all()):
            return None
        if not bool(value.eq(0).logical_or(value.eq(1)).all()):
            return None
    return value.bool()


def gate_depth_ratio(stats, n_early=2):
    """Deep/early gate ratio: mean gate of the deepest layer over the mean of
    the shallowest ``n_early`` layers (E3).

    ``> 1`` means the collaborative normalisation made the deep layers more
    willing to write into the residual stream, i.e. it worked against the
    "deep layers contribute too little" symptom of Pre-LN.
    """
    g = _stats_layer_matrix(stats, "gate")
    if g is None:
        return float("nan")
    if isinstance(n_early, bool):
        return float("nan")
    try:
        n_early = operator.index(n_early)
    except TypeError:
        return float("nan")
    if n_early < 1 or g.size(1) <= n_early:
        return float("nan")
    early = g[:, :n_early].mean()
    deep = g[:, -1].mean()
    return (deep / early).item()


def _centered_unit(values):
    """Centered values scaled so their largest magnitude is one."""
    scale = float(values.abs().max())
    if not (math.isfinite(scale) and scale > 0.0):
        return None
    scaled = values / scale
    centered = scaled - scaled.mean()
    spread = float(centered.abs().max())
    if not (math.isfinite(spread) and spread > 0.0):
        return None
    return centered / spread


def pearson(x, y):
    # Keep the caller's precision: downcasting float64 to float32 can turn
    # distinct values into ties (and a large common offset then looks like a
    # constant vector), silently destroying the correlation.
    x = x.flatten()
    y = y.flatten()
    # ``.to(float64)`` silently drops the imaginary part of a complex
    # tensor; that projection is not the correlation the caller asked for.
    if x.is_complex() or y.is_complex():
        return torch.tensor(float("nan"))
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    if x.numel() == 0 or y.numel() != x.numel():
        return torch.tensor(float("nan"))
    if not (bool(torch.isfinite(x).all())
            and bool(torch.isfinite(y).all())):
        # A non-finite observation has no rank or moment; returning a
        # finite correlation after ranking NaN/Inf would fabricate an
        # association from a malformed sample.
        return torch.tensor(float("nan"))
    dx = _centered_unit(x)
    dy = _centered_unit(y)
    # A constant input is exactly the case with zero centered spread.  The
    # old guard compared that spread with the *raw* magnitude of x, so data
    # with a large common offset (e.g. 1e6, 1e6+1, ...) was misclassified as
    # constant.  Centered spread is the right scale; exact zero is the only
    # genuinely undefined case.
    if dx is None or dy is None:
        return torch.tensor(float("nan"))
    return (dx * dy).sum() / (dx.norm() * dy.norm())


def partial_pearson(x, y, control):
    """Pearson correlation of ``x`` and ``y`` after removing ``control``.

    The diagnostic statistics are observational: a gate that opens on
    diffuse attention can look correlated with entropy even when both
    quantities are just tracking attention concentration (max-prob).  This
    helper answers the controlled question - is the within-layer gate /
    entropy association still present after linearly removing max-prob? -
    by correlating the residuals of ``x`` and ``y`` after regressing each
    on ``control`` (with an intercept).

    At least four observations are required: with fewer, the residual
    correlation has too little residual variation to be informative.  A
    constant control, a control that is collinear with ``x`` or ``y``, a
    non-finite value, or mismatched lengths abstains with NaN rather than
    fabricating a coefficient.
    """
    x = x.flatten()
    y = y.flatten()
    control = control.flatten()
    if x.is_complex() or y.is_complex() or control.is_complex():
        return torch.tensor(float("nan"))
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    control = control.to(torch.float64)
    if x.numel() < 4 or y.numel() != x.numel() \
            or control.numel() != x.numel():
        return torch.tensor(float("nan"))
    if not (bool(torch.isfinite(x).all())
            and bool(torch.isfinite(y).all())
            and bool(torch.isfinite(control).all())):
        return torch.tensor(float("nan"))
    dx = _centered_unit(x)
    dy = _centered_unit(y)
    dc = _centered_unit(control)
    if dx is None or dy is None or dc is None:
        return torch.tensor(float("nan"))
    denom = (dc * dc).sum()
    if not math.isfinite(float(denom)) or float(denom) <= 0.0:
        return torch.tensor(float("nan"))
    rx = dx - (dx * dc).sum() / denom * dc
    ry = dy - (dy * dc).sum() / denom * dc
    if (not bool(torch.isfinite(rx).all())
            or not bool(torch.isfinite(ry).all())
            or rx.abs().max() == 0.0 or ry.abs().max() == 0.0):
        return torch.tensor(float("nan"))
    norm_x = rx.norm()
    norm_y = ry.norm()
    if (not math.isfinite(float(norm_x))
            or not math.isfinite(float(norm_y))
            or float(norm_x) == 0.0 or float(norm_y) == 0.0):
        return torch.tensor(float("nan"))
    return (rx * ry).sum() / (norm_x * norm_y)


def spearman(x, y):
    """Rank correlation (average ranks for ties).

    Robustness companion to ``pearson`` for H1a: the mechanism claim is a
    *monotone* association ("the gate opens when attention is diffuse"),
    which Pearson only captures as long as the relation is linear and no
    outlier dominates it.  Agreement between the two coefficients means
    the effect is not an artifact of the correlation measure.
    """
    # As in ``pearson``, ranking a float32 downcast would invent ties.
    x = x.flatten()
    y = y.flatten()
    # A complex rank correlation has no real interpretation; abstain rather
    # than silently correlating the real projections.
    if x.is_complex() or y.is_complex():
        return torch.tensor(float("nan"))
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    if x.numel() == 0 or y.numel() != x.numel():
        return torch.tensor(float("nan"))
    if not (bool(torch.isfinite(x).all())
            and bool(torch.isfinite(y).all())):
        # ``average_ranks`` has no defined ordering for NaN/Inf, yet it
        # would happily return ranks and a finite coefficient.
        return torch.tensor(float("nan"))
    rx = average_ranks(x)
    ry = average_ranks(y)
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    if dx.abs().max() <= 1e-12 or dy.abs().max() <= 1e-12:
        return torch.tensor(float("nan"))
    return (dx * dy).sum() / (dx.norm() * dy.norm())


def _fisher_mean(rs):
    """Mean of Pearson coefficients, weighted correctly through the
    Fisher z-transform; degenerate (NaN) layers are dropped.
    """
    rs = [r for r in rs if r == r]
    if not rs:
        return float("nan")
    return math.tanh(sum(math.atanh(min(max(r, -0.999999), 0.999999))
                         for r in rs) / len(rs))


def gate_stat_correlations(stats):
    """Correlations between gate value and attention entropy / max-prob /
    correctness, computed *within* each layer and averaged over layers.

    The statistics carry a sample axis and a layer axis.  Flattening both
    into one Pearson r lets the between-layer offsets (deep and shallow
    layers live at different operating points) dominate the between-sample
    variation, so the number answers "are deep gates smaller than shallow
    ones" instead of "does a given layer's gate track that same layer's
    attention statistics".  Per-layer r, aggregated with a Fisher
    z-transform, is the quantity the gate hypothesis is about.
    """
    r = {}
    g = _stats_layer_matrix(stats, "gate")
    if g is None:
        return r
    ent = _stats_layer_matrix(stats, "entropy", g.size(0))
    mp = _stats_layer_matrix(stats, "maxprob", g.size(0))
    corr = _stats_correct_vector(stats, g.size(0))
    if (ent is None or mp is None or corr is None
            or ent.shape != g.shape or mp.shape != g.shape):
        return r
    cols = range(g.size(1))
    r_list = [pearson(g[:, i], ent[:, i]).item() for i in cols]
    s_list = [spearman(g[:, i], ent[:, i]).item() for i in cols]
    r_list_m = [pearson(g[:, i], mp[:, i]).item() for i in cols]
    s_list_m = [spearman(g[:, i], mp[:, i]).item() for i in cols]
    r_list_c = [pearson(g[:, i], corr).item() for i in cols]
    partial_ent = [partial_pearson(g[:, i], ent[:, i], mp[:, i]).item()
                   for i in cols]
    partial_mp = [partial_pearson(g[:, i], mp[:, i], ent[:, i]).item()
                  for i in cols]
    correct_mask = corr > 0.5
    n_correct = int(correct_mask.sum())
    n_error = int((~correct_mask).sum())

    def _stratum_r(mask):
        if int(mask.sum()) < 3:
            return [float("nan")] * len(r_list)
        return [pearson(g[mask, i], ent[mask, i]).item() for i in cols]

    within_correct = _stratum_r(correct_mask)
    within_error = _stratum_r(~correct_mask)
    r["gate_vs_entropy"] = _fisher_mean(r_list)
    r["gate_vs_maxprob"] = _fisher_mean(r_list_m)
    r["gate_vs_correct"] = _fisher_mean(r_list_c)
    r["gate_vs_entropy_spearman"] = _fisher_mean(s_list)
    r["gate_vs_maxprob_spearman"] = _fisher_mean(s_list_m)
    # Quantify cross-layer variability (complement Fisher-mean summary)
    def _zmean_std(rs):
        vals = [x for x in rs if x == x]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        zs = [math.atanh(min(max(x, -0.999999), 0.999999)) for x in vals]
        m = sum(zs) / len(zs)
        if len(zs) < 2:
            return {"mean": math.tanh(m), "std": float("nan"), "n": len(vals)}
        var = sum((z - m) ** 2 for z in zs) / (len(zs) - 1)
        return {"mean": math.tanh(m), "std": math.sqrt(var), "n": len(vals)}
    r["gate_vs_entropy_by_layer"] = _zmean_std(r_list)
    r["gate_vs_maxprob_by_layer"] = _zmean_std(r_list_m)
    r["gate_vs_entropy_spearman_by_layer"] = _zmean_std(s_list)
    r["gate_vs_maxprob_spearman_by_layer"] = _zmean_std(s_list_m)
    r["gate_vs_entropy_partial_maxprob"] = _fisher_mean(partial_ent)
    r["gate_vs_maxprob_partial_entropy"] = _fisher_mean(partial_mp)
    r["gate_vs_entropy_within_correct"] = _fisher_mean(within_correct)
    r["gate_vs_entropy_within_error"] = _fisher_mean(within_error)
    r["gate_vs_entropy_partial_maxprob_by_layer"] = _zmean_std(partial_ent)
    r["gate_vs_maxprob_partial_entropy_by_layer"] = _zmean_std(partial_mp)
    r["gate_vs_entropy_within_correct_by_layer"] = _zmean_std(
        within_correct)
    r["gate_vs_entropy_within_error_by_layer"] = _zmean_std(within_error)
    # Raw per-layer values alongside the Fisher-z summary.  The mean alone
    # cannot reveal a correlation carried by one layer or a sign flip across
    # depth, and the E1 claim is explicitly about *within-layer* association
    # (between-layer operating-point offsets must not be pooled).  NaN
    # (degenerate layer) is stored as null so the JSON stays valid.
    def _per_layer(rs):
        return [float(x) if x == x else None for x in rs]
    r["gate_vs_entropy_per_layer"] = _per_layer(r_list)
    r["gate_vs_maxprob_per_layer"] = _per_layer(r_list_m)
    r["gate_vs_entropy_spearman_per_layer"] = _per_layer(s_list)
    r["gate_vs_maxprob_spearman_per_layer"] = _per_layer(s_list_m)
    r["gate_vs_entropy_partial_maxprob_per_layer"] = _per_layer(partial_ent)
    r["gate_vs_maxprob_partial_entropy_per_layer"] = _per_layer(partial_mp)
    r["gate_vs_entropy_within_correct_per_layer"] = _per_layer(
        within_correct)
    r["gate_vs_entropy_within_error_per_layer"] = _per_layer(within_error)
    r["gate_vs_entropy_partial_maxprob_layers"] = sum(
        1 for x in partial_ent if math.isfinite(x))
    r["gate_vs_maxprob_partial_entropy_layers"] = sum(
        1 for x in partial_mp if math.isfinite(x))
    r["gate_vs_entropy_within_correct_layers"] = sum(
        1 for x in within_correct if math.isfinite(x))
    r["gate_vs_entropy_within_error_layers"] = sum(
        1 for x in within_error if math.isfinite(x))
    r["n_correct"] = n_correct
    r["n_error"] = n_error
    return r


def _false_concentration_vectors(stats):
    """Per-sample max-prob and correctness vectors, or ``None`` if malformed.

    The false-concentration readouts need the *same* sample axis on both
    vectors.  A boolean/integer ``maxprob`` has no real maximum score (and
    negating it raises), a 1-D tensor has no layer axis, and a row-count
    mismatch would silently pair confidence scores with another stream's
    correctness flags.  All of those must abstain rather than escape as an
    ``IndexError``/``NotImplementedError`` from the reporting companion.
    """
    mp = _stats_layer_matrix(stats, "maxprob")
    if mp is None:
        return None
    corr = _stats_correct_vector(stats, mp.size(0))
    if corr is None:
        return None
    return mp.max(dim=1)[0], corr


def _per_sample_layer_mean(stats, key, n_rows):
    """Mean over layers of a per-sample statistic matrix, or ``None``.

    ``collect_stats_and_errors`` stores gate/null-mass as real floating
    ``(samples, layers)`` tensors.  Treat anything else as absent: boolean
    and integer tensors have no real mean (``Tensor.mean`` raises), a 1-D
    tensor has no layer axis, and a matrix with a different sample axis
    would misalign rows with the correctness vector.
    """
    value = _stats_layer_matrix(stats, key, n_rows)
    if value is None:
        return None
    return value.mean(dim=1)


def _validated_maxprob_thresh(value):
    """Coerce a confidence threshold to a finite probability in ``[0, 1]``.

    The false-concentration companions compare ``maxprob`` against this
    value, so a malformed threshold would otherwise escape as an incidental
    ``TypeError``/``NotImplementedError`` from the comparison.  Booleans are
    flags rather than probabilities, complex values have no real threshold,
    and a non-finite or out-of-range number cannot cut a probability axis;
    all of those abstain via ``None``.
    """
    if isinstance(value, bool):
        return None
    if torch.is_tensor(value):
        if value.is_complex() or value.dtype == torch.bool \
                or value.numel() != 1:
            return None
        raw = value.detach().reshape(-1)[0]
    else:
        raw = value
    try:
        threshold = float(raw)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return None
    if not (math.isfinite(threshold) and 0.0 <= threshold <= 1.0):
        return None
    return threshold


def _false_concentration_at(mp, corr, gm, maxprob_thresh):
    """Threshold-specific readout over precomputed per-sample vectors.

    ``false_conc_rate`` is a binomial error share over the confident rows,
    so it carries a Wilson interval: at the stats-stream sizes used here a
    rate of 0.30 [0.22, 0.39] is not distinguishable from 0.36, and a
    variant difference read without that interval would overstate the
    measurement.  The interval is on the *within-stream* sample; training-
    seed variance is a separate uncertainty carried by the paired
    contrast layer.
    """
    high = mp > maxprob_thresh
    r = {
        "n_high_maxprob": int(high.sum()),
        "false_conc_rate": 1.0 - corr[high].float().mean().item() if high.any() else float("nan"),
        "overall_acc": corr.float().mean().item(),
    }
    if high.any():
        n_false = int((high & ~corr).sum())
        ci = wilson_ci(n_false, int(high.sum()))
        r["n_false_conc"] = n_false
        r["false_conc_rate_lo"] = ci["lo"]
        r["false_conc_rate_hi"] = ci["hi"]
    if gm is not None:
        fc = high & ~corr
        cc = high & corr
        r["n_false_conc_gate"] = int(fc.sum())
        r["n_correct_high_gate"] = int(cc.sum())
        r["gate_on_false_conc"] = gm[fc].mean().item() if fc.any() else float("nan")
        r["gate_on_correct"] = gm[cc].mean().item() if cc.any() else float("nan")
    return r


def false_concentration_analysis(stats, maxprob_thresh=0.5):
    """'False concentration': max attention prob high but answer wrong.

    Computes, among high-maxprob samples, the error rate (false-concentration
    rate) and, for gated models, mean gate value on false-concentrated vs
    correct samples - a good calibrated gate should close (lower value)
    when concentration is false.
    """
    vectors = _false_concentration_vectors(stats)
    if vectors is None:
        return {}
    threshold = _validated_maxprob_thresh(maxprob_thresh)
    if threshold is None:
        return {}
    mp, corr = vectors
    gm = _per_sample_layer_mean(stats, "gate", mp.size(0))
    return _false_concentration_at(mp, corr, gm, threshold)


def false_concentration_sweep(stats, thresholds=(0.3, 0.5, 0.7, 0.9)) -> dict:
    """H1b robustness: the false-concentration split at several thresholds.

    ``maxprob_thresh=0.5`` is an a-priori choice, but the whole H1b verdict
    (gate margin = gate on correct - gate on false concentration) depends
    on where "high confidence" is cut.  Sweeping the threshold makes the
    analyst's degree of freedom explicit: a mechanism claim that only
    holds at one arbitrary cut is a property of the cut, not of the
    mechanism.  The pre-registered contrast stays at 0.5; the sweep is
    reported so cherry-picking can be ruled out.

    Returns ``{threshold: false_concentration_analysis(stats, thr)}``.
    """
    vectors = _false_concentration_vectors(stats)
    if vectors is None:
        return {}
    if isinstance(thresholds, (str, bytes, Mapping)):
        return {}
    try:
        candidates = list(thresholds)
    except TypeError:
        return {}
    mp, corr = vectors
    gm = _per_sample_layer_mean(stats, "gate", mp.size(0))
    out = {}
    for candidate in candidates:
        threshold = _validated_maxprob_thresh(candidate)
        if threshold is None:
            continue
        out[threshold] = _false_concentration_at(mp, corr, gm, threshold)
    return out


def false_concentration_by_layer(stats, maxprob_thresh=0.5):
    """Per-layer false-concentration readout (H1b / H1c mechanism check).

    The aggregate :func:`false_concentration_analysis` deliberately collapses
    layers: it uses the maximum max-prob over layers together with the mean
    gate over layers.  That summary is readable but it is not the quantity
    the gate hypothesis is about - a gate is computed *within* a layer from
    that layer's own attention distribution, so a layer with genuinely
    concentrated-but-wrong attention can be diluted by another layer whose
    max-prob is low, and the aggregate gate mean cannot close only where the
    concentration is false.

    This companion keeps the axes aligned: for every layer it forms the
    high-max-prob set from **that layer's** max-prob and compares
    **that layer's** gate (or null-slot mass) on the false-concentrated vs
    correct rows.  It reports the same names as the aggregate readout
    (``false_conc_rate`` / ``gate_on_false_conc`` / ``gate_on_correct`` /
    ``gate_margin``) plus ``null_mass_error_margin`` for the null arm, so the
    layer profile can be read next to - never instead of - the
    pre-registered aggregate H1b/H1c contrasts.

    Returns a list of per-layer dicts; empty when the statistics carry no
    layer axis.  ``gate_margin`` / ``null_mass_error_margin`` are NaN when a
    layer has no false-concentrated or no confident-correct row (a within-
    stream sample-size limit, not a model effect).
    """
    mp = _stats_layer_matrix(stats, "maxprob")
    if mp is None:
        return []
    corr = _stats_correct_vector(stats, mp.size(0))
    if corr is None:
        return []

    def _optional_matrix(key):
        return _stats_layer_matrix(stats, key, mp.size(0))

    gate = _optional_matrix("gate")
    null = _optional_matrix("null_mass")
    threshold = _validated_maxprob_thresh(maxprob_thresh)
    if threshold is None:
        return []
    n_layers = int(mp.size(1))
    rows = []
    for layer in range(n_layers):
        high = mp[:, layer] > threshold
        n_high = int(high.sum())
        n_false = int((high & ~corr).sum())
        row = {
            "layer": int(layer),
            "maxprob_thresh": threshold,
            "n_high_maxprob": n_high,
            "n_false_conc": n_false,
            "false_conc_rate": (n_false / n_high if n_high
                                else float("nan")),
        }
        if n_high:
            ci = wilson_ci(n_false, n_high)
            row["false_conc_rate_lo"] = ci["lo"]
            row["false_conc_rate_hi"] = ci["hi"]
        if gate is not None and layer < gate.size(1):
            gm = gate[:, layer]
            fc, cc = high & ~corr, high & corr
            row["n_false_conc_gate"] = int(fc.sum())
            row["n_correct_high_gate"] = int(cc.sum())
            row["gate_on_false_conc"] = (float(gm[fc].mean()) if bool(fc.any())
                                         else float("nan"))
            row["gate_on_correct"] = (float(gm[cc].mean()) if bool(cc.any())
                                      else float("nan"))
            row["gate_margin"] = (row["gate_on_correct"]
                                  - row["gate_on_false_conc"]
                                  if math.isfinite(row["gate_on_correct"])
                                  and math.isfinite(row["gate_on_false_conc"])
                                  else float("nan"))
        if null is not None and layer < null.size(1):
            nm = null[:, layer]
            fc, cc = high & ~corr, high & corr
            row["null_mass_error"] = (float(nm[fc].mean())
                                      if bool(fc.any()) else float("nan"))
            row["null_mass_correct"] = (float(nm[cc].mean())
                                        if bool(cc.any()) else float("nan"))
            row["null_mass_error_margin"] = (
                row["null_mass_error"] - row["null_mass_correct"]
                if math.isfinite(row["null_mass_error"])
                and math.isfinite(row["null_mass_correct"])
                else float("nan"))
        rows.append(row)
    return rows


def _two_sample_bootstrap_ci(a, b, n_boot: int = 10000, seed: int = 0,
                              alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for ``mean(a) - mean(b)``.

    The two groups are independent samples of different sizes (correct and
    false-concentration rows are disjoint), so their resampling indices
    must be drawn separately; pairing or broadcasting them is invalid.
    """

    def _flat(x):
        # A complex sample has no real mean difference; drop it rather than
        # letting the float64 cast silently discard its imaginary part.
        # Non-numeric/malformed containers must likewise abstain instead of
        # escaping as a TypeError from ``as_tensor``: this is a reporting
        # companion, and one bad group must not abort the whole report.
        if torch.is_tensor(x) and (x.is_complex()
                                   or x.dtype == torch.bool):
            return torch.empty(0, dtype=torch.float64)
        try:
            probe = torch.as_tensor(x)
            if probe.is_complex() or probe.dtype == torch.bool:
                return torch.empty(0, dtype=torch.float64)
            t = torch.as_tensor(x, dtype=torch.float64).detach().reshape(-1)
        except (TypeError, ValueError, RuntimeError, OverflowError):
            return torch.empty(0, dtype=torch.float64)
        if not bool(torch.isfinite(t).all()):
            return torch.empty(0, dtype=torch.float64)
        return t

    alpha = _float_or_nan(alpha)
    a = _flat(a)
    b = _flat(b).to(a.device)
    n_a, n_b = a.numel(), b.numel()
    observed = float(a.mean() - b.mean()) if n_a and n_b else float("nan")
    # One observation in a group carries no within-group variance: the
    # percentile bootstrap would then collapse to a point and report a tiny
    # p-value from a single row per condition.  Abstain (the wrappers turn
    # this into ``applicable=False``) until both groups have >= 2 rows.
    if n_a < 2 or n_b < 2 or not _bootstrap_args_ok(n_boot, alpha):
        return {"mean": observed, "lo": float("nan"), "hi": float("nan"),
                "n": int(n_a + n_b), "n_a": int(n_a), "n_b": int(n_b),
                "sd": float("nan"), "sd_a": float("nan"),
                "sd_b": float("nan"), "se": float("nan"),
                "half_width": float("nan"), "mde80": float("nan"),
                "p_value": float("nan"), "alpha": alpha}
    n_boot = _positive_integer(n_boot)
    gen = torch.Generator(device=a.device).manual_seed(
        _bootstrap_seed(seed))
    idx_a = torch.randint(0, n_a, (n_boot, n_a), generator=gen,
                          device=a.device)
    idx_b = torch.randint(0, n_b, (n_boot, n_b), generator=gen,
                          device=a.device)
    boot = a[idx_a].mean(dim=1) - b[idx_b].mean(dim=1)
    lo = float(torch.quantile(boot, alpha / 2.0))
    hi = float(torch.quantile(boot, 1.0 - alpha / 2.0))
    p_lo = float((boot <= 0.0).to(torch.float64).mean())
    p_hi = float((boot >= 0.0).to(torch.float64).mean())
    p_value = min(
        1.0, 2.0 * (min(p_lo, p_hi) * int(n_boot) + 1.0) / (int(n_boot) + 1.0))
    var_a = float(a.var(unbiased=True)) if n_a > 1 else float("nan")
    var_b = float(b.var(unbiased=True)) if n_b > 1 else float("nan")
    sd_a = math.sqrt(var_a) if math.isfinite(var_a) else float("nan")
    sd_b = math.sqrt(var_b) if math.isfinite(var_b) else float("nan")
    # A group with one observation contributes no within-group variance,
    # but the pooled variance is still estimable from the other group
    # (its (n-1) term is zero).  Treating that one group as NaN threw away
    # the estimable pooled SD/SE; only the degenerate n_a=n_b=1 case has no
    # variance information at all.
    var_a_pool = var_a if n_a > 1 else 0.0
    var_b_pool = var_b if n_b > 1 else 0.0
    df = n_a + n_b - 2
    pooled_var = (((n_a - 1) * var_a_pool
                   + (n_b - 1) * var_b_pool) / df
                  if df > 0 else float("nan"))
    sd = math.sqrt(pooled_var) if math.isfinite(pooled_var) else float("nan")
    if n_a > 1 and n_b > 1:
        se = math.sqrt(var_a / n_a + var_b / n_b)
    elif math.isfinite(pooled_var):
        se = math.sqrt(pooled_var * (1.0 / n_a + 1.0 / n_b))
    else:
        se = float("nan")
    return {
        "mean": observed,
        "lo": lo,
        "hi": hi,
        "n": int(n_a + n_b),
        "n_a": int(n_a),
        "n_b": int(n_b),
        "sd": sd,
        "sd_a": sd_a,
        "sd_b": sd_b,
        "se": se,
        "half_width": (hi - lo) / 2.0,
        "mde80": 2.8015852 * se if math.isfinite(se) else float("nan"),
        "p_value": p_value,
        "alpha": float(alpha),
    }


@torch.inference_mode()
def false_concentration_gate_test(stats, maxprob_thresh=0.5,
                                   n_boot: int = 10000, seed: int = 12345):
    """Bootstrap test of gate_on_correct - gate_on_false_conc for gated models.

    A calibrated gate should close (lower) on false concentration, so the
    difference ``gate_on_correct - gate_on_false_conc`` should be > 0
    and exclude 0 when the mechanism works.

    Correct and false-concentration rows are disjoint, so the two group
    means use an independent two-sample bootstrap (their sizes need not,
    and generally do not, match).
    """
    vectors = _false_concentration_vectors(stats)
    if vectors is None:
        return {"applicable": False,
                "reason": "malformed maxprob/correct stats"}
    mp, corr = vectors
    threshold = _validated_maxprob_thresh(maxprob_thresh)
    if threshold is None:
        return {"applicable": False,
                "reason": "malformed maxprob_thresh"}
    if stats.get("gate") is None:
        return {"applicable": False, "reason": "no gate in stats"}
    gm = _per_sample_layer_mean(stats, "gate", mp.size(0))
    if gm is None:
        return {"applicable": False, "reason": "malformed gate stats"}
    high = mp > threshold
    fc = high & ~corr
    cc = high & corr
    if not bool(fc.any()) or not bool(cc.any()):
        return {"applicable": False, "n_fc": int(fc.sum()), "n_cc": int(cc.sum())}
    ci = _two_sample_bootstrap_ci(gm[cc], gm[fc], n_boot=n_boot, seed=seed)
    ci["n_fc"] = int(fc.sum())
    ci["n_cc"] = int(cc.sum())
    ci["applicable"] = bool(
        ci["n_a"] >= 2 and ci["n_b"] >= 2
        and math.isfinite(ci.get("lo", float("nan")))
        and math.isfinite(ci.get("hi", float("nan"))))
    return ci


@torch.inference_mode()
def null_mass_gate_test(stats, maxprob_thresh=None, n_boot: int = 10000,
                        seed: int = 12345):
    """Within-stream bootstrap for the null slot's error-minus-correct mass.

    H1c is the null arm's mechanism claim: the dedicated null slot should
    absorb the write more on errors than on correct answers.  The seed-level
    margin is tested across training seeds by the H1c contrast, but the
    *finite diagnostic stream* resolves the two conditional levels with its
    own uncertainty - the same second source of noise that
    :func:`false_concentration_gate_test` reports for the gated arms.  This
    companion applies the identical independent two-sample bootstrap to the
    null mass (error rows minus correct rows), optionally restricting to
    high-max-prob rows so the false-concentration reading is available too.
    The error and correct groups are disjoint, so their resampling indices
    are drawn independently.

    Returns ``{"applicable": False}`` for non-null stats or a degenerate
    group, otherwise the bootstrap dict plus the contrast label and group
    sizes.
    """
    if not isinstance(stats, Mapping):
        return {"applicable": False, "reason": "malformed stats"}
    nm = stats.get("null_mass")
    if nm is None:
        return {"applicable": False, "reason": "no null_mass in stats"}
    corr = _stats_correct_vector(stats)
    if corr is None:
        return {"applicable": False, "reason": "malformed correct stats"}
    mass = _per_sample_layer_mean(stats, "null_mass", corr.size(0))
    if mass is None:
        return {"applicable": False,
                "reason": "malformed null_mass stats"}
    if maxprob_thresh is None:
        err_group, correct_group = mass[~corr], mass[corr]
        group = "all"
    else:
        threshold = _validated_maxprob_thresh(maxprob_thresh)
        if threshold is None:
            return {"applicable": False,
                    "reason": "malformed maxprob_thresh"}
        mp = _stats_layer_matrix(stats, "maxprob", mass.size(0))
        if mp is None:
            return {"applicable": False,
                    "reason": "malformed maxprob stats"}
        high = mp.max(dim=1)[0] > threshold
        err_group = mass[high & ~corr]
        correct_group = mass[high & corr]
        group = f"maxprob>{threshold:g}"
    if not bool(err_group.numel()) or not bool(correct_group.numel()):
        return {"applicable": False, "group": group,
                "n_error": int(err_group.numel()),
                "n_correct": int(correct_group.numel()),
                "reason": "empty error or correct group"}
    ci = _two_sample_bootstrap_ci(err_group, correct_group, n_boot=n_boot,
                                  seed=seed)
    ci.update({
        "applicable": bool(
            ci["n_a"] >= 2 and ci["n_b"] >= 2
            and math.isfinite(ci.get("lo", float("nan")))
            and math.isfinite(ci.get("hi", float("nan")))),
        "contrast": "null_mass_error - null_mass_correct",
        "group": group,
        "n_error": int(err_group.numel()),
        "n_correct": int(correct_group.numel()),
    })
    return ci


def error_detection_aucs(stats):
    """Imbalance-aware error-detection summary of attention statistics.

    The positive class is "model answered wrong".  Scores are oriented so
    that **higher means more likely wrong** (``-maxprob``, mean entropy,
    ``-gate``), matching the mechanism hypotheses: concentration should
    track correctness, and a calibrated gate should close when it does
    not.

    Errors are the rare class after convergence, so the metric *regime*
    is measured, not assumed: ``imbalance_ratio`` (negatives per
    positive), ``error_rate`` = prevalence, and ``metric_regime``
    (``no_errors``, ``extreme`` at 1:1000 or worse, ``high`` at 1:10 or
    worse, else ``balanced``).  A stream with **no negatives** (every
    answer wrong, prevalence 1.0) is not "balanced": the ranking metrics
    are undefined there, so it gets its own ``all_errors`` regime instead
    of being confusable with a genuinely usable balance in the report.
    Under ``extreme`` imbalance ROC-AUC grows with the
    huge negative population while precision can still collapse, so the
    headline is **PR-AUC, read against its chance level** - for a random
    ranker average precision equals prevalence, so ``prauc_lift_*``
    (PR-AUC / prevalence) is what says whether a score carries signal,
    and ``primary_metric`` names the headline explicitly.

    Every score also gets ``recall_at_10pct_*`` / ``lift_at_10pct_*``:
    the share of errors captured in the 10% most doubtful answers
    (ranked by that score) and the precision lift of that flagged set
    over prevalence - the operational reading of the same ranking when
    PR-AUC alone is hard to interpret (see
    ``cage.metrics.selection_at_fraction``).
    """
    mp = _stats_layer_matrix(stats, "maxprob")
    if mp is None:
        return {}
    ent = _stats_layer_matrix(stats, "entropy", mp.size(0))
    corr = _stats_correct_vector(stats, mp.size(0))
    if ent is None or corr is None or ent.shape != mp.shape:
        return {}
    gate = _stats_layer_matrix(stats, "gate", mp.size(0))
    err = ~corr
    n = int(corr.numel())
    n_err = int(err.sum())
    n_neg = n - n_err
    prevalence = (n_err / n) if n else float("nan")
    ratio = (n_neg / n_err) if n_err else float("inf")
    if n == 0:
        regime = "no_samples (empty stream; all metrics undefined)"
    elif n_err == 0:
        regime = "no_errors"
    elif n_neg == 0:
        # Degenerate all-wrong stream: ROC/PR-AUC are NaN (no negatives to
        # rank against) and must not be presented as a usable "balanced"
        # error-detection readout.
        regime = "all_errors (no negatives; ranking metrics undefined)"
    elif ratio >= 1000.0:
        regime = "extreme (>=1:1000; PR-AUC is the headline, ROC-AUC is optimistic)"
    elif ratio >= 10.0:
        regime = "high (>=1:10; report both, trust PR-AUC for rare-positive ranking)"
    else:
        regime = "balanced"
    out = {"n": n, "n_errors": n_err, "error_rate": prevalence,
           "prevalence": prevalence, "imbalance_ratio": ratio,
           "metric_regime": regime, "flag_fraction": 0.1}
    scores = {
        "maxprob": -mp.max(dim=1)[0],
        "entropy": ent.mean(dim=1),
    }
    if gate is not None:
        scores["gate"] = -gate.mean(dim=1)
    praucs = {}
    for name in ("maxprob", "entropy", "gate"):
        score = scores.get(name)
        if score is None or n_err == 0 or n_err == n:
            praucs[name] = float("nan")
            out[f"rocauc_{name}_to_error"] = float("nan")
            out[f"prauc_{name}_to_error"] = float("nan")
            out[f"prauc_lift_{name}_to_error"] = float("nan")
            out[f"recall_at_10pct_{name}_to_error"] = float("nan")
            out[f"lift_at_10pct_{name}_to_error"] = float("nan")
            continue
        pa = pr_auc(score, err)
        praucs[name] = pa
        out[f"rocauc_{name}_to_error"] = roc_auc(score, err)
        out[f"prauc_{name}_to_error"] = pa
        out[f"prauc_lift_{name}_to_error"] = (
            pa / prevalence if prevalence > 0.0 else float("nan"))
        sel = selection_at_fraction(score, err, 0.1)
        out[f"recall_at_10pct_{name}_to_error"] = sel["recall"]
        out[f"lift_at_10pct_{name}_to_error"] = sel["lift"]
    # The headline must name a readable ranking: keep the pre-registered
    # preference order (gate > maxprob > entropy) but skip any score whose
    # PR-AUC is non-finite on this stream, so ``primary_metric`` never
    # points at NaN while a finite companion exists.
    primary = next((name for name in ("gate", "maxprob", "entropy")
                    if name in scores
                    and math.isfinite(praucs[name])), None)
    out["primary_metric"] = (f"prauc_{primary}_to_error" if primary
                             else "none (no finite PR-AUC on this stream)")
    return out
