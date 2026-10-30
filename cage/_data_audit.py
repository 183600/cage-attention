# cage data: difficulty, distribution-shift, uniformity and leakage audits.
# Executed into the wrapper module namespace by the loader;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def actual_stream_overlap_audit(n_pairs, vocab, device="cpu",
                                train_specs=None, heldout_specs=None,
                                include_train_association_codes=False):
    """Exact content overlap of the streams a run *actually* consumes.

    ``split_leakage_audit`` checks the first ``batch * batches`` samples of
    every seed at one fixed geometry.  Training, however, reads
    ``settings.steps * settings.batch`` samples, and ``make_batch`` draws
    through shape-dependent random calls, so the audited prefix is not the
    training realization.  This audit replays each specified stream with
    exactly the geometry its consumer uses:

    * ``train_specs`` maps a stream name to ``{"seed", "batch", "steps"}``;
    * ``heldout_specs`` maps a stream name to ``{"seed", "batch",
      "batches"}`` (e.g. final eval, validation monitoring, stats).

    A spec may carry ``"group"`` to name the split it belongs to (training
    specs default to ``"train"``, held-out specs to the spec name).  Two
    realizations of the *same* split (e.g. the validation stream at the
    monitoring geometry and at the final-eval geometry) may legitimately
    share samples; only overlaps between *different* groups - train vs
    held-out, validation vs test, test vs statistics - are leakage.  The
    full pairwise matrix is still reported, but ``max_overlap`` counts
    cross-group pairs only, so re-reading one split at two geometries
    cannot fail the gate.

    Content overlap alone is not sufficient for this task: a split is a
    seed-owned stream, and ``make_batch``'s draws are shape-dependent, so
    the *same* seed read at two different batch sizes can produce two
    disjoint realizations.  Reusing a training seed as a held-out stream
    at a different geometry would therefore show zero duplicate rows while
    still training and evaluating on the same underlying stream.  The
    audit separately compares the canonical 32-bit seed keys of every
    stream pair and reports ``cross_group_seed_collisions``; any such
    collision across groups is leakage even when the content matrices are
    empty.

    **Association exposure.** Exact-row duplicates are the wrong leakage
    unit for a synthetic sequence task: two draws almost never repeat the
    whole sequence, but the same query ``(key, value)`` association can
    recur across splits, and a model can memorise it.  The replay therefore
    also records every item's association code (see
    :func:`association_codes`).  For each non-training group it reports the
    item-weighted share whose exact pair was observed in a training stream
    (``pair_seen_train_rate``) and the share whose query key was observed
    with any value (``key_seen_train_rate``).  This is a *disclosure*, not
    a leakage verdict: a shared vocabulary makes some association overlap
    expected, while only cross-group *row* overlap and seed reuse are hard
    failures.  Pass ``include_train_association_codes=True`` to also return
    the training code set (``train_association_codes``, a Python set that
    is not JSON-serializable) for exposure-stratified evaluation, plus the
    exact per-code replay counts (``train_association_counts``) for the
    exposure-dose readout; callers must pop both before persisting the
    audit.

    It returns per-stream row/duplicate counts, the pairwise overlap matrix
    over row-content fingerprints, and the association-exposure readout.
    Any cross-group row overlap or seed-key collision is leakage; the
    caller decides how to gate on it.
    """
    def _spec_mapping(value, label):
        """Normalise a stream-spec container, rejecting non-mappings.

        ``bytes``/strings are iterable but are not spec mappings (a byte
        string would otherwise fail deep inside ``dict()`` with an
        incidental TypeError), and a bare integer is not iterable at all.
        The audit's input contract is a mapping, so translate every such
        malformed container into a ValueError like the stream-name and
        seed checks do.
        """
        if value is None:
            return {}
        if isinstance(value, (str, bytes)):
            raise ValueError(
                f"{label} must be a mapping of stream name to spec, got "
                f"{type(value).__name__}")
        try:
            return dict(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} must be a mapping of stream name to spec, got "
                f"{type(value).__name__}") from exc

    train_specs = _spec_mapping(train_specs, "train_specs")
    heldout_specs = _spec_mapping(heldout_specs, "heldout_specs")
    if not train_specs or not heldout_specs:
        raise ValueError(
            "actual_stream_overlap_audit needs at least one training and "
            "one held-out stream spec")
    for name in train_specs:
        _audit_stream_name(name, "training")
    for name in heldout_specs:
        _audit_stream_name(name, "held-out")
    shared_names = sorted(set(train_specs) & set(heldout_specs), key=str)
    if shared_names:
        raise ValueError(
            "stream names must be unique across train_specs and "
            f"heldout_specs; duplicated: {shared_names}")
    key_sets, streams, groups, seed_keys = {}, {}, {}, {}
    associations, association_sets, query_key_sets = {}, {}, {}
    all_specs = ([(name, spec, "train")
                  for name, spec in train_specs.items()]
                 + [(name, spec, "heldout")
                    for name, spec in heldout_specs.items()])
    for name, spec, kind in all_specs:
        unit_key = "steps" if kind == "train" else "batches"
        if not isinstance(spec, Mapping):
            raise ValueError(
                f"{kind} spec {name!r} must be a mapping with 'seed', "
                f"'batch' and {unit_key!r}, got {type(spec).__name__}")
        if spec.get("seed") is None:
            raise ValueError(f"{kind} spec {name!r} needs 'seed'")
        if spec.get("batch") is None:
            raise ValueError(f"{kind} spec {name!r} needs 'batch'")
        if spec.get(unit_key) is None:
            raise ValueError(
                f"{kind} spec {name!r} needs {unit_key!r}")
        raw_seed = spec["seed"]
        if isinstance(raw_seed, bool):
            raise ValueError(
                f"{kind} spec {name!r} seed must be an integer, "
                f"got {raw_seed!r}")
        try:
            seed = operator.index(raw_seed)
        except TypeError as exc:
            raise ValueError(
                f"{kind} spec {name!r} seed must be an integer, "
                f"got {raw_seed!r}") from exc
        batch = _positive_audit_count(
            spec["batch"], f"{kind} spec {name!r} batch")
        units = _positive_audit_count(
            spec[unit_key], f"{kind} spec {name!r} {unit_key}")
        seed_key = stream_seed_key(seed)
        replay, n_rows = _draw_stream_keys(
            seed, n_pairs, vocab, device, batch, units)
        keys = replay["row_keys"]
        associations[name] = replay["associations"]
        association_sets[name] = replay["association_set"]
        query_key_sets[name] = replay["key_set"]
        key_sets[name] = keys
        seed_keys[name] = seed_key
        groups[name] = str(spec.get(
            "group", "train" if kind == "train" else name))
        streams[name] = {
            "kind": kind, "group": groups[name],
            "seed": seed, "stream_key": seed_key,
            "batch": batch, "units": units,
            "n_rows": int(n_rows), "n_unique_rows": int(len(keys)),
            "n_duplicate_rows": int(n_rows - len(keys)),
            "n_unique_associations": int(len(association_sets[name])),
        }

    names = list(key_sets)
    pairwise = {}
    pairwise_seed_collision = {}
    cross_group = []
    seed_collisions = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = len(key_sets[a] & key_sets[b])
            pair = f"{a}|{b}"
            pairwise[pair] = overlap
            collision = seed_keys[a] == seed_keys[b]
            pairwise_seed_collision[pair] = collision
            if groups[a] != groups[b]:
                cross_group.append(overlap)
                if collision:
                    seed_collisions.append(pair)
    train_names = list(train_specs)
    heldout_names = list(heldout_specs)
    train_vs_heldout = [
        pairwise[f"{a}|{b}"] for a in train_names for b in heldout_names]
    heldout_vs_heldout = [
        pairwise[f"{a}|{b}"] for i, a in enumerate(heldout_names)
        for b in heldout_names[i + 1:]]

    # Association exposure: the training groups' exact query pairs form the
    # reference; held-out items are only *scored* against it (nothing is
    # fitted on held-out data).  Item-weighted rates are reported because
    # the question is "what share of the evaluated items could be answered
    # from a memorised association?", not "how many distinct pair types
    # recur".  A split that is itself in a training group (e.g. the
    # validation stream re-read at a second geometry) is not held-out and
    # is left out of this readout.
    train_codes, train_keys_seen = set(), set()
    for name in names:
        if groups[name] == "train":
            train_codes.update(association_sets[name])
            train_keys_seen.update(query_key_sets[name])
    vocab_int = int(vocab)
    exposure = {}
    for name in names:
        if groups[name] == "train":
            continue
        codes = associations[name]
        n = len(codes)
        pair_seen = sum(1 for code in codes if code in train_codes)
        key_seen = sum(
            1 for code in codes if code // vocab_int in train_keys_seen)
        unique = set(codes)
        pair_seen_unique = len(unique & train_codes)
        exposure[name] = {
            "group": groups[name],
            "n_items": int(n),
            "n_unique_pairs": int(len(unique)),
            "n_unique_keys": int(len({code // vocab_int for code in codes})),
            "pair_seen_train": int(pair_seen),
            "pair_seen_train_rate": (pair_seen / n if n else float("nan")),
            "key_seen_train": int(key_seen),
            "key_seen_train_rate": (key_seen / n if n else float("nan")),
            # Seen key, unseen exact pair: the item's key occurred in
            # training but not with this value, so a memorised key -> value
            # lookup cannot directly supply the answer.  This is the cohort
            # where in-context retrieval is required.
            "pair_unseen_key_seen": int(key_seen - pair_seen),
            "pair_unseen_key_seen_rate": (
                (key_seen - pair_seen) / n if n else float("nan")),
            "pair_seen_unique": int(pair_seen_unique),
            "pair_seen_unique_rate": (
                pair_seen_unique / len(unique) if unique else float("nan")),
        }
    train_summary = {
        "n_items": int(sum(len(associations[name]) for name in names
                          if groups[name] == "train")),
        "n_unique_pairs": int(len(train_codes)),
        "n_unique_keys": int(len(train_keys_seen)),
        "n_possible_pairs": int((vocab_int - 2) ** 2),
        "n_possible_keys": int(vocab_int - 2),
        "pair_space_coverage": (
            len(train_codes) / ((vocab_int - 2) ** 2)
            if vocab_int > 2 else float("nan")),
    }
    max_exposure = max(
        (row["pair_seen_train_rate"] for row in exposure.values()
         if math.isfinite(row["pair_seen_train_rate"])), default=float("nan"))
    out = {
        "train_specs": {k: dict(v) for k, v in train_specs.items()},
        "heldout_specs": {k: dict(v) for k, v in heldout_specs.items()},
        "streams": streams,
        "pairwise_overlap": pairwise,
        "pairwise_seed_collision": pairwise_seed_collision,
        "cross_group_seed_collisions": seed_collisions,
        "train_vs_heldout_max": max(train_vs_heldout, default=0),
        "heldout_vs_heldout_max": max(heldout_vs_heldout, default=0),
        "max_overlap": max(cross_group, default=0),
        "n_rows_total": int(sum(s["n_rows"] for s in streams.values())),
        "train_association_summary": train_summary,
        "association_exposure": exposure,
        "max_heldout_pair_exposure": float(max_exposure),
        "association_exposure_note": (
            "item-weighted share of held-out query (key, value) pairs whose "
            "exact association was observed in a training-group stream. "
            "This is a memorisation-opportunity disclosure, not a leakage "
            "verdict: exact-pair exposure is expected with a shared "
            "vocabulary, while cross-group whole-row overlap and seed reuse "
            "remain hard failures.  Read clean and counterfactual accuracy "
            "separately on the exposure cohorts before attributing an "
            "accuracy gain to in-context retrieval."),
    }
    if include_train_association_codes:
        # Not JSON-serializable on purpose: the caller must pop these before
        # persisting the audit (the experiment layer does so when it hands
        # the references to exposure-stratified evaluation).
        out["train_association_codes"] = train_codes
        train_counts = {}
        for name in names:
            if groups[name] != "train":
                continue
            for code in associations[name]:
                train_counts[code] = train_counts.get(code, 0) + 1
        out["train_association_counts"] = train_counts
    return out


# Audit fields: pooled tokens, the supervised answer distribution (the
# quantity the accuracy / loss metrics are computed on), the key / value
# sub-populations, and the *structural difficulty* covariates that decide
# whether accuracy is comparable across splits: where the matching key sits
# (retrieval distance), whether the answer value repeats in the context, and
# whether a naive last-match rule would be misled by the query token
# appearing later as a value.  Marginal token frequencies can match perfectly
# while one split contains systematically later keys or more ambiguous
# samples; such a shift changes the ceiling/accuracy of a split and would
# otherwise be invisible to the audit.  All fields are categorical, so a
# distribution shift shows up directly as a total-variation distance and a
# calibrated chi-square.
_AUDIT_FIELDS = ("token", "answer", "key", "value", "key_pos",
                 "value_repeats", "naive_confusable", "difficulty_joint")
_AUDIT_DERIVED_FIELDS = ("key_pos", "value_repeats", "naive_confusable")

def _audit_seed(value) -> int:
    """Validate a Monte Carlo audit seed instead of silently truncating it.

    ``torch.Generator.manual_seed`` raises an incidental ``RuntimeError`` for
    bool/fractional seeds, and an ``int(seed)`` coercion would alias e.g.
    ``0.5`` to stream 0.  Audit p-values are part of the reproducibility
    contract, so every non-integral seed is translated into ``ValueError``
    before a replicate is drawn.
    """
    if isinstance(value, bool):
        raise ValueError(f"audit seed must be an integer, got {value!r}")
    try:
        seed = int(operator.index(value))
    except TypeError as exc:
        raise ValueError(
            f"audit seed must be an integer, got {value!r}") from exc
    # ``torch.Generator.manual_seed`` accepts the signed/unsigned 64-bit
    # range and raises an incidental ValueError outside it.  Reject the
    # out-of-range value here so the audit contract is explicit instead of
    # depending on the sampler's internal unpacking error.
    if not (-(1 << 63) <= seed <= (1 << 64) - 1):
        raise ValueError(
            f"audit seed must fit in 64 bits, got {value!r}")
    return seed


# Calibration of the distribution-shift check.  ``_AUDIT_SIM`` Monte Carlo
# replicates set the p-value resolution to ``1 / (_AUDIT_SIM + 1)`` (500
# replicates resolve the 0.0024 rank-one threshold of a 21-test
# Benjamini-Hochberg family at q=0.05); the fixed seed keeps the audit
# deterministic; ``_AUDIT_SMOOTH`` is the Haldane-Anscombe add-half count
# that keeps unobserved classes testable instead of producing an infinite
# statistic.
_AUDIT_SIM = 500
_AUDIT_SIM_SEED = 90210
_AUDIT_SMOOTH = 0.5

# Generator-calibration audit: the split audit can detect a held-out stream
# that differs from training, but a defect shared by every split (a key/value
# coupling, a non-uniform query rank, values drawn without replacement) is
# invisible to a between-split comparison.  These constants calibrate the
# model-free checks of the declared data-generating process itself.
_GENERATOR_SIM = 500
_GENERATOR_SIM_SEED = 271828
_GENERATOR_FAMILY_ALPHA = 0.05
# The joint independence test materializes an ``n_left * n_right`` table for
# every permutation replicate.  Cap it so an audit of an unusually large
# vocabulary abstains instead of allocating an unbounded matrix.
_GENERATOR_MAX_INDEPENDENCE_CELLS = 4_000_000


# Ambiguity strata used by :func:`difficulty_strata`.  The encoding is a
# two-bit code: bit 0 = naive-confusable, bit 1 = repeated answer value, so
# every id in [0, 4) names one combination and the id is stable across
# audits/records.
_AMBIGUITY_LABELS = ("none", "naive_confusable", "value_repeats", "both")

# Relative retrieval-position bins used by the default difficulty stratum.
# Three bins keep every cell populated at the audit/evaluation sizes while
# still separating early / middle / late retrieval, and the same id means
# the same *relative* position at every sequence length.
DEFAULT_DIFFICULTY_KEY_BINS = 3

_MAX_DIFFICULTY_STRATA = 1 << 20


def _difficulty_int(value, name):
    """Reject bool/fractional metadata instead of silently truncating it."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        value = int(operator.index(value))
    except TypeError as exc:
        raise ValueError(
            f"{name} must be an integer, got {value!r}") from exc
    if value > _MAX_TORCH_INDEX:
        raise ValueError(
            f"{name}={value} exceeds torch's signed 64-bit index range")
    return value


def difficulty_features(idx, answer, n_pairs):
    """Per-sample structural difficulty covariates of one batch.

    Derived from the validated ``[k1 v1 ... kn vn SEP query]`` layout (the
    query is the last column of ``idx``; the answer is stored separately),
    so no fitted statistic is involved:

    * ``key_pos`` - pair index of the matching key (retrieval distance);
    * ``value_repeats`` - the answer value occurs more than once among the
      values, so a perfect key lookup does not uniquely determine it;
    * ``naive_confusable`` - the query token occurs after the matching key
      as a value, so the naive "last visible match wins" rule differs from
      the exact pair lookup.

    These are exactly the task-structural factors that make a length sweep
    non-comparable: with more pairs per sequence the chance that a value
    repeats or that the query token appears later as a value grows, so raw
    accuracy at a longer length mixes a length effect with a data-
    difficulty effect.  :func:`difficulty_strata` turns the covariates into
    a small categorical stratum that can be used for post-stratification
    (:func:`cage.metrics.standardized_proportion`).

    Mirrors ``cage.train._error_case_flags`` so the audit measures exactly
    the difficulty factors used to explain errors downstream.  The batch
    geometry is checked instead of inferred from slicing: a narrower
    ``idx`` would otherwise silently pair the wrong columns and label a
    malformed batch as easy/hard.
    """
    n_pairs = _difficulty_int(n_pairs, "n_pairs")
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    if not _is_integer_tensor(idx) or idx.dim() != 2:
        raise ValueError(
            "idx must be a (batch, 2*n_pairs + 2) integer tensor, got "
            f"{type(idx)}")
    if not _is_integer_tensor(answer) or answer.dim() != 1:
        raise ValueError("answer must be a 1-D integer tensor")
    batch, width = idx.shape
    expected = 2 * n_pairs + 2
    if width != expected:
        raise ValueError(
            f"idx width {width} does not match the [k v]*{n_pairs} + SEP + "
            f"query layout ({expected} positions)")
    if int(answer.numel()) != int(batch):
        raise ValueError(
            f"answer has {int(answer.numel())} entries for {int(batch)} "
            f"sequences")
    query = idx[:, -1]
    # ``meta`` tensors carry no values to read back (they are a shape-only
    # device), so they skip the value-level checks while preserving
    # placement.
    readable = idx.device.type != "meta"
    keys = idx[:, :2 * n_pairs:2]
    if readable and int(idx.size(0)):
        # These are the same structural/label invariants the cleaning gate
        # enforces.  Without them this helper would happily describe a batch
        # (reserved SEP/PAD ids in content slots, duplicate keys, a misplaced
        # SEP) that ``validate_batch`` rejects, and could count the SEP slot
        # as a "later value" of a reserved query token.
        if not bool((idx[:, :-2] >= 2).all()) \
                or not bool((idx[:, -1] >= 2).all()):
            raise ValueError(
                "ids 0=PAD and 1=SEP are reserved; all key/value positions "
                "and the query must be selectable ids >= 2")
        if not bool(idx.eq(SEP_ID).sum(dim=1).eq(1).all()) \
                or not bool(idx[:, -2].eq(SEP_ID).all()):
            raise ValueError(
                "structure: expected exactly one SEP token at the fixed "
                "separator slot (T-2)")
        if keys.size(1) > 1:
            ordered = keys.sort(dim=1).values
            if not bool((ordered[:, 1:] != ordered[:, :-1]).all()):
                raise ValueError(
                    "structure: keys must be distinct within a sequence")
    match = keys.eq(query.unsqueeze(1))
    # A batch with no matching key (or a duplicated key) has no defined
    # retrieval position; ``argmax`` would silently label it as an easy
    # early-key sample.  Refuse it instead, matching the geometry checks.
    if readable and int(idx.size(0)) and not bool(
            match.sum(dim=1).eq(1).all()):
        raise ValueError(
            "idx must contain exactly one key equal to the final query "
            "token in every sequence")
    key_pos = match.to(torch.int64).argmax(dim=1)
    values = idx[:, 1:2 * n_pairs:2]
    # ``answer`` is normally already on ``idx``'s device, but the helper is
    # public: align it rather than failing a valid CPU-metadata/CUDA-batch
    # call (or, worse, broadcasting the wrong side).
    answer = answer.to(device=idx.device)
    paired_value = values.gather(1, key_pos.unsqueeze(1)).squeeze(1)
    if readable and int(idx.size(0)) and not bool(
            paired_value.eq(answer).all()):
        raise ValueError(
            "answer does not match the value paired with the query key; "
            "difficulty features would describe a different label")
    value_repeats = (values.eq(answer.unsqueeze(1)).sum(dim=1)
                     > 1).to(torch.int64)
    # Position indices must live on the batch's device: the comparison
    # below is boolean-ANDed with a tensor derived from ``idx``.
    positions = torch.arange(idx.size(1), device=idx.device).view(1, -1)
    visible = positions < (idx.size(1) - 1)
    last_match = torch.where(idx.eq(query.unsqueeze(1)) & visible,
                             positions,
                             torch.full_like(positions, -1)).max(dim=1).values
    naive_confusable = last_match.ne(key_pos.mul(2)).to(torch.int64)
    return {"key_pos": key_pos, "value_repeats": value_repeats,
            "naive_confusable": naive_confusable}
def difficulty_stratum_labels(n_key_bins=DEFAULT_DIFFICULTY_KEY_BINS):
    """JSON-safe labels for :func:`difficulty_strata` ids.

    Ids are ``key_bin * 4 + ambiguity_code``; the accompanying bounds are
    in *pair-index* space (``key_bin`` is a relative retrieval-position
    tercile, so the same id is comparable across lengths even though the
    absolute distance differs).
    """
    n_key_bins = _difficulty_int(n_key_bins, "n_key_bins")
    if n_key_bins < 1:
        raise ValueError(f"n_key_bins must be >= 1, got {n_key_bins}")
    if 4 * n_key_bins > _MAX_DIFFICULTY_STRATA:
        raise ValueError(
            f"n_key_bins={n_key_bins} would create {4 * n_key_bins} "
            f"strata; the maximum supported encoding is "
            f"{_MAX_DIFFICULTY_STRATA}")
    return [
        {"id": bin_idx * 4 + ambiguity,
         "key_bin": bin_idx,
         "ambiguity": _AMBIGUITY_LABELS[ambiguity]}
        for bin_idx in range(n_key_bins)
        for ambiguity in range(4)
    ]


def difficulty_strata(idx, answer, n_pairs,
                      n_key_bins=DEFAULT_DIFFICULTY_KEY_BINS):
    """Map each sample to a joint retrieval-position x ambiguity stratum.

    ``n_key_bins`` relative retrieval-position bins are crossed with the
    four ambiguity states of :data:`_AMBIGUITY_LABELS`, giving
    ``4 * n_key_bins`` strata.  A stratum is a *pre-treatment* covariate:
    it is fixed by the generated sequence, not by a model, so conditioning
    on it cannot remove the effect of the mechanism under test.  That is
    what makes it valid for post-stratifying accuracy across lengths:
    ``standardized_proportion(correct, n, reference_weights)`` reweights
    each length's per-stratum accuracy to the same reference mix instead
    of comparing compositions that differ by construction.

    Returns ``{"stratum", "key_bin", "ambiguity", "labels",
    "key_bin_bounds"}``.  ``stratum``/``key_bin``/``ambiguity`` are
    per-sample tensors; ``labels`` and ``key_bin_bounds`` describe the
    encoding for records (all JSON-safe).
    """
    n_key_bins = _difficulty_int(n_key_bins, "n_key_bins")
    if n_key_bins < 1:
        raise ValueError(f"n_key_bins must be >= 1, got {n_key_bins}")
    if 4 * n_key_bins > _MAX_DIFFICULTY_STRATA:
        raise ValueError(
            f"n_key_bins={n_key_bins} would create {4 * n_key_bins} "
            f"strata; the maximum supported encoding is "
            f"{_MAX_DIFFICULTY_STRATA}")
    n_pairs = _difficulty_int(n_pairs, "n_pairs")
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    features = difficulty_features(idx, answer, n_pairs)
    key_bin = (features["key_pos"] * n_key_bins // n_pairs).clamp(
        max=n_key_bins - 1).to(torch.long)
    # bit 0 = naive-confusable, bit 1 = repeated value.  Ordering repeat
    # above naive matches the historical ``difficulty_joint`` encoding at
    # n_key_bins == n_pairs, so the two audit views cannot drift apart.
    ambiguity = (features["value_repeats"] * 2
                 + features["naive_confusable"]).to(torch.long)
    stratum = key_bin * 4 + ambiguity
    bounds = [[math.ceil(bin_idx * n_pairs / n_key_bins),
               math.ceil((bin_idx + 1) * n_pairs / n_key_bins)]
              for bin_idx in range(n_key_bins)]
    return {
        "stratum": stratum,
        "key_bin": key_bin,
        "ambiguity": ambiguity,
        "labels": difficulty_stratum_labels(n_key_bins),
        "key_bin_bounds": bounds,
        "n_key_bins": n_key_bins,
        "n_strata": 4 * n_key_bins,
    }


def _difficulty_features(split, n_pairs):
    """Backward-compatible wrapper around :func:`difficulty_features`."""
    return difficulty_features(split["idx"], split["answer"], n_pairs)


def _joint_difficulty_values(features):
    """Encode ``(key_pos, value_repeats, naive_confusable)`` as one category.

    The binary ambiguity flags are the two least-significant digits, so the
    encoding is dense (``0 .. 4*n_pairs-1``) and no joint cell is dropped.
    """
    return (((features["key_pos"].to(torch.long) * 2
              + features["value_repeats"].to(torch.long)) * 2)
            + features["naive_confusable"].to(torch.long))


def _field_values(split, n_pairs, field):
    idx, answer = split["idx"], split["answer"]
    if field == "token":
        return idx.reshape(-1)
    if field == "answer":
        return answer.reshape(-1)
    if field == "key":
        return idx[:, :2 * n_pairs:2].reshape(-1)
    if field == "value":
        return idx[:, 1:2 * n_pairs:2].reshape(-1)
    if field == "difficulty_joint":
        return _joint_difficulty_values(_difficulty_features(split, n_pairs))
    if field in _AUDIT_DERIVED_FIELDS:
        return _difficulty_features(split, n_pairs)[field]
    raise ValueError(f"unknown audit field {field!r}")


def _field_categories(field, vocab, n_pairs):
    """Number of possible categories for a categorical audit field."""
    if field == "key_pos":
        return max(int(n_pairs), 1)
    if field == "difficulty_joint":
        return max(int(n_pairs), 1) * 4
    if field in ("value_repeats", "naive_confusable"):
        return 2
    return int(vocab)


def _counts(values, n_categories):
    if isinstance(n_categories, bool):
        raise ValueError(
            f"n_categories must be a non-negative integer, got "
            f"{n_categories!r}")
    try:
        n_categories = int(operator.index(n_categories))
    except TypeError as exc:
        raise ValueError(
            f"n_categories must be a non-negative integer, got "
            f"{n_categories!r}") from exc
    if n_categories < 0:
        raise ValueError(
            f"n_categories must be non-negative, got {n_categories}")
    def _bad_values(exc=None):
        return ValueError(
            "audit values must be a real numeric vector, got "
            f"{type(values).__name__}")

    if torch.is_tensor(values):
        raw = values
    else:
        try:
            raw = torch.as_tensor(values)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            # A list of finite floats too large for float32 can fail the
            # default conversion; retry directly at float64.
            try:
                raw = torch.as_tensor(values, dtype=torch.float64)
            except (TypeError, ValueError, RuntimeError, OverflowError):
                raise _bad_values(exc) from exc
        if raw.is_floating_point():
            # ``as_tensor`` captures a Python float list as float32, which
            # rounds integral category ids above 2**24 to a different
            # category.  Keep the caller's Python values at float64.
            try:
                raw = torch.as_tensor(values, dtype=torch.float64)
            except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                raise _bad_values(exc) from exc
    if raw.dim() > 1:
        raise ValueError(
            "audit values must be one-dimensional (one category id per "
            f"sample), got shape {tuple(raw.shape)}; flattening a matrix "
            "would merge independent audit samples")
    raw = raw.reshape(-1)
    # ``Tensor.to(long)`` silently discards the imaginary part (with only a
    # warning), which would fabricate a count the caller never supplied.
    if raw.is_complex():
        raise ValueError("audit values must be real integers")
    # A boolean is a flag, not a category id: casting it to long would
    # silently turn ``[True, False]`` into the categories 1 and 0 and
    # fabricate counts the caller never supplied.
    if raw.dtype == torch.bool:
        raise ValueError("audit values must be integers, not booleans")
    if raw.is_floating_point() and not bool(raw.eq(raw.round()).all()):
        raise ValueError("audit values must be integers")
    flat = raw.to(torch.long).cpu()
    if flat.numel():
        lo, hi = int(flat.min()), int(flat.max())
        if lo < 0 or hi >= n_categories:
            raise ValueError(
                f"audit values must lie in [0, {n_categories}), got range "
                f"[{lo}, {hi}]")
    try:
        return torch.bincount(flat, minlength=n_categories).to(torch.int64)
    except (MemoryError, OverflowError, RuntimeError) as exc:
        raise ValueError(
            f"n_categories={n_categories} is too large to count") from exc


def _distribution_from_counts(counts):
    counts = counts.to(torch.float64)
    return counts / counts.sum().clamp_min(1.0)


def _distribution(values, n_categories):
    return _distribution_from_counts(_counts(values, n_categories))


def _total_variation(p, q):
    return float(0.5 * (p - q).abs().sum())


def _noise_floor(stream_values, n_categories):
    """Same-size sampling-noise scale of one audit field.

    With at least two training streams the floor is the mean distance of
    each stream to the pooled distribution of the *other* streams
    (leave-one-out) - exactly the operation the held-out comparison
    performs, at the same sample size.  Including the stream in its own
    reference would understate the floor by ``(m-1)/m``, which would make
    the tolerance tighter than the advertised margin.  With a single
    training stream the stream is split in half instead and the half-size
    distance is rescaled by ``sqrt(2)`` (the TV between two independent
    ``n/2`` draws is ``sqrt(2)`` times the TV between two ``n`` draws).
    All inputs are fitted on training data only.
    """
    if not stream_values:
        # No training stream to fit a reference floor from; the caller
        # should treat the floor as undefined rather than crash on an
        # empty list.
        return float("nan")
    if len(stream_values) >= 2:
        counts = [_counts(values, n_categories) for values in stream_values]
        dists = [_distribution_from_counts(count) for count in counts]
        total = counts[0].clone()
        for count in counts[1:]:
            total.add_(count)
        floors = []
        for j in range(len(counts)):
            # Leave-one-out reference from the cached counts instead of
            # re-binning the concatenation of all other streams.
            floors.append(
                _total_variation(dists[j],
                                 _distribution_from_counts(total - counts[j])))
        return sum(floors) / len(floors)
    values = stream_values[0]
    if values.numel() < 2:
        return float("nan")
    half = values.numel() // 2
    lo = _counts(values[:half], n_categories)
    hi = _counts(values[half:], n_categories)
    floor = _total_variation(_distribution_from_counts(lo),
                             _distribution_from_counts(hi))
    return floor / math.sqrt(2.0)


def _distribution_shift_p(observed, reference, n_sim=_AUDIT_SIM,
                          seed=_AUDIT_SIM_SEED):
    """Monte Carlo p-value for "held-out and training share a distribution".

    ``observed`` / ``reference`` are categorical count vectors at the
    held-out and pooled-training sample sizes, fitted on training data
    only.  The statistic is the Pearson chi-square of the observed counts
    against their expectation under the reference distribution,

    ``X^2 = sum_i (o_i - n_h * p_i)^2 / (n_h * p_i),  p_i = (r_i + c) /
    (N_r + c * K)``

    with the Haldane-Anscombe correction ``c = _AUDIT_SMOOTH`` so classes
    unseen in the reference stay testable (a class with ``r_i = 0`` that
    nevertheless appears held-out then contributes a large finite term
    instead of an infinite one).  The null distribution of ``X^2`` is not
    read off a chi-square table - the reference itself is a finite sample -
    but re-derived by a parametric bootstrap that redraws **both** count
    vectors from the smoothed reference probabilities, reference at
    training size and held-out at held-out size, and recomputes the
    statistic with the simulated reference (Poisson draws are a standard
    stand-in for multinomial counts whose conditioning correction is
    ``O(1/N)`` and negligible against the statistic's sampling variation
    at these sizes).  The returned p-value is
    ``(1 + #{X^2_sim >= X^2_obs}) / (n_sim + 1)``, deterministic in the
    audit seed.

    This replaces a "total variation <= 3 x noise floor" tolerance because
    that point rule goes blind exactly where the audit matters most: the
    supervised answer spreads ~1e3 samples over ``vocab - 2`` classes, so
    even two *same-source* draws already have TV ~ 0.3, three times the
    floor approaches the maximum TV of 1.0, and a held-out split drawn
    from half the class range (TV ~ 0.5) passes unnoticed.  The chi-square
    statistic senses mass moving between classes at the same sample sizes
    and the bootstrap quantiles it against the real null (verified in
    ``python3 -m cage.data``: same-source draws are not flagged, a
    half-range split is).  Returns ``(nan, nan)`` for empty inputs.
    """
    def _as_count_vector(values, name):
        def _bad(exc=None):
            return ValueError(
                f"{name} counts must be a real numeric vector, got "
                f"{type(values).__name__}")

        if torch.is_tensor(values):
            raw = values
        else:
            try:
                raw = torch.as_tensor(values)
            except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                # Finite float values can exceed float32's range even though
                # they are valid counts; retry the conversion at float64.
                try:
                    raw = torch.as_tensor(values, dtype=torch.float64)
                except (TypeError, ValueError, RuntimeError, OverflowError):
                    raise _bad(exc) from exc
            if raw.is_floating_point():
                # Avoid the default float32 capture for Python float lists:
                # it silently rounds large integral counts (and can make a
                # finite count look infinite).
                try:
                    raw = torch.as_tensor(values, dtype=torch.float64)
                except (TypeError, ValueError, RuntimeError,
                        OverflowError) as exc:
                    raise _bad(exc) from exc
        if raw.dim() > 1:
            raise ValueError(
                f"{name} counts must be one-dimensional (one count per "
                f"category), got shape {tuple(raw.shape)}; flattening a "
                f"matrix would invent categories and change the sample size")
        if raw.is_complex():
            raise ValueError(f"{name} counts must be real integers")
        if raw.dtype == torch.bool:
            raise ValueError(
                f"{name} counts must be integers, not booleans")
        try:
            return raw.detach().cpu().reshape(-1).to(torch.float64)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError(
                f"{name} counts must be a real numeric vector") from exc

    obs = _as_count_vector(observed, "observed")
    ref = _as_count_vector(reference, "reference")
    if obs.numel() != ref.numel():
        raise ValueError(
            f"observed and reference count vectors must have the same "
            f"length, got {obs.numel()} and {ref.numel()}")
    for name, counts in (("observed", obs), ("reference", ref)):
        if not bool(torch.isfinite(counts).all()):
            raise ValueError(f"{name} counts contain NaN/Inf")
        if bool((counts < 0).any()):
            raise ValueError(f"{name} counts must be non-negative")
        if not bool(counts.eq(counts.round()).all()):
            raise ValueError(f"{name} counts must be integers")
    n_ref = float(ref.sum())
    n_h = float(obs.sum())
    if not (math.isfinite(n_ref) and math.isfinite(n_h)):
        # Individually finite counts can overflow their sum; no expected
        # count (and therefore no test) is representable in that case.
        return float("nan"), float("nan")
    if n_ref <= 0.0 or n_h <= 0.0:
        return float("nan"), float("nan")
    if isinstance(n_sim, bool):
        return float("nan"), float("nan")
    try:
        n_sim = operator.index(n_sim)
    except TypeError:
        return float("nan"), float("nan")
    if n_sim < 1:
        return float("nan"), float("nan")
    k = ref.numel()
    probs = (ref + _AUDIT_SMOOTH) / (n_ref + _AUDIT_SMOOTH * k)
    expected = n_h * probs
    chi2 = float((((obs - expected) ** 2) / expected).sum())
    if not math.isfinite(chi2):
        return float("nan"), float("nan")

    gen = torch.Generator().manual_seed(_audit_seed(seed))
    try:
        ref_sim = torch.poisson(
            (n_ref * probs).expand(n_sim, k).contiguous(), generator=gen)
        h_sim = torch.poisson(
            (n_h * probs).expand(n_sim, k).contiguous(), generator=gen)
        # Large finite means saturate inside ``torch.poisson`` to negative
        # int64 values instead of raising.  Treat that failed simulation as
        # an abstention rather than comparing the observed statistic with
        # fabricated counts.
        if not (bool(torch.isfinite(ref_sim).all())
                and bool(torch.isfinite(h_sim).all())
                and bool((ref_sim >= 0).all())
                and bool((h_sim >= 0).all())):
            return float("nan"), float("nan")
        probs_sim = (ref_sim + _AUDIT_SMOOTH) / (
            ref_sim.sum(1, keepdim=True) + _AUDIT_SMOOTH * k)
        expected_sim = h_sim.sum(1, keepdim=True) * probs_sim
        chi2_sim = (((h_sim - expected_sim) ** 2) / expected_sim).sum(1)
        p_value = float((1.0 + (chi2_sim >= chi2).to(torch.float64).sum())
                        / (n_sim + 1.0))
    except (TypeError, RuntimeError, ValueError, OverflowError):
        # The audit is a reporting companion; a count magnitude the Poisson
        # sampler cannot represent (or an n_sim too large for torch's size
        # unpacking) must abstain, not abort the report.
        return float("nan"), float("nan")
    return p_value, chi2


# Outlier/tail audit.  The cleaning gate rejects invalid, label-noisy and
# duplicated samples; the calibrated marginal tests above detect a shift in
# each field separately.  Neither catches an *atypical sample* or a heavier
# difficulty tail in a held-out split, so a train-fitted typicality score is
# added here.  Every probability and the tail threshold are fitted on the
# training streams only; held-out splits are only scored (transform-only).
_OUTLIER_FIELDS = ("answer", "key_pos", "value_repeats", "naive_confusable")
_OUTLIER_QUANTILE = 0.99
# Conservative tail-rate bound: flag a held-out split only when its atypical
# rate exceeds both a multiple of the training rate and an absolute floor.
# At the default ~1% training rate this needs >5% of a held-out split to be
# in the training tail - far beyond same-distribution sampling noise (a
# Binomial(1024, 0.01) reaching 5% has probability ~1e-30), so the rule has
# essentially no false-positive rate while still catching a real tail shift.
_OUTLIER_RATE_MULTIPLIER = 5.0
_OUTLIER_RATE_FLOOR = 0.05

# Per-sample fields used by the training-stationarity diagnostic: each has
# exactly one value per sample, so "first half of the stream" maps onto
# whole samples (unlike pooled token fields, which are per-token).
_STATIONARITY_FIELDS = ("answer", "key_pos", "value_repeats",
                        "naive_confusable")


def _typicality_scores(split_values, reference) -> torch.Tensor:
    """Per-sample train-fitted surprisal over the structural audit fields.

    ``split_values`` maps each :data:`_OUTLIER_FIELDS` name to that split's
    per-sample values; ``reference`` holds Haldane-smoothed training
    category probabilities.  The generator draws the fields independently,
    so the joint negative log-likelihood is the sum of the per-field
    surprisals.  A category unseen in training keeps a finite (smoothed)
    probability, so the score is always finite; no held-out statistic enters
    the reference.
    """
    total = None
    for field in _OUTLIER_FIELDS:
        values = split_values[field].reshape(-1).to(torch.long)
        surprisal = -reference[field][values].log()
        total = surprisal if total is None else total + surprisal
    return total


def _outlier_audit(split_values, train_names, names, reference) -> dict:
    """Train-fitted atypical-sample rate per split.

    The threshold is the training ``_OUTLIER_QUANTILE`` of the typicality
    score (fit on the pooled training streams).  A split's outlier rate is
    the share of samples beyond that threshold.  Held-out splits are flagged
    only under the conservative rate rule documented above, so a genuine
    same-distribution split cannot trip the gate while a split whose
    difficulty tail has moved can.
    """
    def scores(name):
        return _typicality_scores(
            {field: split_values[(name, field)] for field in _OUTLIER_FIELDS},
            reference)

    train_scores = torch.cat([scores(name) for name in train_names])
    threshold = torch.quantile(train_scores, _OUTLIER_QUANTILE)
    flagged = {}
    for name in names:
        s = scores(name)
        flagged[name] = int((s > threshold).sum())
    rate = {name: flagged[name] / max(int(
        split_values[(name, _OUTLIER_FIELDS[0])].numel()), 1)
        for name in names}
    n_train = sum(int(split_values[(name, _OUTLIER_FIELDS[0])].numel())
                  for name in train_names)
    train_rate = (sum(flagged[name] for name in train_names)
                  / max(n_train, 1))
    bound = max(_OUTLIER_RATE_MULTIPLIER * train_rate, _OUTLIER_RATE_FLOOR)
    return {
        "fields": list(_OUTLIER_FIELDS),
        "quantile": float(_OUTLIER_QUANTILE),
        "threshold_train": float(threshold),
        "rate_train": float(train_rate),
        "rate_by_split": rate,
        "flagged_by_split": flagged,
        "rate_bound": float(bound),
        "shift_by_split": {
            name: bool(rate[name] > bound)
            for name in names if name not in train_names},
    }


def _support_coverage(pooled_counts, split_values, heldout_names, categories,
                      fields) -> dict:
    """Train-fitted category support and per-split unseen-value rates.

    For each content field, the pooled training counts define which
    selectable categories were ever observed (ids 0=PAD and 1=SEP are never
    selectable content).  A held-out value whose category is absent from
    that support cannot be predicted (an answer) or meaningfully embedded
    (an input token), so its rate is a transparent bound on achievable
    accuracy.  Derived structural fields (key position, repeated value,
    naive-confusable) use their full category range: their zero/one values
    are measurements, not reserved ids.  The support comes from training
    only; held-out values are looked up, never used to define it.
    """
    out = {}
    for field in fields:
        counts = pooled_counts[field]
        seen = counts > 0
        content = field not in _AUDIT_DERIVED_FIELDS
        if content:
            seen[:min(2, int(seen.numel()))] = False
            n_categories = max(int(categories[field]) - 2, 1)
        else:
            n_categories = max(int(categories[field]), 1)
        entry = {
            "n_categories": n_categories,
            "n_seen_train": int(seen.sum()),
            "fraction_seen_train": int(seen.sum()) / n_categories,
            "unseen_by_split": {},
        }
        for name in heldout_names:
            values = split_values[(name, field)].reshape(-1).to(torch.long)
            if content:
                values = values[values >= 2]
            unseen = ~seen[values]
            entry["unseen_by_split"][name] = {
                "n": int(unseen.sum()),
                "rate": (float(unseen.to(torch.float64).mean())
                         if unseen.numel() else 0.0),
            }
        out[field] = entry
    return out


def _train_stationarity(split_values, splits, train_names, categories) -> dict:
    """First-half vs second-half drift inside each training stream.

    The held-out comparison asks whether evaluation data matches training;
    it cannot ask whether *training itself* is stationary.  If an early
    training block differs from a late one, a single held-out split cannot
    represent the future even with zero overlap.  Both halves come from the
    training stream, so this fits and scores on training data only.
    """
    out = {}
    for name in train_names:
        n_rows = int(splits[name]["idx"].size(0))
        half = n_rows // 2
        if half < 2:
            continue
        for field in _STATIONARITY_FIELDS:
            values = split_values[(name, field)].reshape(-1)
            if values.numel() != n_rows:
                continue
            early = _counts(values[:half], categories[field])
            late = _counts(values[half:], categories[field])
            p_value, chi2 = _distribution_shift_p(late, early)
            tv = _total_variation(_distribution_from_counts(late),
                                  _distribution_from_counts(early))
            out[f"{name}:{field}"] = {"p": p_value, "chi2": chi2, "tv": tv}
    return out


# Calibration of the answer-uniformity test (``answer_uniformity_test``):
# the sample size enters the test, so only the simulation count and its
# fixed seed are constants here.
_UNIFORMITY_SIM = 500
_UNIFORMITY_SIM_SEED = 8675309


def _exact_count_vector(values):
    if torch.is_tensor(values):
        flat = values.detach().cpu().reshape(-1)
        if flat.dtype == torch.bool or flat.is_complex():
            return None
        items = flat.tolist()
    elif isinstance(values, (str, bytes, Mapping)):
        return None
    else:
        try:
            items = list(values)
        except TypeError:
            return None
    exact = []
    for item in items:
        if isinstance(item, bool):
            return None
        try:
            exact.append(operator.index(item))
            continue
        except TypeError:
            pass
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number) or not number.is_integer():
            return None
        exact.append(int(number))
    return exact


def _exact_count_total(values):
    exact = _exact_count_vector(values)
    return None if exact is None else sum(exact)


def answer_uniformity_test(counts, n_sim=_UNIFORMITY_SIM,
                           seed=_UNIFORMITY_SIM_SEED) -> dict:
    """Monte Carlo goodness-of-fit of answer counts against *uniform*.

    Why this is a *decision*, not a diagnostic: the primary metric of the
    recall task is plain accuracy, and that choice is only meaningful
    because every value class is (by construction) equally likely - under
    a majority class accuracy degrades into a class-prior metric and the
    explicit F1 variants / PR-AUC companions in ``cage.metrics`` would
    become the headline.  The claim "the task is balanced multi-class"
    must therefore be *tested* on the drawn stream, not inferred from the
    generator description.

    ``counts`` is a 1-D count vector over the selectable classes (ids
    ``2..vocab-1`` - PAD and SEP are never answers; the caller slices
    accordingly).  The statistic is the Pearson chi-square against the
    uniform expectation ``n / K``.  The null is calibrated by a
    parametric bootstrap that redraws counts as independent
    ``Poisson(n / K)`` variates - the standard Poissonization of
    multinomial counts, whose conditioning correction is ``O(1/n)`` and
    negligible at these sizes (same convention as
    :func:`_distribution_shift_p`) - at the observed sample size, so the
    small expected count per class (about 10 at the default audit size,
    where the asymptotic chi-square table is not reliable) does not have
    to be trusted.

    Returns ``{"n", "n_classes", "expected_per_class", "chi2", "df",
    "p_value", "uniform_supported"}`` with
    ``p = (1 + #{chi2_sim >= chi2_obs}) / (n_sim + 1)`` and
    ``uniform_supported = p > 0.05``.  A *passing* test bounds gross
    imbalance at the audit sample size; it is not an equivalence proof
    for arbitrarily small skew (use the reported per-class shares for
    the descriptive bound).  Degenerate input (no samples, fewer than
    two classes) returns NaN fields and ``uniform_supported=False``.
    """
    empty = {"n": 0, "n_classes": 0, "expected_per_class": float("nan"),
             "chi2": float("nan"), "df": 0, "p_value": float("nan"),
             "uniform_supported": False}

    def _exact_payload():
        exact = _exact_count_vector(counts)
        if exact is None:
            return None
        if any(value < 0 for value in exact):
            raise ValueError("answer counts must be non-negative")
        total = sum(exact)
        k = len(exact)
        try:
            expected = float(total / k) if k else float("nan")
        except (OverflowError, ZeroDivisionError):
            expected = float("nan")
        return {"n": int(total), "n_classes": int(k),
                "expected_per_class": expected, "chi2": float("nan"),
                "df": int(max(k - 1, 0)), "p_value": float("nan"),
                "uniform_supported": False}

    try:
        raw = torch.as_tensor(counts).detach()
    except (TypeError, ValueError, RuntimeError, OverflowError):
        # Exact Python integers can exceed int64 while still being valid
        # counts representable in float64.  Retry at float64 before treating
        # the container as non-numeric; the later sum still rejects values
        # whose *total* overflows.
        try:
            raw = torch.as_tensor(counts, dtype=torch.float64).detach()
        except (TypeError, ValueError, RuntimeError, OverflowError):
            exact = _exact_payload()
            if exact is not None:
                return exact
            # A non-numeric container is not a count vector.  Like the other
            # reporting companions, abstain instead of letting ``as_tensor``
            # escape as a TypeError/RuntimeError from an audit readout.
            return empty
    if raw.dim() > 1:
        raise ValueError(
            "answer counts must be one-dimensional (one count per class), "
            f"got shape {tuple(raw.shape)}; flattening a matrix would invent "
            "classes and change the sample size")
    raw = raw.reshape(-1)
    if raw.dtype == torch.bool:
        raise ValueError("answer counts must be integers, not booleans")
    if raw.is_complex():
        raise ValueError("answer counts must be real integers")
    # Convert the *original* container straight to float64.  Converting the
    # probe instead would first capture Python float lists as float32: a
    # finite count as large as 1e39 then becomes +inf (and is rejected as
    # non-finite), while an integral count above 2**24 is rounded and
    # silently changes the sample size.
    try:
        c = torch.as_tensor(counts, dtype=torch.float64).detach(
        ).cpu().reshape(-1)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return empty
    if not bool(torch.isfinite(c).all()):
        raise ValueError("answer counts contain NaN/Inf")
    if bool((c < 0).any()):
        raise ValueError("answer counts must be non-negative")
    if not bool(c.eq(c.round()).all()):
        raise ValueError("answer counts must be integers")
    n = float(c.sum())
    exact_n = _exact_count_total(counts)
    total = exact_n if exact_n is not None else n
    k = c.numel()
    out = {"n": (int(total) if exact_n is not None
                 else (int(n) if math.isfinite(n) else 0)),
           "n_classes": int(k),
           "expected_per_class": (float(total / k)
                                  if k and (exact_n is not None
                                            or math.isfinite(n))
                                  else float("nan")),
           "chi2": float("nan"), "df": int(max(k - 1, 0)),
           "p_value": float("nan"), "uniform_supported": False}
    if not math.isfinite(n):
        # Individually finite counts can still overflow their sum; an
        # infinite expected count cannot be simulated or tested.
        return out
    if k < 2 or n <= 0.0:
        return out
    if isinstance(n_sim, bool):
        return out
    try:
        n_sim_int = operator.index(n_sim)
    except TypeError:
        return out
    if n_sim_int < 1:
        return out
    n_sim = n_sim_int
    expected = total / k
    chi2 = float((((c - expected) ** 2) / expected).sum())
    gen = torch.Generator().manual_seed(_audit_seed(seed))
    try:
        sim = torch.poisson(torch.full((n_sim, k), expected), generator=gen)
        # ``torch.poisson`` silently saturates a mean above the int64 range
        # to the minimum integer instead of raising.  A negative draw is a
        # failed simulation, not evidence for uniformity, so abstain rather
        # than publishing a p-value computed from fabricated counts.
        if not bool(torch.isfinite(sim).all()) \
                or bool((sim < 0).any()):
            return out
        chi2_sim = (((sim - expected) ** 2) / expected).sum(1)
        p_value = float((1.0 + (chi2_sim >= chi2).to(torch.float64).sum())
                        / (n_sim + 1.0))
    except (TypeError, RuntimeError, ValueError, OverflowError):
        # The expected count can be finite but too large for the Poisson
        # sampler's internal float conversion, and an arbitrarily large
        # n_sim overflows ``torch.full``'s size unpacking.  This audit is a
        # reporting companion; abstain rather than abort the whole report.
        return out
    out.update({"chi2": chi2, "p_value": p_value,
                "uniform_supported": bool(p_value > 0.05)})
    return out


def _generator_offset_seed(seed, offset):
    """Deterministic per-test seed without leaving the accepted 64-bit range."""
    return (_audit_seed(seed) + int(offset)) % (1 << 64)


def _category_ids(values, n_categories, label):
    """Validate a categorical vector and return its int64 ids.

    This is the paired-input counterpart of :func:`_counts`: the
    independence audit needs the original left/right alignment, so reducing
    both sides to frequency vectors would destroy the statistic.
    """
    n_categories = _positive_audit_count(
        n_categories, f"{label} category count")
    if n_categories < 2:
        raise ValueError(
            f"{label} needs at least two categories, got {n_categories}")
    if torch.is_tensor(values):
        raw = values.detach()
    else:
        try:
            raw = torch.as_tensor(values)
        except (TypeError, ValueError, RuntimeError, OverflowError):
            try:
                raw = torch.as_tensor(values, dtype=torch.float64)
            except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                raise ValueError(
                    f"{label} must be a real numeric vector, got "
                    f"{type(values).__name__}") from exc
        if raw.is_floating_point():
            # A Python float list is first captured as float32; convert the
            # original container at float64 so an integral id above 2**24
            # is not rounded into a different category.
            try:
                raw = torch.as_tensor(values, dtype=torch.float64)
            except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                raise ValueError(
                    f"{label} must be a real numeric vector, got "
                    f"{type(values).__name__}") from exc
    if raw.dim() > 1:
        raise ValueError(
            f"{label} must be one-dimensional, got shape {tuple(raw.shape)}")
    raw = raw.reshape(-1)
    if raw.is_complex():
        raise ValueError(f"{label} must be real integers")
    if raw.dtype == torch.bool:
        raise ValueError(f"{label} must be integers, not booleans")
    if raw.is_floating_point():
        if not bool(torch.isfinite(raw).all()):
            raise ValueError(f"{label} contains NaN/Inf")
        if not bool(raw.eq(raw.round()).all()):
            raise ValueError(f"{label} must be integers")
    try:
        ids = raw.to(torch.long).cpu()
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            f"{label} must be a real integer vector, got dtype {raw.dtype}"
        ) from exc
    if ids.numel():
        lo, hi = int(ids.min()), int(ids.max())
        if lo < 0 or hi >= n_categories:
            raise ValueError(
                f"{label} must lie in [0, {n_categories}), got range "
                f"[{lo}, {hi}]")
    return ids


def _categorical_independence_test(left, right, n_left, n_right,
                                   n_sim=_GENERATOR_SIM,
                                   seed=_GENERATOR_SIM_SEED) -> dict:
    """Permutation G-test of independence for two aligned categorical vectors.

    The statistic is the mutual-information G statistic of the
    ``n_left x n_right`` contingency table.  Its null distribution is
    generated by permuting the right-hand labels, which preserves both
    marginals and therefore remains valid with the sparse tables this task
    produces (a 510-class key/value table has far more cells than samples,
    where the asymptotic chi-square table is unreliable).  The p-value is
    ``(1 + #{G_perm >= G_obs}) / (n_sim + 1)`` and is deterministic in
    ``seed``.  A large ``n_left * n_right`` table returns ``valid=False``
    instead of allocating it.
    """
    empty = {
        "statistic": float("nan"), "p_value": float("nan"),
        "n": 0, "n_left": 0, "n_right": 0, "n_sim": 0,
        "valid": False, "reason": "",
    }
    n_left = _positive_audit_count(n_left, "left category count")
    n_right = _positive_audit_count(n_right, "right category count")
    cells = n_left * n_right
    if cells > _GENERATOR_MAX_INDEPENDENCE_CELLS:
        empty["reason"] = (
            f"{n_left}x{n_right}={cells} joint cells exceed the "
            f"{_GENERATOR_MAX_INDEPENDENCE_CELLS}-cell audit budget")
        return empty
    try:
        left_ids = _category_ids(left, n_left, "left")
        right_ids = _category_ids(right, n_right, "right")
    except ValueError as exc:
        empty["reason"] = str(exc)
        return empty
    if left_ids.numel() != right_ids.numel():
        empty["reason"] = "left and right vectors must have equal length"
        return empty
    n = int(left_ids.numel())
    if n < 2 or n_left < 2 or n_right < 2:
        empty["reason"] = (
            "independence needs at least two observations and two "
            "categories on each side")
        return empty
    try:
        n_sim = _positive_audit_count(n_sim, "n_sim")
    except ValueError as exc:
        empty["reason"] = str(exc)
        return empty
    seed = _audit_seed(seed)
    left_codes = left_ids * n_right

    def statistic(right_values):
        table = torch.bincount(
            left_codes + right_values, minlength=cells
        ).to(torch.float64).reshape(n_left, n_right)
        row = table.sum(dim=1, keepdim=True)
        col = table.sum(dim=0, keepdim=True)
        expected = row.mul(col).div_(float(n))
        nonzero = table > 0
        return float((table[nonzero]
                      * torch.log(table[nonzero] / expected[nonzero])).sum())

    observed = statistic(right_ids)
    gen = torch.Generator().manual_seed(seed)
    exceed = 0
    for _ in range(n_sim):
        permuted = right_ids[torch.randperm(n, generator=gen)]
        if statistic(permuted) >= observed - 1e-12:
            exceed += 1
    return {
        "statistic": observed,
        "p_value": (1.0 + exceed) / (n_sim + 1.0),
        "n": n, "n_left": int(n_left), "n_right": int(n_right),
        "n_sim": int(n_sim), "valid": True, "reason": "",
    }


def _expected_value_repeat_rate(n_pairs, vocab) -> float:
    """Birthday probability that a row draws at least one repeated value.

    Values are sampled independently with replacement from the
    ``vocab - 2`` selectable ids, so the probability all ``n_pairs`` values
    are distinct is the falling factorial ``k! / (k - n)! / k**n``.  This
    is the DGP-implied companion to the observed
    ``value_repeat_rate``: a generator that silently sampled values without
    replacement would produce a zero rate while leaving every marginal
    uniform.  For a large vocabulary the log-gamma difference between two
    almost equal arguments loses nearly all significant digits, so the
    direct log-product is summed whenever that is cheap.
    """
    n_pairs = _positive_audit_count(n_pairs, "n_pairs")
    vocab = _positive_audit_count(vocab, "vocab")
    selectable = vocab - 2
    if selectable < 2:
        raise ValueError(
            f"vocab={vocab} leaves fewer than two selectable values")
    if n_pairs <= 1:
        return 0.0
    if n_pairs > selectable:
        return 1.0
    if n_pairs <= 1 << 20:
        log_distinct = math.fsum(
            math.log1p(-index / selectable)
            for index in range(1, n_pairs))
    else:
        log_distinct = (math.lgamma(selectable + 1.0)
                        - math.lgamma(selectable - n_pairs + 1.0)
                        - n_pairs * math.log(selectable))
    return min(1.0, max(0.0, -math.expm1(log_distinct)))


def _binomial_rate_test(observed, n, probability, n_sim=_GENERATOR_SIM,
                        seed=_GENERATOR_SIM_SEED) -> dict:
    """Monte Carlo two-sided binomial test for one observed rate."""
    empty = {
        "observed": 0, "n": 0, "observed_rate": float("nan"),
        "expected_rate": float("nan"), "expected": float("nan"),
        "p_value": float("nan"), "n_sim": 0, "valid": False,
        "reason": "",
    }
    try:
        n = _positive_audit_count(n, "n")
    except ValueError as exc:
        empty["reason"] = str(exc)
        return empty
    if isinstance(observed, bool):
        empty["reason"] = "observed count must be an integer, not bool"
        return empty
    try:
        observed = int(operator.index(observed))
    except TypeError:
        empty["reason"] = "observed count must be an integer"
        return empty
    if observed < 0 or observed > n:
        empty["reason"] = "observed count must lie in [0, n]"
        return empty
    try:
        probability = float(probability)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        empty["reason"] = "probability must be a real number in [0, 1]"
        return empty
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        empty["reason"] = "probability must lie in [0, 1]"
        return empty
    try:
        n_sim = _positive_audit_count(n_sim, "n_sim")
    except ValueError as exc:
        empty["reason"] = str(exc)
        return empty
    seed = _audit_seed(seed)
    expected = n * probability
    if probability <= 0.0:
        p_value = 1.0 if observed == 0 else 0.0
    elif probability >= 1.0:
        p_value = 1.0 if observed == n else 0.0
    else:
        gen = torch.Generator().manual_seed(seed)
        trials = torch.full((n_sim,), float(n), dtype=torch.float64)
        probs = torch.full((n_sim,), probability, dtype=torch.float64)
        simulated = torch.binomial(trials, probs,
                                   generator=gen).to(torch.float64)
        deviation = abs(observed - expected)
        p_value = float(
            (1.0 + ((simulated - expected).abs()
                    >= deviation - 1e-12).sum().item())
            / (n_sim + 1.0))
    return {
        "observed": observed, "n": int(n),
        "observed_rate": observed / n,
        "expected_rate": probability,
        "expected": expected, "p_value": p_value,
        "n_sim": int(n_sim), "valid": True, "reason": "",
    }


def _generator_calibration_from_idx(idx, answer, n_pairs, vocab,
                                    n_sim=_GENERATOR_SIM,
                                    sim_seed=_GENERATOR_SIM_SEED) -> dict:
    """Calibrate the declared DGP on an already-validated realized stream."""
    check_data_geometry(n_pairs, vocab)
    n_pairs = _positive_audit_count(n_pairs, "n_pairs")
    vocab = _positive_audit_count(vocab, "vocab")
    if not _is_integer_tensor(idx) or idx.dim() != 2:
        raise ValueError("idx must be a 2-D integer tensor")
    rows, width = int(idx.size(0)), int(idx.size(1))
    if width != 2 * n_pairs + 2:
        raise ValueError(
            f"idx width {width} does not match n_pairs={n_pairs} "
            f"(expected {2 * n_pairs + 2})")
    answer = _category_ids(answer, vocab, "answer").reshape(-1)
    if answer.numel() != rows:
        raise ValueError("answer must carry one entry per row of idx")
    idx = idx.detach().cpu().to(torch.long)
    keys = idx[:, :2 * n_pairs:2]
    values = idx[:, 1:2 * n_pairs:2]
    query = idx[:, -1]
    match = keys.eq(query.unsqueeze(1))
    if not bool(match.sum(dim=1).eq(1).all()):
        raise ValueError(
            "each row must contain its query key exactly once; run "
            "validate_batch before the generator calibration audit")
    query_rank = match.to(torch.long).argmax(dim=1)
    selectable = vocab - 2
    n_pair_obs = rows * n_pairs
    if n_pair_obs == 0:
        raise ValueError("generator calibration needs at least one row")

    marginal_specs = (
        ("query_rank", query_rank, n_pairs),
        ("key", keys.reshape(-1).sub(2), selectable),
        ("value", values.reshape(-1).sub(2), selectable),
        ("query_key", query.sub(2), selectable),
        ("answer", answer.sub(2), selectable),
    )
    marginals = {}
    for offset, (name, values_, categories) in enumerate(marginal_specs, 1):
        try:
            counts = _counts(values_, categories)
        except ValueError as exc:
            test = {
                "n": int(values_.numel()), "n_classes": int(categories),
                "expected_per_class": float("nan"), "chi2": float("nan"),
                "df": 0, "p_value": float("nan"),
                "uniform_supported": False, "reason": str(exc),
                "valid": False,
            }
        else:
            test = answer_uniformity_test(
                counts, n_sim=n_sim,
                seed=_generator_offset_seed(sim_seed, offset))
            test["valid"] = bool(math.isfinite(test["p_value"]))
        test["applicable"] = bool(
            categories >= 2 and values_.numel() > 0
            and math.isfinite(test["p_value"]))
        marginals[name] = test

    independence_specs = (
        ("key_value", keys.reshape(-1).sub(2),
         values.reshape(-1).sub(2), selectable, selectable),
        ("query_rank_answer", query_rank, answer.sub(2),
         n_pairs, selectable),
        ("query_key_answer", query.sub(2), answer.sub(2),
         selectable, selectable),
    )
    independence = {}
    for offset, (name, left, right, n_left, n_right) in enumerate(
            independence_specs, 11):
        test = _categorical_independence_test(
            left, right, n_left, n_right, n_sim=n_sim,
            seed=_generator_offset_seed(sim_seed, offset))
        test["applicable"] = bool(test["valid"])
        independence[name] = test

    sorted_values = values.sort(dim=1).values
    repeated = (sorted_values[:, 1:] == sorted_values[:, :-1]).any(dim=1)
    repeat_test = _binomial_rate_test(
        int(repeated.sum()), rows,
        _expected_value_repeat_rate(n_pairs, vocab), n_sim=n_sim,
        seed=_generator_offset_seed(sim_seed, 21))
    repeat_test["applicable"] = bool(n_pairs >= 2 and repeat_test["valid"])

    checks = {}
    checks.update({f"marginal:{name}": test
                   for name, test in marginals.items()})
    checks.update({f"independence:{name}": test
                   for name, test in independence.items()})
    checks["value_repeat_rate"] = repeat_test
    applicable = {name: test for name, test in checks.items()
                  if test.get("applicable")}
    p_values = [test["p_value"] for test in applicable.values()
                if math.isfinite(test["p_value"])]
    abstained = [name for name, test in applicable.items()
                 if not math.isfinite(test["p_value"])]
    n_tests = len(p_values)
    if not n_tests or abstained:
        per_test_alpha = float("nan")
        rejected = []
        supported = None
    else:
        per_test_alpha = _GENERATOR_FAMILY_ALPHA / n_tests
        rejected = [name for name, test in applicable.items()
                    if test["p_value"] <= per_test_alpha]
        supported = not rejected
    return {
        "n_pairs": int(n_pairs), "vocab": int(vocab),
        "n_rows": rows, "n_pair_observations": int(n_pair_obs),
        "n_selectable": int(selectable),
        "marginal_uniformity": marginals,
        "independence": independence,
        "value_repeat_rate": repeat_test,
        "family": {
            "alpha": _GENERATOR_FAMILY_ALPHA,
            "per_test_alpha": per_test_alpha,
            "n_tests": int(n_tests),
            "rejected": rejected,
            "abstained": abstained,
        },
        "calibration_supported": supported,
        "note": (
            "Model-free goodness-of-fit check of the declared generator. "
            "It complements the split audit: a defect shared by every split "
            "leaves the between-split tests green.  Rejection is a data-"
            "quality warning, not a leakage verdict; the marginal test is "
            "the same Poisson-calibrated uniformity test used elsewhere."),
    }


def generator_calibration_audit(n_pairs, vocab, device="cpu", batch=256,
                                batches=4, seed=TEST_SEED,
                                n_sim=_GENERATOR_SIM) -> dict:
    """Draw a stream and test it against the declared data-generating process.

    The split audit asks whether the held-out streams differ from training.
    This audit asks a different question: did *every* stream come from the
    task the code claims to implement?  A generator defect shared by all
    splits - values silently drawn without replacement, a key/value
    coupling, or a non-uniform query position - passes every between-split
    comparison because both sides inherit it.

    The checks are deliberately model-free and use only realized data:

    * marginal uniformity of the query rank, key ids, value ids, query key
      and supervised answer (Poisson-calibrated Pearson test);
    * mutual-information independence of key/value pairs, query rank and
      answer, and query key and answer (aligned-label permutation null);
    * the row-level value-repeat rate against the birthday probability
      implied by i.i.d. sampling with replacement.

    ``batch * batches`` samples are drawn from ``seed`` at the requested
    geometry, every batch passes :func:`validate_batch`, and the Monte
    Carlo seeds are fixed, so the report is deterministic.  A joint
    independence table above the audit budget abstains rather than
    allocating an unbounded matrix.  The family flag uses a Bonferroni
    threshold over the applicable tests and is a warning, not a
    split-leakage gate.
    """
    check_data_geometry(n_pairs, vocab)
    batch = _positive_audit_count(batch, "batch")
    batches = _positive_audit_count(batches, "batches")
    n_sim = _positive_audit_count(n_sim, "n_sim")
    total = batch * batches
    if total > _MAX_TORCH_INDEX:
        raise ValueError(
            f"batch*batches={total} exceeds torch's signed 64-bit index "
            "range")
    stream_key = stream_seed_key(seed)
    gen = torch.Generator(device=device).manual_seed(stream_key)
    rows, answers = [], []
    for _ in range(batches):
        idx, targets, meta = make_batch(
            batch, n_pairs, vocab, device, gen=gen)
        validate_batch(idx, targets, meta, vocab, diagnostics=False)
        rows.append(idx.detach().cpu())
        answers.append(meta["answer"].detach().cpu())
    out = _generator_calibration_from_idx(
        torch.cat(rows, dim=0), torch.cat(answers, dim=0), n_pairs, vocab,
        n_sim=n_sim, sim_seed=_GENERATOR_SIM_SEED)
    out.update({
        "seed": int(seed), "stream_key": stream_key,
        "batch": int(batch), "batches": int(batches),
        "n_sim": int(n_sim),
    })
    return out


def _fingerprints(idx):
    # ``tolist`` already yields Python ints; no per-token conversion needed.
    return {tuple(row) for row in idx.tolist()}


def split_leakage_audit(n_pairs, vocab, device="cpu", batch=256, batches=4,
                        train_seeds=(0,)):
    """Audit every split of the protocol for leakage and distribution shift.

    Draws samples from every configured training stream and from the
    reserved validation, test and statistics streams (never used for
    fitting), then reports - with every reference statistic fitted on the
    training streams only:

    * **exact-duplicate overlap for every pair of streams** (training seeds
      among themselves, each training stream against each held-out split,
      and the held-out splits against each other).  A shared sample
      between *any* two splits is leakage; a validation/test overlap in
      particular silently couples model selection with the final report
      even though both splits are nominally "held out".  The full
      pairwise matrix is checked, not only train-vs-held-out, because the
      protocol's claim is pairwise disjointness.
    * **per-field distribution shift**: for pooled tokens, supervised
      answers, key / value sub-populations and the structural difficulty
      covariates (key position, repeated answer values, naive-confusable
      queries) and their joint ``(key_pos, value_repeats, naive_confusable)``
      cell, the total-variation distance between each held-out split
      and the pooled training distribution is reported against a same-size
      noise floor measured between disjoint training streams
      (:func:`_noise_floor`), *and* the same-distribution null is tested
      with a Monte Carlo-calibrated Pearson chi-square
      (:func:`_distribution_shift_p`) whose null redraws both count vectors
      at their audit sizes - the TV/floor pair is descriptive, the
      calibrated p-value is what keeps a sparse field (a few hundred
      answer classes at ~1e3 samples) from hiding a real shift behind an
      unattainable tolerance.  The difficulty fields matter because equal
      marginals do not imply equal task difficulty: a held-out split with
      systematically later keys or more ambiguous values has a different
      achievable accuracy, so an accuracy contrast against it would
      attribute a data difference to the mechanism.
    * **category-support coverage**: an answer class absent from the
      training streams is unpredictable and an input token absent from them
      is unembedable; for the answer, token, key and value fields the train
      support and each held-out split's unseen-value rate are reported as a
      bound on achievable accuracy.
    * **atypical-sample / difficulty-tail audit**: a per-sample typicality
      score (the train-fitted surprisal over the answer, key position,
      repeated-value and naive-confusable fields) is thresholded at the
      training tail quantile, and the share of samples beyond that threshold
      is reported for every split.  The reference and the threshold are fit
      on the training streams only.  A held-out split whose *tail* moved can
      pass every marginal test; it is flagged under a conservative rate
      bound (:data:`_OUTLIER_RATE_MULTIPLIER`, :data:`_OUTLIER_RATE_FLOOR`).
    * **training-stream stationarity**: the same calibrated shift test run
      between the first and second halves of each training stream, so
      non-stationary training is visible before a held-out split is trusted
      to represent the future.  This uses training data only.
    * **generator calibration**: the test realization is also checked
      against the *declared* data-generating process (marginal uniformity,
      key/value and query/answer independence, and the i.i.d. value-repeat
      rate).  A generator defect shared by every split would leave the
      between-split comparisons green; the result is reported under
      ``generator_calibration`` and is a warning, not a leakage verdict.
    * **cleaning diagnostics** from :func:`validate_batch`, run on every
      batch of every split: NaN/Inf, out-of-range ids, structural
      violations and ambiguous/misaligned label noise raise; a whole-split
      exact duplicate-row count and the unique-key rate are returned, plus
      the subset of duplicates that occur *within* one batch (a stream-level
      count above the within-batch count means the same sample was drawn
      in two different batches of the same stream; overlaps *between*
      streams are the leakage signal above).

    The audit only *reads* data: it fits nothing on held-out streams and
    writes nothing.  A shared sample or a gap above the noise floor
    invalidates the HB3 decision and every number reported above it.
    """
    check_data_geometry(n_pairs, vocab)
    batch = _positive_audit_count(batch, "batch")
    batches = _positive_audit_count(batches, "batches")
    samples = batch * batches
    if samples > _MAX_TORCH_INDEX:
        raise ValueError(
            f"batch*batches={samples} exceeds torch's signed 64-bit index "
            "range")
    if samples < 2:
        raise ValueError(
            "split_leakage_audit needs at least two samples per split to "
            f"estimate a sampling-noise floor, got batch={batch} x "
            f"batches={batches}=1; the floor would be NaN and the TV/floor "
            f"check would silently pass")
    if isinstance(train_seeds, (str, bytes)) or isinstance(
            train_seeds, Mapping):
        raise ValueError(
            "train_seeds must be an iterable of integer seeds, got "
            f"{type(train_seeds).__name__}")
    try:
        train_seed_iter = iter(train_seeds)
    except TypeError as exc:
        raise ValueError(
            "train_seeds must be an iterable of integer seeds, got "
            f"{type(train_seeds).__name__}") from exc
    unique = {}
    for seed in train_seed_iter:
        unique.setdefault(stream_seed_key(seed), int(seed))
    train_seeds = tuple(unique.values())
    if not train_seeds:
        raise ValueError("train_seeds must contain at least one training seed")
    for seed in train_seeds:
        check_train_seed(seed)
    splits = {f"train:{seed}": _draw_split(n_pairs, vocab, device, batch,
                                           batches, seed)
              for seed in train_seeds}
    for name, seed in (("val", VAL_SEED), ("test", TEST_SEED),
                       ("stats", STATS_SEED)):
        splits[name] = _draw_split(n_pairs, vocab, device, batch, batches, seed)

    # Exact-duplicate overlap for every pair of streams.
    fingerprints = {name: _fingerprints(s["idx"]) for name, s in splits.items()}
    names = list(splits)
    pairwise = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairwise[f"{a}|{b}"] = len(fingerprints[a] & fingerprints[b])

    # Reference statistics are fitted on the training streams only.  The
    # structural difficulty covariates are derived once per split and the
    # per-field values / counts are reused by the floor, shift and TV
    # readouts instead of being recomputed for every threshold/test.
    train_names = [f"train:{seed}" for seed in train_seeds]
    categories = {field: _field_categories(field, vocab, n_pairs)
                  for field in _AUDIT_FIELDS}
    features = {name: _difficulty_features(split, n_pairs)
                for name, split in splits.items()}

    def values(name, field):
        if field in _AUDIT_DERIVED_FIELDS:
            return features[name][field]
        return _field_values(splits[name], n_pairs, field)

    split_values = {(name, field): values(name, field)
                    for name in names for field in _AUDIT_FIELDS}
    pooled = {field: torch.cat([split_values[(name, field)]
                                for name in train_names])
              for field in _AUDIT_FIELDS}
    pooled_counts = {field: _counts(pooled[field], categories[field])
                     for field in _AUDIT_FIELDS}
    reference = {field: _distribution_from_counts(pooled_counts[field])
                 for field in _AUDIT_FIELDS}
    floors = {
        field: _noise_floor([split_values[(name, field)]
                             for name in train_names], categories[field])
        for field in _AUDIT_FIELDS
    }
    observed_counts = {
        (name, field): _counts(split_values[(name, field)], categories[field])
        for name in ("val", "test", "stats") for field in _AUDIT_FIELDS}
    shift_p = {field: {} for field in _AUDIT_FIELDS}
    shift_chi2 = {field: {} for field in _AUDIT_FIELDS}
    for field in _AUDIT_FIELDS:
        for name in ("val", "test", "stats"):
            p_value, chi2 = _distribution_shift_p(
                observed_counts[(name, field)], pooled_counts[field])
            shift_p[field][name] = p_value
            shift_chi2[field][name] = chi2

    # Train-fitted atypical-sample threshold and train-only stationarity.
    # The reference is Haldane-smoothed so a category absent from training
    # still yields a finite surprisal instead of an infinite score.
    smoothed_reference = {
        field: (pooled_counts[field].to(torch.float64) + _AUDIT_SMOOTH)
        / (pooled_counts[field].sum()
           + _AUDIT_SMOOTH * categories[field])
        for field in _AUDIT_FIELDS}
    outliers = _outlier_audit(split_values, train_names, names,
                              smoothed_reference)
    stationarity = _train_stationarity(split_values, splits, train_names,
                                       categories)

    # Data-sufficiency disclosure: a category that never occurs in the
    # training streams is unpredictable (an answer class) or unembedable (an
    # input token), so report the train support and the share of each
    # held-out split's values outside it.  The support is fitted on train
    # counts only and held-out values are merely looked up
    # (transform-only); the rate bounds the accuracy any model can reach,
    # which the marginal shift tests do not report.
    support_coverage = _support_coverage(
        pooled_counts, split_values, ("val", "test", "stats"), categories,
        ("answer", "token", "key", "value"))

    # Descriptive difficulty profile per split: the calibrated tests decide
    # *whether* a shift exists, these shares say *what* differs (mean pair
    # index of the matching key, repeated-answer rate, naive-confusable
    # rate), all computed without any held-out-fitted statistic.
    difficulty = {}
    for name, split in splits.items():
        feat = features[name]
        n = max(int(split["idx"].size(0)), 1)
        difficulty[name] = {
            "mean_key_pos": float(feat["key_pos"].to(torch.float64)
                                  .mean()) if split["idx"].size(0) else
            float("nan"),
            "value_repeat_rate": float(feat["value_repeats"].sum()) / n,
            "naive_confusable_rate": (
                float(feat["naive_confusable"].sum()) / n),
        }

    out = {
        "train_seeds": list(train_seeds),
        "val_seed": VAL_SEED,
        "test_seed": TEST_SEED,
        "stats_seed": STATS_SEED,
        "n_per_split": int(batch) * int(batches),
        "audit_fields": list(_AUDIT_FIELDS),
        "audit_field_categories": categories,
        "tv_noise_floor": floors,
        "token_tv_noise_floor": floors["token"],
        "shift_p_value": shift_p,
        "shift_chi2": shift_chi2,
        "shift_test": {"n_sim": _AUDIT_SIM, "seed": _AUDIT_SIM_SEED,
                       "smoothing": _AUDIT_SMOOTH},
        "difficulty_by_split": difficulty,
        "outlier_audit": outliers,
        "support_coverage": support_coverage,
        "train_stationarity": stationarity,
        "train_stationarity_min_p": min(
            (v["p"] for v in stationarity.values()), default=1.0),
        "pairwise_duplicate_overlap": pairwise,
        "max_pairwise_duplicate_overlap": max(pairwise.values(), default=0),
    }
    per_split_dup = {name: sum(d["n_duplicate_rows"] for d in s["diagnostics"])
                     for name, s in splits.items()}
    stream_dup = {name: _count_duplicate_rows(s["idx"])
                  for name, s in splits.items()}
    out["n_duplicate_rows_train"] = sum(stream_dup[n] for n in train_names)
    out["n_duplicate_rows_within_batch_train"] = sum(
        per_split_dup[n] for n in train_names)
    out["unique_key_rate"] = min(
        d["unique_key_rate"] for s in splits.values()
        for d in s["diagnostics"])
    # A held-out row can match any training stream, not just the first
    # seed's stream; count the overlap against the union so a multi-seed
    # protocol cannot hide leakage behind seed order.
    train_fingerprints = set()
    for train_name in train_names:
        train_fingerprints.update(fingerprints[train_name])
    for name in ("val", "test", "stats"):
        out[f"duplicate_overlap_train_{name}"] = len(
            fingerprints[name] & train_fingerprints)
        out[f"n_duplicate_rows_{name}"] = stream_dup[name]
        out[f"n_duplicate_rows_within_batch_{name}"] = per_split_dup[name]
        for field in _AUDIT_FIELDS:
            out[f"{field}_tv_train_{name}"] = _total_variation(
                reference[field],
                _distribution_from_counts(observed_counts[(name, field)]))
    # Model-free check of the declared DGP, fitted on the same held-out
    # test realization the split audit already drew.  This is intentionally
    # *not* folded into ``hb3_supported``: a calibration rejection is a
    # data-quality warning, while HB3 is the leakage gate.
    calibration = _generator_calibration_from_idx(
        splits["test"]["idx"], splits["test"]["answer"], n_pairs, vocab)
    calibration["seed"] = int(TEST_SEED)
    calibration["stream"] = "test"
    out["generator_calibration"] = calibration
    out["generator_calibration_supported"] = (
        calibration["calibration_supported"])
    return out


