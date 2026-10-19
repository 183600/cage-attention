"""Synthetic task data.

Associative-recall style LM: sequence of key-value pairs plus a query; the
model must recall the value. This makes attention quality (and false
concentration) measurable: we know the *correct* target token. Length
generalisation (E2) evaluates this same task at multiple sequence lengths
via ``length_sweep_lengths``.

Format (vocab 0..V-1):
    [k1 v1 k2 v2 ... kn vn, SEP, qi, <predict vi>]
The label for the position right after qi is vi.
"""

from __future__ import annotations

import hashlib
import math

import torch

SEP_ID = 1
PAD_ID = 0

# ---------------------------------------------------------------------------
# Train / validation / test split protocol
# ---------------------------------------------------------------------------
# A sample belongs to exactly one split, decided *before* it is drawn: every
# split owns a fixed RNG stream and batches are generated only from their own
# split's stream.  Nothing is generated into a common pool that is later
# shuffled or partitioned, and no statistic is ever fitted on validation/test
# data.  The reserved seeds below identify the held-out streams; training
# entry points refuse them via ``check_train_seed`` - training with e.g.
# ``seed=2024`` would draw exactly the batches used for model selection
# (VAL_SEED), the final reported numbers (TEST_SEED) or mechanism statistics
# (STATS_SEED), i.e. train on the evaluation data itself.
#
# The synthetic task is i.i.d.: there is no temporal order and no
# per-entity/group structure, so a seed-owned stream *is* the split and the
# split-before-shuffle requirement reduces to "assign the split before
# generating".  If the generator is ever replaced by a dataset with a time
# axis or with per-entity samples, the same rule applies at a higher level:
# split contiguous time blocks (past -> train, latest -> test, no shuffling
# before the split) and partition whole entity ids before drawing, so no
# entity ever appears in two splits.  The audit below covers both halves of
# that contract: exact overlap for every stream pair, and a calibrated
# same-distribution test per audit field; any fitted preprocessing must be
# fit on the training stream(s) only and applied transform-only to
# val/test; :func:`split_leakage_audit` follows that rule itself: every
# reference distribution and noise floor it uses is fitted on the training
# streams and only *scored* against held-out data.
VAL_SEED = 123      # model/recipe selection only (never reported)
TEST_SEED = 2024    # sealed: final reported numbers + fixed gate rules
STATS_SEED = 7      # held-out mechanism statistics (collect_stats default)
RESERVED_EVAL_SEEDS = (VAL_SEED, TEST_SEED, STATS_SEED)

# Sealing semantics, stated precisely because the short label "sealed" has
# been read as "never touched again".  Sealed means **no model,
# hyperparameter or recipe choice may read the stream**: all selection
# (learning-rate tuning, recipe freezing, baseline choice, early stopping)
# stays on VAL_SEED.  It does *not* mean the stream is evaluated once and
# only once at the very end.  The pre-registered rules fixed in the code -
# HB1's rule ceiling, HB4's label-permuted negative control and the
# HB0/HB6/HB7 competence / saturation marks - are read on the TEST stream
# and gate the grids; those rules are not tunable, and the autonomous CI
# loop re-runs ``--only baselines`` after a code change, so their verdicts
# are *re-measured* (the same fixed rule, applied again), not re-selected.
# Because no decision in that loop can see or react to the individual
# numbers (the rules and the grid are pre-registered and the gate is
# pass/fail), re-measurement cannot turn TEST into an implicit validation
# split; the bias bound is stated in the baselines verdict.


# Tiny-batch overfit sanity check (``cage.train.overfit_sanity_check``): the
# single batch is generated from this stream and the model is *meant* to
# memorise it, so it is not an evaluation split and is deliberately not in
# ``RESERVED_EVAL_SEEDS`` (no reported metric ever reads it).
SANITY_SEED = 31337


def check_train_seed(seed) -> None:
    """Refuse to train on a held-out evaluation stream.

    The data is drawn on the fly from a generator seeded by ``seed``, so
    reusing a reserved evaluation seed makes training consume exactly the
    batches that belong to a held-out split.
    """
    if int(seed) in RESERVED_EVAL_SEEDS:
        raise ValueError(
            f"seed {seed} is reserved for held-out evaluation "
            f"{RESERVED_EVAL_SEEDS}; training on it would fit the model on "
            f"the validation/test data itself. Choose a different seed."
        )


def check_eval_seed(seed, selection: bool = False) -> None:
    """Refuse to read a sealed evaluation stream while selecting a model.

    ``selection=True`` marks a call whose numbers inform a decision
    (learning-rate / recipe tuning, baseline selection, early stopping):
    such a call may read the validation stream but never the sealed test
    *or statistics* stream.  Evaluating candidates on a sealed stream and
    then keeping the winner turns it into an implicit validation set - the
    reported number is then optimistically biased for every future model
    that is picked the same way.  TEST_SEED is legitimate for the final
    reported numbers and for the fixed, pre-registered gate rules (never
    for a choice); STATS_SEED only for the mechanism-statistics readout.
    """
    if selection and int(seed) in (TEST_SEED, STATS_SEED):
        raise ValueError(
            f"stream seed {seed} is sealed (TEST_SEED={TEST_SEED} or "
            f"STATS_SEED={STATS_SEED}): it may be read only for the final "
            f"reported evaluation / pre-registered gate rules / mechanism "
            f"statistics, never for model selection. Use VAL_SEED="
            f"{VAL_SEED} for selection."
        )


def _sample_keys(batch: int, n_pairs: int, vocab: int, gen, device) -> torch.Tensor:
    """Distinct keys per sequence (sampling without replacement).

    Values may still repeat freely (a shared value never makes the target
    ambiguous, only a shared key does).  That makes ``n_pairs`` distinct keys a
    hard requirement of the task, not a nicety: a key that occurs twice gives
    the query two different values to choose from, so the "correct answer" is
    no longer well defined and the accuracy ceiling drops for reasons that have
    nothing to do with the variant under test.  ``vocab - 2`` keys are
    available (0/1 are reserved), so a request beyond that is rejected rather
    than silently downgraded to sampling with replacement.
    """
    n_avail = vocab - 2
    if n_pairs > n_avail:
        raise ValueError(
            f"n_pairs={n_pairs} needs {n_pairs} distinct keys per sequence but "
            f"vocab={vocab} only offers {n_avail} (ids 0=PAD and 1=SEP are "
            f"reserved); use vocab >= {n_pairs + 2} or n_pairs <= {n_avail}. "
            f"Sampling keys with replacement instead would make the recall "
            f"target ambiguous and cap the measured accuracy."
        )
    noise = torch.rand(batch, n_avail, generator=gen, device=device)
    return noise.topk(n_pairs, dim=1).indices.add_(2)


def make_batch(batch: int, n_pairs: int, vocab: int, device, dense_lm: bool = False, gen: torch.Generator | None = None):
    """Returns idx, targets, eval_meta.

    eval_meta: dict with positions needed for false-concentration analysis:
      query_pos  (B,) index of qi in the sequence
      key_pos    (B,) index of the matching ki
      answer     (B,) correct value token
    """
    width = 2 * n_pairs
    seq = torch.empty((batch, width + 3), device=device, dtype=torch.long)
    kv = seq[:, :width].view(batch, n_pairs, 2)
    kv[..., 0] = _sample_keys(batch, n_pairs, vocab, gen, device)
    kv[..., 1] = torch.randint(2, vocab, (batch, n_pairs), generator=gen, device=device)
    qi_idx = torch.randint(0, n_pairs, (batch,), generator=gen, device=device)
    b = torch.arange(batch, device=device)
    query = kv[b, qi_idx, 0]
    answer = kv[b, qi_idx, 1]
    sep = seq[:, width]
    sep.fill_(SEP_ID)
    seq[:, -2].copy_(query)
    seq[:, -1].copy_(answer)
    idx = seq[:, :-1]
    tgt_full = seq[:, 1:]
    key_pos = 2 * qi_idx
    query_pos = torch.full_like(qi_idx, width + 1)
    if dense_lm:
        targets = tgt_full
    else:
        targets = torch.full_like(tgt_full, -1)
        targets[b, query_pos] = answer
    meta = {"query_pos": query_pos, "key_pos": key_pos, "answer": answer}
    return idx, targets, meta


def _count_duplicate_rows(idx) -> int:
    """Exact duplicate rows in ``idx`` (content comparison, no fitting).

    Used both for the per-batch cleaning diagnostics and for the
    whole-stream duplicate count in :func:`split_leakage_audit`: counting
    duplicates is a deterministic set operation, so it is safe on any
    split and needs no train-fitted statistics.  ``torch.unique(dim=0)``
    keeps this cheap enough to run on every training/evaluation batch.
    """
    if idx.numel() == 0:
        return 0
    unique_rows = torch.unique(idx.detach(), dim=0).size(0)
    return int(idx.size(0)) - int(unique_rows)


def validate_batch(idx, targets, meta, vocab) -> dict:
    """Data-cleaning gate for one freshly generated batch.

    Raises ``ValueError`` if any invariant is violated:

    * integrity: 2-D integer ids, no NaN/Inf, all token ids in ``[0, vocab)``;
    * structure: exactly one SEP token per sequence and a valid query
      position, with ``meta`` agreeing with the sequence;
    * label noise: the query key occurs exactly once among the keys, the
      recorded answer is the value paired with that key (a repeated key
      would make the correct answer ambiguous) and the answer id is a
      selectable token (ids 0=PAD and 1=SEP are reserved and never labels);
    * targets (when given): ids in ``[-1, vocab)``, the supervised entry
      at the query position equal to the answer, and *every* entry aligned
      with its position (``-1`` or the true next-token label).  Checking
      more than the single supervised entry catches off-by-one /
      foreign-label target misalignment: a label silently shifted onto the
      wrong position passes the one-spot check whenever the shift happens
      not to touch the query position, but fails the alignment check.

    Returns a small diagnostics dict (sample count, unique-key rate, number
    of exact duplicate rows inside the batch) used by the split audit.
    """
    if not _batch_invariants_hold(idx, targets, meta, vocab):
        _validate_batch_strict(idx, targets, meta, vocab)
    return {"n": int(idx.size(0)), "unique_key_rate": 1.0,
            "n_duplicate_rows": _count_duplicate_rows(idx)}


def _batch_invariants_hold(idx, targets, meta, vocab) -> bool:
    """``validate_batch``'s invariants as one device-side predicate.

    The same conditions as the strict checker below, accumulated on the
    device; anything that cannot be evaluated (non-tensor inputs,
    unexpected meta entries, wrong shapes, ...) reports ``False`` and is
    left to the strict checker, which raises.  The predicate never
    *accepts* a batch the strict checker would reject: every condition of
    the strict checker is mirrored here, shape-level conditions as host
    checks and value-level ones as device checks.
    """
    try:
        if not torch.is_tensor(idx) or idx.dim() != 2:
            return False
        batch_size, width = idx.shape
        checks = [(idx >= 0).all(), (idx < vocab).all()]
        if idx.is_floating_point():
            checks.append(torch.isfinite(idx).all())
        qp = meta["query_pos"].to(device=idx.device, dtype=torch.long)
        if qp.shape != (batch_size,):
            return False
        checks.append((qp >= 0).all())
        checks.append((qp < width).all())
        # Clamped only to keep the gathers below in range: the clamped
        # positions are simultaneously rejected by the range checks above,
        # so an invalid batch can never slip through on the clamped index.
        safe_qp = qp.unsqueeze(1).clamp(0, max(width - 1, 0))
        query = idx.gather(1, safe_qp).squeeze(1)
        answer = meta["answer"].to(device=idx.device, dtype=torch.long)
        answer_ok = answer.shape == (batch_size,)
        checks.append((answer >= 2).all())
        checks.append((answer < vocab).all())
        # Key columns only: ``idx`` is ``[k1 v1 ... kn vn SEP query]``, so
        # the last two columns (SEP at 2n, query at 2n+1) must be excluded.
        keys = idx[:, :-2:2]
        match = keys.eq(query.unsqueeze(1))
        checks.append(match.sum(dim=1).eq(1).all())
        key_pos = match.to(torch.int64).argmax(dim=1).mul(2)
        paired_value = idx.gather(1, (key_pos + 1).unsqueeze(1)).squeeze(1)
        checks.append(paired_value.eq(answer).all())
        checks.append(idx.eq(SEP_ID).sum(dim=1).eq(1).all())
        if "key_pos" in meta:
            kp = meta["key_pos"].to(device=idx.device, dtype=torch.long)
            if kp.shape != (batch_size,):
                return False
            checks.append(kp.eq(key_pos).all())
        if targets is not None:
            if targets.shape != idx.shape:
                return False
            if targets.is_floating_point():
                checks.append(torch.isfinite(targets).all())
            checks.append(((targets >= -1) & (targets < vocab)).all())
            sup = targets.gather(1, safe_qp).squeeze(1)
            checks.append(sup.eq(answer).all())
            # Full label alignment, not only the supervised entry: every
            # position must be unsupervised (``-1``) or carry the true
            # next-token label of that position.
            nxt = targets[:, :-1]
            cont = idx[:, 1:]
            checks.append((nxt.eq(-1) | nxt.eq(cont)).all())
            checks.append((targets[:, -1].eq(-1)
                           | targets[:, -1].eq(answer)).all())
        acc = checks[0]
        for c in checks[1:]:
            acc = acc & c
        return answer_ok and bool(acc)
    except Exception:  # noqa: BLE001 - any unevaluable input is strict's job
        return False


def _validate_batch_strict(idx, targets, meta, vocab) -> dict:
    """``validate_batch``'s checks, one host read-back per condition."""
    if not torch.is_tensor(idx) or idx.dim() != 2:
        raise ValueError(f"idx must be a (B, T) tensor, got {type(idx)}")
    if idx.is_floating_point() and not bool(torch.isfinite(idx).all()):
        raise ValueError("idx contains NaN/Inf")
    if not bool((idx >= 0).all()) or not bool((idx < vocab).all()):
        raise ValueError(f"idx contains token ids outside [0, {vocab})")
    batch_size, width = idx.shape
    qp = meta["query_pos"].to(device=idx.device, dtype=torch.long)
    if qp.shape != (batch_size,) or bool(qp.lt(0).any()) or bool(qp.ge(width).any()):
        raise ValueError("query_pos is missing or out of range")
    query = idx.gather(1, qp.unsqueeze(1)).squeeze(1)
    answer = meta["answer"].to(device=idx.device, dtype=torch.long)
    if answer.shape != (batch_size,):
        raise ValueError("meta['answer'] has the wrong shape")
    if not bool((answer >= 2).all()) or not bool((answer < vocab).all()):
        raise ValueError(
            "label noise: answer ids must be selectable tokens in "
            f"[2, {vocab}) (0=PAD and 1=SEP are reserved and never labels)"
        )
    # Key columns only: ``idx`` is ``[k1 v1 ... kn vn SEP query]``, so the
    # last two columns (SEP at 2n, query at 2n+1) must be excluded - a
    # plain ``0::2`` stride would pick up SEP and silently rely on
    # ``query >= 2`` never matching the reserved SEP id.
    keys = idx[:, :-2:2]
    match = keys.eq(query.unsqueeze(1))
    if not bool(match.sum(dim=1).eq(1).all()):
        raise ValueError(
            "label noise: the query key is not unique among the keys, so "
            "the correct answer is ambiguous"
        )
    key_pos = match.to(torch.int64).argmax(dim=1).mul(2)
    paired_value = idx.gather(1, (key_pos + 1).unsqueeze(1)).squeeze(1)
    if not bool(paired_value.eq(answer).all()):
        raise ValueError("label noise: answer != value paired with the query key")
    if not bool(idx.eq(SEP_ID).sum(dim=1).eq(1).all()):
        raise ValueError("structure: expected exactly one SEP token per row")
    if "key_pos" in meta:
        kp = meta["key_pos"].to(device=idx.device, dtype=torch.long)
        if kp.shape != (batch_size,):
            raise ValueError("meta['key_pos'] has the wrong shape")
        if not bool(kp.eq(key_pos).all()):
            raise ValueError("meta['key_pos'] disagrees with the sequence")
    if targets is not None:
        if targets.shape != idx.shape:
            raise ValueError("targets must have the same shape as idx")
        if targets.is_floating_point() and not bool(torch.isfinite(targets).all()):
            raise ValueError("targets contain NaN/Inf")
        if not bool(((targets >= -1) & (targets < vocab)).all()):
            raise ValueError(f"targets contain ids outside [-1, {vocab})")
        sup = targets.gather(1, qp.unsqueeze(1)).squeeze(1)
        if not bool(sup.eq(answer).all()):
            raise ValueError("supervised target at query_pos != answer")
        nxt = targets[:, :-1]
        cont = idx[:, 1:]
        if not bool((nxt.eq(-1) | nxt.eq(cont)).all()):
            raise ValueError(
                "label alignment: a target entry is neither -1 nor the "
                "true next-token continuation of its position"
            )
        if not bool((targets[:, -1].eq(-1)
                     | targets[:, -1].eq(answer)).all()):
            raise ValueError(
                "label alignment: the final target entry is neither -1 nor "
                "the sample's answer"
            )
    return {"n": int(idx.size(0)), "unique_key_rate": 1.0,
            "n_duplicate_rows": _count_duplicate_rows(idx)}


def batch_fingerprint(idx, tgt, meta) -> str:
    """Content hash of one batch (context, targets, answers).

    One hashing convention shared by ``stream_fingerprint`` and the
    tiny-batch overfit sanity check, so "which data did this run draw or
    memorise" always reduces to the same digest.
    """
    h = hashlib.sha256()
    for tensor in (idx, tgt, meta["answer"]):
        h.update(repr(tensor.detach().cpu().tolist()).encode("ascii"))
    return h.hexdigest()[:16]


def stream_fingerprint(n_pairs, vocab, device="cpu", batch=8, seed=TEST_SEED):
    """Content hash of the first batch of a data stream (dataset versioning).

    Two runs that agree on the fingerprint drew identical data; a changed
    fingerprint means the generation protocol changed and numbers from
    before are not comparable.  Hashing one batch is the smallest
    deterministic summary of a stream that catches every protocol change
    (key/value draw order, SEP/query layout, reserved-id handling), and it
    is recorded with every experiment result next to the code revision.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    idx, tgt, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
    return {"seed": int(seed), "batch": int(batch), "n_pairs": int(n_pairs),
            "vocab": int(vocab), "sha256": batch_fingerprint(idx, tgt, meta)}


def _draw_split(n_pairs, vocab, device, batch, batches, seed):
    """Draw one split's samples and run the cleaning gate on every batch.

    Returns the concatenated contexts, the supervised answers and the
    per-batch ``validate_batch`` diagnostics.  Nothing is fitted here; the
    caller decides which split plays the reference role.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    rows, answers, diagnostics = [], [], []
    for _ in range(batches):
        idx, targets, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        diagnostics.append(validate_batch(idx, targets, meta, vocab))
        rows.append(idx.detach().cpu())
        answers.append(meta["answer"].detach().cpu())
    return {"idx": torch.cat(rows, dim=0),
            "answer": torch.cat(answers, dim=0),
            "diagnostics": diagnostics}


# Audit fields: pooled tokens, the supervised answer distribution (the
# quantity the accuracy / loss metrics are computed on) and the key / value
# sub-populations.  All four are categorical token ids, so a distribution
# shift shows up directly as a total-variation distance.
_AUDIT_FIELDS = ("token", "answer", "key", "value")

# Calibration of the distribution-shift check.  ``_AUDIT_SIM`` Monte Carlo
# replicates set the p-value resolution to ``1 / (_AUDIT_SIM + 1)`` (500
# replicates resolve the 0.0042 rank-one threshold of a 12-test
# Benjamini-Hochberg family at q=0.05); the fixed seed keeps the audit
# deterministic; ``_AUDIT_SMOOTH`` is the Haldane-Anscombe add-half count
# that keeps unobserved classes testable instead of producing an infinite
# statistic.
_AUDIT_SIM = 500
_AUDIT_SIM_SEED = 90210
_AUDIT_SMOOTH = 0.5


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
    raise ValueError(f"unknown audit field {field!r}")


def _counts(values, vocab):
    return torch.bincount(values.reshape(-1).to(torch.long).cpu(),
                          minlength=vocab).to(torch.int64)


def _distribution(values, vocab):
    counts = _counts(values, vocab).to(torch.float64)
    return counts / counts.sum().clamp_min(1.0)


def _total_variation(p, q):
    return float(0.5 * (p - q).abs().sum())


def _noise_floor(stream_values, vocab):
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
    if len(stream_values) >= 2:
        floors = []
        for j, values in enumerate(stream_values):
            others = [v for i, v in enumerate(stream_values) if i != j]
            reference = _distribution(torch.cat(others), vocab)
            floors.append(
                _total_variation(_distribution(values, vocab), reference))
        return sum(floors) / len(floors)
    values = stream_values[0]
    if values.numel() < 2:
        return float("nan")
    half = values.numel() // 2
    floor = _total_variation(_distribution(values[:half], vocab),
                             _distribution(values[half:], vocab))
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
    obs = observed.detach().cpu().to(torch.float64)
    ref = reference.detach().cpu().to(torch.float64)
    n_ref = float(ref.sum())
    n_h = float(obs.sum())
    if n_ref <= 0.0 or n_h <= 0.0:
        return float("nan"), float("nan")
    k = ref.numel()
    probs = (ref + _AUDIT_SMOOTH) / (n_ref + _AUDIT_SMOOTH * k)
    expected = n_h * probs
    chi2 = float((((obs - expected) ** 2) / expected).sum())

    gen = torch.Generator().manual_seed(seed)
    ref_sim = torch.poisson((n_ref * probs).expand(n_sim, k).contiguous(),
                            generator=gen)
    h_sim = torch.poisson((n_h * probs).expand(n_sim, k).contiguous(),
                          generator=gen)
    probs_sim = (ref_sim + _AUDIT_SMOOTH) / (
        ref_sim.sum(1, keepdim=True) + _AUDIT_SMOOTH * k)
    expected_sim = h_sim.sum(1, keepdim=True) * probs_sim
    chi2_sim = (((h_sim - expected_sim) ** 2) / expected_sim).sum(1)
    p_value = float((1.0 + (chi2_sim >= chi2).to(torch.float64).sum())
                    / (n_sim + 1.0))
    return p_value, chi2


# Calibration of the answer-uniformity test (``answer_uniformity_test``):
# the sample size enters the test, so only the simulation count and its
# fixed seed are constants here.
_UNIFORMITY_SIM = 500
_UNIFORMITY_SIM_SEED = 8675309


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
    c = torch.as_tensor(counts).detach().reshape(-1).to(torch.float64)
    n = float(c.sum())
    k = c.numel()
    out = {"n": int(n), "n_classes": int(k),
           "expected_per_class": float(n / k) if k else float("nan"),
           "chi2": float("nan"), "df": int(max(k - 1, 0)),
           "p_value": float("nan"), "uniform_supported": False}
    if k < 2 or n <= 0.0:
        return out
    expected = n / k
    chi2 = float((((c - expected) ** 2) / expected).sum())
    gen = torch.Generator().manual_seed(seed)
    sim = torch.poisson(torch.full((n_sim, k), expected), generator=gen)
    chi2_sim = (((sim - expected) ** 2) / expected).sum(1)
    p_value = float((1.0 + (chi2_sim >= chi2).to(torch.float64).sum())
                    / (n_sim + 1.0))
    out.update({"chi2": chi2, "p_value": p_value,
                "uniform_supported": bool(p_value > 0.05)})
    return out


def _fingerprints(idx):
    return {tuple(int(t) for t in row) for row in idx.tolist()}


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
      answers and key / value sub-populations, the total-variation distance
      between each held-out split and the pooled training distribution is
      reported against a same-size noise floor measured between disjoint
      training streams (:func:`_noise_floor`), *and* the same-distribution
      null is tested with a Monte Carlo-calibrated Pearson chi-square
      (:func:`_distribution_shift_p`) whose null redraws both count vectors
      at their audit sizes - the TV/floor pair is descriptive, the
      calibrated p-value is what keeps a sparse field (a few hundred
      answer classes at ~1e3 samples) from hiding a real shift behind an
      unattainable tolerance.
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
    train_seeds = tuple(dict.fromkeys(int(s) for s in train_seeds))
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

    # Reference statistics are fitted on the training streams only.
    train_names = [f"train:{seed}" for seed in train_seeds]
    pooled = {field: torch.cat([_field_values(splits[name], n_pairs, field)
                                for name in train_names])
              for field in _AUDIT_FIELDS}
    pooled_counts = {field: _counts(pooled[field], vocab)
                     for field in _AUDIT_FIELDS}
    reference = {field: _distribution(pooled[field], vocab)
                 for field in _AUDIT_FIELDS}
    floors = {
        field: _noise_floor([_field_values(splits[name], n_pairs, field)
                             for name in train_names], vocab)
        for field in _AUDIT_FIELDS
    }
    shift_p = {field: {} for field in _AUDIT_FIELDS}
    shift_chi2 = {field: {} for field in _AUDIT_FIELDS}
    for field in _AUDIT_FIELDS:
        for name in ("val", "test", "stats"):
            p_value, chi2 = _distribution_shift_p(
                _counts(_field_values(splits[name], n_pairs, field), vocab),
                pooled_counts[field])
            shift_p[field][name] = p_value
            shift_chi2[field][name] = chi2

    out = {
        "train_seeds": list(train_seeds),
        "val_seed": VAL_SEED,
        "test_seed": TEST_SEED,
        "stats_seed": STATS_SEED,
        "n_per_split": int(batch) * int(batches),
        "audit_fields": list(_AUDIT_FIELDS),
        "tv_noise_floor": floors,
        "token_tv_noise_floor": floors["token"],
        "shift_p_value": shift_p,
        "shift_chi2": shift_chi2,
        "shift_test": {"n_sim": _AUDIT_SIM, "seed": _AUDIT_SIM_SEED,
                       "smoothing": _AUDIT_SMOOTH},
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
    for name in ("val", "test", "stats"):
        out[f"duplicate_overlap_train_{name}"] = pairwise[
            f"{train_names[0]}|{name}"]
        out[f"n_duplicate_rows_{name}"] = stream_dup[name]
        out[f"n_duplicate_rows_within_batch_{name}"] = per_split_dup[name]
        for field in _AUDIT_FIELDS:
            out[f"{field}_tv_train_{name}"] = _total_variation(
                reference[field],
                _distribution(_field_values(splits[name], n_pairs, field),
                              vocab))
    return out


def length_sweep_lengths(base: int = 16, factor: int = 2, n: int = 4,
                         max_len: int = 256):
    """``n`` geometrically spaced ``n_pairs`` values for the length sweep.

    A length ``L`` occupies ``2*L + 2`` positions (pairs + SEP + query),
    so anything that does not fit into ``max_len`` positions cannot
    be run at all and is dropped instead of producing an out-of-range
    position embedding at evaluation time.
    """
    lengths = [base * (factor ** i) for i in range(n)]
    fits = [L for L in lengths if 2 * L + 2 <= max_len]
    if not fits:
        raise ValueError(
            f"no sweep length fits max_len={max_len}: needs >= {2 * base + 2} "
            f"positions for base={base}"
        )
    return fits


def _self_test() -> int:
    """Deterministic self-checks of the split-audit machinery.

    Pins the two properties the distribution-shift decision rests on:
    a same-source held-out draw must not be flagged (calibration), and a
    held-out split drawn from half the class range must be (power) even
    though its total-variation gap stays below the historical
    ``3 x noise floor`` tolerance - the blindness that motivated the
    Monte Carlo-calibrated chi-square.  Also pins the duplicate-row
    accounting used by the split audit, including duplicates that only
    appear when two batches are concatenated.  Run with
    ``python3 -m cage.data``.
    """
    failures = []

    def check(name: str, cond: bool) -> None:
        if not cond:
            failures.append(name)

    vocab, classes = 512, 510
    n_h, n_ref = 1024, 5120

    def draw_counts(k, n, seed):
        gen = torch.Generator().manual_seed(seed)
        return torch.bincount(torch.randint(0, k, (n,), generator=gen),
                              minlength=vocab).to(torch.int64)

    def _tv_between_counts(a, b):
        pa = a.to(torch.float64) / a.sum()
        pb = b.to(torch.float64) / b.sum()
        return 0.5 * float((pa - pb).abs().sum())

    reference = draw_counts(classes, n_ref, 11)
    same_source = draw_counts(classes, n_h, 12)
    p_same, _ = _distribution_shift_p(same_source, reference, n_sim=400,
                                      seed=90210)
    check("same_source_not_flagged", p_same > 0.05)

    half_range = draw_counts(classes // 2, n_h, 13)
    p_shift, chi2_shift = _distribution_shift_p(half_range, reference,
                                                n_sim=400, seed=90210)
    check("half_range_shift_detected", p_shift <= 1.0 / 401.0)
    check("shift_chi2_large", chi2_shift > 900.0)

    floor = _tv_between_counts(same_source, draw_counts(classes, n_h, 14))
    tv_shift = _tv_between_counts(half_range, reference)
    check("old_tv_tolerance_would_pass", tv_shift < 3.0 * floor)

    # Answer-uniformity decision: the balance claim behind "accuracy is the
    # headline metric" must survive a uniform draw and reject a
    # half-range draw at the default audit sample size (5 seeds x 4
    # batches x 256 ~ 5120 samples, ~10 per class).
    n_audit = 5 * 4 * 256
    uni_counts = draw_counts(classes, n_audit, 31)[:classes]
    uni = answer_uniformity_test(uni_counts, n_sim=400, seed=90210)
    check("uniform_draw_supported", uni["uniform_supported"])
    check("uniform_chi2_finite", math.isfinite(uni["chi2"]))
    skewed_counts = draw_counts(classes // 2, n_audit, 32)[:classes]
    skewed = answer_uniformity_test(skewed_counts, n_sim=400, seed=90210)
    check("skewed_draw_rejected", not skewed["uniform_supported"])
    check("skewed_p_below_floor", skewed["p_value"] <= 1.0 / 401.0)
    degenerate = answer_uniformity_test(torch.zeros(1, dtype=torch.int64))
    check("degenerate_input_not_supported",
          not degenerate["uniform_supported"]
          and math.isnan(degenerate["p_value"]))

    # Sealed-stream guard: selection may read the validation stream only.
    # Both the test and the statistics streams are held out, so using
    # either for a choice would turn it into an implicit validation set;
    # reading them *without* selection (final report / pre-registered gate
    # rules / mechanism statistics) stays allowed.
    check_eval_seed(VAL_SEED, selection=True)
    for bad in (TEST_SEED, STATS_SEED):
        try:
            check_eval_seed(bad, selection=True)
            check(f"selection_refused_on_stream_{bad}", False)
        except ValueError:
            check(f"selection_refused_on_stream_{bad}", True)
        check_eval_seed(bad)

    # Stream-level duplicate accounting: rows repeated across *different*
    # batches of one stream must be counted, not only within one batch.
    gen = torch.Generator().manual_seed(21)
    batch_a, _, _ = make_batch(8, 4, 64, "cpu", gen=gen)
    batch_b, _, _ = make_batch(8, 4, 64, "cpu", gen=gen)
    check("clean_batch_has_no_duplicates",
          _count_duplicate_rows(batch_a) == 0)
    check("stream_duplicates_counted",
          _count_duplicate_rows(torch.cat([batch_a, batch_a[:3], batch_b])) == 3)

    # Label-alignment and reserved-id gates: freshly drawn sparse and dense
    # batches pass the cleaning gate; a target that is neither ``-1`` nor
    # the true continuation is rejected (off-by-one / foreign-label
    # misalignment), and a reserved answer id (PAD/SEP) is rejected even
    # when the paired value and the supervised entry agree with it.
    gen = torch.Generator().manual_seed(22)
    sparse_idx, sparse_tgt, sparse_meta = make_batch(8, 4, 64, "cpu", gen=gen)
    check("validate_accepts_sparse_targets",
          validate_batch(sparse_idx, sparse_tgt, sparse_meta, 64)["n"] == 8)
    dense_idx, dense_tgt, dense_meta = make_batch(
        8, 4, 64, "cpu", dense_lm=True, gen=torch.Generator().manual_seed(23))
    check("validate_accepts_dense_targets",
          validate_batch(dense_idx, dense_tgt, dense_meta, 64)["n"] == 8)

    misaligned = sparse_tgt.clone()
    misaligned[:, 0] = (sparse_idx[:, 1] + 1) % 64
    try:
        validate_batch(sparse_idx, misaligned, sparse_meta, 64)
        check("validate_rejects_misaligned_target", False)
    except ValueError:
        check("validate_rejects_misaligned_target", True)

    rows = torch.arange(8)
    reserved_idx = sparse_idx.clone()
    reserved_idx[rows, sparse_meta["key_pos"] + 1] = 0
    reserved_meta = {"query_pos": sparse_meta["query_pos"],
                     "key_pos": sparse_meta["key_pos"],
                     "answer": torch.zeros(8, dtype=torch.long)}
    reserved_tgt = torch.full_like(sparse_tgt, -1)
    reserved_tgt[rows, reserved_meta["query_pos"]] = 0
    try:
        validate_batch(reserved_idx, reserved_tgt, reserved_meta, 64)
        check("validate_rejects_reserved_answer", False)
    except ValueError:
        check("validate_rejects_reserved_answer", True)

    # The predicate that guards every training batch must agree with the
    # strict checker exactly: it may not accept a batch the strict checker
    # rejects (that would silently disable the cleaning gate), and it may
    # not reject a valid one.
    def gate_agrees(i_, t_, m_, vocab_):
        held = _batch_invariants_hold(i_, t_, m_, vocab_)
        try:
            _validate_batch_strict(i_, t_, m_, vocab_)
            strict = True
        except Exception:  # noqa: BLE001 - any raise is a rejection
            strict = False
        return held == strict

    nan_idx = sparse_idx.clone().float()
    nan_idx[0, 0] = float("nan")
    bad_qp = sparse_meta["query_pos"] + 1
    meta_qp = {"query_pos": bad_qp, "key_pos": sparse_meta["key_pos"],
               "answer": sparse_meta["answer"]}
    meta_kp = {"query_pos": sparse_meta["query_pos"],
               "key_pos": sparse_meta["key_pos"] + 2,
               "answer": sparse_meta["answer"]}
    soft_tgt = sparse_tgt.clone()
    soft_tgt[0, 0] = -2
    final_tgt = sparse_tgt.clone()
    final_tgt[:, -1] = sparse_meta["answer"] + 1
    dup_key = sparse_idx.clone()
    dup_key[:, 2] = sparse_idx[:, 0]
    for name, (i_, t_, m_) in {
            "valid_sparse": (sparse_idx, sparse_tgt, sparse_meta),
            "valid_dense": (dense_idx, dense_tgt, dense_meta),
            "misaligned_target": (sparse_idx, misaligned, sparse_meta),
            "reserved_answer": (reserved_idx, reserved_tgt, reserved_meta),
            "idx_out_of_range": (sparse_idx.clone().fill_(64), sparse_tgt,
                                 sparse_meta),
            "idx_not_2d": (sparse_idx.reshape(-1), sparse_tgt, sparse_meta),
            "idx_with_nan": (nan_idx, sparse_tgt, sparse_meta),
            "query_pos_out_of_range": (sparse_idx, sparse_tgt, meta_qp),
            "key_pos_disagrees": (sparse_idx, sparse_tgt, meta_kp),
            "key_not_unique": (dup_key, sparse_tgt, sparse_meta),
            "target_shape_wrong": (sparse_idx, sparse_tgt.reshape(-1),
                                   sparse_meta),
            "target_out_of_range": (sparse_idx, soft_tgt, sparse_meta),
            "final_target_wrong": (sparse_idx, final_tgt, sparse_meta),
    }.items():
        check(f"invariants_agree_{name}", gate_agrees(i_, t_, m_, 64))

    if failures:
        print("data self-test FAILED: " + ", ".join(failures))
        return 1
    print(f"data self-test: all checks passed "
          f"(same-source p={p_same:.3f}, half-range p={p_shift:.4f}, "
          f"chi2={chi2_shift:.1f}, tv={tv_shift:.3f} vs 3x floor="
          f"{3.0 * floor:.3f}; uniformity p={uni['p_value']:.3f} "
          f"chi2={uni['chi2']:.1f}, skewed p={skewed['p_value']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
