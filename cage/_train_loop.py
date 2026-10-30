# cage train: overfit sanity, stats collection and attention diagnostics.
# Executed into the wrapper module namespace by the loader;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _label_audit(idx, meta):
    """Per-sample label-integrity flags on the generated context.

    Answers "is the annotation itself wrong?" *on the very batches where
    errors were found* instead of assuming it: a label is valid when the
    query key occurs exactly once among the keys and the recorded answer
    is the value paired with it.  Mirrors the ``validate_batch``
    invariants but returns a boolean per sample so failures can be
    counted and attached to individual bad cases (``validate_batch``
    raises on the first bad batch, which is the right behaviour for a
    training gate and the wrong one for an error audit).
    """
    query = idx.gather(1, meta["query_pos"].unsqueeze(1)).squeeze(1)
    # Key columns only: drop the trailing SEP and query columns (see
    # ``cage.data.validate_batch``), so the SEP id can never enter the
    # key-uniqueness / answer-pairing judgment.
    keys = idx[:, :-2:2]
    match = keys.eq(query.unsqueeze(1))
    key_pos = match.to(torch.int64).argmax(dim=1).mul(2)
    paired = idx.gather(1, (key_pos + 1).unsqueeze(1)).squeeze(1)
    return match.sum(dim=1).eq(1) & paired.eq(meta["answer"])


# Mutually exclusive bad-case buckets, in priority order (see
# ``collect_stats_and_errors``).  Kept in one place so the per-type
# accumulators, the summary keys and the report tables cannot drift.
_ERROR_TYPES = ("label_error", "hard_structural", "retrieval_miss",
                "found_then_lost", "readout_wrong")


@torch.inference_mode()
def collect_stats_and_errors(model, n_pairs, vocab, device, batch=128,
                             batches=4, seed=STATS_SEED, max_cases=None):
    """One pass over the diagnostic stream producing both the per-layer
    mechanism statistics and the bad-case mining record.

    The layer statistics are read off the same forward that feeds the
    mining hooks.  ``max_cases=None`` collects statistics only (no hooks,
    no case taxonomy) and returns ``None`` for the error record.

    Statistics: per-layer gate / entropy / maxprob / null-mass at query
    positions, plus false-concentration labels (high max-prob but wrong
    answer).  Every batch passes the ``validate_batch`` cleaning gate, so
    the mechanism statistics are computed on the same validated data
    contract as training and the reported accuracy metrics.

    Error mining: each error is recorded with the evidence needed to
    classify *why* it failed:

    * ``confidence`` / ``margin``   - confidently wrong (systematic blind
      spot) vs low-confidence (ambiguous for the model);
    * ``attn_mass_on_key``          - mean attention mass on the correct
      key position in the last block: did the model fail to *find* the
      right key, or did it find it and then *read out* the wrong value?
    * ``attn_mass_on_key_by_layer`` - the same mass at *every* block, so
      a last-layer miss can be told apart from a key that was found in an
      intermediate block and lost again (the fixed readout composes
      match-then-shift, so a mid-stack hit that does not survive to the
      last block is a distinct failure mode);
    * ``pred_in_context``           - prediction is one of the visible
      values (retrieval confusion) vs an out-of-context token (readout
      failure);
    * ``naive_confusable``          - the query token appears as a later
      value, so a copy-last heuristic would fail (task-structural
      ambiguity rather than a model blind spot);
    * ``value_repeats``             - the answer value occurs more than
      once in the context (harder even for a perfect retriever);
    * ``label_ok`` / ``error_type`` - whether the annotation of *this*
      sample is valid (``_label_audit``) and a mutually-exclusive bucket:
      ``label_error`` > ``hard_structural`` > ``retrieval_miss``
      (attention mass < 0.5 on the correct key in *every* block) >
      ``found_then_lost`` (mass >= 0.5 in some block but < 0.5 in the
      last one) > ``readout_wrong`` (key still found in the last block,
      wrong value read out).  ``hard_structural`` is additionally split
      into disjoint naive-only / repeat-only / both buckets so the two
      structural sources can be read apart without changing the primary
      taxonomy;
    * ``key_pair_idx``              - position of the matching pair;
      aggregated into early/mid/late error rates (long-range retrieval).

    The label audit makes "labels cannot be the cause" a *measured* claim:
    ``label_noise_rate`` is computed over every evaluated sample rather
    than assumed to be 0 from the construction.  Every batch additionally
    passes the full ``validate_batch`` cleaning gate, so a structurally
    invalid batch aborts the analysis instead of being taxonomised as a
    model error.

    The taxonomy is quantified, not just tabulated: each bucket carries
    its sample count and a Wilson interval over the error population
    (``taxonomy_<type>_count/_lo/_hi``), and the mean answer confidence
    and top-2 margin *within* each bucket (``conf_by_error_type`` /
    ``margin_by_error_type``).  A high-confidence ``retrieval_miss`` is a
    systematic blind spot; a low-confidence ``hard_structural`` error is
    task ambiguity; a confident ``label_error`` means the annotation is
    wrong rather than the model.  The overall ``error_rate`` and
    ``label_noise_rate`` also carry Wilson intervals, so small
    differences between variants are not read off binomial noise.

    Returns ``(stats, errors)``; cases are capped at ``max_cases``, retaining
    the highest-confidence errors across the whole diagnostic stream rather
    than the first errors encountered, while the summary aggregates every
    evaluated sample.  ``stats`` matches the ``collect_stats`` return value
    exactly.
    """
    raw_batch, raw_batches = batch, batches
    batch = _positive_integer(raw_batch)
    batches = _positive_integer(raw_batches)
    if batch is None:
        raise ValueError(f"batch must be a positive integer, got {raw_batch!r}")
    if batches is None:
        raise ValueError(
            f"batches must be a positive integer, got {raw_batches!r}")
    if max_cases is not None:
        raw_cases = max_cases
        if isinstance(raw_cases, bool):
            raise ValueError(
                f"max_cases must be a non-negative integer or None, got "
                f"{raw_cases!r}")
        try:
            max_cases = operator.index(raw_cases)
        except TypeError as exc:
            raise ValueError(
                f"max_cases must be a non-negative integer or None, got "
                f"{raw_cases!r}") from exc
        if max_cases < 0:
            raise ValueError(
                f"max_cases must be a non-negative integer or None, got "
                f"{raw_cases!r}")
    cfg = model.cfg
    C, H = cfg.d_model, cfg.n_heads
    dk = C // H
    inv_scale = dk ** -0.5
    gen = torch.Generator(device=device).manual_seed(stream_seed_key(seed))
    training_states = {module: module.training for module in model.modules()}
    model.eval()

    with_errors = max_cases is not None

    # Per-batch mechanism statistics (``collect_stats``).
    stats_out = {"gate": [], "entropy": [], "maxprob": [], "null_mass": [],
                 "correct": [], "resid_norm": []}

    per_layer_mass: dict[int, torch.Tensor] = {}
    state: dict = {"meta": None}
    handles = []

    def _make_hook(layer_idx, null_k):
        def _hook(module, inputs, output):
            meta = state["meta"]
            if meta is not None:
                qk = getattr(module, "_qk_capture", None)
                if qk is not None:
                    per_layer_mass[layer_idx] = _attention_mass_from_qk(
                        qk[0], qk[1], meta, H, inv_scale, null_k)
                else:
                    per_layer_mass[layer_idx] = _attention_mass_on_key(
                        inputs[0], module.qkv.weight, meta, H, inv_scale,
                        null_k)
            module._qk_capture = None
        return _hook

    if with_errors:
        for i, blk in enumerate(model.blocks):
            blk.attn._capture_qk = True
            handles.append(blk.attn.register_forward_hook(
                _make_hook(i, getattr(blk.attn, "null_k", None))))

    # Device-side accumulators for the per-batch reductions.
    n_total = 0
    cases = []
    case_order = []
    i64 = torch.int64
    flags_summary = {k: torch.zeros((), dtype=i64, device=device)
                     for k in ("high", "low", "in_ctx", "naive", "repeat")}
    taxonomy = {k: torch.zeros((), dtype=i64, device=device)
                for k in _ERROR_TYPES}
    structural_split = {
        k: torch.zeros((), dtype=i64, device=device)
        for k in ("naive_only", "repeat_only", "both")}
    # Per-error-type confidence / margin sums: the taxonomy alone says
    # *what* failed; the confidence attached to each bucket says whether
    # the failure is a confident blind spot (high confidence on a wrong
    # answer) or an ambiguous / label-noisy sample the model was unsure
    # about - the distinction that drives the next improvement.
    type_conf_sum = {k: torch.zeros((), dtype=torch.float64, device=device)
                     for k in _ERROR_TYPES}
    type_margin_sum = {k: torch.zeros((), dtype=torch.float64, device=device)
                       for k in _ERROR_TYPES}
    type_count = {k: torch.zeros((), dtype=i64, device=device)
                  for k in _ERROR_TYPES}
    n_err = torch.zeros((), dtype=i64, device=device)
    n_label_err = torch.zeros((), dtype=i64, device=device)
    mass_err = torch.zeros((), dtype=torch.float64, device=device)
    mass_err_n = torch.zeros((), dtype=i64, device=device)
    mass_corr = torch.zeros((), dtype=torch.float64, device=device)
    mass_corr_n = torch.zeros((), dtype=i64, device=device)
    layer_mass_err = None   # per-layer sums over error samples (float64)
    layer_mass_corr = None
    n_layers_seen = 0
    tertile_n = torch.zeros(3, dtype=i64, device=device)
    tertile_err = torch.zeros(3, dtype=i64, device=device)
    key_wrong_total = torch.zeros((), dtype=torch.float64, device=device)
    key_wrong_n = torch.zeros((), dtype=i64, device=device)
    key_right_total = torch.zeros((), dtype=torch.float64, device=device)
    key_right_n = torch.zeros((), dtype=i64, device=device)
    tertile_denom = max(1, n_pairs)
    try:
        for i in range(batches):
            idx, tgt, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
            validate_batch(idx, tgt, meta, vocab, diagnostics=False)
            per_layer_mass.clear()
            state["meta"] = meta
            logits, _, layer_stats = model(idx, stat_pos=meta["query_pos"])
            state["meta"] = None

            pred_stats = logits.squeeze(1).argmax(-1)
            stats_out["correct"].append((pred_stats == meta["answer"]).float())
            ents, maxps, nulls, gates, rns = [], [], [], [], []
            for s in layer_stats:
                ents.append(s["entropy"].mean(1))
                maxps.append(s["maxprob"].max(1)[0])
                rns.append(s["resid_norm"])
                if s["gate"] is not None:
                    gates.append(s["gate"].mean(-1))
                if s["null_mass"] is not None:
                    nulls.append(s["null_mass"].mean(1))
            stats_out["entropy"].append(torch.stack(ents, 1))
            stats_out["maxprob"].append(torch.stack(maxps, 1))
            stats_out["resid_norm"].append(torch.stack(rns, 1))
            if gates:
                stats_out["gate"].append(torch.stack(gates, 1))
            if nulls:
                stats_out["null_mass"].append(torch.stack(nulls, 1))
            if not with_errors:
                continue

            layered = torch.stack(
                [per_layer_mass[i] for i in sorted(per_layer_mass)], dim=0)
            mass_last, mass_mean = layered[-1], layered.mean(0)
            if layer_mass_err is None:
                layer_mass_err = torch.zeros(
                    layered.size(0), dtype=torch.float64, device=device)
                layer_mass_corr = torch.zeros_like(layer_mass_err)
                n_layers_seen = layered.size(0)
            probs = logits.squeeze(1).softmax(-1)
            conf, pred = probs.max(-1)
            top2 = probs.topk(2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            answer = meta["answer"]
            wrong = pred != answer

            sample_offset = n_total
            n_total += idx.size(0)
            n_err += wrong.sum()
            label_ok = _label_audit(idx, meta)
            n_label_err += (~label_ok).sum()
            key_pair_idx = (meta["key_pos"] // 2).clamp(max=n_pairs - 1)
            third = (key_pair_idx * 3 // tertile_denom).clamp_(max=2).to(i64)
            tertile_n += torch.bincount(third, minlength=3)
            tertile_err += torch.bincount(third[wrong], minlength=3)
            m_e = mass_last[wrong]
            m_c = mass_last[~wrong]
            mass_err += m_e.sum().double()
            mass_err_n += m_e.numel()
            mass_corr += m_c.sum().double()
            mass_corr_n += m_c.numel()
            layer_mass_err += layered[:, wrong].sum(dim=1).double()
            layer_mass_corr += layered[:, ~wrong].sum(dim=1).double()

            flags = _error_case_flags(idx, meta, pred, n_pairs)
            flags_summary["high"] += (wrong & (conf >= 0.9)).sum()
            flags_summary["low"] += (wrong & (conf < 0.5)).sum()
            flags_summary["in_ctx"] += (wrong & flags["pred_in_context"]).sum()
            flags_summary["naive"] += (wrong & flags["naive_confusable"]).sum()
            flags_summary["repeat"] += (wrong & flags["value_repeats"]).sum()
            key_right = mass_last >= 0.5
            # A key that clears the mass threshold in *some* block but not
            # in the last one was found mid-stack and lost again; calling
            # that a plain retrieval miss would hide where retrieval broke.
            found_any = (layered >= 0.5).any(dim=0)
            key_wrong_total += mass_last[wrong & ~key_right].sum().double()
            key_wrong_n += (wrong & ~key_right).sum()
            key_right_total += mass_last[wrong & key_right].sum().double()
            key_right_n += (wrong & key_right).sum()

            ambiguous = flags["naive_confusable"] | flags["value_repeats"]
            key_missed = ~key_right
            lost = key_missed & found_any
            never_found = key_missed & ~found_any
            masks = {
                "label_error": wrong & ~label_ok,
                "hard_structural": wrong & label_ok & ambiguous,
                "retrieval_miss": wrong & label_ok & ~ambiguous & never_found,
                "found_then_lost": wrong & label_ok & ~ambiguous & lost,
                "readout_wrong": wrong & label_ok & ~ambiguous & ~key_missed,
            }
            for etype, mask in masks.items():
                taxonomy[etype] += mask.sum()
                type_count[etype] += mask.sum()
                type_conf_sum[etype] += conf[mask].sum().double()
                type_margin_sum[etype] += margin[mask].sum().double()

            naive_flag = flags["naive_confusable"]
            repeat_flag = flags["value_repeats"]
            structural = wrong & label_ok
            split_masks = {
                "naive_only": structural & naive_flag & ~repeat_flag,
                "repeat_only": structural & ~naive_flag & repeat_flag,
                "both": structural & naive_flag & repeat_flag,
            }
            for key, mask in split_masks.items():
                structural_split[key] += mask.sum()

            if max_cases > 0:
                rows = torch.nonzero(wrong, as_tuple=False).flatten()
                if rows.numel():
                    if rows.numel() > max_cases:
                        order = torch.argsort(
                            conf.index_select(0, rows), descending=True,
                            stable=True)
                        rows = rows.index_select(0, order[:max_cases])
                    rows_l = rows.tolist()
                    # Pull the per-sample vectors to host for the case
                    # loop below.
                    conf_l = conf.tolist()
                    margin_l = margin.tolist()
                    mass_last_l = mass_last.tolist()
                    mass_mean_l = mass_mean.tolist()
                    mass_layers_l = layered.transpose(0, 1).tolist()
                    pred_l = pred.tolist()
                    answer_l = answer.tolist()
                    kp_l = key_pair_idx.tolist()
                    label_ok_l = label_ok.tolist()
                    ambiguous_l = ambiguous.tolist()
                    key_missed_l = key_missed.tolist()
                    never_found_l = never_found.tolist()
                    pic_l = flags["pred_in_context"].tolist()
                    nc_l = flags["naive_confusable"].tolist()
                    vr_l = flags["value_repeats"].tolist()
                    query_pos_l = meta["query_pos"].tolist()
                    key_pos_l = meta["key_pos"].tolist()
                    for b in rows_l:
                        if not label_ok_l[b]:
                            etype = "label_error"
                        elif ambiguous_l[b]:
                            etype = "hard_structural"
                        elif key_missed_l[b]:
                            etype = ("retrieval_miss" if never_found_l[b]
                                     else "found_then_lost")
                        else:
                            etype = "readout_wrong"
                        context = [int(t) for t in idx[b].tolist()]
                        cases.append({
                            "context": context,
                            "query_pos": int(query_pos_l[b]),
                            "query_key": context[query_pos_l[b]],
                            "key_pos": int(key_pos_l[b]),
                            "key_pair_idx": int(kp_l[b]),
                            "answer": int(answer_l[b]),
                            "pred": int(pred_l[b]),
                            "confidence": float(conf_l[b]),
                            "margin": float(margin_l[b]),
                            "attn_mass_on_key": float(mass_last_l[b]),
                            "attn_mass_on_key_mean_layers": float(mass_mean_l[b]),
                            "attn_mass_on_key_by_layer": [
                                float(v) for v in mass_layers_l[b]],
                            "pred_in_context": bool(pic_l[b]),
                            "naive_confusable": bool(nc_l[b]),
                            "value_repeats": bool(vr_l[b]),
                            "label_ok": bool(label_ok_l[b]),
                            "error_type": etype,
                        })
                        case_order.append(sample_offset + int(b))
                    if len(cases) > max_cases:
                        ranked = sorted(
                            zip(cases, case_order),
                            key=lambda item: (
                                -float(item[0]["confidence"]), item[1]))
                        cases = [case for case, _ in ranked[:max_cases]]
                        case_order = [order for _, order in
                                      ranked[:max_cases]]
    finally:
        for hd in handles:
            hd.remove()
        for blk in model.blocks:
            blk.attn._capture_qk = False
            blk.attn._qk_capture = None
        for module, training in training_states.items():
            module.train(training)

    stats = {k: torch.cat(v).cpu() for k, v in stats_out.items() if v}
    release_eval_scratch()
    if not with_errors:
        return stats, None
    if cases:
        ranked = sorted(
            zip(cases, case_order),
            key=lambda item: (-float(item[0]["confidence"]), item[1]))
        cases = [case for case, _ in ranked]

    n_err = int(n_err)
    n_label_err = int(n_label_err)
    flags_summary = {k: int(v) for k, v in flags_summary.items()}
    taxonomy = {k: int(v) for k, v in taxonomy.items()}
    type_count = {k: int(v) for k, v in type_count.items()}
    type_conf_mean = {
        k: (float(type_conf_sum[k]) / type_count[k]) if type_count[k]
        else float("nan") for k in _ERROR_TYPES}
    type_margin_mean = {
        k: (float(type_margin_sum[k]) / type_count[k]) if type_count[k]
        else float("nan") for k in _ERROR_TYPES}
    err_ci = wilson_ci(n_err, n_total)
    label_ci = wilson_ci(n_label_err, n_total)
    mass_err, mass_err_n = float(mass_err), int(mass_err_n)
    mass_corr, mass_corr_n = float(mass_corr), int(mass_corr_n)
    tertile_n = [int(v) for v in tertile_n.tolist()]
    tertile_err = [int(v) for v in tertile_err.tolist()]
    key_wrong_total, key_wrong_n = float(key_wrong_total), int(key_wrong_n)
    key_right_total, key_right_n = float(key_right_total), int(key_right_n)
    if layer_mass_err is not None:
        layer_mass_err = [float(v) for v in layer_mass_err.tolist()]
        layer_mass_corr = [float(v) for v in layer_mass_corr.tolist()]

    def _share(count):
        return count / n_err if n_err else float("nan")

    summary = {
        "n_samples": int(n_total),
        "n_errors": int(n_err),
        "error_rate": (n_err / n_total) if n_total else float("nan"),
        "error_rate_lo": err_ci["lo"],
        "error_rate_hi": err_ci["hi"],
        "n_label_errors": int(n_label_err),
        "label_noise_rate": (n_label_err / n_total) if n_total else float("nan"),
        "label_noise_rate_lo": label_ci["lo"],
        "label_noise_rate_hi": label_ci["hi"],
        "conf_by_error_type": dict(type_conf_mean),
        "margin_by_error_type": dict(type_margin_mean),
        "taxonomy_label_error": _share(taxonomy["label_error"]),
        "taxonomy_hard_structural": _share(taxonomy["hard_structural"]),
        "taxonomy_retrieval_miss": _share(taxonomy["retrieval_miss"]),
        "taxonomy_found_then_lost": _share(taxonomy["found_then_lost"]),
        "taxonomy_readout_wrong": _share(taxonomy["readout_wrong"]),
        "share_high_conf": _share(flags_summary["high"]),
        "share_low_conf": _share(flags_summary["low"]),
        "share_pred_in_context": _share(flags_summary["in_ctx"]),
        "share_naive_confusable": _share(flags_summary["naive"]),
        "share_value_repeats": _share(flags_summary["repeat"]),
        "attn_mass_on_key_error": (mass_err / mass_err_n) if mass_err_n else float("nan"),
        "attn_mass_on_key_correct": (mass_corr / mass_corr_n) if mass_corr_n else float("nan"),
        "error_rate_key_early": (tertile_err[0] / tertile_n[0]) if tertile_n[0] else float("nan"),
        "error_rate_key_mid": (tertile_err[1] / tertile_n[1]) if tertile_n[1] else float("nan"),
        "error_rate_key_late": (tertile_err[2] / tertile_n[2]) if tertile_n[2] else float("nan"),
        "errors_key_missed_share": ((key_wrong_n) / n_err) if n_err else float("nan"),
        "attn_mass_on_key_error_key_hit": (
            key_right_total / key_right_n) if key_right_n else float("nan"),
        "attn_mass_on_key_error_key_missed": (
            key_wrong_total / key_wrong_n) if key_wrong_n else float("nan"),
    }
    for etype in _ERROR_TYPES:
        ci = wilson_ci(taxonomy[etype], n_err)
        summary[f"taxonomy_{etype}_count"] = int(taxonomy[etype])
        summary[f"taxonomy_{etype}_lo"] = ci["lo"]
        summary[f"taxonomy_{etype}_hi"] = ci["hi"]
    for key, count in structural_split.items():
        count = int(count)
        ci = wilson_ci(count, n_err)
        summary[f"taxonomy_hard_structural_{key}"] = _share(count)
        summary[f"taxonomy_hard_structural_{key}_count"] = count
        summary[f"taxonomy_hard_structural_{key}_lo"] = ci["lo"]
        summary[f"taxonomy_hard_structural_{key}_hi"] = ci["hi"]
    summary["attn_mass_gap"] = summary["attn_mass_on_key_correct"] - \
        summary["attn_mass_on_key_error"]
    if layer_mass_err is not None:
        summary["attn_mass_on_key_error_by_layer"] = [
            (v / mass_err_n) if mass_err_n else float("nan")
            for v in layer_mass_err]
        summary["attn_mass_on_key_correct_by_layer"] = [
            (v / mass_corr_n) if mass_corr_n else float("nan")
            for v in layer_mass_corr]
        summary["attn_mass_on_key_layer_gap"] = [
            c - e for c, e in zip(summary["attn_mass_on_key_correct_by_layer"],
                                  summary["attn_mass_on_key_error_by_layer"])]
        summary["n_layers_seen"] = int(n_layers_seen)
    return stats, {"cases": cases, "summary": summary}


@torch.inference_mode()
def collect_error_cases(model, n_pairs, vocab, device, batch=128, batches=4,
                        seed=STATS_SEED, max_cases=64):
    """Bad-case mining: dump and taxonomise the samples the model gets wrong.

    Single-purpose wrapper around ``collect_stats_and_errors``; returns
    ``{"cases": [...], "summary": {...}}`` with cases capped at
    ``max_cases``, keeping the highest-confidence errors across the stream,
    while the summary aggregates every evaluated sample.
    """
    _, errors = collect_stats_and_errors(
        model, n_pairs, vocab, device, batch=batch, batches=batches, seed=seed,
        max_cases=max_cases)
    return errors


def _attention_mass_on_key(x, weight, meta, n_heads, inv_scale, null_k):
    """Attention mass the query puts on the *correct* key, per sample.

    Read-only: the attention input ``x`` and the shared ``qkv`` weight are
    taken from a pre-forward hook and q/k are recomputed with the *same*
    ``F.linear`` call the fixed kernel uses, so the values are identical
    to the real attention while the implementation stays untouched.  For
    ``null`` gates the learnable null slot is included in the softmax
    exactly as the kernel does, keeping the mass comparable across
    variants.
    """
    from .model import _arange  # local: pinned helper, no cycle

    B, T, _ = x.shape
    C = weight.size(0) // 3
    dk = C // n_heads
    qp = meta["query_pos"]
    kp = meta["key_pos"]
    rows = _arange(B, x.device)
    q_all = F.linear(x[rows, qp], weight[:C]).view(B, n_heads, dk)
    k_all = F.linear(x, weight[C:2 * C]).view(B, T, n_heads, dk)
    k_all = k_all.transpose(1, 2)
    # Batched query-key GEMM in head-major layout (no ``bth`` transpose).
    scores = torch.matmul(q_all.unsqueeze(2),
                          k_all.transpose(-2, -1)).mul_(inv_scale)
    cols = _arange(T, x.device).view(1, 1, 1, T)
    scores.masked_fill_(cols > qp.view(B, 1, 1, 1), float("-inf"))
    if null_k is not None:
        s0 = torch.einsum("bhd,hd->bh", q_all, null_k).mul_(inv_scale)
        scores = torch.cat([s0.view(B, n_heads, 1, 1), scores], dim=-1)
        tapped = torch.softmax(scores, dim=-1).gather(
            3, (kp + 1).view(B, 1, 1, 1).expand(B, n_heads, 1, 1))
    else:
        tapped = torch.softmax(scores, dim=-1).gather(
            3, kp.view(B, 1, 1, 1).expand(B, n_heads, 1, 1))
    return tapped.mean(dim=(1, 2, 3))


def _attention_mass_from_qk(q, k, meta, n_heads, inv_scale, null_k):
    """Attention mass on the correct key from already-projected ``q`` / ``k``.

    ``q`` / ``k`` are the ``(B*h, T, dk)`` tensors the fixed kernel just used
    (``Attention._qk_capture``).  The mass is computed exactly as in
    :func:`_attention_mass_on_key`.
    """
    from .model import _arange  # local: pinned helper, no cycle

    bh, T, dk = k.shape
    B = q.size(0) // n_heads
    h = n_heads
    qp = meta["query_pos"]
    kp = meta["key_pos"]
    rows = _arange(B, q.device)
    q_all = q.view(B, h, T, dk)[rows, :, qp]
    # Match the pre-hook's head-major key layout exactly so the diagnostics
    # stay consistent with the recomputed-key path.
    k_all = k.view(B * h, T, dk)
    scores = torch.bmm(q_all.reshape(B * h, 1, dk),
                       k_all.transpose(1, 2)).view(B, h, 1, T).mul_(inv_scale)
    cols = _arange(T, q.device).view(1, 1, 1, T)
    scores.masked_fill_(cols > qp.view(B, 1, 1, 1), float("-inf"))
    if null_k is not None:
        s0 = torch.einsum("bhd,hd->bh", q_all, null_k).mul_(inv_scale)
        scores = torch.cat([s0.view(B, h, 1, 1), scores], dim=-1)
        tapped = torch.softmax(scores, dim=-1).gather(
            3, (kp + 1).view(B, 1, 1, 1).expand(B, h, 1, 1))
    else:
        tapped = torch.softmax(scores, dim=-1).gather(
            3, kp.view(B, 1, 1, 1).expand(B, h, 1, 1))
    return tapped.mean(dim=(1, 2, 3))


def _error_case_flags(idx, meta, pred, n_pairs):
    """Per-sample structural flags used to classify an error's cause."""
    from .model import _arange  # local: pinned helper, no cycle

    T = idx.size(1)
    pos = _arange(T, idx.device).view(1, -1)
    visible = pos < meta["query_pos"].unsqueeze(1)
    query = idx.gather(1, meta["query_pos"].unsqueeze(1)).squeeze(1)
    answer = meta["answer"]
    vals = idx[:, 1:2 * n_pairs:2]
    match = (idx == query.unsqueeze(1)) & visible
    last = torch.where(match, pos, torch.full_like(pos, -1)).max(dim=1).values
    return {
        "naive_confusable": last != meta["key_pos"],
        "value_repeats": (vals == answer.unsqueeze(1)).sum(dim=1) > 1,
        "pred_in_context": (vals == pred.unsqueeze(1)).any(dim=1),
    }


def _linked_positive_int(value, name):
    """Strict positive integer for the linked-length geometry.

    ``make_batch`` rejects bool/fractional geometry rather than truncating
    it, and this probe constructs/validates batches itself, so it must not
    be looser than the generator it sits on.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        number = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be an integer, got {value!r}") from exc
    if number < 1:
        raise ValueError(f"{name} must be >= 1, got {number!r}")
    return int(number)


@torch.inference_mode()
def eval_linked_lengths(model, n_pairs, vocab, device, lengths, batch=256,
                        batches=4, seed=VAL_SEED, link_seed=271828):
    """Within-item length extension: the paired counterfactual of E2.

    The E2 sweep evaluates *independently generated* batches at every
    ``n_pairs``.  Base-length and long-length accuracy therefore come from
    different items, so the observed drop mixes the length effect with a
    different item/difficulty/retrieval-position composition.
    Post-stratification narrows
    that gap, but only this probe holds the item fixed: every base sample
    is extended in place with fresh distractor pairs
    (:func:`cage.data.make_length_linked_batch`), keeping its query key,
    answer and pair index unchanged.  Each longer sequence is a nested
    extension of the previous one, so the item is literally the same and
    only the amount of context changes.

    For each length the result reports the accuracy, the **retention**
    ``P(correct at L | correct at base)``, the **rescue**
    ``P(correct at L | wrong at base)``, the paired net change and the
    exact McNemar test on the discordant items (``b`` = base wrong / long
    right, ``c`` = base right / long wrong).  A raw E2 drop that also
    appears here is a length effect; a raw drop that disappears under
    linked extension is composition.  Because the target pair index is
    held fixed, this is an added-context / dilution effect, not an estimate
    of the natural target-position distribution.  The probe is a companion:
    it does not replace the pre-registered raw-sweep decision.

    ``lengths`` must start at ``n_pairs``, be strictly increasing and
    contain at least two lengths.  The base stream is drawn with the same
    seed/geometry as :func:`eval_metrics`, so ``base_accuracy`` is directly
    comparable to the ordinary E2 base row.  Extra distractors are drawn
    from a length-specific deterministic stream derived from ``link_seed``;
    the same ``(seed, link_seed, geometry)`` therefore reproduces the exact
    linked items.
    """
    n_pairs = _linked_positive_int(n_pairs, "n_pairs")
    batch = _linked_positive_int(batch, "batch")
    batches = _linked_positive_int(batches, "batches")
    # ``link_seed`` feeds the per-length extension generator, so it is a
    # data-stream seed like ``seed``.  Validate it before the arithmetic
    # below: a bool would otherwise be silently read as 0/1, while None or a
    # string would leak an incidental TypeError from ``link_seed + ...``
    # instead of the documented ValueError.
    stream_seed_key(link_seed)
    if isinstance(lengths, (str, bytes)):
        raise ValueError(
            f"lengths must be an iterable of integers, got "
            f"{type(lengths).__name__}")
    try:
        raw_lengths = list(lengths)
    except TypeError as exc:
        raise ValueError(
            f"lengths must be an iterable of integers, got "
            f"{type(lengths).__name__}") from exc
    parsed = []
    for value in raw_lengths:
        if isinstance(value, bool):
            raise ValueError(
                f"lengths must be integers, got {value!r}")
        try:
            length = operator.index(value)
        except TypeError as exc:
            raise ValueError(
                f"lengths must be integers, got {value!r}") from exc
        if length < 1:
            raise ValueError(f"lengths must be >= 1, got {length!r}")
        parsed.append(int(length))
    if len(parsed) < 2:
        raise ValueError(
            "eval_linked_lengths needs the base length and at least one "
            f"longer length, got {parsed!r}")
    if len(set(parsed)) != len(parsed) or any(
            right <= left for left, right in zip(parsed, parsed[1:])):
        raise ValueError(
            f"lengths must be strictly increasing, got {parsed!r}")
    if parsed[0] != n_pairs:
        raise ValueError(
            f"lengths must start at the base n_pairs={n_pairs}, got "
            f"{parsed!r}")

    training_states = {module: module.training for module in model.modules()}
    try:
        model.eval()
        gen = torch.Generator(device=device).manual_seed(
            stream_seed_key(seed))
        correct = {length: [] for length in parsed}
        for i in range(batches):
            idx, tgt, meta = make_batch(batch, n_pairs, vocab, device,
                                        gen=gen)
            validate_batch(idx, tgt, meta, vocab, diagnostics=False)
            cur_idx, cur_meta, cur_pairs = idx, meta, n_pairs
            for length in parsed:
                if length != cur_pairs:
                    link_gen = torch.Generator(device=device).manual_seed(
                        stream_seed_key(link_seed + 1000003 * length + i))
                    cur_idx, _, cur_meta = make_length_linked_batch(
                        cur_idx, cur_meta, cur_pairs, length, vocab,
                        gen=link_gen)
                    cur_pairs = length
                logits, _, _ = model(cur_idx)
                pred = logits.squeeze(1).argmax(-1)
                correct[length].append(
                    pred.eq(cur_meta["answer"]).detach().cpu())
        cat = {}
        for length in parsed:
            parts = correct[length]
            cat[length] = (torch.cat(parts) if parts else
                           torch.empty(0, dtype=torch.bool))
        base_correct = cat[n_pairs]
        base_n = int(base_correct.numel())
        base_acc = (float(base_correct.to(torch.float64).mean())
                    if base_n else float("nan"))
        n_base_correct = int(base_correct.sum())
        n_base_wrong = int(base_n - n_base_correct)
        out = {
            "base_seed": int(seed),
            "link_seed": int(link_seed),
            "n_pairs": int(n_pairs),
            "lengths": [int(L) for L in parsed],
            "batch": int(batch),
            "batches": int(batches),
            "n_items": int(base_n),
            "base_accuracy": base_acc,
            "base_n_correct": n_base_correct,
            "base_n_wrong": n_base_wrong,
            "nested_extension": True,
            "link_formula": (
                "stream_seed_key(link_seed + 1000003 * length + batch_index)"
            ),
            "accuracy_by_length": {},
            "n_by_length": {},
            "retention_by_length": {},
            "rescue_by_length": {},
            "net_change_by_length": {},
            "mcnemar_by_length": {},
            "note": (
                "within-item nested length extension: each item keeps its "
                "query key/answer/pair index and only gains distractor pairs, "
                "so the paired change isolates length from item/position "
                "composition. "
                "Companion to the raw independent-stream E2 sweep, not a "
                "replacement for its pre-registered decision."),
        }
        for length in parsed:
            rows = cat[length]
            acc = (float(rows.to(torch.float64).mean())
                   if rows.numel() else float("nan"))
            retained = int((base_correct & rows).sum())
            rescued = int((~base_correct & rows).sum())
            retention = (retained / n_base_correct
                         if n_base_correct else None)
            rescue = (rescued / n_base_wrong if n_base_wrong else None)
            mc = mcnemar_test(base_correct, rows)
            out["accuracy_by_length"][length] = acc
            out["n_by_length"][length] = int(rows.numel())
            out["retention_by_length"][length] = retention
            out["rescue_by_length"][length] = rescue
            out["net_change_by_length"][length] = (
                acc - base_acc if math.isfinite(acc)
                and math.isfinite(base_acc) else None)
            out["mcnemar_by_length"][length] = {
                "b": int(mc["b"]), "c": int(mc["c"]),
                "p_exact": float(mc["p_exact"]),
            }
        return out
    finally:
        for module, training in training_states.items():
            module.train(training)
        release_eval_scratch()


def encode_correct_mask(correct: torch.Tensor) -> str:
    """Compact base64 bitmask of per-item correctness.

    The evaluation stream is finite and every training seed is scored on
    the *same* examples.  A compact per-item mask lets a downstream paired
    contrast run the exact item-level test (McNemar) instead of treating
    the shared stream as if it carried no measurement error.  One bit per
    row keeps the run record small (``batch * batches`` bits).

    Only boolean/0-1 correctness values are encodable.  A fractional or
    out-of-range array would otherwise be truncated by the integer cast,
    silently turning e.g. ``0.8`` into a *correct* bit and fabricating a
    mask the model never produced.
    """
    try:
        if torch.is_tensor(correct):
            raw = correct.detach().reshape(-1)
        else:
            raw = torch.as_tensor(correct).detach().reshape(-1)
            if raw.is_floating_point():
                # Avoid the default float32 capture: a fractional value
                # such as 0.9999999999 must not round to an encodable 1.
                raw = torch.as_tensor(
                    correct, dtype=torch.float64).detach().reshape(-1)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            "correctness values must be real booleans/0-1, got "
            f"{type(correct).__name__}") from exc
    if raw.is_complex():
        raise ValueError("correctness values must be real booleans/0-1")
    if raw.is_floating_point() and (
            not bool(torch.isfinite(raw).all())
            or not bool(raw.eq(raw.round()).all())):
        raise ValueError(
            "correctness values must be integral booleans/0-1, got "
            "fractional or non-finite values")
    bits = raw.to(device="cpu", dtype=torch.int64)
    if bits.numel() and not bool(((bits == 0) | (bits == 1)).all()):
        raise ValueError(
            "correctness values must be booleans/0-1; out-of-range values "
            "cannot be encoded as one bit per item")
    pad = (-bits.numel()) % 8
    if pad:
        bits = torch.cat([bits, torch.zeros(pad, dtype=torch.int64)])
    if bits.numel() == 0:
        return ""
    weights = (1 << torch.arange(7, -1, -1)).to(torch.int64)
    packed = (bits.reshape(-1, 8) * weights).sum(dim=1)
    return base64.b64encode(bytes(packed.tolist())).decode("ascii")


def decode_correct_mask(encoded: str, n: int) -> torch.Tensor:
    """Decode :func:`encode_correct_mask` back to a boolean row mask."""
    if isinstance(n, bool):
        raise ValueError("n must be an integer, not bool")
    try:
        n = operator.index(n)
    except TypeError as exc:
        raise ValueError(f"n must be an integer, got {n!r}") from exc
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not isinstance(encoded, str):
        raise TypeError(f"encoded mask must be str, got {type(encoded)}")
    if not encoded:
        if n:
            raise ValueError(
                f"encoded mask is empty but n={n} bits were requested")
        return torch.zeros(0, dtype=torch.bool)
    raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    expected_bytes = (n + 7) // 8
    if len(raw) != expected_bytes:
        raise ValueError(
            f"encoded mask holds {len(raw)} byte(s) ({len(raw) * 8} bits) "
            f"but n={n} requires exactly {expected_bytes} byte(s); a "
            f"mismatched mask cannot be truncated to the recorded item "
            f"count without fabricating or dropping items")
    values = torch.tensor(list(raw), dtype=torch.int64)
    shifts = torch.arange(7, -1, -1).to(torch.int64)
    bits = ((values.reshape(-1, 1) >> shifts) & 1).reshape(-1)
    if n % 8 and bool(bits[n:].any()):
        # The encoder pads the final byte with zeros.  A nonzero padding bit
        # means the mask was corrupted or produced by a different layout;
        # ignoring it would still return n apparently valid item bits.
        raise ValueError(
            "encoded mask has nonzero padding bits after its "
            f"{n} recorded items")
    return bits[:n].bool()


# ---------------------------------------------------------------------------
# Counterfactual (minimal-pair) evaluation probes
# ---------------------------------------------------------------------------
# Accuracy alone does not say *which* rule a model used.  A model can be
# right because it retrieved the value paired with the query key, because it
# copied the token at a fixed position, because it learned a key -> value
# prior, or because it is insensitive to most edits and always emits a
# frequent token.  The task's generator makes every in-context pair
# identifiable, so the *data-level* manipulation that separates these
# explanations is a minimal edit of the already-realized evaluation batch:
#
# * value swap      - change the value paired with the queried key.  A
#                     retrieval model must follow the new value; a model that
#                     memorised key -> value (or copies a fixed position)
#                     keeps the old answer.
# * distractor swap - change a non-queried value.  The correct answer is
#                     unchanged, so a retrieval model should stay stable; a
#                     model that is globally sensitive to token edits fails
#                     the placebo control.
# * pair relocation - move the queried (key, value) pair to another pair
#                     slot.  The correct answer is unchanged, so a
#                     content-based retrieval model stays stable while a
#                     position-only copy rule moves with the position.
#
# Every edit is deterministic, preserves the task's structural contract
# (distinct keys, one SEP, the same query position) and uses only ids that
# the generator could have produced, so no new data distribution is invented.
# The probes are descriptive manipulation checks, not new FDR decision rows.


def _counterfactual_alternative(value, vocab):
    """A deterministic selectable token different from ``value``.

    The task reserves ids 0 (PAD) and 1 (SEP); values live in
    ``[2, vocab)``.  Cycling by one inside that range guarantees a different
    id whenever ``vocab >= 4`` (the generator's geometry contract).
    """
    span = int(vocab) - 2
    if span < 2:
        raise ValueError(
            f"counterfactual edits need at least two selectable values "
            f"(vocab >= 4), got vocab={vocab}")
    return (value.to(torch.long) - 2 + 1) % span + 2


def _counterfactual_edits(idx, meta, n_pairs, vocab):
    """Build the three minimal-pair edits of one validated evaluation batch.

    Returns a dict of edited contexts and their matching metadata, or
    ``None`` for an empty batch.  ``distractor_swap`` / ``pair_relocation``
    are ``None`` when ``n_pairs == 1``: there is no second pair to edit, so
    the probe abstains instead of silently editing the queried pair.

    The edits are made on a copy; the caller's batch is never mutated.
    ``meta['answer']`` is updated only for the value swap (the expected
    answer changes); relocation updates ``meta['key_pos']`` because the
    queried pair moved.
    """
    for name, value in (("n_pairs", n_pairs), ("vocab", vocab)):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, not bool")
        try:
            value = operator.index(value)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be an integer, got {value!r}") from exc
        if name == "n_pairs":
            if value < 1:
                raise ValueError(f"n_pairs must be >= 1, got {value!r}")
            n_pairs = int(value)
        else:
            if value < 4:
                raise ValueError(
                    f"counterfactual edits need at least two selectable "
                    f"values (vocab >= 4), got vocab={value!r}")
            if value > int(torch.iinfo(torch.long).max):
                raise ValueError(
                    f"vocab={value} exceeds torch's signed 64-bit index "
                    f"range")
            vocab = int(value)
    if not torch.is_tensor(idx) or idx.dim() != 2:
        raise ValueError(
            f"idx must be a 2-D integer tensor, got {type(idx)}")
    if idx.dtype == torch.bool or idx.is_floating_point() or idx.is_complex():
        # Token ids are class indices; a float/complex/bool cast with
        # ``torch.long`` would silently truncate or reinterpret them.
        raise ValueError(
            f"idx must be an integer token tensor, got dtype {idx.dtype}")
    # Edits and metadata are long token ids.  Normalise a narrower integer
    # input once instead of assigning long alternatives into it (which
    # raises a destination/source dtype RuntimeError).
    idx_long = idx.to(torch.long)
    if idx.dtype == torch.uint64 and bool((idx_long < 0).any()):
        raise ValueError("idx token ids must fit in int64")
    idx = idx_long
    if int(idx.size(1)) != 2 * int(n_pairs) + 2:
        raise ValueError(
            f"idx width {int(idx.size(1))} does not match the "
            f"[k v]*{int(n_pairs)} + SEP + query layout")
    if not isinstance(meta, Mapping) or "key_pos" not in meta \
            or "answer" not in meta:
        raise ValueError(
            "counterfactual edits need meta with 'key_pos' and 'answer'")
    batch = int(idx.size(0))
    if batch == 0:
        return None
    key_pos = meta["key_pos"]
    answer = meta["answer"]
    if not torch.is_tensor(key_pos) or not torch.is_tensor(answer):
        raise ValueError("meta['key_pos'] and meta['answer'] must be tensors")
    for name, value in (("key_pos", key_pos), ("answer", answer)):
        # The pair/answer geometry is integer metadata.  ``.to(long)``
        # would silently turn 0.5 into 0 or a boolean into a class index;
        # integral JSON-style floats (2.0) remain accepted.
        if value.dtype == torch.bool or value.is_complex():
            raise ValueError(
                f"meta['{name}'] must be an integer tensor, got dtype "
                f"{value.dtype}")
        if value.is_floating_point() and (
                not bool(torch.isfinite(value).all())
                or not bool(value.eq(value.round()).all())):
            raise ValueError(
                f"meta['{name}'] must contain integral values")
    key_pos = key_pos.to(device=idx.device, dtype=torch.long).reshape(-1)
    answer = answer.to(device=idx.device, dtype=torch.long).reshape(-1)
    if key_pos.shape != (batch,) or answer.shape != (batch,):
        raise ValueError(
            "meta['key_pos'] and meta['answer'] must carry one entry per row")
    if bool((key_pos < 0).any()) or bool((key_pos >= 2 * int(n_pairs)).any()):
        raise ValueError("meta['key_pos'] is outside the pair positions")
    if bool((key_pos % 2).ne(0).any()):
        raise ValueError("meta['key_pos'] must be an even key position")
    if bool((answer < 2).any()) or bool((answer >= int(vocab)).any()):
        raise ValueError(
            f"meta['answer'] must be a selectable id in [2, {vocab})")
    rows = torch.arange(batch, device=idx.device)
    query = idx[:, -1]
    paired_key = idx.gather(1, key_pos.unsqueeze(1)).squeeze(1)
    paired_value = idx.gather(1, (key_pos + 1).unsqueeze(1)).squeeze(1)
    if not bool(query.eq(paired_key).all()):
        raise ValueError(
            "the final query token does not match meta['key_pos']; "
            "counterfactual edits would target the wrong pair")
    if not bool(paired_value.eq(answer).all()):
        raise ValueError(
            "meta['answer'] does not match the value at meta['key_pos']+1")
    value_swap_answer = _counterfactual_alternative(answer, vocab)
    value_swap = idx.clone()
    value_swap[rows, key_pos + 1] = value_swap_answer
    value_swap_meta = dict(meta)
    value_swap_meta["answer"] = value_swap_answer
    distractor_swap = distractor_meta = None
    relocation = relocation_meta = None
    if int(n_pairs) >= 2:
        pair_index = key_pos // 2
        other = (pair_index + 1) % int(n_pairs)
        distractor_pos = 2 * other + 1
        current = idx.gather(1, distractor_pos.unsqueeze(1)).squeeze(1)
        distractor_swap = idx.clone()
        distractor_swap[rows, distractor_pos] = _counterfactual_alternative(
            current, vocab)
        distractor_meta = dict(meta)
        relocation = idx.clone()
        # Swap the whole (key, value) blocks; row-wise gather keeps the
        # operation deterministic and independent of advanced-index layout.
        for offset in (0, 1):
            k_col, j_col = 2 * pair_index + offset, 2 * other + offset
            moved = idx.gather(1, k_col.unsqueeze(1)).squeeze(1).clone()
            relocation[rows, k_col] = idx.gather(
                1, j_col.unsqueeze(1)).squeeze(1)
            relocation[rows, j_col] = moved
        relocation_meta = dict(meta)
        relocation_meta["key_pos"] = 2 * other
    return {
        "value_swap": value_swap,
        "value_swap_meta": value_swap_meta,
        "value_swap_answer": value_swap_answer,
        "distractor_swap": distractor_swap,
        "distractor_swap_meta": distractor_meta,
        "relocation": relocation,
        "relocation_meta": relocation_meta,
    }


def _counterfactual_counts(original_pred, answer, value_swap_pred,
                           value_swap_answer, distractor_pred=None,
                           relocation_pred=None) -> dict:
    """Item counts for the three counterfactual probes (no rates yet).

    Splitting counts from rates keeps the aggregation per batch exact
    (integer sums) and lets the summary abstain cleanly when a probe has no
    eligible rows (e.g. ``n_pairs == 1`` has no distractor).
    """
    def _ids(value, name):
        # Token predictions/answers are class indices.  A float cast would
        # silently truncate a fractional value and a complex cast would drop
        # the imaginary part, fabricating a token the caller never supplied;
        # integral JSON-style floats (2.0) remain accepted.
        try:
            if torch.is_tensor(value):
                t = value.detach()
            else:
                t = torch.as_tensor(value).detach()
                if t.is_floating_point():
                    # Do not let float32 round a fractional token id into
                    # an integer the caller never supplied.
                    t = torch.as_tensor(value, dtype=torch.float64).detach()
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError(
                f"{name} must be an integer token tensor, got "
                f"{type(value).__name__}") from exc
        if t.is_complex() or t.dtype == torch.bool:
            raise ValueError(
                f"{name} must be an integer token tensor, got dtype {t.dtype}")
        if t.is_floating_point():
            if (not bool(torch.isfinite(t).all())
                    or not bool(t.eq(t.round()).all())):
                raise ValueError(
                    f"{name} must contain integral token ids, got "
                    f"fractional or non-finite values")
            if not bool(((t >= -2.0 ** 63) & (t < 2.0 ** 63)).all()):
                raise ValueError(
                    f"{name} token ids must fit in int64")
        try:
            converted = t.reshape(-1).to(torch.long)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError(
                f"{name} must be an integer token tensor") from exc
        if t.dtype == torch.uint64 and bool((converted < 0).any()):
            raise ValueError(f"{name} token ids must fit in int64")
        return converted

    pred = _ids(original_pred, "original_pred")
    ans = _ids(answer, "answer").to(device=pred.device)
    swap_pred = _ids(value_swap_pred, "value_swap_pred")
    swap_ans = _ids(value_swap_answer, "value_swap_answer").to(
        device=pred.device)
    for name, value in (("answer", ans), ("value_swap_answer", swap_ans),
                        ("value_swap_pred", swap_pred)):
        if value.shape != pred.shape:
            raise ValueError(
                f"{name} has {int(value.numel())} entries for "
                f"{int(pred.numel())} predictions")
    n = int(pred.numel())
    out = {
        "n": n,
        "n_original_correct": 0,
        "n_value_follow": 0,
        "n_value_stubborn": 0,
        "n_value_flip": 0,
        "n_value_correct": 0,
        "n_distractor": 0,
        "n_distractor_unchanged": 0,
        "n_distractor_correct": 0,
        "n_relocation": 0,
        "n_relocation_unchanged": 0,
        "n_relocation_correct": 0,
    }
    if n == 0:
        return out
    original_correct = pred.eq(ans)
    out["n_original_correct"] = int(original_correct.sum())
    out["n_value_follow"] = int(
        (original_correct & swap_pred.eq(swap_ans)).sum())
    out["n_value_stubborn"] = int(
        (original_correct & swap_pred.eq(ans)).sum())
    out["n_value_flip"] = int(pred.ne(swap_pred).sum())
    out["n_value_correct"] = int(swap_pred.eq(swap_ans).sum())
    if distractor_pred is not None:
        ctrl = _ids(distractor_pred, "distractor_pred")
        if ctrl.shape != pred.shape:
            raise ValueError(
                f"distractor_pred has {int(ctrl.numel())} entries for "
                f"{int(pred.numel())} predictions")
        out["n_distractor"] = n
        out["n_distractor_unchanged"] = int(ctrl.eq(pred).sum())
        out["n_distractor_correct"] = int(ctrl.eq(ans).sum())
    if relocation_pred is not None:
        moved = _ids(relocation_pred, "relocation_pred")
        if moved.shape != pred.shape:
            raise ValueError(
                f"relocation_pred has {int(moved.numel())} entries for "
                f"{int(pred.numel())} predictions")
        out["n_relocation"] = n
        out["n_relocation_unchanged"] = int(moved.eq(pred).sum())
        out["n_relocation_correct"] = int(moved.eq(ans).sum())
    return out


def _nonnegative_count(value):
    """One exact non-negative count from a recorded accumulator row.

    ``bool`` and fractional values are rejected rather than truncated: a
    count that is not a whole number cannot describe evaluated rows.  JSON
    round-trips may turn an integer into ``3.0``, so integral floats are
    accepted, and arbitrarily large Python integers stay exact.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if (isinstance(value, float) and math.isfinite(value)
            and value == int(value)):
        return int(value) if value >= 0 else None
    return None


def _counterfactual_summary(counts) -> dict:
    """Turn summed probe counts into rates plus flat aggregate fields.

    ``follow_rate`` is conditioned on rows the clean model already answered
    correctly: those are the rows where "did the answer track the edited
    value?" is a meaningful question.  If the clean model was never correct,
    the rate is NaN rather than 0 (an undefined diagnostic is not evidence of
    a memorising model).
    """

    # Every field is a row count, so malformed values (fractional, boolean,
    # negative, non-finite, non-numeric) or impossible row inequalities make
    # the whole accumulator unusable.  Report the same empty/NaN payload as a
    # zero-row probe rather than raising, truncating, or showing a rate whose
    # denominator silently disagrees with its components.
    fields = (
        "n", "n_original_correct", "n_value_follow", "n_value_stubborn",
        "n_value_flip", "n_value_correct", "n_distractor",
        "n_distractor_unchanged", "n_distractor_correct", "n_relocation",
        "n_relocation_unchanged", "n_relocation_correct",
    )
    raw = counts if isinstance(counts, Mapping) else {}
    parsed = {}
    for name in fields:
        count = _nonnegative_count(raw.get(name, 0))
        if count is None:
            parsed = None
            break
        parsed[name] = count
    if parsed is not None:
        n = parsed["n"]
        n_correct = parsed["n_original_correct"]
        impossible = not (
            n_correct <= n
            and parsed["n_value_follow"] <= n_correct
            # A follow row is also a value-swap-correct row (both compare
            # the swapped prediction with the swapped answer), so the
            # follow count can never exceed the overall correct count.
            and parsed["n_value_follow"] <= parsed["n_value_correct"]
            # A follow row changes the prediction (the swapped answer is
            # guaranteed to differ from the clean answer), so follow rows
            # are flipped rows and cannot outnumber them.
            and parsed["n_value_follow"] <= parsed["n_value_flip"]
            # A stubborn row keeps the clean prediction, so it is disjoint
            # from the flipped rows.
            and parsed["n_value_stubborn"] + parsed["n_value_flip"] <= n
            and parsed["n_value_stubborn"] <= n_correct
            # Follow and stubborn both require a clean-correct row and are
            # mutually exclusive (new swapped answer vs original answer),
            # so together they fit inside the clean-correct count.
            and (parsed["n_value_follow"] + parsed["n_value_stubborn"]
                 <= n_correct)
            and parsed["n_value_flip"] <= n
            and parsed["n_value_correct"] <= n
            # Inclusion-exclusion: the clean-correct and value-swap-correct
            # sets intersect exactly on the follow rows.
            and (n_correct + parsed["n_value_correct"]
                 - parsed["n_value_follow"] <= n)
            and parsed["n_distractor"] <= n
            and parsed["n_distractor_unchanged"] <= parsed["n_distractor"]
            and parsed["n_distractor_correct"] <= parsed["n_distractor"]
            and parsed["n_relocation"] <= n
            and parsed["n_relocation_unchanged"] <= parsed["n_relocation"]
            and parsed["n_relocation_correct"] <= parsed["n_relocation"])
        if impossible:
            parsed = None
    counts = parsed if parsed is not None else {name: 0 for name in fields}

    def rate(num, den):
        # Exact integer counts above 2**53 (or even above the float range)
        # must not be routed through float separately: Python's int/int
        # division scales first and still returns the representable ratio.
        return num / den if den else float("nan")

    n = int(counts.get("n", 0))
    n_correct = int(counts.get("n_original_correct", 0))
    value_swap = {
        "n": n,
        "n_original_correct": n_correct,
        "n_follow": int(counts.get("n_value_follow", 0)),
        "follow_rate": rate(counts.get("n_value_follow", 0), n_correct),
        "n_stubborn": int(counts.get("n_value_stubborn", 0)),
        "stubborn_rate": rate(counts.get("n_value_stubborn", 0), n_correct),
        "n_correct_after_edit": int(counts.get("n_value_correct", 0)),
        "accuracy_after_edit": rate(counts.get("n_value_correct", 0), n),
        "flip_rate": rate(counts.get("n_value_flip", 0), n),
    }
    distractor_swap = {
        "n": int(counts.get("n_distractor", 0)),
        "n_unchanged": int(counts.get("n_distractor_unchanged", 0)),
        "unchanged_rate": rate(counts.get("n_distractor_unchanged", 0),
                               counts.get("n_distractor", 0)),
        "n_correct_after_edit": int(counts.get("n_distractor_correct", 0)),
        "accuracy_after_edit": rate(counts.get("n_distractor_correct", 0),
                                    counts.get("n_distractor", 0)),
    }
    pair_relocation = {
        "n": int(counts.get("n_relocation", 0)),
        "n_unchanged": int(counts.get("n_relocation_unchanged", 0)),
        "unchanged_rate": rate(counts.get("n_relocation_unchanged", 0),
                               counts.get("n_relocation", 0)),
        "n_correct_after_edit": int(counts.get("n_relocation_correct", 0)),
        "accuracy_after_edit": rate(counts.get("n_relocation_correct", 0),
                                    counts.get("n_relocation", 0)),
    }
    flat = {
        "cf_value_follow_rate": value_swap["follow_rate"],
        "cf_value_stubborn_rate": value_swap["stubborn_rate"],
        "cf_value_accuracy": value_swap["accuracy_after_edit"],
        "cf_value_flip_rate": value_swap["flip_rate"],
        "cf_distractor_unchanged_rate": distractor_swap["unchanged_rate"],
        "cf_distractor_accuracy": distractor_swap["accuracy_after_edit"],
        "cf_relocation_unchanged_rate": pair_relocation["unchanged_rate"],
        "cf_relocation_accuracy": pair_relocation["accuracy_after_edit"],
        # Denominators travel with the rates: a follow_rate conditioned on
        # three clean-correct rows is not the same evidence as one
        # conditioned on three hundred, and an undefined rate must not hide
        # that its denominator was zero.
        "cf_n": n,
        "cf_n_original_correct": n_correct,
        "cf_n_distractor": distractor_swap["n"],
        "cf_n_relocation": pair_relocation["n"],
    }
    return {
        "value_swap": value_swap,
        "distractor_swap": distractor_swap,
        "pair_relocation": pair_relocation,
        "flat": flat,
        "note": (
            "deterministic minimal-pair data edits; descriptive retrieval "
            "manipulation checks, not pre-registered FDR decision rows. "
            "follow_rate is conditioned on rows the clean pass answered "
            "correctly; distractor/relocation unchanged_rate is the placebo "
            "specificity check."),
    }


def _blank_exposure_cohort() -> dict:
    """Integer accumulators for one exposure cohort (paired across batches)."""
    return {
        "n": 0,
        "n_correct": 0,
        "n_value_swap": 0,
        "n_value_correct": 0,
        "n_value_follow": 0,
        "n_value_stubborn": 0,
        "n_distractor": 0,
        "n_distractor_correct": 0,
        "n_relocation": 0,
        "n_relocation_correct": 0,
    }


_EXPOSURE_DOSE_LABELS = ("0", "1", "2", "3-4", "5-9", "10+")
_EXPOSURE_DOSE_BOUNDARIES = (1, 2, 3, 5, 10)


def _blank_exposure_dose() -> list:
    """Integer accumulators for the fixed training-exposure dose bins."""
    return [{"n": 0, "n_correct": 0, "exposure_total": 0,
             "max_exposure": 0} for _ in _EXPOSURE_DOSE_LABELS]


def _association_exposure_dose_accumulate(totals, idx, meta, vocab,
                                          reference_counts, pred,
                                          answer) -> None:
    """Add one batch to the training-exposure dose-response counters.

    ``reference_counts`` maps every exact ``(query key, answer value)``
    association realized by the training streams to the number of times it
    was drawn.  Items are binned by that count (0, 1, 2, 3-4, 5-9, 10+), a
    pre-treatment split that depends only on the training data.  Pooling
    all seen pairs into one cohort can hide a dose-response: if accuracy
    rises with repeated exposure, the binary seen/unseen gap understates
    memorisation and the highest-exposure cohort is the informative one.
    """
    try:
        pred = (pred.detach() if torch.is_tensor(pred)
                else torch.as_tensor(pred).detach()).reshape(-1)
        answer = (answer.detach() if torch.is_tensor(answer)
                  else torch.as_tensor(answer).detach()).reshape(-1)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            "pred and answer must be tensor-like sequences, got "
            f"{type(pred).__name__}/{type(answer).__name__}") from exc
    if pred.numel() != answer.numel():
        raise ValueError(
            f"answer has {int(answer.numel())} entries for "
            f"{int(pred.numel())} predictions")
    codes = association_codes(idx, meta, vocab)["code"].tolist()
    if len(codes) != int(pred.numel()):
        raise ValueError(
            f"association exposure has {len(codes)} entries for "
            f"{int(pred.numel())} rows")
    counts = torch.tensor(
        [int(reference_counts.get(int(code), 0)) for code in codes],
        dtype=torch.long, device=pred.device)
    boundaries = torch.tensor(
        _EXPOSURE_DOSE_BOUNDARIES, dtype=torch.long, device=pred.device)
    bins = torch.bucketize(counts, boundaries, right=True)
    correct = pred.to(device=counts.device).eq(
        answer.to(device=counts.device))
    for bin_idx, entry in enumerate(totals):
        mask = bins.eq(bin_idx)
        n = int(mask.sum())
        if n == 0:
            continue
        entry["n"] += n
        entry["n_correct"] += int(correct[mask].sum())
        entry["exposure_total"] += int(counts[mask].sum())
        entry["max_exposure"] = max(
            entry["max_exposure"], int(counts[mask].max()))


def _association_exposure_accumulate(totals, idx, meta, vocab, reference,
                                     pred, answer, swap_pred=None,
                                     swap_answer=None, distractor_pred=None,
                                     relocation_pred=None) -> None:
    """Add one batch to the exposure-stratified clean/probe counters.

    The cohort label is fixed by the *training* stream (was this exact
    query key-value pair realized there?), never by the model's answer, so
    the comparison is a pre-treatment stratification.  Per-item correctness
    and the counterfactual value-swap outcomes are accumulated separately
    for seen and unseen pairs; no rate is formed until every batch has been
    added, so a zero-row cohort abstains instead of contributing 0.
    """
    try:
        pred = (pred.detach() if torch.is_tensor(pred)
                else torch.as_tensor(pred).detach())
        pred = pred.reshape(-1)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(
            "pred must be a tensor-like sequence of predictions, got "
            f"{type(pred).__name__}") from exc
    n = int(pred.numel())

    def _per_item(value, name):
        # Flatten before comparing: a correctly sized [n, 1] panel must
        # not broadcast against an [n] answer into an [n, n] correctness
        # matrix, and a mismatched length must fail closed instead of
        # indexing/counting only a prefix (or raising a raw torch error).
        try:
            tensor = (value.detach() if torch.is_tensor(value)
                      else torch.as_tensor(value).detach())
            tensor = tensor.reshape(-1)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError(
                f"{name} must be a tensor-like sequence of predictions, "
                f"got {type(value).__name__}") from exc
        if tensor.numel() != n:
            raise ValueError(
                f"{name} has {int(tensor.numel())} entries for "
                f"{n} predictions")
        return tensor.to(device=pred.device)

    answer = _per_item(answer, "answer")
    if swap_pred is not None:
        if swap_answer is None:
            raise ValueError(
                "swap_answer is required when swap_pred is provided")
        swap_pred = _per_item(swap_pred, "swap_pred")
        swap_answer = _per_item(swap_answer, "swap_answer")
    if distractor_pred is not None:
        distractor_pred = _per_item(distractor_pred, "distractor_pred")
    if relocation_pred is not None:
        relocation_pred = _per_item(relocation_pred, "relocation_pred")
    codes = association_codes(idx, meta, vocab)["code"].tolist()
    if not isinstance(reference, frozenset):
        reference = association_reference_codes(reference)
    seen = torch.tensor([int(code) in reference for code in codes],
                        dtype=torch.bool, device=pred.device)
    if seen.numel() != n:
        raise ValueError(
            "association exposure has "
            f"{int(seen.numel())} entries for {n} rows")
    correct = pred.eq(answer)
    for name, mask in (("seen", seen), ("unseen", ~seen)):
        n = int(mask.sum())
        if n == 0:
            continue
        entry = totals.setdefault(name, _blank_exposure_cohort())
        entry["n"] += n
        entry["n_correct"] += int(correct[mask].sum())
        if swap_pred is not None:
            if swap_answer is None:
                raise ValueError(
                    "swap_answer is required when swap_pred is provided")
            swap_correct = swap_pred.eq(swap_answer)
            entry["n_value_swap"] += n
            entry["n_value_correct"] += int(swap_correct[mask].sum())
            entry["n_value_follow"] += int(
                (correct & swap_correct)[mask].sum())
            entry["n_value_stubborn"] += int(
                (correct & swap_pred.eq(answer))[mask].sum())
        if distractor_pred is not None:
            entry["n_distractor"] += n
            entry["n_distractor_correct"] += int(
                distractor_pred.eq(answer)[mask].sum())
        if relocation_pred is not None:
            entry["n_relocation"] += n
            entry["n_relocation_correct"] += int(
                relocation_pred.eq(answer)[mask].sum())
