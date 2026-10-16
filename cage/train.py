"""Training and evaluation utilities."""

from __future__ import annotations

import math

import torch

from .data import make_batch
from .metrics import (
    brier_score,
    expected_calibration_error,
    multiclass_metrics,
    pr_auc,
    roc_auc,
)


def _make_adamw(params, lr):
    """AdamW on the current device."""
    params = list(params)
    kw = dict(lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    if any(p.is_cuda for p in params):
        try:
            return torch.optim.AdamW(params, foreach=False, fused=True, **kw)
        except (TypeError, RuntimeError, ValueError):
            pass
    try:
        return torch.optim.AdamW(params, foreach=True, **kw)
    except (TypeError, RuntimeError, ValueError):
        pass
    return torch.optim.AdamW(params, **kw)


def train_model(model, steps=1500, batch=64, n_pairs=16, lr=3e-4, vocab=512,
                device="cpu", log_every=250, seed=0, dense_lm=False):
    """Train on the recall task.

    ``seed`` governs *all* run-to-run randomness: the data generator and the
    global RNG that draws the attention/MLP dropout masks.  Without the
    latter two runs of the same config would silently differ.
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    params = list(model.parameters())
    opt = _make_adamw(params, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
    hist = []
    model.train()
    for step in range(steps):
        idx, tgt, meta = make_batch(batch, n_pairs, vocab, device, dense_lm=dense_lm, gen=gen)
        opt.zero_grad(set_to_none=True)
        sup_pos = None if dense_lm else meta["query_pos"]
        _, loss, _ = model(idx, tgt, sup_pos=sup_pos)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if (log_every > 0 and step % log_every == 0) or step == steps - 1:
            lv = loss.item()
            hist.append((step, lv))
            print(f"  step {step:5d}  loss {lv:.4f}")
    return hist


@torch.inference_mode()
def eval_accuracy(model, n_pairs, vocab, device, batch=256, batches=4, seed=123):
    """Recall accuracy at the query position."""
    gen = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for _ in range(batches):
        idx, _, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        logits, _, _ = model(idx)
        pred = logits.squeeze(1).argmax(-1)
        correct += int((pred == meta["answer"]).sum())
        total += idx.size(0)
    model.train(was_training)
    return correct / total


@torch.inference_mode()
def eval_metrics(model, n_pairs, vocab, device, batch=256, batches=4, seed=123,
                 topk=5):
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
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    logits_all, tgt_all = [], []
    for _ in range(batches):
        idx, _, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        logits, _, _ = model(idx)
        logits_all.append(logits.squeeze(1).float().cpu())
        tgt_all.append(meta["answer"].cpu())
    model.train(was_training)
    logits_cat = torch.cat(logits_all, dim=0)
    tgt_cat = torch.cat(tgt_all, dim=0)
    out = multiclass_metrics(logits_cat, tgt_cat, topk=topk)
    out["nll"] = float(torch.nn.functional.cross_entropy(logits_cat, tgt_cat))
    probs = logits_cat.softmax(dim=-1)
    confidence, pred = probs.max(dim=-1)
    out["ece_15bin"] = expected_calibration_error(confidence, pred == tgt_cat,
                                                  n_bins=15)
    out["brier"] = brier_score(confidence, pred == tgt_cat)
    return out


@torch.inference_mode()
def collect_stats(model, n_pairs, vocab, device, batch=128, batches=4, seed=7):
    """Collect per-layer gate / entropy / maxprob / null-mass at query positions,
    plus false-concentration labels (high max-prob but wrong answer).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    out = {"gate": [], "entropy": [], "maxprob": [], "null_mass": [],
           "correct": [], "resid_norm": []}
    for _ in range(batches):
        idx, _, meta = make_batch(batch, n_pairs, vocab, device, gen=gen)
        logits, _, layer_stats = model(idx, stat_pos=meta["query_pos"])
        pred = logits.squeeze(1).argmax(-1)
        out["correct"].append((pred == meta["answer"]).float())
        ents, maxps, nulls, gates, rns = [], [], [], [], []
        for s in layer_stats:
            ents.append(s["entropy"].mean(1))
            maxps.append(s["maxprob"].max(1)[0])
            rns.append(s["resid_norm"])
            if s["gate"] is not None:
                gates.append(s["gate"].mean(-1))
            if s["null_mass"] is not None:
                nulls.append(s["null_mass"].mean(1))
        out["entropy"].append(torch.stack(ents, 1))
        out["maxprob"].append(torch.stack(maxps, 1))
        out["resid_norm"].append(torch.stack(rns, 1))
        if gates:
            out["gate"].append(torch.stack(gates, 1))
        if nulls:
            out["null_mass"].append(torch.stack(nulls, 1))
    stats = {k: torch.cat(v).cpu() for k, v in out.items() if v}
    model.train(was_training)
    return stats


def gate_depth_ratio(stats, n_early=2):
    """Deep/early gate ratio: mean gate of the deepest layer over the mean of
    the shallowest ``n_early`` layers (E3).

    ``> 1`` means the collaborative normalisation made the deep layers more
    willing to write into the residual stream, i.e. it worked against the
    "deep layers contribute too little" symptom of Pre-LN.
    """
    g = stats.get("gate")
    if g is None or g.size(1) <= n_early:
        return float("nan")
    early = g[:, :n_early].mean()
    deep = g[:, -1].mean()
    return (deep / early).item()


def pearson(x, y):
    x = x.flatten().float()
    y = y.flatten().float()
    dx = x - x.mean()
    dy = y - y.mean()
    if (dx.abs().max() <= 1e-6 * x.abs().max().clamp_min(1.0)
            or dy.abs().max() <= 1e-6 * y.abs().max().clamp_min(1.0)):
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
    g = stats.get("gate")
    if g is None:
        return r
    ent, mp, corr = stats["entropy"], stats["maxprob"], stats["correct"]
    cols = range(g.size(1))
    r["gate_vs_entropy"] = _fisher_mean(
        [pearson(g[:, i], ent[:, i]).item() for i in cols])
    r["gate_vs_maxprob"] = _fisher_mean(
        [pearson(g[:, i], mp[:, i]).item() for i in cols])
    r["gate_vs_correct"] = _fisher_mean(
        [pearson(g[:, i], corr).item() for i in cols])
    return r


def false_concentration_analysis(stats, maxprob_thresh=0.5):
    """'False concentration': max attention prob high but answer wrong.

    Computes, among high-maxprob samples, the error rate (false-concentration
    rate) and, for gated models, mean gate value on false-concentrated vs
    correct samples - a good calibrated gate should close (lower value)
    when concentration is false.
    """
    mp = stats["maxprob"].max(dim=1)[0]
    high = mp > maxprob_thresh
    corr = stats["correct"].bool()
    r = {
        "n_high_maxprob": int(high.sum()),
        "false_conc_rate": 1.0 - corr[high].float().mean().item() if high.any() else float("nan"),
        "overall_acc": corr.float().mean().item(),
    }
    g = stats.get("gate")
    if g is not None:
        gm = g.mean(1)
        fc = high & ~corr
        r["gate_on_false_conc"] = gm[fc].mean().item() if fc.any() else float("nan")
        r["gate_on_correct"] = gm[corr & high].mean().item() if (corr & high).any() else float("nan")
    return r


def error_detection_aucs(stats):
    """ROC-AUC and PR-AUC of attention statistics as *error detectors*.

    The positive class is "model answered wrong".  Scores are oriented so
    that **higher means more likely wrong** (``-maxprob``, mean entropy,
    ``-gate``), matching the mechanism hypotheses: concentration should
    track correctness, and a calibrated gate should close when it does
    not.  Errors are the rare class after convergence, so PR-AUC is the
    primary summary; ROC-AUC is reported for completeness only (with a
    huge negative population it can look optimistic while precision
    collapses; see ``cage/metrics.py``).
    """
    corr = stats["correct"].bool()
    err = ~corr
    n = int(corr.numel())
    n_err = int(err.sum())
    out = {"n": n, "n_errors": n_err,
           "error_rate": (n_err / n) if n else float("nan")}
    scores = {
        "maxprob": -stats["maxprob"].max(dim=1)[0],
        "entropy": stats["entropy"].mean(dim=1),
    }
    if stats.get("gate") is not None:
        scores["gate"] = -stats["gate"].mean(dim=1)
    for name in ("maxprob", "entropy", "gate"):
        score = scores.get(name)
        if score is None or n_err == 0 or n_err == n:
            out[f"rocauc_{name}_to_error"] = float("nan")
            out[f"prauc_{name}_to_error"] = float("nan")
            continue
        out[f"rocauc_{name}_to_error"] = roc_auc(score, err)
        out[f"prauc_{name}_to_error"] = pr_auc(score, err)
    return out
