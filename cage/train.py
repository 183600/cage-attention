"""Training and evaluation utilities."""

from __future__ import annotations

import math

import torch

from .data import make_batch


def _make_adamw(params, lr):
    """AdamW on the current device, preferring the backends it supports."""
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

    ``dense_lm`` supervises every next-token position of the pair section as
    well.  Those targets are unpredictable in the causal layout (each value
    only ever appears after its own key), so the extra loss buys the marginal
    token frequency and, at ~15:1, drowns the gradient that actually teaches
    the recall operation the accuracy metric measures.  Keep it off unless a
    marginal prior is explicitly part of the study.

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
        if step % log_every == 0 or step == steps - 1:
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
        # logits carry a single query position, so column 0 is a contiguous view
        pred = logits.squeeze(1).argmax(-1)
        correct = correct + (pred == meta["answer"]).sum()
        total += idx.size(0)
    model.train(was_training)
    return correct.item() / total


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
        # the model evaluates the diagnostics on the query positions directly
        logits, _, layer_stats = model(idx, stat_pos=meta["query_pos"])
        pred = logits.squeeze(1).argmax(-1)
        out["correct"].append((pred == meta["answer"]).float())
        ents, maxps, nulls, gates, rns = [], [], [], [], []
        for s in layer_stats:
            ents.append(s["entropy"].mean(1))                # (B,h) -> B
            maxps.append(s["maxprob"].max(1)[0])             # (B,h) -> B
            rns.append(s["resid_norm"])
            if s["gate"] is not None:
                gates.append(s["gate"].mean(-1))             # (B,C) -> B
            if s["null_mass"] is not None:
                nulls.append(s["null_mass"].mean(1))         # (B,h) -> B
        out["entropy"].append(torch.stack(ents, 1))                    # B, L
        out["maxprob"].append(torch.stack(maxps, 1))
        out["resid_norm"].append(torch.stack(rns, 1))                  # B, L
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
    mp = stats["maxprob"].max(dim=1)[0]   # best layer maxprob per sample
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
