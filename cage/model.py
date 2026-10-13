"""CAGE-Attention: Calibrated Attention Gating Experiments.

Implements a small GPT-style decoder-only transformer with pluggable
attention-gate variants and normalization schemes.

Gate variants
-------------
- ``none``     : standard attention (no output gate).
- ``query``    : pure query-dependent gate (Gated Attention style),
                 gate = sigmoid(W_g x) applied to attention output.
- ``calib``    : length + concentration calibrated gate (ours).
                 gate = sigmoid((W_g x - b0 - a*log(T) - c*H_norm) / tau)
                 where H_norm is the per-head normalized attention entropy.
                 The gate therefore depends on the query *and* on how
                 concentrated the attention distribution actually is,
                 corrected for sequence length (entropy grows with log T).
- ``null``     : explicit null-attention / sparse selection baseline
                 (a learnable "null" key-value slot competes with real
                 tokens; choosing null suppresses the attention write).

Normalization schemes
---------------------
- ``pre``      : Pre-LN.
- ``post``     : Post-LN.
- ``mix``      : Mix-LN (Pre-LN in first half of depth, Post-LN in second).
- ``seednorm`` : SeeDNorm-style dynamic normalization - the norm scale is
                 predicted per-token from the hidden state and applied to
                 the residual stream, letting deep layers decide how much
                 to "write".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


_EPS = 1e-9

GATES = ("none", "query", "calib", "null")
NORMS = ("pre", "post", "mix", "seednorm")


@dataclass
class Cfg:
    vocab_size: int = 512
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 8
    max_len: int = 256
    dropout: float = 0.0
    gate: str = "none"          # none | query | calib | null
    norm: str = "pre"           # pre | post | mix | seednorm
    layer_scale: float | None = None  # initial LayerScale value or None
    calib_a: float = 1.0        # length-correction strength (learnable scale)
    calib_c: float = 1.0        # concentration-correction strength
    calib_scale: float = 1.0    # additional concentration scaling
    calib_tau_init: float = 0.9  # initial gate temperature (softplus(tau)+0.1 = this)
    use_gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.gate not in GATES:
            raise ValueError(f"unknown gate {self.gate!r}; valid: {list(GATES)}")
        if self.norm not in NORMS:
            raise ValueError(f"unknown norm {self.norm!r}; valid: {list(NORMS)}")
        if self.n_heads <= 0 or self.d_model % self.n_heads:
            raise ValueError(
                f"d_model={self.d_model} is not a positive multiple of "
                f"n_heads={self.n_heads}"
            )


class Norm(nn.Module):
    __slots__ = ("ln",)
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=True)

    def forward(self, x):
        return self.ln(x)


class SeeDNorm(nn.Module):
    """Dynamic normalization: per-token scale predicted from hidden state."""
    __slots__ = ("ln", "scale")
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale = nn.Linear(dim, dim)
        nn.init.zeros_(self.scale.bias)
        nn.init.normal_(self.scale.weight, std=0.02)

    def forward(self, x):
        s = self.scale(x).sigmoid_()
        return self.ln(x) * s.mul(2.0)


def make_norm(kind: str, dim: int) -> nn.Module:
    return SeeDNorm(dim) if kind == "seednorm" else Norm(dim)


_MAX_MASK_CACHE = 64

_causal_mask_cache = {}
_causal_mask_order = []


def _causal_mask(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Additive causal mask (0 on/below the diagonal, -inf above)."""
    key = (T, device, dtype)
    m = _causal_mask_cache.get(key)
    if m is not None:
        _causal_mask_order.remove(key)
        _causal_mask_order.append(key)
        return m
    if len(_causal_mask_cache) >= _MAX_MASK_CACHE:
        evict_key = _causal_mask_order.pop(0)
        _causal_mask_cache.pop(evict_key)
    m = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
    m.triu_(1)
    _causal_mask_cache[key] = m
    _causal_mask_order.append(key)
    return m


def _mask_rows(pos: torch.Tensor, T: int, h: int, dtype: torch.dtype) -> torch.Tensor:
    """``(B*h, 1, T)`` additive mask rows for the query positions ``pos``.

    Row ``b`` is 0 up to column ``pos[b]`` and ``-inf`` beyond it.  The mask
    depends on the *values* of ``pos``, so it is rebuilt per call; only the
    shared per-length causal mask is cached.
    """
    B = pos.size(0)
    cols = torch.arange(T, device=pos.device)
    m = torch.zeros(B, T, dtype=dtype, device=pos.device)
    m.masked_fill_(cols.view(1, T) > pos.view(B, 1), float("-inf"))
    return m.view(B, 1, T).expand(B, h, T).reshape(B * h, 1, T)


def _softmax_inplace(s: torch.Tensor) -> torch.Tensor:
    """Row softmax written back into ``s`` (no autograd node)."""
    s.sub_(s.amax(dim=-1, keepdim=True)).exp_()
    s.div_(s.sum(dim=-1, keepdim=True))
    return s


class _SoftmaxInPlace(torch.autograd.Function):
    """``softmax(x, -1)`` written back into ``x``."""

    @staticmethod
    def forward(ctx, x):
        ctx.mark_dirty(x)
        x.sub_(x.amax(dim=-1, keepdim=True)).exp_()
        x.div_(x.sum(dim=-1, keepdim=True))
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_out):
        (y,) = ctx.saved_tensors
        gy = grad_out.mul(y)
        return gy.addcmul_(y, gy.sum(-1, keepdim=True), value=-1.0)


class _RowEntropy(torch.autograd.Function):
    """``(p * p.clamp_min(eps).log()).sum(-1).neg()`` computed directly."""

    @staticmethod
    def forward(ctx, p, eps: float):
        ctx.save_for_backward(p)
        ctx.eps = eps
        return _row_entropy(p, eps)

    @staticmethod
    def backward(ctx, grad_out):
        (p,) = ctx.saved_tensors
        g = grad_out.unsqueeze(-1).neg()
        mask = p > ctx.eps
        d = p.clamp_min(ctx.eps).log_()
        d.add_(mask).mul_(mask).mul_(g)
        return d, None


def _batch_index(B: int, device: torch.device) -> torch.Tensor:
    """``arange(B)`` on ``device``: one row is picked per sequence."""
    return torch.arange(B, device=device)


def _pick_rows(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """``x[b, :, pos[b], ...]`` for every ``b``: ``(B, H, T, S) -> (B, H, S)``."""
    return x[torch.arange(x.size(0), device=x.device), :, pos]


def _pick_tokens(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """``x[b, pos[b], ...]`` for every ``b``: ``(B, T, C) -> (B, C)``."""
    return x[torch.arange(x.size(0), device=x.device), pos]


def _query_blocks(bh: int, T: int):
    """``[(start, stop), ...]`` query blocks for a ``bh``-fold batch of ``T``."""
    if T <= 32:
        return ((0, T),)
    max_rows = (1 << 20) // (bh * T)
    if max_rows < 32:
        max_rows = 32
    rows = min(T, max_rows)
    return tuple((i, min(i + rows, T)) for i in range(0, T, rows))


def _row_entropy(p: torch.Tensor, eps: float) -> torch.Tensor:
    """``-(p * log p).sum(-1)`` computed row-wise."""
    return p.clamp_min(eps).log_().mul_(p).sum(dim=-1).neg_()


class _CalibAttn(torch.autograd.Function):
    """Calibrated attention (dropout-free).

    Computes, on the ``(B*H, T, *)`` layout:

        p   = softmax(q k^T / sqrt(dk) + causal)
        y   = p v
        ent = -(p * log p).rowsum

    The probabilities are rebuilt block by block: forward scores them one
    block at a time and backward recomputes them from ``q, k, v``.
    """

    @staticmethod
    def forward(ctx, q, k, v, scale: float, eps: float):
        bh, T = q.size(0), q.size(1)
        blocks = _query_blocks(bh, T)
        mask = _causal_mask(T, q.device, q.dtype)
        kt = k.transpose(1, 2)
        if len(blocks) == 1:
            p = _softmax_inplace(torch.baddbmm(mask, q, kt, beta=1.0, alpha=scale))
            y = torch.bmm(p, v)
            ent = _row_entropy(p, eps)
        else:
            y = q.new_empty(bh, T, v.size(-1))
            ent = q.new_empty(bh, T)
            for i0, i1 in blocks:
                p = _softmax_inplace(torch.baddbmm(mask[i0:i1, :i1], q[:, i0:i1],
                                                   kt[:, :, :i1], beta=1.0, alpha=scale))
                y[:, i0:i1] = torch.bmm(p, v[:, :i1])
                ent[:, i0:i1] = _row_entropy(p, eps)
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        ctx.eps = eps
        return y, ent

    @staticmethod
    def backward(ctx, grad_y, grad_ent):
        q, k, v = ctx.saved_tensors
        scale, eps = ctx.scale, ctx.eps
        bh, T = q.size(0), q.size(1)
        blocks = _query_blocks(bh, T)
        mask = _causal_mask(T, q.device, q.dtype)
        kt = k.transpose(1, 2)
        vt = v.transpose(1, 2)
        # recompute probabilities (identical op sequence to forward)
        if len(blocks) == 1:
            p = _softmax_inplace(torch.baddbmm(mask, q, kt, beta=1.0, alpha=scale))
            d_s = _softmax_grad_from_p(
                torch.bmm(grad_y, vt), p, grad_ent)
            d_v = torch.bmm(p.transpose(1, 2), grad_y)
            d_q = torch.bmm(d_s, k).mul_(scale)
            d_k = torch.bmm(d_s.transpose(1, 2), q).mul_(scale)
            return d_q, d_k, d_v, None, None
        d_q = torch.empty_like(q)
        d_k = torch.zeros_like(k)
        d_v = torch.zeros_like(v)
        for i0, i1 in blocks:
            p = _softmax_inplace(torch.baddbmm(mask[i0:i1, :i1], q[:, i0:i1],
                                               kt[:, :, :i1], beta=1.0, alpha=scale))
            g = grad_y[:, i0:i1]
            d_s = _softmax_grad_from_p(torch.bmm(g, vt[:, :, :i1]), p,
                                       grad_ent[:, i0:i1])
            d_v[:, :i1].add_(torch.bmm(p.transpose(1, 2), g))
            d_q[:, i0:i1] = torch.bmm(d_s, k[:, :i1]).mul_(scale)
            d_k[:, :i1].add_(torch.bmm(d_s.transpose(1, 2), q[:, i0:i1]).mul_(scale))
        return d_q, d_k, d_v, None, None


def _softmax_grad_from_p(d_p: torch.Tensor, p: torch.Tensor,
                         grad_ent: torch.Tensor) -> torch.Tensor:
    """Score gradient for ``p = softmax(s)`` plus the calibrated-entropy term,
    both read off the recomputed probabilities.  Consumes ``d_p``.
    """
    if grad_ent is not None:
        mask = p > _EPS
        logterm = p.clamp_min(_EPS).log_()
        logterm.add_(mask).mul_(mask)
        d_p.addcmul_(logterm, grad_ent.unsqueeze(-1), value=-1.0)
    d_p.mul_(p)
    return d_p.addcmul_(p, d_p.sum(dim=-1, keepdim=True), value=-1.0)


def _null_col(x: torch.Tensor, B: int) -> torch.Tensor:
    """``(h, d)`` -> ``(B*h, d, 1)``: batch-replicated key column."""
    h, d = x.shape
    return x.view(1, h, d).expand(B, h, d).reshape(B * h, d, 1)


def _null_row(x: torch.Tensor, B: int) -> torch.Tensor:
    """``(h, d)`` -> ``(B*h, 1, d)``: batch-replicated value row."""
    h, d = x.shape
    return x.view(1, h, d).expand(B, h, d).reshape(B * h, 1, d)


def _null_probs(q_blk: torch.Tensor, kt: torch.Tensor, i0: int, i1: int,
                mask: torch.Tensor, nkt: torch.Tensor, scale: float):
    """Joint weights of the causally visible tokens and of the null slot for
    the query block ``[i0, i1)``.

    The null score is the rank-1 term ``q . null_k``, so the score matrix
    keeps the width of the real (causally masked) part.  Both are shifted by
    the joint row maximum before the exp, which makes the pair ``(p, p0)`` an
    ordinary softmax over the ``i1 + 1`` columns the null slot adds to the
    block.
    """
    s = torch.baddbmm(mask[i0:i1, :i1], q_blk, kt[:, :, :i1], beta=1.0, alpha=scale)
    s0 = torch.bmm(q_blk, nkt).mul_(scale)
    m = torch.maximum(s.amax(dim=-1, keepdim=True), s0)
    s.sub_(m).exp_()
    e0 = s0.sub_(m).exp_()
    d = e0.add(s.sum(dim=-1, keepdim=True))
    return s.div_(d), e0.div_(d)


class _NullAttn(torch.autograd.Function):
    """Attention against the real keys plus one learnable null slot per head.

    The null slot is one extra score that is *not* causally masked, so it
    competes with the real tokens at every position and "choosing" it
    suppresses the attention write.  Scoring it as ``q . null_k`` keeps the
    score matrix rectangular in the real keys only, and - as for
    ``_CalibAttn`` - the backward pass rebuilds the ``T + 1``-way weights one
    query block at a time.
    """

    @staticmethod
    def forward(ctx, q, k, v, null_k, null_v, scale, B):
        bh, T = q.size(0), q.size(1)
        h = null_k.size(0)
        blocks = _query_blocks(bh, T)
        mask = _causal_mask(T, q.device, q.dtype)
        kt = k.transpose(1, 2)
        nkt = _null_col(null_k, B)
        nvw = _null_row(null_v, B)
        multi = len(blocks) > 1
        y = q.new_empty(bh, T, v.size(-1)) if multi else None
        for i0, i1 in blocks:
            p, p0 = _null_probs(q[:, i0:i1], kt, i0, i1, mask, nkt, scale)
            out = torch.bmm(p, v[:, :i1]).addcmul_(p0, nvw)
            if multi:
                y[:, i0:i1] = out
            else:
                y = out
        ctx.save_for_backward(q, k, v, null_k, null_v)
        ctx.scale = scale
        ctx.shape = (B, h)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        q, k, v, null_k, null_v = ctx.saved_tensors
        scale = ctx.scale
        B, h = ctx.shape
        bh, T = q.size(0), q.size(1)
        dk = null_k.size(-1)
        blocks = _query_blocks(bh, T)
        mask = _causal_mask(T, q.device, q.dtype)
        kt = k.transpose(1, 2)
        vt = v.transpose(1, 2)
        nkt = _null_col(null_k, B)
        nvw = _null_row(null_v, B)
        nkw = nkt.transpose(1, 2)
        multi = len(blocks) > 1
        d_q = torch.empty_like(q) if multi else None
        d_k = torch.zeros_like(k) if multi else None
        d_v = torch.zeros_like(v) if multi else None
        d_null_k = torch.zeros_like(null_k)
        d_null_v = torch.zeros_like(null_v)
        for i0, i1 in blocks:
            # rebuild the joint weights of this block only
            q_blk = q[:, i0:i1]
            g = grad_y[:, i0:i1]
            p, p0 = _null_probs(q_blk, kt, i0, i1, mask, nkt, scale)
            # sum_joint p * d_p = <grad_out, p @ v> = <grad_out, y>: the
            # denominator term follows from the rebuilt weights alone
            e = torch.bmm(g, vt[:, :, :i1])
            gu0 = torch.bmm(g, nvw.mT)
            dot = (p * e).sum(-1, keepdim=True).add_(p0 * gu0)
            d_u = e.sub_(dot)
            gu0.sub_(dot)
            dv = torch.bmm(p.transpose(1, 2), g)
            if multi:
                d_v[:, :i1].add_(dv)
            else:
                d_v = dv
            d_s = d_u.mul_(p).mul_(scale)
            d_s0 = gu0.mul_(p0).mul_(scale)
            dq = torch.bmm(d_s, k[:, :i1]).addcmul_(d_s0, nkw)
            if multi:
                d_q[:, i0:i1] = dq
            else:
                d_q = dq
            dkb = torch.bmm(d_s.transpose(1, 2), q_blk)
            if multi:
                d_k[:, :i1].add_(dkb)
            else:
                d_k = dkb
            # the null slot is a per-head parameter replicated across the batch
            q4 = q_blk.view(B, h, i1 - i0, dk)
            d_null_k.add_((d_s0.view(B, h, -1, 1) * q4).sum((0, 2)))
            d_null_v.add_((g.view(B, h, -1, dk) * p0.view(B, h, -1, 1)).sum((0, 2)))
        return d_q, d_k, d_v, d_null_k, d_null_v, None, None


class Attention(nn.Module):
    __slots__ = ("cfg", "gate_kind", "h", "dk", "scale", "inv_scale",
                 "qkv", "proj", "attn_dropout", "gate", "calib_a", "calib_c",
                 "calib_tau", "calib_scale", "null_k", "null_v", "max_len")
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.gate_kind = cfg.gate
        self.h = cfg.n_heads
        self.dk = cfg.d_model // cfg.n_heads
        self.scale = math.sqrt(self.dk)
        self.inv_scale = 1.0 / self.scale
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.max_len = cfg.max_len

        if cfg.gate in ("query", "calib"):
            # The gate is the only gate-variant-specific module in the model.
            # Drawing it from the global RNG advances the stream that
            # ``CAGE.apply(_init)`` consumes afterwards, so ``query``/``calib``
            # would start from different token embeddings, attention and MLP
            # weights than ``none``/``null`` - E1 is a *controlled* comparison,
            # so that confound would be indistinguishable from the effect of
            # the gate.  Draw it from a forked copy of the CPU generator: same
            # initialisation, shared stream left where it was.
            with torch.random.fork_rng(devices=[]):
                self.gate = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
                nn.init.normal_(self.gate.weight, std=0.02)
                nn.init.constant_(self.gate.bias,
                                  2.0 if cfg.gate == "query" else 6.0)
            self.gate._is_gate = True
        else:
            self.gate = None
        if cfg.gate == "calib":
            self.calib_a = nn.Parameter(torch.tensor(float(cfg.calib_a)))
            self.calib_c = nn.Parameter(torch.tensor(float(cfg.calib_c)))
            tau_init = math.log(math.expm1(cfg.calib_tau_init))
            self.calib_tau = nn.Parameter(torch.tensor(tau_init))
            self.calib_scale = nn.Parameter(torch.tensor(float(cfg.calib_scale)))
            # calib_tau initialized so softplus(tau) + 0.1 = calib_tau_init initially:
            # softplus(log(expm1(calib_tau_init))) = log(1 + exp(log(expm1(calib_tau_init)))) = log(1 + expm1(calib_tau_init)) = log(exp(calib_tau_init)) = calib_tau_init
            # calib_tau_init + 0.1 = target initial temperature
        else:
            self.calib_a = self.calib_c = self.calib_tau = self.calib_scale = None
        if cfg.gate == "null":
            self.null_k = nn.Parameter(torch.zeros(self.h, self.dk))
            self.null_v = nn.Parameter(torch.zeros(self.h, self.dk))
        else:
            self.null_k = self.null_v = None

    def _row_stats(self, rows: torch.Tensor, T: int, is_null: bool):
        """Entropy / max-prob / null-mass for a ``(B, H, S)`` slice of the
        attention distribution.  ``S`` is ``T`` or ``T + 1``, and in the
        ``T + 1`` case the null slot is the *leading* column - the layout both
        the block-wise kernel and the explicit path build.

        The rows always come from the *clean* softmax (attention dropout is
        never applied to them): a dropped row is scaled by the keep-prob and
        therefore is not a distribution any more, which would bias the
        entropy by ``keep*(log keep - 1)`` if it were read as one.
        """
        if is_null:
            # the real-token mass no longer sums to one once the null slot
            # competes, so it has to be renormalised before it means anything
            null_mass = rows[..., 0]
            pr = rows[..., 1:]
            pr = pr / pr.sum(-1, keepdim=True).clamp_min(_EPS)
        else:
            null_mass = None
            pr = rows
        return _RowEntropy.apply(pr, _EPS), pr.max(-1).values, null_mass

    def _row_stats_at(self, q, k, B, h, T, stat_pos, is_null, ent_full=None):
        """Diagnostics for the queried rows only.

        Only the rows at ``stat_pos`` are scored.  ``q`` is ``(B, h, T, dk)``,
        ``k`` the head-major key tensor.  ``ent_full`` optionally supplies the
        full-sequence entropy ``(B, h, T)`` already produced by the calibrated
        kernel, so the entropy of the selected rows is sliced instead of
        recomputed.
        """
        dk = q.size(-1)
        cols = T + 1 if is_null else T
        q_sel = _pick_rows(q, stat_pos).unsqueeze(1).reshape(-1, 1, dk)
        mask_sel = _mask_rows(stat_pos, T, h, q.dtype)
        att_sel = torch.baddbmm(mask_sel, q_sel, k.transpose(1, 2),
                                beta=1.0, alpha=self.inv_scale)
        if is_null:
            s0 = torch.bmm(q_sel, _null_col(self.null_k, B)).mul_(self.inv_scale)
            att_sel = torch.cat([s0, att_sel], dim=-1)
        probs = torch.softmax(att_sel, -1).view(B, h, cols)
        if ent_full is not None and not is_null:
            return _pick_rows(ent_full, stat_pos), probs.max(-1).values, None
        return self._row_stats(probs, T, is_null)

    def forward(self, x, stat_pos: torch.Tensor | None = None):
        B, T, C = x.shape
        h, dk = self.h, self.dk
        bh = B * h
        need_stats = stat_pos is not None
        kind = self.gate_kind
        is_null = kind == "null"
        is_calib = kind == "calib"
        drop_p = self.cfg.dropout
        dropping = self.training and drop_p > 0.0

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        ent = None
        if not dropping:
            # [fixed design, docs/design_notes.md D1] All four gate variants
            # share ONE hand-written kernel family: E1 is a controlled
            # comparison, so the numerics must not depend on the gate variant.
            # The calibrated kernel emits the row entropy as a by-product.
            # Do NOT switch any variant to a fused/flash attention call.
            qb = q.view(B, T, h, dk).transpose(1, 2).reshape(bh, T, dk)
            kb = k.view(B, T, h, dk).transpose(1, 2).reshape(bh, T, dk)
            vb = v.view(B, T, h, dk).transpose(1, 2).reshape(bh, T, dk)
            del qkv, q, k, v
            if is_null:
                y = _NullAttn.apply(qb, kb, vb, self.null_k, self.null_v,
                                    self.inv_scale, B)
            else:
                # none / query / calib run through the same kernel.
                y, ent = _CalibAttn.apply(qb, kb, vb, self.inv_scale, _EPS)
                ent = ent.view(B, h, T)
            y = y.view(B, h, T, dk).transpose(1, 2).reshape(B, T, C)
            stats = None
            if need_stats:
                stats = self._row_stats_at(
                    qb.view(B, h, T, dk), kb, B, h, T, stat_pos, is_null,
                    ent_full=None if is_null else ent)
        else:
            # attention dropout acts on the probabilities; the statistics are
            # read from the clean softmax rows before the dropout.
            q_ = q.view(B, T, h, dk).transpose(1, 2)
            k_ = k.view(B, T, h, dk).transpose(1, 2)
            v_ = v.view(B, T, h, dk).transpose(1, 2)
            mask = _causal_mask(T, x.device, x.dtype)
            if is_null:
                nk = self.null_k.view(1, h, 1, dk).expand(B, h, 1, dk)
                kv_k = torch.cat([nk, k_], dim=2)
                att = torch.matmul(q_, kv_k.transpose(-2, -1)).mul_(self.inv_scale)
                att[..., 1:].add_(mask)
                vb = v_.reshape(bh, T, dk)
                del qkv, q, k, v, nk, kv_k
                p = _SoftmaxInPlace.apply(att)
                if not need_stats:
                    stats = None
                else:
                    stats = self._row_stats(
                        _pick_rows(p.view(B, h, T, -1), stat_pos), T, True)
                if dropping:
                    p = self.attn_dropout(p)
                nv = _null_row(self.null_v, B)
                pb = p.view(bh, T, T + 1)
                y = torch.bmm(pb[..., 1:], vb).addcmul_(pb[..., :1], nv)
                y = y.view(B, h, T, dk).transpose(1, 2).reshape(B, T, C)
            else:
                qb = q_.reshape(bh, T, dk)
                kb = k_.reshape(bh, T, dk)
                vb = v_.reshape(bh, T, dk)
                del qkv, q, k, v
                att = torch.baddbmm(mask, qb, kb.transpose(1, 2),
                                    beta=1.0, alpha=self.inv_scale)
                p = _SoftmaxInPlace.apply(att)
                if is_calib:
                    ent = _RowEntropy.apply(p, _EPS).view(B, h, T)
                if not need_stats:
                    stats = None
                elif is_calib:
                    rows = _pick_rows(p.view(B, h, T, T), stat_pos)
                    stats = (_pick_rows(ent, stat_pos), rows.max(-1).values, None)
                else:
                    stats = self._row_stats(
                        _pick_rows(p.view(B, h, T, -1), stat_pos), T, False)
                if dropping:
                    p = self.attn_dropout(p)
                y = torch.bmm(p, vb).view(B, h, T, dk)
                y = y.transpose(1, 2).reshape(B, T, C)

        # ---- gating ----
        gate_val = None
        if kind == "query":
            gate_val = self.gate(x).sigmoid_()
            y.mul_(gate_val)
        elif is_calib:
            log_len = math.log(max(T, 2))
            conc = ent.mean(dim=1).unsqueeze(-1).div_(log_len)
            gate_val = self.gate(x)
            gate_val.sub_(self.calib_a * log_len)
            gate_val.addcmul_(conc, self.calib_c * self.calib_scale, value=-1.0)
            tau = F.softplus(self.calib_tau).add_(0.1)
            gate_val.div_(tau).sigmoid_()
            y.mul_(gate_val)

        out = self.proj(y)
        if not need_stats:
            return out, None

        gate_stat = None
        if gate_val is not None:
            gate_stat = _pick_tokens(gate_val, stat_pos).detach()
        return out, {
            "entropy": stats[0].detach(),
            "maxprob": None if stats[1] is None else stats[1].detach(),
            "gate": gate_stat,
            "null_mass": None if stats[2] is None else stats[2].detach(),
        }


class _GeluLinear(nn.Linear):
    """``nn.Linear`` fed the *pre-activation*: applies GELU before the
    projection.
    """

    def forward(self, h):
        return F.linear(F.gelu(h), self.weight, self.bias)


class Block(nn.Module):
    __slots__ = ("cfg", "layer_idx", "post", "ln1", "ln2", "attn", "mlp", "ls1", "ls2")
    def __init__(self, cfg: Cfg, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        use_post = cfg.norm == "post" or (cfg.norm == "mix" and layer_idx >= cfg.n_layers // 2)
        self.post = use_post
        self.ln1 = make_norm(cfg.norm, cfg.d_model)
        self.ln2 = make_norm(cfg.norm, cfg.d_model)
        self.attn = Attention(cfg)
        # the GELU is folded into the second projection (see _GeluLinear)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            _GeluLinear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        if cfg.layer_scale is not None:
            ls_val = float(cfg.layer_scale)
            self.ls1 = nn.Parameter(torch.full((cfg.d_model,), ls_val))
            self.ls2 = nn.Parameter(torch.full((cfg.d_model,), ls_val))
        else:
            self.ls1 = self.ls2 = None

    def forward(self, x, stat_pos=None):
        ls1, ls2 = self.ls1, self.ls2
        if self.post:
            a, s = self.attn(x, stat_pos)
            if ls1 is None:
                x = x + a
            else:
                x = torch.addcmul(x, a, ls1)
            x = self.ln1(x)
            if ls2 is None:
                x = x + self.mlp(x)
            else:
                x = torch.addcmul(x, self.mlp(x), ls2)
            x = self.ln2(x)
        else:
            x_norm = self.ln1(x)
            a, s = self.attn(x_norm, stat_pos)
            if ls1 is None:
                x = x + a
            else:
                x = torch.addcmul(x, a, ls1)
            x_norm = self.ln2(x)
            m = self.mlp(x_norm)
            if ls2 is None:
                x = x + m
            else:
                x = torch.addcmul(x, m, ls2)
        return x, s


class CAGE(nn.Module):
    __slots__ = ("cfg", "tok", "pos", "drop", "blocks", "ln_f", "head")
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.ln_f = make_norm(cfg.norm, cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            # The gate was already given its weight and its "start open" bias
            # in Attention.__init__ under a forked RNG.  Re-drawing it here
            # would move the shared stream that this walk consumes, so the
            # four gate variants would stop being init-matched; see
            # Attention.__init__.
            if hasattr(m, "_is_gate") and m._is_gate:
                return
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, stat_pos=None, sup_pos=None):
        """``(logits, loss, layer_stats)``.

        Without ``targets`` this is the eval path and ``logits`` covers the
        last position only.  With ``targets`` it is the training path and
        ``logits`` covers exactly the supervised positions (``targets != -1``),
        in row-major order; the ignored ones carry no gradient.

        ``sup_pos`` optionally gives the supervised column of each row; it
        selects the same rows the ``!= -1`` scan would (``-1`` entries are
        ignored by the loss either way).
        """
        T = idx.size(1)
        if T > self.cfg.max_len:
            raise ValueError(
                f"sequence length {T} exceeds cfg.max_len={self.cfg.max_len}; "
                f"n_pairs={max(0, (T - 2) // 2)} needs {T} positions"
            )
        sup = None
        if targets is not None:
            flat_t = targets.reshape(-1)
            if sup_pos is not None:
                rows = torch.arange(targets.size(0), device=targets.device,
                                    dtype=torch.long)
                sup = rows.mul(targets.size(1)).add_(sup_pos)
            else:
                sup = (flat_t != -1).nonzero(as_tuple=True)[0]
                if sup.numel() == flat_t.numel():
                    sup = None
        x = self.tok(idx)
        x.add_(self.pos.weight[:T])
        x = self.drop(x)
        layer_stats = [] if stat_pos is not None else None
        for blk in self.blocks:
            if self.cfg.use_gradient_checkpointing:
                x, s = torch.utils.checkpoint.checkpoint(
                    blk, x, stat_pos, use_reentrant=False)
            else:
                x, s = blk(x, stat_pos)
            if layer_stats is not None:
                s["resid_norm"] = x.detach().norm(dim=-1).mean(-1)
                layer_stats.append(s)
        if targets is not None:
            if sup is not None:
                flat_t = flat_t.index_select(0, sup)
                h = self.ln_f(x.reshape(-1, self.cfg.d_model).index_select(0, sup))
            else:
                h = self.ln_f(x).reshape(-1, self.cfg.d_model)
            logits = self.head(h)
            loss = F.cross_entropy(logits, flat_t, ignore_index=-1)
        else:
            logits = self.head(self.ln_f(x[:, -1:]))
            loss = None
        return logits, loss, layer_stats


def build(cfg_dict: dict, compile: bool = False) -> CAGE:
    known = {f.name for f in Cfg.__dataclass_fields__.values()}
    unknown = sorted(set(cfg_dict) - known)
    if unknown:
        raise ValueError(
            f"unknown config key(s) {unknown}; valid keys are {sorted(known)}"
        )
    model = CAGE(Cfg(**cfg_dict))
    if compile:
        if hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        else:
            print("Warning: torch.compile not available, skipping compilation")
    return model