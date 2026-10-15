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
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


_EPS = 1e-9

_zero_scalar_cache: dict = {}

def _zero_scalar(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Cached zero scalar for the given device and dtype."""
    key = (device, dtype)
    z = _zero_scalar_cache.get(key)
    if z is None:
        with torch.inference_mode(False):
            z = torch.zeros((), device=device, dtype=dtype)
        _zero_scalar_cache[key] = z
    return z


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
    gate: str = "none"
    norm: str = "pre"
    layer_scale: float | None = None
    calib_a: float = 1.0
    calib_c: float = 1.0
    calib_scale: float = 1.0
    calib_tau_init: float = 0.9
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
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=True)

    def forward(self, x):
        return self.ln(x)


class SeeDNorm(nn.Module):
    """Dynamic normalization: per-token scale predicted from hidden state."""
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale = nn.Linear(dim, dim)
        nn.init.zeros_(self.scale.bias)
        nn.init.normal_(self.scale.weight, std=0.02)

    def forward(self, x):
        s = self.scale(x).sigmoid_()
        return self.ln(x).mul_(s).mul_(2.0)


def make_norm(kind: str, dim: int) -> nn.Module:
    return SeeDNorm(dim) if kind == "seednorm" else Norm(dim)


_MAX_MASK_CACHE = 64

_causal_mask_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()


def _causal_mask(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (T, device, dtype)
    m = _causal_mask_cache.get(key)
    if m is not None:
        _causal_mask_cache.move_to_end(key)
        return m
    with torch.inference_mode(False):
        m = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
        m.triu_(1)
    _causal_mask_cache[key] = m
    if len(_causal_mask_cache) > _MAX_MASK_CACHE:
        _causal_mask_cache.popitem(last=False)
    return m


_MAX_BOOL_CACHE = 64
_causal_bool_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()


def _causal_bool_mask(T: int, device: torch.device) -> torch.Tensor:
    key = (T, device)
    m = _causal_bool_cache.get(key)
    if m is not None:
        _causal_bool_cache.move_to_end(key)
        return m
    with torch.inference_mode(False):
        m = torch.ones(T, T + 1, dtype=torch.bool, device=device)
        m[:, 0] = False
        m[:, 1:].triu_(1)
    _causal_bool_cache[key] = m
    if len(_causal_bool_cache) > _MAX_BOOL_CACHE:
        _causal_bool_cache.popitem(last=False)
    return m


_ARANGE_CACHE_MAX = 128
_arange_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()


def _arange(n: int, device: torch.device) -> torch.Tensor:
    """Read-only ``arange(n)`` shared across calls (index tensors only).

    Allocated with inference mode *off* so one cached entry serves every
    context, including the training-mode forward.  LRU eviction.
    """
    key = (n, device)
    a = _arange_cache.get(key)
    if a is not None:
        _arange_cache.move_to_end(key)
        return a
    with torch.inference_mode(False):
        a = torch.arange(n, device=device)
    _arange_cache[key] = a
    if len(_arange_cache) > _ARANGE_CACHE_MAX:
        _arange_cache.popitem(last=False)
    return a


def _mask_rows(pos: torch.Tensor, T: int, h: int, dtype: torch.dtype) -> torch.Tensor:
    """``(B*h, 1, T)`` additive mask rows for the query positions ``pos``.

    Row ``b`` is 0 up to column ``pos[b]`` and ``-inf`` beyond it.  The mask
    depends on the *values* of ``pos``, so it is rebuilt per call.
    """
    B = pos.size(0)
    cols = _arange(T, pos.device)
    m = _get_scratch(("mask_rows",), pos.device, dtype, B * h * T)
    m = m[:B * h * T].view(B, h, T)
    m.zero_()
    mb = _get_scratch(("mask_rows_b",), pos.device, torch.bool, B * T)
    mb = mb[:B * T].view(B, 1, T)
    torch.gt(cols.view(1, 1, T), pos.view(B, 1, 1), out=mb)
    m.masked_fill_(mb, float("-inf"))
    return m.view(B * h, 1, T)


def _softmax_inplace(s: torch.Tensor) -> torch.Tensor:
    """Row softmax written back into ``s`` (no autograd node)."""
    return torch.softmax(s, dim=-1, out=s)


class _SoftmaxInPlace(torch.autograd.Function):
    """``softmax(x, -1)`` written back into ``x``."""

    @staticmethod
    def forward(ctx, x):
        ctx.mark_dirty(x)
        torch.softmax(x, dim=-1, out=x)
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        if not saved:
            return None
        (y,) = saved
        if grad_out is None:
            return None
        gy = grad_out.mul(y)
        return gy.addcmul_(y, gy.sum(-1, keepdim=True), value=-1.0)


class _RowEntropy(torch.autograd.Function):
    """``(p * p.clamp_min(eps).log()).sum(-1).neg()`` computed directly."""

    @staticmethod
    def forward(ctx, p, eps: float):
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(p)
        ctx.eps = eps
        buf = _get_scratch(("rowent",), p.device, p.dtype, p.numel())
        return _row_entropy(p, eps, buf=buf)

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        if not saved:
            return None, None
        (p,) = saved
        if grad_out is None:
            return None, None
        mask = _get_scratch(("rowent_bwd_m",), p.device, torch.bool,
                            p.numel())
        mask = mask[:p.numel()].view_as(p)
        torch.gt(p, ctx.eps, out=mask)
        d = p.clamp_min(ctx.eps)
        d.log_()
        d.add_(mask).mul_(mask).mul_(grad_out.unsqueeze(-1)).neg_()
        return d, None


def _pick_rows(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """``x[b, :, pos[b], ...]`` for every ``b``: ``(B, H, T, S) -> (B, H, S)``."""
    return x[_arange(x.size(0), x.device), :, pos]


def _pick_tokens(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """``x[b, pos[b], ...]`` for every ``b``: ``(B, T, C) -> (B, C)``."""
    return x[_arange(x.size(0), x.device), pos]


_query_blocks_cache: OrderedDict = OrderedDict()
_QUERY_BLOCKS_CACHE_MAX = 512


_scratch_cache: dict[tuple, torch.Tensor] = {}


def _get_scratch(key: tuple, device: torch.device, dtype: torch.dtype, numel: int) -> torch.Tensor:
    name = key[0] if isinstance(key, tuple) else key
    ckey = (name, device, dtype)
    entry = _scratch_cache.get(ckey)
    if entry is not None and entry.numel() >= numel:
        return entry[:numel]
    with torch.inference_mode(False):
        buf = torch.empty(numel, device=device, dtype=dtype)
    _scratch_cache[ckey] = buf
    return buf[:numel]


def _query_blocks(bh: int, T: int):
    """``[(start, stop), ...]`` query blocks for a ``bh``-fold batch of ``T``."""
    key = (bh, T)
    blocks = _query_blocks_cache.get(key)
    if blocks is not None:
        _query_blocks_cache.move_to_end(key)
        return blocks
    if T <= 32:
        blocks = ((0, T),)
    else:
        max_rows = (1 << 20) // (bh * T)
        if max_rows < 32:
            max_rows = 32
        rows = min(T, max_rows)
        blocks = tuple((i, min(i + rows, T)) for i in range(0, T, rows))
    _query_blocks_cache[key] = blocks
    if len(_query_blocks_cache) > _QUERY_BLOCKS_CACHE_MAX:
        _query_blocks_cache.popitem(last=False)
    return blocks


def _row_entropy(p: torch.Tensor, eps: float,
                 out: torch.Tensor | None = None,
                 buf: torch.Tensor | None = None) -> torch.Tensor:
    """``-(p * log p).sum(-1)`` computed row-wise.

    ``buf`` optionally supplies the ``clamp_min`` workspace and must be a
    flat tensor of at least ``p.numel()`` elements.
    """
    if buf is None:
        t = p.clamp_min(eps)
    else:
        t = buf[:p.numel()].view_as(p)
        torch.clamp_min(p, eps, out=t)
    t.log_().mul_(p)
    if out is None:
        return t.sum(dim=-1).neg_()
    torch.sum(t, dim=-1, out=out)
    return out.neg_()


class _CalibAttn(torch.autograd.Function):
    """Calibrated attention (dropout-free).

    Computes, on the ``(B*H, T, *)`` layout:

        p   = softmax(q k^T / sqrt(dk) + causal)
        y   = p v
        ent = -(p * log p).rowsum

    The probabilities are rebuilt block by block: forward scores them one
    block at a time and backward recomputes them from ``q, k, v``.  The
    entropy output is only produced when ``need_ent`` is set (calibrated
    gate or diagnostics); the attention output itself is identical either
    way, so the kernel stays shared by all gate variants.
    """

    @staticmethod
    def forward(ctx, q, k, v, scale: float, eps: float, need_ent: bool = True):
        ctx.set_materialize_grads(False)
        bh, T = q.size(0), q.size(1)
        blocks = _query_blocks(bh, T)
        mask = _causal_mask(T, q.device, q.dtype)
        kt = k.transpose(1, 2)
        dv = v.size(-1)
        y = q.new_empty(bh, T, dv)
        rows0 = blocks[0][1] - blocks[0][0]
        max_rows = rows0
        max_i1 = T
        s = _get_scratch(("calib_s",), q.device, q.dtype, bh * max_rows * max_i1)
        if len(blocks) == 1:
            p = _softmax_inplace(torch.baddbmm(
                mask, q, kt, beta=1.0, alpha=scale,
                out=s[:bh * T * T].view(bh, T, T)))
            torch.bmm(p, v, out=y)
            if need_ent:
                ent = q.new_empty(bh, T)
                _row_entropy(p, eps, out=ent,
                             buf=_get_scratch(("calib_f",), q.device, q.dtype, bh * T * T))
            else:
                ent = None
        else:
            ent = q.new_empty(bh, T) if need_ent else None
            ebuf = _get_scratch(("calib_f",), q.device, q.dtype, bh * max_rows * max_i1) if need_ent else None
            yb = _get_scratch(("calib_yblk",), q.device, q.dtype, bh * max_rows * dv)
            for i0, i1 in blocks:
                r = i1 - i0
                p = _softmax_inplace(torch.baddbmm(
                    mask[i0:i1, :i1], q[:, i0:i1], kt[:, :, :i1],
                    beta=1.0, alpha=scale,
                    out=s[:bh * r * i1].view(bh, r, i1)))
                yblk = yb[:bh * r * dv].view(bh, r, dv)
                torch.bmm(p, v[:, :i1], out=yblk)
                y[:, i0:i1].copy_(yblk)
                if need_ent:
                    _row_entropy(p, eps, out=ent[:, i0:i1], buf=ebuf)
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(q, k, v)
        else:
            ctx.qkv_saved = False
        ctx.scale = scale
        ctx.eps = eps
        return y, ent

    @staticmethod
    def backward(ctx, grad_y, grad_ent):
        saved = ctx.saved_tensors
        if not saved:
            return None, None, None, None, None, None
        q, k, v = saved
        scale, eps = ctx.scale, ctx.eps
        bh, T = q.size(0), q.size(1)
        if grad_y is None:
            grad_y = q.new_zeros(bh, T, v.size(-1))
        blocks = _query_blocks(bh, T)
        mask = _causal_mask(T, q.device, q.dtype)
        kt = k.transpose(1, 2)
        vt = v.transpose(1, 2)
        rows0 = blocks[0][1] - blocks[0][0]
        if len(blocks) == 1:
            s = _get_scratch(("calib_s",), q.device, q.dtype, bh * rows0 * T)
            p = _softmax_inplace(torch.baddbmm(
                mask, q, kt, beta=1.0, alpha=scale,
                out=s[:bh * T * T].view(bh, T, T)))
            n = p.numel()
            gbuf = mbuf = None
            if grad_ent is not None:
                gbuf = _get_scratch(("calib_f",), q.device, q.dtype, n)
                mbuf = _get_scratch(("calib_bwd_mbuf",), q.device, torch.bool, n)
            d_p = _get_scratch(("calib_dp",), q.device, q.dtype, n).view(bh, T, T)
            torch.bmm(grad_y, vt, out=d_p)
            d_s = _softmax_grad_from_p(d_p, p, grad_ent, buf=gbuf, mbuf=mbuf)
            d_v = q.new_empty(bh, T, v.size(-1))
            torch.bmm(p.transpose(1, 2), grad_y, out=d_v)
            zero = _zero_scalar(d_s.device, d_s.dtype)
            d_q = q.new_empty(bh, T, q.size(-1))
            torch.baddbmm(zero, d_s, k, beta=0.0, alpha=scale, out=d_q)
            d_k = k.new_empty(bh, T, k.size(-1))
            torch.baddbmm(zero, d_s.transpose(1, 2), q, beta=0.0, alpha=scale,
                          out=d_k)
            return d_q, d_k, d_v, None, None, None
        d_q = torch.empty_like(q)
        d_k = torch.empty_like(k)
        d_v = torch.empty_like(v)
        max_rows = rows0
        max_i1 = T
        dk = q.size(-1)
        s = _get_scratch(("calib_s",), q.device, q.dtype, bh * max_rows * max_i1)
        gbuf = _get_scratch(("calib_f",), q.device, q.dtype, bh * max_rows * max_i1) if grad_ent is not None else None
        mbuf = _get_scratch(("calib_bwd_mbuf",), q.device, torch.bool, bh * max_rows * max_i1) if grad_ent is not None else None
        dpb = _get_scratch(("calib_dp",), q.device, q.dtype, bh * max_rows * max_i1)
        tmpb = _get_scratch(("calib_bwd_tmp",), q.device, q.dtype, bh * T * dk)
        dqb = _get_scratch(("calib_bwd_dqb",), q.device, q.dtype,
                           bh * max_rows * dk)
        zero = _zero_scalar(q.device, q.dtype)
        first = True
        for i0, i1 in reversed(blocks):
            r = i1 - i0
            n = bh * r * i1
            p = _softmax_inplace(torch.baddbmm(
                mask[i0:i1, :i1], q[:, i0:i1], kt[:, :, :i1],
                beta=1.0, alpha=scale,
                out=s[:n].view(bh, r, i1)))
            g = grad_y[:, i0:i1]
            ge = None if grad_ent is None else grad_ent[:, i0:i1]
            d_p = dpb[:n].view(bh, r, i1)
            torch.bmm(g, vt[:, :, :i1], out=d_p)
            d_s = _softmax_grad_from_p(d_p, p, ge,
                                       buf=None if gbuf is None else gbuf[:n],
                                       mbuf=None if mbuf is None else mbuf[:n])
            tmp = tmpb[:bh * i1 * dk].view(bh, i1, dk)
            if first:
                torch.bmm(p.transpose(1, 2), g, out=d_v)
            else:
                torch.bmm(p.transpose(1, 2), g, out=tmp)
                d_v[:, :i1].add_(tmp)
            torch.baddbmm(zero, d_s.transpose(1, 2), q[:, i0:i1], beta=0.0,
                          alpha=scale, out=d_k if first else tmp)
            if first:
                first = False
            else:
                d_k[:, :i1].add_(tmp)
            dqblk = dqb[:bh * r * dk].view(bh, r, dk)
            torch.baddbmm(zero, d_s, k[:, :i1], beta=0.0, alpha=scale,
                          out=dqblk)
            d_q[:, i0:i1].copy_(dqblk)
        return d_q, d_k, d_v, None, None, None


def _softmax_grad_from_p(d_p: torch.Tensor, p: torch.Tensor,
                         grad_ent: torch.Tensor | None,
                         buf: torch.Tensor | None = None,
                         mbuf: torch.Tensor | None = None) -> torch.Tensor:
    """Score gradient for ``p = softmax(s)`` plus the calibrated-entropy term,
    both read off the recomputed probabilities.  Consumes ``d_p``.

    ``buf`` optionally supplies a flat scratch for the ``log(p)`` term and
    ``mbuf`` one for the ``p > _EPS`` selector.

    The surviving term is ``log(p) + 1`` on ``p > _EPS`` and ``0`` elsewhere;
    the sub-eps entries are dropped by the trailing select, so clamping ``p``
    to ``_EPS`` in front of the ``log`` cannot change the result.
    """
    if grad_ent is not None:
        if buf is None:
            logterm = p.clamp_min(_EPS).log_()
            mask = p > _EPS
        else:
            logterm = buf[:p.numel()].view_as(p)
            torch.clamp_min(p, _EPS, out=logterm)
            logterm.log_()
            if mbuf is None:
                mask = p > _EPS
            else:
                mask = mbuf[:p.numel()].view_as(p)
                torch.gt(p, _EPS, out=mask)
        logterm.add_(1.0)
        logterm.mul_(mask)
        d_p.addcmul_(logterm, grad_ent.unsqueeze(-1), value=-1.0)
    d_p.mul_(p)
    return d_p.addcmul_(p, d_p.sum(dim=-1, keepdim=True), value=-1.0)


def _null_probs(q_blk: torch.Tensor, kt: torch.Tensor, i0: int, i1: int,
                mask: torch.Tensor, null_k: torch.Tensor, B: int, h: int,
                scale: float, sbuf: torch.Tensor | None = None):
    n, r = q_blk.size(0), q_blk.size(1)
    if sbuf is None:
        s = torch.baddbmm(mask, q_blk, kt,
                          beta=1.0, alpha=scale)
    else:
        s = sbuf[:n * r * i1].view(n, r, i1)
        torch.baddbmm(mask, q_blk, kt,
                      beta=1.0, alpha=scale, out=s)
    s0 = _get_scratch(("null_s0",), q_blk.device, q_blk.dtype, n * r)
    s0 = s0[:n * r]
    torch.matmul(q_blk.view(B, h, r, -1), null_k.view(h, -1, 1),
                 out=s0.view(B, h, r, 1))
    s0mul = s0.view(n, r, 1)
    s0mul.mul_(scale)
    m = _get_scratch(("null_m",), q_blk.device, q_blk.dtype, n * r)
    m = m[:n * r].view(n, r, 1)
    torch.amax(s, dim=-1, keepdim=True, out=m)
    torch.maximum(m, s0mul, out=m)
    s.sub_(m).exp_()
    s0mul.sub_(m).exp_()
    torch.sum(s, dim=-1, keepdim=True, out=m).add_(s0mul)
    s.div_(m)
    s0mul.div_(m)
    return s, s0mul


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
        dv = v.size(-1)
        rows0 = blocks[0][1] - blocks[0][0]
        max_rows = rows0
        max_i1 = T
        y = q.new_empty(bh, T, dv)
        y4 = y.view(B, h, T, dv)
        nv4 = null_v.view(1, h, 1, dv)
        sbuf = _get_scratch(("null_s",), q.device, q.dtype, bh * max_rows * max_i1)
        yb = _get_scratch(("null_yblk",), q.device, q.dtype, bh * max_rows * dv)
        for i0, i1 in blocks:
            r = i1 - i0
            p, p0 = _null_probs(q[:, i0:i1], kt[:, :, :i1], i0, i1, mask[i0:i1, :i1], null_k, B, h,
                                scale, sbuf)
            yblk = yb[:bh * r * dv].view(bh, r, dv)
            torch.bmm(p, v[:, :i1], out=yblk)
            torch.addcmul(yblk.view(B, h, r, dv), p0.view(B, h, r, 1), nv4,
                          out=y4[:, :, i0:i1])
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(q, k, v, null_k, null_v)
        else:
            ctx.qkv_saved = False
        ctx.scale = scale
        ctx.shape = (B, h)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        saved = ctx.saved_tensors
        if not saved:
            return None, None, None, None, None, None, None
        q, k, v, null_k, null_v = saved
        scale = ctx.scale
        B, h = ctx.shape
        bh, T = q.size(0), q.size(1)
        if grad_y is None:
            grad_y = q.new_zeros(bh, T, v.size(-1))
        dk = null_k.size(-1)
        blocks = _query_blocks(bh, T)
        mask = _causal_mask(T, q.device, q.dtype)
        kt = k.transpose(1, 2)
        vt = v.transpose(1, 2)
        multi = len(blocks) > 1
        d_k = torch.empty_like(k) if multi else None
        d_v = torch.empty_like(v) if multi else None
        d_q = torch.empty_like(q)
        d_q4 = d_q.view(B, h, T, dk)
        nk4 = null_k.view(1, h, 1, dk)
        nv4t = null_v.view(1, h, dk, 1)
        rows0 = blocks[0][1] - blocks[0][0]
        max_rows = rows0
        max_i1 = T
        sbuf = _get_scratch(("null_s",), q.device, q.dtype, bh * max_rows * max_i1)
        peb = _get_scratch(("null_bwd_peb",), q.device, q.dtype, bh * max_rows * max_i1)
        eb = _get_scratch(("null_bwd_e",), q.device, q.dtype, bh * max_rows * max_i1)
        gub = _get_scratch(("null_bwd_gu",), q.device, q.dtype, bh * max_rows)
        dotb = _get_scratch(("null_bwd_dot",), q.device, q.dtype, bh * max_rows)
        tmp_rows = T if multi else 1
        tmpb = _get_scratch(("null_bwd_tmp",), q.device, q.dtype,
                            bh * tmp_rows * dk)
        dqb = _get_scratch(("null_bwd_dqb",), q.device, q.dtype,
                           bh * max_rows * dk) if multi else None
        acc_k = _get_scratch(("null_bwd_ak",), q.device, q.dtype, bh * dk).view(bh, 1, dk).zero_()
        acc_v = _get_scratch(("null_bwd_av",), q.device, q.dtype, bh * dk).view(bh, 1, dk).zero_()
        first = True
        for i0, i1 in reversed(blocks):
            q_blk = q[:, i0:i1]
            g = grad_y[:, i0:i1]
            r = i1 - i0
            p, p0 = _null_probs(q_blk, kt[:, :, :i1], i0, i1, mask[i0:i1, :i1],
                                null_k, B, h, scale, sbuf)
            n = bh * r * i1
            e = torch.bmm(g, vt[:, :, :i1], out=eb[:n].view(bh, r, i1))
            gu0 = gub[:bh * r].view(bh, r, 1)
            torch.matmul(g.view(B, h, r, -1), nv4t, out=gu0.view(B, h, r, 1))
            pe = peb[:n].view(bh, r, i1)
            torch.mul(p, e, out=pe)
            dot = dotb[:bh * r].view(bh, r, 1)
            torch.sum(pe, dim=-1, keepdim=True, out=dot)
            dot.addcmul_(p0, gu0)
            d_u = e.sub_(dot)
            gu0.sub_(dot)
            d_s = d_u.mul_(p).mul_(scale)
            d_s0 = gu0.mul_(p0).mul_(scale)
            if multi:
                tmp = tmpb[:bh * i1 * dk].view(bh, i1, dk)
                if first:
                    torch.bmm(p.transpose(1, 2), g, out=d_v)
                else:
                    torch.bmm(p.transpose(1, 2), g, out=tmp)
                    d_v[:, :i1].add_(tmp)
                torch.bmm(d_s.transpose(1, 2), q_blk,
                          out=d_k if first else tmp)
                if first:
                    first = False
                else:
                    d_k[:, :i1].add_(tmp)
                dqblk = dqb[:bh * r * dk].view(bh, r, dk)
                torch.bmm(d_s, k[:, :i1], out=dqblk)
                d_q[:, i0:i1].copy_(dqblk)
            else:
                d_v = q.new_empty(bh, i1, dk)
                d_k = q.new_empty(bh, i1, dk)
                torch.bmm(p.transpose(1, 2), g, out=d_v)
                torch.bmm(d_s.transpose(1, 2), q_blk, out=d_k)
                torch.bmm(d_s, k[:, :i1], out=d_q)
            d_q4[:, :, i0:i1].addcmul_(d_s0.view(B, h, r, 1), nk4)
            acc_blk = tmpb[:bh * dk].view(bh, 1, dk)
            torch.bmm(d_s0.transpose(-2, -1), q_blk, out=acc_blk)
            acc_k.add_(acc_blk)
            torch.bmm(p0.transpose(-2, -1), g, out=acc_blk)
            acc_v.add_(acc_blk)
        d_null_k = acc_k.view(B, h, dk).sum(0)
        d_null_v = acc_v.view(B, h, dk).sum(0)
        return d_q, d_k, d_v, d_null_k, d_null_v, None, None


class Attention(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.gate_kind = cfg.gate
        self.h = cfg.n_heads
        self.dk = cfg.d_model // cfg.n_heads
        self.inv_scale = 1.0 / math.sqrt(self.dk)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)

        if cfg.gate in ("query", "calib"):
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
            self._log_len_cache: dict[int, float] = {}
        else:
            self.calib_a = self.calib_c = self.calib_tau = self.calib_scale = None
            self._log_len_cache = None
        if cfg.gate == "null":
            self.null_k = nn.Parameter(torch.zeros(self.h, self.dk))
            self.null_v = nn.Parameter(torch.zeros(self.h, self.dk))
        else:
            self.null_k = self.null_v = None

    @staticmethod
    def _pack(t: torch.Tensor, B: int, h: int, T: int, dk: int) -> torch.Tensor:
        """``(B, T, h*dk) -> (B*h, T, dk)`` with one contiguous copy."""
        return t.view(B, T, h, dk).transpose(1, 2).contiguous().view(B * h, T, dk)

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
            null_mass = rows[..., 0]
            pr = rows[..., 1:]
            pr.div_(pr.sum(-1, keepdim=True).clamp_min(_EPS))
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
            s0 = torch.matmul(q_sel.reshape(B, h, 1, dk),
                              self.null_k.view(1, h, dk, 1)).reshape(
                                  q_sel.size(0), 1, 1).mul_(self.inv_scale)
            att_sel = torch.cat([s0, att_sel], dim=-1)
        probs = _softmax_inplace(att_sel).view(B, h, cols)
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

        q_lin, k_lin, v_lin = F.linear(x, self.qkv.weight).chunk(3, dim=-1)
        qb = self._pack(q_lin, B, h, T, dk)
        kb = self._pack(k_lin, B, h, T, dk)
        vb = self._pack(v_lin, B, h, T, dk)

        ent = None
        y4 = None
        stats = None
        if not dropping:
            if is_null:
                y4 = _NullAttn.apply(qb, kb, vb, self.null_k, self.null_v,
                                     self.inv_scale, B).view(B, h, T, dk)
            else:
                y, ent = _CalibAttn.apply(qb, kb, vb, self.inv_scale, _EPS,
                                          is_calib)
                y4 = y.view(B, h, T, dk)
                ent = None if ent is None else ent.view(B, h, T)
            if need_stats:
                stats = self._row_stats_at(
                    qb.view(B, h, T, dk), kb, B, h, T, stat_pos, is_null,
                    ent_full=ent)
        else:
            q_ = qb.view(B, h, T, dk)
            k_ = kb.view(B, h, T, dk)
            if is_null:
                nk = self.null_k.view(1, h, 1, dk).expand(B, h, 1, dk)
                kv_k = torch.cat([nk, k_], dim=2)
                att = torch.matmul(q_, kv_k.transpose(-2, -1)).mul_(self.inv_scale)
                att.masked_fill_(_causal_bool_mask(T, x.device),
                                 float("-inf"))
                p = _SoftmaxInPlace.apply(att)
                if not need_stats:
                    stats = None
                else:
                    stats = self._row_stats(
                        _pick_rows(p.view(B, h, T, -1), stat_pos), T, True)
                if dropping:
                    p = self.attn_dropout(p)
                pb = p.view(bh, T, T + 1)
                yb = torch.bmm(pb[..., 1:], vb).view(B, h, T, dk)
                y4 = yb.addcmul_(p.view(B, h, T, T + 1)[..., :1],
                                 self.null_v.view(1, h, 1, dk))
            else:
                mask = _causal_mask(T, x.device, x.dtype)
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
                y4 = torch.bmm(p, vb).view(B, h, T, dk)

        gate_val = None
        if kind == "query":
            gate_val = self.gate(x).sigmoid_()
        elif is_calib:
            log_len = self._log_len_cache.get(T)
            if log_len is None:
                log_len = math.log(max(T, 2))
                self._log_len_cache[T] = log_len
            gate_val = self.gate(x)
            gate_val.sub_(self.calib_a * log_len)
            calib_cs = self.calib_c * self.calib_scale
            conc = ent.mean(dim=1).div_(log_len).unsqueeze(-1)
            conc.mul_(calib_cs)
            gate_val.sub_(conc)
            tau = F.softplus(self.calib_tau).add_(0.1)
            gate_val = gate_val.div_(tau).sigmoid_()
            del ent
            ent = None

        if gate_val is None:
            y = y4.transpose(1, 2).reshape(B, T, C)
        else:
            y = torch.mul(gate_val.view(B, T, h, dk), y4.transpose(1, 2))
            y = y.reshape(B, T, C)

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
    def __init__(self, cfg: Cfg, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        use_post = cfg.norm == "post" or (cfg.norm == "mix" and layer_idx >= cfg.n_layers // 2)
        self.post = use_post
        self.ln1 = make_norm(cfg.norm, cfg.d_model)
        self.ln2 = make_norm(cfg.norm, cfg.d_model)
        self.attn = Attention(cfg)
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
                x = a.add_(x)
            else:
                x = torch.addcmul(x, a, ls1)
            x = self.ln1(x)
            if ls2 is None:
                x = self.mlp(x).add_(x)
            else:
                x = torch.addcmul(x, self.mlp(x), ls2)
            x = self.ln2(x)
        else:
            x_norm = self.ln1(x)
            a, s = self.attn(x_norm, stat_pos)
            if ls1 is None:
                x = a.add_(x)
            else:
                x = torch.addcmul(x, a, ls1)
            x_norm = self.ln2(x)
            m = self.mlp(x_norm)
            if ls2 is None:
                x = m.add_(x)
            else:
                x = torch.addcmul(x, m, ls2)
        return x, s


class CAGE(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout, inplace=True)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.ln_f = make_norm(cfg.norm, cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
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
                sup = _arange(targets.size(0), targets.device).mul(
                    targets.size(1)).add_(sup_pos)
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