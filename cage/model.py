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

_causal_mask_cache: dict[tuple, torch.Tensor] = {}
_causal_mask_order: list[tuple] = []


def _causal_mask(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (T, device, dtype)
    m = _causal_mask_cache.get(key)
    if m is not None:
        _causal_mask_order.remove(key)
        _causal_mask_order.append(key)
        return m
    with torch.inference_mode(False):
        m = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
        m.triu_(1)
    _causal_mask_cache[key] = m
    _causal_mask_order.append(key)
    if len(_causal_mask_cache) > _MAX_MASK_CACHE:
        old_key = _causal_mask_order.pop(0)
        _causal_mask_cache.pop(old_key, None)
    return m


_causal_bool_cache: dict[tuple, torch.Tensor] = {}
_causal_bool_order: list[tuple] = []
_MAX_BOOL_CACHE = 64


def _causal_bool_mask(T: int, device: torch.device) -> torch.Tensor:
    key = (T, device)
    m = _causal_bool_cache.get(key)
    if m is not None:
        _causal_bool_order.remove(key)
        _causal_bool_order.append(key)
        return m
    with torch.inference_mode(False):
        m = torch.ones(T, T + 1, dtype=torch.bool, device=device)
        m[:, 0] = False
        m[:, 1:].triu_(1)
    _causal_bool_cache[key] = m
    _causal_bool_order.append(key)
    if len(_causal_bool_cache) > _MAX_BOOL_CACHE:
        old_key = _causal_bool_order.pop(0)
        _causal_bool_cache.pop(old_key, None)
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
    depends on the *values* of ``pos``, so it is rebuilt per call; only the
    shared per-length causal mask is cached.  The workspace is scratch - the
    only consumer is the ``baddbmm`` right after the call, which copies the
    values out.
    """
    B = pos.size(0)
    cols = _arange(T, pos.device)
    m = _get_scratch(("mask_rows", B, h, T), pos.device, dtype, B * h * T)
    m = m[:B * h * T].view(B, h, T)
    m.zero_()
    # Inline comparison to avoid scratch allocation for the boolean mask
    # cols: (T,) -> (1, 1, T), pos: (B,) -> (B, 1, 1)
    # Result: (B, 1, T) boolean mask where True means mask with -inf
    mask_bool = cols.view(1, 1, T) > pos.view(B, 1, 1)
    m.masked_fill_(mask_bool, float("-inf"))
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
        buf = _get_scratch(("rowent", *p.shape), p.device, p.dtype, p.numel())
        return _row_entropy(p, eps, buf=buf)

    @staticmethod
    def backward(ctx, grad_out):
        (p,) = ctx.saved_tensors
        # the gradient must be a fresh tensor, per the exact D7 formula
        # (log(p)+1 on p > _EPS, else 0, times -grad_out); only the selector is
        # scratch (it is consumed inside this call and never returned)
        mask = _get_scratch(("rowent_bwd_m", *p.shape), p.device, torch.bool,
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


_query_blocks_cache: dict = {}
_QUERY_BLOCKS_CACHE_MAX = 512


class _ScratchCache:
    __slots__ = ("cache", "bytes", "max_entries", "max_bytes")

    def __init__(self, max_entries: int = 128, max_bytes: int = 64 << 20):
        self.cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self.bytes = 0
        self.max_entries = max_entries
        self.max_bytes = max_bytes

    def get(self, key: tuple, device: torch.device, dtype: torch.dtype, numel: int) -> torch.Tensor:
        # The shape dims carried in ``key`` are informational only: buffers are
        # shared by purpose (``key[0]``), device and dtype, so one entry serves
        # every (B, T) shape instead of pinning one buffer per shape.  All call
        # sites use distinct purpose names for concurrently-live temporaries,
        # and every scratch use stays outside autograd tracking (custom
        # Function forward/backward interiors), so sharing is aliasing-safe.
        name = key[0] if isinstance(key, tuple) else key
        ckey = (name, device, dtype)
        entry = self.cache.get(ckey)
        if entry is not None and entry.numel() >= numel:
            # Move to end (MRU)
            self.cache.move_to_end(ckey)
            return entry[:numel]

        nbytes = numel * dtype.itemsize
        # Remove old entry if exists
        if ckey in self.cache:
            old = self.cache.pop(ckey)
            self.bytes -= old.numel() * old.element_size()

        # Evict by count
        while len(self.cache) >= self.max_entries:
            _, old = self.cache.popitem(last=False)
            self.bytes -= old.numel() * old.element_size()

        # Evict by bytes
        if nbytes <= self.max_bytes:
            while self.bytes + nbytes > self.max_bytes and self.cache:
                _, old = self.cache.popitem(last=False)
                self.bytes -= old.numel() * old.element_size()

        with torch.inference_mode(False):
            buf = torch.empty(numel, device=device, dtype=dtype)

        self.cache[ckey] = buf
        self.bytes += buf.numel() * buf.element_size()
        return buf[:numel]


_scratch_cache = _ScratchCache()


def _get_scratch(key: tuple, device: torch.device, dtype: torch.dtype, numel: int) -> torch.Tensor:
    return _scratch_cache.get(key, device, dtype, numel)


def _query_blocks(bh: int, T: int):
    """``[(start, stop), ...]`` query blocks for a ``bh``-fold batch of ``T``."""
    key = (bh, T)
    blocks = _query_blocks_cache.get(key)
    if blocks is not None:
        return blocks
    if T <= 32:
        blocks = ((0, T),)
    else:
        max_rows = (1 << 20) // (bh * T)
        if max_rows < 32:
            max_rows = 32
        rows = min(T, max_rows)
        blocks = tuple((i, min(i + rows, T)) for i in range(0, T, rows))
    if len(_query_blocks_cache) >= _QUERY_BLOCKS_CACHE_MAX:
        _query_blocks_cache.pop(next(iter(_query_blocks_cache)))
    _query_blocks_cache[key] = blocks
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
        yb = _get_scratch(("calib_y",), q.device, q.dtype, bh * max_rows * dv)
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
            for i0, i1 in blocks:
                r = i1 - i0
                p = _softmax_inplace(torch.baddbmm(
                    mask[i0:i1, :i1], q[:, i0:i1], kt[:, :, :i1],
                    beta=1.0, alpha=scale,
                    out=s[:bh * r * i1].view(bh, r, i1)))
                torch.bmm(p, v[:, :i1], out=yb[:bh * r * dv].view(bh, r, dv))
                y[:, i0:i1].copy_(yb[:bh * r * dv].view(bh, r, dv))
                if need_ent:
                    _row_entropy(p, eps, out=ent[:, i0:i1], buf=ebuf)
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
        rows0 = blocks[0][1] - blocks[0][0]
        # recompute probabilities (identical op sequence to forward)
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
        d_k = torch.zeros_like(k)
        d_v = torch.zeros_like(v)
        max_rows = rows0
        max_i1 = T
        dk = q.size(-1)
        s = _get_scratch(("calib_s",), q.device, q.dtype, bh * max_rows * max_i1)
        gbuf = _get_scratch(("calib_f",), q.device, q.dtype, bh * max_rows * max_i1) if grad_ent is not None else None
        mbuf = _get_scratch(("calib_bwd_mbuf",), q.device, torch.bool, bh * max_rows * max_i1) if grad_ent is not None else None
        dpb = _get_scratch(("calib_dp",), q.device, q.dtype, bh * max_rows * max_i1)
        tmpb = _get_scratch(("calib_bwd_tmp",), q.device, q.dtype, bh * T * dk)
        zero = _zero_scalar(q.device, q.dtype)
        for i0, i1 in blocks:
            n = bh * (i1 - i0) * i1
            p = _softmax_inplace(torch.baddbmm(
                mask[i0:i1, :i1], q[:, i0:i1], kt[:, :, :i1],
                beta=1.0, alpha=scale,
                out=s[:n].view(bh, i1 - i0, i1)))
            g = grad_y[:, i0:i1]
            ge = None if grad_ent is None else grad_ent[:, i0:i1]
            d_p = dpb[:n].view(bh, i1 - i0, i1)
            torch.bmm(g, vt[:, :, :i1], out=d_p)
            d_s = _softmax_grad_from_p(d_p, p, ge,
                                       buf=None if gbuf is None else gbuf[:n],
                                       mbuf=None if mbuf is None else mbuf[:n])
            tmp = tmpb[:bh * i1 * dk].view(bh, i1, dk)
            torch.bmm(p.transpose(1, 2), g, out=tmp)
            d_v[:, :i1].add_(tmp)
            torch.baddbmm(zero, d_s.transpose(1, 2), q[:, i0:i1], beta=0.0,
                          alpha=scale, out=tmp)
            d_k[:, :i1].add_(tmp)
            tmp = tmpb[:bh * (i1 - i0) * dk].view(bh, i1 - i0, dk)
            torch.baddbmm(zero, d_s, k[:, :i1], beta=0.0, alpha=scale,
                          out=tmp)
            d_q[:, i0:i1].copy_(tmp)
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
        # ``(p > _EPS) * (log(clamp_min(p, _EPS)) + 1)``: the selector zeroes
        # the sub-eps rows, applied as one select.
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
        s = torch.baddbmm(mask[i0:i1, :i1], q_blk, kt[:, :, :i1],
                          beta=1.0, alpha=scale)
    else:
        s = sbuf[:n * r * i1].view(n, r, i1)
        torch.baddbmm(mask[i0:i1, :i1], q_blk, kt[:, :, :i1],
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
        yb = _get_scratch(("null_y",), q.device, q.dtype, bh * max_rows * dv)
        for i0, i1 in blocks:
            r = i1 - i0
            p, p0 = _null_probs(q[:, i0:i1], kt, i0, i1, mask, null_k, B, h,
                                scale, sbuf)
            yblk = yb[:bh * r * dv].view(bh, r, dv)
            torch.bmm(p, v[:, :i1], out=yblk)
            y[:, i0:i1].copy_(yblk)
            y4[:, :, i0:i1].addcmul_(p0.view(B, h, r, 1), nv4)
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
        multi = len(blocks) > 1
        d_k = torch.zeros_like(k) if multi else None
        d_v = torch.zeros_like(v) if multi else None
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
        tmpb = _get_scratch(("null_bwd_tmp",), q.device, q.dtype, bh * T * dk)
        acc_k = _get_scratch(("null_bwd_ak",), q.device, q.dtype, bh * dk).view(bh, 1, dk).zero_()
        acc_v = _get_scratch(("null_bwd_av",), q.device, q.dtype, bh * dk).view(bh, 1, dk).zero_()
        for i0, i1 in blocks:
            q_blk = q[:, i0:i1]
            g = grad_y[:, i0:i1]
            r = i1 - i0
            p, p0 = _null_probs(q_blk, kt, i0, i1, mask, null_k, B, h, scale, sbuf)
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
                torch.bmm(p.transpose(1, 2), g, out=tmp)
                d_v[:, :i1].add_(tmp)
                torch.bmm(d_s.transpose(1, 2), q_blk, out=tmp)
                d_k[:, :i1].add_(tmp)
                tmp = tmpb[:bh * (i1 - i0) * dk].view(bh, i1 - i0, dk)
                torch.bmm(d_s, k[:, :i1], out=tmp)
                d_q[:, i0:i1].copy_(tmp)
            else:
                d_v = q.new_empty(bh, i1, dk)
                d_k = q.new_empty(bh, i1, dk)
                torch.bmm(p.transpose(1, 2), g, out=d_v)
                torch.bmm(d_s.transpose(1, 2), q_blk, out=d_k)
                torch.bmm(d_s, k[:, :i1], out=d_q[:, i0:i1])
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
            # The gate is the only gate-variant-specific module in the model.
            # It is initialised from a forked copy of the global RNG, so the
            # shared stream ``CAGE.apply(_init)`` consumes afterwards is left
            # exactly where it was and all four variants stay init-matched
            # (E1 is a controlled comparison).
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
            # calib_tau initialized so softplus(tau) = calib_tau_init initially
            # (tau_init = log(expm1(calib_tau_init))):
            # softplus(log(expm1(calib_tau_init))) = log(1 + exp(log(expm1(calib_tau_init)))) = log(1 + expm1(calib_tau_init)) = log(exp(calib_tau_init)) = calib_tau_init
            # gate temperature = softplus(tau) + 0.1 = calib_tau_init + 0.1 (default 0.9 + 0.1 = 1.0)
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

        qkv = self.qkv(x).view(B, T, 3, h, dk).permute(2, 0, 3, 1, 4).contiguous()
        qb = qkv[0].view(bh, T, dk)
        kb = qkv[1].view(bh, T, dk)
        vb = qkv[2].view(bh, T, dk)
        del qkv

        ent = None
        # head-major attention output (B, h, T, dk): the transpose into the
        # model layout happens once, after the gate, and is fused into the
        # per-channel gate multiply below.
        y4 = None
        stats = None
        if not dropping:
            # [fixed design, docs/design_notes.md D1] All four gate variants
            # share ONE hand-written kernel family: E1 is a controlled
            # comparison, so the numerics must not depend on the gate variant.
            # The calibrated kernel emits the row entropy as a by-product.
            # Do NOT switch any variant to a fused/flash attention call.
            if is_null:
                y4 = _NullAttn.apply(qb, kb, vb, self.null_k, self.null_v,
                                     self.inv_scale, B).view(B, h, T, dk)
            else:
                # none / query / calib share the same kernel; the row entropy
                # by-product is materialised only for the calibrated gate
                y, ent = _CalibAttn.apply(qb, kb, vb, self.inv_scale, _EPS,
                                          is_calib)
                y4 = y.view(B, h, T, dk)
                ent = None if ent is None else ent.view(B, h, T)
            if need_stats:
                stats = self._row_stats_at(
                    qb.view(B, h, T, dk), kb, B, h, T, stat_pos, is_null,
                    ent_full=ent)
        else:
            # attention dropout acts on the probabilities; the statistics are
            # read from the clean softmax rows before the dropout.
            q_ = qb.view(B, h, T, dk)
            k_ = kb.view(B, h, T, dk)
            if is_null:
                nk = self.null_k.view(1, h, 1, dk).expand(B, h, 1, dk)
                kv_k = torch.cat([nk, k_], dim=2)
                att = torch.matmul(q_, kv_k.transpose(-2, -1)).mul_(self.inv_scale)
                # fill the real-key columns with -inf and leave the null
                # column untouched; identical to adding the additive 0/-inf
                # mask (the (T+1)-wide bool form comes from the module cache).
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
                y4 = torch.bmm(pb[..., 1:], vb).view(B, h, T, dk)
                y4.addcmul_(p.view(B, h, T, T + 1)[..., :1],
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

        # ---- gating ----
        gate_val = None
        if kind == "query":
            gate_val = self.gate(x)
            gate_val.sigmoid_()
        elif is_calib:
            log_len = math.log(max(T, 2))
            # Fused: gate(x) - calib_a*log_len - (calib_c*calib_scale/log_len) * ent.mean(1, keepdim=True)
            gate_val = self.gate(x)
            gate_val.sub_(self.calib_a * log_len)
            # Subtract (calib_c*calib_scale/log_len) * ent.mean(1, keepdim=True)
            # mean(dim=1) -> (B, T), unsqueeze(-1) -> (B, T, 1) for broadcasting
            conc = ent.mean(dim=1).div_(log_len).unsqueeze(-1)
            gate_val.addcmul_(conc, self.calib_c * self.calib_scale, value=-1.0)
            tau = F.softplus(self.calib_tau).add_(0.1)
            gate_val.div_(tau).sigmoid_()

        # The head-major output is transposed into the model layout once.  For
        # the gated variants the transpose and the per-channel gate multiply
        # share a single pass: ``out[b, t, h, dk] = y[b, h, t, dk] * g[b, t, ...]``
        # is the same elementwise product as transposing first and multiplying.
        if gate_val is None:
            y = y4.transpose(1, 2).reshape(B, T, C)
        else:
            # one pass: the per-channel gate multiply doubles as the
            # head-major -> token-major transpose.  The gate is the leading
            # operand so the product keeps the token-major contiguous
            # layout; the elementwise product is commutative, so this is the
            # same value as transposing first and multiplying afterwards.
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
        # the GELU is folded into the second projection (see _GeluLinear)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            _GeluLinear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout, inplace=True),
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
            # The gate was already given its weight and its "start open" bias
            # in Attention.__init__ under a forked RNG; re-drawing it here
            # would consume the shared stream this walk draws from, so the
            # four gate variants stay init-matched (see Attention.__init__).
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
                # `mul` already yields a private tensor, so the shared arange
                # does not have to be cloned before it is scaled in place
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