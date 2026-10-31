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
                 Only the *product* ``c * calib_scale`` enters the gate, so
                 ``calib_scale`` is a fixed (non-learnable) hyperparameter:
                 a second free scalar next to ``calib_c`` would be
                 unidentifiable (any split of one effective coefficient
                 fits equally well).
                 Note on the divisor: every row is normalised by log(T) of
                 the full sequence, although causal row ``t`` can only see
                 ``t+1`` keys (entropy ceiling log(t+1)).  The task and all
                 collected statistics read the *final* query row, where the
                 visible-key count equals T and the global divisor is exact;
                 the forward-time behaviour on non-final rows can be
                 ablated with ``Cfg.calib_pos_norm``.
                 ``Cfg.calib_use_len`` / ``Cfg.calib_use_ent`` switch one
                 term off at a time for the E1/E2 ablation arms; their
                 defaults (True/True) reproduce the full gate exactly.
                 ``Cfg.calib_shuffle_ent`` (default False) is the E1
                 *active-control* switch: the gate keeps the identical
                 architecture, parameter count and computation, but the
                 entropy it reads is replaced by a fixed position
                 permutation of the same tensor - the same values and
                 marginal distribution, decorrelated from each queried
                 row's own attention - so the entropy *information* can be
                 separated from the mere presence of an extra
                 data-dependent input.  Defaults reproduce the full gate
                 exactly; the flag is ablation-only.
                 ``Cfg.calib_ent_norm`` (default True) is the E2
                 normalisation ablation: with False the concentration term
                 reads the *raw* row-mean entropy instead of
                 ``Hbar = H / log T``, while the architecture, parameter
                 count, computation and length term are unchanged.  The
                 default reproduces the full gate exactly.
                 ``Cfg.calib_pos_norm`` (default False) is the
                 position-conflation ablation: with True the divisor (and
                 the length term) use each row's visible-token length
                 ``log(max(t+1, 2))`` instead of the global ``log T``.  On
                 the final query row the two are identical, so all
                 registered readouts are unchanged; only non-final rows
                 differ.  The default reproduces the full gate exactly.
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
import operator
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

try:
    from torch.compiler import disable as _no_compile
except ImportError:  # torch without torch.compile
    def _no_compile(fn):
        return fn


_EPS = 1e-9
_MAX_TORCH_INDEX = int(torch.iinfo(torch.long).max)

_zero_scalar_cache: dict = {}

def _zero_scalar(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Zero scalar for the given device and dtype."""
    key = (device, dtype)
    z = _zero_scalar_cache.get(key)
    if z is None:
        with torch.inference_mode(False):
            z = torch.zeros((), device=device, dtype=dtype)
        _zero_scalar_cache[key] = z
    return z


def _servable(t: torch.Tensor) -> bool:
    """Whether a tensor may be handed back in the current mode.

    Inference tensors can only be written in place (and used as index
    tensors) inside inference mode, so they are served to an active
    inference mode only.
    """
    return not t.is_inference() or torch.is_inference_mode_enabled()


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
    # Ablation switches for the two calibration terms of ``gate="calib"``.
    # Defaults reproduce the full calibrated gate exactly; the flags exist so
    # E1/E2 can drop one term at a time (factorising the two claimed
    # contributions) without changing the implementation of the full gate.
    calib_use_len: bool = True
    calib_use_ent: bool = True
    # Active-control (placebo) switch for the concentration term: when True
    # the entropy that drives the gate is a fixed per-(layer, T) position
    # permutation of the real entropy.  The architecture, parameter count,
    # computation and marginal signal distribution are unchanged - only the
    # association between a queried row and its own attention concentration
    # is destroyed - so E1 can test whether the calibrated gate's effect is
    # carried by the attention-distribution information or by capacity.
    # Requires the concentration term to be on; default False reproduces
    # the full calibrated gate exactly.
    calib_shuffle_ent: bool = False
    # Ablation switch for the length normalisation of the concentration
    # term.  True (default) is the registered gate
    # sigmoid((W_g x - a*log T - c*scale*H/log T)/tau); False feeds the raw
    # row-mean entropy instead, so E2 can test whether dividing by log T is
    # what makes the concentration signal length-stable.  The length term,
    # parameter count and computation are identical either way.
    calib_ent_norm: bool = True
    # Ablation switch for *which* length normalises each query row.  The
    # registered gate divides every row's head-mean entropy by log(T) of
    # the full sequence; causal row ``t`` can only attend to ``t+1`` keys,
    # so its entropy ceiling is log(t+1) and the global divisor mixes the
    # row's position into its measured concentration on non-final rows.
    # The supervised/statistics row is always the final query row (visible
    # keys = T), where the global divisor is exact; this switch exists so
    # the forward-time behaviour on the other rows can be tested: with
    # True, both the concentration divisor and the length term use
    # log(max(t+1, 2)) per row (visible-token calibration).  Requires the
    # concentration term and its log-length normalisation to be on;
    # default False reproduces the registered gate exactly.
    calib_pos_norm: bool = False
    use_gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.gate not in GATES:
            raise ValueError(f"unknown gate {self.gate!r}; valid: {list(GATES)}")
        if self.norm not in NORMS:
            raise ValueError(f"unknown norm {self.norm!r}; valid: {list(NORMS)}")
        # Ablation switches are semantic booleans; a string such as
        # ``"False"`` is truthy and would silently enable/disable the wrong
        # term (or gradient checkpointing).  Validate them at the boundary.
        for name in ("calib_use_len", "calib_use_ent", "calib_shuffle_ent",
                     "calib_ent_norm", "calib_pos_norm",
                     "use_gradient_checkpointing"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(
                    f"{name} must be a bool, got {getattr(self, name)!r}")
        # Geometry fields index embeddings/loops, so a float (or bool)
        # cannot be coerced later: torch would raise a TypeError halfway
        # through construction, after some parameters had already been
        # allocated.  Validate and canonicalise them at the configuration
        # boundary instead.
        for name in ("vocab_size", "d_model", "n_heads", "n_layers",
                     "max_len"):
            raw = getattr(self, name)
            if isinstance(raw, bool):
                raise ValueError(f"{name} must be an integer, not bool")
            try:
                value = operator.index(raw)
            except TypeError as exc:
                raise ValueError(
                    f"{name} must be an integer, got {raw!r}") from exc
            if value > _MAX_TORCH_INDEX:
                raise ValueError(
                    f"{name}={value} exceeds torch's signed 64-bit index "
                    f"range")
            setattr(self, name, value)
        if self.d_model <= 0 or self.n_heads <= 0 \
                or self.d_model % self.n_heads:
            raise ValueError(
                f"d_model={self.d_model} is not a positive multiple of "
                f"n_heads={self.n_heads}"
            )
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers!r}")
        if self.max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {self.max_len!r}")
        if self.vocab_size < 1:
            raise ValueError(
                f"vocab_size must be >= 1, got {self.vocab_size!r}")
        if self.layer_scale is not None:
            if isinstance(self.layer_scale, bool):
                raise ValueError(
                    "layer_scale must be a finite number or None, not bool")
            try:
                layer_scale = float(self.layer_scale)
            except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                raise ValueError(
                    f"layer_scale must be a finite number or None, got "
                    f"{self.layer_scale!r}") from exc
            if not math.isfinite(layer_scale):
                raise ValueError(
                    f"layer_scale must be finite, got {self.layer_scale!r}")
            self.layer_scale = layer_scale
        if isinstance(self.dropout, bool):
            raise ValueError(
                f"dropout must be a finite number in [0, 1], got "
                f"{self.dropout!r}")
        try:
            dropout = float(self.dropout)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError(
                f"dropout must be a finite number in [0, 1], got "
                f"{self.dropout!r}") from exc
        if not (math.isfinite(dropout) and 0.0 <= dropout <= 1.0):
            raise ValueError(
                f"dropout must be finite and in [0, 1], got "
                f"{self.dropout!r}")
        self.dropout = dropout
        for name in ("calib_a", "calib_c", "calib_scale"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be finite, got {value!r}")
            try:
                value = float(value)
            except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                raise ValueError(
                    f"{name} must be finite, got {getattr(self, name)!r}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            setattr(self, name, value)
        raw_tau_init = self.calib_tau_init
        if isinstance(raw_tau_init, bool):
            raise ValueError(
                f"calib_tau_init must be finite and > 0, got "
                f"{raw_tau_init!r}")
        try:
            tau_init = float(raw_tau_init)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError(
                f"calib_tau_init must be finite and > 0, got "
                f"{raw_tau_init!r}") from exc
        if not math.isfinite(tau_init) or tau_init <= 0.0:
            raise ValueError(
                f"calib_tau_init must be finite and > 0 so that "
                f"log(expm1(calib_tau_init)) is defined, got "
                f"{raw_tau_init!r}")
        try:
            math.log(math.expm1(tau_init))
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                f"calib_tau_init is too large for "
                f"log(expm1(calib_tau_init)) to stay finite, got "
                f"{raw_tau_init!r}") from exc
        self.calib_tau_init = tau_init
        if self.calib_shuffle_ent and (self.gate != "calib"
                                       or not self.calib_use_ent):
            raise ValueError(
                "calib_shuffle_ent is an active-control switch for the "
                "concentration term of gate='calib'; it requires "
                "gate='calib' and calib_use_ent=True"
            )
        if not self.calib_ent_norm and (self.gate != "calib"
                                        or not self.calib_use_ent):
            raise ValueError(
                "calib_ent_norm=False is an ablation of the concentration "
                "term's log-T normalisation; it requires gate='calib' and "
                "calib_use_ent=True"
            )
        if self.calib_pos_norm and (self.gate != "calib"
                                    or not self.calib_use_ent
                                    or not self.calib_ent_norm):
            raise ValueError(
                "calib_pos_norm is an ablation of the concentration term's "
                "length normalisation (per-row visible-token divisors); it "
                "requires gate='calib', calib_use_ent=True and "
                "calib_ent_norm=True"
            )


class Norm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=True)

    def forward(self, x):
        if torch.is_grad_enabled():
            return self.ln(x)
        return _layer_norm_into(self.ln, x)


class SeeDNorm(nn.Module):
    """Dynamic normalization: per-token scale predicted from hidden state."""
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale = nn.Linear(dim, dim)
        nn.init.zeros_(self.scale.bias)
        nn.init.normal_(self.scale.weight, std=0.02)
        # Explicitly initialised above; skip the module-wide re-init.
        self.scale._preinitialized = True

    def forward(self, x):
        if torch.is_grad_enabled():
            s = self.scale(x).sigmoid_()
            return self.ln(x).mul_(s).mul_(2.0)
        shape = x.shape
        n = x.numel() // shape[-1]
        s = _get_scratch(("seed_scale",), x.device, x.dtype, x.numel())
        _linear_into(x.reshape(n, shape[-1]), self.scale.weight,
                     self.scale.bias, s.view(n, shape[-1]))
        s.sigmoid_()
        return _layer_norm_into(self.ln, x).mul_(s.view(shape)).mul_(2.0)


def make_norm(kind: str, dim: int) -> nn.Module:
    return SeeDNorm(dim) if kind == "seednorm" else Norm(dim)


def _norm_out(norm: nn.Module, x: torch.Tensor, key: str) -> torch.Tensor:
    """Eval-path normalization written into the named scratch buffer.

    Mirrors ``Norm.forward`` / ``SeeDNorm.forward`` without gradients, but
    with a caller-chosen output key so the final normalization can never
    alias the block-level ``"ln_out"`` buffer its input may be a view of.
    """
    if isinstance(norm, SeeDNorm):
        shape = x.shape
        n = x.numel() // shape[-1]
        s = _get_scratch(("seed_scale",), x.device, x.dtype, x.numel())
        _linear_into(x.reshape(n, shape[-1]), norm.scale.weight,
                     norm.scale.bias, s.view(n, shape[-1]))
        s.sigmoid_()
        return _layer_norm_into(norm.ln, x, key).mul_(s.view(shape)).mul_(2.0)
    return _layer_norm_into(norm.ln, x, key)


_MAX_MASK_CACHE = 64

_QKV_CHUNK_ELEMS = 3 * (1 << 20)

_MLP_CHUNK_ROWS = 2048

_causal_mask_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()


def _cache_lookup(cache: OrderedDict, key):
    """Return the cached value and mark it as most recently used."""
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_store(cache: OrderedDict, key, value, limit: int):
    """Insert ``value`` and evict the least recently used entry past ``limit``."""
    cache[key] = value
    if len(cache) > limit:
        cache.popitem(last=False)
    return value


def _causal_mask(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (T, device, dtype)
    m = _cache_lookup(_causal_mask_cache, key)
    if m is not None:
        return m
    with torch.inference_mode(False):
        m = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
        m.triu_(1)
    return _cache_store(_causal_mask_cache, key, m, _MAX_MASK_CACHE)


_MAX_NULL_CACHE = 64
_causal_null_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()


def _causal_null_mask(T: int, device: torch.device,
                      dtype: torch.dtype) -> torch.Tensor:
    """Additive causal mask with an *unmasked* null column.

    Shape ``(T, T + 1)``: column 0 (the learnable null slot) is 0 and the
    real-key columns hold the usual strictly-upper ``-inf``.  It is added
    to the ``(B*H, T, T+1)`` scores.
    """
    key = (T, device, dtype)
    m = _cache_lookup(_causal_null_cache, key)
    if m is not None:
        return m
    with torch.inference_mode(False):
        m = torch.full((T, T + 1), float("-inf"), device=device, dtype=dtype)
        m[:, 0] = 0.0
        m[:, 1:].triu_(1)
    return _cache_store(_causal_null_cache, key, m, _MAX_NULL_CACHE)


_ARANGE_CACHE_MAX = 128
_arange_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()


def _arange(n: int, device: torch.device) -> torch.Tensor:
    """Read-only ``arange(n)`` (index tensors only)."""
    key = (n, device)
    a = _cache_lookup(_arange_cache, key)
    if a is not None and _servable(a):
        return a
    with torch.inference_mode(False):
        a = torch.arange(n, device=device)
    return _cache_store(_arange_cache, key, a, _ARANGE_CACHE_MAX)


def _mask_rows(pos: torch.Tensor, T: int, h: int, dtype: torch.dtype) -> torch.Tensor:
    """``(B*h, 1, T)`` additive mask rows for the query positions ``pos``.

    Row ``b`` is 0 up to column ``pos[b]`` and ``-inf`` beyond it - exactly
    row ``pos[b]`` of the cached causal mask, expanded over the heads.
    """
    B = pos.size(0)
    rows = _causal_mask(T, pos.device, dtype).index_select(0, pos)
    if h == 1:
        return rows.view(B, 1, T)
    return rows.unsqueeze(1).expand(B, h, T).reshape(B * h, 1, T)


def _softmax_inplace(s: torch.Tensor) -> torch.Tensor:
    """Row softmax written back into ``s``."""
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
        rows = gy.numel() // gy.size(-1)
        rowsum = _get_scratch(("softmax_rowsum",), gy.device, gy.dtype,
                              rows)[:rows].view(*gy.shape[:-1], 1)
        torch.sum(gy, dim=-1, keepdim=True, out=rowsum)
        return gy.addcmul_(y, rowsum, value=-1.0)


class _RowEntropy(torch.autograd.Function):
    """``(p * p.clamp_min(eps).log()).sum(-1).neg()``."""

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
        d = _get_scratch(("rowent_bwd_d",), p.device, p.dtype, p.numel())
        d = d[:p.numel()].view_as(p)
        torch.clamp_min(p, ctx.eps, out=d)
        d.log_()
        d.add_(1.0)
        torch.where(p > ctx.eps, d, _zero_scalar(p.device, p.dtype), out=d)
        d.mul_(grad_out.unsqueeze(-1))
        d.neg_()
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
    if entry is not None and entry.numel() >= numel and _servable(entry):
        if entry.numel() > 4 * numel:
            with torch.inference_mode(False):
                buf = torch.empty(numel, device=device, dtype=dtype)
            _scratch_cache[ckey] = buf
            return buf
        # Exact-size requests hand out the cached tensor itself; undersized
        # requests receive a slice view.
        return entry if entry.numel() == numel else entry[:numel]
    with torch.inference_mode(False):
        buf = torch.empty(numel, device=device, dtype=dtype)
    _scratch_cache[ckey] = buf
    return buf


# Buffers a dropout-free, gradient-free forward may hold when it returns.
_EVAL_SCRATCH = (
    "embed_out", "packed", "qkv_out", "ln_out", "ln_f_out", "ln_mean",
    "ln_rstd", "seed_scale", "mlp_up", "mlp_down", "mlp_down_big",
    "proj_out", "attn_y", "gate_out", "calib_s", "calib_y", "calib_ent", "calib_ent_shuf", "calib_f", "calib_conc", "null_s",
    "null_s0", "null_y", "null_yblk", "null_m", "rowent", "rowstats_att",
    "rowstats_att_null", "rowstats_sum", "mask_rows", "mask_rows_b",
    # Backward-only buffers: kept across training steps, but dead once an
    # evaluation pass has run; release them too so retained scratch does
    # not hold training-sized work buffers after scoring.
    "calib_bwd_work", "calib_dp", "calib_rowsum", "rowent_bwd_d",
    "softmax_rowsum", "null_bwd_e", "null_bwd_gu", "null_bwd_dot",
    "null_bwd_ak", "null_bwd_av",
)


def release_eval_scratch() -> None:
    """Release the reusable buffers held by the last evaluation pass."""
    _release_scratch(*_EVAL_SCRATCH)


def _release_scratch(*names: str) -> None:
    """Drop the cached buffers with the given names.

    Called at points where every live reference to the named buffers has
    been dropped, so the next ``_get_scratch`` for the name builds a fresh
    buffer instead of serving the old entry.
    """
    if not names:
        return
    drop = set(names)
    for key in [k for k in _scratch_cache if k[0] in drop]:
        del _scratch_cache[key]


def _balanced_step(n: int, cap: int) -> int:
    """Balanced chunk size covering ``n`` units in ``ceil(n/cap)`` chunks.

    Every chunk stays within ``cap`` and all chunks have the same size
    (within one unit), including the tail chunk.
    """
    if n <= cap:
        return n
    n_chunks = -(-n // cap)
    return max(1, min(cap, -(-n // n_chunks)))


def _qkv_step(B: int, T: int, C: int) -> int:
    """Batch chunk size for the staged QKV projection."""
    rows = max(1, _QKV_CHUNK_ELEMS // max(1, T * 3 * C))
    # A zero-row batch still has to yield a legal range() step; the loops
    # below then correctly execute zero chunks.  Returning 0 would abort an
    # otherwise valid empty forward pass.
    return max(1, _balanced_step(B, rows))


def _linear_into(x: torch.Tensor, weight: torch.Tensor,
                 bias: torch.Tensor | None, out: torch.Tensor) -> torch.Tensor:
    """``F.linear(x, weight, bias)`` written into ``out``."""
    if bias is None:
        torch.mm(x, weight.t(), out=out)
    else:
        torch.addmm(bias, x, weight.t(), out=out)
    return out


def _layer_norm_into(ln: nn.LayerNorm, x: torch.Tensor,
                     key: str = "ln_out") -> torch.Tensor:
    """``ln(x)`` for the no-grad forward path.

    The result is written into the buffers named ``key`` /
    ``"ln_mean"`` / ``"ln_rstd"``.

    ``key`` names a *disjoint* output buffer: callers whose input may be a
    view of the block-level ``"ln_out"`` scratch (the final normalization)
    pass their own key so the layer-norm input and output never share
    storage at different coordinates.
    """
    shape = x.shape
    rows = x.numel() // shape[-1]
    out = _get_scratch((key,), x.device, x.dtype, x.numel())
    mean = _get_scratch(("ln_mean",), x.device, x.dtype, rows)
    rstd = _get_scratch(("ln_rstd",), x.device, x.dtype, rows)
    torch.ops.aten.native_layer_norm.out(
        x, [shape[-1]], ln.weight, ln.bias, ln.eps,
        out0=out.view(shape), out1=mean.view(shape[:-1] + (1,)),
        out2=rstd.view(shape[:-1] + (1,)))
    return out.view(shape)


def _query_blocks(bh: int, T: int):
    """``[(start, stop), ...]`` query blocks for a ``bh``-fold batch of ``T``."""
    key = (bh, T)
    blocks = _cache_lookup(_query_blocks_cache, key)
    if blocks is not None:
        return blocks
    if T <= 32 or bh <= 0:
        # A zero-row batch has no query rows to block; returning the single
        # whole-sequence block keeps the empty forward legal (the kernel's
        # zero-sized GEMMs still run) instead of dividing by ``bh`` = 0.
        blocks = ((0, T),)
    else:
        max_rows = (1 << 20) // (bh * T)
        if max_rows < 32:
            max_rows = 32
        # Balanced blocks under D3's cap; every block stays within
        # ``max_rows``.
        rows = _balanced_step(T, max_rows)
        blocks = tuple((i, min(i + rows, T)) for i in range(0, T, rows))
    return _cache_store(_query_blocks_cache, key, blocks,
                        _QUERY_BLOCKS_CACHE_MAX)


def _row_entropy(p: torch.Tensor, eps: float,
                 out: torch.Tensor | None = None,
                 buf: torch.Tensor | None = None) -> torch.Tensor:
    """``-(p * log p).sum(-1)`` computed row-wise.

    ``buf`` optionally supplies a flat tensor of at least ``p.numel()``
    elements as the ``clamp_min`` output.
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

    The entropy output is only produced when ``need_ent`` is set (calibrated
    gate or diagnostics); the attention output itself is identical either
    way, so the kernel stays shared by all gate variants.

    ``ybt`` optionally supplies a ``(B, T, H, dk)`` buffer used by the
    no-grad eval path: each block is written straight into it (transposing
    the block once on the way) and returned in that layout.
    """

    @staticmethod
    @_no_compile
    def forward(ctx, qkv, scale: float, eps: float, need_ent: bool = True,
                ybt: torch.Tensor | None = None):
        ctx.set_materialize_grads(False)
        grad_needed = any(ctx.needs_input_grad)
        if ybt is not None and grad_needed:
            raise RuntimeError("_CalibAttn ybt layout is no-grad only")
        q, k, v = qkv[0], qkv[1], qkv[2]
        bh, T = q.size(0), q.size(1)
        blocks = _query_blocks(bh, T)
        mask = _causal_mask(T, q.device, q.dtype)
        kt = k.transpose(1, 2)
        dv = v.size(-1)
        max_rows = max(i1 - i0 for i0, i1 in blocks)
        max_i1 = max(i1 for _, i1 in blocks)
        s = _get_scratch(("calib_s",), q.device, q.dtype, bh * max_rows * max_i1)
        if ybt is not None:
            B, _, h, dv = ybt.shape
            ent = (_get_scratch(("calib_ent",), q.device, q.dtype,
                                bh * T).view(bh, T)) if need_ent else None
            # One work buffer serves both the block output (copied out
            # before the entropy pass) and the entropy log-term.
            work_numel = bh * max_rows * (max(max_i1, dv) if need_ent else dv)
            yblk = _get_scratch(("calib_f",), q.device, q.dtype, work_numel)
            ebuf = yblk if need_ent else None
            for i0, i1 in blocks:
                r = i1 - i0
                p = _softmax_inplace(torch.baddbmm(
                    mask[i0:i1, :i1], q[:, i0:i1], kt[:, :, :i1],
                    beta=1.0, alpha=scale,
                    out=s[:bh * r * i1].view(bh, r, i1)))
                yb = yblk[:bh * r * dv].view(bh, r, dv)
                torch.bmm(p, v[:, :i1], out=yb)
                ybt[:, i0:i1].copy_(yb.view(B, h, r, dv).transpose(1, 2))
                if need_ent:
                    _row_entropy(p, eps, out=ent[:, i0:i1], buf=ebuf)
            ctx.scale = scale
            ctx.eps = eps
            return ybt, ent
        if grad_needed:
            y = q.new_empty(bh, T, dv)
        else:
            y = _get_scratch(("calib_y",), q.device, q.dtype,
                             bh * T * dv).view(bh, T, dv)
        ent = None
        if need_ent:
            ent = (q.new_empty(bh, T) if grad_needed else
                   _get_scratch(("calib_ent",), q.device, q.dtype,
                                bh * T).view(bh, T))
        if len(blocks) == 1:
            p = _softmax_inplace(torch.baddbmm(
                mask, q, kt, beta=1.0, alpha=scale,
                out=s[:bh * T * T].view(bh, T, T)))
            torch.bmm(p, v, out=y)
            if need_ent:
                _row_entropy(p, eps, out=ent,
                             buf=_get_scratch(("calib_f",), q.device, q.dtype, bh * T * T))
        else:
            # One work buffer serves both the entropy log-term (written
            # before the block output) and the compact block output.
            work_numel = bh * max_rows * (max(max_i1, dv) if need_ent else dv)
            work = _get_scratch(("calib_f",), q.device, q.dtype, work_numel)
            # Each block's result is copied into its row slice of ``y``.
            for i0, i1 in blocks:
                r = i1 - i0
                p = _softmax_inplace(torch.baddbmm(
                    mask[i0:i1, :i1], q[:, i0:i1], kt[:, :, :i1],
                    beta=1.0, alpha=scale,
                    out=s[:bh * r * i1].view(bh, r, i1)))
                if need_ent:
                    _row_entropy(p, eps, out=ent[:, i0:i1], buf=work)
                yb = work[:bh * r * dv].view(bh, r, dv)
                torch.bmm(p, v[:, :i1], out=yb)
                y[:, i0:i1].copy_(yb)
        if grad_needed:
            ctx.save_for_backward(qkv)
        ctx.scale = scale
        ctx.eps = eps
        return y, ent

    @staticmethod
    def backward(ctx, grad_y, grad_ent):
        saved = ctx.saved_tensors
        if not saved:
            return None, None, None, None, None
        (qkv,) = saved
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale, eps = ctx.scale, ctx.eps
        bh, T = q.size(0), q.size(1)
        if grad_y is None:
            grad_y = q.new_zeros(bh, T, v.size(-1))
        blocks = _query_blocks(bh, T)
        mask = _causal_mask(T, q.device, q.dtype)
        kt = k.transpose(1, 2)
        vt = v.transpose(1, 2)
        if len(blocks) == 1:
            s = _get_scratch(("calib_s",), q.device, q.dtype, bh * T * T)
            p = _softmax_inplace(torch.baddbmm(
                mask, q, kt, beta=1.0, alpha=scale,
                out=s[:bh * T * T].view(bh, T, T)))
            n = p.numel()
            gbuf = None
            if grad_ent is not None:
                gbuf = _get_scratch(("calib_f",), q.device, q.dtype, n)
            d_p = _get_scratch(("calib_dp",), q.device, q.dtype, n).view(bh, T, T)
            torch.bmm(grad_y, vt, out=d_p)
            d_s = _softmax_grad_from_p(d_p, p, grad_ent, eps, buf=gbuf)
            d_qkv = torch.empty_like(qkv)
            torch.bmm(p.transpose(1, 2), grad_y, out=d_qkv[2])
            zero = _zero_scalar(d_s.device, d_s.dtype)
            torch.baddbmm(zero, d_s, k, beta=0.0, alpha=scale, out=d_qkv[0])
            torch.baddbmm(zero, d_s.transpose(1, 2), q, beta=0.0, alpha=scale,
                          out=d_qkv[1])
            return d_qkv, None, None, None, None
        d_qkv = torch.empty_like(qkv)
        d_q = d_qkv[0].view_as(q)
        d_k = d_qkv[1].view_as(k)
        d_v = d_qkv[2].view_as(v)
        max_rows = max(i1 - i0 for i0, i1 in blocks)
        max_i1 = max(i1 for _, i1 in blocks)
        s = _get_scratch(("calib_s",), q.device, q.dtype, bh * max_rows * max_i1)
        dpb = _get_scratch(("calib_dp",), q.device, q.dtype, bh * max_rows * max_i1)
        dk = q.size(-1)
        stg_i1 = max(i1 for _, i1 in blocks[:-1])
        # Work buffer for the entropy log-term and block staging.  The
        # first processed block also stages its (possibly largest) dq slice,
        # so the row extent must cover ``max_rows`` as well as the prefix.
        work_numel = bh * max(stg_i1, max_rows) * dk
        if grad_ent is not None:
            work_numel = max(work_numel, bh * max_rows * max_i1)
        work = _get_scratch(("calib_bwd_work",), q.device, q.dtype,
                            work_numel)
        gbuf = work if grad_ent is not None else None
        stg = work
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
            d_s = _softmax_grad_from_p(d_p, p, ge, eps,
                                       buf=None if gbuf is None else gbuf[:n])
            # The first block writes ``d_v`` / ``d_k`` outright; later
            # blocks accumulate into their prefix slice.
            if first:
                torch.bmm(p.transpose(1, 2), g, out=d_v)
                torch.baddbmm(zero, d_s.transpose(1, 2), q[:, i0:i1],
                              beta=0.0, alpha=scale, out=d_k)
                first = False
            else:
                sv = stg[:bh * i1 * dk].view(bh, i1, dk)
                torch.bmm(p.transpose(1, 2), g, out=sv)
                d_v[:, :i1].add_(sv)
                torch.baddbmm(zero, d_s.transpose(1, 2), q[:, i0:i1],
                              beta=0.0, alpha=scale, out=sv)
                d_k[:, :i1].add_(sv)
            dqblk = stg[:bh * r * dk].view(bh, r, dk)
            torch.baddbmm(zero, d_s, k[:, :i1], beta=0.0, alpha=scale,
                          out=dqblk)
            d_q[:, i0:i1].copy_(dqblk)
        return d_qkv, None, None, None, None


def _softmax_grad_from_p(d_p: torch.Tensor, p: torch.Tensor,
                         grad_ent: torch.Tensor | None,
                         eps: float = _EPS,
                         buf: torch.Tensor | None = None) -> torch.Tensor:
    """Score gradient for ``p = softmax(s)`` plus the calibrated-entropy term.
    Consumes ``d_p``.

    ``buf`` optionally supplies a flat buffer for the ``log(p)`` term.

    The surviving term is ``log(p) + 1`` on ``p > eps`` and ``0`` elsewhere;
    the sub-eps entries are dropped by the trailing select, so the value taken
    there is irrelevant.  ``log`` is still fed ``p.clamp_min(eps)`` rather
    than raw ``p``: masked-out entries (``p == 0``) would otherwise take the
    log of zero, and the clamp removes them without changing any surviving
    value (bit-for-bit identical results).
    """
    if grad_ent is not None:
        if buf is None:
            logterm = torch.clamp_min(p, eps).log()
        else:
            logterm = buf[:p.numel()].view_as(p)
            torch.clamp_min(p, eps, out=logterm)
            logterm.log_()
        logterm.add_(1.0)
        torch.where(p > eps, logterm, _zero_scalar(p.device, p.dtype),
                    out=logterm)
        d_p.addcmul_(logterm, grad_ent.unsqueeze(-1), value=-1.0)
    d_p.mul_(p)
    n0, n1 = d_p.size(0), d_p.size(1)
    rowsum = _get_scratch(("calib_rowsum",), d_p.device, d_p.dtype, n0 * n1)[
        :n0 * n1].view(n0, n1, 1)
    torch.sum(d_p, dim=-1, keepdim=True, out=rowsum)
    return d_p.addcmul_(p, rowsum, value=-1.0)


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
    dk = q_blk.size(-1)
    torch.matmul(q_blk.view(B, h, r, dk), null_k.view(h, dk, 1),
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
    score matrix rectangular in the real keys only.
    """

    @staticmethod
    @_no_compile
    def forward(ctx, qkv, null_k, null_v, scale, B, need_grad: bool = True,
                ybt: torch.Tensor | None = None):
        q, k, v = qkv[0], qkv[1], qkv[2]
        bh, T = q.size(0), q.size(1)
        h = null_k.size(0)
        blocks = _query_blocks(bh, T)
        mask = _causal_mask(T, q.device, q.dtype)
        kt = k.transpose(1, 2)
        dv = v.size(-1)
        max_rows = max(i1 - i0 for i0, i1 in blocks)
        max_i1 = max(i1 for _, i1 in blocks)
        if ybt is not None:
            # No-grad eval: write each block (plus its null-slot term)
            # straight into the ``(B, T, h, dv)`` caller buffer.
            if need_grad:
                raise RuntimeError("_NullAttn ybt layout is no-grad only")
            sbuf = _get_scratch(("null_s",), q.device, q.dtype,
                                bh * max_rows * max_i1)
            yblk = _get_scratch(("null_yblk",), q.device, q.dtype,
                                bh * max_rows * dv)
            nv4 = null_v.view(1, h, 1, dv)
            for i0, i1 in blocks:
                r = i1 - i0
                p, p0 = _null_probs(q[:, i0:i1], kt[:, :, :i1], i0, i1,
                                    mask[i0:i1, :i1], null_k, B, h, scale, sbuf)
                yb = yblk[:bh * r * dv].view(bh, r, dv)
                torch.bmm(p, v[:, :i1], out=yb)
                # Fold the null-slot term into the block before the
                # transposed copy into the caller buffer.
                yb.view(B, h, r, dv).addcmul_(p0.view(B, h, r, 1), nv4)
                ybt[:, i0:i1].copy_(yb.view(B, h, r, dv).transpose(1, 2))
            ctx.scale = scale
            ctx.shape = (B, h)
            return ybt
        if need_grad:
            y = q.new_empty(bh, T, dv)
        else:
            y = _get_scratch(("null_y",), q.device, q.dtype,
                             bh * T * dv).view(bh, T, dv)
        y4 = y.view(B, h, T, dv)
        nv4 = null_v.view(1, h, 1, dv)
        sbuf = _get_scratch(("null_s",), q.device, q.dtype, bh * max_rows * max_i1)
        if len(blocks) == 1:
            p, p0 = _null_probs(q, kt, 0, T, mask, null_k, B, h, scale, sbuf)
            torch.bmm(p, v, out=y)
            y4.addcmul_(p0.view(B, h, T, 1), nv4)
        else:
            yblk = _get_scratch(("null_yblk",), q.device, q.dtype,
                                bh * max_rows * dv)
            for i0, i1 in blocks:
                r = i1 - i0
                p, p0 = _null_probs(q[:, i0:i1], kt[:, :, :i1], i0, i1, mask[i0:i1, :i1], null_k, B, h,
                                    scale, sbuf)
                yb = yblk[:bh * r * dv].view(bh, r, dv)
                torch.bmm(p, v[:, :i1], out=yb)
                # Fold the null-slot term into the block before the copy.
                yb.view(B, h, r, dv).addcmul_(p0.view(B, h, r, 1), nv4)
                y[:, i0:i1].copy_(yb)
        if need_grad:
            ctx.save_for_backward(qkv, null_k, null_v)
        ctx.scale = scale
        ctx.shape = (B, h)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        saved = ctx.saved_tensors
        if not saved:
            return None, None, None, None, None, None, None
        qkv, null_k, null_v = saved
        q, k, v = qkv[0], qkv[1], qkv[2]
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
        d_qkv = torch.empty_like(qkv)
        d_q = d_qkv[0]
        d_k = d_qkv[1] if multi else None
        d_v = d_qkv[2] if multi else None
        d_q4 = d_q.view(B, h, T, dk)
        nk4 = null_k.view(1, h, 1, dk)
        nv4t = null_v.view(1, h, dk, 1)
        max_rows = max(i1 - i0 for i0, i1 in blocks)
        max_i1 = max(i1 for _, i1 in blocks)
        sbuf = _get_scratch(("null_s",), q.device, q.dtype, bh * max_rows * max_i1)
        # Work buffer for the block score gradient.
        eb = _get_scratch(("null_bwd_e",), q.device, q.dtype, bh * max_rows * max_i1)
        gub = _get_scratch(("null_bwd_gu",), q.device, q.dtype, bh * max_rows)
        dotb = _get_scratch(("null_bwd_dot",), q.device, q.dtype, bh * max_rows)
        acc_k = _get_scratch(("null_bwd_ak",), q.device, q.dtype, bh * dk).view(bh, 1, dk)
        acc_v = _get_scratch(("null_bwd_av",), q.device, q.dtype, bh * dk).view(bh, 1, dk)
        # Staging buffer for the ``d_v`` / ``d_k`` contribution.
        if multi:
            stg_i1 = max(i1 for _, i1 in blocks[:-1])
            stg = _get_scratch(("null_bwd_stg",), q.device, q.dtype,
                               bh * max(stg_i1, max_rows) * dk)
        else:
            stg = None
        zero = _zero_scalar(q.device, q.dtype)
        first = True
        first_null = True
        for i0, i1 in reversed(blocks):
            q_blk = q[:, i0:i1]
            g = grad_y[:, i0:i1]
            r = i1 - i0
            p, p0 = _null_probs(q_blk, kt[:, :, :i1], i0, i1, mask[i0:i1, :i1],
                                null_k, B, h, scale, sbuf)
            n = bh * r * i1
            e = torch.bmm(g, vt[:, :, :i1], out=eb[:n].view(bh, r, i1))
            gu0 = gub[:bh * r].view(bh, r, 1)
            torch.matmul(g.view(B, h, r, dk), nv4t,
                         out=gu0.view(B, h, r, 1))
            dot = dotb[:bh * r].view(bh, r, 1)
            # Row dot product via a batched GEMM.
            torch.bmm(p.view(bh * r, 1, i1), e.view(bh * r, i1, 1),
                      out=dot.view(bh * r, 1, 1))
            dot.addcmul_(p0, gu0)
            d_u = e.sub_(dot)
            gu0.sub_(dot)
            # Score gradients are unscaled; ``scale`` is applied in the
            # GEMM epilogue (``alpha``).
            d_s = d_u.mul_(p)
            d_s0 = gu0.mul_(p0)
            if multi:
                # The first block writes ``d_v`` / ``d_k`` outright; later
                # blocks add their ``(i1, dk)`` contribution into the
                # leading ``i1`` columns.
                if first:
                    torch.bmm(p.transpose(1, 2), g, out=d_v)
                    torch.baddbmm(zero, d_s.transpose(1, 2), q_blk,
                                  beta=0.0, alpha=scale, out=d_k)
                    first = False
                else:
                    sv = stg[:bh * i1 * dk].view(bh, i1, dk)
                    torch.bmm(p.transpose(1, 2), g, out=sv)
                    d_v[:, :i1].add_(sv)
                    torch.baddbmm(zero, d_s.transpose(1, 2), q_blk,
                                  beta=0.0, alpha=scale, out=sv)
                    d_k[:, :i1].add_(sv)
                dqblk = stg[:bh * r * dk].view(bh, r, dk)
                torch.baddbmm(zero, d_s, k[:, :i1], beta=0.0, alpha=scale,
                              out=dqblk)
                # Fold the null-key term into the staging block before the
                # copy.
                dqblk.view(B, h, r, dk).addcmul_(d_s0.view(B, h, r, 1), nk4,
                                                 value=scale)
                d_q[:, i0:i1].copy_(dqblk)
            else:
                torch.bmm(p.transpose(1, 2), g, out=d_qkv[2])
                torch.baddbmm(zero, d_s.transpose(1, 2), q_blk, beta=0.0,
                              alpha=scale, out=d_qkv[1])
                torch.baddbmm(zero, d_s, k[:, :i1], beta=0.0, alpha=scale,
                              out=d_qkv[0])
                d_q4[:, :, i0:i1].addcmul_(d_s0.view(B, h, r, 1), nk4,
                                           value=scale)
            if first_null:
                torch.baddbmm(zero, d_s0.transpose(-2, -1), q_blk, beta=0.0,
                              alpha=scale, out=acc_k)
                torch.bmm(p0.transpose(-2, -1), g, out=acc_v)
                first_null = False
            else:
                torch.baddbmm(acc_k, d_s0.transpose(-2, -1), q_blk, beta=1.0,
                              alpha=scale, out=acc_k)
                torch.baddbmm(acc_v, p0.transpose(-2, -1), g, beta=1.0,
                              alpha=1.0, out=acc_v)
        d_null_k = acc_k.view(B, h, dk).sum(0)
        d_null_v = acc_v.view(B, h, dk).sum(0)
        return d_qkv, d_null_k, d_null_v, None, None, None, None


_ENT_PLACEBO_SEED = 0x9E3779B1


class Attention(nn.Module):
    def __init__(self, cfg: Cfg, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
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
            self.calib_a = (nn.Parameter(torch.tensor(float(cfg.calib_a)))
                            if cfg.calib_use_len else None)
            self.calib_c = (nn.Parameter(torch.tensor(float(cfg.calib_c)))
                            if cfg.calib_use_ent else None)
            tau_init = math.log(math.expm1(cfg.calib_tau_init))
            self.calib_tau = nn.Parameter(torch.tensor(tau_init))
            # Only the product ``calib_c * calib_scale`` enters the gate, so
            # a second *learnable* scalar would be unidentifiable next to
            # ``calib_c`` (any split of one effective coefficient fits
            # equally well).  The output scale is therefore a fixed
            # hyperparameter recorded as a buffer, and ``calib_c`` carries
            # the whole learnable concentration coefficient
            # (``c_effective = calib_c * calib_scale``).
            self.register_buffer(
                "calib_scale", torch.tensor(float(cfg.calib_scale)))
            self._log_len_cache: dict[int, float] = {}
            self._log_vis_cache: dict[tuple, torch.Tensor] = {}
            self._ent_perm_cache: dict | None = (
                {} if cfg.calib_shuffle_ent else None)
            # True = registered gate (Hbar = H / log T); False = raw-entropy
            # ablation arm (E2/H2g).  Only read when calc_ent is true.
            self.calib_ent_norm = bool(cfg.calib_ent_norm)
            # True = per-row visible-token divisors log(max(t+1, 2))
            # instead of the global log T (position-conflation ablation).
            # Only read when calc_ent is true.
            self.calib_pos_norm = bool(cfg.calib_pos_norm)
        else:
            self.calib_a = self.calib_c = self.calib_tau = self.calib_scale = None
            self._log_len_cache = None
            self._log_vis_cache = None
            self._ent_perm_cache = None
            self.calib_ent_norm = True
            self.calib_pos_norm = False
        if cfg.gate == "null":
            self.null_k = nn.Parameter(torch.zeros(self.h, self.dk))
            self.null_v = nn.Parameter(torch.zeros(self.h, self.dk))
        else:
            self.null_k = self.null_v = None
        # Error-mining diagnostics read the Q/K captured from the forward
        # pass (see ``collect_stats_and_errors``).  The capture is opt-in and
        # only used by the inference-mode statistics pass; ``_qk_capture`` is
        # cleared by the forward hook that consumes it.
        self._capture_qk = False
        self._qk_capture = None

    def _pack_qkv_grad(self, x: torch.Tensor, B: int, h: int, T: int,
                       dk: int) -> torch.Tensor:
        """``(B, T, d_model) -> (3, B*h, T, dk)`` head-major layout.

        The three slices are handed to the attention kernel as a single
        tensor.
        """
        step = _qkv_step(B, T, x.size(-1))
        packed = x.new_empty(3, B * h, T, dk)
        pv = packed.view(3, B, h, T, dk)
        for b0 in range(0, B, step):
            b1 = min(B, b0 + step)
            qkv = F.linear(x[b0:b1], self.qkv.weight)
            pv[:, b0:b1].copy_(
                qkv.view(b1 - b0, T, 3, h, dk).permute(2, 0, 3, 1, 4))
        return packed

    def _ent_perm(self, T: int, device: torch.device) -> torch.Tensor:
        """Fixed position permutation for the shuffled-entropy active control.

        Seeded only from ``(layer_idx, T)``, so every training seed of the
        placebo arm reads the *same* placebo signal: the treatment cannot
        vary with the seed, which keeps the arm paired with its controls
        and leaves the between-seed variance interpretable.  Cached per
        device like the causal mask; created outside inference mode so the
        tensor stays usable in gradient-enabled forwards.
        """
        key = (T, device)
        perm = self._ent_perm_cache.get(key)
        if perm is None:
            seed = (_ENT_PLACEBO_SEED * (self.layer_idx + 1)
                    + 0x85EBCA6B * T) & 0x7FFFFFFF
            with torch.inference_mode(False):
                gen = torch.Generator()
                gen.manual_seed(seed)
                perm = torch.randperm(T, generator=gen).to(device)
            self._ent_perm_cache[key] = perm
        return perm

    def _log_vis_vec(self, T: int, device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
        """Per-row visible-token divisor ``log(max(t+1, 2))``.

        Causal row ``t`` attends to ``t+1`` keys, so its entropy ceiling is
        ``log(t+1)``; the ``max(., 2)`` clamp mirrors the registered gate's
        ``log(max(T, 2))`` so the single-key first row stays finite (its
        entropy is exactly 0, so its normalised value is 0 either way).
        Used only by the ``calib_pos_norm`` ablation.  Cached per
        ``(T, device, dtype)`` and built outside inference mode so both
        the gradient and the no-grad paths can read it.
        """
        key = (T, device, dtype)
        vec = self._log_vis_cache.get(key)
        if vec is None:
            with torch.inference_mode(False):
                t = torch.arange(1, T + 1, device=device, dtype=dtype)
                vec = t.log().clamp_min(math.log(2.0))
            self._log_vis_cache[key] = vec
        return vec

    def _gate_proj(self, x: torch.Tensor, B: int, T: int, C: int,
                   grad_on: bool) -> torch.Tensor:
        """Project the gate.

        With grad on the projection runs on the flattened ``(B*T, C)`` rows;
        without grad the caller consumes the plain ``(B, T, C)`` layout.
        """
        if grad_on:
            return F.linear(x.reshape(B * T, C), self.gate.weight,
                            self.gate.bias)
        out = _get_scratch(("gate_out",), x.device, x.dtype, B * T * C)
        _linear_into(x.reshape(B * T, C), self.gate.weight, self.gate.bias,
                     out.view(B * T, C))
        return out.view(B, T, C)

    def _row_stats(self, rows: torch.Tensor, T: int, is_null: bool):
        """Entropy / max-prob / null-mass for a ``(B, H, S)`` slice of the
        attention distribution.  ``S`` is ``T`` or ``T + 1``, and in the
        ``T + 1`` case the null slot is the *leading* column - the layout both
        the block-wise kernel and the explicit path build.

        The rows always come from the *clean* softmax (attention dropout is
        never applied to them): a dropped row is scaled by the keep-prob and
        therefore is not a distribution any more, which would bias the
        entropy by ``keep*(log keep - 1)`` if it were read as one.

        For the null variant, ``null_mass`` is the leading slot and the
        entropy / max-prob describe the real-key distribution *conditional on
        not choosing null* (the null column is removed and the remainder
        renormalized), keeping those readouts comparable with the non-null
        gates.  The caller's ``rows`` tensor is left unchanged.
        """
        with torch.no_grad():
            if is_null:
                null_mass = rows[..., 0]
                pr = rows[..., 1:]
                n0, n1 = pr.size(0), pr.size(1)
                ssum = _get_scratch(("rowstats_sum",), pr.device, pr.dtype,
                                    n0 * n1)[:n0 * n1].view(n0, n1, 1)
                torch.sum(pr, dim=-1, keepdim=True, out=ssum)
                ssum.clamp_min_(_EPS)
                # Out-of-place: diagnostics must not mutate the attention
                # probabilities they are reading (nor a caller's tensor).
                pr = pr / ssum
            else:
                null_mass = None
                pr = rows
            buf = _get_scratch(("rowent",), pr.device, pr.dtype, pr.numel())
            return _row_entropy(pr, _EPS, buf=buf), pr.max(-1).values, null_mass

    def _row_stats_at(self, q, k, B, h, T, stat_pos, is_null, ent_full=None):
        """Diagnostics for the queried rows only.

        Only the rows at ``stat_pos`` are scored.  ``q`` is ``(B, h, T, dk)``,
        ``k`` the head-major key tensor.  ``ent_full`` optionally supplies the
        full-sequence entropy ``(B, h, T)`` already produced by the calibrated
        kernel.
        """
        dk = q.size(-1)
        cols = T + 1 if is_null else T
        n = B * h
        q_sel = _pick_rows(q, stat_pos).unsqueeze(1).reshape(-1, 1, dk)
        mask_sel = _mask_rows(stat_pos, T, h, q.dtype)
        with torch.no_grad():
            if is_null:
                att_sel = _get_scratch(("rowstats_att_null",), q.device,
                                       q.dtype, n * (T + 1))[
                    :n * (T + 1)].view(n, 1, T + 1)
                torch.baddbmm(mask_sel, q_sel, k.transpose(1, 2),
                              beta=1.0, alpha=self.inv_scale,
                              out=att_sel[..., 1:])
                s0 = att_sel[..., :1]
                torch.matmul(q_sel.reshape(B, h, 1, dk),
                             self.null_k.view(1, h, dk, 1), out=s0.view(B, h, 1, 1))
                s0.mul_(self.inv_scale)
            else:
                att_sel = _get_scratch(("rowstats_att",), q.device, q.dtype,
                                       n * T)[:n * T].view(n, 1, T)
                torch.baddbmm(mask_sel, q_sel, k.transpose(1, 2),
                              beta=1.0, alpha=self.inv_scale, out=att_sel)
            probs = _softmax_inplace(att_sel).view(B, h, cols)
            if ent_full is not None and not is_null:
                return (_pick_rows(ent_full, stat_pos),
                        probs.max(-1).values, None)
            return self._row_stats(probs, T, is_null)

    def forward(self, x, stat_pos: torch.Tensor | None = None):
        B, T, C = x.shape
        h, dk = self.h, self.dk
        bh = B * h
        need_stats = stat_pos is not None
        kind = self.gate_kind
        is_null = kind == "null"
        is_calib = kind == "calib"
        # Ablation switches: with the defaults both are on, so the full
        # calibrated gate is unchanged.  The flags only decide whether the
        # corresponding term is subtracted (and therefore whether the
        # attention entropy has to be produced at all).
        calc_len = is_calib and self.calib_a is not None
        calc_ent = is_calib and self.calib_c is not None
        drop_p = self.cfg.dropout
        dropping = self.training and drop_p > 0.0
        grad_on = torch.is_grad_enabled()

        if grad_on:
            packed = self._pack_qkv_grad(x, B, h, T, dk)
        else:
            packed = _get_scratch(
                ("packed",), x.device, x.dtype, 3 * bh * T * dk).view(3, bh, T, dk)
            pv = packed.view(3, B, h, T, dk)
            # Staging buffer feeding the permute-copy.
            step = _qkv_step(B, T, C)
            qkv_buf = _get_scratch(("qkv_out",), x.device, x.dtype,
                                   step * T * 3 * C).view(step * T, 3 * C)
            for b0 in range(0, B, step):
                b1 = min(B, b0 + step)
                rows = (b1 - b0) * T
                qkv_b = qkv_buf[:rows]
                _linear_into(x[b0:b1].reshape(rows, C), self.qkv.weight,
                             None, qkv_b)
                pv[:, b0:b1].copy_(
                    qkv_b.view(b1 - b0, T, 3, h, dk).permute(2, 0, 3, 1, 4))

        if self._capture_qk and not grad_on:
            self._qk_capture = (packed[0], packed[1])
        ent = None
        y4 = None
        ybt = None
        stats = None
        if not dropping:
            if grad_on and is_null:
                y4 = _NullAttn.apply(packed, self.null_k, self.null_v,
                                     self.inv_scale, B, grad_on).view(B, h, T, dk)
            elif grad_on:
                y, ent = _CalibAttn.apply(packed, self.inv_scale, _EPS,
                                          calc_ent)
                y4 = y.view(B, h, T, dk)
                ent = None if ent is None else ent.view(B, h, T)
            else:
                # No-grad: the kernel writes each block into the
                # ``(B, T, H, dk)`` projection input; a gate, if any, is
                # applied to it below.
                ybt = _get_scratch(("attn_y",), x.device, x.dtype,
                                   B * T * C).view(B, T, h, dk)
                if is_null:
                    _NullAttn.apply(packed, self.null_k, self.null_v,
                                    self.inv_scale, B, False, ybt)
                else:
                    _, ent = _CalibAttn.apply(packed, self.inv_scale,
                                              _EPS, calc_ent, ybt)
                ent = None if ent is None else ent.view(B, h, T)
            if need_stats:
                stats = self._row_stats_at(
                    packed[0].view(B, h, T, dk), packed[1], B, h, T,
                    stat_pos, is_null, ent_full=ent)
            if not grad_on and x.is_cuda:
                del packed
                _release_scratch("packed", "qkv_out")
        else:
            k_ = packed[1].view(B, h, T, dk)
            if is_null:
                nk = self.null_k.view(1, h, 1, dk).expand(B, h, 1, dk)
                kv_k = torch.cat([nk, k_], dim=2)
                att = torch.baddbmm(
                    _causal_null_mask(T, x.device, x.dtype), packed[0],
                    kv_k.reshape(bh, T + 1, dk).transpose(1, 2),
                    beta=1.0, alpha=self.inv_scale)
                p = _SoftmaxInPlace.apply(att)
                if not need_stats:
                    stats = None
                else:
                    stats = self._row_stats(
                        _pick_rows(p.view(B, h, T, T + 1), stat_pos), T,
                        True)
                if dropping:
                    p = self.attn_dropout(p)
                pb = p.view(bh, T, T + 1)
                yb = torch.bmm(pb[..., 1:], packed[2]).view(B, h, T, dk)
                y4 = yb.addcmul_(p.view(B, h, T, T + 1)[..., :1],
                                 self.null_v.view(1, h, 1, dk))
            else:
                mask = _causal_mask(T, x.device, x.dtype)
                att = torch.baddbmm(mask, packed[0], packed[1].transpose(1, 2),
                                    beta=1.0, alpha=self.inv_scale)
                p = _SoftmaxInPlace.apply(att)
                if calc_ent:
                    ent = _RowEntropy.apply(p, _EPS).view(B, h, T)
                if not need_stats:
                    stats = None
                elif calc_ent:
                    rows = _pick_rows(p.view(B, h, T, T), stat_pos)
                    stats = (_pick_rows(ent, stat_pos), rows.max(-1).values, None)
                else:
                    stats = self._row_stats(
                        _pick_rows(p.view(B, h, T, T), stat_pos), T, False)
                if dropping:
                    p = self.attn_dropout(p)
                y4 = torch.bmm(p, packed[2]).view(B, h, T, dk)

        gate_val = None
        if kind == "query":
            gate_val = self._gate_proj(x, B, T, C, grad_on)
            gate_val.sigmoid_()
        elif is_calib:
            gate_val = self._gate_proj(x, B, T, C, grad_on)
            if calc_len or calc_ent:
                log_len = self._log_len_cache.get(T)
                if log_len is None:
                    log_len = math.log(max(T, 2))
                    self._log_len_cache[T] = log_len
            if calc_ent:
                # Active control: the gate reads a position-shuffled copy of
                # the real entropy (same values and marginal distribution,
                # decorrelated from the queried row's own attention).  Only
                # the gate input is shuffled; every reported statistic below
                # still reads the real entropy.
                ent_gate = ent
                if self._ent_perm_cache is not None:
                    perm = self._ent_perm(T, x.device)
                    if grad_on:
                        ent_gate = ent.index_select(2, perm)
                    else:
                        ent_gate = _get_scratch(
                            ("calib_ent_shuf",), x.device, x.dtype,
                            B * h * T).view(B, h, T)
                        torch.index_select(ent, 2, perm, out=ent_gate)
                calib_cs = self.calib_c * self.calib_scale
                # Length normalisation ablation (E2/H2g): the registered
                # gate divides the concentration signal by log T; the raw
                # arm uses the same row-mean entropy without that division.
                # Everything else (parameters, length term, kernel) is
                # identical, so the contrast is single-factor.
                #
                # Position-conflation ablation: the registered gate divides
                # *every* row by the global log T, although row t's entropy
                # ceiling is log(t+1); with calib_pos_norm the divisor (and
                # the length term below) use the per-row visible-token
                # length instead.  On the final query row the two divisors
                # coincide, so registered readouts are unchanged.
                if self.calib_pos_norm:
                    log_div = self._log_vis_vec(T, x.device, ent_gate.dtype)
                else:
                    log_div = log_len
                if grad_on:
                    shift = ent_gate.mean(dim=1)
                    if self.calib_ent_norm:
                        shift = shift.div_(log_div)
                    shift.mul_(calib_cs)
                else:
                    shift = _get_scratch(("calib_conc",), x.device,
                                         x.dtype, B * T)[:B * T].view(B, T)
                    torch.mean(ent_gate, dim=1, out=shift)
                    if self.calib_ent_norm:
                        shift.div_(log_div)
                    shift.mul_(calib_cs)
                if calc_len:
                    shift.add_(self.calib_a * log_div)
                gate_val.sub_(shift.reshape(B * T, 1) if grad_on
                              else shift.unsqueeze(-1))
            elif calc_len:
                gate_val.sub_(self.calib_a * log_len)
            tau = F.softplus(self.calib_tau) + 0.1
            gate_val.mul_(1.0 / tau).sigmoid_()
            del ent
            ent = None

        if ybt is not None:
            # The kernel already wrote the projection input; a gate, if any,
            # is applied to it in place.
            if gate_val is not None:
                torch.mul(gate_val.view(B, T, h, dk), ybt, out=ybt)
            y = ybt.view(B, T, C)
        elif gate_val is None:
            if grad_on:
                y = y4.transpose(1, 2).reshape(B, T, C)
            else:
                y = _get_scratch(("attn_y",), x.device, x.dtype,
                                 B * T * C).view(B, T, h, dk)
                y.copy_(y4.transpose(1, 2))
                y = y.view(B, T, C)
        else:
            if grad_on:
                y = torch.mul(gate_val.view(B, T, h, dk), y4.transpose(1, 2))
                y = y.reshape(B, T, C)
            else:
                y4t = _get_scratch(("attn_y",), x.device, x.dtype, B * T * C)
                torch.mul(gate_val.view(B, T, h, dk), y4.transpose(1, 2),
                          out=y4t.view(B, T, h, dk))
                y = y4t.view(B, T, C)
        del y4

        if grad_on:
            out = self.proj(y)
        else:
            out = _get_scratch(("proj_out",), x.device, x.dtype,
                               B * T * C).view(B, T, C)
            _linear_into(y.view(B * T, C), self.proj.weight, None,
                         out.view(B * T, C))
        if x.is_cuda and not grad_on:
            _release_scratch(
                "qkv_out", "packed", "calib_y", "calib_s",
                "calib_f", "calib_ent", "calib_ent_shuf", "calib_conc",
                "null_y", "null_s", "null_s0",
                "null_m", "rowstats_att", "rowstats_att_null", "rowstats_sum",
                "rowent", "gate_out", "attn_y")
        if not need_stats:
            return out, None

        gate_stat = None
        if gate_val is not None:
            gate_stat = _pick_tokens(gate_val.view(B, T, C), stat_pos).detach()
        return out, {
            "entropy": stats[0].detach(),
            "maxprob": None if stats[1] is None else stats[1].detach(),
            "gate": gate_stat,
            "null_mass": None if stats[2] is None else stats[2].detach().clone(),
        }


class _GeluLinear(nn.Linear):
    """``nn.Linear`` fed the *pre-activation*: applies GELU before the
    projection.
    """

    def forward(self, h):
        if torch.is_grad_enabled():
            return F.linear(F.gelu(h), self.weight, self.bias)
        shape = h.shape
        rows = h.numel() // shape[-1]
        torch.ops.aten.gelu_(h, approximate="none")
        out = _get_scratch(("mlp_down",), h.device, h.dtype,
                           rows * self.out_features)
        _linear_into(h.view(rows, shape[-1]), self.weight, self.bias,
                     out.view(rows, self.out_features))
        return out.view(*shape[:-1], self.out_features)


class Block(nn.Module):
    def __init__(self, cfg: Cfg, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        use_post = cfg.norm == "post" or (cfg.norm == "mix" and layer_idx >= cfg.n_layers // 2)
        self.post = use_post
        self.ln1 = make_norm(cfg.norm, cfg.d_model)
        self.ln2 = make_norm(cfg.norm, cfg.d_model)
        self.attn = Attention(cfg, layer_idx)
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

    @staticmethod
    def _add_residual(x: torch.Tensor, branch: torch.Tensor,
                      scale: torch.Tensor | None,
                      grad_on: bool) -> torch.Tensor:
        """Add ``branch`` to ``x`` with optional LayerScale."""
        if scale is None:
            return branch.add_(x)
        if grad_on:
            return torch.addcmul(x, branch, scale)
        return x.addcmul_(branch, scale)

    def _mlp_residual(self, x: torch.Tensor, residual: torch.Tensor,
                      scale: torch.Tensor | None,
                      grad_on: bool) -> torch.Tensor:
        """Apply the MLP to ``x`` and add its output to ``residual``."""
        if scale is not None:
            return self._add_residual(residual, self._mlp(x), scale, grad_on)
        if grad_on:
            h = x.reshape(-1, x.shape[-1])
            batch_shape = residual.shape
            return self._mlp(h).add_(
                residual.reshape(-1, batch_shape[-1])).view(batch_shape)
        return self._mlp(x).add_(residual)

    def _mlp(self, h: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            return self.mlp(h)
        c = self.cfg.d_model
        n = h.numel() // c
        # Row-chunked up projection.
        if n > _MLP_CHUNK_ROWS:
            step = _balanced_step(n, _MLP_CHUNK_ROWS)
            up = _get_scratch(("mlp_up",), h.device, h.dtype,
                              4 * step * c)
            down = _get_scratch(("mlp_down_big",), h.device, h.dtype, n * c)
            gelu = self.mlp[1]
            h2 = h.reshape(n, c)
            for r0 in range(0, n, step):
                r1 = min(n, r0 + step)
                b = r1 - r0
                upb = up[:4 * b * c].view(b, 4 * c)
                _linear_into(h2[r0:r1], self.mlp[0].weight, self.mlp[0].bias, upb)
                torch.ops.aten.gelu_(upb, approximate="none")
                _linear_into(upb, gelu.weight, gelu.bias,
                             down[r0 * c:(r0 + b) * c].view(b, c))
            return self.mlp[2](down.view_as(h))
        up = _get_scratch(("mlp_up",), h.device, h.dtype, 4 * n * c)
        _linear_into(h.reshape(n, c), self.mlp[0].weight, self.mlp[0].bias,
                     up.view(n, 4 * c))
        m = self.mlp[2](self.mlp[1](up.view(n, 4 * c))).view_as(h)
        if h.is_cuda:
            _release_scratch("mlp_up")
        return m

    @staticmethod
    def _stamp_write_ratio(stats, key, branch, stream, scale):
        """Record ``||scale * branch|| / ||stream||`` into ``stats``.

        E3's ``resid_norm`` is read *after* the block's final LayerNorm in
        post-norm blocks but on the raw residual stream in pre/seednorm
        blocks, so its level is not comparable across normalisation
        schemes.  The relative branch write is: each branch is measured
        against the stream it is actually added to, before any
        normalisation is applied to the sum, which makes the ratio a
        cross-scheme comparable "how much did this layer write" readout.
        Statistics-only (detached); must be called before the in-place
        residual add consumes ``branch``.
        """
        if stats is None:
            return
        with torch.no_grad():
            write = branch.detach()
            if scale is not None:
                write = write * scale.detach()
            num = write.norm(dim=-1).mean(-1)
            den = stream.detach().norm(dim=-1).mean(-1).clamp_min(1e-12)
            stats[key] = num / den

    def forward(self, x, stat_pos=None):
        ls1, ls2 = self.ls1, self.ls2
        grad_on = torch.is_grad_enabled()
        # The write ratios are statistics-only companions: compute them
        # exactly when the per-layer stats dict is being requested.
        want_writes = stat_pos is not None
        if self.post:
            a, s = self.attn(x, stat_pos)
            if want_writes:
                self._stamp_write_ratio(s, "attn_write_ratio", a, x, ls1)
            x = self._add_residual(x, a, ls1, grad_on)
            x = self.ln1(x)
            if want_writes:
                m = self._mlp(x)
                self._stamp_write_ratio(s, "mlp_write_ratio", m, x, ls2)
                x = self._add_residual(x, m, ls2, grad_on)
            else:
                x = self._mlp_residual(x, x, ls2, grad_on)
            x = self.ln2(x)
        else:
            x_norm = self.ln1(x)
            a, s = self.attn(x_norm, stat_pos)
            if want_writes:
                self._stamp_write_ratio(s, "attn_write_ratio", a, x, ls1)
            x = self._add_residual(x, a, ls1, grad_on)
            x_norm = self.ln2(x)
            if want_writes:
                m = self._mlp(x_norm)
                self._stamp_write_ratio(s, "mlp_write_ratio", m, x, ls2)
                x = self._add_residual(x, m, ls2, grad_on)
            else:
                x = self._mlp_residual(x_norm, x, ls2, grad_on)
        return x, s


class CAGE(nn.Module):
    _POS_EXTRAPOLATION_MODES = ("learned", "fold", "zero")

    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        # Eval-time position-extrapolation control (E2 confound audit).
        # "learned" (default) is the registered behaviour: positions are
        # indexed directly, including any position beyond the trained
        # length whose embedding has never been updated by training.
        # "fold" / "zero" remap / remove those untrained rows; see
        # ``set_pos_extrapolation``.
        self.pos_extrapolation = "learned"
        self.pos_extrapolation_from = None
        self.drop = nn.Dropout(cfg.dropout, inplace=True)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.ln_f = make_norm(cfg.norm, cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            if getattr(m, "_is_gate", False) or \
                    getattr(m, "_preinitialized", False):
                return
            if m.weight is self.tok.weight:
                # Already initialised through the shared embedding weight.
                return
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def set_pos_extrapolation(self, mode, trained_len=None):
        """Select the eval-time handling of untrained position rows.

        E2 trains at a fixed length ``T_train`` and then evaluates longer
        sequences; with learned absolute position embeddings every position
        ``>= T_train`` carries an embedding that was never updated by
        training, so a length-sweep accuracy drop mixes the mechanism under
        test with untrained position vectors.  This control changes only
        how those rows are read at evaluation time:

        - ``"learned"`` (default, the registered behaviour): index the
          embedding table directly.
        - ``"fold"``: map position ``p >= trained_len`` to
          ``p % trained_len`` - extrapolated rows reuse *trained*
          embeddings (position folding).
        - ``"zero"``: drop the position signal for extrapolated rows
          (their embedding term is replaced by zeros).

        If folding (or zeroing) recovers most of the accuracy drop, the
        degradation is attributable to position-vector extrapolation
        rather than to attention/gate behaviour at longer ranges.
        ``trained_len`` is required (a positive int) for the control modes
        and ignored for ``"learned"``.
        """
        if mode not in self._POS_EXTRAPOLATION_MODES:
            raise ValueError(
                f"pos extrapolation mode must be one of "
                f"{list(self._POS_EXTRAPOLATION_MODES)}, got {mode!r}")
        if mode == "learned":
            self.pos_extrapolation_from = None
            self.pos_extrapolation = mode
            return
        if isinstance(trained_len, bool) or \
                not isinstance(trained_len, int):
            raise ValueError(
                f"trained_len must be a positive int for mode {mode!r}, "
                f"got {trained_len!r}")
        if trained_len < 1:
            raise ValueError(
                f"trained_len must be a positive int for mode {mode!r}, "
                f"got {trained_len!r}")
        self.pos_extrapolation_from = trained_len
        self.pos_extrapolation = mode

    def _pos_rows(self, T: int) -> torch.Tensor:
        """The ``(T, C)`` position rows added to the token embeddings."""
        mode = self.pos_extrapolation
        if mode == "learned":
            return self.pos.weight[:T]
        start = self.pos_extrapolation_from
        weight = self.pos.weight
        if start is None or T <= start:
            return weight[:T]
        idx = _arange(T, weight.device)
        if mode == "fold":
            folded = torch.where(idx >= start, idx % start, idx)
            return weight.index_select(0, folded)
        keep = (idx < start).to(weight.dtype).unsqueeze(1)
        return weight[:T] * keep

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
                positions = sup_pos.reshape(-1)
                if positions.numel() != targets.size(0):
                    raise ValueError(
                        f"sup_pos must give one position per row "
                        f"({targets.size(0)}), got {positions.numel()}")
                if positions.dtype == torch.bool \
                        or positions.is_floating_point() \
                        or positions.is_complex():
                    raise ValueError(
                        f"sup_pos positions must be integers, got "
                        f"{positions.dtype}")
                if bool((positions >= T).any()):
                    raise ValueError(
                        f"sup_pos positions must be < sequence length {T} "
                        f"(negative entries are ignored), got "
                        f"{int(positions.max())}")
                sup = _arange(targets.size(0), targets.device).mul(
                    targets.size(1)).add_(positions.clamp_min(0))
                sup = sup[positions >= 0]
            else:
                sup = (flat_t != -1).nonzero(as_tuple=True)[0]
                if flat_t.numel() and sup.numel() == flat_t.numel():
                    sup = None
        if sup is not None and sup.numel() == 0:
            raise ValueError(
                "targets contain no supervised position (all entries are "
                "-1, or sup_pos selects only ignored positions): the loss "
                "is undefined, so the batch must not be trained on")
        if sup is not None and not bool(
                (flat_t.index_select(0, sup) != -1).any()):
            raise ValueError(
                "the selected supervised targets are all -1: the loss is "
                "undefined, so the batch must not be trained on")
        if torch.is_grad_enabled():
            x = self.tok(idx)
        else:
            n = idx.numel()
            base = _get_scratch(("embed_out",), idx.device,
                                self.tok.weight.dtype, n * self.cfg.d_model)
            x = base[:n * self.cfg.d_model].view(idx.size(0), T,
                                                 self.cfg.d_model)
            torch.index_select(self.tok.weight, 0, idx.reshape(-1),
                               out=x.view(n, self.cfg.d_model))
        x.add_(self._pos_rows(T))
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
            elif torch.is_grad_enabled():
                h = self.ln_f(x).reshape(-1, self.cfg.d_model)
            else:
                h = _norm_out(self.ln_f, x, "ln_f_out").reshape(
                    -1, self.cfg.d_model)
            logits = self.head(h)
            loss = F.cross_entropy(logits, flat_t, ignore_index=-1)
        else:
            last = x[:, -1:]
            if torch.is_grad_enabled():
                logits = self.head(self.ln_f(last))
            else:
                logits = self.head(_norm_out(self.ln_f, last, "ln_f_out"))
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
            try:
                model = torch.compile(model, mode="reduce-overhead",
                                      fullgraph=False)
            except Exception as exc:  # noqa: BLE001 - optional acceleration
                # ``torch.compile`` can exist but be unusable on this build
                # (missing backend/triton, unsupported Python, ...).  D9 makes
                # it an explicit opt-in optimisation, not a requirement: a
                # failed compile must fall back to the eager model with a
                # warning instead of breaking an otherwise runnable config.
                print(f"Warning: torch.compile failed ({exc}); skipping "
                      f"compilation")
        else:
            print("Warning: torch.compile not available, skipping compilation")
    return model