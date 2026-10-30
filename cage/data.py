# ruff: noqa: F401, F821
"""Synthetic task data.

Associative-recall style LM: sequence of key-value pairs plus a query; the
model must recall the value. This makes attention quality (and false
concentration) measurable: we know the *correct* target token. Length
generalisation (E2) evaluates this same task at multiple sequence lengths
via ``length_sweep_lengths``.  ``make_length_linked_batch`` additionally
extends one validated batch in place for the within-item E2 counterfactual
(the same query/answer/pair index gains distractor pairs).

Format (vocab 0..V-1):
    [k1 v1 k2 v2 ... kn vn, SEP, qi, <predict vi>]
The label for the position right after qi is vi.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import operator
from collections.abc import Mapping

import torch

from ._fragments import load_fragments

SEP_ID = 1
PAD_ID = 0
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32,
                   torch.int64)
_MAX_TORCH_INDEX = int(torch.iinfo(torch.long).max)

# ---------------------------------------------------------------------------
# Implementation layout
# ---------------------------------------------------------------------------
# The implementation is split by responsibility into the fragment files
# below.  They are compiled and executed into this module's namespace in
# order, so every helper keeps its original public/private name and module
# identity for callers and tests.
_FRAGMENTS = (
    "_data_task.py",
    "_data_audit.py",
    "_data_selfcheck.py",
)
load_fragments(globals(), _FRAGMENTS)
del _FRAGMENTS


def make_batch(batch: int, n_pairs: int, vocab: int, device, dense_lm: bool = False, gen: torch.Generator | None = None):
    """Returns idx, targets, eval_meta.

    eval_meta: dict with positions needed for false-concentration analysis:
      query_pos  (B,) index of qi in the sequence
      key_pos    (B,) index of the matching ki
      answer     (B,) correct value token
    """
    canonical = {}
    for name, value in (("batch", batch), ("n_pairs", n_pairs),
                        ("vocab", vocab)):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        try:
            canonical[name] = operator.index(value)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be an integer, got {value!r}") from exc
    if not isinstance(dense_lm, bool):
        raise ValueError(
            f"dense_lm must be a bool, got {dense_lm!r}")
    batch, n_pairs, vocab = (canonical["batch"], canonical["n_pairs"],
                             canonical["vocab"])
    if batch < 0:
        raise ValueError(f"batch must be non-negative, got {batch}")
    if batch > _MAX_TORCH_INDEX:
        raise ValueError(
            f"batch={batch} exceeds torch's signed 64-bit index range")
    check_data_geometry(n_pairs, vocab)
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
        targets = tgt_full.clone()
    else:
        targets = torch.full_like(tgt_full, -1)
        targets[b, query_pos] = answer
    meta = {"query_pos": query_pos, "key_pos": key_pos, "answer": answer}
    return idx, targets, meta

def length_sweep_lengths(base: int = 16, factor: int = 2, n: int = 4,
                         max_len: int = 256):
    """``n`` geometrically spaced ``n_pairs`` values for the length sweep.

    A length ``L`` occupies ``2*L + 2`` positions (pairs + SEP + query),
    so anything that does not fit into ``max_len`` positions cannot
    be run at all and is dropped instead of producing an out-of-range
    position embedding at evaluation time.

    A sweep needs at least two fitted lengths: with a single point the
    base-to-longest drop and entropy drift are identically zero, so H2a
    and H2b would read a degenerate "no degradation" as evidence.  Raise
    rather than return such a one-point grid so the caller cannot train a
    full E2 grid that answers nothing.
    """
    values = {}
    for name, value in (("base", base), ("factor", factor), ("n", n),
                        ("max_len", max_len)):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, not bool")
        try:
            values[name] = operator.index(value)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be an integer, got {value!r}") from exc
    base, factor, n, max_len = (values[name] for name in
                               ("base", "factor", "n", "max_len"))
    if base < 1:
        raise ValueError(f"base must be >= 1, got {base}")
    if factor <= 1:
        raise ValueError(
            f"factor must be > 1 so the sweep contains distinct lengths, "
            f"got {factor}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if max_len < 1:
        raise ValueError(f"max_len must be >= 1, got {max_len}")
    lengths = []
    current = base
    # The sweep is strictly increasing, so once one length no longer fits it
    # and every later length are dropped.  Stop there instead of computing
    # ``factor ** i`` for a huge ``n`` (a Python big integer can grow without
    # bound and turn a configuration error into a long stall).
    for _ in range(n):
        if 2 * current + 2 > max_len:
            break
        lengths.append(current)
        current *= factor
    fits = lengths
    if len(fits) < 2:
        raise ValueError(
            f"length sweep needs at least two fitted lengths, got {fits} "
            f"under max_len={max_len} (base={base}, factor={factor}, n={n}); "
            f"raise max_len or reduce base/n"
        )
    return fits


if __name__ == "__main__":
    raise SystemExit(_self_test())
