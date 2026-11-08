# ruff: noqa: F401, F821
"""Synthetic task data.

Associative-recall style LM: sequence of key-value pairs plus a query; the
model must recall the value. This makes attention quality (and false
concentration) measurable: we know the *correct* target token. Length
generalisation (E2) evaluates this same task at multiple sequence lengths
via ``length_sweep_lengths``; ``length_sweep_plan`` records the *whole*
sweep decision (requested grid, realized grid, dropped lengths and the
measured extrapolation range) so the executed design is never implicit.
``make_length_linked_batch`` additionally extends one validated batch in
place for the within-item E2 counterfactual (the same query/answer/pair
index gains distractor pairs).

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

def length_sweep_plan(base: int = 16, factor: int = 2, n: int = 4,
                       max_len: int = 256, min_lengths: int = 2) -> dict:
    """The *whole* length-sweep decision as one JSON-safe record.

    :func:`length_sweep_lengths` answers "which lengths do I evaluate?".
    This helper answers the question a pre-registration actually has to
    answer: "which lengths did I *ask* for, which ones are actually
    runnable, why was the rest dropped, and over what range does the
    resulting claim hold?".

    A length ``L`` occupies ``2*L + 2`` sequence positions (pairs + SEP +
    query), and the model's absolute position table holds ``max_len``
    positions, so a length that does not fit cannot be run at all - it
    would index an out-of-range position embedding.  The geometric sweep
    therefore stops at the first length that does not fit, and every
    longer length is dropped with it.  That drop used to be invisible to
    the caller: the registered sweep in ``EXPERIMENT_PROTOCOL.md`` names
    four lengths, while the default geometry (``max_len=256``) only
    realizes three of them, so the executed design and the documented
    design could drift apart without any artifact recording it.

    The returned record makes every part of that decision explicit:

    * ``requested_lengths`` - the geometric prefix that was considered,
      including the first length that does *not* fit (the full requested
      grid has ``requested_count`` entries; ``requested_grid_complete``
      says whether all of them were realized);
    * ``realized_lengths`` - the lengths that are actually evaluated;
    * ``dropped`` - the first dropped length together with the number of
      sequence positions it would need and the ``max_len`` it exceeded;
    * ``n_dropped`` - how many of the ``n`` requested lengths are not
      realized (every length beyond the first dropped one is dropped too);
    * ``min_max_len_for_requested`` - the ``max_len`` that would realize
      the whole requested grid;
    * ``coverage`` - the measured extrapolation range in pairs *and* in
      sequence positions, plus the two ratios, so a "degrades more slowly
      with length" claim is always read against the range that was
      actually measured;
    * ``slope_degenerate`` - whether the realized grid has fewer than
      three lengths, in which case the ``accuracy_slope_log_len`` readout
      is not a slope at all but the endpoint difference rescaled, and a
      mid-sweep reversal cannot be detected.

    ``min_lengths`` is the number of realized lengths the caller needs.
    The default of two keeps the historical contract (a one-point sweep
    makes the base-to-longest drop and the entropy drift identically
    zero, so H2a/H2b would read a degenerate "no degradation" as
    evidence).  A caller whose readout needs an interior point - the
    E2 degradation-shape slope - asks for three, and a geometry that
    cannot deliver it fails here instead of producing a sweep that
    answers nothing.

    Raises ``ValueError`` for a non-integer / bool / out-of-range
    geometry and for a grid that realizes fewer than ``min_lengths``
    lengths.
    """
    values = {}
    for name, value in (("base", base), ("factor", factor), ("n", n),
                        ("max_len", max_len), ("min_lengths", min_lengths)):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, not bool")
        try:
            values[name] = operator.index(value)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be an integer, got {value!r}") from exc
    base, factor, n, max_len, min_lengths = (
        values[name] for name in
        ("base", "factor", "n", "max_len", "min_lengths"))
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
    if min_lengths < 1:
        raise ValueError(f"min_lengths must be >= 1, got {min_lengths}")
    fits = []
    first_dropped = None
    current = base
    # The sweep is strictly increasing, so once one length no longer fits it
    # and every later length are dropped.  Stop there instead of computing
    # ``factor ** i`` for a huge ``n`` (a Python big integer can grow without
    # bound and turn a configuration error into a long stall).
    for _ in range(n):
        if 2 * current + 2 > max_len:
            first_dropped = current
            break
        fits.append(current)
        current *= factor
    if len(fits) < min_lengths:
        if min_lengths == 2:
            # Keep the historical wording: this message is asserted by the
            # data-module self-check and quoted in the protocol.
            raise ValueError(
                f"length sweep needs at least two fitted lengths, got {fits} "
                f"under max_len={max_len} (base={base}, factor={factor}, "
                f"n={n}); raise max_len or reduce base/n"
            )
        raise ValueError(
            f"length sweep needs at least {min_lengths} fitted lengths, got "
            f"{fits} under max_len={max_len} (base={base}, factor={factor}, "
            f"n={n}); raise max_len, lower min_lengths or reduce base/n"
        )
    requested = list(fits)
    dropped = []
    n_dropped = max(0, n - len(fits))
    if first_dropped is not None:
        requested.append(first_dropped)
        dropped.append({
            "length": int(first_dropped),
            "required_positions": int(2 * first_dropped + 2),
            "max_len": int(max_len),
            "reason": (
                f"a length of {first_dropped} pairs needs "
                f"{2 * first_dropped + 2} sequence positions, which exceeds "
                f"the registered max_len={max_len} positions; the model has "
                f"no position embedding for those positions, so the length "
                f"cannot be evaluated at all"),
        })
    positions = {str(L): int(2 * L + 2) for L in fits}
    base_positions = positions[str(fits[0])]
    longest_positions = positions[str(fits[-1])]
    coverage = {
        "base_pairs": int(fits[0]),
        "longest_pairs": int(fits[-1]),
        "base_positions": int(base_positions),
        "longest_positions": int(longest_positions),
        "pair_ratio": float(fits[-1] / fits[0]),
        "position_ratio": float(longest_positions / base_positions),
    }
    # The minimal ``max_len`` that would realize the whole requested grid.
    # The last requested length is ``base * factor ** (n - 1)``; when the
    # grid was truncated it is strictly larger than the first dropped
    # length, and for an extreme ``n`` it is not representable - in that
    # case the field is ``None`` and the next-length field below still
    # states what a single extra step would cost.
    last_requested = None
    if first_dropped is None:
        last_requested = fits[-1]
    else:
        remaining = n - len(fits)
        if 1 <= remaining <= 64:
            # ``first_dropped`` is ``base * factor ** k`` with
            # ``k == len(fits)``, so the last requested length is
            # ``base * factor ** (n - 1) == first_dropped * factor ** (n-1-k)``.
            last_requested = first_dropped * (factor ** (remaining - 1))
    return {
        "requested": {"base": int(base), "factor": int(factor),
                      "n": int(n), "min_lengths": int(min_lengths),
                      "max_len": int(max_len)},
        "requested_count": int(n),
        "requested_lengths": [int(L) for L in requested],
        "requested_grid_complete": bool(len(fits) == n),
        "realized_lengths": [int(L) for L in fits],
        "dropped": dropped,
        "n_dropped": int(n_dropped),
        "all_longer_dropped": bool(first_dropped is not None),
        "truncated": bool(first_dropped is not None),
        "min_max_len_for_next_length": (
            int(2 * first_dropped + 2) if first_dropped is not None
            else None),
        "min_max_len_for_requested": (
            int(2 * last_requested + 2) if last_requested is not None
            else None),
        "n_realized": int(len(fits)),
        "max_len": int(max_len),
        "sequence_positions": positions,
        "coverage": coverage,
        "slope_degenerate": bool(len(fits) < 3),
        "note": (
            "the requested grid is the geometric sequence base*factor**i "
            "for i < n; a length is dropped when 2*L+2 exceeds max_len, and "
            "every length beyond the first dropped one is dropped with it. "
            "The realized grid is the executed design - read every "
            "length-generalisation claim against 'coverage', not against "
            "the requested grid."),
    }


def length_sweep_lengths(base: int = 16, factor: int = 2, n: int = 4,
                         max_len: int = 256, min_lengths: int = 2):
    """``n`` geometrically spaced ``n_pairs`` values for the length sweep.

    A length ``L`` occupies ``2*L + 2`` positions (pairs + SEP + query),
    so anything that does not fit into ``max_len`` positions cannot
    be run at all and is dropped instead of producing an out-of-range
    position embedding at evaluation time.

    A sweep needs at least two fitted lengths: with a single point the
    base-to-longest drop and entropy drift are identically zero, so H2a
    and H2b would read a degenerate "no degradation" as evidence.  Raise
    rather than return such a one-point grid so the caller cannot train a
    full E2 grid that answers nothing.  ``min_lengths`` raises that floor
    for callers whose readout needs an interior point (the E2
    degradation-shape slope needs three).

    This is the list-only view of :func:`length_sweep_plan`; use the plan
    when the *requested* grid, the dropped lengths or the measured
    extrapolation range have to be recorded with the results.
    """
    return length_sweep_plan(base=base, factor=factor, n=n, max_len=max_len,
                             min_lengths=min_lengths)["realized_lengths"]


if __name__ == "__main__":
    raise SystemExit(_self_test())
