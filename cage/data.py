"""Synthetic task data.

Tasks
-----
1. ``copy`` / associative-recall style LM: sequence of key-value pairs plus
   a query; the model must recall the value. This makes attention quality
   (and false concentration) measurable: we know the *correct* target token.

2. ``shuffle``: length generalization - evaluate the same task at multiple
   sequence lengths.

Format (vocab 0..V-1):
    [k1 v1 k2 v2 ... kn vn, SEP, qi, <predict vi>]
The label for the position right after qi is vi.

``dense_lm`` additionally scores every next-token position of the pair
section.  Note that in this causal layout those targets are *irreducible*:
at position ``2i`` the model sees ``k_i`` but its target ``v_i`` sits at
``2i+1``, i.e. in the future, so the pair section can only teach the marginal
token frequency.  It therefore dilutes the recall gradient ~15:1 at
``n_pairs=8`` and the recall accuracy that E1/E2 measure collapses to chance
-- supervision should stay on the query position unless the marginal prior
is explicitly wanted.
"""

from __future__ import annotations

import torch

SEP_ID = 1
PAD_ID = 0


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
    n_avail = vocab - 2                      # 0/1 are reserved (PAD / SEP)
    if n_pairs > n_avail:                    # cannot sample without replacement
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
    key_pos = 2 * qi_idx                       # position of matching key in `idx`
    query_pos = torch.full_like(qi_idx, width + 1)  # last position: predict answer here
    if dense_lm:
        targets = tgt_full
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
    """
    lengths = [base * (factor ** i) for i in range(n)]
    fits = [L for L in lengths if 2 * L + 2 <= max_len]
    if not fits:
        raise ValueError(
            f"no sweep length fits max_len={max_len}: needs >= {2 * base + 2} "
            f"positions for base={base}"
        )
    return fits
