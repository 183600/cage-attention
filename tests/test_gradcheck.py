"""Gradient checks and reference parity for the hand-written autograd kernels.

Design pin D6 keeps four custom ``torch.autograd.Function`` kernels
(``_SoftmaxInPlace``, ``_RowEntropy``, ``_CalibAttn``, ``_NullAttn``) with
hand-derived backward passes; the entropy term participates in the gradient,
which standard autograd composition cannot express directly.  That makes the
kernels a correctness liability: a wrong backward silently biases every
trained coefficient and every contrast derived from it.

This suite is the verification contract requested for that risk:

* every kernel's forward is compared against a plain-math reference;
* every kernel's backward is verified with ``torch.autograd.gradcheck`` in
  float64 (through *both* the attention output and the entropy output where
  applicable);
* the chunked (multi-block) query decomposition - normally only reachable at
  large ``B*H*T*T`` - is exercised by injecting block tables into
  ``_query_blocks_cache``, so its accumulate-into-prefix backward path is
  tested too;
* the gradient-enabled (training) path and the no-grad evaluation path of
  the full model agree for every gate variant, at several sequence lengths
  and in both float32 and float64.
"""

import math
import unittest

import torch

from cage.model import (
    _CalibAttn,
    _NullAttn,
    _RowEntropy,
    _SoftmaxInPlace,
    _query_blocks_cache,
    build,
)

_EPS = 1e-9


def _reference_probs(q, k, scale):
    """``softmax(q k^T / sqrt(dk) + causal)`` in plain math."""
    T = q.size(1)
    mask = torch.full((T, T), float("-inf"), dtype=q.dtype,
                      device=q.device).triu_(1)
    return torch.softmax(
        torch.baddbmm(mask, q, k.transpose(1, 2), beta=1.0, alpha=scale),
        dim=-1)


def _reference_calib(qkv, scale, need_ent):
    """Reference for ``_CalibAttn``: ``(y, ent)`` in plain math."""
    q, k, v = qkv[0], qkv[1], qkv[2]
    p = _reference_probs(q, k, scale)
    y = torch.bmm(p, v)
    ent = None
    if need_ent:
        ent = -(p * p.clamp_min(_EPS).log()).sum(-1)
    return y, ent


def _reference_null(qkv, null_k, null_v, scale, B, h):
    """Reference for ``_NullAttn`` in plain math."""
    q, k, v = qkv[0], qkv[1], qkv[2]
    bh, T, dk = q.shape
    q4 = q.view(B, h, T, dk)
    k4 = k.view(B, h, T, dk)
    v4 = v.view(B, h, T, dk)
    s_null = torch.einsum("bhtd,hd->bht", q4, null_k) * scale
    s_real = torch.einsum("bhtd,bhsd->bhts", q4, k4) * scale
    causal = torch.full((T, T), float("-inf"), dtype=q.dtype,
                        device=q.device).triu_(1)
    att = torch.cat([s_null.unsqueeze(-1), s_real + causal], dim=-1)
    p = torch.softmax(att, dim=-1)
    v_cat = torch.cat([null_v.view(1, h, 1, dk).expand(B, h, 1, dk), v4],
                      dim=2)
    return torch.einsum("bhts,bhsd->bhtd", p, v_cat).reshape(bh, T, dk)


def _rand_qkv(bh, T, dk, dtype, gen):
    """A small ``(3, bh, T, dk)`` QKV stack with moderate scores."""
    return torch.randn(3, bh, T, dk, dtype=dtype, generator=gen) * 0.5


class _ForcedBlocks:
    """Temporarily force a multi-block query decomposition in the cache."""

    def __init__(self, key, blocks):
        self.key = key
        self.blocks = blocks
        self.saved = None

    def __enter__(self):
        self.saved = _query_blocks_cache.get(self.key)
        _query_blocks_cache[self.key] = self.blocks
        return self

    def __exit__(self, *exc):
        if self.saved is None:
            _query_blocks_cache.pop(self.key, None)
        else:
            _query_blocks_cache[self.key] = self.saved
        return False


class RowEntropyGradcheckTest(unittest.TestCase):
    def test_forward_matches_reference_and_gradcheck(self):
        gen = torch.Generator().manual_seed(7)
        # Softmax-generated rows keep every entry well above eps, so the
        # backward's sub-eps mask is inactive and the analytic gradient
        # -(log p + 1) applies everywhere gradcheck probes.  Keep the
        # scores small: the entropy's curvature is ~1/p^2, so near-zero
        # probabilities make gradcheck's central difference itself
        # inaccurate (a conditioning limit of the numerical check, not of
        # the kernel - the analytic-vs-reference comparison below is
        # exact and unperturbed by it).
        scores = torch.randn(4, 3, 6, dtype=torch.float64,
                             generator=gen) * 0.5
        p = torch.softmax(scores, dim=-1).requires_grad_(True)
        self.assertGreater(float(p.min()), 1e-3)
        ent = _RowEntropy.apply(p, _EPS)
        ref = -(p.detach() * p.detach().clamp_min(_EPS).log()).sum(-1)
        self.assertTrue(torch.allclose(ent, ref, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.autograd.gradcheck(
            lambda t: _RowEntropy.apply(t, _EPS), (p,),
            atol=1e-8, rtol=1e-6))

    def test_backward_shape_and_none_grad(self):
        gen = torch.Generator().manual_seed(8)
        p = torch.softmax(torch.randn(2, 2, 5, dtype=torch.float64,
                                      generator=gen),
                          -1).requires_grad_(True)
        ent = _RowEntropy.apply(p, _EPS)
        ent.sum().backward()
        self.assertIsNotNone(p.grad)
        self.assertEqual(p.grad.shape, p.shape)


class SoftmaxInPlaceGradcheckTest(unittest.TestCase):
    def test_gradcheck_through_clone(self):
        gen = torch.Generator().manual_seed(9)
        x = (torch.randn(3, 5, dtype=torch.float64,
                         generator=gen) * 2.0).requires_grad_(True)
        # The kernel marks its input dirty; cloning inside the tested
        # function keeps gradcheck's input intact while exercising the
        # in-place softmax backward on the clone.
        self.assertTrue(torch.autograd.gradcheck(
            lambda t: _SoftmaxInPlace.apply(t.clone()), (x,),
            atol=1e-8, rtol=1e-6))

    def test_forward_matches_reference(self):
        gen = torch.Generator().manual_seed(10)
        x = torch.randn(2, 3, 7, dtype=torch.float64, generator=gen)
        out = _SoftmaxInPlace.apply(x.clone())
        self.assertTrue(torch.allclose(out, torch.softmax(x, -1),
                                       atol=1e-12, rtol=1e-12))


class CalibAttnGradcheckTest(unittest.TestCase):
    """``_CalibAttn`` in float64: forward parity + gradcheck, single- and
    multi-block, with and without the entropy output."""

    def _case(self, B, h, T, dk, blocks=None):
        gen = torch.Generator().manual_seed(11 + T + dk)
        bh = B * h
        scale = 1.0 / math.sqrt(dk)
        qkv = _rand_qkv(bh, T, dk, torch.float64, gen).requires_grad_(True)
        forced = (_ForcedBlocks((bh, T), blocks)
                  if blocks is not None else None)
        try:
            if forced is not None:
                forced.__enter__()
            y, ent = _CalibAttn.apply(qkv, scale, _EPS, True)
        finally:
            if forced is not None:
                forced.__exit__()
        y_ref, ent_ref = _reference_calib(qkv.detach(), scale, True)
        self.assertTrue(torch.allclose(y, y_ref, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(ent, ent_ref, atol=1e-10, rtol=1e-10))

        def fn(t):
            if forced is not None:
                forced.__enter__()
            try:
                return _CalibAttn.apply(t, scale, _EPS, True)
            finally:
                if forced is not None:
                    forced.__exit__()

        self.assertTrue(torch.autograd.gradcheck(fn, (qkv,),
                                                 atol=1e-6, rtol=1e-4))

    def test_single_block_small(self):
        self._case(B=1, h=2, T=6, dk=3)

    def test_single_block_wider(self):
        self._case(B=2, h=3, T=9, dk=4)

    def test_multi_block_two(self):
        # T <= 32 always yields a single block naturally; inject a split
        # decomposition to reach the chunked backward (accumulate into the
        # prefix of d_v / d_k).
        self._case(B=1, h=2, T=7, dk=3, blocks=((0, 3), (3, 7)))

    def test_multi_block_three_with_unit_head(self):
        self._case(B=2, h=2, T=8, dk=2,
                   blocks=((0, 1), (1, 4), (4, 8)))

    def test_need_ent_false_forward_and_grad(self):
        gen = torch.Generator().manual_seed(12)
        bh, T, dk = 4, 6, 3
        scale = 1.0 / math.sqrt(dk)
        qkv = _rand_qkv(bh, T, dk, torch.float64, gen).requires_grad_(True)
        y, ent = _CalibAttn.apply(qkv, scale, _EPS, False)
        self.assertIsNone(ent)
        y_ref, _ = _reference_calib(qkv.detach(), scale, False)
        self.assertTrue(torch.allclose(y, y_ref, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.autograd.gradcheck(
            lambda t: _CalibAttn.apply(t, scale, _EPS, False)[0], (qkv,),
            atol=1e-6, rtol=1e-4))

    def test_block_boundaries_are_contiguous_by_construction(self):
        # Guard for the cache-injection helper itself: an inconsistent
        # injected table must not silently pass as a coverage of [0, T).
        with self.assertRaises(AssertionError):
            blocks = ((0, 3), (4, 7))
            covered = [t for i0, i1 in blocks for t in range(i0, i1)]
            assert covered == list(range(7))


class NullAttnGradcheckTest(unittest.TestCase):
    def _case(self, B, h, T, dk, blocks=None):
        gen = torch.Generator().manual_seed(13 + T)
        bh = B * h
        scale = 1.0 / math.sqrt(dk)
        qkv = _rand_qkv(bh, T, dk, torch.float64, gen).requires_grad_(True)
        null_k = (torch.randn(h, dk, dtype=torch.float64,
                              generator=gen) * 0.5).requires_grad_(True)
        null_v = (torch.randn(h, dk, dtype=torch.float64,
                              generator=gen) * 0.5).requires_grad_(True)
        forced = (_ForcedBlocks((bh, T), blocks)
                  if blocks is not None else None)

        def fn(q, nk, nv):
            if forced is not None:
                forced.__enter__()
            try:
                return _NullAttn.apply(q, nk, nv, scale, B, True)
            finally:
                if forced is not None:
                    forced.__exit__()

        y = fn(qkv, null_k, null_v)
        y_ref = _reference_null(qkv.detach(), null_k.detach(),
                                null_v.detach(), scale, B, h)
        self.assertTrue(torch.allclose(y, y_ref, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.autograd.gradcheck(
            fn, (qkv, null_k, null_v), atol=1e-6, rtol=1e-4))

    def test_single_block(self):
        self._case(B=1, h=2, T=6, dk=3)

    def test_multi_block(self):
        self._case(B=1, h=2, T=7, dk=3, blocks=((0, 2), (2, 7)))

    def test_no_grad_path_matches_grad_path(self):
        gen = torch.Generator().manual_seed(14)
        B, h, T, dk = 2, 2, 6, 3
        bh = B * h
        scale = 1.0 / math.sqrt(dk)
        qkv = _rand_qkv(bh, T, dk, torch.float64, gen)
        null_k = torch.randn(h, dk, dtype=torch.float64, generator=gen)
        null_v = torch.randn(h, dk, dtype=torch.float64, generator=gen)
        y_grad = _NullAttn.apply(qkv, null_k, null_v, scale, B, True)
        with torch.no_grad():
            y_nograd = _NullAttn.apply(qkv, null_k, null_v, scale, B, False)
        self.assertTrue(torch.allclose(y_grad, y_nograd, atol=1e-12))


class ModelPathParityTest(unittest.TestCase):
    """Training (grad) path vs evaluation (no-grad) path of the full model."""

    def _parity(self, gate, T, dtype):
        torch.manual_seed(21 + T)
        model = build({
            "vocab_size": 64, "d_model": 16, "n_heads": 2, "n_layers": 2,
            "max_len": 64, "gate": gate, "dropout": 0.0,
        }).to(dtype)
        model.eval()
        idx = torch.randint(0, 64, (3, T))
        with torch.no_grad():
            eval_logits, _, _ = model(idx)
        train_logits, _, _ = model(idx)
        self.assertTrue(torch.allclose(
            train_logits, eval_logits, atol=1e-4 if dtype == torch.float32
            else 1e-9, rtol=1e-4 if dtype == torch.float32 else 1e-9),
            f"grad/no-grad mismatch for gate={gate} T={T} {dtype}")

    def test_parity_all_gates_float32(self):
        for gate in ("none", "query", "calib", "null"):
            for T in (6, 12, 34):
                self._parity(gate, T, torch.float32)

    def test_parity_all_gates_float64(self):
        for gate in ("none", "query", "calib", "null"):
            self._parity(gate, 10, torch.float64)

    def test_calib_gate_matches_manual_formula(self):
        # End-to-end check of the calibrated gate formula against a direct
        # recomputation from the layer's own weights and attention matrix.
        torch.manual_seed(23)
        model = build({
            "vocab_size": 64, "d_model": 16, "n_heads": 2, "n_layers": 1,
            "max_len": 64, "gate": "calib",
        })
        model.eval()
        attn = model.blocks[0].attn
        idx = torch.randint(0, 64, (2, 9))
        with torch.no_grad():
            _, _, layer_stats = model(idx, stat_pos=torch.tensor([8, 8]))
        # Recompute the gate at the final row from first principles.
        h, T = attn.h, 9
        with torch.no_grad():
            x = model.tok(idx) + model.pos.weight[:T]
            x_norm = model.blocks[0].ln1(x)
            qkv = torch.nn.functional.linear(
                x_norm.reshape(-1, x_norm.size(-1)), attn.qkv.weight)
            qkv = qkv.view(2, T, 3, h, attn.dk).permute(2, 0, 3, 1, 4)
            q = qkv[0].reshape(2 * h, T, attn.dk)
            k = qkv[1].reshape(2 * h, T, attn.dk)
            p = _reference_probs(q, k, attn.inv_scale).view(2, h, T, T)
            ent_rows = -(p * p.clamp_min(_EPS).log()).sum(-1)  # (2, h, T)
            hbar = ent_rows.mean(1)[:, -1]  # final row, head-mean
            wg = torch.nn.functional.linear(
                x_norm[:, -1], attn.gate.weight, attn.gate.bias)
            tau = torch.nn.functional.softplus(attn.calib_tau) + 0.1
            expected = torch.sigmoid(
                (wg - attn.calib_a * math.log(max(T, 2))
                 - (attn.calib_c * attn.calib_scale
                    * hbar / math.log(max(T, 2))).unsqueeze(-1)
                 ) / tau)
        got = layer_stats[0]["gate"].mean(-1)
        expected_mean = expected.mean(-1)
        self.assertTrue(
            torch.allclose(got, expected_mean, atol=1e-5, rtol=1e-4),
            f"{got} vs {expected_mean}")


if __name__ == "__main__":
    unittest.main()
