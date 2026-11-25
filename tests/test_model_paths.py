"""Regression tests for the explicit attention paths.

The design-pinned kernels are shared by all gate variants, but the
attention-dropout branch is a separate explicit-math path.  These checks
exercise its statistics readout (a shape bug there is invisible in the
kernel path) and the empty-batch forward (a legal data batch whose zero
row count must still produce a legal chunk step).
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from cage.data import make_batch
from cage.model import GATES, build


class DropoutStatsPathTest(unittest.TestCase):
    def test_every_gate_returns_finite_stats_with_dropout(self):
        idx, tgt, meta = make_batch(
            3, 4, 17, "cpu", gen=torch.Generator().manual_seed(1))
        for gate in GATES:
            with self.subTest(gate=gate):
                model = build({
                    "vocab_size": 17, "d_model": 16, "n_heads": 4,
                    "n_layers": 2, "max_len": 32, "gate": gate,
                    "dropout": 0.1,
                })
                model.train()
                _, loss, stats = model(
                    idx, tgt, sup_pos=meta["query_pos"],
                    stat_pos=meta["query_pos"])
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for layer in stats:
                    for key, value in layer.items():
                        if isinstance(value, torch.Tensor) and value.numel():
                            self.assertTrue(
                                torch.isfinite(value).all(),
                                f"{gate}: non-finite {key}")


class EmptyBatchTest(unittest.TestCase):
    def test_empty_batch_forward_and_stats(self):
        idx, _, meta = make_batch(
            0, 3, 17, "cpu", gen=torch.Generator().manual_seed(2))
        for gate in GATES:
            with self.subTest(gate=gate):
                model = build({
                    "vocab_size": 17, "d_model": 16, "n_heads": 4,
                    "n_layers": 1, "max_len": 32, "gate": gate,
                })
                logits, loss, stats = model(idx)
                self.assertIsNone(loss)
                self.assertEqual(tuple(logits.shape), (0, 1, 17))
                _, _, layer_stats = model(idx, stat_pos=meta["query_pos"])
                self.assertEqual(len(layer_stats), 1)

    def test_empty_batch_targets_are_not_a_silent_nan_loss(self):
        idx, tgt, _meta = make_batch(
            0, 3, 17, "cpu", gen=torch.Generator().manual_seed(2))
        model = build({
            "vocab_size": 17, "d_model": 16, "n_heads": 4,
            "n_layers": 1, "max_len": 32, "gate": "none",
        })
        with self.assertRaises(ValueError):
            model(idx, tgt)


if __name__ == "__main__":
    unittest.main()


class EmptyBatchLongSequenceTest(unittest.TestCase):
    def test_empty_batch_with_blocked_sequence(self):
        # A zero-row batch at T > 32 still enters _query_blocks; the block
        # size formula divides by B*H, so it must not divide by zero.
        idx, _, meta = make_batch(
            0, 16, 64, "cpu", gen=torch.Generator().manual_seed(3))
        self.assertGreater(idx.size(1), 32)
        for gate in GATES:
            with self.subTest(gate=gate):
                model = build({
                    "vocab_size": 64, "d_model": 32, "n_heads": 4,
                    "n_layers": 1, "max_len": 64, "gate": gate,
                })
                logits, loss, _ = model(idx)
                self.assertIsNone(loss)
                self.assertEqual(tuple(logits.shape), (0, 1, 64))
                _, _, layer_stats = model(idx, stat_pos=meta["query_pos"])
                self.assertEqual(len(layer_stats), 1)


class UnbalancedQueryBlockTest(unittest.TestCase):
    """The chunked kernels must size scratch from the actual block extents.

    Production ``_query_blocks`` is balanced, but the kernels accept any
    contiguous decomposition; sizing from ``blocks[0]`` alone underflows the
    score/staging buffers for an unbalanced (e.g. tail-heavy) decomposition.
    """

    T = 6
    DK = 4

    def _qkv(self, grad):
        gen = torch.Generator()
        q, k, v = (torch.randn(1, self.T, self.DK,
                               generator=gen.manual_seed(s))
                   for s in (1, 2, 3))
        qkv = torch.stack([q, k, v])
        return qkv.requires_grad_(grad)

    def _run_calib(self, blocks, need_ent, ybt=False):
        import cage.model as model_mod
        qkv = self._qkv(grad=not ybt)
        args = (qkv, 0.37, 1e-12, need_ent)
        if ybt:
            args += (torch.empty(1, self.T, 1, self.DK),)
        if blocks is None:
            y, ent = model_mod._CalibAttn.apply(*args)
        else:
            with mock.patch.object(model_mod, "_query_blocks",
                                   new=lambda bh, T: blocks):
                y, ent = model_mod._CalibAttn.apply(*args)
        if not ybt:
            (y.square().sum()
             + (ent.square().sum() if ent is not None else 0)).backward()
            return (y.detach(), None if ent is None else ent.detach(),
                    qkv.grad.detach().clone())
        return y.detach(), None if ent is None else ent.detach(), None

    def _run_null(self, blocks, ybt=False):
        import cage.model as model_mod
        qkv = self._qkv(grad=not ybt)
        nk = torch.randn(1, self.DK, generator=torch.Generator().manual_seed(4))
        nv = torch.randn(1, self.DK, generator=torch.Generator().manual_seed(5))
        nk.requires_grad_(not ybt)
        nv.requires_grad_(not ybt)
        args = (qkv, nk, nv, 0.37, 1)
        if ybt:
            args += (False, torch.empty(1, self.T, 1, self.DK))
        if blocks is None:
            y = model_mod._NullAttn.apply(*args)
        else:
            with mock.patch.object(model_mod, "_query_blocks",
                                   new=lambda bh, T: blocks):
                y = model_mod._NullAttn.apply(*args)
        if not ybt:
            y.square().sum().backward()
            return (y.detach(), qkv.grad.detach().clone(),
                    nk.grad.detach().clone(), nv.grad.detach().clone())
        return y.detach(), None, None, None

    def test_calib_unbalanced_blocks_match_reference(self):
        for blocks in (((0, 1), (1, self.T)),
                       ((0, 2), (2, 5), (5, self.T))):
            for need_ent in (False, True):
                with self.subTest(blocks=blocks, need_ent=need_ent):
                    want = self._run_calib(None, need_ent)
                    got = self._run_calib(blocks, need_ent)
                    self.assertTrue(torch.allclose(want[0], got[0]))
                    if need_ent:
                        self.assertTrue(torch.allclose(want[1], got[1]))
                    self.assertTrue(torch.allclose(want[2], got[2]))

    def test_calib_unbalanced_ybt_matches_reference(self):
        for blocks in (((0, 1), (1, self.T)),
                       ((0, 2), (2, 5), (5, self.T))):
            with self.subTest(blocks=blocks):
                want_y, want_ent, _ = self._run_calib(None, True)
                got_y, got_ent, _ = self._run_calib(blocks, True, ybt=True)
                self.assertTrue(torch.allclose(
                    want_y.view(1, 1, self.T, self.DK).transpose(1, 2),
                    got_y))
                self.assertTrue(torch.allclose(want_ent, got_ent))

    def test_null_unbalanced_blocks_match_reference(self):
        for blocks in (((0, 1), (1, self.T)),
                       ((0, 2), (2, 5), (5, self.T))):
            with self.subTest(blocks=blocks):
                want = self._run_null(None)
                got = self._run_null(blocks)
                for expected, actual in zip(want, got):
                    self.assertTrue(torch.allclose(expected, actual))

    def test_null_unbalanced_ybt_matches_reference(self):
        for blocks in (((0, 1), (1, self.T)),
                       ((0, 2), (2, 5), (5, self.T))):
            with self.subTest(blocks=blocks):
                want_y, _, _, _ = self._run_null(None)
                got_y, _, _, _ = self._run_null(blocks, ybt=True)
                self.assertTrue(torch.allclose(
                    want_y.view(1, 1, self.T, self.DK).transpose(1, 2),
                    got_y))
