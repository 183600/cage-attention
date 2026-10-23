"""Regression tests for the explicit attention paths.

The design-pinned kernels are shared by all gate variants, but the
attention-dropout branch is a separate explicit-math path.  These checks
exercise its statistics readout (a shape bug there is invisible in the
kernel path) and the empty-batch forward (a legal data batch whose zero
row count must still produce a legal chunk step).
"""

from __future__ import annotations

import unittest

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
