"""Data-contract tests: split leakage guards and batch validation."""

from __future__ import annotations

import unittest

import torch

from cage.data import (
    RESERVED_EVAL_SEEDS,
    assert_split_partition,
    batch_fingerprint,
    check_train_seed,
    make_batch,
    stream_fingerprint,
    stream_seed_key,
    validate_batch,
)


class SplitContractTest(unittest.TestCase):
    def test_reserved_training_seed_is_refused(self):
        for seed in RESERVED_EVAL_SEEDS:
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    check_train_seed(seed)
                with self.assertRaises(ValueError):
                    check_train_seed(seed + 2 ** 32)

    def test_aliased_stream_fingerprint_is_identical(self):
        a = stream_fingerprint(4, 32, "cpu", batch=4, seed=0)
        b = stream_fingerprint(4, 32, "cpu", batch=4, seed=2 ** 32)
        self.assertEqual(a["sha256"], b["sha256"])

    def test_geometry_digest_changes_with_batch(self):
        # make_batch's draws are shape-dependent, so one seed at two batch
        # sizes is two realizations and must not share a dataset version.
        a = stream_fingerprint(4, 32, "cpu", batch=4, seed=0)
        b = stream_fingerprint(4, 32, "cpu", batch=8, seed=0)
        self.assertNotEqual(a["sha256"], b["sha256"])

    def test_overlapping_partition_is_rejected(self):
        with self.assertRaises(ValueError):
            assert_split_partition(
                4, {"train": torch.tensor([0, 1]), "test": torch.tensor([1, 2])})


class BatchValidationTest(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(1234)
        self.idx, self.tgt, self.meta = make_batch(4, 3, 32, "cpu",
                                                   gen=self.gen)

    def test_valid_batch_passes(self):
        diagnostics = validate_batch(self.idx, self.tgt, self.meta, 32)
        self.assertEqual(diagnostics["n"], 4)
        self.assertEqual(diagnostics["n_duplicate_rows"], 0)

    def test_misaligned_target_is_rejected(self):
        bad = self.tgt.clone()
        bad[:, -1] = -1
        with self.assertRaises(ValueError):
            validate_batch(self.idx, bad, self.meta, 32)

    def test_wrong_answer_label_is_rejected(self):
        bad = dict(self.meta)
        bad["answer"] = (self.meta["answer"] + 1) % 30 + 2
        with self.assertRaises(ValueError):
            validate_batch(self.idx, self.tgt, bad, 32)

    def test_batch_fingerprint_is_stable_for_same_batch(self):
        self.assertEqual(batch_fingerprint(self.idx, self.tgt, self.meta),
                         batch_fingerprint(self.idx.clone(), self.tgt.clone(),
                                           {k: v.clone() for k, v in
                                            self.meta.items()}))
        self.assertEqual(stream_seed_key(7), 7)


if __name__ == "__main__":
    unittest.main()
