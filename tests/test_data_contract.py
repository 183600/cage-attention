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
    stream_content_digest,
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


class StreamContentDigestTest(unittest.TestCase):
    def _batches(self):
        gen = torch.Generator().manual_seed(99)
        return [make_batch(4, 3, 32, "cpu", gen=gen) for _ in range(3)]

    def test_same_stream_has_same_digest(self):
        a = stream_content_digest(self._batches())
        b = stream_content_digest(self._batches())
        self.assertEqual(a, b)
        self.assertEqual(a["n_batches"], 3)
        self.assertEqual(a["n_rows"], 12)

    def test_digest_is_order_sensitive(self):
        batches = self._batches()
        forward = stream_content_digest(batches)
        reverse = stream_content_digest(list(reversed(batches)))
        self.assertNotEqual(forward["sha256"], reverse["sha256"])

    def test_digest_covers_later_batches_and_metadata(self):
        batches = self._batches()
        base = stream_content_digest(batches)["sha256"]
        later = [(idx.clone(), tgt.clone(), dict(meta))
                 for idx, tgt, meta in batches]
        later[-1][1][0, 0] = (later[-1][1][0, 0] + 1) % 30
        self.assertNotEqual(base, stream_content_digest(later)["sha256"])
        meta = [(idx.clone(), tgt.clone(), dict(meta_))
                for idx, tgt, meta_ in batches]
        meta[0][2]["query_pos"] = meta[0][2]["query_pos"].clone()
        meta[0][2]["query_pos"][0] += 1
        self.assertNotEqual(base, stream_content_digest(meta)["sha256"])
        # The last batch has fewer rows: the digest must include the
        # shape, not only the flattened content.
        shape_changed = [(idx.clone(), tgt.clone(), dict(meta_))
                         for idx, tgt, meta_ in batches]
        idx, tgt, meta_ = shape_changed[-1]
        shape_changed[-1] = (idx[:2], tgt[:2], {
            key: value[:2] for key, value in meta_.items()})
        self.assertNotEqual(base, stream_content_digest(shape_changed)["sha256"])

    def test_empty_stream_has_a_defined_digest(self):
        empty = stream_content_digest([])
        self.assertEqual(empty["n_batches"], 0)
        self.assertEqual(empty["n_rows"], 0)
        self.assertEqual(len(empty["sha256"]), 16)

    def test_malformed_batches_raise(self):
        idx, tgt, meta = self._batches()[0]
        bad_cases = [
            ("not-a-triple", object()),
            ("bad_idx", (idx.reshape(-1), tgt, meta)),
            ("bad_targets", (idx, tgt.reshape(-1), meta)),
            ("missing_answer", (idx, tgt, {"query_pos": meta["query_pos"]})),
            ("bad_answer", (idx, tgt, {"answer": meta["answer"][:1]})),
            ("bad_query_pos", (idx, tgt, {
                "answer": meta["answer"],
                "query_pos": meta["query_pos"][:1]})),
        ]
        for label, batch in bad_cases:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    stream_content_digest([batch])


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
