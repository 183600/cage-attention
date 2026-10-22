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


class UniformityInputTest(unittest.TestCase):
    def test_malformed_container_abstains_instead_of_raising(self):
        from cage.data import answer_uniformity_test

        for bad in (None, "x", object()):
            with self.subTest(bad=type(bad).__name__):
                out = answer_uniformity_test(bad)
                self.assertFalse(out["uniform_supported"])
                self.assertTrue(out["p_value"] != out["p_value"])

    def test_boolean_counts_are_rejected(self):
        from cage.data import answer_uniformity_test

        with self.assertRaises(ValueError):
            answer_uniformity_test(torch.tensor([True, False]))


class AuditCountInputTest(unittest.TestCase):
    def test_boolean_counts_are_not_silently_cast_to_categories(self):
        from cage.data import _counts, _distribution_shift_p

        for values in (torch.tensor([True, False]),
                       [True, False, True]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    _counts(values, 2)
                with self.assertRaises(ValueError):
                    _distribution_shift_p(values, torch.tensor([1, 1]))

    def test_malformed_shift_count_containers_raise_value_error(self):
        from cage.data import _distribution_shift_p

        for bad in (None, "x", object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(ValueError):
                    _distribution_shift_p(bad, torch.tensor([1, 1]))
                with self.assertRaises(ValueError):
                    _distribution_shift_p(torch.tensor([1, 1]), bad)


class SplitAuditInputTest(unittest.TestCase):
    def test_non_iterable_train_seeds_raise_value_error(self):
        from cage.data import split_leakage_audit

        for bad in (None, 5, "x"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    split_leakage_audit(3, 17, batch=4, batches=1,
                                        train_seeds=bad)

    def test_malformed_stream_spec_containers_raise_value_error(self):
        from cage.data import actual_stream_overlap_audit

        good = {"seed": 0, "batch": 4, "steps": 1}
        heldout = {"h": {"seed": 1, "batch": 4, "batches": 1}}
        for bad in (5, "x", b"\x00", object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(ValueError):
                    actual_stream_overlap_audit(
                        3, 17, train_specs=bad, heldout_specs=heldout)
                with self.assertRaises(ValueError):
                    actual_stream_overlap_audit(
                        3, 17, train_specs={"t": good},
                        heldout_specs=bad)


class TimeOrderInputTest(unittest.TestCase):
    def test_malformed_timestamp_container_raises_value_error(self):
        from cage.data import assert_time_ordered

        for bad in (None, "x", object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(ValueError):
                    assert_time_ordered(bad, {"train": [0]})


class SplitContainerInputTest(unittest.TestCase):
    def test_non_mapping_split_containers_raise_value_error(self):
        from cage.data import (assert_group_disjoint,
                               assert_split_partition,
                               assert_time_ordered)

        for bad in (5, "x", b"\x00", [("train", [0])]):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(ValueError):
                    assert_split_partition(2, bad)
                with self.assertRaises(ValueError):
                    assert_group_disjoint([0, 1], bad)
                with self.assertRaises(ValueError):
                    assert_time_ordered([0, 1], bad)


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

    def test_non_iterable_batches_raise_value_error(self):
        for bad in (None, 42, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(ValueError):
                    stream_content_digest(bad)

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

    def test_fractional_or_bool_vocab_is_rejected(self):
        for vocab in (8.5, 8.0, True):
            with self.subTest(vocab=vocab):
                with self.assertRaises(ValueError):
                    validate_batch(self.idx, self.tgt, self.meta, vocab)

    def test_odd_width_sequence_is_rejected(self):
        idx = torch.tensor([[5, 10, 7, 1, 5]])
        tgt = torch.tensor([[-1, -1, -1, -1, 10]])
        meta = {"query_pos": torch.tensor([4]),
                "key_pos": torch.tensor([0]),
                "answer": torch.tensor([10])}
        with self.assertRaises(ValueError):
            validate_batch(idx, tgt, meta, 32)

    def test_batch_fingerprint_is_stable_for_same_batch(self):
        self.assertEqual(batch_fingerprint(self.idx, self.tgt, self.meta),
                         batch_fingerprint(self.idx.clone(), self.tgt.clone(),
                                           {k: v.clone() for k, v in
                                            self.meta.items()}))
        self.assertEqual(stream_seed_key(7), 7)


if __name__ == "__main__":
    unittest.main()
