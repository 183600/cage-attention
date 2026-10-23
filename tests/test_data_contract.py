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

    def test_float_counts_are_not_truncated_through_float32(self):
        from cage.data import answer_uniformity_test

        # 2**24 + 1 is not representable in float32; the original Python
        # float container must be widened straight to float64 so the audit
        # sees the exact sample size instead of rounding it down.
        large = float(2 ** 24 + 1)
        out = answer_uniformity_test([large, large], n_sim=1)
        self.assertEqual(out["n"], 2 * (2 ** 24 + 1))

    def test_overflowing_count_sum_and_sampler_abstain(self):
        from cage.data import answer_uniformity_test

        for counts in ([1e308, 1e308], [1e39, 1e39]):
            with self.subTest(counts=counts):
                out = answer_uniformity_test(counts, n_sim=1)
                self.assertFalse(out["uniform_supported"])
                self.assertTrue(out["p_value"] != out["p_value"])


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

    def test_malformed_category_count_raises_value_error(self):
        from cage.data import _counts

        for bad in (True, 1.5, object(), None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _counts([0], bad)

    def test_python_float_categories_keep_float64_precision(self):
        from cage.data import _counts

        # float32 rounds 16777217 to 16777216, which would merge two
        # distinct audit categories and fabricate a duplicate count.
        counts = _counts([16777216.0, 16777217.0], 16777218)
        self.assertEqual(int(counts[16777216]), 1)
        self.assertEqual(int(counts[16777217]), 1)

    def test_fractional_or_boolean_audit_seeds_are_rejected(self):
        from cage.data import _distribution_shift_p, answer_uniformity_test

        for seed in (True, 0.5, 1.0, "1", None, object()):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    _distribution_shift_p(
                        torch.tensor([10, 10]), torch.tensor([8, 12]),
                        n_sim=10, seed=seed)
                with self.assertRaises(ValueError):
                    answer_uniformity_test(
                        torch.tensor([10, 10]), n_sim=10, seed=seed)

    def test_overflowing_shift_count_sums_abstain(self):
        from cage.data import _distribution_shift_p

        huge = [1e308, 1e308]
        for observed, reference in ((huge, [1, 1]), ([1, 1], huge)):
            with self.subTest(observed=observed):
                p, chi2 = _distribution_shift_p(
                    observed, reference, n_sim=2)
                self.assertTrue(p != p)
                self.assertTrue(chi2 != chi2)

    def test_malformed_shift_count_containers_raise_value_error(self):
        from cage.data import _distribution_shift_p

        for bad in (None, "x", object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(ValueError):
                    _distribution_shift_p(bad, torch.tensor([1, 1]))
                with self.assertRaises(ValueError):
                    _distribution_shift_p(torch.tensor([1, 1]), bad)


class SplitFractionInputTest(unittest.TestCase):
    def test_malformed_split_fractions_raise_value_error(self):
        from cage.data import group_disjoint_split, time_ordered_split

        for train_frac, val_frac in ((None, 0.15), (0.7, None),
                                     (0.7, object())):
            with self.subTest(train_frac=train_frac, val_frac=val_frac):
                with self.assertRaises(ValueError):
                    time_ordered_split(10, train_frac, val_frac)
                with self.assertRaises(ValueError):
                    group_disjoint_split([0, 1, 2, 3],
                                         train_frac=train_frac,
                                         val_frac=val_frac)

    def test_empty_stream_list_has_no_noise_floor(self):
        from cage.data import _noise_floor

        self.assertTrue(_noise_floor([], 3) != _noise_floor([], 3))


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

    def test_python_float_timestamps_keep_float64_precision(self):
        from cage.data import assert_time_ordered

        # float32 would collapse these distinct instants and let a newer
        # training row pass as older than validation data.
        times = [16777216.0, 16777217.0]
        with self.assertRaises(ValueError):
            assert_time_ordered(times, {"train": [1], "val": [0],
                                        "test": []})
        assert_time_ordered(times, {"train": [0], "val": [1], "test": []})


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


class LengthSweepBoundsTest(unittest.TestCase):
    def test_huge_sweep_request_stops_at_the_first_unfitted_length(self):
        from cage.data import length_sweep_lengths

        self.assertEqual(
            length_sweep_lengths(base=1, factor=2, n=10 ** 9, max_len=16),
            [1, 2, 4])
        # A huge factor/n must fail fast, not build an unbounded big integer.
        with self.assertRaises(ValueError):
            length_sweep_lengths(base=2, factor=10 ** 6, n=10 ** 9,
                                 max_len=1000)


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


class BatchSwitchAndEmptyRateTest(unittest.TestCase):
    def test_dense_lm_must_be_a_real_boolean(self):
        from cage.data import make_batch

        for value in ("False", 0, 1, None):
            with self.subTest(dense_lm=value):
                with self.assertRaises(ValueError):
                    make_batch(2, 3, 32, "cpu", dense_lm=value)

    def test_empty_batch_has_an_undefined_unique_key_rate(self):
        from cage.data import make_batch, validate_batch

        import math

        idx, tgt, meta = make_batch(
            0, 3, 32, "cpu", gen=torch.Generator().manual_seed(1))
        out = validate_batch(idx, tgt, meta, 32)
        self.assertEqual(out["n"], 0)
        self.assertTrue(math.isnan(out["unique_key_rate"]))


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
