"""Data-level robustness of the shared finite evaluation stream.

The seed bootstrap treats training seeds as independent units, but every
seed is scored on one finite set of evaluation items.  These tests pin the
fold jackknife over those items and the content-digest guard that prevents
two different evaluation realizations from being paired by position.
"""

from __future__ import annotations

import math
import unittest

import torch

from cage.data import TEST_SEED
from cage.experiments import Settings, _model_cfg, _paired_stream_test
from cage.metrics import evaluation_fold_audit
from cage.model import build
from cage.repro import seed_everything
from cage.train import encode_correct_mask, eval_metrics


class EvaluationFoldAuditTest(unittest.TestCase):
    def test_uniform_effect_is_not_fragile(self):
        a = [1] * 16
        b = [0] * 16
        out = evaluation_fold_audit(a, b, n_folds=4)
        self.assertTrue(out["valid"])
        self.assertEqual(out["n_folds"], 4)
        self.assertEqual(out["n_items"], 16)
        self.assertAlmostEqual(out["overall_effect"], 1.0)
        self.assertEqual(out["fold_effects"], [1.0] * 4)
        self.assertEqual(out["fold_sign_consistency"], 1.0)
        self.assertTrue(out["fold_all_same_sign"])
        self.assertFalse(out["loo_sign_flip"])
        self.assertFalse(out["fragile"])
        self.assertEqual(out["n_unit_fold_sign_flips"], 0)

    def test_one_fold_driving_the_effect_is_flagged(self):
        # Overall +1/16, but the first fold carries +4 net items and the
        # second -3.  Dropping the first fold flips the pooled effect to
        # -3/12, so the item-level "win" is a property of one stream slice.
        a = [1, 1, 1, 1] + [0] * 12
        b = [0, 0, 0, 0] + [1, 1, 1] + [0] * 9
        out = evaluation_fold_audit(a, b, n_folds=4)
        self.assertTrue(out["valid"])
        self.assertAlmostEqual(out["overall_effect"], 1.0 / 16.0)
        self.assertAlmostEqual(out["fold_effects"][0], 1.0)
        self.assertAlmostEqual(out["fold_effects"][1], -0.75)
        self.assertTrue(out["loo_sign_flip"])
        self.assertGreater(out["dominant_fold_share"], 0.5)
        self.assertTrue(out["fragile"])

    def test_per_seed_fold_sign_flip_is_counted(self):
        # Unit 0 is uniformly positive; unit 1 has the same concentrated
        # pattern as above, so its own fold-1 effect reverses direction.
        a = torch.tensor([
            [1] * 16,
            [1, 1, 1, 1] + [0] * 12,
        ], dtype=torch.bool)
        b = torch.tensor([
            [0] * 16,
            [0, 0, 0, 0, 1, 1, 1, 0] + [0] * 8,
        ], dtype=torch.bool)
        out = evaluation_fold_audit(a, b, n_folds=4)
        self.assertTrue(out["valid"])
        self.assertEqual(out["n_units"], 2)
        self.assertAlmostEqual(out["overall_effect"], 17.0 / 32.0)
        self.assertEqual(out["unit_fold_sign_flips"], [1])
        self.assertEqual(out["n_unit_fold_sign_flips"], 1)
        self.assertAlmostEqual(out["unit_fold_sign_consistency"][1], 0.5)

    def test_fold_count_is_capped_to_keep_folds_nonempty(self):
        out = evaluation_fold_audit([1] * 10, [0] * 10, n_folds=8)
        self.assertTrue(out["valid"])
        self.assertEqual(out["requested_folds"], 8)
        self.assertEqual(out["n_folds"], 5)
        self.assertEqual(out["fold_sizes"], [2] * 5)

    def test_one_dimensional_input_is_one_unit(self):
        out = evaluation_fold_audit([1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 1, 1, 0], n_folds=4)
        self.assertTrue(out["valid"])
        self.assertEqual(out["n_units"], 1)
        self.assertEqual(len(out["unit_effects"]), 1)

    def test_degenerate_or_malformed_input_abstains(self):
        cases = [
            ([], []),
            ([1, 0, 1], [1, 0]),
            ([1, 0, 1, 0], [1, 0, 1]),
            ([1, 0, 1, 1], [1, 2, 0, 1]),
            ([1, 0, float("nan"), 1], [1, 0, 1, 0]),
            ([1, 0, 1], [1, 0, 0]),
            ([1, 0, 1, 0], [1, 0, 1, 0, 1]),
            ([[1, 0, 1, 0]], [[1, 0, 1, 0], [1, 0, 1, 0]]),
        ]
        for a, b in cases:
            with self.subTest(a=a, b=b):
                self.assertFalse(evaluation_fold_audit(a, b)["valid"])
        for bad_folds in (1, 0, -2, True, 2.5, "4", None):
            with self.subTest(n_folds=bad_folds):
                self.assertFalse(
                    evaluation_fold_audit([1] * 8, [0] * 8,
                                          n_folds=bad_folds)["valid"])


class EvalMetricsDigestTest(unittest.TestCase):
    def test_eval_metrics_stamps_the_whole_stream(self):
        settings = Settings.quick()
        seed_everything(0)
        model = build(_model_cfg(settings), compile=False)
        kwargs = dict(batch=settings.eval_batch,
                      batches=settings.eval_batches)
        first = eval_metrics(model, settings.n_pairs, settings.vocab, "cpu",
                             seed=TEST_SEED, **kwargs)
        repeat = eval_metrics(model, settings.n_pairs, settings.vocab, "cpu",
                              seed=TEST_SEED, **kwargs)
        other = eval_metrics(model, settings.n_pairs, settings.vocab, "cpu",
                             seed=TEST_SEED + 1, **kwargs)
        self.assertEqual(first["stream_digest"], repeat["stream_digest"])
        self.assertNotEqual(first["stream_digest"], other["stream_digest"])
        self.assertEqual(first["stream_n_batches"], settings.eval_batches)
        self.assertEqual(first["stream_n_rows"],
                         settings.eval_batch * settings.eval_batches)


class PairedStreamDigestTest(unittest.TestCase):
    def _records(self, mask_a, mask_b, digest="abc123"):
        by_a, by_b = {}, {}
        n = len(mask_a[0])
        for seed, (a, b) in enumerate(zip(mask_a, mask_b)):
            by_a[seed] = {
                "seed": seed, "n": n,
                "correct_bitmask": encode_correct_mask(a),
                "stream_digest": digest,
            }
            by_b[seed] = {
                "seed": seed, "n": n,
                "correct_bitmask": encode_correct_mask(b),
                "stream_digest": digest,
            }
        return by_a, by_b

    def test_matching_digest_enables_fold_companion(self):
        a = [[1] * 16, [1] * 16]
        b = [[0] * 16, [0] * 16]
        by_a, by_b = self._records(a, b)
        out = _paired_stream_test(by_a, by_b, [0, 1], "accuracy", 1.0)
        self.assertTrue(out["applicable"])
        self.assertTrue(out["digest_verified"])
        self.assertEqual(out["stream_digest"], "abc123")
        self.assertTrue(out["fold_audit"]["valid"])
        self.assertEqual(out["fold_audit"]["n_units"], 2)

    def test_digest_mismatch_refuses_item_pairing(self):
        a = [[1] * 16]
        b = [[0] * 16]
        by_a, by_b = self._records(a, b, digest="aaa")
        by_b[0]["stream_digest"] = "bbb"
        out = _paired_stream_test(by_a, by_b, [0], "accuracy", 1.0)
        self.assertFalse(out["applicable"])
        self.assertIn("digest", out["reason"])

    def test_within_arm_digest_mismatch_is_rejected(self):
        a = [[1] * 16, [1] * 16]
        b = [[0] * 16, [0] * 16]
        by_a, by_b = self._records(a, b, digest="aaa")
        by_a[1]["stream_digest"] = "bbb"
        out = _paired_stream_test(by_a, by_b, [0, 1], "accuracy", 1.0)
        self.assertFalse(out["applicable"])
        self.assertIn("digest", out["reason"])

    def test_mixed_or_one_sided_digest_is_not_verified(self):
        a = [[1] * 16]
        b = [[0] * 16]
        for side in ("a", "b"):
            with self.subTest(side=side):
                by_a, by_b = self._records(a, b)
                {"a": by_a, "b": by_b}[side][0].pop("stream_digest")
                out = _paired_stream_test(by_a, by_b, [0], "accuracy", 1.0)
                self.assertTrue(out["applicable"])
                self.assertFalse(out["digest_verified"])
                self.assertEqual(out["stream_digest"], "abc123")

    def test_partial_digest_across_seeds_is_not_verified(self):
        a = [[1] * 16, [1] * 16]
        b = [[0] * 16, [0] * 16]
        by_a, by_b = self._records(a, b)
        by_b[1].pop("stream_digest")
        out = _paired_stream_test(by_a, by_b, [0, 1], "accuracy", 1.0)
        self.assertTrue(out["applicable"])
        self.assertFalse(out["digest_verified"])

    def test_legacy_records_without_digest_stay_available(self):
        a = [[1] * 16]
        b = [[0] * 16]
        by_a, by_b = self._records(a, b)
        by_a[0].pop("stream_digest")
        by_b[0].pop("stream_digest")
        out = _paired_stream_test(by_a, by_b, [0], "accuracy", 1.0)
        self.assertTrue(out["applicable"])
        self.assertFalse(out["digest_verified"])
        self.assertIsNone(out["stream_digest"])
        self.assertTrue(out["fold_audit"]["valid"])

    def test_non_string_digest_is_rejected(self):
        a = [[1] * 16]
        b = [[0] * 16]
        by_a, by_b = self._records(a, b)
        by_a[0]["stream_digest"] = 123
        out = _paired_stream_test(by_a, by_b, [0], "accuracy", 1.0)
        self.assertFalse(out["applicable"])
        self.assertIn("digest", out["reason"])

    def test_blank_digest_is_rejected_not_verified(self):
        a = [[1] * 16]
        b = [[0] * 16]
        for blank in ("", "   "):
            with self.subTest(blank=repr(blank)):
                by_a, by_b = self._records(a, b)
                by_a[0]["stream_digest"] = blank
                by_b[0]["stream_digest"] = blank
                out = _paired_stream_test(by_a, by_b, [0], "accuracy", 1.0)
                self.assertFalse(out["applicable"])
                self.assertIn("digest", out["reason"])

    def test_non_accuracy_metric_still_abstains(self):
        a = [[1] * 16]
        b = [[0] * 16]
        by_a, by_b = self._records(a, b)
        out = _paired_stream_test(by_a, by_b, [0], "loss", 1.0)
        self.assertFalse(out["applicable"])
        self.assertIn("accuracy", out["reason"])


if __name__ == "__main__":
    unittest.main()
