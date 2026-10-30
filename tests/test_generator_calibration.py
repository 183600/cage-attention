"""Model-free calibration tests for the declared synthetic data generator."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch

import cage.data as data


class GeneratorCalibrationTest(unittest.TestCase):
    def test_declared_generator_passes_the_model_free_family(self):
        out = data.generator_calibration_audit(
            4, 32, batch=64, batches=2, seed=0, n_sim=200)
        self.assertEqual(out["n_rows"], 128)
        self.assertEqual(out["n_pair_observations"], 512)
        self.assertEqual(out["family"]["n_tests"], 10)
        self.assertEqual(out["family"]["abstained"], [])
        self.assertTrue(out["calibration_supported"])
        self.assertTrue(all(
            math.isfinite(test["p_value"])
            for test in (*out["marginal_uniformity"].values(),
                         *out["independence"].values(),
                         out["value_repeat_rate"],
                         out["answer_as_key_rate"]) if test["applicable"]))

    def test_single_pair_geometry_abstains_on_unavailable_checks(self):
        out = data.generator_calibration_audit(
            1, 4, batch=16, batches=2, seed=0, n_sim=50)
        self.assertFalse(
            out["marginal_uniformity"]["query_rank"]["applicable"])
        self.assertFalse(out["value_repeat_rate"]["applicable"])
        self.assertEqual(out["family"]["n_tests"], 7)
        self.assertTrue(out["calibration_supported"])

    def test_biased_key_value_coupling_is_rejected(self):
        original = data.make_batch

        def coupled(batch, n_pairs, vocab, device, dense_lm=False, gen=None):
            idx, targets, meta = original(
                batch, n_pairs, vocab, device, dense_lm=dense_lm, gen=gen)
            idx = idx.clone()
            keys = idx[:, :2 * n_pairs:2]
            values = (keys - 2 + 1) % (vocab - 2) + 2
            idx[:, 1:2 * n_pairs:2] = values
            rows = torch.arange(batch, device=idx.device)
            answer = idx.gather(
                1, (meta["key_pos"] + 1).unsqueeze(1)).squeeze(1)
            targets = torch.full_like(idx, -1)
            targets[rows, meta["query_pos"]] = answer
            meta = dict(meta)
            meta["answer"] = answer
            data.validate_batch(idx, targets, meta, vocab,
                                diagnostics=False)
            return idx, targets, meta

        with patch.object(data, "make_batch", coupled):
            out = data.generator_calibration_audit(
                4, 32, batch=64, batches=2, seed=0, n_sim=200)
        self.assertFalse(out["calibration_supported"])
        self.assertIn("independence:key_value",
                      out["family"]["rejected"])
        self.assertLessEqual(
            out["independence"]["key_value"]["p_value"], 1.0 / 201.0)

    def test_answer_copying_a_key_is_rejected(self):
        original = data.make_batch

        def copy_key(batch, n_pairs, vocab, device, dense_lm=False, gen=None):
            idx, targets, meta = original(
                batch, n_pairs, vocab, device, dense_lm=dense_lm, gen=gen)
            idx = idx.clone()
            rows = torch.arange(batch, device=idx.device)
            answer = idx[:, -1]
            idx[rows, meta["key_pos"] + 1] = answer
            targets = torch.full_like(idx, -1)
            targets[rows, meta["query_pos"]] = answer
            meta = dict(meta)
            meta["answer"] = answer
            data.validate_batch(idx, targets, meta, vocab,
                                diagnostics=False)
            return idx, targets, meta

        with patch.object(data, "make_batch", copy_key):
            out = data.generator_calibration_audit(
                4, 32, batch=64, batches=2, seed=0, n_sim=200)
        self.assertFalse(out["calibration_supported"])
        self.assertIn("answer_as_key_rate", out["family"]["rejected"])
        self.assertEqual(out["answer_as_key_rate"]["observed"], 128)
        self.assertLessEqual(
            out["answer_as_key_rate"]["p_value"], 1.0 / 201.0)

    def test_without_replacement_values_are_rejected(self):
        original = data.make_batch

        def no_repeat(batch, n_pairs, vocab, device, dense_lm=False, gen=None):
            idx, targets, meta = original(
                batch, n_pairs, vocab, device, dense_lm=dense_lm, gen=gen)
            idx = idx.clone()
            scores = torch.rand(
                (batch, vocab - 2), generator=gen, device=device)
            values = scores.topk(n_pairs, dim=1).indices.add(2)
            idx[:, 1:2 * n_pairs:2] = values
            query_rank = meta["key_pos"].div(2, rounding_mode="floor")
            answer = values.gather(
                1, query_rank.unsqueeze(1)).squeeze(1)
            targets = torch.full_like(idx, -1)
            rows = torch.arange(batch, device=idx.device)
            targets[rows, meta["query_pos"]] = answer
            meta = dict(meta)
            meta["answer"] = answer
            data.validate_batch(idx, targets, meta, vocab,
                                diagnostics=False)
            return idx, targets, meta

        with patch.object(data, "make_batch", no_repeat):
            out = data.generator_calibration_audit(
                4, 32, batch=64, batches=2, seed=0, n_sim=200)
        self.assertFalse(out["calibration_supported"])
        self.assertIn("value_repeat_rate", out["family"]["rejected"])
        self.assertEqual(out["value_repeat_rate"]["observed"], 0)
        self.assertLessEqual(
            out["value_repeat_rate"]["p_value"], 1.0 / 201.0)

    def test_independence_helper_detects_coupling_and_abstains_safely(self):
        left = torch.arange(3000) % 10
        coupled = data._categorical_independence_test(
            left, (left + 1) % 10, 10, 10, n_sim=200, seed=0)
        independent = data._categorical_independence_test(
            left, torch.arange(3000) % 7, 10, 10, n_sim=200, seed=0)
        self.assertTrue(coupled["valid"] and independent["valid"])
        self.assertLessEqual(coupled["p_value"], 1.0 / 201.0)
        self.assertGreater(independent["p_value"], 0.05)
        too_wide = data._categorical_independence_test(
            left, left, 10_000, 10_000, n_sim=10, seed=0)
        self.assertFalse(too_wide["valid"])
        self.assertIn("audit budget", too_wide["reason"])

    def test_expected_answer_as_key_rate_matches_the_dgp(self):
        self.assertAlmostEqual(
            data._expected_answer_as_key_rate(1, 4), 0.5)
        self.assertAlmostEqual(
            data._expected_answer_as_key_rate(3, 8), 0.5)
        self.assertEqual(data._expected_answer_as_key_rate(5, 4), 1.0)

    def test_expected_value_repeat_rate_matches_the_birthday_probability(self):
        self.assertEqual(data._expected_value_repeat_rate(1, 4), 0.0)
        self.assertAlmostEqual(data._expected_value_repeat_rate(2, 4), 0.5)
        # More draws than selectable values guarantees a collision.
        self.assertEqual(data._expected_value_repeat_rate(5, 4), 1.0)

    def test_repeat_rate_stays_accurate_for_large_vocabularies(self):
        for vocab in (10**6, 10**9, 10**12, 10**15):
            selectable = vocab - 2
            self.assertAlmostEqual(
                data._expected_value_repeat_rate(2, vocab) * selectable,
                1.0, places=12)
            expected_three = 3.0 / selectable - 2.0 / selectable ** 2
            self.assertAlmostEqual(
                data._expected_value_repeat_rate(3, vocab)
                / expected_three,
                1.0, places=12)

    def test_split_audit_stamps_generator_calibration(self):
        audit = data.split_leakage_audit(
            3, 17, batch=8, batches=2, train_seeds=(0,))
        calibration = audit["generator_calibration"]
        self.assertEqual(calibration["stream"], "test")
        self.assertEqual(calibration["seed"], data.TEST_SEED)
        self.assertIn("family", calibration)
        self.assertIn("calibration_supported", calibration)
        self.assertIs(audit["generator_calibration_supported"],
                      calibration["calibration_supported"])


if __name__ == "__main__":
    unittest.main()
