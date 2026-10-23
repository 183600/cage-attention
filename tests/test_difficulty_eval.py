"""Difficulty strata and composition-standardized evaluation.

A length sweep changes ``n_pairs`` together with the sequence length, so the
chance that an answer value repeats or that the query token appears later
as a value grows with length.  These tests pin the pre-treatment stratum
encoding and the post-stratification estimator that removes that mix
difference from accuracy comparisons.
"""

from __future__ import annotations

import contextlib
import io
import math
import pathlib
import tempfile
import unittest

import torch

from cage.data import (
    DEFAULT_DIFFICULTY_KEY_BINS,
    difficulty_features,
    difficulty_strata,
    difficulty_stratum_labels,
    make_batch,
)
from cage.metrics import standardized_proportion
from cage.model import build
from cage.train import eval_metrics


class CrossLengthStandardizationTest(unittest.TestCase):
    def test_base_mix_reproduces_base_accuracy_and_standardizes_long(self):
        from cage.experiments import _stamp_difficulty_standardization

        model = build({
            "vocab_size": 17, "d_model": 16, "n_heads": 4,
            "n_layers": 1, "max_len": 64, "gate": "query",
        }, compile=False)
        blocks = {}
        for n_pairs in (3, 6):
            blocks[n_pairs] = eval_metrics(
                model, n_pairs, 17, "cpu", batch=8, batches=4, seed=123)
        rec = {"metrics_by_length": blocks}
        _stamp_difficulty_standardization(rec, 3, [3, 6])
        self.assertAlmostEqual(
            rec["difficulty_standardized_accuracy_at_base"],
            blocks[3]["accuracy"], places=12)
        self.assertEqual(
            rec["difficulty_standardized_coverage_by_length"][3], 1.0)
        self.assertTrue(0.0 <= rec["difficulty_standardized_accuracy_by_length"][6]
                        <= 1.0)
        self.assertGreater(
            rec["difficulty_standardized_coverage_by_length"][6], 0.0)
        self.assertLessEqual(
            rec["difficulty_standardized_coverage_by_length"][6], 1.0)
        self.assertTrue(
            rec["difficulty_mix_tv_by_length"][6] >= 0.0)


class JsonLengthKeyTest(unittest.TestCase):
    @staticmethod
    def _block(n, correct):
        return {"difficulty": {
            "n_strata": 12,
            "strata": [{"id": 0, "n": n, "correct": correct}],
        }}

    def test_standardization_reads_string_length_keys(self):
        from cage.experiments import _stamp_difficulty_standardization

        blocks = {3: self._block(10, 8), 6: self._block(10, 5)}
        by_int = {"metrics_by_length": dict(blocks)}
        by_str = {"metrics_by_length": {str(k): v for k, v in blocks.items()}}
        _stamp_difficulty_standardization(by_int, 3, [3, 6])
        _stamp_difficulty_standardization(by_str, 3, [3, 6])
        for key in ("difficulty_standardized_accuracy_by_length",
                    "difficulty_standardized_coverage_by_length",
                    "difficulty_mix_tv_by_length"):
            self.assertEqual(by_int[key], by_str[key])
        self.assertAlmostEqual(
            by_str["difficulty_standardized_accuracy_drop"], 0.3)

    def test_e2_companions_read_string_length_keys(self):
        from cage.experiments import _E2_VARIANTS, _e2_difficulty_companions

        def run(seed, gate, std_acc):
            return {
                "seed": seed, "gate": gate, "model_cfg": {"gate": gate},
                "accuracy_at_longest": 0.5,
                "accuracy_drop": 0.1,
                "difficulty_standardized_accuracy_at_longest": std_acc,
                "difficulty_standardized_accuracy_drop": 0.1,
                "difficulty_standardized_coverage_at_longest": 1.0,
                "difficulty_standardized_effective_n_at_longest": 10.0,
                "difficulty_mix_tv_at_longest": 0.0,
                "metrics_by_length": {
                    "3": {"accuracy": 0.6},
                    "6": {"accuracy": 0.5},
                },
                "eval_n_by_length": {"3": 10, "6": 10},
            }

        by_variant = {
            label: [run(0, gate, 0.4 if gate == "calib" else 0.3)]
            for label, gate, *_ in _E2_VARIANTS
        }
        with contextlib.redirect_stdout(io.StringIO()):
            payload = _e2_difficulty_companions(by_variant, [3, 6])
        self.assertAlmostEqual(
            payload["by_arm"]["calib"][
                "raw_accuracy_at_longest"]["mean"], 0.5)
        self.assertAlmostEqual(
            payload["contrasts"]["calib - none"][
                "difficulty_standardized_accuracy_at_longest"]["mean"], 0.1)


class E2DifficultyCompanionTest(unittest.TestCase):
    @staticmethod
    def _run(gate):
        return {
            "seed": 0, "params": 100, "lr": 1e-3,
            "eval_seed": 2024, "stats_seed": 7, "config": {},
            "training_recipe": {}, "steps_run": 10,
            "stopped_early": False, "restored_best": False,
            "data_fingerprints": None, "gate": gate,
            "model_cfg": {"gate": gate},
            "accuracy_at_longest": 0.40 if gate == "none" else 0.50,
            "accuracy_drop": 0.20 if gate == "none" else 0.10,
            "difficulty_standardized_accuracy_at_longest":
                0.45 if gate == "none" else 0.55,
            "difficulty_standardized_accuracy_drop":
                0.15 if gate == "none" else 0.08,
            "difficulty_standardized_coverage_at_longest": 1.0,
            "difficulty_standardized_effective_n_at_longest": 100.0,
            "difficulty_mix_tv_at_longest": 0.10,
        }

    def test_companion_reports_consistent_and_reversed_signs(self):
        from cage.experiments import _E2_VARIANTS, _e2_difficulty_companions

        by_variant = {label: [self._run(gate)]
                      for label, gate, *_ in _E2_VARIANTS}
        with contextlib.redirect_stdout(io.StringIO()):
            payload = _e2_difficulty_companions(by_variant, [3, 6])
        self.assertEqual(payload["reference_length"], 3)
        self.assertEqual(payload["endpoint_length"], 6)
        contrast = payload["contrasts"]["calib - none"]
        self.assertEqual(contrast["accuracy_sign_agreement"], "consistent")
        self.assertEqual(contrast["drop_sign_agreement"], "consistent")
        self.assertTrue(contrast["raw_and_standardized_agree"])
        # Force a raw/std sign reversal: the companion must flag it rather
        # than reporting the raw endpoint as a clean effect.
        by_variant["calib"][0]["difficulty_standardized_accuracy_at_longest"] = 0.35
        by_variant["calib"][0]["difficulty_standardized_accuracy_drop"] = 0.25
        with contextlib.redirect_stdout(io.StringIO()):
            reversed_payload = _e2_difficulty_companions(by_variant, [3, 6])
        reversed_contrast = reversed_payload["contrasts"]["calib - none"]
        self.assertEqual(reversed_contrast["accuracy_sign_agreement"],
                         "REVERSED")
        self.assertFalse(reversed_contrast["raw_and_standardized_agree"])



class DifficultyStrataTest(unittest.TestCase):
    def setUp(self):
        # [k1 v1 k2 v2 k3 v3 SEP query]; the answer is stored separately.
        self.idx = torch.tensor(
            [[5, 10, 7, 7, 9, 12, 1, 7],
             [3, 20, 4, 20, 5, 9, 1, 4]], dtype=torch.long)
        self.answer = torch.tensor([7, 20], dtype=torch.long)

    def test_features_match_the_sequence_structure(self):
        features = difficulty_features(self.idx, self.answer, 3)
        self.assertEqual(features["key_pos"].tolist(), [1, 1])
        self.assertEqual(features["value_repeats"].tolist(), [0, 1])
        self.assertEqual(features["naive_confusable"].tolist(), [1, 0])

    def test_stratum_reproduces_the_historical_joint_encoding(self):
        # With one bin per pair, the stratum id must equal the audit's
        # ``difficulty_joint`` encoding: key_pos * 4 + repeat * 2 + naive.
        strata = difficulty_strata(self.idx, self.answer, 3, n_key_bins=3)
        self.assertEqual(strata["stratum"].tolist(), [5, 6])
        self.assertEqual(strata["key_bin"].tolist(), [1, 1])
        self.assertEqual(strata["ambiguity"].tolist(), [1, 2])

    def test_labels_cover_every_stratum(self):
        labels = difficulty_stratum_labels(DEFAULT_DIFFICULTY_KEY_BINS)
        self.assertEqual(len(labels), 4 * DEFAULT_DIFFICULTY_KEY_BINS)
        self.assertEqual([row["id"] for row in labels],
                         list(range(len(labels))))
        self.assertEqual({row["ambiguity"] for row in labels},
                         {"none", "naive_confusable", "value_repeats",
                          "both"})

    def test_generated_batches_use_the_same_layout(self):
        idx, _, meta = make_batch(
            8, 4, 32, "cpu", gen=torch.Generator().manual_seed(3))
        strata = difficulty_strata(idx, meta["answer"], 4)
        self.assertEqual(strata["stratum"].shape, (8,))
        self.assertTrue(bool((strata["stratum"] >= 0).all()))
        self.assertTrue(bool((strata["stratum"]
                              < 4 * DEFAULT_DIFFICULTY_KEY_BINS).all()))

    def test_malformed_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            difficulty_features(self.idx[:, :-1], self.answer, 3)
        with self.assertRaises(ValueError):
            difficulty_features(self.idx, self.answer[:-1], 3)
        with self.assertRaises(ValueError):
            difficulty_strata(self.idx, self.answer, 3, n_key_bins=0)
        with self.assertRaises(ValueError):
            difficulty_features(self.idx, self.answer, True)

    def test_missing_or_duplicated_query_key_is_rejected(self):
        # No key matches the final query token: argmax would otherwise
        # label pair 0, inventing an easy retrieval sample.
        no_match = self.idx.clone()
        no_match[:, -1] = 31
        with self.assertRaises(ValueError):
            difficulty_features(no_match, self.answer, 3)
        # Two keys equal the query: the paired answer is ambiguous.
        duplicate = self.idx.clone()
        duplicate[:, 2] = duplicate[:, 0]
        with self.assertRaises(ValueError):
            difficulty_features(duplicate, self.answer, 3)

    def test_answer_must_match_the_paired_value(self):
        with self.assertRaises(ValueError):
            difficulty_features(self.idx, self.answer + 1, 3)


class EmptySweepCompanionTest(unittest.TestCase):
    def test_empty_sweep_companions_abstain(self):
        from cage.experiments import (
            _e2_difficulty_companions,
            _e2_endpoint_class_companions,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            difficulty = _e2_difficulty_companions({}, [])
            endpoint = _e2_endpoint_class_companions({}, [])
        self.assertEqual(difficulty["by_arm"], {})
        self.assertEqual(difficulty["contrasts"], {})
        self.assertIsNone(difficulty["reference_length"])
        self.assertEqual(endpoint["rows"], [])

    def test_empty_sweep_cost_and_tuned_companions_abstain(self):
        from cage.experiments import (
            Settings,
            _e2_per_variant_tuning,
            _length_efficiency_companion,
        )

        settings = Settings.quick()
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                cost = _length_efficiency_companion(
                    settings, out, "e2", [], [])
                tuned = _e2_per_variant_tuning(
                    settings, out, 1e-3, {}, [])
        self.assertEqual(cost["lengths"], [])
        self.assertEqual(cost["cells"], {})
        self.assertFalse(tuned["available"])
        self.assertIn("empty length sweep", tuned["reason"])


class MalformedLengthRecordTest(unittest.TestCase):
    def test_companions_abstain_on_non_mapping_length_rows(self):
        from cage.experiments import (
            _E2_VARIANTS,
            _counterfactual_companion,
            _e2_difficulty_companions,
            _e2_endpoint_class_companions,
            _stamp_difficulty_standardization,
        )

        def run(gate):
            return {
                "seed": 0, "gate": gate, "model_cfg": {"gate": gate},
                "accuracy": 0.4,
                "accuracy_at_longest": 0.3,
                "accuracy_drop": 0.1,
                "metrics_by_length": {3: "not-a-record",
                                      6: ["not", "a", "record"]},
                "accuracy_by_length": {3: 0.4, 6: 0.3},
                "eval_n_by_length": {3: 10, 6: 10},
                "cf_value_follow_rate": 0.5,
                "cf_distractor_unchanged_rate": 1.0,
                "cf_relocation_unchanged_rate": 1.0,
            }

        by_variant = {label: [run(gate)]
                      for label, gate, *_ in _E2_VARIANTS}
        with contextlib.redirect_stdout(io.StringIO()):
            diff = _e2_difficulty_companions(by_variant, [3, 6])
            endpoint = _e2_endpoint_class_companions(by_variant, [3, 6])
            with tempfile.TemporaryDirectory() as tmp:
                cf = _counterfactual_companion(
                    pathlib.Path(tmp), "e2", by_variant, lengths=[3, 6],
                    reference="none")
        self.assertIn("calib - none", diff["contrasts"])
        self.assertTrue(endpoint["rows"])
        self.assertIn("calib - none", cf["contrasts"])
        rec = {"metrics_by_length": {3: "not-a-record",
                                     6: "not-a-record"}}
        _stamp_difficulty_standardization(rec, 3, [3, 6])
        self.assertTrue(math.isnan(
            rec["difficulty_standardized_accuracy_at_base"]))

    def test_report_companions_skip_non_mapping_run_records(self):
        from cage.experiments import (
            _E2_VARIANTS,
            Settings,
            _counterfactual_companion,
            _e2_difficulty_companions,
            _e2_endpoint_class_companions,
            _efficiency_companion,
            _length_efficiency_companion,
        )

        bad = "not-a-record"
        by_variant = {label: [bad]
                      for label, gate, *_ in _E2_VARIANTS}
        cells = [(label, [bad]) for label, gate, *_ in _E2_VARIANTS]
        settings = Settings.quick()
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                diff = _e2_difficulty_companions(by_variant, [3, 6])
                endpoint = _e2_endpoint_class_companions(
                    by_variant, [3, 6])
                cf = _counterfactual_companion(
                    out, "e2", by_variant, lengths=[3, 6],
                    reference="none")
                cost = _length_efficiency_companion(
                    settings, out, "e2", cells, [3, 6])
                flat = _efficiency_companion(
                    settings, out, "e2", cells)
        self.assertEqual(
            diff["contrasts"]["calib - none"]
            ["raw_accuracy_at_longest"]["n"], 0)
        self.assertTrue(all(row["n"] == 0 for row in endpoint["rows"]))
        self.assertEqual(
            cf["contrasts"]["calib - none"]["3"]["n"], 0)
        for label, _ in cells:
            self.assertFalse(cost["cells"][label]["available"])
            self.assertFalse(flat["cells"][label]["available"])


class StandardizedProportionTest(unittest.TestCase):
    def test_self_weights_reproduce_pooled_rate(self):
        out = standardized_proportion([9, 1], [10, 10])
        self.assertAlmostEqual(out["rate"], 0.5, places=12)
        self.assertAlmostEqual(out["pooled_rate"], 0.5, places=12)
        self.assertEqual(out["coverage"], 1.0)
        self.assertAlmostEqual(out["effective_n"], 20.0, places=12)
        self.assertEqual(out["min_stratum_n"], 10)
        self.assertLessEqual(out["lo"], out["rate"])
        self.assertLessEqual(out["rate"], out["hi"])

    def test_reference_mix_is_applied(self):
        out = standardized_proportion([9, 1], [10, 10], [90, 10])
        self.assertAlmostEqual(out["rate"], 0.82, places=12)
        self.assertEqual(out["n_used"], 2)

    def test_unobserved_stratum_is_dropped_and_reported(self):
        out = standardized_proportion([9, 0], [10, 0], [100, 100])
        self.assertAlmostEqual(out["rate"], 0.9, places=12)
        self.assertEqual(out["coverage"], 0.5)
        self.assertEqual(out["n_used"], 1)

    def test_malformed_input_abstains(self):
        for successes, totals, reference in (
                ([], [], None),
                ([1], [1, 1], None),
                ([0.5, 1], [1, 1], None),
                ([2, 0], [1, 1], None),
                ([-1, 1], [1, 1], None),
                ([1, 1], [2, 2], [-1, 1]),
                ([1, 1], [2, 2], [0, 0]),
                ([1, 1], [2, 2], [1]),
        ):
            with self.subTest(successes=successes, totals=totals,
                              reference=reference):
                out = standardized_proportion(successes, totals, reference)
                self.assertFalse(out["valid"])
                self.assertTrue(out["reason"])


class MissingDifficultyRecordTest(unittest.TestCase):
    def test_stamp_does_not_require_a_mapping(self):
        from cage.experiments import _stamp_difficulty_standardization

        self.assertIsNone(_stamp_difficulty_standardization(None, 3, [3, 6]))
        self.assertEqual(
            _stamp_difficulty_standardization([], 3, [3, 6]), [])

    def test_stamp_abstains_without_metrics_by_length(self):
        from cage.experiments import _stamp_difficulty_standardization

        for rec in ({}, {"metrics_by_length": None},
                    {"metrics_by_length": {3: {}}}):
            with self.subTest(rec=rec):
                _stamp_difficulty_standardization(rec, 3, [3, 6])
                self.assertTrue(math.isnan(
                    rec["difficulty_standardized_accuracy_at_base"]))
                self.assertTrue(math.isnan(
                    rec["difficulty_standardized_accuracy_at_longest"]))
                self.assertEqual(rec["difficulty_standardized_coverage_by_length"][3], 0.0)

    def test_stamp_abstains_on_an_empty_sweep(self):
        from cage.experiments import _stamp_difficulty_standardization

        rec = {"metrics_by_length": {3: {"difficulty": {
            "strata": [{"id": 0, "n": 2, "correct": 1}]}}}}
        _stamp_difficulty_standardization(rec, 3, [])
        self.assertTrue(math.isnan(
            rec["difficulty_standardized_accuracy_at_base"]))
        self.assertTrue(math.isnan(
            rec["difficulty_standardized_accuracy_slope_log_len"]))
        self.assertEqual(rec["difficulty_standardized_min_stratum_n_at_longest"], 0)


class EvalDifficultyRecordTest(unittest.TestCase):
    def test_eval_metrics_stamps_per_stratum_counts(self):
        model = build({
            "vocab_size": 17, "d_model": 16, "n_heads": 4,
            "n_layers": 1, "max_len": 32, "gate": "calib",
        }, compile=False)
        out = eval_metrics(model, 3, 17, "cpu", batch=4, batches=2,
                           seed=123)
        block = out["difficulty"]
        self.assertEqual(block["n_pairs"], 3)
        self.assertEqual(len(block["strata"]), 12)
        self.assertEqual(sum(row["n"] for row in block["strata"]), out["n"])
        self.assertEqual(sum(row["correct"] for row in block["strata"]),
                         round(out["accuracy"] * out["n"]))
        self.assertGreaterEqual(block["observed_strata"], 1)
        self.assertLessEqual(block["observed_strata"], 12)
        self.assertAlmostEqual(
            block["coverage_of_possible"],
            block["observed_strata"] / 12, places=12)


if __name__ == "__main__":
    unittest.main()


class DifficultyVectorAlignmentTest(unittest.TestCase):
    def test_reordered_strata_are_aligned_by_id(self):
        from cage.experiments import _difficulty_vectors

        block = {"strata": [
            {"id": 1, "n": 10, "correct": 9},
            {"id": 0, "n": 90, "correct": 9},
        ]}
        totals, correct = _difficulty_vectors(block)
        self.assertEqual(totals, [90, 10])
        self.assertEqual(correct, [9, 9])

    def test_integral_float_ids_are_a_json_round_trip(self):
        from cage.experiments import _difficulty_vectors

        totals, correct = _difficulty_vectors({
            "n_strata": 4.0,
            "strata": [{"id": 2.0, "n": 5, "correct": 1}],
        })
        self.assertEqual(totals, [0, 0, 5, 0])
        self.assertEqual(correct, [0, 0, 1, 0])

    def test_partial_strata_keep_their_id_slot(self):
        from cage.experiments import _difficulty_vectors

        totals, correct = _difficulty_vectors(
            {"strata": [{"id": 2, "n": 5, "correct": 1}]})
        self.assertEqual(totals, [0, 0, 5])
        self.assertEqual(correct, [0, 0, 1])

    def test_declared_encoding_pads_compact_partial_records(self):
        from cage.experiments import _difficulty_vectors

        totals, correct = _difficulty_vectors({
            "n_strata": 4,
            "strata": [{"id": 2, "n": 5, "correct": 1}],
        })
        self.assertEqual(totals, [0, 0, 5, 0])
        self.assertEqual(correct, [0, 0, 1, 0])
        # JSON round-trips may spell the declared size as an integral float.
        totals, correct = _difficulty_vectors({
            "n_strata": 4.0,
            "strata": [{"id": 2, "n": 5, "correct": 1}],
        })
        self.assertEqual(totals, [0, 0, 5, 0])
        self.assertEqual(correct, [0, 0, 1, 0])
        for bad_size in (True, 3.5, -1, "4"):
            with self.subTest(n_strata=bad_size):
                self.assertEqual(_difficulty_vectors({
                    "n_strata": bad_size,
                    "strata": [{"id": 0, "n": 1, "correct": 1}],
                }), ([], []))

    def test_legacy_rows_without_ids_fall_back_to_order(self):
        from cage.experiments import _difficulty_vectors

        totals, correct = _difficulty_vectors(
            {"strata": [{"n": 10, "correct": 9}, {"n": 90, "correct": 9}]})
        self.assertEqual((totals, correct), ([10, 90], [9, 9]))

    def test_malformed_counts_abstain(self):
        from cage.experiments import _difficulty_vectors

        for block in (
                {"strata": [{"id": 0, "n": -1, "correct": 0}]},
                {"strata": [{"id": 0, "n": 2, "correct": 3}]},
                {"strata": [{"id": 0, "n": 2.5, "correct": 1}]},
                {"strata": [{"id": True, "n": 2, "correct": 1}]},
                {"strata": [{"id": 0, "n": 2}]},
                {"strata": [{"id": 0, "correct": 1}]},
                {"strata": [{"n": 2}]},
                {"strata": [{"id": 0, "n": 2}, {"n": 2, "correct": 1}]},
                {"strata": "not-a-list"},
                {"strata": [["not", "a", "row"]]},
                {"n_strata": 1, "strata": [{"id": 2, "n": 1, "correct": 1}]},
                ["not", "a", "block"]):
            with self.subTest(block=block):
                self.assertEqual(_difficulty_vectors(block), ([], []))

    def test_undetermined_endpoint_is_not_reported_as_agreement(self):
        from cage.experiments import _e2_difficulty_companions

        def run(gate, raw_acc, std_acc, raw_drop, std_drop):
            return {
                "seed": 0, "gate": gate, "model_cfg": {"gate": gate},
                "accuracy_at_longest": raw_acc,
                "accuracy_drop": raw_drop,
                "difficulty_standardized_accuracy_at_longest": std_acc,
                "difficulty_standardized_accuracy_drop": std_drop,
            }

        from cage.experiments import _E2_VARIANTS

        by_variant = {
            label: [run(gate, 0.4, 0.45, 0.2, 0.15)]
            for label, gate, *_ in _E2_VARIANTS
        }
        by_variant["calib"] = [run("calib", None, None, None, None)]
        with contextlib.redirect_stdout(io.StringIO()):
            payload = _e2_difficulty_companions(by_variant, [3, 6])
        contrast = payload["contrasts"]["calib - none"]
        self.assertEqual(contrast["accuracy_sign_agreement"], "undetermined")
        self.assertEqual(contrast["drop_sign_agreement"], "undetermined")
        self.assertFalse(contrast["raw_and_standardized_agree"])


class DifficultyDeviceTest(unittest.TestCase):
    def test_position_indices_follow_the_batch_device(self):
        # A CUDA batch with CPU-side answer metadata must not compare a
        # device-local tensor with a CPU ``arange``.  ``meta`` exercises the
        # same placement rule without requiring a GPU.
        idx = torch.zeros((1, 8), dtype=torch.long, device="meta")
        answer = torch.zeros(1, dtype=torch.long)
        features = difficulty_features(idx, answer, 3)
        for value in features.values():
            self.assertEqual(value.device, idx.device)
