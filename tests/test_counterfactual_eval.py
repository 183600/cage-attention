"""Deterministic minimal-pair data edits used to audit *how* a model answers.

Accuracy on a fresh sequence mixes genuine in-context retrieval with
memorised key -> value associations, fixed-position copying and global
insensitivity to token edits.  These tests pin the three data-level probes
(value swap, distractor swap, pair relocation), their aggregate rates and
the fact that a perfect retrieval model passes all of them.
"""

from __future__ import annotations

import contextlib
import io
import math
import pathlib
import tempfile
import unittest

import torch
import torch.nn as nn

from cage.data import make_batch
from cage.experiments import _counterfactual_companion
from cage.train import (
    _association_exposure_summary,
    _blank_exposure_cohort,
    _counterfactual_counts,
    _counterfactual_edits,
    _counterfactual_summary,
    eval_metrics,
)


class CounterfactualEditTest(unittest.TestCase):
    def setUp(self):
        self.vocab = 17
        self.n_pairs = 4
        self.idx, _, self.meta = make_batch(
            6, self.n_pairs, self.vocab, "cpu",
            gen=torch.Generator().manual_seed(11))
        self.edits = _counterfactual_edits(
            self.idx, self.meta, self.n_pairs, self.vocab)

    def test_value_swap_changes_only_the_paired_value(self):
        swapped = self.edits["value_swap"]
        changed = (swapped != self.idx).sum(dim=1)
        self.assertEqual(changed.tolist(), [1] * self.idx.size(0))
        rows = torch.arange(self.idx.size(0))
        self.assertTrue(bool(swapped[rows, self.meta["key_pos"] + 1].eq(
            self.edits["value_swap_answer"]).all()))
        self.assertFalse(bool(
            self.edits["value_swap_answer"].eq(self.meta["answer"]).any()))
        self.assertTrue(bool((self.edits["value_swap_answer"] >= 2).all()))
        self.assertTrue(bool((self.edits["value_swap_answer"]
                              < self.vocab).all()))
        self.assertTrue(torch.equal(
            self.edits["value_swap_meta"]["answer"],
            self.edits["value_swap_answer"]))
        # The caller's batch is never mutated.
        self.assertTrue(torch.equal(
            self.idx, make_batch(
                6, self.n_pairs, self.vocab, "cpu",
                gen=torch.Generator().manual_seed(11))[0]))

    def test_distractor_swap_leaves_the_answer_unchanged(self):
        swapped = self.edits["distractor_swap"]
        self.assertIsNotNone(swapped)
        changed = (swapped != self.idx).sum(dim=1)
        self.assertEqual(changed.tolist(), [1] * self.idx.size(0))
        rows = torch.arange(self.idx.size(0))
        keys = self.idx.gather(1, self.meta["key_pos"].unsqueeze(1)).squeeze(1)
        self.assertTrue(bool(swapped.gather(
            1, self.meta["key_pos"].unsqueeze(1)).squeeze(1).eq(keys).all()))
        self.assertEqual(self.edits["distractor_swap_meta"]["answer"].tolist(),
                         self.meta["answer"].tolist())

    def test_pair_relocation_preserves_the_key_value_association(self):
        moved = self.edits["relocation"]
        self.assertIsNotNone(moved)
        new_pos = self.edits["relocation_meta"]["key_pos"]
        rows = torch.arange(self.idx.size(0))
        query = moved[:, -1]
        self.assertTrue(bool(moved.gather(1, new_pos.unsqueeze(1)).squeeze(1)
                              .eq(query).all()))
        self.assertTrue(bool(moved.gather(
            1, (new_pos + 1).unsqueeze(1)).squeeze(1)
            .eq(self.meta["answer"]).all()))
        # The queried pair moved to a different pair slot.
        self.assertTrue(bool(new_pos.ne(self.meta["key_pos"]).all()))
        # The edit is a permutation of pair blocks: the multiset of tokens
        # is unchanged.
        self.assertTrue(torch.equal(moved.sort(dim=1).values,
                                    self.idx.sort(dim=1).values))

    def test_single_pair_has_no_distractor_or_relocation_probe(self):
        idx, _, meta = make_batch(
            3, 1, self.vocab, "cpu",
            gen=torch.Generator().manual_seed(12))
        edits = _counterfactual_edits(idx, meta, 1, self.vocab)
        self.assertIsNotNone(edits["value_swap"])
        self.assertIsNone(edits["distractor_swap"])
        self.assertIsNone(edits["relocation"])

    def test_malformed_metadata_is_rejected(self):
        bad = dict(self.meta)
        bad["key_pos"] = bad["key_pos"] + 1
        with self.assertRaises(ValueError):
            _counterfactual_edits(self.idx, bad, self.n_pairs, self.vocab)
        bad_answer = dict(self.meta)
        bad_answer["answer"] = bad_answer["answer"] + 1
        with self.assertRaises(ValueError):
            _counterfactual_edits(self.idx, bad_answer, self.n_pairs,
                                  self.vocab)
        with self.assertRaises(ValueError):
            _counterfactual_edits(self.idx, self.meta, self.n_pairs, 3)
        for n_pairs in (0, -1, True, 4.0, "4"):
            with self.subTest(n_pairs=n_pairs):
                with self.assertRaises(ValueError):
                    _counterfactual_edits(self.idx, self.meta, n_pairs,
                                          self.vocab)
        for vocab in (True, 3.0, 17.5, "17"):
            with self.subTest(vocab=vocab):
                with self.assertRaises(ValueError):
                    _counterfactual_edits(self.idx, self.meta, self.n_pairs,
                                          vocab)

    def test_non_integral_tokens_and_metadata_are_rejected(self):
        bad_idx = self.idx.float()
        with self.assertRaises(ValueError):
            _counterfactual_edits(bad_idx, self.meta, self.n_pairs,
                                  self.vocab)
        for field, value in (("key_pos", 0.5), ("answer", 2.5)):
            with self.subTest(field=field):
                bad = dict(self.meta)
                bad[field] = self.meta[field].float() + value
                with self.assertRaises(ValueError):
                    _counterfactual_edits(self.idx, bad, self.n_pairs,
                                          self.vocab)
        bad_bool = dict(self.meta, key_pos=self.meta["key_pos"].bool())
        with self.assertRaises(ValueError):
            _counterfactual_edits(self.idx, bad_bool, self.n_pairs,
                                  self.vocab)

    def test_edits_are_deterministic(self):
        again = _counterfactual_edits(
            self.idx.clone(), self.meta, self.n_pairs, self.vocab)
        for key in ("value_swap", "distractor_swap", "relocation"):
            self.assertTrue(torch.equal(self.edits[key], again[key]))

    def test_narrow_integer_batch_is_widened_for_edits(self):
        edits = _counterfactual_edits(
            self.idx.to(torch.int32), self.meta, self.n_pairs, self.vocab)
        for key in ("value_swap", "distractor_swap", "relocation"):
            self.assertEqual(edits[key].dtype, torch.long)
        self.assertTrue(torch.equal(
            edits["value_swap"], self.edits["value_swap"]))
        # The input is not mutated in place.
        self.assertEqual(self.idx.dtype, torch.long)


class CounterfactualCountsTest(unittest.TestCase):
    def _counts(self):
        # Rows 0-2 are clean-correct; row 3 is clean-wrong.
        original = torch.tensor([5, 5, 5, 7])
        answer = torch.tensor([5, 5, 5, 5])
        swapped_answer = torch.tensor([6, 6, 6, 6])
        swapped = torch.tensor([6, 6, 5, 6])
        distractor = torch.tensor([5, 5, 6, 7])
        relocation = torch.tensor([5, 5, 5, 7])
        return _counterfactual_counts(
            original, answer, swapped, swapped_answer,
            distractor, relocation)

    def test_counts_condition_on_clean_correct_rows(self):
        counts = self._counts()
        self.assertEqual(counts["n"], 4)
        self.assertEqual(counts["n_original_correct"], 3)
        # Two of the three clean-correct rows follow the new value; the
        # third stubbornly repeats the old answer.
        self.assertEqual(counts["n_value_follow"], 2)
        self.assertEqual(counts["n_value_stubborn"], 1)
        self.assertEqual(counts["n_value_flip"], 3)
        self.assertEqual(counts["n_value_correct"], 3)
        self.assertEqual(counts["n_distractor_unchanged"], 3)
        self.assertEqual(counts["n_distractor_correct"], 2)
        self.assertEqual(counts["n_relocation_unchanged"], 4)
        self.assertEqual(counts["n_relocation_correct"], 3)

    def test_summary_reports_rates_and_flat_fields(self):
        out = _counterfactual_summary(self._counts())
        self.assertAlmostEqual(out["value_swap"]["follow_rate"], 2 / 3)
        self.assertAlmostEqual(out["value_swap"]["stubborn_rate"], 1 / 3)
        self.assertAlmostEqual(out["value_swap"]["flip_rate"], 3 / 4)
        self.assertAlmostEqual(out["distractor_swap"]["unchanged_rate"], 3 / 4)
        self.assertAlmostEqual(out["pair_relocation"]["unchanged_rate"], 1.0)
        self.assertAlmostEqual(out["flat"]["cf_value_follow_rate"], 2 / 3)
        self.assertAlmostEqual(
            out["flat"]["cf_relocation_unchanged_rate"], 1.0)

    def test_non_integral_predictions_are_rejected(self):
        valid = torch.tensor([5, 5, 5, 7])
        for bad in (torch.tensor([5.5, 5.0, 5.0, 7.0]),
                    torch.tensor([True, False, True, False]),
                    torch.tensor([5 + 1j, 5, 5, 7]),
                    # float32 would round this to the valid token 5.
                    [5.0, 5.0, 5.0, 5.0000000001]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _counterfactual_counts(
                        bad, valid, valid, valid, valid, valid)

    def test_empty_counts_abstain(self):
        for counts in ({}, None, [], "bad", object()):
            with self.subTest(counts=type(counts).__name__):
                out = _counterfactual_summary(counts)
                self.assertEqual(out["value_swap"]["n"], 0)
                self.assertTrue(math.isnan(
                    out["value_swap"]["follow_rate"]))
                self.assertTrue(math.isnan(
                    out["distractor_swap"]["unchanged_rate"]))
                self.assertTrue(math.isnan(
                    out["pair_relocation"]["unchanged_rate"]))

    def test_malformed_or_impossible_counts_abstain(self):
        cases = (
            {"n": "x"}, {"n": None}, {"n": 1.5}, {"n": float("nan")},
            {"n": float("inf")}, {"n": True}, {"n": -1},
            {"n": 1, "n_original_correct": 2},
            {"n": 1, "n_value_follow": 2},
            {"n": 1, "n_value_stubborn": 2},
            {"n": 1, "n_value_flip": 2},
            {"n": 1, "n_value_correct": 2},
            {"n": 2, "n_distractor": 1, "n_distractor_unchanged": 2},
            {"n": 2, "n_distractor": 1, "n_distractor_correct": 2},
            {"n": 2, "n_distractor": 3},
            {"n": 2, "n_relocation": 1, "n_relocation_unchanged": 2},
            {"n": 2, "n_relocation_correct": 3},
        )
        rate_keys = (
            "cf_value_follow_rate", "cf_value_stubborn_rate",
            "cf_value_accuracy", "cf_value_flip_rate",
            "cf_distractor_unchanged_rate", "cf_distractor_accuracy",
            "cf_relocation_unchanged_rate", "cf_relocation_accuracy",
        )
        for counts in cases:
            with self.subTest(counts=counts):
                out = _counterfactual_summary(counts)
                self.assertEqual(out["value_swap"]["n"], 0)
                self.assertEqual(out["flat"]["cf_n"], 0)
                for key in rate_keys:
                    self.assertTrue(math.isnan(out["flat"][key]), key)

    def test_integral_json_float_counts_are_accepted(self):
        counts = {key: float(value) for key, value in self._counts().items()}
        out = _counterfactual_summary(counts)
        self.assertEqual(out["value_swap"]["n"], 4)
        self.assertEqual(out["value_swap"]["n_correct_after_edit"], 3)
        self.assertAlmostEqual(out["value_swap"]["follow_rate"], 2 / 3)


class AssociationExposureSummaryTest(unittest.TestCase):
    def test_malformed_or_impossible_cohorts_abstain(self):
        blank = _blank_exposure_cohort()
        cases = (
            {"seen": {}},
            {"seen": None},
            {"seen": []},
            {"seen": {"n": "x"}},
            {"seen": dict(blank, n=-1)},
            {"seen": dict(blank, n=1, n_correct=2)},
            {"seen": dict(blank, n=2, n_value_swap=3)},
            {"seen": dict(blank, n=2, n_value_swap=2,
                           n_value_correct=3)},
            {"seen": dict(blank, n=2, n_correct=1,
                           n_value_swap=1, n_value_follow=2)},
            {"seen": dict(blank, n=2, n_distractor=1,
                           n_distractor_correct=2)},
            {"seen": dict(blank, n=2, n_relocation=1,
                           n_relocation_correct=2)},
        )
        for totals in cases:
            with self.subTest(totals=totals):
                out = _association_exposure_summary(totals)
                self.assertEqual(out["n_items"], 0)
                self.assertEqual(out["seen"]["n"], 0)
                self.assertEqual(out["unseen"]["n"], 0)
                self.assertTrue(math.isnan(out["seen"]["accuracy"]))
                self.assertTrue(math.isnan(
                    out["accuracy_gap_seen_minus_unseen"]))

    def test_missing_cohort_is_a_zero_row_stratum_not_a_malformed_row(self):
        seen = dict(_blank_exposure_cohort(), n=10, n_correct=8)
        out = _association_exposure_summary({"seen": seen})
        self.assertEqual(out["n_items"], 10)
        self.assertAlmostEqual(out["seen"]["accuracy"], 0.8)
        self.assertEqual(out["unseen"]["n"], 0)
        self.assertTrue(math.isnan(out["unseen"]["accuracy"]))
        self.assertTrue(math.isnan(out["accuracy_gap_seen_minus_unseen"]))


class _OracleRetrieval(nn.Module):
    """Perfect retrieval model: copy the value paired with the query key."""

    def __init__(self, vocab: int):
        super().__init__()
        self.vocab = int(vocab)
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, idx):
        batch = idx.size(0)
        keys = idx[:, :-2:2]
        query = idx[:, -1]
        match = keys.eq(query.unsqueeze(1))
        value_pos = match.to(torch.int64).argmax(dim=1).mul(2).add(1)
        answer = idx.gather(1, value_pos.unsqueeze(1)).squeeze(1)
        logits = torch.full((batch, 1, self.vocab), -50.0,
                            device=idx.device)
        logits.scatter_(2, answer.view(batch, 1, 1), 50.0)
        return logits, None, None


class CounterfactualEvalIntegrationTest(unittest.TestCase):
    def test_perfect_retrieval_follows_every_probe(self):
        model = _OracleRetrieval(17)
        out = eval_metrics(model, 3, 17, "cpu", batch=8, batches=2,
                           seed=123)
        self.assertAlmostEqual(out["accuracy"], 1.0)
        self.assertAlmostEqual(out["cf_value_follow_rate"], 1.0)
        self.assertAlmostEqual(out["cf_value_stubborn_rate"], 0.0)
        self.assertAlmostEqual(out["cf_distractor_unchanged_rate"], 1.0)
        self.assertAlmostEqual(out["cf_relocation_unchanged_rate"], 1.0)
        self.assertEqual(out["counterfactual"]["value_swap"]["n"], out["n"])
        # The probes reuse the finite stream, so their batch size matches.
        self.assertEqual(
            out["counterfactual"]["distractor_swap"]["n"], out["n"])
        self.assertEqual(
            out["counterfactual"]["pair_relocation"]["n"], out["n"])

    def test_companion_writes_seed_level_aggregates(self):
        def run(seed, gate, follow):
            return {
                "seed": seed, "gate": gate, "accuracy": 0.5,
                "cf_value_follow_rate": follow,
                "cf_value_stubborn_rate": 1.0 - follow,
                "cf_value_accuracy": 0.5 * follow,
                "cf_value_flip_rate": follow,
                "cf_distractor_unchanged_rate": 1.0,
                "cf_distractor_accuracy": 0.5,
                "cf_relocation_unchanged_rate": 1.0,
                "cf_relocation_accuracy": 0.5,
                "model_cfg": {"gate": gate},
                "training_recipe": {}, "config": {},
            }

        by_arm = {
            "none": [run(0, "none", 0.2), run(1, "none", 0.3)],
            "calib": [run(0, "calib", 0.8), run(1, "calib", 0.9)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _counterfactual_companion(
                    out_dir, "e1", by_arm, reference="none")
            self.assertTrue(
                (out_dir / "e1_counterfactual.json").exists())
        self.assertAlmostEqual(
            payload["arms"]["calib"]["cf_value_follow_rate"]["mean"], 0.85)
        ci = payload["contrasts"]["calib - none"]["cf_value_accuracy"]
        self.assertAlmostEqual(ci["mean"], 0.3)
        self.assertIn("cf_value_follow_rate",
                      payload["contrasts"]["calib - none"])
        self.assertIn("note", payload)

    def test_length_sweep_companion_reads_per_length_metrics(self):
        def run(seed, gate, follow):
            return {
                "seed": seed, "gate": gate, "accuracy": 0.5,
                "cf_value_follow_rate": follow,
                "cf_distractor_unchanged_rate": 1.0,
                "cf_relocation_unchanged_rate": 1.0,
                "metrics_by_length": {
                    3: {"cf_value_follow_rate": follow,
                        "cf_value_accuracy": follow,
                        "cf_distractor_unchanged_rate": 1.0,
                        "cf_relocation_unchanged_rate": 1.0},
                    6: {"cf_value_follow_rate": follow * 0.5,
                        "cf_value_accuracy": follow * 0.5,
                        "cf_distractor_unchanged_rate": 1.0,
                        "cf_relocation_unchanged_rate": 1.0},
                },
                "model_cfg": {"gate": gate},
                "training_recipe": {}, "config": {},
            }

        by_arm = {"none": [run(0, "none", 0.2)],
                  "calib": [run(0, "calib", 0.8)]}
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _counterfactual_companion(
                    pathlib.Path(tmp), "e2", by_arm, lengths=[3, 6],
                    reference="none")
        self.assertEqual(payload["lengths"], [3, 6])
        self.assertAlmostEqual(
            payload["arms"]["calib"]["by_length"]["6"]
            ["cf_value_follow_rate"]["mean"], 0.4)
        self.assertAlmostEqual(
            payload["contrasts"]["calib - none"]["6"]["mean"], 0.3)

    def test_length_sweep_companion_reads_json_string_keys(self):
        def run(seed, gate, follow):
            return {
                "seed": seed, "gate": gate, "accuracy": 0.5,
                "cf_value_follow_rate": follow,
                "cf_distractor_unchanged_rate": 1.0,
                "cf_relocation_unchanged_rate": 1.0,
                "metrics_by_length": {
                    "3": {"cf_value_follow_rate": follow,
                          "cf_value_accuracy": follow,
                          "cf_distractor_unchanged_rate": 1.0,
                          "cf_relocation_unchanged_rate": 1.0},
                    "6": {"cf_value_follow_rate": follow * 0.5,
                          "cf_value_accuracy": follow * 0.5,
                          "cf_distractor_unchanged_rate": 1.0,
                          "cf_relocation_unchanged_rate": 1.0},
                },
                "model_cfg": {"gate": gate},
                "training_recipe": {}, "config": {},
            }

        by_arm = {"none": [run(0, "none", 0.2)],
                  "calib": [run(0, "calib", 0.8)]}
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _counterfactual_companion(
                    pathlib.Path(tmp), "e2", by_arm, lengths=[3, 6],
                    reference="none")
        self.assertAlmostEqual(
            payload["arms"]["calib"]["by_length"]["6"]
            ["cf_value_follow_rate"]["mean"], 0.4)
        self.assertAlmostEqual(
            payload["contrasts"]["calib - none"]["6"]["mean"], 0.3)

    def test_length_sweep_accepts_generators_and_empty_containers(self):
        def run(seed, gate, follow):
            return {
                "seed": seed, "gate": gate, "accuracy": 0.5,
                "cf_value_follow_rate": follow,
                "cf_distractor_unchanged_rate": 1.0,
                "cf_relocation_unchanged_rate": 1.0,
                "metrics_by_length": {
                    3: {"cf_value_follow_rate": follow,
                        "cf_value_accuracy": follow,
                        "cf_distractor_unchanged_rate": 1.0,
                        "cf_relocation_unchanged_rate": 1.0},
                    6: {"cf_value_follow_rate": follow * 0.5,
                        "cf_value_accuracy": follow * 0.5,
                        "cf_distractor_unchanged_rate": 1.0,
                        "cf_relocation_unchanged_rate": 1.0},
                },
                "model_cfg": {"gate": gate},
                "training_recipe": {}, "config": {},
            }

        by_arm = {"none": [run(0, "none", 0.2)],
                  "calib": [run(0, "calib", 0.8)]}
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _counterfactual_companion(
                    pathlib.Path(tmp), "e2", by_arm,
                    lengths=(length for length in (3, 6)), reference="none")
        self.assertEqual(payload["lengths"], [3, 6])
        self.assertAlmostEqual(
            payload["arms"]["calib"]["by_length"]["6"]
            ["cf_value_follow_rate"]["mean"], 0.4)
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _counterfactual_companion(
                    pathlib.Path(tmp), "e2", by_arm, lengths=object(),
                    reference="none")
        self.assertEqual(payload["lengths"], [])
        self.assertNotIn(
            "3", payload["contrasts"].get("calib - none", {}))

    def test_callable_reference_keeps_the_comparison_within_cell(self):
        def run(seed, gate, follow):
            return {
                "seed": seed, "gate": gate,
                "cf_value_follow_rate": follow,
                "cf_distractor_unchanged_rate": 1.0,
                "cf_relocation_unchanged_rate": 1.0,
                "model_cfg": {"gate": gate},
                "training_recipe": {}, "config": {},
            }

        by_cell = {
            ("calib", "pre", None): [run(0, "calib", 0.8)],
            ("query", "pre", None): [run(0, "query", 0.2)],
            ("calib", "seednorm", 1e-2): [run(0, "calib", 0.6)],
            ("query", "seednorm", 1e-2): [run(0, "query", 0.3)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = _counterfactual_companion(
                    pathlib.Path(tmp), "e3", by_cell,
                    reference=lambda cell: ("query", cell[1], cell[2]))
        self.assertIn("calib|pre|off - query|pre|off",
                      payload["contrasts"])
        self.assertIn("calib|seednorm|0.01 - query|seednorm|0.01",
                      payload["contrasts"])
        self.assertEqual(payload["reference"], "per-arm control mapping")


if __name__ == "__main__":
    unittest.main()
