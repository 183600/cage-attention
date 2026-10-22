"""Association-level exposure audit and exposure-stratified evaluation.

Exact-row duplicate audits cannot see the relevant memorisation unit of the
synthetic recall task: the same ``(query key, answer value)`` association can
occur in training and evaluation even when no full sequence repeats.  These
tests pin the association code, the replay/exposure readout, and the
pre-treatment ``seen`` / ``unseen`` split stamped on evaluation records.
"""

from __future__ import annotations

import json
import math
import pathlib
import tempfile
import unittest

import torch
import torch.nn as nn

from cage.data import (
    TEST_SEED,
    VAL_SEED,
    actual_stream_overlap_audit,
    association_codes,
    association_reference_codes,
    make_batch,
    stream_seed_key,
)
from cage.train import eval_metrics


def _draw(seed, batch=4, batches=2, n_pairs=3, vocab=32):
    gen = torch.Generator().manual_seed(stream_seed_key(seed))
    return [make_batch(batch, n_pairs, vocab, "cpu", gen=gen)
            for _ in range(batches)]


class AssociationCodeTest(unittest.TestCase):
    def test_code_is_the_exact_query_pair(self):
        idx, _, meta = _draw(5)[0]
        out = association_codes(idx, meta, 32)
        self.assertTrue(torch.equal(out["key"], idx[:, -1]))
        self.assertTrue(torch.equal(out["value"], meta["answer"]))
        self.assertTrue(torch.equal(
            out["code"], out["key"] * 32 + out["value"]))

    def test_malformed_batches_are_rejected(self):
        idx, _, meta = _draw(6)[0]
        bad_meta = [
            {"answer": meta["answer"]},
            {"key_pos": meta["key_pos"]},
            {"key_pos": meta["key_pos"][:1], "answer": meta["answer"]},
            {"key_pos": meta["key_pos"] + 1, "answer": meta["answer"]},
            {"key_pos": meta["key_pos"], "answer": meta["answer"] + 1},
            {"key_pos": meta["key_pos"] * 2 + 1, "answer": meta["answer"]},
        ]
        for bad in bad_meta:
            with self.subTest(meta=bad.keys()):
                with self.assertRaises(ValueError):
                    association_codes(idx, bad, 32)
        with self.assertRaises(ValueError):
            association_codes(idx.reshape(-1), meta, 32)
        with self.assertRaises(ValueError):
            association_codes(idx, meta, 3)


class AssociationReferenceTest(unittest.TestCase):
    def test_integral_float_codes_are_accepted(self):
        self.assertEqual(association_reference_codes([1, 2.0, 3]),
                         frozenset({1, 2, 3}))

    def test_invalid_references_are_rejected(self):
        for bad in (None, 3, "12", {"a": 1}, [True], [1.5],
                    [float("nan")], [-1], [1 + 2j]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    association_reference_codes(bad)


class AssociationExposureAuditTest(unittest.TestCase):
    def _audit(self):
        return actual_stream_overlap_audit(
            3, 32, "cpu",
            train_specs={
                "train:0": {"seed": 0, "batch": 4, "steps": 8},
            },
            heldout_specs={
                "test": {"seed": TEST_SEED, "batch": 4, "batches": 2},
                "val_monitor": {"seed": VAL_SEED, "batch": 4, "batches": 1,
                                "group": "val"},
                "train_geom2": {"seed": 0, "batch": 5, "batches": 2,
                                "group": "train"},
            },
            include_train_association_codes=True)

    def test_exposure_is_item_weighted_and_consistent(self):
        audit = self._audit()
        train_codes = audit.pop("train_association_codes")
        self.assertIsInstance(train_codes, set)
        self.assertTrue(train_codes)
        self.assertEqual(set(audit["association_exposure"]),
                         {"test", "val_monitor"})
        for name, row in audit["association_exposure"].items():
            with self.subTest(stream=name):
                self.assertGreater(row["n_items"], 0)
                self.assertGreaterEqual(row["pair_seen_train_rate"], 0.0)
                self.assertLessEqual(row["pair_seen_train_rate"], 1.0)
                self.assertGreaterEqual(row["key_seen_train_rate"], 0.0)
                self.assertLessEqual(row["key_seen_train_rate"], 1.0)
                self.assertLessEqual(row["pair_seen_train"],
                                     row["key_seen_train"])
                self.assertEqual(row["pair_unseen_key_seen"],
                                 row["key_seen_train"]
                                 - row["pair_seen_train"])
                self.assertAlmostEqual(
                    row["pair_seen_train_rate"],
                    row["pair_seen_train"] / row["n_items"])
                self.assertAlmostEqual(
                    row["pair_unseen_key_seen_rate"],
                    row["pair_unseen_key_seen"] / row["n_items"])
        self.assertGreaterEqual(
            audit["max_heldout_pair_exposure"],
            max(row["pair_seen_train_rate"]
                for row in audit["association_exposure"].values()))

    def test_exposure_counts_match_an_independent_replay(self):
        audit = self._audit()
        train_codes = audit.pop("train_association_codes")
        test_batches = _draw(TEST_SEED, batch=4, batches=2, n_pairs=3,
                            vocab=32)
        test_codes = [
            int(code) for idx, _, meta in test_batches
            for code in association_codes(idx, meta, 32)["code"].tolist()]
        row = audit["association_exposure"]["test"]
        self.assertEqual(row["n_items"], len(test_codes))
        self.assertEqual(
            row["pair_seen_train"],
            sum(1 for code in test_codes if code in train_codes))
        train_keys = {code // 32 for code in train_codes}
        self.assertEqual(
            row["key_seen_train"],
            sum(1 for code in test_codes if code // 32 in train_keys))

    def test_persistable_payload_has_no_code_set(self):
        audit = actual_stream_overlap_audit(
            3, 32, "cpu",
            train_specs={"train:0": {"seed": 0, "batch": 4, "steps": 2}},
            heldout_specs={"test": {"seed": TEST_SEED, "batch": 4,
                                    "batches": 1}})
        self.assertNotIn("train_association_codes", audit)
        self.assertGreater(
            audit["streams"]["train:0"]["n_unique_associations"], 0)
        json.dumps(audit)  # must stay serializable without the code set

    def test_experiment_layer_keeps_codes_off_disk(self):
        from cage.experiments import Settings, _audit_data_splits

        settings = Settings.quick()
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            audit = _audit_data_splits(settings, out, "unit")
            self.assertIsInstance(settings._train_association_codes,
                                  frozenset)
            self.assertTrue(settings._train_association_codes)
            self.assertIn("association_exposure", audit)
            self.assertNotIn("train_association_codes", audit)
            self.assertNotIn(
                "train_association_codes",
                audit["actual_stream_overlap"])
            payload = json.loads(
                (out / "unit_split_audit.json").read_text(encoding="utf-8"))
            self.assertIn("association_exposure", payload)
            self.assertNotIn("train_association_codes", payload)


class _OracleRetrieval(nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.vocab = int(vocab)
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, idx):
        batch = idx.size(0)
        keys = idx[:, :-2:2]
        match = keys.eq(idx[:, -1].unsqueeze(1))
        value_pos = match.to(torch.int64).argmax(dim=1).mul(2).add(1)
        answer = idx.gather(1, value_pos.unsqueeze(1)).squeeze(1)
        logits = torch.full((batch, 1, self.vocab), -50.0)
        logits.scatter_(2, answer.view(batch, 1, 1), 50.0)
        return logits, None, None


class _MemorisedLookup(nn.Module):
    """Ignore the context and answer from a frozen key -> value table."""

    def __init__(self, mapping, vocab: int, default: int = 2):
        super().__init__()
        self.mapping = dict(mapping)
        self.vocab = int(vocab)
        self.default = int(default)
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, idx):
        batch = idx.size(0)
        pred = torch.tensor(
            [self.mapping.get(int(k), self.default)
             for k in idx[:, -1].tolist()], dtype=torch.long)
        logits = torch.full((batch, 1, self.vocab), -50.0)
        logits.scatter_(2, pred.view(batch, 1, 1), 50.0)
        return logits, None, None


class ExposureStratifiedEvalTest(unittest.TestCase):
    def setUp(self):
        self.batches = _draw(123, batch=8, batches=2, n_pairs=3, vocab=17)
        self.eval_codes = [
            int(code) for idx, _, meta in self.batches
            for code in association_codes(idx, meta, 17)["code"].tolist()]

    def test_oracle_is_correct_in_both_cohorts(self):
        reference = frozenset(self.eval_codes[::2])
        out = eval_metrics(_OracleRetrieval(17), 3, 17, "cpu", batch=8,
                           batches=2, seed=123,
                           train_association_codes=reference)
        block = out["association_exposure"]
        self.assertEqual(block["n_items"], out["n"])
        self.assertEqual(block["n_seen"] + block["n_unseen"], out["n"])
        self.assertEqual(
            block["n_seen"],
            sum(1 for code in self.eval_codes if code in reference))
        self.assertAlmostEqual(block["seen"]["accuracy"], 1.0)
        self.assertAlmostEqual(block["unseen"]["accuracy"], 1.0)
        self.assertAlmostEqual(
            block["seen"]["cf_value_follow_rate"], 1.0)
        self.assertAlmostEqual(
            block["unseen"]["cf_value_follow_rate"], 1.0)

    def test_memorised_lookup_scores_above_chance_only_on_seen_pairs(self):
        reference = frozenset(self.eval_codes[::2])
        mapping = {code // 17: code % 17 for code in reference}
        out = eval_metrics(_MemorisedLookup(mapping, 17), 3, 17, "cpu",
                           batch=8, batches=2, seed=123,
                           train_association_codes=reference)
        block = out["association_exposure"]
        self.assertGreater(block["seen"]["n"], 0)
        self.assertGreater(block["unseen"]["n"], 0)
        self.assertGreater(block["seen"]["accuracy"],
                           block["unseen"]["accuracy"])
        # The frozen table ignores the edited context, so it never follows
        # the swapped value on a clean-correct row.
        self.assertEqual(block["seen"]["cf_value_follow_rate"], 0.0)
        self.assertGreater(block["accuracy_gap_seen_minus_unseen"], 0.0)

    def test_without_a_reference_no_exposure_block_is_claimed(self):
        out = eval_metrics(_OracleRetrieval(17), 3, 17, "cpu", batch=8,
                           batches=2, seed=123)
        self.assertNotIn("association_exposure", out)
        self.assertFalse(any(key.startswith("assoc_exposure_")
                             for key in out))

    def test_bad_reference_is_rejected(self):
        for bad in ([True], [1.5], [-1], "12"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    eval_metrics(_OracleRetrieval(17), 3, 17, "cpu",
                                 batch=4, batches=1, seed=123,
                                 train_association_codes=bad)

    def test_single_pair_cohort_abstains_on_placebo_probes(self):
        # ``n_pairs == 1`` has no distractor pair to edit, so the
        # distractor/relocation probes abstain.  Reporting 0.0 against the
        # value-swap denominator would look like a failed placebo check
        # instead of an unavailable one.
        reference = frozenset(self.eval_codes)
        out = eval_metrics(_OracleRetrieval(17), 1, 17, "cpu", batch=8,
                           batches=2, seed=123,
                           train_association_codes=reference)
        block = out["association_exposure"]
        for cohort in ("seen", "unseen"):
            row = block[cohort]
            self.assertEqual(row["n_value_swappable"], row["n"])
            self.assertEqual(row["n_distractor"], 0)
            self.assertEqual(row["n_relocation"], 0)
            self.assertTrue(math.isnan(row["distractor_accuracy"]))
            self.assertTrue(math.isnan(row["relocation_accuracy"]))

    def test_empty_cohort_rates_abstain(self):
        # Every evaluated pair is in the reference: the unseen cohort has
        # n=0 and must report NaN, not accuracy 0.
        reference = frozenset(self.eval_codes)
        out = eval_metrics(_OracleRetrieval(17), 3, 17, "cpu", batch=8,
                           batches=2, seed=123,
                           train_association_codes=reference)
        block = out["association_exposure"]
        self.assertEqual(block["n_unseen"], 0)
        self.assertTrue(math.isnan(block["unseen"]["accuracy"]))
        self.assertTrue(math.isnan(
            block["unseen"]["cf_value_accuracy"]))


if __name__ == "__main__":
    unittest.main()
