"""Within-item length extension: paired counterfactual for the E2 sweep.

The independent-stream sweep changes length and item composition together.
These tests pin the data transform that holds the item fixed (association,
pair index and query preserved; only distractors added) and the evaluation
plumbing that turns it into paired accuracy / retention / McNemar readouts.
"""

from __future__ import annotations

import unittest

import torch

from cage.data import (
    TEST_SEED,
    association_codes,
    make_batch,
    make_length_linked_batch,
    validate_batch,
)
from cage.experiments import Settings, _stamp_linked_length_extension
from cage.model import build
from cage.train import eval_accuracy, eval_linked_lengths


class PairLookupRule(torch.nn.Module):
    """Exact structural rule as a model, so correctness is known a priori."""

    def __init__(self, vocab: int):
        super().__init__()
        self.vocab = int(vocab)

    def forward(self, idx):
        batch = int(idx.size(0))
        keys = idx[:, :-2:2]
        query = idx[:, -1]
        match = keys.eq(query.unsqueeze(1))
        key_pos = match.to(torch.long).argmax(dim=1).mul(2)
        pred = idx.gather(1, (key_pos + 1).unsqueeze(1)).squeeze(1)
        logits = torch.full((batch, 1, self.vocab), -1e4,
                            dtype=torch.float32, device=idx.device)
        rows = torch.arange(batch, device=idx.device)
        logits[rows, 0, pred] = 0.0
        return logits, None, None


class LinkedBatchTest(unittest.TestCase):
    def setUp(self):
        self.vocab = 32
        self.base_pairs = 4
        self.gen = torch.Generator().manual_seed(71)
        self.idx, self.tgt, self.meta = make_batch(
            6, self.base_pairs, self.vocab, "cpu", gen=self.gen)

    def _extend(self, idx, meta, source, target, seed=72):
        return make_length_linked_batch(
            idx, meta, source, target, self.vocab,
            gen=torch.Generator().manual_seed(seed))

    def test_association_and_structure_are_preserved(self):
        idx, tgt, meta = self._extend(
            self.idx, self.meta, self.base_pairs, 8)
        self.assertEqual(tuple(idx.shape), (6, 2 * 8 + 2))
        validate_batch(idx, tgt, meta, self.vocab)
        self.assertTrue(torch.equal(
            idx[:, :2 * self.base_pairs],
            self.idx[:, :2 * self.base_pairs]))
        self.assertTrue(torch.equal(idx[:, -1], self.idx[:, -1]))
        self.assertTrue(torch.equal(meta["answer"], self.meta["answer"]))
        self.assertTrue(torch.equal(meta["key_pos"], self.meta["key_pos"]))
        base_codes = association_codes(
            self.idx, self.meta, self.vocab)["code"]
        self.assertTrue(torch.equal(
            association_codes(idx, meta, self.vocab)["code"], base_codes))
        keys = idx[:, :2 * 8:2]
        ordered = keys.sort(dim=1).values
        self.assertTrue(bool((ordered[:, 1:] != ordered[:, :-1]).all()))
        # Fresh keys must not duplicate any key of the source batch.
        old_keys = self.idx[:, :2 * self.base_pairs:2]
        new_keys = idx[:, 2 * self.base_pairs:2 * 8:2]
        self.assertFalse(bool(
            (new_keys.unsqueeze(2) == old_keys.unsqueeze(1)).any()))

    def test_same_length_is_a_noop(self):
        idx, tgt, meta = self._extend(
            self.idx, self.meta, self.base_pairs, self.base_pairs)
        self.assertTrue(torch.equal(idx, self.idx))
        self.assertTrue(torch.equal(tgt, self.tgt))
        self.assertTrue(torch.equal(meta["key_pos"], self.meta["key_pos"]))
        self.assertTrue(torch.equal(meta["answer"], self.meta["answer"]))

    def test_narrow_integer_ids_are_widened_not_wrapped(self):
        vocab = 300
        idx, _tgt, meta = make_batch(
            2, 3, vocab, "cpu", gen=torch.Generator().manual_seed(1))
        expected = association_codes(idx, meta, vocab)["code"]
        narrow = association_codes(idx.to(torch.int16), meta, vocab)["code"]
        self.assertEqual(narrow.dtype, torch.long)
        self.assertTrue(torch.equal(narrow, expected))
        out, tgt, out_meta = make_length_linked_batch(
            idx.to(torch.int32), meta, 3, 5, vocab,
            gen=torch.Generator().manual_seed(2))
        self.assertEqual(out.dtype, torch.long)
        self.assertEqual(tgt.dtype, torch.long)
        validate_batch(out, tgt, out_meta, vocab)
        self.assertTrue(torch.equal(
            association_codes(out, out_meta, vocab)["code"], expected))

    def test_nested_extension_keeps_the_shorter_prefix(self):
        mid_idx, mid_tgt, mid_meta = self._extend(
            self.idx, self.meta, self.base_pairs, 8)
        long_idx, long_tgt, long_meta = self._extend(
            mid_idx, mid_meta, 8, 12, seed=73)
        validate_batch(long_idx, long_tgt, long_meta, self.vocab)
        self.assertTrue(torch.equal(long_idx[:, :2 * 8], mid_idx[:, :2 * 8]))
        self.assertTrue(torch.equal(
            association_codes(long_idx, long_meta, self.vocab)["code"],
            association_codes(self.idx, self.meta, self.vocab)["code"]))
        self.assertTrue(torch.equal(
            long_meta["key_pos"], self.meta["key_pos"]))

    def test_extension_is_deterministic_in_its_generator(self):
        first = self._extend(self.idx, self.meta, self.base_pairs, 8, seed=72)
        second = self._extend(self.idx, self.meta, self.base_pairs, 8, seed=72)
        third = self._extend(self.idx, self.meta, self.base_pairs, 8, seed=73)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertFalse(torch.equal(first[0], third[0]))

    def test_invalid_requests_raise_value_error(self):
        for source, target in ((4, 3), (4, 0), (0, 8), (4, 40)):
            with self.subTest(source=source, target=target):
                with self.assertRaises(ValueError):
                    make_length_linked_batch(
                        self.idx, self.meta, source, target, self.vocab)
        for bad_pairs in (4.0, "4", True, None):
            with self.subTest(n_pairs=bad_pairs):
                with self.assertRaises(ValueError):
                    make_length_linked_batch(
                        self.idx, self.meta, bad_pairs, 8, self.vocab)
        with self.assertRaises(ValueError):
            make_length_linked_batch(
                self.idx[:, :-1], self.meta, 4, 8, self.vocab)
        with self.assertRaises(ValueError):
            make_length_linked_batch(
                self.idx, None, 4, 8, self.vocab)


class LinkedLengthEvalTest(unittest.TestCase):
    def setUp(self):
        self.vocab = 32
        self.rule = PairLookupRule(self.vocab)
        self.kwargs = dict(batch=8, batches=2, seed=TEST_SEED)

    def test_rule_model_is_perfect_at_every_linked_length(self):
        out = eval_linked_lengths(
            self.rule, 4, self.vocab, "cpu", [4, 8, 16], **self.kwargs)
        self.assertEqual(out["lengths"], [4, 8, 16])
        self.assertEqual(out["n_items"], 16)
        self.assertTrue(out["nested_extension"])
        self.assertEqual(out["base_n_correct"], 16)
        self.assertEqual(out["base_n_wrong"], 0)
        for length in (4, 8, 16):
            self.assertAlmostEqual(out["accuracy_by_length"][length], 1.0)
            self.assertEqual(out["n_by_length"][length], 16)
            self.assertAlmostEqual(out["retention_by_length"][length], 1.0)
            self.assertIsNone(out["rescue_by_length"][length])
            self.assertAlmostEqual(out["net_change_by_length"][length], 0.0)
            self.assertEqual(out["mcnemar_by_length"][length]["b"], 0)
            self.assertEqual(out["mcnemar_by_length"][length]["c"], 0)
            self.assertAlmostEqual(
                out["mcnemar_by_length"][length]["p_exact"], 1.0)

    def test_linked_base_reproduces_the_plain_eval_stream(self):
        out = eval_linked_lengths(
            self.rule, 4, self.vocab, "cpu", [4, 8], **self.kwargs)
        plain = eval_accuracy(
            self.rule, 4, self.vocab, "cpu", batch=8, batches=2,
            seed=TEST_SEED)
        self.assertAlmostEqual(out["base_accuracy"], plain)
        self.assertAlmostEqual(out["accuracy_by_length"][4], plain)

    def test_tiny_transformer_runs_through_the_probe(self):
        model = build({
            "d_model": 16, "n_layers": 1, "n_heads": 2,
            "max_len": 64, "vocab_size": self.vocab, "gate": "none",
        }, compile=False)
        out = eval_linked_lengths(
            model, 4, self.vocab, "cpu", [4, 8], batch=4, batches=1,
            seed=TEST_SEED)
        self.assertEqual(out["n_items"], 4)
        self.assertEqual(out["n_by_length"][8], 4)
        for value in out["accuracy_by_length"].values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_length_contract_is_validated(self):
        for bad in ([], [4], [8, 4], [4, 4, 8], [4, 8, 8],
                    [4.0, 8.0], [True, 8], "48", None):
            with self.subTest(lengths=bad):
                with self.assertRaises(ValueError):
                    eval_linked_lengths(
                        self.rule, 4, self.vocab, "cpu", bad, **self.kwargs)
        with self.assertRaises(ValueError):
            eval_linked_lengths(
                self.rule, 4, self.vocab, "cpu", [4, 40], **self.kwargs)

    def test_run_record_stamp_exposes_the_paired_fields(self):
        settings = Settings.quick()
        settings.n_pairs = 4
        settings.vocab = self.vocab
        settings.eval_batch = 8
        settings.eval_batches = 1
        rec = {"accuracy_by_length": {4: 1.0, 8: 1.0}}
        _stamp_linked_length_extension(
            rec, self.rule, settings, [4, 8], TEST_SEED)
        self.assertEqual(rec["linked_length_extension"]["lengths"], [4, 8])
        self.assertAlmostEqual(rec["linked_accuracy_at_base"], 1.0)
        self.assertAlmostEqual(rec["linked_accuracy_at_longest"], 1.0)
        self.assertAlmostEqual(rec["linked_accuracy_drop"], 0.0)
        self.assertAlmostEqual(rec["linked_retention_at_longest"], 1.0)
        self.assertIsNone(rec["linked_rescue_at_longest"])
        self.assertAlmostEqual(rec["linked_net_change_at_longest"], 0.0)
        self.assertAlmostEqual(
            rec["linked_composition_effect_at_longest"], 0.0)
        self.assertAlmostEqual(rec["linked_mcnemar_p_at_longest"], 1.0)
        self.assertTrue(rec["linked_base_matches_raw"])
        # ``raw-long - linked-long`` is the composition effect, not a
        # difference of p-values: with a perfect linked probe it equals the
        # raw long-length loss.
        rec2 = {"accuracy_by_length": {4: 1.0, 8: 0.5}}
        _stamp_linked_length_extension(
            rec2, self.rule, settings, [4, 8], TEST_SEED)
        self.assertAlmostEqual(
            rec2["linked_composition_effect_at_longest"], 0.5)

    def test_run_record_stamp_abstains_without_a_mapping(self):
        from cage.experiments import Settings, _stamp_linked_length_extension

        settings = Settings.quick()
        self.assertIsNone(_stamp_linked_length_extension(
            None, self.rule, settings, [], 123))

    def test_run_record_stamp_abstains_on_an_empty_sweep(self):
        settings = Settings.quick()
        rec = {}
        _stamp_linked_length_extension(
            rec, self.rule, settings, [], TEST_SEED)
        self.assertIsNone(rec["linked_length_extension"])
        self.assertEqual(rec["linked_accuracy_by_length"], {})
        self.assertFalse(rec["linked_base_matches_raw"])
        self.assertIsNone(rec["linked_retention_at_longest"])

    def test_run_record_stamp_accepts_a_length_generator(self):
        settings = Settings.quick()
        settings.n_pairs = 4
        settings.vocab = self.vocab
        settings.eval_batch = 8
        settings.eval_batches = 1
        rec = {"accuracy_by_length": {4: 1.0, 8: 1.0}}
        _stamp_linked_length_extension(
            rec, self.rule, settings,
            (length for length in (4, 8)), TEST_SEED)
        self.assertEqual(rec["linked_length_extension"]["lengths"], [4, 8])
        self.assertAlmostEqual(rec["linked_accuracy_at_base"], 1.0)
        self.assertAlmostEqual(rec["linked_accuracy_at_longest"], 1.0)
        self.assertTrue(rec["linked_base_matches_raw"])

    def test_link_seed_must_be_an_integer_stream_seed(self):
        for bad in (True, False, 1.5, None, "3"):
            with self.subTest(link_seed=bad):
                with self.assertRaises(ValueError):
                    eval_linked_lengths(
                        self.rule, 4, self.vocab, "cpu", [4, 8],
                        link_seed=bad, **self.kwargs)

    def test_geometry_and_count_contract_is_validated(self):
        for kwargs in ({"batch": 0}, {"batch": 8.0}, {"batches": 0},
                       {"batches": True}):
            with self.subTest(kwargs=kwargs):
                merged = dict(self.kwargs)
                merged.update(kwargs)
                with self.assertRaises(ValueError):
                    eval_linked_lengths(
                        self.rule, 4, self.vocab, "cpu", [4, 8], **merged)
        for n_pairs in (0, 4.0, True, None):
            with self.subTest(n_pairs=n_pairs):
                with self.assertRaises(ValueError):
                    eval_linked_lengths(
                        self.rule, n_pairs, self.vocab, "cpu", [4, 8],
                        **self.kwargs)


if __name__ == "__main__":
    unittest.main()
