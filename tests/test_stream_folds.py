"""Data-level robustness of the shared finite evaluation stream.

The seed bootstrap treats training seeds as independent units, but every
seed is scored on one finite set of evaluation items.  These tests pin the
fold jackknife over those items and the content-digest guard that prevents
two different evaluation realizations from being paired by position.
"""

from __future__ import annotations

import base64
import math
import unittest

import torch

from cage.data import TEST_SEED
from cage.experiments import (
    Settings,
    _model_cfg,
    _paired_diff,
    _paired_did,
    _paired_stream_test,
    _paired_benchmark,
    _curve_health,
    _curve_valid_runs,
    _curve_warn,
    _factorial_main_effect,
    _grid_level_main_effect,
    _one_sample_contrast,
    _curve_record,
    _steps_run_from_hist,
)
from cage.metrics import (
    crossed_cluster_bootstrap_ci,
    evaluation_fold_audit,
)
from cage.model import build
from cage.repro import seed_everything
from cage.train import (
    decode_correct_mask,
    encode_correct_mask,
    eval_metrics,
    training_curve_verdict,
)


class CurveHistoryInputTest(unittest.TestCase):
    def test_best_effort_step_count_handles_generators_and_bad_rows(self):
        rows = [
            {"step": 0, "train_loss": 1.0},
            {"step": 1, "train_loss": 0.9},
            {"step": 2, "train_loss": 0.8},
        ]
        self.assertEqual(_steps_run_from_hist(rows), 3)
        self.assertEqual(_steps_run_from_hist((r for r in rows)), 3)
        self.assertEqual(_steps_run_from_hist(None), 0)
        self.assertEqual(_steps_run_from_hist([None, 3, "x"]), 0)
        self.assertEqual(
            _steps_run_from_hist([{"train_loss": 1.0}]), 0)
        skipped = rows + [{"step": 3, "train_loss": float("nan")}]
        self.assertEqual(_steps_run_from_hist(skipped), 3)

    def test_curve_record_does_not_require_a_well_formed_history(self):
        for hist in (None, [None], [{"train_loss": 1.0}],
                     (r for r in [{"step": 0, "train_loss": 1.0}])):
            with self.subTest(hist=hist):
                rec = _curve_record(hist, early_stop_info="bad",
                                    seeding="bad")
                self.assertIn("curve_verdict", rec)
                self.assertGreaterEqual(rec["steps_run"], 0)
                self.assertEqual(rec["seeding"], {})
        hist = [{"step": 0, "train_loss": 1.0}]
        for bad in (True, 1.9, -1, "5", None):
            with self.subTest(steps_run=bad):
                rec = _curve_record(hist, {"steps_run": bad})
                self.assertEqual(rec["steps_run"], 1)

    def test_malformed_optional_curve_fields_abstain(self):
        # A curve restored from JSON can carry a hand-edited/malformed
        # optional field.  That point must abstain (None in the series /
        # NaN in the verdict), not raise out of the run-record writer.
        series = {
            "val_loss": "curve_val_loss",
            "val_loss_se": "curve_val_loss_se",
            "val_acc": "curve_val_acc",
            "grad_norm": "curve_grad_norm",
            "lr": "curve_lr",
            "train_loss_window": "curve_train_loss_window",
        }
        for field, series_key in series.items():
            for bad in ("bad", True, False, float("nan"), float("inf"),
                        float("-inf"), "nan"):
                with self.subTest(field=field, bad=bad):
                    hist = [{"step": 0, "train_loss": 1.0, field: bad}]
                    rec = _curve_record(hist)
                    self.assertIsNone(rec[series_key][0])
                    verdict = training_curve_verdict(hist)
                    self.assertTrue(math.isfinite(verdict["n_points"]))
        self.assertEqual(
            _curve_record([{"step": 0, "train_loss": 1.0}])["steps_run"], 1)
        # A point without a usable step is not a curve point.
        out = training_curve_verdict(
            [{"step": "x", "train_loss": 1.0}])
        self.assertEqual(out["n_points"], 0)
        # A null/missing mandatory loss is a non-finite point, not a crash.
        for missing in (None, {"train_loss": None}):
            with self.subTest(missing=missing):
                hist = [{"step": 0, "val_loss": 2.0}] if missing is None \
                    else [{"step": 0, **missing}]
                rec = _curve_record(hist)
                self.assertEqual(rec["curve_verdict"],
                                 "numerical_instability")
        # A boolean mandatory loss/step is a flag, not a measurement: it
        # must abstain instead of being read as 0/1 on the curve.
        for bad in (True, False):
            with self.subTest(field="train_loss", bad=bad):
                out = training_curve_verdict(
                    [{"step": 0, "train_loss": bad}])
                self.assertEqual(out["n_points"], 0)
                self.assertEqual(out["verdict"], "numerical_instability")
            with self.subTest(field="step", bad=bad):
                out = training_curve_verdict(
                    [{"step": bad, "train_loss": 1.0}])
                self.assertEqual(out["n_points"], 0)


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

    def test_zero_mean_unit_with_opposite_folds_is_flagged(self):
        out = evaluation_fold_audit([1, 1, 0, 0], [0, 0, 1, 1], n_folds=2)
        self.assertTrue(out["valid"])
        self.assertEqual(out["fold_effects"], [1.0, -1.0])
        self.assertEqual(out["unit_fold_sign_flips"], [0])
        self.assertEqual(out["n_unit_fold_sign_flips"], 1)
        self.assertTrue(math.isnan(out["unit_fold_sign_consistency"][0]))

    def test_zero_mean_unit_with_one_fold_direction_is_not_flagged(self):
        out = evaluation_fold_audit([1, 1, 0, 0], [0, 0, 0, 0], n_folds=2)
        self.assertTrue(out["valid"])
        self.assertEqual(out["unit_fold_sign_flips"], [])

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


class CrossedClusterBootstrapTest(unittest.TestCase):
    """Seed x item bootstrap: the stream is a second sampling axis."""

    def test_uniform_effect_is_not_widened(self):
        out = crossed_cluster_bootstrap_ci(
            [[1] * 8, [1] * 8], [[0] * 8, [0] * 8], n_boot=500, seed=0)
        self.assertTrue(out["valid"])
        self.assertEqual(out["mean"], 1.0)
        self.assertEqual(out["lo"], 1.0)
        self.assertEqual(out["hi"], 1.0)
        self.assertEqual(out["sd_total"], 0.0)
        self.assertTrue(out["ci_excludes_zero"])

    def test_single_item_effect_widens_the_interval(self):
        # Every seed has the same +1/8 mean, so the seed-only interval is
        # exactly [0.125, 0.125].  The gain nevertheless sits entirely in
        # one evaluation item; resampling items must expose that a new
        # stream would often show no effect at all.
        out = crossed_cluster_bootstrap_ci(
            [[1] + [0] * 7] * 4, [[0] * 8] * 4, n_boot=1000, seed=0)
        self.assertTrue(out["valid"])
        self.assertAlmostEqual(out["mean"], 0.125)
        self.assertGreater(out["seed_lo"], 0.0)
        self.assertLessEqual(out["item_lo"], 0.0)
        self.assertLessEqual(out["lo"], 0.0)
        self.assertEqual(out["dominant_axis"], "item")

    def test_same_seed_is_deterministic(self):
        first = crossed_cluster_bootstrap_ci(
            [[1, 0, 1, 0], [0, 1, 0, 1]], [[0, 0, 0, 0], [1, 1, 1, 1]],
            n_boot=200, seed=3)
        second = crossed_cluster_bootstrap_ci(
            [[1, 0, 1, 0], [0, 1, 0, 1]], [[0, 0, 0, 0], [1, 1, 1, 1]],
            n_boot=200, seed=3)
        self.assertEqual(first["lo"], second["lo"])
        self.assertEqual(first["hi"], second["hi"])

    def test_malformed_or_unestimable_input_abstains(self):
        cases = [
            ([], []),
            ([1, 0, 1, 0], [0, 0, 1, 0]),          # one seed
            ([[1], [1]], [[0], [0]]),               # one item
            ([[1, 0], [1, 0]], [[0, 1]]),          # shape mismatch
            ([[1, 0], [1, 0]], [[0, 1], [0, 0.5]]),  # non-binary
        ]
        for a, b in cases:
            with self.subTest(a=a, b=b):
                self.assertFalse(crossed_cluster_bootstrap_ci(a, b)["valid"])
        for kwargs in ({"n_boot": 0}, {"n_boot": 2.5}, {"alpha": 2.0},
                       {"alpha": 0.0}):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(crossed_cluster_bootstrap_ci(
                    [[1, 0], [1, 0]], [[0, 1], [0, 1]],
                    **kwargs)["valid"])


class CorrectMaskPaddingTest(unittest.TestCase):
    def test_nonzero_padding_bits_are_rejected(self):
        # Three recorded items occupy the top three bits; the remaining
        # five bits are padding and must be zero.
        encoded = base64.b64encode(bytes([0b11100001])).decode("ascii")
        with self.assertRaises(ValueError):
            decode_correct_mask(encoded, 3)
        decoded = decode_correct_mask(
            encode_correct_mask([True, True, False, True]), 4)
        self.assertEqual(decoded.tolist(), [True, True, False, True])

    def test_zero_padding_round_trips(self):
        for n in (0, 1, 2, 7, 8, 9, 16):
            with self.subTest(n=n):
                expected = [bool(i % 3 == 0) for i in range(n)]
                decoded = decode_correct_mask(
                    encode_correct_mask(expected), n)
                self.assertEqual(decoded.tolist(), expected)


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

    def test_crossed_bootstrap_flags_a_single_item_effect(self):
        # Four seeds all gain exactly one item and lose none.  The seed-level
        # CI is therefore [1/8, 1/8] and excludes 0, while resampling items
        # shows that the gain does not generalise beyond this stream.
        n_items = 8
        a = [[1] + [0] * (n_items - 1) for _ in range(4)]
        b = [[0] * n_items for _ in range(4)]
        by_a, by_b = self._records(a, b)
        for seed in range(4):
            by_a[seed]["accuracy"] = 1.0 / n_items
            by_b[seed]["accuracy"] = 0.0
        paired = _paired_diff(list(by_a.values()), list(by_b.values()),
                              "accuracy")
        crossed = paired["stream_test"]["crossed_bootstrap"]
        self.assertTrue(crossed["valid"])
        self.assertTrue(crossed["seed_ci_excludes_zero"])
        self.assertTrue(crossed["stream_limited"])
        self.assertFalse(crossed["decision_agrees"])

    def test_paired_diff_abstains_on_malformed_training_seeds(self):
        for bad in ("not-a-record", None, 3, object(),
                    {"seed": [0], "accuracy": 1.0},
                    {"seed": True, "accuracy": 1.0},
                    {"seed": 0.5, "accuracy": 1.0}):
            with self.subTest(record=bad):
                paired = _paired_diff(
                    [bad], [{"seed": 0, "accuracy": 0.0}], "accuracy")
                self.assertEqual(paired["n"], 0)
                self.assertFalse(paired["controlled"]["controlled"])
                self.assertGreaterEqual(len(paired["seeds_unpaired"]), 1)

    def test_one_sample_contrast_drops_malformed_runs(self):
        for bad in ("not-a-record", {"seed": [0], "accuracy": 1.0},
                    {"seed": True, "accuracy": 1.0},
                    {"seed": 0.5, "accuracy": 1.0}):
            with self.subTest(record=bad):
                ci = _one_sample_contrast([bad], "accuracy")
                self.assertEqual(ci["n"], 0)
                self.assertTrue(ci["seeds_dropped"])

    def test_one_sample_contrast_flags_aliased_training_streams(self):
        runs = [
            {"seed": 0, "accuracy": 0.5},
            {"seed": 1, "accuracy": 0.6},
            {"seed": 2 ** 32, "accuracy": 0.7},
        ]
        ci = _one_sample_contrast(runs, "accuracy")
        self.assertEqual(ci["n"], 3)
        self.assertEqual(ci["seed_aliases"], [0, 2 ** 32])
        self.assertFalse(ci["controlled"]["controlled"])

    def test_paired_did_drops_malformed_runs(self):
        good = {"seed": 0, "accuracy": 1.0}
        for bad in ("not-a-record", {"seed": [0], "accuracy": 1.0},
                    {"seed": True, "accuracy": 1.0},
                    {"seed": 0.5, "accuracy": 1.0}):
            with self.subTest(record=bad):
                ci = _paired_did([bad], [good], [good], [good],
                                 "accuracy")
                self.assertEqual(ci["n"], 0)
                self.assertFalse(ci["controlled"]["controlled"])

    def test_factorial_and_grid_effects_drop_malformed_runs(self):
        good = {"seed": 0, "accuracy": 1.0}
        for bad in ("not-a-record", None, 3, object(),
                    {"seed": [0], "accuracy": 1.0}):
            with self.subTest(record=bad):
                cells = {cell: [good] for cell in
                         ("full", "calib_len_only", "calib_ent_only",
                          "calib_no_terms")}
                cells["full"] = [bad]
                ci = _factorial_main_effect(cells, "accuracy", "length")
                self.assertEqual(ci["n"], 0)
                self.assertFalse(ci["controlled"]["controlled"])

                grid = _grid_level_main_effect(
                    "accuracy", [[bad]], [[good]])
                self.assertEqual(grid["n"], 0)
                self.assertFalse(grid["controlled"]["controlled"])

    def test_factorial_and_grid_materialize_generator_cells_once(self):
        good = {"seed": 0, "accuracy": 1.0}
        cells = {cell: (r for r in [good]) for cell in
                 ("full", "calib_len_only", "calib_ent_only",
                  "calib_no_terms")}
        ci = _factorial_main_effect(cells, "accuracy", "length")
        self.assertEqual(ci["n"], 1)
        self.assertAlmostEqual(ci["mean"], 0.0)
        grid = _grid_level_main_effect(
            "accuracy", ((r for r in [good]),),
            ((r for r in [good]),))
        self.assertEqual(grid["n"], 1)
        self.assertAlmostEqual(grid["mean"], 0.0)
        with self.assertRaises(ValueError):
            _factorial_main_effect(None, "accuracy", "length")
        fresh = {cell: [good] for cell in
                 ("full", "calib_len_only", "calib_ent_only",
                  "calib_no_terms")}
        with self.assertRaises(ValueError):
            _factorial_main_effect(fresh, "accuracy", "unknown")

    def test_curve_helpers_skip_non_mapping_runs(self):
        good = {"seed": 0, "accuracy": 1.0, "curve_verdict": "ok"}
        for bad in ("not-a-record", None, 3, object()):
            with self.subTest(record=bad):
                ci = _paired_benchmark([bad], [good], "accuracy", 0.01,
                                       "baseline")
                self.assertEqual(ci["n"], 0)
                self.assertEqual(_curve_valid_runs([bad, good]), [good])
                self.assertEqual(_curve_health([bad, good]), ({"ok": 1}, 0))
                self.assertEqual(_curve_warn(("a", [bad])), "")

    def test_curve_helpers_treat_non_string_verdicts_as_malformed(self):
        good = {"seed": 0, "curve_verdict": "ok"}
        bad = {"seed": 1, "curve_verdict": ["plateau"]}
        self.assertEqual(_curve_valid_runs([bad, good]), [good])
        self.assertEqual(_curve_health([bad, good]),
                         ({"malformed": 1, "ok": 1}, 1))
        self.assertIn("malformed", _curve_warn(("a", [bad])))

    def test_curve_helpers_treat_non_iterable_runs_as_empty(self):
        self.assertEqual(_curve_valid_runs(None), [])
        self.assertEqual(_curve_health(None), ({}, 0))
        self.assertEqual(_curve_warn(("a", None)), "")

    def test_curve_helpers_materialize_a_generator_once(self):
        runs = ({"seed": seed, "curve_verdict": "ok"} for seed in (0, 1))
        self.assertEqual(len(_curve_valid_runs(runs)), 2)
        self.assertEqual(_curve_health(iter([{"curve_verdict": "ok"}])),
                         ({"ok": 1}, 0))
        self.assertEqual(
            _curve_warn(("a", iter([{"curve_verdict": "plateau"}]))),
            "WARN a: 1/1 curve-invalid (plateau*1)")

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

    def test_blank_or_non_hex_digest_is_rejected_not_verified(self):
        a = [[1] * 16]
        b = [[0] * 16]
        for bad in ("", "   ", "not-a-digest", "abc-123", "xyz"):
            with self.subTest(digest=repr(bad)):
                by_a, by_b = self._records(a, b)
                by_a[0]["stream_digest"] = bad
                by_b[0]["stream_digest"] = bad
                out = _paired_stream_test(by_a, by_b, [0], "accuracy", 1.0)
                self.assertFalse(out["applicable"])
                self.assertIn("digest", out["reason"])

    def test_hex_digest_still_verifies(self):
        a = [[1] * 16]
        b = [[0] * 16]
        by_a, by_b = self._records(a, b, digest="ABC123")
        out = _paired_stream_test(by_a, by_b, [0], "accuracy", 1.0)
        self.assertTrue(out["applicable"])
        self.assertTrue(out["digest_verified"])
        self.assertEqual(out["stream_digest"], "ABC123")

    def test_malformed_run_record_abstains(self):
        a = [[1] * 16]
        b = [[0] * 16]
        by_a, by_b = self._records(a, b)
        out = _paired_stream_test({0: "bad-record"}, by_b, [0],
                                  "accuracy", 1.0)
        self.assertFalse(out["applicable"])
        self.assertIn("malformed", out["reason"])
        out = _paired_stream_test(by_a, {0: None}, [0], "accuracy", 1.0)
        self.assertFalse(out["applicable"])
        self.assertIn("missing", out["reason"])
        out = _paired_stream_test([], {}, [0], "accuracy", 1.0)
        self.assertFalse(out["applicable"])
        self.assertIn("mapping", out["reason"])

    def test_non_accuracy_metric_still_abstains(self):
        a = [[1] * 16]
        b = [[0] * 16]
        by_a, by_b = self._records(a, b)
        out = _paired_stream_test(by_a, by_b, [0], "loss", 1.0)
        self.assertFalse(out["applicable"])
        self.assertIn("accuracy", out["reason"])


if __name__ == "__main__":
    unittest.main()
