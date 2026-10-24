"""Run the package's deterministic scientific self-checks under unittest."""

from __future__ import annotations

import contextlib
import io
import math
import unittest

from cage import data, metrics


class MetricInputValidationTest(unittest.TestCase):
    def test_calibration_metrics_preserve_float64_precision(self):
        self.assertAlmostEqual(metrics.brier_score([0.9], [0]), 0.81,
                               places=12)
        self.assertAlmostEqual(
            metrics.expected_calibration_error([0.9], [0]), 0.9,
            places=12)

    def test_non_binary_correctness_abstains(self):
        self.assertTrue(math.isnan(
            metrics.expected_calibration_error([0.5], [0.5])))
        self.assertTrue(math.isnan(
            metrics.expected_calibration_error([0.5], [2.0])))
        self.assertTrue(math.isnan(metrics.brier_score([0.5], [0.5])))
        self.assertTrue(math.isnan(metrics.brier_score([0.5], [-1.0])))

    def test_near_binary_labels_do_not_round_through_float32(self):
        # 0.9999999999999999 and 1.0000000001 are not exactly 0/1; the
        # default float32 capture would round both to an accepted label.
        for bad in ([0.9999999999999999], [1.0000000001]):
            with self.subTest(bad=bad):
                self.assertIsNone(metrics._binary_labels(bad))
                self.assertFalse(metrics.evaluation_fold_audit(
                    bad * 8, [0.0] * 8)["valid"])
                self.assertFalse(metrics.crossed_cluster_bootstrap_ci(
                    [bad * 8, bad * 8], [[0.0] * 8, [0.0] * 8],
                    n_boot=10)["valid"])

    def test_fractional_multiclass_targets_abstain(self):
        import torch

        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        for targets in ([0.5, 1.0], [1.9, 0.0], [-0.5, 0.0],
                        [float("nan"), 0.0]):
            with self.subTest(targets=targets):
                out = metrics.multiclass_metrics(logits, targets)
                self.assertEqual(out["n"], 0)
                self.assertTrue(math.isnan(out["accuracy"]))
        # Integral floating-point labels are a valid JSON round-trip of the
        # token ids and must still be scored.
        out = metrics.multiclass_metrics(logits, [1.0, 0.0])
        self.assertEqual(out["n"], 2)
        self.assertTrue(math.isfinite(out["accuracy"]))

    def test_boolean_multiclass_targets_abstain(self):
        import torch

        # A boolean is a correctness flag, not a class id; casting it to
        # long would silently invent the labels 1/0.
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        out = metrics.multiclass_metrics(logits,
                                         torch.tensor([True, False]))
        self.assertEqual(out["n"], 0)
        self.assertTrue(math.isnan(out["accuracy"]))

    def test_python_float_logits_keep_their_precision(self):
        # A Python list is captured as float32 by ``as_tensor`` by default;
        # that would turn this near-tie into an exact tie and flip accuracy.
        out = metrics.multiclass_metrics([[1.0, 1.0 + 1e-9]], [1])
        self.assertEqual(out["n"], 1)
        self.assertEqual(out["accuracy"], 1.0)

    def test_fractional_float_targets_are_not_rounded_to_integers(self):
        # 0.99999999 is not a class id.  A float32 capture would round it to
        # 1.0 and silently accept it.
        out = metrics.multiclass_metrics(
            [[2.0, 0.0], [0.0, 2.0]], [0.99999999, 1.0])
        self.assertEqual(out["n"], 0)
        self.assertTrue(math.isnan(out["accuracy"]))

    def test_boolean_logits_abstain(self):
        import torch

        out = metrics.multiclass_metrics(
            torch.tensor([[True, False]]), [0])
        self.assertEqual(out["n"], 0)
        self.assertTrue(math.isnan(out["accuracy"]))

    def test_complex_multiclass_logits_abstain(self):
        import torch

        # ``argmax`` has no complex ordering; the reporting metric must
        # abstain instead of raising (or scoring the real projection).
        logits = torch.tensor([[1 + 2j, 0j], [0j, 1 + 1j]])
        out = metrics.multiclass_metrics(logits, [0, 1])
        self.assertEqual(out["n"], 0)
        self.assertTrue(math.isnan(out["accuracy"]))
        self.assertTrue(math.isnan(out["nll"]))

    def test_multiclass_abstention_keeps_the_scored_key_schema(self):
        import torch

        scored = metrics.multiclass_metrics(
            torch.tensor([[0.0, 1.0], [1.0, 0.0]]), [1, 0])
        abstained = metrics.multiclass_metrics(None, [0, 1])
        # The top-k key is named after the clamped class count in the scored
        # panel and after the requested k when there is no class count; every
        # other metric key is still part of the fixed schema.
        scored_keys = {key for key in scored if not key.endswith("_accuracy")
                       or key == "accuracy"}
        abstained_keys = {key for key in abstained
                          if not key.endswith("_accuracy")
                          or key == "accuracy"}
        self.assertEqual(scored_keys, abstained_keys)
        for key in ("f1_macro_all_labels", "precision_macro_all_labels",
                    "recall_macro_all_labels"):
            self.assertTrue(math.isnan(abstained[key]))

    def test_malformed_multiclass_containers_abstain(self):
        import torch

        for logits, targets in ((None, [0]), ("x", [0]),
                                (object(), [0]),
                                (torch.zeros(2, 2, 2), [0, 1])):
            with self.subTest(logits=type(logits).__name__,
                              targets=targets):
                out = metrics.multiclass_metrics(logits, targets)
                self.assertEqual(out["n"], 0)
                self.assertTrue(math.isnan(out["accuracy"]))
                self.assertTrue(math.isnan(out["nll"]))

    def test_malformed_calibration_containers_abstain(self):
        for probs, correct in ((None, [1.0]), ("x", [1.0]),
                               ([0.5], object()),
                               (object(), [1.0])):
            with self.subTest(probs=probs, correct=correct):
                self.assertTrue(math.isnan(
                    metrics.expected_calibration_error(probs, correct)))
                self.assertTrue(math.isnan(
                    metrics.brier_score(probs, correct)))

    def test_complex_metric_samples_are_rejected_not_truncated(self):
        import torch

        # ``Tensor.to(float64)`` silently discards the imaginary part with
        # only a warning; a complex paired difference must not be read as
        # its real projection.
        self.assertEqual(
            metrics.average_ranks(torch.tensor([1.0 + 2.0j,
                                                3.0 + 4.0j])).numel(), 0)
        ci = metrics.paired_bootstrap_ci(
            torch.tensor([0.1 + 0.1j, 0.2 + 0j]))
        self.assertEqual(ci["n"], 0)
        self.assertTrue(math.isnan(ci["mean"]))
        # Companions that document an abstention instead of raising still
        # abstain rather than silently truncating.
        self.assertFalse(metrics.sign_flip_test(
            torch.tensor([0.1 + 0.1j, 0.2 + 0j]))["valid"])
        self.assertFalse(metrics.replication_interval(
            torch.tensor([0.1 + 0.1j, 0.2 + 0j, 0.3 + 0j]))["valid"])

    def test_complex_calibration_and_count_inputs_abstain(self):
        import torch

        # ``.float()`` would silently drop the imaginary part, inventing a
        # calibration sample; both helpers must abstain instead.
        self.assertTrue(math.isnan(
            metrics.expected_calibration_error([0.5 + 0.1j], [1.0])))
        self.assertTrue(math.isnan(
            metrics.expected_calibration_error([0.5], [1.0 + 0.0j])))
        self.assertTrue(math.isnan(
            metrics.brier_score([0.5 + 0.1j], [1.0])))
        self.assertTrue(math.isnan(
            metrics.brier_score([0.5], [1.0 + 0.0j])))
        # Counts are real integers, never complex/bool flags.
        for successes, totals in (([1 + 0j], [2 + 0j]),
                                  (True, 2), ([True], [2])):
            with self.subTest(successes=successes, totals=totals):
                out = metrics.clustered_proportion_ci(successes, totals)
                self.assertFalse(out["valid"])
                self.assertTrue(out["reason"])

    def test_wilson_and_bh_arguments_are_validated(self):
        # A boolean is a flag, not a success count or a z quantile; a NaN
        # z would silently collapse the interval to its point estimate.
        for kwargs in ({"successes": True, "total": 2},
                       {"successes": 1, "total": True}):
            out = metrics.wilson_ci(**kwargs)
            self.assertTrue(math.isnan(out["lo"]))
            self.assertTrue(math.isnan(out["hi"]))
        for z in (float("nan"), 0.0, -1.0, True, 1 + 0j):
            out = metrics.wilson_ci(1, 2, z=z)
            self.assertTrue(math.isnan(out["p"]))
            self.assertTrue(math.isnan(out["lo"]))
        # An invalid FDR level must not turn ``True`` into a 1.0 threshold
        # that rejects the whole family; malformed p-values are untestable.
        bad_level = metrics.benjamini_hochberg([0.01, 0.2], q=True)
        self.assertFalse(any(row["reject"] for row in bad_level))
        mixed = metrics.benjamini_hochberg([0.01, 0.2 + 1j, "x", None], q=0.05)
        self.assertTrue(mixed[0]["reject"])
        for row in mixed[1:]:
            self.assertFalse(row["reject"])
            self.assertTrue(math.isnan(row["p"]))
        # A p-value outside [0, 1] is malformed, not an extreme test
        # result.  Clamping a negative value to zero would fabricate the
        # strongest possible rejection.
        outside = metrics.benjamini_hochberg([-0.1, 1.5, 0.01], q=0.05)
        self.assertFalse(outside[0]["reject"])
        self.assertTrue(math.isnan(outside[0]["p"]))
        self.assertFalse(outside[1]["reject"])
        self.assertTrue(math.isnan(outside[1]["p"]))
        self.assertTrue(outside[2]["reject"])
        # A string/bytes/mapping is iterable but does not name p-values;
        # list() would silently turn it into characters/keys/byte values.
        for bad in (0.1, "0.05", b"\x01", {"p": 0.05}, object()):
            with self.subTest(p_values=bad):
                with self.assertRaises(ValueError):
                    metrics.benjamini_hochberg(bad)

    def test_out_of_range_selection_fraction_abstains(self):
        for frac in (-0.1, 1.5, float("inf")):
            with self.subTest(frac=frac):
                out = metrics.selection_at_fraction(
                    [1.0, 2.0, 3.0], [False, True, True], frac=frac)
                self.assertEqual(out["k"], 0)
                self.assertTrue(math.isnan(out["frac"]))
                self.assertTrue(math.isnan(out["precision"]))

    def test_tiny_positive_selection_fraction_still_flags_one_row(self):
        # Rounding the product before ``ceil`` used to turn any genuinely
        # positive product below 5e-10 into zero, contradicting the
        # documented ``ceil(frac * n)`` rule.
        out = metrics.selection_at_fraction(
            [1.0, 2.0, 3.0], [False, True, True], frac=1e-10)
        self.assertEqual(out["k"], 1)
        self.assertAlmostEqual(out["frac"], 1.0 / 3.0)
        self.assertTrue(math.isfinite(out["precision"]))
        # The floating-point-noise case the snap exists for still works.
        exact = metrics.selection_at_fraction(
            list(range(100)), [1] * 100, frac=0.07)
        self.assertEqual(exact["k"], 7)
        # A genuine excess above an integer is not floating-point noise:
        # 0.2000000001 * 5 == 1.0000000005, so ceil() must flag two rows.
        over = metrics.selection_at_fraction(
            list(range(5)), [0] * 5, frac=0.2000000001)
        self.assertEqual(over["k"], 2)
        self.assertAlmostEqual(over["frac"], 0.4)

    def test_max_exact_is_capped_not_an_abstention(self):
        # The docstring says the threshold is capped at 20; a caller asking
        # for more should still get the exact test for small samples, not a
        # silently invalid result.
        result = metrics.sign_flip_test(
            [1.0, 2.0, 3.0], max_exact=21)
        self.assertTrue(result["valid"], result["reason"])
        self.assertEqual(result["method"], "exact")
        self.assertEqual(result["n_permutations"], 8)

    def test_paired_auc_companions_always_carry_the_item_count(self):
        for args in (([], [], []),
                     ([1.0, 2.0], [1.0], [0, 1]),
                     ([1.0, float("inf")], [1.0, 2.0], [0, 1])):
            with self.subTest(args=args):
                out = metrics.auc_paired_bootstrap_ci(*args, n_boot=20)
                self.assertIn("n", out)
                self.assertTrue(math.isnan(out["mean_diff"]))
        # An invalid bootstrap count is still a measurable pair of panels.
        out = metrics.auc_paired_bootstrap_ci(
            [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [0, 1, 1],
            n_boot=float("nan"))
        self.assertEqual(out["n"], 3)
        self.assertTrue(math.isnan(out["lo"]))

    def test_zero_leave_one_out_mean_is_not_a_sign_flip(self):
        ci = metrics.paired_bootstrap_ci([1.0, -1.0, 2.0], n_boot=200)
        self.assertGreater(ci["mean"], 0.0)
        self.assertEqual(ci["loo_mean_min"], 0.0)
        self.assertFalse(ci["loo_sign_flip"])

    def test_malformed_cluster_inputs_abstain(self):
        for successes, totals in ((["not-a-count"], [1]),
                                  ([1], ["not-a-count"]),
                                  ({"not": "a-count"}, [1])):
            with self.subTest(successes=successes, totals=totals):
                out = metrics.clustered_proportion_ci(successes, totals)
                self.assertFalse(out["valid"])
                self.assertTrue(out["reason"])


class PositiveIntegerValidationTest(unittest.TestCase):
    def test_large_counts_are_exact_or_rejected_not_truncated(self):
        from cage.metrics import _positive_integer

        exact = 2 ** 53 + 1
        self.assertEqual(_positive_integer(exact), exact)
        self.assertEqual(_positive_integer(str(exact)), exact)
        self.assertIsNone(_positive_integer(10 ** 400))
        self.assertIsNone(_positive_integer(True))
        self.assertIsNone(_positive_integer(2.5))
        self.assertEqual(_positive_integer("10.0"), 10)
        exact = 2 ** 53 + 1
        self.assertEqual(metrics.wilson_ci(exact, exact)["n"], exact)


class BootstrapReplicateCountTest(unittest.TestCase):
    def test_large_replicate_counts_are_not_rounded_through_float(self):
        import torch
        from unittest import mock

        from cage import metrics

        exact = 2 ** 53 + 1
        seen = []

        def fake_means(diffs, n_boot, seed):
            seen.append(n_boot)
            return torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)

        calls = (
            lambda: metrics.paired_bootstrap_ci(
                [0.1, 0.2, 0.3], n_boot=exact),
            lambda: metrics.equivalence_test(
                [0.1, 0.2, 0.3], 0.5, n_boot=exact),
            lambda: metrics.noninferiority_test(
                [0.1, 0.2, 0.3], 0.5, n_boot=exact),
            lambda: metrics.minimum_effect_test(
                [0.1, 0.2, 0.3], 0.01, n_boot=exact),
            lambda: metrics.bootstrap_upper_p(
                [0.1, 0.2, 0.3], 0.5, n_boot=exact),
        )
        with mock.patch.object(metrics, "_bootstrap_means", fake_means):
            for call in calls:
                call()
        self.assertEqual(seen, [exact] * len(calls))
        # An unrepresentable count must abstain before sizing a loop.
        self.assertIsNone(metrics._positive_integer(10 ** 400))
        out = metrics.paired_bootstrap_ci(
            [0.1, 0.2, 0.3], n_boot=10 ** 400)
        self.assertTrue(math.isnan(out["lo"]))
        # The independent two-sample companion has the same contract.
        from cage.train import _two_sample_bootstrap_ci

        shapes = []

        def fake_randint(low, high, size, generator=None, device=None):
            shapes.append(tuple(size))
            return torch.zeros((2, size[1]), dtype=torch.long)

        with mock.patch("cage.train.torch.randint",
                        side_effect=fake_randint):
            _two_sample_bootstrap_ci(
                [1.0, 2.0, 3.0], [0.0, 1.0, 2.0], n_boot=exact)
        self.assertEqual(shapes, [(exact, 3), (exact, 3)])


class TinyAlphaQuantileTest(unittest.TestCase):
    def test_tiny_alpha_does_not_raise_from_normal_quantile(self):
        from cage.metrics import (
            paired_bootstrap_ci,
            replication_interval,
            seeds_for_power,
        )

        self.assertTrue(math.isfinite(
            seeds_for_power(1.0, 0.5, alpha=1e-16)))
        ci = paired_bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0],
                                 n_boot=20, alpha=1e-16)
        self.assertEqual(ci["n"], 5)
        # The tiny alpha may put the requested quantile beyond the finite
        # bootstrap grid, but it must be an abstention, not an exception.
        self.assertTrue(math.isfinite(ci["lo"]) or math.isnan(ci["lo"]))
        rep = replication_interval([0.1, 0.2, 0.3, 0.4],
                                   alpha=1e-16)
        self.assertTrue(isinstance(rep["valid"], bool))


class FixedBatchDtypeTest(unittest.TestCase):
    def test_narrow_integer_fixed_batch_is_widened(self):
        import torch

        from cage.data import make_batch
        from cage.model import build
        from cage.train import train_model

        idx, tgt, meta = make_batch(
            2, 3, 300, "cpu", gen=torch.Generator().manual_seed(1))
        model = build({
            "vocab_size": 300, "d_model": 16, "n_heads": 2,
            "n_layers": 1, "max_len": 32, "gate": "none",
        })
        hist = train_model(
            model, steps=1, batch=2, n_pairs=3, vocab=300, seed=None,
            fixed_batch=(idx.to(torch.int16), tgt.to(torch.int32), meta),
            log_every=1)
        self.assertEqual(len(hist), 1)


class AnalyticFlopsValidationTest(unittest.TestCase):
    def test_head_tokens_cannot_exceed_sequence_length(self):
        from cage.efficiency import analytic_matmul_flops
        from cage.model import Cfg

        cfg = Cfg(vocab_size=8, d_model=4, n_heads=1, n_layers=1,
                  max_len=8)
        with self.assertRaises(ValueError):
            analytic_matmul_flops(cfg, 4, head_tokens=5)


class TrainingFlagValidationTest(unittest.TestCase):
    def test_truthy_non_boolean_training_flags_are_rejected(self):
        from cage.train import train_model

        for kwargs in ({"dense_lm": "False"},
                       {"permute_labels": "False"},
                       {"restore_best": "False"},
                       {"disable_regularization": 0},
                       {"batch_metrics": None}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    train_model(None, steps=1, **kwargs)


class BootstrapSeedValidationTest(unittest.TestCase):
    def test_fractional_or_boolean_resampling_seeds_are_rejected(self):
        from cage.baselines import probe_vs_chance_test
        from cage.metrics import (
            auc_paired_bootstrap_ci,
            bootstrap_upper_p,
            clustered_proportion_ci,
            crossed_cluster_bootstrap_ci,
            equivalence_test,
            minimum_effect_test,
            noninferiority_test,
            paired_bootstrap_ci,
            pr_auc_ci,
            roc_auc_ci,
        )
        from cage.train import _two_sample_bootstrap_ci

        calls = (
            lambda seed: roc_auc_ci([1.0, 2.0, 0.0], [0, 1, 0],
                                    n_boot=10, seed=seed),
            lambda seed: pr_auc_ci([1.0, 2.0, 0.0], [0, 1, 0],
                                   n_boot=10, seed=seed),
            lambda seed: auc_paired_bootstrap_ci(
                [1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0, 1, 0],
                n_boot=10, seed=seed),
            lambda seed: paired_bootstrap_ci([0.1, 0.2, 0.3],
                                             n_boot=10, seed=seed),
            lambda seed: equivalence_test([0.1, 0.2, 0.3], 0.5,
                                          n_boot=10, seed=seed),
            lambda seed: noninferiority_test([0.1, 0.2, 0.3], 0.5,
                                             n_boot=10, seed=seed),
            lambda seed: minimum_effect_test([0.1, 0.2, 0.3], 0.01,
                                             n_boot=10, seed=seed),
            lambda seed: bootstrap_upper_p([0.1, 0.2, 0.3], 0.5,
                                           n_boot=10, seed=seed),
            lambda seed: clustered_proportion_ci([1, 0], [2, 2],
                                                 n_boot=10, seed=seed),
            lambda seed: crossed_cluster_bootstrap_ci(
                [[1, 0], [0, 1]], [[0, 1], [1, 0]],
                n_boot=10, seed=seed),
            lambda seed: probe_vs_chance_test(
                [0.01, 0.02], 0.01, n_boot=10, seed=seed,
                n_per_seed=100),
            lambda seed: _two_sample_bootstrap_ci(
                [1.0, 2.0], [0.0, 1.0], n_boot=10, seed=seed),
        )
        for seed in (True, 0.5, 1.0, "1", None, object()):
            for index, call in enumerate(calls):
                with self.subTest(seed=seed, call=index):
                    with self.assertRaises(ValueError):
                        call(seed)


class NegativeControlPermutationTest(unittest.TestCase):
    def test_permuted_labels_never_keep_a_rows_own_answer(self):
        import torch

        from cage.train import _permute_batch_labels

        targets = torch.arange(32).view(8, 4)
        gen = torch.Generator().manual_seed(0)
        permuted = _permute_batch_labels(targets, gen)
        self.assertFalse(torch.equal(permuted, targets))
        for row in range(targets.size(0)):
            self.assertFalse(torch.equal(permuted[row], targets[row]),
                             f"row {row} kept its own labels")
        # The control is still a permutation of the same label rows.
        self.assertTrue(torch.equal(permuted.sort(dim=0).values,
                                    targets.sort(dim=0).values))

    def test_sparse_targets_keep_one_deranged_label_per_row(self):
        import torch

        from cage.train import _permute_batch_labels

        # The recall task stores one supervised column per row.  A row
        # shift would move that label to the donor's column, leaving the
        # receiving row's own query column at -1; the control must instead
        # move the label value and keep every row trainable.
        targets = torch.full((4, 4), -1, dtype=torch.long)
        columns = torch.arange(4)
        targets[torch.arange(4), columns] = torch.tensor([10, 11, 12, 13])
        permuted = _permute_batch_labels(targets,
                                         torch.Generator().manual_seed(0))
        self.assertTrue(bool(permuted.ne(-1).sum(dim=1).eq(1).all()))
        for row in range(4):
            self.assertNotEqual(int(permuted[row, columns[row]]),
                                int(targets[row, columns[row]]))

    def test_permutation_needs_a_partner_row(self):
        import torch

        from cage.train import _permute_batch_labels

        with self.assertRaises(ValueError):
            _permute_batch_labels(torch.arange(4).view(1, 4),
                                  torch.Generator().manual_seed(0))


class TwoSampleBootstrapValidationTest(unittest.TestCase):
    def test_malformed_groups_abstain_instead_of_raising(self):
        from cage.train import _two_sample_bootstrap_ci

        for left, right in (
                ([1.0 + 1.0j, 2.0], [0.0, 1.0]),
                (None, [0.0, 1.0]),
                ("not-a-sample", [0.0, 1.0]),
                ([0.0, 1.0], object())):
            with self.subTest(left=left, right=right):
                out = _two_sample_bootstrap_ci(left, right, n_boot=50)
                self.assertTrue(math.isnan(out["mean"]))
                self.assertTrue(math.isnan(out["lo"]))
                self.assertTrue(math.isnan(out["p_value"]))


class CorrelationInputValidationTest(unittest.TestCase):
    def test_complex_correlations_abstain(self):
        import torch

        from cage.train import pearson, spearman

        x = torch.tensor([1.0 + 2.0j, 2.0 + 3.0j, 3.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        for fn in (pearson, spearman):
            self.assertTrue(math.isnan(float(fn(x, y))))
            self.assertTrue(math.isnan(float(fn(y, x))))

    def test_nonfinite_correlations_abstain(self):
        import torch

        from cage.train import pearson, spearman

        y = torch.tensor([1.0, 2.0, 3.0])
        for bad in ([1.0, float("nan"), 3.0],
                    [1.0, float("inf"), 3.0],
                    [1.0, float("-inf"), 3.0]):
            with self.subTest(bad=bad):
                x = torch.tensor(bad)
                for fn in (pearson, spearman):
                    self.assertTrue(math.isnan(float(fn(x, y))))
                    self.assertTrue(math.isnan(float(fn(y, x))))


class CorrectMaskInputTest(unittest.TestCase):
    def test_malformed_mask_input_raises_value_error(self):
        from cage.train import encode_correct_mask

        for bad in (None, "x", object(),
                    [0.9999999999999999], [1.0000000001]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    encode_correct_mask(bad)


class BootstrapOverflowTest(unittest.TestCase):
    def test_unrepresentable_bootstrap_moments_abstain(self):
        from cage.metrics import (
            bootstrap_upper_p,
            equivalence_test,
            minimum_effect_test,
            noninferiority_test,
            paired_bootstrap_ci,
        )

        # Every input is finite, but the sum/SD overflows float64.  The
        # result must not present an infinite effect with a tiny p-value.
        huge = [1e308, 1e308, 1e308, 1e308]
        for test, keys in (
                (lambda: paired_bootstrap_ci(huge, n_boot=200),
                 ("mean", "lo", "hi", "p_value")),
                (lambda: equivalence_test(huge, 0.1, n_boot=200),
                 ("mean", "lo", "hi", "p_value")),
                (lambda: noninferiority_test(huge, 0.1, n_boot=200),
                 ("mean", "lo", "hi", "p_value")),
                (lambda: minimum_effect_test(huge, 0.1, n_boot=200),
                 ("mean", "lo", "hi", "p_value"))):
            out = test()
            for key in keys:
                self.assertTrue(math.isnan(out[key]), (key, out))
            self.assertFalse(out.get("equivalent", False))
            self.assertFalse(out.get("noninferior", False))
            self.assertFalse(out.get("beyond_sesoi", False))
        self.assertTrue(math.isnan(bootstrap_upper_p(huge, 0.1, n_boot=200)))
        rep = metrics.replication_interval(huge, direction="+")
        self.assertFalse(rep["valid"])
        self.assertTrue(math.isnan(rep["mean"]))
        self.assertTrue(math.isnan(rep["lo"]))
        # Mixed signs overflow resampled sums even though the full-sample
        # mean is zero and the full-sample SD is finite.
        mixed = [-1e308, 1e308, 1e308, -1e308]
        self.assertTrue(math.isnan(
            paired_bootstrap_ci(mixed, n_boot=200)["lo"]))
        # The exact sign-flip companion must abstain too: a finite sample
        # can overflow its own mean/sign-flip sums.
        for sample in (huge, mixed):
            with self.subTest(sample=sample):
                sf = metrics.sign_flip_test(sample)
                self.assertFalse(sf["valid"])
                self.assertTrue(math.isnan(sf["p_value"]))
        self.assertFalse(
            paired_bootstrap_ci(huge, n_boot=10)["sign_flip_valid"])
        cluster = metrics.clustered_proportion_ci(
            [1.5e308, 1.5e308], [1.5e308, 1.5e308], n_boot=3)
        self.assertFalse(cluster["valid"])
        self.assertIn("overflow", cluster["reason"])


class BaselineArgumentValidationTest(unittest.TestCase):
    def test_probe_chance_must_be_a_real_probability(self):
        from cage.baselines import probe_vs_chance_test

        for chance in (False, True, float("nan"), -0.1, 0.5, 1 + 0j):
            with self.subTest(chance=chance):
                with self.assertRaises(ValueError):
                    probe_vs_chance_test([0.01, 0.02], chance)

    def test_zero_between_seed_spread_reports_a_finite_power_plan(self):
        from cage.baselines import probe_vs_chance_test

        # Identical seed accuracies (zero between-seed SD) with a large
        # probe stream are certified; the planning companion must not call
        # that unattainable just because the seed term is zero.
        out = probe_vs_chance_test([0.001] * 5, 0.01, n_per_seed=1000)
        self.assertEqual(out["hb2_state"], "certified")
        self.assertTrue(math.isfinite(out["seeds_for_certification"]))
        self.assertLessEqual(out["seeds_for_certification"], 5)

    def test_malformed_probe_accuracies_abstain(self):
        from cage.baselines import probe_vs_chance_test

        for accuracies in (None, "not-a-sample", object(),
                           [0.1 + 0.1j, 0.2 + 0j],
                           [1.5, 1.6], [-0.5, 0.1], [0.1, 1.2]):
            with self.subTest(accuracies=accuracies):
                out = probe_vs_chance_test(accuracies, 0.01)
                self.assertEqual(out["n"], 0)
                self.assertEqual(out["hb2_state"], "untestable")
                self.assertFalse(out["hb2_supported"])
                # The shared-stream scoring draw must not clamp an
                # impossible accuracy into a fabricated interval.
                out = probe_vs_chance_test(
                    accuracies, 0.01, n_per_seed=100)
                self.assertEqual(out["n"], 0)
                self.assertEqual(out["hb2_state"], "untestable")
                self.assertFalse(out["hb2_supported"])
                self.assertTrue(math.isnan(out["lo"]))


    def test_malformed_probe_grids_and_seeds_raise_value_error(self):
        from cage.baselines import tune_linear_probe

        for kwargs in ({"lr_grid": 5}, {"lr_grid": b"\x01"},
                       {"lr_grid": []}, {"lr_grid": {"1e-3": 1}},
                       {"seeds": 5}, {"seeds": b"\x00"},
                       {"seeds": []}, {"seeds": {"0": 1}},
                       {"steps_grid": 5}, {"steps_grid": b"\x01"},
                       {"steps_grid": 0}, {"steps_grid": False},
                       {"steps_grid": ""},
                       {"steps_grid": {"10": 1}}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    tune_linear_probe(3, 16, **kwargs)


class ProjectSelfCheckTest(unittest.TestCase):
    def test_metrics_self_check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            failures = metrics._self_test()
        self.assertEqual(failures, 0, "cage.metrics self-test failed")

    def test_data_self_check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            failures = data._self_test()
        self.assertEqual(failures, 0, "cage.data self-test failed")


if __name__ == "__main__":
    unittest.main()
