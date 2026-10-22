"""Run the package's deterministic scientific self-checks under unittest."""

from __future__ import annotations

import contextlib
import io
import math
import unittest

from cage import data, metrics


class MetricInputValidationTest(unittest.TestCase):
    def test_non_binary_correctness_abstains(self):
        self.assertTrue(math.isnan(
            metrics.expected_calibration_error([0.5], [0.5])))
        self.assertTrue(math.isnan(
            metrics.expected_calibration_error([0.5], [2.0])))
        self.assertTrue(math.isnan(metrics.brier_score([0.5], [0.5])))
        self.assertTrue(math.isnan(metrics.brier_score([0.5], [-1.0])))

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

    def test_out_of_range_selection_fraction_abstains(self):
        for frac in (-0.1, 1.5, float("inf")):
            with self.subTest(frac=frac):
                out = metrics.selection_at_fraction(
                    [1.0, 2.0, 3.0], [False, True, True], frac=frac)
                self.assertEqual(out["k"], 0)
                self.assertTrue(math.isnan(out["frac"]))
                self.assertTrue(math.isnan(out["precision"]))

    def test_malformed_cluster_inputs_abstain(self):
        for successes, totals in ((["not-a-count"], [1]),
                                  ([1], ["not-a-count"]),
                                  ({"not": "a-count"}, [1])):
            with self.subTest(successes=successes, totals=totals):
                out = metrics.clustered_proportion_ci(successes, totals)
                self.assertFalse(out["valid"])
                self.assertTrue(out["reason"])


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


class BaselineArgumentValidationTest(unittest.TestCase):
    def test_probe_chance_must_be_a_real_probability(self):
        from cage.baselines import probe_vs_chance_test

        for chance in (False, True, float("nan"), -0.1, 0.5, 1 + 0j):
            with self.subTest(chance=chance):
                with self.assertRaises(ValueError):
                    probe_vs_chance_test([0.01, 0.02], chance)

    def test_malformed_probe_accuracies_abstain(self):
        from cage.baselines import probe_vs_chance_test

        for accuracies in (None, "not-a-sample", object(),
                           [0.1 + 0.1j, 0.2 + 0j]):
            with self.subTest(accuracies=accuracies):
                out = probe_vs_chance_test(accuracies, 0.01)
                self.assertEqual(out["n"], 0)
                self.assertEqual(out["hb2_state"], "untestable")
                self.assertFalse(out["hb2_supported"])


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
