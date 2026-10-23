"""Dependence-robust multiplicity companions.

The pre-registered family uses Benjamini-Hochberg, whose FDR guarantee
assumes independent or positively dependent p-values.  Contrasts in one
experiment share training seeds and a sealed evaluation stream, so these
tests pin the BY (arbitrary dependence) and Holm (family-wise error)
companions and their integration into ``ContrastLog.finalize``.  The
companions annotate the BH decision; they never replace it.
"""

from __future__ import annotations

import contextlib
import io
import math
import pathlib
import tempfile
import unittest

from cage.experiments import ContrastLog
from cage.metrics import (
    benjamini_hochberg,
    benjamini_yekutieli,
    holm_bonferroni,
    paired_bootstrap_ci,
    simes_test,
)


class MultiplicityProcedureTest(unittest.TestCase):
    def test_by_uses_the_harmonic_dependence_factor(self):
        p_values = [0.001, 0.02, 0.03, 0.04]
        bh = benjamini_hochberg(p_values, q=0.05)
        by = benjamini_yekutieli(p_values, q=0.05)
        factor = 1.0 + 1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0
        self.assertAlmostEqual(by[0]["p_adj"], 0.004 * factor)
        self.assertAlmostEqual(by[1]["p_adj"], 0.04 * factor)
        for bh_row, by_row in zip(bh, by):
            self.assertGreaterEqual(by_row["p_adj"], bh_row["p_adj"])
        self.assertEqual([r["reject"] for r in bh], [True] * 4)
        self.assertEqual([r["reject"] for r in by],
                         [True, False, False, False])

    def test_holm_step_down_rejects_only_while_each_null_is_cleared(self):
        holm = holm_bonferroni([0.001, 0.02, 0.03, 0.04], q=0.05)
        self.assertAlmostEqual(holm[0]["p_adj"], 0.004)
        self.assertAlmostEqual(holm[1]["p_adj"], 0.06)
        self.assertEqual([r["reject"] for r in holm],
                         [True, False, False, False])

    def test_nan_and_invalid_level_abstain(self):
        by = benjamini_yekutieli(
            [0.001, float("nan"), None], q=0.05)
        self.assertFalse(by[1]["reject"])
        self.assertTrue(math.isnan(by[1]["p_adj"]))
        self.assertFalse(by[2]["reject"])
        self.assertTrue(math.isnan(by[2]["p_adj"]))
        for procedure in (benjamini_yekutieli, holm_bonferroni):
            with self.subTest(procedure=procedure.__name__):
                rows = procedure([0.001, 0.02], q=True)
                self.assertFalse(any(r["reject"] for r in rows))
                self.assertTrue(all(math.isfinite(r["p_adj"])
                                    for r in rows))

    def test_simes_omnibus_is_the_minimum_ranked_bound(self):
        simes = simes_test([0.001, 0.02, 0.03, 0.04], alpha=0.05)
        self.assertTrue(simes["valid"])
        self.assertTrue(simes["reject"])
        self.assertAlmostEqual(simes["p"], 0.004)
        null = simes_test([0.5, 0.6], alpha=0.05)
        self.assertTrue(null["valid"])
        self.assertFalse(null["reject"])
        self.assertAlmostEqual(null["p"], 0.6)
        self.assertFalse(simes_test([])["valid"])
        self.assertFalse(simes_test([0.01], alpha=2.0)["valid"])


class ContrastDependenceCompanionTest(unittest.TestCase):
    @staticmethod
    def _ci(diffs):
        ci = paired_bootstrap_ci(diffs)
        ci["controlled"] = {
            "controlled": True,
            "mismatches": [],
            "varying_factors": {"gate": {"a": ["calib"], "b": ["query"]}},
            "capacity": {"matched": True, "a": 10, "b": 10,
                         "delta_b_minus_a": 0},
        }
        return ci

    def _finalize(self, p_values):
        log = ContrastLog("e1", hypotheses=("H1a",), min_seeds=3)
        diffs = [0.20, 0.19, 0.22, 0.18, 0.21]
        with contextlib.redirect_stdout(io.StringIO()):
            for i, p in enumerate(p_values):
                ci = self._ci(diffs)
                ci["p_value"] = p
                log.print(f"row {i}", "H1a", "abs_gate_vs_entropy", ci)
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                payload = log.finalize(pathlib.Path(tmp), q=0.05)
        return payload, captured.getvalue()

    def test_bh_claims_are_flagged_when_dependence_robust_models_dissent(self):
        payload, output = self._finalize([0.001, 0.02, 0.03, 0.04])
        rows = payload["rows"]
        self.assertTrue(all(row["supports_hypothesis"] for row in rows))
        self.assertTrue(rows[0]["by_supports"])
        self.assertTrue(rows[0]["holm_supports"])
        self.assertFalse(any(row["by_supports"] for row in rows[1:]))
        self.assertFalse(any(row["holm_supports"] for row in rows[1:]))
        self.assertTrue(all(row["by_disagrees"] for row in rows[1:]))
        self.assertTrue(all(row["holm_disagrees"] for row in rows[1:]))
        self.assertEqual(payload["n_by_supporting"], 1)
        self.assertEqual(payload["n_holm_supporting"], 1)
        self.assertEqual(payload["n_by_disagree"], 3)
        self.assertEqual(payload["n_holm_disagree"], 3)
        self.assertEqual(payload["n_dependence_weaker"], 3)
        self.assertAlmostEqual(payload["simes_family_p"], 0.004)
        self.assertTrue(payload["simes_family_reject"])
        self.assertEqual(len(payload["dependence_weaker_contrasts"]), 3)
        self.assertIn("multiplicity-model sensitive", output)
        self.assertIn("pre-registered BH decision unchanged", output)

    def test_agreement_leaves_no_conflict(self):
        payload, _ = self._finalize([0.001, 0.002, 0.003, 0.004])
        self.assertEqual(payload["n_by_disagree"], 0)
        self.assertEqual(payload["n_holm_disagree"], 0)
        self.assertEqual(payload["n_dependence_weaker"], 0)


if __name__ == "__main__":
    unittest.main()
