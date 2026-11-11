"""Tests for the prospective decision-rule calibration.

The calibration measures the *registered* decision chain (bootstrap
p-value -> BH at q -> direction -> SESOI -> seed floor).  Two properties
of its report matter for a reader:

* the null scenario has no row with a true effect, so it must not report
  a target-claim rate - that field is the power of the claim gate on the
  row that carries the effect, and under the null the same number is a
  false-claim rate;
* an alternative scenario whose row was not shifted carries no true
  effect either, so booking that row as a true discovery would hide its
  claims from the false-discovery count and understate the family FDR.
"""

import unittest

from cage import metrics


def _calibration(**overrides):
    kwargs = dict(
        n_units=5, family_sizes=(1, 4), sd=1.0, sesoi=0.5, effect=0.5,
        q=0.05, direction="greater", n_replicates=120, n_boot=400,
        n_random=400, correlation=0.5, seed=0, min_seeds=3)
    kwargs.update(overrides)
    n_units = kwargs.pop("n_units")
    return metrics.decision_rule_calibration(n_units, **kwargs)


class CalibrationTargetRowTest(unittest.TestCase):

    def test_null_scenario_reports_no_target_claim_rate(self):
        cal = _calibration()
        for rule in cal["rules"]:
            for family in cal["rules"][rule]["null"]["families"].values():
                self.assertIsNone(family["target_claim_rate"])

    def test_alternative_scenario_reports_the_target_power(self):
        cal = _calibration()
        for rule in cal["rules"]:
            for family in cal["rules"][rule]["alternative"]["families"].values():
                self.assertIsNotNone(family["target_claim_rate"])

    def test_scenario_without_an_effect_has_no_target_row(self):
        for effect in (None, 0.0):
            with self.subTest(effect=effect):
                cal = _calibration(effect=effect)
                for scenario in ("null", "alternative"):
                    for family in cal["rules"]["bootstrap"][scenario][
                            "families"].values():
                        self.assertIsNone(family["target_claim_rate"])

    def test_scenario_without_an_effect_counts_every_claim_as_false(self):
        # Without an effect the alternative scenario *is* a null scenario:
        # its false-discovery rate must not be lower than the null's,
        # because no row carries a true effect.
        cal = _calibration(effect=None)
        null_fdr = cal["rules"]["bootstrap"]["null"]["families"]["4"][
            "claim_fdr"]
        alternative_fdr = cal["rules"]["bootstrap"]["alternative"][
            "families"]["4"]["claim_fdr"]
        self.assertGreaterEqual(alternative_fdr, null_fdr - 0.05)


if __name__ == "__main__":
    unittest.main()
