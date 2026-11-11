"""Within-model calibration dose-response / sign-reversal ablation probe.

The retrained term ablations remove a component and re-optimise; the on/off
knockout zeroes a learned coefficient at fixed weights.  Neither separates a
signed, graded correction from a generic gate perturbation.  These tests pin
the graded intervention, its record stamp and the E2 aggregation companion.
"""

from __future__ import annotations

import contextlib
import io
import unittest

import torch

from cage.model import build
from cage.train import (calib_term_dose_response_eval,
                        calib_term_knockout_eval, eval_accuracy,
                        train_model)
from cage.experiments import (Settings, _e2_term_dose_response_companion,
                              _stamp_calib_term_dose_response)

_TINY = {
    "vocab_size": 64, "d_model": 32, "n_heads": 4, "n_layers": 2,
    "max_len": 128,
}


def _trained_calib(steps=3):
    torch.manual_seed(0)
    model = build({**_TINY, "gate": "calib"})
    train_model(model, steps=steps, batch=4, n_pairs=6, vocab=64,
                log_every=0)
    return model


def _settings():
    return Settings(device="cpu", d_model=32, n_heads=4, n_layers=2,
                    max_len=128, vocab=64, batch=4, n_pairs=4, steps=3,
                    seeds=(0,), eval_batch=4, eval_batches=1,
                    stats_batch=4, stats_batches=1)


class DoseResponseProbeTest(unittest.TestCase):
    def test_non_calib_abstains_and_validation(self):
        torch.manual_seed(20)
        plain = build(_TINY)
        out = calib_term_dose_response_eval(  # SECURITY: eval() usage - manual review needed
            plain, 4, 64, "cpu", batch=4, batches=1)
        self.assertFalse(out["applicable"])
        self.assertEqual(out["n_pairs"], 4)

        for bad_terms in ("length", ("bogus",), 3):
            with self.subTest(bad_terms=bad_terms):
                with self.assertRaises(ValueError):
                    calib_term_dose_response_eval(  # SECURITY: eval() usage - manual review needed
                        plain, 4, 64, "cpu", batch=4, batches=1,
                        terms=bad_terms)

        for bad_scales in ("0.5", (True,), (float("nan"),),
                           (float("inf"),), (None,), (object(),), 3,
                           {0.0: 1}):
            with self.subTest(bad_scales=bad_scales):
                with self.assertRaises(ValueError):
                    calib_term_dose_response_eval(  # SECURITY: eval() usage - manual review needed
                        plain, 4, 64, "cpu", batch=4, batches=1,
                        scales=bad_scales)

    def test_scale_one_is_baseline_zero_matches_knockout_and_restores(self):
        model = _trained_calib(steps=4)
        parameters = [p for p in model.parameters()]
        before = [p.detach().clone() for p in parameters]
        out = calib_term_dose_response_eval(  # SECURITY: eval() usage - manual review needed
            model, 6, 64, "cpu", batch=4, batches=2, seed=123,
            terms=("both",), scales=(1.0, 0.0, 0.5, 2.0, -1.0))
        self.assertTrue(out["applicable"])
        self.assertEqual(out["scales"], [1.0, 0.0, 0.5, 2.0, -1.0])
        rows = out["by_term"]["both"]["by_scale"]
        self.assertEqual(sorted(rows), ["-1", "0", "0.5", "1", "2"])
        self.assertEqual(rows["1"]["accuracy"], out["baseline"])
        self.assertEqual(rows["1"]["delta"], 0.0)
        for key in ("0", "0.5", "2", "-1"):
            self.assertEqual(rows[key]["delta"],
                             rows[key]["accuracy"] - out["baseline"])

        knockout = calib_term_knockout_eval(  # SECURITY: eval() usage - manual review needed
            model, 6, 64, "cpu", batch=4, batches=2, seed=123,
            terms=("both",))
        self.assertAlmostEqual(
            rows["0"]["accuracy"],
            knockout["by_term"]["both"]["accuracy"], places=12)
        self.assertAlmostEqual(
            rows["0"]["delta"], knockout["by_term"]["both"]["delta"],
            places=12)
        for parameter, value in zip(model.parameters(), before):
            self.assertTrue(torch.equal(parameter.detach(), value))

    def test_scale_keys_are_stable_for_integral_floats(self):
        model = _trained_calib(steps=2)
        out = calib_term_dose_response_eval(  # SECURITY: eval() usage - manual review needed
            model, 4, 64, "cpu", batch=4, batches=1,
            terms=("length",), scales=(0, 2, -1, 2.0, 0.0))
        self.assertEqual(out["scales"], [0.0, 2.0, -1.0])
        self.assertEqual(sorted(out["by_term"]["length"]["by_scale"]),
                         ["-1", "0", "2"])


class DoseResponseStampTest(unittest.TestCase):
    def test_stamp_records_base_and_longest_and_restores_model(self):
        torch.manual_seed(21)
        settings = _settings()
        model = build({**_TINY, "gate": "calib"})
        train_model(model, steps=3, batch=4, n_pairs=4, vocab=64,
                    log_every=0)
        lengths = [4, 8]
        rec = {"accuracy_by_length": {
            length: eval_accuracy(model, length, 64, "cpu", batch=4,
                                  batches=1, seed=123)
            for length in lengths}}
        before = [p.detach().clone() for p in model.parameters()]
        payload = _stamp_calib_term_dose_response(
            rec, model, settings, lengths, eval_seed=123)
        self.assertTrue(payload["applicable"])
        self.assertEqual(payload["base_length"], 4)
        self.assertEqual(payload["longest"], 8)
        self.assertEqual(sorted(int(k) for k in payload["by_length"]),
                         [4, 8])
        self.assertEqual(payload["scales"], [0.0, 0.5, 2.0, -1.0])
        self.assertTrue(rec["calib_term_dose_response"] is payload)
        for entry in payload["by_length"].values():
            for term in ("length", "concentration", "both"):
                self.assertEqual(
                    sorted(entry["by_term"][term]["by_scale"]),
                    ["-1", "0", "0.5", "2"])
        self.assertTrue(rec["calib_term_dose_response"]["base_matches_raw"])
        for parameter, value in zip(model.parameters(), before):
            self.assertTrue(torch.equal(parameter.detach(), value))

    def test_stamp_abstains_on_empty_lengths_and_bad_record(self):
        model = _trained_calib(steps=2)
        rec = {}
        self.assertIsNone(_stamp_calib_term_dose_response(
            rec, model, _settings(), [], eval_seed=123))
        self.assertFalse(rec["calib_term_dose_response"]["applicable"])
        self.assertIsNone(_stamp_calib_term_dose_response(
            None, model, _settings(), [4], eval_seed=123))


class DoseResponseCompanionTest(unittest.TestCase):
    TERMS = ("length", "concentration", "both")

    def _entry(self, baseline, deltas):
        by_scale = {}
        for scale, delta in deltas.items():
            by_scale[f"{float(scale):.12g}"] = {
                "scale": float(scale),
                "accuracy": baseline + delta,
                "delta": delta,
            }
        return {
            "applicable": True,
            "baseline": baseline,
            "by_term": {
                term: {"applicable": True, "by_scale": by_scale}
                for term in self.TERMS
            },
        }

    def _record(self, seed, base, long, deltas):
        return {
            "seed": seed,
            "calib_term_dose_response": {
                "applicable": True,
                "by_length": {
                    "4": self._entry(base, deltas),
                    "8": self._entry(long, deltas),
                },
            },
        }

    def test_graded_directional_pattern_and_paired_ci(self):
        deltas = {0.0: -0.2, 0.5: -0.1, 2.0: -0.05, -1.0: -0.3}
        by_variant = {
            "calib": [self._record(0, 0.8, 0.7, deltas),
                      self._record(1, 0.7, 0.6, deltas)],
            "calib_raw_ent": [self._record(0, 0.7, 0.6, deltas)],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            payload = _e2_term_dose_response_companion(by_variant, [4, 8])
        self.assertTrue(payload["applicable"])
        self.assertEqual(payload["lengths"], [4, 8])
        self.assertEqual(sorted(payload["arms"]), ["calib", "calib_raw_ent"])
        short = payload["arms"]["calib"]["by_length"]["4"]
        self.assertEqual(short["n"], 2)
        term = short["by_term"]["both"]
        self.assertEqual(term["by_scale"]["0"]["n"], 2)
        self.assertAlmostEqual(term["by_scale"]["0"]["delta"]["mean"], -0.2)
        self.assertEqual(term["by_scale"]["0"]["delta"]["seeds_used"],
                         [0, 1])
        for key in ("lo", "hi", "p_value", "sd", "mde80",
                    "sign_flip_p_two_sided"):
            self.assertIn(key, term["by_scale"]["0"]["delta"])
        self.assertTrue(term["knockout_hurts"])
        self.assertTrue(term["reversal_worse_than_knockout"])
        self.assertTrue(term["trained_scale_is_peak"])
        self.assertTrue(term["half_dose_between"])
        self.assertTrue(term["double_dose_hurts"])
        self.assertTrue(term["direction_specific"])
        self.assertTrue(term["graded"])

    def test_malformed_rows_are_skipped_not_fatal(self):
        deltas = {0.0: -0.1, 0.5: -0.05, 2.0: -0.02, -1.0: -0.2}
        good = self._record(0, 0.8, 0.7, deltas)
        bad_payload = dict(good, calib_term_dose_response="not-a-mapping")
        with contextlib.redirect_stdout(io.StringIO()):
            payload = _e2_term_dose_response_companion(
                {"calib": [None, "x", bad_payload, good]}, [4, 8])
        self.assertTrue(payload["applicable"])
        self.assertEqual(
            payload["arms"]["calib"]["by_length"]["4"]["n"], 1)

    def test_no_probe_and_empty_lengths_abstain(self):
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            payload = _e2_term_dose_response_companion({}, [4, 8])
        self.assertFalse(payload["applicable"])
        self.assertIn("n/a", buf.getvalue())
        self.assertFalse(_e2_term_dose_response_companion(
            {}, [])["applicable"])

    def test_flags_follow_the_mean_curve(self):
        deltas = {0.0: 0.0, 0.5: 0.0, 2.0: 0.0, -1.0: 0.0}
        by_variant = {"calib": [self._record(0, 0.8, 0.7, deltas)]}
        with contextlib.redirect_stdout(io.StringIO()):
            payload = _e2_term_dose_response_companion(by_variant, [4])
        term = payload["arms"]["calib"]["by_length"]["4"]["by_term"]["both"]
        self.assertFalse(term["knockout_hurts"])
        self.assertFalse(term["direction_specific"])
        self.assertFalse(term["graded"])


if __name__ == "__main__":
    unittest.main()
