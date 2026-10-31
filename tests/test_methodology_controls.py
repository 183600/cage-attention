"""Tests for the methodology-control additions.

Covers the review-driven controls added next to the pre-registered
readouts:

* ``calib_pos_norm`` - per-row visible-token normalisation ablation of the
  calibrated gate (identical on the final query row by construction);
* the frozen ``calib_scale`` buffer (parameter identifiability);
* the cross-normalisation ``attn_write_ratio`` / ``mlp_write_ratio``
  statistics and their collection;
* ``layer_knockout_eval`` - the interventional per-layer probe;
* the null-aware full-distribution false-concentration readouts and the
  gate-decile companions;
* the eval-time position-extrapolation control (``fold`` / ``zero``) and
  its E2 stamping helper;
* the E1 record's learned-coefficient stamping.
"""

import math
import unittest

import torch

from cage.data import make_batch
from cage.model import Cfg, build
from cage.train import (collect_stats, eval_accuracy,
                        false_concentration_analysis,
                        false_concentration_sweep, layer_knockout_eval,
                        train_model)

_TINY = {
    "vocab_size": 64, "d_model": 32, "n_heads": 4, "n_layers": 2,
    "max_len": 128,
}


def _trained_calib(pos_norm=False, steps=3):
    torch.manual_seed(0)
    model = build({**_TINY, "gate": "calib", "calib_pos_norm": pos_norm})
    train_model(model, steps=steps, batch=4, n_pairs=6, vocab=64,
                log_every=0)
    return model


class CalibPosNormTest(unittest.TestCase):
    def test_flag_validation(self):
        from cage.model import Cfg

        for bad in (
            {"gate": "query"},
            {"gate": "calib", "calib_use_ent": False},
            {"gate": "calib", "calib_ent_norm": False},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Cfg(vocab_size=8, d_model=8, n_heads=2, n_layers=1,
                        max_len=8, calib_pos_norm=True, **bad)
        with self.assertRaises(ValueError):
            Cfg(vocab_size=8, d_model=8, n_heads=2, n_layers=1, max_len=8,
                gate="calib", calib_pos_norm="False")

    def test_final_query_row_is_unchanged_nonfinal_rows_differ(self):
        # On the final query row the visible-token divisor equals the
        # global log T, so the registered readout is identical; on a
        # non-final row the two arms must differ (that is the ablation).
        torch.manual_seed(1)
        base = build({**_TINY, "gate": "calib"})
        torch.manual_seed(1)
        ablated = build({**_TINY, "gate": "calib", "calib_pos_norm": True})
        idx, _, _ = make_batch(4, 6, 64, "cpu")
        T = idx.size(1)
        with torch.no_grad():
            _, _, s_base_last = base(idx, stat_pos=torch.full((4,), T - 1))
            _, _, s_abl_last = ablated(idx, stat_pos=torch.full((4,), T - 1))
            _, _, s_base_mid = base(idx, stat_pos=torch.full((4,), 3))
            _, _, s_abl_mid = ablated(idx, stat_pos=torch.full((4,), 3))
        g_base_last = s_base_last[0]["gate"].mean(-1)
        g_abl_last = s_abl_last[0]["gate"].mean(-1)
        g_base_mid = s_base_mid[0]["gate"].mean(-1)
        g_abl_mid = s_abl_mid[0]["gate"].mean(-1)
        self.assertTrue(torch.allclose(g_base_last, g_abl_last,
                                        atol=1e-6, rtol=1e-5))
        self.assertFalse(torch.allclose(g_base_mid, g_abl_mid,
                                        atol=1e-4, rtol=1e-3))

    def test_per_row_divisor_values(self):
        torch.manual_seed(2)
        model = build({**_TINY, "gate": "calib"})
        attn = model.blocks[0].attn
        vec = attn._log_vis_vec(6, torch.device("cpu"), torch.float32)
        expected = [max(math.log(t + 1), math.log(2.0)) for t in range(6)]
        self.assertTrue(torch.allclose(
            vec, torch.tensor(expected), atol=1e-6))
        self.assertEqual(attn._log_vis_vec(6, torch.device("cpu"),
                                           torch.float32) is vec, True)

    def test_training_smoke_with_flag(self):
        model = _trained_calib(pos_norm=True)
        acc = eval_accuracy(model, 6, 64, "cpu", batch=4, batches=1)
        self.assertTrue(0.0 <= acc <= 1.0)


class FrozenCalibScaleTest(unittest.TestCase):
    def test_scale_is_buffer_not_parameter(self):
        torch.manual_seed(3)
        model = build({**_TINY, "gate": "calib", "calib_scale": 0.5})
        param_names = {name for name, _ in model.named_parameters()}
        for i in range(2):
            self.assertIn(f"blocks.{i}.attn.calib_c", param_names)
            self.assertIn(f"blocks.{i}.attn.calib_tau", param_names)
            self.assertNotIn(f"blocks.{i}.attn.calib_scale", param_names)
        state = model.state_dict()
        self.assertIn("blocks.0.attn.calib_scale", state)
        self.assertEqual(float(state["blocks.0.attn.calib_scale"]), 0.5)
        attn = model.blocks[0].attn
        self.assertEqual(float(attn.calib_scale.detach()), 0.5)

    def test_effective_coefficient_readable_and_trainable(self):
        model = _trained_calib()
        attn = model.blocks[0].attn
        self.assertTrue(attn.calib_c.requires_grad)
        self.assertFalse(attn.calib_scale.requires_grad)
        self.assertIsNotNone(attn.calib_c.grad)
        # The E2 coefficient reader must keep working with the buffer.
        from cage.experiments import _calibration_coefficients

        coeffs = _calibration_coefficients(model)
        self.assertTrue(coeffs["applicable"])
        self.assertEqual(coeffs["n_layers"], 2)
        self.assertTrue(math.isfinite(coeffs["c_effective_mean"]))
        self.assertTrue(math.isfinite(coeffs["scale_mean"]))


class WriteRatioStatsTest(unittest.TestCase):
    def test_stats_contain_write_ratios_pre_and_post(self):
        for norm in ("pre", "post", "seednorm"):
            with self.subTest(norm=norm):
                torch.manual_seed(4)
                model = build({**_TINY, "gate": "calib", "norm": norm})
                train_model(model, steps=2, batch=4, n_pairs=6, vocab=64,
                            log_every=0)
                stats = collect_stats(model, 6, 64, "cpu", batch=4,
                                      batches=1)
                for key in ("attn_write_ratio", "mlp_write_ratio"):
                    self.assertIn(key, stats)
                    self.assertEqual(stats[key].shape,
                                     (4, model.cfg.n_layers))
                    self.assertTrue(torch.isfinite(stats[key]).all())
                    self.assertTrue((stats[key] > 0).all())

    def test_post_resid_norm_is_normalized_but_write_ratio_is_free(self):
        # The documented incomparability: post-norm resid_norm levels are
        # pinned by LayerNorm, pre-norm levels grow freely.  The write
        # ratios live on a shared scale in both schemes.
        torch.manual_seed(5)
        pre = build({**_TINY, "gate": "none", "norm": "pre"})
        post = build({**_TINY, "gate": "none", "norm": "post"})
        for model in (pre, post):
            train_model(model, steps=2, batch=4, n_pairs=6, vocab=64,
                        log_every=0)
        s_pre = collect_stats(pre, 6, 64, "cpu", batch=4, batches=1)
        s_post = collect_stats(post, 6, 64, "cpu", batch=4, batches=1)
        pre_growth = (s_pre["resid_norm"][:, -1]
                      / s_pre["resid_norm"][:, 0]).mean()
        post_growth = (s_post["resid_norm"][:, -1]
                       / s_post["resid_norm"][:, 0]).mean()
        # Post-LN norms are ~scale-pinned; the pre-LN stream is not.
        self.assertLess(abs(float(post_growth) - 1.0), 0.2)
        self.assertGreater(float(pre_growth), 1.0)
        # Write ratios are ordinary positive magnitudes in both schemes.
        for stats in (s_pre, s_post):
            self.assertTrue((stats["mlp_write_ratio"] > 0).all())


class LayerKnockoutTest(unittest.TestCase):
    def test_probe_deterministic_and_restores_model(self):
        torch.manual_seed(6)
        model = build({**_TINY, "gate": "calib"})
        train_model(model, steps=3, batch=4, n_pairs=6, vocab=64,
                    log_every=0)
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        blocks_before = list(model.blocks)
        out1 = layer_knockout_eval(model, 6, 64, "cpu", batch=4, batches=1)
        out2 = layer_knockout_eval(model, 6, 64, "cpu", batch=4, batches=1)
        self.assertEqual(out1, out2)
        self.assertEqual(out1["n_layers"], 2)
        self.assertEqual(sorted(out1["by_layer"]), [0, 1])
        self.assertIsNotNone(out1["embedding_only"])
        for entry in out1["by_layer"].values():
            self.assertTrue(0.0 <= entry["accuracy"] <= 1.0)
            self.assertAlmostEqual(entry["delta"],
                                   entry["accuracy"] - out1["baseline"],
                                   places=12)
        for (name, p), (_, q) in zip(params_before.items(),
                                     model.named_parameters()):
            self.assertTrue(torch.equal(p, q))
        self.assertEqual(list(model.blocks), blocks_before)
        # A second ordinary evaluation is unchanged by the probe.
        acc = eval_accuracy(model, 6, 64, "cpu", batch=4, batches=1)
        self.assertEqual(acc, out1["baseline"])

    def test_layer_selection_and_validation(self):
        torch.manual_seed(7)
        model = build({**_TINY, "gate": "none"})
        out = layer_knockout_eval(model, 6, 64, "cpu", batch=4, batches=1,
                                  layers=(1,))
        self.assertEqual(out["layers"], [1])
        self.assertIsNone(out["embedding_only"])
        for bad in ((2,), (-1,), (True,), "0", {"0": 1}, (0.5,)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    layer_knockout_eval(model, 6, 64, "cpu", batch=4,
                                        batches=1, layers=bad)


class NullAwareFalseConcentrationTest(unittest.TestCase):
    def _stats(self, mp, nm, corr, gate=None):
        n, L = mp.shape
        stats = {"maxprob": mp, "null_mass": nm, "correct": corr}
        if gate is not None:
            stats["gate"] = gate
        return stats

    def test_full_distribution_split(self):
        # Two samples: both "conditionally concentrated" (maxprob 0.9 after
        # removing null) but sample 0 routes 80% of its mass to the null
        # slot.  At the 0.5 cut both stay confident in the full
        # distribution too (max(0.8, 0.2*0.9) = 0.8); the arithmetic and
        # the split are verified at a higher threshold below, where the
        # null-heavy sample drops out of the confident set.
        mp = torch.tensor([[0.9, 0.9], [0.9, 0.9]])
        nm = torch.tensor([[0.8, 0.8], [0.1, 0.1]])
        corr = torch.tensor([0.0, 1.0])
        out = false_concentration_analysis(self._stats(mp, nm, corr))
        self.assertEqual(out["n_high_maxprob"], 2)
        self.assertEqual(out["n_high_maxprob_full"], 2)
        self.assertEqual(out["n_false_conc_full"], 1)
        self.assertAlmostEqual(out["false_conc_rate_full"], 0.5)
        self.assertAlmostEqual(out["mean_null_mass_on_high"], 0.45)
        # Full max-prob arithmetic: sample 0 -> max(0.8, 0.2*0.99) = 0.8,
        # below a 0.9 cut; its conditional maxprob 0.99 stays above it.
        mp2 = torch.tensor([[0.99, 0.99]])
        nm2 = torch.tensor([[0.8, 0.8]])
        corr2 = torch.tensor([0.0])
        out3 = false_concentration_analysis(
            self._stats(mp2, nm2, corr2), maxprob_thresh=0.9)
        self.assertEqual(out3["n_high_maxprob"], 1)
        self.assertEqual(out3["n_high_maxprob_full"], 0)

    def test_no_null_mass_leaves_keys_out(self):
        mp = torch.tensor([[0.9, 0.2], [0.1, 0.8]])
        corr = torch.tensor([1.0, 0.0])
        out = false_concentration_analysis(
            {"maxprob": mp, "correct": corr})
        self.assertNotIn("n_high_maxprob_full", out)
        self.assertNotIn("mean_null_mass_on_high", out)

    def test_gate_deciles_present(self):
        mp = torch.rand(40, 2) * 0.5 + 0.5
        nm = torch.rand(40, 2) * 0.2
        corr = torch.rand(40).round()
        gate = torch.rand(40, 2)
        out = false_concentration_analysis(
            self._stats(mp, nm, corr, gate=gate))
        for key in ("gate_on_correct_q10", "gate_on_correct_q50",
                    "gate_on_correct_q90", "gate_on_false_conc_q10",
                    "gate_on_false_conc_q50", "gate_on_false_conc_q90"):
            self.assertIn(key, out)
        sweep = false_concentration_sweep(self._stats(mp, nm, corr, gate=gate))
        for row in sweep.values():
            self.assertIn("gate_on_correct_q50", row)
            if "n_high_maxprob_full" in row:
                self.assertIn("false_conc_rate_full", row)


class PositionExtrapolationControlTest(unittest.TestCase):
    def test_set_mode_validation(self):
        torch.manual_seed(8)
        model = build(_TINY)
        model.set_pos_extrapolation("learned")
        self.assertEqual(model.pos_extrapolation, "learned")
        self.assertIsNone(model.pos_extrapolation_from)
        for bad_args in (("folds", 4), ("fold", None), ("fold", 0),
                         ("fold", True), ("fold", 2.0), ("zero", -1)):
            with self.subTest(bad_args=bad_args):
                with self.assertRaises(ValueError):
                    model.set_pos_extrapolation(*bad_args)
        model.set_pos_extrapolation("fold", 10)
        self.assertEqual(model.pos_extrapolation, "fold")
        self.assertEqual(model.pos_extrapolation_from, 10)

    def test_pos_rows_semantics(self):
        torch.manual_seed(9)
        model = build(_TINY)
        weight = model.pos.weight.detach()
        T, start = 20, 7
        model.set_pos_extrapolation("learned")
        self.assertTrue(torch.equal(model._pos_rows(T), weight[:T]))
        model.set_pos_extrapolation("fold", start)
        idx = torch.arange(T)
        folded = torch.where(idx >= start, idx % start, idx)
        self.assertTrue(torch.equal(model._pos_rows(T),
                                    weight.index_select(0, folded)))
        # Shorter than the trained range: control is a no-op.
        self.assertTrue(torch.equal(model._pos_rows(start), weight[:start]))
        model.set_pos_extrapolation("zero", start)
        zero_rows = model._pos_rows(T)
        self.assertTrue(torch.equal(zero_rows[:start], weight[:start]))
        self.assertTrue(torch.equal(zero_rows[start:],
                                    torch.zeros_like(weight[start:T])))

    def test_eval_modes_run_and_default_is_learned(self):
        torch.manual_seed(10)
        model = build({**_TINY, "gate": "calib"})
        train_model(model, steps=3, batch=4, n_pairs=4, vocab=64,
                    log_every=0)
        # n_pairs=4 -> trained T=10; evaluate at n_pairs=12 -> T=26.
        acc_learned = eval_accuracy(model, 12, 64, "cpu", batch=4,
                                    batches=1)
        model.set_pos_extrapolation("fold", 10)
        try:
            acc_fold = eval_accuracy(model, 12, 64, "cpu", batch=4,
                                     batches=1)
        finally:
            model.set_pos_extrapolation("learned")
        self.assertTrue(0.0 <= acc_fold <= 1.0)
        self.assertTrue(0.0 <= acc_learned <= 1.0)
        self.assertEqual(model.pos_extrapolation, "learned")

    def test_e2_stamp_helper(self):
        from cage.experiments import Settings
        from cage.experiments import _stamp_position_extrapolation_control

        torch.manual_seed(11)
        settings = Settings(device="cpu", d_model=32, n_heads=4,
                            n_layers=2, max_len=128, vocab=64, batch=4,
                            n_pairs=4, steps=3, seeds=(0,),
                            eval_batch=4, eval_batches=1, stats_batch=4,
                            stats_batches=1)
        model = build({**_TINY, "gate": "calib"})
        train_model(model, steps=3, batch=4, n_pairs=4, vocab=64,
                    log_every=0)
        lengths = [4, 8, 16]
        rec = {"accuracy_by_length": {
            L: eval_accuracy(model, L, 64, "cpu", batch=4, batches=1)
            for L in lengths}}
        _stamp_position_extrapolation_control(rec, model, settings,
                                              lengths, eval_seed=123)
        control = rec["pos_extrapolation_control"]
        self.assertEqual(control["trained_seq_len"], 10)
        self.assertEqual(sorted(int(k) for k in control["by_length"]),
                         [8, 16])
        for entry in control["by_length"].values():
            for key in ("learned", "fold", "zero", "fold_minus_learned",
                        "zero_minus_learned"):
                self.assertIn(key, entry)
        self.assertIn("fold_recovery_at_longest", control)
        self.assertEqual(model.pos_extrapolation, "learned")
        self.assertIsNone(model.pos_extrapolation_from)
        # A sweep entirely within the trained length stamps nothing.
        rec2 = {"accuracy_by_length": {2: 0.5}}
        _stamp_position_extrapolation_control(rec2, model, settings, [2],
                                              eval_seed=123)
        self.assertNotIn("pos_extrapolation_control", rec2)


class E1CoefficientStampTest(unittest.TestCase):
    def test_e1_run_records_learned_coefficients(self):
        from cage.experiments import Settings, _e1_run

        settings = Settings(device="cpu", d_model=32, n_heads=4,
                            n_layers=2, max_len=128, vocab=64, batch=4,
                            n_pairs=6, steps=3, seeds=(0,), eval_batch=4,
                            eval_batches=1, stats_batch=4, stats_batches=1,
                            probe_steps=2, probe_eval_batch=4,
                            probe_eval_batches=1)
        rec, _ = _e1_run(settings, seed=0, lr=3e-4, gate="calib",
                         record_extra={"variant": "calib"})
        self.assertIn("calib_coefficients", rec)
        self.assertTrue(rec["calib_coefficients"]["applicable"])
        for key in ("a_mean", "c_mean", "c_effective_mean",
                    "tau_effective_mean"):
            self.assertIn(key, rec)
            self.assertTrue(math.isfinite(rec[key]))
        # The same stamp on a non-calib arm marks itself inapplicable.
        rec_q, _ = _e1_run(settings, seed=0, lr=3e-4, gate="query",
                           record_extra={"variant": "query"})
        self.assertEqual(rec_q["calib_coefficients"],
                         {"applicable": False})


class InitialGateOpennessTest(unittest.TestCase):
    def test_query_and_calib_initial_openness_comparable(self):
        # The query gate initialises its bias at 2 (openness ~0.88) and the
        # calib gate at 6; the calib bias is chosen so that after
        # subtracting a*log T + c*Hbar at the default training length the
        # two variants start from a comparable openness.  This test pins
        # that otherwise-undocumented matching so the constants cannot
        # silently drift apart.
        torch.manual_seed(12)
        query = build({**_TINY, "gate": "query"})
        torch.manual_seed(12)
        calib = build({**_TINY, "gate": "calib"})
        idx, _, _ = make_batch(8, 6, 64, "cpu")
        with torch.no_grad():
            _, _, sq = query(idx, stat_pos=torch.full((8,), idx.size(1) - 1))
            _, _, sc = calib(idx, stat_pos=torch.full((8,), idx.size(1) - 1))
        q_open = float(sq[0]["gate"].mean())
        c_open = float(sc[0]["gate"].mean())
        self.assertTrue(0.5 < q_open < 0.99)
        self.assertTrue(0.5 < c_open < 0.99)
        self.assertLess(abs(q_open - c_open), 0.2)


if __name__ == "__main__":
    unittest.main()
