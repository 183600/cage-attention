"""Regression tests for the run-level reproducibility layer."""

from __future__ import annotations

import os
import random
import unittest

import torch

from cage.data import batch_fingerprint, make_batch
from cage.repro import canonical_seed, seed_everything, stream_generator


class CanonicalSeedTest(unittest.TestCase):
    def test_alias_is_the_same_run(self):
        first = seed_everything(0, deterministic=False)
        draws_a = (random.random(), float(torch.rand(())))
        second = seed_everything(2 ** 32, deterministic=False)
        draws_b = (random.random(), float(torch.rand(())))
        self.assertEqual(first["seed"], 0)
        self.assertEqual(second["seed"], 0)
        self.assertEqual(second["requested_seed"], 2 ** 32)
        self.assertEqual(draws_a, draws_b)

    def test_negative_seed_is_canonicalised(self):
        requested, key = canonical_seed(-1)
        self.assertEqual(requested, -1)
        self.assertEqual(key, 2 ** 32 - 1)

    def test_invalid_seeds_are_rejected(self):
        for bad in (True, False, 1.5, "3", None):
            with self.subTest(seed=bad):
                with self.assertRaises(ValueError):
                    canonical_seed(bad)
                with self.assertRaises(ValueError):
                    seed_everything(bad)

    def test_data_generator_uses_the_same_key(self):
        def draw(seed):
            gen = stream_generator(seed)
            return batch_fingerprint(
                *make_batch(4, 3, 32, "cpu", gen=gen))

        self.assertEqual(draw(0), draw(2 ** 32))
        self.assertNotEqual(draw(0), draw(1))


class SeedRecordTest(unittest.TestCase):
    def setUp(self):
        self._old_hashseed = os.environ.get("PYTHONHASHSEED")

    def tearDown(self):
        if self._old_hashseed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = self._old_hashseed

    def test_hashseed_is_the_canonical_key(self):
        info = seed_everything(2 ** 32 + 7, deterministic=False)
        self.assertEqual(info["seed"], 7)
        self.assertEqual(info["pythonhashseed"], "7")

    def test_record_says_what_was_seeded(self):
        info = seed_everything(11, deterministic=False)
        for key in ("seed", "seed_key", "requested_seed", "python_random",
                    "numpy", "torch", "cuda_devices", "deterministic",
                    "cudnn", "tf32", "torch_num_threads"):
            self.assertIn(key, info)


if __name__ == "__main__":
    unittest.main()
