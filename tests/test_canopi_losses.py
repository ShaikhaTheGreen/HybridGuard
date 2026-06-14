"""Unit tests for canopi.losses (numpy reference path — runs without torch).

Each test encodes one property the JOINT objective must have. The torch versions
share the formulas and are exercised on the GPU; these pin the math.
"""
import unittest

import numpy as np

from hybridguard.canopi.losses import (
    LossWeights,
    drift_np,
    hardneg_np,
    pauc_toppush_np,
    supcon_inv_np,
)


class TestPAUC(unittest.TestCase):
    def test_low_fpr_separation_lowers_loss(self):
        neg = np.linspace(0, 1, 50)
        pos_far = np.full(20, 3.0)   # clearly above all negs
        pos_near = np.full(20, 0.9)  # overlap the hard negatives
        self.assertLess(
            pauc_toppush_np(pos_far, neg, beta=0.1),
            pauc_toppush_np(pos_near, neg, beta=0.1),
        )

    def test_only_top_beta_negatives_matter(self):
        # beta small -> only the single hardest negative enters the surrogate.
        neg = np.array([0.0, 0.1, 0.2, 0.9])  # n=4, beta=0.01 -> k=1
        pos = np.array([1.0, 1.0])
        base = pauc_toppush_np(pos, neg, beta=0.01)
        neg2 = neg.copy()
        neg2[0] = -5.0  # change a NON-top negative
        self.assertAlmostEqual(base, pauc_toppush_np(pos, neg2, beta=0.01), places=9)
        neg3 = neg.copy()
        neg3[-1] = 5.0  # change THE top negative -> loss must change
        self.assertNotAlmostEqual(base, pauc_toppush_np(pos, neg3, beta=0.01), places=6)


class TestSupConInv(unittest.TestCase):
    def test_clustered_views_beat_scrambled(self):
        # two intents, three views each, well separated -> low SupCon loss.
        a = np.array([[1, 0, 0, 0]] * 3, dtype=float) + 0.01
        b = np.array([[0, 1, 0, 0]] * 3, dtype=float) + 0.01
        z = np.vstack([a, b])
        good_groups = [0, 0, 0, 1, 1, 1]
        bad_groups = [0, 1, 0, 1, 0, 1]  # same vectors, wrong grouping
        self.assertLess(supcon_inv_np(z, good_groups), supcon_inv_np(z, bad_groups))


class TestHardNeg(unittest.TestCase):
    def test_similar_pairs_cost_more(self):
        a = np.array([[1.0, 0.0]])
        near = np.array([[0.95, 0.05]])
        far = np.array([[0.0, 1.0]])
        self.assertGreater(hardneg_np(a, near), hardneg_np(a, far))

    def test_orthogonal_pairs_zero_cost(self):
        a = np.array([[1.0, 0.0]])
        far = np.array([[0.0, 1.0]])
        self.assertEqual(hardneg_np(a, far, margin=0.2), 0.0)


class TestDrift(unittest.TestCase):
    def test_zero_when_identical(self):
        z = np.random.default_rng(0).standard_normal((5, 8))
        self.assertAlmostEqual(drift_np(z, z), 0.0, places=12)

    def test_positive_when_different(self):
        z = np.zeros((3, 4))
        v = np.ones((3, 4))
        self.assertGreater(drift_np(z, v), 0.0)


class TestToggles(unittest.TestCase):
    def test_baselines_as_lambda_settings(self):
        b6 = LossWeights.from_config({"loss": {"lam4_pauc": 0.0}})  # invariance-only
        self.assertFalse(b6.active()["pauc"])
        self.assertTrue(b6.active()["inv"])
        b7 = LossWeights.from_config({"loss": {"lam1_inv": 0, "lam2_hardneg": 0, "lam3_drift": 0}})
        self.assertTrue(b7.active()["pauc"])
        self.assertFalse(any([b7.active()["inv"], b7.active()["hardneg"], b7.active()["drift"]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
