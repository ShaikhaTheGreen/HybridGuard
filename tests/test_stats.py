"""
test_stats.py — WS7: the statistics layer (code/stats.py).

Covers the four primitives the paper relies on:
  * threshold_at_fpr respects the 1%-FPR budget and matches the optimizer's convention
  * bootstrap_ci has approximately nominal coverage on synthetic data
  * mcnemar gives the right sign and an exact p-value on a constructed disagreement
  * holm_bonferroni controls the family-wise error / orders correctly
"""
import numpy as np
import pytest

import stats
from npl_adaptive_experiment_v2 import threshold_at_fpr as opt_threshold


def _synth(seed=0, n=2000, sep=0.8):
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    s = np.where(y == 1, rng.normal(sep, 0.25, n), rng.normal(0.0, 0.25, n))
    return y, np.clip(s, 0, 1)


# --- threshold_at_fpr ---------------------------------------------------------

def test_threshold_respects_fpr_budget():
    y, s = _synth(seed=1)
    thr = stats.threshold_at_fpr(y, s, fpr=0.01)
    neg = s[y == 0]
    fpr = float((neg >= thr).mean())
    assert fpr <= 0.01 + 1e-9, f"FPR budget exceeded: {fpr}"


def test_threshold_matches_optimizer_convention():
    # stats.threshold_at_fpr must be identical to the convention frozen in the
    # adaptive optimizer, otherwise thresholds would differ across experiments.
    y, s = _synth(seed=2)
    assert stats.threshold_at_fpr(y, s, 0.01) == opt_threshold(y, s, 0.01)


# --- bootstrap CI -------------------------------------------------------------

def test_bootstrap_ci_brackets_point_estimate():
    y, s = _synth(seed=3)
    thr = stats.threshold_at_fpr(y, s)
    pt, lo, hi = stats.bootstrap_ci(lambda yy, ss: stats.recall_at(yy, ss, thr),
                                    y, s, n_boot=500, seed=7)
    assert lo <= pt <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_coverage_is_approximately_nominal():
    # Across many resimulations, a 95% CI for AUROC should contain the large-sample
    # estimate clearly more often than not (a loose, fast coverage sanity check).
    covered = 0
    trials = 40
    y0, s0 = _synth(seed=100, n=8000)
    ref = stats.auroc(y0, s0)
    for t in range(trials):
        y, s = _synth(seed=200 + t, n=600)
        _, lo, hi = stats.bootstrap_ci(stats.auroc, y, s, n_boot=300, seed=t)
        covered += int(lo <= ref <= hi)
    assert covered >= int(0.8 * trials), f"coverage too low: {covered}/{trials}"


def test_aggregate_seeds_record():
    rec = stats.aggregate_seeds([0.96, 0.95, 0.97, 0.96, 0.94])
    assert rec["n_seeds"] == 5
    assert 0.94 <= rec["mean"] <= 0.97
    assert rec["std"] > 0
    assert rec["ci_lo"] <= rec["mean"] <= rec["ci_hi"]


# --- McNemar ------------------------------------------------------------------

def test_mcnemar_detects_directional_disagreement():
    # Construct A clearly better than B: A correct everywhere, B wrong on a chunk.
    n = 400
    y = np.array([1] * n)
    sa = np.full(n, 0.9)          # A flags all positives -> all correct at thr=0.5
    sb = np.concatenate([np.full(n // 4, 0.1), np.full(3 * n // 4, 0.9)])  # B misses a quarter
    res = stats.mcnemar(y, sa, sb, 0.5, 0.5)
    assert res["b01"] > res["b10"]          # A right & B wrong dominates
    assert res["p_exact"] < 0.001
    assert res["n_discordant"] == n // 4


def test_mcnemar_symmetric_when_equal():
    y, s = _synth(seed=9)
    thr = stats.threshold_at_fpr(y, s)
    res = stats.mcnemar(y, s, s, thr, thr)     # a detector vs itself
    assert res["n_discordant"] == 0
    assert res["p_exact"] == 1.0


# --- multiple testing ---------------------------------------------------------

def test_holm_rejects_small_and_orders():
    res = stats.holm_bonferroni([0.001, 0.02, 0.04, 0.3], alpha=0.05)
    assert res["reject"][0] is True
    assert res["reject"][3] is False
    # Holm-adjusted p-values are monotone non-decreasing in the sorted order
    adj = np.array(res["adjusted"])
    order = np.argsort([0.001, 0.02, 0.04, 0.3])
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_bonferroni_is_more_conservative_than_holm():
    p = [0.001, 0.02, 0.04, 0.3]
    b = np.array(stats.bonferroni(p)["adjusted"])
    h = np.array(stats.holm_bonferroni(p)["adjusted"])
    assert np.all(b >= h - 1e-12)
