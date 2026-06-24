"""
stats.py — the shared statistics layer for the Computers & Security run.

Every empirical table in the paper is reported as a 5-seed mean +/- s.d. with a
bootstrap 95% CI, and the headline comparisons carry a paired McNemar test under a
multiple-testing correction. This module is the single implementation of those
primitives so the diamond, adaptive, multilingual, and over-defense experiments
all compute them identically.

Contents
--------
* threshold_at_fpr(y, scores, fpr)        frozen 1%-FPR operating point (the EXACT
                                          convention used in the adaptive optimizer)
* recall_at / metric helpers              R@thr, AUROC, AUPRC
* bootstrap_ci(metric_fn, y, scores, ...) percentile bootstrap 95% CI of any metric
* aggregate_seeds(values)                 mean, std, ci_lo, ci_hi, n across seeds
* mcnemar(y, scores_a, scores_b, ...)     paired test: chi^2 (cc) + exact binomial p
* holm_bonferroni(pvals) / bonferroni     multiple-testing correction

Pure numpy/scipy; runs on CPU.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "threshold_at_fpr", "recall_at", "auroc", "auprc",
    "bootstrap_ci", "aggregate_seeds", "mcnemar",
    "bonferroni", "holm_bonferroni",
]


# ---------------------------------------------------------------------------
# Operating point and point metrics
# ---------------------------------------------------------------------------
def threshold_at_fpr(y, scores, fpr: float = 0.01) -> float:
    """Frozen threshold achieving AT MOST `fpr` false-positive rate on the negatives.

    Canonical convention for the whole repo: sort the negative scores ascending and
    take the one at index ceil((1-fpr) * n_neg). Flagging scores >= this threshold
    then admits at most ceil(fpr * n_neg) negatives, so the realized FPR never
    exceeds `fpr` (it can only undershoot due to ties/discretization). This is the
    single source of truth; npl_adaptive_experiment_v2 and npl_diamond_experiment
    both delegate here, so the frozen 1%-FPR threshold is identical across every
    experiment for a given (y, scores). Determinism: depends only on the (y, scores)
    pair, never on order."""
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    neg = np.sort(s[y == 0])
    if neg.size == 0:
        return 0.5
    idx = min(int(np.ceil((1 - fpr) * neg.size)), neg.size - 1)
    return float(neg[idx])


def recall_at(y, scores, thr: float) -> float:
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    pos = y == 1
    return float((s[pos] >= thr).mean()) if pos.any() else float("nan")


def auroc(y, scores) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def auprc(y, scores) -> float:
    """Average precision (AUPRC). Reported as a co-primary metric to AUROC because
    it separates the saturated models that AUROC ranks as tied."""
    from sklearn.metrics import average_precision_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, np.asarray(scores, dtype=float)))


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_ci(metric_fn: Callable[[np.ndarray, np.ndarray], float],
                 y, scores, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Percentile bootstrap (1-alpha) CI for `metric_fn(y, scores)`.

    Resamples example indices with replacement n_boot times. Returns
    (point_estimate, ci_lo, ci_hi). Deterministic given `seed`. Resamples that
    degenerate (e.g. a single class, so the metric is NaN) are dropped before
    taking percentiles, which keeps a low-FPR recall metric well-defined."""
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    point = float(metric_fn(y, s))
    stats: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        val = metric_fn(y[idx], s[idx])
        if val is not None and np.isfinite(val):
            stats.append(float(val))
    if not stats:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(stats, 100 * (alpha / 2)))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return point, lo, hi


def aggregate_seeds(values: Sequence[float], alpha: float = 0.05) -> Dict[str, float]:
    """Aggregate a per-seed metric across seeds into the reporting record
    {mean, std, ci_lo, ci_hi, n_seeds}. The CI here is the across-seed percentile
    interval (the spread the paper reports as `mean +/- s.d.` with a 95% CI). With
    few seeds the percentile interval collapses toward the min/max, which is the
    honest, non-parametric summary; the per-example bootstrap_ci above is used where
    the paper claims a separation between two cells."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n_seeds": 0}
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "ci_lo": float(np.percentile(v, 100 * (alpha / 2))),
        "ci_hi": float(np.percentile(v, 100 * (1 - alpha / 2))),
        "n_seeds": int(v.size),
    }


# ---------------------------------------------------------------------------
# Paired McNemar test
# ---------------------------------------------------------------------------
def mcnemar(y, scores_a, scores_b, thr_a: float, thr_b: float) -> Dict[str, float]:
    """Paired McNemar test on the CORRECTNESS of two detectors at their frozen
    thresholds. Returns the chi^2 statistic (with Edwards' continuity correction),
    the EXACT two-sided binomial p-value over the discordant pairs, and the
    discordant counts b01 (a right / b wrong) and b10 (a wrong / b right).

    The exact binomial p is used (not the chi^2 asymptotic p) because the discordant
    counts on a saturated benchmark are small, where the asymptotic approximation is
    unreliable; the chi^2 statistic is returned for completeness."""
    from scipy.stats import binomtest
    y = np.asarray(y)
    pa = (np.asarray(scores_a, dtype=float) >= thr_a).astype(int)
    pb = (np.asarray(scores_b, dtype=float) >= thr_b).astype(int)
    ca = (pa == y)
    cb = (pb == y)
    b01 = int(np.sum(ca & ~cb))   # a correct, b wrong
    b10 = int(np.sum(~ca & cb))   # a wrong, b correct
    nd = b01 + b10
    if nd == 0:
        chi2 = 0.0
    else:
        chi2 = (abs(b01 - b10) - 1) ** 2 / nd
    p_exact = float(binomtest(min(b01, b10), nd, 0.5).pvalue) if nd > 0 else 1.0
    return {"chi2_cc": float(chi2), "p_exact": p_exact,
            "b01": b01, "b10": b10, "n_discordant": nd}


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------
def bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> Dict[str, object]:
    """Bonferroni: reject p_i < alpha/m. Returns adjusted p-values and reject flags."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.minimum(p * m, 1.0)
    return {"adjusted": adj.tolist(), "reject": (p < alpha / m).tolist(), "alpha": alpha}


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> Dict[str, object]:
    """Holm step-down. Uniformly more powerful than Bonferroni at the same FWER.
    Returns Holm-adjusted p-values (monotone, in the ORIGINAL order) and reject
    flags."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj_sorted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * p[idx], 1.0)
        running = max(running, val)   # enforce monotonicity
        adj_sorted[idx] = running
    reject = [bool(adj_sorted[i] < alpha) for i in range(m)]
    return {"adjusted": adj_sorted.tolist(), "reject": reject, "alpha": alpha}


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = np.array([0] * 500 + [1] * 500)
    # detector A slightly better than B
    sa = np.clip(np.where(y == 1, rng.normal(0.7, 0.2, 1000), rng.normal(0.3, 0.2, 1000)), 0, 1)
    sb = np.clip(np.where(y == 1, rng.normal(0.6, 0.25, 1000), rng.normal(0.35, 0.25, 1000)), 0, 1)
    ta, tb = threshold_at_fpr(y, sa), threshold_at_fpr(y, sb)
    print("thr_a", round(ta, 3), "R@1%FPR", round(recall_at(y, sa, ta), 3))
    print("AUROC", round(auroc(y, sa), 3), "AUPRC", round(auprc(y, sa), 3))
    pt, lo, hi = bootstrap_ci(lambda yy, ss: recall_at(yy, ss, ta), y, sa, n_boot=500, seed=1)
    print("R@1%FPR boot CI", round(pt, 3), [round(lo, 3), round(hi, 3)])
    print("McNemar A vs B", mcnemar(y, sa, sb, ta, tb))
    print("Holm", holm_bonferroni([0.001, 0.02, 0.04, 0.3]))
