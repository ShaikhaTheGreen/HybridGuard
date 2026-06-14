"""
canopi.metrics
==============
Evaluation metrics for CANOPI, re-implemented to match the HybridGuard
orchestrator's *frozen* protocol exactly so new numbers stay comparable to the
published results. Pure numpy/scipy — no torch — so they run identically on a
laptop (smoke tests) and on the Colab GPU run.

Protocol invariants (do NOT change — they define comparability):
  * threshold_at_fpr: tau is the smallest score that holds FPR <= target on the
    NEGATIVES of the set it is computed on. In the pipeline tau is computed ONCE
    on the validation split and FROZEN before any test read. (Kth-order statistic
    on negative scores, matching orchestrator / npl_eval_fixed.threshold_at_fpr.)
  * recall_at: fraction of POSITIVES with score >= tau.
  * ece: confidence-binned with conf = max(p, 1-p) (Guo et al. 2017), 15 bins
    (matches the corrected ECE that took HG 0.70 -> 0.009).
  * bootstrap_ci: percentile CI, >=500 resamples, fixed seed.
  * mcnemar: exact/continuity-corrected on the discordant pair counts.

Pre-registered PRIMARY metric is recall_at(y, p, threshold_at_fpr(y_val, p_val, 0.01)).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "threshold_at_fpr",
    "recall_at",
    "fpr_at",
    "auroc",
    "auprc",
    "f1_at",
    "ece",
    "brier",
    "bootstrap_ci",
    "mcnemar",
    "BootstrapResult",
]


def _as_arrays(y, p):
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y and p must match: {y.shape} vs {p.shape}")
    return y, p


def threshold_at_fpr(y, p, target_fpr: float = 0.01) -> float:
    """Smallest threshold tau such that FPR(neg scores >= tau) <= target_fpr.

    Computed on whatever set is passed; in the pipeline this is the VALIDATION
    set and the returned tau is frozen for the test read. Kth-order statistic on
    the negative scores (identical to the orchestrator helper).
    """
    y, p = _as_arrays(y, p)
    neg = p[y == 0]
    if neg.size == 0:
        # No negatives to calibrate on: fall back to max score (recall 0).
        return float(np.nextafter(p.max(), np.inf)) if p.size else 1.0
    neg_sorted = np.sort(neg)  # ascending
    # We allow at most floor(target_fpr * n_neg) negatives to exceed tau.
    n_allowed = int(np.floor(target_fpr * neg.size))
    # Index of the highest negative we are NOT allowed to exceed (1-based from top).
    # tau set just above the (n_allowed+1)-th largest negative.
    k_from_top = n_allowed + 1
    if k_from_top > neg.size:
        # target_fpr large enough to admit all negatives -> threshold below min.
        return float(np.nextafter(neg_sorted[0], -np.inf))
    boundary = neg_sorted[neg.size - k_from_top]
    return float(np.nextafter(boundary, np.inf))


def recall_at(y, p, tau: float) -> float:
    """Fraction of positives scoring >= tau (TPR at the frozen threshold)."""
    y, p = _as_arrays(y, p)
    pos = p[y == 1]
    if pos.size == 0:
        return float("nan")
    return float((pos >= tau).mean())


def fpr_at(y, p, tau: float) -> float:
    """Realized FPR at threshold tau (negatives scoring >= tau)."""
    y, p = _as_arrays(y, p)
    neg = p[y == 0]
    if neg.size == 0:
        return float("nan")
    return float((neg >= tau).mean())


def _roc_points(y, p):
    y, p = _as_arrays(y, p)
    order = np.argsort(-p, kind="mergesort")
    y = y[order]
    P = max(int((y == 1).sum()), 1)
    N = max(int((y == 0).sum()), 1)
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    tpr = np.concatenate([[0.0], tps / P])
    fpr = np.concatenate([[0.0], fps / N])
    return fpr, tpr


def auroc(y, p) -> float:
    """Area under ROC via Mann-Whitney U (rank) statistic; ties handled."""
    y, p = _as_arrays(y, p)
    pos = p[y == 1]
    neg = p[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="mergesort")) + 1
    # Average ranks for ties for an exact AUC.
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allp = np.concatenate([pos, neg])[order]
    rank = np.empty(allp.size, dtype=float)
    i = 0
    while i < allp.size:
        j = i
        while j + 1 < allp.size and allp[j + 1] == allp[i]:
            j += 1
        rank[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    inv = np.empty_like(rank)
    inv[order] = rank
    rank_pos = inv[: pos.size]
    u = rank_pos.sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def auprc(y, p) -> float:
    """Average precision (area under precision-recall), step interpolation."""
    y, p = _as_arrays(y, p)
    if (y == 1).sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y = y[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / max(int((y == 1).sum()), 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum(np.diff(recall) * precision[1:]))


def f1_at(y, p, tau: float) -> float:
    """F1 at the frozen operating threshold tau (NOT at 0.5)."""
    y, p = _as_arrays(y, p)
    pred = (p >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def ece(y, p, n_bins: int = 15) -> float:
    """Expected Calibration Error, confidence = max(p, 1-p) (Guo et al. 2017)."""
    y, p = _as_arrays(y, p)
    conf = np.maximum(p, 1.0 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = y.size
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        total += abs(correct[m].mean() - conf[m].mean()) * m.sum() / n
    return float(total)


def brier(y, p) -> float:
    y, p = _as_arrays(y, p)
    return float(np.mean((p - y) ** 2))


@dataclass
class BootstrapResult:
    point: float
    lo: float
    hi: float
    n_resamples: int

    def as_row(self, prefix: str) -> dict:
        return {prefix: self.point, f"{prefix}_lo": self.lo, f"{prefix}_hi": self.hi}


def bootstrap_ci(
    y,
    p,
    metric_fn,
    n_resamples: int = 500,
    alpha: float = 0.05,
    seed: int = 1337,
    **metric_kwargs,
) -> BootstrapResult:
    """Stratified percentile bootstrap CI for any metric_fn(y, p, **kw) -> float.

    Resamples positives and negatives separately to preserve prevalence. Fixed
    seed for reproducibility. Used for R@1%FPR / AUROC / AUPRC / F1.
    """
    y, p = _as_arrays(y, p)
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    point = float(metric_fn(y, p, **metric_kwargs))
    stats = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        bi = np.concatenate(
            [
                rng.choice(pos_idx, size=pos_idx.size, replace=True),
                rng.choice(neg_idx, size=neg_idx.size, replace=True),
            ]
        )
        stats[b] = metric_fn(y[bi], p[bi], **metric_kwargs)
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return BootstrapResult(point, float("nan"), float("nan"), n_resamples)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return BootstrapResult(point, lo, hi, n_resamples)


def mcnemar(y, pred_a, pred_b, correction: bool = True) -> dict:
    """Paired McNemar test comparing two models' correctness against labels y.

    Returns {'b': n_a_right_b_wrong, 'c': n_a_wrong_b_right, 'statistic', 'pvalue'}.
    Used vs the strongest neural baseline AND vs B5 (require p < 0.01).
    """
    from scipy.stats import chi2

    y = np.asarray(y).astype(int).ravel()
    ca = (np.asarray(pred_a).astype(int).ravel() == y)
    cb = (np.asarray(pred_b).astype(int).ravel() == y)
    b = int((ca & ~cb).sum())  # a right, b wrong
    c = int((~ca & cb).sum())  # a wrong, b right
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "pvalue": 1.0}
    if n < 25:
        # Exact binomial two-sided.
        from scipy.stats import binomtest

        pvalue = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
        stat = float(min(b, c))
    else:
        cont = 1.0 if correction else 0.0
        stat = (abs(b - c) - cont) ** 2 / (b + c)
        pvalue = float(chi2.sf(stat, df=1))
    return {"b": b, "c": c, "statistic": float(stat), "pvalue": float(pvalue)}
