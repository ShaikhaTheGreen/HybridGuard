"""
canopi.eval
===========
The extended evaluation protocol (E1, E3-E7, E10), decoupled from torch via a
`score_fn: texts -> np.ndarray[0,1]` so each function is testable on CPU and
reused verbatim by the CANOPI orchestrator notebook. tau is ALWAYS frozen on
validation (passed in); no function re-thresholds on its own test set.

All metric definitions come from canopi.metrics (the orchestrator's frozen
protocol). Outputs are plain dict rows / DataFrames written via canopi.runs.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import metrics as M

__all__ = [
    "main_metrics",
    "crosslingual_at_tau",
    "threshold_transfer",
    "overdefense_sweep",
    "calibration",
    "temperature_scale",
    "invariance_drift",
    "intent_preservation_gap",
    "adaptive_residual",
]

ScoreFn = Callable[[Sequence[str]], np.ndarray]


def main_metrics(y_val, p_val, y_test, p_test, model_name: str,
                 target_fpr: float = 0.01, n_boot: int = 1000, seed: int = 1337) -> dict:
    """E1 row: tau frozen on val; R@1%FPR / AUROC / AUPRC / F1@op on test, all
    with stratified bootstrap CIs. The PRIMARY metric is recall_at_1pctfpr."""
    y_val = np.asarray(y_val).astype(int)
    y_test = np.asarray(y_test).astype(int)
    tau = M.threshold_at_fpr(y_val, p_val, target_fpr)
    r = M.bootstrap_ci(y_test, p_test, M.recall_at, n_resamples=n_boot, seed=seed, tau=tau)
    au = M.bootstrap_ci(y_test, p_test, M.auroc, n_resamples=n_boot, seed=seed)
    ap = M.bootstrap_ci(y_test, p_test, M.auprc, n_resamples=n_boot, seed=seed)
    f1 = M.bootstrap_ci(y_test, p_test, M.f1_at, n_resamples=n_boot, seed=seed, tau=tau)
    row = {"model": model_name, "tau": tau, "fpr_test": M.fpr_at(y_test, p_test, tau)}
    row.update(r.as_row("recall_at_1pctfpr"))
    row.update(au.as_row("auroc"))
    row.update(ap.as_row("auprc"))
    row.update(f1.as_row("f1_at_op"))
    return row


def crosslingual_at_tau(scores_by_lang: Dict[str, tuple], tau: float, model_name: str,
                        n_boot: int = 1000, seed: int = 1337) -> pd.DataFrame:
    """E3 HEADLINE: R@1%FPR per language at the SAME frozen tau. scores_by_lang =
    {lang: (y, p)}. Returns rows with bootstrap CIs (the ~0.000 -> non-zero plot)."""
    rows = []
    for lang, (y, p) in scores_by_lang.items():
        y = np.asarray(y).astype(int)
        r = M.bootstrap_ci(y, np.asarray(p), M.recall_at, n_resamples=n_boot, seed=seed, tau=tau)
        row = {"model": model_name, "lang": lang, "tau": tau, "n_pos": int((y == 1).sum())}
        row.update(r.as_row("recall_at_1pctfpr"))
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_transfer(corpora_scores: Dict[str, tuple], tau: float, model_name: str) -> pd.DataFrame:
    """E4: apply the xTRam1-frozen tau to other corpora. Report realized FPR
    (drift) and recall at tau. corpora_scores = {corpus: (y, p)}."""
    rows = []
    for corpus, (y, p) in corpora_scores.items():
        y = np.asarray(y).astype(int)
        p = np.asarray(p)
        rows.append({
            "model": model_name, "corpus": corpus, "tau": tau,
            "fpr_at_tau": M.fpr_at(y, p, tau),
            "recall_at_tau": M.recall_at(y, p, tau),
            "auroc": M.auroc(y, p),
        })
    return pd.DataFrame(rows)


def overdefense_sweep(y_val, p_val, y_bench, p_bench, model_name: str,
                      fpr_targets: Sequence[float] = (0.01, 0.02, 0.05, 0.10)) -> pd.DataFrame:
    """E5: for each in-domain FPR target, freeze tau on val, measure benign FPR on
    the hard-negative bench (NotInject). Lower is better (less over-defense)."""
    y_val = np.asarray(y_val).astype(int)
    y_bench = np.asarray(y_bench).astype(int)
    rows = []
    for t in fpr_targets:
        tau = M.threshold_at_fpr(y_val, p_val, t)
        rows.append({"model": model_name, "fpr_target": t, "tau": tau,
                     "overdefense_fpr": M.fpr_at(y_bench, p_bench, tau)})
    return pd.DataFrame(rows)


def temperature_scale(y_val, logit_val, y_test, logit_test) -> tuple:
    """Fit a single temperature T on val NLL; return (T, p_test_scaled)."""
    from scipy.optimize import minimize_scalar

    yv = np.asarray(y_val).astype(float)
    zv = np.asarray(logit_val, dtype=float)

    def nll(T):
        p = 1.0 / (1.0 + np.exp(-zv / max(T, 1e-3)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))

    T = float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)
    p_test = 1.0 / (1.0 + np.exp(-np.asarray(logit_test, dtype=float) / T))
    return T, p_test


def calibration(y, p, model_name: str, logit_val=None, y_val=None, logit_test=None) -> dict:
    """E10: ECE + Brier; if logits provided, also post-temperature-scaling."""
    row = {"model": model_name, "ece": M.ece(y, p), "brier": M.brier(y, p)}
    if logit_val is not None and logit_test is not None and y_val is not None:
        T, p_scaled = temperature_scale(y_val, logit_val, y, logit_test)
        row.update({"temperature": T, "ece_post": M.ece(y, p_scaled), "brier_post": M.brier(y, p_scaled)})
    return row


def invariance_drift(project_fn: Callable[[Sequence[str]], np.ndarray],
                     anchors: Sequence[str], views_by_family: Dict[str, Sequence[str]],
                     model_name: str) -> pd.DataFrame:
    """E6: delta = mean ||P(anchor) - P(view)|| per transform family (lower = more
    invariant). Run before/after training (pass the relevant project_fn)."""
    za = project_fn(list(anchors))
    rows = []
    for fam, views in views_by_family.items():
        zv = project_fn(list(views))
        delta = float(np.mean(np.linalg.norm(za - zv, axis=1)))
        rows.append({"model": model_name, "family": fam, "drift_delta": delta})
    return pd.DataFrame(rows)


def intent_preservation_gap(score_fn: ScoreFn, anchors: Sequence[str],
                            preserved_views: Sequence[str], model_name: str) -> dict:
    """E6: mean |s(anchor) - s(intent-preserving view)|. Small = score stable
    under meaning-preserving transforms."""
    sa = score_fn(list(anchors))
    sv = score_fn(list(preserved_views))
    return {"model": model_name, "intent_preservation_gap": float(np.mean(np.abs(sa - sv)))}


def adaptive_residual(score_fn: ScoreFn, attacker, pos_texts: Sequence[str],
                      neg_for_tau: tuple, strengths: Sequence[float], tau: float,
                      model_name: str) -> pd.DataFrame:
    """E7: R@1%FPR of malicious prompts as the adaptive semantic-rewrite strength
    increases (the residual curve). tau frozen; we DO NOT hide the residual.
    neg_for_tau=(y_neg, ignored) only used if recomputing fpr; here tau is fixed."""
    sweep = attacker.strength_sweep(list(pos_texts), strengths)
    rows = []
    for s, rewritten in sweep.items():
        p = score_fn(rewritten)
        y = np.ones(len(rewritten), dtype=int)
        rows.append({"model": model_name, "rewrite_strength": s,
                     "recall_at_1pctfpr": float((p >= tau).mean())})
    return pd.DataFrame(rows)
