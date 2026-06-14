"""
canopi.losses
=============
The JOINT objective. Four config-toggleable terms; the combination is the
claimed novelty (pAUC couples invariance to the deployment operating point):

    L = lam1 * L_inv      multi-view supervised contrastive (same-intent views,
                          incl. AR/ES back-translations, pulled together)
      + lam2 * L_hardneg  push attacks away from benign paraphrases that SHARE a
                          trigger token (e.g. "ignore")
      + lam3 * L_drift    penalize ||P(x) - P(view)|| across transform families
      + lam4 * L_pAUC     partial-AUC / top-push surrogate over FPR in [0, beta]

Each term has a pure-numpy reference (`*_np`, exercised by the laptop unit tests)
and a torch version (`*_torch`, used by train.py for autograd on the GPU). The
formulas are identical; the numpy refs are the executable spec.

Baselines are just lambda settings:
    B5 plain embedding clf : lam1=lam2=lam3=0, lam4=0 (detector head only)
    B6 invariance-only     : lam4=0
    B7 pAUC-only           : lam1=lam2=lam3=0
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "supcon_inv_np",
    "hardneg_np",
    "drift_np",
    "pauc_toppush_np",
    "supcon_inv_torch",
    "hardneg_torch",
    "drift_torch",
    "pauc_toppush_torch",
    "joint_loss_torch",
    "LossWeights",
]


# ---------------------------------------------------------------------------
# numpy reference implementations (the executable spec; used by unit tests)
# ---------------------------------------------------------------------------

def supcon_inv_np(z: np.ndarray, groups: Sequence[int], tau: float = 0.1) -> float:
    """Supervised contrastive loss (Khosla et al. 2020), L2-normalized z [n, d].

    `groups[i]` is the intent/anchor id; all views of one intent (paraphrase,
    AR/ES back-translation, persona, encoding) share a group id and are pulled
    together. Returns mean loss over anchors that have >=1 positive partner.
    """
    z = np.asarray(z, dtype=float)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    g = np.asarray(groups)
    n = z.shape[0]
    sim = (z @ z.T) / tau
    np.fill_diagonal(sim, -np.inf)  # exclude self from denominator
    losses = []
    for i in range(n):
        pos_mask = (g == g[i])
        pos_mask[i] = False
        if not pos_mask.any():
            continue
        logits = sim[i]
        m = np.max(logits[np.isfinite(logits)])
        denom = np.log(np.sum(np.exp(logits - m)) + 1e-12) + m
        log_prob = sim[i][pos_mask] - denom
        losses.append(-np.mean(log_prob))
    return float(np.mean(losses)) if losses else 0.0


def hardneg_np(z_attack: np.ndarray, z_benign: np.ndarray, margin: float = 0.2) -> float:
    """Hinge pushing trigger-sharing (attack, benign) pairs apart in cos space.

    z_attack[i], z_benign[i] share a trigger token but differ in intent; we
    penalize cosine similarity above `margin`. Returns mean over pairs.
    """
    a = np.asarray(z_attack, dtype=float)
    b = np.asarray(z_benign, dtype=float)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    cos = np.sum(a * b, axis=1)
    return float(np.mean(np.maximum(cos - margin, 0.0)))


def drift_np(z_anchor: np.ndarray, z_view: np.ndarray) -> float:
    """Invariance regularizer: mean squared L2 distance between anchor and view."""
    a = np.asarray(z_anchor, dtype=float)
    v = np.asarray(z_view, dtype=float)
    return float(np.mean(np.sum((a - v) ** 2, axis=1)))


def pauc_toppush_np(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    beta: float = 0.01,
    margin: float = 1.0,
) -> float:
    """Partial-AUC (low-FPR) top-push surrogate over FPR in [0, beta].

    Only the top ceil(beta * n_neg) hardest negatives matter — these are exactly
    the negatives that set the 1%-FPR threshold. Squared hinge pushes every
    positive above each top negative by `margin`. Minimizing this directly
    optimizes the deployment operating point. (DeepTopPush / SOPA-style.)
    """
    sp = np.asarray(pos_scores, dtype=float).ravel()
    sn = np.asarray(neg_scores, dtype=float).ravel()
    if sp.size == 0 or sn.size == 0:
        return 0.0
    k = max(1, int(np.ceil(beta * sn.size)))
    top_neg = np.sort(sn)[-k:]  # k hardest negatives
    diff = margin - (sp[:, None] - top_neg[None, :])  # [n_pos, k]
    return float(np.mean(np.maximum(diff, 0.0) ** 2))


# ---------------------------------------------------------------------------
# torch implementations (autograd; used by train.py). Lazy import.
# ---------------------------------------------------------------------------

def _torch():
    import torch  # lazy

    return torch


def supcon_inv_torch(z, groups, tau: float = 0.1):
    torch = _torch()
    z = torch.nn.functional.normalize(z, dim=1)
    g = groups if torch.is_tensor(groups) else torch.as_tensor(np.asarray(groups), device=z.device)
    n = z.shape[0]
    sim = (z @ z.t()) / tau
    diag = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag, float("-inf"))
    eq = g.unsqueeze(0) == g.unsqueeze(1)
    pos_mask = eq & ~diag
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return z.sum() * 0.0
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count
    return -(mean_log_prob_pos[has_pos].mean())


def hardneg_torch(z_attack, z_benign, margin: float = 0.2):
    torch = _torch()
    a = torch.nn.functional.normalize(z_attack, dim=1)
    b = torch.nn.functional.normalize(z_benign, dim=1)
    cos = (a * b).sum(dim=1)
    return torch.clamp(cos - margin, min=0.0).mean()


def drift_torch(z_anchor, z_view):
    return ((z_anchor - z_view) ** 2).sum(dim=1).mean()


def pauc_toppush_torch(pos_scores, neg_scores, beta: float = 0.01, margin: float = 1.0):
    torch = _torch()
    sp = pos_scores.ravel()
    sn = neg_scores.ravel()
    if sp.numel() == 0 or sn.numel() == 0:
        return sp.sum() * 0.0
    k = max(1, int(np.ceil(beta * sn.numel())))
    top_neg, _ = torch.topk(sn, k)
    diff = margin - (sp.unsqueeze(1) - top_neg.unsqueeze(0))
    return torch.clamp(diff, min=0.0).pow(2).mean()


class LossWeights:
    """lam1..lam4 with on/off semantics. A term is OFF iff its lambda == 0."""

    def __init__(self, lam1=1.0, lam2=1.0, lam3=1.0, lam4=1.0,
                 tau=0.1, hardneg_margin=0.2, beta=0.01, pauc_margin=1.0):
        self.lam1, self.lam2, self.lam3, self.lam4 = lam1, lam2, lam3, lam4
        self.tau = tau
        self.hardneg_margin = hardneg_margin
        self.beta = beta
        self.pauc_margin = pauc_margin

    def active(self) -> dict:
        return {"inv": self.lam1 != 0, "hardneg": self.lam2 != 0,
                "drift": self.lam3 != 0, "pauc": self.lam4 != 0}

    @classmethod
    def from_config(cls, cfg: dict) -> "LossWeights":
        loss = dict(cfg.get("loss", {}))
        return cls(
            lam1=loss.get("lam1_inv", loss.get("lam1", 1.0)),
            lam2=loss.get("lam2_hardneg", loss.get("lam2", 1.0)),
            lam3=loss.get("lam3_drift", loss.get("lam3", 1.0)),
            lam4=loss.get("lam4_pauc", loss.get("lam4", 1.0)),
            tau=loss.get("tau", 0.1),
            hardneg_margin=loss.get("hardneg_margin", 0.2),
            beta=loss.get("beta", 0.01),
            pauc_margin=loss.get("pauc_margin", 1.0),
        )


def joint_loss_torch(batch: dict, w: LossWeights):
    """Combine active terms. `batch` carries the tensors each active term needs:
        z_all, groups            -> L_inv
        z_attack, z_benign       -> L_hardneg
        z_anchor, z_view         -> L_drift
        pos_scores, neg_scores   -> L_pAUC
    Returns (total_loss, components_dict). Missing inputs for an inactive term
    are simply skipped.
    """
    torch = _torch()
    comps = {}
    total = None

    def _add(val, lam, key):
        nonlocal total
        comps[key] = float(val.detach().cpu()) if torch.is_tensor(val) else float(val)
        term = lam * val
        total = term if total is None else total + term

    a = w.active()
    if a["inv"] and "z_all" in batch:
        _add(supcon_inv_torch(batch["z_all"], batch["groups"], w.tau), w.lam1, "L_inv")
    if a["hardneg"] and "z_attack" in batch:
        _add(hardneg_torch(batch["z_attack"], batch["z_benign"], w.hardneg_margin), w.lam2, "L_hardneg")
    if a["drift"] and "z_anchor" in batch:
        _add(drift_torch(batch["z_anchor"], batch["z_view"]), w.lam3, "L_drift")
    if a["pauc"] and "pos_scores" in batch:
        _add(pauc_toppush_torch(batch["pos_scores"], batch["neg_scores"], w.beta, w.pauc_margin), w.lam4, "L_pAUC")

    if total is None:
        total = batch.get("pos_scores", batch.get("z_all")).sum() * 0.0
    comps["L_total"] = float(total.detach().cpu()) if torch.is_tensor(total) else float(total)
    return total, comps
