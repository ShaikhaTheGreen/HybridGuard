"""
canopi.train
============
Training driver for one CANOPI config / one seed. Reuses the orchestrator's
protocol: frozen encoder, joint loss, val-frozen tau at 1% FPR, per-seed
checkpoint, results written in the runs/<run_id>/ convention.

Heavy path (torch + a semantic encoder) runs on the Colab GPU. The function is
structured so the CANOPI orchestrator notebook can call `train_one_seed(cfg,
data, seed)` inside the existing 5-seed loop with the existing X_/y_ splits.

Config (YAML under configs/) drives encoder backend, head geometry, loss lambdas
(a baseline is just a lambda setting), augmentation families, and epochs.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .augment import IntentPreservationFilter, NLLBTranslator, TransformationBank, build_view_set
from .encoders import get_encoder
from .losses import LossWeights, joint_loss_torch
from .metrics import recall_at, threshold_at_fpr
from .model import CanopiModel

__all__ = ["load_config", "set_determinism", "train_one_seed", "TrainResult"]


def load_config(path: str) -> dict:
    import yaml  # lazy ([full] extra)

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_determinism(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


@dataclass
class TrainResult:
    model: CanopiModel
    tau: float
    val_recall_at_1pct: float
    history: List[dict] = field(default_factory=list)
    accept_rates: dict = field(default_factory=dict)
    aligned_pairs: list = field(default_factory=list)


def _hardneg_pairs(view_texts, view_family, labels):
    """Pair each attack (label 1) with a benign (label 0) sharing a trigger token."""
    triggers = ("ignore", "disregard", "system", "prompt", "reveal", "override")
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    pairs = []
    for i in pos:
        ti = next((t for t in triggers if t in view_texts[i].lower()), None)
        if ti is None:
            continue
        for j in neg:
            if ti in view_texts[j].lower():
                pairs.append((i, j))
                break
    return pairs


def train_one_seed(cfg: dict, data: dict, seed: int) -> TrainResult:
    """Train CANOPI for one seed.

    data: {'X_train','y_train','X_val','y_val'} (lists/arrays of texts/labels).
    Optional 'X_ar','y_ar','X_es','y_es' curated cross-lingual (held out from tau).
    """
    import torch

    set_determinism(seed)
    tcfg = cfg.get("train", {})
    weights = LossWeights.from_config(cfg)

    # 1. Frozen encoder.
    encoder = get_encoder(cfg.get("encoder", {"backend": "auto"}))

    # 2. Build augmented view set (training-time only).
    translator = None
    if cfg.get("augment", {}).get("crosslingual") and weights.active()["inv"]:
        translator = NLLBTranslator(model_name=cfg["augment"].get("nllb", "facebook/nllb-200-distilled-600M"))
    bank = TransformationBank(translator=translator, seed=seed)
    filt = None
    if cfg.get("augment", {}).get("intent_filter", True):
        filt = IntentPreservationFilter(encoder, tau_keep=cfg.get("augment", {}).get("tau_keep", 0.5))
    vs = build_view_set(
        list(data["X_train"]), list(data["y_train"]), bank, filt,
        families=cfg.get("augment", {}).get("families", ("paraphrase", "persona", "encoding:homoglyph")),
        crosslingual=cfg.get("augment", {}).get("crosslingual", ()),
        crosslingual_max=cfg.get("augment", {}).get("crosslingual_max", 300),
    )

    # 3. Precompute frozen embeddings for all view rows + anchors once.
    Z = encoder.encode(vs["view_texts"])  # [N, d]
    in_dim = Z.shape[1]
    groups = np.asarray(vs["group_ids"])
    vlabels = np.asarray(vs["labels"])
    hn_pairs = _hardneg_pairs(vs["view_texts"], vs["view_family"], vs["labels"])
    # drift pairs: anchor (family == 'anchor') vs each non-anchor view in same group
    anchor_row = {g: i for i, (g, f) in enumerate(zip(vs["group_ids"], vs["view_family"])) if f == "anchor"}
    drift_pairs = [(anchor_row[g], i) for i, (g, f) in enumerate(zip(vs["group_ids"], vs["view_family"]))
                   if f != "anchor" and g in anchor_row]

    # 4. Model + optimizer.
    model = CanopiModel(
        encoder=encoder, in_dim=in_dim,
        hidden=cfg.get("model", {}).get("hidden", 256),
        out_dim=cfg.get("model", {}).get("out_dim", 128),
        depth=cfg.get("model", {}).get("head_depth", 2),
        dropout=cfg.get("model", {}).get("dropout", 0.1),
    )
    dev = model._device
    opt = torch.optim.AdamW(model.net.parameters(), lr=tcfg.get("lr", 1e-3), weight_decay=tcfg.get("wd", 1e-4))
    bce = torch.nn.BCEWithLogitsLoss()
    Zt = torch.as_tensor(Z, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(vlabels, dtype=torch.float32, device=dev)
    rng = np.random.default_rng(seed)
    epochs = tcfg.get("epochs", 30)
    bs = tcfg.get("batch_size", 256)
    history = []

    for ep in range(epochs):
        model.net.train()
        idx = rng.permutation(len(Z))
        for s in range(0, len(idx), bs):
            bi = idx[s : s + bs]
            e = Zt[bi]
            z = model.project_tensor(e)
            logits = model.net.det(z)
            batch = {"z_all": z, "groups": torch.as_tensor(groups[bi], device=dev)}
            # detector supervision (always on) — anchors the score scale
            sup = bce(logits, yt[bi])
            # pAUC over this batch's pos/neg logits
            if weights.active()["pauc"]:
                pos = logits[yt[bi] == 1]
                neg = logits[yt[bi] == 0]
                batch["pos_scores"], batch["neg_scores"] = pos, neg
            # hard-neg pairs intersecting the batch
            if weights.active()["hardneg"] and hn_pairs:
                bset = set(bi.tolist())
                sel = [(a, b) for a, b in hn_pairs if a in bset and b in bset]
                if sel:
                    ia = torch.as_tensor([a for a, _ in sel], device=dev)
                    ib = torch.as_tensor([b for _, b in sel], device=dev)
                    batch["z_attack"] = model.project_tensor(Zt[ia])
                    batch["z_benign"] = model.project_tensor(Zt[ib])
            # drift pairs intersecting the batch
            if weights.active()["drift"] and drift_pairs:
                bset = set(bi.tolist())
                sel = [(a, b) for a, b in drift_pairs if a in bset and b in bset]
                if sel:
                    ia = torch.as_tensor([a for a, _ in sel], device=dev)
                    ib = torch.as_tensor([b for _, b in sel], device=dev)
                    batch["z_anchor"] = model.project_tensor(Zt[ia])
                    batch["z_view"] = model.project_tensor(Zt[ib])
            joint, comps = joint_loss_torch(batch, weights)
            loss = sup + joint
            opt.zero_grad()
            loss.backward()
            opt.step()
        history.append({"epoch": ep, "sup": float(sup.detach().cpu()), **comps})

    # 5. Freeze tau on validation at 1% FPR (BEFORE any test read).
    val_p = model.score(list(data["X_val"]))
    yv = np.asarray(data["y_val"]).astype(int)
    tau = threshold_at_fpr(yv, val_p, cfg.get("eval", {}).get("target_fpr", 0.01))
    val_recall = recall_at(yv, val_p, tau)

    return TrainResult(
        model=model, tau=tau, val_recall_at_1pct=val_recall, history=history,
        accept_rates=vs["accept_rates"], aligned_pairs=vs["aligned_pairs"],
    )
