"""
canopi.data
===========
Importable re-implementation of the HybridGuard orchestrator's data protocol so
the CANOPI notebook runs standalone on Colab while staying NUMERICALLY COMPARABLE
to the published splits: SHA-256 exact dedup, SimHash near-dup removal, a
deterministic stratified 60/20/20 split at seed 1337, and a train/val/test
leakage report (must be empty).

If the orchestrator has already populated X_train/.../y_test in the notebook
globals, reuse those instead of calling this — same protocol, same seed. The
pure-logic parts (dedup, simhash, split, leakage) are numpy/pandas only and unit
tested; HF dataset loads lazy-import `datasets`.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from hybridguard.canonicalize import canonicalize

__all__ = [
    "Splits",
    "sha256_dedup",
    "simhash_dedup",
    "stratified_split",
    "leakage_report",
    "prepare_dataset",
    "load_xtram1",
    "load_deepset",
    "load_notinject",
    "load_jbb",
    "mt_multilingual_testset",
]

SPLIT_SEED = 1337
SPLIT_RATIOS = (0.6, 0.2, 0.2)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_dedup(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Drop exact duplicates by SHA-256 of the raw text (first occurrence kept)."""
    h = df[text_col].astype(str).map(_sha256)
    return df.loc[~h.duplicated()].reset_index(drop=True)


def _simhash(text: str, bits: int = 64) -> int:
    """64-bit SimHash over word shingles (size 2). Pure python, deterministic."""
    tokens = re.findall(r"\w+", text.lower())
    shingles = [" ".join(tokens[i : i + 2]) for i in range(max(len(tokens) - 1, 1))] or tokens or [text]
    v = [0] * bits
    for sh in shingles:
        hv = int.from_bytes(hashlib.blake2b(sh.encode("utf-8"), digest_size=8).digest(), "little")
        for b in range(bits):
            v[b] += 1 if (hv >> b) & 1 else -1
    out = 0
    for b in range(bits):
        if v[b] > 0:
            out |= 1 << b
    return out


def simhash_dedup(df: pd.DataFrame, text_col: str = "text", max_hamming: int = 3) -> pd.DataFrame:
    """Remove near-duplicates whose SimHash is within `max_hamming` bits of a kept
    row. O(n^2) worst case; fine for the paper's corpus sizes. First occurrence kept."""
    hashes = df[text_col].astype(str).map(_simhash).tolist()
    keep, kept_hashes = [], []
    for i, h in enumerate(hashes):
        dup = any((h ^ kh).bit_count() <= max_hamming for kh in kept_hashes)
        if not dup:
            keep.append(i)
            kept_hashes.append(h)
    return df.iloc[keep].reset_index(drop=True)


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    indices: Dict[str, List[int]] = field(default_factory=dict)

    def xy(self, split: str, text_col="text", label_col="label"):
        df = getattr(self, split)
        return df[text_col].tolist(), df[label_col].astype(int).tolist()


def stratified_split(df: pd.DataFrame, label_col: str = "label",
                     ratios: Tuple[float, float, float] = SPLIT_RATIOS,
                     seed: int = SPLIT_SEED) -> Splits:
    """Deterministic stratified 60/20/20 split at seed 1337 (orchestrator protocol)."""
    rng = np.random.default_rng(seed)
    tr_idx, va_idx, te_idx = [], [], []
    for lab, grp in df.groupby(label_col):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(ratios[0] * n))
        n_va = int(round(ratios[1] * n))
        tr_idx += idx[:n_tr].tolist()
        va_idx += idx[n_tr : n_tr + n_va].tolist()
        te_idx += idx[n_tr + n_va :].tolist()
    return Splits(
        train=df.loc[tr_idx].reset_index(drop=True),
        val=df.loc[va_idx].reset_index(drop=True),
        test=df.loc[te_idx].reset_index(drop=True),
        indices={"train": sorted(tr_idx), "val": sorted(va_idx), "test": sorted(te_idx)},
    )


def leakage_report(splits: Splits, text_col: str = "text") -> dict:
    """SHA-256 overlap across splits. The 'overlaps' lists MUST be empty."""
    h = {s: set(getattr(splits, s)[text_col].astype(str).map(_sha256)) for s in ("train", "val", "test")}
    return {
        "train_val": sorted(h["train"] & h["val"]),
        "train_test": sorted(h["train"] & h["test"]),
        "val_test": sorted(h["val"] & h["test"]),
        "n_train": len(h["train"]), "n_val": len(h["val"]), "n_test": len(h["test"]),
        "clean": not (h["train"] & h["val"] or h["train"] & h["test"] or h["val"] & h["test"]),
    }


def prepare_dataset(df: pd.DataFrame, text_col="text", label_col="label",
                    do_simhash: bool = True, seed: int = SPLIT_SEED) -> Tuple[Splits, dict]:
    """Full pipeline: dedup -> simhash -> split -> leakage report. Returns (Splits, report)."""
    df = df[[text_col, label_col]].dropna().reset_index(drop=True)
    df = sha256_dedup(df, text_col)
    if do_simhash:
        df = simhash_dedup(df, text_col)
    splits = stratified_split(df, label_col, seed=seed)
    return splits, leakage_report(splits, text_col)


# --- HF dataset loaders (lazy) ---------------------------------------------

def _coerce_label(v) -> int:
    if isinstance(v, str):
        return 1 if v.strip().lower() in {"1", "true", "injection", "jailbreak", "malicious", "unsafe"} else 0
    return int(bool(v))


def load_xtram1() -> pd.DataFrame:
    """Primary corpus: xTRam1/safe-guard-prompt-injection -> df[text,label]."""
    from datasets import load_dataset

    ds = load_dataset("xTRam1/safe-guard-prompt-injection")
    parts = [pd.DataFrame(ds[s]) for s in ds.keys()]
    df = pd.concat(parts, ignore_index=True)
    txt = "text" if "text" in df.columns else df.columns[0]
    lab = "label" if "label" in df.columns else df.columns[-1]
    return pd.DataFrame({"text": df[txt].astype(str), "label": df[lab].map(_coerce_label)})


def load_deepset() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("deepset/prompt-injections")
    df = pd.concat([pd.DataFrame(ds[s]) for s in ds.keys()], ignore_index=True)
    return pd.DataFrame({"text": df["text"].astype(str), "label": df["label"].map(_coerce_label)})


def load_notinject() -> pd.DataFrame:
    """Over-defense benign set (all label 0)."""
    from datasets import load_dataset

    ds = load_dataset("leolee99/NotInject")
    df = pd.concat([pd.DataFrame(ds[s]) for s in ds.keys()], ignore_index=True)
    col = "text" if "text" in df.columns else df.columns[0]
    return pd.DataFrame({"text": df[col].astype(str), "label": 0})


def load_jbb() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    df = pd.concat([pd.DataFrame(ds[s]) for s in ds.keys()], ignore_index=True)
    col = "Goal" if "Goal" in df.columns else ("Behavior" if "Behavior" in df.columns else df.columns[0])
    return pd.DataFrame({"text": df[col].astype(str), "label": 1})


def mt_multilingual_testset(translator, en_pos_texts, langs, src: str = "en", max_n: int = 200):
    """Scalable held-out multilingual eval (E3): translate English POSITIVE test
    injections into each target language via NLLB. Held out from training and from
    tau (tau is frozen on the English val split). Synthetic — flag vs the curated
    AR/ES gold. `translator` is a duck-typed object with .translate(texts, src, tgt).
    Returns {f'{lang}_mt': (texts, [1,...])}; recall is measured on these positives
    at the English-frozen tau, so labels are all 1.
    """
    en = list(en_pos_texts)[:max_n]
    out = {}
    for lang in langs:
        if lang == src:
            continue
        translated = translator.translate(en, src, lang)
        out[f"{lang}_mt"] = (translated, [1] * len(translated))
    return out
