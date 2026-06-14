"""
Shared helpers for the CANOPI figure generators (scripts/make_canopi_*_figure.py).

Conventions match the existing make_*_figure.py scripts: read a CSV from
paper/paper_v2_extract/canopi/, write a {.png,.pdf} pair to paper/figures/<name>,
grayscale-legible (distinct hatches + markers, not color alone).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRACT = REPO_ROOT / "paper" / "paper_v2_extract" / "canopi"
FIGDIR = REPO_ROOT / "paper" / "figures"

# Grayscale-safe: every series distinguished by hatch + marker, color is secondary.
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "++", "ooo"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
GRAYS = ["#222222", "#555555", "#888888", "#aaaaaa", "#404040", "#6f6f6f", "#999999"]


def mean_std(df, metric: str):
    """Return (mean, std) columns for `metric`, tolerating *_mean/*_std,
    *_lo/*_hi (CI -> half-width std proxy), or a bare point column."""
    import numpy as np

    if f"{metric}_mean" in df.columns:
        mean = df[f"{metric}_mean"].to_numpy(float)
        std = df[f"{metric}_std"].to_numpy(float) if f"{metric}_std" in df.columns else np.zeros(len(df))
        return mean, std
    if f"{metric}_lo" in df.columns and f"{metric}_hi" in df.columns:
        lo, hi = df[f"{metric}_lo"].to_numpy(float), df[f"{metric}_hi"].to_numpy(float)
        mid = df[metric].to_numpy(float) if metric in df.columns else (lo + hi) / 2
        return mid, (hi - lo) / 2
    return df[metric].to_numpy(float), np.zeros(len(df))


def save(fig, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = out_stem.with_suffix(".png"), out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"saved {png.relative_to(REPO_ROOT)} + {pdf.name}")


def require(csv: Path):
    if not csv.exists():
        raise SystemExit(f"input CSV not found: {csv}\n  (run the CANOPI orchestrator notebook first)")
    import pandas as pd

    return pd.read_csv(csv)
