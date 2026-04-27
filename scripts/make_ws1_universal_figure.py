"""Reproducible figure: WS1 universal recovery across detectors.

Usage:
    # From repo root, with default paths:
    python scripts/make_ws1_universal_figure.py

    # With explicit paths:
    python scripts/make_ws1_universal_figure.py \\
        --csv  paper/paper_v2_extract/ws1_universal/results.csv \\
        --out  paper/figures/ws1_universal_recovery

The figure has TWO panels:
    (1) Main panel: Recall@1%FPR per detector under three conditions
        (clean / perturbed / perturbed + canonical) drawn as connected
        lines so the V-shape is the visual centerpiece. The InjecGuard
        recovery (~24x) is annotated as the headline.
    (2) Inset / small panel: AUROC under the same three conditions, for
        the threshold-independent ranking sanity check.

The V-shape on InjecGuard, regex, and TF-IDF in panel (1) is the central
empirical claim of Section 5.11; DeBERTa-v3 stays flat (already robust),
which is the safety-property claim.

LaTeX in main.tex references the figure stem (no extension), so pdflatex
picks the PDF automatically.

Source data: HybridGuard_WS1_Universal_Eval.ipynb run 2026-04-25. The
notebook saves a CSV with the schema:

    detector,condition,auroc,recall_at_1pctfpr

where condition is one of {clean, perturbed, perturbed_canonical}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "paper" / "paper_v2_extract" / "ws1_universal" / "results.csv"
DEFAULT_OUT = REPO_ROOT / "paper" / "figures" / "ws1_universal_recovery"

# Per-detector colors. InjecGuard gets the highlight color since it's the
# headline recovery story.
DETECTOR_COLORS = {
    "InjecGuard":      "#d62728",   # accent red — headline
    "tfidf_linearsvm": "#1f77b4",   # blue
    "regex_heuristic": "#7f7f7f",   # neutral gray
    "DeBERTa-v3":      "#2ca02c",   # green — the "already robust" outlier
}
DETECTOR_DISPLAY = {
    "InjecGuard":      "InjecGuard",
    "tfidf_linearsvm": "TF-IDF + LinearSVM",
    "regex_heuristic": "Regex heuristic",
    "DeBERTa-v3":      "DeBERTa-v3 (SOTA)",
}
DETECTOR_ORDER = ["InjecGuard", "tfidf_linearsvm", "regex_heuristic", "DeBERTa-v3"]

CONDITION_KEYS = ["clean", "perturbed", "perturbed_canonical"]
CONDITION_LABELS = ["clean", "perturbed", "perturbed\n+ canon."]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = {"detector", "condition", "auroc", "recall_at_1pctfpr"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV at {csv_path} is missing columns {missing}. Got: {list(df.columns)}"
        )
    bad = set(df["condition"]) - set(CONDITION_KEYS)
    if bad:
        raise ValueError(f"Unrecognized condition values: {bad}.")
    return df


def _values(df: pd.DataFrame, detector: str, metric: str) -> list[float]:
    return [
        float(df[(df.detector == detector) & (df.condition == c)][metric].iloc[0])
        for c in CONDITION_KEYS
    ]


def render(df: pd.DataFrame, out_stem: Path) -> None:
    fig = plt.figure(figsize=(12.0, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 0.05, 1.0])
    ax_recall = fig.add_subplot(gs[0, 0])
    ax_auroc = fig.add_subplot(gs[0, 2])

    x = np.arange(len(CONDITION_KEYS))

    # ---------- Main panel: Recall@1%FPR ----------
    for det in DETECTOR_ORDER:
        if det not in df.detector.values:
            continue
        ys = _values(df, det, "recall_at_1pctfpr")
        color = DETECTOR_COLORS[det]
        is_headline = det == "InjecGuard"
        ax_recall.plot(
            x, ys,
            marker="o",
            markersize=11 if is_headline else 8,
            linewidth=3.0 if is_headline else 1.8,
            color=color,
            label=DETECTOR_DISPLAY[det],
            zorder=4 if is_headline else 2,
        )
        # Value labels at each point
        for xi, yi in zip(x, ys):
            offset = 0.04 if is_headline else 0.025
            ax_recall.annotate(
                f"{yi:.3f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 9 if yi < 0.9 else -16),
                ha="center", va="bottom",
                fontsize=8.5,
                color=color,
                fontweight="bold" if is_headline else "normal",
            )

    # Annotation: largest-recovery-factor detector. Computes the fold-change
    # dynamically rather than hardcoding it, so re-running the script after a
    # data refresh produces an annotation that matches the underlying numbers.
    fold_changes = []
    for det in DETECTOR_ORDER:
        if det not in df.detector.values:
            continue
        ys = _values(df, det, "recall_at_1pctfpr")
        if ys[1] is None or ys[2] is None or ys[1] <= 0 or ys[2] <= 0:
            continue
        fold = ys[2] / ys[1]
        if 1.5 < fold < 1000:    # ignore degenerate or trivial cases
            fold_changes.append((det, fold, ys[1], ys[2]))
    if fold_changes:
        # Pick the detector with the largest fold-change as the headline.
        headline_det, headline_fold, y_pert, y_can = max(fold_changes, key=lambda t: t[1])
        det_color = DETECTOR_COLORS[headline_det]
        ax_recall.annotate(
            f"$\\sim$ {headline_fold:.0f}$\\times$ recovery\n({y_pert:.3f} → {y_can:.3f})",
            xy=(2, y_can),
            xytext=(1.45, 0.78),
            ha="center", va="center",
            fontsize=9.5, fontweight="bold",
            color=det_color,
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="white",
                      edgecolor=det_color,
                      linewidth=1.4),
            arrowprops=dict(arrowstyle="->",
                            color=det_color,
                            linewidth=1.4,
                            connectionstyle="arc3,rad=-0.18"),
        )

    ax_recall.set_xticks(x)
    ax_recall.set_xticklabels(CONDITION_LABELS, fontsize=10)
    ax_recall.set_ylabel("Recall @ 1% FPR", fontsize=11)
    ax_recall.set_title(
        "Canonicalization restores three of four detectors to clean-baseline R@1%FPR",
        fontsize=11, pad=10,
    )
    ax_recall.set_ylim(-0.05, 1.08)
    ax_recall.grid(axis="y", linestyle=":", alpha=0.5)
    ax_recall.set_axisbelow(True)
    ax_recall.legend(loc="lower left", fontsize=9, framealpha=0.95, ncol=2)

    # Light vertical guide-lines between conditions
    for xi in [0.5, 1.5]:
        ax_recall.axvline(xi, color="#cccccc", linewidth=0.6, linestyle="--", zorder=0)

    # ---------- Side panel: AUROC ----------
    for det in DETECTOR_ORDER:
        if det not in df.detector.values:
            continue
        ys = _values(df, det, "auroc")
        color = DETECTOR_COLORS[det]
        ax_auroc.plot(
            x, ys,
            marker="o", markersize=6, linewidth=1.5,
            color=color,
            label=DETECTOR_DISPLAY[det],
        )

    ax_auroc.set_xticks(x)
    ax_auroc.set_xticklabels(CONDITION_LABELS, fontsize=9)
    ax_auroc.set_ylabel("AUROC", fontsize=10)
    ax_auroc.set_title("AUROC (threshold-independent)", fontsize=10, pad=8)
    ax_auroc.set_ylim(0.45, 1.05)
    ax_auroc.grid(axis="y", linestyle=":", alpha=0.5)
    ax_auroc.set_axisbelow(True)

    # ---------- Save ----------
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_stem.with_suffix(".png")
    out_pdf = out_stem.with_suffix(".pdf")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"saved {out_png.relative_to(REPO_ROOT)}  ({out_png.stat().st_size/1024:.1f} KB)")
    print(f"saved {out_pdf.relative_to(REPO_ROOT)}  ({out_pdf.stat().st_size/1024:.1f} KB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help=f"Input CSV (default: {DEFAULT_CSV.relative_to(REPO_ROOT)})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output stem without extension (default: {DEFAULT_OUT.relative_to(REPO_ROOT)})")
    args = p.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"input CSV not found: {args.csv}")

    df = load_data(args.csv)
    render(df, args.out)


if __name__ == "__main__":
    main()
