"""
Reproducible figure: cost-quality Pareto frontier across detectors.

Reads paper/paper_v2_extract/llm_judge/results.csv and emits
paper/figures/llm_judge_pareto.{png,pdf}.

Source data: HybridGuard_LLM_Judge_Baseline.ipynb run 2026-04-25
plus prior small-model baselines (5-seed in-domain run 2026-04-24).

The plot positions every detector on a (log10 cost-per-million, R@1%FPR)
plane. The HybridGuard point sits on the achievable Pareto frontier for
sub-10ms on-prem deployment; the Claude Haiku point sits orders of
magnitude to the right (higher cost) and substantially higher in recall.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "paper" / "paper_v2_extract" / "llm_judge" / "results.csv"
DEFAULT_OUT = REPO_ROOT / "paper" / "figures" / "llm_judge_pareto"

LABEL = {
    "regex_heuristic":   "Regex",
    "tfidf_linearsvm":   "TF-IDF+SVM",
    "sota_injecguard":   "InjecGuard",
    "sota_deberta_v3":   "DeBERTa-v3",
    "hg_multifeat":      "HybridGuard\n(this work)",
    "claude_haiku_4_5":  "Claude\nHaiku 4.5",
}

COLOR = {
    "regex_heuristic":   "#9e9e9e",
    "tfidf_linearsvm":   "#9e9e9e",
    "sota_injecguard":   "#ffa726",
    "sota_deberta_v3":   "#ffa726",
    "hg_multifeat":      "#1f77b4",
    "claude_haiku_4_5":  "#7e57c2",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"input CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)

    # Mid-cost for log placement (geometric mean of low/high band)
    df["cost_mid"] = (df["cost_per_million_low"] * df["cost_per_million_high"]) ** 0.5

    # Plot each point individually so we can label cleanly
    for _, row in df.iterrows():
        det = row["detector"]
        x = row["cost_mid"]
        y = row["recall_at_1pctfpr"]
        # Error bar showing cost band (only meaningful for the LLM)
        if row["cost_per_million_low"] != row["cost_per_million_high"]:
            xerr_low  = x - row["cost_per_million_low"]
            xerr_high = row["cost_per_million_high"] - x
            ax.errorbar(x, y, xerr=[[xerr_low], [xerr_high]],
                        fmt="o", color=COLOR.get(det, "#888"),
                        markersize=12, markeredgecolor="black", markeredgewidth=0.6,
                        capsize=4, ecolor=COLOR.get(det, "#888"))
        else:
            ax.plot(x, y, "o", color=COLOR.get(det, "#888"),
                    markersize=12, markeredgecolor="black", markeredgewidth=0.6)

        # Label placement: hand-tuned offsets per point
        offsets = {
            "regex_heuristic":   (0.13, 0.02),
            "tfidf_linearsvm":   (0.13, -0.04),
            "sota_injecguard":   (0.13, -0.04),
            "sota_deberta_v3":   (0.13, 0.06),
            "hg_multifeat":      (-0.55, 0.04),
            "claude_haiku_4_5":  (-2.6,  0.02),
        }
        dx, dy = offsets.get(det, (0.1, 0.02))
        ax.annotate(LABEL.get(det, det), xy=(x, y), xytext=(x * 10**dx, y + dy),
                    fontsize=9.5, ha="left",
                    color=COLOR.get(det, "#444"), fontweight="600")

    ax.set_xscale("log")
    ax.set_xlabel("Inference cost per million samples (USD, log scale)", fontsize=11)
    ax.set_ylabel("Recall at 1% FPR (higher is better)", fontsize=11)
    ax.set_title("Cost-quality Pareto frontier: HybridGuard vs LLM-as-judge",
                 fontsize=11.5)
    ax.set_xlim(0.05, 5e6)
    ax.set_ylim(-0.04, 1.0)
    ax.grid(True, which="major", linestyle=":", alpha=0.45)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)

    # Annotate the regimes
    ax.axvspan(0.05, 1.0,    alpha=0.06, color="green", label="_nolegend_")
    ax.axvspan(10000, 5e6,   alpha=0.06, color="purple", label="_nolegend_")
    ax.text(0.3, 0.92, "Low-cost,\non-prem regime",
            fontsize=8.5, ha="center", color="#2e7d32", fontstyle="italic")
    ax.text(2e5, 0.92, "API LLM-as-judge\nregime",
            fontsize=8.5, ha="center", color="#5e35b1", fontstyle="italic")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_png = args.out.with_suffix(".png")
    out_pdf = args.out.with_suffix(".pdf")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"saved {out_png.relative_to(REPO_ROOT)}  ({out_png.stat().st_size/1024:.1f} KB)")
    print(f"saved {out_pdf.relative_to(REPO_ROOT)}  ({out_pdf.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
