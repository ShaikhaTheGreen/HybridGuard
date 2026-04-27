"""
fig_ablation.py
===============
Horizontal bar chart showing AUROC drop for each ablation component.
Saves figures/ablation.png.

Run:  python figures/fig_ablation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils.load_results import Results

mpl.rcParams.update({"figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
                      "axes.edgecolor": "#30363d", "text.color": "#c9d1d9",
                      "axes.labelcolor": "#c9d1d9", "xtick.color": "#8b949e",
                      "ytick.color": "#8b949e", "grid.color": "#21262d",
                      "legend.facecolor": "#161b22", "legend.edgecolor": "#30363d",
                      "axes.titlecolor": "#7ecfff"})

def main():
    R = Results()
    df = R.ablation.copy()

    full_auroc = df[df["ablation"] == "full_model"]["auroc"].values[0]
    df_ab = df[df["ablation"] != "full_model"].sort_values("auroc_drop")

    colors = ["#ef5350" if v > 0.025 else "#ffa726" for v in df_ab["auroc_drop"]]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(df_ab["ablation"], df_ab["auroc_drop"], color=colors, edgecolor="#30363d")

    for bar, v in zip(bars, df_ab["auroc_drop"]):
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                f"−{v:.3f}", va="center", ha="left", fontsize=9, color="#c9d1d9")

    ax.axvline(0, color="#555", linewidth=1)
    ax.set_xlabel("AUROC Drop (lower = less important, larger = critical component)", fontsize=10)
    ax.set_title(f"Ablation Study – HybridGuard (Full AUROC = {full_auroc:.3f})",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, df_ab["auroc_drop"].max() * 1.35)
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#ef5350", label="Critical (drop > 0.025)"),
        Patch(facecolor="#ffa726", label="Moderate (drop ≤ 0.025)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    out = Path(__file__).parent / "ablation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
