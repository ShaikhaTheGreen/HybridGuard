"""
fig_robustness.py
=================
Heatmap of AUROC drop under text perturbations.
Saves figures/robustness_heatmap.png.

Run:  python figures/fig_robustness.py
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
                      "ytick.color": "#8b949e", "grid.color": "#21262d"})

def main():
    R = Results()
    df = R.robustness.copy()

    pivot = df.pivot_table(index="perturbation", columns="model", values="auroc_drop")

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.4), 4.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=pivot.values.max())

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7.5, color="white" if val > 0.06 else "#c9d1d9")

    plt.colorbar(im, ax=ax, shrink=0.8, label="AUROC Drop (↓ = more robust)")
    ax.set_title("Robustness – AUROC Drop under Text Perturbations",
                 fontsize=12, fontweight="bold", pad=12)

    out = Path(__file__).parent / "robustness_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
