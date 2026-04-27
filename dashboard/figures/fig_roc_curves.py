"""
fig_roc_curves.py
=================
Standalone script – generates ROC curves figure and saves to
figures/roc_curves.png  (relative to script location).

Run:  python figures/fig_roc_curves.py
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

TYPE_COLORS = {"baseline": "#78909c", "sota": "#ffa726", "hybridguard": "#42a5f5"}

def main():
    R = Results()
    curves = R.roc_curves
    model_types = R.main_results.set_index("model")["type"].to_dict()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot([0, 1], [0, 1], ":", color="#555", label="Random (AUC=0.50)")

    for model, data in curves.items():
        col = TYPE_COLORS.get(model_types.get(model, "baseline"), "#aaa")
        lw  = 2.5 if model_types.get(model) == "hybridguard" else 1.5
        ls  = "-" if model_types.get(model) == "hybridguard" else "--"
        ax.plot(data["fpr"], data["tpr"],
                color=col, linewidth=lw, linestyle=ls,
                label=f"{model} ({data['auroc']:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves – HybridGuard vs Baselines", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Annotate 1% FPR operating point
    ax.axvline(0.01, color="#ef5350", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.text(0.012, 0.05, "1% FPR", color="#ef5350", fontsize=8)

    out = Path(__file__).parent / "roc_curves.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
