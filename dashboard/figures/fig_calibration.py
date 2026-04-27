"""
fig_calibration.py
==================
Reliability diagrams (before / after temperature scaling) for all
HybridGuard variants.  Saves figures/calibration.png.

Run:  python figures/fig_calibration.py
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
    cal_data = R.calibration
    hg_models = [m for m, t in R.main_results.set_index("model")["type"].items()
                 if t == "hybridguard"]

    n = len(hg_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, hg_models):
        d = cal_data.get(model, {})
        if not d:
            continue
        bins = np.array(d["bins"])
        ax.plot([0, 1], [0, 1], ":", color="#555555", label="Perfect", linewidth=1)
        ax.bar(bins, d["raw_frac_positive"], width=0.09, alpha=0.4,
               color="#ef5350", label=f"Raw (ECE={d['ece_raw']:.3f})")
        ax.bar(bins, d["cal_frac_positive"], width=0.06, alpha=0.7,
               color="#66bb6a", label=f"Cal (ECE={d['ece_cal']:.3f})")
        ax.plot([0,1],[0,1],":",color="#555",linewidth=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(model.replace("_seed1337", ""), fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Predicted Prob", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Fraction Positive", fontsize=9)
    fig.suptitle("Calibration Reliability Diagrams – HybridGuard Variants",
                 fontsize=13, fontweight="bold", y=1.02)

    out = Path(__file__).parent / "calibration.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
