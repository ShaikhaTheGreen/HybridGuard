"""
fig_sanitization.py
===================
Security–Utility trade-off scatter for sanitisation modes.
Saves figures/sanitization_tradeoff.png.

Run:  python figures/fig_sanitization.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib as mpl
from utils.load_results import Results

mpl.rcParams.update({"figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
                      "axes.edgecolor": "#30363d", "text.color": "#c9d1d9",
                      "axes.labelcolor": "#c9d1d9", "xtick.color": "#8b949e",
                      "ytick.color": "#8b949e", "grid.color": "#21262d",
                      "legend.facecolor": "#161b22", "legend.edgecolor": "#30363d",
                      "axes.titlecolor": "#7ecfff"})

COLOR_MAP = {
    "off":                  "#78909c",
    "rule_strip":           "#42a5f5",
    "context_isolation":    "#ffa726",
    "llm_rewrite_optional": "#66bb6a",
}

def main():
    R = Results()
    df = R.sanitization.copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in df.iterrows():
        mode = row["sanitize_mode"]
        x = float(row["utility_semantic_similarity_mean"])
        y = float(row["asrr_proxy_mean_delta_sanitized"])
        col = COLOR_MAP.get(mode, "#ffffff")
        ax.scatter(x, y, s=280, color=col, edgecolors="white", linewidths=0.8, zorder=5)
        ax.annotate(mode.replace("_", "\n"), (x, y),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=8.5, color=col)

    ax.set_xlabel("Utility: Mean Semantic Similarity ↑", fontsize=11)
    ax.set_ylabel("Security: ASRR Proxy (Mean Δp among sanitised) ↑", fontsize=11)
    ax.set_title("Sanitisation Security–Utility Trade-Off", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)

    # Ideal corner annotation
    ax.annotate("← ideal (high utility, high security)",
                xy=(1.0, max(df["asrr_proxy_mean_delta_sanitized"].astype(float))),
                fontsize=8, color="#8b949e", ha="right")

    out = Path(__file__).parent / "sanitization_tradeoff.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
