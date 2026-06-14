"""F9 [Results E3] — MULTILINGUAL HEADLINE.

Recall@1%FPR on AR/ES (curated + NLLB MT) at the SAME frozen tau, CANOPI vs
B5/B6/B8 with bootstrap CIs — the ~0.000 -> non-zero plot that is the paper's
minimum-viable-breakthrough. Grayscale-legible (hatch + marker per model).

Input : paper/paper_v2_extract/canopi/crosslingual.csv
        schema: model,lang,recall_at_1pctfpr_mean,recall_at_1pctfpr_std (+ baselines)
Output: paper/figures/canopi_multilingual_headline.{png,pdf}

Usage : python scripts/make_canopi_multilingual_figure.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from canopi_figutils import EXTRACT, FIGDIR, GRAYS, HATCHES, mean_std, require, save

MODEL_ORDER = ["CANOPI", "B6_invariance_only", "B5_embedding", "B8_l0_only"]
DISPLAY = {"CANOPI": "CANOPI", "B6_invariance_only": "B6 inv-only",
           "B5_embedding": "B5 embedding", "B8_l0_only": "B8 L0-only"}


def render(df, out_stem: Path):
    import matplotlib.pyplot as plt

    langs = list(dict.fromkeys(df["lang"]))
    models = [m for m in MODEL_ORDER if m in set(df["model"])] or list(dict.fromkeys(df["model"]))
    x = np.arange(len(langs))
    w = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    for i, m in enumerate(models):
        sub = df[df["model"] == m].set_index("lang").reindex(langs).reset_index()
        mean, std = mean_std(sub, "recall_at_1pctfpr")
        ax.bar(x + i * w - 0.4 + w / 2, mean, w, yerr=std, capsize=3,
               color=GRAYS[i % len(GRAYS)], hatch=HATCHES[i % len(HATCHES)],
               edgecolor="black", linewidth=0.7, label=DISPLAY.get(m, m))
        for xi, yi in zip(x + i * w - 0.4 + w / 2, mean):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in langs], fontsize=11)
    ax.set_ylabel("Recall @ 1% FPR (same frozen $\\tau$)", fontsize=11)
    ax.set_title("CANOPI recovers cross-lingual recall that non-invariant baselines lose",
                 fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.95)
    save(fig, out_stem)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", type=Path, default=EXTRACT / "crosslingual.csv")
    p.add_argument("--out", type=Path, default=FIGDIR / "canopi_multilingual_headline")
    args = p.parse_args()
    if not args.csv.exists():
        print(f"skip F9: {args.csv.name} not found (run the CANOPI orchestrator notebook first)")
        return
    render(require(args.csv), args.out)


if __name__ == "__main__":
    main()
