"""HEADLINE figure — the encoder x objective mechanism.

Two panels, Recall@1%FPR per language at each detector's own frozen tau:
  (A) MONOLINGUAL encoder: B5 (no-invariance probe) collapses cross-lingually;
      CANOPI's invariance objective recovers it -> the objective SUBSTITUTES for
      multilingual pretraining. 5-seed, real (run_ablation_monolingual).
  (B) MULTILINGUAL encoder: the encoder already supplies cross-lingual ability, so
      CANOPI ~ B5 -> the control that localizes the objective's value.

Input : paper/paper_v2_extract/canopi/mechanism_mono.csv (5-seed)
        paper/paper_v2_extract/canopi/mechanism_multi_SMOKE.csv (provisional)
Output: paper/figures/canopi_mechanism.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from canopi_figutils import EXTRACT, FIGDIR, mean_std, save

LANGS = ["es", "es_mt", "fr_mt", "de_mt", "zh_mt", "pt_mt", "ar", "ru_mt"]
LABELS = ["es", "es*", "fr*", "de*", "zh*", "pt*", "ar", "ru*"]  # * = MT panel


def _panel(ax, df, title, provisional=False):
    import matplotlib.pyplot as plt

    x = np.arange(len(LANGS))
    w = 0.38
    for i, (model, color, hatch, lab) in enumerate([
        ("CANOPI", "#222222", "", "CANOPI (joint objective)"),
        ("B5_embedding", "#bbbbbb", "///", "B5 (no-invariance probe)"),
    ]):
        sub = df[df["model"] == model].set_index("lang").reindex(LANGS).reset_index()
        mean, std = mean_std(sub, "recall_at_1pctfpr")
        mean = np.nan_to_num(mean)
        ax.bar(x + (i - 0.5) * w, mean, w, yerr=std, capsize=2.5, color=color,
               hatch=hatch, edgecolor="black", linewidth=0.7, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall @ 1% FPR", fontsize=10)
    ax.set_title(title + ("  [provisional]" if provisional else ""), fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)


def render(mono, multi, out_stem: Path):
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
    _panel(axA, mono, "(A) Monolingual encoder — objective recovers cross-lingual recall")
    _panel(axB, multi, "(B) Multilingual encoder — encoder already provides it (control)",
           provisional=True)
    axA.text(0.5, -0.18, "* = MT-translated held-out test set; bars are mean$\\pm$sd over 5 seeds",
             transform=axA.transAxes, ha="center", fontsize=7, color="#555")
    save(fig, out_stem)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mono", type=Path, default=EXTRACT / "mechanism_mono.csv")
    p.add_argument("--multi", type=Path, default=EXTRACT / "mechanism_multi_SMOKE.csv")
    p.add_argument("--out", type=Path, default=FIGDIR / "canopi_mechanism")
    args = p.parse_args()
    import pandas as pd

    if not args.mono.exists():
        print("skip mechanism fig: mono CSV missing"); return
    multi = pd.read_csv(args.multi) if args.multi.exists() else pd.read_csv(args.mono)
    render(pd.read_csv(args.mono), multi, args.out)


if __name__ == "__main__":
    main()
