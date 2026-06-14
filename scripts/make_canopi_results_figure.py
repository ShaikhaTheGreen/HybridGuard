"""CANOPI results figures (data-driven, from runs/<run_id>/ via paper_v2_extract/canopi/).

Emits, when its CSV exists (skips gracefully otherwise):
  F8  main_results.csv        -> canopi_main_bars        R@1%FPR bars + CIs, B1-B8+CANOPI
  F10 threshold_transfer.csv  -> canopi_threshold_transfer  FPR @ fixed tau across corpora
  F11 overdefense.csv         -> canopi_overdefense      over-defense FPR vs op-point sweep
  F13 drift.csv               -> canopi_drift            invariance-drift delta per family
  F14 adaptive.csv            -> canopi_adaptive_residual R@1%FPR vs rewrite strength
  F15 detector_transfer.csv   -> canopi_detector_transfer recall recovery per detector
  F18 ablation.csv            -> canopi_ablation_deltas  Delta R@1%FPR per removed term/aug

Usage: python scripts/make_canopi_results_figure.py   # all available
       python scripts/make_canopi_results_figure.py --only F9,F18
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from canopi_figutils import EXTRACT, FIGDIR, GRAYS, HATCHES, MARKERS, mean_std, save


def _bars(df, group_col, value_metric, title, ylabel, out_name, series_col="model"):
    import matplotlib.pyplot as plt

    groups = list(dict.fromkeys(df[group_col]))
    series = list(dict.fromkeys(df[series_col]))
    x = np.arange(len(groups))
    w = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(groups)), 4.4), constrained_layout=True)
    for i, s in enumerate(series):
        sub = df[df[series_col] == s].set_index(group_col).reindex(groups).reset_index()
        mean, std = mean_std(sub, value_metric)
        ax.bar(x + i * w - 0.4 + w / 2, mean, w, yerr=std, capsize=3,
               color=GRAYS[i % len(GRAYS)], hatch=HATCHES[i % len(HATCHES)],
               edgecolor="black", linewidth=0.7, label=str(s))
    ax.set_xticks(x); ax.set_xticklabels([str(g) for g in groups], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11); ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    if len(series) > 1:
        ax.legend(fontsize=8, ncol=2, framealpha=0.95)
    save(fig, FIGDIR / out_name)


def f8_main_real(df):
    import matplotlib.pyplot as plt

    df = df.copy()
    mean, std = mean_std(df, "recall_at_1pctfpr")
    order = np.argsort(-mean)
    models = df["model"].to_numpy()[order]
    fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    x = np.arange(len(models))
    ax.bar(x, mean[order], yerr=std[order], capsize=3, color=GRAYS[0], hatch=HATCHES[1],
           edgecolor="black", linewidth=0.7)
    for xi, yi in zip(x, mean[order]):
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Recall @ 1% FPR", fontsize=11); ax.set_ylim(0, 1.08)
    ax.set_title("In-domain Recall @ 1% FPR (5 seeds, 95% CI)", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    save(fig, FIGDIR / "canopi_main_bars")


def f10_transfer(df):
    _bars(df, "corpus", "fpr_at_tau", "Threshold transfer: realized FPR at the fixed $\\tau$",
          "FPR @ frozen $\\tau$", "canopi_threshold_transfer")


def f11_overdefense(df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    for i, m in enumerate(dict.fromkeys(df["model"])):
        sub = df[df["model"] == m].sort_values("fpr_target")
        od_mean, od_std = mean_std(sub, "overdefense_fpr")
        ax.errorbar(sub["fpr_target"].to_numpy(float), od_mean, yerr=od_std, marker=MARKERS[i % 7],
                    color=GRAYS[i % len(GRAYS)], linewidth=1.8, capsize=3, label=str(m))
    ax.set_xlabel("In-domain FPR target", fontsize=11)
    ax.set_ylabel("NotInject over-defense FPR", fontsize=11)
    ax.set_title("Over-defense vs operating point (lower is better)", fontsize=11)
    ax.grid(linestyle=":", alpha=0.5); ax.legend(fontsize=9); save(fig, FIGDIR / "canopi_overdefense")


def f13_drift(df):
    _bars(df, "family", "drift_delta", "Invariance drift $\\delta$ per transform family (lower = invariant)",
          "$\\delta = \\|P(x)-P(\\mathrm{view})\\|$", "canopi_drift",
          series_col="model" if "model" in df.columns else "family")


def f14_adaptive(df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    for i, m in enumerate(dict.fromkeys(df["model"])):
        sub = df[df["model"] == m].sort_values("rewrite_strength")
        rm, rs = mean_std(sub, "recall_at_1pctfpr")
        ax.errorbar(sub["rewrite_strength"].to_numpy(float), rm, yerr=rs, marker=MARKERS[i % 7],
                    color=GRAYS[i % len(GRAYS)], linewidth=2.0, capsize=3, label=str(m))
    ax.set_xlabel("Adaptive semantic-rewrite strength", fontsize=11)
    ax.set_ylabel("Recall @ 1% FPR", fontsize=11); ax.set_ylim(0, 1.05)
    ax.set_title("E7 residual: recall vs rewrite strength (reported, not hidden)", fontsize=11)
    ax.grid(linestyle=":", alpha=0.5); ax.legend(fontsize=9); save(fig, FIGDIR / "canopi_adaptive_residual")


def f15_detector(df):
    _bars(df, "detector", "recall_at_1pctfpr", "Detector-agnostic recall recovery (CANOPI front-end)",
          "Recall @ 1% FPR", "canopi_detector_transfer",
          series_col="condition" if "condition" in df.columns else "detector")


def f18_ablation(df):
    import matplotlib.pyplot as plt

    df = df.copy()
    dm, ds = mean_std(df, "delta_recall_at_1pctfpr") if any(c.startswith("delta_recall") for c in df.columns) else mean_std(df, "recall_at_1pctfpr")
    order = np.argsort(dm)
    labels = df["ablation"].to_numpy() if "ablation" in df.columns else df.iloc[:, 0].to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    y = np.arange(len(labels))
    ax.barh(y, dm[order], xerr=ds[order], capsize=3, color=GRAYS[2], hatch=HATCHES[2],
            edgecolor="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(y); ax.set_yticklabels(labels[order], fontsize=9)
    ax.set_xlabel("$\\Delta$ Recall @ 1% FPR vs full CANOPI (CI must exclude 0)", fontsize=10)
    ax.set_title("E2 ablation: contribution of each loss term / augmentation family", fontsize=11)
    ax.grid(axis="x", linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    save(fig, FIGDIR / "canopi_ablation_deltas")


FIGS = {
    "F8": ("main_results.csv", f8_main_real),
    "F10": ("threshold_transfer.csv", f10_transfer),
    "F11": ("overdefense.csv", f11_overdefense),
    "F13": ("drift.csv", f13_drift),
    "F14": ("adaptive.csv", f14_adaptive),
    "F15": ("detector_transfer.csv", f15_detector),
    "F18": ("ablation.csv", f18_ablation),
}


def main():
    import pandas as pd

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--only", type=str, default=None, help="comma list e.g. F8,F18")
    args = p.parse_args()
    want = set(args.only.split(",")) if args.only else set(FIGS)
    made = 0
    for fid, (csv_name, fn) in FIGS.items():
        if fid not in want:
            continue
        csv = EXTRACT / csv_name
        if not csv.exists():
            print(f"skip {fid}: {csv.name} not found (run the notebook first)")
            continue
        fn(pd.read_csv(csv)); made += 1
    print(f"made {made} figure(s)")


if __name__ == "__main__":
    main()
