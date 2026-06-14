"""CANOPI method/intro schematics (mostly data-free; grayscale vector PDFs).

  F1 canonicalization hierarchy (L0 orthographic -> L1 intent)
  F2 motivation: baseline R@1%FPR collapse under obfuscation + multilingual (bar)
  F3 CANOPI architecture block diagram (c -> frozen E -> P -> detector -> tau)
  F4 joint-objective schematic (views pulled across languages; hard negs pushed;
     pAUC low-FPR tail highlighted)
  F6 data/augmentation pipeline with intent-preservation accept rates
  F19 hyperparameter/encoder ablation (head depth, frozen vs finetune, mono vs multi)

F2/F6/F19 use a CSV if present (paper_v2_extract/canopi/), else illustrative
placeholders clearly labelled "schematic". Usage:
    python scripts/make_canopi_schematics_figure.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from canopi_figutils import EXTRACT, FIGDIR, GRAYS, HATCHES, save


def _box(ax, x, y, w, h, text, fc="white"):
    import matplotlib.patches as mp

    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                 linewidth=1.4, edgecolor="black", facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.4))


def f3_architecture():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 2.6); ax.axis("off")
    stages = [("x\n(raw prompt)", "white"), ("c(x)\nL0 canonicalize", "#eeeeee"),
              ("E(.)\nFROZEN encoder", "#dddddd"), ("P(.)\ntrained head\n(unit-norm)", "#cccccc"),
              ("detector\nhead", "#eeeeee"), ("s(x) $\\geq \\tau$\ndecision", "white")]
    w = 1.55
    xs = np.linspace(0.2, 9.3, len(stages))
    for (txt, fc), x in zip(stages, xs):
        _box(ax, x, 0.8, w, 1.0, txt, fc)
    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(ax, a + w, 1.3, b, 1.3)
    ax.text(5.5, 2.45, "Inference pipeline — NO generative model in the path",
            ha="center", fontsize=10, style="italic")
    ax.text(xs[2] + w / 2, 0.55, "snowflake: frozen", ha="center", fontsize=7, color="#555")
    save(fig, FIGDIR / "canopi_architecture")


def f4_joint_objective():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal"); ax.axis("off")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="#bbbbbb", linewidth=1.0)
    rng = np.random.default_rng(0)
    # same-intent views (EN/AR/ES) clustered -> pulled together
    c = np.array([0.55, 0.35])
    pts = c + 0.12 * rng.standard_normal((6, 2))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    for p in pts:
        ax.scatter(*p, s=60, marker="o", color=GRAYS[0])
    ax.scatter([], [], marker="o", color=GRAYS[0], label="same-intent views (EN/AR/ES) — $L_{inv}$ pulls")
    for i in range(len(pts) - 1):
        ax.plot([pts[i, 0], pts[i + 1, 0]], [pts[i, 1], pts[i + 1, 1]], color=GRAYS[2], linewidth=0.7)
    # hard negative (trigger-sharing benign) pushed away
    hn = np.array([-0.7, 0.2]); hn = hn / np.linalg.norm(hn)
    ax.scatter(*hn, s=90, marker="X", color="black", label="trigger-sharing benign — $L_{hardneg}$ pushes")
    _arrow_xy(ax, c / np.linalg.norm(c), hn)
    # pAUC low-FPR tail
    ax.scatter([-0.2], [-0.85], s=80, marker="s", color=GRAYS[1],
               label="low-FPR tail — $L_{pAUC}$ top-push")
    ax.add_patch(plt.matplotlib.patches.Wedge((0, 0), 1.0, -120, -60, width=0.25,
                 facecolor="#999999", alpha=0.35))
    ax.set_title("Joint objective on the unit sphere", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(-0.15, 1.18), framealpha=0.95)
    save(fig, FIGDIR / "canopi_joint_objective")


def _arrow_xy(ax, p, q):
    ax.annotate("", xy=(q[0] * 0.85, q[1] * 0.85), xytext=(p[0] * 0.85, p[1] * 0.85),
                arrowprops=dict(arrowstyle="-|>", color="#444", linewidth=1.2, linestyle="--"))


def f1_hierarchy():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 3.0)); ax.axis("off")
    ax.set_xlim(0, 8.5); ax.set_ylim(0, 3)
    _box(ax, 0.3, 1.1, 2.2, 0.9, "L0 — orthographic\n(NFKC, confusables,\nencoding unwrap)", "#dddddd")
    _box(ax, 3.2, 1.1, 2.2, 0.9, "L1 — intent\n(learned invariance:\nparaphrase, language)", "#cccccc")
    _box(ax, 6.1, 1.1, 2.1, 0.9, "decision\nat fixed $\\tau$", "white")
    _arrow(ax, 2.5, 1.55, 3.2, 1.55); _arrow(ax, 5.4, 1.55, 6.1, 1.55)
    ax.text(4.25, 2.5, "Canonicalization hierarchy", ha="center", fontsize=11, fontweight="bold")
    ax.text(1.4, 0.7, "deterministic", ha="center", fontsize=8, style="italic")
    ax.text(4.3, 0.7, "CANOPI (trained)", ha="center", fontsize=8, style="italic")
    save(fig, FIGDIR / "canopi_hierarchy")


def f2_motivation():
    import matplotlib.pyplot as plt
    import pandas as pd

    csv = EXTRACT / "motivation.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        conds, vals = df["condition"].tolist(), df["recall_at_1pctfpr"].tolist()
    else:
        conds = ["clean", "obfuscated", "Arabic", "Spanish"]
        vals = [0.98, 0.30, 0.00, 0.35]  # SCHEMATIC illustrative collapse
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.bar(np.arange(len(conds)), vals, color=GRAYS[1], hatch=HATCHES[1], edgecolor="black")
    ax.set_xticks(np.arange(len(conds))); ax.set_xticklabels(conds, fontsize=10)
    ax.set_ylabel("Baseline Recall @ 1% FPR", fontsize=11); ax.set_ylim(0, 1.05)
    title = "Baseline collapse under obfuscation + language shift"
    if not csv.exists():
        title += "  (schematic)"
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    save(fig, FIGDIR / "canopi_motivation")


def f6_pipeline():
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, ax = plt.subplots(figsize=(10.5, 3.0)); ax.axis("off")
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 3)
    fams = ["anchor", "paraphrase", "persona", "encoding", "AR (NLLB)", "ES (NLLB)"]
    csv = EXTRACT / "accept_rates.csv"
    rates = {}
    if csv.exists():
        d = pd.read_csv(csv); rates = dict(zip(d["family"], d["accept_rate"]))
    xs = np.linspace(0.2, 8.8, len(fams))
    for f, x in zip(fams, xs):
        _box(ax, x, 1.2, 1.4, 0.9, f, "#eeeeee")
        r = rates.get(f.split()[0], None)
        ax.text(x + 0.7, 0.95, f"accept {r:.2f}" if r is not None else "accept --",
                ha="center", fontsize=7, color="#444")
    _box(ax, 9.0, 1.2, 1.3, 0.9, "intent\nfilter", "#cccccc")
    ax.text(5.2, 2.6, "Augmentation pipeline + intent-preservation accept rates",
            ha="center", fontsize=11, fontweight="bold")
    save(fig, FIGDIR / "canopi_aug_pipeline")


def f19_hyperparam():
    import matplotlib.pyplot as plt
    import pandas as pd

    csv = EXTRACT / "hyperparam.csv"
    if not csv.exists():
        print("skip F19: hyperparam.csv not found"); return
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    x = np.arange(len(df))
    ax.bar(x, df["recall_at_1pctfpr_mean"], yerr=df.get("recall_at_1pctfpr_std", None),
           capsize=3, color=GRAYS[2], hatch=HATCHES[3], edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(df.iloc[:, 0], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Recall @ 1% FPR", fontsize=11)
    ax.set_title("Encoder / head-depth ablation", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.5); save(fig, FIGDIR / "canopi_hyperparam")


ALL = [f1_hierarchy, f2_motivation, f3_architecture, f4_joint_objective, f6_pipeline, f19_hyperparam]


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    for fn in ALL:
        try:
            fn()
        except Exception as e:  # one figure failing must not block the rest
            print(f"skip {fn.__name__}: {e}")


if __name__ == "__main__":
    main()
