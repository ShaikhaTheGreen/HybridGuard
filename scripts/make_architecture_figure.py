"""Generate the HybridGuard architecture diagram for the paper.

Produces ``paper/figures/architecture.pdf`` and ``paper/figures/architecture.png``.

The diagram visualizes the input-side pipeline:
    raw input -> canonicalization c(x) -> sanitization g(x) -> feature extraction
                                                              -> fusion -> score s(x).

It also shows the four feature-extraction branches and the four core variants
(HG_MULTIFEAT, HG_CNN_TRANS, HG_ENSEMBLE, HG_RAV) as parallel branch alternatives.

Run from the repo root::

    python scripts/make_architecture_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "paper" / "figures"


def _box(ax, x, y, w, h, label, *, fc="#f4f4f4", ec="#222", lw=1.0, fontsize=9):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def _arrow(ax, p0, p1, *, color="#222", lw=1.0, style="-|>"):
    arrow = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=10,
        color=color,
        linewidth=lw,
    )
    ax.add_patch(arrow)


def render() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.set_axis_off()

    # Stage 1 — input
    _box(ax, 1, 22, 11, 6, "Raw input\n$x$\n(prompt / RAG\nchunk / tool out)", fc="#eef5fb")

    # Stage 2 — canonicalization c(x)
    _box(
        ax,
        15,
        22,
        16,
        6,
        "Canonicalize $c(x)$\nNFKC + invisibles strip\n+ TR39 fold + collapse\n+ depth-bounded unwrap",
        fc="#dceefc",
        lw=1.4,
    )

    # Stage 3 — sanitization g(x)
    _box(
        ax,
        34,
        22,
        14,
        6,
        "Sanitize $g(\\hat{x})$\n(rule-strip /\ncontext-isolation /\nrewrite [optional])",
        fc="#f5e6cc",
    )

    # Stage 4 — feature branches (parallel)
    branches = [
        ("Transformer\nembedding\n$\\phi_{\\rm tr}$", "#e7d6f0"),
        ("Char-CNN\nfeatures\n$\\phi_{\\rm cnn}$", "#e7d6f0"),
        ("Engineered\nstats\n$\\phi_{\\rm eng}$", "#e7d6f0"),
        ("Rule score\n$r(\\tilde{x})$", "#e7d6f0"),
        ("Retrieval\nsim. + margin", "#e7d6f0"),
    ]
    bx = 51
    by_top = 44
    bw = 11
    bh = 5
    gap = 0.5
    branch_centers = []
    for i, (label, fc) in enumerate(branches):
        y = by_top - i * (bh + gap)
        _box(ax, bx, y, bw, bh, label, fc=fc)
        branch_centers.append((bx, y + bh / 2, bx + bw, y + bh / 2))

    # Fusion + classifier
    _box(
        ax,
        67,
        22,
        13,
        6,
        "Fusion head\nMLP / LogReg /\nstacking",
        fc="#d6f0d6",
        lw=1.2,
    )

    # Score + threshold
    _box(
        ax,
        83,
        22,
        15,
        6,
        r"$s(x) \to \hat{y} = \mathbb{1}[s(x) \geq \tau_{1\%}]$"
        "\n"
        r"val-selected $\tau_{1\%}$",
        fc="#fbe1e1",
    )

    # Arrows: main pipeline (input -> c -> g -> branches column anchor)
    _arrow(ax, (12, 25), (15, 25))
    _arrow(ax, (31, 25), (34, 25))
    _arrow(ax, (48, 25), (51, 25))   # g -> branches anchor

    # Arrows from g to each branch (fan out)
    for (xL, yL, xR, yR) in branch_centers:
        _arrow(ax, (49, 25), (xL, yL), color="#666", lw=0.8)

    # Arrows from branches to fusion head (fan in)
    for (xL, yL, xR, yR) in branch_centers:
        _arrow(ax, (xR, yR), (67, 25), color="#666", lw=0.8)

    # Fusion -> score
    _arrow(ax, (80, 25), (83, 25))

    # Variant labels (below the branches column, indicating which branches each variant uses)
    variant_y = 1.5
    _box(
        ax,
        15,
        variant_y,
        82,
        7,
        "Core variants (parallel branch selections):  "
        "HG_MULTIFEAT = $\\phi_{\\rm tr}$ + $\\phi_{\\rm eng}$  ·  "
        "HG_CNN_TRANS = $\\phi_{\\rm tr}$ + $\\phi_{\\rm cnn}$  ·  "
        "HG_ENSEMBLE = $r$ + $\\phi_{\\rm tr,128}$  ·  "
        "HG_RAV = retrieval similarity + margin (with optional veto when margin $\\geq m$)",
        fc="#f7f7f7",
        ec="#888",
        fontsize=8.5,
    )

    # Title
    ax.text(
        50,
        49.2,
        "HybridGuard input-side pipeline:  $x \\to c(x) \\to g(\\hat{x}) \\to \\phi(\\tilde{x}) \\to f \\to s(x)$",
        ha="center",
        va="bottom",
        fontsize=11,
        weight="bold",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_out = OUT_DIR / "architecture.pdf"
    png_out = OUT_DIR / "architecture.png"
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(png_out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {pdf_out}")
    print(f"wrote {png_out}")


if __name__ == "__main__":
    render()
