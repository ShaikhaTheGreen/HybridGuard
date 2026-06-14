"""
canopi.runs
===========
Persistence in the HybridGuard `runs/<run_id>/` convention so the dashboard and
pack_overleaf.sh pick CANOPI results up unchanged:

    runs/<run_id>/
        run_metadata.json                  config, seeds, git sha, versions
        seed_<s>/<name>.csv                per-seed eval frames
        aggregated/<name>_mean_std.csv     mean +/- std across seeds (+ CIs)
        tables/<name>.tex                  rendered LaTeX (booktabs)
        tables/<name>.csv                  dashboard-compatible mirror

This mirrors the orchestrator's per-seed + aggregated + LaTeX layout. The
dashboard loader is additively pointed at runs/ (see load_results patch); the
`tables/` CSV mirror matches the loader's `results/run_*/tables/*.csv` schema.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["RunWriter", "aggregate_mean_std", "render_booktabs"]


def _esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def render_booktabs(df: pd.DataFrame, caption: str = "", label: str = "",
                    float_fmt: str = "%.3f") -> str:
    """Render a DataFrame as a booktabs table without pandas.to_latex (which on
    pandas>=3 pulls in jinja2). Self-contained, deterministic, grayscale-safe."""
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)

    def cell(v):
        if isinstance(v, float):
            return float_fmt % v if v == v else "--"  # NaN -> --
        return _esc(v)

    header = " & ".join(_esc(c) for c in cols) + r" \\"
    rows = [" & ".join(cell(v) for v in row) + r" \\" for row in df.itertuples(index=False, name=None)]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\begin{tabular}{" + align + "}", r"\toprule",
        header, r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
    ]
    if caption:
        lines.append(r"\caption{" + caption + "}")
    if label:
        lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def aggregate_mean_std(frames: Sequence[pd.DataFrame], key_cols: Sequence[str]) -> pd.DataFrame:
    """Stack per-seed frames and reduce numeric columns to mean/std over seeds."""
    if not frames:
        return pd.DataFrame()
    allf = pd.concat(list(frames), ignore_index=True)
    # `seed` is a grouping artifact, never a metric to average over.
    num_cols = [c for c in allf.columns
                if c not in key_cols and c != "seed" and pd.api.types.is_numeric_dtype(allf[c])]
    g = allf.groupby(list(key_cols), dropna=False)
    out = g[num_cols].agg(["mean", "std"]).reset_index()
    out.columns = list(key_cols) + [f"{c}_{stat}" for c, stat in out.columns[len(key_cols):]]
    return out


class RunWriter:
    """Write a run under <root>/<run_id>/.

    NOTE: the dashboard loader (`dashboard/utils/load_results.py:_latest_run_dir`)
    only discovers run directories whose name starts with ``run_``, and reads
    ``tables/<name>.csv`` + ``run_metadata.json``. So pass run_ids like
    ``run_canopi_main_20260614`` and name dashboard-facing tables to match the
    loader (main_results, overdefense, ablation, fairness, robustness, ...).
    """

    def __init__(self, run_id: str, root: str | Path = "runs"):
        if not run_id.startswith("run_"):
            # Keep dashboard discovery working without surprising the caller.
            run_id = f"run_{run_id}"
        self.run_id = run_id
        self.dir = Path(root) / run_id
        (self.dir / "aggregated").mkdir(parents=True, exist_ok=True)
        (self.dir / "tables").mkdir(parents=True, exist_ok=True)

    # --- per-seed ----------------------------------------------------------
    def write_seed(self, seed: int, name: str, df: pd.DataFrame) -> Path:
        d = self.dir / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{name}.csv"
        df.to_csv(p, index=False)
        return p

    def read_all_seeds(self, name: str) -> List[pd.DataFrame]:
        frames = []
        for d in sorted(self.dir.glob("seed_*")):
            p = d / f"{name}.csv"
            if p.exists():
                df = pd.read_csv(p)
                df["seed"] = int(d.name.split("_")[1])
                frames.append(df)
        return frames

    # --- aggregated --------------------------------------------------------
    def aggregate(self, name: str, key_cols: Sequence[str]) -> Optional[pd.DataFrame]:
        frames = self.read_all_seeds(name)
        if not frames:
            return None
        agg = aggregate_mean_std(frames, key_cols)
        agg.to_csv(self.dir / "aggregated" / f"{name}_mean_std.csv", index=False)
        return agg

    # --- tables ------------------------------------------------------------
    def write_table(self, name: str, df: pd.DataFrame, caption: str = "", label: str = "",
                    float_fmt: str = "%.3f", dashboard_mirror: bool = True) -> Path:
        tex = self.dir / "tables" / f"{name}.tex"
        tex.write_text(render_booktabs(df, caption, label, float_fmt), encoding="utf-8")
        if dashboard_mirror:
            df.to_csv(self.dir / "tables" / f"{name}.csv", index=False)
        return tex

    # --- metadata ----------------------------------------------------------
    def write_metadata(self, meta: Dict) -> Path:
        p = self.dir / "run_metadata.json"
        p.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return p

    def write_artifact_json(self, name: str, obj) -> Path:
        p = self.dir / f"{name}.json"
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return p
