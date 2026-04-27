"""
load_results.py
===============
Tries to load real HybridGuard run artefacts from the notebook's output
directory.  Falls back silently to synthetic data when artefacts are absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.synthetic_data import (
    get_ablation_results,
    get_calibration_data,
    get_fairness_results,
    get_jailbreak_results,
    get_main_results,
    get_overdefense_results,
    get_robustness_results,
    get_roc_curve_data,
    get_sanitization_results,
)


def _infer_type(model: str) -> str:
    m = model.lower()
    if m.startswith("hg_"):
        return "hybridguard"
    if m.startswith("sota::") or "protectai" in m or "injecguard" in m or "deberta" in m:
        return "sota"
    return "baseline"


def _patch_type(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    df = df.copy()
    # Normalise column names to what the dashboard expects
    df.rename(columns={
        "recall_at_1pctfpr":   "recall_1pct_fpr",
        "latency_s_per_sample": "latency_ms_per_sample",
        "overdef_fpr_at_thr":  "overdefense_fpr",
    }, inplace=True)
    # Convert latency from seconds to milliseconds if it came in as seconds
    if "latency_ms_per_sample" in df.columns:
        # values < 1 are almost certainly in seconds, not ms
        if df["latency_ms_per_sample"].median() < 1:
            df["latency_ms_per_sample"] = df["latency_ms_per_sample"] * 1000
    if "type" not in df.columns and "model" in df.columns:
        df["type"] = df["model"].apply(_infer_type)
    return df


def _patch_ablation(ablation_df: pd.DataFrame, main_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Add missing full_model row and auroc_drop column expected by the dashboard."""
    if ablation_df is None:
        return ablation_df
    df = ablation_df.copy()

    # Derive full-model AUROC: look up the model in main_results, else use max ablation auroc
    full_auroc = None
    if main_df is not None and "model" in df.columns and "auroc" in df.columns:
        model_name = df["model"].iloc[0] if len(df) else None
        if model_name is not None:
            match = main_df[main_df["model"] == model_name]
            if not match.empty:
                full_auroc = float(match["auroc"].iloc[0])
    if full_auroc is None:
        full_auroc = float(df["auroc"].max()) if "auroc" in df.columns else 1.0

    # Add full_model row if missing
    if "ablation" in df.columns and "full_model" not in df["ablation"].values:
        row = {c: None for c in df.columns}
        row["ablation"] = "full_model"
        row["auroc"] = full_auroc
        if "model" in df.columns:
            row["model"] = df["model"].iloc[0] if len(df) else ""
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Compute auroc_drop if missing
    if "auroc_drop" not in df.columns and "auroc" in df.columns:
        df["auroc_drop"] = full_auroc - df["auroc"].astype(float)

    return df


def _patch_fairness(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide fairness CSV into long format expected by the dashboard."""
    if df is None:
        return df
    # Already in expected format
    if "group" in df.columns and "gap_vs_overall" in df.columns:
        return df
    rows = []
    for _, r in df.iterrows():
        model = r.get("model", "")
        if "lang_recall_gap" in df.columns:
            group = str(r.get("worst_lang", "lang")) or "lang"
            rows.append({"model": model, "group": f"lang: {group}", "gap_vs_overall": r.get("lang_recall_gap", 0)})
        if "len_recall_gap" in df.columns:
            group = str(r.get("worst_len", "len")) or "len"
            rows.append({"model": model, "group": f"len: {group}", "gap_vs_overall": r.get("len_recall_gap", 0)})
    return pd.DataFrame(rows) if rows else df


def _patch_robustness(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    df = df.copy()
    if "transform" in df.columns and "perturbation" not in df.columns:
        df.rename(columns={"transform": "perturbation"}, inplace=True)
    if "auroc_drop" not in df.columns and "auroc" in df.columns and "perturbation" in df.columns:
        baseline = df[df["perturbation"] == "none"].set_index("model")["auroc"]
        df["auroc_drop"] = df.apply(
            lambda r: float(baseline.get(r["model"], r["auroc"])) - float(r["auroc"]),
            axis=1,
        )
    return df


def _latest_run_dir(results_root: Path) -> Optional[Path]:
    """Return the most-recently created run_* subdirectory."""
    candidates = sorted(
        [d for d in results_root.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _try_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return None


class Results:
    """
    Container for all dashboard data.

    Attributes are pandas DataFrames or dicts mirroring synthetic_data.py.
    `.source` is either "live" (read from notebook artefacts) or "demo".
    """

    def __init__(self, results_root: Optional[str] = None):
        self.source = "demo"
        self.run_dir: Optional[Path] = None
        self.run_meta: dict = {}

        # Try to find real results
        if results_root is None:
            # Search relative to this file's parent (dashboard/) and one level up.
            # Also check the downloaded Colab results staging area under archive/.
            repo_root = Path(__file__).parent.parent.parent  # /HybridGuard
            self._repo_root = repo_root
            search = [
                repo_root / "results",
                repo_root / "archive" / "colab_results_20260425" / "results",
                repo_root.parent / "results",
                Path("results"),
            ]
        else:
            search = [Path(results_root)]
            self._repo_root = Path(results_root).parent

        for root in search:
            if root.exists():
                rd = _latest_run_dir(root)
                if rd is not None:
                    self.run_dir = rd
                    break

        if self.run_dir:
            self._load_live()
        else:
            self._load_demo()

        # Always try to load revision-pass artifacts (bootstrap CIs + standard ECE)
        # from paper/paper_v2_extract/. These live at a fixed manuscript-aligned
        # location and are populated by the HybridGuard_RevisionAddons.ipynb notebook.
        self.universal_recovery_ci = self._load_universal_recovery_ci()
        self.calibration_std_ece   = self._load_calibration_std_ece()

    # ── live loading ──────────────────────────────────────────────────────────
    def _load_live(self):
        rd = self.run_dir
        self.source = "live"

        # Run metadata
        meta_path = rd / "run_metadata.json"
        if meta_path.exists():
            try:
                self.run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                self.run_meta = {}

        tables = rd / "tables"

        def _csv_or_demo(name, demo_fn):
            df = _try_csv(tables / f"{name}.csv")
            return df if df is not None else demo_fn()

        self.main_results      = _patch_type(_csv_or_demo("main_results",  get_main_results))
        self.overdefense       = _patch_type(_csv_or_demo("overdefense",   get_overdefense_results))
        self.jailbreak         = _csv_or_demo("jailbreak_results",  get_jailbreak_results)
        self.robustness        = _patch_robustness(_csv_or_demo("robustness", get_robustness_results))
        self.sanitization      = _csv_or_demo("sanitization",       get_sanitization_results)
        self.ablation          = _patch_ablation(_csv_or_demo("ablation", get_ablation_results), self.main_results)
        self.fairness          = _patch_fairness(_csv_or_demo("fairness", get_fairness_results))

        # ROC + calibration: always synthesised (no CSV export in notebook)
        self.roc_curves        = get_roc_curve_data()
        self.calibration       = get_calibration_data()

    # ── demo / synthetic loading ──────────────────────────────────────────────
    def _load_demo(self):
        self.source            = "demo"
        self.main_results      = get_main_results()
        self.overdefense       = get_overdefense_results()
        self.jailbreak         = get_jailbreak_results()
        self.robustness        = get_robustness_results()
        self.sanitization      = get_sanitization_results()
        self.ablation          = get_ablation_results()
        self.fairness          = get_fairness_results()
        self.roc_curves        = get_roc_curve_data()
        self.calibration       = get_calibration_data()

    # ── revision-pass artifacts (bootstrap CIs + standard ECE) ────────────────
    def _load_universal_recovery_ci(self) -> Optional[pd.DataFrame]:
        """Load Table 9 with bootstrap 95% CIs.

        Source: paper/paper_v2_extract/ws1_universal/results_with_ci.csv
        Produced by notebooks/HybridGuard_RevisionAddons.ipynb (Section A1).
        Returns None if the file is missing — the dashboard falls back to the
        point-estimate CSV in that case.
        """
        repo = getattr(self, "_repo_root", Path(__file__).parent.parent.parent)
        path = repo / "paper" / "paper_v2_extract" / "ws1_universal" / "results_with_ci.csv"
        df = _try_csv(path)
        if df is None:
            # Fall back to the point-estimate CSV (no CIs); the dashboard still
            # renders, just without error bars.
            fallback = repo / "paper" / "paper_v2_extract" / "ws1_universal" / "results.csv"
            df = _try_csv(fallback)
            if df is not None:
                # Add empty CI columns so downstream code doesn't have to special-case.
                for col, src in [
                    ("auroc_lo", "auroc"), ("auroc_hi", "auroc"),
                    ("recall_lo", "recall_at_1pctfpr"), ("recall_hi", "recall_at_1pctfpr"),
                ]:
                    df[col] = df[src]
        return df

    def _load_calibration_std_ece(self) -> Optional[pd.DataFrame]:
        """Load standard confidence-based ECE (Guo et al. 2017) per detector.

        Source: paper/paper_v2_extract/evaluation/in_domain/calibration_std_ece.csv
        Produced by notebooks/HybridGuard_RevisionAddons.ipynb (Section A2).
        Returns None if the file is missing (the existing custom-binning ECE in
        the calibration view remains unaffected).
        """
        repo = getattr(self, "_repo_root", Path(__file__).parent.parent.parent)
        path = repo / "paper" / "paper_v2_extract" / "evaluation" / "in_domain" / "calibration_std_ece.csv"
        return _try_csv(path)

    # ── convenience ───────────────────────────────────────────────────────────
    def info_banner(self) -> str:
        bits = []
        if self.source == "live":
            rd = str(self.run_dir)
            mode = self.run_meta.get("config", {}).get("FAST_MODE", "?")
            ts   = self.run_meta.get("start_time_iso", "unknown")
            bits.append(f"📂 Live run · {rd} · started {ts} · FAST_MODE={mode}")
        else:
            bits.append("🔬 Demo mode – showing synthetic results (run the notebook to see live data)")
        if self.universal_recovery_ci is not None and "auroc_lo" in self.universal_recovery_ci.columns and self.universal_recovery_ci["auroc_lo"].nunique() > 1:
            bits.append("✓ Bootstrap CIs loaded")
        if self.calibration_std_ece is not None:
            bits.append("✓ Standard ECE loaded")
        return " · ".join(bits)
