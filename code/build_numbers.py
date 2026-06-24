"""
build_numbers.py — WS6: consolidate every experiment artifact into ONE source of
truth, results_COSE/NUMBERS.json.

It merges:
  * every numbers_snapshot_*.json under results_COSE/  (machine-readable scalars)
  * every *_agg.csv under results_COSE/                (5-seed aggregates:
                                                        mean,std,ci_lo,ci_hi,n_seeds)

and records, per declared experiment, whether real data is present or the cell is
still PENDING (e.g. the SOTA/CANOPI cells that require a GPU run). Nothing is
invented: an experiment with no artifact is listed under `_meta.pending`, never
filled with a placeholder number. emit_tables.py and check_consistency.py both read
ONLY this file, so the manuscript and the data cannot silently diverge.

Usage:  python -m build_numbers   (run from code/, or import and call build())
"""
from __future__ import annotations

import csv
import glob
import json
import os
from typing import Dict, List

# Experiments the manuscript reports. Each maps to the snapshot/agg basename(s) that,
# when present under results_COSE/, supply its numbers. Missing -> pending.
DECLARED = {
    "certificate": ["numbers_snapshot_certificate"],
    "adaptive":    ["numbers_snapshot_adaptive_v2", "adaptive_v2_matrix_agg"],
    "in_domain":   ["main_results_agg", "numbers_snapshot_diamond"],
    "recovery":    ["recovery_matrix_agg"],
    "multilingual":["numbers_snapshot_mling", "linguistic_recovery_agg"],
    "overdefense": ["numbers_snapshot_overdef", "overdefense_agg"],
}

SEEDS = [42, 2025, 7, 1337, 314]

# Subtrees that are NOT manuscript-canonical and must never feed NUMBERS.json:
#   seed_*           per-seed raw matrices (the aggregate is the canonical artifact)
#   cpu_validation/  CPU-only mechanism validation on the synthetic corpus
_EXCLUDE = ("cpu_validation", "seed_")


def _is_excluded(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    return any(seg == "cpu_validation" or seg.startswith("seed_") for seg in parts)


def _load_json_snapshots(root: str) -> Dict[str, dict]:
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "numbers_snapshot_*.json"), recursive=True)):
        if _is_excluded(p):
            continue
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            out[name] = json.load(open(p))
        except Exception as e:  # pragma: no cover
            out[name] = {"_error": str(e)}
    return out


def _load_agg_csvs(root: str) -> Dict[str, List[dict]]:
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "*_agg.csv"), recursive=True)):
        if _is_excluded(p):
            continue
        name = os.path.splitext(os.path.basename(p))[0]
        with open(p, newline="") as f:
            out[name] = list(csv.DictReader(f))
    return out


def build(root: str = "results_COSE", out_path: str = "results_COSE/NUMBERS.json") -> dict:
    snapshots = _load_json_snapshots(root)
    aggregates = _load_agg_csvs(root)
    present = set(snapshots) | set(aggregates)

    sections, pending = {}, []
    for exp, basenames in DECLARED.items():
        have = [b for b in basenames if b in present]
        sec = {}
        for b in have:
            if b in snapshots:
                sec[b] = snapshots[b]
            if b in aggregates:
                sec[b] = aggregates[b]
        if sec:
            sections[exp] = sec
        else:
            pending.append(exp)

    numbers = {
        "_meta": {
            "generated_by": "code/build_numbers.py",
            "source_root": root,
            "seeds": SEEDS,
            "snapshots_merged": sorted(snapshots),
            "aggregates_merged": sorted(aggregates),
            "pending": sorted(pending),
            "pending_note": ("Experiments with no artifact under results_COSE/ yet. "
                             "The SOTA/CANOPI cells require a GPU run (transformers + "
                             "sentence-transformers); the certificate and stats are CPU."),
        },
        **sections,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(numbers, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"NUMBERS.json <- {len(snapshots)} snapshots + {len(aggregates)} aggregates; "
          f"sections={sorted(sections)}; pending={sorted(pending)}")
    return numbers


if __name__ == "__main__":
    build()
