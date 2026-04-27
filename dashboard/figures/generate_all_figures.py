"""
generate_all_figures.py
========================
Convenience script – runs all individual figure generators in sequence
and prints a summary of saved files.

Run:  python figures/generate_all_figures.py
"""

import importlib.util
import sys
from pathlib import Path

SCRIPTS = [
    "fig_roc_curves",
    "fig_calibration",
    "fig_ablation",
    "fig_sanitization",
    "fig_robustness",
]

def run_script(name: str):
    script_path = Path(__file__).parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Generating all HybridGuard publication figures")
    print("="*55)
    for s in SCRIPTS:
        print(f"\n▶ {s}")
        try:
            run_script(s)
        except Exception as exc:
            print(f"  ✗ Failed: {exc}")
    print("\n" + "="*55)
    print("  All figures written to:  figures/")
    print("="*55 + "\n")
