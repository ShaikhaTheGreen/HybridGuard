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

    # CANOPI (NPL) paper figures live under scripts/ and read runs/<run_id>/.
    # Refresh them too; each generator skips gracefully if its CSV is absent.
    import subprocess
    repo_root = Path(__file__).resolve().parent.parent.parent
    paper_figs = repo_root / "scripts" / "make_paper_figures.py"
    if paper_figs.exists():
        print("\n▶ CANOPI paper figures (scripts/make_paper_figures.py)")
        try:
            subprocess.run([sys.executable, str(paper_figs)], cwd=str(repo_root))
        except Exception as exc:
            print(f"  ✗ Failed: {exc}")

    print("\n" + "="*55)
    print("  All figures written to:  figures/")
    print("="*55 + "\n")
