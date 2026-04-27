"""
Regenerate every figure the paper currently uses.

Usage:
    python scripts/make_paper_figures.py

This is the single entry point for paper-figure regeneration. Each individual
figure has its own script in scripts/ that reads a CSV from
paper/paper_v2_extract/<topic>/ and writes a {.png, .pdf} pair into
paper/figures/<name>.{png,pdf}.

To add a new figure:
  1. Save the source data as a CSV under paper/paper_v2_extract/<topic>/
  2. Write scripts/make_<topic>_figure.py reading that CSV
     (use scripts/make_ws1_universal_figure.py as a template)
  3. Add an entry to FIGURES below.
  4. Reference the figure in main.tex via \\includegraphics{figures/<name>}
     (no extension — pdflatex picks the PDF automatically).

Running this script is a no-op if all CSVs are already present and unchanged.
It will fail loud if any input CSV is missing.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

# Add new figures here as the paper grows.
# Each entry: (script_filename, short_label_for_logging)
FIGURES = [
    ("make_ws1_universal_figure.py", "Fig 1 — WS1 universal recovery"),
    ("make_llm_judge_figure.py",     "Fig 2 — cost-quality Pareto frontier (LLM-as-judge)"),
    # ("make_robustness_figure.py",     "Fig 3 — robustness with vs without canonicalization"),
    # ("make_calibration_figure.py",    "Fig 4 — calibration reliability diagrams"),
]


def run_one(script_name: str, label: str) -> bool:
    script = SCRIPTS / script_name
    print(f"\n=== {label} ===")
    if not script.exists():
        print(f"  skipped — {script.name} not found")
        return False
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, end="", file=sys.stderr)
        return False
    return True


def main() -> None:
    ok = 0
    fail = 0
    for script_name, label in FIGURES:
        if run_one(script_name, label):
            ok += 1
        else:
            fail += 1

    print(f"\n=== summary: {ok} ok, {fail} failed ===")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
