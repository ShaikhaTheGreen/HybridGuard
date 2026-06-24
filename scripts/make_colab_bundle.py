#!/usr/bin/env python3
"""
make_colab_bundle.py — zip the source the Colab notebook needs into
Q1_Refresh_2026-06/cose_colab_bundle.zip.

The bundle carries the CURRENT local code (which may not be pushed to GitHub), so the
notebook reproduces exactly what is on disk. Regenerate it whenever code changes:

    python scripts/make_colab_bundle.py

Included: code/, src/, configs/, tests/, conftest.py, pyproject.toml, and the
manuscript_COSE/ skeleton (so emit_tables can \\input the regenerated tables and the
adaptive figure can be copied into place). Heavy result blobs and caches are excluded.
"""
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Q1_Refresh_2026-06", "cose_colab_bundle.zip")

INCLUDE_DIRS = ["code", "src", "configs", "tests"]
INCLUDE_FILES = ["conftest.py", "pyproject.toml", "README.md", "CITATION.cff"]
# manuscript: ship the tex + an (empty) tables/ and figures/ target
MANUSCRIPT = "manuscript_COSE"

EXCLUDE_SUFFIX = (".pyc",)
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def _add_dir(z, rel):
    base = os.path.join(ROOT, rel)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn.endswith(EXCLUDE_SUFFIX):
                continue
            full = os.path.join(dirpath, fn)
            z.write(full, os.path.relpath(full, ROOT))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for d in INCLUDE_DIRS:
            if os.path.isdir(os.path.join(ROOT, d)):
                _add_dir(z, d)
        for f in INCLUDE_FILES:
            p = os.path.join(ROOT, f)
            if os.path.exists(p):
                z.write(p, f)
        # manuscript tex + tables/figures dirs (create placeholders so they exist)
        mtex = os.path.join(ROOT, MANUSCRIPT, "main_COSE.tex")
        if os.path.exists(mtex):
            z.write(mtex, os.path.join(MANUSCRIPT, "main_COSE.tex"))
        for sub in ("tables", "figures"):
            d = os.path.join(ROOT, MANUSCRIPT, sub)
            if os.path.isdir(d):
                _add_dir(z, os.path.join(MANUSCRIPT, sub))
            else:
                z.writestr(os.path.join(MANUSCRIPT, sub, ".keep"), "")
        # ship the certificate CPU snapshot so a CPU-only re-run still consolidates
        cert = os.path.join(ROOT, "results_COSE", "certificate")
        if os.path.isdir(cert):
            _add_dir(z, os.path.join("results_COSE", "certificate"))
    size = os.path.getsize(OUT) / 1e6
    print(f"wrote {OUT} ({size:.2f} MB)")


if __name__ == "__main__":
    main()
