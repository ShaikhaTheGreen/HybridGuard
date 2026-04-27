#!/usr/bin/env bash
# ============================================================
# pack_overleaf.sh
# ------------------------------------------------------------
# Build a self-contained zip of paper/ ready to upload to Overleaf.
#
# Usage (from repo root):
#     ./scripts/pack_overleaf.sh
#
# Produces: HybridGuard_overleaf_<YYYYMMDD>.zip in the repo root.
# Open https://www.overleaf.com, click "New Project" -> "Upload Project",
# drag in the zip. Overleaf will auto-detect main.tex; the Springer sn-jnl.cls
# and sn-mathphys-num.bst ship in the zip so no template selection is needed.
#
# Implementation note: uses Python (zipfile module) rather than the `zip`
# CLI so it works identically in any sandbox / Colab / Mac environment.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ ! -d "paper" ]; then
    echo "[pack] ERROR: paper/ folder not found at ${REPO_ROOT}/paper" >&2
    exit 1
fi

STAMP="$(date +%Y%m%d)"
ZIP_NAME="HybridGuard_overleaf_${STAMP}.zip"

python3 - <<PYEOF
import os, shutil, sys, tempfile, zipfile
from pathlib import Path

REPO = Path('.')
zip_name = '${ZIP_NAME}'

required = [
    ('paper/main.tex',                                      'main.tex'),
    ('paper/sn-bibliography.bib',                           'sn-bibliography.bib'),
    ('paper/sn-jnl.cls',                                    'sn-jnl.cls'),
    ('paper/sn-mathphys-num.bst',                           'sn-mathphys-num.bst'),
    ('paper/sn-basic.bst',                                  'sn-basic.bst'),
    ('paper/sn-aps.bst',                                    'sn-aps.bst'),
    ('paper/paper_v2_extract/tables_v2/canonical_recovery.tex',
     'paper_v2_extract/tables_v2/canonical_recovery.tex'),
]
optional = [
    ('paper/main.pdf', 'main.pdf'),
]

stage = Path(tempfile.mkdtemp(prefix='hgo_'))
try:
    missing = []
    for src, dst in required:
        sp = REPO / src
        if not sp.exists():
            missing.append(src); continue
        dp = stage / dst
        dp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sp, dp)
    if missing:
        print(f'[pack] ERROR: required files missing:', file=sys.stderr)
        for m in missing: print(f'  - {m}', file=sys.stderr)
        sys.exit(2)
    for src, dst in optional:
        sp = REPO / src
        if sp.exists():
            dp = stage / dst
            dp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sp, dp)

    # Include any PNG/PDF/JPG figures under paper/figures/
    fig_src_dir = REPO / 'paper' / 'figures'
    if fig_src_dir.exists():
        for f in sorted(fig_src_dir.iterdir()):
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.pdf', '.eps'):
                dp = stage / 'figures' / f.name
                dp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dp)

    (stage / 'README.md').write_text(
        '# HybridGuard - Overleaf bundle\n\n'
        'Main file: main.tex. Class: sn-jnl.cls. Bibliography: sn-bibliography.bib.\n'
        'Compile with default pdflatex + bibtex + pdflatex + pdflatex on Overleaf.\n\n'
        'Code implementation: https://github.com/ShaikhaTheGreen/HybridGuard\n',
        encoding='utf-8'
    )

    zp = REPO / zip_name
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(stage):
            for f in sorted(files):
                full = Path(root) / f
                rel = full.relative_to(stage)
                z.write(full, arcname=str(rel))
    size_kb = zp.stat().st_size / 1024
    print(f'[pack] done: {zip_name} ({size_kb:.1f} KB)')
    print(f'[pack] upload at https://www.overleaf.com -> New Project -> Upload Project')
finally:
    shutil.rmtree(stage, ignore_errors=True)
PYEOF
