#!/usr/bin/env python3
"""
make_colab_notebook.py — emit Q1_Refresh_2026-06/CANOPI_COSE_Colab.ipynb.

The notebook drives the REAL GPU pipeline end-to-end on Colab: it loads xTRam1,
builds the four heterogeneous detectors, trains CANOPI + the B5/B6/B7 ablations over
the five frozen seeds, runs the diamond recovery and the defense-aware adaptive LOFO,
the multilingual and over-defense studies, the certificate (now with the SOTA
detectors in the invariance check), then consolidates to NUMBERS.json, regenerates
the manuscript tables, runs the drift gate, and downloads everything (with a Drive
checkpoint after every section so an interrupted session loses nothing).

Run:  python scripts/make_colab_notebook.py
"""
import json
import os

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                  "source": src.strip("\n").splitlines(keepends=True)})


# ----------------------------------------------------------------------------
md(r"""# Certified Canonicalization (Computers & Security) — GPU result run

This notebook produces **every number the manuscript needs that requires a GPU**:
`tab:main` (in-domain, CANOPI + B5/B6/B7 + four detectors), `tab:adaptive`
(defense-aware leave-one-family-out), the detector-agnostic recovery matrix, the
over-defense and multilingual studies, and the certificate's per-detector invariance
across the real SOTA detectors. It then consolidates to `results_COSE/NUMBERS.json`,
regenerates the `.tex` tables, runs the drift gate, and downloads everything.

**Determinism is fixed by the code**: seeds `{42, 2025, 7, 1337, 314}`, the xTRam1
60/20/20 split frozen at seed 1337, thresholds frozen on validation at 1% FPR. Nothing
here invents numbers — each cell writes artifacts under `results_COSE/` that
`build_numbers.py` reads.

**Runtime**: a few hours on an A100 (CANOPI's NLLB back-translation augmentation is the
slow part). A **Drive checkpoint is written after every section**, so if the session
drops you can re-run from where it stopped and still download partial results.

### Before you start
1. Runtime → Change runtime type → **GPU (A100/L4/T4)**.
2. Run the cells top to bottom. Cell 2 asks you to **upload `cose_colab_bundle.zip`**
   (regenerate it locally with `python scripts/make_colab_bundle.py` so it has the
   latest code), or set `USE_GIT = True` to clone the repo instead.
""")

# --- 1. install -------------------------------------------------------------
md("## 1 · Install dependencies")
code(r"""
# Colab GPU images ship torch; we add the rest. (~2-3 min)
!pip -q install "transformers>=4.41" "sentence-transformers>=2.7" "datasets>=2.20" scikit-learn scipy pandas matplotlib confusable_homoglyphs accelerate sentencepiece sacremoses xxhash pyyaml >/dev/null 2>&1
print("deps installed")
""")

# --- 2. get code ------------------------------------------------------------
md("""## 2 · Get the code

Upload `cose_colab_bundle.zip` (default) **or** set `USE_GIT=True` to clone. The bundle
is the reliable path because it carries your latest local edits, which may not be
pushed.""")
code(r"""
import os, sys, zipfile, subprocess
USE_GIT = False                      # True -> git clone instead of zip upload
GIT_URL = "https://github.com/ShaikhaTheGreen/HybridGuard.git"
GIT_BRANCH = "canopi-npl"
HG = "/content/hg"
os.makedirs(HG, exist_ok=True)

if USE_GIT:
    subprocess.run(["rm", "-rf", HG], check=False)
    subprocess.run(["git", "clone", "--branch", GIT_BRANCH, "--depth", "1", GIT_URL, HG], check=True)
else:
    from google.colab import files
    up = files.upload()                       # choose cose_colab_bundle.zip
    zname = next(iter(up))
    with zipfile.ZipFile(zname) as z:
        z.extractall(HG)
    # if the zip nested everything under a top folder, descend into it
    entries = [d for d in os.listdir(HG) if os.path.isdir(os.path.join(HG, d))]
    if "code" not in os.listdir(HG) and len(entries) == 1:
        HG = os.path.join(HG, entries[0])

os.chdir(HG)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=False)
# Import directly from the source tree (robust even if the editable install above
# does not take on Colab): hybridguard lives under src/, the flat experiment modules
# under code/.
for sub in ("src", "code"):
    p = os.path.join(HG, sub)
    if p not in sys.path:
        sys.path.insert(0, p)
import hybridguard  # noqa: F401  -- fail loudly here, not three cells later
print("repo at", HG, "| hybridguard ->", hybridguard.__file__)
""")

# --- 3. env + checkpoint ----------------------------------------------------
md("## 3 · GPU check, seeds, and the Drive checkpoint helper")
code(r"""
import os, json, time, shutil
import numpy as np
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(CPU only!)")

SEEDS = [42, 2025, 7, 1337, 314]
RUN_CANOPI = True          # set False to skip the learned head (four-detector run only)
FAST_DEBUG = False         # True -> 1 seed, tiny epochs, no NLLB (smoke test only)
if FAST_DEBUG:
    SEEDS = [42]

RESULTS = os.path.join(HG, "results_COSE")
os.makedirs(RESULTS, exist_ok=True)

# Drive checkpointing so an interrupted session never loses finished work.
try:
    from google.colab import drive
    drive.mount("/content/drive")
    CKPT_DIR = "/content/drive/MyDrive/CANOPI_COSE_results"
except Exception as e:
    CKPT_DIR = os.path.join(HG, "checkpoints")
    print("Drive unavailable, checkpointing locally:", e)
os.makedirs(CKPT_DIR, exist_ok=True)

STATUS = {}
def checkpoint(tag):
    z = shutil.make_archive(os.path.join(CKPT_DIR, f"results_COSE_{tag}"), "zip", RESULTS)
    print(f"  [checkpoint] {tag} -> {z}")
def mark(section, ok, detail=""):
    STATUS[section] = {"ok": bool(ok), "detail": str(detail)}
    print(f"  [{ '✓' if ok else '✗' }] {section}: {detail}")
print("seeds:", SEEDS, "| RUN_CANOPI:", RUN_CANOPI, "| FAST_DEBUG:", FAST_DEBUG)
""")

# --- 4. data ----------------------------------------------------------------
md("## 4 · Load xTRam1 (frozen 60/20/20 split) and auxiliary corpora")
code(r"""
from hybridguard.canopi import data
df = data.load_xtram1()
splits, leak = data.prepare_dataset(df, seed=1337)
Xtr, ytr = splits.xy("train"); Xv, yv = splits.xy("val"); Xt, yt = splits.xy("test")
ytr, yv, yt = map(lambda a: np.asarray(a, int), (ytr, yv, yt))
print(f"train/val/test = {len(Xtr)}/{len(Xv)}/{len(Xt)}  leakage_clean={leak.get('clean')}")

# auxiliary corpora (best-effort; used by over-defense / transfer)
AUX = {}
for name, loader in [("deepset", data.load_deepset), ("notinject", data.load_notinject),
                     ("jbb", data.load_jbb)]:
    try:
        AUX[name] = loader(); print(f"loaded {name}: {len(AUX[name])} rows")
    except Exception as e:
        print(f"skip {name}: {e}")
mark("data", True, f"{len(Xtr)}/{len(Xv)}/{len(Xt)}, leakage_clean={leak.get('clean')}")
checkpoint("01_data")
""")

# --- 5. four detectors ------------------------------------------------------
md("""## 5 · Build the four heterogeneous detectors

regex, TF-IDF+LinearSVM (trained on the xTRam1 train split), ProtectAI/DeBERTa-v3, and
InjecGuard. These are the black-box detectors the certificate, recovery and adaptive
tables wrap.""")
code(r"""
from npl_diamond_experiment import build_detectors
DET4 = build_detectors(hg_detectors=None, load_sota=True, train_data=(Xtr, ytr), seed=42)
print("detectors:", list(DET4))
mark("detectors", len(DET4) >= 3, f"{list(DET4)}")
""")

# --- 6. train CANOPI + ablations -------------------------------------------
md("""## 6 · Train CANOPI + B5/B6/B7 over five seeds (the learned reference head)

Each `train_one_seed` returns a model whose `.score(texts)` gives `p_malicious`. We keep
the per-seed models to (a) build `tab:main` and (b) add CANOPI to the recovery/adaptive
runs. **This is the slow section** (NLLB back-translation); it is fully guarded and
checkpointed per seed.""")
code(r"""
CANOPI_MODELS = {}     # name -> {seed: model}; name in {CANOPI,B5_embedding,B6_invariance,B7_pauc}
if RUN_CANOPI:
    import yaml
    from hybridguard.canopi.train import train_one_seed
    CFG = os.path.join(HG, "configs")
    cfg_paths = {
        "CANOPI":        os.path.join(CFG, "canopi_main.yaml"),
        "B5_embedding":  os.path.join(CFG, "baselines", "b5_embedding.yaml"),
        "B6_invariance": os.path.join(CFG, "baselines", "b6_invariance_only.yaml"),
        "B7_pauc":       os.path.join(CFG, "baselines", "b7_pauc_only.yaml"),
    }
    def _load_cfg(p):
        c = yaml.safe_load(open(p))
        if FAST_DEBUG:                       # smoke test: cheap + no NLLB
            c.setdefault("train", {}).update({"epochs": 2})
            c.setdefault("augment", {})["crosslingual"] = []
        return c
    data_dict = {"X_train": list(Xtr), "y_train": list(ytr), "X_val": list(Xv), "y_val": list(yv)}
    for name, p in cfg_paths.items():
        CANOPI_MODELS[name] = {}
        for s in SEEDS:
            try:
                res = train_one_seed(_load_cfg(p), data_dict, seed=s)
                CANOPI_MODELS[name][s] = res.model
                print(f"  trained {name} seed={s}  val_R@1%FPR={res.val_recall_at_1pct:.3f}")
            except Exception as e:
                print(f"  !! {name} seed={s} failed: {type(e).__name__}: {e}")
        checkpoint(f"02_canopi_{name}")
    trained = {n: len(m) for n, m in CANOPI_MODELS.items()}
    mark("canopi_train", any(trained.values()), trained)
else:
    mark("canopi_train", True, "skipped (RUN_CANOPI=False)")
""")

# --- 7. in-domain tab:main --------------------------------------------------
md("""## 7 · In-domain results (`tab:main`) — 5-seed mean ± s.d. with CIs + McNemar

R@1%FPR for every detector at the validation-frozen threshold. The four black-box
detectors are deterministic (their across-seed s.d. is ~0, reported honestly); CANOPI
and its ablations vary by training seed. McNemar (Holm-corrected) compares CANOPI to
B5/B6/B7 and to the strongest neural baseline.""")
code(r"""
import csv
import stats
OUT = os.path.join(RESULTS, "in_domain"); os.makedirs(OUT, exist_ok=True)

def det_scores(detfn, texts):
    return np.asarray(detfn(list(texts)), float)

# per-seed R@1%FPR rows for all detectors
per_seed = []
score_cache = {}     # (name, seed) -> (p_val, p_test) for McNemar
for s in SEEDS:
    rows = []
    # black-box / SOTA detectors (deterministic except tfidf which is reseeded)
    det_s = build_detectors(hg_detectors=None, load_sota=True, train_data=(Xtr, ytr), seed=s)
    pool = {n: (lambda f: (lambda T: det_scores(f, T)))(f) for n, f in det_s.items()}
    # CANOPI + ablations for this seed
    for name, per in CANOPI_MODELS.items():
        if s in per:
            pool[name] = (lambda m: (lambda T: np.asarray(m.score(list(T)), float)))(per[s])
    for name, fn in pool.items():
        pv, pt = fn(Xv), fn(Xt)
        thr = stats.threshold_at_fpr(yv, pv, 0.01)
        rows.append({"detector": name,
                     "R_at_1pctFPR": round(stats.recall_at(yt, pt, thr), 4),
                     "AUROC": round(stats.auroc(yt, pt), 4),
                     "AUPRC": round(stats.auprc(yt, pt), 4)})
        score_cache[(name, s)] = (yv, pv, yt, pt, thr)
    per_seed.append(rows)

# aggregate across seeds
import run_multiseed as rm
agg = rm._aggregate(per_seed, ["detector"], ["R_at_1pctFPR", "AUROC", "AUPRC"])
# emit_tables.emit_main reads the PRIMARY metric from 'mean'/'std'; alias R@1%FPR there
for r in agg:
    r["mean"] = r["R_at_1pctFPR_mean"]
    r["std"] = r["R_at_1pctFPR_std"]
with open(os.path.join(OUT, "main_results_agg.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(agg[0])); w.writeheader(); w.writerows(agg)

# McNemar: CANOPI vs each ablation / strongest neural baseline (seed 42 test split)
mcn = {}
if ("CANOPI", SEEDS[0]) in score_cache:
    _, _, yT, pC, tC = score_cache[("CANOPI", SEEDS[0])]
    for other in ["B5_embedding", "B6_invariance", "B7_pauc", "protectai/deberta", "InjecGuard"]:
        if (other, SEEDS[0]) in score_cache:
            _, _, _, pO, tO = score_cache[(other, SEEDS[0])]
            mcn[f"CANOPI_vs_{other}"] = stats.mcnemar(yT, pC, pO, tC, tO)
    pmap = {k: v["p_exact"] for k, v in mcn.items()}
    holm = stats.holm_bonferroni(list(pmap.values())) if pmap else {}
    mcn["_holm"] = {"order": list(pmap), **holm}
json.dump({"per_seed": per_seed, "mcnemar": mcn},
          open(os.path.join(OUT, "numbers_snapshot_diamond.json"), "w"), indent=2, default=float)
print("tab:main rows:", [(r["detector"], r["R_at_1pctFPR_mean"]) for r in agg])
mark("in_domain", True, f"{len(agg)} detectors")
checkpoint("03_in_domain")
""")

# --- 8. recovery + adaptive -------------------------------------------------
md("""## 8 · Recovery matrix + defense-aware adaptive LOFO (5-seed, all detectors)

Drives the real `npl_diamond_experiment.run` and `npl_adaptive_experiment_v2.run` per
seed with the four detectors **plus CANOPI**, then aggregates with mean/std/CI. Writes
`recovery_matrix_agg.csv` and `adaptive_v2_matrix_agg.csv` (the basenames
`build_numbers` consumes) and the `fig2_adaptive_heldout` figure.""")
code(r"""
from npl_diamond_experiment import run as diamond_run
from npl_adaptive_experiment_v2 import run as adaptive_run

REC = os.path.join(RESULTS, "recovery"); ADP = os.path.join(RESULTS, "adaptive")
dia_rows, adp_rows = [], []
class _CanopiProba:
    # wrap a CANOPI model's .score() in the predict_proba interface build_detectors expects
    def __init__(self, mm): self.mm = mm
    def predict_proba(self, texts):
        p = np.asarray(self.mm.score(list(texts)), float); return np.vstack([1 - p, p]).T

for s in SEEDS:
    hg = None
    if CANOPI_MODELS.get("CANOPI", {}).get(s) is not None:
        hg = {"CANOPI": _CanopiProba(CANOPI_MODELS["CANOPI"][s])}
    diamond_run(Xv, yv, Xt, yt, hg_detectors=hg, out_dir=os.path.join(REC, f"seed_{s}"),
                load_sota=True, max_pos=400, seed=s, train_data=(Xtr, ytr))
    adaptive_run(Xv, yv, Xt, yt, hg_detectors=hg, out_dir=os.path.join(ADP, f"seed_{s}"),
                 load_sota=True, max_pos=400, restarts=6, seed=s, train_data=(Xtr, ytr))
    dia_rows.append(rm._read_csv(os.path.join(REC, f"seed_{s}", "recovery_matrix.csv")))
    adp_rows.append(rm._read_csv(os.path.join(ADP, f"seed_{s}", "adaptive_v2_matrix.csv")))
    checkpoint(f"04_recovery_adaptive_seed_{s}")

dia_agg = rm._aggregate(dia_rows, ["detector", "attack"],
                        ["recall_clean", "recall_attacked", "recall_recovered"])
adp_agg = rm._aggregate(adp_rows, ["detector", "heldout_family"], ["clean", "none", "c", "cplus"])
rm._write_csv(os.path.join(REC, "recovery_matrix_agg.csv"), dia_agg)
rm._write_csv(os.path.join(ADP, "adaptive_v2_matrix_agg.csv"), adp_agg)

# fig2: adaptive LOFO on InjecGuard (the detector most vulnerable to obfuscation)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fams = [r["heldout_family"] for r in adp_agg if r["detector"] == "InjecGuard"]
if fams:
    sub = [r for r in adp_agg if r["detector"] == "InjecGuard"]
    x = np.arange(len(sub)); w = 0.27
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, (k, lab) in enumerate([("none_mean", "no defense"), ("c_mean", "$c$"), ("cplus_mean", "$c^{+}$")]):
        ax.bar(x + (i - 1) * w, [float(r[k]) for r in sub], w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([r["heldout_family"].replace("_", " ") for r in sub], rotation=30, ha="right")
    ax.set_ylabel("worst-case R@1%FPR"); ax.set_ylim(0, 1); ax.legend(); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ADP, f"fig2_adaptive_heldout.{ext}"), dpi=160)
    figdir = os.path.join(HG, "manuscript_COSE", "figures"); os.makedirs(figdir, exist_ok=True)
    shutil.copy(os.path.join(ADP, "fig2_adaptive_heldout.pdf"), os.path.join(figdir, "fig2_adaptive_heldout.pdf"))
mark("recovery_adaptive", True, f"recovery rows={len(dia_agg)}, adaptive rows={len(adp_agg)}")
checkpoint("04_recovery_adaptive")
""")

# --- 9. over-defense + multilingual ----------------------------------------
md("""## 9 · Over-defense (NotInject) and multilingual recovery (best-effort)

Over-defense: benign FPR at the 1% operating point on NotInject, raw vs +canon, per
detector — confirms canonicalization is FPR-neutral. Multilingual: per-language
R@1%FPR raw vs after linguistic canonicalization. Both guarded; whatever loads runs.""")
code(r"""
from canonicalize import canonicalize
OD = os.path.join(RESULTS, "overdefense"); os.makedirs(OD, exist_ok=True)
try:
    nb = AUX["notinject"]
    Xb = nb["text"].tolist() if "text" in nb.columns else nb.iloc[:, 0].tolist()
    rows = []
    for name, fn in DET4.items():
        thr = stats.threshold_at_fpr(yv, np.asarray(fn(list(Xv)), float), 0.01)
        raw = float((np.asarray(fn(Xb), float) >= thr).mean())
        can = float((np.asarray(fn([canonicalize(t) for t in Xb]), float) >= thr).mean())
        rows.append({"detector": name, "benign_fpr_raw": round(raw, 4),
                     "benign_fpr_canon": round(can, 4), "delta": round(can - raw, 4)})
    rm._write_csv(os.path.join(OD, "overdefense_agg.csv"), rows)
    json.dump({"rows": rows, "fpr_neutral": all(abs(r["delta"]) < 1e-9 for r in rows)},
              open(os.path.join(OD, "numbers_snapshot_overdef.json"), "w"), indent=2)
    mark("overdefense", True, f"{len(rows)} detectors, FPR-neutral={all(abs(r['delta'])<1e-9 for r in rows)}")
except Exception as e:
    mark("overdefense", False, f"{type(e).__name__}: {e}")

# multilingual (curated AR/ES if available in the repo, else MT testset)
ML = os.path.join(RESULTS, "multilingual"); os.makedirs(ML, exist_ok=True)
try:
    from multilingual_injections import get_ml_sets    # repo-provided curated AR/ES (+romanized)
    sets = get_ml_sets()                                # {lang: (texts, labels)}; positives only
    mrows = []
    for lang, (texts, _yl) in sets.items():
        for name, fn in DET4.items():
            thr = stats.threshold_at_fpr(yv, np.asarray(fn(list(Xv)), float), 0.01)
            raw = float((np.asarray(fn(list(texts)), float) >= thr).mean())
            can = float((np.asarray(fn([canonicalize(t) for t in texts]), float) >= thr).mean())
            mrows.append({"lang": lang, "detector": name, "raw": round(raw, 4), "canon": round(can, 4)})
    rm._write_csv(os.path.join(ML, "linguistic_recovery_agg.csv"), mrows)
    json.dump({"rows": mrows}, open(os.path.join(ML, "numbers_snapshot_mling.json"), "w"), indent=2)
    mark("multilingual", True, f"{len(mrows)} rows")
except Exception as e:
    mark("multilingual", False, f"{type(e).__name__}: {e}  (curated sets optional; cross-lingual figure may come from the FULL notebook)")
checkpoint("05_overdefense_multilingual")
""")

# --- 10. certificate with SOTA ---------------------------------------------
md("""## 10 · Certificate with the real detectors in the invariance check""")
code(r"""
import run_certificate_tables as rct
snap = rct.run(out_dir=os.path.join(RESULTS, "certificate"), load_sota=True, seed=42)
mark("certificate", snap["idempotence"] == 1.0,
     f"idem={snap['idempotence']} soundness {snap['closure_soundness_base_map']}->{snap['closure_soundness_full_table']} "
     f"invariance={snap['per_detector_invariance']}")
checkpoint("06_certificate")
""")

# --- 11. consolidate --------------------------------------------------------
md("""## 11 · Consolidate → NUMBERS.json → tables → drift gate → tests""")
code(r"""
import build_numbers, emit_tables, check_consistency, importlib
for m in (build_numbers, emit_tables, check_consistency):
    importlib.reload(m)
numbers = build_numbers.build()
emit = emit_tables.emit()
gate = check_consistency.run()
print("pending sections:", numbers["_meta"]["pending"])
print("tables written:", emit.get("written"), "pending:", emit.get("pending"))
print("drift gate exit code:", gate)
mark("consolidate", True, f"pending={numbers['_meta']['pending']} drift_gate={gate}")

# regenerate RUN_REPORT and run the test suite
import subprocess
subprocess.run([sys.executable, "-m", "code.reproduce", "--seeds", "all"], cwd=HG, check=False)
print(subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=HG, capture_output=True, text=True).stdout[-1500:])
checkpoint("07_consolidate")
""")

# --- 12. download -----------------------------------------------------------
md("""## 12 · Package everything and download

Bundles `results_COSE/`, the regenerated `manuscript_COSE/tables/`, the figures,
`NUMBERS.json`, and `RUN_REPORT.md`. Also left on Drive at `CANOPI_COSE_results/`.""")
code(r"""
import shutil
stage = "/content/cose_deliverable"; shutil.rmtree(stage, ignore_errors=True)
os.makedirs(stage, exist_ok=True)
for src in ["results_COSE", os.path.join("manuscript_COSE", "tables"),
            os.path.join("manuscript_COSE", "figures"), "RUN_REPORT.md"]:
    p = os.path.join(HG, src)
    if os.path.exists(p):
        dst = os.path.join(stage, src.replace(os.sep, "__"))
        (shutil.copytree if os.path.isdir(p) else shutil.copy)(p, dst)
zip_path = shutil.make_archive("/content/CANOPI_COSE_deliverable", "zip", stage)
shutil.copy(zip_path, os.path.join(CKPT_DIR, os.path.basename(zip_path)))
print("\n=== SECTION STATUS ===")
for k, v in STATUS.items():
    print(f"  [{'✓' if v['ok'] else '✗'}] {k}: {v['detail']}")
print("\ndeliverable:", zip_path, "(also on Drive at", CKPT_DIR + ")")
try:
    from google.colab import files; files.download(zip_path)
except Exception as e:
    print("auto-download unavailable; grab it from Drive:", e)
""")

# ----------------------------------------------------------------------------
nb = {"cells": CELLS,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(here, "Q1_Refresh_2026-06", "CANOPI_COSE_Colab.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, "with", len(CELLS), "cells")
