"""
Generate notebooks/HybridGuard_CANOPI_Orchestrator.ipynb.

The notebook imports the installed `hybridguard.canopi` package and REUSES the
data/eval protocol (it does not duplicate the orchestrator's cells). Built from a
script so it is reproducible and reviewable in diffs. Run:

    python scripts/build_canopi_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "HybridGuard_CANOPI_Orchestrator.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in lines]}


CELLS = [
    md("# HybridGuard — CANOPI Orchestrator (NPL)",
       "",
       "Trains the CANOPI projection head (frozen L0 canonicalizer + frozen encoder",
       "+ JOINT objective) over 5 seeds and runs E1, E3–E7, E10. Imports the installed",
       "`hybridguard.canopi` package and REUSES `canopi.data` (orchestrator 60/20/20 @",
       "seed 1337) + `canopi.eval` (frozen-τ protocol). No LLM at inference.",
       "",
       "**Run on Colab Pro+ (GPU):** Runtime → GPU, then Run all. Outputs land in",
       "`runs/<run_id>/` (per-seed CSV + aggregated mean±std + LaTeX tables)."),

    md("## 0 · Config — edit once"),
    code(
        "REPO_URL = 'https://github.com/ShaikhaTheGreen/HybridGuard.git'",
        "BRANCH   = 'canopi-npl'",
        "CONFIG   = 'configs/canopi_main.yaml'   # or a baselines/ ablations/ yaml",
        "MOUNT_DRIVE = True                       # persist runs/ to Drive",
        "DRIVE_DIR   = '/content/drive/MyDrive/HybridGuard'",
        "RUN_ID   = None                          # None -> derived from config name + date",
        "SEEDS    = [42, 2025, 7, 1337, 314]      # pre-registered; do not change",
        "SMOKE    = False                         # True: 1 seed + tiny subset for a dry run",
    ),

    md("## 1 · Bootstrap — Drive, repo, deps"),
    code(
        "import os, sys, subprocess",
        "if MOUNT_DRIVE:",
        "    try:",
        "        from google.colab import drive; drive.mount('/content/drive')",
        "        os.makedirs(DRIVE_DIR, exist_ok=True)",
        "    except Exception as e:",
        "        print('Drive mount skipped:', e); MOUNT_DRIVE = False",
        "ROOT = '/content/HybridGuard'",
        "if not os.path.isdir(ROOT):",
        "    subprocess.run(['git','clone','--branch',BRANCH,REPO_URL,ROOT], check=True)",
        "else:",
        "    subprocess.run(['git','-C',ROOT,'fetch','origin'], check=True)",
        "    subprocess.run(['git','-C',ROOT,'checkout',BRANCH], check=True)",
        "    subprocess.run(['git','-C',ROOT,'pull','origin',BRANCH], check=True)",
        "subprocess.run([sys.executable,'-m','pip','install','-q','-e',ROOT+'[full]'], check=True)",
        "os.chdir(ROOT); sys.path.insert(0, ROOT+'/src')",
        "print('repo at', ROOT, '| branch', BRANCH)",
    ),

    md("## 2 · Imports + determinism"),
    code(
        "import numpy as np, pandas as pd",
        "from datetime import date",
        "from hybridguard.canopi import data as D, eval as EV",
        "from hybridguard.canopi.train import load_config, set_determinism, train_one_seed",
        "from hybridguard.canopi.encoders import get_encoder",
        "from hybridguard.canopi.augment import TransformationBank, NLLBTranslator",
        "from hybridguard.canopi.attacks import SemanticRewriteAttacker",
        "from hybridguard.canopi.runs import RunWriter",
        "from hybridguard.canopi import metrics as M",
        "cfg = load_config(CONFIG)",
        "RUN_ID = RUN_ID or f\"run_{cfg['name']}_{date.today().strftime('%Y%m%d')}\"",
        "RUN_ROOT = (DRIVE_DIR + '/runs') if MOUNT_DRIVE else 'runs'",
        "writer = RunWriter(RUN_ID, root=RUN_ROOT)",
        "print('RUN_ID', writer.run_id, '->', writer.dir)",
    ),

    md("## 3 · Data — reuse orchestrator protocol (60/20/20 @ 1337, leakage-checked)",
       "If the orchestrator already populated `X_train/.../y_test` in globals, reuse",
       "those; otherwise build them with `canopi.data` (identical seed/ratios)."),
    code(
        "if all(v in globals() for v in ['X_train','y_train','X_val','y_val','X_test','y_test']):",
        "    print('Reusing orchestrator splits from globals.')",
        "    splits = None",
        "else:",
        "    df = D.load_xtram1()",
        "    if SMOKE: df = df.sample(800, random_state=1337).reset_index(drop=True)",
        "    splits, report = D.prepare_dataset(df, do_simhash=not SMOKE)",
        "    assert report['clean'], f'LEAKAGE: {report}'",
        "    writer.write_artifact_json('leakage_report', report)",
        "    X_train,y_train = splits.xy('train'); X_val,y_val = splits.xy('val'); X_test,y_test = splits.xy('test')",
        "    print('split', len(X_train), len(X_val), len(X_test), '| leakage clean:', report['clean'])",
        "data = {'X_train':X_train,'y_train':y_train,'X_val':X_val,'y_val':y_val}",
    ),

    md("## 4 · Cross-corpus + multilingual eval sets (held out from training/τ)"),
    code(
        "# deepset + JBB for threshold transfer (E4); NotInject for over-defense (E5).",
        "extra = {}",
        "for name, loader in [('deepset', D.load_deepset), ('jbb', D.load_jbb), ('notinject', D.load_notinject)]:",
        "    try: extra[name] = loader(); print(name, len(extra[name]))",
        "    except Exception as e: print('skip', name, e)",
        "# Curated AR/ES (+NLLB MT). Reuse code/multilingual_injections.py if present.",
        "ml = {}",
        "try:",
        "    sys.path.insert(0, ROOT+'/code'); from multilingual_injections import get_ml_sets",
        "    raw = get_ml_sets(include_negatives=True)",
        "    for lang,(txts,labs) in raw.items(): ml[lang] = (list(txts), list(np.asarray(labs).astype(int)))",
        "    print('multilingual sets:', {k: len(v[0]) for k,v in ml.items()})",
        "except Exception as e:",
        "    print('multilingual set unavailable (provide code/multilingual_injections.py):', e)",
        "if not ml:",
        "    print('='*64); print('  E3 (multilingual HEADLINE / F9) WILL BE SKIPPED'); ",
        "    print('  -> code/multilingual_injections.py not importable on this runtime.'); print('='*64)",
        "else:",
        "    print('E3 ready:', sum(len(v[0]) for v in ml.values()), 'multilingual prompts across', list(ml))",
    ),

    md("## 5 · Baselines (B1, B5; B2/B3 if available). B8 from a prior HG run if loaded."),
    code(
        "from sklearn.feature_extraction.text import TfidfVectorizer",
        "from sklearn.svm import LinearSVC; from sklearn.linear_model import LogisticRegression",
        "from hybridguard.canonicalize import canonicalize",
        "def _canon(ts): return [canonicalize(t).canonical for t in ts]",
        "baselines = {}",
        "# B1 TF-IDF + LinearSVM (McNemar reference)",
        "vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2)).fit(_canon(X_train))",
        "svm = LinearSVC().fit(vec.transform(_canon(X_train)), y_train)",
        "def b1_score(ts):",
        "    d = svm.decision_function(vec.transform(_canon(ts))); return 1/(1+np.exp(-d))",
        "baselines['B1_tfidf_svm'] = b1_score",
        "# B5 plain embedding clf on E(c(x)) -- no-invariance floor",
        "enc = get_encoder(cfg.get('encoder', {'backend':'auto'}))",
        "lr = LogisticRegression(max_iter=1000).fit(enc.encode(X_train), y_train)",
        "baselines['B5_embedding'] = lambda ts: lr.predict_proba(enc.encode(ts))[:,1]",
        "# B2/B3 optional HF detectors",
        "def _hf(model_id):",
        "    from transformers import pipeline; pipe = pipeline('text-classification', model=model_id, truncation=True, max_length=256)",
        "    pos = {'INJECTION','LABEL_1','jailbreak','unsafe'}",
        "    def f(ts):",
        "        out = pipe(_canon(list(ts)))",
        "        return np.array([o['score'] if o['label'] in pos else 1-o['score'] for o in out])",
        "    return f",
        "for tag, mid in [('B2_deberta','protectai/deberta-v3-base-prompt-injection-v2'),('B3_injecguard','leolee99/InjecGuard')]:",
        "    try: baselines[tag] = _hf(mid); print('loaded', tag)",
        "    except Exception as e: print('skip', tag, e)",
        "print('baselines:', list(baselines))",
    ),

    md("## 6 · Train CANOPI over 5 seeds (+ per-seed eval, frozen-τ)"),
    code(
        "seeds = [SEEDS[0]] if SMOKE else SEEDS",
        "for s in seeds:",
        "    set_determinism(s)",
        "    res = train_one_seed(cfg, data, s)",
        "    tau = res.tau; model = res.model",
        "    writer.write_artifact_json(f'accept_rates_seed{s}', res.accept_rates)",
        "    if res.aligned_pairs:",
        "        pd.DataFrame(res.aligned_pairs, columns=['en','foreign','lang']).to_csv(writer.dir/f'aligned_pairs_seed{s}.csv', index=False)",
        "    p_val = model.score(X_val); p_test = model.score(X_test)",
        "    # E1 main: CANOPI + baselines (same frozen-tau-on-val protocol)",
        "    rows = [EV.main_metrics(y_val, p_val, y_test, p_test, 'CANOPI')]",
        "    for name, fn in baselines.items():",
        "        rows.append(EV.main_metrics(y_val, fn(X_val), y_test, fn(X_test), name))",
        "    writer.write_seed(s, 'main_results', pd.DataFrame(rows))",
        "    # E3 multilingual @ each detector's OWN frozen tau (HEADLINE / F9)",
        "    if ml:",
        "        ml_rows = [EV.crosslingual_at_tau({lg:(lab, model.score(tx)) for lg,(tx,lab) in ml.items()}, tau, 'CANOPI')]",
        "        for bname, bfn in baselines.items():",
        "            tb = M.threshold_at_fpr(y_val, bfn(X_val), 0.01)   # baseline's own 1%-FPR tau",
        "            ml_rows.append(EV.crosslingual_at_tau({lg:(lab, bfn(tx)) for lg,(tx,lab) in ml.items()}, tb, bname))",
        "        writer.write_seed(s, 'crosslingual', pd.concat(ml_rows, ignore_index=True))",
        "    # E4 threshold transfer",
        "    corp = {c: (df_['label'].values, model.score(df_['text'].tolist())) for c,df_ in extra.items() if c in ('deepset','jbb')}",
        "    if corp: writer.write_seed(s, 'threshold_transfer', EV.threshold_transfer(corp, tau, 'CANOPI'))",
        "    # E5 over-defense on NotInject",
        "    if 'notinject' in extra:",
        "        nj = extra['notinject']; writer.write_seed(s, 'overdefense', EV.overdefense_sweep(y_val, p_val, nj['label'].values, model.score(nj['text'].tolist()), 'CANOPI'))",
        "    # E7 adaptive residual",
        "    atk = SemanticRewriteAttacker(model.encoder, intent_floor=cfg.get('eval',{}).get('intent_floor',0.5))",
        "    pos_test = [t for t,y in zip(X_test,y_test) if y==1][:200]",
        "    writer.write_seed(s, 'adaptive', EV.adaptive_residual(model.score, atk, pos_test, (None,None), [0,0.25,0.5,0.75,1.0], tau, 'CANOPI'))",
        "    # E10 calibration",
        "    writer.write_seed(s, 'calibration', pd.DataFrame([EV.calibration(y_test, p_test, 'CANOPI')]))",
        "    print(f'seed {s}: tau={tau:.4f} val_R@1%={res.val_recall_at_1pct:.3f}')",
    ),

    md("## 7 · Aggregate across seeds + render LaTeX tables"),
    code(
        "for name, keys in [('main_results',['model']), ('crosslingual',['model','lang']),",
        "                   ('threshold_transfer',['model','corpus']), ('overdefense',['model','fpr_target']),",
        "                   ('adaptive',['model','rewrite_strength']), ('calibration',['model'])]:",
        "    agg = writer.aggregate(name, key_cols=keys)",
        "    if agg is not None:",
        "        writer.write_table(name, agg, caption=f'CANOPI {name} (mean±std, 5 seeds)', label=f'tab:canopi_{name}')",
        "        print('aggregated', name, agg.shape)",
    ),

    md("## 8 · McNemar vs strongest neural baseline AND vs B5 (require p<0.01)"),
    code(
        "if not SMOKE:",
        "    set_determinism(SEEDS[0]); res = train_one_seed(cfg, data, SEEDS[0])",
        "    tau = res.tau; pred_c = (res.model.score(X_test) >= tau).astype(int)",
        "    mc = []",
        "    for name, fn in baselines.items():",
        "        tb = M.threshold_at_fpr(y_val, fn(X_val), 0.01); predb = (fn(X_test) >= tb).astype(int)",
        "        r = M.mcnemar(y_test, pred_c, predb); mc.append({'vs':name, **r})",
        "    writer.write_seed(SEEDS[0], 'mcnemar', pd.DataFrame(mc)); print(pd.DataFrame(mc))",
    ),

    md("## 9 · Export figure CSVs + run metadata"),
    code(
        "import subprocess, json",
        "meta = {'run_id': writer.run_id, 'config': cfg, 'seeds': seeds, 'protocol':'val-frozen tau @1%FPR',",
        "        'branch': BRANCH}",
        "try: meta['git_sha'] = subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD']).decode().strip()",
        "except Exception: pass",
        "writer.write_metadata(meta)",
        "# Mirror key tables to paper/paper_v2_extract for the figure generators.",
        "import shutil, os",
        "for name in ['main_results','crosslingual','threshold_transfer','overdefense','adaptive','calibration']:",
        "    src = writer.dir/'tables'/f'{name}.csv'",
        "    if src.exists():",
        "        dst = f'{ROOT}/paper/paper_v2_extract/canopi/{name}.csv'; os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(src, dst)",
        "print('wrote', writer.dir/'run_metadata.json')",
    ),

    md("## 10 · Generate figures + print checklist"),
    code(
        "subprocess.run([sys.executable, 'scripts/make_paper_figures.py'], cwd=ROOT)",
        "import os",
        "print('\\n=== CANOPI run checklist ===')",
        "for name in ['main_results','crosslingual','threshold_transfer','overdefense','adaptive','calibration']:",
        "    ok = (writer.dir/'aggregated'/f'{name}_mean_std.csv').exists()",
        "    print(f'  [{\"x\" if ok else \" \"}] {name}')",
        "print('runs dir:', writer.dir)",
    ),
]


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU", "colab": {"provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", OUT, f"({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
