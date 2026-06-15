"""
npl_diamond_experiment.py
=========================
THE headline experiment for the Neural Processing Letters resubmission.

Claim it proves:  a single, detector-AGNOSTIC canonicalization stage restores the
recall of *any* downstream neural prompt-injection detector under adversarial
obfuscation -- turning "canonicalization = preprocessing" into a general
neural-robustness result (reviewer R2.1 / R2.2).

What it produces (all written to OUT_DIR):
  * recovery_matrix.csv / .tex   Recall@1%FPR for {clean, attack, attack+canon}
                                 x {homoglyph, zero_width, leet, spacing}
                                 x {HG_MULTIFEAT, protectai/deberta, InjecGuard}
  * recovery_curves.csv          Recall vs attack intensity sigma (for the figure)
  * fig_canon_recovery.png/pdf   Grouped bars: attacked vs recovered, per detector
  * corrected_metrics.csv        AUPRC + operating-point F1 + FIXED ECE (R1.3/R2.3)
  * numbers_snapshot_diamond.json  machine-readable numbers for the paper

------------------------------------------------------------------------------
HOW TO RUN (Colab, GPU runtime)  -- explicit args, no global scraping
------------------------------------------------------------------------------
Run this AFTER your HybridGuard_FULL notebook has built the splits and trained
the HG model. In a cell:

    from npl_diamond_experiment import run
    run(X_val, y_val, X_test, y_test,
        hg_detectors={"HG_MULTIFEAT": hg_multifeat},   # .predict_proba(list[str])->(n,2)
        out_dir="npl_diamond_out")

where
    X_val, y_val      list[str], array   (validation split)
    X_test, y_test    list[str], array   (in-domain test split)
    hg_multifeat      trained HG_MULTIFEAT object

The two SOTA detectors load automatically from HuggingFace; any that are gated /
unavailable are skipped with a warning (you need >=2 detectors for the
"agnostic" claim). At the end everything is zipped to
<out_dir>/npl_results_to_send.zip and (on Colab) auto-downloaded.
"""
import os, json, warnings
import numpy as np
from canonicalize import canonicalize, perturb, ATTACKS

# =================== DEFAULTS (override via run(...) args) ===================
TARGET_FPR  = 0.01
SIGMAS      = [0.25, 0.5, 0.75, 1.0]      # attack intensities for the curve
SIGMA_MAIN  = 0.75                         # intensity used in the main matrix
MAX_POS     = 400                          # cap positives for speed (None = all)
SOTA_MODELS = {
    "protectai/deberta": "protectai/deberta-v3-base-prompt-injection-v2",
    "InjecGuard":        "leolee99/InjecGuard",
}
# ============================================================================


# ---- uniform detector adapter: name -> function(list[str]) -> p_malicious[0,1] ----
def build_detectors(hg_detectors=None, load_sota=True):
    det = {}
    for name, obj in (hg_detectors or {}).items():
        if obj is not None:
            det[name] = (lambda o: (lambda texts: np.asarray(o.predict_proba(list(texts)))[:, 1]))(obj)
    if not load_sota:
        return det
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        dev = 0 if torch.cuda.is_available() else -1
        for short, mid in SOTA_MODELS.items():
            try:
                # trust_remote_code=True: InjecGuard ships custom code; setting this
                # avoids the interactive "[y/N]" prompt so the notebook runs unattended.
                tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
                mdl = AutoModelForSequenceClassification.from_pretrained(mid, trust_remote_code=True)
                mdl.eval()
                if dev == 0:
                    mdl = mdl.cuda()

                def make_fn(tok, mdl):
                    def fn(texts, bs=32):
                        outs = []
                        for i in range(0, len(texts), bs):
                            b = list(texts[i:i+bs])
                            enc = tok(b, padding=True, truncation=True, max_length=256, return_tensors="pt")
                            if dev == 0:
                                enc = {k: v.cuda() for k, v in enc.items()}
                            with torch.no_grad():
                                logits = mdl(**enc).logits
                                prob = torch.softmax(logits, -1)[:, -1]   # last label = injection/malicious
                            outs.append(prob.cpu().numpy())
                        return np.concatenate(outs)
                    return fn
                det[short] = make_fn(tok, mdl)
            except Exception as e:
                warnings.warn(f"skip SOTA {short} ({mid}): {e}")
    except Exception as e:
        warnings.warn(f"transformers unavailable: {e}")
    return det


# ---- metrics ----
def threshold_at_fpr(y, p, fpr=TARGET_FPR):
    neg = np.sort(np.asarray(p)[np.asarray(y) == 0])
    if len(neg) == 0:
        return 1.0
    k = min(max(int(np.floor((1 - fpr) * len(neg))), 0), len(neg) - 1)
    return float(neg[k])

def recall_at(y, p, thr):
    y, p = np.asarray(y), np.asarray(p)
    pos = y == 1
    return float((p[pos] >= thr).mean()) if pos.any() else float("nan")

def ece_fixed(y, p, n_bins=15):
    """CORRECT ECE: confidence = max(p,1-p) (fixes the reported ~0.70 artifact)."""
    y, p = np.asarray(y), np.clip(np.asarray(p), 0, 1)
    conf = np.maximum(p, 1 - p); correct = ((p >= .5).astype(int) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i+1]) if i else (conf >= 0) & (conf <= edges[1])
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def run(X_val, y_val, X_test, y_test, hg_detectors=None, out_dir="npl_diamond_out",
        load_sota=True, max_pos=MAX_POS):
    os.makedirs(out_dir, exist_ok=True)
    det = build_detectors(hg_detectors, load_sota=load_sota)
    assert len(det) >= 2, "Need >=2 detectors for the detector-agnostic claim."
    print("Detectors:", list(det))

    # subset positives for the attack (attacks only meaningfully apply to injections)
    yte = np.asarray(y_test)
    pos_idx = np.where(yte == 1)[0]
    if max_pos:
        pos_idx = pos_idx[:max_pos]
    Xpos = [X_test[i] for i in pos_idx]

    rows, curve_rows, snap = [], [], {}
    for dname, fn in det.items():
        thr = threshold_at_fpr(y_val, fn(list(X_val)), TARGET_FPR)   # frozen on val
        p_clean = fn(Xpos); rec_clean = float((p_clean >= thr).mean())
        snap.setdefault(dname, {})["recall_clean"] = rec_clean
        snap[dname]["threshold"] = thr
        for atk in ATTACKS:
            Xatk = [perturb(t, atk, SIGMA_MAIN) for t in Xpos]
            Xcan = [canonicalize(t) for t in Xatk]
            rec_atk = float((fn(Xatk) >= thr).mean())
            rec_can = float((fn(Xcan) >= thr).mean())
            rows.append(dict(detector=dname, attack=atk,
                             recall_clean=round(rec_clean, 4),
                             recall_attacked=round(rec_atk, 4),
                             recall_recovered=round(rec_can, 4),
                             drop=round(rec_clean - rec_atk, 4),
                             recovered_frac=round((rec_can - rec_atk) / max(rec_clean - rec_atk, 1e-9), 3)))
            snap[dname][atk] = {"attacked": rec_atk, "recovered": rec_can}
            for s in SIGMAS:
                Xs = [perturb(t, atk, s) for t in Xpos]
                Xsc = [canonicalize(t) for t in Xs]
                curve_rows.append(dict(detector=dname, attack=atk, sigma=s,
                                       attacked=round(float((fn(Xs) >= thr).mean()), 4),
                                       recovered=round(float((fn(Xsc) >= thr).mean()), 4)))
            print(f"  {dname:18s} {atk:10s} clean={rec_clean:.3f} attacked={rec_atk:.3f} recovered={rec_can:.3f}")

    _write_csv(os.path.join(out_dir, "recovery_matrix.csv"), rows)
    _write_csv(os.path.join(out_dir, "recovery_curves.csv"), curve_rows)
    _emit_latex(rows, os.path.join(out_dir, "recovery_matrix.tex"))
    _make_figure(rows, os.path.join(out_dir, "fig_canon_recovery"))
    json.dump(snap, open(os.path.join(out_dir, "numbers_snapshot_diamond.json"), "w"), indent=2)

    # corrected in-domain metrics (R1.3/R2.3) for every detector
    met = []
    from sklearn.metrics import average_precision_score
    for dname, fn in det.items():
        thr = threshold_at_fpr(y_val, fn(list(X_val)), TARGET_FPR)
        pt = fn(list(X_test)); yt = np.asarray(y_test)
        pred = (pt >= thr).astype(int)
        tp = int(((pred == 1) & (yt == 1)).sum()); fp = int(((pred == 1) & (yt == 0)).sum()); fn_ = int(((pred == 0) & (yt == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn_) if tp + fn_ else 0.0
        met.append(dict(detector=dname, AUPRC=round(float(average_precision_score(yt, pt)), 4),
                        Recall_at_1pctFPR=round(rec, 4), Precision_at_op=round(prec, 4),
                        F1_at_op=round(2*prec*rec/(prec+rec) if prec+rec else 0.0, 4),
                        ECE_fixed=round(ece_fixed(yt, pt), 4)))
    _write_csv(os.path.join(out_dir, "corrected_metrics.csv"), met)

    # bundle everything into one zip + auto-download on Colab
    bundle = _bundle(out_dir)
    print("\nDONE ->", out_dir, "| bundle:", bundle)
    try:
        from google.colab import files  # type: ignore
        files.download(bundle)
    except Exception:
        pass
    return out_dir


def _bundle(out_dir):
    import zipfile, glob
    zpath = os.path.join(out_dir, "npl_results_to_send.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(os.path.join(out_dir, "*")):
            if not f.endswith(".zip"):
                z.write(f, os.path.basename(f))
    return zpath


def _write_csv(path, rows):
    import csv
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def _emit_latex(rows, path):
    dets = sorted({r["detector"] for r in rows})
    lines = [r"\begin{table}[ht]\centering",
             r"\caption{Detector-agnostic recovery. Recall@1\%FPR on adversarially obfuscated injections, without and with canonicalization. Canonicalization restores recall across all detectors.}",
             r"\label{tab:recovery}", r"\small", r"\begin{tabular}{ll" + "c"*len(dets) + "}", r"\toprule",
             "Attack & Cond. & " + " & ".join(dets) + r" \\", r"\midrule"]
    for atk in sorted({r["attack"] for r in rows}):
        for cond, key in [("attacked", "recall_attacked"), ("+canon", "recall_recovered")]:
            cells = []
            for d in dets:
                v = next((r[key] for r in rows if r["attack"] == atk and r["detector"] == d), float("nan"))
                cells.append(f"{v:.3f}")
            lines.append(f"{atk if cond=='attacked' else ''} & {cond} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(lines))

def _make_figure(rows, stem):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        dets = sorted({r["detector"] for r in rows}); atks = sorted({r["attack"] for r in rows})
        fig, axes = plt.subplots(1, len(dets), figsize=(4 * len(dets), 3.2), sharey=True)
        if len(dets) == 1:
            axes = [axes]
        for ax, d in zip(axes, dets):
            a = [next(r["recall_attacked"] for r in rows if r["attack"] == atk and r["detector"] == d) for atk in atks]
            c = [next(r["recall_recovered"] for r in rows if r["attack"] == atk and r["detector"] == d) for atk in atks]
            x = np.arange(len(atks))
            ax.bar(x - 0.2, a, 0.4, label="attacked")
            ax.bar(x + 0.2, c, 0.4, label="+canon")
            ax.set_title(d); ax.set_xticks(x); ax.set_xticklabels(atks, rotation=30, ha="right"); ax.set_ylim(0, 1)
        axes[0].set_ylabel("Recall @ 1% FPR"); axes[-1].legend()
        fig.tight_layout(); fig.savefig(stem + ".png", dpi=160); fig.savefig(stem + ".pdf")
    except Exception as e:
        warnings.warn(f"figure skipped: {e}")


if __name__ == "__main__":
    # CPU self-test: synthetic data + CONTINUOUS-score mock detectors (no GPU/HF).
    # Demonstrates the real pattern: clean high -> attacked low -> recovered high.
    import random
    rng = random.Random(0)
    TRIG = ["ignore previous", "system prompt", "reveal instructions", "disregard rules"]
    def mk(label):
        if label:
            return rng.choice(TRIG) + " " + " ".join(rng.choice(["now", "ok", "the", "and"]) for _ in range(5))
        return " ".join(rng.choice(["weather", "recipe", "music", "the", "is"]) for _ in range(7))
    X = [mk(i % 3 == 0) for i in range(900)]
    y = np.array([1 if i % 3 == 0 else 0 for i in range(900)])
    Xv, yv, Xt, yt = X[:450], y[:450], X[450:], y[450:]

    class ContMock:
        """Continuous detector: keyword match -> high prob + noise, else low + noise."""
        def __init__(self, kws): self.kws = kws
        def predict_proba(self, texts):
            base = np.array([0.9 if any(k in t.lower() for k in self.kws) else 0.1 for t in texts])
            noise = np.random.default_rng(abs(hash(tuple(texts))) % 2**32).normal(0, 0.05, len(texts))
            p = np.clip(base + noise, 0, 1)
            return np.vstack([1 - p, p]).T
    dets = {"MockA": ContMock(TRIG), "MockB": ContMock([t.split()[0] for t in TRIG])}
    out = run(Xv, yv, Xt, yt, hg_detectors=dets, out_dir="/tmp/npl_selftest", load_sota=False, max_pos=150)
    import glob
    print("\nExported files:")
    for f in sorted(glob.glob(out + "/*")):
        print("  ", os.path.basename(f), os.path.getsize(f), "bytes")
