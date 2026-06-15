"""
npl_eval_fixed.py
=================
Corrected, referee-proof evaluation for the prompt-injection guardrail study,
addressing NPL reviewer comments R1.3, R1.4, R1.5, R2.3.

Why this module exists
----------------------
The submitted tables had three measurement bugs:

  * F1 was reported at a FIXED 0.5 probability threshold. After calibration on a
    severely imbalanced set, scores are squashed toward 0, so the 0.5-threshold
    F1 collapses to 0.000 -- which is why the ablation "baseline" showed F1=0 yet
    Recall@1%FPR was 0.277. THE FIX: report precision/recall/F1 at the SAME
    operating threshold used for Recall@1%FPR (the 1%-FPR validation threshold),
    never at 0.5.

  * AUROC saturates near 0.998 under 30% prevalence and hides everything. THE FIX:
    report AUPRC and Recall@fixed-FPR as the primary metrics (R2.3).

  * ECE was reported ~0.70 while Brier was ~0.006 -- mathematically impossible,
    indicating a binning bug. THE FIX: standard 15-bin equal-width ECE below.

Thresholds are selected on the VALIDATION negatives at a target FPR and then
FROZEN for the test split (no test-set peeking). If a model file has no val
split, the script falls back to an in-sample test threshold and flags it.

Usage
-----
    python npl_eval_fixed.py --run_dir <RUN_DIR> --out metrics_fixed.csv

Outputs a tidy CSV and prints a table. Pure-Python + numpy/sklearn only.
"""
import argparse, csv, glob, os, sys
import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
except Exception:                                   # graceful fallback
    roc_auc_score = average_precision_score = brier_score_loss = None


# ----------------------------- metric primitives -----------------------------
def auroc(y, p):
    if roc_auc_score:           return float(roc_auc_score(y, p))
    order = np.argsort(p); ranks = np.empty_like(order, float); ranks[order] = np.arange(len(p))
    pos = y == 1; npos, nneg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - npos*(npos-1)/2) / (npos*nneg)) if npos and nneg else float("nan")

def auprc(y, p):
    if average_precision_score: return float(average_precision_score(y, p))
    return float("nan")

def threshold_at_fpr(y, p, target_fpr):
    """Largest threshold whose FPR on negatives is <= target (conservative)."""
    neg = np.sort(p[y == 0])
    if len(neg) == 0: return 1.0
    k = int(np.floor((1.0 - target_fpr) * len(neg)))
    k = min(max(k, 0), len(neg) - 1)
    return float(neg[k])

def prf_at_threshold(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return prec, rec, f1

def fpr_at_threshold(y, p, thr):
    neg = p[y == 0]
    return float((neg >= thr).mean()) if len(neg) else float("nan")

def ece(y, p, n_bins=15):
    """Standard equal-width ECE in [0,1]. Confidence = max(p,1-p)."""
    p = np.clip(p, 0, 1)
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.sum():
            e += (m.sum()/n) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


# ----------------------------- data loading ----------------------------------
def load(path):
    rows = list(csv.DictReader(open(path)))
    split = np.array([r.get("split", "test") for r in rows])
    y = np.array([int(float(r["y"])) for r in rows])
    p = np.array([float(r["p_malicious"]) for r in rows])
    return split, y, p


def evaluate_file(path, target_fpr=0.01):
    split, y, p = load(path)
    has_val = (split == "val").any() and (split == "test").any()
    if has_val:
        m_val, m_te = split == "val", split == "test"
        thr = threshold_at_fpr(y[m_val], p[m_val], target_fpr)
        yt, pt, src = y[m_te], p[m_te], "val->test (frozen)"
    else:
        thr = threshold_at_fpr(y, p, target_fpr)
        yt, pt, src = y, p, "test in-sample (no val saved)"
    prec, rec, f1_op = prf_at_threshold(yt, pt, thr)
    _, _, f1_05 = prf_at_threshold(yt, pt, 0.5)          # the legacy/buggy number
    return dict(
        model=os.path.basename(path).replace("pred_", "").replace("_preds", "").replace(".csv", ""),
        n_test=int(len(yt)), pos_rate=round(float(yt.mean()), 3),
        AUROC=round(auroc(yt, pt), 4), AUPRC=round(auprc(yt, pt), 4),
        thr=round(thr, 6), FPR_test=round(fpr_at_threshold(yt, pt, thr), 4),
        Recall_at_1pctFPR=round(rec, 4), Precision_at_op=round(prec, 4),
        F1_at_op=round(f1_op, 4), F1_at_0p5_LEGACY=round(f1_05, 4),
        ECE15=round(ece(yt, pt), 4), thr_source=src,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default="metrics_fixed.csv")
    ap.add_argument("--target_fpr", type=float, default=0.01)
    a = ap.parse_args()
    files = sorted(set(glob.glob(os.path.join(a.run_dir, "**", "*pred*csv"), recursive=True)))
    files = [f for f in files if "index" not in f]
    rows = []
    for f in files:
        try:
            rows.append(evaluate_file(f, a.target_fpr))
        except Exception as e:
            print(f"skip {f}: {e}", file=sys.stderr)
    rows.sort(key=lambda r: -r["Recall_at_1pctFPR"])
    cols = ["model","n_test","pos_rate","AUROC","AUPRC","Recall_at_1pctFPR",
            "Precision_at_op","F1_at_op","F1_at_0p5_LEGACY","ECE15","FPR_test","thr_source"]
    w = max(len(r["model"]) for r in rows)
    print(f"{'model':<{w}} " + " ".join(f"{c:>16}" for c in cols[1:]))
    for r in rows:
        print(f"{r['model']:<{w}} " + " ".join(f"{str(r[c]):>16}" for c in cols[1:]))
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader()
        for r in rows: wr.writerow({c: r[c] for c in cols})
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
