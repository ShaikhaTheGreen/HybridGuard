"""
synthetic_data.py
=================
Generates realistic synthetic results that mirror what the HybridGuard
notebook produces.  Used when no real run artefacts are available.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(1337)

# ── model catalogue ──────────────────────────────────────────────────────────
MODELS = [
    # baselines
    "regex_heuristic",
    "tfidf_logreg",
    "tfidf_linearsvm",
    # SOTA
    "protectai_deberta",
    # HybridGuard variants
    "HG_CNN_TRANS_seed1337",
    "HG_ENSEMBLE_seed1337",
    "HG_MULTIFEAT_seed1337",
    "HG_RAV_seed1337",
]

MODEL_TYPE = {
    "regex_heuristic":     "baseline",
    "tfidf_logreg":        "baseline",
    "tfidf_linearsvm":     "baseline",
    "protectai_deberta":   "sota",
    "HG_CNN_TRANS_seed1337":  "hybridguard",
    "HG_ENSEMBLE_seed1337":   "hybridguard",
    "HG_MULTIFEAT_seed1337":  "hybridguard",
    "HG_RAV_seed1337":        "hybridguard",
}

# Approximate realistic numbers
_BASE = {
    "regex_heuristic":    dict(auroc=0.78, auprc=0.74, f1=0.72, precision=0.79, recall=0.66, recall_1pct_fpr=0.58, ece=0.22, brier=0.20, ece_cal=0.08, brier_cal=0.13, latency_ms=0.8),
    "tfidf_logreg":       dict(auroc=0.87, auprc=0.84, f1=0.82, precision=0.84, recall=0.80, recall_1pct_fpr=0.71, ece=0.14, brier=0.14, ece_cal=0.05, brier_cal=0.11, latency_ms=1.2),
    "tfidf_linearsvm":    dict(auroc=0.89, auprc=0.86, f1=0.84, precision=0.86, recall=0.82, recall_1pct_fpr=0.74, ece=0.15, brier=0.13, ece_cal=0.06, brier_cal=0.11, latency_ms=1.1),
    "protectai_deberta":  dict(auroc=0.93, auprc=0.91, f1=0.89, precision=0.91, recall=0.87, recall_1pct_fpr=0.81, ece=0.09, brier=0.10, ece_cal=0.04, brier_cal=0.09, latency_ms=18.4),
    "HG_CNN_TRANS_seed1337":  dict(auroc=0.95, auprc=0.93, f1=0.91, precision=0.92, recall=0.90, recall_1pct_fpr=0.85, ece=0.08, brier=0.08, ece_cal=0.03, brier_cal=0.07, latency_ms=22.1),
    "HG_ENSEMBLE_seed1337":   dict(auroc=0.96, auprc=0.95, f1=0.93, precision=0.94, recall=0.92, recall_1pct_fpr=0.88, ece=0.07, brier=0.07, ece_cal=0.03, brier_cal=0.06, latency_ms=19.8),
    "HG_MULTIFEAT_seed1337":  dict(auroc=0.94, auprc=0.92, f1=0.90, precision=0.91, recall=0.89, recall_1pct_fpr=0.84, ece=0.09, brier=0.09, ece_cal=0.04, brier_cal=0.08, latency_ms=20.5),
    "HG_RAV_seed1337":        dict(auroc=0.93, auprc=0.91, f1=0.89, precision=0.90, recall=0.88, recall_1pct_fpr=0.83, ece=0.10, brier=0.10, ece_cal=0.04, brier_cal=0.08, latency_ms=25.3),
}

CI_WIDTH = 0.015  # ±95 % CI half-width

def _ci(val):
    return max(0.0, val - CI_WIDTH), min(1.0, val + CI_WIDTH)


# ── public builders ───────────────────────────────────────────────────────────

def get_main_results() -> pd.DataFrame:
    rows = []
    for m, d in _BASE.items():
        lo_a, hi_a = _ci(d["auroc"])
        lo_r, hi_r = _ci(d["recall_1pct_fpr"])
        rows.append({
            "model": m,
            "type": MODEL_TYPE[m],
            "auroc": d["auroc"],
            "auroc_ci_lo": lo_a,
            "auroc_ci_hi": hi_a,
            "auprc": d["auprc"],
            "f1": d["f1"],
            "precision": d["precision"],
            "recall": d["recall"],
            "recall_1pct_fpr": d["recall_1pct_fpr"],
            "recall_1pct_fpr_ci_lo": lo_r,
            "recall_1pct_fpr_ci_hi": hi_r,
            "ece": d["ece"],
            "brier": d["brier"],
            "ece_temp_scaled": d["ece_cal"],
            "brier_temp_scaled": d["brier_cal"],
            "latency_ms_per_sample": d["latency_ms"],
        })
    return pd.DataFrame(rows)


def get_overdefense_results() -> pd.DataFrame:
    """False positive rate on benign (NotInject) prompts."""
    fpr_map = {
        "regex_heuristic":    0.28,
        "tfidf_logreg":       0.14,
        "tfidf_linearsvm":    0.13,
        "protectai_deberta":  0.07,
        "HG_CNN_TRANS_seed1337":  0.06,
        "HG_ENSEMBLE_seed1337":   0.05,
        "HG_MULTIFEAT_seed1337":  0.07,
        "HG_RAV_seed1337":        0.08,
    }
    return pd.DataFrame([
        {"model": m, "type": MODEL_TYPE[m], "overdefense_fpr": v}
        for m, v in fpr_map.items()
    ])


def get_jailbreak_results() -> pd.DataFrame:
    jb_map = {
        "regex_heuristic":    dict(auroc=0.62, f1=0.58, recall=0.54),
        "tfidf_logreg":       dict(auroc=0.72, f1=0.68, recall=0.65),
        "tfidf_linearsvm":    dict(auroc=0.74, f1=0.70, recall=0.67),
        "protectai_deberta":  dict(auroc=0.88, f1=0.84, recall=0.82),
        "HG_CNN_TRANS_seed1337":  dict(auroc=0.90, f1=0.86, recall=0.84),
        "HG_ENSEMBLE_seed1337":   dict(auroc=0.91, f1=0.88, recall=0.86),
        "HG_MULTIFEAT_seed1337":  dict(auroc=0.89, f1=0.85, recall=0.83),
        "HG_RAV_seed1337":        dict(auroc=0.88, f1=0.84, recall=0.82),
    }
    return pd.DataFrame([
        {"model": m, "type": MODEL_TYPE[m], **v}
        for m, v in jb_map.items()
    ])


def get_robustness_results() -> pd.DataFrame:
    """AUROC drop under obfuscation transforms."""
    perturbations = ["base64_encode", "unicode_swap", "leet_speak", "whitespace_inject", "synonym_sub"]
    rows = []
    for m, d in _BASE.items():
        base_auroc = d["auroc"]
        for pert in perturbations:
            drop = RNG.uniform(0.02, 0.12) if MODEL_TYPE[m] == "baseline" else RNG.uniform(0.01, 0.06)
            rows.append({
                "model": m,
                "type": MODEL_TYPE[m],
                "perturbation": pert,
                "auroc_original": base_auroc,
                "auroc_perturbed": round(max(0.5, base_auroc - drop), 4),
                "auroc_drop": round(drop, 4),
            })
    return pd.DataFrame(rows)


def get_sanitization_results() -> pd.DataFrame:
    modes = ["off", "rule_strip", "context_isolation", "llm_rewrite_optional"]
    utility  = {"off": 1.00, "rule_strip": 0.91, "context_isolation": 0.87, "llm_rewrite_optional": 0.84}
    security = {"off": 0.00, "rule_strip": 0.18, "context_isolation": 0.22, "llm_rewrite_optional": 0.31}
    rows = []
    for mode in modes:
        rows.append({
            "sanitize_mode": mode,
            "utility_semantic_similarity_mean": utility[mode],
            "asrr_proxy_mean_delta_sanitized": security[mode],
            "frac_sanitized": 0.0 if mode == "off" else RNG.uniform(0.35, 0.55),
            "utility_token_retention_mean": 1.0 if mode == "off" else RNG.uniform(0.82, 0.96),
            "latency_infer_seconds": RNG.uniform(0.02, 0.08),
        })
    return pd.DataFrame(rows)


def get_ablation_results() -> pd.DataFrame:
    ablations = [
        "full_model",
        "remove_cnn_branch",
        "remove_rules_branch",
        "remove_retrieval_branch",
        "remove_engineered_features",
        "remove_calibration",
        "remove_sanitization",
    ]
    base_auroc = 0.96  # HG_ENSEMBLE
    drops = {
        "full_model":                  0.000,
        "remove_cnn_branch":           0.028,
        "remove_rules_branch":         0.022,
        "remove_retrieval_branch":     0.035,
        "remove_engineered_features":  0.018,
        "remove_calibration":          0.004,
        "remove_sanitization":         0.008,
    }
    rows = []
    for ab in ablations:
        a = round(base_auroc - drops[ab], 4)
        rows.append({
            "ablation": ab,
            "auroc": a,
            "auroc_drop": round(drops[ab], 4),
            "f1": round(a - 0.03, 4),
            "recall_1pct_fpr": round(a - 0.08, 4),
        })
    return pd.DataFrame(rows)


def get_fairness_results() -> pd.DataFrame:
    """Worst-group metrics by language and prompt-length bin."""
    groups = ["en", "ar", "fr", "de", "short_prompt", "medium_prompt", "long_prompt"]
    rows = []
    for m in ["HG_ENSEMBLE_seed1337", "protectai_deberta", "tfidf_logreg"]:
        for g in groups:
            base = _BASE[m]["auroc"]
            gap = RNG.uniform(0.01, 0.06)
            rows.append({
                "model": m,
                "group": g,
                "auroc": round(base - gap, 4),
                "f1": round(base - gap - 0.03, 4),
                "gap_vs_overall": round(-gap, 4),
            })
    return pd.DataFrame(rows)


def get_roc_curve_data() -> dict:
    """Per-model ROC curve points (50 thresholds)."""
    thresholds = np.linspace(0, 1, 50)
    curves = {}
    for m, d in _BASE.items():
        auroc = d["auroc"]
        # Simulate a plausible ROC shape using beta distribution trick
        fpr = thresholds
        # steeper = better model
        power = 1.0 / max(0.05, (1 - auroc) * 4)
        tpr = (1 - (1 - fpr) ** power) ** (1 / max(0.1, power * 0.8))
        tpr = np.clip(tpr, 0, 1)
        curves[m] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auroc": auroc}
    return curves


def get_calibration_data() -> dict:
    """Reliability diagram data: mean predicted prob vs fraction positive per bin."""
    bins = np.linspace(0.05, 0.95, 10)
    data = {}
    for m, d in _BASE.items():
        ece_raw = d["ece"]
        ece_cal = d["ece_cal"]
        # Raw: over-confident (shifted away from diagonal)
        raw_frac = bins + RNG.uniform(-ece_raw, ece_raw, size=len(bins))
        raw_frac = np.clip(raw_frac, 0, 1)
        # Calibrated: closer to diagonal
        cal_frac = bins + RNG.uniform(-ece_cal, ece_cal, size=len(bins))
        cal_frac = np.clip(cal_frac, 0, 1)
        data[m] = {
            "bins": bins.tolist(),
            "raw_frac_positive": raw_frac.tolist(),
            "cal_frac_positive": cal_frac.tolist(),
            "ece_raw": ece_raw,
            "ece_cal": ece_cal,
        }
    return data
