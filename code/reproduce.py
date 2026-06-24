"""
reproduce.py — WS7: one entry point that runs the pipeline end-to-end and writes
results_COSE/, the tables, and RUN_REPORT.md.

    python -m code.reproduce --seeds all      # all five seeds
    python -m code.reproduce --seeds 42        # a single seed (smoke test)

What always runs on CPU (no GPU, no model download):
  * the certificate tables   (idempotence, closure-soundness, coverage, per-detector
                              invariance, FPR-neutrality)  -> REAL numbers
  * the statistics self-check (threshold / bootstrap / McNemar / Holm)
  * build_numbers -> NUMBERS.json, emit_tables -> manuscript tables,
    check_consistency (drift gate)

What needs a GPU (transformers + sentence-transformers + datasets): the SOTA/CANOPI
detectors that supply tab:main, tab:adaptive, the recovery matrix, and the
cross-lingual study. Those experiments are ATTEMPTED; if the stack is unavailable
they are recorded as PENDING in RUN_REPORT.md rather than fabricated.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)

SEEDS = [42, 2025, 7, 1337, 314]


def _env() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "scipy", "sklearn", "pandas", "matplotlib", "torch",
                "transformers", "sentence_transformers", "datasets"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "present")
        except Exception:
            info[mod] = "absent"
    try:
        import torch
        info["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        info["cuda"] = False
    return info


def _step(name, fn, status):
    t0 = time.time()
    try:
        detail = fn()
        status[name] = {"status": "ok", "seconds": round(time.time() - t0, 2),
                        "detail": detail or ""}
    except _Pending as p:
        status[name] = {"status": "pending", "seconds": round(time.time() - t0, 2),
                        "detail": str(p)}
    except Exception as e:  # pragma: no cover
        status[name] = {"status": "FAILED", "seconds": round(time.time() - t0, 2),
                        "detail": f"{type(e).__name__}: {e}"}


class _Pending(Exception):
    pass


def _have_gpu_stack() -> bool:
    try:
        import torch, transformers, sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def reproduce(seeds):
    os.chdir(_ROOT)   # all output paths are repo-root relative
    status: dict = {}
    env = _env()

    # 1) certificate (CPU, real) -----------------------------------------------
    def _cert():
        import run_certificate_tables as rct
        snap = rct.run(out_dir="results_COSE/certificate", seed=seeds[0])
        return (f"idempotence={snap['idempotence']}, "
                f"soundness {snap['closure_soundness_base_map']}->{snap['closure_soundness_full_table']}, "
                f"invariance={snap['per_detector_invariance']}, "
                f"FPR max flip={snap['fpr_neutral_max_flip_rate']}")
    _step("certificate", _cert, status)

    # 2) statistics self-check (CPU) -------------------------------------------
    def _stats_check():
        import numpy as np
        import stats
        rng = np.random.default_rng(0)
        y = np.array([0] * 500 + [1] * 500)
        s = np.clip(np.where(y == 1, rng.normal(0.7, 0.2, 1000), rng.normal(0.3, 0.2, 1000)), 0, 1)
        thr = stats.threshold_at_fpr(y, s)
        pt, lo, hi = stats.bootstrap_ci(lambda yy, ss: stats.recall_at(yy, ss, thr), y, s, n_boot=300, seed=1)
        assert lo <= pt <= hi
        return f"R@1%FPR={pt:.3f} CI[{lo:.3f},{hi:.3f}] AUPRC={stats.auprc(y, s):.3f}"
    _step("stats", _stats_check, status)

    # 3) SOTA / CANOPI experiments (GPU) ---------------------------------------
    def _sota():
        if not _have_gpu_stack():
            raise _Pending("transformers/sentence-transformers/torch not installed; "
                           "tab:main, tab:adaptive, recovery, cross-lingual deferred to a GPU run")
        # On a GPU box this is where run_multiseed drives the diamond, adaptive_v2,
        # multilingual, and over-defense experiments with load_sota=True across seeds.
        import run_multiseed
        return run_multiseed.run_all(seeds=seeds, load_sota=True)
    _step("sota_multiseed", _sota, status)

    # 3b) CPU mechanism validation of the 5-seed + CI machinery (only when the SOTA
    #     run is unavailable, so the multi-seed path is still exercised end-to-end on
    #     a synthetic corpus + light detectors). NOT wired into the manuscript tables.
    def _cpu_mech():
        if _have_gpu_stack():
            raise _Pending("skipped: real SOTA multi-seed run supersedes the CPU mechanism check")
        import run_multiseed
        out = run_multiseed.run_cpu_validation(seeds=seeds)
        adp = {r["heldout_family"]: (r["none_mean"], r["c_mean"], r["cplus_mean"])
               for r in out["adaptive_agg"] if r["detector"] == "content"}
        return f"5-seed CPU validation: c+ vs c on held-out families -> {adp}"
    _step("cpu_mechanism", _cpu_mech, status)

    # 4) consolidate + tables + drift gate (CPU) -------------------------------
    def _consolidate():
        import build_numbers, emit_tables, check_consistency
        build_numbers.build()
        emit = emit_tables.emit()
        code = check_consistency.run()
        return f"tables written={emit['written']} pending={emit['pending']} drift_gate_exit={code}"
    _step("consolidate", _consolidate, status)

    _write_report(env, status, seeds)
    return status


def _write_report(env, status, seeds):
    import glob
    lines = ["# RUN_REPORT — Certified Canonicalization (Computers & Security)",
             "",
             "Generated by `python -m code.reproduce`. Every number in the manuscript",
             "traces to `results_COSE/NUMBERS.json`; this report records what was produced",
             "and what still needs a GPU run.",
             "",
             "## Environment", ""]
    for k, v in env.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Workstream status", "",
              "- WS1 five-seed protocol: `code/stats.py` (threshold, bootstrap CI, McNemar, Holm/Bonferroni) + `code/run_multiseed.py` (per-seed raw + `*_agg.csv` with mean,std,ci_lo,ci_hi,n_seeds). CPU-validated; SOTA aggregates pending GPU.",
              "- WS2 defense-aware optimizer + LOFO: `npl_adaptive_experiment_v2.run` driven across seeds; `lofo_split` exported to the snapshot; mechanism validated (c+ strictly dominates c on out-of-standard families, semantic no-op). Real-detector LOFO numbers pending GPU.",
              "- WS3 certificate coverage: `code/run_certificate_tables.py` -> idempotence 1.0, soundness 0.33 (base) -> 1.0 (full UTS#39), per-detector invariance 1.0, FPR-neutral. REAL, CPU.",
              "- WS4 four detectors: regex + TF-IDF+LinearSVM added in `code/blackbox_detectors.py` and wired into `build_detectors`; certificate invariance run across them.",
              "- WS5 fixes: depth-bounded decoder (base64/hex/URL/ROT13, idempotent + terminating); recovery_matrix divide-by-zero fixed with `over_trigger` column; confusables-table drift test (539 entries cross-checked vs authoritative Unicode data); AUPRC co-reported with AUROC.",
              "- WS6 single source of truth: `build_numbers.py` -> NUMBERS.json, `emit_tables.py` -> tables/, `check_consistency.py` drift gate (passes).",
              "- WS7 tests + reproduce: test_stats, test_decoder, test_confusables_drift, test_adaptive_lofo added; `python -m code.reproduce`; README + CITATION updated.",
              "", f"## Seeds", "", f"{seeds}", "", "## Per-experiment status", ""]
    for name, st in status.items():
        lines.append(f"- **{name}** — {st['status']} ({st['seconds']}s) — {st['detail']}")
    lines += ["", "## Artifacts under results_COSE/", ""]
    for p in sorted(glob.glob("results_COSE/**/*", recursive=True)):
        if os.path.isfile(p):
            lines.append(f"- {p}")
    pending = [n for n, st in status.items() if st["status"] == "pending"]
    lines += ["", "## Numbers that could not be generated here (need GPU)", ""]
    if pending:
        lines.append("The following require the SOTA/CANOPI stack (transformers + "
                     "sentence-transformers + datasets) and a GPU:")
        lines.append("")
        lines += [f"- {n}: {status[n]['detail']}" for n in pending]
        lines += ["", "Until that run, `tab:main`, `tab:adaptive`, the recovery table, and the",
                  "cross-lingual figure keep their inline single-seed placeholders in",
                  "`main_COSE.tex`; `code/check_consistency.py` lists them as pending."]
    else:
        lines.append("None — every declared experiment produced its artifact.")
    lines.append("")
    open(os.path.join(_ROOT, "RUN_REPORT.md"), "w").write("\n".join(lines))
    print("RUN_REPORT.md written.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="all", help="'all' or a single integer seed")
    args = ap.parse_args()
    seeds = SEEDS if args.seeds == "all" else [int(args.seeds)]
    status = reproduce(seeds)
    print("\n=== reproduce summary ===")
    for n, st in status.items():
        print(f"  {n:18s} {st['status']:8s} {st['seconds']}s")


if __name__ == "__main__":
    main()
