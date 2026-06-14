# CANOPI — Results Summary (NPL extension)

**Status:** code + configs + protocol complete and unit-tested on CPU; headline
numbers are produced by the Colab GPU run (frozen encoder + NLLB-200 + 5 seeds).
Cells below are filled from `runs/<run_id>/aggregated/` after that run. **No
number is written here until its run produces it** — `TBD(run)` marks a pending
GPU result, never a guess.

Branch: `canopi-npl`. Reproduce any row from a single command (see each E#).

## Central claim & verdict
> A projection head trained with a JOINT objective (multi-view intent-invariance +
> partial-AUC low-FPR term) on frozen L0 canonicalization yields a representation
> whose strict-FPR recall (a) matches/beats baselines in-domain, (b) recovers
> cross-lingual recall, (c) holds its operating threshold across corpora — no LLM
> at inference.

**Primary metric:** Recall@1%FPR, τ frozen on validation before any test read.
**Falsification rule:** if CANOPI does not beat **B6 (invariance-only)** AND
**B7 (pAUC-only)** jointly, the claim FAILS — reported honestly either way.

**Overall verdict:** `TBD(run)`

## Pre-registration (frozen before test)
- Seeds `{42, 2025, 7, 1337, 314}`; split 60/20/20 stratified @ seed 1337 (reused).
- τ at 1% FPR selected on validation only; no per-corpus re-thresholding.
- Bootstrap CIs ≥500 resamples (we use 1000), fixed seed 1337; McNemar vs strongest
  neural baseline AND vs B5, p<0.01; every ablation Δ reported with CI (matters iff CI excludes 0).

## Baselines
B1 TF-IDF+LinearSVM · B2 DeBERTa-v3 PI-v2 · B3 InjecGuard · B4 PromptGuard 2
(`TBD(run)`; document exclusion if license/endpoint blocks) · B5 plain embedding
clf on E(c(x)) [no-invariance floor] · B6 invariance-only (λ4=0) · B7 pAUC-only
(λ1=λ2=λ3=0) · B8 HybridGuard L0-only.

## Experiments — headline · CI · verdict · reproduce
| E# | What | Headline metric | Result (mean±std [95% CI]) | Verdict | Reproduce |
|----|------|-----------------|----------------------------|---------|-----------|
| E1 | Main in-domain (xTRam1) | R@1%FPR; CANOPI vs B1–B8 | `TBD(run)` | `TBD` | `papermill notebooks/HybridGuard_CANOPI_Orchestrator.ipynb -p CONFIG configs/canopi_main.yaml` |
| E2 | Ablations (loss/aug/encoder/depth) | ΔR@1%FPR w/ CI | `TBD(run)` | `TBD` | `for c in configs/ablations/*.yaml; do run $c; done` |
| **E3** | **Multilingual AR/ES @ same τ (HEADLINE)** | **R@1%FPR AR, ES** | `TBD(run)` | `TBD` | `run configs/canopi_main.yaml --eval crosslingual` |
| E4 | Threshold transfer (deepset, JBB) | FPR drift + R@τ | `TBD(run)` | `TBD` | CrossCorpus_Eval w/ CANOPI front-end |
| E5 | Over-defense (NotInject) | FPR at FPR-target sweep | `TBD(run)` | `TBD` | `run --eval overdefense` |
| E6 | Invariance diagnostics | δ per family, intent gap | `TBD(run)` | `TBD` | `run --eval invariance` |
| E7 | Adaptive attacker (residual) | R@1%FPR vs rewrite strength | `TBD(run)` | `TBD` | `run --eval adaptive` |
| E8 | Detector-agnostic transfer | recall recovery / detector | `TBD(run)` | `TBD` | WS1_Universal_Eval w/ CANOPI front-end |
| E9 | Latency + LLM-judge cost | ms/sample, $/1k | `TBD(run)` | `TBD` | `run --eval latency`; LLM_Judge_Baseline |
| E10 | Calibration | ECE, Brier pre/post temp | `TBD(run)` | `TBD` | `run --eval calibration` |

## Minimum-viable-breakthrough check (E3)
AR/ES R@1%FPR rising from ~0.000 to clearly non-zero with non-overlapping CIs,
while NotInject over-defense FPR stays ~0: `TBD(run)`.

## Figure manifest (F1–F19) — generated via `python scripts/make_paper_figures.py`
Each reads `runs/<run_id>/` (or `paper/paper_v2_extract/<topic>/`) and emits PDF+PNG,
grayscale-legible. MAIN: F1–F11, F13–F15, F18. SUPP: F12, F16, F17, F19.

| F# | Section | Generator | Exists |
|----|---------|-----------|--------|
| F1 | Intro | canonicalization-hierarchy schematic | `TBD` |
| F2 | Intro | baseline collapse (obfusc.+multiling.) | `TBD` |
| F3 | Method | CANOPI architecture block diagram | `TBD` |
| F4 | Method | joint-objective schematic | `TBD` |
| F5 | Setup | study map | reuse `make_study_map_figure.py` |
| F6 | Setup | data/augmentation pipeline + accept rates | `TBD` |
| F7 | E1 | ROC + low-FPR inset | reuse `fig_roc_curves.py` |
| F8 | E1 | R@1%FPR bars + CIs (B1–B8+CANOPI) | `TBD` |
| **F9** | **E3** | **AR/ES R@1%FPR (HEADLINE, ~0→nonzero)** | `TBD` |
| F10 | E4 | threshold transfer FPR @ fixed τ | `TBD` |
| F11 | E5 | over-defense FPR vs operating point | `TBD` |
| F12 | E6 | UMAP/t-SNE before/after (intent, lang) | `TBD` (supp) |
| F13 | E6 | invariance-drift δ per family + intent gap | `TBD` |
| F14 | E7 | adaptive residual curve | `TBD` |
| F15 | E8 | detector-agnostic recall recovery | `TBD` |
| F16 | E9 | latency–recall Pareto + LLM cost | `TBD` (supp) |
| F17 | E10 | calibration reliability pre/post | reuse `fig_calibration.py` |
| F18 | E2 | ΔR@1%FPR per removed term/aug | reuse/extend `fig_ablation.py` |
| F19 | E2 | head depth / freeze / mono-vs-multi | `TBD` (supp) |

## Reproducibility
- Versions pinned in `pyproject.toml [full]`; lockfile: `TBD(run)` (`pip freeze` from the Colab run).
- Persisted per run: split indices, per-seed checkpoints, per-seed CSVs, aggregated
  mean±std, all configs, all figure CSVs, `run_metadata.json`, leakage_report (empty).
- Safe-excerpt only; malicious prompts never logged verbatim (`canopi.attacks.safe_excerpt`).

## Final checklist (printed when complete)
- [ ] E1 … E10 each ran and wrote `runs/<run_id>/aggregated/`
- [ ] F1 … F19 each exist as PDF+PNG under `paper/figures/`
- [ ] McNemar vs strongest-neural and vs B5 (p<0.01) recorded
- [ ] τ frozen-on-val confirmed in `run_metadata.json`
- [ ] Verdict (claim supported / FAILED) stated above with CIs
