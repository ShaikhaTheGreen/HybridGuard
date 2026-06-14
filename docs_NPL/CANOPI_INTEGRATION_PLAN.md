# CANOPI Integration Plan (NPL extension)

Branch: `canopi-npl` (off `main` @ 085d5a0). Commit per stage, only CANOPI files;
the existing NPL working-tree WIP (`code/`, `results_NPL/`, `*_NPL` dirs) is left
uncommitted and untouched. Public APIs of `hybridguard.canonicalize` / `cli` are frozen.

## Central claim (falsifiable)
A projection head trained with a JOINT objective (multi-view intent-invariance + partial-AUC
low-FPR term) on top of deterministic L0 canonicalization yields a representation whose
strict-FPR recall (a) matches/beats baselines in-domain, (b) recovers cross-lingual recall,
(c) holds its operating threshold across corpora — with NO LLM at inference.
**Primary metric:** Recall@1%FPR (τ frozen on val before any test read).
**Headline:** AR/ES Recall@1%FPR at the SAME τ.
**Falsification:** if CANOPI does not jointly beat B6 (invariance-only) and B7 (pAUC-only), the claim FAILS — report honestly.

## Audited conventions (reuse verbatim)
- **L0:** `from hybridguard import canonicalize` → `.canonical/.lowercase/.trace/.decoded_payloads`. REUSE.
- **Data/eval protocol** (orchestrator notebook): xTRam1 primary; SHA-256 dedup + SimHash + leakage check;
  deterministic 60/20/20 stratified split seed **1337**; 5 seeds **{42,2025,7,1337,314}**; metrics
  `threshold_at_fpr(y,p,0.01)` (τ on val, frozen on test) → `recall_at`, AUROC, AUPRC, F1@op, bootstrap CI,
  McNemar, ECE; over-defense FPR sweep; multilingual fairness. REUSE.
- **Persistence:** `runs/<run_id>/seed_<s>/eval_*.csv` + `aggregated/*_mean_std.csv` + `tables/*.tex` + `run_metadata.json`.
  Standardize CANOPI here; additively extend `dashboard/utils/load_results.py` search to include `runs/` (non-breaking).
- **Figures:** `scripts/make_canopi_<topic>_figure.py` reads `paper/paper_v2_extract/<topic>/*.csv` → writes
  `paper/figures/<name>.{png,pdf}`; register in `scripts/make_paper_figures.py:FIGURES` and
  `dashboard/figures/generate_all_figures.py`. Reuse existing ROC/calibration/ablation/robustness/sanitization generators.
- **Existing attack bank:** `code/canonicalize.py:perturb(text,attack,sigma)` / `ATTACKS` (homoglyph/zero_width/leet/spacing)
  and `code/npl_adaptive_experiment.py:attack(text,kind,...)` (confusable_known/unknown, emoji_smuggle, paraphrase) — reuse as view/attack sources.

## REUSE / EXTEND / ADD map

### REUSE (unchanged)
- [ ] R1 L0 canonicalizer `hybridguard.canonicalize`.
- [ ] R2 Orchestrator data cells: dataset load, dedup/SimHash/leakage, 60/20/20@1337, 5 seeds, leakage_report.
- [ ] R3 Orchestrator metric helpers: `threshold_at_fpr`, `recall_at`, bootstrap CI, McNemar, ECE, fairness, over-defense.
- [ ] R4 WS1_Universal_Eval 4-detector harness (regex/TF-IDF-SVM/InjecGuard/DeBERTa) → E8 base.
- [ ] R5 CrossCorpus_Eval threshold-transfer harness (deepset, JBB) → E4 base.
- [ ] R6 LLM_Judge_Baseline → E9 cost point.
- [ ] R7 Existing HG variants become baselines B1 (TF-IDF+SVM), B2 (DeBERTa-v3), B3 (InjecGuard), B8 (HG L0-only).

### EXTEND
- [ ] X1 `src/hybridguard/canopi/` installed package (see ADD).
- [ ] X2 `notebooks/HybridGuard_CANOPI_Orchestrator.ipynb` — imports canopi src; re-runs (not duplicates) orchestrator data/eval cells via `%run`/exec; adds CANOPI train+eval loop over 5 seeds.
- [ ] X3 Eval extensions: cross-lingual R@1%FPR (E3), threshold-transfer retention (E4), invariance-drift δ (E6), intent-preservation gap (E6) — persisted in `runs/<run_id>/`.
- [ ] X4 Extend WS1_Universal_Eval to test trained CANOPI representation as a front-end (E8).
- [ ] X5 `dashboard/utils/load_results.py` search list += `runs/` (additive).
- [ ] X6 `pyproject.toml [full]` += pinned deps (libauc/pAUC surrogate is hand-rolled to avoid new dep; NLLB via existing transformers); add lockfile.

### ADD — code
- [ ] A1 `canopi/encoders.py` — frozen encoder E (sentence-transformers / HF), `encode(texts)->np.ndarray`, deterministic, cached.
- [ ] A2 `canopi/model.py` — `CanopiModel`: c → frozen E → projection head P (MLP, unit-norm) → detector head; `score(texts)->[0,1]`.
- [ ] A3 `canopi/losses.py` — `L_inv` (multi-view SupCon incl AR/ES), `L_hardneg`, `L_drift`, `L_pAUC` (DeepTopPush-style top-push surrogate, FPR∈[0,β=0.01]); each config-toggleable via λ.
- [ ] A4 `canopi/augment.py` — transformation bank (paraphrase, BT EN↔AR/EN↔ES via NLLB-200, persona templating, encoding wraps) + intent-preservation filter (NLI/embedding-sim), logs accept rates; emits aligned AR/ES pairs artifact.
- [ ] A5 `canopi/train.py` — training loop, λ config, per-seed checkpoint, τ selection on val (frozen), writes `runs/<run_id>/`.
- [ ] A6 `canopi/attacks.py` — Tier-4 adaptive semantic-rewrite attacker (E7) maximizing embedding distance under intent-preservation constraint.
- [ ] A7 `canopi/runs.py` — results IO matching `runs/<run_id>/` (per-seed CSV, aggregated mean±std, LaTeX table renderer, run_metadata.json).
- [ ] A8 Baselines B4 PromptGuard 2 (document exclusion if license/endpoint blocks), B5 plain embedding clf on E(c(x)) [no-invariance floor], B6 invariance-only (λ4=0), B7 pAUC-only (λ1=λ2=λ3=0) — as canopi configs.

### ADD — configs (one YAML per experiment/ablation under `configs/`)
- [ ] C1 `configs/canopi_main.yaml` (full joint), `b5_embedding.yaml`, `b6_invariance_only.yaml`, `b7_pauc_only.yaml`.
- [ ] C2 ablation YAMLs: each loss off, each aug family off, frozen/finetune, head depth {1,2,3}, mono/multilingual encoder.

### ADD — figures (`scripts/make_canopi_*_figure.py`, PDF+PNG, grayscale-legible)
- [ ] F-new: F2,F3,F4,F5,F6,F8,F9,F10,F11,F13,F14,F15,F16,F18,F19 + reuse F1(schematic), F7(ROC), F12(UMAP), F17(calibration). Each reads `runs/`/`paper_v2_extract/` via `load_results.py`; hook into `make_paper_figures.py` + `generate_all_figures.py`.

### ADD — tests
- [ ] T1 `tests/test_canopi_losses.py` (each loss term: gradient sign, toggling, pAUC focuses low-FPR).
- [ ] T2 `tests/test_canopi_augment.py` (view families, intent filter accept/reject, determinism).
- [ ] T3 `tests/test_canopi_pipeline.py` (canonicalize→encode→project unit-norm shape/determinism, CPU smoke).

## Experiment series → figure map
E1 main in-domain (F7,F8) · E2 ablations (F18,F19) · E3 multilingual HEADLINE (F9) · E4 threshold transfer (F10) ·
E5 over-defense (F11) · E6 invariance diagnostics (F12,F13) · E7 adaptive attacker (F14) · E8 detector-agnostic (F15) ·
E9 latency + LLM-judge (F16) · E10 calibration (F17). Setup/method: F1,F3,F4,F5,F6. Motivation: F2.

## Statistical protocol (non-negotiable)
5 seeds, mean±std + 95% CI; bootstrap ≥500 resamples fixed seed; McNemar vs strongest neural baseline AND vs B5 (p<0.01);
every ablation Δ with CI (matters only if CI excludes 0); τ frozen before any test read; no per-corpus re-thresholding.

## Execution reality
Heavy training/eval (frozen encoder, NLLB-200 BT, 5 seeds) runs on Colab GPU per the orchestrator pattern.
In this environment: build all installable code + configs + tests + notebook + figure generators, and validate the
`canonicalize→encode→project→loss` path with CPU smoke tests on tiny synthetic data. Real numbers are produced by the
GPU run; no metric is fabricated. `RESULTS_SUMMARY.md` is scaffolded with the reproduction command per E# and filled from `runs/`.

## Staging (commit per stage)
1. Plan + branch (this file).  2. `canopi/` package: encoders, model, losses, augment, attacks, runs (+CPU smoke tests).
3. configs YAMLs + baselines.  4. CANOPI orchestrator notebook (reuses orchestrator cells).  5. Eval extensions (E3/E4/E6 metrics) + WS1 E8 extension.  6. Figure generators + hooks.  7. RESULTS_SUMMARY.md scaffold + reproduction checklist + deps/lockfile.
