# Run CANOPI on Colab Pro+ (one-command path)

Everything is on GitHub branch **`canopi-npl`** (repo `ShaikhaTheGreen/HybridGuard`).

## 1. Open the notebook
Open this in Colab (Pro+, A100/L4):

    https://colab.research.google.com/github/ShaikhaTheGreen/HybridGuard/blob/canopi-npl/notebooks/HybridGuard_CANOPI_Orchestrator.ipynb

(Colab → File → Open notebook → GitHub → `ShaikhaTheGreen/HybridGuard`, branch
`canopi-npl`, pick `HybridGuard_CANOPI_Orchestrator.ipynb`.)

## 2. Set the runtime + run
- Runtime → Change runtime type → **GPU**.
- Cell 0 config: leave defaults (`CONFIG=configs/canopi_main.yaml`, `MOUNT_DRIVE=True`).
  For a 3-minute dry run first, set `SMOKE = True`, Run all, confirm it completes,
  then set `SMOKE = False` and Run all again.
- **Run all.** The notebook clones the branch, `pip install -e .[full]`, builds
  splits (leakage-checked), trains CANOPI over 5 seeds, runs E1/E3–E7/E10, writes
  `runs/<run_id>/`, runs McNemar, exports figure CSVs, regenerates figures, prints
  the checklist.

## 3. What you get (persisted to Drive if mounted)
    runs/run_canopi_main_<date>/
      run_metadata.json            config, seeds, git sha, val-frozen-tau protocol
      seed_<s>/*.csv               per-seed E1/E3–E7/E10 frames
      aggregated/*_mean_std.csv    mean±std across 5 seeds
      tables/*.tex, *.csv          booktabs LaTeX + dashboard mirror
      leakage_report.json          must be clean
      aligned_pairs_seed*.csv      AR/ES intent-aligned pairs (release artifact)
    paper/figures/canopi_*.{pdf,png}   F1–F4,F6,F8–F11,F13–F15,F18,F19

## 4. Run the baselines / ablations (for E2 and B6/B7)
Re-run the notebook with a different `CONFIG`:
- `configs/baselines/b6_invariance_only.yaml`  (must lose to CANOPI on E3 or claim FAILS)
- `configs/baselines/b7_pauc_only.yaml`         (must lose to CANOPI or claim FAILS)
- `configs/ablations/*.yaml`                     (E2 ΔR@1%FPR; regenerate via
  `python configs/ablations/generate_ablations.py`)

## 5. Wire the cross-lingual + L0 baselines
- Upload `code/multilingual_injections.py` to the repo `code/` dir (the notebook
  auto-imports `get_ml_sets` for E3). Without it, E3 is skipped with a clear message.
- B8 (HybridGuard L0-only) and B2/B3 (DeBERTa/InjecGuard) load automatically if the
  HF models / a prior orchestrator run are reachable; otherwise documented as excluded.

## 6. Pre-registration reminders (do not violate)
- τ is frozen on validation at 1% FPR before any test read; no per-corpus re-thresholding.
- Seeds `{42,2025,7,1337,314}` fixed; bootstrap 1000 resamples seed 1337.
- Report ALL negatives (E7 residual, any failed language). CANOPI must beat **B6 and
  B7 jointly** or the central claim FAILS — state the verdict either way in RESULTS_SUMMARY.md.

## 7. After the run
Fill `RESULTS_SUMMARY.md` from `runs/<run_id>/aggregated/`, then
`bash scripts/pack_overleaf.sh` to package the manuscript with the new tables/figures.
