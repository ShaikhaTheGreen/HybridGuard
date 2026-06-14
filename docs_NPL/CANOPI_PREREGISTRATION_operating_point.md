# Pre-registration — operating-point-robustness headline (locked 2026-06-14)

Registered **before** the experiment B run (baselines through E4/E5/E7) and the A
mechanism run, so the alternative headline cannot be reverse-engineered from the
result. Git-timestamped on branch `canopi-npl`. Supersedes nothing; complements the
original pre-registration (cross-lingual Recall@1%FPR), which is **demoted to a
supporting result** if and only if the primary hypothesis below confirms.

## Context
The smoke run (see `CANOPI_SMOKE_FINDINGS.md`) showed CANOPI ≈ B5 on raw cross-lingual
recall — the frozen multilingual encoder drives recovery, not the joint objective. The
partial-AUC term is, by construction, an *operating-point* loss, so its value (if any)
should appear on operating-point tasks, not on average recall. We test that directly.

## Primary hypothesis (H1)
The joint objective yields a more robust **operating point** than the no-invariance
floor B5 and the single-term ablations B6 (invariance-only) and B7 (pAUC-only), on
three axes, with τ frozen on xTRam1 validation at 1% FPR and never re-tuned:

- **H1a Threshold transfer (E4):** lower FPR drift |realized_FPR − 0.01| AND higher
  recall@τ on out-of-corpus deepset (and JBB recall@τ) than B5.
- **H1b Over-defense (E5):** lower NotInject FPR than B5 at matched in-domain FPR
  targets {1,2,5,10}%.
- **H1c Adaptive robustness (E7):** higher recall under semantic-rewrite attack at
  fixed τ (area under the residual curve, strengths {0,.25,.5,.75,1}) than B5.

## Metrics (pre-specified, 5 seeds)
Primary: **threshold-transfer retention** = recall@τ on deepset at the xTRam1-frozen τ.
Secondary: FPR drift (H1a), NotInject FPR @1% target (H1b), residual-curve AUC (H1c).
All with stratified bootstrap 95% CIs (≥500 resamples, seed 1337) and mean±std over
seeds {42,2025,7,1337,314}.

## Decision rule (confirm / refute — fixed in advance)
- **CONFIRM H1** (→ operating-point robustness becomes the PRIMARY headline; cross-lingual
  recovery vs surface/English detectors becomes supporting) iff CANOPI beats **B5** on
  **≥2 of {H1a, H1b, H1c}** with **non-overlapping 95% CIs**, AND does not lose to B5 on
  the third, AND also beats **B6 and B7** on the primary metric (else the *joint* objective
  is not the source).
- **REFUTE H1** (→ fall back to the simple-method paper, Position 4: report the joint
  objective as an honest negative ablation against B5) otherwise.

## Guardrails (unchanged)
τ frozen before any test read; no per-corpus re-thresholding; report ALL axes including
losses; the E7 residual and any failed language (e.g. Russian) are reported, not hidden.
B2 DeBERTa's cross-lingual numbers require multilingual NEGATIVES before they count
(smoke run showed it over-triggers at τ≈0.0008).

## Status
PENDING run B (`CONFIG=configs/canopi_main.yaml`, SMOKE=False) + B6/B7 config runs, and
run A (`configs/ablations/monolingual.yaml`). Verdict to be filled from
`runs/<run_id>/aggregated/` — no number entered here until the run produces it.
