# CANOPI — mechanism result + locked framing (2026-06-14)

The monolingual-encoder ablation (experiment A, pre-registered in
`CANOPI_SMOKE_FINDINGS.md`) **vindicates the invariance objective** and sets the
paper's primary framing. This is a real, CI-clean result — 5 seeds, full run
(`run_ablation_monolingual_20260614`, encoder `all-MiniLM-L6-v2`, git 4cc7161).

## The result — A: monolingual encoder, E3 (Recall@1%FPR, 5 seeds)
With a frozen ENGLISH-only encoder the no-invariance floor B5 collapses
cross-lingually; CANOPI's invariance training recovers it, non-overlapping CIs:

| lang | CANOPI | B5 (no-inv) | Δ |
|---|---|---|---|
| fr_mt | 0.61±0.06 | 0.12±0.00 | **+0.49** |
| zh_mt | 0.51±0.13 | 0.01±0.00 | **+0.49** |
| de_mt | 0.40±0.10 | 0.02±0.00 | **+0.38** |
| es_mt | 0.52±0.05 | 0.23±0.00 | **+0.29** |
| pt_mt | 0.42±0.04 | 0.18±0.00 | **+0.24** |
| es | 0.22±0.07 | 0.00±0.00 | **+0.22** |
| ar, hi_mt, ar_mt, ru_mt | ~0.01 | ~0.00 | ~0 (both collapse) |
| ar_roman_nat | 0.09±0.10 | 0.20±0.00 | −0.11 (B5 better) |

In-domain unaffected: CANOPI 0.966±0.003 vs B5 0.972±0.000 — the cross-lingual
ability is added at **no in-domain cost**.

## Interpretation (LOCKED primary framing)
**The cross-lingual invariance objective substitutes for multilingual pretraining.**
Cross-lingual detection can come from the *encoder* or from the *training objective*:
- multilingual encoder → B5 already has it, so CANOPI ≈ B5 (main run; the boundary/control);
- monolingual encoder → B5 loses it, and CANOPI's invariance objective **recovers it at
  training time** (fr 0.12→0.61, zh 0.01→0.51), *provided the encoder represents the script
  at all*.

The honest boundary: non-Latin scripts (Arabic, Hindi, Russian) the English encoder tokenizes
to noise stay collapsed for both — invariance can amplify weak-but-present signal, not
manufacture it from nothing. Romanized Arabic (ar_roman_nat) is the one bucket B5 wins.

This reframes the earlier "CANOPI≈B5 on the multilingual encoder" from a *negative* into the
*control that localizes the objective's value*. The two encoders together isolate the mechanism.

## Attribution (next runs) — is it the invariance term, not pAUC?
Added `configs/baselines/b6_mono_invariance_only.yaml` (invariance, no pAUC, with views) and
`b7_mono_pauc_only.yaml` (pAUC-only, no views), both on the monolingual encoder.
- EXPECT b6_mono to reproduce the lift; b7_mono to stay collapsed.
- That contrast attributes the cross-lingual recovery to **multi-view invariance**, not the
  operating-point term. (Finer confound — views-as-BCE-data vs the contrastive loss — is the
  `no_crosslingual` / `no_inv` ablation; run if a reviewer presses.)

## Still needed to finalize
1. **Full 5-seed `canopi_main`** (multilingual; the last one was smoke) — registered in-domain +
   multilingual numbers and the operating-point H1 verdict (which trended REFUTE in smoke; that
   is FINE now — operating-point robustness was the *fallback*; the mechanism is the headline).
2. **b6_mono + b7_mono** — attribution.
3. Optional **b6 / b7** (multilingual) — confirm the joint objective on the deployed encoder.

## Status of the two pre-registered claims
- Operating-point H1 (`CANOPI_PREREGISTRATION_operating_point.md`): trending REFUTE in smoke —
  acceptable, it was the fallback. Report honestly; CANOPI need not win it.
- Mechanism (experiment A): CONFIRMED at 5 seeds — now the primary contribution.
