# CANOPI — interim findings from the first run (2026-06-14)

**STATUS: SMOKE run, not publishable.** `run_metadata.json`: `seeds=[42]`, split
480/160/159 (SMOKE subsample), 30 epochs. Wide CIs. Recorded here for honesty;
the full 5-seed numbers supersede these. One pattern below is *structural* and
will likely persist at scale.

## Headline verdict (interim): central claim NOT supported as framed
The pre-registered test — AR/ES Recall@1%FPR non-zero with CIs **not overlapping the
no-invariance floor B5** — is **not met**, because **B5 is not near-zero**: it shares
the frozen multilingual encoder and already recovers cross-lingual recall. CANOPI ≈ B5
everywhere, CIs fully overlapping. The encoder, not the invariance objective, drives
recovery. This is the honest negative the protocol required us to surface.

### E3 cross-lingual — Recall@1%FPR at each detector's own frozen τ (seed 42)
| lang | CANOPI | B5 (no-inv floor) | B1 TF-IDF | B3 InjecGuard | B2 DeBERTa* |
|---|---|---|---|---|---|
| ar | 0.62 [.53,.69] | 0.62 [.55,.69] | 0.03 | 0.0 | 1.0* |
| es | 0.70 [.50,.90] | 0.85 [.70,1.0] | 0.15 | 0.05 | 1.0* |
| ar_mt | 0.84 | 0.82 | 0.04 | 0.0 | 1.0* |
| fr_mt | 0.90 | 0.86 | 0.26 | 0.0 | 0.98* |
| de_mt | 0.92 | 0.84 | 0.02 | 0.02 | 0.90* |
| zh_mt | 0.90 | 0.92 | 0.04 | 0.02 | 0.86* |
| hi_mt | 0.92 | 0.88 | 0.0 | 0.0 | 1.0* |
| ru_mt | 0.28 | 0.28 | 0.12 | 0.0 | 1.0* |
| pt_mt | 0.92 | 0.88 | 0.04 | 0.04 | 0.98* |
| ar_roman_nat | 0.20 | 0.13 | 0.40 | 0.13 | 0.93* |

- ✅ CANOPI crushes **surface/English** detectors (B1 0.03, B3 0.0 → 0.62–0.92).
- ❌ CANOPI ≈ **B5** (shared multilingual encoder); invariance objective not shown as the source.
- `*` **B2 DeBERTa over-triggers** (τ≈0.0008 → flags everything as injection); its "1.0" is an
  artifact, not discrimination. Needs multilingual NEGATIVES to expose its FPR.

## Other honest negatives (seed 42)
- **E1 in-domain** R@1%FPR: CANOPI 0.92 [.84,.98], B1 0.98, B5 0.90, B2 0.90, B3 0.0 (InjecGuard
  AUROC 0.69). Near-saturated on 159 samples; not the contribution.
- **E5 over-defense (NotInject):** CANOPI FPR **0.348** at the 1% target (rises with target). Far
  from the "~0" success criterion. (Shared with B5 — encoder-level, not CANOPI-specific.)
- **E10 calibration:** ECE **0.20**, Brier 0.076 (pre temp-scaling; poor).
- **Russian fails:** ru_mt 0.28, consistent with its **18% back-translation accept rate** (encoder/MT ceiling).
- **Romanized Arabic** (ar_roman_nat) 0.20 — open challenge.
- **E4 threshold transfer:** CANOPI holds τ at ~1% FPR on deepset (fpr_at_tau 0.010 ✓) but
  recall_at_tau only 0.14 (AUROC 0.75); JBB recall 0.23. Cross-corpus generalization gap.
- ✅ Working: leakage clean (0 overlap); adaptive residual degrades smoothly 0.92→0.63 (E7);
  augmentation accept rates sane (>0.97 for 7/8 langs).

## Why, and the decided plan (A + B + C)
The multilingual encoder masks the invariance objective's value. Decided 2026-06-14:
- **A — monolingual-encoder mechanism test** (`configs/ablations/monolingual.yaml`): with a frozen
  EN encoder B5 should collapse cross-lingually; if CANOPI lifts it, the objective is vindicated.
  (Caveat: a frozen monolingual encoder gives the head little non-English signal, so this may
  confirm the negative — but it answers the question definitively.)
- **B — baselines through E4/E5/E7** (notebook now scores B1/B2/B3/B5 at their own τ on threshold
  transfer / over-defense / adaptive): tests whether the JOINT objective buys *operating-point
  robustness* even when raw recall ties B5. The pAUC term is an operating-point loss, so this is
  its most defensible home.
- **C — reframe** to "canonicalization + multilingual representation recovers detection that
  surface/English detectors lose, at fixed 1%-FPR, no LLM" — the floor narrative, to be
  *justified by* A/B rather than dodging the B5 question.

Falsification stance: if A and B both come back negative, the honest outcome is a simple-method
paper with the joint objective reported as a negative ablation. We do not spin.

## Reproduce
`runs/run_canopi_main_20260614/` (seed_42 CSVs + aggregated). Re-run full: notebook with
`SMOKE=False`. A: `CONFIG=configs/ablations/monolingual.yaml`. B6/B7: their `configs/baselines/` yamls.
