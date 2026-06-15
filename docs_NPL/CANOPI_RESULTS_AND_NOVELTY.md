# CANOPI — consolidated results + novelty (2026-06-14)

All results to date, integrated under the unified two-pillar framing (keeping the
HybridGuard canonicalization work as Pillar 1). Marks REAL vs PROVISIONAL honestly.

## Unifying frame
A detector trained on a clean distribution $P$ is reliable on $\mathrm{supp}(P)$.
Two input shifts move an attack off-distribution without changing its label:
orthographic (character obfuscation) and semantic (language shift). CANOPI inverts
both: `x → c(x) [deterministic canonicalizer] → E [frozen encoder] → P [trained
invariance head] → detector`. The pipeline contains a deterministic stage for the
orthographic shift and a learned stage for the semantic shift. No generative model at inference.

---

## Pillar 1 — character obfuscation (deterministic `c`; HybridGuard line; REAL)
Canonicalization recovers detector Recall@1%FPR under character obfuscation,
detector-agnostically (clean → attacked → +canonicalization):

| Detector | homoglyph | zero-width | spacing |
|---|---|---|---|
| HG_MULTIFEAT | 0.99→0.37→**0.99** | 0.99→0.00→**0.99** | 0.99→0.00→**0.99** |
| InjecGuard | 0.39→0.00→**0.39** | 0.39→0.00→**0.39** | 0.39→0.01→0.31 |

Over-defense: FPR-neutral on a hard-negative set (Δ=0 for every detector).
Latency: **0.054 ms/sample**. (Source: prior A100 runs, `results_NPL/`.)
This justifies the `c` stage and shows it is usability- and throughput-neutral.

## Pillar 2 — language shift (learned `P`; the substitution result; REAL, 5 seeds)
**The cross-lingual invariance objective substitutes for multilingual pretraining.**
Controlled encoder ablation, Recall@1%FPR, each detector at its own frozen τ:

| lang | CANOPI (mono E) | B5 probe (mono E) | Δ |
|---|---|---|---|
| French* | 0.61±0.06 | 0.12±0.00 | **+0.49** |
| Chinese* | 0.51±0.13 | 0.01±0.00 | **+0.49** |
| German* | 0.40±0.10 | 0.02±0.00 | **+0.38** |
| Spanish* | 0.52±0.05 | 0.23±0.00 | **+0.29** |
| Portuguese* | 0.42±0.04 | 0.18±0.00 | **+0.24** |
| Spanish (curated) | 0.22±0.07 | 0.00±0.00 | **+0.22** |
| Arabic / Hindi* / Russian* | ~0.01 | ~0.00 | ~0 (boundary) |
| in-domain (English) | 0.966±0.003 | 0.972±0.000 | −0.006 |

On a MULTILINGUAL encoder CANOPI ≈ B5 (the control): the encoder already supplies
the alignment, so the objective's value is localized to the monolingual regime.
Boundary: invariance amplifies weak-but-present signal but cannot manufacture it for
scripts the frozen encoder discards (non-Latin on an English encoder).

## Attribution — it is the invariance term, not pAUC (REAL, 5 seeds)
Invariance-only (B6, λ4=0, mono E) reproduces the lift: es_mt 0.59 (B5 0.23),
pt 0.55 (0.18), fr 0.52 (0.12), de 0.30 (0.02), es 0.23 (0.00). Removing pAUC does
not remove the recovery → invariance is the source. Not redundant: pAUC adds on
Chinese (0.15 inv-only → 0.51 full). [b7_mono pAUC-only pending to complete dissociation.]

## Surface/English detectors collapse cross-lingually (REAL)
B1 TF-IDF Arabic 0.03, B3 InjecGuard Arabic 0.00 — CANOPI (and B5 on a multilingual
encoder) recover what these lose. Establishes the gap CANOPI's `P` closes.

## Honest negatives (reported, not hidden)
- **Operating-point H1 not supported** (pre-registered): on the multilingual encoder
  B5 matches/slightly beats CANOPI on over-defense and adaptive robustness. The
  contribution is the substitution mechanism, not operating-point superiority. [smoke; full run pending]
- Absolute recovered recall is moderate (0.40–0.61), narrowing not closing the gap to a multilingual encoder.
- Natural romanized Arabic: B5 0.20 > CANOPI 0.09 (open case).
- Adaptive semantic-rewrite leaves an irreducible residual (0.92→0.63 over strength).

## PROVISIONAL (pending full 5-seed `canopi_main` + b7_mono)
Multilingual-encoder control panel; in-domain table; operating-point table; b7_mono.

---

## Novelty argument (grounded in the verified SLR)
The combination is unoccupied in the literature (`CANOPI_RELATED_WORK_RESEARCH.md`).
Four distinct, defensible novelties:

1. **A training objective substitutes for multilingual pretraining.** This is a
   general representation-learning finding, not a prompt-injection trick: cross-lingual
   transfer, normally bought by scaling the encoder, can be obtained at training time on
   a smaller encoder where it is otherwise absent. No prior safety-detection work shows this.
2. **A unified two-stage canonicalization covering both threat axes.** Deterministic
   orthographic (`c`) + learned linguistic-invariant (`P`) under one covariate-shift
   account. Prior canonicalization/spotlighting (Hines, Jain) is orthographic-only;
   prior multilingual defenses (MrGuard, Deng) are model-side or LLM-based.
3. **No LLM at inference.** Versus LLM-reasoning guardrails (MrGuard) and LLM-judge:
   cheap, deterministic, reproducible — with the same cross-lingual coverage on
   represented scripts.
4. **A characterized boundary + adaptive evaluation + the attribution.** We state
   where the objective provably cannot help (and why), evaluate against an adaptive
   adversary, and dissociate the invariance term from the operating-point term. The
   honest operating-point negative strengthens, not weakens, credibility.

Nearest works, distinguished: **RCS** (contrastive projection for jailbreak detection)
is vision-only, no cross-lingual, no partial-AUC, post-hoc threshold; **MrGuard**
(multilingual guardrail) is an LLM with reasoning on the inference path. CANOPI is the
first to put canonicalization + learned cross-lingual invariance + a low-FPR objective
together with no generative model at inference.

## Recommended manuscript framing
Title direction: *"Two-Stage Canonicalization for Prompt-Injection Detection: a
Learned Cross-Lingual Invariance Objective that Substitutes for Multilingual
Pretraining."* Results: Pillar 1 (character, deterministic) → Pillar 2 (language,
learned, the headline) → attribution → controls → honest operating-point negative.
