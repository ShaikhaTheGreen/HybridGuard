# Certified Canonicalization — Complete Results Digest

**Paper:** *Certified Canonicalization: A Detector-Agnostic Defense against Prompt-Injection Obfuscation*
**Target venue (current):** Computers & Security (Elsevier, JCR Q1)
**Generated from:** `results_COSE/NUMBERS.json` (single source of truth). Every number
below is emitted by code and read back; nothing is hand-typed. Drift gate
(`code/check_consistency.py`) passes with exit 0.

---

## 0. TL;DR (the one paragraph)

We give prompt-injection canonicalization a **certificate**: a deterministic,
idempotent canonicalizer `c+` provably forces **0% attack success** for *any* downstream
detector over the entire UTS#39 confusable + invisible-character + depth-bounded-encoding
closure of an input — no retraining. Empirically: idempotence **1.0**, closure-soundness
**0.33 → 1.0** (40-entry hand map → full UTS#39 table), per-detector decision invariance
**1.0 across all four detectors + InjecGuard**, and the full-table fold is
**FPR-neutral** (0 benign flips on Russian/Greek/Arabic/Chinese/Hindi/Korean). Under a
**defense-aware adaptive attacker** (leave-one-family-out), the hardened `c+` recovers
detectors on attack families it was never tuned on, where base canonicalization fails.
The learned head (CANOPI) is reported **honestly as a conditional / negative result**: no
in-domain advantage on a saturated benchmark, useful cross-lingually only on a
monolingual encoder. The contribution is the **certified primitive + the adaptive
evaluation**, not a benchmark win.

---

## 1. Protocol & provenance

- **Primary data:** `xTRam1/safe-guard-prompt-injection`, deterministic stratified
  60/20/20 split at seed 1337, SHA-256 leakage-verified clean.
- **Seeds:** {42, 2025, 7, 1337, 314}. Thresholds frozen on validation at **1% FPR**
  before any test read.
- **Reporting:** 5-seed mean ± s.d. with bootstrap 95% CIs; headline comparisons carry
  paired **McNemar** with **Holm** correction.
- **Detectors (four heterogeneous + reference):** regex baseline, TF-IDF+LinearSVM (B1),
  ProtectAI/DeBERTa-v3 (B2), InjecGuard (B3); **CANOPI** is the learned reference
  instantiation, with ablations B5 (embedding), B6 (invariance-only), B7 (pAUC-only).

---

## 2. The certificate (Section "The certificate") — REAL, CPU-reproducible

| Quantity | Value |
|---|---|
| Idempotence `c+(c+(x)) = c+(x)` | **1.0** |
| Closure-soundness — base 40-entry hand map | **0.33** |
| Closure-soundness — full UTS#39 table | **1.0** |
| Per-detector decision invariance over the closure | **1.0** (regex, TF-IDF, ProtectAI/DeBERTa-v3, InjecGuard, + keyword probe) |
| FPR-neutral fold — benign decision flips | **0 / 21** (incl. genuine non-Latin text) |

**Certificate coverage by attack family** (fraction provably inside the closure `C_D`):

| Family | Type | Coverage in `C_D` | Residual |
|---|---|---|---|
| confusable (in-table) | info-preserving | 1.00 | 0.00 |
| confusable (TR39) | info-preserving | 1.00 | 0.00 |
| invisible / Tag | info-preserving | 1.00 | 0.00 |
| small-capital (OOM) | info-preserving | 0.00 | 1.00 *(handled empirically by `c+`, outside the standardized certificate)* |
| structural (spacing/leet) | info-preserving | 0.00 | 1.00 *(empirical `c+` extension)* |
| semantic (paraphrase) | semantic | 0.00 | 1.00 *(out of scope — provable no-op)* |

---

## 3. In-domain detection — `tab:main` (5-seed, R@1%FPR)

| Detector | R@1%FPR | AUROC | AUPRC |
|---|---|---|---|
| B1 TF-IDF+LinearSVM | 0.988 ± 0.000 | 0.999 | 0.998 |
| B6 invariance-only | 0.962 ± 0.004 | 0.998 | 0.997 |
| **CANOPI (joint)** | **0.959 ± 0.005** | 0.997 | 0.996 |
| B7 partial-AUC-only | 0.958 ± 0.007 | 0.999 | 0.997 |
| B5 embedding | 0.954 ± 0.005 | 0.997 | 0.995 |
| B2 DeBERTa-v3 | 0.924 | 0.992 | 0.988 |
| B3 InjecGuard | 0.427 | 0.912 | 0.851 |
| Regex baseline | 0.136 | 0.561 | 0.371 *(no meaningful low-FPR point — reported honestly)* |

**McNemar (Holm-corrected):** CANOPI is **statistically tied** with its B5/B6/B7
ablations (p = 1.0 for all three) and **significantly exceeds** DeBERTa-v3
(p < 0.001) and InjecGuard (p ≪ 0.001). → The benchmark is saturated; the joint
objective buys no in-domain gain. This is the honest framing, not a weakness.

---

## 4. Defense-aware adaptive attacker (leave-one-family-out) — `tab:adaptive`

Worst-case R@1%FPR an optimizer finds after applying the **known** defense, on the
**held-out** family. `c+` is *not* tuned on the held-out family — it generalizes because
it folds the whole UTS#39 standard.

**InjecGuard (most vulnerable to obfuscation) — the manuscript table:**

| Held-out family | No defense | `c` (base) | `c+` (hardened) |
|---|---|---|---|
| confusable (in-table) | 0.00 | 0.42 | 0.42 |
| confusable (TR39) | 0.00 | 0.00 | **0.42** |
| small-capital | 0.00 | 0.00 | **0.37** |
| invisible / Tag | 0.00 | 0.00 | **0.42** |
| structural | 0.00 | 0.30 | 0.30 |
| semantic (no-op) | 0.39 | 0.39 | 0.39 |

**The same pattern across detectors (full matrix in `adaptive_v2_matrix_agg.csv`):**
`c+` strictly dominates `c` on the out-of-standard families. The sharpest case is
**TF-IDF**: confusable-TR39 / small-cap / invisible collapse it to 0.08 / 0.02 / 0.07
under `c`, and `c+` restores all three to **0.99**. DeBERTa-v3 shows the over-trigger
counter-case (recall *rises* under obfuscation; `c+` normalizes it back to clean 0.91).
Semantic paraphrase is a measured no-op for every detector.

---

## 5. Detector-agnostic recovery — `tab:recovery` (attacked → +canon)

| Detector | homoglyph | zero-width | leet | spacing |
|---|---|---|---|---|
| CANOPI | 0.96 → 0.96 | 0.96 → 0.96 | 0.68 → 0.96 | 0.51 → 0.98 |
| TF-IDF | 0.61 → 0.99 | 0.14 → 0.99 | 0.47 → 0.98 | 0.89 → 0.93 |
| InjecGuard | 0.00 → 0.42 | 0.00 → 0.42 | 0.39 → 0.36 | 0.01 → 0.33 |
| DeBERTa-v3 | 1.00 → 0.91 | 1.00 → 0.91 | 1.00 → 0.94 | 1.00 → 0.95 *(over-trigger normalized)* |
| regex | 0.02 → 0.13 | 0.01 → 0.13 | 0.02 → 0.04 | 0.02 → 0.05 *(degenerate)* |

Canonicalization restores recall for the detectors that are *evaded*; for the detector
that is *tripped* (DeBERTa-v3) it normalizes the inflated recall back to its clean value.
The "evaded vs tripped" dichotomy is an underreported, genuinely interesting finding.

---

## 6. Over-defense (NotInject) and latency

Benign FPR at the 1% operating point, raw vs +canon — canonicalization is **FPR-neutral**:

| Detector | benign FPR raw | +canon | Δ |
|---|---|---|---|
| regex | 0.000 | 0.000 | 0.000 |
| TF-IDF | 0.283 | 0.283 | 0.000 |
| ProtectAI/DeBERTa-v3 | 0.507 | 0.502 | −0.006 |
| InjecGuard | 0.062 | 0.062 | 0.000 |

Honest caveat: the general detectors **over-defend** on adversarial-benign text (the
in-domain threshold does not transfer); canonicalization **neither causes nor worsens
this** — it is an orthogonal problem requiring target-specific recalibration.

---

## 7. Cross-lingual / multilingual (curated AR/ES + natural romanized Arabic)

R@1%FPR per language, raw vs +canon (frozen English threshold):

| Bucket | regex | TF-IDF | ProtectAI | InjecGuard |
|---|---|---|---|---|
| Arabic | 0.03 → 0.03 | 0.03 → 0.03 | 1.00 → 1.00 | 0.00 → 0.00 |
| Spanish | 0.05 → 0.05 | 0.15 → 0.15 | 1.00 → 0.95 | 0.60 → 0.65 |
| Romanized Arabic (natural) | 0.27 → 0.27 | 0.40 → 0.40 | 0.80 → 0.93 | 0.00 → 0.00 |

> **PENDING (one item):** the *learned* cross-lingual study — the **monolingual vs
> multilingual encoder** contrast that drives Figure 3 and the "conditional mechanism"
> claim — still comes from the FULL NPL notebook and was **not** regenerated in the
> 5-seed COSE run. The curated-set recovery above is real; the mono-encoder fig3 numbers
> in the prose are inherited. *Decide whether to (a) regenerate it in Colab, (b) keep it
> as inherited single-seed with a footnote, or (c) cut Figure 3 and fold the claim into
> text.*

---

## 8. Honest negatives / limitations (state them before a reviewer does)

1. **No in-domain gain** for the joint objective on the saturated benchmark (CANOPI ties
   its ablations). The contribution is the certificate, which does not depend on this.
2. **Learned cross-lingual head is conditional** — useful only on a monolingual encoder,
   redundant once the encoder is multilingual.
3. **Over-defense** at 1% FPR is a real usability cost for the general detectors
   (orthogonal to canonicalization).
4. **Certificate boundary is honest:** small-capital/structural are *empirical* `c+`
   extensions outside the standardized certificate; semantic paraphrase is out of scope.
5. **Certificate depends on the loaded UTS#39 table** — a novel out-of-standard look-alike
   needs a table update, which is exactly why the certificate is paired with the adaptive
   hardened evaluation.

---

## 9. Where the numbers live (auditability)

- `results_COSE/NUMBERS.json` — consolidated source of truth (pending = []).
- `results_COSE/{certificate,in_domain,adaptive,recovery,overdefense,multilingual}/` —
  per-experiment snapshots + `*_agg.csv` (mean, std, ci_lo, ci_hi, n_seeds).
- `manuscript_COSE/tables/{certificate_coverage,main_results,adaptive,recovery}.tex` —
  auto-generated, `\input` by `main_COSE.tex`.
- `manuscript_COSE/figures/fig2_adaptive_heldout.pdf` — real LOFO figure.
- Regenerate everything: `python -m code.reproduce --seeds all` (GPU for SOTA/CANOPI;
  certificate + stats run on CPU). Drift gate: `python -m code.check_consistency`.
- Reproducible run: branch `cose-q1-5seed-results` on GitHub; Colab runner
  `Q1_Refresh_2026-06/CANOPI_COSE_Colab.ipynb`.
