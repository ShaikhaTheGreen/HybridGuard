# CANOPI — Results summary

Headline results from the manuscript *"Certified Canonicalization: A Detector-Agnostic Defense against Prompt-Injection Obfuscation."* All numbers are reproduced by `notebooks/CANOPI_Manuscript_Results.ipynb`; the certificate numbers also reproduce on CPU via `code/certify.py` and `pytest tests/`.

## The certificate (core contribution)

- Idempotence of the canonicalizer c+: **1.0**.
- Per-detector decision invariance over the closure: **1.0** across four heterogeneous detectors.
- Closure-soundness on the standardized class (full UTS#39 fold, invisible-character stripping, bounded decode): **≈1.0** (0.9999 at corpus scale).
- Consequence (Theorem 1): attack success over the closure is exactly **0**, for any detector, with no retraining.

## FPR neutrality

- **0** benign decision flips on Russian, Greek, Arabic, and Chinese text, enforced by the mixed-script gate.

## Adaptive, held-out evaluation

- Under a defense-aware optimizer (leave-one-family-out), the hardened canonicalizer c+ restores detectors on attack families that base canonicalization misses (for example, InjecGuard from **0.00 to 0.43** at 1% FPR on the held-out family).

## Learned objective (transparent negative result)

- The learned cross-lingual invariance head recovers cross-lingual recall only on a monolingual encoder, is redundant once the encoder is multilingual, and gives no in-domain advantage on the saturated benchmark. This is reported honestly; it does not affect the certified primitive, which is the contribution.

## Reproduce

- Certificate (CPU): `python code/certify.py`; `pytest tests/`.
- Full tables and figures (GPU): `notebooks/CANOPI_Manuscript_Results.ipynb`.
