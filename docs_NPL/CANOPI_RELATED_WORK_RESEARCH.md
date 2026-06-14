# CANOPI — Related Work research (verified 2026-06-14)

Source: deep-research pass (105 agents, 23 sources, 97 claims → 25 adversarially
verified, 24 confirmed / 1 killed). Confidence tags are from 3-vote verification.
**Flagged items still need a manual arXiv check before they enter the .bib.**

## A. Verified citations + reported limitation

| key | work | arXiv / id | year | reported limitation | conf |
|---|---|---|---|---|---|
| liu2024injecguard | InjecGuard + **NotInject** benchmark | 2410.22770 | 2024 | SOTA guards drop to ~60% (near-random) on benign instruction-like text → **over-defense** | high |
| metapromptguard2 | Meta **Prompt Guard 2** (86M/22M) | HF model card | 2025 | 86M: 97.5% R@1%FPR, AUC .998; 22M: 88.7%. Only 86M (mDeBERTa) is multilingual. **Vendor self-reported, not peer-reviewed.** | high |
| protectai2024 | ProtectAI **deberta-v3-base-prompt-injection-v2** | HF model card | 2024 | **English-only; does not detect jailbreaks; produces false positives** | high |
| hackett2025 | Character-injection / **emoji smuggling** evasion study | 2504.11168 | 2025 | Up to **100% evasion** across six systems (ProtectAI v2 → 20.26%); does **not** address cross-lingual or canonicalization | high |
| yong2023 | **Low-Resource Languages Jailbreak GPT-4** (Yong et al.) | 2310.02446 | 2023 | Translation attack ~79% success; **defense is model-side only** | high |
| deng2024 | **Multilingual Jailbreak Challenges** (Deng et al.) | 2310.06474 | ICLR 2024 | 80.9%/40.7%; proposed **Self-Defense = safety fine-tuning (model-side), not detector-side** | high |
| benchmark2025 | 2025 guardrail/detector benchmarking study | 2502.15427 | 2025 | Detector performance varies substantially; **many fail multilingual** | high |
| hines2024 | **Spotlighting** (Hines et al., Microsoft) | 2403.14720 | 2024 | Cuts attack >50%→<2% but **orthographic/structural delimiting only** | high |
| jain2023 | Paraphrase / retokenization baseline defenses (Jain et al.) | 2309.00614 | 2023 | **Input preprocessing only; no multilingual** | high |
| rcs2025 | **RCS** — contrastive safety detector (NEAREST competitor) | 2512.12069 | 2025 | Reaches FPR<1% by **post-hoc threshold calibration, NOT partial-AUC optimization**; no cross-lingual | high |

## B. The gap (medium confidence — convergent absence + RCS datum)
**Unoccupied:** no published method combines [deterministic canonicalization +
learned cross-lingual intent-invariance + partial-AUC low-FPR objective, no LLM at
inference]. CANOPI's headline — **cross-lingual Recall@1%FPR at a fixed threshold** —
is unclaimed. RCS (2512.12069) is the closest: contrastive + low-FPR, but the FPR is
hit by post-hoc calibration (not optimized) and it has no cross-lingual component.

## C. Refuted assumption (KILLED 1-2) — and why it HELPS us
"Cross-lingual vulnerability is concentrated in *low-resource* languages" → **refuted.**
High-/mid-resource languages (e.g., Spanish, French) also bypass detectors. → **Do NOT
frame the multilingual claim as low-resource-specific.** This *strengthens* the diverse-8
panel: language shift defeats detectors generally, so high-resource languages are fair game.

## D. FLAGGED — not verified by this pass; confirm manually before citing
- **khosla2020** Supervised Contrastive Learning (NeurIPS 2020, likely arXiv 2004.11362) — standard, but verify id.
- **deeptoppush** DeepTopPush (Adam/Mácha et al.) — partial-AUC/top-push; **verify exact citation** (could not confirm in pass).
- **libauc / yuan** AUC-margin / partial-AUC library (Yuan et al.) — verify id (likely ICCV 2021 / 2012.03173).
- **carlini** adaptive-attack evaluation norm — verify ("On Evaluating Adversarial Robustness", 2019).
- **Lakera Guard / Llama Guard** — **NO evidence found**; do not cite specific numbers. Drop or mark proprietary.
- 5 additional 2025 detection/adaptive papers surfaced but not characterized: 2510.09023, 2506.10597, 2511.22047, 2512.01353, 2507.07417 — fetch titles before deciding whether to cite.

## E. Related Work draft (~250 words, citation keys above)
Prompt-injection and jailbreak detection has converged on fine-tuned encoder
classifiers—ProtectAI's DeBERTa-v3 \cite{protectai2024}, Meta's Prompt Guard~2
\cite{metapromptguard2}, and InjecGuard \cite{liu2024injecguard}. These reach strong
in-distribution accuracy (Prompt Guard~2 reports 97.5\% recall at 1\% FPR
\cite{metapromptguard2}) but show two systematic failures. First, they \emph{over-defend}:
on NotInject, state-of-the-art guards collapse to near-random ($\sim$60\%) accuracy on
benign instruction-like text \cite{liu2024injecguard}. Second, they are brittle
off-distribution—orthographic perturbations such as emoji smuggling evade them almost
completely (up to 100\% across six systems) \cite{hackett2025}, and most are English-only
\cite{protectai2024}. A parallel attack literature shows that simply \emph{translating} an
unsafe prompt defeats aligned models and their detectors \cite{yong2023,deng2024}, and this
weakness is \emph{not} confined to low-resource languages. Yet the only defenses are
model-side (multilingual safety fine-tuning \cite{deng2024}); no detector-side method
recovers cross-lingual detection. Input-transformation defenses—NFKC normalization,
spotlighting \cite{hines2024}, paraphrase/retokenization \cite{jain2023}—neutralize
orthographic and structural attacks but address neither semantic nor cross-lingual
paraphrase. Finally, recent contrastive detectors attain a strict $<$1\% FPR operating point
only by \emph{post-hoc} threshold calibration \cite{rcs2025}, not by optimizing the low-FPR
region during training; partial-AUC objectives \cite{deeptoppush,libauc} remain unused in
this domain. CANOPI is, to our knowledge, the first detector to unify deterministic
canonicalization, a learned cross-lingual intent-invariant representation, and a partial-AUC
low-FPR objective—closing the gap none of these threads address: recovering cross-lingual
Recall@1\%FPR at a fixed operating threshold, with no LLM in the inference path.
