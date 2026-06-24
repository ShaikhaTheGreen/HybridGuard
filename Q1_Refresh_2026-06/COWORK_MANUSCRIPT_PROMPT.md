# Cowork prompt — think with me on the manuscript and where to submit

> Paste everything below the line into a fresh Claude session. It is **fully
> self-contained** — no files, repo, or attachments required. (If you happen to also have
> the manuscript draft or results digest handy, you can paste them in, but the prompt
> does not depend on them.)

---

I'm a Kuwait University faculty member writing up a security/ML paper and I want you to
be a **critical thinking partner and co-author**, not a yes-machine. Push back, surface
the weakest points before a reviewer does, and ask me clarifying questions before
committing to a direction. Work with me iteratively — one decision at a time.

## What the paper is

**Title (working):** *Certified Canonicalization: A Detector-Agnostic Defense against
Prompt-Injection Obfuscation.*

**The contribution, honestly stated:** Canonicalization-as-defense already exists (NFKC
shields, an open-source tool, a USPTO patent). Our delta is **not** the idea — it's:
1. A **decision-invariance certificate**: a deterministic, idempotent canonicalizer `c+`
   provably yields **0% attack success for ANY downstream detector** over the entire
   UTS#39 confusable + invisible-character + depth-bounded-encoding closure of an input,
   with no retraining. (A short theorem + an executable proof-obligation check.)
2. An **FPR-safe full-table fold** under a mixed-script gate (proven + measured not to
   harm genuine non-Latin text).
3. An **adaptive, leave-one-family-out evaluation**: a defense-aware optimizer that knows
   the defense; `c+` recovers detectors on attack families it was never tuned on.
4. A **transparent negative/conditional result** on the learned head (no in-domain gain;
   cross-lingual value only on a monolingual encoder).

**Key results** (all 5-seed, real detectors, thresholds frozen on validation at 1% FPR):
- Idempotence 1.0; closure-soundness 0.33→1.0 (hand map→full UTS#39); per-detector
  invariance 1.0 across four heterogeneous detectors (regex, TF-IDF+LinearSVM,
  ProtectAI/DeBERTa-v3, InjecGuard); FPR-neutral fold (0 benign flips on non-Latin text).
- Adaptive LOFO: `c+` strictly dominates base `c` on out-of-standard families (e.g.
  TF-IDF confusable-TR39/small-cap/invisible 0.08/0.02/0.07 → 0.99; InjecGuard 0.00 →
  0.42); semantic paraphrase is a measured no-op.
- In-domain: saturated benchmark; the joint objective ties its ablations (McNemar,
  Holm-corrected) — we **report this as a feature of the framing**, not a win.

**The framing I want to protect:** the contribution is the **certified primitive + the
adaptive evaluation**. The paper must NOT live or die on the learned head winning a
benchmark. Keep the honesty; do not let me oversell.

## My constraints and history (these matter for venue)

- For promotion, Kuwait University requires the venue to be in an **explicit Computer
  Science / Artificial Intelligence category** (Scopus/JCR). This is a hard filter.
- History: desk-rejected by *Applied Intelligence* (novelty/fit) → accepted a Springer
  transfer to *Neural Processing Letters* (Q2, Scopus AI) → we then rebuilt the work on
  correct numbers and added the certificate + adaptive evaluation to clear the novelty
  bar. We are **currently formatting for *Computers & Security* (Elsevier, JCR Q1)**.
- I have Arabic/Kuwaiti linguistic expertise (romanized-Arabic / Arabizi injection is a
  niche I can own that English-centric labs won't scoop).
- Author-side: I prefer a **reputable Q1, reasonable time-to-decision, and APC I can
  justify**. Open access is a plus but not required.

## What I want from you, in order

**A. Story & structure.** Stress-test the narrative arc. Is "certificate + adaptive
evaluation, learned head honestly conditional" the strongest framing, or is there a
sharper one (e.g. lead harder on the *certified* angle and demote the learned head to a
short section)? Propose 2–3 alternative framings with trade-offs. Tell me what a
skeptical reviewer attacks first and how to disarm it in the Introduction.

**B. Reviewer-proofing.** Given the honest negatives (no in-domain gain; conditional
cross-lingual head; over-defense at 1% FPR; certificate depends on the loaded UTS#39
table), draft the 3–4 most likely reviewer objections and the crisp response to each.
Flag anything that reads as overclaiming in the current abstract/contributions.

**C. Venue decision.** Recommend where to submit, as a **ranked shortlist** with explicit
trade-offs, scoring each on: (1) CS/AI category for KU promotion, (2) topicality/fit for
"certified defense + adaptive evaluation + LLM security", (3) tier/prestige, (4) typical
time-to-first-decision, (5) APC, (6) single vs double blind. Consider at least:
*Computers & Security*, and credible alternatives (security venues vs AI/ML journals vs
selective conferences). Be honest if you think C&S is the right call or if something fits
better. End with a clear recommendation and the one risk of that choice.

**D. The one open experimental decision.** The learned cross-lingual study
(monolingual-vs-multilingual encoder, Figure 3) is still single-seed from an older
notebook. Help me decide: (a) regenerate it 5-seed, (b) keep it with a footnote, or
(c) cut the figure and fold the claim into text. Weigh cost vs reviewer risk.

Start by asking me up to **3 clarifying questions** (e.g. promotion timeline, appetite for
a conference vs journal, whether I want to expand the Arabizi angle into its own threat
section). Then give me your first concrete recommendation on **(A) the framing** — don't
try to cover everything at once.
