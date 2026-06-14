# CANOPI — Neural Processing Letters submission (Overleaf)

Self-contained Overleaf project. **Upload to Overleaf:** New Project → Upload Project
→ drag `overleaf_canopi.zip` (or this folder zipped). Overleaf auto-detects `main.tex`;
the Springer `sn-jnl.cls` + `.bst` ship in the folder, so no template selection is needed.
Set the compiler to **pdfLaTeX** and the bibliography runs via BibTeX.

## Files
- `main.tex` — the manuscript (Springer `sn-jnl`, `sn-mathphys-num` numbered refs).
- `related_work.tex` — `\input` by main; CANOPI-positioned, verified citations.
- `references_npl.bib` — bibliography (arXiv IDs confirmed; a few author lists marked
  to expand before camera-ready).
- `sn-jnl.cls`, `sn-mathphys-num.bst`, `sn-basic.bst` — Springer template.
- `figures/` — `canopi_mechanism.pdf` (headline), `canopi_architecture.pdf` (Fig.1),
  `canopi_joint_objective.pdf` (Fig.2). All vector PDF, grayscale-legible.

## Status — what is FINAL vs PROVISIONAL
- **FINAL (5-seed, real):** the primary *substitution result* (Section 5.1, Table 1,
  Fig. 3A) — the monolingual-encoder mechanism. This is the paper's headline and is
  complete.
- **PROVISIONAL (smoke pass / pending runs), clearly marked in the text:**
  - Fig. 3B (multilingual-encoder control) — pending the full 5-seed `canopi_main` run.
  - Section 5.3 attribution (B6/B7 mono) — configs released, runs pending.
  - Section 5.4 in-domain + operating-point — smoke (seed 42); full run pending.

Replace each `[PROVISIONAL]` block with the 5-seed aggregated numbers from
`runs/<run_id>/aggregated/` when the runs complete. No fabricated numbers are present.

## Before submission (checklist)
- [ ] Run full 5-seed `canopi_main` (multilingual) → fill Fig. 3B + Section 5.4.
- [ ] Run `b6_mono` / `b7_mono` → fill the Section 5.3 attribution table.
- [ ] Verify the live NPL guidelines for current reference style (author-year vs numbered).
- [ ] Expand `and others` author lists in `references_npl.bib`.
- [ ] Confirm ORCIDs and the Declarations (funding/contributions) with all co-authors.
- [ ] Compile on Overleaf; resolve any `\orcidID`/`sn-jnl` warnings.
