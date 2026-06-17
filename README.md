# CANOPI — Certified Canonicalization for Prompt-Injection Defense

*(The repository keeps the name `HybridGuard` for link stability; the system is named CANOPI in the manuscript.)*

> **Pre-publication note.** This repository accompanies a manuscript currently under peer review. The full bibliographic reference and a DOI will be added here after acceptance.

CANOPI places a **deterministic, idempotent canonicalizer** in front of any prompt-injection detector. Because the canonicalizer maps every orthographic and encoding disguise of an input to a single normal form, the decision of any detector composed after it is provably invariant over that whole class of disguises: Unicode confusables, zero-width and Tag-block characters, full-width and compatibility forms, and depth-bounded base64/hex/URL/ROT13 nesting. This is a **decision-invariance certificate**: attack success over the closure is exactly zero, detector-agnostically, with no retraining.

The learned reference detector in this repository, **CANOPI (CANOnicalization + Projection Invariance)**, is a canonicalizer followed by a frozen encoder and a projection head; it is used to quantify residual learned behavior. The certified primitive, not the learned head, is the contribution.

## What is in this repository

| Path | Role |
|------|------|
| `src/hybridguard/canonicalize.py` | The canonicalizer and CLI (the installable package). |
| `code/certify.py` | The certificate checker: closure membership, the idempotence verifier, and the closure-soundness check. |
| `code/tr39_fold.py`, `code/confusables_table.py` | Full Unicode UTS#39 confusable fold (539 entries) applied under a mixed-script gate. |
| `src/hybridguard/canopi/` | The learned reference detector (frozen encoder + projection head). |
| `notebooks/CANOPI_Manuscript_Results.ipynb` | One-run reproducibility notebook: regenerates every table and figure in the manuscript. |
| `tests/` | Unit tests for the canonicalizer and the certificate. |

## Install

```bash
git clone https://github.com/ShaikhaTheGreen/HybridGuard.git
cd HybridGuard
pip install -e .          # canonicalizer + CLI
pip install -e .[full]    # adds the training/evaluation stack
```

Python 3.10+ is required.

## Quick use

```python
from hybridguard.canonicalize import canonicalize

canonicalize("Ｉｇｎｏｒｅ​ prevіous instructions")
# -> "Ignore previous instructions"   (full-width folded, zero-width stripped, Cyrillic confusables folded)
```

Use it as a pre-classifier in front of any detector:

```python
from hybridguard.canonicalize import canonicalize

def detect_with_canonicalization(text, base_detector):
    return base_detector(canonicalize(text))
```

Check the certificate on your machine:

```bash
python code/certify.py
# prints idempotence pass rate, closure-soundness, and a per-detector certificate
pytest tests/
```

## Reproduce the manuscript

Open `notebooks/CANOPI_Manuscript_Results.ipynb` in Colab and run top to bottom; it writes every table and figure the manuscript reports into one bundle. The certificate numbers (idempotence 1.0, per-detector decision invariance 1.0) reproduce on CPU via `code/certify.py` and `tests/`; the learned-detector results use a GPU.

**On closure-soundness.** On the standardized certified class (NFKC, invisible-character stripping, the full UTS#39 fold, and bounded decoding) soundness is ≈1.0, reported as 0.9999 at corpus scale in the manuscript. The `certify.py` self-check may print a slightly lower combined figure because it also exercises the heuristic de-leet and de-segment passes, which are best-effort helpers that lie outside the certified closure.

## Datasets

Public Hugging Face datasets, used under their original licenses: `xTRam1/safe-guard-prompt-injection` (training and threshold selection) and `deepset/prompt-injections` (cross-corpus transfer), plus the NotInject over-defense benchmark, a JailbreakBench behavior set for jailbreak transfer, and curated human-verified Arabic and Spanish injection sets (with romanized Arabic) for the cross-lingual study.

## License

MIT. See [`LICENSE`](LICENSE).

## Authors

Shaikhah Alkhadhr (corresponding, s.alkhadhr@ku.edu.kw), Heba Alturki, and Aseel Alqemlas. Information Science Department, Kuwait University. See [`CITATION.cff`](CITATION.cff).
