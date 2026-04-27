# HybridGuard

**Canonicalization as a detector-agnostic primitive for prompt-injection detection in LLM pipelines.**

> ⚠️ **Pre-publication note.** This repository accompanies a manuscript currently under peer review. The work has *not* yet been published. Citation details and corresponding-author contact will be added to this README once the manuscript is accepted; until then, please treat any results in this repository as preliminary and refrain from citing.

HybridGuard ships a deterministic canonicalization front-end — NFKC normalization, zero-width / Tag-block stripping, Unicode TR39 confusable folding, and depth-bounded base64/hex/URL/ROT13 unwrap — that any downstream prompt-injection detector can adopt as a pre-classifier defense, independent of HybridGuard itself. Quantitative results, benchmark numbers, and a per-detector evaluation are reported in the accompanying manuscript (under peer review).

🤗 **Live demo:** https://huggingface.co/spaces/ProfSK/hybridguard-canonicalize

---

## Install

```bash
git clone https://github.com/ShaikhaTheGreen/HybridGuard.git
cd HybridGuard

# Minimal install — canonicalization front-end + CLI
pip install -e .

# Full install — adds training/evaluation dependencies
pip install -e .[full]
```

Python 3.10+ is required.

---

## Quick use

### Canonicalize text (Python)

```python
from hybridguard import canonicalize

result = canonicalize("Ｉｇｎｏｒｅ\u200b all previous instructions")
print(result.canonical)         # "Ignore all previous instructions"
print(result.lowercase)         # "ignore all previous instructions"
print(result.trace)             # ['strip_invisibles', 'nfkc', 'fold_confusables']
print(result.decoded_payloads)  # []  (no base64/hex/URL payloads to unwrap)
```

### Canonicalize text (command line)

```bash
hybridguard canonicalize "Ｉｇｎｏｒｅ\u200b all previous instructions"
# Ignore all previous instructions
```

### Use canonicalization as a pre-processor for any other detector

```python
from hybridguard import canonicalize

# In your existing pipeline, just prepend:
def my_detector_with_canonicalization(text):
    canonical_text = canonicalize(text).canonical
    return my_existing_detector(canonical_text)
```

The accompanying manuscript measures the recovery this gives across four heterogeneous detectors.

---

## Datasets

HybridGuard is evaluated against four public Hugging Face prompt-injection datasets, redistributed under their original licenses. The full inventory — per-dataset source URLs, licenses, sizes, splits, and the role each one plays in the evaluation — will be published with the manuscript.

---

## Pipeline architecture

The HybridGuard input-side pipeline applies two stages, in order:

```
input  →  canonicalize(c)  →  sanitize(g)  →  feature_extraction  →  classifier
```

**Stage 1 — Canonicalization** (`c`) is the empirically-evaluated contribution of this work: deterministic, idempotent, audit-traceable, lossless. Tested across four heterogeneous detectors as part of the under-review manuscript.

**Stage 2 — Sanitization** (`g`) is a defined interface with four modes: `off`, `rule_strip`, `context_isolation`, `llm_rewrite_optional`. The non-generative modes are implemented and run in our pipeline; the ablation reports them as a no-op on detection AUROC. The generative mode (`llm_rewrite_optional`) is implemented as an interface but its security-utility trade-off is not centrally evaluated in this work — the manuscript explains the architectural choice to lean on canonicalization rather than generative sanitization, and lists a full sanitization study as a deferred topic. The HF Space demo includes the two non-generative sanitization modes for illustrative purposes.

---

## Reproduce the experiments

The training and evaluation pipeline is currently driven by the Colab orchestrator notebook at [`notebooks/HybridGuard_Colab_Orchestrator.ipynb`](notebooks/HybridGuard_Colab_Orchestrator.ipynb).

```bash
# 1. Open the orchestrator notebook in Colab (A100 GPU recommended)
#    https://colab.research.google.com/github/ShaikhaTheGreen/HybridGuard/blob/main/notebooks/HybridGuard_Colab_Orchestrator.ipynb

# 2. Follow docs/COLAB_RUNBOOK_v2.md for the step-by-step recipe.
#    ~5–8 hours for all 5 seeds × 4 HybridGuard variants + baselines on an A100.

# 3. Outputs land in /content/drive/MyDrive/HybridGuard/runs/<run_id>/
#    Per-seed CSVs, aggregated mean ± std, and rendered LaTeX tables.
```

Standalone evaluation notebooks reproduce specific experiments:

| Notebook | Reproduces |
|---|---|
| `HybridGuard_Colab_Orchestrator.ipynb` | Full 5-seed pipeline (main detection, robustness, ablation, calibration, multilingual fairness) |
| `HybridGuard_CrossCorpus_Eval.ipynb` | Cross-corpus generalization (held-out `deepset/prompt-injections`) |
| `HybridGuard_WS1_Universal_Eval.ipynb` | Canonicalization applied to four heterogeneous detectors (universal pre-classifier defense) |
| `HybridGuard_LLM_Judge_Baseline.ipynb` | Cost–quality comparison against an LLM-as-judge baseline |

---

## Evaluation dashboard

An interactive evaluation dashboard is included for local exploration of run results:

```bash
cd dashboard
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 dashboard.py     # opens http://127.0.0.1:8050
```

The dashboard auto-detects run directories and surfaces ROC curves, calibration reliability diagrams, ablation contributions, robustness heatmaps, fairness gaps, over-defense behavior, architecture schematics, and a sanitization trade-off view. See [`dashboard/README.md`](dashboard/README.md).

---

## Repo layout

```
HybridGuard/
├── src/hybridguard/        # The installable library (canonicalization + CLI)
├── tests/                  # pytest unit tests for the canonicalization module
├── scripts/                # Research utilities (figure generation, packaging)
├── configs/                # YAML configs for reproducibility
├── notebooks/              # Colab orchestrator + standalone evaluation notebooks
├── demo/                   # HF Space Gradio app (canonicalize + sanitize)
└── dashboard/              # Interactive evaluation dashboard (Dash + Plotly)
```

Reproducibility artifacts (bibliography, generated tables, per-seed evaluation snapshots, paper figures) and full documentation will be added to this repository once the work is published.

---

## Citation

This repository accompanies a research manuscript that is currently undergoing peer review and has not yet been published. Bibliographic details — title, authors, journal, volume/issue, page numbers, and DOI — will be added to this section after acceptance. Please refrain from citing the work until that point.

---

## Contact

Author and institutional contact details will be added to this section after the manuscript is accepted for publication.
