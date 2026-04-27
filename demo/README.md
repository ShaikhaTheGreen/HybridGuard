---
title: HybridGuard Canonicalize
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.5.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# HybridGuard Canonicalization — Hugging Face Space

This folder contains a self-contained Gradio app that demonstrates the HybridGuard canonicalization front-end as a public web demo.

## Live demo (after deployment)

`https://huggingface.co/spaces/ProfSK/hybridguard-canonicalize` *(deploy and update the URL once live)*

## Files

| File | Purpose |
|---|---|
| `app.py` | The Gradio interface (single file, runs on CPU; ~50 ms per inference) |
| `requirements.txt` | Pip dependencies (gradio + hybridguard from GitHub) |
| `README.md` | This file (also serves as the Space's README on HF) |

## Deploy to Hugging Face Spaces (5 minutes)

1. **Sign in** at https://huggingface.co (free account).
2. Go to https://huggingface.co/new-space.
3. **Owner**: your HF username (`ProfSK`). **Space name**: `hybridguard-canonicalize`. **License**: MIT. **SDK**: **Gradio**. **Hardware**: CPU basic (free).
4. Click **Create Space**. You land on a new git repo.
5. **In a terminal**, clone the new Space and copy these files in:
   ```bash
   git clone https://huggingface.co/spaces/ProfSK/hybridguard-canonicalize
   cd hybridguard-canonicalize
   cp ~/path/to/HybridGuard/demo/app.py .
   cp ~/path/to/HybridGuard/demo/requirements.txt .
   cp ~/path/to/HybridGuard/demo/README.md .
   git add -A
   git commit -m "Initial deploy: HybridGuard canonicalization demo"
   git push
   ```
6. Wait ~2 minutes for the Space to build (HF will install gradio + hybridguard from your GitHub repo).
7. Visit `https://huggingface.co/spaces/ProfSK/hybridguard-canonicalize` — your demo is live.

## Test locally before deploying

```bash
cd demo
pip install -r requirements.txt
python app.py
# Browser opens at http://127.0.0.1:7860
```

