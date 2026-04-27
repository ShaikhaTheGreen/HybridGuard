# HybridGuard Dashboard

Interactive evaluation dashboard for the **HybridGuard** prompt-injection & jailbreak defence framework.

> **Refreshed 2026-04-25** — header now links to the live HF Space demo
> (`ProfSK/hybridguard-canonicalize`) and the public GitHub repo, and prominently labels the
> 5-seed evaluation set `{42, 2025, 7, 1337, 314}`.

> **Status:** This folder currently lives under `archive/`. To bring it back into active use,
> move it to the repo root: `git mv archive/HybridGuard_Dashboard dashboard/` (do this on Day 1
> of the housekeeping checklist).

> **Live demo (HF Space):** https://huggingface.co/spaces/ProfSK/hybridguard-canonicalize
> &mdash; consumes the same canonicalization library this dashboard visualizes.

---

## Quick Start

### Easiest way — double-click a launcher (for coauthors)

If you're a coauthor and just want to view the dashboard, you don't need to use the terminal.

| Your OS | Double-click this file |
|---|---|
| **macOS** | `Launch Dashboard.command` |
| **Windows** | `Launch Dashboard.bat` |

The launcher will:
1. Find Python 3.10+ on your machine.
2. Create a small private environment in your home folder (`~/.hybridguard-dashboard-venv` on Mac, `%USERPROFILE%\.hybridguard-dashboard-venv` on Windows). This is **outside** OneDrive on purpose, so it doesn't try to sync hundreds of MB of installed packages.
3. Install the dashboard's dependencies into that environment the first time (~30 seconds).
4. Start the local server at http://127.0.0.1:8050 and open the dashboard in your browser.

To stop the dashboard, close the Terminal/Command Prompt window or press **Ctrl+C** inside it.

**macOS first-launch note:** macOS may block the `.command` file the first time as an "unidentified developer". To unblock it: **right-click → Open** (instead of double-clicking), then click *Open* in the dialog. After that, double-click works normally.

**No Python installed?** The launcher will tell you and link to the installer. On Windows, be sure to check **"Add Python to PATH"** during install.

**Cleaning up:** If you ever want to delete the dashboard's environment (e.g. to free disk space or force a clean reinstall), just delete the folder above. The launcher will rebuild it the next time you run it.

---

### Manual way — terminal commands

If the launcher doesn't work or you prefer the command line:

```bash
cd dashboard
pip install -r requirements.txt
python dashboard.py
```

Then open **http://127.0.0.1:8050** in your browser.

### 3 – Generate static figures

```bash
python figures/generate_all_figures.py
```

All PNG files are written to `figures/`.

---

## VS Code

Open the `HybridGuard_Dashboard/` folder as the workspace root.
The `.vscode/launch.json` provides ready-to-use run configurations:

| Configuration | What it does |
|---|---|
| **Run Dashboard** | Starts the interactive Dash app |
| **Generate All Figures** | Batch-generates all publication PNGs |
| **Fig – ROC Curves** | Single figure script |
| **Fig – Calibration** | Reliability diagrams |
| **Fig – Ablation** | Ablation bar chart |
| **Fig – Sanitization** | Trade-off scatter |
| **Fig – Robustness** | Perturbation heat-map |

Use **F5** (or the Run panel) to launch any configuration.

---

## Live data vs Demo mode

The dashboard auto-detects whether a real HybridGuard notebook run exists:

- **Live mode** – reads CSVs from the notebook's `results/run_<timestamp>/tables/` directory.
  Point the loader at a custom path by editing `utils/load_results.py → Results(results_root=...)`.
- **Demo mode** – uses realistic synthetic data (shown when no run artefacts are found).

---

## Downloading figures and tables

Every chart and table on the dashboard has small `↓` buttons at the bottom-right.

| What | Buttons |
|---|---|
| **Any chart** | `↓ PDF` (Springer-accepted vector, ≈300 dpi) · `↓ PNG` (high-DPI raster) · `↓ CSV` (the chart's underlying data — open in Excel to redraw) |
| **Any table** | `↓ CSV` (open in Excel) · `↓ LaTeX` (booktabs, drop straight into your paper) |

For dropdown-driven charts (ROC, Calibration, Robustness, Overview metric), the download reflects the **currently selected** options — so you can change a model dropdown and download the version you're looking at.

PDF/PNG exports require `kaleido` + a private Chrome binary. The launcher provisions both automatically the first time it runs (~20 sec one-off). If your machine is offline or behind a firewall during that step, CSV and LaTeX downloads still work — only PDF/PNG buttons get disabled with a tooltip telling you what to install.

To enable PDF/PNG manually after the fact, run inside the dashboard's venv:

```bash
~/.hybridguard-dashboard-venv/bin/pip install kaleido
~/.hybridguard-dashboard-venv/bin/plotly_get_chrome -y
```

LaTeX output uses booktabs and includes a header comment reminding you to add `\usepackage{booktabs}` to your preamble. Special characters (`% & _ # $ { } \ ~ ^`) are escaped automatically.

---

## Dashboard Tabs

| Tab | Contents |
|---|---|
| 📊 **Overview** | KPI cards · full results table · AUROC/F1/Recall@1%FPR bar chart · Over-defense FPR |
| 🎯 **ROC & Calibration** | Interactive ROC curves · reliability diagrams (before/after temp-scaling) |
| 🔀 **Robustness** | Perturbation heat-map · jailbreak detection table |
| 🧹 **Sanitization** | Security-utility scatter · metrics table |
| 🔬 **Ablations** | Component ablation chart · fairness gap chart |
| ✅ **Logic Validation** | Documented logic issues, architecture summary, sanitisation semantics |

---

## Logic Validation Summary

| Severity | Count | Examples |
|---|---|---|
| ⚠️ Warning | 4 | Unused `_sigmoid`, unnormalised rule scores, `HG_OBJECTS` never set, approximate ablations |
| ℹ️ Info | 6 | `context_isolation` is formatting only, non-determinism may be silenced, `torch` not pinned |

All issues are documented in the **Logic Validation** tab with remediation notes.

---

## File Structure

```
HybridGuard_Dashboard/
├── dashboard.py              ← Main Dash app (entry point)
├── requirements.txt
├── README.md
├── .vscode/
│   ├── launch.json           ← VS Code run configs
│   └── settings.json
├── figures/
│   ├── fig_roc_curves.py
│   ├── fig_calibration.py
│   ├── fig_ablation.py
│   ├── fig_sanitization.py
│   ├── fig_robustness.py
│   └── generate_all_figures.py
├── utils/
│   ├── __init__.py
│   ├── synthetic_data.py     ← Realistic demo data
│   └── load_results.py       ← Live/demo data loader
└── assets/
    └── style.css
```
