#!/bin/bash
# HybridGuard Dashboard Launcher — macOS
# Double-click this file to start the dashboard in your default browser.
#
# What this does:
#   1. Finds Python 3.10 or newer on your machine.
#   2. Creates a small private environment in ~/.hybridguard-dashboard-venv
#      (outside OneDrive so it doesn't sync to the cloud).
#   3. Installs dashboard dependencies into that environment (first run only).
#   4. Starts the local server (http://127.0.0.1:8050) and opens your browser.
#
# To stop the dashboard, close this Terminal window or press Ctrl+C.

# Always run from the directory containing this script.
cd "$(dirname "$0")" || exit 1

echo "============================================================"
echo "  HybridGuard Dashboard — macOS launcher"
echo "============================================================"

# ── 1. Find a usable Python interpreter ─────────────────────────────────────
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3.13 python3.14 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        if "$cmd" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    echo "ERROR: Python 3.10 or newer is not installed on this machine."
    echo ""
    echo "Install it from one of these sources, then double-click this file again:"
    echo "  • https://www.python.org/downloads/macos/  (recommended)"
    echo "  • Homebrew:  brew install python@3.11"
    echo ""
    echo "Press any key to close this window..."
    read -n 1
    exit 1
fi

echo "Using $PYTHON ($("$PYTHON" --version 2>&1))"

# ── 2. Create or reuse a private virtual environment ────────────────────────
# We put the venv OUTSIDE the OneDrive-synced folder so it doesn't waste
# bandwidth or quota by syncing hundreds of MB of installed packages.
VENV="$HOME/.hybridguard-dashboard-venv"
VENV_PY="$VENV/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo ""
    echo "Creating a private dashboard environment (one-time, ~10 seconds)..."
    echo "Location: $VENV"
    if ! "$PYTHON" -m venv "$VENV"; then
        echo ""
        echo "ERROR: Could not create the virtual environment."
        echo "If your Python is missing the 'venv' module, install it with:"
        echo "  $PYTHON -m pip install virtualenv"
        echo "  $PYTHON -m virtualenv \"$VENV\""
        echo ""
        echo "Press any key to close this window..."
        read -n 1
        exit 1
    fi
fi

# ── 3. Install dependencies if missing ──────────────────────────────────────
if ! "$VENV_PY" -c "import dash, dash_bootstrap_components, plotly, pandas, numpy" 2>/dev/null; then
    echo ""
    echo "Installing dashboard dependencies (first run only, ~30 seconds)..."
    "$VENV_PY" -m pip install --upgrade --quiet pip
    # No pipe to tail — we want pip's real exit code and any error message.
    if ! "$VENV_PY" -m pip install --quiet -r requirements.txt; then
        echo ""
        echo "ERROR: Could not install dependencies."
        echo "To debug, open Terminal and run:"
        echo "  \"$VENV_PY\" -m pip install -r \"$(pwd)/requirements.txt\""
        echo ""
        echo "If install keeps failing, you can reset the environment by deleting it:"
        echo "  rm -rf \"$VENV\""
        echo ""
        echo "Press any key to close this window..."
        read -n 1
        exit 1
    fi
    echo "Dependencies installed."
fi

# ── 3b. Provision Chrome for kaleido (figure PDF/PNG export) ─────────────────
# Kaleido 1.x needs a private Chrome binary to render Plotly figures.
# We run plotly_get_chrome silently — if it fails (no internet, behind a
# firewall, etc.) the dashboard still works for CSV / LaTeX downloads;
# only PDF/PNG figure exports become unavailable, with a helpful console
# message at dashboard startup.
"$VENV_PY" -c "import kaleido" 2>/dev/null && {
    PLOTLY_GET_CHROME="$VENV/bin/plotly_get_chrome"
    if [ -x "$PLOTLY_GET_CHROME" ]; then
        # Skip if Chrome already provisioned. Heuristic: try a tiny render.
        if ! "$VENV_PY" -c "import plotly.io as pio, plotly.graph_objects as go; pio.to_image(go.Figure(go.Scatter(x=[0], y=[0])), format='png', width=10, height=10)" 2>/dev/null; then
            echo ""
            echo "Provisioning Chrome for figure exports (one-time, ~20 seconds)..."
            "$PLOTLY_GET_CHROME" -y >/dev/null 2>&1 || \
                echo "  (Chrome provisioning failed — CSV/LaTeX downloads will still work; "
                echo "   PDF/PNG figure exports will be disabled until you run plotly_get_chrome -y manually.)"
        fi
    fi
}

# ── 4. Start the server, open the browser after a short delay ───────────────
( sleep 2 && open "http://127.0.0.1:8050" ) &

echo ""
echo "Starting dashboard at http://127.0.0.1:8050"
echo "Close this window or press Ctrl+C to stop."
echo "============================================================"
echo ""

"$VENV_PY" dashboard.py

# Keep the window open after dashboard.py exits so any error is readable.
echo ""
echo "============================================================"
echo "Dashboard stopped. Press any key to close this window..."
read -n 1
