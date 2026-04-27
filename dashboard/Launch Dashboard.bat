@echo off
REM HybridGuard Dashboard Launcher -- Windows
REM Double-click this file to start the dashboard in your default browser.
REM
REM What this does:
REM   1. Finds Python 3.10 or newer on your machine.
REM   2. Creates a small private environment in %USERPROFILE%\.hybridguard-dashboard-venv
REM      (outside OneDrive so it doesn't sync to the cloud).
REM   3. Installs dashboard dependencies into that environment (first run only).
REM   4. Starts the local server (http://127.0.0.1:8050) and opens your browser.
REM
REM To stop the dashboard, close this window or press Ctrl+C.

setlocal
cd /d "%~dp0"

echo ============================================================
echo   HybridGuard Dashboard -- Windows launcher
echo ============================================================

REM -- 1. Find a usable Python interpreter --------------------------------
set "PYTHON="
where py >nul 2>nul && set "PYTHON=py -3"
if not defined PYTHON (
    where python >nul 2>nul && set "PYTHON=python"
)
if not defined PYTHON (
    where python3 >nul 2>nul && set "PYTHON=python3"
)

if not defined PYTHON (
    echo.
    echo ERROR: Python is not installed on this machine.
    echo.
    echo Install Python 3.10 or newer from:
    echo   https://www.python.org/downloads/windows/
    echo.
    echo During installation, check the box "Add Python to PATH".
    echo Then double-click this file again.
    echo.
    pause
    exit /b 1
)

%PYTHON% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Found Python but it is older than 3.10.
    echo Please install Python 3.10+ from https://www.python.org/downloads/windows/
    echo.
    pause
    exit /b 1
)

echo Using Python:
%PYTHON% --version

REM -- 2. Create or reuse a private virtual environment -------------------
REM We put the venv OUTSIDE the OneDrive-synced folder so it doesn't waste
REM bandwidth or quota by syncing hundreds of MB of installed packages.
set "VENV=%USERPROFILE%\.hybridguard-dashboard-venv"
set "VENV_PY=%VENV%\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo.
    echo Creating a private dashboard environment ^(one-time, ~10 seconds^)...
    echo Location: %VENV%
    %PYTHON% -m venv "%VENV%"
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create the virtual environment.
        echo Try running this in a Command Prompt:
        echo   %PYTHON% -m pip install virtualenv
        echo   %PYTHON% -m virtualenv "%VENV%"
        echo.
        pause
        exit /b 1
    )
)

REM -- 3. Install dependencies if missing ---------------------------------
"%VENV_PY%" -c "import dash, dash_bootstrap_components, plotly, pandas, numpy" 2>nul
if errorlevel 1 (
    echo.
    echo Installing dashboard dependencies ^(first run only, ~30 seconds^)...
    "%VENV_PY%" -m pip install --upgrade --quiet pip
    "%VENV_PY%" -m pip install --quiet -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Could not install dependencies.
        echo To debug, open a Command Prompt and run:
        echo   "%VENV_PY%" -m pip install -r "%CD%\requirements.txt"
        echo.
        echo If install keeps failing, reset the environment by deleting it:
        echo   rmdir /S /Q "%VENV%"
        echo.
        pause
        exit /b 1
    )
    echo Dependencies installed.
)

REM -- 3b. Provision Chrome for kaleido (figure PDF/PNG export) -----------
REM Kaleido 1.x needs a private Chrome binary to render Plotly figures.
REM If Chrome provisioning fails, CSV and LaTeX downloads still work;
REM only PDF/PNG figure exports become unavailable.
"%VENV_PY%" -c "import kaleido" 2>nul
if not errorlevel 1 (
    REM Tiny render test: if it fails, we need Chrome.
    "%VENV_PY%" -c "import plotly.io as pio, plotly.graph_objects as go; pio.to_image(go.Figure(go.Scatter(x=[0], y=[0])), format='png', width=10, height=10)" 2>nul
    if errorlevel 1 (
        echo.
        echo Provisioning Chrome for figure exports ^(one-time, ~20 seconds^)...
        "%VENV%\Scripts\plotly_get_chrome.exe" -y >nul 2>&1
        if errorlevel 1 (
            echo   ^(Chrome provisioning failed -- CSV/LaTeX downloads will still work;
            echo    PDF/PNG figure exports will be disabled until you run plotly_get_chrome -y manually.^)
        )
    )
)

REM -- 4. Start the server, open the browser after a short delay ----------
start "" /B cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8050"

echo.
echo Starting dashboard at http://127.0.0.1:8050
echo Close this window or press Ctrl+C to stop.
echo ============================================================
echo.

"%VENV_PY%" dashboard.py

echo.
echo ============================================================
echo Dashboard stopped.
pause
