@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PY="

for %%V in (314 313 312 311 310) do (
    if not defined PY (
        set "CAND=python3.%%V"
        where !CAND! >nul 2>nul
        if not errorlevel 1 (
            !CAND! -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
            if not errorlevel 1 set "PY=!CAND!"
        )
    )
)

if not defined PY (
    set "CAND=python"
    where !CAND! >nul 2>nul
    if not errorlevel 1 (
        !CAND! -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
        if not errorlevel 1 set "PY=!CAND!"
    )
)

if not defined PY (
    set "CAND=py -3"
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)

if not defined PY (
    echo.
    echo Python 3.10+ is required but was not found in PATH.
    pause
    exit /b 1
)

%PY% setup.py
if errorlevel 1 (
    echo.
    echo Setup failed. Make sure Python 3.10+ is installed and on PATH.
    pause
)

endlocal
