@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PY="
set "VENV_PY=%~dp0..\.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=%VENV_PY%"
)

if not defined PY (
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
    echo Re-run docs\install.ps1, it creates a virtualenv in .venv and uses that.
    pause
    exit /b 1
)

echo Using interpreter: %PY%
echo.

%PY% setup.py
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo Setup failed with exit code %RC%. Make sure Python 3.10+ is installed and on PATH.
    pause
    exit /b %RC%
)

endlocal
exit /b 0
