@echo off
setlocal enabledelayedexpansion
title DanyAPI Token Utility
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
    echo [ERROR] Python 3.10+ not found. Install it from https://python.org
    echo         and make sure "Add Python to PATH" is checked.
    pause
    exit /b 1
)

echo Using interpreter: %PY%
echo.

%PY% token_utility.py %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [ERROR] Script exited with code %RC%.
    pause
    exit /b %RC%
)

endlocal
exit /b 0
