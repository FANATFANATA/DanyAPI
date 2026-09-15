@echo off
setlocal
title DanyAPI Token Utility
cd /d "%~dp0"

rem Locate a Python interpreter (any version 3.6+, 32 or 64-bit - no deps needed)
set "PY="

where python >nul 2>nul && set "PY=python"
if not defined PY (
    where py >nul 2>nul && set "PY=py -3"
)
if not defined PY (
    if exist "%LocalAppData%\Programs\Python\Python310-32\python.exe" (
        set "PY=%LocalAppData%\Programs\Python\Python310-32\python.exe"
    )
)

if not defined PY (
    echo [ERROR] Python not found. Install Python 3.6+ from https://python.org
    echo         and make sure "Add Python to PATH" is checked.
    pause
    exit /b 1
)

echo Using interpreter: %PY%
echo.

%PY% token_utility.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] Script exited with an error.
    pause
    exit /b 1
)

endlocal
