@echo off
setlocal
cd /d "%~dp0"
python setup.py
if errorlevel 1 (
    echo.
    echo Setup failed. Make sure Python 3.13+ is installed and on PATH.
    pause
)
endlocal
