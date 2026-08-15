@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if not errorlevel 1 goto run_python
where py >nul 2>nul
if not errorlevel 1 goto run_py
echo.
echo Python 3.13+ is required but was not found in PATH.
pause
exit /b 1
:run_python
python setup.py
goto check
:run_py
py setup.py
:check
if errorlevel 1 (
    echo.
    echo Setup failed. Make sure Python 3.13+ is installed and on PATH.
    pause
)
endlocal