@echo off
setlocal
cd /d "%~dp0"

if not exist .env (
    copy .env.example .env >nul
    echo No .env found, created one from .env.example.
    echo Fill in the tokens DEEPSEEK_TOKENS / QWEN_TOKENS and run this script again.
    exit /b 1
)

python -m danyapi
if errorlevel 1 (
    echo.
    echo Server failed to start. Install dependencies with: pip install -r requirements.txt
    pause
)
endlocal
