#!/usr/bin/env sh
set -e
cd "$(dirname "$0")"

if [ ! -f .env ]; then
    cp .env.example .env
    echo "No .env found, created one from .env.example."
    echo "Fill in the tokens (DEEPSEEK_TOKENS / QWEN_TOKENS) and run this script again."
    exit 1
fi

python -m danyapi
