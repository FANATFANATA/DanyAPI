#!/usr/bin/env sh
set -e
cd "$(dirname "$0")"
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "Python 3.10+ is required but was not found in PATH."
    exit 1
fi
"$PY" setup.py
