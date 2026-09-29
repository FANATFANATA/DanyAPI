#!/usr/bin/env sh
set -e
cd "$(dirname "$0")"
VENV_PY="$(cd .. && pwd)/.venv/bin/python"
if [ -x "$VENV_PY" ]; then
    PY="$VENV_PY"
else
    PY=""
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 \
            && "$candidate" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; then
            PY="$candidate"
            break
        fi
    done
fi
if [ -z "$PY" ]; then
    echo "Python 3.10+ is required but was not found in PATH."
    echo "Re-run docs/install.sh, it creates a virtualenv in .venv and uses that."
    exit 1
fi
echo "Using interpreter: $PY"
echo
"$PY" setup.py
