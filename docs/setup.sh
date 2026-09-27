#!/usr/bin/env sh
set -e
cd "$(dirname "$0")"
PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 \
        && "$candidate" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "Python 3.10+ is required but was not found in PATH."
    exit 1
fi
"$PY" setup.py
