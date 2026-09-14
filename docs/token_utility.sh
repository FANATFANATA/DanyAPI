#!/usr/bin/env sh
set -e
cd "$(dirname "$0")"

# Locate a usable Python 3 interpreter. command -v alone is not enough: some
# systems ship a python3 launcher that is broken or points at a dead install,
# so each candidate is actually executed before it is accepted.
PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import sys" >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "Python 3 is required but was not found in PATH."
    exit 1
fi

echo "Using interpreter: $PY"
echo

"$PY" token_utility.py "$@"
