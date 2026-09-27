#!/usr/bin/env sh
set -e
REPO_URL="https://github.com/FANATFANATA/DanyAPI"
BRANCH="prod"
TARGET="${DANYAPI_DIR:-$HOME/DanyAPI}"
ZIP_URL="$REPO_URL/archive/refs/heads/$BRANCH.zip"
ENV_BACKUP=""

cleanup() {
    restore_env
}

save_env() {
    ENV_BACKUP=""
    if [ -f "$TARGET/.env" ]; then
        ENV_BACKUP="$TARGET.env.danyapi-backup"
        if ! cp "$TARGET/.env" "$ENV_BACKUP"; then
            echo "Could not back up $TARGET/.env, aborting." >&2
            exit 1
        fi
    fi
}

restore_env() {
    if [ -n "$ENV_BACKUP" ] && [ -f "$ENV_BACKUP" ]; then
        if [ -f "$TARGET/.env" ]; then
            rm -f "$ENV_BACKUP"
        elif mkdir -p "$TARGET" && cp "$ENV_BACKUP" "$TARGET/.env"; then
            echo "Restored your existing $TARGET/.env"
            rm -f "$ENV_BACKUP"
        else
            echo "Could not restore $TARGET/.env, your settings are still in $ENV_BACKUP" >&2
        fi
    fi
    ENV_BACKUP=""
}

trap cleanup EXIT

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

from_zip() {
    tmp="${TMPDIR:-/tmp}/danyapi-download"
    rm -rf "$tmp"
    mkdir -p "$tmp"
    echo "Downloading $ZIP_URL"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$ZIP_URL" -o "$tmp/repo.zip"
    else
        wget -q "$ZIP_URL" -O "$tmp/repo.zip"
    fi
    (
        cd "$tmp"
        if command -v unzip >/dev/null 2>&1; then
            unzip -q repo.zip
        else
            "$PY" -c "import zipfile; zipfile.ZipFile('repo.zip').extractall('.')"
        fi
    )
    rm -rf "$TARGET"
    mv "$tmp/DanyAPI-$BRANCH" "$TARGET"
    rm -rf "$tmp"
}

echo "DanyAPI will be installed into: $TARGET"

save_env

if [ -d "$TARGET/.git" ]; then
    echo "Updating existing checkout..."
    (cd "$TARGET" && git pull --ff-only)
elif command -v git >/dev/null 2>&1; then
    if [ -d "$TARGET" ]; then
        rm -rf "$TARGET"
    fi
    echo "Cloning $REPO_URL ..."
    if git clone "$REPO_URL" "$TARGET"; then
        :
    else
        echo "git clone failed, trying the source archive."
        rm -rf "$TARGET"
        from_zip
    fi
else
    echo "git not found, downloading the source archive instead."
    from_zip
fi

restore_env

if [ ! -f "$TARGET/docs/setup.py" ]; then
    echo "Could not find $TARGET/docs/setup.py in the checkout."
    exit 1
fi

"$PY" "$TARGET/docs/setup.py"
