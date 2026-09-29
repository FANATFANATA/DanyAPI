#!/usr/bin/env sh
set -e
REPO_URL="https://github.com/FANATFANATA/DanyAPI"
BRANCH="prod"
TARGET="${DANYAPI_DIR:-$HOME/DanyAPI}"
ZIP_URL="$REPO_URL/archive/refs/heads/$BRANCH.zip"
VENV_DIR="$TARGET/.venv"
ENV_BACKUP=""

cleanup() {
    restore_env
}

save_env() {
    ENV_BACKUP=""
    if [ -f "$TARGET/.env" ]; then
        ENV_BACKUP="$(dirname "$TARGET")/$(basename "$TARGET").env.danyapi-backup"
        if ! cp "$TARGET/.env" "$ENV_BACKUP"; then
            echo "Could not back up $TARGET/.env, aborting." >&2
            exit 1
        fi
        chmod 600 "$ENV_BACKUP" 2>/dev/null || true
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

# Refuse to delete a target that is not empty and does not look like a DanyAPI
# install, so a mistyped DANYAPI_DIR cannot destroy an unrelated directory.
target_is_removable() {
    if [ ! -e "$TARGET" ]; then
        return 0
    fi
    if [ -d "$TARGET/.git" ] || [ -f "$TARGET/app.py" ] || [ -f "$TARGET/docs/setup.py" ]; then
        return 0
    fi
    if [ -z "$(ls -A "$TARGET" 2>/dev/null)" ]; then
        return 0
    fi
    return 1
}

remove_target() {
    if target_is_removable; then
        rm -rf "$TARGET"
    else
        echo "$TARGET is not empty and does not look like a DanyAPI install, refusing to delete it." >&2
        exit 1
    fi
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
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/danyapi-download.XXXXXX")"
    echo "Downloading $ZIP_URL"
    echo "No checksum is published for this archive, so it is applied unverified."
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
    if [ ! -f "$tmp/DanyAPI-$BRANCH/app.py" ] || [ ! -d "$tmp/DanyAPI-$BRANCH/danyapi" ]; then
        echo "The downloaded archive does not contain a DanyAPI checkout, aborting." >&2
        rm -rf "$tmp"
        exit 1
    fi
    remove_target
    mkdir -p "$(dirname "$TARGET")"
    mv "$tmp/DanyAPI-$BRANCH" "$TARGET"
    rm -rf "$tmp"
}

ensure_venv() {
    if [ -x "$VENV_DIR/bin/python" ]; then
        return 0
    fi
    echo "Creating a virtualenv in $VENV_DIR"
    if ! "$PY" -m venv "$VENV_DIR"; then
        echo "Could not create a virtualenv. On Debian and Ubuntu install it with: sudo apt install python3-venv" >&2
        exit 1
    fi
}

echo "DanyAPI will be installed into: $TARGET"

save_env

if [ -d "$TARGET/.git" ]; then
    echo "Updating existing checkout..."
    (cd "$TARGET" && git pull --ff-only)
elif command -v git >/dev/null 2>&1; then
    remove_target
    echo "Cloning $REPO_URL ..."
    if git clone "$REPO_URL" "$TARGET"; then
        :
    else
        echo "git clone failed, trying the source archive."
        remove_target
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

ensure_venv

"$VENV_DIR/bin/python" "$TARGET/docs/setup.py"
