#!/usr/bin/env sh
set -e
REPO_URL="https://github.com/FANATFANATA/DanyAPI"
BRANCH="main"
TARGET="${DANYAPI_DIR:-$HOME/DanyAPI}"
ZIP_URL="$REPO_URL/archive/refs/heads/$BRANCH.zip"

if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
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
        elif command -v tar >/dev/null 2>&1; then
            tar -xzf repo.zip
        else
            echo "Neither unzip nor tar is available." >&2
            exit 1
        fi
    )
    rm -rf "$TARGET"
    mv "$tmp/DanyAPI-$BRANCH" "$TARGET"
    rm -rf "$tmp"
}

echo "DanyAPI will be installed into: $TARGET"

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

if [ ! -f "$TARGET/docs/setup.py" ]; then
    echo "Could not find $TARGET/docs/setup.py in the checkout."
    exit 1
fi

"$PY" "$TARGET/docs/setup.py"
