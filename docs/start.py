from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "FANATFANATA/DanyAPI"
API = f"https://api.github.com/repos/{REPO}/releases/latest"
TAG_FILE = ROOT / ".installed-release"
TAG_CACHE = ROOT / ".latest-release-cache"
TAG_CACHE_TTL = 10 * 60
ENV_FILE = ROOT / ".env"
POW_SOLVER_DIR = ROOT / "danyapi" / "deepseek"
USER_FILES = (".env", str(POW_SOLVER_DIR / "pow_solver"), str(POW_SOLVER_DIR / "pow_solver.exe"))


def env_flag(name, default):
    value = os.environ.get(name)
    if value is None and ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            key, _, raw = line.partition("=")
            if key.strip() != name:
                continue
            value = raw.strip().strip('"').strip("'")
            break
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def cached_latest_tag():
    try:
        stat = TAG_CACHE.stat()
        if time.time() - stat.st_mtime > TAG_CACHE_TTL:
            return None, False
        tag = TAG_CACHE.read_text(encoding="utf-8").strip()
        return (tag or None), True
    except OSError:
        return None, False


def api_latest_tag():
    cached, fresh = cached_latest_tag()
    if fresh:
        return cached
    req = urllib.request.Request(API, headers={"User-Agent": "DanyAPI", "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tag = data.get("tag_name")
    except Exception:
        return None
    if tag:
        try:
            TAG_CACHE.write_text(tag, encoding="utf-8")
        except OSError:
            pass
    return tag


def git_local_tag():
    try:
        out = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], cwd=str(ROOT), stderr=subprocess.DEVNULL)
        tag = out.decode("utf-8", "replace").strip()
        return tag or None
    except Exception:
        return None


def file_local_tag():
    if TAG_FILE.exists():
        tag = TAG_FILE.read_text(encoding="utf-8").strip()
        return tag or None
    return None


def run(cmd, cwd=None):
    return subprocess.call(cmd, cwd=str(cwd or ROOT))


def git_branch():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL)
        branch = out.decode("utf-8", "replace").strip()
    except Exception:
        return None
    return branch or None


def git_dirty_paths():
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=str(ROOT), stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return [line[3:].strip() for line in out.decode("utf-8", "replace").splitlines() if line.strip()]


def git_remote_tag_sha(tag):
    try:
        out = subprocess.check_output(
            ["git", "ls-remote", "--tags", "origin", "refs/tags/" + tag, "refs/tags/" + tag + "^{}"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    peeled = None
    plain = None
    for line in out.decode("utf-8", "replace").splitlines():
        sha, _, ref = line.partition("\t")
        sha = sha.strip()
        if ref.strip().endswith("^{}"):
            peeled = sha
        elif ref.strip() == "refs/tags/" + tag:
            plain = sha
    return peeled or plain


def git_update(tag):
    dirty = git_dirty_paths()
    if dirty is None:
        print("DanyAPI: cannot read git status, aborting update.")
        return False
    if dirty:
        print("DanyAPI: refusing to update, the working tree has local changes:")
        for path in dirty[:20]:
            print(f"  {path}")
        if len(dirty) > 20:
            print(f"  ... and {len(dirty) - 20} more")
        print("Commit or stash them, or set DANYAPI_AUTO_UPDATE=0 to skip updates.")
        return False
    branch = git_branch()
    if run(["git", "fetch", "origin"]) != 0:
        return False
    expected = git_remote_tag_sha(tag)
    resolved = run(["git", "rev-parse", "--verify", "--quiet", "refs/tags/" + tag + "^{commit}"]) == 0
    if not resolved:
        if run(["git", "fetch", "--unshallow", "origin"]) != 0:
            print("DanyAPI: cannot complete the git history, aborting update.")
            return False
        if run(["git", "fetch", "origin"]) != 0:
            return False
        resolved = run(["git", "rev-parse", "--verify", "--quiet", "refs/tags/" + tag + "^{commit}"]) == 0
    if not resolved:
        print(f"DanyAPI: tag {tag} is not in the fetched history, aborting update.")
        return False
    if expected is not None:
        head = subprocess.run(
            ["git", "rev-parse", "refs/tags/" + tag + "^{commit}"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        local_sha = head.stdout.strip()
        if local_sha and local_sha != expected:
            print(f"DanyAPI: {tag} resolves to {local_sha[:12]} locally but origin says {expected[:12]}, aborting update.")
            return False
    if run(["git", "checkout", "-f", tag]) != 0:
        return False
    if run(["git", "reset", "--hard", tag]) != 0:
        return False
    if branch and branch != "HEAD" and run(["git", "checkout", "-f", branch]) != 0:
        return False
    return True


def _move_retry(src, dst, attempts=4):
    for i in range(attempts):
        try:
            shutil.move(src, dst)
            return True
        except OSError:
            if i == attempts - 1:
                return False
            time.sleep(0.6 * (i + 1))
    return False


TAG_RE = re.compile(r"^v\d+\.\d+\.\d+$")
ARCHIVE_TIMEOUT_SEC = 60
ARCHIVE_MAX_BYTES = 200 * 1024 * 1024


def zip_update(tag):
    if not TAG_RE.match(tag or ""):
        print(f"DanyAPI: refusing to download an unexpected tag name {tag!r}")
        return False
    parent = ROOT.parent
    url = f"https://github.com/{REPO}/archive/refs/tags/{tag}.zip"
    print(f"DanyAPI: downloading {url}")
    print("DanyAPI: no published checksum for this archive, the tag commit is verified against origin when the install is a git checkout.")
    try:
        staging = Path(tempfile.mkdtemp(prefix=".DanyAPI.staging-", dir=str(parent)))
    except OSError as exc:
        print(f"DanyAPI: cannot create a staging directory next to the install: {exc}")
        return False
    tmp_zip = staging / "update.zip"
    extract = staging / "extracted"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "DanyAPI"}), timeout=ARCHIVE_TIMEOUT_SEC) as resp:
            written = 0
            with open(tmp_zip, "wb") as handle:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > ARCHIVE_MAX_BYTES:
                        raise OSError(f"update archive is above the {ARCHIVE_MAX_BYTES} byte limit")
                    handle.write(chunk)
        with zipfile.ZipFile(tmp_zip) as zf:
            names = [n for n in zf.namelist() if n]
            roots = {Path(n).parts[0] for n in names}
            if len(roots) != 1:
                print("DanyAPI: update archive layout unexpected, aborting")
                _remove_staging(staging)
                return False
            if next(iter(roots)) != f"DanyAPI-{tag}":
                print(f"DanyAPI: update archive root {next(iter(roots))!r} does not match the tag, aborting")
                _remove_staging(staging)
                return False
            extract.mkdir(parents=True, exist_ok=True)
            zf.extractall(str(extract))
    except Exception as exc:
        print(f"DanyAPI: update download failed: {exc}")
        _remove_staging(staging)
        return False
    tmp_dir = extract / f"DanyAPI-{tag}"
    if not tmp_dir.is_dir() or not (tmp_dir / "app.py").is_file() or not (tmp_dir / "danyapi").is_dir():
        print("DanyAPI: update archive layout unexpected, aborting")
        _remove_staging(staging)
        return False
    old_root = Path(tempfile.mkdtemp(prefix=".DanyAPI.old-", dir=str(parent)))
    old_dir = old_root / "previous"
    try:
        os.chdir(str(parent))
    except OSError:
        pass
    tmp_dir_str = str(tmp_dir)
    root_str = str(ROOT)
    old_dir_str = str(old_dir)
    if not _move_retry(root_str, old_dir_str):
        merged = False
        for item in sorted(tmp_dir.rglob("*")):
            rel = item.relative_to(tmp_dir)
            target = ROOT / rel
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(item), str(target))
                merged = True
            except Exception as exc:
                print(f"DanyAPI: warning, could not replace {rel}: {exc}")
        if not merged:
            print("DanyAPI: could not replace installation: directory is locked")
            _remove_staging(staging)
            _remove_staging(old_root)
            return False
        print(f"DanyAPI: your previous install is kept at {old_dir}")
        _remove_staging(staging)
        return True
    try:
        if not _move_retry(tmp_dir_str, root_str):
            _move_retry(old_dir_str, root_str)
            _remove_staging(staging)
            _remove_staging(old_root)
            print("DanyAPI: could not replace installation: directory is locked")
            return False
    except Exception as exc:
        _move_retry(old_dir_str, root_str)
        _remove_staging(staging)
        _remove_staging(old_root)
        print(f"DanyAPI: could not replace installation: {exc}")
        return False
    for name in USER_FILES:
        if (old_dir / name).exists():
            try:
                dest = ROOT / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(old_dir / name), str(dest))
            except Exception as exc:
                print(f"DanyAPI: warning, could not restore {name}: {exc}")
    _remove_staging(staging)
    _remove_staging(old_root)
    return True


def _remove_staging(path):
    if path is None:
        return
    if not path.is_dir() or not path.name.startswith((".DanyAPI.staging-", ".DanyAPI.old-")):
        print(f"DanyAPI: refusing to remove {path}, it is not a directory this updater created")
        return
    shutil.rmtree(str(path), ignore_errors=True)


def git_head():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL)
    except Exception:
        return ""
    return out.decode("utf-8", "replace").strip()


def update_to(tag):
    print(f"DanyAPI: updating to {tag} ...")
    previous_head = ""
    if (ROOT / ".git").exists():
        previous_head = git_head()
        if not git_update(tag):
            print("DanyAPI: git update failed, aborting update.")
            return
    else:
        if not zip_update(tag):
            return
    if run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]) != 0:
        print("DanyAPI: pip install failed, aborting update.")
        return
    if previous_head:
        try:
            TAG_FILE.with_name(".previous-release").write_text(previous_head + "\n", encoding="utf-8")
            print(f"DanyAPI: the previous commit was {previous_head[:12]}, undo with git reset --hard {previous_head}")
        except OSError as exc:
            print(f"DanyAPI: could not record the previous commit: {exc}")
    TAG_FILE.write_text(tag, encoding="utf-8")
    print(f"DanyAPI: updated to {tag}")


def main():
    if not env_flag("DANYAPI_AUTO_UPDATE", True):
        return run([sys.executable, "-m", "danyapi"])
    latest = api_latest_tag()
    if not latest:
        print("DanyAPI: could not check for updates, starting anyway.")
        return run([sys.executable, "-m", "danyapi"])
    local = git_local_tag() or file_local_tag()
    if local != latest:
        try:
            update_to(latest)
        except Exception as exc:
            print(f"DanyAPI: update failed: {exc}")
    else:
        print(f"DanyAPI: already at {latest}")
    return run([sys.executable, "-m", "danyapi"])


if __name__ == "__main__":
    sys.exit(main())
