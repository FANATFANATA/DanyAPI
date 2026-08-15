import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "FANATFANATA/DanyAPI"
API = f"https://api.github.com/repos/{REPO}/releases/latest"
TAG_FILE = ROOT / ".installed-release"
ENV_FILE = ROOT / ".env"


def env_flag(name, default):
    value = os.environ.get(name)
    if value is None and ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(name + "="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def api_latest_tag():
    req = urllib.request.Request(API, headers={"User-Agent": "DanyAPI", "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("tag_name")
    except Exception:
        return None


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


def git_update(tag):
    for cmd in (
        ["git", "fetch", "origin", "--tags", "--force"],
        ["git", "checkout", "-f", tag],
        ["git", "reset", "--hard", tag],
    ):
        if run(cmd) != 0:
            return False
    return True


def zip_update(tag):
    parent = ROOT.parent
    tmp_zip = parent / "danyapi-update.zip"
    tmp_dir = parent / ("DanyAPI-" + tag)
    old_dir = parent / ".DanyAPI.old"
    url = f"https://github.com/{REPO}/archive/refs/tags/{tag}.zip"
    print(f"DanyAPI: downloading {url}")
    try:
        urllib.request.urlretrieve(url, str(tmp_zip))
        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(str(parent))
    except Exception as exc:
        print(f"DanyAPI: update download failed: {exc}")
        tmp_zip.unlink(missing_ok=True)
        return False
    if not tmp_dir.exists():
        print("DanyAPI: update archive layout unexpected, aborting")
        tmp_zip.unlink(missing_ok=True)
        return False
    if old_dir.exists():
        shutil.rmtree(str(old_dir), ignore_errors=True)
    shutil.move(str(ROOT), str(old_dir))
    shutil.move(str(tmp_dir), str(ROOT))
    tmp_zip.unlink(missing_ok=True)
    shutil.rmtree(str(old_dir), ignore_errors=True)
    return True


def update_to(tag):
    print(f"DanyAPI: updating to {tag} ...")
    if (ROOT / ".git").exists():
        if not git_update(tag):
            print("DanyAPI: git update failed, aborting update.")
            return
    else:
        if not zip_update(tag):
            return
    if run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]) != 0:
        print("DanyAPI: pip install failed, aborting update.")
        return
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
        update_to(latest)
    else:
        print(f"DanyAPI: already at {latest}")
    return run([sys.executable, "-m", "danyapi"])


if __name__ == "__main__":
    sys.exit(main())
