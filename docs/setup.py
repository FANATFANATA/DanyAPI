import datetime
import getpass
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
EXAMPLE_FILE = ROOT / ".env.example"

DS_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
DS_HEADERS = {
    "x-client-bundle-id": "com.deepseek.chat",
    "x-client-platform": "web",
    "x-client-version": "2.3.0",
    "x-client-locale": "en-US",
    "x-client-timezone-offset": "0",
}

QWEN_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
QWEN_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://chat.qwen.ai",
    "Referer": "https://chat.qwen.ai/",
    "source": "web",
    "version": "0.2.83",
    "sec-ch-ua": '"Not=A?Brand";v="99", "Google Chrome";v="151", "Chromium";v="151"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

GROUPS = [
    (
        "Server",
        [
            ("DANYAPI_HOST", "Address the API server binds to", None),
            ("DANYAPI_PORT", "Port the API server listens on", "int"),
            ("DANYAPI_TIMEOUT", "Upstream request timeout in seconds", "int"),
            ("DANYAPI_ACQUIRE_TIMEOUT", "Seconds to wait for a free account (empty = forever)", "int_empty"),
        ],
    ),
    (
        "Sessions",
        [
            ("DANYAPI_SESSION_CACHE_SIZE", "Max server-side chats cached per provider", "int"),
            ("DANYAPI_SESSION_TTL_SECONDS", "Seconds an unused session stays reusable (0 = never)", "int"),
            ("DANYAPI_CACHE_DIR", "On-disk session cache directory (empty = system temp)", None),
            ("DANYAPI_CACHE_DISABLED", "Disable on-disk cache (1/true/yes/on)", None),
        ],
    ),
    (
        "Logging",
        [
            ("DANYAPI_LOG_LEVEL", "Log level (DEBUG/INFO/WARNING/ERROR)", "level"),
            ("DANYAPI_LOG_FILE", "Log file path (empty = console only)", None),
            ("DANYAPI_LOG_MAX_BYTES", "Max log file size in bytes before rotation", "int"),
            ("DANYAPI_LOG_BACKUP_COUNT", "Rotated log files to keep", "int"),
        ],
    ),
    (
        "Request behaviour",
        [
            ("DANYAPI_HUMAN_DELAY_MIN", "Minimum delay in seconds before sending a request", "float"),
            ("DANYAPI_HUMAN_DELAY_MAX", "Maximum delay in seconds (0/0 disables)", "float"),
            ("DANYAPI_AUTO_UPDATE", "Auto-update to the latest GitHub release on start (1/0)", "flag"),
        ],
    ),
]


def ask(question, default):
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        try:
            raw = input(f"{question}{suffix}: ").strip().lower()
        except EOFError:
            return default
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("    answer y or n")


def run_pip(req):
    cmd = [sys.executable, "-m", "pip", "install", "-r", req]
    print(f"Running: {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        print(f"pip install failed for {req} (exit {rc})")
        sys.exit(rc)


def prompt(key, label, kind, current, default=""):
    while True:
        try:
            if current:
                raw = input(f"  {key} - {label} [{current}]: ").strip()
            else:
                raw = input(f"  {key} - {label}: ").strip()
        except EOFError:
            return current
        if raw == "":
            return current
        if raw == "!clear":
            return ""
        if raw == "!reset":
            return default
        try:
            if kind == "int":
                int(raw)
            elif kind == "int_empty":
                if raw != "":
                    int(raw)
            elif kind == "float":
                float(raw)
            elif kind == "level":
                if raw not in ("DEBUG", "INFO", "WARNING", "ERROR"):
                    raise ValueError("must be DEBUG, INFO, WARNING or ERROR")
            elif kind == "flag":
                if raw not in ("0", "1"):
                    raise ValueError("must be 0 or 1")
            return raw
        except ValueError as e:
            print(f"    invalid: {e}")


def quote(value):
    if value == "":
        return ""
    if value != value.strip() or "#" in value or "\\" in value:
        if '"' in value:
            print("    warning: value contains a double quote, writing it raw")
            return value
        return f'"{value}"'
    return value


def parse_env(path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if len(val) >= 2 and val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        values[key] = val
    return values


def load_env():
    return parse_env(ENV_FILE)


def load_defaults():
    return parse_env(EXAMPLE_FILE)


def update_env(values):
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
    for key, value in values.items():
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
        found = False
        for i, line in enumerate(lines):
            if pattern.match(line):
                lines[i] = f"{key}={quote(value)}\n"
                found = True
                break
        if not found:
            lines.append(f"{key}={quote(value)}\n")
    fd, path = tempfile.mkstemp(dir=str(ROOT), suffix=".env.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.replace(path, ENV_FILE)
    except OSError:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def _request(url, headers, payload=None, timeout=25):
    data = None
    method = "POST" if payload is not None else "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers = dict(headers)
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace") if exc.fp else str(exc)
        return exc.code, body
    except Exception as exc:
        return None, str(exc)


def _ds_headers(token=None):
    headers = {
        "User-Agent": DS_UA,
        "Referer": "https://chat.deepseek.com/",
        "Origin": "https://chat.deepseek.com",
        "Accept": "*/*",
    }
    headers.update(DS_HEADERS)
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def _qwen_headers(token=None):
    now = datetime.datetime.now().astimezone()
    offset = now.strftime("%z")
    timezone = f"{now.strftime('%a %b %d %Y %H:%M:%S')} GMT{offset}"
    headers = {
        "User-Agent": QWEN_UA,
        "X-Request-Id": str(uuid.uuid4()),
        "Timezone": timezone,
    }
    headers.update(QWEN_HEADERS)
    if token:
        headers["Authorization"] = "Bearer " + token
        headers["Cookie"] = "token=" + token
    return headers


def check_deepseek_token(token):
    did = str(uuid.uuid4())
    url = f"https://chat.deepseek.com/api/v0/client/settings?did={did}&scope=main"
    status, body = _request(url, _ds_headers(token))
    if status is None:
        return False, f"network error: {body}"
    try:
        payload = json.loads(body)
        if payload.get("code") == 0:
            return True, ""
        return False, "server rejected the token"
    except ValueError:
        return False, f"unexpected response: {body[:200]}"


def check_deepseek_login(email, password):
    payload = {
        "email": email,
        "mobile": None,
        "password": password,
        "area_code": "",
        "device_id": str(uuid.uuid4()),
        "os": "web",
    }
    status, body = _request("https://chat.deepseek.com/api/v0/users/login", _ds_headers(), payload)
    if status is None:
        return False, f"network error: {body}"
    try:
        resp = json.loads(body)
        if resp.get("code"):
            return False, f"code {resp.get('code')}: {resp.get('msg') or resp.get('message') or ''}"
        data = resp.get("data") or {}
        if data.get("biz_code"):
            return False, f"biz {data.get('biz_code')}: {data.get('biz_msg', '')}"
        user = (data.get("biz_data") or {}).get("user") or {}
        return bool(user.get("token")), "login failed: no token in response"
    except ValueError:
        return False, f"unexpected response: {body[:200]}"


def check_qwen_token(token):
    status, body = _request("https://chat.qwen.ai/api/v1/auths/", _qwen_headers(token))
    if status != 200:
        return False, f"http {status}: {body[:200]}"
    try:
        payload = json.loads(body)
        if payload.get("success") is True:
            return True, ""
        return False, "server rejected the token"
    except ValueError:
        return False, f"unexpected response: {body[:200]}"


def check_qwen_login(email, password):
    payload = {"email": email, "password": hashlib.sha256(password.encode("utf-8")).hexdigest()}
    status, body = _request("https://chat.qwen.ai/api/v2/auths/signin", _qwen_headers(), payload)
    if status is None:
        return False, f"network error: {body}"
    try:
        resp = json.loads(body)
        if resp.get("success") is not True:
            data = resp.get("data")
            if isinstance(data, dict) and data.get("code"):
                return False, f"code {data.get('code')}: {data.get('details') or data.get('message') or ''}"
            return False, "login failed"
        biz = resp.get("data") or {}
        token = biz.get("token") or (biz.get("user") or {}).get("token")
        return bool(token), "login failed: no token in response"
    except ValueError:
        return False, f"unexpected response: {body[:200]}"


def split_tokens(raw):
    return [t.strip() for t in re.split(r"[, ]+", raw) if t.strip()]


def read_value(message, current, default=""):
    try:
        raw = input(message).strip()
    except EOFError:
        return current
    if raw == "":
        return current
    if raw == "!clear":
        return ""
    if raw == "!reset":
        return default
    return raw


def collect_provider(name, current, defaults):
    upper = name.upper()
    tokens_key = upper + "_TOKENS"
    single_key = upper + "_TOKEN"
    email_key = upper + "_EMAIL"
    password_key = upper + "_PASSWORD"
    host = "chat.deepseek.com" if name == "DeepSeek" else "chat.qwen.ai"
    storage = "userToken" if name == "DeepSeek" else "token"
    print()
    print(f"[ {name} ]")
    print(f"  Grab a token: open {host} -> DevTools -> Application -> Local Storage -> {storage}")
    tokens = read_value(
        f"  {name} tokens, comma-separated [{current.get(tokens_key, '') or '(empty)'}]: ",
        current.get(tokens_key, ""),
        defaults.get(tokens_key, ""),
    )
    single = read_value(
        f"  {name} single token, fallback [{current.get(single_key, '') or '(empty)'}]: ",
        current.get(single_key, ""),
        defaults.get(single_key, ""),
    )
    email = read_value(
        f"  {name} email, login fallback [{current.get(email_key, '') or '(empty)'}]: ",
        current.get(email_key, ""),
        defaults.get(email_key, ""),
    )
    if current.get(password_key, ""):
        print(f"  {name} password (current set, Enter keeps, !clear erases):")
    else:
        print(f"  {name} password (empty keeps unset):")
    password = getpass.getpass("    ").strip()
    if password == "":
        password = current.get(password_key, "")
    elif password == "!clear":
        password = ""
    return {
        tokens_key: tokens,
        single_key: single,
        email_key: email,
        password_key: password,
    }


def check_provider(name, creds):
    upper = name.upper()
    tokens = split_tokens(creds.get(upper + "_TOKENS", ""))
    if not tokens:
        single = creds.get(upper + "_TOKEN", "").strip()
        if single:
            tokens = [single]
    if tokens:
        if name == "DeepSeek":
            for token in tokens:
                ok, detail = check_deepseek_token(token)
                if not ok:
                    return False, f"token invalid: {detail}"
            return True, ""
        for token in tokens:
            ok, detail = check_qwen_token(token)
            if not ok:
                return False, f"token invalid: {detail}"
        return True, ""
    email = creds.get(upper + "_EMAIL", "").strip()
    password = creds.get(upper + "_PASSWORD", "")
    if email and password:
        if name == "DeepSeek":
            return check_deepseek_login(email, password)
        return check_qwen_login(email, password)
    return True, ""


def validate_provider(name, creds, defaults):
    while True:
        ok, detail = check_provider(name, creds)
        if ok:
            print(f"  {name} credentials OK.")
            return creds
        print(f"  {name} credentials INVALID: {detail}")
        if not ask(f"  Re-enter {name} credentials?", True):
            print(f"  Keeping {name} credentials as entered; the server may fail at startup.")
            return creds
        creds = collect_provider(name, creds, defaults)


def _desktop_dir():
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        desktop = os.path.expanduser("~/Desktop")
        if not os.path.isdir(desktop):
            os.makedirs(desktop, exist_ok=True)
        return desktop
    return os.path.join(os.path.expanduser("~"), "Desktop")


def _ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def create_shortcut():
    py = sys.executable
    root = str(ROOT)
    launcher = str(ROOT / "docs" / "start.py")
    if sys.platform.startswith("win"):
        script = "\n".join(
            [
                "$d=[Environment]::GetFolderPath('Desktop')",
                "$ws=New-Object -ComObject WScript.Shell",
                "$sc=$ws.CreateShortcut((Join-Path $d 'DanyAPI.lnk'))",
                "$sc.TargetPath=" + _ps_quote(py),
                "$sc.Arguments=" + _ps_quote(launcher),
                "$sc.WorkingDirectory=" + _ps_quote(root),
                "$sc.IconLocation=" + _ps_quote(py + ",0"),
                "$sc.Description='Start the DanyAPI server'",
                "$sc.Save()",
            ]
        )
        subprocess.check_call(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])
        return os.path.join(_desktop_dir(), "DanyAPI.lnk")
    if sys.platform.startswith("linux"):
        desktop = _desktop_dir()
        content = f"[Desktop Entry]\nType=Application\nName=DanyAPI\nComment=Start the DanyAPI server\nExec={py} {launcher}\nPath={root}\nTerminal=true\n"
        path = os.path.join(desktop, "DanyAPI.desktop")
        Path(path).write_text(content, encoding="utf-8")
        os.chmod(path, 0o755)
        return path
    desktop = _desktop_dir()
    content = f"#!/bin/zsh\ncd {shlex.quote(root)}\nexec {shlex.quote(py)} {shlex.quote(launcher)}\n"
    path = os.path.join(desktop, "DanyAPI.command")
    Path(path).write_text(content, encoding="utf-8")
    os.chmod(path, 0o755)
    return path


def build_pow_solver():
    src = ROOT / "danyapi" / "deepseek" / "pow_solver.c"
    out = ROOT / "danyapi" / "deepseek" / ("pow_solver.exe" if os.name == "nt" else "pow_solver")
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        print("No C compiler (clang/gcc/cc) found; skipping the native PoW solver build.")
        print("The Python and Node fallbacks will still solve challenges, just slower.")
        return
    cc_name = os.path.basename(cc)
    if sys.platform == "darwin":
        base = ["-O3", "-funroll-loops", "-flto", "-fomit-frame-pointer"]
    elif os.name == "nt":
        base = ["-O3", "-funroll-loops", "-flto", "-fomit-frame-pointer", "-march=native", "-mtune=native"]
        if "clang" in cc_name:
            base.append("-fuse-ld=lld")
    else:
        base = ["-O3", "-pthread", "-funroll-loops", "-flto", "-fomit-frame-pointer", "-march=native", "-mtune=native"]
    variants = [base]
    if cc_name in ("gcc", "cc") or "gcc" in cc_name:
        variants.insert(0, [*base, "-ffast-math"])
    for i, flags in enumerate(variants):
        cmd = [cc, *flags, "-o", str(out), str(src)]
        print(f"Building native PoW solver ({'gcc tuned' if i == 0 and len(variants) > 1 else 'base'}): {' '.join(cmd)}")
        if subprocess.call(cmd, cwd=str(ROOT)) == 0:
            print(f"Native PoW solver built: {out}")
            return
    flags = [f for f in base if f not in ("-march=native", "-mtune=native", "-flto", "-ffast-math")]
    cmd = [cc, *flags, "-o", str(out), str(src)]
    print(f"Falling back to portable build: {' '.join(cmd)}")
    if subprocess.call(cmd, cwd=str(ROOT)) == 0:
        print(f"Native PoW solver built: {out}")
    else:
        print("Native PoW solver build failed; Python/Node fallbacks still work.")


def main():
    print("DanyAPI setup")
    print("==============")
    if not ENV_FILE.exists():
        if EXAMPLE_FILE.exists():
            shutil.copyfile(EXAMPLE_FILE, ENV_FILE)
            print("Created .env from .env.example.")
        else:
            ENV_FILE.write_text("", encoding="utf-8")
            print("Created an empty .env.")
    else:
        print("Found existing .env, keeping it.")
    if ask("Install dependencies now (pip install -r requirements.txt)?", True):
        run_pip("requirements.txt")
    if ask("Install development dependencies (tests + linting)?", False):
        run_pip("requirements-dev.txt")

    build_pow_solver()

    current = load_env()
    defaults = load_defaults()
    values = dict(current)
    deepseek = validate_provider("DeepSeek", collect_provider("DeepSeek", current, defaults), defaults)
    qwen = validate_provider("Qwen", collect_provider("Qwen", current, defaults), defaults)
    values.update(deepseek)
    values.update(qwen)

    print()
    print("Now the rest of the settings. Enter to keep the current value, !clear to erase, !reset to restore the default.")
    for title, fields in GROUPS:
        print()
        print(f"[ {title} ]")
        for key, label, kind in fields:
            values[key] = prompt(key, label, kind, values.get(key, ""), defaults.get(key, ""))

    update_env(values)
    print()
    print("Configuration saved to .env")

    has_ds = any(v for v in deepseek.values() if v)
    has_qwen = any(v for v in qwen.values() if v)
    if not has_ds and not has_qwen:
        print("Warning: no provider credentials configured. The server will not start until you add DeepSeek or Qwen tokens.")

    if ask("Create a DanyAPI launcher shortcut on the desktop?", True):
        try:
            path = create_shortcut()
            print(f"Shortcut created: {path}")
        except Exception as exc:
            print(f"Shortcut creation failed: {exc}")

    print()
    print(f"DanyAPI runs from: {ROOT}")
    print("The server auto-updates to the latest GitHub release at every launch (DANYAPI_AUTO_UPDATE=0 disables).")
    print("Start it anytime with the desktop shortcut or:")
    print("  run.bat   (Windows)")
    print("  ./run.sh  (Linux/macOS)")
    print("  python -m danyapi")


if __name__ == "__main__":
    main()
