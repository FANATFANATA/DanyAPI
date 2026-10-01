from __future__ import annotations

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
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
EXAMPLE_FILE = ROOT / ".env.example"
ANDROID_MARKER = Path("/system/build.prop")

TRUE_WORDS = ("1", "true", "yes", "on")
FALSE_WORDS = ("0", "false", "no", "off")
GIGACHAT_SCOPES = ("GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP")

GROUPS = [
    (
        "Server",
        [
            ("DANYAPI_HOST", "Address the API server binds to", None),
            ("DANYAPI_PORT", "Port the API server listens on", "int"),
            ("DANYAPI_TIMEOUT", "Upstream request timeout in seconds", "float"),
            (
                "DANYAPI_ACQUIRE_TIMEOUT",
                "Seconds to wait for a free account (empty = forever)",
                "float_opt",
            ),
            (
                "DANYAPI_CORS_ORIGINS",
                "Extra browser origins allowed to call the API, comma separated (empty = any)",
                None,
            ),
            ("DANYAPI_ADMIN_TOKEN", "Bearer token that enables POST /v1/tokens (empty keeps it off)", "secret"),
        ],
    ),
    (
        "GigaChat",
        [
            (
                "GIGACHAT_SCOPE",
                "GigaChat scope: " + ", ".join(GIGACHAT_SCOPES),
                "scope",
            ),
            (
                "DANYAPI_GIGACHAT_CA_FILE",
                "Path to a CA bundle for GigaChat (empty = bundled Russian root CA)",
                None,
            ),
        ],
    ),
    (
        "Unofficial providers",
        [
            ("ALICE_ACCOUNTS", "Concurrent Alice connections to keep open", "int"),
            ("DUCKAI_ACCOUNTS", "Concurrent Duck.ai connections to keep open", "int"),
        ],
    ),
    (
        "Public instance",
        [
            (
                "BYOK_MODE",
                "Bring your own key mode, every caller sends its own tokens (1/0)",
                "flag",
            ),
            (
                "DANYAPI_BYOK_MODE",
                "Alias of BYOK_MODE, used only when BYOK_MODE is empty (1/0)",
                "flag",
            ),
            (
                "DANYAPI_BYOK_AUTH_TTL_SECONDS",
                "Seconds a bring your own key token check is reused",
                "float",
            ),
            (
                "DANYAPI_MODELS_REFRESH_SECONDS",
                "Seconds between provider model list refreshes (0 = off)",
                "float",
            ),
        ],
    ),
    (
        "Sessions",
        [
            (
                "DANYAPI_SESSION_CACHE_SIZE",
                "Max server-side chats cached per provider",
                "int",
            ),
            (
                "DANYAPI_SESSION_TTL_SECONDS",
                "Seconds an unused session stays reusable (0 = never)",
                "float",
            ),
            (
                "DANYAPI_CACHE_DIR",
                "On-disk session cache directory (empty = system temp)",
                None,
            ),
            ("DANYAPI_CACHE_DISABLED", "Disable on-disk cache (1/true/yes/on)", None),
            ("DANYAPI_RESPONSES_MAX_RECORDS", "Recent /v1/responses records kept", "int"),
        ],
    ),
    (
        "Usage",
        [
            (
                "DANYAPI_USAGE_ENABLED",
                "Enable usage tracking / token counter stats (1/0)",
                "onoff",
            ),
            (
                "DANYAPI_USAGE_MAX_RECORDS",
                "Max recent usage records kept for /v1/usage",
                "int",
            ),
        ],
    ),
    (
        "Logging",
        [
            ("DANYAPI_LOG_LEVEL", "Log level (DEBUG/INFO/WARNING/ERROR)", "level"),
            ("DANYAPI_LOG_FILE", "Log file path (empty = console only)", None),
            (
                "DANYAPI_LOG_MAX_BYTES",
                "Max log file size in bytes before rotation",
                "int",
            ),
            ("DANYAPI_LOG_BACKUP_COUNT", "Rotated log files to keep", "int"),
        ],
    ),
    (
        "Request behaviour",
        [
            (
                "DANYAPI_AUTO_UPDATE",
                "Auto-update to the latest GitHub release on start (1/0)",
                "onoff",
            ),
        ],
    ),
]


class _InputState:
    eof_seen = False


_input_state = _InputState()


def _read_input(prompt):
    try:
        return input(prompt)
    except EOFError:
        _input_state.eof_seen = True
        return None


def ask(question, default):
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        raw = _read_input(f"{question}{suffix}: ")
        if raw is None:
            return default
        raw = raw.strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("    answer y or n")


@cache
def is_termux():
    return "com.termux" in sys.prefix or Path("/data/data/com.termux").exists()


@cache
def is_android():
    return ANDROID_MARKER.exists()


def ensure_rust_on_termux():
    if shutil.which("rustc"):
        return True
    pkg = shutil.which("pkg")
    if pkg is None:
        print("Termux detected but 'pkg' is missing; cannot install Rust automatically.")
        print("Run: pkg install -y rust binutils clang cmake")
        return False
    print("Rust not found. Installing Rust toolchain for Termux (needed to build pydantic-core)...")
    rc = subprocess.call([pkg, "install", "-y", "rust", "binutils", "clang", "cmake"])
    return rc == 0


def rustup_target_reachable():
    try:
        proc = subprocess.run(
            ["rustup", "target", "list", "--installed"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    lines = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    return "aarch64-unknown-linux-android" in lines


def run_pip(req):
    cmd = [sys.executable, "-m", "pip", "install", "-r", req]
    env = os.environ.copy()
    if is_termux():
        if not ensure_rust_on_termux():
            print(f"Rust unavailable; skipping {req} install.")
            return
        env["CRATE_CC_NO_DEFAULTS"] = "1"
        env["CARGO_BUILD_TARGET"] = "aarch64-linux-android"
    elif is_android() and not rustup_target_reachable():
        print("Rustup lacks the required target; relying on a system Rust toolchain.")
        print("Install it with: rustup target add aarch64-unknown-linux-android")
    print(f"Running: {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    if rc != 0:
        print(f"pip install failed for {req} (exit {rc})")
        sys.exit(rc)


def prompt(key, label, kind, current, default=""):
    while True:
        if current:
            shown = mask_secrets(current) if kind in ("tokens", "secret") else current
            raw = _read_input(f"  {key} - {label} [{shown}]: ")
        else:
            raw = _read_input(f"  {key} - {label}: ")
        if raw is None:
            return current
        raw = raw.strip()
        if raw == "":
            return current
        if raw == "!clear":
            return ""
        if raw == "!reset":
            return default
        try:
            if kind == "int":
                int(raw)
            elif kind == "float":
                float(raw)
            elif kind == "float_opt":
                if raw != "":
                    value = float(raw)
                    if not (value > 0):
                        raise ValueError("must be greater than 0 or left empty")
            elif kind == "level":
                if raw not in ("DEBUG", "INFO", "WARNING", "ERROR"):
                    raise ValueError("must be DEBUG, INFO, WARNING or ERROR")
            elif kind == "scope":
                if raw not in GIGACHAT_SCOPES:
                    raise ValueError("must be " + ", ".join(GIGACHAT_SCOPES))
            elif kind == "flag":
                if raw not in ("0", "1"):
                    raise ValueError("must be 0 or 1")
            elif kind == "onoff":
                if raw.lower() not in TRUE_WORDS + FALSE_WORDS:
                    raise ValueError("must be one of " + ", ".join(TRUE_WORDS + FALSE_WORDS))
            return raw
        except ValueError as e:
            print(f"    invalid: {e}")


def quote(value):
    if value == "":
        return ""
    if "\n" in value or "\r" in value:
        raise ValueError("value contains a line break and cannot be written to .env")
    if value != value.strip() or "#" in value or "\\" in value or "'" in value:
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    return value


_SINGLE_ESCAPES = {"'": "'", "\\": "\\"}
_DOUBLE_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}


def _unquote(val):
    if len(val) < 2:
        return val, False
    quote_char = val[0]
    if quote_char not in ("'", '"') or val[-1] != quote_char:
        return val, False
    table = _SINGLE_ESCAPES if quote_char == "'" else _DOUBLE_ESCAPES
    out = []
    index = 1
    end = len(val) - 1
    while index < end:
        char = val[index]
        if char == "\\" and index + 1 < end and val[index + 1] in table:
            out.append(table[val[index + 1]])
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out), True


INLINE_COMMENT_RE = re.compile(r"\s+#.*")


def parse_env(path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$", line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2).strip()
        val, quoted = _unquote(raw)
        if quoted:
            values[key] = val
        else:
            values[key] = INLINE_COMMENT_RE.sub("", raw).rstrip()
    return values


def load_env():
    return parse_env(ENV_FILE)


def load_defaults():
    return parse_env(EXAMPLE_FILE)


def update_env(values):
    for key, value in load_defaults().items():
        values.setdefault(key, value)
    raw = ENV_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
    lines = []
    seen = set()
    try:
        for line in raw:
            m = re.match(r"^\s*([A-Za-z0-9_]+)\s*=", line)
            key = m.group(1) if m else None
            if key is not None and key in values:
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"{key}={quote(values[key])}\n")
            else:
                lines.append(line)
        for key, value in values.items():
            if key not in seen:
                lines.append(f"{key}={quote(value)}\n")
    except ValueError as exc:
        raise SystemExit(f"Refusing to write .env: {exc}") from exc
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
    if os.name != "nt":
        os.chmod(ENV_FILE, 0o600)


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
    from danyapi.deepseek.client import CLIENT_HEADERS, USER_AGENT

    headers = {
        "User-Agent": USER_AGENT,
        "Referer": "https://chat.deepseek.com/",
        "Origin": "https://chat.deepseek.com",
        "Accept": "*/*",
    }
    headers.update(CLIENT_HEADERS)
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def _qwen_headers(token=None):
    from danyapi.qwen.client import COMMON_HEADERS, USER_AGENT, new_uuid, timezone_header

    headers = {
        "User-Agent": USER_AGENT,
        "X-Request-Id": new_uuid(),
        "Timezone": timezone_header(),
    }
    headers.update(COMMON_HEADERS)
    if token:
        headers["Authorization"] = "Bearer " + token
        headers["Cookie"] = "token=" + token
    return headers


def check_deepseek_token(token):
    from danyapi.deepseek.client import BASE_URL, new_device_id

    url = f"{BASE_URL}/api/v0/client/settings?did={new_device_id()}&scope=main"
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


def _gigachat_token_status(key):
    from danyapi.gigachat.client import AUTH_URL, DEFAULT_SCOPE, USER_AGENT
    from danyapi.gigachat.tls import resolve_ca

    context = resolve_ca()
    handler = urllib.request.HTTPSHandler(context=context)
    opener = urllib.request.build_opener(handler)
    req = urllib.request.Request(
        AUTH_URL,
        data=f"scope={DEFAULT_SCOPE}".encode(),
        headers={
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {key}",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with opener.open(req, timeout=25) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace") if exc.fp else str(exc)
        return exc.code, body
    except Exception as exc:
        return None, str(exc)


def check_gigachat_key(key):
    status, body = _gigachat_token_status(key)
    if status is None:
        return False, f"network error: {body}"
    if status != 200:
        try:
            message = json.loads(body).get("message", "")
        except ValueError:
            message = body[:200]
        return False, f"http {status}: {message}"[:200]
    try:
        payload = json.loads(body)
    except ValueError:
        return False, f"unexpected response: {body[:200]}"
    if payload.get("access_token"):
        return True, ""
    return False, "no access token in the response"


def check_qwen_token(token):
    from danyapi.qwen.client import BASE_URL

    status, body = _request(f"{BASE_URL}/api/v1/auths/", _qwen_headers(token))
    if status is None:
        return False, f"network error: {body}"
    if status != 200:
        return False, f"http {status}: {body[:200]}"
    try:
        payload = json.loads(body)
    except ValueError:
        return False, f"unexpected response: {body[:200]}"
    if payload.get("success") is True or payload.get("id"):
        return True, ""
    data = payload.get("data")
    if isinstance(data, dict) and data.get("id"):
        return True, ""
    return False, "server rejected the token"


def split_tokens(raw):
    return [t.strip() for t in re.split(r"[, ]+", raw) if t.strip()]


def mask_secrets(raw):
    values = split_tokens(raw)
    if not values:
        return "(empty)"
    shown = ", ".join(f"{value[:4]}... ({len(value)} chars)" for value in values[:2])
    if len(values) > 2:
        shown += f", +{len(values) - 2} more"
    return f"{shown} ({len(values)} set)"


def read_value(message, current, default=""):
    raw = _read_input(message)
    if raw is None:
        return current
    raw = raw.strip()
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
    host = "chat.deepseek.com" if name == "DeepSeek" else "chat.qwen.ai"
    storage = "userToken" if name == "DeepSeek" else "token"
    print()
    print(f"[ {name} ]")
    print("  Easiest: run docs/token_utility.sh (or .bat on Windows), it shows both tokens in your browser at /results,")
    print("  then copy them from there into the prompt below.")
    print(f"  Or grab a token by hand: open {host} -> DevTools -> Application -> Local Storage -> {storage}")
    tokens = read_value(
        f"  {name} tokens, comma-separated [{mask_secrets(current.get(tokens_key, ''))}]: ",
        current.get(tokens_key, ""),
        defaults.get(tokens_key, ""),
    )
    return {tokens_key: tokens}


def _guard_checker(checker):
    def run(token):
        try:
            return checker(token)
        except Exception as exc:
            return False, f"check failed: {type(exc).__name__}: {exc}"

    return run


def check_provider(name, creds):
    upper = name.upper()
    tokens = split_tokens(creds.get(upper + "_TOKENS", ""))
    if not tokens:
        return True, ""
    checker = _guard_checker(check_deepseek_token if name == "DeepSeek" else check_qwen_token)
    with ThreadPoolExecutor(max_workers=max(1, min(len(tokens), 8))) as pool:
        try:
            results = list(pool.map(checker, tokens))
        except Exception as exc:
            return False, f"credential check failed: {type(exc).__name__}: {exc}"
    for ok, detail in results:
        if not ok:
            return False, f"token invalid: {detail}"
    return True, ""


def collect_gigachat(current, defaults):
    print()
    print("[ GigaChat ]")
    print("  Official GigaChat API, free freemium quota for a new project.")
    print("  In the GigaChat Studio account open 'API settings' and generate an authorization key.")
    print("  The key is the base64 of client_id:client_secret, copy it as is.")
    keys = read_value(
        "  GigaChat authorization keys, comma-separated [" + mask_secrets(current.get("GIGACHAT_KEYS", "")) + "]: ",
        current.get("GIGACHAT_KEYS", ""),
        defaults.get("GIGACHAT_KEYS", ""),
    )
    return {"GIGACHAT_KEYS": keys}


def validate_gigachat(creds, defaults):
    while True:
        ok, detail = True, ""
        for key in split_tokens(creds.get("GIGACHAT_KEYS", "")):
            ok, detail = check_gigachat_key(key)
            if not ok:
                break
        if ok:
            print("  GigaChat credentials OK.")
            return creds
        print(f"  GigaChat credentials INVALID: {detail}")
        if _input_state.eof_seen or not ask("  Re-enter GigaChat credentials?", True):
            print("  Keeping GigaChat credentials as entered; the server may fail at startup.")
            return creds
        creds = collect_gigachat(creds, defaults)


def collect_opencode(current, defaults):
    print()
    print("[ OpenCode Zen ]")
    print("  The curated OpenCode model gateway at https://opencode.ai/zen, metered per token.")
    print("  Sign in at https://opencode.ai/auth and copy the API key. A few models are free")
    print("  for a limited time and answer without one, everything else needs the key.")
    keys = read_value(
        "  OpenCode Zen API keys, comma-separated [" + mask_secrets(current.get("OPENCODE_KEYS", "")) + "]: ",
        current.get("OPENCODE_KEYS", ""),
        defaults.get("OPENCODE_KEYS", ""),
    )
    values = {"OPENCODE_KEYS": keys}
    if not keys.strip():
        print("  No key given: the Zen free tier can still be served, without any key.")
        enabled = read_value(
            "  Serve the Zen free tier anyway? yes/no [" + (current.get("OPENCODE_ENABLED", "") or "no") + "]: ",
            current.get("OPENCODE_ENABLED", ""),
            defaults.get("OPENCODE_ENABLED", ""),
        )
        if enabled.strip().lower() in ("n", "no", "off", "0", "false", "нет"):
            enabled = ""
        elif enabled.strip():
            enabled = "1"
        values["OPENCODE_ENABLED"] = enabled
    else:
        values["OPENCODE_ENABLED"] = current.get("OPENCODE_ENABLED", "") or defaults.get("OPENCODE_ENABLED", "")
    return values


def collect_alice(current, defaults):
    print()
    print("[ Yandex Alice, unofficial ]")
    print("  Free and keyless, but it speaks an undocumented internal protocol of a consumer service.")
    print("  Yandex's terms forbid that, and the protocol can change without notice. See the README.")
    enabled = read_value(
        "  Enable the unofficial Alice provider? yes/no [" + (current.get("ALICE_ENABLED", "") or "no") + "]: ",
        current.get("ALICE_ENABLED", ""),
        defaults.get("ALICE_ENABLED", ""),
    )
    if enabled.strip().lower() in ("n", "no", "off", "0", "false", "нет"):
        enabled = ""
    elif enabled.strip():
        enabled = "1"
    return {"ALICE_ENABLED": enabled}


def collect_duckai(current, defaults):
    print()
    print("[ Duck.ai, unofficial ]")
    print("  Free and keyless, but it speaks an undocumented internal protocol of a consumer service")
    print("  and proves a browser fingerprint. DuckDuckGo's terms forbid that, and it refuses")
    print("  datacenter addresses outright. Needs Node.js on the host. See the README.")
    enabled = read_value(
        "  Enable the unofficial Duck.ai provider? yes/no [" + (current.get("DUCKAI_ENABLED", "") or "no") + "]: ",
        current.get("DUCKAI_ENABLED", ""),
        defaults.get("DUCKAI_ENABLED", ""),
    )
    if enabled.strip().lower() in ("n", "no", "off", "0", "false", "нет"):
        enabled = ""
    elif enabled.strip():
        enabled = "1"
    return {"DUCKAI_ENABLED": enabled}


def validate_provider(name, creds, defaults):
    while True:
        ok, detail = check_provider(name, creds)
        if ok:
            print(f"  {name} credentials OK.")
            return creds
        print(f"  {name} credentials INVALID: {detail}")
        if _input_state.eof_seen or not ask(f"  Re-enter {name} credentials?", True):
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


def write_private(target, data):
    fd, path = tempfile.mkstemp(dir=str(ROOT), suffix=".env.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(path, target)
    except BaseException:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    if os.name != "nt":
        os.chmod(target, 0o600)


def _ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def _windows_desktop_dir():
    script = "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; [Environment]::GetFolderPath('Desktop')"
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode == 0:
        path = (result.stdout or "").strip().splitlines()
        if path and path[-1].strip():
            return path[-1].strip()
    return os.path.join(os.path.expanduser("~"), "Desktop")


def create_shortcut():
    py = sys.executable
    root = str(ROOT)
    launcher = str(ROOT / "docs" / "start.py")
    if sys.platform.startswith("win"):
        desktop = _windows_desktop_dir()
        script = "\n".join(
            [
                "$d=" + _ps_quote(desktop),
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
        return os.path.join(desktop, "DanyAPI.lnk")
    if sys.platform.startswith("linux"):
        desktop = _desktop_dir()
        content = (
            "[Desktop Entry]\nType=Application\nName=DanyAPI\n"
            "Comment=Start the DanyAPI server\n"
            f"Exec={shlex.quote(py)} {shlex.quote(launcher)}\n"
            f"Path={shlex.quote(root)}\n"
            "Terminal=true\n"
        )
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
        base = [
            "-O3",
            "-funroll-loops",
            "-flto",
            "-fomit-frame-pointer",
            "-march=native",
            "-mtune=native",
        ]
        if "clang" in cc_name:
            base.append("-fuse-ld=lld")
    else:
        base = [
            "-O3",
            "-pthread",
            "-funroll-loops",
            "-flto",
            "-fomit-frame-pointer",
            "-march=native",
            "-mtune=native",
        ]
    variants = [base]
    if cc_name in ("gcc", "cc") or "gcc" in cc_name:
        variants.insert(0, [*base, "-ffast-math"])
    for i, flags in enumerate(variants):
        cmd = [cc, *flags, "-o", str(out), str(src)]
        print(f"Building native PoW solver ({'gcc tuned' if i == 0 and len(variants) > 1 else 'base'}): {' '.join(cmd)}")
        if subprocess.call(cmd, cwd=str(ROOT)) == 0:
            print(f"Native PoW solver built: {out}")
            return
    flags = [
        f
        for f in base
        if f
        not in (
            "-march=native",
            "-mtune=native",
            "-flto",
            "-ffast-math",
            "-fuse-ld=lld",
        )
    ]
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
            write_private(ENV_FILE, EXAMPLE_FILE.read_bytes())
            print("Created .env from .env.example.")
        else:
            write_private(ENV_FILE, b"")
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
    gigachat = validate_gigachat(collect_gigachat(current, defaults), defaults)
    opencode = collect_opencode(current, defaults)
    alice = collect_alice(current, defaults)
    duckai = collect_duckai(current, defaults)
    values.update(deepseek)
    values.update(qwen)
    values.update(gigachat)
    values.update(opencode)
    values.update(alice)
    values.update(duckai)

    print()
    print("Now the rest of the settings. Enter to keep the current value, !clear to erase, !reset to restore the default.")
    for title, fields in GROUPS:
        print()
        print(f"[ {title} ]")
        for key, label, kind in fields:
            current_value = values.get(key) or defaults.get(key, "")
            values[key] = prompt(key, label, kind, current_value, defaults.get(key, ""))

    update_env(values)
    print()
    print("Configuration saved to .env")

    has_ds = any(v for v in deepseek.values() if v)
    has_qwen = any(v for v in qwen.values() if v)
    has_gigachat = any(v for v in gigachat.values() if v)
    has_opencode = any(v for v in opencode.values() if v)
    has_alice = any(v for v in alice.values() if v)
    has_duckai = any(v for v in duckai.values() if v)
    if not (has_ds or has_qwen or has_gigachat or has_opencode or has_alice or has_duckai):
        print("Warning: no provider credentials configured.")
        print(
            "The server will not start until you add DEEPSEEK_TOKENS, QWEN_TOKENS, GIGACHAT_KEYS, OPENCODE_KEYS,"
            " ALICE_ENABLED=1, OPENCODE_ENABLED=1 or DUCKAI_ENABLED=1."
        )

    if ask("Create a DanyAPI launcher shortcut on the desktop?", True):
        try:
            path = create_shortcut()
            print(f"Shortcut created: {path}")
        except Exception as exc:
            print(f"Shortcut creation failed: {exc}")

    print()
    print(f"DanyAPI runs from: {ROOT}")
    print("docs/start.py auto-updates to the latest GitHub release at every launch (DANYAPI_AUTO_UPDATE=0 disables).")
    print("Start it anytime with the desktop shortcut or:")
    print("  python app.py        (from the DanyAPI folder)")
    print("  python -m danyapi")
    print("  python docs/start.py (auto-updates, then runs)")


if __name__ == "__main__":
    main()
