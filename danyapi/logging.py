from __future__ import annotations

import ipaddress
import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_FORMAT = "(%(asctime)s) %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"
RESET = "\033[0m"
LEVEL_COLORS = {
    "INFO": "\033[37m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[31m",
}
SUCCESS_COLOR = "\033[32m"
SUCCESS_PATTERN = re.compile(r"\b(ok|ready|success)\b", re.IGNORECASE)
LIFECYCLE_MESSAGES = {
    "Waiting for application startup.",
    "Application startup complete.",
    "Waiting for application shutdown.",
    "Application shutdown complete.",
}
LIFECYCLE_PREFIXES = ("Started server process", "Finished server process")
UVICORN_RUNNING = "Uvicorn running on"
DANYAPI_RUNNING = "DanyAPI running on"
CONSOLE_HANDLER_NAME = "danyapi-console"
FILE_HANDLER_NAME = "danyapi-file"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 3
_FALLBACK_LEVEL_NAMES = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


def _level_names() -> set[str]:
    get_mapping = getattr(logging, "getLevelNamesMapping", None)
    if get_mapping is not None:
        return set(get_mapping())
    return set(_FALLBACK_LEVEL_NAMES)


def _resolve_level(level: str | None) -> str:
    normalized = str(level or "").strip().upper()
    if normalized in _level_names():
        return normalized
    return DEFAULT_LOG_LEVEL


def _coerce_max_bytes(value: int) -> int:
    if value and value > 0:
        return value
    return DEFAULT_MAX_BYTES


def _coerce_backup_count(value: int) -> int:
    if value >= 0:
        return value
    return DEFAULT_BACKUP_COUNT


def _make_file_handler(log_file: str, max_bytes: int, backup_count: int) -> RotatingFileHandler:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATEFMT))
    return handler


class _LifecycleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "uvicorn.access":
            return False
        message = record.getMessage()
        if message in LIFECYCLE_MESSAGES:
            return False
        if message.startswith(LIFECYCLE_PREFIXES):
            return False
        if UVICORN_RUNNING in message:
            record.msg = message.replace(UVICORN_RUNNING, DANYAPI_RUNNING)
            record.args = ()
        return True


def _is_success(message: str) -> bool:
    return SUCCESS_PATTERN.search(message) is not None


class _ColorFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(DEFAULT_FORMAT, DEFAULT_DATEFMT)
        stream = sys.stderr if sys.stderr is not None else sys.stdout
        self._use_color = bool(stream and stream.isatty())

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self._use_color:
            return text
        color = LEVEL_COLORS.get(record.levelname, "")
        if record.levelname == "INFO" and _is_success(record.getMessage()):
            color = SUCCESS_COLOR
        if not color:
            return text
        return f"{color}{text}{RESET}"


def _has_handler(root: logging.Logger, name: str) -> bool:
    return any(getattr(handler, "name", None) == name for handler in root.handlers)


def _enable_windows_vt() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel32.GetStdHandle.restype = wintypes.HANDLE
        kernel32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetConsoleMode.restype = wintypes.BOOL
        kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetConsoleMode.restype = wintypes.BOOL
        enable_virtual_terminal_processing = 0x0004
        invalid_handle = wintypes.HANDLE(-1).value
        for std_handle in (-11, -12):
            handle = kernel32.GetStdHandle(std_handle)
            if not handle or handle == invalid_handle:
                continue
            mode = wintypes.DWORD()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                continue
            if not mode.value & enable_virtual_terminal_processing:
                kernel32.SetConsoleMode(handle, mode.value | enable_virtual_terminal_processing)
    except Exception:
        logging.getLogger(__name__).debug("failed to enable windows VT mode", exc_info=True)


def configure() -> None:
    from danyapi.config import settings

    _enable_windows_vt()

    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    level = _resolve_level(settings.log_level)
    root = logging.getLogger()
    root.setLevel(level)

    if not _has_handler(root, CONSOLE_HANDLER_NAME):
        console = logging.StreamHandler(sys.stdout)
        console.name = CONSOLE_HANDLER_NAME
        console.setLevel(level)
        console.setFormatter(_ColorFormatter())
        console.addFilter(_LifecycleFilter())
        root.addHandler(console)

    if settings.log_file and not _has_handler(root, FILE_HANDLER_NAME):
        path = Path(settings.log_file)
        if path.is_dir():
            logging.getLogger(__name__).warning(
                "log file %s is a directory, using console only",
                path,
            )
        else:
            file_handler = _make_file_handler(
                str(path),
                _coerce_max_bytes(settings.log_max_bytes),
                _coerce_backup_count(settings.log_backup_count),
            )
            file_handler.name = FILE_HANDLER_NAME
            file_handler.setLevel(level)
            file_handler.addFilter(_LifecycleFilter())
            root.addHandler(file_handler)


def uvicorn_log_config() -> dict:
    from danyapi.config import settings

    level = _resolve_level(settings.log_level)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "loggers": {
            "uvicorn": {"handlers": [], "level": level, "propagate": True},
            "uvicorn.error": {"handlers": [], "level": level, "propagate": True},
            "uvicorn.access": {"handlers": [], "level": level, "propagate": True},
        },
    }


IP_CHECK_ENDPOINTS = [
    "https://ifconfig.me/ip",
    "https://api.ipify.org",
    "https://icanhazip.com",
]


def _is_valid_ip(text: str) -> bool:
    try:
        ipaddress.ip_address(text.strip())
        return True
    except ValueError:
        return False


def get_outgoing_ip(proxy: str | None = None, timeout: float = 4.0) -> tuple[str | None, str | None]:
    proxy_url = proxy if (isinstance(proxy, str) and proxy.strip()) else None
    last_err: str | None = None

    try:
        from .config import settings

        ua = getattr(settings, "user_agent", "curl/7.88.1")
    except Exception:
        ua = "curl/7.88.1"

    # 1. Try httpx
    try:
        import httpx

        with httpx.Client(headers={"User-Agent": ua}, proxy=proxy_url, timeout=timeout) as client:
            for url in IP_CHECK_ENDPOINTS:
                try:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        candidate = resp.text.strip()
                        if candidate and _is_valid_ip(candidate):
                            return candidate, None
                except Exception as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
                    continue
    except Exception as exc:
        last_err = f"{type(exc).__name__}: {exc}"

    # 2. Fallback to curl if available
    try:
        import shutil
        import subprocess

        curl_path = shutil.which("curl")
        if curl_path:
            for url in IP_CHECK_ENDPOINTS:
                cmd = [curl_path, "-s", "-A", ua, "--max-time", str(int(timeout))]
                if proxy_url:
                    if proxy_url.startswith("socks5://") or proxy_url.startswith("socks5h://"):
                        socks_addr = proxy_url.split("://", 1)[1]
                        cmd.extend(["--socks5-hostname", socks_addr])
                    else:
                        cmd.extend(["-x", proxy_url])
                cmd.append(url)
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 1, check=False)
                    candidate = res.stdout.strip()
                    if res.returncode == 0 and candidate and _is_valid_ip(candidate):
                        return candidate, None
                except Exception:
                    continue
    except Exception as exc:
        last_err = str(exc)

    return None, last_err


def log_startup_info() -> None:
    from danyapi.config import settings

    log = logging.getLogger("danyapi")

    raw_proxy = getattr(settings, "proxy", None)
    proxy = raw_proxy if (isinstance(raw_proxy, str) and raw_proxy.strip()) else None
    proxy_desc = f"via proxy {proxy}" if proxy else "direct, no proxy"

    try:
        ip, err = get_outgoing_ip(proxy=proxy, timeout=4.0)
        if ip:
            log.info("outgoing IP: %s (%s)", ip, proxy_desc)
        elif err:
            log.warning("could not determine outgoing IP (%s): %s", proxy_desc, err)
        else:
            log.warning("could not determine outgoing IP (%s)", proxy_desc)
    except Exception as exc:
        log.warning("could not determine outgoing IP (%s): %s", proxy_desc, exc)

    raw_api_key = getattr(settings, "api_key", None)
    api_key = raw_api_key if (isinstance(raw_api_key, str) and raw_api_key.strip()) else None
    if api_key:
        log.info("authentication: Bearer token required")
    else:
        log.info("authentication: open (no API_KEY set)")
