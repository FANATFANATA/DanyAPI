from __future__ import annotations

import logging
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
SUCCESS_MARKERS = (" ok", "ready", "success")
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
    lowered = message.lower()
    return any(marker in lowered for marker in SUCCESS_MARKERS)


class _ColorFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(DEFAULT_FORMAT, DEFAULT_DATEFMT)
        self._use_color = bool(sys.stdout and sys.stdout.isatty())

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if isinstance(message, str):
            message = message.replace("{", "{{").replace("}", "}}")
        record.msg = message
        record.args = ()
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

        kernel32 = ctypes.windll.kernel32
        for std_handle in (-11, -12):
            handle = kernel32.GetStdHandle(std_handle)
            if not handle or handle == -1:
                continue
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                continue
            if not mode.value & 0x0004:
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def configure() -> None:
    from danyapi.config import settings

    _enable_windows_vt()

    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    level = _resolve_level(settings.log_level)
    root = logging.getLogger()
    root.setLevel(level)

    if not _has_handler(root, CONSOLE_HANDLER_NAME):
        console = logging.StreamHandler()
        console.name = CONSOLE_HANDLER_NAME
        console.setLevel(level)
        console.setFormatter(_ColorFormatter())
        console.addFilter(_LifecycleFilter())
        root.addHandler(console)

    if settings.log_file and not _has_handler(root, FILE_HANDLER_NAME):
        path = Path(settings.log_file)
        if path.is_dir():
            logging.getLogger("danyapi.logging").warning(
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
