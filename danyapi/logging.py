from __future__ import annotations

import atexit
import logging
import queue
import re
import sys
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path

DEFAULT_FORMAT = "(%(asctime)s) %(levelname)s %(name)s %(message)s"
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
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_CONTROL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
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
_FILE_QUEUE_MAX = 10000
_FALLBACK_LEVEL_NAMES = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


def escape_control(message: str) -> str:
    if not message:
        return message
    return _CONTROL_RE.sub(lambda match: _CONTROL_ESCAPES.get(match.group(0), f"\\x{ord(match.group(0)):02x}"), message)


def _level_names() -> set[str]:
    get_mapping = getattr(logging, "getLevelNamesMapping", None)
    if get_mapping is not None:
        return set(get_mapping())
    return set(_FALLBACK_LEVEL_NAMES)


_LEVEL_NAMES = _level_names()
_queue_listeners: list[QueueListener] = []
_file_handler_state: dict[str, tuple[str, int, int]] = {}


def _resolve_level(level: str | None) -> str:
    normalized = str(level or "").strip().upper()
    if normalized in _LEVEL_NAMES:
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
    handler.setFormatter(_EscapingFormatter(DEFAULT_FORMAT, DEFAULT_DATEFMT))
    return handler


def _record_message(record: logging.LogRecord) -> str:
    cached = getattr(record, "_danyapi_message", None)
    if isinstance(cached, str):
        return cached
    if isinstance(record.msg, str) and not record.args:
        message = record.msg
    else:
        try:
            message = record.getMessage()
        except Exception:
            message = f"{record.msg!r} {record.args!r}"
            _set_record_message(record, message)
            return message
    record._danyapi_message = message  # type: ignore[attr-defined]
    return message


def _set_record_message(record: logging.LogRecord, message: str) -> None:
    record.msg = message
    record.args = ()
    record._danyapi_message = message  # type: ignore[attr-defined]


class _LifecycleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "uvicorn.access":
            return False
        return self._apply(record, _record_message(record))

    def _apply(self, record: logging.LogRecord, message: str) -> bool:
        if message in LIFECYCLE_MESSAGES:
            return False
        if message.startswith(LIFECYCLE_PREFIXES):
            return False
        if UVICORN_RUNNING in message:
            _set_record_message(record, message.replace(UVICORN_RUNNING, DANYAPI_RUNNING))
        return True


def _is_success(message: str) -> bool:
    return SUCCESS_PATTERN.search(message) is not None


class _EscapingFormatter(logging.Formatter):
    def formatMessage(self, record: logging.LogRecord) -> str:
        _set_record_message(record, _record_message(record))
        return escape_control(super().formatMessage(record))


class _ColorFormatter(_EscapingFormatter):
    def __init__(self) -> None:
        super().__init__(DEFAULT_FORMAT, DEFAULT_DATEFMT)
        stream = sys.stderr if sys.stderr is not None else sys.stdout
        self._use_color = bool(stream and stream.isatty())

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self._use_color:
            return text
        color = LEVEL_COLORS.get(record.levelname, "")
        if record.levelname == "INFO" and _is_success(_record_message(record)):
            color = SUCCESS_COLOR
        if not color:
            return text
        return f"{color}{text}{RESET}"


class _DroppingQueueHandler(QueueHandler):
    dropped = 0

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            type(self).dropped += 1


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
        kernel32.GetConsoleMode.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetConsoleMode.restype = wintypes.BOOL
        kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetConsoleMode.restype = wintypes.BOOL
        enable_virtual_terminal_processing = 0x0004
        invalid_handle = wintypes.HANDLE(-1).value
        for std_handle in (0xFFFFFFF5, 0xFFFFFFF4):
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


def _find_handler(root: logging.Logger, name: str) -> logging.Handler | None:
    for handler in root.handlers:
        if getattr(handler, "name", None) == name:
            return handler
    return None


def _drop_file_handler(root: logging.Logger) -> None:
    handler = _find_handler(root, FILE_HANDLER_NAME)
    if handler is None:
        return
    root.removeHandler(handler)
    _DroppingQueueHandler.dropped = 0
    handler.close()


def configure() -> None:
    from danyapi.config import settings

    _enable_windows_vt()

    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    level = _resolve_level(settings.log_level)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        if getattr(handler, "name", None) in (CONSOLE_HANDLER_NAME, FILE_HANDLER_NAME):
            handler.setLevel(level)

    console = _find_handler(root, CONSOLE_HANDLER_NAME)
    if console is None:
        console = logging.StreamHandler()
        console.name = CONSOLE_HANDLER_NAME
        console.setLevel(level)
        console.setFormatter(_ColorFormatter())
        console.addFilter(_LifecycleFilter())
        root.addHandler(console)

    target = settings.log_file
    max_bytes = _coerce_max_bytes(settings.log_max_bytes)
    backup_count = _coerce_backup_count(settings.log_backup_count)
    existing = _find_handler(root, FILE_HANDLER_NAME)
    if not target:
        _drop_file_handler(root)
        _file_handler_state.clear()
        return
    if existing is not None and _file_handler_state.get("target") == (target, max_bytes, backup_count):
        return
    _drop_file_handler(root)
    path = Path(target)
    if path.is_dir():
        logging.getLogger(__name__).warning(
            "log file %s is a directory, using console only",
            path,
        )
        return
    try:
        file_target = _make_file_handler(target, max_bytes, backup_count)
    except OSError as exc:
        logging.getLogger(__name__).warning("cannot open log file %s: %s, using console only", path, exc)
        return
    queue_for_file: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=_FILE_QUEUE_MAX)
    queue_handler = _DroppingQueueHandler(queue_for_file)
    queue_handler.name = FILE_HANDLER_NAME
    queue_handler.setLevel(level)
    queue_handler.addFilter(_LifecycleFilter())
    listener = QueueListener(queue_for_file, file_target)
    listener.start()
    _queue_listeners.append(listener)
    root.addHandler(queue_handler)
    _file_handler_state["target"] = (target, max_bytes, backup_count)


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


def shutdown() -> None:
    while _queue_listeners:
        listener = _queue_listeners[-1]
        try:
            listener.stop()
        except Exception as exc:
            logging.getLogger(__name__).warning("log queue listener did not stop: %s", exc)
            break
        _queue_listeners.pop()
    if _DroppingQueueHandler.dropped:
        logging.getLogger(__name__).warning("%d log record(s) were dropped because the file log queue was full", _DroppingQueueHandler.dropped)


atexit.register(shutdown)
