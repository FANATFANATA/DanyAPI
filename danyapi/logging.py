from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
CONSOLE_HANDLER_NAME = "danyapi-console"
FILE_HANDLER_NAME = "danyapi-file"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 3


def _resolve_level(level: str | None) -> str:
    normalized = str(level or "").strip().upper()
    if normalized in logging.getLevelNamesMapping():
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
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    return handler


def _has_handler(root: logging.Logger, name: str) -> bool:
    return any(getattr(handler, "name", None) == name for handler in root.handlers)


def configure() -> None:
    from danyapi.config import settings

    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    level = _resolve_level(settings.log_level)
    root = logging.getLogger()
    root.setLevel(level)

    if not _has_handler(root, CONSOLE_HANDLER_NAME):
        console = logging.StreamHandler()
        console.name = CONSOLE_HANDLER_NAME
        console.setLevel(level)
        console.setFormatter(logging.Formatter(DEFAULT_FORMAT))
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
