from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
CONSOLE_HANDLER_NAME = "danyapi-console"
FILE_HANDLER_NAME = "danyapi-file"


def _make_file_handler(log_file: str, max_bytes: int, backup_count: int) -> RotatingFileHandler:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    return handler


def _has_handler(root: logging.Logger, name: str) -> bool:
    return any(getattr(handler, "name", None) == name for handler in root.handlers)


def configure() -> None:
    from danyapi.config import settings

    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    if not _has_handler(root, CONSOLE_HANDLER_NAME):
        console = logging.StreamHandler()
        console.name = CONSOLE_HANDLER_NAME
        console.setFormatter(logging.Formatter(DEFAULT_FORMAT))
        root.addHandler(console)
    if settings.log_file and not _has_handler(root, FILE_HANDLER_NAME):
        file_handler = _make_file_handler(settings.log_file, settings.log_max_bytes, settings.log_backup_count)
        file_handler.name = FILE_HANDLER_NAME
        root.addHandler(file_handler)
