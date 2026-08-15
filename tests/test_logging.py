import logging
import logging.handlers

import pytest

from danyapi import logging as dlog
from danyapi.config import settings


def root_handler_names(root):
    return [getattr(h, "name", None) for h in root.handlers]


def test_has_handler():
    logger = logging.Logger("probe")
    assert not dlog._has_handler(logger, "x")
    handler = logging.StreamHandler()
    handler.name = "x"
    logger.addHandler(handler)
    assert dlog._has_handler(logger, "x")
    logger.removeHandler(handler)


def test_make_file_handler_creates_parent(tmp_path):
    path = tmp_path / "sub" / "app.log"
    handler = dlog._make_file_handler(str(path), 1024, 2)
    assert path.exists()
    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    handler.close()


def test_resolve_level_valid():
    assert dlog._resolve_level("DEBUG") == "DEBUG"
    assert dlog._resolve_level(" debug ") == "DEBUG"


@pytest.mark.parametrize("value", ["NOPE", "", None])
def test_resolve_level_invalid(value):
    assert dlog._resolve_level(value) == dlog.DEFAULT_LOG_LEVEL


def test_coerce_max_bytes():
    assert dlog._coerce_max_bytes(2048) == 2048
    assert dlog._coerce_max_bytes(0) == dlog.DEFAULT_MAX_BYTES
    assert dlog._coerce_max_bytes(-5) == dlog.DEFAULT_MAX_BYTES


def test_coerce_backup_count():
    assert dlog._coerce_backup_count(5) == 5
    assert dlog._coerce_backup_count(0) == 0
    assert dlog._coerce_backup_count(-1) == dlog.DEFAULT_BACKUP_COUNT


def test_uvicorn_log_config():
    cfg = dlog.uvicorn_log_config()
    assert cfg["disable_existing_loggers"] is False
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger_cfg = cfg["loggers"][name]
        assert logger_cfg["handlers"] == []
        assert logger_cfg["propagate"] is True
        assert logger_cfg["level"] == "INFO"


@pytest.fixture
def clean_handlers():
    root = logging.getLogger()
    before = list(root.handlers)
    for handler in list(root.handlers):
        if getattr(handler, "name", None) in (dlog.CONSOLE_HANDLER_NAME, dlog.FILE_HANDLER_NAME):
            root.removeHandler(handler)
            handler.close()
    yield
    for handler in list(root.handlers):
        if getattr(handler, "name", None) in (dlog.CONSOLE_HANDLER_NAME, dlog.FILE_HANDLER_NAME):
            root.removeHandler(handler)
            handler.close()
    for handler in before:
        if handler not in root.handlers:
            root.addHandler(handler)


@pytest.fixture
def saved_settings():
    saved = {k: getattr(settings, k) for k in ("log_level", "log_file", "log_max_bytes", "log_backup_count")}
    yield
    for k, v in saved.items():
        setattr(settings, k, v)


def test_configure_adds_console_once(clean_handlers, saved_settings):
    root = logging.getLogger()
    dlog.configure()
    assert root_handler_names(root).count(dlog.CONSOLE_HANDLER_NAME) == 1
    dlog.configure()
    assert root_handler_names(root).count(dlog.CONSOLE_HANDLER_NAME) == 1


def test_configure_sets_root_level(clean_handlers, saved_settings):
    settings.log_level = "DEBUG"
    dlog.configure()
    assert logging.getLogger().level == logging.DEBUG


def test_configure_sets_noisy_loggers(clean_handlers, saved_settings):
    settings.log_level = "INFO"
    dlog.configure()
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


def test_configure_adds_file_handler(clean_handlers, saved_settings, tmp_path):
    root = logging.getLogger()
    settings.log_file = str(tmp_path / "danyapi.log")
    settings.log_max_bytes = 1024
    settings.log_backup_count = 1
    dlog.configure()
    assert root_handler_names(root).count(dlog.FILE_HANDLER_NAME) == 1
    logging.getLogger("danyapi.test").warning("hello file")
    assert (tmp_path / "danyapi.log").exists()


def test_configure_skips_directory_log_file(clean_handlers, saved_settings, tmp_path):
    root = logging.getLogger()
    settings.log_file = str(tmp_path)
    settings.log_max_bytes = 1024
    settings.log_backup_count = 1
    dlog.configure()
    assert root_handler_names(root).count(dlog.FILE_HANDLER_NAME) == 0
