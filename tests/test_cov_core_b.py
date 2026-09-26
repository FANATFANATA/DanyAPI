import asyncio
import io
import logging
import sys
from types import SimpleNamespace

import pytest

from danyapi import logging as dlog
from danyapi.pow import _parse_number, deepseek_hash_v1_hex, solve_python
from danyapi.sessions import SessionRegistry, _as_int
from danyapi.usage import UsageTracker

requires_windows = pytest.mark.skipif(sys.platform != "win32", reason="windows only")


def test_level_names_fallback(monkeypatch):
    monkeypatch.delattr(logging, "getLevelNamesMapping", raising=False)
    assert dlog._level_names() == set(dlog._FALLBACK_LEVEL_NAMES)


def test_color_formatter_uncolored_level(monkeypatch):
    class TTY(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(dlog.sys, "stderr", TTY())
    fmt = dlog._ColorFormatter()
    record = logging.LogRecord("x", logging.DEBUG, "", 0, "trace", (), None)
    text = fmt.format(record)
    assert "trace" in text
    assert dlog.RESET not in text


def test_enable_windows_vt_non_windows(monkeypatch):
    monkeypatch.setattr(dlog.sys, "platform", "linux")
    dlog._enable_windows_vt()


@requires_windows
def test_enable_windows_vt_invalid_handle(monkeypatch):
    import ctypes

    kernel32 = ctypes.windll.kernel32
    monkeypatch.setattr(kernel32, "GetStdHandle", lambda handle: 0)
    dlog._enable_windows_vt()


@requires_windows
def test_enable_windows_vt_sets_console_mode(monkeypatch):
    import ctypes

    kernel32 = ctypes.windll.kernel32
    seen = []

    def get_handle(handle):
        return 1234

    def get_mode(handle, mode):
        return 1

    def set_mode(handle, mode):
        seen.append(mode)
        return 1

    monkeypatch.setattr(kernel32, "GetStdHandle", get_handle)
    monkeypatch.setattr(kernel32, "GetConsoleMode", get_mode)
    monkeypatch.setattr(kernel32, "SetConsoleMode", set_mode)
    dlog._enable_windows_vt()
    assert seen


@requires_windows
def test_enable_windows_vt_failure_swallowed(monkeypatch, caplog):
    import ctypes

    kernel32 = ctypes.windll.kernel32

    def boom(handle):
        raise RuntimeError("no std handle")

    monkeypatch.setattr(kernel32, "GetStdHandle", boom)
    with caplog.at_level(logging.DEBUG, logger="danyapi.logging"):
        dlog._enable_windows_vt()
    assert any("failed to enable windows VT mode" in record.getMessage() for record in caplog.records)


def test_parse_number_bool_and_float():
    assert _parse_number(True) is None
    assert _parse_number(3) == 3
    assert _parse_number(1.5) == 1.5
    assert _parse_number(float("nan")) is None
    assert _parse_number(float("inf")) is None


def test_parse_number_strings():
    assert _parse_number("") is None
    assert _parse_number("42") == 42
    assert _parse_number("3.5") == 3.5
    assert _parse_number("abc") is None
    assert _parse_number("nan") is None
    assert _parse_number(None) is None


def test_solve_python_long_prefix_fallback():
    salt = "S" * 140
    expire_at = 1
    prefix = f"{salt}_{expire_at}_".encode()
    target = deepseek_hash_v1_hex(prefix + b"5")
    assert solve_python(target, salt, expire_at, 100) == 5


def test_usage_evicts_model_overflow():
    tracker = UsageTracker(max_records=1)
    tracker.record("p", "m1", 1, 1, 2)
    tracker.record("p", "m2", 1, 1, 2)
    snap = tracker.snapshot()
    assert "m1" not in snap["by_model"]
    assert "m2" in snap["by_model"]


class _BoomSet:
    def get(self, key):
        return None

    def set(self, key, value):
        raise RuntimeError("boom")

    def discard(self, key):
        pass

    def flush(self):
        pass


class _BoomDiscard:
    def get(self, key):
        return None

    def set(self, key, value):
        pass

    def discard(self, key):
        raise RuntimeError("boom")

    def flush(self):
        pass


def test_usage_record_store_write_failure_swallowed():
    tracker = UsageTracker(store=_BoomSet())
    tracker.record("p", "m", 1, 1, 2)


def test_usage_flush_store_failure_swallowed():
    tracker = UsageTracker(store=_BoomSet())
    tracker.flush()


def test_usage_reset_store_failure_swallowed():
    tracker = UsageTracker(store=_BoomDiscard())
    tracker.reset()


class _Client:
    async def create_session(self, **kwargs):
        return SimpleNamespace(id="s1")


def test_as_int_bool_and_strings():
    assert _as_int(True) == 0
    assert _as_int(5) == 5
    assert _as_int(5.9) == 5
    assert _as_int("7") == 7
    assert _as_int("7.9") == 7
    assert _as_int("abc") == 0
    assert _as_int(None) == 0


def test_release_session_lock_missing_refs():
    reg = SessionRegistry(_Client())
    asyncio.run(reg._release_session_lock("missing"))


def test_release_session_lock_decrements_existing_refs():
    reg = SessionRegistry(_Client())

    async def run():
        await reg._session_lock("k")
        await reg._session_lock("k")
        await reg._release_session_lock("k")
        assert reg._session_refs["k"] == 1

    asyncio.run(run())
