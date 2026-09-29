import ctypes
import logging
import queue
import sys
import types

import pytest

from danyapi import logging as dlog
from danyapi.config import settings


def _wintypes():
    try:
        from ctypes import wintypes
    except ImportError:
        wintypes = types.SimpleNamespace(DWORD=ctypes.c_uint32, HANDLE=ctypes.c_void_p, BOOL=ctypes.c_int)
        module = types.ModuleType("ctypes.wintypes")
        module.DWORD = ctypes.c_uint32
        module.HANDLE = ctypes.c_void_p
        module.BOOL = ctypes.c_int
        sys.modules.setdefault("ctypes.wintypes", module)
        ctypes.wintypes = sys.modules["ctypes.wintypes"]
    return wintypes


WINTYPES = _wintypes()


@pytest.fixture
def logging_state():
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level
    listeners_before = list(dlog._queue_listeners)
    saved = {key: getattr(settings, key) for key in ("log_level", "log_file", "log_max_bytes", "log_backup_count")}
    dlog._queue_listeners.clear()

    def _restore():
        for handler in list(root.handlers):
            if getattr(handler, "name", None) in (dlog.CONSOLE_HANDLER_NAME, dlog.FILE_HANDLER_NAME):
                root.removeHandler(handler)
                handler.close()
        for handler in handlers_before:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(level_before)
        for key, value in saved.items():
            setattr(settings, key, value)
        for listener in dlog._queue_listeners:
            listener.stop()
        dlog._queue_listeners.clear()
        dlog._queue_listeners.extend(listeners_before)
        dlog._file_handler_state.clear()
        dlog._DroppingQueueHandler.dropped = 0

    yield
    _restore()


def _file_handler():
    return next(h for h in logging.getLogger().handlers if getattr(h, "name", None) == dlog.FILE_HANDLER_NAME)


def test_escape_control_returns_an_empty_message_unchanged():
    assert dlog.escape_control("") == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a\nb", "a\\nb"),
        ("a\rb", "a\\rb"),
        ("a\tb", "a\\tb"),
        ("a\x00b", "a\\x00b"),
        ("a\x1fb", "a\\x1fb"),
        ("a\x7fb", "a\\x7fb"),
        ("plain", "plain"),
    ],
)
def test_escape_control_handles_every_class(raw, expected):
    assert dlog.escape_control(raw) == expected


def test_escape_control_covers_the_whole_control_range():
    escaped = dlog.escape_control("".join(chr(code) for code in range(0x20)) + "\x7f")
    assert "\n" not in escaped
    assert "\t" not in escaped
    assert "\x00" not in escaped
    assert escaped.count("\\x") == 30
    assert escaped.endswith("\\x7f")


def test_level_names_uses_the_stdlib_mapping(monkeypatch):
    monkeypatch.setattr(logging, "getLevelNamesMapping", lambda: {"INFO": 20, "WARNING": 30}, raising=False)
    assert dlog._level_names() == {"INFO", "WARNING"}


def test_level_names_falls_back_without_the_stdlib_mapping(monkeypatch):
    monkeypatch.delattr(logging, "getLevelNamesMapping", raising=False)
    assert dlog._level_names() == dlog._FALLBACK_LEVEL_NAMES


def test_record_message_replaces_a_mismatched_args_pair():
    record = logging.LogRecord("x", logging.INFO, "", 0, "value is %s", ("a", "b"), None)
    assert dlog._record_message(record) == "'value is %s' ('a', 'b')"
    assert record.msg == "'value is %s' ('a', 'b')"
    assert record.args == ()
    assert dlog._record_message(record) == "'value is %s' ('a', 'b')"


def test_lifecycle_filter_formats_a_malformed_record_without_raising():
    record = logging.LogRecord("x", logging.INFO, "", 0, "value is %s", ("a", "b"), None)
    assert dlog._LifecycleFilter().filter(record) is True
    assert dlog._record_message(record) == "'value is %s' ('a', 'b')"


def test_filter_and_formatter_share_one_formatting_of_a_record():
    calls = []

    class _CountingRecord(logging.LogRecord):
        def getMessage(self):
            calls.append(1)
            return super().getMessage()

    record = _CountingRecord("x", logging.INFO, "", 0, "count %s", ("once",), None)
    formatter = dlog._EscapingFormatter(dlog.DEFAULT_FORMAT, dlog.DEFAULT_DATEFMT)
    assert dlog._record_message(record) == "count once"
    assert calls == [1]
    assert dlog._LifecycleFilter().filter(record) is True
    assert calls == [1]
    assert formatter.format(record).endswith("count once")
    assert calls == [1, 1]


def test_color_formatter_without_a_colour_level_is_plain(monkeypatch):
    import io

    class TTY(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(dlog.sys, "stderr", TTY())
    formatter = dlog._ColorFormatter()
    assert formatter._use_color is True
    record = logging.LogRecord("x", logging.NOTSET, "", 0, "no level colour", (), None)
    text = formatter.format(record)
    assert "\033[" not in text
    assert text.endswith("NOTSET x no level colour")


class _FakeFunction:
    def __init__(self, handler):
        self.argtypes = None
        self.restype = None
        self._handler = handler

    def __call__(self, *args):
        return self._handler(*args)


class _FakeKernel32:
    def __init__(self, std_handles, console_modes):
        self.set_calls = []
        self.GetStdHandle = _FakeFunction(lambda std: std_handles[std])
        self.GetConsoleMode = _FakeFunction(lambda handle, mode: _write_mode(mode, console_modes[handle]))
        self.SetConsoleMode = _FakeFunction(self._set)

    def _set(self, handle, mode):
        self.set_calls.append((handle, mode))
        return 1


def _write_mode(pointer, value):
    ctypes.cast(pointer, ctypes.POINTER(WINTYPES.DWORD))[0] = value
    return 1


class _FakeWindll:
    def __init__(self, kernel32):
        self.kernel32 = kernel32


@pytest.fixture
def fake_windows(monkeypatch):
    monkeypatch.setattr(dlog.sys, "platform", "win32")

    def _install(std_handles, console_modes):
        kernel32 = _FakeKernel32(std_handles, console_modes)
        monkeypatch.setattr(ctypes, "windll", _FakeWindll(kernel32), raising=False)
        return kernel32

    return _install


STD_INPUT = 0xFFFFFFF5
STD_OUTPUT = 0xFFFFFFF4
INVALID_HANDLE = WINTYPES.HANDLE(-1).value
VT = 0x0004


def test_enable_windows_vt_is_skipped_off_windows(monkeypatch):
    monkeypatch.setattr(dlog.sys, "platform", "linux")
    kernel32 = _FakeKernel32({STD_INPUT: 10, STD_OUTPUT: 11}, {10: 0, 11: 0})
    monkeypatch.setattr(ctypes, "windll", _FakeWindll(kernel32), raising=False)
    assert dlog._enable_windows_vt() is None
    assert kernel32.set_calls == []


def test_enable_windows_vt_turns_the_flag_on_for_both_handles(fake_windows):
    kernel32 = fake_windows({STD_INPUT: 10, STD_OUTPUT: 11}, {10: 0x0001, 11: 0})
    assert dlog._enable_windows_vt() is None
    assert kernel32.set_calls == [(10, 0x0001 | VT), (11, VT)]


def test_enable_windows_vt_leaves_an_enabled_mode_alone(fake_windows):
    kernel32 = fake_windows({STD_INPUT: 10, STD_OUTPUT: 11}, {10: VT, 11: VT})
    dlog._enable_windows_vt()
    assert kernel32.set_calls == []


def test_enable_windows_vt_skips_an_invalid_handle(fake_windows):
    kernel32 = fake_windows({STD_INPUT: INVALID_HANDLE, STD_OUTPUT: 11}, {11: 0})
    dlog._enable_windows_vt()
    assert kernel32.set_calls == [(11, VT)]


def test_enable_windows_vt_skips_a_handle_without_a_console_mode(fake_windows):
    kernel32 = _FakeKernel32({STD_INPUT: 10, STD_OUTPUT: 11}, {10: 0, 11: 0})
    kernel32.GetConsoleMode = _FakeFunction(lambda handle, mode: 0)
    assert dlog._enable_windows_vt() is None
    assert kernel32.set_calls == []


def test_enable_windows_vt_reports_a_ctypes_failure(monkeypatch, caplog):
    monkeypatch.setattr(dlog.sys, "platform", "win32")

    def _boom(std):
        raise OSError("no console")

    kernel32 = _FakeKernel32({STD_INPUT: 10, STD_OUTPUT: 11}, {10: 0, 11: 0})
    kernel32.GetStdHandle = _FakeFunction(_boom)
    monkeypatch.setattr(ctypes, "windll", _FakeWindll(kernel32), raising=False)
    with caplog.at_level(logging.DEBUG, logger="danyapi.logging"):
        assert dlog._enable_windows_vt() is None
    assert kernel32.set_calls == []
    records = [record for record in caplog.records if record.name == "danyapi.logging"]
    assert [record.getMessage() for record in records] == ["failed to enable windows VT mode"]
    assert records[0].exc_info[0] is OSError


def test_full_queue_is_counted_and_reported_at_shutdown(caplog, logging_state):
    already_dropped = dlog._DroppingQueueHandler.dropped
    pending: queue.Queue = queue.Queue(maxsize=1)
    handler = dlog._DroppingQueueHandler(pending)
    records = [logging.LogRecord("x", logging.INFO, "", 0, f"n{index}", (), None) for index in range(3)]
    for record in records:
        handler.enqueue(record)
    assert pending.qsize() == 1
    assert pending.get() is records[0]
    assert dlog._DroppingQueueHandler.dropped == already_dropped + 2
    with caplog.at_level(logging.WARNING, logger="danyapi.logging"):
        dlog.shutdown()
    messages = [record.getMessage() for record in caplog.records if record.name == "danyapi.logging"]
    assert messages == [f"{already_dropped + 2} log record(s) were dropped because the file log queue was full"]


def test_shutdown_is_silent_without_drops(caplog, logging_state):
    dlog._DroppingQueueHandler.dropped = 0
    with caplog.at_level(logging.WARNING, logger="danyapi.logging"):
        dlog.shutdown()
    assert [record.getMessage() for record in caplog.records if record.name == "danyapi.logging"] == []


def test_configure_keeps_the_file_handler_when_the_target_is_unchanged(logging_state, tmp_path):
    settings.log_file = str(tmp_path / "same.log")
    settings.log_max_bytes = 4096
    settings.log_backup_count = 2
    dlog.configure()
    handler = _file_handler()
    dlog.configure()
    assert _file_handler() is handler
    assert len(dlog._queue_listeners) == 1
    assert dlog._file_handler_state["target"] == (str(tmp_path / "same.log"), 4096, 2)


def test_configure_swaps_the_file_handler_when_the_path_changes(logging_state, tmp_path):
    settings.log_file = str(tmp_path / "first.log")
    dlog.configure()
    first = _file_handler()
    dlog._DroppingQueueHandler.dropped = 5
    settings.log_file = str(tmp_path / "second.log")
    dlog.configure()
    second = _file_handler()
    assert second is not first
    assert dlog._DroppingQueueHandler.dropped == 0
    assert dlog._file_handler_state["target"] == (str(tmp_path / "second.log"), settings.log_max_bytes, settings.log_backup_count)
    names = [getattr(handler, "name", None) for handler in logging.getLogger().handlers]
    assert names.count(dlog.FILE_HANDLER_NAME) == 1
    assert dlog.shutdown() is None
    assert dlog._queue_listeners == []


@pytest.mark.parametrize(("field", "value", "expected"), [("log_max_bytes", 128, (128, 2)), ("log_backup_count", 7, (4096, 7))])
def test_configure_swaps_the_file_handler_when_rotation_changes(logging_state, tmp_path, field, value, expected):
    settings.log_file = str(tmp_path / "rotate.log")
    settings.log_max_bytes = 4096
    settings.log_backup_count = 2
    dlog.configure()
    first = _file_handler()
    setattr(settings, field, value)
    dlog.configure()
    assert _file_handler() is not first
    assert dlog._file_handler_state["target"] == (str(tmp_path / "rotate.log"), *expected)


def test_configure_removes_the_file_handler_when_the_path_is_cleared(logging_state, tmp_path):
    target = tmp_path / "dropped.log"
    settings.log_file = str(target)
    dlog.configure()
    handler = _file_handler()
    logging.getLogger("danyapi.test").warning("before clear")
    handler.queue.join()
    assert target.read_text(encoding="utf-8").count("before clear") == 1
    settings.log_file = ""
    dlog.configure()
    names = [getattr(handler, "name", None) for handler in logging.getLogger().handlers]
    assert dlog.FILE_HANDLER_NAME not in names
    assert dlog._file_handler_state == {}
    logging.getLogger("danyapi.test").warning("after clear")
    assert target.read_text(encoding="utf-8").count("after clear") == 0


def test_configure_reapplies_a_changed_level_to_existing_handlers(logging_state, tmp_path):
    settings.log_level = "INFO"
    settings.log_file = str(tmp_path / "level.log")
    dlog.configure()
    console = next(handler for handler in logging.getLogger().handlers if getattr(handler, "name", None) == dlog.CONSOLE_HANDLER_NAME)
    assert logging.getLogger().level == logging.INFO
    assert console.level == logging.INFO
    assert _file_handler().level == logging.INFO
    settings.log_level = "DEBUG"
    dlog.configure()
    assert logging.getLogger().level == logging.DEBUG
    assert console.level == logging.DEBUG
    assert _file_handler().level == logging.DEBUG


def test_configure_applies_noisy_logger_levels(logging_state):
    settings.log_level = "INFO"
    dlog.configure()
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
