from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from danyapi import store as store_mod
from danyapi.store import JsonStore

USAGE_RECORD = {"recent": [{"user": "alice", "session_id": "s1"}]}


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod.settings, "cache_dir", str(tmp_path))
    monkeypatch.setattr(store_mod.settings, "cache_enabled", True)
    return tmp_path


class _Clock:
    def __init__(self, start: float) -> None:
        self.now = start
        self.on_sleep: Callable[[], None] | None = None

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        hook = self.on_sleep
        if hook is not None:
            self.on_sleep = None
            hook()


def _no_background_flush(monkeypatch) -> None:
    settled: Future[None] = Future()
    settled.set_result(None)
    monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", lambda executor, func, *args: settled)


def test_cache_disabled_keeps_usage_out_of_the_cache_directory(cache_dir, monkeypatch):
    def exercise() -> tuple[JsonStore, list[tuple[str, Any]]]:
        store = JsonStore("usage", "default")
        store.set("usage", USAGE_RECORD)
        store.flush()
        return store, store.items()

    enabled, enabled_items = exercise()
    assert enabled.enabled is True
    assert enabled_items == [("usage", USAGE_RECORD)]
    assert json.loads((cache_dir / "usage-default.json").read_text(encoding="utf-8")) == {"usage": USAGE_RECORD}
    (cache_dir / "usage-default.json").unlink()

    monkeypatch.setattr(store_mod.settings, "cache_enabled", False)
    disabled, disabled_items = exercise()
    assert disabled.enabled is False
    assert disabled._path is None
    assert disabled_items == enabled_items
    assert list(cache_dir.iterdir()) == []


def test_fsync_dir_syncs_the_directory_on_posix(tmp_path, monkeypatch):
    calls: list[tuple[Any, ...]] = []

    class FakeOs:
        name = "posix"
        O_RDONLY = 0

        @staticmethod
        def open(path: Any, flags: int) -> int:
            calls.append(("open", path, flags))
            return 11

        @staticmethod
        def fsync(fd: int) -> None:
            calls.append(("fsync", fd))

        @staticmethod
        def close(fd: int) -> None:
            calls.append(("close", fd))

    monkeypatch.setattr(store_mod, "os", FakeOs)
    assert store_mod._fsync_dir(tmp_path) is None
    assert calls == [("open", tmp_path, 0), ("fsync", 11), ("close", 11)]


def test_fsync_dir_never_opens_the_directory_on_windows(tmp_path):
    assert store_mod.os.name == "nt"
    assert store_mod._fsync_dir(tmp_path / "not-created") is None


def test_report_flush_failure_logs_the_error(caplog):
    future: Future[None] = Future()
    future.set_exception(RuntimeError("disk gone"))
    with caplog.at_level(logging.WARNING, logger="danyapi.store"):
        store_mod._report_flush_failure(future)
    assert [record.getMessage() for record in caplog.records] == ["cache flush failed: disk gone"]


async def test_report_flush_failure_ignores_a_cancelled_future(caplog):
    future = asyncio.get_running_loop().create_future()
    future.cancel()
    with caplog.at_level(logging.WARNING, logger="danyapi.store"):
        store_mod._report_flush_failure(future)
    assert caplog.records == []


async def test_report_flush_failure_ignores_an_unfinished_future(caplog):
    future = asyncio.get_running_loop().create_future()
    with caplog.at_level(logging.WARNING, logger="danyapi.store"):
        store_mod._report_flush_failure(future)
    assert caplog.records == []
    future.cancel()


async def test_a_burst_of_writes_coalesces_into_one_commit(cache_dir, monkeypatch):
    monkeypatch.setattr(store_mod, "_FLUSH_DEBOUNCE", 0.0)
    monkeypatch.setattr(store_mod, "time", _Clock(1000.0))
    store = JsonStore("burst", "default")
    _no_background_flush(monkeypatch)
    commits: list[dict] = []
    monkeypatch.setattr(store, "_commit", lambda data: commits.append(dict(data)))

    for index in range(20):
        store.set(f"k{index}", index)
    assert store._pending is True
    assert store._dirty is True
    assert store._idle.is_set() is False

    store._flush_background()
    assert commits == [{f"k{index}": index for index in range(20)}]
    assert store._pending is False
    assert store._dirty is False
    assert store._idle.is_set() is True


async def test_an_update_landing_during_the_debounce_reaches_the_disk(cache_dir, monkeypatch):
    monkeypatch.setattr(store_mod, "_FLUSH_DEBOUNCE", 0.25)
    clock = _Clock(1000.0)
    monkeypatch.setattr(store_mod, "time", clock)
    store = JsonStore("nodrop", "default")
    _no_background_flush(monkeypatch)
    store.set("a", 1)
    commits: list[dict] = []
    real_commit = store._commit

    def commit(data: Any) -> None:
        commits.append(dict(data))
        real_commit(data)

    monkeypatch.setattr(store, "_commit", commit)
    clock.on_sleep = lambda: store.set("late", 2)
    store._flush_background()
    assert commits == [{"a": 1, "late": 2}, {"a": 1, "late": 2}]
    assert clock.now == 1000.5
    assert store._pending is False
    assert store._dirty is False
    assert store._path is not None
    assert json.loads(store._path.read_text(encoding="utf-8")) == {"a": 1, "late": 2}


async def test_a_background_flush_with_nothing_to_write_clears_the_pending_flag(cache_dir, monkeypatch):
    monkeypatch.setattr(store_mod, "_FLUSH_DEBOUNCE", 0.0)
    monkeypatch.setattr(store_mod, "time", _Clock(1000.0))
    store = JsonStore("nothing", "default")
    _no_background_flush(monkeypatch)
    commits: list[dict] = []
    monkeypatch.setattr(store, "_commit", lambda data: commits.append(dict(data)))

    store.set("a", 1)
    store._dirty = False
    store._flush_background()
    assert commits == []
    assert store._pending is False
    assert store._dirty is False
    assert store._idle.is_set() is True

    store.set("b", 2)
    assert store._pending is True
    store._removed = True
    store._flush_background()
    assert commits == []
    assert store._pending is False
    assert store._dirty is False
    assert store._idle.is_set() is True


def test_a_removed_store_keeps_an_unserialisable_value_in_memory(cache_dir, caplog):
    store = JsonStore("removed-probe", "default")
    store.set("k", "v")
    store.remove()
    caplog.clear()
    value = object()
    with caplog.at_level(logging.WARNING, logger="danyapi.store"):
        store.set("bad", value)
    assert caplog.records == []
    assert store.get("bad") is value
    assert store._path is not None
    assert not store._path.exists()


def test_remove_logs_when_the_file_cannot_be_deleted(cache_dir, monkeypatch, caplog):
    store = JsonStore("undeletable", "default")
    store.set("k", "v")

    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("locked by another process")

    monkeypatch.setattr(Path, "unlink", boom)
    with caplog.at_level(logging.WARNING, logger="danyapi.store"):
        store.remove()
    assert [record.getMessage() for record in caplog.records] == [f"cache file delete failed for {store._path}: locked by another process"]
    assert store.enabled is False
    assert store._path is not None
    assert store._path.exists()


def test_a_load_trims_a_file_that_is_over_the_maxsize(cache_dir):
    path = store_mod.cache_root() / "capped-default.json"
    path.write_text(json.dumps({"a": 1, "b": 2, "c": 3}), encoding="utf-8")
    store = JsonStore("capped", "default", maxsize=2)
    assert sorted(store.items()) == [("b", 2), ("c", 3)]
    store.set("d", 4)
    assert sorted(store.items()) == [("c", 3), ("d", 4)]
    assert sorted(JsonStore("capped", "default", maxsize=2).items()) == [("c", 3), ("d", 4)]


def test_a_failed_commit_leaves_no_temp_file_behind(cache_dir, monkeypatch, caplog):
    store = JsonStore("failed-commit", "default")
    store.set("k", "v")
    store.flush()
    before = sorted(path.name for path in cache_dir.iterdir())
    monkeypatch.setattr(os, "replace", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("denied")))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="danyapi.store"):
        store.set("k2", "v2")
    assert [record.getMessage() for record in caplog.records] == [f"cache write failed for {store._path}: denied"]
    assert store.get("k2") == "v2"
    assert sorted(path.name for path in cache_dir.iterdir()) == before
    assert not list(cache_dir.glob("*.tmp"))


def test_a_temp_file_that_cannot_be_removed_is_only_logged(cache_dir, monkeypatch, caplog):
    store = JsonStore("stuck-tmp", "default")
    store.set("k", "v")
    store.flush()
    monkeypatch.setattr(os, "replace", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("denied")))
    monkeypatch.setattr(Path, "unlink", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("still locked")))
    with caplog.at_level(logging.WARNING, logger="danyapi.store"):
        store.set("k2", "v2")
    assert [record.getMessage() for record in caplog.records] == [f"cache write failed for {store._path}: denied"]
    assert store.get("k2") == "v2"


async def test_a_closed_loop_falls_back_to_a_synchronous_write(cache_dir, monkeypatch):
    store = JsonStore("closed-loop", "default")
    monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", _raise_closed_loop)
    store.set("k", "v")
    assert store._pending is False
    assert store._dirty is False
    assert store._idle.is_set() is True
    assert store._path is not None
    assert json.loads(store._path.read_text(encoding="utf-8")) == {"k": "v"}


def _raise_closed_loop(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("Event loop is closed")
