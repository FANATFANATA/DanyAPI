import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from danyapi import store as store_mod
from danyapi.store import JsonStore


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod.settings, "cache_dir", str(tmp_path))
    return tmp_path


def test_cache_root_uses_override(cache_dir):
    assert store_mod.cache_root() == cache_dir


def test_cache_root_falls_back_to_temp(monkeypatch):
    monkeypatch.setattr(store_mod.settings, "cache_dir", "")
    root = store_mod.cache_root()
    assert isinstance(root, Path)
    assert root.is_absolute()


def test_disabled_without_scope(cache_dir):
    store = JsonStore("n", None)
    assert not store.enabled
    store.set("k", "v")
    assert store.get("k2") is None
    assert "k" in store


def test_enabled_with_scope(cache_dir):
    assert JsonStore("sessions", "default").enabled


def test_set_get_roundtrip(cache_dir):
    store = JsonStore("a", "default")
    store.set("k", {"nested": [1, 2, 3]})
    assert store.get("k") == {"nested": [1, 2, 3]}


def test_get_default(cache_dir):
    store = JsonStore("b", "default")
    assert store.get("missing") is None
    assert store.get("missing", 42) == 42


def test_pop_existing_and_missing(cache_dir):
    store = JsonStore("c", "default")
    store.set("k", "v")
    assert store.pop("k") == "v"
    assert "k" not in store
    assert store.pop("k", "dflt") == "dflt"


def test_discard(cache_dir):
    store = JsonStore("d", "default")
    store.set("k", "v")
    store.discard("k")
    assert "k" not in store
    store.discard("absent")


def test_clear(cache_dir):
    store = JsonStore("e", "default")
    store.set("a", 1)
    store.set("b", 2)
    store.clear()
    assert len(store) == 0


def test_items_len_contains(cache_dir):
    store = JsonStore("f", "default")
    store.set("a", 1)
    store.set("b", 2)
    assert sorted(store.items()) == [("a", 1), ("b", 2)]
    assert len(store) == 2
    assert "a" in store
    assert "z" not in store


def test_flush_writes_file(cache_dir):
    store = JsonStore("persist", "default")
    store.set("key", "value")
    assert store._path is not None
    raw = json.loads(store._path.read_text(encoding="utf-8"))
    assert raw == {"key": "value"}


def test_reload_from_disk(cache_dir):
    JsonStore("reload", "default").set("key", "value")
    store2 = JsonStore("reload", "default")
    assert store2.get("key") == "value"


def test_corrupted_json_ignored(cache_dir):
    path = store_mod.cache_root() / "corrupt-scope.json"
    path.write_text("not json", encoding="utf-8")
    store = JsonStore("corrupt", "scope")
    assert len(store) == 0


def test_non_dict_json_ignored(cache_dir):
    path = store_mod.cache_root() / "list-scope.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    store = JsonStore("list", "scope")
    assert len(store) == 0


def test_scope_sanitization(cache_dir):
    store = JsonStore("weird", "a b/c\\d")
    assert store._path is not None
    assert store._path.name.endswith(".json")


def test_cache_root_mkdir_error(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("file", encoding="utf-8")
    monkeypatch.setattr(store_mod.settings, "cache_dir", str(blocker))
    assert store_mod.cache_root() == blocker


def test_load_skips_disabled():
    store = JsonStore("n", None)
    store._load()
    assert store._data == {}


def test_write_error_logged(cache_dir, monkeypatch):
    import os

    store = JsonStore("w", "default")
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("denied")))
    store.set("k", "v")
    assert store.get("k") == "v"


def test_set_unchanged_skips_write(cache_dir):
    store = JsonStore("g", "default")
    store.set("k", "v")
    store._write = MagicMock()
    store.set("k", "v")
    store._write.assert_not_called()
    assert store.get("k") == "v"


def test_flush_bounded_wait(cache_dir, caplog):
    store = JsonStore("cov-flush-bounded", "default")
    store._pending = True
    store._idle.clear()
    with caplog.at_level(logging.WARNING, logger="danyapi.store"):
        store.flush(timeout=0.05)
    assert any("timed out" in record.getMessage() for record in caplog.records)
    assert store._pending is True
    store._pending = False
    store._idle.set()


def test_flush_default_timeout_is_bounded():
    assert store_mod._FLUSH_MAX_WAIT == 5.0
    assert store_mod.JsonStore.flush.__defaults__ == (store_mod._FLUSH_MAX_WAIT,)


def test_load_logs_read_failure(cache_dir, monkeypatch, caplog):
    store = JsonStore("cov-read", "default")

    def boom(*args, **kwargs):
        raise OSError("denied")

    monkeypatch.setattr(Path, "read_text", boom)
    with caplog.at_level(logging.WARNING, logger="danyapi.store"):
        store._load()
    assert any("cache read failed" in record.getMessage() for record in caplog.records)
    assert store._data == {}


def test_load_missing_file_is_silent(cache_dir, caplog):
    store = JsonStore("cov-missing", "default")
    store._path.unlink(missing_ok=True)
    with caplog.at_level(logging.WARNING, logger="danyapi.store"):
        store._load()
    assert not [record for record in caplog.records if record.name == "danyapi.store"]


def test_load_logs_corrupt_and_unexpected_root(cache_dir, caplog):
    path = store_mod.cache_root() / "corrupt-load-sc.json"
    path.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="danyapi.store"):
        corrupt = JsonStore("corrupt-load", "sc")
    assert any("corrupt" in record.getMessage() for record in caplog.records)
    assert len(corrupt) == 0

    path2 = store_mod.cache_root() / "list-load-sc.json"
    path2.write_text("[1, 2]", encoding="utf-8")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="danyapi.store"):
        listed = JsonStore("list-load", "sc")
    assert any("list" in record.getMessage() for record in caplog.records)
    assert len(listed) == 0


def test_commit_writes_restrictive_file(cache_dir):
    store = JsonStore("cov-perm", "default")
    store.set("k", "v")
    assert store._path is not None
    assert store._path.exists()
    if os.name != "nt":
        assert oct(store._path.stat().st_mode & 0o777) == oct(0o600)


def test_clear_empty_returns_early(cache_dir):
    store = JsonStore("h", "default")
    store.clear()
    assert len(store) == 0


def test_unserializable_value_is_rejected(cache_dir):
    store = JsonStore("ns", "default")
    store.set("bad", object())
    assert "bad" not in store
    store.set("good", {"k": 1})
    assert store.get("good") == {"k": 1}


def test_unserializable_value_warns_and_leaves_no_tmp(cache_dir, caplog):
    store = JsonStore("ns2", "default")
    with caplog.at_level(logging.WARNING, logger="danyapi.store"):
        store.set("bad", object())
    assert any("not serialisable" in record.getMessage() for record in caplog.records)
    assert store._path is not None
    assert not (store._path.with_name(store._path.name + ".tmp")).exists()
    assert not list(store._path.parent.glob("*.tmp"))
