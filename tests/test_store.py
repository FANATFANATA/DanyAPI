import json
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


def test_clear_empty_returns_early(cache_dir):
    store = JsonStore("h", "default")
    store.clear()
    assert len(store) == 0
