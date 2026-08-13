import json
import tempfile
import unittest
from pathlib import Path

from danyapi import store as store_mod
from danyapi.store import JsonStore


class JsonStoreTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = store_mod.settings.cache_dir
        store_mod.settings.cache_dir = self._tmp.name

    def tearDown(self):
        store_mod.settings.cache_dir = self._orig
        self._tmp.cleanup()


class TestCacheRoot(JsonStoreTestBase):
    def test_cache_root_uses_override(self):
        self.assertEqual(store_mod.cache_root(), Path(self._tmp.name))

    def test_cache_root_falls_back_to_temp(self):
        store_mod.settings.cache_dir = ""
        root = store_mod.cache_root()
        self.assertIsInstance(root, Path)
        self.assertTrue(root.is_absolute())


class TestJsonStoreBasic(JsonStoreTestBase):
    def test_disabled_without_scope(self):
        store = JsonStore("n", None)
        self.assertFalse(store.enabled)
        store.set("k", "v")
        self.assertIsNone(store.get("k2"))
        self.assertIn("k", store)

    def test_enabled_with_scope(self):
        store = JsonStore("sessions", "default")
        self.assertTrue(store.enabled)

    def test_set_get_roundtrip(self):
        store = JsonStore("a", "default")
        store.set("k", {"nested": [1, 2, 3]})
        self.assertEqual(store.get("k"), {"nested": [1, 2, 3]})

    def test_get_default(self):
        store = JsonStore("b", "default")
        self.assertIsNone(store.get("missing"))
        self.assertEqual(store.get("missing", 42), 42)

    def test_pop_existing_and_missing(self):
        store = JsonStore("c", "default")
        store.set("k", "v")
        self.assertEqual(store.pop("k"), "v")
        self.assertNotIn("k", store)
        self.assertEqual(store.pop("k", "dflt"), "dflt")

    def test_discard(self):
        store = JsonStore("d", "default")
        store.set("k", "v")
        store.discard("k")
        self.assertNotIn("k", store)
        store.discard("absent")

    def test_clear(self):
        store = JsonStore("e", "default")
        store.set("a", 1)
        store.set("b", 2)
        store.clear()
        self.assertEqual(len(store), 0)

    def test_items_len_contains(self):
        store = JsonStore("f", "default")
        store.set("a", 1)
        store.set("b", 2)
        self.assertEqual(sorted(store.items()), [("a", 1), ("b", 2)])
        self.assertEqual(len(store), 2)
        self.assertIn("a", store)
        self.assertNotIn("z", store)


class TestJsonStorePersistence(JsonStoreTestBase):
    def test_flush_writes_file(self):
        store = JsonStore("persist", "default")
        store.set("key", "value")
        assert store._path is not None
        raw = json.loads(store._path.read_text(encoding="utf-8"))
        self.assertEqual(raw, {"key": "value"})

    def test_reload_from_disk(self):
        store = JsonStore("reload", "default")
        store.set("key", "value")
        store2 = JsonStore("reload", "default")
        self.assertEqual(store2.get("key"), "value")

    def test_corrupted_json_ignored(self):
        path = store_mod.cache_root() / "corrupt-scope.json"
        path.write_text("not json", encoding="utf-8")
        store = JsonStore("corrupt", "scope")
        self.assertEqual(len(store), 0)

    def test_non_dict_json_ignored(self):
        path = store_mod.cache_root() / "list-scope.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        store = JsonStore("list", "scope")
        self.assertEqual(len(store), 0)

    def test_scope_sanitization(self):
        store = JsonStore("weird", "a b/c\\d")
        self.assertIsNotNone(store._path)
        assert store._path is not None
        self.assertTrue(store._path.name.endswith(".json"))


if __name__ == "__main__":
    unittest.main()
