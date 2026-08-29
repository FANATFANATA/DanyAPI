import pytest

from danyapi import store as store_mod
from danyapi.store import JsonStore
from danyapi.usage import UsageTracker, get_tracker, init_tracker, record_usage, reset_tracker


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod.settings, "cache_dir", str(tmp_path))
    return tmp_path


def test_record_and_snapshot():
    tracker = UsageTracker()
    tracker.record("deepseek", "deepseek-v4-flash", 10, 20, 30)
    tracker.record("qwen", "qwen3.8-max", 5, 7, 12, user="alice")
    snap = tracker.snapshot()
    assert snap["totals"] == {"requests": 2, "prompt_tokens": 15, "completion_tokens": 27, "total_tokens": 42}
    assert snap["by_model"]["deepseek-v4-flash"]["requests"] == 1
    assert snap["by_model"]["deepseek-v4-flash"]["total_tokens"] == 30
    assert snap["by_provider"]["qwen"]["total_tokens"] == 12
    assert snap["by_user"]["alice"]["requests"] == 1
    assert len(snap["recent"]) == 2
    assert snap["recent"][0]["model"] == "deepseek-v4-flash"
    assert snap["recent"][1]["user"] == "alice"


def test_record_clamps_negative():
    tracker = UsageTracker()
    tracker.record("deepseek", "m", -5, -3, -8)
    snap = tracker.snapshot()
    assert snap["totals"] == {"requests": 1, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_record_fills_total():
    tracker = UsageTracker()
    tracker.record("deepseek", "m", 10, 20, 0)
    snap = tracker.snapshot()
    assert snap["totals"]["total_tokens"] == 30


def test_max_records_bounded():
    tracker = UsageTracker(max_records=3)
    for _ in range(10):
        tracker.record("deepseek", "m", 1, 1, 2)
    snap = tracker.snapshot()
    assert len(snap["recent"]) == 3
    assert snap["totals"]["requests"] == 10


def test_persist_restore(cache_dir):
    store = JsonStore("usage-test", "default")
    tracker = UsageTracker(store=store)
    tracker.record("deepseek", "m", 10, 20, 30, user="bob")
    tracker2 = UsageTracker(store=store)
    snap = tracker2.snapshot()
    assert snap["totals"]["requests"] == 1
    assert snap["totals"]["total_tokens"] == 30
    assert snap["by_user"]["bob"]["requests"] == 1


def test_reset(cache_dir):
    store = JsonStore("usage-reset", "default")
    tracker = UsageTracker(store=store)
    tracker.record("deepseek", "m", 1, 1, 2)
    tracker.reset()
    snap = tracker.snapshot()
    assert snap["totals"]["requests"] == 0
    assert snap["recent"] == []
    assert store.get("usage") is None


def test_module_singleton():
    reset_tracker()
    assert get_tracker() is None
    tracker = init_tracker()
    assert get_tracker() is tracker
    record_usage("deepseek", "m", 1, 1, 2)
    assert get_tracker().snapshot()["totals"]["requests"] == 1
    reset_tracker()
    assert get_tracker() is None


def test_record_usage_no_tracker():
    reset_tracker()
    record_usage("deepseek", "m", 1, 1, 2)
    assert get_tracker() is None
