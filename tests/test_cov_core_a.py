import asyncio
import os
import threading
import time
from pathlib import Path

import pytest

from danyapi import store as store_mod
from danyapi.accounts import AccountPool, AccountPoolBusy, ContextIndex
from danyapi.store import _MAX_AFFINITY, JsonStore
from danyapi.tokens import StreamBudget, count_message_tokens, estimate_tokens


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod.settings, "cache_dir", str(tmp_path))
    return tmp_path


class FakeClient:
    def __init__(self, ok=True, exc=False):
        self.ok = ok
        self.exc = exc

    async def check_auth(self):
        if self.exc:
            raise RuntimeError("nope")
        return self.ok


class FakeAccount:
    def __init__(self, index, stable_id=None, broken=False, broken_at=None, client=None):
        self.index = index
        self.stable_id = stable_id
        self.broken = broken
        self.broken_at = broken_at
        self.sem = asyncio.Semaphore(1)
        self.client = FakeClient() if client is None else client

    @property
    def label(self):
        return f"acct#{self.index}"


class BadSessions:
    def flush(self):
        raise RuntimeError("session flush failed")


class BadContextStore:
    def items(self):
        return []

    def flush(self):
        raise RuntimeError("context flush failed")

    def discard(self, key):
        pass

    def set(self, key, value):
        pass


def _trim(text, budget):
    if budget is None or estimate_tokens(text) <= budget:
        return text
    return text


def test_context_index_flush_store_error():
    ContextIndex(16, store=BadContextStore()).flush()


def test_affinity_record_uses_stable_id(cache_dir):
    store = JsonStore("cov-aff-stable", "default")
    pool = AccountPool([FakeAccount(0, stable_id="stable-0")], affinity_store=store)
    pool.register(0, "s1")
    assert store.get("s1") == "stable-0"


def test_restore_affinity_bad_record(cache_dir):
    store = JsonStore("cov-aff-bad", "default")
    store.set("s1", {"nested": 1})
    pool = AccountPool([FakeAccount(0)], affinity_store=store)
    assert pool.account_for_session("s1") is None


def test_register_evicts_oldest():
    pool = AccountPool([FakeAccount(0)])
    now = time.monotonic()
    for i in range(_MAX_AFFINITY + 1):
        pool._by_session[f"s{i}"] = (0, now)
    pool.register(0, "new")
    assert "new" in pool._by_session
    assert len(pool._by_session) <= _MAX_AFFINITY


def test_account_for_session_refreshes_ts():
    pool = AccountPool([FakeAccount(0)], ttl=100.0)
    pool.register(0, "s1")
    time.sleep(0.01)
    assert pool.account_for_session("s1") is not None


def test_pool_flush_store_errors(cache_dir, monkeypatch):
    aff = JsonStore("cov-aff-err", "default")

    def boom():
        raise RuntimeError("affinity flush failed")

    monkeypatch.setattr(aff, "flush", boom)
    a0 = FakeAccount(0)
    a1 = FakeAccount(1)
    a1.sessions = BadSessions()
    pool = AccountPool([a0, a1], affinity_store=aff)
    pool.flush()


async def test_acquire_busy_when_revive_fails():
    a0 = FakeAccount(0, broken=True, broken_at=time.monotonic())
    pool = AccountPool([a0])
    with pytest.raises(AccountPoolBusy):
        await pool.acquire(None)


async def test_acquire_after_revive():
    a0 = FakeAccount(0, broken=True, broken_at=time.monotonic() - 1000.0)
    pool = AccountPool([a0])
    acct, sid = await pool.acquire(None)
    assert acct is a0
    assert sid is None
    assert a0.broken is False
    assert a0.broken_at is None


async def test_revive_skips_no_client():
    a0 = FakeAccount(0, broken=True, broken_at=time.monotonic() - 1000.0, client=None)
    a0.client = None
    pool = AccountPool([a0])
    assert await pool.revive_broken() is None


async def test_revive_auth_exception():
    a0 = FakeAccount(0, broken=True, broken_at=time.monotonic() - 1000.0, client=FakeClient(exc=True))
    pool = AccountPool([a0])
    assert await pool.revive_broken() is None


async def test_revive_skips_healthy():
    pool = AccountPool([FakeAccount(0)])
    assert await pool.revive_broken() is None


async def test_wait_free_busy_no_candidates():
    a0 = FakeAccount(0, broken=True, broken_at=time.monotonic())
    pool = AccountPool([a0])
    with pytest.raises(AccountPoolBusy):
        await pool._wait_free(a0, 1.0, None)


def test_add_account():
    pool = AccountPool([FakeAccount(0)])
    a1 = FakeAccount(1, stable_id="stable-1")
    pool.add_account(a1)
    assert pool.accounts[-1] is a1
    assert pool._stable_to_idx["stable-1"] == 1
    a2 = FakeAccount(2)
    pool.add_account(a2)
    assert pool.accounts[-1] is a2


def test_stream_budget_count_cjk_only():
    budget = StreamBudget(5, _trim)
    assert budget.feed("你好") == "你好"
    assert budget._count() == 2


def test_stream_budget_trim_returns_full_text():
    budget = StreamBudget(1, lambda text, limit: text)
    assert budget.feed("Hello world") == "Hello world"


def test_stream_budget_trim_not_prefix():
    budget = StreamBudget(1, lambda text, limit: "X")
    assert budget.feed("abcd") == "abcd"
    assert budget.feed("efgh") == ""


class _Msg:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


def test_count_message_tokens_object():
    msg = _Msg("Hello")
    assert count_message_tokens(msg) == 3 + estimate_tokens("Hello")


def test_count_message_tokens_list_of_str():
    assert count_message_tokens({"content": ["Hello"]}) == 3 + estimate_tokens("Hello")


def test_count_message_tokens_tool_calls_non_dict():
    assert count_message_tokens({"content": "", "tool_calls": ["x"]}) == 3


def test_evict_on_set(cache_dir):
    store = JsonStore("cov-evict", "default", maxsize=1)
    store.set("a", 1)
    store.set("b", 2)
    assert len(store) == 1
    assert store.get("a") is None
    assert store.get("b") == 2


def test_commit_disabled_noop():
    store = JsonStore("cov-commit-off", None)
    store._commit({"a": 1})
    assert store._data == {}


def test_commit_unlink_error(cache_dir, monkeypatch):
    store = JsonStore("cov-unlink", "default")

    def boom(*args, **kwargs):
        raise OSError("denied")

    monkeypatch.setattr(os, "replace", boom)
    monkeypatch.setattr(Path, "unlink", boom)
    store.set("k", "v")
    assert store.get("k") == "v"


def test_write_disabled_noop():
    JsonStore("cov-write-off", None)._write()


async def test_note_changed_executor_failure(cache_dir, monkeypatch):
    store = JsonStore("cov-exec", "default")
    loop = asyncio.get_running_loop()

    def boom(*args, **kwargs):
        raise RuntimeError("no executor")

    monkeypatch.setattr(loop, "run_in_executor", boom)
    store.set("k", "v")
    assert store.get("k") == "v"
    assert store._pending is False


def test_flush_waits_for_pending(cache_dir):
    store = JsonStore("cov-flush", "default")
    store._pending = True
    store._idle.clear()
    thread = threading.Thread(target=store.flush)
    thread.start()
    time.sleep(0.05)
    store._pending = False
    store._idle.set()
    thread.join(2.0)
    assert not thread.is_alive()
