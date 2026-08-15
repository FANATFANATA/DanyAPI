import asyncio
import time
from unittest.mock import MagicMock

import pytest

from danyapi.accounts import AccountPool, AccountPoolBusy, ContextIndex, DeepSeekAccount, account_lock
from danyapi.deepseek.client import DeepSeekClient
from danyapi.store import JsonStore


def make_acct(i):
    client = MagicMock(spec=DeepSeekClient)
    client.index = i
    return DeepSeekAccount(i, client)


async def test_no_accounts_raises():
    pool = AccountPool([])
    with pytest.raises(RuntimeError):
        await pool.acquire(None)


async def test_session_affinity():
    a0, a1 = make_acct(0), make_acct(1)
    pool = AccountPool([a0, a1])
    pool.register(1, "sess-abc")
    acct, sid = await pool.acquire("sess-abc")
    assert acct is a1
    assert sid == "sess-abc"
    acct, sid = await pool.acquire("sess-zzz")
    assert sid is None


async def test_broken_account_releases_session():
    a0, a1 = make_acct(0), make_acct(1)
    a0.broken = True
    pool = AccountPool([a0, a1])
    pool.register(0, "sess-broken")
    acct, sid = await pool.acquire("sess-broken")
    assert acct is a1
    assert sid is None
    assert "sess-broken" not in pool._by_session


async def test_round_robin_frees_first():
    a0, a1, a2 = make_acct(0), make_acct(1), make_acct(2)
    pool = AccountPool([a0, a1, a2])
    await a1.sem.acquire()
    await asyncio.sleep(0)
    assert a1.sem.locked()
    acct, _ = await pool.acquire(None)
    assert acct is a0
    await a0.sem.acquire()
    await asyncio.sleep(0)
    acct, _ = await pool.acquire(None)
    assert acct is a2
    await a2.sem.acquire()
    await asyncio.sleep(0)
    acct, _ = await pool.acquire(None)
    assert acct is not None
    a1.sem.release()
    a0.sem.release()
    a2.sem.release()


async def test_acquire_waits_for_free_account():
    a0, a1 = make_acct(0), make_acct(1)
    pool = AccountPool([a0, a1])
    await a0.sem.acquire()
    await a1.sem.acquire()

    async def acquire():
        return await pool.acquire(None, max_wait=5)

    task = asyncio.create_task(acquire())
    await asyncio.sleep(0.1)
    assert not task.done()
    a1.sem.release()
    acct, sid = await asyncio.wait_for(task, timeout=2)
    assert acct is a1
    assert sid is None
    a0.sem.release()


async def test_acquire_times_out_when_busy():
    a0, a1 = make_acct(0), make_acct(1)
    pool = AccountPool([a0, a1])
    await a0.sem.acquire()
    await a1.sem.acquire()
    with pytest.raises(AccountPoolBusy):
        await pool.acquire(None, max_wait=0.2)
    a0.sem.release()
    a1.sem.release()


async def test_session_affinity_waits_for_its_account():
    a0, a1 = make_acct(0), make_acct(1)
    pool = AccountPool([a0, a1])
    pool.register(1, "sess-x")
    await a1.sem.acquire()

    async def acquire():
        return await pool.acquire("sess-x", max_wait=5)

    task = asyncio.create_task(acquire())
    await asyncio.sleep(0.1)
    assert not task.done()
    a1.sem.release()
    acct, sid = await asyncio.wait_for(task, timeout=2)
    assert acct is a1
    assert sid == "sess-x"


async def test_account_lock_no_timeout():
    sem = asyncio.Semaphore(1)
    async with account_lock(sem):
        assert sem.locked()
    assert not sem.locked()


async def test_account_lock_timeout_free():
    sem = asyncio.Semaphore(1)
    async with account_lock(sem, 1.0):
        assert sem.locked()
    assert not sem.locked()


async def test_account_lock_timeout_busy():
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    try:
        with pytest.raises(AccountPoolBusy):
            async with account_lock(sem, 0.05):
                pass
    finally:
        sem.release()


async def test_account_lock_exception_releases():
    sem = asyncio.Semaphore(1)
    with pytest.raises(ValueError):
        async with account_lock(sem, 1.0):
            raise ValueError("boom")
    assert not sem.locked()


@pytest.fixture
def ctx_store(tmp_path, monkeypatch):
    import danyapi.store as store_mod

    monkeypatch.setattr(store_mod.settings, "cache_dir", str(tmp_path))
    return JsonStore("ctx-persist", "default")


def test_restore_from_store(ctx_store):
    ctx_store.set("s1", ["a"])
    ctx_store.set("s2", ["b"])
    idx = ContextIndex(16, store=ctx_store)
    assert idx.lookup(("a",)) == "s1"
    assert idx.lookup(("b",)) == "s2"


def test_index_persists(ctx_store):
    idx = ContextIndex(16, store=ctx_store)
    idx.index("s1", ("a",))
    idx2 = ContextIndex(16, store=JsonStore("ctx-persist", "default"))
    assert idx2.lookup(("a",)) == "s1"


def test_eviction_discards_from_store(ctx_store):
    idx = ContextIndex(2, store=ctx_store)
    idx.index("s1", ("a",))
    idx.index("s2", ("b",))
    idx.index("s3", ("c",))
    assert "s1" not in ctx_store


def test_forget_discards_from_store(ctx_store):
    idx = ContextIndex(16, store=ctx_store)
    idx.index("s1", ("a",))
    idx.forget("s1")
    assert "s1" not in ctx_store
    assert idx.lookup(("a",)) is None


def test_expired_discards_from_store(ctx_store):
    idx = ContextIndex(16, ttl=0.1, store=ctx_store)
    idx.index("s1", ("a",))
    idx._ts["s1"] = time.monotonic() - 10
    assert idx.lookup(("a",)) is None
    assert "s1" not in ctx_store


def test_index_touches_superset():
    idx = ContextIndex(16)
    idx.index("s1", ("a", "b"))
    idx.index("s1", ("a",))
    assert idx.lookup(("a", "b", "c")) == "s1"


def test_index_supersedes_shorter():
    idx = ContextIndex(16)
    idx.index("s1", ("a",))
    idx.index("s1", ("a", "b", "c"))
    assert idx.lookup(("a", "b", "c")) == "s1"


def test_empty_index_counts_miss():
    idx = ContextIndex(16)
    assert idx.lookup(("a",)) is None
    assert idx.misses == 1


def test_min_maxsize():
    assert ContextIndex(0)._maxsize == 1
    assert ContextIndex(-5)._maxsize == 1


def test_restore_skips_garbage(ctx_store):
    ctx_store.set("s1", ["a"])
    ctx_store.set("bad", 42)
    ctx_store.set("", ["x"])
    idx = ContextIndex(16, store=ctx_store)
    assert idx.lookup(("a",)) == "s1"
    assert idx.lookup(("x",)) is None


def test_exact_match_reuses():
    idx = ContextIndex(16)
    idx.index("s1", ("a",))
    assert idx.lookup(("a",)) == "s1"


def test_continuation_matches_prefix():
    idx = ContextIndex(16)
    idx.index("s1", ("a",))
    idx.index("s2", ("x",))
    assert idx.lookup(("a", "b")) == "s1"
    assert idx.lookup(("x", "y", "z")) == "s2"


def test_prefers_longest_prefix():
    idx = ContextIndex(16)
    idx.index("s1", ("a",))
    idx.index("s2", ("a", "b"))
    assert idx.lookup(("a", "b")) == "s2"
    assert idx.lookup(("a", "b", "c")) == "s2"


def test_no_match():
    idx = ContextIndex(16)
    idx.index("s1", ("a",))
    assert idx.lookup(("c",)) is None
    assert idx.lookup(()) is None


def test_eviction():
    idx = ContextIndex(2)
    idx.index("s1", ("a",))
    idx.index("s2", ("b",))
    idx.index("s3", ("c",))
    assert idx.lookup(("a",)) is None
    assert idx.lookup(("b",)) == "s2"


def test_recency_tiebreak():
    idx = ContextIndex(16)
    idx.index("s1", ("a",))
    idx.index("s2", ("a",))
    assert idx.lookup(("a",)) == "s2"


def test_stats_counters():
    idx = ContextIndex(16)
    idx.index("s1", ("a",))
    assert idx.lookup(("x",)) is None
    assert idx.lookup(("a",)) == "s1"
    assert idx.hits == 1
    assert idx.misses == 1


@pytest.fixture
def pool_store(tmp_path, monkeypatch):
    import danyapi.store as store_mod

    monkeypatch.setattr(store_mod.settings, "cache_dir", str(tmp_path))


def test_register_persists_affinity(pool_store):
    store = JsonStore("aff", "default")
    AccountPool([make_acct(0)], affinity_store=store).register(0, "s1")
    pool2 = AccountPool([make_acct(0)], affinity_store=JsonStore("aff", "default"))
    assert pool2.account_for_session("s1") is not None


def test_register_ttl_cleanup():
    pool = AccountPool([make_acct(0)], ttl=0.05)
    for i in range(4097):
        pool._by_session[f"s{i}"] = (0, time.monotonic())
    pool._by_session["stale"] = (0, time.monotonic() - 10)
    pool.register(0, "fresh")
    assert "stale" not in pool._by_session
    assert "fresh" in pool._by_session


def test_account_for_session_expired():
    pool = AccountPool([make_acct(0)], ttl=0.05)
    pool.register(0, "s1")
    pool._by_session["s1"] = (0, time.monotonic() - 10)
    assert pool.account_for_session("s1") is None
    assert "s1" not in pool._by_session


def test_account_for_session_broken():
    a0 = make_acct(0)
    a0.broken = True
    pool = AccountPool([a0])
    pool.register(0, "s1")
    assert pool.account_for_session("s1") is None
    assert "s1" not in pool._by_session


def test_stats():
    pool = AccountPool([make_acct(0), make_acct(1)])
    pool.register(0, "s1")
    pool.index_context("s1", ("a",))
    stats = pool.stats()
    assert stats["accounts"] == 2
    assert stats["session_affinities"] == 1
    assert stats["context_entries"] == 1
    assert stats["context_cache_size"] == 128


def test_context_forget():
    pool = AccountPool([make_acct(0)])
    pool.index_context("s1", ("a",))
    assert pool.resolve_context(("a",)) == "s1"
    pool.forget_context("s1")
    assert pool.resolve_context(("a",)) is None


async def test_acquire_no_healthy():
    a0 = make_acct(0)
    a0.broken = True
    pool = AccountPool([a0])
    with pytest.raises(RuntimeError):
        await pool.acquire(None)


def test_mark_broken_idempotent():
    a0 = make_acct(0)
    a0.mark_broken()
    a0.mark_broken()
    assert a0.broken
    assert a0.label == "acct#0"


def test_resolve_routes_to_owner():
    a0, a1 = make_acct(0), make_acct(1)
    pool = AccountPool([a0, a1])
    pool.register(1, "sess-a")
    pool.index_context("sess-a", ("u1",))
    assert pool.resolve_context(("u1",)) == "sess-a"


def test_forget_removes_mapping():
    pool = AccountPool([make_acct(0)])
    pool.register(0, "s1")
    assert pool.account_for_session("s1").index == 0
    pool.forget("s1")
    assert pool.account_for_session("s1") is None


def test_restore_skips_empty_sequence(ctx_store):
    ctx_store.set("s1", [])
    ctx_store.set("s2", ["a"])
    ctx_store.set("", ["x"])
    ctx_store.set("s3", 42)
    idx = ContextIndex(16, store=ctx_store)
    assert idx.lookup(("a",)) == "s2"


def test_lookup_skips_longer_sequences():
    idx = ContextIndex(16)
    idx.index("s1", ("a", "b"))
    assert idx.lookup(("a",)) is None


def test_index_ignores_empty():
    idx = ContextIndex(16)
    idx.index("", ("a",))
    idx.index("s1", ())
    assert len(idx._seqs) == 0


def test_register_ttl_cleanup_with_store(pool_store):
    store = JsonStore("aff2", "default")
    pool = AccountPool([make_acct(0)], ttl=0.05, affinity_store=store)
    for i in range(4097):
        pool._by_session[f"s{i}"] = (0, time.monotonic())
    pool._by_session["stale"] = (0, time.monotonic() - 10)
    pool.register(0, "fresh")
    assert "stale" not in store


def test_forget_discards_affinity(pool_store):
    store = JsonStore("aff3", "default")
    pool = AccountPool([make_acct(0)], affinity_store=store)
    pool.register(0, "s1")
    assert "s1" in store
    pool.forget("s1")
    assert "s1" not in store


def test_account_for_session_expired_discards(pool_store):
    store = JsonStore("aff4", "default")
    pool = AccountPool([make_acct(0)], ttl=0.05, affinity_store=store)
    pool.register(0, "s1")
    pool._by_session["s1"] = (0, time.monotonic() - 10)
    assert pool.account_for_session("s1") is None
    assert "s1" not in store


def test_restore_affinities_variants(pool_store):
    store = JsonStore("aff5", "default")
    store.set("good", 0)
    store.set("as-list", [0])
    store.set("bad-bool", True)
    store.set("bad-str", "x")
    store.set("out-of-range", 5)
    store.set("", 0)
    pool = AccountPool([make_acct(0)], affinity_store=store)
    assert pool.account_for_session("good") is not None
    assert pool.account_for_session("as-list") is not None
    assert pool.account_for_session("bad-bool") is None
    assert pool.account_for_session("bad-str") is None
    assert pool.account_for_session("out-of-range") is None
