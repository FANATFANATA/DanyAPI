from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from itertools import pairwise
from typing import Any

import pytest

import danyapi.accounts as accounts_mod
from danyapi import store as store_mod
from danyapi.accounts import AccountPool, AccountPoolBusy, ContextIndex, DeepSeekAccount, _free_account
from danyapi.store import _MAX_AFFINITY, JsonStore


class FakeClient:
    def __init__(self, index: int, check_auth: Any = None) -> None:
        self.index = index
        self._check_auth = check_auth

    async def check_auth(self) -> bool:
        if self._check_auth is None:
            return True
        return await self._check_auth()


class Gate:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.waiting = asyncio.Event()
        self.open = asyncio.Event()
        self.cancelled = False

    async def __call__(self) -> bool:
        self.waiting.set()
        try:
            await self.open.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return self.result


class SignallingLock:
    def __init__(self) -> None:
        self.attempted = threading.Event()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        self.attempted.set()
        return self._lock.acquire()

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> SignallingLock:
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


class FrozenClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def monotonic(self) -> float:
        return self.value


def make_acct(index: int, stable_id: str | None = None, check_auth: Any = None) -> DeepSeekAccount:
    client: Any = FakeClient(index, check_auth)
    return DeepSeekAccount(index, client, stable_id=stable_id)


def refuse() -> Any:
    async def run() -> bool:
        return False

    return run


def stale(acct: DeepSeekAccount) -> DeepSeekAccount:
    acct.broken = True
    acct.broken_at = time.monotonic() - AccountPool._REVIVE_COOLDOWN - 60
    return acct


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod.settings, "cache_dir", str(tmp_path))
    monkeypatch.setattr(store_mod.settings, "cache_enabled", True)
    return tmp_path


async def test_free_account_hands_out_the_free_slot_and_leaves_the_held_one_alone():
    a0, a1 = make_acct(0), make_acct(1)
    await a0.sem.acquire()
    ready = await _free_account([a0, a1], 5)
    assert [(acct.index, held) for acct, held in ready] == [(0, False), (1, True)]
    assert a1.sem.locked()
    assert a0.sem.locked()
    a1.sem.release()
    assert a1.sem.locked() is False
    a0.sem.release()


async def test_free_account_reports_a_cancelled_waiter_as_not_held():
    a0 = make_acct(0)
    await a0.sem.acquire()
    assert await _free_account([a0], 0) == [(a0, False)]
    assert a0.sem.locked()
    a0.sem.release()


async def test_a_spent_budget_raises_before_any_account_is_awaited():
    a0, a1 = make_acct(0), make_acct(1)
    pool = AccountPool([a0, a1])
    await a0.sem.acquire()
    await a1.sem.acquire()
    with pytest.raises(AccountPoolBusy):
        await pool.acquire(None, max_wait=-1)
    assert a0.sem.locked()
    assert a1.sem.locked()
    a0.sem.release()
    a1.sem.release()


async def test_two_concurrent_acquires_both_receive_the_revived_account():
    gate = Gate(True)
    a0 = stale(make_acct(0, check_auth=gate))
    stale(make_acct(1, check_auth=gate))
    pool = AccountPool([a0])

    first = asyncio.create_task(pool.acquire(None, max_wait=5))
    await gate.waiting.wait()
    assert pool._revive_lock.locked()

    second = asyncio.create_task(pool.acquire(None, max_wait=5))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not second.done()
    assert pool._revive_lock.locked()

    gate.open.set()
    first_acct, first_sid = await first
    second_acct, second_sid = await second
    assert first_acct is a0
    assert second_acct is a0
    assert first_sid is None
    assert second_sid is None
    assert a0.broken is False
    assert a0.broken_at is None
    assert pool._revive_lock.locked() is False


async def test_revive_broken_waits_for_the_pass_that_is_already_running():
    pool = AccountPool([make_acct(0), make_acct(1)])
    await pool._revive_lock.acquire()
    task = asyncio.create_task(pool.revive_broken())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not task.done()
    assert pool._revive_lock.locked()
    pool._revive_lock.release()
    assert await task is pool.accounts[0]
    assert pool._revive_lock.locked() is False


async def test_revive_broken_falls_back_to_the_first_healthy_account_when_the_wait_times_out():
    a0 = make_acct(0)
    stale(make_acct(1))
    pool = AccountPool([a0])
    await pool._revive_lock.acquire()
    assert await pool.revive_broken(max_wait=0) is a0
    assert pool._revive_lock.locked()
    pool._revive_lock.release()


async def test_revive_broken_finds_no_healthy_account_when_every_one_is_broken():
    pool = AccountPool([stale(make_acct(0))])
    await pool._revive_lock.acquire()
    assert await pool.revive_broken(max_wait=0) is None
    assert pool._revive_lock.locked()
    pool._revive_lock.release()


async def test_the_revive_pass_gives_up_once_its_deadline_is_spent(caplog):
    a0 = stale(make_acct(0, check_auth=refuse()))
    a1 = stale(make_acct(1, check_auth=refuse()))
    pool = AccountPool([a0, a1])
    with caplog.at_level(logging.WARNING, logger="danyapi.accounts"):
        assert await pool.revive_broken(max_wait=0) is None
    assert a0.broken is True
    assert a1.broken is True
    assert [record.getMessage() for record in caplog.records] == ["deepseek revive budget exhausted after 2 account(s)"]


async def test_a_hung_auth_recheck_is_cut_off_by_the_per_account_timeout(monkeypatch, caplog):
    gate = Gate(True)
    a0 = stale(make_acct(0, check_auth=gate))
    pool = AccountPool([a0])
    monkeypatch.setattr(AccountPool, "_REVIVE_AUTH_TIMEOUT", 0.01)
    with caplog.at_level(logging.WARNING, logger="danyapi.accounts"):
        task = asyncio.create_task(pool.revive_broken())
        await gate.waiting.wait()
        assert task.done() is False
        assert await task is None
    assert gate.cancelled is True
    assert a0.broken is True
    assert [record.getMessage() for record in caplog.records] == ["acct#0 auth recheck timed out"]
    gate.open.set()


async def test_the_revive_pass_revives_an_account_and_resets_its_cooldown(caplog):
    a0 = stale(make_acct(0))
    pool = AccountPool([a0])
    with caplog.at_level(logging.INFO, logger="danyapi.accounts"):
        assert await pool.revive_broken() is a0
    assert a0.broken is False
    assert a0.broken_at is None
    assert [record.getMessage() for record in caplog.records] == ["acct#0 revived after auth recheck"]


async def test_the_revive_pass_skips_an_account_with_no_client():
    a0 = stale(make_acct(0))
    missing: Any = None
    a0.client = missing
    pool = AccountPool([a0])
    assert await pool.revive_broken() is None
    assert a0.broken is True


async def test_the_revive_pass_skips_an_account_that_is_not_broken():
    healthy = make_acct(0)
    stale_refusal = stale(make_acct(1, check_auth=refuse()))
    pool = AccountPool([healthy, stale_refusal])
    assert await pool.revive_broken() is None
    assert healthy.broken is False
    assert stale_refusal.broken is True


async def test_wait_free_gives_up_when_the_preferred_account_breaks_while_waiting():
    a0 = make_acct(0)
    pool = AccountPool([a0])
    pool.register(0, "s1")
    await a0.sem.acquire()
    task = asyncio.create_task(pool.acquire("s1", max_wait=0.5))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not task.done()
    a0.broken = True
    a0.broken_at = time.monotonic()
    with pytest.raises(AccountPoolBusy):
        await asyncio.wait_for(task, timeout=2)
    a0.broken = False
    a0.sem.release()


async def test_wait_free_gives_up_immediately_when_no_account_is_healthy():
    a0 = make_acct(0)
    pool = AccountPool([a0])
    a0.broken = True
    a0.broken_at = time.monotonic()
    with pytest.raises(AccountPoolBusy):
        await pool._wait_free(a0, 5, "s1")
    assert a0.sem.locked() is False


class Exploding:
    def __init__(self) -> None:
        self.flushed = 0

    def items(self) -> list[tuple[str, Any]]:
        return []

    def get(self, key: str, default: Any = None) -> Any:
        return default

    def flush(self) -> None:
        self.flushed += 1
        raise OSError("disk gone")

    def discard(self, key: str) -> None:
        return None


def test_pool_flush_reports_every_store_that_fails(caplog):
    context = Exploding()
    affinity = Exploding()
    a0 = make_acct(0)
    without_sessions: Any = SimpleNamespace(index=1, broken=False, sem=asyncio.Semaphore(1), label="acct#1")
    with_sessions: Any = SimpleNamespace(index=2, broken=False, sem=asyncio.Semaphore(1), label="acct#2", sessions=Exploding())
    pool = AccountPool([a0], context_store=context, affinity_store=affinity)
    pool.accounts.extend([without_sessions, with_sessions])
    with caplog.at_level(logging.WARNING, logger="danyapi.accounts"):
        pool.flush()
    assert [record.getMessage() for record in caplog.records] == [
        "context store flush failed: disk gone",
        "affinity store flush failed: disk gone",
        "session store flush failed for acct#2: disk gone",
    ]
    assert context.flushed == 1
    assert affinity.flushed == 1


class SimpleNamespace:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


def test_an_affinity_record_of_an_unknown_shape_resolves_to_nothing(cache_dir):
    store = JsonStore("affshape", "default")
    store.set("list-record", ["alpha", 1])
    store.set("empty-list", [])
    store.set("float", 1.5)
    store.set("none", None)
    pool = AccountPool([make_acct(0, stable_id="alpha")], affinity_store=JsonStore("affshape", "default"))
    assert pool.account_for_session("list-record") is pool.accounts[0]
    for key in ("empty-list", "float", "none"):
        assert pool.account_for_session(key) is None
        assert key in store


async def test_the_revive_pass_skips_an_account_still_inside_its_cooldown():
    a0 = make_acct(0)
    a0.broken = True
    a0.broken_at = time.monotonic()
    pool = AccountPool([a0])
    assert await pool.revive_broken() is None
    assert a0.broken is True


async def test_a_failed_recheck_pushes_the_cooldown_out_again():
    a0 = stale(make_acct(0, check_auth=refuse()))
    pool = AccountPool([a0])
    before = time.monotonic()
    assert await pool.revive_broken() is None
    assert a0.broken is True
    assert a0.broken_at is not None
    assert before <= a0.broken_at
    assert a0.broken_at <= time.monotonic()


async def test_a_raising_auth_recheck_leaves_the_account_broken():
    async def boom() -> bool:
        raise RuntimeError("upstream 500")

    a0 = stale(make_acct(0, check_auth=boom))
    pool = AccountPool([a0])
    assert await pool.revive_broken() is None
    assert a0.broken is True


async def test_acquire_reports_a_busy_pool_once_the_revive_pass_fails():
    a0 = stale(make_acct(0, check_auth=refuse()))
    pool = AccountPool([a0])
    with pytest.raises(AccountPoolBusy):
        await pool.acquire(None, max_wait=0)
    assert a0.broken is True


async def test_acquire_forwards_its_budget_to_the_revive_pass(monkeypatch):
    a0 = stale(make_acct(0))
    pool = AccountPool([a0])
    seen: list[float | None] = []

    async def fake_revive(max_wait: float | None = None) -> Any:
        seen.append(max_wait)
        a0.broken = False
        return a0

    monkeypatch.setattr(pool, "revive_broken", fake_revive)
    acct, sid = await pool.acquire(None, max_wait=0.25)
    assert seen == [0.25]
    assert acct is a0
    assert sid is None


def test_register_ignores_an_index_that_is_not_a_plain_integer():
    pool = AccountPool([make_acct(0), make_acct(1)])
    boolean: Any = True
    text: Any = "1"
    number: Any = 1.0
    pool.register(boolean, "s1")
    pool.register(text, "s2")
    pool.register(number, "s3")
    assert pool._by_session == {}
    pool.register(1, "s4")
    assert list(pool._by_session) == ["s4"]
    assert pool.account_for_session("s4").index == 1


def test_register_ignores_an_index_outside_the_pool_and_an_empty_session_id():
    pool = AccountPool([make_acct(0)])
    pool.register(5, "s1")
    pool.register(-1, "s2")
    pool.register(0, "")
    assert pool._by_session == {}
    pool.register(0, "s3")
    assert list(pool._by_session) == ["s3"]


def test_an_account_with_a_stable_id_is_persisted_by_name(cache_dir):
    store = JsonStore("aff-stable", "default")
    AccountPool([make_acct(0, stable_id="alpha")], affinity_store=store).register(0, "s1")
    assert store.get("s1") == "alpha"
    store.set("s1", 0)
    pool = AccountPool([make_acct(0, stable_id="alpha")], affinity_store=JsonStore("aff-stable", "default"))
    assert pool.account_for_session("s1") is not None


def test_restore_affinities_stops_once_the_cap_is_reached(cache_dir):
    path = store_mod.cache_root() / "affcap-default.json"
    path.write_text(json.dumps({f"s{index}": 0 for index in range(_MAX_AFFINITY + 1)}), encoding="utf-8")
    pool = AccountPool([make_acct(0)], affinity_store=JsonStore("affcap", "default"))
    assert len(pool._by_session) == _MAX_AFFINITY
    assert f"s{_MAX_AFFINITY}" not in pool._by_session
    assert f"s{_MAX_AFFINITY - 1}" in pool._by_session


def test_restore_keeps_affinity_records_it_cannot_resolve_yet(cache_dir):
    store = JsonStore("affpending", "default")
    store.set("s1", 0)
    store.set("pending-name", "an-account-this-process-does-not-have")
    store.set("out-of-range", 9)
    pool = AccountPool([make_acct(0, stable_id="alpha")], affinity_store=JsonStore("affpending", "default"))
    assert pool.account_for_session("s1") is not None
    assert pool.account_for_session("pending-name") is None
    assert pool.account_for_session("out-of-range") is None
    assert sorted(store.items()) == [
        ("out-of-range", 9),
        ("pending-name", "an-account-this-process-does-not-have"),
        ("s1", 0),
    ]
    store.set("pending-name", "alpha")
    restarted = AccountPool([make_acct(0, stable_id="alpha")], affinity_store=JsonStore("affpending", "default"))
    assert restarted.account_for_session("pending-name") is not None


def test_a_restored_affinity_is_stamped_with_the_clock_rather_than_its_write_time(cache_dir, monkeypatch):
    store = JsonStore("affstamp", "default")
    store.set("s1", 0)
    monkeypatch.setattr(accounts_mod, "time", FrozenClock(4242.0))
    pool = AccountPool([make_acct(0)], ttl=0.0001, affinity_store=JsonStore("affstamp", "default"))
    assert pool._by_session["s1"][1] == 4242.0
    assert pool.account_for_session("s1") is not None


def test_a_restored_context_is_stamped_with_the_clock_rather_than_its_write_time(cache_dir, monkeypatch):
    store = JsonStore("ctxstamp", "default")
    store.set("s1", ["a"])
    monkeypatch.setattr(accounts_mod, "time", FrozenClock(4242.0))
    idx = ContextIndex(16, ttl=0.0001, store=JsonStore("ctxstamp", "default"))
    assert idx._ts["s1"] == 4242.0
    assert idx.lookup(("a",)) == "s1"


def test_context_restore_discards_records_that_can_never_resolve(cache_dir):
    store = JsonStore("ctxjunk", "default")
    store.set("good", ["a"])
    store.set("not-a-list", 42)
    store.set("no-strings", [1, 2, 3])
    store.set("empty", [])
    store.set("", ["x"])
    store.set("partial", ["a", 5, "b"])
    idx = ContextIndex(16, store=JsonStore("ctxjunk", "default"))
    assert idx.lookup(("a",)) == "good"
    assert idx.lookup(("a", "b")) == "partial"
    assert idx.lookup(("x",)) is None
    assert sorted(store.items()) == [("good", ["a"]), ("partial", ["a", 5, "b"])]


def test_a_second_restore_does_not_bring_the_discarded_records_back(cache_dir):
    store = JsonStore("ctxjunk2", "default")
    store.set("junk", 42)
    store.set("worse", "text")
    ContextIndex(16, store=JsonStore("ctxjunk2", "default"))
    assert len(store) == 0
    ContextIndex(16, store=JsonStore("ctxjunk2", "default"))
    assert len(store) == 0


def test_lookup_against_an_empty_index_counts_one_miss_and_keeps_the_store(cache_dir):
    store = JsonStore("ctxempty", "default")
    store.set("unrelated", ["z"])
    idx = ContextIndex(16, store=store)
    assert idx.lookup(("a",)) is None
    assert idx.hits == 0
    assert idx.misses == 1
    assert store.get("unrelated") == ["z"]


def test_context_resolution_through_the_pool(cache_dir):
    pool = AccountPool([make_acct(0)], context_store=JsonStore("ctxpool", "default"))
    pool.index_context("s1", ("a", "b"))
    assert pool.resolve_context(("a", "b", "c")) == "s1"
    stats = pool.stats()
    assert stats["context_hits"] == 1
    assert stats["context_entries"] == 1
    assert stats["context_misses"] == 0
    pool.forget_context("s1")
    assert pool.resolve_context(("a", "b", "c")) is None
    assert pool.stats()["context_entries"] == 0
    assert pool.stats()["context_misses"] == 1


def test_context_counters_stay_exact_while_stats_reads_them_unlocked():
    pool = AccountPool([make_acct(0)])
    pool.index_context("s1", ("a",))
    rounds = 250
    counted = [[0, 0], [0, 0]]

    def hammer(slot: int) -> None:
        for _ in range(rounds):
            assert pool.resolve_context(("a",)) == "s1"
            counted[slot][0] += 1
            assert pool.resolve_context(("zzz",)) is None
            counted[slot][1] += 1

    workers = [threading.Thread(target=hammer, args=(slot,)) for slot in range(2)]
    for worker in workers:
        worker.start()
    samples = [pool.stats()["context_hits"] for _ in range(rounds * 4)]
    for worker in workers:
        worker.join(10)
    assert all(later >= earlier for earlier, later in pairwise(samples))
    assert pool.stats()["context_hits"] == rounds * 2
    assert pool.stats()["context_misses"] == rounds * 2
    assert sum(count[0] for count in counted) == rounds * 2
    assert sum(count[1] for count in counted) == rounds * 2


def test_add_account_holds_the_affinity_lock_for_the_whole_mutation():
    a0 = make_acct(0, stable_id="alpha")
    pool = AccountPool([a0])
    probe = SignallingLock()
    pool._affinity_lock = probe
    landed: list[str] = []
    probe.acquire()

    def worker() -> None:
        pool.add_account(make_acct(1, stable_id="beta"))
        landed.append("added")

    thread = threading.Thread(target=worker)
    thread.start()
    assert probe.attempted.wait(2)
    thread.join(0.05)
    assert thread.is_alive()
    assert landed == []
    assert pool.accounts == [a0]
    assert pool._stable_to_idx == {"alpha": 0}

    probe.release()
    thread.join(5)
    assert landed == ["added"]
    assert pool._stable_to_idx == {"alpha": 0, "beta": 1}
    assert pool.stats()["accounts"] == 2
    assert pool.stats()["healthy"] == 2


async def test_an_account_added_at_runtime_is_acquirable_by_its_stable_id():
    a0 = make_acct(0, stable_id="alpha")
    pool = AccountPool([a0])
    a1 = make_acct(1, stable_id="beta")
    pool.add_account(a1)
    assert pool.healthy == [a0, a1]
    pool.register(1, "s1")
    assert pool.account_for_session("s1") is a1
    acct, sid = await pool.acquire("s1")
    assert acct is a1
    assert sid == "s1"


async def test_an_account_added_at_runtime_without_a_stable_id_is_registered_by_index():
    pool = AccountPool([make_acct(0)])
    anonymous = make_acct(1)
    pool.add_account(anonymous)
    assert pool._stable_to_idx == {}
    assert pool._affinity_record(1) == 1
    pool.register(1, "s1")
    assert pool.account_for_session("s1") is anonymous
    acct, sid = await pool.acquire("s1")
    assert acct is anonymous
    assert sid == "s1"
