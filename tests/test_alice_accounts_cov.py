import logging
import time
from typing import Any

from danyapi.alice.accounts import AliceAccount
from danyapi.alice.client import AliceClient


class _ScriptedClock:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)
        self.calls = 0

    def monotonic(self) -> float:
        self.calls += 1
        if not self.values:
            raise AssertionError("the account clock was read more times than the test scripted")
        return self.values.pop(0)

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def test_account_surface_is_pinned_to_slots_without_a_shared_base() -> None:
    assert AliceAccount.__slots__ == ("broken", "broken_at", "client", "index", "sem", "stable_id")


def test_fresh_account_is_healthy_and_carries_its_identity() -> None:
    client = AliceClient(timeout=12.0)
    account = AliceAccount(index=3, client=client, stable_id="stable-3")
    assert account.index == 3
    assert account.client is client
    assert account.stable_id == "stable-3"
    assert account.broken is False
    assert account.broken_at is None
    assert account.sem.locked() is False


def test_account_without_a_stable_id_keeps_none() -> None:
    assert AliceAccount(index=0, client=AliceClient()).stable_id is None


def test_label_names_the_provider_and_index() -> None:
    assert AliceAccount(index=7, client=AliceClient()).label == "alice-acct#7"
    assert AliceAccount(index=0, client=AliceClient()).label == "alice-acct#0"


def test_mark_broken_stamps_the_clock_and_warns(monkeypatch, caplog) -> None:
    clock = _ScriptedClock([1234.5])
    monkeypatch.setattr("danyapi.alice.accounts.time", clock)
    account = AliceAccount(index=2, client=AliceClient())
    with caplog.at_level(logging.WARNING, logger="danyapi.alice"):
        account.mark_broken()
    assert account.broken is True
    assert account.broken_at == 1234.5
    assert clock.calls == 1
    assert caplog.messages == ["alice account #2 marked broken"]


def test_mark_broken_is_idempotent_and_keeps_the_first_stamp(monkeypatch, caplog) -> None:
    clock = _ScriptedClock([10.0, 20.0])
    monkeypatch.setattr("danyapi.alice.accounts.time", clock)
    account = AliceAccount(index=1, client=AliceClient())
    with caplog.at_level(logging.WARNING, logger="danyapi.alice"):
        account.mark_broken()
        account.mark_broken()
    assert account.broken is True
    assert account.broken_at == 10.0
    assert clock.calls == 1
    assert caplog.messages == ["alice account #1 marked broken"]


async def test_mark_broken_does_not_touch_the_account_semaphore() -> None:
    account = AliceAccount(index=0, client=AliceClient())
    await account.sem.acquire()
    account.mark_broken()
    assert account.sem.locked() is True
    account.sem.release()
    assert account.sem.locked() is False
