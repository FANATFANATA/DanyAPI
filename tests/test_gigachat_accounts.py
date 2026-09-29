from __future__ import annotations

import asyncio
import base64
import logging
import time

import pytest

from danyapi.gigachat.accounts import GigaChatAccount
from danyapi.gigachat.client import GigaChatClient

KEY = base64.b64encode(b"9f2c1a4e-0b7d-4c8e-9a1b-2f3c4d5e6f70").decode()
UNKNOWN = "extra"


def _account(index: int = 0, stable_id: str | None = None) -> GigaChatAccount:
    return GigaChatAccount(index=index, client=GigaChatClient(key=KEY), stable_id=stable_id)


def test_account_starts_healthy_with_a_free_semaphore():
    client = GigaChatClient(key=KEY)
    account = GigaChatAccount(index=2, client=client, stable_id="acct-2")

    assert account.index == 2
    assert account.client is client
    assert account.stable_id == "acct-2"
    assert account.sem.locked() is False
    assert account.broken is False
    assert account.broken_at is None
    assert account.label == "gigachat-acct#2"


def test_account_without_a_stable_id_keeps_none():
    assert _account().stable_id is None


def test_account_label_follows_the_index():
    assert [_account(index=index).label for index in (0, 1, 13)] == ["gigachat-acct#0", "gigachat-acct#1", "gigachat-acct#13"]


def test_mark_broken_records_the_flag_the_timestamp_and_the_label(caplog):
    account = _account(index=4)
    before = time.monotonic()

    with caplog.at_level(logging.WARNING, logger="danyapi.gigachat"):
        account.mark_broken()

    assert account.broken is True
    assert account.broken_at is not None
    assert before <= account.broken_at <= time.monotonic()
    assert [record.getMessage() for record in caplog.records] == ["gigachat account #4 marked broken (invalid/expired authorization key)"]
    assert account.label == "gigachat-acct#4"


def test_mark_broken_is_idempotent(caplog):
    account = _account(index=7)

    with caplog.at_level(logging.WARNING, logger="danyapi.gigachat"):
        account.mark_broken()
        first = account.broken_at
        account.mark_broken()

    assert account.broken is True
    assert account.broken_at == first
    assert len(caplog.records) == 1


def test_account_rejects_attributes_outside_its_slots():
    account = _account()

    with pytest.raises(AttributeError, match="'GigaChatAccount' object has no attribute 'extra'"):
        setattr(account, UNKNOWN, 1)

    assert GigaChatAccount.__slots__ == ("broken", "broken_at", "client", "index", "sem", "stable_id")
    assert not hasattr(account, UNKNOWN)
    assert not hasattr(account, "__dict__")


async def test_semaphore_serialises_concurrent_account_use():
    account = _account()
    order: list[str] = []

    async def worker(name: str) -> None:
        async with account.sem:
            order.append(f"enter-{name}")
            await asyncio.sleep(0)
            order.append(f"exit-{name}")

    await asyncio.gather(worker("a"), worker("b"))

    assert order == ["enter-a", "exit-a", "enter-b", "exit-b"]
    assert account.sem.locked() is False


async def test_semaphore_starts_held_after_acquiring():
    account = _account()

    async with account.sem:
        assert account.sem.locked() is True

    assert account.sem.locked() is False
