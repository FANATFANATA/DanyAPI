from __future__ import annotations

import asyncio
import logging
import time

from .client import GigaChatClient

log = logging.getLogger("danyapi.gigachat")


class GigaChatAccount:
    __slots__ = ("broken", "broken_at", "client", "index", "sem", "stable_id")

    def __init__(
        self,
        index: int,
        client: GigaChatClient,
        stable_id: str | None = None,
    ) -> None:
        self.index = index
        self.client = client
        self.sem = asyncio.Semaphore(1)
        self.stable_id = stable_id
        self.broken = False
        self.broken_at: float | None = None

    def mark_broken(self) -> None:
        if not self.broken:
            self.broken = True
            self.broken_at = time.monotonic()
            log.warning("gigachat account #%d marked broken (invalid/expired authorization key)", self.index)

    @property
    def label(self) -> str:
        return f"gigachat-acct#{self.index}"
