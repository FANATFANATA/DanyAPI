from __future__ import annotations

import asyncio
import logging
import time

from .client import AliceClient

log = logging.getLogger("danyapi.alice")


class AliceAccount:
    __slots__ = ("broken", "broken_at", "client", "index", "sem", "stable_id")

    def __init__(
        self,
        index: int,
        client: AliceClient,
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
            log.warning("alice account #%d marked broken", self.index)

    @property
    def label(self) -> str:
        return f"alice-acct#{self.index}"
