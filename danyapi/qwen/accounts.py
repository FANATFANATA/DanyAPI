from __future__ import annotations

import asyncio
import logging

from ..sessions import SessionRegistry
from .client import QwenClient, QwenSession

log = logging.getLogger("danyapi.qwen")


class QwenSessionRegistry(SessionRegistry):
    def __init__(self, client: QwenClient, maxsize: int = 128, ttl: float = 0.0) -> None:
        super().__init__(client, maxsize, ttl)

    async def _create(self, **kwargs) -> QwenSession:
        model = kwargs.get("model") or ""
        chat_id = await self._client.create_chat(model=model, chat_mode="normal")
        return QwenSession(id=chat_id, model=model)

    def _reuse(self, session: QwenSession, session_id: str, **kwargs) -> bool:
        return session.model == (kwargs.get("model") or "")

    def _update_last(self, session: QwenSession, message_id: str) -> None:
        session.last_response_id = message_id

    async def obtain(self, session_id: str | None = None, model: str | None = None, **kwargs) -> tuple[QwenSession, str]:
        return await super().obtain(session_id, model=model or "")


class QwenAccount:
    __slots__ = ("broken", "client", "index", "sem", "sessions")

    def __init__(self, index: int, client: QwenClient, session_cache_size: int = 128, ttl: float = 0.0) -> None:
        self.index = index
        self.client = client
        self.sem = asyncio.Semaphore(1)
        self.sessions = QwenSessionRegistry(client, session_cache_size, ttl)
        self.broken = False

    def mark_broken(self) -> None:
        if not self.broken:
            self.broken = True
            log.warning("qwen account #%d marked broken (invalid/expired token)", self.index)

    @property
    def label(self) -> str:
        return f"qwen-acct#{self.index}"
