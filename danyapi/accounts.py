"""Пул DeepSeek-аккаунтов.

Каждый аккаунт может генерировать одно сообщение одновременно
(ограничение chat.deepseek.com). Чтобы обслуживать конкурентные запросы,
DanyAPI держит пул аккаунтов и распределяет нагрузку между ними.

Сессии привязаны к аккаунту, на котором созданы (история диалога живёт
серверно) — повторные запросы с тем же session_id маршрутизируются туда же.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .deepseek.client import DeepSeekClient
from .pow import PowManager
from .sessions import SessionRegistry

log = logging.getLogger("danyapi.accounts")


class DeepSeekAccount:
    __slots__ = ("index", "client", "pow", "sem", "sessions")

    def __init__(self, index: int, client: DeepSeekClient) -> None:
        self.index = index
        self.client = client
        self.pow = PowManager()
        self.sem = asyncio.Semaphore(1)
        self.sessions = SessionRegistry(client)

    @property
    def label(self) -> str:
        return f"acct#{self.index}"


class AccountPool:
    def __init__(self, accounts: list[DeepSeekAccount]) -> None:
        self.accounts = accounts
        self._by_session: dict[str, int] = {}
        self._rr = 0

    def register(self, account_index: int, session_id: str) -> None:
        self._by_session[session_id] = account_index

    def account_for_session(self, session_id: str) -> Optional[DeepSeekAccount]:
        idx = self._by_session.get(session_id)
        if idx is None:
            return None
        acct = self.accounts[idx]
        return acct if acct is not None else None

    async def acquire(
        self, session_id: Optional[str]
    ) -> tuple[DeepSeekAccount, Optional[str]]:
        """Возвращает (аккаунт, существующий session_id или None).

        Сем семафора НЕ захватывается — вызывающий берёт `async with acct.sem`
        вокруг всей генерации.
        """
        if not self.accounts:
            raise RuntimeError("no deepseek accounts configured")
        if session_id:
            acct = self.account_for_session(session_id)
            if acct is not None:
                return acct, session_id
        n = len(self.accounts)
        start = self._rr % n
        for i in range(n):
            idx = (start + i) % n
            acct = self.accounts[idx]
            if not acct.sem.locked():
                self._rr = (idx + 1) % n
                return acct, None
        acct = self.accounts[start]
        return acct, None
