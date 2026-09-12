from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("danyapi.byok")


class ByokUser:
    __slots__ = ("id", "password_hash", "username")

    def __init__(self, id: str, username: str, password_hash: str) -> None:
        self.id = id
        self.username = username
        self.password_hash = password_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "password_hash": self.password_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ByokUser:
        return cls(
            id=data["id"],
            username=data["username"],
            password_hash=data["password_hash"],
        )


class ByokAccount:
    __slots__ = ("id", "provider", "token")

    def __init__(self, id: str, provider: str, token: str) -> None:
        self.id = id
        self.provider = provider
        self.token = token

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "provider": self.provider, "token": self.token}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ByokAccount:
        return cls(id=data["id"], provider=data["provider"], token=data["token"])


class UserTokens:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.accounts: list[ByokAccount] = []
        self.updated_at: float = 0.0

    def add_or_replace(self, account: ByokAccount) -> None:
        for i, acc in enumerate(self.accounts):
            if acc.provider == account.provider:
                self.accounts[i] = account
                return
        self.accounts.append(account)

    def remove_by_provider(self, provider: str) -> bool:
        before = len(self.accounts)
        self.accounts = [a for a in self.accounts if a.provider != provider]
        return len(self.accounts) < before

    def get_by_provider(self, provider: str) -> ByokAccount | None:
        for acc in self.accounts:
            if acc.provider == provider:
                return acc
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "accounts": [a.to_dict() for a in self.accounts],
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserTokens:
        ut = cls(user_id=data["user_id"])
        ut.accounts = [ByokAccount.from_dict(a) for a in data.get("accounts", [])]
        ut.updated_at = data.get("updated_at", 0.0)
        return ut
