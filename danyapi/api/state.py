from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI

from ..store import JsonStore

app = FastAPI(title="DanyAPI")


def _byok_mode() -> bool:
    return bool(getattr(app.state, "byok", False))


async def _byok_pools_state() -> dict[str, dict[str, Any]]:
    pools = getattr(app.state, "byok_pools", None)
    if pools is None:
        pools = {"deepseek": {}, "qwen": {}}
        app.state.byok_pools = pools
    return pools


async def _byok_locks_state() -> dict[str, asyncio.Lock]:
    locks = getattr(app.state, "byok_locks", None)
    if locks is None:
        locks = {"deepseek": asyncio.Lock(), "qwen": asyncio.Lock()}
        app.state.byok_locks = locks
    return locks


async def _byok_auth_state() -> dict[str, dict[str, Any]]:
    auth = getattr(app.state, "byok_auth", None)
    if auth is None:
        auth = {"deepseek": {}, "qwen": {}}
        app.state.byok_auth = auth
    return auth


async def _byok_stores_state() -> dict[str, dict[str, list[JsonStore]]]:
    stores = getattr(app.state, "byok_stores", None)
    if stores is None:
        stores = {"deepseek": {}, "qwen": {}}
        app.state.byok_stores = stores
    return stores
