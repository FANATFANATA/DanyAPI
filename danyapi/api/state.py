from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI

from ..config import settings
from ..store import JsonStore

app = FastAPI(title="DanyAPI")

BYOK_PROVIDERS = ("deepseek", "qwen", "gigachat", "opencode", "alice", "duckai", "mistral")

KEYLESS_PROVIDERS = ("alice", "duckai")

KEY_OPTIONAL_PROVIDERS = ("opencode",)

MODEL_ATTRS = {
    "deepseek": "deepseek_models",
    "qwen": "qwen_models",
    "gigachat": "gigachat_models",
    "opencode": "opencode_models",
    "alice": "alice_models",
    "duckai": "duckai_models",
    "mistral": "mistral_models",
}

POOL_ATTRS_BY_PROVIDER = {
    "deepseek": "pool",
    "qwen": "qwen_pool",
    "gigachat": "gigachat_pool",
    "opencode": "opencode_pool",
    "alice": "alice_pool",
    "duckai": "duckai_pool",
    "mistral": "mistral_pool",
}


def _byok_mode() -> bool:
    return bool(getattr(app.state, "byok", False))


def provider_needs_api_key(provider: str) -> bool:
    return provider not in KEYLESS_PROVIDERS


def provider_models(provider: str) -> list[dict]:
    if not settings.provider_enabled(provider):
        return []
    attr = MODEL_ATTRS.get(provider)
    if attr is None:
        return []
    return list(getattr(app.state, attr, None) or [])


def provider_pool(provider: str) -> Any:
    if not settings.provider_enabled(provider):
        return None
    attr = POOL_ATTRS_BY_PROVIDER.get(provider)
    if attr is None:
        return None
    return getattr(app.state, attr, None)


def _blank_byok_state() -> dict[str, Any]:
    return {provider: {} for provider in BYOK_PROVIDERS}


def _blank_byok_locks() -> dict[str, asyncio.Lock]:
    return {provider: asyncio.Lock() for provider in BYOK_PROVIDERS}


def _blank_byok_stores() -> dict[str, dict[str, list[JsonStore]]]:
    return {provider: {} for provider in BYOK_PROVIDERS}


def _byok_pools_state() -> dict[str, dict[str, Any]]:
    pools = getattr(app.state, "byok_pools", None)
    if pools is None:
        pools = _blank_byok_state()
        app.state.byok_pools = pools
    return pools


def _byok_locks_state() -> dict[str, asyncio.Lock]:
    locks = getattr(app.state, "byok_locks", None)
    if locks is None:
        locks = _blank_byok_locks()
        app.state.byok_locks = locks
    return locks


def _byok_auth_state() -> dict[str, dict[str, Any]]:
    auth = getattr(app.state, "byok_auth", None)
    if auth is None:
        auth = _blank_byok_state()
        app.state.byok_auth = auth
    return auth


def _byok_stores_state() -> dict[str, dict[str, list[JsonStore]]]:
    stores = getattr(app.state, "byok_stores", None)
    if stores is None:
        stores = _blank_byok_stores()
        app.state.byok_stores = stores
    return stores
