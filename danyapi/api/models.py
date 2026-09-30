from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

from fastapi import HTTPException, Request

from ..alice.client import MODEL_ALIASES as ALICE_MODEL_IDS
from ..alice.client import MODEL_NAMES as ALICE_MODEL_NAMES
from ..config import settings
from ..deepseek.client import DEFAULT_MODEL_TYPE as DEEPSEEK_DEFAULT_MODEL_TYPE
from ..deepseek.client import DeepSeekClient
from ..duckai.client import DuckAIClient
from ..duckai.client import catalog_models as duckai_catalog_models
from ..gigachat.client import GigaChatClient
from ..opencode.client import MODEL_PREFIX as OPENCODE_MODEL_PREFIX
from ..opencode.client import OpenCodeClient
from ..qwen.client import QwenClient
from .state import BYOK_PROVIDERS, MODEL_ATTRS, _byok_mode, app, provider_models, provider_pool

log = logging.getLogger("danyapi.api")

STATUS_TO_FINISH_REASON = {
    "FINISHED": "stop",
    "CONTEXT_LENGTH_EXCEEDED": "length",
    "CONTENT_FILTER": "content_filter",
    "INCOMPLETE": "length",
    "WIP": "length",
    "TIMEOUT": "length",
}

MODEL_CREATED_AT = int(time.time())

PROVIDER_NAMES = BYOK_PROVIDERS

GIGACHAT_EMBEDDING_MARKER = "embed"
GIGACHAT_CHAT_TYPES = frozenset({"chat"})

REASONING_SUFFIXES = ("-thinking",)

DEEPSEEK_LEGACY_ALIASES = frozenset({"deepseek-v4.1-flash"})

_MODEL_CACHE: dict[str, Any] = {
    "key": None,
    "models": None,
    "index": None,
    "deepseek_ids": None,
    "qwen_ids": None,
    "gigachat_ids": None,
    "opencode_ids": None,
    "alice_ids": None,
    "duckai_ids": None,
}

_REFRESH_LOCKS: dict[str, asyncio.Lock] = {provider: asyncio.Lock() for provider in BYOK_PROVIDERS}


def _header_api_key(request: Request) -> str:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        key = auth[7:].strip()
        if key:
            return key
    return (request.headers.get("x-api-key") or "").strip()


def _pool_client(provider: str) -> Any:
    pool = provider_pool(provider)
    accounts = getattr(pool, "accounts", None) or []
    return accounts[0].client if accounts else None


def _probe_client(provider: str, api_key: str | None) -> Any:
    if provider == "deepseek":
        return DeepSeekClient(token=api_key or None, timeout=settings.timeout)
    if provider == "qwen":
        return QwenClient(token=api_key or None, timeout=settings.timeout)
    if provider == "duckai":
        return DuckAIClient(timeout=settings.timeout)
    if provider == "gigachat" and api_key:
        return GigaChatClient(key=api_key, scope=settings.gigachat_scope, timeout=settings.timeout)
    if provider == "opencode":
        return OpenCodeClient(key=api_key or "", timeout=settings.timeout)
    return None


async def _close_probe_client(client: Any) -> None:
    try:
        await client.aclose()
    except Exception as exc:
        log.info("model probe client close failed: %s", exc)


async def _fetch_deepseek_models(client: DeepSeekClient) -> list[dict]:
    raw = await client.fetch_models()
    models: list[dict] = []
    for entry in raw:
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        models.append(
            {
                "id": model_id,
                "name": entry.get("name") or model_id,
                "owned_by": "deepseek",
                "model_type": "chat",
                "upstream_type": entry.get("model_type") or model_id,
                "is_default": bool(entry.get("is_default")),
                "supports_thinking": bool(entry.get("supports_thinking")),
                "supports_search": bool(entry.get("supports_search")),
            }
        )
    return models


async def _fetch_qwen_models(client: QwenClient) -> list[dict]:
    raw = await client.fetch_models()
    models = []
    for model in raw:
        if not isinstance(model, dict) or not model.get("id"):
            continue
        info = model.get("info")
        meta = (info or {}).get("meta") if isinstance(info, dict) else None
        chat_types = (meta or {}).get("chat_type") if isinstance(meta, dict) else None
        chat_types = chat_types or []
        if "t2t" in chat_types:
            model_type = "chat"
        elif "t2i" in chat_types:
            model_type = "image"
        elif "t2v" in chat_types:
            model_type = "video"
        else:
            model_type = "chat"
        models.append(
            {
                "id": model["id"],
                "name": model.get("name") or model["id"],
                "owned_by": "qwen",
                "model_type": model_type,
                "chat_types": chat_types,
            }
        )
    return models


async def _fetch_gigachat_models(client: GigaChatClient) -> list[dict]:
    raw = await client.fetch_models()
    models: list[dict] = []
    for model in raw:
        if not isinstance(model, dict) or not model.get("id"):
            continue
        model_id = str(model["id"])
        if GIGACHAT_EMBEDDING_MARKER in model_id.lower():
            continue
        model_type = model.get("type")
        if isinstance(model_type, str) and model_type not in GIGACHAT_CHAT_TYPES:
            continue
        models.append(
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "owned_by": "gigachat",
                "model_type": "chat",
            }
        )
    return models


async def _fetch_opencode_models(client: OpenCodeClient) -> list[dict]:
    raw = await client.fetch_models()
    models: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        model_id = str(entry["id"])
        models.append(
            {
                "id": model_id,
                "name": entry.get("name") or model_id,
                "owned_by": "opencode",
                "model_type": "chat",
            }
        )
    return models


async def _fetch_alice_models(client: Any = None) -> list[dict]:
    return [
        {
            "id": alias,
            "name": name,
            "owned_by": "alice",
            "model_type": "chat",
        }
        for alias, name in ALICE_MODEL_NAMES.items()
    ]


async def _fetch_duckai_models(client: DuckAIClient) -> list[dict]:
    raw = await client.fetch_models()
    models: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        models.append(
            {
                "id": entry["id"],
                "name": entry.get("name") or entry["id"],
                "owned_by": "duckai",
                "model_type": entry.get("model_type", "chat"),
                "provider": entry.get("provider", ""),
                "efforts": list(entry.get("efforts") or ()),
            }
        )
    return models


MODEL_FETCHERS: dict[str, Any] = {
    "deepseek": _fetch_deepseek_models,
    "qwen": _fetch_qwen_models,
    "gigachat": _fetch_gigachat_models,
    "opencode": _fetch_opencode_models,
    "alice": _fetch_alice_models,
    "duckai": _fetch_duckai_models,
}


async def _store_models(provider: str, client: Any) -> list[dict]:
    fetcher = MODEL_FETCHERS.get(provider)
    if fetcher is None:
        return provider_models(provider)
    lock = _REFRESH_LOCKS.setdefault(provider, asyncio.Lock())
    async with lock:
        fetched: list[dict] = []
        try:
            fetched = await fetcher(client)
        except Exception as exc:
            log.warning("%s models fetch failed: %s", provider, exc)
        if not fetched:
            kept = provider_models(provider)
            log.warning("%s models fetch returned nothing, keeping %d known models", provider, len(kept))
            return kept
        setattr(app.state, MODEL_ATTRS[provider], fetched)
        log.info("%s models refreshed: %d", provider, len(fetched))
        return fetched


async def refresh_provider_models(provider: str, client: Any) -> list[dict]:
    if provider not in MODEL_FETCHERS:
        return []
    return await _store_models(provider, client)


async def refresh_models(api_key: str | None = None, providers: Sequence[str] | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for provider in providers or BYOK_PROVIDERS:
        if provider == "alice":
            counts[provider] = len(await _store_models(provider, None))
            continue
        client = _pool_client(provider)
        owned = client is not None
        if client is None:
            try:
                client = _probe_client(provider, api_key)
            except (RuntimeError, OSError) as exc:
                log.warning("%s model refresh skipped, client unusable: %s", provider, exc)
                continue
        if client is None:
            log.info("%s model refresh skipped, no credentials available", provider)
            continue
        try:
            counts[provider] = len(await _store_models(provider, client))
        finally:
            if not owned:
                await _close_probe_client(client)
    return counts


async def model_refresh_loop() -> None:
    interval = settings.models_refresh_seconds
    if interval <= 0:
        return
    while True:
        await asyncio.sleep(interval)
        try:
            await refresh_models()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("model refresh cycle failed: %s", exc)


def _resolve_model(model: str) -> str:
    base = _strip_reasoning_suffix(model)
    for entry in provider_models("deepseek"):
        if str(entry.get("id", "")).lower() != base:
            continue
        return str(entry.get("upstream_type") or entry["id"])
    if base in DEEPSEEK_LEGACY_ALIASES:
        return _default_deepseek_model_type()
    raise HTTPException(404, f"Unknown model: {model}")


def _default_deepseek_model_type() -> str:
    entries = provider_models("deepseek")
    for entry in entries:
        if entry.get("is_default"):
            return str(entry.get("upstream_type") or entry["id"])
    if entries:
        return str(entries[0].get("upstream_type") or entries[0]["id"])
    return DEEPSEEK_DEFAULT_MODEL_TYPE


def _strip_reasoning_suffix(model: str) -> str:
    for suffix in REASONING_SUFFIXES:
        if model.endswith(suffix):
            return model[: -len(suffix)].lower()
    return model.lower()


def _is_reasoning_model(model: str) -> bool:
    return any(model.endswith(suffix) for suffix in REASONING_SUFFIXES)


def _finish_reason(status: Any) -> str:
    if isinstance(status, str):
        return STATUS_TO_FINISH_REASON.get(status, "stop")
    return "stop"


def _output_truncated(status: Any) -> bool:
    return _finish_reason(status) == "length"


def _model_source() -> list[dict]:
    models: list[dict] = []
    for provider in BYOK_PROVIDERS:
        models.extend(provider_models(provider))
    return models


def _model_cache_key() -> tuple[tuple[Any, ...], ...]:
    return tuple((m.get("id"), m.get("name"), m.get("owned_by"), m.get("model_type")) for m in _model_source())


def _models_state() -> list[dict]:
    key = _model_cache_key()
    cached = _MODEL_CACHE
    if cached["key"] == key and cached["models"] is not None:
        return cached["models"]
    source = _model_source()
    models: list[dict] = []
    for model in source:
        owner = model.get("owned_by") or "qwen"
        entry = {
            "id": model["id"],
            "object": "model",
            "created": MODEL_CREATED_AT,
            "owned_by": owner,
            "name": model.get("name"),
            "model_type": model.get("model_type", "chat"),
        }
        models.append(entry)
        if owner == "deepseek":
            for suffix in REASONING_SUFFIXES:
                models.append({**entry, "id": f"{model['id']}{suffix}"})
    cached["key"] = key
    cached["models"] = models
    cached["index"] = {m["id"]: m for m in models}
    for provider in BYOK_PROVIDERS:
        cached[f"{provider}_ids"] = {str(m.get("id", "")).lower() for m in provider_models(provider) if m.get("id")}
    return models


def _all_models() -> list[dict]:
    return _models_state()


@app.get("/v1/models")
async def list_models(request: Request, refresh: int = 0) -> dict:
    api_key = _header_api_key(request)
    if refresh or api_key:
        await refresh_models(api_key=api_key or None)
    return {"object": "list", "data": _all_models()}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> dict:
    _models_state()
    model = _MODEL_CACHE["index"].get(model_id)
    if model is None:
        raise HTTPException(404, f"The model '{model_id}' does not exist")
    return model


def _cached_ids(key: str) -> set[str]:
    _models_state()
    value = _MODEL_CACHE.get(key)
    return value if isinstance(value, set) else set()


def _known_ids(provider: str) -> set[str]:
    ids = _cached_ids(f"{provider}_ids")
    if provider == "duckai":
        ids = ids | {str(entry.get("id", "")).lower() for entry in duckai_catalog_models() if entry.get("id")}
    elif provider == "alice":
        ids = ids | {alias.lower() for alias in ALICE_MODEL_IDS}
    return ids


def _is_deepseek_model(lowered: str) -> bool:
    if _strip_reasoning_suffix(lowered) in _known_ids("deepseek"):
        return True
    if _strip_reasoning_suffix(lowered) in DEEPSEEK_LEGACY_ALIASES:
        return True
    return lowered.startswith("deepseek")


def _resolve_provider(model: str) -> str:
    lowered = model.lower()
    if lowered.startswith(OPENCODE_MODEL_PREFIX):
        return "opencode"
    if lowered.startswith("qwen"):
        return "qwen"
    if lowered.startswith("gigachat"):
        return "gigachat"
    if lowered in ALICE_MODEL_IDS:
        return "alice"
    if lowered.startswith("duckai") or lowered in _known_ids("duckai"):
        return "duckai"
    if _is_deepseek_model(lowered):
        return "deepseek"
    for provider in BYOK_PROVIDERS:
        if lowered in _known_ids(provider):
            return provider
    raise HTTPException(404, f"Unknown model: {model}")


def provider_enabled(provider: str) -> bool:
    if _byok_mode():
        return True
    return provider_pool(provider) is not None
