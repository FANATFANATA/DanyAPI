from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import HTTPException

from ..alice.client import AliceClient
from ..gigachat.client import GigaChatClient
from ..qwen.client import QwenClient
from .state import _byok_mode, app

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

PROVIDER_NAMES = ("deepseek", "qwen", "gigachat", "alice")

CHAT_ONLY_GIGACHAT_TYPES = frozenset({"chat", "aicheck", "embedder", None})

GIGACHAT_EMBEDDING_PREFIXES = ("embed",)

ALICE_MODEL_IDS = ("alice", "alice-ai", "yagpt")

_MODEL_CACHE: dict[str, Any] = {
    "key": None,
    "models": None,
    "index": None,
    "qwen_ids": None,
    "gigachat_ids": None,
    "alice_ids": None,
}


MODEL_TYPE_BY_NAME = {
    "deepseek-v4.1-flash": "default",
}

REASONING_SUFFIXES = ("-thinking",)


QWEN_DEFAULT_MODELS = [
    {
        "id": "qwen3.8-max",
        "name": "Qwen3.8-Max",
        "owned_by": "qwen",
        "model_type": "chat",
    },
    {
        "id": "qwen3.7-plus",
        "name": "Qwen3.7-Plus",
        "owned_by": "qwen",
        "model_type": "chat",
    },
    {
        "id": "qwen3.7-max",
        "name": "Qwen3.7-Max",
        "owned_by": "qwen",
        "model_type": "chat",
    },
]


async def _fetch_qwen_models(client: QwenClient) -> list[dict]:
    try:
        raw = await client.fetch_models()
    except Exception as exc:
        log.warning("qwen models fetch failed, using defaults: %s", exc)
        return QWEN_DEFAULT_MODELS
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
    if not models:
        log.warning("qwen models fetch returned nothing, using defaults")
        return QWEN_DEFAULT_MODELS
    return models


ALICE_DEFAULT_MODELS = [
    {
        "id": "alice",
        "name": "Alice AI (Yandex)",
        "owned_by": "alice",
        "model_type": "chat",
    },
    {
        "id": "alice-ai",
        "name": "Alice AI (Yandex)",
        "owned_by": "alice",
        "model_type": "chat",
    },
    {
        "id": "yagpt",
        "name": "YaGPT (Yandex)",
        "owned_by": "alice",
        "model_type": "chat",
    },
]

GIGACHAT_DEFAULT_MODELS = [
    {
        "id": "GigaChat",
        "name": "GigaChat 2 Lite",
        "owned_by": "gigachat",
        "model_type": "chat",
    },
    {
        "id": "GigaChat-2-Pro",
        "name": "GigaChat 2 Pro",
        "owned_by": "gigachat",
        "model_type": "chat",
    },
    {
        "id": "GigaChat-2-Max",
        "name": "GigaChat 2 Max",
        "owned_by": "gigachat",
        "model_type": "chat",
    },
]


async def _fetch_gigachat_models(client: GigaChatClient) -> list[dict]:
    try:
        raw = await client.fetch_models()
    except Exception as exc:
        log.warning("gigachat models fetch failed, using defaults: %s", exc)
        return GIGACHAT_DEFAULT_MODELS
    models: list[dict] = []
    for model in raw:
        if not isinstance(model, dict) or not model.get("id"):
            continue
        model_id = str(model["id"])
        if model_id.lower().startswith(GIGACHAT_EMBEDDING_PREFIXES):
            continue
        model_type = model.get("type")
        if model_type not in CHAT_ONLY_GIGACHAT_TYPES:
            continue
        models.append(
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "owned_by": "gigachat",
                "model_type": "chat",
            }
        )
    if not models:
        log.warning("gigachat models fetch returned no chat models, using defaults")
        return GIGACHAT_DEFAULT_MODELS
    return models


async def _fetch_alice_models(client: AliceClient) -> list[dict]:
    return list(ALICE_DEFAULT_MODELS)


def _resolve_model(model: str) -> str:
    model_type = MODEL_TYPE_BY_NAME.get(model)
    if model_type is not None:
        return model_type
    for suffix in REASONING_SUFFIXES:
        if not model.endswith(suffix):
            continue
        base_type = MODEL_TYPE_BY_NAME.get(model[: -len(suffix)])
        if base_type is not None:
            return base_type
        break
    raise HTTPException(404, f"Unknown model: {model}")


def _is_reasoning_model(model: str) -> bool:
    return any(model.endswith(suffix) for suffix in REASONING_SUFFIXES)


def _finish_reason(status: Any) -> str:
    if isinstance(status, str):
        return STATUS_TO_FINISH_REASON.get(status, "stop")
    return "stop"


def _output_truncated(status: Any) -> bool:
    return _finish_reason(status) == "length"


def _model_source() -> list[dict]:
    sources: list[dict] = []
    sources.extend(getattr(app.state, "qwen_models", None) or [])
    sources.extend(getattr(app.state, "gigachat_models", None) or [])
    sources.extend(getattr(app.state, "alice_models", None) or [])
    if not sources and _byok_mode():
        return [*QWEN_DEFAULT_MODELS, *GIGACHAT_DEFAULT_MODELS, *ALICE_DEFAULT_MODELS]
    return sources


def _model_cache_key() -> tuple[tuple[Any, ...], ...]:
    return tuple((m.get("id"), m.get("name"), m.get("owned_by"), m.get("model_type")) for m in _model_source())


def _models_state() -> list[dict]:
    key = _model_cache_key()
    cached = _MODEL_CACHE
    if cached["key"] == key and cached["models"] is not None:
        return cached["models"]
    models: list[dict] = []
    for name, model_type in MODEL_TYPE_BY_NAME.items():
        models.append(
            {
                "id": name,
                "object": "model",
                "created": MODEL_CREATED_AT,
                "owned_by": "deepseek",
                "model_type": model_type,
            }
        )
        for suffix in REASONING_SUFFIXES:
            models.append(
                {
                    "id": f"{name}{suffix}",
                    "object": "model",
                    "created": MODEL_CREATED_AT,
                    "owned_by": "deepseek",
                    "model_type": model_type,
                }
            )
    for model in _model_source():
        models.append(
            {
                "id": model["id"],
                "object": "model",
                "created": MODEL_CREATED_AT,
                "owned_by": model.get("owned_by", "qwen"),
                "name": model.get("name"),
                "model_type": model.get("model_type", "chat"),
            }
        )
    cached["key"] = key
    cached["models"] = models
    cached["index"] = {m["id"]: m for m in models}
    cached["qwen_ids"] = {m.get("id") for m in _model_source() if m.get("owned_by") == "qwen"}
    cached["gigachat_ids"] = {m.get("id") for m in _model_source() if m.get("owned_by") == "gigachat"}
    cached["alice_ids"] = {m.get("id") for m in _model_source() if m.get("owned_by") == "alice"}
    return models


def _all_models() -> list[dict]:
    return _models_state()


@app.get("/v1/models")
async def list_models() -> dict:
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


def _resolve_provider(model: str) -> str:
    lowered = model.lower()
    if lowered.startswith("qwen"):
        return "qwen"
    if lowered.startswith("gigachat"):
        return "gigachat"
    if lowered in ALICE_MODEL_IDS:
        return "alice"
    if model in MODEL_TYPE_BY_NAME or lowered.startswith("deepseek"):
        return "deepseek"
    if model in _cached_ids("qwen_ids"):
        return "qwen"
    if model in _cached_ids("gigachat_ids"):
        return "gigachat"
    if model in _cached_ids("alice_ids"):
        return "alice"
    raise HTTPException(404, f"Unknown model: {model}")
