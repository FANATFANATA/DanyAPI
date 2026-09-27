from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import HTTPException

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

_MODEL_CACHE: dict[str, Any] = {"key": None, "models": None, "index": None, "qwen_ids": None}


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
    qwen_models = getattr(app.state, "qwen_models", None) or []
    if not qwen_models and _byok_mode():
        return QWEN_DEFAULT_MODELS
    return qwen_models


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
    cached["qwen_ids"] = {m.get("id") for m in _model_source()}
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


def _resolve_provider(model: str) -> str:
    if model.startswith("qwen"):
        return "qwen"
    if model in MODEL_TYPE_BY_NAME or model.startswith("deepseek"):
        return "deepseek"
    _models_state()
    qwen_ids = _MODEL_CACHE.get("qwen_ids")
    if isinstance(qwen_ids, set) and model in qwen_ids:
        return "qwen"
    raise HTTPException(404, f"Unknown model: {model}")
