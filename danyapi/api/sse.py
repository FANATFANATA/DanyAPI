from __future__ import annotations

import json
import logging
import time
import uuid

from ..accounts import AccountPoolBusy
from .core import INTERNAL_ERROR_MESSAGE

log = logging.getLogger("danyapi.api")


_JSON_ENCODE = json.JSONEncoder(ensure_ascii=False).encode


def _sse(data: dict) -> str:
    return f"data: {_JSON_ENCODE(data)}\n\n"


def _delta_json(delta: dict, finish: str | None) -> str:
    if finish is None:
        return f'{{"index":0,"delta":{_JSON_ENCODE(delta)}}}'
    return f'{{"index":0,"delta":{_JSON_ENCODE(delta)},"finish_reason":{_JSON_ENCODE(finish)}}}'


def _stream_error_sse(
    chunk_id: str,
    created: int,
    model: str,
    message: str,
    session_key: str | None = None,
    error_finish: str | None = None,
    choice_finish: str | None = None,
) -> tuple[str, str]:
    error: dict = {"message": message}
    if error_finish is not None:
        error["finish_reason"] = error_finish
    error_chunk = _sse(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "session_id": session_key,
            "error": error,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice_finish or "stop"}],
        }
    )
    return error_chunk, "data: [DONE]\n\n"


def _chunk_id_from_line(line: str) -> str | None:
    if not isinstance(line, str) or not line.startswith("data: "):
        return None
    payload = line[len("data: ") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        parsed = json.loads(payload)
    except ValueError:
        return None
    chunk_id = parsed.get("id") if isinstance(parsed, dict) else None
    return chunk_id if isinstance(chunk_id, str) and chunk_id else None


async def _close_generator(gen) -> None:
    aclose = getattr(gen, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception as exc:
        log.debug("stream generator close failed: %s", exc)


async def _stream_guard(gen, model: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    seen_id: str | None = None
    try:
        async for item in gen:
            if seen_id is None:
                seen_id = _chunk_id_from_line(item)
            yield item
    except AccountPoolBusy:
        for line in _stream_error_sse(seen_id or chunk_id, created, model, "all accounts are busy, try again later"):
            yield line
    except Exception as exc:
        log.exception("stream generator failed: %s", exc)
        for line in _stream_error_sse(seen_id or chunk_id, created, model, INTERNAL_ERROR_MESSAGE):
            yield line
    finally:
        await _close_generator(gen)
