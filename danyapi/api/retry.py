from __future__ import annotations

import logging
import random

from fastapi import HTTPException

log = logging.getLogger("danyapi.api")

MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 1.0
RETRY_BACKOFF_MAX_SEC = 8.0
RETRY_BACKOFF_JITTER = 0.25

RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
STALE_SESSION_STATUSES = {400, 404}


def _retry_delay(attempt: int, retry_after: object = None) -> float:
    hinted = _retry_after_hint(retry_after)
    if hinted is not None:
        return min(max(hinted, 0.0), RETRY_BACKOFF_MAX_SEC)
    base = min(RETRY_BACKOFF_SEC * (2 ** (max(1, attempt) - 1)), RETRY_BACKOFF_MAX_SEC)
    spread = base * RETRY_BACKOFF_JITTER
    return min(RETRY_BACKOFF_MAX_SEC, base + random.uniform(-spread, spread))


def _retry_after_hint(retry_after: object) -> float | None:
    if isinstance(retry_after, bool) or not isinstance(retry_after, (int, float, str)):
        return None
    try:
        seconds = float(retry_after)
    except (TypeError, ValueError):
        return None
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return None
    return seconds


def _retry_after_header(response: object) -> object:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    return getter("retry-after")


def _is_retryable_http(exc: HTTPException) -> bool:
    return exc.status_code in RETRYABLE_HTTP_STATUSES


async def _try_stop_stream(client, session_id: str, message_id: str | None) -> None:
    if not session_id or not message_id:
        return
    try:
        await client.stop_stream(session_id, message_id)
    except Exception as exc:
        log.debug("stop_stream failed for %s: %s", session_id, exc)
