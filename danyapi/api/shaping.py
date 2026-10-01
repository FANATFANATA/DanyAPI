from __future__ import annotations

from typing import Any

from ..config import MAX_CHOICES
from ..sseutil import split_stop
from ..tokens import estimate_tokens, trim_to_tokens
from .schemas import ChatCompletionRequest

MAX_STREAM_CHOICES = MAX_CHOICES


def _bounded_choices(n: int | None) -> int:
    if not isinstance(n, int) or n <= 1:
        return 1
    return min(n, MAX_STREAM_CHOICES)


def _max_calls(parallel_tool_calls: bool | None) -> int | None:
    return 1 if parallel_tool_calls is False else None


def _apply_stop(text: str, stop: Any) -> str:
    cut = -1
    for marker in split_stop(stop):
        position = text.find(marker)
        if position != -1 and (cut == -1 or position < cut):
            cut = position
    return text[:cut] if cut != -1 else text


def _apply_limits(content: str, max_tokens: int | None, stop: Any) -> tuple[str, str]:
    text = _apply_stop(content or "", stop)
    if text != (content or ""):
        return text, "stop"
    trimmed = trim_to_tokens(text, max_tokens)
    if trimmed != text:
        return trimmed, "length"
    return trimmed, "stop"


def _include_usage(req: ChatCompletionRequest) -> bool:
    opts = getattr(req, "stream_options", None)
    if not isinstance(opts, dict):
        return False
    return bool(opts.get("include_usage"))


def _safe_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    try:
        return int(value)
    except (OverflowError, ValueError):
        return 0


def _deepseek_usage(total: int, prompt: str = "", provider_usage: dict | None = None, completion_text: str | None = None) -> dict:
    prompt_tokens = 0
    if isinstance(provider_usage, dict):
        p_tokens = provider_usage.get("prompt_tokens")
        if isinstance(p_tokens, int) and p_tokens > 0:
            prompt_tokens = p_tokens
    if not prompt_tokens:
        prompt_tokens = estimate_tokens(prompt)
    total_tokens = max(0, _safe_int(total))
    if total_tokens < prompt_tokens:
        total_tokens = prompt_tokens
    completion_tokens = max(0, total_tokens - prompt_tokens)
    if not completion_tokens and completion_text:
        completion_tokens = estimate_tokens(completion_text)
        total_tokens = prompt_tokens + completion_tokens
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}


def _usage_with_details(usage: dict, reasoning_text: str | None = None, reasoning_tokens: int | None = None) -> dict:
    result = dict(usage)
    if not isinstance(result.get("prompt_tokens_details"), dict):
        result["prompt_tokens_details"] = {"cached_tokens": 0}
    if not isinstance(result.get("completion_tokens_details"), dict):
        if reasoning_tokens is None:
            reasoning_tokens = estimate_tokens(reasoning_text or "")
        result["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return result


USAGE_TOTAL_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _usage_number(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _merge_usage(previous: dict | None, current: dict) -> dict:
    if previous is None:
        return dict(current)
    merged = dict(previous)
    for field in USAGE_TOTAL_FIELDS:
        left = _usage_number(merged.get(field))
        right = _usage_number(current.get(field))
        if left is not None and right is not None:
            merged[field] = left + right
        elif right is not None:
            merged[field] = right
        elif left is not None:
            merged[field] = left
    return merged


def _advance_session_usage(session, accumulated_total: int) -> int:
    prev = max(0, int(getattr(session, "accumulated_tokens", 0) or 0))
    current = max(0, int(accumulated_total or 0))
    session.accumulated_tokens = max(prev, current)
    return max(0, current - prev)
