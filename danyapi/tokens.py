from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache
from typing import Any

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")
_IMAGE_TOKEN_COST = 85


@lru_cache(maxsize=4096)
def _cjk_count(text: str) -> int:
    return len(_CJK_RE.findall(text))


@lru_cache(maxsize=4096)
def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    cjk = _cjk_count(text)
    other = len(text) - cjk
    if other == 0:
        return cjk
    return cjk + max(1, other // 4)


class StreamBudget:
    __slots__ = ("_budget", "_trim", "done", "text")

    def __init__(self, budget: int | None, trim: Callable[[str, int | None], str]) -> None:
        self._budget = budget
        self._trim = trim
        self.text = ""
        self.done = False

    def feed(self, piece: str | None) -> str:
        if not piece or self.done:
            return ""
        if self._budget is None:
            self.text += piece
            return piece
        candidate = self.text + piece
        trimmed = self._trim(candidate, self._budget)
        if trimmed == candidate:
            self.text = candidate
            return piece
        self.done = True
        if not trimmed.startswith(self.text):
            return ""
        send = trimmed[len(self.text) :]
        self.text = trimmed
        return send


def count_message_tokens(message: Any) -> int:
    if isinstance(message, dict):
        content = message.get("content")
        tool_calls = message.get("tool_calls")
    elif hasattr(message, "content"):
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None)
    else:
        return 0
    tokens = 3
    if isinstance(content, str):
        tokens += estimate_tokens(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    tokens += estimate_tokens(text)
                elif item.get("type") == "image_url":
                    tokens += _IMAGE_TOKEN_COST
            elif isinstance(item, str):
                tokens += estimate_tokens(item)
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str):
                    tokens += estimate_tokens(name)
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    tokens += estimate_tokens(arguments)
    return tokens


def count_messages_tokens(messages: list[Any]) -> int:
    return sum(count_message_tokens(message) for message in messages)


def count_prompt_tokens(prompt: str | None) -> int:
    return estimate_tokens(prompt)
