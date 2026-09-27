from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")
_IMAGE_TOKEN_COST = 85


def _cjk_count(text: str) -> int:
    return len(_CJK_RE.findall(text))


def _units(cjk: int, other: int) -> int:
    if other == 0:
        return cjk
    return cjk + max(1, other // 4)


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    cjk = _cjk_count(text)
    return _units(cjk, len(text) - cjk)


def _head_length(word: str, budget: int) -> int:
    cjk = 0
    length = 0
    for char in word:
        candidate_cjk = cjk + _cjk_count(char)
        if _units(candidate_cjk, length + 1 - candidate_cjk) > budget:
            break
        cjk = candidate_cjk
        length += 1
    return length


def trim_to_tokens(text: str, budget: int | None) -> str:
    if budget is None or not text or estimate_tokens(text) <= budget:
        return text
    words = text.split(" ")
    total_len = 0
    total_cjk = 0
    for index, word in enumerate(words):
        candidate_len = total_len + len(word) + (1 if index else 0)
        candidate_cjk = total_cjk + _cjk_count(word)
        if _units(candidate_cjk, candidate_len - candidate_cjk) > budget:
            break
        total_len = candidate_len
        total_cjk = candidate_cjk
    if total_len:
        return text[:total_len].rstrip() or text[:total_len]
    head = words[0]
    if not head:
        return text
    return text[: _head_length(head, budget)]


class StreamBudget:
    __slots__ = ("_budget", "_chars", "_chunks", "_cjk", "_trim", "done")

    def __init__(self, budget: int | None, trim: Callable[[str, int | None], str]) -> None:
        self._budget = budget
        self._trim = trim
        self._chunks: list[str] = []
        self.done = False
        self._cjk = 0
        self._chars = 0

    @property
    def text(self) -> str:
        chunks = self._chunks
        if len(chunks) == 1:
            return chunks[0]
        return "".join(chunks)

    def _count(self) -> int:
        if self._chars == 0:
            return self._cjk
        return self._cjk + max(1, self._chars // 4)

    def feed(self, piece: str | None) -> str:
        if not piece or self.done:
            return ""
        self._chunks.append(piece)
        if self._budget is None:
            return piece
        cjk = _cjk_count(piece)
        self._cjk += cjk
        self._chars += len(piece) - cjk
        if self._count() <= self._budget:
            return piece
        text = "".join(self._chunks)
        trimmed = self._trim(text, self._budget)
        if trimmed == text:
            return piece
        self.done = True
        self._chunks = [trimmed]
        previous = text[: len(text) - len(piece)]
        if not trimmed.startswith(previous):
            return ""
        return trimmed[len(previous) :]


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
