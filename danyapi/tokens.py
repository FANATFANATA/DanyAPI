from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_CJK_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x3040, 0x30FF), (0xAC00, 0xD7AF))
_CJK_RE = re.compile("[" + "".join(f"{chr(low)}-{chr(high)}" for low, high in _CJK_RANGES) + "]")
_CJK_MATCH_RE = _CJK_RE.match
_IMAGE_TOKEN_COST = 85
MESSAGE_OVERHEAD_TOKENS = 3


def _is_cjk(char: str) -> bool:
    return _CJK_MATCH_RE(char) is not None


def _cjk_count(text: str) -> int:
    if text.isascii():
        return 0
    return sum(1 for _ in _CJK_RE.finditer(text))


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
    if budget <= 0:
        return 0
    size = len(word)
    if not _cjk_count(word):
        return min(size, budget * 4 + 3)
    cjk = 0
    length = 0
    for char in word:
        candidate_cjk = cjk + 1 if _is_cjk(char) else cjk
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
        stripped = text.lstrip()
        if not stripped:
            return ""
        return stripped[: _head_length(stripped, budget)]
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
            self.done = True
            self._chunks = [text]
            return piece
        self.done = True
        self._chunks = [trimmed]
        previous = text[: len(text) - len(piece)]
        if not trimmed.startswith(previous):
            return ""
        return trimmed[len(previous) :]


def _message_fields(message: Any) -> tuple[Any, Any] | None:
    if isinstance(message, dict):
        return message.get("content"), message.get("tool_calls")
    content = getattr(message, "content", None)
    if content is None and getattr(message, "tool_calls", None) is None:
        return None
    return content, getattr(message, "tool_calls", None)


def count_message_tokens(message: Any) -> int:
    fields = _message_fields(message)
    if fields is None:
        return 0
    content, tool_calls = fields
    tokens = MESSAGE_OVERHEAD_TOKENS
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
