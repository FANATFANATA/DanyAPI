from __future__ import annotations

import re
from typing import Any

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")
_IMAGE_TOKEN_COST = 85


def _is_cjk(ch: str) -> bool:
    return _CJK_RE.match(ch) is not None


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    if other == 0:
        return cjk
    return cjk + max(1, other // 4)


def count_message_tokens(message: Any) -> int:
    if not isinstance(message, dict):
        return 0
    tokens = 3
    content = message.get("content")
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
    tool_calls = message.get("tool_calls")
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
