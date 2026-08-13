from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .sse import SSEEvent, parse_sse

MAIN_RESPONSE_TYPES = ("RESPONSE", "TEMPLATE_RESPONSE")
THINK_TYPES = ("THINK",)


class IncrementalSSE:
    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, chunk: bytes) -> Iterator[SSEEvent]:
        self._buffer += chunk
        while True:
            idx = self._buffer.find(b"\n\n")
            if idx == -1:
                idx = self._buffer.find(b"\r\n\r\n")
                if idx == -1:
                    break
                block = self._buffer[:idx].decode("utf-8", errors="replace")
                self._buffer = self._buffer[idx + 4 :]
            else:
                block = self._buffer[:idx].decode("utf-8", errors="replace")
                self._buffer = self._buffer[idx + 2 :]
            yield from parse_sse(block)

    def finish(self) -> Iterator[SSEEvent]:
        if self._buffer.strip():
            yield from parse_sse(self._buffer.decode("utf-8", errors="replace"))
        self._buffer = b""


def _normalise_key(key: str) -> str:
    return "id" if key == "message_id" else key


def _navigate(node: Any, parts: list[str]) -> Any:
    cur: Any = node
    for raw_part in parts:
        part = _normalise_key(raw_part)
        if isinstance(cur, list):
            idx = int(part)
            cur = cur[idx]
        elif isinstance(cur, dict):
            cur = cur[part]
        else:
            return None
    return cur


def _set_path(target: dict, parts: list[str], value: Any) -> None:
    node: Any = target
    for i, raw_part in enumerate(parts):
        part = _normalise_key(raw_part)
        if i == len(parts) - 1:
            if isinstance(node, dict):
                node[part] = value
            return
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict):
            if part not in node or not isinstance(node[part], (dict, list)):
                node[part] = {}
            node = node[part]
        else:
            return


def _init_message(message: dict, value: Any) -> None:
    source = value.get("response") if isinstance(value, dict) else None
    if not isinstance(source, dict):
        return
    message.clear()
    for key, val in source.items():
        message[_normalise_key(key)] = val


def _apply_delta(message: dict, op: str, path: str, value: Any) -> None:
    if op == "BATCH":
        values = value if isinstance(value, list) else []
        for sub in values:
            if not isinstance(sub, dict):
                continue
            sub_op = sub.get("o", "SET")
            sub_path = sub.get("p", "")
            if sub_path and path and not sub_path.startswith("response/"):
                sub_path = f"{path}/{sub_path}"
            _apply_delta(message, sub_op, sub_path, sub.get("v"))
        return

    parts = [p for p in (path or "").split("/") if p]
    if not parts:
        _init_message(message, value)
        return
    if parts[0] != "response":
        return
    rest = parts[1:]
    if not rest:
        if op == "SET":
            _init_message(message, value)
        return

    if op == "APPEND":
        node = _navigate(message, rest[:-1])
        key = _normalise_key(rest[-1])
        if isinstance(node, list):
            idx = int(key)
            if idx >= len(node) or idx < 0:
                return
            cur = node[idx]
            if isinstance(cur, str) and isinstance(value, str):
                node[idx] = cur + value
        elif isinstance(node, dict):
            if key not in node:
                node[key] = value
            elif isinstance(node[key], str) and isinstance(value, str):
                node[key] += value
            elif isinstance(node[key], list):
                if isinstance(value, list):
                    node[key].extend(value)
                else:
                    node[key].append(value)
            else:
                node[key] = value
        return

    if op == "SET":
        _set_path(message, rest, value)


def _fragment_text(fragment: Any) -> str:
    if isinstance(fragment, str):
        return fragment
    if not isinstance(fragment, dict):
        return ""
    content = fragment.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                out.append(item["text"])
        return "".join(out)
    return ""


class MessageReconstructor:
    def __init__(self) -> None:
        self.message: dict = {}
        self._last_op = "SET"
        self._last_path = ""
        self._prev_content = ""
        self._prev_reasoning = ""
        self.response_message_id: str | None = None
        self.hint_error: dict | None = None

    def handle(self, event: SSEEvent) -> None:
        if event.event == "ready":
            if isinstance(event.data, dict):
                self.response_message_id = event.data.get("response_message_id")
            return
        if event.event in ("toast", "hint"):
            if isinstance(event.data, dict) and event.data.get("type") == "error":
                self.hint_error = {
                    "message": event.data.get("content", ""),
                    "finish_reason": event.data.get("finish_reason"),
                }
            return
        if event.event not in (None, "delta"):
            return
        data = event.data
        if not isinstance(data, dict) or "v" not in data:
            return
        op = data.get("o", self._last_op)
        path = data.get("p", self._last_path)
        self._last_op = op
        self._last_path = path
        _apply_delta(self.message, op, path, data["v"])

    def _aggregate(self, types: tuple[str, ...]) -> str:
        fragments = self.message.get("fragments") or []
        parts = []
        for frag in fragments:
            if isinstance(frag, dict) and frag.get("type") in types:
                parts.append(_fragment_text(frag))
        return "".join(parts)

    @property
    def content(self) -> str:
        return self._aggregate(MAIN_RESPONSE_TYPES)

    @property
    def reasoning(self) -> str:
        return self._aggregate(THINK_TYPES)

    def take_diffs(self) -> tuple[str, str]:
        content, reasoning = self.content, self.reasoning
        c_diff = content.removeprefix(self._prev_content)
        r_diff = reasoning.removeprefix(self._prev_reasoning)
        self._prev_content, self._prev_reasoning = content, reasoning
        return c_diff, r_diff

    @property
    def status(self) -> str | None:
        return self.message.get("status")

    @property
    def id(self) -> str | None:
        return self.message.get("id")

    @property
    def accumulated_tokens(self) -> int:
        value = self.message.get("accumulated_token_usage")
        return int(value) if isinstance(value, (int, float)) and value > 0 else 0

    @property
    def usage(self) -> dict:
        total = self.accumulated_tokens
        return {"prompt_tokens": 0, "completion_tokens": total, "total_tokens": total}
