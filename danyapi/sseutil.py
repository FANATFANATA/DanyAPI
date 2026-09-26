from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any


class SSEEvent:
    __slots__ = ("data", "event")

    def __init__(self, event: str | None, data: Any) -> None:
        self.event = event
        self.data = data


def _decode(raw_data: str) -> Any:
    try:
        return json.loads(raw_data)
    except json.JSONDecodeError:
        return raw_data


def parse_sse(data: str) -> list[SSEEvent]:
    events: list[SSEEvent] = []
    event_name: str | None = None
    data_lines: list[str] = []
    for raw_line in data.split("\n"):
        line = raw_line.strip("\r")
        if line == "":
            if data_lines:
                events.append(SSEEvent(event_name, _decode("\n".join(data_lines))))
            event_name = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    if data_lines:
        events.append(SSEEvent(event_name, _decode("\n".join(data_lines))))
    return events


MAIN_RESPONSE_TYPES = ("RESPONSE", "TEMPLATE_RESPONSE")
THINK_TYPES = ("THINK",)


_COMPACT_THRESHOLD = 8192


class IncrementalSSE:
    __slots__ = ("_buffer", "_pos")

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._pos = 0

    def feed(self, chunk: bytes) -> Iterator[SSEEvent]:
        if not isinstance(self._buffer, bytearray):
            self._buffer = bytearray(self._buffer)
        self._buffer += chunk
        while True:
            idx = self._buffer.find(b"\n\n", self._pos)
            if idx == -1:
                idx = self._buffer.find(b"\r\n\r\n", self._pos)
                if idx == -1:
                    break
                block = self._buffer[self._pos : idx].decode("utf-8", errors="replace")
                self._pos = idx + 4
            else:
                block = self._buffer[self._pos : idx].decode("utf-8", errors="replace")
                self._pos = idx + 2
            yield from parse_sse(block)
        if self._pos and (self._pos >= _COMPACT_THRESHOLD or self._pos == len(self._buffer)):
            del self._buffer[: self._pos]
            self._pos = 0

    def finish(self) -> Iterator[SSEEvent]:
        tail = self._buffer[self._pos :]
        if tail.strip():
            yield from parse_sse(tail.decode("utf-8", errors="replace"))
        self._buffer = bytearray()
        self._pos = 0


def _normalise_key(key: str) -> str:
    return "id" if key == "message_id" else key


def _navigate(node: Any, parts: list[str]) -> Any:
    cur: Any = node
    for raw_part in parts:
        part = _normalise_key(raw_part)
        try:
            if isinstance(cur, list):
                idx = int(part)
                if idx >= len(cur) or idx < -len(cur):
                    return None
                cur = cur[idx]
            elif isinstance(cur, dict):
                cur = cur[part]
            else:
                return None
        except (KeyError, ValueError, TypeError):
            return None
    return cur


def _set_path(target: dict, parts: list[str], value: Any) -> None:
    node: Any = target
    for i, raw_part in enumerate(parts):
        part = _normalise_key(raw_part)
        if i == len(parts) - 1:
            if isinstance(node, dict):
                node[part] = value
            elif isinstance(node, list):
                try:
                    idx = int(part)
                except ValueError:
                    return
                if -len(node) <= idx < len(node):
                    node[idx] = value
            return
        try:
            if isinstance(node, list):
                idx = int(part)
                if idx >= len(node) or idx < -len(node):
                    return
                node = node[idx]
            elif isinstance(node, dict):
                if part not in node or not isinstance(node[part], (dict, list)):
                    node[part] = {}
                node = node[part]
            else:
                return
        except (KeyError, ValueError, TypeError):
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
            try:
                idx = int(key)
            except ValueError:
                return
            if idx < -len(node) or idx >= len(node):
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


def _diff_suffix(previous: str, current: str, prefix_ok: bool) -> str:
    if current == previous:
        return ""
    if prefix_ok and len(current) > len(previous) and current.startswith(previous):
        return current[len(previous) :]
    return current.removeprefix(previous)


def _touches_aggregated_fragment(op: str, path: str, value: Any, frag_idx: int) -> bool:
    if op == "BATCH":
        values = value if isinstance(value, list) else []
        for sub in values:
            if not isinstance(sub, dict):
                continue
            sub_path = sub.get("p", "")
            if sub_path and path and not sub_path.startswith("response/"):
                sub_path = f"{path}/{sub_path}"
            if _touches_aggregated_fragment(sub.get("o", "SET"), sub_path, sub.get("v"), frag_idx):
                return True
        return False
    parts = [p for p in (path or "").split("/") if p]
    if len(parts) < 3 or parts[0] != "response" or parts[1] != "fragments":
        return False
    try:
        idx = int(parts[2])
    except ValueError:
        return False
    return idx < frag_idx


class MessageReconstructor:
    __slots__ = (
        "_agg_fragments",
        "_aggregate_dirty",
        "_content",
        "_content_prefix_ok",
        "_diffs_revision",
        "_frag_idx",
        "_last_op",
        "_last_path",
        "_prev_content",
        "_prev_reasoning",
        "_reasoning",
        "_reasoning_prefix_ok",
        "_revision",
        "hint_error",
        "message",
        "response_message_id",
    )

    def __init__(self) -> None:
        self.message: dict = {}
        self._last_op = "SET"
        self._last_path = ""
        self._prev_content = ""
        self._prev_reasoning = ""
        self.response_message_id: str | None = None
        self.hint_error: dict | None = None
        self._revision = 0
        self._agg_fragments: Any = None
        self._frag_idx = 0
        self._content = ""
        self._reasoning = ""
        self._aggregate_dirty = True
        self._content_prefix_ok = True
        self._reasoning_prefix_ok = True
        self._diffs_revision = -1

    def handle(self, event: SSEEvent) -> None:
        if event.event == "ready":
            if isinstance(event.data, dict):
                self.response_message_id = event.data.get("response_message_id")
            return
        if event.event in ("toast", "hint"):
            if isinstance(event.data, dict) and event.data.get("type") == "error":
                self.hint_error = {
                    "message": event.data.get("content") or event.data.get("message") or "",
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
        if op != "BATCH":
            self._last_op = op
            self._last_path = path
        _apply_delta(self.message, op, path, data["v"])
        self._revision += 1
        self._aggregate_dirty = True
        if self._fast_append_tail(op, path, data["v"]):
            self._aggregate_dirty = False
        elif self._frag_idx and _touches_aggregated_fragment(op, path, data["v"], self._frag_idx):
            self._agg_fragments = None

    def _fast_append_tail(self, op: str, path: str, value: Any) -> bool:
        if op != "APPEND" or not isinstance(value, str):
            return False
        frags = self.message.get("fragments")
        if not isinstance(frags, list) or frags is not self._agg_fragments or not frags:
            return False
        if self._frag_idx != len(frags):
            return False
        parts = [p for p in path.split("/") if p]
        if len(parts) != 4 or parts[0] != "response" or parts[1] != "fragments" or parts[3] != "content":
            return False
        try:
            index = int(parts[2])
        except ValueError:
            return False
        if index != self._frag_idx - 1:
            return False
        tail = frags[self._frag_idx - 1]
        if not isinstance(tail, dict):
            return False
        frag_type = tail.get("type")
        if frag_type in MAIN_RESPONSE_TYPES:
            self._content += value
        elif frag_type in THINK_TYPES:
            self._reasoning += value
        else:
            return False
        return True

    def _aggregates(self) -> tuple[str, str]:
        frags = self.message.get("fragments")
        if frags is self._agg_fragments and not self._aggregate_dirty:
            return self._content, self._reasoning
        if isinstance(frags, list) and frags is self._agg_fragments and len(frags) > self._frag_idx:
            for i in range(self._frag_idx, len(frags)):
                frag = frags[i]
                if isinstance(frag, dict):
                    frag_type = frag.get("type")
                    if frag_type in MAIN_RESPONSE_TYPES:
                        self._content += _fragment_text(frag)
                    elif frag_type in THINK_TYPES:
                        self._reasoning += _fragment_text(frag)
            self._frag_idx = len(frags)
            self._aggregate_dirty = False
            return self._content, self._reasoning
        if isinstance(frags, list):
            content = ""
            reasoning = ""
            for frag in frags:
                if isinstance(frag, dict):
                    frag_type = frag.get("type")
                    if frag_type in MAIN_RESPONSE_TYPES:
                        content += _fragment_text(frag)
                    elif frag_type in THINK_TYPES:
                        reasoning += _fragment_text(frag)
        else:
            content = ""
            reasoning = ""
        self._content = content
        self._reasoning = reasoning
        self._frag_idx = len(frags) if isinstance(frags, list) else 0
        self._agg_fragments = frags
        self._aggregate_dirty = False
        self._content_prefix_ok = False
        self._reasoning_prefix_ok = False
        return content, reasoning

    @property
    def content(self) -> str:
        return self._aggregates()[0]

    @property
    def reasoning(self) -> str:
        return self._aggregates()[1]

    def take_diffs(self) -> tuple[str, str]:
        if self._revision == self._diffs_revision:
            return "", ""
        content, reasoning = self._aggregates()
        c_diff = _diff_suffix(self._prev_content, content, self._content_prefix_ok)
        r_diff = _diff_suffix(self._prev_reasoning, reasoning, self._reasoning_prefix_ok)
        self._prev_content, self._prev_reasoning = content, reasoning
        self._content_prefix_ok = True
        self._reasoning_prefix_ok = True
        self._diffs_revision = self._revision
        return c_diff, r_diff

    def extend_with(self, other: MessageReconstructor) -> None:
        old_content, old_reasoning = self._aggregates()
        other_fragments = (other.message or {}).get("fragments")
        if isinstance(other_fragments, list) and other_fragments:
            fragments = self.message.get("fragments")
            if isinstance(fragments, list):
                fragments.extend(other_fragments)
            else:
                self.message["fragments"] = list(other_fragments)
        if other.id:
            self.message["id"] = other.id
        if other.status:
            self.message["status"] = other.status
        if other.accumulated_tokens:
            self.message["accumulated_token_usage"] = other.accumulated_tokens
        self.hint_error = other.hint_error
        self._prev_content = old_content
        self._prev_reasoning = old_reasoning
        self._revision += 1
        self._aggregate_dirty = True

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
