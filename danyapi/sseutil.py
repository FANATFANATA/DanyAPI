from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Sequence
from functools import lru_cache
from typing import Any

log = logging.getLogger("danyapi.sseutil")

_SSE_LINE_RE = re.compile(r"\r\n|\r|\n")

MAX_BUFFER_BYTES = 8 * 1024 * 1024


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
    for line in _SSE_LINE_RE.split(data):
        if line == "":
            if data_lines:
                events.append(SSEEvent(event_name, _decode("\n".join(data_lines))))
            event_name = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].removeprefix(" ")
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].removeprefix(" "))
    if data_lines:
        events.append(SSEEvent(event_name, _decode("\n".join(data_lines))))
    return events


def split_stop(stop: Any) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    if isinstance(stop, list):
        return [item for item in stop if isinstance(item, str) and item]
    return []


MAIN_RESPONSE_TYPES = ("RESPONSE", "TEMPLATE_RESPONSE")
THINK_TYPES = ("THINK",)


_COMPACT_THRESHOLD = 8192


def _decode_text(raw: bytes | bytearray) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        log.warning("sse payload is not valid utf-8 at byte %d (%s), replaced %d byte(s)", exc.start, exc.reason, len(raw))
        return raw.decode("utf-8", errors="replace")


class IncrementalSSE:
    __slots__ = ("_buffer", "_pos")

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._pos = 0

    def _next_boundary(self) -> tuple[int, int]:
        buffer = self._buffer
        pos = self._pos
        lf = buffer.find(b"\n\n", pos)
        crlf = buffer.find(b"\r\n\r\n", pos)
        cr = buffer.find(b"\r\r", pos)
        best = len(buffer)
        step = 2
        for candidate, width in ((lf, 2), (crlf, 4), (cr, 2)):
            if candidate != -1 and candidate < best:
                best = candidate
                step = width
        return best, step

    def feed(self, chunk: bytes) -> Iterator[SSEEvent]:
        self._buffer += chunk
        buffer = self._buffer
        while True:
            idx, step = self._next_boundary()
            if idx == len(buffer):
                break
            start = self._pos
            self._pos = idx + step
            yield from parse_sse(_decode_text(buffer[start:idx]))
        if self._pos and (self._pos >= _COMPACT_THRESHOLD or self._pos == len(buffer)):
            del self._buffer[: self._pos]
            self._pos = 0
        if len(self._buffer) > MAX_BUFFER_BYTES:
            self._buffer = bytearray()
            self._pos = 0
            raise ValueError(f"sse buffer exceeded {MAX_BUFFER_BYTES} bytes without an event boundary")

    def finish(self) -> Iterator[SSEEvent]:
        tail = self._buffer[self._pos :]
        self._buffer = bytearray()
        self._pos = 0
        if tail.strip():
            yield from parse_sse(_decode_text(tail))


class StreamStopFilter:
    __slots__ = ("_buf", "_hold", "_markers")

    def __init__(self, markers: list[str]) -> None:
        self._markers = markers
        self._hold = max((len(marker) for marker in markers), default=1) - 1
        self._buf = ""

    def feed(self, piece: str) -> tuple[str, bool]:
        if not piece:
            return "", False
        text = self._buf + piece
        cut = -1
        for marker in self._markers:
            pos = text.find(marker)
            if pos != -1 and (cut == -1 or pos < cut):
                cut = pos
        if cut != -1:
            self._buf = ""
            return text[:cut], True
        hold = self._hold
        if hold <= 0:
            self._buf = ""
            return text, False
        if len(text) > hold:
            self._buf = text[-hold:]
            return text[:-hold], False
        self._buf = text
        return "", False

    def flush(self) -> str:
        out = self._buf
        self._buf = ""
        return out


@lru_cache(maxsize=2048)
def _path_parts(path: str) -> tuple[str, ...]:
    return tuple(part for part in path.split("/") if part)


def _normalise_key(key: str) -> str:
    return "id" if key == "message_id" else key


def _navigate(node: Any, parts: Sequence[str]) -> Any:
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


def _set_path(target: dict, parts: Sequence[str], value: Any) -> None:
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
                if part not in node:
                    if i:
                        return
                    node[part] = {}
                elif not isinstance(node[part], (dict, list)):
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
        name = _normalise_key(key)
        message[name] = list(val) if isinstance(val, list) else dict(val) if isinstance(val, dict) else val


def _delta_op(value: Any, default: str) -> str:
    return value if isinstance(value, str) else default


def _apply_delta(message: dict, op: str, path: str, value: Any) -> None:
    if op == "BATCH":
        values = value if isinstance(value, list) else []
        for sub in values:
            if not isinstance(sub, dict):
                continue
            sub_op = _delta_op(sub.get("o"), "SET")
            sub_path = sub.get("p", "")
            if sub_path and path and not sub_path.startswith("response/"):
                sub_path = f"{path}/{sub_path}"
            _apply_delta(message, sub_op, sub_path, sub.get("v"))
        return

    parts = _path_parts(path or "")
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


def _diff_suffix(previous: str, current: str) -> str:
    if current == previous:
        return ""
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
            if _touches_aggregated_fragment(_delta_op(sub.get("o"), "SET"), sub_path, sub.get("v"), frag_idx):
                return True
        return False
    parts = _path_parts(path or "")
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
        "_append_only",
        "_content_base",
        "_content_parts",
        "_diffs_revision",
        "_frag_idx",
        "_last_op",
        "_last_path",
        "_prev_content",
        "_prev_reasoning",
        "_reasoning_base",
        "_reasoning_parts",
        "_reported_c",
        "_reported_r",
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
        self._content_base = ""
        self._reasoning_base = ""
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._reported_c = 0
        self._reported_r = 0
        self._append_only = True
        self._aggregate_dirty = True
        self._diffs_revision = -1

    @property
    def _content(self) -> str:
        parts = self._content_parts
        if not parts:
            return self._content_base
        return self._content_base + "".join(parts)

    @_content.setter
    def _content(self, value: str) -> None:
        self._content_base = value
        self._content_parts.clear()
        self._reported_c = 0

    @property
    def _reasoning(self) -> str:
        parts = self._reasoning_parts
        if not parts:
            return self._reasoning_base
        return self._reasoning_base + "".join(parts)

    @_reasoning.setter
    def _reasoning(self, value: str) -> None:
        self._reasoning_base = value
        self._reasoning_parts.clear()
        self._reported_r = 0

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
        op = _delta_op(data.get("o"), self._last_op)
        path = data.get("p", self._last_path)
        if not isinstance(path, str):
            path = self._last_path
        if op != "BATCH":
            self._last_op = op
            self._last_path = path
        _apply_delta(self.message, op, path, data["v"])
        self._revision += 1
        self._aggregate_dirty = True
        if self._fast_append_tail(op, path, data["v"]):
            self._aggregate_dirty = False
        else:
            self._append_only = False
            if self._frag_idx and _touches_aggregated_fragment(op, path, data["v"], self._frag_idx):
                self._agg_fragments = None

    def _fast_append_tail(self, op: str, path: str, value: Any) -> bool:
        if op != "APPEND" or not isinstance(value, str):
            return False
        frags = self.message.get("fragments")
        if not isinstance(frags, list) or frags is not self._agg_fragments or not frags:
            return False
        if self._frag_idx != len(frags):
            return False
        parts = _path_parts(path)
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
            self._content_parts.append(value)
        elif frag_type in THINK_TYPES:
            self._reasoning_parts.append(value)
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
                        self._content_parts.append(_fragment_text(frag))
                    elif frag_type in THINK_TYPES:
                        self._reasoning_parts.append(_fragment_text(frag))
            self._frag_idx = len(frags)
            self._aggregate_dirty = False
            return self._content, self._reasoning
        if isinstance(frags, list):
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            for frag in frags:
                if isinstance(frag, dict):
                    frag_type = frag.get("type")
                    if frag_type in MAIN_RESPONSE_TYPES:
                        content_parts.append(_fragment_text(frag))
                    elif frag_type in THINK_TYPES:
                        reasoning_parts.append(_fragment_text(frag))
        else:
            content_parts = []
            reasoning_parts = []
        self._content = "".join(content_parts)
        self._reasoning = "".join(reasoning_parts)
        self._frag_idx = len(frags) if isinstance(frags, list) else 0
        self._agg_fragments = frags
        self._aggregate_dirty = False
        return self._content, self._reasoning

    @property
    def content(self) -> str:
        return self._aggregates()[0]

    @property
    def reasoning(self) -> str:
        return self._aggregates()[1]

    def _fold_content(self) -> str:
        parts = self._content_parts
        reported = self._reported_c
        if not parts:
            return ""
        if reported:
            self._content_base = self._content_base + "".join(parts[:reported])
            del parts[:reported]
        piece = "".join(parts)
        self._reported_c = len(parts)
        self._prev_content = self._content_base + piece
        return piece

    def _fold_reasoning(self) -> str:
        parts = self._reasoning_parts
        reported = self._reported_r
        if not parts:
            return ""
        if reported:
            self._reasoning_base = self._reasoning_base + "".join(parts[:reported])
            del parts[:reported]
        piece = "".join(parts)
        self._reported_r = len(parts)
        self._prev_reasoning = self._reasoning_base + piece
        return piece

    def take_diffs(self) -> tuple[str, str]:
        if self._revision == self._diffs_revision:
            return "", ""
        self._diffs_revision = self._revision
        frags = self.message.get("fragments")
        if self._append_only and frags is self._agg_fragments and not self._aggregate_dirty:
            return self._fold_content(), self._fold_reasoning()
        self._append_only = True
        content, reasoning = self._aggregates()
        c_diff = _diff_suffix(self._prev_content, content)
        r_diff = _diff_suffix(self._prev_reasoning, reasoning)
        self._prev_content, self._prev_reasoning = content, reasoning
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
