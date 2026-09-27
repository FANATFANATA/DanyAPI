from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from .dsml import _XML_HTML_TAGS

TOOL_STREAM_TAGS = (
    "tool_calls",
    "tool_call",
    "function_calls",
    "function_call",
    "functions",
    "function",
    "tools",
    "calls",
    "_calls",
    "toolinvoke",
    "tool_invoke",
    "use_tool",
    "tool_use",
    "invoke",
    "call",
    "action",
    "run",
)

TOOL_STREAM_JSON_KEYS = (
    "tool_calls",
    "calls",
    "_calls",
    "function_call",
    "tool_name",
    "name",
    "tool",
    "action",
    "call",
)

TOOL_STREAM_MARKERS = tuple([f'{{"{key}"' for key in TOOL_STREAM_JSON_KEYS] + [f"<{tag}" for tag in TOOL_STREAM_TAGS] + ["tool_calls:", "[{"])

TOOL_STREAM_MARKER_MAX = max(len(marker) for marker in TOOL_STREAM_MARKERS)

_MARKER_PREFIXES = frozenset(marker[:size] for marker in TOOL_STREAM_MARKERS for size in range(1, len(marker) + 1))
_JSON_KEY_PREFIXES = frozenset(key[:size] for key in TOOL_STREAM_JSON_KEYS for size in range(1, len(key) + 1))

_TOOL_STREAM_TAG_RE = re.compile(
    r"<\s*/?\s*(?:" + "|".join(TOOL_STREAM_TAGS) + r")\b[^<>]*>",
    re.IGNORECASE,
)
_TOOL_STREAM_JSON_RE = re.compile(r"\{\s*['\"]?(?:" + "|".join(TOOL_STREAM_JSON_KEYS) + r")['\"]?\s*:")
_TOOL_STREAM_ARRAY_RE = re.compile(r"\[\s*\{")
_TOOL_STREAM_YAML_RE = re.compile(r"tool_calls\s*:")
_TOOL_STREAM_NAME_ATTR_RE = re.compile(
    r"<\s*/?\s*(?!(?:" + "|".join(sorted(_XML_HTML_TAGS)) + r")\b)[A-Za-z_][A-Za-z0-9_.-]*[^<>]*\bname\s*=",
    re.IGNORECASE,
)
_DSML_STREAM_START = re.compile(
    r"<\s*/?\s*(?:[|]|[^\x00-\x7f]){1,8}\s*DSML\s*(?:[|]|[^\x00-\x7f]){1,8}",
    re.IGNORECASE | re.DOTALL,
)
_BOUNDARY_SCAN_RE = re.compile(r"[{\[<>\n]")
_TAG_NAME_RUN_RE = re.compile(r"[A-Za-z0-9_.-]{1,256}")
_TAG_HOLD_PREFIXES = frozenset(tag[:size] for tag in TOOL_STREAM_TAGS for size in range(1, len(tag) + 1))
_MARKER_GROUPS: dict[str, tuple[str, ...]] = {}
for _marker in TOOL_STREAM_MARKERS:
    _MARKER_GROUPS[_marker[0]] = (*_MARKER_GROUPS.get(_marker[0], ()), _marker)
_PREFIX_BY_LEN: dict[int, tuple[str, ...]] = {}
for _size in range(1, TOOL_STREAM_MARKER_MAX + 1):
    _found = tuple(sorted(prefix for prefix in _MARKER_PREFIXES if len(prefix) == _size))
    if _found:
        _PREFIX_BY_LEN[_size] = _found


@lru_cache(maxsize=64)
def _stream_patterns(names: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    patterns = [
        _TOOL_STREAM_TAG_RE,
        _TOOL_STREAM_JSON_RE,
        _TOOL_STREAM_ARRAY_RE,
        _TOOL_STREAM_NAME_ATTR_RE,
    ]
    if names:
        escaped = "|".join(re.escape(name) for name in names)
        patterns.append(re.compile(rf"<\s*/?\s*(?:{escaped})\b", re.IGNORECASE))
    return tuple(patterns)


def _line_head(text: str, pos: int) -> int:
    while pos > 0 and text[pos - 1] in " \t":
        pos -= 1
    return pos if pos == 0 or text[pos - 1] == "\n" else -1


def _yaml_marker(text: str, start: int) -> int:
    pos = start
    while True:
        match = _TOOL_STREAM_YAML_RE.search(text, pos)
        if match is None:
            return -1
        head = _line_head(text, match.start())
        if head != -1:
            return head
        pos = match.end()


def _call_marker(text: str, start: int, names: tuple[str, ...]) -> int:
    best = -1
    size = len(text)
    for name in names:
        at = text.find(name, start)
        while at != -1:
            head = _line_head(text, at)
            if head != -1:
                after = at + len(name)
                while after < size and text[after] in " \t":
                    after += 1
                if after < size and text[after] == "(":
                    if best == -1 or head < best:
                        best = head
                    break
            at = text.find(name, at + 1)
    return best


def _stream_names(tool_schemas: dict[str, dict[str, Any]] | None) -> tuple[str, ...]:
    if not tool_schemas or not isinstance(tool_schemas, dict):
        return ()
    return _stream_names_keys(tuple(tool_schemas))


@lru_cache(maxsize=64)
def _stream_names_keys(keys: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(sorted(name.lower() for name in keys if isinstance(name, str) and name))


def _literal_hold(text: str, start: int) -> int:
    length = len(text)
    max_size = min(TOOL_STREAM_MARKER_MAX - 1, length - start)
    if max_size <= 0:
        return -1
    tail = text[length - max_size :]
    for size in range(len(tail), 0, -1):
        if tail.endswith(_PREFIX_BY_LEN[size]):
            return length - size
    return -1


def _json_hold(text: str, start: int) -> int:
    brace = text.rfind("{", start)
    if brace == -1 or "}" in text[brace:]:
        return -1
    body = text[brace + 1 : brace + 24].lstrip()
    if not body:
        return brace
    if body[0] in "'\"":
        body = body[1:]
    key = body.lower()
    if key in _JSON_KEY_PREFIXES:
        return brace
    return -1


def _tag_hold(text: str, start: int, names: tuple[str, ...]) -> int:
    lt = text.rfind("<", start)
    if lt == -1:
        return -1
    tail = text[lt:]
    if ">" in tail:
        return -1
    body = tail[1:].lstrip()
    if body.startswith("/"):
        body = body[1:].lstrip()
    if not body:
        return lt
    first = body[0]
    if first == "|" or first > "\x7f":
        return lt
    name_match = _TAG_NAME_RUN_RE.match(body)
    name = name_match.group(0).lower() if name_match is not None else ""
    if not name:
        return -1
    if name in _XML_HTML_TAGS:
        return -1
    if name in _TAG_HOLD_PREFIXES or any(candidate.startswith(name) for candidate in names):
        return lt
    lowered = body.lower()
    for suffix in ("name", "nam", "na", "n"):
        if lowered.endswith(suffix):
            before = lowered[: len(lowered) - len(suffix)]
            if not before or before[-1] in " \t_-\"'=<>":
                return lt
    if "name" in lowered:
        return lt
    return -1


def _array_hold(text: str, start: int) -> int:
    bracket = text.rfind("[", start)
    if bracket == -1 or "]" in text[bracket:]:
        return -1
    if not text[bracket + 1 :].strip():
        return bracket
    return -1


_boundary_cache: dict[tuple[str, ...], tuple[str, int, bool]] = {}
_BOUNDARY_CACHE_MAX = 256
_BOUNDARY_CACHE_PREFIX = 256


def _boundary_hold(text: str, start: int, names: tuple[str, ...]) -> int:
    hold = -1
    for candidate in (
        _literal_hold(text, start),
        _json_hold(text, start),
        _tag_hold(text, start, names),
        _array_hold(text, start),
    ):
        if candidate != -1 and (hold == -1 or candidate < hold):
            hold = candidate
    return hold


def _boundary_may_start(text: str, start: int, names: tuple[str, ...]) -> bool:
    if _BOUNDARY_SCAN_RE.search(text, start) is not None:
        return True
    if start > 0 and text[start - 1] != "\n":
        return False
    head = text[start : start + 24].lstrip(" \t")
    if not head:
        return bool(text[start : start + 1])
    first = head[0].lower()
    return first == "t" or any(name.startswith(first) for name in names)


def tool_call_boundary(
    text: str,
    start: int = 0,
    tool_schemas: dict[str, dict[str, Any]] | None = None,
) -> tuple[int, bool]:
    names = _stream_names(tool_schemas)
    cached = _boundary_cache.get(names)
    if cached is not None:
        old_text, old_best, old_complete = cached
        prefix_ok = len(text) >= len(old_text) and text.startswith(old_text)
        if old_complete and 0 <= old_best < len(old_text) and start <= old_best and prefix_ok:
            hold = _boundary_hold(text, start, names)
            if hold != -1 and hold < old_best:
                return hold, False
            return old_best, True
    if not _boundary_may_start(text, start, names):
        hold = _boundary_hold(text, start, names)
        return (hold, False) if hold != -1 else (-1, False)
    best = -1
    complete = False
    for pattern in _stream_patterns(names):
        match = pattern.search(text, start)
        if match is not None and (best == -1 or match.start() < best):
            best = match.start()
            complete = True
    if best == start:
        return best, True
    head = _yaml_marker(text, start)
    if head != -1 and (best == -1 or head < best):
        best = head
        complete = True
    if names:
        head = _call_marker(text, start, names)
        if head != -1 and (best == -1 or head < best):
            best = head
            complete = True
    match = _DSML_STREAM_START.search(text, start)
    if match is not None and (best == -1 or match.start() < best):
        best = match.start()
        complete = True
    for char, group in _MARKER_GROUPS.items():
        at = text.find(char, start)
        if at == -1 or (best != -1 and at > best):
            continue
        for marker in group:
            pos = text.find(marker, start)
            if pos != -1 and (best == -1 or pos < best):
                best = pos
                complete = True
                if best == start:
                    break
        if best == start:
            break
    hold = _boundary_hold(text, start, names)
    if hold != -1 and (best == -1 or hold < best):
        return hold, False
    if best != -1 and best < _BOUNDARY_CACHE_PREFIX and (not _boundary_cache or len(_boundary_cache) < _BOUNDARY_CACHE_MAX):
        _boundary_cache[names] = (text[:_BOUNDARY_CACHE_PREFIX], best, complete)
    return best, complete


def tool_visible(
    content_buf: str,
    shown: int,
    hidden: bool,
    tool_schemas: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, int, bool]:
    if hidden:
        return "", shown, True
    boundary, complete = tool_call_boundary(content_buf, shown, tool_schemas)
    if boundary < 0:
        return content_buf[shown:], len(content_buf), False
    return content_buf[shown:boundary], boundary, complete
