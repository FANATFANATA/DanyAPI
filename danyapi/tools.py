from __future__ import annotations

import hashlib
import json
import re
import uuid
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache
from typing import Any

_DSML_PIPE = r"|\u00a6\u01c0\u01c1\u05c0\u2016\u2223\u2502\u2551\u2758\ufe31\uff5c"
_DSML_CHAR = r"(?:[|]|[^\x00-\x7f])"
_DSML_MARKER = rf"{_DSML_CHAR}+\s*DSML\s*{_DSML_CHAR}+"
_DSML_PIPE_ANGLE = rf"[{_DSML_PIPE}<>]"
_DSML_BLOCK = re.compile(
    rf"<{_DSML_PIPE_ANGLE}+\s*[a-zA-Z_][^<>]*\s*{_DSML_PIPE_ANGLE}+>\s*DSML\s*<{_DSML_PIPE_ANGLE}+\s*[a-zA-Z_][^<>]*\s*{_DSML_PIPE_ANGLE}+>",
    re.IGNORECASE,
)
_DSML_WRAP = re.compile(
    rf"{_DSML_CHAR}+\s*>\s*DSML\s*<\s*{_DSML_CHAR}+",
    re.IGNORECASE,
)
_DSML_XML_NORMALIZE = re.compile(rf"<\s*(/?)\s*{_DSML_MARKER}\s*([a-zA-Z_][^<>]*)>", re.IGNORECASE)
_DSML_TAG = re.compile(rf"<\s*/?\s*{_DSML_MARKER}\s*[^<>]*>", re.IGNORECASE)
_DSML_NAKED = re.compile(rf"{_DSML_MARKER}", re.IGNORECASE)

_DSML_TOOL_CALLS_BLOCK = re.compile(
    rf"<{_DSML_MARKER}\s*tool_calls\b[^<>]*>(.*?)</{_DSML_MARKER}\s*tool_calls\s*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_INVOKE = re.compile(
    rf"<{_DSML_MARKER}\s*invoke\b[^>]*?\sname\s*=\s*([\"']?)([^\s>\"']+)\1[^>]*>(.*?)</{_DSML_MARKER}\s*invoke\s*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_PARAMETER = re.compile(
    rf"<{_DSML_MARKER}\s*parameter\s+name\s*=\s*([\"']?)([^\"']+)\1[^>]*>(.*?)</{_DSML_MARKER}\s*parameter\s*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_HIDDEN_NAMES = (
    r"thinking|reasoning|thought|analysis|summary|abbreviation|"
    r"ds_safety|ds_sensitive|ds_core|ds_middle|ds_end|ds_pii|ds_related|"
    r"ds_rephrase|ds_translate|ds_bilingual|ds_inner|ds_header|ds_web_search|"
    r"search|result|reference|quote"
)
_DSML_HIDDEN_PATS = tuple(
    re.compile(
        rf"<{_DSML_MARKER}\s*{name}\b[^<>]*>.*?</{_DSML_MARKER}\s*{name}\s*>",
        re.DOTALL | re.IGNORECASE,
    )
    for name in _DSML_HIDDEN_NAMES.split("|")
)
_DSML_HIDDEN_NAKED_PATS = tuple(
    re.compile(
        rf"{_DSML_MARKER}\s*<{name}\b[^<>]*>.*?</{name}>\s*{_DSML_MARKER}",
        re.DOTALL | re.IGNORECASE,
    )
    for name in _DSML_HIDDEN_NAMES.split("|")
)
_DSML_EQUALS = r"=\uff1d"
_DSML_LAX_MARKER = rf"(?:{_DSML_MARKER}|{_DSML_CHAR}+)"
_DSML_LAX_SKIP_TAGS = frozenset(
    {"parameter", "tool_calls", "tool_call", "function_call", "function_calls", "call", "calls", "tool", "functions", "tools", "name"}
    | set(_DSML_HIDDEN_NAMES.split("|"))
)
_DSML_LAX_TAG = re.compile(
    rf"</?{_DSML_LAX_MARKER}\s*[a-zA-Z_][a-zA-Z0-9_-]*\b[^<>]*>",
    re.IGNORECASE,
)
_DSML_LAX_BLOCK = re.compile(
    rf"<{_DSML_LAX_MARKER}\s*(?:tool_calls|calls)\b[^<>]*>(?P<body>.*?)</{_DSML_LAX_MARKER}\s*(?:tool_calls|calls)\s*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_LAX_OPENANY = re.compile(
    rf"<(?P<sep>{_DSML_LAX_MARKER})\s*(?P<tagname>[a-zA-Z_][a-zA-Z0-9_-]*)\b(?P<attrs>[^>]*)>",
    re.IGNORECASE,
)
_DSML_LAX_NAME_ATTR = re.compile(rf"\bname\s*[{_DSML_EQUALS}]\s*([\"'])([^\"']+)\1", re.IGNORECASE)
_DSML_LAX_TOOLNAME_TAIL = re.compile(rf"(?<!parameter\s)\bname\s*[{_DSML_EQUALS}]\s*([\"'])([^\"']+)\1", re.IGNORECASE)
_DSML_LAX_PARAMETER = re.compile(
    rf"(?:<{_DSML_LAX_MARKER}\s*)?parameter\b\s+name\s*[{_DSML_EQUALS}]\s*"
    rf"([\"']?)(?P<name>[^\"']+)\1[^>]*>(?P<value>.*?)"
    rf"(?:</?{_DSML_LAX_MARKER}\s*parameter\s*>|/?\s*parameter\s*>|</?parameter\s*>|/?parameter\s*>)",
    re.DOTALL | re.IGNORECASE,
)
_XML_SELFCLOSE = re.compile(r"<([a-zA-Z_][a-zA-Z0-9_-]*)\b([^>]*?)/>", re.DOTALL | re.IGNORECASE)
_XML_OPEN_TAG_SCAN = re.compile(r"<\s*([a-zA-Z_][a-zA-Z0-9_-]*)\b([^>]*)>", re.IGNORECASE)


def _find_xml_close(text: str, name: str, start: int, name_space: bool, tail_space: bool) -> tuple[int, int] | None:
    signature = name.lower()
    signature_len = len(signature)
    found = start
    length = len(text)
    while True:
        lt = text.find("</", found)
        if lt == -1:
            return None
        j = lt + 2
        if name_space:
            while j < length and text[j] in " \t\r\n":
                j += 1
        if text[j : j + signature_len].lower() == signature:
            k = j + signature_len
            if tail_space:
                while k < length and text[k] in " \t\r\n":
                    k += 1
            if k < length and text[k] == ">":
                return lt, k + 1
        found = lt + 1


def _scan_xml_pairs(
    text: str,
    name_filter: frozenset[str] | None = None,
    name_space: bool = False,
    tail_space: bool = False,
) -> Iterator[tuple[int, int, str, str, str]]:
    pos = 0
    length = len(text)
    while pos < length:
        open_match = _XML_OPEN_TAG_SCAN.search(text, pos)
        if open_match is None:
            return
        name = open_match.group(1)
        if name_filter is not None and name.lower() not in name_filter:
            pos = open_match.end()
            continue
        close = _find_xml_close(text, name, open_match.end(), name_space, tail_space)
        if close is None:
            body_start = open_match.end()
            if name_filter is None or open_match.group(2).rstrip().endswith("/"):
                pos = body_start
                continue
            if text.rfind("<", body_start) > text.rfind(">", body_start):
                pos = body_start
                continue
            wrapper_close = _XML_WRAPPER_CLOSE_RE.search(text, body_start)
            trunc = wrapper_close.start() if wrapper_close is not None else length
            if _XML_STRAY_TOOL_CLOSE_RE.search(text, body_start, trunc):
                pos = body_start
                continue
            close = (trunc, trunc)
        close_start, end = close
        yield open_match.start(), end, name, open_match.group(2), text[open_match.end() : close_start]
        pos = end


class _IntervalSet:
    __slots__ = ("ends", "starts")

    def __init__(self) -> None:
        self.starts: list[int] = []
        self.ends: list[int] = []

    def add(self, start: int, end: int) -> None:
        i = bisect_right(self.ends, start)
        if i > 0 and start <= self.ends[i - 1]:
            i -= 1
            start = min(start, self.starts[i])
            end = max(end, self.ends[i])
            while i + 1 < len(self.ends) and self.starts[i + 1] <= end:
                end = max(end, self.ends[i + 1])
                del self.starts[i + 1]
                del self.ends[i + 1]
            self.starts[i] = start
            self.ends[i] = end
        else:
            self.starts.insert(i, start)
            self.ends.insert(i, end)

    def contains(self, start: int, end: int) -> bool:
        i = bisect_right(self.starts, start) - 1
        return i >= 0 and self.ends[i] >= end


def _blanked(text: str, mask: bytearray) -> str:
    parts: list[str] = []
    cursor = 0
    i = 0
    n = len(text)
    while i < n:
        if mask[i]:
            parts.append(text[cursor:i])
            while i < n and mask[i]:
                i += 1
            parts.append(" ")
            cursor = i
        else:
            i += 1
    parts.append(text[cursor:])
    return "".join(parts)


_XML_WRAPPER_OPEN = re.compile(
    r"<(?:tool_calls|tool_call|function_calls|function_call|tools|calls|_calls)\b[^>]*>",
    re.IGNORECASE,
)
_XML_SKIP_ELEMENTS = frozenset(
    {
        "calls",
        "_calls",
        "tool_calls",
        "tool_call",
        "function_calls",
        "function_call",
        "functions",
        "tools",
        "invoke",
        "use_tool",
        "tool_use",
        "tool",
        "function",
        "parameter",
        "thinking",
        "reasoning",
        "thought",
        "analysis",
    }
)
_XML_GENERIC_TOOL_TAGS = frozenset(
    {
        "invoke",
        "toolinvoke",
        "tool_invoke",
        "use_tool",
        "tool_use",
        "call",
        "tool_call",
        "function_call",
        "action",
        "run",
    }
)
_XML_OPEN_TAG = re.compile(
    r"<(?:tool_calls|tool_call|function_calls|function_call|functions|function|tools|calls|_calls)\b[^>]*>",
    re.IGNORECASE,
)
_XML_CLOSE_TAG = re.compile(
    r"</(?:tool_calls|tool_call|function_calls|function_call|functions|function|tools|calls|_calls)\s*>",
    re.IGNORECASE,
)
_XML_HTML_TAGS = frozenset(
    {
        "a",
        "abbr",
        "address",
        "area",
        "article",
        "aside",
        "audio",
        "b",
        "base",
        "bdi",
        "bdo",
        "blockquote",
        "body",
        "br",
        "button",
        "canvas",
        "caption",
        "cite",
        "code",
        "col",
        "colgroup",
        "data",
        "datalist",
        "dd",
        "del",
        "details",
        "dfn",
        "dialog",
        "div",
        "dl",
        "dt",
        "em",
        "embed",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "head",
        "header",
        "hgroup",
        "hr",
        "html",
        "i",
        "iframe",
        "img",
        "input",
        "ins",
        "kbd",
        "label",
        "legend",
        "li",
        "link",
        "main",
        "map",
        "mark",
        "menu",
        "meta",
        "meter",
        "nav",
        "noscript",
        "object",
        "ol",
        "optgroup",
        "option",
        "output",
        "p",
        "param",
        "picture",
        "pre",
        "progress",
        "q",
        "rp",
        "rt",
        "ruby",
        "s",
        "samp",
        "script",
        "section",
        "select",
        "slot",
        "small",
        "source",
        "span",
        "strong",
        "style",
        "sub",
        "summary",
        "sup",
        "table",
        "tbody",
        "td",
        "template",
        "textarea",
        "tfoot",
        "th",
        "thead",
        "time",
        "title",
        "tr",
        "track",
        "u",
        "ul",
        "var",
        "video",
        "wbr",
    }
)
_ARGS_ALIASES = ("arguments", "args", "params", "parameters", "input")
_NAME_ALIASES = ("name", "tool", "action", "tool_name", "call")
_JSON_TYPE_ATTRS = frozenset({"string", "boolean", "integer", "number", "object", "array", "null"})
_FENCES_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)
_XML_PARAM_RE = re.compile(
    r'<\s*parameter\b[^>]*?\bname\s*=\s*(["\'])([^"\']+)\1[^>]*?>'
    r"(.*?)"
    r"(?:</\s*parameter\s*>|(?=</?\s*(?:tool_calls|tool_call|function_calls|function_call|calls|invoke|parameter)\b)|$)",
    re.DOTALL | re.IGNORECASE,
)
_XML_ATTR_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_.-]*)\s*=\s*(\"[^\"]*\"|'[^']*')", re.IGNORECASE)
_XML_NESTED_RE = re.compile(r"<[a-zA-Z_]")
_XML_WRAPPER_CLOSE_RE = re.compile(r"</(?:tool_calls|tool_call|function_calls|function_call|tools|calls|_calls)\s*>", re.IGNORECASE)
_XML_STRAY_TOOL_CLOSE_RE = re.compile(
    r"</\s*(?:invoke|toolinvoke|tool_invoke|use_tool|tool_use|call|function|tool)\s*>",
    re.IGNORECASE,
)
_XML_TOOL_NAMES = r"invoke|toolinvoke|tool_invoke|use_tool|tool_use|call|function|tool"
_TOOL_TAG_NAMES = frozenset(name.strip().lower() for name in _XML_TOOL_NAMES.split("|"))
_XML_TOOL_SELFCLOSE_RE = re.compile(
    r"<(?:invoke|toolinvoke|tool_invoke|use_tool|tool_use|call|function|tool)\b([^>]*?)/>",
    re.DOTALL | re.IGNORECASE,
)
_XML_NAME_ATTR_RE = re.compile(r"\bname\s*=\s*([\"']?)([^\s>\"']+)\1", re.IGNORECASE)
_XML_NAME_ATTR_STRIP_RE = re.compile(r"\bname\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)", re.IGNORECASE)
_XML_TOOL_CALL_BLOCK_RE = re.compile(
    r"<(?:tool_call|function_call)>(.*?)</(?:tool_call|function_call)>",
    re.DOTALL | re.IGNORECASE,
)
_XML_CHILD_NAME_RE = re.compile(r"<name\b[^>]*>(.*?)</name\s*>", re.DOTALL | re.IGNORECASE)


def _replace_dsml_tag(match: re.Match) -> str:
    tag = match.group(0)
    extracted = _extract_json_object(tag)
    if extracted is not None:
        obj, start, end = extracted
        if _extract_calls(obj) is not None:
            return tag[start : end + 1]
    return " "


def _strip_dsml(text: str) -> str:
    if not text:
        return text
    result = text
    if "dsml" not in result.casefold():
        return result
    for _ in range(10):
        updated = _DSML_BLOCK.sub(" ", result)
        updated = _DSML_WRAP.sub(" ", updated)
        for pattern in _DSML_HIDDEN_PATS:
            updated = pattern.sub(" ", updated)
        for pattern in _DSML_HIDDEN_NAKED_PATS:
            updated = pattern.sub(" ", updated)
        if updated == result:
            break
        result = updated
        if "dsml" not in result.casefold():
            break
    result = _DSML_XML_NORMALIZE.sub(r"<\1\2>", result)
    result = _DSML_TAG.sub(_replace_dsml_tag, result)
    return _DSML_NAKED.sub(" ", result)


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

_TOOL_STREAM_TAG_RE = re.compile(
    r"<\s*/?\s*(?:" + "|".join(TOOL_STREAM_TAGS) + r")\b[^<>]*>",
    re.IGNORECASE,
)
_TOOL_STREAM_JSON_RE = re.compile(r"\{\s*['\"]?(?:" + "|".join(TOOL_STREAM_JSON_KEYS) + r")['\"]?\s*:")
_TOOL_STREAM_ARRAY_RE = re.compile(r"\[\s*\{")
_TOOL_STREAM_YAML_RE = re.compile(r"(?m)^[ \t]*tool_calls\s*:")
_TOOL_STREAM_NAME_ATTR_RE = re.compile(
    r"<\s*/?\s*(?!(?:" + "|".join(sorted(_XML_HTML_TAGS)) + r")\b)[A-Za-z_][A-Za-z0-9_.-]*[^<>]*\bname\s*=",
    re.IGNORECASE,
)
_DSML_STREAM_START = re.compile(
    r"<\s*/?\s*(?:[|]|[^\x00-\x7f]){1,8}\s*DSML\s*(?:[|]|[^\x00-\x7f]){1,8}",
    re.IGNORECASE | re.DOTALL,
)


@lru_cache(maxsize=64)
def _stream_patterns(names: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    patterns = [
        _TOOL_STREAM_TAG_RE,
        _TOOL_STREAM_JSON_RE,
        _TOOL_STREAM_ARRAY_RE,
        _TOOL_STREAM_YAML_RE,
        _TOOL_STREAM_NAME_ATTR_RE,
    ]
    if names:
        escaped = "|".join(re.escape(name) for name in names)
        patterns.append(re.compile(rf"<\s*/?\s*(?:{escaped})\b", re.IGNORECASE))
        patterns.append(re.compile(rf"(?m)^[ \t]*(?:{escaped})[ \t]*\(", re.IGNORECASE))
    return tuple(patterns)


def _stream_names(tool_schemas: dict[str, dict[str, Any]] | None) -> tuple[str, ...]:
    if not tool_schemas:
        return ()
    return _stream_names_keys(tuple(tool_schemas))


@lru_cache(maxsize=64)
def _stream_names_keys(keys: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(sorted(name.lower() for name in keys if isinstance(name, str) and name))


def _literal_hold(text: str, start: int) -> int:
    length = len(text)
    max_size = min(TOOL_STREAM_MARKER_MAX - 1, length - start)
    for size in range(max_size, 0, -1):
        if text[length - size :] in _MARKER_PREFIXES:
            return length - size
    return -1


def _json_hold(text: str, start: int) -> int:
    brace = text.rfind("{")
    if brace < start or "}" in text[brace:]:
        return -1
    body = text[brace + 1 :].lstrip()
    if not body:
        return brace
    if body[0] in "'\"":
        body = body[1:]
    key = body.lower()
    if any(candidate.startswith(key) for candidate in TOOL_STREAM_JSON_KEYS):
        return brace
    return -1


def _tag_hold(text: str, start: int, names: tuple[str, ...]) -> int:
    lt = text.rfind("<")
    if lt < start:
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
    if first == "|" or ord(first) > 127:
        return lt
    chars: list[str] = []
    for char in body:
        if char.isascii() and (char.isalnum() or char in "_-."):
            chars.append(char)
        else:
            break
    name = "".join(chars).lower()
    if not name:
        return -1
    if name in _XML_HTML_TAGS:
        return -1
    for candidate in TOOL_STREAM_TAGS:
        if candidate.startswith(name):
            return lt
    for candidate in names:
        if candidate.startswith(name):
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
    bracket = text.rfind("[")
    if bracket < start or "]" in text[bracket:]:
        return -1
    if not text[bracket + 1 :].strip():
        return bracket
    return -1


_boundary_cache: dict[tuple[str, ...], tuple[str, int, bool]] = {}
_BOUNDARY_CACHE_MAX = 256


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


def tool_call_boundary(
    text: str,
    start: int = 0,
    tool_schemas: dict[str, dict[str, Any]] | None = None,
) -> tuple[int, bool]:
    names = _stream_names(tool_schemas)
    cached = _boundary_cache.get(names)
    if cached is not None:
        old_text, old_best, old_complete = cached
        if old_complete and old_best != -1 and start <= old_best and text.startswith(old_text):
            hold = _boundary_hold(text, start, names)
            if hold != -1 and hold < old_best:
                return hold, False
            return old_best, True
    best = -1
    complete = False
    for pattern in _stream_patterns(names):
        match = pattern.search(text, start)
        if match is not None and (best == -1 or match.start() < best):
            best = match.start()
            complete = True
    if "dsml" in text.casefold():
        match = _DSML_STREAM_START.search(text, start)
        if match is not None and (best == -1 or match.start() < best):
            best = match.start()
            complete = True
    for marker in TOOL_STREAM_MARKERS:
        pos = text.find(marker, start)
        if pos != -1 and (best == -1 or pos < best):
            best = pos
            complete = True
    hold = _boundary_hold(text, start, names)
    if hold != -1 and (best == -1 or hold < best):
        return hold, False
    if best != -1 and (not _boundary_cache or len(_boundary_cache) < _BOUNDARY_CACHE_MAX):
        _boundary_cache[names] = (text, best, complete)
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


TOOL_CALL_INSTRUCTION = (
    "{functions}\n\n"
    "To call a function, reply with ONLY:\n"
    "<tool_calls>\n"
    '<invoke name="FN">\n'
    '<parameter name="ARG">value</parameter>\n'
    "</invoke>\n"
    "</tool_calls>\n"
    "Use the exact function names and argument keys from the list above.\n"
    'The numbers (1, 2, ...) only help you scan the list; always write the real function name in <invoke name="...">.\n'
    "Never invent a function name or an argument key that is not in the list.\n"
    "Use only argument values that are real and present in the conversation; never guess or fabricate a value.\n"
    "If no listed function fits or a required value is unknown, reply with normal text instead of calling a function.\n"
    "Put independent calls in separate sibling <invoke> elements.\n"
    "Argument values must be JSON-compatible: numbers without quotes, true or false without quotes, objects and arrays as JSON.\n"
    "If you already tried to call a function but received no tool result, do not repeat the same broken output. "
    "Look at the format above and re-emit the tool call exactly in that format.\n"
    "If your previous reply was empty or cut off, re-emit the full tool call in the format above.\n"
    "No text before or after the <tool_calls> block.\n"
    "{choice}"
)

TOOL_TAIL_REMINDER = (
    "Continue the conversation and provide the final answer based on the tool results.\n"
    "If another function call is needed, reply with only the <tool_calls> XML block in the defined format.\n"
    "The format is exactly:\n"
    "<tool_calls>\n"
    '<invoke name="FN">\n'
    '<parameter name="ARG">value</parameter>\n'
    "</invoke>\n"
    "</tool_calls>\n"
    "If a previous attempt to call a function produced no result, look at the format and re-emit the call in it. Do not invent a different format.\n"
    "If not, reply with your final answer."
)

CHOICE_INSTRUCTIONS = {
    "required": "You MUST call one or more functions from the list above. Call no function that is not in the list.",
    "function": "You MUST call a function from the list above. Call no function that is not in the list.",
}

JSON_MODE_INSTRUCTION = (
    "You must reply with ONLY a valid JSON object.{constraints}\nDo not wrap the JSON in markdown fences. Do not add any text before or after the JSON object."
)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    @classmethod
    def create(cls, name: str, arguments: Any) -> ToolCall:
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        if name == "edit" and isinstance(arguments, dict):
            arguments = dict(arguments)
            if "oldString" in arguments and not isinstance(arguments["oldString"], str):
                arguments["oldString"] = json.dumps(arguments["oldString"], ensure_ascii=False)
            if "newString" in arguments and not isinstance(arguments["newString"], str):
                arguments["newString"] = json.dumps(arguments["newString"], ensure_ascii=False)
        if isinstance(arguments, dict):
            args_text = json.dumps(arguments, ensure_ascii=False)
        elif isinstance(arguments, str):
            args_text = arguments
        elif arguments is None:
            args_text = "{}"
        else:
            args_text = json.dumps(arguments, ensure_ascii=False)
        return cls(call_id, name, args_text)


def _tool_function(tool: Any) -> dict | None:
    if not isinstance(tool, dict):
        return None
    if "function" in tool:
        fn = tool["function"]
        return fn if isinstance(fn, dict) else None
    if isinstance(tool.get("name"), str):
        return tool
    return None


def _choice_name(tool_choice: Any) -> str | None:
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type in ("none", "required"):
            return choice_type
        fn = tool_choice.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            return fn["name"]
        if isinstance(choice_type, str) and choice_type not in ("auto", "function"):
            return choice_type
    return None


def _argument_summary(fn: dict) -> str | None:
    params = fn.get("parameters")
    if not isinstance(params, dict):
        return None
    properties = params.get("properties")
    if not isinstance(properties, dict) or not properties:
        return None
    required = params.get("required")
    required_names = set(required) if isinstance(required, list) else set()
    parts: list[str] = []
    for key, prop in properties.items():
        if not isinstance(key, str) or not key:
            continue
        prop_type = prop.get("type") if isinstance(prop, dict) else (prop if isinstance(prop, str) else None)
        if isinstance(prop_type, str) and prop_type:
            parts.append(f"{key} ({prop_type}{', required' if key in required_names else ', optional'})")
        elif key in required_names:
            parts.append(f"{key} (required)")
        else:
            parts.append(key)
    return ", ".join(parts) if parts else None


def render_tool_schema(tools: list[Any] | None, tool_choice: Any = None) -> str | None:
    if not tools:
        return None
    functions: list[dict] = []
    for tool in tools:
        fn = _tool_function(tool)
        if fn is not None and isinstance(fn.get("name"), str) and fn["name"]:
            functions.append(fn)
    if not functions:
        return None
    choice = _choice_name(tool_choice)
    if choice == "none":
        return None
    lines = []
    for i, fn in enumerate(functions, start=1):
        lines.append(f"{i}. name: {fn['name']}")
        if fn.get("description"):
            lines.append(f"   description: {fn['description']}")
        params = fn.get("parameters")
        if params is not None:
            if isinstance(params, str):
                params_json = params
            else:
                params_json = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
            lines.append(f"   parameters: {params_json}")
            argument_summary = _argument_summary(fn)
            if argument_summary:
                lines.append(f"   arguments: {argument_summary}")
    if choice in CHOICE_INSTRUCTIONS:
        choice_line = CHOICE_INSTRUCTIONS[choice]
    elif isinstance(choice, str) and choice not in ("auto", "none", "required"):
        choice_line = f"You MUST call exactly the function {choice} and no other functions."
    else:
        choice_line = "If you do not need to call any function, reply normally with your answer and do not invent a tool call."
    return TOOL_CALL_INSTRUCTION.format(
        functions="\n".join(lines),
        choice=choice_line,
    )


def _content_text(content: Any, *, with_images: bool = False, separator: str = "") -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "image_url" and with_images:
                    image_url = item.get("image_url")
                    if isinstance(image_url, str):
                        parts.append(image_url)
                    elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                        parts.append(image_url["url"])
        result = separator.join(parts)
        return result if separator else result.strip()
    return ""


def _msg_field(msg: Any, key: str, default: Any = None) -> Any:
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def context_sequence(messages: list[Any], user: str | None = None) -> tuple[str, ...]:
    sequence: list[str] = []
    scope = f"\0{user or ''}"
    for msg in messages:
        role = _msg_field(msg, "role", "user")
        if role not in ("system", "user"):
            continue
        content = _content_text(_msg_field(msg, "content", ""), with_images=True, separator="\n")
        if not content.strip():
            continue
        digest = hashlib.sha256(f"{role}\0{content}{scope}".encode()).hexdigest()
        sequence.append(digest)
    return tuple(sequence)


def _render_tool_call_mention(call: Any) -> str:
    if not isinstance(call, dict):
        return ""
    fn = call.get("function")
    if isinstance(fn, dict):
        name = fn.get("name") or ""
        args = fn.get("arguments") or ""
    else:
        name = call.get("name") or ""
        args = call.get("arguments") or ""
    if isinstance(args, (dict, list)):
        args = json.dumps(args, ensure_ascii=False)
    return f"[assistant called {name}({args})]"


def render_message(msg: Any) -> str:
    role = _msg_field(msg, "role", "user")
    text = _strip_dsml(_content_text(_msg_field(msg, "content", "")))
    if role in ("user", "system"):
        return text
    if role == "assistant":
        parts = []
        if text:
            parts.append(text)
        for call in _msg_field(msg, "tool_calls", None) or []:
            mention = _render_tool_call_mention(call)
            if mention:
                parts.append(mention)
        content = _msg_field(msg, "content", None)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_call":
                    parts.append(_render_tool_call_mention(item))
        return "; ".join(parts)
    if role == "tool":
        tool_call_id = _msg_field(msg, "tool_call_id", None) or ""
        prefix = f"Tool result ({tool_call_id})" if tool_call_id else "Tool result"
        return f"{prefix}: {text}"
    if role == "function":
        name = _msg_field(msg, "name", None) or ""
        return f"Function {name} returned: {text}"
    return text


def _render_history(messages: list[Any]) -> str:
    parts = []
    for msg in messages:
        text = render_message(msg)
        if text:
            role = _msg_field(msg, "role", "user")
            parts.append(f"{role.capitalize()}: {text}")
    return "\n".join(parts)


def _render_tool_tail(messages: list[Any]) -> str:
    parts = []
    for msg in messages:
        role = _msg_field(msg, "role", None)
        if role in ("tool", "function"):
            parts.append(render_message(msg))
    parts.append(TOOL_TAIL_REMINDER)
    return "\n".join(parts)


def extract_last_user(messages: list[Any]) -> str:
    if not messages:
        raise ValueError("messages is required")
    for msg in reversed(messages):
        if _msg_field(msg, "role", None) != "user":
            continue
        content = _msg_field(msg, "content", None)
        if content is None:
            continue
        if isinstance(content, str):
            return _strip_dsml(content)
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif item.get("type") == "image_url":
                        continue
            text = _strip_dsml("".join(parts)).strip()
            if text:
                return text
            continue
        raise ValueError("unsupported message content")
    raise ValueError("no user message found")


def is_tool_round(messages: list[Any]) -> bool:
    for msg in messages:
        role = _msg_field(msg, "role", None)
        if role in ("tool", "function"):
            return True
        if role == "assistant" and _msg_field(msg, "tool_calls", None):
            return True
        content = _msg_field(msg, "content", None)
        if role == "assistant" and isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_call":
                    return True
    return False


def _has_history(messages: list[Any]) -> bool:
    user_count = 0
    for msg in messages:
        role = _msg_field(msg, "role", None)
        if role == "user":
            user_count += 1
            continue
        text = _content_text(_msg_field(msg, "content", ""))
        if role == "assistant" and (text or _msg_field(msg, "tool_calls", None)):
            return True
        if role in ("tool", "function") and text:
            return True
    return user_count > 1


def _tail_after_last_user(messages: list[Any]) -> list[Any]:
    index = -1
    for i, msg in enumerate(messages):
        if _msg_field(msg, "role", None) in ("user", "system"):
            index = i
    if index < 0:
        return list(messages)
    return list(messages[index + 1 :])


def extract_system(messages: list[Any]) -> str:
    parts = []
    for msg in messages:
        if _msg_field(msg, "role", None) == "system":
            text = _strip_dsml(_content_text(_msg_field(msg, "content", ""))).strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def render_json_mode(response_format: Any) -> str | None:
    if response_format is None:
        return None
    constraints = ""
    schema: Any = None
    if isinstance(response_format, str):
        if response_format != "json_object":
            return None
    elif isinstance(response_format, dict):
        rtype = response_format.get("type")
        if rtype == "json_schema":
            raw = response_format.get("json_schema")
            schema = raw.get("schema") if isinstance(raw, dict) else None
        elif rtype != "json_object":
            return None
    else:
        return None
    if schema is not None:
        constraints = f"\nThe JSON object must match this JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}"
    return JSON_MODE_INSTRUCTION.format(constraints=constraints)


def build_prompt(
    messages: list[Any],
    tools: list[Any] | None = None,
    tool_choice: Any = None,
    has_session: bool = False,
    response_format: Any = None,
) -> tuple[str, bool]:
    schema = render_tool_schema(tools, tool_choice)
    tools_present = schema is not None
    json_block = render_json_mode(response_format)

    if has_session:
        tail = _tail_after_last_user(messages)
        if is_tool_round(tail):
            return _render_tool_tail(tail), True
        base = extract_last_user(messages)
        blocks = []
        if json_block:
            blocks.append(json_block)
        choice = _choice_name(tool_choice)
        if schema and choice is not None and choice not in ("auto", "none"):
            blocks.append(schema)
        blocks.append(base)
        return "\n\n".join(blocks), tools_present

    tool_round_active = is_tool_round(messages)
    if tool_round_active or _has_history(messages):
        prompt = _render_history(messages)
        if schema:
            prompt = f"{schema}\n\n{prompt}"
        if json_block:
            prompt = f"{json_block}\n\n{prompt}"
        if not prompt.strip():
            prompt = schema or extract_last_user(messages)
        return prompt, tools_present or tool_round_active

    base = extract_last_user(messages)
    blocks = []
    system = extract_system(messages)
    if system:
        blocks.append(system)
    if schema:
        blocks.append(schema)
    if json_block:
        blocks.append(json_block)
    blocks.append(base)
    return "\n\n".join(blocks), tools_present


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    match = _FENCES_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _strip_trailing_commas(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    quote = ""
    escaped = False
    while i < n:
        ch = text[i]
        if quote:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _normalize_single_quotes(text: str) -> str:
    out: list[str] = []
    in_double = False
    in_single = False
    escaped = False
    for ch in text:
        if in_double:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_double = False
            continue
        if in_single:
            if escaped:
                if ch == "'":
                    out.append("'")
                elif ch == '"':
                    out.append('\\"')
                elif ch == "\\":
                    out.append("\\\\")
                else:
                    out.append("\\" + ch)
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "'":
                out.append('"')
                in_single = False
            elif ch == '"':
                out.append('\\"')
            else:
                out.append(ch)
            continue
        if ch == '"':
            in_double = True
            out.append(ch)
        elif ch == "'":
            in_single = True
            out.append('"')
        else:
            out.append(ch)
    return "".join(out)


_NUMBER_RE = re.compile(r"-?\d+(\.\d+)?([eE][+-]?\d+)?")
_BARE_LITERALS = frozenset({"true", "false", "null"})


def _is_bare_literal(token: str) -> bool:
    if token.lower() in _BARE_LITERALS:
        return True
    return _NUMBER_RE.fullmatch(token) is not None


_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


def _url_like_after(text: str, i: int) -> bool:
    fragment = text[i : i + 32]
    if not fragment:
        return False
    if fragment.startswith("://"):
        return True
    if _URL_SCHEME_RE.match(fragment):
        return True
    if fragment[0] == ":":
        rest = fragment[1:]
        for k, ch in enumerate(rest):
            if k >= 16:
                break
            if ch in "{}[],\"'\\ \t\r\n":
                if ch in "[],":
                    return False
                break
        else:
            return True
        return False
    return False


def _normalize_bare_json(text: str) -> str | None:
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    escaped = False
    prev: str = ""
    changed = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            prev = '"'
            out.append(ch)
            i += 1
            continue
        if ch.isspace():
            out.append(ch)
            i += 1
            continue
        if ch in "{[":
            prev = ch
            out.append(ch)
            i += 1
            continue
        if ch in "}]":
            prev = ch
            out.append(ch)
            i += 1
            continue
        if ch in ":,":
            prev = ch
            out.append(ch)
            i += 1
            continue
        start = i
        while i < n and (text[i].isalnum() or text[i] in "_-."):
            i += 1
        token = text[start:i]
        if not token:
            prev = ch
            out.append(ch)
            i += 1
            continue
        j = i
        while j < n and text[j].isspace():
            j += 1
        nxt = text[j] if j < n else ""
        if prev in ("{", "[", ",") and nxt == ":":
            out.append('"')
            out.append(token)
            out.append('"')
            prev = '"'
            i = j
            changed = True
            continue
        if prev == ":" and not _is_bare_literal(token) and not _url_like_after(text, i):
            out.append('"')
            out.append(token)
            out.append('"')
            prev = '"'
            changed = True
            continue
        if prev in ("[", ",") and nxt in (",", "]") and not _is_bare_literal(token):
            out.append('"')
            out.append(token)
            out.append('"')
            prev = '"'
            changed = True
            continue
        prev = token
        out.append(token)
    if not changed:
        return None
    return "".join(out)


def _json_candidates(text: str) -> Iterator[str]:
    yield text
    no_trailing = _strip_trailing_commas(text)
    if no_trailing != text:
        yield no_trailing
    normalized = _normalize_single_quotes(no_trailing)
    if normalized != no_trailing:
        yield normalized
    bare = _normalize_bare_json(normalized)
    if bare is not None and bare != normalized:
        yield bare
    fixed = _fix_unbalanced_json(normalized)
    if fixed is not None and fixed != normalized:
        yield fixed


def _loads_lenient(text: str) -> Any:
    text = text.strip()
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    raise ValueError("invalid json")


def _fix_unbalanced_json(text: str) -> str | None:
    stack: list[str] = []
    insert_before: dict[int, int] = {}
    drop: set[int] = set()
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
            else:
                drop.add(i)
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
            elif stack:
                count = 0
                while stack and stack[-1] != "[":
                    count += 1
                    stack.pop()
                insert_before[i] = count
                if stack:
                    stack.pop()
                else:
                    drop.add(i)
            else:
                drop.add(i)

    if not insert_before and not drop and not stack and not in_string:
        return None

    result: list[str] = []
    for i, ch in enumerate(text):
        if i in insert_before:
            result.append("}" * insert_before[i])
        if i in drop:
            continue
        result.append(ch)
    if in_string:
        if escaped:
            result.append("\\")
        result.append('"')
    result.append("".join("}" if opening == "{" else "]" for opening in reversed(stack)))
    return "".join(result)


def _balanced_json(text: str) -> tuple[int, int] | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, i
    return None


def _extract_json_object(text: str) -> tuple[dict, int, int] | None:
    if not isinstance(text, str):
        return None
    stripped = _strip_fences(text)
    bounds = _balanced_json(stripped)
    if bounds is None:
        return None
    start, end = bounds
    try:
        obj = _loads_lenient(stripped[start : end + 1])
        if not isinstance(obj, dict):
            return None
    except ValueError:
        return None
    return obj, start, end


def _call_item_fields(item: dict) -> tuple[Any, Any]:
    fn = item.get("function")
    if isinstance(fn, dict):
        return fn.get("name"), fn.get("arguments")
    name = item.get("name")
    if not isinstance(name, str) or not name:
        for key in _NAME_ALIASES[1:]:
            candidate = item.get(key)
            if isinstance(candidate, str) and candidate:
                name = candidate
                break
    arguments = item.get("arguments")
    if arguments is None:
        for key in _ARGS_ALIASES[1:]:
            candidate = item.get(key)
            if candidate is not None:
                arguments = candidate
                break
    if arguments is None:
        arguments = {}
    return name, arguments


def _extract_one_call(item: Any) -> ToolCall | None:
    if not isinstance(item, dict):
        return None
    name, arguments = _call_item_fields(item)
    if not isinstance(name, str) or not name:
        return None
    return ToolCall.create(name, arguments)


def _is_jsonish_arguments(value: Any) -> bool:
    if isinstance(value, dict):
        return True
    if isinstance(value, str):
        return value.strip().startswith("{")
    return False


def _extract_wrapped_calls(obj: dict) -> list[ToolCall] | None:
    calls: list[ToolCall] = []
    raw = obj.get("tool_calls")
    if raw is None:
        raw = obj.get("calls")
    if raw is None:
        raw = obj.get("_calls")
    if isinstance(raw, list):
        for item in raw:
            call = _extract_one_call(item)
            if call is not None:
                calls.append(call)
    elif isinstance(raw, dict):
        call = _extract_one_call(raw)
        if call is not None:
            calls.append(call)
    else:
        legacy = obj.get("function_call")
        if isinstance(legacy, dict):
            call = _extract_one_call(legacy)
            if call is not None:
                calls.append(call)
    return calls or None


def _extract_calls(obj: dict) -> list[ToolCall] | None:
    if not isinstance(obj, dict):
        return None
    calls = _extract_wrapped_calls(obj)
    if calls is not None:
        return calls
    name, arguments = _call_item_fields(obj)
    if isinstance(name, str) and name and _is_jsonish_arguments(arguments):
        return [ToolCall.create(name, arguments)]
    return None


_XML_ENTITY_RE = re.compile(r"&(amp|lt|gt|quot|apos);")
_XML_ENTITY_MAP = {"lt": "<", "gt": ">", "quot": '"', "apos": "'", "amp": "&"}


def _unescape_xml(text: str) -> str:
    return _XML_ENTITY_RE.sub(lambda m: _XML_ENTITY_MAP.get(m.group(1), m.group(0)), text)


def _coerce_scalar(value: str, json_type: Any) -> Any:
    if not isinstance(value, str):
        return value
    if json_type in ("integer", "number"):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value
    if json_type == "boolean":
        low = value.strip().lower()
        if low == "true":
            return True
        if low == "false":
            return False
        return value
    if json_type == "null":
        return None
    return value


@lru_cache(maxsize=8192)
def _casefold(text: str) -> str:
    return text.casefold()


@lru_cache(maxsize=8192)
def _name_key(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", _casefold(name))


@lru_cache(maxsize=256)
def _folded_keys(keys: tuple[Any, ...]) -> dict[str, str]:
    folded: dict[str, str] = {}
    for key in keys:
        if isinstance(key, str):
            folded.setdefault(_casefold(key), key)
    return folded


@lru_cache(maxsize=256)
def _folded_names(keys: tuple[Any, ...]) -> dict[str, str]:
    return {_casefold(key): key for key in keys}


@lru_cache(maxsize=256)
def _compact_names(keys: tuple[Any, ...]) -> dict[str, str]:
    return {_name_key(key): key for key in keys}


def _schema_for_name(tool_schemas: dict[str, dict[str, Any]] | None, name: str) -> dict[str, Any] | None:
    if not tool_schemas or not name:
        return None
    if not isinstance(tool_schemas, dict):
        return None
    if name in tool_schemas:
        spec = tool_schemas[name]
        return spec if isinstance(spec, dict) else None
    key = _folded_keys(tuple(tool_schemas)).get(_casefold(name))
    if key is not None:
        spec = tool_schemas[key]
        return spec if isinstance(spec, dict) else None
    resolved = _resolve_alias(name, tool_schemas)
    if resolved is not None and resolved in tool_schemas:
        spec = tool_schemas[resolved]
        return spec if isinstance(spec, dict) else None
    return None


@lru_cache(maxsize=256)
def _alias_rows(seed: tuple[tuple[str, tuple[Any, ...]], ...]) -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = []
    for known, aliases in seed:
        for alias in aliases:
            if not isinstance(alias, str) or not alias:
                continue
            rows.append((_name_key(alias), _casefold(alias), known))
    return tuple(rows)


@lru_cache(maxsize=256)
def _schema_name_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_name_key(key) for key in keys)


def _resolve_alias(name: str, tool_schemas: dict[str, dict[str, Any]] | None) -> str | None:
    if not tool_schemas:
        return None
    folded = _casefold(name)
    key = _name_key(name)
    if not key:
        return None
    seed = tuple((known, tuple((spec or {}).get("_aliases") or ())) for known, spec in (tool_schemas or {}).items())
    for alias_key, alias_folded, known in _alias_rows(seed):
        if alias_key == key or alias_folded == folded:
            return known
    return None


def _fuzzy_known_name(name: str, tool_schemas: dict[str, dict[str, Any]]) -> str | None:
    key = _name_key(name)
    if len(key) < 4:
        return None
    keys = _schema_name_keys(tuple(tool_schemas))
    matches = get_close_matches(key, keys, n=2, cutoff=0.8)
    if len(matches) != 1:
        return None
    hit = matches[0]
    for known in tool_schemas:
        if _name_key(known) == hit:
            return known
    return None


def _normalize_call_name(name: str, tool_schemas: dict[str, dict[str, Any]] | None) -> str:
    if not tool_schemas or name in tool_schemas:
        return name
    keys = tuple(tool_schemas)
    hit = _folded_names(keys).get(_casefold(name))
    if hit is not None:
        return hit
    hit = _compact_names(keys).get(_name_key(name))
    if hit is not None:
        return hit
    return _resolve_alias(name, tool_schemas) or _fuzzy_known_name(name, tool_schemas) or name


_tool_schema_map_cache: dict[int, tuple[tuple[int, ...], dict[str, dict[str, Any]]]] = {}
_TOOL_SCHEMA_MAP_CACHE_MAX = 256


def tool_schema_map(tools: list[Any] | None) -> dict[str, dict[str, Any]]:
    if not tools or not isinstance(tools, list):
        return {}
    key = id(tools)
    fingerprint = tuple(id(item) for item in tools)
    cached = _tool_schema_map_cache.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    result: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = _tool_function(tool)
        if not fn or not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        params = fn.get("parameters")
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (ValueError, TypeError, AttributeError):
                params = None
        prop_types: dict[str, Any] = {}
        properties = params.get("properties") if isinstance(params, dict) else None
        if isinstance(properties, dict):
            for prop, spec in properties.items():
                if not isinstance(prop, str) or not isinstance(spec, dict):
                    continue
                typ = spec.get("type")
                if isinstance(typ, list):
                    for candidate in ("integer", "number", "boolean", "null", "string"):
                        if candidate in typ:
                            typ = candidate
                            break
                if typ:
                    prop_types[prop] = typ
        aliases = fn.get("aliases")
        if isinstance(aliases, (list, tuple)):
            cleaned = [a for a in aliases if isinstance(a, str) and a.strip()]
            if cleaned:
                prop_types["_aliases"] = cleaned
        result[name] = prop_types
    if len(_tool_schema_map_cache) >= _TOOL_SCHEMA_MAP_CACHE_MAX:
        _tool_schema_map_cache.clear()
    _tool_schema_map_cache[key] = (fingerprint, result)
    return result


_FIX_MODES = frozenset({"report", "safe", "full"})
_FIX_FUZZY_PARAM_MIN = 4
_FIX_FUZZY_PARAM_CUTOFF = 0.85
_FIX_FUZZY_ENUM_MIN = 2
_FIX_FUZZY_ENUM_CUTOFF = 0.85


def tool_schema_detail(tools: list[Any] | None) -> dict[str, dict[str, Any]]:
    if not tools or not isinstance(tools, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = _tool_function(tool)
        if not fn or not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        params = fn.get("parameters")
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (ValueError, TypeError, AttributeError):
                params = None
        types: dict[str, Any] = {}
        enums: dict[str, list[Any]] = {}
        defaults: dict[str, Any] = {}
        bounds: dict[str, dict[str, Any]] = {}
        param_aliases: dict[str, list[str]] = {}
        required: list[str] = []
        if isinstance(params, dict):
            req = params.get("required")
            if isinstance(req, list):
                required = [str(item) for item in req if isinstance(item, str) and item]
            properties = params.get("properties")
            if isinstance(properties, dict):
                for prop, spec in properties.items():
                    if not isinstance(prop, str) or not isinstance(spec, dict):
                        continue
                    typ = spec.get("type")
                    if isinstance(typ, list):
                        for candidate in ("integer", "number", "boolean", "null", "string"):
                            if candidate in typ:
                                typ = candidate
                                break
                    if typ:
                        types[prop] = typ
                    enum_values = spec.get("enum")
                    if isinstance(enum_values, list) and enum_values:
                        enums[prop] = list(enum_values)
                    if "default" in spec:
                        defaults[prop] = spec["default"]
                    entry: dict[str, Any] = {}
                    low = spec.get("minimum")
                    high = spec.get("maximum")
                    if isinstance(low, (int, float)) and not isinstance(low, bool):
                        entry["minimum"] = low
                    if isinstance(high, (int, float)) and not isinstance(high, bool):
                        entry["maximum"] = high
                    if entry:
                        bounds[prop] = entry
                    aliases = spec.get("aliases")
                    if isinstance(aliases, (list, tuple)):
                        cleaned = [a for a in aliases if isinstance(a, str) and a.strip()]
                        if cleaned:
                            param_aliases[prop] = cleaned
        name_aliases: list[str] = []
        raw_aliases = fn.get("aliases")
        if isinstance(raw_aliases, (list, tuple)):
            name_aliases = [a for a in raw_aliases if isinstance(a, str) and a.strip()]
        result[name] = {
            "types": types,
            "required": required,
            "enums": enums,
            "defaults": defaults,
            "bounds": bounds,
            "param_aliases": param_aliases,
            "name_aliases": name_aliases,
        }
    return result


def _resolve_arg_key(
    key: str,
    known_props: tuple[str, ...],
    param_aliases: dict[str, Any],
) -> tuple[str | None, str]:
    if key in known_props:
        return key, "exact"
    if not known_props:
        return None, "unknown"
    folded_map = _folded_names(known_props)
    hit = folded_map.get(_casefold(key))
    if hit is not None:
        return hit, "casefold"
    compact_map = _compact_names(known_props)
    hit = compact_map.get(_name_key(key))
    if hit is not None:
        return hit, "compact"
    folded = _casefold(key)
    compact = _name_key(key)
    for prop, aliases in param_aliases.items():
        for alias in aliases:
            if _casefold(alias) == folded or _name_key(alias) == compact:
                return prop, "alias"
    if len(compact) >= _FIX_FUZZY_PARAM_MIN:
        matches = get_close_matches(compact, tuple(compact_map), n=2, cutoff=_FIX_FUZZY_PARAM_CUTOFF)
        if len(matches) == 1:
            return compact_map[matches[0]], "fuzzy"
    return None, "unknown"


def _match_enum(value: Any, enum_values: list[Any]) -> tuple[Any, str] | None:
    try:
        if value in enum_values:
            return value, "exact"
    except TypeError:
        pass
    if isinstance(value, str):
        folded = _casefold(value)
        for item in enum_values:
            if isinstance(item, str) and _casefold(item) == folded:
                return item, "casefold"
        str_values = [item for item in enum_values if isinstance(item, str)]
        if str_values and len(value) >= _FIX_FUZZY_ENUM_MIN:
            matches = get_close_matches(value, str_values, n=2, cutoff=_FIX_FUZZY_ENUM_CUTOFF)
            if len(matches) == 1:
                return matches[0], "fuzzy"
    return None


def _coerce_by_type(value: Any, json_type: Any) -> tuple[Any, bool]:
    if json_type is None:
        return value, False
    if json_type == "string":
        if isinstance(value, str):
            return value, False
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False), True
        if value is None:
            return "", True
        return json.dumps(value, ensure_ascii=False), True
    if json_type in ("integer", "number"):
        if isinstance(value, bool):
            return value, False
        if isinstance(value, (int, float)):
            if json_type == "integer" and isinstance(value, float) and value.is_integer():
                return int(value), True
            return value, False
        if isinstance(value, str):
            coerced = _coerce_scalar(value, json_type)
            if isinstance(coerced, (int, float)) and not isinstance(coerced, bool):
                return coerced, True
        return value, False
    if json_type == "boolean":
        if isinstance(value, bool):
            return value, False
        if isinstance(value, str):
            low = value.strip().lower()
            if low == "true":
                return True, True
            if low == "false":
                return False, True
        return value, False
    if json_type == "null":
        if value is None:
            return value, False
        if isinstance(value, str) and value.strip().lower() in ("null", "none", "~"):
            return None, True
        return value, False
    return value, False


def fix_tool_calls(
    calls: list[ToolCall],
    tool_schemas: dict[str, dict[str, Any]] | None,
    tool_details: dict[str, dict[str, Any]] | None = None,
    mode: str = "report",
    report: dict[str, Any] | None = None,
) -> list[ToolCall]:
    if not calls:
        return calls
    if mode not in _FIX_MODES:
        mode = "report"
    schemas = tool_schemas if isinstance(tool_schemas, dict) else {}
    details = tool_details if isinstance(tool_details, dict) else {}
    fixes: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    result: list[ToolCall] = []
    apply_fixes = mode in ("safe", "full")
    drop_unknown = mode == "full"
    for call in calls:
        spec = details.get(call.name)
        if not isinstance(spec, dict):
            if schemas and _schema_for_name(schemas, call.name) is None:
                warnings.append({"call_id": call.id, "kind": "unknown_tool", "name": call.name})
                result.append(call)
                continue
            spec = {}
        raw_types = spec.get("types")
        types: dict[str, Any] = raw_types if isinstance(raw_types, dict) else {}
        raw_enums = spec.get("enums")
        enums: dict[str, Any] = raw_enums if isinstance(raw_enums, dict) else {}
        raw_defaults = spec.get("defaults")
        defaults: dict[str, Any] = raw_defaults if isinstance(raw_defaults, dict) else {}
        raw_bounds = spec.get("bounds")
        bounds: dict[str, Any] = raw_bounds if isinstance(raw_bounds, dict) else {}
        raw_param_aliases = spec.get("param_aliases")
        param_aliases: dict[str, Any] = raw_param_aliases if isinstance(raw_param_aliases, dict) else {}
        raw_required = spec.get("required")
        required: list[Any] = raw_required if isinstance(raw_required, list) else []
        try:
            parsed = _loads_lenient(call.arguments) if call.arguments and call.arguments.strip() else {}
        except ValueError:
            result.append(call)
            continue
        if not isinstance(parsed, dict):
            result.append(call)
            continue
        known_props = tuple(types)
        rebuilt: dict[str, Any] = {}
        unknowns: list[str] = []
        for key, value in parsed.items():
            resolved_key, how = _resolve_arg_key(key, known_props, param_aliases)
            if resolved_key is None:
                unknowns.append(key)
                rebuilt[key] = value
                continue
            if resolved_key in rebuilt:
                continue
            if resolved_key != key:
                fixes.append({"call_id": call.id, "kind": "rename", "from": key, "to": resolved_key, "confidence": how})
            rebuilt[resolved_key] = value
        coerced: dict[str, Any] = {}
        for key, value in rebuilt.items():
            new_value, changed = _coerce_by_type(value, types.get(key))
            if changed:
                fixes.append({"call_id": call.id, "kind": "coerce", "param": key, "from": value, "to": new_value})
            enum_values = enums.get(key)
            if isinstance(enum_values, list) and enum_values:
                match = _match_enum(new_value, enum_values)
                if match is not None:
                    if match[0] != new_value:
                        fixes.append({"call_id": call.id, "kind": "enum", "param": key, "from": new_value, "to": match[0]})
                        new_value = match[0]
                else:
                    warnings.append({"call_id": call.id, "kind": "enum_mismatch", "param": key, "value": new_value})
            bound = bounds.get(key)
            if isinstance(bound, dict) and isinstance(new_value, (int, float)) and not isinstance(new_value, bool):
                low = bound.get("minimum")
                high = bound.get("maximum")
                if isinstance(low, (int, float)) and new_value < low:
                    warnings.append({"call_id": call.id, "kind": "out_of_range", "param": key, "value": new_value, "minimum": low})
                if isinstance(high, (int, float)) and new_value > high:
                    warnings.append({"call_id": call.id, "kind": "out_of_range", "param": key, "value": new_value, "maximum": high})
            coerced[key] = new_value
        for key, default_value in defaults.items():
            if key not in coerced:
                coerced[key] = default_value
                fixes.append({"call_id": call.id, "kind": "default", "param": key, "to": default_value})
        for name in required:
            if name not in coerced:
                warnings.append({"call_id": call.id, "kind": "missing_required", "param": name})
        for key in unknowns:
            warnings.append({"call_id": call.id, "kind": "unknown_param", "param": key})
        if apply_fixes:
            if drop_unknown:
                for key in unknowns:
                    coerced.pop(key, None)
            new_arguments = json.dumps(coerced, ensure_ascii=False)
        else:
            new_arguments = call.arguments
        result.append(ToolCall(call.id, call.name, new_arguments))
    if report is not None:
        existing_fixes = report.get("fixes")
        if isinstance(existing_fixes, list):
            existing_fixes.extend(fixes)
        else:
            report["fixes"] = fixes
        existing_warnings = report.get("warnings")
        if isinstance(existing_warnings, list):
            existing_warnings.extend(warnings)
        else:
            report["warnings"] = warnings
    return result


def _xml_set_param(params: dict[str, Any], key: str, value: Any) -> None:
    if key in params:
        existing = params[key]
        if isinstance(existing, list):
            existing.append(value)
        else:
            params[key] = [existing, value]
    else:
        params[key] = value


def _xml_value(raw: str, json_type: Any) -> Any:
    stripped = raw.strip()
    if stripped.startswith(("{", "[")):
        try:
            return _loads_lenient(stripped)
        except ValueError:
            pass
    if json_type == "string":
        return _unescape_xml(stripped)
    if _XML_NESTED_RE.search(stripped):
        nested = _xml_invoke_arguments(stripped, None, False)
        if nested is not None:
            return nested
    return _coerce_scalar(_unescape_xml(stripped), json_type)


def _xml_invoke_arguments(body: str, param_types: dict[str, Any] | None = None, allow_content: bool = True) -> dict[str, Any] | None:
    stripped = body.strip()
    if stripped.startswith("{"):
        try:
            return _loads_lenient(stripped)
        except ValueError:
            pass
    params: dict[str, Any] = {}
    for match in _XML_PARAM_RE.finditer(body):
        key = match.group(2).strip()
        _xml_set_param(params, key, _xml_value(match.group(3), (param_types or {}).get(key)))
    if params:
        return params
    for _, _, key, _, inner in _scan_xml_pairs(body):
        lowered = key.strip().lower()
        if lowered in _XML_SKIP_ELEMENTS:
            continue
        if lowered in _XML_HTML_TAGS and lowered not in _ARGS_ALIASES:
            continue
        _xml_set_param(params, key.strip(), _xml_value(inner, (param_types or {}).get(key.strip())))
    if params:
        if len(params) == 1:
            for key in _ARGS_ALIASES:
                if key in params and isinstance(params[key], dict) and (param_types is None or key not in param_types):
                    return params[key]
        return params
    if not allow_content:
        return None
    if param_types is not None and all(key == "_aliases" for key in param_types):
        return None
    inner = _unescape_xml(stripped)
    if inner:
        return {"content": inner}
    return None


def _xml_tag_attrs(body: str, param_types: dict[str, Any] | None = None) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for match in _XML_ATTR_RE.finditer(body):
        key = match.group(1)
        raw = match.group(2)
        value = raw[1:-1]
        if key.casefold() in _JSON_TYPE_ATTRS and value.casefold() in (
            "true",
            "false",
            "null",
        ):
            continue
        attrs[key] = _coerce_scalar(_unescape_xml(value), (param_types or {}).get(key))
    return attrs


def _iter_xml_call_wrappers(text: str) -> Iterator[tuple[int, int, int, str]]:
    pos = 0
    length = len(text)
    while pos < length:
        match = _XML_WRAPPER_OPEN.search(text, pos)
        if match is None:
            return
        content_start = match.end()
        close = _XML_WRAPPER_CLOSE_RE.search(text, content_start)
        if close is None:
            close = _XML_WRAPPER_OPEN.search(text, content_start)
        end = length if close is None else close.start()
        yield match.start(), content_start, end, text[content_start:end]
        pos = max(match.end(), end)


@lru_cache(maxsize=512)
def _schema_xml_patterns(tool_name: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    escaped = re.escape(tool_name)
    return (
        re.compile(rf"<{escaped}(?=[\s/>])([^>]*?)>(.*?)</{escaped}>", re.DOTALL | re.IGNORECASE),
        re.compile(rf"<{escaped}(?=[\s/>])([^>]*?)/>", re.DOTALL | re.IGNORECASE),
    )


def _parse_xml_tool_calls(text: str, tool_schemas: dict[str, dict[str, Any]] | None = None) -> tuple[list[ToolCall] | None, str]:
    calls: list[ToolCall] = []
    mask = bytearray(len(text))
    consumed = _IntervalSet()

    def blank(start: int, end: int) -> None:
        mask[start:end] = b" " * (end - start)

    for start, end, _tag, attrs_text, element_body in _scan_xml_pairs(text, _TOOL_TAG_NAMES, tail_space=True):
        body = element_body
        name_match = _XML_NAME_ATTR_RE.search(attrs_text)
        tool_name = name_match.group(2) if name_match else None
        if not tool_name:
            child_name = _XML_CHILD_NAME_RE.search(body)
            if child_name is None:
                continue
            tool_name = _unescape_xml(child_name.group(1).strip())
            body = body[: child_name.start()] + " " + body[child_name.end() :]
        if not tool_name:
            continue
        param_types = _schema_for_name(tool_schemas, tool_name)
        arguments = _xml_tag_attrs(_XML_NAME_ATTR_STRIP_RE.sub("", attrs_text), param_types)
        arguments.update(_xml_invoke_arguments(body, param_types) or {})
        calls.append(ToolCall.create(tool_name, arguments))
        blank(start, end)
        consumed.add(start, end)
    for match in _XML_TOOL_SELFCLOSE_RE.finditer(text):
        start, end = match.span()
        if consumed.contains(start, end):
            continue
        attrs_text = match.group(1)
        name_match = _XML_NAME_ATTR_RE.search(attrs_text)
        if name_match is None:
            continue
        tool_name = name_match.group(2)
        param_types = _schema_for_name(tool_schemas, tool_name)
        arguments = _xml_tag_attrs(_XML_NAME_ATTR_STRIP_RE.sub("", attrs_text), param_types)
        calls.append(ToolCall.create(tool_name, arguments))
        blank(start, end)
        consumed.add(start, end)
    for match in _XML_TOOL_CALL_BLOCK_RE.finditer(text):
        parsed = _extract_json_object(match.group(1))
        if parsed is None:
            continue
        obj, _, _ = parsed
        extracted = _extract_calls(obj)
        if extracted:
            calls.extend(extracted)
            start, end = match.span()
            blank(start, end)
            consumed.add(start, end)
    for start, content_start, end, inner in _iter_xml_call_wrappers(text):
        if consumed.contains(start, end):
            continue
        stripped_inner = inner.strip()
        if stripped_inner.startswith("["):
            array_calls = _parse_bare_array_calls(stripped_inner)
            if array_calls:
                calls.extend(array_calls)
                blank(start, end)
                consumed.add(start, end)
                continue
        json_parsed = _extract_json_object(stripped_inner)
        if json_parsed is not None:
            extracted = _extract_calls(json_parsed[0])
            if extracted:
                calls.extend(extracted)
                blank(start, end)
                consumed.add(start, end)
                continue
        pending_name: str | None = None
        block_calls = 0
        for element_start_rel, element_end_rel, element_raw_name, element_attrs, element_body in _scan_xml_pairs(inner):
            raw_name = element_raw_name
            element_name = raw_name.strip().lower()
            if element_name in _XML_SKIP_ELEMENTS:
                continue
            element_start = content_start + element_start_rel
            element_end = content_start + element_end_rel
            if consumed.contains(element_start, element_end):
                continue
            if element_name == "name":
                raw = _unescape_xml(element_body.strip())
                if raw:
                    pending_name = raw
                continue
            if element_name in _ARGS_ALIASES:
                container = _xml_invoke_arguments(element_body, None)
                if isinstance(container, dict) and pending_name:
                    calls.append(ToolCall.create(pending_name, container))
                    consumed.add(element_start, element_end)
                    block_calls += 1
                    pending_name = None
                continue
            param_types = _schema_for_name(tool_schemas, raw_name)
            arguments = _xml_tag_attrs(element_attrs, param_types)
            arguments.update(_xml_invoke_arguments(element_body, param_types) or {})
            if param_types is None and isinstance(arguments.get("name"), str) and arguments["name"].strip():
                raw_name = arguments.pop("name")
                param_types = _schema_for_name(tool_schemas, raw_name)
            elif param_types is None and raw_name.casefold() in _XML_GENERIC_TOOL_TAGS:
                continue
            if not arguments and param_types is None:
                continue
            calls.append(ToolCall.create(raw_name, arguments))
            consumed.add(element_start, element_end)
            block_calls += 1
        for element in _XML_SELFCLOSE.finditer(inner):
            element_name = element.group(1).strip().lower()
            if element_name in _XML_SKIP_ELEMENTS:
                continue
            element_start = content_start + element.start()
            element_end = content_start + element.end()
            if consumed.contains(element_start, element_end):
                continue
            tool_name = element.group(1).strip()
            param_types = _schema_for_name(tool_schemas, tool_name)
            arguments = _xml_tag_attrs(element.group(2), param_types)
            if not arguments and param_types is None:
                continue
            calls.append(ToolCall.create(tool_name, arguments))
            consumed.add(element_start, element_end)
            block_calls += 1
        if not block_calls and not any(_scan_xml_pairs(inner, _TOOL_TAG_NAMES)):
            bare_params: dict[str, Any] = {}
            for param in _XML_PARAM_RE.finditer(inner):
                key = param.group(2).strip()
                _xml_set_param(bare_params, key, _xml_value(param.group(3), None))
            if bare_params:
                inferred = _infer_tool_name_from_schemas(set(bare_params), tool_schemas)
                if inferred is not None:
                    calls.append(ToolCall.create(inferred, bare_params))
                    consumed.add(start, end)
                    block_calls += 1
        if block_calls:
            blank(start, end)
            consumed.add(start, end)
    for tool_name, param_types in (tool_schemas or {}).items():
        open_pattern, selfclose_pattern = _schema_xml_patterns(tool_name)
        for match in open_pattern.finditer(text):
            start, end = match.span()
            if consumed.contains(start, end):
                continue
            merged = _xml_tag_attrs(match.group(1), param_types)
            merged.update(_xml_invoke_arguments(match.group(2), param_types) or {})
            calls.append(ToolCall.create(tool_name, merged))
            consumed.add(start, end)
            blank(start, end)
        for match in selfclose_pattern.finditer(text):
            start, end = match.span()
            if consumed.contains(start, end):
                continue
            arguments = _xml_tag_attrs(match.group(1), param_types)
            calls.append(ToolCall.create(tool_name, arguments))
            consumed.add(start, end)
            blank(start, end)

    def _bare_eligible(name: str) -> bool:
        return name not in _XML_SKIP_ELEMENTS and name not in _XML_HTML_TAGS

    bare_candidates: list[tuple[int, int, bool, str, str, str]] = []
    for start, end, raw_name, attrs, body in _scan_xml_pairs(text):
        if _bare_eligible(raw_name.strip().lower()) and not consumed.contains(start, end):
            bare_candidates.append((start, end, False, raw_name, attrs, body))
    for m in _XML_SELFCLOSE.finditer(text):
        if _bare_eligible(m.group(1).strip().lower()) and not consumed.contains(m.start(), m.end()):
            bare_candidates.append((m.start(), m.end(), True, m.group(1), m.group(2), ""))
    bare_candidates.sort(key=lambda item: (item[0], -item[1]))
    seen = _IntervalSet()
    for start, end, self_closed, bare_raw_name, attrs, body in bare_candidates:
        raw_name = bare_raw_name
        if seen.contains(start, end):
            continue
        if consumed.contains(start, end):
            continue
        seen.add(start, end)
        param_types = _schema_for_name(tool_schemas, raw_name)
        arguments = _xml_tag_attrs(attrs, param_types)
        if not self_closed:
            arguments.update(_xml_invoke_arguments(body, param_types, False) or {})
        if param_types is None and isinstance(arguments.get("name"), str) and arguments["name"].strip():
            raw_name = arguments.pop("name")
            param_types = _schema_for_name(tool_schemas, raw_name)
        elif param_types is None and raw_name.casefold() in _XML_GENERIC_TOOL_TAGS and not arguments:
            continue
        if not arguments and param_types is None:
            continue
        calls.append(ToolCall.create(raw_name, arguments))
        consumed.add(start, end)
        blank(start, end)
    if not calls:
        return None, ""
    remainder = _blanked(text, mask)
    remainder = _XML_OPEN_TAG.sub(" ", remainder)
    remainder = _XML_CLOSE_TAG.sub(" ", remainder)
    wrapper = " ".join(remainder.split())
    return calls, wrapper


def _parse_bare_array_calls(text: str) -> list[ToolCall] | None:
    stripped = _strip_fences(text).strip()
    if not stripped.startswith("["):
        return None
    try:
        items = _loads_lenient(stripped)
    except ValueError:
        return None
    calls: list[ToolCall] = []
    for item in items:
        call = _extract_one_call(item)
        if call is not None:
            calls.append(call)
    return calls or None


_MAX_JSON_SCAN = 200_000
_MAX_JSON_CANDIDATES = 2000
_JSON_KEY_START = frozenset("_-.'" + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def _iter_json_objects(text: str) -> Iterator[tuple[dict, int, int]]:
    i = 0
    length = len(text)
    scanned = 0
    attempts = 0
    while scanned < _MAX_JSON_SCAN and attempts < _MAX_JSON_CANDIDATES:
        start = text.find("{", i)
        if start == -1:
            return
        probe = start + 1
        while probe < length and text[probe] in " \t\r\n":
            probe += 1
        if probe < length and text[probe] != '"' and text[probe] != "}" and text[probe] not in _JSON_KEY_START:
            i = start + 1
            continue
        depth = 0
        in_string = False
        escaped = False
        end = start
        closed = False
        while end < length:
            ch = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    closed = True
                    break
            end += 1
        scanned += end - start + 1
        attempts += 1
        if not closed:
            return
        candidate = text[start : end + 1]
        try:
            obj = _loads_lenient(candidate)
            yield obj, start, end
        except ValueError:
            pass
        i = end + 1


_YAML_KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$")
_YAML_TOOL_CALLS_RE = re.compile(r"^tool_calls\s*:?\s*(.*)$", re.IGNORECASE)


def _yaml_key_value(line: str) -> tuple[str | None, str]:
    match = _YAML_KEY_VALUE_RE.match(line)
    if match is None:
        return None, ""
    return match.group(1), match.group(2).strip()


def _yaml_name(raw: str) -> str | None:
    name = raw.strip()
    if not name:
        return None
    if len(name) > 1 and name[0] in ("'", '"') and name[-1] == name[0]:
        name = name[1:-1]
    return name


def _yaml_value(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return None
    if value.startswith(("{", "[")):
        try:
            return _loads_lenient(value)
        except (ValueError, TypeError, AttributeError):
            return value
    if value[0] in ("'", '"'):
        if len(value) < 2 or value[-1] != value[0]:
            return value
        if value[0] == '"':
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value[1:-1]
        return value[1:-1]
    low = value.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    return value


def _parse_yaml_calls(text: str) -> list[ToolCall] | None:
    lines = [line.rstrip() for line in text.splitlines()]
    if not lines:
        return None
    root = lines[0].strip()
    root_match = _YAML_TOOL_CALLS_RE.match(root)
    if root_match is None:
        return None
    inline = root_match.group(1).strip()
    rest = [line.strip() for line in lines[1:] if line.strip()]
    if inline:
        if inline.startswith("["):
            array_calls = _parse_bare_array_calls(inline)
            if array_calls:
                return array_calls
        return None
    calls: list[ToolCall] = []
    current_name: str | None = None
    current_args: dict[str, Any] = {}
    args_mode = False
    for line in rest:
        if line.startswith("- "):
            if current_name:
                calls.append(ToolCall.create(current_name, current_args))
            current_name = None
            current_args = {}
            args_mode = False
            item_text = line[2:].strip()
            if ":" in item_text:
                key, value = _yaml_key_value(item_text)
                if key == "name":
                    current_name = _yaml_name(value)
                continue
            else:
                current_name = _yaml_name(item_text)
            continue
        if current_name is None:
            key, value = _yaml_key_value(line)
            if key == "name":
                current_name = _yaml_name(value)
            continue
        key, value = _yaml_key_value(line)
        if key is None:
            continue
        if key in _ARGS_ALIASES:
            args_mode = True
            if value:
                parsed = _yaml_value(value)
                if isinstance(parsed, dict):
                    current_args.update(parsed)
            continue
        if args_mode:
            current_args[key] = _yaml_value(value)
        elif key != "name":
            current_args[key] = _yaml_value(value)
    if current_name:
        calls.append(ToolCall.create(current_name, current_args))
    return calls or None


def _parse_dsml_tool_calls(text: str, tool_schemas: dict[str, dict[str, Any]] | None = None) -> tuple[list[ToolCall], str] | None:
    if "dsml" not in text.casefold():
        return None
    block_match = _DSML_TOOL_CALLS_BLOCK.search(text)
    if block_match is None:
        return None
    calls: list[ToolCall] = []
    for inv in _DSML_INVOKE.finditer(block_match.group(1)):
        tool_name = inv.group(2).strip()
        body = inv.group(3)
        params: dict[str, Any] = {}
        param_types = _schema_for_name(tool_schemas, tool_name)
        for param in _DSML_PARAMETER.finditer(body):
            key = param.group(2).strip()
            params[key] = _xml_value(param.group(3), (param_types or {}).get(key))
        if not params:
            normalized = _DSML_XML_NORMALIZE.sub(r"<\1\2>", body)
            parsed = _xml_invoke_arguments(normalized, param_types)
            if parsed:
                params = parsed
        calls.append(ToolCall.create(tool_name, params))
    if not calls:
        return None
    before = text[: block_match.start()]
    after = text[block_match.end() :]
    wrapper = _strip_dsml((before + " " + after).strip()).strip()
    return calls, wrapper


def _lax_tool_name(attrs: str) -> str | None:
    match = _DSML_LAX_NAME_ATTR.search(attrs)
    if match is not None:
        return match.group(2).strip()
    return None


def _infer_tool_name_from_schemas(param_keys: set[str], tool_schemas: dict[str, dict[str, Any]] | None) -> str | None:
    if not param_keys or not tool_schemas:
        return None
    candidates: list[tuple[int, str]] = []
    for name, spec in tool_schemas.items():
        if not isinstance(spec, dict):
            continue
        properties = set(spec) - {"_aliases"}
        if not properties:
            continue
        candidates.append((len(properties & param_keys), str(name)))
    if not candidates:
        return None
    best = max(candidates, key=lambda item: (item[0], -len(item[1])))
    tied = [item for item in candidates if item[0] == best[0]]
    if len(tied) != 1:
        return None
    return best[1]


def _parse_dsml_lax_tool_calls(text: str, tool_schemas: dict[str, dict[str, Any]] | None = None) -> tuple[list[ToolCall], str] | None:
    if _DSML_LAX_TAG.search(text) is None:
        return None
    block_match = _DSML_LAX_BLOCK.search(text)
    block = block_match.group("body") if block_match is not None else text
    opens = list(_DSML_LAX_OPENANY.finditer(block))
    invokes = [o for o in opens if o.group("tagname").strip().lower() not in _DSML_LAX_SKIP_TAGS]
    params = list(_DSML_LAX_PARAMETER.finditer(block))
    calls: list[ToolCall] = []
    for index, invoke in enumerate(invokes):
        tool_name = _lax_tool_name(invoke.group("attrs"))
        if not tool_name:
            next_start = invokes[index + 1].start() if index + 1 < len(invokes) else len(block)
            tail = block[invoke.end() : next_start]
            tail_name = _DSML_LAX_TOOLNAME_TAIL.search(tail)
            if tail_name is not None:
                tool_name = tail_name.group(2).strip()
        if not tool_name:
            continue
        param_types = _schema_for_name(tool_schemas, tool_name)
        params_by_call: dict[str, Any] = {}
        for param in params:
            if param.start() <= invoke.start():
                continue
            if index + 1 < len(invokes) and param.start() >= invokes[index + 1].start():
                continue
            key = param.group("name").strip()
            params_by_call[key] = _xml_value(param.group("value"), (param_types or {}).get(key))
        calls.append(ToolCall.create(tool_name, params_by_call))
    if not calls and block_match is not None and params:
        inferred = _infer_tool_name_from_schemas({item.group("name").strip() for item in params}, tool_schemas)
        if inferred is not None:
            param_types = _schema_for_name(tool_schemas, inferred)
            inferred_params: dict[str, Any] = {}
            for param in params:
                key = param.group("name").strip()
                inferred_params[key] = _xml_value(param.group("value"), (param_types or {}).get(key))
            calls.append(ToolCall.create(inferred, inferred_params))
    if not calls:
        return None
    spans: list[tuple[int, int]] = [(o.start(), o.end()) for o in _DSML_LAX_OPENANY.finditer(text)]
    spans.extend((p.start(), p.end()) for p in _DSML_LAX_PARAMETER.finditer(text))
    spans.sort()
    wrapper_parts: list[str] = []
    cursor = 0
    for start, end in spans:
        if end < cursor:
            continue
        if start > cursor:
            wrapper_parts.append(text[cursor:start])
        cursor = end
    wrapper_parts.append(text[cursor:])
    wrapper = " ".join(_DSML_NAKED.sub(" ", _DSML_LAX_TAG.sub(" ", " ".join(wrapper_parts))).split())
    return calls, wrapper


def _parse_tool_calls_impl(
    text: str,
    tool_schemas: dict[str, dict[str, Any]] | None,
    report: dict[str, Any] | None,
) -> tuple[list[ToolCall], str] | None:
    if not text or not text.strip():
        return None
    dsml_parsed = _parse_dsml_tool_calls(text, tool_schemas)
    if dsml_parsed is not None:
        if report is not None:
            report["strategies"].append("dsml")
        return dsml_parsed
    dsml_lax_parsed = _parse_dsml_lax_tool_calls(text, tool_schemas)
    if dsml_lax_parsed is not None:
        if report is not None:
            report["strategies"].append("dsml_lax")
        return dsml_lax_parsed
    stripped = _strip_fences(_strip_dsml(text))
    extracted = _extract_json_object(stripped)
    if extracted is not None:
        obj, start, end = extracted
        wrapped_calls = _extract_wrapped_calls(obj)
        if wrapped_calls is not None:
            wrapper_parts = []
            surrounding = (stripped[:start].strip() + " " + stripped[end + 1 :].strip()).strip()
            surrounding = _XML_OPEN_TAG.sub(" ", surrounding)
            surrounding = _XML_CLOSE_TAG.sub(" ", surrounding)
            surrounding = " ".join(surrounding.split())
            if surrounding:
                wrapper_parts.append(surrounding)
            inner = obj.get("content")
            if isinstance(inner, str) and inner.strip():
                wrapper_parts.append(inner.strip())
            if report is not None:
                report["strategies"].append("json_wrapped")
            return wrapped_calls, " ".join(wrapper_parts).strip()
    array_calls = _parse_bare_array_calls(stripped)
    if array_calls:
        if report is not None:
            report["strategies"].append("json_array")
        return array_calls, ""
    xml_calls, wrapper = _parse_xml_tool_calls(stripped, tool_schemas)
    if xml_calls:
        if report is not None:
            report["strategies"].append("xml")
        return xml_calls, wrapper
    yaml_calls = _parse_yaml_calls(stripped)
    if yaml_calls:
        if report is not None:
            report["strategies"].append("yaml")
        return yaml_calls, ""
    calls: list[ToolCall] = []
    removed = bytearray(len(stripped))
    for obj, start, end in _iter_json_objects(stripped):
        found = _extract_calls(obj)
        if found:
            calls.extend(found)
            removed[start : end + 1] = b" " * (end - start + 1)
    if calls:
        wrapper = _blanked(stripped, removed)
        if report is not None:
            report["strategies"].append("json_in_prose")
        return calls, " ".join(wrapper.split())
    return None


def parse_tool_calls(
    text: str,
    tool_schemas: dict[str, dict[str, Any]] | None = None,
    tool_details: dict[str, dict[str, Any]] | None = None,
    fix_mode: str | None = None,
) -> tuple[list[ToolCall], str] | None:
    result = _parse_tool_calls_impl(text, tool_schemas, None)
    if result is None:
        return None
    calls, wrapper = result
    normalized = [ToolCall(call.id, _normalize_call_name(call.name, tool_schemas), call.arguments) for call in calls]
    if fix_mode:
        normalized = fix_tool_calls(normalized, tool_schemas, tool_details, fix_mode)
    return normalized, wrapper


def parse_tool_calls_debug(
    text: str,
    tool_schemas: dict[str, dict[str, Any]] | None = None,
    tool_details: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    stripped = _strip_fences(_strip_dsml(text))
    report: dict[str, Any] = {
        "text": text,
        "stripped": stripped,
        "parsed": False,
        "strategies": [],
        "calls": [],
        "renamed": [],
        "wrapper": "",
        "unrecognized": stripped,
        "fixes": [],
        "warnings": [],
    }
    result = _parse_tool_calls_impl(text, tool_schemas, report)
    if result is not None:
        calls, wrapper = result
        normalized = [(call, _normalize_call_name(call.name, tool_schemas)) for call in calls]
        renamed = [{"from": call.name, "to": name} for call, name in normalized if call.name != name]
        applied = [ToolCall(call.id, name, call.arguments) for call, name in normalized]
        if tool_details is not None:
            applied = fix_tool_calls(applied, tool_schemas, tool_details, "report", report)
        report["parsed"] = True
        report["renamed"] = renamed
        report["calls"] = [{"id": call.id, "name": call.name, "arguments": call.arguments} for call in applied]
        report["wrapper"] = wrapper
        report["unrecognized"] = wrapper
    return report


def format_tool_message(tool_calls: list[ToolCall], text: str, reasoning: str | None = None) -> dict:
    message: dict = {"role": "assistant", "content": text}
    message["tool_calls"] = [
        {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        }
        for call in tool_calls
    ]
    if reasoning:
        message["reasoning_content"] = reasoning
    return message


def tool_call_deltas(tool_calls: list[ToolCall], text: str | None = None) -> list[dict]:
    deltas: list[dict] = []
    if text:
        deltas.append({"role": "assistant", "content": text})
    for index, call in enumerate(tool_calls):
        deltas.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": index,
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": ""},
                    }
                ],
            }
        )
        arguments = call.arguments
        if arguments:
            step = max(1, (len(arguments) + 5) // 6)
            for offset in range(0, len(arguments), step):
                deltas.append(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "function": {"arguments": arguments[offset : offset + step]},
                            }
                        ]
                    }
                )
    return deltas
