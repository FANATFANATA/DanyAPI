from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from typing import Any

from .common import _ARGS_ALIASES, _FENCE_CLOSE, _FENCE_OPEN_RE, _NAME_ALIASES, ToolCall


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith(_FENCE_CLOSE):
        return stripped
    open_match = _FENCE_OPEN_RE.match(stripped)
    body_start = open_match.end() if open_match is not None else len(stripped)
    close = len(stripped) - len(_FENCE_CLOSE)
    if close < body_start or not stripped.startswith(_FENCE_CLOSE, close):
        return stripped
    if close > body_start and stripped[close - 1] == "\n":
        close -= 1
    return stripped[body_start:close].strip()


_MAX_FIX_DEPTH = 200

_TRAILING_COMMA_RE = re.compile(r",\s*[}\]]")


def _strip_trailing_commas(text: str) -> str:
    if _TRAILING_COMMA_RE.search(text) is None:
        return text
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
    if "'" not in text:
        return text
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
_SEPARATED_NUMBER_RE = re.compile(r"-?\d[\d_]*(\.\d[\d_]*)?([eE][+-]?\d[\d_]*)?")
_NON_FINITE_RE = re.compile(r"[+-]?(nan|inf(inity)?)", re.IGNORECASE)
_BARE_LITERALS = frozenset({"true", "false", "null", "nan", "inf", "+nan", "-nan", "+inf", "-inf", "+infinity", "-infinity", "infinity"})


def _canonical_number(token: str) -> str:
    if "_" not in token:
        return token
    return token.replace("_", "")


def _is_bare_literal(token: str) -> bool:
    if token.lower() in _BARE_LITERALS:
        return True
    if _NUMBER_RE.fullmatch(token) is not None:
        return True
    return _SEPARATED_NUMBER_RE.fullmatch(token) is not None


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
        if _is_bare_literal(token):
            canonical = _canonical_number(token)
            if canonical != token:
                token = canonical
                changed = True
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
        except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
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
            if len(stack) >= _MAX_FIX_DEPTH:
                return None
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
            number = float(value)
        except ValueError:
            return value
        return number if math.isfinite(number) else value
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
