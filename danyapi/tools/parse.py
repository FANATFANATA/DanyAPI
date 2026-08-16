from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from .prompt import ToolCall, _tool_function

_DSML_TAG = re.compile(
    r"<\s*/?\s*(?:\||\uff5c){2}\s*DSML\s*(?:\||\uff5c){2}[^>]*>"
    r"|(?:\||\uff5c){2}\s*DSML\s*(?:\||\uff5c){2}",
    re.IGNORECASE,
)


def _strip_dsml(text: str) -> str:
    if not text:
        return text
    return _DSML_TAG.sub(" ", text)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$", stripped, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def _strip_trailing_commas(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    escaped = False
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


def _loads_lenient(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        return json.loads(_strip_trailing_commas(text))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    normalized = _normalize_single_quotes(_strip_trailing_commas(text))
    if normalized != text:
        try:
            return json.loads(normalized)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    fixed = _fix_unbalanced_json(text)
    if fixed is not None and fixed != text:
        try:
            return json.loads(fixed)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
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
    stripped = _strip_fences(text)
    bounds = _balanced_json(stripped)
    if bounds is None:
        return None
    start, end = bounds
    try:
        obj = _loads_lenient(stripped[start : end + 1])
    except ValueError:
        return None
    if isinstance(obj, dict):
        return obj, start, end


def _extract_one_call(item: Any) -> ToolCall | None:
    if not isinstance(item, dict):
        return None
    fn = item.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        arguments = fn.get("arguments", {})
    else:
        name = item.get("name")
        arguments = item.get("arguments", {})
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
    calls = _extract_wrapped_calls(obj)
    if calls is not None:
        return calls
    if isinstance(obj.get("name"), str) and obj["name"] and _is_jsonish_arguments(obj.get("arguments")):
        call = _extract_one_call(obj)
        if call is not None:
            return [call]
    return None


def _unescape_xml(text: str) -> str:
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&")


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


def tool_schema_map(tools: list[Any] | None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not tools:
        return result
    for tool in tools:
        fn = _tool_function(tool)
        if not fn:
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        params = fn.get("parameters")
        if not isinstance(params, dict):
            continue
        properties = params.get("properties")
        if not isinstance(properties, dict):
            continue
        prop_types: dict[str, Any] = {}
        for prop, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            typ = spec.get("type")
            if isinstance(typ, list):
                for candidate in ("integer", "number", "boolean", "null", "string"):
                    if candidate in typ:
                        typ = candidate
                        break
            if typ:
                prop_types[prop] = typ
        if prop_types:
            result[name] = prop_types
    return result


def _xml_invoke_arguments(body: str, param_types: dict[str, Any] | None = None) -> dict[str, Any] | None:
    stripped = body.strip()
    if stripped.startswith("{"):
        try:
            parsed = _loads_lenient(stripped)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
    params: dict[str, Any] = {}
    for match in re.finditer(r'<parameter\s+name=["\']([^"\']+)["\']\s*>(.*?)</parameter>', body, re.DOTALL | re.IGNORECASE):
        params[match.group(1).strip()] = _coerce_scalar(_unescape_xml(match.group(2).strip()), (param_types or {}).get(match.group(1).strip()))
    if params:
        return params
    has_children = False
    for match in re.finditer(r"<([a-zA-Z_][a-zA-Z0-9_-]*)\s*>([^<]*)</\1>", body, re.DOTALL | re.IGNORECASE):
        params[match.group(1).strip()] = _coerce_scalar(_unescape_xml(match.group(2).strip()), (param_types or {}).get(match.group(1).strip()))
        has_children = True
    if has_children:
        return params
    inner = _unescape_xml(body.strip())
    if inner:
        return {"content": inner}
    return None


def _xml_tag_attrs(body: str, param_types: dict[str, Any] | None = None) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    pattern = re.compile(r"([a-zA-Z_][a-zA-Z0-9_.-]*)\s*=\s*(\"[^\"]*\"|'[^']*')", re.IGNORECASE)
    for match in pattern.finditer(body):
        key = match.group(1)
        raw = match.group(2)
        value = raw[1:-1]
        attrs[key] = _coerce_scalar(_unescape_xml(value), (param_types or {}).get(key))
    return attrs


def _parse_xml_tool_calls(text: str, tool_schemas: dict[str, dict[str, Any]] | None = None) -> tuple[list[ToolCall] | None, str]:
    calls: list[ToolCall] = []
    remainder = text
    consumed: list[tuple[int, int]] = []
    invoke_pattern = re.compile(
        r"<(?:invoke|use_tool|tool_use)\s+name=([\"']?)([^\s>\"']+)\1\s*>(.*?)</(?:invoke|use_tool|tool_use)>",
        re.DOTALL | re.IGNORECASE,
    )
    for match in invoke_pattern.finditer(text):
        tool_name = match.group(2)
        arguments = _xml_invoke_arguments(match.group(3), (tool_schemas or {}).get(tool_name))
        if arguments is None:
            continue
        calls.append(ToolCall.create(tool_name, arguments))
        remainder = remainder.replace(match.group(0), " ")
        consumed.append(match.span())
    block_pattern = re.compile(r"<(?:tool_call|function_call)>(.*?)</(?:tool_call|function_call)>", re.DOTALL | re.IGNORECASE)
    for match in block_pattern.finditer(text):
        parsed = _extract_json_object(match.group(1))
        if parsed is None:
            continue
        obj, _, _ = parsed
        extracted = _extract_calls(obj)
        if extracted:
            calls.extend(extracted)
            remainder = remainder.replace(match.group(0), " ")
            consumed.append(match.span())
    for tool_name, param_types in (tool_schemas or {}).items():
        escaped = re.escape(tool_name)
        open_pattern = re.compile(rf"<{escaped}(?=[\s/>])([^>]*?)>(.*?)</{escaped}>", re.DOTALL | re.IGNORECASE)
        selfclose_pattern = re.compile(rf"<{escaped}(?=[\s/>])([^>]*?)/>", re.DOTALL | re.IGNORECASE)
        for match in open_pattern.finditer(text):
            start, end = match.span()
            if any(s <= start and end <= e for s, e in consumed):
                continue
            merged = _xml_tag_attrs(match.group(1), param_types)
            merged.update(_xml_invoke_arguments(match.group(2), param_types) or {})
            if not merged:
                continue
            calls.append(ToolCall.create(tool_name, merged))
            consumed.append((start, end))
            remainder = remainder.replace(match.group(0), " ")
        for match in selfclose_pattern.finditer(text):
            start, end = match.span()
            if any(s <= start and end <= e for s, e in consumed):
                continue
            arguments = _xml_tag_attrs(match.group(1), param_types)
            if not arguments:
                continue
            calls.append(ToolCall.create(tool_name, arguments))
            consumed.append((start, end))
            remainder = remainder.replace(match.group(0), " ")
    if not calls:
        return None, ""
    remainder = re.sub(r"<(?:tool_calls|function_calls|tool_call|function_call)>", " ", remainder, flags=re.IGNORECASE)
    remainder = re.sub(r"</(?:tool_calls|function_calls|tool_call|function_call)>", " ", remainder, flags=re.IGNORECASE)
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


def _iter_json_objects(text: str) -> Iterator[tuple[dict, int, int]]:
    i = 0
    length = len(text)
    while True:
        start = text.find("{", i)
        if start == -1:
            return
        depth = 0
        in_string = False
        escaped = False
        end = start
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
                    candidate = text[start : end + 1]
                    try:
                        obj = _loads_lenient(candidate)
                        if isinstance(obj, dict):
                            yield obj, start, end
                    except ValueError:
                        pass
                    break
            end += 1
        i = end + 1


def parse_tool_calls(text: str, tool_schemas: dict[str, dict[str, Any]] | None = None) -> tuple[list[ToolCall], str] | None:
    if not text or not text.strip():
        return None
    stripped = _strip_fences(_strip_dsml(text))
    extracted = _extract_json_object(stripped)
    if extracted is not None:
        obj, start, end = extracted
        wrapped_calls = _extract_wrapped_calls(obj)
        if wrapped_calls is not None:
            wrapper_parts = []
            surrounding = (stripped[:start].strip() + " " + stripped[end + 1 :].strip()).strip()
            if surrounding:
                wrapper_parts.append(surrounding)
            inner = obj.get("content")
            if isinstance(inner, str) and inner.strip():
                wrapper_parts.append(inner.strip())
            return wrapped_calls, " ".join(wrapper_parts).strip()
    array_calls = _parse_bare_array_calls(stripped)
    if array_calls:
        return array_calls, ""
    xml_calls, wrapper = _parse_xml_tool_calls(stripped, tool_schemas)
    if xml_calls:
        return xml_calls, wrapper
    calls: list[ToolCall] = []
    removed = bytearray(len(stripped))
    for obj, start, end in _iter_json_objects(stripped):
        found = _extract_calls(obj)
        if found:
            calls.extend(found)
            removed[start : end + 1] = b" " * (end - start + 1)
    if calls:
        wrapper = "".join((stripped[i] if removed[i] == 0 else " ") for i in range(len(stripped)))
        return calls, " ".join(wrapper.split())
    return None
