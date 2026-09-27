from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from .common import (
    _ARGS_ALIASES,
    _JSON_TYPE_ATTRS,
    _TOOL_TAG_NAMES,
    _XML_ATTR_RE,
    _XML_CHILD_NAME_RE,
    _XML_NAME_ATTR_RE,
    _XML_NAME_ATTR_STRIP_RE,
    _XML_NESTED_RE,
    _XML_PARAM_RE,
    _XML_STRAY_TOOL_CLOSE_RE,
    _XML_TOOL_CALL_BLOCK_RE,
    _XML_TOOL_SELFCLOSE_RE,
    _XML_WRAPPER_CLOSE_RE,
    ToolCall,
)
from .dsml import (
    _DSML_INVOKE,
    _DSML_LAX_BLOCK,
    _DSML_LAX_NAME_ATTR,
    _DSML_LAX_OPENANY,
    _DSML_LAX_PARAMETER,
    _DSML_LAX_SKIP_TAGS,
    _DSML_LAX_TAG,
    _DSML_LAX_TOOLNAME_TAIL,
    _DSML_NAKED,
    _DSML_PARAMETER,
    _DSML_TOOL_CALLS_BLOCK,
    _DSML_XML_NORMALIZE,
    _XML_CLOSE_TAG,
    _XML_GENERIC_TOOL_TAGS,
    _XML_HTML_TAGS,
    _XML_OPEN_TAG,
    _XML_SELFCLOSE,
    _XML_SKIP_ELEMENTS,
    _XML_WRAPPER_OPEN,
    _blanked,
    _dsml_present,
    _IntervalSet,
    _scan_xml_pairs,
    _strip_dsml,
    strip_dsml,
)
from .jsonfix import (
    _coerce_scalar,
    _extract_calls,
    _extract_json_object,
    _extract_one_call,
    _extract_wrapped_calls,
    _loads_lenient,
    _strip_fences,
    _unescape_xml,
)
from .names import _normalize_call_name, _schema_for_name, fix_tool_calls


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
    for _, _, raw_key, _, inner in _scan_xml_pairs(body):
        key = raw_key.strip()
        lowered = key.lower()
        if lowered in _XML_SKIP_ELEMENTS:
            continue
        if lowered in _XML_HTML_TAGS and lowered not in _ARGS_ALIASES:
            continue
        _xml_set_param(params, key, _xml_value(inner, (param_types or {}).get(key)))
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
        value = raw[1:-1] if raw[:1] in ('"', "'") else raw
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
        mask[start:end] = b"\x01" * (end - start)

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
        if not block_calls and not any(_scan_xml_pairs(inner, _TOOL_TAG_NAMES, tail_space=True)):
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
    schema_items = tool_schemas.items() if isinstance(tool_schemas, dict) else ()
    for tool_name, raw_types in schema_items:
        if not isinstance(tool_name, str) or not tool_name:
            continue
        param_types = raw_types if isinstance(raw_types, dict) else None
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
            i = start + 1
            continue
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
        number = float(value)
    except (ValueError, TypeError):
        pass
    else:
        if math.isfinite(number):
            return number
    return value


def _parse_yaml_calls(text: str) -> list[ToolCall] | None:
    body = [line.strip() for line in text.splitlines() if line.strip()]
    if not body:
        return None
    root = body[0]
    root_match = _YAML_TOOL_CALLS_RE.match(root)
    if root_match is None:
        return None
    inline = root_match.group(1).strip()
    rest = body[1:]
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
    if not _dsml_present(text):
        return None
    blocks = list(_DSML_TOOL_CALLS_BLOCK.finditer(text))
    if not blocks:
        return None
    calls: list[ToolCall] = []
    for block_match in blocks:
        for inv in _DSML_INVOKE.finditer(block_match.group(1)):
            tool_name = inv.group(2).strip()
            body = inv.group(3)
            params: dict[str, Any] = {}
            param_types = _schema_for_name(tool_schemas, tool_name)
            for param in _DSML_PARAMETER.finditer(body):
                key = param.group(2).strip()
                raw = _DSML_XML_NORMALIZE.sub(r"<\1\2>", param.group(3))
                _xml_set_param(params, key, _xml_value(raw, (param_types or {}).get(key)))
            if not params:
                normalized = _DSML_XML_NORMALIZE.sub(r"<\1\2>", body)
                parsed = _xml_invoke_arguments(normalized, param_types)
                if parsed:
                    params = parsed
            calls.append(ToolCall.create(tool_name, params))
    if not calls:
        return None
    outside: list[str] = []
    cursor = 0
    for block_match in blocks:
        outside.append(text[cursor : block_match.start()])
        cursor = block_match.end()
    outside.append(text[cursor:])
    wrapper = _strip_dsml(" ".join(outside).strip()).strip()
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
            _xml_set_param(params_by_call, key, _xml_value(param.group("value"), (param_types or {}).get(key)))
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
    joined = " ".join(wrapper_parts)
    joined = _XML_STRAY_TOOL_CLOSE_RE.sub(" ", joined)
    joined = _XML_OPEN_TAG.sub(" ", joined)
    joined = _XML_CLOSE_TAG.sub(" ", joined)
    wrapper = " ".join(_DSML_NAKED.sub(" ", _DSML_LAX_TAG.sub(" ", joined)).split())
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
            removed[start : end + 1] = b"\x01" * (end - start + 1)
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
        message["reasoning_content"] = strip_dsml(reasoning)
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
