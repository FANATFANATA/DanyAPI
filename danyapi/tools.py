from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import jsonschema

_DSML_TAG = re.compile(
    r"<\s*/?\s*(?:\||\uff5c){2}\s*DSML\s*(?:\||\uff5c){2}[^>]*>"
    r"|(?:\||\uff5c){2}\s*DSML\s*(?:\||\uff5c){2}",
    re.IGNORECASE,
)


def _strip_dsml(text: str) -> str:
    if not text:
        return text
    return _DSML_TAG.sub(" ", text)


TOOL_CALL_INSTRUCTION = (
    "You have access to the following functions. Call them when the user's request requires it.\n\n"
    "{functions}\n\n"
    "When you need to call one or more functions, reply with ONLY a single valid JSON object "
    "(no markdown fences, no code blocks, no extra text, no explanation) in exactly this format:\n"
    '{{"tool_calls": [{{"name": "get_weather", "arguments": {{"city": "Moscow"}}}}]}}\n\n'
    "Use the exact function names and JSON argument keys from the definitions above. "
    'The value of "arguments" must be a JSON object with only those keys. '
    'You may include multiple entries in the "tool_calls" array to call several functions at once. '
    "Large tool results are truncated to fit the context window, so keep tool inputs "
    "(file line ranges, output sizes) as small as practical.\n"
    "{choice}"
)

CHOICE_INSTRUCTIONS = {
    "required": "You MUST call one or more functions from the list above.",
    "function": "You MUST call exactly the function specified below and no other functions.",
}

# Default cap for a single tool result when rendering the upstream prompt.
# A full file read (up to ~100 KB) would otherwise blow the upstream context.
# 0 disables truncation.
TOOL_RESULT_MAX_CHARS = 8000


def _truncate_tool_result(text: str, limit: int) -> str:
    if limit > 0 and len(text) > limit:
        omitted = len(text) - limit
        return f"{text[:limit]} ... [truncated: {omitted} more characters]"
    return text


JSON_MODE_INSTRUCTION = """You must reply with ONLY a valid JSON object, no markdown fences, no extra text, no explanations, no comments inside the JSON.
{constraints}"""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    @classmethod
    def create(cls, name: str, arguments: Any) -> ToolCall:
        call_id = f"call_{uuid.uuid4().hex[:12]}"
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
        fn = tool_choice.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            return fn["name"]
    return None


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
        if fn.get("strict"):
            lines.append("   strict: true (output arguments must match the schema exactly)")
        params = fn.get("parameters")
        if params is not None:
            if isinstance(params, str):
                params_json = params
            else:
                params_json = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
            lines.append(f"   parameters (JSON Schema): {params_json}")
    if choice in CHOICE_INSTRUCTIONS:
        choice_line = CHOICE_INSTRUCTIONS[choice]
    elif isinstance(choice, str) and choice not in ("auto", "none", "required"):
        choice_line = f"You MUST call exactly the function {choice} and no other functions."
    else:
        choice_line = "If you do not need to call any function, reply normally with your answer."
    return TOOL_CALL_INSTRUCTION.format(functions="\n".join(lines), choice=choice_line)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "image_url":
                    continue
        return "".join(parts).strip()
    return ""


def _content_fingerprint(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, str):
                        parts.append(image_url)
                    elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                        parts.append(image_url["url"])
        return "\n".join(parts)
    return ""


def context_sequence(messages: list[Any], user: str | None = None) -> tuple[str, ...]:
    sequence: list[str] = []
    scope = f"\0{user or ''}"
    for msg in messages:
        role = getattr(msg, "role", "user")
        if role not in ("system", "user"):
            continue
        content = _content_fingerprint(getattr(msg, "content", ""))
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


def render_message(msg: Any, tool_result_limit: int = TOOL_RESULT_MAX_CHARS) -> str:
    role = getattr(msg, "role", "user")
    text = _content_text(getattr(msg, "content", ""))
    if role in ("user", "system"):
        return text
    if role == "assistant":
        parts = []
        if text:
            parts.append(text)
        for call in getattr(msg, "tool_calls", None) or []:
            mention = _render_tool_call_mention(call)
            if mention:
                parts.append(mention)
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_call":
                    mention = _render_tool_call_mention(item)
                    if mention:
                        parts.append(mention)
        return "; ".join(parts)
    if role == "tool":
        tool_call_id = getattr(msg, "tool_call_id", None) or ""
        prefix = f"Tool result ({tool_call_id})" if tool_call_id else "Tool result"
        return f"{prefix}: {_truncate_tool_result(text, tool_result_limit)}"
    if role == "function":
        name = getattr(msg, "name", None) or ""
        return f"Function {name} returned: {_truncate_tool_result(text, tool_result_limit)}"
    return text


def _render_history(messages: list[Any], tool_result_limit: int = TOOL_RESULT_MAX_CHARS) -> str:
    parts = []
    for msg in messages:
        text = render_message(msg, tool_result_limit)
        if text:
            role = getattr(msg, "role", "user")
            parts.append(f"{role.capitalize()}: {text}")
    return "\n".join(parts)


def _render_tool_tail(messages: list[Any], tool_result_limit: int = TOOL_RESULT_MAX_CHARS) -> str:
    parts = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role in ("tool", "function"):
            text = render_message(msg, tool_result_limit)
            if text:
                parts.append(text)
    parts.append("Continue the conversation and provide the final answer based on the tool results.")
    return "\n".join(parts)


def extract_last_user(messages: list[Any]) -> str:
    if not messages:
        raise ValueError("messages is required")
    for msg in reversed(messages):
        if getattr(msg, "role", None) in ("user", "system"):
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                return content
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
                text = "".join(parts).strip()
                if text:
                    return text
                continue
            raise ValueError("unsupported message content")
    raise ValueError("no user message found")


def is_tool_round(messages: list[Any]) -> bool:
    for msg in messages:
        role = getattr(msg, "role", None)
        if role in ("tool", "function"):
            return True
        if role == "assistant" and getattr(msg, "tool_calls", None):
            return True
        if role == "assistant" and isinstance(getattr(msg, "content", None), list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "tool_call":
                    return True
    return False


def _has_history(messages: list[Any]) -> bool:
    user_count = 0
    for msg in messages:
        role = getattr(msg, "role", None)
        if role == "user":
            user_count += 1
            continue
        text = _content_text(getattr(msg, "content", ""))
        if role == "assistant" and (text or getattr(msg, "tool_calls", None)):
            return True
        if role in ("tool", "function") and text:
            return True
    return user_count > 1


def _tail_after_last_user(messages: list[Any]) -> list[Any]:
    index = -1
    for i, msg in enumerate(messages):
        if getattr(msg, "role", None) in ("user", "system"):
            index = i
    if index < 0:
        return list(messages)
    return list(messages[index + 1 :])


def _is_tool_round_tail(messages: list[Any]) -> bool:
    for msg in messages:
        role = getattr(msg, "role", None)
        if role in ("tool", "function"):
            return True
        if role == "assistant" and getattr(msg, "tool_calls", None):
            return True
        if role == "assistant" and isinstance(getattr(msg, "content", None), list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "tool_call":
                    return True
    return False


def extract_system(messages: list[Any]) -> str:
    parts = []
    for msg in messages:
        if getattr(msg, "role", None) == "system":
            text = _content_text(getattr(msg, "content", "")).strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def render_json_mode(response_format: Any) -> str | None:
    if response_format is None:
        return None
    constraints = "The JSON object must be the only thing in your reply."
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
        constraints = f"The JSON object must match this JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}"
    return JSON_MODE_INSTRUCTION.format(constraints=constraints)


def response_format_schema(response_format: Any) -> dict | None:
    """The JSON Schema from a response_format of type "json_schema", if any.

    Only a dict is returned: validation against a non-dict schema would raise
    jsonschema.SchemaError, which the handlers do not catch.
    """
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") != "json_schema":
        return None
    raw = response_format.get("json_schema")
    schema = raw.get("schema") if isinstance(raw, dict) else None
    return schema if isinstance(schema, dict) else None


def validate_json_response(response: Any, schema: dict | None) -> None:
    """Validate a parsed JSON response against the schema.

    Raises:
        ValueError: If validation fails.
    """
    if schema is None:
        return
    try:
        jsonschema.validate(instance=response, schema=schema)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"Response does not match JSON schema: {exc.message}") from exc


def build_prompt(
    messages: list[Any],
    tools: list[Any] | None = None,
    tool_choice: Any = None,
    has_session: bool = False,
    response_format: Any = None,
    tool_result_limit: int = TOOL_RESULT_MAX_CHARS,
) -> tuple[str, bool]:
    schema = render_tool_schema(tools, tool_choice)
    tools_present = schema is not None
    json_block = render_json_mode(response_format)

    if has_session:
        tail = _tail_after_last_user(messages)
        if _is_tool_round_tail(tail):
            tail_prompt = _render_tool_tail(tail, tool_result_limit)
            return tail_prompt, True
        base = extract_last_user(messages)
        blocks = []
        if json_block:
            blocks.append(json_block)
        blocks.append(base)
        return "\n\n".join(blocks), tools_present

    tool_round_active = is_tool_round(messages)
    if tool_round_active or _has_history(messages):
        prompt = _render_history(messages, tool_result_limit)
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


_TOOL_CALLS_BAD_SEP = re.compile(r'"tool_calls"\s*([^\s":{\[\],\d\w-])')


def _repair_tool_calls_key(text: str) -> str:
    """Repair a corrupted separator after the `"tool_calls"` key.

    Models occasionally emit e.g. `"tool_calls">` instead of `"tool_calls": [`,
    dropping the colon and the opening bracket. Re-insert the colon, and the
    opening bracket as well unless the value already starts with one, so that
    a single object or a comma-separated list of objects parses as an array.
    """

    def _fix(match: re.Match) -> str:
        if text[match.end() :].lstrip().startswith("["):
            return '"tool_calls": '
        return '"tool_calls": ['

    return _TOOL_CALLS_BAD_SEP.sub(_fix, text)


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
    # Repair a corrupted "tool_calls" key separator (e.g. `"tool_calls">`)
    repaired = _repair_tool_calls_key(text)
    if repaired != text:
        for candidate in (repaired, _strip_trailing_commas(repaired), _fix_unbalanced_json(repaired)):
            if candidate is None:
                continue
            try:
                return json.loads(candidate)
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


def _coerce_json_container(value: str) -> Any:
    """Parse a JSON array/object embedded in a string parameter value.

    DeepSeek emits nested structures as pre-serialized JSON strings; if the
    value does not parse, or parses to a scalar, keep the raw string."""
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        parsed = _loads_lenient(stripped)
    except ValueError:
        return value
    return parsed if isinstance(parsed, (list, dict)) else value


def _coerce_scalar(value: str, json_type: Any) -> Any:
    if not isinstance(value, str):
        return value
    if json_type in ("array", "object"):
        return _coerce_json_container(value)
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


# Non-null types win: for union/nullable specs we coerce to the real value type,
# not to None. "null" is the fallback for nullable-only properties. Containers
# (array/object) rank above string: a failed JSON parse falls back to the raw
# string, while a string hint would never attempt a parse.
_TYPE_PRIORITY = ("integer", "number", "boolean", "array", "object", "string", "null")


def _collect_declared_types(spec: dict) -> set[str]:
    """All JSON types declared in a property spec, including types nested
    inside anyOf/oneOf/allOf combinators (how union types are expressed)."""
    types: set[str] = set()
    typ = spec.get("type")
    if isinstance(typ, str):
        types.add(typ)
    elif isinstance(typ, list):
        types.update(t for t in typ if isinstance(t, str))
    for combinator in ("anyOf", "oneOf", "allOf"):
        variants = spec.get(combinator)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if isinstance(variant, dict):
                types |= _collect_declared_types(variant)
    return types


def _coerce_untyped(value: str) -> Any:
    """Coerce a parameter value whose type is unknown from the schema,
    using JSON literal semantics. Non-literals are kept as strings."""
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return _coerce_json_container(value)


_STRING_ATTR_FALSE = re.compile(r'string\s*=\s*["\']?false["\']?', re.IGNORECASE)


def _string_attr_is_false(attrs: str) -> bool:
    """Whether a <parameter> tag carries the model's `string="false"` hint,
    i.e. the value is a non-string JSON literal rather than text."""
    return _STRING_ATTR_FALSE.search(attrs) is not None


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
            declared = _collect_declared_types(spec)
            if declared:
                for candidate in _TYPE_PRIORITY:
                    if candidate in declared:
                        prop_types[prop] = candidate
                        break
        if prop_types:
            result[name] = prop_types
    return result


def _xml_invoke_arguments(body: str, param_types: dict[str, Any] | None = None) -> dict[str, Any] | None:
    stripped = body.strip()
    if stripped and stripped[0] in "{[":
        try:
            parsed = _loads_lenient(stripped)
            if isinstance(parsed, (dict, list)):
                return parsed
        except ValueError:
            pass
    params: dict[str, Any] = {}
    for match in re.finditer(r'<parameter\s+name=["\']([^"\']+)["\']([^>]*)>(.*?)</parameter>', body, re.DOTALL | re.IGNORECASE):
        name = match.group(1).strip()
        value = _unescape_xml(match.group(3).strip())
        type_hint = (param_types or {}).get(name)
        if type_hint is None and _string_attr_is_false(match.group(2)):
            value = _coerce_untyped(value)
        params[name] = _coerce_scalar(value, type_hint)
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
    # Empty body = zero-argument call (e.g. EnterPlanMode) → {}, not "no call".
    return {}


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
                deltas.append({"tool_calls": [{"index": index, "function": {"arguments": arguments[offset : offset + step]}}]})
    return deltas
