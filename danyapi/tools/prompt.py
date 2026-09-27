from __future__ import annotations

import hashlib
import json
from typing import Any

from .common import _tool_function
from .dsml import _strip_dsml

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


def _choice_name(tool_choice: Any) -> str | None:
    if isinstance(tool_choice, str):
        return "required" if tool_choice == "any" else tool_choice
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
    required_names = {item for item in required if isinstance(item, str)} if isinstance(required, list) else set()
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
