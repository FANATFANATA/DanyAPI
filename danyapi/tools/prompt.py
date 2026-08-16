from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any


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


TOOL_CALL_INSTRUCTION = (
    "You have access to the following functions. Call them when the user's request requires it.\n\n"
    "{functions}\n\n"
    "When you need to call one or more functions, reply with ONLY a single valid JSON object "
    "(no markdown fences, no code blocks, no extra text, no explanation) in exactly this format:\n"
    '{{"tool_calls": [{{"name": "get_weather", "arguments": {{"city": "Moscow"}}}}]}}\n\n'
    "Use the exact function names and JSON argument keys from the definitions above. "
    'The value of "arguments" must be a JSON object with only those keys. '
    'You may include multiple entries in the "tool_calls" array to call several functions at once.\n'
    "{choice}"
)

CHOICE_INSTRUCTIONS = {
    "required": "You MUST call one or more functions from the list above.",
    "function": "You MUST call exactly the function specified below and no other functions.",
}

JSON_MODE_INSTRUCTION = """You must reply with ONLY a valid JSON object, no markdown fences, no extra text, no explanations, no comments inside the JSON.
{constraints}"""




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


def render_message(msg: Any) -> str:
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
        return f"{prefix}: {text}"
    if role == "function":
        name = getattr(msg, "name", None) or ""
        return f"Function {name} returned: {text}"
    return text


def _render_history(messages: list[Any]) -> str:
    parts = []
    for msg in messages:
        text = render_message(msg)
        if text:
            role = getattr(msg, "role", "user")
            parts.append(f"{role.capitalize()}: {text}")
    return "\n".join(parts)


def _render_tool_tail(messages: list[Any]) -> str:
    parts = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role in ("tool", "function"):
            text = render_message(msg)
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
        if _is_tool_round_tail(tail):
            tail_prompt = _render_tool_tail(tail)
            return tail_prompt, True
        base = extract_last_user(messages)
        blocks = []
        if json_block:
            blocks.append(json_block)
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

