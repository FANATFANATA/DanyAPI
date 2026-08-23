from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any

_DSML_PIPE = r"|\u00a6\u01c0\u01c1\u05c0\u2016\u2223\u2502\u2551\u2758\ufe31\uff5c"
_DSML_JUNK = r"[^\x00-\x7f]"
_DSML_MARKER = rf"(?:[{_DSML_PIPE}]|{_DSML_JUNK})+\s*DSML\s*(?:[{_DSML_PIPE}]|{_DSML_JUNK})+"
_DSML_PIPE_ANGLE = rf"[{_DSML_PIPE}<>]"
_DSML_BLOCK = re.compile(
    rf"<{_DSML_PIPE_ANGLE}+\s*[a-zA-Z_][^<>]*\s*{_DSML_PIPE_ANGLE}+>\s*DSML\s*<{_DSML_PIPE_ANGLE}+\s*[a-zA-Z_][^<>]*\s*{_DSML_PIPE_ANGLE}+>",
    re.IGNORECASE,
)
_DSML_WRAP = re.compile(
    rf"(?:[{_DSML_PIPE}]|{_DSML_JUNK})+\s*>\s*DSML\s*<\s*(?:[{_DSML_PIPE}]|{_DSML_JUNK})+",
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
_DSML_HIDDEN = re.compile(
    rf"<{_DSML_MARKER}\s*({_DSML_HIDDEN_NAMES})\b[^<>]*>.*?</{_DSML_MARKER}\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_HIDDEN_NAKED = re.compile(
    rf"{_DSML_MARKER}\s*<({_DSML_HIDDEN_NAMES})\b[^<>]*>.*?</\1>\s*{_DSML_MARKER}",
    re.DOTALL | re.IGNORECASE,
)
_XML_ELEMENT = re.compile(r"<([a-zA-Z_][a-zA-Z0-9_-]*)\b([^>]*)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
_XML_SELFCLOSE = re.compile(r"<([a-zA-Z_][a-zA-Z0-9_-]*)\b([^>]*?)/>", re.DOTALL | re.IGNORECASE)
_XML_WRAPPER_OPEN = re.compile(
    r"<(?:tool_calls|tool_call|function_calls|function_call|tools)\b[^>]*>",
    re.IGNORECASE,
)
_XML_SKIP_ELEMENTS = frozenset(
    {
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
    r"<(?:tool_calls|tool_call|function_calls|function_call|functions|function|tools)\b[^>]*>",
    re.IGNORECASE,
)
_XML_CLOSE_TAG = re.compile(
    r"</(?:tool_calls|tool_call|function_calls|function_call|functions|function|tools)\s*>",
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
    r'<parameter\b[^>]*?\bname\s*=\s*(["\'])([^"\']+)\1[^>]*>(.*?)</parameter\s*>',
    re.DOTALL | re.IGNORECASE,
)
_XML_ATTR_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_.-]*)\s*=\s*(\"[^\"]*\"|'[^']*')", re.IGNORECASE)
_XML_NESTED_RE = re.compile(r"<[a-zA-Z_]")
_PYTHON_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_PYTHON_CALL_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_-]*)\s*\(")
_XML_WRAPPER_CLOSE_RE = re.compile(r"</(?:tool_calls|tool_call|function_calls|function_call|tools)\s*>", re.IGNORECASE)
_XML_TOOL_NAMES = r"invoke|toolinvoke|tool_invoke|use_tool|tool_use|call|function|tool"
_XML_TOOL_ELEMENT_RE = re.compile(
    rf"<(?:{_XML_TOOL_NAMES})\b([^>]*)>(.*?)</(?:{_XML_TOOL_NAMES})\s*>",
    re.DOTALL | re.IGNORECASE,
)
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
    for _ in range(10):
        updated = _DSML_BLOCK.sub(" ", result)
        updated = _DSML_WRAP.sub(" ", updated)
        updated = _DSML_HIDDEN.sub(" ", updated)
        updated = _DSML_HIDDEN_NAKED.sub(" ", updated)
        if updated == result:
            break
        result = updated
    result = _DSML_XML_NORMALIZE.sub(r"<\1\2>", result)
    result = _DSML_TAG.sub(_replace_dsml_tag, result)
    return _DSML_NAKED.sub(" ", result)


TOOL_CALL_INSTRUCTION = (
    "You have access to the following functions. Call them when the user's request requires it.\n\n"
    "{functions}\n\n"
    "When you need to call one or more functions, reply with ONLY an XML block in exactly this format:\n"
    "<tool_calls>\n"
    '<invoke name="TOOL_NAME">\n'
    '<parameter name="ARG_NAME">value</parameter>\n'
    "</invoke>\n"
    "</tool_calls>\n\n"
    "Example:\n"
    "{example}\n\n"
    "Rules:\n"
    "- Call ONLY functions from the list above; copy each function name character-for-character.\n"
    "- Never invent, rename, abbreviate, translate, or guess function names.\n"
    "- Use the exact argument keys from the definitions above; never invent or rename argument keys.\n"
    "- Put every function call in its own <invoke> element inside the <tool_calls> block.\n"
    "- Emit several sibling <invoke> elements for several independent calls.\n"
    "- Put every argument in its own <parameter> element; the name attribute must be the exact argument key.\n"
    "- If a function has no parameters, or you need no arguments, emit its <invoke> element with no <parameter> children.\n"
    "- Omit optional arguments you do not need.\n"
    "- If none of the functions fit the request, reply normally with your answer instead of calling anything.\n"
    "- No markdown fences, no code blocks, no comments, no text before or after the XML block, no other XML tags.\n"
    "{choice}"
)

TOOL_TAIL_REMINDER = (
    "If another function call is needed, reply with ONLY an XML block:\n"
    "<tool_calls>\n"
    '<invoke name="FUNCTION_NAME">\n'
    '<parameter name="ARG_NAME">value</parameter>\n'
    "</invoke>\n"
    "</tool_calls>\n"
    "Call functions ONLY by their exact names listed here; never invent function names.\n"
    "Available functions: {names}\n"
    "If no further function call is needed, reply with your final answer."
)

TOOL_HISTORY_REMINDER = (
    "Remember: to call any function, reply ONLY with the <tool_calls> XML block using the exact function names and argument keys defined above; "
    "if no function call is needed, reply with your final answer."
)

CHOICE_INSTRUCTIONS = {
    "required": "You MUST call one or more functions from the list above.",
    "function": "You MUST call exactly the function specified below and no other functions.",
}

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


def _tool_names(tools: list[Any] | None) -> list[str]:
    names: list[str] = []
    if not tools:
        return names
    for tool in tools:
        fn = _tool_function(tool)
        if fn is not None and isinstance(fn.get("name"), str) and fn["name"]:
            names.append(fn["name"])
    return names


_EXAMPLE_MAX_FUNCTIONS = 3
_EXAMPLE_MAX_PARAMETERS = 3


def _example_value(spec: Any) -> str:
    ptype = spec.get("type") if isinstance(spec, dict) else None
    if isinstance(ptype, list):
        ptype = next((t for t in ptype if isinstance(t, str) and t != "null"), None)
    return {
        "integer": "1",
        "number": "1",
        "boolean": "true",
        "array": "[]",
        "object": "{}",
    }.get(ptype if isinstance(ptype, str) else "", "value")


def _example_tool_call(fn: dict) -> str:
    name = fn.get("name") or "tool"
    params = fn.get("parameters")
    properties = params.get("properties") if isinstance(params, dict) else None
    if isinstance(properties, dict) and properties:
        lines = [f'<parameter name="{key}">{_example_value(spec)}</parameter>' for key, spec in list(properties.items())[:_EXAMPLE_MAX_PARAMETERS]]
        joined = "\n".join(lines)
        return f'<invoke name="{name}">\n{joined}\n</invoke>'
    return f'<invoke name="{name}"></invoke>'


def _examples_block(functions: list[dict]) -> str:
    invokes = "\n".join(_example_tool_call(fn) for fn in functions[:_EXAMPLE_MAX_FUNCTIONS])
    return f"<tool_calls>\n{invokes}\n</tool_calls>"


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
        aliases = fn.get("aliases")
        if isinstance(aliases, (list, tuple)):
            rendered = ", ".join(a for a in aliases if isinstance(a, str) and a)
            if rendered:
                lines.append(f"   aliases accepted: {rendered}")
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
    return TOOL_CALL_INSTRUCTION.format(
        functions="\n".join(lines),
        example=_examples_block(functions),
        choice=choice_line,
    )


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
    text = _strip_dsml(_content_text(getattr(msg, "content", "")))
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
                    parts.append(_render_tool_call_mention(item))
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


def _render_tool_tail(messages: list[Any], tool_names: list[str] | None = None) -> str:
    parts = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role in ("tool", "function"):
            parts.append(render_message(msg))
    parts.append("Continue the conversation and provide the final answer based on the tool results.")
    if tool_names:
        parts.append(TOOL_TAIL_REMINDER.format(names=", ".join(tool_names)))
    return "\n".join(parts)


def extract_last_user(messages: list[Any]) -> str:
    if not messages:
        raise ValueError("messages is required")
    for msg in reversed(messages):
        if getattr(msg, "role", None) in ("user", "system"):
            content = getattr(msg, "content", "")
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
    return is_tool_round(messages)


def extract_system(messages: list[Any]) -> str:
    parts = []
    for msg in messages:
        if getattr(msg, "role", None) == "system":
            text = _strip_dsml(_content_text(getattr(msg, "content", ""))).strip()
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
            tail_prompt = _render_tool_tail(tail, _tool_names(tools) if tools_present else None)
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
        if schema:
            prompt = f"{prompt}\n\n{TOOL_HISTORY_REMINDER}"
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


_NUMBER_RE = re.compile(r"-?\d+(\.\d+)?([eE][+-]?\d+)?")
_BARE_LITERALS = frozenset({"true", "false", "null"})


def _is_bare_literal(token: str) -> bool:
    if token.lower() in _BARE_LITERALS:
        return True
    return _NUMBER_RE.fullmatch(token) is not None


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
        if prev in ("{", "[", ",") and j < n and text[j] == ":":
            out.append('"')
            out.append(token)
            out.append('"')
            prev = '"'
            i = j
            changed = True
            continue
        if prev == ":" and not _is_bare_literal(token):
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
    name, arguments = _call_item_fields(obj)
    if isinstance(name, str) and name and _is_jsonish_arguments(arguments):
        return [ToolCall.create(name, arguments)]
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


def _schema_for_name(tool_schemas: dict[str, dict[str, Any]] | None, name: str) -> dict[str, Any] | None:
    if not tool_schemas:
        return None
    if name in tool_schemas:
        return tool_schemas[name]
    for known, spec in tool_schemas.items():
        if known.casefold() == name.casefold():
            return spec
    resolved = _resolve_alias(name, tool_schemas)
    if resolved is not None:
        return tool_schemas[resolved]
    return None


def _name_key(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", name.casefold())


def _resolve_alias(name: str, tool_schemas: dict[str, dict[str, Any]] | None) -> str | None:
    if not tool_schemas:
        return None
    folded = name.casefold()
    key = _name_key(name)
    if not key:
        return None
    for known, spec in tool_schemas.items():
        aliases = (spec or {}).get("_aliases") or []
        for alias in aliases:
            if isinstance(alias, str) and (_name_key(alias) == key or alias.casefold() == folded):
                return known
    return None


def _fuzzy_known_name(name: str, tool_schemas: dict[str, dict[str, Any]]) -> str | None:
    key = _name_key(name)
    if len(key) < 4:
        return None
    matches = get_close_matches(key, [_name_key(known) for known in tool_schemas], n=2, cutoff=0.8)
    if len(matches) != 1:
        return None
    for known in tool_schemas:
        if _name_key(known) == matches[0]:
            return known
    return None


def _normalize_call_name(name: str, tool_schemas: dict[str, dict[str, Any]] | None) -> str:
    if not tool_schemas or name in tool_schemas:
        return name
    folded = {known.casefold(): known for known in tool_schemas}
    hit = folded.get(name.casefold())
    if hit is not None:
        return hit
    compact = {_name_key(known): known for known in tool_schemas}
    hit = compact.get(_name_key(name))
    if hit is not None:
        return hit
    return _resolve_alias(name, tool_schemas) or _fuzzy_known_name(name, tool_schemas) or name


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
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except ValueError:
                params = None
        prop_types: dict[str, Any] = {}
        properties = params.get("properties") if isinstance(params, dict) else None
        if isinstance(properties, dict):
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
        aliases = fn.get("aliases")
        if isinstance(aliases, (list, tuple)):
            cleaned = [a for a in aliases if isinstance(a, str) and a]
            if cleaned:
                prop_types["_aliases"] = cleaned
        result[name] = prop_types
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
    for match in _XML_ELEMENT.finditer(body):
        key = match.group(1).strip()
        if key.lower() in _XML_SKIP_ELEMENTS:
            continue
        _xml_set_param(params, key, _xml_value(match.group(3), (param_types or {}).get(key)))
    if params:
        if len(params) == 1:
            for key in _ARGS_ALIASES:
                if key in params and isinstance(params[key], dict):
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
        rest = text[content_start:]
        close = _XML_WRAPPER_CLOSE_RE.search(rest)
        if close is None:
            close = _XML_WRAPPER_OPEN.search(rest)
        end = length if close is None else content_start + close.start()
        yield match.start(), content_start, end, text[content_start:end]
        pos = max(match.end(), end)


def _parse_xml_tool_calls(text: str, tool_schemas: dict[str, dict[str, Any]] | None = None) -> tuple[list[ToolCall] | None, str]:
    calls: list[ToolCall] = []
    mask = bytearray(len(text))
    consumed: list[tuple[int, int]] = []

    def blank(start: int, end: int) -> None:
        mask[start:end] = b" " * (end - start)

    for match in _XML_TOOL_ELEMENT_RE.finditer(text):
        start, end = match.span()
        attrs_text = match.group(1)
        body = match.group(2)
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
        consumed.append((start, end))
    for match in _XML_TOOL_SELFCLOSE_RE.finditer(text):
        start, end = match.span()
        if any(s <= start and end <= e for s, e in consumed):
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
        consumed.append((start, end))
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
            consumed.append((start, end))
    for start, content_start, end, inner in _iter_xml_call_wrappers(text):
        if any(s <= start and end <= e for s, e in consumed):
            continue
        stripped_inner = inner.strip()
        if stripped_inner.startswith("["):
            array_calls = _parse_bare_array_calls(stripped_inner)
            if array_calls:
                calls.extend(array_calls)
                blank(start, end)
                consumed.append((start, end))
                continue
        json_parsed = _extract_json_object(stripped_inner)
        if json_parsed is not None:
            extracted = _extract_calls(json_parsed[0])
            if extracted:
                calls.extend(extracted)
                blank(start, end)
                consumed.append((start, end))
                continue
        elements = list(_XML_ELEMENT.finditer(inner))
        top_level = [m for m in elements if not any(o is not m and o.start() < m.start() and o.end() > m.end() for o in elements)]
        pending_name: str | None = None
        block_calls = 0
        for element in top_level:
            element_name = element.group(1).strip().lower()
            if element_name in _XML_SKIP_ELEMENTS:
                continue
            element_start = content_start + element.start()
            element_end = content_start + element.end()
            if any(s <= element_start and element_end <= e for s, e in consumed):
                continue
            if element_name == "name":
                raw = _unescape_xml(element.group(3).strip())
                if raw:
                    pending_name = raw
                continue
            if element_name in _ARGS_ALIASES:
                container = _xml_invoke_arguments(element.group(3), None)
                if isinstance(container, dict) and pending_name:
                    calls.append(ToolCall.create(pending_name, container))
                    consumed.append((element_start, element_end))
                    block_calls += 1
                    pending_name = None
                continue
            raw_name = element.group(1).strip()
            param_types = _schema_for_name(tool_schemas, raw_name)
            arguments = _xml_tag_attrs(element.group(2), param_types)
            arguments.update(_xml_invoke_arguments(element.group(3), param_types) or {})
            if param_types is None and isinstance(arguments.get("name"), str) and arguments["name"].strip():
                raw_name = arguments.pop("name")
                param_types = _schema_for_name(tool_schemas, raw_name)
            elif param_types is None and raw_name.casefold() in _XML_GENERIC_TOOL_TAGS:
                continue
            if not arguments and param_types is None:
                continue
            calls.append(ToolCall.create(raw_name, arguments))
            consumed.append((element_start, element_end))
            block_calls += 1
        for element in _XML_SELFCLOSE.finditer(inner):
            element_name = element.group(1).strip().lower()
            if element_name in _XML_SKIP_ELEMENTS:
                continue
            element_start = content_start + element.start()
            element_end = content_start + element.end()
            if any(s <= element_start and element_end <= e for s, e in consumed):
                continue
            tool_name = element.group(1).strip()
            param_types = _schema_for_name(tool_schemas, tool_name)
            arguments = _xml_tag_attrs(element.group(2), param_types)
            if not arguments and param_types is None:
                continue
            calls.append(ToolCall.create(tool_name, arguments))
            consumed.append((element_start, element_end))
            block_calls += 1
        if block_calls:
            blank(start, end)
            consumed.append((start, end))
    for tool_name, param_types in (tool_schemas or {}).items():
        escaped = re.escape(tool_name)
        open_pattern = re.compile(
            rf"<{escaped}(?=[\s/>])([^>]*?)>(.*?)</{escaped}>",
            re.DOTALL | re.IGNORECASE,
        )
        selfclose_pattern = re.compile(rf"<{escaped}(?=[\s/>])([^>]*?)/>", re.DOTALL | re.IGNORECASE)
        for match in open_pattern.finditer(text):
            start, end = match.span()
            if any(s <= start and end <= e for s, e in consumed):
                continue
            merged = _xml_tag_attrs(match.group(1), param_types)
            merged.update(_xml_invoke_arguments(match.group(2), param_types) or {})
            calls.append(ToolCall.create(tool_name, merged))
            consumed.append((start, end))
            blank(start, end)
        for match in selfclose_pattern.finditer(text):
            start, end = match.span()
            if any(s <= start and end <= e for s, e in consumed):
                continue
            arguments = _xml_tag_attrs(match.group(1), param_types)
            calls.append(ToolCall.create(tool_name, arguments))
            consumed.append((start, end))
            blank(start, end)

    def _bare_eligible(match: re.Match) -> bool:
        name = match.group(1).strip().lower()
        return name not in _XML_SKIP_ELEMENTS and name not in _XML_HTML_TAGS

    bare_candidates: list[tuple[int, int, bool, re.Match]] = []
    bare_candidates.extend((m.start(), m.end(), False, m) for m in _XML_ELEMENT.finditer(text) if _bare_eligible(m))
    bare_candidates.extend((m.start(), m.end(), True, m) for m in _XML_SELFCLOSE.finditer(text) if _bare_eligible(m))
    bare_candidates.sort(key=lambda item: (item[0], -item[1]))
    for start, end, self_closed, match in bare_candidates:
        if any(s <= start and end <= e for s, e, _, _ in bare_candidates if (s, e) != (start, end)):
            continue
        if any(s <= start and end <= e for s, e in consumed):
            continue
        raw_name = match.group(1).strip()
        param_types = _schema_for_name(tool_schemas, raw_name)
        if self_closed:
            arguments = _xml_tag_attrs(match.group(2), param_types)
        else:
            arguments = _xml_tag_attrs(match.group(2), param_types)
            arguments.update(_xml_invoke_arguments(match.group(3), param_types, False) or {})
        if param_types is None and isinstance(arguments.get("name"), str) and arguments["name"].strip():
            raw_name = arguments.pop("name")
            param_types = _schema_for_name(tool_schemas, raw_name)
        elif param_types is None and raw_name.casefold() in _XML_GENERIC_TOOL_TAGS:
            continue
        if not arguments and param_types is None:
            continue
        calls.append(ToolCall.create(raw_name, arguments))
        consumed.append((start, end))
        blank(start, end)
    if not calls:
        return None, ""
    remainder = "".join(text[i] if mask[i] == 0 else " " for i in range(len(text)))
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
                        yield obj, start, end
                    except ValueError:
                        pass
                    break
            end += 1
        i = end + 1


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            current.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            current.append(ch)
        elif ch in "{([":
            depth += 1
            current.append(ch)
        elif ch in "})]":
            depth -= 1
            current.append(ch)
        elif ch == delimiter and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _python_value(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith(("{", "[")):
        try:
            return _loads_lenient(stripped)
        except ValueError:
            return stripped
    if stripped[0] in ("'", '"'):
        if stripped[-1] != stripped[0]:
            return stripped
        if stripped[0] == '"':
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return stripped[1:-1]
        normalized = _normalize_single_quotes(stripped)
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            return stripped[1:-1].replace("\\'", "'")
    low = stripped.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return stripped


def _parse_python_call_args(text: str) -> dict[str, Any] | None:
    args: dict[str, Any] = {}
    for raw_part in _split_top_level(text):
        part = raw_part.strip()
        if not part:
            continue
        eq = part.find("=")
        if eq <= 0:
            return None
        key = part[:eq].strip()
        if not _PYTHON_KEY_RE.fullmatch(key):
            return None
        args[key] = _python_value(part[eq + 1 :])
    return args


def _python_call_match(text: str) -> tuple[str, str] | None:
    match = _PYTHON_CALL_RE.match(text)
    if match is None:
        return None
    depth = 1
    in_string = False
    escaped = False
    for i in range(match.end(), len(text)):
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
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                if text[i + 1 :].strip():
                    return None
                return match.group(1), text[match.end() : i]
    return None


def _parse_python_calls(text: str) -> tuple[list[ToolCall], str] | None:
    stripped = text.strip()
    if not stripped:
        return None
    single = _python_call_match(stripped)
    if single is not None:
        name, args_text = single
        if not args_text.strip():
            return [ToolCall.create(name, {})], ""
        args = _parse_python_call_args(args_text)
        if args is not None:
            return [ToolCall.create(name, args)], ""
        return None
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    first = -1
    for i, line in enumerate(lines):
        if _python_call_match(line) is not None:
            first = i
            break
    if first < 0:
        return None
    calls: list[ToolCall] = []
    for line in lines[first:]:
        parsed = _python_call_match(line)
        if parsed is None:
            return None
        name, args_text = parsed
        if not args_text.strip():
            calls.append(ToolCall.create(name, {}))
        else:
            args = _parse_python_call_args(args_text)
            if args is None:
                return None
            calls.append(ToolCall.create(name, args))
    return calls, " ".join(lines[:first])


def _yaml_key_value(line: str) -> tuple[str | None, str]:
    match = re.match(r"([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$", line)
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
        except ValueError:
            return value
    if value[0] in ("'", '"'):
        if value[-1] != value[0]:
            return value
        if value[0] == '"':
            try:
                return json.loads(value)
            except json.JSONDecodeError:
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
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_yaml_calls(text: str) -> list[ToolCall] | None:
    lines = [line.rstrip() for line in text.splitlines()]
    if not lines:
        return None
    root = lines[0].strip()
    root_match = re.match(r"^tool_calls\s*:?\s*(.*)$", root, re.IGNORECASE)
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
    python_calls = _parse_python_calls(stripped)
    if python_calls is not None:
        py_calls, py_wrapper = python_calls
        if report is not None:
            report["strategies"].append("python_call")
        return py_calls, py_wrapper
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
        wrapper = "".join((stripped[i] if removed[i] == 0 else " ") for i in range(len(stripped)))
        if report is not None:
            report["strategies"].append("json_in_prose")
        return calls, " ".join(wrapper.split())
    return None


def parse_tool_calls(text: str, tool_schemas: dict[str, dict[str, Any]] | None = None) -> tuple[list[ToolCall], str] | None:
    result = _parse_tool_calls_impl(text, tool_schemas, None)
    if result is None:
        return None
    calls, wrapper = result
    return [ToolCall(call.id, _normalize_call_name(call.name, tool_schemas), call.arguments) for call in calls], wrapper


def parse_tool_calls_debug(text: str, tool_schemas: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "text": text,
        "stripped": _strip_fences(_strip_dsml(text)),
        "parsed": False,
        "strategies": [],
        "calls": [],
        "renamed": [],
        "wrapper": "",
        "unrecognized": _strip_fences(_strip_dsml(text)),
    }
    result = _parse_tool_calls_impl(text, tool_schemas, report)
    if result is not None:
        calls, wrapper = result
        renamed = [{"from": call.name, "to": _normalize_call_name(call.name, tool_schemas)} for call in calls]
        renamed = [item for item in renamed if item["from"] != item["to"]]
        report["parsed"] = True
        report["renamed"] = renamed
        report["calls"] = [
            {
                "id": call.id,
                "name": _normalize_call_name(call.name, tool_schemas),
                "arguments": call.arguments,
            }
            for call in calls
        ]
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
