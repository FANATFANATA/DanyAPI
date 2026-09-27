from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

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
_XML_ATTR_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_.-]*)\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s\"'=<>`]+)", re.IGNORECASE)
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
