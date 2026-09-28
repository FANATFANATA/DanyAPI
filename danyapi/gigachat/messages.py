from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import httpx
from fastapi import HTTPException

from ..api.attachments import _split_data_uri
from .client import IMAGE_MIME_TYPES, GigaChatClient

log = logging.getLogger("danyapi.gigachat.messages")

MAX_IMAGES_PER_MESSAGE = 1
MAX_IMAGES_PER_REQUEST = 10
MAX_IMAGE_BYTES = 15 * 1024 * 1024
REMOTE_SCHEMES = ("http://", "https://")

FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "function_call": "tool_calls",
    "blacklist": "content_filter",
    "error": "stop",
}


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in ("text", "input_text"):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _image_uris(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    uris: list[str] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") not in ("image_url", "input_image"):
            continue
        image_url = item.get("image_url")
        if isinstance(image_url, str):
            uris.append(image_url)
        elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
            uris.append(image_url["url"])
    return uris


def _extension_for(content_type: str) -> str:
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/tiff": ".tiff",
        "image/bmp": ".bmp",
    }
    return mapping.get(content_type, ".bin")


def _gigachat_arguments(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except ValueError:
            return {"input": value}
        if isinstance(parsed, dict):
            return parsed
        return {"input": parsed}
    if value is None:
        return {}
    return {"input": value}


def _tool_call_name(tool_call: dict) -> tuple[str, Any]:
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = tool_call.get("name")
        arguments = tool_call.get("arguments")
    if not isinstance(name, str) or not name:
        name = ""
    return name, _gigachat_arguments(arguments)


def _as_function_result(content: Any) -> str:
    text = _text_of(content).strip()
    if not text:
        return "{}"
    return text


def _system_text(content: Any) -> str:
    return _text_of(content).strip()


def _developer_to_system(role: str) -> str:
    return "system" if role == "developer" else role


EMPTY_PARAMETERS = {"type": "object", "properties": {}}


def _build_functions(tools: Any, functions: Any) -> list[dict]:
    specs: list[dict] = []

    def add(name: Any, description: Any, parameters: Any) -> None:
        if not isinstance(name, str) or not name:
            return
        if isinstance(description, str) and description:
            spec: dict[str, Any] = {"name": name, "description": description}
        else:
            spec = {"name": name}
        if isinstance(parameters, dict) and parameters:
            spec["parameters"] = parameters
        else:
            spec["parameters"] = dict(EMPTY_PARAMETERS)
        if spec not in specs:
            specs.append(spec)

    for item in tools or []:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        add(function.get("name"), function.get("description"), function.get("parameters"))
    for item in functions or []:
        if not isinstance(item, dict):
            continue
        add(item.get("name"), item.get("description"), item.get("parameters"))
    return specs


def _function_call_value(tool_choice: Any, function_call: Any) -> str | dict | None:
    choice = function_call if function_call is not None else tool_choice
    if choice is None:
        return "auto"
    if isinstance(choice, str):
        if choice in ("none", "auto", "required"):
            return "none" if choice == "required" else choice
        return {"name": choice}
    if isinstance(choice, dict):
        function = choice.get("function")
        name = function.get("name") if isinstance(function, dict) else choice.get("name")
        if isinstance(name, str) and name:
            return {"name": name}
    return "auto"


async def _upload_images(
    client: GigaChatClient,
    uris: list[str],
    budget: list[int],
) -> list[str]:
    if not uris:
        return []
    if len(uris) > MAX_IMAGES_PER_MESSAGE:
        raise HTTPException(400, f"gigachat accepts at most {MAX_IMAGES_PER_MESSAGE} image per message")
    if budget[0] + len(uris) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(400, f"gigachat accepts at most {MAX_IMAGES_PER_REQUEST} images per request")
    file_ids: list[str] = []
    for uri in uris:
        if uri.startswith("data:"):
            content_type, data = _split_data_uri(uri)
        elif uri.startswith(REMOTE_SCHEMES):
            try:
                resp = await client.http.get(uri, timeout=30.0)
                resp.raise_for_status()
                data = resp.content
            except httpx.HTTPError as exc:
                raise HTTPException(400, f"gigachat could not fetch remote image: {exc}") from exc
            content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        else:
            raise HTTPException(400, "gigachat images must be data URIs or http(s) URLs")
        if content_type not in IMAGE_MIME_TYPES:
            raise HTTPException(400, f"gigachat does not accept image type {content_type or 'unknown'}")
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(400, "gigachat accepts images up to 15 MB")
        name = f"image{_extension_for(content_type)}"
        file_ids.append(await client.upload_file(name, data, content_type))
    budget[0] += len(uris)
    return file_ids


async def build_messages(
    client: GigaChatClient,
    messages: list[Any],
    tools: Any = None,
    tool_choice: Any = None,
    functions: Any = None,
    function_call: Any = None,
) -> tuple[list[dict], list[dict], str | dict | None]:
    system_parts: list[str] = []
    body: list[dict] = []
    id_to_name: dict[str, str] = {}
    image_budget = [0]
    pending_assistant: dict | None = None

    def release_pending() -> None:
        nonlocal pending_assistant
        if pending_assistant is not None:
            body.append(pending_assistant)
            pending_assistant = None

    for message in messages:
        role = _developer_to_system(getattr(message, "role", "user") or "user")
        content = getattr(message, "content", "")
        if role == "system":
            text = _system_text(content)
            if text:
                system_parts.append(text)
            continue
        if role in ("tool", "function"):
            name = getattr(message, "name", None)
            call_id = getattr(message, "tool_call_id", None)
            if not isinstance(name, str) or not name:
                name = id_to_name.get(call_id or "", "")
            if not name:
                continue
            if pending_assistant is not None and pending_assistant["function_call"]["name"] == name:
                release_pending()
            body.append({"role": "function", "content": _as_function_result(content), "name": name})
            continue
        if role == "assistant":
            text = _text_of(content).strip()
            raw_calls = getattr(message, "tool_calls", None) or []
            call_name = ""
            call_args: dict = {}
            if raw_calls:
                first = raw_calls[0]
                if isinstance(first, dict):
                    call_name, call_args = _tool_call_name(first)
                    call_id = first.get("id")
                    if isinstance(call_id, str) and call_id:
                        id_to_name[call_id] = call_name
            release_pending()
            if call_name:
                pending_assistant = {
                    "role": "assistant",
                    "content": text,
                    "function_call": {"name": call_name, "arguments": call_args},
                    "functions_state_id": str(uuid.uuid4()),
                }
            else:
                body.append({"role": "assistant", "content": text})
            continue
        release_pending()
        uris = _image_uris(content)
        text = _text_of(content).strip()
        if not text and not uris:
            continue
        entry: dict[str, Any] = {"role": "user", "content": text}
        file_ids = await _upload_images(client, uris, image_budget)
        if file_ids:
            entry["attachments"] = file_ids
        body.append(entry)

    pending_assistant = None

    while body and body[0]["role"] != "user":
        body.pop(0)
    if not body:
        body.append({"role": "user", "content": "Hello"})

    specs = _build_functions(tools, functions)
    call_value = _function_call_value(tool_choice, function_call) if specs else None

    out: list[dict] = []
    if system_parts:
        out.append({"role": "system", "content": "\n\n".join(system_parts)})
    out.extend(body)

    if specs:
        return out, specs, call_value
    return out, [], None


def request_body(
    messages: list[dict],
    functions: list[dict],
    function_call: Any,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    response_format: Any,
) -> dict:
    body: dict[str, Any] = {"messages": messages}
    if functions:
        body["functions"] = functions
        if function_call is not None:
            body["function_call"] = function_call
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if max_tokens is not None and max_tokens > 0:
        body["max_tokens"] = max_tokens
    if isinstance(response_format, dict):
        body["response_format"] = response_format
    return body


def normalize_finish_reason(value: Any) -> str:
    if not isinstance(value, str):
        return "stop"
    return FINISH_REASON_MAP.get(value, "stop")


def normalize_usage(payload: Any) -> dict:
    if not isinstance(payload, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = int(payload.get("prompt_tokens") or 0)
    completion = int(payload.get("completion_tokens") or 0)
    total = int(payload.get("total_tokens") or 0)
    cached = int(payload.get("precached_prompt_tokens") or 0)
    if total <= 0:
        total = prompt + completion
    if total < prompt + completion:
        total = prompt + completion
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    return usage
