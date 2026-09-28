import base64

import httpx
import pytest

from danyapi.api.schemas import ChatMessage
from danyapi.gigachat import messages as gm

KEY = base64.b64encode(b"client-id:client-secret").decode()

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


class _Stub:
    def __init__(self) -> None:
        self.http = httpx.AsyncClient()
        self.uploads: list[tuple[str, bytes, str]] = []

    async def upload_file(self, filename: str, data: bytes, content_type: str, purpose: str = "general") -> str:
        self.uploads.append((filename, data, content_type))
        return f"file-{len(self.uploads)}"


@pytest.mark.asyncio
async def test_system_messages_merge_into_one_leading_message():
    stub = _Stub()
    out, specs, call = await gm.build_messages(
        stub,
        [
            ChatMessage(role="system", content="first"),
            ChatMessage(role="developer", content="second"),
            ChatMessage(role="user", content="hello"),
        ],
    )
    assert specs == []
    assert call is None
    assert out[0] == {"role": "system", "content": "first\n\nsecond"}
    assert out[1] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_tools_are_converted_to_gigachat_functions():
    stub = _Stub()
    out, specs, call = await gm.build_messages(
        stub,
        [ChatMessage(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "get_weather", "description": "d", "parameters": {"type": "object"}}}],
        tool_choice="auto",
    )
    assert specs == [{"name": "get_weather", "description": "d", "parameters": {"type": "object"}}]
    assert call == "auto"
    assert out[0]["function_call"] == "auto"
    assert out[0]["role"] == "user"


@pytest.mark.asyncio
async def test_tool_choice_object_resolves_to_name():
    stub = _Stub()
    _out, _specs, call = await gm.build_messages(
        stub,
        [ChatMessage(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "f"}}],
        tool_choice={"type": "function", "function": {"name": "f"}},
    )
    assert call == {"name": "f"}


@pytest.mark.asyncio
async def test_assistant_tool_call_waits_for_matching_function_message():
    stub = _Stub()
    out, _specs, _call = await gm.build_messages(
        stub,
        [
            ChatMessage(role="user", content="weather?"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Moscow"}'}}],
            ),
            ChatMessage(role="tool", tool_call_id="call_1", content='{"temp": 27}'),
        ],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )
    assistant = [m for m in out if isinstance(m.get("function_call"), dict)]
    assert len(assistant) == 1
    assert assistant[0]["function_call"]["name"] == "get_weather"
    assert assistant[0]["functions_state_id"]
    functions = [m for m in out if m["role"] == "function"]
    assert functions[0]["name"] == "get_weather"
    assert functions[0]["content"] == '{"temp": 27}'
    roles = [m["role"] for m in out if m["role"] != "user" or m.get("function_call") == "auto"]
    assert roles.index("assistant") < roles.index("function")


@pytest.mark.asyncio
async def test_dangling_tool_call_is_dropped():
    stub = _Stub()
    out, _specs, _call = await gm.build_messages(
        stub,
        [
            ChatMessage(role="user", content="weather?"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
            ),
        ],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )
    assert not [m for m in out if isinstance(m.get("function_call"), dict)]


@pytest.mark.asyncio
async def test_leading_function_message_is_removed():
    stub = _Stub()
    out, _specs, _call = await gm.build_messages(
        stub,
        [
            ChatMessage(role="function", name="f", content="{}"),
            ChatMessage(role="user", content="hi"),
        ],
    )
    assert out[0]["role"] == "user"


@pytest.mark.asyncio
async def test_image_data_uri_is_uploaded_and_attached():
    stub = _Stub()
    out, _specs, _call = await gm.build_messages(
        stub,
        [ChatMessage(role="user", content=[{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": PNG}}])],
    )
    assert out[0]["attachments"] == ["file-1"]
    assert out[0]["content"] == "what is this"
    assert stub.uploads[0][0] == "image.png"
    assert stub.uploads[0][2] == "image/png"


@pytest.mark.asyncio
async def test_unsupported_image_mime_is_rejected():
    from fastapi import HTTPException

    stub = _Stub()
    gif = "data:image/gif;base64,R0lGODlhAQABAAAAACw="
    with pytest.raises(HTTPException):
        await gm.build_messages(stub, [ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": gif}}])])


@pytest.mark.asyncio
async def test_too_many_images_in_one_message_is_rejected():
    from fastapi import HTTPException

    stub = _Stub()
    content = [
        {"type": "image_url", "image_url": {"url": PNG}},
        {"type": "image_url", "image_url": {"url": PNG}},
    ]
    with pytest.raises(HTTPException):
        await gm.build_messages(stub, [ChatMessage(role="user", content=content)])


@pytest.mark.asyncio
async def test_empty_message_list_gets_default_prompt():
    stub = _Stub()
    out, _specs, _call = await gm.build_messages(stub, [])
    assert out == [{"role": "user", "content": "Hello"}]
