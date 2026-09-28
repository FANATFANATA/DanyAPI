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
    assert "function_call" not in out[0]
    assert out[0]["role"] == "user"


@pytest.mark.asyncio
async def test_message_level_function_call_is_never_set():
    stub = _Stub()
    out, _specs, _call = await gm.build_messages(
        stub,
        [ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="hey")],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        tool_choice="auto",
    )
    assert not any("function_call" in m for m in out)


@pytest.mark.asyncio
async def test_tool_without_parameters_gets_empty_object_schema():
    stub = _Stub()
    _out, specs, _call = await gm.build_messages(
        stub,
        [ChatMessage(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "ping"}}],
    )
    assert specs == [{"name": "ping", "parameters": {"type": "object", "properties": {}}}]


@pytest.mark.asyncio
async def test_tool_parameters_stay_a_json_object_not_a_string():
    stub = _Stub()
    _out, specs, _call = await gm.build_messages(
        stub,
        [ChatMessage(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}}] ,
    )
    assert isinstance(specs[0]["parameters"], dict)


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
    assert out.index(assistant[0]) < out.index(functions[0])


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
async def test_leading_non_user_messages_are_dropped():
    stub = _Stub()
    out, _specs, _call = await gm.build_messages(
        stub,
        [
            ChatMessage(role="function", name="f", content="{}"),
            ChatMessage(role="assistant", content="thinking out loud"),
            ChatMessage(role="user", content="hi"),
        ],
    )
    assert [m["role"] for m in out] == ["user"]


@pytest.mark.asyncio
async def test_history_starting_with_assistant_is_normalized():
    stub = _Stub()
    out, _specs, _call = await gm.build_messages(
        stub,
        [ChatMessage(role="assistant", content="only assistant"), ChatMessage(role="user", content="hi")],
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


def test_translate_message_serializes_object_arguments():
    from danyapi.gigachat.api import _translate_message

    message = _translate_message(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "function_call": {"id": "fc-1", "name": "get_weather", "arguments": {"city": "Moscow"}},
            },
            "finish_reason": "function_call",
        }
    )
    call = message["tool_calls"][0]
    assert call["id"] == "fc-1"
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert call["function"]["arguments"] == '{"city": "Moscow"}'


def test_translate_message_synthesizes_tool_call_id_when_absent():
    from danyapi.gigachat.api import _translate_message

    message = _translate_message({"message": {"content": "", "function_call": {"name": "f", "arguments": "{}"}}})
    assert message["tool_calls"][0]["id"].startswith("call_")


@pytest.mark.parametrize(("upstream", "expected"), [(404, 404), (401, 401), (429, 429), (422, 400), (500, 502)])
def test_status_mapping(upstream, expected):
    from danyapi.gigachat.api import _detail_for, _status_for
    from danyapi.gigachat.client import GigaChatError

    assert _status_for(GigaChatError(upstream, "boom")) == expected
    assert "boom" in _detail_for(GigaChatError(upstream, "boom"))


def test_gigachat_arguments_handles_all_shapes():
    assert gm._gigachat_arguments({"a": 1}) == {"a": 1}
    assert gm._gigachat_arguments('{"a": 1}') == {"a": 1}
    assert gm._gigachat_arguments(None) == {}
    assert gm._gigachat_arguments("") == {}
    assert gm._gigachat_arguments("not json") == {"input": "not json"}
    assert gm._gigachat_arguments("[1,2]") == {"input": [1, 2]}


@pytest.mark.asyncio
async def test_outgoing_tool_call_arguments_are_an_object_not_a_string():
    stub = _Stub()
    out, _specs, _call = await gm.build_messages(
        stub,
        [
            ChatMessage(role="user", content="hi"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"city": "Moscow"}'}}],
            ),
            ChatMessage(role="tool", tool_call_id="c1", name="f", content="{}"),
        ],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
    )
    assistant = [m for m in out if isinstance(m.get("function_call"), dict)]
    assert isinstance(assistant[0]["function_call"]["arguments"], dict)
    assert assistant[0]["function_call"]["arguments"] == {"city": "Moscow"}


def test_image_capability_error_is_translated():
    from danyapi.gigachat.api import _detail_for
    from danyapi.gigachat.client import GigaChatError

    detail = _detail_for(GigaChatError(400, "Model does not support image"))
    assert "Pro" in detail
    assert "GigaChat-2-Pro" in detail
    assert "does not support image" not in detail
    assert _detail_for(GigaChatError(400, "some other problem")) == "GigaChat error: some other problem"


def test_multipart_requests_omit_json_content_type():
    import inspect

    from danyapi.gigachat.client import GigaChatClient

    sig = inspect.signature(GigaChatClient._api_headers)
    assert "json_body" in sig.parameters
    client = GigaChatClient(key="x" * 8)
    assert "Content-Type" not in client._api_headers("t", json_body=False)
    assert client._api_headers("t")["Content-Type"] == "application/json"
