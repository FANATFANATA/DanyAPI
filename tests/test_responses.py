import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import danyapi.api.openai as openai_mod
from danyapi.api import responses as resp
from danyapi.api.openai import app
from danyapi.store import JsonStore

OK_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"WIP","fragments":[{"id":2,"type":"RESPONSE","content":"Hi"}]}}}\n'
    "\n"
    'data: {"p":"response/status","o":"SET","v":"FINISHED"}\n'
    "\n"
)

TOOL_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"WIP","fragments":[{"id":2,"type":"RESPONSE","content":"<tool_calls>\\n'
    '<invoke name=\\"get_weather\\">\\n<parameter name=\\"city\\">Paris</parameter>\\n</invoke>\\n</tool_calls>"}]}}}\n'
    "\n"
    'data: {"p":"response/status","o":"SET","v":"FINISHED"}\n'
    "\n"
)


class FakeSession:
    def __init__(self, sid="c1", last_message_id=None):
        self.id = sid
        self.last_message_id = last_message_id
        self.accumulated_tokens = 0


class FakeResp:
    def __init__(self, body=None, sse_text=None, status=200, content_type="text/event-stream; charset=utf-8"):
        self.status_code = status
        self.headers = {"content-type": content_type}
        self._b = (sse_text if sse_text is not None else (body or "")).encode()

    async def aiter_bytes(self):
        yield self._b

    async def aclose(self):
        pass

    async def aread(self):
        return self._b


class FakeAccount:
    def __init__(self, sse_list=None):
        self.index = 0
        self.broken = False
        from unittest.mock import AsyncMock, MagicMock

        self.client = MagicMock()
        self.client.completion = AsyncMock(side_effect=[FakeResp(sse_text=s) for s in (sse_list or [OK_SSE])])
        self.client.create_pow_challenge = AsyncMock(return_value={})
        self.pow = MagicMock()
        self.pow.make_header = AsyncMock(return_value={})
        self.pow_upload = MagicMock()
        self.pow_upload.make_header = AsyncMock(return_value={})
        self.sem = asyncio.Semaphore(1)
        self.sessions = MagicMock()
        self.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
        self.sessions.touch_last_message = MagicMock()
        self.sessions.forget = MagicMock()

    def mark_broken(self):
        self.broken = True


def make_pool(sse_list=None):
    from unittest.mock import AsyncMock, MagicMock

    acct = FakeAccount(sse_list)
    pool = MagicMock()
    pool.acquire = AsyncMock(return_value=(acct, None))
    return pool, acct


@pytest.fixture(autouse=True)
def clean_state():
    saved = (
        getattr(app.state, "pool", None),
        getattr(app.state, "qwen_pool", None),
        getattr(app.state, "qwen_models", None),
        getattr(app.state, "responses_store", None),
    )
    app.state.pool = None
    app.state.qwen_pool = None
    app.state.qwen_models = []
    app.state.responses_store = JsonStore("responses-test", None)
    yield
    (
        app.state.pool,
        app.state.qwen_pool,
        app.state.qwen_models,
        app.state.responses_store,
    ) = saved


@pytest.fixture(autouse=True)
def zero_backoff():
    orig = openai_mod.RETRY_BACKOFF_SEC
    openai_mod.RETRY_BACKOFF_SEC = 0.0
    yield
    openai_mod.RETRY_BACKOFF_SEC = orig


async def _agen(items):
    for item in items:
        yield item


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


def test_normalize_input_string():
    assert resp.normalize_input("hello") == [{"role": "user", "content": "hello"}]


def test_normalize_input_message_parts():
    messages = resp.normalize_input(
        [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                ],
            }
        ]
    )
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0] == {"type": "text", "text": "look"}
    assert messages[0]["content"][1] == {"type": "image_url", "image_url": "data:image/png;base64,AAAA"}


def test_normalize_input_developer_role():
    assert resp.normalize_input([{"role": "developer", "content": "be nice"}]) == [{"role": "system", "content": "be nice"}]


def test_normalize_input_function_call():
    messages = resp.normalize_input([{"type": "function_call", "call_id": "call_1", "name": "f", "arguments": '{"a":1}'}])
    assert messages[0]["tool_calls"][0]["id"] == "call_1"
    assert messages[0]["tool_calls"][0]["function"]["name"] == "f"


def test_normalize_input_function_call_output():
    messages = resp.normalize_input([{"type": "function_call_output", "call_id": "call_1", "output": "42"}])
    assert messages == [{"role": "tool", "tool_call_id": "call_1", "content": "42"}]


def test_normalize_input_unsupported_role():
    with pytest.raises(resp.ResponsesInputError):
        resp.normalize_input([{"role": "nobody", "content": "x"}])


def test_convert_tools_flat():
    tools = resp.convert_tools([{"type": "function", "name": "f", "description": "d", "parameters": {"type": "object"}}])
    assert tools == [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]


def test_convert_tools_nested_kept():
    nested = {"type": "function", "function": {"name": "f"}}
    assert resp.convert_tools([nested]) == [nested]


def test_convert_tool_choice():
    assert resp.convert_tool_choice("auto") == "auto"
    assert resp.convert_tool_choice({"type": "required"}) == "required"
    assert resp.convert_tool_choice({"type": "function", "name": "f"}) == {"type": "function", "function": {"name": "f"}}
    assert resp.convert_tool_choice(None) is None


def test_extract_response_format():
    assert resp.extract_response_format(None, None) is None
    assert resp.extract_response_format({"format": {"type": "text"}}, None) is None
    assert resp.extract_response_format({"format": {"type": "json_object"}}, None) == {"type": "json_object"}
    fmt = resp.extract_response_format({"format": {"type": "json_schema", "name": "x", "schema": {"type": "object"}}}, None)
    assert fmt == {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}


def test_response_text_format():
    assert resp.response_text_format({"format": {"type": "json_object"}}) == {"type": "json_object"}
    assert resp.response_text_format("json_object") == {"type": "json_object"}
    assert resp.response_text_format(None) is None


def test_output_items_from_message_text():
    items = resp.output_items_from_message({"role": "assistant", "content": "hi"})
    assert items[0]["type"] == "message"
    assert items[0]["content"][0]["text"] == "hi"


def test_output_items_from_message_tools():
    message = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "function": {"name": "f", "arguments": "{}"}}]}
    items = resp.output_items_from_message(message)
    assert len(items) == 1
    assert items[0]["type"] == "function_call"
    assert items[0]["name"] == "f"


def test_messages_from_output_roundtrip():
    output = resp.output_items_from_message({"role": "assistant", "content": "hi"})
    assert resp.messages_from_output(output) == [{"role": "assistant", "content": "hi"}]
    output = resp.output_items_from_message(
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "function": {"name": "f", "arguments": "{}"}}]}
    )
    messages = resp.messages_from_output(output)
    assert messages[-1]["tool_calls"][0]["id"] == "call_1"


def test_build_response_object_shape():
    info = resp.RequestInfo(model="deepseek-v4.1-flash", instructions="sys")
    obj = resp.build_response_object(info, "resp_1", 123, output=[], status="in_progress")
    assert obj["id"] == "resp_1"
    assert obj["object"] == "response"
    assert obj["status"] == "in_progress"
    assert obj["usage"] is None
    assert obj["text"] == {"format": {"type": "text"}}


def test_response_from_chat_completed():
    info = resp.RequestInfo(model="m")
    chat = {
        "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    obj = resp.response_from_chat(chat, info, "resp_1", 1)
    assert obj["status"] == "completed"
    assert obj["output"][0]["content"][0]["text"] == "Hi"
    assert obj["usage"]["input_tokens"] == 1
    assert obj["usage"]["output_tokens"] == 2


def test_response_from_chat_incomplete_length():
    info = resp.RequestInfo(model="m")
    chat = {"choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "length"}]}
    obj = resp.response_from_chat(chat, info, "resp_1", 1)
    assert obj["status"] == "incomplete"
    assert obj["incomplete_details"] == {"reason": "max_output_tokens"}


async def test_translate_stream_text():
    info = resp.RequestInfo(model="m")
    chat_stream = _agen(
        [
            'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
            'data: {"id":"x","choices":[{"index":0,"delta":{"content":"He"},"finish_reason":null}]}\n\n',
            'data: {"id":"x","choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":null}]}\n\n',
            'data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n',
            "data: [DONE]\n\n",
        ]
    )
    lines = await _collect(resp.translate_stream(chat_stream, info, "resp_1", 1))
    joined = "".join(lines)
    assert "event: response.created" in joined
    assert "event: response.output_text.delta" in joined
    assert '"delta": "He"' in joined
    assert "event: response.output_text.done" in joined
    assert "event: response.completed" in joined
    assert '"input_tokens": 1' in joined


async def test_translate_stream_tools():
    info = resp.RequestInfo(model="m")
    chat_stream = _agen(
        [
            'data: {"id":"x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",'
            '"function":{"name":"f","arguments":""}}]},"finish_reason":null}]}\n\n',
            'data: {"id":"x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\":1}"}}]},"finish_reason":null}]}\n\n',
            'data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
        ]
    )
    joined = "".join(await _collect(resp.translate_stream(chat_stream, info, "resp_1", 1)))
    assert "event: response.function_call_arguments.delta" in joined
    assert "response.function_call_arguments.done" in joined
    assert '"name": "f"' in joined


async def test_translate_stream_error():
    info = resp.RequestInfo(model="m")
    chat_stream = _agen(['data: {"id":"x","error":{"message":"boom","finish_reason":"server_busy"},"choices":[]}\n\n'])
    joined = "".join(await _collect(resp.translate_stream(chat_stream, info, "resp_1", 1)))
    assert "event: error" in joined
    assert "boom" in joined
    assert "response.completed" not in joined


def test_endpoint_requires_known_model():
    client = TestClient(app)
    resp_obj = client.post("/v1/responses", json={"model": "gpt-4", "input": "hi"})
    client.close()
    assert resp_obj.status_code == 404


def test_endpoint_provider_not_configured():
    client = TestClient(app)
    resp_obj = client.post("/v1/responses", json={"model": "deepseek-v4.1-flash", "input": "hi"})
    client.close()
    assert resp_obj.status_code == 503


def test_endpoint_bad_input():
    pool, _ = make_pool()
    app.state.pool = pool
    client = TestClient(app)
    resp_obj = client.post("/v1/responses", json={"model": "deepseek-v4.1-flash", "input": 123})
    client.close()
    assert resp_obj.status_code == 400


def test_endpoint_non_stream_and_store():
    pool, _ = make_pool()
    app.state.pool = pool
    client = TestClient(app)
    resp_obj = client.post("/v1/responses", json={"model": "deepseek-v4.1-flash", "input": "hi", "instructions": "be brief"})
    client.close()
    assert resp_obj.status_code == 200
    data = resp_obj.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"
    assert data["output"][0]["content"][0]["text"] == "Hi"
    assert data["instructions"] == "be brief"

    client = TestClient(app)
    fetched = client.get(f"/v1/responses/{data['id']}")
    client.close()
    assert fetched.status_code == 200
    assert fetched.json()["id"] == data["id"]

    client = TestClient(app)
    deleted = client.delete(f"/v1/responses/{data['id']}")
    client.close()
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True

    client = TestClient(app)
    missing = client.get(f"/v1/responses/{data['id']}")
    client.close()
    assert missing.status_code == 404


def test_endpoint_previous_response_id():
    pool, _ = make_pool([OK_SSE, OK_SSE])
    app.state.pool = pool
    client = TestClient(app)
    first = client.post("/v1/responses", json={"model": "deepseek-v4.1-flash", "input": "hi"}).json()
    client.close()

    client = TestClient(app)
    second = client.post("/v1/responses", json={"model": "deepseek-v4.1-flash", "input": "again", "previous_response_id": first["id"]})
    client.close()
    assert second.status_code == 200
    record = app.state.responses_store.get(second.json()["id"])
    assert any(message.get("content") == "again" for message in record["conversation"])
    assert any(message.get("role") == "assistant" and message.get("content") == "Hi" for message in record["conversation"])


def test_endpoint_previous_response_missing():
    pool, _ = make_pool()
    app.state.pool = pool
    client = TestClient(app)
    resp_obj = client.post("/v1/responses", json={"model": "deepseek-v4.1-flash", "input": "hi", "previous_response_id": "resp_nope"})
    client.close()
    assert resp_obj.status_code == 404


def test_endpoint_stream():
    pool, _ = make_pool()
    app.state.pool = pool
    client = TestClient(app)
    resp = client.post("/v1/responses", json={"model": "deepseek-v4.1-flash", "input": "hi", "stream": True})
    client.close()
    assert resp.status_code == 200
    assert "event: response.created" in resp.text
    assert "event: response.output_text.delta" in resp.text
    assert "event: response.completed" in resp.text
    assert "Hi" in resp.text
    record_id = resp.text.split("response.completed", 1)[1]
    response_ids = [
        line.split('"id": "', 1)[1].split('"', 1)[0] for line in resp.text.splitlines() if line.startswith("data:") and '"object": "response"' in line
    ]
    assert response_ids
    stored = app.state.responses_store.get(response_ids[-1])
    assert stored is not None
    assert "conversation" in stored
    assert record_id


def test_endpoint_stream_with_tools():
    pool, _ = make_pool([TOOL_SSE])
    app.state.pool = pool
    client = TestClient(app)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "deepseek-v4.1-flash",
            "input": "weather in paris",
            "stream": True,
            "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        },
    )
    client.close()
    assert resp.status_code == 200
    assert "response.function_call_arguments.delta" in resp.text
    assert "get_weather" in resp.text


def test_store_disabled_not_persisted():
    pool, _ = make_pool()
    app.state.pool = pool
    client = TestClient(app)
    data = client.post("/v1/responses", json={"model": "deepseek-v4.1-flash", "input": "hi", "store": False}).json()
    client.close()
    assert app.state.responses_store.get(data["id"]) is None


def test_conversation_stored_as_json_safe():
    pool, _ = make_pool()
    app.state.pool = pool
    client = TestClient(app)
    data = client.post("/v1/responses", json={"model": "deepseek-v4.1-flash", "input": "hi"}).json()
    client.close()
    record = app.state.responses_store.get(data["id"])
    json.dumps(record)
