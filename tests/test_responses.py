import asyncio
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import danyapi.api.deepseek as deepseek_mod
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


class FakeUpstream:
    def __init__(self, chunks, fail_at=None):
        self.chunks = list(chunks)
        self.fail_at = fail_at
        self.closed = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for index, chunk in enumerate(self.chunks):
            if self.fail_at is not None and index == self.fail_at:
                raise RuntimeError("upstream boom")
            yield chunk

    async def aclose(self):
        self.closed = True


def make_pool(sse_list=None):
    from unittest.mock import AsyncMock, MagicMock

    acct = FakeAccount(sse_list)
    pool = MagicMock()
    pool.acquire = AsyncMock(return_value=(acct, None))
    return pool, acct


def _frames(text: str) -> list[tuple[str, dict]]:
    frames: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if data:
            frames.append((event, json.loads(data)))
    return frames


def _named(frames: list[tuple[str, dict]], name: str) -> list[dict]:
    return [payload for event, payload in frames if event == name]


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
    deepseek_mod.RETRY_BACKOFF_SEC = 0.0
    yield
    deepseek_mod.RETRY_BACKOFF_SEC = orig


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


def test_response_from_chat_reasoning_tokens():
    info = resp.RequestInfo(model="m")
    chat = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hi",
                    "reasoning_content": "Let me think about this carefully first",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    obj = resp.response_from_chat(chat, info, "resp_1", 1)
    details = obj["usage"]["output_tokens_details"]
    assert details["reasoning_tokens"] > 0
    assert details["reasoning_tokens"] == 9


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


async def test_translate_stream_tool_name_after_args():
    info = resp.RequestInfo(model="m")
    chat_stream = _agen(
        [
            'data: {"id":"x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",'
            '"function":{"arguments":"{\\"a\\":1}"}}]},"finish_reason":null}]}\n\n',
            'data: {"id":"x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":""}}]},"finish_reason":null}]}\n\n',
            'data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
        ]
    )
    joined = "".join(await _collect(resp.translate_stream(chat_stream, info, "resp_1", 1)))
    added_at = joined.index("response.output_item.added")
    delta_at = joined.index("response.function_call_arguments.delta")
    done_at = joined.index("response.function_call_arguments.done")
    assert added_at < delta_at < done_at
    assert '"name": "f"' in joined
    assert '"delta": "{\\"a\\":1}"' in joined


async def test_translate_stream_interrupted_tool_call_closed():
    info = resp.RequestInfo(model="m")
    chat_stream = _agen(
        [
            'data: {"id":"x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",'
            '"function":{"arguments":"{\\"a\\":"}}]},"finish_reason":null}]}\n\n',
        ]
    )
    joined = "".join(await _collect(resp.translate_stream(chat_stream, info, "resp_1", 1)))
    added_at = joined.index("response.output_item.added")
    delta_at = joined.index("response.function_call_arguments.delta")
    done_at = joined.index("response.function_call_arguments.done")
    item_done_at = joined.index("event: response.output_item.done")
    assert added_at < delta_at < done_at < item_done_at
    assert '"arguments": "{\\"a\\":"}' in joined


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


def test_iter_sse_payloads_joins_multiline_data():
    payloads = list(resp._iter_sse_payloads('data: {"choices":\ndata: [{"delta":\ndata: {"content":"Hi"}}]}\n\n'))
    assert payloads == [{"choices": [{"delta": {"content": "Hi"}}]}]


def test_iter_sse_payloads_strips_carriage_return():
    assert list(resp._iter_sse_payloads('data: {"a": 1}\r\n\r\n')) == [{"a": 1}]


def test_iter_sse_payloads_ignores_non_object_and_comments():
    assert list(resp._iter_sse_payloads(": ping\n\ndata: [1, 2]\n\n")) == []


def test_iter_sse_payloads_mixed_frames():
    payloads = list(resp._iter_sse_payloads('data: {"a": 1}\n\ndata: [DONE]\n\ndata: {"b": 2}\n\n'))
    assert payloads == [{"a": 1}, None, {"b": 2}]


def test_response_from_chat_error_is_failed():
    info = resp.RequestInfo(model="m")
    obj = resp.response_from_chat(
        {"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": None}], "error": {"message": "bad"}},
        info,
        "r",
        1,
    )
    assert obj["status"] == "failed"
    assert obj["error"] == {"message": "bad"}
    assert obj["incomplete_details"] is None


def test_response_from_chat_reduced_context_incomplete():
    info = resp.RequestInfo(model="m")
    chat = {
        "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "response_incomplete"}],
        "error": {"message": "reduced", "finish_reason": "response_incomplete"},
    }
    obj = resp.response_from_chat(chat, info, "r", 1)
    assert obj["status"] == "incomplete"
    assert obj["incomplete_details"] == {"reason": "max_output_tokens"}
    assert obj["error"]["message"] == "reduced"


def test_response_from_chat_reduced_context_by_finish_reason_only():
    info = resp.RequestInfo(model="m")
    chat = {"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "response_incomplete"}]}
    obj = resp.response_from_chat(chat, info, "r", 1)
    assert obj["status"] == "incomplete"
    assert obj["incomplete_details"] == {"reason": "max_output_tokens"}


def test_response_from_chat_invalid_is_server_fault():
    with pytest.raises(HTTPException) as excinfo:
        resp.response_from_chat(5, resp.RequestInfo(model="m"), "r", 1)
    assert excinfo.value.status_code == 500


def test_messages_from_output_single_message_with_text_and_calls():
    output = [
        {"id": "msg_1", "type": "message", "content": [{"type": "output_text", "text": "hi"}]},
        {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"},
    ]
    assert resp.messages_from_output(output) == [
        {"role": "assistant", "content": "hi", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}
    ]


def test_input_message_item_detail_and_file():
    items = resp._input_message_item(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://x/y.png", "detail": "high"}},
                {"type": "image_url", "image_url": "https://x/z.png", "detail": "weird"},
                {"type": "image_url", "image_url": "https://x/w.png", "detail": "low"},
                {"type": "input_file", "filename": "f.txt", "file": {"id": "x"}},
                {"type": "input_file", "filename": "g.txt"},
            ],
        }
    )
    parts = items[0]["content"]
    assert parts[0] == {"type": "input_image", "image_url": "https://x/y.png", "detail": "high"}
    assert parts[1] == {"type": "input_image", "image_url": "https://x/z.png"}
    assert parts[2] == {"type": "input_image", "image_url": "https://x/w.png", "detail": "low"}
    assert parts[3] == {"type": "input_file", "file": {"id": "x"}, "filename": "f.txt"}
    assert len(parts) == 4


def test_normalize_content_keeps_valid_image_detail():
    assert resp._normalize_content([{"type": "input_image", "image_url": {"url": "data:x", "detail": "high"}}]) == [
        {"type": "image_url", "image_url": "data:x", "detail": "high"}
    ]
    assert resp._normalize_content([{"type": "input_image", "image_url": {"url": "data:x", "detail": "bogus"}}]) == [
        {"type": "image_url", "image_url": "data:x"}
    ]


def test_request_info_has_no_extras_field():
    assert not hasattr(resp.RequestInfo(model="m"), "extras")


async def test_translate_stream_error_event_keeps_discriminator():
    info = resp.RequestInfo(model="m")
    upstream = FakeUpstream(['data: {"id":"x","error":{"message":"boom","finish_reason":"server_busy"},"choices":[]}\n\n'])
    seen = []
    frames = _frames("".join(await _collect(resp.translate_stream(upstream, info, "r", 1, on_complete=seen.append))))
    errors = [payload for event, payload in frames if event == "error"]
    assert len(errors) == 1
    assert errors[0]["type"] == "error"
    assert errors[0]["message"] == "boom"
    assert errors[0]["code"] == "server_busy"
    failed = [payload for event, payload in frames if event == "response.failed"]
    assert len(failed) == 1
    assert failed[0]["response"]["status"] == "failed"
    assert seen and seen[0]["status"] == "failed"
    assert upstream.closed


async def test_translate_stream_reduced_context_is_incomplete():
    info = resp.RequestInfo(model="m")
    upstream = FakeUpstream(
        [
            'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
            'data: {"id":"x","error":{"message":"reduced","finish_reason":"response_incomplete"},'
            '"choices":[{"index":0,"delta":{},"finish_reason":"response_incomplete"}]}\n\n',
        ]
    )
    seen = []
    frames = _frames("".join(await _collect(resp.translate_stream(upstream, info, "r", 1, on_complete=seen.append))))
    events = {event for event, _ in frames}
    assert "response.incomplete" in events
    assert "response.failed" not in events
    assert "error" not in events
    incomplete = _named(frames, "response.incomplete")
    assert incomplete[0]["response"]["status"] == "incomplete"
    assert incomplete[0]["response"]["incomplete_details"] == {"reason": "max_output_tokens"}
    assert seen and seen[0]["status"] == "incomplete"
    assert upstream.closed


async def test_translate_stream_closes_upstream_on_failure():
    info = resp.RequestInfo(model="m")
    upstream = FakeUpstream(
        [
            'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
            'data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        ],
        fail_at=1,
    )
    with pytest.raises(RuntimeError):
        await _collect(resp.translate_stream(upstream, info, "r", 1))
    assert upstream.closed


async def test_translate_stream_closes_upstream_on_client_disconnect():
    info = resp.RequestInfo(model="m")
    upstream = FakeUpstream(
        [
            'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
            'data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        ]
    )
    stream = resp.translate_stream(upstream, info, "r", 1)
    await stream.__anext__()
    await stream.aclose()
    assert upstream.closed


async def test_translate_stream_closes_upstream_on_success():
    info = resp.RequestInfo(model="m")
    upstream = FakeUpstream(['data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'])
    await _collect(resp.translate_stream(upstream, info, "r", 1))
    assert upstream.closed


def test_endpoint_stream_failure_is_persisted():
    from unittest.mock import AsyncMock

    pool, acct = make_pool()
    acct.sessions.obtain = AsyncMock(side_effect=HTTPException(503, "provider down"))
    app.state.pool = pool
    client = TestClient(app)
    stream = client.post("/v1/responses", json={"model": "deepseek-v4.1-flash", "input": "hi", "stream": True})
    text = stream.text
    client.close()
    assert stream.status_code == 200
    frames = _frames(text)
    assert _named(frames, "response.failed")
    errors = _named(frames, "error")
    assert errors and errors[0]["type"] == "error"
    failed = _named(frames, "response.failed")[0]["response"]
    assert failed["status"] == "failed"
    stored = app.state.responses_store.get(failed["id"])
    assert stored is not None
    assert stored["public"]["status"] == "failed"
    client = TestClient(app)
    fetched = client.get(f"/v1/responses/{failed['id']}")
    client.close()
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "failed"


def test_endpoint_stream_error_type_is_not_error_type():
    from unittest.mock import AsyncMock

    pool, acct = make_pool()
    acct.sessions.obtain = AsyncMock(side_effect=HTTPException(503, "provider down"))
    app.state.pool = pool
    client = TestClient(app)
    stream = client.post("/v1/responses", json={"model": "deepseek-v4.1-flash", "input": "hi", "stream": True})
    client.close()
    errors = _named(_frames(stream.text), "error")
    assert len(errors) == 1
    assert set(errors[0]) >= {"type", "sequence_number", "message", "code", "param"}


async def test_translate_stream_output_order_matches_announced_indices():
    info = resp.RequestInfo(model="m")
    upstream = FakeUpstream(
        [
            'data: {"id":"x","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}\n\n',
            'data: {"id":"x","choices":[{"index":0,"delta":{"reasoning_content":"late"},"finish_reason":null}]}\n\n',
            'data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        ]
    )
    frames = _frames("".join(await _collect(resp.translate_stream(upstream, info, "r", 1))))
    added = {payload["output_index"]: payload["item"]["type"] for event, payload in frames if event == "response.output_item.added"}
    assert added == {0: "message", 1: "reasoning"}
    completed = _named(frames, "response.completed")
    assert [item["type"] for item in completed[0]["response"]["output"]] == [added[index] for index in sorted(added)]
