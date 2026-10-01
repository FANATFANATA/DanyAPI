import asyncio
import json
import logging

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import danyapi.api.retry as retry_mod
from danyapi.api import anthropic as ant
from danyapi.api.openai import app

PNG = "iVBORw0KGgo="


class FakeSession:
    def __init__(self, sid="c1", last_message_id=None):
        self.id = sid
        self.last_message_id = last_message_id
        self.last_response_id = None
        self.accumulated_tokens = 0


class FakeResp:
    def __init__(self, sse_text):
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream; charset=utf-8"}
        self._b = sse_text.encode()

    async def aiter_bytes(self):
        yield self._b

    async def aclose(self):
        pass

    async def aread(self):
        return self._b


def _sse(*payloads: str) -> str:
    return "".join(f"data: {item}\n\n" for item in payloads)


class FakeAccount:
    def __init__(self, sse_text):
        from unittest.mock import AsyncMock, MagicMock

        self.index = 0
        self.broken = False
        self.client = MagicMock()
        self.client.completion = AsyncMock(return_value=FakeResp(sse_text))
        self.client.create_pow_challenge = AsyncMock(return_value={})
        self.client.upload_file = AsyncMock(return_value={"id": "file_1"})
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


def make_pool(sse_text):
    from unittest.mock import AsyncMock, MagicMock

    acct = FakeAccount(sse_text)
    pool = MagicMock()
    pool.acquire = AsyncMock(return_value=(acct, None))
    return pool


TOOL_XML = '<tool_calls>\n<invoke name="get_weather">\n<parameter name="city">Paris</parameter>\n</invoke>\n</tool_calls>'


def _ds_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _ds_ready() -> str:
    ready = json.dumps({"request_message_id": 1, "response_message_id": 2, "model_type": "default"})
    return f"event: ready\ndata: {ready}\n\n"


def _ds_sse(content="Hi", reasoning=None, tool_calls_xml=None, finish="FINISHED") -> str:
    fragments = []
    if reasoning:
        fragments.append({"id": 2, "type": "THINK", "content": reasoning})
    body = tool_calls_xml if tool_calls_xml is not None else content
    if body:
        fragments.append({"id": 2, "type": "RESPONSE", "content": body})
    out = _ds_ready()
    if fragments:
        out += _ds_event({"v": {"response": {"message_id": 2, "parent_id": 1, "status": "WIP", "fragments": fragments}}})
    out += _ds_event({"p": "response/status", "o": "SET", "v": finish})
    return out


def _ds_stream_sse(*deltas: str) -> str:
    out = _ds_ready()
    for delta in deltas:
        out += _ds_event({"p": "response/fragments/0/content", "o": "APPEND", "v": delta})
    out += _ds_event({"p": "response/status", "o": "SET", "v": "FINISHED"})
    return out


def _chat_sse(content=None, reasoning=None, tool_calls=None, finish="stop", usage=True) -> str:
    parts = []
    if reasoning:
        parts.append(json.dumps({"choices": [{"index": 0, "delta": {"reasoning_content": reasoning}}]}))
    if content is not None:
        parts.append(json.dumps({"choices": [{"index": 0, "delta": {"content": content}}]}))
    if tool_calls:
        parts.append(json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": tool_calls}}]}))
    chunk: dict = {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
    if usage:
        chunk["usage"] = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    parts.append(json.dumps(chunk))
    return "".join(f"data: {item}\n\n" for item in parts)


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


def _named(frames, name):
    return [payload for event, payload in frames if event == name]


def _block_events(frames):
    return [(event, payload["index"]) for event, payload in frames if event in ("content_block_start", "content_block_delta", "content_block_stop")]


def _tool_starts(frames):
    return [payload for event, payload in frames if event == "content_block_start" and payload["content_block"]["type"] == "tool_use"]


def _tool_deltas(frames):
    return [payload for event, payload in frames if event == "content_block_delta" and payload["delta"]["type"] == "input_json_delta"]


@pytest.fixture(autouse=True)
def clean_state():
    saved = (getattr(app.state, "pool", None), getattr(app.state, "qwen_pool", None))
    app.state.pool = None
    app.state.qwen_pool = None
    yield
    app.state.pool, app.state.qwen_pool = saved


@pytest.fixture(autouse=True)
def zero_backoff(monkeypatch):
    monkeypatch.setattr(retry_mod, "RETRY_BACKOFF_SEC", 0.0)


def _info(model="claude-sonnet-4-5", **kwargs):
    return ant.RequestInfo(model=model, max_tokens=100, **kwargs)


async def _agen_chunks(*texts: str):
    for text in texts:
        yield text


def test_normalize_system_string():
    assert ant.normalize_system("be nice") == "be nice"


def test_normalize_system_blocks():
    assert ant.normalize_system([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"


def test_normalize_system_none():
    assert ant.normalize_system(None) is None
    assert ant.normalize_system("") is None


def test_normalize_messages_plain():
    assert ant.normalize_messages([{"role": "user", "content": "hi"}]) == [{"role": "user", "content": "hi"}]


def test_normalize_messages_text_blocks_collapse():
    messages = ant.normalize_messages([{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
    assert messages == [{"role": "user", "content": "ab"}]


def test_normalize_messages_mixed_blocks():
    messages = ant.normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
                ],
            }
        ]
    )
    assert messages[0]["content"][0] == {"type": "text", "text": "look"}
    assert messages[0]["content"][1] == {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}


def test_normalize_messages_image_url_source():
    messages = ant.normalize_messages([{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}]}])
    assert messages[0]["content"] == [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]


def test_normalize_messages_image_without_source():
    with pytest.raises(ant.AnthropicInputError):
        ant.normalize_messages([{"role": "user", "content": [{"type": "image"}]}])


def test_normalize_messages_tool_use_and_result():
    messages = ant.normalize_messages(
        [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "f", "input": {"a": 1}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "42"}]},
        ]
    )
    assert len(messages) == 2
    assert messages[0]["tool_calls"][0]["id"] == "tu1"
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert messages[1] == {"role": "tool", "tool_call_id": "tu1", "content": "42"}


def test_normalize_messages_tool_result_only_message_adds_no_trailing_user_message():
    messages = ant.normalize_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "f", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "42"}]},
        ]
    )
    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]


def test_normalize_messages_tool_error_result_is_prefixed():
    messages = ant.normalize_messages([{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "bad", "is_error": True}]}])
    assert messages == [{"role": "tool", "tool_call_id": "tu1", "content": "[tool_error] bad"}]


def test_normalize_messages_rejects_unknown_block_type():
    with pytest.raises(ant.AnthropicInputError):
        ant.normalize_messages([{"role": "user", "content": [{"type": "mystery", "value": 1}]}])


def test_normalize_messages_skips_ignorable_block_types():
    messages = ant.normalize_messages([{"role": "assistant", "content": [{"type": "mcp_tool_result", "content": "x"}, {"type": "text", "text": "ok"}]}])
    assert messages == [{"role": "assistant", "content": "ok"}]


def test_normalize_messages_skips_thinking_blocks():
    messages = ant.normalize_messages([{"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "ok"}]}])
    assert messages == [{"role": "assistant", "content": "ok"}]


def test_normalize_messages_rejects_system_role():
    with pytest.raises(ant.AnthropicInputError):
        ant.normalize_messages([{"role": "system", "content": "no"}])


def test_normalize_messages_rejects_empty():
    with pytest.raises(ant.AnthropicInputError):
        ant.normalize_messages([])


def test_normalize_messages_rejects_bad_type():
    with pytest.raises(ant.AnthropicInputError):
        ant.normalize_messages("nope")


def test_convert_tools_input_schema():
    tools = ant.convert_tools([{"name": "f", "description": "d", "input_schema": {"type": "object"}}])
    assert tools == [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]


def test_convert_tools_passthrough_openai():
    openai_tools = [{"type": "function", "function": {"name": "f"}}]
    assert ant.convert_tools(openai_tools) == openai_tools


def test_convert_tool_choice_variants():
    assert ant.convert_tool_choice({"type": "auto"}) == "auto"
    assert ant.convert_tool_choice({"type": "any"}) == "required"
    assert ant.convert_tool_choice({"type": "none"}) == "none"
    assert ant.convert_tool_choice({"type": "tool", "name": "f"}) == {"type": "function", "function": {"name": "f"}}


def test_convert_stop_sequences():
    assert ant.convert_stop_sequences(["a", "b"]) == ["a", "b"]
    assert ant.convert_stop_sequences("a") == ["a"]
    assert ant.convert_stop_sequences(None) is None


def test_convert_stop_sequences_accepts_long_list_and_rejects_non_strings():
    long_list = [f"seq-{index}" for index in range(32)]
    assert ant.convert_stop_sequences(long_list) == long_list
    with pytest.raises(ant.AnthropicInputError):
        ant.convert_stop_sequences([1])
    with pytest.raises(ant.AnthropicInputError):
        ant.convert_stop_sequences(["ok", ""])
    with pytest.raises(ant.AnthropicInputError):
        ant.convert_stop_sequences([None])


def test_convert_stop_sequences_rejects_non_list():
    with pytest.raises(ant.AnthropicInputError):
        ant.convert_stop_sequences(5)


def test_endpoint_accepts_long_stop_sequences():
    app.state.pool = make_pool(_ds_sse("Hi"))
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "deepseek-v4.1-flash",
            "max_tokens": 10,
            "stop_sequences": [f"s{index}" for index in range(32)],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    client.close()
    assert r.status_code == 200
    assert r.json()["type"] == "message"


def test_endpoint_rejects_non_string_stop_sequence():
    app.state.pool = make_pool(_ds_sse("Hi"))
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={"model": "deepseek-v4.1-flash", "max_tokens": 10, "stop_sequences": ["ok", 5], "messages": [{"role": "user", "content": "hi"}]},
    )
    client.close()
    assert r.status_code == 400
    assert r.json() == {"type": "error", "error": {"type": "invalid_request_error", "message": "stop_sequences[1] must be a non-empty string"}}


def test_as_max_tokens():
    assert ant.as_max_tokens(None) == ant.DEFAULT_MAX_TOKENS
    assert ant.as_max_tokens(64) == 64
    with pytest.raises(ant.AnthropicInputError):
        ant.as_max_tokens(0)
    with pytest.raises(ant.AnthropicInputError):
        ant.as_max_tokens(-1)
    with pytest.raises(ant.AnthropicInputError):
        ant.as_max_tokens("x")
    with pytest.raises(ant.AnthropicInputError):
        ant.as_max_tokens(10.7)
    with pytest.raises(ant.AnthropicInputError):
        ant.as_max_tokens(True)


def test_convert_tools_rejects_invalid_entries():
    with pytest.raises(ant.AnthropicInputError):
        ant.convert_tools(["nope"])
    with pytest.raises(ant.AnthropicInputError):
        ant.convert_tools([{"description": "d"}])
    with pytest.raises(ant.AnthropicInputError):
        ant.convert_tools([{"name": ""}])
    with pytest.raises(ant.AnthropicInputError):
        ant.convert_tools([{"name": 5}])


def test_build_chat_request_validates_sampling_params():
    base = {"messages": [{"role": "user", "content": "hi"}]}
    assert ant.build_chat_request({**base, "temperature": 0.0}, "m")["temperature"] == 0.0
    assert ant.build_chat_request({**base, "top_p": 1}, "m")["top_p"] == 1.0
    assert "top_k" not in ant.build_chat_request({**base, "top_k": None}, "m")
    for field, value in (("temperature", 1.5), ("top_p", -0.1), ("top_p", "0.5"), ("top_k", 5), ("top_k", 0), ("top_k", 5.5), ("top_k", True)):
        with pytest.raises(ant.AnthropicInputError):
            ant.build_chat_request({**base, field: value}, "m")


def test_build_chat_request_names_top_k_in_the_rejection():
    with pytest.raises(ant.AnthropicInputError) as excinfo:
        ant.build_chat_request({"messages": [{"role": "user", "content": "hi"}], "top_k": 5}, "m")
    assert str(excinfo.value) == "top_k is not supported by the upstream providers, remove it from the request"


def test_as_text_is_depth_bounded():
    nested: object = "leaf"
    for _ in range(ant.MAX_TEXT_DEPTH + 20):
        nested = [{"content": nested}]
    assert ant._as_text(nested) == ""
    assert ant._as_text([{"text": "a"}, {"content": ["b", {"text": "c"}]}]) == "abc"


def test_build_chat_request_full():
    payload = ant.build_chat_request(
        {
            "model": "claude-sonnet-4-5",
            "max_tokens": 32,
            "system": "be brief",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "f", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"},
            "stop_sequences": ["END"],
            "temperature": 0.5,
            "top_p": 0.9,
            "stream": True,
            "metadata": {"user_id": "u1"},
        },
        "deepseek-v4.1-flash",
    )
    assert payload["model"] == "deepseek-v4.1-flash"
    assert payload["messages"][0] == {"role": "system", "content": "be brief"}
    assert payload["messages"][1] == {"role": "user", "content": "hi"}
    assert payload["max_tokens"] == 32
    assert payload["tool_choice"] == "required"
    assert payload["stop"] == ["END"]
    assert payload["stream"] is True
    assert payload["user"] == "u1"
    assert payload["tools"][0]["function"]["name"] == "f"


def test_build_chat_request_minimal():
    payload = ant.build_chat_request({"messages": [{"role": "user", "content": "hi"}]}, "m")
    assert payload["max_tokens"] == ant.DEFAULT_MAX_TOKENS
    assert "system" not in payload["messages"][0]
    assert "tools" not in payload


def test_build_chat_request_rejects_bad_input():
    with pytest.raises(ant.AnthropicInputError):
        ant.build_chat_request({"messages": [{"role": "nope", "content": "x"}]}, "m")


def test_build_content_text():
    blocks = ant.build_content({"message": {"content": "hello"}})
    assert blocks == [{"type": "text", "text": "hello"}]


def test_build_content_thinking_first():
    blocks = ant.build_content({"message": {"reasoning_content": "why", "content": "hello"}})
    assert blocks[0]["type"] == "thinking"
    assert blocks[0]["thinking"] == "why"
    assert blocks[1] == {"type": "text", "text": "hello"}


def test_build_content_tool_use_decodes_arguments():
    blocks = ant.build_content({"message": {"content": "", "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": '{"a":1}'}}]}})
    assert blocks == [{"type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1}}]


def test_build_content_empty_returns_no_blocks():
    assert ant.build_content({"message": {"content": ""}}) == []
    assert ant.build_content({"message": {}}) == []
    assert ant.build_content({}) == []
    assert ant.build_content({"message": {"content": "hi"}}) == [{"type": "text", "text": "hi"}]


def test_usage_maps_provider_fields():
    assert ant._usage({"prompt_tokens": 5, "completion_tokens": 3}) == {"input_tokens": 5, "output_tokens": 3}
    assert ant._usage({"input_tokens": 2, "output_tokens": 4}) == {"input_tokens": 2, "output_tokens": 4}
    assert ant._usage(None) == {"input_tokens": 0, "output_tokens": 0}


def test_stop_reason_mapping():
    assert ant.stop_reason_for("stop") == "end_turn"
    assert ant.stop_reason_for("length") == "max_tokens"
    assert ant.stop_reason_for("tool_calls") == "tool_use"
    assert ant.stop_reason_for(None) == "end_turn"


def test_build_message_shape():
    message = ant.build_message(
        _info(), "msg_1", {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}
    )
    assert message["id"] == "msg_1"
    assert message["type"] == "message"
    assert message["role"] == "assistant"
    assert message["model"] == "claude-sonnet-4-5"
    assert message["content"] == [{"type": "text", "text": "hi"}]
    assert message["stop_reason"] == "end_turn"
    assert message["stop_sequence"] is None
    assert message["usage"] == {"input_tokens": 4, "output_tokens": 2}


def test_build_message_reports_matched_stop_sequence():
    message = ant.build_message(
        _info(stop_sequences=["END", "STOP"]),
        "msg_1",
        {"choices": [{"message": {"content": "say END now"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
    )
    assert message["stop_reason"] == "stop_sequence"
    assert message["stop_sequence"] == "END"


def test_build_message_stop_reason_ignores_unmatched_stop_sequences():
    message = ant.build_message(
        _info(stop_sequences=["NOPE"]),
        "msg_1",
        {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]},
    )
    assert message["stop_reason"] == "end_turn"
    assert message["stop_sequence"] is None


def test_build_message_error_maps_to_end_turn():
    message = ant.build_message(
        _info(), "msg_1", {"error": {"message": "boom"}, "choices": [{"message": {"content": "partial"}, "finish_reason": "tool_calls"}]}
    )
    assert message["stop_reason"] == "end_turn"
    assert message["stop_sequence"] is None
    assert message["content"] == [{"type": "text", "text": "partial"}]
    assert message["usage"] == {"input_tokens": 0, "output_tokens": 0}


@pytest.mark.parametrize(
    "payload",
    [{}, {"choices": []}, {"choices": "junk"}, {"choices": ["junk"]}],
)
def test_build_message_refuses_a_response_without_choices(payload):
    with pytest.raises(HTTPException) as excinfo:
        ant.build_message(_info(), "msg_1", payload)
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "upstream returned a response without choices"


def test_build_message_keeps_the_upstream_message_when_there_are_no_choices():
    with pytest.raises(HTTPException) as excinfo:
        ant.build_message(_info(), "msg_1", {"error": {"message": "boom"}})
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "boom"


def test_count_input_tokens_includes_system():
    without = ant.count_input_tokens([{"role": "user", "content": "hello"}], None)
    with_system = ant.count_input_tokens([{"role": "user", "content": "hello"}], "a longer system prompt")
    assert with_system > without


def test_endpoint_unknown_model():
    client = TestClient(app)
    r = client.post("/v1/messages", json={"model": "gpt-4", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
    client.close()
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "not_found_error"


def test_endpoint_provider_not_configured():
    client = TestClient(app)
    r = client.post("/v1/messages", json={"model": "deepseek-v4.1-flash", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
    client.close()
    assert r.status_code == 503


def test_endpoint_bad_messages():
    app.state.pool = make_pool(_ds_sse("Hi"))
    client = TestClient(app)
    r = client.post("/v1/messages", json={"model": "deepseek-v4.1-flash", "max_tokens": 10, "messages": [{"role": "system", "content": "x"}]})
    client.close()
    assert r.status_code == 400
    assert r.json()["type"] == "error"
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_endpoint_non_stream():
    app.state.pool = make_pool(_ds_sse("Hi"))
    client = TestClient(app)
    r = client.post("/v1/messages", json={"model": "deepseek-v4.1-flash", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
    client.close()
    assert r.status_code == 200
    data = r.json()
    assert data["type"] == "message"
    assert data["id"].startswith("msg_")
    assert data["content"] == [{"type": "text", "text": "Hi"}]
    assert data["stop_reason"] == "end_turn"
    assert set(data["usage"]) == {"input_tokens", "output_tokens"}
    assert data["usage"]["input_tokens"] >= 0


def test_endpoint_non_stream_with_reasoning():
    app.state.pool = make_pool(_ds_sse("Hi", reasoning="because"))
    client = TestClient(app)
    r = client.post("/v1/messages", json={"model": "deepseek-v4.1-flash-thinking", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
    client.close()
    data = r.json()
    assert data["content"][0] == {"type": "thinking", "thinking": "because", "signature": ""}
    assert data["content"][1] == {"type": "text", "text": "Hi"}


def test_endpoint_non_stream_with_tool():
    app.state.pool = make_pool(_ds_sse(tool_calls_xml=TOOL_XML))
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "deepseek-v4.1-flash",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"name": "get_weather", "description": "d", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        },
    )
    client.close()
    data = r.json()
    assert data["stop_reason"] == "tool_use"
    assert len(data["content"]) == 1
    block = data["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "get_weather"
    assert block["input"] == {"city": "Paris"}


def test_endpoint_non_stream_with_image():
    app.state.pool = make_pool(_ds_sse("A cat"))
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "deepseek-v4.1-flash",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
                    ],
                }
            ],
        },
    )
    client.close()
    assert r.status_code == 200
    assert r.json()["content"] == [{"type": "text", "text": "A cat"}]


def test_endpoint_stream_event_order():
    app.state.pool = make_pool(_ds_sse("Hello"))
    client = TestClient(app)
    with client.stream(
        "POST", "/v1/messages", json={"model": "deepseek-v4.1-flash", "max_tokens": 10, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    ) as r:
        text = "".join(r.iter_text())
    client.close()
    frames = _frames(text)
    events = [event for event, _ in frames]
    assert events[0] == "message_start"
    assert events[-1] == "message_stop"
    assert "content_block_start" in events
    assert "content_block_delta" in events
    assert "content_block_stop" in events
    assert "message_delta" in events
    starts = _named(frames, "content_block_start")
    assert starts[0]["content_block"] == {"type": "text", "text": ""}
    deltas = _named(frames, "content_block_delta")
    assert "".join(d["delta"]["text"] for d in deltas) == "Hello"
    final = _named(frames, "message_delta")
    assert final[0]["delta"] == {"stop_reason": "end_turn", "stop_sequence": None}
    assert "output_tokens" in final[0]["usage"]
    started = _named(frames, "message_start")[0]
    assert started["message"]["usage"]["input_tokens"] > 0
    assert started["message"]["usage"]["output_tokens"] == 0
    assert final[0]["usage"]["output_tokens"] > 0


def test_translate_stream_thinking_block():
    async def run():
        stream = ant.translate_stream(_agen_chunks(_chat_sse("Hi", reasoning="think")), _info(), "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    starts = _named(frames, "content_block_start")
    assert [payload["content_block"] for payload in starts] == [
        {"type": "thinking", "thinking": "", "signature": ""},
        {"type": "text", "text": ""},
    ]
    deltas = _named(frames, "content_block_delta")
    assert deltas[0]["delta"] == {"type": "thinking_delta", "thinking": "think"}
    assert deltas[1]["delta"] == {"type": "text_delta", "text": "Hi"}
    assert _named(frames, "message_delta")[0]["delta"] == {"stop_reason": "end_turn", "stop_sequence": None}


def test_translate_stream_tool_use_block():
    calls = [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": '{"a":1}'}}]
    text = _chat_sse(finish="tool_calls", tool_calls=calls)

    async def run():
        stream = ant.translate_stream(_agen_chunks(text), _info(), "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    assert _tool_starts(frames) == [{"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "c1", "name": "f", "input": {}}}]
    assert _tool_deltas(frames) == [{"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"a":1}'}}]
    assert _block_events(frames) == [
        ("content_block_start", 0),
        ("content_block_delta", 0),
        ("content_block_stop", 0),
    ]
    assert _named(frames, "message_delta")[0]["delta"] == {"stop_reason": "tool_use", "stop_sequence": None}


def test_translate_stream_tool_use_starts_once_when_id_repeats():
    text = _sse(
        json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{"}}]}}]}),
        json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"arguments": '"a":1}'}}]}}]}),
        json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    )

    async def run():
        stream = ant.translate_stream(_agen_chunks(text), _info(), "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    starts = _tool_starts(frames)
    assert len(starts) == 1
    assert starts[0]["content_block"] == {"type": "tool_use", "id": "c1", "name": "f", "input": {}}
    assert [(payload["index"], payload["delta"]["partial_json"]) for payload in _tool_deltas(frames)] == [(0, "{"), (0, '"a":1}')]


def test_translate_stream_tool_use_starts_without_provider_id():
    text = _sse(
        json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"name": "search", "arguments": "{}"}}]}}]}),
        json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"q":"x"}'}}]}}]}),
        json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    )

    async def run():
        stream = ant.translate_stream(_agen_chunks(text), _info(), "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    starts = _tool_starts(frames)
    assert len(starts) == 1
    assert starts[0]["content_block"]["name"] == "search"
    assert starts[0]["content_block"]["id"].startswith("call_")
    assert [(payload["index"], payload["delta"]["partial_json"]) for payload in _tool_deltas(frames)] == [(0, "{}"), (0, '{"q":"x"}')]
    assert _named(frames, "message_delta")[0]["delta"]["stop_reason"] == "tool_use"


def test_translate_stream_tool_use_keyed_by_call_index():
    text = _sse(
        json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "a", "function": {"name": "fa", "arguments": "{}"}},
                                {"index": 1, "id": "b", "function": {"name": "fb", "arguments": "{}"}},
                            ]
                        },
                    }
                ]
            }
        ),
        json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "function": {"arguments": '{"x":1}'}}]}}]}),
        json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    )

    async def run():
        stream = ant.translate_stream(_agen_chunks(text), _info(), "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    starts = [(payload["index"], payload["content_block"]["id"], payload["content_block"]["name"]) for payload in _tool_starts(frames)]
    assert starts == [(0, "a", "fa"), (1, "b", "fb")]
    assert [(payload["index"], payload["delta"]["partial_json"]) for payload in _tool_deltas(frames)] == [(0, "{}"), (1, "{}"), (1, '{"x":1}')]
    assert [payload["index"] for event, payload in frames if event == "content_block_stop"] == [0, 1]


def test_translate_stream_error_event():
    text = 'data: {"error":{"message":"boom"}}\n\n'

    async def run():
        stream = ant.translate_stream(_agen_chunks(text), _info(), "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    assert [event for event, _ in frames] == ["message_start", "error", "message_delta", "message_stop"]
    assert _named(frames, "error")[0]["error"] == {"type": "api_error", "message": "boom"}
    assert _named(frames, "message_delta")[0]["delta"] == {"stop_reason": "end_turn", "stop_sequence": None}
    assert _named(frames, "message_delta")[0]["usage"] == {"input_tokens": _info().prompt_tokens, "output_tokens": 0}


def test_error_message_does_not_forward_non_dict_error():
    assert ant._error_message({"message": "boom"}) == ("api_error", "boom")
    assert ant._error_message({"message": 5}) == ("api_error", "upstream stream error")
    assert ant._error_message({}) == ("api_error", "upstream stream error")
    assert ant._error_message("secret token") == ("api_error", "upstream stream error")


def test_translate_stream_closes_upstream():
    closed = {"value": False}

    class Stream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield _chat_sse("Hi")

        async def aclose(self):
            closed["value"] = True

    async def run():
        stream = ant.translate_stream(Stream(), _info(), "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    assert closed["value"] is True
    assert _named(frames, "message_delta")[0]["delta"] == {"stop_reason": "end_turn", "stop_sequence": None}


def test_translate_stream_upstream_exception_closes_blocks_and_errors():
    async def gen():
        yield _sse(json.dumps({"choices": [{"index": 0, "delta": {"content": "hi"}}]}))
        yield _sse(json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{}"}}]}}]}))
        raise RuntimeError("upstream died")

    collected: list[str] = []

    async def run():
        async for line in ant.translate_stream(gen(), _info(), "msg_1"):
            collected.append(line)

    with pytest.raises(RuntimeError, match="upstream died"):
        asyncio.run(run())

    frames = _frames("".join(collected))
    assert [event for event, _ in frames] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "error",
    ]
    assert _named(frames, "error")[0]["error"] == {"type": "api_error", "message": "upstream stream failed"}


def test_translate_stream_reraises_generator_exit_without_a_spurious_error(caplog):
    async def gen():
        yield _sse(json.dumps({"choices": [{"index": 0, "delta": {"content": "hi"}}]}))

    async def run():
        stream = ant.translate_stream(gen(), _info(), "msg_1")
        for _ in range(3):
            await stream.__anext__()
        await stream.aclose()

    with caplog.at_level(logging.WARNING, logger="danyapi.api.anthropic"):
        asyncio.run(run())
    assert "aborted by the upstream generator" not in caplog.text


def test_translate_stream_decodes_bytes_chunks():
    chunk = b'data: {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]}\n\n'

    async def gen():
        yield chunk

    async def run():
        stream = ant.translate_stream(gen(), _info(), "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    assert [payload["delta"]["text"] for payload in _named(frames, "content_block_delta")] == ["hi"]
    assert _named(frames, "message_delta")[0]["delta"] == {"stop_reason": "end_turn", "stop_sequence": None}


def test_translate_stream_merges_usage_across_chunks():
    text = _sse(
        json.dumps({"choices": [{"index": 0, "delta": {"content": "hi"}}], "usage": {"prompt_tokens": 42}}),
        json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 9}}),
    )

    async def run():
        stream = ant.translate_stream(_agen_chunks(text), _info(prompt_tokens=17), "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    assert _named(frames, "message_start")[0]["message"]["usage"] == {"input_tokens": 17, "output_tokens": 0}
    assert _named(frames, "message_delta")[0]["usage"] == {"input_tokens": 42, "output_tokens": 9}


def test_translate_stream_awaits_async_on_complete():
    seen: list[tuple] = []

    async def on_complete(reason, usage):
        seen.append((reason, dict(usage)))

    text = _sse(
        json.dumps({"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}})
    )

    async def run():
        stream = ant.translate_stream(_agen_chunks(text), _info(), "msg_1", on_complete)
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    assert seen == [("end_turn", {"input_tokens": 5, "output_tokens": 3})]
    assert _named(frames, "message_delta")[0]["usage"] == {"input_tokens": 5, "output_tokens": 3}


def test_translate_stream_reports_matched_stop_sequence():
    text = _sse(
        json.dumps({"choices": [{"index": 0, "delta": {"content": "say END now"}}]}),
        json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    )

    async def run():
        stream = ant.translate_stream(_agen_chunks(text), _info(stop_sequences=["END"]), "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    assert _named(frames, "message_delta")[0]["delta"] == {"stop_reason": "stop_sequence", "stop_sequence": "END"}


def test_endpoint_stream_length_maps_to_max_tokens():
    app.state.pool = make_pool(_ds_sse("partial", finish="INCOMPLETE"))
    client = TestClient(app)
    with client.stream(
        "POST", "/v1/messages", json={"model": "deepseek-v4.1-flash", "max_tokens": 4, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    ) as r:
        text = "".join(r.iter_text())
    client.close()
    assert _named(_frames(text), "message_delta")[0]["delta"]["stop_reason"] == "max_tokens"


def test_count_tokens_endpoint():
    client = TestClient(app)
    r = client.post(
        "/v1/messages/count_tokens",
        json={"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "hello there"}], "system": "be brief"},
    )
    client.close()
    assert r.status_code == 200
    assert r.json()["input_tokens"] > 0


def test_count_tokens_endpoint_rejects_bad_messages():
    client = TestClient(app)
    r = client.post("/v1/messages/count_tokens", json={"model": "deepseek-v4.1-flash", "messages": [{"role": "system", "content": "x"}]})
    client.close()
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_endpoint_accepts_qwen_model():
    app.state.qwen_pool = make_pool("")
    app.state.qwen_models = [{"id": "qwen3.8-max", "name": "Qwen3.8-Max", "owned_by": "qwen", "model_type": "chat"}]
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={"model": "qwen3.8-max", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
    )
    client.close()
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "qwen3.8-max"


def test_endpoint_unknown_model_is_anthropic_error():
    client = TestClient(app)
    r = client.post("/v1/messages", json={"model": "nope", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})
    client.close()
    assert r.status_code == 404
    assert r.json() == {"type": "error", "error": {"type": "not_found_error", "message": "Unknown model: nope"}}


def test_endpoint_upstream_http_error_is_anthropic_error():
    from unittest.mock import AsyncMock

    from danyapi.accounts import AccountPoolBusy

    app.state.pool = make_pool("")
    app.state.pool.acquire = AsyncMock(side_effect=AccountPoolBusy())
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={"model": "deepseek-v4.1-flash", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
    )
    client.close()
    assert r.status_code == 429
    assert r.json() == {"type": "error", "error": {"type": "rate_limit_error", "message": "all accounts are busy, try again later"}}


def test_translate_stream_closes_thinking_before_text():
    text = _chat_sse(reasoning="think", content="answer")

    async def run():
        stream = ant.translate_stream(_agen_chunks(text), _info(), "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    assert _block_events(frames) == [
        ("content_block_start", 0),
        ("content_block_delta", 0),
        ("content_block_stop", 0),
        ("content_block_start", 1),
        ("content_block_delta", 1),
        ("content_block_stop", 1),
    ]


def test_translate_stream_closes_text_before_tool_use():
    calls = [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{}"}}]
    text = _chat_sse(reasoning="think", content="answer", tool_calls=calls, finish="tool_calls")

    async def run():
        stream = ant.translate_stream(_agen_chunks(text), _info(), "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    assert [payload["content_block"]["type"] for event, payload in frames if event == "content_block_start"] == ["thinking", "text", "tool_use"]
    assert _block_events(frames) == [
        ("content_block_start", 0),
        ("content_block_delta", 0),
        ("content_block_stop", 0),
        ("content_block_start", 1),
        ("content_block_delta", 1),
        ("content_block_stop", 1),
        ("content_block_start", 2),
        ("content_block_delta", 2),
        ("content_block_stop", 2),
    ]
    assert _named(frames, "message_delta")[0]["delta"] == {"stop_reason": "tool_use", "stop_sequence": None}


def test_translate_stream_keeps_known_input_tokens():
    chunk = json.dumps({"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 0, "completion_tokens": 5}})
    info = _info(prompt_tokens=17)

    async def run():
        stream = ant.translate_stream(_agen_chunks(f"data: {chunk}\n\n"), info, "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    assert _named(frames, "message_start")[0]["message"]["usage"] == {"input_tokens": 17, "output_tokens": 0}
    usage = _named(frames, "message_delta")[0]["usage"]
    assert usage["input_tokens"] == 17
    assert usage["output_tokens"] == 5


def test_endpoint_stream_counts_system_prompt_once():
    system = "you are a helpful assistant with a long system prompt that repeats itself many times over"
    app.state.pool = make_pool(_ds_sse("Hello"))
    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/messages",
        json={"model": "deepseek-v4.1-flash", "max_tokens": 10, "stream": True, "system": system, "messages": [{"role": "user", "content": "hi"}]},
    ) as r:
        text = "".join(r.iter_text())
    client.close()
    reported = _named(_frames(text), "message_start")[0]["message"]["usage"]["input_tokens"]
    once = ant.count_input_tokens([{"role": "system", "content": system}, {"role": "user", "content": "hi"}], None)
    assert reported == once
    assert reported < once + ant.estimate_tokens(system)
