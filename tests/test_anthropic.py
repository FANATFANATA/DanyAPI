import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import danyapi.api.deepseek as deepseek_mod
import danyapi.api.openai as openai_mod
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


@pytest.fixture(autouse=True)
def clean_state():
    saved = (getattr(app.state, "pool", None), getattr(app.state, "qwen_pool", None))
    app.state.pool = None
    app.state.qwen_pool = None
    yield
    app.state.pool, app.state.qwen_pool = saved


@pytest.fixture(autouse=True)
def zero_backoff():
    orig = openai_mod.RETRY_BACKOFF_SEC
    deepseek_mod.RETRY_BACKOFF_SEC = 0.0
    yield
    deepseek_mod.RETRY_BACKOFF_SEC = orig


def _info(model="claude-sonnet-4-5"):
    return ant.RequestInfo(model=model, upstream_model="default", max_tokens=100)


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
    assert messages[0]["tool_calls"][0]["id"] == "tu1"
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert messages[1] == {"role": "tool", "tool_call_id": "tu1", "content": "42"}


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


def test_convert_stop_sequences_limit():
    with pytest.raises(ant.AnthropicInputError):
        ant.convert_stop_sequences(["a", "b", "c", "d", "e"])


def test_as_max_tokens():
    assert ant.as_max_tokens(None) == ant.DEFAULT_MAX_TOKENS
    assert ant.as_max_tokens(64) == 64
    with pytest.raises(ant.AnthropicInputError):
        ant.as_max_tokens(0)
    with pytest.raises(ant.AnthropicInputError):
        ant.as_max_tokens("x")


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


def test_build_content_empty_falls_back():
    assert ant.build_content({"message": {"content": ""}}) == [{"type": "text", "text": ""}]


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
    assert message["stop_reason"] == "end_turn"
    assert message["stop_sequence"] is None
    assert message["usage"] == {"input_tokens": 4, "output_tokens": 2}


def test_build_message_error_maps_to_end_turn():
    message = ant.build_message(_info(), "msg_1", {"error": {"message": "boom"}, "choices": []})
    assert message["stop_reason"] == "end_turn"


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
    assert starts[0]["content_block"]["type"] == "thinking"
    deltas = _named(frames, "content_block_delta")
    assert deltas[0]["delta"] == {"type": "thinking_delta", "thinking": "think"}
    assert deltas[1]["delta"] == {"type": "text_delta", "text": "Hi"}


def test_translate_stream_tool_use_block():
    calls = [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": '{"a":1}'}}]
    text = _chat_sse(finish="tool_calls", tool_calls=calls)

    async def run():
        stream = ant.translate_stream(_agen_chunks(text), _info(), "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    starts = _named(frames, "content_block_start")
    assert starts[0]["content_block"] == {"type": "tool_use", "id": "c1", "name": "f", "input": {}}
    deltas = _named(frames, "content_block_delta")
    assert deltas[0]["delta"] == {"type": "input_json_delta", "partial_json": '{"a":1}'}
    assert _named(frames, "message_delta")[0]["delta"]["stop_reason"] == "tool_use"


def test_translate_stream_error_event():
    text = 'data: {"error":{"message":"boom"}}\n\n'

    async def run():
        stream = ant.translate_stream(_agen_chunks(text), _info(), "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    errors = _named(frames, "error")
    assert errors[0]["error"] == {"type": "api_error", "message": "boom"}
    assert "message_stop" not in [event for event, _ in frames]


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

    asyncio.run(run())
    assert closed["value"] is True


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
    order = [(event, payload.get("index")) for event, payload in frames if event in ("content_block_start", "content_block_stop")]
    assert order == [
        ("content_block_start", 0),
        ("content_block_stop", 0),
        ("content_block_start", 1),
        ("content_block_stop", 1),
    ]


def test_translate_stream_closes_text_before_tool_use():
    calls = [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{}"}}]
    text = _chat_sse(reasoning="think", content="answer", tool_calls=calls, finish="tool_calls")

    async def run():
        stream = ant.translate_stream(_agen_chunks(text), _info(), "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    types = [payload["content_block"]["type"] for event, payload in frames if event == "content_block_start"]
    assert types == ["thinking", "text", "tool_use"]
    stopped = [payload["index"] for event, payload in frames if event == "content_block_stop"]
    assert stopped[:2] == [0, 1]


def test_translate_stream_keeps_known_input_tokens():
    chunk = json.dumps({"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 0, "completion_tokens": 5}})
    info = ant.RequestInfo(model="claude-sonnet-4-5", upstream_model="default", max_tokens=100, prompt_tokens=17)

    async def run():
        stream = ant.translate_stream(_agen_chunks(f"data: {chunk}\n\n"), info, "msg_1")
        return [line async for line in stream]

    frames = _frames("".join(asyncio.run(run())))
    usage = _named(frames, "message_delta")[0]["usage"]
    assert usage["input_tokens"] == 17
    assert usage["output_tokens"] == 5
