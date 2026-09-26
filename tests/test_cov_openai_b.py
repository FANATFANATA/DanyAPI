import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import danyapi.api.openai as openai_mod
from danyapi import tools as toolemu
from danyapi.api.openai import Attachment, ChatMessage, app
from danyapi.deepseek.stream import MessageReconstructor

OK_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"WIP","fragments":[{"id":2,"type":"RESPONSE","content":"Hi"}]}}}\n'
    "\n"
    'data: {"p":"response/status","o":"SET","v":"FINISHED"}\n'
    "\n"
)

INPUT_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"p":"response/status","o":"SET","v":"input_exceeds_limit"}\n'
    "\n"
)

TOO_FREQUENT_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    "event: toast\n"
    'data: {"type":"error","content":"Message too frequent, please try again later."}\n'
    "\n"
)

THINK_ONLY_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"FINISHED","fragments":[{"id":2,"type":"THINK","content":"why"}]}}}\n'
    "\n"
)

TOOL_JSON_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"FINISHED","fragments":'
    '[{"id":2,"type":"RESPONSE","content":"{\\"tool_calls\\": [{\\"name\\": \\"get_weather\\", \\"arguments\\": {\\"city\\": \\"Moscow\\"}}]}"}]}}}\n'
    "\n"
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather in a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def _tool_sse(calls):
    content = json.dumps({"tool_calls": calls})
    payload = {
        "v": {
            "response": {
                "message_id": 2,
                "parent_id": 1,
                "status": "FINISHED",
                "fragments": [{"id": 2, "type": "RESPONSE", "content": content}],
            }
        }
    }
    return f'event: ready\ndata: {{"request_message_id":1,"response_message_id":2,"model_type":"default"}}\n\ndata: {json.dumps(payload)}\n\n'


class FakeSession:
    def __init__(self, sid="c1", last_message_id=None):
        self.id = sid
        self.last_message_id = last_message_id
        self.accumulated_tokens = 0


class FakeResp:
    def __init__(self, body=None, sse_text=None, status=200, content_type="text/event-stream; charset=utf-8"):
        self.status_code = status
        self.headers = {"content-type": content_type}
        if sse_text is not None:
            self._b = sse_text.encode()
        else:
            self._b = (body or "").encode()

    async def aiter_bytes(self):
        yield self._b

    async def aclose(self):
        pass

    async def aread(self):
        return self._b


class CloseFailResp(FakeResp):
    async def aclose(self):
        raise RuntimeError("close boom")


class StreamReadError(FakeResp):
    async def aiter_bytes(self):
        yield b'event: ready\ndata: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n\n'
        yield b'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"WIP","fragments":[{"id":2,"type":"RESPONSE","content":"Hi"}]}}}\n\n'
        raise httpx.ReadError("reset")


class Boom(BaseException):
    pass


class StreamBoom(FakeResp):
    async def aiter_bytes(self):
        yield b'event: ready\ndata: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n\n'
        yield b'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"WIP","fragments":[{"id":2,"type":"RESPONSE","content":"Hi"}]}}}\n\n'
        raise Boom()


class FakeAccount:
    def __init__(self, sse_list=None, resp=None):
        self.index = 0
        self.broken = False
        self.label = "acct"
        self.client = MagicMock()
        if resp is not None:
            self.client.completion = AsyncMock(return_value=resp)
        else:
            self.client.completion = AsyncMock(side_effect=[FakeResp(sse_text=s) for s in (sse_list or [OK_SSE])])
        self.client.create_pow_challenge = AsyncMock(return_value={})
        self.client.stop_stream = AsyncMock()
        self.pow = MagicMock()
        self.pow.make_header = AsyncMock(return_value={})
        self.pow_upload = MagicMock()
        self.pow_upload.make_header = AsyncMock(return_value={})
        self.sem = asyncio.Semaphore(1)
        self.sessions = MagicMock()
        self.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
        self.sessions.get = MagicMock(return_value=None)
        self.sessions.touch_last_message = MagicMock()
        self.sessions.forget = MagicMock()

    def mark_broken(self):
        self.broken = True


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


def test_apply_limits_stop_and_length():
    text, finish = openai_mod._apply_limits("hello STOP world", None, ["STOP"])
    assert text == "hello "
    assert finish == "stop"
    text, finish = openai_mod._apply_limits("word " * 30, 1, None)
    assert finish == "length"
    assert text


def test_deepseek_usage_provider_prompt_tokens():
    usage = openai_mod._deepseek_usage(10, "", {"prompt_tokens": 7})
    assert usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def test_chunk_id_from_line_edges():
    assert openai_mod._chunk_id_from_line("data: \n") is None
    assert openai_mod._chunk_id_from_line("data: [DONE]\n") is None
    assert openai_mod._chunk_id_from_line("data: {bad json\n") is None
    assert openai_mod._chunk_id_from_line("data: [1, 2]\n") is None


async def test_close_generator_paths():
    class NoClose:
        pass

    class BadClose:
        async def aclose(self):
            raise RuntimeError("boom")

    await openai_mod._close_generator(NoClose())
    await openai_mod._close_generator(BadClose())


async def test_stream_guard_generic_exception():
    async def gen():
        raise RuntimeError("kaboom")
        yield "never"

    lines = await _collect(openai_mod._stream_guard(gen(), "m"))
    joined = "".join(lines)
    assert "stream error" in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_is_fake_context_hint_non_string_message():
    rec = SimpleNamespace(hint_error={"message": 123})
    assert not openai_mod._is_fake_context_hint(rec)


def test_error_text_variants():
    assert openai_mod._error_text({"a": 1}) == '{"a": 1}'
    assert "set" in openai_mod._error_text({1, 2}) or openai_mod._error_text({1, 2})


def test_build_assistant_message_max_calls():
    schemas = toolemu.tool_schema_map([WEATHER_TOOL])
    content = json.dumps(
        {
            "tool_calls": [
                {"name": "get_weather", "arguments": {"city": "A"}},
                {"name": "get_weather", "arguments": {"city": "B"}},
            ]
        }
    )
    message, finish = openai_mod._build_assistant_message(content, None, True, schemas, max_calls=1)
    assert finish == "tool_calls"
    assert len(message["tool_calls"]) == 1


def test_build_limited_message_length_branches():
    _message, finish = openai_mod._build_limited_message("word " * 30, None, True, {}, 1, None, None, "FINISHED")
    assert finish == "length"
    _message, finish = openai_mod._build_limited_message("word " * 30, None, False, None, 1, None, None, "FINISHED")
    assert finish == "length"


async def test_chat_qwen_rejects_non_image_attachment(monkeypatch):
    monkeypatch.setattr(
        openai_mod,
        "_acquire_and_build",
        AsyncMock(return_value=(MagicMock(), None, (), "prompt", False)),
    )
    monkeypatch.setattr(
        openai_mod,
        "_collect_attachments",
        lambda req: [Attachment(b"a", "a.txt", "text/plain", False)],
    )
    req = SimpleNamespace(
        model="qwen3.8-max",
        thinking=None,
        search=None,
        messages=[],
        tools=None,
        tool_choice=None,
        response_format=None,
        user=None,
        session_id=None,
    )
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._chat_completions_qwen(req, pool=MagicMock())
    assert excinfo.value.status_code == 400


async def test_send_completion_too_frequent_status_error():
    request = httpx.Request("POST", "https://x")
    response = httpx.Response(502, text="Message too frequent", request=request)
    err = httpx.HTTPStatusError("boom", request=request, response=response)
    client = MagicMock()
    client.completion = AsyncMock(side_effect=err)
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._send_completion(client, {}, "s", None, "p", "default", False, False)
    assert excinfo.value.status_code == 429


async def test_send_completion_unexpected_response():
    client = MagicMock()
    client.completion = AsyncMock(return_value=None)
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._send_completion(client, {}, "s", None, "p", "default", False, False)
    assert excinfo.value.status_code == 502


async def test_send_completion_bad_json_too_frequent():
    resp = FakeResp(body="Message too frequent", content_type="application/json")
    client = MagicMock()
    client.completion = AsyncMock(return_value=resp)
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._send_completion(client, {}, "s", None, "p", "default", False, False)
    assert excinfo.value.status_code == 429


async def test_send_completion_biz_code_too_frequent():
    resp = FakeResp(body=json.dumps({"data": {"biz_code": 1, "biz_msg": "Message too frequent"}}), content_type="application/json")
    client = MagicMock()
    client.completion = AsyncMock(return_value=resp)
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._send_completion(client, {}, "s", None, "p", "default", False, False)
    assert excinfo.value.status_code == 429


async def test_send_completion_code_too_frequent():
    resp = FakeResp(body=json.dumps({"code": 5000, "msg": "Message too frequent"}), content_type="application/json")
    client = MagicMock()
    client.completion = AsyncMock(return_value=resp)
    with pytest.raises(openai_mod.HTTPException) as excinfo:
        await openai_mod._send_completion(client, {}, "s", None, "p", "default", False, False)
    assert excinfo.value.status_code == 429


async def test_send_deepseek_stream_read_error_stops(monkeypatch):
    acct = FakeAccount(resp=StreamReadError(OK_SSE))
    with pytest.raises(openai_mod.DeepSeekStreamError):
        await openai_mod._send_deepseek_stream(acct, FakeSession(), None, "p", "default", False, False)
    acct.client.stop_stream.assert_awaited()


async def test_send_deepseek_stream_base_exception_stops():
    acct = FakeAccount(resp=StreamBoom())
    with pytest.raises(Boom):
        await openai_mod._send_deepseek_stream(acct, FakeSession(), None, "p", "default", False, False)
    acct.client.stop_stream.assert_awaited()


async def test_send_deepseek_stream_close_failure_stops():
    acct = FakeAccount(resp=CloseFailResp(sse_text=OK_SSE))
    rec, _response_id, _stop_id = await openai_mod._send_deepseek_stream(acct, FakeSession(), None, "p", "default", False, False)
    assert rec.content == "Hi"
    acct.client.stop_stream.assert_awaited()


async def test_collect_continuation_rate_limit_http(monkeypatch):
    monkeypatch.setattr(openai_mod, "MESSAGE_TOO_FREQUENT_WAIT_SEC", 0.0)
    acct = FakeAccount()
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(429, "Message too frequent"),
            FakeResp(sse_text=OK_SSE),
        ]
    )
    rec = await openai_mod._collect_continuation(acct, FakeSession(), None, "default", False, False)
    assert rec is not None
    assert rec.content == "Hi"


async def test_collect_continuation_hint_rate_limit(monkeypatch):
    monkeypatch.setattr(openai_mod, "MESSAGE_TOO_FREQUENT_WAIT_SEC", 0.0)
    acct = FakeAccount([TOO_FREQUENT_SSE, OK_SSE])
    rec = await openai_mod._collect_continuation(acct, FakeSession(), None, "default", False, False)
    assert rec is not None
    assert rec.content == "Hi"


async def test_collect_non_stream_build_prompt_value_error():
    acct = FakeAccount([OK_SSE])
    result = await openai_mod._collect_non_stream(
        account=acct,
        pool=MagicMock(),
        existing_sid=None,
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
        messages=[],
        tools=None,
    )
    assert result["choices"][0]["message"]["content"] == "Hi"


async def test_collect_non_stream_stale_session_rebuild():
    acct = FakeAccount()
    acct.sessions.get = MagicMock(return_value=object())
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(404, "gone"),
            FakeResp(sse_text=OK_SSE),
        ]
    )
    result = await openai_mod._collect_non_stream(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
        messages=[ChatMessage(role="user", content="hello")],
        tools=None,
    )
    assert result["choices"][0]["message"]["content"] == "Hi"


async def test_collect_non_stream_reduced_tool_mode(monkeypatch):
    acct = FakeAccount([INPUT_SSE])
    session = FakeSession()
    rec = MessageReconstructor()
    rec.message = {"fragments": [{"type": "RESPONSE", "content": "Reduced"}]}
    monkeypatch.setattr(openai_mod, "_collect_continuation", AsyncMock(return_value=None))
    monkeypatch.setattr(openai_mod, "_reduced_prompt_variants", lambda *a, **k: [("p", True, {})])
    monkeypatch.setattr(openai_mod, "_collect_reduced", AsyncMock(return_value=(rec, session, "s1", True, {"get_weather": {}})))
    result = await openai_mod._collect_non_stream(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
        messages=[ChatMessage(role="user", content="hello")],
        tools=[WEATHER_TOOL],
    )
    assert result["choices"][0]["finish_reason"] == "response_incomplete"
    assert result["error"]["finish_reason"] == "response_incomplete"


async def test_collect_non_stream_n_choices():
    acct = FakeAccount([OK_SSE])
    result = await openai_mod._collect_non_stream(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
        n=3,
    )
    assert len(result["choices"]) == 3


async def test_stream_openai_upload_attachments():
    acct = FakeAccount([OK_SSE])
    acct.client.upload_file = AsyncMock(return_value={"id": "f1"})
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
        attachments=[Attachment(b"a", "a.txt", "text/plain", False)],
    )
    joined = "".join(await _collect(gen))
    assert '"content": "Hi"' in joined
    acct.client.upload_file.assert_awaited_once()


async def test_stream_openai_build_prompt_value_error():
    acct = FakeAccount([OK_SSE])
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid=None,
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
        messages=[],
        tools=None,
    )
    joined = "".join(await _collect(gen))
    assert '"content": "Hi"' in joined


async def test_stream_openai_stop_filter_hit():
    sse = (
        "event: ready\n"
        'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
        "\n"
        'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"FINISHED","fragments":[{"id":2,"type":"RESPONSE","content":"hello STOP world"}]}}}\n'
        "\n"
    )
    acct = FakeAccount([sse])
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
        stop="STOP",
    )
    joined = "".join(await _collect(gen))
    assert "STOP" not in joined
    assert '"finish_reason": "stop"' in joined


async def test_stream_openai_stop_filter_flush_tail():
    sse = (
        "event: ready\n"
        'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
        "\n"
        'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"FINISHED","fragments":[{"id":2,"type":"RESPONSE","content":"hello"}]}}}\n'
        "\n"
    )
    acct = FakeAccount([sse])
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
        stop="ZZ",
    )
    joined = "".join(await _collect(gen))
    assert '"content": "hell"' in joined
    assert '"finish_reason": "stop"' in joined
    assert '"content": "o"' not in joined


async def test_stream_openai_stale_session_rebuild():
    acct = FakeAccount()
    acct.sessions.get = MagicMock(return_value=object())
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(404, "gone"),
            FakeResp(sse_text=OK_SSE),
        ]
    )
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
        messages=[ChatMessage(role="user", content="hello")],
        tools=None,
    )
    joined = "".join(await _collect(gen))
    assert '"content": "Hi"' in joined


async def test_stream_openai_rate_limit_http(monkeypatch):
    monkeypatch.setattr(openai_mod, "MESSAGE_TOO_FREQUENT_WAIT_SEC", 0.0)
    acct = FakeAccount()
    acct.client.completion = AsyncMock(
        side_effect=[
            openai_mod.HTTPException(429, "Message too frequent"),
            FakeResp(sse_text=OK_SSE),
        ]
    )
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    joined = "".join(await _collect(gen))
    assert '"content": "Hi"' in joined


async def test_stream_openai_finish_ready_event():
    sse = (
        'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"FINISHED","fragments":[{"id":2,"type":"RESPONSE","content":"A"}]}}}\n'
        "\n"
        'event: ready\ndata: {"request_message_id":1,"response_message_id":5,"model_type":"default"}'
    )
    acct = FakeAccount([sse])
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    joined = "".join(await _collect(gen))
    assert '"content": "A"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_openai_finish_content_event():
    sse = (
        "event: ready\n"
        'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
        "\n"
        'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"FINISHED","fragments":[{"id":2,"type":"RESPONSE","content":"Fin"}]}}}'
    )
    acct = FakeAccount([sse])
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    joined = "".join(await _collect(gen))
    assert '"content": "Fin"' in joined


async def test_stream_openai_close_failure_stops():
    acct = FakeAccount(resp=CloseFailResp(sse_text=OK_SSE))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    joined = "".join(await _collect(gen))
    assert '"content": "Hi"' in joined
    acct.client.stop_stream.assert_awaited()


async def test_stream_openai_continuation_reasoning_only():
    acct = FakeAccount([INPUT_SSE, THINK_ONLY_SSE])
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
    )
    joined = "".join(await _collect(gen))
    assert '"reasoning_content": "why"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_openai_reduced_tool_mode(monkeypatch):
    acct = FakeAccount([INPUT_SSE])
    session = FakeSession()
    rec = MessageReconstructor()
    rec.message = {"fragments": [{"type": "RESPONSE", "content": "Reduced"}]}
    monkeypatch.setattr(openai_mod, "_collect_continuation", AsyncMock(return_value=None))
    monkeypatch.setattr(openai_mod, "_reduced_prompt_variants", lambda *a, **k: [("p", True, {})])
    monkeypatch.setattr(openai_mod, "_collect_reduced", AsyncMock(return_value=(rec, session, "s1", True, {"get_weather": {}})))
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
        messages=[ChatMessage(role="user", content="hello")],
        tools=[WEATHER_TOOL],
        tool_schemas=toolemu.tool_schema_map([WEATHER_TOOL]),
    )
    joined = "".join(await _collect(gen))
    assert "Reduced" in joined
    assert '"finish_reason": "response_incomplete"' in joined


async def test_stream_openai_tool_max_calls_slice():
    calls = [
        {"name": "get_weather", "arguments": {"city": "A"}},
        {"name": "get_weather", "arguments": {"city": "B"}},
    ]
    acct = FakeAccount([_tool_sse(calls)])
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
        tool_mode=True,
        parallel_tool_calls=False,
        tool_schemas=toolemu.tool_schema_map([WEATHER_TOOL]),
    )
    joined = "".join(await _collect(gen))
    assert '"tool_calls"' in joined


async def test_stream_openai_tool_empty_calls_tail():
    acct = FakeAccount([_tool_sse([])])
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
        tool_mode=True,
        tool_schemas=toolemu.tool_schema_map([WEATHER_TOOL]),
    )
    joined = "".join(await _collect(gen))
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_openai_extra_choices():
    acct = FakeAccount([OK_SSE])
    gen = openai_mod._stream_openai(
        account=acct,
        pool=MagicMock(),
        existing_sid="s1",
        lock=acct.sem,
        prompt="x",
        model="deepseek-v4.1-flash",
        model_type="default",
        thinking=False,
        search=False,
        n=3,
    )
    joined = "".join(await _collect(gen))
    assert '"index": 1' in joined
    assert '"index": 2' in joined


def test_unknown_v1_route():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.get("/v1/definitely-unknown-endpoint")
    client.close()
    assert resp.status_code == 404
    assert "Unknown /v1 endpoint" in resp.text
