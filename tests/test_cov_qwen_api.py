import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import danyapi.qwen.api as qwen_api
from danyapi.qwen.client import QwenError
from danyapi.qwen.stream import QwenStreamReconstructor

OK_SSE = (
    'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1","response_index":"0"}} \n'
    "\n"
    'data: {"choices": [{"delta": {"role": "assistant", "content": "Hello", "phase": "answer", "status": "typing"}}], "response_id": "r1"}\n'
    "\n"
    'data: {"choices": [{"delta": {"role": "assistant", "content": " world", "phase": "answer", "status": "typing"}}], "response_id": "r1"}\n'
    "\n"
    'data: {"choices": [{"delta": {"content": "", "role": "assistant", "status": "finished", "phase": "answer"}}], "response_id": "r1"}\n'
    "\n"
)

BUSY_SSE = 'data: {"error": {"code": "Too_Many_Requests", "details": "please slow down"}, "response_id": "r1"}\n\n'

CTX_SSE = 'data: {"error": {"code": "ContextLengthExceeded", "details": "too long"}, "response_id": "r1"}\n\n'

AUTH_SSE = 'data: {"error": {"code": "unauthorized", "details": "bad token"}, "response_id": "r1"}\n\n'

THINK_FEED_SSE = (
    'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
    'data: {"choices": [{"delta": {"content": "step", "phase": "think"}}], "response_id": "r1"}\n\n'
    'data: {"choices": [{"delta": {"content": "", "status": "finished", "phase": "answer"}}], "response_id": "r1"}\n\n'
)

IMG_SSE = (
    'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
    'data: {"choices": [{"delta": {"content": "![img](https://cdn.qwenlm.ai/a.png)", "phase": "image"}}], "response_id": "r1"}\n\n'
    'data: {"choices": [{"delta": {"content": "", "status": "finished", "phase": "image"}}], "response_id": "r1"}\n\n'
)

TOOL_JSON = '{"tool_calls":[{"name":"get_weather","arguments":{"city":"Moscow"}}]}'

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]

TOOL_SCHEMAS = qwen_api.toolemu.tool_schema_map(TOOLS)


def _created() -> str:
    return 'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'


def _trail(body) -> str:
    return "data: " + json.dumps(body)


FINISH_ANSWER_SSE = _created() + _trail({"choices": [{"delta": {"content": "Hi", "phase": "answer"}}], "response_id": "r1"})

FINISH_THINK_SSE = _created() + _trail({"choices": [{"delta": {"content": "step", "phase": "think"}}], "response_id": "r1"})

FINISH_USAGE_SSE = _created() + _trail({"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}})

TOOLISH_SSE = _created() + _trail({"choices": [{"delta": {"content": "hi " + TOOL_JSON, "phase": "answer"}}], "response_id": "r1"})

TOOL_PREFIX_SSE = _created() + _trail({"choices": [{"delta": {"content": "Sure " + TOOL_JSON, "phase": "answer"}}], "response_id": "r1"})

COLLECT_TAIL_SSE = _created() + _trail({"choices": [{"delta": {"content": "Hi", "phase": "answer"}}], "response_id": "r1"})


class FakeSession:
    def __init__(self, sid: str = "c1", last_response_id: str | None = None) -> None:
        self.id = sid
        self.last_response_id = last_response_id
        self.accumulated_input_tokens = 0
        self.accumulated_output_tokens = 0


class FakeResp:
    def __init__(self, sse_text):
        self._b = sse_text.encode()
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream; charset=utf-8"}

    async def aiter_bytes(self):
        yield self._b

    async def aclose(self):
        pass

    async def aread(self):
        return self._b


class CloseErrorResp(FakeResp):
    async def aclose(self):
        raise RuntimeError("close boom")


class ErrorResp(FakeResp):
    async def aiter_bytes(self):
        yield b'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
        raise httpx.ReadError("connection reset")


class RawResp:
    def __init__(self, body: bytes, content_type: str = "application/json", status_code: int = 200):
        self._b = body
        self.status_code = status_code
        self.headers = {"content-type": content_type}

    async def aread(self):
        return self._b

    async def aclose(self):
        pass


class JsonResp:
    def __init__(self, body):
        self._b = body.encode()
        self.status_code = 200
        self.headers = {"content-type": "application/json"}

    async def aread(self):
        return self._b

    async def aclose(self):
        pass


class FakeAccount:
    def __init__(self, sse_list):
        self.index = 0
        self.broken = False
        self.client = MagicMock()
        self.client.completion = AsyncMock(side_effect=[FakeResp(s) for s in sse_list])
        self.sem = asyncio.Semaphore(1)
        self.sessions = MagicMock()
        self.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
        self.sessions.touch_last_message = MagicMock()
        self.sessions.forget = MagicMock()

    def mark_broken(self):
        self.broken = True


class _ImgMsg:
    def __init__(self, content):
        self.content = content


@pytest.fixture(autouse=True)
def zero_backoff():
    orig = qwen_api.RETRY_BACKOFF_SEC
    qwen_api.RETRY_BACKOFF_SEC = 0.0
    yield
    qwen_api.RETRY_BACKOFF_SEC = orig


def _args(acct, pool=None, existing_sid="s1", tool_mode=False, tool_schemas=None, **extra):
    args = {
        "account": acct,
        "pool": pool or MagicMock(),
        "existing_sid": existing_sid,
        "lock": acct.sem,
        "prompt": "x",
        "model": "qwen3.8-max",
        "model_id": "qwen3.8-max",
        "thinking": False,
        "search": False,
        "tool_mode": tool_mode,
    }
    if tool_schemas is not None:
        args["tool_schemas"] = tool_schemas
    args.update(extra)
    return args


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


def _rec_with(text, reasoning=None):
    rec = QwenStreamReconstructor()
    rec._content_parts.append(text)
    rec._content_joined_cache = None
    if reasoning is not None:
        rec._reasoning_parts.append(reasoning)
        rec._reasoning_joined_cache = None
    return rec


def test_retry_delay_variants():
    orig = qwen_api.RETRY_BACKOFF_SEC
    qwen_api.RETRY_BACKOFF_SEC = 1.0
    try:
        assert qwen_api._retry_delay(1) == 1.0
        assert qwen_api._retry_delay(10) == 8.0
    finally:
        qwen_api.RETRY_BACKOFF_SEC = orig


def test_is_retryable_http_variants():
    assert qwen_api._is_retryable_http(qwen_api.HTTPException(429, "x"))
    assert not qwen_api._is_retryable_http(qwen_api.HTTPException(418, "x"))


def test_append_image_markdown_branches():
    messages = [
        _ImgMsg(
            [
                {"type": "image_url", "image_url": {"nourl": 1}},
                {"type": "image_url", "image_url": 42},
                {"type": "image_url", "image_url": {"url": "ftp://x"}},
            ]
        )
    ]
    assert qwen_api._append_image_markdown("hello", messages) == "hello"


def test_append_image_markdown_empty_prompt():
    messages = [_ImgMsg([{"type": "image_url", "image_url": "https://x/y.png"}])]
    prompt = qwen_api._append_image_markdown("", messages)
    assert prompt == "![image](https://x/y.png)"


def test_is_context_limit_code_variants():
    assert qwen_api._is_context_limit_code("Context_Length_Exceeded")
    assert qwen_api._is_context_limit_code("TOKEN LIMIT")
    assert not qwen_api._is_context_limit_code("nope")


def test_error_status_variants():
    assert qwen_api._error_status("Too_Many_Requests") == 429
    assert qwen_api._error_status("Forbidden") == 401
    assert qwen_api._error_status("Other") == 502


def test_handle_account_error_branches():
    acct = FakeAccount([])
    acct.mark_broken = MagicMock()
    qwen_api._handle_account_error(acct, QwenError("unauthorized", "bad"))
    acct.mark_broken.assert_called_once_with()
    acct2 = FakeAccount([])
    acct2.mark_broken = MagicMock()
    qwen_api._handle_account_error(acct2, QwenError(500, "bad"))
    acct2.mark_broken.assert_not_called()


async def test_prepare_session_qwen_error():
    acct = FakeAccount([])
    acct.sessions.obtain = AsyncMock(side_effect=QwenError("unauthorized", "bad"))
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api._prepare_session(acct, MagicMock(), None, "m")
    assert excinfo.value.status_code == 401


async def test_prepare_session_index_context():
    acct = FakeAccount([])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(sid="new"), "new"))
    pool = MagicMock()
    _session, key = await qwen_api._prepare_session(acct, pool, "old", "m", ("u1",))
    assert key == "new"
    pool.index_context.assert_called_once_with("new", ("u1",))


async def test_prepare_session_no_pool_new_key():
    acct = FakeAccount([])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(sid="new"), "new"))
    _session, key = await qwen_api._prepare_session(acct, None, "old", "m")
    assert key == "new"
    acct.sessions.forget.assert_called_once_with("old")


async def test_send_completion_status_error():
    client = MagicMock()
    response = MagicMock(status_code=500, text="boom")
    client.completion = AsyncMock(side_effect=httpx.HTTPStatusError("x", request=MagicMock(), response=response))
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api._send_completion(client, FakeSession(), "p", "m", False, False)
    assert excinfo.value.status_code == 500


async def test_send_completion_http_error():
    client = MagicMock()
    client.completion = AsyncMock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api._send_completion(client, FakeSession(), "p", "m", False, False)
    assert excinfo.value.status_code == 502


async def test_send_completion_non_200():
    client = MagicMock()
    client.completion = AsyncMock(return_value=RawResp(b"err", "text/plain", 500))
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api._send_completion(client, FakeSession(), "p", "m", False, False)
    assert excinfo.value.status_code == 500


async def test_send_completion_waf_html():
    client = MagicMock()
    client.completion = AsyncMock(return_value=RawResp(b"<html>x</html>", "text/html", 200))
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api._send_completion(client, FakeSession(), "p", "m", False, False)
    assert excinfo.value.status_code == 502


async def test_send_completion_waf_requestinfo():
    client = MagicMock()
    client.completion = AsyncMock(return_value=RawResp(b"requestInfo", "application/json", 200))
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api._send_completion(client, FakeSession(), "p", "m", False, False)
    assert excinfo.value.status_code == 502


async def test_send_completion_invalid_json():
    client = MagicMock()
    client.completion = AsyncMock(return_value=RawResp(b"not json", "application/json", 200))
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api._send_completion(client, FakeSession(), "p", "m", False, False)
    assert excinfo.value.status_code == 502


async def test_send_completion_error_dict_context():
    client = MagicMock()
    client.completion = AsyncMock(return_value=JsonResp('{"error": {"code": "ContextLengthExceeded"}}'))
    with pytest.raises(qwen_api.ContextLimitError):
        await qwen_api._send_completion(client, FakeSession(), "p", "m", False, False)


async def test_send_completion_error_dict_non_context():
    client = MagicMock()
    client.completion = AsyncMock(return_value=JsonResp('{"error": {"code": "Too_Many_Requests", "details": "x"}}'))
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api._send_completion(client, FakeSession(), "p", "m", False, False)
    assert excinfo.value.status_code == 429


async def test_send_completion_data_context():
    client = MagicMock()
    client.completion = AsyncMock(return_value=JsonResp('{"data": {"code": "TokenLimit"}}'))
    with pytest.raises(qwen_api.ContextLimitError):
        await qwen_api._send_completion(client, FakeSession(), "p", "m", False, False)


async def test_send_completion_data_non_context():
    client = MagicMock()
    client.completion = AsyncMock(return_value=JsonResp('{"data": {"code": "quotaLimited", "details": "x"}}'))
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api._send_completion(client, FakeSession(), "p", "m", False, False)
    assert excinfo.value.status_code == 429


async def test_send_completion_non_dict_payload():
    client = MagicMock()
    client.completion = AsyncMock(return_value=JsonResp("[]"))
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api._send_completion(client, FakeSession(), "p", "m", False, False)
    assert excinfo.value.status_code == 502


def test_error_body_variants():
    rec = MagicMock()
    rec.error = {"code": "x", "details": "boom"}
    body = json.loads(qwen_api._error_body(rec))
    assert body["error"]["message"] == "boom"
    rec2 = MagicMock()
    rec2.error = {}
    body2 = json.loads(qwen_api._error_body(rec2))
    assert body2["error"]["code"] is None


def test_stream_error_lines_with_code():
    lines = list(qwen_api._stream_error_lines("id", 0, "m", "msg", "s", "code"))
    assert len(lines) == 2
    assert '"code": "code"' in lines[0]


def test_stream_context_limit_lines():
    lines = list(qwen_api._stream_context_limit_lines("id", 0, "m", "s"))
    assert lines[-1] == "data: [DONE]\n\n"


async def test_try_stop_stream_error():
    client = MagicMock()
    client.stop_stream = AsyncMock(side_effect=Exception("x"))
    await qwen_api._try_stop_stream(client, "c1", "r1")


async def test_human_delay_sleeps(monkeypatch):
    monkeypatch.setattr(qwen_api.random, "uniform", lambda a, b: 1.0)
    slept = []

    async def fake_sleep(value):
        slept.append(value)

    monkeypatch.setattr(qwen_api.asyncio, "sleep", fake_sleep)
    await qwen_api._human_delay()
    assert slept == [1.0]


def test_split_stop_variants():
    assert qwen_api._split_stop(None) == []
    assert qwen_api._split_stop("x") == ["x"]
    assert qwen_api._split_stop("") == []
    assert qwen_api._split_stop(["a", 1, "", "b"]) == ["a", "b"]
    assert qwen_api._split_stop(5) == []


def test_token_estimate_variants():
    assert qwen_api._token_estimate(5, 0) == 5
    assert qwen_api._token_estimate(0, 8) == 2


def test_trim_to_tokens_trims():
    assert qwen_api._trim_to_tokens("aaaa bbbb cccc", 2) == "aaaa bbbb"
    assert qwen_api._trim_to_tokens("hello", None) == "hello"
    assert qwen_api._trim_to_tokens("", 2) == ""


def test_apply_limits_stop_cut():
    text, finish = qwen_api._apply_limits("hello world", None, ["world"])
    assert text == "hello "
    assert finish == "stop"


def test_apply_limits_length_finish():
    text, finish = qwen_api._apply_limits("aaaa bbbb cccc", 2, None)
    assert text == "aaaa bbbb"
    assert finish == "length"


def test_build_limited_message_parallel_false():
    rec = _rec_with(TOOL_JSON)
    message, finish = qwen_api._build_limited_message(rec, True, TOOL_SCHEMAS, None, None, False)
    assert finish == "tool_calls"
    assert len(message["tool_calls"]) == 1


def test_build_limited_message_tail_trim():
    rec = _rec_with("prefix text " + TOOL_JSON)
    message, finish = qwen_api._build_limited_message(rec, True, TOOL_SCHEMAS, 1, None, None)
    assert finish == "length"
    assert message["content"] == "prefix"


def test_build_limited_message_fallback_stop_with_reasoning():
    rec = _rec_with("plain", "think")
    message, finish = qwen_api._build_limited_message(rec, True, None, None, None, None)
    assert finish == "stop"
    assert message["reasoning_content"] == "think"


def test_build_limited_message_plain_reasoning():
    rec = _rec_with("plain", "think")
    message, finish = qwen_api._build_limited_message(rec, False, None, None, None, None)
    assert finish == "stop"
    assert message["reasoning_content"] == "think"


async def test_collect_non_stream_context_limit_json():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=JsonResp('{"error": {"code": "ContextLengthExceeded"}}'))
    pool = MagicMock()
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api.collect_non_stream(**_args(acct, pool=pool))
    assert excinfo.value.status_code == 400
    pool.forget.assert_called_once_with("s1")


async def test_collect_non_stream_stale_build_error():
    acct = FakeAccount([])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(sid="new"), "new"))
    acct.client.completion = AsyncMock(side_effect=qwen_api.HTTPException(400, "stale"))
    args = _args(acct)
    args["messages"] = [{"role": "system", "content": "sys"}]
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api.collect_non_stream(**args)
    assert excinfo.value.status_code == 400


async def test_collect_non_stream_outer_build_value_error():
    acct = FakeAccount([OK_SSE])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(sid="new"), "new"))
    args = _args(acct)
    args["messages"] = [{"role": "system", "content": "sys"}]
    result = await qwen_api.collect_non_stream(**args)
    assert result["choices"][0]["message"]["content"] == "Hello world"


async def test_collect_non_stream_retryable_http():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(side_effect=[qwen_api.HTTPException(429, "slow"), FakeResp(OK_SSE)])
    result = await qwen_api.collect_non_stream(**_args(acct))
    assert result["choices"][0]["message"]["content"] == "Hello world"
    assert acct.client.completion.await_count == 2


async def test_collect_non_stream_retryable_stream_error():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(side_effect=[FakeResp(BUSY_SSE), FakeResp(OK_SSE)])
    result = await qwen_api.collect_non_stream(**_args(acct))
    assert result["choices"][0]["message"]["content"] == "Hello world"


async def test_collect_non_stream_finish_tail():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=FakeResp(COLLECT_TAIL_SSE))
    result = await qwen_api.collect_non_stream(**_args(acct))
    assert result["choices"][0]["message"]["content"] == "Hi"


async def test_collect_non_stream_stream_processing_error():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=ErrorResp(OK_SSE))
    acct.client.stop_stream = AsyncMock()
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api.collect_non_stream(**_args(acct))
    assert excinfo.value.status_code == 502


async def test_collect_non_stream_close_error():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=CloseErrorResp(OK_SSE))
    result = await qwen_api.collect_non_stream(**_args(acct))
    assert result["choices"][0]["message"]["content"] == "Hello world"


async def test_collect_non_stream_context_limit_rec():
    acct = FakeAccount([CTX_SSE])
    pool = MagicMock()
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api.collect_non_stream(**_args(acct, pool=pool))
    assert excinfo.value.status_code == 400
    pool.forget.assert_called_once_with("s1")


async def test_collect_non_stream_error_no_content():
    acct = FakeAccount([AUTH_SSE])
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api.collect_non_stream(**_args(acct))
    assert excinfo.value.status_code == 401


async def test_collect_non_stream_n_choices():
    acct = FakeAccount([OK_SSE])
    result = await qwen_api.collect_non_stream(**_args(acct, n=3))
    assert [c["index"] for c in result["choices"]] == [0, 1, 2]


async def test_collect_non_stream_cancel_stops_upstream():
    acct = FakeAccount([])
    started = asyncio.Event()

    class BlockingResp(FakeResp):
        async def aiter_bytes(self):
            yield b'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
            started.set()
            await asyncio.Event().wait()
            yield b""

    acct.client.completion = AsyncMock(return_value=BlockingResp(OK_SSE))
    acct.client.stop_stream = AsyncMock()
    task = asyncio.create_task(qwen_api.collect_non_stream(**_args(acct)))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    acct.client.stop_stream.assert_awaited_once()


async def test_stream_prepare_session_error():
    acct = FakeAccount([OK_SSE])
    acct.sessions.obtain = AsyncMock(side_effect=qwen_api.HTTPException(401, "bad"))
    gen = qwen_api.stream_openai(**_args(acct))
    joined = "".join(await _collect(gen))
    assert '"error"' in joined
    assert "bad" in joined


async def test_stream_outer_build_value_error():
    acct = FakeAccount([OK_SSE])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(sid="new"), "new"))
    args = _args(acct)
    args["messages"] = [{"role": "system", "content": "sys"}]
    joined = "".join(await _collect(qwen_api.stream_openai(**args)))
    assert '"content": "Hello"' in joined


async def test_stream_context_limit_json():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=JsonResp('{"error": {"code": "ContextLengthExceeded"}}'))
    pool = MagicMock()
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct, pool=pool))))
    assert '"finish_reason": "length"' in joined
    pool.forget.assert_called_once_with("s1")


async def test_stream_stale_build_error():
    acct = FakeAccount([])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(sid="new"), "new"))
    acct.client.completion = AsyncMock(side_effect=qwen_api.HTTPException(400, "stale"))
    args = _args(acct)
    args["messages"] = [{"role": "system", "content": "sys"}]
    joined = "".join(await _collect(qwen_api.stream_openai(**args)))
    assert '"error"' in joined


async def test_stream_stale_prepare_error():
    acct = FakeAccount([])
    acct.sessions.obtain = AsyncMock(side_effect=[(FakeSession(sid="new"), "new"), qwen_api.HTTPException(401, "prep bad")])
    acct.client.completion = AsyncMock(side_effect=qwen_api.HTTPException(400, "stale"))
    args = _args(acct)
    args["messages"] = [{"role": "user", "content": "hi"}]
    joined = "".join(await _collect(qwen_api.stream_openai(**args)))
    assert "prep bad" in joined


async def test_stream_stale_rebuild_success():
    acct = FakeAccount([])
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(sid="new"), "new"))
    acct.client.completion = AsyncMock(side_effect=[qwen_api.HTTPException(400, "stale"), FakeResp(OK_SSE)])
    args = _args(acct)
    args["messages"] = [{"role": "user", "content": "hi"}]
    joined = "".join(await _collect(qwen_api.stream_openai(**args)))
    assert '"content": "Hello"' in joined


async def test_stream_retryable_http():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(side_effect=[qwen_api.HTTPException(429, "slow"), FakeResp(OK_SSE)])
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct))))
    assert '"content": "Hello"' in joined


async def test_stream_401_marks_broken():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(side_effect=qwen_api.HTTPException(401, "bad"))
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct))))
    assert acct.broken
    assert '"error"' in joined


async def test_stream_retryable_error():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(side_effect=[FakeResp(BUSY_SSE), FakeResp(OK_SSE)])
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct))))
    assert '"content": "Hello"' in joined


async def test_stream_context_limit_rec():
    acct = FakeAccount([CTX_SSE])
    pool = MagicMock()
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct, pool=pool))))
    assert '"finish_reason": "length"' in joined
    pool.forget.assert_called_once_with("s1")


async def test_stream_error_no_content():
    acct = FakeAccount([AUTH_SSE])
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct))))
    assert '"error"' in joined
    assert "unauthorized" in joined


async def test_stream_usage_payload():
    acct = FakeAccount([OK_SSE])
    args = _args(acct, include_usage=True)
    joined = "".join(await _collect(qwen_api.stream_openai(**args)))
    assert '"usage"' in joined


async def test_stream_finish_reasoning_feed():
    acct = FakeAccount([THINK_FEED_SSE])
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct))))
    assert "reasoning_content" in joined


async def test_stream_disconnect_stops_upstream():
    acct = FakeAccount([OK_SSE])
    acct.client.stop_stream = AsyncMock()
    gen = qwen_api.stream_openai(**_args(acct))
    first = await gen.__anext__()
    assert first.startswith("data: ")
    await gen.aclose()
    acct.client.stop_stream.assert_awaited_once()


async def test_stream_close_error_stops_upstream():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=CloseErrorResp(OK_SSE))
    acct.client.stop_stream = AsyncMock()
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct))))
    assert '"content": "Hello"' in joined
    acct.client.stop_stream.assert_awaited_once()


async def test_stream_tool_delta_content():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=FakeResp(TOOL_PREFIX_SSE))
    args = _args(acct, tool_mode=True, tool_schemas=TOOL_SCHEMAS)
    joined = "".join(await _collect(qwen_api.stream_openai(**args)))
    assert '"content": "Sure "' in joined
    assert '"tool_calls"' in joined


async def test_stream_tool_parse_none_remainder():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=FakeResp(TOOLISH_SSE))
    args = _args(acct, tool_mode=True)
    joined = "".join(await _collect(qwen_api.stream_openai(**args)))
    assert "tool_calls" in joined
    assert '"finish_reason": "tool_calls"' in joined


async def test_stream_tool_empty_calls(monkeypatch):
    monkeypatch.setattr(qwen_api.toolemu, "parse_tool_calls", lambda *a, **k: ([], "wrapper"))
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=FakeResp(TOOLISH_SSE))
    args = _args(acct, tool_mode=True)
    joined = "".join(await _collect(qwen_api.stream_openai(**args)))
    assert "tool_calls" in joined


async def test_stream_empty_response_role_delta():
    sse = (
        'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
        'data: {"choices": [{"delta": {"content": "", "role": "assistant", "status": "finished", "phase": "answer"}}], "response_id": "r1"}\n\n'
    )
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=FakeResp(sse))
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct))))
    assert '"role": "assistant"' in joined
    assert '"finish_reason": "stop"' in joined


async def test_collect_image_success():
    acct = FakeAccount([IMG_SSE])
    args = {
        "account": acct,
        "pool": MagicMock(),
        "existing_sid": "s1",
        "lock": acct.sem,
        "prompt": "a cat",
        "model": "qwen-image-gen",
        "model_id": "qwen-image-gen",
    }
    result = await qwen_api.collect_image(**args)
    assert result["image_urls"] == ["https://cdn.qwenlm.ai/a.png"]
    assert result["session_id"] == "s1"


async def test_collect_image_retry_then_success():
    acct = FakeAccount([BUSY_SSE, IMG_SSE])
    args = {
        "account": acct,
        "pool": MagicMock(),
        "existing_sid": "s1",
        "lock": acct.sem,
        "prompt": "a cat",
        "model": "qwen-image-gen",
        "model_id": "qwen-image-gen",
    }
    result = await qwen_api.collect_image(**args)
    assert result["image_urls"] == ["https://cdn.qwenlm.ai/a.png"]


async def test_collect_image_context_limit():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=JsonResp('{"error": {"code": "ContextLengthExceeded"}}'))
    pool = MagicMock()
    args = {
        "account": acct,
        "pool": pool,
        "existing_sid": "s1",
        "lock": acct.sem,
        "prompt": "a cat",
        "model": "qwen-image-gen",
        "model_id": "qwen-image-gen",
    }
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api.collect_image(**args)
    assert excinfo.value.status_code == 400


async def test_collect_image_auth_error():
    acct = FakeAccount([AUTH_SSE])
    args = {
        "account": acct,
        "pool": MagicMock(),
        "existing_sid": "s1",
        "lock": acct.sem,
        "prompt": "a cat",
        "model": "qwen-image-gen",
        "model_id": "qwen-image-gen",
    }
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api.collect_image(**args)
    assert excinfo.value.status_code == 401


async def test_collect_image_stream_error():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=ErrorResp(IMG_SSE))
    acct.client.stop_stream = AsyncMock()
    args = {
        "account": acct,
        "pool": MagicMock(),
        "existing_sid": "s1",
        "lock": acct.sem,
        "prompt": "a cat",
        "model": "qwen-image-gen",
        "model_id": "qwen-image-gen",
    }
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api.collect_image(**args)
    assert excinfo.value.status_code == 502
