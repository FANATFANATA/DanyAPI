import asyncio
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import danyapi.api.retry as retry_mod
import danyapi.qwen.api as qwen_api
from danyapi.api import shaping
from danyapi.qwen.client import QwenError, QwenSession
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

THINK_SSE = (
    'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1","response_index":"0"}} \n'
    "\n"
    'data: {"choices": [{"delta": {"role": "assistant", "content": "", "phase": "thinking_summary",'
    ' "extra": {"summary_thought": {"content": ["Think step"]}}, "status": "typing"}}],'
    ' "response_id": "r1"}\n'
    "\n"
    'data: {"choices": [{"delta": {"content": "Answer", "phase": "answer", "status": "typing"}}], "response_id": "r1"}\n'
    "\n"
    'data: {"choices": [{"delta": {"content": "", "role": "assistant", "status": "finished", "phase": "answer"}}], "response_id": "r1"}\n'
    "\n"
)

BUSY_SSE = 'data: {"error": {"code": "Too_Many_Requests", "details": "please slow down"}, "response_id": "r1"}\n\n'

NUMERIC_CODE_SSE = 'data: {"error": {"code": 40014, "details": "numeric upstream failure"}, "response_id": "r1"}\n\n'

CTX_SSE = 'data: {"error": {"code": "ContextLengthExceeded", "details": "too long"}, "response_id": "r1"}\n\n'

TOOL_JSON = '{"tool_calls":[{"name":"get_weather","arguments":{"city":"Moscow"}}]}'

TOOL_SSE = (
    'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1","response_index":"0"}} \n'
    "\n"
    'data: {"choices": [{"delta": {"role": "assistant", "content": "", "phase": "thinking_summary",'
    ' "extra": {"summary_thought": {"content": ["Think step"]}}, "status": "typing"}}],'
    ' "response_id": "r1"}\n'
    "\n"
    f'data: {{"choices": [{{"delta": {{"content": {json.dumps(TOOL_JSON)}, "phase": "answer", "status": "typing"}}}}], "response_id": "r1"}}\n'
    "\n"
    'data: {"choices": [{"delta": {"content": "", "role": "assistant", "status": "finished", "phase": "answer"}}], "response_id": "r1"}\n'
    "\n"
)


class JsonResp:
    def __init__(self, body):
        self._b = body.encode()
        self.status_code = 200
        self.headers = {"content-type": "application/json"}

    async def aiter_bytes(self):
        yield self._b

    async def aclose(self):
        pass

    async def aread(self):
        return self._b


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

    def mark_broken(self):
        self.broken = True


@pytest.fixture(autouse=True)
def zero_backoff():
    orig = retry_mod.RETRY_BACKOFF_SEC
    retry_mod.RETRY_BACKOFF_SEC = 0.0
    yield
    retry_mod.RETRY_BACKOFF_SEC = orig


def _args(acct, pool=None, existing_sid: str | None = "s1", tool_mode=False, **extra):
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
    args.update(extra)
    return args


def _content_text(joined: str) -> str:
    parts: list[str] = []
    for match in re.finditer(r'"content": "((?:[^"\\]|\\.)*)"', joined):
        parts.append(json.loads(f'"{match.group(1)}"'))
    return "".join(parts)


def _payloads(joined: str) -> list[dict]:
    out: list[dict] = []
    for line in joined.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        out.append(json.loads(line[len("data: ") :]))
    return out


async def _send(resp):
    client = MagicMock()
    client.completion = AsyncMock(return_value=resp)
    session = FakeSession()
    return await qwen_api._send_completion(client, session, "p", "m", False, False)


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


async def test_non_stream_collects_content():
    acct = FakeAccount([OK_SSE])
    result = await qwen_api.collect_non_stream(**_args(acct))
    assert result["choices"][0]["message"]["content"] == "Hello world"
    assert result["session_id"] == "s1"
    assert result["choices"][0]["message"]["role"] == "assistant"
    acct.sessions.touch_last_message.assert_called_once_with("s1", "r1")


async def test_non_stream_collects_reasoning():
    acct = FakeAccount([THINK_SSE])
    result = await qwen_api.collect_non_stream(**_args(acct))
    message = result["choices"][0]["message"]
    assert message["content"] == "Answer"
    assert message["reasoning_content"] == "Think step"


async def test_non_stream_retries_then_success():
    acct = FakeAccount([BUSY_SSE, OK_SSE])
    result = await qwen_api.collect_non_stream(**_args(acct))
    assert result["choices"][0]["message"]["content"] == "Hello world"
    assert acct.client.completion.await_count == 2


async def test_non_stream_raises_429_after_retries():
    acct = FakeAccount([BUSY_SSE] * (qwen_api.MAX_RETRIES + 1))
    with pytest.raises(Exception) as excinfo:
        await qwen_api.collect_non_stream(**_args(acct))
    assert isinstance(excinfo.value, qwen_api.HTTPException)
    assert excinfo.value.status_code == 429


async def test_non_stream_rebuilds_prompt_for_replaced_session():
    acct = FakeAccount([OK_SSE])
    fresh = FakeSession()
    acct.sessions.obtain = AsyncMock(return_value=(fresh, "s1"))
    messages = [
        SimpleNamespace(role="user", content="earlier turn"),
        SimpleNamespace(role="assistant", content="earlier answer"),
        SimpleNamespace(role="user", content="now"),
    ]
    result = await qwen_api.collect_non_stream(**_args(acct, messages=messages, tools=None, tool_choice=None, cached_session=FakeSession()))
    assert result["choices"][0]["message"]["content"] == "Hello world"
    sent = acct.client.completion.await_args
    assert "earlier turn" in str(sent)


async def test_non_stream_keeps_delta_prompt_for_reused_session():
    acct = FakeAccount([OK_SSE])
    reused = FakeSession()
    acct.sessions.obtain = AsyncMock(return_value=(reused, "s1"))
    messages = [SimpleNamespace(role="user", content="earlier turn"), SimpleNamespace(role="user", content="now")]
    await qwen_api.collect_non_stream(**_args(acct, messages=messages, tools=None, tool_choice=None, cached_session=reused))
    sent = acct.client.completion.await_args
    assert "earlier turn" not in str(sent)


async def test_stream_rebuilds_prompt_for_replaced_session():
    acct = FakeAccount([OK_SSE])
    fresh = FakeSession()
    acct.sessions.obtain = AsyncMock(return_value=(fresh, "s1"))
    messages = [SimpleNamespace(role="user", content="earlier turn"), SimpleNamespace(role="user", content="now")]
    gen = qwen_api.stream_openai(**_args(acct, messages=messages, tools=None, tool_choice=None, cached_session=FakeSession()))
    await _collect(gen)
    sent = acct.client.completion.await_args
    assert "earlier turn" in str(sent)


async def test_stream_emits_error_event_after_retries():
    acct = FakeAccount([BUSY_SSE] * (qwen_api.MAX_RETRIES + 1))
    gen = qwen_api.stream_openai(**_args(acct))
    lines = await _collect(gen)
    assert acct.client.completion.await_count == qwen_api.MAX_RETRIES + 1
    joined = "".join(lines)
    assert "Too_Many_Requests" in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_success_streams_content():
    acct = FakeAccount([OK_SSE])
    gen = qwen_api.stream_openai(**_args(acct))
    lines = await _collect(gen)
    joined = "".join(lines)
    assert '"content": "Hello"' in joined
    assert '"content": " world"' in joined
    assert '"finish_reason": "stop"' in joined
    assert '"session_id": "s1"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_role_delta_emitted_once():
    acct = FakeAccount([OK_SSE])
    gen = qwen_api.stream_openai(**_args(acct))
    lines = await _collect(gen)
    joined = "".join(lines)
    assert joined.count('"role": "assistant"') == 1


async def test_stream_success_streams_reasoning():
    acct = FakeAccount([THINK_SSE])
    gen = qwen_api.stream_openai(**_args(acct))
    lines = await _collect(gen)
    joined = "".join(lines)
    assert "reasoning_content" in joined
    assert '"content": "Answer"' in joined


async def test_stream_emits_usage_when_requested():
    acct = FakeAccount([OK_SSE])
    args = _args(acct)
    args["include_usage"] = True
    gen = qwen_api.stream_openai(**args)
    lines = await _collect(gen)
    joined = "".join(lines)
    assert '"usage"' in joined
    assert '"completion_tokens"' in joined


async def test_new_session_registered():
    acct = FakeAccount([OK_SSE])
    pool = MagicMock()
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(sid="new1"), "new1"))
    await qwen_api.collect_non_stream(**_args(acct, pool=pool, existing_sid=None))
    pool.register.assert_called_once_with(0, "new1")


async def test_usage_reported():
    acct = FakeAccount([OK_SSE])
    result = await qwen_api.collect_non_stream(**_args(acct))
    assert "usage" in result


async def test_stream_emits_error_when_completion_fails():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(side_effect=httpx.ConnectError("boom"))
    gen = qwen_api.stream_openai(**_args(acct))
    lines = await _collect(gen)
    joined = "".join(lines)
    assert '"error"' in joined
    assert "Qwen request failed" in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_emits_error_when_json_error():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=JsonResp('{"ret":["FAIL_SYS_USER_VALIDATE"]}'))
    gen = qwen_api.stream_openai(**_args(acct))
    lines = await _collect(gen)
    joined = "".join(lines)
    assert '"error"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_non_stream_context_limit_drops_session_and_raises_400():
    acct = FakeAccount([CTX_SSE])
    pool = MagicMock()
    with pytest.raises(Exception) as excinfo:
        await qwen_api.collect_non_stream(**_args(acct, pool=pool))
    assert isinstance(excinfo.value, qwen_api.HTTPException)
    assert excinfo.value.status_code == 400
    pool.forget.assert_called_once_with("s1")
    pool.forget_context.assert_called_once_with("s1")
    acct.sessions.forget.assert_called_once_with("s1")


async def test_stream_context_limit_drops_session_and_emits_length():
    acct = FakeAccount([CTX_SSE])
    pool = MagicMock()
    gen = qwen_api.stream_openai(**_args(acct, pool=pool))
    lines = await _collect(gen)
    joined = "".join(lines)
    assert '"finish_reason": "length"' in joined
    assert "context length exceeded" in joined
    assert joined.rstrip().endswith("data: [DONE]")
    pool.forget.assert_called_once_with("s1")
    pool.forget_context.assert_called_once_with("s1")
    acct.sessions.forget.assert_called_once_with("s1")


async def test_non_stream_context_limit_json_raises_400():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=JsonResp('{"error": {"code": "ContextLengthExceeded", "details": "too long"}}'))
    pool = MagicMock()
    with pytest.raises(Exception) as excinfo:
        await qwen_api.collect_non_stream(**_args(acct, pool=pool))
    assert isinstance(excinfo.value, qwen_api.HTTPException)
    assert excinfo.value.status_code == 400
    pool.forget.assert_called_once_with("s1")
    pool.forget_context.assert_called_once_with("s1")
    acct.sessions.forget.assert_called_once_with("s1")


async def test_stream_context_limit_json_emits_length():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=JsonResp('{"error": {"code": "ContextLengthExceeded", "details": "too long"}}'))
    pool = MagicMock()
    gen = qwen_api.stream_openai(**_args(acct, pool=pool))
    lines = await _collect(gen)
    joined = "".join(lines)
    assert '"finish_reason": "length"' in joined
    assert joined.rstrip().endswith("data: [DONE]")
    pool.forget.assert_called_once_with("s1")
    pool.forget_context.assert_called_once_with("s1")
    acct.sessions.forget.assert_called_once_with("s1")


async def test_stream_disconnect_stops_upstream():
    acct = FakeAccount([OK_SSE])
    acct.client.stop_stream = AsyncMock()
    gen = qwen_api.stream_openai(**_args(acct))
    first = await gen.__anext__()
    assert first.startswith("data: ")
    await gen.aclose()
    acct.client.stop_stream.assert_awaited_once()
    args, _ = acct.client.stop_stream.call_args
    assert args[0] == "c1"
    assert args[1] == "r1"


async def test_non_stream_cancel_stops_upstream():
    acct = FakeAccount([])
    started = asyncio.Event()

    class BlockingResp(FakeResp):
        async def aiter_bytes(self):
            yield (b'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1","response_index":"0"}} \n\n')
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
    args, _ = acct.client.stop_stream.call_args
    assert args[0] == "c1"
    assert args[1] == "r1"


async def test_stream_full_consumption_does_not_stop_upstream():
    acct = FakeAccount([OK_SSE])
    acct.client.stop_stream = AsyncMock()
    gen = qwen_api.stream_openai(**_args(acct))
    lines = await _collect(gen)
    assert any(line.startswith("data: ") for line in lines)
    acct.client.stop_stream.assert_not_awaited()


async def test_non_stream_stream_error_stops_upstream():
    acct = FakeAccount([])

    class ErrorResp(FakeResp):
        async def aiter_bytes(self):
            yield b'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
            raise httpx.ReadError("connection reset")

    acct.client.completion = AsyncMock(return_value=ErrorResp(OK_SSE))
    acct.client.stop_stream = AsyncMock()
    with pytest.raises(Exception) as excinfo:
        await qwen_api.collect_non_stream(**_args(acct))
    assert isinstance(excinfo.value, qwen_api.HTTPException)
    assert excinfo.value.status_code == 502
    acct.client.stop_stream.assert_awaited()
    args, _ = acct.client.stop_stream.call_args
    assert args[0] == "c1"
    assert args[1] == "r1"


async def test_stream_stream_error_stops_upstream():
    acct = FakeAccount([])

    class ErrorResp(FakeResp):
        async def aiter_bytes(self):
            yield b'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
            raise httpx.ReadError("connection reset")

    acct.client.completion = AsyncMock(return_value=ErrorResp(OK_SSE))
    acct.client.stop_stream = AsyncMock()
    gen = qwen_api.stream_openai(**_args(acct))
    with pytest.raises(httpx.ReadError):
        await _collect(gen)
    acct.client.stop_stream.assert_awaited_once()
    args, _ = acct.client.stop_stream.call_args
    assert args[0] == "c1"
    assert args[1] == "r1"


def test_error_status():
    assert qwen_api._error_status("Too_Many_Requests") == 429
    assert qwen_api._error_status("RateLimited") == 429
    assert qwen_api._error_status("unauthorized") == 401
    assert qwen_api._error_status("Other") == 502
    assert qwen_api._error_status(None) == 502


def test_is_context_limit_code():
    assert qwen_api._is_context_limit_code("ContextLengthExceeded")
    assert qwen_api._is_context_limit_code("The input token limit is exceeded")
    assert not qwen_api._is_context_limit_code("Too_Many_Requests")
    assert not qwen_api._is_context_limit_code("")
    assert not qwen_api._is_context_limit_code(None)
    assert not qwen_api._is_context_limit_code(123)


def test_is_retryable_error():
    rec = MagicMock()
    rec.error = {"code": "Too_Many_Requests"}
    rec.has_content = False
    assert qwen_api._is_retryable_error(rec)
    rec2 = MagicMock()
    rec2.error = {"code": "Other"}
    rec2.has_content = False
    assert not qwen_api._is_retryable_error(rec2)
    rec3 = MagicMock()
    rec3.error = None
    assert not qwen_api._is_retryable_error(rec3)
    rec4 = MagicMock()
    rec4.error = {"code": "Too_Many_Requests"}
    rec4.has_content = True
    assert not qwen_api._is_retryable_error(rec4)


def test_error_detail():
    rec = MagicMock()
    rec.error = {"code": "x", "details": "boom"}
    body = qwen_api._error_detail(rec)
    assert body == "boom"
    assert isinstance(body, str)
    rec2 = MagicMock()
    rec2.error = {}
    body2 = qwen_api._error_detail(rec2)
    assert body2 == "Qwen server error, try again later"
    rec3 = MagicMock()
    rec3.error = {"message": "only message"}
    assert qwen_api._error_detail(rec3) == "only message"
    rec4 = MagicMock()
    rec4.error = None
    assert qwen_api._error_detail(rec4) == "Qwen server error, try again later"


def test_accumulate_usage():
    session = FakeSession()
    rec = QwenStreamReconstructor()
    rec.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    usage = qwen_api._accumulate_usage(session, rec)
    assert usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    usage2 = qwen_api._accumulate_usage(session, rec)
    assert usage2 == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert session.accumulated_input_tokens == 10
    assert session.accumulated_output_tokens == 5
    rec2 = QwenStreamReconstructor()
    rec2.usage = {"input_tokens": 13, "output_tokens": 8}
    usage3 = qwen_api._accumulate_usage(session, rec2)
    assert usage3 == {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}
    assert session.accumulated_input_tokens == 13
    assert session.accumulated_output_tokens == 8


def test_drop_session():
    pool = MagicMock()
    acct = FakeAccount([])
    acct.sessions.forget = MagicMock()
    qwen_api._drop_session(pool, acct, "s1")
    pool.forget.assert_called_once_with("s1")
    pool.forget_context.assert_called_once_with("s1")
    acct.sessions.forget.assert_called_once_with("s1")


def test_handle_account_error_auth():
    acct = FakeAccount([])
    acct.mark_broken = MagicMock()
    qwen_api._handle_account_error(acct, QwenError("unauthorized", "bad"))
    acct.mark_broken.assert_called_once_with()


def test_handle_account_error_other():
    acct = FakeAccount([])
    acct.mark_broken = MagicMock()
    qwen_api._handle_account_error(acct, QwenError(500, "bad"))
    acct.mark_broken.assert_not_called()


async def test_try_stop_stream():
    client = MagicMock()
    client.stop_stream = AsyncMock()
    await qwen_api._try_stop_stream(client, "c1", "r1")
    client.stop_stream.assert_awaited_once_with("c1", "r1")
    await qwen_api._try_stop_stream(client, "", "r1")
    await qwen_api._try_stop_stream(client, "c1", None)
    client.stop_stream.assert_awaited_once()


async def test_try_stop_stream_error():
    client = MagicMock()
    client.stop_stream = AsyncMock(side_effect=Exception("x"))
    await qwen_api._try_stop_stream(client, "c1", "r1")


async def test_auth_error_401():
    acct = FakeAccount([])
    acct.sessions.obtain = AsyncMock(side_effect=QwenError("unauthorized", "bad"))
    with pytest.raises(Exception) as excinfo:
        await qwen_api._prepare_session(acct, MagicMock(), None, "m")
    assert excinfo.value.status_code == 401
    assert acct.broken


async def test_other_error_502():
    acct = FakeAccount([])
    acct.sessions.obtain = AsyncMock(side_effect=QwenError(500, "bad"))
    with pytest.raises(Exception) as excinfo:
        await qwen_api._prepare_session(acct, MagicMock(), None, "m")
    assert excinfo.value.status_code == 502
    assert not acct.broken


async def test_prepare_session_new_session_registered():
    acct = FakeAccount([])
    acct.sessions.obtain = AsyncMock(return_value=(QwenSession(id="new"), "new"))
    pool = MagicMock()
    _session, key = await qwen_api._prepare_session(acct, pool, "old", "m", ("u1",))
    assert key == "new"
    pool.register.assert_called_once_with(0, "new")
    pool.index_context.assert_called_once_with("new", ("u1",))
    pool.forget.assert_not_called()
    acct.sessions.forget.assert_not_called()


async def test_existing_session_reused():
    acct = FakeAccount([])
    pool = MagicMock()
    _session, key = await qwen_api._prepare_session(acct, pool, "s1", "m")
    assert key == "s1"
    pool.register.assert_called_once_with(0, "s1")


async def test_json_error_dict():
    with pytest.raises(Exception) as excinfo:
        await _send(JsonResp('{"error": {"code": "Too_Many_Requests", "details": "slow"}}'))
    assert excinfo.value.status_code == 429


async def test_json_context_limit():
    with pytest.raises(qwen_api.ContextLimitError):
        await _send(JsonResp('{"error": {"code": "ContextLengthExceeded", "details": "long"}}'))


async def test_json_data_code():
    with pytest.raises(Exception) as excinfo:
        await _send(JsonResp('{"data": {"code": "quotaLimited", "details": "no quota"}}'))
    assert excinfo.value.status_code == 429


async def test_json_data_context_limit():
    with pytest.raises(qwen_api.ContextLimitError):
        await _send(JsonResp('{"data": {"code": "TokenLimit", "details": "long"}}'))


async def test_bad_json():
    with pytest.raises(Exception) as excinfo:
        await _send(JsonResp("not json"))
    assert excinfo.value.status_code == 502


async def test_waf_html():
    resp = JsonResp("<html>challenge</html>")
    resp.headers = {"content-type": "text/html"}
    with pytest.raises(Exception) as excinfo:
        await _send(resp)
    assert excinfo.value.status_code == 502
    assert "WAF" in str(excinfo.value.detail)


OPENAI_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter", "function_call"}
STREAM_ERROR_FINISH_REASON = "error"

TOOL_SCHEMA_LIST = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


async def test_stream_error_lines_carry_the_code_and_a_valid_finish_reason():
    lines = list(qwen_api._stream_error_lines("id", 0, "m", "slow down", "s", "Too_Many_Requests"))
    assert len(lines) == 2
    payload = json.loads(lines[0][len("data: ") :])
    assert payload["error"] == {"message": "slow down", "code": "Too_Many_Requests"}
    finish = payload["choices"][0]["finish_reason"]
    assert finish == STREAM_ERROR_FINISH_REASON
    assert finish != "Too_Many_Requests"
    assert lines[1] == "data: [DONE]\n\n"


async def test_stream_context_limit_finish_reason_is_an_openai_reason():
    lines = list(qwen_api._stream_context_limit_lines("id", 0, "m", "s"))
    payload = json.loads(lines[0][len("data: ") :])
    assert payload["error"]["code"] == "context_length_exceeded"
    assert payload["choices"][0]["finish_reason"] in OPENAI_FINISH_REASONS


async def test_stream_error_payload_never_reuses_the_upstream_code_as_finish_reason():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=FakeResp(BUSY_SSE))
    lines = await _collect(qwen_api.stream_openai(**_args(acct)))
    joined = "".join(lines)
    payloads = _payloads(joined)
    errors = [payload for payload in payloads if "error" in payload]
    assert errors
    assert errors[-1]["error"]["message"] == "please slow down"
    assert errors[-1]["error"]["code"] == "Too_Many_Requests"
    for payload in payloads:
        for choice in payload["choices"]:
            assert choice["finish_reason"] in OPENAI_FINISH_REASONS | {STREAM_ERROR_FINISH_REASON}
            assert choice["finish_reason"] != "Too_Many_Requests"
    assert joined.rstrip().endswith("data: [DONE]")


async def test_non_stream_error_detail_is_a_plain_string():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=FakeResp(BUSY_SSE))
    with pytest.raises(Exception) as excinfo:
        await qwen_api.collect_non_stream(**_args(acct))
    assert excinfo.value.status_code == 429
    assert excinfo.value.detail == "please slow down"
    assert isinstance(excinfo.value.detail, str)


async def test_stream_retry_backoff_does_not_hold_the_account_lock():
    orig = retry_mod.RETRY_BACKOFF_SEC
    retry_mod.RETRY_BACKOFF_SEC = 0.2
    try:
        order: list[str] = []
        state = {"n": 0}
        acct = FakeAccount([])
        acct.client.stop_stream = AsyncMock()

        async def completion(**kwargs):
            state["n"] += 1
            order.append(f"completion-{state['n']}")
            return FakeResp(BUSY_SSE if state["n"] == 1 else OK_SSE)

        async def probe():
            async with acct.sem:
                order.append("probe")

        acct.client.completion = AsyncMock(side_effect=completion)
        probe_task = asyncio.create_task(probe())
        lines = await _collect(qwen_api.stream_openai(**_args(acct)))
        probe_task.cancel()
        try:
            await probe_task
        except asyncio.CancelledError:
            pass
        assert _content_text("".join(lines)) == "Hello world"
        assert order.index("probe") < order.index("completion-2")
    finally:
        retry_mod.RETRY_BACKOFF_SEC = orig


async def test_stream_tail_is_emitted_after_the_account_lock_is_released():
    acct = FakeAccount([OK_SSE])
    lines = await _collect(qwen_api.stream_openai(**_args(acct)))
    assert lines[-1] == "data: [DONE]\n\n"
    assert '"finish_reason": "stop"' in lines[-2]
    assert not acct.sem.locked()


async def test_stream_retry_after_hidden_tool_content_does_not_resurrect_stale_text(monkeypatch):
    monkeypatch.setattr(qwen_api, "_is_retryable_error", lambda rec: True)
    stale = (
        'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
        'data: {"choices": [{"delta": {"role": "assistant", "content": '
        + json.dumps(TOOL_JSON)
        + ', "phase": "answer", "status": "typing"}}], "response_id": "r1"}\n\n'
        'data: {"error": {"code": "Too_Many_Requests", "details": "please slow down"}, "response_id": "r1"}\n\n'
    )
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(side_effect=[FakeResp(stale), FakeResp(OK_SSE)])
    args = _args(acct, tool_mode=True, tool_schemas=qwen_api.toolemu.tool_schema_map(TOOL_SCHEMA_LIST))
    lines = await _collect(qwen_api.stream_openai(**args))
    joined = "".join(lines)
    assert acct.client.completion.await_count == 2
    assert _content_text(joined) == "Hello world"
    assert "Moscow" not in joined


async def test_numeric_upstream_code_is_502_and_context_limit_code_is_400():
    numeric = FakeAccount([])
    numeric.client.completion = AsyncMock(return_value=FakeResp(NUMERIC_CODE_SSE))
    with pytest.raises(Exception) as excinfo:
        await qwen_api.collect_non_stream(**_args(numeric))
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "numeric upstream failure"

    limited = FakeAccount([])
    limited.client.completion = AsyncMock(return_value=FakeResp(CTX_SSE))
    with pytest.raises(Exception) as excinfo2:
        await qwen_api.collect_non_stream(**_args(limited))
    assert excinfo2.value.status_code == 400


async def test_http_status_error():
    client = MagicMock()
    client.completion = AsyncMock(side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock(status_code=500)))
    with pytest.raises(Exception) as excinfo:
        await qwen_api._send_completion(client, FakeSession(), "p", "m", False, False)
    assert excinfo.value.status_code == 502
    assert "Qwen request failed" in excinfo.value.detail


async def test_http_error():
    client = MagicMock()
    client.completion = AsyncMock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(Exception) as excinfo:
        await qwen_api._send_completion(client, FakeSession(), "p", "m", False, False)
    assert excinfo.value.status_code == 502


async def test_non_200():
    resp = FakeResp("data: x\n\n")
    resp.status_code = 429
    with pytest.raises(Exception) as excinfo:
        await _send(resp)
    assert excinfo.value.status_code == 429


async def test_collect_non_stream_retryable_http():
    acct = FakeAccount([OK_SSE])
    acct.client.completion = AsyncMock(
        side_effect=[
            qwen_api.HTTPException(429, "slow"),
            FakeResp(OK_SSE),
        ]
    )
    result = await qwen_api.collect_non_stream(**_args(acct))
    assert result["choices"][0]["message"]["content"] == "Hello world"
    assert acct.client.completion.await_count == 2


async def test_non_stream_tool_mode_falls_back_to_content():
    acct = FakeAccount([OK_SSE])
    args = _args(acct, tool_mode=True)
    result = await qwen_api.collect_non_stream(**args)
    choice = result["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    assert "Hello world" in choice["message"]["content"]


async def test_stream_tool_mode_falls_back_to_content():
    acct = FakeAccount([OK_SSE])
    args = _args(acct, tool_mode=True)
    gen = qwen_api.stream_openai(**args)
    lines = await _collect(gen)
    joined = "".join(lines)
    assert '"tool_calls"' not in joined
    assert '"content": "Hello"' in joined
    assert '"content": " world"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_tool_mode_streams_prefix_then_tool_deltas():
    sse = (
        'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1","response_index":"0"}} \n'
        "\n"
        'data: {"choices": [{"delta": {"content": "Sure, here: ", "phase": "answer", "status": "typing"}}], "response_id": "r1"}\n'
        "\n"
        f'data: {{"choices": [{{"delta": {{"content": {json.dumps(TOOL_JSON)}, "phase": "answer", "status": "typing"}}}}], "response_id": "r1"}}\n'
        "\n"
        'data: {"choices": [{"delta": {"content": "", "role": "assistant", "status": "finished", "phase": "answer"}}], "response_id": "r1"}\n'
        "\n"
    )
    acct = FakeAccount([sse])
    args = _args(acct, tool_mode=True)
    gen = qwen_api.stream_openai(**args)
    joined = "".join(await _collect(gen))
    assert '"content": "Sure, here: "' in joined
    assert '"tool_calls"' in joined
    assert json.dumps(TOOL_JSON) not in joined
    assert '"finish_reason": "tool_calls"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


class _ImgMsg:
    def __init__(self, content):
        self.content = content


def test_append_image_markdown():
    messages = [
        _ImgMsg("plain"),
        _ImgMsg([{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]),
        _ImgMsg([{"type": "image_url", "image_url": "data:image/png;base64,AAAA"}]),
        _ImgMsg([{"type": "image_url", "image_url": 42}]),
    ]
    prompt = qwen_api._append_image_markdown("hello", messages)
    assert prompt.startswith("hello")
    assert "![image](https://x/y.png)" in prompt
    assert "![image](data:image/png;base64,AAAA)" in prompt
    assert qwen_api._append_image_markdown("hello", None) == "hello"
    assert qwen_api._append_image_markdown("hello", [_ImgMsg("nope")]) == "hello"


REJECTED_IMAGE_URIS = [
    "",
    "httpx://x/y.png",
    "http:/x.png",
    "ftp://x/y.png",
    "javascript:alert(1)",
    "data:image/png;base64,",
    "data:image/png,AAAA",
    "data:image/png;base64,AA*A",
    "data:image/png;base64,AAA",
    "https://x/a b.png",
    "https://x/a\tb.png",
    "https://x/a\nb.png",
    "https://x/a(b).png",
    "https://x/<script>.png",
    "![x](https://evil/y.png)",
    "![image](https://evil/y.png)",
]


def test_image_uri_validation_rejects_injection():
    for uri in REJECTED_IMAGE_URIS:
        assert not qwen_api._valid_image_uri(uri), uri
        messages = [_ImgMsg([{"type": "image_url", "image_url": {"url": uri}}])]
        assert qwen_api._append_image_markdown("hello", messages) == "hello", uri


def test_image_uri_validation_accepts_real_uris():
    for uri in ("http://x/y.png", "https://x/y.png?q=1#a", "data:image/png;base64,QUJDRA==", "data:image/jpeg;base64,AAAA"):
        assert qwen_api._valid_image_uri(uri), uri
        messages = [_ImgMsg([{"type": "image_url", "image_url": {"url": uri}}])]
        assert qwen_api._append_image_markdown("hi", messages) == f"hi\n\n![image]({uri})", uri


def test_append_image_markdown_keeps_order_and_dedup():
    messages = [
        _ImgMsg([{"type": "image_url", "image_url": "https://x/b.png"}, {"type": "image_url", "image_url": "https://x/a.png"}]),
        _ImgMsg([{"type": "image_url", "image_url": "https://x/b.png"}]),
    ]
    prompt = qwen_api._append_image_markdown("", messages)
    assert prompt == "![image](https://x/b.png)\n![image](https://x/a.png)"


def test_append_image_markdown_skips_already_present_tag():
    messages = [_ImgMsg([{"type": "image_url", "image_url": "https://x/a.png"}])]
    prompt = qwen_api._append_image_markdown("see ![image](https://x/a.png) here", messages)
    assert prompt == "see ![image](https://x/a.png) here"


async def test_stream_stop_truncates_content():
    sse = (
        'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
        'data: {"choices": [{"delta": {"content": "Hello wor", "phase": "answer"}}], "response_id": "r1"}\n\n'
        'data: {"choices": [{"delta": {"content": "ld STOP tail", "phase": "answer"}}], "response_id": "r1"}\n\n'
        'data: {"choices": [{"delta": {"status": "finished", "phase": "answer"}}], "response_id": "r1"}\n\n'
    )
    acct = FakeAccount([sse])
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct, stop="STOP"))))
    assert _content_text(joined) == "Hello world "
    assert '"finish_reason": "stop"' in joined


async def test_stream_stop_keeps_prefix_before_marker():
    sse = (
        'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
        'data: {"choices": [{"delta": {"content": "one<END>", "phase": "answer"}}], "response_id": "r1"}\n\n'
        'data: {"choices": [{"delta": {"content": "two", "phase": "answer"}}], "response_id": "r1"}\n\n'
    )
    acct = FakeAccount([sse])
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct, stop=["<END>"]))))
    assert _content_text(joined) == "one"
    assert '"finish_reason": "stop"' in joined


async def test_stream_stop_marker_split_across_chunks():
    sse = (
        'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
        'data: {"choices": [{"delta": {"content": "aaa<HAL", "phase": "answer"}}], "response_id": "r1"}\n\n'
        'data: {"choices": [{"delta": {"content": "T>bbb", "phase": "answer"}}], "response_id": "r1"}\n\n'
    )
    acct = FakeAccount([sse])
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct, stop="<HALT>"))))
    assert _content_text(joined) == "aaa"
    assert '"finish_reason": "stop"' in joined


async def test_stream_stop_absent_marker_keeps_content():
    acct = FakeAccount([OK_SSE])
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct, stop="NOTPRESENT"))))
    assert _content_text(joined) == "Hello world"
    assert '"finish_reason": "stop"' in joined


async def test_stream_choices_n_expands_index():
    acct = FakeAccount([OK_SSE])
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct, n=3))))
    assert '"index": 0' in joined
    assert '"index": 1' in joined
    assert '"index": 2' in joined
    assert '"index": 3' not in joined
    assert '"content": "Hello world"' in joined
    assert joined.count('"index": 2') == 2


async def test_stream_choices_n_bounded():
    acct = FakeAccount([OK_SSE])
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct, n=99))))
    assert f'"index": {shaping.MAX_STREAM_CHOICES - 1}' in joined
    assert f'"index": {shaping.MAX_STREAM_CHOICES}' not in joined


async def test_stream_choices_n_one_stays_single():
    acct = FakeAccount([OK_SSE])
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct, n=1))))
    assert '"index": 1' not in joined


async def test_stream_choices_n_empty_text_skips_delta():
    sse = (
        'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
        'data: {"choices": [{"delta": {"content": "", "role": "assistant", "status": "finished", "phase": "answer"}}], "response_id": "r1"}\n\n'
    )
    acct = FakeAccount([sse])
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct, n=2))))
    assert '"index": 1' in joined
    assert '"content"' not in joined


async def test_non_stream_rejects_injected_image_uri():
    acct = FakeAccount([OK_SSE])
    args = _args(acct)
    args["messages"] = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/a.png) evil ![x](https://evil/y"}}]}]
    await qwen_api.collect_non_stream(**args)
    sent = acct.client.completion.await_args.kwargs["prompt"]
    assert "![image](https://x/a.png) evil" not in sent
    assert "https://evil/y" not in sent


async def test_stream_rejects_injected_image_uri():
    acct = FakeAccount([OK_SSE])
    args = _args(acct)
    args["messages"] = [{"role": "user", "content": [{"type": "image_url", "image_url": "data:image/png;base64,<script>"}]}]
    await _collect(qwen_api.stream_openai(**args))
    sent = acct.client.completion.await_args.kwargs["prompt"]
    assert "![image](" not in sent

    acct = FakeAccount([])
    acct.client.completion = AsyncMock(side_effect=qwen_api.HTTPException(403, "forbidden"))
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct))))
    assert acct.broken
    assert '"error"' in joined


async def test_non_stream_403_marks_broken():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(side_effect=qwen_api.HTTPException(403, "forbidden"))
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api.collect_non_stream(**_args(acct))
    assert excinfo.value.status_code == 403
    assert acct.broken


async def test_stream_empty_response_sends_role_delta():
    sse = (
        'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n'
        "\n"
        'data: {"choices": [{"delta": {"content": "", "role": "assistant", "status": "finished", "phase": "answer"}}], "response_id": "r1"}\n'
        "\n"
    )
    acct = FakeAccount([sse])
    gen = qwen_api.stream_openai(**_args(acct))
    joined = "".join(await _collect(gen))
    assert '"role": "assistant"' in joined
    assert '"finish_reason": "stop"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_prepare_session_error():
    from fastapi import HTTPException

    acct = FakeAccount([OK_SSE])
    acct.sessions.obtain = AsyncMock(side_effect=HTTPException(401, "bad"))
    gen = qwen_api.stream_openai(**_args(acct))
    lines = await _collect(gen)
    joined = "".join(lines)
    assert '"error"' in joined
    assert "bad" in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_collect_non_stream_retryable_http_exhausted():
    acct = FakeAccount([OK_SSE])
    acct.client.completion = AsyncMock(side_effect=[qwen_api.HTTPException(429, "slow")] * (qwen_api.MAX_RETRIES + 1))
    with pytest.raises(Exception) as excinfo:
        await qwen_api.collect_non_stream(**_args(acct))
    assert excinfo.value.status_code == 429


async def test_collect_non_stream_finish_buffer():
    sse = (
        'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n'
        "\n"
        'data: {"choices": [{"delta": {"content": "Hello", "phase": "answer"}}], "response_id": "r1"}\n'
        "\n"
        'data: {"choices": [{"delta": {"status": "finished", "phase": "answer"}}], "response_id": "r1"}'
    )
    acct = FakeAccount([sse])
    result = await qwen_api.collect_non_stream(**_args(acct))
    assert result["choices"][0]["message"]["content"] == "Hello"


async def test_non_stream_tool_mode_with_reasoning():
    acct = FakeAccount([THINK_SSE])
    args = _args(acct, tool_mode=True)
    result = await qwen_api.collect_non_stream(**args)
    message = result["choices"][0]["message"]
    assert message["content"] == "Answer"
    assert message["reasoning_content"] == "Think step"


async def test_json_non_dict_payload():
    with pytest.raises(Exception) as excinfo:
        await _send(JsonResp("[]"))
    assert excinfo.value.status_code == 502


NO_RID_SSE = (
    'data: {"choices": [{"delta": {"content": "Hi", "phase": "answer"}}]}\n'
    "\n"
    'data: {"choices": [{"delta": {"content": "", "status": "finished", "phase": "answer"}}]}\n'
    "\n"
)


async def test_non_stream_without_response_id():
    acct = FakeAccount([NO_RID_SSE])
    result = await qwen_api.collect_non_stream(**_args(acct))
    assert result["choices"][0]["message"]["content"] == "Hi"


async def test_stream_without_response_id():
    acct = FakeAccount([NO_RID_SSE])
    gen = qwen_api.stream_openai(**_args(acct))
    joined = "".join(await _collect(gen))
    assert '"content": "Hi"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_tool_mode_reasoning_only():
    sse = (
        'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n'
        "\n"
        'data: {"choices": [{"delta": {"content": "step", "phase": "think"}}], "response_id": "r1"}\n'
        "\n"
        'data: {"choices": [{"delta": {"content": "", "status": "finished", "phase": "answer"}}], "response_id": "r1"}\n'
        "\n"
    )
    acct = FakeAccount([sse])
    args = _args(acct, tool_mode=True)
    gen = qwen_api.stream_openai(**args)
    joined = "".join(await _collect(gen))
    assert "reasoning_content" in joined
    assert '"finish_reason": "stop"' in joined
