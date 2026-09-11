import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import danyapi.qwen.api as qwen_api

IMG_SSE = (
    'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1","response_index":"0"}} \n'
    "\n"
    'data: {"choices": [{"delta": {"role": "assistant", "content": "![img](https://cdn.qwenlm.ai/a.png)",'
    ' "phase": "image", "status": "typing"}}], "response_id": "r1"}\n'
    "\n"
    'data: {"choices": [{"delta": {"content": "", "role": "assistant", "status": "finished", "phase": "image"}}], "response_id": "r1"}\n'
    "\n"
)

BUSY_SSE = 'data: {"error": {"code": "Too_Many_Requests", "details": "please slow down"}, "response_id": "r1"}\n\n'

CTX_SSE = 'data: {"error": {"code": "ContextLengthExceeded", "details": "too long"}, "response_id": "r1"}\n\n'

AUTH_SSE = 'data: {"error": {"code": "unauthorized", "details": "bad token"}, "response_id": "r1"}\n\n'


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


class FakeSession:
    def __init__(self, sid: str = "c1") -> None:
        self.id = sid
        self.last_response_id = None


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


@pytest.fixture(autouse=True)
def zero_backoff():
    orig = qwen_api.RETRY_BACKOFF_SEC
    qwen_api.RETRY_BACKOFF_SEC = 0.0
    yield
    qwen_api.RETRY_BACKOFF_SEC = orig


def _args(acct, pool=None):
    return {
        "account": acct,
        "pool": pool or MagicMock(),
        "existing_sid": "s1",
        "lock": acct.sem,
        "prompt": "a cat",
        "model": "qwen-image-gen",
        "model_id": "qwen-image-gen",
    }


async def test_collect_image_success():
    acct = FakeAccount([IMG_SSE])
    result = await qwen_api.collect_image(**_args(acct))
    assert result["image_urls"] == ["https://cdn.qwenlm.ai/a.png"]
    assert result["revised_prompt"] == "![img](https://cdn.qwenlm.ai/a.png)"
    assert result["session_id"] == "s1"
    assert "usage" in result
    acct.sessions.touch_last_message.assert_called_once_with("s1", "r1")


async def test_collect_image_retryable_then_success():
    acct = FakeAccount([BUSY_SSE, IMG_SSE])
    result = await qwen_api.collect_image(**_args(acct))
    assert result["image_urls"] == ["https://cdn.qwenlm.ai/a.png"]
    assert acct.client.completion.await_count == 2


async def test_collect_image_context_limit_drops_session():
    acct = FakeAccount([CTX_SSE])
    pool = MagicMock()
    with pytest.raises(Exception) as excinfo:
        await qwen_api.collect_image(**_args(acct, pool=pool))
    assert isinstance(excinfo.value, qwen_api.HTTPException)
    assert excinfo.value.status_code == 400
    pool.forget.assert_called_once_with("s1")
    pool.forget_context.assert_called_once_with("s1")
    acct.sessions.forget.assert_called_once_with("s1")


async def test_collect_image_auth_error_raises_401():
    acct = FakeAccount([AUTH_SSE])
    with pytest.raises(Exception) as excinfo:
        await qwen_api.collect_image(**_args(acct))
    assert isinstance(excinfo.value, qwen_api.HTTPException)
    assert excinfo.value.status_code == 401
    body = json.loads(excinfo.value.detail)
    assert body["error"]["code"] == "unauthorized"


async def test_collect_image_stream_error_stops_upstream():
    acct = FakeAccount([])

    class ErrorResp(FakeResp):
        async def aiter_bytes(self):
            yield b'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
            raise httpx.ReadError("connection reset")

    acct.client.completion = AsyncMock(return_value=ErrorResp(IMG_SSE))
    acct.client.stop_stream = AsyncMock()
    with pytest.raises(Exception) as excinfo:
        await qwen_api.collect_image(**_args(acct))
    assert isinstance(excinfo.value, qwen_api.HTTPException)
    assert excinfo.value.status_code == 502
    acct.client.stop_stream.assert_awaited()
    args, _ = acct.client.stop_stream.call_args
    assert args[0] == "c1"
    assert args[1] == "r1"
