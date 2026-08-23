import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import danyapi.api.openai as openai_mod
from danyapi.api.openai import ChatMessage, _collect_non_stream, _stream_openai

BUSY_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"expert"}\n'
    "\n"
    "event: hint\n"
    'data: {"type":"error","content":"Server is busy. Try again later, or use Instant Mode.","clear_response":true,"finish_reason":"expert_busy_use_default"}\n'
    "\n"
    "event: close\n"
    'data: {"click_behavior":"retry","auto_resume":false}\n'
    "\n"
)

OK_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"WIP","fragments":[{"id":2,"type":"RESPONSE","content":"При"}]}}}\n'
    "\n"
    'data: {"p":"response/fragments/-1/content","o":"APPEND","v":"вет"}\n'
    "\n"
    'data: {"p":"response/status","o":"SET","v":"FINISHED"}\n'
    "\n"
)

CTX_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"p":"response/status","o":"SET","v":"CONTEXT_LENGTH_EXCEEDED"}\n'
    "\n"
)


class FakeSession:
    def __init__(self, sid: str = "c1", last_message_id: str | None = None) -> None:
        self.id = sid
        self.last_message_id = last_message_id


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
        self.client.create_pow_challenge = AsyncMock(return_value={})
        self.pow = MagicMock()
        self.pow.make_header = AsyncMock(return_value={})
        self.pow_upload = MagicMock()
        self.pow_upload.make_header = AsyncMock(return_value={})
        self.sem = asyncio.Semaphore(1)
        self.sessions = MagicMock()
        self.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
        self.sessions.touch_last_message = MagicMock()

    def mark_broken(self):
        self.broken = True


@pytest.fixture(autouse=True)
def zero_backoff():
    orig = openai_mod.RETRY_BACKOFF_SEC
    openai_mod.RETRY_BACKOFF_SEC = 0.0
    yield
    openai_mod.RETRY_BACKOFF_SEC = orig


def _args(acct, pool=None, existing_sid: str | None = "s1"):
    return {
        "account": acct,
        "pool": pool or MagicMock(),
        "existing_sid": existing_sid,
        "lock": acct.sem,
        "prompt": "x",
        "model": "deepseek-v4-flash",
        "model_type": "default",
        "thinking": False,
        "search": False,
    }


async def test_non_stream_retries_then_success():
    acct = FakeAccount([BUSY_SSE, OK_SSE])
    result = await _collect_non_stream(**_args(acct))
    assert result["choices"][0]["message"]["content"] == "Привет"
    assert acct.pow.make_header.await_count == 2
    assert acct.client.completion.await_count == 2


async def test_non_stream_raises_429_after_retries():
    acct = FakeAccount([BUSY_SSE] * (openai_mod.MAX_RETRIES + 1))
    with pytest.raises(Exception) as excinfo:
        await _collect_non_stream(**_args(acct))
    exc = excinfo.value
    assert isinstance(exc, openai_mod.HTTPException)
    assert exc.status_code == 429
    assert "busy" in exc.detail.lower()


async def test_stream_emits_error_event_after_retries():
    acct = FakeAccount([BUSY_SSE] * (openai_mod.MAX_RETRIES + 1))
    gen = _stream_openai(**_args(acct))
    lines = list(await _collect(gen))
    assert acct.client.completion.await_count == openai_mod.MAX_RETRIES + 1
    joined = "".join(lines)
    assert '"error"' in joined
    assert "expert_busy_use_default" in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_success_streams_content():
    acct = FakeAccount([OK_SSE])
    gen = _stream_openai(**_args(acct))
    lines = list(await _collect(gen))
    joined = "".join(lines)
    assert '"content": "При"' in joined
    assert '"content": "вет"' in joined
    assert '"finish_reason": "stop"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_emits_usage_when_requested():
    acct = FakeAccount([OK_SSE])
    args = _args(acct)
    args["include_usage"] = True
    gen = _stream_openai(**args)
    lines = list(await _collect(gen))
    joined = "".join(lines)
    assert '"usage"' in joined
    assert '"completion_tokens"' in joined


async def test_stream_omits_usage_by_default():
    acct = FakeAccount([OK_SSE])
    gen = _stream_openai(**_args(acct))
    lines = list(await _collect(gen))
    assert '"usage"' not in "".join(lines)


async def test_obtain_waits_for_lock():
    acct = FakeAccount([OK_SSE])
    state = {"under_lock": False}

    async def fake_obtain(sid):
        state["under_lock"] = acct.sem.locked()
        return FakeSession(), "s1"

    acct.sessions.obtain = fake_obtain
    await _collect_non_stream(**_args(acct))
    assert state["under_lock"]


async def test_new_session_registered():
    acct = FakeAccount([OK_SSE])
    pool = MagicMock()
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(sid="new1"), "new1"))
    await _collect_non_stream(**_args(acct, pool=pool, existing_sid=None))
    pool.register.assert_called_once_with(0, "new1")


async def test_prepare_session_auth_error_is_401():
    from danyapi.deepseek.client import DeepSeekError

    acct = FakeAccount([OK_SSE])
    acct.sessions.obtain = AsyncMock(side_effect=DeepSeekError(40001, "token invalid"))
    with pytest.raises(Exception) as excinfo:
        await _collect_non_stream(**_args(acct))
    exc = excinfo.value
    assert isinstance(exc, openai_mod.HTTPException)
    assert exc.status_code == 401
    assert acct.broken


async def test_prepare_session_non_auth_error_is_502():
    from danyapi.deepseek.client import DeepSeekError

    acct = FakeAccount([OK_SSE])
    acct.sessions.obtain = AsyncMock(side_effect=DeepSeekError(5000, "session boom"))
    with pytest.raises(Exception) as excinfo:
        await _collect_non_stream(**_args(acct))
    exc = excinfo.value
    assert isinstance(exc, openai_mod.HTTPException)
    assert exc.status_code == 502
    assert not acct.broken


async def test_completion_http_401_marks_broken():
    from fastapi import HTTPException

    acct = FakeAccount([OK_SSE])
    acct.client.completion = AsyncMock(side_effect=HTTPException(401, "unauthorized"))
    with pytest.raises(Exception) as excinfo:
        await _collect_non_stream(**_args(acct))
    exc = excinfo.value
    assert isinstance(exc, openai_mod.HTTPException)
    assert exc.status_code == 401
    assert acct.broken


async def test_stream_emits_error_when_completion_fails():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(side_effect=openai_mod.HTTPException(502, "boom"))
    gen = _stream_openai(**_args(acct))
    lines = list(await _collect(gen))
    joined = "".join(lines)
    assert '"error"' in joined
    assert "boom" in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_emits_error_when_pow_header_fails():
    from danyapi.deepseek.client import DeepSeekError

    acct = FakeAccount([])
    acct.pow.make_header = AsyncMock(side_effect=DeepSeekError(5000, "pow boom"))
    gen = _stream_openai(**_args(acct))
    lines = list(await _collect(gen))
    joined = "".join(lines)
    assert '"error"' in joined
    assert "pow boom" in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_non_stream_context_limit_drops_session_and_raises_400():
    acct = FakeAccount([CTX_SSE])
    pool = MagicMock()
    with pytest.raises(Exception) as excinfo:
        await _collect_non_stream(**_args(acct, pool=pool))
    exc = excinfo.value
    assert isinstance(exc, openai_mod.HTTPException)
    assert exc.status_code == 400
    pool.forget.assert_called_once_with("s1")
    pool.forget_context.assert_called_once_with("s1")
    acct.sessions.forget.assert_called_once_with("s1")


async def test_stream_context_limit_drops_session_and_emits_length():
    acct = FakeAccount([CTX_SSE])
    pool = MagicMock()
    gen = _stream_openai(**_args(acct, pool=pool))
    lines = list(await _collect(gen))
    joined = "".join(lines)
    assert '"finish_reason": "length"' in joined
    assert "context length exceeded" in joined
    assert joined.rstrip().endswith("data: [DONE]")
    pool.forget.assert_called_once_with("s1")
    pool.forget_context.assert_called_once_with("s1")
    acct.sessions.forget.assert_called_once_with("s1")


async def test_non_stream_cancel_stops_upstream():
    acct = FakeAccount([])
    started = asyncio.Event()

    class BlockingResp(FakeResp):
        async def aiter_bytes(self):
            yield (b'event: ready\ndata: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n\n')
            started.set()
            await asyncio.Event().wait()
            yield b""

    acct.client.completion = AsyncMock(return_value=BlockingResp(OK_SSE))
    acct.client.stop_stream = AsyncMock()

    task = asyncio.create_task(_collect_non_stream(**_args(acct)))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    acct.client.stop_stream.assert_awaited_once()
    args, _ = acct.client.stop_stream.call_args
    assert args[0] == "c1"
    assert args[1] == 2


async def test_stream_disconnect_stops_upstream():
    acct = FakeAccount([OK_SSE])
    acct.client.stop_stream = AsyncMock()
    gen = _stream_openai(**_args(acct))

    first = await gen.__anext__()
    assert first.startswith("data: ")
    await gen.aclose()
    acct.client.stop_stream.assert_awaited_once()
    args, _ = acct.client.stop_stream.call_args
    assert args[0] == "c1"
    assert args[1] == 2


async def test_stream_full_consumption_does_not_stop_upstream():
    acct = FakeAccount([OK_SSE])
    acct.client.stop_stream = AsyncMock()
    gen = _stream_openai(**_args(acct))
    lines = list(await _collect(gen))
    assert any(line.startswith("data: ") for line in lines)
    acct.client.stop_stream.assert_not_awaited()


def test_model_type_mapping():
    assert openai_mod.MODEL_TYPE_BY_NAME == {
        "deepseek-v4-flash": "default",
        "deepseek-v4-pro": "expert",
        "deepseek-v4-vision": "vision",
    }


async def test_search_gated_to_flash_and_thinking_allowed():
    captured = {}
    orig = openai_mod._collect_non_stream

    async def fake_collect(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    openai_mod._collect_non_stream = fake_collect
    try:
        pool = MagicMock()
        pool.acquire = AsyncMock(return_value=(FakeAccount([OK_SSE]), None))
        openai_mod.app.state.pool = pool

        async def run(model, search, thinking):
            req = SimpleNamespace(
                model=model,
                stream=False,
                thinking=thinking,
                search=search,
                session_id=None,
                files=None,
                messages=[ChatMessage(role="user", content="x")],
            )
            await openai_mod._chat_completions_deepseek(req)

        await run("deepseek-v4-flash", search=True, thinking=None)
        assert captured["model_type"] == "default"
        assert captured["search"] is True
        assert captured["thinking"] is False

        await run("deepseek-v4-pro", search=True, thinking=None)
        assert captured["model_type"] == "expert"
        assert captured["search"] is False
        assert captured["thinking"] is False

        await run("deepseek-v4-pro-thinking", search=True, thinking=None)
        assert captured["model_type"] == "expert"
        assert captured["search"] is False
        assert captured["thinking"] is True

        await run("deepseek-v4-vision", search=True, thinking=True)
        assert captured["model_type"] == "vision"
        assert captured["search"] is False
        assert captured["thinking"] is True
    finally:
        openai_mod._collect_non_stream = orig


def test_pro_rejects_all_files():
    from danyapi.api.openai import Attachment, _validate_attachments

    with pytest.raises(Exception) as excinfo:
        _validate_attachments([Attachment(b"x", "a.txt", "text/plain", False)], "expert")
    exc = excinfo.value
    assert isinstance(exc, openai_mod.HTTPException)
    assert exc.status_code == 400


def test_vision_rejects_text_files():
    from danyapi.api.openai import Attachment, _validate_attachments

    with pytest.raises(Exception) as excinfo:
        _validate_attachments([Attachment(b"x", "a.txt", "text/plain", False)], "vision")
    exc = excinfo.value
    assert isinstance(exc, openai_mod.HTTPException)
    assert exc.status_code == 400


def test_vision_accepts_images():
    from danyapi.api.openai import Attachment, _validate_attachments

    _validate_attachments([Attachment(b"x", "a.png", "image/png", True)], "vision")


def test_too_many_files_rejected():
    from danyapi.api.openai import MAX_FILES_PER_REQUEST, Attachment, _validate_attachments

    many = [Attachment(b"x", f"{i}.txt", "text/plain", False) for i in range(MAX_FILES_PER_REQUEST + 1)]
    with pytest.raises(Exception) as excinfo:
        _validate_attachments(many, "default")
    exc = excinfo.value
    assert isinstance(exc, openai_mod.HTTPException)
    assert exc.status_code == 400


def test_collect_attachments_from_image_url_and_files():
    import base64 as b64

    from danyapi.api.openai import ChatMessage, _collect_attachments

    img_b64 = b64.b64encode(b"pngdata").decode()
    file_b64 = b64.b64encode(b"hello").decode()
    req = SimpleNamespace(
        model="deepseek-v4-flash",
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            )
        ],
        files=[SimpleNamespace(name="doc.txt", content=file_b64, content_type="text/plain")],
    )
    atts = _collect_attachments(req)
    assert len(atts) == 2
    assert atts[0].is_image
    assert atts[0].data == b"pngdata"
    assert not atts[1].is_image
    assert atts[1].data == b"hello"


async def test_upload_attachments_returns_ids():
    from danyapi.api.openai import Attachment, _upload_attachments

    acct = FakeAccount([OK_SSE])
    acct.client.upload_file = AsyncMock(
        side_effect=[
            {"id": "file-1", "status": "PENDING"},
            {"id": "file-2", "status": "PENDING"},
        ]
    )
    acct.pow_upload = MagicMock()
    acct.pow_upload.make_header = AsyncMock(return_value={"X-DS-PoW-Response": "x"})
    ids = await _upload_attachments(
        acct,
        [Attachment(b"a", "a.txt", "text/plain", False), Attachment(b"b", "b.txt", "text/plain", False)],
        "default",
        False,
    )
    assert ids == ["file-1", "file-2"]
    assert acct.client.upload_file.await_count == 2


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out
