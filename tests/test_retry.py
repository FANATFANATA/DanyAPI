import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


class TestRetry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_backoff = openai_mod.RETRY_BACKOFF_SEC
        openai_mod.RETRY_BACKOFF_SEC = 0.0

    @classmethod
    def tearDownClass(cls):
        openai_mod.RETRY_BACKOFF_SEC = cls._orig_backoff

    def _args(self, acct, pool=None, existing_sid: str | None = "s1"):
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

    def test_non_stream_retries_then_success(self):
        acct = FakeAccount([BUSY_SSE, OK_SSE])
        result = asyncio.run(_collect_non_stream(**self._args(acct)))
        self.assertEqual(result["choices"][0]["message"]["content"], "Привет")
        self.assertEqual(acct.pow.make_header.await_count, 2)
        self.assertEqual(acct.client.completion.await_count, 2)

    def test_non_stream_raises_429_after_retries(self):
        acct = FakeAccount([BUSY_SSE] * (openai_mod.MAX_RETRIES + 1))
        with self.assertRaises(Exception) as ctx:
            asyncio.run(_collect_non_stream(**self._args(acct)))
        exc = ctx.exception
        assert isinstance(exc, openai_mod.HTTPException)
        self.assertEqual(exc.status_code, 429)
        self.assertIn("busy", exc.detail.lower())

    def test_stream_emits_error_event_after_retries(self):
        acct = FakeAccount([BUSY_SSE] * (openai_mod.MAX_RETRIES + 1))
        gen = _stream_openai(**self._args(acct))
        lines = list(asyncio.run(_collect(gen)))
        self.assertEqual(acct.client.completion.await_count, openai_mod.MAX_RETRIES + 1)
        joined = "".join(lines)
        self.assertIn('"error"', joined)
        self.assertIn("expert_busy_use_default", joined)
        self.assertTrue(joined.rstrip().endswith("data: [DONE]"))

    def test_stream_success_streams_content(self):
        acct = FakeAccount([OK_SSE])
        gen = _stream_openai(**self._args(acct))
        lines = list(asyncio.run(_collect(gen)))
        joined = "".join(lines)
        self.assertIn('"content": "При"', joined)
        self.assertIn('"content": "вет"', joined)
        self.assertIn('"finish_reason": "stop"', joined)
        self.assertTrue(joined.rstrip().endswith("data: [DONE]"))

    def test_stream_emits_usage_when_requested(self):
        acct = FakeAccount([OK_SSE])
        args = self._args(acct)
        args["include_usage"] = True
        gen = _stream_openai(**args)
        lines = list(asyncio.run(_collect(gen)))
        joined = "".join(lines)
        self.assertIn('"usage"', joined)
        self.assertIn('"completion_tokens"', joined)

    def test_stream_omits_usage_by_default(self):
        acct = FakeAccount([OK_SSE])
        gen = _stream_openai(**self._args(acct))
        lines = list(asyncio.run(_collect(gen)))
        self.assertNotIn('"usage"', "".join(lines))

    def test_obtain_waits_for_lock(self):
        acct = FakeAccount([OK_SSE])
        state = {"under_lock": False}

        async def fake_obtain(sid):
            state["under_lock"] = acct.sem.locked()
            return FakeSession(), "s1"

        acct.sessions.obtain = fake_obtain
        asyncio.run(_collect_non_stream(**self._args(acct)))
        self.assertTrue(state["under_lock"])

    def test_new_session_registered(self):
        acct = FakeAccount([OK_SSE])
        pool = MagicMock()
        acct.sessions.obtain = AsyncMock(return_value=(FakeSession(sid="new1"), "new1"))
        asyncio.run(_collect_non_stream(**self._args(acct, pool=pool, existing_sid=None)))
        pool.register.assert_called_once_with(0, "new1")

    def test_prepare_session_auth_error_is_401(self):
        from danyapi.deepseek.client import DeepSeekError

        acct = FakeAccount([OK_SSE])
        acct.sessions.obtain = AsyncMock(side_effect=DeepSeekError(40001, "token invalid"))
        with self.assertRaises(Exception) as ctx:
            asyncio.run(_collect_non_stream(**self._args(acct)))
        exc = ctx.exception
        assert isinstance(exc, openai_mod.HTTPException)
        self.assertEqual(exc.status_code, 401)
        self.assertTrue(acct.broken)

    def test_prepare_session_non_auth_error_is_502(self):
        from danyapi.deepseek.client import DeepSeekError

        acct = FakeAccount([OK_SSE])
        acct.sessions.obtain = AsyncMock(side_effect=DeepSeekError(5000, "session boom"))
        with self.assertRaises(Exception) as ctx:
            asyncio.run(_collect_non_stream(**self._args(acct)))
        exc = ctx.exception
        assert isinstance(exc, openai_mod.HTTPException)
        self.assertEqual(exc.status_code, 502)
        self.assertFalse(acct.broken)

    def test_completion_http_401_marks_broken(self):
        from fastapi import HTTPException

        acct = FakeAccount([OK_SSE])
        acct.client.completion = AsyncMock(side_effect=HTTPException(401, "unauthorized"))
        with self.assertRaises(Exception) as ctx:
            asyncio.run(_collect_non_stream(**self._args(acct)))
        exc = ctx.exception
        assert isinstance(exc, openai_mod.HTTPException)
        self.assertEqual(exc.status_code, 401)
        self.assertTrue(acct.broken)

    def test_stream_emits_error_when_completion_fails(self):
        acct = FakeAccount([])
        acct.client.completion = AsyncMock(side_effect=openai_mod.HTTPException(502, "boom"))
        gen = _stream_openai(**self._args(acct))
        lines = list(asyncio.run(_collect(gen)))
        joined = "".join(lines)
        self.assertIn('"error"', joined)
        self.assertIn("boom", joined)
        self.assertTrue(joined.rstrip().endswith("data: [DONE]"))

    def test_stream_emits_error_when_pow_header_fails(self):
        from danyapi.deepseek.client import DeepSeekError

        acct = FakeAccount([])
        acct.pow.make_header = AsyncMock(side_effect=DeepSeekError(5000, "pow boom"))
        gen = _stream_openai(**self._args(acct))
        lines = list(asyncio.run(_collect(gen)))
        joined = "".join(lines)
        self.assertIn('"error"', joined)
        self.assertIn("pow boom", joined)
        self.assertTrue(joined.rstrip().endswith("data: [DONE]"))

    def test_non_stream_context_limit_drops_session_and_raises_400(self):
        acct = FakeAccount([CTX_SSE])
        pool = MagicMock()
        with self.assertRaises(Exception) as ctx:
            asyncio.run(_collect_non_stream(**self._args(acct, pool=pool)))
        exc = ctx.exception
        assert isinstance(exc, openai_mod.HTTPException)
        self.assertEqual(exc.status_code, 400)
        pool.forget.assert_called_once_with("s1")
        pool.forget_context.assert_called_once_with("s1")
        acct.sessions.forget.assert_called_once_with("s1")

    def test_stream_context_limit_drops_session_and_emits_length(self):
        acct = FakeAccount([CTX_SSE])
        pool = MagicMock()
        gen = _stream_openai(**self._args(acct, pool=pool))
        lines = list(asyncio.run(_collect(gen)))
        joined = "".join(lines)
        self.assertIn('"finish_reason": "length"', joined)
        self.assertIn("context length exceeded", joined)
        self.assertTrue(joined.rstrip().endswith("data: [DONE]"))
        pool.forget.assert_called_once_with("s1")
        pool.forget_context.assert_called_once_with("s1")
        acct.sessions.forget.assert_called_once_with("s1")


class TestModelConfig(unittest.TestCase):
    def test_model_type_mapping(self):
        self.assertEqual(
            openai_mod.MODEL_TYPE_BY_NAME,
            {
                "deepseek-v4-flash": "default",
                "deepseek-v4-pro": "expert",
                "deepseek-v4-vision": "vision",
            },
        )

    def test_search_gated_to_flash_and_thinking_allowed(self):
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

            asyncio.run(run("deepseek-v4-flash", search=True, thinking=None))
            self.assertEqual(captured["model_type"], "default")
            self.assertEqual(captured["search"], True)
            self.assertEqual(captured["thinking"], False)

            asyncio.run(run("deepseek-v4-pro", search=True, thinking=None))
            self.assertEqual(captured["model_type"], "expert")
            self.assertEqual(captured["search"], False)
            self.assertEqual(captured["thinking"], True)

            asyncio.run(run("deepseek-v4-vision", search=True, thinking=True))
            self.assertEqual(captured["model_type"], "vision")
            self.assertEqual(captured["search"], False)
            self.assertEqual(captured["thinking"], True)
        finally:
            openai_mod._collect_non_stream = orig


class TestAttachments(unittest.TestCase):
    def test_pro_rejects_all_files(self):
        from danyapi.api.openai import Attachment, _validate_attachments

        with self.assertRaises(Exception) as ctx:
            _validate_attachments([Attachment(b"x", "a.txt", "text/plain", False)], "expert")
        exc = ctx.exception
        assert isinstance(exc, openai_mod.HTTPException)
        self.assertEqual(exc.status_code, 400)

    def test_vision_rejects_text_files(self):
        from danyapi.api.openai import Attachment, _validate_attachments

        with self.assertRaises(Exception) as ctx:
            _validate_attachments([Attachment(b"x", "a.txt", "text/plain", False)], "vision")
        exc = ctx.exception
        assert isinstance(exc, openai_mod.HTTPException)
        self.assertEqual(exc.status_code, 400)

    def test_vision_accepts_images(self):
        from danyapi.api.openai import Attachment, _validate_attachments

        _validate_attachments([Attachment(b"x", "a.png", "image/png", True)], "vision")

    def test_too_many_files_rejected(self):
        from danyapi.api.openai import MAX_FILES_PER_REQUEST, Attachment, _validate_attachments

        many = [Attachment(b"x", f"{i}.txt", "text/plain", False) for i in range(MAX_FILES_PER_REQUEST + 1)]
        with self.assertRaises(Exception) as ctx:
            _validate_attachments(many, "default")
        exc = ctx.exception
        assert isinstance(exc, openai_mod.HTTPException)
        self.assertEqual(exc.status_code, 400)

    def test_collect_attachments_from_image_url_and_files(self):
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
        self.assertEqual(len(atts), 2)
        self.assertTrue(atts[0].is_image)
        self.assertEqual(atts[0].data, b"pngdata")
        self.assertFalse(atts[1].is_image)
        self.assertEqual(atts[1].data, b"hello")

    def test_upload_attachments_returns_ids(self):
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
        ids = asyncio.run(
            _upload_attachments(
                acct,
                [Attachment(b"a", "a.txt", "text/plain", False), Attachment(b"b", "b.txt", "text/plain", False)],
                "default",
                False,
            )
        )
        self.assertEqual(ids, ["file-1", "file-2"])
        self.assertEqual(acct.client.upload_file.await_count, 2)


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


if __name__ == "__main__":
    unittest.main()
