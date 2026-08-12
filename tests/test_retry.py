import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import danyapi.api.openai as openai_mod
from danyapi.api.openai import _collect_non_stream, _stream_openai

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
        self.client = MagicMock()
        self.client.completion = AsyncMock(side_effect=[FakeResp(s) for s in sse_list])
        self.client.create_pow_challenge = AsyncMock(return_value={})
        self.pow = MagicMock()
        self.pow.make_header = AsyncMock(return_value={})
        self.sem = asyncio.Semaphore(1)
        self.sessions = MagicMock()
        self.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
        self.sessions.touch_last_message = MagicMock()


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
            "model": "deepseek-chat",
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


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


if __name__ == "__main__":
    unittest.main()
