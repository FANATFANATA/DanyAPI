import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import danyapi.qwen.api as qwen_api

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


class FakeSession:
    def __init__(self, sid: str = "c1", last_response_id: str | None = None) -> None:
        self.id = sid
        self.last_response_id = last_response_id


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
        self.sem = asyncio.Semaphore(1)
        self.sessions = MagicMock()
        self.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
        self.sessions.touch_last_message = MagicMock()


class TestQwenAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_backoff = qwen_api.RETRY_BACKOFF_SEC
        qwen_api.RETRY_BACKOFF_SEC = 0.0

    @classmethod
    def tearDownClass(cls):
        qwen_api.RETRY_BACKOFF_SEC = cls._orig_backoff

    def _args(self, acct, pool=None, existing_sid: str | None = "s1"):
        return {
            "account": acct,
            "pool": pool or MagicMock(),
            "existing_sid": existing_sid,
            "lock": acct.sem,
            "prompt": "x",
            "model": "qwen3.8-max",
            "model_id": "qwen3.8-max",
            "thinking": False,
            "search": False,
        }

    def test_non_stream_collects_content(self):
        acct = FakeAccount([OK_SSE])
        result = asyncio.run(qwen_api.collect_non_stream(**self._args(acct)))
        self.assertEqual(result["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(result["session_id"], "s1")
        self.assertEqual(result["choices"][0]["message"]["role"], "assistant")
        acct.sessions.touch_last_message.assert_called_once_with("s1", "r1")

    def test_non_stream_collects_reasoning(self):
        acct = FakeAccount([THINK_SSE])
        result = asyncio.run(qwen_api.collect_non_stream(**self._args(acct)))
        message = result["choices"][0]["message"]
        self.assertEqual(message["content"], "Answer")
        self.assertEqual(message["reasoning_content"], "Think step")

    def test_non_stream_retries_then_success(self):
        acct = FakeAccount([BUSY_SSE, OK_SSE])
        result = asyncio.run(qwen_api.collect_non_stream(**self._args(acct)))
        self.assertEqual(result["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(acct.client.completion.await_count, 2)

    def test_non_stream_raises_429_after_retries(self):
        acct = FakeAccount([BUSY_SSE] * (qwen_api.MAX_RETRIES + 1))
        with self.assertRaises(Exception) as ctx:
            asyncio.run(qwen_api.collect_non_stream(**self._args(acct)))
        exc = ctx.exception
        assert isinstance(exc, qwen_api.HTTPException)
        self.assertEqual(exc.status_code, 429)

    def test_stream_emits_error_event_after_retries(self):
        acct = FakeAccount([BUSY_SSE] * (qwen_api.MAX_RETRIES + 1))
        gen = qwen_api.stream_openai(**self._args(acct))
        lines = list(asyncio.run(_collect(gen)))
        self.assertEqual(acct.client.completion.await_count, qwen_api.MAX_RETRIES + 1)
        joined = "".join(lines)
        self.assertIn("Too_Many_Requests", joined)
        self.assertTrue(joined.rstrip().endswith("data: [DONE]"))

    def test_stream_success_streams_content(self):
        acct = FakeAccount([OK_SSE])
        gen = qwen_api.stream_openai(**self._args(acct))
        lines = list(asyncio.run(_collect(gen)))
        joined = "".join(lines)
        self.assertIn('"content": "Hello"', joined)
        self.assertIn('"content": " world"', joined)
        self.assertIn('"finish_reason": "stop"', joined)
        self.assertIn('"session_id": "s1"', joined)
        self.assertTrue(joined.rstrip().endswith("data: [DONE]"))

    def test_stream_success_streams_reasoning(self):
        acct = FakeAccount([THINK_SSE])
        gen = qwen_api.stream_openai(**self._args(acct))
        lines = list(asyncio.run(_collect(gen)))
        joined = "".join(lines)
        self.assertIn("reasoning_content", joined)
        self.assertIn('"content": "Answer"', joined)

    def test_stream_emits_usage_when_requested(self):
        acct = FakeAccount([OK_SSE])
        args = self._args(acct)
        args["include_usage"] = True
        gen = qwen_api.stream_openai(**args)
        lines = list(asyncio.run(_collect(gen)))
        joined = "".join(lines)
        self.assertIn('"usage"', joined)
        self.assertIn('"completion_tokens"', joined)

    def test_new_session_registered(self):
        acct = FakeAccount([OK_SSE])
        pool = MagicMock()
        acct.sessions.obtain = AsyncMock(return_value=(FakeSession(sid="new1"), "new1"))
        asyncio.run(qwen_api.collect_non_stream(**self._args(acct, pool=pool, existing_sid=None)))
        pool.register.assert_called_once_with(0, "new1")

    def test_usage_reported(self):
        acct = FakeAccount([OK_SSE])
        result = asyncio.run(qwen_api.collect_non_stream(**self._args(acct)))
        self.assertIn("usage", result)


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


if __name__ == "__main__":
    unittest.main()
