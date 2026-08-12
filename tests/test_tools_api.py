import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

import danyapi.api.openai as openai_mod
import danyapi.qwen.api as qwen_api

TOOL_JSON = '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Moscow"}}]}'

DS_TOOL_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"WIP","fragments":['
    '{"id":2,"type":"RESPONSE","content":"{\\"tool_calls\\": [{\\"name\\": \\"get_weather\\""}]}}}\n'
    "\n"
    'data: {"p":"response/fragments/-1/content","o":"APPEND","v":", \\"arguments\\": '
    '{\\"city\\": \\"Moscow\\"}}]}"}\n'
    "\n"
    'data: {"p":"response/status","o":"SET","v":"FINISHED"}\n'
    "\n"
)

DS_PLAIN_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"WIP","fragments":['
    '{"id":2,"type":"RESPONSE","content":"The weather in Moscow is 22C and sunny."}]}}}\n'
    "\n"
    'data: {"p":"response/status","o":"SET","v":"FINISHED"}\n'
    "\n"
)

DS_XML_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"WIP","fragments":['
    '{"id":2,"type":"RESPONSE","content":"<tool_calls>\\n<invoke name=\\"bash\\">\\n<command>Get-ChildItem -Name</command>\\n</invoke>\\n</tool_calls>"}]}}}\n'
    "\n"
    'data: {"p":"response/status","o":"SET","v":"FINISHED"}\n'
    "\n"
)

QWEN_TOOL_SSE = (
    'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1","response_index":"0"}} \n'
    "\n"
    'data: {"choices": [{"delta": {"role": "assistant", "content": '
    + json.dumps(TOOL_JSON)
    + ', "phase": "answer", "status": "typing"}}], "response_id": "r1"}\n'
    "\n"
    'data: {"choices": [{"delta": {"content": "", "role": "assistant", "status": "finished", "phase": "answer"}}], "response_id": "r1"}\n'
    "\n"
)


class FakeSession:
    def __init__(self, sid="c1", last_message_id=None, last_response_id=None):
        self.id = sid
        self.last_message_id = last_message_id
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


async def collect_stream(gen):
    return [line async for line in gen]


class TestDeepSeekToolCalls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_backoff = openai_mod.RETRY_BACKOFF_SEC
        openai_mod.RETRY_BACKOFF_SEC = 0.0

    @classmethod
    def tearDownClass(cls):
        openai_mod.RETRY_BACKOFF_SEC = cls._orig_backoff

    def _args(self, acct, tool_mode=True):
        return {
            "account": acct,
            "pool": MagicMock(),
            "existing_sid": "s1",
            "lock": acct.sem,
            "prompt": "x",
            "model": "deepseek-v4-flash",
            "model_type": "default",
            "thinking": False,
            "search": False,
            "tool_mode": tool_mode,
        }

    def test_non_stream_formats_tool_calls(self):
        acct = FakeAccount([DS_TOOL_SSE])
        result = asyncio.run(openai_mod._collect_non_stream(**self._args(acct)))
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        message = choice["message"]
        self.assertEqual(len(message["tool_calls"]), 1)
        call = message["tool_calls"][0]
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "get_weather")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Moscow"})

    def test_non_stream_falls_back_to_content(self):
        acct = FakeAccount([DS_PLAIN_SSE])
        result = asyncio.run(openai_mod._collect_non_stream(**self._args(acct)))
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertNotIn("tool_calls", choice["message"])
        self.assertIn("22C", choice["message"]["content"])

    def test_stream_emits_tool_call_deltas(self):
        acct = FakeAccount([DS_TOOL_SSE])
        gen = openai_mod._stream_openai(**self._args(acct))
        lines = asyncio.run(collect_stream(gen))
        joined = "".join(lines)
        self.assertIn('"tool_calls"', joined)
        self.assertIn('"name": "get_weather"', joined)
        self.assertIn('"finish_reason": "tool_calls"', joined)
        arguments = ""
        for line in lines:
            if not line.startswith("data: ") or line.startswith("data: [DONE]"):
                continue
            payload = json.loads(line[6:])
            for chunk in payload.get("choices") or []:
                for tc in (chunk.get("delta") or {}).get("tool_calls") or []:
                    arguments += tc.get("function", {}).get("arguments", "")
        self.assertEqual(json.loads(arguments), {"city": "Moscow"})

    def test_stream_falls_back_to_content(self):
        acct = FakeAccount([DS_PLAIN_SSE])
        gen = openai_mod._stream_openai(**self._args(acct))
        lines = asyncio.run(collect_stream(gen))
        joined = "".join(lines)
        self.assertNotIn('"tool_calls"', joined)
        self.assertIn('"content": "The weather in Moscow is 22C and sunny."', joined)
        self.assertIn('"finish_reason": "stop"', joined)


class TestQwenToolCalls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_backoff = qwen_api.RETRY_BACKOFF_SEC
        qwen_api.RETRY_BACKOFF_SEC = 0.0

    @classmethod
    def tearDownClass(cls):
        qwen_api.RETRY_BACKOFF_SEC = cls._orig_backoff

    def _args(self, acct, tool_mode=True):
        return {
            "account": acct,
            "pool": MagicMock(),
            "existing_sid": "s1",
            "lock": acct.sem,
            "prompt": "x",
            "model": "qwen3.8-max",
            "model_id": "qwen3.8-max",
            "thinking": False,
            "search": False,
            "tool_mode": tool_mode,
        }

    def test_non_stream_formats_tool_calls(self):
        acct = FakeAccount([QWEN_TOOL_SSE])
        result = asyncio.run(qwen_api.collect_non_stream(**self._args(acct)))
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        message = choice["message"]
        self.assertEqual(len(message["tool_calls"]), 1)
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "get_weather")

    def test_stream_emits_tool_call_deltas(self):
        acct = FakeAccount([QWEN_TOOL_SSE])
        gen = qwen_api.stream_openai(**self._args(acct))
        lines = asyncio.run(collect_stream(gen))
        joined = "".join(lines)
        self.assertIn('"tool_calls"', joined)
        self.assertIn('"finish_reason": "tool_calls"', joined)
        self.assertTrue(joined.rstrip().endswith("data: [DONE]"))


if __name__ == "__main__":
    unittest.main()
