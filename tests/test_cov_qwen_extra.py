import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import danyapi.qwen.api as qwen_api
from danyapi.qwen.stream import QwenStreamReconstructor


def _created() -> str:
    return 'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'


def _trail(body) -> str:
    return "data: " + json.dumps(body)


TWO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "time",
            "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}},
        },
    },
]

TWO_TOOL_SCHEMAS = qwen_api.toolemu.tool_schema_map(TWO_TOOLS)

TWO_CALL_JSON = json.dumps(
    {
        "tool_calls": [
            {"name": "get_weather", "arguments": {"city": "Moscow"}},
            {"name": "get_time", "arguments": {"zone": "MSK"}},
        ]
    }
)

TWO_CALL_SSE = _created() + _trail({"choices": [{"delta": {"content": TWO_CALL_JSON, "phase": "answer"}}], "response_id": "r1"})

OK_SSE = (
    'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
    'data: {"choices": [{"delta": {"role": "assistant", "content": "Hello", "phase": "answer"}}], "response_id": "r1"}\n\n'
    'data: {"choices": [{"delta": {"content": "", "role": "assistant", "status": "finished", "phase": "answer"}}], "response_id": "r1"}\n\n'
)

FINISH_ONLY_SSE = (
    _created()
    + _trail({"choices": [{"delta": {"role": "assistant", "status": "typing", "phase": "answer"}}], "response_id": "r1"})
    + "\n\n"
    + _trail({"choices": [{"delta": {"content": "", "status": "finished", "phase": "answer"}}], "response_id": "r1"})
    + "\n\n"
)


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


def test_image_markdown_non_image_item():
    messages = [_ImgMsg([{"type": "text", "text": "hi"}])]
    assert qwen_api._append_image_markdown("hello", messages) == "hello"


def test_build_limited_message_tool_fallback_length():
    rec = _rec_with("aaaa bbbb cccc", None)
    _message, finish = qwen_api._build_limited_message(rec, True, None, 2, None, None)
    assert finish == "length"


async def test_collect_non_stream_http_401_marks_broken():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(side_effect=qwen_api.HTTPException(401, "bad"))
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api.collect_non_stream(**_args(acct))
    assert excinfo.value.status_code == 401
    assert acct.broken


async def test_collect_non_stream_stale_rebuild_success():
    acct = FakeAccount([])
    acct.sessions.get = MagicMock(return_value=object())
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(sid="new"), "new"))
    acct.client.completion = AsyncMock(side_effect=[qwen_api.HTTPException(404, "stale"), FakeResp(OK_SSE)])
    args = _args(acct)
    args["messages"] = [{"role": "user", "content": "hi"}]
    result = await qwen_api.collect_non_stream(**args)
    assert result["choices"][0]["message"]["content"] == "Hello"


async def test_collect_non_stream_stale_rebuild_value_error():
    acct = FakeAccount([])
    acct.sessions.get = MagicMock(return_value=object())
    acct.sessions.obtain = AsyncMock(return_value=(FakeSession(sid="new"), "new"))
    acct.client.completion = AsyncMock(side_effect=qwen_api.HTTPException(404, "stale"))
    args = _args(acct)
    args["messages"] = [{"role": "system", "content": "sys"}]
    with pytest.raises(qwen_api.HTTPException) as excinfo:
        await qwen_api.collect_non_stream(**args)
    assert excinfo.value.status_code == 404


async def test_stream_tool_two_calls_parallel_false():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=FakeResp(TWO_CALL_SSE))
    args = _args(acct, tool_mode=True, tool_schemas=TWO_TOOL_SCHEMAS, parallel_tool_calls=False)
    joined = "".join(await _collect(qwen_api.stream_openai(**args)))
    assert '"tool_calls"' in joined


async def test_stream_finish_only_events():
    acct = FakeAccount([])
    acct.client.completion = AsyncMock(return_value=FakeResp(FINISH_ONLY_SSE))
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct))))
    assert '"finish_reason": "stop"' in joined
