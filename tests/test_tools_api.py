import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

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

DS_JSON_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"WIP","fragments":['
    + '{"id":2,"type":"RESPONSE","content":'
    + json.dumps('{"answer": 42}')
    + "}]}}}\n"
    "\n"
    'data: {"p":"response/status","o":"SET","v":"FINISHED"}\n'
    "\n"
)

DS_BAD_JSON_SSE = (
    "event: ready\n"
    'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}\n'
    "\n"
    'data: {"v":{"response":{"message_id":2,"parent_id":1,"status":"WIP","fragments":['
    + '{"id":2,"type":"RESPONSE","content":'
    + json.dumps("not json at all")
    + "}]}}}\n"
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

QWEN_JSON_SSE = (
    'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1","response_index":"0"}} \n'
    "\n"
    'data: {"choices": [{"delta": {"role": "assistant", "content": '
    + json.dumps('{"answer": 42}')
    + ', "phase": "answer", "status": "typing"}}], "response_id": "r1"}\n'
    "\n"
    'data: {"choices": [{"delta": {"content": "", "role": "assistant", "status": "finished", "phase": "answer"}}], "response_id": "r1"}\n'
    "\n"
)

QWEN_BAD_JSON_SSE = (
    'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1","response_index":"0"}} \n'
    "\n"
    'data: {"choices": [{"delta": {"role": "assistant", "content": '
    + json.dumps("not json at all")
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


@pytest.fixture(autouse=True)
def zero_backoff():
    orig_ds = openai_mod.RETRY_BACKOFF_SEC
    orig_qwen = qwen_api.RETRY_BACKOFF_SEC
    openai_mod.RETRY_BACKOFF_SEC = 0.0
    qwen_api.RETRY_BACKOFF_SEC = 0.0
    yield
    openai_mod.RETRY_BACKOFF_SEC = orig_ds
    qwen_api.RETRY_BACKOFF_SEC = orig_qwen


async def collect_stream(gen):
    return [line async for line in gen]


def _deepseek_args(acct, tool_mode=True):
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


def _qwen_args(acct, tool_mode=True):
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


async def test_non_stream_formats_tool_calls():
    acct = FakeAccount([DS_TOOL_SSE])
    result = await openai_mod._collect_non_stream(**_deepseek_args(acct))
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert len(message["tool_calls"]) == 1
    call = message["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Moscow"}


async def test_non_stream_falls_back_to_content():
    acct = FakeAccount([DS_PLAIN_SSE])
    result = await openai_mod._collect_non_stream(**_deepseek_args(acct))
    choice = result["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    assert "22C" in choice["message"]["content"]


async def test_stream_emits_tool_call_deltas():
    acct = FakeAccount([DS_TOOL_SSE])
    gen = openai_mod._stream_openai(**_deepseek_args(acct))
    lines = await collect_stream(gen)
    joined = "".join(lines)
    assert '"tool_calls"' in joined
    assert '"name": "get_weather"' in joined
    assert '"finish_reason": "tool_calls"' in joined
    arguments = ""
    for line in lines:
        if not line.startswith("data: ") or line.startswith("data: [DONE]"):
            continue
        payload = json.loads(line[6:])
        for chunk in payload.get("choices") or []:
            for tc in (chunk.get("delta") or {}).get("tool_calls") or []:
                arguments += tc.get("function", {}).get("arguments", "")
    assert json.loads(arguments) == {"city": "Moscow"}


async def test_stream_falls_back_to_content():
    acct = FakeAccount([DS_PLAIN_SSE])
    gen = openai_mod._stream_openai(**_deepseek_args(acct))
    lines = await collect_stream(gen)
    joined = "".join(lines)
    assert '"tool_calls"' not in joined
    assert '"content": "The weather in Moscow is 22C and sunny."' in joined
    assert '"finish_reason": "stop"' in joined


async def test_qwen_non_stream_formats_tool_calls():
    acct = FakeAccount([QWEN_TOOL_SSE])
    result = await qwen_api.collect_non_stream(**_qwen_args(acct))
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"


async def test_qwen_stream_emits_tool_call_deltas():
    acct = FakeAccount([QWEN_TOOL_SSE])
    gen = qwen_api.stream_openai(**_qwen_args(acct))
    lines = await collect_stream(gen)
    joined = "".join(lines)
    assert '"tool_calls"' in joined
    assert '"finish_reason": "tool_calls"' in joined
    assert joined.rstrip().endswith("data: [DONE]")


JSON_ANSWER_SCHEMA = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "integer"}}}


async def test_json_schema_valid_response_passes():
    acct = FakeAccount([DS_JSON_SSE])
    result = await openai_mod._collect_non_stream(**{**_deepseek_args(acct, tool_mode=False), "json_schema": JSON_ANSWER_SCHEMA})
    choice = result["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == '{"answer": 42}'


async def test_json_schema_invalid_json_raises():
    acct = FakeAccount([DS_BAD_JSON_SSE])
    with pytest.raises(openai_mod.HTTPException) as exc:
        await openai_mod._collect_non_stream(**{**_deepseek_args(acct, tool_mode=False), "json_schema": JSON_ANSWER_SCHEMA})
    assert exc.value.status_code == 400
    assert "Invalid JSON response" in str(exc.value.detail)


async def test_json_schema_mismatch_raises():
    acct = FakeAccount([DS_JSON_SSE])
    with pytest.raises(openai_mod.HTTPException) as exc:
        await openai_mod._collect_non_stream(**{**_deepseek_args(acct, tool_mode=False), "json_schema": {"type": "object", "required": ["score"]}})
    assert exc.value.status_code == 400
    assert "does not match JSON schema" in str(exc.value.detail)


async def test_json_schema_no_schema_no_validation():
    acct = FakeAccount([DS_BAD_JSON_SSE])
    result = await openai_mod._collect_non_stream(**{**_deepseek_args(acct, tool_mode=False), "json_schema": None})
    assert result["choices"][0]["finish_reason"] == "stop"


async def test_stream_json_schema_error_terminates_stream():
    acct = FakeAccount([DS_BAD_JSON_SSE])
    gen = openai_mod._stream_openai(**{**_deepseek_args(acct, tool_mode=False), "json_schema": JSON_ANSWER_SCHEMA})
    lines = await collect_stream(gen)
    joined = "".join(lines)
    assert '"finish_reason": "error"' in joined
    assert "Invalid JSON response" in joined
    assert joined.rstrip().endswith("data: [DONE]")


async def test_qwen_json_schema_valid_response_passes():
    acct = FakeAccount([QWEN_JSON_SSE])
    result = await qwen_api.collect_non_stream(**{**_qwen_args(acct, tool_mode=False), "json_schema": JSON_ANSWER_SCHEMA})
    choice = result["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == '{"answer": 42}'


async def test_qwen_json_schema_invalid_json_raises():
    acct = FakeAccount([QWEN_BAD_JSON_SSE])
    with pytest.raises(openai_mod.HTTPException) as exc:
        await qwen_api.collect_non_stream(**{**_qwen_args(acct, tool_mode=False), "json_schema": JSON_ANSWER_SCHEMA})
    assert exc.value.status_code == 400


async def test_qwen_stream_json_schema_error_terminates_stream():
    acct = FakeAccount([QWEN_BAD_JSON_SSE])
    gen = qwen_api.stream_openai(**{**_qwen_args(acct, tool_mode=False), "json_schema": JSON_ANSWER_SCHEMA})
    lines = await collect_stream(gen)
    joined = "".join(lines)
    assert '"finish_reason": "error"' in joined
    assert "Invalid JSON response" in joined
    assert joined.rstrip().endswith("data: [DONE]")


def test_resolve_thinking_explicit_flag_wins():
    req = openai_mod.ChatCompletionRequest(thinking=False, reasoning_effort="high")
    assert openai_mod._resolve_thinking(req, default=True) is False
    req = openai_mod.ChatCompletionRequest(thinking=True, reasoning_effort="none")
    assert openai_mod._resolve_thinking(req, default=False) is True


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_resolve_thinking_effort_enables_thinking(effort):
    req = openai_mod.ChatCompletionRequest(reasoning_effort=effort)
    assert openai_mod._resolve_thinking(req, default=False) is True


@pytest.mark.parametrize("effort", ["none", "off", "", "   "])
def test_resolve_thinking_off_effort_falls_back_to_default(effort):
    req = openai_mod.ChatCompletionRequest(reasoning_effort=effort)
    assert openai_mod._resolve_thinking(req, default=False) is False
    assert openai_mod._resolve_thinking(req, default=True) is True


def test_resolve_thinking_effort_is_case_and_space_insensitive():
    req = openai_mod.ChatCompletionRequest(reasoning_effort="  High ")
    assert openai_mod._resolve_thinking(req, default=False) is True
    req = openai_mod.ChatCompletionRequest(reasoning_effort="NONE")
    assert openai_mod._resolve_thinking(req, default=False) is False


def test_resolve_thinking_no_effort_uses_default():
    req = openai_mod.ChatCompletionRequest()
    assert openai_mod._resolve_thinking(req, default=False) is False
    assert openai_mod._resolve_thinking(req, default=True) is True
