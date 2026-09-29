import asyncio
import json
import logging
import pathlib
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException

import danyapi.api.retry as retry_mod
import danyapi.qwen.api as qwen_api
import danyapi.qwen.client as qwen_client
import danyapi.sseutil as sseutil
from danyapi.qwen.accounts import QwenAccount, QwenSessionRegistry
from danyapi.qwen.client import QwenClient, QwenError, QwenSession, _cached_timezone_header, _error_code, new_uuid
from danyapi.qwen.stream import QwenStreamReconstructor, error_code
from danyapi.store import JsonStore
from danyapi.tokens import estimate_tokens

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

REAL_SLEEP = asyncio.sleep

OPENAI_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter", "function_call", "error"}

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

CTX_AFTER_CONTENT_SSE = (
    'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
    'data: {"choices": [{"delta": {"content": "Partial", "phase": "answer"}}], "response_id": "r1"}\n\n'
    'data: {"error": {"code": "ContextLengthExceeded", "details": "too long"}, "response_id": "r1"}\n\n'
)

TAIL_IMAGE_SSE = (
    'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
    'data: {"choices": [{"delta": {"content": "see ![img](https://cdn.qwenlm.ai/", "phase": "answer"}}], "response_id": "r1"}\n\n'
    'data: {"choices": [{"delta": {"content": "tail.png", "phase": "answer"}}], "response_id": "r1"}\n\n'
)

TOOL_JSON = '{"tool_calls":[{"name":"get_weather","arguments":{"city":"Moscow"}}]}'

TWO_TOOLS_JSON = '{"tool_calls":[{"name":"get_weather","arguments":{"city":"Moscow"}},{"name":"get_time","arguments":{}}]}'

TOOLS = [
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
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_SCHEMAS = qwen_api.toolemu.tool_schema_map(TOOLS)


def _tool_sse(payload: str) -> str:
    return (
        'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n'
        "\n"
        f'data: {{"choices": [{{"delta": {{"content": {json.dumps(payload)}, "phase": "answer", "status": "typing"}}}}], "response_id": "r1"}}\n'
        "\n"
        'data: {"choices": [{"delta": {"content": "", "role": "assistant", "status": "finished", "phase": "answer"}}], "response_id": "r1"}\n'
        "\n"
    )


TWO_TOOL_SSE = _tool_sse(f"Here: {TWO_TOOLS_JSON}")

ONE_TOOL_SSE = _tool_sse(f"Here: {TOOL_JSON}")


class FakeSession:
    def __init__(self, sid: str = "c1", last_response_id: str | None = None) -> None:
        self.id = sid
        self.last_response_id = last_response_id
        self.accumulated_input_tokens = 0
        self.accumulated_output_tokens = 0


class FakeResp:
    def __init__(self, sse_text: str, status_code: int = 200, content_type: str = "text/event-stream; charset=utf-8") -> None:
        self._b = sse_text.encode()
        self.status_code = status_code
        self.headers = {"content-type": content_type}

    async def aiter_bytes(self):
        yield self._b

    async def aclose(self):
        return None

    async def aread(self):
        return self._b


class CloseErrorResp(FakeResp):
    def __init__(self, sse_text: str) -> None:
        super().__init__(sse_text)
        self.close_attempts = 0

    async def aclose(self):
        self.close_attempts += 1
        raise RuntimeError("close boom")


class FakeAccount:
    def __init__(self, sse_list: list[str] | None = None) -> None:
        self.index = 0
        self.broken = False
        self.client = MagicMock()
        self.client.completion = AsyncMock(side_effect=[FakeResp(item) for item in (sse_list or [])])
        self.client.stop_stream = AsyncMock()
        self.sem = asyncio.Semaphore(1)
        self.sessions = MagicMock()
        self.sessions.obtain = AsyncMock(return_value=(FakeSession(), "s1"))
        self.sessions.get = MagicMock(return_value=None)
        self.sessions.touch_last_message = MagicMock()
        self.sessions.forget = MagicMock()

    def mark_broken(self):
        self.broken = True


@pytest.fixture(autouse=True)
def zero_backoff():
    original = retry_mod.RETRY_BACKOFF_SEC
    retry_mod.RETRY_BACKOFF_SEC = 0.0
    yield
    retry_mod.RETRY_BACKOFF_SEC = original


def _args(acct, pool=None, existing_sid: str | None = "s1", tool_mode: bool = False, **extra):
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


async def _collect(agen) -> list[str]:
    return [item async for item in agen]


def _payloads(joined: str) -> list[dict]:
    out: list[dict] = []
    for line in joined.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        out.append(json.loads(line[len("data: ") :]))
    return out


def _deltas(joined: str, key: str = "content") -> list[str]:
    return [payload["choices"][0]["delta"][key] for payload in _payloads(joined) if payload.get("choices") and key in payload["choices"][0]["delta"]]


def _content(joined: str) -> str:
    return "".join(_deltas(joined))


def _reasons(joined: str) -> list[str | None]:
    return [choice["finish_reason"] for payload in _payloads(joined) if payload.get("choices") for choice in payload["choices"]]


def _tool_deltas(joined: str) -> list[tuple[int, dict]]:
    out: list[tuple[int, dict]] = []
    for payload in _payloads(joined):
        for choice in payload.get("choices") or []:
            for call in choice["delta"].get("tool_calls") or []:
                out.append((choice["index"], call))
    return out


def _tool_names(joined: str, index: int = 0) -> list[str]:
    names: list[str] = []
    for choice_index, call in _tool_deltas(joined):
        name = call["function"].get("name")
        if choice_index == index and name is not None:
            names.append(name)
    return names


def _rec(content: str = "", reasoning: str = "", error: dict | None = None) -> QwenStreamReconstructor:
    rec = QwenStreamReconstructor()
    if content:
        rec.handle(sseutil.SSEEvent(None, {"choices": [{"delta": {"content": content, "phase": "answer"}}]}))
    if reasoning:
        rec.handle(sseutil.SSEEvent(None, {"choices": [{"delta": {"content": reasoning, "phase": "think"}}]}))
    if error is not None:
        rec.handle(sseutil.SSEEvent(None, {"error": error}))
    return rec


def test_incremental_sse_comes_from_sseutil_not_deepseek():
    assert qwen_api.IncrementalSSE is sseutil.IncrementalSSE
    source = (REPO_ROOT / "danyapi" / "qwen" / "api.py").read_text(encoding="utf-8")
    assert "from ..sseutil import IncrementalSSE" in source
    assert "deepseek" not in source


def test_collect_request_defaults_and_in_out_hybrids():
    req = qwen_api._make_request(
        "account",
        "pool",
        "sid",
        "prompt",
        "model",
        True,
        True,
        True,
        {"f": {}},
        ("u1", "u2"),
        None,
        None,
        None,
        None,
        None,
        "t2i",
    )
    assert req.prompt == "prompt"
    assert req.prompt_with_images == "prompt"
    assert req.thinking is True
    assert req.search is True
    assert req.chat_type == "t2i"
    assert req.tool_mode is True
    assert req.tool_schemas == {"f": {}}
    assert req.context_seq == ("u1", "u2")
    assert req.session is None
    assert req.session_key is None
    assert req.had_cached_session is False
    assert req.prepared is False
    assert req.prompt_with_images == qwen_api._append_image_markdown("prompt", None)


def test_collect_request_rebuild_prompt_returns_early_without_messages():
    req = qwen_api._make_request("a", "p", "s", "keep me", "m", False, False, False, None, None, None, None, None, None, None)
    req.rebuild_prompt()
    assert req.prompt == "keep me"
    assert req.prompt_with_images == "keep me"
    assert req.tool_schemas is None
    assert req.tool_mode is False


def test_collect_request_rebuild_prompt_carries_history_tools_and_images():
    messages = [
        SimpleNamespace(role="system", content="be terse"),
        SimpleNamespace(role="user", content="earlier"),
        SimpleNamespace(role="assistant", content="answered"),
        SimpleNamespace(content=[{"type": "image_url", "image_url": {"url": "https://x/a.png"}}]),
    ]
    req = qwen_api._make_request("a", "p", "s", "delta only", "m", False, False, False, None, None, messages, TOOLS, "auto", None, None)
    req.rebuild_prompt()
    expected, expected_mode = qwen_api.toolemu.build_prompt(messages, TOOLS, "auto", False, None)
    assert req.prompt == expected
    assert req.tool_mode is expected_mode
    assert req.tool_schemas == TOOL_SCHEMAS
    assert req.prompt_with_images == f"{expected}\n\n![image](https://x/a.png)"


async def test_collect_request_prepared_flag_short_circuits_ensure():
    req = qwen_api._make_request("a", "p", "s", "prompt", "m", False, False, False, None, None, None, None, None, None, None)
    req.prepared = True
    await qwen_api._ensure_prepared(req)
    assert req.session is None
    assert req.had_cached_session is False


def test_image_uris_skip_content_that_is_not_a_list_of_parts():
    assert list(qwen_api._iter_image_uris([SimpleNamespace(content="just text")])) == []
    message = SimpleNamespace(content=[{"type": "text", "text": "hi"}, "junk", {"type": "image_url", "image_url": 42}])
    assert list(qwen_api._iter_image_uris([message])) == []


def test_error_detail_is_a_plain_string_in_every_arm():
    assert qwen_api._error_detail(_rec(error={"code": "x", "details": "boom"})) == "boom"
    assert qwen_api._error_detail(_rec(error={"code": "x", "message": "only message"})) == "only message"
    assert qwen_api._error_detail(_rec(error={"code": "x", "details": "", "message": ""})) == "Qwen server error, try again later"
    assert qwen_api._error_detail(_rec(error=None)) == "Qwen server error, try again later"
    assert qwen_api._error_detail(_rec(error={"code": "x", "details": 42})) == "42"
    assert qwen_api._error_detail(_rec(error={"code": "x", "details": {}})) == "Qwen server error, try again later"


def test_error_code_normalises_every_arm():
    assert error_code(None) is None
    assert error_code({}) is None
    assert error_code({"code": "Busy"}) == "Busy"
    assert error_code({"code": 40014}) == "40014"
    assert error_code({"code": 40014.7}) == "40014"
    assert error_code({"code": True}) is None
    assert error_code({"code": False}) is None
    assert error_code({"code": ["x"]}) is None
    assert error_code({"code": {"x": 1}}) is None


def test_numeric_code_is_not_mistaken_for_a_named_class():
    assert error_code({"code": 40014}) not in qwen_api.RETRYABLE_ERROR_CODES | qwen_api.AUTH_ERROR_CODES
    assert qwen_api._error_status(error_code({"code": 40014})) == 502
    assert qwen_api._is_context_limit(_rec(error={"code": 40014, "details": "too long"})) is False
    assert error_code({"code": 400001}) not in qwen_api.RETRYABLE_ERROR_CODES


def test_error_code_arms_change_the_http_classification():
    assert qwen_api._error_status(error_code({"code": "Too_Many_Requests"})) == 429
    assert qwen_api._error_status(error_code({"code": 429})) == 502
    assert qwen_api._error_status(error_code({"code": ["Too_Many_Requests"]})) == 502
    assert qwen_api._error_status(error_code({"code": True})) == 502


def test_summary_and_think_reasoning_are_joined_without_duplication():
    rec = QwenStreamReconstructor()
    rec.handle(sseutil.SSEEvent(None, {"choices": [{"delta": {"content": "thinking hard", "phase": "think"}}]}))
    assert rec.take_diffs() == ("", "thinking hard")
    summary = {"choices": [{"delta": {"phase": "thinking_summary", "extra": {"summary_thought": {"content": ["a summary"]}}}}]}
    rec.handle(sseutil.SSEEvent(None, summary))
    assert rec.take_diffs() == ("", "a summary")
    assert rec.reasoning == "thinking hard\n\na summary"
    assert rec.take_diffs() == ("", "")


def test_summary_then_think_keeps_both_parts_in_order():
    rec = QwenStreamReconstructor()
    items = {"choices": [{"delta": {"phase": "thinking_summary", "extra": {"summary_thought": {"content": [{"text": "first"}, "second", 7, {}]}}}}]}
    rec.handle(sseutil.SSEEvent(None, items))
    assert rec.take_diffs() == ("", "first\n\nsecond")
    rec.handle(sseutil.SSEEvent(None, {"choices": [{"delta": {"content": "later", "phase": "think"}}]}))
    assert rec.take_diffs() == ("", "later")
    assert rec.reasoning == "later\n\nfirst\n\nsecond"
    assert rec.take_diffs() == ("", "")


def test_summary_only_and_think_only_reasoning():
    only_summary = QwenStreamReconstructor()
    payload = {"choices": [{"delta": {"phase": "thinking_summary", "extra": {"summary_thought": {"content": ["just a summary"]}}}}]}
    only_summary.handle(sseutil.SSEEvent(None, payload))
    assert only_summary.take_diffs() == ("", "just a summary")
    assert only_summary.reasoning == "just a summary"
    only_think = QwenStreamReconstructor()
    only_think.handle(sseutil.SSEEvent(None, {"choices": [{"delta": {"content": "just thinking", "phase": "think"}}]}))
    assert only_think.take_diffs() == ("", "just thinking")
    assert only_think.reasoning == "just thinking"


def test_a_diverging_summary_resends_nothing():
    rec = QwenStreamReconstructor()
    first = {"choices": [{"delta": {"phase": "thinking_summary", "extra": {"summary_thought": {"content": ["original"]}}}}]}
    rec.handle(sseutil.SSEEvent(None, first))
    assert rec.take_diffs() == ("", "original")
    second = {"choices": [{"delta": {"phase": "thinking_summary", "extra": {"summary_thought": {"content": ["a rewrite"]}}}}]}
    rec.handle(sseutil.SSEEvent(None, second))
    assert rec.take_diffs() == ("", "")
    assert rec.reasoning == "a rewrite"


def test_summary_delta_helper_arms():
    from danyapi.qwen.stream import _summary_delta

    assert _summary_delta("", "fresh") == "fresh"
    assert _summary_delta("par", "parented") == "ented"
    assert _summary_delta("original", "a rewrite") == ""


def test_image_tail_above_the_window_limit_is_still_scanned():
    rec = QwenStreamReconstructor()
    payload = "z" * (4096 + 10) + " ![x](https://cdn.qwenlm.ai/tail.png)"
    rec.handle(sseutil.SSEEvent(None, {"choices": [{"delta": {"content": payload, "phase": "image"}}]}))
    assert rec.image_urls == ["https://cdn.qwenlm.ai/tail.png"]


def test_empty_delta_does_not_start_an_image_scan():
    rec = QwenStreamReconstructor()
    rec._collect_image_urls("")
    assert rec._image_scan_tail == ""
    rec.handle(sseutil.SSEEvent(None, {"choices": [{"delta": {"content": "", "phase": "answer"}}]}))
    assert rec._image_scan_tail == ""


def test_token_int_reads_numeric_strings_and_bounds_floats():
    rec = QwenStreamReconstructor()
    rec.usage = {"input_tokens": "12", "output_tokens": "1e3", "total_tokens": "nan"}
    assert rec.usage_tokens == {"prompt_tokens": 12, "completion_tokens": 1000, "total_tokens": 0}
    rec.usage = {"input_tokens": 1e30, "output_tokens": -1e30, "total_tokens": 0}
    assert rec.usage_tokens == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    rec.usage = {"input_tokens": "not a number", "output_tokens": "  ", "total_tokens": "inf"}
    assert rec.usage_tokens == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_finalize_recovers_a_url_split_at_the_very_end():
    rec = QwenStreamReconstructor()
    incremental = sseutil.IncrementalSSE()
    for event in incremental.feed(TAIL_IMAGE_SSE.encode()):
        rec.handle(event)
    assert rec.image_urls == []
    rec.finalize()
    assert rec.image_urls == ["https://cdn.qwenlm.ai/tail.png"]
    rec.finalize()
    assert rec.image_urls == ["https://cdn.qwenlm.ai/tail.png"]


def test_extra_image_url_string_is_deduped():
    rec = QwenStreamReconstructor()
    delta = {"phase": "image", "extra": {"url": "https://cdn.qwenlm.ai/direct.png"}}
    rec.handle(sseutil.SSEEvent(None, {"choices": [{"delta": dict(delta)}]}))
    rec.handle(sseutil.SSEEvent(None, {"choices": [{"delta": dict(delta)}]}))
    assert rec.image_urls == ["https://cdn.qwenlm.ai/direct.png"]


def test_qwen_error_code_is_always_hashable_and_classifiable():
    with pytest.raises(QwenError) as excinfo:
        QwenClient._biz({"success": False, "data": {"code": {"nested": True}, "details": "nope"}})
    assert excinfo.value.code == -1
    assert isinstance(excinfo.value.code, int)
    assert hash(excinfo.value.code) == hash(-1)
    assert qwen_api._error_status(excinfo.value.code) == 502
    with pytest.raises(QwenError) as listy:
        QwenClient._biz({"success": False, "code": ["Too_Many_Requests"], "details": "nope"})
    assert listy.value.code == -1
    with pytest.raises(QwenError) as booly:
        QwenClient._biz({"success": False, "code": True})
    assert booly.value.code == -1
    with pytest.raises(QwenError) as numeric:
        QwenClient._biz({"success": False, "code": 40014.4, "message": "weird"})
    assert numeric.value.code == 40014
    assert numeric.value.message == "weird"


def test_error_code_helper_arms():
    assert _error_code(True) == -1
    assert _error_code(1.9) == 1
    assert _error_code("Busy") == "Busy"
    assert _error_code({"a": 1}) == -1
    assert _error_code([1]) == -1
    assert _error_code(None) == -1


def test_timezone_cache_double_check_returns_the_winner(monkeypatch):
    calls: list[int] = []

    def fake_timezone_header() -> str:
        calls.append(1)
        return "DIRECT-VALUE"

    class RacingLock:
        def __init__(self) -> None:
            self._real = threading.Lock()

        def __enter__(self):
            held = self._real.__enter__()
            qwen_client._tz_cache.value = "RACE-VALUE"
            qwen_client._tz_cache.ts = qwen_client.time.monotonic()
            return held

        def __exit__(self, *exc):
            return self._real.__exit__(*exc)

    monkeypatch.setattr(qwen_client, "_tz_cache", qwen_client._TzCache())
    monkeypatch.setattr(qwen_client, "_tz_lock", RacingLock())
    monkeypatch.setattr(qwen_client, "timezone_header", fake_timezone_header)
    assert _cached_timezone_header() == "RACE-VALUE"
    assert calls == []


def test_timezone_cache_is_shared_until_the_ttl_expires(monkeypatch):
    monkeypatch.setattr(qwen_client, "_tz_cache", qwen_client._TzCache())
    first = _cached_timezone_header()
    assert "GMT" in first
    assert _cached_timezone_header() == first
    monkeypatch.setattr(qwen_client, "_tz_cache", qwen_client._TzCache())
    assert _cached_timezone_header() == first


def test_new_uuid_is_a_uuid():
    value = new_uuid()
    assert len(value) == 36
    assert value.count("-") == 4


async def test_non_json_body_is_logged_and_never_embedded_in_the_error(caplog):
    client = QwenClient()
    resp = SimpleNamespace(headers={"content-type": "text/html"}, text="<html>secret marker</html>")
    try:
        with caplog.at_level(logging.WARNING, logger="danyapi.qwen"):
            with pytest.raises(QwenError) as excinfo:
                client._parse_json(resp, "/api/v1/auths/")
    finally:
        await client.aclose()
    assert excinfo.value.code == -1
    assert "secret marker" not in str(excinfo.value)
    assert excinfo.value.message == "unexpected non-JSON response from /api/v1/auths/"
    assert "secret marker" in caplog.text
    assert "/api/v1/auths/" in caplog.text


async def test_only_completion_widens_the_read_timeout():
    client = QwenClient(timeout=60.0)
    sent: dict = {}
    original_build = client.http.build_request

    def build_request(*args, **kwargs):
        sent["timeout"] = kwargs.get("timeout")
        return original_build(*args, **kwargs)

    async def send(request, **kwargs):
        if kwargs.get("stream"):
            return SimpleNamespace(status_code=200)
        return SimpleNamespace(headers={"content-type": "application/json"}, json=lambda: {"success": True, "data": {"id": "chat1"}})

    client.http.build_request = build_request
    client.http.send = send
    try:
        assert client.http.timeout.read == 60.0
        assert client._stream_timeout.read == 300.0
        assert client._stream_timeout.connect == 60.0
        await client.completion(chat_session_id="c1", prompt="p", parent_message_id=None, model="m")
        assert sent["timeout"] is client._stream_timeout
        sent.pop("timeout")
        await client.stop_stream(chat_session_id="c1", response_id="r1")
        assert sent["timeout"] is httpx.USE_CLIENT_DEFAULT
        sent.pop("timeout")
        assert await client.create_chat(model="m") == "chat1"
        assert sent["timeout"] is httpx.USE_CLIENT_DEFAULT
    finally:
        client.http.build_request = original_build
        await client.aclose()


async def test_check_auth_accepts_every_documented_shape():
    payloads = {
        "success": ({"success": True}, True),
        "nested": ({"data": {"id": "u1"}}, True),
        "flat": ({"id": "u1"}, True),
        "empty": ({}, False),
        "list": ([1, 2], False),
        "no-id": ({"foo": "bar"}, False),
    }
    for name, (payload, expected) in payloads.items():
        client = QwenClient()
        resp = SimpleNamespace(status_code=200)
        resp.json = MagicMock(return_value=payload)
        client.http.get = AsyncMock(return_value=resp)
        try:
            assert await client.check_auth() is expected, name
        finally:
            await client.aclose()


def test_serialize_reads_every_session_field():
    registry = QwenSessionRegistry(MagicMock(), 8, 0)
    session = QwenSession(id="c1", title="t", last_response_id="r1", model="m", accumulated_input_tokens=5, accumulated_output_tokens=7)
    assert registry._serialize(session) == {
        "id": "c1",
        "title": "t",
        "last_response_id": "r1",
        "model": "m",
        "accumulated_input_tokens": 5,
        "accumulated_output_tokens": 7,
    }


def test_serialize_cannot_hide_a_renamed_field():
    registry = QwenSessionRegistry(MagicMock(), 8, 0)
    with pytest.raises(AttributeError, match="title"):
        registry._serialize(SimpleNamespace(id="c1"))


def test_deserialize_rebuilds_every_session_field():
    registry = QwenSessionRegistry(MagicMock(), 8, 0)
    record = {"id": "c1", "title": "t", "last_response_id": "r1", "model": "m", "accumulated_input_tokens": "5", "accumulated_output_tokens": 7.9}
    assert registry._deserialize(record) == QwenSession(
        id="c1", title="t", last_response_id="r1", model="m", accumulated_input_tokens=5, accumulated_output_tokens=7
    )
    assert registry._deserialize({"id": "c2", "title": "", "last_response_id": None, "model": None}) == QwenSession(id="c2")
    for bad in (None, [], {}, {"id": ""}):
        with pytest.raises(ValueError, match="invalid session record"):
            registry._deserialize(bad)


async def test_obtain_writes_the_serialized_session_to_the_store():
    client = MagicMock(spec=QwenClient)
    client.create_chat = AsyncMock(return_value="fresh")
    store = JsonStore("qwen-cov2")
    registry = QwenSessionRegistry(client, 8, 0, store=store, key_prefix="0:")
    session, key = await registry.obtain(None, model="qwen3.8-max")
    assert key == "fresh"
    assert session.model == "qwen3.8-max"
    assert store.get("0:fresh") == {
        "id": "fresh",
        "title": "",
        "last_response_id": None,
        "model": "qwen3.8-max",
        "accumulated_input_tokens": 0,
        "accumulated_output_tokens": 0,
    }
    registry.touch_last_message("fresh", "r1")
    assert registry.get("fresh").last_response_id == "r1"
    assert store.get("0:fresh")["last_response_id"] == "r1"
    restored = QwenSessionRegistry(client, 8, 0, store=store, key_prefix="0:")
    assert restored.get("fresh") == QwenSession(id="fresh", last_response_id="r1", model="qwen3.8-max")


async def test_reuse_is_a_pure_predicate_and_a_stale_session_is_replaced():
    client = MagicMock(spec=QwenClient)
    client.create_chat = AsyncMock(return_value="brand-new")
    registry = QwenSessionRegistry(client, 8, 0)
    stale = QwenSession(id="c1", model=None)
    registry._sessions["c1"] = (stale, registry._now())
    assert registry._reuse(stale, "c1", model="qwen3.8-max") is False
    assert stale.model is None
    assert registry.can_reuse("c1", model="qwen3.8-max") is False
    session, key = await registry.obtain("c1", model="qwen3.8-max")
    assert key == "c1"
    assert session is not stale
    assert session.id == "brand-new"
    assert session.model == "qwen3.8-max"
    assert stale.model is None
    assert registry.get("c1") is session
    assert registry._reuse(session, "c1", model="qwen3.8-max") is True
    assert registry._reuse(session, "c1", model="other") is False
    assert session.model == "qwen3.8-max"


async def test_obtain_defaults_the_model_to_an_empty_string():
    client = MagicMock(spec=QwenClient)
    client.create_chat = AsyncMock(return_value="c9")
    registry = QwenSessionRegistry(client, 8, 0)
    session, key = await registry.obtain(None)
    assert (key, session.model) == ("c9", "")
    client.create_chat.assert_awaited_once_with(model="", chat_mode="normal")


async def test_account_tracks_state_and_marks_itself_broken_once():
    client = MagicMock(spec=QwenClient)
    client.create_chat = AsyncMock(return_value="c1")
    account = QwenAccount(3, client, session_cache_size=4, stable_id="qwen-3")
    assert account.index == 3
    assert account.client is client
    assert account.stable_id == "qwen-3"
    assert account.label == "qwen-acct#3"
    assert account.sem.locked() is False
    assert isinstance(account.sessions, QwenSessionRegistry)
    assert account.broken is False
    assert account.broken_at is None
    account.mark_broken()
    assert account.broken is True
    stamp = account.broken_at
    assert stamp is not None
    account.mark_broken()
    assert account.broken_at == stamp


def test_duckai_account_keeps_the_shared_account_shape():
    from danyapi.duckai.accounts import DuckAIAccount

    assert DuckAIAccount.__slots__ == ("broken", "broken_at", "client", "index", "sem", "stable_id")
    account = DuckAIAccount(1, MagicMock(), stable_id="duckai")
    assert account.label == "duckai-acct#1"
    assert account.broken is False
    account.mark_broken()
    assert account.broken is True
    assert account.broken_at is not None


def test_build_limited_message_returns_tool_calls_and_trims_to_the_budget():
    prefix = "Here is a rather long prefix before the call "
    parsed = _rec(f"{prefix}{TWO_TOOLS_JSON}")
    message, finish = qwen_api._build_limited_message(parsed, True, TOOL_SCHEMAS, None, None, None)
    assert finish == "tool_calls"
    assert [call["function"]["name"] for call in message["tool_calls"]] == ["get_weather", "get_time"]
    assert [call["function"]["arguments"] for call in message["tool_calls"]] == ['{"city": "Moscow"}', "{}"]
    assert message["content"] == prefix.rstrip()
    trimmed, trimmed_finish = qwen_api._build_limited_message(parsed, True, TOOL_SCHEMAS, 2, None, None)
    assert trimmed_finish == "length"
    assert len(trimmed["content"]) < len(message["content"])
    assert message["content"].startswith(trimmed["content"][:4])
    single, _ = qwen_api._build_limited_message(parsed, True, TOOL_SCHEMAS, None, None, False)
    assert [call["function"]["name"] for call in single["tool_calls"]] == ["get_weather"]


def test_build_limited_message_reports_length_for_filtered_tool_mode_text():
    long_text = "word " * 200
    message, finish = qwen_api._build_limited_message(_rec(long_text), True, TOOL_SCHEMAS, 2, None, None)
    assert finish == "length"
    assert message == {"role": "assistant", "content": long_text[: len(message["content"])]}
    plain, plain_finish = qwen_api._build_limited_message(_rec(long_text, reasoning="because"), False, None, 2, None, None)
    assert plain_finish == "length"
    assert plain["reasoning_content"] == "because"
    stopped, stopped_finish = qwen_api._build_limited_message(_rec("keep<END>drop"), False, None, None, "<END>", None)
    assert (stopped["content"], stopped_finish) == ("keep", "stop")


async def test_collect_response_rebuilds_a_stale_session_and_bills_the_rebuilt_history():
    messages = [
        SimpleNamespace(role="user", content="earlier turn"),
        SimpleNamespace(role="assistant", content="earlier answer"),
        SimpleNamespace(role="user", content="now"),
    ]
    acct = FakeAccount()
    cached = FakeSession()
    fresh = FakeSession(sid="c2")
    acct.sessions.get = MagicMock(return_value=cached)
    acct.sessions.obtain = AsyncMock(side_effect=[(cached, "s1"), (fresh, "c2")])
    acct.client.completion = AsyncMock(side_effect=[HTTPException(400, "stale chat"), FakeResp(OK_SSE)])
    result = await qwen_api.collect_non_stream(**_args(acct, messages=messages, cached_session=cached))
    assert result["choices"][0]["message"]["content"] == "Hello world"
    assert result["session_id"] == "c2"
    assert acct.client.completion.await_count == 2
    assert "earlier turn" in acct.client.completion.await_args_list[1].kwargs["prompt"]
    expected_prompt, _ = qwen_api.toolemu.build_prompt(messages, None, None, False, None)
    assert result["usage"]["prompt_tokens"] == estimate_tokens(expected_prompt)


async def test_collect_response_reraises_the_stale_error_when_the_rebuild_fails():
    messages = [SimpleNamespace(role="system", content="be terse")]
    acct = FakeAccount()
    acct.sessions.get = MagicMock(return_value=FakeSession())
    acct.client.completion = AsyncMock(side_effect=[HTTPException(400, "stale chat")])
    pool = MagicMock()
    with pytest.raises(HTTPException) as excinfo:
        await qwen_api.collect_non_stream(**_args(acct, pool=pool, messages=messages))
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "stale chat"
    assert acct.client.completion.await_count == 1
    pool.forget.assert_called_once_with("s1")
    assert acct.broken is False


async def test_collect_response_survives_a_failing_response_close():
    acct = FakeAccount()
    acct.client.completion = AsyncMock(return_value=CloseErrorResp(OK_SSE))
    result = await qwen_api.collect_non_stream(**_args(acct))
    assert result["choices"][0]["message"]["content"] == "Hello world"


async def test_collect_response_sleeps_after_releasing_the_account_lock(monkeypatch):
    asked: list[float] = []

    async def recorder(delay: float) -> None:
        asked.append(delay)
        await REAL_SLEEP(0)

    monkeypatch.setattr(qwen_api.asyncio, "sleep", recorder)
    monkeypatch.setattr(retry_mod, "RETRY_BACKOFF_SEC", 3.0)
    order: list[str] = []
    acct = FakeAccount()
    acct.client.completion = AsyncMock(side_effect=[HTTPException(429, "slow"), FakeResp(OK_SSE)])

    async def probe() -> None:
        async with acct.sem:
            order.append("probe")

    task = asyncio.create_task(probe())
    result = await qwen_api.collect_non_stream(**_args(acct))
    await task
    assert result["choices"][0]["message"]["content"] == "Hello world"
    assert asked == [3.0]
    assert order == ["probe"]


async def test_collect_non_stream_expands_choices_for_n():
    acct = FakeAccount([OK_SSE])
    result = await qwen_api.collect_non_stream(**_args(acct, n=3))
    assert [choice["index"] for choice in result["choices"]] == [0, 1, 2]
    assert len({id(choice) for choice in result["choices"]}) == 3
    assert result["choices"][1]["message"] == result["choices"][0]["message"]


async def test_collect_image_returns_urls_and_bills_the_t2i_usage():
    image_sse = (
        'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
        'data: {"choices": [{"delta": {"content": "![img](https://cdn.qwenlm.ai/a.png)", "phase": "image"}}], "response_id": "r1"}\n\n'
        'data: {"usage": {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42}}\n\n'
    )
    acct = FakeAccount()
    session = FakeSession()
    acct.sessions.obtain = AsyncMock(return_value=(session, "c1"))
    acct.client.completion = AsyncMock(return_value=FakeResp(image_sse))
    result = await qwen_api.collect_image(acct, MagicMock(), None, acct.sem, "draw a cat", "qwen3.8-max", "qwen3.8-max", ("u1",), user="alice")
    assert result["image_urls"] == ["https://cdn.qwenlm.ai/a.png"]
    assert result["revised_prompt"] == "![img](https://cdn.qwenlm.ai/a.png)"
    assert result["usage"] == {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42}
    assert result["session_id"] == "c1"
    assert session.accumulated_input_tokens == 30
    assert session.accumulated_output_tokens == 12
    assert acct.client.completion.await_args.kwargs["chat_type"] == "t2i"
    assert acct.client.completion.await_args.kwargs["prompt"] == "draw a cat"
    again = await qwen_api.collect_image(acct, MagicMock(), None, acct.sem, "draw a dog", "qwen3.8-max", "qwen3.8-max")
    assert again["usage"] == {
        "prompt_tokens": estimate_tokens("draw a dog"),
        "completion_tokens": estimate_tokens("![img](https://cdn.qwenlm.ai/a.png)"),
        "total_tokens": 10,
    }


async def test_collect_image_rejects_a_context_limit():
    acct = FakeAccount()
    acct.client.completion = AsyncMock(return_value=FakeResp(CTX_SSE))
    pool = MagicMock()
    with pytest.raises(HTTPException) as excinfo:
        await qwen_api.collect_image(acct, pool, None, acct.sem, "p", "m", "m")
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == qwen_api.CONTEXT_LIMIT_MESSAGE
    pool.forget.assert_called_once_with("s1")


async def test_collect_image_surfaces_a_stream_error():
    acct = FakeAccount()
    acct.client.completion = AsyncMock(return_value=FakeResp(BUSY_SSE))
    with pytest.raises(HTTPException) as excinfo:
        await qwen_api.collect_image(acct, MagicMock(), None, acct.sem, "p", "m", "m")
    assert excinfo.value.status_code == 429
    assert excinfo.value.detail == "please slow down"


async def test_stream_calls_finalize_so_a_split_tail_image_url_is_recovered(monkeypatch):
    seen: list[list[str]] = []
    original = QwenStreamReconstructor.finalize

    def spy(self):
        original(self)
        seen.append(list(self.image_urls))

    monkeypatch.setattr(QwenStreamReconstructor, "finalize", spy)
    acct = FakeAccount([TAIL_IMAGE_SSE])
    lines = await _collect(qwen_api.stream_openai(**_args(acct)))
    assert seen == [["https://cdn.qwenlm.ai/tail.png"]]
    assert _content("".join(lines)) == "see ![img](https://cdn.qwenlm.ai/tail.png"


async def test_stream_reports_a_failed_session_setup_as_an_in_band_error():
    acct = FakeAccount()
    pool = MagicMock()
    acct.sessions.obtain = AsyncMock(side_effect=QwenError("unauthorized", "bad token"))
    lines = await _collect(qwen_api.stream_openai(**_args(acct, pool=pool)))
    joined = "".join(lines)
    assert acct.broken is True
    payload = json.loads(lines[0][len("data: ") :])
    assert payload["error"]["message"] == "Qwen error: Qwen error unauthorized: bad token"
    assert _reasons(joined) == ["error"]
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_rebuilds_a_stale_session_and_reuses_the_fresh_one():
    messages = [
        SimpleNamespace(role="user", content="earlier turn"),
        SimpleNamespace(role="assistant", content="earlier answer"),
        SimpleNamespace(role="user", content="now"),
    ]
    acct = FakeAccount()
    cached = FakeSession()
    fresh = FakeSession(sid="c2")
    acct.sessions.get = MagicMock(return_value=cached)
    acct.sessions.obtain = AsyncMock(side_effect=[(cached, "s1"), (fresh, "c2")])
    acct.client.completion = AsyncMock(side_effect=[HTTPException(404, "stale chat"), FakeResp(OK_SSE)])
    lines = await _collect(qwen_api.stream_openai(**_args(acct, messages=messages, cached_session=cached)))
    joined = "".join(lines)
    assert acct.client.completion.await_count == 2
    assert "earlier turn" in acct.client.completion.await_args_list[1].kwargs["prompt"]
    assert '"session_id": "c2"' in joined
    assert _content(joined) == "Hello world"
    acct.client.stop_stream.assert_not_awaited()


async def test_stream_reports_a_stale_session_whose_rebuild_fails():
    messages = [SimpleNamespace(role="system", content="be terse")]
    acct = FakeAccount()
    acct.sessions.get = MagicMock(return_value=FakeSession())
    acct.client.completion = AsyncMock(side_effect=[HTTPException(400, "stale chat")])
    lines = await _collect(qwen_api.stream_openai(**_args(acct, messages=messages)))
    joined = "".join(lines)
    assert acct.client.completion.await_count == 1
    assert "stale chat" in joined
    assert _reasons(joined) == ["error"]
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_reports_a_stale_session_whose_fresh_chat_cannot_be_created():
    messages = [
        SimpleNamespace(role="user", content="earlier turn"),
        SimpleNamespace(role="assistant", content="earlier answer"),
        SimpleNamespace(role="user", content="now"),
    ]
    acct = FakeAccount()
    acct.sessions.get = MagicMock(return_value=FakeSession())
    acct.sessions.obtain = AsyncMock(side_effect=[(FakeSession(), "s1"), QwenError("unauthorized", "bad token")])
    acct.client.completion = AsyncMock(side_effect=[HTTPException(400, "stale chat")])
    lines = await _collect(qwen_api.stream_openai(**_args(acct, messages=messages)))
    joined = "".join(lines)
    assert "bad token" in joined
    assert "stale chat" not in joined
    assert _reasons(joined) == ["error"]
    assert acct.broken is True
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_pumps_the_events_flushed_by_the_incremental_parser():
    tail_only = 'data: {"choices": [{"delta": {"content": "TailOnly", "phase": "answer"}}]}'
    acct = FakeAccount()
    acct.client.completion = AsyncMock(return_value=FakeResp(tail_only))
    lines = await _collect(qwen_api.stream_openai(**_args(acct)))
    assert _content("".join(lines)) == "TailOnly"


async def test_stream_stops_upstream_when_the_response_cannot_be_closed():
    acct = FakeAccount()
    acct.client.completion = AsyncMock(return_value=CloseErrorResp(OK_SSE))
    lines = await _collect(qwen_api.stream_openai(**_args(acct)))
    joined = "".join(lines)
    assert _content(joined) == "Hello world"
    acct.client.stop_stream.assert_awaited_once_with("c1", "r1")


async def test_stream_retry_after_a_filtered_tool_attempt_rebuilds_the_visible_offset(monkeypatch):
    monkeypatch.setattr(qwen_api, "_is_retryable_error", lambda rec: True)
    stale = (
        'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
        f'data: {{"choices": [{{"delta": {{"content": {json.dumps(TOOL_JSON)}, "phase": "answer", "status": "typing"}}}}], "response_id": "r1"}}\n\n'
        'data: {"error": {"code": "Too_Many_Requests", "details": "please slow down"}, "response_id": "r1"}\n\n'
    )
    acct = FakeAccount()
    acct.client.completion = AsyncMock(side_effect=[FakeResp(stale), FakeResp(OK_SSE)])
    args = _args(acct, tool_mode=True, tool_schemas=TOOL_SCHEMAS, max_tokens=4)
    joined = "".join(await _collect(qwen_api.stream_openai(**args)))
    assert acct.client.completion.await_count == 2
    assert _content(joined) == "Hello world"
    assert "Moscow" not in joined
    assert _tool_names(joined) == []
    assert _reasons(joined)[-1] == "stop"
    acct.client.stop_stream.assert_awaited_once_with("c1", "r1")


async def test_stream_retry_after_a_stopped_attempt_rebuilds_the_filters(monkeypatch):
    monkeypatch.setattr(qwen_api, "_is_retryable_error", lambda rec: True)
    stopped = (
        'data: {"response.created":{"response_id":"r1"}} \n\n'
        'data: {"choices": [{"delta": {"content": "<END>tail", "phase": "answer", "status": "typing"}}], "response_id": "r1"}\n\n'
        'data: {"error": {"code": "Too_Many_Requests", "details": "please slow down"}, "response_id": "r1"}\n\n'
    )
    acct = FakeAccount()
    acct.client.completion = AsyncMock(side_effect=[FakeResp(stopped), FakeResp(OK_SSE)])
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct, stop="<END>", max_tokens=4))))
    assert acct.client.completion.await_count == 2
    assert _content(joined) == "Hello world"
    assert "tail" not in joined
    assert _reasons(joined)[-1] == "stop"


async def test_stream_retries_an_attempt_whose_only_delta_was_never_delivered():
    both = (
        'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
        'data: {"choices": [{"delta": {"content": "dropped", "phase": "answer"}}],'
        ' "error": {"code": "Too_Many_Requests", "details": "please slow down"}, "response_id": "r1"}\n\n'
    )
    acct = FakeAccount()
    acct.client.completion = AsyncMock(side_effect=[FakeResp(both), FakeResp(OK_SSE)])
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct))))
    assert acct.client.completion.await_count == 2
    assert _content(joined) == "Hello world"
    assert "dropped" not in joined
    assert _reasons(joined)[-1] == "stop"


async def test_stream_context_limit_after_partial_content_is_reported_as_length():
    acct = FakeAccount([CTX_AFTER_CONTENT_SSE])
    pool = MagicMock()
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct, pool=pool))))
    assert '"finish_reason": "length"' in joined
    assert _reasons(joined)[-1] == "length"
    assert qwen_api.CONTEXT_LIMIT_MESSAGE in joined
    assert '"usage"' not in joined
    pool.forget.assert_called_once_with("s1")
    acct.sessions.forget.assert_called_once_with("s1")
    assert acct.broken is False


async def test_every_stream_finish_reason_is_an_openai_reason():
    cases = {
        "busy": BUSY_SSE,
        "context": CTX_SSE,
        "ok": OK_SSE,
        "empty": 'data: {"choices": [{"delta": {"content": "", "status": "finished", "phase": "answer"}}]}\n\n',
    }
    for name, payload in cases.items():
        acct = FakeAccount()
        acct.client.completion = AsyncMock(side_effect=[FakeResp(payload)] * (qwen_api.MAX_RETRIES + 1))
        joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct))))
        reasons = [reason for reason in _reasons(joined) if reason is not None]
        assert reasons, name
        for reason in reasons:
            assert reason in OPENAI_FINISH_REASONS, (name, reason)
        assert "Too_Many_Requests" not in set(reasons)
        assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_releases_the_account_before_the_terminator():
    acct = FakeAccount([OK_SSE])
    seen: list[tuple[str, bool]] = []
    async for line in qwen_api.stream_openai(**_args(acct, include_usage=True)):
        seen.append((line, acct.sem.locked()))
    finish_index = next(index for index, (line, _) in enumerate(seen) if '"finish_reason": "stop"' in line)
    assert [locked for _, locked in seen[:finish_index]] == [True] * finish_index
    assert [locked for _, locked in seen[finish_index:]] == [False] * (len(seen) - finish_index)
    assert '"usage"' in seen[finish_index + 1][0]
    assert seen[-1][0] == "data: [DONE]\n\n"


async def test_stream_hands_the_tail_to_the_next_session():
    acct = FakeAccount([OK_SSE])
    acquired = asyncio.Event()
    order: list[str] = []
    finish_at = -1

    async def probe() -> None:
        await acct.sem.acquire()
        order.append("probe")
        acquired.set()
        acct.sem.release()

    async for line in qwen_api.stream_openai(**_args(acct)):
        order.append("chunk")
        if '"finish_reason": "stop"' in line:
            finish_at = len(order) - 1
            assert acct.sem.locked() is False
            task = asyncio.create_task(probe())
            await asyncio.wait_for(acquired.wait(), 1)
            assert task.done()
    assert order.count("probe") == 1
    assert order.index("probe") > finish_at
    assert acct.sem.locked() is False


async def test_stream_tool_mode_emits_the_visible_prefix_and_the_tool_deltas():
    acct = FakeAccount([TWO_TOOL_SSE])
    args = _args(acct, tool_mode=True, tool_schemas=TOOL_SCHEMAS)
    joined = "".join(await _collect(qwen_api.stream_openai(**args)))
    assert _tool_names(joined) == ["get_weather", "get_time"]
    assert _content(joined) == "Here: "
    arguments = "".join(call["function"].get("arguments", "") for _index, call in _tool_deltas(joined) if "arguments" in call["function"])
    assert arguments == '{"city": "Moscow"}{}'
    assert _reasons(joined)[-1] == "tool_calls"


async def test_stream_tool_mode_trims_to_one_call_and_repeats_it_for_each_choice():
    acct = FakeAccount([TWO_TOOL_SSE])
    args = _args(acct, tool_mode=True, tool_schemas=TOOL_SCHEMAS, parallel_tool_calls=False, n=2)
    joined = "".join(await _collect(qwen_api.stream_openai(**args)))
    assert _tool_names(joined, 0) == ["get_weather"]
    assert _tool_names(joined, 1) == ["get_weather"]
    assert _reasons(joined)[-1] == "tool_calls"
    assert _reasons(joined).count("tool_calls") == 2


async def test_stream_tool_mode_falls_back_to_the_remaining_text(monkeypatch):
    monkeypatch.setattr(qwen_api.toolemu, "parse_tool_calls", lambda text, schemas, *args, **kwargs: None)
    acct = FakeAccount([ONE_TOOL_SSE])
    args = _args(acct, tool_mode=True, tool_schemas=TOOL_SCHEMAS)
    joined = "".join(await _collect(qwen_api.stream_openai(**args)))
    assert _content(joined) == f"Here: {TOOL_JSON}"
    assert _tool_names(joined) == []
    assert _reasons(joined)[-1] == "stop"


async def test_stream_stop_marker_found_in_the_flushed_tail():
    sse = (
        'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
        'data: {"choices": [{"delta": {"content": "keep<", "phase": "answer"}}], "response_id": "r1"}\n\n'
    )
    acct = FakeAccount()
    acct.client.completion = AsyncMock(return_value=FakeResp(sse))
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct, stop="<"))))
    assert _content(joined) == "keep"
    assert _reasons(joined)[-1] == "stop"


async def test_stream_emits_the_reasoning_tail_after_the_finish_prelude():
    tail_reasoning = (
        'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1"}} \n\n'
        'data: {"choices": [{"delta": {"content": "Answer", "phase": "answer"}}], "response_id": "r1"}\n\n'
        'data: {"choices": [{"delta": {"content": "partial thin<", "phase": "think"}}], "response_id": "r1"}\n\n'
    )
    acct = FakeAccount()
    acct.client.completion = AsyncMock(return_value=FakeResp(tail_reasoning))
    lines = await _collect(qwen_api.stream_openai(**_args(acct)))
    joined = "".join(lines)
    assert _deltas(joined, "reasoning_content") == ["partial thin", "<"]
    assert _content(joined) == "Answer"
    assert _reasons(joined)[-1] == "stop"
    assert lines[-1] == "data: [DONE]\n\n"


async def test_send_completion_maps_both_http_error_shapes_to_502():
    status_error = httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock(status_code=500))
    for error in (status_error, httpx.ConnectError("boom")):
        client = MagicMock()
        client.completion = AsyncMock(side_effect=error)
        with pytest.raises(HTTPException) as excinfo:
            await qwen_api._send_completion(client, FakeSession(), "p", "m", False, False)
        assert excinfo.value.status_code == 502
        assert excinfo.value.detail.startswith("Qwen request failed: ")


async def test_send_completion_reads_the_body_for_a_non_200():
    resp = FakeResp("data: nope\n\n", status_code=429, content_type="text/event-stream")
    with pytest.raises(HTTPException) as excinfo:
        await qwen_api._send_completion(MagicMock(completion=AsyncMock(return_value=resp)), FakeSession(), "p", "m", False, False)
    assert excinfo.value.status_code == 429
    assert excinfo.value.detail == "data: nope\n\n"


async def test_stream_reports_a_transport_error_without_a_status_code():
    acct = FakeAccount()
    acct.client.completion = AsyncMock(side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock(status_code=500)))
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct))))
    assert "Qwen request failed" in joined
    assert _reasons(joined) == ["error"]
    assert joined.rstrip().endswith("data: [DONE]")


async def test_stream_reports_an_unmapped_waf_body():
    waf = FakeResp("<html>challenge</html>", content_type="text/html")
    acct = FakeAccount()
    acct.client.completion = AsyncMock(return_value=waf)
    joined = "".join(await _collect(qwen_api.stream_openai(**_args(acct))))
    assert "WAF challenge" in joined
    assert "requestInfo" not in joined
