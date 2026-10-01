import asyncio
import json
import logging
import time
from collections.abc import Sequence

import pytest
from fastapi import HTTPException

from danyapi.alice import api as alice_api
from danyapi.alice.accounts import AliceAccount
from danyapi.alice.client import AliceClient, AliceError, AliceStream
from danyapi.api import retry as retry_module
from danyapi.api.schemas import ChatMessage
from danyapi.api.sse import STREAM_ERROR_FINISH


class _StubClient(AliceClient):
    def __init__(self, stream: AliceStream | None = None, errors: Sequence[BaseException] | None = None) -> None:
        super().__init__()
        self.stream = stream
        self.errors = list(errors or [])
        self.prompts: list[str] = []
        self.closed = 0
        self.timeline: list[str] = []
        self.delays: list[int] = []
        self.slept: list[float] = []

    async def ask(self, prompt: str) -> AliceStream:
        self.timeline.append("ask")
        self.prompts.append(prompt)
        if self.errors:
            raise self.errors.pop(0)
        if self.stream is None:
            raise AssertionError("the stub client was asked without an answer to give")
        return self.stream

    async def aclose(self) -> None:
        self.timeline.append("aclose")
        self.closed += 1


def _stream(content: str, version: str = "") -> AliceStream:
    stream = AliceStream()
    stream.content = content
    stream.done = True
    stream.version = version
    return stream


def _account(client: _StubClient) -> AliceAccount:
    return AliceAccount(index=0, client=client)


@pytest.fixture
def instant_backoff(monkeypatch):
    real_sleep = asyncio.sleep

    def _install(client: _StubClient) -> None:
        def _delay(attempt: int) -> float:
            client.delays.append(attempt)
            return 0.0

        async def _sleep(delay: float) -> None:
            client.slept.append(delay)
            await real_sleep(0)

        monkeypatch.setattr(alice_api, "_retry_delay", _delay)
        monkeypatch.setattr(asyncio, "sleep", _sleep)

    return _install


def _payloads(lines: list[str]) -> list[dict]:
    return [json.loads(line[len("data: ") :]) for line in lines if line.startswith("data: ") and line[len("data: ") :].strip() != "[DONE]"]


def _warnings(caplog, logger_name: str) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == logger_name]


def test_max_retries_is_reexported_from_the_shared_retry_helper() -> None:
    assert alice_api.MAX_RETRIES is retry_module.MAX_RETRIES
    assert alice_api.MAX_RETRIES == 5


def test_status_for_reports_upstream_trouble_as_bad_gateway() -> None:
    assert alice_api._status_for(AliceError(1006, "closed", retryable=True)) == 502
    assert alice_api._status_for(AliceError(1009, "timeout", retryable=True)) == 502
    assert alice_api._status_for(AliceError(1011, "alice error: boom")) == 502


def test_status_for_reports_a_non_upstream_error_as_bad_request() -> None:
    assert alice_api._status_for(AliceError(1008, "empty answer")) == 400
    assert alice_api._status_for(AliceError(1003, "handshake failed")) == 400


def test_usage_for_counts_prompt_and_completion_tokens() -> None:
    assert alice_api._usage_for("hello world", "hi there") == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


async def test_retryable_failure_reconnects_and_backs_off_before_retrying(instant_backoff) -> None:
    client = _StubClient(stream=_stream("391"), errors=[AliceError(1006, "closed", retryable=True)])
    instant_backoff(client)
    stream = await alice_api._ask(_account(client), "17*23?")
    assert stream.content == "391"
    assert client.prompts == ["17*23?", "17*23?"]
    assert client.timeline == ["ask", "aclose", "ask"]
    assert client.closed == 1
    assert client.delays == [1]
    assert client.slept == [0.0]


async def test_retry_budget_is_exhausted_after_max_retries_attempts(instant_backoff) -> None:
    errors = [AliceError(1006, f"closed {index}", retryable=True) for index in range(alice_api.MAX_RETRIES + 1)]
    client = _StubClient(errors=errors)
    instant_backoff(client)
    with pytest.raises(AliceError) as excinfo:
        await alice_api._ask(_account(client), "hi")
    assert excinfo.value.code == 1006
    assert excinfo.value.message == f"closed {alice_api.MAX_RETRIES}"
    assert len(client.prompts) == alice_api.MAX_RETRIES + 1
    assert client.closed == alice_api.MAX_RETRIES
    assert client.delays == list(range(1, alice_api.MAX_RETRIES + 1))
    assert client.slept == [0.0] * alice_api.MAX_RETRIES


async def test_non_retryable_failure_is_raised_without_a_reconnect() -> None:
    client = _StubClient(errors=[AliceError(1008, "empty answer", retryable=False)])
    with pytest.raises(AliceError) as excinfo:
        await alice_api._ask(_account(client), "hi")
    assert excinfo.value.code == 1008
    assert client.prompts == ["hi"]
    assert client.closed == 0
    assert client.timeline == ["ask"]


@pytest.mark.parametrize("code", [1002, 1011])
async def test_auth_rejected_and_connect_fatal_mark_the_account_broken(code, caplog) -> None:
    account = _account(_StubClient(errors=[AliceError(code, "boom", retryable=False)]))
    with caplog.at_level(logging.WARNING, logger="danyapi.alice"):
        with pytest.raises(AliceError):
            await alice_api._ask(account, "hi")
    assert account.broken is True
    assert account.broken_at is not None
    assert _warnings(caplog, "danyapi.alice.api") == [f"alice account #0 marked broken by upstream error {code}: boom"]
    assert _warnings(caplog, "danyapi.alice") == ["alice account #0 marked broken"]


@pytest.mark.parametrize("code", [1002, 1011])
async def test_broken_marking_happens_on_every_attempt(code, instant_backoff, caplog) -> None:
    client = _StubClient(
        stream=_stream("391"),
        errors=[AliceError(code, "boom", retryable=True) for _ in range(2)],
    )
    instant_backoff(client)
    account = _account(client)
    with caplog.at_level(logging.WARNING, logger="danyapi.alice.api"):
        stream = await alice_api._ask(account, "hi")
    assert stream.content == "391"
    assert account.broken is True
    assert client.delays == [1, 2]
    assert _warnings(caplog, "danyapi.alice.api") == [f"alice account #0 marked broken by upstream error {code}: boom"] * 2


async def test_unattributable_failure_does_not_mark_the_account_broken(caplog) -> None:
    account = _account(_StubClient(errors=[AliceError(1006, "closed", retryable=False)]))
    with caplog.at_level(logging.WARNING, logger="danyapi.alice"):
        with pytest.raises(AliceError):
            await alice_api._ask(account, "hi")
    assert account.broken is False
    assert account.broken_at is None
    assert _warnings(caplog, "danyapi.alice.api") == []


async def test_collect_non_stream_returns_the_openai_shape_and_records_usage(monkeypatch) -> None:
    recorded: list[tuple] = []
    monkeypatch.setattr(alice_api, "record_usage_dict", lambda *args, **kwargs: recorded.append((args, kwargs)))
    account = _account(_StubClient(stream=_stream("391", version="v1")))
    before = int(time.time())
    result = await alice_api.collect_non_stream(
        account,
        messages=[ChatMessage(role="user", content="17*23?")],
        model="alice",
        user="u1",
        session_id="s1",
    )
    after = int(time.time())
    assert result["object"] == "chat.completion"
    assert result["model"] == "alice"
    assert result["system_fingerprint"] == "v1"
    assert result["session_id"] == "s1"
    assert result["id"].startswith("chatcmpl-")
    assert len(result["id"]) == len("chatcmpl-") + 32
    assert before <= result["created"] <= after
    assert result["choices"] == [{"index": 0, "message": {"role": "assistant", "content": "391"}, "finish_reason": "stop", "logprobs": None}]
    assert result["usage"]["total_tokens"] > 0
    assert recorded == [(("alice", "alice", result["usage"]), {"user": "u1", "session_id": "s1"})]
    assert account.sem.locked() is False


async def test_collect_non_stream_falls_back_to_the_fingerprint_sentinel(monkeypatch) -> None:
    monkeypatch.setattr(alice_api, "record_usage_dict", lambda *args, **kwargs: None)
    account = _account(_StubClient(stream=_stream("391")))
    result = await alice_api.collect_non_stream(account, messages=[ChatMessage(role="user", content="hi")])
    assert result["system_fingerprint"] == "fp_danyapi"


async def test_collect_non_stream_applies_stop_inside_the_lock(monkeypatch) -> None:
    monkeypatch.setattr(alice_api, "record_usage_dict", lambda *args, **kwargs: None)
    client = _StubClient(stream=_stream("keep STOP drop"))
    account = _account(client)
    result = await alice_api.collect_non_stream(
        account,
        messages=[ChatMessage(role="user", content="hi")],
        stop=["STOP"],
    )
    assert result["choices"][0]["message"]["content"] == "keep "
    assert client.prompts == ["User: hi"]


async def test_collect_non_stream_prefers_the_raw_prompt_over_folded_messages(monkeypatch) -> None:
    monkeypatch.setattr(alice_api, "record_usage_dict", lambda *args, **kwargs: None)
    client = _StubClient(stream=_stream("391"))
    await alice_api.collect_non_stream(_account(client), messages=[ChatMessage(role="user", content="folded")], prompt="raw prompt")
    assert client.prompts == ["raw prompt"]


async def test_collect_non_stream_maps_a_retryable_error_to_bad_gateway(monkeypatch, instant_backoff) -> None:
    monkeypatch.setattr(alice_api, "record_usage_dict", lambda *args, **kwargs: None)
    client = _StubClient(errors=[AliceError(1006, "closed", retryable=True) for _ in range(alice_api.MAX_RETRIES + 1)])
    instant_backoff(client)
    account = _account(client)
    with pytest.raises(HTTPException) as excinfo:
        await alice_api.collect_non_stream(account, messages=[ChatMessage(role="user", content="hi")])
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "Alice error: closed"
    assert len(client.prompts) == alice_api.MAX_RETRIES + 1
    assert account.sem.locked() is False


async def test_collect_non_stream_maps_a_fatal_error_to_bad_gateway(monkeypatch) -> None:
    monkeypatch.setattr(alice_api, "record_usage_dict", lambda *args, **kwargs: None)
    account = _account(_StubClient(errors=[AliceError(1011, "alice error: boom")]))
    with pytest.raises(HTTPException) as excinfo:
        await alice_api.collect_non_stream(account, messages=[ChatMessage(role="user", content="hi")])
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "Alice error: alice error: boom"


async def test_collect_non_stream_maps_a_protocol_error_to_bad_request(monkeypatch) -> None:
    monkeypatch.setattr(alice_api, "record_usage_dict", lambda *args, **kwargs: None)
    account = _account(_StubClient(errors=[AliceError(1008, "alice declined to answer: nope")]))
    with pytest.raises(HTTPException) as excinfo:
        await alice_api.collect_non_stream(account, messages=[ChatMessage(role="user", content="hi")])
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "Alice error: alice declined to answer: nope"


async def test_stream_releases_the_account_lock_before_yielding_any_chunk(monkeypatch) -> None:
    recorded: list[tuple] = []
    monkeypatch.setattr(alice_api, "record_usage_dict", lambda *args, **kwargs: recorded.append((args, kwargs)))
    account = _account(_StubClient(stream=_stream("hello", version="v2")))
    stream = alice_api.stream_openai(
        account,
        messages=[ChatMessage(role="user", content="hi")],
        model="alice",
        include_usage=True,
    )
    seen: list[str] = []
    while True:
        assert account.sem.locked() is False
        try:
            seen.append(await stream.__anext__())
        except StopAsyncIteration:
            break
    assert seen[-1] == alice_api.DONE_LINE
    payloads = _payloads(seen)
    assert len(payloads) == 3
    assert payloads[0]["choices"][0] == {"index": 0, "delta": {"role": "assistant", "content": "hello"}, "finish_reason": None}
    assert payloads[1]["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "stop"}
    assert payloads[2]["choices"][0] == {"index": 0, "delta": {}, "finish_reason": None}
    assert payloads[2]["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    assert all(payload["model"] == "alice" for payload in payloads)
    assert len({payload["id"] for payload in payloads}) == 1
    assert recorded == [(("alice", "alice", payloads[2]["usage"]), {"user": None, "session_id": None})]
    assert account.sem.locked() is False


async def test_stream_delivers_no_incremental_text_above_the_first_chunk(monkeypatch) -> None:
    monkeypatch.setattr(alice_api, "record_usage_dict", lambda *args, **kwargs: None)
    account = _account(_StubClient(stream=_stream("a long single answer")))
    lines = [line async for line in alice_api.stream_openai(account, messages=[ChatMessage(role="user", content="hi")])]
    deltas = [payload["choices"][0]["delta"] for payload in _payloads(lines)]
    assert deltas == [{"role": "assistant", "content": "a long single answer"}, {}]
    assert account.sem.locked() is False


async def test_stream_omits_the_usage_chunk_by_default(monkeypatch) -> None:
    recorded: list[tuple] = []
    monkeypatch.setattr(alice_api, "record_usage_dict", lambda *args, **kwargs: recorded.append((args, kwargs)))
    account = _account(_StubClient(stream=_stream("391")))
    lines = [line async for line in alice_api.stream_openai(account, messages=[ChatMessage(role="user", content="hi")])]
    payloads = _payloads(lines)
    assert len(payloads) == 2
    assert all("usage" not in payload for payload in payloads)
    assert lines[-1] == "data: [DONE]\n\n"
    assert len(recorded) == 1
    assert recorded[0][0][:2] == ("alice", "alice")


async def test_stream_reports_the_error_in_band_and_frees_the_lock(monkeypatch, instant_backoff) -> None:
    recorded: list[tuple] = []
    monkeypatch.setattr(alice_api, "record_usage_dict", lambda *args, **kwargs: recorded.append((args, kwargs)))
    client = _StubClient(errors=[AliceError(1006, "closed", retryable=True) for _ in range(alice_api.MAX_RETRIES + 1)])
    instant_backoff(client)
    account = _account(client)
    stream = alice_api.stream_openai(account, messages=[ChatMessage(role="user", content="hi")], session_id="sess")
    lines = []
    while True:
        assert account.sem.locked() is False
        try:
            lines.append(await stream.__anext__())
        except StopAsyncIteration:
            break
    assert len(lines) == 2
    assert lines[1] == "data: [DONE]\n\n"
    error_payload = json.loads(lines[0][len("data: ") :])
    assert error_payload["error"] == {"message": "Alice error: closed"}
    assert error_payload["session_id"] == "sess"
    assert error_payload["choices"] == [{"index": 0, "delta": {}, "finish_reason": STREAM_ERROR_FINISH}]
    assert recorded == []
    assert len(client.prompts) == alice_api.MAX_RETRIES + 1
    assert account.sem.locked() is False


async def test_stream_applies_stop_and_records_usage(monkeypatch) -> None:
    recorded: list[tuple] = []
    monkeypatch.setattr(alice_api, "record_usage_dict", lambda *args, **kwargs: recorded.append((args, kwargs)))
    account = _account(_StubClient(stream=_stream("before STOP after")))
    lines = [
        line
        async for line in alice_api.stream_openai(
            account,
            messages=[ChatMessage(role="user", content="hi")],
            prompt="hi",
            stop=["STOP"],
            user="u2",
            session_id="s3",
        )
    ]
    assert _payloads(lines)[0]["choices"][0]["delta"] == {"role": "assistant", "content": "before "}
    assert recorded[0][0] == (
        "alice",
        "alice",
        {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    )
    assert recorded[0][1] == {"user": "u2", "session_id": "s3"}
    assert account.sem.locked() is False
