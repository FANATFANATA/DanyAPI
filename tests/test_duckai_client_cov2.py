import asyncio
import base64
import json
import pathlib
import threading
from types import SimpleNamespace

import httpx
import pytest

import danyapi.duckai.client as client_mod
from danyapi.duckai import attest
from danyapi.duckai.client import (
    BASE_URL,
    CATALOG_BODY_LIMIT,
    CATALOG_ID_LIMIT,
    CATALOG_SCAN_LIMIT,
    MODEL_CATALOG,
    REASONING_EFFORTS,
    DuckAIClient,
    DuckAIError,
    _catalog_efforts,
    _catalog_field,
    _catalog_terminators,
    _error_for_payload,
    _first_mark,
    _iter_lines,
    _offsets,
    _reject_oversized_bundle,
    model_efforts,
    normalize_effort,
    parse_catalog,
    parse_event,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

STREAM_HEADERS = {"content-type": "text/event-stream"}

PAGE = '<html><script src="/dist/entry.duckai.9f8e7d.js"></script></html>'

ENTRIES = (
    '{model:"gpt-5.4-mini",modelName:"GPT-5.4 mini",modelShortName:"GPT-5.4 mini",createdBy:"OpenAI",modelType:"chat",'
    'supportedReasoningEffort:["none","low"],availableTo:["free.Free","paid.Pro"],aliases:{model:"nested"},costRank:1},'
    '{model:"claude-haiku-4-5",modelName:"Claude Haiku 4.5",modelVariant:"Fast",modelShortName:"Haiku",createdBy:"Anthropic",'
    'modelType:"chat",supportedReasoningEffort:[ ],availableTo:["free.Free"],costRank:2},'
    '{model:"gpt-5.6-luna",modelName:"Luna",modelShortName:"Luna",createdBy:"OpenAI",modelType:"chat",'
    'supportedReasoningEffort:["none"],availableTo:["paid.Pro"],costRank:0},'
    '{model:"gpt-5.6-terra",modelShortName:"Terra",createdBy:"OpenAI",modelType:"chat",'
    'supportedReasoningEffort:["none","medium"],availableTo:["free.Free"]}'
    '{model:"paid-only",modelShortName:"Paid",availableTo:["paid.Pro"],costRank:3}'
    f'{{model:"{"x" * 300}",modelShortName:"Long",availableTo:["free.Free"],costRank:4}}'
)

BUNDLE = f"window.__x={{catalog:[{ENTRIES}]}};var catalogDone=1"
BRACE_BUNDLE = f"window.__x={{catalog:[{ENTRIES}]}}"
VAR_BUNDLE = f"window.__x={{catalog:[{ENTRIES});var catalogDone=1"

EXPECTED_CATALOG = (
    {
        "id": "gpt-5.4-mini",
        "name": "GPT-5.4 mini",
        "owned_by": "duckai",
        "model_type": "chat",
        "provider": "openai",
        "efforts": ("none", "low"),
        "cost_rank": 1,
    },
    {
        "id": "claude-haiku-4-5",
        "name": "Claude Haiku 4.5 Fast",
        "owned_by": "duckai",
        "model_type": "chat",
        "provider": "anthropic",
        "efforts": REASONING_EFFORTS,
        "cost_rank": 2,
    },
    {
        "id": "gpt-5.6-terra",
        "name": "Terra",
        "owned_by": "duckai",
        "model_type": "chat",
        "provider": "openai",
        "efforts": ("none", "medium"),
        "cost_rank": 3,
    },
)


def _solve_stub(monkeypatch, header: str = "solved-jsa") -> list[str]:
    seen: list[str] = []

    async def _fake_header_for(script: str, user_agent: str, origin: str = BASE_URL) -> str:
        seen.append(script)
        return header

    monkeypatch.setattr(attest, "header_for", _fake_header_for)
    return seen


def _stream_body(lines: list[str]) -> bytes:
    return ("\n".join(f"data: {line}" for line in lines) + "\n\n").encode()


def _client(router) -> DuckAIClient:
    client = DuckAIClient()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(router), base_url=BASE_URL)
    return client


def _default_router(chat_response: httpx.Response | None = None):
    def router(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text='data-version-tag="v1" data-version-sha="sha1"')
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={"status": "0"})
        if chat_response is not None:
            return chat_response
        return httpx.Response(200, content=_stream_body(["[DONE]"]), headers=STREAM_HEADERS)

    return router


@pytest.fixture(autouse=True)
async def _drain_background_tasks():
    yield
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def test_empty_credential_is_gone_from_the_duckai_provider():
    assert not hasattr(client_mod, "EMPTY_CREDENTIAL")
    duckai_sources = sorted((REPO_ROOT / "danyapi" / "duckai").glob("*.py"))
    offenders = [path.name for path in duckai_sources if path.name != "attest.py" and "EMPTY_CREDENTIAL" in path.read_text(encoding="utf-8")]
    assert offenders == []


def test_scan_primitives_locate_offsets_and_terminators():
    text = 'a{model:"x"b{model:"y"c},{model:"z"d}]e; var f'
    assert _offsets(text, '{model:"') == [1, 12, 25]
    assert _offsets("nothing here", '{model:"') == []
    marks = _catalog_terminators(text)
    assert marks == sorted(marks)
    assert text[marks[0] : marks[0] + 11] == '},{model:"z'
    assert _first_mark(marks, 0, marks[0]) == marks[0]
    assert _first_mark(marks, 0, marks[0] - 1) is None
    assert _first_mark(marks, len(text) + 1, len(text) + 2) is None


def test_catalog_field_and_effort_helpers():
    body = 'modelName:"N",modelVariant:null,supportedReasoningEffort:["none" ," medium"],costRank:4'
    assert _catalog_field(body, "name") == "N"
    assert _catalog_field(body, "variant") is None
    assert _catalog_field(body, "cost_rank") == "4"
    assert _catalog_field(body, "model_type") is None
    assert _catalog_efforts(None) == REASONING_EFFORTS
    assert _catalog_efforts("") == REASONING_EFFORTS
    assert _catalog_efforts(" ") == REASONING_EFFORTS
    assert _catalog_efforts(' "none" , "medium" ') == ("none", "medium")


def test_parse_catalog_reads_the_free_entries_in_cost_order():
    assert parse_catalog(BUNDLE) == EXPECTED_CATALOG
    assert parse_catalog(BRACE_BUNDLE) == EXPECTED_CATALOG


def test_parse_catalog_matches_on_every_terminator_variant():
    assert BRACE_BUNDLE.endswith("}]}")
    assert VAR_BUNDLE.endswith(";var catalogDone=1")
    assert not VAR_BUNDLE.endswith("]")
    assert parse_catalog(BRACE_BUNDLE) == EXPECTED_CATALOG
    assert parse_catalog(VAR_BUNDLE) == EXPECTED_CATALOG


def test_parse_catalog_ignores_a_bundle_without_anchors():
    assert parse_catalog("nothing to see here") == ()
    assert parse_catalog("") == ()


def test_parse_catalog_ignores_an_unterminated_anchor():
    assert parse_catalog('{model:"') == ()


def test_parse_catalog_drops_an_id_beyond_the_id_limit():
    assert CATALOG_ID_LIMIT == 256
    assert "x" * (CATALOG_ID_LIMIT + 1) in BUNDLE
    assert all(entry["id"] != "x" * (CATALOG_ID_LIMIT + 1) for entry in parse_catalog(BUNDLE))


def test_parse_catalog_drops_a_body_beyond_the_body_limit():
    assert CATALOG_BODY_LIMIT == 4000
    filler = "f" * (CATALOG_BODY_LIMIT + 500)
    huge = f'{{model:"huge",filler:"{filler}",modelShortName:"Huge",availableTo:["free.Free"],costRank:1}},'
    assert parse_catalog(BUNDLE.replace("{catalog:[", "{catalog:[" + huge, 1)) == EXPECTED_CATALOG


def test_parse_catalog_stops_at_the_scan_limit(monkeypatch):
    assert CATALOG_SCAN_LIMIT == 8 * 1024 * 1024
    assert parse_catalog("x" * CATALOG_SCAN_LIMIT + BUNDLE) == ()
    monkeypatch.setattr(client_mod, "CATALOG_SCAN_LIMIT", 100)
    assert parse_catalog(BUNDLE) == ()


def test_reject_oversized_bundle_checks_the_declared_length(monkeypatch):
    monkeypatch.setattr(client_mod, "CATALOG_SCAN_LIMIT", 10)
    with pytest.raises(DuckAIError) as excinfo:
        _reject_oversized_bundle(SimpleNamespace(headers={"content-length": "11"}, content=b"tiny"))
    assert excinfo.value.code == "catalog"
    assert excinfo.value.message == "duck.ai entry bundle is above the 10 byte scan limit"
    _reject_oversized_bundle(SimpleNamespace(headers={"content-length": "not a number"}, content=b"tiny"))
    _reject_oversized_bundle(SimpleNamespace(headers={}, content=b"tiny"))


def test_reject_oversized_bundle_checks_the_body_length(monkeypatch):
    monkeypatch.setattr(client_mod, "CATALOG_SCAN_LIMIT", 10)
    with pytest.raises(DuckAIError) as excinfo:
        _reject_oversized_bundle(SimpleNamespace(headers={}, content=b"x" * 11))
    assert excinfo.value.message == "duck.ai entry bundle is above the 10 byte scan limit"
    _reject_oversized_bundle(SimpleNamespace(headers={"content-length": "4"}, content=b"x" * 10))


def test_model_efforts_falls_back_for_an_unknown_model():
    assert model_efforts("gpt-5.4-mini") == ("none", "low", "medium")
    assert model_efforts("nope") == REASONING_EFFORTS
    assert normalize_effort("nope", "high") == "none"
    assert normalize_effort("nope", "medium") == "medium"


def test_entrypoint_code_is_recognised():
    assert DuckAIError("ERR_BN_LIMIT", "x").is_entrypoint is True
    assert DuckAIError(500, "an Unsupported Entrypoint was served").is_entrypoint is True
    assert DuckAIError(500, "other").is_entrypoint is False


def test_source_entries_accept_a_singular_and_a_plural_shape():
    event = parse_event(
        {"action": "success", "role": "assistant", "message": "x", "sources": [{"url": "https://a", "title": "A"}, "junk", {"title": "no url"}]}
    )
    assert event.sources == [{"url": "https://a", "title": "A", "site": ""}]
    assert parse_event({"action": "success", "role": "assistant", "message": "x", "sources": "nope"}).sources == []


async def test_fe_version_retries_after_a_transport_failure():
    state = {"n": 0}

    def router(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/":
            return httpx.Response(404)
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.ConnectError("no route", request=request)
        return httpx.Response(200, text='data-version-tag="v1.2.3" data-version-sha="abc123"')

    client = _client(router)
    try:
        assert await client.fe_version() == "dev-hash"
        assert client._fe_version == ""
        assert await client.fe_version() == "v1.2.3-abc123"
        assert state["n"] == 2
        assert await client.fe_version() == "v1.2.3-abc123"
        assert state["n"] == 2
    finally:
        await client.aclose()


async def test_fe_version_retries_after_an_error_page():
    state = {"n": 0}

    def router(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/":
            return httpx.Response(404)
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(503, text="down")
        return httpx.Response(200, text="<html>no version here</html>")

    client = _client(router)
    try:
        assert await client.fe_version() == "dev-hash"
        assert client._fe_version == ""
        assert await client.fe_version() == "dev-hash"
        assert client._fe_version == "dev-hash"
        assert state["n"] == 2
    finally:
        await client.aclose()


async def test_cached_attestation_short_circuits_a_solve(monkeypatch):
    solves = _solve_stub(monkeypatch)
    client = _client(_default_router())
    try:
        client._jsa = "cached"
        assert await client.attestation() == "cached"
        assert solves == []
    finally:
        await client.aclose()


async def test_attestation_assigns_under_the_jsa_lock(monkeypatch):
    started = asyncio.Event()

    async def _fake_header_for(script: str, user_agent: str, origin: str = BASE_URL) -> str:
        started.set()
        return "fresh-jsa"

    monkeypatch.setattr(attest, "header_for", _fake_header_for)
    client = _client(_default_router())
    try:
        client._jsa = "old-jsa"
        await client._jsa_lock.acquire()
        task = asyncio.create_task(client._refresh_attestation("script"))
        await started.wait()
        assert client._jsa == "old-jsa"
        assert not task.done()
        client._jsa_lock.release()
        assert await task == "fresh-jsa"
        assert client._jsa == "fresh-jsa"
    finally:
        await client.aclose()


async def test_attestation_serialises_the_status_and_solve_pipeline(monkeypatch):
    entered: list[str] = []
    release = asyncio.Event()
    calls: list[int] = []

    async def _fake_status() -> dict:
        calls.append(1)
        entered.append("enter")
        await release.wait()
        entered.append("exit")
        return {"status": "0"}

    client = _client(_default_router())
    monkeypatch.setattr(client, "status", _fake_status)
    try:
        first = asyncio.create_task(client.attestation(force=True))
        while not entered:
            await asyncio.sleep(0)
        second = asyncio.create_task(client.attestation(force=True))
        await asyncio.sleep(0)
        assert entered == ["enter"]
        release.set()
        assert await first == attest.INITIAL_JSA
        assert await second == attest.INITIAL_JSA
        assert entered == ["enter", "exit", "enter", "exit"]
        assert len(calls) == 2
    finally:
        await client.aclose()


async def test_attestation_double_checks_the_cache_after_taking_the_refresh_lock(monkeypatch):
    client = _client(_default_router())
    status_calls: list[int] = []

    async def _fake_status() -> dict:
        status_calls.append(1)
        return {"status": "0"}

    monkeypatch.setattr(client, "status", _fake_status)
    try:
        client._jsa = attest.INITIAL_JSA
        held = asyncio.Event()
        release = asyncio.Event()

        async def holder() -> None:
            async with client._refresh_lock:
                held.set()
                await release.wait()

        guard = asyncio.create_task(holder())
        await held.wait()
        call = asyncio.create_task(client.attestation())
        await asyncio.sleep(0)
        assert not call.done()
        client._jsa = "solved-by-a-concurrent-warm"
        release.set()
        await guard
        assert await call == "solved-by-a-concurrent-warm"
        assert status_calls == []
    finally:
        await client.aclose()


async def test_attestation_warm_ignores_an_empty_or_repeated_script(monkeypatch):
    solves = _solve_stub(monkeypatch)
    client = _client(_default_router())
    try:
        client._start_attestation_warm("")
        assert client._jsa_warm is None
        client._start_attestation_warm("script-1")
        await client._join_attestation_warm()
        assert solves == ["script-1"]
        assert client._jsa_warm is None
        client._start_attestation_warm("script-1")
        assert client._jsa_warm is None
        assert solves == ["script-1"]
    finally:
        await client.aclose()


async def test_attestation_warm_keeps_one_solve_in_flight(monkeypatch):
    release = asyncio.Event()
    solves: list[str] = []

    async def _fake_header_for(script: str, user_agent: str, origin: str = BASE_URL) -> str:
        solves.append(script)
        await release.wait()
        return f"jsa-for-{script}"

    monkeypatch.setattr(attest, "header_for", _fake_header_for)
    client = _client(_default_router())
    try:
        client._start_attestation_warm("script-1")
        first = client._jsa_warm
        while not solves:
            await asyncio.sleep(0)
        client._start_attestation_warm("script-2")
        assert client._jsa_warm is first
        release.set()
        await first
        assert solves == ["script-1"]
        assert client._jsa == "jsa-for-script-1"
    finally:
        await client.aclose()


async def test_attestation_warm_swallows_a_solve_failure(monkeypatch):
    async def _boom(script: str, user_agent: str, origin: str = BASE_URL) -> str:
        raise attest.AttestationError("unsupported fragment")

    monkeypatch.setattr(attest, "header_for", _boom)
    client = _client(_default_router())
    try:
        client._start_attestation_warm("script-1")
        await client._join_attestation_warm()
        assert client._jsa == attest.INITIAL_JSA
        assert client._jsa_warm is None
    finally:
        await client.aclose()


async def test_chat_streams_the_first_event_before_the_solver_finishes(monkeypatch):
    release = asyncio.Event()
    order: list[str] = []

    async def _fake_header_for(script: str, user_agent: str, origin: str = BASE_URL) -> str:
        order.append("solve-start")
        await release.wait()
        order.append("solve-end")
        return "rotated-jsa"

    monkeypatch.setattr(attest, "header_for", _fake_header_for)
    body = _stream_body([json.dumps({"action": "success", "role": "assistant", "message": "he"}), "[DONE]"])
    client = _client(_default_router(httpx.Response(200, content=body, headers={**STREAM_HEADERS, attest.JSA_HEADER: "script-2"})))
    try:
        client._jsa = "old-jsa"
        gen = client.chat([{"role": "user", "content": []}])
        first = await asyncio.wait_for(gen.__anext__(), 2)
        assert first.delta == "he"
        assert "solve-end" not in order
        assert client._jsa == "old-jsa"
        while not order:
            await asyncio.sleep(0)
        assert order == ["solve-start"]
        assert client._jsa == "old-jsa"
        release.set()
        rest = [event async for event in gen]
        assert [event.finish for event in rest] == ["stop"]
        assert order == ["solve-start", "solve-end"]
        assert client._jsa == "rotated-jsa"
        assert client._jsa_warm is None
    finally:
        await client.aclose()


def test_tool_call_ids_are_unique_and_carry_the_tool_name():
    payload = {"action": "success", "role": "tool-invocation", "state": "call", "toolName": "WebSearch", "toolArguments": {}}
    first = parse_event(dict(payload)).tool_calls[0]["id"]
    second = parse_event(dict(payload)).tool_calls[0]["id"]
    assert first != second
    assert first.startswith("call_WebSearch_")
    assert len(first.rsplit("_", 1)[1]) == 16
    assert parse_event({**payload, "toolCallId": "c9"}).tool_calls[0]["id"] == "c9"


async def test_chat_dedupes_sources_and_keeps_their_order():
    def source(url: str) -> dict:
        return {"action": "success", "role": "source", "source": {"url": url, "title": url}}

    body = _stream_body(
        [
            json.dumps(source("https://a")),
            json.dumps(source("https://b")),
            json.dumps(source("https://a")),
            json.dumps(source("https://c")),
            "[DONE]",
        ]
    )
    client = _client(_default_router(httpx.Response(200, content=body, headers=STREAM_HEADERS)))
    try:
        collected = [entry for event in [e async for e in client.chat([{"role": "user", "content": []}])] for entry in event.sources]
        assert [entry["url"] for entry in collected] == ["https://a", "https://b", "https://c"]
    finally:
        await client.aclose()


async def test_check_auth_requires_a_status_code_in_the_ok_set(monkeypatch):
    _solve_stub(monkeypatch)
    bodies = {
        "ok": {"status": "0"},
        "empty-status": {"status": ""},
        "numeric-status": {"status": 0},
        "bad-status": {"status": "ERR_CHALLENGE"},
        "no-status": {"foo": "bar"},
    }

    def make(body) -> DuckAIClient:
        return _client(lambda request: httpx.Response(200, json=body) if request.url.path.endswith("/status") else httpx.Response(404))

    clients = {name: make(body) for name, body in bodies.items()}
    failing = _client(lambda request: httpx.Response(418, json={"type": "ERR_CHALLENGE"}))
    try:
        assert await clients["ok"].check_auth() is True
        assert await clients["empty-status"].check_auth() is True
        assert await clients["numeric-status"].check_auth() is True
        assert await clients["bad-status"].check_auth() is False
        assert await clients["no-status"].check_auth() is False
        assert await failing.check_auth() is False
    finally:
        for client in [*clients.values(), failing]:
            await client.aclose()


def test_error_for_payload_reads_type_message_and_challenge():
    assert _error_for_payload(500, None).message == "upstream returned 500"
    assert _error_for_payload(500, None).code == 500
    assert _error_for_payload(500, []).message == "upstream returned 500"
    typed = _error_for_payload(500, {"type": "ERR_UPSTREAM", "message": "down"})
    assert (typed.code, typed.message) == ("ERR_UPSTREAM", "down")
    assert _error_for_payload(500, {"type": "", "message": ""}).code == 500
    assert _error_for_payload(418, {"type": "ERR_CHALLENGE", "cd": {"gk": "abc"}}).message == "bot check failed (challenge abc)"
    assert _error_for_payload(418, {"type": "ERR_CHALLENGE", "message": "nope", "cd": {"gk": "abc"}}).message == "nope (challenge abc)"
    assert _error_for_payload(418, {"type": "ERR_CHALLENGE", "cd": {"gk": ""}}).message == "upstream returned 418"
    assert _error_for_payload(418, {"type": "ERR_CHALLENGE", "cd": "not a dict"}).message == "upstream returned 418"


async def test_fail_reads_the_body_before_building_the_error():
    client = _client(_default_router())
    try:
        typed = await client._fail(httpx.Response(418, json={"type": "ERR_CHALLENGE", "message": "refused"}))
        assert (typed.code, typed.message) == ("ERR_CHALLENGE", "refused")
        assert typed.is_challenge is True
        plain = await client._fail(httpx.Response(500, text="not json"))
        assert (plain.code, plain.message) == (500, "upstream returned 500")
    finally:
        await client.aclose()


async def test_fetch_models_publishes_the_catalog_atomically():
    bundle_ready = asyncio.Event()
    release = asyncio.Event()
    seen: list[str] = []

    class FakeHttp:
        async def get(self, path: str, headers=None) -> httpx.Response:
            seen.append(path)
            if path == "/":
                return httpx.Response(200, text=PAGE)
            bundle_ready.set()
            await release.wait()
            return httpx.Response(200, text=BUNDLE)

        async def aclose(self) -> None:
            return None

    client = _client(_default_router())
    client.http = FakeHttp()
    try:
        task = asyncio.create_task(client.fetch_models())
        await bundle_ready.wait()
        assert seen == ["/", "/dist/entry.duckai.9f8e7d.js"]
        assert client._known_catalog() == MODEL_CATALOG
        assert client.efforts_for("gpt-5.4-mini") == ("none", "low", "medium")
        release.set()
        models = await task
        assert [entry["id"] for entry in models] == [entry["id"] for entry in EXPECTED_CATALOG]
        assert client._known_catalog() == EXPECTED_CATALOG
        assert client.efforts_for("gpt-5.6-terra") == ("none", "medium")
        assert client.efforts_for("unknown-model") == REASONING_EFFORTS
    finally:
        await client.aclose()


async def test_concurrent_fetch_models_never_sees_a_partial_catalog():
    def router(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text=PAGE)
        return httpx.Response(200, text=BUNDLE)

    client = _client(router)
    try:
        first, second = await asyncio.gather(client.fetch_models(), client.fetch_models())
        assert first == second == [dict(entry) for entry in EXPECTED_CATALOG]
        assert client._known_catalog() == EXPECTED_CATALOG
    finally:
        await client.aclose()


async def test_fetch_models_keeps_the_known_catalog_when_the_page_has_no_bundle():
    client = _client(lambda request: httpx.Response(200, text="<html>no script</html>") if request.url.path == "/" else httpx.Response(200, text=BUNDLE))
    try:
        assert await client.fetch_models() == [dict(entry) for entry in MODEL_CATALOG]
        assert client._known_catalog() == MODEL_CATALOG
    finally:
        await client.aclose()


async def test_fetch_models_keeps_the_known_catalog_when_the_bundle_has_no_free_entries():
    client = _client(lambda request: httpx.Response(200, text=PAGE) if request.url.path == "/" else httpx.Response(200, text="nothing here"))
    try:
        assert await client.fetch_models() == [dict(entry) for entry in MODEL_CATALOG]
        assert client._known_catalog() == MODEL_CATALOG
    finally:
        await client.aclose()


async def test_fetch_models_keeps_the_known_catalog_for_an_oversized_bundle():
    oversized = "x" * (CATALOG_SCAN_LIMIT + 1)

    def router(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text=PAGE)
        return httpx.Response(200, content=oversized.encode(), headers={"content-type": "application/javascript"})

    client = _client(router)
    try:
        assert await client.fetch_models() == [dict(entry) for entry in MODEL_CATALOG]
        assert client._known_catalog() == MODEL_CATALOG
    finally:
        await client.aclose()


async def test_fetch_models_keeps_the_known_catalog_on_a_transport_failure():
    def router(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    client = _client(router)
    try:
        assert await client.fetch_models() == [dict(entry) for entry in MODEL_CATALOG]
    finally:
        await client.aclose()


async def test_chat_ignores_a_control_event_without_a_finish_and_a_broken_line():
    body = _stream_body(["[PING]", "[CHAT_TITLE:Title here]", "not json at all", json.dumps({"action": "success", "role": "assistant", "message": "hi"})])
    client = _client(_default_router(httpx.Response(200, content=body, headers=STREAM_HEADERS)))
    try:
        events = [event async for event in client.chat([{"role": "user", "content": []}])]
        assert [event.title for event in events if event.title] == ["Title here"]
        assert "".join(event.delta for event in events) == "hi"
    finally:
        await client.aclose()


async def test_iter_lines_skips_empty_chunks_and_caps_the_buffer():
    class _Resp:
        def __init__(self, chunks: list[str]) -> None:
            self._chunks = chunks

        async def aiter_text(self):
            for chunk in self._chunks:
                yield chunk

    assert [line async for line in _iter_lines(_Resp(["", "data: a\n", "", "\ndata: b"]))] == ["a", "b"]

    with pytest.raises(DuckAIError) as excinfo:
        _ = [line async for line in _iter_lines(_Resp(["x" * (client_mod.MAX_SSE_LINE_CHARS + 1)]))]
    assert excinfo.value.code == 502
    assert excinfo.value.message == f"duckai sent an SSE line above the {client_mod.MAX_SSE_LINE_CHARS} character limit without a newline"


def test_catalog_publish_uses_the_thread_lock():
    real = threading.Lock()
    order: list[str] = []

    class CountingLock:
        def __enter__(self):
            order.append("acquire")
            real.acquire()
            return self

        def __exit__(self, *exc):
            order.append("release")
            real.release()
            return False

    client = DuckAIClient()
    client._catalog_lock = CountingLock()
    assert client._known_catalog() == MODEL_CATALOG
    assert client.efforts_for("claude-opus-4-8") == ("none", "low", "medium")
    assert order == ["acquire", "release", "acquire", "release"]


def test_attestation_header_constant_is_stable():
    assert attest.JSA_HEADER == "X-Vqd-Hash-1"
    assert base64.b64decode("") == b""
