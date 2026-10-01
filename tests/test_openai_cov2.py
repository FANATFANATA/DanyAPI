import asyncio
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import danyapi.api.core as core_mod
import danyapi.api.openai as openai_mod
from danyapi import store as store_mod
from danyapi.api import responses as responses_api
from danyapi.api import state as state_mod
from danyapi.api.openai import ResponsesRequest, app, settings
from danyapi.store import JsonStore
from danyapi.usage import UsageTracker

POOL_ATTRS = ("pool", "qwen_pool", "gigachat_pool", "alice_pool", "duckai_pool")


class _FakeRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}
        self.client = None
        self.url = type("_Url", (), {"path": "/v1/responses"})()


def _dispatcher(handler):
    async def dispatcher(model, request):
        return handler

    return dispatcher


async def _agen(*chunks):
    for chunk in chunks:
        yield chunk


class _ChatResponse:
    def __init__(self, chunks):
        if hasattr(chunks, "__aiter__"):
            self.body_iterator = chunks
        else:
            self.body_iterator = _agen(*chunks)


async def _drain(stream):
    return [line async for line in stream]


def _chat_chunk(content="hi", finish="stop"):
    payload = {
        "id": "c1",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }
    return f"data: {json.dumps(payload)}\n\n"


class _FakePool:
    def __init__(self, accounts=(), healthy=0):
        self.accounts = list(accounts)
        self.healthy = healthy

    def stats(self):
        return {"accounts": len(self.accounts), "healthy": self.healthy, "broken": 0}


class _BrokenPool(_FakePool):
    def stats(self):
        raise RuntimeError("stats exploded")


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    saved = dict(app.state._state)
    saved_root = SimpleNamespace(html=openai_mod._root_ctx.html, checked=openai_mod._root_ctx.checked, stamp=openai_mod._root_ctx.stamp)
    openai_mod._INFLIGHT_RESPONSES.clear()
    app.state.responses_store = None
    for attr in POOL_ATTRS:
        setattr(app.state, attr, None)
    app.state.byok = False
    app.state.byok_pools = {provider: {} for provider in state_mod.BYOK_PROVIDERS}
    app.state.usage = None
    monkeypatch.setattr(core_mod, "_POOL_RATE_CACHE", {})
    yield
    openai_mod._INFLIGHT_RESPONSES.clear()
    openai_mod._root_ctx.html = saved_root.html
    openai_mod._root_ctx.checked = saved_root.checked
    openai_mod._root_ctx.stamp = saved_root.stamp
    app.state._state.clear()
    app.state._state.update(saved)


def _reset_root_ctx():
    openai_mod._root_ctx.html = None
    openai_mod._root_ctx.checked = False
    openai_mod._root_ctx.stamp = None


class _FakeLeaf:
    def __init__(self, events):
        self.events = events
        self.text = "<html>dashboard</html>"
        self.mtime = 1000
        self.size = 21
        self.reads = 0
        self.stat_error = None
        self.read_error = None

    def stat(self):
        if self.stat_error is not None:
            raise self.stat_error
        return SimpleNamespace(st_mtime_ns=self.mtime, st_size=self.size)

    def read_text(self, encoding="utf-8"):
        self.reads += 1
        self.events.append("read")
        if self.read_error is not None:
            raise self.read_error
        return self.text


class _FakeWebPath:
    def __init__(self, leaf):
        self.leaf = leaf

    def __truediv__(self, other):
        return self

    def resolve(self):
        return self

    @property
    def parents(self):
        return (self, self, self)

    def stat(self):
        return self.leaf.stat()

    def read_text(self, encoding="utf-8"):
        return self.leaf.read_text(encoding)


class _TrackingLock:
    def __init__(self, events):
        self.events = events
        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        self.events.append("acquire")
        return self

    def __exit__(self, *exc_info):
        self.events.append("release")
        self._lock.release()
        return False


def _install_web(monkeypatch, leaf, events):
    _reset_root_ctx()
    monkeypatch.setattr(openai_mod, "Path", lambda value: _FakeWebPath(leaf))
    monkeypatch.setattr(openai_mod, "_root_lock", _TrackingLock(events))


def test_concurrent_first_requests_read_the_dashboard_once(monkeypatch):
    events: list[str] = []
    leaf = _FakeLeaf(events)
    _install_web(monkeypatch, leaf, events)
    parties = 8
    start = threading.Barrier(parties)

    def worker():
        start.wait()
        openai_mod._load_root_html()

    with ThreadPoolExecutor(max_workers=parties) as pool:
        list(pool.map(lambda _: worker(), range(parties)))
    assert leaf.reads == 1
    assert openai_mod._root_ctx.html == "<html>dashboard</html>"
    assert openai_mod._root_ctx.stamp == (1000, 21)
    assert events.count("acquire") == parties
    assert events.count("release") == parties
    assert events[0] == "acquire"
    assert events[1] == "read"
    assert events[2] == "release"
    for index, event in enumerate(events):
        if event == "read":
            assert events[index - 1] == "acquire"
            assert events[index + 1] == "release"


def test_an_unchanged_dashboard_is_not_reread(monkeypatch):
    events: list[str] = []
    leaf = _FakeLeaf(events)
    _install_web(monkeypatch, leaf, events)
    openai_mod._load_root_html()
    openai_mod._load_root_html()
    openai_mod._load_root_html()
    assert leaf.reads == 1
    assert events.count("read") == 1


@pytest.mark.parametrize("change", ["mtime", "size", "both"])
def test_a_changed_mtime_or_size_invalidates_the_cached_dashboard(monkeypatch, change):
    events: list[str] = []
    leaf = _FakeLeaf(events)
    _install_web(monkeypatch, leaf, events)
    openai_mod._load_root_html()
    expected = [1000, 21]
    if change in ("mtime", "both"):
        leaf.mtime = 2000
        expected[0] = 2000
    if change in ("size", "both"):
        leaf.size = 99
        expected[1] = 99
    openai_mod._load_root_html()
    assert leaf.reads == 2
    assert openai_mod._root_ctx.stamp == tuple(expected)
    openai_mod._load_root_html()
    assert leaf.reads == 2


def test_a_vanished_dashboard_invalidates_the_cache_and_warns_once(monkeypatch, caplog):
    events: list[str] = []
    leaf = _FakeLeaf(events)
    _install_web(monkeypatch, leaf, events)
    openai_mod._load_root_html()
    leaf.stat_error = FileNotFoundError("web/index.html is gone")
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        openai_mod._load_root_html()
        openai_mod._load_root_html()
    assert leaf.reads == 1
    assert openai_mod._root_ctx.html is None
    assert openai_mod._root_ctx.stamp is None
    assert openai_mod._root_ctx.checked is True
    assert [record.getMessage() for record in caplog.records] == ["web interface is not readable: web/index.html is gone"]


def test_a_missing_dashboard_warns_on_the_first_probe(monkeypatch, caplog):
    events: list[str] = []
    leaf = _FakeLeaf(events)
    leaf.stat_error = OSError("no web directory")
    _install_web(monkeypatch, leaf, events)
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        openai_mod._load_root_html()
    assert "web interface is not readable: no web directory" in caplog.records[0].getMessage()
    assert leaf.reads == 0


@pytest.mark.parametrize("error", [OSError("read failed"), UnicodeError("bad utf-8")])
def test_a_read_failure_is_logged_and_does_not_raise(monkeypatch, caplog, error):
    events: list[str] = []
    leaf = _FakeLeaf(events)
    leaf.read_error = error
    _install_web(monkeypatch, leaf, events)
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        openai_mod._load_root_html()
    assert "web interface read failed" in caplog.records[0].getMessage()
    assert openai_mod._root_ctx.html is None
    assert openai_mod._root_ctx.stamp is None
    assert openai_mod._root_ctx.checked is True


def test_a_read_failure_invalidates_the_previous_html(monkeypatch):
    events: list[str] = []
    leaf = _FakeLeaf(events)
    _install_web(monkeypatch, leaf, events)
    openai_mod._load_root_html()
    assert openai_mod._root_ctx.html == "<html>dashboard</html>"
    leaf.mtime = 3000
    leaf.read_error = OSError("read failed")
    openai_mod._load_root_html()
    assert openai_mod._root_ctx.html is None
    assert openai_mod._root_ctx.stamp is None
    assert openai_mod._root_ctx.checked is True


async def test_root_serves_the_cached_dashboard(monkeypatch):
    events: list[str] = []
    leaf = _FakeLeaf(events)
    _install_web(monkeypatch, leaf, events)
    assert await openai_mod.root() == "<html>dashboard</html>"
    assert (await openai_mod.root()) == "<html>dashboard</html>"
    assert leaf.reads == 1


async def test_root_falls_back_to_a_404_page(monkeypatch):
    events: list[str] = []
    leaf = _FakeLeaf(events)
    leaf.stat_error = OSError("no web directory")
    _install_web(monkeypatch, leaf, events)
    response = await openai_mod.root()
    assert response.status_code == 404
    assert response.body == b"<h1>DanyAPI</h1><p>Web interface not found</p>"


async def test_favicon_is_an_empty_204():
    response = await openai_mod.favicon()
    assert response.status_code == 204
    assert response.body == b""


def test_pool_stats_logs_and_returns_none_for_a_broken_pool(caplog):
    assert openai_mod._pool_stats(None) is None
    assert openai_mod._pool_stats(_FakePool()) == {"accounts": 0, "healthy": 0, "broken": 0}
    with caplog.at_level(logging.DEBUG, logger="danyapi.api"):
        assert openai_mod._pool_stats(_BrokenPool()) is None
    assert "pool stats unavailable: stats exploded" in caplog.records[0].getMessage()


def test_byok_pools_for_merges_the_keyless_singleton():
    keyed = _FakePool()
    singleton = _FakePool()
    app.state.byok_alice_pool = singleton
    assert openai_mod._byok_pools_for("deepseek", {"deepseek": {"k1": keyed}}) == [keyed]
    assert openai_mod._byok_pools_for("alice", {"alice": {"k1": singleton}}) == [singleton]
    assert openai_mod._byok_pools_for("alice", {"alice": {}}) == [singleton]
    assert openai_mod._byok_pools_for("alice", {"alice": {"k1": None}}) == [singleton]
    assert openai_mod._byok_pools_for("duckai", {"duckai": {}}) == []


def test_byok_provider_stats_sums_every_pool():
    app.state.alice_models = [{"id": "yagpt"}]
    app.state.byok_alice_pool = _FakePool(healthy=2)
    stats = openai_mod._byok_provider_stats("alice", {"alice": {"k1": _FakePool(healthy=1), "k2": _BrokenPool()}})
    assert stats == {"pools": 3, "accounts": 0, "healthy": 3, "broken": 0, "models": 1}
    assert openai_mod._byok_provider_stats("duckai", {})["models"] == 0


def test_usage_summary_warns_and_returns_none_for_a_broken_tracker(caplog):
    assert openai_mod._usage_summary() is None

    class _Broken:
        def snapshot(self):
            raise RuntimeError("tracker corrupt")

    app.state.usage = _Broken()
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        assert openai_mod._usage_summary() is None
    assert "usage snapshot unavailable: tracker corrupt" in caplog.records[0].getMessage()
    app.state.usage = UsageTracker()
    app.state.usage.record("deepseek", "deepseek-v4.1-flash", 3, 1, 4)
    assert openai_mod._usage_summary() == {"requests": 1, "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}


def test_health_is_admin_gated():
    client = TestClient(app)
    anonymous = client.get("/health")
    assert anonymous.status_code == 200
    assert anonymous.json() == {"status": "ok"}
    wrong = client.get("/health", headers={"x-api-key": "not-the-admin-token"})
    assert wrong.json() == {"status": "ok"}
    with_token = client.get("/health", headers={"x-api-key": settings.admin_token})
    assert with_token.json()["status"] == "ok"
    assert with_token.json()["usage"] is None
    assert set(with_token.json()) == {
        "status",
        "usage",
        "deepseek",
        "qwen",
        "gigachat",
        "opencode",
        "alice",
        "duckai",
        "deepseek_stats",
        "qwen_stats",
        "gigachat_stats",
        "opencode_stats",
        "alice_stats",
        "duckai_stats",
    }
    assert with_token.json()["deepseek"] is False
    assert with_token.json()["deepseek_stats"] is None
    client.close()


def test_health_reports_the_byok_detail_to_an_admin():
    app.state.byok = True
    app.state.byok_pools = {provider: {} for provider in state_mod.BYOK_PROVIDERS}
    app.state.byok_pools["deepseek"] = {"k1": _FakePool(), "k2": _FakePool()}
    app.state.byok_alice_pool = _FakePool(healthy=1)
    client = TestClient(app)
    detail = client.get("/health", headers={"x-api-key": settings.admin_token}).json()
    assert detail["byok"] is True
    assert detail["byok_pools"] == {"deepseek": 2, "qwen": 0, "gigachat": 0, "opencode": 0, "alice": 0, "duckai": 0}
    assert detail["byok_api_key_required"] == {
        "deepseek": True,
        "qwen": True,
        "gigachat": True,
        "opencode": True,
        "alice": False,
        "duckai": False,
    }
    assert detail["deepseek"] is True
    assert detail["alice_stats"] == {"pools": 1, "accounts": 0, "healthy": 1, "broken": 0, "models": 0}
    assert detail["duckai_stats"]["pools"] == 0
    client.close()


def test_usage_endpoint_hides_user_identifiers_from_anonymous_callers():
    tracker = UsageTracker()
    tracker.record("deepseek", "deepseek-v4.1-flash", 3, 1, 4, user="alice-secret", session_id="sess-live")
    app.state.usage = tracker
    client = TestClient(app)
    anonymous = client.get("/v1/usage")
    assert anonymous.status_code == 200
    body = anonymous.json()
    assert set(body) == {"totals", "by_model"}
    assert body["totals"] == {"requests": 1, "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    assert "alice-secret" not in anonymous.text
    assert "sess-live" not in anonymous.text
    full = client.get("/v1/usage", headers={"x-api-key": settings.admin_token})
    assert set(full.json()) == {"totals", "by_model", "by_provider", "by_user", "recent"}
    assert full.json()["by_user"]["alice-secret"]["requests"] == 1
    client.close()


def test_usage_endpoint_is_404_when_tracking_is_disabled():
    app.state.usage = None
    client = TestClient(app)
    response = client.get("/v1/usage", headers={"x-api-key": settings.admin_token})
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "usage tracking is disabled"
    client.close()


def test_responses_chat_request_carries_the_merged_conversation():
    req = ResponsesRequest(
        model="deepseek-v4.1-flash",
        instructions="be brief",
        stream=True,
        temperature=0.5,
        top_p=0.9,
        thinking=True,
        search=True,
        user="alice",
        max_output_tokens=64,
        tools=[{"type": "function", "name": "f", "input_schema": {"type": "object"}}],
        tool_choice="auto",
        text={"format": {"type": "json_object"}},
    )
    chat = openai_mod._responses_chat_request(req, [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], "sess-1")
    assert chat.model == "deepseek-v4.1-flash"
    assert [(message.role, message.content) for message in chat.messages] == [("system", "be brief"), ("user", "hi")]
    assert chat.stream is True
    assert chat.stream_options == {"include_usage": True}
    assert chat.temperature == 0.5
    assert chat.top_p == 0.9
    assert chat.thinking is True
    assert chat.search is True
    assert chat.user == "alice"
    assert chat.max_tokens == 64
    assert chat.session_id == "sess-1"
    assert chat.tools == responses_api.convert_tools(req.tools)
    assert chat.tools == [{"type": "function", "function": {"name": "f"}}]
    assert chat.tool_choice == "auto"
    assert chat.response_format == responses_api.extract_response_format(req.text, req.response_format)
    assert openai_mod._responses_chat_request(ResponsesRequest(model="m"), [{"role": "user", "content": "hi"}], None).stream_options is None


async def test_cancellable_stream_delegates_and_stops_on_cancel():
    source = _agen("a", "b")
    entry = {"cancel": False}
    stream = openai_mod._CancellableStream(source, entry)
    assert stream.__aiter__() is stream
    assert await stream.__anext__() == "a"
    entry["cancel"] = True
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()
    entry["cancel"] = False
    assert await stream.__anext__() == "b"
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


async def test_cancellable_stream_close_is_best_effort(caplog):
    class _NoClose:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class _BadClose:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            raise RuntimeError("close exploded")

    closed = []

    class _GoodClose:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            closed.append(True)

    await openai_mod._CancellableStream(_NoClose(), {"cancel": False}).aclose()
    await openai_mod._CancellableStream(_GoodClose(), {"cancel": False}).aclose()
    assert closed == [True]
    with caplog.at_level(logging.DEBUG, logger="danyapi.api"):
        await openai_mod._CancellableStream(_BadClose(), {"cancel": False}).aclose()
    assert "cancellable stream close failed: close exploded" in caplog.records[0].getMessage()


async def test_store_set_writes_from_a_worker_thread():
    calls = []
    loop_thread = threading.current_thread()

    class _Store:
        def set(self, key, value):
            calls.append((key, value, threading.current_thread()))

    await openai_mod._store_set(_Store(), "resp_1", {"public": {"status": "completed"}})
    assert len(calls) == 1
    key, value, thread = calls[0]
    assert key == "resp_1"
    assert value == {"public": {"status": "completed"}}
    assert thread is not loop_thread


def test_cancelled_public_marks_the_response_cancelled():
    assert openai_mod._cancelled_public({"id": "resp_1", "status": "completed"}) == {
        "id": "resp_1",
        "status": "cancelled",
        "incomplete_details": {"reason": "cancelled"},
    }


@pytest.fixture
def responses_store(monkeypatch):
    store = JsonStore("openai-cov2-responses", None)
    app.state.responses_store = store
    return store


def _record(store, response_id, **public):
    store.set(response_id, {"public": {"id": response_id, "status": "completed", **public}, "conversation": [{"role": "user", "content": "hi"}]})


async def test_create_response_validates_the_input_before_dispatch(monkeypatch):
    app.state.responses_store = JsonStore("openai-cov2-a", None)
    dispatched = []

    async def dispatcher(model, request):
        dispatched.append(model)
        return None

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", dispatcher)
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod.create_response(ResponsesRequest(model="gpt-4", input="   "), _FakeRequest())
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "input must contain at least one input item"
    assert dispatched == []


async def test_create_response_rejects_a_malformed_input(monkeypatch):
    app.state.responses_store = JsonStore("openai-cov2-b", None)
    monkeypatch.setattr(openai_mod, "_chat_dispatcher", _dispatcher(None))
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod.create_response(ResponsesRequest(model="deepseek-v4.1-flash", input=123), _FakeRequest())
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "input must be a string or an array of input items"


async def test_create_response_requires_a_known_previous_response(monkeypatch, responses_store):
    dispatched = []

    async def dispatcher(model, request):
        dispatched.append(model)
        return None

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", dispatcher)
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod.create_response(ResponsesRequest(model="deepseek-v4.1-flash", input="hi", previous_response_id="resp_missing"), _FakeRequest())
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "response resp_missing not found"
    assert dispatched == []


async def test_create_response_validates_the_tool_chain_over_the_merged_conversation(monkeypatch, responses_store):
    responses_store.set(
        "resp_prev", {"public": {"id": "resp_prev", "status": "completed"}, "conversation": [{"role": "tool", "tool_call_id": "call_ghost", "content": "42"}]}
    )
    dispatched = []

    async def dispatcher(model, request):
        dispatched.append(model)
        return None

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", dispatcher)
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod.create_response(ResponsesRequest(model="deepseek-v4.1-flash", input="hi", previous_response_id="resp_prev"), _FakeRequest())
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "messages[0] is a tool result for an unknown call_id: 'call_ghost'"
    assert dispatched == []


async def test_create_response_tolerates_a_previous_record_without_a_conversation(monkeypatch, responses_store):
    responses_store.set("resp_prev", {"public": {"id": "resp_prev", "status": "completed"}})

    async def provider_call(chat_req):
        return {
            "id": "c1",
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", _dispatcher(provider_call))
    body = await openai_mod.create_response(
        ResponsesRequest(model="deepseek-v4.1-flash", input="hi", previous_response_id="resp_prev", session_id="sess-9"), _FakeRequest()
    )
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"] == "Hi"
    record = responses_store.get(body["id"])
    assert [message["content"] for message in record["conversation"]] == ["hi", "Hi"]
    assert all(message["role"] != "system" for message in record["conversation"])


async def test_create_response_non_stream_persists_through_a_worker_thread(monkeypatch, responses_store):
    threads = []
    real_set = responses_store.set

    def recording_set(key, value):
        threads.append(threading.current_thread())
        real_set(key, value)

    monkeypatch.setattr(responses_store, "set", recording_set)
    loop_thread = threading.current_thread()

    async def provider_call(chat_req):
        assert chat_req.messages[0].role == "system"
        assert chat_req.messages[0].content == "be brief"
        return {
            "id": "c1",
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", _dispatcher(provider_call))
    body = await openai_mod.create_response(ResponsesRequest(model="deepseek-v4.1-flash", input="hi", instructions="be brief"), _FakeRequest())
    assert body["status"] == "completed"
    record = responses_store.get(body["id"])
    assert record["public"]["status"] == "completed"
    assert [message["content"] for message in record["conversation"]] == ["hi", "Hi"]
    assert len(threads) == 2
    assert all(thread is not loop_thread for thread in threads)


async def test_create_response_skips_the_store_when_store_is_false(monkeypatch, responses_store):
    writes = []
    monkeypatch.setattr(responses_store, "set", lambda key, value: writes.append(key))

    async def provider_call(chat_req):
        return {"id": "c1", "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", _dispatcher(provider_call))
    body = await openai_mod.create_response(ResponsesRequest(model="deepseek-v4.1-flash", input="hi", store=False), _FakeRequest())
    assert body["status"] == "completed"
    assert writes == []
    assert body["id"] not in openai_mod._INFLIGHT_RESPONSES


async def test_create_response_persists_a_cancelled_non_stream_result(monkeypatch, responses_store):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def provider_call(chat_req):
        entered.set()
        await release.wait()
        return {"id": "c1", "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", _dispatcher(provider_call))
    request = ResponsesRequest(model="deepseek-v4.1-flash", input="hi")
    task = asyncio.create_task(openai_mod.create_response(request, _FakeRequest()))
    await entered.wait()
    response_id = next(iter(openai_mod._INFLIGHT_RESPONSES))
    assert responses_store.get(response_id)["public"]["status"] == "in_progress"
    cancelled = await openai_mod.cancel_response(response_id)
    assert cancelled["status"] == "cancelled"
    release.set()
    body = await task
    assert body["status"] == "cancelled"
    assert body["incomplete_details"] == {"reason": "cancelled"}
    assert responses_store.get(response_id)["public"]["status"] == "cancelled"
    assert response_id not in openai_mod._INFLIGHT_RESPONSES


async def test_create_response_drops_the_inflight_entry_when_the_provider_fails(monkeypatch, responses_store):
    async def provider_call(chat_req):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", _dispatcher(provider_call))
    with pytest.raises(RuntimeError, match="provider exploded"):
        await openai_mod.create_response(ResponsesRequest(model="deepseek-v4.1-flash", input="hi"), _FakeRequest())
    assert openai_mod._INFLIGHT_RESPONSES == {}


class _ControlledStream:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.yielded = 0
        self.closed = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.yielded >= len(self.chunks):
            raise StopAsyncIteration
        index = self.yielded
        self.yielded += 1
        if index == 0:
            self.started.set()
        await self.release.wait()
        return self.chunks[index]

    async def aclose(self):
        self.closed = True


async def test_cancel_stops_the_upstream_and_persists_the_cancelled_status(monkeypatch, responses_store):
    loop_thread = threading.current_thread()
    store_threads = []
    real_set = responses_store.set

    def recording_set(key, value):
        store_threads.append(threading.current_thread())
        real_set(key, value)

    monkeypatch.setattr(responses_store, "set", recording_set)
    upstream = _ControlledStream([_chat_chunk("Hi"), _chat_chunk(" there"), _chat_chunk(" again")])

    async def provider_call(chat_req):
        assert chat_req.stream is True
        assert chat_req.stream_options == {"include_usage": True}
        return _ChatResponse(upstream)

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", _dispatcher(provider_call))
    request = ResponsesRequest(model="deepseek-v4.1-flash", input="hi", stream=True)
    response = await openai_mod.create_response(request, _FakeRequest())
    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache"
    body_iterator = response.body_iterator
    created = json.loads((await body_iterator.__anext__())[len("event: response.created\ndata: ") :])
    response_id = created["response"]["id"]
    assert created["response"]["status"] == "in_progress"
    assert (await body_iterator.__anext__()).startswith("event: response.in_progress")
    pending = asyncio.create_task(body_iterator.__anext__())
    await upstream.started.wait()

    assert responses_store.get(response_id)["public"]["status"] == "in_progress"
    entry = openai_mod._INFLIGHT_RESPONSES[response_id]
    assert entry == {"cancel": False}

    cancelled = await openai_mod.cancel_response(response_id)
    assert cancelled["status"] == "cancelled"
    assert cancelled["incomplete_details"] == {"reason": "cancelled"}
    assert entry["cancel"] is True
    assert responses_store.get(response_id)["public"]["status"] == "cancelled"

    upstream.release.set()
    resumed = await pending
    assert resumed.startswith("event: ")
    tail = await _drain(body_iterator)
    assert any("response.output_text.delta" in line for line in [resumed, *tail])
    assert upstream.yielded == 1
    assert upstream.closed is True
    assert any("response.completed" in line for line in tail)
    assert response_id not in openai_mod._INFLIGHT_RESPONSES
    assert responses_store.get(response_id)["public"]["status"] == "cancelled"
    assert responses_store.get(response_id)["public"]["status"] != "completed"
    assert all(thread is not loop_thread for thread in store_threads)


async def test_create_response_stream_persists_the_completed_status(monkeypatch, responses_store):
    async def provider_call(chat_req):
        return _ChatResponse([_chat_chunk("Hi")])

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", _dispatcher(provider_call))
    response = await openai_mod.create_response(ResponsesRequest(model="deepseek-v4.1-flash", input="hi", stream=True), _FakeRequest())
    lines = await _drain(response.body_iterator)
    response_id = json.loads(lines[0][len("event: response.created\ndata: ") :])["response"]["id"]
    assert responses_store.get(response_id)["public"]["status"] == "completed"
    assert [message["content"] for message in responses_store.get(response_id)["conversation"]] == ["hi", "Hi"]
    assert response_id not in openai_mod._INFLIGHT_RESPONSES


async def test_create_response_stream_skips_the_store_when_store_is_false(monkeypatch, responses_store):
    writes = []
    monkeypatch.setattr(responses_store, "set", lambda key, value: writes.append(key))

    async def provider_call(chat_req):
        return _ChatResponse([_chat_chunk("Hi")])

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", _dispatcher(provider_call))
    response = await openai_mod.create_response(ResponsesRequest(model="deepseek-v4.1-flash", input="hi", stream=True, store=False), _FakeRequest())
    lines = await _drain(response.body_iterator)
    assert any("response.completed" in line for line in lines)
    assert writes == []
    assert openai_mod._INFLIGHT_RESPONSES == {}


async def test_responses_stream_close_failure_is_logged(monkeypatch, responses_store, caplog):
    class _BadStream:
        def __init__(self):
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return "event: response.created\ndata: {}\n\n"

        async def aclose(self):
            raise RuntimeError("close exploded")

    monkeypatch.setattr(responses_api, "translate_stream", lambda *args, **kwargs: _BadStream())

    async def provider_call(chat_req):
        return _ChatResponse([_chat_chunk("Hi")])

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", _dispatcher(provider_call))
    response = await openai_mod.create_response(ResponsesRequest(model="deepseek-v4.1-flash", input="hi", stream=True), _FakeRequest())
    with caplog.at_level(logging.DEBUG, logger="danyapi.api"):
        assert await _drain(response.body_iterator) == ["event: response.created\ndata: {}\n\n"]
    assert "responses stream close failed: close exploded" in caplog.records[0].getMessage()
    assert openai_mod._INFLIGHT_RESPONSES == {}


def test_stored_response_404s_without_a_store_or_a_record(monkeypatch):
    app.state.responses_store = None
    with pytest.raises(HTTPException) as excinfo:
        openai_mod._stored_response("resp_1")
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "response resp_1 not found"
    store = JsonStore("openai-cov2-404", None)
    app.state.responses_store = store
    store.set("resp_2", "not a record")
    with pytest.raises(HTTPException) as excinfo:
        openai_mod._stored_response("resp_2")
    assert excinfo.value.detail == "response resp_2 not found"


async def test_get_response_returns_404_for_a_record_without_a_public_object(responses_store):
    responses_store.set("resp_1", {"conversation": []})
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod.get_response("resp_1")
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "response resp_1 not found"
    responses_store.set("resp_2", {"public": "not a dict"})
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod.get_response("resp_2")
    assert excinfo.value.detail == "response resp_2 not found"


async def test_get_and_delete_round_trip(responses_store):
    _record(responses_store, "resp_1")
    assert (await openai_mod.get_response("resp_1"))["id"] == "resp_1"
    assert await openai_mod.delete_response("resp_1") == {"id": "resp_1", "object": "response.deleted", "deleted": True}
    assert responses_store.get("resp_1") is None
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod.delete_response("resp_1")
    assert excinfo.value.status_code == 404


def test_a_read_does_not_create_a_responses_cache_file(monkeypatch, tmp_path):
    monkeypatch.setattr(store_mod.settings, "cache_enabled", True)
    monkeypatch.setattr(store_mod.settings, "cache_dir", str(tmp_path))
    app.state.responses_store = None
    client = TestClient(app)
    response = client.get("/v1/responses/resp_never_stored")
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "response resp_never_stored not found"
    assert list(tmp_path.iterdir()) == []
    client.close()


async def _input_items(response_id, **kwargs):
    kwargs.setdefault("limit", openai_mod.INPUT_ITEMS_DEFAULT_LIMIT)
    kwargs.setdefault("order", "desc")
    return await openai_mod.get_response_input_items(response_id, **kwargs)


def _items(store, response_id, count, **record):
    conversation = [{"role": "user", "content": f"message {index}"} for index in range(count)]
    store.set(response_id, {"public": {"id": response_id, "status": "completed"}, "conversation": conversation, **record})


async def test_input_items_pages_in_both_orders(responses_store):
    _items(responses_store, "resp_1", 5)
    default_page = await _input_items("resp_1")
    assert default_page["object"] == "response.input_items_list"
    assert default_page["has_more"] is False
    assert default_page["first_id"] == default_page["data"][0]["id"]
    assert default_page["last_id"] == default_page["data"][-1]["id"]
    assert [item["content"][0]["text"] for item in default_page["data"]] == [f"message {index}" for index in reversed(range(5))]
    assert len(default_page["data"]) == 5
    ascending = await _input_items("resp_1", order="asc")
    assert [item["content"][0]["text"] for item in ascending["data"]] == [f"message {index}" for index in range(5)]
    assert ascending["first_id"] == ascending["data"][0]["id"]
    assert ascending["last_id"] == ascending["data"][-1]["id"]
    assert [item["id"] for item in default_page["data"]] == list(reversed([item["id"] for item in ascending["data"]]))
    first_two = await openai_mod.get_response_input_items("resp_1", order="asc", limit=2)
    assert [item["content"][0]["text"] for item in first_two["data"]] == ["message 0", "message 1"]
    assert first_two["has_more"] is True
    assert first_two["last_id"] == first_two["data"][-1]["id"]
    newest_two = await _input_items("resp_1", limit=2)
    assert [item["content"][0]["text"] for item in newest_two["data"]] == ["message 3", "message 4"]
    assert newest_two["has_more"] is True
    assert newest_two["first_id"] == newest_two["data"][0]["id"]
    assert newest_two["last_id"] == newest_two["data"][-1]["id"]


async def test_input_items_slices_after_a_cursor(responses_store):
    _items(responses_store, "resp_1", 4)
    ascending = await _input_items("resp_1", order="asc")
    cursor = ascending["data"][1]["id"]
    after = await _input_items("resp_1", after=cursor, order="asc")
    assert [item["content"][0]["text"] for item in after["data"]] == ["message 2", "message 3"]
    assert after["has_more"] is False
    assert after["first_id"] == after["data"][0]["id"]
    unknown = await _input_items("resp_1", after="msg_not_here", order="asc")
    assert unknown == {"object": "response.input_items_list", "data": [], "first_id": None, "last_id": None, "has_more": False}
    last_cursor = ascending["data"][-1]["id"]
    assert (await _input_items("resp_1", after=last_cursor, order="asc"))["data"] == []


async def test_input_items_falls_back_to_the_public_input(responses_store):
    responses_store.set("resp_1", {"public": {"id": "resp_1", "status": "completed", "input": [{"role": "user", "content": "from public"}]}})
    page = await _input_items("resp_1", order="asc")
    assert [item["content"][0]["text"] for item in page["data"]] == ["from public"]
    responses_store.set("resp_2", {"public": {"id": "resp_2", "status": "completed", "input": "not a list"}})
    assert (await _input_items("resp_2"))["data"] == []
    responses_store.set("resp_3", {"conversation": "not a list", "public": {"id": "resp_3", "status": "completed"}})
    empty = await _input_items("resp_3")
    assert empty == {"object": "response.input_items_list", "data": [], "first_id": None, "last_id": None, "has_more": False}


def test_input_items_limit_and_order_bounds(responses_store):
    _items(responses_store, "resp_1", 3)
    client = TestClient(app)
    ok = client.get("/v1/responses/resp_1/input_items", params={"order": "asc"})
    assert ok.status_code == 200
    assert len(ok.json()["data"]) == 3
    too_small = client.get("/v1/responses/resp_1/input_items", params={"limit": 0})
    assert too_small.status_code == 400
    assert "limit" in too_small.json()["error"]["message"]
    too_large = client.get("/v1/responses/resp_1/input_items", params={"limit": 101})
    assert too_large.status_code == 400
    assert "limit" in too_large.json()["error"]["message"]
    assert client.get("/v1/responses/resp_1/input_items", params={"limit": 1}).status_code == 200
    assert client.get("/v1/responses/resp_1/input_items", params={"limit": 100}).status_code == 200
    bad_order = client.get("/v1/responses/resp_1/input_items", params={"order": "sideways"})
    assert bad_order.status_code == 400
    assert "order" in bad_order.json()["error"]["message"]
    assert client.get("/v1/responses/resp_1/input_items", params={"after": "msg_x"}).status_code == 200
    client.close()
    assert openai_mod.INPUT_ITEMS_DEFAULT_LIMIT == 20
    assert openai_mod.INPUT_ITEMS_MAX_LIMIT == 100


async def test_cancel_rejects_a_terminal_response(responses_store):
    _record(responses_store, "resp_1")
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod.cancel_response("resp_1")
    assert excinfo.value.status_code == 409
    assert excinfo.value.detail == "response resp_1 is not cancellable in its current state"
    responses_store.set("resp_2", {"public": {"id": "resp_2", "status": "queued"}})
    cancelled = await openai_mod.cancel_response("resp_2")
    assert cancelled["status"] == "cancelled"
    assert cancelled["incomplete_details"] == {"reason": "cancelled"}
    assert responses_store.get("resp_2") == {"public": {"id": "resp_2", "status": "cancelled", "incomplete_details": {"reason": "cancelled"}}}


async def test_cancel_marks_the_inflight_entry_when_one_exists(responses_store):
    responses_store.set("resp_1", {"public": {"id": "resp_1", "status": "in_progress"}})
    entry = {"cancel": False}
    openai_mod._INFLIGHT_RESPONSES["resp_1"] = entry
    assert (await openai_mod.cancel_response("resp_1"))["status"] == "cancelled"
    assert entry == {"cancel": True}


async def test_unknown_v1_route_is_a_404():
    with pytest.raises(HTTPException) as excinfo:
        await openai_mod.unknown_v1_route("nope")
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "Unknown /v1 endpoint: /v1/nope"


def test_unknown_v1_route_over_http():
    client = TestClient(app)
    for method in ("get", "post", "put", "patch", "delete"):
        response = getattr(client, method)("/v1/definitely-not-a-route")
        assert response.status_code == 404
        assert response.json()["error"]["message"] == "Unknown /v1 endpoint: /v1/definitely-not-a-route"
    client.close()


def test_model_cache_is_the_one_from_models():
    from danyapi.api import models as models_mod

    assert openai_mod._MODEL_CACHE is models_mod._MODEL_CACHE
    assert not hasattr(openai_mod, "_resolve_model")
    assert not hasattr(openai_mod, "_upstream_model_for")
    assert "_resolve_model" not in openai_mod.__all__
    assert "_upstream_model_for" not in openai_mod.__all__
    assert openai_mod.POOL_ATTRS == core_mod.POOL_ATTRS


def test_exported_names_all_resolve():
    missing = [name for name in openai_mod.__all__ if not hasattr(openai_mod, name)]
    assert missing == []


def test_anthropic_endpoints_answer_unauthenticated_404_for_unknown_models():
    client = TestClient(app)
    messages = client.post("/v1/messages", json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
    assert messages.status_code == 404
    assert messages.json() == {"type": "error", "error": {"type": "not_found_error", "message": "Unknown model: gpt-4"}}
    count_tokens = client.post("/v1/messages/count_tokens", json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
    assert count_tokens.status_code == 404
    assert count_tokens.json()["error"]["type"] == "not_found_error"
    client.close()


def test_anthropic_count_tokens_over_http():
    client = TestClient(app)
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "weather in paris"}], "system": "be brief"}
    response = client.post("/v1/messages/count_tokens", json=body)
    assert response.status_code == 200
    assert response.json() == {"input_tokens": _expected_tokens(body)}
    client.close()


def _expected_tokens(body):
    from danyapi.api import anthropic as anthropic_api
    from danyapi.tokens import estimate_tokens

    total = anthropic_api.count_input_tokens(anthropic_api.normalize_messages(body["messages"]), anthropic_api.normalize_system(body["system"]))
    for converted in anthropic_api.convert_tools(body.get("tools")) or []:
        total += estimate_tokens(openai_mod._tool_token_text(converted))
    return total


async def test_anthropic_messages_rejects_a_response_without_choices(monkeypatch):
    async def provider_call(chat_req):
        return {}

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", _dispatcher(provider_call))
    client = TestClient(app)
    response = client.post(
        "/v1/messages",
        json={"model": "deepseek-v4.1-flash", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
    )
    client.close()
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"
    assert response.json()["error"]["message"] == "upstream returned a response without choices"


def test_anthropic_count_tokens_reports_a_pydantic_failure_as_a_400(monkeypatch):
    from danyapi.api import anthropic as anthropic_api

    def exploding(**kwargs):
        raise ValueError("n is out of range")

    monkeypatch.setattr(openai_mod, "ChatCompletionRequest", exploding)
    client = TestClient(app)
    response = client.post("/v1/messages", json={"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "hi"}], "n": 999})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert anthropic_api.DEFAULT_MAX_TOKENS == 4096
    client.close()


async def test_create_response_stream_persists_a_terminal_record_when_the_client_leaves(monkeypatch, responses_store):
    gate = asyncio.Event()

    async def provider_call(chat_req):
        async def body():
            yield 'data: {"id": "c1", "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": "stop"}]}\n\n'
            await gate.wait()

        return SimpleNamespace(body_iterator=body())

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", _dispatcher(provider_call))
    request = ResponsesRequest(model="deepseek-v4.1-flash", input="hi", stream=True)
    response = await openai_mod.create_response(request, _FakeRequest())
    stream = response.body_iterator
    first = await stream.__anext__()
    assert first.startswith("event: response.created")
    response_id = next(iter(openai_mod._INFLIGHT_RESPONSES))
    assert responses_store.get(response_id)["public"]["status"] == "in_progress"
    await stream.aclose()
    stored = responses_store.get(response_id)["public"]
    assert stored["status"] == "cancelled"
    assert stored["incomplete_details"] == {"reason": "cancelled"}
    assert response_id not in openai_mod._INFLIGHT_RESPONSES


async def test_create_response_stream_keeps_the_completed_record_when_the_client_leaves(monkeypatch, responses_store):
    async def provider_call(chat_req):
        async def body():
            yield 'data: {"id": "c1", "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": "stop"}]}\n\n'

        return SimpleNamespace(body_iterator=body())

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", _dispatcher(provider_call))
    request = ResponsesRequest(model="deepseek-v4.1-flash", input="hi", stream=True)
    response = await openai_mod.create_response(request, _FakeRequest())
    lines = [line async for line in response.body_iterator]
    assert any(line.startswith("event: response.completed") for line in lines)
    response_id = next(iter(stored_ids(responses_store)), None)
    assert response_id is not None
    assert responses_store.get(response_id)["public"]["status"] == "completed"


def stored_ids(store):
    return list(store._data)
