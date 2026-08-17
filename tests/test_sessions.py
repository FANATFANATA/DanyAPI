import asyncio
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import danyapi.api.openai as openai_mod
import danyapi.qwen.api as qwen_api
from danyapi.accounts import AccountPool, DeepSeekAccount
from danyapi.deepseek.client import DeepSeekClient
from danyapi.qwen.accounts import QwenSessionRegistry
from danyapi.sessions import SessionRegistry
from danyapi.store import JsonStore
from danyapi.tools import build_prompt, context_sequence

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


class Message:
    def __init__(self, role="user", content: Any = "", tool_calls=None, tool_call_id=None, name=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.name = name


class FakeSessionClient:
    def __init__(self):
        self.counter = 0

    async def create_session(self):
        self.counter += 1
        return SimpleNamespace(id=f"s{self.counter}")

    async def create_chat(self, model="", chat_mode="normal"):
        self.counter += 1
        return f"c{self.counter}"


def make_acct(i):
    client = MagicMock(spec=DeepSeekClient)
    client.index = i
    return DeepSeekAccount(i, client)


def test_same_context_same_sequence():
    assert context_sequence([Message("user", "hello")]) == context_sequence([Message("user", "hello")])


def test_assistant_and_tool_excluded():
    full = [
        Message("user", "hi"),
        Message("assistant", "hello"),
        Message("assistant", tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]),
        Message("tool", "42", tool_call_id="c1"),
    ]
    assert context_sequence(full) == context_sequence([Message("user", "hi")])


def test_system_included():
    seq = context_sequence([Message("system", "sys"), Message("user", "q")])
    assert len(seq) == 2


def test_content_change_changes_fingerprint():
    assert context_sequence([Message("user", "one")]) != context_sequence([Message("user", "two")])


def test_growth_keeps_prefix():
    one = context_sequence([Message("user", "hi")])
    two = context_sequence([Message("user", "hi"), Message("assistant", "yo"), Message("user", "next")])
    assert one == two[:1]
    assert len(two) == 2


def test_list_content_fingerprint():
    a = [Message("user", [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}}])]
    b = [Message("user", [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,yy"}}])]
    assert context_sequence(a) != context_sequence(b)


def test_context_empty():
    assert context_sequence([]) == ()
    assert context_sequence([Message("assistant", "a")]) == ()


def test_user_scopes_fingerprint():
    base = [Message("user", "hello")]
    assert context_sequence(base) == context_sequence(base)
    assert context_sequence(base, user="alice") != context_sequence(base, user="bob")
    assert context_sequence(base, user="alice") == context_sequence(base, user="alice")


def test_user_keeps_continuation_prefix():
    one = context_sequence([Message("user", "hi")], user="alice")
    two = context_sequence([Message("user", "hi"), Message("assistant", "yo"), Message("user", "next")], user="alice")
    assert one == two[:1]
    assert context_sequence([Message("user", "hi")], user="alice") != context_sequence([Message("user", "hi")], user="bob")


def test_session_reuses_explicit():
    reg = SessionRegistry(FakeSessionClient())
    s1, k1 = asyncio.run(reg.obtain(None))
    s2, k2 = asyncio.run(reg.obtain(k1))
    assert s1 is s2
    assert k2 == k1


def test_session_evicts_oldest():
    reg = SessionRegistry(FakeSessionClient(), maxsize=2)
    asyncio.run(reg.obtain(None))
    asyncio.run(reg.obtain(None))
    asyncio.run(reg.obtain(None))
    assert reg.get("s1") is None
    assert reg.get("s2") is not None
    assert reg.get("s3") is not None


def test_session_touch_keeps_fresh():
    reg = SessionRegistry(FakeSessionClient(), maxsize=2)
    asyncio.run(reg.obtain(None))
    asyncio.run(reg.obtain(None))
    reg.get("s1")
    asyncio.run(reg.obtain(None))
    assert reg.get("s2") is None
    assert reg.get("s1") is not None


def test_session_touch_last_message():
    reg = SessionRegistry(FakeSessionClient())
    _, k = asyncio.run(reg.obtain(None))
    reg.touch_last_message(k, "m1")
    assert reg.get(k).last_message_id == "m1"


def test_session_forget_removes():
    reg = SessionRegistry(FakeSessionClient())
    _, k = asyncio.run(reg.obtain(None))
    assert reg.get(k) is not None
    reg.forget(k)
    assert reg.get(k) is None
    _, k2 = asyncio.run(reg.obtain(k))
    assert k2 != k


def test_session_ttl_expires():
    reg = SessionRegistry(FakeSessionClient(), maxsize=16, ttl=0.1)
    s, k = asyncio.run(reg.obtain(None))
    assert reg.get(k) is not None
    reg._sessions[k] = (s, time.monotonic() - 10)
    assert reg.get(k) is None


def test_qwen_session_reuses_and_evicts():
    reg = QwenSessionRegistry(FakeSessionClient(), maxsize=2)
    _, k1 = asyncio.run(reg.obtain(None, "qwen3.8-max"))
    _, k2 = asyncio.run(reg.obtain(None, "qwen3.8-max"))
    _, _k3 = asyncio.run(reg.obtain(None, "qwen3.8-max"))
    assert reg.get(k1) is None
    assert reg.get(k2) is not None
    s, k = asyncio.run(reg.obtain(k2, "qwen3.8-max"))
    assert k == k2
    assert s.id == k2


def test_qwen_session_touch_last_message():
    reg = QwenSessionRegistry(FakeSessionClient())
    s, k = asyncio.run(reg.obtain(None, "qwen3.8-max"))
    reg.touch_last_message(k, "r1")
    assert s.last_response_id == "r1"


def test_qwen_model_switch_creates_new_chat():
    reg = QwenSessionRegistry(FakeSessionClient())
    _, k1 = asyncio.run(reg.obtain(None, "qwen3.8-max"))
    _, k2 = asyncio.run(reg.obtain(k1, "qwen3.7-plus"))
    assert k2 != k1
    s, k3 = asyncio.run(reg.obtain(k1, "qwen3.8-max"))
    assert k3 == k1
    assert s.model == "qwen3.8-max"


@pytest.fixture
def sessions_store(tmp_path, monkeypatch):
    import danyapi.store as store_mod

    monkeypatch.setattr(store_mod.settings, "cache_dir", str(tmp_path))
    return JsonStore("sessions", "default")


def test_obtain_persists(sessions_store):
    reg = SessionRegistry(FakeSessionClient(), store=sessions_store)
    _, key = asyncio.run(reg.obtain(None))
    assert key in sessions_store
    reg2 = SessionRegistry(FakeSessionClient(), store=JsonStore("sessions", "default"))
    assert reg2.get(key) is not None


def test_restore_respects_prefix(sessions_store):
    sessions_store.set("9:s9", {"id": "cs9"})
    sessions_store.set("0:s0", {"id": "cs0"})
    reg = SessionRegistry(FakeSessionClient(), store=JsonStore("sessions", "default"), key_prefix="0:")
    assert reg.get("s9") is None
    assert reg.get("s0") is not None


def test_restore_skips_invalid_records(sessions_store):
    sessions_store.set("0:ok", {"id": "cs1"})
    sessions_store.set("0:bad", "garbage")
    sessions_store.set("0:no-id", {})
    sessions_store.set("0:", {"id": "x"})
    reg = SessionRegistry(FakeSessionClient(), store=JsonStore("sessions", "default"), key_prefix="0:")
    assert reg.get("ok") is not None
    assert reg.get("bad") is None
    assert reg.get("no-id") is None
    assert len(reg._sessions) == 1


def test_restore_trims_to_maxsize(sessions_store):
    for i in range(5):
        sessions_store.set(f"0:s{i}", {"id": f"cs{i}"})
    reg = SessionRegistry(FakeSessionClient(), maxsize=2, store=JsonStore("sessions", "default"), key_prefix="0:")
    assert len(reg._sessions) == 2
    assert "0:s0" not in JsonStore("sessions", "default")


def test_expiry_discards_from_store(sessions_store):
    reg = SessionRegistry(FakeSessionClient(), maxsize=16, ttl=0.1, store=sessions_store)
    _, key = asyncio.run(reg.obtain(None))
    assert key in sessions_store
    reg._sessions[key] = (reg._sessions[key][0], time.monotonic() - 10)
    assert reg.get(key) is None
    assert key not in sessions_store


def test_eviction_discards_from_store(sessions_store):
    reg = SessionRegistry(FakeSessionClient(), maxsize=1, store=sessions_store)
    _, k1 = asyncio.run(reg.obtain(None))
    _, k2 = asyncio.run(reg.obtain(None))
    assert k1 not in sessions_store
    assert k2 in sessions_store


def test_touch_last_message_persists(sessions_store):
    reg = SessionRegistry(FakeSessionClient(), store=sessions_store)
    _, key = asyncio.run(reg.obtain(None))
    reg.touch_last_message(key, "m9")
    reg2 = SessionRegistry(FakeSessionClient(), store=JsonStore("sessions", "default"))
    session = reg2.get(key)
    assert session is not None
    assert session.last_message_id == "m9"


def test_forget_removes_from_store(sessions_store):
    reg = SessionRegistry(FakeSessionClient(), store=sessions_store)
    _, key = asyncio.run(reg.obtain(None))
    reg.forget(key)
    assert key not in sessions_store


def test_close_all(sessions_store):
    reg = SessionRegistry(FakeSessionClient(), store=sessions_store, key_prefix="0:")
    _, k0 = asyncio.run(reg.obtain(None))
    other = SessionRegistry(FakeSessionClient(), store=sessions_store, key_prefix="1:")
    _, k1 = asyncio.run(other.obtain(None))
    reg.close_all()
    assert len(reg._sessions) == 0
    assert "0:" + k0 not in sessions_store
    assert "1:" + k1 in sessions_store


def test_get_missing_and_empty():
    reg = SessionRegistry(FakeSessionClient())
    assert reg.get(None) is None
    assert reg.get("") is None


def test_qwen_persist_roundtrip(sessions_store):
    reg = QwenSessionRegistry(FakeSessionClient(), store=sessions_store)
    _, key = asyncio.run(reg.obtain(None, "qwen3.8-max"))
    reg.touch_last_message(key, "r1")
    reg2 = QwenSessionRegistry(FakeSessionClient(), store=JsonStore("sessions", "default"))
    session = reg2.get(key)
    assert session is not None
    assert session.model == "qwen3.8-max"
    assert session.last_response_id == "r1"


def test_qwen_broken_records_skipped(sessions_store):
    sessions_store.set("0:bad", {"id": ""})
    sessions_store.set("0:ok", {"id": "cs1", "model": "m"})
    reg = QwenSessionRegistry(FakeSessionClient(), store=JsonStore("sessions", "default"), key_prefix="0:")
    assert reg.get("ok") is not None
    assert reg.get("bad") is None


def test_account_pool_resolve_routes_to_owner():
    a0, a1 = make_acct(0), make_acct(1)
    pool = AccountPool([a0, a1])
    pool.register(1, "sess-a")
    pool.index_context("sess-a", ("u1",))
    cached = pool.resolve_context(("u1",))
    assert cached == "sess-a"


def test_account_pool_forget_removes_mapping():
    pool = AccountPool([make_acct(0)])
    pool.register(0, "s1")
    assert pool.account_for_session("s1").index == 0
    pool.forget("s1")
    assert pool.account_for_session("s1") is None


def test_account_pool_stats_reports_cache_state():
    pool = AccountPool([make_acct(0), make_acct(1)])
    pool.register(0, "s1")
    pool.index_context("s1", ("a",))
    stats = pool.stats()
    assert stats["accounts"] == 2
    assert stats["healthy"] == 2
    assert stats["session_affinities"] == 1
    assert stats["context_entries"] == 1
    pool.resolve_context(("a",))
    assert pool.stats()["context_hits"] == 1


def test_plain_continuation_sends_only_last_user():
    messages = [
        Message("user", "What is the weather?"),
        Message("assistant", "It is 22C."),
        Message("user", "And in Rome?"),
    ]
    prompt, tool_mode = build_prompt(messages, has_session=True)
    assert not tool_mode
    assert prompt == "And in Rome?"


def test_continuation_after_tool_round_with_new_user():
    messages = [
        Message("user", "What is the weather?"),
        Message("assistant", tool_calls=[{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Moscow"}'}}]),
        Message("tool", "22C, sunny", tool_call_id="c1"),
        Message("assistant", "It is 22C in Moscow."),
        Message("user", "And in Rome?"),
    ]
    prompt, tool_mode = build_prompt(messages, has_session=True)
    assert not tool_mode
    assert prompt == "And in Rome?"
    assert "22C, sunny" not in prompt


def test_immediate_tool_round_uses_tail():
    messages = [
        Message("user", "What is the weather?"),
        Message("assistant", tool_calls=[{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Moscow"}'}}]),
        Message("tool", "22C, sunny", tool_call_id="c1"),
    ]
    prompt, tool_mode = build_prompt(messages, has_session=True)
    assert tool_mode
    assert "22C, sunny" in prompt
    assert "Continue the conversation" in prompt
    assert "What is the weather?" not in prompt


def test_session_continuation_with_tools():
    messages = [
        Message("user", "What is the weather?"),
        Message("assistant", "It is 22C."),
        Message("user", "And in Rome?"),
    ]
    prompt, tool_mode = build_prompt(messages, [WEATHER_TOOL], has_session=True)
    assert tool_mode
    assert "And in Rome?" in prompt
    assert "get_weather" not in prompt


def test_session_skips_system():
    messages = [Message("system", "Be concise."), Message("user", "hi")]
    prompt, _ = build_prompt(messages, has_session=True)
    assert prompt == "hi"


def test_deepseek_stateless_request_resolves_cached_session():
    captured = {}
    orig = openai_mod._collect_non_stream

    async def fake_collect(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    openai_mod._collect_non_stream = fake_collect
    try:
        pool = MagicMock()
        pool.acquire = AsyncMock(return_value=(MagicMock(), "sess-a"))
        pool.resolve_context = MagicMock(return_value="sess-a")
        openai_mod.app.state.pool = pool
        req = SimpleNamespace(
            model="deepseek-v4-flash",
            stream=False,
            thinking=False,
            search=False,
            session_id=None,
            files=None,
            tools=None,
            tool_choice=None,
            response_format=None,
            messages=[openai_mod.ChatMessage(role="user", content="hello")],
        )
        asyncio.run(openai_mod._chat_completions_deepseek(req))
        pool.resolve_context.assert_called_once()
        assert captured["existing_sid"] == "sess-a"
    finally:
        openai_mod._collect_non_stream = orig


def test_deepseek_cached_missing_session_renders_full_history():
    captured = {}
    orig = openai_mod._collect_non_stream

    async def fake_collect(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    openai_mod._collect_non_stream = fake_collect
    try:
        account = MagicMock()
        account.sessions.get.return_value = None
        pool = MagicMock()
        pool.acquire = AsyncMock(return_value=(account, "sess-a"))
        pool.resolve_context = MagicMock(return_value="sess-a")
        openai_mod.app.state.pool = pool
        req = SimpleNamespace(
            model="deepseek-v4-flash",
            stream=False,
            thinking=False,
            search=False,
            session_id=None,
            files=None,
            tools=None,
            tool_choice=None,
            response_format=None,
            messages=[
                openai_mod.ChatMessage(role="user", content="remember alpha"),
                openai_mod.ChatMessage(role="assistant", content="alpha noted"),
                openai_mod.ChatMessage(role="user", content="what did I ask you to remember?"),
            ],
        )
        asyncio.run(openai_mod._chat_completions_deepseek(req))
        assert "remember alpha" in captured["prompt"]
        assert "alpha noted" in captured["prompt"]
        assert "what did I ask you to remember?" in captured["prompt"]
    finally:
        openai_mod._collect_non_stream = orig


def test_deepseek_explicit_session_bypasses_context_resolution():
    captured = {}
    orig = openai_mod._collect_non_stream

    async def fake_collect(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    openai_mod._collect_non_stream = fake_collect
    try:
        pool = MagicMock()
        pool.acquire = AsyncMock(return_value=(MagicMock(), "explicit-1"))
        pool.resolve_context = MagicMock()
        openai_mod.app.state.pool = pool
        req = SimpleNamespace(
            model="deepseek-v4-flash",
            stream=False,
            thinking=False,
            search=False,
            session_id="explicit-1",
            files=None,
            tools=None,
            tool_choice=None,
            response_format=None,
            messages=[openai_mod.ChatMessage(role="user", content="hello")],
        )
        asyncio.run(openai_mod._chat_completions_deepseek(req))
        pool.acquire.assert_called_once()
        pool.resolve_context.assert_not_called()
        assert captured["existing_sid"] == "explicit-1"
    finally:
        openai_mod._collect_non_stream = orig


def test_qwen_stateless_request_resolves_cached_session():
    captured = {}
    orig = qwen_api.collect_non_stream

    async def fake_collect(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    qwen_api.collect_non_stream = fake_collect
    try:
        pool = MagicMock()
        pool.acquire = AsyncMock(return_value=(MagicMock(), "sess-q"))
        pool.resolve_context = MagicMock(return_value="sess-q")
        openai_mod.app.state.qwen_pool = pool
        req = SimpleNamespace(
            model="qwen3.8-max",
            stream=False,
            thinking=False,
            search=False,
            session_id=None,
            files=None,
            tools=None,
            tool_choice=None,
            response_format=None,
            messages=[openai_mod.ChatMessage(role="user", content="hello")],
        )
        asyncio.run(openai_mod._chat_completions_qwen(req))
        pool.resolve_context.assert_called_once()
        assert captured["existing_sid"] == "sess-q"
    finally:
        qwen_api.collect_non_stream = orig


def test_qwen_cached_wrong_model_renders_full_history():
    captured = {}
    orig = qwen_api.collect_non_stream

    async def fake_collect(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    qwen_api.collect_non_stream = fake_collect
    try:
        account = MagicMock()
        account.sessions.get.return_value = SimpleNamespace(id="sess-q", model="qwen-old")
        account.sessions._reuse.return_value = False
        pool = MagicMock()
        pool.acquire = AsyncMock(return_value=(account, "sess-q"))
        pool.resolve_context = MagicMock(return_value="sess-q")
        openai_mod.app.state.qwen_pool = pool
        req = SimpleNamespace(
            model="qwen3.8-max",
            stream=False,
            thinking=False,
            search=False,
            session_id=None,
            files=None,
            tools=None,
            tool_choice=None,
            response_format=None,
            messages=[
                openai_mod.ChatMessage(role="user", content="remember beta"),
                openai_mod.ChatMessage(role="assistant", content="beta noted"),
                openai_mod.ChatMessage(role="user", content="what did I ask you to remember?"),
            ],
        )
        asyncio.run(openai_mod._chat_completions_qwen(req))
        account.sessions._reuse.assert_called_once()
        assert "remember beta" in captured["prompt"]
        assert "beta noted" in captured["prompt"]
        assert "what did I ask you to remember?" in captured["prompt"]
    finally:
        qwen_api.collect_non_stream = orig


def test_deepseek_non_stream_accepts_include_usage():
    assert "include_usage" in openai_mod._collect_non_stream.__code__.co_varnames


def test_qwen_non_stream_accepts_include_usage():
    assert "include_usage" in qwen_api.collect_non_stream.__code__.co_varnames


def test_restore_empty_key_without_prefix(sessions_store):
    sessions_store.set("", {"id": "cs1"})
    reg = SessionRegistry(FakeSessionClient(), store=JsonStore("sessions", "default"))
    assert len(reg._sessions) == 0


def test_touch_last_message_missing():
    reg = SessionRegistry(FakeSessionClient())
    reg.touch_last_message("nope", "m1")
    assert reg.get("nope") is None


def test_close_all_without_store():
    reg = SessionRegistry(FakeSessionClient())
    asyncio.run(reg.obtain(None))
    reg.close_all()
    assert len(reg._sessions) == 0
