import asyncio
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import danyapi.api.openai as openai_mod
import danyapi.qwen.api as qwen_api
from danyapi.accounts import AccountPool, ContextIndex, DeepSeekAccount
from danyapi.deepseek.client import DeepSeekClient
from danyapi.qwen.accounts import QwenSessionRegistry
from danyapi.sessions import SessionRegistry
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


class TestContextSequence(unittest.TestCase):
    def test_same_context_same_sequence(self):
        self.assertEqual(
            context_sequence([Message("user", "hello")]),
            context_sequence([Message("user", "hello")]),
        )

    def test_assistant_and_tool_excluded(self):
        full = [
            Message("user", "hi"),
            Message("assistant", "hello"),
            Message("assistant", tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]),
            Message("tool", "42", tool_call_id="c1"),
        ]
        self.assertEqual(context_sequence(full), context_sequence([Message("user", "hi")]))

    def test_system_included(self):
        seq = context_sequence([Message("system", "sys"), Message("user", "q")])
        self.assertEqual(len(seq), 2)

    def test_content_change_changes_fingerprint(self):
        self.assertNotEqual(
            context_sequence([Message("user", "one")]),
            context_sequence([Message("user", "two")]),
        )

    def test_growth_keeps_prefix(self):
        one = context_sequence([Message("user", "hi")])
        two = context_sequence([Message("user", "hi"), Message("assistant", "yo"), Message("user", "next")])
        self.assertEqual(one, two[:1])
        self.assertEqual(len(two), 2)

    def test_list_content_fingerprint(self):
        a = [Message("user", [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}}])]
        b = [Message("user", [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,yy"}}])]
        self.assertNotEqual(context_sequence(a), context_sequence(b))

    def test_empty(self):
        self.assertEqual(context_sequence([]), ())
        self.assertEqual(context_sequence([Message("assistant", "a")]), ())


class TestContextIndex(unittest.TestCase):
    def test_exact_match_reuses(self):
        idx = ContextIndex(16)
        idx.index("s1", ("a",))
        self.assertEqual(idx.lookup(("a",)), "s1")

    def test_continuation_matches_prefix(self):
        idx = ContextIndex(16)
        idx.index("s1", ("a",))
        idx.index("s2", ("x",))
        self.assertEqual(idx.lookup(("a", "b")), "s1")
        self.assertEqual(idx.lookup(("x", "y", "z")), "s2")

    def test_prefers_longest_prefix(self):
        idx = ContextIndex(16)
        idx.index("s1", ("a",))
        idx.index("s2", ("a", "b"))
        self.assertEqual(idx.lookup(("a", "b")), "s2")
        self.assertEqual(idx.lookup(("a", "b", "c")), "s2")

    def test_no_match(self):
        idx = ContextIndex(16)
        idx.index("s1", ("a",))
        self.assertIsNone(idx.lookup(("c",)))
        self.assertIsNone(idx.lookup(()))

    def test_eviction(self):
        idx = ContextIndex(2)
        idx.index("s1", ("a",))
        idx.index("s2", ("b",))
        idx.index("s3", ("c",))
        self.assertIsNone(idx.lookup(("a",)))
        self.assertEqual(idx.lookup(("b",)), "s2")
        self.assertEqual(idx.lookup(("c",)), "s3")

    def test_never_shrinks_stored_sequence(self):
        idx = ContextIndex(16)
        idx.index("s1", ("a", "b"))
        idx.index("s1", ("a",))
        self.assertEqual(idx.lookup(("a", "b", "c")), "s1")

    def test_forget(self):
        idx = ContextIndex(16)
        idx.index("s1", ("a",))
        idx.forget("s1")
        self.assertIsNone(idx.lookup(("a",)))

    def test_recency_tiebreak(self):
        idx = ContextIndex(16)
        idx.index("s1", ("a",))
        idx.index("s2", ("a",))
        self.assertEqual(idx.lookup(("a",)), "s2")


class TestSessionRegistry(unittest.TestCase):
    def test_reuses_explicit_session(self):
        reg = SessionRegistry(FakeSessionClient())
        s1, k1 = asyncio.run(reg.obtain(None))
        s2, k2 = asyncio.run(reg.obtain(k1))
        self.assertIs(s1, s2)
        self.assertEqual(k2, k1)

    def test_evicts_oldest(self):
        reg = SessionRegistry(FakeSessionClient(), maxsize=2)
        asyncio.run(reg.obtain(None))
        asyncio.run(reg.obtain(None))
        asyncio.run(reg.obtain(None))
        self.assertIsNone(reg.get("s1"))
        self.assertIsNotNone(reg.get("s2"))
        self.assertIsNotNone(reg.get("s3"))

    def test_touch_keeps_fresh(self):
        reg = SessionRegistry(FakeSessionClient(), maxsize=2)
        asyncio.run(reg.obtain(None))
        asyncio.run(reg.obtain(None))
        reg.get("s1")
        asyncio.run(reg.obtain(None))
        self.assertIsNone(reg.get("s2"))
        self.assertIsNotNone(reg.get("s1"))

    def test_touch_last_message(self):
        reg = SessionRegistry(FakeSessionClient())
        _, k = asyncio.run(reg.obtain(None))
        reg.touch_last_message(k, "m1")
        self.assertEqual(reg.get(k).last_message_id, "m1")


class TestQwenSessionRegistry(unittest.TestCase):
    def test_reuses_and_evicts(self):
        reg = QwenSessionRegistry(FakeSessionClient(), maxsize=2)
        _, k1 = asyncio.run(reg.obtain(None, "qwen3.8-max"))
        _, k2 = asyncio.run(reg.obtain(None, "qwen3.8-max"))
        _, k3 = asyncio.run(reg.obtain(None, "qwen3.8-max"))
        self.assertIsNone(reg.get(k1))
        self.assertIsNotNone(reg.get(k2))
        self.assertIsNotNone(reg.get(k3))
        s, k = asyncio.run(reg.obtain(k2, "qwen3.8-max"))
        self.assertEqual(k, k2)
        self.assertEqual(s.id, k2)

    def test_touch_last_message(self):
        reg = QwenSessionRegistry(FakeSessionClient())
        s, k = asyncio.run(reg.obtain(None, "qwen3.8-max"))
        reg.touch_last_message(k, "r1")
        self.assertEqual(s.last_response_id, "r1")


class TestAccountPoolContext(unittest.TestCase):
    def test_resolve_routes_to_owner(self):
        async def run():
            a0, a1 = make_acct(0), make_acct(1)
            pool = AccountPool([a0, a1])
            pool.register(1, "sess-a")
            pool.index_context("sess-a", ("u1",))
            cached = pool.resolve_context(("u1",))
            self.assertEqual(cached, "sess-a")
            acct, sid = await pool.acquire(cached)
            self.assertIs(acct, a1)
            self.assertEqual(sid, "sess-a")

        asyncio.run(run())

    def test_broken_owner_drops_context(self):
        async def run():
            a0, a1 = make_acct(0), make_acct(1)
            a0.broken = True
            pool = AccountPool([a0, a1])
            pool.register(0, "sess-broken")
            pool.index_context("sess-broken", ("u1",))
            acct, sid = await pool.acquire("sess-broken")
            self.assertIs(acct, a1)
            self.assertIsNone(sid)
            self.assertIsNone(pool.resolve_context(("u1",)))

        asyncio.run(run())

    def test_forget_removes_mapping(self):
        pool = AccountPool([make_acct(0)])
        pool.register(0, "s1")
        self.assertEqual(pool.account_for_session("s1").index, 0)
        pool.forget("s1")
        self.assertIsNone(pool.account_for_session("s1"))


class TestBuildPromptSession(unittest.TestCase):
    def test_plain_continuation_sends_only_last_user(self):
        messages = [
            Message("user", "What is the weather?"),
            Message("assistant", "It is 22C."),
            Message("user", "And in Rome?"),
        ]
        prompt, tool_mode = build_prompt(messages, has_session=True)
        self.assertFalse(tool_mode)
        self.assertEqual(prompt, "And in Rome?")

    def test_continuation_after_tool_round_with_new_user(self):
        messages = [
            Message("user", "What is the weather?"),
            Message("assistant", tool_calls=[{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Moscow"}'}}]),
            Message("tool", "22C, sunny", tool_call_id="c1"),
            Message("assistant", "It is 22C in Moscow."),
            Message("user", "And in Rome?"),
        ]
        prompt, tool_mode = build_prompt(messages, has_session=True)
        self.assertFalse(tool_mode)
        self.assertEqual(prompt, "And in Rome?")
        self.assertNotIn("22C, sunny", prompt)

    def test_immediate_tool_round_uses_tail(self):
        messages = [
            Message("user", "What is the weather?"),
            Message("assistant", tool_calls=[{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Moscow"}'}}]),
            Message("tool", "22C, sunny", tool_call_id="c1"),
        ]
        prompt, tool_mode = build_prompt(messages, has_session=True)
        self.assertTrue(tool_mode)
        self.assertIn("22C, sunny", prompt)
        self.assertIn("Continue the conversation", prompt)
        self.assertNotIn("What is the weather?", prompt)

    def test_session_continuation_with_tools(self):
        messages = [
            Message("user", "What is the weather?"),
            Message("assistant", "It is 22C."),
            Message("user", "And in Rome?"),
        ]
        prompt, tool_mode = build_prompt(messages, [WEATHER_TOOL], has_session=True)
        self.assertTrue(tool_mode)
        self.assertIn("And in Rome?", prompt)
        self.assertIn("get_weather", prompt)

    def test_session_skips_system(self):
        messages = [Message("system", "Be concise."), Message("user", "hi")]
        prompt, _ = build_prompt(messages, has_session=True)
        self.assertEqual(prompt, "hi")


class TestDeepSeekCachedSession(unittest.TestCase):
    def test_stateless_request_resolves_cached_session(self):
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
            self.assertEqual(captured["existing_sid"], "sess-a")
            self.assertEqual(captured["context_seq"], openai_mod.toolemu.context_sequence(req.messages))
        finally:
            openai_mod._collect_non_stream = orig

    def test_explicit_session_bypasses_context_resolution(self):
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
            self.assertEqual(captured["existing_sid"], "explicit-1")
        finally:
            openai_mod._collect_non_stream = orig


class TestQwenCachedSession(unittest.TestCase):
    def test_stateless_request_resolves_cached_session(self):
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
            self.assertEqual(captured["existing_sid"], "sess-q")
            self.assertEqual(captured["context_seq"], openai_mod.toolemu.context_sequence(req.messages))
        finally:
            qwen_api.collect_non_stream = orig


class TestNonStreamUsageKwarg(unittest.TestCase):
    def test_deepseek_non_stream_accepts_include_usage(self):
        self.assertIn("include_usage", openai_mod._collect_non_stream.__code__.co_varnames)

    def test_qwen_non_stream_accepts_include_usage(self):
        self.assertIn("include_usage", qwen_api.collect_non_stream.__code__.co_varnames)


if __name__ == "__main__":
    unittest.main()
