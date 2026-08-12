import unittest

from danyapi.deepseek.sse import SSEEvent, parse_sse
from danyapi.deepseek.stream import IncrementalSSE, MessageReconstructor


def first_set_delta():
    return SSEEvent(
        None,
        {
            "o": "SET",
            "p": "response",
            "v": {
                "response": {
                    "message_id": "asst_1",
                    "parent_id": "user_1",
                    "role": "assistant",
                    "thinking_enabled": True,
                    "inserted_at": 1786469543117,
                    "accumulated_token_usage": 0,
                    "search_enabled": False,
                    "search_triggered": None,
                    "status": "WIP",
                    "ban_regenerate": False,
                    "feedback": None,
                    "fragments": [
                        {"type": "THINK", "content": ""},
                        {"type": "RESPONSE", "content": ""},
                    ],
                    "has_pending_fragment": False,
                    "conversation_mode": "DEFAULT",
                }
            },
        },
    )


class TestSSEParser(unittest.TestCase):
    def test_parse_sse(self):
        raw = (
            "event: ready\n"
            'data: {"response_message_id":"a","request_message_id":"b","model_type":"default"}\n'
            "\n"
            'data: {"o":"BATCH","v":[{"o":"SET","p":"response/fragments/0/content","v":"hi"}]}\n'
            "\n"
            "event: finish\n"
            "data: {}\n"
            "\n"
        )
        events = parse_sse(raw)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].event, "ready")
        self.assertEqual(events[0].data["response_message_id"], "a")
        self.assertIsNone(events[1].event)
        self.assertEqual(events[1].data["o"], "BATCH")
        self.assertEqual(events[2].event, "finish")

    def test_incremental_split(self):
        raw = 'event: ready\ndata: {"a":1}\n\ndata: {"o":"SET","p":"response/x","v":1}\n\n'
        inc = IncrementalSSE()
        collected = []
        mid = len(raw) // 2
        collected.extend(inc.feed(raw[:mid].encode()))
        collected.extend(inc.feed(raw[mid:].encode()))
        collected.extend(inc.finish())
        self.assertEqual(len(collected), 2)
        self.assertEqual(collected[0].event, "ready")

    def test_crlf_separators(self):
        raw = 'event: ready\r\ndata: {"a":1}\r\n\r\ndata: {"o":"SET","p":"response/x","v":1}\r\n\r\n'
        inc = IncrementalSSE()
        collected = list(inc.feed(raw.encode())) + list(inc.finish())
        self.assertEqual(len(collected), 2)
        self.assertEqual(collected[0].event, "ready")
        self.assertEqual(collected[1].data["v"], 1)


class TestReconstruction(unittest.TestCase):
    def test_real_stream_pattern(self):
        rec = MessageReconstructor()
        rec.handle(
            SSEEvent(
                "ready",
                {
                    "request_message_id": 1,
                    "response_message_id": 2,
                    "model_type": "default",
                },
            )
        )
        self.assertEqual(rec.response_message_id, 2)
        rec.handle(
            SSEEvent(
                None,
                {
                    "v": {
                        "response": {
                            "message_id": 2,
                            "parent_id": 1,
                            "role": "ASSISTANT",
                            "status": "WIP",
                            "fragments": [{"id": 2, "type": "RESPONSE", "content": "Г"}],
                        }
                    }
                },
            )
        )
        self.assertEqual(rec.message["id"], 2)
        rec.handle(SSEEvent(None, {"p": "response/fragments/-1/content", "o": "APPEND", "v": "рави"}))
        for token in ("тация", " -", " это"):
            rec.handle(SSEEvent(None, {"v": token}))
        self.assertEqual(rec.content, "Гравитация - это")
        rec.handle(
            SSEEvent(
                None,
                {
                    "p": "response",
                    "o": "BATCH",
                    "v": [
                        {"p": "accumulated_token_usage", "v": 103},
                        {"p": "quasi_status", "v": "FINISHED"},
                    ],
                },
            )
        )
        rec.handle(SSEEvent(None, {"p": "response/status", "o": "SET", "v": "FINISHED"}))
        self.assertEqual(rec.status, "FINISHED")
        self.assertEqual(rec.message.get("accumulated_token_usage"), 103)

    def test_thinking_then_response_fragment(self):
        rec = MessageReconstructor()
        rec.handle(
            SSEEvent(
                None,
                {
                    "v": {
                        "response": {
                            "message_id": 2,
                            "parent_id": 1,
                            "status": "WIP",
                            "fragments": [{"id": 2, "type": "THINK", "content": "Мы"}],
                        }
                    }
                },
            )
        )
        rec.handle(SSEEvent(None, {"p": "response/fragments/-1/elapsed_secs", "o": "SET", "v": 3.5}))
        rec.handle(
            SSEEvent(
                None,
                {
                    "p": "response/fragments",
                    "o": "APPEND",
                    "v": [{"id": 3, "type": "RESPONSE", "content": "Если"}],
                },
            )
        )
        rec.handle(SSEEvent(None, {"p": "response/fragments/-1/content", "v": " вз"}))
        for token in ("ять", " четыре"):
            rec.handle(SSEEvent(None, {"v": token}))
        self.assertEqual(rec.reasoning, "Мы")
        self.assertEqual(rec.content, "Если взять четыре")

    def test_hint_error_capture(self):
        rec = MessageReconstructor()
        rec.handle(
            SSEEvent(
                "ready",
                {
                    "request_message_id": 1,
                    "response_message_id": 2,
                    "model_type": "expert",
                },
            )
        )
        rec.handle(
            SSEEvent(
                "hint",
                {
                    "type": "error",
                    "content": "Server is busy. Try again later, or use Instant Mode.",
                    "clear_response": True,
                    "finish_reason": "expert_busy_use_default",
                },
            )
        )
        self.assertIsNotNone(rec.hint_error)
        hint = rec.hint_error
        assert hint is not None
        self.assertEqual(hint["finish_reason"], "expert_busy_use_default")
        self.assertIn("busy", hint["message"].lower())
        rec2 = MessageReconstructor()
        rec2.handle(SSEEvent("hint", {"type": "info", "content": "ok", "finish_reason": None}))
        self.assertIsNone(rec2.hint_error)

    def test_apply_deltas(self):
        rec = MessageReconstructor()
        rec.handle(first_set_delta())
        self.assertEqual(rec.message["id"], "asst_1")
        rec.handle(
            SSEEvent(
                None,
                {
                    "o": "BATCH",
                    "v": [
                        {
                            "o": "SET",
                            "p": "response/fragments/0/content",
                            "v": "Let me think step by step",
                        },
                        {"o": "SET", "p": "response/fragments/1/content", "v": "Hello"},
                    ],
                },
            )
        )
        self.assertEqual(rec.reasoning, "Let me think step by step")
        self.assertEqual(rec.content, "Hello")
        c_diff, r_diff = rec.take_diffs()
        self.assertEqual(c_diff, "Hello")
        self.assertEqual(r_diff, "Let me think step by step")
        rec.handle(
            SSEEvent(
                None,
                {
                    "o": "BATCH",
                    "v": [
                        {
                            "o": "SET",
                            "p": "response/fragments/1/content",
                            "v": "Hello world",
                        },
                        {"o": "SET", "p": "response/status", "v": "FINISHED"},
                    ],
                },
            )
        )
        c_diff, r_diff = rec.take_diffs()
        self.assertEqual(c_diff, " world")
        self.assertEqual(rec.content, "Hello world")
        self.assertEqual(rec.status, "FINISHED")


if __name__ == "__main__":
    unittest.main()
