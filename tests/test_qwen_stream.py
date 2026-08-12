import unittest

from danyapi.deepseek.sse import SSEEvent
from danyapi.deepseek.stream import IncrementalSSE
from danyapi.qwen.stream import QwenStreamReconstructor


class TestQwenStream(unittest.TestCase):
    def test_incremental_parse_real_stream(self):
        raw = (
            'data: {"response.created":{"chat_id":"c1","parent_id":"p0","response_id":"r1","response_index":"0"}} \n'
            "\n"
            'data: {"choices": [{"delta": {"role": "assistant", "content": "", "phase": "answer", "status": "typing"}}], "response_id": "r1"}\n'
            "\n"
            'data: {"choices": [{"delta": {"role": "assistant", "content": "Hello", "phase": "answer", "status": "typing"}}], "response_id": "r1"}\n'
            "\n"
            'data: {"choices": [{"delta": {"content": "", "role": "assistant", "status": "finished", "phase": "answer"}}], "response_id": "r1"}\n'
            "\n"
        )
        rec = QwenStreamReconstructor()
        inc = IncrementalSSE()
        mid = len(raw) // 2
        for chunk in (raw[:mid].encode(), raw[mid:].encode()):
            for event in inc.feed(chunk):
                rec.handle(event)
        for event in inc.finish():
            rec.handle(event)
        self.assertEqual(rec.response_id, "r1")
        self.assertEqual(rec.content, "Hello")
        self.assertTrue(rec.finished)

    def test_thinking_summary(self):
        events = [
            {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": "",
                            "phase": "thinking_summary",
                            "extra": {
                                "summary_title": {"content": ["Title A"]},
                                "summary_thought": {"content": ["First thought"]},
                            },
                            "status": "typing",
                        }
                    }
                ],
                "response_id": "r1",
            },
            {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": "",
                            "phase": "thinking_summary",
                            "extra": {
                                "summary_title": {"content": ["Title A", "Title B"]},
                                "summary_thought": {"content": ["First thought", "Second thought"]},
                            },
                            "status": "typing",
                        }
                    }
                ],
                "response_id": "r1",
            },
            {
                "choices": [{"delta": {"role": "assistant", "content": "Answer", "phase": "answer", "status": "typing"}}],
                "response_id": "r1",
            },
            {
                "choices": [{"delta": {"role": "assistant", "content": " is here", "phase": "answer", "status": "typing"}}],
                "response_id": "r1",
            },
        ]
        rec = QwenStreamReconstructor()
        for data in events:
            rec.handle(SSEEvent(None, data))
        self.assertEqual(rec.reasoning, "First thought\n\nSecond thought")
        self.assertEqual(rec.content, "Answer is here")

    def test_think_incremental(self):
        rec = QwenStreamReconstructor()
        for text in ("Let me think", " step by step"):
            rec.handle(SSEEvent(None, {"choices": [{"delta": {"role": "assistant", "content": text, "phase": "think"}}]}))
        c_diff, r_diff = rec.take_diffs()
        self.assertEqual(c_diff, "")
        self.assertEqual(r_diff, "Let me think step by step")
        self.assertEqual(rec.reasoning, "Let me think step by step")

    def test_diffs(self):
        rec = QwenStreamReconstructor()
        rec.handle(SSEEvent(None, {"choices": [{"delta": {"content": "Hel", "phase": "answer"}}]}))
        c, _ = rec.take_diffs()
        self.assertEqual(c, "Hel")
        rec.handle(SSEEvent(None, {"choices": [{"delta": {"content": "lo", "phase": "answer"}}]}))
        c, _ = rec.take_diffs()
        self.assertEqual(c, "lo")
        self.assertEqual(rec.content, "Hello")

    def test_error_capture(self):
        rec = QwenStreamReconstructor()
        rec.handle(SSEEvent(None, {"error": {"code": "Too_Many_Requests", "details": "slow down"}}))
        assert rec.error is not None
        self.assertEqual(rec.error["code"], "Too_Many_Requests")
        self.assertTrue(rec.error["details"])

    def test_done_and_stopped(self):
        rec = QwenStreamReconstructor()
        rec.handle(SSEEvent(None, {"done": True}))
        self.assertTrue(rec.finished)
        rec2 = QwenStreamReconstructor()
        rec2.handle(SSEEvent(None, {"response.stopped": {"response_id": "r1"}}))
        self.assertTrue(rec2.finished)

    def test_usage(self):
        rec = QwenStreamReconstructor()
        rec.handle(
            SSEEvent(
                None,
                {
                    "choices": [{"delta": {"content": "x", "phase": "answer"}}],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                },
            )
        )
        self.assertEqual(rec.usage_tokens["prompt_tokens"], 10)
        self.assertEqual(rec.usage_tokens["completion_tokens"], 5)
        self.assertEqual(rec.usage_tokens["total_tokens"], 15)


if __name__ == "__main__":
    unittest.main()
