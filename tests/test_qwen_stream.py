from danyapi.deepseek.sse import SSEEvent
from danyapi.deepseek.stream import IncrementalSSE
from danyapi.qwen.stream import QwenStreamReconstructor


def test_incremental_parse_real_stream():
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
    assert rec.response_id == "r1"
    assert rec.content == "Hello"
    assert rec.finished


def test_thinking_summary():
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
    assert rec.reasoning == "First thought\n\nSecond thought"
    assert rec.content == "Answer is here"


def test_think_incremental():
    rec = QwenStreamReconstructor()
    for text in ("Let me think", " step by step"):
        rec.handle(SSEEvent(None, {"choices": [{"delta": {"role": "assistant", "content": text, "phase": "think"}}]}))
    c_diff, r_diff = rec.take_diffs()
    assert c_diff == ""
    assert r_diff == "Let me think step by step"
    assert rec.reasoning == "Let me think step by step"


def test_diffs():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"content": "Hel", "phase": "answer"}}]}))
    c, _ = rec.take_diffs()
    assert c == "Hel"
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"content": "lo", "phase": "answer"}}]}))
    c, _ = rec.take_diffs()
    assert c == "lo"
    assert rec.content == "Hello"


def test_error_capture():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"error": {"code": "Too_Many_Requests", "details": "slow down"}}))
    assert rec.error is not None
    assert rec.error["code"] == "Too_Many_Requests"
    assert rec.error["details"]


def test_done_and_stopped():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"done": True}))
    assert rec.finished
    rec2 = QwenStreamReconstructor()
    rec2.handle(SSEEvent(None, {"response.stopped": {"response_id": "r1"}}))
    assert rec2.finished


def test_usage():
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
    assert rec.usage_tokens == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_handle_non_dict_data():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, "text"))
    assert rec.content == ""


def test_handle_error_non_dict():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"error": "plain"}))
    assert rec.error == {"code": "Internal_Server_Error", "details": "plain"}


def test_handle_choices_empty():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": []}))
    assert rec.content == ""


def test_handle_delta_not_dict():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": [{"delta": "x"}]}))
    assert rec.content == ""


def test_handle_unknown_phase():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"phase": "other", "content": "x"}}]}))
    assert rec.content == ""


def test_think_empty_text():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"content": "", "phase": "think"}}]}))
    assert rec.reasoning == ""


def test_summary_extra_not_dict():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"phase": "thinking_summary", "extra": "x"}}]}))
    assert rec.reasoning == ""


def test_summary_thought_not_dict():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"phase": "thinking_summary", "extra": {"summary_thought": "x"}}}]}))
    assert rec.reasoning == ""


def test_summary_content_not_list():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"phase": "thinking_summary", "extra": {"summary_thought": {"content": "x"}}}}]}))
    assert rec.reasoning == ""


def test_summary_content_empty():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"phase": "thinking_summary", "extra": {"summary_thought": {"content": []}}}}]}))
    assert rec.reasoning == ""
