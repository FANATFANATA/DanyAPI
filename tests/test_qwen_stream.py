from danyapi.deepseek.sse import SSEEvent
from danyapi.deepseek.stream import IncrementalSSE
from danyapi.qwen.stream import QwenStreamReconstructor, error_code


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
    assert not rec.error


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


def test_done_and_stopped_are_ignored():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"done": True}))
    rec.handle(SSEEvent(None, {"response.stopped": {"response_id": "r1"}}))
    rec.handle(SSEEvent(None, {"response.stopped": False}))
    assert rec.content == ""
    assert not hasattr(rec, "finished")
    assert not hasattr(rec, "image_size")


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


def test_handle_choice_not_a_dict():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": ["boom"]}))
    rec.handle(SSEEvent(None, {"choices": [None]}))
    rec.handle(SSEEvent(None, {"choices": [7]}))
    assert rec.content == ""
    assert rec.reasoning == ""
    assert rec.has_content is False


def test_handle_keeps_reading_after_a_choice_that_is_not_a_dict():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": ["boom"]}))
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"phase": "answer", "content": "Hello"}}]}))
    assert rec.content == "Hello"


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


def test_image_phase_collects_markdown_url():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"content": "![cat](https://cdn.qwenlm.ai/cat.png)", "phase": "image"}}]}))
    assert rec.image_urls == ["https://cdn.qwenlm.ai/cat.png"]
    assert rec.content == "![cat](https://cdn.qwenlm.ai/cat.png)"
    assert rec.has_content


def test_image_phase_deduplicates_urls():
    rec = QwenStreamReconstructor()
    delta = {"content": "![x](https://cdn.qwenlm.ai/x.png)", "phase": "image"}
    rec.handle(SSEEvent(None, {"choices": [{"delta": dict(delta)}]}))
    rec.handle(SSEEvent(None, {"choices": [{"delta": dict(delta)}]}))
    assert rec.image_urls == ["https://cdn.qwenlm.ai/x.png"]


def test_image_phase_image_url_field():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"phase": "image", "image_url": "https://cdn.qwenlm.ai/direct.png"}}]}))
    assert rec.image_urls == ["https://cdn.qwenlm.ai/direct.png"]


def test_image_phase_extra_hw_ignored():
    rec = QwenStreamReconstructor()
    for hw in ([[1152, 2048]], [], [[0, 100]], [["a", "b"]], "bad"):
        rec.handle(SSEEvent(None, {"choices": [{"delta": {"phase": "image", "extra": {"output_image_hw": hw}}}]}))
    assert rec.image_urls == []
    assert rec.content == ""


def test_image_phase_extra_url_variants():
    rec = QwenStreamReconstructor()
    rec.handle(
        SSEEvent(
            None,
            {
                "choices": [
                    {
                        "delta": {
                            "phase": "image",
                            "extra": {
                                "image_urls": ["https://cdn.qwenlm.ai/1.png", {"url": "https://cdn.qwenlm.ai/2.png"}, "nope"],
                                "images": [{"url": "https://cdn.qwenlm.ai/3.png"}],
                            },
                        }
                    }
                ]
            },
        )
    )
    assert rec.image_urls == [
        "https://cdn.qwenlm.ai/1.png",
        "https://cdn.qwenlm.ai/2.png",
        "https://cdn.qwenlm.ai/3.png",
    ]


def test_answer_phase_collects_cdn_urls():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"content": "see https://cdn.qwenlm.ai/pic.webp here", "phase": "answer"}}]}))
    assert rec.image_urls == ["https://cdn.qwenlm.ai/pic.webp"]
    assert rec.content == "see https://cdn.qwenlm.ai/pic.webp here"


def test_error_code_variants():
    assert error_code({"code": "Busy"}) == "Busy"
    assert error_code({"code": 123}) == "123"
    assert error_code({"code": 123.0}) == "123"
    assert error_code({"code": 123.9}) == "123"
    assert error_code({"code": True}) is None
    assert error_code({"code": None}) is None
    assert error_code({"code": ["x"]}) is None
    assert error_code({}) is None
    assert error_code(None) is None


def _rec_error(error):
    rec = QwenStreamReconstructor()
    rec.error = error
    return rec


def test_numeric_upstream_code_is_preserved_instead_of_reading_as_absent():
    from danyapi.qwen import api as qwen_api

    rec = _rec_error({"code": 40014, "details": "nope"})
    code = error_code(rec.error)
    assert code == "40014"
    assert code is not None
    assert code not in (qwen_api.RETRYABLE_ERROR_CODES | qwen_api.AUTH_ERROR_CODES | qwen_api.RATE_LIMIT_ERROR_CODES)
    assert hash(code)
    assert qwen_api._error_status(code) == 502
    assert qwen_api._is_context_limit(rec) is False
    assert qwen_api._is_retryable_error(rec) is False
    assert qwen_api._error_detail(rec) == "nope"


def test_usage_tokens_defaults():
    rec = QwenStreamReconstructor()
    assert rec.usage_tokens == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_usage_tokens_parses_string_numbers():
    rec = QwenStreamReconstructor()
    rec.usage = {"input_tokens": "1234", "output_tokens": "56.0", "total_tokens": "1290.5"}
    assert rec.usage_tokens == {"prompt_tokens": 1234, "completion_tokens": 56, "total_tokens": 1290}


def test_usage_tokens_booleans_zero():
    rec = QwenStreamReconstructor()
    rec.usage = {"input_tokens": True, "output_tokens": False, "total_tokens": True}
    assert rec.usage_tokens == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_usage_tokens_ignores_invalid_values():
    rec = QwenStreamReconstructor()
    rec.usage = {"input_tokens": None, "output_tokens": "5", "total_tokens": [15]}
    assert rec.usage_tokens == {"prompt_tokens": 0, "completion_tokens": 5, "total_tokens": 0}
    rec.usage = {"input_tokens": 10.0, "output_tokens": 5.0, "total_tokens": 15.0}
    assert rec.usage_tokens == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_image_url_split_across_answer_chunks():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"content": "see ![cat](https://cdn.qwenlm.ai/", "phase": "answer"}}]}))
    assert rec.image_urls == []
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"content": "cat.png) here", "phase": "answer"}}]}))
    assert rec.image_urls == ["https://cdn.qwenlm.ai/cat.png"]
    assert rec.content == "see ![cat](https://cdn.qwenlm.ai/cat.png) here"


def test_image_phase_url_split_across_chunks():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"content": "![img](https://cdn.qwenlm.ai/", "phase": "image"}}]}))
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"content": "img.webp)", "phase": "image"}}]}))
    assert rec.image_urls == ["https://cdn.qwenlm.ai/img.webp"]
    assert rec.content == "![img](https://cdn.qwenlm.ai/img.webp)"


def test_has_content_false_initially():
    rec = QwenStreamReconstructor()
    assert not rec.has_content
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"content": "step", "phase": "think"}}]}))
    assert rec.has_content
