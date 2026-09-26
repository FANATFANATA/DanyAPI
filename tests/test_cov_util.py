import runpy
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from danyapi.deepseek.client import DeepSeekClient, DeepSeekError
from danyapi.qwen.client import QwenClient
from danyapi.qwen.stream import (
    QwenStreamReconstructor,
    _extract_image_urls,
    _incomplete_tail,
    _summary_text,
    _trailing_open_url,
)
from danyapi.sseutil import (
    IncrementalSSE,
    MessageReconstructor,
    SSEEvent,
    _set_path,
    _touches_aggregated_fragment,
)


def _ds_resp(payload=None, status=200, text=""):
    resp = SimpleNamespace(status_code=status, text=text)
    if payload is not None:
        resp.json = MagicMock(return_value=payload)
    else:
        resp.json = MagicMock(side_effect=ValueError("no json"))
    resp.raise_for_status = MagicMock()
    return resp


def _ds_client():
    client = DeepSeekClient(token="tok")
    client.http = MagicMock()
    return client


def _fast_rec(frags, idx, content="", reasoning=""):
    rec = MessageReconstructor()
    rec.message = {"fragments": frags}
    rec._agg_fragments = frags
    rec._frag_idx = idx
    rec._content = content
    rec._reasoning = reasoning
    return rec


def test_incremental_feed_converts_bytes_buffer():
    inc = IncrementalSSE()
    inc._buffer = b'data: {"x": 1}\n\n'
    events = list(inc.feed(b""))
    assert len(events) == 1
    assert events[0].data == {"x": 1}


def test_set_path_list_final_non_int_key():
    target = {"a": {"b": [1, 2]}}
    _set_path(target, ["a", "b", "zz"], 9)
    assert target == {"a": {"b": [1, 2]}}


def test_append_to_list_with_non_int_key():
    rec = MessageReconstructor()
    rec.message = {"items": ["a"]}
    rec.handle(SSEEvent(None, {"p": "response/items/zz", "o": "APPEND", "v": "x"}))
    assert rec.message == {"items": ["a"]}


def test_touches_batch_with_non_dict_sub():
    assert _touches_aggregated_fragment("BATCH", "response", [42, "x"], 2) is False


def test_touches_non_int_fragment_index():
    assert _touches_aggregated_fragment("SET", "response/fragments/xx/content", 1, 5) is False


def test_handle_batch_non_dict_sub_hits_touches():
    rec = MessageReconstructor()
    rec.message = {"fragments": [{"type": "RESPONSE", "content": "a"}]}
    rec._frag_idx = 2
    rec.handle(SSEEvent(None, {"o": "BATCH", "v": [42]}))
    assert rec.message["fragments"][0]["content"] == "a"


def test_handle_touches_invalid_fragment_index():
    rec = MessageReconstructor()
    rec._frag_idx = 1
    rec.handle(SSEEvent(None, {"o": "SET", "p": "response/fragments/xx/content", "v": "x"}))
    assert rec.message == {"fragments": {"xx": {"content": "x"}}}


def test_handle_fast_append_marks_clean():
    frags = [{"type": "RESPONSE", "content": "a"}]
    rec = _fast_rec(frags, 1, content="a")
    rec.handle(SSEEvent(None, {"o": "APPEND", "p": "response/fragments/0/content", "v": "b"}))
    assert rec._aggregate_dirty is False
    assert rec.content == "ab"


def test_fast_append_tail_frag_idx_mismatch():
    frags = [{"type": "RESPONSE", "content": "a"}]
    rec = _fast_rec(frags, 0)
    assert rec._fast_append_tail("APPEND", "response/fragments/0/content", "b") is False


def test_fast_append_tail_wrong_path_length():
    frags = [{"type": "RESPONSE", "content": "a"}]
    rec = _fast_rec(frags, 1)
    assert rec._fast_append_tail("APPEND", "response/fragments/0", "b") is False


def test_fast_append_tail_non_int_index():
    frags = [{"type": "RESPONSE", "content": "a"}]
    rec = _fast_rec(frags, 1)
    assert rec._fast_append_tail("APPEND", "response/fragments/xx/content", "b") is False


def test_fast_append_tail_non_dict_tail():
    frags = [42]
    rec = _fast_rec(frags, 1)
    assert rec._fast_append_tail("APPEND", "response/fragments/0/content", "b") is False


def test_fast_append_tail_response_appends_content():
    frags = [{"type": "RESPONSE", "content": "a"}]
    rec = _fast_rec(frags, 1)
    assert rec._fast_append_tail("APPEND", "response/fragments/0/content", "b") is True
    assert rec._content == "b"


def test_fast_append_tail_think_appends_reasoning():
    frags = [{"type": "THINK", "content": "a"}]
    rec = _fast_rec(frags, 1)
    assert rec._fast_append_tail("APPEND", "response/fragments/0/content", "b") is True
    assert rec._reasoning == "b"


def test_fast_append_tail_unknown_type_returns_false():
    frags = [{"type": "OTHER", "content": "a"}]
    rec = _fast_rec(frags, 1)
    assert rec._fast_append_tail("APPEND", "response/fragments/0/content", "b") is False


def test_aggregates_incremental_think():
    frags = [{"type": "THINK", "content": "why"}]
    rec = MessageReconstructor()
    rec.message = {"fragments": frags}
    rec._agg_fragments = frags
    rec._frag_idx = 0
    rec._aggregate_dirty = True
    assert rec.reasoning == "why"


def test_main_exits_on_old_python(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 9))
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("danyapi.__main__", run_name="__main__")
    assert exc.value.code == 1


async def test_deepseek_post_invalid_json():
    client = _ds_client()
    client.http.post = AsyncMock(return_value=_ds_resp())
    with pytest.raises(DeepSeekError) as exc:
        await client._post("/api/x")
    assert exc.value.biz_code == -1


async def test_deepseek_check_auth_invalid_json():
    client = _ds_client()
    client.http.get = AsyncMock(return_value=_ds_resp())
    assert await client.check_auth() is False


async def test_qwen_check_auth_non_dict_payload():
    client = QwenClient()
    resp = SimpleNamespace(status_code=200)
    resp.json = MagicMock(return_value=["x"])
    client.http.get = AsyncMock(return_value=resp)
    assert await client.check_auth() is False


async def test_qwen_check_auth_nested_id():
    client = QwenClient()
    resp = SimpleNamespace(status_code=200)
    resp.json = MagicMock(return_value={"data": {"id": "u1"}})
    client.http.get = AsyncMock(return_value=resp)
    assert await client.check_auth() is True


def test_trailing_open_url_returns_start():
    assert _trailing_open_url("see https://cdn.qwenlm.ai/partial") is not None


def test_extract_image_urls_strips_open_tail():
    assert _extract_image_urls("see https://cdn.qwenlm.ai/partial") == []


def test_incomplete_tail_truncates_long_window():
    assert _incomplete_tail("a" * 5000) == ""


def test_summary_text_invalid_returns_empty():
    assert _summary_text(42) == ""
    assert _summary_text({"foo": 1}) == ""


def test_collect_image_urls_empty():
    rec = QwenStreamReconstructor()
    rec._collect_image_urls("")
    assert rec.image_urls == []


def test_finalize_commits_trailing_url():
    rec = QwenStreamReconstructor()
    rec._image_scan_tail = "https://cdn.qwenlm.ai/z.png"
    rec.finalize()
    assert rec.image_urls == ["https://cdn.qwenlm.ai/z.png"]
    assert rec.has_content


def test_image_phase_extra_string_url():
    rec = QwenStreamReconstructor()
    rec.handle(SSEEvent(None, {"choices": [{"delta": {"phase": "image", "extra": {"image_url": "https://cdn.qwenlm.ai/e.png"}}}]}))
    assert rec.image_urls == ["https://cdn.qwenlm.ai/e.png"]


def test_usage_tokens_non_numeric_string():
    rec = QwenStreamReconstructor()
    rec.usage = {"input_tokens": "abc", "output_tokens": "5.5", "total_tokens": "7"}
    assert rec.usage_tokens == {"prompt_tokens": 0, "completion_tokens": 5, "total_tokens": 7}
