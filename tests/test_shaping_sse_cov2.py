import asyncio
import base64 as b64
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import danyapi.api.attachments as attachments_mod
import danyapi.api.schemas as schemas_mod
import danyapi.api.shaping as shaping_mod
import danyapi.api.sse as sse_mod
from danyapi.accounts import AccountPoolBusy
from danyapi.api.core import INTERNAL_ERROR_MESSAGE
from danyapi.api.schemas import ChatMessage
from danyapi.deepseek.client import DeepSeekError


def _b64(data: bytes) -> str:
    return b64.b64encode(data).decode()


def test_allowed_roles_text_is_sorted_and_module_level():
    assert schemas_mod.ALLOWED_ROLES_TEXT == "assistant, developer, function, system, tool, user"
    assert schemas_mod.ALLOWED_ROLES == frozenset({"user", "assistant", "system", "developer", "tool", "function"})
    assert ChatMessage.validate_role.__func__.__globals__["ALLOWED_ROLES"] is schemas_mod.ALLOWED_ROLES


def test_invalid_role_error_text_is_exact_and_deterministic():
    texts = set()
    for _ in range(5):
        with pytest.raises(ValidationError) as excinfo:
            ChatMessage(role="wizard")
        texts.add(str(excinfo.value))
    assert len(texts) == 1
    assert "Invalid role: wizard. Allowed roles: assistant, developer, function, system, tool, user" in texts.pop()


def test_valid_roles_are_accepted_verbatim():
    for role in sorted(schemas_mod.ALLOWED_ROLES):
        assert ChatMessage(role=role).role == role


def test_tool_calls_none_passes_through():
    assert ChatMessage(role="user", content="x", tool_calls=None).tool_calls is None


def test_tool_calls_must_be_a_list():
    with pytest.raises(ValidationError) as excinfo:
        ChatMessage(role="user", tool_calls={"function": {"name": "f"}})
    assert "tool_calls must be a list" in str(excinfo.value)


def test_tool_call_items_must_be_dicts():
    with pytest.raises(ValidationError) as excinfo:
        ChatMessage(role="user", tool_calls=["get_weather"])
    assert "Each tool_call must be a dictionary" in str(excinfo.value)


def test_tool_call_needs_function_or_name():
    with pytest.raises(ValidationError) as excinfo:
        ChatMessage(role="user", tool_calls=[{"arguments": "{}"}])
    assert "Each tool_call must contain 'function' or 'name'" in str(excinfo.value)


def test_tool_calls_with_function_and_name_are_kept_verbatim():
    calls = [{"function": {"name": "f", "arguments": "{}"}}, {"name": "g"}]
    message = ChatMessage(role="assistant", tool_calls=calls)
    assert message.tool_calls == calls


def test_max_calls_collapses_to_one_when_parallel_is_disabled():
    assert shaping_mod._max_calls(False) == 1
    assert shaping_mod._max_calls(True) is None
    assert shaping_mod._max_calls(None) is None


def test_include_usage_requires_a_stream_options_dict():
    assert shaping_mod._include_usage(SimpleNamespace(stream_options=None)) is False
    assert shaping_mod._include_usage(SimpleNamespace(stream_options={"include_usage": True})) is True
    assert shaping_mod._include_usage(SimpleNamespace(stream_options={"include_usage": 0})) is False


def test_bounded_choices_clamps_to_max_stream_choices():
    assert shaping_mod._bounded_choices(20) == shaping_mod.MAX_STREAM_CHOICES
    assert shaping_mod.MAX_STREAM_CHOICES == 8
    assert shaping_mod._bounded_choices(3) == 3
    assert shaping_mod._bounded_choices(0) == 1
    assert shaping_mod._bounded_choices(None) == 1
    assert shaping_mod._bounded_choices("2") == 1


def test_apply_limits_reports_length_when_trimmed():
    text, finish = shaping_mod._apply_limits("a b c d e f g h i j k l m n o p", 2, None)
    assert finish == "length"
    assert text == shaping_mod.trim_to_tokens("a b c d e f g h i j k l m n o p", 2)


def test_apply_limits_reports_stop_when_no_stop_marker_and_no_trim():
    assert shaping_mod._apply_limits("hello", 100, None) == ("hello", "stop")
    assert shaping_mod._apply_limits("", None, "END") == ("", "stop")


def test_safe_int_rejects_non_numeric_payloads():
    assert shaping_mod._safe_int("10") == 0
    assert shaping_mod._safe_int(float("inf")) == 0
    assert shaping_mod._safe_int(float("nan")) == 0
    assert shaping_mod._safe_int(None) == 0
    assert shaping_mod._safe_int({"n": 1}) == 0
    assert shaping_mod._safe_int([1]) == 0
    assert shaping_mod._safe_int(True) == 0
    assert shaping_mod._safe_int(12) == 12
    assert shaping_mod._safe_int(12.9) == 12


def test_deepseek_usage_total_never_raises_for_bad_types():
    assert shaping_mod._deepseek_usage("x") == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert shaping_mod._deepseek_usage(4.7)["total_tokens"] == 4
    assert shaping_mod._deepseek_usage(True)["total_tokens"] == 0
    assert shaping_mod._deepseek_usage(None)["total_tokens"] == 0
    assert shaping_mod._deepseek_usage({"total": 3})["total_tokens"] == 0
    assert shaping_mod._deepseek_usage([3])["total_tokens"] == 0
    assert shaping_mod._deepseek_usage(9)["total_tokens"] == 9


def test_deepseek_usage_ignores_non_int_provider_prompt_tokens():
    usage = shaping_mod._deepseek_usage(10, "hello world", {"prompt_tokens": "4"})
    assert usage == {"prompt_tokens": 2, "completion_tokens": 8, "total_tokens": 10}
    usage = shaping_mod._deepseek_usage(10, "", {"prompt_tokens": 4})
    assert usage == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}


def test_deepseek_usage_uses_completion_text_when_total_is_zero():
    usage = shaping_mod._deepseek_usage(0, "hello world", None, "one two three four")
    assert usage["completion_tokens"] == shaping_mod.estimate_tokens("one two three four")
    assert usage["total_tokens"] == 2 + shaping_mod.estimate_tokens("one two three four")


def test_merge_usage_first_step_is_not_aliased():
    provider_usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    merged = shaping_mod._merge_usage(None, provider_usage)
    merged["total_tokens"] = 999
    assert provider_usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


def test_merge_usage_sums_numeric_total_fields_only():
    merged = shaping_mod._merge_usage(
        {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "extra": "keep"},
        {"prompt_tokens": 1.0, "completion_tokens": "ignored", "total_tokens": 4},
    )
    assert merged == {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 9, "extra": "keep"}


def test_merge_usage_keeps_previous_when_current_field_missing():
    assert shaping_mod._merge_usage({"total_tokens": 7}, {}) == {"total_tokens": 7}


def test_usage_with_details_fills_missing_detail_blocks():
    assert shaping_mod._usage_with_details({"prompt_tokens": 1}) == {
        "prompt_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    filled = shaping_mod._usage_with_details(
        {"prompt_tokens": 1, "prompt_tokens_details": {"cached_tokens": 3}},
        "why not",
    )
    assert filled["prompt_tokens_details"] == {"cached_tokens": 3}
    assert filled["completion_tokens_details"] == {"reasoning_tokens": shaping_mod.estimate_tokens("why not")}
    explicit = shaping_mod._usage_with_details(
        {"prompt_tokens": 1, "completion_tokens_details": {"reasoning_tokens": 9}},
        "why not",
        4,
    )
    assert explicit["completion_tokens_details"] == {"reasoning_tokens": 9}


def test_advance_session_usage_clamps_negative_inputs():
    session = SimpleNamespace(accumulated_tokens=-5)
    assert shaping_mod._advance_session_usage(session, -10) == 0
    assert session.accumulated_tokens == 0


def test_apply_stop_picks_earliest_marker():
    assert shaping_mod._apply_stop("hello STOP world", ["STOP", "hello"]) == ""
    assert shaping_mod._apply_stop("hello world", None) == "hello world"


def test_delta_json_without_and_with_finish():
    assert sse_mod._delta_json({"content": "x"}, None) == '{"index":0,"delta":{"content": "x"}}'
    assert sse_mod._delta_json({"content": "x"}, "stop") == '{"index":0,"delta":{"content": "x"},"finish_reason":"stop"}'


def test_chunk_id_from_line_rejects_non_str_and_bad_json():
    assert sse_mod._chunk_id_from_line(b'data: {"id": "x"}') is None
    assert sse_mod._chunk_id_from_line("data: ") is None
    assert sse_mod._chunk_id_from_line("event: ping") is None
    assert sse_mod._chunk_id_from_line("data: not-json") is None
    assert sse_mod._chunk_id_from_line("data: [DONE]") is None
    assert sse_mod._chunk_id_from_line('data: {"id": 7}') is None
    assert sse_mod._chunk_id_from_line("data: [1, 2]") is None
    assert sse_mod._chunk_id_from_line('data: {"id": "chatcmpl-abc"}') == "chatcmpl-abc"


async def _drain(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


async def test_stream_guard_hides_internal_error_and_logs_detail(caplog):
    async def gen():
        yield 'data: {"id": "chatcmpl-1"}\n\n'
        raise RuntimeError("httpx failed for https://internal.example/path C:\\secret")

    with caplog.at_level(logging.DEBUG, logger="danyapi.api"):
        lines = await _drain(sse_mod._stream_guard(gen(), "m1"))
    body = "".join(lines)
    assert "https://internal.example/path" not in body
    assert "C:\\secret" not in body
    assert "stream error" not in body
    assert INTERNAL_ERROR_MESSAGE in body
    error_chunk = json.loads(body.split("data: ")[2])
    assert error_chunk["id"] == "chatcmpl-1"
    assert error_chunk["error"] == {"message": INTERNAL_ERROR_MESSAGE}
    assert error_chunk["choices"] == [{"index": 0, "delta": {}, "finish_reason": "error"}]
    assert lines[-1] == "data: [DONE]\n\n"
    assert any("httpx failed" in record.getMessage() for record in caplog.records)
    assert any(record.exc_info for record in caplog.records)


async def test_stream_guard_surfaces_the_busy_hint():
    async def gen():
        raise AccountPoolBusy()
        yield "never"

    lines = await _drain(sse_mod._stream_guard(gen(), "m1"))
    payload = json.loads(lines[0][len("data: ") :])
    assert payload["error"] == {"message": "all accounts are busy, try again later"}
    assert payload["choices"][0]["finish_reason"] == "error"
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_guard_uses_generated_id_when_stream_has_none():
    async def gen():
        raise RuntimeError("boom")
        yield "never"

    lines = await _drain(sse_mod._stream_guard(gen(), "m1"))
    payload = json.loads(lines[0][len("data: ") :])
    assert payload["id"].startswith("chatcmpl-")
    assert payload["session_id"] is None


async def test_close_generator_skips_objects_without_aclose():
    class NoClose:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    await sse_mod._close_generator(NoClose())
    closed = []

    class Failing:
        async def aclose(self):
            closed.append(1)
            raise RuntimeError("nope")

    await sse_mod._close_generator(Failing())
    assert closed == [1]


def test_stream_error_sse_carries_finish_reasons():
    error_chunk, done = sse_mod._stream_error_sse("cid", 10, "m", "boom", session_key="s1", error_finish="length", choice_finish="length")
    assert done == "data: [DONE]\n\n"
    payload = json.loads(error_chunk[len("data: ") :])
    assert payload["error"] == {"message": "boom", "finish_reason": "length"}
    assert payload["session_id"] == "s1"
    assert payload["choices"][0]["finish_reason"] == "length"


def test_attachment_size_limits_are_aligned():
    assert attachments_mod.MAX_FILE_SIZE == attachments_mod.MAX_ATTACHMENT_TOTAL_SIZE
    assert attachments_mod.MAX_FILE_SIZE == 10 * 1024 * 1024
    assert attachments_mod.MAX_FILE_NAME_LENGTH == 128
    assert attachments_mod.MAX_ATTACHMENT_CONCURRENCY == 4


def test_safe_file_name_neutralises_control_and_path_characters():
    assert attachments_mod._safe_file_name("a\nb\r\tc.png") == "a_b_c.png"
    assert attachments_mod._safe_file_name("../../etc/passwd") == "_.._etc_passwd"
    assert attachments_mod._safe_file_name("...hidden") == "hidden"
    assert attachments_mod._safe_file_name("") == "file"
    assert attachments_mod._safe_file_name("...") == "file"
    long_name = "x" * 300 + ".png"
    cleaned = attachments_mod._safe_file_name(long_name)
    assert len(cleaned) == 128
    assert cleaned == "x" * 128


def test_generated_data_uri_name_is_sanitised():
    uri = "data:image/png\nEvil;charset=utf-8;base64," + _b64(b"png")
    req = SimpleNamespace(messages=[ChatMessage(role="user", content=[{"type": "image_url", "image_url": uri}])], files=[])
    attachments = attachments_mod._collect_attachments(req)
    assert len(attachments) == 1
    assert attachments[0].name == "image_0.png_Evil"
    assert attachments[0].content_type == "image/png\nEvil"
    assert attachments[0].data == b"png"


def test_generated_data_uri_name_falls_back_when_extension_is_empty():
    uri = "data:image/;base64," + _b64(b"png")
    req = SimpleNamespace(messages=[ChatMessage(role="user", content=[{"type": "image_url", "image_url": uri}])], files=[])
    attachments = attachments_mod._collect_attachments(req)
    assert attachments[0].name == "image_0.bin"


def test_data_uri_is_parsed_once_per_attachment(monkeypatch):
    calls: list[str] = []
    real = attachments_mod._data_uri_parts

    def counting(uri: str):
        calls.append(uri)
        return real(uri)

    monkeypatch.setattr(attachments_mod, "_data_uri_parts", counting)
    uri = "data:image/png;base64," + _b64(b"png")
    req = SimpleNamespace(messages=[ChatMessage(role="user", content=[{"type": "image_url", "image_url": uri}])], files=[])
    attachments_mod._collect_attachments(req)
    assert len(calls) == 1
    assert calls[0] == uri


def test_data_uri_missing_payload_is_400():
    with pytest.raises(HTTPException) as excinfo:
        attachments_mod._data_uri_parts("data:image/png;base64,")
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "invalid data URI: missing base64 payload"


def test_compact_data_uri_length_counts_decoded_bytes():
    assert attachments_mod._compact_data_uri_length(_b64(b"a")) == 1
    assert attachments_mod._compact_data_uri_length(_b64(b"ab")) == 2
    assert attachments_mod._compact_data_uri_length(_b64(b"abc")) == 3
    assert attachments_mod._compact_data_uri_length(_b64(b"abcd")) == 4
    assert attachments_mod._compact_data_uri_length(_b64(b"abcde")) == 5
    assert attachments_mod._compact_data_uri_length(_b64(b"abcdef")) == 6
    assert attachments_mod._raw_data_uri_length("data:image/png;base64," + _b64(b"abcde")) == 5


def test_invalid_base64_inside_file_content_is_rejected():
    req = SimpleNamespace(
        messages=[],
        files=[SimpleNamespace(name="a.txt", content="aGVsbG8*", content_type="text/plain")],
    )
    with pytest.raises(HTTPException) as excinfo:
        attachments_mod._collect_attachments(req)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "invalid base64 in file a.txt"


def test_remote_image_is_skipped_only_when_remote_is_allowed():
    uri = "https://example.com/a.png"
    req = SimpleNamespace(messages=[ChatMessage(role="user", content=[{"type": "image_url", "image_url": uri}])], files=[])
    assert attachments_mod._collect_attachments(req, allow_remote=True) == []
    with pytest.raises(HTTPException) as excinfo:
        attachments_mod._collect_attachments(req, allow_remote=False)
    assert excinfo.value.detail == "image_url must be a data URI (data:<mime>;base64,...)"


def test_validate_attachments_allows_empty_list():
    assert attachments_mod._validate_attachments([]) is None


def test_oversized_attachment_is_413():
    oversized = attachments_mod.Attachment(b"x" * (attachments_mod.MAX_FILE_SIZE + 1), "big.bin", "application/octet-stream", False)
    with pytest.raises(HTTPException) as excinfo:
        attachments_mod._validate_attachments([oversized])
    assert excinfo.value.status_code == 413
    assert excinfo.value.detail == "file big.bin exceeds 10 MB limit"


def test_upload_attachments_without_attachments_does_no_io():
    account = MagicMock()
    account.client.upload_file = AsyncMock()
    assert asyncio.run(attachments_mod._upload_attachments(account, [], "default", False)) == []
    assert account.client.upload_file.await_count == 0


async def test_pow_header_solves_respect_the_shared_semaphore(monkeypatch):
    account = MagicMock()
    live = 0
    peak = 0

    async def fake_pow(_account):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)
        live -= 1
        return {"X-DS-PoW-Response": "ok"}

    monkeypatch.setattr(attachments_mod, "_fresh_pow_upload_headers", fake_pow)
    monkeypatch.setattr(account.client, "upload_file", AsyncMock(side_effect=lambda *a, **k: _id(a[0])))
    attachments = [attachments_mod.Attachment(bytes([index]), f"f{index}.bin", "application/octet-stream", False) for index in range(20)]
    file_ids = await attachments_mod._upload_attachments(account, attachments, "default", False)
    assert file_ids == [f"id-{index}" for index in range(20)]
    assert peak == attachments_mod.MAX_ATTACHMENT_CONCURRENCY


async def test_failing_pow_header_surfaces_and_siblings_finish(monkeypatch):
    account = MagicMock()
    finished: list[int] = []

    async def fake_pow(_account):
        await asyncio.sleep(0)
        finished.append(1)
        raise RuntimeError("pow died")

    monkeypatch.setattr(attachments_mod, "_fresh_pow_upload_headers", fake_pow)
    upload = AsyncMock()
    monkeypatch.setattr(account.client, "upload_file", upload)
    attachments = [attachments_mod.Attachment(b"x", f"f{index}.bin", "application/octet-stream", False) for index in range(6)]
    with pytest.raises(RuntimeError, match="pow died"):
        await attachments_mod._upload_attachments(account, attachments, "default", False)
    assert len(finished) == 6
    assert upload.await_count == 0


async def test_upload_error_stops_siblings_and_reports_status(monkeypatch):
    account = MagicMock()
    monkeypatch.setattr(attachments_mod, "_fresh_pow_upload_headers", AsyncMock(return_value={}))
    monkeypatch.setattr(account.client, "upload_file", AsyncMock(side_effect=RuntimeError("upload died")))
    attachments = [attachments_mod.Attachment(b"x", f"f{index}.bin", "application/octet-stream", False) for index in range(4)]
    with pytest.raises(RuntimeError, match="upload died"):
        await attachments_mod._upload_attachments(account, attachments, "default", False)


def _id(data: bytes) -> dict:
    return {"id": f"id-{data[0]}"}


def test_decode_data_uri_rejects_invalid_base64():
    with pytest.raises(HTTPException) as excinfo:
        attachments_mod._decode_data_uri("image/png", "aGVsbG8*")
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "invalid base64 in image_url"


def test_split_data_uri_round_trip():
    assert attachments_mod._split_data_uri("data:image/png;base64," + _b64(b"abc")) == ("image/png", b"abc")
    assert attachments_mod._split_data_uri("data:;base64," + _b64(b"abc")) == ("application/octet-stream", b"abc")


def test_data_uri_prefix_must_be_data():
    with pytest.raises(HTTPException) as excinfo:
        attachments_mod._data_uri_parts("http://x/y.png")
    assert excinfo.value.detail == "image_url must be a data URI (data:<mime>;base64,...)"


def test_non_dict_content_items_are_skipped():
    req = SimpleNamespace(
        messages=[ChatMessage(role="user", content=[{"type": "image_url", "image_url": "data:image/png;base64,eA=="}, 42, "text"])],
        files=[],
    )
    attachments = attachments_mod._collect_attachments(req)
    assert [att.name for att in attachments] == ["image_0.png"]
    assert attachments[0].is_image is True


def test_image_url_accepts_string_and_dict_forms():
    payload = "data:image/png;base64," + _b64(b"png")
    for value in (payload, {"url": payload}):
        req = SimpleNamespace(messages=[ChatMessage(role="user", content=[{"type": "image_url", "image_url": value}])], files=[])
        assert attachments_mod._collect_attachments(req)[0].data == b"png"


def test_image_url_value_of_the_wrong_type_is_400():
    req = SimpleNamespace(messages=[ChatMessage(role="user", content=[{"type": "image_url", "image_url": 42}])], files=[])
    with pytest.raises(HTTPException) as excinfo:
        attachments_mod._collect_attachments(req)
    assert excinfo.value.detail == "invalid image_url value"


def test_data_uri_counts_against_the_same_aggregate_cap_as_files():
    half = "data:image/png;base64," + _b64(b"x" * (attachments_mod.MAX_ATTACHMENT_TOTAL_SIZE // 2))
    chunk = _b64(b"x" * (attachments_mod.MAX_ATTACHMENT_TOTAL_SIZE // 4))
    req = SimpleNamespace(
        messages=[ChatMessage(role="user", content=[{"type": "image_url", "image_url": half}])],
        files=[SimpleNamespace(name=f"part{index}.bin", content=chunk, content_type="application/octet-stream") for index in range(3)],
    )
    with pytest.raises(HTTPException) as excinfo:
        attachments_mod._collect_attachments(req)
    assert excinfo.value.status_code == 413
    assert excinfo.value.detail == "attachments too large"


def test_oversized_data_uri_alone_is_413():
    uri = "data:image/png;base64," + _b64(b"x" * (attachments_mod.MAX_ATTACHMENT_TOTAL_SIZE + 1))
    req = SimpleNamespace(messages=[ChatMessage(role="user", content=[{"type": "image_url", "image_url": uri}])], files=[])
    with pytest.raises(HTTPException) as excinfo:
        attachments_mod._collect_attachments(req)
    assert excinfo.value.status_code == 413


def test_plain_text_message_content_contributes_no_attachment():
    req = SimpleNamespace(messages=[ChatMessage(role="user", content="just text")], files=[])
    assert attachments_mod._collect_attachments(req) == []


def test_file_needs_name_and_content():
    for spec in (SimpleNamespace(name="", content="aGk=", content_type="text/plain"), SimpleNamespace(name="a.txt", content="", content_type="text/plain")):
        with pytest.raises(HTTPException) as excinfo:
            attachments_mod._collect_attachments(SimpleNamespace(messages=[], files=[spec]))
        assert excinfo.value.detail == "each file needs name and base64 content"


def test_file_content_counts_against_the_aggregate_cap():
    chunk = _b64(b"x" * (attachments_mod.MAX_ATTACHMENT_TOTAL_SIZE // 4))
    req = SimpleNamespace(
        messages=[],
        files=[SimpleNamespace(name=f"part{index}.bin", content=chunk, content_type="application/octet-stream") for index in range(5)],
    )
    with pytest.raises(HTTPException) as excinfo:
        attachments_mod._collect_attachments(req)
    assert excinfo.value.status_code == 413


def test_file_content_type_defaults_and_image_flag():
    req = SimpleNamespace(messages=[], files=[SimpleNamespace(name="a.bin", content="aGk=", content_type=None)])
    attachments = attachments_mod._collect_attachments(req)
    assert attachments[0].content_type == "application/octet-stream"
    assert attachments[0].is_image is False
    req = SimpleNamespace(messages=[], files=[SimpleNamespace(name="a.png", content="aGk=", content_type="image/png")])
    assert attachments_mod._collect_attachments(req)[0].is_image is True


def test_too_many_files_is_400():
    req = SimpleNamespace(
        messages=[],
        files=[SimpleNamespace(name=f"f{index}.txt", content="aGk=", content_type="text/plain") for index in range(attachments_mod.MAX_FILES_PER_REQUEST + 1)],
    )
    with pytest.raises(HTTPException) as excinfo:
        attachments_mod._collect_attachments(req)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "too many files: max 50 per request"


def test_too_many_image_parts_is_400_while_collecting():
    uri = "data:image/png;base64,aGk="
    parts = [{"type": "image_url", "image_url": {"url": uri}} for _ in range(attachments_mod.MAX_FILES_PER_REQUEST + 1)]
    req = SimpleNamespace(messages=[SimpleNamespace(content=parts)], files=None)
    with pytest.raises(HTTPException) as excinfo:
        attachments_mod._collect_attachments(req)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "too many files: max 50 per request"
    at_limit = SimpleNamespace(
        messages=[SimpleNamespace(content=parts[: attachments_mod.MAX_FILES_PER_REQUEST])],
        files=None,
    )
    assert len(attachments_mod._collect_attachments(at_limit)) == attachments_mod.MAX_FILES_PER_REQUEST


def test_an_oversized_file_is_rejected_before_it_is_stored():
    oversized = b64.b64encode(b"x" * (attachments_mod.MAX_FILE_SIZE // 2 + 1)).decode()
    req = SimpleNamespace(
        messages=[],
        files=[
            SimpleNamespace(name="a.bin", content=oversized, content_type="application/octet-stream"),
            SimpleNamespace(name="b.bin", content=oversized, content_type="application/octet-stream"),
        ],
    )
    with pytest.raises(HTTPException) as excinfo:
        attachments_mod._collect_attachments(req)
    assert excinfo.value.status_code == 413
    assert excinfo.value.detail == "attachments too large"


async def test_upload_deepseek_error_is_mapped_and_marks_the_account(monkeypatch):
    account = MagicMock()
    account.mark_broken = MagicMock()
    monkeypatch.setattr(attachments_mod, "_fresh_pow_upload_headers", AsyncMock(return_value={}))
    monkeypatch.setattr(account.client, "upload_file", AsyncMock(side_effect=DeepSeekError(40001, "bad token")))
    attachments = [attachments_mod.Attachment(b"x", "a.bin", "application/octet-stream", False)]
    with pytest.raises(HTTPException) as excinfo:
        await attachments_mod._upload_attachments(account, attachments, "default", False)
    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == "file upload failed: DeepSeek biz error 40001: bad token"
    account.mark_broken.assert_called_once()


async def test_upload_without_file_id_is_502(monkeypatch):
    account = MagicMock()
    monkeypatch.setattr(attachments_mod, "_fresh_pow_upload_headers", AsyncMock(return_value={}))
    monkeypatch.setattr(account.client, "upload_file", AsyncMock(return_value={}))
    attachments = [attachments_mod.Attachment(b"x", "a.bin", "application/octet-stream", False)]
    with pytest.raises(HTTPException) as excinfo:
        await attachments_mod._upload_attachments(account, attachments, "default", False)
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "file upload failed for a.bin: no file id"


def test_merge_usage_takes_the_only_side_that_has_a_number():
    assert shaping_mod._merge_usage({}, {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}) == {
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "total_tokens": 12,
    }
    assert shaping_mod._merge_usage({"total_tokens": 4}, {}) == {"total_tokens": 4}
    assert shaping_mod._merge_usage({}, {"total_tokens": 4}) == {"total_tokens": 4}
    assert shaping_mod._merge_usage({"prompt_tokens": "x"}, {"prompt_tokens": 3}) == {"prompt_tokens": 3}
    assert shaping_mod._merge_usage({"prompt_tokens": True}, {"prompt_tokens": 3}) == {"prompt_tokens": 3}
    assert shaping_mod._merge_usage({"prompt_tokens": 2}, {"prompt_tokens": None}) == {"prompt_tokens": 2}


def test_apply_limits_reports_stop_when_the_stop_sequence_truncated_the_text():
    assert shaping_mod._apply_limits("alpha beta gamma", 1, "gamma") == ("alpha beta ", "stop")
    assert shaping_mod._apply_limits("alpha beta gamma", None, "gamma") == ("alpha beta ", "stop")
    assert shaping_mod._apply_limits("alpha beta gamma", 1, None) == ("alpha", "length")


@pytest.mark.parametrize(
    "model, field",
    [
        (schemas_mod.ChatCompletionRequest, "max_tokens"),
        (schemas_mod.ChatCompletionRequest, "max_completion_tokens"),
        (schemas_mod.CompletionRequest, "max_tokens"),
        (schemas_mod.ResponsesRequest, "max_output_tokens"),
    ],
)
def test_token_budgets_reject_a_non_positive_value(model, field):
    assert model(**{field: 1}).model_dump()[field] == 1
    assert model().model_dump()[field] is None
    for value in (0, -1, -4096):
        with pytest.raises(ValidationError):
            model(**{field: value})
