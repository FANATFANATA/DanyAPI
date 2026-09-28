import json

import pytest

from danyapi.alice import api as alice_api
from danyapi.alice.client import (
    AliceClient,
    AliceError,
    AliceStream,
    fold_messages,
    is_placeholder,
    looks_like_refusal,
    trim_prompt,
)
from danyapi.api.schemas import ChatMessage


class _Account:
    def __init__(self, client):
        import asyncio as _asyncio

        self.client = client
        self.sem = _asyncio.Semaphore(1)
        self.index = 0
        self.broken = False
        self.broken_at = None

    @property
    def label(self) -> str:
        return "alice-acct#0"


def test_fold_messages_labels_roles():
    text = fold_messages(
        [
            ChatMessage(role="system", content="be brief"),
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]
    )
    assert text.splitlines() == ["System: be brief", "User: hi", "Assistant: hello"]


def test_fold_messages_maps_developer_to_system():
    assert fold_messages([ChatMessage(role="developer", content="rules")]) == "System: rules"


def test_fold_messages_skips_empty_and_defaults():
    assert fold_messages([]) == "Hello"
    assert fold_messages([ChatMessage(role="user", content="   ")]) == "Hello"


def test_fold_messages_handles_content_parts():
    content = [{"type": "text", "text": "part one"}]
    assert "part one" in fold_messages([ChatMessage(role="user", content=content)])


def test_trim_prompt_keeps_tail():
    from danyapi.alice.client import HARD_MAX_PROMPT

    text = "a" * (HARD_MAX_PROMPT + 100) + "TAIL"
    trimmed = trim_prompt(text)
    assert len(trimmed) == HARD_MAX_PROMPT
    assert trimmed.endswith("TAIL")


def test_is_placeholder_detects_waiting_text():
    assert is_placeholder("Одну секунду...") is True
    assert is_placeholder("одну секунду") is True
    assert is_placeholder("391") is False
    assert is_placeholder("") is False


def test_looks_like_refusal_detects_canned_lines():
    assert looks_like_refusal("На этом устройстве я не могу с Вами познакомиться.") is True
    assert looks_like_refusal("391") is False


def test_prompt_message_uses_uuid4_and_empty_dialog_id():
    client = AliceClient()
    message, request_id = client._prompt_message("hi")
    payload = message["event"]["payload"]
    assert payload["header"]["dialog_id"] == ""
    assert payload["header"]["request_id"] == request_id
    assert payload["header"]["prev_req_id"] is None
    assert payload["request"]["event"] == {"type": "text_input", "text": "hi"}
    assert payload["request"]["voice_session"] is False
    assert payload["application"]["app_id"] == "ru.yandex.web.desktop"
    assert len(payload["application"]["uuid"]) == 36


def test_continuation_message_shape():
    client = AliceClient()
    _message, request_id = client._prompt_message("hi")
    cont, cont_id = client._continuation_message()
    event = cont["event"]["payload"]["request"]["event"]
    assert event["type"] == "server_action"
    assert event["name"] == "@@mm_stack_engine_get_next"
    assert event["payload"]["stack_session_id"] == request_id
    assert event["payload"]["@scenario_name"] == "Dialogovo"
    assert cont["event"]["payload"]["header"]["prev_req_id"] == request_id
    assert cont_id != request_id


def test_extract_reads_card_text():
    client = AliceClient()
    stream = AliceStream()
    directive = {"payload": {"response": {"card": {"type": "simple_text", "text": "привет"}}}}
    client._extract(directive, stream)
    assert stream.content == "привет"
    assert stream.done is True


def test_extract_reads_directive_text_and_version():
    client = AliceClient()
    stream = AliceStream()
    directive = {
        "payload": {
            "version": "tags/releases/alice/megamind/stable-1@1",
            "response": {"directives": [{"name": "print_text", "payload": {"text": "answer", "is_end": True}}]},
        }
    }
    client._extract(directive, stream)
    assert stream.content == "answer"
    assert stream.version == "tags/releases/alice/megamind/stable-1@1"


def test_extract_reads_chat_dialog_update():
    client = AliceClient()
    stream = AliceStream()
    directive = {
        "payload": {
            "response": {
                "chat_dialog_update": [
                    {"add_message_request": {"messages": [{"content": {"plain_response_text": "из апдейта"}}]}}
                ]
            }
        }
    }
    client._extract(directive, stream)
    assert stream.content == "из апдейта"


def test_extract_treats_placeholder_as_not_done():
    client = AliceClient()
    stream = AliceStream()
    directive = {"payload": {"response": {"directives": [{"payload": {"text": "Одну секунду..."}}]}}}
    client._extract(directive, stream)
    assert stream.content == ""
    assert stream.placeholder == "Одну секунду..."
    assert stream.done is False


@pytest.mark.asyncio
async def test_pong_payload_shape():
    client = AliceClient()
    sent: list[dict] = []

    async def _capture(msg):
        sent.append(msg)

    client._send = _capture
    await client._send({"event": {"header": {"namespace": "System", "name": "Pong", "refMessageId": "x"}, "payload": {}}})
    assert sent[0]["event"]["header"]["refMessageId"] == "x"
    assert sent[0]["event"]["payload"] == {}


def test_error_codes_are_classified():
    from danyapi.alice.client import RETRYABLE_ERRORS, TERMINAL_ERRORS

    assert 1006 in RETRYABLE_ERRORS
    assert 1011 not in RETRYABLE_ERRORS
    assert 1011 in TERMINAL_ERRORS


@pytest.mark.asyncio
async def test_collect_non_stream_success(monkeypatch):
    class _Client(AliceClient):
        def __init__(self):
            super().__init__()
            self.closed = False

        async def ask(self, prompt):
            stream = AliceStream()
            stream.content = "391"
            stream.version = "v1"
            return stream

        async def aclose(self):
            self.closed = True

    account = _Account(_Client())
    result = await alice_api.collect_non_stream(
        account,
        messages=[ChatMessage(role="user", content="17*23?")],
        model="alice",
    )
    assert result["choices"][0]["message"]["content"] == "391"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] > 0
    assert result["model"] == "alice"


@pytest.mark.asyncio
async def test_collect_non_stream_maps_error(monkeypatch):
    from fastapi import HTTPException

    class _Client(AliceClient):
        async def ask(self, prompt):
            raise AliceError(1006, "closed", retryable=True)

    account = _Account(_Client())
    with pytest.raises(HTTPException) as excinfo:
        await alice_api.collect_non_stream(account, messages=[ChatMessage(role="user", content="hi")], model="alice")
    assert excinfo.value.status_code == 502


@pytest.mark.asyncio
async def test_stream_openai_emits_role_content_finish_and_done():
    class _Client(AliceClient):
        async def ask(self, prompt):
            stream = AliceStream()
            stream.content = "привет"
            return stream

    account = _Account(_Client())
    lines = [line async for line in alice_api.stream_openai(account, messages=[ChatMessage(role="user", content="hi")], model="alice")]
    assert lines[-1] == "data: [DONE]\n\n"
    payloads = [json.loads(line[6:]) for line in lines if line.startswith("data: ") and line[6:].strip() != "[DONE]"]
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant", "content": "привет"}
    assert payloads[1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_stream_openai_reports_error_in_band():
    class _Client(AliceClient):
        async def ask(self, prompt):
            raise AliceError(1006, "closed", retryable=True)

    account = _Account(_Client())
    lines = [line async for line in alice_api.stream_openai(account, messages=[ChatMessage(role="user", content="hi")], model="alice")]
    joined = "".join(lines)
    assert "Alice error" in joined
    assert "[DONE]" in joined


@pytest.mark.asyncio
async def test_ask_retries_then_gives_up():
    attempts: list[int] = []

    class _Client(AliceClient):
        async def ask(self, prompt):
            attempts.append(1)
            raise AliceError(1006, "closed", retryable=True)

        async def aclose(self):
            return None

    account = _Account(_Client())
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        await alice_api.collect_non_stream(account, messages=[ChatMessage(role="user", content="hi")], model="alice")
    assert len(attempts) == alice_api.MAX_RETRIES + 1
