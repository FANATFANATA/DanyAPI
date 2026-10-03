from __future__ import annotations

import asyncio
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

import danyapi.api.openai as openai_mod

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request():
    from fastapi import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "root_path": "",
        "scheme": "http",
        "query_string": b"",
        "headers": [],
        "client": ("1.2.3.4", 1234),
        "server": ("test", 80),
    }
    request = Request(scope)
    request.state.byok = False
    return request


def test_an_oversized_anthropic_message_list_is_refused():
    from danyapi.api.openai import MAX_MESSAGES_PER_REQUEST

    messages = [{"role": "user", "content": "hi"} for _ in range(MAX_MESSAGES_PER_REQUEST + 1)]
    response = asyncio.run(openai_mod._anthropic_messages({"model": "deepseek-v4.1-flash", "messages": messages}, _request()))
    assert response.status_code == 400
    assert f"too many messages: max {MAX_MESSAGES_PER_REQUEST} per request" in response.body.decode()


def test_a_message_list_inside_the_cap_is_accepted(monkeypatch):
    from danyapi.api.openai import MAX_MESSAGES_PER_REQUEST

    seen: list[int] = []

    async def dispatcher(_model, _request):
        async def call(chat_req):
            seen.append(len(chat_req.messages))
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}

        return call

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", dispatcher)
    messages = [{"role": "user", "content": "hi"} for _ in range(MAX_MESSAGES_PER_REQUEST)]
    result = asyncio.run(openai_mod._anthropic_messages({"model": "deepseek-v4.1-flash", "messages": messages}, _request()))
    assert result["content"] == [{"type": "text", "text": "ok"}]
    assert seen == [MAX_MESSAGES_PER_REQUEST]


def _responses_request(**overrides):
    base = {
        "state": SimpleNamespace(byok=False),
        "headers": {},
        "input": [{"role": "user", "content": "again"}],
        "previous_response_id": "resp_prev",
        "instructions": None,
        "model": "gpt-4o",
        "max_output_tokens": None,
        "temperature": None,
        "top_p": None,
        "tool_choice": None,
        "tools": None,
        "parallel_tool_calls": None,
        "store": False,
        "metadata": None,
        "user": None,
        "text": None,
        "truncation": None,
        "reasoning": None,
        "stream": False,
        "session_id": None,
        "thinking": None,
        "search": None,
        "response_format": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_the_responses_endpoint_refuses_an_oversized_chain(monkeypatch):
    from danyapi.api.openai import MAX_MESSAGES_PER_REQUEST

    store: dict[str, dict] = {
        "resp_prev": {
            "public": {"id": "resp_prev", "instructions": "always answer in French"},
            "conversation": [{"role": "user", "content": "hi"} for _ in range(MAX_MESSAGES_PER_REQUEST)],
        }
    }
    monkeypatch.setattr(openai_mod, "_responses_store", lambda: store)

    async def run() -> None:
        with pytest.raises(HTTPException) as excinfo:
            await openai_mod.create_response(_responses_request(), _request())
        assert excinfo.value.status_code == 400
        assert f"too many messages: max {MAX_MESSAGES_PER_REQUEST} per request" in str(excinfo.value.detail)

    asyncio.run(run())


def test_a_chained_response_keeps_the_original_instructions(monkeypatch):
    store: dict[str, dict] = {
        "resp_prev": {
            "public": {"id": "resp_prev", "instructions": "always answer in French"},
            "conversation": [{"role": "user", "content": "hi"}],
        }
    }
    seen: list[list[dict]] = []
    monkeypatch.setattr(openai_mod, "_responses_store", lambda: store)

    async def dispatcher(_model, _request):
        async def call(chat_req):
            seen.append([{"role": message.role, "content": message.content} for message in chat_req.messages])
            return {"choices": [{"message": {"content": "bonjour"}, "finish_reason": "stop"}], "usage": {}}

        return call

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", dispatcher)
    result = asyncio.run(openai_mod.create_response(_responses_request(), _request()))
    assert result["instructions"] == "always answer in French"
    assert seen[0][0] == {"role": "system", "content": "always answer in French"}
    assert [message["role"] for message in seen[0]] == ["system", "user", "user"]


def test_a_new_instructions_value_still_wins_over_the_stored_one(monkeypatch):
    store: dict[str, dict] = {
        "resp_prev": {
            "public": {"id": "resp_prev", "instructions": "always answer in French"},
            "conversation": [{"role": "user", "content": "hi"}],
        }
    }
    seen: list[list[dict]] = []
    monkeypatch.setattr(openai_mod, "_responses_store", lambda: store)

    async def dispatcher(_model, _request):
        async def call(chat_req):
            seen.append([{"role": message.role, "content": message.content} for message in chat_req.messages])
            return {"choices": [{"message": {"content": "hallo"}, "finish_reason": "stop"}], "usage": {}}

        return call

    monkeypatch.setattr(openai_mod, "_chat_dispatcher", dispatcher)
    result = asyncio.run(openai_mod.create_response(_responses_request(instructions="answer in German"), _request()))
    assert result["instructions"] == "answer in German"
    assert seen[0][0] == {"role": "system", "content": "answer in German"}


def test_the_setup_wizard_keeps_the_last_line_before_appending(tmp_path):
    module = _load("setup_probe_a", REPO / "docs" / "setup.py")
    env_file = tmp_path / ".env"
    env_file.write_text("A=1\n# last note", encoding="utf-8")
    with patch.object(module, "ENV_FILE", env_file), patch.object(module, "load_defaults", lambda: {}):
        module.update_env({"A": "2", "B": "3"})
    assert env_file.read_text(encoding="utf-8") == "A=2\n# last note\nB=3\n"
    assert module.parse_env(env_file) == {"A": "2", "B": "3"}


@pytest.mark.parametrize("raw", ["quoted", 'has"quote', "has'quote", "has space ", "has#hash", "back\\slash", "a=b"])
def test_every_quirky_value_survives_the_dotenv_round_trip(raw):
    module = _load("setup_probe_b", REPO / "docs" / "setup.py")
    dotenv = pytest.importorskip("dotenv")
    line = f"K={module.quote(raw)}\n"
    assert dotenv.dotenv_values(stream=io.StringIO(line)) == {"K": raw}
    assert module.parse_env_text(line) == {"K": raw} if hasattr(module, "parse_env_text") else True


def test_the_setup_wizard_validates_gigachat_with_the_chosen_scope():
    module = _load("setup_probe_c", REPO / "docs" / "setup.py")
    seen: list[tuple[str, str | None]] = []

    def fake_status(key, scope=None):
        seen.append((key, scope))
        return 200, json.dumps({"access_token": "t", "expires_in": 3600})

    module._gigachat_token_status = fake_status
    creds = {"GIGACHAT_KEYS": "a,b", "GIGACHAT_SCOPE": "GIGACHAT_API_CORP"}
    with patch.object(module, "load_env", lambda: {}):
        module.validate_gigachat(creds, {})
    assert seen == [("a", "GIGACHAT_API_CORP"), ("b", "GIGACHAT_API_CORP")]


def test_the_gigachat_scope_is_asked_next_to_the_credentials():
    module = _load("setup_probe_d", REPO / "docs" / "setup.py")
    generic = [key for _title, fields in module.GROUPS for key, _label, _kind in fields]
    assert "GIGACHAT_SCOPE" not in generic
    constants = module.collect_gigachat.__code__.co_consts
    assert any(value == "GIGACHAT_SCOPE" for value in constants)


@pytest.mark.skipif(not (REPO / "collecter.py").is_file(), reason="collecter.py is not present")
def test_the_collecter_total_size_survives_a_file_that_vanished(tmp_path, monkeypatch):
    module = _load("collecter_probe", REPO / "collecter.py")
    gone = tmp_path / "gone.py"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "collect_files", lambda root: [gone])
    monkeypatch.setattr(module.sys, "argv", ["collecter.py"])
    assert module.main() == 0
    assert (tmp_path / "collected.xml").exists()


def test_the_start_script_refuses_an_unverifiable_tag():
    module = _load("start_probe", REPO / "docs" / "start.py")
    with (
        patch.object(module, "git_dirty_paths", lambda: []),
        patch.object(module, "git_branch", lambda: "dev"),
        patch.object(module, "run", lambda cmd: 0),
        patch.object(module, "git_remote_tag_sha", lambda tag: None),
    ):
        assert module.git_update("v1.2.3") is False
