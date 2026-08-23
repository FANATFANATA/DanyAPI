import email.message
import importlib.util
import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import patch

DOCS = Path(__file__).resolve().parents[1] / "docs"
_SPEC = importlib.util.spec_from_file_location("danyapi_setup", DOCS / "setup.py")
assert _SPEC is not None
assert _SPEC.loader is not None
setup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(setup)


class FakeResp:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    def read(self):
        return self._body.encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _patch_urlopen(status, body):
    return patch.object(setup.urllib.request, "urlopen", return_value=FakeResp(status, body))


def test_check_qwen_token_success_flag():
    with _patch_urlopen(200, json.dumps({"success": True, "id": "u1"})):
        ok, detail = setup.check_qwen_token("tok")
    assert ok is True
    assert detail == ""


def test_check_qwen_token_user_object_without_success():
    with _patch_urlopen(200, json.dumps({"id": "u1", "name": "n", "email": "a@b.c"})):
        ok, detail = setup.check_qwen_token("tok")
    assert ok is True
    assert detail == ""


def test_check_qwen_token_rejects_unknown_payload():
    with _patch_urlopen(200, json.dumps({"foo": "bar"})):
        ok, detail = setup.check_qwen_token("tok")
    assert ok is False
    assert "rejected" in detail


def test_check_qwen_token_http_401():
    error = urllib.error.HTTPError("https://chat.qwen.ai/api/v1/auths/", 401, "Unauthorized", email.message.Message(), io.BytesIO(b"denied"))
    with patch.object(setup.urllib.request, "urlopen", side_effect=error):
        ok, detail = setup.check_qwen_token("bad")
    assert ok is False
    assert "401" in detail


def test_check_qwen_token_network_error():
    with patch.object(setup.urllib.request, "urlopen", side_effect=OSError("no route")):
        ok, detail = setup.check_qwen_token("tok")
    assert ok is False
    assert "network error" in detail


def test_check_deepseek_token_ok():
    with _patch_urlopen(200, json.dumps({"code": 0})):
        ok, _ = setup.check_deepseek_token("tok")
    assert ok is True


def test_check_deepseek_token_rejected():
    with _patch_urlopen(200, json.dumps({"code": 401, "msg": "bad"})):
        ok, detail = setup.check_deepseek_token("tok")
    assert ok is False
    assert "rejected" in detail


def test_split_tokens_variants():
    assert setup.split_tokens("a, b  c,d") == ["a", "b", "c", "d"]
    assert setup.split_tokens(" , ") == []
    assert setup.split_tokens("") == []


def test_check_provider_valid_tokens():
    with (
        patch.object(setup, "check_deepseek_token", side_effect=[(True, ""), (True, "")]),
    ):
        ok, detail = setup.check_provider("DeepSeek", {"DEEPSEEK_TOKENS": "t1, t2"})
    assert ok is True
    assert detail == ""


def test_check_provider_invalid_token_fails_fast():
    calls = []

    def checker(token):
        calls.append(token)
        if token == "t1":
            return True, ""
        return False, "http 401: denied"

    with patch.object(setup, "check_deepseek_token", side_effect=checker):
        ok, detail = setup.check_provider("DeepSeek", {"DEEPSEEK_TOKENS": "t1,t2,t3"})
    assert ok is False
    assert "http 401" in detail
    assert calls == ["t1", "t2"]


def test_check_provider_empty_is_ok():
    ok, detail = setup.check_provider("Qwen", {"QWEN_TOKENS": ""})
    assert ok is True
    assert detail == ""


def test_collect_provider_returns_tokens_key_only(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "tok1,tok2")
    creds = setup.collect_provider("Qwen", {}, {})
    assert creds == {"QWEN_TOKENS": "tok1,tok2"}
