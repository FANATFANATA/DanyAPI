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


class _Proc:
    def __init__(self, stdout):
        self.stdout = stdout


def test_rustup_target_reachable_present(monkeypatch):
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: _Proc("aarch64-unknown-linux-android\n"))
    assert setup.rustup_target_reachable() is True


def test_rustup_target_reachable_missing(monkeypatch):
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: _Proc("x86_64-unknown-linux-gnu\n"))
    assert setup.rustup_target_reachable() is False


def test_rustup_target_reachable_no_rustup(monkeypatch):
    def boom(*a, **k):
        raise OSError("no rustup")

    monkeypatch.setattr(setup.subprocess, "run", boom)
    assert setup.rustup_target_reachable() is True


def test_update_env_removes_duplicates(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("A=old1\nB=keep\nA=old2\n", encoding="utf-8")
    monkeypatch.setattr(setup, "ENV_FILE", env_file)
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    setup.update_env({"A": "new"})
    text = env_file.read_text(encoding="utf-8")
    assert text.count("A=") == 1
    assert "A=new" in text
    assert "B=keep" in text


def test_update_env_appends_new_key(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("A=1\n", encoding="utf-8")
    monkeypatch.setattr(setup, "ENV_FILE", env_file)
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    setup.update_env({"C": "3"})
    text = env_file.read_text(encoding="utf-8")
    assert "A=1" in text
    assert "C=3" in text
