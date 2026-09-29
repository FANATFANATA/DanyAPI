import asyncio
import importlib
import json
import logging
import os
import sys
import threading
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import danyapi.api.envtokens as envtokens
from danyapi import config as config_mod
from danyapi.api.core import _token_stable_id
from danyapi.api.state import app
from danyapi.config import settings

LOGGER = "danyapi.api"


class _FakeClient:
    def __init__(
        self,
        ok: bool = True,
        auth_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.ok = ok
        self.auth_error = auth_error
        self.close_error = close_error
        self.auth_calls = 0
        self.closed = 0

    async def check_auth(self) -> bool:
        self.auth_calls += 1
        if self.auth_error is not None:
            raise self.auth_error
        return self.ok

    async def aclose(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


class _FakeOs:
    O_RDONLY = os.O_RDONLY

    def __init__(self) -> None:
        self.name = "posix"
        self.opened: list[tuple[Any, int]] = []
        self.fsynced: list[int] = []
        self.closed_fds: list[int] = []

    def open(self, path: Any, flags: int) -> int:
        self.opened.append((path, flags))
        return 91

    def fsync(self, fd: int) -> None:
        self.fsynced.append(fd)

    def close(self, fd: int) -> None:
        self.closed_fds.append(fd)


class _CountingLock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._guard = threading.Lock()
        self.inside = 0
        self.max_inside = 0
        self.acquires = 0

    def __enter__(self) -> Any:
        self._lock.acquire()
        with self._guard:
            self.acquires += 1
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
        return self

    def __exit__(self, *exc: object) -> None:
        with self._guard:
            self.inside -= 1
        self._lock.release()


class _RefreshRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def __call__(self, provider: str, client: Any) -> list[dict[str, str]]:
        self.calls.append((provider, client))
        return []


class _QwenModelsRecorder:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def __call__(self, client: Any) -> list[dict[str, str]]:
        self.calls.append(client)
        return [{"id": "qwen-live", "name": "Q", "owned_by": "qwen", "model_type": "chat"}]


def _request(headers: dict[str, str] | None = None, body: bytes = b"", host: str | None = "203.0.113.5") -> Request:
    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    pairs = [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
    client: Any = (host, 54321) if host is not None else None
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/tokens",
        "query_string": b"",
        "headers": pairs,
        "client": client,
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope, receive=receive)


async def _asgi_post(headers: list[tuple[bytes, bytes]], body: bytes) -> tuple[int, Any]:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/tokens",
        "raw_path": b"/v1/tokens",
        "query_string": b"",
        "root_path": "",
        "headers": [*headers, (b"content-length", str(len(body)).encode())],
        "client": ("198.51.100.9", 51234),
        "server": ("testserver", 80),
        "app": app,
    }
    sent: list[MutableMapping[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)

    await app(scope, receive, send)
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    payload = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    return status, json.loads(payload.decode("utf-8"))


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    monkeypatch.setattr(config_mod, "_ENV_PATH", env_file)
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(settings, "deepseek_tokens", [])
    monkeypatch.setattr(settings, "qwen_tokens", [])
    monkeypatch.setattr(envtokens, "_TOKENS_LOCK", asyncio.Lock())
    saved = dict(app.state._state)
    app.state.pool = None
    app.state.qwen_pool = None
    app.state.byok = False
    yield env_file
    app.state._state.clear()
    app.state._state.update(saved)


def test_env_path_follows_config_env_path(monkeypatch, tmp_path):
    first = tmp_path / "first.env"
    second = tmp_path / "second.env"
    monkeypatch.setattr(config_mod, "ENV_PATH", first, raising=False)
    assert envtokens._env_path() == first
    monkeypatch.setattr(config_mod, "ENV_PATH", "not-a-path", raising=False)
    monkeypatch.setattr(config_mod, "_ENV_PATH", second)
    assert envtokens._env_path() == second
    monkeypatch.delattr(config_mod, "ENV_PATH")
    monkeypatch.delattr(config_mod, "_ENV_PATH")
    assert envtokens._env_path() == Path(config_mod.__file__).resolve().parents[1] / ".env"


def test_unquote_env_value_strips_matching_quotes_only():
    assert envtokens._unquote_env_value('"a,b"') == "a,b"
    assert envtokens._unquote_env_value("'a,b'") == "a,b"
    assert envtokens._unquote_env_value('""') == ""
    assert envtokens._unquote_env_value('"a') == '"a'
    assert envtokens._unquote_env_value('a"') == 'a"'
    assert envtokens._unquote_env_value("plain") == "plain"


def test_strip_inline_comment_respects_quotes_and_spacing():
    assert envtokens._strip_inline_comment("#whole line") == ""
    assert envtokens._strip_inline_comment("value # note") == "value "
    assert envtokens._strip_inline_comment("value#note") == "value#note"
    assert envtokens._strip_inline_comment('"value # note" # tail') == '"value # note" '
    assert envtokens._strip_inline_comment("'a # b'") == "'a # b'"
    assert envtokens._strip_inline_comment("plain") == "plain"
    assert envtokens._strip_inline_comment("a\t# b") == "a\t"


def test_fallback_env_values_scanner():
    text = "\n".join(
        [
            "# leading comment",
            "",
            "OTHER=ignored",
            "  export DEEPSEEK_TOKENS = a,b ",
            "QWEN_TOKENS='q1' # trailing",
            "export QWEN_TOKENS_EXTRA=not-a-credential",
            "no-equals-here",
            'DEEPSEEK_OTHER="quoted # kept"',
        ]
    )
    assert envtokens._fallback_env_values(text) == {"DEEPSEEK_TOKENS": "a,b", "QWEN_TOKENS": "q1"}


def test_split_token_list_handles_escapes_like_the_config_parser():
    cases = [
        "",
        ",",
        " , , ",
        "a",
        "a,b",
        "a,,b",
        "  a  ,  b  ",
        "a\\,b",
        "a\\qb",
        "a\\",
        "\\\\",
        "a\\,b,c",
        "\\,",
        "one,two,three",
    ]
    for raw in cases:
        assert envtokens._split_token_list(raw) == config_mod._split_env_list(raw), raw
    assert envtokens._split_token_list("a,b") == ["a", "b"]
    assert envtokens._split_token_list("a\\,b") == ["a,b"]
    assert envtokens._split_token_list("a\\qb") == ["a\\qb"]
    assert envtokens._split_token_list("a\\") == ["a\\"]
    assert envtokens._split_token_list("a,,b") == ["a", "b"]
    assert envtokens._split_token_list(",") == []


def test_read_env_values_returns_empty_for_a_missing_file(_isolated):
    assert envtokens._read_env_values(_isolated) == {}


def test_read_env_values_uses_python_dotenv_semantics(_isolated):
    _isolated.write_text(
        "\n".join(
            [
                'DEEPSEEK_TOKENS="quoted,value"',
                "export QWEN_TOKENS='single'",
                "TRAILING=value # note",
                'HASH="keep # this"',
                "EMPTY=",
            ]
        ),
        encoding="utf-8",
    )
    assert envtokens._read_env_values(_isolated) == {
        "DEEPSEEK_TOKENS": "quoted,value",
        "QWEN_TOKENS": "single",
        "TRAILING": "value",
        "HASH": "keep # this",
        "EMPTY": "",
    }


def test_read_env_values_falls_back_to_the_builtin_scanner(monkeypatch, _isolated):
    monkeypatch.setattr(envtokens, "_dotenv_values", None)
    _isolated.write_text('export DEEPSEEK_TOKENS = a,b\nQWEN_TOKENS="c" # note\n', encoding="utf-8")
    assert envtokens._read_env_values(_isolated) == {"DEEPSEEK_TOKENS": "a,b", "QWEN_TOKENS": "c"}


def test_read_env_values_reraises_an_http_exception(monkeypatch, _isolated):
    _isolated.write_text("DEEPSEEK_TOKENS=a\n", encoding="utf-8")

    def boom(_path: Path) -> dict[str, str]:
        raise HTTPException(418, "teapot")

    monkeypatch.setattr(envtokens, "_dotenv_values", boom)
    with pytest.raises(HTTPException) as excinfo:
        envtokens._read_env_values(_isolated)
    assert excinfo.value.status_code == 418
    assert excinfo.value.detail == "teapot"


def test_read_env_values_turns_a_directory_into_a_500(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(envtokens, "_dotenv_values", None)
    env_dir = tmp_path / "envdir"
    env_dir.mkdir()
    with caplog.at_level(logging.WARNING, logger=LOGGER), pytest.raises(HTTPException) as excinfo:
        envtokens._read_env_values(env_dir)
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "cannot read the credentials file"
    assert "cannot read the credentials file:" in caplog.text


def test_read_env_values_turns_invalid_utf8_into_a_500(monkeypatch, _isolated, caplog):
    monkeypatch.setattr(envtokens, "_dotenv_values", None)
    _isolated.write_bytes(b"DEEPSEEK_TOKENS=\xff\xfe\n")
    with caplog.at_level(logging.WARNING, logger=LOGGER), pytest.raises(HTTPException) as excinfo:
        envtokens._read_env_values(_isolated)
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "cannot read the credentials file"
    assert "codec can't decode" in caplog.text


def test_fsync_dir_opens_and_fsyncs_the_directory_on_posix(monkeypatch, tmp_path):
    fake = _FakeOs()
    monkeypatch.setattr(envtokens, "os", fake)
    envtokens._fsync_dir(tmp_path)
    assert fake.opened == [(tmp_path, os.O_RDONLY)]
    assert fake.fsynced == [91]
    assert fake.closed_fds == [91]


def test_fsync_dir_is_a_noop_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(envtokens.os, "name", "nt")
    assert envtokens._fsync_dir(tmp_path) is None


def test_atomic_write_text_fsyncs_the_temp_file_and_the_directory(monkeypatch, tmp_path):
    target = tmp_path / "out.env"
    fsynced: list[int] = []
    dir_syncs: list[Path] = []

    def record_fsync(fd: int) -> None:
        fsynced.append(fd)

    def record_dir(path: Path) -> None:
        dir_syncs.append(path)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(envtokens, "_fsync_dir", record_dir)
    envtokens._atomic_write_text(target, "hello\n")
    assert target.read_text(encoding="utf-8") == "hello\n"
    assert len(fsynced) == 1
    assert dir_syncs == [tmp_path]
    assert [p.name for p in tmp_path.iterdir()] == ["out.env"]


def test_atomic_write_text_serialises_concurrent_writers(monkeypatch, tmp_path):
    target = tmp_path / "out.env"
    lock = _CountingLock()
    monkeypatch.setattr(envtokens, "_ENV_WRITE_LOCK", lock)
    payloads = [f"writer-{index}\n" for index in range(8)]
    start = threading.Barrier(len(payloads))
    seen: list[str] = []

    def worker(payload: str) -> None:
        start.wait()
        for _ in range(15):
            envtokens._atomic_write_text(target, payload)
        seen.append(payload)

    threads = [threading.Thread(target=worker, args=(payload,)) for payload in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert lock.acquires == 8 * 15
    assert lock.max_inside == 1
    assert sorted(seen) == sorted(payloads)
    assert target.read_text(encoding="utf-8") in payloads
    assert [p.name for p in tmp_path.iterdir()] == ["out.env"]


def test_atomic_write_text_removes_the_temp_file_when_replace_fails(monkeypatch, tmp_path):
    target = tmp_path / "out.env"

    def boom(_src: Any, _dst: Any) -> None:
        raise OSError("replace refused")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError) as excinfo:
        envtokens._atomic_write_text(target, "hello")
    assert str(excinfo.value) == "replace refused"
    assert list(tmp_path.iterdir()) == []
    assert not target.exists()


async def test_read_env_tokens_sync_and_async(_isolated):
    assert envtokens._read_env_tokens_sync() == ([], [])
    _isolated.write_text("DEEPSEEK_TOKENS=a,b\nQWEN_TOKENS=c\n", encoding="utf-8")
    assert envtokens._read_env_tokens_sync() == (["a", "b"], ["c"])
    assert await envtokens._read_env_tokens() == (["a", "b"], ["c"])


def test_write_env_tokens_rewrites_every_form_and_drops_duplicates(_isolated):
    _isolated.write_text(
        "\n".join(
            [
                "# comment",
                "DEEPSEEK_TOKENS=old",
                "   QWEN_TOKENS = oldq",
                "export DEEPSEEK_TOKENS = duplicate",
                "OTHER=1",
                "DEEPSEEK_TOKENS_EXTRA=untouched",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    envtokens._write_env_tokens_sync(["a", "b"], ["c"])
    assert _isolated.read_text(encoding="utf-8") == (
        "\n".join(
            [
                "# comment",
                "DEEPSEEK_TOKENS=a,b",
                "   QWEN_TOKENS = c",
                "OTHER=1",
                "DEEPSEEK_TOKENS_EXTRA=untouched",
            ]
        )
        + "\n"
    )
    assert _isolated.read_text(encoding="utf-8").count("DEEPSEEK_TOKENS=") == 1


def test_write_env_tokens_appends_missing_names(_isolated):
    envtokens._write_env_tokens_sync(["a"], ["b"])
    assert _isolated.read_text(encoding="utf-8") == "DEEPSEEK_TOKENS=a\nQWEN_TOKENS=b\n"


def test_write_env_tokens_preserves_existing_lines_without_a_match(_isolated):
    _isolated.write_text("KEEP=1\nDEEPSEEK_TOKENS=old\n", encoding="utf-8")
    envtokens._write_env_tokens_sync([], ["b"])
    assert _isolated.read_text(encoding="utf-8") == "KEEP=1\nDEEPSEEK_TOKENS=\nQWEN_TOKENS=b\n"


def test_write_env_tokens_turns_an_unreadable_file_into_a_500(monkeypatch, tmp_path, caplog):
    env_dir = tmp_path / "envdir"
    env_dir.mkdir()
    monkeypatch.setattr(config_mod, "_ENV_PATH", env_dir)
    with caplog.at_level(logging.WARNING, logger=LOGGER), pytest.raises(HTTPException) as excinfo:
        envtokens._write_env_tokens_sync(["a"], ["b"])
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "cannot read the credentials file"
    assert "cannot read the credentials file before writing" in caplog.text


def test_write_env_tokens_turns_a_failed_write_into_a_500(monkeypatch, _isolated, caplog):
    def boom(_target: Path, _text: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(envtokens, "_atomic_write_text", boom)
    with caplog.at_level(logging.WARNING, logger=LOGGER), pytest.raises(HTTPException) as excinfo:
        envtokens._write_env_tokens_sync(["a"], ["b"])
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "cannot write the credentials file"
    assert "disk full" in caplog.text


def test_write_env_tokens_reports_a_missing_parent_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(config_mod, "_ENV_PATH", tmp_path / "missing" / ".env")
    with pytest.raises(HTTPException) as excinfo:
        envtokens._write_env_tokens_sync(["a"], ["b"])
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "cannot write the credentials file"


async def test_write_env_tokens_async(_isolated):
    await envtokens._write_env_tokens(["x", "y"], ["z"])
    assert _isolated.read_text(encoding="utf-8") == "DEEPSEEK_TOKENS=x,y\nQWEN_TOKENS=z\n"
    assert await envtokens._read_env_tokens() == (["x", "y"], ["z"])


def test_validate_token_accepts_a_plain_token():
    assert envtokens._validate_token("sk-abc_123", "deepseek_tokens") == "sk-abc_123"
    at_limit = "a" * envtokens.MAX_TOKEN_LENGTH
    assert envtokens._validate_token(at_limit, "deepseek_tokens") == at_limit


def test_validate_token_rejects_an_over_long_value():
    with pytest.raises(HTTPException) as excinfo:
        envtokens._validate_token("a" * (envtokens.MAX_TOKEN_LENGTH + 1), "qwen_tokens")
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "qwen_tokens entries must be at most 4096 characters"


@pytest.mark.parametrize("raw", ["a\nb", "a\rb", "\n", "\r"])
def test_validate_token_rejects_line_breaks(raw):
    with pytest.raises(HTTPException) as excinfo:
        envtokens._validate_token(raw, "deepseek_tokens")
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "deepseek_tokens entries must not contain line breaks"


@pytest.mark.parametrize("bad", [",", " ", '"', "'", "#", "\\", "\x00", "\x1f", "\x7f", "a,b", "a b"])
def test_validate_token_rejects_forbidden_characters(bad):
    with pytest.raises(HTTPException) as excinfo:
        envtokens._validate_token(bad, "deepseek_tokens")
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "deepseek_tokens entries must not contain commas, quotes, backslashes, '#' or spaces"


def test_env_token_list_none_is_empty():
    assert envtokens._env_token_list(None, "deepseek_tokens") == []


def test_env_token_list_rejects_a_non_list():
    with pytest.raises(HTTPException) as excinfo:
        envtokens._env_token_list("a,b", "qwen_tokens")
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "qwen_tokens must be a list of strings"


def test_env_token_list_rejects_non_string_items():
    with pytest.raises(HTTPException) as excinfo:
        envtokens._env_token_list(["ok", 7], "deepseek_tokens")
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "deepseek_tokens must contain only strings"


def test_env_token_list_strips_dedupes_and_drops_empty_entries():
    assert envtokens._env_token_list([" a ", "a", "", "   ", "b"], "deepseek_tokens") == ["a", "b"]


def test_env_token_list_validates_every_kept_entry():
    with pytest.raises(HTTPException) as excinfo:
        envtokens._env_token_list(["ok", "bad,token"], "deepseek_tokens")
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "deepseek_tokens entries must not contain commas, quotes, backslashes, '#' or spaces"


def test_env_token_list_caps_at_max_tokens_per_request():
    cap = envtokens.MAX_TOKENS_PER_REQUEST
    assert cap == 64
    allowed = [f"token-{index}" for index in range(cap)]
    assert envtokens._env_token_list(allowed, "deepseek_tokens") == allowed
    with pytest.raises(HTTPException) as excinfo:
        envtokens._env_token_list([*allowed, "one-too-many"], "deepseek_tokens")
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "too many tokens in deepseek_tokens: max 64 per request"


def test_request_client_reports_the_peer_or_a_dash():
    assert envtokens._request_client(_request(host="203.0.113.5")) == "203.0.113.5"
    assert envtokens._request_client(_request(host=None)) == "-"
    assert envtokens._request_client(_request(host="")) == "-"


def test_presented_admin_token_prefers_x_api_key():
    assert envtokens._presented_admin_token(_request({"x-api-key": "  k1  "})) == "k1"
    assert envtokens._presented_admin_token(_request({"Authorization": "Bearer b1"})) == "b1"
    assert envtokens._presented_admin_token(_request({"Authorization": "bearer b2"})) == "b2"
    assert envtokens._presented_admin_token(_request({"x-api-key": "k3", "Authorization": "Bearer b3"})) == "k3"
    assert envtokens._presented_admin_token(_request({"Authorization": "Basic abc"})) == ""
    assert envtokens._presented_admin_token(_request({"x-api-key": "   "})) == ""
    assert envtokens._presented_admin_token(_request()) == ""


def test_admin_token_matches(monkeypatch):
    request = _request({"x-api-key": "test-admin-token"})
    assert envtokens.admin_token_matches(request) is True
    monkeypatch.setattr(settings, "admin_token", "")
    assert envtokens.admin_token_matches(request) is False
    monkeypatch.setattr(settings, "admin_token", "test-admin-token")
    assert envtokens.admin_token_matches(_request()) is False
    assert envtokens.admin_token_matches(_request({"x-api-key": "wrong"})) is False
    assert envtokens.admin_token_matches(_request({"x-api-key": "\u00ff"})) is False
    assert envtokens.admin_token_matches(_request({"Authorization": "Bearer \u00e9"})) is False


def test_require_admin_token_hides_the_route_in_byok_mode():
    app.state.byok = True
    with pytest.raises(HTTPException) as excinfo:
        envtokens._require_admin_token(_request({"x-api-key": "test-admin-token"}))
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "Unknown /v1 endpoint: /v1/tokens"


def test_require_admin_token_is_disabled_without_a_configured_token(monkeypatch):
    monkeypatch.setattr(settings, "admin_token", "")
    with pytest.raises(HTTPException) as excinfo:
        envtokens._require_admin_token(_request({"x-api-key": "anything"}))
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "token management is disabled, set DANYAPI_ADMIN_TOKEN to enable it"


def test_require_admin_token_logs_the_client_ip_and_never_the_secret(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER), pytest.raises(HTTPException) as excinfo:
        envtokens._require_admin_token(_request({"x-api-key": "leaked-secret-value"}))
    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == "invalid or missing admin token"
    assert len(caplog.records) == 1
    assert caplog.records[0].getMessage() == "rejected POST /v1/tokens from 203.0.113.5: wrong admin token"
    assert "leaked-secret-value" not in caplog.text


def test_require_admin_token_logs_a_missing_token(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER), pytest.raises(HTTPException) as excinfo:
        envtokens._require_admin_token(_request({"Authorization": "Bearer "}))
    assert excinfo.value.status_code == 401
    assert caplog.records[0].getMessage() == "rejected POST /v1/tokens from 203.0.113.5: no admin token presented"


def test_require_admin_token_accepts_the_configured_token():
    assert envtokens._require_admin_token(_request({"x-api-key": "test-admin-token"})) is None
    assert envtokens._require_admin_token(_request({"Authorization": "Bearer test-admin-token"})) is None


async def test_post_tokens_with_a_non_ascii_x_api_key_is_401_not_500():
    status, payload = await _asgi_post(
        [(b"host", b"testserver"), (b"content-type", b"application/json"), (b"x-api-key", b"\xff")],
        b'{"deepseek_tokens": []}',
    )
    assert status == 401
    assert payload["error"]["message"] == "invalid or missing admin token"
    assert payload["error"]["type"] == "authentication_error"


async def test_post_tokens_with_a_non_ascii_authorization_is_401_not_500():
    status, payload = await _asgi_post(
        [(b"host", b"testserver"), (b"content-type", b"application/json"), (b"authorization", b"Bearer \xff")],
        b'{"deepseek_tokens": []}',
    )
    assert status == 401
    assert payload["error"]["message"] == "invalid or missing admin token"
    assert payload["error"]["type"] == "authentication_error"


async def test_post_tokens_with_a_valid_token_reaches_the_handler(monkeypatch):
    monkeypatch.setattr(envtokens, "DeepSeekClient", lambda **_kwargs: _FakeClient(ok=False))
    status, payload = await _asgi_post(
        [(b"host", b"testserver"), (b"content-type", b"application/json"), (b"x-api-key", b"test-admin-token")],
        b'{"deepseek_tokens": ["nope"]}',
    )
    assert status == 200
    assert payload["skipped"]["deepseek"] == 1
    assert payload["added"] == {"deepseek": 0, "qwen": 0}
    assert payload["message"] == "No valid tokens to add."


def test_add_tokens_request_forbids_unknown_keys():
    with pytest.raises(envtokens.ValidationError) as excinfo:
        envtokens.AddTokensRequest.model_validate({"deepseek_tokens": ["a"], "nope": 1})
    assert excinfo.value.errors()[0]["type"] == "extra_forbidden"
    assert excinfo.value.errors()[0]["loc"] == ("nope",)
    assert envtokens.AddTokensRequest.model_config.get("extra") == "forbid"


def test_add_tokens_request_rejects_wrong_types():
    with pytest.raises(envtokens.ValidationError) as excinfo:
        envtokens.AddTokensRequest.model_validate({"deepseek_tokens": "a,b"})
    assert excinfo.value.errors()[0]["type"] == "list_type"


def test_coerce_tokens_passes_models_through_and_wraps_failures():
    model = envtokens.AddTokensRequest(deepseek_tokens=["a"])
    assert envtokens._coerce_tokens(model) is model
    assert envtokens._coerce_tokens({"deepseek_tokens": ["a"]}) == model
    with pytest.raises(HTTPException) as excinfo:
        envtokens._coerce_tokens({"deepseek_tokens": ["a"], "nope": 1})
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail.startswith("invalid request body: ")
    assert "extra_forbidden" in excinfo.value.detail


def test_pool_account_by_stable():
    assert envtokens._pool_account_by_stable(None, "sid") is None
    first = envtokens.DeepSeekAccount(0, _FakeClient(), stable_id="sid-0")
    second = envtokens.DeepSeekAccount(1, _FakeClient(), stable_id="sid-1")
    pool = envtokens.AccountPool([first, second])
    assert envtokens._pool_account_by_stable(pool, "sid-1") is second
    assert envtokens._pool_account_by_stable(pool, "missing") is None


def test_plan_tokens_classifies_every_state():
    live = envtokens.DeepSeekAccount(0, _FakeClient(), stable_id=_token_stable_id("live-token"))
    broken = envtokens.DeepSeekAccount(1, _FakeClient(), stable_id=_token_stable_id("broken-token"))
    broken.mark_broken()
    pool = envtokens.AccountPool([live, broken])
    plan = envtokens._plan_tokens(["live-token", "broken-token", "known-token", "brand-new"], {"known-token"}, pool)
    assert [(state, acct is None) for _token, state, acct in plan] == [
        (envtokens._PLAN_LIVE, False),
        (envtokens._PLAN_BROKEN, False),
        (envtokens._PLAN_UNTRACKED, True),
        (envtokens._PLAN_NEW, True),
    ]
    assert plan[0][2] is live
    assert plan[1][2] is broken


def test_plan_tokens_with_no_pool_reports_untracked_or_new():
    plan = envtokens._plan_tokens(["known", "fresh"], {"known"}, None)
    assert [state for _token, state, _acct in plan] == [envtokens._PLAN_UNTRACKED, envtokens._PLAN_NEW]


def test_known_tokens_drops_empty_entries():
    assert envtokens._known_tokens(["a", "", "b"], ["", "c"]) == {"a", "b", "c"}
    assert envtokens._known_tokens([], []) == set()


async def test_check_token_auth(caplog):
    assert await envtokens._check_token_auth(_FakeClient(ok=True)) is True
    failing = _FakeClient(auth_error=ValueError("boom"))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert await envtokens._check_token_auth(failing) is False
    assert caplog.records[0].getMessage() == "token auth check failed: ValueError: boom"


async def test_close_client_swallows_close_failures(caplog):
    client = _FakeClient()
    await envtokens._close_client(client)
    assert client.closed == 1
    broken = _FakeClient(close_error=OSError("nope"))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await envtokens._close_client(broken)
    assert broken.closed == 1
    assert caplog.records[0].getMessage() == "token client close failed: nope"


async def test_close_client_later_ignores_none_and_closes_the_rest():
    envtokens._close_client_later(None)
    client = _FakeClient()
    envtokens._close_client_later(client)
    assert len(envtokens._CLOSE_TASKS) == 1
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert client.closed == 1
    assert envtokens._CLOSE_TASKS == set()


async def test_validated_tokens_skips_live_accounts_without_a_round_trip():
    live = _FakeClient()
    acct = envtokens.DeepSeekAccount(0, live, stable_id="sid")
    validated, skipped = await envtokens._validated_tokens([("t", envtokens._PLAN_LIVE, acct)], "deepseek", _FakeClient)
    assert validated == []
    assert skipped == 0
    assert live.auth_calls == 0


async def test_validated_tokens_rechecks_a_broken_account_on_its_existing_client():
    client = _FakeClient(ok=True)
    acct = envtokens.DeepSeekAccount(0, client, stable_id="sid")
    acct.mark_broken()
    validated, skipped = await envtokens._validated_tokens([("t", envtokens._PLAN_BROKEN, acct)], "deepseek", _FakeClient)
    assert validated == [("t", envtokens._PLAN_BROKEN, acct, client)]
    assert skipped == 0
    assert client.auth_calls == 1
    assert client.closed == 0


async def test_validated_tokens_drops_a_broken_account_that_still_fails():
    client = _FakeClient(ok=False)
    acct = envtokens.DeepSeekAccount(0, client, stable_id="sid")
    acct.mark_broken()
    validated, skipped = await envtokens._validated_tokens([("t", envtokens._PLAN_BROKEN, acct)], "deepseek", _FakeClient)
    assert validated == []
    assert skipped == 0


async def test_validated_tokens_counts_only_new_rejections(caplog):
    clients = {"new": _FakeClient(ok=False), "untracked": _FakeClient(ok=False)}
    plan = [("new", envtokens._PLAN_NEW, None), ("untracked", envtokens._PLAN_UNTRACKED, None)]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        validated, skipped = await envtokens._validated_tokens(plan, "qwen", lambda token: clients[token])
    assert validated == []
    assert skipped == 1
    assert clients["new"].closed == 1
    assert clients["untracked"].closed == 1
    assert caplog.text.count("new qwen token invalid/expired, skipping") == 2


async def test_validated_tokens_keeps_accepted_new_tokens():
    client = _FakeClient(ok=True)
    validated, skipped = await envtokens._validated_tokens([("t", envtokens._PLAN_NEW, None)], "deepseek", lambda _token: client)
    assert validated == [("t", envtokens._PLAN_NEW, None, client)]
    assert skipped == 0
    assert client.closed == 0


async def test_validated_tokens_bounds_concurrency():
    assert envtokens.AUTH_CONCURRENCY == 8
    live = 0
    peak = 0

    class _CountingClient(_FakeClient):
        async def check_auth(self) -> bool:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            live -= 1
            return True

    tokens = [f"t{index}" for index in range(24)]
    plan = [(token, envtokens._PLAN_NEW, None) for token in tokens]
    validated, skipped = await envtokens._validated_tokens(plan, "deepseek", _CountingClient)
    assert len(validated) == 24
    assert skipped == 0
    assert peak == envtokens.AUTH_CONCURRENCY


async def test_validated_tokens_closes_the_pending_client_on_cancellation(caplog):
    client = _FakeClient(auth_error=asyncio.CancelledError())
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        validated, skipped = await envtokens._validated_tokens([("t", envtokens._PLAN_NEW, None)], "deepseek", lambda _token: client)
    assert validated == []
    assert skipped == 0
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert client.closed == 1
    assert "token auth check did not finish: CancelledError: " in caplog.text


def test_account_decision():
    assert envtokens._account_decision(None, "sid") == ("add", None)
    healthy = envtokens.DeepSeekAccount(0, _FakeClient(), stable_id="sid")
    pool = envtokens.AccountPool([healthy])
    assert envtokens._account_decision(pool, "sid") == ("skip", healthy)
    healthy.mark_broken()
    assert envtokens._account_decision(pool, "sid") == ("revive", healthy)
    assert envtokens._account_decision(pool, "other") == ("add", None)


def test_apply_deepseek_creates_a_pool_when_there_is_none():
    token = "brand-new-token"
    client = _FakeClient()
    pool, counts = envtokens._apply_deepseek([(token, envtokens._PLAN_NEW, None, client)], None, set(), None, None, None)
    assert counts == {"added": 1, "reactivated": 0, "activated": 0}
    assert app.state.pool is pool
    assert pool.label == "deepseek"
    assert [acct.index for acct in pool.accounts] == [0]
    assert pool.accounts[0].stable_id == _token_stable_id(token)
    assert pool.accounts[0].client is client


def test_apply_deepseek_appends_with_contiguous_indexes():
    existing = envtokens.DeepSeekAccount(0, _FakeClient(), stable_id="other")
    pool = envtokens.AccountPool([existing])
    prepared = [("t1", envtokens._PLAN_NEW, None, _FakeClient()), ("t2", envtokens._PLAN_NEW, None, _FakeClient())]
    pool, counts = envtokens._apply_deepseek(prepared, pool, set(), None, None, None)
    assert counts == {"added": 2, "reactivated": 0, "activated": 0}
    assert [acct.index for acct in pool.accounts] == [0, 1, 2]
    assert [acct.stable_id for acct in pool.accounts[1:]] == [_token_stable_id("t1"), _token_stable_id("t2")]


def test_apply_deepseek_activates_a_persisted_token_without_an_account():
    prepared = [("persisted", envtokens._PLAN_UNTRACKED, None, _FakeClient())]
    pool, counts = envtokens._apply_deepseek(prepared, None, {"persisted"}, None, None, None)
    assert counts == {"added": 0, "reactivated": 0, "activated": 1}
    assert len(pool.accounts) == 1


async def test_apply_deepseek_revives_a_broken_account_and_closes_the_replacement():
    stable = _token_stable_id("t")
    stale_client = _FakeClient()
    broken = envtokens.DeepSeekAccount(0, stale_client, stable_id=stable)
    broken.mark_broken()
    pool = envtokens.AccountPool([broken])
    fresh_client = _FakeClient()
    replacement = envtokens.DeepSeekAccount(0, fresh_client, stable_id=stable)
    prepared = [("t", envtokens._PLAN_BROKEN, replacement, fresh_client)]
    pool, counts = envtokens._apply_deepseek(prepared, pool, set(), None, None, None)
    assert counts == {"added": 0, "reactivated": 1, "activated": 0}
    assert pool.accounts[0] is broken
    assert broken.broken is False
    assert broken.broken_at is None
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert fresh_client.closed == 1
    assert stale_client.closed == 0


async def test_apply_deepseek_skips_a_healthy_account_and_closes_an_orphan_client():
    healthy = envtokens.DeepSeekAccount(0, _FakeClient(), stable_id=_token_stable_id("t"))
    pool = envtokens.AccountPool([healthy])
    orphan = _FakeClient()
    result, counts = envtokens._apply_deepseek([("t", envtokens._PLAN_UNTRACKED, None, orphan)], pool, set(), None, None, None)
    assert result is pool
    assert counts == {"added": 0, "reactivated": 0, "activated": 0}
    assert pool.accounts == [healthy]
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert orphan.closed == 1


async def test_apply_qwen_mirrors_deepseek():
    client = _FakeClient()
    pool, counts = envtokens._apply_qwen([("qwen-token", envtokens._PLAN_NEW, None, client)], None, set(), None, None, None)
    assert counts == {"added": 1, "reactivated": 0, "activated": 0}
    assert app.state.qwen_pool is pool
    assert pool.label == "qwen"
    assert pool.accounts[0].client is client

    pool, counts = envtokens._apply_qwen([("second", envtokens._PLAN_NEW, None, _FakeClient())], pool, set(), None, None, None)
    assert counts == {"added": 1, "reactivated": 0, "activated": 0}
    assert [acct.index for acct in pool.accounts] == [0, 1]

    broken = pool.accounts[0]
    broken.mark_broken()
    pool, counts = envtokens._apply_qwen(
        [("qwen-token", envtokens._PLAN_BROKEN, broken, _FakeClient())],
        pool,
        set(),
        None,
        None,
        None,
    )
    assert counts == {"added": 0, "reactivated": 1, "activated": 0}
    assert broken.broken is False

    orphan = _FakeClient()
    result, counts = envtokens._apply_qwen(
        [("qwen-token", envtokens._PLAN_UNTRACKED, None, orphan)],
        pool,
        {"qwen-token"},
        None,
        None,
        None,
    )
    assert result is pool
    assert counts == {"added": 0, "reactivated": 0, "activated": 0}
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert orphan.closed == 1


async def test_apply_qwen_closes_the_replacement_client_on_revival():
    stable = _token_stable_id("dup")
    stale = envtokens.QwenAccount(0, _FakeClient(), stable_id=stable)
    stale.mark_broken()
    pool = envtokens.AccountPool([stale], label="qwen")
    fresh_client = _FakeClient()
    replacement = envtokens.QwenAccount(0, fresh_client, stable_id=stable)
    pool, counts = envtokens._apply_qwen([("dup", envtokens._PLAN_BROKEN, replacement, fresh_client)], pool, set(), None, None, None)
    assert counts == {"added": 0, "reactivated": 1, "activated": 0}
    assert stale.broken is False
    assert stale.broken_at is None
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert fresh_client.closed == 1


async def test_refresh_models_skips_an_empty_pool(monkeypatch):
    recorder = _RefreshRecorder()
    monkeypatch.setattr(envtokens, "refresh_provider_models", recorder)
    await envtokens._refresh_models("deepseek", envtokens.AccountPool([]))
    assert recorder.calls == []


async def test_refresh_models_wraps_deepseek(monkeypatch):
    recorder = _RefreshRecorder()
    monkeypatch.setattr(envtokens, "refresh_provider_models", recorder)
    client = _FakeClient()
    pool = envtokens.AccountPool([envtokens.DeepSeekAccount(0, client, stable_id="sid")])
    await envtokens._refresh_models("deepseek", pool)
    assert recorder.calls == [("deepseek", client)]


async def test_refresh_models_wraps_qwen(monkeypatch):
    models = _QwenModelsRecorder()
    monkeypatch.setattr(envtokens, "_fetch_qwen_models", models)
    client = _FakeClient()
    pool = envtokens.AccountPool([envtokens.QwenAccount(0, client, stable_id="sid")], label="qwen")
    await envtokens._refresh_models("qwen", pool)
    assert models.calls == [client]
    assert app.state.qwen_models == [{"id": "qwen-live", "name": "Q", "owned_by": "qwen", "model_type": "chat"}]


async def test_refresh_models_logs_a_failure(monkeypatch, caplog):
    async def boom(_provider: str, _client: Any) -> list[dict[str, str]]:
        raise OSError("no route to host")

    monkeypatch.setattr(envtokens, "refresh_provider_models", boom)
    client = _FakeClient()
    pool = envtokens.AccountPool([envtokens.DeepSeekAccount(0, client, stable_id="sid")])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await envtokens._refresh_models("deepseek", pool)
    assert caplog.records[0].getMessage() == "failed to refresh deepseek models: no route to host"


def test_token_result_messages():
    assert envtokens._token_result(1, 0, 2, 3, 0, 0, 0, 0) == {
        "success": True,
        "message": "Tokens added and activated.",
        "added": {"deepseek": 1, "qwen": 0},
        "skipped": {"deepseek": 2, "qwen": 3},
        "reactivated": {"deepseek": 0, "qwen": 0},
        "activated": {"deepseek": 0, "qwen": 0},
    }
    assert envtokens._token_result(0, 0, 0, 0, 0, 1, 0, 0)["message"] == "Tokens reactivated."
    assert envtokens._token_result(0, 0, 0, 0, 0, 0, 0, 1)["message"] == "Tokens reactivated."
    assert envtokens._token_result(0, 0, 0, 0, 0, 0, 0, 0)["message"] == "No valid tokens to add."


async def test_add_tokens_requires_at_least_one_token():
    with pytest.raises(HTTPException) as excinfo:
        await envtokens.add_tokens(envtokens.AddTokensRequest())
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "no tokens provided"
    with pytest.raises(HTTPException) as empty:
        await envtokens.add_tokens(envtokens.AddTokensRequest(deepseek_tokens=["", "   "]))
    assert empty.value.status_code == 400
    assert empty.value.detail == "no tokens provided"


async def test_add_tokens_writes_the_file_and_adds_the_accounts(monkeypatch, _isolated):
    client = _FakeClient()
    recorder = _RefreshRecorder()
    monkeypatch.setattr(envtokens, "DeepSeekClient", lambda **_kwargs: client)
    monkeypatch.setattr(envtokens, "refresh_provider_models", recorder)
    body = await envtokens.add_tokens(envtokens.AddTokensRequest(deepseek_tokens=["t1", "t2", "t1"]))
    assert body == {
        "success": True,
        "message": "Tokens added and activated.",
        "added": {"deepseek": 2, "qwen": 0},
        "skipped": {"deepseek": 0, "qwen": 0},
        "reactivated": {"deepseek": 0, "qwen": 0},
        "activated": {"deepseek": 0, "qwen": 0},
    }
    assert _isolated.read_text(encoding="utf-8") == "DEEPSEEK_TOKENS=t1,t2\nQWEN_TOKENS=\n"
    assert settings.deepseek_tokens == ["t1", "t2"]
    assert settings.qwen_tokens == []
    assert [acct.index for acct in app.state.pool.accounts] == [0, 1]
    assert all(acct.client is client for acct in app.state.pool.accounts)
    assert recorder.calls == [("deepseek", client)]


async def test_add_tokens_refreshes_deepseek_models_on_an_add(monkeypatch, _isolated):
    monkeypatch.setattr(envtokens, "DeepSeekClient", lambda **_kwargs: _FakeClient())
    seen: list[str] = []

    async def recorder(provider: str, _client: Any) -> list[dict[str, str]]:
        seen.append(provider)
        return []

    monkeypatch.setattr(envtokens, "refresh_provider_models", recorder)
    await envtokens.add_tokens(envtokens.AddTokensRequest(deepseek_tokens=["fresh"]))
    assert seen == ["deepseek"]


async def test_add_tokens_refreshes_qwen_models_on_an_add(monkeypatch, _isolated):
    monkeypatch.setattr(envtokens, "QwenClient", lambda **_kwargs: _FakeClient())
    monkeypatch.setattr(envtokens, "refresh_provider_models", _RefreshRecorder())
    models = _QwenModelsRecorder()
    monkeypatch.setattr(envtokens, "_fetch_qwen_models", models)
    body = await envtokens.add_tokens(envtokens.AddTokensRequest(qwen_tokens=["q1"]))
    assert body["added"] == {"deepseek": 0, "qwen": 1}
    assert body["message"] == "Tokens added and activated."
    assert _isolated.read_text(encoding="utf-8") == "DEEPSEEK_TOKENS=\nQWEN_TOKENS=q1\n"
    assert len(models.calls) == 1
    assert app.state.qwen_pool.label == "qwen"
    assert app.state.qwen_pool.accounts[0].stable_id == _token_stable_id("q1")


async def test_add_tokens_leaves_the_pool_untouched_when_the_write_fails(monkeypatch, _isolated):
    monkeypatch.setattr(envtokens, "DeepSeekClient", lambda **_kwargs: _FakeClient())
    monkeypatch.setattr(envtokens, "refresh_provider_models", _RefreshRecorder())

    def boom(_ds: list[str], _qw: list[str]) -> None:
        raise HTTPException(500, "cannot write the credentials file")

    monkeypatch.setattr(envtokens, "_write_env_tokens", boom)
    with pytest.raises(HTTPException) as excinfo:
        await envtokens.add_tokens(envtokens.AddTokensRequest(deepseek_tokens=["t"]))
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "cannot write the credentials file"
    assert app.state.pool is None
    assert not _isolated.exists()


async def test_add_tokens_does_not_rewrite_the_file_for_a_token_from_the_environment(monkeypatch, _isolated):
    monkeypatch.setattr(settings, "deepseek_tokens", ["env-token"])
    monkeypatch.setattr(envtokens, "DeepSeekClient", lambda **_kwargs: _FakeClient())
    monkeypatch.setattr(envtokens, "refresh_provider_models", _RefreshRecorder())

    def boom(_ds: list[str], _qw: list[str]) -> None:
        raise AssertionError("the credentials file must not be rewritten when the merged list is unchanged")

    monkeypatch.setattr(envtokens, "_write_env_tokens", boom)
    body = await envtokens.add_tokens(envtokens.AddTokensRequest(deepseek_tokens=["env-token"]))
    assert body["added"] == {"deepseek": 1, "qwen": 0}
    assert not _isolated.exists()
    assert len(app.state.pool.accounts) == 1


async def test_add_tokens_skips_a_duplicate_account_with_the_same_stable_id(monkeypatch, _isolated):
    monkeypatch.setattr(envtokens, "DeepSeekClient", lambda **_kwargs: _FakeClient())
    monkeypatch.setattr(envtokens, "refresh_provider_models", _RefreshRecorder())
    healthy = envtokens.DeepSeekAccount(0, _FakeClient(), stable_id=_token_stable_id("dup"))
    app.state.pool = envtokens.AccountPool([healthy])
    with pytest.raises(HTTPException) as excinfo:
        await envtokens.add_tokens(envtokens.AddTokensRequest(deepseek_tokens=["dup"]))
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "all provided tokens already exist"
    assert app.state.pool.accounts == [healthy]
    assert not _isolated.exists()


async def test_add_tokens_activates_a_persisted_token_without_an_account(monkeypatch, _isolated):
    _isolated.write_text("DEEPSEEK_TOKENS=kept\n", encoding="utf-8")
    monkeypatch.setattr(envtokens, "DeepSeekClient", lambda **_kwargs: _FakeClient())
    monkeypatch.setattr(envtokens, "refresh_provider_models", _RefreshRecorder())

    def boom(_ds: list[str], _qw: list[str]) -> None:
        raise AssertionError("a persisted token must not be written again")

    monkeypatch.setattr(envtokens, "_write_env_tokens", boom)
    body = await envtokens.add_tokens(envtokens.AddTokensRequest(deepseek_tokens=["kept"]))
    assert body["message"] == "Tokens reactivated."
    assert body["activated"] == {"deepseek": 1, "qwen": 0}
    assert body["added"] == {"deepseek": 0, "qwen": 0}
    assert len(app.state.pool.accounts) == 1


async def test_add_tokens_reactivates_a_broken_account(monkeypatch, _isolated):
    broken = envtokens.DeepSeekAccount(0, _FakeClient(), stable_id=_token_stable_id("old"))
    broken.mark_broken()
    app.state.pool = envtokens.AccountPool([broken])
    monkeypatch.setattr(envtokens, "DeepSeekClient", lambda **_kwargs: _FakeClient())
    monkeypatch.setattr(envtokens, "refresh_provider_models", _RefreshRecorder())
    body = await envtokens.add_tokens(envtokens.AddTokensRequest(deepseek_tokens=["old"]))
    assert body["message"] == "Tokens reactivated."
    assert body["reactivated"] == {"deepseek": 1, "qwen": 0}
    assert broken.broken is False
    assert broken.broken_at is None
    assert app.state.pool.accounts == [broken]


async def test_add_tokens_dedupes_the_merged_file_list(monkeypatch, _isolated):
    _isolated.write_text("DEEPSEEK_TOKENS=kept\n", encoding="utf-8")
    monkeypatch.setattr(envtokens, "DeepSeekClient", lambda **_kwargs: _FakeClient())
    monkeypatch.setattr(envtokens, "refresh_provider_models", _RefreshRecorder())
    body = await envtokens.add_tokens(envtokens.AddTokensRequest(deepseek_tokens=["kept", "extra", "extra"]))
    assert body["added"] == {"deepseek": 1, "qwen": 0}
    assert body["activated"] == {"deepseek": 1, "qwen": 0}
    assert _isolated.read_text(encoding="utf-8") == "DEEPSEEK_TOKENS=kept,extra\nQWEN_TOKENS=\n"
    assert len(app.state.pool.accounts) == 2


async def test_add_tokens_reports_a_rejected_new_token(monkeypatch, _isolated, caplog):
    monkeypatch.setattr(envtokens, "DeepSeekClient", lambda **_kwargs: _FakeClient(ok=False))
    monkeypatch.setattr(envtokens, "refresh_provider_models", _RefreshRecorder())
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        body = await envtokens.add_tokens(envtokens.AddTokensRequest(deepseek_tokens=["bad"]))
    assert body["skipped"] == {"deepseek": 1, "qwen": 0}
    assert body["added"] == {"deepseek": 0, "qwen": 0}
    assert body["message"] == "No valid tokens to add."
    assert app.state.pool is None
    assert not _isolated.exists()
    assert "new deepseek token invalid/expired, skipping" in caplog.text


def test_module_imports_without_python_dotenv():
    saved = sys.modules.get("dotenv")
    assert envtokens._dotenv_values is not None
    try:
        sys.modules["dotenv"] = None
        assert importlib.reload(envtokens)._dotenv_values is None
    finally:
        if saved is not None:
            sys.modules["dotenv"] = saved
        importlib.reload(envtokens)
    assert envtokens._dotenv_values is not None
