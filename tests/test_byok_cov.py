import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile
from starlette.requests import Request

import danyapi.api.byok as byok_mod
from danyapi.api.state import BYOK_PROVIDERS, app
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


class _FakeSemaphore:
    def __init__(self, acquire_now: bool = True) -> None:
        self.acquire_now = acquire_now
        self.releases = 0
        self._never = asyncio.Event()

    def locked(self) -> bool:
        return not self.acquire_now

    async def acquire(self) -> bool:
        if not self.acquire_now:
            await self._never.wait()
        return True

    def release(self) -> None:
        self.releases += 1


def _make_request(
    headers: dict[str, str] | None = None,
    body: bytes = b"",
    *,
    app_scope: bool = False,
    receive: Any = None,
) -> Request:
    async def default_receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    pairs = [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "query_string": b"",
        "headers": pairs,
        "client": ("127.0.0.1", 5555),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    if app_scope:
        scope["app"] = app
    return Request(scope, receive=receive or default_receive)


def _multipart(boundary: str, fields: dict[str, str], files: dict[str, tuple[str, bytes, str]] | None = None) -> bytes:
    out = bytearray()
    for name, value in fields.items():
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
    for name, (filename, payload, content_type) in (files or {}).items():
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n').encode()
        out += payload
        out += b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out)


class _RefreshRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def __call__(self, provider: str, client: Any) -> list[dict[str, str]]:
        self.calls.append((provider, client))
        return []


def _discarding_wait_for(inner: Any) -> Any:
    async def wrapper(awaitable: Any, timeout: float) -> Any:
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()
        return await inner(awaitable, timeout)

    return wrapper


@pytest.fixture(autouse=True)
def _byok_state(monkeypatch):
    saved = dict(app.state._state)
    app.state.byok = False
    app.state.byok_auth = {provider: {} for provider in BYOK_PROVIDERS}
    app.state.byok_pools = {provider: {} for provider in BYOK_PROVIDERS}
    app.state.byok_stores = {provider: {} for provider in BYOK_PROVIDERS}
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(byok_mod, "_KEY_LOCKS", {})
    monkeypatch.setattr(byok_mod, "_ALICE_BYOK_POOL", [None])
    monkeypatch.setattr(byok_mod, "_DUCKAI_BYOK_POOL", [None])
    monkeypatch.setattr(byok_mod, "_ALICE_BYOK_LOCK", asyncio.Lock())
    monkeypatch.setattr(byok_mod, "_DUCKAI_BYOK_LOCK", asyncio.Lock())
    yield
    app.state._state.clear()
    app.state._state.update(saved)


def test_cached_auth_rejects_a_disabled_ttl():
    assert byok_mod._cached_auth({}, "sid", 0.0, 100.0) is None
    assert byok_mod._cached_auth({}, "sid", -1.0, 100.0) is None


@pytest.mark.parametrize("record", ["nope", [True], [True, 1.0, 3], {"ok": True}, None])
def test_cached_auth_rejects_a_malformed_record(record):
    assert byok_mod._cached_auth({"sid": record}, "sid", 300.0, 100.0) is None


@pytest.mark.parametrize("stamp", ["not-a-number", None, [1.0]])
def test_cached_auth_rejects_a_malformed_timestamp(stamp):
    assert byok_mod._cached_auth({"sid": [True, stamp]}, "sid", 300.0, 100.0) is None


def test_cached_auth_honours_the_ttl():
    store = {"sid": [True, 100.0], "old": [False, 10.0]}
    assert byok_mod._cached_auth(store, "sid", 300.0, 100.0) is True
    assert byok_mod._cached_auth(store, "old", 300.0, 100.0) is False
    assert byok_mod._cached_auth(store, "sid", 300.0, 500.0) is None
    assert byok_mod._cached_auth(store, "old", 300.0, 500.0) is None
    assert byok_mod._cached_auth(store, "missing", 300.0, 100.0) is None


def test_evict_auth_drops_the_least_recently_used_entry(monkeypatch):
    monkeypatch.setattr(byok_mod, "BYOK_AUTH_LIMIT", 3)
    store: dict[str, Any] = {"a": [True, 1.0], "b": [True, 1.0], "c": [True, 1.0]}
    byok_mod._touch_auth(store, "a")
    assert list(store) == ["b", "c", "a"]
    store["d"] = [True, 1.0]
    byok_mod._evict_auth(store)
    assert list(store) == ["c", "a", "d"]
    assert "a" in store
    assert "b" not in store


def test_touch_auth_ignores_a_missing_entry():
    store: dict[str, Any] = {}
    byok_mod._touch_auth(store, "absent")
    assert store == {}


def test_touch_pool_cache_ignores_a_missing_entry():
    cache: dict[str, Any] = {}
    byok_mod._touch_pool_cache(cache, "absent")
    assert cache == {}


def test_touch_pool_cache_moves_a_hit_to_the_end():
    first = object()
    second = object()
    cache = {"a": first, "b": second}
    byok_mod._touch_pool_cache(cache, "a")
    assert list(cache) == ["b", "a"]
    assert cache["a"] is first


def test_load_byok_salt_creates_the_file_once_and_reuses_it(monkeypatch, tmp_path):
    monkeypatch.setattr(byok_mod, "cache_root", lambda: tmp_path)
    salt_file = tmp_path / byok_mod.BYOK_SALT_FILE
    assert not salt_file.exists()
    first = byok_mod._load_byok_salt()
    assert salt_file.exists()
    assert len(first) == 32
    assert salt_file.read_bytes() == first
    second = byok_mod._load_byok_salt()
    assert second == first
    assert salt_file.read_bytes() == first


def test_load_byok_salt_regenerates_a_short_salt(monkeypatch, tmp_path):
    monkeypatch.setattr(byok_mod, "cache_root", lambda: tmp_path)
    salt_file = tmp_path / byok_mod.BYOK_SALT_FILE
    salt_file.write_bytes(b"too-short")
    salt = byok_mod._load_byok_salt()
    assert len(salt) == 32
    assert salt != b"too-short"
    assert salt_file.read_bytes() == salt


def test_load_byok_salt_truncates_an_oversized_salt(monkeypatch, tmp_path):
    monkeypatch.setattr(byok_mod, "cache_root", lambda: tmp_path)
    salt_file = tmp_path / byok_mod.BYOK_SALT_FILE
    salt_file.write_bytes(bytes(range(64)))
    assert byok_mod._load_byok_salt() == bytes(range(32))


def test_load_byok_salt_survives_an_unreadable_salt_file(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(byok_mod, "cache_root", lambda: tmp_path)
    (tmp_path / byok_mod.BYOK_SALT_FILE).mkdir()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        salt = byok_mod._load_byok_salt()
    assert len(salt) == 32
    assert "byok affinity salt is not readable, a new one is generated" in caplog.text


def test_load_byok_salt_survives_an_unwritable_cache_dir(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(byok_mod, "cache_root", lambda: tmp_path / "missing")

    def boom(_self: Path, _data: bytes) -> int:
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "write_bytes", boom)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        salt = byok_mod._load_byok_salt()
    assert len(salt) == 32
    assert "byok affinity salt cannot be persisted, session affinity is lost on restart" in caplog.text


def test_byok_stable_id_is_deterministic_for_a_fixed_salt(monkeypatch):
    monkeypatch.setattr(byok_mod, "_BYOK_SALT", b"\x01" * 32)
    assert byok_mod._byok_stable_id("tok") == "6ed68981996dbe17"
    assert byok_mod._byok_stable_id("tok") == byok_mod._byok_stable_id("tok")
    assert byok_mod._byok_stable_id("other") != byok_mod._byok_stable_id("tok")
    assert "tok" not in byok_mod._byok_stable_id("tok")


def test_byok_stable_id_is_salted_and_not_a_plain_digest(monkeypatch):
    plain = hashlib.sha256(b"tok").hexdigest()[:16]
    monkeypatch.setattr(byok_mod, "_BYOK_SALT", b"\x01" * 32)
    assert byok_mod._byok_stable_id("tok") != plain


def test_byok_scope_hides_the_cache_key_and_follows_the_cache_setting(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", True)
    scope = byok_mod._byok_scope("cache-key")
    assert scope == "byok-89dd5bff4fb6fd83"
    assert "cache-key" not in scope
    monkeypatch.setattr(settings, "cache_enabled", False)
    assert byok_mod._byok_scope("cache-key") is None


def test_key_lock_is_per_key_and_prunes_unlocked_entries(monkeypatch):
    monkeypatch.setattr(byok_mod, "BYOK_KEY_LOCK_LIMIT", 4)
    locks = [byok_mod._key_lock("deepseek", f"k{index}") for index in range(4)]
    assert len({id(lock) for lock in locks}) == 4
    assert len(byok_mod._KEY_LOCKS) == 4
    fresh = byok_mod._key_lock("deepseek", "k4")
    assert len(byok_mod._KEY_LOCKS) < 5
    assert byok_mod._KEY_LOCKS["deepseek:k4"] is fresh
    assert byok_mod._key_lock("deepseek", "k4") is fresh
    assert byok_mod._key_lock("qwen", "k4") is not fresh


async def test_key_lock_never_prunes_a_held_lock(monkeypatch):
    monkeypatch.setattr(byok_mod, "BYOK_KEY_LOCK_LIMIT", 2)
    held = byok_mod._key_lock("deepseek", "held")
    byok_mod._key_lock("deepseek", "other")
    await held.acquire()
    try:
        assert held.locked() is True
        byok_mod._key_lock("deepseek", "third")
        assert "deepseek:held" in byok_mod._KEY_LOCKS
        assert "deepseek:other" not in byok_mod._KEY_LOCKS
    finally:
        held.release()


async def test_api_key_from_form_does_not_close_a_cached_form():
    boundary = "----cov"
    body = _multipart(boundary, {"api_key": "  cached-key "}, {"image": ("a.png", b"png", "image/png")})
    request = _make_request(headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, body=body)
    form = await request.form()
    upload = form["image"]
    assert isinstance(upload, UploadFile)
    assert upload.file.closed is False
    assert await byok_mod._api_key_from_form(request) == "cached-key"
    assert upload.file.closed is False
    assert await upload.read() == b"png"


async def test_api_key_from_form_closes_a_form_it_parsed_itself():
    boundary = "----cov"
    body = _multipart(boundary, {"api_key": "fresh-key"}, {"image": ("a.png", b"png", "image/png")})
    request = _make_request(headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, body=body)
    assert getattr(request, "_form", None) is None
    assert await byok_mod._api_key_from_form(request) == "fresh-key"
    form = request._form
    assert form is not None
    assert form["image"].file.closed is True


async def test_api_key_from_form_returns_none_without_a_field():
    boundary = "----cov"
    body = _multipart(boundary, {"model": "qwen"})
    request = _make_request(headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, body=body)
    assert await byok_mod._api_key_from_form(request) is None
    blank = _make_request(headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, body=_multipart(boundary, {"api_key": "   "}))
    assert await byok_mod._api_key_from_form(blank) is None


async def test_api_key_from_form_rejects_an_oversized_body():
    request = _make_request(
        headers={"Content-Type": "multipart/form-data; boundary=zz", "content-length": str(byok_mod.BYOK_FORM_MAX_BYTES + 1)},
    )
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._api_key_from_form(request)
    assert excinfo.value.status_code == 413
    assert excinfo.value.detail == "multipart body too large"


async def test_api_key_from_form_rejects_a_bad_content_length():
    request = _make_request(headers={"Content-Type": "multipart/form-data; boundary=zz", "content-length": "abc"})
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._api_key_from_form(request)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "invalid content-length header"


async def test_api_key_from_form_reports_a_malformed_body():
    request = _make_request(headers={"Content-Type": "multipart/form-data; boundary=zz"}, body=b"garbage")
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._api_key_from_form(request)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "malformed multipart request body"


async def test_api_key_from_form_honours_the_file_and_field_limits():
    boundary = "----cov"
    at_limit = _multipart(
        boundary,
        {f"field{index}": "v" for index in range(byok_mod.BYOK_FORM_MAX_FIELDS)},
        {f"file{index}": (f"{index}.bin", b"x", "application/octet-stream") for index in range(byok_mod.BYOK_FORM_MAX_FILES)},
    )
    request = _make_request(headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, body=at_limit)
    assert await byok_mod._api_key_from_form(request) is None

    over_files = _multipart(
        boundary,
        {"api_key": "k"},
        {f"file{index}": (f"{index}.bin", b"x", "application/octet-stream") for index in range(byok_mod.BYOK_FORM_MAX_FILES + 1)},
    )
    too_many_files = _make_request(headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, body=over_files)
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._api_key_from_form(too_many_files)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "malformed multipart request body"

    over_fields = _multipart(
        boundary,
        {"api_key": "k", **{f"field{index}": "v" for index in range(byok_mod.BYOK_FORM_MAX_FIELDS + 1)}},
    )
    too_many_fields = _make_request(headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, body=over_fields)
    with pytest.raises(HTTPException) as field_exc:
        await byok_mod._api_key_from_form(too_many_fields)
    assert field_exc.value.status_code == 400
    assert field_exc.value.detail == "malformed multipart request body"


async def test_api_key_from_form_reports_a_parser_rejection_as_a_bad_request(caplog):
    boundary = "----cov"
    body = _multipart(
        boundary,
        {"api_key": "k"},
        {f"file{index}": (f"{index}.bin", b"x", "application/octet-stream") for index in range(byok_mod.BYOK_FORM_MAX_FILES + 1)},
    )
    request = _make_request(headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, body=body, app_scope=True)
    with caplog.at_level(logging.INFO, logger=LOGGER), pytest.raises(HTTPException) as excinfo:
        await byok_mod._api_key_from_form(request)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "malformed multipart request body"
    assert "byok multipart body could not be parsed: 400: Too many files. Maximum number of files is 8." in caplog.text


async def test_extract_request_api_key_uses_the_cached_body_without_reading_again():
    async def explode() -> dict[str, Any]:
        raise AssertionError("the request body must not be read a second time")

    request = _make_request(headers={"Content-Type": "application/json"}, receive=explode)
    request._body = b'{"api_key": "cached-json-key"}'
    assert await byok_mod._extract_request_api_key(request) == "cached-json-key"


async def test_extract_request_api_key_reads_the_body_when_it_is_not_cached():
    request = _make_request(headers={"Content-Type": "application/json"}, body=b'{"api_key": "streamed"}')
    assert await byok_mod._extract_request_api_key(request) == "streamed"


async def test_extract_request_api_key_skips_an_oversized_json_body(caplog):
    oversized = b'{"api_key": "' + b"x" * byok_mod.BYOK_MAX_JSON_BODY + b'"}'
    request = _make_request(headers={"Content-Type": "application/json"}, body=oversized)
    request._body = oversized
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert await byok_mod._extract_request_api_key(request) is None
    assert "the json body is ignored for bodies over 1048576 bytes" in caplog.text


async def test_extract_request_api_key_rejects_an_oversized_declared_json_body():
    request = _make_request(
        headers={"Content-Type": "application/json", "content-length": str(byok_mod.MAX_REQUEST_BODY + 1)},
    )
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._extract_request_api_key(request)
    assert excinfo.value.status_code == 413
    assert excinfo.value.detail == "request body too large"


async def test_extract_request_api_key_ignores_a_bad_declared_json_length():
    request = _make_request(headers={"Content-Type": "application/json", "content-length": "abc"})
    assert await byok_mod._extract_request_api_key(request) is None


async def test_extract_request_api_key_reads_the_api_key_from_a_multipart_form():
    boundary = "----cov"
    body = _multipart(boundary, {"api_key": " form-key ", "model": "qwen"})
    request = _make_request(headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, body=body)
    assert await byok_mod._extract_request_api_key(request) == "form-key"


async def test_extract_request_api_key_survives_an_unreadable_body():
    async def explode() -> dict[str, Any]:
        raise OSError("client disconnected")

    request = _make_request(headers={"Content-Type": "application/json"}, receive=explode)
    assert await byok_mod._extract_request_api_key(request) is None


async def test_extract_request_api_key_ignores_an_empty_or_broken_json_body():
    empty = _make_request(headers={"Content-Type": "application/json"}, body=b"")
    assert await byok_mod._extract_request_api_key(empty) is None
    broken = _make_request(headers={"Content-Type": "application/json"}, body=b"{not json")
    assert await byok_mod._extract_request_api_key(broken) is None
    undecodable = _make_request(headers={"Content-Type": "application/json"}, body=b"\xff\xfe\x00")
    assert await byok_mod._extract_request_api_key(undecodable) is None
    not_a_dict = _make_request(headers={"Content-Type": "application/json"}, body=b"[1, 2]")
    assert await byok_mod._extract_request_api_key(not_a_dict) is None
    wrong_type = _make_request(headers={"Content-Type": "application/json"}, body=b'{"api_key": 7}')
    assert await byok_mod._extract_request_api_key(wrong_type) is None
    blank = _make_request(headers={"Content-Type": "application/json"}, body=b'{"api_key": "  "}')
    assert await byok_mod._extract_request_api_key(blank) is None


async def test_extract_request_api_key_ignores_unsupported_content_types():
    assert await byok_mod._extract_request_api_key(_make_request(headers={"Content-Type": "text/plain"})) is None
    assert await byok_mod._extract_request_api_key(_make_request(headers={})) is None


async def test_close_client_reports_a_close_failure(caplog):
    broken = _FakeClient(close_error=OSError("already gone"))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        await byok_mod._close_client(broken)
    assert broken.closed == 1
    assert "byok client close failed: already gone" in caplog.text


async def test_close_client_later_ignores_none_and_schedules_the_close():
    byok_mod._close_client_later(None)
    assert byok_mod._deferred_close_tasks == set()
    client = _FakeClient()
    byok_mod._close_client_later(client)
    assert len(byok_mod._deferred_close_tasks) == 1
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert client.closed == 1
    assert byok_mod._deferred_close_tasks == set()


async def test_close_pool_reports_every_cleanup_failure(caplog):
    bad_sessions = MagicMock()
    bad_sessions.close_all.side_effect = RuntimeError("registry gone")
    bad_flush = MagicMock()
    bad_flush.side_effect = RuntimeError("flush gone")
    bad_store = MagicMock()
    bad_store.remove.side_effect = RuntimeError("unlink gone")
    bad_client = MagicMock()
    bad_client.aclose = AsyncMock(side_effect=RuntimeError("client gone"))
    account = MagicMock()
    account.label = "acct#0"
    account.sessions = bad_sessions
    account.client = bad_client
    account.sem = None
    pool = MagicMock()
    pool.accounts = [account]
    pool.flush = bad_flush
    with caplog.at_level(logging.INFO, logger=LOGGER):
        await byok_mod._close_pool(pool, [bad_store])
    assert bad_sessions.close_all.call_count == 1
    assert bad_store.remove.call_count == 1
    assert bad_client.aclose.await_count == 1
    assert "session cleanup failed for byok account 'acct#0': registry gone" in caplog.text
    assert "pool store flush failed: flush gone" in caplog.text
    assert "byok cache file delete failed: unlink gone" in caplog.text
    assert "client close failed for byok account 'acct#0': client gone" in caplog.text


async def test_close_pool_defers_a_busy_account(caplog):
    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    client = _FakeClient()
    account = MagicMock()
    account.label = "acct#1"
    account.sessions = MagicMock()
    account.client = client
    account.sem = semaphore
    pool = MagicMock()
    pool.accounts = [account]
    pool.flush = None
    with caplog.at_level(logging.INFO, logger=LOGGER):
        await byok_mod._close_pool(pool)
    assert client.closed == 0
    assert "schedule deferred client close for busy byok account 'acct#1'" in caplog.text
    assert len(byok_mod._deferred_close_tasks) == 1
    semaphore.release()
    pending = list(byok_mod._deferred_close_tasks)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    assert client.closed == 1
    assert byok_mod._deferred_close_tasks == set()


async def test_close_busy_client_closes_after_acquiring():
    semaphore = _FakeSemaphore()
    client = _FakeClient()
    account = MagicMock()
    account.label = "acct#2"
    account.client = client
    await byok_mod._close_busy_client(account, semaphore)
    assert client.closed == 1
    assert semaphore.releases == 1


async def test_close_busy_client_gives_up_after_the_timeout(monkeypatch, caplog):
    semaphore = _FakeSemaphore(acquire_now=False)
    client = _FakeClient()
    account = MagicMock()
    account.label = "acct#3"
    account.client = client

    async def immediate_timeout(_awaitable: Any, timeout: float) -> Any:
        assert timeout == 300
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", _discarding_wait_for(immediate_timeout))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        await byok_mod._close_busy_client(account, semaphore)
    assert client.closed == 0
    assert semaphore.releases == 0
    assert "give up deferred client close for busy byok account 'acct#3'" in caplog.text


async def test_close_busy_client_closes_the_client_and_reraises_on_cancel():
    semaphore = _FakeSemaphore(acquire_now=False)
    client = _FakeClient()
    account = MagicMock()
    account.label = "acct#4"
    account.client = client
    task = asyncio.create_task(byok_mod._close_busy_client(account, semaphore))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert semaphore.releases == 0
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert client.closed == 1


async def test_close_busy_client_reports_a_close_failure(caplog):
    semaphore = _FakeSemaphore()
    client = _FakeClient(close_error=OSError("gone"))
    account = MagicMock()
    account.label = "acct#5"
    account.client = client
    with caplog.at_level(logging.INFO, logger=LOGGER):
        await byok_mod._close_busy_client(account, semaphore)
    assert semaphore.releases == 1
    assert "client close failed for byok account 'acct#5': gone" in caplog.text


async def test_byok_validate_caches_a_rejection_and_logs_it(caplog):
    client = _FakeClient(ok=False)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert await byok_mod._byok_validate("deepseek", "rejected-token", client) is False
    store = app.state.byok_auth["deepseek"]
    stable = byok_mod._byok_stable_id("rejected-token")
    assert list(store) == [stable]
    assert store[stable][0] is False
    assert "byok deepseek api key was rejected upstream" in caplog.text
    assert client.auth_calls == 1

    monkey_client = _FakeClient()
    assert await byok_mod._byok_validate("deepseek", "rejected-token", monkey_client) is False
    assert monkey_client.auth_calls == 0
    assert list(store) == [stable]


async def test_byok_validate_does_not_cache_a_transport_failure(caplog):
    before = byok_mod._auth_indeterminate_count()
    client = _FakeClient(auth_error=OSError("connection reset"))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert await byok_mod._byok_validate("deepseek", "flaky-token", client) is False
    assert app.state.byok_auth["deepseek"] == {}
    assert byok_mod._auth_indeterminate_count() == before + 1
    assert "byok deepseek auth check failed before a verdict, the key is not cached: OSError: connection reset" in caplog.text


async def test_byok_validate_reraises_a_cancellation():
    client = _FakeClient(auth_error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await byok_mod._byok_validate("deepseek", "cancelled-token", client)
    assert app.state.byok_auth["deepseek"] == {}


async def test_byok_validate_evicts_when_the_store_is_full(monkeypatch):
    monkeypatch.setattr(byok_mod, "BYOK_AUTH_LIMIT", 1)
    assert await byok_mod._byok_validate("deepseek", "first", _FakeClient()) is True
    assert await byok_mod._byok_validate("deepseek", "second", _FakeClient()) is True
    store = app.state.byok_auth["deepseek"]
    assert list(store) == [byok_mod._byok_stable_id("second")]


async def test_build_accounts_indexes_accounts_after_skips(caplog):
    clients = {"a": _FakeClient(ok=True), "b": _FakeClient(ok=True), "c": _FakeClient(ok=True)}

    def make_client(token: str) -> Any:
        if token == "b":
            raise RuntimeError("no client for b")
        if token == "c":
            return _FakeClient(ok=False)
        return clients[token]

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        accounts = await byok_mod._build_accounts(
            "deepseek",
            ["a", "b", "c"],
            make_client,
            lambda index, client, _token: _acct(index, client),
        )
    assert [account.index for account in accounts] == [0]
    assert accounts[0].client is clients["a"]
    assert "byok deepseek client unusable, skipping key #1: no client for b" in caplog.text
    assert "byok deepseek token invalid/expired, skipping" in caplog.text


async def test_build_accounts_closes_everything_on_cancellation(monkeypatch):
    clients = {"first": _FakeClient(), "second": _FakeClient()}

    async def validate(_provider: str, token: str, _client: Any) -> bool:
        if token == "first":
            return True
        raise asyncio.CancelledError

    monkeypatch.setattr(byok_mod, "_byok_validate", validate)
    with pytest.raises(asyncio.CancelledError):
        await byok_mod._build_accounts(
            "deepseek",
            ["first", "second"],
            lambda token: clients[token],
            lambda index, client, _token: _acct(index, client),
        )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert clients["first"].closed == 1
    assert clients["second"].closed == 1


def _acct(index: int, client: Any) -> Any:
    return byok_mod.DeepSeekAccount(index, client, stable_id=byok_mod._byok_stable_id(f"key-{index}"))


async def test_build_byok_pool_reports_no_valid_deepseek_key(monkeypatch):
    monkeypatch.setattr(byok_mod, "DeepSeekClient", lambda **_kwargs: _FakeClient(ok=False))
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._build_byok_pool("deepseek", ["bad"], None)
    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == byok_mod._INVALID_KEY_DETAIL.format(provider="deepseek")


async def test_build_byok_pool_builds_a_qwen_pool(monkeypatch):
    monkeypatch.setattr(byok_mod, "QwenClient", lambda **_kwargs: _FakeClient())
    recorder = _RefreshRecorder()
    monkeypatch.setattr(byok_mod, "refresh_provider_models", recorder)
    pool, created = await byok_mod._build_byok_pool("qwen", ["q1", "q2"], None)
    assert created == []
    assert pool.label == "qwen"
    assert [account.index for account in pool.accounts] == [0, 1]
    assert pool.accounts[0].stable_id == byok_mod._byok_stable_id("q1")
    assert recorder.calls == [("qwen", pool.accounts[0].client)]


async def test_build_byok_pool_gives_qwen_its_own_scoped_stores(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path))
    monkeypatch.setattr(byok_mod, "QwenClient", lambda **_kwargs: _FakeClient())
    monkeypatch.setattr(byok_mod, "refresh_provider_models", _RefreshRecorder())
    scope = byok_mod._byok_scope("qwen-cache-key")
    _pool, created = await byok_mod._build_byok_pool("qwen", ["q1"], scope)
    assert [store._path.name for store in created] == [
        f"qwen-sessions-{scope}.json",
        f"qwen-contexts-{scope}.json",
        f"qwen-affinities-{scope}.json",
    ]


async def test_build_byok_pool_reports_no_valid_qwen_key(monkeypatch):
    monkeypatch.setattr(byok_mod, "QwenClient", lambda **_kwargs: _FakeClient(ok=False))
    monkeypatch.setattr(byok_mod, "refresh_provider_models", _RefreshRecorder())
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._build_byok_pool("qwen", ["bad"], None)
    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == byok_mod._INVALID_KEY_DETAIL.format(provider="qwen")


async def test_build_byok_pool_rejects_an_unknown_provider():
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._build_byok_pool("mistral", ["key"], None)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "provider mistral does not accept a caller supplied api key"


async def test_build_byok_pool_builds_gigachat_accounts(monkeypatch, caplog):
    clients = [_FakeClient(ok=True), _FakeClient(ok=True)]

    class _Giga:
        def __init__(self, **_kwargs: Any) -> None:
            self.inner = clients.pop(0)

        async def check_auth(self) -> bool:
            return await self.inner.check_auth()

        async def aclose(self) -> None:
            await self.inner.aclose()

    monkeypatch.setattr(byok_mod, "GigaChatClient", _Giga)
    recorder = _RefreshRecorder()
    monkeypatch.setattr(byok_mod, "refresh_provider_models", recorder)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        pool, created = await byok_mod._build_byok_pool("gigachat", ["g1", "g2"], None)
    assert created == []
    assert pool.label == "gigachat"
    assert [account.index for account in pool.accounts] == [0, 1]
    assert pool.accounts[0].stable_id == byok_mod._byok_stable_id("g1")
    assert recorder.calls == [("gigachat", pool.accounts[0].client)]
    assert "byok gigachat key set received with 2 key(s)" in caplog.text


async def test_build_byok_pool_reports_a_rejected_gigachat_key(monkeypatch):
    monkeypatch.setattr(byok_mod, "GigaChatClient", lambda **_kwargs: _FakeClient(ok=False))
    monkeypatch.setattr(byok_mod, "refresh_provider_models", _RefreshRecorder())
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._build_byok_pool("gigachat", ["bad"], None)
    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == byok_mod._INVALID_KEY_DETAIL.format(provider="gigachat")


async def test_build_byok_pool_reports_an_unreachable_provider(monkeypatch):
    monkeypatch.setattr(byok_mod, "QwenClient", lambda **_kwargs: _FakeClient(auth_error=OSError("dns")))
    monkeypatch.setattr(byok_mod, "refresh_provider_models", _RefreshRecorder())
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._build_byok_pool("qwen", ["key"], None)
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == byok_mod._UNREACHABLE_DETAIL.format(provider="qwen")


async def test_build_byok_pool_gives_the_replacement_the_persisted_stores(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path))
    monkeypatch.setattr(byok_mod, "DeepSeekClient", lambda **_kwargs: _FakeClient())
    monkeypatch.setattr(byok_mod, "refresh_provider_models", _RefreshRecorder())
    cache_key = byok_mod._byok_cache_key(["persisted-token"])
    scope = byok_mod._byok_scope(cache_key)
    assert scope is not None
    (tmp_path / f"deepseek-contexts-{scope}.json").write_text(json.dumps({"s1": ["hello", "world"]}), encoding="utf-8")
    first, created = await byok_mod._build_byok_pool("deepseek", ["persisted-token"], scope)
    assert first.resolve_context(("hello", "world")) == "s1"
    assert [store._path.name for store in created] == [
        f"deepseek-sessions-{scope}.json",
        f"deepseek-contexts-{scope}.json",
        f"deepseek-affinities-{scope}.json",
    ]

    for account in first.accounts:
        account.mark_broken()
    assert first.healthy == []
    app.state.byok_pools["deepseek"][cache_key] = first
    app.state.byok_stores["deepseek"][cache_key] = created

    replacement = await byok_mod._byok_pool("deepseek", ["persisted-token"])
    assert replacement is not first
    assert replacement.resolve_context(("hello", "world")) == "s1"
    assert [account.index for account in replacement.accounts] == [0]
    assert first.accounts[0].client.closed == 1
    assert app.state.byok_pools["deepseek"][cache_key] is replacement
    assert [store._path.name for store in app.state.byok_stores["deepseek"][cache_key]] == [
        f"deepseek-sessions-{scope}.json",
        f"deepseek-contexts-{scope}.json",
        f"deepseek-affinities-{scope}.json",
    ]


async def test_byok_pool_rejects_an_unknown_provider():
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._byok_pool("mistral", [])
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "unknown provider: mistral"


async def test_byok_pool_rejects_more_keys_than_the_limit():
    tokens = [f"key-{index}" for index in range(byok_mod.BYOK_MAX_KEYS + 1)]
    assert byok_mod.BYOK_MAX_KEYS == 16
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._byok_pool("deepseek", tokens)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "too many api keys for deepseek: at most 16 keys per request"


async def test_byok_pool_returns_the_cached_healthy_pool(monkeypatch):
    monkeypatch.setattr(byok_mod, "refresh_provider_models", _RefreshRecorder())
    cached = byok_mod.AccountPool([_acct(0, _FakeClient())])
    other = byok_mod.AccountPool([_acct(0, _FakeClient())])
    cache_key = byok_mod._byok_cache_key(["cached-key"])
    app.state.byok_pools["deepseek"][cache_key] = cached
    app.state.byok_pools["deepseek"][byok_mod._byok_cache_key(["other-key"])] = other
    assert await byok_mod._byok_pool("deepseek", ["cached-key"]) is cached
    assert list(app.state.byok_pools["deepseek"]) == [byok_mod._byok_cache_key(["other-key"]), cache_key]


async def test_byok_pool_dedupes_the_key_list(monkeypatch):
    monkeypatch.setattr(byok_mod, "DeepSeekClient", lambda **_kwargs: _FakeClient())
    monkeypatch.setattr(byok_mod, "refresh_provider_models", _RefreshRecorder())
    pool = await byok_mod._byok_pool("deepseek", ["same", "same", "same"])
    assert len(pool.accounts) == 1


async def test_byok_pool_routes_keyless_providers(monkeypatch):
    alice = object()
    duckai = object()
    monkeypatch.setattr(byok_mod, "_byok_alice_pool", AsyncMock(return_value=alice))
    monkeypatch.setattr(byok_mod, "_byok_duckai_pool", AsyncMock(return_value=duckai))
    assert await byok_mod._byok_pool("alice", []) is alice
    assert await byok_mod._byok_pool("duckai", ["ignored"]) is duckai


async def test_byok_pool_builds_once_for_the_same_key_set(monkeypatch):
    builds = 0

    async def build(_provider: str, tokens: list[str], _scope: str | None) -> tuple[Any, list[Any]]:
        nonlocal builds
        builds += 1
        await asyncio.sleep(0)
        return byok_mod.AccountPool([_acct(0, _FakeClient())]), []

    monkeypatch.setattr(byok_mod, "_build_byok_pool", build)
    first, second = await asyncio.gather(byok_mod._byok_pool("deepseek", ["a"]), byok_mod._byok_pool("deepseek", ["a"]))
    assert builds == 1
    assert first is second


async def test_byok_pool_lets_unrelated_key_sets_proceed_concurrently(monkeypatch):
    started = 0
    release = asyncio.Event()

    async def build(_provider: str, _tokens: list[str], _scope: str | None) -> tuple[Any, list[Any]]:
        nonlocal started
        started += 1
        if started == 2:
            release.set()
        await release.wait()
        return byok_mod.AccountPool([_acct(0, _FakeClient())]), []

    monkeypatch.setattr(byok_mod, "_build_byok_pool", build)
    first, second = await asyncio.wait_for(
        asyncio.gather(byok_mod._byok_pool("deepseek", ["a"]), byok_mod._byok_pool("deepseek", ["b"])),
        2,
    )
    assert started == 2
    assert first is not second


async def test_byok_pool_closes_the_stale_pool_after_the_replacement_is_cached(monkeypatch):
    monkeypatch.setattr(byok_mod, "DeepSeekClient", lambda **_kwargs: _FakeClient())
    monkeypatch.setattr(byok_mod, "refresh_provider_models", _RefreshRecorder())
    stale_client = _FakeClient()
    stale_account = _acct(0, stale_client)
    stale_account.mark_broken()
    stale = byok_mod.AccountPool([stale_account])
    cache_key = byok_mod._byok_cache_key(["reused"])
    app.state.byok_pools["deepseek"][cache_key] = stale
    fresh = await byok_mod._byok_pool("deepseek", ["reused"])
    assert stale_client.closed == 1
    assert fresh is not stale
    assert app.state.byok_pools["deepseek"][cache_key] is fresh
    assert list(app.state.byok_pools["deepseek"]) == [cache_key]


async def test_byok_pool_evicts_the_oldest_key_over_the_limit(monkeypatch):
    monkeypatch.setattr(byok_mod, "BYOK_POOL_LIMIT", 1)
    monkeypatch.setattr(byok_mod, "DeepSeekClient", lambda **_kwargs: _FakeClient())
    monkeypatch.setattr(byok_mod, "refresh_provider_models", _RefreshRecorder())
    removed: list[str] = []
    store = MagicMock()
    store.remove = MagicMock(side_effect=lambda: removed.append("gone"))
    first = await byok_mod._byok_pool("deepseek", ["first-key"])
    first_key = byok_mod._byok_cache_key(["first-key"])
    first_client = first.accounts[0].client
    app.state.byok_stores["deepseek"][first_key] = [store]
    second = await byok_mod._byok_pool("deepseek", ["second-key"])
    assert first is not second
    assert list(app.state.byok_pools["deepseek"]) == [byok_mod._byok_cache_key(["second-key"])]
    assert first_key not in app.state.byok_stores["deepseek"]
    assert removed == ["gone"]
    assert first_client.closed == 1


async def test_byok_alice_accounts_closes_a_rejected_client(monkeypatch):
    created: list[_FakeClient] = []

    class _Alice:
        def __init__(self, **_kwargs: Any) -> None:
            self.inner = _FakeClient(ok=False)
            created.append(self.inner)

        async def check_auth(self) -> bool:
            return await self.inner.check_auth()

        async def aclose(self) -> None:
            await self.inner.aclose()

    monkeypatch.setattr(byok_mod, "AliceClient", _Alice)
    assert await byok_mod._byok_alice_accounts() == []
    assert created[0].closed == 1


async def test_byok_alice_accounts_closes_the_client_on_cancellation(monkeypatch):
    created: list[_FakeClient] = []

    class _Alice:
        def __init__(self, **_kwargs: Any) -> None:
            self.inner = _FakeClient(auth_error=asyncio.CancelledError())
            created.append(self.inner)

        async def check_auth(self) -> bool:
            return await self.inner.check_auth()

        async def aclose(self) -> None:
            await self.inner.aclose()

    monkeypatch.setattr(byok_mod, "AliceClient", _Alice)
    with pytest.raises(asyncio.CancelledError):
        await byok_mod._byok_alice_accounts()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert created[0].closed == 1


def _install_alice(monkeypatch, ok: bool = True) -> list[_FakeClient]:
    created: list[_FakeClient] = []

    class _Alice:
        def __init__(self, **_kwargs: Any) -> None:
            self.inner = _FakeClient(ok=ok)
            created.append(self.inner)

        async def check_auth(self) -> bool:
            await asyncio.sleep(0)
            return await self.inner.check_auth()

        async def aclose(self) -> None:
            await self.inner.aclose()

    monkeypatch.setattr(byok_mod, "AliceClient", _Alice)
    return created


async def test_byok_alice_pool_registers_and_closes_the_replaced_pool(monkeypatch):
    monkeypatch.setattr(byok_mod, "refresh_provider_models", _RefreshRecorder())
    stale_client = _FakeClient()
    stale = byok_mod.AccountPool([_alice_acct(0, stale_client)], label="alice")
    stale.accounts[0].mark_broken()
    byok_mod._ALICE_BYOK_POOL[0] = stale
    created = _install_alice(monkeypatch)
    pool = await byok_mod._byok_alice_pool()
    assert stale_client.closed == 1
    assert byok_mod._ALICE_BYOK_POOL[0] is pool
    assert app.state.byok_alice_pool is pool
    assert app.state.byok_pools["alice"][byok_mod.KEYLESS_POOL_KEY] is pool
    assert pool.label == "alice"
    assert pool.accounts[0].stable_id == "alice"
    assert created[0].auth_calls == 1


async def test_byok_alice_pool_returns_the_healthy_cached_pool(monkeypatch):
    created = _install_alice(monkeypatch)
    cached = byok_mod.AccountPool([_alice_acct(0, _FakeClient())], label="alice")
    byok_mod._ALICE_BYOK_POOL[0] = cached
    assert await byok_mod._byok_alice_pool() is cached
    assert created == []


async def test_byok_alice_pool_builds_once_under_concurrency(monkeypatch):
    monkeypatch.setattr(byok_mod, "refresh_provider_models", _RefreshRecorder())
    created = _install_alice(monkeypatch)
    first, second = await asyncio.gather(byok_mod._byok_alice_pool(), byok_mod._byok_alice_pool())
    assert first is second
    assert len(created) == 1


async def test_byok_alice_pool_reports_an_unreachable_endpoint(monkeypatch):
    _install_alice(monkeypatch, ok=False)
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._byok_alice_pool()
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "alice endpoint is unreachable"


def _alice_acct(index: int, client: Any) -> Any:
    return byok_mod.AliceAccount(index, client, stable_id="alice")


def _duckai_acct(index: int, client: Any) -> Any:
    return byok_mod.DuckAIAccount(index, client, stable_id="duckai")


def _install_duckai(monkeypatch, ok: bool = True) -> list[_FakeClient]:
    created: list[_FakeClient] = []

    class _Duck:
        def __init__(self, **_kwargs: Any) -> None:
            self.inner = _FakeClient(ok=ok)
            created.append(self.inner)

        async def check_auth(self) -> bool:
            await asyncio.sleep(0)
            return await self.inner.check_auth()

        async def aclose(self) -> None:
            await self.inner.aclose()

    monkeypatch.setattr(byok_mod, "DuckAIClient", _Duck)
    return created


async def test_byok_duckai_accounts_closes_a_rejected_client(monkeypatch):
    created = _install_duckai(monkeypatch, ok=False)
    assert await byok_mod._byok_duckai_accounts() == []
    assert created[0].closed == 1


async def test_byok_duckai_accounts_closes_the_client_on_cancellation(monkeypatch):
    created: list[_FakeClient] = []

    class _Duck:
        def __init__(self, **_kwargs: Any) -> None:
            self.inner = _FakeClient(auth_error=asyncio.CancelledError())
            created.append(self.inner)

        async def check_auth(self) -> bool:
            return await self.inner.check_auth()

        async def aclose(self) -> None:
            await self.inner.aclose()

    monkeypatch.setattr(byok_mod, "DuckAIClient", _Duck)
    with pytest.raises(asyncio.CancelledError):
        await byok_mod._byok_duckai_accounts()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert created[0].closed == 1


async def test_byok_duckai_pool_registers_and_closes_the_replaced_pool(monkeypatch):
    monkeypatch.setattr(byok_mod, "refresh_provider_models", _RefreshRecorder())
    stale_client = _FakeClient()
    stale = byok_mod.AccountPool([_duckai_acct(0, stale_client)], label="duckai")
    stale.accounts[0].mark_broken()
    byok_mod._DUCKAI_BYOK_POOL[0] = stale
    created = _install_duckai(monkeypatch)
    pool = await byok_mod._byok_duckai_pool()
    assert stale_client.closed == 1
    assert byok_mod._DUCKAI_BYOK_POOL[0] is pool
    assert app.state.byok_duckai_pool is pool
    assert app.state.byok_pools["duckai"][byok_mod.KEYLESS_POOL_KEY] is pool
    assert pool.accounts[0].stable_id == "duckai"
    assert created[0].auth_calls == 1


async def test_byok_duckai_pool_returns_the_healthy_cached_pool(monkeypatch):
    created = _install_duckai(monkeypatch)
    cached = byok_mod.AccountPool([_duckai_acct(0, _FakeClient())], label="duckai")
    byok_mod._DUCKAI_BYOK_POOL[0] = cached
    assert await byok_mod._byok_duckai_pool() is cached
    assert created == []


async def test_byok_duckai_pool_builds_once_under_concurrency(monkeypatch):
    monkeypatch.setattr(byok_mod, "refresh_provider_models", _RefreshRecorder())
    created = _install_duckai(monkeypatch)
    first, second = await asyncio.gather(byok_mod._byok_duckai_pool(), byok_mod._byok_duckai_pool())
    assert first is second
    assert len(created) == 1


async def test_byok_duckai_pool_reports_a_blocked_endpoint(monkeypatch):
    _install_duckai(monkeypatch, ok=False)
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._byok_duckai_pool()
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == byok_mod.duckai_api.BLOCKED_HINT


async def test_register_keyless_pool_ignores_a_broken_cache(monkeypatch):
    monkeypatch.setattr(app.state, "byok_pools", {"alice": "not-a-dict"})
    pool = byok_mod.AccountPool([], label="alice")
    await byok_mod._register_keyless_pool("alice", pool)
    assert app.state.byok_pools["alice"] == "not-a-dict"


async def test_register_keyless_pool_drops_per_key_pools(monkeypatch):
    app.state.byok_pools["alice"] = {"stale-key": object(), byok_mod.KEYLESS_POOL_KEY: object()}
    pool = byok_mod.AccountPool([], label="alice")
    await byok_mod._register_keyless_pool("alice", pool)
    assert app.state.byok_pools["alice"] == {byok_mod.KEYLESS_POOL_KEY: pool}


def test_byok_caller_id_reads_the_context_var():
    assert byok_mod._byok_caller_id() == ""
    token = byok_mod._CALLER_ID.set(byok_mod._caller_id_for(["b", "a"]))
    try:
        assert byok_mod._byok_caller_id() == byok_mod._caller_id_for(["a", "b"])
        assert byok_mod._byok_caller_id() != byok_mod._caller_id_for(["a"])
    finally:
        byok_mod._CALLER_ID.reset(token)


async def test_byok_pool_for_answers_the_same_detail_for_a_missing_and_a_rejected_key(monkeypatch, caplog):
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(byok_mod, "DeepSeekClient", lambda **_kwargs: _FakeClient(ok=False))
    with pytest.raises(HTTPException) as missing:
        await byok_mod._byok_pool_for("deepseek", _make_request(headers={}))
    with caplog.at_level(logging.WARNING, logger=LOGGER), pytest.raises(HTTPException) as rejected:
        await byok_mod._byok_pool_for("deepseek", _make_request(headers={"x-api-key": "rejected"}))
    assert missing.value.status_code == rejected.value.status_code == 401
    assert missing.value.detail == rejected.value.detail
    assert missing.value.detail == byok_mod._INVALID_KEY_DETAIL.format(provider="deepseek")
    assert "invalid deepseek api key" not in missing.value.detail
    assert "byok deepseek api key was rejected upstream" in caplog.text
    assert app.state.byok_pools["deepseek"] == {}


async def test_byok_pool_for_answers_503_when_the_provider_is_unreachable(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(byok_mod, "DeepSeekClient", lambda **_kwargs: _FakeClient(auth_error=OSError("network down")))
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._byok_pool_for("deepseek", _make_request(headers={"x-api-key": "unreachable"}))
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == byok_mod._UNREACHABLE_DETAIL.format(provider="deepseek")
    assert app.state.byok_pools["deepseek"] == {}


async def test_byok_pool_for_rejects_more_keys_than_the_limit():
    header = ",".join(f"key-{index}" for index in range(byok_mod.BYOK_MAX_KEYS + 1))
    request = _make_request(headers={"x-api-key": header})
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._byok_pool_for("deepseek", request)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "too many api keys for deepseek: at most 16 keys per request"
    assert app.state.byok_pools["deepseek"] == {}


async def test_byok_pool_for_rejects_an_unknown_provider():
    request = _make_request(headers={"Authorization": "Bearer k"})
    with pytest.raises(HTTPException) as excinfo:
        await byok_mod._byok_pool_for("mistral", request)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "unknown provider: mistral"


async def test_byok_pool_for_sets_the_caller_id(monkeypatch):
    monkeypatch.setattr(byok_mod, "DeepSeekClient", lambda **_kwargs: _FakeClient())
    monkeypatch.setattr(byok_mod, "refresh_provider_models", _RefreshRecorder())
    pool = await byok_mod._byok_pool_for("deepseek", _make_request(headers={"Authorization": "Bearer key-one"}))
    assert len(pool.accounts) == 1
    assert byok_mod._byok_caller_id() == byok_mod._caller_id_for(["key-one"])


async def test_byok_pool_for_clears_the_caller_id_for_keyless_providers(monkeypatch):
    monkeypatch.setattr(byok_mod, "_byok_alice_pool", AsyncMock(return_value=object()))
    token = byok_mod._CALLER_ID.set("stale-caller")
    try:
        await byok_mod._byok_pool_for("alice", _make_request(headers={}))
        assert byok_mod._byok_caller_id() == ""
    finally:
        byok_mod._CALLER_ID.reset(token)
