from __future__ import annotations

import asyncio
import base64
import json
import ssl
import time
import uuid
from typing import Any

import httpx
import pytest

import danyapi.api.retry as retry_module
from danyapi.api.retry import MAX_RETRIES, RETRY_BACKOFF_JITTER, RETRY_BACKOFF_MAX_SEC, RETRY_BACKOFF_SEC, RETRYABLE_HTTP_STATUSES
from danyapi.config import settings
from danyapi.gigachat import client as client_mod
from danyapi.gigachat import tls
from danyapi.gigachat.client import (
    AUTH_URL,
    BASE_URL,
    TOKEN_ERROR_STATUSES,
    TOKEN_EXPIRY_BUFFER_SEC,
    TOKEN_LIFETIME_SEC,
    TOKEN_RETRY_LIMIT_SEC,
    USER_AGENT,
    GigaChatClient,
    GigaChatError,
    _expires_at_seconds,
    is_authorization_key,
)

KEY_RAW = b"9f2c1a4e-0b7d-4c8e-9a1b-2f3c4d5e6f70"
KEY = base64.b64encode(KEY_RAW).decode()
FOREVER = 4102444800


class _AsyncioShim:
    def __init__(self, delays: list[float]) -> None:
        self.delays = delays

    def Lock(self) -> asyncio.Lock:
        return asyncio.Lock()

    async def sleep(self, delay: float) -> None:
        self.delays.append(delay)


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now


class _GatedTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.MockTransport, gate: asyncio.Event) -> None:
        self._inner = inner
        self._gate = gate

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/oauth"):
            await self._gate.wait()
        return await self._inner.handle_async_request(request)


def _route(
    token_calls: list[int],
    api_handler: Any,
    token_handler: Any = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/oauth"):
            token_calls.append(1)
            if token_handler is None:
                return httpx.Response(200, json={"access_token": f"tok{len(token_calls)}", "expires_at": FOREVER})
            return token_handler(request, len(token_calls))
        return api_handler(request)

    return httpx.MockTransport(handler)


def _client(transport: httpx.AsyncBaseTransport, timeout: float = 60.0) -> GigaChatClient:
    client = GigaChatClient(key=KEY, timeout=timeout)
    client.http = httpx.AsyncClient(transport=transport, base_url=BASE_URL)
    return client


def _fail(status: int, message: str) -> dict:
    return {"status": status, "message": message}


def _sleep_recorder(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr(client_mod, "asyncio", _AsyncioShim(delays))
    return delays


def _lowest_jitter(low: float, high: float) -> float:
    return low


@pytest.fixture(autouse=True)
def _isolated_tls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tls, "_resolved", [False])


def test_is_authorization_key_accepts_a_36_byte_uuid():
    assert len(KEY_RAW) == 36
    assert str(uuid.UUID(KEY_RAW.decode("ascii"))) == KEY_RAW.decode("ascii")
    assert is_authorization_key(KEY) is True
    assert is_authorization_key(f"\t{KEY}\n") is True
    assert is_authorization_key(base64.b64encode(b"\x7f" + KEY_RAW).decode()) is False


@pytest.mark.parametrize("payload", [b"x" * 35, b"x" * 37, b"", b"x" * 12])
def test_is_authorization_key_rejects_a_wrong_decoded_length(payload: bytes):
    assert is_authorization_key(base64.b64encode(payload).decode()) is False


def test_is_authorization_key_rejects_an_unparsable_uuid():
    assert is_authorization_key(base64.b64encode(b"y" * 36).decode()) is False
    assert is_authorization_key(base64.b64encode(b"z" * 36).decode()) is False


def test_is_authorization_key_rejects_bytes_outside_the_base64_alphabet():
    assert is_authorization_key("not base64!!") is False
    assert is_authorization_key("QU JD") is False
    assert is_authorization_key("ü" * 48) is False


def test_is_authorization_key_rejects_broken_padding():
    assert is_authorization_key("QUJD=") is False
    assert is_authorization_key("AAAAA") is False
    assert is_authorization_key("AB") is False


def test_gigachat_error_reports_auth_statuses():
    assert GigaChatError(401, "Unauthorized").is_auth is True
    assert GigaChatError(403, "Forbidden").is_auth is True
    assert GigaChatError(429, "slow down").is_auth is False
    error = GigaChatError(503, "gateway down")
    assert error.code == 503
    assert error.message == "gateway down"
    assert str(error) == "GigaChat error 503: gateway down"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (-1, 0.0),
        (0.0, 0.0),
        (FOREVER, FOREVER),
        (FOREVER * 1000, FOREVER),
        (str(FOREVER), FOREVER),
        (None, 0.0),
        ("nonsense", 0.0),
        ([], 0.0),
    ],
)
def test_expires_at_seconds_normalizes_every_shape(value, expected):
    assert _expires_at_seconds(value) == expected


async def test_http_is_lazy_verified_and_built_once(monkeypatch):
    calls: list[str] = []

    def _resolve() -> ssl.SSLContext:
        calls.append("resolve")
        return ssl.create_default_context()

    monkeypatch.setattr(client_mod, "resolve_ca", _resolve)
    client = GigaChatClient(key=KEY, timeout=60.0)
    assert calls == []

    first = client.http

    assert calls == ["resolve"]
    assert client.http is first
    assert str(first.base_url) == f"{BASE_URL}/"
    assert first.follow_redirects is True
    assert first.headers["User-Agent"] == USER_AGENT
    assert first.headers["Accept"] == "application/json"
    assert (first.timeout.connect, first.timeout.read) == (60.0, 300.0)
    await client.aclose()
    assert client._http is None


async def test_http_read_timeout_scales_with_a_short_timeout(monkeypatch):
    monkeypatch.setattr(client_mod, "resolve_ca", ssl.create_default_context)
    client = GigaChatClient(key=KEY, timeout=2.0)
    assert client.http.timeout.read == 300.0
    await client.aclose()

    wide = GigaChatClient(key=KEY, timeout=120.0)
    assert wide.http.timeout.read == 600.0
    await wide.aclose()


async def test_aclose_never_builds_a_client_it_did_not_use(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(client_mod, "resolve_ca", lambda: calls.append("resolve") or ssl.create_default_context())
    client = GigaChatClient(key=KEY)

    await client.aclose()

    assert calls == []
    assert client._http is None


async def test_http_setter_replaces_the_stored_client(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(client_mod, "resolve_ca", lambda: calls.append("resolve") or ssl.create_default_context())
    client = GigaChatClient(key=KEY)
    replacement = httpx.AsyncClient()

    client.http = replacement

    assert client.http is replacement
    assert calls == []
    await client.aclose()
    assert client._http is None
    assert replacement.is_closed is True


async def test_lazy_http_writes_the_bundle_next_to_the_configured_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path))
    client = GigaChatClient(key=KEY)

    built = client.http

    assert (tmp_path / "gigachat-ca-bundle.pem").is_file()
    assert client.http is built
    await client.aclose()


async def test_cached_token_raises_the_deferred_error_until_it_expires():
    client = _client(_route([], lambda request: httpx.Response(200, json={})))
    before = time.time()

    client._defer_token(403, "invalid client")

    assert client._token == ""
    assert client._token_expires_at == 0.0
    assert before + TOKEN_RETRY_LIMIT_SEC <= client._token_deferred_until <= time.time() + TOKEN_RETRY_LIMIT_SEC
    assert client._token_error == (403, "invalid client")
    with pytest.raises(GigaChatError) as exc:
        client._cached_token(client._token_deferred_until - 1.0, False)
    assert (exc.value.code, exc.value.message) == (403, "invalid client")
    assert client._cached_token(client._token_deferred_until + 1.0, False) is None


async def test_cached_token_serves_a_fresh_token_and_force_always_misses():
    client = _client(_route([], lambda request: httpx.Response(200, json={})))
    client._token = "tok"
    client._token_expires_at = 2000.0
    edge = 2000.0 - TOKEN_EXPIRY_BUFFER_SEC

    assert client._cached_token(edge - 1.0, False) == "tok"
    assert client._cached_token(edge, False) is None
    assert client._cached_token(edge - 1.0, True) is None


async def test_a_second_caller_reuses_the_token_filled_in_by_the_first(monkeypatch):
    token_calls: list[int] = []
    gate = asyncio.Event()
    transport = _GatedTransport(_route(token_calls, lambda request: httpx.Response(200, json={})), gate)
    client = _client(transport)

    first = asyncio.create_task(client.access_token())
    await asyncio.sleep(0)
    second = asyncio.create_task(client.access_token())
    await asyncio.sleep(0)
    assert len(token_calls) == 0

    gate.set()

    assert await first == "tok1"
    assert await second == "tok1"
    assert len(token_calls) == 1
    assert client._token == "tok1"


async def test_token_transport_failure_is_negatively_cached(monkeypatch):
    token_calls: list[int] = []

    def _down(request: httpx.Request, attempt: int) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(_route(token_calls, lambda request: httpx.Response(200, json={}), _down))
    before = time.time()

    with pytest.raises(GigaChatError) as exc:
        await client.access_token()

    assert (exc.value.code, exc.value.message) == (503, "authorization endpoint transport error: connection refused")
    assert str(exc.value) == "GigaChat error 503: authorization endpoint transport error: connection refused"
    assert client._token_deferred_until >= before + TOKEN_RETRY_LIMIT_SEC

    with pytest.raises(GigaChatError) as again:
        await client.access_token()

    assert (again.value.code, again.value.message) == (503, "authorization endpoint transport error: connection refused")
    assert len(token_calls) == 1


@pytest.mark.parametrize("status", sorted(TOKEN_ERROR_STATUSES))
async def test_token_endpoint_statuses_are_negatively_cached(status: int):
    token_calls: list[int] = []

    def _denied(request: httpx.Request, attempt: int) -> httpx.Response:
        return httpx.Response(status, json=_fail(status, "invalid client"))

    client = _client(_route(token_calls, lambda request: httpx.Response(200, json={}), _denied))

    for _ in range(2):
        with pytest.raises(GigaChatError) as exc:
            await client.access_token()
        assert (exc.value.code, exc.value.message) == (status, "invalid client")
        assert str(exc.value) == f"GigaChat error {status}: invalid client"

    assert len(token_calls) == 1
    assert client._token_error == (status, "invalid client")
    assert client._token_deferred_until > time.time()


async def test_a_token_endpoint_status_outside_the_negatively_cached_set_is_not_deferred():
    token_calls: list[int] = []

    def _broken(request: httpx.Request, attempt: int) -> httpx.Response:
        return httpx.Response(500, json=_fail(500, "upstream exploded"))

    client = _client(_route(token_calls, lambda request: httpx.Response(200, json={}), _broken))

    for _ in range(2):
        with pytest.raises(GigaChatError) as exc:
            await client.access_token()
        assert (exc.value.code, exc.value.message) == (500, "upstream exploded")

    assert len(token_calls) == 2
    assert client._token_deferred_until == 0.0


async def test_non_json_token_response_is_raised_without_deferring():
    token_calls: list[int] = []

    def _html(request: httpx.Request, attempt: int) -> httpx.Response:
        return httpx.Response(200, content=b"<html>gateway</html>")

    client = _client(_route(token_calls, lambda request: httpx.Response(200, json={}), _html))

    for _ in range(2):
        with pytest.raises(GigaChatError) as exc:
            await client.access_token()
        assert (exc.value.code, exc.value.message) == (200, "authorization endpoint returned non-JSON")

    assert len(token_calls) == 2
    assert client._token_deferred_until == 0.0


@pytest.mark.parametrize("payload", [[], "token", 7, {"expires_in": 3600}, {"access_token": ""}, {"access_token": 5}])
async def test_unusable_token_payloads_are_rejected(payload: Any):
    token_calls: list[int] = []
    client = _client(_route(token_calls, lambda request: httpx.Response(200, json={}), lambda request, attempt: httpx.Response(200, json=payload)))

    with pytest.raises(GigaChatError) as exc:
        await client.access_token()

    assert exc.value.code == 200
    assert exc.value.message in {"unexpected authorization payload", "authorization response has no access_token"}
    assert len(token_calls) == 1


async def test_a_token_without_an_expiry_gets_the_default_lifetime():
    token_calls: list[int] = []
    client = _client(
        _route(token_calls, lambda request: httpx.Response(200, json={}), lambda request, attempt: httpx.Response(200, json={"access_token": "tok"}))
    )
    before = time.time()

    assert await client.access_token() == "tok"
    assert before + TOKEN_LIFETIME_SEC <= client._token_expires_at <= time.time() + TOKEN_LIFETIME_SEC
    assert await client.access_token() == "tok"
    assert len(token_calls) == 1


async def test_an_already_expired_access_token_is_negatively_cached():
    token_calls: list[int] = []

    def _stale(request: httpx.Request, attempt: int) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "stale", "expires_at": time.time() + 5.0})

    client = _client(_route(token_calls, lambda request: httpx.Response(200, json={}), _stale))

    for _ in range(2):
        with pytest.raises(GigaChatError) as exc:
            await client.access_token()
        assert (exc.value.code, exc.value.message) == (200, "authorization response returned an already expired access_token")

    assert len(token_calls) == 1
    assert client._token_error == (200, "authorization response returned an already expired access_token")
    assert client._token_deferred_until > time.time()


async def test_invalidate_token_drops_the_cached_token_but_keeps_the_authorization_backoff():
    token_calls: list[int] = []
    calls = {"n": 0}

    def _flaky(request: httpx.Request, attempt: int) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(401, json=_fail(401, "invalid client"))
        return httpx.Response(200, json={"access_token": f"tok{attempt}", "expires_at": FOREVER})

    client = _client(_route(token_calls, lambda request: httpx.Response(200, json={}), _flaky))

    with pytest.raises(GigaChatError):
        await client.access_token()
    deferred_until = client._token_deferred_until
    assert deferred_until > time.time()

    client.invalidate_token()

    assert client._token == ""
    assert client._token_expires_at == 0.0
    assert client._token_deferred_until == deferred_until
    assert client._token_error == (401, "invalid client")
    with pytest.raises(GigaChatError) as blocked:
        await client.access_token()
    assert (blocked.value.code, blocked.value.message) == (401, "invalid client")
    assert len(token_calls) == 1


async def test_a_deferred_401_is_not_cleared_by_an_api_401():
    token_calls: list[int] = []
    api_calls: list[int] = []
    calls = {"n": 0}

    def _flaky(request: httpx.Request, attempt: int) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(401, json=_fail(401, "invalid client"))
        return httpx.Response(200, json={"access_token": f"tok{attempt}", "expires_at": FOREVER})

    def _unauthorized(request: httpx.Request) -> httpx.Response:
        api_calls.append(1)
        return httpx.Response(401, json=_fail(401, "Unauthorized"))

    client = _client(_route(token_calls, _unauthorized, _flaky))

    with pytest.raises(GigaChatError) as blocked:
        await client.fetch_models()

    assert (blocked.value.code, blocked.value.message) == (401, "invalid client")
    assert len(token_calls) == 1
    assert len(api_calls) == 0

    client.invalidate_token()

    with pytest.raises(GigaChatError) as exc:
        await client.fetch_models()

    assert (exc.value.code, exc.value.message) == (401, "invalid client")
    assert len(token_calls) == 1
    assert len(api_calls) == 0


async def test_a_token_inside_the_expiry_buffer_is_refreshed_and_one_outside_it_is_served_from_cache(monkeypatch):
    clock = _Clock(1000.0)
    monkeypatch.setattr(client_mod, "time", clock)
    token_calls: list[int] = []
    payloads = [
        {"access_token": "tok-1", "expires_at": 1090.0},
        {"access_token": "tok-2", "expires_at": 2000.0},
    ]
    client = _client(
        _route(token_calls, lambda request: httpx.Response(200, json={}), lambda request, attempt: httpx.Response(200, json=payloads[attempt - 1]))
    )

    assert await client.access_token() == "tok-1"
    assert client._token_expires_at == 1090.0
    assert await client.access_token() == "tok-1"
    assert len(token_calls) == 1

    clock.now = 1060.0

    assert await client.access_token() == "tok-2"
    assert len(token_calls) == 2
    assert client._token == "tok-2"


@pytest.mark.parametrize("status", sorted(RETRYABLE_HTTP_STATUSES))
async def test_retryable_statuses_are_retried_until_the_api_answers(monkeypatch, status: int):
    monkeypatch.setattr(client_mod, "_retry_delay", lambda attempt: 0.0)
    token_calls: list[int] = []
    api_calls: list[int] = []

    def _flaky(request: httpx.Request) -> httpx.Response:
        api_calls.append(1)
        if len(api_calls) == 1:
            return httpx.Response(status, json=_fail(status, "later"))
        return httpx.Response(200, json={"object": "list", "data": [{"id": "GigaChat"}]})

    client = _client(_route(token_calls, _flaky))

    assert await client.fetch_models() == [{"id": "GigaChat"}]
    assert len(api_calls) == 2
    assert len(token_calls) == 1


async def test_exhausted_retries_surface_the_upstream_error(monkeypatch):
    monkeypatch.setattr(client_mod, "_retry_delay", lambda attempt: 0.0)
    token_calls: list[int] = []
    api_calls: list[int] = []

    def _down(request: httpx.Request) -> httpx.Response:
        api_calls.append(1)
        return httpx.Response(503, json=_fail(503, "upstream down"))

    client = _client(_route(token_calls, _down))

    with pytest.raises(GigaChatError) as exc:
        await client.fetch_models()

    assert (exc.value.code, exc.value.message) == (503, "upstream down")
    assert len(api_calls) == MAX_RETRIES + 1
    assert len(token_calls) == 1


@pytest.mark.parametrize(("header", "expected"), [("120", RETRY_BACKOFF_MAX_SEC), ("3", 3.0), (None, RETRY_BACKOFF_SEC * (1 - RETRY_BACKOFF_JITTER))])
async def test_retry_after_drives_the_sleep_before_the_retry(monkeypatch, header, expected):
    monkeypatch.setattr(retry_module, "RETRY_BACKOFF_SEC", RETRY_BACKOFF_SEC)
    monkeypatch.setattr(retry_module.random, "uniform", _lowest_jitter)
    delays = _sleep_recorder(monkeypatch)
    token_calls: list[int] = []
    api_calls: list[int] = []

    def _flaky(request: httpx.Request) -> httpx.Response:
        api_calls.append(1)
        if len(api_calls) == 1:
            headers = {} if header is None else {"Retry-After": header}
            return httpx.Response(429, json=_fail(429, "slow down"), headers=headers)
        return httpx.Response(200, json={"object": "list", "data": []})

    client = _client(_route(token_calls, _flaky))

    assert await client.fetch_models() == []
    assert delays == [expected]
    assert len(api_calls) == 2
    assert len(token_calls) == 1


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({}, RETRY_BACKOFF_SEC * (1 - RETRY_BACKOFF_JITTER)),
        ({"Retry-After": "120"}, RETRY_BACKOFF_MAX_SEC),
        ({"Retry-After": "2.5"}, 2.5),
        ({"Retry-After": "0"}, RETRY_BACKOFF_SEC * (1 - RETRY_BACKOFF_JITTER)),
        ({"Retry-After": "-1"}, RETRY_BACKOFF_SEC * (1 - RETRY_BACKOFF_JITTER)),
        ({"Retry-After": "later"}, RETRY_BACKOFF_SEC * (1 - RETRY_BACKOFF_JITTER)),
        ({"Retry-After": " 3 "}, 3.0),
    ],
)
def test_retry_after_seconds_clones_or_falls_back_to_the_backoff(monkeypatch, headers, expected):
    monkeypatch.setattr(retry_module, "RETRY_BACKOFF_SEC", RETRY_BACKOFF_SEC)
    monkeypatch.setattr(retry_module.random, "uniform", _lowest_jitter)
    assert client_mod._retry_delay(1) == RETRY_BACKOFF_SEC * (1 - RETRY_BACKOFF_JITTER)
    assert client_mod._retry_after_seconds(httpx.Response(429, headers=headers)) == expected


async def test_401_is_refreshed_once_and_then_returned(monkeypatch):
    monkeypatch.setattr(client_mod, "_retry_delay", lambda attempt: 0.0)
    token_calls: list[int] = []
    api_calls: list[int] = []

    def _unauthorized(request: httpx.Request) -> httpx.Response:
        api_calls.append(1)
        return httpx.Response(401, json=_fail(401, "Unauthorized"))

    client = _client(_route(token_calls, _unauthorized))

    with pytest.raises(GigaChatError) as exc:
        await client.fetch_models()

    assert (exc.value.code, exc.value.message) == (401, "Unauthorized")
    assert len(api_calls) == 2
    assert len(token_calls) == 2
    assert client._token == "tok2"


async def test_403_is_also_treated_as_an_auth_failure(monkeypatch):
    token_calls: list[int] = []
    api_calls: list[int] = []

    def _forbidden(request: httpx.Request) -> httpx.Response:
        api_calls.append(1)
        return httpx.Response(403, json=_fail(403, "Forbidden"))

    client = _client(_route(token_calls, _forbidden))

    with pytest.raises(GigaChatError) as exc:
        await client.fetch_models()

    assert (exc.value.code, exc.value.message) == (403, "Forbidden")
    assert len(api_calls) == 2
    assert len(token_calls) == 2


async def test_a_json_request_against_a_non_json_body_raises(monkeypatch):
    token_calls: list[int] = []
    client = _client(_route(token_calls, lambda request: httpx.Response(200, content=b"<html>maintenance</html>")))

    with pytest.raises(GigaChatError) as exc:
        await client.fetch_models()

    assert (exc.value.code, exc.value.message) == (200, "unexpected non-JSON response from /models")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [([], []), ("nope", []), ({"data": "nope"}, []), ({"data": ["x", {"id": "GigaChat"}]}, [{"id": "GigaChat"}])],
)
async def test_fetch_models_tolerates_unexpected_payload_shapes(payload: Any, expected: list[dict]):
    token_calls: list[int] = []
    client = _client(_route(token_calls, lambda request: httpx.Response(200, json=payload)))

    assert await client.fetch_models() == expected


async def test_check_auth_reports_false_for_a_non_object_payload():
    token_calls: list[int] = []
    client = _client(_route(token_calls, lambda request: httpx.Response(200, json=[])))

    assert await client.check_auth() is False


async def test_check_auth_reports_false_on_a_transport_failure(monkeypatch):
    monkeypatch.setattr(client_mod, "_retry_delay", lambda attempt: 0.0)
    token_calls: list[int] = []
    api_calls: list[int] = []

    def _down(request: httpx.Request) -> httpx.Response:
        api_calls.append(1)
        return httpx.Response(500, text="boom")

    client = _client(_route(token_calls, _down))

    assert await client.check_auth() is False
    assert len(api_calls) == MAX_RETRIES + 1


@pytest.mark.parametrize("payload", [[], "nope", 7])
async def test_upload_file_rejects_a_non_object_payload(payload: Any):
    token_calls: list[int] = []
    seen: list[httpx.Request] = []

    def _files(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload)

    client = _client(_route(token_calls, _files))

    with pytest.raises(GigaChatError) as exc:
        await client.upload_file("a.png", b"data", "image/png")

    assert (exc.value.code, exc.value.message) == (200, "unexpected file upload payload")
    assert seen[0].headers["content-type"].startswith("multipart/form-data; boundary=")
    assert b'name="purpose"' in seen[0].content
    assert b'filename="a.png"' in seen[0].content


async def test_upload_file_returns_the_id_from_the_response():
    token_calls: list[int] = []
    client = _client(_route(token_calls, lambda request: httpx.Response(200, json={"object": "file", "id": "file-uuid"})))

    assert await client.upload_file("a.png", b"data", "image/png", purpose="vision") == "file-uuid"


@pytest.mark.parametrize("payload", [{"object": "file"}, {"id": ""}, {"id": 7}])
async def test_upload_file_without_a_usable_id_raises(payload: Any):
    token_calls: list[int] = []
    client = _client(_route(token_calls, lambda request: httpx.Response(200, json=payload)))

    with pytest.raises(GigaChatError) as exc:
        await client.upload_file("a.png", b"data", "image/png")

    assert (exc.value.code, exc.value.message) == (200, "file upload returned no id")


async def test_the_auth_post_targets_the_oauth_endpoint_with_basic_auth():
    token_calls: list[int] = []
    seen: list[httpx.Request] = []

    def _models(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"object": "list", "data": []})

    client = _client(_route(token_calls, _models))
    await client.fetch_models()

    assert str(seen[0].url) == f"{BASE_URL}/models"
    assert seen[0].headers["Authorization"] == "Bearer tok1"
    assert seen[0].headers["Content-Type"] == "application/json"
    assert uuid.UUID(seen[0].headers["X-Request-ID"])
    assert client._auth_headers()["Authorization"] == f"Basic {KEY}"
    assert client._auth_headers()["Content-Type"] == "application/x-www-form-urlencoded"
    assert client._api_headers("t", {"X-Extra": "1"})["X-Extra"] == "1"
    assert "Content-Type" not in client._api_headers("t", json_body=False)
    assert client._request_id() != client._request_id()


async def test_check_auth_true_for_a_populated_model_list():
    token_calls: list[int] = []
    client = _client(_route(token_calls, lambda request: httpx.Response(200, json={"object": "list", "data": [{"id": "GigaChat"}]})))

    assert await client.check_auth() is True


async def test_multipart_and_stream_requests_carry_the_expected_headers():
    token_calls: list[int] = []
    seen: list[httpx.Request] = []

    def _chat(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"data: {}\n\n")

    client = _client(_route(token_calls, _chat))
    response = await client.chat({"messages": [{"role": "user", "content": "hi"}], "stream": True}, "GigaChat-2-Max")
    await response.aclose()

    assert seen[0].headers["Accept"] == "text/event-stream"
    assert seen[0].headers["X-Accel-Buffering"] == "no"
    assert json.loads(seen[0].content) == {
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "model": "GigaChat-2-Max",
    }
    assert client._auth_headers()["RqUID"]
    assert AUTH_URL == "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
