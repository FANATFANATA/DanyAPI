import asyncio
import base64
import logging
import socket
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import Request

import danyapi.api.images as images_mod
from danyapi.api.images import (
    IMAGE_DOWNLOAD_CONCURRENCY,
    IMAGE_FETCH_REDIRECTS,
    IMAGE_FETCH_TIMEOUT_SEC,
    MAX_IMAGE_DIM,
    MIN_IMAGE_DIM,
    QWEN_UNAVAILABLE_MESSAGE,
    app,
)
from danyapi.api.schemas import ImageGenerationRequest
from danyapi.qwen import api as qwen_api

PUBLIC_ADDRESS = "93.184.216.34"


def _request(path: str = "/v1/images/generations") -> Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 5555),
            "server": ("testserver", 80),
            "scheme": "http",
        },
        receive=receive,
    )


class FakeUpload:
    def __init__(self, data: bytes, content_type: str | None = "image/png", filename: str = "a.png") -> None:
        self.data = data
        self.content_type = content_type
        self.filename = filename
        self.bytes_served = 0

    async def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.data) - self.bytes_served
        chunk = self.data[self.bytes_served : self.bytes_served + size]
        self.bytes_served += len(chunk)
        return chunk


class FakeImageResponse:
    def __init__(self, status: int = 200, headers: dict | None = None, chunks=(), url: str = "https://a.example/i.png") -> None:
        self.status_code = status
        self.headers = headers or {}
        self.url = httpx.URL(url)
        self._chunks = list(chunks)

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _StreamContext:
    def __init__(self, response: FakeImageResponse) -> None:
        self._response = response

    async def __aenter__(self) -> FakeImageResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeImageClient:
    def __init__(self, responses=()) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, str, dict]] = []

    def stream(self, method: str, url: str, **kwargs):
        self.requests.append((method, url, kwargs))
        return _StreamContext(self.responses.pop(0))


class LockSpy:
    def __init__(self) -> None:
        self.entered = 0

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.fixture(autouse=True)
def _clean_image_state():
    saved = {attr: getattr(app.state, attr, None) for attr in ("qwen_pool", "http_client")}
    app.state.qwen_pool = None
    app.state.http_client = None
    yield
    for attr, value in saved.items():
        setattr(app.state, attr, value)


@pytest.fixture
def public_dns(monkeypatch):
    def resolve(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_ADDRESS, 0))]

    monkeypatch.setattr(images_mod.socket, "getaddrinfo", resolve)
    return resolve


def _resolve_to(address: str):
    def resolve(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))]

    return resolve


def _account():
    return SimpleNamespace(sem=LockSpy(), index=0)


def _install_collect_image(monkeypatch, results):
    calls: list[dict] = []

    async def fake(**kwargs):
        calls.append(kwargs)
        return results[len(calls) - 1]

    monkeypatch.setattr(qwen_api, "collect_image", fake)
    return calls


def _pool(account=None):
    account = account or _account()
    pool = MagicMock()
    pool.acquire = AsyncMock(return_value=(account, None))
    return pool, account


def _png(size=(32, 32), fmt: str = "PNG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (255, 0, 0)).save(buffer, format=fmt)
    return buffer.getvalue()


def test_parse_image_size_accepts_every_separator():
    assert images_mod._parse_image_size(None) is None
    assert images_mod._parse_image_size("   ") is None
    assert images_mod._parse_image_size("1152x2048") == (1152, 2048)
    assert images_mod._parse_image_size("1152*2048") == (1152, 2048)
    assert images_mod._parse_image_size("1152\u00d72048") == (1152, 2048)
    assert images_mod._parse_image_size("1152, 2048") == (1152, 2048)
    assert images_mod._parse_image_size(f"{MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}") == (MIN_IMAGE_DIM, MIN_IMAGE_DIM)
    assert images_mod._parse_image_size(f"{MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}") == (MAX_IMAGE_DIM, MAX_IMAGE_DIM)


def test_parse_image_size_rejects_garbage_with_the_exact_message():
    with pytest.raises(HTTPException) as excinfo:
        images_mod._parse_image_size("abc")
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "invalid size 'abc': expected WIDTHxHEIGHT (e.g. 1152x2048 or 1152*2048)"


def test_parse_image_size_rejects_out_of_range_dimensions():
    with pytest.raises(HTTPException) as excinfo:
        images_mod._parse_image_size("9999x9999")
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "size out of range: both dimensions must be within 16..8192"


def test_resize_is_skipped_without_dimensions():
    payload = b"raw"
    assert images_mod._resize_image_bytes(payload, None) is payload


def test_resize_returns_a_png_of_the_requested_size():
    resized = images_mod._resize_image_bytes(_png(), (16, 24))
    assert Image.open(BytesIO(resized)).size == (16, 24)


def test_resize_converts_jpeg_to_rgb():
    buffer = BytesIO()
    Image.new("CMYK", (32, 32), (10, 20, 30, 40)).save(buffer, format="JPEG")
    resized = images_mod._resize_image_bytes(buffer.getvalue(), (16, 16))
    opened = Image.open(BytesIO(resized))
    assert opened.format == "JPEG"
    assert opened.mode == "RGB"
    assert opened.size == (16, 16)


def test_resize_failure_returns_the_original_and_logs(caplog):
    payload = b"not an image"
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        assert images_mod._resize_image_bytes(payload, (16, 16)) is payload
    assert "image resize to (16, 16) failed, returning original" in caplog.text


def test_host_is_public_accepts_a_routable_address(public_dns):
    assert images_mod._host_is_public("a.example") is True


def test_host_is_public_rejects_private_link_local_and_special_ranges():
    for address in ("10.0.0.1", "192.168.1.1", "169.254.169.254", "127.0.0.1", "0.0.0.0", "224.0.0.1", "240.0.0.1", "::1"):
        assert images_mod._host_is_public("blocked.example") is False, address


def test_host_is_public_rejects_unresolvable_hosts(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("Name or service not known")

    monkeypatch.setattr(images_mod.socket, "getaddrinfo", boom)
    assert images_mod._host_is_public("nope.invalid") is False


def test_host_is_public_rejects_garbage_results(monkeypatch):
    monkeypatch.setattr(images_mod.socket, "getaddrinfo", lambda *a, **k: [])
    assert images_mod._host_is_public("empty.example") is False
    monkeypatch.setattr(images_mod.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ())])
    assert images_mod._host_is_public("short.example") is False
    monkeypatch.setattr(images_mod.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("not-an-ip", 0))])
    assert images_mod._host_is_public("text.example") is False


async def test_check_image_url_refuses_a_non_http_scheme(public_dns):
    with pytest.raises(ValueError) as excinfo:
        await images_mod._check_image_url("ftp://a.example/i.png")
    assert str(excinfo.value) == "unsupported image url scheme 'ftp'"


async def test_check_image_url_refuses_a_url_without_a_host(public_dns):
    with pytest.raises(ValueError) as excinfo:
        await images_mod._check_image_url("http:///i.png")
    assert str(excinfo.value) == "image url has no host"


async def test_check_image_url_refuses_a_metadata_address(monkeypatch):
    monkeypatch.setattr(images_mod.socket, "getaddrinfo", _resolve_to("169.254.169.254"))
    with pytest.raises(ValueError) as excinfo:
        await images_mod._check_image_url("http://metadata.example/latest/meta-data/")
    assert str(excinfo.value) == "refusing to fetch image from the non-public address metadata.example"


async def test_check_image_url_refuses_a_private_address(monkeypatch):
    monkeypatch.setattr(images_mod.socket, "getaddrinfo", _resolve_to("10.1.2.3"))
    with pytest.raises(ValueError) as excinfo:
        await images_mod._check_image_url("http://internal.example/i.png")
    assert str(excinfo.value) == "refusing to fetch image from the non-public address internal.example"


async def test_download_image_returns_the_body_and_disables_redirect_following(public_dns):
    client = FakeImageClient([FakeImageResponse(chunks=[b"ab", b"cd"])])
    assert await images_mod._download_image(client, "https://a.example/i.png") == b"abcd"
    method, url, kwargs = client.requests[0]
    assert method == "GET"
    assert url == "https://a.example/i.png"
    assert kwargs["follow_redirects"] is False
    assert kwargs["timeout"] == IMAGE_FETCH_TIMEOUT_SEC


async def test_download_image_follows_a_redirect_and_revalidates_the_target(public_dns, monkeypatch):
    resolved: list[str] = []

    def resolve(host, *args, **kwargs):
        resolved.append(host)
        if host == "b.example":
            return [(2, 1, 6, "", ("169.254.169.254", 0))]
        return [(2, 1, 6, "", (PUBLIC_ADDRESS, 0))]

    monkeypatch.setattr(images_mod.socket, "getaddrinfo", resolve)
    client = FakeImageClient([FakeImageResponse(status=302, headers={"location": "https://b.example/i.png"}, url="https://a.example/i.png")])
    with pytest.raises(ValueError) as excinfo:
        await images_mod._download_image(client, "https://a.example/i.png")
    assert str(excinfo.value) == "refusing to fetch image from the non-public address b.example"
    assert resolved == ["a.example", "b.example"]
    assert [request[1] for request in client.requests] == ["https://a.example/i.png"]


async def test_download_image_refuses_a_redirect_without_a_location(public_dns):
    client = FakeImageClient([FakeImageResponse(status=302, headers={})])
    with pytest.raises(ValueError) as excinfo:
        await images_mod._download_image(client, "https://a.example/i.png")
    assert str(excinfo.value) == "image redirect without a location header"
    assert len(client.requests) == 1


async def test_download_image_rejects_a_non_200_status(public_dns):
    client = FakeImageClient([FakeImageResponse(status=500)])
    with pytest.raises(ValueError) as excinfo:
        await images_mod._download_image(client, "https://a.example/i.png")
    assert str(excinfo.value) == "image download failed with status 500"


async def test_download_image_hop_limit_is_exact(public_dns):
    client = FakeImageClient(
        [
            FakeImageResponse(status=302, headers={"location": "/n1"}, url="https://a.example/i.png"),
            FakeImageResponse(status=302, headers={"location": "/n2"}, url="https://a.example/n1"),
            FakeImageResponse(status=302, headers={"location": "/n3"}, url="https://a.example/n2"),
            FakeImageResponse(status=302, headers={"location": "/n4"}, url="https://a.example/n3"),
            FakeImageResponse(status=302, headers={"location": "/n5"}, url="https://a.example/n4"),
        ]
    )
    with pytest.raises(ValueError) as excinfo:
        await images_mod._download_image(client, "https://a.example/i.png")
    assert str(excinfo.value) == "image download exceeded the redirect limit"
    assert len(client.requests) == IMAGE_FETCH_REDIRECTS + 1


async def test_download_image_succeeds_on_the_last_allowed_hop(public_dns):
    client = FakeImageClient(
        [
            FakeImageResponse(status=302, headers={"location": "/n1"}, url="https://a.example/i.png"),
            FakeImageResponse(status=302, headers={"location": "/n2"}, url="https://a.example/n1"),
            FakeImageResponse(status=302, headers={"location": "/n3"}, url="https://a.example/n2"),
            FakeImageResponse(chunks=[b"ok"], url="https://a.example/n3"),
        ]
    )
    assert await images_mod._download_image(client, "https://a.example/i.png") == b"ok"
    assert len(client.requests) == IMAGE_FETCH_REDIRECTS + 1


async def test_download_image_aborts_at_the_size_cap(public_dns, monkeypatch):
    monkeypatch.setattr(images_mod, "MAX_FILE_SIZE", 1024)
    client = FakeImageClient([FakeImageResponse(chunks=[b"x" * 600, b"x" * 600])])
    with pytest.raises(ValueError) as excinfo:
        await images_mod._download_image(client, "https://a.example/i.png")
    assert str(excinfo.value) == "image exceeds 0 MB"


async def test_image_http_client_reuses_the_application_client():
    sentinel = MagicMock()
    app.state.http_client = sentinel
    assert await images_mod._image_http_client() is sentinel


async def test_image_http_client_is_created_once_and_follows_redirects():
    app.state.http_client = None
    first = await images_mod._image_http_client()
    second = await images_mod._image_http_client()
    try:
        assert first is second
        assert app.state.http_client is first
        assert first.follow_redirects is True
    finally:
        await first.aclose()


async def test_b64encode_matches_standard_base64():
    payload = b"abc"
    assert await images_mod._b64encode(payload) == base64.b64encode(payload).decode()
    big = b"z" * (images_mod._ASYNC_B64_THRESHOLD + 1)
    assert await images_mod._b64encode(big) == base64.b64encode(big).decode()


async def test_image_markdown_embeds_a_data_uri():
    assert await images_mod._image_markdown(b"png", "image/png") == "![image](data:image/png;base64,cG5n)"
    assert await images_mod._image_markdown(b"png", "") == "![image](data:image/png;base64,cG5n)"


async def test_read_upload_joins_chunks_and_normalises_the_content_type():
    upload = FakeUpload(b"x" * (images_mod.UPLOAD_CHUNK_SIZE * 2 + 5), content_type="image/png; charset=binary")
    content, content_type = await images_mod._read_upload(upload)
    assert len(content) == images_mod.UPLOAD_CHUNK_SIZE * 2 + 5
    assert content_type == "image/png"
    assert upload.bytes_served == images_mod.UPLOAD_CHUNK_SIZE * 2 + 5


async def test_read_upload_defaults_the_content_type():
    assert await images_mod._read_upload(FakeUpload(b"a", content_type=None)) == (b"a", "application/octet-stream")
    assert await images_mod._read_upload(FakeUpload(b"a", content_type="   ")) == (b"a", "application/octet-stream")
    assert await images_mod._read_upload(FakeUpload(b"", content_type=None)) == (b"", "application/octet-stream")


async def test_read_upload_rejects_without_draining_the_upload(monkeypatch):
    monkeypatch.setattr(images_mod, "MAX_FILE_SIZE", 1000)
    upload = FakeUpload(b"y" * 5000)
    with pytest.raises(HTTPException) as excinfo:
        await images_mod._read_upload(upload)
    assert excinfo.value.status_code == 413
    assert excinfo.value.detail == "uploaded file exceeds 0 MB limit"
    assert upload.bytes_served == 1001


def test_image_edit_req_joins_prompt_image_and_mask():
    image_md = "![image](data:image/png;base64,a)"
    mask_md = "![image](data:image/png;base64,b)"
    assert images_mod._image_edit_req("  make it blue  ", image_md, None) == f"make it blue\n{image_md}"
    assert images_mod._image_edit_req("   ", image_md, None) == image_md
    assert images_mod._image_edit_req("p", image_md, mask_md) == f"p\n{image_md}\n{mask_md}"


def test_image_request_maps_validation_errors_to_400():
    req = images_mod._image_request(model="qwen-image-gen", prompt="p", n=1)
    assert req.n == 1
    assert req.response_format == "url"
    for bad_n in (0, 99):
        with pytest.raises(HTTPException) as excinfo:
            images_mod._image_request(model="qwen-image-gen", prompt="p", n=bad_n)
        assert excinfo.value.status_code == 400
        assert excinfo.value.detail == f"invalid request body: n: Input should be {'greater than or equal to 1' if bad_n == 0 else 'less than or equal to 4'}"
    with pytest.raises(HTTPException) as excinfo:
        images_mod._image_request(model="qwen-image-gen")
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "invalid request body: prompt: Field required"


async def test_image_pool_without_configuration_is_503():
    with pytest.raises(HTTPException) as excinfo:
        await images_mod._image_pool(_request())
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == QWEN_UNAVAILABLE_MESSAGE


async def test_image_pool_uses_the_application_pool():
    pool, _account = _pool()
    app.state.qwen_pool = pool
    assert await images_mod._image_pool(_request()) is pool


async def test_image_pool_in_byok_mode_asks_the_byok_manager(monkeypatch):
    pool, _account = _pool()
    byok_pool_for = AsyncMock(return_value=pool)
    monkeypatch.setattr(images_mod, "_byok_mode", lambda: True)
    monkeypatch.setattr(images_mod, "_byok_pool_for", byok_pool_for)
    request = _request()
    assert await images_mod._image_pool(request) is pool
    assert byok_pool_for.await_args.args[0] == "qwen"
    assert byok_pool_for.await_args.args[1] is request


async def test_image_generations_without_a_pool_is_503():
    req = ImageGenerationRequest(prompt="a cat")
    with pytest.raises(HTTPException) as excinfo:
        await images_mod._image_generations(req)
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == QWEN_UNAVAILABLE_MESSAGE


async def test_size_without_b64_json_is_400():
    req = ImageGenerationRequest(prompt="a cat", size="512x512")
    with pytest.raises(HTTPException) as excinfo:
        await images_mod._image_generations(req, MagicMock())
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "size requires response_format=b64_json"


async def test_url_response_format_never_touches_the_network(monkeypatch, public_dns):
    pool, account = _pool()
    step = {"image_urls": ["https://a.example/i.png"], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}
    calls = _install_collect_image(monkeypatch, [{**step, "session_id": "s9"}, dict(step)])
    client = FakeImageClient()
    monkeypatch.setattr(images_mod, "_image_http_client", AsyncMock(return_value=client))
    req = ImageGenerationRequest(prompt="a cat", n=2)
    result = await images_mod._image_generations(req, pool)
    assert result["data"] == [{"url": "https://a.example/i.png"}, {"url": "https://a.example/i.png"}]
    assert result["session_id"] == "s9"
    assert result["usage"] == {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}
    assert isinstance(result["created"], int)
    assert client.requests == []
    assert len(calls) == 2
    assert calls[0]["prompt"] == "a cat"
    assert calls[1]["existing_sid"] == "s9"
    assert calls[0]["lock"] is account.sem


async def test_b64_json_response_downloads_and_encodes(monkeypatch, public_dns):
    pool, _account = _pool()
    _install_collect_image(monkeypatch, [{"image_urls": ["https://a.example/i.png"], "usage": None}])
    client = FakeImageClient([FakeImageResponse(chunks=[b"ab", b"cd"])])
    monkeypatch.setattr(images_mod, "_image_http_client", AsyncMock(return_value=client))
    req = ImageGenerationRequest(prompt="a cat", response_format="b64_json")
    result = await images_mod._image_generations(req, pool)
    assert result["data"] == [{"b64_json": base64.b64encode(b"abcd").decode()}]
    assert result["usage"] is None
    assert len(client.requests) == 1


async def test_b64_json_with_size_resizes_before_encoding(monkeypatch, public_dns):
    pool, _account = _pool()
    _install_collect_image(monkeypatch, [{"image_urls": ["https://a.example/i.png"], "usage": None}])
    client = FakeImageClient([FakeImageResponse(chunks=[_png((32, 32))])])
    monkeypatch.setattr(images_mod, "_image_http_client", AsyncMock(return_value=client))
    req = ImageGenerationRequest(prompt="a cat", response_format="b64_json", size="16x24")
    result = await images_mod._image_generations(req, pool)
    assert Image.open(BytesIO(base64.b64decode(result["data"][0]["b64_json"]))).size == (16, 24)


async def test_b64_json_download_failure_is_502_and_never_falls_back_to_url(monkeypatch, caplog):
    pool, _account = _pool()
    _install_collect_image(monkeypatch, [{"image_urls": ["https://a.example/i.png"], "usage": None}])
    client = FakeImageClient()
    monkeypatch.setattr(images_mod, "_image_http_client", AsyncMock(return_value=client))
    monkeypatch.setattr(images_mod.socket, "getaddrinfo", _resolve_to("169.254.169.254"))
    req = ImageGenerationRequest(prompt="a cat", response_format="b64_json")
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        with pytest.raises(HTTPException) as excinfo:
            await images_mod._image_generations(req, pool)
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "image download failed"
    assert "refusing to fetch image from the non-public address a.example" in caplog.text
    assert client.requests == []


async def test_image_generations_without_urls_is_502(monkeypatch, public_dns):
    pool, _account = _pool()
    _install_collect_image(monkeypatch, [{"image_urls": [], "usage": None}])
    monkeypatch.setattr(images_mod, "_image_http_client", AsyncMock(return_value=FakeImageClient()))
    with pytest.raises(HTTPException) as excinfo:
        await images_mod._image_generations(ImageGenerationRequest(prompt="a cat"), pool)
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "image generation returned no data"


async def test_image_generations_translates_a_busy_pool_to_429(monkeypatch):
    from danyapi.accounts import AccountPoolBusy

    account = _account()

    async def busy(**kwargs):
        raise AccountPoolBusy()

    monkeypatch.setattr(qwen_api, "collect_image", busy)
    monkeypatch.setattr(images_mod, "_image_http_client", AsyncMock(return_value=FakeImageClient()))
    pool = MagicMock()
    pool.acquire = AsyncMock(return_value=(account, None))
    with pytest.raises(HTTPException) as excinfo:
        await images_mod._image_generations(ImageGenerationRequest(prompt="a cat"), pool)
    assert excinfo.value.status_code == 429
    assert excinfo.value.detail == "all accounts are busy, try again later"


async def test_image_generations_limits_concurrent_downloads(monkeypatch, public_dns):
    pool, _account = _pool()
    _install_collect_image(monkeypatch, [{"image_urls": [f"https://a.example/{index}.png" for index in range(12)], "usage": None}])
    live = 0
    peak = 0

    async def fake_download(_hc, _url):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)
        live -= 1
        return b"png"

    monkeypatch.setattr(images_mod, "_download_image", fake_download)
    req = ImageGenerationRequest(prompt="a cat", response_format="b64_json")
    result = await images_mod._image_generations(req, pool)
    assert len(result["data"]) == 12
    assert all("b64_json" in entry for entry in result["data"])
    assert peak == IMAGE_DOWNLOAD_CONCURRENCY


def test_image_edits_endpoint_rejects_out_of_range_n(monkeypatch, public_dns):
    pool, _account = _pool()
    app.state.qwen_pool = pool
    _install_collect_image(monkeypatch, [{"image_urls": ["https://a.example/i.png"], "usage": None}])
    monkeypatch.setattr(images_mod, "_image_http_client", AsyncMock(return_value=FakeImageClient()))
    client = TestClient(app)
    try:
        for bad_n, reason in (("0", "greater than or equal to 1"), ("99", "less than or equal to 4")):
            response = client.post("/v1/images/edits", files={"image": ("a.png", b"raw-png", "image/png")}, data={"prompt": "p", "n": bad_n})
            assert response.status_code == 400
            assert response.json()["error"]["message"] == f"invalid request body: n: Input should be {reason}"
        response = client.post("/v1/images/edits", files={"image": ("a.png", b"raw-png", "image/png")}, data={"prompt": "p", "n": "1"})
        assert response.status_code == 200
        assert response.json()["data"] == [{"url": "https://a.example/i.png"}]
    finally:
        client.close()


def test_image_edits_embeds_prompt_upload_and_mask(monkeypatch, public_dns):
    pool, _account = _pool()
    app.state.qwen_pool = pool
    calls = _install_collect_image(monkeypatch, [{"image_urls": ["https://a.example/i.png"], "usage": None}])
    monkeypatch.setattr(images_mod, "_image_http_client", AsyncMock(return_value=FakeImageClient()))
    client = TestClient(app)
    try:
        response = client.post(
            "/v1/images/edits",
            files={"image": ("a.png", b"raw-png", "image/png"), "mask": ("m.png", b"raw-mask", "image/png")},
            data={"prompt": "  blue cat  ", "response_format": "url", "user": "alice"},
        )
        assert response.status_code == 200
    finally:
        client.close()
    prompt = calls[0]["prompt"]
    assert prompt == "blue cat\n![image](data:image/png;base64,cmF3LXBuZw==)\n![image](data:image/png;base64,cmF3LW1hc2s=)"
    assert calls[0]["user"] == "alice"


def test_image_variations_forwards_the_prompt_and_uses_the_generation_path(monkeypatch, public_dns):
    pool, _account = _pool()
    app.state.qwen_pool = pool
    calls = _install_collect_image(monkeypatch, [{"image_urls": ["https://a.example/v.png"], "usage": None}])
    monkeypatch.setattr(images_mod, "_image_http_client", AsyncMock(return_value=FakeImageClient()))
    client = TestClient(app)
    try:
        response = client.post("/v1/images/variations", files={"image": ("a.png", b"raw-png", "image/png")}, data={"prompt": "warmer"})
        body = response.json()
    finally:
        client.close()
    assert response.status_code == 200
    assert body["data"] == [{"url": "https://a.example/v.png"}]
    assert calls[0]["prompt"] == "warmer\n![image](data:image/png;base64,cmF3LXBuZw==)"
    assert calls[0]["model"] == "qwen-image-gen"


def test_image_variations_defaults_the_prompt_to_an_empty_prefix(monkeypatch, public_dns):
    pool, _account = _pool()
    app.state.qwen_pool = pool
    calls = _install_collect_image(monkeypatch, [{"image_urls": ["https://a.example/v.png"], "usage": None}])
    monkeypatch.setattr(images_mod, "_image_http_client", AsyncMock(return_value=FakeImageClient()))
    client = TestClient(app)
    try:
        client.post("/v1/images/variations", files={"image": ("a.png", b"raw-png", "image/png")})
    finally:
        client.close()
    assert calls[0]["prompt"] == "![image](data:image/png;base64,cmF3LXBuZw==)"


def test_image_variations_rejects_out_of_range_n(monkeypatch, public_dns):
    pool, _account = _pool()
    app.state.qwen_pool = pool
    _install_collect_image(monkeypatch, [{"image_urls": ["https://a.example/v.png"], "usage": None}])
    monkeypatch.setattr(images_mod, "_image_http_client", AsyncMock(return_value=FakeImageClient()))
    client = TestClient(app)
    try:
        response = client.post("/v1/images/variations", files={"image": ("a.png", b"raw-png", "image/png")}, data={"n": "0"})
    finally:
        client.close()
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "invalid request body: n: Input should be greater than or equal to 1"


def test_image_edits_and_variations_are_503_without_a_pool():
    client = TestClient(app)
    try:
        edits = client.post("/v1/images/edits", files={"image": ("a.png", b"raw-png", "image/png")})
        variations = client.post("/v1/images/variations", files={"image": ("a.png", b"raw-png", "image/png")})
    finally:
        client.close()
    assert edits.status_code == 503
    assert edits.json()["error"]["message"] == QWEN_UNAVAILABLE_MESSAGE
    assert variations.status_code == 503
    assert variations.json()["error"]["message"] == QWEN_UNAVAILABLE_MESSAGE


def test_image_generations_endpoint_uses_the_request_pool(monkeypatch, public_dns):
    pool, _account = _pool()
    app.state.qwen_pool = pool
    _install_collect_image(monkeypatch, [{"image_urls": ["https://a.example/g.png"], "usage": None}])
    monkeypatch.setattr(images_mod, "_image_http_client", AsyncMock(return_value=FakeImageClient()))
    client = TestClient(app)
    try:
        response = client.post("/v1/images/generations", json={"prompt": "a cat"})
        body = response.json()
    finally:
        client.close()
    assert response.status_code == 200
    assert body["data"] == [{"url": "https://a.example/g.png"}]
    assert set(body) == {"created", "data", "usage", "session_id"}


def test_qwen_unavailable_message_is_shared_with_the_pool_lookup():
    assert images_mod.QWEN_UNAVAILABLE_MESSAGE == "qwen provider is not configured (required for image generation)"
