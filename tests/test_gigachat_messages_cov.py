import asyncio
import base64
from typing import Any

import httpx
import pytest
from fastapi import HTTPException

from danyapi.api.schemas import ChatMessage
from danyapi.gigachat import messages as gm

PNG_BYTES = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
PNG_URI = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()
PUBLIC_IP = "93.184.216.34"
ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class _Stub:
    def __init__(self) -> None:
        self.http: Any = None
        self.uploads: list[tuple[str, bytes, str]] = []

    async def upload_file(self, filename: str, data: bytes, content_type: str, purpose: str = "general") -> str:
        self.uploads.append((filename, data, content_type))
        return f"file-{len(self.uploads)}"


class _RemoteResponse:
    def __init__(self, chunks: list[bytes] | None = None, status: int = 200, headers: dict[str, str] | None = None, fail: Exception | None = None) -> None:
        self.status_code = status
        self.headers = httpx.Headers(headers or {})
        self._chunks = list(chunks or [])
        self._fail = fail

    async def aiter_bytes(self) -> Any:
        for chunk in self._chunks:
            yield chunk
        if self._fail is not None:
            raise self._fail


class _StreamContext:
    def __init__(self, response: _RemoteResponse) -> None:
        self._response = response
        self.exited = False

    async def __aenter__(self) -> _RemoteResponse:
        return self._response

    async def __aexit__(self, *exc: Any) -> bool:
        self.exited = True
        return False


class _RemoteHttp:
    def __init__(self, response: _RemoteResponse) -> None:
        self._response = response
        self.calls: list[tuple[str, str, dict]] = []
        self.contexts: list[_StreamContext] = []

    def stream(self, method: str, url: str, **kwargs: Any) -> _StreamContext:
        self.calls.append((method, url, kwargs))
        context = _StreamContext(self._response)
        self.contexts.append(context)
        return context


def _public(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gm.socket, "getaddrinfo", lambda *args, **kwargs: [(2, 1, 6, "", (PUBLIC_IP, 443))])


def _resolving_to(monkeypatch: pytest.MonkeyPatch, *addresses: str) -> None:
    monkeypatch.setattr(gm.socket, "getaddrinfo", lambda *args, **kwargs: [(2, 1, 6, "", (address, 443)) for address in addresses])


async def _drain(turns: int = 12) -> None:
    for _ in range(turns):
        await asyncio.sleep(0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("plain", "plain"),
        (["a", {"type": "text", "text": "b"}, {"type": "input_text", "text": "c"}, {"type": "image_url", "text": "d"}], "abc"),
        ([{"type": "text", "text": 5}, {"type": "text"}, "d"], "d"),
        ([], ""),
        (None, ""),
        (True, ""),
        (False, ""),
        (7, "7"),
        (1.5, "1.5"),
        ({"a": 1}, ""),
        (b"bytes", ""),
        (("t",), ""),
    ],
)
def test_text_of_renders_only_supported_shapes(value, expected):
    assert gm._text_of(value) == expected


def test_text_of_never_leaks_a_python_repr_of_a_dict():
    assert gm._text_of({"type": "text", "text": "secret"}) == ""
    assert gm._text_of({"text": "secret"}) == ""
    assert "secret" not in gm._text_of({"text": "secret"})


def test_image_uris_picks_up_both_accepted_shapes():
    assert gm._image_uris("not a list") == []
    assert gm._image_uris([{"type": "text", "text": "hi"}, "raw", {"type": "image_url"}, {"type": "image_url", "image_url": 7}]) == []
    assert gm._image_uris([{"type": "image_url", "image_url": "https://x/1.png"}]) == ["https://x/1.png"]
    assert gm._image_uris([{"type": "input_image", "image_url": {"url": "https://x/2.png"}}]) == ["https://x/2.png"]


def test_extension_for_maps_every_supported_type():
    assert gm._extension_for("image/jpeg") == ".jpg"
    assert gm._extension_for("image/png") == ".png"
    assert gm._extension_for("image/tiff") == ".tiff"
    assert gm._extension_for("image/bmp") == ".bmp"
    assert gm._extension_for("image/webp") == ".bin"


def test_gigachat_arguments_wraps_a_bare_value():
    assert gm._gigachat_arguments(5) == {"input": 5}
    assert gm._gigachat_arguments([1, 2]) == {"input": [1, 2]}
    assert gm._gigachat_arguments(True) == {"input": True}
    assert gm._gigachat_arguments('  {"a": 1}  ') == {"a": 1}


def test_tool_call_name_accepts_the_flat_legacy_shape():
    assert gm._tool_call_name({"name": "f", "arguments": '{"a": 1}'}) == ("f", {"a": 1})
    assert gm._tool_call_name({"name": "f"}) == ("f", {})
    assert gm._tool_call_name({"name": 7, "arguments": "{}"}) == ("", {})


def test_tool_call_name_uses_an_empty_name_when_none_is_declared():
    assert gm._tool_call_name({"function": {"arguments": "x"}}) == ("", {"input": "x"})


def test_as_function_result_of_blank_content_is_an_empty_object():
    assert gm._as_function_result("   ") == "{}"
    assert gm._as_function_result({"temp": 27}) == "{}"
    assert gm._as_function_result(None) == "{}"
    assert gm._as_function_result('  {"temp": 27}  ') == '{"temp": 27}'


def test_system_text_strips_the_whitespace():
    assert gm._system_text("  be nice \n") == "be nice"
    assert gm._system_text(None) == ""
    assert gm._developer_to_system("developer") == "system"
    assert gm._developer_to_system("user") == "user"


def test_build_functions_drops_unusable_and_non_dict_items():
    specs = gm._build_functions(
        [
            "not a dict",
            {"type": "function"},
            {"type": "function", "function": {"name": "", "description": "nameless"}},
            {"type": "function", "function": {"name": None}},
            {"type": "function", "function": {"name": "f"}},
        ],
        ["also not a dict", {"name": "g", "description": "d", "parameters": {"type": "object"}}, {"description": "no name"}],
    )
    assert specs == [
        {"name": "f", "parameters": {"type": "object", "properties": {}}},
        {"name": "g", "description": "d", "parameters": {"type": "object"}},
    ]


def test_build_functions_collapses_duplicate_definitions_to_one():
    definition = {"name": "f", "description": "d", "parameters": {"type": "object"}}
    specs = gm._build_functions(
        [
            {"type": "function", "function": dict(definition)},
            {"type": "function", "function": dict(definition)},
        ],
        [dict(definition)],
    )
    assert specs == [definition]


def test_build_functions_dedup_is_independent_of_key_order():
    specs = gm._build_functions(
        [
            {"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"parameters": {"type": "object"}, "description": "d", "name": "f"}},
        ],
        None,
    )
    assert len(specs) == 1


def test_build_functions_keeps_definitions_that_differ():
    specs = gm._build_functions(
        [
            {"type": "function", "function": {"name": "f", "description": "one"}},
            {"type": "function", "function": {"name": "f", "description": "two"}},
        ],
        None,
    )
    assert [spec["description"] for spec in specs] == ["one", "two"]
    assert all(spec["name"] == "f" for spec in specs)


def test_build_functions_tolerates_non_json_parameters():
    specs = gm._build_functions([{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "tags": {1, 2}}}}], None)
    assert specs[0]["parameters"] == {"type": "object", "tags": {1, 2}}


def test_function_call_value_none_and_auto_pass_through():
    assert gm._function_call_value("none", None, ["f"]) == "none"
    assert gm._function_call_value("auto", None, ["f"]) == "auto"
    assert gm._function_call_value(None, None, ["f"]) == "auto"
    assert gm._function_call_value("auto", "none", ["f"]) == "none"


def test_function_call_value_required_with_one_function_forces_that_call():
    assert gm._function_call_value("required", None, ["f"]) == {"name": "f"}
    assert gm._function_call_value("required", None, ["f"]) != "none"
    assert gm._function_call_value("any", None, ["f"]) == {"name": "f"}
    assert gm._function_call_value("any", None, ["f"]) != "none"


def test_function_call_value_required_with_two_functions_is_rejected():
    with pytest.raises(HTTPException) as exc:
        gm._function_call_value("required", None, ["f", "g"])
    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat requires a function call only when exactly one function is provided, name it in tool_choice"


def test_function_call_value_any_with_two_functions_is_rejected():
    with pytest.raises(HTTPException) as exc:
        gm._function_call_value("any", None, ["f", "g"])
    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat requires a function call only when exactly one function is provided, name it in tool_choice"


def test_function_call_value_required_without_any_function_is_rejected():
    with pytest.raises(HTTPException) as exc:
        gm._function_call_value("required", None, [])
    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat requires a function call only when exactly one function is provided, name it in tool_choice"


def test_function_call_value_named_choice_passes_through():
    assert gm._function_call_value("g", None, ["f", "g"]) == {"name": "g"}
    assert gm._function_call_value(None, "f", ["f", "g"]) == {"name": "f"}
    assert gm._function_call_value("g", "f", ["f", "g"]) == {"name": "f"}


def test_function_call_value_unknown_named_choice_is_rejected():
    with pytest.raises(HTTPException) as exc:
        gm._function_call_value("nope", None, ["f", "g"])
    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat does not know function nope, provide one of: f, g"


def test_function_call_value_without_a_declared_list_accepts_any_name():
    assert gm._function_call_value("whatever", None) == {"name": "whatever"}
    assert gm._function_call_value({"name": "whatever"}, None) == {"name": "whatever"}


def test_function_call_value_object_choice_is_validated():
    assert gm._function_call_value({"type": "function", "function": {"name": "f"}}, None, ["f", "g"]) == {"name": "f"}
    assert gm._function_call_value({"name": "g"}, None, ["f", "g"]) == {"name": "g"}


def test_function_call_value_object_choice_with_an_unknown_name_is_rejected():
    with pytest.raises(HTTPException) as exc:
        gm._function_call_value({"function": {"name": "zz"}}, None, ["f", "g"])
    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat does not know function zz, provide one of: f, g"


@pytest.mark.parametrize("choice", [{"name": ""}, {"function": {}}, {"function": {"name": 7}}, 5, ["f"]])
def test_function_call_value_falls_back_to_auto(choice):
    assert gm._function_call_value(choice, None, ["f"]) == "auto"


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"\x89PNG\r\n\x1a\nrest", "image/png"),
        (b"\xff\xd8\xff\xe0rest", "image/jpeg"),
        (b"BM\x00\x00\x00rest", "image/bmp"),
        (b"II\x2a\x00rest", "image/tiff"),
        (b"MM\x00\x2arest", "image/tiff"),
        (b"GIF89a-rest", ""),
        (b"", ""),
    ],
)
def test_sniff_image_type_reads_the_payload_not_the_header(head, expected):
    assert gm._sniff_image_type(head) == expected


def test_host_is_public_accepts_a_public_address(monkeypatch):
    _public(monkeypatch)
    assert gm._host_is_public("images.example.com") is True


@pytest.mark.parametrize(
    "address",
    ["10.0.0.1", "127.0.0.1", "169.254.169.254", "224.0.0.1", "0.0.0.0", "192.0.2.1", "240.0.0.1", "::1", "fc00::1", "fe80::1"],
)
def test_host_is_public_refuses_non_public_addresses(monkeypatch, address):
    _resolving_to(monkeypatch, address)
    assert gm._host_is_public("images.example.com") is False


@pytest.mark.parametrize("error", [OSError("nxdomain"), UnicodeError("idna failed")])
def test_host_is_public_refuses_when_resolution_fails(monkeypatch, error):
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(gm.socket, "getaddrinfo", boom)
    assert gm._host_is_public("images.example.com") is False


def test_host_is_public_refuses_an_empty_answer(monkeypatch):
    monkeypatch.setattr(gm.socket, "getaddrinfo", lambda *args, **kwargs: [])
    assert gm._host_is_public("images.example.com") is False


def test_host_is_public_refuses_an_unparsable_address(monkeypatch):
    _resolving_to(monkeypatch, "not-an-ip")
    assert gm._host_is_public("images.example.com") is False


def test_host_is_public_refuses_an_address_tuple_without_a_host(monkeypatch):
    monkeypatch.setattr(gm.socket, "getaddrinfo", lambda *args, **kwargs: [(2, 1, 6, "", ())])
    assert gm._host_is_public("images.example.com") is False


def test_host_is_public_checks_every_resolved_address(monkeypatch):
    _resolving_to(monkeypatch, PUBLIC_IP, "127.0.0.1")
    assert gm._host_is_public("rebind.example.com") is False


async def test_remote_image_refuses_a_private_address_before_any_request(monkeypatch):
    _resolving_to(monkeypatch, "10.1.2.3")
    client = _Stub()
    client.http = _RemoteHttp(_RemoteResponse([PNG_BYTES], headers={"content-type": "image/png"}))

    with pytest.raises(HTTPException) as exc:
        await gm._fetch_remote_image(client, "http://internal.example/pic.png")

    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat refuses to fetch images from the private address internal.example"
    assert client.http.calls == []


async def test_remote_image_refuses_a_link_local_metadata_address(monkeypatch):
    _resolving_to(monkeypatch, "169.254.169.254")
    client = _Stub()
    client.http = _RemoteHttp(_RemoteResponse([PNG_BYTES], headers={"content-type": "image/png"}))

    with pytest.raises(HTTPException) as exc:
        await gm._fetch_remote_image(client, "http://169.254.169.254/latest/meta-data")

    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat refuses to fetch images from the private address 169.254.169.254"
    assert client.http.calls == []


async def test_remote_image_refuses_a_url_without_a_host(monkeypatch):
    _public(monkeypatch)
    client = _Stub()
    client.http = _RemoteHttp(_RemoteResponse([PNG_BYTES]))

    with pytest.raises(HTTPException) as exc:
        await gm._fetch_remote_image(client, "file:///etc/passwd")

    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat remote image URL has no host"
    assert client.http.calls == []


async def test_remote_image_aborts_the_body_at_the_cap(monkeypatch):
    _public(monkeypatch)
    monkeypatch.setattr(gm, "MAX_IMAGE_BYTES", 1024 * 1024)
    resp = _RemoteResponse([b"\x89PNG\r\n\x1a\n" + b"a" * (1024 * 1024), b"b" * 16], headers={"content-type": "image/png"})
    client = _Stub()
    client.http = _RemoteHttp(resp)

    with pytest.raises(HTTPException) as exc:
        await gm._fetch_remote_image(client, "https://cdn.example/pic.png")

    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat accepts images up to 1 MB"
    assert len(client.http.calls) == 1
    assert client.http.contexts[0].exited is True


async def test_remote_image_rejects_a_non_image_body_declared_as_png(monkeypatch):
    _public(monkeypatch)
    client = _Stub()
    client.http = _RemoteHttp(_RemoteResponse([b"<html>not an image</html>"], headers={"content-type": "image/png"}))

    with pytest.raises(HTTPException) as exc:
        await gm._fetch_remote_image(client, "https://cdn.example/pic.png")

    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat could not confirm the image type from the payload"


async def test_remote_image_accepts_a_real_png_declared_as_jpeg(monkeypatch):
    _public(monkeypatch)
    client = _Stub()
    client.http = _RemoteHttp(_RemoteResponse([PNG_BYTES], headers={"content-type": "image/jpeg"}))

    assert await gm._fetch_remote_image(client, "https://cdn.example/pic.png") == ("image/png", PNG_BYTES)


async def test_remote_image_never_follows_redirects(monkeypatch):
    _public(monkeypatch)
    client = _Stub()
    client.http = _RemoteHttp(_RemoteResponse([PNG_BYTES], headers={"content-type": "image/png"}))

    await gm._fetch_remote_image(client, "https://cdn.example/pic.png")

    assert client.http.calls == [("GET", "https://cdn.example/pic.png", {"timeout": gm.REMOTE_TIMEOUT_SEC, "follow_redirects": False})]


async def test_remote_image_reports_an_upstream_error_status(monkeypatch):
    _public(monkeypatch)
    client = _Stub()
    client.http = _RemoteHttp(_RemoteResponse([b"missing"], 404))

    with pytest.raises(HTTPException) as exc:
        await gm._fetch_remote_image(client, "https://cdn.example/pic.png")

    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat could not fetch remote image: upstream returned 404"


async def test_remote_image_refuses_a_declared_type_outside_the_allowed_set(monkeypatch):
    _public(monkeypatch)
    client = _Stub()
    client.http = _RemoteHttp(_RemoteResponse([PNG_BYTES], headers={"content-type": "text/html; charset=utf-8"}))

    with pytest.raises(HTTPException) as exc:
        await gm._fetch_remote_image(client, "https://cdn.example/pic.png")

    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat does not accept image type text/html"


async def test_remote_image_accepts_a_response_without_a_content_type(monkeypatch):
    _public(monkeypatch)
    client = _Stub()
    client.http = _RemoteHttp(_RemoteResponse([PNG_BYTES]))

    assert await gm._fetch_remote_image(client, "https://cdn.example/pic.png") == ("image/png", PNG_BYTES)


async def test_remote_image_converts_a_transport_error_into_400(monkeypatch):
    _public(monkeypatch)
    client = _Stub()
    client.http = _RemoteHttp(_RemoteResponse([PNG_BYTES], fail=httpx.ConnectError("connection refused")))

    with pytest.raises(HTTPException) as exc:
        await gm._fetch_remote_image(client, "https://cdn.example/pic.png")

    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat could not fetch remote image: connection refused"


async def test_remote_image_fetches_a_public_address_resolved_locally(monkeypatch):
    client = _Stub()
    client.http = _RemoteHttp(_RemoteResponse([PNG_BYTES], headers={"content-type": "image/png"}))

    assert gm._host_is_public(PUBLIC_IP) is True
    content_type, data = await gm._fetch_remote_image(client, f"http://{PUBLIC_IP}/pic.png")

    assert (content_type, data) == ("image/png", PNG_BYTES)
    assert client.http.calls[0][0:2] == ("GET", f"http://{PUBLIC_IP}/pic.png")


async def test_resolve_images_without_pending_uploads_does_nothing():
    stub = _Stub()
    await gm._resolve_images(stub, [])
    assert stub.uploads == []


async def test_resolve_images_accepts_exactly_one_image_per_message():
    stub = _Stub()
    entry = {"role": "user", "content": "look"}

    await gm._resolve_images(stub, [(entry, [PNG_URI])])

    assert entry["attachments"] == ["file-1"]
    assert stub.uploads == [("image.png", PNG_BYTES, "image/png")]


async def test_resolve_images_rejects_a_second_image_in_the_same_message():
    stub = _Stub()
    with pytest.raises(HTTPException) as exc:
        await gm._resolve_images(stub, [({"role": "user", "content": "look"}, [PNG_URI, PNG_URI])])
    assert exc.value.status_code == 400
    assert exc.value.detail == f"gigachat accepts at most {gm.MAX_IMAGES_PER_MESSAGE} image(s) per message"
    assert stub.uploads == []


async def test_resolve_images_mixes_data_and_remote_images_in_order(monkeypatch):
    fetched: list[str] = []

    async def fake_fetch(_client: Any, uri: str) -> tuple[str, bytes]:
        fetched.append(uri)
        return "image/png", b"\x89PNG\r\n\x1a\nremote"

    monkeypatch.setattr(gm, "_fetch_remote_image", fake_fetch)
    stub = _Stub()
    first = {"role": "user", "content": "look"}
    second = {"role": "user", "content": "and this"}

    await gm._resolve_images(stub, [(first, [PNG_URI]), (second, ["https://cdn.example/1.png"])])

    assert fetched == ["https://cdn.example/1.png"]
    assert [name for name, _data, _ctype in stub.uploads] == ["image.png", "image.png"]
    assert [ctype for _name, _data, ctype in stub.uploads] == ["image/png", "image/png"]
    assert [data for _name, data, _ctype in stub.uploads] == [PNG_BYTES, b"\x89PNG\r\n\x1a\nremote"]
    assert first["attachments"] == ["file-1"]
    assert second["attachments"] == ["file-2"]


async def test_resolve_images_uploads_in_message_order_when_a_later_fetch_finishes_first(monkeypatch):
    started: list[str] = []
    gates = {0: asyncio.Event(), 1: asyncio.Event()}
    both_in_flight = asyncio.Event()

    async def fake_fetch(_client: Any, uri: str) -> tuple[str, bytes]:
        started.append(uri)
        if len(started) == 2:
            both_in_flight.set()
        await gates[int(uri.rsplit("/", 1)[1][0])].wait()
        return "image/png", uri.encode()

    monkeypatch.setattr(gm, "_fetch_remote_image", fake_fetch)
    stub = _Stub()
    uris = ["https://cdn.example/0.png", "https://cdn.example/1.png"]
    pending = [({"role": "user", "content": f"m{index}"}, [uri]) for index, uri in enumerate(uris)]

    task = asyncio.ensure_future(gm._resolve_images(stub, pending))
    await asyncio.wait_for(both_in_flight.wait(), timeout=5.0)
    assert started == uris

    gates[1].set()
    await _drain(3)
    assert stub.uploads == []

    gates[0].set()
    await task

    assert [data.decode() for _name, data, _ctype in stub.uploads] == uris
    assert pending[0][0]["attachments"] == ["file-1"]
    assert pending[1][0]["attachments"] == ["file-2"]


async def test_resolve_images_bounds_the_fetch_concurrency(monkeypatch):
    inflight = 0
    peak = 0
    started: list[str] = []

    async def fake_fetch(_client: Any, uri: str) -> tuple[str, bytes]:
        nonlocal inflight, peak
        started.append(uri)
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0)
        inflight -= 1
        return "image/png", uri.encode()

    monkeypatch.setattr(gm, "_fetch_remote_image", fake_fetch)
    stub = _Stub()
    uris = [f"https://cdn.example/{index}.png" for index in range(7)]
    pending = [({"role": "user", "content": f"m{index}"}, [uri]) for index, uri in enumerate(uris)]

    await gm._resolve_images(stub, pending)

    assert gm.MAX_IMAGE_FETCH_CONCURRENCY == 4
    assert peak == 4
    assert len(started) == 7
    assert started[:4] == uris[:4]
    assert [data.decode() for _name, data, _ctype in stub.uploads] == uris
    assert [entry["attachments"] for entry, _uris in pending] == [[f"file-{index + 1}"] for index in range(7)]


async def test_resolve_images_attempts_every_image_and_surfaces_the_failure(monkeypatch):
    attempted: list[str] = []

    async def fake_fetch(_client: Any, uri: str) -> tuple[str, bytes]:
        attempted.append(uri)
        if uri.endswith("/1.png"):
            raise HTTPException(400, "gigachat could not confirm the image type from the payload")
        return "image/png", b"\x89PNG\r\n\x1a\n"

    monkeypatch.setattr(gm, "_fetch_remote_image", fake_fetch)
    stub = _Stub()
    uris = ["https://cdn.example/0.png", "https://cdn.example/1.png", "https://cdn.example/2.png"]
    pending = [({"role": "user", "content": f"m{index}"}, [uri]) for index, uri in enumerate(uris)]

    with pytest.raises(HTTPException) as exc:
        await gm._resolve_images(stub, pending)

    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat could not confirm the image type from the payload"
    assert sorted(attempted) == sorted(uris)
    assert stub.uploads == []
    assert all("attachments" not in entry for entry, _uris in pending)


async def test_resolve_images_rejects_a_data_uri_over_the_byte_cap(monkeypatch):
    monkeypatch.setattr(gm, "MAX_IMAGE_BYTES", 1024 * 1024)
    stub = _Stub()
    oversized = "data:image/png;base64," + base64.b64encode(PNG_BYTES + b"x" * (1024 * 1024)).decode()

    with pytest.raises(HTTPException) as exc:
        await gm._resolve_images(stub, [({"role": "user", "content": "look"}, [oversized])])

    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat accepts images up to 1 MB"
    assert stub.uploads == []


async def test_resolve_images_rejects_a_data_uri_of_an_unsupported_type():
    stub = _Stub()
    gif = "data:image/gif;base64,R0lGODlhAQABAAAAACw="

    with pytest.raises(HTTPException) as exc:
        await gm._resolve_images(stub, [({"role": "user", "content": "look"}, [gif])])

    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat does not accept image type image/gif"
    assert stub.uploads == []


@pytest.mark.parametrize("uri", ["file:///etc/passwd", "ftp://cdn.example/pic.png", "s3://bucket/key.png", "cdn.example/pic.png"])
async def test_resolve_images_rejects_a_url_that_is_neither_data_nor_http(uri):
    stub = _Stub()

    with pytest.raises(HTTPException) as exc:
        await gm._resolve_images(stub, [({"role": "user", "content": "look"}, [uri])])

    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat images must be data URIs or http(s) URLs"
    assert stub.uploads == []


async def test_build_messages_keeps_the_message_text_grammar_neutral_with_an_image():
    stub = _Stub()
    content = [{"type": "text", "text": "look at this"}, {"type": "image_url", "image_url": {"url": PNG_URI}}]

    out, specs, call = await gm.build_messages(stub, [ChatMessage(role="user", content=content)])

    assert out == [{"role": "user", "content": "look at this", "attachments": ["file-1"]}]
    assert specs == []
    assert call is None


async def test_build_messages_rejects_a_second_image_in_one_message():
    stub = _Stub()
    content = [
        {"type": "text", "text": "look at these"},
        {"type": "image_url", "image_url": {"url": PNG_URI}},
        {"type": "image_url", "image_url": {"url": PNG_URI}},
    ]

    with pytest.raises(HTTPException) as exc:
        await gm.build_messages(stub, [ChatMessage(role="user", content=content)])

    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat accepts at most 1 image(s) per message"
    assert stub.uploads == []


async def test_build_messages_fetches_a_remote_image_for_a_surviving_user_message(monkeypatch):
    fetched: list[str] = []

    async def fake_fetch(_client: Any, uri: str) -> tuple[str, bytes]:
        fetched.append(uri)
        return "image/png", b"\x89PNG\r\n\x1a\n"

    monkeypatch.setattr(gm, "_fetch_remote_image", fake_fetch)
    stub = _Stub()
    messages = [
        ChatMessage(role="function", name="f", content="{}"),
        ChatMessage(role="assistant", content="thinking"),
        ChatMessage(role="user", content=[{"type": "text", "text": "kept"}, {"type": "image_url", "image_url": {"url": "https://cdn.example/a.png"}}]),
    ]

    out, _specs, _call = await gm.build_messages(stub, messages)

    assert fetched == ["https://cdn.example/a.png"]
    assert [message["role"] for message in out] == ["user"]
    assert out[0]["attachments"] == ["file-1"]


async def test_build_messages_rejects_an_image_url_that_is_neither_data_nor_http():
    stub = _Stub()
    messages = [
        ChatMessage(role="user", content=""),
        ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": "ftp://cdn.example/bad.png"}}]),
        ChatMessage(role="user", content="hi"),
    ]

    with pytest.raises(HTTPException) as exc:
        await gm.build_messages(stub, messages)

    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat images must be data URIs or http(s) URLs"
    assert stub.uploads == []


async def test_build_messages_attaches_images_only_to_messages_that_carry_them():
    stub = _Stub()
    messages = [
        ChatMessage(role="user", content="first"),
        ChatMessage(role="user", content=[{"type": "text", "text": "second"}, {"type": "image_url", "image_url": {"url": PNG_URI}}]),
        ChatMessage(role="user", content="third"),
    ]

    out, _specs, _call = await gm.build_messages(stub, messages)

    assert out[0] == {"role": "user", "content": "first"}
    assert out[1] == {"role": "user", "content": "second", "attachments": ["file-1"]}
    assert out[2] == {"role": "user", "content": "third"}


async def test_build_messages_drops_empty_user_messages():
    stub = _Stub()
    messages = [
        ChatMessage(role="user", content=""),
        ChatMessage(role="user", content=[{"type": "text", "text": ""}]),
        ChatMessage(role="user", content=[]),
        ChatMessage(role="user", content="hi"),
    ]

    out, _specs, _call = await gm.build_messages(stub, messages)

    assert out == [{"role": "user", "content": "hi"}]


async def test_build_messages_drops_a_tool_message_without_a_resolvable_name():
    stub = _Stub()
    messages = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="tool", tool_call_id="unknown", content="{}"),
        ChatMessage(role="tool", name="", tool_call_id="unknown", content="{}"),
    ]

    out, _specs, _call = await gm.build_messages(stub, messages)

    assert [message["role"] for message in out] == ["user"]


async def test_build_messages_forces_a_single_function_call_when_required():
    stub = _Stub()
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    _out, specs, call = await gm.build_messages(stub, [ChatMessage(role="user", content="hi")], tools=tools, tool_choice="required")

    assert [spec["name"] for spec in specs] == ["get_weather"]
    assert call == {"name": "get_weather"}


async def test_build_messages_rejects_required_with_two_functions():
    stub = _Stub()
    tools = [{"type": "function", "function": {"name": "f"}}, {"type": "function", "function": {"name": "g"}}]

    with pytest.raises(HTTPException) as exc:
        await gm.build_messages(stub, [ChatMessage(role="user", content="hi")], tools=tools, tool_choice="required")

    assert exc.value.status_code == 400
    assert exc.value.detail == "gigachat requires a function call only when exactly one function is provided, name it in tool_choice"
    assert stub.uploads == []


def test_request_body_keeps_a_json_object_response_format():
    body = gm.request_body([{"role": "user", "content": "hi"}], [], None, None, None, 0, {"type": "json_object"})
    assert body == {"messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_object"}}


def test_request_body_drops_a_non_dict_response_format_and_a_non_positive_budget():
    assert "response_format" not in gm.request_body([{"role": "user", "content": "hi"}], [], None, None, None, 0, "json")
    assert "max_tokens" not in gm.request_body([{"role": "user", "content": "hi"}], [], None, None, None, -5, None)
    assert "function_call" not in gm.request_body([{"role": "user", "content": "hi"}], [{"name": "f"}], None, None, None, None, None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (12, 12),
        (0, 0),
        (-3, 0),
        ("42", 42),
        (" 7 ", 7),
        ("8.9", 8),
        ("-3", 0),
        ("abc", 0),
        ("", 0),
        (4.7, 4),
        (0.4, 0),
        (-1.5, 0),
        (True, 0),
        (False, 0),
        (None, 0),
        ({"a": 1}, 0),
        ([1], 0),
    ],
)
def test_token_count_coerces_only_real_numbers(value, expected):
    assert gm._token_count(value) == expected


def test_normalize_usage_sums_the_parts_when_the_total_is_missing():
    assert gm.normalize_usage({"prompt_tokens": 10, "completion_tokens": 5})["total_tokens"] == 15
    assert gm.normalize_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 0})["total_tokens"] == 15


def test_normalize_usage_raises_a_total_that_is_below_the_parts():
    assert gm.normalize_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 3})["total_tokens"] == 15


def test_normalize_usage_keeps_a_larger_upstream_total_and_the_cached_tokens():
    usage = gm.normalize_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 99, "precached_prompt_tokens": 4})
    assert usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 99,
        "prompt_tokens_details": {"cached_tokens": 4},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


def test_normalize_usage_of_a_non_dict_is_all_zero():
    assert gm.normalize_usage("nope") == ZERO_USAGE
    assert gm.normalize_usage(None) == ZERO_USAGE
    assert gm.normalize_usage([1, 2]) == ZERO_USAGE


def test_normalize_usage_of_an_empty_dict_carries_the_detail_blocks():
    assert gm.normalize_usage({}) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


def test_normalize_usage_rejects_bools_and_containers_in_the_token_fields():
    usage = gm.normalize_usage({"prompt_tokens": True, "completion_tokens": [1, 2], "total_tokens": {"a": 1}, "precached_prompt_tokens": None})
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0
    assert usage["prompt_tokens_details"] == {"cached_tokens": 0}


def test_normalize_finish_reason_falls_back_to_stop():
    assert gm.normalize_finish_reason("stop") == "stop"
    assert gm.normalize_finish_reason("error") == "stop"
    assert gm.normalize_finish_reason(7) == "stop"
    assert gm.normalize_finish_reason("") == "stop"
