from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import ClassVar

import pytest
from fastapi import HTTPException, Request

import danyapi.api.deepseek as deepseek_mod
import danyapi.api.envtokens as envtokens_mod
import danyapi.gigachat.api as gigachat_api
import danyapi.gigachat.messages as gigachat_messages
import danyapi.opencode.api as opencode_api
from danyapi.api.anthropic import build_chat_request, convert_stop_sequences


class FakeResp:
    def __init__(self, body="", status=200, content_type="text/event-stream; charset=utf-8") -> None:
        self.status_code = status
        self.headers = {"content-type": content_type}
        self._b = body.encode() if isinstance(body, str) else body

    async def aiter_bytes(self):
        yield self._b

    async def aclose(self):
        return None

    async def aread(self):
        return self._b


def _null_lock(_sem, _timeout=None):
    class _Ctx:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *_exc):
            return False

    return _Ctx()


def _aiter(items):
    async def inner():
        for item in items:
            yield item

    return inner()


def _drain(generator) -> str:
    async def run() -> str:
        return "".join([chunk async for chunk in generator])

    return asyncio.run(run())


def _text_event(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


def test_a_byok_form_key_is_read_from_the_cached_form_however_large():
    import danyapi.api.byok as byok_mod

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "root_path": "",
        "scheme": "http",
        "query_string": b"",
        "headers": [(b"content-length", b"9000000")],
        "client": ("1.2.3.4", 1234),
        "server": ("test", 80),
    }
    request = Request(scope)
    request._form = {"api_key": "  secret  "}
    assert asyncio.run(byok_mod._api_key_from_form(request)) == "secret"


def test_a_byok_form_without_a_cached_form_still_guards_the_size():
    import danyapi.api.byok as byok_mod

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "root_path": "",
        "scheme": "http",
        "query_string": b"",
        "headers": [(b"content-length", str(byok_mod.BYOK_FORM_MAX_BYTES + 1).encode())],
        "client": ("1.2.3.4", 1234),
        "server": ("test", 80),
    }
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(byok_mod._api_key_from_form(Request(scope)))
    assert excinfo.value.status_code == 413


class _TokenClient:
    def __init__(self, name: str, accepted: bool | None) -> None:
        self.name = name
        self.accepted = accepted
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True

    async def check_auth(self) -> bool:
        if self.accepted is None:
            raise envtokens_mod.TokenCheckFailed("upstream unreachable")
        return self.accepted


def test_an_indeterminate_token_check_closes_every_client_it_accepted():
    closed: list[str] = []
    good = _TokenClient("good", True)
    bad = _TokenClient("bad", None)
    clients = {"t1": good, "t2": bad}
    plan = [("t1", envtokens_mod._PLAN_NEW, None), ("t2", envtokens_mod._PLAN_NEW, None)]

    async def run() -> None:
        with pytest.raises(HTTPException) as excinfo:
            await envtokens_mod._validated_tokens(plan, "deepseek", lambda token: clients[token])
        assert excinfo.value.status_code == 503

    asyncio.run(run())
    assert good.closed is True
    assert bad.closed is True
    assert closed == []


def test_an_indeterminate_token_check_never_closes_a_pool_owned_client():
    good = _TokenClient("owned", True)
    acct = SimpleNamespace(client=good)
    broken = _TokenClient("broken", None)
    plan = [
        ("t1", envtokens_mod._PLAN_BROKEN, acct),
        ("t2", envtokens_mod._PLAN_BROKEN, SimpleNamespace(client=broken)),
    ]

    async def run() -> None:
        with pytest.raises(HTTPException):
            await envtokens_mod._validated_tokens(plan, "deepseek", lambda token: _TokenClient(token, True))

    asyncio.run(run())
    assert good.closed is False
    assert broken.closed is False


@pytest.mark.parametrize("body", ["[]", '"boom"', "123", "null", "true"])
def test_a_non_object_json_body_is_a_bad_gateway(body):
    client = SimpleNamespace(completion=_returns(FakeResp(body, content_type="application/json")))

    async def run() -> None:
        with pytest.raises(HTTPException) as excinfo:
            await deepseek_mod._send_completion(client, {}, "s", None, "p", "default", False, False)
        assert excinfo.value.status_code == 502

    asyncio.run(run())


def test_a_non_dict_data_member_does_not_break_the_error_scan():
    client = SimpleNamespace(completion=_returns(FakeResp(json.dumps({"data": [1, 2, 3]}), content_type="application/json")))

    async def run() -> None:
        with pytest.raises(HTTPException) as excinfo:
            await deepseek_mod._send_completion(client, {}, "s", None, "p", "default", False, False)
        assert excinfo.value.status_code == 502

    asyncio.run(run())


def _returns(value):
    async def inner(*_args, **_kwargs):
        return value

    return inner


def _true():
    async def inner(*_args, **_kwargs):
        return True

    return inner()


def test_gigachat_drops_everything_after_the_stop_marker(monkeypatch):
    monkeypatch.setattr(gigachat_api, "account_lock", _null_lock)
    monkeypatch.setattr(gigachat_api, "build_messages", _returns(([], [], "hi")))
    monkeypatch.setattr(gigachat_api, "request_body", lambda *args, **kwargs: {})
    monkeypatch.setattr(gigachat_api, "_send", _returns(FakeResp()))
    monkeypatch.setattr(gigachat_api, "_iter_sse", lambda resp: _aiter([_text_event("say "), _text_event("END"), _text_event(" and more")]))
    account = SimpleNamespace(sem=asyncio.Semaphore(1), client=object())
    text = _drain(gigachat_api.stream_openai(account, [], "m", stop=["END"]))
    assert "say " in text
    assert "and more" not in text


def test_opencode_drops_everything_after_the_stop_marker(monkeypatch):
    monkeypatch.setattr(opencode_api, "account_lock", _null_lock)
    monkeypatch.setattr(opencode_api, "build_messages", lambda messages: [])
    monkeypatch.setattr(opencode_api, "request_body", lambda *args, **kwargs: {})
    monkeypatch.setattr(opencode_api, "_send", _returns(FakeResp()))
    monkeypatch.setattr(opencode_api, "_iter_sse", lambda resp: _aiter([_text_event("say "), _text_event("END"), _text_event(" and more")]))
    account = SimpleNamespace(sem=asyncio.Semaphore(1), client=object())
    text = _drain(opencode_api.stream_openai(account, [], "m", stop=["END"]))
    assert "say " in text
    assert "and more" not in text


def test_gigachat_keeps_a_function_call_continuation():
    delta, _finish = gigachat_api._delta_from_event({"choices": [{"delta": {"function_call": {"arguments": '{"a":'}}}]})
    call = delta["tool_calls"][0]
    assert call["id"] is None
    assert call["function"] == {"name": "", "arguments": '{"a":'}


class _PeerStream:
    def __init__(self, address: str) -> None:
        self.address = address

    def get_extra_info(self, name):
        return (self.address, 443) if name == "server_addr" else None


def _peer_response(address: str):
    import httpx

    request = httpx.Request("GET", "https://cdn.example/pic.png")
    return httpx.Response(200, request=request, extensions={"network_stream": _PeerStream(address)})


def test_the_gigachat_peer_address_is_rechecked_after_connect():
    assert gigachat_messages._peer_is_public(_peer_response("127.0.0.1")) is False
    assert gigachat_messages._peer_is_public(_peer_response("169.254.169.254")) is False
    assert gigachat_messages._peer_is_public(_peer_response("93.184.216.34")) is True


def test_a_gigachat_image_that_rebinds_is_refused(monkeypatch):
    monkeypatch.setattr(gigachat_messages, "_host_is_public", lambda host: True)

    class _Resp:
        status_code: ClassVar[int] = 200
        headers: ClassVar[dict[str, str]] = {"content-type": "image/png"}
        extensions: ClassVar[dict[str, object]] = {"network_stream": _PeerStream("169.254.169.254")}

        async def aiter_bytes(self):
            yield b""

    class _Http:
        def stream(self, *_args, **_kwargs):
            class _Ctx:
                async def __aenter__(self):
                    return _Resp()

                async def __aexit__(self, *_exc):
                    return False

            return _Ctx()

    client = SimpleNamespace(http=_Http())

    async def run() -> None:
        with pytest.raises(HTTPException) as excinfo:
            await gigachat_messages._fetch_remote_image(client, "https://cdn.example/pic.png")
        assert excinfo.value.status_code == 400
        assert "non-public address on connect" in excinfo.value.detail

    asyncio.run(run())


def test_the_anthropic_gateway_does_not_forward_the_stop_sequences():
    payload = build_chat_request(
        {
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "stop_sequences": ["END"],
        },
        "deepseek-v4.1-flash",
    )
    assert "stop" not in payload
    assert convert_stop_sequences(["END"]) == ["END"]


def test_a_stale_byok_pool_never_unlinks_a_store_the_new_pool_adopted(monkeypatch):
    import danyapi.api.byok as byok_mod

    closed: list[list[str]] = []
    monkeypatch.setattr(byok_mod.settings, "cache_enabled", True)
    monkeypatch.setattr(byok_mod, "_close_pool_later", lambda pool, stores: closed.append([store.path.name for store in stores or ()]))
    monkeypatch.setattr(byok_mod, "_evict_stale_cache", lambda cache, stores, incoming=0: None)
    monkeypatch.setattr(byok_mod, "_evict_pools", lambda entries, reserve=0: None)
    monkeypatch.setattr(byok_mod, "_mark_pool_touched", lambda pool: None)
    monkeypatch.setattr(byok_mod, "_make_room_for_pool", lambda key: _true())
    monkeypatch.setattr(byok_mod, "_byok_cache_key", lambda tokens: "key")
    monkeypatch.setattr(byok_mod, "_byok_scope", lambda key: "scope")
    monkeypatch.setattr(byok_mod.app.state, "byok_pools", {"deepseek": {}}, raising=False)
    monkeypatch.setattr(byok_mod.app.state, "byok_stores", {"deepseek": {}}, raising=False)

    def _store(name: str):
        return SimpleNamespace(path=SimpleNamespace(name=name), enabled=True)

    def _pool():
        return SimpleNamespace(healthy=[], accounts=[])

    stale = _pool()
    shared = _store("deepseek-contexts-scope.json")
    orphan = _store("deepseek-sessions-scope.json")
    byok_mod.app.state.byok_pools["deepseek"]["key"] = stale
    byok_mod.app.state.byok_stores["deepseek"]["key"] = [shared, orphan]

    async def build(_provider, _tokens, _scope):
        return _pool(), [_store("deepseek-contexts-scope.json")]

    monkeypatch.setattr(byok_mod, "_build_byok_pool", build)

    async def run() -> None:
        pool = await byok_mod._byok_pool("deepseek", ["token"])
        assert pool is not stale
        assert byok_mod.app.state.byok_pools["deepseek"]["key"] is pool

    asyncio.run(run())
    assert closed == [["deepseek-sessions-scope.json"]]
