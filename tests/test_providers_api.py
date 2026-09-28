import base64

import httpx
import pytest
from fastapi.testclient import TestClient

from danyapi.accounts import AccountPool
from danyapi.alice import api as alice_api
from danyapi.alice.accounts import AliceAccount
from danyapi.alice.client import AliceClient, AliceStream
from danyapi.api.models import _resolve_provider
from danyapi.api.openai import app
from danyapi.gigachat import api as gigachat_api
from danyapi.gigachat.accounts import GigaChatAccount
from danyapi.gigachat.client import GigaChatClient

KEY = base64.b64encode(b"id:secret").decode()


@pytest.fixture(autouse=True)
def _clean():
    saved = {}
    for attr in ("gigachat_pool", "alice_pool", "gigachat_models", "alice_models"):
        saved[attr] = getattr(app.state, attr, None)
        setattr(app.state, attr, None)
    app.state.byok = False
    yield
    for attr, value in saved.items():
        setattr(app.state, attr, value)


def test_resolve_provider_routes_all_four():
    assert _resolve_provider("deepseek-v4.1-flash") == "deepseek"
    assert _resolve_provider("qwen3.8-max") == "qwen"
    assert _resolve_provider("GigaChat") == "gigachat"
    assert _resolve_provider("GigaChat-2-Max") == "gigachat"
    assert _resolve_provider("alice") == "alice"
    assert _resolve_provider("alice-ai") == "alice"
    assert _resolve_provider("yagpt") == "alice"


def test_resolve_provider_rejects_unknown():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        _resolve_provider("totally-unknown")
    assert excinfo.value.status_code == 404


def test_health_reports_new_providers():
    body = TestClient(app).get("/health").json()
    assert "gigachat" in body
    assert "alice" in body
    assert body["gigachat"] is False
    assert body["alice"] is False


def test_models_endpoint_includes_new_providers():
    app.state.gigachat_models = [{"id": "GigaChat", "name": "GigaChat", "owned_by": "gigachat", "model_type": "chat"}]
    app.state.alice_models = [{"id": "alice", "name": "Alice AI", "owned_by": "alice", "model_type": "chat"}]
    data = TestClient(app).get("/v1/models").json()["data"]
    owners = {m["id"]: m["owned_by"] for m in data}
    assert owners["GigaChat"] == "gigachat"
    assert owners["alice"] == "alice"


def test_chat_completions_503_when_gigachat_not_configured():
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "GigaChat", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 503
    assert "gigachat" in resp.json()["error"]["message"]


def test_chat_completions_503_when_alice_not_configured():
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "alice", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 503
    assert "ALICE_ENABLED" in resp.json()["error"]["message"]


class _AliceStub(AliceClient):
    async def ask(self, prompt):
        stream = AliceStream()
        stream.content = "ok"
        return stream

    async def aclose(self):
        return None


class _GigaStub(GigaChatClient):
    def __init__(self):
        super().__init__(key=KEY)
        self.sent: list[dict] = []

    async def chat(self, body, model):
        self.sent.append(body)
        payload = {
            "id": "chatcmpl-x",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "привет"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5, "precached_prompt_tokens": 1},
        }
        return httpx.Response(200, json=payload, request=httpx.Request("POST", "https://api.giga.chat/v1/chat/completions"))

    async def upload_file(self, filename, data, content_type, purpose="general"):
        return "file-1"

    async def aclose(self):
        return None


def test_gigachat_chat_completion_end_to_end():
    client = _GigaStub()
    app.state.gigachat_pool = AccountPool([GigaChatAccount(0, client, stable_id="k1")], label="gigachat")
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "GigaChat", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "привет"
    assert body["usage"]["prompt_tokens"] == 3
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 1
    assert client.sent[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_alice_chat_completion_end_to_end():
    app.state.alice_pool = AccountPool([AliceAccount(0, _AliceStub(), stable_id="alice")], label="alice")
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "alice", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "ok"
    assert body["model"] == "alice"


def test_alice_stream_end_to_end():
    app.state.alice_pool = AccountPool([AliceAccount(0, _AliceStub(), stable_id="alice")], label="alice")
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "alice", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    assert '"content":"ok"' in resp.text or '"content": "ok"' in resp.text
    assert "data: [DONE]" in resp.text


def test_alice_rejects_file_attachments():
    app.state.alice_pool = AccountPool([AliceAccount(0, _AliceStub(), stable_id="alice")], label="alice")
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "alice",
            "messages": [{"role": "user", "content": "hi"}],
            "files": [{"name": "a.txt", "content": "x"}],
        },
    )
    assert resp.status_code == 400
    assert "does not support file attachments" in resp.json()["error"]["message"]


def test_completions_endpoint_uses_gigachat():
    app.state.gigachat_pool = AccountPool([GigaChatAccount(0, _GigaStub(), stable_id="k1")], label="gigachat")
    resp = TestClient(app).post("/v1/completions", json={"model": "GigaChat", "prompt": "hi"})
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["text"] == "привет"


def test_anthropic_messages_accepts_gigachat_model():
    app.state.gigachat_pool = AccountPool([GigaChatAccount(0, _GigaStub(), stable_id="k1")], label="gigachat")
    resp = TestClient(app).post(
        "/v1/messages",
        json={
            "model": "GigaChat",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "привет"


def test_handler_registry_covers_all_providers():
    from danyapi.api.chats import CHAT_HANDLERS

    assert set(CHAT_HANDLERS) == {"deepseek", "qwen", "gigachat", "alice"}


def test_provider_apis_are_importable():
    assert callable(gigachat_api.collect_non_stream)
    assert callable(gigachat_api.stream_openai)
    assert callable(alice_api.collect_non_stream)
    assert callable(alice_api.stream_openai)
