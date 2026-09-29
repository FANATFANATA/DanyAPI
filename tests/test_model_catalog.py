import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

import danyapi.api.models as models_mod
import danyapi.api.openai as openai_mod
from danyapi.api.openai import app, settings

DEEPSEEK_RAW = [
    {
        "id": "default",
        "name": "Instant",
        "owned_by": "deepseek",
        "model_type": "default",
        "is_default": True,
        "supports_thinking": True,
        "supports_search": True,
    },
    {
        "id": "expert",
        "name": "Expert",
        "owned_by": "deepseek",
        "model_type": "expert",
        "is_default": False,
        "supports_thinking": True,
        "supports_search": False,
    },
]

DUCKAI_BUNDLE = (
    'x=[{model:"gpt-5.6-luna",modelName:"GPT-5.6",modelVariant:"Luna",modelShortName:"5.6 Luna",'
    'createdBy:"OpenAI",modelType:"reasoning",supportedReasoningEffort:["none","low"],moderationLevel:"HIGH",'
    "availableTo:[n.ac.Internal,n.ac.Free],inputCharLimit:16e3,costRank:2},"
    '{model:"gpt-5.6-terra",modelName:"GPT-5.6",modelVariant:"Terra",modelShortName:"5.6 Terra",'
    'createdBy:"OpenAI",modelType:"reasoning",supportedReasoningEffort:["none","low","medium"],'
    "availableTo:[n.ac.Plus,n.ac.Pro],costRank:7},"
    '{model:"voice-mode",serviceModelId:"gpt-realtime",modelName:"Voice Chat",modelShortName:"Voice Chat",'
    'createdBy:"OpenAI",modelType:"general",availableTo:[],costRank:1},'
    '{model:"gpt-4o",deprecated:true}],y=1'
)

DUCKAI_PAGE = '<html><head><script src="/dist/duckai-dist/entry.duckai.abc123.js"></script></head></html>'


@pytest.fixture(autouse=True)
def _clean_model_state():
    saved = {attr: list(getattr(app.state, attr, None) or []) for attr in models_mod.MODEL_ATTRS.values()}
    for attr in saved:
        setattr(app.state, attr, [])
    saved_qwen_pool = getattr(app.state, "qwen_pool", None)
    openai_mod._MODEL_CACHE["key"] = None
    yield
    for attr, value in saved.items():
        setattr(app.state, attr, value)
    app.state.qwen_pool = saved_qwen_pool
    openai_mod._MODEL_CACHE["key"] = None


def _request(headers: dict[str, str] | None = None) -> Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    header_bytes = [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/v1/models",
            "query_string": b"",
            "headers": header_bytes,
            "client": ("127.0.0.1", 5555),
            "server": ("testserver", 80),
            "scheme": "http",
        },
        receive=receive,
    )


def test_parse_catalog_keeps_only_free_tier():
    from danyapi.duckai.client import parse_catalog

    entries = parse_catalog(DUCKAI_BUNDLE)
    assert [entry["id"] for entry in entries] == ["gpt-5.6-luna"]
    assert entries[0]["name"] == "GPT-5.6 Luna"
    assert entries[0]["provider"] == "openai"
    assert entries[0]["efforts"] == ("none", "low")


def test_parse_catalog_orders_by_cost_rank():
    from danyapi.duckai.client import parse_catalog

    bundle = DUCKAI_BUNDLE.replace("availableTo:[n.ac.Plus,n.ac.Pro]", "availableTo:[n.ac.Free,n.ac.Pro]")
    entries = parse_catalog(bundle)
    assert [entry["id"] for entry in entries] == ["gpt-5.6-luna", "gpt-5.6-terra"]
    assert [entry["cost_rank"] for entry in entries] == [1, 2]


def test_parse_catalog_of_garbage_is_empty():
    from danyapi.duckai.client import parse_catalog

    assert parse_catalog("nothing here") == ()


async def test_duckai_fetch_models_scrapes_page_and_bundle():
    from danyapi.duckai.client import DuckAIClient

    client = DuckAIClient(timeout=5.0)
    pages = {
        "/": DUCKAI_PAGE,
        "/dist/duckai-dist/entry.duckai.abc123.js": DUCKAI_BUNDLE,
    }
    seen: list[str] = []

    async def fake_get(path, headers=None):
        seen.append(path)
        return MagicMock(text=pages[path], headers={})

    client.http.get = fake_get
    try:
        models = await client.fetch_models()
    finally:
        await client.aclose()
    assert [model["id"] for model in models] == ["gpt-5.6-luna"]
    assert seen == ["/", "/dist/duckai-dist/entry.duckai.abc123.js"]


async def test_duckai_fetch_models_keeps_catalog_when_page_has_no_bundle():
    from danyapi.duckai.client import MODEL_CATALOG, DuckAIClient

    client = DuckAIClient(timeout=5.0)

    async def fake_get(path, headers=None):
        return MagicMock(text="<html></html>", headers={})

    client.http.get = fake_get
    try:
        models = await client.fetch_models()
    finally:
        await client.aclose()
    assert [model["id"] for model in models] == [entry["id"] for entry in MODEL_CATALOG]


async def test_duckai_client_uses_live_efforts():
    from danyapi.duckai.client import DuckAIClient

    client = DuckAIClient(timeout=5.0)
    try:
        client._catalog = ({"id": "m", "efforts": ("low", "medium")},)
        body = client._request_body([], "m", "medium", can_use_tools=False, can_use_web_search=False)
        assert body["reasoningEffort"] == "medium"
        unsupported = client._request_body([], "m", "high", can_use_tools=False, can_use_web_search=False)
        assert unsupported["reasoningEffort"] == "none"
        absent = client._request_body([], "unknown", "low", can_use_tools=False, can_use_web_search=False)
        assert absent["reasoningEffort"] == "low"
    finally:
        await client.aclose()


async def test_deepseek_fetch_models_maps_upstream_types():
    client = MagicMock()
    client.fetch_models = AsyncMock(return_value=DEEPSEEK_RAW)
    models = await openai_mod._fetch_deepseek_models(client)
    assert [model["id"] for model in models] == ["default", "expert"]
    assert models[0]["upstream_type"] == "default"
    assert models[0]["model_type"] == "chat"
    assert models[0]["is_default"] is True
    assert models[1]["is_default"] is False


async def test_store_models_deepseek_then_resolve_and_route():
    client = MagicMock()
    client.fetch_models = AsyncMock(return_value=DEEPSEEK_RAW)
    await openai_mod._store_models("deepseek", client)
    assert models_mod._resolve_model("default") == "default"
    assert models_mod._resolve_model("expert-thinking") == "expert"
    assert openai_mod._resolve_provider("expert") == "deepseek"
    assert openai_mod._resolve_provider("default-thinking") == "deepseek"
    assert openai_mod._default_deepseek_model_type() == "default"


async def test_legacy_alias_resolves_to_live_default():
    client = MagicMock()
    client.fetch_models = AsyncMock(return_value=DEEPSEEK_RAW)
    await openai_mod._store_models("deepseek", client)
    assert models_mod._resolve_model("deepseek-v4.1-flash") == "default"
    assert models_mod._resolve_model("deepseek-v4.1-flash-thinking") == "default"
    assert openai_mod._resolve_provider("deepseek-v4.1-flash") == "deepseek"


async def test_resolve_model_unknown_raises_404():
    with pytest.raises(HTTPException) as excinfo:
        models_mod._resolve_model("nope")
    assert excinfo.value.status_code == 404
    with pytest.raises(HTTPException) as excinfo:
        models_mod._resolve_model("nope-thinking")
    assert excinfo.value.status_code == 404


def test_models_list_exposes_deepseek_ids_and_thinking_siblings():
    app.state.deepseek_models = [
        {
            "id": "default",
            "name": "Instant",
            "owned_by": "deepseek",
            "model_type": "chat",
            "upstream_type": "default",
            "is_default": True,
        }
    ]
    client = TestClient(app)
    data = client.get("/v1/models").json()
    client.close()
    ids = [entry["id"] for entry in data["data"]]
    assert "default" in ids
    assert "default-thinking" in ids
    owner = {entry["id"]: entry["owned_by"] for entry in data["data"]}
    assert owner["default"] == "deepseek"


def test_models_endpoint_refreshes_with_api_key(monkeypatch):
    calls: list[dict] = []

    async def fake_refresh(api_key=None, providers=None):
        calls.append({"api_key": api_key, "providers": providers})
        return {}

    monkeypatch.setattr(models_mod, "refresh_models", fake_refresh)
    client = TestClient(app)
    client.get("/v1/models", headers={"Authorization": "Bearer tok"})
    client.get("/v1/models?refresh=1")
    client.get("/v1/models", headers={"x-api-key": "hdr-key"})
    client.close()
    assert calls[0]["api_key"] == "tok"
    assert calls[1]["api_key"] is None
    assert calls[2]["api_key"] == "hdr-key"


def test_models_endpoint_without_key_does_not_refresh(monkeypatch):
    calls: list[str] = []

    async def fake_refresh(api_key=None, providers=None):
        calls.append(str(api_key))
        return {}

    monkeypatch.setattr(models_mod, "refresh_models", fake_refresh)
    client = TestClient(app)
    client.get("/v1/models")
    client.close()
    assert calls == []


async def test_refresh_models_uses_pool_client_when_present(monkeypatch):
    pooled = MagicMock()
    pooled.fetch_models = AsyncMock(return_value=[{"id": "m1", "name": "M1"}])
    account = MagicMock()
    account.client = pooled
    pool = MagicMock()
    pool.accounts = [account]
    app.state.qwen_pool = pool
    captured: list[str] = []

    real_probe = models_mod._probe_client

    def fake_probe(provider, api_key):
        captured.append(provider)
        return real_probe(provider, api_key)

    monkeypatch.setattr(models_mod, "_probe_client", fake_probe)
    counts = await openai_mod.refresh_models(providers=["qwen"])
    assert counts["qwen"] == 1
    assert captured == []
    assert app.state.qwen_models[0]["id"] == "m1"


async def test_refresh_models_skips_providers_without_credentials():
    counts = await openai_mod.refresh_models(api_key=None, providers=["gigachat"])
    assert "gigachat" not in counts


async def test_refresh_models_gigachat_uses_supplied_key(monkeypatch):
    client = MagicMock()
    client.fetch_models = AsyncMock(
        return_value=[
            {"id": "GigaChat-2", "name": "GigaChat 2", "type": "chat"},
            {"id": "GigaChat-embeddings", "name": "emb", "type": "chat"},
            {"id": "GigaChat-3", "name": "GigaChat 3", "type": "image"},
        ]
    )
    client.aclose = AsyncMock()
    monkeypatch.setattr(models_mod, "_probe_client", lambda provider, api_key: client)
    counts = await openai_mod.refresh_models(api_key="auth-key", providers=["gigachat"])
    assert counts["gigachat"] == 1
    assert [entry["id"] for entry in app.state.gigachat_models] == ["GigaChat-2"]
    client.aclose.assert_awaited()


async def test_refresh_models_alice_needs_no_client():
    counts = await openai_mod.refresh_models(providers=["alice"])
    assert counts["alice"] == 3
    assert [entry["id"] for entry in app.state.alice_models] == ["alice", "alice-ai", "yagpt"]


async def test_refresh_models_keeps_last_known_on_failure(monkeypatch):
    client = MagicMock()
    client.fetch_models = AsyncMock(side_effect=RuntimeError("down"))
    client.aclose = AsyncMock()
    monkeypatch.setattr(models_mod, "_probe_client", lambda provider, api_key: client)
    app.state.qwen_models = [{"id": "kept", "name": "kept", "owned_by": "qwen", "model_type": "chat"}]
    counts = await openai_mod.refresh_models(providers=["qwen"])
    assert counts["qwen"] == 1
    assert [entry["id"] for entry in app.state.qwen_models] == ["kept"]


async def test_model_refresh_loop_disabled_by_zero(monkeypatch):
    monkeypatch.setattr(settings, "models_refresh_seconds", 0.0)
    await asyncio.wait_for(openai_mod.model_refresh_loop(), timeout=1.0)


async def test_model_refresh_loop_survives_cycle_error(monkeypatch):
    calls: list[int] = []

    async def fake_refresh(api_key=None, providers=None):
        calls.append(1)
        raise RuntimeError("boom")

    async def fake_sleep(_delay):
        if len(calls) >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(settings, "models_refresh_seconds", 1.0)
    monkeypatch.setattr(models_mod, "refresh_models", fake_refresh)
    monkeypatch.setattr(models_mod.asyncio, "sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        await openai_mod.model_refresh_loop()
    assert len(calls) == 2


def test_provider_needs_api_key_matches_keyless_providers():
    from danyapi.api.state import provider_needs_api_key

    assert provider_needs_api_key("alice") is False
    assert provider_needs_api_key("duckai") is False
    assert provider_needs_api_key("deepseek") is True
    assert provider_needs_api_key("qwen") is True
    assert provider_needs_api_key("gigachat") is True


def test_provider_enabled_follows_byok_mode():
    app.state.byok = True
    try:
        assert openai_mod.provider_enabled("deepseek") is True
        assert openai_mod.provider_enabled("gigachat") is True
    finally:
        app.state.byok = False
    app.state.alice_pool = None
    assert openai_mod.provider_enabled("alice") is False
    app.state.alice_pool = MagicMock()
    try:
        assert openai_mod.provider_enabled("alice") is True
    finally:
        app.state.alice_pool = None


def test_duckai_routing_falls_back_to_known_catalog_without_state():
    from danyapi.duckai.client import MODEL_CATALOG

    for entry in MODEL_CATALOG:
        assert openai_mod._resolve_provider(entry["id"]) == "duckai"


def test_alice_routing_uses_aliases_without_state():
    for alias in openai_mod.ALICE_MODEL_IDS:
        assert openai_mod._resolve_provider(alias) == "alice"


def test_health_reports_known_models_per_provider():
    app.state.alice_models = [
        {"id": "alice", "name": "Alice", "owned_by": "alice", "model_type": "chat"},
        {"id": "yagpt", "name": "YaGPT", "owned_by": "alice", "model_type": "chat"},
    ]
    app.state.alice_pool = MagicMock()
    app.state.alice_pool.stats.return_value = {"accounts": 1, "healthy": 1, "broken": 0}
    try:
        client = TestClient(app)
        payload = client.get("/health", headers={"x-api-key": settings.admin_token}).json()
        public = client.get("/health").json()
        client.close()
        assert payload["alice"] is True
        assert payload["alice_stats"]["models"] == 2
        assert public == {"status": "ok"}
    finally:
        app.state.alice_pool = None
