import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import danyapi.api.models as models_mod
from danyapi.alice.client import MODEL_NAMES as ALICE_MODEL_NAMES
from danyapi.api.models import app, settings
from danyapi.api.state import POOL_ATTRS_BY_PROVIDER
from danyapi.gigachat.client import DEFAULT_SCOPE, GigaChatClient

DEEPSEEK_RAW = [
    {
        "id": "default",
        "name": "Instant",
        "owned_by": "deepseek",
        "model_type": "default",
        "is_default": True,
        "supports_thinking": True,
        "supports_search": True,
    }
]


class LockSpy:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.entered = 0

    async def __aenter__(self):
        self.entered += 1
        return await self.lock.acquire()

    async def __aexit__(self, exc_type, exc, tb):
        self.lock.release()


@pytest.fixture(autouse=True)
def _clean_model_state():
    model_attrs = models_mod.MODEL_ATTRS
    pool_attrs = POOL_ATTRS_BY_PROVIDER
    saved = {attr: list(getattr(app.state, attr, None) or []) for attr in model_attrs.values()}
    saved_pools = {attr: getattr(app.state, attr, None) for attr in pool_attrs.values()}
    for attr in saved:
        setattr(app.state, attr, [])
    for attr in saved_pools:
        setattr(app.state, attr, None)
    models_mod._MODEL_CACHE["key"] = None
    yield
    for attr, value in saved.items():
        setattr(app.state, attr, value)
    for attr, value in saved_pools.items():
        setattr(app.state, attr, value)
    models_mod._MODEL_CACHE["key"] = None


def test_probe_client_builds_a_gigachat_client_for_the_supplied_key():
    probe = models_mod._probe_client("gigachat", "auth-key")
    assert isinstance(probe, GigaChatClient)
    assert probe.key == "auth-key"
    assert probe.scope == DEFAULT_SCOPE
    assert models_mod._probe_client("gigachat", None) is None
    assert models_mod._probe_client("nosuch", "k") is None


async def test_close_probe_client_logs_and_swallows_a_failing_close(caplog):
    client = MagicMock()
    client.aclose = AsyncMock(side_effect=RuntimeError("already closed"))
    with caplog.at_level(logging.INFO, logger="danyapi.api"):
        await models_mod._close_probe_client(client)
    assert "model probe client close failed: already closed" in caplog.text


async def test_fetch_deepseek_models_skips_entries_without_a_string_id():
    client = MagicMock()
    client.fetch_models = AsyncMock(return_value=[{"name": "no id"}, {"id": "", "name": "empty"}, {"id": 7}, DEEPSEEK_RAW[0]])
    models = await models_mod._fetch_deepseek_models(client)
    assert models == [
        {
            "id": "default",
            "name": "Instant",
            "owned_by": "deepseek",
            "model_type": "chat",
            "upstream_type": "default",
            "is_default": True,
            "supports_thinking": True,
            "supports_search": True,
        }
    ]


async def test_fetch_deepseek_models_defaults_name_and_upstream_type_to_the_id():
    client = MagicMock()
    client.fetch_models = AsyncMock(return_value=[{"id": "solo"}])
    assert await models_mod._fetch_deepseek_models(client) == [
        {
            "id": "solo",
            "name": "solo",
            "owned_by": "deepseek",
            "model_type": "chat",
            "upstream_type": "solo",
            "is_default": False,
            "supports_thinking": False,
            "supports_search": False,
        }
    ]


async def test_fetch_qwen_models_classifies_t2i_and_t2v():
    client = MagicMock()
    client.fetch_models = AsyncMock(
        return_value=[
            {"id": "img", "name": "Image", "info": {"meta": {"chat_type": ["t2i"]}}},
            {"id": "vid", "info": {"meta": {"chat_type": ["t2v"]}}},
            {"id": "odd", "info": {"meta": {"chat_type": ["search"]}}},
        ]
    )
    models = await models_mod._fetch_qwen_models(client)
    assert [(model["id"], model["model_type"]) for model in models] == [("img", "image"), ("vid", "video"), ("odd", "chat")]
    assert models[0]["name"] == "Image"
    assert models[1]["name"] == "vid"
    assert models[2]["chat_types"] == ["search"]


async def test_fetch_qwen_models_stores_a_string_chat_type_verbatim():
    client = MagicMock()
    client.fetch_models = AsyncMock(return_value=[{"id": "weird", "info": {"meta": {"chat_type": "x-t2i"}}}])
    models = await models_mod._fetch_qwen_models(client)
    assert models[0]["chat_types"] == "x-t2i"
    assert models[0]["model_type"] == "image"


async def test_fetch_qwen_models_without_meta_defaults_to_chat():
    client = MagicMock()
    client.fetch_models = AsyncMock(return_value=[{"id": "plain"}, {"id": "badmeta", "info": "nope"}])
    models = await models_mod._fetch_qwen_models(client)
    assert [model["chat_types"] for model in models] == [[], []]
    assert [model["model_type"] for model in models] == ["chat", "chat"]


async def test_fetch_gigachat_models_skips_non_dict_and_embedding_entries():
    client = MagicMock()
    client.fetch_models = AsyncMock(
        return_value=[
            "garbage",
            {"name": "no id"},
            {"id": "GigaChat-embeddings", "type": "chat"},
            {"id": "GigaChat-Pro", "name": "Pro"},
        ]
    )
    assert await models_mod._fetch_gigachat_models(client) == [{"id": "GigaChat-Pro", "name": "Pro", "owned_by": "gigachat", "model_type": "chat"}]


async def test_fetch_gigachat_models_stringifies_numeric_ids():
    client = MagicMock()
    client.fetch_models = AsyncMock(return_value=[{"id": 7, "type": "chat"}])
    assert await models_mod._fetch_gigachat_models(client) == [{"id": "7", "name": "7", "owned_by": "gigachat", "model_type": "chat"}]


async def test_fetch_duckai_models_skips_non_dict_and_keeps_effort_tuple():
    client = MagicMock()
    client.fetch_models = AsyncMock(
        return_value=[
            "garbage",
            {"name": "no id"},
            {"id": "gpt-5.6-luna", "name": "Luna", "model_type": "reasoning", "provider": "openai", "efforts": ("low", "medium")},
            {"id": "bare"},
        ]
    )
    assert await models_mod._fetch_duckai_models(client) == [
        {
            "id": "gpt-5.6-luna",
            "name": "Luna",
            "owned_by": "duckai",
            "model_type": "reasoning",
            "provider": "openai",
            "efforts": ["low", "medium"],
        },
        {"id": "bare", "name": "bare", "owned_by": "duckai", "model_type": "chat", "provider": "", "efforts": []},
    ]


async def test_fetch_alice_models_ignores_its_client_argument(monkeypatch):
    spy = LockSpy()
    monkeypatch.setitem(models_mod._REFRESH_LOCKS, "alice", spy)
    client = MagicMock()
    client.fetch_models = AsyncMock(return_value=[{"id": "leaked"}])
    stored = await models_mod._store_models("alice", client)
    assert stored == [{"id": alias, "name": name, "owned_by": "alice", "model_type": "chat"} for alias, name in ALICE_MODEL_NAMES.items()]
    assert client.fetch_models.await_count == 0
    assert spy.entered == 1
    assert app.state.alice_models == stored


async def test_store_models_for_an_unknown_provider_never_fetches():
    app.state.duckai_models = [{"id": "d1", "name": "D", "owned_by": "duckai", "model_type": "chat"}]
    client = MagicMock()
    client.fetch_models = AsyncMock(return_value=[{"id": "nope"}])
    assert await models_mod._store_models("nope", client) == []
    assert client.fetch_models.await_count == 0
    assert await models_mod.refresh_provider_models("nope", client) == []


async def test_store_models_reuses_one_lock_object_per_provider(monkeypatch):
    spy = LockSpy()
    monkeypatch.setitem(models_mod._REFRESH_LOCKS, "qwen", spy)
    client = MagicMock()
    client.fetch_models = AsyncMock(return_value=[{"id": "q1", "name": "Q1", "owned_by": "qwen", "model_type": "chat"}])
    assert [model["id"] for model in await models_mod._store_models("qwen", client)] == ["q1"]
    assert [model["id"] for model in await models_mod._store_models("qwen", client)] == ["q1"]
    assert spy.entered == 2
    assert models_mod._REFRESH_LOCKS["qwen"] is spy


async def test_store_models_registers_a_lock_once_for_a_new_provider(monkeypatch):
    monkeypatch.delitem(models_mod._REFRESH_LOCKS, "duckai")
    client = MagicMock()
    client.fetch_models = AsyncMock(return_value=[{"id": "d1", "name": "D", "owned_by": "duckai", "model_type": "chat"}])
    await models_mod._store_models("duckai", client)
    first = models_mod._REFRESH_LOCKS["duckai"]
    assert first is not None
    assert first.locked() is False
    await models_mod._store_models("duckai", client)
    assert models_mod._REFRESH_LOCKS["duckai"] is first


async def test_refresh_models_skips_a_provider_whose_probe_client_raises(monkeypatch, caplog):
    def boom(provider, api_key):
        raise OSError("certificate verify failed")

    monkeypatch.setattr(models_mod, "_probe_client", boom)
    with caplog.at_level(logging.WARNING, logger="danyapi.api"):
        counts = await models_mod.refresh_models(api_key="k", providers=["gigachat"])
    assert counts == {}
    assert "gigachat model refresh skipped, client unusable: certificate verify failed" in caplog.text


async def test_refresh_models_closes_the_probe_client_it_created(monkeypatch):
    client = MagicMock()
    client.fetch_models = AsyncMock(return_value=[{"id": "d1", "name": "D", "owned_by": "duckai", "model_type": "chat"}])
    client.aclose = AsyncMock()
    monkeypatch.setattr(models_mod, "_probe_client", lambda provider, api_key: client)
    counts = await models_mod.refresh_models(api_key="k", providers=["duckai"])
    assert counts == {"duckai": 1}
    client.aclose.assert_awaited_once()
    assert client.fetch_models.await_count == 1


async def test_refresh_models_never_closes_a_pooled_client(monkeypatch):
    pooled = MagicMock()
    pooled.fetch_models = AsyncMock(return_value=[{"id": "q1", "name": "Q", "owned_by": "qwen", "model_type": "chat"}])
    pooled.aclose = AsyncMock()
    account = MagicMock()
    account.client = pooled
    pool = MagicMock()
    pool.accounts = [account]
    app.state.qwen_pool = pool
    counts = await models_mod.refresh_models(providers=["qwen"])
    assert counts == {"qwen": 1}
    pooled.aclose.assert_not_awaited()


async def test_model_refresh_loop_propagates_cancellation_from_refresh(monkeypatch):
    async def fake_sleep(_delay):
        return None

    async def fake_refresh(api_key=None, providers=None):
        raise asyncio.CancelledError

    monkeypatch.setattr(settings, "models_refresh_seconds", 1.0)
    monkeypatch.setattr(models_mod, "refresh_models", fake_refresh)
    monkeypatch.setattr(models_mod.asyncio, "sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        await models_mod.model_refresh_loop()


def test_default_deepseek_model_type_falls_back_to_the_first_entry():
    app.state.deepseek_models = [
        {"id": "expert", "upstream_type": "expert", "is_default": False},
        {"id": "default", "upstream_type": "default", "is_default": False},
    ]
    assert models_mod._default_deepseek_model_type() == "expert"
    app.state.deepseek_models = []
    assert models_mod._default_deepseek_model_type() == models_mod.DEEPSEEK_DEFAULT_MODEL_TYPE


def test_default_deepseek_model_type_uses_the_flagged_entry():
    app.state.deepseek_models = [
        {"id": "expert", "upstream_type": "expert", "is_default": False},
        {"id": "default", "is_default": True},
    ]
    assert models_mod._default_deepseek_model_type() == "default"


def test_resolve_provider_routes_by_prefix_alias_and_stored_catalog():
    assert models_mod._resolve_provider("GigaChat-Pro") == "gigachat"
    app.state.alice_models = [{"id": "custom-alice", "name": "C", "owned_by": "alice", "model_type": "chat"}]
    models_mod._MODEL_CACHE["key"] = None
    assert models_mod._resolve_provider("custom-alice") == "alice"


def test_resolve_provider_unknown_is_404():
    with pytest.raises(HTTPException) as excinfo:
        models_mod._resolve_provider("gpt-4")
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "Unknown model: gpt-4"


def test_resolve_provider_rebuilds_the_whole_catalog_per_routing_decision(monkeypatch):
    measurements: list[int] = []
    real_source = models_mod._model_source

    def measured():
        built = real_source()
        measurements.append(len(built))
        return built

    monkeypatch.setattr(models_mod, "_model_source", measured)
    app.state.alice_models = [{"id": "custom-alice", "name": "C", "owned_by": "alice", "model_type": "chat"}]
    models_mod._MODEL_CACHE["key"] = None
    assert models_mod._resolve_provider("custom-alice") == "alice"
    single_calls = len(measurements)
    single_elements = sum(measurements)

    app.state.alice_models = [{"id": f"custom-alice-{index}", "name": "C", "owned_by": "alice", "model_type": "chat"} for index in range(20)]
    models_mod._MODEL_CACHE["key"] = None
    measurements.clear()
    assert models_mod._resolve_provider("custom-alice-19") == "alice"
    assert len(measurements) == single_calls
    assert sum(measurements) == 20 * single_elements


async def test_get_model_returns_the_cached_entry_and_404s_for_unknown():
    app.state.deepseek_models = [
        {"id": "default", "name": "Instant", "owned_by": "deepseek", "model_type": "chat", "upstream_type": "default", "is_default": True}
    ]
    found = await models_mod.get_model("default")
    client = TestClient(app)
    listed = {entry["id"]: entry for entry in client.get("/v1/models").json()["data"]}
    over_http = client.get("/v1/models/default")
    missing = client.get("/v1/models/nope")
    client.close()
    assert found == listed["default"]
    assert over_http.json() == found
    assert found["owned_by"] == "deepseek"
    assert found["object"] == "model"
    assert missing.status_code == 404
    with pytest.raises(HTTPException) as excinfo:
        await models_mod.get_model("nope")
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "The model 'nope' does not exist"


def test_models_state_cache_is_reused_until_the_source_changes():
    app.state.qwen_models = [{"id": "q1", "name": "Q", "owned_by": "qwen", "model_type": "chat"}]
    first = models_mod._models_state()
    assert models_mod._models_state() is first
    assert models_mod._models_state() is models_mod._all_models()
    app.state.qwen_models = [{"id": "q2", "name": "Q", "owned_by": "qwen", "model_type": "chat"}]
    second = models_mod._models_state()
    assert second is not first
    assert [entry["id"] for entry in second] == ["q2"]


def test_models_state_defaults_owner_and_model_type():
    app.state.qwen_models = [{"id": "bare"}]
    assert models_mod._models_state() == [
        {
            "id": "bare",
            "object": "model",
            "created": models_mod.MODEL_CREATED_AT,
            "owned_by": "qwen",
            "name": None,
            "model_type": "chat",
        }
    ]


def test_cached_ids_returns_an_empty_set_for_an_unknown_key():
    assert models_mod._cached_ids("nosuch_provider_ids") == set()


def test_known_ids_falls_back_to_the_static_catalogs():
    assert "gpt-5.6-luna" in models_mod._known_ids("duckai")
    assert set(models_mod.ALICE_MODEL_IDS) <= models_mod._known_ids("alice")
    assert models_mod._known_ids("deepseek") == set()


def test_is_deepseek_model_accepts_prefixes_aliases_and_known_ids():
    assert models_mod._is_deepseek_model("deepseek-anything") is True
    assert models_mod._is_deepseek_model("deepseek-v4.1-flash") is True
    assert models_mod._is_deepseek_model("deepseek-v4.1-flash-thinking") is True
    assert models_mod._is_deepseek_model("qwen3.8-max") is False


def test_is_reasoning_model_and_output_truncated():
    assert models_mod._is_reasoning_model("default-thinking") is True
    assert models_mod._is_reasoning_model("default") is False
    assert models_mod._output_truncated("CONTEXT_LENGTH_EXCEEDED") is True
    assert models_mod._output_truncated("FINISHED") is False


def test_resolve_model_uses_the_matching_upstream_type():
    app.state.deepseek_models = [
        {"id": "default", "upstream_type": "default", "is_default": True},
        {"id": "expert", "upstream_type": "expert", "is_default": False},
    ]
    assert models_mod._resolve_model("EXPERT") == "expert"
    assert models_mod._resolve_model("expert-thinking") == "expert"
    app.state.deepseek_models = [{"id": "expert", "is_default": False}]
    assert models_mod._resolve_model("expert") == "expert"


def test_provider_enabled_follows_pool_presence_outside_byok():
    assert models_mod.provider_enabled("alice") is False
    app.state.alice_pool = MagicMock()
    assert models_mod.provider_enabled("alice") is True
    assert models_mod.provider_enabled("nosuch") is False
