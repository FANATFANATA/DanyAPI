from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def enabled_app():
    import tempfile

    from danyapi.api.openai import app
    from danyapi.byok import ByokManager, set_manager

    tmpdir = tempfile.mkdtemp()
    with patch("danyapi.api.openai.settings") as mock_settings:
        mock_settings.byok_mode = True
        mock_settings.session_ttl = 3600.0
        mgr = ByokManager.from_settings.__func__(None) if False else None  # noqa: F841
        store_path = Path(tmpdir) / "byok"
        store_path.mkdir(parents=True, exist_ok=True)
        from danyapi.byok.manager import ByokManager as BM
        from danyapi.byok.store import UserStore

        ustore = UserStore(store_path)
        byok_mgr = BM(store=ustore, salt="test-salt-for-api")
        set_manager(byok_mgr)
        yield app, byok_mgr
        set_manager(None)


def test_byok_login_register(enabled_app):
    app, _mgr = enabled_app
    client = TestClient(app)
    res = client.post("/byok/register", json={"username": "testuser", "password": "secret"})
    assert res.status_code == 200
    data = res.json()
    assert data["registered"] is True
    assert "user_id" in data

    res2 = client.post("/byok/register", json={"username": "testuser", "password": "secret"})
    assert res2.status_code == 200
    assert res2.json()["registered"] is False

    res3 = client.post("/byok/login", json={"username": "testuser", "password": "secret"})
    assert res3.status_code == 200
    assert "session_key" in res3.json()


def test_byok_add_token(enabled_app):
    app, _mgr = enabled_app
    client = TestClient(app)
    # register + login to get cookie
    client.post("/byok/register", json={"username": "tokenuser", "password": "p"})
    res = client.post("/byok/login", json={"username": "tokenuser", "password": "p"})
    session_key = res.json()["session_key"]
    resp = client.post(
        "/byok/token",
        cookies={"byok_session": session_key},
        json={"provider": "deepseek", "token": "sk-abc123"},
    )
    assert resp.status_code == 200
    assert resp.json()["token"] == "sk-abc123"


def test_byok_get_tokens(enabled_app):
    app, _mgr = enabled_app
    client = TestClient(app)
    client.post("/byok/register", json={"username": "listuser", "password": "p"})
    res = client.post("/byok/login", json={"username": "listuser", "password": "p"})
    session_key = res.json()["session_key"]
    client.post(
        "/byok/token",
        cookies={"byok_session": session_key},
        json={"provider": "qwen", "token": "qw-token"},
    )
    res2 = client.get("/byok/tokens", cookies={"byok_session": session_key})
    assert res2.status_code == 200
    tokens_data = res2.json()["tokens"]
    assert len(tokens_data) == 1
    assert tokens_data[0]["provider"] == "qwen"


def test_byok_delete_token(enabled_app):
    app, _mgr = enabled_app
    client = TestClient(app)
    client.post("/byok/register", json={"username": "deluser", "password": "p"})
    res = client.post("/byok/login", json={"username": "deluser", "password": "p"})
    session_key = res.json()["session_key"]
    client.post(
        "/byok/token",
        cookies={"byok_session": session_key},
        json={"provider": "deepseek", "token": "sk-del"},
    )
    del_res = client.delete("/byok/token/deepseek", cookies={"byok_session": session_key})
    assert del_res.status_code == 200
    assert del_res.json()["removed"] is True


def test_byok_me(enabled_app):
    app, _mgr = enabled_app
    client = TestClient(app)
    client.post("/byok/register", json={"username": "meuser", "password": "p"})
    res = client.post("/byok/login", json={"username": "meuser", "password": "p"})
    session_key = res.json()["session_key"]
    me_res = client.get("/byok/me", cookies={"byok_session": session_key})
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["authenticated"] is True
    assert me_data["username"] == "meuser"


def test_byok_status_enabled(enabled_app):
    app, _mgr = enabled_app
    client = TestClient(app)
    res = client.get("/byok/status")
    assert res.status_code == 200
    assert res.json()["enabled"] is True


def test_byok_401_without_cookie(enabled_app):
    app, _mgr = enabled_app
    client = TestClient(app)
    res = client.get("/byok/me")
    assert res.status_code == 401
