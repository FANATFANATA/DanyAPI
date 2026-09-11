import tempfile
from pathlib import Path

from danyapi.byok.manager import ByokManager, _hash_password
from danyapi.byok.store import UserStore


def _make_mgr():
    tmpdir = tempfile.mkdtemp()
    store = UserStore(Path(tmpdir) / "byok")
    return ByokManager(store=store, salt="test-salt")


def test_login_success():
    mgr = _make_mgr()
    ok, uid = mgr.register("alice", "pass123")
    assert ok is True
    session = mgr.login("alice", "pass123")
    assert session is not None
    assert mgr.auth_check(session) == uid


def test_login_wrong_password():
    mgr = _make_mgr()
    pw_hash = _hash_password("correct", "test-salt")
    mgr._store.register_user("uid1", "bob", pw_hash)
    result = mgr.login("bob", "wrong")
    assert result is None


def test_login_unknown_user():
    mgr = _make_mgr()
    result = mgr.login("nobody", "nope")
    assert result is None


def test_register_twice_fails():
    mgr = _make_mgr()
    ok1, _uid1 = mgr.register("newuser", "password")
    assert ok1 is True
    ok2, _uid2 = mgr.register("newuser", "password")
    assert ok2 is False


def test_logout_invalidates_session():
    mgr = _make_mgr()
    ok, uid = mgr.register("logme", "correctpass")
    assert ok is True
    session = mgr.login("logme", "correctpass")
    assert session is not None
    assert mgr.auth_check(session) == uid
    mgr.logout(session)
    assert mgr.auth_check(session) is None


def test_add_and_get_token():
    mgr = _make_mgr()
    uid = "uid_tok"
    mgr._store.register_user(uid, "tokener", "h")
    result = mgr.add_token(uid, "deepseek", "sk-xxx")
    assert result["provider"] == "deepseek"
    token = mgr.get_provider_token(uid, "deepseek")
    assert token == "sk-xxx"
    assert mgr.user_has_token(uid, "deepseek") is True
    assert mgr.user_has_token(uid, "qwen") is False


def test_remove_token():
    mgr = _make_mgr()
    uid = "uid_rem"
    mgr._store.register_user(uid, "removr", "h")
    mgr.add_token(uid, "qwen", "qw-token")
    removed = mgr.remove_token(uid, "qwen")
    assert removed is True
    assert mgr.get_provider_token(uid, "qwen") is None


def test_build_account_returns_none_when_missing():
    mgr = _make_mgr()
    uid = "uid_acc"
    mgr._store.register_user(uid, "accnt", "h")
    acc = mgr.build_account(uid, "deepseek")
    assert acc is None
