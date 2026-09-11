import tempfile
from pathlib import Path

from danyapi.byok.store import UserStore


def _make_store() -> UserStore:
    tmpdir = tempfile.mkdtemp()
    return UserStore(Path(tmpdir) / "byok")


def test_register_and_auth():
    store = _make_store()
    ok = store.register_user("uid1", "alice", "hash_alice")
    assert ok is True
    ok2 = store.register_user("uid1", "bob", "hash_bob")
    assert ok2 is False
    user = store.get_user_by_name("alice")
    assert user is not None
    uid_val = user["id"]
    assert uid_val == "uid1"
    user2 = store.get_user("uid1")
    assert user2 is not None
    assert user2["username"] == "alice"


def test_token_add_remove():
    store = _make_store()
    uid = "uid_tok"
    store.register_user(uid, "tester", "h")
    store.add_token(uid, "deepseek", "sk-1")
    assert store.get_token_count(uid, "deepseek") == 1
    assert store.get_token(uid, "deepseek") == "sk-1"
    tokens_list = store.get_tokens_list(uid)
    assert len(tokens_list) == 1
    store.remove_token(uid, "deepseek")
    assert store.get_token(uid, "deepseek") is None
    assert store.get_token_count(uid, "deepseek") == 0


def test_token_replace():
    store = _make_store()
    uid = "uid_rep"
    store.register_user(uid, "replacer", "h")
    store.add_token(uid, "qwen", "qw-old")
    store.add_token(uid, "qwen", "qw-new")
    assert store.get_token(uid, "qwen") == "qw-new"
    assert store.get_token_count(uid, "qwen") == 1


def test_delete_user():
    store = _make_store()
    store.register_user("uid_del", "delme", "h")
    store.add_token("uid_del", "deepseek", "sk")
    result = store.delete_user("uid_del")
    assert result is True
    assert store.get_user("uid_del") is None
    assert store.get_user_by_name("delme") is None
    assert store.delete_user("uid_del") is False


def test_multiple_providers():
    store = _make_store()
    uid = "uid_multi"
    store.register_user(uid, "multi", "h")
    store.add_token(uid, "deepseek", "ds-token")
    store.add_token(uid, "qwen", "qw-token")
    assert store.get_token_count(uid, "deepseek") == 1
    assert store.get_token_count(uid, "qwen") == 1
    assert len(store.get_tokens_list(uid)) == 2
