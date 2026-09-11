from danyapi.byok.models import ByokAccount, ByokUser, UserTokens


def test_byok_user_serde():
    u = ByokUser(id="u1", username="alice", password_hash="abc123")
    data = u.to_dict()
    assert data == {"id": "u1", "username": "alice", "password_hash": "abc123"}
    restored = ByokUser.from_dict(data)
    assert restored.id == "u1"
    assert restored.username == "alice"
    assert restored.password_hash == "abc123"


def test_byok_account_serde():
    a = ByokAccount(id="a1", provider="deepseek", token="sk-xxx")
    data = a.to_dict()
    assert data["provider"] == "deepseek"
    restored = ByokAccount.from_dict(data)
    assert restored.token == "sk-xxx"


def test_user_tokens_add_replace():
    ut = UserTokens(user_id="u1")
    acc = ByokAccount("a1", "deepseek", "sk-old")
    ut.add_or_replace(acc)
    assert len(ut.accounts) == 1
    new_acc = ByokAccount("a2", "deepseek", "sk-new")
    ut.add_or_replace(new_acc)
    assert len(ut.accounts) == 1
    assert ut.accounts[0].token == "sk-new"
    qwen_acc = ByokAccount("a3", "qwen", "qw-token")
    ut.add_or_replace(qwen_acc)
    assert len(ut.accounts) == 2
    assert ut.get_by_provider("qwen").token == "qw-token"


def test_user_tokens_remove():
    ut = UserTokens(user_id="u1")
    ut.add_or_replace(ByokAccount("a1", "deepseek", "sk"))
    ut.add_or_replace(ByokAccount("a2", "qwen", "qw"))
    assert ut.remove_by_provider("deepseek") is True
    assert len(ut.accounts) == 1
    assert ut.get_by_provider("deepseek") is None


def test_user_tokens_serde():
    ut = UserTokens(user_id="u1")
    ut.add_or_replace(ByokAccount("a1", "deepseek", "sk"))
    ut.updated_at = 12345.0
    data = ut.to_dict()
    restored = UserTokens.from_dict(data)
    assert restored.user_id == "u1"
    assert len(restored.accounts) == 1
    assert restored.accounts[0].provider == "deepseek"
