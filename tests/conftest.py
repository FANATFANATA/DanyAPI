import os

import pytest

os.environ["BYOK"] = "0"
os.environ["BYOK_MODE"] = "0"
os.environ["DANYAPI_BYOK_MODE"] = "0"
for _key in (
    "DEEPSEEK_TOKENS",
    "QWEN_TOKENS",
    "GIGACHAT_KEYS",
    "GIGACHAT_SCOPE",
    "DANYAPI_GIGACHAT_SCOPE",
    "DANYAPI_GIGACHAT_CA_FILE",
    "OPENCODE_KEYS",
    "OPENCODE_ENABLED",
    "ALICE_ENABLED",
    "ALICE_ACCOUNTS",
    "DUCKAI_ENABLED",
    "DUCKAI_ACCOUNTS",
):
    os.environ[_key] = ""


@pytest.fixture(autouse=True)
def _admin_token(monkeypatch):
    from danyapi.config import settings

    monkeypatch.setattr(settings, "admin_token", "test-admin-token", raising=False)
