import os

import pytest

from danyapi.config import Settings


@pytest.fixture
def settings_for():
    def _make(env):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "environ", env, raising=False)
            return Settings()

    return _make


def test_defaults(settings_for):
    s = settings_for({})
    assert s.host == "0.0.0.0"
    assert s.port == 8000
    assert s.deepseek_tokens == []
    assert s.qwen_tokens == []
    assert s.deepseek_email == ""
    assert s.deepseek_password == ""
    assert s.qwen_email == ""
    assert s.qwen_password == ""
    assert s.timeout == 60.0
    assert s.acquire_timeout is None
    assert s.session_cache_size == 128
    assert s.session_ttl == 3600.0
    assert s.log_level == "INFO"
    assert s.log_file == ""
    assert s.log_max_bytes == 10485760
    assert s.log_backup_count == 3
    assert s.cache_dir == ""
    assert s.cache_enabled


def test_invalid_port_falls_back(settings_for):
    assert settings_for({"DANYAPI_PORT": "abc"}).port == 8000
    assert settings_for({"DANYAPI_PORT": "9000"}).port == 9000


def test_invalid_timeout_falls_back(settings_for):
    assert settings_for({"DANYAPI_TIMEOUT": "xyz"}).timeout == 60.0
    assert settings_for({"DANYAPI_TIMEOUT": "12.5"}).timeout == 12.5


def test_acquire_timeout(settings_for):
    assert settings_for({"DANYAPI_ACQUIRE_TIMEOUT": ""}).acquire_timeout is None
    assert settings_for({"DANYAPI_ACQUIRE_TIMEOUT": "bad"}).acquire_timeout is None
    assert settings_for({"DANYAPI_ACQUIRE_TIMEOUT": "3"}).acquire_timeout == 3.0


def test_invalid_cache_size_falls_back(settings_for):
    assert settings_for({"DANYAPI_SESSION_CACHE_SIZE": "bad"}).session_cache_size == 128
    assert settings_for({"DANYAPI_SESSION_CACHE_SIZE": "7"}).session_cache_size == 7


def test_invalid_ttl_falls_back(settings_for):
    assert settings_for({"DANYAPI_SESSION_TTL_SECONDS": "bad"}).session_ttl == 3600.0
    assert settings_for({"DANYAPI_SESSION_TTL_SECONDS": "42"}).session_ttl == 42.0


def test_log_level_empty_uses_info(settings_for):
    assert settings_for({"DANYAPI_LOG_LEVEL": ""}).log_level == "INFO"
    assert settings_for({"DANYAPI_LOG_LEVEL": "DEBUG"}).log_level == "DEBUG"


def test_invalid_log_bytes_falls_back(settings_for):
    assert settings_for({"DANYAPI_LOG_MAX_BYTES": "bad"}).log_max_bytes == 10485760
    assert settings_for({"DANYAPI_LOG_MAX_BYTES": "2048"}).log_max_bytes == 2048


def test_invalid_log_backup_falls_back(settings_for):
    assert settings_for({"DANYAPI_LOG_BACKUP_COUNT": "bad"}).log_backup_count == 3
    assert settings_for({"DANYAPI_LOG_BACKUP_COUNT": "5"}).log_backup_count == 5


def test_deepseek_tokens_comma(settings_for):
    s = settings_for({"DEEPSEEK_TOKENS": "a, b , c"})
    assert s.deepseek_tokens == ["a", "b", "c"]


def test_deepseek_single_token_fallback(settings_for):
    assert settings_for({"DEEPSEEK_TOKEN": "solo"}).deepseek_tokens == ["solo"]


def test_deepseek_tokens_win_over_single(settings_for):
    s = settings_for({"DEEPSEEK_TOKENS": "a,b", "DEEPSEEK_TOKEN": "solo"})
    assert s.deepseek_tokens == ["a", "b"]


def test_qwen_tokens_comma(settings_for):
    assert settings_for({"QWEN_TOKENS": "x , y"}).qwen_tokens == ["x", "y"]


def test_qwen_single_token_fallback(settings_for):
    assert settings_for({"QWEN_TOKEN": "only"}).qwen_tokens == ["only"]


def test_empty_tokens(settings_for):
    s = settings_for({"DEEPSEEK_TOKENS": " , , ", "QWEN_TOKENS": ""})
    assert s.deepseek_tokens == []
    assert s.qwen_tokens == []


def test_cache_dir(settings_for):
    assert settings_for({"DANYAPI_CACHE_DIR": "C:/tmp/cache"}).cache_dir == "C:/tmp/cache"


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "On"])
def test_cache_disabled_variants(settings_for, value):
    assert settings_for({"DANYAPI_CACHE_DISABLED": value}).cache_enabled is False


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "nope"])
def test_cache_enabled_variants(settings_for, value):
    assert settings_for({"DANYAPI_CACHE_DISABLED": value}).cache_enabled is True
