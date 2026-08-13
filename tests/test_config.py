import unittest
from unittest.mock import patch

from danyapi.config import Settings


def _settings(env):
    with patch.dict("os.environ", env, clear=True):
        return Settings()


class TestSettingsDefaults(unittest.TestCase):
    def test_defaults(self):
        s = _settings({})
        self.assertEqual(s.host, "0.0.0.0")
        self.assertEqual(s.port, 8000)
        self.assertEqual(s.deepseek_tokens, [])
        self.assertEqual(s.qwen_tokens, [])
        self.assertEqual(s.deepseek_email, "")
        self.assertEqual(s.deepseek_password, "")
        self.assertEqual(s.qwen_email, "")
        self.assertEqual(s.qwen_password, "")
        self.assertEqual(s.timeout, 60.0)
        self.assertIsNone(s.acquire_timeout)
        self.assertEqual(s.session_cache_size, 128)
        self.assertEqual(s.session_ttl, 3600.0)
        self.assertEqual(s.log_level, "INFO")
        self.assertEqual(s.log_file, "")
        self.assertEqual(s.log_max_bytes, 10485760)
        self.assertEqual(s.log_backup_count, 3)
        self.assertEqual(s.cache_dir, "")
        self.assertTrue(s.cache_enabled)

    def test_invalid_port_falls_back(self):
        self.assertEqual(_settings({"DANYAPI_PORT": "abc"}).port, 8000)
        self.assertEqual(_settings({"DANYAPI_PORT": "9000"}).port, 9000)

    def test_invalid_timeout_falls_back(self):
        self.assertEqual(_settings({"DANYAPI_TIMEOUT": "xyz"}).timeout, 60.0)
        self.assertEqual(_settings({"DANYAPI_TIMEOUT": "12.5"}).timeout, 12.5)

    def test_acquire_timeout(self):
        self.assertIsNone(_settings({"DANYAPI_ACQUIRE_TIMEOUT": ""}).acquire_timeout)
        self.assertIsNone(_settings({"DANYAPI_ACQUIRE_TIMEOUT": "bad"}).acquire_timeout)
        self.assertEqual(_settings({"DANYAPI_ACQUIRE_TIMEOUT": "3"}).acquire_timeout, 3.0)

    def test_invalid_cache_size_falls_back(self):
        self.assertEqual(_settings({"DANYAPI_SESSION_CACHE_SIZE": "bad"}).session_cache_size, 128)
        self.assertEqual(_settings({"DANYAPI_SESSION_CACHE_SIZE": "7"}).session_cache_size, 7)

    def test_invalid_ttl_falls_back(self):
        self.assertEqual(_settings({"DANYAPI_SESSION_TTL_SECONDS": "bad"}).session_ttl, 3600.0)
        self.assertEqual(_settings({"DANYAPI_SESSION_TTL_SECONDS": "42"}).session_ttl, 42.0)

    def test_log_level_empty_uses_info(self):
        self.assertEqual(_settings({"DANYAPI_LOG_LEVEL": ""}).log_level, "INFO")
        self.assertEqual(_settings({"DANYAPI_LOG_LEVEL": "DEBUG"}).log_level, "DEBUG")

    def test_invalid_log_bytes_falls_back(self):
        self.assertEqual(_settings({"DANYAPI_LOG_MAX_BYTES": "bad"}).log_max_bytes, 10485760)
        self.assertEqual(_settings({"DANYAPI_LOG_MAX_BYTES": "2048"}).log_max_bytes, 2048)

    def test_invalid_log_backup_falls_back(self):
        self.assertEqual(_settings({"DANYAPI_LOG_BACKUP_COUNT": "bad"}).log_backup_count, 3)
        self.assertEqual(_settings({"DANYAPI_LOG_BACKUP_COUNT": "5"}).log_backup_count, 5)


class TestSettingsTokens(unittest.TestCase):
    def test_deepseek_tokens_comma(self):
        s = _settings({"DEEPSEEK_TOKENS": "a, b , c"})
        self.assertEqual(s.deepseek_tokens, ["a", "b", "c"])

    def test_deepseek_single_token_fallback(self):
        s = _settings({"DEEPSEEK_TOKEN": "solo"})
        self.assertEqual(s.deepseek_tokens, ["solo"])

    def test_deepseek_tokens_win_over_single(self):
        s = _settings({"DEEPSEEK_TOKENS": "a,b", "DEEPSEEK_TOKEN": "solo"})
        self.assertEqual(s.deepseek_tokens, ["a", "b"])

    def test_qwen_tokens_comma(self):
        s = _settings({"QWEN_TOKENS": "x , y"})
        self.assertEqual(s.qwen_tokens, ["x", "y"])

    def test_qwen_single_token_fallback(self):
        s = _settings({"QWEN_TOKEN": "only"})
        self.assertEqual(s.qwen_tokens, ["only"])

    def test_empty_tokens(self):
        s = _settings({"DEEPSEEK_TOKENS": " , , ", "QWEN_TOKENS": ""})
        self.assertEqual(s.deepseek_tokens, [])
        self.assertEqual(s.qwen_tokens, [])


class TestSettingsCache(unittest.TestCase):
    def test_cache_dir(self):
        self.assertEqual(_settings({"DANYAPI_CACHE_DIR": "C:/tmp/cache"}).cache_dir, "C:/tmp/cache")

    def test_cache_disabled_variants(self):
        for value in ("1", "true", "yes", "on", "TRUE", "On"):
            with self.subTest(value=value):
                self.assertFalse(_settings({"DANYAPI_CACHE_DISABLED": value}).cache_enabled)

    def test_cache_enabled_variants(self):
        for value in ("", "0", "false", "no", "off", "nope"):
            with self.subTest(value=value):
                self.assertTrue(_settings({"DANYAPI_CACHE_DISABLED": value}).cache_enabled)


if __name__ == "__main__":
    unittest.main()
