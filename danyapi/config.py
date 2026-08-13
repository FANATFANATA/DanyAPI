from __future__ import annotations

import os
from pathlib import Path


def _noop_load_dotenv(*args, **kwargs) -> bool:
    return False


try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = _noop_load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)


class Settings:
    def __init__(self) -> None:
        self.host = os.environ.get("DANYAPI_HOST", "0.0.0.0")
        try:
            self.port = int(os.environ.get("DANYAPI_PORT", "8000"))
        except ValueError:
            self.port = 8000
        tokens = [t.strip() for t in os.environ.get("DEEPSEEK_TOKENS", "").split(",") if t.strip()]
        single = os.environ.get("DEEPSEEK_TOKEN", "").strip()
        if not tokens and single:
            tokens = [single]
        self.deepseek_tokens = tokens
        self.deepseek_email = os.environ.get("DEEPSEEK_EMAIL", "")
        self.deepseek_password = os.environ.get("DEEPSEEK_PASSWORD", "")
        qwen_tokens = [t.strip() for t in os.environ.get("QWEN_TOKENS", "").split(",") if t.strip()]
        qwen_single = os.environ.get("QWEN_TOKEN", "").strip()
        if not qwen_tokens and qwen_single:
            qwen_tokens = [qwen_single]
        self.qwen_tokens = qwen_tokens
        self.qwen_email = os.environ.get("QWEN_EMAIL", "")
        self.qwen_password = os.environ.get("QWEN_PASSWORD", "")
        try:
            self.timeout = float(os.environ.get("DANYAPI_TIMEOUT", "60"))
        except ValueError:
            self.timeout = 60.0
        raw_acquire = os.environ.get("DANYAPI_ACQUIRE_TIMEOUT", "")
        try:
            self.acquire_timeout = float(raw_acquire) if raw_acquire else None
        except ValueError:
            self.acquire_timeout = None
        raw_cache = os.environ.get("DANYAPI_SESSION_CACHE_SIZE", "128")
        try:
            self.session_cache_size = int(raw_cache)
        except ValueError:
            self.session_cache_size = 128
        raw_ttl = os.environ.get("DANYAPI_SESSION_TTL_SECONDS", "3600")
        try:
            self.session_ttl = float(raw_ttl)
        except ValueError:
            self.session_ttl = 3600.0
        raw_level = os.environ.get("DANYAPI_LOG_LEVEL", "INFO").strip()
        self.log_level = raw_level or "INFO"
        self.log_file = os.environ.get("DANYAPI_LOG_FILE", "").strip()
        try:
            self.log_max_bytes = int(os.environ.get("DANYAPI_LOG_MAX_BYTES", "10485760"))
        except ValueError:
            self.log_max_bytes = 10 * 1024 * 1024
        try:
            self.log_backup_count = int(os.environ.get("DANYAPI_LOG_BACKUP_COUNT", "3"))
        except ValueError:
            self.log_backup_count = 3
        self.cache_dir = os.environ.get("DANYAPI_CACHE_DIR", "").strip()
        self.cache_enabled = os.environ.get("DANYAPI_CACHE_DISABLED", "").strip().lower() not in ("1", "true", "yes", "on")


settings = Settings()
