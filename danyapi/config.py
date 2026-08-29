from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _noop_load_dotenv(*args: Any, **kwargs: Any) -> bool:
    return False


try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = _noop_load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        return default


def _env_float_opt(key: str) -> float | None:
    raw = os.environ.get(key, "")
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def _env_str(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


class Settings:
    def __init__(self) -> None:
        self.host = os.environ.get("DANYAPI_HOST", "0.0.0.0")
        self.port = _env_int("DANYAPI_PORT", 8000)
        tokens = [t.strip() for t in os.environ.get("DEEPSEEK_TOKENS", "").split(",") if t.strip()]
        self.deepseek_tokens = tokens
        qwen_tokens = [t.strip() for t in os.environ.get("QWEN_TOKENS", "").split(",") if t.strip()]
        self.qwen_tokens = qwen_tokens
        self.timeout = _env_float("DANYAPI_TIMEOUT", 60.0)
        self.acquire_timeout = _env_float_opt("DANYAPI_ACQUIRE_TIMEOUT")
        self.session_cache_size = _env_int("DANYAPI_SESSION_CACHE_SIZE", 128)
        self.session_ttl = _env_float("DANYAPI_SESSION_TTL_SECONDS", 3600.0)
        self.log_level = _env_str("DANYAPI_LOG_LEVEL", "INFO") or "INFO"
        self.log_file = _env_str("DANYAPI_LOG_FILE")
        self.log_max_bytes = _env_int("DANYAPI_LOG_MAX_BYTES", 10 * 1024 * 1024)
        self.log_backup_count = _env_int("DANYAPI_LOG_BACKUP_COUNT", 3)
        self.cache_dir = _env_str("DANYAPI_CACHE_DIR")
        self.cache_enabled = os.environ.get("DANYAPI_CACHE_DISABLED", "").strip().lower() not in ("1", "true", "yes", "on")
        self.human_delay_min = _env_float("DANYAPI_HUMAN_DELAY_MIN", 0.5)
        self.human_delay_max = _env_float("DANYAPI_HUMAN_DELAY_MAX", 3.0)
        self.usage_enabled = os.environ.get("DANYAPI_USAGE_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
        self.usage_max_records = _env_int("DANYAPI_USAGE_MAX_RECORDS", 1000)


settings = Settings()
