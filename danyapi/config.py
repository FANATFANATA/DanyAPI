from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Any

_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off")

log = logging.getLogger("danyapi.config")

MIN_PORT = 1
MAX_PORT = 65535
MAX_CHOICES = 8
MAX_ALICE_ACCOUNTS = 4
MAX_DUCKAI_ACCOUNTS = 4

CREDENTIAL_ENV_NAMES = (
    "DEEPSEEK_TOKENS",
    "QWEN_TOKENS",
    "GIGACHAT_KEYS",
    "GIGACHAT_SCOPE",
    "DANYAPI_GIGACHAT_SCOPE",
    "OPENCODE_KEYS",
    "BYOK",
    "BYOK_MODE",
    "DANYAPI_BYOK_MODE",
    "DANYAPI_ADMIN_TOKEN",
)
_NON_CREDENTIAL_ENV_NAMES = frozenset(
    {
        "DANYAPI_HOST",
        "DANYAPI_PORT",
        "ALICE_ENABLED",
        "ALICE_ACCOUNTS",
        "OPENCODE_ENABLED",
        "DUCKAI_ENABLED",
        "DUCKAI_ACCOUNTS",
        "DANYAPI_TIMEOUT",
        "DANYAPI_ACQUIRE_TIMEOUT",
        "DANYAPI_SESSION_CACHE_SIZE",
        "DANYAPI_SESSION_TTL_SECONDS",
        "DANYAPI_LOG_LEVEL",
        "DANYAPI_LOG_FILE",
        "DANYAPI_LOG_MAX_BYTES",
        "DANYAPI_LOG_BACKUP_COUNT",
        "DANYAPI_CACHE_DIR",
        "DANYAPI_CACHE_DISABLED",
        "DANYAPI_BYOK_AUTH_TTL_SECONDS",
        "DANYAPI_MODELS_REFRESH_SECONDS",
        "DANYAPI_USAGE_ENABLED",
        "DANYAPI_USAGE_MAX_RECORDS",
        "DANYAPI_AUTO_UPDATE",
        "DANYAPI_CORS_ORIGINS",
        "DANYAPI_RESPONSES_MAX_RECORDS",
    }
)
_ENV_NAME_RE = re.compile(r"_env_(?:int|float|positive_float|float_opt|str|list|on|off|first)\(\s*\"([A-Za-z0-9_]+)\"")


def audit_credential_env_names() -> None:
    try:
        source = Path(__file__).resolve().read_text(encoding="utf-8")
    except OSError as exc:
        logging.getLogger(__name__).warning("cannot audit credential env names: %s", exc)
        return
    read = set(_ENV_NAME_RE.findall(source))
    missing = sorted(read - set(CREDENTIAL_ENV_NAMES) - _NON_CREDENTIAL_ENV_NAMES)
    if missing:
        logging.getLogger(__name__).warning(
            "config reads env names that are neither credential nor listed as non credential: %s",
            ", ".join(missing),
        )


_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _noop_load_dotenv(*args: Any, **kwargs: Any) -> bool:
    logging.getLogger(__name__).warning("python-dotenv is not installed, skipping %s", _ENV_PATH)
    return False


try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = _noop_load_dotenv

load_dotenv(dotenv_path=_ENV_PATH, override=False)


def _env_int(key: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        value = int(os.environ.get(key, default))
    except (TypeError, ValueError):
        log.warning("%s=%r is not a valid integer, using %r", key, os.environ.get(key), default)
        return default
    if minimum is not None and value < minimum:
        log.warning("%s=%r is below the minimum %r, clamped", key, value, minimum)
        return minimum
    if maximum is not None and value > maximum:
        log.warning("%s=%r is above the maximum %r, clamped", key, value, maximum)
        return maximum
    return value


def _env_float(key: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.environ.get(key, default))
    except (TypeError, ValueError):
        log.warning("%s=%r is not a valid number, using %r", key, os.environ.get(key), default)
        return default
    if not math.isfinite(value):
        log.warning("%s=%r is not finite, using %r", key, value, default)
        return default
    if value < minimum:
        log.warning("%s=%r is below the minimum %r, clamped", key, value, minimum)
        return minimum
    return value


def _env_positive_float(key: str, default: float) -> float:
    try:
        value = float(os.environ.get(key, default))
    except (TypeError, ValueError):
        log.warning("%s=%r is not a valid number, using %r", key, os.environ.get(key), default)
        return default
    if not math.isfinite(value) or value <= 0:
        log.warning("%s=%r must be finite and positive, using %r", key, value, default)
        return default
    return value


def _env_float_opt(key: str) -> float | None:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log.warning("%s=%r is not a valid number, ignoring it", key, raw)
        return None
    if not math.isfinite(value) or value <= 0:
        log.warning("%s=%r must be finite and positive, ignoring it", key, raw)
        return None
    return value


def _env_str(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _split_env_list(raw: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    escaped = False
    for char in raw:
        if escaped:
            escaped = False
            if char == ",":
                current.append(char)
                continue
            current.append("\\")
            current.append(char)
            continue
        if char == "\\":
            escaped = True
            continue
        if char == ",":
            items.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if escaped:
        current.append("\\")
    items.append("".join(current).strip())
    return [item for item in items if item]


def _env_list(key: str) -> list[str]:
    return _split_env_list(os.environ.get(key, ""))


def _env_on(key: str, default: str) -> bool:
    return os.environ.get(key, default).strip().lower() in _TRUE_VALUES


def _env_off(key: str, default: str) -> bool:
    return os.environ.get(key, default).strip().lower() in _FALSE_VALUES


def _env_first(*keys: str) -> str:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return ""


class Settings:
    def __init__(self) -> None:
        self.host = _env_str("DANYAPI_HOST", "0.0.0.0")
        self.port = _env_int("DANYAPI_PORT", 8000, MIN_PORT, MAX_PORT)
        self.deepseek_tokens = _env_list("DEEPSEEK_TOKENS")
        self.qwen_tokens = _env_list("QWEN_TOKENS")
        self.gigachat_keys = _env_list("GIGACHAT_KEYS")
        self.gigachat_scope = _env_first("GIGACHAT_SCOPE", "DANYAPI_GIGACHAT_SCOPE").strip() or "GIGACHAT_API_PERS"
        self.opencode_keys = _env_list("OPENCODE_KEYS")
        self.opencode_enabled = _env_on("OPENCODE_ENABLED", "")
        self.alice_enabled = _env_on("ALICE_ENABLED", "")
        self.alice_accounts = _env_int("ALICE_ACCOUNTS", 1, 1, MAX_ALICE_ACCOUNTS)
        self.duckai_enabled = _env_on("DUCKAI_ENABLED", "")
        self.duckai_accounts = _env_int("DUCKAI_ACCOUNTS", 1, 1, MAX_DUCKAI_ACCOUNTS)
        self.byok = _env_first("BYOK", "BYOK_MODE", "DANYAPI_BYOK_MODE").strip().lower() in _TRUE_VALUES
        self.timeout = _env_positive_float("DANYAPI_TIMEOUT", 60.0)
        self.acquire_timeout = _env_float_opt("DANYAPI_ACQUIRE_TIMEOUT")
        self.session_cache_size = _env_int("DANYAPI_SESSION_CACHE_SIZE", 128, 1)
        self.session_ttl = _env_float("DANYAPI_SESSION_TTL_SECONDS", 3600.0, minimum=1.0)
        self.log_level = _env_str("DANYAPI_LOG_LEVEL", "INFO") or "INFO"
        self.log_file = _env_str("DANYAPI_LOG_FILE")
        self.log_max_bytes = _env_int("DANYAPI_LOG_MAX_BYTES", 10 * 1024 * 1024, 1)
        self.log_backup_count = _env_int("DANYAPI_LOG_BACKUP_COUNT", 3, 0)
        self.cache_dir = _env_str("DANYAPI_CACHE_DIR")
        self.cache_enabled = not _env_on("DANYAPI_CACHE_DISABLED", "")
        self.byok_auth_ttl = _env_float("DANYAPI_BYOK_AUTH_TTL_SECONDS", 300.0)
        self.models_refresh_seconds = _env_float("DANYAPI_MODELS_REFRESH_SECONDS", 900.0)
        self.usage_enabled = not _env_off("DANYAPI_USAGE_ENABLED", "1")
        self.usage_max_records = _env_int("DANYAPI_USAGE_MAX_RECORDS", 1000, 1)
        self.auto_update = not _env_off("DANYAPI_AUTO_UPDATE", "1")
        self.cors_origins = _env_list("DANYAPI_CORS_ORIGINS")
        self.responses_max_records = _env_int("DANYAPI_RESPONSES_MAX_RECORDS", 1024, 1)
        self.admin_token = _env_str("DANYAPI_ADMIN_TOKEN")


settings = Settings()
audit_credential_env_names()
