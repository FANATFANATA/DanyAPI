"""Конфигурация DanyAPI (env-переменные + автозагрузка .env)."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = lambda *a, **k: False  # type: ignore[assignment]

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)


class Settings:
    def __init__(self) -> None:
        self.host = os.environ.get("DANYAPI_HOST", "0.0.0.0")
        self.port = int(os.environ.get("DANYAPI_PORT", "8000"))
        # Учётные записи DeepSeek (пул; каждый аккаунт обрабатывает одно
        # сообщение одновременно). Токены через запятую.
        tokens = [
            t.strip()
            for t in os.environ.get("DEEPSEEK_TOKENS", "").split(",")
            if t.strip()
        ]
        single = os.environ.get("DEEPSEEK_TOKEN", "").strip()
        if not tokens and single:
            tokens = [single]
        self.deepseek_tokens = tokens
        # Либо одна учётка email+пароль (логин при старте).
        self.deepseek_email = os.environ.get("DEEPSEEK_EMAIL", "")
        self.deepseek_password = os.environ.get("DEEPSEEK_PASSWORD", "")
        # Таймауты
        self.timeout = float(os.environ.get("DANYAPI_TIMEOUT", "60"))


settings = Settings()
