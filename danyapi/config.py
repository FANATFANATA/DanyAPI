"""Конфигурация DanyAPI (env-переменные)."""

from __future__ import annotations

import os


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
        self.deepseek_device_id = os.environ.get("DEEPSEEK_DEVICE_ID", "")
        # Таймауты
        self.timeout = float(os.environ.get("DANYAPI_TIMEOUT", "60"))


settings = Settings()
