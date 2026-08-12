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


settings = Settings()
