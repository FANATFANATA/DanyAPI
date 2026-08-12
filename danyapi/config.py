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
        self.port = int(os.environ.get("DANYAPI_PORT", "8000"))
        tokens = [t.strip() for t in os.environ.get("DEEPSEEK_TOKENS", "").split(",") if t.strip()]
        single = os.environ.get("DEEPSEEK_TOKEN", "").strip()
        if not tokens and single:
            tokens = [single]
        self.deepseek_tokens = tokens
        self.deepseek_email = os.environ.get("DEEPSEEK_EMAIL", "")
        self.deepseek_password = os.environ.get("DEEPSEEK_PASSWORD", "")
        self.timeout = float(os.environ.get("DANYAPI_TIMEOUT", "60"))


settings = Settings()
