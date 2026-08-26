from __future__ import annotations

import os
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
ROOT = Path(__file__).resolve().parent


def ensure_env() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        return
    example = ROOT / ".env.example"
    if example.exists():
        example.copy(env_file)
        print("Created .env from .env.example")
        print("Fill in DEEPSEEK_TOKENS / QWEN_TOKENS and run again.")
        sys.exit(1)
    print("No .env or .env.example found.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if sys.version_info < MIN_PYTHON:
        print(
            f"DanyAPI requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+, got {sys.version.split()[0]}",
            file=sys.stderr,
        )
        sys.exit(1)

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    ensure_env()

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)

    import uvicorn

    from danyapi.config import settings
    from danyapi.logging import uvicorn_log_config

    print(f"DanyAPI starting on {settings.host}:{settings.port}")
    uvicorn.run(
        "danyapi.api.openai:app",
        host=settings.host,
        port=settings.port,
        log_config=uvicorn_log_config(),
    )


if __name__ == "__main__":
    main()
