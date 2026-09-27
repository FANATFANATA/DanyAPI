# DanyAPI

OpenAI compatible HTTP API built on Python + FastAPI. Instead of the paid APIs it talks to the internal APIs of the free web clients.

[![CI](https://img.shields.io/github/actions/workflow/status/FANATFANATA/DanyAPI/ci.yml?branch=prod)](https://github.com/FANATFANATA/DanyAPI/actions)
[![GitHub Release](https://img.shields.io/github/v/release/FANATFANATA/DanyAPI?sort=semver)](https://github.com/FANATFANATA/DanyAPI/releases)
[![Python](https://img.shields.io/badge/Python-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/GHCR-ghcr.io%2Ffanatfanata%2Fdanyapi-blue)](https://github.com/FANATFANATA/DanyAPI/pkgs/container/danyapi)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/FANATFANATA/DanyAPI/blob/prod/LICENSE)

## Public hosted instance

A public instance is already running in production (BYOK_MODE=1):

- API base URL: `https://danyapi.cloudpub.ru/v1/`

Point any OpenAI compatible client at API base URL with a valid tokens, unauthenticated requests are rejected with 401. The API key should be the raw token (e.g. "token1,token2", same in .env).

### Example request

```bash
curl -X POST https://danyapi.cloudpub.ru/v1/chat/completions \
  -H "Authorization: Bearer token1,token2,token3" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4.1-flash-thinking",
    "messages": [
      {"role": "user", "content": "Hi from example request!"}
    ]
  }'
```

## Install & Upgrade

Requires Python 3.10+.

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/FANATFANATA/DanyAPI/prod/docs/install.ps1 | iex
```

Linux/macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/FANATFANATA/DanyAPI/prod/docs/install.sh | bash
```

Docker:

```bash
docker run -d -p 8000:8000 \
  -e DEEPSEEK_TOKENS="token1,token2" \
  -e QWEN_TOKENS="token3" \
  ghcr.io/fanatfanata/danyapi:latest
```

## Run locally

Clone or download the repo, install the dependencies, then start the server from the DanyAPI folder:

```bash
python app.py
```

Two equivalent entry points, both reading the same `.env`:

```bash
python -m danyapi
python docs/start.py
```

`app.py` and `python -m danyapi` start the server as is. `docs/start.py` first pulls the latest GitHub release (see `DANYAPI_AUTO_UPDATE`), then starts it, and it is what the desktop shortcut runs.

Defaults: binds `0.0.0.0:8000`, so the API is at `http://127.0.0.1:8000/v1/`, the landing page at `http://127.0.0.1:8000/` and the health check at `http://127.0.0.1:8000/health`.

## Configuration

All settings live in `.env` at the repo root. `docs/setup.py` writes it for you, and `.env.example` lists every key with its default. `.env` is git-ignored and holds your tokens, so keep it out of version control.

Credentials:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DEEPSEEK_TOKENS` | empty | Comma-separated DeepSeek web tokens. Required unless only Qwen is used |
| `QWEN_TOKENS` | empty | Comma-separated Qwen web tokens. Required unless only DeepSeek is used |
| `BYOK` / `BYOK_MODE` / `DANYAPI_BYOK_MODE` | empty | `1` runs in bring-your-own-key mode: every request supplies its own provider token instead of using the pools above |

Server:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DANYAPI_HOST` | `0.0.0.0` | Address the server binds to |
| `DANYAPI_PORT` | `8000` | Port the server listens on |
| `DANYAPI_TIMEOUT` | `60` | Upstream request timeout in seconds |
| `DANYAPI_ACQUIRE_TIMEOUT` | empty | Seconds to wait for a free account, empty means wait forever |
| `DANYAPI_HUMAN_DELAY_MIN` | `0.5` | Minimum delay in seconds before a request is sent |
| `DANYAPI_HUMAN_DELAY_MAX` | `3.0` | Maximum delay in seconds |
| `DANYAPI_CORS_ORIGINS` | empty | Comma-separated extra browser origins allowed to call the API |
| `DANYAPI_AUTO_UPDATE` | `1` | `docs/start.py` updates to the latest GitHub release before starting |

Sessions and cache:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DANYAPI_SESSION_CACHE_SIZE` | `128` | Chats cached per provider |
| `DANYAPI_SESSION_TTL_SECONDS` | `3600` | How long an unused session stays reusable, `0` never expires |
| `DANYAPI_CACHE_DIR` | empty | On-disk cache directory, empty means `$TMPDIR/danyapi` |
| `DANYAPI_CACHE_DISABLED` | empty | `1` keeps everything in memory only |
| `DANYAPI_RESPONSES_MAX_RECORDS` | `1024` | Recent `/v1/responses` records kept |

Usage and logging:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DANYAPI_USAGE_ENABLED` | `1` | `0` disables the usage counters behind `GET /v1/usage` |
| `DANYAPI_USAGE_MAX_RECORDS` | `1000` | Recent usage records kept |
| `DANYAPI_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |
| `DANYAPI_LOG_FILE` | empty | Log file path, empty logs to the console only |
| `DANYAPI_LOG_MAX_BYTES` | `10485760` | Size at which the log file rotates |
| `DANYAPI_LOG_BACKUP_COUNT` | `3` | Rotated log files kept |
| `DANYAPI_BYOK_AUTH_TTL_SECONDS` | `300` | How long a BYOK token check is reused |

## Token utility

The installer asks for provider tokens by hand, but `docs/token_utility.py` reads them out of your browser for you. It starts a small local server on `127.0.0.1:8765`, walks you through DeepSeek and then Qwen, and prints both tokens to copy into `.env`. It needs no dependencies, and nothing leaves your machine.

Linux/macOS:

```bash
sh docs/token_utility.sh
```

Windows:

```
docs\token_utility.bat
```

Or run it directly with any Python 3.10+ interpreter:

```bash
python docs/token_utility.py
```

Use `--port` to move it off the default port and `--no-browser` to skip opening the page. The server binds to `127.0.0.1` only and stops when you close it.

## Contacts

[Creator](https://t.me/DanyaVoredom) · [Telegram channel](https://t.me/DanyAPIFree) · [Website](https://fanatfanata.github.io/DanyAPI/)
