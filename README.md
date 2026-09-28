# DanyAPI

OpenAI compatible HTTP API built on Python + FastAPI. Instead of the paid APIs it talks to the internal APIs of the free web clients, and to the free tier of GigaChat.

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

### Endpoints

- `POST /v1/chat/completions` and `POST /v1/completions`: OpenAI compatible
- `POST /v1/responses`: OpenAI Responses API
- `POST /v1/messages` and `POST /v1/messages/count_tokens`: Anthropic Messages API
- `GET /v1/models`, `GET /health`, `GET /v1/usage`

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
  -e GIGACHAT_KEYS="<authorization_key>" \
  -e ALICE_ENABLED=1 \
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
| `DEEPSEEK_TOKENS` | empty | Comma-separated DeepSeek web tokens. Required unless another provider is used |
| `QWEN_TOKENS` | empty | Comma-separated Qwen web tokens. Required unless another provider is used |
| `GIGACHAT_KEYS` | empty | Comma-separated GigaChat authorization keys, base64 of `client_id:client_secret` from the GigaChat Studio account |
| `GIGACHAT_SCOPE` | `GIGACHAT_API_PERS` | GigaChat scope: `GIGACHAT_API_PERS`, `GIGACHAT_API_B2B` or `GIGACHAT_API_CORP` |
| `DANYAPI_GIGACHAT_CA_FILE` | empty | Path to a CA bundle for GigaChat, empty uses the bundled Russian root CA |
| `ALICE_ENABLED` | empty | `1` enables the unofficial Yandex Alice provider, see the warning below |
| `ALICE_ACCOUNTS` | `1` | Concurrent Alice connections, `1` to `4` |
| `BYOK` / `BYOK_MODE` / `DANYAPI_BYOK_MODE` | empty | `1` runs in bring-your-own-key mode: every request supplies its own provider token instead of using the pools above |
| `DANYAPI_ADMIN_TOKEN` | empty | Bearer token required by `POST /v1/tokens`, empty keeps that endpoint disabled |

Server:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DANYAPI_HOST` | `0.0.0.0` | Address the server binds to |
| `DANYAPI_PORT` | `8000` | Port the server listens on |
| `DANYAPI_TIMEOUT` | `60` | Upstream request timeout in seconds |
| `DANYAPI_ACQUIRE_TIMEOUT` | empty | Seconds to wait for a free account, empty means wait forever |
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

## GigaChat

`GigaChat` and `GigaChat-2` (Lite), `GigaChat-Pro` and `GigaChat-2-Pro`, `GigaChat-Max` and `GigaChat-2-Max` are served from the official GigaChat API at `https://api.giga.chat/v1`, which speaks the OpenAI chat completions format. Every new GigaChat project starts with a free freemium quota.

Get the authorization key in the GigaChat Studio account under "Настройки API". It is the base64 of `client_id:client_secret`, not the secret on its own:

```bash
GIGACHAT_KEYS="<authorization_key>"
```

Access tokens live 30 minutes and are refreshed automatically. GigaChat issues its certificates under the Russian Trusted Root CA, which is absent from most Python CA bundles, so a root CA is shipped in `danyapi/gigachat/russian_trusted_root_ca.pem` and combined with the system roots at runtime. Point `DANYAPI_GIGACHAT_CA_FILE` at your own bundle to override it.

Differences from the OpenAI API to keep in mind: function calling uses the legacy `functions` plus `function_call` pair, which DanyAPI maps from `tools` for you, there is no `n`, `seed`, `stop` or penalty support, and images are uploaded to the GigaChat file storage first, one image per message and ten per request.

## Yandex Alice, unofficial

`alice`, `alice-ai` and `yagpt` route to the Yandex Alice consumer endpoint `wss://uniproxy.alice.yandex.net/uni.ws`. It needs no key, no account and no token.

Read this before enabling it. Yandex has no public API for this, so the provider speaks an undocumented internal protocol of a consumer service, and `yandex.ru/legal/alice_chat` clause 4.2 forbids circumventing technical protections and imitating the service's functioning. Yandex also versions and reshapes this protocol without notice, so the provider can break at any time. It is disabled unless you set `ALICE_ENABLED=1`, which is your acknowledgement of the above.

Behaviour to expect from the answers: the endpoint is stateless, so the whole conversation is folded into one prompt; it routes prompts across scenario handlers rather than to a single model, so many prompts get a canned Alice reply or a persona refusal instead of an answer; and there is no incremental text, so streaming sends the finished text in a single chunk.

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
