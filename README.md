# DanyAPI

OpenAI-compatible HTTP API built on Python + FastAPI. Instead of the paid APIs it talks to the internal APIs of the free web clients [chat.deepseek.com](https://chat.deepseek.com) and [chat.qwen.ai](https://chat.qwen.ai) using server-side accounts created from your own free provider tokens (`DEEPSEEK_TOKENS` / `QWEN_TOKENS`). API consumers need no keys - all upstream requests are made by the configured server tokens.

For multi-user deployments, enable [BYOK mode](#byok-bring-your-own-key) so each user can register their own tokens.

[![CI](https://img.shields.io/github/actions/workflow/status/FANATFANATA/DanyAPI/ci.yml?branch=main)](https://github.com/FANATFANATA/DanyAPI/actions)
[![GitHub Release](https://img.shields.io/github/v/release/FANATFANATA/DanyAPI?sort=semver)](https://github.com/FANATFANATA/DanyAPI/releases)
[![Python](https://img.shields.io/badge/Python-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/GHCR-ghcr.io%2Ffanatfanata%2Fdanyapi-blue)](https://github.com/FANATFANATA/DanyAPI/pkgs/container/danyapi)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/FANATFANATA/DanyAPI/blob/main/LICENSE)

## Public hosted instance (full free)

Don't want to self-host? A public, fully free instance is already running in production - no signup, no keys, no limits on your side:

- API base URL: `https://danyapi.cloudpub.ru/v1/`
- Landing page: `https://danyapi.cloudpub.ru/`

Point any OpenAI-compatible client at `https://danyapi.cloudpub.ru/v1/` with a dummy `api_key` and it just works. The instance is backed by the same free provider tokens described below; treat it as best-effort.

## Install & Upgrade

Requires Python 3.10+ (CI tests 3.10-3.14).

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/FANATFANATA/DanyAPI/main/docs/install.ps1 | iex
```

Linux/macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/FANATFANATA/DanyAPI/main/docs/install.sh | bash
```

Docker:

```bash
docker run -d -p 8000:8000 \
  -e DEEPSEEK_TOKENS="token1,token2" \
  -e QWEN_TOKENS="token3" \
  ghcr.io/fanatfanata/danyapi:latest
```

For multi-user BYOK mode:

```bash
docker run -d -p 8000:8000 \
  -e DEEPSEEK_TOKENS="" \
  -e QWEN_TOKENS="" \
  -e BYOK_MODE=1 \
  ghcr.io/fanatfanata/danyapi:latest
```

## BYOK (Bring Your Own Key)

By default DanyAPI runs in single-mode using server tokens (`DEEPSEEK_TOKENS`, `QWEN_TOKENS`). Set `BYOK_MODE=true` in `.env` to enable per-user token management: users register an account, authenticate via cookie, and add their own provider tokens. Tokens are stored encrypted on disk per user and used exclusively when that user makes requests.

Enable in `.env`:

```env
BYOK_MODE=1
BYOK_SALT=       # auto-generated if empty — keep persistent across restarts
```

API endpoints (available only when `BYOK_MODE=true`):

| Method | Path | Description |
|--------|------|-------------|
| POST | `/byok/register` | Create account `{username, password}` |
| POST | `/byok/login` | Authenticate, sets `byok_session` cookie |
| GET | `/byok/logout` | Invalidate session, clear cookie |
| GET | `/byok/status` | `{"enabled": true/false}` |
| GET | `/byok/me` | Authenticated user info |
| POST | `/byok/token` | Add/replace provider token `{provider, token}` |
| GET | `/byok/tokens` | List providers where user has stored tokens |
| DELETE | `/byok/token/{provider}` | Remove token for given provider |

Tokens are stored in `$DANYAPI_CACHE_DIR/byok/tokens_{user_id}.json`. Each request resolves the user's token from their session cookie and uses it instead of the server pool.

## Contacts

[Creator](https://t.me/DanyaVoredom) · [Telegram channel](https://t.me/DanyAPIFree) · [Website](https://fanatfanata.github.io/DanyAPI/)
