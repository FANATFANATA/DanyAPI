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

## Contacts

[Creator](https://t.me/DanyaVoredom) · [Telegram channel](https://t.me/DanyAPIFree) · [Website](https://fanatfanata.github.io/DanyAPI/)
