# DanyAPI

OpenAI compatible API built on Python + FastAPI. Instead of the paid APIs it talks to the internal APIs of the free web clients using server side accounts created from your own free provider tokens (DEEPSEEK_TOKENS / QWEN_TOKENS). API consumers need no keys, all upstream requests are made by the configured server tokens.

[![CI](https://img.shields.io/github/actions/workflow/status/FANATFANATA/DanyAPI/ci.yml?branch=main)](https://github.com/FANATFANATA/DanyAPI/actions)
[![GitHub Release](https://img.shields.io/github/v/release/FANATFANATA/DanyAPI?sort=semver)](https://github.com/FANATFANATA/DanyAPI/releases)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/GHCR-ghcr.io%2Ffanatfanata%2Fdanyapi-blue)](https://github.com/FANATFANATA/DanyAPI/pkgs/container/danyapi)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/FANATFANATA/DanyAPI/blob/main/LICENSE)

## Public hosted (full free)

- API baseURL:

```text
https://danyapi.cloudpub.ru/v1/
```

- Landing:

```text
https://danyapi.cloudpub.ru/
```

## Install & Upgrade

Requires Python 3.10+

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

For multi user BYOK mode:

```bash
docker run -d -p 8000:8000 \
  -e DEEPSEEK_TOKENS="" \
  -e QWEN_TOKENS="" \
  -e BYOK_MODE=1 \
  ghcr.io/fanatfanata/danyapi:latest
```

## BYOK (Bring Your Own Key)

By default DanyAPI runs in single mode using server tokens (DEEPSEEK_TOKENS, QWEN_TOKENS). Set BYOK_MODE=true in .env to enable per user token management: users register an account, authenticate via cookie, and add their own provider tokens. Tokens are stored encrypted on disk per user and used exclusively when that user makes requests.

Enable in .env:

```env
BYOK_MODE=1
BYOK_SALT=
```

## Contacts

[Creator](https://t.me/DanyaVoredom) · [Telegram channel](https://t.me/DanyAPIFree) · [Website](https://fanatfanata.github.io/DanyAPI/)
