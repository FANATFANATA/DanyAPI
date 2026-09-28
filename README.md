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

Point any OpenAI compatible client at API base URL with a valid tokens, unauthenticated requests are rejected with 401. The API key should be the raw token (e.g. "token1,token2", same in .env). The `alice`, `alice-ai`, `yagpt` and Duck.ai models need no key and are reachable without one; DeepSeek, Qwen and GigaChat models return 401 without a key. `GET /health` reports every provider as enabled in this mode, with `byok_api_key_required` naming the ones that need a key.

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
  -e DUCKAI_ENABLED=1 \
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
| `DUCKAI_ENABLED` | empty | `1` enables the unofficial Duck.ai provider, see the warning below |
| `DUCKAI_ACCOUNTS` | `1` | Concurrent Duck.ai connections, `1` to `4` |
| `BYOK` / `BYOK_MODE` / `DANYAPI_BYOK_MODE` | empty | `1` runs in bring-your-own-key mode: DeepSeek, Qwen and GigaChat requests supply their own key, Alice and Duck.ai need none. `GET /health` reports every provider as enabled and reports the per-key pools |
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
| `DANYAPI_MODELS_REFRESH_SECONDS` | `900` | How often provider model lists are refetched, `0` disables the background refresh |

## GigaChat

`GigaChat-2` (Lite), `GigaChat-2-Pro`, `GigaChat-2-Max`, `GigaChat-3-Lightning`, `GigaChat-3-Pro` and `GigaChat-3-Ultra` are served from the official GigaChat API at `https://api.giga.chat/v1`, which speaks the OpenAI chat completions format. The list is read from `GET /models` at startup, so it follows whatever your account is granted. Every new GigaChat project starts with a free freemium quota.

Get the authorization key in the GigaChat Studio account under "Настройки API". It is the base64 of `client_id:client_secret`, not the secret on its own:

```bash
GIGACHAT_KEYS="<authorization_key>"
```

Access tokens live 30 minutes and are refreshed automatically. GigaChat issues its certificates under the Russian Trusted Root CA, which is absent from most Python CA bundles, so a root CA is shipped in `danyapi/gigachat/russian_trusted_root_ca.pem` and combined with the system roots at runtime. Point `DANYAPI_GIGACHAT_CA_FILE` at your own bundle to override it.

Differences from the OpenAI API to keep in mind: function calling uses the legacy `functions` plus `function_call` pair, which DanyAPI maps from `tools` for you, there is no `n`, `seed`, `stop` or penalty support, and images are uploaded to the GigaChat file storage first, one image per message and ten per request.

Images work on the Pro, Max and Ultra tiers only. `GigaChat-2` and `GigaChat-3-Lightning` are Lite models and reject attachments; the bridge turns that into a 400 naming the models that accept them instead of a raw upstream error.

## Yandex Alice, unofficial

`alice`, `alice-ai` and `yagpt` route to the Yandex Alice consumer endpoint `wss://uniproxy.alice.yandex.net/uni.ws`. It needs no key, no account and no token.

Read this before enabling it. Yandex has no public API for this, so the provider speaks an undocumented internal protocol of a consumer service, and `yandex.ru/legal/alice_chat` clause 4.2 forbids circumventing technical protections and imitating the service's functioning. Yandex also versions and reshapes this protocol without notice, so the provider can break at any time. It is disabled unless you set `ALICE_ENABLED=1`, which is your acknowledgement of the above.

Behaviour to expect from the answers: the endpoint is stateless, so the whole conversation is folded into one prompt; it routes prompts across scenario handlers rather than to a single model, so many prompts get a canned Alice reply or a persona refusal instead of an answer; and there is no incremental text, so streaming sends the finished text in a single chunk.

## Duck.ai, unofficial

The free-tier Duck.ai models, currently `mistral-small-2603`, `gpt-5.6-luna`, `tinfoil/gemma4-31b`, `gpt-5.4-mini`, `tinfoil/gpt-oss-120b` and `claude-haiku-4-5`, route to DuckDuckGo's public Duck.ai chat at `https://duck.ai/duckchat/v1`. The list is not fixed: it is parsed from the model table the Duck.ai web bundle currently ships, filtered to the models DuckDuckGo grants the free tier, and refetched on the model refresh interval. It needs no key, no account and no token.

Read this before enabling it. DuckDuckGo has no public API for this, so the provider speaks an undocumented internal protocol of a consumer service, and the attestation it demands is a browser fingerprint. The Duck.ai terms at `https://duckduckgo.com/duckduckgo-help-pages/duckai` forbid circumventing technical protections and imitating the service's functioning, and DuckDuckGo reshapes this protocol without notice, so the provider can break at any time. It is disabled unless you set `DUCKAI_ENABLED=1`, which is your acknowledgement of the above.
It does work, and it was measured end to end through the real HTTP surface on a host with no proxy: nine of twelve consecutive requests answered with real model output, streaming included. Two things make that possible and are easy to get wrong. duck.ai compares the build identifier in `x-fe-version` against the one it is currently serving, so the provider reads it from the landing page on startup; a stale value is refused as an unsupported entrypoint. And every request carries a proof that a real browser made it, computed by evaluating a script DuckDuckGo serves fresh on each request against a browser environment. The solver in `danyapi/duckai/jsa_solver.js` reproduces that environment: DOM prototype chain, error and stack API, a box model, and an HTML fragment parser. Some of those scripts are HTML parser differentials that only a real parser answers correctly, so a minority of requests is refused with `403` and an explanation. The provider re-solves a freshly served attestation and retries those, keeps a startup miss from stopping the server, and never disables the account over one.

Now the part worth planning around. After a few dozen automated requests in a row, DuckDuckGo stops judging the proof and refuses the client outright: `ERR_BN_LIMIT`, the same "unsupported entrypoint" wording, returned in about a tenth of a second, before any attestation is evaluated. A real Chrome on the same host kept answering normally throughout, and roughly twenty minutes of quiet did not clear it. So this is not your address being blocked and not a solver bug; it is DuckDuckGo deciding a Python client is a client, and the decision is sticky. The honest reading is that a provider which has to impersonate a browser will be recognised eventually, and there is no amount of header tuning in Python that changes that, because what is missing is a browser engine, not a header. Treat this provider as something that works interactively and in light use, not as a dependable backend route, and keep the other providers for anything that has to be reliable.

Differences from the OpenAI API to keep in mind: there is no `system` role, so system and developer messages are folded into the first user turn; `reasoningEffort` is clamped to what the chosen model supports, which is read from the same live table as the model list; there is no `n`, `seed`, penalty or `response_format` support; images must be inline data URIs, at most three per message and ten per request, and file attachments are rejected; and there is no usage accounting upstream, so token counts are estimated from the text. Tool calls are native, and web search and image generation are switched off because the free tier does not grant them.

## Models

Model lists are not hardcoded. Every provider is asked where it runs and the answer is what `GET /v1/models` serves:

| Provider | Source | Auth |
| --- | --- | --- |
| DeepSeek | `model_configs` in the web client settings at `scope=model` | none |
| Qwen | `GET /api/v2/models/` on the account | a token gives the account list, without one it serves the visitor list |
| GigaChat | `GET /models` on the official API | authorization key |
| Duck.ai | the model table in the Duck.ai web bundle, filtered to the free tier | none |
| Alice | the provider's own aliases, upstream serves no catalog | none |

The lists are fetched at startup and refetched every `DANYAPI_MODELS_REFRESH_SECONDS`, and a fetch that fails or comes back empty keeps the last good list rather than emptying `GET /v1/models`. Send `?refresh=1` to `GET /v1/models`, or pass a key in `Authorization` or `x-api-key`, to force a refetch right now and, for GigaChat, to read the list your own key is granted.

Only models the upstream marks enabled are listed, so a DeepSeek model type the account is not given does not appear. DeepSeek ids are its `model_type` values, `default` today, with a `-thinking` sibling for each that toggles reasoning; `deepseek-v4.1-flash` still resolves as an alias of the default one. In BYOK mode the keyless providers are catalogued from the same endpoints without any key, and GigaChat joins the list as soon as a request arrives with a key.

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
