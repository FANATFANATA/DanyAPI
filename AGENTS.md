# AGENTS.md

## Gate: `python tests.py` is the only thing that counts

One command runs all 18 checks (lint, typecheck, tests). Run it before every commit; do not substitute individual tools.

```bash
python tests.py                   # all steps, sequential
python tests.py -j 2              # parallel (what CI uses)
python tests.py -k qwen           # extra args pass through to pytest
python tests.py tests/test_byok.py
```

`OPTIONAL_STEPS = {"ruff format --check", "clang-format", "clang-tidy"}` only ever report `WARN`, never `FAIL`. Every other step must be `PASS`.

Run steps individually while iterating, then the full gate:

```bash
python -m ruff check . && python -m mypy danyapi tests docs && python -m pytest -q
```

### `repo guards` bans text

`tests.py:check_repo_guards` fails on these substrings anywhere in `docs/**` (`*.html *.js *.css *.py *.md *.sh *.bat`) and `README.md`:

- em dash (U+2014)
- en dash (U+2013)
- `Zero keys.` / `Ноль ключей.`
- `live demo`

The first two entries are matched by codepoint, so they are listed that way rather than pasted: a literal em or en dash anywhere in `docs/**` or `README.md` fails the gate. Root-level `AGENTS.md` is not a guard target, so this file is free of them.

It also validates UTF-8 and parses every `*.json` plus `.github/**/*.yml`. Use plain hyphens, not dashes.

## Python 3.10 is the hard floor

`tests/test_syntax_compat.py` parses all of `danyapi/**/*.py`, `docs/setup.py` and `docs/start.py` with `ast.parse(..., feature_version=(3, 10))`. CI matrixes 3.10 through 3.14, `app.py` enforces `MIN_PYTHON = (3, 10)`.

So: no PEP 695 generics, no `except*`. Keep `from __future__ import annotations` at the top of every module. `ruff target-version = "py310"` even though mypy is set to 3.12.

## Architecture: four providers, two shapes

`danyapi/api/chats.py` dispatches via the `CHAT_HANDLERS` dict; `danyapi/api/models.py:_resolve_provider` maps a model-id prefix to a provider. Adding a provider touches both, plus `danyapi/api/state.py:BYOK_PROVIDERS`, `danyapi/api/core.py:POOL_ATTRS`, and `danyapi/api/byok.py:_byok_pool`.

| Provider | Transport | Server-side sessions | Credentials |
| --- | --- | --- | --- |
| `deepseek` | internal web API, custom SSE | yes | `DEEPSEEK_TOKENS` |
| `qwen` | internal web API, custom SSE | yes | `QWEN_TOKENS` |
| `gigachat` | **official** `api.giga.chat` OpenAI-shaped | no, stateless | `GIGACHAT_KEYS` |
| `alice` | unofficial `wss://uniproxy...` WebSocket | no, stateless | none |

`DeepSeekAccount` and `QwenAccount` own a `sessions` attribute (`SessionRegistry` / `QwenSessionRegistry`). `GigaChatAccount` and `AliceAccount` deliberately have none, so `AccountPool` reads it via `getattr(acct, "sessions", None)`. Do not add `sessions` to the stateless providers.

Session-based providers fold message history into a single prompt via `danyapi/tools/prompt.py:build_prompt` (re-exported as `toolemu.build_prompt`); the stateless ones forward the real `messages` array.

## `openai.py` is a deliberate re-export surface

`danyapi/api/openai.py` has a ~270-entry `__all__` and imports many api-package symbols that appear exactly once, in the import line. This is intentional:

- `vulture` runs with `min_confidence = 90` over `paths = ["danyapi", "docs"]` and has **no whitelist file**. Symbols listed in `__all__` count as used.
- It gives tests and other modules one canonical import point.

If you add a public symbol to the `danyapi/api` package or a provider, add it to `__all__` in `openai.py` or vulture will report it as dead code. Keep the list sorted (ruff `RUF022` enforces it).

## `docs/` is linted, type-checked and tested like the rest of the package

`PYTHON_DIRS = ["danyapi", "tests", "docs"]`, and pyright includes `docs`.

- `docs/setup.py` is an interactive installer and has real tests in `tests/test_setup.py` (loaded via `importlib.util.spec_from_file_location`). Changing it means changing those tests.
- `docs/index.html` ships **English as static text** with `data-i18n` keys; `docs/script.js` holds the `ru` dict that replaces it, and an `en` dict with overrides only for keys that differ. To add copy: put English in the HTML, add the Russian string to the `ru` dict, add the `en` entry only if it differs from the HTML.
- Cyrillic string literals trip ruff `RUF001`; new files with Russian text need a `per-file-ignores` entry in `pyproject.toml`.
- `vulture` also scans `docs`, so dead code in `docs/*.py` is reported too.

## Adding a setting or a provider

A new env var is not one edit. Touch all of: `danyapi/config.py` (`Settings.__init__`), `.env.example`, the README env table, `docs/setup.py` (interactive collection), and `tests/conftest.py` (`os.environ.setdefault` stub). A new provider additionally needs `danyapi/api/state.py:BYOK_PROVIDERS`, `danyapi/api/core.py:POOL_ATTRS`, `danyapi/api/byok.py:_byok_pool`, a branch in `danyapi/api/chats.py:CHAT_HANDLERS`, and model routing in `danyapi/api/models.py`. Tests live in `tests/test_<provider>_*.py`.

## Tests

```bash
python -m pytest -q                          # full
python -m pytest tests/test_byok.py -q       # one file
python -m pytest -k "gigachat" -q            # by keyword
```

`pyproject.toml` sets `asyncio_mode = "auto"`, so `async def test_*` needs no decorator.

- Naming: `test_<topic>.py` for behavior, `test_cov_<area>_<part>.py` (split `a`/`b`) for coverage sweeps.
- `tests/conftest.py` sets `BYOK=0` and stubs provider env vars with `os.environ.setdefault`, so a developer's real `.env` cannot leak into a run.
- Import the app as `from danyapi.api.openai import app`. Importing `danyapi.api.state` alone does not register routes, and `TestClient(app)` entered as a context manager runs the real lifespan, which needs credentials and builds a live account pool. For pure endpoint tests use `TestClient(app)` without `with`.
- HTTP errors come back in an OpenAI envelope, not FastAPI's `{"detail": ...}`: assert on `resp.json()["error"]["message"]`.

## Provider-specific gotchas

GigaChat is official and OpenAI-shaped, but not identical to OpenAI:

- `tools` maps onto the legacy `functions` + `function_call` pair. `function_call` is valid **only** at the top level of the request, never on a message object.
- `functions[].parameters` must be a JSON **object**. A JSON string yields `400 invalid JSON syntax`.
- In a returned `function_call`, `arguments` is an **object** even though the docs say string. `danyapi/gigachat/api.py:_translate_message` re-serialises it for OpenAI clients, and `_gigachat_arguments` does the reverse on the way in.
- Images need `purpose="general"` on `POST /files`, and a `Content-Type` header there would break the multipart boundary. `_api_headers(..., json_body=False)` handles that.
- `GigaChat-2` and `GigaChat-3-Lightning` are Lite and reject images; only Pro/Max/Ultra see them. The model list is read from `GET /models` at startup, so do not hardcode model ids.
- The plain `GigaChat` id does not exist and returns 404. Embedding models must be filtered by `"embed" in id.lower()`, not `startswith("embed")` (`GigaEmbeddings-3B-2025-09` slips through the latter).
- TLS needs the bundled Russian root CA. `danyapi/gigachat/tls.py:resolve_ca` returns an `ssl.SSLContext` combining it with certifi, and caches it in `settings.cache_dir`. `httpx` deprecates `verify=<str>`, so pass a context, not a path. The root ships at `danyapi/gigachat/russian_trusted_root_ca.pem`; there is no fetchable official URL for it.

Alice is an undocumented internal protocol of a consumer service. It is **off by default** behind `ALICE_ENABLED=1`, because Yandex's terms for the Alice chat page forbid exactly this and because the protocol shifts without notice. Re-read the terms before restating that claim; do not treat the section number here as authoritative. If you touch the code:

- The server sends `System`/`Ping` application-level directives that the `websockets` library does not answer for you. `_read_loop` must reply `System`/`Pong` with `refMessageId`, or the connection dies.
- Every id in the protocol must be a uuid4 in hex. A base36 id yields `Invalid uuid`.
- `dialog_id` must be `""`. A non-empty one is silently ignored.
- The endpoint is stateless, so history is folded into a prompt, and there is no incremental text, so streaming emits the whole answer in one chunk.

## Git and CI

Branches are `dev` and `prod`. Pull requests target **`prod`**, not dev. `.github/workflows/sync-dev.yml` auto-merges `prod` back into `dev` after every push to prod, so `dev` legitimately falls behind and pushes to it are rejected until you rebase on `origin/dev`.

The Docker image is only pushed for `prod` refs and `v*` tags; the `latest` tag comes from `prod` only.

```bash
git fetch origin && git rebase origin/dev   # resolve, don't force-push
```

## Native PoW solver

`danyapi/deepseek/pow_solver.c` is the only C file. CI builds it with clang before the tests, and `clang-format`/`clang-tidy` gate on it. `app.py:build_solver` compiles it at startup with a portable fallback, and there are Python and JS fallbacks if no compiler exists.
