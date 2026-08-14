# DanyAPI

OpenAI-compatible HTTP API built on Python + FastAPI. Instead of the paid
APIs it talks to the internal APIs of the free web clients
[chat.deepseek.com](https://chat.deepseek.com) and
[chat.qwen.ai](https://chat.qwen.ai) using server-side accounts.
API users need no keys - all requests are made by the server accounts.

## Features

- `GET /v1/models` - model list
- `POST /v1/chat/completions` - generation (stream and non-stream)
- DeepSeek models: `deepseek-v4-flash` (default), `deepseek-v4-pro`
  (expert), `deepseek-v4-vision` (vision). Internal `model_type`: `default`,
  `expert`, `vision`.
- Thinking is available for all DeepSeek models; web search works only for
  `deepseek-v4-flash`.
- Attachments: `deepseek-v4-vision` accepts images only; `deepseek-v4-flash`
  accepts images (OCR) and text files; `deepseek-v4-pro` accepts no files.
  Per request: max 50 files, 100 MB each.
- Qwen models: fetched from the account at startup
  (`qwen3.8-max`, `qwen3.7-plus`, ...)
- Thinking and web search (DeepSeek); thinking and search (Qwen). Thinking
  traces are exposed as `reasoning_content` (streamed live, both providers)
- Tool calling (emulated): `tools` / `tool_choice` / `parallel_tool_calls`
  with proper `finish_reason: "tool_calls"` responses
- JSON mode (emulated): `response_format` with `json_object` / `json_schema`
- `system` messages are injected as the model's system prompt
- Real token usage in responses, accumulated per conversation like the
  official API; streaming usage via `stream_options.include_usage`
- On-disk session cache: conversations survive server restarts
- `GET /health` - readiness probe
- Multi-session: the message chain is stored server-side
  (`session_id` in the response), like the web clients. Stateless requests
  (no `session_id`) reuse the same server-side chat automatically based on
  the message context, so plain OpenAI-protocol clients keep their
  conversation too.

## Install

Requires Python 3.13+.

```bash
pip install -r requirements.txt
```

For development (tests + linting) install the dev extras:

```bash
pip install -r requirements-dev.txt
```

## Account setup

### DeepSeek

Set a pool of tokens (from different accounts), comma-separated, or a single
token via `DEEPSEEK_TOKEN`:

```bash
export DEEPSEEK_TOKENS="token1,token2,token3"
# or
export DEEPSEEK_TOKEN="token1"
```

Each account can generate **one** message at a time, so a pool of N tokens
gives up to N parallel generations. Grab a token in the browser:
DevTools -> Application -> Local Storage -> https://chat.deepseek.com -> `userToken`.

Or a single email + password account (login happens at startup):

```bash
export DEEPSEEK_EMAIL="you@example.com"
export DEEPSEEK_PASSWORD="secret"
```

### Qwen

Same model, different token location (single token via `QWEN_TOKEN`):

```bash
export QWEN_TOKENS="token1,token2,token3"
# or
export QWEN_TOKEN="token1"
```

Grab a token in the browser:
DevTools -> Application -> Local Storage -> https://chat.qwen.ai -> `token`.

For Qwen accounts used by DanyAPI, disable the built-in **Tools** switch
(code interpreter, image generation, and other Qwen built-ins) in the
chat.qwen.ai web interface. When it is enabled, Qwen's built-in tools fire
their own response phases on every request, which DanyAPI cannot parse, and
the server-side conversation history grows fast (hitting the input token
limit quickly). With the switch off, DanyAPI's emulated tool calling
(`tools` / `tool_choice`) and plain chat work as intended.

Or a single email + password account (login happens at startup):

```bash
export QWEN_EMAIL="you@example.com"
export QWEN_PASSWORD="secret"
```

Both providers are optional. Run at least one of them (or both) - requests
are routed to the right provider by the model name (`deepseek-*` / `qwen*`).

At startup every token is validated against its provider; invalid or expired
tokens are skipped with a warning (the server refuses to start when no valid
credential remains). The Qwen model list for `/v1/models` is fetched from the
first Qwen account at startup (text-chat models only); if the fetch fails,
a built-in default list is used. New requests without a `session_id` are
distributed across healthy accounts round-robin.

## Run

The `.env` file (gitignored, created from `.env.example`) is loaded
automatically at startup:

```bash
cp .env.example .env   # fill in tokens
python -m danyapi
# or
uvicorn danyapi.api.openai:app --host 0.0.0.0 --port 8000
```

Or with the helper scripts:

```bash
run.bat   # Windows
./run.sh  # Linux/macOS
```

## Logging

By default logs go to the console. To persist them to a file, set
`DANYAPI_LOG_FILE` in `.env` (or as an environment variable):

```bash
DANYAPI_LOG_FILE=/var/log/danyapi.log   # or danyapi.log for the working dir
DANYAPI_LOG_LEVEL=INFO                  # DEBUG / INFO / WARNING / ERROR
DANYAPI_LOG_MAX_BYTES=10485760          # rotate at 10 MB per file
DANYAPI_LOG_BACKUP_COUNT=3              # keep 3 rotated files
```

The file is rotated by size (`DANYAPI_LOG_MAX_BYTES`, default 10 MB) keeping
`DANYAPI_LOG_BACKUP_COUNT` backups (default 3). When started via
`python -m danyapi`, uvicorn's own startup/access logs are routed through the
same root logger and also land in the file.

## Docker

```bash
docker build -t danyapi .
docker run -d -p 8000:8000 \
  -e DEEPSEEK_TOKENS="token1,token2" \
  -e QWEN_TOKENS="token3" \
  danyapi
```

The `.env` file is not baked into the image; pass credentials as environment
variables or mount your `.env` as a volume (`-v /path/to/.env:/app/.env`).

### Prebuilt image (GHCR)

CI builds and pushes the image to the GitHub Container Registry on every push
to `main` (`latest` and `sha-*` tags) and on version tags (`v1.2.3`):

```bash
docker run -d -p 8000:8000 \
  -e DEEPSEEK_TOKENS="token1,token2" \
  -e QWEN_TOKENS="token3" \
  ghcr.io/fanatfanata/danyapi:latest
```

## Environment variables

All configuration is environment-driven (`.env` or exported vars). Provider
credentials are documented in [Account setup](#account-setup).

| Variable | Default | Description |
| --- | --- | --- |
| `DANYAPI_HOST` | `0.0.0.0` | Address the API server binds to |
| `DANYAPI_PORT` | `8000` | Port the API server listens on |
| `DANYAPI_TIMEOUT` | `60` | Upstream request timeout in seconds |
| `DANYAPI_ACQUIRE_TIMEOUT` | (empty) | Seconds to wait for a free account before returning 429; empty = wait forever |
| `DANYAPI_SESSION_CACHE_SIZE` | `128` | Max server-side chats cached per provider (LRU) for stateless session reuse |
| `DANYAPI_SESSION_TTL_SECONDS` | `3600` | Seconds an unused session/context stays reusable; `0` = never expire |
| `DANYAPI_CACHE_DIR` | (empty) | Directory for the on-disk session cache; empty = system temp dir (`%TEMP%\danyapi` / `/tmp/danyapi`) |
| `DANYAPI_CACHE_DISABLED` | (empty) | Set to `1`/`true`/`yes`/`on` to disable the on-disk cache entirely |
| `DANYAPI_LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `DANYAPI_LOG_FILE` | (empty) | File path for persistent logs; empty = console only |
| `DANYAPI_LOG_MAX_BYTES` | `10485760` | Max log file size in bytes before rotation |
| `DANYAPI_LOG_BACKUP_COUNT` | `3` | Number of rotated log files to keep |
| `DANYAPI_HUMAN_DELAY_MIN` | `0.5` | Minimum delay in seconds before sending a request (uniform random jitter) |
| `DANYAPI_HUMAN_DELAY_MAX` | `3.0` | Maximum delay in seconds; set both to `0` to disable |

## Usage

OpenAI SDK usage (drop-in replacement for the official API):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy")
r = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(r.choices[0].message.content)
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Or with a Qwen model:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-max", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Multi-turn: the response includes `session_id`; pass it in the next request
to continue the same conversation.

```json
{
  "model": "deepseek-v4-flash",
  "messages": [{"role": "user", "content": "2+2?"}],
  "session_id": "<id from the previous response>",
  "thinking": true,
  "search": false,
  "stream": true
}
```

Stateless clients (no `session_id`) don't lose context either: the server
derives a fingerprint from the `system`/`user` messages and reuses the
matching server-side chat. If the message context is identical - the same
chat is used; if the context is a continuation of a previously seen one -
that chat is continued, so multi-turn conversations work even when the
client never echoes `session_id`. Pass `session_id` to force an exact
conversation (or to branch off into an independent chat). The in-memory
cache is LRU-bounded per provider (`DANYAPI_SESSION_CACHE_SIZE`, default
128).

The context fingerprint is scoped by the `user` field when the client sends
it: two different `user` values never share a server-side chat even for
identical messages, so stateless multi-tenant clients stay isolated. Cache
entries expire after `DANYAPI_SESSION_TTL_SECONDS` (default 3600, `0`
disables expiry) so stale chats are dropped instead of being reused.
Qwen chats are also model-aware: a `session_id` created for one Qwen model
is automatically migrated to a new chat if a request switches models.

No context is ever duplicated or lost:

- A reused chat receives **only the delta**: the new user message, or the
  tool round tail (tool results) in the case of a tool call. The full
  conversation history lives server-side in the chat.
- The tool schema / system prompt are injected **once**, into the first
  message of a chat, and are not repeated in every follow-up message.
- If a chat cannot be matched (cache miss or eviction), the whole message
  history is replayed into a fresh chat, so the model always sees the full
  conversation.

### Request fields

`POST /v1/chat/completions` accepts:

| Field | Default | Notes |
| --- | --- | --- |
| `model` | `deepseek-v4-flash` | `deepseek-*` routes to DeepSeek, `qwen*` to Qwen; anything else is HTTP 404 |
| `messages` | `[]` | OpenAI format; `content` may be a string or a list of `text`/`image_url` parts |
| `stream` | `false` | SSE stream (`data:` chunks + `data: [DONE]`) |
| `thinking` | provider-specific | DeepSeek: off by default, on for `deepseek-v4-pro`; Qwen: on by default |
| `search` | `false` | Web search; DeepSeek honors it only for `deepseek-v4-flash` |
| `session_id` | `null` | Continue an exact server-side conversation |
| `user` | `null` | Scopes the stateless context fingerprint (multi-tenant isolation) |
| `files` | `null` | DeepSeek attachments: `{name, content (base64), content_type}` |
| `tools`, `tool_choice`, `parallel_tool_calls` | `null` | Emulated tool calling (see below) |
| `response_format` | `null` | Emulated JSON mode (see below) |
| `stream_options` | `null` | `{"include_usage": true}` adds `usage` to the final SSE chunk |
| `temperature`, `top_p` | `null` | Accepted for compatibility but ignored: the upstream web APIs have no sampling parameters |

The response additionally carries `reasoning_content` (the thinking trace)
in `message` (non-stream) or `delta` (stream) when thinking is enabled,
and `session_id` to continue the conversation.

### Session persistence

The session registry (chat ids, context fingerprints, session -> account
affinity, accumulated usage counters) is written to disk as JSON files, so
conversations survive server restarts and keep pointing at the same
server-side chats and accounts:

- Location: `DANYAPI_CACHE_DIR`, default is the system temp dir
  (`%TEMP%\danyapi` on Windows, `/tmp/danyapi` on Linux/macOS).
- Files: `<provider>-sessions-default.json`, `<provider>-contexts-default.json`,
  `<provider>-affinities-default.json`.
- Writes are atomic (temp file + rename), so a crash mid-write cannot corrupt
  the cache.
- Set `DANYAPI_CACHE_DISABLED=1` to keep everything in memory only.
- In Docker, mount `DANYAPI_CACHE_DIR` as a volume if you want sessions to
  survive container recreation.

Note that the upstream chats themselves live on the provider side; the local
cache only maps your `session_id` / message context to them.

## File attachments (DeepSeek)

Send files as base64 in the `files` field, or as `image_url` (data URI) parts
inside a message:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "What magic number is in the file?"}],
    "files": [{"name": "secret.txt", "content": "<base64>", "content_type": "text/plain"}]
  }'
```

```json
{
  "model": "deepseek-v4-vision",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe the image."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64>"}}
    ]
  }]
}
```

Per-model limits: `deepseek-v4-vision` accepts images only;
`deepseek-v4-flash` accepts images (OCR) and text files; `deepseek-v4-pro`
accepts no files. Max 50 files, 100 MB each per request.

## Tool calling (emulated)

Neither chat.deepseek.com nor chat.qwen.ai exposes a native function-calling
API, so DanyAPI emulates it at the proxy layer with prompt injection. The
OpenAI-compatible `tools`, `tool_choice` and `parallel_tool_calls` request
fields are accepted:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "What is the weather in Moscow?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the weather in a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

How it works:

1. When `tools` are present, the function schema and a strict JSON instruction
   (with a concrete example, no template placeholders) are injected into the
   prompt sent to the upstream model.
2. The model replies with a tool call - DanyAPI understands several formats
   and normalizes them all to a proper OpenAI response:
   - JSON `{"tool_calls": [{"name": "...", "arguments": {...}}]}`;
   - legacy `{"function_call": {...}}`;
   - a bare dict `{"name": "...", "arguments": {...}}` (Qwen/DeepSeek style)
     or a bare array `[...]` of them;
   - XML/Anthropic style `<tool_calls><invoke name="...">...</invoke></tool_calls>`
     (arguments as child tags, `<parameter name="...">`, or inline JSON).
   The result is `message.tool_calls` (non-stream) or streamed
   `delta.tool_calls` chunks, both with `finish_reason: "tool_calls"`.
   Any number of calls in one reply are supported (`parallel_tool_calls`),
   so clients that ship many tools (e.g. opencode) work out of the box.
3. You run the tool, then send back the result:
   `{"role": "tool", "tool_call_id": "<id>", "content": "22C, sunny"}`.
   DanyAPI renders the tool results into the prompt and continues the
   conversation until the model answers (or calls more tools).

Notes:

- `tool_choice`: `"auto"` (default), `"none"` (tools are accepted but no
  schema is injected), `"required"`, or
  `{"type": "function", "function": {"name": "<tool>"}}`.
- Pass the `session_id` from the first response back in the tool-result
  request to keep the conversation server-side. Stateless clients work too:
  the context cache reuses the chat from the previous round, so the tool
  results are sent as the continuation of the same server-side conversation
  instead of replaying the whole history into a brand-new chat. If a session
  cannot be matched (e.g. the cache was evicted), the whole message history
  (including tool results) is replayed into the prompt instead, so plain
  OpenAI-protocol clients keep working.
- While `tools` are present, streamed replies are buffered until the model
  finishes so the reply can be classified as a tool call or plain text.
  Reasoning (`reasoning_content`) is streamed live in both cases.

## JSON mode (emulated)

`response_format` is emulated the same way as tool calling - the JSON
constraint (and an optional schema) is injected into the prompt:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Extract the city and temperature."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "schema": {
          "type": "object",
          "properties": {"city": {"type": "string"}, "temperature": {"type": "number"}},
          "required": ["city", "temperature"]
        }
      }
    }
  }'
```

`response_format` accepts `"json_object"`, `{"type": "json_object"}` and
`{"type": "json_schema", "json_schema": {...}}`. As with tool calling this is
prompt-level emulation: replies are JSON in practice but not guaranteed
schema-valid - validate on the client side.

## System prompt and health

- `system` messages are collected and injected as the model's system prompt
  in front of the first user turn (upstream web APIs have no dedicated
  `system` field).
- `GET /health` returns `{"status": "ok", "deepseek": true, "qwen": true}`
  plus per-provider cache stats (`deepseek_stats` / `qwen_stats`: account
  health, session-affinity count, context-cache size and hit/miss counters) -
  useful for readiness probes and load balancers.
- When the client disconnects mid-generation, DanyAPI tells the upstream
  provider to stop the stream, so the server-side chat does not keep a
  partial response.

## Token usage

Every response carries an OpenAI-style `usage` object, and both providers
report it **accumulated per conversation** (like the official API, where the
counter grows with every turn of the same chat):

- **Qwen** - the upstream reports per-turn prompt/completion counts; DanyAPI
  sums them per chat. `prompt_tokens` = total input tokens processed by this
  conversation so far, `completion_tokens` = total output generated,
  `total_tokens` = their sum.
- **DeepSeek** - the web API only exposes a single cumulative counter
  (`accumulated_token_usage`), so `completion_tokens` is the total generated
  in this conversation so far and `prompt_tokens` is always `0`
  (`total_tokens` equals `completion_tokens`).

The counters live on the session, so a new conversation starts from zero and
continuing a `session_id` keeps counting. They are also persisted in the
on-disk session cache (survive restarts).

Streaming usage: pass `"stream_options": {"include_usage": true}` and the
final SSE chunk carries the same accumulated `usage` (like the official API).

## Error handling

Non-stream requests get a plain HTTP error; stream requests get an SSE
`error` event (and `data: [DONE]`) once the stream is open.

| Status | When |
| --- | --- |
| `400` | Bad request: invalid base64/files, per-model attachment rules violated, context length exceeded |
| `401` | DeepSeek auth error (invalid/expired token); the account is then marked broken and excluded from the pool |
| `404` | Unknown model name |
| `429` | All accounts busy (`DANYAPI_ACQUIRE_TIMEOUT` expired) or upstream throttling after retries are exhausted |
| `502` | Upstream request failed (network, file upload, Qwen WAF challenge) |
| `503` | Provider not configured (no tokens/email for that provider) or all its accounts are broken |

Retries: `expert_busy_use_default` / `parallel_chat_limit` / `server_busy` /
`busy` (DeepSeek) and `Too_Many_Requests` / `RateLimited` / `quotaLimited`
(Qwen) are retried automatically up to 5 times with exponential backoff
(1s, capped at 8s) before surfacing as errors. Responses can also end with
`finish_reason: "content_filter"` when the upstream moderates the output.
See [Account limits](#account-limits).

## Tests

```bash
python -m pytest
```

Lint and format (ruff):

```bash
ruff check .
ruff format .
```

Tests, lint and the native solver build also run in CI on every push
(`.github/workflows/ci.yml`).

## How it works

### DeepSeek

Protocol reverse-engineered from the chat.deepseek.com main bundle
(`fe-static.deepseek.com/chat/static/main.4e922c397f.js`) and the wasm module
`sha3_wasm_bg.7b9ca65ddd.wasm`:

- Auth: `POST /api/v0/users/login` -> `data.biz_data.user.token`,
  then `Authorization: Bearer <token>`.
- Headers: `x-client-bundle-id`, `x-client-platform`, `x-client-version`,
  `x-client-locale`, `x-client-timezone-offset`.
- Session: `POST /api/v0/chat_session/create` (empty body) -> `chat_session.id`.
- Generation: `POST /api/v0/chat/completion`:
  `{chat_session_id, parent_message_id, model_type, prompt, ref_file_ids,
  thinking_enabled, search_enabled, action, preempt}`.
- Response - `text/event-stream`: `ready` events, deltas
  (`SET`/`APPEND`/`BATCH`, `response/...` paths), `finish`, `close`.
- PoW header `X-DS-PoW-Response` - base64 of
  `{algorithm, challenge, salt, answer, signature, target_path}`.
  The challenge is single-use: `answer` = minimal counter c where
  `DeepSeekHashV1(f"{salt}_{expire_at}_" + str(c))` matches `challenge`
  (32 bytes). The server iterates c in `[0, difficulty)`.

### Qwen

Protocol reverse-engineered from the chat.qwen.ai frontend bundle
(`assets.alicdn.com/g/qwenweb/qwen-chat-fe/0.2.83/js/main.js`):

- Auth: `POST /api/v2/auths/signin` with `{email, password}` where the
  password is SHA-256 hex of the plain text -> `data.token` (JWT).
  Requests send it as `Authorization: Bearer <token>` and the `token` cookie.
- Headers: `source: web`, `version: 0.2.83`, `X-Request-Id`, `Timezone`,
  browser `sec-ch-ua`/`User-Agent`/`Origin`/`Referer`.
- Session: `POST /api/v2/chats/new` (`{chatId, models, chat_type: "t2t",
  chat_mode: "normal", timestamp}`) -> `data.id` (the chat id).
- Generation: `POST /api/v2/chat/completions?chat_id=<id>` with
  `{stream, version: "2.1", incremental_output, chat_id, model, parent_id,
  messages: [{fid, parentId, role, content, chat_type: "t2t", feature_config:
  {thinking_enabled, output_schema: "phase", ...}}]}`.
  The chat history lives server-side; `parent_id` points at the last assistant
  response id, so the next turn continues the same conversation.
- Response - `text/event-stream` of OpenAI-style JSON chunks:
  `{"choices": [{"delta": {"role", "content", "phase", "status"}}],
  "response_id", "usage"}`. The `response.created` chunk opens the stream with
  the assistant `response_id`; content is streamed in the `answer` phase,
  thinking in `think`/`DeepThinking`/`thinking_summary` phases, and the stream
  ends with a chunk whose `delta.status` is `finished`.

## Native PoW solver (optional)

If you have a C compiler (`clang` or `gcc`), build the binary for maximum speed:

```bash
clang -O2 -o danyapi/deepseek/pow_solver.exe danyapi/deepseek/pow_solver.c  # Windows
clang -O2 -o danyapi/deepseek/pow_solver danyapi/deepseek/pow_solver.c      # Linux/macOS
```

The prebuilt Docker image compiles this binary at build time, so container
users get the native solver out of the box.

Solvers are tried in order until one succeeds: native binary -> Node solver
(the site's wasm module) -> pure-Python Keccak fallback (capped at 2M
iterations, so it only handles low difficulties). Each solved header is
single-use; the next challenge is prefetched in the background so the next
request does not wait.

## Account limits

- One chat.deepseek.com account can generate **one message at a time**
  (otherwise the server replies `parallel_chat_limit`). DanyAPI keeps an
  **account pool** and distributes concurrent requests across accounts;
  if all are busy, requests wait in a queue. More tokens = more parallel
  generations. The same applies to chat.qwen.ai accounts (Qwen uses its own
  pool, so DeepSeek and Qwen parallel generations are independent).
- Sessions are tied to the account they were created on: repeat requests with
  the same `session_id` (or the same cached message context) route to the
  same account, so the conversation history stays intact server-side.
- The in-memory session/context cache is LRU-bounded per account and per
  provider. When an entry is evicted, the corresponding chat is no longer
  reused and a new one is created on the next request; the explicit
  `session_id` remains the reliable way to pin a conversation. Unused
  entries also expire after `DANYAPI_SESSION_TTL_SECONDS`, and session-id ->
  account affinity is cleaned up together with the cache, so memory stays
  bounded on long-running instances.
- DeepSeek may throttle accounts (especially the expert model
  `deepseek-v4-pro` - "limited resource"). Responses with `finish_reason`
  `expert_busy_use_default` / `parallel_chat_limit` / `server_busy` / `busy`
  are retried automatically (up to 5 retries with exponential backoff). If all
  attempts are exhausted:
  - non-stream requests get HTTP 429 with the DeepSeek error text;
  - stream requests get an SSE `error` event with `finish_reason`.
- Qwen may reply `Too_Many_Requests` / `RateLimited` / `quotaLimited`;
  those are retried automatically the same way (up to 5 retries), then
  reported as HTTP 429 or an SSE `error` event.
- The DeepSeek PoW challenge is single-use - a new one is solved per request
  (the next one is prefetched in advance so you don't wait).
- When all accounts are busy, requests wait for a free account. Set
  `DANYAPI_ACQUIRE_TIMEOUT` (seconds) to cap that wait and get an HTTP 429
  ("all accounts are busy") instead of waiting forever.
- Long server-side conversations eventually exceed the model's context window.
  When that happens DanyAPI detects the context-limit error, discards the
  overflowing chat (so it is never reused) and reports the failure: HTTP 400
  for non-stream requests, or an SSE `error` event with
  `finish_reason: "length"` for stream requests. The next request automatically
  starts with a fresh conversation. To avoid hitting the limit often, keep Qwen
  built-in tools disabled (see above) and rotate long conversations client-side.
