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
- Thinking and web search (DeepSeek); thinking and search (Qwen)
- Multi-session: the message chain is stored server-side
  (`session_id` in the response), like the web clients

## Install

```bash
pip install -r requirements.txt
```

For development (tests + linting) install the dev extras:

```bash
pip install -r requirements-dev.txt
```

## Account setup

### DeepSeek

Set a pool of tokens (from different accounts), comma-separated:

```bash
export DEEPSEEK_TOKENS="token1,token2,token3"
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

Same model, different token location:

```bash
export QWEN_TOKENS="token1,token2,token3"
```

Grab a token in the browser:
DevTools -> Application -> Local Storage -> https://chat.qwen.ai -> `token`.

Or a single email + password account (login happens at startup):

```bash
export QWEN_EMAIL="you@example.com"
export QWEN_PASSWORD="secret"
```

Both providers are optional. Run at least one of them (or both) - requests
are routed to the right provider by the model name (`deepseek-*` / `qwen*`).

## Run

The `.env` file (gitignored, created from `.env.example`) is loaded
automatically at startup:

```bash
cp .env.example .env   # fill in tokens
python -m danyapi
# or
uvicorn danyapi.api.openai:app --host 0.0.0.0 --port 8000
```

## Usage

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
  "model": "deepseek-reasoner",
  "messages": [{"role": "user", "content": "2+2?"}],
  "session_id": "<id from the previous response>",
  "thinking": true,
  "search": false,
  "stream": true
}
```

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
   are injected into the prompt sent to the upstream model.
2. The model replies with a JSON object like
   `{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Moscow"}}]}`.
   DanyAPI parses it and returns a proper OpenAI response:
   `message.tool_calls` (non-stream) or streamed `delta.tool_calls` chunks,
   both with `finish_reason: "tool_calls"`. Multiple calls in one reply are
   supported (`parallel_tool_calls`).
3. You run the tool, then send back the result:
   `{"role": "tool", "tool_call_id": "<id>", "content": "22C, sunny"}`.
   DanyAPI renders the tool results into the prompt and continues the
   conversation until the model answers (or calls more tools).

Notes:

- `tool_choice`: `"auto"` (default), `"none"` (tools are accepted but no
  schema is injected), `"required"`, or
  `{"type": "function", "function": {"name": "<tool>"}}`.
- Pass the `session_id` from the first response back in the tool-result
  request to keep the conversation server-side. Without it the whole message
  history (including tool results) is replayed into the prompt instead, so
  plain OpenAI-protocol clients work too.
- While `tools` are present, streamed replies are buffered until the model
  finishes so the reply can be classified as a tool call or plain text.
  Reasoning (`reasoning_content`) is streamed live in both cases.

## Tests

```bash
python -m unittest tests.test_pow tests.test_stream tests.test_accounts tests.test_retry tests.test_qwen_stream tests.test_qwen_api -v
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

If you have a C compiler, build the binary for maximum speed:

```bash
clang -O2 -o danyapi/deepseek/pow_solver.exe danyapi/deepseek/pow_solver.c  # Windows
clang -O2 -o danyapi/deepseek/pow_solver danyapi/deepseek/pow_solver.c      # Linux/macOS
```

Without it the server falls back to the Node solver (the site's wasm module)
or the pure-Python fallback. All three produce the same answer.

## Account limits

- One chat.deepseek.com account can generate **one message at a time**
  (otherwise the server replies `parallel_chat_limit`). DanyAPI keeps an
  **account pool** and distributes concurrent requests across accounts;
  if all are busy, requests wait in a queue. More tokens = more parallel
  generations. The same applies to chat.qwen.ai accounts (Qwen uses its own
  pool, so DeepSeek and Qwen parallel generations are independent).
- Sessions are tied to the account they were created on: repeat requests with
  the same `session_id` route to the same account (conversation history is
  stored server-side on the account).
- DeepSeek may throttle accounts (especially the expert model
  `deepseek-v4-pro` - "limited resource"). Responses with `finish_reason`
  `expert_busy_use_default` / `parallel_chat_limit` are retried automatically
  (up to 3 attempts with exponential backoff). If all attempts are exhausted:
  - non-stream requests get HTTP 429 with the DeepSeek error text;
  - stream requests get an SSE `error` event with `finish_reason`.
- Qwen may reply `Too_Many_Requests` / `RateLimited` / `quotaLimited`;
  those are retried automatically the same way (up to 3 attempts), then
  reported as HTTP 429 or an SSE `error` event.
- The DeepSeek PoW challenge is single-use - a new one is solved per request
  (the next one is prefetched in advance so you don't wait).
