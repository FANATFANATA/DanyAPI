# DanyAPI

OpenAI-compatible HTTP API built on Python + FastAPI. Instead of the paid
DeepSeek API it talks to the internal API of the free web client
[chat.deepseek.com](https://chat.deepseek.com) using a server-side account.
API users need no keys - all requests are made by the server account.

## Features

- `GET /v1/models` - model list
- `POST /v1/chat/completions` - generation (stream and non-stream)
- Models: `deepseek-chat`, `deepseek-reasoner`, `deepseek-vision`
  (internal `model_type`: `default`, `expert`, `vision`)
- Thinking (R1 reasoning) and web search
- Multi-session: the message chain is stored server-side
  (`session_id` in the response), like the web client
- Built-in reverse-engineered PoW hash **DeepSeekHashV1** (23-round Keccak
  with rate 136 and shifted round constants) + a fast native C solver
  (`danyapi/deepseek/pow_solver.c`, compiles with clang)

## Install

```bash
pip install -r requirements.txt
```

## Account setup

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
  -d '{"model": "deepseek-chat", "messages": [{"role": "user", "content": "Hello!"}]}'
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

## Tests

```bash
python -m unittest tests.test_pow tests.test_stream -v
```

## How it works

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

## Native PoW solver (optional)

If you have a C compiler, build the binary for maximum speed:

```bash
clang -O2 -o danyapi/deepseek/pow_solver.exe danyapi/deepseek/pow_solver.c
```

Without it the server falls back to the Node solver (the site's wasm module)
or the pure-Python fallback. All three produce the same answer.

## Account limits

- One chat.deepseek.com account can generate **one message at a time**
  (otherwise the server replies `parallel_chat_limit`). DanyAPI keeps an
  **account pool** and distributes concurrent requests across accounts;
  if all are busy, requests wait in a queue. More tokens = more parallel
  generations.
- Sessions are tied to the account they were created on: repeat requests with
  the same `session_id` route to the same account (conversation history is
  stored server-side on the account).
- DeepSeek may throttle accounts (especially the expert model
  `deepseek-reasoner` - "limited resource"). Responses with `finish_reason`
  `expert_busy_use_default` / `parallel_chat_limit` are retried automatically
  (up to 3 attempts with exponential backoff). If all attempts are exhausted:
  - non-stream requests get HTTP 429 with the DeepSeek error text;
  - stream requests get an SSE `error` event with `finish_reason`.
- The PoW challenge is single-use - a new one is solved per request (the next
  one is prefetched in advance so you don't wait).
