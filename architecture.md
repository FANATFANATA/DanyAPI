# DanyAPI Architecture

FastAPI proxy that exposes a unified OpenAI-compatible `/v1/chat/completions` endpoint over both DeepSeek and Qwen (Tongyi Qianwen) providers. Manages multiple accounts per provider, handles sessions, context deduplication, retry logic with backoff, tool calls, streaming, and file attachments.

## Module Map

```
danyapi/
├── __init__.py
├── __main__.py              # entry point (uvicorn launch)
├── config.py                # settings from env / .env
├── logging.py               # log level setup
│
├── api/                     # OpenAI-compatible HTTP layer
│   ├── openai.py            # FastAPI app: lifespan, routes, provider dispatch
│   ├── models.py            # pydantic models (ChatMessage, ChatCompletionRequest), model constants
│   ├── attachments.py       # file attachment collection and upload to DeepSeek
│   ├── deepseek.py          # DeepSeek-specific helpers: session prep, auth, retry logic
│   └── streaming.py         # non-streaming + streaming OpenAI response generators (DeepSeek)
│
├── tools/                   # prompt building, tool-call parsing/formatting
│   ├── prompt.py            # build_prompt, context_sequence, render_*, ToolCall
│   ├── parse.py             # DSML regexes, JSON/XML tool-call parsers
│   └── format.py            # format_tool_message, tool_call_deltas (SSE)
│
├── deepseek/                # DeepSeek provider internals
│   ├── client.py            # HTTP client: auth, login, completion, upload_file
│   ├── stream.py            # IncrementalSSE parser, MessageReconstructor
│   └── sse.py               # SSE event types
│
├── qwen/                    # Qwen provider internals
│   ├── api.py               # collect_non_stream + stream_openai (Qwen)
│   ├── client.py            # HTTP client: auth, login, completion, fetch_models
│   ├── accounts.py          # QwenAccount, QwenSessionRegistry
│   └── stream.py            # QwenStreamReconstructor
│
├── retry.py                 # shared retry constants and helpers (MAX_RETRIES, backoff, _drop_session)
├── accounts.py              # AccountPool, DeepSeekAccount, ContextIndex, account_lock
├── sessions.py              # SessionRegistry (LRU cache with persistence)
├── store.py                 # JsonStore (file-backed JSON key-value store)
└── pow.py                   # Proof-of-Work manager for DeepSeek auth
```

## Request Flow

```
Client POST /v1/chat/completions
  │
  ├─ openai.py: chat_completions(req)
  │   └─ _resolve_provider(model) → "deepseek" | "qwen"
  │
  ├─ DeepSeek path:
  │   ├─ build_prompt() + context dedup via pool.resolve_context()
  │   ├─ acquire account from AccountPool (session affinity / round-robin)
  │   ├─ streaming.py: _stream_openai or _collect_non_stream
  │   │   ├─ deepseek.py: _prepare_session → obtain/create session
  │   │   ├─ retry loop with backoff (up to MAX_RETRIES=5)
  │   │   ├─ _fresh_pow_headers + _send_with_auth → SSE stream
  │   │   ├─ MessageReconstructor parses SSE events
  │   │   ├─ context continuation via _collect_continuation if input_exceeds_limit
  │   │   └─ emit OpenAI-format response (stream or non-stream)
  │   └─ tool-call parsing: tools/parse.py → format_tool_message / tool_call_deltas
  │
  └─ Qwen path:
      ├─ build_prompt() + context dedup
      ├─ acquire account from qwen_pool
      ├─ qwen/api.py: stream_openai or collect_non_stream
      │   ├─ _prepare_session → obtain/create session (with model_id)
      │   ├─ retry loop with backoff
      │   ├─ QwenStreamReconstructor parses SSE events
      │   └─ emit OpenAI-format response
      └─ tool-call parsing: same tools/ module
```

## Key Design Decisions

### Provider Dispatch
`_resolve_provider(model)` routes by model name prefix: `qwen*` → Qwen, everything else (including `deepseek*`) → DeepSeek. Both paths share the same `ChatCompletionRequest` schema and return OpenAI-compatible responses.

### Account Pooling
`AccountPool` manages multiple accounts per provider with:
- **Session affinity**: a given session always goes to the account that created it.
- **Context deduplication**: `ContextIndex` matches message fingerprints across sessions so identical conversations reuse existing sessions.
- **Round-robin fallback**: new requests distribute evenly when no session is specified.
- **Concurrency control**: each account has an `asyncio.Semaphore(1)`; `account_lock()` enforces exclusive access with a configurable timeout that raises `AccountPoolBusy` on contention.

### Retry Logic
Shared constants in `retry.py`:
- `MAX_RETRIES = 5`, exponential backoff from 1s to 8s max.
- `RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}`.
- Provider-specific retryable conditions (DeepSeek: busy hints; Qwen: error codes) are handled separately in their respective modules.

### Session Persistence
`SessionRegistry` is an LRU cache backed by `JsonStore` on disk. Sessions survive restarts within TTL. Both DeepSeek and Qwen use the same pattern — Qwen overrides `_create`, `_reuse`, and deserialization to pass `model_id`.

### Streaming
Both providers yield OpenAI-compatible SSE chunks (`data: {...}\n\n`). The response generator buffers deltas until content arrives, then flushes in order to avoid sending empty role-only prefixes prematurely. On disconnect or cancellation, the upstream stream is stopped via provider-specific `stop_stream()` calls.

## File Sizes (post-refactor)

| Module | Lines | Role |
|--------|-------|------|
| `api/openai.py` | 397 | FastAPI app, routes, provider dispatch, re-exports |
| `api/streaming.py` | 563 | DeepSeek streaming + non-stream response generators |
| `qwen/api.py` | 631 | Qwen streaming + non-stream response generators |
| `tools/parse.py` | 541 | DSML, JSON, XML tool-call parsing |
| `accounts.py` | 316 | AccountPool, DeepSeekAccount, ContextIndex |
| `api/deepseek.py` | 268 | DeepSeek session/auth/retry helpers |
| `tools/prompt.py` | 384 | Prompt building, context sequence extraction |
| `sessions.py` | 148 | SessionRegistry (LRU + persistence) |
| `api/attachments.py` | 108 | File attachment handling |
| `api/models.py` | 102 | Pydantic models, model constants |
| `retry.py` | 27 | Shared retry constants and helpers |
| `tools/format.py` | 44 | Tool message formatting for SSE |
