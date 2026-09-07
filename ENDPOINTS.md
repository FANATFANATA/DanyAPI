# DanyAPI Endpoints Reference

DanyAPI provides an OpenAI-compatible interface with reverse-engineered support for DeepSeek and Qwen web platforms, including session management, stateful multi-turn continuations, and direct file/image uploads.

---

## 1. Chat & Completions

### `POST /v1/chat/completions`
OpenAI-compatible chat completion endpoint. Supports streaming SSE and non-streaming responses.

- **Supported Models**:
  - DeepSeek: `deepseek-v4-flash` (default), `deepseek-v4-pro`, `deepseek-v4-vision` (add `-thinking` for reasoning trace)
  - Qwen: `qwen3.8-max`, `qwen-plus`, `qwen-turbo`, etc.
- **Stateful Sessions**: Pass `"session_id": "your-alias"` to reuse conversation context across requests.
- **File Attachments**: Uploaded files via `/v1/files` are **automatically attached** to your completion. You can also explicitly pass `file_ids: ["file-..."]` or inline base64 `files: [...]`.

#### Example Request (Basic)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Hello world"}]
  }'
```

#### Example Request (With Session & Explicit Files)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "session_id": "my-session",
    "file_ids": ["file-12345678"],
    "messages": [{"role": "user", "content": "Analyze the attached file."}]
  }'
```

---

## 2. File & Image Uploads

Files uploaded via DanyAPI are streamed directly in-memory without saving to local disk.
- **DeepSeek**: Solves Proof of Work (PoW) challenges and streams to DeepSeek's upload endpoint.
- **Qwen**: Obtains temporary STS upload credentials and streams binary bytes directly to Alibaba Cloud OSS (`qwen-webui-prod.oss-accelerate.aliyuncs.com`).

### `POST /v1/files`
Upload a file or image to DeepSeek or Qwen. Uploaded files are **automatically staged** and attached to your next chat completion for that `session_id`.

- **Provider Resolution**:
  - Pass `model=qwen3.7-plus` (or any Qwen model) to upload to **Qwen**.
  - Pass `model=deepseek-v4-vision` (or any DeepSeek model) to upload to **DeepSeek**.
  - If `model` is omitted, DanyAPI uses the provider from the active session (`session_id`), or whichever provider pool is configured.

- **Supported Upload Formats**:
  1. Standard `multipart/form-data` (`file=@path/to/file`)
  2. JSON body with base64 payload (`{"file": "<base64>", "filename": "...", "session_id": "...", "model": "..."}`)
- **Parameters**:
  - `file`: The binary file or base64 string (required).
  - `session_id`: Optional string. Associates and stages the file specifically for that chat session.
  - `model`: Optional string (e.g. `"qwen3.7-plus"` or `"deepseek-v4-vision"`).
  - `purpose`: Optional string (default: `"assistants"`).

#### Multipart Upload for Qwen (curl)
```bash
curl -X POST http://localhost:8000/v1/files \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@person.jpg" \
  -F "session_id=my-qwen-session" \
  -F "model=qwen3.7-plus"
```

#### Multipart Upload for DeepSeek (curl)
```bash
curl -X POST http://localhost:8000/v1/files \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@document.pdf" \
  -F "session_id=my-ds-session" \
  -F "model=deepseek-v4-flash"
```

#### JSON Base64 Upload
```bash
curl -X POST http://localhost:8000/v1/files \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "file": "'$(base64 -i document.pdf)'",
    "filename": "document.pdf",
    "session_id": "my-session"
  }'
```

#### Response Format
```json
{
  "id": "file-893049103940",
  "object": "file",
  "bytes": 245120,
  "created_at": 1726000000,
  "filename": "document.pdf",
  "purpose": "assistants",
  "session_id": "my-session",
  "status": "processed"
}
```

### `GET /v1/files/{file_id}`
Retrieve metadata and status for an uploaded file.

```bash
curl http://localhost:8000/v1/files/file-893049103940 \
  -H "Authorization: Bearer $API_KEY"
```

---

## 3. Session Management

Interact with server-side chat sessions on DeepSeek.

### `GET /v1/sessions`
List chat sessions across accounts.

- **Query Parameters**:
  - `account`: Optional integer account index (e.g. `0`). If omitted, returns sessions across all healthy accounts.
  - `pinned`: Optional boolean (default: `false`).
  - `count`: Optional integer (default: `20`, max: `100`).

```bash
curl "http://localhost:8000/v1/sessions?count=10" \
  -H "Authorization: Bearer $API_KEY"
```

#### Response Format
```json
{
  "object": "list",
  "data": [
    {
      "id": "sess-uuid-1",
      "title": "Project Planning",
      "account_index": 0,
      "created_at": 1726000000,
      "updated_at": 1726000000
    }
  ]
}
```

### `GET /v1/sessions/{session_id}`
Retrieve the full message history and conversation contents for a specific session.

```bash
curl http://localhost:8000/v1/sessions/sess-uuid-1 \
  -H "Authorization: Bearer $API_KEY"
```

#### Response Format
```json
{
  "id": "sess-uuid-1",
  "object": "chat.session",
  "account_index": 0,
  "messages": [
    {
      "message_id": 1,
      "role": "USER",
      "content": "What is Python?"
    },
    {
      "message_id": 2,
      "role": "ASSISTANT",
      "content": "Python is a high-level programming language..."
    }
  ]
}
```

### `DELETE /v1/sessions/{session_id}`
Permanently delete a chat session from DeepSeek and evict it from DanyAPI's local cache.

```bash
curl -X DELETE http://localhost:8000/v1/sessions/sess-uuid-1 \
  -H "Authorization: Bearer $API_KEY"
```

---

## 4. Models & Image Generation

### `GET /v1/models`
Lists all available models currently loaded and supported by the active token accounts.

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer $API_KEY"
```

### `POST /v1/images/generations`
Generate images using Qwen accounts.

```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "prompt": "A futuristic digital city at night",
    "size": "1024x1024"
  }'
```

---

## 5. Health, Stats & Administration

### `GET /health`
Returns system status, active account pool counts, health flags, and cumulative usage.

```bash
curl http://localhost:8000/health
```

### `GET /v1/usage`
Returns token usage statistics and request counters.

```bash
curl http://localhost:8000/v1/usage \
  -H "Authorization: Bearer $API_KEY"
```

### `POST /v1/tokens`
Dynamically add new DeepSeek or Qwen tokens to the running pool without restarting the server. Also appends them to `.env`.

```bash
curl -X POST http://localhost:8000/v1/tokens \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "tokens": ["your-deepseek-user-token-here"]
  }'
```
