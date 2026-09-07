# DANYAPI DOCS
--------------
The DanyAPI provides a server-side dispatcher to communicate with Deepseek/Qwen.

# CONFIGURATION
The config has 3 main areas: tokens, connections and settings (session, logging).

### Tokens
Edit the *.env* file with the tokens.

Start up Deepseek or Qwen and login. Then open the Developer-Debugger and look at the network requests. Look for a request with 'completion' in it. Click on that and look for the Authorization: Bearer XXXXXXXXXXX.

That sequence of characters is what you need to put into the DEEPSEEK/QWEN tokens. It will send the requests like your user-id based on that token. Don't let other people use this API as it may get your user banned. Treat this system as a privilege to use.

### Connections
By default it binds to all IPs on the system at port 8000. If you want to run it through a proxy (ex. to make it appear to have a residential IP), you would set the proxy in DANYAPI_PROXY.

If you want to force incoming requests to you to have a Bearer-token, you can set that in DANYAPI_API_KEY. If it's not set, all requests are accepted (only run it locally then!)

You can customize the upstream User-Agent header sent to DeepSeek and Qwen by setting DANYAPI_USERAGENT (defaults to a modern Chrome browser string).

### Settings
The 'session' is like a chat conversation. So each call you make can be grouped together in the same session (which saves lots of context tokens, as it remembers prior conversation). You can set the session_id to whatever, ex a name, etc. It saves 128 by default for 7 days, but examine the .env.example and change.

### Logging & more
By default it logs everything to the docker log, or stdout but you can choose what to log, or even if a file.

You can log tokens used, and auto-update.


# RUNNING
You can run it on your system, or more easily you can run it from docker:

```
docker compose up
```

It will show log output like:
 ✔ Image danyapi-danyapi    Built     149.6s
 ✔ Container danyapi        Recreated 0.5s
Attaching to danyapi

danyapi  | (21:37:33) outgoing IP: 99.88.77.66 (via proxy socks5://host.docker.internal:1080)
danyapi  | (21:37:33) authentication: open (no API_KEY set)
danyapi  | (21:37:35) deepseek accounts ready: 1 (deepseek-v4-flash, deepseek-v4-pro, deepseek-v4-vision)
danyapi  | (21:37:35) default model: deepseek-v4-flash
danyapi  | (21:37:35) DanyAPI running on http://0.0.0.0:8000 (Press CTRL+C to quit)
danyapi  | (21:42:35) deepseek create session success (1520ms)
danyapi  | (21:42:49) deepseek completion success (14234ms)
danyapi  | (21:42:49) POST /v1/chat/completions success (17800ms)
danyapi  | (21:59:23) deepseek completion success (2910ms)
danyapi  | (21:59:23) POST /v1/chat/completions success (3650ms)


# ENDPOINTS

### LLM Endpoints (OpenAI-Compatible)
- POST **/v1/chat/completions** — Chat, reasoning (thinking), search, tools, sessions (session_id), file attachments, and streaming.
    * messages (array, required): List of message objects (role: system | user | assistant, content: string).
    * optional: model (string), stream (boolean, default: false), thinking (boolean), search (boolean), session_id (string): continue conversation, tools (array), file_ids (array): explicit file IDs to attach. Uploaded files for this session are also automatically attached.
- POST **/v1/images/generations** — Text-to-image generation powered by Qwen.
    * prompt (string, required): Text description of the image to generate.
    * optional: model (string, default: qwen-image-gen), n (integer, default: 1), size (string, default: 1024x1024), response_format (string, default: url).

- _Note: If Bearer Auth given, that will be required in calls_.

### File & Image Uploads (DeepSeek)
- POST **/v1/files** — Upload files or images to DeepSeek with dynamic Proof-of-Work (PoW) challenge solving. Files are streamed in-memory directly to DeepSeek (no disk storage) and automatically staged to attach to your next chat completion.
    * file (binary / multipart, or base64 JSON string, required)
    * optional: session_id (string): associates the file with a session and pins to that account, purpose (string, default: assistants), model (string).
- GET **/v1/files/{file_id}** — Retrieve status and metadata for an uploaded file.

### Session Management (DeepSeek)
- GET **/v1/sessions** — List active chat sessions stored on DeepSeek across accounts.
    * optional: account (integer): filter by account index, pinned (boolean, default: false), count (integer, default: 20, max: 100).
- GET **/v1/sessions/{session_id}** — Retrieve full conversation history and messages for a session.
- DELETE **/v1/sessions/{session_id}** — Delete a chat session from DeepSeek and evict from cache.

### Models & Token Management
- GET **/v1/models** — List of all available DeepSeek and Qwen models.
- GET **/v1/usage** — Real-time request and token consumption statistics.
- POST **/v1/tokens** — Hot-add new tokens without restarting the server.

### Health & Diagnostics
- GET **/health** — Provider readiness, active accounts, and cache metrics.

### Web UI & Documentation
- GET **/** — Interactive token management and usage dashboard.
- GET **/docs/** — Landing page, playground, and guides.
- GET **/openapi.json** & **GET /redoc** — OpenAPI specs and ReDoc interactive viewer.


# EXAMPLES
See the models available:
```
curl -s http://localhost:8000/v1/models
```

By default the first model with a valid token is selected if none specified (so if deepseek token available, deepseek-v4-flash is selected).

We will use curl to show the same prompts:
```
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant that summarizes news articles."},
      {"role": "user", "content": "Please summarize the results of this poll: https://slashdot.org/poll/3284/how-much-of-your-coding-is-done-by-ai-coding-agents-these-days"}
    ],
    "session_id": "george"}'
```

Then follow-up chats woulds keep the same session_id, ex.
```
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant that summarizes news articles."},
      {"role": "user", "content": "What was the top pick?"}
    ],
    "session_id": "george"}'
```

### Image Generation (Qwen)

If you have a Qwen token in this, you can generate images:
```
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Futuristic skyline at dusk, cyberpunk style, digital art",
    "size": "1024x1024"
  }'
```  

### Image Uploads (Deepseek)
Uploading with CURL in this example:

```
curl -X POST http://localhost:8000/v1/files \
  -F "file=@/Users/george/Downloads/person.jpg" \
  -F "session_id=george"
```

Then querying it (need deepseek-v4-vision):
```
curl -s -m 120 http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"deepseek-v4-vision","messages":[{"role":"user","content":"Describe the attached image."}],"session_id":"george"}' 
```

### Usage Tracking
You can use the webpage (if it's enabled to see from a browser), or:

```
curl -s http://localhost:8000/v1/usage
```

### File & Image Uploads (with Auto-Attachment)

You can upload files or images directly to DeepSeek. Files are streamed in-memory (no local disk storage) and automatically solved with cryptographic Proof of Work (PoW).

**1. Upload via multipart form (associating with session "george"):**
```
curl -s -X POST http://localhost:8000/v1/files \
  -F "file=@annual_report.pdf" \
  -F "session_id=george"
```

**2. Or upload via JSON with base64 data:**
```
curl -s -X POST http://localhost:8000/v1/files \
  -H "Content-Type: application/json" \
  -d '{
    "file": "SGVsbG8gV29ybGQ=",
    "filename": "notes.txt",
    "session_id": "george"
  }'
```

**3. Automatic Attachment in Chat:**
Now when you send a prompt with `"session_id": "george"`, the uploaded file(s) are **automatically attached** to the model prompt:
```
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "george",
    "messages": [
      {"role": "user", "content": "Please summarize the file I just uploaded."}
    ]
  }'
```

**4. Explicit Attachment by File ID:**
You can also re-use previously uploaded files by passing their file IDs explicitly:
```
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "george",
    "file_ids": ["file-xxxxxxxx"],
    "messages": [
      {"role": "user", "content": "What are the main findings in this file?"}
    ]
  }'
```

**5. Inspect File Status:**
```
curl -s http://localhost:8000/v1/files/file-xxxxxxxx
```

### Session Management

Inspect, retrieve message history, or delete server-side sessions on DeepSeek:

**List active sessions:**
```
curl -s "http://localhost:8000/v1/sessions?count=10"
```

**Get session details and message contents:**
```
curl -s http://localhost:8000/v1/sessions/george
```

**Delete a session:**
```
curl -s -X DELETE http://localhost:8000/v1/sessions/george
```
