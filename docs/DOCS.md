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
danyapi  | (21:37:35) deepseek accounts ready: 1
danyapi  | (21:37:35) default model: deepseek-v4-flash
danyapi  | (21:37:35) DanyAPI running on http://0.0.0.0:8000 (Press CTRL+C to quit)
danyapi  | (21:42:35) deepseek create session success (1520ms)
danyapi  | (21:42:49) deepseek completion success (14234ms)
danyapi  | (21:42:49) POST /v1/chat/completions success (17800ms)
danyapi  | (21:59:23) deepseek completion success (2910ms)
danyapi  | (21:59:23) POST /v1/chat/completions success (3650ms)


# ENDPOINTS

### LLM Endpoints (OpenAI-Compatible)
- POST **/v1/chat/completions** — Chat, reasoning (thinking), search, tools, sessions (session_id), and streaming.
    * messages (array, required): List of message objects (role: system | user | assistant, content: string).
    * optional: model (string), stream (boolean, default: false), thinking (boolean), search (boolean), session_id (string): continue conversation, tools (array).
- POST **/v1/images/generations** — Text-to-image generation powered by Qwen.
    * prompt (string, required): Text description of the image to generate.
    * optional: model (string, default: qwen-image-gen), n (integer, default: 1), size (string, default: 1024x1024), response_format (string, default: url).

- _Note: If Bearer Auth given, that will be required in calls_.

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

### Usage Tracking
You can use the webpage (if it's enabled to see from a browser), or:

```
curl -s http://localhost:8000/v1/usage
```
