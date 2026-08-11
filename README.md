# DanyAPI

OpenAI-совместимый HTTP API на Python + FastAPI, который вместо платного
DeepSeek API ходит во внутренний API бесплатного веб-клиента
[chat.deepseek.com](https://chat.deepseek.com) под одним серверным аккаунтом.
Пользователям API не нужны ключи - все запросы выполняет серверный аккаунт.

## Возможности

- `GET /v1/models` - список моделей
- `POST /v1/chat/completions` - генерация (stream и non-stream)
- Модели: `deepseek-chat`, `deepseek-reasoner`, `deepseek-vision`
  (внутренние `model_type`: `default`, `expert`, `vision`)
- Thinking (рассуждения R1) и web-поиск
- Многосессионность: цепочка сообщений хранится серверно
  (`session_id` в ответе), как в веб-клиенте
- Встроенный реверс PoW-хэша **DeepSeekHashV1** (23-раундовый Keccak
  с rate 136 и сдвинутыми round constants) + быстрый нативный солвер на C
  (`danyapi/deepseek/pow_solver.c`, компилируется clang)

## Установка

```bash
pip install -r requirements.txt
```

## Настройка аккаунта

Задайте пул токенов (с разных аккаунтов) через запятую:

```bash
export DEEPSEEK_TOKENS="token1,token2,token3"
```

Каждый аккаунт может генерировать **одно** сообщение одновременно, поэтому
пул из N токенов даёт до N параллельных генераций. Токен берётся в браузере:
DevTools -> Application -> Local Storage -> https://chat.deepseek.com -> `userToken`.

Либо одна учётка email + пароль (логин выполнится при старте):

```bash
export DEEPSEEK_EMAIL="you@example.com"
export DEEPSEEK_PASSWORD="secret"
```

## Деплой

### Hugging Face Spaces (бесплатно)

1. Заведи аккаунт на huggingface.co и создай **новый Space**:
   SDK выбери **Docker**, Hardware — любой, например "CPU basic / Free".
2. В настройках Space (Settings -> Variables and secrets) добавь секрет:
   `DEEPSEEK_TOKENS` = токены через запятую (без коммита в репозиторий).
3. Запушь код в Space (Space — это git-репозиторий):

```bash
git clone https://huggingface.co/spaces/<user>/<space-name>
cp -r danyapi Dockerfile requirements.txt .dockerignore <space>/
cd <space> && git add -A && git commit -m "DanyAPI" && git push
```

4. Образ соберётся автоматически и поднимет сервер на порту 7860.
   API будет доступен по адресу `https://<user>-<space-name>.hf.space`.

### VPS / Docker

```bash
docker build -t danyapi .
docker run -d -p 8000:7860 \
  -e DEEPSEEK_TOKENS="token1,token2" \
  danyapi
```

## Запуск локально

Файл `.env` (в гитигноре, создаётся из `.env.example`) подхватывается
автоматически при старте:

```bash
cp .env.example .env   # вписать токены
python -m danyapi
# или
uvicorn danyapi.api.openai:app --host 0.0.0.0 --port 8000
```

## Деплой на Android-сервере (Termux)

Для домашнего сервера на телефоне (Termux):

```bash
cd deploy
bash termux_setup.sh      # установка: python, clang, зависимости, сборка PoW-солвера
nano ~/DanyAPI/.env       # вписать DEEPSEEK_TOKENS
bash termux_service.sh    # автозапуск (termux-services) + cloudflared-туннель
```

После этого API доступен локально на `http://<ip-телефона>:8000`, а наружу
открывается публичным https-адресом от cloudflared (печатается при запуске).




## Использование

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-chat", "messages": [{"role": "user", "content": "Привет!"}]}'
```

Многоходовость: в ответе приходит `session_id`; передайте его в следующий
запрос, чтобы продолжить тот же диалог.

```json
{
  "model": "deepseek-reasoner",
  "messages": [{"role": "user", "content": "2+2?"}],
  "session_id": "<id из предыдущего ответа>",
  "thinking": true,
  "search": false,
  "stream": true
}
```

## Тесты

```bash
python -m unittest tests.test_pow tests.test_stream -v
```

## Как это работает

Реверс протокола выполнен по main-бандлу chat.deepseek.com
(`fe-static.deepseek.com/chat/static/main.4e922c397f.js`) и wasm-модулю
`sha3_wasm_bg.7b9ca65ddd.wasm`:

- Авторизация: `POST /api/v0/users/login` -> `data.biz_data.user.token`,
  дальше `Authorization: Bearer <token>`.
- Заголовки: `x-client-bundle-id`, `x-client-platform`, `x-client-version`,
  `x-client-locale`, `x-client-timezone-offset`.
- Сессия: `POST /api/v0/chat_session/create` (пустое тело) -> `chat_session.id`.
- Генерация: `POST /api/v0/chat/completion`:
  `{chat_session_id, parent_message_id, model_type, prompt, ref_file_ids,
  thinking_enabled, search_enabled, action, preempt}`.
- Ответ - `text/event-stream`: события `ready`, дельты
  (`SET`/`APPEND`/`BATCH`, пути `response/...`), `finish`, `close`.
- PoW-заголовок `X-DS-PoW-Response` - base64 от
  `{algorithm, challenge, salt, answer, signature, target_path}`.
  Чаллендж одноразовый: `answer` = минимальный counter c, при котором
  `DeepSeekHashV1(f"{salt}_{expire_at}_" + str(c))` совпадает с `challenge`
  (32 байта). Сервер перебирает c в диапазоне `[0, difficulty)`.

## Нативный PoW-солвер (необязательно)

Если есть компилятор C, соберите экзешник для максимальной скорости:

```bash
clang -O2 -o danyapi/deepseek/pow_solver.exe danyapi/deepseek/pow_solver.c
```

Без него сервер использует Node-солвер (wasm-модуль сайта) или
чисто-питоновский fallback. Все три варианта дают одинаковый ответ.

## Ограничения аккаунта

- Один аккаунт chat.deepseek.com может генерировать **одно сообщение
  одновременно** (иначе сервер отвечает `parallel_chat_limit`). DanyAPI
  держит **пул аккаунтов** и распределяет конкурентные запросы между ними;
  если все заняты - запросы ждут в очереди. Больше токенов = больше
  параллельных генераций.
- Сессии привязаны к аккаунту, на котором созданы: повторные запросы с тем же
  `session_id` маршрутизируются на тот же аккаунт (история диалога хранится
  серверно на аккаунте).
- DeepSeek может временно троттлить аккаунты (особенно экспертную модель
  `deepseek-reasoner` - "limited resource"). Ответы с `finish_reason`
  `expert_busy_use_default` / `parallel_chat_limit` автоматически ретраются
  (до 3 попыток с экспоненциальным backoff). Если заняты все попытки:
  - non-stream запрос получает HTTP 429 с текстом ошибки DeepSeek;
  - stream запрос получает SSE-событие с `error` и `finish_reason`.
- Чаллендж PoW одноразовый - на каждый запрос решается новый (префетчится
  следующий заранее, чтобы не ждать).
