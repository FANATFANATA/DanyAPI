# Рефакторинг DanyAPI — статус работы

## Контекст

Цель: рефакторинг **только оригинального кода** (FANATFANATA/DanyAPI) — разбивка огромных файлов на модули. Наши собственные фишки сохранены отдельно, чтобы потом применить их к уже переделанному проекту.

- Ветка `refactor` — чистый `origin/main` + рефакторинговые коммиты (текущая ветка).
- Ветка `features-ours` (3401f69) — 5 наших коммитов с «фишками». Чтобы применить их к переделанному проекту: `git rebase features-ours` поверх или cherry-pick 5 коммитов. Патчи, которые касались `danyapi/tools.py`, будут конфликтовать с новым пакетом `tools/` — придётся адаптировать под `prompt.py` / `parse.py`.
- Тесты: `DANYAPI_LOG_LEVEL= .venv/bin/python -m pytest tests/ -q` (переменная обязательна, иначе `test_logging` падает из-за DEBUG в локальном `.env`).

## Что уже сделано

### 1. tools.py → пакет tools/ — ГОТОВО, закоммичено (3fc5465)
Было один файл `danyapi/tools.py` (~700 строк). Теперь:
- `danyapi/tools/prompt.py` — ToolCall, константы инструкций, build_prompt, render_*, context_sequence.
- `danyapi/tools/parse.py` — DSML-регулярки, `_strip_dsml`, все JSON/XML парсеры tool-calls.
- `danyapi/tools/format.py` — format_tool_message, tool_call_deltas.
- `danyapi/tools/__init__.py` реэкспортирует ВСЕ имена (включая `_`-приватные), чтобы старые импорты работали без изменений.

Проверено: 639 passed + ruff чисто.

### 2. api/openai.py → модули — В РАБОТЕ, НЕ закоммичено
Было один файл `danyapi/api/openai.py` (1407 строк). Созданы новые файлы:
- `danyapi/api/models.py` (102 строки) — pydantic-модели ChatMessage/FileSpec/ChatCompletionRequest, константы моделей (MODEL_TYPE_BY_NAME, QWEN_DEFAULT_MODELS, STATUS_TO_FINISH_REASON и т.д.), `_resolve_model`, `_finish_reason`, `_deepseek_usage`, `_include_usage`.
- `danyapi/api/attachments.py` (108 строк) — Attachment, MAX_FILES_PER_REQUEST/MAX_FILE_SIZE, `_split_data_uri`, `_collect_attachments`, `_validate_attachments`, `_fresh_pow_upload_headers`, `_upload_attachments`.
- `danyapi/api/deepseek.py` (195 строк) — retry-константы (RETRYABLE_FINISH_REASONS, MAX_RETRIES=5, RETRY_BACKOFF_SEC=1.0, RETRY_BACKOFF_MAX_SEC=8.0, DEEPSEEK_AUTH_ERROR_CODES), `_human_delay`, `_prepare_session`, `_send_completion`, все `_is_retryable_*`/`_input_exceeds_hint_from_http`/`_drop_session`/`_deepseek_status`/`_send_with_auth`/`_fresh_pow_headers`/`_busy_error_body`/`_try_stop_stream`, `_collect_continuation`.
- `danyapi/api/streaming.py` (533 строки) — `_sse`, `_stream_guard`, `_chat_completions_deepseek`, `_chat_completions_qwen`, `_collect_non_stream`, `_stream_openai`.

В самом `openai.py` остаются: lifespan, app, health, list_models, chat_completions, `_acquire_account`, `_resolve_provider` + **re-export блок** всех `_`-имён (тесты используют `openai_mod._X`).

## Что осталось доделать

### А) ДОВЕРШИТЬ расщепление openai.py (срочно, в процессе)
Текущий файл `danyapi/api/openai.py` на диске — битая промежуточная версия (319 строк): при импорте падает `IndentationError: line 25`. Нужно:

1. Переписать ЧИСТО через Write-тул новый `openai.py`:
   - header с импортами (`json`, `logging`, `time`, `uuid`, `asynccontextmanager`, `Any`, FastAPI, HTTPException, StreamingResponse, `toolemu` из tools, AccountPool/AccountPoolBusy/DeepSeekAccount, settings, DeepSeekClient, qwen_api, QwenAccount, QwenClient/QwenError, JsonStore).
   - тело: lifespan (107–200), `_fetch_qwen_models` (201–236) — из оригинала `git show HEAD:danyapi/api/openai.py`.
   - строка `app = FastAPI(title="DanyAPI", lifespan=lifespan)` (оригинальная 238).
   - `_pool_stats`, health, list_models (345–406).
   - `_resolve_provider`, chat_completions, `_acquire_account` (413–437).
   - `_sse` + `_stream_guard` (448–479).
   - `_chat_completions_deepseek` + `_chat_completions_qwen` (480–593).
   - re-export блок в конце.

2. Проверка импорта: `.venv/bin/python -c "import danyapi.api.openai as m; print(hasattr(m,'_collect_non_stream'), hasattr(m,'Attachment'), hasattr(m,'app'))"`.

3. Тесты: `DANYAPI_LOG_LEVEL= .venv/bin/python -m pytest tests/test_api_helpers.py tests/test_retry.py tests/test_tools_api.py tests/test_qwen_api.py -q`.
   Критично: тесты патчат `openai_mod.RETRY_BACKOFF_SEC` и `openai_mod.MAX_RETRIES` — значит deepseek.py/streaming.py должны импортировать эти имена ОТДЕЛЬНО (`from .models import MAX_RETRIES, ...`), не через `import *`, чтобы monkeypatch атрибута модуля работал.

4. Коммит: «refactor(api): split openai.py into models/attachments/deepseek/streaming modules».

### Б) Убрать дублирование retry/error-логики (фаза 2b) — ГОТОВО, закоммичено (c822c42)
Создан `danyapi/retry.py` с общими константами: MAX_RETRIES, RETRY_BACKOFF_SEC/MAX_SEC, RETRYABLE_HTTP_STATUSES, `_is_retryable_http`, `_drop_session`. Оба потребителя (`deepseek.py` и `qwen/api.py`) импортируют их как локальные имена — monkeypatch в тестах работает.

### В) Финальная проверка
- Весь pytest: `DANYAPI_LOG_LEVEL= .venv/bin/python -m pytest tests/ -q`.
- Ruff: `.venv/bin/ruff check danyapi/ tests/`.
- Публичный API без изменений (все имена, которые импортируют тесты и внешние потребители, доступны из `danyapi.api.openai`).

### Г) Чистка черновиков
В `.tmp/` лежат черновики (`part_parse_head.txt`, `exact.txt`, `head_section.txt`, `mid_section.txt`, `routes*.txt`) — НЕ коммитить. В конце удалить или оставить в gitignore (`.tmp/` уже игнорируется).

## Точные границы секций оригинала openai.py (для отсылок)
- imports 1–29; константы моделей 31–71; pydantic ChatMessage/FileSpec/ChatCompletionRequest 74–105.
- lifespan 107–200; _fetch_qwen_models 201–236; `app = FastAPI(...)` строка 238 (проверить!).
- MAX_FILES_PER_REQUEST=50/MAX_FILE_SIZE + Attachment+_split_data_uri+_collect_attachments+_validate_attachments+_fresh_pow_upload_headers+_upload_attachments — 240–331.
- _resolve_model 332–337; _finish_reason 339–343; _pool_stats 345–353.
- health (маршрут) ~368–380; list_models ~383–406.
- RETRYABLE_FINISH_REASONS..DEEPSEEK_AUTH_ERROR_CODES + MAX_RETRIES/RETRY_BACKOFF_SEC/MAX — 407–427.
- _human_delay ~429; _resolve_provider ~436; @app.post chat_completions ~455; _acquire_account ~460.
- _include_usage/_deepseek_usage/_sse/_stream_guard ~480–510.
- _chat_completions_deepseek 512–575; _chat_completions_qwen 577–630.
- _prepare_session 632+; _send_completion ~655; retry/error-функции 710–800; _busy_error_body/_try_stop_stream ~800–815.
- _collect_continuation 818–870; _collect_non_stream 872–1012; _stream_openai 1014–конце (1407).
