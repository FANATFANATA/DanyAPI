(function () {
    var KEY = 'danyapi-lang';
    var SUPPORTED = ['en', 'ru'];

    var RU = {
        theme_label: 'Тема',
        theme_group: 'Тема',
        lang_group: 'Язык',
        sidebar_label: 'Разделы',
        toc_title: 'На этой странице',
        copy: 'Копировать',
        copied: 'Скопировано!',
        download_env: 'Скачать шаблон .env',
        footer_note: 'DanyAPI · на FastAPI и Python',
        nav_setup: 'Старт',
        nav_overview: 'Обзор',
        nav_install: 'Установка и аккаунты',
        nav_config: 'Конфигурация',
        nav_usage: 'Использование',
        nav_internals: 'Как это работает',
        title_index: 'DanyAPI · установка',
        title_overview: 'Обзор · DanyAPI',
        title_setup: 'Установка и аккаунты · DanyAPI',
        title_config: 'Конфигурация · DanyAPI',
        title_usage: 'Использование · DanyAPI',
        title_internals: 'Как это работает · DanyAPI',
        idx_subtitle: 'Установка одной командой.',
        idx_intro: 'OpenAI-совместимый HTTP API на Python + FastAPI. Одна команда скачает его, поставит зависимости, настроит <code>.env</code>, проверит токены у провайдеров и создаст на рабочем столе ярлык <code>DanyAPI</code>. Сервер сам обновляется до последнего GitHub-релиза при каждом запуске.',
        idx_cmd_label: 'Открой терминал и вставь это:',
        idx_cmd_hint: 'Или выбери свою платформу:',
        idx_creator: '<a href="https://t.me/DanyaVoredom" target="_blank">Создатель</a> · <a href="https://t.me/DanyAPIFree" target="_blank">Телеграм-канал</a>',
        idx_term_note: 'Это выглядит так: скрипт выполнит всё это в твоём терминале. Здесь просто наглядная демонстрация.',
        idx_manual: 'Предпочитаешь ручную установку? Открой <a href="setup.html#setup-script">пошаговую инструкцию</a>.',
        idx_after: 'После установки',
        idx_after_text: 'Запусти сервер:',
        idx_run_alt: 'Или вспомогательными скриптами:',
        idx_download_note: 'Не хочешь заполнять настройки в терминале? Скачай шаблон и отредактируй его вручную.',
        idx_links_more: 'См. <a href="setup.html#account-setup">Настройку аккаунтов</a> и <a href="configuration.html">все переменные</a>.',
        ov_subtitle: 'Что умеет DanyAPI.',
        ov_p1: 'Вместо платных API он работает с внутренними API бесплатных веб-клиентов <a href="https://chat.deepseek.com" target="_blank">chat.deepseek.com</a> и <a href="https://chat.qwen.ai" target="_blank">chat.qwen.ai</a> через серверные аккаунты. Пользователям API не нужны ключи - все запросы делают серверные аккаунты.',
        ov_features: 'Возможности',
        ov_f1: '<code>GET /v1/models</code> - список моделей',
        ov_f2: '<code>POST /v1/chat/completions</code> - генерация (stream и non-stream)',
        ov_f3: 'Модели DeepSeek: <span class="model">deepseek-v4-flash</span> (по умолчанию), <span class="model">deepseek-v4-pro</span> (экспертная), <span class="model">deepseek-v4-vision</span> (vision). Внутренний <code>model_type</code>: <code>default</code>, <code>expert</code>, <code>vision</code>.',
        ov_f4: 'Размышления доступны для всех моделей DeepSeek; веб-поиск работает только у <span class="model">deepseek-v4-flash</span>.',
        ov_f5: 'Вложения: <span class="model">deepseek-v4-vision</span> принимает только изображения; <span class="model">deepseek-v4-flash</span> - изображения (OCR) и текстовые файлы; <span class="model">deepseek-v4-pro</span> - ничего. За запрос: максимум 50 файлов по 100 МБ.',
        ov_f6: 'Модели Qwen: подтягиваются из аккаунта при старте (<span class="model">qwen3.8-max</span>, <span class="model">qwen3.7-plus</span>, …)',
        ov_f7: 'Размышления и веб-поиск (DeepSeek); размышления и поиск (Qwen). Трассы размышлений отдаются как <code>reasoning_content</code> (стримятся вживую, оба провайдера)',
        ov_f8: 'Вызов инструментов (эмуляция): <code>tools</code> / <code>tool_choice</code> / <code>parallel_tool_calls</code> с корректными ответами <code>finish_reason: "tool_calls"</code>',
        ov_f9: 'JSON-режим (эмуляция): <code>response_format</code> с <code>json_object</code> / <code>json_schema</code>',
        ov_f10: 'Сообщения <code>system</code> внедряются как системный промпт модели',
        ov_f11: 'Реальный расход токенов в ответах, накапливается за диалог как в официальном API; стриминговый расход через <code>stream_options.include_usage</code>',
        ov_f12: 'Дисковый кэш сессий: диалоги переживают перезапуск сервера',
        ov_f13: '<code>GET /health</code> - readiness-проба',
        ov_f14: 'Мультисессионность: цепочка сообщений хранится на сервере (<code>session_id</code> в ответе), как в веб-клиентах. Stateless-запросы (без <code>session_id</code>) автоматически переиспользуют тот же серверный чат по контексту сообщений, поэтому обычные OpenAI-клиенты сохраняют диалог.',
        ov_usage: 'Быстрый старт',
        ov_usage_text: 'Использование через OpenAI SDK (прямая замена официального API):',
        ov_usage_more: 'Многопоточные диалоги, сессии, поля запросов, вложения, вызов инструментов, JSON-режим и обработка ошибок описаны на странице <a href="usage.html">Использование</a>.',
        ov_tests: 'Тесты',
        ov_lint: 'Линт и форматирование (ruff):',
        ov_ci: 'Тесты, линт и сборка нативного солвера также запускаются в CI при каждом пуше (<code>.github/workflows/ci.yml</code>).',
        st_subtitle: 'Установка, настройка аккаунтов и запуск DanyAPI.',
        pl_script: 'Скрипт установки',
        pl_install: 'Установка',
        pl_accounts: 'Настройка аккаунтов',
        pl_run: 'Запуск',
        pl_logging: 'Логирование',
        st_script_p1: 'Установка одной командой: <code>irm https://fanatfanata.github.io/DanyAPI/install.ps1 | iex</code> в PowerShell или <code>curl -fsSL https://fanatfanata.github.io/DanyAPI/install.sh | bash</code> на Linux/macOS - команда скачает репозиторий и запустит <code>docs/setup.py</code>. Скрипт ставит зависимости, создаёт <code>.env</code> из <code>.env.example</code>, если его нет, спрашивает учётные данные провайдеров и настройки сервера, живым запросом проверяет токены, записывает всё в <code>.env</code> и создаёт на рабочем столе ярлык <code>DanyAPI</code>. Сервер сам обновляется до последнего релиза при каждом запуске. Нажимай <code>Enter</code>, чтобы оставить текущее значение, или введи <code>!clear</code>, чтобы стереть его.',
        st_script_p2: 'Требуется Python 3.13+. Пароли маскируются при вводе и пишутся в <code>.env</code>, который в gitignore и никогда не попадает в репозиторий.',
        st_install_p: 'Требуется Python 3.13+.',
        st_dev: 'Для разработки (тесты + линт) установи dev-зависимости:',
        st_ds_h: 'DeepSeek',
        st_ds_p1: 'Задай пул токенов (с разных аккаунтов) через запятую или один токен через <span class="env-var">DEEPSEEK_TOKEN</span>:',
        st_ds_p2: 'Каждый аккаунт может генерировать <strong>одно</strong> сообщение одновременно, поэтому пул из N токенов даёт до N параллельных генераций. Возьми токен в браузере: DevTools → Application → Local Storage → <a href="https://chat.deepseek.com" target="_blank">https://chat.deepseek.com</a> → <code>userToken</code>.',
        st_ds_p3: 'Или один аккаунт «email + пароль» (вход происходит при старте):',
        st_qw_h: 'Qwen',
        st_qw_p1: 'Та же модель, другое место токена (одиночный токен через <span class="env-var">QWEN_TOKEN</span>):',
        st_qw_p2: 'Возьми токен в браузере: DevTools → Application → Local Storage → <a href="https://chat.qwen.ai" target="_blank">https://chat.qwen.ai</a> → <code>token</code>.',
        st_qw_blockquote: '<strong>Для аккаунтов Qwen, используемых DanyAPI, отключи встроенный переключатель <em>Tools</em></strong> (интерпретатор кода, генерация изображений и другие встроенные инструменты Qwen) в веб-интерфейсе chat.qwen.ai. Пока он включён, встроенные инструменты Qwen запускают собственные фазы ответа на каждый запрос, которые DanyAPI не может распарсить, а серверная история диалога быстро разрастается (и упирается в лимит входных токенов). При выключенном переключателе эмулируемый вызов инструментов (<code>tools</code> / <code>tool_choice</code>) и обычный чат работают как надо.',
        st_qw_p3: 'Или один аккаунт «email + пароль» (вход происходит при старте):',
        st_both: 'Оба провайдера опциональны. Запусти хотя бы один (или оба) - запросы маршрутизируются на нужного провайдера по имени модели (<code>deepseek-*</code> / <code>qwen*</code>).',
        st_valid: 'При старте каждый токен проверяется у своего провайдера; невалидные или истёкшие токены пропускаются с предупреждением (сервер отказывается запускаться, если не осталось ни одного валидного токена). Список моделей Qwen для <code>/v1/models</code> забирается с первого Qwen-аккаунта при старте (только text-chat модели); если получить его не удалось, используется встроенный список по умолчанию. Новые запросы без <code>session_id</code> распределяются по здоровым аккаунтам round-robin.',
        st_run_p1: 'Файл <code>.env</code> (в gitignore, создаётся из <code>.env.example</code>) подхватывается автоматически при старте:',
        st_run_p2: 'Или вспомогательными скриптами:',
        st_log_p1: 'По умолчанию логи идут в консоль. Чтобы сохранять их в файл, задай <span class="env-var">DANYAPI_LOG_FILE</span> в <code>.env</code> (или как переменную окружения):',
        st_log_p2: 'Файл ротируется по размеру (<span class="env-var">DANYAPI_LOG_MAX_BYTES</span>, по умолчанию 10 МБ), сохраняя <span class="env-var">DANYAPI_LOG_BACKUP_COUNT</span> копий (по умолчанию 3). При запуске через <code>python -m danyapi</code> собственные логи старта/доступа uvicorn идут через тот же корневой логгер и тоже попадают в файл.',
        cf_subtitle: 'Переменные окружения, Docker и опциональный нативный PoW-солвер.',
        pl_env: 'Переменные окружения',
        pl_docker: 'Docker',
        pl_pow: 'Нативный PoW-солвер',
        cf_env_intro: 'Вся конфигурация управляется окружением (<code>.env</code> или экспортированные переменные). Учётные данные провайдеров описаны в разделе <a href="setup.html#account-setup">Настройка аккаунтов</a>.',
        cf_th_var: 'Переменная',
        cf_th_default: 'По умолчанию',
        cf_th_desc: 'Описание',
        cf_r1_desc: 'Адрес, на котором слушает API-сервер',
        cf_r2_desc: 'Порт, на котором слушает API-сервер',
        cf_r3_desc: 'Таймаут запроса к апстриму в секундах',
        cf_r4_desc: 'Секунды ожидания свободного аккаунта до 429; пусто = ждать вечно',
        cf_r5_desc: 'Максимум серверных чатов в кэше на провайдера (LRU) для stateless-переиспользования сессий',
        cf_r6_desc: 'Секунды, в течение которых неиспользуемая сессия/контекст остаётся переиспользуемой; <code>0</code> = никогда не истекает',
        cf_r7_desc: 'Директория дискового кэша сессий; пусто = системная временная (<code>%TEMP%\\danyapi</code> / <code>/tmp/danyapi</code>)',
        cf_r8_desc: 'Задай <code>1</code>/<code>true</code>/<code>yes</code>/<code>on</code>, чтобы полностью отключить дисковый кэш',
        cf_r9_desc: 'Уровень логов: <code>DEBUG</code>, <code>INFO</code>, <code>WARNING</code>, <code>ERROR</code>',
        cf_r10_desc: 'Путь к файлу логов; пусто = только консоль',
        cf_r11_desc: 'Максимальный размер файла логов в байтах до ротации',
        cf_r12_desc: 'Сколько ротированных файлов логов хранить',
        cf_r13_desc: 'Минимальная задержка в секундах перед отправкой запроса (равномерный случайный джиттер)',
        cf_r14_desc: 'Максимальная задержка в секундах; задай оба в <code>0</code>, чтобы отключить',
        cf_r15_desc: 'Автообновление до последнего GitHub-релиза при каждом запуске; <code>0</code> отключает',
        cf_docker_p1: 'Файл <code>.env</code> не запекается в образ; передавай учётные данные переменными окружения или примонтируй свой <code>.env</code> как том (<code>-v /path/to/.env:/app/.env</code>).',
        cf_docker_h3: 'Готовый образ (GHCR)',
        cf_docker_p2: 'CI собирает и пушит образ в GitHub Container Registry при каждом пуше в <code>main</code> (теги <code>latest</code> и <code>sha-*</code>) и на тегах версий (<code>v1.2.3</code>):',
        cf_pow_p1: 'Если есть C-компилятор (<code>clang</code> или <code>gcc</code>), собери бинарь для максимальной скорости:',
        cf_pow_p2: 'Готовый Docker-образ компилирует этот бинарь во время сборки, поэтому пользователи контейнера получают нативный солвер из коробки.',
        cf_pow_p3: 'Солверы перебираются по порядку, пока один не сработает: нативный бинарь → Node-солвер (wasm-модуль сайта) → чистый Python-фолбэк на Keccak (ограничен 2 млн итераций, поэтому тянет только низкие сложности). Каждый решённый заголовок одноразовый; следующий вызов префетчится в фоне, так что следующий запрос не ждёт.',
        ug_subtitle: 'Чат, сессии, вложения, вызов инструментов, JSON-режим и ошибки.',
        pl_ug_basic: 'Базовое использование',
        pl_ug_sessions: 'Сессии',
        pl_ug_fields: 'Поля запросов',
        pl_ug_persist: 'Персистентность сессий',
        pl_ug_files: 'Вложения файлов',
        pl_ug_tools: 'Вызов инструментов',
        pl_ug_json: 'JSON-режим',
        pl_ug_health: 'Системный промпт и health',
        pl_ug_tokens: 'Расход токенов',
        pl_ug_errors: 'Обработка ошибок',
        ug_h_basic: 'Базовое использование',
        ug_sdk_p: 'Использование через OpenAI SDK (прямая замена официального API):',
        ug_qwen_p: 'Или с Qwen-моделью:',
        ug_h_sessions: 'Сессии',
        ug_s1: 'Многопоточность: ответ содержит <code>session_id</code>; передай его в следующем запросе, чтобы продолжить тот же диалог.',
        ug_s2: 'Stateless-клиенты (без <code>session_id</code>) тоже не теряют контекст: сервер вычисляет отпечаток из сообщений <code>system</code>/<code>user</code> и переиспользует подходящий серверный чат. Если контекст сообщений идентичен - используется тот же чат; если это продолжение ранее виденного - чат продолжается, поэтому многопоточные диалоги работают, даже когда клиент никогда не присылает <code>session_id</code>. Передай <code>session_id</code>, чтобы зафиксировать точный диалог (или ответвиться в независимый чат). В памяти кэш ограничен LRU на провайдера (<span class="env-var">DANYAPI_SESSION_CACHE_SIZE</span>, по умолчанию 128).',
        ug_s3: 'Отпечаток контекста изолирован по полю <code>user</code>, когда клиент его шлёт: два разных <code>user</code> никогда не делят серверный чат даже при одинаковых сообщениях, поэтому stateless-мультитенантные клиенты остаются изолированными. Записи кэша истекают через <span class="env-var">DANYAPI_SESSION_TTL_SECONDS</span> (по умолчанию 3600, <code>0</code> отключает истечение), так что устаревшие чаты выкидываются вместо переиспользования. Qwen-чаты также учитывают модель: <code>session_id</code>, созданный для одной Qwen-модели, автоматически переносится в новый чат, если запрос переключил модель.',
        ug_no_context: 'Никакой контекст никогда не дублируется и не теряется:',
        ug_s_b1: 'Переиспользуемый чат получает <strong>только дельту</strong>: новое сообщение пользователя или хвост раунда инструментов (результаты инструментов) при вызове. Полная история диалога живёт на сервере в чате.',
        ug_s_b2: 'Схема инструментов / системный промпт внедряются <strong>один раз</strong>, в первое сообщение чата, и не повторяются в каждом последующем.',
        ug_s_b3: 'Если чат не удалось сопоставить (промах кэша или вытеснение), вся история сообщений переигрывается в новый чат, так что модель всегда видит полный диалог.',
        ug_h_fields: 'Поля запросов',
        ug_fields_p: '<code>POST /v1/chat/completions</code> принимает:',
        ug_th_field: 'Поле',
        ug_th_default: 'По умолчанию',
        ug_th_notes: 'Примечания',
        ug_r1_note: '<code>deepseek-*</code> маршрутизируется на DeepSeek, <code>qwen*</code> - на Qwen; всё остальное даёт HTTP 404',
        ug_r2_note: 'Формат OpenAI; <code>content</code> может быть строкой или списком частей <code>text</code>/<code>image_url</code>',
        ug_r3_note: 'SSE-стрим (<code>data:</code> чанки + <code>data: [DONE]</code>)',
        ug_r4_note: 'DeepSeek: выключен по умолчанию, включён для <code>deepseek-v4-pro</code>; Qwen: включён по умолчанию',
        ug_r5_note: 'Веб-поиск; DeepSeek учитывает его только для <code>deepseek-v4-flash</code>',
        ug_r6_note: 'Продолжить точный серверный диалог',
        ug_r7_note: 'Изолирует stateless-отпечаток контекста (мультитенантная изоляция)',
        ug_r8_note: 'Вложения DeepSeek: <code>{name, content (base64), content_type}</code>',
        ug_r9_note: 'Эмулируемый вызов инструментов (см. ниже)',
        ug_r10_note: 'Эмулируемый JSON-режим (см. ниже)',
        ug_r11_note: '<code>{"include_usage": true}</code> добавляет <code>usage</code> в финальный SSE-чанк',
        ug_r12_note: 'Принимаются для совместимости, но игнорируются: у веб-API апстрима нет параметров сэмплинга',
        ug_fields_note: 'Ответ дополнительно несёт <code>reasoning_content</code> (трассу размышлений) в <code>message</code> (не-stream) или <code>delta</code> (stream), когда включены размышления, а также <code>session_id</code> для продолжения диалога.',
        ug_h_persist: 'Персистентность сессий',
        ug_persist_p: 'Реестр сессий (id чатов, отпечатки контекста, привязка сессия → аккаунт, накопленные счётчики расхода) пишется на диск в JSON-файлы, поэтому диалоги переживают перезапуски сервера и продолжают указывать на те же серверные чаты и аккаунты:',
        ug_persist_b1: 'Расположение: <span class="env-var">DANYAPI_CACHE_DIR</span>, по умолчанию системная временная директория (<code>%TEMP%\\danyapi</code> на Windows, <code>/tmp/danyapi</code> на Linux/macOS).',
        ug_persist_b2: 'Файлы: <code>&lt;provider&gt;-sessions-default.json</code>, <code>&lt;provider&gt;-contexts-default.json</code>, <code>&lt;provider&gt;-affinities-default.json</code>.',
        ug_persist_b3: 'Запись атомарная (временный файл + переименование), поэтому сбой посреди записи не повредит кэш.',
        ug_persist_b4: 'Задай <span class="env-var">DANYAPI_CACHE_DISABLED=1</span>, чтобы всё держать только в памяти.',
        ug_persist_b5: 'В Docker примонтируй <span class="env-var">DANYAPI_CACHE_DIR</span> как том, если хочешь, чтобы сессии переживали пересоздание контейнера.',
        ug_persist_note: 'Обрати внимание: сами чаты апстрима живут на стороне провайдера; локальный кэш лишь сопоставляет твой <code>session_id</code> / контекст сообщений с ними.',
        ug_h_files: 'Вложения файлов (DeepSeek)',
        ug_files_p: 'Отправляй файлы как base64 в поле <code>files</code> или как <code>image_url</code> (data URI) внутри сообщения:',
        ug_files_limits: 'Лимиты по моделям: <span class="model">deepseek-v4-vision</span> принимает только изображения; <span class="model">deepseek-v4-flash</span> - изображения (OCR) и текстовые файлы; <span class="model">deepseek-v4-pro</span> - ничего. Максимум 50 файлов по 100 МБ за запрос.',
        ug_h_tools: 'Вызов инструментов (эмуляция)',
        ug_tools_p: 'Ни chat.deepseek.com, ни chat.qwen.ai не предоставляют нативного function-calling API, поэтому DanyAPI эмулирует его на уровне прокси через промпт-инъекцию. Принимаются OpenAI-совместимые поля <code>tools</code>, <code>tool_choice</code> и <code>parallel_tool_calls</code>:',
        ug_tools_how: 'Как это работает:',
        ug_tools_li1: 'Когда есть <code>tools</code>, схема функций и строгая JSON-инструкция (с конкретным примером, без шаблонных плейсхолдеров) внедряются в промпт, уходящий к модели апстрима.',
        ug_tools_li2_intro: 'Модель отвечает вызовом инструмента - DanyAPI понимает несколько форматов и нормализует их все в корректный OpenAI-ответ:',
        ug_tools_li2_b1: 'JSON <code>{"tool_calls": [{"name": "...", "arguments": {...}}]}</code>;',
        ug_tools_li2_b2: 'легаси <code>{"function_call": {...}}</code>;',
        ug_tools_li2_b3: 'голый dict <code>{"name": "...", "arguments": {...}}</code> (стиль Qwen/DeepSeek) или голый массив <code>[...]</code> из них;',
        ug_tools_li2_b4: 'XML/Anthropic-стиль <code>&lt;tool_calls&gt;&lt;invoke name="..."&gt;...&lt;/invoke&gt;&lt;/tool_calls&gt;</code> (аргументы дочерними тегами, <code>&lt;parameter name="..."&gt;</code> или инлайн-JSON).',
        ug_tools_li2_outro: 'Результат - <code>message.tool_calls</code> (не-stream) или стриминговые <code>delta.tool_calls</code>, оба с <code>finish_reason: "tool_calls"</code>. Поддерживается любое число вызовов в одном ответе (<code>parallel_tool_calls</code>), поэтому клиенты с кучей инструментов (например, opencode) работают из коробки.',
        ug_tools_li3: 'Ты выполняешь инструмент и отправляешь результат обратно: <code>{"role": "tool", "tool_call_id": "&lt;id&gt;", "content": "22C, sunny"}</code>. DanyAPI рендерит результаты инструментов в промпт и продолжает диалог, пока модель не ответит (или не вызовет ещё инструментов).',
        ug_tools_notes: 'Примечания:',
        ug_tools_n1: '<code>tool_choice</code>: <code>"auto"</code> (по умолчанию), <code>"none"</code> (инструменты принимаются, но схема не внедряется), <code>"required"</code> или <code>{"type": "function", "function": {"name": "&lt;tool&gt;"}}</code>.',
        ug_tools_n2: 'Передавай <code>session_id</code> из первого ответа обратно в запрос с результатом инструмента, чтобы диалог остался на сервере. Stateless-клиенты тоже работают: кэш контекста переиспользует чат с предыдущего раунда, поэтому результаты инструментов уходят как продолжение того же серверного диалога, а не как переигровка всей истории в новый чат. Если сессию не удалось сопоставить (например, кэш вытеснили), вся история сообщений (включая результаты инструментов) переигрывается в промпт, и обычные OpenAI-клиенты продолжают работать.',
        ug_tools_n3: 'Пока есть <code>tools</code>, стриминговые ответы буферизуются до конца генерации, чтобы ответ можно было классифицировать как вызов инструмента или обычный текст. Размышления (<code>reasoning_content</code>) стримятся вживую в обоих случаях.',
        ug_h_json: 'JSON-режим (эмуляция)',
        ug_json_p1: '<code>response_format</code> эмулируется так же, как вызов инструментов - JSON-ограничение (и опциональная схема) внедряются в промпт:',
        ug_json_p2: '<code>response_format</code> принимает <code>"json_object"</code>, <code>{"type": "json_object"}</code> и <code>{"type": "json_schema", "json_schema": {...}}</code>. Как и вызов инструментов, это эмуляция на уровне промпта: ответы на практике JSON, но валидность по схеме не гарантируется - валидируй на стороне клиента.',
        ug_h_health: 'Системный промпт и health',
        ug_health_b1: 'Сообщения <code>system</code> собираются и внедряются как системный промпт модели перед первым сообщением пользователя (у веб-API апстрима нет отдельного поля <code>system</code>).',
        ug_health_b2: '<code>GET /health</code> возвращает <code>{"status": "ok", "deepseek": true, "qwen": true}</code> плюс статистику кэша по провайдерам (<code>deepseek_stats</code> / <code>qwen_stats</code>: здоровье аккаунтов, количество привязок сессий, размер кэша контекста и счётчики hit/miss) - полезно для readiness-проб и балансировщиков.',
        ug_health_b3: 'Когда клиент отключается посреди генерации, DanyAPI сообщает провайдеру апстрима остановить стрим, чтобы серверный чат не хранил частичный ответ.',
        ug_h_tokens: 'Расход токенов',
        ug_tokens_p: 'Каждый ответ несёт OpenAI-совместимый объект <code>usage</code>, и оба провайдера сообщают его <strong>накопительно за диалог</strong> (как в официальном API, где счётчик растёт с каждым ходом того же чата):',
        ug_tokens_b1: '<strong>Qwen</strong> - апстрим сообщает попарные счётчики prompt/completion за ход; DanyAPI суммирует их по чату. <code>prompt_tokens</code> = всего входных токенов, обработанных этим диалогом, <code>completion_tokens</code> = всего сгенерированного, <code>total_tokens</code> = их сумма.',
        ug_tokens_b2: '<strong>DeepSeek</strong> - веб-API отдаёт только один кумулятивный счётчик (<code>accumulated_token_usage</code>), поэтому <code>completion_tokens</code> - это всего сгенерированного в этом диалоге, а <code>prompt_tokens</code> всегда <code>0</code> (<code>total_tokens</code> равен <code>completion_tokens</code>).',
        ug_tokens_p2: 'Счётчики живут на сессии, поэтому новый диалог начинается с нуля, а продолжение по <code>session_id</code> продолжает считать. Они также персистятся в дисковом кэше сессий (переживают перезапуски).',
        ug_tokens_p3: 'Стриминговый расход: передай <code>"stream_options": {"include_usage": true}</code>, и финальный SSE-чанк понесёт тот же накопленный <code>usage</code> (как в официальном API).',
        ug_h_errors: 'Обработка ошибок',
        ug_errors_p: 'Не-stream запросы получают обычную HTTP-ошибку; stream-запросы получают SSE-событие <code>error</code> (и <code>data: [DONE]</code>) после открытия стрима.',
        ug_th_status: 'Статус',
        ug_th_when: 'Когда',
        ug_e1: 'Плохой запрос: невалидный base64/файлы, нарушены правила вложений по моделям, превышен лимит длины контекста',
        ug_e2: 'Ошибка авторизации DeepSeek (невалидный/истёкший токен); аккаунт помечается сломанным и исключается из пула',
        ug_e3: 'Неизвестное имя модели',
        ug_e4: 'Все аккаунты заняты (истёк <span class="env-var">DANYAPI_ACQUIRE_TIMEOUT</span>) или троттлинг апстрима после исчерпания ретраев',
        ug_e5: 'Сбой запроса к апстриму (сеть, загрузка файла, WAF-челлендж Qwen)',
        ug_e6: 'Провайдер не сконфигурирован (нет токенов/email для этого провайдера) или все его аккаунты сломаны',
        ug_errors_retries: 'Ретраи: <code>expert_busy_use_default</code> / <code>parallel_chat_limit</code> / <code>server_busy</code> / <code>busy</code> (DeepSeek) и <code>Too_Many_Requests</code> / <code>RateLimited</code> / <code>quotaLimited</code> (Qwen) автоматически ретраятся до 5 раз с экспоненциальным бэкоффом (от 1 с до 8 с) перед тем, как выйти как ошибки. Ответы также могут завершаться <code>finish_reason: "content_filter"</code>, когда апстрим модерирует вывод. См. <a href="internals.html#account-limits">Лимиты аккаунтов</a>.',
        in_subtitle: 'Реверс-инженерные протоколы провайдеров и лимиты аккаунтов.',
        pl_in_ds: 'DeepSeek',
        pl_in_qw: 'Qwen',
        pl_in_limits: 'Лимиты аккаунтов',
        in_ds_p: 'Протокол восстановлен из главного бандла chat.deepseek.com (<code>fe-static.deepseek.com/chat/static/main.4e922c397f.js</code>) и wasm-модуля <code>sha3_wasm_bg.7b9ca65ddd.wasm</code>:',
        in_ds_b1: 'Auth: <code>POST /api/v0/users/login</code> → <code>data.biz_data.user.token</code>, затем <code>Authorization: Bearer &lt;token&gt;</code>.',
        in_ds_b2: 'Заголовки: <code>x-client-bundle-id</code>, <code>x-client-platform</code>, <code>x-client-version</code>, <code>x-client-locale</code>, <code>x-client-timezone-offset</code>.',
        in_ds_b3: 'Сессия: <code>POST /api/v0/chat_session/create</code> (пустое тело) → <code>chat_session.id</code>.',
        in_ds_b4: 'Генерация: <code>POST /api/v0/chat/completion</code>: <code>{chat_session_id, parent_message_id, model_type, prompt, ref_file_ids, thinking_enabled, search_enabled, action, preempt}</code>.',
        in_ds_b5: 'Ответ - <code>text/event-stream</code>: события <code>ready</code>, дельты (<code>SET</code>/<code>APPEND</code>/<code>BATCH</code>, пути <code>response/...</code>), <code>finish</code>, <code>close</code>.',
        in_ds_b6: 'PoW-заголовок <code>X-DS-PoW-Response</code> - base64 от <code>{algorithm, challenge, salt, answer, signature, target_path}</code>. Челлендж одноразовый: <code>answer</code> = минимальный счётчик c, где <code>DeepSeekHashV1(f"{salt}_{expire_at}_" + str(c))</code> совпадает с <code>challenge</code> (32 байта). Сервер перебирает c в <code>[0, difficulty)</code>.',
        in_qw_p: 'Протокол восстановлен из фронтенд-бандла chat.qwen.ai (<code>assets.alicdn.com/g/qwenweb/qwen-chat-fe/0.2.83/js/main.js</code>):',
        in_qw_b1: 'Auth: <code>POST /api/v2/auths/signin</code> с <code>{email, password}</code>, где пароль - это SHA-256 hex от текста → <code>data.token</code> (JWT). Запросы шлют его как <code>Authorization: Bearer &lt;token&gt;</code> и cookie <code>token</code>.',
        in_qw_b2: 'Заголовки: <code>source: web</code>, <code>version: 0.2.83</code>, <code>X-Request-Id</code>, <code>Timezone</code>, браузерные <code>sec-ch-ua</code>/<code>User-Agent</code>/<code>Origin</code>/<code>Referer</code>.',
        in_qw_b3: 'Сессия: <code>POST /api/v2/chats/new</code> (<code>{chatId, models, chat_type: "t2t", chat_mode: "normal", timestamp}</code>) → <code>data.id</code> (id чата).',
        in_qw_b4: 'Генерация: <code>POST /api/v2/chat/completions?chat_id=&lt;id&gt;</code> с <code>{stream, version: "2.1", incremental_output, chat_id, model, parent_id, messages: [{fid, parentId, role, content, chat_type: "t2t", feature_config: {thinking_enabled, output_schema: "phase", ...}}]}</code>. История чата живёт на сервере; <code>parent_id</code> указывает на id последнего ответа ассистента, поэтому следующий ход продолжает тот же диалог.',
        in_qw_b5: 'Ответ - <code>text/event-stream</code> из JSON-чанков в стиле OpenAI: <code>{"choices": [{"delta": {"role", "content", "phase", "status"}}], "response_id", "usage"}</code>. Чанк <code>response.created</code> открывает стрим с <code>response_id</code> ассистента; контент стримится в фазе <code>answer</code>, размышления - в фазах <code>think</code>/<code>DeepThinking</code>/<code>thinking_summary</code>, а стрим заканчивается чанком, у которого <code>delta.status</code> = <code>finished</code>.',
        in_limits_h: 'Лимиты аккаунтов',
        in_l1: 'Один аккаунт chat.deepseek.com может генерировать <strong>одно сообщение за раз</strong> (иначе сервер отвечает <code>parallel_chat_limit</code>). DanyAPI держит <strong>пул аккаунтов</strong> и распределяет параллельные запросы по аккаунтам; если все заняты, запросы ждут в очереди. Больше токенов = больше параллельных генераций. То же касается аккаунтов chat.qwen.ai (у Qwen свой пул, поэтому параллельные генерации DeepSeek и Qwen независимы).',
        in_l2: 'Сессии привязаны к аккаунту, на котором созданы: повторные запросы с тем же <code>session_id</code> (или тем же закэшированным контекстом сообщений) маршрутизируются на тот же аккаунт, и история диалога сохраняется на сервере.',
        in_l3: 'Кэш сессий/контекста в памяти ограничен LRU на аккаунт и на провайдера. Когда запись вытесняется, соответствующий чат больше не переиспользуется и при следующем запросе создаётся новый; явный <code>session_id</code> остаётся надёжным способом закрепить диалог. Неиспользуемые записи также истекают через <span class="env-var">DANYAPI_SESSION_TTL_SECONDS</span>, а привязка session-id → аккаунт очищается вместе с кэшем, так что память остаётся ограниченной на долго работающих инстансах.',
        in_l4: 'DeepSeek может троттлить аккаунты (особенно экспертную модель <span class="model">deepseek-v4-pro</span> - «limited resource»). Ответы с <code>finish_reason</code> <code>expert_busy_use_default</code> / <code>parallel_chat_limit</code> / <code>server_busy</code> / <code>busy</code> автоматически ретраятся (до 5 ретраев с экспоненциальным бэкоффом). Если все попытки исчерпаны:',
        in_l4_1: 'не-stream запросы получают HTTP 429 с текстом ошибки DeepSeek;',
        in_l4_2: 'stream-запросы получают SSE-событие <code>error</code> с <code>finish_reason</code>.',
        in_l5: 'Qwen может отвечать <code>Too_Many_Requests</code> / <code>RateLimited</code> / <code>quotaLimited</code>; они ретраятся автоматически так же (до 5 раз), затем отдаются как HTTP 429 или SSE-событие <code>error</code>.',
        in_l6: 'PoW-челлендж DeepSeek одноразовый - новый решается на каждый запрос (следующий префетчится заранее, чтобы не ждать).',
        in_l7: 'Когда все аккаунты заняты, запросы ждут свободный аккаунт. Задай <span class="env-var">DANYAPI_ACQUIRE_TIMEOUT</span> (секунды), чтобы ограничить ожидание и получить HTTP 429 («все аккаунты заняты») вместо вечного ожидания.',
        in_l8: 'Длинные серверные диалоги со временем превышают окно контекста модели. Тогда DanyAPI обнаруживает ошибку лимита контекста, выбрасывает переполненный чат (он больше не переиспользуется) и сообщает об ошибке: HTTP 400 для не-stream запросов или SSE-событие <code>error</code> с <code>finish_reason: "length"</code> для stream. Следующий запрос автоматически начинает свежий диалог. Чтобы реже упираться в лимит, держи встроенные инструменты Qwen выключенными (см. <a href="setup.html#account-setup">Настройку аккаунтов</a>) и ротируй длинные диалоги на стороне клиента.',
    };

    function read() {
        try { return localStorage.getItem(KEY); } catch (e) { return null; }
    }
    function write(v) {
        try { localStorage.setItem(KEY, v); } catch (e) {}
    }
    function detect() {
        var saved = read();
        if (saved && SUPPORTED.indexOf(saved) !== -1) return saved;
        var nav = (navigator.language || navigator.userLanguage || '').toLowerCase();
        return nav.indexOf('ru') === 0 ? 'ru' : 'en';
    }

    var currentLang = detect();

    function capture(el) {
        if (el.hasAttribute('data-i18n') && !('_i18nText' in el)) el._i18nText = el.textContent;
        if (el.hasAttribute('data-i18n-html') && !('_i18nHtml' in el)) el._i18nHtml = el.innerHTML;
        if (el.hasAttribute('data-i18n-aria') && !('_i18nAria' in el)) el._i18nAria = el.getAttribute('aria-label');
    }

    function apply(lang) {
        var t = lang === 'en' ? null : RU;
        document.documentElement.setAttribute('lang', lang);
        document.querySelectorAll('[data-i18n]').forEach(function (el) {
            capture(el);
            if (t && t[el.getAttribute('data-i18n')] !== undefined) {
                el.textContent = t[el.getAttribute('data-i18n')];
            } else {
                el.textContent = el._i18nText;
            }
        });
        document.querySelectorAll('[data-i18n-html]').forEach(function (el) {
            capture(el);
            if (t && t[el.getAttribute('data-i18n-html')] !== undefined) {
                el.innerHTML = t[el.getAttribute('data-i18n-html')];
            } else {
                el.innerHTML = el._i18nHtml;
            }
        });
        document.querySelectorAll('[data-i18n-aria]').forEach(function (el) {
            capture(el);
            var key = el.getAttribute('data-i18n-aria');
            el.setAttribute('aria-label', t && t[key] !== undefined ? t[key] : el._i18nAria);
        });
        var titleKey = document.documentElement.getAttribute('data-title-key') || 'title_index';
        if (t && t[titleKey]) document.title = t[titleKey];
        document.querySelectorAll('[data-lang-btn]').forEach(function (btn) {
            btn.classList.toggle('active', btn.getAttribute('data-lang-btn') === lang);
        });
    }

    function run() {
        document.querySelectorAll('[data-i18n], [data-i18n-html], [data-i18n-aria]').forEach(capture);
        apply(currentLang);
    }

    function setLang(lang) {
        if (SUPPORTED.indexOf(lang) === -1) lang = 'en';
        currentLang = lang;
        write(lang);
        apply(lang);
        document.dispatchEvent(new CustomEvent('danyapi:lang', { detail: { lang: lang } }));
    }

    document.documentElement.setAttribute('lang', currentLang);
    var t0 = currentLang === 'ru' ? RU : null;
    var titleKey0 = document.documentElement.getAttribute('data-title-key') || 'title_index';
    if (t0 && t0[titleKey0]) document.title = t0[titleKey0];

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', run);
    } else {
        run();
    }

    window.DANYAPI_I18N = {
        current: function () { return currentLang; },
        setLang: setLang,
        apply: apply,
        t: function (lang, key) {
            if (lang || currentLang) {
                if ((lang || currentLang) === 'ru') return RU[key];
            }
            return undefined;
        }
    };
})();
