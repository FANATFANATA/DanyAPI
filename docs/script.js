(function () {
    "use strict";

    /* ---------------- nav scrolled state ---------------- */
    var nav = document.getElementById("nav");
    function onScroll() {
        nav.classList.toggle("scrolled", window.scrollY > 20);
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    /* ---------------- docs dropdown ---------------- */
    var docsNav = document.querySelector(".docs-nav");
    if (docsNav) {
        var docsBtn = docsNav.querySelector(".docs-btn");
        function closeDocs() {
            docsNav.classList.remove("open");
            docsBtn.setAttribute("aria-expanded", "false");
        }
        docsBtn.addEventListener("click", function (e) {
            e.stopPropagation();
            var open = docsNav.classList.toggle("open");
            docsBtn.setAttribute("aria-expanded", open ? "true" : "false");
        });
        document.addEventListener("click", closeDocs);
        document.addEventListener("keydown", function (e) {
            if (e.key === "Escape") closeDocs();
        });
    }

    /* ---------------- reveal on scroll ---------------- */
    var revealEls = document.querySelectorAll(".reveal");
    if ("IntersectionObserver" in window) {
        var io = new IntersectionObserver(function (entries) {
            entries.forEach(function (e) {
                if (e.isIntersecting) {
                    e.target.classList.add("revealed");
                    io.unobserve(e.target);
                }
            });
        }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
        revealEls.forEach(function (el) { io.observe(el); });
    } else {
        revealEls.forEach(function (el) { el.classList.add("revealed"); });
    }

    /* ---------------- code tabs ---------------- */
    var tabs = document.querySelectorAll(".tab");
    tabs.forEach(function (tab) {
        tab.addEventListener("click", function () {
            var name = tab.getAttribute("data-tab");
            tabs.forEach(function (t) { t.classList.toggle("active", t === tab); });
            document.querySelectorAll(".code-pane").forEach(function (pane) {
                pane.classList.toggle("active", pane.getAttribute("data-pane") === name);
            });
        });
    });

    /* ---------------- copy buttons ---------------- */
    var COPY_LABELS = {
        en: "Copy", ru: "Копировать",
        okEn: "Copied!", okRu: "Скопировано!"
    };
    function copyText(text) {
        if (navigator.clipboard && window.isSecureContext) {
            return navigator.clipboard.writeText(text);
        }
        return new Promise(function (resolve, reject) {
            var ta = document.createElement("textarea");
            ta.value = text;
            ta.style.position = "fixed";
            ta.style.opacity = "0";
            document.body.appendChild(ta);
            ta.select();
            try { document.execCommand("copy"); resolve(); }
            catch (e) { reject(e); }
            document.body.removeChild(ta);
        });
    }
    document.querySelectorAll("[data-copy]").forEach(function (btn) {
        btn.addEventListener("click", function () {
            copyText(btn.getAttribute("data-copy")).then(function () {
                var prev = btn.textContent;
                var isRu = document.documentElement.lang === "ru";
                btn.textContent = isRu ? COPY_LABELS.okRu : COPY_LABELS.okEn;
                btn.classList.add("copied");
                setTimeout(function () {
                    btn.textContent = prev;
                    btn.classList.remove("copied");
                }, 1600);
            });
        });
    });

    /* ---------------- hero terminal animation ---------------- */
    var termBody = document.getElementById("termBody");
    var TERM_LINES = {
        en: [
            { cls: "term-prompt", text: "$ python -m danyapi" },
            { cls: "term-info", text: "Loading .env - deepseek + qwen" },
            { cls: "term-ok", text: "✓ DeepSeek: 3 tokens valid" },
            { cls: "term-ok", text: "✓ Qwen: 1 token valid" },
            { cls: "term-info", text: "Uvicorn running on http://0.0.0.0:8000" },
            { cls: "term-prompt", text: "→ POST /v1/chat/completions · deepseek-v4-flash" },
            { cls: "term-data", text: 'data: {"choices":[{"delta":{"role":"assistant"}}]}' },
            { cls: "term-data", text: 'data: {"choices":[{"delta":{"content":"Hi"}}]}' },
            { cls: "term-data", text: 'data: {"choices":[{"delta":{"content":" there! How can I help?"}}]}' },
            { cls: "term-data", text: 'data: [DONE]' },
            { cls: "term-ok", text: "✓ session_id: 8f3a...c21e · tokens: 42" }
        ],
        ru: [
            { cls: "term-prompt", text: "$ python -m danyapi" },
            { cls: "term-info", text: "Загружаю .env - deepseek + qwen" },
            { cls: "term-ok", text: "✓ DeepSeek: 3 токена валидны" },
            { cls: "term-ok", text: "✓ Qwen: 1 токен валиден" },
            { cls: "term-info", text: "Uvicorn запущен на http://0.0.0.0:8000" },
            { cls: "term-prompt", text: "→ POST /v1/chat/completions · deepseek-v4-flash" },
            { cls: "term-data", text: 'data: {"choices":[{"delta":{"role":"assistant"}}]}' },
            { cls: "term-data", text: 'data: {"choices":[{"delta":{"content":"Привет"}}]}' },
            { cls: "term-data", text: 'data: {"choices":[{"delta":{"content":"! Чем могу помочь?"}}]}' },
            { cls: "term-data", text: 'data: [DONE]' },
            { cls: "term-ok", text: "✓ session_id: 8f3a...c21e · tokens: 42" }
        ]
    };
    var termIdx = 0;
    var termStarted = false;

    function typeLine(line, resolve) {
        var row = document.createElement("div");
        row.className = "term-line " + line.cls;
        termBody.appendChild(row);
        var i = 0;
        var t = line.text;
        (function step() {
            if (i <= t.length) {
                row.textContent = t.slice(0, i);
                i++;
                setTimeout(step, 14);
            } else {
                resolve();
            }
        })();
    }

    function playTerminal() {
        if (!termBody) return;
        var lines = TERM_LINES[document.documentElement.lang] || TERM_LINES.en;
        if (termIdx >= lines.length) {
            setTimeout(playTerminal, 9000);
            return;
        }
        var line = lines[termIdx];
        termIdx++;
        var delay = line.cls === "term-prompt" ? 650 : 200;
        setTimeout(function () {
            typeLine(line, function () {
                termBody.scrollTop = termBody.scrollHeight;
                setTimeout(playTerminal, delay);
            });
        }, delay);
    }

    function startTerminal() {
        if (termStarted || !termBody) return;
        termStarted = true;
        playTerminal();
    }

    function restartTerminal() {
        if (!termStarted || !termBody) return;
        termBody.innerHTML = "";
        termIdx = 0;
        playTerminal();
    }

    if ("IntersectionObserver" in window) {
        var termObserver = new IntersectionObserver(function (entries) {
            entries.forEach(function (e) {
                if (e.isIntersecting) {
                    startTerminal();
                    termObserver.disconnect();
                }
            });
        }, { threshold: 0.4 });
        termObserver.observe(termBody);
    } else {
        startTerminal();
    }

    /* ---------------- i18n (EN / RU) ---------------- */
    var LANGS = ["en", "ru"];
    var STORE_KEY = "danyapi-lang";

    var I18N = {
        ru: {
            nav_features: "Возможности",
            nav_models: "Модели",
            nav_quickstart: "Быстрый старт",
            nav_how: "Как работает",
            nav_faq: "FAQ",
            nav_docs: "Доки",
            nav_overview: "Обзор",
            nav_install: "Установка и аккаунты",
            nav_config: "Конфигурация",
            nav_usage: "Использование",
            nav_internals: "Как это работает",
            hero_badge: "Бесплатно · Open Source · OpenAI-совместимо",
            hero_h1_a: "API больших моделей.",
            hero_h1_b: "Ноль ключей.",
            hero_h1_c: "Ноль затрат.",
            hero_sub: "DanyAPI - это OpenAI-совместимый HTTP API на Python + FastAPI. Вместо платных API он работает с внутренними API бесплатных веб-клиентов chat.deepseek.com и chat.qwen.ai через серверные аккаунты. Вашим пользователям не нужен ни один API-ключ.",
            hero_cta_start: "Быстрый старт",
            hero_cta_gh: "на GitHub",
            hero_term_note: "Одна команда устанавливает, настраивает и запускает сервер. Запросы обслуживаются бесплатными серверными аккаунтами.",
            stat_zero: "за токен, навсегда",
            stat_providers: "бесплатных провайдера",
            stat_cmd: "команда для установки",
            stat_retries: "авторетраев с бэкоффом",
            stat_cache: "серверных чатов в кэше",
            features_eyebrow: "Возможности",
            features_title: "Всё, что есть у платных API.",
            features_title_hi: "Без их счетов.",
            features_sub: "Прямая замена официального OpenAI API - ваш существующий код продолжит работать как есть.",
            f1_t: "OpenAI-совместимый",
            f1_d: "<code>GET /v1/models</code> и <code>POST /v1/chat/completions</code> в официальном формате. Поменяйте <code>base_url</code> - и клиент работает.",
            f2_t: "SSE-стриминг",
            f2_d: "Реальный стрим через чанки <code>data:</code> и <code>data: [DONE]</code>, с живыми трассами размышлений как <code>reasoning_content</code>.",
            f3_t: "Трассы размышлений",
            f3_d: "Рассуждения DeepSeek и Qwen доступны вашему приложению - стримятся вживую, у обоих провайдеров, как в официальном API.",
            f4_t: "Вызов инструментов",
            f4_d: "Эмуляция <code>tools</code>, <code>tool_choice</code> и <code>parallel_tool_calls</code> с корректными ответами <code>finish_reason: \"tool_calls\"</code>.",
            f5_t: "JSON-режим",
            f5_d: "<code>response_format</code> с <code>json_object</code> и <code>json_schema</code> - структурированный вывод для ваших агентов.",
            f6_t: "Серверные сессии",
            f6_d: "Многопоточные чаты живут на сервере. Stateless-клиенты автоматически сохраняют контекст через отпечаток сообщений.",
            f7_t: "Веб-поиск",
            f7_d: "Включайте актуальные ответы флагом <code>search</code> на моделях DeepSeek.",
            f8_t: "Вложения файлов",
            f8_d: "Изображения и текстовые файлы как base64 или data URI. Vision, OCR и анализ файлов - до 50 файлов по 100 МБ.",
            f9_t: "Реальный расход токенов",
            f9_d: "OpenAI-совместимый <code>usage</code>, накапливаемый за диалог, включая стриминг через <code>stream_options.include_usage</code>.",
            f10_t: "Переживает рестарты",
            f10_d: "Сессии, привязки аккаунтов и счётчики расхода персистятся на диск. Диалоги переживают перезапуск сервера.",
            f11_t: "Пул аккаунтов",
            f11_d: "N токенов = N параллельных генераций. Запросы распределяются round-robin с ретраями и экспоненциальным бэкоффом.",
            f12_t: "Health для прод",
            f12_d: "<code>GET /health</code> отдаёт статус провайдеров и статистику кэша - готово для readiness-проб и балансировщиков.",
            models_eyebrow: "Модели",
            models_title: "Два бесплатных провайдера.",
            models_title_hi: "Один OpenAI API.",
            models_sub: "Маршрутизация по имени модели - <code>deepseek-*</code> или <code>qwen*</code>. Оба провайдера опциональны и работают вместе.",
            models_ds_tag: "chat.deepseek.com",
            models_ds_flash: "по умолчанию · поиск · OCR",
            models_ds_pro: "эксперт · размышления",
            models_ds_vision: "vision",
            models_ds_note: "Размышления на всех моделях. Веб-поиск на flash. Файлы: vision = изображения, flash = изображения + текст, pro = нет.",
            models_qw_tag: "chat.qwen.ai",
            models_qw_1: "топ-модель",
            models_qw_2: "быстрая",
            models_qw_3: "подтягиваются из аккаунта",
            models_qw_note: "Размышления и поиск встроены. Список моделей забирается из аккаунта при старте - новые появляются автоматически.",
            qs_eyebrow: "Быстрый старт",
            qs_title: "Запуск за",
            qs_title_hi: "меньше минуты.",
            qs_sub: "Установка одной командой на Windows, Linux и macOS. Без танцев с Python.",
            qs_tab_install: "Установка",
            qs_tab_python: "OpenAI SDK",
            qs_tab_curl: "curl streaming",
            qs_tab_docker: "Docker",
            qs_pane_install: "PowerShell",
            qs_pane_install_alt: "Linux / macOS",
            qs_install_note: "Скрипт клонирует репозиторий, ставит зависимости, создаёт <code>.env</code>, живым запросом проверяет токены провайдеров и запускает сервер. Обновляется сам при каждом старте.",
            qs_python_note: "<code>api_key</code> обязателен для SDK, но не проверяется - все запросы делают серверные аккаунты.",
            qs_curl_note: "Трассы размышлений стримятся вживую как <code>reasoning_content</code>. Добавьте <code>\"stream_options\": {\"include_usage\": true}</code> для расхода в финальном чанке.",
            qs_docker_note: "Готовый образ, пушится на каждый коммит и тег версии. Нативный PoW-солвер собирается в образ для максимальной скорости.",
            copy: "Копировать",
            how_eyebrow: "Как работает",
            how_title: "Умный прокси к",
            how_title_hi: "бесплатным моделям.",
            how_sub: "Реверс-инженерные протоколы провайдеров, пул аккаунтов, решение PoW и кэш сессий - всё спрятано за чистым OpenAI API.",
            flow_in: "Ваше приложение",
            flow_api: "FastAPI · /v1",
            flow_pool: "Пул аккаунтов",
            flow_pool_detail: "round-robin · ретраи · PoW",
            how1_t: "Ключей нет",
            how1_d: "Все запросы делают серверные аккаунты. Потребители API просто указывают ваш инстанс - <code>api_key</code> не проверяется.",
            how2_t: "Параллельность",
            how2_d: "Один аккаунт = одно сообщение за раз. DanyAPI держит пул токенов и распределяет параллельные запросы, ставя в очередь, когда все заняты.",
            how3_t: "Сессии, что живут",
            how3_d: "Чаты живут на сервере, как в веб-клиентах. <code>session_id</code> фиксирует диалог; stateless-клиенты сопоставляются по отпечатку сообщений.",
            how4_t: "Эмуляция и нормализация",
            how4_d: "Вызов инструментов и JSON-режим эмулируются на уровне прокси через промпт-инъекцию и нормализуются в идеальные OpenAI-ответы.",
            faq_eyebrow: "FAQ",
            faq_title: "Вопросы?",
            faq_title_hi: "Есть ответы.",
            q1: "Это правда бесплатно?",
            a1: "Да. DanyAPI работает с внутренними API бесплатных веб-клиентов chat.deepseek.com и chat.qwen.ai через серверные аккаунты. Никаких тарифов, лимитов и ключей.",
            q2: "Нужен ли пользователям API-ключ?",
            a2: "Нет. SDK OpenAI требует поле <code>api_key</code>, но DanyAPI его не проверяет - передайте любое значение. Все запросы к апстриму делают настроенные вами серверные аккаунты.",
            q3: "Какие провайдеры и модели?",
            a3: "DeepSeek (<code>deepseek-v4-flash</code>, <code>deepseek-v4-pro</code>, <code>deepseek-v4-vision</code>) и Qwen (<code>qwen3.8-max</code>, <code>qwen3.7-plus</code>, … - подтягиваются из аккаунта). Маршрутизация по имени модели; оба могут работать одновременно.",
            q4: "Подходит для агентов и инструментов?",
            a4: "Да - эмуляция <code>tools</code>, <code>tool_choice</code> и <code>parallel_tool_calls</code> с <code>finish_reason: \"tool_calls\"</code>. Клиенты со множеством инструментов (например, opencode) работают из коробки.",
            q5: "Какие лимиты аккаунтов?",
            a5: "Один аккаунт генерирует одно сообщение за раз, поэтому пул из N токенов даёт N параллельных генераций. Пулы DeepSeek и Qwen независимы. Ошибки занятости автоматически ретраятся до 5 раз с бэкоффом.",
            q6: "Диалоги переживают рестарты?",
            a6: "Да. Id сессий, отпечатки контекста, привязка аккаунтов и расход токенов персистятся на диск (JSON) с атомарной записью. В Docker примонтируйте <code>DANYAPI_CACHE_DIR</code> как том.",
            cta_title: "Бесплатные модели.",
            cta_sub: "Ваш API.",
            cta_gh: "GitHub",
            cta_tg: "Телеграм-канал",
            footer_creator: "Создатель",
            footer_channel: "Телеграм-канал",
            footer_docs: "Документация",
            footer_note: "Сделано на FastAPI и Python · реверс-инжиниринг, не аффилировано с DeepSeek или Alibaba",
            meta_title: "DanyAPI - бесплатный OpenAI-совместимый API без ключей",
            lang_en: "Английский",
            lang_ru: "Русский"
        }
    };

    function detectLang() {
        try {
            var saved = localStorage.getItem(STORE_KEY);
            if (saved && LANGS.indexOf(saved) !== -1) return saved;
        } catch (e) {}
        var nav = (navigator.language || navigator.userLanguage || "").toLowerCase();
        return nav.indexOf("ru") === 0 ? "ru" : "en";
    }

    function applyLang(lang) {
        var t = I18N[lang];
        document.documentElement.lang = lang;

        document.querySelectorAll("[data-i18n]").forEach(function (el) {
            var key = el.getAttribute("data-i18n");
            if (t && t[key] !== undefined) {
                el.innerHTML = t[key];
            }
        });
        document.querySelectorAll("[data-i18n-aria]").forEach(function (el) {
            var key = el.getAttribute("data-i18n-aria");
            if (t && t[key] !== undefined) {
                el.setAttribute("aria-label", t[key]);
            }
        });
        if (t && t.meta_title) document.title = t.meta_title;

        document.querySelectorAll(".lang-btn").forEach(function (btn) {
            btn.classList.toggle("active", btn.getAttribute("data-lang") === lang);
        });
        document.querySelectorAll(".copy-btn").forEach(function (cb) {
            if (!cb.classList.contains("copied")) cb.textContent = t && t.copy ? t.copy : "Copy";
        });

        var main = document.querySelector("main");
        if (main) {
            main.classList.remove("i18n-swap");
            void main.offsetWidth;
            main.classList.add("i18n-swap");
        }

        restartTerminal();
    }

    function formatStarCount(n) {
        if (n >= 1000000) return (n / 1000000).toFixed(1).replace(/\.0$/, "") + "M";
        if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "k";
        return n.toString();
    }

    function fetchGitHubStars() {
        var starsEl = document.getElementById("gh-stars");
        if (!starsEl) return;
        fetch("https://api.github.com/repos/FANATFANATA/DanyAPI")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data && data.stargazers_count != null) {
                    starsEl.textContent = "★ " + formatStarCount(data.stargazers_count) + " ";
                    starsEl.style.opacity = "1";
                }
            })
            .catch(function () {
                starsEl.style.opacity = "0";
            });
    }

    var currentLang = detectLang();
    applyLang(currentLang);
    fetchGitHubStars();

    document.querySelectorAll(".lang-btn").forEach(function (btn) {
        btn.addEventListener("click", function () {
            var lang = btn.getAttribute("data-lang");
            if (lang === currentLang) return;
            currentLang = lang;
            try { localStorage.setItem(STORE_KEY, currentLang); } catch (e) {}
            applyLang(currentLang);
        });
    });
})();