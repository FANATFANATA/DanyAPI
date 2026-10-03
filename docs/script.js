(function () {
    "use strict";
    var nav = document.getElementById("nav");
    var scrollTicking = false;
    function updateNav() {
        scrollTicking = false;
        nav.classList.toggle("scrolled", window.scrollY > 20);
    }
    function onScroll() {
        if (scrollTicking) return;
        scrollTicking = true;
        requestAnimationFrame(updateNav);
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    updateNav();
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
    var COPY_LABELS = {
        okEn: "Copied", okRu: "Скопировано"
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
    var LANGS = ["en", "ru"];
    var STORE_KEY = "danyapi-lang";

    var I18N = {
        ru: {
            nav_features: "Возможности",
            nav_hosted: "Публичный",
            nav_models: "Модели",
            nav_quickstart: "Быстрый старт",
            nav_faq: "FAQ",
            hero_badge: "Бесплатно · Open Source · OpenAI-совместимо",
            hero_h1_b: "Ноль затрат.",
            hero_sub: "DanyAPI - бесплатный OpenAI-совместимый API, который запускает веб-клиенты DeepSeek и Qwen, официальный API GigaChat, шлюз OpenCode Zen и неофициальные Alice и Duck.ai на сервере из ваших токенов. Без платных ключей и лимитов.",
            hero_cta_start: "Быстрый старт",
            hero_cta_gh: "на GitHub",
            features_eyebrow: "Возможности",
            features_title: "Всё, что есть у платных API.",
            features_title_hi: "Без их счетов.",
            features_sub: "Прямая замена официальному OpenAI API - поменяйте <code>base_url</code>, и ваш код заработает.",
            f1_t: "OpenAI-совместимый",
            f1_d: "<code>GET /v1/models</code> и <code>POST /v1/chat/completions</code> в официальном формате. Поменяйте <code>base_url</code> - и клиент работает.",
            f2_t: "Стриминг",
            f2_d: "Живой стрим через чанки <code>data:</code> и <code>data: [DONE]</code>, плюс трассы размышлений как <code>reasoning_content</code>.",
            f3_t: "Трассы размышлений",
            f3_d: "Рассуждения DeepSeek и Qwen доступны вашему приложению - стримятся вживую, у обоих провайдеров.",
            f4_t: "Вызов инструментов",
            f4_d: "Эмуляция <code>tools</code>, <code>tool_choice</code> и <code>parallel_tool_calls</code> с ответами <code>finish_reason: \"tool_calls\"</code>.",
            f5_t: "JSON-режим",
            f5_d: "<code>response_format</code> с <code>json_object</code> и <code>json_schema</code> - структурированный вывод для агентов.",
            f7_t: "Веб-поиск",
            f7_d: "Включайте актуальные ответы флагом <code>search</code> на моделях DeepSeek.",
            f8_t: "Вложения файлов",
            f8_d: "Изображения и текстовые файлы как base64 или data URI. Vision, OCR и анализ файлов по модели.",
            models_eyebrow: "Модели",
            models_title: "Шесть провайдеров.",
            models_title_hi: "Один OpenAI API.",
            models_sub: "Маршрутизация по имени модели - <code>deepseek-*</code>, <code>qwen*</code>, <code>GigaChat*</code>, <code>opencode/*</code>, <code>alice</code> или модель Duck.ai. Все провайдеры опциональны и работают вместе.",
            models_ds_flash: "размышления · поиск · файлы · vision",
            models_ds_pro: "размышления · поиск · файлы · vision",
            models_ds_note: "Список читается из настроек веб-клиента, поэтому показываются только те типы моделей, которые выданы аккаунту. Веб-поиск через флаг <code>search</code>; размышления через суффикс <code>-thinking</code> или флаг <code>thinking</code>.",
            models_qw_1: "топ-модель",
            models_qw_2: "быстрая",
            models_qw_3: "подтягиваются из аккаунта",
            models_qw_note: "Размышления и поиск встроены. Список моделей забирается из аккаунта и обновляется по таймеру - новые появляются автоматически.",
            models_gc_lite: "лайт · текст",
            models_gc_pro: "инструменты · vision",
            models_gc_max: "инструменты · vision",
            models_gc_note: "Официальный API GigaChat с бесплатной фремиум-квотой. Нужен ключ авторизации из Studio. Список моделей читается из аккаунта и обновляется по таймеру. Картинки работают только на Pro, Max и Ultra.",
            models_oc_gpt: "инструменты · vision",
            models_oc_claude: "инструменты · vision",
            models_oc_kimi: "инструменты",
            models_oc_note_list: "подтягиваются с шлюза",
            models_oc_note: "Курируемый шлюз моделей OpenCode на opencode.ai/zen с оплатой по токенам. Нужен API-ключ. Часть идентификаторов совпадает с Qwen и DeepSeek, такие нужно писать с префиксом <code>opencode/</code>. Обслуживается только половина каталога с форматом chat completions.",
            models_al_keyless: "без ключа",
            models_al_note: "Неофициальный провайдер, включается через <code>ALICE_ENABLED=1</code>, по умолчанию выключен. У Яндекса нет публичного API, поэтому используется недокументированный внутренний протокол, который может отвалиться в любой момент. Без состояния и без потокового текста, поэтому история сворачивается в один промпт. Нужно хотя бы одно сообщение, вызовы инструментов не поддерживаются, <code>max_tokens</code> обрезает ответ.",
            models_duck_keyless: "без ключа",
            models_duck_note: "Неофициальный провайдер, включается через <code>DUCKAI_ENABLED=1</code>, по умолчанию выключен. У DuckDuckGo нет публичного API, поэтому используется недокументированный внутренний протокол с проверкой отпечатка браузера, который может отвалиться в любой момент. Нужен Node.js на хосте. Прямо отказывает дата-центровым адресам, так что запускайте с домашней или мобильной линии.",
            models_mistral_keyless: "без ключа",
            models_mistral_note: "Неофициальный провайдер, включается через <code>MISTRAL_ENABLED=1</code> и <code>MISTRAL_LOGINS</code> с парами <code>email:password</code>, по умолчанию выключен. У Mistral нет публичного API, поэтому провайдер говорит с Le Chat с сессией аккаунта, как мобильное приложение. Бесплатные аккаунты ограничены по числу сообщений. Может отвалиться в любой момент.",
            qs_eyebrow: "Быстрый старт",
            qs_need: "Нужен только бесплатный токен DeepSeek или Qwen, ключ GigaChat из Studio или API-ключ OpenCode Zen - всё остальное сделает скрипт.",
            qs_title: "Запуск за",
            qs_title_hi: "меньше минуты.",
            qs_sub: "Установка одной командой на Windows, Linux и macOS.",
            qs_tab_install: "Установка",
            qs_tab_docker: "Docker",
            qs_pane_install: "PowerShell",
            qs_pane_install_alt: "Linux / macOS",
            qs_install_note: "Скрипт клонирует репозиторий, ставит зависимости, создаёт <code>.env</code>, проверяет токены и подсказывает, как запустить сервер. Обновляется сам при каждом старте. Не хочется доставать токены из хранилища браузера вручную? Запустите <code>docs/token_utility.sh</code> (или <code>docs\\token_utility.bat</code> на Windows) - он вытащит их за вас.",
            qs_docker_note: "Готовый образ, пушится на каждый пуш в prod и dev, а также на тег версии. Нативный PoW-солвер собран в образ для максимальной скорости.",
            copy: "Копировать",
            faq_eyebrow: "FAQ",
            faq_title: "Вопросы?",
            faq_title_hi: "Есть ответы.",
            q1: "Это правда бесплатно?",
            a1: "Да. DanyAPI использует внутренние API бесплатных веб-клиентов через аккаунты из ваших бесплатных токенов. Неофициальные Alice и Duck.ai вообще не требуют никаких credentials. Никаких тарифов и лимитов.",
            q2: "Нужен ли пользователям API-ключ?",
            a2: "При локальном запуске или личном сервере API-ключ не требуется (передайте любое значение). На публичном инстансе (режим BYOK) передайте ваш токен провайдера как Bearer-токен.",
            q3: "Какие провайдеры и модели?",
            a3: "Списки моделей не захардкожены: DeepSeek (<code>default</code> и другие типы, которые выданы аккаунту), Qwen (<code>qwen3.8-max</code>, <code>qwen3.7-plus</code>, ...), GigaChat (<code>GigaChat-2</code>, <code>GigaChat-2-Pro</code> и другие), OpenCode Zen (<code>opencode/gpt-5.6-sol</code>, <code>opencode/kimi-k3</code>, ...) и Duck.ai (модели бесплатного тарифа) читаются с эндпоинтов провайдеров и обновляются по таймеру, Alice (<code>alice</code>, <code>yagpt</code>) отдаёт свои алиасы. Маршрутизация по имени модели; все работают одновременно.",
            q7: "Есть ли лимиты или забанят токен?",
            a7: "Запросы идут через бесплатные веб-клиенты в человекоподобном темпе. Добавьте больше токенов в пул для параллелизма - проект держится в рамках нормального использования, но бесплатные токены - best-effort.",
            q8: "Есть ли учёт использования?",
            a8: "<code>GET /v1/usage</code> возвращает итоги по токенам, разбивку по моделям и пользователям. Отключается через <code>DANYAPI_USAGE_ENABLED=0</code>.",
            cta_title: "Бесплатные модели.",
            cta_title_hi: "Ваш API.",
            cta_sub: "Установите прямо сейчас, поставьте звезду и подпишитесь на канал.",
            cta_gh: "GitHub",
            cta_tg: "Телеграм-канал",
            footer_creator: "Создатель",
            footer_channel: "Телеграм-канал",
            footer_note: "Сделано на FastAPI и Python · реверс-инжиниринг, не аффилировано с DeepSeek, Alibaba, Sber или Яндекс",
            hosted_eyebrow: "Публичный инстанс",
            hosted_title: "Нет сервера?",
            hosted_title_hi: "Берите бесплатный.",
            hosted_sub: "Публичный, полностью бесплатный инстанс DanyAPI уже работает в продакшене. Без регистрации, ключей и настройки - просто направьте на него свой клиент.",
            hosted_pane_api: "Базовый URL API",
            hosted_pane_site: "Лендинг",
            hosted_note: "Используйте любой OpenAI-совместимый клиент с вашим токеном провайдера в качестве API-ключа (режим BYOK). Инстанс работает на ваших токенах без комиссий и посредников.",
            meta_title: "DanyAPI Документация",
            lang_en: "Английский",
            lang_ru: "Русский"
        },
        en: {
            nav_features: "Features",
            nav_hosted: "Public",
            nav_models: "Models",
            nav_quickstart: "Quick start",
            nav_faq: "FAQ",
            hero_badge: "Free · Open source · OpenAI-compatible",
            hero_h1_b: "Zero cost.",
            hero_sub: "A free, OpenAI-compatible API that runs the DeepSeek and Qwen web clients, the official GigaChat API, the OpenCode Zen gateway and the unofficial Alice and Duck.ai endpoints server-side from your own provider tokens. No paid keys, no quotas.",
            hero_cta_start: "Get started",
            hero_cta_gh: "Star on GitHub",
            features_eyebrow: "Features",
            features_title: "Everything the paid APIs have.",
            features_title_hi: "None of the bills.",
            features_sub: "A drop-in replacement for the official OpenAI API - swap <code>base_url</code> and your existing client code just works.",
            f1_t: "OpenAI-compatible",
            f1_d: "<code>GET /v1/models</code> and <code>POST /v1/chat/completions</code> in the official format. Change <code>base_url</code>, keep your client.",
            f2_t: "Streaming",
            f2_d: "Live streaming with <code>data:</code> chunks and <code>data: [DONE]</code>, plus thinking traces as <code>reasoning_content</code>.",
            f3_t: "Thinking traces",
            f3_d: "DeepSeek and Qwen reasoning exposed to your app - streamed live, on both providers.",
            f4_t: "Tool calling",
            f4_d: "Emulated <code>tools</code>, <code>tool_choice</code> and <code>parallel_tool_calls</code> with <code>finish_reason: \"tool_calls\"</code> responses.",
            f5_t: "JSON mode",
            f5_d: "<code>response_format</code> with <code>json_object</code> and <code>json_schema</code> - structured output for your agents.",
            f7_t: "Web search",
            f7_d: "Turn on grounded, up-to-date answers with the <code>search</code> flag on DeepSeek models.",
            f8_t: "File attachments",
            f8_d: "Images and text files as base64 or data URIs. Vision, OCR and file analysis per model.",
            models_eyebrow: "Models",
            models_title: "Six providers.",
            models_title_hi: "One OpenAI API.",
            models_sub: "Route by model name - <code>deepseek-*</code>, <code>qwen*</code>, <code>GigaChat*</code>, <code>opencode/*</code>, <code>alice</code> or a Duck.ai model. All optional, all can run together.",
            models_ds_flash: "thinking · search · files · vision",
            models_ds_pro: "thinking · search · files · vision",
            models_ds_note: "Read live from the web client settings, so only the model types the account is granted are listed. Web search via the <code>search</code> flag; reasoning via the <code>-thinking</code> suffix or <code>thinking</code> flag.",
            models_qw_1: "top model",
            models_qw_2: "fast",
            models_qw_3: "fetched live from the account",
            models_qw_note: "Thinking and search built in. The model list is read from the account and refetched on a timer, so new models appear automatically.",
            models_gc_lite: "lite · text",
            models_gc_pro: "tools · vision",
            models_gc_max: "tools · vision",
            models_gc_note: "Official GigaChat API with a free freemium quota. Needs a Studio authorization key. The list is read from the account and refetched on a timer. Images work on Pro, Max and Ultra only.",
            models_oc_gpt: "tools · vision",
            models_oc_claude: "tools · vision",
            models_oc_kimi: "tools",
            models_oc_note_list: "fetched live from the gateway",
            models_oc_note: "The curated OpenCode gateway at opencode.ai/zen, metered per token. Needs an API key. Some ids collide with Qwen or DeepSeek, prefix those with <code>opencode/</code>. Only the chat completions half of the catalogue is served.",
            models_al_keyless: "no key needed",
            models_al_note: "Unofficial, opt-in via <code>ALICE_ENABLED=1</code>, off by default. Yandex has no public API here, so it uses an undocumented internal protocol that can break at any time. Stateless and no incremental text, so history is folded into one prompt. At least one message is required, tool calls are not supported, and <code>max_tokens</code> trims the answer.",
            models_duck_keyless: "no key needed",
            models_duck_note: "Unofficial, opt-in via <code>DUCKAI_ENABLED=1</code>, off by default. DuckDuckGo has no public API here, so it uses an undocumented internal protocol and a browser fingerprint attestation that can break at any time. Needs Node.js on the host. Refuses datacenter addresses outright, so run it from a residential or mobile line.",
            models_mistral_keyless: "no key needed",
            models_mistral_note: "Unofficial, opt-in via <code>MISTRAL_ENABLED=1</code> plus <code>MISTRAL_LOGINS</code> with <code>email:password</code> pairs, off by default. Mistral has no public API here, so the provider speaks the Le Chat mobile flow with an account session. Free accounts are rate limited per message count. Can break at any time.",
            qs_eyebrow: "Quick start",
            qs_need: "All you need is a free DeepSeek or Qwen token, a GigaChat Studio authorization key, or an OpenCode Zen API key - the script does the rest.",
            qs_title: "Up and running in",
            qs_title_hi: "under a minute.",
            qs_sub: "One-command install on Windows, Linux and macOS. No Python setup gymnastics required.",
            qs_tab_install: "Install",
            qs_tab_docker: "Docker",
            qs_pane_install: "PowerShell",
            qs_pane_install_alt: "Linux / macOS",
            qs_install_note: "The script clones the repo, installs dependencies, creates <code>.env</code>, live-checks your provider tokens and tells you how to start the server. It even auto-updates itself on each start. Prefer not to dig tokens out of browser storage by hand? Run <code>docs/token_utility.sh</code> (or <code>docs\\token_utility.bat</code> on Windows) and it pulls them out of your browser for you.",
            qs_docker_note: "Prebuilt image, pushed on every push to prod and dev plus every version tag. The native PoW solver is compiled into the image for maximum speed.",
            copy: "Copy",
            faq_eyebrow: "FAQ",
            faq_title: "Questions?",
            faq_title_hi: "Answered.",
            q1: "Is it really free?",
            a1: "Yes. DanyAPI uses the internal APIs of the free web clients chat.deepseek.com and chat.qwen.ai through accounts made from your own free provider tokens, plus the free tier of the official GigaChat API and the OpenCode Zen gateway. The unofficial Alice and Duck.ai providers need no credentials at all. No billing, no quotas.",
            q2: "Do my users need an API key?",
            a2: "When running DanyAPI locally or in private hosting, no API key is required (pass any dummy value). When using the public hosted instance (BYOK mode), provide your DeepSeek or Qwen token as the Bearer token.",
            q3: "Which providers and models?",
            a3: "Model lists are not hardcoded: DeepSeek (<code>default</code> and whatever other types the account is granted), Qwen (<code>qwen3.8-max</code>, <code>qwen3.7-plus</code>, ...), GigaChat (<code>GigaChat-2</code>, <code>GigaChat-2-Pro</code> and the rest), OpenCode Zen (<code>opencode/gpt-5.6-sol</code>, <code>opencode/kimi-k3</code>, ...) and Duck.ai (the free tier models) are read from the provider endpoints and refetched on a timer, while Alice (<code>alice</code>, <code>yagpt</code>) serves its own aliases. Route by model name; all of them can run at once.",
            q7: "Are there rate limits or will my token get banned?",
            a7: "Requests go through the free web clients at a human-like pace. Add more tokens to the pool for extra parallelism - the project stays within normal usage, but treat free tokens as best-effort.",
            q8: "Is there usage tracking?",
            a8: "<code>GET /v1/usage</code> returns token usage totals, per-model and per-user breakdowns. Disable with <code>DANYAPI_USAGE_ENABLED=0</code>.",
            cta_title: "Free models.",
            cta_title_hi: "Your API.",
            cta_sub: "Install it now, or drop a star and follow the channel.",
            cta_gh: "GitHub",
            cta_tg: "Telegram channel",
            footer_creator: "Creator",
            footer_channel: "Telegram channel",
            footer_note: "Built with FastAPI &amp; Python · reverse-engineered, not affiliated with DeepSeek, Alibaba, Sber or Yandex",
            hosted_eyebrow: "Public instance",
            hosted_title: "No server?",
            hosted_title_hi: "Use the free one.",
            hosted_sub: "A public, fully free DanyAPI instance is already live in production. No signup, no keys, no setup - just point your client at it.",
            hosted_pane_api: "API base URL",
            hosted_pane_site: "Landing page",
            hosted_note: "Use any OpenAI-compatible client with your provider token as the API key (BYOK mode). The instance runs on your provider tokens with zero middleman fees.",
            meta_title: "DanyAPI Documentation",
            lang_en: "English",
            lang_ru: "Russian",
        }
    };

    function detectLang() {
        try {
            var saved = localStorage.getItem(STORE_KEY);
            if (saved && LANGS.indexOf(saved) !== -1) return saved;
        } catch (e) {}
        var browserLang = (navigator.language || navigator.userLanguage || "").toLowerCase();
        return browserLang.indexOf("ru") === 0 ? "ru" : "en";
    }

    function escapeText(value) {
        return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function leadingText(markup) {
        var cut = markup.indexOf("<");
        return cut === -1 ? markup : markup.slice(0, cut);
    }

    function translateNode(el, t) {
        var key = el.getAttribute("data-i18n");
        var original = el.getAttribute("data-i18n-src");
        if (original === null) {
            original = el.innerHTML;
            el.setAttribute("data-i18n-src", original);
        }
        var value = t ? t[key] : undefined;
        if (!/data-i18n/.test(original)) {
            if (value !== undefined) el.innerHTML = value;
            return;
        }
        var head = value === undefined ? leadingText(original) : value;
        var cut = original.indexOf("<");
        var tail = cut === -1 ? "" : original.slice(cut);
        el.innerHTML = head.indexOf("<") === -1 ? escapeText(head) + tail : head + tail;
        var nested = el.querySelectorAll("[data-i18n]");
        for (var i = 0; i < nested.length; i++) translateNode(nested[i], t);
    }

    function applyLang(lang) {
        var t = I18N[lang];
        document.documentElement.lang = lang;

        var els = document.querySelectorAll("[data-i18n], [data-i18n-aria]");
        for (var i = 0; i < els.length; i++) {
            var el = els[i];
            if (el.hasAttribute("data-i18n-aria")) {
                var ariaKey = el.getAttribute("data-i18n-aria");
                if (t && t[ariaKey] !== undefined) el.setAttribute("aria-label", t[ariaKey]);
            }
            if (!el.hasAttribute("data-i18n")) continue;
            if (el.parentElement && el.parentElement.closest("[data-i18n]")) continue;
            translateNode(el, t);
        }
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
            requestAnimationFrame(function () {
                requestAnimationFrame(function () {
                    main.classList.add("i18n-swap");
                });
            });
        }
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

    var MODEL_LIST_MAX = 6;

    function renderModelList(list, models, labelKey, labelText) {
        if (!models.length) return;
        list.innerHTML = "";
        models.slice(0, MODEL_LIST_MAX).forEach(function (model) {
            var li = document.createElement("li");
            var code = document.createElement("code");
            code.textContent = model.id;
            li.appendChild(code);
            var chip = document.createElement("span");
            chip.className = "model-chip";
            chip.setAttribute("data-i18n", labelKey);
            if (I18N.en[labelKey] !== undefined) chip.setAttribute("data-i18n-src", I18N.en[labelKey]);
            chip.textContent = labelText;
            li.appendChild(chip);
            list.appendChild(li);
        });
        if (models.length > MODEL_LIST_MAX) {
            var more = document.createElement("li");
            var rest = document.createElement("span");
            rest.textContent = models.slice(MODEL_LIST_MAX).map(function (m) { return m.id; }).join(", ");
            more.appendChild(rest);
            list.appendChild(more);
        }
    }

    function fetchLiveModels() {
        var lists = document.querySelectorAll("[data-model-provider]");
        if (!lists.length) return;
        var t = I18N[currentLang] || I18N.en;
        var en = I18N.en;
        function label(key) {
            if (t && t[key] !== undefined) return t[key];
            if (en[key] !== undefined) return en[key];
            return "";
        }
        fetch("/v1/models")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (!data || !data.data) return;
                lists.forEach(function (list) {
                    var provider = list.getAttribute("data-model-provider");
                    var models = data.data.filter(function (m) { return m.owned_by === provider; });
                    if (!models.length) return;
                    if (provider === "deepseek" || provider === "duckai") {
                        renderModelList(list, models, "models_duck_keyless", label("models_duck_keyless"));
                    } else if (provider === "alice") {
                        renderModelList(list, models, "models_al_keyless", label("models_al_keyless"));
                    } else if (provider === "gigachat") {
                        renderModelList(list, models, "models_gc_pro", label("models_gc_pro"));
                    } else if (provider === "opencode") {
                        renderModelList(list, models, "models_oc_note_list", label("models_oc_note_list"));
                    } else {
                        renderModelList(list, models, "models_qw_3", label("models_qw_3"));
                    }
                });
            })
            .catch(function () {});
    }

    var currentLang = detectLang();
    applyLang(currentLang);
    fetchGitHubStars();
    fetchLiveModels();

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
