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
            hero_sub: "DanyAPI - бесплатный OpenAI-совместимый API, который запускает веб-клиенты DeepSeek и Qwen, официальный API GigaChat и неофициальные Alice и Duck.ai на сервере из ваших бесплатных токенов. Без платных ключей и лимитов.",
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
            models_title: "Четыре провайдера.",
            models_title_hi: "Один OpenAI API.",
            models_sub: "Маршрутизация по имени модели - <code>deepseek-*</code>, <code>qwen*</code>, <code>GigaChat*</code>, <code>alice</code> или модель Duck.ai. Все провайдеры опциональны и работают вместе.",
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
            models_al_keyless: "без ключа",
            models_al_note: "Неофициальный провайдер, включается через <code>ALICE_ENABLED=1</code>, по умолчанию выключен. У Яндекса нет публичного API, поэтому используется недокументированный внутренний протокол, который может отвалиться в любой момент. Без состояния и без потокового текста, поэтому история сворачивается в один промпт.",
            models_duck_keyless: "без ключа",
            models_duck_note: "Неофициальный провайдер, включается через <code>DUCKAI_ENABLED=1</code>, по умолчанию выключен. У DuckDuckGo нет публичного API, поэтому используется недокументированный внутренний протокол с проверкой отпечатка браузера, который может отвалиться в любой момент. Нужен Node.js на хосте. Прямо отказывает дата-центровым адресам, так что запускайте с домашней или мобильной линии.",
            qs_eyebrow: "Быстрый старт",
            qs_need: "Нужен только бесплатный токен DeepSeek или Qwen, либо ключ GigaChat из Studio - всё остальное сделает скрипт.",
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
            a3: "Списки моделей не захардкожены: DeepSeek (<code>default</code> и другие типы, которые выданы аккаунту), Qwen (<code>qwen3.8-max</code>, <code>qwen3.7-plus</code>, ...), GigaChat (<code>GigaChat-2</code>, <code>GigaChat-2-Pro</code> и другие) и Duck.ai (модели бесплатного тарифа) читаются с эндпоинтов провайдеров и обновляются по таймеру, Alice (<code>alice</code>, <code>yagpt</code>) отдаёт свои алиасы. Маршрутизация по имени модели; все работают одновременно.",
            q7: "Есть ли лимиты или забанят токен?",
            a7: "Запросы идут через бесплатные веб-клиенты в человекоподобном темпе. Добавьте больше токенов в пул для параллелизма - проект держится в рамках нормального использования, но бесплатные токены - best-effort.",
            q8: "Есть ли учёт использования?",
            a8: "<code>GET /v1/usage</code> возвращает итоги по токенам, разбивку по моделям и пользователям. Отключается через <code>DANYAPI_USAGE_ENABLED=0</code>.",
            cta_title: "Бесплатные модели.",
            cta_sub: "Ваш API.",
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
            meta_title: "DanyAPI Documentation",
            nav_hosted: "Public",
            hosted_eyebrow: "Public instance",
            hosted_title: "No server?",
            hosted_title_hi: "Use the free one.",
            hosted_sub: "A public, fully free DanyAPI instance is already live in production. No signup, no keys, no setup - just point your client at it.",
            hosted_pane_api: "API base URL",
            hosted_pane_site: "Landing page",
            hosted_note: "Use any OpenAI-compatible client with your provider token as the API key (BYOK mode). The instance runs on your provider tokens with zero middleman fees."
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

    function applyLang(lang) {
        var t = I18N[lang];
        document.documentElement.lang = lang;

        var els = document.querySelectorAll("[data-i18n], [data-i18n-aria]");
        for (var i = 0; i < els.length; i++) {
            var el = els[i];
            if (el.hasAttribute("data-i18n")) {
                var key = el.getAttribute("data-i18n");
                if (t && t[key] !== undefined) el.innerHTML = t[key];
            }
            if (el.hasAttribute("data-i18n-aria")) {
                var ariaKey = el.getAttribute("data-i18n-aria");
                if (t && t[ariaKey] !== undefined) el.setAttribute("aria-label", t[ariaKey]);
            }
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
        fetch("/v1/models")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (!data || !data.data) return;
                lists.forEach(function (list) {
                    var provider = list.getAttribute("data-model-provider");
                    var models = data.data.filter(function (m) { return m.owned_by === provider; });
                    if (!models.length) return;
                    if (provider === "deepseek" || provider === "duckai") {
                        renderModelList(list, models, "models_duck_keyless", t.models_duck_keyless);
                    } else if (provider === "alice") {
                        renderModelList(list, models, "models_al_keyless", t.models_al_keyless);
                    } else if (provider === "gigachat") {
                        renderModelList(list, models, "models_gc_pro", t.models_gc_pro);
                    } else {
                        renderModelList(list, models, "models_qw_3", t.models_qw_3);
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
