(function () {
    "use strict";
    var nav = document.getElementById("nav");
    function onScroll() {
        nav.classList.toggle("scrolled", window.scrollY > 20);
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
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
    var LANGS = ["en", "ru"];
    var STORE_KEY = "danyapi-lang";

    var I18N = {
        ru: {
            nav_features: "Возможности",
            nav_models: "Модели",
            nav_quickstart: "Быстрый старт",
            nav_faq: "FAQ",
            hero_badge: "Бесплатно · Open Source · OpenAI-совместимо",
            hero_h1_b: "Ноль затрат.",
            hero_sub: "DanyAPI - бесплатный OpenAI-совместимый API, который запускает веб-клиенты DeepSeek и Qwen на сервере из ваших бесплатных токенов. Без платных ключей и лимитов.",
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
            models_title: "Два бесплатных провайдера.",
            models_title_hi: "Один OpenAI API.",
            models_sub: "Маршрутизация по имени модели - <code>deepseek-*</code> или <code>qwen*</code>. Оба провайдера опциональны и работают вместе.",
            models_ds_flash: "по умолчанию · поиск · OCR",
            models_ds_pro: "эксперт · размышления",
            models_ds_vision: "vision",
            models_ds_note: "Размышления на всех моделях. Веб-поиск на flash. Файлы: vision = изображения, flash = изображения + текст, pro = нет.",
            models_qw_1: "топ-модель",
            models_qw_2: "быстрая",
            models_qw_3: "подтягиваются из аккаунта",
            models_qw_note: "Размышления и поиск встроены. Список моделей забирается из аккаунта при старте - новые появляются автоматически.",
            qs_eyebrow: "Быстрый старт",
            qs_need: "Нужен только бесплатный токен DeepSeek или Qwen - всё остальное сделает скрипт.",
            qs_title: "Запуск за",
            qs_title_hi: "меньше минуты.",
            qs_sub: "Установка одной командой на Windows, Linux и macOS.",
            qs_tab_install: "Установка",
            qs_tab_docker: "Docker",
            qs_pane_install: "PowerShell",
            qs_pane_install_alt: "Linux / macOS",
            qs_install_note: "Скрипт клонирует репозиторий, ставит зависимости, создаёт <code>.env</code>, проверяет токены и подсказывает, как запустить сервер. Обновляется сам при каждом старте.",
            qs_docker_note: "Готовый образ, пушится на каждый коммит и тег версии. Нативный PoW-солвер собран в образ для максимальной скорости.",
            copy: "Копировать",
            faq_eyebrow: "FAQ",
            faq_title: "Вопросы?",
            faq_title_hi: "Есть ответы.",
            q1: "Это правда бесплатно?",
            a1: "Да. DanyAPI использует внутренние API бесплатных веб-клиентов через аккаунты из ваших бесплатных токенов. Никаких тарифов и лимитов.",
            q2: "Нужен ли пользователям API-ключ?",
            a2: "Нет. SDK требует <code>api_key</code>, но DanyAPI его не проверяет - передайте любое значение. Все запросы делают ваши серверные аккаунты.",
            q3: "Какие провайдеры и модели?",
            a3: "DeepSeek (<code>deepseek-v4-flash</code>, <code>deepseek-v4-pro</code>, <code>deepseek-v4-vision</code>) и Qwen (<code>qwen3.8-max</code>, <code>qwen3.7-plus</code>, … - подтягиваются из аккаунта). Маршрутизация по имени модели; оба работают одновременно.",
            q7: "Есть ли лимиты или забанят токен?",
            a7: "Запросы идут через бесплатные веб-клиенты в человекоподобном темпе. Добавьте больше токенов в пул для параллелизма - проект держится в рамках нормального использования, но бесплатные токены - best-effort.",
            cta_title: "Бесплатные модели.",
            cta_sub: "Ваш API.",
            cta_gh: "GitHub",
            cta_tg: "Телеграм-канал",
            footer_creator: "Создатель",
            footer_channel: "Телеграм-канал",
            footer_note: "Сделано на FastAPI и Python · реверс-инжиниринг, не аффилировано с DeepSeek или Alibaba",
            meta_title: "DanyAPI Документация",
            lang_en: "Английский",
            lang_ru: "Русский"
        },
        en: {
            meta_title: "DanyAPI Documentation"
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
