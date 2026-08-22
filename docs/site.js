(function () {
    var I18N = window.DANYAPI_I18N;

    var THEME_KEY = 'danyapi-theme';
    var THEMES = ['black', 'hacker', 'monokai'];

    function readTheme() {
        try { return localStorage.getItem(THEME_KEY); } catch (e) { return null; }
    }
    function writeTheme(t) {
        try { localStorage.setItem(THEME_KEY, t); } catch (e) {}
    }
    function themeCurrent() {
        var t = document.documentElement.getAttribute('data-theme');
        return THEMES.indexOf(t) !== -1 ? t : 'black';
    }
    function applyTheme(t) {
        document.documentElement.setAttribute('data-theme', t);
        var sel = document.getElementById('theme-select');
        if (sel) sel.value = t;
    }

    function initTheme() {
        var saved = readTheme();
        applyTheme(THEMES.indexOf(saved) !== -1 ? saved : themeCurrent());
        var sel = document.getElementById('theme-select');
        if (sel) {
            sel.addEventListener('change', function () {
                applyTheme(sel.value);
                writeTheme(sel.value);
                var sw = document.querySelector('.theme-switcher');
                if (sw) {
                    sw.classList.remove('flash');
                    void sw.offsetWidth;
                    sw.classList.add('flash');
                    setTimeout(function () { sw.classList.remove('flash'); }, 450);
                }
            });
        }
    }

    function initDocsNav() {
        var nav = document.querySelector('.docs-nav');
        if (!nav) return;
        var btn = nav.querySelector('.docs-btn');
        var dd = nav.querySelector('.docs-dropdown');
        var path = (location.pathname.split('/').pop() || 'index.html');
        var matched = false;
        dd.querySelectorAll('a').forEach(function (a) {
            if (a.getAttribute('href') === path) {
                a.classList.add('active');
                matched = true;
            }
        });
        btn.classList.toggle('active', matched);
        function close() {
            nav.classList.remove('open');
            btn.setAttribute('aria-expanded', 'false');
        }
        btn.addEventListener('click', function (e) {
            e.stopPropagation();
            var open = nav.classList.toggle('open');
            btn.setAttribute('aria-expanded', open ? 'true' : 'false');
        });
        document.addEventListener('click', close);
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') close();
        });
    }

    function initLangButtons() {
        document.querySelectorAll('[data-lang-btn]').forEach(function (btn) {
            btn.addEventListener('click', function () {
                I18N.setLang(btn.getAttribute('data-lang-btn'));
            });
        });
    }

    function copyFallback(text, done) {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch (e) {}
        document.body.removeChild(ta);
        done();
    }

    function copyText(text, btn) {
        var settled = false;
        function doneSafe() {
            if (settled) return;
            settled = true;
            var label = I18N.t(undefined, 'copied') || 'Copied!';
            var old = btn.textContent;
            btn.textContent = label;
            setTimeout(function () { btn.textContent = old; }, 1600);
        }
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(doneSafe, function () { doneSafe(); copyFallback(text, function () {}); });
            setTimeout(function () { doneSafe(); copyFallback(text, function () {}); }, 900);
        } else {
            doneSafe();
            copyFallback(text, function () {});
        }
    }

    function initCopy() {
        document.querySelectorAll('[data-copy]').forEach(function (btn) {
            btn.addEventListener('click', function () {
                btn.style.animation = 'none';
                void btn.offsetWidth;
                btn.style.animation = 'click-bounce 0.25s ease';
                copyText(btn.getAttribute('data-copy'), btn);
            });
        });
    }

    var ENV_TEMPLATE = [
        '# DeepSeek provider: one or more bearer tokens, comma-separated',
        'DEEPSEEK_TOKENS=',
        '',
        '# Qwen provider: one or more bearer tokens, comma-separated',
        'QWEN_TOKENS=',
        '',
        '# address the API server binds to',
        'DANYAPI_HOST=0.0.0.0',
        '# port the API server listens on',
        'DANYAPI_PORT=8000',
        '# upstream request timeout in seconds',
        'DANYAPI_TIMEOUT=60',
        '# seconds to wait for a free account before returning 429 (empty = wait forever)',
        'DANYAPI_ACQUIRE_TIMEOUT=',
        '# max server-side chats cached per provider (LRU) for stateless session reuse',
        'DANYAPI_SESSION_CACHE_SIZE=128',
        '# seconds an unused session/context stays reusable (0 = never expire)',
        'DANYAPI_SESSION_TTL_SECONDS=3600',
        '# directory for on-disk session cache (default = system temp dir, e.g. %TEMP%\\danyapi)',
        'DANYAPI_CACHE_DIR=',
        '# set to 1/true/yes to disable on-disk cache entirely',
        'DANYAPI_CACHE_DISABLED=',
        '# log level: DEBUG, INFO, WARNING, ERROR',
        'DANYAPI_LOG_LEVEL=INFO',
        '# file path for persistent logs (empty = console only)',
        'DANYAPI_LOG_FILE=',
        '# max log file size in bytes before rotation',
        'DANYAPI_LOG_MAX_BYTES=10485760',
        '# number of rotated log files to keep',
        'DANYAPI_LOG_BACKUP_COUNT=3',
        '# minimum delay in seconds before sending a request (uniform random jitter)',
        'DANYAPI_HUMAN_DELAY_MIN=0.5',
        '# maximum delay in seconds; set both to 0 to disable',
        'DANYAPI_HUMAN_DELAY_MAX=3.0',
        '# auto-update from the latest GitHub release on each start (0 disables)',
        'DANYAPI_AUTO_UPDATE=1',
        ''
    ].join('\n');

    function downloadEnv() {
        var blob = new Blob([ENV_TEMPLATE], { type: 'text/plain;charset=utf-8' });
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = '.env';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
    }

    function initDownload() {
        document.querySelectorAll('[data-download-env]').forEach(function (btn) {
            btn.addEventListener('click', downloadEnv);
        });
    }

    var TERM = {
        en: [
            { type: 'cmd', text: 'irm https://raw.githubusercontent.com/FANATFANATA/DanyAPI/main/docs/install.ps1 | iex' },
            { type: 'ok', text: 'Cloning DanyAPI into $HOME\\DanyAPI' },
            { type: 'ok', text: 'Python 3.10+ detected' },
            { type: 'ok', text: '.env created from .env.example' },
            { type: 'ok', text: 'Installing requirements.txt ... done' },
            { type: 'ask', text: 'DeepSeek tokens (comma-separated): ' },
            { type: 'ask', text: 'Qwen tokens (comma-separated): ' },
            { type: 'ok', text: 'Tokens validated against the providers' },
            { type: 'ok', text: 'DanyAPI shortcut created on the desktop' },
            { type: 'ok', text: 'Auto-update: checking the latest GitHub release' },
            { type: 'info', text: 'Next: double-click the shortcut to start the server' }
        ],
        ru: [
            { type: 'cmd', text: 'irm https://raw.githubusercontent.com/FANATFANATA/DanyAPI/main/docs/install.ps1 | iex' },
            { type: 'ok', text: 'Клонирование DanyAPI в ~\\DanyAPI' },
            { type: 'ok', text: 'Python 3.10+ найден' },
            { type: 'ok', text: '.env создан из .env.example' },
            { type: 'ok', text: 'Установка requirements.txt ... готово' },
            { type: 'ask', text: 'DeepSeek токены (через запятую): ' },
            { type: 'ask', text: 'Qwen токены (через запятую): ' },
            { type: 'ok', text: 'Токены проверены у провайдеров' },
            { type: 'ok', text: 'Ярлык DanyAPI создан на рабочем столе' },
            { type: 'ok', text: 'Автообновление: проверка последнего релиза' },
            { type: 'info', text: 'Дальше: запусти ярлык' }
        ]
    };

    var termVersion = 0;

    var tocItems = [];

    function initScrollReveal() {
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

        var selectors = [
            { sel: '.reveal, .reveal-left, .reveal-right, .reveal-scale, .reveal-code', cls: 'revealed' },
            { sel: '.reveal-stagger', cls: 'revealed' }
        ];

        var all = [];
        selectors.forEach(function (s) {
            document.querySelectorAll(s.sel).forEach(function (el) {
                all.push({ el: el, cls: s.cls });
            });
        });

        if (!all.length) return;

        if (!('IntersectionObserver' in window)) {
            all.forEach(function (item) { item.el.classList.add(item.cls); });
            return;
        }

        var observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    var item = all.find(function (i) { return i.el === entry.target; });
                    if (item) {
                        setTimeout(function () {
                            item.el.classList.add(item.cls);
                        }, 50);
                    }
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

        all.forEach(function (item) { observer.observe(item.el); });
    }

    function autoMarkReveal() {
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

        var main = document.querySelector('.main');
        if (!main) return;

        var container = main.querySelector('.container');
        if (!container) return;

        Array.prototype.forEach.call(container.children, function (el) {
            if (el.classList.contains('reveal') || el.classList.contains('reveal-left') ||
                el.classList.contains('reveal-right') || el.classList.contains('reveal-scale') ||
                el.classList.contains('reveal-stagger') || el.classList.contains('reveal-code') ||
                el.tagName === 'HR') return;

            var tag = el.tagName.toLowerCase();

            if (tag === 'h1' || tag === 'h2' || tag === 'h3' || tag === 'h4') {
                el.classList.add('reveal');
            } else if (tag === 'p' || tag === 'blockquote') {
                el.classList.add('reveal');
            } else if (tag === 'ul' || tag === 'ol') {
                el.classList.add('reveal-stagger');
            } else if (tag === 'div' && (el.classList.contains('table-wrap') || el.classList.contains('code-header'))) {
                el.classList.add('reveal-scale');
            } else if (tag === 'pre') {
                el.classList.add('reveal-code');
            } else if (tag === 'div') {
                el.classList.add('reveal');
            }
        });
    }

    function buildToc() {
        tocItems = [];
        var aside = document.querySelector('.toc-aside');
        var main = document.querySelector('.main');
        var list = document.querySelector('.toc-list');
        if (!aside || !main || !list) return;
        list.innerHTML = '';
        var heads = Array.prototype.filter.call(main.querySelectorAll('h2[id], h3[id], h4[id]'), function (h) {
            return h.closest('[data-toc-exclude]') === null;
        });
        if (heads.length < 2) {
            aside.classList.add('is-hidden');
            return;
        }
        aside.classList.remove('is-hidden');
        var uls = { 2: list };
        heads.forEach(function (h) {
            var level = parseInt(h.tagName.charAt(1), 10);
            if (!uls[level]) {
                var parentLevel = level - 1;
                var parentLi = uls[parentLevel].lastElementChild;
                var sub = document.createElement('ul');
                if (parentLi) {
                    parentLi.appendChild(sub);
                } else {
                    uls[2].appendChild(sub);
                }
                uls[level] = sub;
            }
            var li = document.createElement('li');
            var a = document.createElement('a');
            a.href = '#' + h.id;
            a.textContent = h.textContent;
            li.appendChild(a);
            uls[level].appendChild(li);
            tocItems.push({ link: a, heading: h });
        });
        updateToc();
    }

    function updateToc() {
        if (!tocItems.length) return;
        var pos = window.scrollY + 120;
        var current = null;
        tocItems.forEach(function (it) {
            if (it.heading.getBoundingClientRect().top + window.scrollY <= pos) current = it;
        });
        tocItems.forEach(function (it) {
            it.link.classList.toggle('active', it === current);
        });
    }

    function renderTerminal() {
        var term = document.querySelector('[data-term]');
        if (!term) return;
        var body = term.querySelector('.terminal-body');
        body.innerHTML = '';
        var lang = I18N.current();
        var lines = TERM[lang] || TERM.en;
        var v = ++termVersion;
        var i = 0;

        function next() {
            if (v !== termVersion) return;
            if (i >= lines.length) {
                term.classList.add('is-done');
                body.scrollTop = body.scrollHeight;
                return;
            }
            addLine(lines[i++]);
        }

        function addLine(line) {
            if (v !== termVersion) return;
            var div = document.createElement('div');
            div.className = 'term-line term-' + line.type;
            if (line.type === 'cmd') {
                var prompt = document.createElement('span');
                prompt.className = 'term-prompt';
                prompt.textContent = '$ ';
                div.appendChild(prompt);
                var span = document.createElement('span');
                span.className = 'term-text';
                div.appendChild(span);
                body.appendChild(div);
                var j = 0;
                var timer = setInterval(function () {
                    if (v !== termVersion) { clearInterval(timer); return; }
                    if (j < line.text.length) {
                        span.textContent += line.text[j];
                        j++;
                    } else {
                        clearInterval(timer);
                        setTimeout(next, 380);
                    }
                }, 42);
            } else {
                if (line.type === 'ask') {
                    div.textContent = line.text;
                    var cur = document.createElement('span');
                    cur.className = 'term-cursor';
                    cur.textContent = '█';
                    div.appendChild(cur);
                } else {
                    div.textContent = line.text;
                }
                body.appendChild(div);
                body.scrollTop = body.scrollHeight;
                setTimeout(next, 300);
            }
        }

        next();
    }

    function init() {
        initTheme();
        initDocsNav();
        initLangButtons();
        initCopy();
        initDownload();
        renderTerminal();
        buildToc();
        autoMarkReveal();
        initScrollReveal();
        var topbar = document.querySelector('.topbar');
        if (topbar) {
            window.addEventListener('scroll', function () {
                requestAnimationFrame(function () {
                    topbar.classList.toggle('scrolled', window.scrollY > 20);
                });
            }, { passive: true });
        }
        window.addEventListener('scroll', function () {
            requestAnimationFrame(updateToc);
        }, { passive: true });
        document.addEventListener('danyapi:lang', function () {
            renderTerminal();
            buildToc();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
