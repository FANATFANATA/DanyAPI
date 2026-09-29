FROM python:3.14.2-slim-bookworm@sha256:e87711ef5c86aaeaa7031718a69db79d334d94c545c709583f651b8185870941

WORKDIR /app

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev nodejs \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

COPY danyapi ./danyapi
COPY web ./web
COPY docs/index.html ./docs/index.html
COPY docs/style.css ./docs/style.css
COPY docs/script.js ./docs/script.js
COPY docs/deepseek-logo.svg ./docs/deepseek-logo.svg
COPY docs/qwen-logo.svg ./docs/qwen-logo.svg

RUN gcc -O3 -pthread -funroll-loops -flto -fomit-frame-pointer -o danyapi/deepseek/pow_solver danyapi/deepseek/pow_solver.c \
    && apt-get purge -y gcc libc6-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ENV DANYAPI_HOST=0.0.0.0
ENV DANYAPI_PORT=8000

RUN groupadd --gid 10001 danyapi \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin danyapi \
    && mkdir -p /tmp/danyapi \
    && chown -R 10001:10001 /app /tmp/danyapi

EXPOSE 8000

USER 10001:10001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('DANYAPI_PORT', '8000') + '/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "danyapi"]
