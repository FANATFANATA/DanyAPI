# DanyAPI: OpenAI-совместимый API поверх chat.deepseek.com
# Универсальный образ: работает на HF Spaces (порт 7860) и любом VPS.
# Токены задаются env-переменной DEEPSEEK_TOKENS (секрет на платформе).

FROM python:3.12-slim

WORKDIR /app

# gcc нужен только чтобы собрать нативный PoW-солвер
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY danyapi ./danyapi

# Сборка нативного солвера под Linux (бинарник без расширения .exe)
RUN cc -O2 -o danyapi/deepseek/pow_solver danyapi/deepseek/pow_solver.c

ENV DANYAPI_HOST=0.0.0.0
ENV DANYAPI_PORT=7860

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "danyapi.api.openai:app", "--host", "0.0.0.0", "--port", "7860"]
