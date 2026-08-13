FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY danyapi ./danyapi

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && gcc -O2 -o danyapi/deepseek/pow_solver danyapi/deepseek/pow_solver.c \
    && apt-get purge -y gcc libc6-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ENV DANYAPI_HOST=0.0.0.0
ENV DANYAPI_PORT=8000

EXPOSE 8000

CMD ["python", "-m", "danyapi"]
