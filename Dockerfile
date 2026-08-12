FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY danyapi ./danyapi

ENV DANYAPI_HOST=0.0.0.0
ENV DANYAPI_PORT=8000

EXPOSE 8000

CMD ["python", "-m", "danyapi"]
