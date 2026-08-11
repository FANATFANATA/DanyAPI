#!/data/data/com.termux/files/usr/bin/bash
# DanyAPI: установка на Termux (Android)
# Запуск: bash termux_setup.sh
set -e

echo "==> Обновление пакетов Termux"
pkg update -y && pkg upgrade -y

echo "==> Установка зависимостей (python, clang для PoW-солвера, git)"
pkg install -y python clang git

echo "==> Клонирование репозитория"
if [ ! -d "$HOME/DanyAPI" ]; then
  git clone https://github.com/FANATFANATA/DanyAPI.git "$HOME/DanyAPI"
fi
cd "$HOME/DanyAPI"
git pull || true

echo "==> Python-зависимости"
pip install --upgrade pip
pip install -r requirements.txt || pip install fastapi "uvicorn>=0.30" httpx "pydantic>=2.8" python-dotenv

echo "==> Сборка нативного PoW-солвера (aarch64)"
clang -O2 -o danyapi/deepseek/pow_solver danyapi/deepseek/pow_solver.c

echo "==> Файл .env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "ВАЖНО: отредактируй .env и впиши токены:"
  echo "  nano $HOME/DanyAPI/.env"
fi

echo ""
echo "Готово. Дальше:"
echo "  1) nano $HOME/DanyAPI/.env   -> вписать DEEPSEEK_TOKENS"
echo "  2) запуск: python -m uvicorn danyapi.api.openai:app --host 0.0.0.0 --port 8000"
echo "     (из папки $HOME/DanyAPI)"
echo "  3) публичный доступ: pkg install cloudflared && cloudflared tunnel --url http://localhost:8000"
