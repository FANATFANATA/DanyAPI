#!/data/data/com.termux/files/usr/bin/bash
# DanyAPI: автозапуск + публичный туннель на Termux (термукс-сервис)
# Запуск: bash termux_service.sh
set -e

echo "==> termux-services (автозапуск сервисов при включении телефона)"
pkg install -y termux-services

mkdir -p "$PREFIX/var/service/danyapi"
cat > "$PREFIX/var/service/danyapi/run" <<'EOF'
#!/data/data/com.termux/files/usr/bin/bash
exec 2>&1
cd "$HOME/DanyAPI"
exec python -m uvicorn danyapi.api.openai:app --host 0.0.0.0 --port 8000
EOF
chmod +x "$PREFIX/var/service/danyapi/run"

echo "==> cloudflared (публичный https-туннель)"
pkg install -y cloudflared

echo "==> Запуск сервиса danyapi"
sv-enable danyapi
sv up danyapi

sleep 3
echo "Локально: http://127.0.0.1:8000  (в локальной сети: http://<ip-телефона>:8000)"

echo ""
echo "Публичный адрес (временный trycloudflare URL, меняется при перезапуске):"
cloudflared tunnel --url http://localhost:8000
