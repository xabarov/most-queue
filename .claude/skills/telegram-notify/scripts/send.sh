#!/usr/bin/env bash
# Send a Telegram text message to the owner (agent notifications).
# Usage:
#   send.sh "короткий текст"
#   send.sh <<'EOF'
#   многострочный
#   текст
#   EOF
# Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from the repo .env.
# Never print the token; only the API response is surfaced on failure.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ENV_FILE="${ROOT}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "telegram-notify: .env not found at ${ENV_FILE}" >&2
  exit 2
fi

read_env_var() {
  local key="$1" value=""
  value="$(grep -E "^${key}=" "${ENV_FILE}" | tail -n 1 | cut -d= -f2- \
    | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
          -e 's/^"//' -e 's/"$//' | tr -d '\r' || true)"
  printf '%s' "${value}"
}

TOKEN="$(read_env_var TELEGRAM_BOT_TOKEN)"
CHAT_ID="$(read_env_var TELEGRAM_CHAT_ID)"

if [[ -z "${TOKEN}" || -z "${CHAT_ID}" ]]; then
  echo "telegram-notify: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set in .env" >&2
  exit 2
fi

TEXT="${1:-}"
if [[ -z "${TEXT}" ]]; then
  TEXT="$(cat)"
fi

if [[ -z "${TEXT}" ]]; then
  echo "telegram-notify: empty message" >&2
  exit 2
fi

send_once() {
  curl -sS --max-time 15 -X POST \
    "https://api.telegram.org/bot${TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${CHAT_ID}" \
    --data-urlencode "text=${TEXT}"
}

RESP="$(send_once)"
if [[ "${RESP}" != *'"ok":true'* ]]; then
  RESP="$(send_once)"
fi

if [[ "${RESP}" != *'"ok":true'* ]]; then
  echo "telegram-notify: send failed: ${RESP}" >&2
  exit 1
fi

echo "telegram-notify: delivered"
