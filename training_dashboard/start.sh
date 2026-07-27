#!/usr/bin/env bash
set -euo pipefail

DASHBOARD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_ROOT="${TRAINING_ROOT:-/root/autodl-tmp/FuckAnthropic/rfdetr_runs}"
DASHBOARD_PORT="${DASHBOARD_PORT:-6006}"
PID_FILE="${DASHBOARD_DIR}/dashboard.pid"
LOG_FILE="${DASHBOARD_DIR}/dashboard.log"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "Dashboard is already running (PID ${existing_pid})."
    exit 0
  fi
fi

cd "${DASHBOARD_DIR}"
nohup python server.py \
  --training-root "${TRAINING_ROOT}" \
  --host 127.0.0.1 \
  --port "${DASHBOARD_PORT}" \
  >"${LOG_FILE}" 2>&1 &
echo "$!" >"${PID_FILE}"

echo "Dashboard started (PID $(cat "${PID_FILE}"), port ${DASHBOARD_PORT})."
