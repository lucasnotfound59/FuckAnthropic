#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/autodl-tmp/FuckAnthropic"
RUN_DIR="${PROJECT_DIR}/rfdetr_runs/rfdetr_2xl_1360_refine"
LOG_DIR="${PROJECT_DIR}/rfdetr_runs/submission_logs"
SCREEN_NAME="rfdetr_submit_watch"

if [[ "${ACCEPT_RFDETR_PML:-}" != "YES" ]]; then
  echo "Set ACCEPT_RFDETR_PML=YES only after accepting PML-1.0."
  exit 2
fi
if [[ ! -f "${RUN_DIR}/last.ckpt" ]]; then
  echo "Missing ${RUN_DIR}/last.ckpt"
  exit 3
fi
if screen -list 2>/dev/null | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "Submission watcher is already running in screen '${SCREEN_NAME}'."
  exit 0
fi

mkdir -p "${LOG_DIR}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR}/watcher_${timestamp}.log"

cd "${PROJECT_DIR}"
screen -dmS "${SCREEN_NAME}" bash -lc \
  "set -o pipefail; cd '${PROJECT_DIR}'; \
  python -u rfdetr/epoch_submission_watcher.py \
    --accept-pml \
    --project-root '${PROJECT_DIR}' \
    --run-dir '${RUN_DIR}' \
    --images '${PROJECT_DIR}/rfdetr_dataset_v2/test/images' \
    --thresholds '${PROJECT_DIR}/rfdetr/thresholds_epoch006.json' \
    2>&1 | tee '${log_file}'"

echo "Submission watcher started in screen '${SCREEN_NAME}'."
echo "Log: ${log_file}"
