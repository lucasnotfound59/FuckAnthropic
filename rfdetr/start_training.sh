#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${RFDETR_SCREEN_NAME:-rfdetr2xl}"
LOG_DIR="${PROJECT_DIR}/rfdetr_runs/logs"

if [[ "${ACCEPT_RFDETR_PML:-}" != "YES" ]]; then
  cat >&2 <<'EOF'
RF-DETR-2XL requires acceptance of the Platform Model License 1.0.
Read: https://github.com/roboflow/rf-detr_plus/blob/main/LICENSE
If you accept it, rerun with:
  ACCEPT_RFDETR_PML=YES bash rfdetr/start_training.sh
EOF
  exit 2
fi

if [[ ! -f "${PROJECT_DIR}/rfdetr_dataset/data.yaml" ]]; then
  echo "Missing ${PROJECT_DIR}/rfdetr_dataset/data.yaml" >&2
  exit 3
fi

if screen -list | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
  echo "Training screen '${SESSION_NAME}' is already running." >&2
  exit 4
fi

if pgrep -af "python .*rfdetr/train.py" >/dev/null; then
  echo "An RF-DETR training process is already running." >&2
  pgrep -af "python .*rfdetr/train.py" >&2
  exit 5
fi

available_kb="$(df -Pk "${PROJECT_DIR}" | awk 'NR==2 {print $4}')"
if (( available_kb < 10 * 1024 * 1024 )); then
  echo "At least 10GB of free data-disk space is required." >&2
  exit 6
fi

mkdir -p "${LOG_DIR}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR}/train_${timestamp}.log"
command=(python -u rfdetr/train.py --accept-pml "$@")
printf -v quoted_command '%q ' "${command[@]}"
printf -v quoted_project '%q' "${PROJECT_DIR}"
printf -v quoted_log '%q' "${log_file}"

screen -dmS "${SESSION_NAME}" bash -lc \
  "set -o pipefail; cd ${quoted_project}; ${quoted_command} 2>&1 | tee ${quoted_log}"

sleep 1
if ! screen -list | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
  echo "Training failed to stay running. Inspect ${log_file}" >&2
  exit 7
fi

echo "RF-DETR training started in screen '${SESSION_NAME}'."
echo "Log: ${log_file}"
echo "Attach: screen -r ${SESSION_NAME}"
echo "Detach: Ctrl+A, then D"
