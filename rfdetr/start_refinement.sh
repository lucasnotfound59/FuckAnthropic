#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/autodl-tmp/FuckAnthropic"
DATASET_DIR="${PROJECT_DIR}/rfdetr_dataset_v2"
OUTPUT_DIR="${PROJECT_DIR}/rfdetr_runs/rfdetr_2xl_1360_refine"
WEIGHTS="${PROJECT_DIR}/rfdetr_runs/snapshots/refinement_source_best_ema.pth"
LOG_DIR="${PROJECT_DIR}/rfdetr_runs/logs"
SCREEN_NAME="rfdetr2xl_refine"

if [[ "${ACCEPT_RFDETR_PML:-}" != "YES" ]]; then
  echo "Set ACCEPT_RFDETR_PML=YES only after accepting PML-1.0."
  exit 2
fi
if [[ ! -f "${DATASET_DIR}/data.yaml" ]]; then
  echo "Missing ${DATASET_DIR}/data.yaml"
  exit 3
fi
if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Missing refinement source checkpoint: ${WEIGHTS}"
  exit 4
fi
if pgrep -af "rfdetr/train.py" >/dev/null; then
  echo "Another RF-DETR training process is already running."
  exit 5
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR}/refine_${timestamp}.log"

cd "${PROJECT_DIR}"
screen -dmS "${SCREEN_NAME}" bash -lc \
  "set -o pipefail; cd '${PROJECT_DIR}'; \
  python -u rfdetr/train.py \
    --accept-pml \
    --dataset '${DATASET_DIR}' \
    --output '${OUTPUT_DIR}' \
    --pretrain-weights '${WEIGHTS}' \
    --epochs 60 \
    --lr 4e-5 \
    --lr-encoder 1e-5 \
    --warmup-epochs 1 \
    --weight-decay 1e-4 \
    --drop-path 0.05 \
    --smooth-alpha 0.005 \
    --skip-best-epochs 2 \
    --patience 12 \
    --target-f1 0.95 \
    --checkpoint-interval 5 \
    2>&1 | tee '${log_file}'"

echo "RF-DETR refinement started in screen '${SCREEN_NAME}'."
echo "Log: ${log_file}"
