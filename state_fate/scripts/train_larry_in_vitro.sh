#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
PROCESSED_DIR="${PROCESSED_DIR:-${ROOT_DIR}/state_fate/data/processed/larry_in_vitro_d2_d6}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/state_fate/runs/larry_in_vitro}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/state_fate/data/mpl_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/state_fate/data/xdg_cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

cd "${ROOT_DIR}"
PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" state_fate/train.py \
  --data-root "${PROCESSED_DIR}" \
  --out-dir "${RUN_ROOT}" \
  --steps "${STEPS:-20000}" \
  --batch-size "${BATCH_SIZE:-512}" \
  --test-batch-size "${TEST_BATCH_SIZE:-512}" \
  --hidden-dim "${HIDDEN_DIM:-512}" \
  --num-blocks "${NUM_BLOCKS:-6}" \
  --lr "${LR:-2e-4}" \
  --K "${SDE_K:-0.5}" \
  --log-every "${LOG_EVERY:-100}" \
  --eval-every "${EVAL_EVERY:-1000}" \
  --ckpt-every "${CKPT_EVERY:-5000}" \
  --eval-num-steps "${EVAL_NUM_STEPS:-250}" \
  --seed "${SEED:-0}" \
  "$@"
