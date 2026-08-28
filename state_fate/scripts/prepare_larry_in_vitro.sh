#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
DATASET="${DATASET:-in_vitro}"
RAW_ROOT="${RAW_ROOT:-${ROOT_DIR}/state_fate/data/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-${ROOT_DIR}/state_fate/data/processed/larry_in_vitro_d2_d6}"

cd "${ROOT_DIR}"
PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" state_fate/data_utils/prepare_larry.py \
  --dataset "${DATASET}" \
  --raw-root "${RAW_ROOT}" \
  --out-dir "${PROCESSED_DIR}" \
  --early-day "${EARLY_DAY:-2}" \
  --late-day "${LATE_DAY:-6}" \
  --n-hvgs "${N_HVGS:-2000}" \
  --latent-dim "${LATENT_DIM:-64}" \
  --pairs-per-clone "${PAIRS_PER_CLONE:-32}" \
  --seed "${SEED:-0}" \
  "$@"

