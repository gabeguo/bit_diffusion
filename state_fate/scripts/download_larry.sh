#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
DATASET="${DATASET:-in_vitro}"
RAW_ROOT="${RAW_ROOT:-${ROOT_DIR}/state_fate/data/raw}"

cd "${ROOT_DIR}"
PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" state_fate/data_utils/download_larry.py \
  --dataset "${DATASET}" \
  --raw-root "${RAW_ROOT}" \
  "$@"

