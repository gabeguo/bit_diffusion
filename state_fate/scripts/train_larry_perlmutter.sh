#!/usr/bin/env bash
#SBATCH --job-name=state_fate_dit
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=a100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=state_fate_perlmutter_%j.out

set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATASET="${DATASET:-in_vitro}"
RAW_ROOT="${RAW_ROOT:-${ROOT_DIR}/state_fate/data/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-${ROOT_DIR}/state_fate/data/processed/larry_in_vitro_d2_d6}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/state_fate/runs/larry_in_vitro_dit}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUN_ROOT}/mpl_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUN_ROOT}/xdg_cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${RUN_ROOT}/slurm_logs"

cd "${ROOT_DIR}"

if [[ "${USE_UV:-1}" == "1" ]]; then
  UV_BIN="${UV_BIN:-uv}"
  UV_VENV="${UV_VENV:-${ROOT_DIR}/state_fate/.venv}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-${RUN_ROOT}/uv_cache}"
  mkdir -p "${UV_CACHE_DIR}"

  if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
    echo "uv not found. Install uv or submit with USE_UV=0." >&2
    exit 1
  fi

  uv_venv_args=()
  if [[ -n "${UV_PYTHON:-}" ]]; then
    uv_venv_args+=(--python "${UV_PYTHON}")
  fi
  uv_venv_args+=("${UV_VENV}")
  "${UV_BIN}" venv "${uv_venv_args[@]}"

  if [[ "${SKIP_UV_SYNC:-0}" != "1" ]]; then
    "${UV_BIN}" pip install \
      --python "${UV_VENV}/bin/python" \
      -r "${ROOT_DIR}/state_fate/requirements.txt"
  fi

  PYTHON_BIN="${UV_VENV}/bin/python"
  TORCHRUN_BIN="${UV_VENV}/bin/torchrun"
else
  PYTHON_BIN="${PYTHON:-python}"
  TORCHRUN_BIN="${TORCHRUN:-torchrun}"
fi

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  download_args=()
  if [[ "${FORCE_DOWNLOAD:-0}" == "1" ]]; then
    download_args+=(--force)
  fi
  PYTHON="${PYTHON_BIN}" DATASET="${DATASET}" RAW_ROOT="${RAW_ROOT}" \
    state_fate/scripts/download_larry.sh "${download_args[@]}"
fi

if [[ "${FORCE_PREPARE:-0}" == "1" || ! -f "${PROCESSED_DIR}/pairs.npz" ]]; then
  PYTHON="${PYTHON_BIN}" \
  DATASET="${DATASET}" \
  RAW_ROOT="${RAW_ROOT}" \
  PROCESSED_DIR="${PROCESSED_DIR}" \
  EARLY_DAY="${EARLY_DAY:-2}" \
  LATE_DAY="${LATE_DAY:-6}" \
  N_HVGS="${N_HVGS:-2000}" \
  LATENT_DIM="${LATENT_DIM:-64}" \
  PAIRS_PER_CLONE="${PAIRS_PER_CLONE:-32}" \
  SEED="${SEED:-0}" \
    state_fate/scripts/prepare_larry_in_vitro.sh
fi

PYTHONPATH="${ROOT_DIR}" "${TORCHRUN_BIN}" --standalone --nproc-per-node=4 state_fate/train.py \
  --data-root "${PROCESSED_DIR}" \
  --out-dir "${RUN_ROOT}" \
  --arch "${ARCH:-dit}" \
  --hidden-dim "${HIDDEN_DIM:-768}" \
  --num-blocks "${NUM_BLOCKS:-12}" \
  --dit-num-heads "${DIT_NUM_HEADS:-12}" \
  --dit-token-dim "${DIT_TOKEN_DIM:-8}" \
  --model-sharing "${MODEL_SHARING:-shared}" \
  --objective "${OBJECTIVE:-score}" \
  --sde "${SDE:-uniform}" \
  --K "${SDE_K:-0.5}" \
  --precision "${PRECISION:-bf16}" \
  --steps "${STEPS:-20000}" \
  --batch-size "${BATCH_SIZE:-512}" \
  --test-batch-size "${TEST_BATCH_SIZE:-512}" \
  --lr "${LR:-2e-4}" \
  --grad-clip "${GRAD_CLIP:-1.0}" \
  --log-every "${LOG_EVERY:-100}" \
  --eval-every "${EVAL_EVERY:-5000}" \
  --ckpt-every "${CKPT_EVERY:-5000}" \
  --eval-num-steps "${EVAL_NUM_STEPS:-250}" \
  --seed "${SEED:-0}" \
  "$@"
