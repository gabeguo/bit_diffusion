#!/usr/bin/env bash
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --gres=gpu:4
#SBATCH --job-name=larry_benchmark
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=larry_benchmark_%j.out

set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RAW_ROOT="${RAW_ROOT:-${ROOT_DIR}/state_fate/data/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-${ROOT_DIR}/state_fate/data/processed/larry_in_vitro_d2_d6}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/state_fate/runs/larry_in_vitro_benchmark}"
SEED="${SEED:-0}"
DATA_SEED="${DATA_SEED:-0}"
ENV_PREFIX="${ENV_PREFIX:-}"

if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer, got ${SEED}" >&2
  exit 2
fi
if [[ ! "${DATA_SEED}" =~ ^[0-9]+$ ]]; then
  echo "DATA_SEED must be a non-negative integer, got ${DATA_SEED}" >&2
  exit 2
fi

if [[ -n "${ENV_PREFIX}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
  TORCHRUN_BIN="${TORCHRUN_BIN:-${ENV_PREFIX}/bin/torchrun}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
  TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! command -v "${TORCHRUN_BIN}" >/dev/null 2>&1; then
  echo "torchrun executable not found: ${TORCHRUN_BIN}" >&2
  exit 2
fi

SEED_RUN_ROOT="${RUN_ROOT}/seed_${SEED}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SEED_RUN_ROOT}/mpl_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SEED_RUN_ROOT}/xdg_cache}"
mkdir -p "${SEED_RUN_ROOT}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
cd "${ROOT_DIR}"

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install -r state_fate/requirements.txt
fi

if [[ "${FORCE_PREPARE:-0}" == "1" || ! -f "${PROCESSED_DIR}/pairs.npz" ]]; then
  if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
    download_args=()
    if [[ "${INSECURE_SSL:-0}" == "1" ]]; then
      download_args+=(--insecure-ssl)
    fi
    PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" state_fate/data_utils/download_larry.py \
      --dataset in_vitro \
      --raw-root "${RAW_ROOT}" \
      "${download_args[@]}"
  fi
  PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" state_fate/data_utils/prepare_larry.py \
    --dataset in_vitro \
    --raw-root "${RAW_ROOT}" \
    --out-dir "${PROCESSED_DIR}" \
    --early-day "${EARLY_DAY:-2}" \
    --late-day "${LATE_DAY:-6}" \
    --n-hvgs "${N_HVGS:-2000}" \
    --latent-dim "${LATENT_DIM:-64}" \
    --pairs-per-clone "${PAIRS_PER_CLONE:-32}" \
    --seed "${DATA_SEED}"
fi

STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
PRECISION="${PRECISION:-bf16}"
LOG_EVERY="${LOG_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
CKPT_EVERY="${CKPT_EVERY:-5000}"
EVAL_NUM_STEPS="${EVAL_NUM_STEPS:-250}"
MANIFEST="${SEED_RUN_ROOT}/benchmark_manifest.tsv"

if [[ ! -f "${MANIFEST}" ]]; then
  printf "variant\tmethod\tobjective\tsde\tseed\tdata_seed\tcheckpoint_dir\ttest_dir\n" > "${MANIFEST}"
fi

run_method() {
  local variant="$1"
  local method="$2"
  local objective="$3"
  local sde="$4"
  shift 4

  echo "Starting ${method}, seed ${SEED}"
  PYTHONPATH="${ROOT_DIR}" "${TORCHRUN_BIN}" --standalone --nproc-per-node=4 state_fate/train.py \
    --data-root "${PROCESSED_DIR}" \
    --out-dir "${SEED_RUN_ROOT}/${variant}" \
    --arch dit \
    --hidden-dim 768 \
    --num-blocks 12 \
    --dit-num-heads 12 \
    --dit-token-dim 8 \
    --model-sharing shared \
    --objective "${objective}" \
    --sde "${sde}" \
    --precision "${PRECISION}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --test-batch-size "${TEST_BATCH_SIZE}" \
    --lr "${LR:-2e-4}" \
    --grad-clip "${GRAD_CLIP:-1.0}" \
    --log-every "${LOG_EVERY}" \
    --eval-every "${EVAL_EVERY}" \
    --ckpt-every "${CKPT_EVERY}" \
    --eval-num-steps "${EVAL_NUM_STEPS}" \
    --seed "${SEED}" \
    "$@"

  local latest_run
  latest_run="$(find "${SEED_RUN_ROOT}/${variant}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${variant}" "${method}" "${objective}" "${sde}" "${SEED}" "${DATA_SEED}" \
    "${latest_run}/checkpoints" "${latest_run}/test" >> "${MANIFEST}"
}

START_VARIANT="${START_VARIANT:-0}"
END_VARIANT="${END_VARIANT:-5}"

for variant in $(seq "${START_VARIANT}" "${END_VARIANT}"); do
  case "${variant}" in
    0)
      run_method bit_uniform_shared "BIT uniform shared" score uniform --K 0.5
      ;;
    1)
      run_method bit_uniform_separate "BIT uniform separate" score uniform \
        --K 0.5 --model-sharing separate
      ;;
    2)
      run_method bit_cosine_shared "BIT cosine shared" score cosine_decay \
        --K 0.5 --cosine-sde-eps 0.03
      ;;
    3)
      run_method noise_to_data "Noise-to-data diffusion" noise uniform --K 0.5
      ;;
    4)
      run_method rectified_flow "Rectified flow" flow uniform --K 0.5
      ;;
    5)
      run_method endpoint_regression "Endpoint regression" endpoint uniform --K 0.5
      ;;
    *)
      echo "Unknown variant index: ${variant}" >&2
      exit 2
      ;;
  esac
done
