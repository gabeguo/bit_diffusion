#!/bin/bash
# Shared train/eval recipe for the two BIT follow-up experiments. Source only.
set -eo pipefail
if [[ -n "${SLURM_JOB_ID:-}" && "${DRYRUN:-0}" != 1 ]]; then
  module load python
  module load nccl/2.29.2-cu13
  module load conda
  conda activate "${CONDA_ENV:-dit_env}"
fi
# Conda's activation hooks read unset variables, so enable nounset afterwards.
set -u
for option in "$@"; do
  case "$option" in
    --data-root|--data-root=*|--out-dir|--out-dir=*|--steps|--steps=*|--ckpt-every|--ckpt-every=*|\
    --resume|--resume=*|--auto-resume|--eval-only|--no-forward|--no-reverse|\
    --text-as-noise|--image-as-noise|--sde|--sde=*|--wandb-name|--wandb-name=*)
      echo "$option is managed by this experiment; use its environment settings instead" >&2
      exit 2 ;;
  esac
done

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  prefix=${PREFIX_DIR:-/pscratch/sd/g/gabeguo}
  TRAIN_DATA_ROOT=${TRAIN_DATA_ROOT:-$prefix/datasets/text_to_image/gpic_latents/train}
  EVAL_DATA_ROOT=${EVAL_DATA_ROOT:-$prefix/datasets/text_to_image/gpic_latents_TEST/TEST}
  DINO_DIR=${DINO_DIR:-$prefix/datasets/text_to_image/gpic_latents_dino/train}
  export NPROC=${NPROC:-4}
  if [[ "${DRYRUN:-0}" != 1 ]]; then
    if [[ -z "${MASTER_ADDR:-}" ]]; then
      MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:?Missing Slurm node list}" | head -n1)
    fi
    export MASTER_ADDR
  fi
  export MASTER_PORT=${MASTER_PORT:-29500}
else
  prefix=${PREFIX_DIR:-/data}
  TRAIN_DATA_ROOT=${TRAIN_DATA_ROOT:-$prefix/gpic_text_to_image_dataset/gpic_latents/gpic_latents/train}
  EVAL_DATA_ROOT=${EVAL_DATA_ROOT:-$prefix/gpic_text_to_image_dataset/gpic_latents_TEST/TEST}
  DINO_DIR=${DINO_DIR:-$prefix/gpic_text_to_image_dataset/gpic_latents/gpic_dino/train}
  export NPROC=${NPROC:-8}
fi
export HF_HOME=${HF_HOME:-$prefix/cache_sub}
export PYTHONPATH=".:..${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
SPECIALIST_STEPS=${SPECIALIST_STEPS:-200000}
CKPT_EVERY=${CKPT_EVERY:-10000}
if [[ ! "$SPECIALIST_STEPS" =~ ^[1-9][0-9]*$ || ! "$CKPT_EVERY" =~ ^[1-9][0-9]*$ ]] ||
   (( SPECIALIST_STEPS % CKPT_EVERY != 0 )); then
  echo "SPECIALIST_STEPS must be a positive multiple of CKPT_EVERY" >&2
  exit 2
fi

run() {
  if [[ "${DRYRUN:-0}" == 1 ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

# Exact steps only: never silently substitute a partial run's latest checkpoint.
checkpoint_at() {
  local root=$1 step=$2 filename
  printf -v filename 'step_%07d.pt' "$step"
  if [[ "${DRYRUN:-0}" == 1 ]]; then
    printf '%s/<timestamp>/checkpoints/%s\n' "$root" "$filename"
    return
  fi
  local matches=()
  shopt -s nullglob
  matches=("$root"/*/checkpoints/"$filename")
  shopt -u nullglob
  if (( ${#matches[@]} != 1 )); then
    echo "Expected exactly one $filename under $root; found ${#matches[@]}" >&2
    return 1
  fi
  printf '%s\n' "${matches[0]}"
}

bit_run() {
  local mode=$1 name=$2 root=$3 data=$4 steps=$5 checkpoint=$6
  shift 6
  # Matches _launch_ablation_sweep.sh and _launch_ablation_eval.sh, including
  # image REPA in BOTH directions and the original SDE-only 500-step eval.
  local args=(--wandb-name "$name" --use-token-text-bridge --token-layout row_major
    --x0-cond-source x0 --data-root "$data" --out-dir "$root"
    --log-every 100 --ckpt-every "$CKPT_EVERY" --eval-every 10000 --steps "$steps"
    --unconditional-percent 0.3 --model DiTXA-L/2 --global-batch-size 512
    --lr 1.5e-4 --warmup-steps 5000 --ema-decay 0.9995 --eps 9.9e-4
    --sde periodic --periodic_sde_alpha 0.95 --periodic_sde_k 1.0 --periodic_sde_eps 0.05
    --repa-image --repa-image-lambda 0.5 --repa-image-warmup-steps 0
    --repa-image-layer 8 --repa-phase equal --dino-dir "$DINO_DIR"
    --eval-cfg-scales 0.0 0.5 --eval-num-sde-steps 500
    --eval-fid-batch-size 16 --eval-i2t-batch-size 32
    --eval-text-decode-include-padding-in-accuracy)
  if [[ "$mode" == train ]]; then
    args+=(--auto-resume --eval-fid-num-samples 1024 --eval-i2t-num-samples 1024)
  else
    args+=(--eval-only --resume "$checkpoint" --val-fraction 1.0
      --eval-fid-num-samples 50000 --eval-i2t-num-samples 10000)
  fi
  args+=("$@")
  local launch=(torchrun --standalone --nproc-per-node="$NPROC" train.py)
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    launch=(srun --ntasks="${SLURM_NNODES:-1}" --ntasks-per-node=1
      --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" bash -c '
        phase=$1; shift
        job_id=$SLURM_JOB_ID
        # train.py uses SLURM_JOB_ID as its W&B id. Scope this override to the
        # trainer so train and separate checkpoint evals cannot share a run.
        export SLURM_JOB_ID="${job_id}-${phase}"
        exec torchrun --nnodes="${SLURM_NNODES:-1}" --nproc-per-node="$NPROC" \
          --node-rank="$SLURM_NODEID" --rdzv-id="${job_id}-${phase}" \
          --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
          train.py "$@"
      ' _ "$name")
  fi
  run "${launch[@]}" "${args[@]}"
}
