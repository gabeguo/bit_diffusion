#!/bin/bash
# Shared recipe copied from _launch_ablation_sweep.sh's flow configuration.
# Wrappers change only the endpoint noise, conditioning, and trained direction.
set -euo pipefail
variant=$1
mode=$2
shift 2
cd "$(dirname "${BASH_SOURCE[0]}")"
prefix=${PREFIX_DIR:-/data}
split=train
[[ "$mode" == eval ]] && split=test
data_root=${DATA_ROOT:-${prefix}/gpic_text_to_image_dataset/gpic_latents/gpic_latents/${split}}
dino_root=${DINO_DIR:-${prefix}/gpic_text_to_image_dataset/gpic_latents/gpic_dino/${split}}
name=abl_flow_matching_${variant}
args=(--sde flow_matching)
case "$variant" in
  original) args+=(--force-unconditional --no-reverse) ;;
  text_noise) args+=(--flow-text-sigma 0.01 --no-reverse) ;;
  image_noise) args+=(--flow-image-sigma 0.01 --no-forward) ;;
  *) echo "Unknown flow variant: $variant" >&2; exit 2 ;;
esac
args+=(--wandb-name "$name" --use-token-text-bridge --token-layout row_major
  --x0-cond-source x0 --data-root "$data_root"
  --out-dir "${OUT_DIR:-${prefix}/BiB_results/flow_comparison/${name}}"
  --min-throughput 1.1 --min-throughput-windows 3
  --log-every 100 --ckpt-every 5000 --eval-every 20000 --steps 200000
  --unconditional-percent 0.3 --model DiTXA-L/2 --global-batch-size 512
  --lr 1.5e-4 --warmup-steps 5000 --ema-decay 0.9995 --eps 9.9e-4
  --repa-image --repa-image-lambda 0.5 --repa-image-warmup-steps 0
  --repa-image-layer 8 --repa-phase equal --dino-dir "$dino_root"
  --eval-cfg-scales 0.0 0.5 --eval-fid-batch-size 16 --eval-i2t-batch-size 32
  --eval-text-decode-include-padding-in-accuracy)
if [[ "$mode" == eval ]]; then
  : "${CHECKPOINT:?Set CHECKPOINT to the matching training checkpoint}"
  args+=(--eval-only --resume "$CHECKPOINT" --val-fraction 1.0
    --eval-fid-num-samples 50000 --eval-i2t-num-samples 10000
    --eval-num-ode-steps "${EVAL_STEPS:-500}" --eval-num-sde-steps "${EVAL_STEPS:-500}")
elif [[ "$mode" == train ]]; then
  args+=(--auto-resume --eval-fid-num-samples 1024 --eval-i2t-num-samples 1024)
else
  echo "Mode must be train or eval" >&2; exit 2
fi
export HF_HOME=${HF_HOME:-${prefix}/cache_sub}
export PYTHONPATH=".:..${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${SLURM_NNODES:-1}" -gt 1 ]]; then
  # SLURM_NODEID exists only inside an srun step, so multi-node runs must go
  # through `srun --ntasks-per-node=1`; MASTER_ADDR comes from the batch script.
  launch=(torchrun --nnodes="$SLURM_NNODES" --nproc-per-node="${NPROC:-4}"
    --node-rank="${SLURM_NODEID:?Launch multi-node runs under srun}"
    --rdzv-id="$SLURM_JOB_ID" --rdzv-backend=c10d
    --rdzv-endpoint="${MASTER_ADDR:?Set MASTER_ADDR to the first node}:${MASTER_PORT:-29500}")
else
  launch=(torchrun --standalone --nproc-per-node="${NPROC:-8}")
fi
cmd=("${launch[@]}" train.py "${args[@]}" "$@")
if [[ "${DRYRUN:-0}" == 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  exec "${cmd[@]}"
fi
