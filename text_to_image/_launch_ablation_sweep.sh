#!/bin/bash
#
# Launches the full ablation sweep as one SLURM job per configuration.
#
#   sweep = {direction: 5} x {sde: 3} = 9 jobs (after the skip guards below)
#
# Each job runs on 2 nodes (8 GPUs) with the same restart/requeue safety as
# _multi_node_train.sh: an in-script retry loop resumes from the latest
# checkpoint on a process-level failure, and #SBATCH --requeue re-runs the whole
# allocation on preemption / node failure.
#
# Each job gets:
#   * its own --out-dir          (so --auto-resume never crosses configs)
#   * a unique name              (SBATCH --job-name, --wandb-name, log file)
#
# Usage:
#   ./_launch_ablation_sweep.sh           # submit all jobs
#   DRYRUN=1 ./_launch_ablation_sweep.sh  # print what would be submitted, submit nothing
#
# Run this from the text_to_image/ directory (same place you'd sbatch
# _train_ablation.sh from).

set -euo pipefail

PREFIX_DIR=/data # TODO: change to your own
TODAY=$(date +%m_%d_%y)
SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_ROOT=${PREFIX_DIR}/BiB_results/${TODAY}/ablation_sweep
LOG_DIR=${PREFIX_DIR}/BiB_results/slurm_logs
data_root=${DATA_ROOT:-${PREFIX_DIR}/gpic_text_to_image_dataset/gpic_latents/gpic_latents/train} # TODO: change to your own

export HF_HOME=${PREFIX_DIR}/cache_sub

# ---------------------------------------------------------------------------
# Sweep axes. Tags feed the unique name; flags are passed verbatim to train.py.
# ---------------------------------------------------------------------------

# This for original sweep:
sde_tags=(periodic)
sde_flags=(
  "--sde periodic --periodic_sde_alpha 0.95 --periodic_sde_k 1.0 --periodic_sde_eps 0.05"
)
# # This for flow matching baseline:
# sde_tags=(flow_matching)
# sde_flags=(
#   "--sde flow_matching --force-unconditional"
# )

# Original sweep:
dir_tags=(fwd rev fwdnoise revnoise)
dir_flags=(
  "--no-reverse"
  "--no-forward"
  "--no-reverse --text-as-noise"
  "--no-forward --image-as-noise"
)
# # This for flow matching baseline:
# dir_tags=(bidir_fm)
# dir_flags=(
#   "--no-reverse"
# )

eps=9.9e-4
# eps=0

repa_img_on_args="--repa-image \
  --repa-image-lambda 0.5 \
  --repa-image-warmup-steps 0 \
  --repa-image-layer 8 \
  --repa-phase equal \
  --dino-dir /data/gpic_text_to_image_dataset/gpic_latents/gpic_dino/train"

# ---------------------------------------------------------------------------

submit_one() {
  local name="$1" out_dir="$2" cmd="$3"
  mkdir -p "${LOG_DIR}"
  if [[ "${DRYRUN:-0}" == "1" ]]; then
    echo "==== ${name} ===="
    echo "  out-dir: ${out_dir}"
    echo "  ${cmd}"
    echo
    return
  fi
  sbatch --job-name="${name}" --output="${LOG_DIR}/${name}_%j.out" <<EOF
#!/bin/bash
#SBATCH --account=m5319
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --reservation=run_evals
#SBATCH --requeue

module load python
module load nccl/2.29.2-cu13
module load conda
conda activate dit_env

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME=${PREFIX_DIR}/cache_sub
cd ${SUBMIT_DIR}

export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

MAX_RETRIES=10
for attempt in \$(seq 1 \$MAX_RETRIES); do
echo "=== launch attempt \$attempt/\$MAX_RETRIES ==="
PYTHONPATH=.:.. srun --ntasks-per-node=1 --cpus-per-task=64 bash -c '
torchrun \\
    --nnodes=\$SLURM_NNODES \\
    --nproc-per-node=4 \\
    --node-rank=\$SLURM_NODEID \\
    --rdzv-id=\$SLURM_JOB_ID \\
    --rdzv-backend=c10d \\
    --rdzv-endpoint='"\$MASTER_ADDR:\$MASTER_PORT"' \\
    train.py ${cmd}
' && break
echo "=== attempt \$attempt exited non-zero; resuming from latest checkpoint ==="
sleep 10
done
EOF
}

export HF_HOME=${PREFIX_DIR}/cache_sub

n=0
for si in "${!sde_tags[@]}"; do
    for di in "${!dir_tags[@]}"; do
      sde_tag="${sde_tags[$si]}"
      dir_tag="${dir_tags[$di]}"

      name="abl_${sde_tag}_${dir_tag}"
      out_dir="${SWEEP_ROOT}/${name}"

      sde_flag="${sde_flags[$si]}"
      dir_flag="${dir_flags[$di]}"

      # if [[ "${dir_flag}" == *"--no-forward"* ]]; then
      #   curr_repa_args=""
      # else
      #   curr_repa_args="${repa_img_on_args}"
      #   # only makes sense to run REPA when generating images: makes no sense for text gen
      # fi
      curr_repa_args="${repa_img_on_args}"

      # The trailing backslashes are consumed by THIS shell, so cmd ends up as
      # a single line of train.py flags (the launcher lives in submit_one);
      # empty dir flags just collapse to extra spaces.
      cmd="--wandb-name ${name} \
        --use-token-text-bridge \
        --token-layout row_major \
        --x0-cond-source x0 \
        --data-root ${data_root} \
        --out-dir ${out_dir} \
        --log-every 100 \
        --ckpt-every 10000 \
        --eval-every 10000 \
        --steps 200_000 \
        --unconditional-percent 0.3 \
        --model DiTXA-L/2 \
        --global-batch-size 512 \
        --lr 1.5e-4 \
        --warmup-steps 5000 \
        --ema-decay 0.9995 \
        --eps ${eps} \
        ${sde_flag} \
        ${curr_repa_args} \
        --eval-cfg-scales 0.0 0.5 \
        --eval-fid-num-samples 1024 \
        --eval-fid-batch-size 16 \
        --eval-i2t-num-samples 1024 \
        --eval-i2t-batch-size 32 \
        --eval-text-decode-include-padding-in-accuracy \
        ${dir_flag} \
        --auto-resume"
      
      PYTHONPATH=.:.. torchrun --nproc-per-node=8 train.py ${cmd}

      # submit_one "${name}" "${out_dir}" "${cmd}"
      n=$((n + 1))
    done
done

echo "Submitted ${n} jobs (DRYRUN=${DRYRUN:-0})."
