#!/bin/bash
#
# SLURM entrypoint for the flow wrappers. Same restart safety as
# _multi_node_train.sh: the retry loop resumes from the latest checkpoint on a
# process-level failure, --requeue re-runs the allocation on preemption.
#
#   sbatch --job-name=flow_text_noise --export=ALL,VARIANT=text_noise _sbatch_flow.sh
#   sbatch --nodes=4 --export=ALL,VARIANT=image_noise,MODE=eval,CHECKPOINT=... _sbatch_flow.sh
#
# Flags after the script name are forwarded to train.py.
#
#SBATCH --account=m5319
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --requeue
#SBATCH --output=/pscratch/sd/g/gabeguo/BiB_results/slurm_logs/%x_%j.out

# No `set -u`: dit_env's libblas_mkl_activate.sh hook reads MKL_INTERFACE_LAYER
# while it is unset, so `conda activate` would abort under it.

module load python
module load nccl/2.29.2-cu13
module load conda
conda activate dit_env

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PREFIX_DIR=${PREFIX_DIR:-/pscratch/sd/g/gabeguo} # TODO: change to your own
export HF_HOME=${HF_HOME:-${PREFIX_DIR}/cache_sub}
export NPROC=${NPROC:-4} # Perlmutter nodes have 4 A100s

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29500}

variant=${VARIANT:?Set VARIANT to original, text_noise, or image_noise}
mode=${MODE:-train}
# Perlmutter's layout, not the /data one _flow_common.sh defaults to.
if [[ "$mode" == eval ]]; then
  export DATA_ROOT=${DATA_ROOT:-${PREFIX_DIR}/datasets/text_to_image/gpic_latents_TEST/TEST}
else
  export DATA_ROOT=${DATA_ROOT:-${PREFIX_DIR}/datasets/text_to_image/gpic_latents/train}
fi
# Only the train split has DINO features, and eval needs them too: the flow
# recipe always passes --repa-image, which reads dino_config.json at startup.
export DINO_DIR=${DINO_DIR:-${PREFIX_DIR}/datasets/text_to_image/gpic_latents_dino/train}

# Mirrors _flow_common.sh's default so a chained eval can locate the training
# run. Every process start writes its own timestamp directory, so checkpoints
# are spread across them; rank by step rather than by mtime.
run_root=${OUT_DIR:-${PREFIX_DIR}/BiB_results/flow_comparison/abl_flow_matching_${variant}}
if [[ "$mode" == eval && -z "${CHECKPOINT:-}" ]]; then
  CHECKPOINT=$(ls -1 "$run_root"/*/checkpoints/step_*.pt 2>/dev/null \
    | awk -F'step_' '{print $NF+0, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
  [[ -n "$CHECKPOINT" ]] || { echo "No checkpoint under $run_root to evaluate" >&2; exit 2; }
  export CHECKPOINT
  echo "=== resolved CHECKPOINT=$CHECKPOINT ==="
fi
# Optional guard (set by queue_rebuttal_experiments.sh): refuse to evaluate a
# partial run, e.g. when training hit its wall time before the final step.
if [[ "$mode" == eval && -n "${REQUIRE_STEP:-}" ]]; then
  printf -v required 'step_%07d.pt' "$REQUIRE_STEP"
  [[ "$(basename "$CHECKPOINT")" == "$required" ]] || {
    echo "CHECKPOINT=$CHECKPOINT is not $required; training has not finished" >&2; exit 2; }
fi

cd "$SLURM_SUBMIT_DIR"

script="_${mode}_flow_${variant}.sh"
[[ -f "$script" ]] || { echo "No such wrapper: $script" >&2; exit 2; }

MAX_RETRIES=10
for attempt in $(seq 1 $MAX_RETRIES); do
echo "=== launch attempt $attempt/$MAX_RETRIES ($script) ==="
srun --ntasks-per-node=1 --cpus-per-task=64 bash "$script" "$@" && break
echo "=== attempt $attempt exited non-zero; resuming from latest checkpoint ==="
sleep 10
done
