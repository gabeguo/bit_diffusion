#!/bin/bash
#
# SLURM entrypoint for COCO evaluation. Generation is a distributed srun step;
# scoring runs once afterward, because coco_eval.py refuses to score under
# torchrun and requires every shard to be on disk first.
#
#   sbatch --job-name=coco_t2i \
#     --export=ALL,TASK=t2i,CHECKPOINT=/path/step_0200000.pt,DATA_ROOT=/path/manifests,COCO_ROOT=/path/coco,OUT_DIR=/path/eval/model_t2i \
#     _sbatch_coco_eval.sh
#
# Flags after the script name are forwarded to the generation step. There is no
# retry loop and no --requeue: OUT_DIR must be new, and a partially generated
# directory cannot be resumed or reused.
#
#SBATCH --account=m5319
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --output=/pscratch/sd/g/gabeguo/BiB_results/slurm_logs/%x_%j.out

set -eo pipefail
# No `set -u`: dit_env's libblas_mkl_activate.sh hook reads MKL_INTERFACE_LAYER
# while it is unset, so `conda activate` would abort under it.

module load python
module load nccl/2.29.2-cu13
module load conda
module load cudatoolkit/13.0
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
cd "$SLURM_SUBMIT_DIR"

task=${TASK:?Set TASK to t2i or i2t}

export CHECKPOINT=${CHECKPOINT:-"hf://therealgabeguo/BiB_generative/large_scale/06_29_26/gpic_16_nodes_repa_15/step_0136000.pt"}
export DATA_ROOT=${DATA_ROOT:-"/pscratch/sd/g/gabeguo/datasets/text_to_image/gpic_latents/train"}
export COCO_ROOT=${COCO_ROOT:-"/pscratch/sd/g/gabeguo/datasets/coco"}
export OUT_DIR=${OUT_DIR:-"${PREFIX_DIR}/BiB_results/coco_eval/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${task}"}

echo "=== generating ($task, $SLURM_NNODES nodes x $NPROC GPUs) ==="
PHASE=generate srun --ntasks-per-node=1 --cpus-per-task=64 bash _eval_coco.sh "$task" "$@"

echo "=== scoring ==="
PHASE=score srun --nodes=1 --ntasks=1 --cpus-per-task=64 bash _eval_coco.sh "$task"
