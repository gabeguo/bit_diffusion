#!/bin/bash
#
#SBATCH --account=m5319
#SBATCH --job-name=dino-encode
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=30:00:00
#SBATCH --nodes=8                  # 8-node scale_up reservation
#SBATCH --gpus-per-node=a100:4
#SBATCH --ntasks-per-node=4        # one task per GPU (no torchrun; SLURM ranks)
#SBATCH --cpus-per-task=16         # 4 tasks * 16 = 64 cores/node
#SBATCH --output=/pscratch/sd/g/gabeguo/BiB_results/slurm_logs/%j.out
#SBATCH --reservation=encode_data
#SBATCH --requeue

module load python
module load conda
conda activate dit_env

export PREFIX_DIR=/pscratch/sd/g/gabeguo # TODO: change to your own
export HF_HOME=${PREFIX_DIR}/cache_sub
export HF_TOKEN=$(cat "$HF_HOME/token" 2>/dev/null)
export TORCH_HOME=${PREFIX_DIR}/cache_sub/torch # DINOv2 hub weights cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pre-cache the DINOv2 weights once on a login node (compute nodes read $TORCH_HOME):
#   TORCH_HOME=${PREFIX_DIR}/cache_sub/torch python -c \
#     "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitl14')"

SRC=${PREFIX_DIR}/datasets/text_to_image/gpic_latents_flux/train
DINO=${PREFIX_DIR}/datasets/text_to_image/gpic_latents_flux_dino/train

# cd ${PREFIX_DIR}/BiB/text_to_image

# Resume is driven by the per-shard dino_filled bitmaps (a row is skipped once
# its bit is set), so on a requeue/crash it picks up where it left off, and the
# world size may differ from prior attempts. Loop until a clean exit.
MAX_RETRIES=10
for attempt in $(seq 1 $MAX_RETRIES); do
echo "=== encode attempt $attempt/$MAX_RETRIES ==="
# PYTHONPATH=.:.. torchrun --standalone --nproc_per_node=8 \
#     -m data_utils.encode_dino_features \
PYTHONPATH=.:.. srun --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 \
    python -m data_utils.encode_dino_features \
        --source-root ${SRC} \
        --dino-dir ${DINO} \
        --cache-dir ${HF_HOME} \
        --dino-model dinov2_vitb14 \
        --store patch \
        --img-size 224 \
        --batch 128 \
        --percent 10 \
    && break
echo "=== attempt $attempt exited non-zero; resuming ==="
sleep 10
done
