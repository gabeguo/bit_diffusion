#!/bin/bash
#
#SBATCH --account=m5319
#SBATCH --job-name=large_scale
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=72:00:00
#SBATCH --nodes=16                # Multi node
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=64       # CPUs for the job
#SBATCH --ntasks-per-node=1            # Number of tasks (one per node)
#SBATCH --output=/pscratch/sd/g/gabeguo/BiB_results/slurm_logs/%j.out
#SBATCH --reservation=large_scale
#SBATCH --requeue

module load python
module load nccl/2.29.2-cu13
module load conda
conda activate dit_env

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# Abort (crash loudly) on a stuck/errored NCCL collective instead of hanging,
# so the short PG timeout + retry loop below can resume from the latest ckpt.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Reduce CUDA caching-allocator fragmentation (notably after eval, which loads
# FID/CLIP nets and decodes many images inside the training process).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


export PREFIX_DIR=/pscratch/sd/g/gabeguo
export HF_HOME=${PREFIX_DIR}/cache_sub

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

MAX_RETRIES=10
for attempt in $(seq 1 $MAX_RETRIES); do
echo "=== launch attempt $attempt/$MAX_RETRIES ==="
PYTHONPATH=.:.. srun --ntasks-per-node=1 --cpus-per-task=64 bash -c '
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=4 \
    --node-rank=$SLURM_NODEID \
    --rdzv-id=$SLURM_JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
    train.py \
        --wandb-name mega_run_multi_ema \
        --min-throughput 0.65 \
        --min-throughput-windows 5 \
        --use-token-text-bridge \
        --token-layout row_major \
        --x0-cond-source x0 \
        --data-root ${PREFIX_DIR}/datasets/text_to_image/gpic_latents/train \
        --out-dir ${PREFIX_DIR}/BiB_results/06_29_26/gpic/mega_run_multi_ema \
        --log-every 100 \
        --ckpt-every 10000 \
        --eval-every 10000 \
        --model DiTXA-XL/2 \
        --global-batch-size 1792 \
        --lr 2.5e-4 \
        --warmup-steps 5000 \
        --ema-decay 0.9995 \
        --alt-ema-decays 0.9997 0.9998 \
        --eps 9.9e-4 \
        --sde periodic \
        --periodic_sde_alpha 0.95 \
        --periodic_sde_k 1.0 \
        --periodic_sde_eps 0.05 \
        --eval-cfg-scales 0.0 0.5 \
        --eval-fid-num-samples 4096 \
        --eval-fid-batch-size 16 \
        --eval-i2t-num-samples 4096 \
        --eval-i2t-batch-size 32 \
        --eval-text-decode-include-padding-in-accuracy \
        --repa-image \
        --repa-image-lambda 0.75 \
        --repa-image-warmup-steps 0 \
        --repa-image-layer 8 \
        --repa-phase equal \
        --dino-dir ${PREFIX_DIR}/datasets/text_to_image/gpic_latents_dino/train \
        --time-sampler logit_normal \
        --time-sampler-logit-normal-mean 0.4 \
        --time-sampler-logit-normal-std 0.7 \
        --auto-resume
' && break
echo "=== attempt $attempt exited non-zero; resuming from latest checkpoint ==="
sleep 10
done
