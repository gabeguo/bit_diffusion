#!/bin/bash
#
#SBATCH --account=m5319
#SBATCH --job-name=encode-gpic
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=8
#SBATCH --gpus-per-node=a100:4
#SBATCH --ntasks-per-node=4          # one task per GPU (no NCCL; embarrassingly parallel)
#SBATCH --cpus-per-task=16           # CPU JPEG decode keeps the VAE fed
# NOTE: all 4 GPUs are visible to every task; each task selects cuda:$SLURM_LOCALID.
# (Do NOT use --gpus-per-task=1, which would expose only cuda:0 per task.)
#SBATCH --licenses=scratch,cfs
#SBATCH --output=/pscratch/sd/g/gabeguo/BiB_results/slurm_logs/%j.out
#SBATCH --requeue
#SBATCH --reservation=encode_data

module load python
module load conda
conda activate dit_env

set -Eeuo pipefail

export PREFIX_DIR=/pscratch/sd/g/gabeguo
export HF_HOME=${PREFIX_DIR}/cache_sub

# Stream source tars from CFS; write memmaps to fast scratch (random-access at
# train time wants NVMe). Copy outputs to CFS for archival once complete.
TARS_DIR=$CFS/m5319/gabeguo/gpic/train
OUTPUT_DIR=${OUTPUT_DIR:-${PREFIX_DIR}/datasets/text_to_image/gpic_latents_flux/train}
SPLITS=${SPLITS:-train}

MAX_RETRIES=10
for attempt in $(seq 1 $MAX_RETRIES); do
echo "=== launch attempt $attempt/$MAX_RETRIES ==="
PYTHONPATH=.:..:../data_utils srun --cpu-bind=cores \
    python ../data_utils/encode_gpic.py \
        --bridge-preset "flux" \
        --tars-dir "$TARS_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --splits $SPLITS \
        --shard-size 262144 \
        --vae-batch 256 \
        --text-batch 256 \
        --log-every 6250 \
        --no-store-logvar \
    && break
echo "=== attempt $attempt exited non-zero; resuming from per-rank progress ==="
sleep 10
done
