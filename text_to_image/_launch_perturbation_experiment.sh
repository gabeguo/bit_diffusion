#!/bin/bash
#SBATCH --account=m5319
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --reservation=run_evals
#SBATCH --requeue

module load python
module load nccl/2.29.2-cu13
module load conda
conda activate dit_env

PREFIX_DIR=$SCRATCH # TODO: change to your own
export HF_HOME=$PREFIX_DIR/cache_sub
TODAY=$(date +%m_%d_%y)

PYTHONPATH=.:.. python perturbation_slerp_experiment.py \
    --forward-ckpt hf://therealgabeguo/BiB_generative/large_scale/06_29_26/gpic_16_nodes_repa_15/step_0100000.pt \
    --num-perturb-tokens 16 \
    --max-token-length 64 \
    --perturb-mode llm \
    --llm-max-tries 5 \
    --nfe 400 \
    --infer-nfe 400 \
    --infer-noise \
    --batch-size 4 \
    --num-slerp 5 \
    --cfg-scale 0 \
    --num-images 64 \
    --text-source image \
    --out $PREFIX_DIR/BiB_results/perturbation_slerp_results/${TODAY}/llm_edit_infer_noise_new_prompt
