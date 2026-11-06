#!/bin/bash
# One shared BIT, trained to 400K, then evaluated at 200K and 400K.
# bash text_to_image/_launch_bit_bidirectional.sh [shared train.py options]
# sbatch text_to_image/_launch_bit_bidirectional.sh
# DRYRUN=1 prints all commands; EVAL_ONLY=1 reuses existing checkpoints.
#SBATCH --account=m5319
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --requeue

set -eo pipefail
# sbatch copies this file; submit from the repo root or text_to_image/.
cd "${BIT_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}}"
if [[ -d text_to_image ]]; then cd text_to_image; fi
source ./_bit_ablation_common.sh
root=${OUT_DIR:-$prefix/BiB_results/bit_bidirectional}
final_steps=$((2 * SPECIALIST_STEPS))
if [[ "${EVAL_ONLY:-0}" != 1 ]]; then
  bit_run train bit_bidirectional "$root/train" "$TRAIN_DATA_ROOT" "$final_steps" "" "$@"
fi
for step in "$SPECIALIST_STEPS" "$final_steps"; do
  checkpoint=$(checkpoint_at "$root/train" "$step")
  bit_run eval "bit_bidirectional_eval_${step}" "$root/eval_${step}" \
    "$EVAL_DATA_ROOT" "$final_steps" "$checkpoint" "$@"
done
