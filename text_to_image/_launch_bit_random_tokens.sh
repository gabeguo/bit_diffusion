#!/bin/bash
# Fixed random token vocabulary; train/eval only the two ablated specialists.
# bash text_to_image/_launch_bit_random_tokens.sh [shared train.py options]
# sbatch text_to_image/_launch_bit_random_tokens.sh
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
cd "${BIT_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}}"
if [[ -d text_to_image ]]; then cd text_to_image; fi
source ./_bit_ablation_common.sh
root=${OUT_DIR:-$prefix/BiB_results/bit_random_tokens_seed${TOKEN_SEED:-0}}
random_data=${RANDOM_DATA_ROOT:-$root/data}
run python data_utils/prepare_random_token_data.py --train-root "$TRAIN_DATA_ROOT" \
  --eval-root "$EVAL_DATA_ROOT" --out-dir "$random_data" --seed "${TOKEN_SEED:-0}"
for direction in fwd rev; do
  flag=--no-reverse
  [[ "$direction" == rev ]] && flag=--no-forward
  name=bit_random_tokens_seed${TOKEN_SEED:-0}_${direction}
  if [[ "${EVAL_ONLY:-0}" != 1 ]]; then
    bit_run train "$name" "$root/$direction/train" "$random_data/train" \
      "$SPECIALIST_STEPS" "" "$@" "$flag"
  fi
  checkpoint=$(checkpoint_at "$root/$direction/train" "$SPECIALIST_STEPS")
  bit_run eval "${name}_eval" "$root/$direction/eval" "$random_data/test" \
    "$SPECIALIST_STEPS" "$checkpoint" "$@" "$flag"
done
