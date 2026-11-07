#!/bin/bash
# One shared BIT (both directions per step). By default it trains to
# SPECIALIST_STEPS (200K), the point matching each specialist's per-direction
# examples and the pair's combined compute, and evaluates there.
#   CONTINUE=1        train on to 2*SPECIALIST_STEPS (400K); evaluate at 200K and 400K
#   TRAIN_STEPS=N     any other final step (a multiple of CKPT_EVERY, >= SPECIALIST_STEPS)
#   EVAL_CFG_SCALES   guidance scales to evaluate (default "0.0"; "0.0 0.5" = old behavior)
#   EVAL_ONLY=1       skip training; DRYRUN=1 prints all commands; FORCE_EVAL=1 redoes evals
# bash text_to_image/_launch_bit_bidirectional.sh [shared train.py options]
# sbatch text_to_image/_launch_bit_bidirectional.sh
# Resumes runs started by the earlier 400K version of this script: same output
# directory and run name, and --auto-resume picks the latest checkpoint. The LR
# is constant after warmup, so --steps does not change the trajectory, and the
# node count may differ from the original run (global batch stays 512).
# Reruns skip training once step TRAIN_STEPS exists and skip each evaluation
# whose eval/eval_metrics.json is present.
#SBATCH --account=m5319
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=48:00:00
#SBATCH --nodes=4
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
if [[ "${CONTINUE:-0}" == 1 ]]; then
  TRAIN_STEPS=${TRAIN_STEPS:-$((2 * SPECIALIST_STEPS))}
fi
TRAIN_STEPS=${TRAIN_STEPS:-$SPECIALIST_STEPS}
if [[ ! "$TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]] || (( TRAIN_STEPS % CKPT_EVERY != 0 || TRAIN_STEPS < SPECIALIST_STEPS )); then
  echo "TRAIN_STEPS=$TRAIN_STEPS must be a multiple of CKPT_EVERY=$CKPT_EVERY and >= SPECIALIST_STEPS=$SPECIALIST_STEPS" >&2
  exit 2
fi
EVAL_CFG_SCALES=${EVAL_CFG_SCALES:-0.0}
# Placed before "$@", so an explicit --eval-cfg-scales argument still wins.
# shellcheck disable=SC2206  # EVAL_CFG_SCALES is a word list
cfg_args=(--eval-cfg-scales $EVAL_CFG_SCALES)
eval_steps=("$SPECIALIST_STEPS")
(( TRAIN_STEPS != SPECIALIST_STEPS )) && eval_steps+=("$TRAIN_STEPS")

printf -v final_ckpt 'step_%07d.pt' "$TRAIN_STEPS"
if [[ "${EVAL_ONLY:-0}" != 1 ]]; then
  if [[ "${DRYRUN:-0}" != 1 ]] && compgen -G "$root/train/*/checkpoints/$final_ckpt" > /dev/null; then
    echo "Training already reached $TRAIN_STEPS steps under $root/train; skipping training."
  else
    bit_run train bit_bidirectional "$root/train" "$TRAIN_DATA_ROOT" "$TRAIN_STEPS" "" "${cfg_args[@]}" "$@"
  fi
fi
for step in "${eval_steps[@]}"; do
  if [[ "${FORCE_EVAL:-0}" != 1 && "${DRYRUN:-0}" != 1 ]] &&
     compgen -G "$root/eval_${step}/*/eval/eval_metrics.json" > /dev/null; then
    echo "Evaluation at step $step already written under $root/eval_${step}; skipping."
    continue
  fi
  checkpoint=$(checkpoint_at "$root/train" "$step")
  bit_run eval "bit_bidirectional_eval_${step}" "$root/eval_${step}" \
    "$EVAL_DATA_ROOT" "$step" "$checkpoint" "${cfg_args[@]}" "$@"
done
