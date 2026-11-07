#!/bin/bash
# One leg of the small-scale shared-vs-specialist control (DiTXA-B/2).
# Submitted by _launch_bit_small_scale.sh; can also be sbatch'ed directly:
#   sbatch --export=ALL,RUN=shared,LEG=1 _bit_small_scale_leg.sh
# RUN=fwd (T2I specialist) | rev (I2T specialist) | shared (both directions per step)
# LEG=1 trains to LEG_STEPS (100K) and evaluates that checkpoint; LEG=2 resumes
# (model AND data order) to 2*LEG_STEPS (200K) and evaluates that checkpoint.
# There are no training-time evaluations. Evaluation: CFG 0 only, 500 SDE
# steps, SMALL_EVAL_SAMPLES (10K) held-out samples per direction.
# Recipe otherwise identical to the Table 1 controls (_bit_ablation_common.sh):
# batch 512, LR 1.5e-4, 5K warmup, EMA 0.9995, periodic SDE, image REPA 0.5,
# checkpoints every SMALL_CKPT_EVERY (10K: 20 per run, incl. 100K and 200K).
# Training is retried in the allocation up to TRAIN_RETRIES (10) times; a
# watchdog kills an attempt after MIN_THROUGHPUT_WINDOWS (5) consecutive
# 100-step windows below MIN_THROUGHPUT steps/s (shared 1.3, specialists 2.0).
# DRYRUN=1 prints commands.
#SBATCH --account=m5319
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --requeue

set -eo pipefail
cd "${BIT_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}}"
if [[ -d text_to_image ]]; then cd text_to_image; fi
# Set before sourcing: _bit_ablation_common.sh keeps an existing CKPT_EVERY and
# checks that it divides SPECIALIST_STEPS (200K); LEG_STEPS is checked below.
CKPT_EVERY=${SMALL_CKPT_EVERY:-10000}
source ./_bit_ablation_common.sh
RUN=${RUN:?Set RUN to fwd, rev, or shared}
LEG=${LEG:?Set LEG to 1 or 2}
LEG_STEPS=${LEG_STEPS:-100000}
SMALL_MODEL=${SMALL_MODEL:-DiTXA-B/2}
# REPA aligns 1/3 of the way up the trunk in the Table 1 recipe (layer 8 of 24);
# layer 4 keeps that relative depth for the 12-block B/2 trunk.
SMALL_REPA_LAYER=${SMALL_REPA_LAYER:-4}
SMALL_EVAL_SAMPLES=${SMALL_EVAL_SAMPLES:-10000}
# A shared step trains both directions, so it runs slower than a specialist step.
case "$RUN" in
  fwd) direction=(--no-reverse); default_min_throughput=2.0 ;;
  rev) direction=(--no-forward); default_min_throughput=2.0 ;;
  shared) direction=(); default_min_throughput=1.3 ;;
  *) echo "RUN must be fwd, rev, or shared" >&2; exit 2 ;;
esac
[[ "$LEG" == 1 || "$LEG" == 2 ]] || { echo "LEG must be 1 or 2" >&2; exit 2; }
if (( LEG_STEPS % CKPT_EVERY != 0 )); then
  echo "LEG_STEPS=$LEG_STEPS must be a multiple of CKPT_EVERY=$CKPT_EVERY" >&2; exit 2
fi
TRAIN_RETRIES=${TRAIN_RETRIES:-10}
watchdog=(--min-throughput "${MIN_THROUGHPUT:-$default_min_throughput}"
  --min-throughput-windows "${MIN_THROUGHPUT_WINDOWS:-5}")
root=${OUT_DIR:-$prefix/BiB_results/bit_small_scale}/$RUN
target=$((LEG * LEG_STEPS))
common=(--model "$SMALL_MODEL" --repa-image-layer "$SMALL_REPA_LAYER"
  --eval-cfg-scales 0.0 "${direction[@]}")

printf -v ckpt 'step_%07d.pt' "$target"
if [[ "${DRYRUN:-0}" != 1 ]] && compgen -G "$root/train/*/checkpoints/$ckpt" > /dev/null; then
  echo "[$RUN leg $LEG] training already reached $target steps; skipping training."
else
  # --eval-every beyond the run: no training-time evaluations at all.
  bit_train "bit_small_${RUN}" "$root/train" "$TRAIN_DATA_ROOT" "$target" "" \
    "${common[@]}" --eval-every 1000000000 --resume-data-position "${watchdog[@]}" "$@"
fi

if [[ "${FORCE_EVAL:-0}" != 1 && "${DRYRUN:-0}" != 1 ]] &&
   compgen -G "$root/eval_${target}/*/eval/eval_metrics.json" > /dev/null; then
  echo "[$RUN leg $LEG] evaluation at $target already written; skipping."
  exit 0
fi
checkpoint=$(checkpoint_at "$root/train" "$target")
bit_run eval "bit_small_${RUN}_eval_${target}" "$root/eval_${target}" "$EVAL_DATA_ROOT" \
  "$target" "$checkpoint" "${common[@]}" \
  --eval-fid-num-samples "$SMALL_EVAL_SAMPLES" --eval-i2t-num-samples "$SMALL_EVAL_SAMPLES" "$@"
