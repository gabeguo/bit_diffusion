#!/bin/bash
# The same entrypoint evaluates original flow, noisy flow, and SDE checkpoints.
# PHASE=generate|score|all (default all) splits the distributed generation step
# from the single-process scoring step. Multi-node runs must split them: the
# shards carry no collectives to synchronize on, so scoring can only start once
# every node's srun task has exited.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
: "${COCO_ROOT:?Set COCO_ROOT}"
: "${OUT_DIR:?Set OUT_DIR to a new output directory}"
task=${1:?Usage: bash _eval_coco.sh t2i|i2t [extra generation flags]}
shift
phase=${PHASE:-all}
export PYTHONPATH=".:..${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$phase" == all && "${SLURM_NNODES:-1}" -gt 1 ]]; then
  echo "Multi-node: run PHASE=generate under srun, then PHASE=score once (see _sbatch_coco_eval.sh)." >&2
  exit 2
fi

run() {
  if [[ "${DRYRUN:-0}" == 1 ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}

if [[ "$phase" != score ]]; then
  : "${CHECKPOINT:?Set CHECKPOINT}"
  : "${DATA_ROOT:?Set DATA_ROOT to training manifests and the token vocabulary table}"
  if [[ "${SLURM_NNODES:-1}" -gt 1 ]]; then
    # SLURM_NODEID exists only inside an srun step; MASTER_ADDR comes from the batch script.
    launch=(torchrun --nnodes="$SLURM_NNODES" --nproc-per-node="${NPROC:-8}"
      --node-rank="${SLURM_NODEID:?Launch multi-node runs under srun}"
      --rdzv-id="$SLURM_JOB_ID" --rdzv-backend=c10d
      --rdzv-endpoint="${MASTER_ADDR:?Set MASTER_ADDR to the first node}:${MASTER_PORT:-29500}")
  else
    launch=(torchrun --standalone --nproc-per-node="${NPROC:-8}")
  fi
  run "${launch[@]}" coco_eval.py generate \
    --checkpoint "$CHECKPOINT" --data-root "$DATA_ROOT" --coco-root "$COCO_ROOT" \
    --task "$task" --output "$OUT_DIR" --steps "${EVAL_STEPS:-500}" \
    --cfg-scale "${CFG_SCALE:-0}" --seed "${SEED:-0}" "$@"
fi

if [[ "$phase" != generate ]]; then
  run python coco_eval.py score --coco-root "$COCO_ROOT" --output "$OUT_DIR" \
    --fid-mode "${FID_MODE:-clean}" ${NO_SPICE:+--no-spice} ${CLIP_SCORE:+--clip-score}
fi
