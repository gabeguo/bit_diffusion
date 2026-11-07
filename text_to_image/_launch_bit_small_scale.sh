#!/bin/bash
# Small-scale (DiTXA-B/2) control: T2I specialist, I2T specialist, and one
# shared bidirectional model, same recipe, 200K steps each, in two legs:
#   leg 1: train to 100K, evaluate step 100K
#   leg 2 (afterok leg 1): continue model and data order to 200K, evaluate step 200K
# Run from a login node:
#   bash _launch_bit_small_scale.sh               # submit everything not done yet
#   DRYRUN=1 bash _launch_bit_small_scale.sh      # print sbatch commands only
#   STATUS=1 bash _launch_bit_small_scale.sh      # print per-run status only
# Submitted job ids are printed on stdout (one per line), progress on stderr.
#   RUNS="shared" bash _launch_bit_small_scale.sh # subset of fwd rev shared
# Settings: RESERVATION (default iclr_rebuttal; empty for none), ACCOUNT (m5319),
# QOS (regular), NODES (2), LEG_TIME (24:00:00), PREFIX_DIR, OUT_DIR, LOG_DIR,
# plus the leg settings in _bit_small_scale_leg.sh (LEG_STEPS, SMALL_MODEL, ...).
# Finished legs are skipped; resubmitting after a failure resumes from the last
# checkpoint and data position. Do not resubmit while jobs are still queued.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PREFIX_DIR=${PREFIX_DIR:-/pscratch/sd/g/gabeguo}
export PREFIX_DIR
OUT_DIR=${OUT_DIR:-$PREFIX_DIR/BiB_results/bit_small_scale}
export OUT_DIR
LOG_DIR=${LOG_DIR:-$OUT_DIR/slurm_logs}
RUNS=${RUNS:-"fwd rev shared"}
RESERVATION=${RESERVATION-iclr_rebuttal}
ACCOUNT=${ACCOUNT:-m5319}
QOS=${QOS:-regular}
NODES=${NODES:-2}
LEG_TIME=${LEG_TIME:-24:00:00}
LEG_STEPS=${LEG_STEPS:-100000}
export LEG_STEPS

have() { compgen -G "$1" > /dev/null; }
ckpt_done() { have "$OUT_DIR/$1/train/*/checkpoints/$(printf 'step_%07d.pt' "$2")"; }
eval_done() { have "$OUT_DIR/$1/eval_$2/*/eval/eval_metrics.json"; }

submit() {  # run leg dependency -> job id
  local run=$1 leg=$2 dep=$3
  local args=(--parsable --job-name="small_${run}_leg${leg}" --account="$ACCOUNT" --qos="$QOS"
    --nodes="$NODES" --time="$LEG_TIME" --output="$LOG_DIR/%x_%j.out"
    --export="ALL,RUN=$run,LEG=$leg")
  [[ -n "$RESERVATION" ]] && args+=(--reservation="$RESERVATION")
  # A failed or timed-out leg 1 cancels leg 2 instead of leaving it pending forever.
  [[ -n "$dep" ]] && args+=(--dependency="afterok:$dep" --kill-on-invalid-dep=yes)
  if [[ "${DRYRUN:-0}" == 1 ]]; then
    echo "sbatch ${args[*]} _bit_small_scale_leg.sh" >&2
    echo "<${run}_leg${leg}>"
    return
  fi
  mkdir -p "$LOG_DIR"
  sbatch "${args[@]}" _bit_small_scale_leg.sh
}

for run in $RUNS; do
  case "$run" in fwd|rev|shared) ;; *) echo "Unknown run '$run' (fwd, rev, shared)" >&2; exit 2 ;; esac
  s1=$LEG_STEPS s2=$((2 * LEG_STEPS))
  leg1_done=0; ckpt_done "$run" "$s1" && eval_done "$run" "$s1" && leg1_done=1
  leg2_done=0; ckpt_done "$run" "$s2" && eval_done "$run" "$s2" && leg2_done=1
  if [[ "${STATUS:-0}" == 1 ]]; then
    printf '%-7s leg1(%s)=%s leg2(%s)=%s\n' "$run" "$s1" "$( ((leg1_done)) && echo "done" || echo "todo")" \
      "$s2" "$( ((leg2_done)) && echo "done" || echo "todo")"
    continue
  fi
  jid=""
  if (( ! leg1_done )); then
    jid=$(submit "$run" 1 "")
    echo "small_${run}_leg1: ${jid}" >&2
    echo "$jid"
  fi
  if (( ! leg2_done )); then
    jid2=$(submit "$run" 2 "$jid")
    echo "small_${run}_leg2: ${jid2}${jid:+ (afterok ${jid})}" >&2
    echo "$jid2"
  fi
  if (( leg1_done && leg2_done )); then echo "small_${run}: done (100K and 200K evaluated)" >&2; fi
done
