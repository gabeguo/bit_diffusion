#!/bin/bash
#
# Submits training plus a dependent held-out evaluation for each noisy-endpoint
# flow variant, through _sbatch_flow.sh.
#
#   ./_launch_flow_pair.sh                    # text_noise + image_noise
#   DRYRUN=1 ./_launch_flow_pair.sh           # print the sbatch lines, submit nothing
#   VARIANTS=text_noise ./_launch_flow_pair.sh
#   RESERVATION= ACCOUNT=m5319 ./_launch_flow_pair.sh   # regular queue instead
#
# Defaults target the `bit` reservation, which admits only account m1266_g and
# holds 4 nodes; two variants at 2 nodes each fill it exactly. TRAIN_TIME plus
# EVAL_TIME must fit inside the reservation window or Slurm refuses the job.
#
# The eval jobs pass no CHECKPOINT: _sbatch_flow.sh resolves the highest-step
# checkpoint under the training run's output directory when the job starts.
# They depend on `afterany`, not `afterok`, because _sbatch_flow.sh's retry
# loop exits 0 even when every attempt failed. Check the resolved path that
# each eval job logs before trusting its numbers.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VARIANTS=${VARIANTS:-"text_noise image_noise"}
ACCOUNT=${ACCOUNT:-m1266}
RESERVATION=${RESERVATION-bit}
NODES=${NODES:-2}
TRAIN_TIME=${TRAIN_TIME:-60:00:00}
EVAL_TIME=${EVAL_TIME:-08:00:00}

common=(--account="$ACCOUNT" --nodes="$NODES")
[[ -n "$RESERVATION" ]] && common+=(--reservation="$RESERVATION")

submit() {
  if [[ "${DRYRUN:-0}" == 1 ]]; then
    { printf 'sbatch --parsable '; printf '%q ' "$@"; printf '\n'; } >&2
    echo "<jobid>"
  else
    sbatch --parsable "$@"
  fi
}

for variant in $VARIANTS; do
  train=$(submit "${common[@]}" --time="$TRAIN_TIME" \
    --job-name="flow_${variant}" \
    --export="ALL,VARIANT=${variant}" _sbatch_flow.sh)
  eval_job=$(submit "${common[@]}" --time="$EVAL_TIME" \
    --dependency="afterany:${train}" \
    --job-name="flow_${variant}_eval" \
    --export="ALL,VARIANT=${variant},MODE=eval" _sbatch_flow.sh)
  echo "${variant}: train=${train} eval=${eval_job}"
done
