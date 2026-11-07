#!/bin/bash
#
# Queue the ICLR 2027 rebuttal experiments in Slurm with one command.
# Every experiment is checked for existing results first; finished work is
# skipped, partial work is resumed. See REBUTTAL_EXPERIMENTS.md.
#
#   bash queue_rebuttal_experiments.sh                       # everything not done yet
#   bash queue_rebuttal_experiments.sh --list                # status only
#   bash queue_rebuttal_experiments.sh --dry-run             # print jobs, submit nothing
#   bash queue_rebuttal_experiments.sh --exclude bidir_400k,noisy_flow
#   bash queue_rebuttal_experiments.sh --only flowtok,crossflow
#   bash queue_rebuttal_experiments.sh --include roundtrip_n2d_cond
#   RESERVATION= bash queue_rebuttal_experiments.sh          # no reservation
#
# Run from anywhere; jobs run in text_to_image/. Settings are environment
# variables (see the CONFIG block); the defaults follow the existing launchers.

set -euo pipefail
shopt -s inherit_errexit  # a failed sbatch inside $(...) must abort the queue
SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SUBMIT_DIR"

DEFAULT_EXPERIMENTS=(flowtok crossflow roundtrip_cfg editing_cfg bidir_400k noisy_flow)
OPTIONAL_EXPERIMENTS=(roundtrip_n2d_cond)
declare -A DESCRIPTION=(
  [flowtok]="FlowTok-XL T2I row for Table 2 (GPIC protocol, 20 steps) [6E8U]"
  [crossflow]="CrossFlow T2I row for Table 2 (GPIC protocol, 50 steps) [6E8U]"
  [roundtrip_cfg]="Fig. 3 round-trip variation with BIT at CFG ROUNDTRIP_CFG_SCALES [4qvB Q2]"
  [editing_cfg]="Fig. 5 image editing at CFG EDIT_CFG_SCALES + 4-row figure [4qvB Q2]"
  [bidir_400k]="Shared bidirectional BIT to BIDIR_TRAIN_STEPS (200K; 400K with BIDIR_CONTINUE=1) [xYk4, 6E8U]"
  [noisy_flow]="Flow ODE with stochastic source-endpoint noise, T2I + I2T [6E8U]"
  [roundtrip_n2d_cond]="(opt-in) round trip for noise-to-data restoring WITH inferred side information [4qvB W1]"
)

usage() {
  sed -n '3,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo "Experiments (default): ${DEFAULT_EXPERIMENTS[*]}"
  echo "Experiments (opt-in):  ${OPTIONAL_EXPERIMENTS[*]}"
  echo "Options: --only LIST | --exclude LIST | --include LIST | --list | --dry-run | --force"
  echo "  --force reruns evaluations whose results exist; it never deletes checkpoints or retrains."
}

only="" exclude="" include="" LIST=0 DRY=${DRYRUN:-0} FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) only=$2; shift 2 ;;
    --only=*) only=${1#*=}; shift ;;
    --exclude) exclude=$2; shift 2 ;;
    --exclude=*) exclude=${1#*=}; shift ;;
    --include) include=$2; shift 2 ;;
    --include=*) include=${1#*=}; shift ;;
    --list|--status) LIST=1; shift ;;
    --dry-run|-n) DRY=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ALL_EXPERIMENTS=("${DEFAULT_EXPERIMENTS[@]}" "${OPTIONAL_EXPERIMENTS[@]}")
for key in ${only//,/ } ${exclude//,/ } ${include//,/ }; do
  [[ -n "${DESCRIPTION[$key]+x}" ]] || { echo "Unknown experiment '$key' (known: ${ALL_EXPERIMENTS[*]})" >&2; exit 2; }
done
read -r -a only_keys <<< "${only//,/ }"
read -r -a exclude_keys <<< "${exclude//,/ }"
read -r -a include_keys <<< "${include//,/ }"
if (( ${#only_keys[@]} )); then
  candidates=("${only_keys[@]}")
else
  candidates=("${DEFAULT_EXPERIMENTS[@]}" "${include_keys[@]}")
fi
SELECTED=()
for key in "${candidates[@]}"; do
  [[ " ${exclude_keys[*]} " == *" $key "* || " ${SELECTED[*]} " == *" $key "* ]] || SELECTED+=("$key")
done
selected() { [[ " ${SELECTED[*]} " == *" $1 "* ]]; }

# ---------------------------------------------------------------- CONFIG ---
PREFIX_DIR=${PREFIX_DIR:-/pscratch/sd/g/gabeguo}             # TODO: change to your own
RESULTS_ROOT=${RESULTS_ROOT:-$PREFIX_DIR/BiB_results/iclr_rebuttal}
LOG_DIR=${LOG_DIR:-$RESULTS_ROOT/slurm_logs}
SCRIPT_DIR=${SCRIPT_DIR:-$RESULTS_ROOT/slurm_scripts}
RESERVATION=${RESERVATION-iclr_rebuttal}   # RESERVATION= (empty) submits without one
ACCOUNT=${ACCOUNT:-m5319}
QOS=${QOS:-regular}
CONSTRAINT=${CONSTRAINT:-gpu&hbm80g}
GPU_SPEC=${GPU_SPEC:-a100:4}               # --gpus-per-node
NPROC=${NPROC:-4}                          # GPUs (torchrun workers) per node
CPUS_PER_TASK=${CPUS_PER_TASK:-64}
CONDA_ENV=${CONDA_ENV:-dit_env}
MODULES=${MODULES:-"python nccl/2.29.2-cu13 conda"}
EXTRA_SBATCH=${EXTRA_SBATCH:-}             # e.g. "--mail-type=END --mail-user=me@x"
HF_HOME=${HF_HOME:-$PREFIX_DIR/cache_sub}
WANDB_MODE=${WANDB_MODE:-online}
# Data: Table 1/2 evaluations use EVAL_DATA_ROOT; Fig. 3/5 launchers use EDIT_DATA_ROOT.
EVAL_DATA_ROOT=${EVAL_DATA_ROOT:-$PREFIX_DIR/datasets/text_to_image/gpic_latents_TEST/TEST}
EDIT_DATA_ROOT=${EDIT_DATA_ROOT:-$PREFIX_DIR/datasets/text_to_image/gpic_latents_sd_test/TEST}
# External baselines (Table 2).
EXT_ROOT=${EXT_ROOT:-$PREFIX_DIR/external_baselines}
EXT_NUM_SAMPLES=${EXT_NUM_SAMPLES:-50000}
EXT_BATCH=${EXT_BATCH:-32}
EXT_SPLIT_SEED=${EXT_SPLIT_SEED:-0}        # train.py --seed of the BIT evaluation
EXT_TIME=${EXT_TIME:-08:00:00}
FLOWTOK_STEPS=${FLOWTOK_STEPS:-20};   FLOWTOK_CFG=${FLOWTOK_CFG:-2.0}
CROSSFLOW_STEPS=${CROSSFLOW_STEPS:-50}; CROSSFLOW_CFG=${CROSSFLOW_CFG:-7.0}
# Round-trip variation (Fig. 3; matches _launch_editing_experiment.sh).
ROUNDTRIP_CFG_SCALES=${ROUNDTRIP_CFG_SCALES:-0.5}
RT_FWD=${RT_FWD:-hf://therealgabeguo/BiB_generative/ablations/07_06_26/data_to_data/fwd/step_0200000.pt}
RT_REV=${RT_REV:-hf://therealgabeguo/BiB_generative/ablations/07_06_26/data_to_data/rev/step_0200000.pt}
RT_N2D_FWD=${RT_N2D_FWD:-hf://therealgabeguo/BiB_generative/ablations/07_06_26/noise_to_data/fwd/step_0200000.pt}
RT_N2D_REV=${RT_N2D_REV:-hf://therealgabeguo/BiB_generative/ablations/07_06_26/noise_to_data/rev/step_0200000.pt}
RT_NUM_IMAGES=${RT_NUM_IMAGES:-500}
RT_NUM_VARIATIONS=${RT_NUM_VARIATIONS:-8}
RT_TIME_FRACTIONS=${RT_TIME_FRACTIONS:-"0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"}
RT_NFE=${RT_NFE:-250}
RT_TIME=${RT_TIME:-06:00:00}
ROUNDTRIP_BASELINE_IMAGE_JSONS=${ROUNDTRIP_BASELINE_IMAGE_JSONS:-}   # existing Fig. 3 runs to overlay
ROUNDTRIP_BASELINE_TEXT_JSONS=${ROUNDTRIP_BASELINE_TEXT_JSONS:-}
# Image editing (Fig. 5; matches _launch_perturbation_experiment.sh).
EDIT_CFG_SCALES=${EDIT_CFG_SCALES:-0.5}
EDIT_CKPT=${EDIT_CKPT:-hf://therealgabeguo/BiB_generative/large_scale/06_29_26/gpic_16_nodes_repa_15/step_0100000.pt}
EDIT_NUM_IMAGES=${EDIT_NUM_IMAGES:-64}
EDIT_TIME=${EDIT_TIME:-24:00:00}
FIG5_IMAGES=${FIG5_IMAGES:-}               # dataset indices of the 4 rows; default first 4
# Shared bidirectional model (existing _launch_bit_bidirectional.sh).
BIDIR_OUT=${BIDIR_OUT:-$PREFIX_DIR/BiB_results/bit_bidirectional}
BIDIR_NODES=${BIDIR_NODES:-4}
BIDIR_TIME=${BIDIR_TIME:-48:00:00}
BIDIR_CHAIN=${BIDIR_CHAIN:-3}              # chained resubmissions (each resumes)
BIDIR_ARGS=${BIDIR_ARGS:-}                 # extra train.py flags, e.g. "--wandb-project bit-controls"
SPECIALIST_STEPS=${SPECIALIST_STEPS:-200000}
BIDIR_CONTINUE=${BIDIR_CONTINUE:-0}        # 1: train on to 2*SPECIALIST_STEPS (400K)
if [[ "$BIDIR_CONTINUE" == 1 ]]; then BIDIR_TRAIN_STEPS=${BIDIR_TRAIN_STEPS:-$((2 * SPECIALIST_STEPS))}; fi
BIDIR_TRAIN_STEPS=${BIDIR_TRAIN_STEPS:-$SPECIALIST_STEPS}
BIDIR_EVAL_CFG_SCALES=${BIDIR_EVAL_CFG_SCALES:-0.0}
# Stochastic-endpoint flow (existing _sbatch_flow.sh / _flow_common.sh).
FLOW_VARIANTS=${FLOW_VARIANTS:-"text_noise image_noise"}
FLOW_ROOT_BASE=${FLOW_ROOT_BASE:-$PREFIX_DIR/BiB_results/flow_comparison}
FLOW_NODES=${FLOW_NODES:-2}
FLOW_TRAIN_TIME=${FLOW_TRAIN_TIME:-48:00:00}
FLOW_EVAL_TIME=${FLOW_EVAL_TIME:-08:00:00}
FLOW_CHAIN=${FLOW_CHAIN:-2}
FLOW_STEPS=200000                          # fixed by _flow_common.sh
# ---------------------------------------------------------------------------

have() { compgen -G "$1" > /dev/null; }
q() { printf '%q ' "$@"; }   # shell-quoted words, each followed by a space
q1() { printf '%q' "$1"; }    # one quoted word, no trailing space

SUBMITTED=()
# submit NAME DEPENDENCY NODES TIME BODY [requeue] -> prints job id on stdout
submit() {
  local name=$1 dep=$2 nodes=$3 time=$4 body=$5 requeue=${6:-}
  local args=(--parsable --job-name="rb_${name}" --account="$ACCOUNT" --qos="$QOS"
    --constraint="$CONSTRAINT" --nodes="$nodes" --gpus-per-node="$GPU_SPEC"
    --ntasks-per-node=1 --cpus-per-task="$CPUS_PER_TASK" --time="$time"
    --output="$LOG_DIR/%x_%j.out")
  [[ -n "$RESERVATION" ]] && args+=(--reservation="$RESERVATION")
  [[ -n "$dep" ]] && args+=(--dependency="$dep")
  [[ -n "$requeue" ]] && args+=(--requeue)
  # shellcheck disable=SC2206
  args+=($EXTRA_SBATCH)
  local module_lines="" m
  for m in $MODULES; do module_lines+="module load $m"$'\n'; done
  local script="$SCRIPT_DIR/${name}.sbatch"
  local content
  content=$(cat <<EOF
#!/bin/bash
# Generated by queue_rebuttal_experiments.sh on $(date -Is)
set -eo pipefail  # no -u: conda activation hooks read unset variables
${module_lines}conda activate $(q "$CONDA_ENV")
export PREFIX_DIR=$(q "$PREFIX_DIR") HF_HOME=$(q "$HF_HOME") TORCH_HOME=$(q "$HF_HOME/torch")
export PYTHONPATH=.:.. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NPROC=$(q "$NPROC") WANDB_MODE=$(q "$WANDB_MODE")
cd $(q "$SUBMIT_DIR")
$body
EOF
)
  if [[ "$DRY" == 1 ]]; then
    { echo "---- sbatch $(q "${args[@]}")$script"; echo "$content" | sed 's/^/    | /'; } >&2
    echo "<${name}>"
    return
  fi
  mkdir -p "$LOG_DIR" "$SCRIPT_DIR"
  printf '%s\n' "$content" > "$script"
  local jid
  jid=$(sbatch "${args[@]}" "$script")
  echo "submitted rb_${name} as job ${jid}${dep:+ (dependency ${dep})}" >&2
  echo "$jid"
}
note() { echo "[$1] $2" >&2; }

# ------------------------------------------------------------ status checks --
ext_out() { echo "$RESULTS_ROOT/external/$1"; }
status_external() { if [[ -s "$(ext_out "$1")/results.json" ]]; then echo "done"; else echo "todo"; fi; }
rt_out() { echo "$RESULTS_ROOT/roundtrip/cfg$1/$2"; }
edit_out() { echo "$RESULTS_ROOT/editing/cfg$1"; }
edit_complete() { [[ -s "$1/manifest.json" ]] && grep -q '"complete": true' "$1/manifest.json"; }
bidir_eval_done() { have "$BIDIR_OUT/eval_$1/*/eval/eval_metrics.json"; }
bidir_eval_steps() {
  echo "$SPECIALIST_STEPS"
  if (( BIDIR_TRAIN_STEPS != SPECIALIST_STEPS )); then echo "$BIDIR_TRAIN_STEPS"; fi
}
flow_root() { echo "$FLOW_ROOT_BASE/abl_flow_matching_$1"; }
flow_final_ckpt() { printf '%s/*/checkpoints/step_%07d.pt' "$(flow_root "$1")" "$FLOW_STEPS"; }
flow_eval_done() { have "$(flow_root "$1")/*/eval/eval_metrics.json"; }

status_line() {
  local key=$1 s="" w v
  case "$key" in
    flowtok) s="$(status_external flowtok_xl): $(ext_out flowtok_xl)/results.json" ;;
    crossflow) s="$(status_external crossflow_256_clip): $(ext_out crossflow_256_clip)/results.json" ;;
    roundtrip_cfg|roundtrip_n2d_cond)
      local prefix=d2d; [[ "$key" == roundtrip_n2d_cond ]] && prefix=n2d_cond
      for w in $(rt_scales "$key"); do for v in image text; do
        [[ -s "$(rt_out "$w" "${prefix}_$v")/results.json" ]] && s+="cfg$w/$v=done " || s+="cfg$w/$v=todo "
      done; done ;;
    editing_cfg)
      for w in $EDIT_CFG_SCALES; do
        if edit_complete "$(edit_out "$w")"; then
          [[ -s "$(edit_out "$w")/figure5_cfg$w.pdf" ]] && s+="cfg$w=done " || s+="cfg$w=figure-todo "
        else s+="cfg$w=todo "; fi
      done ;;
    bidir_400k)
      local ck
      printf -v ck 'step_%07d.pt' "$BIDIR_TRAIN_STEPS"
      have "$BIDIR_OUT/train/*/checkpoints/$ck" && s+="train$BIDIR_TRAIN_STEPS=done " || s+="train$BIDIR_TRAIN_STEPS=todo "
      for v in $(bidir_eval_steps); do bidir_eval_done "$v" && s+="eval$v=done " || s+="eval$v=todo "; done
      s+="($BIDIR_OUT)" ;;
    noisy_flow)
      for v in $FLOW_VARIANTS; do
        if flow_eval_done "$v"; then s+="$v=done "
        elif have "$(flow_final_ckpt "$v")"; then s+="$v=eval-todo "
        else s+="$v=train-todo "; fi
      done
      s+="($FLOW_ROOT_BASE)" ;;
  esac
  echo "$s"
}

rt_scales() {  # n2d_cond also runs the unguided reference
  if [[ "$1" == roundtrip_n2d_cond ]]; then
    printf '%s\n' 0.0 $ROUNDTRIP_CFG_SCALES | awk '!seen[$0+0]++' | tr '\n' ' '
  else
    echo "$ROUNDTRIP_CFG_SCALES"
  fi
}

if [[ "$LIST" == 1 ]]; then
  for key in "${ALL_EXPERIMENTS[@]}"; do
    mark=" "; selected "$key" && mark="*"
    printf '%s %-19s %s\n      %s\n' "$mark" "$key" "${DESCRIPTION[$key]}" "$(status_line "$key")"
  done
  echo "(* = selected by the current options)"
  exit 0
fi
(( ${#SELECTED[@]} )) || { echo "Nothing selected." >&2; exit 0; }
echo "Selected: ${SELECTED[*]}  (reservation=${RESERVATION:-none}, account=$ACCOUNT, results=$RESULTS_ROOT)" >&2

# ------------------------------------------------------------- preflight ---
if selected flowtok || selected crossflow; then
  if [[ ! -e "$EXT_ROOT/.setup_complete" ]]; then
    if [[ "$DRY" == 1 ]]; then
      note setup "would run _setup_external_baselines.sh (EXT_ROOT=$EXT_ROOT)"
    else
      note setup "fetching FlowTok/CrossFlow code and checkpoints into $EXT_ROOT"
      PREFIX_DIR="$PREFIX_DIR" EXT_ROOT="$EXT_ROOT" HF_HOME="$HF_HOME" bash _setup_external_baselines.sh || {
        echo "External-baseline setup failed; fix it (see message above) or use --exclude flowtok,crossflow." >&2
        exit 1; }
    fi
  fi
fi

# ------------------------------------------------------------- experiments --
queue_external() {  # key model name extra-args...
  local key=$1 model=$2 name=$3; shift 3
  local out; out=$(ext_out "$name")
  if [[ "$(status_external "$name")" == "done" && "$FORCE" != 1 ]]; then
    note "$key" "results exist: $out/results.json"; return; fi
  local cmd=(torchrun --standalone --nproc-per-node="$NPROC" external_baselines.py --model "$model"
    --data-root "$EVAL_DATA_ROOT" --split-seed "$EXT_SPLIT_SEED" --num-samples "$EXT_NUM_SAMPLES"
    --batch-size "$EXT_BATCH" --out "$out" "$@")
  [[ "$FORCE" == 1 ]] && cmd+=(--overwrite)
  SUBMITTED+=("$(submit "$name" "" 1 "$EXT_TIME" "$(q "${cmd[@]}")")")
}

queue_roundtrip() {  # key
  local key=$1 w v jids=() prefix=d2d fwd=$RT_FWD rev=$RT_REV
  if [[ "$key" == roundtrip_n2d_cond ]]; then prefix=n2d_cond fwd=$RT_N2D_FWD rev=$RT_N2D_REV; fi
  for w in $(rt_scales "$key"); do
    for v in image text; do
      local name="${prefix}_$v" out; out=$(rt_out "$w" "$name")
      if [[ -s "$out/results.json" && "$FORCE" != 1 ]]; then
        note "$key" "results exist: $out/results.json"; continue; fi
      # shellcheck disable=SC2206  # RT_TIME_FRACTIONS is a word list
      local cmd=(torchrun --standalone --nproc-per-node="$NPROC" editing_experiment.py
        --forward-ckpt "$fwd" --reverse-ckpt "$rev" --generate-modality "$v"
        --proportional-nfe --infer-nfe "$RT_NFE" --restore-nfe "$RT_NFE" --noise-nfe "$RT_NFE"
        --data-root "$EDIT_DATA_ROOT" --num-images "$RT_NUM_IMAGES" --num-variations "$RT_NUM_VARIATIONS"
        --time-fractions $RT_TIME_FRACTIONS --cfg-scale "$w" --viz-every 10
        --out "$out/results.json" --wandb-project bib-editing --wandb-name "rebuttal_${name}_cfg$w"
        --wandb-mode "$WANDB_MODE" --plot)
      [[ "$prefix" == d2d ]] && cmd+=(--cache-perturbed-samples)
      jids+=("$(submit "rt_${name}_cfg$w" "" 1 "$RT_TIME" "mkdir -p $(q "$out")"$'\n'"$(q "${cmd[@]}")")")
    done
  done
  (( ${#jids[@]} )) || return 0
  SUBMITTED+=("${jids[@]}")
  # Overlay every finished round-trip run (plus optional existing Fig. 3 JSONs).
  local plot_body="shopt -s nullglob"$'\n'
  for v in image text; do
    local base=$ROUNDTRIP_BASELINE_IMAGE_JSONS; [[ "$v" == text ]] && base=$ROUNDTRIP_BASELINE_TEXT_JSONS
    plot_body+="runs=($base $(q1 "$RESULTS_ROOT/roundtrip")/cfg*/*_$v/results.json)"$'\n'
    plot_body+="if (( \${#runs[@]} )); then python editing_plot.py --results \"\${runs[@]}\" --out $(q "$RESULTS_ROOT/roundtrip/compare_$v.png"); fi"$'\n'
  done
  local dep
  dep="afterany:$(IFS=:; echo "${jids[*]}")"
  [[ "$DRY" == 1 ]] && dep="afterany:<round-trip jobs>"
  SUBMITTED+=("$(submit "rt_plot_${key}" "$dep" 1 00:20:00 "$plot_body")")
}

queue_editing() {
  local w
  for w in $EDIT_CFG_SCALES; do
    local out; out=$(edit_out "$w")
    local fig=(python editing_figure.py --run-dir "$out" --out "$out/figure5_cfg$w.pdf" "$out/figure5_cfg$w.png"
      --contact-sheet "$out/contact_sheet.png" --title "")
    # shellcheck disable=SC2206
    [[ -n "$FIG5_IMAGES" ]] && fig+=(--images $FIG5_IMAGES)
    local run=(python perturbation_slerp_experiment.py --forward-ckpt "$EDIT_CKPT" --data-root "$EDIT_DATA_ROOT"
      --num-perturb-tokens 16 --max-token-length 64 --perturb-mode llm --llm-max-tries 5
      --nfe 400 --infer-nfe 400 --infer-noise --batch-size 4 --num-slerp 5
      --cfg-scale "$w" --num-images "$EDIT_NUM_IMAGES" --text-source image --out "$out")
    if edit_complete "$out" && [[ "$FORCE" != 1 ]]; then
      if [[ -s "$out/figure5_cfg$w.pdf" ]]; then note editing_cfg "results exist: $out"; continue; fi
      SUBMITTED+=("$(submit "edit_figure_cfg$w" "" 1 00:20:00 "$(q "${fig[@]}")")")
      continue
    fi
    SUBMITTED+=("$(submit "edit_cfg$w" "" 1 "$EDIT_TIME" "$(q "${run[@]}")"$'\n'"$(q "${fig[@]}")")")
  done
}

queue_bidir() {
  local v all_done=1
  for v in $(bidir_eval_steps); do bidir_eval_done "$v" || all_done=0; done
  if [[ "$all_done" == 1 && "$FORCE" != 1 ]]; then
    note bidir_400k "evaluations at $(bidir_eval_steps | tr '\n' ' ')exist under $BIDIR_OUT"; return; fi
  local ck n=$BIDIR_CHAIN prev="" k
  printf -v ck 'step_%07d.pt' "$BIDIR_TRAIN_STEPS"
  have "$BIDIR_OUT/train/*/checkpoints/$ck" && n=1   # only evaluations remain
  local env=(OUT_DIR="$BIDIR_OUT" SPECIALIST_STEPS="$SPECIALIST_STEPS"
    TRAIN_STEPS="$BIDIR_TRAIN_STEPS" EVAL_CFG_SCALES="$BIDIR_EVAL_CFG_SCALES"
    TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-$PREFIX_DIR/datasets/text_to_image/gpic_latents/train}"
    EVAL_DATA_ROOT="$EVAL_DATA_ROOT"
    DINO_DIR="${DINO_DIR:-$PREFIX_DIR/datasets/text_to_image/gpic_latents_dino/train}")
  [[ "$FORCE" == 1 ]] && env+=(FORCE_EVAL=1)
  local body
  # shellcheck disable=SC2086  # BIDIR_ARGS is a word list
  body="$(q env "${env[@]}" bash _launch_bit_bidirectional.sh $BIDIR_ARGS)"
  for ((k = 1; k <= n; k++)); do
    prev=$(submit "bidir_400k_part$k" "${prev:+afterany:$prev}" "$BIDIR_NODES" "$BIDIR_TIME" "$body" requeue)
    SUBMITTED+=("$prev")
  done
}

queue_flow() {
  local v
  for v in $FLOW_VARIANTS; do
    local root; root=$(flow_root "$v")
    if flow_eval_done "$v" && [[ "$FORCE" != 1 ]]; then
      note noisy_flow "$v evaluation exists under $root"; continue; fi
    local prev="" k
    if ! have "$(flow_final_ckpt "$v")"; then
      for ((k = 1; k <= FLOW_CHAIN; k++)); do
        prev=$(submit "flow_${v}_train$k" "${prev:+afterany:$prev}" "$FLOW_NODES" "$FLOW_TRAIN_TIME" \
          "$(q env VARIANT="$v" MODE=train OUT_DIR="$root" bash _sbatch_flow.sh)" requeue)
        SUBMITTED+=("$prev")
      done
    fi
    SUBMITTED+=("$(submit "flow_${v}_eval" "${prev:+afterany:$prev}" "$FLOW_NODES" "$FLOW_EVAL_TIME" \
      "$(q env VARIANT="$v" MODE=eval OUT_DIR="$root" REQUIRE_STEP="$FLOW_STEPS" bash _sbatch_flow.sh)")")
  done
}

for key in "${SELECTED[@]}"; do
  case "$key" in
    flowtok) queue_external flowtok flowtok flowtok_xl --repo "$EXT_ROOT/FlowTok" \
      --checkpoint "$EXT_ROOT/checkpoints/FlowTok-XL.pth" \
      --tokenizer-checkpoint "$EXT_ROOT/checkpoints/FlowTiTok_512.bin" \
      --steps "$FLOWTOK_STEPS" --cfg-scale "$FLOWTOK_CFG" ;;
    crossflow) queue_external crossflow crossflow crossflow_256_clip --repo "$EXT_ROOT/CrossFlow" \
      --checkpoint "$EXT_ROOT/checkpoints/pretrained_models/t2i_256px_clip_dimr.pth" \
      --autoencoder-checkpoint "$EXT_ROOT/checkpoints/assets/stable-diffusion/autoencoder_kl.pth" \
      --steps "$CROSSFLOW_STEPS" --cfg-scale "$CROSSFLOW_CFG" ;;
    roundtrip_cfg|roundtrip_n2d_cond) queue_roundtrip "$key" ;;
    editing_cfg) queue_editing ;;
    bidir_400k) queue_bidir ;;
    noisy_flow) queue_flow ;;
  esac
done

if (( ${#SUBMITTED[@]} )); then
  echo "Queued ${#SUBMITTED[@]} job(s): ${SUBMITTED[*]}" >&2
  if [[ "$DRY" == 1 ]]; then echo "(dry run: nothing was submitted)" >&2; fi
else
  echo "Nothing to submit: every selected experiment already has results (use --force to rerun evaluations)." >&2
fi
