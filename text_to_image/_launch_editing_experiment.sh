#!/bin/bash
#
# Launches the cross-modal round-trip editing experiment as ONE SLURM job per
# configuration, so nothing is blocked by sequential processing. A final plot
# job is submitted with an afterany dependency on all runs and draws the
# image-modality and text-modality comparison figures.
#
# Configurations (see editing_experiment.py for what each mode does):
#   (1)   data-to-data, two models      -> periodic fwd + periodic rev
#   (2.1) noise-to-data, two models      -> cos-decay (noise->image) + (noise->text)
#   (2.2) noise-to-data, single (->text) -> cos-decay reverse only
#   (2.3) noise-to-data, single (->image)-> cos-decay forward only
#   (3)   data-to-data, single (bidir)   -> mega_run bidirectional model
#
# Modality note: (2.2) can only make text and (2.3) can only make image. The
# other three configs are direction-agnostic, so we run them for BOTH image and
# text generation to complete each comparison. That yields 8 experiment jobs +
# 1 plot job. Drop lines from the job table below to trim.
#
# Usage:
#   ./_launch_editing_experiment.sh           # submit all jobs
#   DRYRUN=1 ./_launch_editing_experiment.sh  # print, submit nothing
#
# Run from the text_to_image/ directory.

set -euo pipefail

PREFIX_DIR=/pscratch/sd/g/gabeguo # TODO: change to your own
TODAY=$(date +%m_%d_%y)
SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT=${PREFIX_DIR}/BiB_results/${TODAY}/editing_experiment
LOG_DIR=${PREFIX_DIR}/BiB_results/slurm_logs

export HF_HOME=${PREFIX_DIR}/cache_sub

# ---- Checkpoints ----------------------------------------------------------
FWD_BRIDGE="hf://therealgabeguo/BiB_generative/ablations/07_06_26/data_to_data/fwd/step_0200000.pt"
REV_BRIDGE="hf://therealgabeguo/BiB_generative/ablations/07_06_26/data_to_data/rev/step_0200000.pt"
NOISE_TO_IMG_BRIDGE="hf://therealgabeguo/BiB_generative/ablations/07_06_26/noise_to_data/fwd/step_0200000.pt"
NOISE_TO_TXT_BRIDGE="hf://therealgabeguo/BiB_generative/ablations/07_06_26/noise_to_data/rev/step_0200000.pt"
# MEGA=${PREFIX_DIR}/BiB_results/06_29_26/gpic/mega_run_multi_ema/20260701_031004/checkpoints/step_0100000.pt

# ---- Shared experiment knobs (override via env) ---------------------------
QOS=${QOS:-regular}
TIME=${TIME:-2:40:00}
NUM_IMAGES=${NUM_IMAGES:-500}
NUM_VARIATIONS=${NUM_VARIATIONS:-8}
TIME_FRACTIONS=${TIME_FRACTIONS:-"0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"}
NFE_BASE=${NFE_BASE:-250}
VIZ_EVERY=${VIZ_EVERY:-10}
WANDB_PROJECT=${WANDB_PROJECT:-bib-editing}
WANDB_MODE=${WANDB_MODE:-online}
DATA_ROOT=${DATA_ROOT:-${PREFIX_DIR}/datasets/text_to_image/gpic_latents_sd_test/TEST} # TODO: change to your own

# ---- Job table: name | modality | forward-ckpt | reverse-ckpt -------------
# Use "-" for an absent model.
job_names=(
  d2d_two_image  n2d_single_image
  d2d_two_text   n2d_single_text
)
job_modality=(
  image          image
  text           text
)
job_fwd=(
  "${FWD_BRIDGE}" "${NOISE_TO_IMG_BRIDGE}"
  "${FWD_BRIDGE}" "-"
)
job_rev=(
  "${REV_BRIDGE}" "-"
  "${REV_BRIDGE}" "${NOISE_TO_TXT_BRIDGE}"
)

# ---------------------------------------------------------------------------

submit_editing() {
  # echoes the SLURM job id on stdout (empty in DRYRUN); logs go to stderr.
  local name="$1" modality="$2" fwd="$3" rev="$4"
  local out_dir="${RESULTS_ROOT}/${name}"
  local ckpt_flags=""
  [[ "${fwd}" != "-" ]] && ckpt_flags+=" --forward-ckpt ${fwd}"
  [[ "${rev}" != "-" ]] && ckpt_flags+=" --reverse-ckpt ${rev}"

  local cache_perturbed_samples=""
  if [[ "${name}" =~ "d2d" ]]; then
    cache_perturbed_samples="--cache-perturbed-samples"
  fi

  local cmd="${ckpt_flags} \
    --generate-modality ${modality} \
    --proportional-nfe \
    --infer-nfe ${NFE_BASE} \
    --restore-nfe ${NFE_BASE} \
    --noise-nfe ${NFE_BASE} \
    --data-root ${DATA_ROOT} \
    --num-images ${NUM_IMAGES} \
    --num-variations ${NUM_VARIATIONS} \
    --time-fractions ${TIME_FRACTIONS} \
    ${cache_perturbed_samples} \
    --viz-every ${VIZ_EVERY} \
    --out ${out_dir}/results.json \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-name ${name} \
    --wandb-mode ${WANDB_MODE} \
    --plot"

  if [[ "${DRYRUN:-0}" == "1" ]]; then
    { echo "==== ${name} (${modality}) ===="; echo "  python editing_experiment.py ${cmd}"; echo; } >&2
    return
  fi
  mkdir -p "${LOG_DIR}" "${out_dir}"

  # PYTHONPATH=.:.. torchrun --standalone --nproc-per-node=4  editing_experiment.py ${cmd}

  sbatch --parsable --job-name="${name}" --output="${LOG_DIR}/${name}_%j.out" <<EOF
#!/bin/bash
#SBATCH --account=m5319
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=${QOS}
#SBATCH --time=${TIME}
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --requeue

module load python
module load conda
conda activate dit_env

export HF_HOME=${PREFIX_DIR}/cache_sub
export TORCH_HOME=${PREFIX_DIR}/cache_sub/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ${SUBMIT_DIR}

PYTHONPATH=.:.. torchrun --standalone --nproc-per-node=4  editing_experiment.py ${cmd}
EOF
}

submit_plot() {
  # $1 = colon-joined dependency ids; $2.. = nothing (uses globals).
  local dep="$1"
  local dep_flag=""
  [[ -n "${dep}" ]] && dep_flag="--dependency=afterany:${dep}"

  if [[ "${DRYRUN:-0}" == "1" ]]; then
    { echo "==== plot (dep=${dep}) ===="
      echo "  image: ${IMAGE_JSONS}"
      echo "  text : ${TEXT_JSONS}"; echo; } >&2
    return
  fi
  mkdir -p "${LOG_DIR}"
  sbatch --parsable ${dep_flag} --job-name="editing_plot" \
         --output="${LOG_DIR}/editing_plot_%j.out" <<EOF
#!/bin/bash
#SBATCH --account=m5319
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

module load python
module load conda
conda activate dit_env
cd ${SUBMIT_DIR}

PYTHONPATH=.:.. python editing_plot.py --results ${IMAGE_JSONS} --out ${RESULTS_ROOT}/compare_image.png
PYTHONPATH=.:.. python editing_plot.py --results ${TEXT_JSONS}  --out ${RESULTS_ROOT}/compare_text.png
EOF
}

# ---- Submit all experiment jobs, collecting ids + json paths --------------
all_ids=()
IMAGE_JSONS=""
TEXT_JSONS=""
for i in "${!job_names[@]}"; do
  name="${job_names[$i]}"
  modality="${job_modality[$i]}"
  json="${RESULTS_ROOT}/${name}/results.json"
  if [[ ! "${name}" =~ "d2d" ]]; then
    continue
  fi
  jid=$(submit_editing "${name}" "${modality}" "${job_fwd[$i]}" "${job_rev[$i]}")
  [[ -n "${jid}" ]] && all_ids+=("${jid}")
  if [[ "${modality}" == "image" ]]; then IMAGE_JSONS+=" ${json}"; else TEXT_JSONS+=" ${json}"; fi
  echo "submitted ${name} (${modality})${jid:+ as job ${jid}}"
done

# dep=$(IFS=:; echo "${all_ids[*]}")
# pjid=$(submit_plot "${dep}")
# echo "submitted editing_plot${pjid:+ as job ${pjid}} (afterany ${#all_ids[@]} jobs)"
# echo "Done (DRYRUN=${DRYRUN:-0}). Results under ${RESULTS_ROOT}"
