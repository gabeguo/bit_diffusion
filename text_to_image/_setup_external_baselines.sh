#!/bin/bash
# Fetch the official FlowTok and CrossFlow code (pinned commits) and released
# checkpoints, then verify the Python packages their inference code imports.
# Idempotent; run on a node with internet access (e.g. a login node):
#
#   bash _setup_external_baselines.sh                # fetch + verify
#   INSTALL_DEPS=1 bash _setup_external_baselines.sh # also pip-install missing packages
#   DRYRUN=1 bash _setup_external_baselines.sh       # print commands only
#
# Layout under EXT_ROOT (default $PREFIX_DIR/external_baselines):
#   FlowTok/  CrossFlow/                         official repositories
#   checkpoints/FlowTok-XL.pth  checkpoints/FlowTiTok_512.bin
#   checkpoints/pretrained_models/t2i_256px_clip_dimr.pth
#   checkpoints/assets/stable-diffusion/autoencoder_kl.pth  (from assets.tar)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PREFIX_DIR=${PREFIX_DIR:-/pscratch/sd/g/gabeguo}
EXT_ROOT=${EXT_ROOT:-$PREFIX_DIR/external_baselines}
export HF_HOME=${HF_HOME:-$PREFIX_DIR/cache_sub}
FLOWTOK_COMMIT=${FLOWTOK_COMMIT:-d4d43c5e3aa241e9a43855f2facfb55359cd56f2}
CROSSFLOW_COMMIT=${CROSSFLOW_COMMIT:-8493373c6e5c53b2b02f38586a4cf8e45e287c7d}
CKPT=$EXT_ROOT/checkpoints

run() {
  if [[ "${DRYRUN:-0}" == 1 ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}

clone_at() {
  local url=$1 dir=$2 commit=$3
  if [[ ! -d "$dir/.git" ]]; then
    run git clone "$url" "$dir"
  fi
  if [[ "${DRYRUN:-0}" != 1 && "$(git -C "$dir" rev-parse HEAD)" == "$commit" ]]; then
    return
  fi
  run git -C "$dir" fetch --quiet origin
  run git -C "$dir" checkout --quiet "$commit"
}

run mkdir -p "$EXT_ROOT" "$CKPT"
clone_at https://github.com/TACJu/FlowTok.git "$EXT_ROOT/FlowTok" "$FLOWTOK_COMMIT"
clone_at https://github.com/qihao067/CrossFlow.git "$EXT_ROOT/CrossFlow" "$CROSSFLOW_COMMIT"

# Released checkpoints (HF repos named in each README).
fetch() {
  local repo=$1 file=$2
  [[ -s "$CKPT/$file" ]] && return
  run python -c 'import sys; from huggingface_hub import hf_hub_download as d; print(d(repo_id=sys.argv[1], filename=sys.argv[2], local_dir=sys.argv[3]))' \
    "$repo" "$file" "$CKPT"
}
fetch turkeyju/FlowTok FlowTok-XL.pth
fetch turkeyju/FlowTok FlowTiTok_512.bin
fetch QHL067/CrossFlow pretrained_models/t2i_256px_clip_dimr.pth
if [[ ! -s "$CKPT/assets/stable-diffusion/autoencoder_kl.pth" ]]; then
  fetch QHL067/CrossFlow assets.tar
  run tar -xf "$CKPT/assets.tar" -C "$CKPT"
  if [[ "${DRYRUN:-0}" != 1 && ! -s "$CKPT/assets/stable-diffusion/autoencoder_kl.pth" ]]; then
    echo "assets.tar did not contain assets/stable-diffusion/autoencoder_kl.pth; inspect $CKPT" >&2
    exit 1
  fi
fi

# Python packages imported by the external inference code.
missing=$(python - <<'EOF'
import importlib.util
mods = {"open_clip": "open_clip_torch", "ml_collections": "ml_collections", "absl": "absl-py",
        "einops": "einops", "timm": "timm", "ftfy": "ftfy", "torchdiffeq": "torchdiffeq",
        "fairscale": "fairscale", "huggingface_hub": "huggingface_hub"}
print(" ".join(pkg for mod, pkg in mods.items() if importlib.util.find_spec(mod) is None))
EOF
)
if [[ -n "$missing" ]]; then
  if [[ "${INSTALL_DEPS:-0}" == 1 ]]; then
    run python -m pip install $missing
  else
    echo "Missing Python packages for the external baselines: $missing" >&2
    echo "Install them in this environment (INSTALL_DEPS=1 bash $0, or pip install -r requirements-external-baselines.txt)." >&2
    exit 1
  fi
fi

# Pre-download the text encoders their samplers construct, for offline nodes.
run python - <<'EOF'
import open_clip
from transformers import CLIPTextModel, CLIPTokenizer
open_clip.create_model_and_transforms("ViT-L-14-336", pretrained="openai")  # FlowTok
CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")             # CrossFlow
CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
print("text encoders cached")
EOF
[[ "${DRYRUN:-0}" == 1 ]] || touch "$EXT_ROOT/.setup_complete"
echo "External baselines ready under $EXT_ROOT"
