# Invertible Text–Image Diffusion Bridges

Research code for bidirectional diffusion bridges between text and images, building on [ABC Diffusion](https://github.com/gabeguo/abc_diffusion).

A single cross-attention Diffusion Transformer learns both:

- **Text → image:** token embeddings to VAE image latents
- **Image → text:** VAE image latents to token embeddings

The repository includes distributed training, evaluation, data encoding, cross-modal editing, and text-embedding perturbation experiments.

## Method

Text token embeddings are packed into the same tensor geometry as image latents, allowing one DiT to transport samples between modalities.

| Preset | Text representation | Image latent | VAE |
|---|---|---|---|
| `sd` | 64 × 64 token features | 4 × 32 × 32 | `stabilityai/sd-vae-ft-mse` |
| `flux` *(not tested)* | 128 × 128 token features | 16 × 32 × 32 | FLUX.1-dev VAE |

Implemented transport processes include:

- Uniform-volatility diffusion bridges
- Periodic-volatility diffusion bridges
- Cosine-decaying volatility
- Rectified flow matching *(baseline that fails by deterministic construction)*
- Optional EDM-style preconditioning *(not extensively tested)*
- Forward-only, reverse-only, and bidirectional training
- Noise-to-data ablations
- Text and image REPA objectives

## Repository layout

```text
sde_utils/
├── loss.py                 # DSM, flow-matching, EDM, and REPA losses
├── precond.py              # EDM-style bridge preconditioning *(not extensively tested)*
└── sde.py                  # SDE and flow-matching dynamics

text_to_image/
├── train.py                # Distributed training and evaluation
├── token_bridge.py         # Token/latent bridge representations
├── eval_metrics.py         # FID and image-to-text metrics
├── eval_plot.py            # Samples and trajectory visualizations
├── editing_experiment.py   # Cross-modal round-trip editing
├── perturbation_slerp_experiment.py
├── models/
│   ├── dit.py              # DiT and cross-attention DiT models
│   └── token_decoder.py    # Token-embedding decoder
└── data_utils/
    ├── encode_gpic.py      # Creates the dataset embeddings
    ├── encode_dino_features.py     # Gets REPA features
    ├── encode_global_text.py       # *(not extensively tested)*
    ├── latent_dataset.py           # Dataset loading code
    └── encode_common_catalog.py    # *legacy code*
```

## Installation

A CUDA-enabled PyTorch installation is required. Dependency versions are currently unpinned.

```bash
pip install torch torchvision
pip install \
  diffusers transformers timm wandb \
  numpy pyarrow pillow tqdm matplotlib \
  imageio imageio-ffmpeg opencv-python-headless \
  torchmetrics torch-fidelity pycocoevalcap \
  huggingface-hub
```

## Data preparation

The encoder consumes WebDataset tar files from
[GPIC](https://huggingface.co/datasets/stanford-vision-lab/gpic) and writes
sharded memory-mapped VAE latents, token IDs, masks, and metadata.

Run commands from `text_to_image/`:

```bash
cd text_to_image

PYTHONPATH=.:..:data_utils python data_utils/encode_gpic.py \
  --tars-dir /path/to/gpic/train \
  --output-dir /path/to/gpic_latents/train \
  --bridge-preset sd \
  --splits train
```

The encoder is designed for one process per GPU and can also be launched with
SLURM `srun`. See `new_dataset/encode_gpic.sh` for a multi-node example ***(will need to change the directories)***.

### Optional DINOv2 sidecars

DINOv2 patch features are required for image REPA:

```bash
PYTHONPATH=.:.. python -m data_utils.encode_dino_features \
  --source-root /path/to/gpic_latents/train \
  --dino-dir /path/to/gpic_dino/train \
  --dino-model dinov2_vitb14 \
  --store patch
```

See `_encode_dino.sh` for a multi-node example ***(will need to change the directories)***.

## Training

The training entry point uses NCCL distributed data parallelism and should be
started with `torchrun`:

```bash
cd text_to_image

PYTHONPATH=.:.. torchrun --standalone --nproc-per-node=8 train.py \
  --data-root /path/to/gpic_latents/train \
  --out-dir /path/to/results \
  --use-token-text-bridge \
  --token-layout row_major \
  --x0-cond-source x0 \
  --model DiTXA-L/2 \
  --sde periodic \
  --periodic_sde_alpha 0.95 \
  --periodic_sde_k 1.0 \
  --periodic_sde_eps 0.05 \
  --global-batch-size 512 \
  --wandb-name bridge-training
```

Image REPA can be enabled with:

```text
--repa-image --dino-dir /path/to/gpic_dino/train
```

For multi-node SLURM training, adapt `_multi_node_train.sh`. The committed
launch scripts contain site-specific accounts, reservations, environments, and
filesystem paths that must be changed for your cluster.

## Evaluation

Training periodically evaluates:

- Text-to-image FID
- CLIPScore
- Image-to-text token accuracy
- CIDEr, when installed
- Generative perplexity
- Decoded images, captions, and bridge trajectory videos

Evaluate a checkpoint without further training:

```bash
PYTHONPATH=.:.. torchrun --standalone --nproc-per-node=8 train.py \
  --data-root /path/to/eval_latents \
  --out-dir /path/to/eval_results \
  --resume /path/to/checkpoint.pt \
  --eval-only \
  [the same model, bridge, SDE, and direction arguments used for training]
```

Checkpoint semantics are validated during `--eval-only`; incompatible model or
bridge arguments are rejected.

### Checkpoints

Our checkpoints may also be referenced as:

```text
hf://therealgabeguo/BiB_generative/path/to/checkpoint.pt
```

Visit [https://huggingface.co/therealgabeguo/BiB_generative](https://huggingface.co/therealgabeguo/BiB_generative).

## Cross-modal editing

`editing_experiment.py` performs SDEdit-style round trips. It partially moves a
real sample toward the opposite modality and then returns to the source
modality, measuring fidelity and diversity along the way.

```bash
PYTHONPATH=.:.. python editing_experiment.py \
  --forward-ckpt /path/to/forward.pt \
  --reverse-ckpt /path/to/reverse.pt \
  --generate-modality image \
  --proportional-nfe \
  --restore-nfe 250 \
  --noise-nfe 250 \
  --data-root /path/to/eval_latents \
  --num-images 256 \
  --num-variations 16 \
  --time-fractions 0.0 0.25 0.5 0.75 1.0 \
  --out editing_results/results.json \
  --plot
```

Compare several runs:

```bash
python editing_plot.py \
  --results run_a/results.json run_b/results.json \
  --out comparison.png
```

See `_launch_editing_experiment.sh` for SLURM version.

## Token perturbation experiments

`perturbation_slerp_experiment.py` perturbs text token embeddings on the unit
hypersphere, interpolates between the original and perturbed captions, and
generates images with a fixed Brownian path.

```bash
PYTHONPATH=.:.. python perturbation_slerp_experiment.py \
    --forward-ckpt /path/to/forward.pt \
    --data-root /path/to/eval_latents \
    --num-perturb-tokens 16 \
    --max-token-length 64 \
    --perturb-mode llm \
    --llm-max-tries 5 \
    --nfe 400 \
    --infer-nfe 400 \
    --infer-noise \
    --batch-size 4 \
    --num-slerp 5 \
    --cfg-scale 0 \
    --num-images 64 \
    --text-source image \
    --out perturbation_results
```

Both angular perturbations and LLM-generated caption rewrites are supported.

## Acknowledgements

This repository builds on:

- [ABC Diffusion](https://github.com/gabeguo/abc_diffusion)
- [DiT](https://github.com/facebookresearch/DiT)