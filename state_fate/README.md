# LARRY state-fate benchmark

This benchmark learns bidirectional maps between day-2 progenitors and day-6
descendants in the LARRY lineage-tracing data. Endpoint pairs share a clone
barcode and are represented by 64-dimensional gene-expression latents.

## Protocol

Preprocessing splits clones, rather than individual pairs, to prevent lineage
leakage. With the default in-vitro configuration and data seed 0, it produces
31,840 training pairs and 3,968 test pairs. A second clone partition is reserved
and is not exposed to training or evaluation.

There is no validation split or checkpoint selection. Each method trains for a
fixed 20,000 steps, checkpoints every 5,000 steps, and reports the final
checkpoint. Intermediate test-batch evaluations are diagnostic only. The test
batch contains 512 pairs.

`--batch-size` is per process. The four-GPU launchers therefore use an effective
global batch size of 2,048 when the default value is 512.

## Installation

Run commands from the repository root:

```bash
pip install -r state_fate/requirements.txt
```

## Data preparation

Download the public LARRY files:

```bash
PYTHONPATH=. python state_fate/data_utils/download_larry.py \
  --dataset in_vitro \
  --raw-root state_fate/data/raw
```

Prepare clone-paired endpoints:

```bash
PYTHONPATH=. python state_fate/data_utils/prepare_larry.py \
  --dataset in_vitro \
  --raw-root state_fate/data/raw \
  --out-dir state_fate/data/processed/larry_in_vitro_d2_d6 \
  --early-day 2 \
  --late-day 6 \
  --n-hvgs 2000 \
  --latent-dim 64 \
  --pairs-per-clone 32 \
  --seed 0
```

The processed directory contains `latents.npy`, `pairs.npz`, compressed cell
metadata, and a JSON record of preprocessing parameters and label mappings.

## Training

A local training run can be launched with:

```bash
state_fate/scripts/train_larry_in_vitro.sh
```

The training entry point also supports direct invocation:

```bash
PYTHONPATH=. python state_fate/train.py \
  --data-root state_fate/data/processed/larry_in_vitro_d2_d6 \
  --out-dir state_fate/runs/larry_in_vitro \
  --steps 20000
```

The full benchmark compares six configurations:

- BIT with a shared network and uniform volatility
- BIT with separate forward and reverse networks
- BIT with a shared network and cosine-decay volatility
- conditional noise-to-data diffusion
- rectified flow
- endpoint regression

### Atlas

The Atlas launcher is site-specific but contains no user paths or node pins.
Set `ENV_PREFIX` to an existing Python environment, or set `PYTHON_BIN` and
`TORCHRUN_BIN` directly:

```bash
ENV_PREFIX=/path/to/environment SEED=1 \
  sbatch state_fate/scripts/train_larry_sc_atlas_benchmark.sh
```

The launcher accepts `ROOT_DIR`, `RAW_ROOT`, `PROCESSED_DIR`, `RUN_ROOT`,
`SEED`, and `DATA_SEED`. Set `INSTALL_DEPS=1` to install dependencies inside the
allocation. HTTPS verification remains enabled unless `INSECURE_SSL=1` is set
explicitly.

### Perlmutter

```bash
ROOT_DIR=/path/to/bit_diffusion \
PROCESSED_DIR=/path/to/larry_in_vitro_d2_d6 \
RUN_ROOT=/path/to/larry_runs \
  sbatch state_fate/scripts/train_larry_perlmutter.sh
```

## Evaluation

Evaluate a saved checkpoint on the fixed test batch:

```bash
PYTHONPATH=. python state_fate/evaluate.py \
  --data-root state_fate/data/processed/larry_in_vitro_d2_d6 \
  --checkpoint state_fate/runs/larry_in_vitro/<run>/checkpoints/step_0020000.pt
```

Evaluation reports forward and reverse MSE, RBF-MMD, fate and clone nearest-
neighbor accuracy, and cycle MSE. It also saves the generated tensors and
trajectory figures. Cached tensors can be replotted without sampling again:

```bash
PYTHONPATH=. python state_fate/evaluate.py \
  --replot \
  --out-dir state_fate/runs/larry_in_vitro/<run>/test
```

## Data source

The downloader uses the public files released with the LARRY state-fate study
by Weinreb et al. (2020):

- in-vitro differentiation
- cytokine perturbation
- in-vivo differentiation
