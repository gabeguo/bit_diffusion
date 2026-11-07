# COCO evaluation

Use the existing training environment, plus:

```bash
pip install -r requirements-coco.txt
# Install Java 8 in your environment; confirm `java -version` shows 1.8.
python data_utils/prepare_spice.py --cache-dir /data/cache/spice
python data_utils/download_coco.py --root /data/coco
```

The downloader uses public COCO-hosted 2014 validation images (~6.65 GB zip),
official train/validation annotations (~253 MB zip), and the
[BLIP-distributed Karpathy test split](https://github.com/salesforce/BLIP#pre-training-datasets-download).
Allow space for archives and extracted files. Downloads resume after interruption;
zip extraction checks CRCs. `--metadata-only` skips images. The SPICE setup uses
the original Stanford CoreNLP 3.6 jars over HTTPS, avoiding the upstream package's
old HTTP downloader. Prepare it before running on offline nodes; the installed
`pycocoevalcap/spice` directory must be writable for its cache/temp files. SPICE
uses a Java heap of up to 8 GB. Newer Java versions may fail with reflective-access
errors; the supported upstream setup is Java 8.

## Generate once, score afterward

From `text_to_image/`, use the same launcher for SDE, original flow, and noisy flow:

```bash
CHECKPOINT=/data/run/checkpoints/step_0200000.pt DATA_ROOT=/data/training_manifests \
  COCO_ROOT=/data/coco OUT_DIR=/data/eval/model_t2i NPROC=8 \
  bash _eval_coco.sh t2i

CHECKPOINT=/data/reverse_run/checkpoints/step_0200000.pt DATA_ROOT=/data/training_manifests \
  COCO_ROOT=/data/coco OUT_DIR=/data/eval/model_i2t NPROC=8 \
  bash _eval_coco.sh i2t
```

`DATA_ROOT` must contain the same training `config.json`,
`token_embed_config.json`, and (for T2I) token vocabulary `.npy` table; the loader
uses the checkpoint's EMA weights and trained token decoder. T2I currently
supports the repository's table-based token representation. It truncates/pads
prompts to the checkpoint bridge length, normalizes each token vector, and uses
the training scale. I2T uses training-compatible area resize/center crop and the
VAE posterior mean, including upsampling small COCO images without dropping them.
The SD preset generates 256×256 images. FLUX follows its manifest but, like the
rest of this repository, remains unvalidated here.

Defaults: seed 0, 500 Euler steps, CFG 0, one output per selected prompt/image.
Set `EVAL_STEPS`, `CFG_SCALE`, and `SEED` identically across comparisons. Pass
`--ode` after the task for an SDE checkpoint's probability-flow ODE; flow matching
always uses an ODE. Positive CFG costs two network evaluations per step. The
original unconditional flow only supports CFG 0. For a smoke run append
`--num-samples 8 --steps 2`; those are **not benchmark scores**.

Generation requires a new `OUT_DIR` and saves `protocol.json`, per-rank completion
receipts, PNGs or captions. Do not reuse a partially completed directory: use a
new one. Scoring rejects missing/mixed shards and duplicate/missing outputs.
The manifest records exact caption/image IDs, annotation/split hashes, checkpoint
arguments and identity, steps, CFG, seed, batch size, world size, and preprocessing.
Keep batch size/world size fixed for reproducible random draws. T2I and I2T are
separate runs so different directional checkpoints can be evaluated fairly.

On SLURM, submit `_sbatch_coco_eval.sh` instead; it generates across nodes and
then scores once:

```bash
sbatch --job-name=coco_t2i --nodes=2 \
  --export=ALL,TASK=t2i,CHECKPOINT=/data/run/checkpoints/step_0200000.pt,DATA_ROOT=/data/training_manifests,COCO_ROOT=/data/coco,OUT_DIR=/data/eval/model_t2i \
  _sbatch_coco_eval.sh
```

`_eval_coco.sh` switches to a c10d rendezvous when `SLURM_NNODES` exceeds 1, so
multi-node generation must run under `srun --ntasks-per-node=1`. It also refuses
the default `PHASE=all` there: generation writes independent shards with no
collectives, so scoring is only safe after every node's task has exited, and
`coco_eval.py score` rejects a non-unit `WORLD_SIZE` anyway. The wrapper
therefore runs `PHASE=generate` under `srun` and `PHASE=score` afterward.
Keep the node count fixed across compared runs, since the manifest records
world size and the random draws depend on it. The job has no retry loop or
`--requeue`, because a partially generated `OUT_DIR` cannot be reused.

To rerun scoring without loading a checkpoint or generating again:

```bash
PYTHONPATH=.:.. python coco_eval.py score --coco-root /data/coco \
  --output /data/eval/model_t2i --fid-mode clean --clip-score
PYTHONPATH=.:.. python coco_eval.py score --coco-root /data/coco \
  --output /data/eval/model_i2t
```

`scores.json` records scorer versions and conventions; I2T also exports standard
COCO-format `predictions.json` (`image_id`, `caption`). `--no-spice` explicitly
skips SPICE for a quick check. Requested scorers otherwise fail on missing
dependencies or invalid outputs rather than silently omitting a metric.

## Protocols and published comparisons

| Task | Default sample selection | Scoring |
|---|---|---|
| T2I | 30,000 distinct caption annotations sampled without replacement from COCO 2014 validation; multiple captions may refer to one image | FID against **all original val2014 JPEGs**; optional CLIP B/16 cosine ×100 |
| I2T | Canonical Karpathy 5,000 test image IDs | One prediction per image, all official reference captions, COCO PTB-tokenized CIDEr and SPICE |

T2I follows the 30k validation-caption/full-validation-reference pattern described
by [Imagen, §3](https://arxiv.org/abs/2205.11487). It saves its own selected prompt
IDs; this does not claim to recover an unpublished paper's exact sample list.
FID defaults to `clean-fid==0.1.35`, `mode=clean`, Inception 2048 features, using
original reference images directly. `legacy_pytorch` and `legacy_tensorflow` modes
are available. Choose the **same implementation, resizing, reference statistics,
image resolution and sample protocol** as a cited result; changing the mode name
alone does not prove equivalence. The existing GPIC evaluation continues to use
VAE-reconstructed real images and therefore measures a different quantity.
The optional CLIP metric is explicitly named `clip_b16_cosine_x100`; do not equate
it with papers using different backbones, clamping, scaling or prompt handling.

Caption scoring uses the unmodified [COCOEvalCap implementation](https://github.com/salaniz/pycocoevalcap):
PTB tokenization before the standard `Cider` and `Spice` scorers. CIDEr's IDF is
computed on the entire selected reference corpus, not per batch or GPU. `Cider`
uses the COCO implementation's clipping and Gaussian length penalty. Scores are
saved in native units **and** ×100 units (e.g. CIDEr 1.2 → 120, SPICE 0.2 → 20).
`cider_legacy` is also saved: raw whitespace tokenization over the same full COCO
references, to isolate the scorer difference. It is not the old GPIC score,
which has a different dataset and a single truncated reference per image.
The official Karpathy subset has five references for 4,990 images and six for
10 images; all are retained, matching the distributed split annotations.

Published numbers are valid context **only when these protocols match**. Label
COCO-trained/fine-tuned methods separately from models evaluated without COCO
training; verify training-set overlap before claiming zero-shot evaluation.
Karpathy test is not the hidden COCO server test split. A limited smoke subset,
different FID implementation, different split, or different number of references
must be labeled and must not be presented as a matched leaderboard comparison.
For causal comparisons between this project's methods, run them through the same
saved sample selection and scorer settings. These scripts enable that experiment;
they do not establish the resulting model quality in advance.

## Existing training evaluation

Add `--eval-coco-cider --eval-spice` to `train.py` or a flow eval launcher to enable
the additional caption metrics on GPIC. Existing `cider_*` keys and behavior remain
unchanged. New `cider_coco_*` and `spice_*` keys use the full available caption,
but GPIC still only has **one** reference; these are not COCO benchmark numbers.
Standard scoring runs once on rank 0 after gathering the full corpus, and sends
either scores or an explicit error to all ranks. These opt-ins avoid imposing
Java/SPICE overhead on existing training runs.

## Checks

```bash
OMP_NUM_THREADS=1 PYTHONPATH=.:.. python -m unittest discover -s tests -v
RUN_COCO_METRICS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:.. \
  python -m unittest discover -s tests -v
```

The second command also checks actual PTB/CIDEr/SPICE parity against the upstream
scorers and requires Java and the Stanford models. Tests cover noise direction,
zero-noise RNG compatibility, clean conditioning/decoder training, checkpoint
defaults, COCO sample/shard integrity, and resumable/safe archive handling. Full
GPU training and full-dataset generation/scoring must be run in the experiment
environment with the project's checkpoints and training manifests.
