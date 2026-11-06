# Unified BIT and random-token controls

These launchers extend the periodic-SDE **data-to-data** controls in
`_launch_ablation_sweep.sh`. They train and evaluate only the new models.
No trainer, model, loss, or evaluation Python changes are needed.

## Existing specialist baselines

Both specialists train for 200,000 optimizer steps. Their training commands
are in `_launch_ablation_sweep.sh`; their held-out evaluation commands are in
`_launch_ablation_eval.sh`. The following checkpoint files were verified to
exist on Hugging Face on September 22, 2026. This verifies checkpoint availability,
not successful completion of prior evaluation jobs or their scores.

| Direction | Training flag | Checkpoint in `therealgabeguo/BiB_generative` |
|---|---|---|
| Text → image | `--no-reverse` | `ablations/07_06_26/data_to_data/fwd/step_0200000.pt` |
| Image → text | `--no-forward` | `ablations/07_16_26/data_to_data/rev_repa/step_0200000.pt` |

Use the **REPA reverse baseline** above, not the older reverse checkpoint that
is commented out in the evaluation script. Historical checkpoint hyperparameters
and evaluation results should still be checked against the current recipe before
reporting a matched comparison.

## Run

From the repository root, with the usual training environment active:

```bash
bash text_to_image/_launch_bit_bidirectional.sh
bash text_to_image/_launch_bit_random_tokens.sh
```

Both scripts also accept `sbatch` from the repository root or `text_to_image/`:

```bash
sbatch text_to_image/_launch_bit_bidirectional.sh
sbatch text_to_image/_launch_bit_random_tokens.sh
```

Each allocation runs its phases sequentially. Defaults are eight GPUs locally,
or two Perlmutter nodes with four A100s each, account `m5319`, no reservation,
and `dit_env`. Standard `sbatch --account=... --time=...` overrides apply.
Training uses `--auto-resume`, and a failed phase stops the script before any
subsequent evaluation. Rerunning resumes training; completed evaluations can
run again. Wall-time termination requires resubmission. `EVAL_ONLY=1` skips
training and evaluates the required existing checkpoints.

Environment settings:

| Setting | Purpose/default |
|---|---|
| `PREFIX_DIR` | `/data` locally; `/pscratch/sd/g/gabeguo` under Slurm |
| `TRAIN_DATA_ROOT`, `EVAL_DATA_ROOT` | Original GPIC train and held-out roots; use absolute paths |
| `DINO_DIR` | Original **training** DINO sidecar, also needed for model construction during eval |
| `OUT_DIR` | Experiment output root; defaults to `BiB_results/bit_bidirectional` or `BiB_results/bit_random_tokens_seed0` under the prefix |
| `NPROC` | Workers per node: 8 locally, 4 under Slurm |
| `TOKEN_SEED` | Random vocabulary seed, default 0; independent of training seed |
| `RANDOM_DATA_ROOT` | Prepared dataset views, default `$OUT_DIR/data` for the random experiment |
| `DRYRUN` | `1` prints every preparation/train/eval command without creating data or launching workers |
| `EVAL_ONLY` | `1` evaluates without training |
| `SPECIALIST_STEPS`, `CKPT_EVERY` | 200000 and 10000; step budget must be a positive multiple of checkpoint interval |

Other options are forwarded to every `train.py` invocation, for example
`--wandb-project bit-controls --seed 0`. Dataset/output paths, direction,
SDE, checkpoint, and step settings are managed by the scripts. Defaults match
the checked-in baseline recipe; changing hyperparameters or evaluation options
also requires matching the historical comparison. Keep output roots separate
when changing training seeds or configurations. Slurm phases get separate W&B
run IDs, so evaluations do not overwrite the training run's configuration.

The held-out recipe matches `_launch_ablation_eval.sh`: 50,000 T2I samples,
10,000 I2T samples, 500 SDE steps, CFG 0 and 0.5, and the same metrics and
padding treatment. This uses the existing GPIC evaluation, not COCO. The
original train/test splits, tokenization, REPA sidecars, and model defaults
remain part of the comparison.

## Unified versus specialist models

`_launch_bit_bidirectional.sh` trains one `DiTXA-L/2` with both directions to
400K steps, then evaluates **both directions** at exactly 200K and 400K.
Evaluations use separate output directories. Missing or ambiguous checkpoints
raise an error; a partial run's latest checkpoint is never substituted.

The shared model and each specialist use the same trunk and token decoder.
Parameter matching is approximate: the shared model activates both conditioning
embedders and both image-REPA heads. Relative to one specialist, the extra
active parameters are one 17,408-parameter conditioning patch embedder plus
one training-only REPA head. For hidden size 1024 that head has
`2 * 1024 * 1025 + 1025 * d_dino` parameters (about 2.9M–3.1M for DINO widths
768–1024), under 0.5% of the roughly 700M-parameter trunk. Stored parameter
counts differ from active counts because some unused modules remain frozen.
Use the existing `params/total`, `params/trainable`, and `params/used` logs.
The shared model saves approximately half the model storage compared with
deploying two specialists of this size.

The existing loss sums two direction-specific forward passes into one backward
call. That call traverses **both graphs**; one `.backward()` does not make the
two directions cost the same as a single direction.

| Run | Optimizer steps | Directional training passes | Interpretation |
|---|---:|---:|---|
| Each specialist | 200K | 200K | Existing per-direction baseline |
| Two specialists combined | 400K total | 400K total | Total baseline training cost |
| Shared model, halfway | 200K | 400K | Approximately matches the combined specialist compute and per-direction exposure |
| Shared model, final | 400K | 800K | Twice the steps of each specialist; approximately twice the combined specialist compute |

Report both shared checkpoints, and use the trainer's existing
`flops/train_step_fwd_bwd_global` and cumulative FLOP logs for the actual
comparison. The token decoder is trained once per shared step, while two
specialists each train their own decoder, so the ratios above are approximate.
Optimizer/communication overhead and wall time need not follow the same ratio.
The shared 200K checkpoint is the appropriate primary compute control; 400K
shows the effect of additional joint training. LR warmup is 5K followed by a
constant LR, so the 200K prefix does not suffer a stretched decay schedule.

## Fixed random token embeddings

`_launch_bit_random_tokens.sh` prepares one random vocabulary, then trains and
evaluates a 200K-step T2I specialist and a 200K-step I2T specialist. Compare
each against the corresponding existing Qwen-based specialist above.

For every vocabulary ID `v`, draw and freeze a Gaussian vector. At the runtime
dimension `d`, the existing loader constructs

$$z_v \sim \mathcal N(0,I_d),\qquad e_v=\sqrt d\,z_v/\|z_v\|_2.$$

Thus each valid token has norm `sqrt(d)` and expected squared coordinate 1,
matching the normalized/scaled Qwen token vectors. With the default SD bridge,
`d=64` and the scale is **8**; the Flux bridge uses `d=128` and `sqrt(128)`.
These are **normalized Gaussian draws** (uniform directions on a sphere),
not unnormalized Gaussian vectors with fluctuating norms. Padding stays zero.

The helper `data_utils/prepare_random_token_data.py` reads the training table's
shape, generates a seeded float32 Gaussian table with NumPy PCG64, and creates
`train/` and `test/` views referencing that exact shared table. The original
latent shards, token IDs, masks, lengths, and captions are symlinked, not copied
or re-encoded. Training requires the current GPIC table-mode dataset; held-out
data may use table or older dense-memmap storage. Tokenizers must agree.
The manifest records sources, seed, shape, RNG, and table checksum. Identical
reruns reuse the preparation; changing its configuration or table requires a
fresh output directory. Keep these prepared views alongside the checkpoints:
the table is frozen input data, not a learned parameter in the checkpoint.
Existing GPIC and COCO encoders consume the views without Python changes.

This tests dependence on pretrained text-embedding geometry while retaining
the same Qwen tokenizer, sequence length, token identity, and trainable decoder.
The random vectors are fixed per **vocabulary item**, not resampled per
occurrence or epoch; resampling would remove the stable text signal. No global
Qwen conditioning or text REPA is enabled. Performance differences can also
reflect the difficulty of decoding random codes; inspect the existing token
decoder accuracy alongside generation metrics. For uncertainty estimates,
repeat with multiple vocabulary seeds in separate output roots.
