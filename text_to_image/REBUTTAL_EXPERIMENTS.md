# ICLR 2027 rebuttal experiments

One command queues every rebuttal experiment in Slurm. It checks each one for
existing results first, skips finished work, and resumes partial work:

```bash
cd text_to_image
bash queue_rebuttal_experiments.sh --list      # status of every experiment, submits nothing
bash queue_rebuttal_experiments.sh --dry-run   # print every job script, submits nothing
bash queue_rebuttal_experiments.sh             # submit everything not done yet
```

| Key | Reviewer | What runs | Result checked before submitting |
|---|---|---|---|
| `flowtok` | 6E8U | FlowTok-XL T2I under the Table 2 GPIC protocol (20 Euler steps, guidance 2.0) | `$RESULTS_ROOT/external/flowtok_xl/results.json` |
| `crossflow` | 6E8U | CrossFlow 256 px CLIP DiMR T2I, same protocol (50 Euler steps, guidance 7.0) | `$RESULTS_ROOT/external/crossflow_256_clip/results.json` |
| `roundtrip_cfg` | 4qvB Q2 | Fig. 3 round-trip variation with BIT at ω = `ROUNDTRIP_CFG_SCALES` (default 0.5), image and text, plus overlay plots | `$RESULTS_ROOT/roundtrip/cfg<ω>/d2d_{image,text}/results.json` |
| `editing_cfg` | 4qvB Q2 | Fig. 5 image editing at ω = `EDIT_CFG_SCALES` (default 0.5), plus a four-row Figure-5-style PDF/PNG | `$RESULTS_ROOT/editing/cfg<ω>/manifest.json` (complete) and `figure5_cfg<ω>.pdf` |
| `bidir_400k` | xYk4, 6E8U | Existing `_launch_bit_bidirectional.sh`: shared model trained to 400K, evaluated at 200K and 400K | `$BIDIR_OUT/eval_{200000,400000}/*/eval/eval_metrics.json`; 400K checkpoint means evaluation only |
| `noisy_flow` | 6E8U | Existing `_sbatch_flow.sh` stochastic-endpoint flow: T2I `text_noise`, I2T `image_noise` (σ = 0.01) | `<flow root>/*/eval/eval_metrics.json`; 200K checkpoint means evaluation only |
| `roundtrip_n2d_cond` (opt-in) | 4qvB W1 | Round trip for the noise-to-data baseline restoring **with** inferred side information (`n2d_two`), at ω = 0 and each `ROUNDTRIP_CFG_SCALES` value | `$RESULTS_ROOT/roundtrip/cfg<ω>/n2d_cond_{image,text}/results.json` |

Selecting experiments:

```bash
bash queue_rebuttal_experiments.sh --exclude bidir_400k,noisy_flow
bash queue_rebuttal_experiments.sh --only flowtok,crossflow
bash queue_rebuttal_experiments.sh --include roundtrip_n2d_cond   # add opt-in keys
bash queue_rebuttal_experiments.sh --only editing_cfg --force      # recompute even if results exist
```

`--force` reruns evaluations whose results already exist. It never deletes
checkpoints and never retrains a finished model. Rerunning the command is
always safe. After a wall-time limit or a failure, run it again to resume.

## Slurm settings

All settings are environment variables; defaults follow the existing launchers.

| Variable | Default |
|---|---|
| `RESERVATION` | `iclr_rebuttal` (`RESERVATION=` submits without one) |
| `ACCOUNT`, `QOS`, `CONSTRAINT` | `m5319`, `regular`, `gpu&hbm80g` |
| `GPU_SPEC`, `NPROC`, `CPUS_PER_TASK` | `a100:4`, `4`, `64` |
| `PREFIX_DIR` | `/pscratch/sd/g/gabeguo` |
| `RESULTS_ROOT` | `$PREFIX_DIR/BiB_results/iclr_rebuttal` (stable, not dated, so reruns find old results) |
| `LOG_DIR`, `SCRIPT_DIR` | `$RESULTS_ROOT/slurm_logs`, `$RESULTS_ROOT/slurm_scripts` (every generated job script is kept) |
| `CONDA_ENV`, `MODULES` | `dit_env`, `python nccl/2.29.2-cu13 conda` |
| `EXTRA_SBATCH` | extra `sbatch` flags for every job |
| `EVAL_DATA_ROOT` | `$PREFIX_DIR/datasets/text_to_image/gpic_latents_TEST/TEST` (Tables 1–2) |
| `EDIT_DATA_ROOT` | `$PREFIX_DIR/datasets/text_to_image/gpic_latents_sd_test/TEST` (Figs. 3 and 5) |
| `*_TIME` | `EXT_TIME=08:00:00`, `RT_TIME=06:00:00`, `EDIT_TIME=24:00:00`, `BIDIR_TIME=48:00:00`, `FLOW_TRAIN_TIME=48:00:00`, `FLOW_EVAL_TIME=08:00:00` |
| `BIDIR_CHAIN`, `FLOW_CHAIN` | `3`, `2` chained `afterany` training jobs. Each one resumes; extras exit quickly once training is done |

If the reservation has an end time, every job's time limit must fit inside
it, or Slurm will not start the job.

## External baselines (`external_baselines.py`)

The first time `flowtok` or `crossflow` is selected, the queue command runs
`_setup_external_baselines.sh` on the submitting node. That script clones the
official repositories at pinned commits (FlowTok `d4d43c5`, CrossFlow
`8493373`), downloads the released checkpoints from HuggingFace
(`turkeyju/FlowTok`, `QHL067/CrossFlow`), and caches the text encoders. It
also verifies the extra packages in `requirements-external-baselines.txt`. Use
`INSTALL_DEPS=1 bash _setup_external_baselines.sh` to install missing packages.

Protocol (identical to `train.py --eval-only` for BIT):

- **Prompts and references:** the first 50,000 items of the held-out split, built with `--val-fraction 1.0 --seed 0` (`EXT_SPLIT_SEED`).
- **FID:** torchmetrics, 2048-d Inception features, float64. The references are the SD-VAE decodings of the stored latents, the same 256×256 images used for BIT.
- **CLIP T2I:** ViT-B/16 score against the full caption, computed with the code in `eval_metrics.py`.
- **Resolution:** FlowTok-XL generates at 512 px and is resized to 256 px (bicubic with antialiasing). CrossFlow 256 px is native.
- **Sampling:** each model keeps its own released sampler: text encoder, text variational encoder, Euler ODE, decoder, and guidance.
- **Guidance convention:** both libraries apply `uncond + s (cond - uncond)`, so s = 2.0 and 7.0 correspond to BIT ω = 1 and ω = 6.
- **Image-to-text:** neither release includes an I2T checkpoint, so those Table 2 cells stay n/a.

The step counts and guidance values come from the released code
(`eval_t2i_mjhq.py`, `configs/t2i_256px_clip_dimr.py`). Override them with
`FLOWTOK_STEPS`, `FLOWTOK_CFG`, `CROSSFLOW_STEPS`, and `CROSSFLOW_CFG`.

## Guidance (ω) in round trips and editing

- **Round trip (`editing_experiment.py`):** in BIT mode, `--cfg-scale` only
  affects the endpoint-conditioned corruption leg. The restoration leg is
  state-only (x_cond = ∅), and Eq. 10 reduces to the unguided drift there. The
  paper's noise-to-data baseline restores unconditionally, so ω has no effect
  on it. Set `ROUNDTRIP_BASELINE_IMAGE_JSONS` / `ROUNDTRIP_BASELINE_TEXT_JSONS`
  to the existing Fig. 3 `results.json` files to overlay them, or set
  `ROUNDTRIP_CFG_SCALES="0.0 0.5"` to recompute the ω = 0 reference. Legends
  show ω when it is nonzero.
- **Editing (`perturbation_slerp_experiment.py`):** `--cfg-scale` is applied to
  the image→text inference (override with `--infer-cfg-scale`). It is also
  applied, identically, to noise backsolving (Eq. 61) and replay (Alg. 2). Using
  the same guided drift in both keeps the λ = 0 column an exact reconstruction;
  `meta.json` records the relative error. Each run also writes
  `rows/img####/{col*.png,source.png,meta.json}` and `manifest.json`.
  `editing_figure.py` turns these into the four-row figure:

```bash
python editing_figure.py --run-dir RUN --contact-sheet contact.png   # choose rows
python editing_figure.py --run-dir RUN --images 3 17 22 40 --out fig5_cfg0.5.pdf
FIG5_IMAGES="3 17 22 40" bash queue_rebuttal_experiments.sh --only editing_cfg
```

`--show-source` adds the VAE-decoded source image as a leftmost column. The
paper's Figure 5 does not have this column.

## Notes

- `train.py --eval-only` now also writes `<out>/<timestamp>/eval/eval_metrics.json`.
  Evaluations that ran before this change exist only in W&B and are **not**
  detected. If they are already done, exclude that experiment.
- The editing default checkpoint is the one in `_launch_perturbation_experiment.sh`
  (`large_scale/.../step_0100000.pt`). Set `EDIT_CKPT` if Figure 5 used another one.
- The `noisy_flow` evaluation job refuses a partial checkpoint (`REQUIRE_STEP`).
  Training at the wall time is resumed by the next chained job or the next queue run.
- Shared-model compute: each shared step runs both directions on the same
  batch. 400K shared steps match the two specialists' combined optimizer
  steps and sampled pairs, but cost about twice their combined FLOPs. The 200K
  checkpoint is the FLOP- and per-direction-matched point (see `BIT_ABLATIONS.md`).
- The LaTeX descriptions are in `../rebuttal_experimental_description/`.
