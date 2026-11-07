# Additions relative to the paper

Base: `main` at `cbc85553333755564aeb06b98b800686bcace375`.
Reference draft: [arXiv:2608.27885](https://arxiv.org/abs/2608.27885).

- Add a stochastic conditional flow baseline with independently configurable
  Gaussian text/image endpoint perturbations (defaults zero). Train with the
  perturbed endpoints in both the linear interpolant and velocity target; keep
  clean source conditioning. Sampling draws source noise once before ODE
  integration. The original unconditional zero-noise flow remains available.
  This extends the draft's deterministic flow baseline using the conditional
  noisy-base idea of [Albergo et al.](https://arxiv.org/abs/2310.03725).
- Add separate sigma=0.01 text-noise T2I and image-noise I2T training/evaluation
  recipes. Match the checked-in original launch recipe's hyperparameters,
  including epsilon 9.9e-4; this is a code-recipe comparison, not a claim that
  every setting equals the draft. New standalone evaluation defaults to 500
  steps for matched inference cost; rerun any historical 100-step flow result.
- Add COCO 2014 validation-caption T2I and Karpathy-test I2T evaluation with
  official image/caption downloads. COCO FID references original images rather
  than the VAE reconstructions used by existing GPIC evaluation. Protocols and
  compatibility limits are documented in `text_to_image/COCO_EVALUATION.md`.
- Retain legacy CIDEr and add opt-in PTB-tokenized COCO CIDEr and standard SPICE.
  The standalone COCO scorer uses all reference captions; GPIC remains a
  single-reference evaluation. No new experimental scores are asserted by
  these code additions.
- Add controlled BIT follow-up experiments on top of
  `09-21-2026/claude/multi-node-flow-launchers` at `8ce528f`: a shared
  bidirectional model trained to 400K with 200K/400K evaluations, and two
  200K specialists using a fixed normalized Gaussian token vocabulary.
  Both reuse the existing ablation architecture, training, and GPIC evaluation;
  only random-vocabulary data preparation needs new Python. The 200K shared
  checkpoint approximately matches the **combined** compute of two 200K
  specialists, whereas 400K costs approximately twice as much. Protocol,
  parameter-count qualifications, scaling, and existing specialist checkpoint
  references are in `text_to_image/BIT_ABLATIONS.md`. These are additional
  experimental controls, not reported results or changes to the BIT objective.
- ICLR rebuttal additions, built on
  `09-21-2026/claude/multi-node-flow-launchers` at `fd9a74b` (protocols in
  `text_to_image/REBUTTAL_EXPERIMENTS.md`, LaTeX in
  `rebuttal_experimental_description/`). No reported number changes:
  - **New Table 2 rows (§6.3):** FlowTok-XL and CrossFlow (256 px, CLIP, DiMR),
    from their released checkpoints and samplers (pinned commits). Released
    settings: 20 Euler steps with guidance 2.0, and 50 steps with 7.0. Scoring
    uses BIT's held-out GPIC protocol: the same 50K captions, SD-VAE reference
    images, FID-2048, and CLIP ViT-B/16. FlowTok's 512 px outputs are resized
    to 256 px. Their guidance `s` equals BIT `omega = s - 1`. I2T is n/a
    because no I2T checkpoints are released.
  - **Guided variants of Fig. 3 and Fig. 5 (ω = 0.5):** the paper uses ω = 0.
    In the round trip, guidance only affects the endpoint-conditioned
    corruption leg; restoration is state-only, so ω is a no-op there. For
    editing, `perturbation_slerp_experiment.py` previously ignored
    `--cfg-scale` when `--infer-noise` was on. It now applies the Eq. 10 drift
    in both backsolving (Eq. 61) and replay, which keeps the λ = 0 replay exact.
    At ω = 0 the calls and RNG stream are unchanged. New `--infer-cfg-scale` defaults to
    `--cfg-scale`, which preserves the old image→text behavior.
  - `editing_experiment.py`, `n2d_two` mode only (not used in the paper): with
    `--proportional-nfe`, the cross-modal inference leg now uses the full
    `--infer-nfe` instead of `--infer-nfe` × corruption fraction. That leg
    always traverses the whole bridge.
  - Bookkeeping only, no effect on results: `train.py --eval-only` also writes
    `eval/eval_metrics.json`. `_launch_bit_bidirectional.sh` skips finished
    phases, and `_sbatch_flow.sh` can refuse a partial checkpoint
    (`REQUIRE_STEP`).
- The shared bidirectional control (`_launch_bit_bidirectional.sh`) now trains
  to 200K by default. That is the point matching each specialist's
  per-direction examples and the pair's combined compute. It evaluates there at
  CFG 0 only and runs on four nodes (global batch unchanged at 512).
  `CONTINUE=1` restores the 400K run with evaluations at 200K and 400K. Runs
  started by the earlier version resume unchanged.
