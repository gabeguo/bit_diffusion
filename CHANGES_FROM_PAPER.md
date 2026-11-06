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
