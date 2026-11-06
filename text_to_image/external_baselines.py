"""
Evaluate public text-to-image checkpoints of FlowTok-XL and CrossFlow under the
GPIC held-out protocol used for BIT's Table 2 (``train.py --eval-only``).

Protocol (shared with ``compute_fid_distributed``):
  * Prompts/references: the first ``--num-samples`` items of the held-out split,
    built exactly like train.py with ``--val-fraction 1.0 --seed <split seed>``.
  * FID: torchmetrics, 2048-d Inception features, float64, uint8 inputs. The
    reference images are the stabilityai/sd-vae-ft-mse decodings of the stored
    latents, i.e. the same 256x256 references used for BIT.
  * CLIP T2I: CLIP ViT-B/16, 100 * cosine between each generated image and its
    full GPIC caption (``eval_metrics._clip_score_manual``).
  * Generated images are resized to the reference resolution (256x256, bicubic
    with antialiasing) only when the model's native output differs.

Each external model keeps its own released inference recipe: text encoder,
text variational encoder, Euler ODE solver, decoder, and guidance. Defaults are
the released settings: FlowTok-XL 20 Euler steps and guidance 2.0
(``eval_t2i_mjhq.py``); CrossFlow 256px CLIP DiMR 50 Euler steps and guidance
7.0 (``configs/t2i_256px_clip_dimr.py``). Both libraries use
``uncond + s * (cond - uncond)``; BIT's Eq. 10 weight is ``omega = s - 1``.
Neither release includes an image-to-text checkpoint, so there is no I2T run.

Launch with torchrun from text_to_image (PYTHONPATH=.:..). Every rank generates
a disjoint round-robin shard; FID/CLIP statistics are all-reduced. Writes
``<out>/results.json`` (the completion marker) and a few sample images.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

MODEL_DEFAULTS = {
    "flowtok": {"config": "configs/FlowTok-XL-Stage3.py", "steps": 20, "cfg_scale": 2.0,
                "label": "FlowTok-XL"},
    "crossflow": {"config": "configs/t2i_256px_clip_dimr.py", "steps": 50, "cfg_scale": 7.0,
                  "label": "CrossFlow (256px, CLIP, DiMR)"},
}


# ---------------------------------------------------------------------------
# Protocol helpers (no external code)
# ---------------------------------------------------------------------------

def gpic_eval_split(data_root: str, split_seed: int, num_samples: int):
    """Held-out dataset view + eval indices exactly as in train.py's eval-only path."""
    from transformers import AutoTokenizer
    from data_utils.latent_dataset import CommonCatalogLatentDataset, random_split_indices
    from token_bridge import bridge_config_from_manifest

    runtime = bridge_config_from_manifest(data_root, preset="auto")
    token_cfg = json.loads((Path(data_root) / "token_embed_config.json").read_text())["config"]
    tokenizer = AutoTokenizer.from_pretrained(token_cfg["text_model"])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ds_kwargs = {
        "token_pad_id": int(pad_id), "token_scale": runtime.token_scale, "config": runtime.bridge,
        "latent_scale": runtime.latent_scale, "latent_shift": runtime.latent_shift,
    }
    n_total = len(CommonCatalogLatentDataset(data_root, cast_dtype=torch.float32, **ds_kwargs))
    _, val_idx = random_split_indices(n_total, val_fraction=1.0, seed=split_seed)
    val_ds = CommonCatalogLatentDataset(data_root, cast_dtype=torch.float32, indices=val_idx,
                                        return_caption=True, **ds_kwargs)
    return val_ds, list(range(min(num_samples, len(val_ds)))), runtime


def to_reference_uint8(images: torch.Tensor, size: int) -> torch.Tensor:
    """[0, 1] float RGB (B, 3, H, W) -> uint8 at ``size`` x ``size``."""
    images = images.float().clamp(0, 1)
    if images.shape[-2:] != (size, size):
        images = F.interpolate(images, size=(size, size), mode="bicubic",
                               align_corners=False, antialias=True).clamp(0, 1)
    return (images * 255).round().to(torch.uint8)


def _load_py_config(path: Path):
    spec = importlib.util.spec_from_file_location(f"_ext_config_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# External generators. Each mirrors the released sampling function verbatim
# apart from batching; ``generate`` returns float RGB in [0, 1].
# ---------------------------------------------------------------------------

class FlowTokGenerator:
    """FlowTok-XL + FlowTiTok-512 decoder (``eval_t2i_mjhq.py::ode_fm_solver_sample``)."""

    def __init__(self, repo: Path, config_path: Path, checkpoint: Path, tokenizer_checkpoint: Path,
                 device, steps: int, cfg_scale: float):
        import open_clip
        import utils as ext_utils
        from diffusion.flow_matching import ODEEulerFlowMatchingSolver
        from libs.flowtitok import FlowTiTok

        self.config = config = _load_py_config(config_path)
        config.tokenizer_checkpoint = str(tokenizer_checkpoint)
        self.device, self.steps, self.cfg_scale = device, steps, cfg_scale
        self.Solver = ODEEulerFlowMatchingSolver
        self.nnet = ext_utils.get_nnet(**config.nnet)
        self.nnet.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.nnet.to(device).eval().requires_grad_(False)
        self.autoencoder = FlowTiTok(config)
        self.autoencoder.load_state_dict(torch.load(tokenizer_checkpoint, map_location="cpu"))
        self.autoencoder.to(device).eval().requires_grad_(False)
        clip_encoder, _, _ = open_clip.create_model_and_transforms("ViT-L-14-336", pretrained="openai")
        del clip_encoder.visual
        self.clip = clip_encoder.to(device).eval().requires_grad_(False)
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-L-14-336")

    def _encode_text(self, captions):
        clip = self.clip
        tokens = self.clip_tokenizer(captions).to(self.device)
        cast_dtype = clip.transformer.get_cast_dtype()
        x = clip.token_embedding(tokens).to(cast_dtype) + clip.positional_embedding.to(cast_dtype)
        # FlowTok sets batch_first=False and feeds LND; feeding the layout the
        # installed open_clip expects is the same computation on any version.
        if getattr(clip.transformer, "batch_first", False):
            x = clip.transformer(x, attn_mask=clip.attn_mask)
        else:
            x = clip.transformer(x.permute(1, 0, 2), attn_mask=clip.attn_mask).permute(1, 0, 2)
        return clip.ln_final(x)

    @torch.no_grad()
    def generate(self, captions):
        text_tokens = self._encode_text(captions)
        z_x0, _, _ = self.nnet(text_tokens, text_encoder=True)
        if self.config.nnet.model_args.noising_type != "none":
            z_x0 = z_x0 + torch.randn_like(z_x0) * self.config.sample.noise_scale
        solver = self.Solver(self.nnet, step_size_type="step_in_dsigma", guidance_scale=self.cfg_scale)
        z, _ = solver.sample(x_T=z_x0, batch_size=len(captions), sample_steps=self.steps,
                             unconditional_guidance_scale=self.cfg_scale, has_null_indicator=True)
        z = z.permute(0, 2, 1).unsqueeze(2)
        return self.autoencoder.decode_tokens(z / self.config.vq_model.scale_factor,
                                              text_guidance=text_tokens).float()


class CrossFlowGenerator:
    """CrossFlow DiMR (``demo_t2i.py::ode_fm_solver_sample`` with the SD autoencoder)."""

    def __init__(self, repo: Path, config_path: Path, checkpoint: Path, autoencoder_checkpoint: Path,
                 device, steps: int, cfg_scale: float):
        import libs.autoencoder
        import utils as ext_utils
        from diffusion.flow_matching import ODEEulerFlowMatchingSolver
        from libs.clip import FrozenCLIPEmbedder

        self.config = config = _load_py_config(config_path)
        if config.nnet.model_args.clip_dim != 768:
            raise ValueError("Only the CLIP-conditioned CrossFlow checkpoints are wired up.")
        if cfg_scale <= 1:
            raise ValueError("CrossFlow's sampler asserts a guidance scale > 1.")
        self.device, self.steps, self.cfg_scale = device, steps, cfg_scale
        self.Solver = ODEEulerFlowMatchingSolver
        self.nnet = ext_utils.get_nnet(**config.nnet)
        self.nnet.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.nnet.to(device).eval().requires_grad_(False)
        self.clip = FrozenCLIPEmbedder(device=str(device)).to(device).eval()
        self.autoencoder = libs.autoencoder.get_model(pretrained_path=str(autoencoder_checkpoint),
                                                      scale_factor=config.autoencoder.scale_factor)
        self.autoencoder.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def generate(self, captions):
        _, others = self.clip.encode(captions)
        context, token_mask = others["token_embedding"].detach(), others["token_mask"].detach()
        shape = (len(captions), *self.config.z_shape)
        z_x0, _, _ = self.nnet(context, text_encoder=True, shape=shape, mask=token_mask)
        solver = self.Solver(self.nnet, bdv_model_fn=None, step_size_type="step_in_dsigma",
                             guidance_scale=self.cfg_scale)
        z, _ = solver.sample(x_T=z_x0.reshape(shape), batch_size=len(captions), sample_steps=self.steps,
                             unconditional_guidance_scale=self.cfg_scale,
                             has_null_indicator=hasattr(self.config.nnet.model_args, "cfg_indicator"))
        with torch.autocast(device_type="cuda", dtype=torch.float16):  # demo_t2i.py decode()
            images = self.autoencoder.decode(z)
        return (0.5 * (images.float() + 1.0)).clamp(0.0, 1.0)


def build_generator(args, device):
    repo = args.repo.resolve()
    # Their top-level packages (utils, libs, diffusion, ...) shadow nothing we
    # import, and all of OUR modules are already imported before this point.
    sys.path.insert(0, str(repo))
    config_path = repo / (args.config or MODEL_DEFAULTS[args.model]["config"])
    if args.model == "flowtok":
        return FlowTokGenerator(repo, config_path, args.checkpoint, args.tokenizer_checkpoint,
                                device, args.steps, args.cfg_scale)
    return CrossFlowGenerator(repo, config_path, args.checkpoint, args.autoencoder_checkpoint,
                              device, args.steps, args.cfg_scale)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args):
    from diffusers.models import AutoencoderKL
    from eval_metrics import _clip_score_manual, _get_or_make_clip, _get_or_make_fid, _make_shard_loader
    from eval_plot import decode_latents
    from PIL import Image
    from tqdm import tqdm

    out = Path(args.out)
    results_path = out / "results.json"
    ddp = "RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world, device = 0, 1, torch.device(args.device)
    if results_path.exists() and not args.overwrite:
        if rank == 0:
            print(f"[external] {results_path} exists; skipping (pass --overwrite to recompute)")
        if ddp:
            dist.destroy_process_group()
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    val_ds, indices, runtime = gpic_eval_split(args.data_root, args.split_seed, args.num_samples)
    if len(indices) < args.num_samples and rank == 0:
        print(f"[external] only {len(indices)} held-out items (requested {args.num_samples})")
    loader = _make_shard_loader(val_ds, indices, rank, world, args.batch_size, None, drop_last=False)
    vae_kwargs = {"subfolder": runtime.vae_subfolder} if runtime.vae_subfolder else {}
    vae = AutoencoderKL.from_pretrained(runtime.vae_model, **vae_kwargs).to(device, torch.bfloat16).eval()
    fid = _get_or_make_fid(2048, device)
    clip_model, clip_processor = _get_or_make_clip(device)
    size = int(runtime.bridge.height * 8)  # SD VAE: 32x32 latents -> 256x256 references

    generator = build_generator(args, device)  # external imports happen only here
    torch.manual_seed(args.sample_seed * 1_000_003 + rank)
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.precision == "bf16" else torch.autocast(device_type="cuda", enabled=False))

    clip_sum = clip_sq = 0.0
    n_local = 0
    saved = 0
    sample_dir = out / "samples"
    t0 = time.time()
    for batch in tqdm(loader, desc=f"{args.model} T2I", disable=rank != 0):
        captions = list(batch["caption"])
        with torch.no_grad():
            with autocast:
                fake = to_reference_uint8(generator.generate(captions), size)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                real = decode_latents(vae, batch["latent"].to(device).float(), to_cpu=False,
                                      latent_scale=runtime.latent_scale, latent_shift=runtime.latent_shift)
            if real.shape[-2:] != fake.shape[-2:]:
                raise RuntimeError(f"reference {tuple(real.shape)} vs generated {tuple(fake.shape)}")
            fid.update(real, real=True)
            fid.update(fake, real=False)
            scores = _clip_score_manual(clip_model, clip_processor, fake, captions, device)
        clip_sum += float(scores.sum())
        clip_sq += float(scores.square().sum())
        n_local += len(captions)
        if rank == 0 and saved < args.save_images:
            sample_dir.mkdir(parents=True, exist_ok=True)
            for k in range(min(len(captions), args.save_images - saved)):
                Image.fromarray(fake[k].permute(1, 2, 0).cpu().numpy()).save(sample_dir / f"{saved:04d}.png")
                (sample_dir / f"{saved:04d}.txt").write_text(captions[k] + "\n")
                saved += 1

    fid_value = float(fid.compute())  # all-reduces the Inception statistics
    stats = torch.tensor([n_local, clip_sum, clip_sq], device=device, dtype=torch.float64)
    if ddp:
        dist.all_reduce(stats)
    n, s, sq = (float(v) for v in stats)
    if rank == 0:
        clip_mean = s / n
        info = MODEL_DEFAULTS[args.model]
        result = {
            "model": args.model, "label": info["label"],
            "protocol": "GPIC held-out split (train.py --eval-only --val-fraction 1.0), first N items",
            "fid": fid_value, "clipscore_t2i": clip_mean,
            "clipscore_t2i_std": math.sqrt(max(sq / n - clip_mean ** 2, 0.0)),
            "num_samples": int(n), "steps": args.steps, "cfg_scale": args.cfg_scale,
            "cfg_convention": "uncond + s * (cond - uncond); BIT omega = s - 1",
            "bit_equivalent_omega": args.cfg_scale - 1.0, "reference_resolution": size,
            "precision": args.precision, "split_seed": args.split_seed,
            "sample_seed": args.sample_seed, "world_size": world, "data_root": str(args.data_root),
            "repo": str(args.repo), "repo_commit": _git_commit(args.repo),
            "config": args.config or MODEL_DEFAULTS[args.model]["config"],
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256(args.checkpoint) if args.hash_checkpoints else None,
            "i2t": "not evaluated: no public image-to-text checkpoint",
            "elapsed_sec": time.time() - t0,
        }
        out.mkdir(parents=True, exist_ok=True)
        tmp = results_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(results_path)
        print(f"[external] {info['label']}: FID={fid_value:.3f} CLIP={clip_mean:.3f} "
              f"(n={int(n)}) -> {results_path}")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(MODEL_DEFAULTS), required=True)
    p.add_argument("--repo", type=Path, required=True, help="Clone of the model's official repository.")
    p.add_argument("--config", default=None, help="Config path relative to --repo (default per model).")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer-checkpoint", type=Path, help="FlowTok: FlowTiTok_512.bin.")
    p.add_argument("--autoencoder-checkpoint", type=Path,
                   help="CrossFlow: assets/stable-diffusion/autoencoder_kl.pth from assets.tar.")
    p.add_argument("--steps", type=int, default=None, help="ODE steps (default: released setting).")
    p.add_argument("--cfg-scale", type=float, default=None, help="Guidance s (default: released setting).")
    p.add_argument("--data-root", required=True, help="GPIC held-out latents (gpic_latents_TEST/TEST).")
    p.add_argument("--split-seed", type=int, default=0, help="train.py --seed used for the held-out order.")
    p.add_argument("--num-samples", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--save-images", type=int, default=64, help="Generated samples kept by rank 0.")
    p.add_argument("--hash-checkpoints", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    defaults = MODEL_DEFAULTS[args.model]
    args.steps = defaults["steps"] if args.steps is None else args.steps
    args.cfg_scale = defaults["cfg_scale"] if args.cfg_scale is None else args.cfg_scale
    if args.model == "flowtok" and args.tokenizer_checkpoint is None:
        p.error("--tokenizer-checkpoint is required for FlowTok")
    if args.model == "crossflow" and args.autoencoder_checkpoint is None:
        p.error("--autoencoder-checkpoint is required for CrossFlow")
    return args


if __name__ == "__main__":
    run(parse_args())
