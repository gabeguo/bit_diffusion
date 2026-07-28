"""
DINO sidecar for latents written by encode_gpic.py.

Reads each source shard's filled latents directly, decodes them with the same
SD VAE, runs DINOv2, and writes features + a per-shard dino_filled bitmap that
latent_dataset.py consumes (dino_{sid}.memmap / dino_filled_{sid}.memmap).

- Layout: shards are discovered by globbing the source filled_*.bin bitmaps, so
  the sparse, per-rank-reserved GPIC shard ids work unchanged.
- World size is independent of the encode run: ranks split the shard list and
  resume is driven by the dino_filled bitmap (a row is skipped once its bit is
  set), so any world size can resume any prior run with no migration.
- --percent keeps a PREFIX of shards covering the first `percent` of all filled
  rows (global shard order). Kept shards are written densely and the rest get no
  files at all, so there are no subsetting-induced sparse files to inflate on
  copy/archive, and larger percents are supersets of smaller ones.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from tqdm import tqdm

try:
    from encode_common_catalog import ensure_sized_file
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from encode_common_catalog import ensure_sized_file  # noqa: E402

_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def dist_env() -> tuple[int, int, int]:
    """(global_rank, world_size, local_rank) from torchrun or SLURM env."""
    g = lambda *ks, d=0: int(next((os.environ[k] for k in ks if k in os.environ), d))
    return (g("RANK", "SLURM_PROCID"),
            g("WORLD_SIZE", "SLURM_NTASKS", d=1),
            g("LOCAL_RANK", "SLURM_LOCALID"))


@torch.inference_mode()
def decode_and_encode(vae, dino, latents, *, device, img_size, store, mean, std):
    """latents: (B, 4, 32, 32) fp16 array -> DINO features (B, flat_dim) fp16."""
    z = torch.from_numpy(latents).to(device=device, dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        decoded = vae.decode(z).sample
    x = (decoded.float().clamp(-1, 1) + 1) / 2
    x = F.interpolate(x, img_size, mode="bicubic", align_corners=False, antialias=True)
    x = (x - mean) / std
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = dino.forward_features(x)
    v = f["x_norm_clstoken"] if store == "cls" else f["x_norm_patchtokens"]
    return v.float().reshape(v.shape[0], -1).cpu().numpy().astype(np.float16)


def run_encode(args) -> None:
    rank, world, local = dist_env()
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)
    src, out = Path(args.source_root), Path(args.dino_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())["config"]
    S = int(cfg["shard_size"])
    latent_shape = cfg.get("latent_shape")
    if latent_shape is None:
        if rank == 0:
            print("Warning: latent_shape is not set, using default latent shape: (4, 32, 32)")
        latent_shape = (4, 32, 32)
    else:
        latent_shape = tuple(latent_shape)
    vae_model = cfg.get("vae_model")
    if vae_model is None:
        if rank == 0:
            print("Warning: vae_model is not set, using default vae model: stabilityai/sd-vae-ft-mse")
        vae_model = "stabilityai/sd-vae-ft-mse"

    vae_kwargs = {"torch_dtype": torch.bfloat16, "cache_dir": args.cache_dir}
    vae_subfolder = cfg.get("vae_subfolder")   # e.g. Flux VAE lives in "<repo>/vae"
    if vae_subfolder is not None:
        vae_kwargs["subfolder"] = vae_subfolder
    vae = AutoencoderKL.from_pretrained(vae_model, **vae_kwargs).to(device).eval()
    dino = torch.hub.load("facebookresearch/dinov2", args.dino_model).to(device).eval()
    probe = dino.forward_features(
        torch.zeros(1, 3, args.img_size, args.img_size, device=device))
    assert args.store == "patch"
    tok = probe["x_norm_clstoken"] if args.store == "cls" else probe["x_norm_patchtokens"]
    num_tokens, token_dim = (1, tok.shape[-1]) if args.store == "cls" else (tok.shape[1], tok.shape[2])
    flat_dim = num_tokens * token_dim
    mean, std = _MEAN.to(device), _STD.to(device)

    if rank == 0:
        (out / "dino_config.json").write_text(json.dumps({"config": {
            "source_root": str(src.resolve()), "shard_size": S, "vae_model": vae_model,
            "dino_model": args.dino_model, "store": args.store, "img_size": args.img_size,
            "num_tokens": num_tokens, "token_dim": token_dim, "flat_dim": flat_dim,
            "percent": args.percent,
        }}, indent=2))

    shard_ids = sorted(int(p.stem.split("_")[1]) for p in src.glob("filled_*.bin"))
    # Prefix selection: keep whole shards covering the first `percent` of all
    # filled rows (global shard order). Kept shards are written densely; the rest
    # get no files, so there are no subsetting-induced sparse files.
    counts = np.array([int(np.count_nonzero(
        np.fromfile(src / f"filled_{sid:05d}.bin", np.uint8, S))) for sid in shard_ids])
    target = counts.sum() * args.percent / 100.0
    before = np.concatenate([[0], np.cumsum(counts)[:-1]])
    keep = {sid for sid, b in zip(shard_ids, before) if b < target}

    count_by_sid = {sid: int(c) for sid, c in zip(shard_ids, counts)}
    my_shards = [sid for sid in shard_ids[rank::world] if sid in keep]

    def done_count(sid: int) -> int:
        p = out / f"dino_filled_{sid:05d}.memmap"
        return int(np.count_nonzero(np.fromfile(p, np.uint8, S))) if p.exists() else 0

    encoded = 0
    pbar = tqdm(total=sum(count_by_sid[sid] for sid in my_shards),
                initial=sum(done_count(sid) for sid in my_shards),
                disable=(rank != 0), desc="dino", unit="img")
    for sid in my_shards:
        src_filled = np.fromfile(src / f"filled_{sid:05d}.bin", np.uint8, S)
        feat_path, fill_path = out / f"dino_{sid:05d}.memmap", out / f"dino_filled_{sid:05d}.memmap"
        ensure_sized_file(feat_path, S * flat_dim * 2)
        ensure_sized_file(fill_path, S)
        lat = np.memmap(src / f"latents_{sid:05d}.memmap", np.float16, "r", shape=(S, *latent_shape))
        feats = np.memmap(feat_path, np.float16, "r+", shape=(S, flat_dim))
        done = np.memmap(fill_path, np.uint8, "r+", shape=(S,))

        todo = np.flatnonzero((src_filled != 0) & (done == 0))
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            feats[chunk] = decode_and_encode(
                vae, dino, np.asarray(lat[chunk]), device=device,
                img_size=args.img_size, store=args.store, mean=mean, std=std)
            feats.flush()              # data durable before the row is marked done
            done[chunk] = 1
            done.flush()
            encoded += len(chunk)
            pbar.update(len(chunk))
        del lat, feats, done
        pbar.set_postfix_str(f"shard {sid} done")
    pbar.close()

    print(f"rank {rank}: done, encoded {encoded} rows.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", required=True, help="encode_gpic.py output dir (one preset).")
    p.add_argument("--dino-dir", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dino-model", default="dinov2_vitl14")
    p.add_argument("--store", default="patch", choices=["patch", "cls"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--percent", type=float, default=100.0,
                   help="Keep a prefix of shards covering the first %% of filled rows.")
    return p.parse_args()


if __name__ == "__main__":
    run_encode(parse_args())
