"""Reproducible COCO generation and offline scoring. See COCO_EVALUATION.md.

Run from text_to_image with PYTHONPATH=.:.. . `generate` accepts torchrun's
rank environment and writes disjoint shards, without distributed collectives.
Run `score` once after every generation process has finished successfully.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random

from data_utils.download_coco import karpathy_ids


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_versions():
    versions = {}
    for package in ("pycocoevalcap", "clean-fid", "torch", "transformers", "diffusers",
                    "numpy", "Pillow", "opencv-python-headless"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def load_annotations(root):
    path = Path(root) / "annotations/captions_val2014.json"
    annotations = json.loads(path.read_text())
    images = {row["id"]: row["file_name"] for row in annotations["images"]}
    references = {key: [] for key in images}
    for row in annotations["annotations"]:
        references[row["image_id"]].append(row["caption"])
    if any(len(captions) < 5 for captions in references.values()):
        raise ValueError("Expected at least five official COCO references per image.")
    return images, references, annotations["annotations"]


def select_samples(root, task, count, seed):
    """Stable caption-ID sampling for T2I; canonical test image IDs for I2T."""
    images, _, captions = load_annotations(root)
    if task == "t2i":
        count = 30000 if count is None else count
        if not 1 <= count <= len(captions):
            raise ValueError("Invalid number of T2I captions.")
        chosen = random.Random(seed).sample(sorted(captions, key=lambda x: x["id"]), count)
        return [{"index": i, "image_id": row["image_id"], "caption_id": row["id"],
                 "caption": row["caption"]} for i, row in enumerate(chosen)]
    ids = karpathy_ids(Path(root) / "coco_karpathy_test.json")
    count = len(ids) if count is None else count
    if not 1 <= count <= len(ids):
        raise ValueError("I2T uses at most the 5,000 Karpathy test images.")
    return [{"index": i, "image_id": key, "file_name": images[key]}
            for i, key in enumerate(ids[:count])]


def encode_prompts(captions, tokenizer, table, runtime, device):
    """Match table-mode CommonCatalogLatentDataset normalization and padding."""
    import numpy as np
    import torch
    from token_bridge import token_flat_to_bridge
    bc = runtime.bridge
    tokens = tokenizer(captions, add_special_tokens=False, padding="max_length",
                       truncation=True, max_length=bc.token_seq_len, return_tensors="np")
    ids, mask = tokens["input_ids"], tokens["attention_mask"]
    embeddings = np.array(table[ids, :bc.token_emb_dim], dtype=np.float32, copy=True)
    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-12
    embeddings *= mask[..., None] * runtime.token_scale
    flat = torch.from_numpy(embeddings.reshape(len(captions), -1)).to(device)
    return token_flat_to_bridge(flat, layout="row_major", config=bc)


def load_image(path, size):
    """Training-compatible OpenCV area resize / center crop, including small images."""
    import cv2
    import numpy as np
    import torch
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode {path}")
    h, w = img.shape[:2]
    new_h, new_w = (size, round(w * size / h)) if h < w else (round(h * size / w), size)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top, left = (new_h - size) // 2, (new_w - size) // 2
    img = cv2.cvtColor(img[top:top + size, left:left + size], cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)


def generate(args):
    import numpy as np
    import torch
    from diffusers import AutoencoderKL
    from PIL import Image
    from transformers import AutoTokenizer
    from checkpoint_utils import _resolve_ckpt, load_model
    from eval_plot import decode_latents
    from token_bridge import bridge_config_from_manifest, bridge_to_token_flat, norm_based_token_stops

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(args.seed + rank)
    checkpoint = _resolve_ckpt(args.checkpoint)
    loaded = load_model(checkpoint, str(args.data_root), device)
    if not loaded.generates("image" if args.task == "t2i" else "text"):
        raise ValueError(f"Checkpoint was not trained for {args.task} generation.")
    if args.cfg_scale and loaded.args.get("force_unconditional"):
        raise ValueError("The original unconditional flow requires --cfg-scale 0.")
    if args.task == "i2t" and loaded.token_decoder is None:
        raise ValueError("I2T requires a trained token decoder in the checkpoint.")
    runtime = bridge_config_from_manifest(args.data_root, preset=loaded.args.get("bridge_preset", "auto"))
    bc = runtime.bridge
    tcfg_path = args.data_root / "token_embed_config.json"
    tcfg = json.loads(tcfg_path.read_text())["config"]
    tokenizer = loaded.tokenizer or AutoTokenizer.from_pretrained(tcfg["text_model"])
    # GPIC's encoder manually pads on the right, independent of tokenizer defaults.
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    table = None
    if args.task == "t2i":
        if tcfg.get("token_emb_storage") != "table":
            raise ValueError("COCO prompt encoding requires the training token vocabulary table (table storage).")
        table = np.load(args.data_root / tcfg.get("vocab_table", "token_vocab_table.npy"), mmap_mode="r")
        if table.shape[0] != len(tokenizer) or table.shape[1] < bc.token_emb_dim:
            raise ValueError("Training vocabulary table does not match tokenizer/bridge dimensions.")
    vae_kwargs = {}
    if runtime.vae_subfolder and not loaded.args.get("vae_ckpt"):
        vae_kwargs["subfolder"] = runtime.vae_subfolder
    vae = AutoencoderKL.from_pretrained(loaded.args.get("vae_ckpt") or runtime.vae_model,
                                      torch_dtype=dtype, **vae_kwargs).to(device).eval()
    resolution = bc.height * 2 ** (len(vae.config.block_out_channels) - 1)
    samples = select_samples(args.coco_root, args.task, args.num_samples, args.seed)
    # All ranks compute this same fingerprint. Shard receipts prevent scoring
    # missing ranks or mixing outputs from different runs after a failed launch.
    protocol = {
        "task": args.task, "samples": samples, "seed": args.seed,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_bytes": Path(checkpoint).stat().st_size,
        "checkpoint_mtime_ns": Path(checkpoint).stat().st_mtime_ns,
        "checkpoint_args": loaded.args, "steps": args.steps,
        "cfg_scale": args.cfg_scale, "ode": args.ode or loaded.args["sde"] == "flow_matching",
        "network_evaluations": args.steps * (2 if args.cfg_scale > 0 else 1),
        "batch_size": args.batch_size, "world_size": world, "dtype": str(dtype),
        "resolution": resolution, "prompt_kind_label": 0,
        "image_encoding": "OpenCV INTER_AREA short-side resize, center crop; VAE posterior mean",
        "text_encoding": "training vocab table; truncate/pad; per-token L2 normalization and bridge scale",
        "annotations_sha256": digest(args.coco_root / "annotations/captions_val2014.json"),
        "token_config_sha256": digest(tcfg_path),
        "package_versions": package_versions(),
    }
    if args.task == "i2t":
        protocol["karpathy_sha256"] = digest(args.coco_root / "coco_karpathy_test.json")
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output / f"rank-{rank:05d}"
    shard_dir.mkdir()  # Never overwrite or combine previous generation runs.
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)
    if rank == 0:
        write_json(args.output / "protocol.json", {**protocol, "fingerprint": fingerprint})
    local = samples[rank::world]
    predictions = []
    loaded.sde.score_network = loaded.net
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        for start in range(0, len(local), args.batch_size):
            rows = local[start:start + args.batch_size]
            if args.task == "t2i":
                clean = encode_prompts([r["caption"] for r in rows], tokenizer, table, runtime, device)
                source = torch.randn_like(clean) if loaded.args.get("text_as_noise") else clean
            else:
                pixels = torch.stack([load_image(args.coco_root / "val2014" / r["file_name"], resolution) for r in rows]).to(device, dtype=dtype)
                pixels = pixels / 127.5 - 1.0
                clean = (vae.encode(pixels).latent_dist.mean.float() - runtime.latent_shift) * runtime.latent_scale
                source = torch.randn_like(clean) if loaded.args.get("image_as_noise") else clean
            y = torch.zeros(len(rows), device=device, dtype=torch.long)
            generated = loaded.sde.simulate(source, num_steps=args.steps, reverse=args.task == "i2t",
                                            cfg_scale=args.cfg_scale, ode=protocol["ode"], x_cond=clean, y=y)
            if args.task == "t2i":
                pixels = decode_latents(vae, generated, latent_scale=runtime.latent_scale, latent_shift=runtime.latent_shift)
                for row, pixel in zip(rows, pixels):
                    Image.fromarray(pixel.permute(1, 2, 0).numpy()).save(image_dir / f"{row['index']:06d}.png")
            else:
                flat = bridge_to_token_flat(generated.float(), config=bc)
                ids = loaded.token_decoder(flat).argmax(dim=-1).cpu().tolist()
                stops, _ = norm_based_token_stops(generated.float(), token_scale=runtime.token_scale, config=bc)
                predictions.extend({"image_id": row["image_id"], "caption": tokenizer.decode(tokens[:int(stop)], skip_special_tokens=True)}
                                   for row, tokens, stop in zip(rows, ids, stops))
            print(f"rank {rank}: {min(start + len(rows), len(local))}/{len(local)}", flush=True)
    if args.task == "i2t":
        write_json(shard_dir / "predictions.json", predictions)
    write_json(shard_dir / "done.json", {"fingerprint": fingerprint, "indices": [r["index"] for r in local]})


def validate_outputs(output, protocol):
    """Require every shard and exactly the selected output IDs before scoring."""
    output = Path(output)
    predictions = []
    world = protocol["world_size"]
    expected_dirs = {f"rank-{rank:05d}" for rank in range(world)}
    if {p.name for p in output.glob("rank-*")} != expected_dirs:
        raise ValueError("Generation rank directories do not match protocol.json.")
    for rank in range(world):
        directory = output / f"rank-{rank:05d}"
        receipt = json.loads((directory / "done.json").read_text())
        expected = [r["index"] for r in protocol["samples"][rank::world]]
        if receipt != {"fingerprint": protocol["fingerprint"], "indices": expected}:
            raise ValueError("Incomplete or mixed generation shards.")
        if protocol["task"] == "i2t":
            predictions.extend(json.loads((directory / "predictions.json").read_text()))
    if protocol["task"] == "t2i":
        expected = {f"{r['index']:06d}.png" for r in protocol["samples"]}
        if {p.name for p in (output / "images").iterdir()} != expected:
            raise ValueError("Generated image files do not match the requested samples.")
    else:
        ids = [row["image_id"] for row in predictions]
        if len(ids) != len(set(ids)) or set(ids) != {r["image_id"] for r in protocol["samples"]}:
            raise ValueError("Caption outputs contain missing, extra, or duplicate image IDs.")
    return predictions


def score(args):
    protocol = json.loads((args.output / "protocol.json").read_text())
    if digest(args.coco_root / "annotations/captions_val2014.json") != protocol["annotations_sha256"]:
        raise ValueError("Scoring annotations differ from generation annotations.")
    predictions = validate_outputs(args.output, protocol)
    images, references, _ = load_annotations(args.coco_root)
    metrics = {}
    details = {"fingerprint": protocol["fingerprint"], "num_samples": len(protocol["samples"])}
    details["full_protocol_sample_count"] = len(protocol["samples"]) == (5000 if protocol["task"] == "i2t" else 30000)
    if protocol["task"] == "i2t":
        from caption_metrics import score_captions
        preds = {row["image_id"]: [row["caption"]] for row in predictions}
        refs = {key: references[key] for key in preds}
        metrics = score_captions(refs, preds, spice=not args.no_spice)
        metrics.update({f"{key}_x100": 100 * value for key, value in list(metrics.items())})
        details.update(split="Karpathy test", references="all official COCO captions per image",
                       references_per_image=dict(Counter(len(r) for r in refs.values())),
                       units="native scorer units and explicitly suffixed x100 values")
        write_json(args.output / "predictions.json", sorted(predictions, key=lambda x: x["image_id"]))
    else:
        import torch
        from cleanfid import fid
        # Use the original full validation images, never VAE reconstructions.
        real_dir = args.coco_root / "val2014"
        if {p.name for p in real_dir.iterdir()} != set(images.values()):
            raise ValueError("FID needs exactly the full original val2014 image folder.")
        metrics["fid"] = float(fid.compute_fid(str(args.output / "images"), str(real_dir),
                                              mode=args.fid_mode, device=torch.device(args.device),
                                              batch_size=args.batch_size, num_workers=args.workers,
                                              use_dataparallel=False))
        details.update(split="COCO 2014 validation captions", fid_mode=args.fid_mode,
                       real_images=len(images), real_preprocessing="original JPEGs; selected clean-fid mode")
        if args.clip_score:
            metrics["clip_b16_cosine_x100"] = clip_score(args.output, protocol["samples"], args.device, args.batch_size)
    details["package_versions"] = package_versions()
    report = {"metrics": metrics, "protocol": details}
    write_json(args.output / "scores.json", report)
    print(json.dumps(report, indent=2))


def clip_score(output, samples, device, batch_size):
    """Explicitly named B/16 cosine metric, matching the existing GPIC helper."""
    import torch
    from PIL import Image
    from eval_metrics import _clip_score_manual, _get_or_make_clip
    model, processor = _get_or_make_clip(torch.device(device))
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            rows = samples[start:start + batch_size]
            import numpy as np
            pixels = []
            for row in rows:
                with Image.open(output / "images" / f"{row['index']:06d}.png") as image:
                    pixels.append(torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1))
            total += _clip_score_manual(model, processor, torch.stack(pixels), [r["caption"] for r in rows], device).sum().item()
    return total / len(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    gen = subparsers.add_parser("generate", help="Generate once; does not compute metrics.")
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("--data-root", type=Path, required=True, help="Training manifests and token vocabulary table.")
    gen.add_argument("--task", choices=["t2i", "i2t"], required=True)
    gen.add_argument("--num-samples", type=int, help="Default: 30k captions T2I / 5k Karpathy images I2T.")
    gen.add_argument("--steps", type=int, default=500)
    gen.add_argument("--cfg-scale", type=float, default=0.0)
    gen.add_argument("--ode", action="store_true", help="Use probability-flow ODE for SDE checkpoints; FM is always ODE.")
    gen.add_argument("--seed", type=int, default=0)
    scorer = subparsers.add_parser("score", help="Score completed output using original COCO references.")
    scorer.add_argument("--no-spice", action="store_true", help="Explicitly skip SPICE (e.g. smoke tests).")
    scorer.add_argument("--fid-mode", choices=["clean", "legacy_pytorch", "legacy_tensorflow"], default="clean")
    scorer.add_argument("--clip-score", action="store_true", help="Also compute CLIP B/16 cosine x100.")
    scorer.add_argument("--device", default="cuda")
    scorer.add_argument("--workers", type=int, default=4)
    for sub in (gen, scorer):
        sub.add_argument("--coco-root", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.command == "generate":
        import math
        if args.steps <= 0 or not math.isfinite(args.cfg_scale) or args.cfg_scale < 0:
            parser.error("--steps must be positive and --cfg-scale finite and nonnegative.")
        generate(args)
    else:
        if int(os.environ.get("WORLD_SIZE", 1)) != 1:
            parser.error("Run scoring once, outside torchrun.")
        score(args)


if __name__ == "__main__":
    main()
