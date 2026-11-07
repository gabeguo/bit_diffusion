"""Create dataset views with a shared, fixed Gaussian token vocabulary.

Only the vocabulary is new; latent shards, token IDs, masks, and captions are
symlinked. The existing table-mode loader turns z ~ N(0, I_d) into
sqrt(d) * z / ||z|| per valid token and keeps padding zero. No encoder or
training changes are needed. Run once before launching distributed workers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(train_root: Path, eval_root: Path, out_dir: Path, seed: int = 0) -> None:
    """Build immutable train/test views atomically; identical reruns are safe."""
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    sources = {"train": train_root.resolve(), "test": eval_root.resolve()}
    out_dir = out_dir.resolve()
    configs, shapes, fingerprints = {}, {}, {}
    for split, source in sources.items():
        if source == out_dir or source in out_dir.parents or out_dir in source.parents:
            raise ValueError("Output must be separate from the source datasets")
        token_manifest = source / "token_embed_config.json"
        cfg = json.loads(token_manifest.read_text())["config"]
        if cfg.get("token_emb_storage", "memmap") == "table":
            table = np.load(source / cfg.get("vocab_table", "token_vocab_table.npy"), mmap_mode="r")
            if table.ndim != 2 or min(table.shape) <= 0:
                raise ValueError(f"Invalid vocabulary shape in {source}: {table.shape}")
            shapes[split] = list(table.shape)
            del table
        elif split == "train" or cfg.get("token_emb_storage", "memmap") != "memmap":
            raise ValueError(f"{source}: training requires GPIC table-mode token sidecars")
        configs[split] = cfg
        fingerprints[split] = {
            "root": str(source), "config_sha256": _sha256(source / "config.json"),
            "token_config_sha256": _sha256(token_manifest),
        }
    if (shapes["train"] != shapes.get("test", shapes["train"])
            or configs["train"]["text_model"] != configs["test"]["text_model"]):
        raise ValueError("Train/eval must use the same tokenizer and vocabulary shape")

    metadata = {
        "version": 1, "seed": seed, "sources": fingerprints,
        "shape": shapes["train"], "dtype": "float32",
        "rng": "numpy.PCG64.standard_normal", "distribution": "N(0, 1)",
        "runtime_scaling": "sqrt(d) * z[:d] / norm(z[:d]); padding zero",
    }
    table_name = "token_vocab_table.npy"
    receipt = out_dir / "random_tokens.json"
    if out_dir.exists():
        saved = json.loads(receipt.read_text())
        checksum = saved.pop("table_sha256")
        if saved != metadata or _sha256(out_dir / table_name) != checksum:
            raise ValueError("Random dataset configuration changed; use a fresh --out-dir")
        print(f"Reusing random token data: {out_dir}")
        return

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    # Publish the entire view only when complete, so a failed preparation cannot
    # leave a plausible but incomplete dataset for a later training invocation.
    with tempfile.TemporaryDirectory(prefix=".random-tokens-", dir=out_dir.parent) as tmp:
        staging = Path(tmp) / "dataset"
        staging.mkdir()
        rng = np.random.Generator(np.random.PCG64(seed))
        table = rng.standard_normal(tuple(shapes["train"]), dtype=np.float32)
        np.save(staging / table_name, table)
        metadata["table_sha256"] = _sha256(staging / table_name)
        for split, source in sources.items():
            view = staging / split
            view.mkdir()
            cfg = dict(configs[split])
            old_table = (source / cfg.get("vocab_table", table_name)).resolve()
            for item in source.iterdir():
                if item.name in ("token_embed_config.json", table_name) or item.resolve() == old_table:
                    continue
                (view / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
            (view / table_name).symlink_to(Path("..") / table_name)
            cfg.update(vocab_table=table_name, token_emb_storage="table", mrl_dim=shapes["train"][1],
                       source_root=str(source),
                       random_token_seed=seed, random_token_table_sha256=metadata["table_sha256"])
            (view / "token_embed_config.json").write_text(json.dumps({"config": cfg}, indent=2) + "\n")
        (staging / receipt.name).write_text(json.dumps(metadata, indent=2) + "\n")
        os.rename(staging, out_dir)
    print(f"Created random token data: {out_dir} (seed={seed}, shape={shapes['train']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    prepare(args.train_root, args.eval_root, args.out_dir, args.seed)
