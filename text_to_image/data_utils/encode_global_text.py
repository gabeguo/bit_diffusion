"""
Global text-embedding sidecar for tokens written by encode_gpic.py.

Post-hoc twin of encode_gpic.py's --global-text-emb: instead of re-running the
VAE pass, it reads each source shard's stored Qwen tokens directly, decodes them
back to text, re-encodes each caption AS A WHOLE SENTENCE (attention, last-token
pool -- contextual, not the context-independent per-token vocab table), MRL
truncates to 2048-d, L2 normalizes, and writes features + a per-shard
gtext_filled bitmap (gtext_{sid}.memmap / gtext_filled_{sid}.memmap).

- Layout: shards are discovered by globbing the source filled_*.bin bitmaps, so
  the sparse, per-rank-reserved GPIC shard ids work unchanged.
- World size is independent of the encode run: ranks split the shard list and
  resume is driven by the gtext_filled bitmap (a row is skipped once its bit is
  set), so any world size can resume any prior run with no migration.
- --percent keeps a PREFIX of shards covering the first `percent` of all filled
  rows (global shard order). Kept shards are written densely and the rest get no
  files at all, so there are no subsetting-induced sparse files to inflate on
  copy/archive, and larger percents are supersets of smaller ones.
- Stored tokens were tokenized with add_special_tokens=False and capped at
  token_seq_len, so detok->re-tok caps content at that length but re-adds the
  EOS that Qwen3-Embedding's last-token pooling expects.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm

try:
    from encode_common_catalog import _last_token_pool, ensure_sized_file, load_text_encoder
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from encode_common_catalog import (  # noqa: E402
        _last_token_pool, ensure_sized_file, load_text_encoder)


def dist_env() -> tuple[int, int, int]:
    """(global_rank, world_size, local_rank) from torchrun or SLURM env."""
    g = lambda *ks, d=0: int(next((os.environ[k] for k in ks if k in os.environ), d))
    return (g("RANK", "SLURM_PROCID"),
            g("WORLD_SIZE", "SLURM_NTASKS", d=1),
            g("LOCAL_RANK", "SLURM_LOCALID"))


@torch.inference_mode()
def encode(model, tok, texts, *, device, max_length, mrl_dim):
    """list[str] -> (B, mrl_dim) fp16: last-token pool, MRL truncate, L2 norm."""
    enc = tok(texts, padding=True, truncation=True, max_length=max_length,
              return_tensors="pt").to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**enc, use_cache=False, output_hidden_states=False)
    v = _last_token_pool(out.last_hidden_state, enc.attention_mask)[:, :mrl_dim]
    v = F.normalize(v.float(), p=2, dim=1)      # truncate first, then renormalize
    return v.cpu().numpy().astype(np.float16)


def run_encode(args) -> None:
    rank, world, local = dist_env()
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)
    src, out = Path(args.source_root), Path(args.text_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())["config"]
    S = int(cfg["shard_size"])
    T = int(cfg["token_seq_len"])
    src_model = cfg["text_model"]               # tokenizer that produced the stored ids

    src_tok = AutoTokenizer.from_pretrained(src_model, cache_dir=args.cache_dir)
    tok, model = load_text_encoder(args.text_model, device, cache_dir=args.cache_dir)
    with torch.inference_mode():
        probe = model(**tok(["x"], return_tensors="pt").to(device))
    native = probe.last_hidden_state.shape[-1]
    assert args.mrl_dim <= native, f"mrl_dim {args.mrl_dim} > native dim {native}"
    flat_dim = args.mrl_dim

    if rank == 0:
        (out / "gtext_config.json").write_text(json.dumps({"config": {
            "source_root": str(src.resolve()), "shard_size": S,
            "src_text_model": src_model, "text_model": args.text_model,
            "native_dim": native, "mrl_dim": flat_dim, "flat_dim": flat_dim,
            "max_length": args.max_length, "store": "global", "percent": args.percent,
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
    tdir = src / "token_original"

    def done_count(sid: int) -> int:
        p = out / f"gtext_filled_{sid:05d}.memmap"
        return int(np.count_nonzero(np.fromfile(p, np.uint8, S))) if p.exists() else 0

    encoded = 0
    pbar = tqdm(total=sum(count_by_sid[sid] for sid in my_shards),
                initial=sum(done_count(sid) for sid in my_shards),
                disable=(rank != 0), desc="gtext", unit="cap")
    for sid in my_shards:
        src_filled = np.fromfile(src / f"filled_{sid:05d}.bin", np.uint8, S)
        feat_path, fill_path = out / f"gtext_{sid:05d}.memmap", out / f"gtext_filled_{sid:05d}.memmap"
        ensure_sized_file(feat_path, S * flat_dim * 2)
        ensure_sized_file(fill_path, S)
        ids = np.memmap(tdir / f"token_original_token_ids_{sid:05d}.memmap", np.int32, "r", shape=(S, T))
        lens = np.memmap(tdir / f"token_original_lengths_{sid:05d}.memmap", np.uint16, "r", shape=(S,))
        feats = np.memmap(feat_path, np.float16, "r+", shape=(S, flat_dim))
        done = np.memmap(fill_path, np.uint8, "r+", shape=(S,))

        todo = np.flatnonzero((src_filled != 0) & (done == 0))
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            texts = src_tok.batch_decode(
                [ids[r][:int(lens[r])].tolist() for r in chunk], skip_special_tokens=True)
            feats[chunk] = encode(model, tok, texts, device=device,
                                  max_length=args.max_length, mrl_dim=flat_dim)
            feats.flush()              # data durable before the row is marked done
            done[chunk] = 1
            done.flush()
            encoded += len(chunk)
            pbar.update(len(chunk))
        del ids, lens, feats, done
        pbar.set_postfix_str(f"shard {sid} done")
    pbar.close()

    print(f"rank {rank}: done, encoded {encoded} rows.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", required=True, help="encode_gpic.py output dir (one preset).")
    p.add_argument("--text-dir", required=True)
    p.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    p.add_argument("--text-model", default="Qwen/Qwen3-Embedding-4B")
    p.add_argument("--mrl-dim", type=int, default=2048)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--percent", type=float, default=100.0,
                   help="Keep a prefix of shards covering the first %% of filled rows.")
    return p.parse_args()


if __name__ == "__main__":
    run_encode(parse_args())
