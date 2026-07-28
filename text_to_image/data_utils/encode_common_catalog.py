# NOTE: this is a legacy script from which we import a few functions.

"""
Stream common-canvas/commoncatalog-cc-by-sa, encode images with SD-VAE-ft-mse and
captions with Qwen3-Embedding-8B, and write the results to sharded numpy memmaps
suitable for fast random-access training.

One process per GPU, both models co-located on each GPU. CPU producers feed
bounded queues with decoded crops + captions; GPU consumers batch and write
their own non-overlapping rows of the global memmap.

Run:
    torchrun --standalone --nproc_per_node=8 scripts/encode_commoncatalog.py \
        --output-dir /data/cc_by_sa_latents \
        --num-pairs 10_000_000 \
        --shard-size 262144 \
        --vae-batch 256 \
        --text-batch 64 \
        --log-samples-every 2000

Resume the same run by re-invoking with the SAME flags; progress.json is read
automatically. Use --skip-input N to additionally skip N rows of this rank's
already-partitioned source stream before the saved cursor (rarely needed).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import fcntl
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
"""
from datasets import load_dataset
try:
    from datasets.distributed import split_dataset_by_node
except ImportError:  # older datasets versions
    split_dataset_by_node = None
"""
from diffusers import AutoencoderKL
from PIL import Image
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

"""
DATASET_NAME = "common-canvas/commoncatalog-cc-by-sa"
"""
VAE_MODEL = "stabilityai/sd-vae-ft-mse"
TEXT_MODEL = "Qwen/Qwen3-Embedding-8B"

IMAGE_SIZE = 256
LATENT_SHAPE = (4, 32, 32)         # (C, H, W) for 256x256 with f8 KL VAE
TEXT_DIM = 4096
SD_LATENT_SCALE = 0.18215          # only used for the decode preview
"""
TEXT_MAX_TOKENS = 48

CAPTION_FIELD = "blip2_caption"
IMAGE_FIELD_CANDIDATES = ("jpg", "image", "jpeg", "png")
URL_FIELD_CANDIDATES = ("url", "image_url", "photo_url", "flickr_url")
ID_FIELD_CANDIDATES = ("photoid", "id", "key", "image_id")

LOG = logging.getLogger("encode_cc")
"""


# ---------------------------------------------------------------------------
# Config & hashing (so we refuse to resume across incompatible runs)
# ---------------------------------------------------------------------------
"""
@dataclass
class Config:
    output_dir: str
    cache_dir: str
    num_pairs: int
    shard_size: int
    vae_batch: int
    text_batch: int
    cpu_workers: int
    world_size: int
    queue_size: int
    min_resolution: int
    log_samples_every: int
    log_samples_count: int
    checkpoint_every: int
    store_logvar: bool
    dataset_split: str
    dataset_name: str = DATASET_NAME
    vae_model: str = VAE_MODEL
    text_model: str = TEXT_MODEL
    image_size: int = IMAGE_SIZE
    text_dim: int = TEXT_DIM
    text_max_tokens: int = TEXT_MAX_TOKENS
    caption_field: str = CAPTION_FIELD
    dtype: str = "fp16"
    stream_partitioning: str = "split_dataset_by_node_v1"

    def hash(self) -> str:
        # Hash of fields that affect the *contents* of the output. We exclude
        # things like batch sizes and worker counts.
        relevant = {
            k: v for k, v in dataclasses.asdict(self).items()
            if k not in {
                "output_dir", "num_pairs", "vae_batch", "text_batch",
                "cpu_workers", "queue_size", "log_samples_every",
                "log_samples_count", "checkpoint_every",
            }
        }
        return hashlib.sha256(
            json.dumps(relevant, sort_keys=True).encode()
        ).hexdigest()


# ---------------------------------------------------------------------------
# Memmap helpers (sharded)
# ---------------------------------------------------------------------------

def shard_paths(output_dir: Path, shard_id: int) -> dict[str, Path]:
    return {
        "latents": output_dir / f"latents_{shard_id:05d}.memmap",
        "logvar":  output_dir / f"logvar_{shard_id:05d}.memmap",
        "text":    output_dir / f"text_{shard_id:05d}.memmap",
        "filled":  output_dir / f"filled_{shard_id:05d}.bin",  # uint8 bitmap
        "meta":    output_dir / f"meta_{shard_id:05d}.parquet",
    }
"""

def ensure_sized_file(path: Path, nbytes: int) -> None:
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        if path.exists():
            actual = path.stat().st_size
            if actual != nbytes:
                raise RuntimeError(
                    f"{path} has size {actual}, expected {nbytes}. "
                    "Refusing to continue."
                )
            return

        with open(path, "xb") as f:
            f.truncate(nbytes)
"""
def ensure_shard(output_dir: Path, shard_id: int, shard_size: int,
                 store_logvar: bool) -> None:
    paths = shard_paths(output_dir, shard_id)
    lat_bytes = shard_size * int(np.prod(LATENT_SHAPE)) * 2
    txt_bytes = shard_size * TEXT_DIM * 2
    fill_bytes = shard_size

    ensure_sized_file(paths["latents"], lat_bytes)
    if store_logvar:
        ensure_sized_file(paths["logvar"], lat_bytes)
    ensure_sized_file(paths["text"], txt_bytes)
    ensure_sized_file(paths["filled"], fill_bytes)

class ShardWriter:
    """"""Holds open memmaps for the current shard. Reopens on shard change.

    To avoid torn writes across the (latents, text, filled) triple, we never
    set filled[i] = 1 in memory until flush() has durably persisted the
    corresponding latent / text / logvar rows. flush() then sets the pending
    filled bits and msyncs the filled memmap last. As a result, any reader
    that observes filled[i] == 1 on disk can assume latents[i] and text[i]
    are also on disk (modulo the underlying device's write cache).
    """"""

    def __init__(self, output_dir: Path, shard_size: int, store_logvar: bool):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.store_logvar = store_logvar
        self._shard_id: Optional[int] = None
        self._latents: Optional[np.memmap] = None
        self._logvar: Optional[np.memmap] = None
        self._text: Optional[np.memmap] = None
        self._filled: Optional[np.memmap] = None
        self._pending_filled: list[int] = []

    def _open(self, shard_id: int):
        ensure_shard(self.output_dir, shard_id, self.shard_size,
                     self.store_logvar)
        paths = shard_paths(self.output_dir, shard_id)
        self._shard_id = shard_id
        self._latents = np.memmap(
            paths["latents"], dtype=np.float16, mode="r+",
            shape=(self.shard_size, *LATENT_SHAPE),
        )
        if self.store_logvar:
            self._logvar = np.memmap(
                paths["logvar"], dtype=np.float16, mode="r+",
                shape=(self.shard_size, *LATENT_SHAPE),
            )
        self._text = np.memmap(
            paths["text"], dtype=np.float16, mode="r+",
            shape=(self.shard_size, TEXT_DIM),
        )
        self._filled = np.memmap(
            paths["filled"], dtype=np.uint8, mode="r+",
            shape=(self.shard_size,),
        )

    def write(self, global_row: int, latent: np.ndarray,
              logvar: Optional[np.ndarray], text: np.ndarray) -> None:
        shard_id, local_row = divmod(global_row, self.shard_size)
        if shard_id != self._shard_id:
            self.flush()
            self._open(shard_id)
        self._latents[local_row] = latent
        if self.store_logvar and logvar is not None:
            self._logvar[local_row] = logvar
        self._text[local_row] = text
        # Intentionally do NOT set self._filled[local_row] yet; flush() owns
        # that, so the filled bit can only reach disk after the data has.
        self._pending_filled.append(local_row)

    def flush(self):
        for mm in (self._latents, self._logvar, self._text):
            if mm is not None:
                mm.flush()
        if self._filled is not None and self._pending_filled:
            for r in self._pending_filled:
                self._filled[r] = 1
            self._filled.flush()
        self._pending_filled = []


# ---------------------------------------------------------------------------
# CPU producer: stream + decode + crop
# ---------------------------------------------------------------------------

@dataclass
class RawSample:
    input_idx: int        # row index in this rank's source stream
    sample_id: str
    image: np.ndarray     # uint8 HWC RGB, exactly IMAGE_SIZE x IMAGE_SIZE
    caption: str


@dataclass
class FailedSample:
    input_idx: int
    sample_id: str
    url: str
    caption: str
    reason: str


def _pick_field(row: dict, candidates) -> Optional[object]:
    for k in candidates:
        if k in row and row[k] is not None:
            return row[k]
    return None
"""

def _decode_and_crop(img_bytes: bytes, size: int,
                     min_res: int) -> Optional[np.ndarray]:
    """JPEG/PNG bytes -> uint8 HWC RGB, resized-shorter-side then center-crop."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR HWC uint8
    if img is None:
        return None
    h, w = img.shape[:2]
    if min(h, w) < min_res:
        return None
    # resize shorter side to `size`
    if h < w:
        new_h, new_w = size, int(round(w * size / h))
    else:
        new_h, new_w = int(round(h * size / w)), size
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (new_h - size) // 2
    left = (new_w - size) // 2
    img = img[top:top + size, left:left + size]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

"""
def producer_loop(rank: int, world_size: int, cfg: Config,
                  start_source_cursor: int, extra_skip: int,
                  out_q: "mp.Queue[RawSample]",
                  fail_q: "mp.Queue[FailedSample]",
                  stop: "mp.Event") -> None:
    """"""One producer process per rank.

    We partition the streaming dataset by rank up-front so each rank consumes
    only its own stream shard and does not scan every upstream row.
    """"""
    try:
        ds = load_dataset(
            cfg.dataset_name, split=cfg.dataset_split, streaming=True,
            cache_dir=cfg.cache_dir,
        )
        if world_size > 1:
            if split_dataset_by_node is not None:
                ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)
            elif hasattr(ds, "shard"):
                raise ValueError("not supported")
                LOG.warning(
                    "rank %d: datasets.distributed unavailable; falling back "
                    "to IterableDataset.shard", rank
                )
                try:
                    ds = ds.shard(num_shards=world_size, index=rank, contiguous=True)
                except TypeError:
                    ds = ds.shard(num_shards=world_size, index=rank)
            else:
                raise RuntimeError(
                    "This datasets version cannot partition streaming data by rank. "
                    "Please upgrade `datasets`."
                )
    except Exception:
        LOG.exception("rank %d: failed to open dataset", rank)
        stop.set()
        return

    skip_total = start_source_cursor + extra_skip
    if skip_total > 0:
        ds = ds.skip(skip_total)

    input_idx = skip_total
    for row in ds:
        if stop.is_set():
            break

        sample_id = str(_pick_field(row, ID_FIELD_CANDIDATES) or
                        f"r{rank}_{input_idx}")
        url = str(_pick_field(row, URL_FIELD_CANDIDATES) or "")
        caption = row.get(cfg.caption_field) or ""
        caption = caption.strip() if isinstance(caption, str) else ""

        img_obj = _pick_field(row, IMAGE_FIELD_CANDIDATES)

        def fail(reason: str):
            try:
                fail_q.put(FailedSample(input_idx, sample_id, url,
                                        caption, reason), timeout=10)
            except queue.Full:
                pass

        if img_obj is None:
            fail("no_image_bytes")
            input_idx += 1
            continue
        if not caption:
            fail("no_caption")
            input_idx += 1
            continue

        # `datasets` may give us bytes, a dict {bytes, path}, or a PIL image.
        try:
            if isinstance(img_obj, dict) and "bytes" in img_obj and img_obj["bytes"]:
                img_bytes = img_obj["bytes"]
            elif isinstance(img_obj, (bytes, bytearray)):
                img_bytes = bytes(img_obj)
            elif isinstance(img_obj, Image.Image):
                buf = io.BytesIO()
                img_obj.convert("RGB").save(buf, format="JPEG", quality=95)
                img_bytes = buf.getvalue()
            else:
                fail("no_image_bytes")
                input_idx += 1
                continue
        except Exception:
            fail("image_extract_error")
            input_idx += 1
            continue

        cropped = _decode_and_crop(img_bytes, cfg.image_size, cfg.min_resolution)
        if cropped is None:
            fail("decode_or_too_small")
            input_idx += 1
            continue

        sample = RawSample(input_idx=input_idx, sample_id=sample_id,
                           image=cropped, caption=caption)
        # Block until queue has space — backpressure.
        while not stop.is_set():
            try:
                out_q.put(sample, timeout=1.0)
                break
            except queue.Full:
                continue

        input_idx += 1

    # signal end-of-stream for this rank
    while not stop.is_set():
        try:
            out_q.put(None, timeout=1.0)
            break
        except queue.Full:
            continue
"""

# ---------------------------------------------------------------------------
# GPU worker
# ---------------------------------------------------------------------------

# Last-non-pad-token pool, Qwen3-Embedding convention. We refuse to handle
# right-padded inputs explicitly because the existing pipeline always uses
# `padding_side="left"`; silently supporting both would let a tokenizer
# misconfiguration slip through to subtly wrong embeddings.
def _last_token_pool(hidden: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
    """Last-non-pad-token pooling for Qwen3-Embedding (left-padded inputs)."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if not left_padding:
        raise ValueError(
            "_last_token_pool expects left-padded inputs (Qwen3 convention). "
            "Make sure the tokenizer was constructed with padding_side='left'."
        )
    return hidden[:, -1]


def load_text_encoder(
    model_name: str,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: Optional[str] = None,
):
    """Load a Qwen3-Embedding-style model with a LEFT-padding tokenizer.

    Returns (tokenizer, model). The model is moved to `device` and set to eval().
    """
    tok = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", cache_dir=cache_dir,
    )
    text_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        cache_dir=cache_dir,
    ).to(device).eval()
    return tok, text_model


@torch.inference_mode()
def encode_texts(
    text_model,
    tokenizer,
    texts: list[str],
    *,
    device: torch.device,
    max_length: int,
    batch_size: int,
    autocast_dtype: torch.dtype = torch.bfloat16,
    out_dtype=np.float16,
) -> np.ndarray:
    """Tokenize, forward, last-token pool, L2 normalize. Returns (N, TEXT_DIM).

    Matches the encoder's inline implementation bit-for-bit (same padding,
    same truncation, same autocast, same pooling, same normalization,
    same fp16 cast on the way out). Stage B (re-embedding new captions)
    imports this so we cannot drift across versions.
    """
    n = len(texts)
    if n == 0:
        return np.empty((0, TEXT_DIM), dtype=out_dtype)
    embs = np.empty((n, TEXT_DIM), dtype=out_dtype)
    for i in range(0, n, batch_size):
        sub = texts[i:i + batch_size]
        enc = tokenizer(
            sub, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=True):
            out = text_model(
                **enc, use_cache=False, output_hidden_states=False,
            )
        pooled = _last_token_pool(out.last_hidden_state, enc.attention_mask)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        assert pooled.shape == (len(sub), TEXT_DIM), \
            f"unexpected pooled shape {tuple(pooled.shape)} for batch of {len(sub)}"
        if out_dtype == np.float16:
            embs[i:i + len(sub)] = pooled.to(torch.float16).cpu().numpy()
        elif out_dtype == np.float32:
            embs[i:i + len(sub)] = pooled.to(torch.float32).cpu().numpy()
        else:
            raise ValueError(f"unsupported out_dtype: {out_dtype}")
    return embs

"""
def _rows_for_rank(total_rows: int, rank: int, world_size: int) -> int:
    if rank >= total_rows:
        return 0
    return ((total_rows - 1 - rank) // world_size) + 1


def _rows_done_from_next_row(next_row: int, rank: int, world_size: int) -> int:
    if next_row <= rank:
        return 0
    return ((next_row - 1 - rank) // world_size) + 1


def gpu_worker(rank: int, world_size: int, cfg: Config,
               extra_skip: int) -> None:
    """"""One process per GPU. Owns its share of memmap rows.""""""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    is_main = (rank == 0)
    next_row, start_source_cursor = load_rank_resume(cfg, rank, world_size)

    # ---- load models ----
    LOG.info("rank %d: loading VAE", rank)
    vae = AutoencoderKL.from_pretrained(cfg.vae_model, torch_dtype=torch.bfloat16, cache_dir=cfg.cache_dir)
    vae = vae.to(device).eval()
    try:
        vae.encoder = torch.compile(vae.encoder, mode="reduce-overhead")
    except Exception:
        LOG.warning("rank %d: torch.compile on VAE failed, continuing", rank)

    LOG.info("rank %d: loading %s", rank, cfg.text_model)
    tok, text_model = load_text_encoder(
        cfg.text_model, device, dtype=torch.bfloat16, cache_dir=cfg.cache_dir,
    )

    # ---- queues + producer thread (in-process; CPU-bound work runs in
    #      a thread pool of decoders). We keep the actual HF stream in a
    #      separate *process* so its GIL/IO doesn't fight the GPU thread. ----
    ctx = mp.get_context("spawn")
    raw_q: "mp.Queue[RawSample]" = ctx.Queue(maxsize=cfg.queue_size)
    fail_q: "mp.Queue[FailedSample]" = ctx.Queue(maxsize=cfg.queue_size)
    stop = ctx.Event()

    prod = ctx.Process(
        target=producer_loop,
        args=(rank, world_size, cfg, start_source_cursor, extra_skip,
              raw_q, fail_q, stop),
        daemon=True,
    )
    prod.start()

    # ---- writer ----
    output_dir = Path(cfg.output_dir)
    writer = ShardWriter(output_dir, cfg.shard_size, cfg.store_logvar)

    # Each rank owns rows {rank, rank+world_size, rank+2*world_size, ...}
    valid_count_local = _rows_done_from_next_row(next_row, rank, world_size)
    failed_count_local = 0
    source_cursor = start_source_cursor
    ended = False

    # In-memory metadata buffer flushed to per-shard parquet
    meta_buffer: dict[int, list[dict]] = {}
    flush_seq: dict[int, int] = {}

    pbar_total = _rows_for_rank(cfg.num_pairs, rank, world_size)
    pbar = tqdm(total=pbar_total, initial=min(valid_count_local, pbar_total),
                disable=not is_main, desc=f"rank {rank}", smoothing=0.05)

    fail_writer_path = output_dir / f"failures_rank{rank}.jsonl"
    fail_writer_path.parent.mkdir(parents=True, exist_ok=True)
    fail_f = open(fail_writer_path, "a", buffering=1)

    last_ckpt_t = time.time()

    def drain_failures():
        nonlocal failed_count_local, source_cursor
        while True:
            try:
                fs = fail_q.get_nowait()
            except queue.Empty:
                return
            fail_f.write(json.dumps(dataclasses.asdict(fs)) + "\n")
            failed_count_local += 1
            source_cursor = max(source_cursor, fs.input_idx + 1)

    def maybe_log_samples(images_b: torch.Tensor, latents_b: torch.Tensor,
                          captions: list[str]):
        if not is_main or cfg.log_samples_every <= 0:
            return
        if (valid_count_local // cfg.log_samples_every) == \
           ((valid_count_local - len(captions)) // cfg.log_samples_every):
            return
        n = min(cfg.log_samples_count, latents_b.shape[0])
        with torch.inference_mode():
            decoded = vae.decode(latents_b[:n]).sample
        # We stored RAW mean (unscaled), so just pass it directly.
        decoded = (decoded.float().clamp(-1, 1) + 1) / 2
        orig = (images_b[:n].float().clamp(-1, 1) + 1) / 2
        grid = make_grid(torch.cat([orig.cpu(), decoded.cpu()], dim=0),
                         nrow=n, padding=2)
        out_dir = output_dir / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"step_{valid_count_local:09d}"
        save_image(grid, out_dir / f"{tag}.png")
        with open(out_dir / f"{tag}.txt", "w") as f:
            for c in captions[:n]:
                f.write(c + "\n")

    def write_checkpoint():
        # Reduce per-rank counts via files (cheap, no NCCL needed).
        ckpt = {
            "config_hash": cfg.hash(),
            "rank": rank,
            "next_row": next_row,
            "source_cursor": source_cursor,
            "valid_count_local": valid_count_local,
            "failed_count_local": failed_count_local,
        }
        tmp = output_dir / f".progress_rank{rank}.json.tmp"
        final = output_dir / f"progress_rank{rank}.json"
        tmp.write_text(json.dumps(ckpt))
        os.replace(tmp, final)

    # ---- main loop ----
    pending: list[RawSample] = []

    while next_row < cfg.num_pairs and not ended:
        # gather a batch
        while len(pending) < cfg.vae_batch:
            try:
                item = raw_q.get(timeout=5.0)
            except queue.Empty:
                drain_failures()
                if not prod.is_alive():
                    prod.join(timeout=0)
                    raise RuntimeError(
                        f"rank {rank}: producer died unexpectedly "
                        f"with exit code {prod.exitcode}"
                    )
                continue
            if item is None:
                ended = True
                break
            pending.append(item)

        drain_failures()

        if not pending:
            continue

        # Sub-batches: VAE batch is large, text batch is smaller. We cut the
        # pending list into VAE-sized chunks and run text in nested chunks.
        batch = pending[: cfg.vae_batch]
        pending = pending[cfg.vae_batch:]

        # ---- image -> tensor ----
        imgs = np.stack([s.image for s in batch], axis=0)        # (B,H,W,3) uint8
        imgs_t = torch.from_numpy(imgs).to(device, non_blocking=True)
        imgs_t = imgs_t.permute(0, 3, 1, 2).contiguous().to(torch.bfloat16) / 127.5 - 1.0

        # ---- VAE encode ----
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=True
        ):
            posterior = vae.encode(imgs_t).latent_dist
        mean = posterior.mean.to(torch.float16).cpu().numpy()
        logvar = posterior.logvar.to(torch.float16).cpu().numpy() \
            if cfg.store_logvar else None
        assert np.isfinite(mean).all()
        if logvar is not None:
            assert np.isfinite(logvar).all()

        # ---- text encode (sub-batches) ----
        # Matches https://huggingface.co/Qwen/Qwen3-Embedding-8B
        captions = [s.caption for s in batch]
        text_emb = encode_texts(
            text_model, tok, captions,
            device=device,
            max_length=cfg.text_max_tokens,
            batch_size=cfg.text_batch,
        )

        # ---- write rows ----
        written_this_batch = 0
        for j, s in enumerate(batch):
            if next_row >= cfg.num_pairs:
                ended = True
                break
            writer.write(
                next_row, mean[j],
                logvar[j] if logvar is not None else None,
                text_emb[j],
            )
            shard_id = next_row // cfg.shard_size
            local_row = next_row % cfg.shard_size
            meta_buffer.setdefault(shard_id, []).append({
                "global_row": next_row,
                "local_row": local_row,
                "input_idx": s.input_idx,
                "sample_id": s.sample_id,
                "caption": s.caption,
            })
            next_row += world_size
            valid_count_local += 1
            written_this_batch += 1
            source_cursor = max(source_cursor, s.input_idx + 1)

        pbar.update(written_this_batch)

        # decode-preview hook
        maybe_log_samples(imgs_t, posterior.mean, captions)

        # checkpoint
        if (valid_count_local % cfg.checkpoint_every) < cfg.vae_batch \
                or (time.time() - last_ckpt_t) > 60 * 60:
            writer.flush()
            _flush_meta(output_dir, meta_buffer, rank, flush_seq)
            write_checkpoint()
            last_ckpt_t = time.time()

    pbar.close()
    writer.flush()
    _flush_meta(output_dir, meta_buffer, rank, flush_seq)
    write_checkpoint()
    fail_f.close()
    stop.set()
    prod.join(timeout=30)
    LOG.info("rank %d: done. valid=%d failed=%d",
             rank, valid_count_local, failed_count_local)


def _flush_meta(output_dir: Path, meta_buffer: dict[int, list[dict]],
                rank: int, flush_seq: dict[int, int]) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for shard_id, rows in list(meta_buffer.items()):
        if not rows:
            continue
        seq = flush_seq.get(shard_id, 0)
        # discover existing seq on first flush after resume
        if seq == 0:
            existing = sorted(meta_dir.glob(
                f"meta_{shard_id:05d}_rank{rank}_*.parquet"))
            if existing:
                last = existing[-1].stem.rsplit("_", 1)[-1]
                seq = int(last) + 1
        path = meta_dir / f"meta_{shard_id:05d}_rank{rank}_{seq:06d}.parquet"
        tmp  = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(pa.Table.from_pylist(rows), tmp)
        os.replace(tmp, path)
        flush_seq[shard_id] = seq + 1
        meta_buffer[shard_id] = []

# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", type=str, default=DATASET_NAME)
    p.add_argument("--output-dir", type=str, default="/data/common_catalog_cc_by_sa")
    p.add_argument("--cache-dir", type=str, default="/data/cache_sub")
    p.add_argument("--num-pairs", type=int, default=8_000_000,
                   help="Total dense rows to produce across all GPUs.")
    p.add_argument("--shard-size", type=int, default=262144)
    p.add_argument("--vae-batch", type=int, default=256)
    p.add_argument("--text-batch", type=int, default=64)
    p.add_argument("--cpu-workers", type=int, default=4,
                   help="Reserved for future use; producer is currently 1/rank.")
    p.add_argument("--queue-size", type=int, default=512)
    p.add_argument("--min-resolution", type=int, default=256)
    p.add_argument("--log-samples-every", type=int, default=2000,
                   help="0 to disable.")
    p.add_argument("--log-samples-count", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument("--store-logvar", action="store_true")
    p.add_argument("--dataset-split", type=str, default="train")
    p.add_argument("--skip-input", type=int, default=0,
                   help="Skip N additional rows in this rank's source stream "
                        "BEFORE the saved cursor.")
    p.add_argument("--dry-run", action="store_true",
                   help="Process at most 100 samples per GPU.")
    p.add_argument("--world-size", type=int,
                   default=int(os.environ.get("WORLD_SIZE", "1")))
    return p.parse_args()


def ensure_config_compatible(cfg: Config) -> None:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "config.json"
    if cfg_path.exists():
        prev = json.loads(cfg_path.read_text())
        if prev.get("hash") != cfg.hash():
            raise RuntimeError(
                f"Config hash mismatch. Existing run used different settings.\n"
                f"  existing: {prev.get('hash')}\n"
                f"  current : {cfg.hash()}\n"
                f"Refusing to resume. Use a different --output-dir or delete it."
            )
    else:
        payload = json.dumps({
            "hash": cfg.hash(),
            "config": dataclasses.asdict(cfg),
        }, indent=2)

        tmp = out / f".config.{os.getpid()}.json.tmp"
        tmp.write_text(payload)
        os.replace(tmp, cfg_path)

def load_rank_resume(cfg: Config, rank: int, world_size: int) -> tuple[int, int]:
    """"""Returns (next_row, source_cursor) for a single rank.""""""
    out = Path(cfg.output_dir)
    p = out / f"progress_rank{rank}.json"
    if not p.exists():
        return rank, 0
    try:
        d = json.loads(p.read_text())
    except Exception:
        LOG.warning("rank %d: failed to parse %s, starting fresh",
                    rank, str(p))
        return rank, 0

    next_row = int(d.get("next_row", rank))
    source_cursor = int(d.get("source_cursor", 0))

    if next_row < rank:
        next_row = rank
    if (next_row % world_size) != rank:
        raise RuntimeError(
            f"rank {rank}: progress file has next_row={next_row} which is not "
            f"compatible with world_size={world_size}. This means the saved run "
            f"used a different world_size and the config-hash check missed it. "
            f"Refusing to continue."
        )

    return next_row, max(0, source_cursor)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(levelname)s %(message)s",
    )
    args = parse_args()
    if args.dry_run:
        args.num_pairs = min(args.num_pairs, 100 * max(args.world_size, 1))
        args.checkpoint_every = 50

    cfg = Config(
        dataset_name=args.dataset_name,
        world_size=args.world_size,
        output_dir=args.output_dir,
        num_pairs=args.num_pairs,
        shard_size=args.shard_size,
        vae_batch=args.vae_batch,
        text_batch=args.text_batch,
        cpu_workers=args.cpu_workers,
        queue_size=args.queue_size,
        min_resolution=args.min_resolution,
        log_samples_every=args.log_samples_every,
        log_samples_count=args.log_samples_count,
        checkpoint_every=args.checkpoint_every,
        store_logvar=args.store_logvar,
        dataset_split=args.dataset_split,
        cache_dir=args.cache_dir,
    )

    ensure_config_compatible(cfg)

    world_size = args.world_size
    if world_size <= 1:
        gpu_worker(0, 1, cfg, args.skip_input)
        return

    # When launched via torchrun, each rank already has its own process. We
    # detect that by LOCAL_RANK and just run the worker directly.
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        rank = int(local_rank)
        gpu_worker(rank, world_size, cfg, args.skip_input)
        return

    # Fallback: spawn ourselves.
    mp.spawn(
        gpu_worker,
        args=(world_size, cfg, args.skip_input),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
"""