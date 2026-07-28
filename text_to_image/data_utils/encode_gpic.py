"""
Encode GPIC (stanford-vision-lab/gpic) webdataset tars into sharded memmaps that
are byte-compatible with the CommonCatalog latent pipeline.

Per accepted sample we store: SD-VAE-ft-mse latent mean (+logvar), Qwen3 token
ids/mask/length (context-independent embeddings are reconstructed at train time
by gathering a shared vocab table), the caption_type, and key/caption metadata.
The global 4096-d pooled text embedding is optional (--global-text-emb).

Embarrassingly parallel: one process per GPU (rank = SLURM_PROCID), NO NCCL.
Each rank owns a disjoint shard-id range so every file has a single writer
(multi-node safe). Tars are read in place from --tars-dir (expected on $SCRATCH).
Resume is tar-granular: re-invoke with the SAME world size and flags.

    srun --ntasks=$((NODES*4)) --gpus-per-task=1 \
        python encode_gpic.py --tars-dir $CFS/.../gpic/train \
            --output-dir $SCRATCH/datasets/text_to_image/gpic_latents
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import logging
import os
import shutil
import tarfile
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from diffusers import AutoencoderKL
from torchvision.utils import make_grid, save_image
from transformers import AutoModel, AutoTokenizer

from encode_common_catalog import (
    IMAGE_SIZE,
    TEXT_DIM,
    TEXT_MODEL,
    _decode_and_crop,
    encode_texts,
    ensure_sized_file,
    load_text_encoder,
)
from token_bridge import runtime_from_preset

LOG = logging.getLogger("encode_gpic")

CAPTION_TYPES = ("tag", "short", "medium", "long")
CTYPE_TO_IDX = {c: i for i, c in enumerate(CAPTION_TYPES)}
TOKEN_SEQ_LEN = 128
TOKEN_EMB_DIM = 128
MAX_SENTENCE_LENGTH = 192
MRL_DIM = 128                 # stored raw; train-time loader truncates+normalizes
IMAGE_EXTS = ("jpg", "jpeg", "png", "webp")
STAGE_PREFETCH = 2            # tars prefetched to scratch ahead of the GPU


def dist_env() -> tuple[int, int, int]:
    """(global_rank, world_size, local_rank) from SLURM or torchrun."""
    g = lambda *ks, d=0: int(next((os.environ[k] for k in ks if k in os.environ), d))
    return (g("RANK", "SLURM_PROCID"),
            g("WORLD_SIZE", "SLURM_NTASKS", d=1),
            g("LOCAL_RANK", "SLURM_LOCALID"))


# --------------------------------------------------------------------------- #
# Config / manifests
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    tars_dir: str
    output_dir: str
    cache_dir: str
    shard_size: int
    shards_per_rank: int
    min_resolution: int
    store_logvar: bool
    store_text: bool
    vae_batch: int
    text_batch: int
    decode_workers: int
    log_every: int
    checkpoint_every: int
    splits: tuple[str, ...]
    bridge_preset: str
    latent_shape: tuple[int, int, int]
    vae_model: str
    vae_subfolder: str | None
    vae_kind: str
    latent_scale: float
    latent_shift: float
    image_size: int = IMAGE_SIZE
    text_dim: int = TEXT_DIM
    token_seq_len: int = TOKEN_SEQ_LEN
    token_emb_dim: int = TOKEN_EMB_DIM
    mrl_dim: int = MRL_DIM
    text_model: str = TEXT_MODEL
    dataset: str = "gpic"

    def hash(self) -> str:
        skip = {"output_dir", "cache_dir", "vae_batch", "text_batch", "decode_workers",
                "log_every", "checkpoint_every", "shards_per_rank"}
        rel = {k: v for k, v in dataclasses.asdict(self).items() if k not in skip}
        return hashlib.sha256(json.dumps(rel, sort_keys=True).encode()).hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def ensure_manifests(cfg: Config) -> None:
    """Write config.json + token_embed_config.json, refusing incompatible resumes."""
    out = Path(cfg.output_dir)
    cfg_path = out / "config.json"
    if cfg_path.exists():
        prev = json.loads(cfg_path.read_text()).get("hash")
        if prev != cfg.hash():
            raise RuntimeError(
                f"config.json hash mismatch ({prev} != {cfg.hash()}); use a fresh "
                "--output-dir or delete the existing one."
            )
        return
    _write_json_atomic(cfg_path, {"hash": cfg.hash(), "config": dataclasses.asdict(cfg)})
    _write_json_atomic(out / "token_embed_config.json", {"config": {
        "source_root": str(out.resolve()),
        "shard_size": cfg.shard_size,
        "token_seq_len": cfg.token_seq_len,
        "token_emb_dim": cfg.token_emb_dim,
        "mrl_dim": cfg.mrl_dim,
        "token_emb_storage": "table",
        "vocab_table": "token_vocab_table.npy",
        "text_model": cfg.text_model,
    }})


def ensure_tar_manifest(cfg: Config, tars: list[Path]) -> None:
    """Freeze the tar set on first run so resumes index a stable list."""
    path = Path(cfg.output_dir) / "tars.json"
    if not path.exists():
        _write_json_atomic(path, {"tars": [t.name for t in tars]})


def load_frozen_tars(cfg: Config) -> list[Path]:
    """Resolve the frozen tar names back to paths (rglob handles subdirs)."""
    names = json.loads((Path(cfg.output_dir) / "tars.json").read_text())["tars"]
    by_name = {p.name: p for p in Path(cfg.tars_dir).rglob("*.tar")}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise RuntimeError(
            f"{len(missing)} tar(s) from tars.json are missing under {cfg.tars_dir} "
            f"(e.g. {missing[:3]}); the tar set must be stable across resume.")
    return [by_name[n] for n in names]


# --------------------------------------------------------------------------- #
# Vocab table (context-independent token embeddings; built once)
# --------------------------------------------------------------------------- #

@torch.inference_mode()
def _build_table(model, device, mrl_dim: int, batch: int = 2048) -> np.ndarray:
    vocab = int(model.config.vocab_size)
    table = np.empty((vocab, mrl_dim), dtype=np.float16)
    for s in range(0, vocab, batch):
        ids = torch.arange(s, min(s + batch, vocab), device=device).unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                        use_cache=False, output_hidden_states=False)
        # RAW first mrl_dim dims (no normalization; loader normalizes at use dim).
        table[s:s + ids.shape[0]] = out.last_hidden_state[:, 0, :mrl_dim].float().cpu().numpy()
    return table


def ensure_vocab_table(cfg: Config, device, model=None) -> None:
    path = Path(cfg.output_dir) / "token_vocab_table.npy"
    if path.exists():
        return
    lock = open(path.with_suffix(".npy.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not path.exists():
            LOG.info("building vocab table (%d-d) ...", cfg.mrl_dim)
            own = model is None
            if own:
                model = AutoModel.from_pretrained(
                    cfg.text_model, torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa", cache_dir=cfg.cache_dir,
                ).to(device).eval()
            table = _build_table(model, device, cfg.mrl_dim)
            tmp = path.with_suffix(".npy.tmp")
            with open(tmp, "wb") as f:        # file handle => np.save won't append .npy
                np.save(f, table)
            os.replace(tmp, path)
            if own:
                del model
                torch.cuda.empty_cache()
    except BlockingIOError:
        pass
    finally:
        lock.close()
    while not path.exists():            # built by another rank
        time.sleep(2.0)


# --------------------------------------------------------------------------- #
# Sharded writer (one rank owns its shard range; filled bitmaps flushed last)
# --------------------------------------------------------------------------- #

def shard_paths(out: Path, sid: int) -> dict[str, Path]:
    tok = out / "token_original"
    return {
        "latents": out / f"latents_{sid:05d}.memmap",
        "logvar": out / f"logvar_{sid:05d}.memmap",
        "text": out / f"text_{sid:05d}.memmap",
        "filled": out / f"filled_{sid:05d}.bin",
        "ctype": out / f"caption_type_{sid:05d}.memmap",
        "ids": tok / f"token_original_token_ids_{sid:05d}.memmap",
        "mask": tok / f"token_original_mask_{sid:05d}.memmap",
        "len": tok / f"token_original_lengths_{sid:05d}.memmap",
        "tok_filled": tok / f"token_original_filled_{sid:05d}.memmap",
    }


class ShardWriter:
    def __init__(self, cfg: Config):
        self.out = Path(cfg.output_dir)
        self.S = cfg.shard_size
        self.store_logvar = cfg.store_logvar
        self.store_text = cfg.store_text
        self.latent_shape = cfg.latent_shape
        self.latent_elems = int(np.prod(cfg.latent_shape))
        self.token_seq_len = cfg.token_seq_len
        self.text_dim = cfg.text_dim
        self._sid: Optional[int] = None
        self._mm: dict[str, np.memmap] = {}
        self._pending: list[int] = []

    def _open(self, sid: int) -> None:
        p = shard_paths(self.out, sid)
        p["ids"].parent.mkdir(parents=True, exist_ok=True)
        S = self.S
        ensure_sized_file(p["latents"], S * self.latent_elems * 2)
        ensure_sized_file(p["filled"], S)
        ensure_sized_file(p["ctype"], S)
        ensure_sized_file(p["ids"], S * self.token_seq_len * 4)
        ensure_sized_file(p["mask"], S * self.token_seq_len)
        ensure_sized_file(p["len"], S * 2)
        ensure_sized_file(p["tok_filled"], S)
        if self.store_logvar:
            ensure_sized_file(p["logvar"], S * self.latent_elems * 2)
        if self.store_text:
            ensure_sized_file(p["text"], S * self.text_dim * 2)

        f16, u8 = np.float16, np.uint8
        mm = {
            "latents": np.memmap(p["latents"], f16, "r+", shape=(S, *self.latent_shape)),
            "filled": np.memmap(p["filled"], u8, "r+", shape=(S,)),
            "ctype": np.memmap(p["ctype"], u8, "r+", shape=(S,)),
            "ids": np.memmap(p["ids"], np.int32, "r+", shape=(S, self.token_seq_len)),
            "mask": np.memmap(p["mask"], u8, "r+", shape=(S, self.token_seq_len)),
            "len": np.memmap(p["len"], np.uint16, "r+", shape=(S,)),
            "tok_filled": np.memmap(p["tok_filled"], u8, "r+", shape=(S,)),
        }
        if self.store_logvar:
            mm["logvar"] = np.memmap(p["logvar"], f16, "r+", shape=(S, *self.latent_shape))
        if self.store_text:
            mm["text"] = np.memmap(p["text"], f16, "r+", shape=(S, self.text_dim))
        self._mm, self._sid = mm, sid

    def write(self, global_row: int, *, latent, logvar, text, ctype, ids, mask, length) -> None:
        sid, row = divmod(global_row, self.S)
        if sid != self._sid:
            self.flush()
            self._open(sid)
        m = self._mm
        m["latents"][row] = latent
        m["ctype"][row] = ctype
        m["ids"][row] = ids
        m["mask"][row] = mask
        m["len"][row] = length
        if self.store_logvar:
            m["logvar"][row] = logvar
        if self.store_text:
            m["text"][row] = text
        self._pending.append(row)       # filled bits set only in flush()

    def flush(self) -> None:
        if not self._mm:
            return
        for k, mm in self._mm.items():
            if k not in ("filled", "tok_filled"):
                mm.flush()
        for row in self._pending:
            self._mm["filled"][row] = 1
            self._mm["tok_filled"][row] = 1
        self._mm["filled"].flush()
        self._mm["tok_filled"].flush()
        self._pending.clear()


# --------------------------------------------------------------------------- #
# Per-rank encode
# --------------------------------------------------------------------------- #

def list_tars(cfg: Config) -> list[Path]:
    root = Path(cfg.tars_dir)
    tars = sorted(p for p in root.rglob("*.tar")
                  if not cfg.splits or any(s in p.name for s in cfg.splits))
    if not tars:
        raise RuntimeError(f"no .tar files under {root} (splits={cfg.splits})")
    return tars


def tokenize(tokenizer, caption: str):
    ids = tokenizer(caption, add_special_tokens=False, truncation=True,
                    max_length=TOKEN_SEQ_LEN)["input_ids"][:TOKEN_SEQ_LEN]
    assert tokenizer.pad_token_id not in ids
    n = len(ids)
    out_ids = np.zeros(TOKEN_SEQ_LEN, np.int32)
    mask = np.zeros(TOKEN_SEQ_LEN, np.uint8)
    out_ids[:n] = ids
    mask[:n] = 1
    return out_ids, mask, n


def load_vae_for_config(cfg: Config, device: torch.device):
    kwargs = {"torch_dtype": torch.bfloat16, "cache_dir": cfg.cache_dir}
    if cfg.vae_subfolder is not None:
        kwargs["subfolder"] = cfg.vae_subfolder
    vae = AutoencoderKL.from_pretrained(cfg.vae_model, **kwargs).to(device).eval()
    cfg.latent_scale = float(getattr(vae.config, "scaling_factor", cfg.latent_scale))
    shift = getattr(vae.config, "shift_factor", None)
    cfg.latent_shift = cfg.latent_shift if shift is None else float(shift)
    return vae


def iter_tar(path: Path):
    """Stream a tar, pairing {key}.json with its image. We strip only the known
    trailing extension, so keys containing '.' group correctly (unlike wds)."""
    metas: dict[str, Optional[tuple]] = {}
    imgs: dict[str, bytes] = {}
    with tarfile.open(path, "r|*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            data = tf.extractfile(m).read()
            name = m.name
            if name.endswith(".json"):
                key = name[:-5]
                try:
                    meta = json.loads(data)
                    metas[key] = (str(meta["key"]), meta.get("caption_type"),
                                  (meta.get("caption") or "").strip())
                except (json.JSONDecodeError, KeyError, TypeError):
                    metas[key] = None
                if key not in imgs:
                    continue
            else:
                ext = name.rsplit(".", 1)[-1].lower()
                if ext not in IMAGE_EXTS:
                    continue
                key = name[:-(len(ext) + 1)]
                imgs[key] = data
                if key not in metas:
                    continue
            meta, img = metas.pop(key), imgs.pop(key)
            yield (None, None, None, None) if meta is None else (*meta, img)


def _parallel(records, fn, ex: ThreadPoolExecutor, lookahead: int):
    """Map fn over records on a thread pool, preserving order with bounded
    look-ahead so decode overlaps GPU work without buffering a whole tar."""
    buf: deque = deque()
    for rec in records:
        buf.append(ex.submit(fn, rec))
        if len(buf) >= lookahead:
            yield buf.popleft().result()
    while buf:
        yield buf.popleft().result()


def _is_on_cfs(path: Path) -> bool:
    """True if path lives on the (slow, archival) Community File System."""
    rp = str(path.resolve())
    cfs = os.environ.get("CFS")
    if cfs and rp.startswith(str(Path(cfs).resolve())):
        return True
    return rp.startswith("/global/cfs")


def _stage_tar(src: Path, staging_dir: Path) -> tuple[Path, bool]:
    """Copy a CFS tar onto (fast) scratch so the GPU reads locally; tars already
    on scratch are read in place. Returns (path_to_read, is_temp_copy)."""
    if not _is_on_cfs(src):
        return src, False
    staging_dir.mkdir(parents=True, exist_ok=True)
    dst = staging_dir / src.name
    tmp = dst.with_suffix(dst.suffix + f".{os.getpid()}.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)            # atomic publish so a partial copy is never read
    return dst, True


def run(cfg: Config) -> None:
    rank, world, local = dist_env()
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)
    is_main = rank == 0
    out = Path(cfg.output_dir)

    vae = load_vae_for_config(cfg, device)
    try:
        vae.encoder = torch.compile(vae.encoder, mode="reduce-overhead")
    except Exception:
        LOG.warning("rank %d: torch.compile(vae) failed; continuing", rank)

    if is_main:
        ensure_manifests(cfg)
        ensure_tar_manifest(cfg, list_tars(cfg))
    while not ((out / "config.json").exists() and (out / "tars.json").exists()):
        time.sleep(1.0)

    # shards_per_rank is authoritative from the manifest: it fixes each rank's
    # base_row, so a resume with a different --max-images can't relocate shards.
    cfg.shards_per_rank = int(
        json.loads((out / "config.json").read_text())["config"]["shards_per_rank"])
    rows_per_rank = cfg.shards_per_rank * cfg.shard_size
    base_row = rank * rows_per_rank

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model, cache_dir=cfg.cache_dir)
    text_tok = text_model = None
    if cfg.store_text:
        text_tok, text_model = load_text_encoder(cfg.text_model, device, cache_dir=cfg.cache_dir)
    ensure_vocab_table(cfg, device, model=text_model)

    my_tars = load_frozen_tars(cfg)[rank::world]

    prog_path = out / f"progress_rank{rank}.json"
    # Progress is tar-granular, which means that if we restart after a crash in the middle of a tar, we resume from the beginning of that tar. This prevents desync and data misordering.
    tar_pos, local_row = 0, 0
    if prog_path.exists():
        d = json.loads(prog_path.read_text())
        if d["world_size"] != world:
            raise RuntimeError(
                f"rank {rank}: saved world_size={d['world_size']} != {world}; "
                "resume with the original world size or start fresh.")
        tar_pos, local_row = d["tar_pos"], d["local_row"]

    writer = ShardWriter(cfg)
    meta_buf: dict[int, list[dict]] = {}
    meta_seq: dict[int, int] = {}
    fail_f = open(out / f"failures_rank{rank}.jsonl", "a", buffering=1)
    done_dir = out / "tars_done"          # visible ledger of completed tars
    done_dir.mkdir(parents=True, exist_ok=True)

    def fail(key, reason):
        fail_f.write(json.dumps({"key": key, "reason": reason}) + "\n")

    def flush_meta():
        mdir = out / "meta"
        mdir.mkdir(parents=True, exist_ok=True)
        for sid, rows in meta_buf.items():
            if not rows:
                continue
            seq = meta_seq.get(sid)
            if seq is None:
                existing = sorted(mdir.glob(f"meta_{sid:05d}_rank{rank}_*.parquet"))
                seq = int(existing[-1].stem.rsplit("_", 1)[-1]) + 1 if existing else 0
            p = mdir / f"meta_{sid:05d}_rank{rank}_{seq:06d}.parquet"
            tmp = p.with_suffix(".tmp")
            pq.write_table(pa.Table.from_pylist(rows), tmp)
            os.replace(tmp, p)
            meta_seq[sid] = seq + 1
        meta_buf.clear()

    def checkpoint():
        writer.flush()
        flush_meta()
        tmp = prog_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"world_size": world, "tar_pos": tar_pos, "local_row": local_row}))
        os.replace(tmp, prog_path)

    @torch.inference_mode()
    def encode(batch: list[tuple]) -> None:
        nonlocal local_row
        b = len(batch)
        imgs = np.stack([s[3] for s in batch])
        if b < cfg.vae_batch:              # pad to the compiled graph's fixed shape
            imgs = np.concatenate([imgs, np.repeat(imgs[-1:], cfg.vae_batch - b, 0)], 0)
        t = torch.from_numpy(imgs).to(device, non_blocking=True)
        t = t.permute(0, 3, 1, 2).contiguous().to(torch.bfloat16) / 127.5 - 1.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            post = vae.encode(t).latent_dist
        mean = post.mean[:b].to(torch.float16).cpu().numpy()
        logvar = post.logvar[:b].to(torch.float16).cpu().numpy() if cfg.store_logvar else None
        finite = np.isfinite(mean.reshape(b, -1)).all(1)
        if logvar is not None:
            finite &= np.isfinite(logvar.reshape(b, -1)).all(1)
        text = (encode_texts(text_model, text_tok, [s[2] for s in batch],
                             device=device, max_length=MAX_SENTENCE_LENGTH,
                             batch_size=cfg.text_batch) if cfg.store_text else None)
        start_local = local_row
        for j, (key, ctype, caption, _) in enumerate(batch):
            if not finite[j]:              # leave the row unfilled, don't consume one
                fail(key, "nonfinite_latent")
                continue
            grow = base_row + local_row
            ids, mask, n = tokenize(tokenizer, caption)
            writer.write(grow, latent=mean[j], logvar=logvar[j] if logvar is not None else None,
                         text=text[j] if text is not None else None,
                         ctype=CTYPE_TO_IDX[ctype], ids=ids, mask=mask, length=n)
            meta_buf.setdefault(grow // cfg.shard_size, []).append(
                {"global_row": grow, "local_row": grow % cfg.shard_size, # two variables named local_row, but they're different
                 "key": key, "caption": caption, "caption_type": ctype})
            local_row += 1
            if local_row >= rows_per_rank:
                raise RuntimeError(
                    f"rank {rank}: exhausted reserved rows ({rows_per_rank}); "
                    "increase --shards-per-rank.")
        if is_main and cfg.log_every > 0 and start_local // cfg.log_every != local_row // cfg.log_every:
            _log_samples(out, vae, tokenizer, t, post.mean, batch, local_row)

    def decode_one(rec):                   # runs on the decode pool (cv2 frees GIL)
        key, ctype, caption, img_bytes = rec
        if not key:
            return ("fail", key, "bad_record")
        if ctype not in CTYPE_TO_IDX or not caption:
            return ("fail", key, "bad_caption_type" if caption else "empty_caption")
        img = _decode_and_crop(img_bytes, cfg.image_size, cfg.min_resolution)
        if img is None:
            return ("fail", key, "decode_or_too_small")
        return ("ok", (key, ctype, caption, img))

    # decode_pool decodes JPEGs in parallel so the GPU isn't starved while it
    # encodes the current batch. CFS tars are first staged to scratch (below).
    decode_pool = ThreadPoolExecutor(max_workers=cfg.decode_workers)

    # stage_pool copies the next tars from CFS to scratch in the background so the
    # GPU never blocks on a slow CFS read. One worker => ordered, bounded prefetch;
    # each staged copy is deleted right after its tar is processed.
    staging_dir = out / "_staging" / f"rank{rank}"
    shutil.rmtree(staging_dir, ignore_errors=True)   # drop leftovers from a prior crash
    stage_pool = ThreadPoolExecutor(max_workers=1)
    stage_futs: dict[int, Future] = {}

    def _submit_stage(idx: int) -> None:
        if tar_pos <= idx < len(my_tars) and idx not in stage_futs:
            stage_futs[idx] = stage_pool.submit(_stage_tar, my_tars[idx], staging_dir)

    for idx in range(tar_pos, min(tar_pos + STAGE_PREFETCH + 1, len(my_tars))):
        _submit_stage(idx)

    pending: list[tuple] = []
    for i in range(tar_pos, len(my_tars)):
        src = my_tars[i]
        _submit_stage(i + STAGE_PREFETCH + 1)         # keep the prefetch window full
        local, staged = stage_futs.pop(i).result()    # blocks only if staging lags
        try:
            # look-ahead >= one batch so the next batch can decode during GPU encode
            for res in _parallel(iter_tar(local), decode_one, decode_pool,
                                 cfg.vae_batch + cfg.decode_workers):
                if res[0] == "fail":
                    fail(res[1], res[2])
                    continue
                pending.append(res[1])
                if len(pending) >= cfg.vae_batch:
                    assert len(pending) == cfg.vae_batch
                    encode(pending)
                    pending = []
            if pending:
                encode(pending)
                pending = []
        finally:
            if staged:
                local.unlink(missing_ok=True)          # free scratch immediately
        tar_pos = i + 1             # this tar is fully written; persist as resume point
        if tar_pos % cfg.checkpoint_every == 0 or tar_pos == len(my_tars):
            checkpoint()
        (done_dir / f"{src.name}.done").touch()   # informational; progress file is authoritative

    decode_pool.shutdown(wait=True)
    stage_pool.shutdown(wait=True)
    shutil.rmtree(staging_dir, ignore_errors=True)
    checkpoint()
    fail_f.close()
    LOG.info("rank %d: done. tars=%d rows=%d", rank, len(my_tars), local_row)


@torch.inference_mode()
def _log_samples(out: Path, vae, tokenizer, imgs_t, means, batch, count: int, n: int = 4) -> None:
    n = min(n, len(batch))
    decoded = ((vae.decode(means[:n]).sample.float().clamp(-1, 1) + 1) / 2).cpu()
    orig = ((imgs_t[:n].float().clamp(-1, 1) + 1) / 2).cpu()
    grid = make_grid(torch.cat([orig, decoded], dim=0), nrow=n, padding=2)
    sdir = out / "samples"
    sdir.mkdir(parents=True, exist_ok=True)
    tag = f"step_{count:09d}"
    save_image(grid, sdir / f"{tag}.png")
    with open(sdir / f"{tag}.txt", "w") as f:
        for key, ctype, caption, _ in batch[:n]:
            ids, _, length = tokenize(tokenizer, caption)
            f.write(f"key={key} caption_type={ctype}\n  caption: {caption}\n"
                    f"  detok : {tokenizer.decode(ids[:length].tolist())}\n")


# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tars-dir", required=True, help="Dir of GPIC .tar files (e.g. $CFS/.../gpic/train).")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--bridge-preset", choices=["sd", "flux", "both"], default="sd",
        help="Latent VAE/bridge preset to encode. 'both' writes output_dir/sd and output_dir/flux.",
    )
    p.add_argument("--vae-model", default=None, help="Override the preset VAE model for single-preset runs.")
    p.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    p.add_argument("--shard-size", type=int, default=262144)
    p.add_argument("--shards-per-rank", type=int, default=0,
                   help="Per-rank reserved shard range. 0 => derive from --max-images.")
    p.add_argument("--max-images", type=int, default=110_000_000,
                   help="Upper bound used to size the per-rank shard reservation.")
    p.add_argument("--min-resolution", type=int, default=IMAGE_SIZE)
    p.add_argument("--store-logvar", action="store_true", default=True)
    p.add_argument("--no-store-logvar", dest="store_logvar", action="store_false")
    p.add_argument("--global-text-emb", dest="store_text", action="store_true",
                   help="Also store the pooled 4096-d Qwen text embedding (expensive).")
    p.add_argument("--vae-batch", type=int, default=256)
    p.add_argument("--text-batch", type=int, default=256)
    p.add_argument("--decode-workers", type=int, default=12,
                   help="Parallel JPEG decode threads feeding the GPU.")
    p.add_argument("--log-every", type=int, default=6250)
    p.add_argument("--checkpoint-every", type=int, default=1, help="Checkpoint every N tars.")
    p.add_argument("--splits", nargs="*", default=["train"],
                   help="Substrings a tar name must match (e.g. train val test). Empty => all.")
    return p.parse_args()


def config_for_preset(a: argparse.Namespace, preset: str, output_dir: Path, shards_per_rank: int) -> Config:
    runtime = runtime_from_preset(preset)
    store_text = bool(a.store_text)
    vae_model = a.vae_model or runtime.vae_model
    vae_subfolder = runtime.vae_subfolder if a.vae_model is None else None
    return Config(
        tars_dir=a.tars_dir,
        output_dir=str(output_dir),
        cache_dir=a.cache_dir or "",
        shard_size=a.shard_size,
        shards_per_rank=shards_per_rank,
        min_resolution=a.min_resolution,
        store_logvar=a.store_logvar,
        store_text=store_text,
        vae_batch=a.vae_batch,
        text_batch=a.text_batch,
        decode_workers=a.decode_workers,
        log_every=a.log_every,
        checkpoint_every=a.checkpoint_every,
        splits=tuple(a.splits),
        bridge_preset=preset,
        latent_shape=runtime.bridge.bridge_shape,
        vae_model=vae_model,
        vae_subfolder=vae_subfolder,
        vae_kind=runtime.vae_kind,
        latent_scale=runtime.latent_scale,
        latent_shift=runtime.latent_shift,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(processName)s] %(levelname)s %(message)s")
    a = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("encode_gpic.py requires a CUDA GPU.")
    _, world, _ = dist_env()
    spr = a.shards_per_rank or (
        -(-a.max_images // max(world, 1) // a.shard_size) + 2)   # ceil + headroom
    if a.bridge_preset == "both" and a.vae_model is not None:
        raise ValueError("--vae-model is ambiguous with --bridge-preset both.")
    presets = ("sd", "flux") if a.bridge_preset == "both" else (a.bridge_preset,)
    base_out = Path(a.output_dir)
    for preset in presets:
        out = base_out / preset if a.bridge_preset == "both" else base_out
        run(config_for_preset(a, preset, out, spr))


if __name__ == "__main__":
    main()
# APPROVED