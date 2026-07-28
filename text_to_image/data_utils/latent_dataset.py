"""
PyTorch Dataset for the sharded memmap output produced by the GPIC encoder
(``data_utils/encode_gpic.py``).

On-disk layout (per shard, in `root`):
    latents_{shard:05d}.memmap     # (shard_size, 4, 32, 32) fp16
    filled_{shard:05d}.bin         # (shard_size,)           uint8 (0 or 1)
    logvar_{shard:05d}.memmap      # optional, same shape as latents
    text_{shard:05d}.memmap        # optional (shard_size, 4096) fp16 pooled text
    meta/meta_{shard:05d}_rank{R}_{seq:06d}.parquet  # caption sidecars
    config.json                    # written by the encoder; describes shapes/dtypes

Each image has exactly one caption, so there is a single token-embedding kind
("original"). The context-independent token embeddings are gathered at read
time from a shared vocab table (``token_emb_storage == "table"``) or read
directly from per-row memmaps (``"memmap"``), depending on the sidecar config.

The encoder may leave gaps (rows where `filled == 0`). This dataset hides them
by building a global "valid index" once at construction time, mapping each
contiguous external index 0..len-1 to a (shard_id, local_row) pair.
"""

from __future__ import annotations

import json
import mmap
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from token_bridge import (
        PROMPT_KIND_TO_LABEL,
        TokenBridgeConfig,
        bridge_config_from_manifest,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from token_bridge import (  # noqa: E402
        PROMPT_KIND_TO_LABEL,
        TokenBridgeConfig,
        bridge_config_from_manifest,
    )

LEGACY_TEXT_DIM = 4096


@dataclass
class _ShardSpec:
    shard_id: int
    shard_size: int
    latents_path: Path
    text_path: Optional[Path]
    filled_path: Path
    logvar_path: Optional[Path]
    token_paths: dict[str, dict[str, Path]] = field(default_factory=dict)
    # Optional DINOv2 patch-token sidecar (encode_dino_features.py). None when
    # this shard wasn't re-encoded (the subset may cover only some shards).
    dino_path: Optional[Path] = None
    dino_filled_path: Optional[Path] = None
    # Optional global text-embedding sidecar (encode_global_text.py). Same
    # per-shard / per-row optionality as DINO (the --percent prefix may cover
    # only some shards).
    gtext_path: Optional[Path] = None
    gtext_filled_path: Optional[Path] = None


# Each image has a single caption, so there is exactly one token-embedding kind.
_TOKEN_KIND = "original"


def _madvise_random(arr: np.memmap) -> None:
    """Hint the kernel that access to this memmap is random.

    Training reads rows in shuffled order, so the default sequential readahead
    just pollutes the page cache with pages we won't use (and evicts ones we
    will). MADV_RANDOM disables that readahead. Best-effort: silently skipped
    on platforms / objects that don't support it.
    """
    mm = getattr(arr, "_mmap", None)
    if mm is None:
        return
    try:
        mm.madvise(mmap.MADV_RANDOM)
    except (AttributeError, OSError, ValueError):
        pass


class CommonCatalogLatentDataset(Dataset):
    """Random-access dataset over sharded VAE latents + Qwen3 text embeddings.

    Args:
        root: directory containing the memmap shards and config.json.
        scale_latents: if True (default), multiplies the stored raw VAE mean
            by 0.18215 (SD convention) before returning. Set False if your
            training code applies the scaling itself.
        return_logvar: if True, also returns the per-row VAE logvar. Requires
            the encoder to have been run with --store-logvar.
        sample_posterior: if True (default False), draws z = mean + sigma*eps
            instead of returning the deterministic mean. Implies the
            existence of logvar; raises if --store-logvar wasn't set.
        return_caption: if True, loads the caption from the per-shard parquet
            sidecars and returns it alongside the embedding.
        cast_dtype: dtype to cast the returned tensors to. Default fp16
            (zero-copy from disk). Use torch.bfloat16 or torch.float32 if
            your training code expects those.
        indices: optional explicit list of valid external indices to expose.
            Useful for train/val splits. If None, all `filled == 1` rows
            in `root` are exposed in shard order.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        *,
        scale_latents: bool = True,
        scale_tokens: bool = True,
        token_scale: Optional[float] = None,
        return_logvar: bool = False,
        sample_posterior: bool = False,
        return_caption: bool = False,
        cast_dtype: torch.dtype = torch.bfloat16,
        indices: Optional[Sequence[int]] = None,
        token_pad_id: Optional[int] = None,
        config: Optional[TokenBridgeConfig] = None,
        bridge_preset: str = "auto",
        latent_scale: Optional[float] = None,
        latent_shift: Optional[float] = None,
        dino_dir: Optional[str | os.PathLike] = None,
        dino_shape: Optional[tuple[int, int]] = None,
        gtext_dir: Optional[str | os.PathLike] = None,
    ):
        self.root = Path(root)
        self.scale_latents = scale_latents
        self.scale_tokens = scale_tokens
        self.token_pad_id = int(token_pad_id) if token_pad_id is not None else None
        self.return_logvar = return_logvar or sample_posterior
        self.sample_posterior = sample_posterior
        self.return_caption = return_caption
        self.cast_dtype = cast_dtype

        # Token embeddings live under the dataset root (the GPIC encoder writes
        # token_original/*, token_embed_config.json, and token_vocab_table.npy
        # there). Storage is "table" (gather a shared vocab table) or "memmap"
        # (dense per-row), per the sidecar config.
        self._token_storage = "memmap"
        self._token_table: Optional[np.ndarray] = None
        self.dino_dir: Optional[Path] = (
            Path(dino_dir) if dino_dir is not None else None
        )
        self.gtext_dir: Optional[Path] = (
            Path(gtext_dir) if gtext_dir is not None else None
        )

        cfg_path = self.root / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing {cfg_path}")
        cfg = json.loads(cfg_path.read_text())["config"]
        self._shard_size = int(cfg["shard_size"])
        self._store_logvar_on_disk = bool(cfg.get("store_logvar", False))
        runtime = bridge_config_from_manifest(self.root, preset=bridge_preset)
        self.bridge_config = config or runtime.bridge
        self.bridge_config.validate()
        if tuple(runtime.bridge.bridge_shape) != tuple(self.bridge_config.bridge_shape):
            raise ValueError(
                f"{self.root} latent shape {runtime.bridge.bridge_shape} does not "
                f"match requested {self.bridge_config.bridge_shape}."
            )
        self.latent_shape = self.bridge_config.bridge_shape
        self.latent_scale = runtime.latent_scale if latent_scale is None else float(latent_scale)
        self.latent_shift = runtime.latent_shift if latent_shift is None else float(latent_shift)
        self.text_dim = int(cfg.get("text_dim", LEGACY_TEXT_DIM))
        self.token_seq_len = self.bridge_config.token_seq_len
        self.token_emb_dim = self.bridge_config.token_emb_dim
        self.token_flat_dim = self.bridge_config.token_flat_dim
        self.token_scale = (
            self.token_emb_dim ** 0.5 if token_scale is None else token_scale
        )

        self.validate_token_config()

        # DINO sidecar shapes. When dino_dir is set we read them from the
        # sidecar's config; when only dino_shape is passed (e.g. an extra
        # dataset with no DINO files) we still emit zero-filled placeholders so
        # ConcatDataset batches stay uniform. Neither set => DINO disabled.
        self._dino_enabled = False
        self._dino_tokens = self._dino_tdim = 0
        if self.dino_dir is not None:
            self.validate_dino_dir_compatibility()
            dcfg = json.loads(
                (self.dino_dir / "dino_config.json").read_text()
            )["config"]
            self._dino_tokens = int(dcfg["num_tokens"])
            self._dino_tdim = int(dcfg["token_dim"])
            self._dino_enabled = True
        elif dino_shape is not None:
            self._dino_tokens, self._dino_tdim = int(dino_shape[0]), int(dino_shape[1])
            self._dino_enabled = True

        # Global text-embedding sidecar. When set, its per-row pooled vector is
        # exposed as `text_emb` (the text REPA target), with a presence flag for
        # rows the sidecar's --percent prefix didn't cover.
        self._gtext_enabled = False
        self._gtext_dim = 0
        if self.gtext_dir is not None:
            self.validate_gtext_dir_compatibility()
            gcfg = json.loads(
                (self.gtext_dir / "gtext_config.json").read_text()
            )["config"]
            self._gtext_dim = int(gcfg["flat_dim"])
            self._gtext_enabled = True
        # TODO: may need else branch later

        if self.return_logvar and not self._store_logvar_on_disk:
            raise ValueError(
                "Asked for logvar / posterior sampling, but the encoder was "
                "run without --store-logvar."
            )

        self._shards: dict[int, _ShardSpec] = self._discover_shards()
        if not self._shards:
            raise RuntimeError(f"No shards found in {self.root}")

        self._has_text = any(s.text_path is not None for s in self._shards.values())

        # Build the global valid index ONCE. Reading every `filled` bitmap is
        # cheap (a few hundred KB per shard) and saves us from per-row I/O.
        self._index = self._build_valid_index(indices)

        # Lazy per-worker handles. We keep these as None at construction
        # time and open them inside the worker process; multiprocessing
        # forking + open file descriptors is a known footgun.
        self._handles: dict[int, dict[str, np.memmap]] = {}
        self._captions_cache: dict[int, dict[int, str]] = {}


    ###
    # Sanity checks
    ###

    def validate_token_config(self) -> None:
        cfg_path = self.root / "token_embed_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing {cfg_path}")
        cfg = json.loads(cfg_path.read_text())["config"]
        if int(cfg["shard_size"]) != self._shard_size:
            raise ValueError(
                f"token_embed_config shard_size={cfg['shard_size']} does not match "
                f"dataset shard_size={self._shard_size}."
            )
        self._stored_token_seq_len = int(cfg.get("token_seq_len", self.token_seq_len))
        self._stored_token_emb_dim = int(cfg.get("token_emb_dim", self.token_emb_dim))
        self._stored_mrl_dim = int(cfg.get("mrl_dim", self._stored_token_emb_dim))
        if self._stored_token_seq_len < self.token_seq_len:
            raise ValueError(
                f"{cfg_path} token_seq_len={self._stored_token_seq_len} is "
                f"shorter than requested {self.token_seq_len}."
            )
        self._token_storage = cfg.get("token_emb_storage", "memmap")
        if self._token_storage == "table":
            table_path = self.root / cfg.get("vocab_table", "token_vocab_table.npy")
            if not table_path.exists():
                raise FileNotFoundError(f"Missing vocab table {table_path}")
            if self._stored_mrl_dim < self.token_emb_dim:
                raise ValueError(f"mrl_dim < {self.token_emb_dim} in {cfg_path}")
        elif self._token_storage == "memmap":
            if self._stored_token_emb_dim < self.token_emb_dim:
                raise ValueError(
                    f"token_emb_dim < {self.token_emb_dim} in {cfg_path}"
                )
        elif self._token_storage != "memmap":
            raise ValueError(f"Unknown token_emb_storage {self._token_storage!r}")

    def validate_dino_dir_compatibility(self) -> None:
        if self.dino_dir is None:
            return
        cfg_path = self.dino_dir / "dino_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing {cfg_path}")
        cfg = json.loads(cfg_path.read_text())["config"]
        dino_source = Path(cfg["source_root"]).resolve()
        if dino_source != self.root.resolve():
            print(
                f"dino_dir was generated from {dino_source}, "
                f"but dataset root is {self.root.resolve()}."
            )
        if int(cfg["shard_size"]) != self._shard_size:
            raise ValueError(
                f"dino_dir shard_size={cfg['shard_size']} does not match "
                f"dataset shard_size={self._shard_size}."
            )

    def validate_gtext_dir_compatibility(self) -> None:
        if self.gtext_dir is None:
            return
        cfg_path = self.gtext_dir / "gtext_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing {cfg_path}")
        cfg = json.loads(cfg_path.read_text())["config"]
        gtext_source = Path(cfg["source_root"]).resolve()
        if gtext_source != self.root.resolve():
            print(
                f"gtext_dir was generated from {gtext_source}, "
                f"but dataset root is {self.root.resolve()}."
            )
        if int(cfg["shard_size"]) != self._shard_size:
            raise ValueError(
                f"gtext_dir shard_size={cfg['shard_size']} does not match "
                f"dataset shard_size={self._shard_size}."
            )

    # ------------------------------------------------------------------
    # discovery & indexing
    # ------------------------------------------------------------------

    def _discover_shards(self) -> dict[int, _ShardSpec]:
        shards: dict[int, _ShardSpec] = {}
        for p in sorted(self.root.glob("latents_*.memmap")):
            shard_id = int(p.stem.split("_")[1])
            text = self.root / f"text_{shard_id:05d}.memmap"
            filled = self.root / f"filled_{shard_id:05d}.bin"
            logvar = self.root / f"logvar_{shard_id:05d}.memmap"
            # `text` (global pooled embedding) is optional: token-bridge datasets
            # (e.g. GPIC) skip it. A shard is valid as long as it has a filled map.
            if not filled.exists():
                continue

            token_paths = self._token_paths_for_shard(shard_id)
            dino_path = dino_filled_path = None
            if self.dino_dir is not None:
                dp = self.dino_dir / f"dino_{shard_id:05d}.memmap"
                fp = self.dino_dir / f"dino_filled_{shard_id:05d}.memmap"
                if dp.exists() and fp.exists():
                    dino_path, dino_filled_path = dp, fp

            gtext_path = gtext_filled_path = None
            if self.gtext_dir is not None:
                gp = self.gtext_dir / f"gtext_{shard_id:05d}.memmap"
                gfp = self.gtext_dir / f"gtext_filled_{shard_id:05d}.memmap"
                if gp.exists() and gfp.exists():
                    gtext_path, gtext_filled_path = gp, gfp

            shards[shard_id] = _ShardSpec(
                shard_id=shard_id,
                shard_size=self._shard_size,
                latents_path=p,
                text_path=text if text.exists() else None,
                filled_path=filled,
                logvar_path=logvar if logvar.exists() else None,
                token_paths=token_paths,
                dino_path=dino_path,
                dino_filled_path=dino_filled_path,
                gtext_path=gtext_path,
                gtext_filled_path=gtext_filled_path,
            )
        return shards

    def _token_paths_for_shard(self, shard_id: int) -> dict[str, dict[str, Path]]:
        kind = _TOKEN_KIND
        subdir = self.root / f"token_{kind}"
        paths = {
            "emb": subdir / f"token_{kind}_{shard_id:05d}.memmap",
            "token_ids": subdir / f"token_{kind}_token_ids_{shard_id:05d}.memmap",
            "mask": subdir / f"token_{kind}_mask_{shard_id:05d}.memmap",
            "lengths": subdir / f"token_{kind}_lengths_{shard_id:05d}.memmap",
            "filled": subdir / f"token_{kind}_filled_{shard_id:05d}.memmap",
        }
        # In table mode there is no per-row emb memmap; ids/mask/lengths/filled
        # are enough (the embedding is gathered from the shared vocab table).
        required = [k for k in paths if not (self._token_storage == "table" and k == "emb")]
        if all(paths[k].exists() for k in required):
            return {kind: paths}
        return {}

    def _build_valid_index(
        self, indices: Optional[Sequence[int]]
    ) -> np.ndarray:
        """Returns an (N, 2) int64 array of (shard_id, local_row) pairs."""
        rows = []
        for shard_id in sorted(self._shards):
            spec = self._shards[shard_id]
            filled = np.memmap(
                spec.filled_path, dtype=np.uint8, mode="r",
                shape=(spec.shard_size,),
            )
            local = np.flatnonzero(filled)
            shard_col = np.full_like(local, shard_id)
            rows.append(np.stack([shard_col, local], axis=1))
            del filled  # release mmap
        full = np.concatenate(rows, axis=0).astype(np.int64)
        if indices is not None:
            full = full[np.asarray(indices, dtype=np.int64)]
        return full

    # ------------------------------------------------------------------
    # per-worker lazy open
    # ------------------------------------------------------------------

    def _get_handles(self, shard_id: int) -> dict[str, np.memmap]:
        h = self._handles.get(shard_id)
        if h is not None:
            return h
        spec = self._shards[shard_id]
        h = {
            "latents": np.memmap(
                spec.latents_path, dtype=np.float16, mode="r",
                shape=(spec.shard_size, *self.latent_shape),
            ),
        }
        if spec.text_path is not None:
            raise ValueError("text_original is not supported yet")
            h["text"] = np.memmap(
                spec.text_path, dtype=np.float16, mode="r",
                shape=(spec.shard_size, self.text_dim),
            )
        if self.return_logvar and spec.logvar_path is not None:
            h["logvar"] = np.memmap(
                spec.logvar_path, dtype=np.float16, mode="r",
                shape=(spec.shard_size, *self.latent_shape),
            )
        for kind, paths in spec.token_paths.items():
            assert kind == "original"
            if self._token_storage != "table":
                h[f"token_{kind}"] = np.memmap(
                    paths["emb"], dtype=np.float16, mode="r",
                    shape=(spec.shard_size, self._stored_token_seq_len * self._stored_token_emb_dim), # stored as opposed to actual for flexibility
                )
            h[f"token_{kind}_ids"] = np.memmap(
                paths["token_ids"], dtype=np.int32, mode="r",
                shape=(spec.shard_size, self._stored_token_seq_len),
            )
            h[f"token_{kind}_mask"] = np.memmap(
                paths["mask"], dtype=np.uint8, mode="r",
                shape=(spec.shard_size, self._stored_token_seq_len),
            )
            h[f"token_{kind}_lengths"] = np.memmap(
                paths["lengths"], dtype=np.uint16, mode="r",
                shape=(spec.shard_size,),
            )
            h[f"token_{kind}_filled"] = np.memmap(
                paths["filled"], dtype=np.uint8, mode="r",
                shape=(spec.shard_size,),
            )
        if self._dino_enabled and spec.dino_path is not None:
            h["dino"] = np.memmap(
                spec.dino_path, dtype=np.float16, mode="r",
                shape=(spec.shard_size, self._dino_tokens * self._dino_tdim),
            )
            h["dino_filled"] = np.memmap(
                spec.dino_filled_path, dtype=np.uint8, mode="r",
                shape=(spec.shard_size,),
            )
        if self._gtext_enabled and spec.gtext_path is not None:
            h["gtext"] = np.memmap(
                spec.gtext_path, dtype=np.float16, mode="r",
                shape=(spec.shard_size, self._gtext_dim),
            )
            h["gtext_filled"] = np.memmap(
                spec.gtext_filled_path, dtype=np.uint8, mode="r",
                shape=(spec.shard_size,),
            )
        # Suppress wasteful sequential readahead on every mapping (access is
        # row-shuffled). Cheap and best-effort.
        for arr in h.values():
            _madvise_random(arr)
        self._handles[shard_id] = h
        return h

    def _get_token_table(self) -> np.ndarray:
        if self._token_table is None:
            cfg = json.loads(
                (self.root / "token_embed_config.json").read_text()
            )["config"]
            path = self.root / cfg.get("vocab_table", "token_vocab_table.npy")
            self._token_table = np.load(path, mmap_mode="r")
        return self._token_table

    def _get_caption(self, shard_id: int, local_row: int) -> str:
        cache = self._captions_cache.get(shard_id)
        if cache is None:
            import pyarrow.parquet as pq
            cache = {}
            paths: list[Path] = []
            paths += sorted((self.root / "meta").glob(
                f"meta_{shard_id:05d}_rank*.parquet"))
            for p in paths:
                tbl = pq.read_table(p, columns=["local_row", "caption"])
                for lr, cap in zip(
                    tbl.column("local_row").to_pylist(),
                    tbl.column("caption").to_pylist(),
                ):
                    if int(lr) in cache:
                        assert cache[int(lr)] == cap # sanity check on our script
                    cache[int(lr)] = cap
            self._captions_cache[shard_id] = cache
        return cache.get(int(local_row), "")

    # ------------------------------------------------------------------
    # Per-worker state management
    # ------------------------------------------------------------------

    def _reset_worker_state(self, worker_seed: Optional[int] = None) -> None:
        """Reset per-worker memmap handles + caption cache.

        Call this from ``worker_init_fn``. ``worker_seed`` is accepted for
        call-site compatibility but unused (no per-row randomness remains).
        """
        self._handles = {}
        self._captions_cache = {}

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._index.shape[0])

    def _token_kind_available(
        self,
        handles: dict[str, np.memmap],
        local_row: int,
        kind: str,
    ) -> bool:
        assert kind == "original"
        filled = handles.get(f"token_{kind}_filled")
        if filled is None or int(filled[local_row]) != 1:
            return False
        if self._token_storage == "table":
            return handles.get(f"token_{kind}_ids") is not None
        return handles.get(f"token_{kind}") is not None

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        shard_id, local_row = self._index[idx]
        shard_id = int(shard_id)
        local_row = int(local_row)
        h = self._get_handles(shard_id)

        # np.array(...) copies the row out of the mmap into RAM. Without
        # this, the returned tensor would be a view into the mmap, which
        # the DataLoader's pin_memory thread can't pin.
        latent_np = np.array(h["latents"][local_row], copy=True)
        latent = torch.from_numpy(latent_np)

        has_text = "text" in h
        text_original = (
            torch.from_numpy(np.array(h["text"][local_row], copy=True))
            if has_text else None
        )

        if self.return_logvar:
            logvar_np = np.array(h["logvar"][local_row], copy=True)
            logvar = torch.from_numpy(logvar_np)
        else:
            logvar = None

        if self.sample_posterior:
            std = torch.exp(0.5 * logvar.float()).to(latent.dtype)
            latent = latent + std * torch.randn_like(latent)

        if self.scale_latents:
            latent = (latent.to(torch.float32) - self.latent_shift) * self.latent_scale
            latent = latent.to(self.cast_dtype)
        else:
            latent = latent.to(self.cast_dtype)

        if has_text:
            text_original = text_original.to(self.cast_dtype)

        out: dict[str, torch.Tensor | str] = {"latent": latent}

        chosen = _TOKEN_KIND
        out["text_emb_kind"] = chosen
        if has_text:
            raise ValueError("text_original is not supported yet")
            out["text_emb"] = text_original

        if not self._token_kind_available(h, local_row, chosen):
            raise RuntimeError(
                f"Missing token embedding for kind={chosen!r}, "
                f"shard={shard_id}, local_row={local_row}."
            )
        token_ids_np = np.array(
            h[f"token_{chosen}_ids"][local_row][: self.token_seq_len], copy=True
        )
        token_mask_np = np.array(
            h[f"token_{chosen}_mask"][local_row][: self.token_seq_len], copy=True
        )
        token_length = min(int(h[f"token_{chosen}_lengths"][local_row]), self.token_seq_len)

        if self._token_storage == "table":
            # Gather raw rows, truncate to runtime dims, L2-normalize (MRL),
            # zero out padding positions (stop-detection relies on this).
            table = self._get_token_table()
            emb = np.asarray(
                table[token_ids_np.astype(np.int64)][:, : self.token_emb_dim],
                dtype=np.float32,
            )
            emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb *= token_mask_np[:, None]
            token_emb_np = emb.reshape(self.token_flat_dim)
        else:
            token_emb_np = np.array(h[f"token_{chosen}"][local_row], copy=True)
            token_emb_np = token_emb_np.reshape(
                self._stored_token_seq_len, self._stored_token_emb_dim
            )[: self.token_seq_len, : self.token_emb_dim].reshape(self.token_flat_dim)

        out["text_token_emb"] = torch.from_numpy(token_emb_np).to(self.cast_dtype)
        out["text_token_ids"] = torch.from_numpy(token_ids_np).to(torch.long)
        out["text_token_mask"] = torch.from_numpy(token_mask_np).to(torch.bool)
        # On disk, padding positions have token id 0 (zero-init leftover),
        # which collides with a real token. Remap them to the tokenizer's
        # true pad id so the discrete targets/stop-detection are honest.
        # Only the discrete ids are touched; the padding *embeddings* must
        # stay zero (the magnitude-based stop detection
        # and diffusion data scales rely on that).
        if self.token_pad_id is not None:
            assert out["text_token_mask"][0] == 1
            out["text_token_ids"] = out["text_token_ids"].masked_fill(
                ~out["text_token_mask"], self.token_pad_id
            )
        out["text_token_length"] = torch.tensor(token_length, dtype=torch.long)
        out["prompt_kind_label"] = torch.tensor(
            PROMPT_KIND_TO_LABEL[chosen],
            dtype=torch.long,
        )
        if self.scale_tokens:
            out["text_token_emb"] = out["text_token_emb"] * self.token_scale

        # DINOv2 patch-token REPA target. Always emit a (tokens, tdim) tensor +
        # presence flag when DINO is enabled (zeros when this shard/row wasn't
        # re-encoded) so ConcatDataset batches collate uniformly.
        if self._dino_enabled:
            dino_h = h.get("dino")
            dino_filled = h.get("dino_filled")
            if (
                dino_h is not None and dino_filled is not None
                and int(dino_filled[local_row]) == 1
            ):
                dino_np = np.array(dino_h[local_row], copy=True).reshape(
                    self._dino_tokens, self._dino_tdim
                )
                out["dino_emb"] = torch.from_numpy(dino_np).to(self.cast_dtype)
                out["dino_present"] = torch.tensor(True)
            else:
                out["dino_emb"] = torch.zeros(
                    (self._dino_tokens, self._dino_tdim), dtype=self.cast_dtype
                )
                out["dino_present"] = torch.tensor(False)

        # Global text-embedding REPA target. Like DINO, emit a flat (gtext_dim,)
        # tensor + presence flag (zeros when this shard/row wasn't covered by
        # the sidecar's --percent prefix) so ConcatDataset batches stay uniform.
        if self._gtext_enabled:
            gtext_h = h.get("gtext")
            gtext_filled = h.get("gtext_filled")
            if (
                gtext_h is not None and gtext_filled is not None
                and int(gtext_filled[local_row]) == 1
            ):
                gtext_np = np.array(gtext_h[local_row], copy=True)
                out["text_emb"] = torch.from_numpy(gtext_np).to(self.cast_dtype)
                out["text_emb_present"] = torch.tensor(True)
            else:
                out["text_emb"] = torch.zeros(self._gtext_dim, dtype=self.cast_dtype)
                out["text_emb_present"] = torch.tensor(False)

        if self.return_caption:
            out["caption"] = self._get_caption(shard_id, local_row)
        return out


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def random_split_indices(
    n: int, val_fraction: float = 0.005, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) into a CommonCatalogLatentDataset of size n."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    return perm[n_val:], perm[:n_val]


def make_dataloader(
    dataset: CommonCatalogLatentDataset,
    *,
    batch_size: int,
    num_workers: int = 8,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """Convenience wrapper with the recommended worker_init_fn."""

    def _worker_init_fn(worker_id: int) -> None:
        # Force each worker to lazily open its OWN memmap handles. Without
        # this, on `fork` start methods, all workers share the parent's
        # memmap fds and the OS page cache behavior gets weird.
        info = torch.utils.data.get_worker_info()
        ds: CommonCatalogLatentDataset = info.dataset  # type: ignore[assignment]
        ds._reset_worker_state(int(info.seed))

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init_fn,
    )

# APPROVED