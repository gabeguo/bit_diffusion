from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_TO_ID = {"train": 0, "test": 1}


class StateFatePairDataset(Dataset):
    """Clone-paired endpoint dataset for vector bit_diffusion training."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: Literal["train", "test"] = "train",
        mmap: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.config = json.loads((self.root / "config.json").read_text())
        self.latents = np.load(
            self.root / "latents.npy",
            mmap_mode="r" if mmap else None,
        )
        pairs = np.load(self.root / "pairs.npz")
        self.early_idx = pairs["early_idx"].astype(np.int64)
        self.late_idx = pairs["late_idx"].astype(np.int64)
        self.clone_id = pairs["clone_id"].astype(np.int64)
        self.context_label = pairs["context_label"].astype(np.int64)
        self.fate_label = pairs["fate_label"].astype(np.int64)
        self.pair_split = pairs["split"].astype(np.int64)

        split_id = SPLIT_TO_ID[split]
        self._indices = np.flatnonzero(self.pair_split == split_id).astype(np.int64)
        if len(self._indices) == 0:
            raise RuntimeError(f"No pairs found for split={split!r} in {self.root}")

    @property
    def x_dim(self) -> int:
        return int(self.latents.shape[1])

    @property
    def num_context_classes(self) -> int:
        return max(1, len(self.config.get("context_map", {"none": 0})))

    @property
    def num_fate_classes(self) -> int:
        return max(1, len(self.config.get("fate_map", {"unknown": 0})))

    def __len__(self) -> int:
        return int(len(self._indices))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair_idx = int(self._indices[idx])
        early = int(self.early_idx[pair_idx])
        late = int(self.late_idx[pair_idx])

        x_0 = torch.from_numpy(
            np.array(self.latents[early], dtype=np.float32, copy=True)
        )
        x_1 = torch.from_numpy(
            np.array(self.latents[late], dtype=np.float32, copy=True)
        )
        return {
            "x_0": x_0,
            "x_1": x_1,
            "y": torch.tensor(int(self.context_label[pair_idx]), dtype=torch.long),
            "fate_label": torch.tensor(
                int(self.fate_label[pair_idx]), dtype=torch.long
            ),
            "clone_id": torch.tensor(int(self.clone_id[pair_idx]), dtype=torch.long),
            "pair_idx": torch.tensor(pair_idx, dtype=torch.long),
            "early_idx": torch.tensor(early, dtype=torch.long),
            "late_idx": torch.tensor(late, dtype=torch.long),
        }
