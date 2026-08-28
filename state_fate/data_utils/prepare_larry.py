"""Prepare clone-paired endpoints from the LARRY lineage-tracing data.

The output directory contains:
    latents.npy            compact cell latent matrix, shape (cells, latent_dim)
    pairs.npz              endpoint pair indices and split labels
    cell_metadata.csv.gz   compact metadata aligned with latents.npy rows
    config.json            preprocessing metadata and label maps
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from state_fate.data_utils.download_larry import LARRY_FILES
except ModuleNotFoundError:
    from download_larry import LARRY_FILES

pd = None
scipy_io = None
sparse = None
TruncatedSVD = None


def _load_preprocess_deps() -> None:
    global pd, scipy_io, sparse, TruncatedSVD
    try:
        import pandas as _pd
        from scipy import io as _scipy_io
        from scipy import sparse as _sparse
        from sklearn.decomposition import TruncatedSVD as _TruncatedSVD
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing preprocessing dependency. Install with:\n"
            "  pip install -r state_fate/requirements.txt"
        ) from exc
    pd = _pd
    scipy_io = _scipy_io
    sparse = _sparse
    TruncatedSVD = _TruncatedSVD


TIME_COLUMNS = ("Time point", "Timepoint", "time point", "time", "day", "Day")
FATE_COLUMNS = (
    "Cell type annotation",
    "Cell type",
    "cell type annotation",
    "cell_type",
    "annotation",
)
CONTEXT_AUTO_COLUMNS = (
    "Condition",
    "condition",
    "Cytokine",
    "cytokine",
    "Treatment",
    "treatment",
    "Well",
    "well",
    "Starting population",
    "starting population",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default="in_vitro",
        choices=sorted(LARRY_FILES),
        help="Downloaded LARRY dataset to prepare.",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="state_fate/data/raw",
        help="Root containing downloader-created per-dataset raw folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Processed output directory.",
    )
    parser.add_argument("--early-day", type=float, default=2.0)
    parser.add_argument("--late-day", type=float, default=6.0)
    parser.add_argument("--n-hvgs", type=int, default=2000)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument(
        "--log1p-scale",
        type=float,
        default=1e4,
        help="Scale normalized counts before log1p. Use 0 to disable log1p.",
    )
    parser.add_argument(
        "--pairs-per-clone",
        type=int,
        default=32,
        help="Random endpoint pairs sampled per eligible clone. Use 0 for all pairs up to --max-pairs-per-clone.",
    )
    parser.add_argument(
        "--max-pairs-per-clone",
        type=int,
        default=256,
        help="Cap used only when --pairs-per-clone 0.",
    )
    parser.add_argument("--min-early-cells", type=int, default=1)
    parser.add_argument("--min-late-cells", type=int, default=1)
    parser.add_argument(
        "--context-column",
        type=str,
        default="auto",
        help="Metadata column used as y context. Use 'none' to disable.",
    )
    parser.add_argument(
        "--fate-column",
        type=str,
        default="auto",
        help="Metadata column used as terminal fate label. Use 'none' to disable.",
    )
    parser.add_argument(
        "--allow-multi-clone",
        action="store_true",
        help="Keep cells assigned to multiple clone columns by taking the first clone id.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--reserve-fraction",
        type=float,
        default=0.1,
        help="Clone fraction excluded from training and evaluation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _filename_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def _raw_path(raw_root: Path, dataset: str, kind: str) -> Path:
    path = raw_root / dataset / _filename_from_url(LARRY_FILES[dataset][kind])
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run:\n"
            f"  PYTHONPATH=. python state_fate/data_utils/download_larry.py "
            f"--dataset {dataset} --raw-root {raw_root}"
        )
    return path


def _read_mtx_gz(path: Path) -> sparse.csr_matrix:
    with gzip.open(path, "rb") as f:
        matrix = scipy_io.mmread(f)
    return matrix.tocsr().astype(np.float32)


def _read_gene_names(path: Path) -> list[str]:
    with gzip.open(path, "rt") as f:
        return [line.strip() for line in f if line.strip()]


def _read_metadata(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, sep="\t")
    if df.shape[1] == 1:
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool,
    purpose: str,
) -> str | None:
    lower_to_original = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        match = lower_to_original.get(candidate.lower())
        if match is not None:
            return match
    if required:
        raise KeyError(
            f"Could not find {purpose} column. Tried {tuple(candidates)}. "
            f"Available columns: {tuple(df.columns)}"
        )
    return None


def _parse_time_values(values: pd.Series) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float32)
    pattern = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
    for i, value in enumerate(values.astype(str)):
        match = pattern.search(value)
        if match:
            out[i] = float(match.group(0))
    return out


def _clone_ids_from_matrix(
    clone_matrix: sparse.csr_matrix,
    *,
    allow_multi_clone: bool,
) -> tuple[np.ndarray, np.ndarray]:
    clone_matrix = clone_matrix.tocsr()
    n_cells = clone_matrix.shape[0]
    nnz = np.diff(clone_matrix.indptr)
    clone_ids = np.full(n_cells, -1, dtype=np.int64)
    for row in np.flatnonzero(nnz > 0):
        if nnz[row] == 1 or allow_multi_clone:
            clone_ids[row] = int(clone_matrix.indices[clone_matrix.indptr[row]])
    keep = clone_ids >= 0
    return clone_ids, keep


def _encode_strings(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    as_str = np.asarray([str(v) if pd.notna(v) else "unknown" for v in values])
    categories = sorted(set(as_str.tolist()))
    mapping = {name: i for i, name in enumerate(categories)}
    labels = np.asarray([mapping[v] for v in as_str], dtype=np.int64)
    return labels, mapping


def _split_clones(
    clone_ids: np.ndarray,
    *,
    test_fraction: float,
    reserve_fraction: float,
    seed: int,
) -> dict[int, int]:
    clones = np.asarray(sorted(set(int(c) for c in clone_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(clones)
    n = len(clones)
    n_reserve = int(round(n * reserve_fraction))
    n_test = int(round(n * test_fraction))
    if n >= 3:
        if reserve_fraction > 0:
            n_reserve = max(1, n_reserve)
        if test_fraction > 0:
            n_test = max(1, n_test)
    n_reserve = min(n_reserve, max(0, n - 1))
    n_test = min(n_test, max(0, n - n_reserve - 1))

    split_by_clone: dict[int, int] = {}
    for clone in clones[:n_reserve]:
        split_by_clone[int(clone)] = 2
    for clone in clones[n_reserve : n_reserve + n_test]:
        split_by_clone[int(clone)] = 1
    for clone in clones[n_reserve + n_test :]:
        split_by_clone[int(clone)] = 0
    return split_by_clone


def _sample_pairs_for_clone(
    early_idx: np.ndarray,
    late_idx: np.ndarray,
    *,
    pairs_per_clone: int,
    max_pairs_per_clone: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if pairs_per_clone == 0:
        all_pairs = np.stack(np.meshgrid(early_idx, late_idx, indexing="ij"), axis=-1)
        all_pairs = all_pairs.reshape(-1, 2)
        if len(all_pairs) > max_pairs_per_clone:
            chosen = rng.choice(len(all_pairs), size=max_pairs_per_clone, replace=False)
            all_pairs = all_pairs[chosen]
        return all_pairs[:, 0], all_pairs[:, 1]

    replace_early = len(early_idx) < pairs_per_clone
    replace_late = len(late_idx) < pairs_per_clone
    return (
        rng.choice(early_idx, size=pairs_per_clone, replace=replace_early),
        rng.choice(late_idx, size=pairs_per_clone, replace=replace_late),
    )


def main() -> None:
    args = parse_args()
    _load_preprocess_deps()
    if args.pairs_per_clone < 0:
        raise ValueError("--pairs-per-clone must be non-negative")
    if not (0 <= args.test_fraction < 1) or not (0 <= args.reserve_fraction < 1):
        raise ValueError("split fractions must be in [0, 1)")
    if args.test_fraction + args.reserve_fraction >= 1:
        raise ValueError("--test-fraction + --reserve-fraction must be < 1")

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    counts_path = _raw_path(raw_root, args.dataset, "counts")
    genes_path = _raw_path(raw_root, args.dataset, "genes")
    metadata_path = _raw_path(raw_root, args.dataset, "metadata")
    clones_path = _raw_path(raw_root, args.dataset, "clones")

    print(f"Reading counts: {counts_path}")
    counts = _read_mtx_gz(counts_path)
    genes = _read_gene_names(genes_path)
    metadata = _read_metadata(metadata_path)
    clone_matrix = _read_mtx_gz(clones_path)

    if counts.shape[0] != len(metadata):
        raise ValueError(
            f"counts rows {counts.shape[0]} != metadata rows {len(metadata)}"
        )
    if counts.shape[1] != len(genes):
        raise ValueError(f"counts cols {counts.shape[1]} != gene names {len(genes)}")
    if clone_matrix.shape[0] != counts.shape[0]:
        raise ValueError(
            f"clone rows {clone_matrix.shape[0]} != counts rows {counts.shape[0]}"
        )

    time_col = _find_column(metadata, TIME_COLUMNS, required=True, purpose="time")
    time_values = _parse_time_values(metadata[time_col])

    if args.fate_column == "none":
        fate_values = np.full(len(metadata), "unknown")
        fate_col = None
    elif args.fate_column == "auto":
        fate_col = _find_column(metadata, FATE_COLUMNS, required=False, purpose="fate")
        fate_values = (
            metadata[fate_col].fillna("unknown").astype(str).to_numpy()
            if fate_col is not None
            else np.full(len(metadata), "unknown")
        )
    else:
        if args.fate_column not in metadata.columns:
            raise KeyError(f"--fate-column {args.fate_column!r} is not in metadata")
        fate_col = args.fate_column
        fate_values = metadata[fate_col].fillna("unknown").astype(str).to_numpy()

    if args.context_column == "none":
        context_values = np.full(len(metadata), "none")
        context_col = None
    elif args.context_column == "auto":
        context_col = _find_column(
            metadata, CONTEXT_AUTO_COLUMNS, required=False, purpose="context"
        )
        context_values = (
            metadata[context_col].fillna("none").astype(str).to_numpy()
            if context_col is not None
            else np.full(len(metadata), "none")
        )
    else:
        if args.context_column not in metadata.columns:
            raise KeyError(
                f"--context-column {args.context_column!r} is not in metadata"
            )
        context_col = args.context_column
        context_values = metadata[context_col].fillna("none").astype(str).to_numpy()

    clone_ids, clone_keep = _clone_ids_from_matrix(
        clone_matrix,
        allow_multi_clone=args.allow_multi_clone,
    )
    early_mask = np.isclose(time_values, args.early_day)
    late_mask = np.isclose(time_values, args.late_day)
    endpoint_mask = clone_keep & (early_mask | late_mask)
    if endpoint_mask.sum() == 0:
        raise RuntimeError(
            f"No clone-assigned cells found at days {args.early_day} and {args.late_day}."
        )

    raw_indices = np.flatnonzero(endpoint_mask)
    compact_index = np.full(counts.shape[0], -1, dtype=np.int64)
    compact_index[raw_indices] = np.arange(len(raw_indices), dtype=np.int64)
    compact_clone_ids = clone_ids[raw_indices]
    compact_time = time_values[raw_indices]
    compact_fate_values = fate_values[raw_indices]
    compact_context_values = context_values[raw_indices]

    print(
        f"Endpoint cells: {len(raw_indices):,} "
        f"(early={np.isclose(compact_time, args.early_day).sum():,}, "
        f"late={np.isclose(compact_time, args.late_day).sum():,})"
    )

    counts = counts.tocsr(copy=True)
    if args.log1p_scale > 0:
        counts.data *= float(args.log1p_scale)
        np.log1p(counts.data, out=counts.data)

    endpoint_counts = counts[raw_indices]
    n_hvgs = min(args.n_hvgs, endpoint_counts.shape[1])
    print(f"Selecting {n_hvgs:,} HVGs")
    mean = np.asarray(endpoint_counts.mean(axis=0)).ravel()
    sq_mean = np.asarray(endpoint_counts.power(2).mean(axis=0)).ravel()
    variance = sq_mean - mean**2
    hvg_idx = np.argsort(variance)[-n_hvgs:]
    hvg_idx.sort()
    hvg_genes = [genes[i] for i in hvg_idx]
    endpoint_hvg = endpoint_counts[:, hvg_idx]

    latent_dim = min(
        args.latent_dim, endpoint_hvg.shape[1] - 1, endpoint_hvg.shape[0] - 1
    )
    if latent_dim < 1:
        raise RuntimeError(
            f"Cannot compute latent_dim={args.latent_dim} from matrix {endpoint_hvg.shape}"
        )
    print(f"Fitting TruncatedSVD latent_dim={latent_dim}")
    svd = TruncatedSVD(n_components=latent_dim, random_state=args.seed)
    latents = svd.fit_transform(endpoint_hvg).astype(np.float32)
    latent_mean = latents.mean(axis=0, keepdims=True)
    latent_std = latents.std(axis=0, keepdims=True)
    latent_std[latent_std < 1e-6] = 1.0
    latents = ((latents - latent_mean) / latent_std).astype(np.float32)

    context_labels_by_cell, context_map = _encode_strings(compact_context_values)
    fate_labels_by_cell, fate_map = _encode_strings(compact_fate_values)

    pair_early: list[np.ndarray] = []
    pair_late: list[np.ndarray] = []
    pair_clone: list[np.ndarray] = []

    compact_early_mask = np.isclose(compact_time, args.early_day)
    compact_late_mask = np.isclose(compact_time, args.late_day)
    for clone in sorted(set(int(c) for c in compact_clone_ids)):
        clone_mask = compact_clone_ids == clone
        early_idx = np.flatnonzero(clone_mask & compact_early_mask)
        late_idx = np.flatnonzero(clone_mask & compact_late_mask)
        if len(early_idx) < args.min_early_cells or len(late_idx) < args.min_late_cells:
            continue
        e, l = _sample_pairs_for_clone(
            early_idx,
            late_idx,
            pairs_per_clone=args.pairs_per_clone,
            max_pairs_per_clone=args.max_pairs_per_clone,
            rng=rng,
        )
        pair_early.append(e.astype(np.int64))
        pair_late.append(l.astype(np.int64))
        pair_clone.append(np.full(len(e), clone, dtype=np.int64))

    if not pair_early:
        raise RuntimeError(
            "No eligible clones after min-cell filtering. Try lower "
            "--min-early-cells/--min-late-cells or check day values."
        )

    early_idx = np.concatenate(pair_early)
    late_idx = np.concatenate(pair_late)
    pair_clone_id = np.concatenate(pair_clone)
    split_by_clone = _split_clones(
        pair_clone_id,
        test_fraction=args.test_fraction,
        reserve_fraction=args.reserve_fraction,
        seed=args.seed,
    )
    split = np.asarray([split_by_clone[int(c)] for c in pair_clone_id], dtype=np.int64)
    context_label = context_labels_by_cell[late_idx].astype(np.int64)
    fate_label = fate_labels_by_cell[late_idx].astype(np.int64)

    split_counts = {
        "train": int((split == 0).sum()),
        "test": int((split == 1).sum()),
        "reserved": int((split == 2).sum()),
    }
    print(
        f"Pairs: {len(early_idx):,}; eligible clones: {len(split_by_clone):,}; "
        f"splits={split_counts}"
    )

    cell_metadata = metadata.iloc[raw_indices].copy()
    cell_metadata.insert(
        0, "compact_cell_id", np.arange(len(raw_indices), dtype=np.int64)
    )
    cell_metadata.insert(1, "raw_cell_id", raw_indices.astype(np.int64))
    cell_metadata["clone_id"] = compact_clone_ids
    cell_metadata["time_numeric"] = compact_time
    cell_metadata["context_label"] = context_labels_by_cell
    cell_metadata["fate_label"] = fate_labels_by_cell

    np.save(out_dir / "latents.npy", latents)
    np.savez_compressed(
        out_dir / "pairs.npz",
        early_idx=early_idx.astype(np.int64),
        late_idx=late_idx.astype(np.int64),
        raw_early_idx=raw_indices[early_idx].astype(np.int64),
        raw_late_idx=raw_indices[late_idx].astype(np.int64),
        clone_id=pair_clone_id.astype(np.int64),
        context_label=context_label,
        fate_label=fate_label,
        split=split,
    )
    cell_metadata.to_csv(out_dir / "cell_metadata.csv.gz", index=False)

    config = {
        "benchmark": "bit_diffusion_larry",
        "dataset": args.dataset,
        "raw_root": str(raw_root),
        "early_day": args.early_day,
        "late_day": args.late_day,
        "time_column": time_col,
        "context_column": context_col,
        "fate_column": fate_col,
        "n_endpoint_cells": int(len(raw_indices)),
        "n_pairs": int(len(early_idx)),
        "n_eligible_clones": int(len(split_by_clone)),
        "latent_dim": int(latent_dim),
        "requested_latent_dim": int(args.latent_dim),
        "n_hvgs": int(n_hvgs),
        "hvg_genes": hvg_genes,
        "svd_explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
        "latent_mean_before_standardization": latent_mean.reshape(-1).tolist(),
        "latent_std_before_standardization": latent_std.reshape(-1).tolist(),
        "context_map": context_map,
        "fate_map": fate_map,
        "split_labels": {"train": 0, "test": 1, "reserved": 2},
        "split_counts": split_counts,
        "source_urls": LARRY_FILES[args.dataset],
        "pairing_note": (
            "Pairs are sampled between early and late cells from the same clone; "
            "they are clone-conditioned endpoint pairs, not same-cell time series."
        ),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote processed benchmark to {out_dir}")


if __name__ == "__main__":
    main()
