"""Download public LARRY state-fate data files.

This script downloads data when invoked from a shell script or manually.
"""

from __future__ import annotations

import argparse
import json
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "https://kleintools.hms.harvard.edu/paper_websites/state_fate2020"

LARRY_FILES: dict[str, dict[str, str]] = {
    "in_vitro": {
        "counts": f"{BASE_URL}/stateFate_inVitro_normed_counts.mtx.gz",
        "genes": f"{BASE_URL}/stateFate_inVitro_gene_names.txt.gz",
        "metadata": f"{BASE_URL}/stateFate_inVitro_metadata.txt.gz",
        "clones": f"{BASE_URL}/stateFate_inVitro_clone_matrix.mtx.gz",
        "neutrophil_pseudotime": f"{BASE_URL}/stateFate_inVitro_neutrophil_pseudotime.txt.gz",
        "neutrophil_monocyte_trajectory": f"{BASE_URL}/stateFate_inVitro_neutrophil_monocyte_trajectory.txt.gz",
    },
    "in_vivo": {
        "counts": f"{BASE_URL}/stateFate_inVivo_normed_counts.mtx.gz",
        "genes": f"{BASE_URL}/stateFate_inVivo_gene_names.txt.gz",
        "metadata": f"{BASE_URL}/stateFate_inVivo_metadata.txt.gz",
        "clones": f"{BASE_URL}/stateFate_inVivo_clone_matrix.mtx.gz",
    },
    "cytokine": {
        "counts": f"{BASE_URL}/stateFate_cytokinePerturbation_normed_counts.mtx.gz",
        "genes": f"{BASE_URL}/stateFate_cytokinePerturbation_gene_names.txt.gz",
        "metadata": f"{BASE_URL}/stateFate_cytokinePerturbation_metadata.txt.gz",
        "clones": f"{BASE_URL}/stateFate_cytokinePerturbation_clone_matrix.mtx.gz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default="in_vitro",
        choices=[*LARRY_FILES.keys(), "all"],
        help="LARRY experiment to download.",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="state_fate/data/raw",
        help="Directory where per-dataset raw folders are created.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retry failed downloads this many times.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=10.0,
        help="Base sleep between retries. The script uses linear backoff.",
    )
    parser.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="Disable HTTPS certificate verification for this public data download.",
    )
    return parser.parse_args()


def _filename_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def _download_one(
    url: str,
    dst: Path,
    *,
    force: bool,
    max_retries: int,
    sleep_s: float,
    insecure_ssl: bool,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        print(f"[skip] {dst} already exists")
        return

    part = dst.with_suffix(dst.suffix + ".part")
    if part.exists():
        part.unlink()

    ssl_context = ssl._create_unverified_context() if insecure_ssl else None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[download] {url}")
            with urllib.request.urlopen(
                url, timeout=120, context=ssl_context
            ) as response:
                with part.open("wb") as f:
                    shutil.copyfileobj(response, f)
            part.replace(dst)
            print(f"[ok] {dst}")
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if part.exists():
                part.unlink()
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Failed to download {url} after {max_retries} attempts"
                ) from exc
            wait = sleep_s * attempt
            print(
                f"[retry] attempt {attempt}/{max_retries} failed for {url}: {exc}. "
                f"Sleeping {wait:.1f}s.",
                file=sys.stderr,
            )
            time.sleep(wait)


def main() -> None:
    args = parse_args()
    datasets = list(LARRY_FILES) if args.dataset == "all" else [args.dataset]
    raw_root = Path(args.raw_root)
    if args.insecure_ssl:
        print("[warn] HTTPS certificate verification is disabled for LARRY downloads.")

    for dataset in datasets:
        dataset_dir = raw_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "dataset": dataset,
            "source": "Klein Lab LARRY state-fate paper-data",
            "files": LARRY_FILES[dataset],
        }
        (dataset_dir / "download_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(f"== {dataset} -> {dataset_dir} ==")
        for _kind, url in LARRY_FILES[dataset].items():
            dst = dataset_dir / _filename_from_url(url)
            _download_one(
                url,
                dst,
                force=args.force,
                max_retries=args.max_retries,
                sleep_s=args.retry_sleep_seconds,
                insecure_ssl=args.insecure_ssl,
            )

    print("Download step complete.")


if __name__ == "__main__":
    main()
