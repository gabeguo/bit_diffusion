"""Download the public COCO 2014 validation images and caption-eval metadata.

Usage: python data_utils/download_coco.py --root /data/coco
Images are ~6.65 GB compressed; annotations ~253 MB. No training images are
needed for either the 30k validation-caption T2I or Karpathy 5k I2T protocol.
Interrupted downloads resume via HTTP Range. Archives are retained in downloads/.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import zipfile


BASE = "https://s3.amazonaws.com/images.cocodataset.org"
KARPATHY_URL = "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json"


def download(url, destination):
    """Atomically finish a download; never mistake a partial file for a result."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        resumed = offset > 0 and response.status == 206
        if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Server returned an unexpected download range.")
        expected = response.headers.get("Content-Length")
        received = 0
        with partial.open("ab" if resumed else "wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                received += len(chunk)
        if expected is not None and received != int(expected):
            raise IOError(f"Incomplete download from {url}; rerun to resume.")
    partial.replace(destination)


def extract(archive, root):
    """CRC-checked zip extraction with path traversal protection."""
    root = Path(root).resolve()
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            path = (root / info.filename).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Unsafe archive member: {info.filename}")
        source.extractall(root)


def karpathy_ids(path):
    """Read the original BLIP-distributed Karpathy test split, not a new split."""
    records = json.loads(Path(path).read_text())
    ids = []
    for row in records:
        match = re.fullmatch(r"val2014/COCO_val2014_(\d+)\.jpg", row["image"])
        if match is None:
            raise ValueError(f"Unexpected Karpathy image path: {row['image']}")
        ids.append(int(match.group(1)))
    if len(ids) != 5000 or len(set(ids)) != 5000:
        raise ValueError("Expected 5,000 distinct Karpathy test images.")
    return sorted(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true", help="Skip the 6.65 GB image archive.")
    args = parser.parse_args()
    archives = [("annotations/annotations_trainval2014.zip", "annotations")]
    if not args.metadata_only:
        archives.append(("zips/val2014.zip", "val2014"))
    for relative, name in archives:
        archive = args.root / "downloads" / Path(relative).name
        marker = args.root / f".{name}-extracted"
        if not marker.exists():
            print(f"Downloading/extracting {BASE}/{relative}", flush=True)
            download(f"{BASE}/{relative}", archive)
            extract(archive, args.root)
            marker.write_text("Extraction completed with zip CRC checks.\n")
    split = args.root / "coco_karpathy_test.json"
    download(KARPATHY_URL, split)
    karpathy_ids(split)
    print(f"Ready: {args.root}; Karpathy split SHA256={hashlib.sha256(split.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
