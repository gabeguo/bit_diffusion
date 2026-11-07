"""Install the unchanged Stanford CoreNLP 3.6 jars used by COCO SPICE.

Run once in a writable Python environment with internet access, before scoring
on offline nodes. Uses HTTPS instead of pycocoevalcap 1.2's obsolete HTTP URL.
This does not change the SPICE scorer, models, or tokenization.
"""

import argparse
from pathlib import Path
import shutil
import zipfile

try:
    from data_utils.download_coco import download
except ModuleNotFoundError:  # direct `python data_utils/prepare_spice.py`
    from download_coco import download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    from pycocoevalcap.spice.get_stanford_models import CORENLP, JAR, SPICEDIR, SPICELIB
    destination = Path(SPICEDIR) / SPICELIB
    jars = [f"{JAR}.jar", f"{JAR}-models.jar"]
    if not all((destination / name).is_file() for name in jars):
        archive = args.cache_dir / f"{CORENLP}.zip"
        download(f"https://nlp.stanford.edu/software/{CORENLP}.zip", archive)
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            for name in jars:
                temporary = destination / (name + ".tmp")
                with source.open(f"{CORENLP}/{name}") as incoming, temporary.open("wb") as output:
                    shutil.copyfileobj(incoming, output)
                temporary.replace(destination / name)
    print(f"SPICE CoreNLP jars ready in {destination}. Java 8 must be on PATH.")


if __name__ == "__main__":
    main()
