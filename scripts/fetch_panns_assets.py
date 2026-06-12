#!/usr/bin/env python3
"""Download PANNs CNN14 assets into data/panns_data/.

Usage:
    python scripts/fetch_panns_assets.py [destination_dir]

Defaults to data/panns_data/.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

PANNS_WEIGHTS_URL = (
    "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
)
LABELS_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"

FILES = {
    "panns_cnn14.pth": PANNS_WEIGHTS_URL,
    "class_labels_indices.csv": LABELS_URL,
}


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PANNs CNN14 assets")
    parser.add_argument(
        "dest",
        nargs="?",
        default="data/panns_data",
        help="Destination directory (default: data/panns_data)",
    )
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for name, url in FILES.items():
        target = dest / name
        if target.exists():
            print(f"{name} already exists, skipping")
            continue
        print(f"Downloading {name}...")
        download(url, target)
        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"  {name} ({size_mb:.1f} MB)")

    print("Done. Files in", dest)
    for p in sorted(dest.iterdir()):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name}  {size_mb:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
