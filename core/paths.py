import hashlib
from importlib.metadata import version as pkg_version
from pathlib import Path

import config


def get_embed_version(
    profile_path: Path | None = None,
    panns_weights_path: Path | None = None,
) -> str:
    if profile_path is None:
        profile_path = config.data_path / "essentia_extractor_profile.yaml"
    if panns_weights_path is None:
        panns_weights_path = config.panns_weights_path
    essentia_version = pkg_version("essentia")
    profile_hash = compute_file_hash(profile_path)
    panns_hash = compute_file_hash(panns_weights_path)
    return f"essentia-{essentia_version}+profile-{profile_hash}+panns-{panns_hash}"


def compute_file_hash(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()
