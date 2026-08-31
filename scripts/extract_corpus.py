"""Re-extract the corpus under a new embed_version (P5.2).

Track sources:
- default: read source_path/set_name verbatim from existing cache parquet
  shards under data/{uid}/features/{from_embed}/full/{set_name}/*.parquet
- --from-dirs: walk the downloaded audio store data/{uid}/liked/*.mp3 and
  data/{uid}/disliked/*.mp3 (mirror of train._build_profile; use when the
  feature cache is gone)

Resumable via the feature-cache probe; rerun the same command to continue.
After extraction every target-embed shard is verified (vector width +
finiteness); a non-zero exit code marks a failed run.

    docker run --rm -v ./data:/app/data -v ./local_data:/app/local_data \
      --cpus 4 --memory 4G tg-zmt-bot:$(poetry version --short) \
      -m scripts.extract_corpus \
      --user-id $OWNER_USER_ID --workers 4 --from-dirs \
      --profile /app/data/essentia_extractor_profile.yaml
"""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np

import config
from audio.features import _DESCRIPTOR_SCHEMA
from core.paths import get_embed_version
from core.storage import FeatureStore
from core.writer import start_extraction_job

logger = logging.getLogger(__name__)

_PANNS_EMBED_DIM = 2048
_EXPECTED_VECTOR_DIM = (
    sum(length for _, length, _ in _DESCRIPTOR_SCHEMA) + _PANNS_EMBED_DIM
)


def _resolve_from_embed(features_root: Path, from_embed: str | None) -> str:
    if from_embed:
        return from_embed
    candidates = sorted(
        (d for d in features_root.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No embed_version dirs under {features_root}")
    chosen = candidates[-1]
    logger.info(f"--from-embed not set, using most recent: {chosen.name}")
    return chosen.name


def _load_tracks(features_root: Path, from_embed: str) -> list[tuple[Path, str]]:
    pattern = f"{features_root}/{from_embed}/full/*/*.parquet"
    rows = duckdb.sql(
        f"SELECT DISTINCT source_path, set_name FROM read_parquet('{pattern}')"
    ).fetchall()
    tracks = []
    missing = 0
    for source_path, set_name in rows:
        path = Path(source_path)
        if path.exists():
            tracks.append((path, set_name))
        else:
            missing += 1
    if missing:
        logger.warning(f"Skipping {missing} cached tracks whose source files are gone")
    logger.info(
        f"Loaded {len(tracks)} tracks from {from_embed} "
        f"({missing} missing sources, sets: "
        f"{sorted({s for _, s in tracks})})"
    )
    return tracks


def _load_tracks_from_dirs(user_id: int) -> list[tuple[Path, str]]:
    liked = sorted(config.get_liked_file_store_path(user_id).glob("*.mp3"))
    disliked = sorted(config.get_disliked_file_store_path(user_id).glob("*.mp3"))
    tracks = [(t, "like") for t in liked] + [(t, "dislike") for t in disliked]
    logger.info(
        f"Loaded {len(tracks)} tracks from the audio store "
        f"(liked={len(liked)}, disliked={len(disliked)})"
    )
    return tracks


def _verify_shards(user_id: int, embed_version: str) -> bool:
    root = config.get_feature_store_root(user_id) / embed_version / "full"
    ok = True
    checked = 0
    for set_name in ("like", "dislike"):
        for shard in sorted((root / set_name).glob("*.parquet")):
            vectors = FeatureStore.load_vectors(shard)
            checked += 1
            if vectors.size == 0:
                logger.error(f"Shard {shard} is empty")
                ok = False
                continue
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            if vectors.shape[1] != _EXPECTED_VECTOR_DIM:
                logger.error(
                    f"Shard {shard} width {vectors.shape[1]} "
                    f"!= expected {_EXPECTED_VECTOR_DIM}"
                )
                ok = False
            if not np.isfinite(vectors).all():
                logger.error(f"Shard {shard} contains non-finite values")
                ok = False
    logger.info(
        f"Shard sanity: checked={checked} expected_dim={_EXPECTED_VECTOR_DIM} "
        f"result={'PASS' if ok else 'FAIL'}"
    )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--from-embed",
        default=None,
        help="embed_version to read source paths from (default: most recent "
        "under data/{uid}/features)",
    )
    source.add_argument(
        "--from-dirs",
        action="store_true",
        help="load tracks from the audio store data/{uid}/{liked,disliked}/*.mp3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="extract only the first N loaded tracks (smoke runs)",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("/app/essentia_profile.yaml"),
        help="essentia profile for the NEW embed_version",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    features_root = config.get_feature_store_root(args.user_id)
    if args.from_dirs:
        tracks = _load_tracks_from_dirs(args.user_id)
    else:
        from_embed = _resolve_from_embed(features_root, args.from_embed)
        tracks = _load_tracks(features_root, from_embed)
        logger.info(f"Source embed_version: {from_embed}")
    if args.limit is not None:
        tracks = tracks[: args.limit]
        logger.info(f"--limit {args.limit}: extraction capped to {len(tracks)} tracks")
    if not tracks:
        raise RuntimeError("No tracks to extract")

    embed_version = get_embed_version(profile_path=args.profile)
    logger.info(f"Target embed_version: {embed_version}")

    job_id = (
        f"extract_corpus_{args.user_id}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    )
    result = start_extraction_job(
        user_id=args.user_id,
        tracks=tracks,
        embed_version=embed_version,
        segment_policy="full",
        job_id=job_id,
        n_workers=args.workers,
        profile_path=args.profile,
    )
    logger.info(
        f"Extraction done: ok={result.ok} failed={result.failed} "
        f"skipped={result.skipped}"
    )

    if not _verify_shards(args.user_id, embed_version):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
