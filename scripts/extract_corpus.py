"""Re-extract the corpus under a new embed_version (P5.2).

Reads source_path/set_name verbatim from existing cache parquet shards under
data/{uid}/features/{from_embed}/full/{set_name}/*.parquet and feeds them to
start_extraction_job() with the new embed_version. Resumable via the
feature-cache probe; rerun the same command to continue.

    docker run --rm -v ./data:/app/data -v ./local_data:/app/local_data \
      --cpus 4 --memory 4G tg-zmt-bot:$(poetry version --short) \
      -m scripts.extract_corpus \
      --user-id $OWNER_USER_ID --workers 4 --profile /app/essentia_profile.yaml
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import duckdb

import config
from core.paths import get_embed_version
from core.writer import start_extraction_job

logger = logging.getLogger(__name__)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--from-embed",
        default=None,
        help="embed_version to read source paths from (default: most recent "
        "under data/{uid}/features)",
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
    from_embed = _resolve_from_embed(features_root, args.from_embed)
    tracks = _load_tracks(features_root, from_embed)
    if not tracks:
        raise RuntimeError("No tracks to extract")

    embed_version = get_embed_version(profile_path=args.profile)
    logger.info(f"Source embed_version: {from_embed}")
    logger.info(f"Target embed_version: {embed_version}")

    job_id = (
        f"extract_corpus_{args.user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
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


if __name__ == "__main__":
    main()
