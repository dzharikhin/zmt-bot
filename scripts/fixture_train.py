"""Fixture driver for _build_profile — runs training outside Telegram.

Usage:
    export PANNS_WEIGHTS_PATH=/path/to/panns_cnn14.pth
    poetry run python scripts/fixture_train.py \
        --user-id 1 --model-id 1 --model-type INCLUDE_LIKED \
        --liked-dir /path/to/liked_mp3s --disliked-dir /path/to/disliked_mp3s
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from models import ModelType
from train import _build_profile, _execute_estimation

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_TYPE_NAMES = {mt.name for mt in ModelType}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run _build_profile directly (no Telegram)"
    )
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--model-id", type=int, required=True)
    parser.add_argument(
        "--model-type",
        required=True,
        choices=sorted(MODEL_TYPE_NAMES),
    )
    parser.add_argument(
        "--liked-dir",
        type=Path,
        required=True,
        help="Directory with liked MP3 files",
    )
    parser.add_argument(
        "--disliked-dir",
        type=Path,
        required=True,
        help="Directory with disliked MP3 files",
    )
    parser.add_argument(
        "--n-tracks",
        type=int,
        default=0,
        help="Cap number of tracks per set (0 = all)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing liked/disliked dirs before symlinking",
    )
    parser.add_argument(
        "--estimate",
        type=Path,
        default=None,
        help="After training, score this MP3 file with the model",
    )
    return parser.parse_args()


def symlink_tracks(source_dir: Path, target_dir: Path, n_tracks: int, clean: bool):
    if clean and target_dir.exists():
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    mp3s = sorted(source_dir.glob("*.mp3"))
    if n_tracks > 0:
        mp3s = mp3s[:n_tracks]

    if not mp3s:
        logger.error(f"No MP3 files found in {source_dir}")
        sys.exit(1)

    for mp3 in mp3s:
        link_path = target_dir / mp3.name
        if link_path.exists():
            continue
        link_path.symlink_to(mp3)

    logger.info(f"Symlinked {len(mp3s)} tracks from {source_dir} -> {target_dir}")


def main():
    args = parse_args()

    if not config.panns_weights_path.exists():
        logger.error(
            f"PANNs weights not found at {config.panns_weights_path}. "
            f"Set PANNS_WEIGHTS_PATH env var or download weights."
        )
        sys.exit(1)

    model_type = ModelType[args.model_type]

    liked_target = config.get_liked_file_store_path(args.user_id)
    disliked_target = config.get_disliked_file_store_path(args.user_id)

    symlink_tracks(args.liked_dir, liked_target, args.n_tracks, args.clean)
    symlink_tracks(args.disliked_dir, disliked_target, args.n_tracks, args.clean)

    logger.info(
        f"Calling _build_profile(user_id={args.user_id}, "
        f"model_id={args.model_id}, model_type={model_type.name})"
    )

    model = _build_profile(args.user_id, args.model_id, model_type)

    logger.info(
        f"Training complete: model_id={model.model_id}, "
        f"accuracy={model.accuracy:.2f}, "
        f"liked={model.liked_tracks_count}, disliked={model.disliked_tracks_count}, "
        f"thresholds={model.thresholds}, embed_version={model.embed_version}"
    )

    if args.estimate:
        logger.info(f"Estimating track: {args.estimate}")
        recommended = _execute_estimation(args.user_id, args.model_id, args.estimate)
        logger.info(f"Recommendation: {recommended}")


if __name__ == "__main__":
    main()
