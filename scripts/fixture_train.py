"""Fixture driver for _build_profile — runs training outside Telegram.

Usage:
    export PANNS_WEIGHTS_PATH=/zmt-bot/data/panns_data/panns_cnn14.pth
    poetry run python scripts/fixture_train.py \
        --user-id 1 --model-id 1

    # With estimation:
    poetry run python scripts/fixture_train.py \
        --user-id 1 --model-id 1 \
        --estimate /path/to/track.mp3 --estimate-model-type INCLUDE_LIKED
"""

import argparse
import logging
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
        "--estimate",
        type=Path,
        default=None,
        help="After training, score this MP3 file with the model",
    )
    parser.add_argument(
        "--estimate-model-type",
        required=False,
        choices=sorted(MODEL_TYPE_NAMES),
        default="INCLUDE_LIKED",
        help="Decision policy for estimation (default: INCLUDE_LIKED)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not config.panns_weights_path.exists():
        logger.error(
            f"PANNs weights not found at {config.panns_weights_path}. "
            f"Set PANNS_WEIGHTS_PATH env var or download weights."
        )
        sys.exit(1)

    liked_dir = config.get_liked_file_store_path(args.user_id)
    disliked_dir = config.get_disliked_file_store_path(args.user_id)
    liked_count = len(list(liked_dir.glob("*.mp3")))
    disliked_count = len(list(disliked_dir.glob("*.mp3")))

    if liked_count == 0 or disliked_count == 0:
        logger.error(
            f"No tracks found: liked={liked_count}, disliked={disliked_count}. "
            f"Place MP3s in {liked_dir} and {disliked_dir}."
        )
        sys.exit(1)

    logger.info(
        f"Found {liked_count} liked, {disliked_count} disliked tracks for user {args.user_id}"
    )
    logger.info(
        f"Calling _build_profile(user_id={args.user_id}, " f"model_id={args.model_id})"
    )

    model = _build_profile(args.user_id, args.model_id)

    logger.info(
        f"Training complete: model_id={model.model_id}, "
        f"disliked_false_accept={model.disliked_false_accept:.2f}, "
        f"liked_false_reject={model.liked_false_reject:.2f}, "
        f"metrics_source={model.metrics_source}, "
        f"liked={model.liked_tracks_count}, disliked={model.disliked_tracks_count}, "
        f"thresholds={model.thresholds}, embed_version={model.embed_version}"
    )

    if args.estimate:
        model_type = ModelType[args.estimate_model_type]
        logger.info(f"Estimating track: {args.estimate} with policy {model_type.name}")
        recommended = _execute_estimation(
            args.user_id, args.model_id, args.estimate, model_type
        )
        logger.info(f"Recommendation: {recommended}")


if __name__ == "__main__":
    main()
