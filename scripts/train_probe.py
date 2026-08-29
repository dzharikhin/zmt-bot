"""Headless end-to-end train + estimate probe (P6 ship validation).

Runs the exact production internal APIs the bot uses —
train._build_profile and train._execute_estimation — without Telegram,
then checks the fitted stats and the gate outcomes against expected
bands. Writes a scratch model under data/{uid}/models/{model_id}
(delete the directory afterwards).

    docker run --rm -v ./data:/app/data -v ./local_data:/app/local_data \
      --cpus 4 --memory 4G tg-zmt-bot:$(poetry version --short) \
      -m scripts.train_probe --user-id $OWNER_USER_ID \
      --liked-track /app/data/$OWNER_USER_ID/liked/first.mp3 \
      --disliked-track /app/data/$OWNER_USER_ID/disliked/second.mp3
"""

import argparse
import json
import logging
from pathlib import Path

import config
from core.modeling import DualOneClassModel
from models import ModelType
from train import _build_profile, _execute_estimation

logger = logging.getLogger(__name__)

EXCLUDE_FP_CAP = 0.07
INCLUDE_FP_CAP = 0.07


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--model-id", type=int, default=999)
    parser.add_argument("--liked-track", type=Path, required=True)
    parser.add_argument("--disliked-track", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info(
        f"like={config.model_like_preprocessor} "
        f"dislike={config.model_dislike_preprocessor} "
        f"decision={config.model_decision_mode} w={config.model_fusion_weight}"
    )

    _build_profile(args.user_id, args.model_id)

    model_store = config.get_model_store_path(args.user_id, args.model_id)
    stats = json.loads(
        (model_store.model_workdir / model_store.model_stats_name).read_text()
    )
    model = DualOneClassModel.load(model_store.model_workdir)

    print("=== stats ===")
    for key in (
        "decision_mode",
        "fusion_weight",
        "liked_tracks_count",
        "disliked_tracks_count",
        "outliers_removed_liked",
        "outliers_removed_disliked",
        "thresholds",
    ):
        print(f"{key}: {stats.get(key)}")
    for key in (
        "metrics_source",
        "exclude_disliked_tp",
        "exclude_disliked_fp",
        "include_liked_tp",
        "include_liked_fp",
    ):
        print(f"operating.{key}: {stats.get(key)}")
    print(f"liked_preprocessor: {type(model.liked_preprocessor).__name__}")
    print(f"disliked_preprocessor: {type(model.disliked_preprocessor).__name__}")

    checks: list[tuple[str, bool]] = [
        ("decision_mode == fused_diff", stats.get("decision_mode") == "fused_diff"),
        ("fusion_weight == 1.0", stats.get("fusion_weight") == 1.0),
        (
            f"exclude_disliked_fp <= {EXCLUDE_FP_CAP}",
            stats.get("exclude_disliked_fp", 1.0) <= EXCLUDE_FP_CAP,
        ),
        (
            f"include_liked_fp <= {INCLUDE_FP_CAP}",
            stats.get("include_liked_fp", 1.0) <= INCLUDE_FP_CAP,
        ),
    ]

    print("=== estimation ===")
    for label, track, model_type, expected in (
        ("liked/include", args.liked_track, ModelType.INCLUDE_LIKED, True),
        ("liked/exclude", args.liked_track, ModelType.EXCLUDE_DISLIKED, False),
        ("disliked/exclude", args.disliked_track, ModelType.EXCLUDE_DISLIKED, True),
        ("disliked/include", args.disliked_track, ModelType.INCLUDE_LIKED, False),
    ):
        got = _execute_estimation(args.user_id, args.model_id, track, model_type)
        print(f"{label}: {got} (expected {expected})")
        checks.append((f"estimation {label}", got == expected))

    print("=== verdict ===")
    ok = True
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
        ok = ok and passed
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
