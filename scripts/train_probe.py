"""Headless end-to-end train + estimate probe (P6 ship validation).

Runs the exact production internal APIs the bot uses —
train._build_profile and train._execute_estimation — without Telegram,
then checks the fitted stats and the gate outcomes. Writes a scratch
model under data/{uid}/models/{model_id} (delete the directory
afterwards).

Semantics: _execute_estimation returns is_recommended (True = the bot
forwards the track). Expected outcomes:
- liked track, INCLUDE_LIKED       -> True  (forwarded as a like)
- liked track, EXCLUDE_DISLIKED    -> True  (not filtered)
- disliked track, EXCLUDE_DISLIKED -> False (filtered)
- disliked track, INCLUDE_LIKED    -> False (not forwarded)

The calibrated gates miss some tracks by design (include tp=0.775,
exclude tp=0.80), so single-track expectations are noisy: the probe
scores --n-tracks tracks per class (evenly spaced through the sorted
audio store) and a check passes when ANY track matches. That catches
systematic/polarity breakage while tolerating the calibrated miss
rate; read the per-track lines for detail.

    docker run --rm -v ./data:/app/data -v ./local_data:/app/local_data \
      --cpus 4 --memory 4G tg-zmt-bot:$(poetry version --short) \
      -m scripts.train_probe --user-id $OWNER_USER_ID
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


def _pick_tracks(store_dir: Path, n: int) -> list[Path]:
    tracks = sorted(store_dir.glob("*.mp3"))
    if not tracks:
        raise RuntimeError(f"No mp3 tracks under {store_dir}")
    if n <= 1 or len(tracks) == 1:
        return [tracks[0]]
    indices = dict.fromkeys(round(k * (len(tracks) - 1) / (n - 1)) for k in range(n))
    return [tracks[i] for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--model-id", type=int, default=999)
    parser.add_argument(
        "--liked-track",
        type=Path,
        default=None,
        help="manual override: probe only this liked track",
    )
    parser.add_argument(
        "--disliked-track",
        type=Path,
        default=None,
        help="manual override: probe only this disliked track",
    )
    parser.add_argument(
        "--n-tracks",
        type=int,
        default=3,
        help="tracks per class probed when no manual override is given",
    )
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
    liked_tracks = (
        [args.liked_track]
        if args.liked_track
        else _pick_tracks(config.get_liked_file_store_path(args.user_id), args.n_tracks)
    )
    disliked_tracks = (
        [args.disliked_track]
        if args.disliked_track
        else _pick_tracks(
            config.get_disliked_file_store_path(args.user_id), args.n_tracks
        )
    )
    for label, tracks, model_type, expected in (
        ("liked/include", liked_tracks, ModelType.INCLUDE_LIKED, True),
        ("liked/exclude", liked_tracks, ModelType.EXCLUDE_DISLIKED, True),
        ("disliked/exclude", disliked_tracks, ModelType.EXCLUDE_DISLIKED, False),
        ("disliked/include", disliked_tracks, ModelType.INCLUDE_LIKED, False),
    ):
        matches = 0
        for track in tracks:
            got = _execute_estimation(args.user_id, args.model_id, track, model_type)
            matches += got == expected
            print(f"{label} [{track.name}]: {got} (expected {expected})")
        checks.append(
            (f"estimation {label} ({matches}/{len(tracks)} match)", matches > 0)
        )

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
