import asyncio
import atexit
import json
import logging
import pathlib
import shutil
import sys
import tempfile
import time
from datetime import datetime
from typing import Literal, cast

from telethon import TelegramClient
from telethon.tl.custom import Message
from telethon.tl.types import (
    DocumentAttributeAudio,
    MessageMediaPhoto,
)
from telethon.tl.types import (
    Message as TlMessage,
)

import config
from audio.extractor import extract_features_for_mp3
from audio.features import prepare_extractor
from bot_utils import get_chat, get_message, obtain_latest_message_id
from core.modeling import DualOneClassModel
from core.outliers import detect_outliers
from core.paths import get_embed_version
from core.preprocessing import NoOpPreprocessor, StandardizeSelectPreprocessor
from core.storage import FeatureStore
from core.writer import start_extraction_job
from models import ModelType

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

atexit_handler_registered = False


def _log_error_on_exit():
    """Log error details if process exits with an error"""
    if sys.exc_info()[0] is not None:
        logger.error("Exit with exception", exc_info=True)


def _register_atexit_handler():
    global atexit_handler_registered
    if not atexit_handler_registered:
        atexit.register(_log_error_on_exit)
        atexit_handler_registered = True


class TrainUnrecoverable(Exception):
    pass


class EstimationUnrecoverable(Exception):
    pass


class Mp3Filter:
    def __init__(self, **params):
        self.min_length_seconds = params["min_seconds"]
        self.max_length_seconds = params["max_seconds"]

    def filter_message(self, message) -> bool:
        if not message:
            return False
        if not isinstance(message, (TlMessage, Message)):
            return False
        if isinstance(message, MessageMediaPhoto):
            return False
        if not hasattr(message, "media") or not hasattr(message.media, "document"):
            return False
        if not hasattr(
            message.media.document, "mime_type"
        ) or message.media.document.mime_type not in {
            "audio/mpeg",
            "audio/mp3",
        }:
            return False
        if not hasattr(message.media.document, "attributes") or not [
            audio_attr := cast(DocumentAttributeAudio, attr)
            for attr in message.media.document.attributes
            if isinstance(attr, DocumentAttributeAudio)
        ]:
            return False
        if (
            audio_attr.duration < self.min_length_seconds
            or audio_attr.duration > self.max_length_seconds
        ):
            return False
        return True

    def __repr__(self):
        return f"Mp3Filter[min_length_seconds={self.min_length_seconds}, "
        f"max_length_seconds={self.max_length_seconds}]"


FILTER = Mp3Filter(
    min_seconds=config.min_track_length_seconds,
    max_seconds=config.max_track_length_seconds,
)


async def save_track_if_not_exists(
    user_id: int, message: Message, channel_type: Literal["liked", "disliked"]
):
    tracks_folder = (
        config.get_disliked_file_store_path(user_id)
        if channel_type == "disliked"
        else config.get_liked_file_store_path(user_id)
    )
    file_path = tracks_folder.joinpath(f"{message.file.id}{message.file.ext}")
    if not file_path.exists():
        await message.download_media(file=file_path)


async def download_audio_from_channel(
    user_id: int,
    channel_id: int,
    latest_message_links: list[str],
    channel_type: Literal["liked", "disliked"],
    bot_client: TelegramClient,
    limit: int | None = None,
):
    channel = await get_chat(channel_id, bot_client)
    if not channel:
        raise TrainUnrecoverable(f"Channel {channel_id} is not available")

    latest_message_id = await obtain_latest_message_id(channel, latest_message_links)
    ids = list(range(latest_message_id + 1))
    if limit:
        ids = ids[-limit:]
    start = time.time()
    async for message in bot_client.iter_messages(channel, ids=ids, reverse=True):
        got_message = time.time()
        if not FILTER.filter_message(message):
            if message:
                logger.info(
                    f"Message {message.stringify()} does not match {FILTER}, skipping"
                )
            continue
        filtered_message = time.time()
        await save_track_if_not_exists(user_id, message, channel_type)
        logger.debug(
            f"Handled msg={message.id}: "
            f"got message in {got_message - start:.2f} sec, "
            f"filtered in {filtered_message - got_message:.2f} sec, "
            f"saved in {time.time() - filtered_message:.2f} sec"
        )
        start = time.time()


def _build_profile(user_id: int, model_id: int) -> config.Model:
    """Build two one-class models from liked/disliked tracks (internal API)"""
    _register_atexit_handler()

    resolved_profile = config.data_path / "essentia_extractor_profile.yaml"
    model_store = config.get_model_store_path(user_id, model_id)
    bundled_profile = model_store.model_workdir / "essentia_profile.yaml"
    shutil.copy(resolved_profile, bundled_profile)

    embed_version = get_embed_version(profile_path=bundled_profile)
    segment_policy = config.segment_policy

    liked_path = config.get_liked_file_store_path(user_id)
    disliked_path = config.get_disliked_file_store_path(user_id)

    if not liked_path.exists() or not disliked_path.exists():
        raise TrainUnrecoverable("Track directories do not exist. Run /init first.")

    liked_tracks = list(liked_path.glob("*.mp3"))
    disliked_tracks = list(disliked_path.glob("*.mp3"))

    if not liked_tracks or not disliked_tracks:
        raise TrainUnrecoverable("No tracks found. Add tracks to channels first.")

    logger.info(
        f"Found {len(liked_tracks)} liked, {len(disliked_tracks)} disliked tracks"
    )

    all_tracks = [(t, "like") for t in liked_tracks] + [
        (t, "dislike") for t in disliked_tracks
    ]

    job_id = (
        f"profile_{user_id}_{model_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )
    logger.info(f"Starting feature extraction job {job_id}")

    def progress_callback(job_id, done, total, status, **kwargs):
        if status == "running":
            pct = (done / total * 100) if total > 0 else 0
            logger.info(
                f"Extraction progress: {done}/{total} ({pct:.1f}%) - "
                f"ok={kwargs.get('ok', 0)}, failed={kwargs.get('failed', 0)}, "
                f"skipped={kwargs.get('skipped', 0)}"
            )

    try:
        start_extraction_job(
            user_id=user_id,
            tracks=all_tracks,
            embed_version=embed_version,
            segment_policy=segment_policy,
            job_id=job_id,
            progress_callback=progress_callback,
            profile_path=bundled_profile,
        )
    except Exception as e:
        raise TrainUnrecoverable(f"Feature extraction failed: {e}") from e

    logger.info("Loading features from parquet cache")
    store = FeatureStore(user_id, embed_version, segment_policy)

    with store.training_view("like") as liked_pq:
        X_liked_raw = FeatureStore.load_vectors(liked_pq)
    with store.training_view("dislike") as disliked_pq:
        X_disliked_raw = FeatureStore.load_vectors(disliked_pq)

    logger.info(
        f"Loaded {len(X_liked_raw)} liked, {len(X_disliked_raw)} disliked features"
    )

    if len(X_liked_raw) == 0 or len(X_disliked_raw) == 0:
        raise TrainUnrecoverable("No features extracted. Check logs for errors.")

    outliers_removed_liked = 0
    outliers_removed_disliked = 0

    if config.model_outlier_threshold > 0:
        logger.info(
            f"Applying outlier detection (threshold={config.model_outlier_threshold})"
        )

        mask_liked, outliers_liked = detect_outliers(
            X_liked_raw,
            threshold=config.model_outlier_threshold,
            knn_k=config.model_knn_k_max,
            n_estimators=200,
            min_set_size=config.model_min_set_size,
        )

        mask_disliked, outliers_disliked = detect_outliers(
            X_disliked_raw,
            threshold=config.model_outlier_threshold,
            knn_k=config.model_knn_k_max,
            n_estimators=200,
            min_set_size=config.model_min_set_size,
        )

        outliers_removed_liked = len(outliers_liked)
        outliers_removed_disliked = len(outliers_disliked)
        logger.info(
            f"Filtered {outliers_removed_liked} liked, "
            f"{outliers_removed_disliked} disliked outliers"
        )

        X_liked = X_liked_raw[mask_liked]
        X_disliked = X_disliked_raw[mask_disliked]
    else:
        logger.info("Outlier detection disabled")
        X_liked = X_liked_raw
        X_disliked = X_disliked_raw

    n_liked = len(X_liked)
    n_disliked = len(X_disliked)
    n_min = max(1, min(n_liked, n_disliked))
    imbalance_ratio = round(max(n_liked, n_disliked) / n_min, 2)

    if imbalance_ratio > config.model_max_imbalance_ratio:
        logger.warning(
            f"Imbalance ratio {imbalance_ratio} exceeds threshold "
            f"{config.model_max_imbalance_ratio} "
            f"(liked={n_liked}, disliked={n_disliked}). "
            f"Consider adding more tracks to the smaller set."
        )
    else:
        logger.info(
            f"Imbalance ratio {imbalance_ratio} "
            f"(liked={n_liked}, disliked={n_disliked})"
        )

    logger.info("Fitting DualOneClassModel")
    preprocessor = (
        StandardizeSelectPreprocessor(n_features=config.model_select_n_features)
        if config.model_preprocessor == "standardize_select"
        else NoOpPreprocessor()
    )
    model = DualOneClassModel(
        knn_k_min=config.model_knn_k_min,
        knn_k_max=config.model_knn_k_max,
        knn_k_scale=config.model_knn_k_scale,
        gmm_components_max=config.model_gmm_components_max,
        gmm_min_points_per_component=config.model_gmm_min_points_per_component,
        cv_folds=config.model_cv_folds,
        exclude_disliked_recall_target=config.model_exclude_disliked_recall_target,
        include_liked_recall_target=config.model_include_liked_recall_target,
        preprocessor=preprocessor,
    )
    model.fit(X_liked, X_disliked)
    model.embed_version = embed_version
    model.segment_policy = segment_policy

    model.save(model_store.model_workdir)

    stats = {
        **model.stats,
        "liked_tracks_count": n_liked,
        "disliked_tracks_count": n_disliked,
        "accuracy": 0.0,
        "thresholds": model.thresholds,
        "embed_version": embed_version,
        "outliers_removed_liked": outliers_removed_liked,
        "outliers_removed_disliked": outliers_removed_disliked,
    }
    with open(model_store.model_workdir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Model saved to {model_store.model_workdir}")

    return config.get_model(user_id, model_id)


async def prepare_model(
    user_id: int,
    bot_client: TelegramClient,
    latest_message_links: list[str],
    model_id: int,
    force: bool = False,
    limit: int | None = None,
):
    try:
        channels = config.get_user_channels(user_id)
        if not channels:
            raise TrainUnrecoverable(f"User {user_id} has no channels initialized")

        if force:
            liked_path = config.get_liked_file_store_path(user_id)
            disliked_path = config.get_disliked_file_store_path(user_id)
            if liked_path.exists():
                shutil.rmtree(liked_path)
            if disliked_path.exists():
                shutil.rmtree(disliked_path)

        await download_audio_from_channel(
            user_id,
            channels.liked_channel_id,
            latest_message_links,
            "liked",
            bot_client,
            limit,
        )
        await download_audio_from_channel(
            user_id,
            channels.disliked_channel_id,
            latest_message_links,
            "disliked",
            bot_client,
            limit,
        )
        await asyncio.get_running_loop().run_in_executor(
            config.get_training_executor(),
            _build_profile,
            user_id,
            model_id,
        )

    except TrainUnrecoverable as e:
        logger.error(
            f"Training failed for user {user_id} model {model_id}: {e}", exc_info=True
        )
        raise TrainUnrecoverable(
            f"Can't train model {model_id} for user {user_id}"
        ) from e


def _execute_estimation(
    user_id: int,
    model_id: int,
    track_to_estimate_path: pathlib.Path,
    model_type: ModelType,
) -> bool:
    """Load model and score track (internal API)"""
    try:
        model_store = config.get_model_store_path(user_id, model_id)

        if not model_store.model_workdir.exists():
            raise EstimationUnrecoverable(
                f"Model {model_id} not found for user {user_id}"
            )

        model = DualOneClassModel.load(model_store.model_workdir)

        bundled_profile = model_store.model_workdir / "essentia_profile.yaml"
        if not bundled_profile.exists():
            raise EstimationUnrecoverable(
                f"Bundled profile not found at {bundled_profile}. "
                f"Please retrain with /train."
            )

        current_embed_version = get_embed_version(profile_path=bundled_profile)
        if current_embed_version != model.embed_version:
            raise EstimationUnrecoverable(
                f"embed_version mismatch: model was trained with "
                f"'{model.embed_version}' but current pipeline produces "
                f"'{current_embed_version}'. Please retrain with /train."
            )

        extractor = prepare_extractor(profile_path=bundled_profile)
        X = extract_features_for_mp3(track_to_estimate_path, extractor).reshape(1, -1)

        scores = model.predict(X)

        is_recommended = model.decide(scores, model_type)

        logger.debug(
            f"Scores: like={scores['like']}, dislike={scores['dislike']}, "
            f"decision={is_recommended}"
        )

        return is_recommended
    except EstimationUnrecoverable:
        raise
    except Exception as e:
        logger.error(
            f"Estimation failed for track {track_to_estimate_path}: {e}", exc_info=True
        )
        raise EstimationUnrecoverable(f"Estimation failed: {e}") from e


async def estimate(
    user_id: int,
    chat_id: int,
    message_id: int,
    model_id: int,
    model_type: ModelType,
    bot_client: TelegramClient,
) -> bool:
    message = await get_message(chat_id, message_id, bot_client)

    tmp_dir = config.get_user_tmp_dir(user_id)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        track_to_estimate_path = pathlib.Path(tmp).joinpath("to-estimate.mp3")
        track_to_estimate_path.unlink(missing_ok=True)
        await message.download_media(file=track_to_estimate_path)
        is_recommended = await asyncio.get_running_loop().run_in_executor(
            config.get_estimation_executor(),
            _execute_estimation,
            user_id,
            model_id,
            track_to_estimate_path,
            model_type,
        )
        logger.info(f"{user_id=} {chat_id=} {message_id=}: {is_recommended=}")
        return is_recommended
