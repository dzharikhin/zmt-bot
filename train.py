import asyncio
import json
import logging
import pathlib
import pickle
import shutil
import tempfile
from typing import Literal, Optional, TypeAlias, cast, Callable

import atomics
import polars as pl
import telethon
from linearboost import LinearBoostClassifier
from mutagen.mp3 import HeaderNotFoundError
from sklearn.metrics import accuracy_score
from soundfile import LibsndfileError
from telethon import TelegramClient
from telethon.tl.custom import Message
from telethon.tl.functions.channels import GetChannelsRequest
from telethon.tl.types import Chat, DocumentAttributeAudio
from telethon.tl.types.messages import Chats

import config
from audio import features
from audio.features import extract_features_for_mp3
from dataset.persistent_dataset_processor import DataSetFromDataManager

logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


ID_COLUMN_NAME = "track_id"
LIKED_COLUMN_NAME = "is_liked"

_estimation_model_cache = {}


def load_genre_model() -> tuple[Optional[object], list[tuple[str, type]]]:
    model_path = config.data_path.joinpath("genre_model.pickle")
    if not model_path.exists():
        logger.warning(
            f"{model_path} is not found. Training and estimation without genre data"
        )
        return None, []
    return pickle.load(model_path), [("genre", int)]


GENRE_MODEL, genre_schema = load_genre_model()


RowType: TypeAlias = tuple[
    *features.AudioFeaturesType, *[e[1] for e in genre_schema], int
]  # *[audio_features], cluster?, is_liked

ROW_SCHEMA = (
    features.AUDIO_FEATURE_TYPE_SCHEMA
    + [e[1] for e in genre_schema]
    + [(LIKED_COLUMN_NAME, int)]
)


class TrainUnrecoverable(Exception):
    pass


class EstimationUnrecoverable(Exception):
    pass


class Mp3Filter:

    def __init__(self, **params):
        self.min_length_seconds = params["min_seconds"]
        self.max_length_seconds = params["max_seconds"]

    def filter_message(self, message) -> bool:
        if not isinstance(message, (telethon.tl.types.Message, Message)):
            return False
        if not message.media or not message.media.document:
            return False
        if message.media.document.mime_type not in {"audio/mpeg", "audio/mp3"}:
            return False
        if not message.media.document.attributes or not [
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
        return f"Mp3Filter[min_length_seconds={self.min_length_seconds}, max_length_seconds={self.max_length_seconds}]"


FILTER = Mp3Filter(
    min_seconds=config.min_track_length_seconds,
    max_seconds=config.max_track_length_seconds,
)


def unwrap_single_chat(chat: Chats) -> Optional[Chat]:
    if not chat or not chat.chats:
        return None
    return chat.chats[0]


async def save_track_if_not_exists(
    user_id: int, message: Message, channel_type: Literal["liked", "disliked"]
):
    tracks_folder = (
        config.get_disliked_file_store_path(user_id)
        if channel_type == "disliked"
        else config.get_liked_file_store_path(user_id)
    )
    file_path = tracks_folder.joinpath(f"{message.document.id}.mp3")
    if not file_path.exists():
        await message.download_media(file=file_path)


async def download_audio_from_channel(
    user_id: int,
    channel_id: int,
    channel_type: Literal["liked", "disliked"],
    bot_client: TelegramClient,
):
    channel = unwrap_single_chat(
        await bot_client(GetChannelsRequest(id=[int(channel_id)]))
    )
    if not channel:
        raise TrainUnrecoverable(f"Channel {channel_id} is not available")

    await bot_client.end_takeout(False)
    async with bot_client.takeout(channels=True) as takeout_client:
        async for message in takeout_client.iter_messages(channel):
            if not FILTER.filter_message(message):
                logging.info(
                    f"Message {message.stringify()} does not match {FILTER}, skipping"
                )
                continue
            await save_track_if_not_exists(user_id, message, channel_type)


def generate_features(
    audio_dir: pathlib.Path,
    row_id: str,
    liked_class: int,
    error_hook: Callable[[str, Exception], RowType],
) -> RowType:
    try:
        audio_features = extract_features_for_mp3(
            track_id=row_id,
            mp3_path=audio_dir.joinpath(f"{row_id}.mp3"),
        )
        return cast(
            RowType,
            (
                *audio_features,
                *([GENRE_MODEL.predict(audio_features[1:-2])] if GENRE_MODEL else []),
                liked_class,
            ),
        )
    except (LibsndfileError, HeaderNotFoundError) as e:
        row = error_hook(row_id, e)
    return row


def prepare_audio_features_dataset(
    *, user_id: int, results_dir: pathlib.Path, is_liked: bool
) -> pathlib.Path:
    audio_dir = (
        config.get_disliked_file_store_path(user_id)
        if not is_liked
        else config.get_liked_file_store_path(user_id)
    )
    counter = atomics.atomic(width=4, atype=atomics.INT)
    dataset_path = results_dir.joinpath(f"audio_features_dataset-{int(is_liked)}.csv")

    fails_path = results_dir.joinpath(f"{dataset_path.stem}-processing_failed.csv")
    fails_path.unlink(missing_ok=True)

    tmp_dir = config.get_user_tmp_dir(user_id)

    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        dataset_manager = DataSetFromDataManager(
            dataset_path,
            row_schema=ROW_SCHEMA,
            index_generator=(f.stem for f in audio_dir.iterdir() if f.is_file()),
            intermediate_results_dir=pathlib.Path(tmp),
            batch_size=10,
        )
        with dataset_manager as ds:

            def analyze_track(row_id: str) -> RowType:
                def error_hook(row: str, e: Exception) -> RowType:
                    logging.error(
                        f"failed to get features for {row}, returning stub",
                        exc_info=e,
                    )
                    return cast(
                        RowType,
                        tuple([row] + [None] * (len(ds.row_schema) - 1)),
                    )

                return generate_features(audio_dir, row_id, int(is_liked), error_hook)

            ds.fill(analyze_track)
            logging.info(
                f"total feature generation calls/dataset_size stat: {counter.load()}/{ds.to_process_rows_count}"
            )
    return dataset_path


def train_model(user_id: int, model_id: int) -> config.Model:
    model_store_ctx = config.get_model_store_path(user_id, model_id)
    liked_data = prepare_audio_features_dataset(
        user_id=user_id,
        results_dir=model_store_ctx.model_workdir,
        is_liked=True,
    )
    disliked_data = prepare_audio_features_dataset(
        user_id=user_id,
        results_dir=model_store_ctx.model_workdir,
        is_liked=False,
    )
    data = (
        pl.concat([pl.scan_csv(liked_data), pl.scan_csv(disliked_data)])
        .collect()
        .sample(fraction=1, shuffle=True)
    )
    data_stats = (
        data.group_by(by=pl.col(LIKED_COLUMN_NAME))
        .agg(pl.col(ID_COLUMN_NAME).count())
        .to_dict(as_series=False)
    )
    bad_weight = data_stats[LIKED_COLUMN_NAME][1] / data_stats[LIKED_COLUMN_NAME][0]
    logging.info(f"Splitting data into train/test")
    test_track_ids = (
        data.group_by(pl.col(LIKED_COLUMN_NAME))
        .agg(
            pl.col(ID_COLUMN_NAME).sample(
                fraction=0.3, with_replacement=False, shuffle=True
            )
        )
        .explode(pl.col(ID_COLUMN_NAME))
        .select(ID_COLUMN_NAME)
    )

    test_data = data.filter(
        pl.col(ID_COLUMN_NAME).is_in(test_track_ids.get_column(ID_COLUMN_NAME))
    )
    train_data = data.filter(
        pl.col(ID_COLUMN_NAME).is_in(test_track_ids.get_column(ID_COLUMN_NAME)).not_()
    )

    model = LinearBoostClassifier(
        algorithm="SAMME.R", class_weight={0: bad_weight, 1: 1.0}
    )
    model.fit(
        train_data.select(pl.all().exclude(ID_COLUMN_NAME, LIKED_COLUMN_NAME)),
        train_data.select(pl.col(LIKED_COLUMN_NAME)),
    )
    y_predicted = model.predict(
        test_data.select(pl.all().exclude(ID_COLUMN_NAME, LIKED_COLUMN_NAME))
    )
    accuracy = accuracy_score(test_data.select(pl.col(LIKED_COLUMN_NAME)), y_predicted)
    logging.info(f"Dumping model")
    model_file_path = model_store_ctx.model_workdir.joinpath(
        model_store_ctx.model_pickle_name
    )
    with model_file_path.open(mode="wb") as model_file:
        pickle.dump(model, model_file, protocol=5)
    logging.info(
        f"Dumped model user_id={user_id} model={model_id} to {model_file_path}"
    )
    model_stats_file_path = model_store_ctx.model_workdir.joinpath(
        model_store_ctx.model_stats_name
    )
    with model_stats_file_path.open("wt") as model_stats_file:
        model_stats_file.write(
            json.dumps(
                {
                    "accuracy": accuracy,
                    "liked_tracks_count": data_stats[LIKED_COLUMN_NAME][1],
                    "disliked_tracks_count": data_stats[LIKED_COLUMN_NAME][0],
                }
            )
        )
    return config.get_model(user_id, model_id)


async def prepare_model(
    user_id: int, bot_client: TelegramClient, model_id: int, force: bool = False
):
    try:
        subscription = config.get_subscription(user_id)
        if not subscription:
            raise TrainUnrecoverable(
                f"User {user_id} is not subscribed - nothing to train"
            )

        if force:
            shutil.rmtree(config.get_liked_file_store_path(user_id))
            shutil.rmtree(config.get_disliked_file_store_path(user_id))

        await download_audio_from_channel(
            user_id, subscription.liked_tracks_channel_id, "liked", bot_client
        )
        await download_audio_from_channel(
            user_id, subscription.disliked_tracks_channel_id, "disliked", bot_client
        )
        # Run the blocking task in the executor
        model = await asyncio.get_running_loop().run_in_executor(
            config.train_threadpool, train_model, user_id, model_id
        )
        config.set_current_model_id(user_id, model.model_id)

    except TrainUnrecoverable as e:
        raise TrainUnrecoverable(
            f"Can't train model {model_id} for user {user_id}"
        ) from e


async def estimate(
    user_id: int, chat_id: int, message_id: int, bot_client: TelegramClient
) -> int:
    actual_model_id = config.get_current_model_id(user_id)
    if not actual_model_id:
        raise EstimationUnrecoverable(f"for user {user_id} no model version set")
    try:
        message = (await bot_client.get_messages(chat_id, ids=[message_id]))[0]

        tmp_dir = config.get_user_tmp_dir(user_id)
        with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
            track_to_estimate_path = pathlib.Path(tmp).joinpath(f"to-estimate.mp3")
            await message.download_media(file=track_to_estimate_path)

            def error_hook(row: str, exc: Exception):
                raise exc

            data = generate_features(
                track_to_estimate_path.parent,
                track_to_estimate_path.name,
                -1,
                error_hook,
            )

        if not data:
            raise EstimationUnrecoverable(
                f"No features for {track_to_estimate_path}. Skipping"
            )
        cached_model_id, model = _estimation_model_cache.get(user_id)

        if not model or actual_model_id != cached_model_id:
            with config.get_model(user_id, actual_model_id).pickle_file_path.open(
                mode="rb"
            ) as model_data:
                _estimation_model_cache[user_id] = actual_model_id, pickle.load(
                    model_data
                )
        _, model = _estimation_model_cache[user_id]
        return model.predict(
            pl.DataFrame([data], schema=ROW_SCHEMA).select(
                pl.all().exclude(ID_COLUMN_NAME, LIKED_COLUMN_NAME)
            )
        )[0]
    except IndexError as e:
        raise EstimationUnrecoverable(
            f"Cannot estimate track chat_id={chat_id} message_id={message_id}"
        ) from e
