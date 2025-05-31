import asyncio
import json
import logging
import math
import pathlib
import pickle
import shutil
import tempfile
import time
from typing import Literal, Optional, TypeAlias, cast, Callable

import atomics
import polars as pl
import telethon
from linearboost import LinearBoostClassifier
from mutagen.mp3 import HeaderNotFoundError
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from soundfile import LibsndfileError
from telethon import TelegramClient
from telethon.tl import types
from telethon.tl.custom import Message
from telethon.tl.functions.channels import GetChannelsRequest
from telethon.tl.types import Chat, DocumentAttributeAudio

import config
from audio import features
from audio.features import extract_features_for_mp3
from dataset.persistent_dataset_processor import DataSetFromDataManager
from utils import unwrap_single_chat, get_message

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
        if not message:
            return False
        if not isinstance(message, (telethon.tl.types.Message, Message)):
            return False
        if isinstance(message, types.MessageMediaPhoto):
            return False
        if not hasattr(message, "media") or not hasattr(message.media, "document"):
            return False
        if not hasattr(
            message.media.document, "mime_type"
        ) or message.media.document.mime_type not in {"audio/mpeg", "audio/mp3"}:
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
        return f"Mp3Filter[min_length_seconds={self.min_length_seconds}, max_length_seconds={self.max_length_seconds}]"


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


async def obtain_latest_message_id(
    channel: Chat, bot_client: TelegramClient, step: int = 1000
) -> int:
    last_message_date = channel.date

    async def binary_search(index_range: list[int]) -> int:
        low, high = index_range[0], index_range[-1]
        mid = low
        while low <= high:
            mid = low + (high - low) // 2
            msg = await get_message(channel, mid, bot_client)
            if msg and msg.date >= last_message_date:  # target found
                return mid
            elif msg and msg.date < last_message_date:  # target is in the right half
                low = mid + 1
            elif not msg:  # target is in the left half
                high = mid - 1
            else:  # should not happen
                raise

        return mid

    max_message_range_start = 0
    max_message_range_end = step
    message = await get_message(channel, max_message_range_end, bot_client)
    while message and message.date < last_message_date:
        max_message_range_start = max_message_range_end
        max_message_range_end += step
        message = await get_message(channel, max_message_range_end, bot_client)

    if message and message.date >= last_message_date:
        return max_message_range_end
    return await binary_search(
        list(range(max_message_range_start, max_message_range_end + 1))
    )


async def download_audio_from_channel(
    user_id: int,
    channel_id: int,
    channel_type: Literal["liked", "disliked"],
    bot_client: TelegramClient,
    limit: int | None = None,
):
    channel = unwrap_single_chat(await bot_client(GetChannelsRequest(id=[channel_id])))
    if not channel:
        raise TrainUnrecoverable(f"Channel {channel_id} is not available")

    latest_message_id = await obtain_latest_message_id(channel, bot_client)
    ids = list(range(latest_message_id + 1))
    if limit:
        ids = ids[-limit:]
    start = time.time()
    async for message in bot_client.iter_messages(channel, ids=ids, reverse=True):
        got_message = time.time()
        if not FILTER.filter_message(message):
            if message:
                logging.info(
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

    # takeout_init_tries = 0
    # while takeout_init_tries < 3:
    #     takeout_init_tries += 1
    #     try:
    #         async with bot_client.takeout(channels=True) as takeout_client:
    #             async for message in takeout_client.iter_messages(channel):
    #                 if not FILTER.filter_message(message):
    #                     logging.info(
    #                         f"Message {message.stringify()} does not match {FILTER}, skipping"
    #                     )
    #                     continue
    #                 await save_track_if_not_exists(user_id, message, channel_type)
    #                 break
    #     except errors.TakeoutInitDelayError as e:
    #         await asyncio.sleep(e.seconds)
    #         await bot_client.end_takeout(True)


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
    except Exception as e:
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
        logger.debug(f"model {results_dir.name}: started to init ds manager")
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
                    if isinstance(e, (LibsndfileError, HeaderNotFoundError)):
                        logging.warning(
                            f"failed to get features for {row}, returning stub",
                            exc_info=e,
                        )
                        return cast(
                            RowType,
                            tuple([row] + [None] * (len(ds.row_schema) - 1)),
                        )

                    logging.error(
                        f"Unexpected error while processing row {row}, propagating",
                        exc_info=e,
                    )
                    raise

                return generate_features(audio_dir, row_id, int(is_liked), error_hook)

            ds.fill(analyze_track)
            logging.info(
                f"total feature generation calls/dataset_size stat: {counter.load()}/{ds.to_process_rows_count}"
            )
    return dataset_path


def train_model(user_id: int, model_id: int, model_type: str) -> config.Model:
    model_store_ctx = config.get_model_store_path(user_id, model_id)
    logger.debug(f"model {model_id}: got store ctx={model_store_ctx}")
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
    liked_dataframe = pl.scan_csv(liked_data)
    disliked_dataframe = pl.scan_csv(disliked_data)
    data = (
        pl.concat([liked_dataframe, disliked_dataframe])
        .collect(engine="streaming")
        .drop_nulls()
        .sample(fraction=1, shuffle=True)
    )
    data_stats = data.group_by(by=pl.col(LIKED_COLUMN_NAME)).agg(
        pl.col(LIKED_COLUMN_NAME).count()
    )
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
    if model_type == "similar":
        bad_fraction = data_stats.select(
            pl.col(LIKED_COLUMN_NAME)
            .filter(pl.col("by") == 1)
            .truediv(pl.col(LIKED_COLUMN_NAME).filter(pl.col("by") == 0))
        ).item(0, 0)
        model, model_accuracy = train_similar_model(
            train_data=train_data, test_data=test_data, bad_fraction=bad_fraction
        )
    elif model_type == "dissimilar":
        model, model_accuracy = train_dissimilar_model(
            train_data=train_data, test_data=test_data, nu=0.66, contamination_fraction=0.2
        )
    else:
        raise Exception(f"{model_type} is not supported")

    logging.info(f"Dumping model")
    model_file_path = model_store_ctx.model_workdir.joinpath(
        model_store_ctx.model_pickle_name
    )
    with model_file_path.open(mode="wb") as model_file:
        pickle.dump(model, model_file, protocol=5)
    logging.info(
        f"Dumped model user_id={model_store_ctx.user_id} model={model_store_ctx.model_id} to {model_file_path}"
    )
    model_stats_file_path = model_store_ctx.model_workdir.joinpath(
        model_store_ctx.model_stats_name
    )
    with model_stats_file_path.open("wt") as model_stats_file:
        model_stats_file.write(
            json.dumps(
                {
                    "model_type": model_type,
                    "accuracy": model_accuracy,
                    "liked_tracks_count": data_stats[LIKED_COLUMN_NAME][1],
                    "disliked_tracks_count": data_stats[LIKED_COLUMN_NAME][0],
                }
            )
        )
    return config.get_model(user_id, model_id)


def train_similar_model(
    *, train_data: pl.DataFrame, test_data: pl.DataFrame, bad_fraction: float
) -> tuple[object, float | int]:

    model = LinearBoostClassifier(
        algorithm="SAMME.R", class_weight={0: bad_fraction, 1: 1.0}
    )
    model.fit(
        train_data.select(pl.all().exclude(ID_COLUMN_NAME, LIKED_COLUMN_NAME)),
        train_data.select(pl.col(LIKED_COLUMN_NAME)),
    )
    y_predicted = model.predict(
        test_data.select(pl.all().exclude(ID_COLUMN_NAME, LIKED_COLUMN_NAME))
    )
    model_accuracy = accuracy_score(
        test_data.select(pl.col(LIKED_COLUMN_NAME)), y_predicted
    )

    return model, model_accuracy


def train_dissimilar_model(
    *, train_data: pl.DataFrame, test_data: pl.DataFrame, contamination_fraction: float
) -> tuple[object, float | int]:
    model = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
    # model = LocalOutlierFactor(novelty=True, contamination=contamination_fraction, metric="cosine")
    model = Pipeline([('scaler', StandardScaler()), ('clf', model)])
    positive_cases = train_data.filter(pl.col(LIKED_COLUMN_NAME) == 0)
    negative_cases = train_data.filter(pl.col(LIKED_COLUMN_NAME) == 1).limit(
        math.ceil(positive_cases.shape[0] * contamination_fraction)
    )
    one_class_train_data = pl.concat([positive_cases, negative_cases]).sample(
        fraction=1, shuffle=True
    )

    model.fit(
        one_class_train_data.select(pl.all().exclude(ID_COLUMN_NAME, LIKED_COLUMN_NAME))
    )
    one_class_test_data = test_data.with_columns(
        pl.when(pl.col(LIKED_COLUMN_NAME) == 0)
        .then(1)
        .otherwise(-1)
        .alias(LIKED_COLUMN_NAME)
    )
    y_predicted = model.predict(
        one_class_test_data.select(pl.all().exclude(ID_COLUMN_NAME, LIKED_COLUMN_NAME))
    )
    model_accuracy = accuracy_score(
        one_class_test_data.select(pl.col(LIKED_COLUMN_NAME)), y_predicted
    )
    print(f"{model_accuracy=}")
    return model, model_accuracy


async def prepare_model(
    user_id: int,
    bot_client: TelegramClient,
    model_id: int,
    model_type: Literal["similar", "dissimilar"],
    force: bool = False,
    limit: int | None = None,
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
            user_id,
            subscription.liked_tracks_channel_id,
            "liked",
            bot_client,
            limit,
        )
        await download_audio_from_channel(
            user_id,
            subscription.disliked_tracks_channel_id,
            "disliked",
            bot_client,
            limit,
        )
        # Run the blocking task in the executor
        model = await asyncio.get_running_loop().run_in_executor(
            config.training_threadpool, train_model, user_id, model_id, model_type
        )
        config.set_current_model_id(user_id, model.model_id)

    except TrainUnrecoverable as e:
        raise TrainUnrecoverable(
            f"Can't train model {model_id} for user {user_id}"
        ) from e


def execute_estimation(user_id: int, track_to_estimate_path: pathlib.Path) -> int:
    actual_model_id = config.get_current_model_id(user_id)
    if not actual_model_id:
        raise EstimationUnrecoverable(f"for user {user_id} no model version set")

    def error_hook(row: str, exc: Exception):
        logger.error(f"Error on processing {row}, propagating", exc_info=exc)
        raise exc

    data = generate_features(
        track_to_estimate_path.parent,
        track_to_estimate_path.stem,
        -1,
        error_hook,
    )

    if not data:
        raise EstimationUnrecoverable(
            f"No features for {track_to_estimate_path}. Skipping"
        )
    cached_model_id, model = _estimation_model_cache.get(user_id, (None, None))

    if not model or actual_model_id != cached_model_id:
        with config.get_model(user_id, actual_model_id).pickle_file_path.open(
            mode="rb"
        ) as model_data:
            _estimation_model_cache[user_id] = (
                actual_model_id,
                pickle.load(model_data),
            )
    _, model = _estimation_model_cache[user_id]
    return model.predict(
        pl.DataFrame([data], schema=ROW_SCHEMA).select(
            pl.all().exclude(ID_COLUMN_NAME, LIKED_COLUMN_NAME)
        )
    )[0]


async def estimate(
    user_id: int, chat_id: int, message_id: int, bot_client: TelegramClient
) -> int:
    message = await get_message(chat_id, message_id, bot_client)

    tmp_dir = config.get_user_tmp_dir(user_id)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        track_to_estimate_path = pathlib.Path(tmp).joinpath(f"to-estimate.mp3")
        track_to_estimate_path.unlink(missing_ok=True)
        await message.download_media(file=track_to_estimate_path)
        return await asyncio.get_running_loop().run_in_executor(
            config.estimation_threadpool,
            execute_estimation,
            user_id,
            track_to_estimate_path,
        )
