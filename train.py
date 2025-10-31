import asyncio
import enum
import json
import logging
import pathlib
import pickle
import shutil
import tempfile
import time
import typing
from typing import Literal, cast

import numpy as np
import polars as pl
import telethon
from clustpy.partition import GMeans
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_is_fitted
from skopt import BayesSearchCV, space
from telethon import TelegramClient
from telethon.tl import types
from telethon.tl.custom import Message
from telethon.tl.functions.channels import GetChannelsRequest
from telethon.tl.types import DocumentAttributeAudio

import config
from audio.features import (
    extract_features_for_mp3,
    prepare_extractor,
    AudioFeatures,
    key_mapping,
    scale_mapping,
    key_columns,
    scale_columns,
)
from bot_utils import unwrap_single_chat, get_message, obtain_latest_message_id
from dataclass_utils import (
    create_wrapper_type,
    unwrap_to_dict,
    convert_type,
    get_field_shape_map,
)
from dataset.persistent_dataset_processor import (
    DataFrameBuilder,
    build_successfully_processed_dataframe,
)

logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

_estimation_model_cache = {}

LIKED_COLUMN_NAME = "liked"
ROW_ID_COLUMN_NAME = "row_id"

MappedAudioFeatures = create_wrapper_type(
    AudioFeatures,
    "MappedAudioFeatures",
    {np.ndarray: list[float], np.float32: float},
    {LIKED_COLUMN_NAME: int},
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


class ModelType(enum.IntEnum):
    INCLUDE_LIKED = 1
    EXCLUDE_DISLIKED = 0

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return ModelType[s]
        except KeyError:
            raise ValueError()


class NoCV(BaseCrossValidator):

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        yield np.arange(X.shape[0]), np.arange(X.shape[0])


class LabelScorerUnsupervised:

    def __init__(
        self, y_data, score_type: ModelType, *, driving_metric_name, main_metric_name
    ):
        super().__init__()
        self.score_type = score_type
        self.y_data = y_data
        self.main_metric_name = main_metric_name
        self.driving_metric_name = driving_metric_name

    def __call__(self, estimator, X, labels_true=None):
        labels = self.get_labels(estimator)
        return target_cluster_data_coverage_fraction_score(
            self.score_type,
            self.y_data,
            labels,
            main_metric_name=self.main_metric_name,
            guiding_metric_name=self.driving_metric_name,
        )

    class HasLabels(typing.Protocol):
        labels_: np.ndarray[tuple[int], np.int32]

    @staticmethod
    def get_labels(estimator: Pipeline | HasLabels):
        """Gets the cluster labels from an estimator or pipeline."""
        if isinstance(estimator, Pipeline):
            check_is_fitted(estimator.steps[-1][1], ["labels_"])
            labels = estimator.steps[-1][1].labels_
        else:
            check_is_fitted(estimator, ["labels_"])
            labels = estimator.labels_
        return labels


class CustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, delegate, target_clusters, model_type):
        self.delegate = delegate
        self.target_clusters = target_clusters
        self.model_type = model_type
        self._classify_vectorized = np.vectorize(self.classify)

    def classify(self, elem):
        if self.model_type == ModelType.INCLUDE_LIKED:
            return 1 if elem in self.target_clusters else 0
        elif self.model_type == ModelType.EXCLUDE_DISLIKED:
            return 0 if elem in self.target_clusters else 1
        else:
            raise Exception(f"Unsupported type {self.model_type}")

    def fit(self, X, y):
        self.delegate.fit(X, y)
        return self

    def predict(self, X):
        predicted_cluster = self.delegate.predict(X)
        return self._classify_vectorized(predicted_cluster)


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

    # takeout_init_tries = 0
    # while takeout_init_tries < 3:
    #     takeout_init_tries += 1
    #     try:
    #         async with bot_client.takeout(channels=True) as takeout_client:
    #             async for message in takeout_client.iter_messages(channel):
    #                 if not FILTER.filter_message(message):
    #                     logger.info(
    #                         f"Message {message.stringify()} does not match {FILTER}, skipping"
    #                     )
    #                     continue
    #                 await save_track_if_not_exists(user_id, message, channel_type)
    #                 break
    #     except errors.TakeoutInitDelayError as e:
    #         await asyncio.sleep(e.seconds)
    #         await bot_client.end_takeout(True)


def _unpack_data(data: pl.LazyFrame, metric_sizes) -> pl.LazyFrame:
    for field_name, _ in filter(lambda i: i[1], metric_sizes.items()):
        data = data.with_columns(
            pl.col(field_name).list.mean().alias(f"{field_name}_mean"),
            pl.col(field_name).list.std().alias(f"{field_name}_std"),
            pl.col(field_name).list.min().alias(f"{field_name}_min"),
            pl.col(field_name).list.max().alias(f"{field_name}_max"),
            pl.col(field_name).list.var().alias(f"{field_name}_var"),
        ).drop(field_name)
    return data


def _remove_outliers_in_parts(
    data: pl.LazyFrame, outliers_fraction, liked
) -> pl.DataFrame:
    raw_data_part = (
        data.filter(pl.col(LIKED_COLUMN_NAME) == liked).select(
            pl.all().exclude(LIKED_COLUMN_NAME, ROW_ID_COLUMN_NAME)
        )
    ).collect(engine="streaming")
    return (
        data.filter(pl.col(LIKED_COLUMN_NAME) == liked)
        .with_columns(
            pl.Series(
                name="outlier",
                values=OneClassSVM(nu=outliers_fraction).fit_predict(raw_data_part),
            )
        )
        .filter(pl.col("outlier") != -1)
        .select(pl.all().exclude("outlier"))
        .collect(engine="streaming")
    )


def _prepare_data_for_processing(
    data: pl.LazyFrame, outliers_fraction: float
) -> pl.DataFrame:
    data = data.fill_null(0.0).fill_nan(0.0)
    data_liked_clean = _remove_outliers_in_parts(data, outliers_fraction, 1)
    data_disliked_clean = _remove_outliers_in_parts(data, outliers_fraction, 0)

    return pl.concat([data_liked_clean, data_disliked_clean]).select(
        pl.all().exclude(ROW_ID_COLUMN_NAME)
    )


def _string_values_to_numbers(data: pl.LazyFrame) -> pl.LazyFrame:
    return data.with_columns(
        *[
            pl.col(column_name).replace_strict(
                key_mapping,
                return_dtype=pl.Struct(
                    {
                        f"{column_name}-sin": pl.Float64,
                        f"{column_name}-cos": pl.Float64,
                    }
                ),
            )
            for column_name in key_columns
        ],
        *[
            pl.col(column_name).replace_strict(scale_mapping, return_dtype=pl.Int64)
            for column_name in scale_columns
        ],
    ).unnest(key_columns)


_extractor = prepare_extractor()


def _convert(
    audio_path: pathlib.Path, is_liked: int, track_id: str, **task_id_args
) -> MappedAudioFeatures:
    features_row = extract_features_for_mp3(audio_path.joinpath(track_id), _extractor)
    target = convert_type(
        features_row, MappedAudioFeatures, {LIKED_COLUMN_NAME: int(is_liked)}
    )
    logger.debug(f"[{task_id_args=},{is_liked=}] extracted features for {track_id=}")
    return target


def train_model(user_id: int, model_id: int, model_type: ModelType) -> config.Model:
    model_store_ctx = config.get_model_store_path(user_id, model_id)
    logger.debug(f"model {model_id}: got store ctx={model_store_ctx}")
    return train_model_inner(
        model_store_ctx,
        model_type,
        config.get_user_tmp_dir(user_id),
        config.get_disliked_file_store_path(user_id),
        config.get_liked_file_store_path(user_id),
    )


def train_model_inner(
    model_store_ctx: config.ModelStoreContext,
    model_type: ModelType,
    tmp_dir: pathlib.Path,
    disliked_audio_path: pathlib.Path,
    liked_audio_path: pathlib.Path,
) -> config.Model:

    def liked_feature_extractor(track_id: str) -> MappedAudioFeatures:
        return _convert(
            liked_audio_path,
            int(True),
            track_id,
            user_id=model_store_ctx.user_id,
            model_id=model_store_ctx.model_id,
        )

    def disliked_feature_extractor(track_id: str) -> MappedAudioFeatures:
        return _convert(
            disliked_audio_path,
            int(False),
            track_id,
            user_id=model_store_ctx.user_id,
            model_id=model_store_ctx.model_id,
        )

    def stats(progress_state: DataFrameBuilder.ProgressStat):
        logger.info(
            f"[{model_store_ctx.user_id=},{model_store_ctx.model_id=}] {progress_state.succeed=}/{progress_state.failed=}/{progress_state.data_size=}"
        )

    with (
        DataFrameBuilder(
            working_dir=tmp_dir.joinpath("disliked_dir"),
            index_generator=DataFrameBuilder.GeneratingParams(
                generator=(
                    f.name for f in disliked_audio_path.iterdir() if f.is_file()
                ),
                result_schema=(ROW_ID_COLUMN_NAME, pl.String),
            ),
            mappers=[disliked_feature_extractor],
            batch_size=25,
            cleanup_on_exit=False,
            progress_tracker=stats,
        ) as disliked_frame_result,
        DataFrameBuilder(
            working_dir=tmp_dir.joinpath("liked_dir"),
            index_generator=DataFrameBuilder.GeneratingParams(
                generator=(f.name for f in liked_audio_path.iterdir() if f.is_file()),
                result_schema=(ROW_ID_COLUMN_NAME, pl.String),
            ),
            mappers=[liked_feature_extractor],
            batch_size=25,
            cleanup_on_exit=False,
            progress_tracker=stats,
        ) as liked_frame_result,
    ):
        data = build_successfully_processed_dataframe(
            [liked_frame_result, disliked_frame_result]
        )
        data = _string_values_to_numbers(data)
        shape_map = get_field_shape_map(AudioFeatures)
        data = _unpack_data(data, shape_map)
        prepared_data = _prepare_data_for_processing(
            data, config.model_data_contamination_fraction
        )

        return train_model_on_data(prepared_data, model_type, model_store_ctx)


def target_cluster_data_coverage_fraction_score(
    score_type: ModelType,
    y_data: pl.DataFrame,
    labels,
    *,
    main_metric_name,
    guiding_metric_name,
):
    cluster_column_name = "cluster"
    cluster_column = pl.Series(name=cluster_column_name, values=labels)
    clustered_data = y_data.with_columns(cluster_column)
    target_clusters = _get_target_clusters(clustered_data, score_type)
    model_accuracy = {
        main_metric_name: 0.0,
        guiding_metric_name: 0.0,
    }
    if target_clusters:
        cluster_coverage_fraction_score = (
            clustered_data.filter(pl.col(LIKED_COLUMN_NAME) == score_type.value)
            .select(pl.col(cluster_column_name))
            .select(
                pl.col(cluster_column_name)
                .filter(pl.col(cluster_column_name).is_in(target_clusters))
                .count()
                / pl.col(cluster_column_name).count(),
                # pl.col("cluster")
                # .filter(pl.col("cluster").is_in(target_clusters))
                # .count()
                # .alias("outliers"),
                # pl.col("cluster").count().alias("total"),
            )
            .item(0, 0)
        )
        max_cluster_size = cluster_column.value_counts().max().item(0, 1)
        average_cluster_size = y_data.shape[0] / cluster_column.n_unique()
        group_clusters_count = (
            cluster_column.value_counts().filter(pl.nth(1) > 1).shape[0]
        )

        if (
            hasattr(config, "model_metric_guide")
            and config.model_metric_guide == "algebraic_weighted"
        ):
            coef = 5.0
            guiding_score = sum(
                scores := [
                    cluster_coverage_fraction_score,
                    group_clusters_count / y_data.shape[0],
                    coef * max_cluster_size / y_data.shape[0],
                ]
            ) / (len(scores) + coef)
        else:
            guiding_score = sum(
                scores := [
                    cluster_coverage_fraction_score,
                    group_clusters_count / y_data.shape[0] * average_cluster_size,
                    max_cluster_size / y_data.shape[0] * average_cluster_size,
                ]
            ) / (len(scores) + 2 * average_cluster_size)

        model_accuracy = {
            main_metric_name: cluster_coverage_fraction_score,
            guiding_metric_name: guiding_score,
        }
    return model_accuracy


def _get_target_clusters(clustered_like_value: pl.DataFrame, score_type):
    target_fraction_column_name = f"{score_type.name}_fraction"
    cluster_member_coverage_fraction = clustered_like_value.group_by(pl.last()).agg(
        (
            pl.last().filter(pl.col(LIKED_COLUMN_NAME) == score_type.value).count()
            / pl.last().count()
        ).alias(target_fraction_column_name),
        # pl.col("cluster")
        # .filter(pl.col(LIKED_COLUMN_NAME) == 0)
        # .count()
        # .alias("disliked"),
        # pl.col("cluster").count().alias("all"),
    )
    target_clusters = (
        cluster_member_coverage_fraction.filter(
            pl.col(target_fraction_column_name)
            > config.model_cluster_target_coverage_threshold
        )
        .to_series(0)
        .to_list()
    )
    return target_clusters


def train_model_on_data(
    data: pl.DataFrame, model_type, model_store_ctx
) -> config.Model:

    x = data.select(pl.all().exclude(LIKED_COLUMN_NAME))
    y = data.select(pl.col(LIKED_COLUMN_NAME))

    max_clusters = (
        int(data.shape[0] * config.model_max_cluster_limit)
        if 0 < config.model_max_cluster_limit <= 1
        else config.model_max_cluster_limit
    )
    grid = {
        "significance": space.Real(0.001, 0.2),
        "n_split_trials": space.Integer(10, max(max_clusters, 11)),
        "n_clusters_init": space.Integer(10, 25),
        "max_n_clusters": space.Integer(
            30,
            max(max_clusters, 30),
        ),
    }

    driving_metric_name = "guiding_score"
    main_metric_name = f"{model_type.name.lower()}_cluster_coverage"
    scorer = LabelScorerUnsupervised(
        y,
        model_type,
        driving_metric_name=driving_metric_name,
        main_metric_name=main_metric_name,
    )

    search = BayesSearchCV(
        estimator=GMeans(),
        search_spaces=grid,
        n_iter=config.model_optimization_iterations,
        scoring=scorer,
        error_score=0.0,
        cv=NoCV(),
        refit=driving_metric_name,
        verbose=0,
        n_points=1,
    )

    pipeline = make_pipeline(
        # PCA(),
        StandardScaler(),
        MinMaxScaler(),
        search,
    )
    pipeline.fit(x)

    best_target_cluster_data_coverage_fraction = search.cv_results_[
        f"mean_test_{main_metric_name}"
    ][search.best_index_]

    best_score_clusterer_pipeline = Pipeline(
        [*pipeline.steps[:-1], ("estimator", search.best_estimator_)]
    )
    cluster_column = pl.Series(
        name="cluster",
        values=LabelScorerUnsupervised.get_labels(search.best_estimator_),
    )

    model = CustomClassifier(
        best_score_clusterer_pipeline,
        _get_target_clusters(y.with_columns(cluster_column), model_type),
        model_type,
    )
    logger.info(f"Dumping model")
    model_file_path = model_store_ctx.model_workdir.joinpath(
        model_store_ctx.model_pickle_name
    )
    with model_file_path.open(mode="wb") as model_file:
        pickle.dump(model, model_file, protocol=5)
    logger.info(
        f"Dumped model user_id={model_store_ctx.user_id} model={model_store_ctx.model_id} to {model_file_path}"
    )
    model_stats_file_path = model_store_ctx.model_workdir.joinpath(
        model_store_ctx.model_stats_name
    )
    with model_stats_file_path.open("wt") as model_stats_file:
        data_stats = dict(data.group_by(pl.col(LIKED_COLUMN_NAME)).len().rows())
        model_stats_file.write(
            json.dumps(
                {
                    "model_type": model_type,
                    "accuracy": best_target_cluster_data_coverage_fraction,
                    "liked_tracks_count": data_stats[True],
                    "disliked_tracks_count": data_stats[False],
                }
            )
        )
    return config.get_model(model_store_ctx.user_id, model_store_ctx.model_id)


async def prepare_model(
    user_id: int,
    bot_client: TelegramClient,
    model_id: int,
    model_type: ModelType,
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


def execute_estimation(user_id: int, track_to_estimate_path: pathlib.Path) -> bool:
    actual_model_id = config.get_current_model_id(user_id)
    if not actual_model_id:
        raise EstimationUnrecoverable(f"for user {user_id} no model version set")

    cached_model_id, model = _estimation_model_cache.get(user_id, (None, None))

    if not model or actual_model_id != cached_model_id:
        model_entry = config.get_model(user_id, actual_model_id)
        _estimation_model_cache[user_id] = (
            actual_model_id,
            _load_model(model_entry.pickle_file_path),
        )
    _, model = _estimation_model_cache[user_id]
    return estimate_inner(model, track_to_estimate_path)


def _load_model(model_file: pathlib.Path) -> typing.Any:
    with model_file.open(mode="rb") as model_data:
        return pickle.load(model_data)


def estimate_inner(model, track_to_estimate_path: pathlib.Path) -> int:
    try:
        features_row = extract_features_for_mp3(track_to_estimate_path, _extractor)
        data = pl.from_dicts([unwrap_to_dict(features_row)]).lazy()
        data = _string_values_to_numbers(data)
        shape_map = get_field_shape_map(AudioFeatures)
        data = _unpack_data(data, shape_map)
        data = data.fill_null(0.0).fill_nan(0.0).collect(engine="streaming")
    except Exception as e:
        raise EstimationUnrecoverable(
            f"Failed to get features for {track_to_estimate_path}. Skipping"
        ) from e
    if data.is_empty():
        raise EstimationUnrecoverable(
            f"No features for {track_to_estimate_path}. Skipping"
        )

    return bool(model.predict(data)[0])


async def estimate(
    user_id: int, chat_id: int, message_id: int, bot_client: TelegramClient
) -> bool:
    message = await get_message(chat_id, message_id, bot_client)

    tmp_dir = config.get_user_tmp_dir(user_id)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        track_to_estimate_path = pathlib.Path(tmp).joinpath(f"to-estimate.mp3")
        track_to_estimate_path.unlink(missing_ok=True)
        await message.download_media(file=track_to_estimate_path)
        is_recommended = await asyncio.get_running_loop().run_in_executor(
            config.estimation_threadpool,
            execute_estimation,
            user_id,
            track_to_estimate_path,
        )
        logger.info(f"{user_id=} {message_id=}: {is_recommended=}")
        return is_recommended


if __name__ == "__main__":
    model_path = pathlib.Path("test/test_model")
    model_path.mkdir(parents=True, exist_ok=True)
    model_store_ctx = config.ModelStoreContext(
        user_id=123,
        model_id=1,
        model_workdir=model_path,
        model_pickle_name=f"test.pickle",
        model_stats_name="stats.json",
    )
    model_type = ModelType.EXCLUDE_DISLIKED
    tmp_dir = pathlib.Path("test/tmp")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    disliked_path = pathlib.Path("data/disliked_short")
    liked_path = pathlib.Path("data/liked_short")
    train_model_inner(model_store_ctx, model_type, tmp_dir, disliked_path, liked_path)

    model = _load_model(
        pathlib.Path(
            model_store_ctx.model_workdir.joinpath(model_store_ctx.model_pickle_name)
        )
    )
    liked_score = estimate_inner(model, next(liked_path.iterdir()))
    print(f"liked track score: {liked_score}")
    disliked_score = estimate_inner(model, next(liked_path.iterdir()))
    print(f"disliked track score: {disliked_score}")
