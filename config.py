import dataclasses
import json
import logging
import multiprocessing
import os
import pathlib
import re
from concurrent.futures.process import ProcessPoolExecutor
from typing import Optional

from dacite import from_dict

from models import ModelType

api_id = os.getenv("API_ID")
api_hash = os.getenv("API_HASH")
bot_token = os.getenv("BOT_TOKEN")
owner_user_id = int(os.getenv("OWNER_USER_ID", "0"))
data_path = pathlib.Path("data").resolve()
local_data_path = pathlib.Path("local_data").resolve()


def v_name(fstr: str) -> str:
    return fstr.split("=", 2)[0]


not_overridable_properties = {
    v_name(f"{api_id=}"),
    v_name(f"{api_hash=}"),
    v_name(f"{bot_token=}"),
    v_name(f"{owner_user_id=}"),
    v_name(f"{data_path=}"),
    v_name(f"{local_data_path=}"),
}

user_client_check_period_seconds = int(
    os.getenv("USER_CLIENT_CHECK_PERIOD_SECONDS", "10")
)
dialog_list_page_size = int(os.getenv("DIALOG_LIST_PAGE_SIZE", "10"))
estimation_post_way = os.getenv("ESTIMATION_POST_WAY", "reply")

max_training_workers = int(os.getenv("MAX_TRAINING_WORKERS", "2"))
max_estimation_workers = int(os.getenv("MAX_ESTIMATION_WORKERS", "2"))
min_track_length_seconds = int(os.getenv("MIN_TRACK_LENGTH_SECONDS", "90"))
max_track_length_seconds = int(os.getenv("MAX_TRACK_LENGTH_SECONDS", "480"))

model_knn_k_min = int(os.getenv("MODEL_KNN_K_MIN", "7"))
model_knn_k_max = int(os.getenv("MODEL_KNN_K_MAX", "19"))
model_knn_k_scale = float(os.getenv("MODEL_KNN_K_SCALE", "0.6530751049738679"))
model_gmm_components_max = int(os.getenv("MODEL_GMM_COMPONENTS_MAX", "28"))
model_gmm_min_points_per_component = int(
    os.getenv("MODEL_GMM_MIN_POINTS_PER_COMPONENT", "80")
)
model_outlier_threshold = float(os.getenv("MODEL_OUTLIER_THRESHOLD", "0.07"))
model_exclude_disliked_recall_target = float(
    os.getenv("MODEL_EXCLUDE_DISLIKED_RECALL", "0.80")
)
model_include_liked_recall_target = float(
    os.getenv("MODEL_INCLUDE_LIKED_RECALL", "0.775")
)
model_min_set_size = int(os.getenv("MODEL_MIN_SET_SIZE", "50"))
model_cv_folds = int(os.getenv("MODEL_CV_FOLDS", "5"))
model_max_imbalance_ratio = float(os.getenv("MODEL_MAX_IMBALANCE_RATIO", "3.0"))
model_preprocessor = os.getenv("MODEL_PREPROCESSOR", "standardize_select")
model_select_n_features = int(os.getenv("MODEL_SELECT_N_FEATURES", "64"))
model_like_preprocessor = os.getenv("MODEL_LIKE_PREPROCESSOR", "welch64")
model_dislike_preprocessor = os.getenv("MODEL_DISLIKE_PREPROCESSOR", "welch64")
model_decision_mode = os.getenv("MODEL_DECISION_MODE", "fused_diff")
model_fusion_weight = float(os.getenv("MODEL_FUSION_WEIGHT", "1.0"))
if model_decision_mode not in ("single", "fused_diff"):
    raise ValueError(
        f"MODEL_DECISION_MODE must be 'single' or 'fused_diff', "
        f"got {model_decision_mode!r}"
    )

model_knn_k = model_knn_k_max
model_gmm_components = model_gmm_components_max
worker_ack_timeout_seconds = int(os.getenv("WORKER_ACK_TIMEOUT_S", "120"))
segment_policy = os.getenv("SEGMENT_POLICY", "full")
panns_weights_path = pathlib.Path(
    os.getenv("PANNS_WEIGHTS_PATH", str(data_path / "panns_data" / "panns_cnn14.pth"))
)


def override():
    overrides = {}
    override_from = data_path.joinpath("config.py")
    if override_from.exists():
        exec(override_from.read_text(), None, overrides)
    for override_key, override_value in filter(
        lambda t: t[0] not in not_overridable_properties, overrides.items()
    ):
        globals()[override_key] = override_value


override()


def _setup_worker_logging():
    from core.logging import setup_logging

    setup_logging(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


_spawn_context = multiprocessing.get_context("spawn")
_training_executor: ProcessPoolExecutor | None = None
_estimation_executor: ProcessPoolExecutor | None = None


def get_training_executor() -> ProcessPoolExecutor:
    global _training_executor
    if _training_executor is None:
        _training_executor = ProcessPoolExecutor(
            max_workers=max_training_workers,
            mp_context=_spawn_context,
            initializer=_setup_worker_logging,
        )
    return _training_executor


def reset_training_executor() -> None:
    global _training_executor
    if _training_executor is not None:
        _training_executor.shutdown(wait=False, cancel_futures=True)
    _training_executor = None


def get_estimation_executor() -> ProcessPoolExecutor:
    global _estimation_executor
    if _estimation_executor is None:
        _estimation_executor = ProcessPoolExecutor(
            max_workers=max_estimation_workers,
            mp_context=_spawn_context,
            initializer=_setup_worker_logging,
        )
    return _estimation_executor


def reset_estimation_executor() -> None:
    global _estimation_executor
    if _estimation_executor is not None:
        _estimation_executor.shutdown(wait=False, cancel_futures=True)
    _estimation_executor = None


def get_existing_users() -> list[int]:
    return [
        int(user_data.name)
        for user_data in data_path.iterdir()
        if re.match("\\d+", user_data.name)
    ]


@dataclasses.dataclass
class UserChannels:
    liked_channel_id: int
    disliked_channel_id: int


def get_user_channels(user_id: int) -> Optional[UserChannels]:
    channels_file = data_path.joinpath(str(user_id)).joinpath("channels.json")
    if not channels_file.exists():
        return None
    return from_dict(
        data_class=UserChannels, data=json.loads(channels_file.read_text())
    )


def set_user_channels(user_id: int, channels: UserChannels):
    channels_file = data_path.joinpath(str(user_id)).joinpath("channels.json")
    channels_file.parent.mkdir(exist_ok=True)
    with channels_file.open(mode="wt") as f:
        f.write(json.dumps(dataclasses.asdict(channels)))


@dataclasses.dataclass
class Subscription:
    estimate_from_channel_id: int
    model_id: int
    model_type: ModelType = ModelType.INCLUDE_LIKED


def _serialize_subscriptions(subscriptions: list[Subscription]) -> list[dict]:
    result = []
    for s in subscriptions:
        d = dataclasses.asdict(s)
        d["model_type"] = s.model_type.name
        result.append(d)
    return result


def _deserialize_subscriptions(items: list[dict]) -> list[Subscription]:
    subscriptions = []
    for item in items:
        if isinstance(item.get("model_type"), str):
            item["model_type"] = ModelType.from_string(item["model_type"])
        elif isinstance(item.get("model_type"), int):
            item["model_type"] = ModelType(item["model_type"])
        subscriptions.append(Subscription(**item))
    return subscriptions


def get_subscriptions(user_id: int) -> list[Subscription]:
    subscriptions_file = data_path.joinpath(str(user_id)).joinpath("subscriptions.json")
    if not subscriptions_file.exists():
        return []
    return _deserialize_subscriptions(json.loads(subscriptions_file.read_text()))


def _save_subscriptions(user_id: int, subscriptions: list[Subscription]):
    subscriptions_file = data_path.joinpath(str(user_id)).joinpath("subscriptions.json")
    subscriptions_file.parent.mkdir(exist_ok=True)
    with subscriptions_file.open(mode="wt") as f:
        f.write(json.dumps(_serialize_subscriptions(subscriptions)))


def add_subscription(user_id: int, subscription: Subscription):
    subscriptions = get_subscriptions(user_id)
    subscriptions.append(subscription)
    _save_subscriptions(user_id, subscriptions)


def get_subscription_by_channel(
    user_id: int, channel_id: int
) -> Optional[Subscription]:
    subscriptions = get_subscriptions(user_id)
    for sub in subscriptions:
        if sub.estimate_from_channel_id == channel_id:
            return sub
    return None


def update_subscription_model(user_id: int, channel_id: int, model_id: int):
    subscriptions = get_subscriptions(user_id)
    for sub in subscriptions:
        if sub.estimate_from_channel_id == channel_id:
            sub.model_id = model_id
            _save_subscriptions(user_id, subscriptions)
            return


def update_subscription_model_type(
    user_id: int, channel_id: int, model_type: ModelType
):
    subscriptions = get_subscriptions(user_id)
    for sub in subscriptions:
        if sub.estimate_from_channel_id == channel_id:
            sub.model_type = model_type
            _save_subscriptions(user_id, subscriptions)
            return


def remove_subscription(user_id: int, channel_id: int):
    subscriptions = get_subscriptions(user_id)
    subscriptions = [
        sub for sub in subscriptions if sub.estimate_from_channel_id != channel_id
    ]
    _save_subscriptions(user_id, subscriptions)


def has_user_channels(user_id: int) -> bool:
    return get_user_channels(user_id) is not None


def get_subscribed_user_ids(channel_id: int) -> list[int]:
    return [
        user_id
        for user_id in get_existing_users()
        if get_subscription_by_channel(user_id, channel_id) is not None
    ]


@dataclasses.dataclass
class Model:
    model_id: int
    pickle_file_path: pathlib.Path
    liked_tracks_count: int
    disliked_tracks_count: int
    metrics_source: Optional[str] = None
    thresholds: Optional[dict] = None
    embed_version: Optional[str] = None
    outliers_removed_liked: Optional[int] = None
    outliers_removed_disliked: Optional[int] = None
    include_liked_tp: Optional[float] = None
    include_liked_tn: Optional[float] = None
    include_liked_fp: Optional[float] = None
    include_liked_fn: Optional[float] = None
    exclude_disliked_tp: Optional[float] = None
    exclude_disliked_tn: Optional[float] = None
    exclude_disliked_fp: Optional[float] = None
    exclude_disliked_fn: Optional[float] = None


_MODEL_FIELDS = {f.name for f in dataclasses.fields(Model)}


def get_models(user_id: int) -> list[Model]:
    models_path = data_path.joinpath(str(user_id)).joinpath("models")
    if not models_path.exists():
        return []
    return [
        model
        for model_path in models_path.iterdir()
        if model_path.is_dir()
        and model_path.joinpath("model.pkl").exists()
        and (model := get_model(user_id, int(model_path.stem))) is not None
    ]


def get_model(user_id: int, model_id: int) -> Optional[Model]:
    model_path = (
        data_path.joinpath(str(user_id)).joinpath("models").joinpath(str(model_id))
    )
    if not model_path.joinpath("model.pkl").exists():
        return None
    stats_path = model_path.joinpath("stats.json")
    if not stats_path.exists():
        return None
    model_stats = json.loads(stats_path.read_text())
    model_stats.pop("model_type", None)
    model_stats = {k: v for k, v in model_stats.items() if k in _MODEL_FIELDS}
    return Model(
        model_id=int(model_path.stem),
        pickle_file_path=model_path.joinpath("model.pkl"),
        **model_stats,
    )


@dataclasses.dataclass
class ModelStoreContext:
    user_id: int
    model_id: int
    model_workdir: pathlib.Path
    model_pickle_name: str
    model_stats_name: str


def get_model_store_path(user_id: int, model_id: int) -> ModelStoreContext:
    model_path = (
        data_path.joinpath(str(user_id)).joinpath("models").joinpath(str(model_id))
    )
    model_path.mkdir(parents=True, exist_ok=True)

    return ModelStoreContext(
        user_id=user_id,
        model_id=model_id,
        model_workdir=model_path,
        model_pickle_name="model.pkl",
        model_stats_name="stats.json",
    )


def get_liked_file_store_path(user_id: int) -> pathlib.Path:
    liked_path = data_path.joinpath(str(user_id)).joinpath("liked")
    liked_path.mkdir(exist_ok=True)
    return liked_path


def get_disliked_file_store_path(user_id: int) -> pathlib.Path:
    disliked_path = data_path.joinpath(str(user_id)).joinpath("disliked")
    disliked_path.mkdir(exist_ok=True)
    return disliked_path


def get_train_queue_path(user_id: int) -> pathlib.Path:
    return local_data_path.joinpath(str(user_id)).joinpath("train-queue.db")


def get_estimate_queue_path(user_id: int) -> pathlib.Path:
    return local_data_path.joinpath(str(user_id)).joinpath("estimate-queue.db")


def get_user_tmp_dir(user_id: int) -> pathlib.Path:
    tmp_path = data_path.joinpath(str(user_id)).joinpath("tmp")
    tmp_path.mkdir(exist_ok=True)
    return tmp_path


def get_feature_store_root(user_id: int) -> pathlib.Path:
    root = data_path / str(user_id) / "features"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_job_store_path(user_id: int) -> pathlib.Path:
    job_dir = local_data_path / str(user_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir / "jobs.duckdb"


def get_training_tmp_dir(user_id: int) -> pathlib.Path:
    tmp_dir = local_data_path / str(user_id) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


profiles_root = data_path.joinpath("profiles")


def get_allowed_to_use_user_ids() -> list[int]:
    whitelist_path = data_path.joinpath("user_whitelist")
    if not whitelist_path.exists():
        return []
    return [int(user.name) for user in whitelist_path.iterdir()]
