import dataclasses
import json
import math
import multiprocessing
import os
import pathlib
import re
from concurrent.futures.process import ProcessPoolExecutor
from typing import Optional, Literal

from dacite import from_dict

api_id = os.getenv("API_ID")
api_hash = os.getenv("API_HASH")
bot_token = os.getenv("BOT_TOKEN")
owner_user_id = int(os.getenv("OWNER_USER_ID", "0"))
data_path = pathlib.Path("data")
local_data_path = pathlib.Path("local_data")


def v_name(fstr: str) -> str:
    return fstr.split("=", 2)[0]


not_overridable_properties = {
    v_name(f"{bot_token=}"),
    v_name(f"{owner_user_id=}"),
    v_name(f"{data_path=}"),
    v_name(f"{local_data_path=}"),
}

user_client_check_period_seconds = 10
dialog_list_page_size = 10
max_training_workers = 10
max_estimation_workers = 10
min_track_length_seconds = 60
max_track_length_seconds = 480
test_samples_fraction = 0.25

model_optimization_iterations = math.floor(math.e**4)
model_data_contamination_fraction = 0.1
model_cluster_target_coverage_threshold = 0.7
model_max_cluster_limit = model_cluster_target_coverage_threshold
model_metric_guide = "weighted"


def override():
    overrides = {}
    override_from = data_path.joinpath("config.py")
    if override_from.exists():
        exec(override_from.read_text(), locals=overrides)
    for override_key, override_value in filter(
        lambda t: t[0] not in not_overridable_properties, overrides.items()
    ):
        globals()[override_key] = override_value


override()

_spawn_context = multiprocessing.get_context("spawn")
training_executor = ProcessPoolExecutor(
    max_workers=max_training_workers,
    mp_context=_spawn_context,
)

estimation_executor = ProcessPoolExecutor(
    max_workers=max_estimation_workers,
    mp_context=_spawn_context,
)


def get_existing_users() -> list[int]:
    return [
        int(user_data.name)
        for user_data in data_path.iterdir()
        if re.match("\\d+", user_data.name)
    ]


@dataclasses.dataclass
class Subscription:
    liked_tracks_channel_id: int
    disliked_tracks_channel_id: int
    estimate_from_channel_id: int


def get_subscription(user_id: int) -> Optional[Subscription]:
    subscription_file = data_path.joinpath(str(user_id)).joinpath("subscription")
    if not subscription_file.exists():
        return None
    return from_dict(
        data_class=Subscription, data=json.loads(subscription_file.read_text())
    )


def get_subscribed_user_ids(channel_id: int) -> list[int]:
    return [
        user_id
        for user_id in get_existing_users()
        if get_subscription(user_id).estimate_from_channel_id == channel_id
    ]


def set_channels(user_id: int, subscription: Subscription):
    subscription_file = data_path.joinpath(str(user_id)).joinpath("subscription")
    subscription_file.parent.mkdir(exist_ok=True)
    with subscription_file.open(mode="wt") as subs:
        subs.write(json.dumps(dataclasses.asdict(subscription)))


@dataclasses.dataclass
class Model:
    model_id: int
    model_type: Literal["similar", "dissimilar"]
    pickle_file_path: pathlib.Path
    accuracy: float
    liked_tracks_count: int
    disliked_tracks_count: int


def get_models(user_id: int) -> list[Model]:
    models_path = data_path.joinpath(str(user_id)).joinpath("models")
    if not models_path.exists():
        return []
    return [
        get_model(user_id, int(model_path.stem))
        for model_path in models_path.iterdir()
        if model_path.is_dir()
        and model_path.joinpath(f"{model_path.name}.pickle").exists()
    ]


def get_model(user_id: int, model_id: int) -> Optional[Model]:
    model_path = (
        data_path.joinpath(str(user_id)).joinpath("models").joinpath(str(model_id))
    )
    if not model_path.exists():
        return None
    model_stats = json.loads(model_path.joinpath("stats.json").read_text())
    return Model(
        model_id=int(model_path.stem),
        pickle_file_path=model_path.joinpath(f"{model_path.stem}.pickle"),
        **model_stats,
    )


def get_current_model_id(user_id: int) -> Optional[int]:
    current_model_id_file = data_path.joinpath(str(user_id)).joinpath(
        "current_model_id.json"
    )
    if not current_model_id_file.exists():
        return None
    return json.loads(current_model_id_file.read_text())


def set_current_model_id(user_id: int, model_id: int):
    current_model_id_file = data_path.joinpath(str(user_id)).joinpath(
        "current_model_id.json"
    )
    with current_model_id_file.open(mode="wt") as model_store:
        model_store.write(json.dumps(model_id))


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
        model_pickle_name=f"{model_id}.pickle",
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


def get_allowed_to_use_user_ids() -> list[int]:
    whitelist_path = data_path.joinpath("user_whitelist")
    if not whitelist_path.exists():
        return []
    return [int(user.name) for user in whitelist_path.iterdir()]
