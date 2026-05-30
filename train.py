import asyncio
import enum
import logging
import pathlib
import pickle
import shutil
import tempfile
import time
import typing
from typing import Literal, cast

import numpy as np
import telethon
from telethon import TelegramClient
from telethon.tl import types
from telethon.tl.custom import Message
from telethon.tl.types import DocumentAttributeAudio

import config
from audio.features import extract_features_for_mp3, prepare_extractor
from bot_utils import get_message, obtain_latest_message_id, get_chat

logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

_estimation_model_cache = {}


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


def train_model(user_id: int, model_id: int, model_type: ModelType) -> config.Model:
    raise NotImplementedError(
        "train_model requires Phase 2 implementation (k-NN + GMM models)"
    )


async def prepare_model(
    user_id: int,
    bot_client: TelegramClient,
    latest_message_links: list[str],
    model_id: int,
    model_type: ModelType,
    force: bool = False,
    limit: int | None = None,
):
    try:
        channels = config.get_user_channels(user_id)
        if not channels:
            raise TrainUnrecoverable(f"User {user_id} has no channels initialized")

        if force:
            shutil.rmtree(config.get_liked_file_store_path(user_id))
            shutil.rmtree(config.get_disliked_file_store_path(user_id))
            shutil.rmtree(config.get_user_tmp_dir(user_id).joinpath(str(model_id)))

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
        model = await asyncio.get_running_loop().run_in_executor(
            config.get_training_executor(),
            train_model,
            user_id,
            model_id,
            model_type,
        )

    except TrainUnrecoverable as e:
        raise TrainUnrecoverable(
            f"Can't train model {model_id} for user {user_id}"
        ) from e


def _load_model(model_file: pathlib.Path) -> typing.Any:
    with model_file.open(mode="rb") as model_data:
        return pickle.load(model_data)


def execute_estimation(
    user_id: int, model_id: int, track_to_estimate_path: pathlib.Path
) -> bool:
    raise NotImplementedError(
        "execute_estimation requires Phase 2 implementation (k-NN + GMM models)"
    )


async def estimate(
    user_id: int,
    chat_id: int,
    message_id: int,
    model_id: int,
    bot_client: TelegramClient,
) -> bool:
    message = await get_message(chat_id, message_id, bot_client)

    tmp_dir = config.get_user_tmp_dir(user_id)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        track_to_estimate_path = pathlib.Path(tmp).joinpath(f"to-estimate.mp3")
        track_to_estimate_path.unlink(missing_ok=True)
        await message.download_media(file=track_to_estimate_path)
        is_recommended = await asyncio.get_running_loop().run_in_executor(
            config.get_estimation_executor(),
            execute_estimation,
            user_id,
            model_id,
            track_to_estimate_path,
        )
        logger.info(f"{user_id=} {chat_id=} {message_id=}: {is_recommended=}")
        return is_recommended
