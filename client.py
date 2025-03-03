import asyncio
import json
import logging
import re
from asyncio import Task
from types import CoroutineType
from typing import cast, Union

from persistqueue import SQLiteAckQueue
from persistqueue.serializers import json as jser
from telethon import TelegramClient, events
from telethon.errors import RPCError
from telethon.events import NewMessage, CallbackQuery

import config
from models import build_model_page_response
from train import prepare_model, TrainUnrecoverable, estimate

# commands to implement:
# - subscribe <link_to_good> <link_to_bad> <link_to_estimate> - set channels to work with
# - train [force] - train new model and set it as current, force - reload all tracks
# - list-models - list available models
# - set-model <model_id> - set some model as current

logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


async def handle_train_queue_tasks(
    user_id: int,
    bot_client: TelegramClient,
):
    while True:
        queue_path = config.get_train_queue_path(user_id)
        if not queue_path.exists():
            await asyncio.sleep(1)
            continue

        queue = get_or_create_train_queue(user_id)
        cmd = await asyncio.to_thread(queue.get)
        try:
            await prepare_model(
                user_id, bot_client, cmd["message_id"], cmd["is_forced"]
            )
            queue.ack(cmd)
        except TrainUnrecoverable as e:
            cmd_id = queue.ack_failed(cmd)
            logger.warning(
                f"cannot handle {cmd_id}: {cmd} - marked as failed",
                exc_info=e,
            )
            await bot_client.send_message(user_id, f"Failed to execute {cmd}: {e}")
        except RPCError as e:
            cmd_id = queue.nack(cmd)
            logger.info(
                f"{cmd_id}: {cmd} - failed with {type(e)}, going to retry",
                exc_info=e,
            )


async def handle_estimate_queue_tasks(
    user_id: int,
    bot_client: TelegramClient,
):
    while True:
        queue_path = config.get_estimate_queue_path(user_id)
        if not queue_path.exists():
            await asyncio.sleep(1)
            continue
        queue = get_or_create_estimate_queue(user_id)
        cmd = await asyncio.to_thread(queue.get)
        try:
            is_recommended = bool(
                await estimate(user_id, cmd["chat_id"], cmd["message_id"], bot_client)
            )
            if is_recommended:
                await bot_client.forward_messages(user_id, cmd["message_id"])
            else:
                channel = await bot_client.get_input_entity(cmd["chat_id"])
                await bot_client.send_message(
                    f"Rated as not recommended: https://t.me/{channel.name}/{cmd["message_id"]}"
                )
            queue.ack(cmd)
        except TrainUnrecoverable as e:
            cmd_id = queue.ack_failed(cmd)
            logger.warning(
                f"cannot handle {cmd_id}: {cmd} - marked as failed",
                exc_info=e,
            )
            await bot_client.send_message(user_id, f"Failed to execute {cmd}: {e}")
        except RPCError as e:
            cmd_id = queue.nack(cmd)
            logger.info(
                f"{cmd_id}: {cmd} - failed with {type(e)}, going to retry",
                exc_info=e,
            )


START_CMD = "(?i)^/start"
SUBSCRIBE_CMD = "(?i)^/subscribe\\s+(-?\\d+)\\s+(-?\\d+)\\s+(-?\\d+)"
TRAIN_CMD = "(?i)^/train\\s+?([\\S]*)"
LIST_MODELS_CMD = "(?i)^/list"
SET_MODEL_CMD = "(?i)^/set\\s+(\\d+)"


def not_matched_command(txt: str) -> bool:
    return not any(
        (
            re.match(pattern, txt)
            for pattern in (
                START_CMD,
                SUBSCRIBE_CMD,
                TRAIN_CMD,
                LIST_MODELS_CMD,
                SET_MODEL_CMD,
            )
        )
    )


def get_or_create_train_queue(user_id: int) -> SQLiteAckQueue:
    queue_path = config.get_train_queue_path(user_id)
    return SQLiteAckQueue(
        str(queue_path.parent),
        serializer=jser,
        multithreading=True,
        auto_commit=True,
        db_file_name=queue_path.name,
    )


def get_or_create_estimate_queue(user_id: int) -> SQLiteAckQueue:
    queue_path = config.get_estimate_queue_path(user_id)
    return SQLiteAckQueue(
        str(queue_path.parent),
        serializer=jser,
        multithreading=True,
        auto_commit=True,
        db_file_name=queue_path.name,
    )


async def check_queue_handlers(
    tasks: dict[str, dict[str, Task]], bot_client: TelegramClient
):
    while True:
        for user_id in config.get_existing_users():
            current_user_tasks = tasks.get(str(user_id), {})
            train_queue_task = current_user_tasks.get("handle_train_queue_tasks")
            if (
                not train_queue_task
                or train_queue_task.cancelled()
                or train_queue_task.done()
            ):
                current_user_tasks["handle_train_queue_tasks"] = asyncio.create_task(
                    handle_train_queue_tasks(user_id, bot_client)
                )

            estimate_queue_task = current_user_tasks.get("handle_estimate_queue_tasks")
            if (
                not estimate_queue_task
                or estimate_queue_task.cancelled()
                or train_queue_task.done()
            ):
                current_user_tasks["handle_estimate_queue_tasks"] = asyncio.create_task(
                    handle_estimate_queue_tasks(user_id, bot_client)
                )
        await asyncio.sleep(10)


async def main():
    bot_client = await cast(
        Union[CoroutineType, TelegramClient],
        TelegramClient(
            config.data_path.joinpath("bot"), config.api_id, config.api_hash
        ).start(bot_token=config.bot_token),
    )
    tasks = {}
    async with bot_client:
        bot_client: TelegramClient
        logger.debug(f"Started bot {await bot_client.get_me()}")

        def filter_not_mapped(event: NewMessage.Event):
            return event.is_channel == False and not_matched_command(
                event.message.message
            )

        @bot_client.on(events.NewMessage(incoming=True, func=filter_not_mapped))
        @bot_client.on(events.NewMessage(incoming=True, pattern=START_CMD))
        async def start_handler(event: NewMessage.Event):

            logger.debug(f"Received unknown command: <{event.message.message}>")
            await event.respond(
                """
/subscribe - to create/edit subscription
/train - to train new recomenation model
/list - to list trained models
/sync - to sync some channels historical data
/set - to set model"""
            )

        @bot_client.on(events.NewMessage(incoming=True, pattern=SUBSCRIBE_CMD))
        async def subscribe_handler(event: NewMessage.Event):
            # if you need a way to get channel id - this is it
            # button = Button(types.KeyboardButtonRequestPeer("ch", 1, RequestPeerTypeBroadcast(), 1),
            #        resize=True, single_use=False, selective=False)
            # await event.respond(f"test", buttons=[button])

            like_channel_id = event.pattern_match.group(1).strip()
            dislike_channel_id = event.pattern_match.group(2).strip()
            estimate_channel_id = event.pattern_match.group(3).strip()
            config.set_channels(
                event.sender_id,
                config.Subscription(
                    int(like_channel_id),
                    int(dislike_channel_id),
                    int(estimate_channel_id),
                ),
            )
            await event.respond(
                f"Successfully subscribed. Please don't forget to add the bot into channels"
            )

        # if you need a way to get channel id - this is it
        # @bot_client.on(events.Raw())
        # async def handle(event: events.Raw):
        #     logger.debug(f"Received unknown command: <{event.message.message}>")

        def filter_subscribed_with_mp3(event: NewMessage.Event):
            return config.get_subscribed_user_ids(event.chat_id)

        @bot_client.on(events.NewMessage(func=filter_subscribed_with_mp3))
        async def handle_estimation_update_handler(event: NewMessage.Event):
            user_ids = config.get_subscribed_user_ids(event.chat_id)
            for user_id in user_ids:
                if not config.get_subscription(user_id):
                    await bot_client.send_message(user_id, f"{SUBSCRIBE_CMD} first")
                    continue

                get_or_create_estimate_queue(user_id).put(
                    {
                        "chat_id": event.chat_id,
                        "message_id": event.message.id,
                    }
                )

        @bot_client.on(events.NewMessage(incoming=True, pattern=TRAIN_CMD))
        async def handle_train_handler(event: NewMessage.Event):
            if not config.get_subscription(event.sender_id):
                event.respond(f"{SUBSCRIBE_CMD} first")
                return
            is_forced = (
                event.pattern_match.group(1)
                and event.pattern_match.group(1).to_lower() == "true"
            )

            get_or_create_train_queue(event.sender_id).put(
                {
                    "chat_id": event.chat_id,
                    "message_id": event.message.id,
                    "forced": is_forced,
                }
            )

        @bot_client.on(events.NewMessage(incoming=True, pattern=LIST_MODELS_CMD))
        async def list_models_handler(event: NewMessage.Event):
            if not config.get_subscription(event.sender_id):
                event.respond(f"{SUBSCRIBE_CMD} first")
                return
            await build_model_page_response(event.sender_id, [])

        @bot_client.on(
            events.CallbackQuery(data=re.compile("^model-list\\(([^:]+):([^:]+)\\)"))
        )
        async def channels_pagination_handler(event: CallbackQuery.Event):
            message = (
                await bot_client.get_messages(event.chat_id, ids=[event.message_id])
            )[0]
            action_type = event.pattern_match.group(1).decode("utf-8").strip()
            target_offset = event.pattern_match.group(2).decode("utf-8").strip()
            value = (await message.download_media(file=bytes)).decode("utf-8")
            offset_stack = json.loads(value)
            await build_model_page_response(
                message.sender_id, offset_stack, (int(target_offset), action_type)
            )

        @bot_client.on(events.NewMessage(incoming=True, pattern=SET_MODEL_CMD))
        async def set_model_handler(event: NewMessage.Event):
            if not config.get_subscription(event.sender_id):
                event.respond(f"{SUBSCRIBE_CMD} first")
                return
            model_id = int(event.pattern_match.group(1).decode("utf-8").strip())
            if not config.get_model(event.sender_id, model_id):
                event.respond(f"Model {model_id} does not exist")
                return

            config.set_current_model_id(event.sender_id, model_id)

        tasks["global"] = {
            "check": asyncio.create_task(check_queue_handlers(tasks, bot_client))
        }
        await bot_client.run_until_disconnected()
    for task_group in tasks.values():
        for task in task_group.values():
            task.cancel("shutdown")


# api_id = os.getenv("API_ID")
# api_hash = os.getenv("API_HASH")
# bot_token = os.getenv("BOT_TOKEN")
if __name__ == "__main__":
    asyncio.run(main())
