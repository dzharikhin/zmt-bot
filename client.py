import asyncio
import functools
import io
import json
import logging
import re
from argparse import ArgumentParser, ArgumentError
from asyncio import Task
from multiprocessing.managers import Namespace
from types import CoroutineType
from typing import cast, Union

import persistqueue
import polars
import telethon
from persistqueue.serializers import json as jser
from telethon import TelegramClient, events
from telethon.errors import RPCError
from telethon.events import NewMessage, CallbackQuery

import config
import train
from bot_utils import get_message, is_allowed_user
from models import build_model_page_response
from train import prepare_model, estimate

# commands to implement:
# - subscribe <link_to_good> <link_to_bad> <link_to_estimate> - set channels to work with
# - train [force] - train new model and set it as current, force - reload all tracks
# - list - list available models
# - set-model <model_id> - set some model as current

logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


async def send_train_queue_task(
    event,
    model_type: train.ModelType,
    limit: int | None,
    is_forced: bool,
):
    get_or_create_train_queue(event.sender_id).put(
        {
            "message_id": event.message.id,
            "model_type": model_type,
            "forced": is_forced,
            "limit": limit,
        }
    )


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
        cmd = None
        try:
            cmd = queue.get_nowait()
            logger.debug(f"Handling train cmd={cmd}")
            await prepare_model(
                user_id,
                bot_client,
                cmd["message_id"],
                cmd["model_type"],
                cmd["forced"],
                cmd.get("limit", 1000),
            )
            model = config.get_model(user_id, cmd["message_id"])
            await bot_client.send_message(
                user_id,
                f"Successfully trained model {model.model_id}: accuracy={model.accuracy} for {model.disliked_tracks_count} disliked tracks and {model.liked_tracks_count} liked tracks",
            )
            queue.ack(cmd)
        except persistqueue.exceptions.Empty:
            await asyncio.sleep(1)
        except telethon.errors.rpcerrorlist.BotMethodInvalidError as e:
            await handle_non_recoverable(
                bot_client, cmd, e, queue, user_id, "cannot train model"
            )
        except RPCError as e:
            cmd_id = queue.nack(cmd)
            logger.info(
                f"{cmd_id}: {cmd} - failed with {type(e)}, going to retry",
                exc_info=e,
            )
        except Exception as e:
            await handle_non_recoverable(
                bot_client, cmd, e, queue, user_id, "cannot train model"
            )


async def handle_non_recoverable(bot_client, cmd, e, queue, user_id, prefix):
    cmd_id = queue.ack_failed(cmd)
    logger.warning(
        f"{prefix} for user {user_id}. {cmd_id}: {cmd} - marked as failed",
        exc_info=e,
    )
    await bot_client.send_message(user_id, f"Failed to execute {cmd}: {e}")


async def send_estimate_queue_task(event, user_id):
    get_or_create_estimate_queue(user_id).put(
        {
            "chat_id": event.chat_id,
            "message_id": event.message.id,
        }
    )
    logger.debug(
        f"Created estimation task for chat_id={event.chat_id} and message_id={event.message.id}"
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
        cmd = None
        try:
            cmd = queue.get_nowait()
            logger.debug(f"Handling estimation cmd={cmd}")
            is_recommended = bool(
                await estimate(user_id, cmd["chat_id"], cmd["message_id"], bot_client)
            )
            message = await get_message(cmd["chat_id"], cmd["message_id"], bot_client)
            if message:
                if is_recommended:
                    await bot_client.forward_messages(user_id, message)
                else:
                    if message.forward:
                        reply_message = f"channel erases forward info, so provide <https://t.me> link explicitly when forwarding to estimation channel"
                    elif m := re.match("https://t.me/\\S+", message.message):
                        reply_message = f"Rated as not recommended: {m.group(0)}"
                    else:
                        reply_message = f"Rated as not recommended: https://t.me/c/{message.input_chat.channel_id}/{message.id}"

                    await bot_client.send_message(user_id, reply_message)
            else:
                alert = f"Message {cmd["message_id"]} from channel {cmd["chat_id"]} seems to be removed"
                logger.info(alert)
                await bot_client.send_message(user_id, alert)
            queue.ack(cmd)
        except persistqueue.exceptions.Empty:
            await asyncio.sleep(1)
        except RPCError as e:
            cmd_id = queue.nack(cmd)
            logger.info(
                f"{cmd_id}: {cmd} - failed with {type(e)}, going to retry",
                exc_info=e,
            )
        except Exception as e:
            await handle_non_recoverable(
                bot_client, cmd, e, queue, user_id, "cannot estimate track"
            )


# START_CMD = "(?i)^/start$"
# SUBSCRIBE_CMD = "(?i)^/subscribe\\s+(-?\\d+)\\s+(-?\\d+)\\s+(-?\\d+)"
# TRAIN_CMD = "(?i)^/train\\s*([\\S\\D]*\\s*([\\S]*)\\s*([\\S]*)"
# LIST_MODELS_CMD = "(?i)^/list"
# SET_MODEL_CMD = "(?i)^/set\\s+(\\d+)"

START_CMD = ArgumentParser(
    prog="start",
    epilog="(?i)^/start.*$",
    description="print available commands",
    exit_on_error=False,
    add_help=False,
)
SUBSCRIBE_CMD = (
    parser := ArgumentParser(
        prog="subscribe",
        epilog="(?i)^/subscribe(.*)$",
        description="create subscription to telegram data",
        exit_on_error=False,
        add_help=False,
    ),
    parser.add_argument(
        "-l",
        "--liked_channel_id",
        required=True,
        type=int,
        help="channel with user-liked tracks. Data for ML. Don't forget to add the bot to channel",
    ),
    parser.add_argument(
        "-d",
        "--disliked_channel_id",
        required=True,
        type=int,
        help="channel with user-disliked tracks. Data for ML. Don't forget to add the bot to channel",
    ),
    parser.add_argument(
        "-e",
        "--estimation_channel_id",
        required=True,
        type=int,
        help="channel to estimate tracks from. Don't forget to add the bot to channel",
    ),
    parser,
)[-1]
TRAIN_CMD = (
    parser := ArgumentParser(
        prog="train",
        epilog="(?i)^/train(.*)$",
        description="train a model to estimate track with and set it as current",
        exit_on_error=False,
        add_help=False,
    ),
    parser.add_argument(
        "-t",
        "--type",
        required=True,
        type=train.ModelType.from_string,
        choices=list(train.ModelType),
        help=f"model type. {train.ModelType.INCLUDE_LIKED} - posts tracks similar to liked ones, {train.ModelType.EXCLUDE_DISLIKED} - posts other than disliked",
    ),
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        help="limit download with only last [limit] tracks. Can be faster",
    ),
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="clear already downloaded tracks and download again",
    ),
    parser,
)[-1]
LIST_MODELS_CMD = ArgumentParser(
    prog="list",
    epilog="(?i)^/list\\s*.*$",
    description="list trained models",
    exit_on_error=False,
    add_help=False,
)
SET_MODEL_CMD = (
    parser := ArgumentParser(
        prog="set",
        epilog="(?i)^/set(.*)$",
        description="set a model to estimate tracks with",
        exit_on_error=False,
        add_help=False,
    ),
    parser.add_argument(
        "-m",
        "--model_id",
        required=True,
        type=int,
        help="model id to set as current estimation model",
    ),
    parser,
)[-1]


def _parse_args(
    arg_parser: ArgumentParser, cmd_line: str
) -> tuple[Namespace | None, str | None]:
    try:
        args = arg_parser.parse_args(cmd_line.split())
        return args, None
    except ArgumentError as e:
        buffer = io.StringIO()
        arg_parser.print_usage(buffer)
        return None, buffer.getvalue()


def _not_matched_command(txt: str) -> bool:
    return not any(
        (
            re.match(pattern, txt)
            for pattern in (
                START_CMD.epilog,
                SUBSCRIBE_CMD.epilog,
                TRAIN_CMD.epilog,
                LIST_MODELS_CMD.epilog,
                SET_MODEL_CMD.epilog,
            )
        )
    )


@functools.cache
def get_or_create_train_queue(user_id: int) -> persistqueue.SQLiteAckQueue:
    queue_path = config.get_train_queue_path(user_id)
    return persistqueue.SQLiteAckQueue(
        str(queue_path.parent),
        serializer=jser,
        multithreading=True,
        auto_commit=True,
        db_file_name=queue_path.name,
    )


@functools.cache
def get_or_create_estimate_queue(user_id: int) -> persistqueue.SQLiteAckQueue:
    queue_path = config.get_estimate_queue_path(user_id)
    return persistqueue.SQLiteAckQueue(
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
    polars.show_versions()
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
            return event.is_channel == False and _not_matched_command(
                event.message.message
            )

        @bot_client.on(events.NewMessage(incoming=True, func=filter_not_mapped))
        @bot_client.on(events.NewMessage(incoming=True, pattern=START_CMD.epilog))
        async def start_handler(event: NewMessage.Event):
            if not is_allowed_user(event.sender_id):
                await bot_client.send_message(
                    config.owner_user_id, f"user {event.sender_id} tries to use zmt-bot"
                )
                return
            logger.debug(f"Received unknown command: <{event.message.message}>")
            buffer = io.StringIO()
            for cmd in [SUBSCRIBE_CMD, TRAIN_CMD, LIST_MODELS_CMD, SET_MODEL_CMD]:
                buffer.write(f"/{cmd.prog}\n")
                cmd.print_usage(buffer)
            await event.respond(buffer.getvalue())

        @bot_client.on(events.NewMessage(incoming=True, pattern=SUBSCRIBE_CMD.epilog))
        async def subscribe_handler(event: NewMessage.Event):
            if not is_allowed_user(event.sender_id):
                await bot_client.send_message(
                    config.owner_user_id, f"user {event.sender_id} tries to use zmt-bot"
                )
                return

                # if you need a way to get channel id - this is it
            # button = Button(types.KeyboardButtonRequestPeer("ch", 1, RequestPeerTypeBroadcast(), 1),
            #        resize=True, single_use=False, selective=False)
            # await event.respond(f"test", buttons=[button])

            args, help_to_print = _parse_args(
                SUBSCRIBE_CMD, event.pattern_match.group(1).strip()
            )
            if help_to_print:
                await event.respond(help_to_print)
                return

            config.set_channels(
                event.sender_id,
                config.Subscription(
                    args.liked_channel_id,
                    args.disliked_channel_id,
                    args.estimation_channel_id,
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
                    await bot_client.send_message(
                        user_id, f"/{SUBSCRIBE_CMD.prog} first"
                    )
                    continue

                await send_estimate_queue_task(event, user_id)

        @bot_client.on(events.NewMessage(incoming=True, pattern=TRAIN_CMD.epilog))
        async def handle_train_handler(event: NewMessage.Event):
            if not is_allowed_user(event.sender_id):
                await bot_client.send_message(
                    config.owner_user_id, f"user {event.sender_id} tries to use zmt-bot"
                )
                return

            if not config.get_subscription(event.sender_id):
                await event.respond(f"/{SUBSCRIBE_CMD.prog} first")
                return

            args, help_to_print = _parse_args(
                TRAIN_CMD, event.pattern_match.group(1).strip()
            )
            if help_to_print:
                await event.respond(help_to_print)
                return

            await send_train_queue_task(event, args.type, args.limit, args.force)
            await event.respond(f"Training task for id={event.message.id} created")

        @bot_client.on(events.NewMessage(incoming=True, pattern=LIST_MODELS_CMD.epilog))
        async def list_models_handler(event: NewMessage.Event):
            if not is_allowed_user(event.sender_id):
                await bot_client.send_message(
                    config.owner_user_id, f"user {event.sender_id} tries to use zmt-bot"
                )
                return

            if not config.get_subscription(event.sender_id):
                await event.respond(f"/{SUBSCRIBE_CMD.prog} first")
                return

            message_text, buttons, (pagination_data, attributes) = (
                await build_model_page_response(event.sender_id, [])
            )
            conditional_params = (
                {"buttons": buttons, "file": pagination_data} if buttons else {}
            )
            await event.respond(
                message_text,
                attributes=attributes,
                **conditional_params,
            )

        @bot_client.on(
            events.CallbackQuery(data=re.compile("^model-list\\(([^:]+):([^:]+)\\)"))
        )
        async def models_pagination_handler(event: CallbackQuery.Event):
            message = await get_message(event.chat_id, event.message_id, bot_client)
            action_type = event.pattern_match.group(1).decode("utf-8").strip()
            target_offset = event.pattern_match.group(2).decode("utf-8").strip()
            value = (await message.download_media(file=bytes)).decode("utf-8")
            offset_stack = json.loads(value)
            message_text, buttons, (pagination_data, attributes) = (
                await build_model_page_response(
                    message.sender_id, offset_stack, (int(target_offset), action_type)
                )
            )
            await event.edit(
                message_text,
                file=pagination_data,
                attributes=attributes,
                buttons=buttons,
            )

        @bot_client.on(events.NewMessage(incoming=True, pattern=SET_MODEL_CMD.epilog))
        async def set_model_handler(event: NewMessage.Event):
            if not is_allowed_user(event.sender_id):
                await bot_client.send_message(
                    config.owner_user_id, f"user {event.sender_id} tries to use zmt-bot"
                )
                return

            if not config.get_subscription(event.sender_id):
                await event.respond(f"/{SUBSCRIBE_CMD.prog} first")
                return

            args, help_to_print = _parse_args(
                SET_MODEL_CMD, event.pattern_match.group(1).strip()
            )
            if help_to_print:
                await event.respond(help_to_print)
                return
            if not config.get_model(event.sender_id, args.model_id):
                await event.respond(f"Model {args.model_id} does not exist")
                return

            config.set_current_model_id(event.sender_id, args.model_id)
            await event.respond(f"Model {args.model_id} set as default")

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
