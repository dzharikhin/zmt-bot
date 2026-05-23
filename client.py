import asyncio
import functools
import io
import json
import logging
import re
from argparse import ArgumentParser, ArgumentError
from asyncio import Task
from concurrent.futures.process import BrokenProcessPool
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
from bot_utils import get_channel_name, get_message, is_allowed_user, get_channel_names
from models import build_model_page_response
from train import prepare_model, estimate


logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


async def send_train_queue_task(
    event,
    latest_message_links: list[str],
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
            "latest_message_links": latest_message_links,
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
            if isinstance(cmd["model_type"], int):
                cmd["model_type"] = train.ModelType(cmd["model_type"])
            await prepare_model(
                user_id,
                bot_client,
                cmd["latest_message_links"],
                cmd["message_id"],
                cmd["model_type"],
                cmd["forced"],
                cmd.get("limit", 1000),
            )
            model = config.get_model(user_id, cmd["message_id"])
            await bot_client.send_message(
                user_id,
                f"Successfully trained model {model.model_id}: accuracy={model.accuracy:.2f} for {model.disliked_tracks_count} disliked tracks and {model.liked_tracks_count} liked tracks",
            )
            queue.ack(cmd)
        except persistqueue.exceptions.Empty:
            await asyncio.sleep(1)
        except telethon.errors.rpcerrorlist.BotMethodInvalidError as e:
            await handle_non_recoverable(
                bot_client, cmd, e, queue, user_id, "cannot train model"
            )
        except (RPCError, BrokenProcessPool) as e:
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


async def send_estimate_queue_task_with_channel(
    event: NewMessage.Event,
    user_id: int,
    subscription: config.Subscription,
    channel_name: str,
):
    get_or_create_estimate_queue(user_id).put(
        {
            "chat_id": event.chat_id,
            "message_id": event.message.id,
            "model_id": subscription.model_id,
            "channel_name": channel_name,
        }
    )
    logger.debug(
        f"Created estimation task for {user_id=} {event.chat_id=} {event.message.id=}"
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
            is_recommended = await estimate(
                user_id, cmd["chat_id"], cmd["message_id"], cmd["model_id"], bot_client
            )
            message = await get_message(cmd["chat_id"], cmd["message_id"], bot_client)
            if message:
                channel_name = cmd.get("channel_name", str(cmd["chat_id"]))

                if is_recommended:
                    if config.estimation_post_way == "reply":
                        await bot_client.send_message(
                            user_id,
                            f"#{channel_name}",
                            file=message.media,
                            reply_to=message,
                        )
                    else:
                        await bot_client.forward_messages(user_id, message)
                else:
                    if message.forward:
                        reply_message = f"[{channel_name}] channel erases forward info, so provide <https://t.me> link explicitly when forwarding to estimation channel"
                    elif m := re.match("https://t.me/\\S+", message.message):
                        reply_message = (
                            f"[{channel_name}] Rated as not recommended: {m.group(0)}"
                        )
                    else:
                        reply_message = f"[{channel_name}] Rated as not recommended: https://t.me/c/{message.input_chat.channel_id}/{message.id}"

                    await bot_client.send_message(user_id, reply_message)
            else:
                alert = f"Message {cmd['message_id']} from channel {cmd['chat_id']} seems to be removed"
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


START_CMD = ArgumentParser(
    prog="start",
    epilog="(?i)^/start.*$",
    description="print available commands",
    exit_on_error=False,
    add_help=False,
)
INIT_CMD = (
    parser := ArgumentParser(
        prog="init",
        epilog="(?i)^/init(.*)$",
        description="initialize user channels (liked/disliked)",
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
    parser,
)[-1]

SUBSCRIBE_CMD = (
    parser := ArgumentParser(
        prog="subscribe",
        epilog="(?i)^/subscribe(.*)$",
        description="create subscription to telegram data",
        exit_on_error=False,
        add_help=False,
    ),
    parser.add_argument(
        "-e",
        "--estimation_channel_id",
        required=True,
        type=int,
        help="channel to estimate tracks from. Don't forget to add the bot to channel",
    ),
    parser.add_argument(
        "-m",
        "--model_id",
        required=True,
        type=int,
        help="model id to use for this subscription",
    ),
    parser,
)[-1]

UNSUBSCRIBE_CMD = (
    parser := ArgumentParser(
        prog="unsubscribe",
        epilog="(?i)^/unsubscribe(.*)$",
        description="remove subscription",
        exit_on_error=False,
        add_help=False,
    ),
    parser.add_argument(
        "-e",
        "--estimation_channel_id",
        required=True,
        type=int,
        help="estimation channel to unsubscribe from",
    ),
    parser,
)[-1]

SET_SUB_MODEL_CMD = (
    parser := ArgumentParser(
        prog="set-subscription-model",
        epilog="(?i)^/set-subscription-model(.*)$",
        description="update model for a subscription",
        exit_on_error=False,
        add_help=False,
    ),
    parser.add_argument(
        "-e",
        "--estimation_channel_id",
        required=True,
        type=int,
        help="estimation channel to update",
    ),
    parser.add_argument(
        "-m",
        "--model_id",
        required=True,
        type=int,
        help="new model id",
    ),
    parser,
)[-1]

SUBSCRIPTIONS_CMD = ArgumentParser(
    prog="subscriptions",
    epilog="(?i)^/subscriptions\\s*.*$",
    description="list all subscriptions",
    exit_on_error=False,
    add_help=False,
)
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
    parser.add_argument(
        "-ll",
        "--latest_message_links",
        required=True,
        type=str,
        nargs="+",
        help="liked and disliked channels latest message links(in any order)",
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

CMDS = [
    START_CMD,  # always first element
    INIT_CMD,
    SUBSCRIBE_CMD,
    UNSUBSCRIBE_CMD,
    SET_SUB_MODEL_CMD,
    SUBSCRIPTIONS_CMD,
    TRAIN_CMD,
    LIST_MODELS_CMD,
]


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
    return not any((re.match(pattern, txt) for pattern in (cmd.epilog for cmd in CMDS)))


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
            config.local_data_path.joinpath("bot"), config.api_id, config.api_hash
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
        async def start_handler(event: NewMessage.Event) -> None:
            if not is_allowed_user(event.sender_id):
                await bot_client.send_message(
                    config.owner_user_id, f"user {event.sender_id} tries to use zmt-bot"
                )
                return
            logger.debug(f"Received unknown command: <{event.message.message}>")
            buffer = io.StringIO()
            for cmd in [
                INIT_CMD,
                SUBSCRIBE_CMD,
                SUBSCRIPTIONS_CMD,
                SET_SUB_MODEL_CMD,
                UNSUBSCRIBE_CMD,
                TRAIN_CMD,
                LIST_MODELS_CMD,
            ]:
                buffer.write(f"/{cmd.prog}\n")
                cmd.print_usage(buffer)
                buffer.write("\n")
            await event.respond(buffer.getvalue())

        @bot_client.on(events.NewMessage(incoming=True, pattern=INIT_CMD.epilog))
        async def init_handler(event: NewMessage.Event) -> None:
            if not is_allowed_user(event.sender_id):
                await bot_client.send_message(
                    config.owner_user_id, f"user {event.sender_id} tries to use zmt-bot"
                )
                return

            args, help_to_print = _parse_args(
                INIT_CMD, event.pattern_match.group(1).strip()
            )
            if help_to_print:
                await event.respond(help_to_print)
                return

            try:
                liked_channel = await bot_client.get_entity(args.liked_channel_id)
                disliked_channel = await bot_client.get_entity(args.disliked_channel_id)
            except Exception as e:
                await event.respond(
                    f"❌ Error: Cannot access one or both channels. Check bot permissions."
                )
                return

            channels = config.UserChannels(
                liked_channel_id=args.liked_channel_id,
                disliked_channel_id=args.disliked_channel_id,
            )
            config.set_user_channels(event.sender_id, channels)

            await event.respond(
                f"Channels initialized. Train a model: /train -t <type> -ll <links>"
            )

        @bot_client.on(events.NewMessage(incoming=True, pattern=SUBSCRIBE_CMD.epilog))
        async def subscribe_handler(event: NewMessage.Event) -> None:
            if not is_allowed_user(event.sender_id):
                await bot_client.send_message(
                    config.owner_user_id, f"user {event.sender_id} tries to use zmt-bot"
                )
                return

            args, help_to_print = _parse_args(
                SUBSCRIBE_CMD, event.pattern_match.group(1).strip()
            )
            if help_to_print:
                await event.respond(help_to_print)
                return

            user_id = event.sender_id
            channels = config.get_user_channels(user_id)
            if not channels:
                await event.respond(
                    f"❌ Error: Initialize channels first. Hint: /init -l <id> -d <id>"
                )
                return

            if not config.get_model(user_id, args.model_id):
                await event.respond(
                    f"❌ Error: Model {args.model_id} does not exist. Hint: /train -t <type> -ll <links>"
                )
                return

            existing_sub = config.get_subscription_by_channel(
                user_id, args.estimation_channel_id
            )
            if existing_sub:
                await event.respond(
                    f"❌ Error: Already subscribed to this channel. Hint: /subscriptions"
                )
                return

            channel_name = await get_channel_name(
                args.estimation_channel_id, bot_client
            )

            subscription = config.Subscription(
                estimate_from_channel_id=args.estimation_channel_id,
                model_id=args.model_id,
            )
            config.add_subscription(user_id, subscription)

            await event.respond(
                f"Subscribed to {channel_name} with model #{args.model_id}"
            )

        @bot_client.on(events.NewMessage(incoming=True, pattern=UNSUBSCRIBE_CMD.epilog))
        async def unsubscribe_handler(event: NewMessage.Event) -> None:
            if not is_allowed_user(event.sender_id):
                await bot_client.send_message(
                    config.owner_user_id, f"user {event.sender_id} tries to use zmt-bot"
                )
                return

            args, help_to_print = _parse_args(
                UNSUBSCRIBE_CMD, event.pattern_match.group(1).strip()
            )
            if help_to_print:
                await event.respond(help_to_print)
                return

            subscription = config.get_subscription_by_channel(
                event.sender_id, args.estimation_channel_id
            )
            if not subscription:
                await event.respond(
                    f"❌ Error: Not subscribed to channel {args.estimation_channel_id}. Hint: /subscriptions"
                )
                return

            try:
                channel_info = await bot_client.get_entity(args.estimation_channel_id)
                channel_name = (
                    getattr(channel_info, "title", None)
                    or getattr(channel_info, "username", None)
                    or str(args.estimation_channel_id)
                )
            except Exception:
                channel_name = str(args.estimation_channel_id)

            config.remove_subscription(event.sender_id, args.estimation_channel_id)

            await event.respond(f"Unsubscribed from {channel_name}")

        @bot_client.on(
            events.NewMessage(incoming=True, pattern=SET_SUB_MODEL_CMD.epilog)
        )
        async def set_sub_model_handler(event: NewMessage.Event) -> None:
            if not is_allowed_user(event.sender_id):
                await bot_client.send_message(
                    config.owner_user_id, f"user {event.sender_id} tries to use zmt-bot"
                )
                return

            args, help_to_print = _parse_args(
                SET_SUB_MODEL_CMD, event.pattern_match.group(1).strip()
            )
            if help_to_print:
                await event.respond(help_to_print)
                return

            subscription = config.get_subscription_by_channel(
                event.sender_id, args.estimation_channel_id
            )
            if not subscription:
                await event.respond(
                    f"❌ Error: Not subscribed to channel {args.estimation_channel_id}"
                )
                return

            if not config.get_model(event.sender_id, args.model_id):
                await event.respond(
                    f"❌ Error: Model {args.model_id} does not exist. Hint: /train -t <type> -ll <links>"
                )
                return

            try:
                channel_info = await bot_client.get_entity(args.estimation_channel_id)
                channel_name = (
                    getattr(channel_info, "title", None)
                    or getattr(channel_info, "username", None)
                    or str(args.estimation_channel_id)
                )
            except Exception:
                channel_name = str(args.estimation_channel_id)

            config.update_subscription_model(
                event.sender_id, args.estimation_channel_id, args.model_id
            )

            await event.respond(f"Updated {channel_name} to use model #{args.model_id}")

        @bot_client.on(
            events.NewMessage(incoming=True, pattern=SUBSCRIPTIONS_CMD.epilog)
        )
        async def subscriptions_handler(event: NewMessage.Event) -> None:
            if not is_allowed_user(event.sender_id):
                await bot_client.send_message(
                    config.owner_user_id, f"user {event.sender_id} tries to use zmt-bot"
                )
                return

            subscriptions = config.get_subscriptions(event.sender_id)
            if not subscriptions:
                await event.respond(
                    f"No subscriptions yet. Add: /subscribe -e <channel> -m <model>"
                )
                return

            buffer = io.StringIO()
            buffer.write("Your subscriptions:\n")
            for idx, sub in enumerate(subscriptions, 1):
                try:
                    channel_info = await bot_client.get_entity(
                        sub.estimate_from_channel_id
                    )
                    channel_name = (
                        getattr(channel_info, "title", None)
                        or getattr(channel_info, "username", None)
                        or str(sub.estimate_from_channel_id)
                    )
                except Exception:
                    channel_name = str(sub.estimate_from_channel_id)
                buffer.write(f"{idx}. {channel_name} - Model #{sub.model_id}\n")

            await event.respond(buffer.getvalue())

        def filter_subscribed_with_mp3(event: NewMessage.Event):
            return config.get_subscribed_user_ids(event.chat_id)

        @bot_client.on(events.NewMessage(func=filter_subscribed_with_mp3))
        async def handle_estimation_update_handler(event: NewMessage.Event):
            user_ids = config.get_subscribed_user_ids(event.chat_id)
            for user_id in user_ids:
                subscription = config.get_subscription_by_channel(
                    user_id, event.chat_id
                )
                if not subscription:
                    await bot_client.send_message(
                        user_id, f"/{SUBSCRIBE_CMD.prog} first"
                    )
                    continue
                channel_name = await get_channel_name(event.chat_id, bot_client)
                await send_estimate_queue_task_with_channel(
                    event, user_id, subscription, channel_name
                )

        @bot_client.on(events.NewMessage(incoming=True, pattern=TRAIN_CMD.epilog))
        async def handle_train_handler(event: NewMessage.Event):
            if not is_allowed_user(event.sender_id):
                await bot_client.send_message(
                    config.owner_user_id, f"user {event.sender_id} tries to use zmt-bot"
                )
                return

            args, help_to_print = _parse_args(
                TRAIN_CMD, event.pattern_match.group(1).strip()
            )
            if help_to_print:
                await event.respond(help_to_print)
                return

            user_id = event.sender_id
            if not config.has_user_channels(user_id):
                await event.respond(
                    f"❌ Error: Initialize channels first. Hint: /init -l <id> -d <id>"
                )
                return

            await send_train_queue_task(
                event, args.latest_message_links, args.type, args.limit, args.force
            )
            await event.respond(f"Training task for id={event.message.id} created")

        @bot_client.on(events.NewMessage(incoming=True, pattern=LIST_MODELS_CMD.epilog))
        async def list_models_handler(event: NewMessage.Event):
            if not is_allowed_user(event.sender_id):
                await bot_client.send_message(
                    config.owner_user_id, f"user {event.sender_id} tries to use zmt-bot"
                )
                return

            if not config.has_user_channels(event.sender_id):
                await event.respond(
                    f"❌ Error: Initialize channels first. Hint: /init -l <id> -d <id>"
                )
                return

            subscription_names = await get_channel_names(
                {
                    s.estimate_from_channel_id: s.model_id
                    for s in config.get_subscriptions(event.sender_id)
                },
                bot_client,
            )
            message_text, buttons, (pagination_data, attributes) = (
                await build_model_page_response(event.sender_id, subscription_names, [])
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
            subscription_names = await get_channel_names(
                {
                    s.estimate_from_channel_id: s.model_id
                    for s in config.get_subscriptions(event.sender_id)
                },
                bot_client,
            )
            message_text, buttons, (pagination_data, attributes) = (
                await build_model_page_response(
                    message.sender_id,
                    subscription_names,
                    offset_stack,
                    (int(target_offset), action_type),
                )
            )
            await event.edit(
                message_text,
                file=pagination_data,
                attributes=attributes,
                buttons=buttons,
            )

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
