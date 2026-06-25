import logging
from collections import defaultdict
from typing import Optional

from telethon import TelegramClient
from telethon.tl.custom import Message
from telethon.tl.functions.channels import GetChannelsRequest
from telethon.tl.types import Channel, Chat
from telethon.tl.types.messages import Chats

import config

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


async def get_chat(chat_id: int, bot_client: TelegramClient):
    return unwrap_single_chat(await bot_client(GetChannelsRequest(id=[chat_id])))


def unwrap_single_chat(chat: Chats) -> Optional[Chat]:
    if not chat or not chat.chats:
        return None
    return chat.chats[0]


async def get_message(
    channel: int | Chat, msg_id: int, bot_client: TelegramClient
) -> Optional[Message]:
    msgs = await bot_client.get_messages(channel, ids=[msg_id])
    return msgs[0] if msgs and msgs[0] else None


def is_allowed_user(user_id: int) -> bool:
    return (
        user_id == config.owner_user_id
        or user_id in config.get_allowed_to_use_user_ids()
    )


async def obtain_latest_message_id(
    channel: Chat | Channel, latest_message_links: list[str]
) -> int:
    link_by_id = [link for link in latest_message_links if str(channel.id) in link]
    if link_by_id:
        return int(link_by_id[0].split(f"{channel.id}/")[-1])

    if isinstance(channel, Channel):
        link_by_name = [
            link for link in latest_message_links if str(channel.username) in link
        ]
        if link_by_name:
            return int(link_by_name[0].split(f"{channel.username}/")[-1])

    raise ValueError(f"{latest_message_links} do not contain {channel} link")


async def get_channel_name(channel_id: int, bot_client: TelegramClient) -> str:
    try:
        channel_entity = await bot_client.get_entity(channel_id)
        channel_name = getattr(channel_entity, "title", None) or getattr(
            channel_entity, "username", None
        )
        if not channel_name:
            logger.warning(
                f"No channel name found for {channel_entity=}. Falling back to id",
            )
        return channel_name or str(channel_id)
    except Exception as e:
        logger.warning(
            "Error on obtaining channel name. Falling back to id",
            exc_info=e,
        )
        return str(channel_id)


async def get_channel_names(
    subscriptions: dict[int, int], bot_client: TelegramClient
) -> dict[int, list[str]]:
    subs_per_model = defaultdict(list)
    for channel_id, model_id in subscriptions.items():
        subs_per_model[model_id].append(await get_channel_name(channel_id, bot_client))
    return subs_per_model
