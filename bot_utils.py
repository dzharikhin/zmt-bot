import logging
from typing import Optional

from telethon import TelegramClient
from telethon.tl import custom
from telethon.tl.functions.channels import GetChannelsRequest
from telethon.tl.types import Chat, Channel
from telethon.tl.types.messages import Chats

import config

logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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
) -> Optional[custom.Message]:
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
        link_by_title = [link for link in latest_message_links if str(channel.username) in link]
        if link_by_title:
            return int(link_by_title[0].split(f"{channel.title}/")[-1])

    raise ValueError(f"{latest_message_links} do not contain {channel} link")
