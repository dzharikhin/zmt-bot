from typing import Optional

from telethon import TelegramClient
from telethon.tl import custom
from telethon.tl.functions.channels import GetChannelsRequest
from telethon.tl.types import Chat
from telethon.tl.types.messages import Chats

import config


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
