from typing import Optional

from telethon import TelegramClient
from telethon.tl import custom
from telethon.tl.types import Chat
from telethon.tl.types.messages import Chats


def unwrap_single_chat(chat: Chats) -> Optional[Chat]:
    if not chat or not chat.chats:
        return None
    return chat.chats[0]


async def get_message(
    channel: int | Chat, msg_id: int, bot_client: TelegramClient
) -> Optional[custom.Message]:
    msgs = await bot_client.get_messages(channel, ids=[msg_id])
    return msgs[0] if msgs and msgs[0] else None
