from __future__ import annotations

import enum
import json
import logging
from typing import Optional

from telethon import Button
from telethon.tl.types import DocumentAttributeFilename
from typing_extensions import Literal

import config


class ModelType(enum.IntEnum):
    EXCLUDE_DISLIKED = 0
    INCLUDE_LIKED = 1

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return ModelType[s]
        except KeyError:
            raise ValueError()


logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


async def build_model_page_response(
    user_id: int,
    subscription_names: dict[int, list[str]],
    offset_stack: list[int],
    action: Optional[tuple[int, Literal["backward", "forward"]]] = None,
) -> tuple[str, list[Button], tuple[bytes, list[DocumentAttributeFilename]]]:
    target_offset, action_type = action if action else (0, None)
    models = config.get_models(user_id)
    models_slice = models[target_offset : target_offset + config.dialog_list_page_size]
    if not action_type:
        previous_offset = None
        offset_stack.append(0)
        next_offset = (
            models_slice[-1]
            if len(models_slice) >= config.dialog_list_page_size
            else None
        )
    elif "forward" == action_type:
        previous_offset = offset_stack[-1] if offset_stack else None
        offset_stack.append(target_offset)
        next_offset = (
            models_slice[-1]
            if len(models_slice) >= config.dialog_list_page_size
            else None
        )
    elif "backward" == action_type:
        offset_stack.pop()
        next_offset = (
            models_slice[-1]
            if len(models_slice) >= config.dialog_list_page_size
            else None
        )
        previous_offset = offset_stack[-2] if len(offset_stack) >= 2 else None
    else:
        raise f"Unknown action type {action_type}"
    logger.debug(
        f"returning for model request {action=}: {offset_stack=},{previous_offset=},{next_offset=}"
    )

    return await format_model_response(
        models_slice,
        subscription_names,
        offset_stack,
        previous_offset,
        next_offset,
    )


async def format_model_response(
    items: list[config.Model],
    subscription_names: dict[int, list[str]],
    offset_stack: list[float | None] | list[int],
    previous_offset: Optional[float],
    next_offset: Optional[float],
) -> tuple[str, list[Button], tuple[bytes, list[DocumentAttributeFilename]]]:
    previous_button = (
        Button.inline("<--", f"model-list(backward:{previous_offset})")
        if previous_offset is not None
        else None
    )
    next_button = (
        Button.inline("-->", f"model-list(forward:{next_offset})")
        if next_offset is not None
        else None
    )
    models_formatted = "\n".join(
        [
            f"* [{",".join(subscription_names[model.model_id]) if model.model_id in subscription_names else ""}] model `{model.model_id}`: {model.accuracy:.2f}(track stats: {model.disliked_tracks_count}disliked / {model.liked_tracks_count}liked)"
            for model in items
        ]
    )
    if not items:
        models_formatted = "No items to show"
    serialized_offset_stack = json.dumps(offset_stack).encode("utf-8")
    file_name = [DocumentAttributeFilename("pagination-state.json")]
    return (
        models_formatted,
        [b for b in [previous_button, next_button] if b],
        (serialized_offset_stack, file_name),
    )
