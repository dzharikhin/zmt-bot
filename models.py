from __future__ import annotations

import enum


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
