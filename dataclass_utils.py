import dataclasses
import functools
import typing
from typing import Literal

import numpy as np


def unwrap_type(v):
    if hasattr(v, "tolist"):
        return v.tolist()
    elif hasattr(v, "item"):
        return v.item()
    else:
        return v


def _get_type_shape(target_type: typing.Type) -> typing.Optional[tuple[int, ...]]:
    if typing.get_origin(target_type) in [np.ndarray]:
        root_size_type = typing.get_args(target_type)[0]
        if typing.get_origin(root_size_type) in [tuple]:
            dimensions = typing.get_args(root_size_type)
            return tuple(
                [
                    (
                        typing.get_args(dimension)[0]
                        if typing.get_origin(dimension) in [Literal]
                        else None
                    )
                    for dimension in dimensions
                ]
            )

    return None


def unwrap_to_dict(dataclass_object):
    return {k: unwrap_type(v) for k, v in dataclasses.asdict(dataclass_object).items()}


@functools.cache
def get_field_shape_map(container_type: typing.Type):
    return {f.name: _get_type_shape(f.type) for f in dataclasses.fields(container_type)}


def convert_type(original_object, target_type: typing.Type, additional_kwargs):
    remapped_values = unwrap_to_dict(original_object)
    return target_type(**remapped_values, **additional_kwargs)


def create_wrapper_type(original_type, name, type_mapping, additional_fields):
    field_definitions = [
        (f.name, type_mapping.get(typing.get_origin(f.type) or f.type, f.type))
        for f in dataclasses.fields(original_type)
    ] + list(additional_fields.items())
    return dataclasses.make_dataclass(name, field_definitions)


def get_class_field_shape_mapping(dataclass_or_instance) -> dict[str, tuple[int, ...]]:
    return {
        field.name: _get_type_shape(field.type)
        for field in dataclasses.fields(dataclass_or_instance)
    }


def get_class_container_fields(
    dataclass_or_instance, target_types: list[typing.Type]
) -> dict[str, typing.ParamSpec]:
    return {
        field.name: field_type
        for field in dataclasses.fields(dataclass_or_instance)
        if (field_type := typing.get_origin(field.type)) in target_types
    }
