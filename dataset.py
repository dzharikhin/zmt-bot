import pathlib
import re
from typing import Any, NamedTuple

import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle

SNIPPET_PATH_COLUMN = "snippet_path"


class Track(NamedTuple):
    spotify_id: str


class Dataset:

    TYPES = {SNIPPET_PATH_COLUMN: str}

    def __init__(self, csv: pathlib.Path, persist_per_update: bool = False):
        self.csv = csv
        self.result_path = csv.parent.joinpath(f"{csv.name}.result")
        self.dataframe = shuffle(self._create_dataset())
        self.persist_per_update = persist_per_update

    def _create_dataset(self) -> DataFrame:
        if self.result_path.exists():
            return pd.read_csv(self.result_path, dtype=Dataset.TYPES)

        with self.csv.open(mode="r") as dataset:
            first_line = dataset.readline()
            headers = [
                column.strip('"') for column in re.split(r",", first_line.strip())
            ]
        initial_dataframe = pd.read_csv(
            self.csv, sep=";", header=None, names=headers, skiprows=1
        )
        initial_dataframe[SNIPPET_PATH_COLUMN] = ""
        return initial_dataframe

    def add_snippet_path(self, spotify_id: str, snippet_path: pathlib.Path | str):
        if snippet_path is pathlib.Path:
            snippet_path = snippet_path.resolve().absolute()
        self.dataframe.loc[
            self.dataframe["spotify_id"] == spotify_id, SNIPPET_PATH_COLUMN
        ] = str(snippet_path)
        if self.persist_per_update:
            self.save()

    def save(self):
        self.dataframe.to_csv(self.result_path, mode="w")

    @staticmethod
    def has_snippet(row: tuple[Any]) -> bool:
        attr_value = getattr(row, SNIPPET_PATH_COLUMN)
        return attr_value is str and attr_value
