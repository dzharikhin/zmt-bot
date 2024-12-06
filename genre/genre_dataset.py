import pathlib
import re
from typing import NamedTuple

import pandas as pd
from pandas import DataFrame, Series
from sklearn.utils import shuffle

SNIPPET_PATH_COLUMN = "snippet_path"


class Track(NamedTuple):
    spotify_id: str


class Dataset:

    def __init__(self, csv: pathlib.Path, persist_per_update: bool = False):
        self.csv = csv
        self.result_path = csv.parent.joinpath(f"{csv.name}.result")
        self.dataframe: DataFrame = self._create_dataset()
        self.dataframe = shuffle(self.dataframe)
        self.persist_per_update = persist_per_update

    def _create_dataset(self) -> DataFrame:
        if self.result_path.exists():
            return pd.read_csv(self.result_path, keep_default_na=False)

        with self.csv.open(mode="r") as dataset:
            skip_indexes = {0}
            for index, line in enumerate(dataset):
                if index == 0:
                    headers = [
                        column.strip('"') for column in re.split(r",", line.strip())
                    ]
                if '"spotify_id' in line:
                    skip_indexes.add(index)

        initial_dataframe = pd.read_csv(
            self.csv, sep=";", header=None, names=headers, skiprows=list(skip_indexes)
        )
        initial_dataframe = initial_dataframe.drop_duplicates(subset=["spotify_id"])
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

    def mark_snippet_failed(self, spotify_id: str):
        self.add_snippet_path(spotify_id, "FAILED")

    def save(self):
        self.dataframe.to_csv(self.result_path, mode="w", index=False)

    def stats(self) -> Series:
        return self.dataframe.where(
            ~self.dataframe[SNIPPET_PATH_COLUMN].isin(["", "FAILED"])
        )["genre_name"].value_counts()

    def has_snippet(self, row: Track) -> bool:
        attr_value = getattr(row, SNIPPET_PATH_COLUMN)
        has_path_in_file = attr_value is str and attr_value
        return (
            has_path_in_file
            or self.dataframe.loc[
                (self.dataframe["spotify_id"] == row.spotify_id)
                & (self.dataframe[SNIPPET_PATH_COLUMN].str.len() > 0),
                SNIPPET_PATH_COLUMN,
            ].all()
        )
