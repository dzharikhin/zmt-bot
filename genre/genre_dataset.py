import pathlib
import re
from typing import NamedTuple, List

import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle


class Track(NamedTuple):
    Index: str


class Dataset:

    def __init__(self, csv: pathlib.Path, persist_per_update: bool = False):
        self.csv = csv
        self.result_path = csv.parent.joinpath(f"{csv.name}.prepared")
        self.dataframe: DataFrame = self._create_dataset()
        self.dataframe = shuffle(self.dataframe)
        self.persist_per_update = persist_per_update

    def _create_dataset(self) -> DataFrame:
        if self.result_path.exists():
            prepared_dataframe = pd.read_csv(self.result_path, keep_default_na=False)
            prepared_dataframe.set_index(["spotify_id"], inplace=True)
            prepared_dataframe.index.name = "spotify_id"
            return prepared_dataframe

        with self.csv.open(mode="r") as dataset:
            skip_indexes = {0}
            for index, line in enumerate(dataset):
                if index == 0:
                    headers = [
                        column.strip('"') for column in re.split(r",", line.strip())
                    ]
                if '"spotify_id' in line:
                    skip_indexes.add(index)

        raw_dataframe = pd.read_csv(
            self.csv, sep=";", header=None, names=headers, skiprows=list(skip_indexes)
        )
        raw_dataframe = raw_dataframe.drop_duplicates(subset=["spotify_id"])
        raw_dataframe.set_index(["spotify_id"], inplace=True, verify_integrity=True)
        raw_dataframe.index.name = "spotify_id"
        return raw_dataframe

    def remove_tracks(self, track_ids: List[str]):
        self.dataframe.drop(index=track_ids, inplace=True)

    def save(self):
        self.dataframe.to_csv(self.result_path, mode="w")
