import pathlib
import shutil
import sys
import tempfile
from collections import OrderedDict
from typing import (
    Generator,
    TypeVar,
    Protocol, Callable, NamedTuple, Generic)

import atomics
import polars as pl
import polars.datatypes as pldt

Id = TypeVar("Id", int, str)
FeatureLineType = TypeVar("FeatureLineType")

class DataSetFromDataManager(Protocol[Id]):
    RAW_FEATURES_COLUMN_NAME = "raw_features"

    def __enter__(self) -> pl.LazyFrame:
        self._intermediate_increased_data_path.unlink(missing_ok=True)

        not_processes_yet_df = self.init_data()
        batch_index = 0
        while not (slice_df := not_processes_yet_df.slice(batch_index * self._batch_size, self._batch_size), slice_df.limit(1).select(pl.nth(0)).collect())[-1].is_empty():
            increment_df = (
                slice_df
                .with_columns(pl.nth(0).map_elements(self._mapping.mapper, pl.Struct(self._mapping.result_types)).alias(self.RAW_FEATURES_COLUMN_NAME))
                .with_columns(pl.col(self.RAW_FEATURES_COLUMN_NAME).struct.unnest())
                .drop(self.RAW_FEATURES_COLUMN_NAME)
            )
            pl.concat([pl.scan_ipc(self._intermediate_base_data_path), increment_df]).sink_ipc(self._intermediate_increased_data_path, engine="streaming", sync_on_close="all")
            shutil.move(self._intermediate_increased_data_path, self._intermediate_base_data_path)
            batch_index += 1
        return pl.scan_ipc(self._intermediate_base_data_path)


    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_val:
            def handle_rm_error(func, error_on_path, exc_info):
                print(f"{error_on_path}: {exc_info}", file=sys.stderr)
            shutil.rmtree(self._working_dir, onexc=handle_rm_error)

    class GeneratingParams(NamedTuple):
        generator: Generator[Id, None, None]
        result_schema: tuple[str, pldt.DataTypeClass]

    class MappingParams(NamedTuple, Generic[FeatureLineType]):
        mapper: Callable[[Id], dict[str, FeatureLineType]]
        result_types: dict[str, pldt.DataTypeClass]


    def __init__(
        self,
        working_dir: pathlib.Path,
        index_generator: GeneratingParams,
        mapping: MappingParams,
        *,
        batch_size: int = 1000,
    ):
        self._working_dir = working_dir
        self._intermediate_base_data_path = self._working_dir.joinpath("base.ipc")
        self._intermediate_increased_data_path = self._working_dir.joinpath("increased.ipc")
        self._index_generator = index_generator
        self._mapping = mapping
        self._batch_size = batch_size
        self._backup_file = None


    def init_data(self) -> pl.LazyFrame:
        intermediate_data_schema = OrderedDict([self._index_generator.result_schema]) | self._mapping.result_types
        if self._intermediate_base_data_path.exists():
            existing_data_df = pl.scan_ipc(self._intermediate_base_data_path, schema=intermediate_data_schema)
        else:
            existing_data_df = pl.LazyFrame([], schema=intermediate_data_schema)
            existing_data_df.sink_ipc(self._intermediate_base_data_path)
        index_buffer_file_path = pathlib.Path(self._working_dir).joinpath("index_buffer")

        with index_buffer_file_path.open(mode="wt") as index_buffer_file:
            for index_value in self._index_generator.generator:
                index_buffer_file.write(
                    f"{index_value}\n"
                )
        index_data = pl.scan_csv(index_buffer_file_path, has_header=False, schema=(dict([self._index_generator.result_schema])))

        not_processes_yet_df = index_data
        if not existing_data_df.limit(1).select(pl.nth(0)).collect().is_empty():
                not_processes_yet_df = index_data.filter(pl.nth(0).is_in(existing_data_df.select(pl.nth(0).implode()).collect(engine="streaming")).not_())
        return not_processes_yet_df


if __name__ == "__main__":
    path = pathlib.Path("snippet-dataset.csv")
    counter = atomics.atomic(width=4, atype=atomics.INT)
    type TestFeature = int | float

    def extract_features(row_id: str) -> dict[str, TestFeature]:
        counter.fetch_inc() + 1
        return dict(a=int(row_id) * 1.0, b=int(row_id) * 2.0, c=int(row_id) * 3.0)

    (temp_base := pathlib.Path("tmp")).mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_base, delete=False) as tmp:
        with DataSetFromDataManager(
            working_dir=pathlib.Path(tmp),
            index_generator=DataSetFromDataManager.GeneratingParams(
                generator=(f.name for f in pathlib.Path("data").iterdir() if f.is_file()),
                result_schema=("row_id", pl.String)
            ),
            mapping=DataSetFromDataManager.MappingParams(extract_features, dict(a=pl.Float64, b=pl.Float64, c=pl.Float64)),
            batch_size=1,
        ) as result_frame:
            result_frame.filter(pl.nth(1).is_null().not_()).sink_csv(path, engine="streaming", sync_on_close="all")
