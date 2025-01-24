import functools
import pathlib
import shutil
import tempfile
from datetime import datetime
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Generator,
    TypeVar,
    Protocol,
    TypeAlias,
    Literal,
    Type,
    Collection,
)

import atomics
import polars as pl
from polars import LazyFrame

ID = TypeVar("ID", int, str)
MapElementsStrategy: TypeAlias = Literal["thread_local", "threading"] | None


class DatasetProcessor(Protocol[ID]):
    def fill(self, row_value_generator: Callable[[ID], tuple[ID, *tuple[Any, ...]]]):
        pass

    size: int


class DataSetFromDataManager(DatasetProcessor[ID]):

    def __init__(
        self,
        csv_path: pathlib.Path,
        *,
        row_schema: tuple[tuple[str, Type[ID]], *tuple[tuple[str, Type[Any]], ...]],
        index_generator: Generator[ID, None, None],
        intermediate_results_dir: pathlib.Path,
        batch_size: int = 1000,
        map_elements_call_strategy: MapElementsStrategy = None,
        cache_fraction: float = 0,
    ):
        self._persist_path = csv_path
        self.row_schema = row_schema
        self._polars_row_schema = [
            (name, pl.DataType.from_python(column_type))
            for name, column_type in row_schema
        ]
        self.size = 0
        self.cache_fraction = cache_fraction
        self._map_elements_call_strategy = map_elements_call_strategy
        self._intermediate_results_dir = intermediate_results_dir
        self._batch_size = batch_size
        self._df = self._init_df(index_generator)
        self._backup_file: pathlib.Path | None

    def fill(self, row_value_generator: Callable[[ID], tuple[ID, *tuple[Any, ...]]]):
        mapper = functools.partial(self._transform_tuple_to_dict, row_value_generator)
        if self.cache_fraction > 0:
            mapper = lru_cache(maxsize=int(self.size * self.cache_fraction))(mapper)
        map_elements_kwargs = (
            {"strategy": self._map_elements_call_strategy}
            if self._map_elements_call_strategy
            else {}
        )
        id_col_name = self._polars_row_schema[0][0]
        generated_data_col_name = "generated_row_data_as_struct"

        def process_unprocessed_rows_in_batch(df: pl.DataFrame) -> pl.DataFrame:
            additional_data = (
                df.with_columns(
                    pl.nth(0)
                    .map_elements(
                        mapper,
                        pl.Struct(dict(self._polars_row_schema)),
                        **map_elements_kwargs,
                    )
                    .alias(generated_data_col_name)
                )
                .with_columns(pl.col(generated_data_col_name).struct.unnest())
                .drop(generated_data_col_name)
            )
            return df.update(additional_data, on=id_col_name, how="left")

        processed_size = (
            self._df.filter(pl.nth(1).is_null().not_())
            .select(pl.len())
            .collect(streaming=True)
            .item()
        )
        not_processes_yet_df = self._df.filter(pl.nth(1).is_null()).map_batches(
            process_unprocessed_rows_in_batch, streamable=True
        )
        print(f"Processing {self.size - processed_size} of total {self.size} rows")
        batch_index = 0
        while not (
            slice_df := not_processes_yet_df.slice(
                batch_index * self._batch_size, self._batch_size
            ).collect(streaming=True)
        ).is_empty():
            with self._persist_path.open("at") as sink:
                slice_df.write_csv(
                    sink,
                    include_header=(self._persist_path.stat().st_size == 0),
                )
            print(
                f"Batch [{batch_index * self._batch_size}, {(batch_index + 1) * self._batch_size}) of not processed yet rows is appended to {self._persist_path}"
            )
            batch_index += 1

    def __enter__(self) -> DatasetProcessor[ID]:
        if self._persist_path.exists():
            backup_name = f"{self._persist_path.name}.{int(datetime.timestamp(datetime.now()))}.bak"
            self._backup_file = self._intermediate_results_dir.joinpath(backup_name)
            shutil.copy(
                self._persist_path.name,
                self._backup_file,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val and self._backup_file and self._backup_file.exists():
            shutil.copy(
                self._backup_file,
                self._persist_path.parent.joinpath(self._backup_file.name),
            )

    def remove_failures_in_place(self, failed_row_ids: Collection[ID]):
        tmp_file = self._intermediate_results_dir.joinpath("cleared_data")
        drop_ids = list(failed_row_ids)
        pl.scan_csv(self._persist_path, schema=dict(self._polars_row_schema)).filter(
            pl.nth(0).is_in(drop_ids).not_()
        ).sink_csv(tmp_file)
        shutil.copy(tmp_file, self._persist_path)

    def _init_df(self, index_generator: Generator[ID, None, None]) -> LazyFrame:
        schema_as_dict = dict(self._polars_row_schema)
        index_column_name = self._polars_row_schema[0][0]

        if self._persist_path.exists() and self._persist_path.stat().st_size > 0:
            existing_data_df = pl.scan_csv(self._persist_path, schema=schema_as_dict)
        else:
            existing_data_df = pl.LazyFrame([], schema=schema_as_dict)

        print(
            f"Using {self._intermediate_results_dir} as tempdir for data initialization"
        )
        index_buffer_file_path = pathlib.Path(self._intermediate_results_dir).joinpath(
            "index_buffer"
        )
        with index_buffer_file_path.open(mode="wt") as index_buffer_file:
            for index_value in index_generator:
                self.size += 1
                index_buffer_file.write(f"{index_value}\n")

        whole_data_df = pl.scan_csv(
            index_buffer_file_path, schema=schema_as_dict, has_header=False
        )
        whole_data_df = whole_data_df.update(
            existing_data_df, on=index_column_name, how="left", include_nulls=True
        )
        data_file = pathlib.Path(self._intermediate_results_dir).joinpath("merged")
        whole_data_df.collect(streaming=True).write_csv(data_file)
        return pl.scan_csv(data_file, schema=schema_as_dict)

    def _transform_tuple_to_dict(
        self,
        row_generator: Callable[[ID], tuple[ID, *tuple[Any, ...]]],
        row_id: ID,
    ) -> dict[str:ID, str:Any]:
        result_tuple = row_generator(row_id)
        result = {}
        fails = {}
        for i, schema in enumerate(self.row_schema):
            element = result_tuple[i]
            if isinstance(element, schema[1]) or (i > 0 and element is None):
                result[schema[0]] = element
            else:
                fails[i] = (schema, element)

        if fails:
            raise Exception(
                ", ".join(
                    [
                        f"element on index {i}={val[1]} not matched by schema={val[0]}"
                        for i, val in fails.items()
                    ]
                )
            )
        return result

    def _build_slice_name(self, index):
        return self._intermediate_results_dir.joinpath(f"slice_{index}.csv")


if __name__ == "__main__":
    path = pathlib.Path("snippet-dataset.csv")
    counter = atomics.atomic(width=4, atype=atomics.INT)
    with tempfile.TemporaryDirectory() as tmp:
        dataset_manager = DataSetFromDataManager(
            path,
            row_schema=(
                ("track_id", str),
                ("col1", float),
                ("col2", float),
                ("col3", float),
            ),
            index_generator=(
                f.name for f in pathlib.Path("data").iterdir() if f.is_file()
            ),
            intermediate_results_dir=pathlib.Path(tmp),
            batch_size=1,
            cache_fraction=0,
        )
        with dataset_manager as ds:

            def generate_value(row_id: str) -> tuple[str, float, float, float]:
                done = counter.fetch_inc() + 1
                print(f"{done}/{ds.size}: {row_id}")
                return (
                    row_id,
                    int(row_id) * 1.0,
                    int(row_id) * 2.0,
                    int(row_id) * 3.0,
                )

            ds.fill(generate_value)
            print(f"totally called generation: {counter.load()}/{ds.size}")
        dataset_manager.remove_failures_in_place({"2"})
