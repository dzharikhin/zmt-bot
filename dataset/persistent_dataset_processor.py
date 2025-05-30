import functools
import pathlib
import shutil
import tempfile
from datetime import datetime
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

ID = TypeVar("ID", int, str)
MapElementsStrategy: TypeAlias = Literal["thread_local", "threading"] | None


class DatasetProcessor(Protocol[ID]):
    def fill(self, row_value_generator: Callable[[ID], tuple[ID, *tuple[Any, ...]]]):
        pass

    total_dataset_rows_count: int
    processed_rows_count: int
    to_process_rows_count: int
    row_schema: tuple[tuple[str, Type[ID]], *tuple[tuple[str, Type[Any]], ...]]


class DataSetFromDataManager(DatasetProcessor[ID]):

    def __init__(
        self,
        csv_path: pathlib.Path,
        *,
        row_schema: tuple[tuple[str, Type[ID]], *tuple[tuple[str, Type[Any]], ...]],
        index_generator: Generator[ID, None, None],
        intermediate_results_dir: pathlib.Path,
        batch_size: int = 1000,
    ):
        self._persist_path = csv_path
        self.row_schema = row_schema
        self._polars_row_schema = [
            (name, pl.DataType.from_python(column_type))
            for name, column_type in row_schema
        ]
        self._batch_size = batch_size
        self._intermediate_results_dir = intermediate_results_dir
        self._df = self._init_df(index_generator)
        size_df = (
            self._df.select(pl.nth(0), pl.nth(1))
            .group_by(pl.nth(1).is_null().not_().alias("processed"))
            .len()
            .collect(engine="streaming")
        )
        self.total_dataset_rows_count = size_df.select("len").sum().item()
        self.processed_rows_count = (
            size_df.filter(pl.col("processed").eq(True)).select("len").sum().item()
        )
        self.to_process_rows_count = (
            size_df.filter(pl.col("processed").eq(False)).select("len").sum().item()
        )
        print(f"Successfully inited {csv_path} manager. Ready to fill")

    def fill(self, row_value_generator: Callable[[ID], tuple[ID, *tuple[Any, ...]]]):
        mapper = functools.partial(self._transform_tuple_to_dict, row_value_generator)
        id_col_name = self._polars_row_schema[0][0]
        generated_data_col_name = "generated_row_data_as_struct"

        # def process_unprocessed_rows_in_batch(df: pl.DataFrame) -> pl.DataFrame:
        #     additional_data = (
        #         df.with_columns(
        #             pl.nth(0)
        #             .map_elements(
        #                 mapper,
        #                 pl.Struct(dict(self._polars_row_schema)),
        #             )
        #             .alias(generated_data_col_name)
        #         )
        #         .with_columns(pl.col(generated_data_col_name).struct.unnest())
        #         .drop(generated_data_col_name)
        #     )
        #     return df.update(additional_data, on=id_col_name, how="left")

        print(
            f"Processing {self.to_process_rows_count} of total {self.total_dataset_rows_count} rows with batch={self._batch_size}"
        )

        # self._df.filter(pl.nth(1).is_null()).map_batches(
        #     process_unprocessed_rows_in_batch, streamable=True
        # ).sink_csv(self._intermediate_results_dir.joinpath("result.csv"), batch_size=self._batch_size, engine="streaming", sync_on_close="all")
        (self._df
         .filter(pl.nth(1).is_null())
         .with_columns(
            pl.nth(0)
                .map_elements(
                    mapper,
                    pl.Struct(dict(self._polars_row_schema)),
                )
                .alias(generated_data_col_name)
        )
         .with_columns(pl.col(generated_data_col_name).struct.unnest())
         .drop(generated_data_col_name)
         .sink_csv(self._intermediate_results_dir.joinpath("result.csv"), batch_size=1, sync_on_close="all"))

        print(
            f"Successfully processed all of {self.total_dataset_rows_count} rows"
        )

    def __enter__(self) -> DatasetProcessor[ID]:
        if self._persist_path.exists():
            backup_name = f"{self._persist_path.name}.{int(datetime.timestamp(datetime.now()))}.bak"
            self._backup_file = self._intermediate_results_dir.joinpath(backup_name)
            shutil.copy(
                self._persist_path,
                self._backup_file,
            )
            self._tmp_result_file = self._intermediate_results_dir.joinpath("result.csv")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val and self._backup_file and self._backup_file.exists():
            shutil.copy(
                self._backup_file,
                self._persist_path.parent.joinpath(self._backup_file.name),
            )
        elif self._tmp_result_file and self._tmp_result_file.exists():
            shutil.copy(
                self._tmp_result_file,
                self._persist_path,
            )


    def remove_failures_in_place(self, failed_row_ids: Collection[ID]):
        tmp_file = self._intermediate_results_dir.joinpath("filtered_data.csv")
        drop_ids = list(failed_row_ids)
        pl.scan_csv(self._persist_path, schema=dict(self._polars_row_schema)).filter(
            pl.nth(0).is_in(drop_ids).not_()
        ).sink_csv(tmp_file)
        shutil.copy(tmp_file, self._persist_path)

    def _init_df(self, index_generator: Generator[ID, None, None]) -> pl.LazyFrame:
        schema_as_dict = dict(self._polars_row_schema)
        index_column_name = self._polars_row_schema[0][0]

        if not self._persist_path.exists() or self._persist_path.stat().st_size <= 0:
            with self._persist_path.open(mode="wt") as index_buffer_file:
                index_buffer_file.write(
                    f"{index_column_name},{",".join([name for name, _ in self.row_schema][1:])}\n"
                )
                for index_value in index_generator:
                    index_buffer_file.write(
                        f"{index_value}{"," * (len(self.row_schema) - 1)}\n"
                    )

        return pl.scan_csv(self._persist_path, schema=schema_as_dict)

    def _transform_tuple_to_dict(
        self,
        row_generator: Callable[[ID], tuple[ID, *tuple[Any, ...]]],
        row_id: ID,
    ) -> dict[ID, Any]:
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
    pl.show_versions()
    path = pathlib.Path("snippet-dataset.csv")
    counter = atomics.atomic(width=4, atype=atomics.INT)
    tmp_path = pathlib.Path("tmp")
    tmp_path.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_path, delete=False) as tmp:
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
        )
        with dataset_manager as ds:

            def generate_value(row_id: str) -> tuple[str, float, float, float]:
                done = counter.fetch_inc() + 1
                print(f"{done}/{ds.to_process_rows_count}: {row_id}")
                # if done == 4:
                #     raise Exception()
                return (
                    row_id,
                    int(row_id) * 1.0,
                    int(row_id) * 2.0,
                    int(row_id) * 3.0,
                )

            ds.fill(generate_value)
        # dataset_manager.remove_failures_in_place({"2"})
        path.unlink(missing_ok=True)
