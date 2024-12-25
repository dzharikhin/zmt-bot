import csv
import functools
import pathlib
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
        map_elements_call_strategy: MapElementsStrategy = None,
        cache_fraction: float = 0.3,
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
        self._df = self._init_df(index_generator)

    def fill(self, row_value_generator: Callable[[ID], tuple[ID, *tuple[Any, ...]]]):
        any_value_column_is_null = functools.reduce(
            lambda a, b: a.or_(b),
            [pl.col(column[0]).is_null() for column in self._polars_row_schema[1:]],
        )
        mapper = lru_cache(maxsize=int(self.size * self.cache_fraction))(
            lambda row_id: self._transform_tuple_to_dict(row_value_generator(row_id))
        )
        map_elements_kwargs = (
            {"strategy": self._map_elements_call_strategy}
            if self._map_elements_call_strategy
            else {}
        )

        def batch_mapper(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(
                pl.col(self._polars_row_schema[0][0])
                .map_elements(
                    mapper,
                    pl.Struct(dict(self._polars_row_schema)),
                    **map_elements_kwargs,
                )
                .struct.unnest()
            )

        additional_data = self._df.filter(any_value_column_is_null).map_batches(
            batch_mapper, streamable=True
        )
        self._df = self._df.update(additional_data, on=self.row_schema[0][0])

    def __enter__(self) -> DatasetProcessor[ID]:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        tmp_file = self._persist_path.parent.joinpath(f"{self._persist_path.name}.out")
        self._df.sink_csv(tmp_file, maintain_order=False)
        tmp_file.rename(self._persist_path)

    def _init_df(self, index_generator: Generator[ID, None, None]) -> LazyFrame:
        if not self._persist_path.exists() or self._persist_path.stat().st_size == 0:
            with self._persist_path.open(mode="a") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([name for name, _ in self._polars_row_schema])
                for index in index_generator:
                    self.size += 1
                    writer.writerow(
                        [index] + [None] * (len(self._polars_row_schema) - 1)
                    )
        else:
            with self._persist_path.open(mode="rb") as f:
                self.size = sum(1 for _ in f) - 1
        return pl.scan_csv(self._persist_path, schema=dict(self._polars_row_schema))

    def _transform_tuple_to_dict(
        self,
        result_tuple: tuple[ID, *tuple[Any, ...]],
    ) -> dict[str:ID, str:Any]:
        result = {}
        fails = {}
        for i, schema in enumerate(self.row_schema):
            element = result_tuple[i]
            if isinstance(element, schema[1]):
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


if __name__ == "__main__":
    path = pathlib.Path("snippet-dataset.csv")

    with DataSetFromDataManager(
        path,
        row_schema=(
            ("track_id", str),
            ("col1", float),
            ("col2", float),
            ("col3", float),
        ),
        index_generator=(f.name for f in pathlib.Path("data").iterdir() if f.is_file()),
        cache_fraction=5,
    ) as ds:
        counter = atomics.atomic(width=4, atype=atomics.INT)

        def generate_value(row_id: str) -> tuple[str, float, float, float]:
            print(row_id)
            counter.inc()
            done = counter.load()
            if done % 2 == 0:
                print(f"{done}/1000\n")
            return (
                row_id,
                1.0,
                2.0,
                3.0,
            )

        ds.fill(generate_value)
