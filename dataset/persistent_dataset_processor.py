import dataclasses
import logging
import pathlib
import shutil
import sys
import tempfile
from collections import OrderedDict
from enum import Enum
from functools import reduce
from typing import (
    Generator,
    TypeVar,
    Callable,
    Generic,
    get_type_hints,
    NamedTuple,
    Any,
    Optional,
    TypedDict,
)

import atomics
import polars as pl
import polars.datatypes as pldt

Id = TypeVar("Id", int, str)
FeatureLineType = TypeVar("FeatureLineType")


class DataFrameBuilder(Generic[Id, FeatureLineType]):
    PROCESSING_SUCCEED_VALUE = "ok"

    class MappingParams(TypedDict):
        mapper: Callable[[Id], FeatureLineType]
        type_schema: OrderedDict[str, pldt.DataTypeClass | type]

    @dataclasses.dataclass()
    class Result:
        ldf: pl.LazyFrame
        processing_status_column: tuple[str, pldt.DataTypeClass]

    @dataclasses.dataclass()
    class ProgressStat:
        class StatPhase(Enum):
            BEFORE_ITERATION = 1
            ON_ITERATION = 2

        phase: StatPhase
        succeed: int
        failed: int
        data_size: int

    def __enter__(self) -> Result:
        self._intermediate_increased_data_path.unlink(missing_ok=True)

        not_processes_yet_df, init_stat = self._init_data()
        self._trigger_progress(init_stat)
        batch_index = 0
        while not (
            slice_df := not_processes_yet_df.slice(
                batch_index * self._batch_size, self._batch_size
            ),
            slice_df.limit(1).select(pl.nth(0)).collect(),
        )[-1].is_empty():
            increment_df = slice_df
            for i, mapper in enumerate(self._mappers):
                struct_alias = f"mapping_{i}"
                mapper_delegate, mapper_type = self._parse_mapper(mapper)
                increment_df = increment_df.with_columns(
                    pl.nth(0)
                    .map_elements(mapper_delegate, pl.Struct(mapper_type))
                    .alias(struct_alias)
                ).unnest(struct_alias)

            slice_file = self._working_dir.joinpath(f"slice_result_{batch_index}.ipc")
            increment_df.sink_ipc(
                slice_file,
                engine="streaming",
                sync_on_close="all",
                maintain_order=self._maintain_order,
            )
            (
                pl.concat(
                    [
                        pl.scan_ipc(self._intermediate_base_data_path),
                        pl.scan_ipc(slice_file),
                    ]
                ).sink_ipc(
                    self._intermediate_increased_data_path,
                    engine="streaming",
                    sync_on_close="all",
                    maintain_order=self._maintain_order,
                )
            )
            shutil.move(
                self._intermediate_increased_data_path,
                self._intermediate_base_data_path,
            )
            success_increment, failed_increment = self._get_stat(
                pl.scan_ipc(slice_file).select(
                    pl.col(self._processing_status_column_definition[0])
                )
            )
            slice_file.unlink(missing_ok=True)
            init_stat = DataFrameBuilder.ProgressStat(
                DataFrameBuilder.ProgressStat.StatPhase.ON_ITERATION,
                init_stat.succeed + success_increment,
                init_stat.failed + failed_increment,
                init_stat.data_size,
            )
            self._trigger_progress(init_stat)
            batch_index += 1
        return DataFrameBuilder.Result(
            pl.scan_ipc(self._intermediate_base_data_path),
            self._processing_status_column_definition,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_val and self._cleanup_on_exit:

            def handle_rm_error(func, error_on_path, exc_info):
                print(f"{error_on_path}: {exc_info}", file=sys.stderr)

            shutil.rmtree(self._working_dir, onexc=handle_rm_error)

    class GeneratingParams(NamedTuple):
        generator: Generator[Id, None, None]
        result_schema: tuple[str, pldt.DataTypeClass]

    def __init__(
        self,
        working_dir: pathlib.Path,
        index_generator: GeneratingParams,
        mappers: list[Callable[[Id], FeatureLineType] | MappingParams],
        *,
        batch_size: int = 1000,
        cleanup_on_exit: bool = True,
        progress_tracker: Optional[Callable[[ProgressStat], None]] = None,
        maintain_order: bool = False,
        processing_status_column_name: str = "processing_status",
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger if logger else logging.getLogger(DataFrameBuilder.__name__)
        working_dir.mkdir(exist_ok=True, parents=True)
        self._processing_status_column_definition = (
            processing_status_column_name,
            pl.String,
        )
        self._working_dir = working_dir
        self._intermediate_base_data_path = self._working_dir.joinpath("base.ipc")
        self._intermediate_increased_data_path = self._working_dir.joinpath(
            "increased.ipc"
        )
        self._index_generator = index_generator
        self._mappers = mappers
        self._progress_tracker = progress_tracker
        self._batch_size = batch_size
        self._backup_file = None
        self._maintain_order = maintain_order
        self._cleanup_on_exit = cleanup_on_exit
        self._id_dtype, self._intermediate_data_schema = self._init_schema()

    def _init_data(self) -> tuple[pl.LazyFrame, ProgressStat]:
        if self._intermediate_base_data_path.exists():
            existing_data_df = pl.scan_ipc(self._intermediate_base_data_path)
        else:
            existing_data_df = pl.LazyFrame([], schema=self._intermediate_data_schema)
            existing_data_df.sink_ipc(self._intermediate_base_data_path)
        index_buffer_file_path = pathlib.Path(self._working_dir).joinpath(
            "index_buffer"
        )

        total_count = 0
        with index_buffer_file_path.open(mode="wt") as index_buffer_file:
            for index_value in self._index_generator.generator:
                index_buffer_file.write(f"{index_value}\n")
                total_count += 1
        index_data = pl.scan_csv(
            index_buffer_file_path, has_header=False, schema={"row_id": self._id_dtype}
        )

        not_processes_yet_df = index_data
        ok_count = 0
        failed_count = 0
        if not existing_data_df.limit(1).select(pl.nth(0)).collect().is_empty():
            id_column_name = self._index_generator.result_schema[0]
            united_data = not_processes_yet_df.join(
                existing_data_df, id_column_name, "left"
            )
            ok_count, failed_count = self._get_stat(united_data)
            not_processes_yet_df = united_data.filter(
                pl.col(self._processing_status_column_definition[0]).is_null()
            ).select(id_column_name)
        return not_processes_yet_df, DataFrameBuilder.ProgressStat(
            phase=DataFrameBuilder.ProgressStat.StatPhase.BEFORE_ITERATION,
            succeed=ok_count,
            failed=failed_count,
            data_size=total_count,
        )

    def _get_stat(self, united_data: pl.LazyFrame) -> tuple[int, int]:
        sizes = (
            united_data.group_by(
                pl.col(self._processing_status_column_definition[0])
                .eq(DataFrameBuilder.PROCESSING_SUCCEED_VALUE)
                .alias("is_ok")
            )
            .agg(pl.nth(0).len())
            .collect(engine="streaming")
        )
        ok_count = (
            ok := sizes.filter(pl.col("is_ok")),
            ok.item(0, 1) if not ok.is_empty() else 0,
        )[-1]
        failed_count = (
            fails := sizes.filter(pl.col("is_ok").not_()),
            fails.item(0, 1) if not fails.is_empty() else 0,
        )[-1]
        return ok_count, failed_count

    def _init_schema(
        self,
    ) -> tuple[pldt.DataTypeClass, OrderedDict[str, pldt.DataTypeClass]]:
        mapping_schema = reduce(
            lambda a, b: a | b,
            (self._extract_mapping_schema(mapper) for mapper in self._mappers),
        )
        return (
            self._index_generator.result_schema[1],
            OrderedDict(
                [
                    self._index_generator.result_schema,
                    self._processing_status_column_definition,
                ]
            )
            | mapping_schema,
        )

    def _parse_mapper(
        self, mapper: Callable[[Id], FeatureLineType] | MappingParams
    ) -> tuple[Callable[[Id], dict[str, type]], dict[str, pldt.DataTypeClass]]:
        type_dict = OrderedDict(
            [self._processing_status_column_definition]
        ) | self._extract_mapping_schema(mapper)

        def map_row_by_id(row_id: Id) -> dict[str, Any]:
            try:
                result = (mapper if callable(mapper) else mapper["mapper"])(row_id)
            except Exception as e:
                stub = OrderedDict({k: None for k in type_dict.keys()}) | {
                    self._processing_status_column_definition[0]: str(e)
                }
                self.logger.warning(
                    f"Failed to extract features for {row_id=}, returning {stub}",
                    exc_info=e,
                )
                return stub

            if not dataclasses.is_dataclass(result):
                raise Exception(f"Mapper must return dataclass")

            dict_result_for_polars_struct = OrderedDict(
                [
                    (
                        self._processing_status_column_definition[0],
                        DataFrameBuilder.PROCESSING_SUCCEED_VALUE,
                    )
                ]
            ) | dataclasses.asdict(result)
            self.logger.debug(
                f"mapped {row_id=}: { {k: f"{type(v)}{v.shape if hasattr(v, "shape") else ""})" for k, v in dict_result_for_polars_struct.items()} }"
            )
            return dict_result_for_polars_struct

        return map_row_by_id, type_dict

    @staticmethod
    def _extract_mapping_schema(
        mapper: Callable[[Id], FeatureLineType] | MappingParams
    ) -> OrderedDict[str, pldt.DataTypeClass]:

        if isinstance(mapper, OrderedDict):
            schema_container = mapper["type_schema"]
        elif callable(mapper):
            schema_container = {
                field.name: field.type
                for field in dataclasses.fields(get_type_hints(mapper)["return"])
            }
        else:
            raise ValueError("Mappings are either Callable or Mapping with schema")
        return OrderedDict([(k, pldt.parse_into_dtype(v)) for k, v in schema_container.items()])

    def _trigger_progress(self, stat: ProgressStat):
        if self._progress_tracker:
            self._progress_tracker(stat)


if __name__ == "__main__":
    path = pathlib.Path("snippet-dataset.csv")
    counter = atomics.atomic(width=4, atype=atomics.INT)

    @dataclasses.dataclass
    class TestFeatures:
        a: float
        b: float
        c: float
        d: list[list[int]]

    def extract_features(row_id: str) -> TestFeatures:
        counter.fetch_inc() + 1
        if row_id == "3":
            raise Exception("failed to process")
        return TestFeatures(
            a=int(row_id) * 1.0,
            b=int(row_id) * 2.0,
            c=int(row_id) * 3.0,
            d=[[1, 2, 3], [4, 5, 6]],
        )

    (temp_base := pathlib.Path("tmp")).mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_base, delete=False) as tmp:
        with DataFrameBuilder(
            working_dir=pathlib.Path(tmp),
            index_generator=DataFrameBuilder.GeneratingParams(
                generator=(
                    f.name
                    for f in sorted(pathlib.Path("data").iterdir())
                    if f.is_file()
                ),
                result_schema=("row_id", pl.String),
            ),
            mappers=[
                OrderedDict(
                    mapper=extract_features,
                    type_schema={
                        "a": pl.Float64,
                        "b": float,
                        "c": float,
                        "d": list[list[int]],
                    },
                )
            ],
            batch_size=1,
        ) as result_frame:
            df = result_frame.ldf.collect(engine="streaming")
            print(f"{df=}")

