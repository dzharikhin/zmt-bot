import duckdb
import numpy as np
import pytest

from core.storage import FeatureStore


def _write_parquet_with_vectors(path, vectors):
    conn = duckdb.connect()
    try:
        conn.execute("CREATE TABLE tmp (vector FLOAT[])")
        for vec in vectors:
            vec_sql = f"[{','.join(str(v) for v in vec)}]"
            conn.execute(f"INSERT INTO tmp VALUES ({vec_sql})")
        conn.execute(f"COPY tmp TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        conn.execute("DROP TABLE tmp")
    finally:
        conn.close()


def _write_empty_parquet(path):
    conn = duckdb.connect()
    try:
        conn.execute("CREATE TABLE tmp (vector FLOAT[])")
        conn.execute(f"COPY tmp TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        conn.execute("DROP TABLE tmp")
    finally:
        conn.close()


class TestLoadVectorsRaisesOnInhomogeneousShapes:
    def test_raises_on_inhomogeneous_shapes(self, tmp_path):
        parquet_path = tmp_path / "inhomogeneous.parquet"
        _write_parquet_with_vectors(
            parquet_path,
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]],
        )
        with pytest.raises(RuntimeError, match="Inhomogeneous"):
            FeatureStore.load_vectors(parquet_path)
        with pytest.raises(RuntimeError, match=str(parquet_path)):
            FeatureStore.load_vectors(parquet_path)


class TestLoadVectorsReturnsStackedArrayOnHomogeneous:
    def test_returns_stacked_array_on_homogeneous(self, tmp_path):
        parquet_path = tmp_path / "homogeneous.parquet"
        _write_parquet_with_vectors(
            parquet_path,
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
            ],
        )
        result = FeatureStore.load_vectors(parquet_path)
        assert result.shape == (3, 5)
        assert result.dtype == np.float32


class TestLoadVectorsReturnsEmptyOnNoRows:
    def test_returns_empty_on_no_rows(self, tmp_path):
        parquet_path = tmp_path / "empty.parquet"
        _write_empty_parquet(parquet_path)
        result = FeatureStore.load_vectors(parquet_path)
        assert isinstance(result, np.ndarray)
        assert result.size == 0
