import numpy as np
import pytest

import config
from core.storage import FeatureStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "data_path", tmp_path / "data")
    monkeypatch.setattr(config, "local_data_path", tmp_path / "local_data")
    return FeatureStore(user_id=123, embed_version="v1", segment_policy="full")


class TestInsertAndProbe:
    def test_insert_and_probe(self, store, tmp_path):
        store.insert_track(
            file_hash="abc123",
            vector=[1.0, 2.0, 3.0],
            source_path="/test.mp3",
            set_name="like",
            duration_s=180.0,
        )
        assert store.has("abc123", "like") is True
        assert store.has("abc123", "dislike") is False
        assert store.has("nonexistent", "like") is False


class TestInsertWritesParquet:
    def test_insert_writes_parquet(self, store, tmp_path):
        store.insert_track(
            file_hash="def456",
            vector=[4.0, 5.0, 6.0],
            source_path="/test2.mp3",
            set_name="dislike",
            duration_s=120.0,
        )
        partition_dir = store.partition_dir("dislike")
        parquet_file = partition_dir / "def456.parquet"
        assert parquet_file.exists()


class TestInsertAtomic:
    def test_insert_atomic(self, store, tmp_path):
        store.insert_track(
            file_hash="ghi789",
            vector=[7.0, 8.0, 9.0],
            source_path="/test3.mp3",
            set_name="like",
            duration_s=90.0,
        )
        partition_dir = store.partition_dir("like")
        tmp_files = list(partition_dir.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestListCachedHashes:
    def test_list_cached_hashes(self, store, tmp_path):
        for i, hash_val in enumerate(["hash1", "hash2", "hash3"]):
            store.insert_track(
                file_hash=hash_val,
                vector=[float(i), float(i + 1), float(i + 2)],
                source_path=f"/test{i}.mp3",
                set_name="like",
                duration_s=100.0,
            )
        cached = store.list_cached_hashes("like")
        assert cached == {"hash1", "hash2", "hash3"}


class TestCountTracks:
    def test_count_tracks(self, store, tmp_path):
        store.insert_track(
            file_hash="h1",
            vector=[1.0],
            source_path="/a.mp3",
            set_name="like",
            duration_s=100.0,
        )
        store.insert_track(
            file_hash="h2",
            vector=[2.0],
            source_path="/b.mp3",
            set_name="like",
            duration_s=100.0,
        )
        store.insert_track(
            file_hash="h3",
            vector=[3.0],
            source_path="/c.mp3",
            set_name="dislike",
            duration_s=100.0,
        )
        counts = store.count_tracks()
        assert counts == {"like": 2, "dislike": 1}


class TestTrainingViewMaterializes:
    def test_training_view_materializes(self, store, tmp_path):
        store.insert_track(
            file_hash="mat1",
            vector=[1.0, 2.0],
            source_path="/a.mp3",
            set_name="like",
            duration_s=100.0,
        )
        store.insert_track(
            file_hash="mat2",
            vector=[3.0, 4.0],
            source_path="/b.mp3",
            set_name="like",
            duration_s=100.0,
        )
        tmp_dir = tmp_path / "local_data" / "123" / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with store.training_view("like") as path:
            assert path.exists()
            merged_files = list(tmp_dir.glob("merged_*.parquet"))
            assert len(merged_files) == 1

        after_files = list(tmp_dir.glob("merged_*.parquet"))
        assert len(after_files) == 0


class TestTrainingViewCleanupOnException:
    def test_training_view_cleanup_on_exception(self, store, tmp_path):
        store.insert_track(
            file_hash="exc1",
            vector=[1.0],
            source_path="/a.mp3",
            set_name="like",
            duration_s=100.0,
        )
        tmp_dir = tmp_path / "local_data" / "123" / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(ValueError):
            with store.training_view("like") as path:
                assert path.exists()
                raise ValueError("test error")

        after_files = list(tmp_dir.glob("merged_*.parquet"))
        assert len(after_files) == 0


class TestLoadVectorsRoundtrip:
    def test_load_vectors_roundtrip(self, store, tmp_path):
        store.insert_track(
            file_hash="rt1",
            vector=[1.0, 2.0, 3.0, 4.0, 5.0],
            source_path="/a.mp3",
            set_name="like",
            duration_s=100.0,
        )

        with store.training_view("like") as path:
            vectors = FeatureStore.load_vectors(path)

        assert vectors.shape == (1, 5)
        np.testing.assert_array_almost_equal(vectors[0], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_load_vectors_empty(self, store, tmp_path):
        store.insert_track(
            file_hash="emp1",
            vector=[1.0],
            source_path="/a.mp3",
            set_name="like",
            duration_s=100.0,
        )
        tmp_dir = tmp_path / "local_data" / "123" / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with store.training_view("like") as path:
            vectors = FeatureStore.load_vectors(path)

        assert vectors.shape == (1, 1)


class TestProbeMissing:
    def test_probe_missing(self, store, tmp_path):
        assert store.has("nonexistent_hash", "like") is False
        assert store.has("another_missing", "dislike") is False
