import numpy as np
import pytest

from core.storage import DuckDBStorage


@pytest.fixture
def storage(tmp_path):
    db_path = tmp_path / "test.duckdb"
    s = DuckDBStorage(db_path)
    yield s
    s.close()


class TestDuckDBStorageInit:
    def test_creates_db_file(self, tmp_path):
        db_path = tmp_path / "subdir" / "test.duckdb"
        s = DuckDBStorage(db_path)
        assert db_path.exists()
        s.close()

    def test_creates_tables(self, storage):
        tables = storage.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "tracks" in table_names
        assert "jobs" in table_names


class TestProbeCache:
    def test_returns_none_for_missing_track(self, storage):
        result = storage.probe_cache("nonexistent_hash", "v1", "full")
        assert result is None

    def test_returns_vector_for_cached_track(self, storage):
        storage.insert_track(
            file_hash="abc123",
            vector=[1.0, 2.0, 3.0],
            source_path="/test.mp3",
            set_name="like",
            duration_s=180.0,
            segment_policy="full",
            embed_version="v1",
        )
        result = storage.probe_cache("abc123", "v1", "full")
        assert result is not None
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_returns_none_for_failed_track(self, storage):
        storage.insert_track(
            file_hash="fail_hash",
            vector=[],
            source_path="/fail.mp3",
            set_name="like",
            duration_s=0.0,
            segment_policy="full",
            embed_version="v1",
            error_code="E_DECODE_FAILED",
            error_msg="bad file",
        )
        result = storage.probe_cache("fail_hash", "v1", "full")
        assert result is None

    def test_returns_none_for_wrong_embed_version(self, storage):
        storage.insert_track(
            file_hash="ver_hash",
            vector=[1.0, 2.0],
            source_path="/test.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v1",
        )
        result = storage.probe_cache("ver_hash", "v2", "full")
        assert result is None

    def test_returns_none_for_wrong_segment_policy(self, storage):
        storage.insert_track(
            file_hash="seg_hash",
            vector=[1.0, 2.0],
            source_path="/test.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v1",
        )
        result = storage.probe_cache("seg_hash", "v1", "segmented")
        assert result is None


class TestInsertTrack:
    def test_insert_ok_track(self, storage):
        storage.insert_track(
            file_hash="ok_hash",
            vector=[0.5, 1.5, 2.5],
            source_path="/ok.mp3",
            set_name="like",
            duration_s=200.0,
            segment_policy="full",
            embed_version="v1",
            bytes_=1024,
            mtime=1234567890.0,
            sample_rate=16000,
        )
        row = storage.conn.execute(
            "SELECT status, error_code FROM tracks WHERE file_hash = 'ok_hash'"
        ).fetchone()
        assert row[0] == "ok"
        assert row[1] is None

    def test_insert_failed_track(self, storage):
        storage.insert_track(
            file_hash="err_hash",
            vector=[],
            source_path="/err.mp3",
            set_name="dislike",
            duration_s=0.0,
            segment_policy="full",
            embed_version="v1",
            error_code="E_DECODE_FAILED",
            error_msg="corrupt",
        )
        row = storage.conn.execute(
            "SELECT status, error_code, error_msg FROM tracks WHERE file_hash = 'err_hash'"
        ).fetchone()
        assert row[0] == "failed"
        assert row[1] == "E_DECODE_FAILED"
        assert row[2] == "corrupt"

    def test_insert_duplicate_does_nothing(self, storage):
        storage.insert_track(
            file_hash="dup_hash",
            vector=[1.0],
            source_path="/dup.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v1",
        )
        storage.insert_track(
            file_hash="dup_hash",
            vector=[2.0, 3.0],
            source_path="/dup2.mp3",
            set_name="dislike",
            duration_s=200.0,
            segment_policy="full",
            embed_version="v1",
        )
        row = storage.conn.execute(
            "SELECT source_path, vector FROM tracks WHERE file_hash = 'dup_hash'"
        ).fetchone()
        assert row[0] == "/dup.mp3"
        np.testing.assert_array_almost_equal(row[1], [1.0])

    def test_insert_same_hash_different_version(self, storage):
        storage.insert_track(
            file_hash="multi_ver",
            vector=[1.0],
            source_path="/a.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v1",
        )
        storage.insert_track(
            file_hash="multi_ver",
            vector=[2.0, 3.0],
            source_path="/a.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v2",
        )
        v1_result = storage.probe_cache("multi_ver", "v1", "full")
        v2_result = storage.probe_cache("multi_ver", "v2", "full")
        np.testing.assert_array_almost_equal(v1_result, [1.0])
        np.testing.assert_array_almost_equal(v2_result, [2.0, 3.0])


class TestLoadFeatures:
    def test_returns_empty_for_no_data(self, storage):
        result = storage.load_features("like")
        assert result.size == 0

    def test_returns_features_for_set(self, storage):
        storage.insert_track(
            file_hash="h1",
            vector=[1.0, 2.0],
            source_path="/a.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v1",
        )
        storage.insert_track(
            file_hash="h2",
            vector=[3.0, 4.0],
            source_path="/b.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v1",
        )
        storage.insert_track(
            file_hash="h3",
            vector=[5.0, 6.0],
            source_path="/c.mp3",
            set_name="dislike",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v1",
        )
        result = storage.load_features("like")
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0])

    def test_filters_by_embed_version(self, storage):
        storage.insert_track(
            file_hash="h1",
            vector=[1.0],
            source_path="/a.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v1",
        )
        storage.insert_track(
            file_hash="h2",
            vector=[2.0],
            source_path="/b.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v2",
        )
        result = storage.load_features("like", embed_version="v1")
        assert result.shape[0] == 1
        np.testing.assert_array_almost_equal(result[0], [1.0])

    def test_filters_by_segment_policy(self, storage):
        storage.insert_track(
            file_hash="h1",
            vector=[1.0],
            source_path="/a.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v1",
        )
        storage.insert_track(
            file_hash="h2",
            vector=[2.0],
            source_path="/b.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="topk_energy:W=30,K=3|agg=mean",
            embed_version="v1",
        )
        result = storage.load_features(
            "like", embed_version="v1", segment_policy="full"
        )
        assert result.shape[0] == 1
        np.testing.assert_array_almost_equal(result[0], [1.0])

    def test_filters_by_segment_policy_and_embed_version(self, storage):
        storage.insert_track(
            file_hash="h1",
            vector=[1.0],
            source_path="/a.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v1",
        )
        storage.insert_track(
            file_hash="h2",
            vector=[2.0],
            source_path="/b.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v2",
        )
        storage.insert_track(
            file_hash="h3",
            vector=[3.0],
            source_path="/c.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="topk_energy:W=30,K=3|agg=mean",
            embed_version="v1",
        )
        result = storage.load_features(
            "like", embed_version="v1", segment_policy="full"
        )
        assert result.shape[0] == 1
        np.testing.assert_array_almost_equal(result[0], [1.0])


class TestJobs:
    def test_update_and_get_job(self, storage):
        storage.update_job(
            job_id="job1",
            kind="extraction",
            status="running",
            progress_total=100,
            progress_done=0,
            started_at="2026-01-01T00:00:00",
        )
        job = storage.get_job("job1")
        assert job is not None
        assert job["job_id"] == "job1"
        assert job["kind"] == "extraction"
        assert job["status"] == "running"
        assert job["progress_total"] == 100

    def test_get_missing_job(self, storage):
        assert storage.get_job("nonexistent") is None

    def test_update_existing_job(self, storage):
        storage.update_job(
            job_id="job2",
            kind="extraction",
            status="running",
            progress_total=50,
            progress_done=0,
        )
        storage.update_job(
            job_id="job2",
            status="done",
            progress_done=50,
            finished_at="2026-01-01T01:00:00",
        )
        job = storage.get_job("job2")
        assert job["status"] == "done"
        assert job["progress_done"] == 50
        assert job["progress_total"] == 50
        assert job["kind"] == "extraction"


class TestCountTracks:
    def test_empty_db(self, storage):
        result = storage.count_tracks("v1", "full")
        assert result == {}

    def test_single_set_single_status(self, storage):
        storage.insert_track(
            file_hash="h1",
            vector=[1.0],
            source_path="/a.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v1",
        )
        result = storage.count_tracks("v1", "full")
        assert result == {"like": {"ok": 1}}

    def test_multiple_sets_and_statuses(self, storage):
        storage.insert_track(
            file_hash="h1",
            vector=[1.0],
            source_path="/a.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v1",
        )
        storage.insert_track(
            file_hash="h2",
            vector=[2.0],
            source_path="/b.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v1",
        )
        storage.insert_track(
            file_hash="h3",
            vector=[],
            source_path="/c.mp3",
            set_name="like",
            duration_s=0.0,
            segment_policy="full",
            embed_version="v1",
            error_code="E_DECODE_FAILED",
            error_msg="bad",
        )
        storage.insert_track(
            file_hash="h4",
            vector=[3.0],
            source_path="/d.mp3",
            set_name="dislike",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v1",
        )
        result = storage.count_tracks("v1", "full")
        assert result == {"like": {"ok": 2, "failed": 1}, "dislike": {"ok": 1}}

    def test_filters_by_embed_version(self, storage):
        storage.insert_track(
            file_hash="h1",
            vector=[1.0],
            source_path="/a.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v1",
        )
        storage.insert_track(
            file_hash="h2",
            vector=[2.0],
            source_path="/b.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v2",
        )
        assert storage.count_tracks("v1", "full") == {"like": {"ok": 1}}
        assert storage.count_tracks("v2", "full") == {"like": {"ok": 1}}

    def test_filters_by_segment_policy(self, storage):
        storage.insert_track(
            file_hash="h1",
            vector=[1.0],
            source_path="/a.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="full",
            embed_version="v1",
        )
        storage.insert_track(
            file_hash="h2",
            vector=[2.0],
            source_path="/b.mp3",
            set_name="like",
            duration_s=100.0,
            segment_policy="topk_energy:W=30,K=3|agg=mean",
            embed_version="v1",
        )
        assert storage.count_tracks("v1", "full") == {"like": {"ok": 1}}
        assert storage.count_tracks("v1", "topk_energy:W=30,K=3|agg=mean") == {
            "like": {"ok": 1}
        }
