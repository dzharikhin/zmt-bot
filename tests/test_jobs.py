import pytest

import config
from core.jobs import JobManager
from core.storage import JobStore


@pytest.fixture
def job_manager(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "local_data_path", tmp_path / "local_data")
    job_store = JobStore(user_id=123)
    mgr = JobManager(job_store)
    yield mgr
    job_store.close()


class TestStartJob:
    def test_start_and_complete_job(self, job_manager):
        job_manager.start_job("j1", "extraction", {"total": 10})
        job = job_manager.job_store.get_job("j1")
        assert job is not None
        assert job["kind"] == "extraction"
        assert job["status"] == "running"

        job_manager.complete_job("j1")
        job = job_manager.job_store.get_job("j1")
        assert job["status"] == "done"
        assert job["finished_at"] is not None

    def test_writes_db_record(self, job_manager):
        job_manager.start_job("j2", "extraction", {"total": 5})
        job = job_manager.job_store.get_job("j2")
        assert job is not None
        assert job["kind"] == "extraction"
        assert job["status"] == "running"


class TestUpdateProgress:
    def test_update_progress(self, job_manager):
        job_manager.start_job("j4", "extraction", {"total": 10})
        job_manager.update_progress("j4", progress_done=5, progress_total=10)
        job = job_manager.job_store.get_job("j4")
        assert job["progress_done"] == 5
        assert job["progress_total"] == 10


class TestCompleteJob:
    def test_sets_status_done(self, job_manager):
        job_manager.start_job("j6", "extraction", {"total": 10})
        job_manager.complete_job("j6")
        job = job_manager.job_store.get_job("j6")
        assert job["status"] == "done"
        assert job["finished_at"] is not None


class TestFailJob:
    def test_fail_job(self, job_manager):
        job_manager.start_job("j8", "extraction", {"total": 10})
        job_manager.fail_job("j8", "something broke")
        job = job_manager.job_store.get_job("j8")
        assert job["status"] == "failed"
        assert job["error_json"] is not None

    def test_get_job(self, job_manager):
        job_manager.start_job("j9", "extraction", {"total": 10})
        job = job_manager.job_store.get_job("j9")
        assert job is not None
        assert job["job_id"] == "j9"
        assert job["kind"] == "extraction"
        assert job["status"] == "running"

        missing = job_manager.job_store.get_job("nonexistent")
        assert missing is None
