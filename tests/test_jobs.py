import json

import pytest

from core.storage import DuckDBStorage
from core.jobs import JobManager


@pytest.fixture
def job_manager(tmp_path):
    db_path = tmp_path / "test.duckdb"
    storage = DuckDBStorage(db_path)
    jobs_root = tmp_path / "jobs"
    mgr = JobManager(storage, jobs_root)
    yield mgr
    storage.close()


class TestStartJob:
    def test_creates_job_dir(self, job_manager):
        job_dir = job_manager.start_job("j1", "extraction", {"total": 10})
        assert job_dir.exists()
        assert (job_dir / "state.json").exists()

    def test_writes_db_record(self, job_manager):
        job_manager.start_job("j2", "extraction", {"total": 5})
        job = job_manager.storage.get_job("j2")
        assert job is not None
        assert job["kind"] == "extraction"
        assert job["status"] == "running"

    def test_state_json_content(self, job_manager):
        job_manager.start_job("j3", "extraction", {"total": 3})
        with open(job_manager.jobs_root / "j3" / "state.json") as f:
            state = json.load(f)
        assert state["status"] == "running"
        assert state["kind"] == "extraction"


class TestUpdateProgress:
    def test_updates_db(self, job_manager):
        job_manager.start_job("j4", "extraction", {"total": 10})
        job_manager.update_progress("j4", progress_done=5, progress_total=10)
        job = job_manager.storage.get_job("j4")
        assert job["progress_done"] == 5
        assert job["progress_total"] == 10

    def test_updates_state_json(self, job_manager):
        job_manager.start_job("j5", "extraction", {"total": 10})
        job_manager.update_progress("j5", progress_done=3, progress_total=10)
        with open(job_manager.jobs_root / "j5" / "state.json") as f:
            state = json.load(f)
        assert state["progress_done"] == 3


class TestCompleteJob:
    def test_sets_status_done(self, job_manager):
        job_manager.start_job("j6", "extraction", {"total": 10})
        job_manager.complete_job("j6")
        job = job_manager.storage.get_job("j6")
        assert job["status"] == "done"
        assert job["finished_at"] is not None

    def test_updates_state_json(self, job_manager):
        job_manager.start_job("j7", "extraction", {"total": 10})
        job_manager.complete_job("j7")
        with open(job_manager.jobs_root / "j7" / "state.json") as f:
            state = json.load(f)
        assert state["status"] == "done"


class TestFailJob:
    def test_sets_status_failed(self, job_manager):
        job_manager.start_job("j8", "extraction", {"total": 10})
        job_manager.fail_job("j8", "something broke")
        job = job_manager.storage.get_job("j8")
        assert job["status"] == "failed"
        assert job["error_json"] is not None

    def test_updates_state_json_with_error(self, job_manager):
        job_manager.start_job("j9", "extraction", {"total": 10})
        job_manager.fail_job("j9", "disk full")
        with open(job_manager.jobs_root / "j9" / "state.json") as f:
            state = json.load(f)
        assert state["status"] == "failed"
        assert state["error"] == "disk full"
