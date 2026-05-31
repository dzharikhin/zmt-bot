import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from core.storage import DuckDBStorage

logger = logging.getLogger(__name__)


class JobManager:
    def __init__(self, storage: DuckDBStorage, jobs_root: Path):
        self.storage = storage
        self.jobs_root = jobs_root
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def start_job(self, job_id: str, kind: str, params: dict) -> Path:
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        self.storage.update_job(
            job_id=job_id,
            kind=kind,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
            params_json=json.dumps(params),
        )

        state = {
            "job_id": job_id,
            "kind": kind,
            "status": "running",
            "progress_total": 0,
            "progress_done": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(job_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        return job_dir

    def update_progress(self, job_id: str, progress_done: int, progress_total: int):
        job_dir = self.jobs_root / job_id

        self.storage.update_job(
            job_id=job_id,
            progress_done=progress_done,
            progress_total=progress_total,
            last_heartbeat_at=datetime.now(timezone.utc).isoformat(),
        )

        state_path = job_dir / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)

            state["progress_done"] = progress_done
            state["progress_total"] = progress_total
            state["last_heartbeat_at"] = datetime.now(timezone.utc).isoformat()

            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)

    def complete_job(self, job_id: str):
        job_dir = self.jobs_root / job_id

        self.storage.update_job(
            job_id=job_id,
            status="done",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

        state_path = job_dir / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)

            state["status"] = "done"
            state["finished_at"] = datetime.now(timezone.utc).isoformat()

            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)

    def fail_job(self, job_id: str, error: str):
        job_dir = self.jobs_root / job_id

        self.storage.update_job(
            job_id=job_id,
            status="failed",
            error_json=json.dumps({"error": str(error)}),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

        state_path = job_dir / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)

            state["status"] = "failed"
            state["error"] = str(error)
            state["finished_at"] = datetime.now(timezone.utc).isoformat()

            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
