import json
import logging
from datetime import datetime, timezone

from core.storage import JobStore

logger = logging.getLogger(__name__)


class JobManager:
    def __init__(self, job_store: JobStore):
        self.job_store = job_store

    def start_job(self, job_id: str, kind: str, params: dict) -> None:
        logger.info(f"Starting job {job_id} of kind '{kind}'")
        self.job_store.update_job(
            job_id=job_id,
            kind=kind,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
            params_json=json.dumps(params),
        )

    def update_progress(self, job_id: str, progress_done: int, progress_total: int):
        self.job_store.update_job(
            job_id=job_id,
            progress_done=progress_done,
            progress_total=progress_total,
            last_heartbeat_at=datetime.now(timezone.utc).isoformat(),
        )
        if progress_total > 0:
            pct = (progress_done / progress_total) * 100
            logger.debug(
                f"Job {job_id}: Progress {progress_done}/{progress_total} ({pct:.1f}%)"
            )

    def complete_job(self, job_id: str):
        logger.info(f"Completing job {job_id}")
        self.job_store.update_job(
            job_id=job_id,
            status="done",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    def fail_job(self, job_id: str, error: str):
        logger.error(f"Job {job_id} failed: {error}", exc_info=True)
        self.job_store.update_job(
            job_id=job_id,
            status="failed",
            error_json=json.dumps({"error": str(error)}),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
