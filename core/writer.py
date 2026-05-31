import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config
from audio.features import CombinedExtractor
from core.storage import DuckDBStorage
from core.jobs import JobManager
from core.paths import compute_file_hash

logger = logging.getLogger(__name__)


class FeatureWriter:
    def __init__(
        self,
        db_path: Path,
        task_queue: mp.Queue,
        ack_queue: mp.Queue,
        heartbeat_interval: float = 2.0,
        heartbeat_path: Path = None,
    ):
        self.db_path = db_path
        self.task_queue = task_queue
        self.ack_queue = ack_queue
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_path = heartbeat_path
        self.running = True

    def run(self):
        storage = DuckDBStorage(self.db_path)
        last_heartbeat = 0

        while self.running:
            try:
                try:
                    item = self.task_queue.get(timeout=1.0)
                except Exception:
                    item = None

                if item is None:
                    now = time.time()
                    if now - last_heartbeat > self.heartbeat_interval:
                        if self.heartbeat_path:
                            self.heartbeat_path.touch()
                        last_heartbeat = now
                    continue

                if item == "STOP":
                    self.running = False
                    break

                file_hash, vector, metadata = item
                storage.insert_track(file_hash, vector, **metadata)

                self.ack_queue.put(file_hash)

                now = time.time()
                if now - last_heartbeat > self.heartbeat_interval:
                    if self.heartbeat_path:
                        self.heartbeat_path.touch()
                    last_heartbeat = now

            except Exception as e:
                logger.error(f"Writer error: {e}")

        storage.close()

    def stop(self):
        self.running = False


@dataclass
class ExtractionResult:
    ok: int
    failed: int
    skipped: int


def _extract_one(task: tuple) -> tuple:
    track_path, file_hash, set_name, metadata = task

    try:
        extractor = _get_or_create_extractor()
        vector = extractor(track_path)
        return (file_hash, vector, metadata)
    except Exception as e:
        metadata_with_error = {
            **metadata,
            "error_code": "E_DECODE_FAILED",
            "error_msg": str(e),
        }
        return (file_hash, None, metadata_with_error)


_extractor: Optional["CombinedExtractor"] = None


def _get_or_create_extractor():
    global _extractor
    if _extractor is None:
        _extractor = CombinedExtractor(config.panns_weights_path)
    return _extractor


def _worker_loop(
    worker_id: int, tasks: list, panns_weights_path: Path, result_queue: mp.Queue
):
    extractor = CombinedExtractor(panns_weights_path)

    for task in tasks:
        track_path, file_hash, set_name, metadata = task
        try:
            vector = extractor(track_path)
            result_queue.put((file_hash, vector, metadata))
        except Exception as e:
            metadata_with_error = {
                **metadata,
                "error_code": "E_DECODE_FAILED",
                "error_msg": str(e),
            }
            result_queue.put((file_hash, None, metadata_with_error))


def start_extraction_job(
    db_path: Path,
    tracks: list[tuple[Path, str]],
    embed_version: str,
    segment_policy: str,
    job_id: str,
    n_workers: int = 4,
    panns_weights_path: Optional[Path] = None,
) -> ExtractionResult:
    if panns_weights_path is None:
        panns_weights_path = config.panns_weights_path

    storage = DuckDBStorage(db_path)

    to_extract = []
    skipped = 0

    for track_path, set_name in tracks:
        file_hash = compute_file_hash(track_path)
        cached = storage.probe_cache(file_hash, embed_version, segment_policy)
        if cached is not None:
            skipped += 1
        else:
            stat = track_path.stat()
            metadata = {
                "source_path": str(track_path),
                "set_name": set_name,
                "duration_s": 0.0,
                "segment_policy": segment_policy,
                "embed_version": embed_version,
                "bytes_": stat.st_size,
                "mtime": stat.st_mtime,
                "sample_rate": 16000,
            }
            to_extract.append((track_path, file_hash, set_name, metadata))

    logger.info(f"Cache probe: {skipped} cached, {len(to_extract)} to extract")

    total = skipped + len(to_extract)
    job_mgr = JobManager(storage, config.jobs_root)
    job_mgr.start_job(
        job_id, "extraction", {"total": total, "to_extract": len(to_extract)}
    )

    if not to_extract:
        job_mgr.update_progress(job_id, progress_done=skipped, progress_total=total)
        job_mgr.complete_job(job_id)
        storage.close()
        return ExtractionResult(ok=0, failed=0, skipped=skipped)

    storage.close()

    task_queue = mp.Queue()
    ack_queue = mp.Queue()
    heartbeat_path = db_path.parent / ".writer_heartbeat"

    writer = FeatureWriter(
        db_path, task_queue, ack_queue, heartbeat_path=heartbeat_path
    )
    writer_proc = mp.Process(target=writer.run, daemon=True)
    writer_proc.start()

    result_queue = mp.Queue()

    _spawn = mp.get_context("spawn")
    workers = []
    for i in range(n_workers):
        p = _spawn.Process(
            target=_worker_loop,
            args=(i, to_extract[i::n_workers], panns_weights_path, result_queue),
            daemon=True,
        )
        p.start()
        workers.append(p)

    ok = 0
    failed = 0
    expected = len(to_extract)

    for _ in range(expected):
        file_hash, vector, metadata = result_queue.get()

        if vector is not None:
            task_queue.put((file_hash, vector, metadata))
            acked_hash = ack_queue.get(timeout=config.worker_ack_timeout_seconds)
            assert acked_hash == file_hash
            ok += 1
        else:
            task_queue.put((file_hash, [], metadata))
            acked_hash = ack_queue.get(timeout=config.worker_ack_timeout_seconds)
            failed += 1

    for w in workers:
        w.join(timeout=30)

    task_queue.put("STOP")
    writer_proc.join(timeout=30)

    if heartbeat_path.exists():
        heartbeat_path.unlink()

    storage = DuckDBStorage(db_path)
    job_mgr = JobManager(storage, config.jobs_root)
    job_mgr.complete_job(job_id)
    storage.close()

    return ExtractionResult(ok=ok, failed=failed, skipped=skipped)
