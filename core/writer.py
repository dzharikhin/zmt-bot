import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config
from audio.features import CombinedExtractor
from audio.segments import SegmentSpec
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
        items_processed = 0

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
                    logger.info(f"Writer received STOP, processed {items_processed} items")
                    self.running = False
                    break

                file_hash, vector, metadata = item
                storage.insert_track(file_hash, vector, **metadata)

                self.ack_queue.put(file_hash)
                items_processed += 1

                now = time.time()
                if now - last_heartbeat > self.heartbeat_interval:
                    if self.heartbeat_path:
                        self.heartbeat_path.touch()
                    last_heartbeat = now

            except Exception as e:
                logger.error(f"Writer error (items_processed={items_processed}): {e}", exc_info=True)

        logger.info(f"Writer shutting down, total items processed: {items_processed}")
        storage.close()

    def stop(self):
        self.running = False


@dataclass
class ExtractionResult:
    ok: int
    failed: int
    skipped: int


def _worker_loop(
    worker_id: int,
    tasks: list,
    panns_weights_path: Path,
    result_queue: mp.Queue,
    profile_path: Path | None = None,
    segment_spec: SegmentSpec | None = None,
):
    logger.info(f"Worker {worker_id}: Starting with {len(tasks)} tasks")
    extractor = CombinedExtractor(panns_weights_path, profile_path=profile_path)

    for idx, task in enumerate(tasks):
        track_path, file_hash, set_name, metadata = task
        try:
            vector = extractor(track_path, segment_spec=segment_spec)
            result_queue.put((file_hash, vector, metadata))
            if (idx + 1) % 10 == 0:
                logger.info(f"Worker {worker_id}: Completed {idx + 1}/{len(tasks)} tracks")
        except Exception as e:
            logger.exception(f"Worker {worker_id}: Failed to extract features from {track_path.name}: {e}")
            metadata_with_error = {
                **metadata,
                "error_code": "E_DECODE_FAILED",
                "error_msg": str(e),
            }
            result_queue.put((file_hash, None, metadata_with_error))

    logger.info(f"Worker {worker_id}: Finished, processed {len(tasks)} tracks")


def start_extraction_job(
    db_path: Path,
    tracks: list[tuple[Path, str]],
    embed_version: str,
    segment_policy: str,
    job_id: str,
    n_workers: int = 4,
    panns_weights_path: Optional[Path] = None,
    profile_path: Optional[Path] = None,
    segment_spec: Optional[SegmentSpec] = None,
    progress_callback=None,
) -> ExtractionResult:
    if panns_weights_path is None:
        panns_weights_path = config.panns_weights_path
    if segment_spec is None and segment_policy != "full":
        segment_spec = SegmentSpec.parse(segment_policy)

    storage = DuckDBStorage(db_path)

    to_extract = []
    skipped = 0

    for track_path, set_name in tracks:
        file_hash = compute_file_hash(track_path)
        cached = storage.probe_cache(file_hash, embed_version, segment_policy)
        if cached is not None:
            skipped += 1
            logger.debug(f"Cache hit: {track_path.name}")
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

    total = skipped + len(to_extract)
    logger.info(f"Job {job_id}: Cache probe complete - {skipped} cached, {len(to_extract)} to extract, {total} total")

    job_mgr = JobManager(storage, config.jobs_root)
    job_mgr.start_job(
        job_id, "extraction", {"total": total, "to_extract": len(to_extract)}
    )

    if progress_callback:
        progress_callback(job_id, 0, total, "starting")

    if not to_extract:
        job_mgr.update_progress(job_id, progress_done=skipped, progress_total=total)
        job_mgr.complete_job(job_id)
        storage.close()
        if progress_callback:
            progress_callback(job_id, total, total, "complete")
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
    logger.info(f"Job {job_id}: Writer process started (PID: {writer_proc.pid})")

    result_queue = mp.Queue()

    _spawn = mp.get_context("spawn")
    workers = []
    for i in range(n_workers):
        p = _spawn.Process(
            target=_worker_loop,
            args=(
                i,
                to_extract[i::n_workers],
                panns_weights_path,
                result_queue,
                profile_path,
                segment_spec,
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)
        logger.info(f"Job {job_id}: Worker {i} started (PID: {p.pid})")

    ok = 0
    failed = 0
    expected = len(to_extract)
    last_progress = 0

    logger.info(f"Job {job_id}: Processing {expected} tracks with {n_workers} workers")

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
            error_msg = metadata.get("error_msg", "Unknown error")
            logger.error(f"Job {job_id}: Feature extraction failed for {metadata.get('source_path')}: {error_msg}")

        current_done = ok + failed + skipped
        if current_done - last_progress >= 10 or current_done == total:
            job_mgr.update_progress(job_id, progress_done=current_done, progress_total=total)
            if progress_callback:
                progress_callback(job_id, current_done, total, "running", ok=ok, failed=failed, skipped=skipped)
            last_progress = current_done

    logger.info(f"Job {job_id}: All workers finished - ok={ok}, failed={failed}")

    for w in workers:
        w.join(timeout=30)

    task_queue.put("STOP")
    writer_proc.join(timeout=30)
    logger.info(f"Job {job_id}: Writer process stopped")

    if heartbeat_path.exists():
        heartbeat_path.unlink()

    storage = DuckDBStorage(db_path)
    job_mgr = JobManager(storage, config.jobs_root)
    job_mgr.complete_job(job_id)
    storage.close()
    if progress_callback:
        progress_callback(job_id, total, total, "complete", ok=ok, failed=failed, skipped=skipped)

    logger.info(f"Job {job_id}: Completed - total={total}, ok={ok}, failed={failed}, skipped={skipped}")

    return ExtractionResult(ok=ok, failed=failed, skipped=skipped)
