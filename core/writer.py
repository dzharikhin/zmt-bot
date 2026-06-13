import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config
from audio.features import CombinedExtractor
from audio.segments import SegmentSpec
from core.jobs import JobManager
from core.paths import compute_file_hash
from core.storage import FeatureStore, JobStore

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    ok: int
    failed: int
    skipped: int


def _worker_loop(
    worker_id: int,
    tasks: list[tuple[Path, str, str, dict]],
    panns_weights_path: Path,
    user_id: int,
    embed_version: str,
    segment_policy: str,
    result_queue: mp.Queue,
    profile_path: Optional[Path],
    segment_spec: Optional[SegmentSpec],
):
    from core.logging import setup_logging

    setup_logging(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info(f"Worker {worker_id}: Starting with {len(tasks)} tasks")
    extractor = CombinedExtractor(panns_weights_path, profile_path=profile_path)
    store = FeatureStore(user_id, embed_version, segment_policy)

    for idx, (track_path, file_hash, set_name, metadata) in enumerate(tasks):
        try:
            vector = extractor(track_path, segment_spec=segment_spec)
            store.insert_track(
                file_hash=file_hash,
                vector=vector.tolist(),
                source_path=str(track_path),
                set_name=set_name,
                duration_s=metadata.get("duration_s", 0.0),
                bytes_=metadata.get("bytes_"),
                mtime=metadata.get("mtime"),
                sample_rate=metadata.get("sample_rate", 16000),
            )
            logger.info(f"Worker {worker_id}: OK {track_path.name}")
            result_queue.put((file_hash, True, None))
            if (idx + 1) % 10 == 0:
                logger.info(
                    f"Worker {worker_id}: Completed {idx + 1}/{len(tasks)} tracks"
                )
        except Exception as e:
            logger.error(f"Worker {worker_id}: FAILED {track_path.name}: {e}")
            result_queue.put((file_hash, False, str(e)))

    logger.info(f"Worker {worker_id}: Finished, processed {len(tasks)} tracks")


def start_extraction_job(
    user_id: int,
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

    store = FeatureStore(user_id, embed_version, segment_policy)

    # Probe phase
    to_extract = []
    skipped = 0
    for track_path, set_name in tracks:
        file_hash = compute_file_hash(track_path)
        if store.has(file_hash, set_name):
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
    logger.info(
        f"Job {job_id}: Cache probe complete - {skipped} cached, "
        f"{len(to_extract)} to extract, {total} total"
    )

    job_store = JobStore(user_id)
    job_mgr = JobManager(job_store)
    job_mgr.start_job(
        job_id, "extraction", {"total": total, "to_extract": len(to_extract)}
    )

    if progress_callback:
        progress_callback(job_id, 0, total, "starting")

    if not to_extract:
        job_mgr.update_progress(job_id, progress_done=skipped, progress_total=total)
        job_mgr.complete_job(job_id)
        job_store.close()
        if progress_callback:
            progress_callback(job_id, total, total, "complete")
        return ExtractionResult(ok=0, failed=0, skipped=skipped)

    job_store.close()

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
                user_id,
                embed_version,
                segment_policy,
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

    job_store = JobStore(user_id)
    job_mgr = JobManager(job_store)

    for _ in range(expected):
        file_hash, success, error_msg = result_queue.get()
        if success:
            ok += 1
            logger.info(f"Job {job_id}: {file_hash} — OK")
        else:
            failed += 1
            logger.error(f"Job {job_id}: {file_hash} — FAILED: {error_msg}")

        current_done = ok + failed + skipped
        if current_done - last_progress >= 10 or current_done == total:
            job_mgr.update_progress(
                job_id, progress_done=current_done, progress_total=total
            )
            if progress_callback:
                progress_callback(
                    job_id,
                    current_done,
                    total,
                    "running",
                    ok=ok,
                    failed=failed,
                    skipped=skipped,
                )
            last_progress = current_done

    logger.info(f"Job {job_id}: All workers finished - ok={ok}, failed={failed}")

    for w in workers:
        w.join(timeout=30)

    job_mgr.complete_job(job_id)
    job_store.close()
    if progress_callback:
        progress_callback(
            job_id, total, total, "complete", ok=ok, failed=failed, skipped=skipped
        )

    logger.info(
        f"Job {job_id}: Completed - total={total}, ok={ok}, failed={failed}, "
        f"skipped={skipped}"
    )

    return ExtractionResult(ok=ok, failed=failed, skipped=skipped)
