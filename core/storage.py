import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

import config

logger = logging.getLogger(__name__)


class FeatureStore:
    def __init__(self, user_id: int, embed_version: str, segment_policy: str):
        self.user_id = user_id
        self.embed_version = embed_version
        self.segment_policy = segment_policy
        self._root = config.get_feature_store_root(user_id)

    def partition_dir(self, set_name: str) -> Path:
        d = self._root / self.embed_version / self.segment_policy / set_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def list_cached_hashes(self, set_name: str) -> set[str]:
        d = self.partition_dir(set_name)
        return {p.stem for p in d.glob("*.parquet")}

    def has(self, file_hash: str, set_name: str) -> bool:
        return (self.partition_dir(set_name) / f"{file_hash}.parquet").exists()

    def insert_track(
        self,
        file_hash: str,
        vector: list[float],
        source_path: str,
        set_name: str,
        duration_s: float,
        bytes_: int | None = None,
        mtime: float | None = None,
        sample_rate: int | None = None,
    ):
        d = self.partition_dir(set_name)
        tmp_path = d / f"{file_hash}.parquet.tmp"
        final_path = d / f"{file_hash}.parquet"

        vector_sql = f"[{','.join(str(v) for v in vector)}]"
        bytes_val = bytes_ if bytes_ is not None else "NULL"
        mtime_val = mtime if mtime is not None else "NULL"
        sample_rate_val = sample_rate if sample_rate is not None else "NULL"

        conn = duckdb.connect()
        try:
            conn.execute(
                "CREATE TABLE temp (file_hash VARCHAR, source_path VARCHAR, set_name VARCHAR, bytes BIGINT, mtime DOUBLE, duration_s DOUBLE, sample_rate INTEGER, extracted_at TIMESTAMP, vector FLOAT[])"
            )
            conn.execute(
                f"INSERT INTO temp VALUES ('{file_hash}', '{source_path}', '{set_name}', {bytes_val}, {mtime_val}, {duration_s}, {sample_rate_val}, CURRENT_TIMESTAMP, {vector_sql})"
            )
            conn.execute(
                f"COPY temp TO '{tmp_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            conn.execute("DROP TABLE temp")
        finally:
            conn.close()

        os.replace(tmp_path, final_path)

    def count_tracks(self) -> dict[str, int]:
        counts = {}
        for set_name in ("like", "dislike"):
            d = self.partition_dir(set_name)
            counts[set_name] = len(list(d.glob("*.parquet")))
        return counts

    @contextmanager
    def training_view(self, set_name: str):
        path = self._materialize(set_name)
        try:
            yield path
        finally:
            self._cleanup(path)

    def _materialize(self, set_name: str) -> Path:
        tmp_dir = config.get_training_tmp_dir(self.user_id)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        local_path = (
            tmp_dir
            / f"merged_{set_name}_{self.embed_version}_{self.segment_policy}_{ts}.parquet"
        )

        nas_dir = self.partition_dir(set_name)
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (SELECT vector FROM read_parquet('{nas_dir}/*.parquet'))
                TO '{local_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
        finally:
            conn.close()
        return local_path

    def _cleanup(self, path: Path):
        if path.exists():
            path.unlink()

    @staticmethod
    def load_vectors(parquet_path: Path) -> np.ndarray:
        conn = duckdb.connect()
        try:
            rows = conn.execute(
                f"SELECT vector FROM read_parquet('{parquet_path}')"
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return np.array([])
        arrays = [np.asarray(row[0], dtype=np.float32) for row in rows]
        shapes = {a.shape for a in arrays}
        if len(shapes) > 1:
            raise RuntimeError(
                f"Inhomogeneous vector shapes in {parquet_path}: "
                f"{sorted(shapes)}. Feature cache is corrupt; purge the "
                f"per-variant feature partition and re-extract."
            )
        return np.stack(arrays)


class JobStore:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS jobs (
      job_id            TEXT PRIMARY KEY,
      kind              TEXT NOT NULL,
      status            TEXT NOT NULL,
      progress_total    INTEGER,
      progress_done     INTEGER,
      started_at        TIMESTAMP,
      last_heartbeat_at TIMESTAMP,
      finished_at       TIMESTAMP,
      params_json       TEXT,
      error_json        TEXT
    );
    """

    def __init__(self, user_id: int):
        db_path = config.get_job_store_path(user_id)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(db_path))
        self.conn.execute(self.SCHEMA)

    def update_job(
        self,
        job_id: str,
        kind: str | None = None,
        status: str | None = None,
        progress_total: int | None = None,
        progress_done: int | None = None,
        started_at: str | None = None,
        last_heartbeat_at: str | None = None,
        finished_at: str | None = None,
        params_json: str | None = None,
        error_json: str | None = None,
    ):
        self.conn.execute(
            """
            INSERT INTO jobs (job_id, kind, status, progress_total, progress_done,
                              started_at, last_heartbeat_at, finished_at, params_json, error_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (job_id) DO UPDATE SET
                kind            = COALESCE(excluded.kind, jobs.kind),
                status          = COALESCE(excluded.status, jobs.status),
                progress_total  = COALESCE(excluded.progress_total, jobs.progress_total),
                progress_done   = COALESCE(excluded.progress_done, jobs.progress_done),
                started_at      = COALESCE(excluded.started_at, jobs.started_at),
                last_heartbeat_at = COALESCE(excluded.last_heartbeat_at, jobs.last_heartbeat_at),
                finished_at     = COALESCE(excluded.finished_at, jobs.finished_at),
                params_json     = COALESCE(excluded.params_json, jobs.params_json),
                error_json      = COALESCE(excluded.error_json, jobs.error_json)
            """,
            [
                job_id,
                kind,
                status,
                progress_total,
                progress_done,
                started_at,
                last_heartbeat_at,
                finished_at,
                params_json,
                error_json,
            ],
        )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        result = self.conn.execute(
            """
            SELECT * FROM jobs WHERE job_id = ?
            """,
            [job_id],
        ).fetchone()

        if not result:
            return None

        return {
            "job_id": result[0],
            "kind": result[1],
            "status": result[2],
            "progress_total": result[3],
            "progress_done": result[4],
            "started_at": result[5],
            "last_heartbeat_at": result[6],
            "finished_at": result[7],
            "params_json": result[8],
            "error_json": result[9],
        }

    def close(self):
        self.conn.close()
