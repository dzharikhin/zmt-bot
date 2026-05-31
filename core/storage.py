import logging
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
  file_hash      TEXT NOT NULL,
  source_path    TEXT NOT NULL,
  set_name       TEXT NOT NULL CHECK (set_name IN ('like','dislike')),
  bytes          BIGINT,
  mtime          DOUBLE,
  duration_s     DOUBLE,
  sample_rate    INTEGER,
  segment_policy TEXT NOT NULL,
  embed_version  TEXT NOT NULL,
  extracted_at   TIMESTAMP,
  status         TEXT NOT NULL CHECK (status IN ('ok', 'failed', 'in_progress')),
  error_code     TEXT,
  error_msg      TEXT,
  vector         FLOAT[],
  PRIMARY KEY (file_hash, embed_version, segment_policy)
);
CREATE INDEX IF NOT EXISTS tracks_set_status ON tracks(set_name, status);

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


class DuckDBStorage:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self.conn.execute(SCHEMA)
        self.conn.execute("PRAGMA wal_autocheckpoint='1GB'")

    def probe_cache(
        self,
        file_hash: str,
        embed_version: str,
        segment_policy: str,
    ) -> Optional[np.ndarray]:
        result = self.conn.execute(
            """
            SELECT vector, status FROM tracks
            WHERE file_hash = ? AND embed_version = ? AND segment_policy = ?
        """,
            [file_hash, embed_version, segment_policy],
        ).fetchone()

        if result and result[1] == "ok":
            return np.asarray(result[0], dtype=np.float32)
        return None

    def insert_track(
        self,
        file_hash: str,
        vector: list[float],
        source_path: str,
        set_name: str,
        duration_s: float,
        segment_policy: str,
        embed_version: str,
        bytes_: int = None,
        mtime: float = None,
        sample_rate: int = None,
        error_code: str = None,
        error_msg: str = None,
    ):
        status = "ok" if error_code is None else "failed"

        self.conn.execute(
            """
            INSERT INTO tracks
            (file_hash, source_path, set_name, bytes, mtime, duration_s, sample_rate,
             segment_policy, embed_version, extracted_at, status, error_code, error_msg, vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
            ON CONFLICT (file_hash, embed_version, segment_policy) DO NOTHING
        """,
            [
                file_hash,
                source_path,
                set_name,
                bytes_,
                mtime,
                duration_s,
                sample_rate,
                segment_policy,
                embed_version,
                status,
                error_code,
                error_msg,
                vector,
            ],
        )

    def load_features(
        self,
        set_name: str,
        status: str = "ok",
        embed_version: str | None = None,
    ) -> np.ndarray:
        if embed_version is not None:
            result = self.conn.execute(
                """
                SELECT vector FROM tracks
                WHERE set_name = ? AND status = ? AND embed_version = ?
            """,
                [set_name, status, embed_version],
            ).fetchall()
        else:
            result = self.conn.execute(
                """
                SELECT vector FROM tracks
                WHERE set_name = ? AND status = ?
            """,
                [set_name, status],
            ).fetchall()

        if not result:
            return np.array([])

        return np.array([np.asarray(row[0], dtype=np.float32) for row in result])

    def update_job(
        self,
        job_id: str,
        kind: str = None,
        status: str = None,
        progress_total: int = None,
        progress_done: int = None,
        started_at: str = None,
        last_heartbeat_at: str = None,
        finished_at: str = None,
        params_json: str = None,
        error_json: str = None,
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

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
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
