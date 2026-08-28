import os
from pathlib import Path

import duckdb
import pytest

from scripts.extract_corpus import _load_tracks, _resolve_from_embed


def _write_shard(root: Path, set_name: str, source_path: str, file_hash: str):
    out_dir = root / "embed_old" / "full" / set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    duckdb.sql(f"""
        COPY (
            SELECT * FROM (VALUES
                ('{file_hash}', '{source_path}', '{set_name}', 1.0)
            ) t(file_hash, source_path, set_name, duration_s)
        )
        TO '{out_dir}/{file_hash}.parquet' (FORMAT PARQUET)
        """)


def test_load_tracks_returns_distinct_source_paths(tmp_path):
    root = tmp_path / "features"
    audio = tmp_path / "liked"
    audio.mkdir()
    track = audio / "a.mp3"
    track.write_bytes(b"x")
    _write_shard(root, "like", str(track), "h1")
    _write_shard(root, "like", str(track), "h1")
    _write_shard(root, "dislike", "/gone/b.mp3", "h2")

    tracks = _load_tracks(root, "embed_old")

    assert tracks == [(track, "like")]


def test_resolve_from_embed_explicit_wins(tmp_path):
    root = tmp_path / "features"
    root.mkdir()
    assert _resolve_from_embed(root, "embed_given") == "embed_given"


def test_resolve_from_embed_picks_most_recent(tmp_path):
    root = tmp_path / "features"
    root.mkdir()
    old = root / "embed_old"
    new = root / "embed_new"
    old.mkdir()
    new.mkdir()
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    assert _resolve_from_embed(root, None) == "embed_new"


def test_resolve_from_embed_empty_root_raises(tmp_path):
    root = tmp_path / "features"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        _resolve_from_embed(root, None)
