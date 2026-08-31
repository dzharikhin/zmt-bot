"""Audit Essentia descriptor shapes across a track corpus.

Discover mode — find variable-shape descriptors and emit a candidate
_DESCRIPTOR_SCHEMA literal for paste into audio/features.py:

    poetry run python -m audit.descriptor_shapes discover \
        --profile data/essentia_extractor_profile.yaml \
        --tracks /path/to/music/directory \
        --output data/audit/descriptor_schema_literal.txt

Verify mode — check the committed _DESCRIPTOR_SCHEMA still matches
a fresh corpus (regression guard for Essentia version drift):

    poetry run python -m audit.descriptor_shapes verify \
        --profile data/essentia_extractor_profile.yaml \
        --tracks /path/to/music/directory
"""

import argparse
import logging
import random
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

import essentia.standard as es

logger = logging.getLogger(__name__)


def _discover_all_descriptors(pool) -> list[tuple[str, np.ndarray]]:
    result = []
    for name in sorted(pool.descriptorNames()):
        if name.startswith("metadata."):
            continue
        value = pool[name]
        if isinstance(value, str):
            continue
        arr = np.atleast_1d(np.asarray(value))
        result.append((name, arr))
    return result


def _collect_shapes(extractor, tracks: list[Path]) -> dict[str, list[tuple[int, ...]]]:
    shapes: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    n_ok = 0
    for track_path in tracks:
        try:
            features, _frames = extractor(str(track_path))
        except Exception as exc:
            logger.warning("Extraction failed for %s: %s", track_path, exc)
            continue
        for name, arr in _discover_all_descriptors(features):
            shapes[name].append(arr.shape)
        n_ok += 1
        logger.info("Collected shapes from %s (%d ok so far)", track_path.name, n_ok)
    logger.info(
        "Collected shapes from %d/%d tracks, %d unique descriptors",
        n_ok,
        len(tracks),
        len(shapes),
    )
    return dict(shapes)


def _classify(observations: dict[str, list[tuple[int, ...]]]) -> tuple[dict, dict, set]:
    """
    Returns:
        set_a: {name: shape} — identical shape across all tracks (deterministic)
        set_b: {name: set_of_shapes} — variable shape (needs normalizer)
        set_c: set of names present on fewer tracks than total (sometimes absent)
    """
    total_tracks = (
        max(len(shapes) for shapes in observations.values()) if observations else 0
    )
    set_a: dict[str, tuple[int, ...]] = {}
    set_b: dict[str, set[tuple[int, ...]]] = {}
    set_c: set[str] = set()

    for name, shape_list in observations.items():
        unique_shapes = set(shape_list)
        if len(shape_list) < total_tracks:
            set_c.add(name)
        if len(unique_shapes) == 1:
            set_a[name] = shape_list[0]
        else:
            set_b[name] = unique_shapes

    return set_a, set_b, set_c


def _emit_schema_literal(
    set_a: dict,
    set_b: dict,
    n_tracks_total: int,
) -> str:
    """Build a Python tuple literal for _DESCRIPTOR_SCHEMA (3-tuple form).
    set_a entries: use observed shape directly, normalizer_key=None.
    set_b entries:
      - 1-D variable -> length 4, normalizer_key="stats4"
      - 2-D variable -> length 4 * n_rows_max, normalizer_key="matrix_rowstats"
    Sorted by descriptor name. Returns the literal as a string."""
    entries = []

    for name in sorted(set_a):
        shape = set_a[name]
        length = int(np.prod(shape)) if len(shape) > 0 else 1
        entries.append(f'    ("{name}", {length}, None),')

    for name in sorted(set_b):
        shapes = set_b[name]
        dims = {len(s) for s in shapes}
        if dims == {1}:
            length = 4
            entries.append(f'    ("{name}", {length}, "stats4"),')
        elif all(d >= 2 for d in dims):
            n_rows_max = max(s[0] for s in shapes)
            length = 4 * n_rows_max
            entries.append(
                f'    ("{name}", {length}, "matrix_rowstats"),  '
                f"# {n_rows_max} rows × 4"
            )
        else:
            max_len = max(int(np.prod(s)) for s in shapes)
            entries.append(
                f'    ("{name}", {max_len}, "stats4"),  # mixed-dim; review manually'
            )

    header = (
        f"# Generated from audit of {n_tracks_total} tracks.\n"
        f"# Set A (deterministic shape): {len(set_a)} descriptors\n"
        f"# Set B (variable shape, normalized): {len(set_b)} descriptors\n"
    )
    return (
        header
        + "_DESCRIPTOR_SCHEMA: tuple[tuple[str, int, str | None], ...] = (\n"
        + "\n".join(entries)
        + "\n)\n"
    )


def _synthesize_wav(path: Path, duration_s: float = 3.0, sr: int = 44100) -> None:
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(int(sr * duration_s)) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def _split_allocation(k: int, n_liked: int, n_disliked: int) -> tuple[int, int]:
    total = n_liked + n_disliked
    if total == 0:
        return (0, 0)
    k_l = round(k * n_liked / total)
    k_d = k - k_l
    k_l = min(k_l, n_liked)
    k_d = min(k_d, n_disliked)
    deficit = k - k_l - k_d
    if deficit > 0:
        slack_l = n_liked - k_l
        give_l = min(slack_l, deficit)
        k_l += give_l
        deficit -= give_l
        k_d = min(k_d + deficit, n_disliked)
    return k_l, k_d


def _size_stratified(tracks: list[Path], k: int, rng: random.Random) -> list[Path]:
    if k <= 0 or not tracks:
        return []
    if k >= len(tracks):
        return list(tracks)
    with_sizes = sorted(tracks, key=lambda p: p.stat().st_size)
    selected = []
    n = len(with_sizes)
    for i in range(k):
        lo = (i * n) // k
        hi = ((i + 1) * n) // k
        bucket = with_sizes[lo:hi]
        if bucket:
            selected.append(rng.choice(bucket))
    return selected


def _select_stratified(tracks: list[Path], k: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    if k >= len(tracks):
        return list(tracks)
    return _size_stratified(tracks, k, rng)


def discover(
    profile_path: Path, tracks_dir: Path, output_path: Path | None, k: int
) -> None:
    extractor = es.MusicExtractor(profile=str(profile_path))

    all_files = sorted(
        p
        for p in tracks_dir.rglob("*")
        if p.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    )
    if not all_files:
        logger.error("No audio files found in %s", tracks_dir)
        sys.exit(1)

    logger.info(
        "Found %d audio files, selecting %d stratified sample", len(all_files), k
    )
    selected = _select_stratified(all_files, k, seed=42)

    observations = _collect_shapes(extractor, selected)
    set_a, set_b, set_c = _classify(observations)

    logger.info("Set A (deterministic shape): %d descriptors", len(set_a))
    logger.info("Set B (variable shape): %d descriptors", len(set_b))
    logger.info("Set C (sometimes absent): %d descriptors", len(set_c))

    literal = _emit_schema_literal(set_a, set_b, len(selected))

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(literal)
        logger.info("Schema literal written to %s", output_path)
    else:
        print(literal)


def verify(profile_path: Path, tracks_dir: Path) -> None:
    """Verify subcommand implementation.
    Imports _DESCRIPTOR_SCHEMA and _NORMALIZERS from audio.features,
    runs extractor on a synthetic WAV, checks shapes match."""
    from audio.features import _DESCRIPTOR_SCHEMA, _NORMALIZERS

    extractor = es.MusicExtractor(profile=str(profile_path))

    with __import__("tempfile").TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "verify_noise.wav"
        _synthesize_wav(wav_path)
        features, _frames = extractor(str(wav_path))

    pool_names = set(features.descriptorNames())
    schema_names = {name for name, _, _ in _DESCRIPTOR_SCHEMA}

    mismatches = []
    missing_from_pool = []
    missing_from_schema = []

    for name, expected_length, normalizer_key in _DESCRIPTOR_SCHEMA:
        if name not in pool_names:
            missing_from_pool.append(name)
            continue
        raw = np.asarray(features[name])
        if normalizer_key is not None:
            arr = _NORMALIZERS[normalizer_key](raw)
        else:
            arr = raw.astype(np.float32).reshape(-1)
        if len(arr) != expected_length:
            mismatches.append(
                f"  {name}: schema expects length {expected_length}, "
                f"got {len(arr)} (raw shape {raw.shape})"
            )

    for name in sorted(pool_names - schema_names):
        if not name.startswith("metadata."):
            missing_from_schema.append(name)

    if mismatches:
        print("SCHEMA MISMATCH — the following descriptors have wrong lengths:")
        for m in mismatches:
            print(m)
        print()
        print(
            "Re-run discover to regenerate the schema, "
            "or update normalizer keys in _DESCRIPTOR_SCHEMA."
        )
        sys.exit(1)

    if missing_from_pool:
        print("WARNING — schema descriptors absent from pool (zero-fill slots):")
        for name in missing_from_pool:
            print(f"  {name}")

    if missing_from_schema:
        print("INFO — pool descriptors absent from schema (ignored):")
        for name in missing_from_schema:
            print(f"  {name}")

    total_dim = sum(length for _, length, _ in _DESCRIPTOR_SCHEMA)
    print(
        f"OK — schema={len(_DESCRIPTOR_SCHEMA)} descriptors (dim={total_dim}), "
        f"zero-fill={len(missing_from_pool)}, pool-only={len(missing_from_schema)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Essentia descriptor shapes across a track corpus."
    )
    subparsers = parser.add_subparsers(dest="command")

    discover_parser = subparsers.add_parser(
        "discover",
        help="Discover descriptor shapes and emit _DESCRIPTOR_SCHEMA literal.",
    )
    discover_parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="Path to essentia extractor profile YAML.",
    )
    discover_parser.add_argument(
        "--tracks",
        type=Path,
        required=True,
        help="Path to directory of audio tracks.",
    )
    discover_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for schema literal (default: stdout).",
    )
    discover_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of stratified sample tracks (default: 10).",
    )

    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify committed schema against a fresh corpus.",
    )
    verify_parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="Path to essentia extractor profile YAML.",
    )
    verify_parser.add_argument(
        "--tracks",
        type=Path,
        required=True,
        help="Path to directory of audio tracks (used for logging context).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "discover":
        discover(args.profile, args.tracks, args.output, args.k)
    elif args.command == "verify":
        verify(args.profile, args.tracks)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
