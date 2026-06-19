import hashlib
import logging
import pathlib
import subprocess
import tempfile
import wave
from typing import Callable

import essentia.standard as es
import librosa
import numpy as np
from panns_inference import AudioTagging

import config
import essentia
from audio.extractor import CombinedExtractor

essentia.EssentiaLogger().warningActive = False

logger = logging.getLogger(__name__)


def _summarize_stats4(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros(4, dtype=np.float32)
    a = arr.astype(np.float32).reshape(-1)
    return np.array([a.mean(), a.std(), a.min(), a.max()], dtype=np.float32)


def _summarize_matrix_rowstats(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    m = np.atleast_2d(arr).astype(np.float32)
    return np.concatenate([m.mean(axis=1), m.std(axis=1), m.min(axis=1), m.max(axis=1)])


_NORMALIZERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "stats4": _summarize_stats4,
    "matrix_rowstats": _summarize_matrix_rowstats,
}

_DESCRIPTOR_SCHEMA: tuple[tuple[str, int, str | None], ...] = (
    # Populated after running the audit script — paste literal here.
)


def schema_fingerprint() -> str:
    return hashlib.sha256(repr(_DESCRIPTOR_SCHEMA).encode()).hexdigest()[:16]


def _synthesize_wav(
    path: pathlib.Path, duration_s: float = 3.0, sr: int = 44100
) -> None:
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(int(sr * duration_s)) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def assert_schema_dim_consistent(profile_path: pathlib.Path | None = None) -> None:
    if not _DESCRIPTOR_SCHEMA:
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = pathlib.Path(tmp_dir) / "dim_check_noise.wav"
        _synthesize_wav(wav_path)
        extractor = get_essentia_extractor(profile_path)
        features, _frames = extractor(str(wav_path))
    pool_names = set(features.descriptorNames())
    mismatches = []
    for name, expected_length, normalizer_key in _DESCRIPTOR_SCHEMA:
        if name not in pool_names:
            continue
        raw = np.asarray(features[name])
        if normalizer_key is not None:
            arr = _NORMALIZERS[normalizer_key](raw)
        else:
            arr = raw.astype(np.float32).reshape(-1)
        if len(arr) != expected_length:
            mismatches.append(
                f"  {name}: schema declares length {expected_length}, "
                f"got {len(arr)} (raw shape {raw.shape})"
            )
    if mismatches:
        bullet_list = "\n".join(mismatches)
        raise RuntimeError(
            f"Schema dimension mismatch:\n{bullet_list}\n"
            f"Update _DESCRIPTOR_SCHEMA or re-run: "
            f"poetry run python -m audit.descriptor_shapes discover ..."
        )


def decode_audio(audio_path: pathlib.Path, sample_rate: int = 16000) -> bytes:
    cmd = [
        "ffmpeg",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout[44:]


def get_essentia_extractor(profile_path: pathlib.Path | None = None):
    if profile_path is None:
        profile_path = config.data_path / "essentia_extractor_profile.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Essentia profile not found: {profile_path}")
    return es.MusicExtractor(profile=str(profile_path))


def _essentia_pool_to_vector(pool) -> np.ndarray:
    pool_names = set(pool.descriptorNames())
    parts = []
    for name, expected_length, normalizer_key in _DESCRIPTOR_SCHEMA:
        if name not in pool_names:
            parts.append(np.zeros(expected_length, dtype=np.float32))
            continue
        raw = np.asarray(pool[name])
        if normalizer_key is not None:
            arr = _NORMALIZERS[normalizer_key](raw)
        else:
            arr = raw.astype(np.float32).reshape(-1)
        if len(arr) < expected_length:
            arr = np.concatenate(
                [arr, np.zeros(expected_length - len(arr), dtype=np.float32)]
            )
        elif len(arr) > expected_length:
            arr = arr[:expected_length]
        parts.append(arr)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def extract_essentia_features(extractor, audio_path) -> np.ndarray:
    features, _frames = extractor(str(audio_path))
    return _essentia_pool_to_vector(features)


def extract_essentia_features_segment(
    extractor,
    audio_path,
    start: float,
    end: float,
) -> np.ndarray:
    cropped_path = _ffmpeg_crop_to_tempwav(audio_path, start, end)
    try:
        features, _frames = extractor(str(cropped_path))
        return _essentia_pool_to_vector(features)
    finally:
        cropped_path.unlink(missing_ok=True)


class PANNsCNN14:
    def __init__(self, weights_path: pathlib.Path):
        self.tagger = AudioTagging(
            checkpoint_path=str(weights_path),
            device="cpu",
        )

    def extract(self, audio_path: pathlib.Path) -> np.ndarray:
        waveform, _sr = librosa.load(str(audio_path), sr=32000, mono=True)
        _clipwise_output, embedding = self.tagger.inference(waveform[None, :])
        return embedding.reshape(-1)

    def extract_segment(
        self, audio_path: pathlib.Path, start_s: float, end_s: float
    ) -> np.ndarray:
        waveform, _sr = librosa.load(
            str(audio_path),
            sr=32000,
            mono=True,
            offset=start_s,
            duration=end_s - start_s,
        )
        if len(waveform) == 0:
            return np.zeros(2048, dtype=np.float32)
        _clipwise_output, embedding = self.tagger.inference(waveform[None, :])
        return embedding.reshape(-1)


def _ffmpeg_crop_to_tempwav(
    audio_path: pathlib.Path, start_s: float, end_s: float
) -> pathlib.Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-ss",
        str(start_s),
        "-to",
        str(end_s),
        "-ac",
        "1",
        "-ar",
        "16000",
        tmp.name,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return pathlib.Path(tmp.name)


def prepare_extractor(
    profile_path: pathlib.Path | None = None,
    panns_weights_path: pathlib.Path | None = None,
) -> CombinedExtractor:
    if panns_weights_path is None:
        panns_weights_path = config.panns_weights_path
    essentia_extractor = get_essentia_extractor(profile_path)
    panns_model = PANNsCNN14(panns_weights_path)
    return CombinedExtractor(
        essentia_extractor=essentia_extractor,
        panns_model=panns_model,
        essentia_extract_fn=extract_essentia_features,
        essentia_extract_segment_fn=extract_essentia_features_segment,
    )
