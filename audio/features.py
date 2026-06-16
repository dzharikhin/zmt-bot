import pathlib
import subprocess
import tempfile

import librosa
import numpy as np
from panns_inference import AudioTagging

import config
import essentia
import essentia.standard as es
from audio.extractor import CombinedExtractor
from core.paths import compute_file_hash

essentia.EssentiaLogger().warningActive = False


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


_DESCRIPTOR_NAMES_BY_PROFILE: dict[str, list[tuple[str, int]]] = {}


def _discover_descriptor_names(pool) -> list[tuple[str, int]]:
    descriptor_info = []
    for name in sorted(pool.descriptorNames()):
        if name.startswith("metadata."):
            continue
        value = pool[name]
        if isinstance(value, str):
            continue
        arr = np.atleast_1d(np.asarray(value))
        if arr.ndim >= 2:
            continue
        if arr.ndim == 1 and len(arr) > 40:
            continue
        descriptor_info.append((name, len(arr)))
    return descriptor_info


def _essentia_pool_to_vector(
    pool, profile_path: pathlib.Path | None = None
) -> np.ndarray:
    if profile_path is None:
        profile_path = config.data_path / "essentia_extractor_profile.yaml"
    profile_key = compute_file_hash(profile_path)

    if profile_key not in _DESCRIPTOR_NAMES_BY_PROFILE:
        _DESCRIPTOR_NAMES_BY_PROFILE[profile_key] = _discover_descriptor_names(pool)

    descriptor_info = _DESCRIPTOR_NAMES_BY_PROFILE[profile_key]

    parts = []
    for name, expected_length in descriptor_info:
        if name not in pool.descriptorNames():
            parts.append(np.zeros(expected_length, dtype=np.float32))
            continue
        value = pool[name]
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if len(arr) < expected_length:
            pad = np.zeros(expected_length - len(arr), dtype=np.float32)
            arr = np.concatenate([arr, pad])
        elif len(arr) > expected_length:
            arr = arr[:expected_length]
        parts.append(arr)

    return np.concatenate(parts)


def extract_essentia_features(
    extractor, audio_path: pathlib.Path, profile_path: pathlib.Path | None = None
) -> np.ndarray:
    features, _frames = extractor(str(audio_path))
    return _essentia_pool_to_vector(features, profile_path)


def extract_essentia_features_segment(
    extractor,
    audio_path: pathlib.Path,
    profile_path: pathlib.Path | None,
    start: float,
    end: float,
) -> np.ndarray:
    cropped_path = _ffmpeg_crop_to_tempwav(audio_path, start, end)
    try:
        return _essentia_pool_to_vector(extractor(str(cropped_path)), profile_path)
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
        profile_path=profile_path,
    )
