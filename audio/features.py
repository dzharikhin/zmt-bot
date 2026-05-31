import pathlib
import subprocess

import essentia
import essentia.standard as es
import librosa
import numpy as np
import yaml
from panns_inference import AudioTagging

import config

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


def get_essentia_extractor():
    profile_path = config.data_path / "essentia_extractor_profile.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Essentia profile not found: {profile_path}")
    return es.MusicExtractor(profile=str(profile_path))


_DESCRIPTOR_NAMES: list[str] | None = None


def _discover_descriptor_names(pool) -> list[str]:
    names = []
    for name in sorted(pool.descriptorNames()):
        if name.startswith("metadata."):
            continue
        value = pool[name]
        if isinstance(value, str):
            continue
        arr = np.asarray(value)
        if arr.ndim >= 2:
            continue
        if arr.ndim == 1 and len(arr) > 40:
            continue
        names.append(name)
    return names


def _essentia_pool_to_vector(pool) -> np.ndarray:
    global _DESCRIPTOR_NAMES
    if _DESCRIPTOR_NAMES is None:
        _DESCRIPTOR_NAMES = _discover_descriptor_names(pool)

    parts = []
    for name in _DESCRIPTOR_NAMES:
        value = pool[name]
        parts.append(np.asarray(value, dtype=np.float32).reshape(-1))

    return np.concatenate(parts)


def extract_essentia_features(extractor, audio_path: pathlib.Path) -> np.ndarray:
    features, _frames = extractor(str(audio_path))
    return _essentia_pool_to_vector(features)


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


class CombinedExtractor:
    def __init__(self, panns_weights_path: pathlib.Path):
        self.essentia_extractor = get_essentia_extractor()
        self.panns_model = PANNsCNN14(panns_weights_path)

    def __call__(self, audio_path: pathlib.Path) -> np.ndarray:
        essentia_vector = extract_essentia_features(self.essentia_extractor, audio_path)
        panns_vector = self.panns_model.extract(audio_path)
        return np.concatenate([essentia_vector, panns_vector])


def prepare_extractor() -> CombinedExtractor:
    return CombinedExtractor(
        panns_weights_path=config.panns_weights_path,
    )


def extract_features_for_mp3(
    audio_path: pathlib.Path,
    extractor: CombinedExtractor,
) -> np.ndarray:
    return extractor(audio_path)
