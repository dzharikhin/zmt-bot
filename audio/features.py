import pathlib
import subprocess

import essentia
import essentia.standard as es
import numpy as np
import yaml

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


def _load_descriptor_names() -> list[str]:
    profile_path = config.data_path / "essentia_extractor_profile.yaml"
    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    names = []
    for group in ("lowLevel", "rhythm", "tonal"):
        for desc in profile.get("outputFrames", {}).get(group, []):
            names.append(f"{group}.{desc}")
    return names


def _essentia_pool_to_vector(pool) -> np.ndarray:
    global _DESCRIPTOR_NAMES
    if _DESCRIPTOR_NAMES is None:
        _DESCRIPTOR_NAMES = _load_descriptor_names()

    parts = []
    for name in _DESCRIPTOR_NAMES:
        value = pool[name]
        if isinstance(value, (list, np.ndarray)):
            parts.append(np.asarray(value, dtype=np.float32).ravel())
        else:
            parts.append(np.array([float(value)], dtype=np.float32))

    return np.concatenate(parts)


def extract_essentia_features(extractor, audio_path: pathlib.Path) -> np.ndarray:
    features, _frames = extractor(str(audio_path))
    return _essentia_pool_to_vector(features)


class PANNsCNN14:
    def __init__(self, weights_path: pathlib.Path):
        from panns_inference import AudioTagging

        self.tagger = AudioTagging(
            checkpoint_path=str(weights_path),
            device="cpu",
        )

    def extract(self, audio_path: pathlib.Path) -> np.ndarray:
        import librosa

        waveform, _sr = librosa.load(str(audio_path), sr=32000, mono=True)
        _clipwise_output, embedding = self.tagger.inference(waveform[None, :])
        mean_pool = embedding.mean(axis=1)
        return mean_pool.squeeze(0)


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
        panns_weights_path=pathlib.Path("/app/models/panns_cnn14.pth"),
    )


def extract_features_for_mp3(
    audio_path: pathlib.Path,
    extractor: CombinedExtractor,
) -> np.ndarray:
    return extractor(audio_path)
