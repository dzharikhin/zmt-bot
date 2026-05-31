import pathlib
import subprocess
import tempfile

import essentia
import essentia.standard as es
import librosa
import numpy as np
from panns_inference import AudioTagging

import config
from audio.aggregation import aggregate
from audio.segments import SegmentSpec, get_segments
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


_DESCRIPTOR_NAMES_BY_PROFILE: dict[str, list[str]] = {}


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


def _essentia_pool_to_vector(
    pool, profile_path: pathlib.Path | None = None
) -> np.ndarray:
    if profile_path is None:
        profile_path = config.data_path / "essentia_extractor_profile.yaml"
    profile_key = compute_file_hash(profile_path)

    if profile_key not in _DESCRIPTOR_NAMES_BY_PROFILE:
        _DESCRIPTOR_NAMES_BY_PROFILE[profile_key] = _discover_descriptor_names(pool)

    names = _DESCRIPTOR_NAMES_BY_PROFILE[profile_key]

    parts = []
    for name in names:
        value = pool[name]
        parts.append(np.asarray(value, dtype=np.float32).reshape(-1))

    return np.concatenate(parts)


def extract_essentia_features(
    extractor, audio_path: pathlib.Path, profile_path: pathlib.Path | None = None
) -> np.ndarray:
    features, _frames = extractor(str(audio_path))
    return _essentia_pool_to_vector(features, profile_path)


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


class CombinedExtractor:
    def __init__(
        self,
        panns_weights_path: pathlib.Path,
        profile_path: pathlib.Path | None = None,
    ):
        self.profile_path = profile_path
        self.essentia_extractor = get_essentia_extractor(profile_path)
        self.panns_model = PANNsCNN14(panns_weights_path)

    def __call__(
        self,
        audio_path: pathlib.Path,
        segment_spec: SegmentSpec | None = None,
    ) -> np.ndarray:
        spec = segment_spec or SegmentSpec(
            type="full", window_s=None, k=None, aggregation="mean"
        )
        segments = get_segments(audio_path, spec)

        essentia_vectors = []
        panns_vectors = []

        for start, end in segments:
            if spec.type == "full":
                essentia_vec = extract_essentia_features(
                    self.essentia_extractor, audio_path, self.profile_path
                )
                panns_vec = self.panns_model.extract(audio_path)
            else:
                cropped_path = _ffmpeg_crop_to_tempwav(audio_path, start, end)
                try:
                    essentia_vec = extract_essentia_features(
                        self.essentia_extractor, cropped_path, self.profile_path
                    )
                finally:
                    cropped_path.unlink(missing_ok=True)
                panns_vec = self.panns_model.extract_segment(audio_path, start, end)

            essentia_vectors.append(essentia_vec)
            panns_vectors.append(panns_vec)

        essentia_agg = aggregate(essentia_vectors, spec.aggregation)
        panns_agg = aggregate(panns_vectors, spec.aggregation)
        return np.concatenate([essentia_agg, panns_agg])


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
    return CombinedExtractor(
        panns_weights_path=panns_weights_path,
        profile_path=profile_path,
    )


def extract_features_for_mp3(
    audio_path: pathlib.Path,
    extractor: CombinedExtractor,
    segment_spec: SegmentSpec | None = None,
) -> np.ndarray:
    return extractor(audio_path, segment_spec=segment_spec)
