import dataclasses
import pathlib
from typing import Literal

import librosa
from librosa import feature
from mutagen.mp3 import MP3
from soundfile import SoundFile


@dataclasses.dataclass
class AudioFeatures:
    mfcc: list[list[float]]
    chroma_cqt: list[list[float]]
    chroma_cens: list[list[float]]
    chroma_stft: list[list[float]]
    zcr: list[list[float]]
    rmse: list[list[float]]
    spectral_centroid: list[list[float]]
    spectral_bandwidth: list[list[float]]
    spectral_flatness: list[list[float]]
    spectral_contrast: list[list[float]]
    spectral_rolloff: list[list[float]]
    tonnetz: list[list[float]]
    tempo: list[float]
    duration: float
    bitrate: int


def extract_features_for_mp3(
    mp3_path: pathlib.Path,
    *,
    n_mfcc: int = 48,
    dct_type: Literal[1, 2, 3] = 2,
    n_chroma: int = 12,
    bins_per_octave_multiplier: int = 3,
    n_octaves: int = 7,
    n_bands: int = 7,
    use_hpss: bool = False,
) -> AudioFeatures:
    f = MP3(mp3_path)
    with SoundFile(mp3_path) as wav:
        audio, sr = librosa.load(wav, sr=None)

        return AudioFeatures(
            mfcc=librosa.feature.mfcc(
                y=audio, sr=sr, n_mfcc=n_mfcc, dct_type=dct_type
            ).tolist(),
            chroma_cqt=librosa.feature.chroma_cqt(
                y=audio,
                sr=sr,
                n_chroma=n_chroma,
                bins_per_octave=n_chroma * bins_per_octave_multiplier,
                n_octaves=n_octaves,
            ).tolist(),
            chroma_cens=librosa.feature.chroma_cens(
                y=audio,
                sr=sr,
                n_chroma=n_chroma,
                bins_per_octave=n_chroma * bins_per_octave_multiplier,
                n_octaves=n_octaves,
            ).tolist(),
            chroma_stft=librosa.feature.chroma_stft(
                y=audio, sr=sr, n_chroma=n_chroma
            ).tolist(),
            zcr=librosa.feature.zero_crossing_rate(y=audio).tolist(),
            rmse=librosa.feature.rms(y=audio).tolist(),
            spectral_centroid=librosa.feature.spectral_centroid(
                y=audio, sr=sr
            ).tolist(),
            spectral_bandwidth=librosa.feature.spectral_bandwidth(
                y=audio, sr=sr
            ).tolist(),
            spectral_flatness=librosa.feature.spectral_flatness(y=audio).tolist(),
            spectral_contrast=librosa.feature.spectral_contrast(
                y=audio, sr=sr, n_bands=n_bands - 1
            ).tolist(),
            spectral_rolloff=librosa.feature.spectral_rolloff(y=audio, sr=sr).tolist(),
            tonnetz=librosa.feature.tonnetz(
                y=audio, sr=sr
            ).tolist(),  # size always eq(6)
            tempo=librosa.feature.tempo(y=audio, sr=sr).tolist(),
            duration=librosa.get_duration(y=audio, sr=sr),
            bitrate=f.info.bitrate // 1000,
        )


if __name__ == "__main__":
    track = pathlib.Path("../data/118517468/liked/CQADAgAD0AADrZa5SR_FU76wvZUrAg.mp3")

    row = extract_features_for_mp3(
        mp3_path=track,
    )

    as_dict = dataclasses.asdict(row)
    print(
        f"row: len={len(as_dict)}, types={ {k: type(v) for k, v in as_dict.items()} }"
    )
