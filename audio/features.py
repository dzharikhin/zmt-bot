import pathlib
from collections import OrderedDict
from typing import Literal

import librosa
import numpy.typing as npt
from librosa import feature
from mutagen.mp3 import MP3
from soundfile import SoundFile



type FeatureType = npt.NDArray | int | float

def extract_features_for_mp3(
    *,
    mp3_path: pathlib.Path,
    n_mfcc: int = 48,
    dct_type: Literal[1,2,3] = 2,
    n_chroma: int = 12,
    bins_per_octave_multiplier: int = 3,
    n_octaves: int = 7,
    n_bands: int = 7,

) -> dict[str, FeatureType]:
    f = MP3(mp3_path)
    with SoundFile(mp3_path) as wav:
        audio, sr = librosa.load(wav, sr=None)
        stft = librosa.stft(audio)


        return OrderedDict(
            mfcc=librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, dct_type=dct_type),
            chroma_cqt=librosa.feature.chroma_cqt(
                y=audio, sr=sr, n_chroma=n_chroma, bins_per_octave=n_chroma * bins_per_octave_multiplier, n_octaves=n_octaves,
            ),
            chroma_cens=librosa.feature.chroma_cens(
                y=audio, sr=sr, n_chroma=n_chroma, bins_per_octave=n_chroma * bins_per_octave_multiplier, n_octaves=n_octaves,
            ),
            chroma_stft=librosa.feature.chroma_stft(
                y=audio, sr=sr, n_chroma=n_chroma
            ),
            zcr=librosa.feature.zero_crossing_rate(y=audio),
            rmse=librosa.feature.rms(y=audio),
            spectral_centroid=librosa.feature.spectral_centroid(y=audio, sr=sr),
            spectral_bandwidth=librosa.feature.spectral_bandwidth(y=audio, sr=sr),
            spectral_flatness=librosa.feature.spectral_flatness(y=audio),
            spectral_contrast=librosa.feature.spectral_contrast(
                y=audio, sr=sr, n_bands=n_bands - 1
            ),
            spectral_rolloff=librosa.feature.spectral_rolloff(y=audio, sr=sr),
            tonnetz=librosa.feature.tonnetz(y=audio, sr=sr),  # size always eq(6)
            tempo=librosa.feature.tempo(y=audio, sr=sr),
            **dict(zip(("harmonic", "percussive"), librosa.decompose.hpss(stft))),
            duration=librosa.get_duration(S=stft, sr=sr),
            bitrate=f.info.bitrate // 1000,
        )


if __name__ == "__main__":
    track = pathlib.Path("../data/118517468/liked/CQADAgAD0AADrZa5SR_FU76wvZUrAg.mp3")

    row = extract_features_for_mp3(
        mp3_path=track,
    )
    print(f"row size: {len(row)}")
