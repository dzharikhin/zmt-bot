import dataclasses
import math
import pathlib
from typing import Literal

import librosa
import numpy as np
from librosa import feature
from mutagen.mp3 import MP3
from soundfile import SoundFile


@dataclasses.dataclass
class AudioFeatures:
    mfcc: np.ndarray[tuple[int], np.float64]
    chroma_cqt: np.ndarray[tuple[int], np.float64]
    chroma_cens: np.ndarray[tuple[int], np.float64]
    chroma_stft: np.ndarray[tuple[int], np.float64]
    zcr: np.ndarray[tuple[Literal[1]], np.float64]
    rms: np.ndarray[tuple[Literal[1]], np.float64]
    spectral_centroid: np.ndarray[tuple[Literal[1]], np.float64]
    spectral_bandwidth: np.ndarray[tuple[Literal[1]], np.float64]
    spectral_flatness: np.ndarray[tuple[Literal[1]], np.float64]
    spectral_contrast: np.ndarray[tuple[int], np.float64]
    spectral_rolloff: np.ndarray[tuple[Literal[1]], np.float64]
    tonnetz: np.ndarray[tuple[Literal[6]], np.float64]
    tempo: np.ndarray[tuple[Literal[1]], np.float64]
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
    n_bands: int = 6,
    hop_length: int = 512,
    fft_hop_multiplier: int = 4,
) -> AudioFeatures:
    f = MP3(mp3_path)
    with SoundFile(mp3_path) as wav:
        audio, sr = librosa.load(wav, sr=None)

        return AudioFeatures(
            mfcc=librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=n_mfcc,
                dct_type=dct_type,
                hop_length=hop_length,
                n_fft=hop_length * fft_hop_multiplier,
            ),
            chroma_cqt=(
                cqt := librosa.cqt(
                    y=audio,
                    sr=sr,
                    hop_length=hop_length,
                    n_bins=n_octaves * n_chroma * bins_per_octave_multiplier,
                    bins_per_octave=n_chroma * bins_per_octave_multiplier,
                    tuning=None,
                ),
                abs_cqt := np.abs(cqt),
                chroma_cqt := librosa.feature.chroma_cqt(
                    y=audio,
                    sr=sr,
                    C=abs_cqt,
                    n_chroma=n_chroma,
                    bins_per_octave=n_chroma * bins_per_octave_multiplier,
                    n_octaves=n_octaves,
                    hop_length=hop_length,
                ),
            )[-1],
            chroma_cens=librosa.feature.chroma_cens(
                y=audio,
                sr=sr,
                C=abs_cqt,
                n_chroma=n_chroma,
                bins_per_octave=n_chroma * bins_per_octave_multiplier,
                n_octaves=n_octaves,
                hop_length=hop_length,
            ),
            chroma_stft=(
                stft := librosa.stft(
                    y=audio,
                    hop_length=hop_length,
                    n_fft=hop_length * fft_hop_multiplier,
                ),
                abs_stft := np.abs(stft),
                librosa.feature.chroma_stft(
                    y=audio,
                    sr=sr,
                    S=abs_stft**2,
                    n_chroma=n_chroma,
                    hop_length=hop_length,
                    n_fft=hop_length * fft_hop_multiplier,
                ),
            )[-1],
            zcr=librosa.feature.zero_crossing_rate(
                y=audio,
                hop_length=hop_length,
            ),
            rms=librosa.feature.rms(
                y=audio,
                hop_length=hop_length,
            ),
            spectral_centroid=librosa.feature.spectral_centroid(
                y=audio,
                sr=sr,
                S=abs_stft,
                hop_length=hop_length,
                n_fft=hop_length * fft_hop_multiplier,
            ),
            spectral_bandwidth=librosa.feature.spectral_bandwidth(
                y=audio,
                sr=sr,
                S=abs_stft,
                hop_length=hop_length,
                n_fft=hop_length * fft_hop_multiplier,
            ),
            spectral_flatness=librosa.feature.spectral_flatness(
                y=audio,
                S=abs_stft,
                hop_length=hop_length,
                n_fft=hop_length * fft_hop_multiplier,
            ),
            spectral_contrast=librosa.feature.spectral_contrast(
                y=audio,
                sr=sr,
                n_bands=n_bands,
                fmin=min(math.floor(sr/(2**n_bands)), 200),
                S=abs_stft,
                hop_length=hop_length,
                n_fft=hop_length * fft_hop_multiplier,
            ),
            spectral_rolloff=librosa.feature.spectral_rolloff(
                y=audio,
                sr=sr,
                S=abs_stft,
                hop_length=hop_length,
                n_fft=hop_length * fft_hop_multiplier,
            ),
            tonnetz=librosa.feature.tonnetz(
                y=audio,
                sr=sr,
                chroma=chroma_cqt,
                hop_length=hop_length,
            ),
            tempo=librosa.feature.tempo(
                y=audio,
                sr=sr,
                hop_length=hop_length,
            ),
            duration=librosa.get_duration(
                y=audio,
                sr=sr,
                hop_length=hop_length,
                n_fft=hop_length * fft_hop_multiplier,
            ),
            bitrate=f.info.bitrate // 1000,
        )


if __name__ == "__main__":
    track = pathlib.Path("../data/118517468/liked/CQADAgAD0AADrZa5SR_FU76wvZUrAg.mp3")

    row = extract_features_for_mp3(mp3_path=track, hop_length=512 * 4)

    as_dict = dataclasses.asdict(row)
    print(
        f"row: len={len(as_dict)}, types={ {k: f"{type(v)}{v.shape if hasattr(v, "shape") else ""}[{v.dtype if hasattr(v,"dtype") else ""}])" for k, v in as_dict.items()} }"
    )
