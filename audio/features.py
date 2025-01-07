import pathlib
from typing import TypeAlias, cast

import librosa
import numpy as np
from librosa import feature
from soundfile import SoundFile

AudioFeaturesType: TypeAlias = tuple[
    str,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]

AUDIO_FEATURE_SCHEMA = [
    ("track_id", str),
    *((f"mfcc_mean{i + 1}", float) for i in range(64)),
    *((f"chroma_mean{i + 1}", float) for i in range(12)),
    *((f"spectral_contrast_mean{i + 1}", float) for i in range(7)),
    ("zcr_mean", float),
    ("rmse_mean", float),
    ("spectral_centroid_mean", float),
    ("spectral_bandwidth_mean", float),
    ("spectral_flatness_mean", float),
    *((f"tonnetz_mean{i + 1}", float) for i in range(6)),
    ("tempo", float),
]


def extract_features_for_mp3(
    *, track_id: str, mp3_path: pathlib.Path
) -> AudioFeaturesType:
    with SoundFile(mp3_path) as wav:
        audio, sr = librosa.load(wav, sr=None)

        mfccs = feature.mfcc(y=audio, sr=sr, n_mfcc=64)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        chroma = feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        spectral_contrast = feature.spectral_contrast(y=audio, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)

        zcr = feature.zero_crossing_rate(y=audio)
        zcr_mean = np.mean(zcr.T, axis=0)

        rmse = librosa.feature.rms(y=audio)
        rmse_mean = np.mean(rmse.T, axis=0)

        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid.T, axis=0)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth.T, axis=0)

        spectral_flatness = librosa.feature.spectral_flatness(y=audio)
        spectral_flatness_mean = np.mean(spectral_flatness.T, axis=0)

        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)

        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo = feature.tempo(y=audio, sr=sr, onset_envelope=onset_env)

        feature_row = np.concatenate(
            (
                mfccs_mean,
                chroma_mean,
                spectral_contrast_mean,
                zcr_mean,
                rmse_mean,
                spectral_centroid_mean,
                spectral_bandwidth_mean,
                spectral_flatness_mean,
                tonnetz_mean,
                tempo,
            )
        )

        return cast(AudioFeaturesType, (track_id, *tuple(feature_row)))


if __name__ == "__main__":
    track = pathlib.Path("0VvR2aTYtAsqLTpaBPbfsw.mp3")

    row = extract_features_for_mp3(track_id=track.stem, mp3_path=track)
    print(row)
