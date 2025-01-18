import pathlib
from math import ceil
from typing import TypeAlias, cast

import librosa
import numpy as np
import polars as pl
from librosa import feature
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
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

# features[name, 'std'] = np.std(values, axis=1)
# features[name, 'skew'] = stats.skew(values, axis=1)
# features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
# features[name, 'median'] = np.median(values, axis=1)
# features[name, 'min'] = np.min(values, axis=1)
# features[name, 'max'] = np.max(values, axis=1)


def extract_features_for_mp3(
    *,
    track_id: str,
    mp3_path: pathlib.Path,
    frames_number: int,
) -> AudioFeaturesType:
    with SoundFile(mp3_path) as wav:
        audio, sr = librosa.load(wav, sr=None)
        frames = librosa.stft(audio).shape[1]

        connectivity_matrix = csr_matrix((frames, frames), dtype=np.int8)
        connectivity_matrix.setdiag(values=[1] * frames, k=-1)
        connectivity_matrix.setdiag(values=[1] * frames, k=1)

        def clusterize_framed_feature(
            feature_data: np.ndarray,
            feature_prefix: str,
            guiding_values_fraction: float = 0.37,
        ) -> pl.DataFrame:
            feature_column_builder = lambda i: f"{feature_prefix}_{i}"
            feature_raw = pl.DataFrame(
                data=feature_data,
                schema=[
                    feature_column_builder(i) for i in range(feature_data.shape[0])
                ],
                orient="col",
            )
            max_std_data_column_indexes = (
                feature_raw.std()
                .transpose(column_names=["std"])
                .with_row_index()
                .sort(by=pl.col("std"), descending=True)
                .head(ceil(feature_raw.shape[1] * guiding_values_fraction))
                .get_column("index")
                .sort()
                .to_list()
            )

            def cluster_values(raw_values: pl.Series) -> pl.Series:
                raw_values = raw_values.to_numpy().reshape(-1, 1)
                agglomerator = AgglomerativeClustering(
                    n_clusters=frames_number, connectivity=connectivity_matrix
                )
                cluster_list = agglomerator.fit_predict(raw_values)
                return pl.Series(cluster_list)

            clusterization_variants = feature_raw.select(
                [
                    pl.col(feature_column_builder(index))
                    .map_batches(cluster_values, return_dtype=pl.Int64)
                    .alias(f"clusterization_{index}")
                    for index in max_std_data_column_indexes
                ]
            )
            clusters = {
                i: (
                    pl.concat(
                        [
                            feature_raw,
                            clusterization_variants.select(
                                pl.col(f"clusterization_{i}")
                            ),
                        ],
                        how="horizontal",
                    )
                    .group_by(by=pl.col(f"clusterization_{i}"))
                    .agg(pl.all().mean())
                    .drop("by", f"clusterization_{i}")
                )
                for i in max_std_data_column_indexes
            }

            def score(index: int) -> float:
                return (
                    clusters[index]
                    .std()
                    .transpose(column_names=["std_of_clusters_means"])
                    .sum()
                    .get_column("std_of_clusters_means")
                    .to_list()[0]
                )

            target_cluster_variant_index = list(
                sorted(
                    [(i, score(i)) for i in max_std_data_column_indexes],
                    key=lambda t: t[1],
                    reverse=True,
                )
            )[0][0]
            feature_aggregate = clusters[target_cluster_variant_index]
            return feature_aggregate.unstack(1)

        mfcc = clusterize_framed_feature(
            feature.mfcc(y=audio, sr=sr, n_mfcc=64),
            "mfcc",
        )

        chroma = feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        spectral_contrast = feature.spectral_contrast(y=audio, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)

        zcr = feature.zero_crossing_rate(y=audio, frame_length=2048, hop_length=512)
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

        # feature_row = np.concatenate(
        #     (
        #         mfccs_mean,
        #         chroma_mean,
        #         spectral_contrast_mean,
        #         zcr_mean,
        #         rmse_mean,
        #         spectral_centroid_mean,
        #         spectral_bandwidth_mean,
        #         spectral_flatness_mean,
        #         tonnetz_mean,
        #         tempo,
        #     )
        # )

        return cast(AudioFeaturesType, (track_id, *tuple(feature_row)))


if __name__ == "__main__":
    track = pathlib.Path("/home/jrx/Downloads/2.mp3")

    row = extract_features_for_mp3(
        track_id=track.stem,
        mp3_path=track,
        frames_number=160,
    )
    print(row)
