import math
import pathlib
import types
from collections import OrderedDict
from functools import reduce
from typing import TypeAlias, Callable, cast, Iterable

import librosa
import numpy as np
import polars as pl
from librosa import feature
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from soundfile import SoundFile

FRAMES_NUMBER = 48  # 10 seconds frame for 8 minutes track
MFCCS_NUMBER = 48
CHROMA_NUMBER = 12
SPECTRAL_CONTRAST_NUMBER = 7
TONNETZ_NUMBER = 6
AGGREGATES = OrderedDict(
    std=lambda col: col.std(),
    skew=lambda col: col.skew(),
    kurtosis=lambda col: col.kurtosis(),
)


def feature_size(**feature_columns: int) -> int:
    feature_columns = list(feature_columns.values())[0]
    return FRAMES_NUMBER * feature_columns + feature_columns * len(AGGREGATES)


def build_schema_for_feature(
    feature_prefix: str, feature_values: int
) -> Iterable[tuple[str, type]]:
    return (
        *(
            (f"{feature_prefix}_{frame + 1}_{column + 1}", float)
            for frame in range(FRAMES_NUMBER)
            for column in range(feature_values)
        ),
        *(
            (f"{feature_prefix}_{aggregation}_{column + 1}", float)
            for aggregation in AGGREGATES
            for column in range(feature_values)
        ),
    )


AUDIO_FEATURE_TYPE_SCHEMA = [
    ("track_id", str),
    *build_schema_for_feature("mfcc", MFCCS_NUMBER),
    *build_schema_for_feature("chroma_cqt", CHROMA_NUMBER),
    *build_schema_for_feature("chroma_cens", CHROMA_NUMBER),
    *build_schema_for_feature("chroma_stft", CHROMA_NUMBER),
    *build_schema_for_feature("zcr", 1),
    *build_schema_for_feature("rmse", 1),
    *build_schema_for_feature("spectral_centroid", 1),
    *build_schema_for_feature("spectral_bandwidth", 1),
    *build_schema_for_feature("spectral_flatness", 1),
    *build_schema_for_feature("spectral_contrast", SPECTRAL_CONTRAST_NUMBER),
    *build_schema_for_feature("spectral_rolloff", 1),
    *build_schema_for_feature("tonnetz", TONNETZ_NUMBER),
    ("tempo", float),
]

FeatureDataType = types.GenericAlias(
    tuple, (float,) * (len(AUDIO_FEATURE_TYPE_SCHEMA) - 1)  # track_id
)
AudioFeaturesType: TypeAlias = tuple[str, *FeatureDataType]


def extract_features_for_mp3(
    *,
    track_id: str,
    mp3_path: pathlib.Path,
) -> AudioFeaturesType:
    with SoundFile(mp3_path) as wav:
        audio, sr = librosa.load(wav, sr=None)
        stft = librosa.stft(audio)
        frames = stft.shape[1]

        connectivity_matrix = csr_matrix((frames, frames), dtype=np.int8)
        connectivity_matrix.setdiag(values=[1] * frames, k=-1)
        connectivity_matrix.setdiag(values=[1] * frames, k=1)

        def build_optimal_frame_clusterization(
            feature_data: np.ndarray,
            guiding_values_fraction: float = 0.1,
        ) -> pl.DataFrame:
            feature_column_builder = lambda i: f"metric_{i}"
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
                .head(math.ceil(feature_raw.shape[1] * guiding_values_fraction))
                .get_column("index")
                .sort()
                .to_list()
            )

            def cluster_values(raw_values: pl.Series) -> pl.Series:
                raw_values = raw_values.to_numpy().reshape(-1, 1)
                agglomerator = AgglomerativeClustering(
                    n_clusters=FRAMES_NUMBER, connectivity=connectivity_matrix
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
            return clusterization_variants.select(
                pl.col(f"clusterization_{target_cluster_variant_index}")
            )

        def wrap_in_df(feature_data: np.ndarray, feature_prefix: str) -> pl.DataFrame:
            feature_column_builder = lambda i: f"{feature_prefix}_{i}"
            df = pl.DataFrame(
                data=(
                    feature_data
                    if feature_data.shape[0] > 1
                    else feature_data.reshape((-1, 1))
                ),
                schema=[
                    feature_column_builder(i) for i in range(feature_data.shape[0])
                ],
                orient="col" if feature_data.shape[0] > 1 else "row",
            )
            if df.shape[0] != feature_data.shape[1]:
                raise Exception(
                    f"shape of feature {feature_prefix} dataframe: {df.shape} does not match feature data shape {feature_data.shape}"
                )
            return df

        def clustered_frames_mean(
            feature_df: pl.DataFrame,
            clusterization: pl.DataFrame,
        ) -> pl.DataFrame:
            return (
                pl.concat(
                    [
                        feature_df,
                        clusterization,
                    ],
                    how="horizontal",
                )
                .group_by(by=pl.col(clusterization.columns[0]))
                .agg(pl.all().mean())
                .drop("by", clusterization.columns[0])
                .unstack(1)
            )

        def aggregate_feature(
            feature_df: pl.DataFrame,
            aggregation_postfix: str,
            column_aggregation: Callable[[pl.Expr], pl.Expr],
        ) -> pl.DataFrame:
            return feature_df.rename(
                {
                    old_column: f"{old_column}_{aggregation_postfix}"
                    for old_column in feature_df.columns
                }
            ).select(
                *(column_aggregation(pl.nth(i)) for i in range(feature_df.shape[1]))
            )

        features_data = OrderedDict(
            mfcc=librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCCS_NUMBER),
            chroma_cqt=librosa.feature.chroma_cqt(
                y=audio, sr=sr, n_chroma=CHROMA_NUMBER
            ),
            chroma_cens=librosa.feature.chroma_cens(
                y=audio, sr=sr, n_chroma=CHROMA_NUMBER
            ),
            chroma_stft=librosa.feature.chroma_stft(
                y=audio, sr=sr, n_chroma=CHROMA_NUMBER
            ),
            zcr=librosa.feature.zero_crossing_rate(y=audio),
            rmse=librosa.feature.rms(y=audio),
            spectral_centroid=librosa.feature.spectral_centroid(y=audio, sr=sr),
            spectral_bandwidth=librosa.feature.spectral_bandwidth(y=audio, sr=sr),
            spectral_flatness=librosa.feature.spectral_flatness(y=audio),
            spectral_contrast=librosa.feature.spectral_contrast(
                y=audio, sr=sr, n_bands=SPECTRAL_CONTRAST_NUMBER - 1
            ),
            spectral_rolloff=librosa.feature.spectral_rolloff(y=audio, sr=sr),
            tonnetz=librosa.feature.tonnetz(y=audio, sr=sr),  # size always eq(6)
        )
        try:
            all_features = np.concatenate(list(features_data.values()))
        except ValueError as e:
            min_size, max_size = reduce(
                lambda x, y: (min(x[0], y), max(x[1], y)),
                (data.shape[1] for data in features_data.values()),
                (float("inf"), float("-inf")),
            )
            diff_fraction = (max_size - min_size) / min_size
            if diff_fraction > 0.2:
                raise Exception(
                    f"feature data size varies from {min_size} to {max_size} - "
                    f"diff is {diff_fraction:.0%} of min value - can not trim, raising"
                ) from e
            features_data = {
                name: data[:, :min_size] for name, data in features_data.items()
            }
            all_features = np.concatenate(list(features_data.values()))

        optimal_frame_clusterization = build_optimal_frame_clusterization(all_features)

        def compress_feature(
            feature_data: np.ndarray, feature_prefix: str
        ) -> pl.DataFrame:
            feature_df = wrap_in_df(feature_data, feature_prefix)
            clustered_frames_mean_feature = clustered_frames_mean(
                feature_df, optimal_frame_clusterization
            )
            return pl.concat(
                [clustered_frames_mean_feature]
                + [
                    aggregate_feature(feature_df, aggregation_name, aggregation_action)
                    for aggregation_name, aggregation_action in AGGREGATES.items()
                ],
                how="horizontal",
            )

        compressed_data = OrderedDict(
            (k, compress_feature(v, k)) for k, v in features_data.items()
        )
        tempo = pl.Series("tempo", librosa.feature.tempo(y=audio, sr=sr)).to_frame()

    feature_row = pl.concat(
        [v for v in compressed_data.values()] + [tempo], how="horizontal"
    )
    return cast(AudioFeaturesType, (track_id, *tuple(feature_row.rows()[0])))


if __name__ == "__main__":
    track = pathlib.Path("test.mp3")

    row = extract_features_for_mp3(
        track_id=track.stem,
        mp3_path=track,
    )
    print(f"schema size: {len(AUDIO_FEATURE_TYPE_SCHEMA)}")
    print(f"row size: {len(row)}")
