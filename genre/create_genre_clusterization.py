import pathlib
from typing import cast

import polars as pl
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

GENRE_COLUMN_NAME = "genre_name"
GENRE_ENCODED_COLUMN_NAME = "genre_name_encoded"


def main(
    *,
    genre_dataset_path: pathlib.Path,
    audio_features_dataset_path: pathlib.Path,
    clustered_genres_dataset_path: pathlib.Path,
):
    genre_dataset = pl.scan_csv(genre_dataset_path)
    audio_features_dataset = pl.scan_csv(audio_features_dataset_path)

    meaningful_genres_data = (
        genre_dataset.group_by(pl.col(GENRE_COLUMN_NAME))
        .agg(pl.len().gt(100).alias("examples_cnt"))
        .filter(pl.col("examples_cnt"))
        .select(pl.col(GENRE_COLUMN_NAME))
        .collect()
    )
    genre_dataset = genre_dataset.select(
        pl.col("spotify_id"), pl.col(GENRE_COLUMN_NAME)
    ).filter(pl.col(GENRE_COLUMN_NAME).is_in(meaningful_genres_data))
    data = audio_features_dataset.join(
        genre_dataset,
        how="inner",
        left_on="track_id",
        right_on="spotify_id",
    )
    X = data.select(
        pl.all().exclude(GENRE_COLUMN_NAME, "track_id", "spotify_id")
    ).collect()
    print(f"{X=}")
    y = data.select(pl.col(GENRE_COLUMN_NAME)).collect()
    print(f"{y=}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    cluster_labels = HDBSCAN().fit_predict(X)
    print(f"clusters: {set(list(cluster_labels))}")
    y = cast(pl.DataFrame, y).insert_column(
        1, pl.Series("cluster", cluster_labels)
    )
    y.group_by("cluster").agg(pl.col(GENRE_COLUMN_NAME).unique()).with_columns(
        pl.col(GENRE_COLUMN_NAME).list.join(";")
    ).write_csv(
        clustered_genres_dataset_path
    )

    # # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)


if __name__ == "__main__":
    main(
        genre_dataset_path=pathlib.Path("csv/songs.csv.prepared"),
        audio_features_dataset_path=pathlib.Path("csv/audio_features_dataset.csv"),
        clustered_genres_dataset_path=pathlib.Path("csv/clustered_genres.csv")
    )
