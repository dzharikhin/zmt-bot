import csv
import pathlib
from typing import cast

import polars as pl
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

GENRE_COLUMN_NAME = "genre_name"
ID_COLUMN_NAME = "spotify_id"
TRACK_ID_COLUMN_NAME = "track_id"


def filter_outliers_per_genre(
    *,
    genre_dataset_path: pathlib.Path,
    audio_features_dataset_path: pathlib.Path,
    result_path: pathlib.Path,
):
    genre_dataset = pl.scan_csv(genre_dataset_path)
    audio_features_dataset = pl.scan_csv(audio_features_dataset_path)

    data = audio_features_dataset.join(
        genre_dataset,
        how="inner",
        left_on="track_id",
        right_on="spotify_id",
    )
    outliners = set()
    outliners_store_path = result_path.parent.joinpath(
        f"{result_path.stem}-outliners.csv"
    )
    with outliners_store_path.open(mode="at") as outliners_file:
        outliners_writer = csv.writer(outliners_file)
        outliners_writer.writerow(("genre", "out/total", "ids"))
        for genre, dataset in (
            data.select(
                pl.all().exclude(
                    ID_COLUMN_NAME,
                    "name",
                    "artist",
                    "position",
                )
            )
            .collect()
            .group_by(pl.col(GENRE_COLUMN_NAME))
        ):
            print(f"Processing {genre[0]}: {dataset.shape}")
            X = dataset.select(
                pl.all().exclude(TRACK_ID_COLUMN_NAME, GENRE_COLUMN_NAME)
            )
            y = dataset.select(pl.col(TRACK_ID_COLUMN_NAME))
            model = IsolationForest()
            # model = EllipticEnvelope(contamination=0.1)
            id_with_filter = cast(pl.DataFrame, y).insert_column(
                1, pl.Series("filter", model.fit_predict(X))
            )
            outliners_for_genre = (
                id_with_filter.filter(pl.col("filter") == -1)
                .get_column(TRACK_ID_COLUMN_NAME)
                .to_list()
            )
            outliners.update(outliners_for_genre)
            outliners_writer.writerow(
                (
                    genre[0],
                    f"{len(outliners_for_genre)}/{dataset.shape[0]}",
                    ";".join(outliners_for_genre),
                )
            )
    print("Saving filtered genre dataset")
    genre_dataset.filter(
        pl.col(ID_COLUMN_NAME).is_in(list(outliners)).not_()
    ).with_columns(pl.col(ID_COLUMN_NAME).alias(TRACK_ID_COLUMN_NAME)).select(
        TRACK_ID_COLUMN_NAME, GENRE_COLUMN_NAME
    ).sink_csv(
        result_path
    )


if __name__ == "__main__":
    filter_outliers_per_genre(
        genre_dataset_path=pathlib.Path("csv/songs-downloaded.csv"),
        audio_features_dataset_path=pathlib.Path("csv/audio_features_dataset.csv"),
        result_path=pathlib.Path("csv/songs-genre_filtered.csv"),
    )
