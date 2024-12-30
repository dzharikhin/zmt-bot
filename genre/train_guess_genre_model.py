import pathlib

import polars as pl
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GENRE_COLUMN_NAME = "genre_name"
GENRE_ENCODED_COLUMN_NAME = "genre_name_encoded"


def main(
    *, genre_dataset_path: pathlib.Path, audio_features_dataset_path: pathlib.Path
):
    genre_dataset = pl.scan_csv(genre_dataset_path)
    audio_features_dataset = pl.scan_csv(audio_features_dataset_path)

    encoded_genre_data = genre_dataset.with_columns(
        (pl.col(GENRE_COLUMN_NAME).rank("dense") - 1).alias(GENRE_ENCODED_COLUMN_NAME)
    )
    mapping = (
        encoded_genre_data.unique([GENRE_COLUMN_NAME, GENRE_ENCODED_COLUMN_NAME])
        .select(pl.col(GENRE_COLUMN_NAME), pl.col(GENRE_ENCODED_COLUMN_NAME))
        .sort(pl.col(GENRE_ENCODED_COLUMN_NAME))
        .collect()
    )
    print(mapping)
    data = audio_features_dataset.join(
        encoded_genre_data.select(
            pl.col("spotify_id"), pl.col(GENRE_ENCODED_COLUMN_NAME)
        ),
        how="inner",
        left_on="track_id",
        right_on="spotify_id",
    )
    X = data.select(pl.all().exclude(GENRE_COLUMN_NAME, GENRE_ENCODED_COLUMN_NAME, "track_id", "spotify_id")).collect()
    y = data.select(pl.col(GENRE_ENCODED_COLUMN_NAME)).collect()
    model = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=3)
    model = make_pipeline(StandardScaler(), model)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model.fit(X_train, y_train, xgbclassifier__eval_set=[(X_test, y_test)])
    print(model.score(X_test, y_test))


if __name__ == "__main__":
    main(
        genre_dataset_path=pathlib.Path("csv/songs.csv.prepared"),
        audio_features_dataset_path=pathlib.Path("csv/audio_features_dataset.csv"),
    )
