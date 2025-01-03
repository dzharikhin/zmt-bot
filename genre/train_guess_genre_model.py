import pathlib

import numpy
import polars as pl
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

GENRE_COLUMN_NAME = "genre_name"
GENRE_ENCODED_COLUMN_NAME = "genre_name_encoded"


def main(
    *, genre_dataset_path: pathlib.Path, audio_features_dataset_path: pathlib.Path
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
    genre_dataset_sample = genre_dataset
    has_unique_genres = True
    while has_unique_genres:
        print("plus one try to select sample with non unique genre examples")
        genre_dataset_sample = genre_dataset.collect().sample(10000, shuffle=True)
        has_unique_genres = (
            genre_dataset_sample.group_by(pl.col(GENRE_COLUMN_NAME))
            .len(name="len")
            .get_column("len")
            .eq(1)
            .any()
        )

    encoded_genre_data = genre_dataset_sample.lazy().with_columns(
        (pl.col(GENRE_COLUMN_NAME).rank("dense") - 1).alias(GENRE_ENCODED_COLUMN_NAME)
    )

    print(f"{encoded_genre_data=}")
    mapping = (
        encoded_genre_data.unique([GENRE_COLUMN_NAME, GENRE_ENCODED_COLUMN_NAME])
        .select(pl.col(GENRE_COLUMN_NAME), pl.col(GENRE_ENCODED_COLUMN_NAME))
        .sort(pl.col(GENRE_ENCODED_COLUMN_NAME))
        .collect()
    )
    print(f"{mapping=}")
    data = audio_features_dataset.join(
        encoded_genre_data.select(
            pl.col("spotify_id"), pl.col(GENRE_ENCODED_COLUMN_NAME)
        ),
        how="inner",
        left_on="track_id",
        right_on="spotify_id",
    )
    X = data.select(
        pl.all().exclude(
            GENRE_COLUMN_NAME, GENRE_ENCODED_COLUMN_NAME, "track_id", "spotify_id"
        )
    ).collect()
    print(f"{X=}")
    y = data.select(pl.col(GENRE_ENCODED_COLUMN_NAME)).collect()
    print(f"{y=}")
    model = xgb.XGBClassifier(
        # eval_metric="mlogloss",
        # tree_method="hist",
        # early_stopping_rounds=3,
        objective="multi:softprob",
        # n_estimators=500,
        # max_depth = 9,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # Using 5-fold cross-validation
    # print(f'Cross-Validation Scores: {cv_scores}')
    # print(f'Mean Cross-Validation Score: {numpy.mean(cv_scores):.2f}')
    print("Fitting")
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    y_predicted = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_predicted):.2f}")
    print(
        classification_report(
            y_test,
            y_predicted,
            target_names=mapping.get_column(GENRE_COLUMN_NAME).to_list(),
        )
    )


if __name__ == "__main__":
    main(
        genre_dataset_path=pathlib.Path("csv/songs.csv.prepared"),
        audio_features_dataset_path=pathlib.Path("csv/audio_features_dataset.csv"),
    )
