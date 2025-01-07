import pathlib

import numpy
import polars as pl
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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
    genre_sample = meaningful_genres_data.sample(fraction=0.1, shuffle=True)
    print(f"{genre_sample=}")
    genre_dataset_sample = genre_dataset.filter(
        pl.col(GENRE_COLUMN_NAME).is_in(genre_sample)
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
    # model = xgb.XGBClassifier(
    #     verbosity=3,
    #     eval_metric="mlogloss",
    #     tree_method="hist",
    #     # early_stopping_rounds=3,
    #     objective="multi:softprob",
    #     # n_estimators=5000,
    #     # max_depth = 30,
    # )
    model = OneVsRestClassifier(LinearSVC(random_state=42), verbose=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean Cross-Validation Score: {numpy.mean(cv_scores):.2f}")
    print("Fitting")
    model.fit(X_train, y_train)

    y_predicted = model.predict(X_test)
    print(f"Average accuracy: {accuracy_score(y_test, y_predicted):.2f}")
    print("Detailed report:")
    report = classification_report(
        y_test,
        y_predicted,
        target_names=mapping.get_column(GENRE_COLUMN_NAME).to_list(),
        output_dict=True,
        zero_division=0.0,
    )

    not_learned_genres = []
    for genre, data in sorted(
        filter(
            lambda t: isinstance(t[1], dict)
            and t[0] not in ["macro avg", "weighted avg"],
            report.items(),
        ),
        key=lambda t: t[1]["precision"],
        reverse=True,
    ):
        if data["precision"] < 0.5:
            not_learned_genres.append(genre)
            continue
        print(
            f"{genre}: {data["precision"]}, supported by {int (data["support"])} examples"
        )
    print(
        f"Not learned for {len(not_learned_genres)}/{genre_sample.shape[0]} genres: {not_learned_genres}"
    )

    result = permutation_importance(
        model,
        X_test.to_pandas(),
        y_test.to_pandas(),
        # scoring="neg_log_loss",
        scoring="top_k_accuracy",
        n_repeats=10,
        random_state=42,
    )

    importance_by_feature = sorted(
        pl.DataFrame(
            [list(result.importances_mean)], orient="row", schema=X_test.schema
        )
        .rows(named=True)[0]
        .items(),
        key=lambda row: row[1],
        reverse=True,
    )
    print(importance_by_feature)


if __name__ == "__main__":
    main(
        genre_dataset_path=pathlib.Path("csv/songs.csv.prepared"),
        audio_features_dataset_path=pathlib.Path("csv/audio_features_dataset.csv"),
    )
