import pathlib
import random

import numpy
import polars as pl
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

ID_COLUMN_NAME = "track_id"
GENRE_COLUMN_NAME = "genre_name_aggregated"


def main(
    *,
    genre_dataset_path: pathlib.Path,
    audio_features_dataset_path: pathlib.Path,
    parallel_threads: int | None = None,
    repetitions: int = 0,
):
    genre_dataset = pl.scan_csv(genre_dataset_path)
    audio_features_dataset = pl.scan_csv(audio_features_dataset_path)

    meaningful_genres_data = (
        genre_dataset.group_by(pl.col(GENRE_COLUMN_NAME))
        .agg(pl.len().gt(100).alias("examples_cnt"))
        .filter(
            pl.col("examples_cnt")
            .and_(pl.col(GENRE_COLUMN_NAME).str.ends_with("folk").not_())
            .and_(pl.col(GENRE_COLUMN_NAME).str.ends_with("instrumental").not_())
        )
        .select(pl.col(GENRE_COLUMN_NAME))
        .collect()
    )
    genre_sample = meaningful_genres_data
    genre_sample = genre_sample.sample(fraction=0.05, shuffle=True)
    print(f"{genre_sample=}")
    genre_dataset_sample = genre_dataset.select(
        pl.col(ID_COLUMN_NAME), pl.col(GENRE_COLUMN_NAME)
    ).filter(pl.col(GENRE_COLUMN_NAME).is_in(genre_sample))

    data = audio_features_dataset.join(
        genre_dataset_sample.select(pl.col(ID_COLUMN_NAME), pl.col(GENRE_COLUMN_NAME)),
        how="inner",
        left_on=ID_COLUMN_NAME,
        right_on=ID_COLUMN_NAME,
    )
    X = data.select(
        pl.all().exclude(
            GENRE_COLUMN_NAME,
            ID_COLUMN_NAME,
        )
    ).collect()
    print(f"{X=}")
    y = data.select(pl.col(GENRE_COLUMN_NAME)).collect()
    print(f"{y=}")
    random_seeds_for_models = {
        name: random.randint(1, 100) for name in ("ovo", "rf", "mlp")
    }
    print(f"{random_seeds_for_models=}")
    model = VotingClassifier(
        verbose=True,
        n_jobs=parallel_threads,
        voting="hard",
        estimators=[
            (
                "ovo",
                OneVsOneClassifier(
                    LinearSVC(random_state=random_seeds_for_models["ovo"]),
                    n_jobs=parallel_threads,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_seeds_for_models["rf"],
                    n_jobs=parallel_threads,
                ),
            ),
            ("mlp", MLPClassifier(random_state=random_seeds_for_models["mlp"])),
        ],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=parallel_threads)
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
        output_dict=True,
        zero_division=0.0,
    )

    not_learned_genres = 0
    for genre, data in sorted(
        filter(
            lambda t: isinstance(t[1], dict)
            and t[0] not in ["macro avg", "weighted avg"],
            report.items(),
        ),
        key=lambda t: t[1]["precision"],
        reverse=True,
    ):
        print(
            f"{genre}: {data["precision"]}, supported by {int (data["support"])} examples"
        )
        if data["precision"] <= 0.5:
            not_learned_genres += 1
    print(f"Not learned for {not_learned_genres}/{genre_sample.shape[0]}")

    if repetitions > 0:
        result = permutation_importance(
            model,
            X_test.to_pandas(),
            y_test.to_pandas(),
            scoring="neg_log_loss",
            n_repeats=repetitions,
            random_state=random.randint(1, 100),
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
        genre_dataset_path=pathlib.Path("csv/songs-genre_filtered.csv"),
        audio_features_dataset_path=pathlib.Path("csv/audio_features_dataset.csv"),
        parallel_threads=10,
    )
