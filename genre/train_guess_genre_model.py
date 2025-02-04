import csv
import pathlib
import pickle
import random

import numpy as np
import polars as pl
from sklearn.inspection import permutation_importance
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsOneClassifier

ID_COLUMN_NAME = "track_id"
GENRE_COLUMN_NAME = "genre_name_aggregated"


def main(
    *,
    working_dir: pathlib.Path,
    snippets_path: pathlib.Path,
    songs_dataset_path: pathlib.Path,
    data_contamination_fraction: float,
    parallel_threads: int | None = None,
    genre_sample_fraction: float = 1.0,
    feature_importance_repetitions: int = 0,
):
    # audio_features_dataset_path = prepare_audio_features_dataset(
    #     working_dir, snippets_path
    # )
    # mapped_genres_dataset_path = working_dir.joinpath("songs-mapped_genres.csv")
    # group_genres(
    #     genre_dataset_path=songs_dataset_path,
    #     grouped_by_genre_path=working_dir.joinpath("songs-grouped_by_genre.csv"),
    #     mapped_genre_dataset_path=mapped_genres_dataset_path,
    # )
    filtered_genre_dataset_path = working_dir.joinpath("songs-genre_filtered.csv")
    # filter_outliers_per_genre(
    #     genre_dataset_path=mapped_genres_dataset_path,
    #     audio_features_dataset_path=audio_features_dataset_path,
    #     result_path=filtered_genre_dataset_path,
    #     contamination_fraction=data_contamination_fraction,
    # )

    audio_features_dataset_path = working_dir.joinpath("audio_features_dataset.csv")
    genre_dataset = pl.scan_csv(filtered_genre_dataset_path)

    meaningful_genres_data = (
        genre_dataset.group_by(pl.col(GENRE_COLUMN_NAME))
        .agg(pl.len().ge(100).alias("examples_cnt"))
        .filter(
            pl.col("examples_cnt")
            .and_(pl.col(GENRE_COLUMN_NAME).str.ends_with("folk").not_())
            .and_(pl.col(GENRE_COLUMN_NAME).str.ends_with("instrumental").not_())
        )
        .select(pl.col(GENRE_COLUMN_NAME))
        .collect()
    )
    genre_sample = meaningful_genres_data
    if genre_sample_fraction < 1.0:
        genre_sample = genre_sample.sample(fraction=genre_sample_fraction, shuffle=True)
    print(f"{genre_sample.shape=}: {list(sorted(genre_sample.get_column(GENRE_COLUMN_NAME).to_list()))=}")
    genre_dataset_sample = genre_dataset.select(
        pl.col(ID_COLUMN_NAME), pl.col(GENRE_COLUMN_NAME)
    ).filter(pl.col(GENRE_COLUMN_NAME).is_in(genre_sample))

    test_track_ids = genre_dataset_sample.group_by(pl.col(GENRE_COLUMN_NAME)).agg(
        pl.col(ID_COLUMN_NAME).sample(fraction=0.3, with_replacement=False, shuffle=True)
    ).explode(pl.col(ID_COLUMN_NAME)).select(ID_COLUMN_NAME).collect()

    normalized_audio_features_dataset = pl.scan_csv(audio_features_dataset_path).drop_nans().select(
        pl.col(ID_COLUMN_NAME),
        (pl.all().exclude(ID_COLUMN_NAME) - pl.all().exclude(ID_COLUMN_NAME).mean()) / pl.all().exclude(ID_COLUMN_NAME).std()
    )
    data = normalized_audio_features_dataset.join(
        genre_dataset_sample.select(pl.col(ID_COLUMN_NAME), pl.col(GENRE_COLUMN_NAME)),
        how="inner",
        left_on=ID_COLUMN_NAME,
        right_on=ID_COLUMN_NAME,
    )
    test_data = data.filter(pl.col(ID_COLUMN_NAME).is_in(test_track_ids))
    train_data = data.filter(pl.col(ID_COLUMN_NAME).is_in(test_track_ids).not_())

    # random_seeds_for_models = {
    #     name: random.randint(1, 100) for name in ("ovo", "rf", "mlp")
    # }
    # print(f"{random_seeds_for_models=}")

    # model = VotingClassifier(
    #     verbose=True,
    #     n_jobs=parallel_threads,
    #     voting="hard",
    #     estimators=[
    #         (
    #             "ovo",
    #             OneVsOneClassifier(
    #                 SGDClassifier(random_state=random_seeds_for_models["ovo"]),
    #                 n_jobs=parallel_threads,
    #             ),
    #         ),
    #         (
    #             "rf",
    #             RandomForestClassifier(
    #                 n_estimators=math.log2(X.shape[0] * X.shape[1]),
    #                 random_state=random_seeds_for_models["rf"],
    #                 n_jobs=parallel_threads,
    #             ),
    #         ),
    #         ("mlp", MLPClassifier(random_state=random_seeds_for_models["mlp"])),
    #     ],
    # )

    # cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=parallel_threads)
    # print(f"Cross-Validation Scores: {cv_scores}")
    # print(f"Mean Cross-Validation Score: {numpy.mean(cv_scores):.2f}")

    # print(f"{data.collect().shape=}={train_data.collect().shape}+{test_data.collect().shape}")
    print(f"Collecting train data")
    training_data = train_data.collect()
    print("Train data loaded. Fitting")
    model = OneVsOneClassifier(
        SGDClassifier(random_state=42),
        n_jobs=parallel_threads,
    )
    for train_chunk in training_data.iter_slices():
        X = train_chunk.select(pl.all().exclude(ID_COLUMN_NAME, GENRE_COLUMN_NAME))
        y = train_chunk.select(GENRE_COLUMN_NAME)
        print(f"{X=}")
        print(f"{np.argwhere(np.isnan(X))=}")
        print(f"{y=}")
        model.partial_fit(X, y, genre_sample)

    print(f"Fitting is done")
    print(f"Collecting test data")
    X_test = test_data.select(pl.all().exclude(ID_COLUMN_NAME, GENRE_COLUMN_NAME)).collect()
    y_test = test_data.select(GENRE_COLUMN_NAME).collect()
    print(f"Collecting model predictions")
    y_predicted = model.predict(X_test)
    print(f"Evaluating predictions test")
    print(
        f"Average accuracy: {accuracy_score(y_test, y_predicted):.2f}"
    )
    report = classification_report(
        y_test,
        y_predicted,
        output_dict=True,
        zero_division=0.0,
    )

    not_learned_genres = 0
    report_rows = [("genre", "precision", "examples in data")]
    for genre, data in sorted(
        filter(
            lambda t: isinstance(t[1], dict)
            and t[0] not in ["macro avg", "weighted avg"],
            report.items(),
        ),
        key=lambda t: t[1]["precision"],
        reverse=True,
    ):
        report_rows.append((genre, data["precision"], int(data["support"])))
        if data["precision"] <= 0.5:
            not_learned_genres += 1

    print(f"Dumping model")
    model_file_path = working_dir.joinpath("genre_model.pickle")
    with model_file_path.open("wb") as model_file:
        pickle.dump(model, model_file, protocol=5)
    print(f"Dumped model to {model_file_path}")

    model_stats_path = model_file_path.parent.joinpath(
        f"{model_file_path.stem}-stat.csv"
    )
    with model_stats_path.open("wt") as report_csv:
        csv.writer(report_csv).writerows(report_rows)
    print(f"Dumped model stats to {model_stats_path}")
    print(f"Not learned for {not_learned_genres}/{genre_sample.shape[0]}")

    if feature_importance_repetitions > 0:
        result = permutation_importance(
            model,
            X_test.to_pandas(),
            y_test.to_pandas(),
            scoring="neg_log_loss",
            n_repeats=feature_importance_repetitions,
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
        working_dir=pathlib.Path("csv"),
        snippets_path=pathlib.Path("/home/jrx/snippets"),
        songs_dataset_path=pathlib.Path("csv/songs-downloaded.csv"),
        data_contamination_fraction=0.3,
        genre_sample_fraction=1.0,
        parallel_threads=None,
    )
