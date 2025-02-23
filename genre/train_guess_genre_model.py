import csv
import logging
import pathlib
import pickle
import random
from functools import reduce

import numpy as np
import polars as pl
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from genre.create_genre_mapping import group_genres
from genre.filter_outliers import filter_outliers_per_genre
from genre.prepare_features_dataset import prepare_audio_features_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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
    audio_features_dataset_path = prepare_audio_features_dataset(
        working_dir, snippets_path
    )
    mapped_genres_dataset_path = working_dir.joinpath("songs-mapped_genres.csv")
    group_genres(
        genre_dataset_path=songs_dataset_path,
        grouped_by_genre_path=working_dir.joinpath("songs-grouped_by_genre.csv"),
        mapped_genre_dataset_path=mapped_genres_dataset_path,
    )
    filtered_genre_dataset_path = working_dir.joinpath("songs-genre_filtered.csv")
    filter_outliers_per_genre(
        genre_dataset_path=mapped_genres_dataset_path,
        audio_features_dataset_path=audio_features_dataset_path,
        result_path=filtered_genre_dataset_path,
        contamination_fraction=data_contamination_fraction,
    )

    genre_mapping = {
        "edm": {
            "trance",
            "minimal techno",
            "psytrance",
            "electro",
            "techno",
            "house",
            "progressive",
            "miami bass",
            "electro tropical",
            "baile funk",
            "hardstyle",
            "darkpsy",
            "big room",
            "pop edm",
        },
        "hip hop": {
            "electronic trap",
            "trap",
            "r&b",
            "drill",
        },
        "drum and bass": {
            "breakcore",
            "neurofunk",
            "jungle",
            "jump up",
            "liquid",
        },
        "dubstep": {
            "minimal dubstep",
            "substep",
        },
        "rock": {
            "grunge",
            "metal rock",
            "ska",
            "death metal",
            "post rock",
            "emo rock",
            "punk rock",
            "nu metal",
            "black metal",
            "metalcore",
            "avantgarde metal",
            "crossover thrash",
            "beatdown",
            "glam metal",
            "goregrind",
            "grim death metal",
            "hardcore",
            "post black metal",
            "post punk rock",
            "thrash metal",
        },
        "pop": {},
        "improvizing": {
            "acid jazz",
            "smooth jazz",
            "blues",
            "funk rock",
            "funky breaks",
        },
        "witch house": {},
    }
    interesting_genres = set(genre_mapping.keys())
    interesting_genres.update(reduce(lambda a, b: a.union(b), genre_mapping.values()))
    genre_dataset = pl.scan_csv(filtered_genre_dataset_path)

    not_included_genres = (
        genre_dataset.select(pl.col(GENRE_COLUMN_NAME).unique().sort())
        .select(pl.implode(GENRE_COLUMN_NAME).sort().alias("all_genres"))
        .collect()
        .insert_column(
            1, pl.Series("interesting_genres", [list(sorted(interesting_genres))])
        )
        .with_columns(
            pl.col("all_genres")
            .list.set_difference(pl.col("interesting_genres"))
            .alias("not_included_genres")
        )
        .select(pl.col("not_included_genres"))
        .explode(pl.col("not_included_genres"))
        .get_column("not_included_genres")
        .sort()
        .unique(maintain_order=True)
        .to_list()
    )

    logging.info(f"{not_included_genres=}")
    inverse_mapping = {
        specific_name: common_name
        for common_name, specific_names in genre_mapping.items()
        for specific_name in specific_names
    }

    def mapper(genre_name: str) -> str:
        return inverse_mapping.get(genre_name, genre_name)

    remapped_genre_dataset = genre_dataset.with_columns(
        pl.col(GENRE_COLUMN_NAME)
        .map_elements(mapper, pl.String)
        .alias("clustered_genre")
    ).with_columns(pl.col("clustered_genre").alias(GENRE_COLUMN_NAME))

    genre_sample = pl.Series(GENRE_COLUMN_NAME, list(genre_mapping.keys())).to_frame()
    if genre_sample_fraction < 1.0:
        genre_sample = genre_sample.sample(fraction=genre_sample_fraction, shuffle=True)
    logging.info(
        f"{genre_sample.shape=}: {genre_sample.sort(by=pl.col(GENRE_COLUMN_NAME)).to_series().to_list()=}"
    )
    genre_dataset_sample = remapped_genre_dataset.select(
        pl.col(ID_COLUMN_NAME), pl.col(GENRE_COLUMN_NAME)
    ).filter(
        pl.col(GENRE_COLUMN_NAME).is_in(genre_sample.get_column(GENRE_COLUMN_NAME))
    )

    test_track_ids = (
        genre_dataset_sample.group_by(pl.col(GENRE_COLUMN_NAME))
        .agg(
            pl.col(ID_COLUMN_NAME).sample(
                fraction=0.3, with_replacement=False, shuffle=True
            )
        )
        .explode(pl.col(ID_COLUMN_NAME))
        .select(ID_COLUMN_NAME)
        .collect()
    )

    logging.info(f"Cleaning data")
    audio_features_dataset = (
        pl.scan_csv(audio_features_dataset_path)
        .drop_nans()
    )

    data = audio_features_dataset.join(
        genre_dataset_sample.select(pl.col(ID_COLUMN_NAME), pl.col(GENRE_COLUMN_NAME)),
        how="inner",
        left_on=ID_COLUMN_NAME,
        right_on=ID_COLUMN_NAME,
    )
    logging.info(f"Collecting data")
    data = data.collect()
    logging.info(f"Splitting train-test")
    test_data = data.filter(
        pl.col(ID_COLUMN_NAME).is_in(test_track_ids.get_column(ID_COLUMN_NAME))
    )
    train_data = data.filter(
        pl.col(ID_COLUMN_NAME).is_in(test_track_ids.get_column(ID_COLUMN_NAME)).not_()
    )

    X_train = train_data.select(pl.all().exclude(ID_COLUMN_NAME, GENRE_COLUMN_NAME))
    y_train = train_data.select(pl.col(GENRE_COLUMN_NAME))

    models = {
        "ovo": OneVsOneClassifier(
            LinearSVC(random_state=random.randint(1, 100)), n_jobs=parallel_threads
        ),
        "rf": RandomForestClassifier(
            n_estimators=data.shape[1] // 10,
            random_state=random.randint(1, 100),
            n_jobs=parallel_threads,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(data.shape[1] // 10,),
            random_state=random.randint(1, 100),
        ),
    }
    model = VotingClassifier(
        n_jobs=parallel_threads,
        voting="hard",
        estimators=list(models.items()),
    )

    random_seeds_for_models = {}
    for model_name, model in models.items():
        if "ovo" == model_name:
            random_seeds_for_models[model_name] = model.estimator.random_state
        else:
            random_seeds_for_models[model_name] = model.random_state
    logging.info(f"{random_seeds_for_models=}")

    logging.info(f"Fitting")
    model.fit(X_train, y_train)
    logging.info(f"Fitting is done")
    logging.info(f"Collecting test data")
    X_test = test_data.select(pl.all().exclude(ID_COLUMN_NAME, GENRE_COLUMN_NAME))
    logging.info(f"Collecting model predictions")

    def partial_predict(chunk):
        predicted_part = model.predict(chunk)
        logging.info(f"Predictions for {chunk.shape=}: {predicted_part.shape}")
        return predicted_part

    y_predicted = np.concatenate(
        [partial_predict(data_chunk) for data_chunk in X_test.iter_slices(1000)]
    )
    logging.info(f"Collecting test predictions")
    y_test = test_data.select(GENRE_COLUMN_NAME)
    logging.info("Saving predictions")
    test_data.select(
        pl.col(ID_COLUMN_NAME), pl.col(GENRE_COLUMN_NAME), pl.col("tempo")
    ).insert_column(1, pl.Series("predicted", y_predicted)).write_csv(
        working_dir.joinpath("test_predictions_match.csv")
    )
    logging.info(f"Evaluating predictions test")
    logging.info(f"Average accuracy: {accuracy_score(y_test, y_predicted):.2f}")
    logging.info(f"Cross-validating")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=parallel_threads)
    logging.info(f"Cross-Validation Scores: {cv_scores}")
    logging.info(
        f"Mean Cross-Validation Score: {np.mean(cv_scores):.2f}+-{np.std(cv_scores)}"
    )
    logging.info(f"Building report")
    report = classification_report(
        y_test,
        y_predicted,
        output_dict=True,
        zero_division=0.0,
    )

    not_learned_genres = 0
    report_rows = [("genre", "precision", "examples in data")]
    for genre, stats in sorted(
        filter(
            lambda t: isinstance(t[1], dict)
            and t[0] not in ["macro avg", "weighted avg"],
            report.items(),
        ),
        key=lambda t: t[1]["precision"],
        reverse=True,
    ):
        report_rows.append((genre, stats["precision"], int(stats["support"])))
        if stats["precision"] <= 0.5:
            not_learned_genres += 1

    logging.info(f"Dumping model")
    model_file_path = working_dir.joinpath("genre_model.pickle")
    with model_file_path.open("wb") as model_file:
        pickle.dump(model, model_file, protocol=5)
    logging.info(f"Dumped model to {model_file_path}")

    model_stats_path = model_file_path.parent.joinpath(
        f"{model_file_path.stem}-stat.csv"
    )
    with model_stats_path.open("wt") as report_csv:
        csv.writer(report_csv).writerows(report_rows)
    logging.info(f"Dumped model stats to {model_stats_path}")
    logging.info(f"Not learned for {not_learned_genres}/{genre_sample.shape[0]}")

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
        logging.info(importance_by_feature)


if __name__ == "__main__":
    main(
        working_dir=pathlib.Path("csv"),
        snippets_path=pathlib.Path("/home/jrx/snippets"),
        songs_dataset_path=pathlib.Path("csv/songs-downloaded.csv"),
        data_contamination_fraction=0.3,
        genre_sample_fraction=1.0,
        parallel_threads=None,
    )
