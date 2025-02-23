import csv
import logging
import pathlib
import pickle
import random

import numpy as np
import polars as pl
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from prepare_features_dataset import prepare_audio_features_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ID_COLUMN_NAME = "track_id"
GENRE_COLUMN_NAME = "genre"


def main(
    *,
    working_dir: pathlib.Path,
    snippets_path: pathlib.Path,
    clusters: int,
    min_cluster_size: int = 100,
    sampling_fraction: float = 0.66,
    contamination_fraction: float = 0.3,
):
    audio_features_dataset_path = prepare_audio_features_dataset(
        working_dir, snippets_path
    )

    clustered_tracks = working_dir.joinpath("tracks-clustered.csv")
    if not clustered_tracks.exists():
        logging.info(f"Collecting data")
        audio_features_dataset = (
            pl.scan_csv(audio_features_dataset_path).drop_nans().collect()
        )
        logging.info(f"Data size: {audio_features_dataset.shape}")
        logging.info(f"Clustering data")
        clusterizer = MiniBatchKMeans(n_clusters=clusters)
        clusters = clusterizer.fit_predict(
            audio_features_dataset.select(pl.all().exclude(ID_COLUMN_NAME))
        )
        data = audio_features_dataset.insert_column(
            audio_features_dataset.shape[1], pl.Series(GENRE_COLUMN_NAME, clusters)
        )
        logging.info(f"Saving clustered tracks")
        data.write_csv(clustered_tracks)
        logging.info(f"Saving grouped by cluster")
        data.select(pl.col(ID_COLUMN_NAME), pl.col(GENRE_COLUMN_NAME)).group_by(
            pl.col(GENRE_COLUMN_NAME)
        ).agg(pl.col(ID_COLUMN_NAME)).with_columns(
            pl.col(ID_COLUMN_NAME).list.join(";").alias(ID_COLUMN_NAME)
        ).sort(
            by=pl.col(GENRE_COLUMN_NAME),
        ).write_csv(
            working_dir.joinpath("tracks-grouped-by-cluster.csv")
        )
        del audio_features_dataset
        del clusterizer
        del clusters

    data = pl.read_csv(clustered_tracks)
    logging.info(f"Sampling data")
    data = (
        data.filter(pl.len().over(GENRE_COLUMN_NAME) >= min_cluster_size)
        .group_by(pl.col(GENRE_COLUMN_NAME))
        .agg(pl.all().sample(fraction=sampling_fraction, shuffle=True))
        .explode(pl.all().exclude(GENRE_COLUMN_NAME))
    )
    logging.info(f"Sampled data size: {data.shape}")

    logging.info(f"Filtering outliers")
    outlier_ids = set()
    for genre, dataset in data.group_by(pl.col(GENRE_COLUMN_NAME)):
        print(f"Processing {genre[0]}: {dataset.shape}")
        X = dataset.select(pl.all().exclude(ID_COLUMN_NAME, GENRE_COLUMN_NAME))
        model = IsolationForest(contamination=contamination_fraction)
        dataset = dataset.insert_column(1, pl.Series("filter", model.fit_predict(X)))
        outliers_for_genre = (
            dataset.select(pl.col(ID_COLUMN_NAME), pl.col("filter"))
            .filter(pl.col("filter") == -1)
            .get_column(ID_COLUMN_NAME)
            .to_list()
        )
        outlier_ids.update(outliers_for_genre)

    outliers = pl.Series("outliers", list(outlier_ids)).to_frame()
    del outlier_ids
    logging.info(f"total outliers: {outliers.shape}")
    outliers.write_csv(working_dir.joinpath("tracks-outliers.csv"))
    data = data.filter(
        pl.col(ID_COLUMN_NAME).is_in(outliers.get_column("outliers")).not_()
    )

    logging.info(f"Splitting data into train/test")
    test_track_ids = (
        data.group_by(pl.col(GENRE_COLUMN_NAME))
        .agg(
            pl.col(ID_COLUMN_NAME).sample(
                fraction=0.3, with_replacement=False, shuffle=True
            )
        )
        .explode(pl.col(ID_COLUMN_NAME))
        .select(ID_COLUMN_NAME)
    )

    test_data = data.filter(
        pl.col(ID_COLUMN_NAME).is_in(test_track_ids.get_column(ID_COLUMN_NAME))
    )
    logging.info(f"test data size: {test_data.shape}")
    train_data = data.filter(
        pl.col(ID_COLUMN_NAME).is_in(test_track_ids.get_column(ID_COLUMN_NAME)).not_()
    )
    logging.info(f"train data size: {train_data.shape}")

    X_train = train_data.select(pl.all().exclude(ID_COLUMN_NAME, GENRE_COLUMN_NAME))
    y_train = train_data.select(pl.col(GENRE_COLUMN_NAME))

    del data
    model = OneVsRestClassifier(
        SVC(decision_function_shape="ovo", random_state=random.randint(1, 100))
    )
    model_file_path = working_dir.joinpath(f"genre_model.pickle")
    model_stats_path = model_file_path.parent.joinpath(
        f"{model_file_path.stem}-stat.csv"
    )
    random_seeds_for_model = model.estimator.random_state
    logging.info(f"random state: {random_seeds_for_model}")

    logging.info(f"Fitting model")
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

    logging.info(f"Average accuracy: {accuracy_score(y_test, y_predicted):.2f}")
    logging.info(f"Cross-validating")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    logging.info(f"Cross-Validation scores: {cv_scores}")
    logging.info(
        f"Mean Cross-Validation score: {np.mean(cv_scores):.2f}+-{np.std(cv_scores)}"
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
    with model_file_path.open("wb") as model_file:
        pickle.dump(model, model_file, protocol=5)
    logging.info(f"Dumped model to {model_file_path}")

    with model_stats_path.open("wt") as report_csv:
        csv.writer(report_csv).writerows(report_rows)
    logging.info(f"Dumped model stats to {model_stats_path}")


if __name__ == "__main__":
    main(
        working_dir=pathlib.Path("csv"),
        snippets_path=pathlib.Path("/home/jrx/snippets"),
        clusters=150,
    )
