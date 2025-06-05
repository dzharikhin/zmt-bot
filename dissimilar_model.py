import math
from itertools import product

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

import config
from train import LIKED_COLUMN_NAME, ID_COLUMN_NAME, _train_dissimilar_model

if __name__ == "__main__":
    results = []
    for frames in [-96, -48, -24, -12, 0, 12, 24, 48, 96]:
        liked_dataframe = pl.scan_csv(f"data/audio_features_dataset_f{frames}-1.csv")
        disliked_dataframe = pl.scan_csv(f"data/audio_features_dataset_f{frames}-0.csv")
        initial_data = pl.concat([liked_dataframe, disliked_dataframe]).collect(engine="streaming").drop_nulls()
        for scale in [True, False]:
            for contamination in [0, 0.05, 0.1, 0.2, 0.25]:
                contamination_safe = contamination if contamination > 0 else "auto"
                for model_name, model in ([(f"svm({nu=})", OneClassSVM(nu=nu)) for nu in [0.1, 0.2, 0.3, 0.5]]
                                          + [(f"forest({contamination_safe=},{estimators=})", IsolationForest(contamination=contamination_safe, n_estimators=estimators)) for estimators in [100, 50, 10, initial_data.shape[1] // 2, initial_data.shape[1] // 10]]
                                          + [(f"lof({contamination_safe=},{leaf_size=},{n_neighbors=})", LocalOutlierFactor(novelty=True, contamination=contamination_safe, leaf_size=leaf_size, n_neighbors=n_neighbors)) for leaf_size, n_neighbors in product([30, initial_data.shape[0] // 10, initial_data.shape[0] // 2], [20, 10, math.ceil(initial_data.shape[0] * contamination) if contamination > 0 else 5, initial_data.shape[0] // 20])]
                ):
                    print(f"Testing {model_name=}: {frames=},{contamination=}")
                    try_accuracies = []
                    for tries in range(20):
                        data = initial_data.sample(fraction=1, shuffle=True)
                        test_track_ids = (
                            data.group_by(pl.col(LIKED_COLUMN_NAME))
                            .agg(
                                pl.col(ID_COLUMN_NAME).sample(
                                    fraction=config.test_samples_fraction, with_replacement=False, shuffle=True
                                )
                            )
                            .explode(pl.col(ID_COLUMN_NAME))
                            .select(ID_COLUMN_NAME)
                        )

                        test_data = data.filter(
                            pl.col(ID_COLUMN_NAME).is_in(test_track_ids.get_column(ID_COLUMN_NAME).implode())
                        )
                        positive_cases = test_data.filter(pl.col(LIKED_COLUMN_NAME) == 0)
                        negative_cases = test_data.filter(pl.col(LIKED_COLUMN_NAME) == 1).limit(positive_cases.shape[0])
                        test_data = pl.concat([positive_cases, negative_cases])

                        train_data = data.filter(
                            pl.col(ID_COLUMN_NAME).is_in(test_track_ids.get_column(ID_COLUMN_NAME).implode()).not_()
                        )
                        model, accuracy = _train_dissimilar_model(
                            train_data=train_data, test_data=test_data, model=model, contamination_fraction=contamination, scale=scale
                        )
                        try_accuracies.append(accuracy)
                    mean_accuracy = float(np.mean(try_accuracies))
                    results.append((mean_accuracy, float(np.std(try_accuracies)), model_name, frames, contamination))

    print()
    for i, (mean_accuracy, accuracy_variance, model_name, frames, contamination) in enumerate(sorted(results, reverse=True, key=lambda r: r[0])):
        print(f"{i}: Stats for {model_name=} on data [{frames=},{contamination=},{scale=}]: accuracy={mean_accuracy:.3f}(+-{accuracy_variance:.3f})")
