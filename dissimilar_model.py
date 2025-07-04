import dataclasses
import json
import logging
import math
import pathlib
import random
import re
import sys
import typing
from collections import defaultdict

import numpy as np
import polars as pl
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.ecod import ECOD
from pyod.models.knn import KNN
from pyod.models.kpca import KPCA
from pyod.models.lof import LOF
from sklearn.metrics import accuracy_score

from audio.features import extract_features_for_mp3, AudioFeatures
from dataset.persistent_dataset_processor import DataFrameBuilder

if __name__ == "__main__":
    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[handler],
    )
    logger = logging.getLogger(__file__)

    # for tweak in options:
    @dataclasses.dataclass
    class MappedAudioFeatures(AudioFeatures):
        mfcc: list[list[float]]
        chroma_cqt: list[list[float]]
        chroma_cens: list[list[float]]
        chroma_stft: list[list[float]]
        zcr: list[list[float]]
        rms: list[list[float]]
        spectral_centroid: list[list[float]]
        spectral_bandwidth: list[list[float]]
        spectral_flatness: list[list[float]]
        spectral_contrast: list[list[float]]
        spectral_rolloff: list[list[float]]
        tonnetz: list[list[float]]
        tempo: list[float]
        liked: int

    test_fraction = 0.33
    contamination_fraction = 0.25
    train_tries = 1
    data_report = pathlib.Path("data_report.csv")
    checked_variants = set()
    if data_report.exists():
        with data_report.open(mode="rt") as f:
            checked_variants = {
                frozenset(json.loads(m.group(0).replace("'", '"')).items())
                for line in f.readlines()
                if (m := re.match("{.+}", line))
            }

    data_params = dict(
        hop_length=512 * 2**3,
        n_mfcc=24,
        dct_type=1,
        n_chroma=12,
        bins_per_octave_multiplier=3,
        n_octaves=7,
        n_bands=4,
        fft_hop_multiplier=2,
    )

    metric_sizes = {
        field.name: size_literal[0]
        for field in dataclasses.fields(AudioFeatures)
        if (root_args := typing.get_args(field.type))
        and (size_literal := typing.get_args(typing.get_args(root_args[0])[0]))
    }

    metric_sizes |= dict(
        mfcc=data_params["n_mfcc"],
        chroma_cqt=data_params["n_chroma"],
        chroma_cens=data_params["n_chroma"],
        chroma_stft=data_params["n_chroma"],
        spectral_contrast=data_params["n_bands"],
    )
    container_field_names = set(
        field.name
        for field in dataclasses.fields(MappedAudioFeatures)
        if field.type
        in [
            list[list[float]],
        ]
    )
    if diff := (container_field_names - metric_sizes.keys()):
        raise Exception(f"For fields {diff} size is unknown")

    def stats(progress_state: DataFrameBuilder.ProgressStat):
        logger.info(
            f"{data_params}: {progress_state.succeed=}/{progress_state.failed=}/{progress_state.data_size=}"
        )

    liked_audio_path = pathlib.Path("data/118517468/liked")
    disliked_audio_path = pathlib.Path("data/118517468/disliked")

    def common_extractor(
        audio_path: pathlib.Path, is_liked: bool, track_id: str
    ) -> MappedAudioFeatures:
        features = extract_features_for_mp3(
            audio_path.joinpath(track_id),
            **data_params,
        )
        remapped_values = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in dataclasses.asdict(features).items()
        }
        logger.debug(f"[{is_liked=}]extracted features for {track_id=}")
        return MappedAudioFeatures(**remapped_values, liked=int(is_liked))

    def liked_feature_extractor(track_id: str) -> MappedAudioFeatures:
        return common_extractor(liked_audio_path, True, track_id)

    def disliked_feature_extractor(track_id: str) -> MappedAudioFeatures:
        return common_extractor(disliked_audio_path, False, track_id)

    with (
        DataFrameBuilder(
            working_dir=pathlib.Path("data/tmp/disliked_dir"),
            index_generator=DataFrameBuilder.GeneratingParams(
                generator=(
                    f.name for f in disliked_audio_path.iterdir() if f.is_file()
                ),
                result_schema=("row_id", pl.String),
            ),
            mappers=[disliked_feature_extractor],
            batch_size=25,
            cleanup_on_exit=False,
            progress_tracker=stats,
        ) as disliked_frame_result,
        DataFrameBuilder(
            working_dir=pathlib.Path("data/tmp/liked_dir"),
            index_generator=DataFrameBuilder.GeneratingParams(
                generator=(
                    f.name for f in liked_audio_path.iterdir() if f.is_file()
                ),
                result_schema=("row_id", pl.String),
            ),
            mappers=[liked_feature_extractor],
            batch_size=25,
            cleanup_on_exit=False,
            progress_tracker=stats,
        ) as liked_frame_result,
    ):
        logger.info(f"{data_params}: Started processing")

        # with tempfile.TemporaryDirectory(dir="data/tmp") as tmp:
        processed_data = (
            pl.concat([disliked_frame_result.ldf, liked_frame_result.ldf])
            .filter(
                pl.col(disliked_frame_result.processing_status_column[0])
                == DataFrameBuilder.PROCESSING_SUCCEED_VALUE
            )
            .select(
                pl.all().exclude(disliked_frame_result.processing_status_column[0])
            )
        )
        max_possible_columns = math.ceil(
            (
                max_track_duration := processed_data.select(
                    pl.col("duration").max()
                )
                .collect(engine="streaming")
                .item(0, 0)
            )
            * 44100
            / data_params["hop_length"]
        )
        processed_data = processed_data.with_columns(
            pl.col("tempo").list.first()
        )
        logger.debug(f"{max_track_duration=}")
        for field_name in (
            field.name
            for field in dataclasses.fields(MappedAudioFeatures)
            if field.type
            in [
                list[list[float]],
            ]
        ):
            processed_data = (
                processed_data.with_columns(
                    pl.col(field_name).list.to_struct(
                        fields=[
                            f"{field_name}_{idx}"
                            for idx in range(metric_sizes[field_name])
                        ],
                        upper_bound=metric_sizes[field_name],
                    )
                )
                .unnest(field_name)
                .with_columns(
                    *[
                        pl.col((col_name := f"{field_name}_{idx}")).list.to_struct(
                            fields=[
                                f"{col_name}_{idx}"
                                for idx in range(max_possible_columns)
                            ],
                            upper_bound=max_possible_columns,
                        )
                        for idx in range(metric_sizes[field_name])
                    ],
                )
                .unnest(
                    [
                        f"{field_name}_{idx}"
                        for idx in range(metric_sizes[field_name])
                    ]
                )
            )

        logger.info(
            f"{data_params}: {processed_data.collect_schema().len()=}"
        )
            # flat_data_path = pathlib.Path(tmp).joinpath("flat.ipc")
            # processed_data.sink_ipc(
            #     flat_data_path,
            #     maintain_order=False,
            #     engine="streaming",
            #     sync_on_close="all",
            # )

        accuracies = defaultdict(list)
        for try_n in range(train_tries):
            shuffle_seed = random.randint(0, 100)
            test_disliked_track_ids = (
                processed_data.filter(pl.col("liked") == 0)
                .select(pl.col("row_id"))
                .with_columns(pl.col("row_id").shuffle(seed=shuffle_seed))
                .with_row_index()
                .filter(pl.col("index") < pl.col("index").max() * test_fraction)
                .select(pl.all().exclude("index"))
            )
            test_liked_track_ids = (
                processed_data.filter(pl.col("liked") == 1)
                .select(pl.col("row_id"))
                .with_columns(pl.col("row_id").shuffle(seed=shuffle_seed))
                .with_row_index()
                .join(test_disliked_track_ids.with_row_index(), "index", "left")
                .filter(pl.col("row_id_right").is_null().not_())
                .select(pl.all().exclude("index", "row_id_right"))
            )

            test_track_ids = pl.concat(
                [
                    test_disliked_track_ids,
                    test_liked_track_ids,
                ]
            )

            train_disliked_track_ids = (
                processed_data.filter(pl.col("liked") == 0)
                .select(pl.col("row_id"))
                .join(test_disliked_track_ids, "row_id", "left", coalesce=False)
                .filter(pl.col("row_id_right").is_null())
                .select(pl.all().exclude("row_id_right"))
            )
            contamination_count = train_disliked_track_ids.with_row_index().filter(
                pl.col("index") < pl.col("index").max() * contamination_fraction
            )
            train_liked_track_ids = (
                processed_data.filter(pl.col("liked") == 1)
                .select(pl.col("row_id"))
                .join(test_liked_track_ids, "row_id", "left", coalesce=False)
                .filter(pl.col("row_id_right").is_null())
                .select(pl.col("row_id"))
                .with_columns(pl.col("row_id").shuffle(seed=shuffle_seed))
                .with_row_index()
                .join(contamination_count, "index", "left")
                .filter(pl.col("row_id_right").is_null().not_())
                .select(pl.col("row_id"))
            )

            train_track_ids = pl.concat(
                [train_disliked_track_ids, train_liked_track_ids]
            )

            if (
                not (
                    train_test_cross := train_track_ids.join(
                        test_track_ids, "row_id"
                    ).collect(engine="streaming")
                ).is_empty()
                or not (
                    test_train_cross := test_track_ids.join(
                        train_track_ids, "row_id"
                    ).collect(engine="streaming")
                ).is_empty()
            ):
                raise Exception(
                    f"Bad split:\n{train_test_cross=}\n{test_train_cross}"
                )

            train_data = (
                processed_data.join(
                    train_track_ids, "row_id", "left", coalesce=False
                )
                .filter(pl.col("row_id_right").is_null().not_())
                .select(pl.all().exclude("row_id_right"))
            )

            test_data = processed_data.join(
                    test_track_ids, "row_id", "left", coalesce=False
                ).filter(pl.col("row_id_right").is_null().not_()).select(pl.all().exclude("row_id_right"))


            logger.info(
                f"{data_params}: {train_data.group_by(by="liked").agg(pl.nth(0).len()).collect(engine="streaming")=}"
            )
            if logger.isEnabledFor(logging.DEBUG):
                train_nulls = train_data.null_count().collect(engine="streaming")
                logger.debug(
                    f"{data_params}: {train_nulls.select(col.name for col in train_nulls.select(pl.all() > 0) if col.all())=}"
                )
            logger.info(
                f"{data_params}: {test_data.group_by(by="liked").agg(pl.nth(0).len()).collect(engine="streaming")=}"
            )
            if logger.isEnabledFor(logging.DEBUG):
                test_nulls = test_data.null_count().collect(engine="streaming")
                logger.debug(
                    f"{data_params}: {test_nulls.select(col.name for col in test_nulls.select(pl.all() > 0) if col.all())=}"
                )

            train_dataset = train_data.select(pl.all().exclude("row_id", "liked")).collect(engine="streaming").fill_null(0.0).sample(fraction=1.0, shuffle=True)
            train_dataset_numpy = train_dataset.to_numpy()

            for model_name, model, *agglomeration_fraction in [
                (
                        "lof-pyod",
                        LOF(contamination=contamination_fraction, novelty=True,
                            n_neighbors=20,
                            leaf_size=30,),
                ),
                (
                    "knn-default",
                    KNN(contamination=contamination_fraction,),
                ),
                (
                    "knn-neighbours=20,leaf_size=30",
                    KNN(contamination=contamination_fraction, n_neighbors=20, leaf_size=30),
                ),
                (
                    "ecod",
                    ECOD(contamination=contamination_fraction,),
                ),
                (
                    "abod",
                    ABOD(contamination=contamination_fraction,),
                ),
                # (
                #     "mad",
                #     MAD(contamination=contamination_fraction,),
                # ),
                (
                    "kpca",
                    KPCA(contamination=contamination_fraction,),
                ),
                (
                    "cof",
                    COF(contamination=contamination_fraction,),
                ),
                (
                    "cblof",
                    CBLOF(contamination=contamination_fraction,),
                ),
                # (
                #     "mo_gaal",
                #     MO_GAAL(contamination=contamination_fraction,),
                # ),
                # (
                #     "anogan",
                #     AnoGAN(contamination=contamination_fraction,),
                # ),
                # (
                #     "lunar",
                #     LUNAR(contamination=contamination_fraction,),
                # ),
                # (
                #     "suod",
                #     SUOD(base_estimators= [LOF(n_neighbors=15, contamination=contamination_fraction), KNN(contamination=contamination_fraction),
                #                            COPOD(contamination=contamination_fraction), IForest(n_estimators=100, contamination=contamination_fraction),
                #                            ABOD(contamination=contamination_fraction,), KNN(contamination=contamination_fraction,),
                #                            AnoGAN(contamination=contamination_fraction,),], n_jobs=2, combination='average',
                #          verbose=False),
                # ),
            ]:

                # 0 stands for inliers and 1 for outliers
                one_class_test_data = test_data.collect(engine="streaming").fill_null(0.0)

                model.fit(train_dataset_numpy)

                test_feature_data = one_class_test_data.select(pl.all().exclude("row_id", "liked")).sample(fraction=1.0, shuffle=True).to_numpy()

                y_predicted = model.predict(
                    test_feature_data
                )
                model_accuracy = accuracy_score(
                    one_class_test_data.select(pl.col("liked")), y_predicted
                )
                logger.info(
                    f"{data_params}[{model_name}], try={try_n + 1}: {model_accuracy:.3f}"
                )
                accuracies[model_name].append(model_accuracy)
        with data_report.open("at") as f:
            lines = [
                f"{data_params}: {", ".join([f"{model_name}={float(np.mean(tries)):.3f}(+-{np.var(tries):.3f})" for model_name, tries in accuracies.items()])}"
            ]
            f.writelines(lines + ["\n"])
