import dataclasses
import logging
import operator
import pathlib
import random
import sys
import typing
from collections import defaultdict
from functools import reduce
from itertools import product
from typing import Callable

import numpy
import numpy as np
import polars as pl
from pyod.models.combination import majority_vote
from pyod.models.ecod import ECOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lscp import LSCP
from pyod.models.suod import SUOD
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from audio.features import extract_features_for_mp3, AudioFeatures, prepare_extractor
from dataset.persistent_dataset_processor import DataFrameBuilder

if __name__ == "__main__":
    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[handler],
    )
    logger = logging.getLogger(__file__)

    class Majority:

        def __init__(self, estimators):
            self.estimators = estimators

        def fit(self, train_dataset_numpy):
            for estimator in self.estimators:
                estimator.fit(train_dataset_numpy)

        def predict(self, test_data):
            predictions = np.stack(
                [estimator.predict(test_data) for estimator in self.estimators], axis=1
            )
            return majority_vote(predictions)


    class Pipe:

        def __init__(self, estimator, input_transformer: Callable):
            self.estimator = estimator
            self.input_transformer = input_transformer

        def fit(self, train_dataset_numpy):
            return self.estimator.fit(self.input_transformer(train_dataset_numpy))

        def predict(self, test_data):
            return self.estimator.predict(self.input_transformer(test_data))

    # for tweak in options:
    @dataclasses.dataclass
    class MappedAudioFeatures(AudioFeatures):
        liked: int
        lowlevel______barkbands______max: list[float]
        lowlevel______barkbands______mean: list[float]
        lowlevel______barkbands______min: list[float]
        lowlevel______barkbands______var: list[float]
        lowlevel______erbbands______max: list[float]
        lowlevel______erbbands______mean: list[float]
        lowlevel______erbbands______min: list[float]
        lowlevel______erbbands______var: list[float]
        lowlevel______gfcc______mean: list[float]
        lowlevel______melbands______max: list[float]
        lowlevel______melbands______mean: list[float]
        lowlevel______melbands______min: list[float]
        lowlevel______melbands______var: list[float]
        lowlevel______melbands128______max: list[float]
        lowlevel______melbands128______mean: list[float]
        lowlevel______melbands128______min: list[float]
        lowlevel______melbands128______var: list[float]
        lowlevel______mfcc______mean: list[float]
        lowlevel______spectral_contrast_coeffs______max: list[float]
        lowlevel______spectral_contrast_coeffs______mean: list[float]
        lowlevel______spectral_contrast_coeffs______min: list[float]
        lowlevel______spectral_contrast_coeffs______var: list[float]
        lowlevel______spectral_contrast_valleys______max: list[float]
        lowlevel______spectral_contrast_valleys______mean: list[float]
        lowlevel______spectral_contrast_valleys______min: list[float]
        lowlevel______spectral_contrast_valleys______var: list[float]
        rhythm______beats_loudness_band_ratio______dmean: list[float]
        rhythm______beats_loudness_band_ratio______dmean2: list[float]
        rhythm______beats_loudness_band_ratio______dvar: list[float]
        rhythm______beats_loudness_band_ratio______dvar2: list[float]
        rhythm______beats_loudness_band_ratio______max: list[float]
        rhythm______beats_loudness_band_ratio______mean: list[float]
        rhythm______beats_loudness_band_ratio______median: list[float]
        rhythm______beats_loudness_band_ratio______min: list[float]
        rhythm______beats_loudness_band_ratio______stdev: list[float]
        rhythm______beats_loudness_band_ratio______var: list[float]
        tonal______hpcp______dmean: list[float]
        tonal______hpcp______dmean2: list[float]
        tonal______hpcp______dvar: list[float]
        tonal______hpcp______dvar2: list[float]
        tonal______hpcp______max: list[float]
        tonal______hpcp______mean: list[float]
        tonal______hpcp______median: list[float]
        tonal______hpcp______min: list[float]
        tonal______hpcp______stdev: list[float]
        tonal______hpcp______var: list[float]
        rhythm______bpm_histogram: list[float]
        tonal______chords_histogram: list[float]
        tonal______thpcp: list[float]
        danceability___msd___musicnn___1______danceable: float
        engagement_regression___discogs___effnet___1______engagement: float
        deam___msd___musicnn___2______valence: float
        emomusic___msd___musicnn___2______valence: float
        engagement_regression___discogs___effnet___1______engagement: float
        mood_acoustic___msd___musicnn___1______acoustic: float
        mood_aggressive___msd___musicnn___1______aggressive: float
        mood_electronic___msd___musicnn___1______electronic: float
        mood_happy___msd___musicnn___1______happy: float
        mood_party___msd___musicnn___1______non_party: float
        mood_relaxed___msd___musicnn___1______non_relaxed: float
        mood_sad___msd___musicnn___1______non_sad: float
        moods_mirex___msd___musicnn___1______passionate_rousing_confident_boisterous_rowdy: (
            float
        )
        moods_mirex___msd___musicnn___1______rollicking_cheerful_fun_sweet_amiable_good_natured: (
            float
        )
        moods_mirex___msd___musicnn___1______literate_poignant_wistful_bittersweet_autumnal_brooding: (
            float
        )
        moods_mirex___msd___musicnn___1______humorous_silly_campy_quirky_whimsical_witty_wry: (
            float
        )
        moods_mirex___msd___musicnn___1______aggressive_fiery_tense_anxious_intense_volatile_visceral: (
            float
        )
        muse___msd___musicnn___2______valence: float
        nsynth_acoustic_electronic___discogs___effnet___1______acoustic: float
        nsynth_bright_dark___discogs___effnet___1______bright: float
        timbre___discogs___effnet___1______bright: float
        tonal_atonal___msd___musicnn___1______atonal: float
        voice_instrumental___msd___musicnn___1______instrumental: float

    n_jobs = 1
    bps_flag = n_jobs != 1
    contamination_fraction = 0.33
    test_fraction = 0.33
    train_tries = 5
    # feature_bagging_n_estimators=20
    # feature_bagging_type=None,ecod,knn,loda
    # lscp_local_region_size=60(->15)
    # loda_n_random_cuts=100(->200)
    # lscp_n_bins=10(->20)
    # suod_combination=average,maximization

    data_param_options = [
        tuple(
            reduce(
                operator.iconcat,
                (e if isinstance(e, list) else [e] for e in variant),
                [],
            )
        )
        for variant in [["contamination_fraction=0.33"]]
    ]
    data_report = pathlib.Path("data_report.csv")
    if data_report.exists():
        processed_variants = data_report.read_text().split("\n")
        data_param_options = [
            v
            for v in data_param_options
            if not any(line.startswith(f"{v}: ") for line in processed_variants)
        ]

    def stats(progress_state: DataFrameBuilder.ProgressStat):
        logger.info(
            f"{progress_state.succeed=}/{progress_state.failed=}/{progress_state.data_size=}"
        )

    liked_audio_path = pathlib.Path("data/118517468/liked")
    disliked_audio_path = pathlib.Path("data/118517468/disliked")

    extractor = prepare_extractor()

    def common_extractor(
        audio_path: pathlib.Path, is_liked: bool, track_id: str
    ) -> MappedAudioFeatures:
        features = extract_features_for_mp3(audio_path.joinpath(track_id), extractor)

        def unwrap(v):
            if hasattr(v, "tolist"):
                return v.tolist()
            elif hasattr(v, "item"):
                return v.item()
            else:
                return v

        remapped_values = {
            k: unwrap(v) for k, v in dataclasses.asdict(features).items()
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
                generator=(f.name for f in liked_audio_path.iterdir() if f.is_file()),
                result_schema=("row_id", pl.String),
            ),
            mappers=[liked_feature_extractor],
            batch_size=25,
            cleanup_on_exit=False,
            progress_tracker=stats,
        ) as liked_frame_result,
    ):
        logger.info(f"Started processing")

        processed_data = (
            pl.concat([disliked_frame_result.ldf, liked_frame_result.ldf])
            .filter(
                pl.col(disliked_frame_result.processing_status_column[0])
                == DataFrameBuilder.PROCESSING_SUCCEED_VALUE
            )
            .select(pl.all().exclude(disliked_frame_result.processing_status_column[0]))
        )

        metric_sizes = {
            field.name: size_literal[0]
            for field in dataclasses.fields(AudioFeatures)
            if (root_args := typing.get_args(field.type))
            and (size_literal := typing.get_args(typing.get_args(root_args[0])[0]))
        }

        container_field_names = set(
            field.name
            for field in dataclasses.fields(MappedAudioFeatures)
            if field.type
            in [
                list[float],
            ]
        )
        if diff := (container_field_names - metric_sizes.keys()):
            raise Exception(f"For fields {diff} size is unknown")

        scale_mapping = {
            "major": 1,
            "minor": 0,
        }
        keys = {
            "C": 0,
            "C#": 1,
            "D": 2,
            "D#": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "G": 7,
            "G#": 8,
            "A": 9,
            "A#": 10,
            "B": 11,
        }
        alias_keys = {
            "Db": keys["C#"],
            "Eb": keys["D#"],
            "Gb": keys["F#"],
            "Ab": keys["G#"],
            "Bb": keys["A#"],
            "Cb": keys["B"],
        }
        key_mapping = {
            k: {
                "sin": numpy.sin(2 * numpy.pi * v / len(keys)),
                "cos": numpy.cos(2 * numpy.pi * v / len(keys)),
            }
            for k, v in (keys | alias_keys).items()
        }

        key_columns = [
            field.name
            for field in dataclasses.fields(MappedAudioFeatures)
            if field.type == str and field.name.endswith("_key")
        ]

        scale_columns = [
            field.name
            for field in dataclasses.fields(MappedAudioFeatures)
            if field.type == str and field.name.endswith("_scale")
        ]
        processed_data = processed_data.with_columns(
            *[
                pl.col(column_name).replace_strict(
                    key_mapping,
                    return_dtype=pl.Struct(
                        {
                            f"{column_name}-sin": pl.Float64,
                            f"{column_name}-cos": pl.Float64,
                        }
                    ),
                )
                for column_name in key_columns
            ],
            *[
                pl.col(column_name).replace_strict(scale_mapping, return_dtype=pl.Int64)
                for column_name in scale_columns
            ],
        ).unnest(key_columns)

        def extract_param(param_row, name, converter=None):
            val = [fract for fract in param_row if fract.startswith(f"{name}=")][
                0
            ].split("=", 2)[1]
            return converter(val) if converter else val

        for dn, data_params in enumerate(data_param_options, 1):
            accuracies = defaultdict(list)
            data_variant = processed_data
            logger.info(
                f"<{dn}/{len(data_param_options)}>{data_params=}: {data_variant.collect_schema().len()=}"
            )

            for field_name, field_size in metric_sizes.items():
                data_variant = data_variant.with_columns(
                    pl.col(field_name).list.mean().alias(f"{field_name}_mean"),
                    pl.col(field_name).list.std().alias(f"{field_name}_std"),
                    pl.col(field_name).list.min().alias(f"{field_name}_min"),
                    pl.col(field_name).list.max().alias(f"{field_name}_max"),
                )
                if "with_var_agg" in data_params:
                    data_variant = data_variant.with_columns(
                        pl.col(field_name).list.var().alias(f"{field_name}_var")
                    )
                data_variant = data_variant.with_columns(
                    pl.col(field_name).list.to_struct(
                        fields=[f"{field_name}_{idx}" for idx in range(field_size)],
                        upper_bound=field_size,
                    )
                ).unnest(field_name)

            logger.info(
                f"<{dn}/{len(data_param_options)}>{data_params=}: {data_variant.collect_schema().len()=}"
            )

            for try_n in range(train_tries):
                shuffle_seed = random.randint(0, 100)
                test_disliked_track_ids = (
                    data_variant.filter(pl.col("liked") == 0)
                    .select(pl.col("row_id"))
                    .with_columns(pl.col("row_id").shuffle(seed=shuffle_seed))
                    .with_row_index()
                    .filter(pl.col("index") < pl.col("index").max() * test_fraction)
                    .select(pl.all().exclude("index"))
                )
                test_liked_track_ids = (
                    data_variant.filter(pl.col("liked") == 1)
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
                    data_variant.filter(pl.col("liked") == 0)
                    .select(pl.col("row_id"))
                    .join(test_disliked_track_ids, "row_id", "left", coalesce=False)
                    .filter(pl.col("row_id_right").is_null())
                    .select(pl.all().exclude("row_id_right"))
                )
                contamination_count = train_disliked_track_ids.with_row_index().filter(
                    pl.col("index") < pl.col("index").max() * contamination_fraction
                )
                train_liked_track_ids = (
                    data_variant.filter(pl.col("liked") == 1)
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
                    data_variant.join(train_track_ids, "row_id", "left", coalesce=False)
                    .filter(pl.col("row_id_right").is_null().not_())
                    .select(pl.all().exclude("row_id_right"))
                )

                test_data = (
                    data_variant.join(test_track_ids, "row_id", "left", coalesce=False)
                    .filter(pl.col("row_id_right").is_null().not_())
                    .select(pl.all().exclude("row_id_right"))
                )

                # logger.info(
                #     f"{train_data.group_by(by="liked").agg(pl.nth(0).len()).collect(engine="streaming")=}"
                # )
                if logger.isEnabledFor(logging.DEBUG):
                    train_nulls = train_data.null_count().collect(engine="streaming")
                    logger.debug(
                        f"{train_nulls.select(col.name for col in train_nulls.select(pl.all() > 0) if col.all())=}"
                    )
                # logger.info(
                #     f"{test_data.group_by(by="liked").agg(pl.nth(0).len()).collect(engine="streaming")=}"
                # )
                if logger.isEnabledFor(logging.DEBUG):
                    test_nulls = test_data.null_count().collect(engine="streaming")
                    logger.debug(
                        f"{test_nulls.select(col.name for col in test_nulls.select(pl.all() > 0) if col.all())=}"
                    )

                train_dataset = (
                    train_data.select(pl.all().exclude("row_id", "liked"))
                    .collect(engine="streaming")
                    .fill_null(0.0)
                    .fill_nan(0.0)
                    .sample(fraction=1.0, shuffle=True)
                )

                one_class_test_data = (
                    test_data.collect(engine="streaming").fill_null(0.0).fill_nan(0.0)
                )
                test_feature_data = one_class_test_data.select(
                    pl.all().exclude("row_id", "liked")
                ).sample(fraction=1.0, shuffle=True)

                scaler = StandardScaler()
                train_dataset_numpy = scaler.fit_transform(train_dataset)
                test_feature_processed_data = scaler.fit_transform(
                    test_feature_data
                )
                logger.info(f"Starting try={try_n + 1}")
                model_cfgs = []

                def create_estimators():
                    return {
                        "ecod": ECOD(contamination=contamination_fraction, n_jobs=n_jobs),
                        "loda": LODA(
                            contamination=contamination_fraction,
                            n_random_cuts=150,
                        ),
                        "knn": KNN(
                            contamination=contamination_fraction,
                            n_neighbors=5,
                            metric="l1",
                            method="mean",
                            n_jobs=n_jobs,
                        ),
                    }

                def create_ensembles():
                    suods = {
                        f"suod[{combination=}]": SUOD(
                            base_estimators=list(create_estimators().values()),
                            contamination=contamination_fraction,
                            combination=combination,
                            bps_flag=bps_flag,
                            n_jobs=n_jobs,
                        )
                        for combination in ["average", "maximization"]
                    }
                    feature_baggings = {
                        f"feature_bagging[estimator_type={etype}]": FeatureBagging(
                            base_estimator=base_estimator,
                            contamination=contamination_fraction,
                            n_estimators=20,
                            n_jobs=n_jobs,
                        )
                        for etype, base_estimator in [((etype, estimators[etype]) if etype else (None, None)) for etype, estimators in product([None, "ecod", "knn", "loda"], [create_estimators()]) ]
                    }
                    lscps = {
                        f"lscp": LSCP(
                            detector_list=list(create_estimators().values()),
                            contamination=contamination_fraction,
                            n_bins=15,
                            local_region_size=27,
                        )
                    }
                    return suods | feature_baggings | lscps

                def create_combinations():
                    suods_of_feature_baggings = {
                        f"suod_of_feature_baggings[{combination=},base_estimators={"+".join([e[0].replace("[", "{").replace("]", "}") for e in feature_baggings])}]": SUOD(
                            base_estimators=[e[1] for e in feature_baggings],
                            contamination=contamination_fraction,
                            combination=combination,
                            bps_flag=bps_flag,
                            n_jobs=n_jobs,
                        ) for combination, feature_baggings in product(
                            ["average", "maximization"],
                            [[e for e in create_ensembles().items() if isinstance(e[1], FeatureBagging)]],
                        )
                    }
                    suods_of_feature_bagging_and_lscp = {
                        f"suod_of_feature_bagging_and_lscp[{combination=},base_estimators={"+".join([e[0].replace("[", "{").replace("]", "}") for e in lscps])}]": SUOD(
                            base_estimators=[e[1] for e in lscps],
                            contamination=contamination_fraction,
                            combination=combination,
                            bps_flag=bps_flag,
                            n_jobs=n_jobs,
                        ) for combination, lscps in product(
                            ["average", "maximization"],
                            [[e for e in create_ensembles().items() if isinstance(e[1], (LSCP, FeatureBagging))]],
                        )
                    }
                    suods_of_feature_bagging_and_lscp = {
                        f"suod_of_feature_bagging_and_lscp[{combination=},base_estimators={"+".join([e[0].replace("[", "{").replace("]", "}") for e in lscps])}]": SUOD(
                            base_estimators=[e[1] for e in lscps],
                            contamination=contamination_fraction,
                            combination=combination,
                            bps_flag=bps_flag,
                            n_jobs=n_jobs,
                        ) for combination, lscps in product(
                            ["average", "maximization"],
                            [[e for e in create_ensembles().items() if isinstance(e[1], (LSCP, FeatureBagging))]],
                        )
                    }
                    # feature_baggings_of_suods = {
                    #     f"feature_baggings_of_suods[base_estimator={suod[0].replace("[", "{").replace("]", "}")}]": FeatureBagging(
                    #         base_estimator=suod[1],
                    #         contamination=contamination_fraction,
                    #         n_estimators=20,
                    #         n_jobs=n_jobs,
                    #     )
                    #     for suod in [e for e in create_ensembles().items() if isinstance(e[1], SUOD)]
                    # }
                    # feature_baggings_of_lscps = {
                    #     f"feature_bagging_of_lscps[base_estimator={lscp[0].replace("[", "{").replace("]", "}")}]": FeatureBagging(
                    #         base_estimator=lscp[1],
                    #         contamination=contamination_fraction,
                    #         n_estimators=20,
                    #         n_jobs=n_jobs,
                    #     )
                    #     for lscp in [e for e in create_ensembles().items() if isinstance(e[1], LSCP)]
                    # }
                    lscps_of_feature_baggings = {
                        f"lscp_of_feature_baggings[detector_list={"+".join([e[0].replace("[", "{").replace("]", "}") for e in feature_baggings])}]": LSCP(
                            detector_list=[e[1] for e in feature_baggings],
                            contamination=contamination_fraction,
                            n_bins=15,
                            local_region_size=27,
                        )
                        for feature_baggings in [[e for e in create_ensembles().items() if isinstance(e[1], FeatureBagging)]]
                    }
                    lscps_of_feature_baggings_and_suods = {
                        f"lscp_of_feature_baggings_and_suods[detector_list={"+".join([e[0].replace("[", "{").replace("]", "}") for e in feature_baggings])}]": LSCP(
                            detector_list=[e[1] for e in feature_baggings],
                            contamination=contamination_fraction,
                            n_bins=15,
                            local_region_size=27,
                        )
                        for feature_baggings in [[e for e in create_ensembles().items() if isinstance(e[1], (FeatureBagging, SUOD))]]
                    }
                    return (suods_of_feature_baggings | suods_of_feature_bagging_and_lscp
                    # | feature_baggings_of_suods | feature_baggings_of_lscps
                    | lscps_of_feature_baggings | lscps_of_feature_baggings_and_suods)

                pca = PCA(min(train_dataset.shape[0], test_feature_data.shape[0]))
                pca.fit(train_dataset)
                model_cfgs.extend([
                    *create_estimators().items(),
                    *{f"pca_{name}": Pipe(value, lambda data: pca.transform(data)) for name, value in create_estimators().items()}.items(),
                    *create_ensembles().items(),
                    *{f"pca_{name}": Pipe(value, lambda data: pca.transform(data)) for name, value in create_ensembles().items()}.items(),
                    *create_combinations().items(),
                    *{f"pca_{name}": Pipe(value, lambda data: pca.transform(data)) for name, value in create_combinations().items()}.items(),
                    (
                        f"majority_estimators",
                        Majority(list(create_estimators().values())),
                    ),
                    (
                        f"pca_majority_estimators",
                        Pipe(Majority(list(create_estimators().values())), lambda data: pca.transform(data)),
                    ),
                    *[(
                        f"majority_ensembles[{"+".join([e[0].replace("[", "{").replace("]", "}") for e in ensemles])}]",
                        Majority([e[1] for e in ensemles]),
                    ) for ensemles in [create_ensembles().items()]],
                    *[(
                        f"pca_majority_ensembles[{"+".join([e[0].replace("[", "{").replace("]", "}") for e in ensemles])}]",
                        Pipe(Majority([e[1] for e in ensemles]), lambda data: pca.transform(data)),
                    ) for ensemles in [create_ensembles().items()]],
                    # *[(
                    #     f"majority_combinations[{"+".join([e[0].replace("[", "{").replace("]", "}") for e in combinations])}]",
                    #     Majority([e[1] for e in combinations]),
                    # ) for combinations in [create_combinations().items()]],
                ])
                vc = 0
                total_models = len(model_cfgs)
                while model_cfgs and (v := model_cfgs.pop(0)):
                    model_name, model = v
                    vc +=1
                    model.fit(train_dataset_numpy)
                    y_predicted = model.predict(test_feature_processed_data)
                    model_accuracy = accuracy_score(
                        one_class_test_data.select(pl.col("liked")), y_predicted
                    )
                    logger.info(
                        f"<{dn}/{len(data_param_options)}>{data_params}, try={try_n + 1} -> {model_name}({vc}/{total_models}): {model_accuracy:.3f}"
                    )
                    accuracies[model_name].append(model_accuracy)

            data_variant_results = list(
                sorted(
                    {
                        k: (float(np.mean(v)), float(np.std(v)))
                        for k, v in accuracies.items()
                    }.items(),
                    key=lambda e: e[1][0],
                    reverse=True,
                )
            )
            line = f"{data_params}: {";".join([f"{model_name}->{mean:.3f}+-{std:.3f}" for model_name, (mean, std) in data_variant_results if mean > 0.5])}"
            with data_report.open("at") as f:
                f.writelines([line] + ["\n"])
            logger.info(f"<{dn}/{len(data_param_options)}>{line}")
