import dataclasses
import json
import logging
import operator
import pathlib
import random
import re
import sys
import typing
from collections import defaultdict
from functools import reduce
from itertools import product

import numpy
import numpy as np
import polars as pl
from pyod.models.abod import ABOD
from pyod.models.anogan import AnoGAN
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.ecod import ECOD
from pyod.models.knn import KNN
from pyod.models.kpca import KPCA
from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR
from pyod.models.mo_gaal import MO_GAAL
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

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
        moods_mirex___msd___musicnn___1______passionate_rousing_confident_boisterous_rowdy: float
        moods_mirex___msd___musicnn___1______rollicking_cheerful_fun_sweet_amiable_good_natured: float
        moods_mirex___msd___musicnn___1______literate_poignant_wistful_bittersweet_autumnal_brooding: float
        moods_mirex___msd___musicnn___1______humorous_silly_campy_quirky_whimsical_witty_wry: float
        moods_mirex___msd___musicnn___1______aggressive_fiery_tense_anxious_intense_volatile_visceral: float
        muse___msd___musicnn___2______valence: float
        nsynth_acoustic_electronic___discogs___effnet___1______acoustic: float
        nsynth_bright_dark___discogs___effnet___1______bright: float
        timbre___discogs___effnet___1______bright: float
        tonal_atonal___msd___musicnn___1______atonal: float
        voice_instrumental___msd___musicnn___1______instrumental: float

    test_fraction = 0.33
    train_tries = 10
    data_param_options = [tuple(reduce(operator.iconcat, (e if isinstance(e, list) else [e] for e in variant), [])) for variant in product(
        [["raw", "aggregates"], "raw", "aggregates"],
        ["standard_scaling", "min_max_scaling", "abs_max_scaling", "no_scaling"],
        ["pca", "no_pca"],
        ["contamination_fraction=0.1", "contamination_fraction=0.25", "contamination_fraction=0.33"],
    )]
    data_report = pathlib.Path("data_report.csv")
    checked_variants = set()
    if data_report.exists():
        with data_report.open(mode="rt") as f:
            checked_variants = {
                frozenset(json.loads(m.group(0).replace("'", '"')).items())
                for line in f.readlines()
                if (m := re.match("{.+}", line))
            }

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
        features = extract_features_for_mp3(
            audio_path.joinpath(track_id),
            extractor
        )
        def unwrap(v):
            if hasattr(v, "tolist"):
                return v.tolist()
            elif hasattr(v, "item"):
                return v.item()
            else:
                return v
        remapped_values = {
            k: unwrap(v)
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
        logger.info(f"Started processing")

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
        key_mapping = {k: {"sin": numpy.sin(2 * numpy.pi * v / len(keys)), "cos": numpy.cos(2 * numpy.pi * v / len(keys))} for k, v in (keys | alias_keys).items()}

        key_columns = [field.name
        for field in dataclasses.fields(MappedAudioFeatures) if field.type == str and field.name.endswith("_key")]

        scale_columns = [field.name
        for field in dataclasses.fields(MappedAudioFeatures) if field.type == str and field.name.endswith("_scale")]
        processed_data = processed_data.with_columns(
            *[
                pl.col(column_name).replace_strict(key_mapping, return_dtype=pl.Struct({f"{column_name}-sin": pl.Float64, f"{column_name}-cos": pl.Float64})) for column_name in key_columns
            ],
            *[
                pl.col(column_name).replace_strict(scale_mapping, return_dtype=pl.Int64) for column_name in scale_columns
            ]
        ).unnest(key_columns)

        accuracies: dict[tuple[str, ...], dict[str, list[float]] | list[tuple[str, tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        for data_params in data_param_options:
            contamination_fraction = float([fract for fract in data_params if fract.startswith("contamination_fraction=")][0].split("=", 2)[1])
            data_variant = processed_data
            logger.info(f"{data_params=}: {data_variant.collect_schema().len()=}")
            for field_name, field_size in metric_sizes.items():
                if "aggregates" in data_params:
                    data_variant = data_variant.with_columns(
                        pl.col(field_name).list.mean().alias(f"{field_name}_mean"),
                        pl.col(field_name).list.var().alias(f"{field_name}_var"),
                        pl.col(field_name).list.std().alias(f"{field_name}_std"),
                        pl.col(field_name).list.min().alias(f"{field_name}_min"),
                        pl.col(field_name).list.max().alias(f"{field_name}_max"),
                    )
                if "raw" in data_params:
                    data_variant = (
                        data_variant.with_columns(
                            pl.col(field_name).list.to_struct(
                                fields=[
                                    f"{field_name}_{idx}"
                                    for idx in range(field_size)
                                ],
                                upper_bound=field_size,
                            )
                        )
                        .unnest(field_name)
                    )
                else:
                    data_variant = data_variant.drop(field_name)

            logger.info(f"{data_params=}: {data_variant.collect_schema().len()=}")

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
                    data_variant.join(
                        train_track_ids, "row_id", "left", coalesce=False
                    )
                    .filter(pl.col("row_id_right").is_null().not_())
                    .select(pl.all().exclude("row_id_right"))
                )

                test_data = data_variant.join(
                        test_track_ids, "row_id", "left", coalesce=False
                    ).filter(pl.col("row_id_right").is_null().not_()).select(pl.all().exclude("row_id_right"))


                logger.info(
                    f"{train_data.group_by(by="liked").agg(pl.nth(0).len()).collect(engine="streaming")=}"
                )
                if logger.isEnabledFor(logging.DEBUG):
                    train_nulls = train_data.null_count().collect(engine="streaming")
                    logger.debug(
                        f"{train_nulls.select(col.name for col in train_nulls.select(pl.all() > 0) if col.all())=}"
                    )
                logger.info(
                    f"{test_data.group_by(by="liked").agg(pl.nth(0).len()).collect(engine="streaming")=}"
                )
                if logger.isEnabledFor(logging.DEBUG):
                    test_nulls = test_data.null_count().collect(engine="streaming")
                    logger.debug(
                        f"{test_nulls.select(col.name for col in test_nulls.select(pl.all() > 0) if col.all())=}"
                    )

                train_dataset = train_data.select(pl.all().exclude("row_id", "liked")).collect(engine="streaming").fill_null(0.0).fill_nan(0.0).sample(fraction=1.0, shuffle=True)

                one_class_test_data = test_data.collect(engine="streaming").fill_null(0.0).fill_nan(0.0)
                test_feature_data = one_class_test_data.select(pl.all().exclude("row_id", "liked")).sample(fraction=1.0, shuffle=True)

                if "no_scaling" in data_params:
                    train_dataset_numpy = train_dataset.to_numpy()
                elif "standard_scaling" in data_params:
                    scaler = StandardScaler()
                    train_dataset_numpy = scaler.fit_transform(train_dataset)
                elif "min_max_scaling" in data_params:
                    scaler = MinMaxScaler()
                    train_dataset_numpy = scaler.fit_transform(train_dataset)
                elif "abs_max_scaling" in data_params:
                    scaler = MaxAbsScaler()
                    train_dataset_numpy = scaler.fit_transform(train_dataset)

                if "pca" in data_params:
                    pca = PCA(min(train_dataset.shape[0], test_feature_data.shape[0]))
                    train_dataset_numpy = pca.fit_transform(train_dataset_numpy)

                for model_name, model in [
                    (
                        "lof-pyod",
                        LOF(contamination=contamination_fraction, novelty=True,
                            n_neighbors=20,
                            leaf_size=30,metric="cosine"),
                    ),
                    (
                        "knn-default",
                        KNN(contamination=contamination_fraction, metric="l1"),
                    ),
                    (
                        "knn-neighbours=20,leaf_size=30",
                        KNN(contamination=contamination_fraction, n_neighbors=20, leaf_size=30, metric="l2"),
                    ),
                    (
                        "ecod",
                        ECOD(contamination=contamination_fraction,),
                    ),
                    (
                        "abod",
                        ABOD(contamination=contamination_fraction, method="default"),
                    ),
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
                    (
                        "mo_gaal",
                        MO_GAAL(contamination=contamination_fraction,),
                    ),
                    (
                        "anogan",
                        AnoGAN(contamination=contamination_fraction,),
                    ),
                    (
                        "lunar",
                        LUNAR(contamination=contamination_fraction,),
                    ),
                    # (
                    #     "suod",
                    #     SUOD(base_estimators= [LOF(n_neighbors=15, contamination=contamination_fraction), KNN(contamination=contamination_fraction),
                    #                            COPOD(contamination=contamination_fraction), IForest(n_estimators=100, contamination=contamination_fraction),
                    #                            ABOD(contamination=contamination_fraction,), KNN(contamination=contamination_fraction,),
                    #                            AnoGAN(contamination=contamination_fraction,),], n_jobs=2, combination='average',
                    #          verbose=False),
                    # ),
                ]:

                    model.fit(train_dataset_numpy)
                    if "no_scaler" in data_params:
                        test_feature_processed_data = test_feature_data.to_numpy()
                    else:
                        test_feature_processed_data = scaler.fit_transform(test_feature_data)
                    if "pca" in data_params:
                        test_feature_processed_data = pca.fit_transform(test_feature_processed_data)

                    y_predicted = model.predict(test_feature_processed_data)
                    model_accuracy = accuracy_score(
                        one_class_test_data.select(pl.col("liked")), y_predicted
                    )
                    logger.info(
                        f"{model_name}[{data_params}], try={try_n + 1}: {model_accuracy:.3f}"
                    )
                    accuracies[data_params][model_name].append(model_accuracy)

            accuracies[data_params] = list(sorted({k: (float(np.mean(v)), float(np.std(v))) for k, v in accuracies[data_params].items()}.items(), key=lambda e: e[1][0], reverse=True))
            print(f"{data_params=}: {";".join([f"{model_name}={mean:.3f},+-{std:.3f}" for model_name, (mean, std) in accuracies[data_params] if mean > 0.5])}")

        with data_report.open("at") as f:
            lines = [
                f"{data_params=}: {";".join([f"{model_name}={mean:.3f},+-{std:.3f}" for model_name, (mean, std) in model_stats if mean > 0.5])}\n" for data_params, model_stats in accuracies.items()
            ]
            f.writelines(lines + ["\n"])
