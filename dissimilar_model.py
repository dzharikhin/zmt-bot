import dataclasses
import logging
import operator
import pathlib
import random
import shutil
import sys
import tempfile
import typing
from collections import defaultdict
from functools import reduce
from itertools import product

import numpy
import numpy as np
import polars as pl
from clustpy.alternative import AutoNR
from clustpy.deep import VaDE, DDC, DeepECT, DipDECK
from clustpy.density import MultiDensityDBSCAN
from clustpy.partition import XMeans, DipMeans, DipNSub, GapStatistic, GMeans, PGMeans, ProjectedDipMeans, SkinnyDip, \
    SpecialK
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_is_fitted
from skopt import BayesSearchCV

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

    def _get_labels(estimator: Pipeline):
        """Gets the cluster labels from an estimator or pipeline."""
        if isinstance(estimator, Pipeline):
            check_is_fitted(estimator.steps[-1][1], ["labels_"])
            labels = estimator.steps[-1][1].labels_
        else:
            check_is_fitted(estimator, ["labels_"])
            labels = estimator.labels_
        return labels


    class NoCV(BaseCrossValidator):

        def get_n_splits(self, X=None, y=None, groups=None):
            return 1

        def split(self, X, y=None, groups=None):
            yield np.arange(X.shape[0]), np.arange(X.shape[0])


    class LabelScorerUnsupervised:

        def __init__(self, scoring_func):
            super().__init__()
            self._scoring_func = scoring_func



        def __call__(self, estimator, X, labels_true=None):
            labels = _get_labels(estimator)
            if isinstance(estimator, Pipeline):
                X = estimator[:-1].transform(X)
            # X, labels = _remove_noise_cluster(X, labels, labels=labels)
            return self._scoring_func(labels)

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
    train_tries = 1
    optimization_iterations = 50
    disliked_fraction_cluster_threshold = 0.7
    interesting_clusterization_limit = 0.66

    data_param_options = [
        tuple(
            reduce(
                operator.iconcat,
                (e if isinstance(e, list) else [e] for e in variant),
                [],
            )
        )
        for variant in product([["raw", "aggregates"], "raw", "aggregates"], ["with_var_agg", "no_var_agg"])
    ]

    report_root = pathlib.Path("model_reports")
    def build_model_report_prefix(data_options, full_model_name):
        return f"{data_options}&{full_model_name}"
    def build_full_model_name(base_name, preprocess_steps):
        return "|".join([*preprocess_steps, base_name])
    def get_base_model_name(full_name):
        return full_name.split("|")[-1].lower()
    def filter_pipelines(pipelines, data_opts):
        def is_already_processed(pipeline):
            m_name = pipeline.steps[-1][0]
            base_model_name = get_base_model_name(m_name)
            processed_variants = []
            if (model_report := report_root.joinpath(f"{base_model_name}.csv")).exists():
                processed_variants = model_report.read_text().split("\n")
            return any(l for l in processed_variants if l.startswith(f"{data_opts}&{m_name}"))

        return [pipeline for pipeline in pipelines if not is_already_processed(pipeline)]

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

        for dn, data_params in enumerate(data_param_options, 1):
            accuracies = defaultdict(list)
            data_variant = processed_data
            logger.info(
                f"<{dn}/{len(data_param_options)}>{data_params=}: initial {data_variant.collect_schema().len()=}"
            )

            for field_name, field_size in metric_sizes.items():
                if "aggregates" in data_params:
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

            logger.info(
                f"<{dn}/{len(data_param_options)}>{data_params=}: transformed {data_variant.collect_schema().len()=}"
            )
            data_variant = data_variant.select(pl.all().exclude("row_id")).collect(engine="streaming").fill_null(0.0).fill_nan(0.0)

            for try_n in range(train_tries):
                shuffle_seed = random.randint(0, 100)
                data_variant = data_variant.sample(fraction=1.0, shuffle=True, seed=shuffle_seed)
                X_numpy = data_variant.select(pl.all().exclude("liked")).to_numpy()
                Y_numpy = data_variant.select(pl.col("liked")).to_numpy()

                def disliked_matching_score(labels, **kwargs) -> float:
                    cluster_column = pl.Series(name="cluster", values=labels)
                    clustered_data_variant = data_variant.with_columns(cluster_column)

                    cluster_outlier_member_fraction = clustered_data_variant.select(pl.col("cluster"), pl.col("liked")).group_by("cluster").agg(
                        (pl.col("cluster").filter(pl.col("liked") == 0).count() / pl.col("cluster").count()).alias(
                            "disliked_fraction"),
                        pl.col("cluster").filter(pl.col("liked") == 0).count().alias("disliked"),
                        pl.col("cluster").count().alias("all"),
                    )
                    outlier_clusters = cluster_outlier_member_fraction.filter(pl.col("disliked_fraction") > disliked_fraction_cluster_threshold).get_column("cluster").to_list()
                    model_accuracy = 0.0
                    if outlier_clusters:
                        model_accuracy = clustered_data_variant.filter(pl.col("liked") == 0).select(pl.col("cluster")).select(
                            pl.col("cluster").filter(pl.col("cluster").is_in(outlier_clusters)).count()/pl.col("cluster").count(),
                            pl.col("cluster").filter(pl.col("cluster").is_in(outlier_clusters)).count().alias("outliers"),
                            pl.col("cluster").count().alias("total"),
                            ).item(0, 0)
                    return model_accuracy

                scorer = LabelScorerUnsupervised(disliked_matching_score)

                standard_scaler = make_pipeline(StandardScaler())
                pca = make_pipeline(PCA())
                minmax_scaler = make_pipeline(MinMaxScaler())
                transformation_variants = [
                    [],
                    [*pca.steps],
                    [*standard_scaler.steps],
                    [*standard_scaler.steps, *pca.steps],
                    [*pca.steps, *standard_scaler.steps],
                    [*minmax_scaler.steps, *pca.steps],
                    [*pca.steps, *minmax_scaler.steps],
                    [*standard_scaler.steps, *minmax_scaler.steps, *pca.steps],
                    [*pca.steps, *standard_scaler.steps, *minmax_scaler.steps],
                    [*standard_scaler.steps, *minmax_scaler.steps],
                    [*minmax_scaler.steps],
                ]

                def make_search_pipelines(estimator_name, estimator, grid, scoring=scorer):
                    def create_optimizer():
                        return BayesSearchCV(
                            estimator=estimator,
                            search_spaces=grid,
                            n_iter=optimization_iterations,
                            scoring=scoring,
                            error_score=0.0,
                            cv=NoCV(),
                            verbose=1,
                        )
                    def build_last_step_name(variation):
                        return build_full_model_name(estimator_name, [step[0] for step in variation])

                    return [Pipeline([*variation, (build_last_step_name(variation), create_optimizer())]) for variation in transformation_variants]

                # logger.info(f"Starting try={try_n + 1}")
                pipelines = sum([
                    # make_search_pipelines(
                    #     "OneClassSVM",
                    #     OneClassSVM(),
                    #     {
                    #         "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    #         "degree": [2, 3, 4, 5],
                    #         "nu": np.arange(0.1, 1, 0.05),
                    #     },
                    #     scoring="accuracy",
                    # ),
                    make_search_pipelines(
                        "hdbscan",
                        HDBSCAN(allow_single_cluster=False, algorithm="brute"),
                        {
                            "min_cluster_size": np.arange(3, 60, 3),
                            "metric": ["euclidean", "cosine"],
                            "cluster_selection_method": ["eom", "leaf"],
                            "cluster_selection_epsilon": np.arange(0.0, 1, 0.1),
                            "leaf_size": np.arange(10, 100, 10),
                            "alpha": [0.5, 1.0, 2.0, 5.0],
                        },
                    ),
                    make_search_pipelines(
                        "DipMeans",
                        DipMeans(),
                        {
                            "significance": [0.001, 0.003, 0.0005],
                            "split_viewers_threshold": [0.01, 0.005, 0.02],
                            "n_boots": [500, 1000, 2000],
                            "n_split_trials": [5, 10, 20],
                            "n_clusters_init": [2, 5, 10],
                            "max_n_clusters": [50, 75, 100, 150, 200, np.inf],
                        },
                    ),
                    make_search_pipelines(
                        "GMeans",
                        GMeans(),
                        {
                            "significance": [0.00005, 0.0001, 0.0002, 0.001],
                            "n_clusters_init": [2, 5, 10],
                            "max_n_clusters": [15, 50, 100, 150, 200, np.inf],
                            "n_split_trials": [5, 10, 20],
                        },
                    ),
                    # make_search_pipelines(
                    #     "PGMeans",
                    #     PGMeans(),
                    #     {
                    #         "significance": [0.0005, 0.001, 0.002, 0.1],
                    #         "n_new_centers": [3, 5, 10],
                    #         "amount_random_centers": [0.25, 0.5, 0.75],
                    #         "n_clusters_init": [2, 5, 10],
                    #         "max_n_clusters": [15, 50, 100, 150, 200, np.inf],
                    #     },
                    # ),
                    make_search_pipelines(
                        "SpecialK",
                        SpecialK(),
                        {
                            "significance": [0.005, 0.01, 0.02, 0.1],
                            "n_dimensions": [100, 200, 300],
                            "similarity_matrix": ["NAM", "SAM"],
                            "n_neighbors": [3, 5, 10],
                            "percentage": [0.7, 0.9, 0.99],
                            "n_cluster_pairs_to_consider": [5, 10, 20, None],
                            "max_n_clusters": [10, 50, 100, 150, 200, np.inf],
                        },
                    ),
                    make_search_pipelines(
                        "XMeans",
                        XMeans(),
                        {
                            "allow_merging": [True, False],
                            "n_clusters_init": [2, 10],
                            "max_n_clusters": [50, 100, 200, 300, np.inf],
                            "n_split_trials": [10, 20, 30],
                        },
                    ),
                    make_search_pipelines(
                        "MultiDensityDBSCAN",
                        MultiDensityDBSCAN(),
                        {
                            "k": np.arange(5, 50, 5),
                            "var": [1.5, 2.5, 5],
                            "min_cluster_size": [3, 5, 10],
                        },
                    ),
                    make_search_pipelines(
                        "GapStatistic",
                        GapStatistic(use_principal_components = False),
                        {
                            "min_n_clusters": [2, 10],
                            "max_n_clusters": [15, 35, 100, 150, 200, 300],
                            "n_boots": [5, 10, 20],
                            "use_log": [True, False],
                        },
                    ),
                    make_search_pipelines(
                        "VaDE",
                        make_pipeline(MinMaxScaler(clip=True), VaDE()),
                        {
                            "vade__batch_size": [256, 512],
                            "vade__clustering_loss_weight": [0.5, 1.0, 2],
                            "vade__ssl_loss_weight": [0.5, 1.0, 2],
                            "vade__embedding_size": [5, 10, 20],
                        },
                    ),
                    make_search_pipelines(
                        "AutoNR",
                        AutoNR(),
                        {
                            "nrkmeans_repetitions": [10, 15, 30],
                            "outliers": [True, False],
                            "max_n_clusters": [10, 50, 100, 150, 200, None],
                            "mdl_for_noisespace": [True, False],
                            "similarity_threshold": [1e-6, 1e-5, 1e-4],
                        },
                    ),
                    make_search_pipelines(
                        "DDC",
                        DDC(),
                        {
                            "ratio": [0.05, 0.1, 0.3],
                            "batch_size": [128, 256, 512],
                            "pretrain_epochs": [50, 100, 200],
                            "embedding_size": [5, 10, 20, 50],
                        },
                    ),
                    make_search_pipelines(
                        "DeepECT",
                        DeepECT(),
                        {
                            "max_n_leaf_nodes": [10, 20, 50],
                            "batch_size": [128, 256, 512],
                            "pretrain_epochs": [50, 100, 200],
                            "clustering_epochs": [100, 150, 200],
                            "grow_interval": [1, 2, 4],
                            "pruning_threshold": [0.05, 0.1, 0.2],
                            "embedding_size": [5, 10, 20, 50],
                        },
                    ),
                    make_search_pipelines(
                        "DipDECK",
                        DipDECK(),
                        {
                            "n_clusters_init": [2, 10, 20, 35],
                            "dip_merge_threshold": [0.75, 0.9, 0.95],
                            "max_n_clusters": [100, 200, 300, np.inf],
                            "min_n_clusters": [2, 10, 30, 50, 100],
                            "batch_size": [128, 256, 512],
                            "pretrain_epochs": [50, 100, 200],
                            "clustering_epochs": [50, 100, 150, 200],
                            "embedding_size": [3, 5, 10, 20, 50],
                            "max_cluster_size_diff_factor": [1.5, 2, 3],
                        },
                    ),
                    make_search_pipelines(
                        "ProjectedDipMeans",
                        ProjectedDipMeans(),
                        {
                            "significance": [0.0005, 0.001, 0.002, 0.1],
                            "n_random_projections": [0, 5, 10],
                            "n_boots": [500, 1000, 2000],
                            "n_split_trials": [5, 10, 20],
                            "n_clusters_init": [2, 5, 10],
                            "max_n_clusters": [15, 50, 100, 150, 200, np.inf],
                        },
                    ),
                    make_search_pipelines(
                        "SkinnyDip",
                        SkinnyDip(),
                        {
                            "significance": [0.005, 0.05, 0.1],
                            "n_boots": [500, 1000, 2000],
                            "add_tails": [True, False],
                            "outliers": [True, False],
                            "max_cluster_size_diff_factor": [1.5, 2, 3, 4],
                        },
                    ),
                    make_search_pipelines(
                        "DipNSub",
                        DipNSub(),
                        {
                            "significance": [0.005, 0.01, 0.05, 0.1],
                            "threshold": [0.1, 0.15, 0.2],
                            "step_size": [0.05, 0.1, 0.2],
                            "momentum": [0.5, 0.75, 0.95, 1, 1.5],
                            "add_tails": [True, False],
                            "outliers": [True, False],
                        },
                    ),

                ], [])
                pipelines = filter_pipelines(pipelines, data_params)
                processed_models_count = 0
                total_models = len(pipelines)
                while pipelines and (pipeline := pipelines.pop(0)):
                    model_name = pipeline.steps[-1][0]
                    pipeline.fit(X_numpy)
                    processed_models_count +=1

                    search = pipeline.named_steps[model_name]
                    unique_clusters = 0
                    if hasattr(search.best_estimator_, "labels_"):
                        unique_clusters = len(numpy.unique(search.best_estimator_.labels_))
                        logger.info(
                            f"<{dn}/{len(data_param_options)}>{data_params} -> {model_name}: {unique_clusters=}"
                        )
                    logger.info(
                        f"<{dn}/{len(data_param_options)}>{data_params} -> {model_name}({processed_models_count}/{total_models})}} {search.best_score_:.3f} for params {search.best_params_} resulting {unique_clusters=}"
                    )
                    line = f"{build_model_report_prefix(data_params, model_name)}: {search.best_score_:.3f} -> {search.best_params_}[{unique_clusters=}]"

                    lines = []
                    data_report = report_root.joinpath(f"{get_base_model_name(model_name)}.csv")
                    if data_report.exists():
                        with data_report.open("rt") as f:
                            lines = f.read().splitlines(keepends=True)

                    if search.best_score_ >= interesting_clusterization_limit:
                        with tempfile.NamedTemporaryFile(mode='at', delete=False) as tmp_report:
                            tmp_report.write(f"{line}\n")
                            tmp_report.writelines(lines)
                        shutil.move(tmp_report.name, data_report)
                    else:
                        with data_report.open("at") as f:
                            f.write(f"{line}\n")
