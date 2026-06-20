import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import yaml
from sklearn.model_selection import KFold

import config
from audio.segments import SegmentSpec
from core.modeling import DualOneClassModel
from core.outliers import detect_outliers
from core.paths import get_embed_version
from core.storage import FeatureStore
from core.writer import start_extraction_job

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def objective(trial, X_liked, X_disliked, w_a, w_b):
    """Optuna objective: weighted sum of exclude_disliked and include_liked recall via k-fold CV."""
    knn_k_min = trial.suggest_int("knn_k_min", 3, 8)
    knn_k_max = trial.suggest_int("knn_k_max", 8, 25)
    if knn_k_max < knn_k_min:
        raise optuna.exceptions.TrialPruned("knn_k_max < knn_k_min")

    knn_k_scale = trial.suggest_float("knn_k_scale", 0.3, 1.0)
    gmm_components_max = trial.suggest_int("gmm_components_max", 8, 32)
    gmm_min_points_per_component = trial.suggest_int(
        "gmm_min_points_per_component", 20, 80
    )
    outlier_threshold = trial.suggest_float("outlier_threshold", 0.01, 0.10)

    n_splits = min(5, len(X_liked), len(X_disliked))
    if n_splits < 2:
        trial.set_user_attr("exclude_disliked_recall", 0.0)
        trial.set_user_attr("include_liked_recall", 0.0)
        return 0.0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    exclude_disliked_recalls, include_liked_recalls = [], []

    folds_liked = list(kf.split(X_liked))
    folds_disliked = list(kf.split(X_disliked))

    for (l_train, l_test), (d_train, d_test) in zip(folds_liked, folds_disliked):
        X_l_tr, X_l_te = X_liked[l_train], X_liked[l_test]
        X_d_tr, X_d_te = X_disliked[d_train], X_disliked[d_test]

        mask_l, _ = detect_outliers(
            X_l_tr,
            threshold=outlier_threshold,
            knn_k=knn_k_max,
            min_set_size=config.model_min_set_size,
        )
        mask_d, _ = detect_outliers(
            X_d_tr,
            threshold=outlier_threshold,
            knn_k=knn_k_max,
            min_set_size=config.model_min_set_size,
        )
        X_l_tr_f, X_d_tr_f = X_l_tr[mask_l], X_d_tr[mask_d]

        if len(X_l_tr_f) < 2 or len(X_d_tr_f) < 2:
            trial.set_user_attr("exclude_disliked_recall", 0.0)
            trial.set_user_attr("include_liked_recall", 0.0)
            return 0.0

        try:
            model = DualOneClassModel(
                knn_k_min=knn_k_min,
                knn_k_max=knn_k_max,
                knn_k_scale=knn_k_scale,
                gmm_components_max=gmm_components_max,
                gmm_min_points_per_component=gmm_min_points_per_component,
                cv_folds=None,
                exclude_disliked_recall_target=config.model_exclude_disliked_recall_target,
                include_liked_recall_target=config.model_include_liked_recall_target,
            )
            model.fit(X_l_tr_f, X_d_tr_f)
        except ValueError as e:
            logger.warning(
                f"Trial failed with ValueError: {e}. "
                f"Params: knn_k_min={knn_k_min}, knn_k_max={knn_k_max}, "
                f"knn_k_scale={knn_k_scale}, gmm_components_max={gmm_components_max}, "
                f"gmm_min_points_per_component={gmm_min_points_per_component}, "
                f"outlier_threshold={outlier_threshold}"
            )
            trial.set_user_attr("exclude_disliked_recall", 0.0)
            trial.set_user_attr("include_liked_recall", 0.0)
            return 0.0

        try:
            model = DualOneClassModel(
                knn_k_min=knn_k_min,
                knn_k_max=knn_k_max,
                knn_k_scale=knn_k_scale,
                gmm_components_max=gmm_components_max,
                gmm_min_points_per_component=gmm_min_points_per_component,
                cv_folds=None,
                exclude_disliked_recall_target=config.model_exclude_disliked_recall_target,
                include_liked_recall_target=config.model_include_liked_recall_target,
            )
            model.fit(X_l_tr_f, X_d_tr_f)
        except ValueError as e:
            logger.warning(
                f"Trial failed with ValueError: {e}. "
                f"Params: knn_k_min={knn_k_min}, knn_k_max={knn_k_max}, "
                f"knn_k_scale={knn_k_scale}, gmm_components_max={gmm_components_max}, "
                f"gmm_min_points_per_component={gmm_min_points_per_component}, "
                f"outlier_threshold={outlier_threshold}"
            )
            trial.set_user_attr("exclude_disliked_recall", 0.0)
            trial.set_user_attr("include_liked_recall", 0.0)
            return 0.0

        d_scores = [
            model.dislike_model.score(x.reshape(1, -1))["calibrated"] for x in X_d_te
        ]
        exclude_disliked_recalls.append(
            np.mean([s < model.thresholds["exclude_disliked"] for s in d_scores])
        )

        l_scores = [
            model.liked_model.score(x.reshape(1, -1))["calibrated"] for x in X_l_te
        ]
        include_liked_recalls.append(
            np.mean([s > model.thresholds["include_liked"] for s in l_scores])
        )

    exclude_disliked = (
        float(np.mean(exclude_disliked_recalls)) if exclude_disliked_recalls else 0.0
    )
    include_liked = (
        float(np.mean(include_liked_recalls)) if include_liked_recalls else 0.0
    )
    trial.set_user_attr("exclude_disliked_recall", exclude_disliked)
    trial.set_user_attr("include_liked_recall", include_liked)
    weighted = w_a * exclude_disliked + w_b * include_liked
    return weighted


def optimize_embedding(X_liked, X_disliked, w_a, w_b, n_iterations=50):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective(trial, X_liked, X_disliked, w_a, w_b),
        n_trials=n_iterations,
        show_progress_bar=True,
    )

    best = study.best_trial
    return {
        "best_params": {
            "knn_k_min": int(best.params["knn_k_min"]),
            "knn_k_max": int(best.params["knn_k_max"]),
            "knn_k_scale": float(best.params["knn_k_scale"]),
            "gmm_components_max": int(best.params["gmm_components_max"]),
            "gmm_min_points_per_component": int(
                best.params["gmm_min_points_per_component"]
            ),
            "outlier_threshold": float(best.params["outlier_threshold"]),
        },
        "weighted_recall": float(best.value),
        "exclude_disliked_recall": float(best.user_attrs["exclude_disliked_recall"]),
        "include_liked_recall": float(best.user_attrs["include_liked_recall"]),
        "n_trials": n_iterations,
        "trial_history": [
            {
                "params": t.params,
                "value": t.value,
                "exclude_disliked_recall": t.user_attrs.get("exclude_disliked_recall"),
                "include_liked_recall": t.user_attrs.get("include_liked_recall"),
            }
            for t in study.trials
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare embedding variants")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to embedding variants YAML config",
    )
    parser.add_argument(
        "--objective-weights",
        type=float,
        nargs=2,
        default=[0.5, 0.5],
        help="Weights for exclude_disliked and include_liked recall",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output JSON report path"
    )
    parser.add_argument(
        "--user-id", type=int, required=True, help="Telegram user ID for DuckDB access"
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=50,
        help="Number of Optuna trials per variant",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Extraction worker processes per variant",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Regex to filter variant names (partial match)",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config_data = yaml.safe_load(f)

    w_a, w_b = args.objective_weights
    user_id = args.user_id

    existing_report = {}
    if Path(args.output).exists():
        with open(args.output) as f:
            existing_report = json.load(f)

    processed_names = {r["name"] for r in existing_report.get("results", [])}

    liked_path = config.get_liked_file_store_path(user_id)
    disliked_path = config.get_disliked_file_store_path(user_id)
    liked_tracks = list(liked_path.glob("*.mp3"))
    disliked_tracks = list(disliked_path.glob("*.mp3"))
    all_tracks = [(t, "like") for t in liked_tracks] + [
        (t, "dislike") for t in disliked_tracks
    ]

    results = existing_report.get("results", [])

    for embedding_var in config_data["embeddings"]:
        name = embedding_var["name"]
        if args.variants and args.variants not in name:
            continue

        if name in processed_names:
            logger.info(f"Skipping: {name} (already in report)")
            continue

        logger.info(f"Processing: {name}")

        essentia_profile = embedding_var["essentia_profile"]
        profile_path = Path(essentia_profile)
        if not profile_path.is_absolute():
            profile_path = config.data_path / "benchmark" / essentia_profile

        panns_weights_path = Path(embedding_var["panns_weights"])

        sp_dict = embedding_var["segment_policy"]
        segment_spec = SegmentSpec(
            type=sp_dict["type"],
            window_s=sp_dict.get("window_s"),
            k=sp_dict.get("k"),
            aggregation=embedding_var.get("aggregation", "mean"),
        )

        variant_embed_version = get_embed_version(profile_path, panns_weights_path)
        variant_segment_policy = segment_spec.canonical()

        job_id = (
            f"bench_{user_id}_{name}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )

        t0 = time.monotonic()
        logger.info(
            f"Job {job_id}: Starting feature extraction for {len(all_tracks)} tracks"
        )
        extraction_result = start_extraction_job(
            user_id=user_id,
            tracks=all_tracks,
            embed_version=variant_embed_version,
            segment_policy=variant_segment_policy,
            job_id=job_id,
            n_workers=args.n_workers,
            panns_weights_path=panns_weights_path,
            profile_path=profile_path,
            segment_spec=segment_spec,
        )
        extraction_time_s = time.monotonic() - t0
        logger.info(
            f"Job {job_id}: Extraction completed in {extraction_time_s:.1f}s "
            f"({extraction_result.ok} succeeded, {extraction_result.failed} failed, "
            f"{extraction_result.skipped} cached)"
        )

        store = FeatureStore(user_id, variant_embed_version, variant_segment_policy)
        track_counts = store.count_tracks()
        logger.info(
            f"Job {job_id}: Feature counts — liked: {track_counts.get('like', 0)} ok / "
            f"{len(liked_tracks)} total, disliked: {track_counts.get('dislike', 0)} "
            f"ok / {len(disliked_tracks)} total"
        )
        with store.training_view("like") as liked_pq:
            X_liked = FeatureStore.load_vectors(liked_pq)
        with store.training_view("dislike") as disliked_pq:
            X_disliked = FeatureStore.load_vectors(disliked_pq)

        liked_ok = track_counts.get("like", 0)
        disliked_ok = track_counts.get("dislike", 0)

        extraction_stats = {
            "time_s": round(extraction_time_s, 3),
            "liked": {
                "total": len(liked_tracks),
                "ok": liked_ok,
            },
            "disliked": {
                "total": len(disliked_tracks),
                "ok": disliked_ok,
            },
            "cached": extraction_result.skipped,
            "newly_extracted": extraction_result.ok,
        }

        if liked_ok < 10 or disliked_ok < 10:
            logger.warning(
                f"Skipping — insufficient data ({liked_ok} liked, {disliked_ok} "
                f"disliked)"
            )
            continue

        opt_result = optimize_embedding(
            X_liked,
            X_disliked,
            w_a,
            w_b,
            n_iterations=args.n_iterations,
        )

        results.append(
            {
                "name": name,
                "embedding": variant_embed_version,
                "segment_policy": variant_segment_policy,
                "config": embedding_var,
                "feature_dim": X_liked.shape[1],
                "extraction": extraction_stats,
                "optimization": opt_result,
            }
        )

        print(
            f"  Extraction: {extraction_time_s:.1f}s "
            f"({extraction_result.ok} new, {extraction_result.skipped} cached, "
            f"{extraction_result.failed} failed)"
        )
        logger.info(
            f"Feature counts — Liked: {track_counts.get('like', 0)} ok / "
            f"{len(liked_tracks)} total, Disliked: {track_counts.get('dislike', 0)} "
            f"ok / {len(disliked_tracks)} total"
        )
        print(f"  Best weighted recall: {opt_result['weighted_recall']:.3f}")
        print(f"  Best params: {opt_result['best_params']}")

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "objective_weights": {"exclude_disliked": w_a, "include_liked": w_b},
            "threshold_regime": {
                "exclude_disliked_recall_target": config.model_exclude_disliked_recall_target,
                "include_liked_recall_target": config.model_include_liked_recall_target,
                "cv_folds_in_benchmark": None,
            },
            "n_iterations": args.n_iterations,
            "results": results,
            "best_variant": (
                max(results, key=lambda r: r["optimization"]["weighted_recall"])
                if results
                else None
            ),
        }

        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {args.output}")

    if results:
        bv = max(results, key=lambda r: r["optimization"]["weighted_recall"])
        logger.info(f"Best variant: {bv['name']}")
        logger.info(f"   Segment policy: {bv['segment_policy']}")
        logger.info(f"   Weighted recall: {bv['optimization']['weighted_recall']:.3f}")
        logger.info(f"   Best params: {bv['optimization']['best_params']}")
        ext = bv["extraction"]
        logger.info(
            f"   Extraction time: {ext['time_s']:.1f}s "
            f"(liked: {ext['liked']['ok']}/{ext['liked']['total']}, "
            f"disliked: {ext['disliked']['ok']}/{ext['disliked']['total']})"
        )


if __name__ == "__main__":
    main()
