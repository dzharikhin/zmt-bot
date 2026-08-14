import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import config
from audio.segments import SegmentSpec
from core.modeling import DualOneClassModel
from core.outliers import detect_outliers
from core.paths import get_embed_version
from core.preprocessing import NoOpPreprocessor, StandardizeSelectPreprocessor
from core.storage import FeatureStore
from core.writer import start_extraction_job

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def objective(trial, X_liked, X_disliked, w_a, w_b, make_preprocessor=None):
    """Optuna objective: weighted sum of exclude_disliked and include_liked AUC
    via k-fold CV."""
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
        trial.set_user_attr("auc_include", 0.5)
        trial.set_user_attr("auc_exclude", 0.5)
        trial.set_user_attr("disliked_false_accept", 0.5)
        trial.set_user_attr("liked_false_reject", 0.5)
        trial.set_user_attr("liked_recall", 0.5)
        trial.set_user_attr("disliked_recall", 0.5)
        return 0.0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    s_like_on_liked = []
    s_like_on_disliked = []
    s_dislike_on_disliked = []
    s_dislike_on_liked = []

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
            trial.set_user_attr("auc_include", 0.5)
            trial.set_user_attr("auc_exclude", 0.5)
            trial.set_user_attr("disliked_false_accept", 0.5)
            trial.set_user_attr("liked_false_reject", 0.5)
            trial.set_user_attr("liked_recall", 0.5)
            trial.set_user_attr("disliked_recall", 0.5)
            return 0.0

        prep = make_preprocessor() if make_preprocessor else NoOpPreprocessor()
        X_combined_tr = np.concatenate([X_l_tr_f, X_d_tr_f])
        y_combined_tr = np.concatenate(
            [np.ones(len(X_l_tr_f)), np.zeros(len(X_d_tr_f))]
        )
        prep.fit(X_combined_tr, y_combined_tr)

        X_l_tr_f = prep.transform(X_l_tr_f)
        X_d_tr_f = prep.transform(X_d_tr_f)
        X_l_te = prep.transform(X_l_te)
        X_d_te = prep.transform(X_d_te)

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
            trial.set_user_attr("auc_include", 0.5)
            trial.set_user_attr("auc_exclude", 0.5)
            trial.set_user_attr("disliked_false_accept", 0.5)
            trial.set_user_attr("liked_false_reject", 0.5)
            trial.set_user_attr("liked_recall", 0.5)
            trial.set_user_attr("disliked_recall", 0.5)
            return 0.0

        s_like_on_liked.extend(
            [model.liked_model.score(x.reshape(1, -1))["calibrated"] for x in X_l_te]
        )
        s_like_on_disliked.extend(
            [model.liked_model.score(x.reshape(1, -1))["calibrated"] for x in X_d_te]
        )
        s_dislike_on_disliked.extend(
            [model.dislike_model.score(x.reshape(1, -1))["calibrated"] for x in X_d_te]
        )
        s_dislike_on_liked.extend(
            [model.dislike_model.score(x.reshape(1, -1))["calibrated"] for x in X_l_te]
        )

    try:
        y_inc = [1] * len(s_like_on_liked) + [0] * len(s_like_on_disliked)
        auc_include = roc_auc_score(y_inc, s_like_on_liked + s_like_on_disliked)
    except ValueError:
        auc_include = 0.5

    try:
        y_exc = [1] * len(s_dislike_on_disliked) + [0] * len(s_dislike_on_liked)
        auc_exclude = roc_auc_score(y_exc, s_dislike_on_disliked + s_dislike_on_liked)
    except ValueError:
        auc_exclude = 0.5

    t_include = np.percentile(
        s_like_on_liked, 100 * (1 - config.model_include_liked_recall_target)
    )
    liked_recall = np.mean(np.array(s_like_on_liked) > t_include)
    disliked_false_accept = np.mean(np.array(s_like_on_disliked) > t_include)

    t_exclude = np.percentile(
        s_dislike_on_disliked, 100 * (1 - config.model_exclude_disliked_recall_target)
    )
    disliked_recall = np.mean(np.array(s_dislike_on_disliked) >= t_exclude)
    liked_false_reject = np.mean(np.array(s_dislike_on_liked) >= t_exclude)

    trial.set_user_attr("auc_include", float(auc_include))
    trial.set_user_attr("auc_exclude", float(auc_exclude))
    trial.set_user_attr("disliked_false_accept", float(disliked_false_accept))
    trial.set_user_attr("liked_false_reject", float(liked_false_reject))
    trial.set_user_attr("liked_recall", float(liked_recall))
    trial.set_user_attr("disliked_recall", float(disliked_recall))

    return float(w_a * auc_exclude + w_b * auc_include)


def optimize_embedding(
    X_liked, X_disliked, w_a, w_b, n_iterations=50, make_preprocessor=None
):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective(
            trial, X_liked, X_disliked, w_a, w_b, make_preprocessor
        ),
        n_trials=n_iterations,
        show_progress_bar=False,
    )

    best = study.best_trial

    top_n = min(5, len(study.trials))
    top_trials_no_none = [t for t in study.trials if t.value is not None]
    top_trials = sorted(top_trials_no_none, key=lambda t: t.value, reverse=True)[:top_n]
    median_metrics = {
        "auc_include": float(
            np.median([t.user_attrs.get("auc_include", 0.5) for t in top_trials])
        ),
        "auc_exclude": float(
            np.median([t.user_attrs.get("auc_exclude", 0.5) for t in top_trials])
        ),
        "disliked_false_accept": float(
            np.median(
                [t.user_attrs.get("disliked_false_accept", 0.5) for t in top_trials]
            )
        ),
        "liked_false_reject": float(
            np.median([t.user_attrs.get("liked_false_reject", 0.5) for t in top_trials])
        ),
        "liked_recall": float(
            np.median([t.user_attrs.get("liked_recall", 0.5) for t in top_trials])
        ),
        "disliked_recall": float(
            np.median([t.user_attrs.get("disliked_recall", 0.5) for t in top_trials])
        ),
    }

    return {
        "objective": float(best.value),
        "objective_top5_median": float(
            np.median([t.value for t in top_trials_no_none])
        ),
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
        "metrics_best": {
            "auc_include": float(best.user_attrs.get("auc_include", 0.5)),
            "auc_exclude": float(best.user_attrs.get("auc_exclude", 0.5)),
            "disliked_false_accept": float(
                best.user_attrs.get("disliked_false_accept", 0.5)
            ),
            "liked_false_reject": float(best.user_attrs.get("liked_false_reject", 0.5)),
            "liked_recall": float(best.user_attrs.get("liked_recall", 0.5)),
            "disliked_recall": float(best.user_attrs.get("disliked_recall", 0.5)),
        },
        "metrics_top5_median": median_metrics,
        "n_trials": n_iterations,
        "trial_history": [
            {
                "params": t.params,
                "value": t.value,
                "auc_include": t.user_attrs.get("auc_include", 0.5),
                "auc_exclude": t.user_attrs.get("auc_exclude", 0.5),
                "disliked_false_accept": t.user_attrs.get("disliked_false_accept", 0.5),
                "liked_false_reject": t.user_attrs.get("liked_false_reject", 0.5),
                "liked_recall": t.user_attrs.get("liked_recall", 0.5),
                "disliked_recall": t.user_attrs.get("disliked_recall", 0.5),
            }
            for t in study.trials
        ],
    }


def _best_result(results):
    """Best non-failed result by objective, or None if no valid results."""
    valid = [r for r in results if not r.get("failed") and "optimization" in r]
    if not valid:
        return None
    return max(valid, key=lambda r: r["optimization"]["objective"])


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
        "--output",
        type=str,
        required=True,
        help="Output JSON report path (default: report_v2.json for new metrics)",
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
        default=2,
        help="Extraction worker processes per variant",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Regex to filter variant names (partial match)",
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="standardize+select_64",
        help="Preprocessor to use (e.g., standardize+select_64, none)",
    )

    args = parser.parse_args()

    def make_preprocessor_factory(preprocessor_str: str):
        if preprocessor_str == "standardize+select_64":
            n = int(preprocessor_str.split("_")[-1])
            return lambda: StandardizeSelectPreprocessor(n_features=n)
        elif preprocessor_str == "none":
            return lambda: NoOpPreprocessor()
        else:
            raise ValueError(f"Unknown preprocessor: {preprocessor_str}")

    make_preprocessor = make_preprocessor_factory(args.preprocessor)

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

        use_panns = embedding_var.get("panns", True)
        panns_weights_path = Path(embedding_var["panns_weights"]) if use_panns else None

        sp_dict = embedding_var["segment_policy"]
        segment_spec = SegmentSpec(
            type=sp_dict["type"],
            window_s=sp_dict.get("window_s"),
            k=sp_dict.get("k"),
            aggregation=embedding_var.get("aggregation", "mean"),
        )

        variant_embed_version = get_embed_version(
            profile_path, panns_weights_path, use_panns=use_panns
        )
        variant_segment_policy = segment_spec.canonical()

        job_id = (
            f"bench_{user_id}_{name}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )

        t0 = time.monotonic()
        logger.info(
            f"Job {job_id}: Starting feature extraction for {len(all_tracks)} tracks"
        )
        try:
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
                use_panns=use_panns,
            )
            extraction_time_s = time.monotonic() - t0
            logger.info(
                f"Job {job_id}: Extraction completed in {extraction_time_s:.1f}s "
                f"(ok={extraction_result.ok}, fail={extraction_result.failed}, "
                f"cached={extraction_result.skipped})"
            )

            store = FeatureStore(user_id, variant_embed_version, variant_segment_policy)
            track_counts = store.count_tracks()
            logger.info(
                f"Job {job_id}: Feature counts — "
                f"liked: {track_counts.get('like', 0)}/{len(liked_tracks)}, "
                f"disliked: {track_counts.get('dislike', 0)}/{len(disliked_tracks)}"
            )
            with store.training_view("like") as liked_pq:
                X_liked = FeatureStore.load_vectors(liked_pq)
            with store.training_view("dislike") as disliked_pq:
                X_disliked = FeatureStore.load_vectors(disliked_pq)

            liked_ok = track_counts.get("like", 0)
            disliked_ok = track_counts.get("dislike", 0)

            extraction_stats = {
                "time_s": round(extraction_time_s, 3),
                "liked": {"total": len(liked_tracks), "ok": liked_ok},
                "disliked": {"total": len(disliked_tracks), "ok": disliked_ok},
                "cached": extraction_result.skipped,
                "newly_extracted": extraction_result.ok,
            }

            if liked_ok < 10 or disliked_ok < 10:
                logger.warning(
                    f"Skipping — insufficient data ({liked_ok} liked, {disliked_ok} "
                    f"disliked)"
                )
                results.append(
                    {
                        "name": name,
                        "embedding": variant_embed_version,
                        "segment_policy": variant_segment_policy,
                        "failed": True,
                        "error": "insufficient data",
                        "extraction": extraction_stats,
                    }
                )
            else:
                opt_result = optimize_embedding(
                    X_liked,
                    X_disliked,
                    w_a,
                    w_b,
                    n_iterations=args.n_iterations,
                    make_preprocessor=make_preprocessor,
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
                    f"Feature counts — "
                    f"Liked: {track_counts.get('like', 0)}/{len(liked_tracks)}, "
                    f"Disliked: {track_counts.get('dislike', 0)}/{len(disliked_tracks)}"
                )

                opt = opt_result
                metrics_best = opt["metrics_best"]

                print(f"  Best params: {opt['best_params']}")

                verdict_include = (
                    "strong"
                    if metrics_best["auc_include"] >= 0.90
                    and metrics_best["disliked_false_accept"] < 0.20
                    else (
                        "decent"
                        if metrics_best["auc_include"] >= 0.80
                        and metrics_best["disliked_false_accept"] < 0.40
                        else "weak" if metrics_best["auc_include"] >= 0.70 else "broken"
                    )
                )
                print(
                    f"  includeLiked: AUC={metrics_best['auc_include']:.3f}, "
                    f"false_accept={metrics_best['disliked_false_accept']:.2f}, "
                    f"recall={metrics_best['liked_recall']:.2f} [{verdict_include}]"
                )

                verdict_exclude = (
                    "strong"
                    if metrics_best["auc_exclude"] >= 0.90
                    and metrics_best["liked_false_reject"] < 0.15
                    else (
                        "decent"
                        if metrics_best["auc_exclude"] >= 0.80
                        and metrics_best["liked_false_reject"] < 0.35
                        else "weak" if metrics_best["auc_exclude"] >= 0.70 else "broken"
                    )
                )
                print(
                    f"  excludeDisliked: AUC={metrics_best['auc_exclude']:.3f}, "
                    f"false_reject={metrics_best['liked_false_reject']:.2f}, "
                    f"recall={metrics_best['disliked_recall']:.2f} [{verdict_exclude}]"
                )
        except Exception as e:
            logger.error(f"Variant {name} failed: {e}", exc_info=True)
            results.append({"name": name, "failed": True, "error": str(e)})

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "objective_weights": {"auc_exclude": w_a, "auc_include": w_b},
            "threshold_regime": {
                "exclude_disliked_recall_target": (
                    config.model_exclude_disliked_recall_target
                ),
                "include_liked_recall_target": (
                    config.model_include_liked_recall_target
                ),
                "cv_folds_in_benchmark": None,
            },
            "n_iterations": args.n_iterations,
            "results": results,
            "best_variant": _best_result(results),
        }

        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {args.output}")

    bv = _best_result(results)
    if bv is not None:
        logger.info(f"Best variant: {bv['name']}")
        logger.info(f"   Segment policy: {bv['segment_policy']}")
        logger.info(f"   Objective: {bv['optimization']['objective']:.3f}")
        logger.info(f"   Best params: {bv['optimization']['best_params']}")
        ext = bv["extraction"]
        logger.info(
            f"   Extraction time: {ext['time_s']:.1f}s "
            f"(liked: {ext['liked']['ok']}/{ext['liked']['total']}, "
            f"disliked: {ext['disliked']['ok']}/{ext['disliked']['total']})"
        )


if __name__ == "__main__":
    main()
