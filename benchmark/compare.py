import argparse
import json
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


def objective(trial, X_liked, X_disliked, w_a, w_b):
    """Optuna objective: weighted sum of mode_a and mode_b recall via k-fold CV."""
    knn_k = trial.suggest_int("knn_k", 3, 15)
    gmm_components = trial.suggest_int("gmm_components", 8, 32)
    outlier_threshold = trial.suggest_float("outlier_threshold", 0.01, 0.10)

    n_splits = min(5, len(X_liked), len(X_disliked))
    if n_splits < 2:
        trial.set_user_attr("mode_a_recall", 0.0)
        trial.set_user_attr("mode_b_recall", 0.0)
        return 0.0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mode_a_recalls, mode_b_recalls = [], []

    folds_liked = list(kf.split(X_liked))
    folds_disliked = list(kf.split(X_disliked))

    for (l_train, l_test), (d_train, d_test) in zip(folds_liked, folds_disliked):
        X_l_tr, X_l_te = X_liked[l_train], X_liked[l_test]
        X_d_tr, X_d_te = X_disliked[d_train], X_disliked[d_test]

        mask_l, _ = detect_outliers(
            X_l_tr,
            threshold=outlier_threshold,
            knn_k=knn_k,
            min_set_size=config.model_min_set_size,
        )
        mask_d, _ = detect_outliers(
            X_d_tr,
            threshold=outlier_threshold,
            knn_k=knn_k,
            min_set_size=config.model_min_set_size,
        )
        X_l_tr_f, X_d_tr_f = X_l_tr[mask_l], X_d_tr[mask_d]

        if len(X_l_tr_f) < 2 or len(X_d_tr_f) < 2:
            trial.set_user_attr("mode_a_recall", 0.0)
            trial.set_user_attr("mode_b_recall", 0.0)
            return 0.0

        model = DualOneClassModel(knn_k=knn_k, gmm_components=gmm_components)
        model.fit(X_l_tr_f, X_d_tr_f)

        d_scores = [
            model.dislike_model.score(x.reshape(1, -1))["calibrated"] for x in X_d_te
        ]
        mode_a_recalls.append(
            np.mean([s < model.thresholds["mode_a"] for s in d_scores])
        )

        l_scores = [
            model.liked_model.score(x.reshape(1, -1))["calibrated"] for x in X_l_te
        ]
        mode_b_recalls.append(
            np.mean([s > model.thresholds["mode_b"] for s in l_scores])
        )

    mode_a = float(np.mean(mode_a_recalls)) if mode_a_recalls else 0.0
    mode_b = float(np.mean(mode_b_recalls)) if mode_b_recalls else 0.0
    weighted = w_a * mode_a + w_b * mode_b
    trial.set_user_attr("mode_a_recall", mode_a)
    trial.set_user_attr("mode_b_recall", mode_b)
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
            "knn_k": int(best.params["knn_k"]),
            "gmm_components": int(best.params["gmm_components"]),
            "outlier_threshold": float(best.params["outlier_threshold"]),
        },
        "weighted_recall": float(best.value),
        "mode_a_recall": float(best.user_attrs["mode_a_recall"]),
        "mode_b_recall": float(best.user_attrs["mode_b_recall"]),
        "n_trials": n_iterations,
        "trial_history": [
            {
                "params": t.params,
                "value": t.value,
                "mode_a_recall": t.user_attrs.get("mode_a_recall"),
                "mode_b_recall": t.user_attrs.get("mode_b_recall"),
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
        help="Weights for mode_a and mode_b recall",
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

    liked_path = config.get_liked_file_store_path(user_id)
    disliked_path = config.get_disliked_file_store_path(user_id)
    liked_tracks = list(liked_path.glob("*.mp3"))
    disliked_tracks = list(disliked_path.glob("*.mp3"))
    all_tracks = [(t, "like") for t in liked_tracks] + [
        (t, "dislike") for t in disliked_tracks
    ]

    results = []

    for embedding_var in config_data["embeddings"]:
        name = embedding_var["name"]
        if args.variants and args.variants not in name:
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")

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

        job_id = f"bench_{user_id}_{name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        t0 = time.monotonic()
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

        store = FeatureStore(user_id, variant_embed_version, variant_segment_policy)
        track_counts = store.count_tracks()
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

        if len(X_liked) < 10 or len(X_disliked) < 10:
            print(
                f"  Skipping — insufficient data ({len(X_liked)} liked, {len(X_disliked)} disliked)"
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
        print(f"  Liked: {track_counts.get('like', 0)} ok / {len(liked_tracks)} total")
        print(
            f"  Disliked: {track_counts.get('dislike', 0)} ok / {len(disliked_tracks)} total"
        )
        print(f"  Best weighted recall: {opt_result['weighted_recall']:.3f}")
        print(f"  Best params: {opt_result['best_params']}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "objective_weights": {"mode_a": w_a, "mode_b": w_b},
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

    print(f"\nReport saved to {args.output}")

    if report["best_variant"]:
        bv = report["best_variant"]
        print(f"\nBest variant: {bv['name']}")
        print(f"   Segment policy: {bv['segment_policy']}")
        print(f"   Weighted recall: {bv['optimization']['weighted_recall']:.3f}")
        print(f"   Best params: {bv['optimization']['best_params']}")
        ext = bv["extraction"]
        print(
            f"   Extraction time: {ext['time_s']:.1f}s "
            f"(liked: {ext['liked']['ok']}/{ext['liked']['total']}, "
            f"disliked: {ext['disliked']['ok']}/{ext['disliked']['total']})"
        )


if __name__ == "__main__":
    main()
