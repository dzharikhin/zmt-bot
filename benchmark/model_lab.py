import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import config
from audio.features import _DESCRIPTOR_SCHEMA
from core.modeling import DualOneClassModel
from core.outliers import detect_outliers
from core.preprocessing import NoOpPreprocessor, StandardizeSelectPreprocessor
from core.storage import FeatureStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def build_dim_names(panns_dim: int) -> list[str]:
    """Build a list mapping each flat vector index to a descriptor name."""
    names = []
    for name, length, _ in _DESCRIPTOR_SCHEMA:
        names.extend([name] * length)
    # PANNs dims (derived from embed_version, not hardcoded)
    names.extend(["panns_backbone"] * panns_dim)
    return names


def get_essentia_dim() -> int:
    """Compute Essentia block dimension from _DESCRIPTOR_SCHEMA."""
    return sum(length for _, length, _ in _DESCRIPTOR_SCHEMA)


class StandardizePreprocessor:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean_ = X.mean(axis=0)
        self.std_ = np.where(X.std(axis=0) < 1e-9, 1.0, X.std(axis=0))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


class BlockStandardizePreprocessor:
    def __init__(self, essentia_dim: int):
        self.essentia_dim = essentia_dim

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        E, P = X[:, : self.essentia_dim], X[:, self.essentia_dim :]
        self.e_mean = E.mean(axis=0)
        self.e_std = np.where(E.std(axis=0) < 1e-9, 1.0, E.std(axis=0))
        self.p_mean = P.mean(axis=0)
        self.p_std = np.where(P.std(axis=0) < 1e-9, 1.0, P.std(axis=0))
        e_norms = np.linalg.norm((E - self.e_mean) / self.e_std, axis=1, keepdims=True)
        p_norms = np.linalg.norm((P - self.p_mean) / self.p_std, axis=1, keepdims=True)
        self.e_scale = float(np.median(e_norms)) or 1.0
        self.p_scale = float(np.median(p_norms)) or 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        E = (X[:, : self.essentia_dim] - self.e_mean) / self.e_std / self.e_scale
        P = (X[:, self.essentia_dim :] - self.p_mean) / self.p_std / self.p_scale
        return np.concatenate([E, P], axis=1)


class StandardizePCAPreprocessor:
    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean_ = X.mean(axis=0)
        self.std_ = np.where(X.std(axis=0) < 1e-9, 1.0, X.std(axis=0))
        Xs = (X - self.mean_) / self.std_
        U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
        k = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:k]
        self.explained_var_ = (S[:k] ** 2) / (len(X) - 1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean_) / self.std_
        # Whitening: divide by sqrt(explained variance) so all PCs have unit
        # variance. This prevents high-variance PCs from re-dominating Euclidean
        # distance after standardize.
        return (Xs @ self.components_.T) / np.sqrt(self.explained_var_)


PREPROCESSOR_CONFIGS = [
    {"name": "none"},
    {"name": "standardize"},
    {"name": "block_standardize"},
    {"name": "standardize+pca_32", "type": "pca", "k": 32},
    {"name": "standardize+pca_64", "type": "pca", "k": 64},
    {"name": "standardize+pca_128", "type": "pca", "k": 128},
    {"name": "standardize+pca_256", "type": "pca", "k": 256},
    {"name": "standardize+select_32", "type": "select", "k": 32},
    {"name": "standardize+select_64", "type": "select", "k": 64},
    {"name": "standardize+select_128", "type": "select", "k": 128},
    {"name": "standardize+select_256", "type": "select", "k": 256},
]


def make_preprocessor_factory(cfg: dict[str, Any], essentia_dim: int) -> callable:
    """Return a zero-arg factory that creates a fresh preprocessor instance."""
    name = cfg["name"]

    if name == "none":
        return lambda: NoOpPreprocessor()
    elif name == "standardize":
        return lambda: StandardizePreprocessor()
    elif name == "block_standardize":
        return lambda: BlockStandardizePreprocessor(essentia_dim)
    elif cfg.get("type") == "pca":
        k = cfg["k"]
        return lambda: StandardizePCAPreprocessor(k)
    elif cfg.get("type") == "select":
        k = cfg["k"]
        return lambda: StandardizeSelectPreprocessor(k)
    else:
        raise ValueError(f"Unknown preprocessor config: {cfg}")


def compute_diagnostics(
    X_liked: np.ndarray,
    X_disliked: np.ndarray,
    essentia_dim: int,
    dim_names: list[str],
) -> dict[str, Any]:
    """Compute Phase A diagnostics on X_all = concat(liked, disliked)."""
    X_all = np.concatenate([X_liked, X_disliked])
    per_dim_std = X_all.std(axis=0)

    top20_indices = np.argsort(per_dim_std)[::-1][:20]

    def dim_name(idx: int) -> str:
        if 0 <= idx < len(dim_names):
            return dim_names[idx]
        return f"unknown_{idx}"

    return {
        "std_max": float(per_dim_std.max()),
        "std_median": float(np.median(per_dim_std)),
        "std_max_over_median": float(
            per_dim_std.max() / (np.median(per_dim_std) + 1e-9)
        ),
        "top20_high_variance_dims": [
            {"dim_idx": int(i), "std": float(per_dim_std[i]), "name": dim_name(i)}
            for i in top20_indices
        ],
        "essentia_block_median_l2": float(
            np.median(np.linalg.norm(X_all[:, :essentia_dim], axis=1))
        ),
        "panns_block_median_l2": float(
            np.median(np.linalg.norm(X_all[:, essentia_dim:], axis=1))
        ),
        "top_dims_variance_share": {
            "top10": float(
                np.sort(per_dim_std**2)[::-1][:10].sum() / (per_dim_std**2).sum()
            ),
            "top50": float(
                np.sort(per_dim_std**2)[::-1][:50].sum() / (per_dim_std**2).sum()
            ),
            "top100": float(
                np.sort(per_dim_std**2)[::-1][:100].sum() / (per_dim_std**2).sum()
            ),
        },
    }


def objective_with_preprocessor(
    trial: optuna.Trial,
    X_liked: np.ndarray,
    X_disliked: np.ndarray,
    w_a: float,
    w_b: float,
    make_preprocessor: callable,
) -> float:
    """Optuna objective: weighted sum of exclude_disliked and include_liked AUC
    via k-fold CV, with preprocessor fit on train partition only."""
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

        # Fit preprocessor on combined training set (leak-safe)
        prep = make_preprocessor()
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
                exclude_disliked_recall_target=(
                    config.model_exclude_disliked_recall_target
                ),
                include_liked_recall_target=(config.model_include_liked_recall_target),
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


def optimize_with_preprocessor(
    X_liked: np.ndarray,
    X_disliked: np.ndarray,
    w_a: float,
    w_b: float,
    n_iterations: int,
    make_preprocessor: callable,
) -> dict[str, Any]:
    """Run Optuna optimization and return metrics."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective_with_preprocessor(
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


def run_cell(
    segment_policy: str,
    preproc_cfg: dict[str, Any],
    X_liked: np.ndarray,
    X_disliked: np.ndarray,
    w_a: float,
    w_b: float,
    n_iterations: int,
    essentia_dim: int,
    dim_names: list[str],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Run one (segment_policy, preprocessor) cell and return result dict."""
    cell_name = f"{segment_policy}__{preproc_cfg['name']}"

    logger.info(f"Running cell: {cell_name}")

    # Compute feature_dim_out by dry-running the preprocessor
    prep_factory = make_preprocessor_factory(preproc_cfg, essentia_dim)
    prep = prep_factory()
    X_combined = np.concatenate([X_liked, X_disliked])
    y_combined = np.concatenate([np.ones(len(X_liked)), np.zeros(len(X_disliked))])
    prep.fit(X_combined, y_combined)
    X_transformed = prep.transform(X_combined)
    feature_dim_out = X_transformed.shape[1]

    t0 = time.monotonic()
    opt_result = optimize_with_preprocessor(
        X_liked, X_disliked, w_a, w_b, n_iterations, prep_factory
    )
    elapsed_s = time.monotonic() - t0

    opt_metrics = opt_result["metrics_best"]
    logger.info(
        f"  {cell_name}: AUC_inc={opt_metrics['auc_include']:.3f} "
        f"AUC_exc={opt_metrics['auc_exclude']:.3f} "
        f"obj={opt_result['objective']:.3f} "
        f"({elapsed_s:.1f}s)"
    )

    return {
        "name": cell_name,
        "segment_policy": segment_policy,
        "preprocessor": preproc_cfg["name"],
        "feature_dim_in": X_liked.shape[1],
        "feature_dim_out": feature_dim_out,
        "diagnostics": diagnostics,
        "optimization": opt_result,
    }


def main():
    parser = argparse.ArgumentParser(description="Model Lab: preprocessing sweep")
    parser.add_argument("--user-id", type=int, required=True, help="Telegram user ID")
    parser.add_argument(
        "--embed-version",
        type=str,
        required=True,
        help="Embed version string (must have cached features)",
    )
    parser.add_argument(
        "--segment-policies",
        type=str,
        nargs="+",
        required=True,
        help="Canonical segment policy strings (space-separated)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON path (cells are skipped by name if already present)",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=50,
        help="Optuna trials per cell",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Substring filter on cell names",
    )
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Run Phase A diagnostics only, skip Phase B sweep",
    )
    parser.add_argument(
        "--objective-weights",
        type=float,
        nargs=2,
        default=[0.5, 0.5],
        help="Weights for exclude_disliked and include_liked AUC",
    )

    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    w_a, w_b = args.objective_weights

    existing_report = {}
    if Path(args.output).exists():
        with open(args.output) as f:
            existing_report = json.load(f)

    processed_names = {r["name"] for r in existing_report.get("results", [])}

    essentia_dim = get_essentia_dim()

    results = []
    all_diagnostics = {}

    for segment_policy in args.segment_policies:
        logger.info(f"Loading features for segment_policy: {segment_policy}")
        store = FeatureStore(args.user_id, args.embed_version, segment_policy)
        with store.training_view("like") as p:
            X_liked = FeatureStore.load_vectors(p)
        with store.training_view("dislike") as p:
            X_disliked = FeatureStore.load_vectors(p)

        logger.info(
            f"  Loaded: {len(X_liked)} liked, {len(X_disliked)} disliked, "
            f"{X_liked.shape[1]} dims"
        )

        if len(X_liked) < 10 or len(X_disliked) < 10:
            logger.warning(
                f"Skipping segment_policy {segment_policy}: insufficient data "
                f"({len(X_liked)} liked, {len(X_disliked)} disliked)"
            )
            continue

        # Compute PANNs dim and build _DIM_NAMES
        panns_dim = X_liked.shape[1] - essentia_dim
        dim_names = build_dim_names(panns_dim)

        # Phase A: diagnostics
        diagnostics = compute_diagnostics(X_liked, X_disliked, essentia_dim, dim_names)
        all_diagnostics[segment_policy] = diagnostics

        logger.info(
            f"  Diagnostics: std_max={diagnostics['std_max']:.3f}, "
            f"std_median={diagnostics['std_median']:.3f}, "
            f"std_max_over_median={diagnostics['std_max_over_median']:.3f}, "
            f"essentia_median_l2={diagnostics['essentia_block_median_l2']:.3f}, "
            f"panns_median_l2={diagnostics['panns_block_median_l2']:.3f}"
        )

        if args.diagnose_only:
            continue

        # Phase B: sweep
        for preproc_cfg in PREPROCESSOR_CONFIGS:
            cell_name = f"{segment_policy}__{preproc_cfg['name']}"

            if args.variants and args.variants not in cell_name:
                logger.info(f"Skipping: {cell_name} (filtered by --variants)")
                continue

            if cell_name in processed_names:
                logger.info(f"Skipping: {cell_name} (already in report)")
                continue

            result = run_cell(
                segment_policy=segment_policy,
                preproc_cfg=preproc_cfg,
                X_liked=X_liked,
                X_disliked=X_disliked,
                w_a=w_a,
                w_b=w_b,
                n_iterations=args.n_iterations,
                essentia_dim=essentia_dim,
                dim_names=dim_names,
                diagnostics=diagnostics,
            )
            results.append(result)

            # Write incremental checkpoint
            checkpoint_report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "embed_version": args.embed_version,
                "n_iterations": args.n_iterations,
                "results": results,
                "best_cell": (
                    max(results, key=lambda r: r["optimization"]["objective"])
                    if results
                    else None
                ),
            }
            with open(args.output, "w") as f:
                json.dump(checkpoint_report, f, indent=2)

    # Final output
    if args.diagnose_only:
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "embed_version": args.embed_version,
            "diagnostics": all_diagnostics,
        }
    else:
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "embed_version": args.embed_version,
            "n_iterations": args.n_iterations,
            "results": results,
            "best_cell": (
                max(results, key=lambda r: r["optimization"]["objective"])
                if results
                else None
            ),
        }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
