"""dFA gate study: operating-point sweep (A/B) + liked-only search (C) in one session.

Stage 1 (baseline): one 5-fold CV pass with the shipped params, exact
benchmark/compare.py estimator. The saved OOF score arrays yield:
  - A-curve: include threshold anchored on liked scores (recall targets)
  - B-curve: include threshold anchored on disliked scores (dFA caps)
  - exclude curve: exclude threshold sweep (disliked_recall vs lFR)

Stage 2 (search): Optuna over liked-model params only (disliked pinned at
shipped params). Objective = liked_recall at dFA = DFA_TARGET (threshold
anchored on disliked OOF scores — mechanism B evaluated per trial).

Stage 3 (analysis): winner + top5-median stability, gate check, ship verdict.

Gate: dFA <= ~DFA_TARGET (buffer under the 0.15 cap for +/-1.5pp SE on 603
tracks) AND liked_recall >= RECALL_FLOOR AND auc_include >= AUC_INCLUDE_FLOOR
AND auc_exclude >= AUC_EXCLUDE_FLOOR (disliked side pinned; drift guard).
"""

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
from core.preprocessing import StandardizeSelectPreprocessor
from core.storage import FeatureStore
from core.writer import start_extraction_job

logger = logging.getLogger(__name__)

SHIPPED_LIKED_PARAMS = {
    "knn_k_min": 7,
    "knn_k_max": 19,
    "knn_k_scale": 0.6530751049738679,
    "gmm_components_max": 28,
    "gmm_min_points_per_component": 80,
}
SHIPPED_DISLIKED_PARAMS = dict(SHIPPED_LIKED_PARAMS)
SHIPPED_OUTLIER_THRESHOLD = 0.08721065119224632

DFA_TARGET = 0.145
RECALL_FLOOR = 0.75
AUC_INCLUDE_FLOOR = 0.85
AUC_EXCLUDE_FLOOR = 0.785
IMPROVEMENT_MARGIN = 0.005

INCLUDE_RECALL_GRID = (0.85, 0.80, 0.75, 0.70, 0.65, 0.60)
DFA_CAP_GRID = (0.16, 0.15, 0.14)
EXCLUDE_RECALL_GRID = (0.95, 0.90, 0.85, 0.80, 0.75)

EXPECTED_BASELINE_AUC_INCLUDE = 0.8824241197115199
EXPECTED_BASELINE_AUC_EXCLUDE = 0.7903819082880172


def run_cv(
    X_liked: np.ndarray,
    X_disliked: np.ndarray,
    liked_model_params: dict,
    disliked_model_params: dict,
    liked_outlier_threshold: float,
    disliked_outlier_threshold: float,
) -> dict | None:
    """One CV pass mirroring the benchmark/compare.py estimator exactly.

    Per fold: per-set outlier detection (each side's knn_k_max), preprocessor
    refit on the filtered combined training data, DualOneClassModel fit with
    per-model params (cv_folds=None — CV happens here, as in compare.py).

    Returns dict of four OOF score arrays, or None for a degenerate trial.
    """
    n_splits = min(5, len(X_liked), len(X_disliked))
    if n_splits < 2:
        return None

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
            threshold=liked_outlier_threshold,
            knn_k=liked_model_params["knn_k_max"],
            min_set_size=config.model_min_set_size,
        )
        mask_d, _ = detect_outliers(
            X_d_tr,
            threshold=disliked_outlier_threshold,
            knn_k=disliked_model_params["knn_k_max"],
            min_set_size=config.model_min_set_size,
        )
        X_l_tr_f, X_d_tr_f = X_l_tr[mask_l], X_d_tr[mask_d]

        if len(X_l_tr_f) < 2 or len(X_d_tr_f) < 2:
            return None

        prep = StandardizeSelectPreprocessor(n_features=config.model_select_n_features)
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
                liked_params=liked_model_params,
                disliked_params=disliked_model_params,
                cv_folds=None,
                exclude_disliked_recall_target=config.model_exclude_disliked_recall_target,
                include_liked_recall_target=config.model_include_liked_recall_target,
            )
            model.fit(X_l_tr_f, X_d_tr_f)
        except ValueError as e:
            logger.warning(f"CV fold failed with ValueError: {e}")
            return None

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

    return {
        "like_on_liked": np.array(s_like_on_liked),
        "like_on_disliked": np.array(s_like_on_disliked),
        "dislike_on_disliked": np.array(s_dislike_on_disliked),
        "dislike_on_liked": np.array(s_dislike_on_liked),
    }


def include_curve(s_like_on_liked, s_like_on_disliked, recall_targets):
    """A-curve: include threshold anchored on liked-score percentiles"""
    rows = []
    for rt in recall_targets:
        threshold = float(np.percentile(s_like_on_liked, 100 * (1 - rt)))
        dfa = float(np.mean(s_like_on_disliked > threshold))
        rows.append(
            {
                "recall_target": float(rt),
                "threshold": threshold,
                "liked_recall": float(np.mean(s_like_on_liked > threshold)),
                "disliked_false_accept": dfa,
                "disliked_false_accept_se": float(
                    np.sqrt(dfa * (1 - dfa) / len(s_like_on_disliked))
                ),
            }
        )
    return rows


def dfa_cap_curve(s_like_on_liked, s_like_on_disliked, dfa_caps):
    """B-curve: include threshold anchored on disliked-score percentiles"""
    rows = []
    for cap in dfa_caps:
        threshold = float(np.percentile(s_like_on_disliked, 100 * (1 - cap)))
        rows.append(
            {
                "dfa_cap": float(cap),
                "threshold": threshold,
                "liked_recall": float(np.mean(s_like_on_liked > threshold)),
                "disliked_false_accept": float(np.mean(s_like_on_disliked > threshold)),
            }
        )
    return rows


def exclude_curve(s_dislike_on_disliked, s_dislike_on_liked, recall_targets):
    """Exclude-threshold sweep (report-only)"""
    rows = []
    for rt in recall_targets:
        threshold = float(np.percentile(s_dislike_on_disliked, 100 * (1 - rt)))
        rows.append(
            {
                "recall_target": float(rt),
                "threshold": threshold,
                "disliked_recall": float(np.mean(s_dislike_on_disliked >= threshold)),
                "liked_false_reject": float(np.mean(s_dislike_on_liked >= threshold)),
            }
        )
    return rows


def recall_at_dfa(s_like_on_liked, s_like_on_disliked, dfa_cap):
    """Mechanism-B operating point: recall kept at a dFA cap (objective)"""
    threshold = float(np.percentile(s_like_on_disliked, 100 * (1 - dfa_cap)))
    return {
        "threshold": threshold,
        "liked_recall": float(np.mean(s_like_on_liked > threshold)),
        "disliked_false_accept": float(np.mean(s_like_on_disliked > threshold)),
    }


def dfa_at_recall(s_like_on_liked, s_like_on_disliked, recall_target):
    """Mechanism-A operating point: dFA paid at a recall target"""
    threshold = float(np.percentile(s_like_on_liked, 100 * (1 - recall_target)))
    return {
        "threshold": threshold,
        "liked_recall": float(np.mean(s_like_on_liked > threshold)),
        "disliked_false_accept": float(np.mean(s_like_on_disliked > threshold)),
    }


def compute_metrics(scores: dict) -> dict:
    """AUCs + operating points for one set of OOF score arrays"""
    s_ll = scores["like_on_liked"]
    s_ld = scores["like_on_disliked"]
    s_dd = scores["dislike_on_disliked"]
    s_dl = scores["dislike_on_liked"]

    y_inc = [1] * len(s_ll) + [0] * len(s_ld)
    y_exc = [1] * len(s_dd) + [0] * len(s_dl)
    try:
        auc_include = float(roc_auc_score(y_inc, np.concatenate([s_ll, s_ld])))
    except ValueError:
        auc_include = 0.5
    try:
        auc_exclude = float(roc_auc_score(y_exc, np.concatenate([s_dd, s_dl])))
    except ValueError:
        auc_exclude = 0.5

    at_target = recall_at_dfa(s_ll, s_ld, DFA_TARGET)
    at_floor = dfa_at_recall(s_ll, s_ld, RECALL_FLOOR)
    at_80 = dfa_at_recall(s_ll, s_ld, 0.80)

    return {
        "auc_include": auc_include,
        "auc_exclude": auc_exclude,
        "recall_at_dfa_target": at_target["liked_recall"],
        "dfa_at_dfa_target": at_target["disliked_false_accept"],
        "dfa_at_recall_floor": at_floor["disliked_false_accept"],
        "dfa_at_recall_80": at_80["disliked_false_accept"],
    }


def evaluate_gate(recall_at_target: float, auc_include: float, auc_exclude: float):
    """Gate check: dFA <= target (by construction for mechanism B) plus floors"""
    checks = {
        "recall_floor": recall_at_target >= RECALL_FLOOR,
        "auc_include_floor": auc_include >= AUC_INCLUDE_FLOOR,
        "auc_exclude_floor": auc_exclude >= AUC_EXCLUDE_FLOOR,
    }
    return {"passed": all(checks.values()), "checks": checks}


def decide_shipment(
    winner_value: float,
    winner_gate: dict,
    baseline_value: float,
    baseline_gate: dict,
    margin: float = IMPROVEMENT_MARGIN,
) -> tuple[str, str]:
    """Ship verdict from the best search trial vs the baseline operating point"""
    if winner_gate["passed"] and winner_value > baseline_value + margin:
        return (
            "ship_C",
            f"Winner keeps recall {winner_value:.4f} at dFA={DFA_TARGET} vs "
            f"baseline {baseline_value:.4f} (+{winner_value - baseline_value:.4f} "
            f"> margin {margin}); ship per-model liked params + winner's "
            f"operating point",
        )
    if winner_gate["passed"] and not baseline_gate["passed"]:
        return (
            "ship_C",
            "Winner passes the gate but baseline does not; ship per-model "
            "liked params + winner's operating point",
        )
    if baseline_gate["passed"]:
        return (
            "ship_baseline_A",
            f"Search did not beat baseline by margin {margin} "
            f"(winner {winner_value:.4f} vs baseline {baseline_value:.4f}); "
            f"ship shipped params + baseline operating point via mechanism A "
            f"(recall target in config)",
        )
    return (
        "no_pass",
        "Neither baseline nor winner satisfies the gate at the recall floor; "
        "operating-point levers exhausted — revisit lever D / more disliked "
        "data before shipping",
    )


def objective(trial, X_liked, X_disliked):
    """Optuna objective: liked_recall at dFA=DFA_TARGET (disliked side pinned)"""
    knn_k_min = trial.suggest_int("liked_knn_k_min", 3, 8)
    knn_k_max = trial.suggest_int("liked_knn_k_max", 8, 25)
    if knn_k_max < knn_k_min:
        raise optuna.exceptions.TrialPruned("liked_knn_k_max < liked_knn_k_min")

    knn_k_scale = trial.suggest_float("liked_knn_k_scale", 0.3, 1.0)
    gmm_components_max = trial.suggest_int("liked_gmm_components_max", 8, 32)
    gmm_min_points_per_component = trial.suggest_int(
        "liked_gmm_min_points_per_component", 20, 80
    )
    liked_outlier_threshold = trial.suggest_float("liked_outlier_threshold", 0.01, 0.10)

    liked_params = {
        "knn_k_min": knn_k_min,
        "knn_k_max": knn_k_max,
        "knn_k_scale": knn_k_scale,
        "gmm_components_max": gmm_components_max,
        "gmm_min_points_per_component": gmm_min_points_per_component,
    }

    scores = run_cv(
        X_liked,
        X_disliked,
        liked_model_params=liked_params,
        disliked_model_params=SHIPPED_DISLIKED_PARAMS,
        liked_outlier_threshold=liked_outlier_threshold,
        disliked_outlier_threshold=SHIPPED_OUTLIER_THRESHOLD,
    )
    if scores is None:
        for key in (
            "auc_include",
            "auc_exclude",
            "recall_at_dfa_target",
            "dfa_at_dfa_target",
            "dfa_at_recall_floor",
            "dfa_at_recall_80",
        ):
            trial.set_user_attr(key, 0.5)
        return 0.0

    metrics = compute_metrics(scores)
    for key, value in metrics.items():
        trial.set_user_attr(key, value)
    return metrics["recall_at_dfa_target"]


def load_variant_matrices(config_path: str, user_id: int, n_workers: int):
    """Load liked/disliked feature matrices for the 'full' variant (cache-aware)"""
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    embedding_var = next(
        (e for e in config_data["embeddings"] if e["name"] == "full"), None
    )
    if embedding_var is None:
        raise SystemExit("Variant 'full' not found in config")

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

    embed_version = get_embed_version(
        profile_path, panns_weights_path, use_panns=use_panns
    )
    segment_policy = segment_spec.canonical()

    liked_path = config.get_liked_file_store_path(user_id)
    disliked_path = config.get_disliked_file_store_path(user_id)
    liked_tracks = list(liked_path.glob("*.mp3"))
    disliked_tracks = list(disliked_path.glob("*.mp3"))
    all_tracks = [(t, "like") for t in liked_tracks] + [
        (t, "dislike") for t in disliked_tracks
    ]

    job_id = (
        f"dfa_study_{user_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    t0 = time.monotonic()
    logger.info(f"Job {job_id}: probing feature cache for {len(all_tracks)} tracks")
    extraction_result = start_extraction_job(
        user_id=user_id,
        tracks=all_tracks,
        embed_version=embed_version,
        segment_policy=segment_policy,
        job_id=job_id,
        n_workers=n_workers,
        panns_weights_path=panns_weights_path,
        profile_path=profile_path,
        segment_spec=segment_spec,
        use_panns=use_panns,
    )
    logger.info(
        f"Job {job_id}: extraction done in {time.monotonic() - t0:.1f}s "
        f"(ok={extraction_result.ok}, failed={extraction_result.failed}, "
        f"cached={extraction_result.skipped})"
    )

    store = FeatureStore(user_id, embed_version, segment_policy)
    with store.training_view("like") as liked_pq:
        X_liked = FeatureStore.load_vectors(liked_pq)
    with store.training_view("dislike") as disliked_pq:
        X_disliked = FeatureStore.load_vectors(disliked_pq)
    return X_liked, X_disliked


def print_curve(title: str, rows: list[dict], cols: list[str]):
    print(title)
    for row in rows:
        cells = "  ".join(
            f"{c}={row[c]:.4f}" if isinstance(row[c], float) else f"{c}={row[c]}"
            for c in cols
        )
        print(f"  {cells}")


def main():
    parser = argparse.ArgumentParser(
        description="dFA gate study: operating-point sweep + liked-only search"
    )
    parser.add_argument(
        "--config", type=str, default="benchmark/full_only.yaml", help="Variants YAML"
    )
    parser.add_argument(
        "--user-id", type=int, required=True, help="Telegram user ID for DuckDB access"
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=40,
        help="Optuna trials for the liked-only search",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/benchmark/dfa_gate_study.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--scores-output",
        type=str,
        default="data/benchmark/dfa_gate_scores.npz",
        help="OOF score arrays (baseline + winner) path",
    )
    parser.add_argument(
        "--n-workers", type=int, default=2, help="Extraction worker processes"
    )
    parser.add_argument(
        "--sanity-tolerance",
        type=float,
        default=0.005,
        help="Max |AUC - expected| for the Stage-1 estimator sanity check",
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip the Stage-1 reproduction check (e.g. corpus changed)",
    )
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_liked, X_disliked = load_variant_matrices(
        args.config, args.user_id, args.n_workers
    )
    logger.info(
        f"Loaded {len(X_liked)} liked, {len(X_disliked)} disliked feature vectors"
    )

    # Stage 1: baseline pass with shipped params
    print("=== Stage 1: baseline (shipped params, one CV pass) ===")
    baseline_scores = run_cv(
        X_liked,
        X_disliked,
        liked_model_params=SHIPPED_LIKED_PARAMS,
        disliked_model_params=SHIPPED_DISLIKED_PARAMS,
        liked_outlier_threshold=SHIPPED_OUTLIER_THRESHOLD,
        disliked_outlier_threshold=SHIPPED_OUTLIER_THRESHOLD,
    )
    if baseline_scores is None:
        raise SystemExit("Baseline CV pass failed")

    baseline_metrics = compute_metrics(baseline_scores)
    baseline_inc = include_curve(
        baseline_scores["like_on_liked"],
        baseline_scores["like_on_disliked"],
        INCLUDE_RECALL_GRID,
    )
    baseline_cap = dfa_cap_curve(
        baseline_scores["like_on_liked"],
        baseline_scores["like_on_disliked"],
        DFA_CAP_GRID,
    )
    baseline_exc = exclude_curve(
        baseline_scores["dislike_on_disliked"],
        baseline_scores["dislike_on_liked"],
        EXCLUDE_RECALL_GRID,
    )

    diff_inc = abs(baseline_metrics["auc_include"] - EXPECTED_BASELINE_AUC_INCLUDE)
    diff_exc = abs(baseline_metrics["auc_exclude"] - EXPECTED_BASELINE_AUC_EXCLUDE)
    sanity_ok = (diff_inc <= args.sanity_tolerance) and (
        diff_exc <= args.sanity_tolerance
    )
    if not args.skip_sanity_check and not sanity_ok:
        raise SystemExit(
            f"Stage-1 sanity check FAILED: "
            f"auc_include={baseline_metrics['auc_include']:.4f} "
            f"(expected {EXPECTED_BASELINE_AUC_INCLUDE:.4f}), "
            f"auc_exclude={baseline_metrics['auc_exclude']:.4f} "
            f"(expected {EXPECTED_BASELINE_AUC_EXCLUDE:.4f}). "
            f"If the corpus changed since segment_report.json, rerun with "
            f"--skip-sanity-check."
        )

    print(
        f"auc_include={baseline_metrics['auc_include']:.4f} "
        f"(expected {EXPECTED_BASELINE_AUC_INCLUDE:.4f}), "
        f"auc_exclude={baseline_metrics['auc_exclude']:.4f} "
        f"(expected {EXPECTED_BASELINE_AUC_EXCLUDE:.4f})"
    )
    print(
        f"Baseline operating points: recall@dfa{DFA_TARGET}="
        f"{baseline_metrics['recall_at_dfa_target']:.4f} "
        f"(dfa={baseline_metrics['dfa_at_dfa_target']:.4f}), "
        f"dfa@recall{RECALL_FLOOR}={baseline_metrics['dfa_at_recall_floor']:.4f}, "
        f"dfa@recall0.80={baseline_metrics['dfa_at_recall_80']:.4f}"
    )
    print_curve(
        "A-curve (recall target -> dFA):",
        baseline_inc,
        [
            "recall_target",
            "threshold",
            "liked_recall",
            "disliked_false_accept",
            "disliked_false_accept_se",
        ],
    )
    print_curve(
        "B-curve (dFA cap -> recall):",
        baseline_cap,
        ["dfa_cap", "threshold", "liked_recall", "disliked_false_accept"],
    )
    print_curve(
        "Exclude curve (recall target -> lFR):",
        baseline_exc,
        ["recall_target", "threshold", "disliked_recall", "liked_false_reject"],
    )

    baseline_gate = evaluate_gate(
        baseline_metrics["recall_at_dfa_target"],
        baseline_metrics["auc_include"],
        baseline_metrics["auc_exclude"],
    )

    # Stage 2: liked-only search (disliked pinned)
    print(f"=== Stage 2: liked-only Optuna search ({args.n_iterations} trials) ===")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective(trial, X_liked, X_disliked),
        n_trials=args.n_iterations,
        show_progress_bar=False,
    )

    valid_trials = [t for t in study.trials if t.value is not None]
    if valid_trials:
        winner = max(valid_trials, key=lambda t: t.value)
    else:
        winner = None

    trial_history = [
        {
            "params": t.params,
            "value": t.value,
            **{
                k: t.user_attrs.get(k, 0.5)
                for k in (
                    "auc_include",
                    "auc_exclude",
                    "recall_at_dfa_target",
                    "dfa_at_dfa_target",
                    "dfa_at_recall_floor",
                    "dfa_at_recall_80",
                )
            },
        }
        for t in study.trials
    ]

    top_trials = sorted(valid_trials, key=lambda t: t.value, reverse=True)[
        : min(5, len(valid_trials))
    ]
    top5_median = {
        key: float(np.median([t.user_attrs.get(key, 0.5) for t in top_trials]))
        for key in (
            "recall_at_dfa_target",
            "auc_include",
            "auc_exclude",
            "dfa_at_recall_floor",
        )
    }

    if winner is not None:
        winner_gate = evaluate_gate(
            winner.user_attrs.get("recall_at_dfa_target", 0.5),
            winner.user_attrs.get("auc_include", 0.5),
            winner.user_attrs.get("auc_exclude", 0.5),
        )
    else:
        winner_gate = evaluate_gate(0.0, 0.0, 0.0)

    # Stage 3: verdict
    verdict, reason = decide_shipment(
        winner.value if winner is not None else 0.0,
        winner_gate,
        baseline_metrics["recall_at_dfa_target"],
        baseline_gate,
    )

    print("=== Stage 3: analysis ===")
    if winner is not None:
        print(
            f"Winner: value={winner.value:.4f} "
            f"auc_include={winner.user_attrs.get('auc_include', 0.5):.4f} "
            f"auc_exclude={winner.user_attrs.get('auc_exclude', 0.5):.4f} "
            f"dfa@recall{RECALL_FLOOR}="
            f"{winner.user_attrs.get('dfa_at_recall_floor', 0.5):.4f}"
        )
        print(f"Winner params: {winner.params}")
    else:
        print("No valid search trials produced")
    print(f"Top5 median: {top5_median}")
    print(f"Baseline gate: {baseline_gate}")
    print(f"Winner gate: {winner_gate}")
    print(f"VERDICT: {verdict} — {reason}")

    # Winner's full tables: recompute its scores
    winner_scores = None
    winner_tables = None
    if winner is not None:
        winner_scores = run_cv(
            X_liked,
            X_disliked,
            liked_model_params={
                "knn_k_min": winner.params["liked_knn_k_min"],
                "knn_k_max": winner.params["liked_knn_k_max"],
                "knn_k_scale": winner.params["liked_knn_k_scale"],
                "gmm_components_max": winner.params["liked_gmm_components_max"],
                "gmm_min_points_per_component": winner.params[
                    "liked_gmm_min_points_per_component"
                ],
            },
            disliked_model_params=SHIPPED_DISLIKED_PARAMS,
            liked_outlier_threshold=winner.params["liked_outlier_threshold"],
            disliked_outlier_threshold=SHIPPED_OUTLIER_THRESHOLD,
        )
        if winner_scores is not None:
            winner_tables = {
                "include_curve": include_curve(
                    winner_scores["like_on_liked"],
                    winner_scores["like_on_disliked"],
                    INCLUDE_RECALL_GRID,
                ),
                "dfa_cap_curve": dfa_cap_curve(
                    winner_scores["like_on_liked"],
                    winner_scores["like_on_disliked"],
                    DFA_CAP_GRID,
                ),
                "exclude_curve": exclude_curve(
                    winner_scores["dislike_on_disliked"],
                    winner_scores["dislike_on_liked"],
                    EXCLUDE_RECALL_GRID,
                ),
            }
            print("Winner A-curve:")
            print_curve(
                "",
                winner_tables["include_curve"],
                ["recall_target", "threshold", "liked_recall", "disliked_false_accept"],
            )
            print("Winner B-curve:")
            print_curve(
                "",
                winner_tables["dfa_cap_curve"],
                ["dfa_cap", "threshold", "liked_recall", "disliked_false_accept"],
            )

    # Save OOF score arrays for offline analysis
    scores_payload = {f"baseline_{k}": v for k, v in baseline_scores.items()}
    if winner_scores is not None:
        scores_payload.update({f"winner_{k}": v for k, v in winner_scores.items()})
    scores_path = Path(args.scores_output)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scores_path, **scores_payload)
    print(f"Saved OOF score arrays to {scores_path}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "study_config": {
            "dfa_target": DFA_TARGET,
            "recall_floor": RECALL_FLOOR,
            "auc_include_floor": AUC_INCLUDE_FLOOR,
            "auc_exclude_floor": AUC_EXCLUDE_FLOOR,
            "improvement_margin": IMPROVEMENT_MARGIN,
            "n_iterations": args.n_iterations,
            "disliked_params_pinned": SHIPPED_DISLIKED_PARAMS,
            "disliked_outlier_threshold_pinned": SHIPPED_OUTLIER_THRESHOLD,
        },
        "baseline": {
            "params": SHIPPED_LIKED_PARAMS,
            "outlier_threshold": SHIPPED_OUTLIER_THRESHOLD,
            "metrics": baseline_metrics,
            "gate": baseline_gate,
            "include_curve": baseline_inc,
            "dfa_cap_curve": baseline_cap,
            "exclude_curve": baseline_exc,
            "sanity": {
                "expected_auc_include": EXPECTED_BASELINE_AUC_INCLUDE,
                "expected_auc_exclude": EXPECTED_BASELINE_AUC_EXCLUDE,
                "abs_diff_auc_include": diff_inc,
                "abs_diff_auc_exclude": diff_exc,
                "tolerance": args.sanity_tolerance,
                "skipped": args.skip_sanity_check,
                "ok": bool(sanity_ok or args.skip_sanity_check),
            },
        },
        "optimization": {
            "objective": f"liked_recall at dFA={DFA_TARGET}",
            "n_trials": args.n_iterations,
            "best_trial": (
                {
                    "params": winner.params,
                    "value": winner.value,
                    "metrics": {
                        k: winner.user_attrs.get(k, 0.5)
                        for k in (
                            "auc_include",
                            "auc_exclude",
                            "recall_at_dfa_target",
                            "dfa_at_dfa_target",
                            "dfa_at_recall_floor",
                            "dfa_at_recall_80",
                        )
                    },
                    "gate": winner_gate,
                }
                if winner is not None
                else None
            ),
            "top5_median": top5_median,
            "trial_history": trial_history,
        },
        "decision": {"verdict": verdict, "reason": reason},
        "winner_tables": winner_tables,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
