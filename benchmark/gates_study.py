"""Gates study: outlier method x budget x selection, multi-objective (NSGA-II).

Objectives: minimize (lfr@0.80, dfa@0.775) computed from 5-fold OOF calibrated
scores, averaged over 3 seeds. Per-model params pinned at shipped values; the
feature set is fixed and loaded directly from parquet shards (no extraction),
so this study isolates data-cleaning and preprocessing levers:

  outlier_method: prod_fused | knn | iforest | std_fused | lof_std
  outlier_budget: 0.02 .. 0.12 (prod fused removes less than nominal: its
                  rank fusion behaves as a consensus rule)
  selection:      welch64 | ridge_select64 | fused_welch_ridge64 |
                  quota64 | pls_project64

Baseline row = shipped config (prod_fused @ 0.08721065119224632 + welch64).
Guideline (owner): lfr@0.80 <= 0.20 AND dfa@0.775 <= 0.20; stretch ~0.12/0.08.

Extra tooling for the Phase 5 feature ablation:
  --slice-arms FEATURES_DIR  write column-sliced copies (baseline 4368 /
                             +key-scale 4380 / full essentia block, panns
                             tail kept) of an extracted features dir and exit
  --essentia-dims K          run the study on a sliced arm dir (builds the
                             quota family layout for the arm's essentia width)
  --extra-cells              also evaluate the pinned parity cells (prod
                             baseline + ship candidate) with full metric dicts
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import optuna
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

import config
from audio.features import _DESCRIPTOR_SCHEMA, descriptor_family_layout
from core.modeling import OneClassSetModel
from core.outliers import detect_outliers
from core.preprocessing import (
    QuotaSelectPreprocessor,
    RidgeSelectPreprocessor,
    StandardizeSelectPreprocessor,
    welch_scores,
)

logger = logging.getLogger(__name__)

SHIPPED_MODEL_PARAMS = {
    "knn_k_min": 7,
    "knn_k_max": 19,
    "knn_k_scale": 0.6530751049738679,
    "gmm_components_max": 28,
    "gmm_min_points_per_component": 80,
}
SHIPPED_OUTLIER_BUDGET = 0.08721065119224632

LFR_RECALL_TARGETS = (0.70, 0.80, 0.90)
DFA_RECALL_TARGETS = (0.80, 0.775)
GUIDELINE_LFR_CAP = 0.20
GUIDELINE_DFA_CAP = 0.20
STRETCH_LFR = 0.12
STRETCH_DFA = 0.08

ESSENTIA_DIMS = sum(_len for _, _len, _ in _DESCRIPTOR_SCHEMA)
FAMILY_QUOTA = {
    "lowlevel": 16,
    "tonal": 12,
    "rhythm": 12,
    "panns": 24,
    "frames": 12,
}
BASELINE_ARM_ENTRY = "tonal.chords_key"
KEYSCALE_ARM_ENTRY = "frames.pitch"


def family_layout(essentia_dims: int | None = None) -> list[tuple[str, int, int]]:
    """Family slices for a vector whose essentia block is the first
    essentia_dims dims (panns tail last). None means the full schema."""
    total = ESSENTIA_DIMS if essentia_dims is None else essentia_dims
    layout = []
    for name, start, end in descriptor_family_layout():
        if start >= total:
            continue
        layout.append((name, start, min(end, total)))
    layout.append(("panns", total, -1))
    return layout


def _knn_ranks(X: np.ndarray, k: int) -> np.ndarray:
    knn = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = knn.kneighbors(X)
    return stats.rankdata(dists.mean(axis=1)) / len(X)


def _iforest_ranks(X: np.ndarray) -> np.ndarray:
    ifo = IsolationForest(n_estimators=200, random_state=42, contamination="auto")
    ifo.fit_predict(X)
    return stats.rankdata(-ifo.score_samples(X)) / len(X)


def _prod_fused_mask(X: np.ndarray, budget: float) -> np.ndarray:
    mask, _ = detect_outliers(
        X,
        threshold=budget,
        knn_k=SHIPPED_MODEL_PARAMS["knn_k_max"],
        n_estimators=200,
        min_set_size=config.model_min_set_size,
    )
    return mask


def _knn_mask(X: np.ndarray, budget: float) -> np.ndarray:
    return _knn_ranks(X, SHIPPED_MODEL_PARAMS["knn_k_max"]) < (1.0 - budget)


def _iforest_mask(X: np.ndarray, budget: float) -> np.ndarray:
    return _iforest_ranks(X) < (1.0 - budget)


def _std_fused_mask(X: np.ndarray, budget: float) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    fused = (_knn_ranks(Xs, SHIPPED_MODEL_PARAMS["knn_k_max"]) + _iforest_ranks(Xs)) / 2
    return fused < (1.0 - budget)


def _lof_std_mask(X: np.ndarray, budget: float) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    lof = LocalOutlierFactor(n_neighbors=SHIPPED_MODEL_PARAMS["knn_k_max"])
    lof.fit_predict(Xs)
    return stats.rankdata(-lof.negative_outlier_factor_) / len(Xs) < (1.0 - budget)


OUTLIER_METHODS = {
    "prod_fused": _prod_fused_mask,
    "knn": _knn_mask,
    "iforest": _iforest_mask,
    "std_fused": _std_fused_mask,
    "lof_std": _lof_std_mask,
}


class FusedWelchRidgePreprocessor:
    """Rank-fusion of Welch t-stat and |logistic coefficient|, top-n dims"""

    def __init__(self, n_features: int = 64, C: float = 0.01):
        self.n_features = n_features
        self.C = C

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        lr = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000).fit(
            Xs, y
        )
        w_rank = stats.rankdata(np.abs(lr.coef_[0])) / Xs.shape[1]
        t_rank = stats.rankdata(welch_scores(Xs, y)) / Xs.shape[1]
        fused = (w_rank + t_rank) / 2
        self.selected_ = np.argsort(fused)[::-1][: self.n_features]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler_.transform(X)[:, self.selected_]


class PlsProjectPreprocessor:
    """Standardize + PLS projection to n components (dense, supervised)"""

    def __init__(self, n_features: int = 64):
        self.n_features = n_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        n_comp = min(self.n_features, Xs.shape[0] - 1, Xs.shape[1])
        self.pls_ = PLSRegression(n_components=n_comp).fit(Xs, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pls_.transform(self.scaler_.transform(X))


def _quota(
    n_features: int, essentia_dims: int | None = None
) -> QuotaSelectPreprocessor:
    return QuotaSelectPreprocessor(
        n_features=n_features,
        families=family_layout(essentia_dims),
        family_quota=FAMILY_QUOTA,
    )


SELECTION_VARIANTS = {
    "welch64": lambda d=None: StandardizeSelectPreprocessor(n_features=64),
    "welch32": lambda d=None: StandardizeSelectPreprocessor(n_features=32),
    "ridge_select32": lambda d=None: RidgeSelectPreprocessor(n_features=32),
    "ridge_select64": lambda d=None: RidgeSelectPreprocessor(n_features=64),
    "ridge_select128": lambda d=None: RidgeSelectPreprocessor(n_features=128),
    "ridge64_c001": lambda d=None: RidgeSelectPreprocessor(n_features=64, C=0.001),
    "ridge64_c01": lambda d=None: RidgeSelectPreprocessor(n_features=64, C=0.1),
    "fused_welch_ridge64": lambda d=None: FusedWelchRidgePreprocessor(),
    "quota32": lambda d=None: _quota(32, d),
    "quota64": lambda d=None: _quota(64, d),
    "quota128": lambda d=None: _quota(128, d),
    "pls_project64": lambda d=None: PlsProjectPreprocessor(),
}

# Per-model selections: "per:LIKE/DISLIKE" — each one-class model gets its own
# preprocessor (both fitted on the combined labeled training data, as usual).
PER_MODEL_SELECTIONS = (
    "per:welch64/ridge_select64",
    "per:quota64/ridge_select64",
    "per:welch64/fused_welch_ridge64",
    "per:quota64/ridge_select128",
    "per:welch32/ridge_select64",
)

# Focused per-model search space (gates_study --focus per_model): like-side
# variants that drove dfa down, dislike-side variants that drove lfr down.
PER_MODEL_FOCUS = (
    "per:quota64/ridge_select64",
    "per:quota64/ridge_select128",
    "per:quota64/ridge64_c001",
    "per:quota64/ridge64_c01",
    "per:quota128/ridge_select64",
    "per:quota128/ridge_select128",
    "per:quota128/ridge64_c001",
    "per:welch64/ridge_select64",
    "per:welch64/ridge_select128",
    "per:welch64/ridge64_c001",
    "per:quota32/ridge_select64",
)
FOCUS_OUTLIER_METHODS = ("prod_fused", "knn")


def _parse_selection(selection: str) -> tuple[str, str]:
    """Return (like_selection, dislike_selection) for a selection string.

    Plain names apply to both models; "per:LIKE/DISLIKE" splits per model.
    """
    if selection.startswith("per:"):
        like_sel, dislike_sel = selection[len("per:") :].split("/")
        return like_sel, dislike_sel
    return selection, selection


def make_preprocessor(name: str, essentia_dims: int | None = None):
    factory = SELECTION_VARIANTS.get(name)
    if factory is None:
        raise ValueError(f"Unknown selection variant: {name}")
    return factory(essentia_dims)


def load_features(features_dir: str) -> tuple[np.ndarray, np.ndarray]:
    def load(set_name: str) -> np.ndarray:
        rows = duckdb.sql(f"""
            select vector from read_parquet('{features_dir}/{set_name}/*.parquet')
            order by file_hash
            """).fetchall()
        return np.asarray([r[0] for r in rows], dtype=np.float64)

    return load("like"), load("dislike")


def run_cv(
    X_liked: np.ndarray,
    X_disliked: np.ndarray,
    outlier_method: str,
    outlier_budget: float,
    selection: str,
    model_params: dict | None = None,
    seeds: tuple[int, ...] = (42, 43, 44),
    essentia_dims: int | None = None,
) -> dict | None:
    """5-fold OOF x seeds; returns mean metrics or None for a degenerate trial"""
    params = dict(model_params or SHIPPED_MODEL_PARAMS)
    outlier_fn = OUTLIER_METHODS[outlier_method]
    per_seed: list[dict] = []

    for seed in seeds:
        n_splits = min(5, len(X_liked), len(X_disliked))
        if n_splits < 2:
            return None
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        s_like_on_liked = []
        s_like_on_disliked = []
        s_dislike_on_disliked = []
        s_dislike_on_liked = []

        folds_liked = list(kf.split(X_liked))
        folds_disliked = list(kf.split(X_disliked))

        for (l_train, l_test), (d_train, d_test) in zip(folds_liked, folds_disliked):
            X_l_tr, X_l_te = X_liked[l_train], X_liked[l_test]
            X_d_tr, X_d_te = X_disliked[d_train], X_disliked[d_test]
            mask_l = outlier_fn(X_l_tr, outlier_budget)
            mask_d = outlier_fn(X_d_tr, outlier_budget)
            X_l_tr_f, X_d_tr_f = X_l_tr[mask_l], X_d_tr[mask_d]
            if len(X_l_tr_f) < 2 or len(X_d_tr_f) < 2:
                return None

            like_sel, dis_sel = _parse_selection(selection)
            prep_like = make_preprocessor(like_sel, essentia_dims)
            prep_dis = make_preprocessor(dis_sel, essentia_dims)
            X_combined_tr = np.concatenate([X_l_tr_f, X_d_tr_f])
            y_combined_tr = np.concatenate(
                [np.ones(len(X_l_tr_f)), np.zeros(len(X_d_tr_f))]
            )
            prep_like.fit(X_combined_tr, y_combined_tr)
            prep_dis.fit(X_combined_tr, y_combined_tr)
            # each model trains in its own space...
            X_l_tr_like = prep_like.transform(X_l_tr_f)
            X_d_tr_dis = prep_dis.transform(X_d_tr_f)
            # ...and each test set is scored in BOTH spaces
            X_l_te_like = prep_like.transform(X_l_te)
            X_d_te_like = prep_like.transform(X_d_te)
            X_d_te_dis = prep_dis.transform(X_d_te)
            X_l_te_dis = prep_dis.transform(X_l_te)

            try:
                like_model = OneClassSetModel(**params)
                dis_model = OneClassSetModel(**params)
                like_model.fit(X_l_tr_like)
                dis_model.fit(X_d_tr_dis)
            except ValueError as e:
                logger.warning(f"CV fold failed with ValueError: {e}")
                return None

            s_like_on_liked.extend(
                like_model.score(x.reshape(1, -1))["calibrated"] for x in X_l_te_like
            )
            s_like_on_disliked.extend(
                like_model.score(x.reshape(1, -1))["calibrated"] for x in X_d_te_like
            )
            s_dislike_on_disliked.extend(
                dis_model.score(x.reshape(1, -1))["calibrated"] for x in X_d_te_dis
            )
            s_dislike_on_liked.extend(
                dis_model.score(x.reshape(1, -1))["calibrated"] for x in X_l_te_dis
            )

        scores = {
            "like_on_liked": np.asarray(s_like_on_liked),
            "like_on_disliked": np.asarray(s_like_on_disliked),
            "dislike_on_disliked": np.asarray(s_dislike_on_disliked),
            "dislike_on_liked": np.asarray(s_dislike_on_liked),
        }
        per_seed.append(compute_metrics(scores))

    metrics: dict = {}
    for key in per_seed[0]:
        vals = np.array([m[key] for m in per_seed], dtype=float)
        metrics[key] = float(vals.mean())
        if key in ("lfr_at_0.8", "dfa_at_0.775"):
            metrics[f"{key}_std"] = float(vals.std())
    return metrics


def compute_metrics(scores: dict) -> dict:
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

    metrics = {"auc_include": auc_include, "auc_exclude": auc_exclude}
    for rt in LFR_RECALL_TARGETS:
        thr = float(np.percentile(s_dd, 100 * (1 - rt)))
        metrics[f"lfr_at_{rt}"] = float(np.mean(s_dl >= thr))
    for rt in DFA_RECALL_TARGETS:
        thr = float(np.percentile(s_ll, 100 * (1 - rt)))
        metrics[f"dfa_at_{rt}"] = float(np.mean(s_ld > thr))
    return metrics


def _verdict(lfr: float, dfa: float) -> str:
    if lfr <= STRETCH_LFR and dfa <= STRETCH_DFA:
        return "stretch"
    if lfr <= GUIDELINE_LFR_CAP and dfa <= GUIDELINE_DFA_CAP:
        return "guideline"
    return "fail"


def verdict(metrics: dict) -> str:
    return _verdict(metrics["lfr_at_0.8"], metrics["dfa_at_0.775"])


def verdict_at_0_9(metrics: dict) -> str:
    return _verdict(metrics["lfr_at_0.9"], metrics["dfa_at_0.775"])


def objective(trial, X_liked, X_disliked, focus: str = "full", essentia_dims=None):
    if focus == "per_model":
        outlier_method = trial.suggest_categorical(
            "outlier_method", list(FOCUS_OUTLIER_METHODS)
        )
        outlier_budget = trial.suggest_float("outlier_budget", 0.03, 0.10)
        selection = trial.suggest_categorical("selection", list(PER_MODEL_FOCUS))
    else:
        outlier_method = trial.suggest_categorical(
            "outlier_method", list(OUTLIER_METHODS)
        )
        outlier_budget = trial.suggest_float("outlier_budget", 0.02, 0.12)
        selection = trial.suggest_categorical(
            "selection", list(SELECTION_VARIANTS) + list(PER_MODEL_SELECTIONS)
        )

    metrics = run_cv(
        X_liked,
        X_disliked,
        outlier_method=outlier_method,
        outlier_budget=outlier_budget,
        selection=selection,
        essentia_dims=essentia_dims,
    )
    if metrics is None:
        trial.set_user_attr("failed", True)
        return [1.0, 1.0]
    for key, value in metrics.items():
        trial.set_user_attr(key, value)
    trial.set_user_attr("verdict", verdict(metrics))
    return [metrics["lfr_at_0.8"], metrics["dfa_at_0.775"]]


EXTRA_CELLS = (
    {
        "name": "prod_baseline",
        "outlier_method": "prod_fused",
        "outlier_budget": SHIPPED_OUTLIER_BUDGET,
        "selection": "welch64",
    },
    {
        "name": "ship_candidate",
        "outlier_method": "prod_fused",
        "outlier_budget": 0.07,
        "selection": "per:quota64/ridge_select64",
    },
)


def _schema_offset(entry_name: str) -> int:
    offset = 0
    for name, length, _ in _DESCRIPTOR_SCHEMA:
        if name == entry_name:
            return offset
        offset += length
    raise KeyError(f"schema entry not found: {entry_name}")


def ablation_arm_dims() -> dict[str, int]:
    return {
        "baseline": _schema_offset(BASELINE_ARM_ENTRY),
        "keyscale": _schema_offset(KEYSCALE_ARM_ENTRY),
        "full": ESSENTIA_DIMS,
    }


def _feature_width(features_dir: Path) -> int:
    row = duckdb.sql(
        f"SELECT len(vector) FROM read_parquet('{features_dir}/*/*.parquet') LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError(f"No parquet shards under {features_dir}")
    return int(row[0])


def write_ablation_arms(features_dir: str | Path) -> dict[str, str]:
    """Column-slice a full-width features dir into the 3 ablation arms.

    Each arm keeps the first essentia_dims essentia columns plus the tail
    after the full essentia block (panns). Output dirs are named
    {src}_arm{essentia_dims} beside the source dir; run gates_study on them
    with --essentia-dims {essentia_dims}.
    """
    src = Path(features_dir)
    width = _feature_width(src)
    if width < ESSENTIA_DIMS:
        raise ValueError(
            f"features under {src} are {width} wide, expected >= {ESSENTIA_DIMS}"
        )
    set_names = sorted(
        row[0]
        for row in duckdb.sql(
            f"SELECT DISTINCT set_name FROM read_parquet('{src}/*/*.parquet')"
        ).fetchall()
    )
    if not set_names:
        raise RuntimeError(f"No shards under {src}")
    arms = {}
    for arm, essentia_dims in ablation_arm_dims().items():
        out_dir = src.parent / f"{src.name}_arm{essentia_dims}"
        for set_name in set_names:
            out_set = out_dir / set_name
            out_set.mkdir(parents=True, exist_ok=True)
            duckdb.sql(f"""
                COPY (
                    SELECT * REPLACE (
                        list_slice(vector, 1, {essentia_dims})
                        || list_slice(vector, {ESSENTIA_DIMS + 1}, len(vector))
                        AS vector
                    )
                    FROM read_parquet('{src}/{set_name}/*.parquet')
                )
                TO '{out_set}/data.parquet'
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """)
        arms[arm] = str(out_dir)
        logger.info(
            f"Wrote ablation arm {arm} ({essentia_dims} essentia dims + "
            f"{width - ESSENTIA_DIMS} tail dims) to {out_dir}"
        )
    return arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--n-iterations", type=int, default=60)
    parser.add_argument(
        "--focus",
        choices=("full", "per_model"),
        default="full",
        help="full: whole space; per_model: focused per-model selection grid",
    )
    parser.add_argument("--output", default="data/benchmark/gates_study.json")
    parser.add_argument(
        "--extra-cells",
        action="store_true",
        help="also evaluate the pinned parity cells (prod baseline + ship "
        "candidate) with full metric dicts",
    )
    parser.add_argument(
        "--essentia-dims",
        type=int,
        default=None,
        help="essentia block width of the features dir (for column-sliced "
        "ablation arms); default: full schema width",
    )
    parser.add_argument(
        "--slice-arms",
        default=None,
        metavar="FEATURES_DIR",
        help="write column-sliced 3-arm ablation copies of FEATURES_DIR and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.slice_arms:
        arms = write_ablation_arms(args.slice_arms)
        for arm, out_dir in arms.items():
            print(f"{arm}: {out_dir}")
        return

    essentia_dims = args.essentia_dims

    X_liked, X_disliked = load_features(args.features_dir)
    logger.info(
        f"Loaded liked={len(X_liked)} disliked={len(X_disliked)} dim={X_liked.shape[1]}"
    )

    start = time.time()
    baseline = run_cv(
        X_liked,
        X_disliked,
        outlier_method="prod_fused",
        outlier_budget=SHIPPED_OUTLIER_BUDGET,
        selection="welch64",
        essentia_dims=essentia_dims,
    )
    logger.info(f"Baseline metrics ({time.time() - start:.0f}s): {baseline}")

    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=42),
    )
    study.optimize(
        lambda t: objective(
            t, X_liked, X_disliked, focus=args.focus, essentia_dims=essentia_dims
        ),
        n_trials=args.n_iterations,
    )

    front = [
        {
            "number": t.number,
            "params": t.params,
            "values": t.values,
            "metrics": {
                k: t.user_attrs[k]
                for k in t.user_attrs
                if k not in ("failed", "verdict")
            },
            "verdict": t.user_attrs.get("verdict"),
        }
        for t in study.best_trials
        if not t.user_attrs.get("failed")
    ]
    for row in front:
        row["verdict"] = verdict(row["metrics"])
        row["verdict_at_0.9"] = verdict_at_0_9(row["metrics"])

    history = [
        {
            "number": t.number,
            "params": t.params,
            "values": t.values,
            "lfr_at_0.8": t.user_attrs.get("lfr_at_0.8"),
            "dfa_at_0.775": t.user_attrs.get("dfa_at_0.775"),
            "metrics": {
                k: v for k, v in t.user_attrs.items() if k not in ("failed", "verdict")
            },
            "verdict": t.user_attrs.get("verdict"),
            "verdict_at_0.9": (
                verdict_at_0_9(t.user_attrs) if "lfr_at_0.9" in t.user_attrs else None
            ),
        }
        for t in study.trials
    ]

    extra_cells = None
    if args.extra_cells:
        extra_cells = []
        for cell in EXTRA_CELLS:
            cell_start = time.time()
            metrics = run_cv(
                X_liked,
                X_disliked,
                outlier_method=cell["outlier_method"],
                outlier_budget=cell["outlier_budget"],
                selection=cell["selection"],
                essentia_dims=essentia_dims,
            )
            logger.info(
                f"Extra cell {cell['name']} ({time.time() - cell_start:.0f}s): "
                f"{metrics}"
            )
            extra_cells.append(
                {
                    **cell,
                    "metrics": metrics,
                    "verdict": verdict(metrics),
                    "verdict_at_0.9": verdict_at_0_9(metrics),
                }
            )

    passing = [row for row in front if row["verdict"] != "fail"]
    summary = (
        f"{len(passing)}/{len(front)} front trials pass the guideline"
        if passing
        else "no front trial passes the guideline"
    )

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features_dir": args.features_dir,
        "focus": args.focus,
        "n_iterations": args.n_iterations,
        "essentia_dims": essentia_dims,
        "objectives": ["lfr_at_0.8", "dfa_at_0.775"],
        "guideline": {"lfr_cap": GUIDELINE_LFR_CAP, "dfa_cap": GUIDELINE_DFA_CAP},
        "baseline": baseline,
        "baseline_verdict": verdict(baseline),
        "baseline_verdict_at_0.9": verdict_at_0_9(baseline),
        "extra_cells": extra_cells,
        "pareto_front": front,
        "summary": summary,
        "trial_history": history,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Wrote {out_path} ({time.time() - start:.0f}s total) — {summary}")


if __name__ == "__main__":
    main()
