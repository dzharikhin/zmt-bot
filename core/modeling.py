import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from models import ModelType

logger = logging.getLogger(__name__)

_MODEL_SCHEMA_VERSION = 4


class ModelLoadError(Exception):
    """Raised when model loading fails (incompatible format, corrupt file, etc.)."""


class OneClassSetModel:
    """One-class model for a single set (liked or disliked)

    Calibration: raw density scores are mapped to rank-percentile within the set's
    own distribution via IsotonicRegression. "calibrated" ≈ 1.0 means the point
    is well inside the set's density; ≈ 0.0 means it's at the fringe.

    Model capacity (knn_k, gmm_components) is adaptively derived from training
    set size to avoid overfitting on small sets and underfitting on large ones.
    """

    def __init__(
        self,
        knn_k_min: int = 5,
        knn_k_max: int = 15,
        knn_k_scale: float = 0.5,
        gmm_components_max: int = 16,
        gmm_min_points_per_component: int = 40,
    ):
        self.knn_k_min = knn_k_min
        self.knn_k_max = knn_k_max
        self.knn_k_scale = knn_k_scale
        self.gmm_components_max = gmm_components_max
        self.gmm_min_points_per_component = gmm_min_points_per_component

        self.knn = None
        self.gmm = None
        self.knn_calibrator = None
        self.gmm_calibrator = None
        self.X_fit = None
        self._knn_score_range = None
        self._gmm_score_range = None

        self.knn_k_used: int | None = None
        self.gmm_components_used: int | None = None

    def _effective_knn_k(self, n: int) -> int:
        k = round(self.knn_k_scale * np.sqrt(n))
        k = max(self.knn_k_min, min(self.knn_k_max, k))
        return min(k, n - 1)

    def _effective_gmm_components(self, n: int) -> int:
        by_density = max(1, n // self.gmm_min_points_per_component)
        return max(2, min(self.gmm_components_max, by_density))

    def fit(self, X: np.ndarray):
        """Fit k-NN, GMM, and isotonic calibrators on full training data"""
        if len(X) == 0:
            raise ValueError("Cannot fit on empty set")

        X_fit = X
        self.X_fit = X_fit
        n = len(X_fit)

        knn_k_eff = self._effective_knn_k(n)
        gmm_n_eff = self._effective_gmm_components(n)
        self.knn_k_used = knn_k_eff
        self.gmm_components_used = gmm_n_eff

        logger.info(
            f"OneClassSetModel.fit: n={n}, knn_k={knn_k_eff}, "
            f"gmm_components={gmm_n_eff}"
        )

        self.knn = NearestNeighbors(n_neighbors=knn_k_eff + 1)
        self.knn.fit(X_fit)

        reg_covar = 1e-4
        max_retries = 5
        retry_count = 0
        n_components = gmm_n_eff
        while retry_count <= max_retries:
            try:
                self.gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type="diag",
                    random_state=42,
                    n_init=3,
                    reg_covar=reg_covar,
                )
                self.gmm.fit(X_fit)
                break
            except ValueError:
                retry_count += 1
                if retry_count <= max_retries:
                    n_components = max(1, n_components // 2)
                    reg_covar = min(1e-2, reg_covar * 10)
                    if retry_count == max_retries:
                        n_components = 1
                        reg_covar = 1e-2
                        self.gmm = GaussianMixture(
                            n_components=n_components,
                            covariance_type="diag",
                            random_state=42,
                            n_init=3,
                            reg_covar=reg_covar,
                        )
                        self.gmm.fit(X_fit)
        self.gmm_components_used = n_components

        knn_dists_fit, _ = self.knn.kneighbors(X_fit)
        knn_scores_fit = knn_dists_fit[:, 1:].mean(axis=1)
        gmm_scores_fit = self.gmm.score_samples(X_fit)

        knn_sorted = np.sort(knn_scores_fit)
        knn_targets = np.arange(1, len(knn_sorted) + 1) / len(knn_sorted)
        self.knn_calibrator = IsotonicRegression(y_min=0.0, y_max=1.0)
        self.knn_calibrator.fit(knn_sorted, knn_targets)
        self._knn_score_range = (float(knn_sorted[0]), float(knn_sorted[-1]))

        gmm_sorted = np.sort(gmm_scores_fit)
        gmm_targets = np.arange(1, len(gmm_sorted) + 1) / len(gmm_sorted)
        self.gmm_calibrator = IsotonicRegression(y_min=0.0, y_max=1.0)
        self.gmm_calibrator.fit(gmm_sorted, gmm_targets)
        self._gmm_score_range = (float(gmm_sorted[0]), float(gmm_sorted[-1]))

        return self

    def score(self, X: np.ndarray) -> dict:
        """Return raw and calibrated scores

        Returns:
            Dict with calibrated (mean of knn+gmm calibrated), raw_knn, raw_gmm_loglik
        """
        knn_dist, _ = self.knn.kneighbors(X)
        knn_score = knn_dist[:, 1:].mean(axis=1)[0]

        gmm_loglik = self.gmm.score_samples(X)[0]

        knn_cal_raw = self.knn_calibrator.predict([knn_score])[0]
        gmm_cal_raw = self.gmm_calibrator.predict([gmm_loglik])[0]

        knn_calibrated = self._clamp_calibrated(
            knn_cal_raw, knn_score, self._knn_score_range, "knn"
        )
        gmm_calibrated = self._clamp_calibrated(
            gmm_cal_raw, gmm_loglik, self._gmm_score_range, "gmm"
        )
        calibrated = (knn_calibrated + gmm_calibrated) / 2

        return {
            "calibrated": calibrated,
            "raw_knn": float(knn_score),
            "raw_gmm_loglik": float(gmm_loglik),
        }

    @staticmethod
    def _clamp_calibrated(cal_raw, raw_score, score_range, direction):
        """Clamp isotonic regression output for out-of-range inputs

        Args:
            cal_raw: Raw isotonic regression prediction (may be NaN)
            raw_score: The input raw score
            score_range: Tuple of (min, max) from calibration data
            direction: "knn" (higher raw = lower calibrated) or
                       "gmm" (higher raw = higher calibrated)
        """
        if not np.isnan(cal_raw):
            return float(np.clip(cal_raw, 0.0, 1.0))

        lo, hi = score_range
        if raw_score < lo:
            return 1.0 if direction == "knn" else 0.0
        return 0.0 if direction == "knn" else 1.0

    def save(self, path: Path):
        """Save model artifacts"""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path):
        """Load model from disk"""
        with open(path / "model.pkl", "rb") as f:
            return pickle.load(f)


class DualOneClassModel:
    """Two one-class models: one for liked, one for disliked

    Thresholds are computed via k-fold cross-validated out-of-fold scoring
    when cv_folds >= 2, giving honest recall guarantees. Falls back to
    in-sample percentiles when cv_folds is None or < 2.
    """

    def __init__(
        self,
        knn_k_min: int = 5,
        knn_k_max: int = 15,
        knn_k_scale: float = 0.5,
        gmm_components_max: int = 16,
        gmm_min_points_per_component: int = 40,
        cv_folds: int | None = None,
        exclude_disliked_recall_target: float = 0.90,
        include_liked_recall_target: float = 0.80,
    ):
        self.knn_k_min = knn_k_min
        self.knn_k_max = knn_k_max
        self.knn_k_scale = knn_k_scale
        self.gmm_components_max = gmm_components_max
        self.gmm_min_points_per_component = gmm_min_points_per_component
        self.cv_folds = cv_folds
        self.exclude_disliked_recall_target = exclude_disliked_recall_target
        self.include_liked_recall_target = include_liked_recall_target

        self.liked_model = OneClassSetModel(
            knn_k_min=knn_k_min,
            knn_k_max=knn_k_max,
            knn_k_scale=knn_k_scale,
            gmm_components_max=gmm_components_max,
            gmm_min_points_per_component=gmm_min_points_per_component,
        )
        self.dislike_model = OneClassSetModel(
            knn_k_min=knn_k_min,
            knn_k_max=knn_k_max,
            knn_k_scale=knn_k_scale,
            gmm_components_max=gmm_components_max,
            gmm_min_points_per_component=gmm_min_points_per_component,
        )
        self.thresholds = {"exclude_disliked": 0.55, "include_liked": 0.65}
        self.embed_version = None
        self.segment_policy = None
        self.stats = {}

    def fit(self, X_liked: np.ndarray, X_disliked: np.ndarray):
        """Fit both models and compute thresholds"""
        self.liked_model.fit(X_liked)
        self.dislike_model.fit(X_disliked)
        self._compute_thresholds(X_liked, X_disliked)

        n_liked = len(X_liked)
        n_disliked = len(X_disliked)
        n_min = max(1, min(n_liked, n_disliked))
        imbalance_ratio = round(max(n_liked, n_disliked) / n_min, 2)

        self.stats = {
            "liked_n": n_liked,
            "disliked_n": n_disliked,
            "imbalance_ratio": imbalance_ratio,
            "liked_knn_k_used": self.liked_model.knn_k_used,
            "liked_gmm_components_used": self.liked_model.gmm_components_used,
            "disliked_knn_k_used": self.dislike_model.knn_k_used,
            "disliked_gmm_components_used": self.dislike_model.gmm_components_used,
            "cv_folds_used": (
                self.cv_folds if self.cv_folds and self.cv_folds >= 2 else None
            ),
            "exclude_disliked_recall_target": self.exclude_disliked_recall_target,
            "include_liked_recall_target": self.include_liked_recall_target,
        }
        return self

    def _cv_scores(self, X: np.ndarray) -> np.ndarray:
        """Out-of-fold calibrated scores via k-fold CV

        Each fold trains a fresh OneClassSetModel (same hyperparams) and scores
        the held-out partition. The concatenated held-out scores approximate
        the distribution a new point would face at inference time.
        """
        n_splits = min(self.cv_folds, len(X))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_scores = np.empty(len(X), dtype=np.float64)

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            fold_model = OneClassSetModel(
                knn_k_min=self.knn_k_min,
                knn_k_max=self.knn_k_max,
                knn_k_scale=self.knn_k_scale,
                gmm_components_max=self.gmm_components_max,
                gmm_min_points_per_component=self.gmm_min_points_per_component,
            )
            fold_model.fit(X_train)
            for i, idx in enumerate(test_idx):
                oof_scores[idx] = fold_model.score(X_test[i].reshape(1, -1))[
                    "calibrated"
                ]

        return oof_scores

    def _compute_thresholds(self, X_liked: np.ndarray, X_disliked: np.ndarray):
        """Determine thresholds for exclude_disliked and include_liked

        exclude_disliked: Reject definite dislikes (recall_target on dislikes)
        include_liked: Accept definite likes (recall_target on likes)

        When cv_folds >= 2, thresholds are derived from out-of-fold scores
        for honest recall guarantees. Otherwise in-sample scores are used.
        """
        use_cv = self.cv_folds is not None and self.cv_folds >= 2

        if use_cv:
            dislike_scores = self._cv_scores(X_disliked)
            like_scores = self._cv_scores(X_liked)
        else:
            dislike_scores = np.array(
                [
                    self.dislike_model.score(x.reshape(1, -1))["calibrated"]
                    for x in X_disliked
                ]
            )
            like_scores = np.array(
                [
                    self.liked_model.score(x.reshape(1, -1))["calibrated"]
                    for x in X_liked
                ]
            )

        self.thresholds["exclude_disliked"] = float(
            np.percentile(
                dislike_scores, 100 * (1 - self.exclude_disliked_recall_target)
            )
        )
        self.thresholds["include_liked"] = float(
            np.percentile(like_scores, 100 * (1 - self.include_liked_recall_target))
        )

    def predict(self, X: np.ndarray) -> dict:
        """Score a track against both models"""
        return {
            "like": self.liked_model.score(X),
            "dislike": self.dislike_model.score(X),
            "thresholds_at_build": self.thresholds,
        }

    def decide(self, scores: dict, model_type: ModelType) -> bool:
        """Apply decision logic based on model type"""
        if model_type == ModelType.EXCLUDE_DISLIKED:
            return bool(
                scores["dislike"]["calibrated"] < self.thresholds["exclude_disliked"]
            )
        elif model_type == ModelType.INCLUDE_LIKED:
            return bool(scores["like"]["calibrated"] > self.thresholds["include_liked"])
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def save(self, path: Path):
        """Save model and all metadata in a single pickle file"""
        path.mkdir(parents=True, exist_ok=True)

        artifact = {
            "schema_version": _MODEL_SCHEMA_VERSION,
            "model": self,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "embed_version": self.embed_version,
            "segment_policy": self.segment_policy,
            "stats": self.stats,
            "thresholds": self.thresholds,
            "config": {
                "knn_k_min": self.knn_k_min,
                "knn_k_max": self.knn_k_max,
                "knn_k_scale": self.knn_k_scale,
                "gmm_components_max": self.gmm_components_max,
                "gmm_min_points_per_component": self.gmm_min_points_per_component,
                "cv_folds": self.cv_folds,
                "exclude_disliked_recall_target": self.exclude_disliked_recall_target,
                "include_liked_recall_target": self.include_liked_recall_target,
            },
        }

        with open(path / "model.pkl", "wb") as f:
            pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path):
        """Load model artifact from disk

        Requires schema_version 2. Raises ModelLoadError on incompatible
        formats (missing schema_version, old v1 pickles, corrupt files).
        """
        with open(path / "model.pkl", "rb") as f:
            artifact = pickle.load(f)

        if isinstance(artifact, dict) and "schema_version" in artifact:
            if artifact["schema_version"] != _MODEL_SCHEMA_VERSION:
                raise ModelLoadError(
                    f"Model schema version {artifact.get('schema_version')} "
                    f"is no longer supported (required: {_MODEL_SCHEMA_VERSION}). "
                    f"Please retrain with /train."
                )
            return artifact["model"]

        if isinstance(artifact, dict) and "model" in artifact:
            raise ModelLoadError(
                "Model schema version is too old. Please retrain with /train."
            )

        raise ModelLoadError("Incompatible model format. Please retrain with /train.")
