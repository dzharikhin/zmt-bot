import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from models import ModelType

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when model loading fails (incompatible format, corrupt file, etc.)."""


class OneClassSetModel:
    """One-class model for a single set (liked or disliked)

    Calibration: raw density scores are mapped to rank-percentile within the set's
    own distribution via IsotonicRegression. "calibrated" ≈ 1.0 means the point
    is well inside the set's density; ≈ 0.0 means it's at the fringe.
    """

    def __init__(self, knn_k: int = 5, gmm_components: int = 16):
        self.knn_k = knn_k
        self.gmm_components = gmm_components
        self.knn = None
        self.gmm = None
        self.knn_calibrator = None
        self.gmm_calibrator = None
        self.X_fit = None
        self._knn_score_range = None
        self._gmm_score_range = None

    def fit(self, X: np.ndarray):
        """Fit k-NN, GMM, and isotonic calibrators on full training data"""
        if len(X) == 0:
            raise ValueError("Cannot fit on empty set")

        X_fit = X
        self.X_fit = X_fit

        # Fit k-NN — k+1 neighbors, slice [:, 1:] to exclude self-distance
        self.knn = NearestNeighbors(n_neighbors=min(self.knn_k + 1, len(X_fit) - 1))
        self.knn.fit(X_fit)

        # Fit GMM with retry logic for numerical stability
        n_components = min(self.gmm_components, max(1, len(X_fit) // 2))
        reg_covar = 1e-4
        max_retries = 5
        retry_count = 0
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

        # Compute raw scores for the fit set (used as calibration reference)
        # k-NN: exclude self-distance by slicing [:, 1:]
        knn_dists_fit, _ = self.knn.kneighbors(X_fit)
        knn_scores_fit = knn_dists_fit[:, 1:].mean(axis=1)
        gmm_scores_fit = self.gmm.score_samples(X_fit)

        # Isotonic calibration: rank-percentile within the fit distribution
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
        # k-NN distance (lower = more similar to set)
        # Skip first neighbor to exclude self-distance, consistent with fit()
        knn_dist, _ = self.knn.kneighbors(X)
        knn_score = knn_dist[:, 1:].mean(axis=1)[0]

        # GMM log-likelihood (higher = more likely under set distribution)
        gmm_loglik = self.gmm.score_samples(X)[0]

        # Calibrate via isotonic regression
        # Out-of-range values get boundary calibration (0.0 or 1.0)
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
    """Two one-class models: one for liked, one for disliked"""

    def __init__(self, knn_k: int = 5, gmm_components: int = 16):
        self.knn_k = knn_k
        self.gmm_components = gmm_components
        self.liked_model = OneClassSetModel(knn_k, gmm_components)
        self.dislike_model = OneClassSetModel(knn_k, gmm_components)
        self.thresholds = {"mode_a": 0.55, "mode_b": 0.65}
        self.embed_version = None
        self.segment_policy = None
        self.stats = {}

    def fit(self, X_liked: np.ndarray, X_disliked: np.ndarray):
        """Fit both models and compute thresholds"""
        self.liked_model.fit(X_liked)
        self.dislike_model.fit(X_disliked)
        self._compute_thresholds(X_liked, X_disliked)
        self.stats = {
            "liked_n": len(X_liked),
            "disliked_n": len(X_disliked),
        }
        return self

    def _compute_thresholds(self, X_liked: np.ndarray, X_disliked: np.ndarray):
        """Determine thresholds for mode (a) and mode (b)

        Mode (a): Reject definite dislikes (90% recall on dislikes)
        Mode (b): Accept definite likes (80% recall on likes)
        """
        dislike_scores = [
            self.dislike_model.score(x.reshape(1, -1))["calibrated"] for x in X_disliked
        ]
        self.thresholds["mode_a"] = np.percentile(dislike_scores, 10)

        like_scores = [
            self.liked_model.score(x.reshape(1, -1))["calibrated"] for x in X_liked
        ]
        self.thresholds["mode_b"] = np.percentile(like_scores, 20)

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
            return bool(scores["dislike"]["calibrated"] < self.thresholds["mode_a"])
        elif model_type == ModelType.INCLUDE_LIKED:
            return bool(scores["like"]["calibrated"] > self.thresholds["mode_b"])
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def save(self, path: Path):
        """Save model and all metadata in a single pickle file"""
        path.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": self,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "embed_version": self.embed_version,
            "segment_policy": self.segment_policy,
            "stats": self.stats,
            "thresholds": self.thresholds,
            "config": {
                "knn_k": self.knn_k,
                "gmm_components": self.gmm_components,
            },
        }

        with open(path / "model.pkl", "wb") as f:
            pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path):
        """Load model artifact from disk

        No backward compatibility with old GMeans models — raises on
        incompatible pickles.
        """
        with open(path / "model.pkl", "rb") as f:
            artifact = pickle.load(f)

        if isinstance(artifact, dict) and "model" in artifact:
            model = artifact["model"]
        else:
            raise ModelLoadError(
                "Incompatible model format. Please retrain with /train."
            )

        return model
