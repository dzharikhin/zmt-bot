import logging

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def detect_outliers(
    X: np.ndarray,
    threshold: float = 0.05,
    knn_k: int = 5,
    n_estimators: int = 200,
    min_set_size: int = 50,
) -> tuple[np.ndarray, list[dict]]:
    """Detect outliers using k-NN + IsolationForest rank fusion

    Args:
        X: Feature matrix (n_samples, n_features)
        threshold: Fraction of outliers to remove (percentile)
        knn_k: Number of neighbors for k-NN
        n_estimators: Number of trees for IsolationForest
        min_set_size: Minimum set size to apply outlier detection

    Returns:
        mask: Boolean array (True = keep, False = outlier)
        outlier_report: List of outlier details
    """
    if len(X) < min_set_size:
        logger.warning(
            f"Set size {len(X)} < min_set_size {min_set_size}, skipping outlier detection"
        )
        return np.ones(len(X), dtype=bool), []

    if threshold <= 0.0:
        return np.ones(len(X), dtype=bool), []

    knn = NearestNeighbors(n_neighbors=knn_k)
    knn.fit(X)
    knn_dists, _ = knn.kneighbors(X)
    knn_scores = knn_dists.mean(axis=1)

    iforest = IsolationForest(
        n_estimators=n_estimators, random_state=42, contamination="auto"
    )
    iforest.fit_predict(X)
    iforest_scores = -iforest.score_samples(X)

    knn_ranks = stats.rankdata(knn_scores, method="average") / len(X)
    iforest_ranks = stats.rankdata(iforest_scores, method="average") / len(X)

    fused_scores = (knn_ranks + iforest_ranks) / 2

    mask = fused_scores < (1.0 - threshold)

    outlier_indices = np.where(~mask)[0]
    outlier_report = [
        {
            "index": int(i),
            "fused_score": float(fused_scores[i]),
            "knn_rank": float(knn_ranks[i]),
            "iforest_rank": float(iforest_ranks[i]),
        }
        for i in outlier_indices
    ]

    return mask, outlier_report
