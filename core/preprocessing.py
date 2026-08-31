from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class Preprocessor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def transform(self, X: np.ndarray) -> np.ndarray: ...


class NoOpPreprocessor:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


def welch_scores(Xs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-dim Welch t-statistic |mean_liked - mean_disliked| / pooled std"""
    Xl = Xs[y == 1]
    Xd = Xs[y == 0]
    pooled_std = (
        np.sqrt(
            (Xl.var(axis=0) * len(Xl) + Xd.var(axis=0) * len(Xd)) / (len(Xl) + len(Xd))
        )
        + 1e-9
    )
    return np.abs(Xl.mean(axis=0) - Xd.mean(axis=0)) / pooled_std


class StandardizeSelectPreprocessor:
    def __init__(self, n_features: int):
        self.n_features = n_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean_ = X.mean(axis=0)
        self.std_ = np.where(X.std(axis=0) < 1e-9, 1.0, X.std(axis=0))
        Xs = (X - self.mean_) / self.std_
        scores = welch_scores(Xs, y)
        self.selected_ = np.argsort(scores)[::-1][: self.n_features]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_)[:, self.selected_]


class RidgeSelectPreprocessor:
    """Standardize + select top-n dims by |logistic coefficient| (multivariate)

    Unlike Welch selection (per-dim mean separation), this ranks dims by how
    much a regularized linear discriminant uses them jointly.
    """

    def __init__(self, n_features: int = 64, C: float = 0.01):
        self.n_features = n_features
        self.C = C

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        lr = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000).fit(
            Xs, y
        )
        self.selected_ = np.argsort(np.abs(lr.coef_[0]))[::-1][: self.n_features]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler_.transform(X)[:, self.selected_]


class QuotaSelectPreprocessor:
    """Family-quota Welch selection; quotas scale with n_features

    families: list of (name, start, end) dim slices covering the whole vector
    (including the panns block). Within each family, dims are ranked by Welch
    t-statistic; leftovers are padded by global Welch rank.
    """

    def __init__(
        self,
        n_features: int = 64,
        families: list[tuple[str, int, int]] | None = None,
        family_quota: dict[str, int] | None = None,
    ):
        self.n_features = n_features
        self.families = families or [("default", 0, -1)]
        self.family_quota = family_quota or {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        scores = welch_scores(Xs, y)
        n_dims = X.shape[1]
        # merge dim indices per family name (families may be interleaved)
        family_dims: dict[str, list[int]] = {}
        for name, start, end in self.families:
            end = n_dims if end == -1 else end
            family_dims.setdefault(name, []).extend(range(start, end))
        default_quota = max(1, self.n_features // len(family_dims))
        scale = self.n_features / sum(
            self.family_quota.get(name, default_quota) for name in family_dims
        )
        chosen: list[int] = []
        for name, dims in family_dims.items():
            quota = max(1, round(self.family_quota.get(name, default_quota) * scale))
            idx = np.array(dims)
            top = idx[np.argsort(scores[idx])[::-1][:quota]]
            chosen.extend(top.tolist())
        remaining = [i for i in np.argsort(scores)[::-1] if i not in set(chosen)]
        chosen.extend(remaining[: self.n_features - len(chosen)])
        self.selected_ = np.array(sorted(chosen[: self.n_features]))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler_.transform(X)[:, self.selected_]
