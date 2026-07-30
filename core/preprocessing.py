from typing import Protocol

import numpy as np


class Preprocessor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def transform(self, X: np.ndarray) -> np.ndarray: ...


class NoOpPreprocessor:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


class StandardizeSelectPreprocessor:
    def __init__(self, n_features: int):
        self.n_features = n_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean_ = X.mean(axis=0)
        self.std_ = np.where(X.std(axis=0) < 1e-9, 1.0, X.std(axis=0))
        Xs = (X - self.mean_) / self.std_
        Xl = Xs[y == 1]
        Xd = Xs[y == 0]
        pooled_std = (
            np.sqrt(
                (Xl.var(axis=0) * len(Xl) + Xd.var(axis=0) * len(Xd))
                / (len(Xl) + len(Xd))
            )
            + 1e-9
        )
        scores = np.abs(Xl.mean(axis=0) - Xd.mean(axis=0)) / pooled_std
        self.selected_ = np.argsort(scores)[::-1][: self.n_features]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_)[:, self.selected_]
