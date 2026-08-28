import pathlib

import numpy as np
import pytest

import config
import train
from core.preprocessing import (
    NoOpPreprocessor,
    QuotaSelectPreprocessor,
    RidgeSelectPreprocessor,
    StandardizeSelectPreprocessor,
)
from models import ModelType


def test_execute_estimation_missing_model_pickle(tmp_path, monkeypatch):
    """Regression: a model workdir without model.pkl (legacy layout or a
    failed train that only copied the profile) must raise a clear
    unrecoverable error instead of a raw FileNotFoundError from pickle."""
    monkeypatch.setattr(config, "data_path", tmp_path)

    store = config.get_model_store_path(1, 6177)
    assert store.model_workdir.exists()

    with pytest.raises(
        train.EstimationUnrecoverable,
        match=r"Model 6177 not found for user 1.*retrain with /train",
    ):
        train._execute_estimation(
            1,
            6177,
            pathlib.Path("/nonexistent/to-estimate.mp3"),
            ModelType.INCLUDE_LIKED,
        )


class TestBuildPreprocessor:
    def test_noop(self):
        assert isinstance(train.build_preprocessor("noop"), NoOpPreprocessor)
        assert isinstance(train.build_preprocessor(""), NoOpPreprocessor)

    def test_standardize_select_uses_config(self, monkeypatch):
        monkeypatch.setattr(config, "model_select_n_features", 17)
        prep = train.build_preprocessor("standardize_select")
        assert isinstance(prep, StandardizeSelectPreprocessor)
        assert prep.n_features == 17

    def test_welch_prefixed(self):
        prep = train.build_preprocessor("welch32")
        assert isinstance(prep, StandardizeSelectPreprocessor)
        assert prep.n_features == 32

    def test_ridge_prefixed(self):
        prep = train.build_preprocessor("ridge_select64")
        assert isinstance(prep, RidgeSelectPreprocessor)
        assert prep.n_features == 64

    def test_quota_covers_all_dims(self):
        prep = train.build_preprocessor("quota64")
        assert isinstance(prep, QuotaSelectPreprocessor)
        n_dims = sum(end - start for _, start, end in prep.families[:-1])
        assert prep.families[-1] == ("panns", n_dims, -1)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, n_dims + 32))
        y = np.concatenate([np.ones(30), np.zeros(30)])
        prep.fit(X, y)
        assert prep.transform(X).shape == (60, 64)
        n_pan = sum(1 for i in prep.selected_ if i >= n_dims)
        expected = round(
            train.PANNS_FAMILY_QUOTA["panns"]
            * 64
            / sum(train.PANNS_FAMILY_QUOTA.values())
        )
        assert expected <= n_pan <= expected + 1  # quota + possible leftover pad

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown preprocessor"):
            train.build_preprocessor("banana")
