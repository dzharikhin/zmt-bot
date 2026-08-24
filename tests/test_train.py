import pathlib

import pytest

import config
import train
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
