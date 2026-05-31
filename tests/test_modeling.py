import pickle

import numpy as np
import pytest

from core.modeling import DualOneClassModel, ModelLoadError, OneClassSetModel
from models import ModelType


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def liked_data(rng):
    return rng.normal(loc=0.0, scale=1.0, size=(60, 5))


@pytest.fixture
def disliked_data(rng):
    return rng.normal(loc=5.0, scale=1.0, size=(60, 5))


def test_one_class_set_model_fit_sets_attributes(rng):
    X = rng.normal(size=(30, 4))
    model = OneClassSetModel(knn_k=3, gmm_components=4)
    model.fit(X)

    assert model.knn is not None
    assert model.gmm is not None
    assert model.knn_calibrator is not None
    assert model.gmm_calibrator is not None
    assert model.X_fit is not None
    np.testing.assert_array_equal(model.X_fit, X)


def test_one_class_set_model_fit_raises_on_empty():
    model = OneClassSetModel()
    with pytest.raises(ValueError, match="Cannot fit on empty set"):
        model.fit(np.array([]).reshape(0, 5))


def test_one_class_set_model_score_keys(rng):
    X = rng.normal(size=(30, 4))
    model = OneClassSetModel(knn_k=3, gmm_components=4)
    model.fit(X)

    sample = X[0].reshape(1, -1)
    result = model.score(sample)

    assert "calibrated" in result
    assert "raw_knn" in result
    assert "raw_gmm_loglik" in result


def test_one_class_set_model_score_calibrated_range(rng):
    X = rng.normal(size=(30, 4))
    model = OneClassSetModel(knn_k=3, gmm_components=4)
    model.fit(X)

    scores = [model.score(X[i].reshape(1, -1))["calibrated"] for i in range(len(X))]
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_one_class_set_model_save_load(tmp_path, rng):
    X = rng.normal(size=(30, 4))
    model = OneClassSetModel(knn_k=3, gmm_components=4)
    model.fit(X)

    sample = X[:3]
    original_scores = [model.score(s.reshape(1, -1)) for s in sample]

    model.save(tmp_path / "oneclass")
    loaded = OneClassSetModel.load(tmp_path / "oneclass")

    loaded_scores = [loaded.score(s.reshape(1, -1)) for s in sample]

    for orig, loaded in zip(original_scores, loaded_scores):
        assert orig["calibrated"] == pytest.approx(loaded["calibrated"])
        assert orig["raw_knn"] == pytest.approx(loaded["raw_knn"])
        assert orig["raw_gmm_loglik"] == pytest.approx(loaded["raw_gmm_loglik"])


def test_dual_one_class_model_fit_computes_thresholds(liked_data, disliked_data):
    model = DualOneClassModel(knn_k=3, gmm_components=4)
    model.fit(liked_data, disliked_data)

    assert "mode_a" in model.thresholds
    assert "mode_b" in model.thresholds
    assert isinstance(model.thresholds["mode_a"], float)
    assert isinstance(model.thresholds["mode_b"], float)


def test_dual_one_class_model_predict_keys(liked_data, disliked_data):
    model = DualOneClassModel(knn_k=3, gmm_components=4)
    model.fit(liked_data, disliked_data)

    sample = liked_data[0].reshape(1, -1)
    result = model.predict(sample)

    assert "like" in result
    assert "dislike" in result
    assert "thresholds_at_build" in result
    assert result["like"]["calibrated"] is not None
    assert result["dislike"]["calibrated"] is not None


def test_dual_one_class_model_decide_exclude_disliked(liked_data, disliked_data):
    model = DualOneClassModel(knn_k=3, gmm_components=4)
    model.fit(liked_data, disliked_data)

    sample = liked_data[0].reshape(1, -1)
    scores = model.predict(sample)
    result = model.decide(scores, ModelType.EXCLUDE_DISLIKED)

    assert isinstance(result, bool)


def test_dual_one_class_model_decide_include_liked(liked_data, disliked_data):
    model = DualOneClassModel(knn_k=3, gmm_components=4)
    model.fit(liked_data, disliked_data)

    sample = liked_data[0].reshape(1, -1)
    scores = model.predict(sample)
    result = model.decide(scores, ModelType.INCLUDE_LIKED)

    assert isinstance(result, bool)


def test_dual_one_class_model_decide_raises_on_unknown_type(liked_data, disliked_data):
    model = DualOneClassModel(knn_k=3, gmm_components=4)
    model.fit(liked_data, disliked_data)

    sample = liked_data[0].reshape(1, -1)
    scores = model.predict(sample)

    with pytest.raises(ValueError, match="Unknown model_type"):
        model.decide(scores, 999)


def test_dual_one_class_model_save_load(tmp_path, liked_data, disliked_data):
    model = DualOneClassModel(knn_k=3, gmm_components=4)
    model.fit(liked_data, disliked_data)

    sample = liked_data[0].reshape(1, -1)
    original_scores = model.predict(sample)

    model.save(tmp_path / "dual")
    loaded = DualOneClassModel.load(tmp_path / "dual")

    loaded_scores = loaded.predict(sample)

    assert original_scores["like"]["calibrated"] == pytest.approx(
        loaded_scores["like"]["calibrated"]
    )
    assert original_scores["dislike"]["calibrated"] == pytest.approx(
        loaded_scores["dislike"]["calibrated"]
    )
    assert loaded.thresholds == model.thresholds


def test_dual_one_class_model_load_raises_on_incompatible_format(tmp_path):
    bad_path = tmp_path / "bad_model"
    bad_path.mkdir()
    with open(bad_path / "model.pkl", "wb") as f:
        pickle.dump({"not_model": True}, f)

    with pytest.raises(ModelLoadError, match="Incompatible model format"):
        DualOneClassModel.load(bad_path)


def test_isotonic_calibration_in_distribution_higher_than_outliers(rng):
    in_dist = rng.normal(loc=0.0, scale=1.0, size=(80, 5))
    model = OneClassSetModel(knn_k=5, gmm_components=8)
    model.fit(in_dist)

    in_dist_scores = [model.score(x.reshape(1, -1))["calibrated"] for x in in_dist[:20]]
    outlier_points = rng.normal(loc=10.0, scale=1.0, size=(20, 5))
    outlier_scores = [
        model.score(x.reshape(1, -1))["calibrated"] for x in outlier_points
    ]

    mean_in_dist = np.mean(in_dist_scores)
    mean_outlier = np.mean(outlier_scores)
    assert mean_in_dist > mean_outlier
