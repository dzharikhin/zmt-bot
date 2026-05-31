import numpy as np
import pytest

from core.outliers import detect_outliers


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_detect_outliers_normal_data_returns_all_true(rng):
    X = rng.normal(loc=0.0, scale=1.0, size=(200, 5))
    mask, report = detect_outliers(X, threshold=0.0)

    assert mask.dtype == bool
    assert mask.all()
    assert len(report) == 0


def test_detect_outliers_with_injected_outliers(rng):
    X_normal = rng.normal(loc=0.0, scale=1.0, size=(200, 5))
    n_outliers = 10
    X_outliers = rng.uniform(low=50.0, high=100.0, size=(n_outliers, 5))
    X = np.vstack([X_normal, X_outliers])

    mask, report = detect_outliers(X, threshold=0.05)

    assert not mask.all()
    assert len(report) > 0


def test_outlier_report_structure(rng):
    X_normal = rng.normal(loc=0.0, scale=1.0, size=(200, 5))
    X_outliers = rng.uniform(low=50.0, high=100.0, size=(10, 5))
    X = np.vstack([X_normal, X_outliers])

    _, report = detect_outliers(X, threshold=0.05)

    assert len(report) > 0
    for entry in report:
        assert "index" in entry
        assert "fused_score" in entry
        assert "knn_rank" in entry
        assert "iforest_rank" in entry
        assert isinstance(entry["index"], int)
        assert isinstance(entry["fused_score"], float)
        assert isinstance(entry["knn_rank"], float)
        assert isinstance(entry["iforest_rank"], float)


def test_detect_outliers_min_set_size_skips_small_set(rng):
    X = rng.normal(size=(30, 5))
    mask, report = detect_outliers(X, threshold=0.05, min_set_size=50)

    assert mask.all()
    assert len(report) == 0


def test_detect_outliers_threshold_zero_no_removal(rng):
    X_normal = rng.normal(loc=0.0, scale=1.0, size=(100, 5))
    X_outliers = rng.uniform(low=50.0, high=100.0, size=(10, 5))
    X = np.vstack([X_normal, X_outliers])

    mask, report = detect_outliers(X, threshold=0.0)

    assert mask.all()
    assert len(report) == 0


def test_detect_outliers_higher_threshold_more_outliers_removed(rng):
    X_normal = rng.normal(loc=0.0, scale=1.0, size=(200, 5))
    X_outliers = rng.uniform(low=50.0, high=100.0, size=(20, 5))
    X = np.vstack([X_normal, X_outliers])

    _, report_low = detect_outliers(X, threshold=0.05)
    _, report_high = detect_outliers(X, threshold=0.20)

    assert len(report_high) >= len(report_low)
