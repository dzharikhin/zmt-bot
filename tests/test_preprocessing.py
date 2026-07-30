import numpy as np
import pytest

from core.preprocessing import NoOpPreprocessor, StandardizeSelectPreprocessor


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def separable_data(rng):
    X_liked = rng.normal(loc=0.0, scale=1.0, size=(80, 20))
    X_disliked = rng.normal(loc=5.0, scale=1.0, size=(80, 20))
    y_liked = np.ones(len(X_liked))
    y_disliked = np.zeros(len(X_disliked))
    X = np.concatenate([X_liked, X_disliked])
    y = np.concatenate([y_liked, y_disliked])
    return X, y


class TestStandardizeSelectPreprocessor:
    def test_transform_shape(self, separable_data):
        X, y = separable_data
        prep = StandardizeSelectPreprocessor(n_features=10)
        prep.fit(X, y)
        X_transformed = prep.transform(X)
        assert X_transformed.shape == (160, 10)

    def test_selects_discriminative_features(self, separable_data):
        X, y = separable_data
        prep = StandardizeSelectPreprocessor(n_features=5)
        prep.fit(X, y)
        assert len(prep.selected_) == 5
        assert all(0 <= idx < X.shape[1] for idx in prep.selected_)

    def test_deterministic(self, separable_data):
        X, y = separable_data
        prep1 = StandardizeSelectPreprocessor(n_features=5)
        prep1.fit(X, y)
        prep2 = StandardizeSelectPreprocessor(n_features=5)
        prep2.fit(X, y)
        np.testing.assert_array_equal(prep1.selected_, prep2.selected_)

    def test_fit_transform_consistency(self, separable_data):
        X, y = separable_data
        prep = StandardizeSelectPreprocessor(n_features=10)
        prep.fit(X, y)
        X_transformed_1 = prep.transform(X)
        X_transformed_2 = prep.transform(X)
        np.testing.assert_array_equal(X_transformed_1, X_transformed_2)


class TestNoOpPreprocessor:
    def test_identity(self):
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, size=50)
        prep = NoOpPreprocessor()
        prep.fit(X, y)
        X_transformed = prep.transform(X)
        np.testing.assert_array_equal(X_transformed, X)
