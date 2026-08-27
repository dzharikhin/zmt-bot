import numpy as np
import pytest

from core.preprocessing import (
    NoOpPreprocessor,
    QuotaSelectPreprocessor,
    RidgeSelectPreprocessor,
    StandardizeSelectPreprocessor,
    welch_scores,
)


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


class TestWelchScores:
    def test_ranks_separating_dim_highest(self, rng):
        Xl = rng.normal(size=(60, 5))
        Xd = rng.normal(size=(60, 5))
        Xd[:, 2] += 10.0
        X = np.vstack([Xl, Xd])
        y = np.concatenate([np.ones(60), np.zeros(60)])
        scores = welch_scores(X, y)
        assert np.argmax(scores) == 2

    def test_zero_separation_zero_score(self, rng):
        # identical sets -> identical means -> exactly zero scores
        X = rng.normal(size=(60, 5))
        X = np.vstack([X, X])
        y = np.concatenate([np.ones(60), np.zeros(60)])
        assert np.allclose(welch_scores(X, y), 0.0, atol=1e-9)


class TestRidgeSelectPreprocessor:
    def test_selects_jointly_discriminative_dims(self, rng):
        # dims 0 and 1 individually weak, jointly separable (XOR-ish means)
        n = 100
        a = rng.normal(size=(n, 6))
        b = rng.normal(size=(n, 6))
        a[:, 0] += 0.5
        a[:, 1] -= 0.5
        b[:, 0] -= 0.5
        b[:, 1] += 0.5
        X = np.vstack([a, b])
        y = np.concatenate([np.ones(n), np.zeros(n)])
        prep = RidgeSelectPreprocessor(n_features=2)
        prep.fit(X, y)
        assert sorted(prep.selected_.tolist()) in ([0, 1], [0, 4], [1, 4])

    def test_transform_shape_and_picklable(self, separable_data):
        import pickle

        X, y = separable_data
        prep = RidgeSelectPreprocessor(n_features=7)
        prep.fit(X, y)
        assert prep.transform(X).shape == (160, 7)
        clone = pickle.loads(pickle.dumps(prep))
        np.testing.assert_array_equal(prep.transform(X), clone.transform(X))


class TestQuotaSelectPreprocessor:
    def test_respects_family_quotas(self, separable_data):
        X, y = separable_data
        X = np.hstack([X, X[:, :4]])  # 24 dims: 20 ess + 4 panns
        fams = [("ess", 0, 20), ("panns", 20, -1)]
        prep = QuotaSelectPreprocessor(
            n_features=6, families=fams, family_quota={"ess": 4, "panns": 2}
        )
        prep.fit(X, y)
        assert len(prep.selected_) == 6
        n_ess = sum(1 for i in prep.selected_ if i < 20)
        assert n_ess == 4
        assert sum(1 for i in prep.selected_ if i >= 20) == 2

    def test_default_layout_single_block(self, separable_data):
        X, y = separable_data
        prep = QuotaSelectPreprocessor(n_features=5)
        prep.fit(X, y)
        assert prep.transform(X).shape == (160, 5)

    def test_quotas_scale_with_n_features(self, separable_data):
        X, y = separable_data
        X = np.hstack([X, X[:, :8]])  # 28 dims: 20 ess + 8 panns
        fams = [("ess", 0, 20), ("panns", 20, -1)]
        prep = QuotaSelectPreprocessor(
            n_features=24, families=fams, family_quota={"ess": 4, "panns": 2}
        )
        prep.fit(X, y)
        n_ess = sum(1 for i in prep.selected_ if i < 20)
        assert n_ess == 16  # 4 * (24/6)
        assert sum(1 for i in prep.selected_ if i >= 20) == 8  # 2 * (24/6)
