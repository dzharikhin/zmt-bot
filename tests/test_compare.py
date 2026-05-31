import numpy as np
import optuna
import pytest

from benchmark.compare import objective, optimize_embedding


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def liked_data(rng):
    return rng.normal(loc=0.0, scale=1.0, size=(80, 10))


@pytest.fixture
def disliked_data(rng):
    return rng.normal(loc=5.0, scale=1.0, size=(80, 10))


class TestObjective:
    def test_returns_float(self, liked_data, disliked_data):
        study = optuna.create_study()
        trial = study.ask()
        result = objective(trial, liked_data, disliked_data, 0.5, 0.5)
        assert isinstance(result, float)

    def test_weighted_recall_in_range(self, liked_data, disliked_data):
        study = optuna.create_study()
        trial = study.ask()
        result = objective(trial, liked_data, disliked_data, 0.5, 0.5)
        assert 0.0 <= result <= 1.0

    def test_sets_user_attrs(self, liked_data, disliked_data):
        study = optuna.create_study()
        trial = study.ask()
        objective(trial, liked_data, disliked_data, 0.5, 0.5)
        frozen = study.trials[0]
        assert "mode_a_recall" in frozen.user_attrs
        assert "mode_b_recall" in frozen.user_attrs
        assert isinstance(frozen.user_attrs["mode_a_recall"], float)
        assert isinstance(frozen.user_attrs["mode_b_recall"], float)

    def test_equal_weights(self, liked_data, disliked_data):
        study = optuna.create_study()
        trial = study.ask()
        result = objective(trial, liked_data, disliked_data, 0.5, 0.5)
        frozen = study.trials[0]
        mode_a = frozen.user_attrs["mode_a_recall"]
        mode_b = frozen.user_attrs["mode_b_recall"]
        expected = 0.5 * mode_a + 0.5 * mode_b
        assert result == pytest.approx(expected)

    def test_asymmetric_weights(self, liked_data, disliked_data):
        study = optuna.create_study()
        trial = study.ask()
        result = objective(trial, liked_data, disliked_data, 0.8, 0.2)
        frozen = study.trials[0]
        mode_a = frozen.user_attrs["mode_a_recall"]
        mode_b = frozen.user_attrs["mode_b_recall"]
        expected = 0.8 * mode_a + 0.2 * mode_b
        assert result == pytest.approx(expected)

    def test_separable_data_high_recall(self, rng):
        X_liked = rng.normal(loc=0.0, scale=0.5, size=(100, 5))
        X_disliked = rng.normal(loc=20.0, scale=0.5, size=(100, 5))
        study = optuna.create_study()
        trial = study.ask()
        result = objective(trial, X_liked, X_disliked, 0.5, 0.5)
        assert result > 0.3

    def test_too_few_samples_for_cv_returns_zero(self, rng):
        X_liked = rng.normal(size=(1, 5))
        X_disliked = rng.normal(size=(1, 5))
        study = optuna.create_study()
        trial = study.ask()
        result = objective(trial, X_liked, X_disliked, 0.5, 0.5)
        assert result == 0.0


class TestOptimizeEmbedding:
    def test_returns_expected_keys(self, liked_data, disliked_data):
        result = optimize_embedding(liked_data, disliked_data, 0.5, 0.5, n_iterations=3)
        assert "best_params" in result
        assert "weighted_recall" in result
        assert "mode_a_recall" in result
        assert "mode_b_recall" in result
        assert "n_trials" in result
        assert "trial_history" in result

    def test_best_params_keys(self, liked_data, disliked_data):
        result = optimize_embedding(liked_data, disliked_data, 0.5, 0.5, n_iterations=3)
        assert "knn_k" in result["best_params"]
        assert "gmm_components" in result["best_params"]
        assert "outlier_threshold" in result["best_params"]

    def test_best_params_types(self, liked_data, disliked_data):
        result = optimize_embedding(liked_data, disliked_data, 0.5, 0.5, n_iterations=3)
        assert isinstance(result["best_params"]["knn_k"], int)
        assert isinstance(result["best_params"]["gmm_components"], int)
        assert isinstance(result["best_params"]["outlier_threshold"], float)

    def test_best_params_in_range(self, liked_data, disliked_data):
        result = optimize_embedding(liked_data, disliked_data, 0.5, 0.5, n_iterations=3)
        assert 3 <= result["best_params"]["knn_k"] <= 15
        assert 8 <= result["best_params"]["gmm_components"] <= 32
        assert 0.01 <= result["best_params"]["outlier_threshold"] <= 0.10

    def test_n_trials_matches(self, liked_data, disliked_data):
        result = optimize_embedding(liked_data, disliked_data, 0.5, 0.5, n_iterations=5)
        assert result["n_trials"] == 5
        assert len(result["trial_history"]) == 5

    def test_trial_history_structure(self, liked_data, disliked_data):
        result = optimize_embedding(liked_data, disliked_data, 0.5, 0.5, n_iterations=3)
        for trial_entry in result["trial_history"]:
            assert "params" in trial_entry
            assert "value" in trial_entry
            assert "mode_a_recall" in trial_entry
            assert "mode_b_recall" in trial_entry

    def test_weighted_recall_is_float(self, liked_data, disliked_data):
        result = optimize_embedding(liked_data, disliked_data, 0.5, 0.5, n_iterations=3)
        assert isinstance(result["weighted_recall"], float)

    def test_recalls_are_float(self, liked_data, disliked_data):
        result = optimize_embedding(liked_data, disliked_data, 0.5, 0.5, n_iterations=3)
        assert isinstance(result["mode_a_recall"], float)
        assert isinstance(result["mode_b_recall"], float)

    def test_deterministic_with_same_seed(self, liked_data, disliked_data):
        result1 = optimize_embedding(
            liked_data, disliked_data, 0.5, 0.5, n_iterations=3
        )
        result2 = optimize_embedding(
            liked_data, disliked_data, 0.5, 0.5, n_iterations=3
        )
        assert result1["best_params"] == result2["best_params"]
        assert result1["weighted_recall"] == pytest.approx(result2["weighted_recall"])
