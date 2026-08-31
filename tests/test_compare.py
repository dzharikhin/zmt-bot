import numpy as np
import optuna
import pytest

from benchmark.compare import objective, optimize_embedding
from core.preprocessing import StandardizeSelectPreprocessor


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
        trial.suggest_int("knn_k_min", 3, 8)
        trial.suggest_int("knn_k_max", 8, 25)
        trial.suggest_float("knn_k_scale", 0.3, 1.0)
        trial.suggest_int("gmm_components_max", 8, 32)
        trial.suggest_int("gmm_min_points_per_component", 20, 80)
        trial.suggest_float("outlier_threshold", 0.01, 0.10)
        result = objective(trial, liked_data, disliked_data, 0.5, 0.5)
        assert isinstance(result, float)

    def test_objective_is_weighted_auc(self, liked_data, disliked_data):
        study = optuna.create_study()
        trial = study.ask()
        trial.suggest_int("knn_k_min", 3, 8)
        trial.suggest_int("knn_k_max", 8, 25)
        trial.suggest_float("knn_k_scale", 0.3, 1.0)
        trial.suggest_int("gmm_components_max", 8, 32)
        trial.suggest_int("gmm_min_points_per_component", 20, 80)
        trial.suggest_float("outlier_threshold", 0.01, 0.10)
        result = objective(trial, liked_data, disliked_data, 0.5, 0.5)
        assert 0.0 <= result <= 1.0

    def test_sets_user_attrs(self, liked_data, disliked_data):
        study = optuna.create_study()
        trial = study.ask()
        trial.suggest_int("knn_k_min", 3, 8)
        trial.suggest_int("knn_k_max", 8, 25)
        trial.suggest_float("knn_k_scale", 0.3, 1.0)
        trial.suggest_int("gmm_components_max", 8, 32)
        trial.suggest_int("gmm_min_points_per_component", 20, 80)
        trial.suggest_float("outlier_threshold", 0.01, 0.10)
        objective(trial, liked_data, disliked_data, 0.5, 0.5)
        frozen = study.trials[0]
        assert "auc_include" in frozen.user_attrs
        assert "auc_exclude" in frozen.user_attrs
        assert "disliked_false_accept" in frozen.user_attrs
        assert "liked_false_reject" in frozen.user_attrs
        assert "liked_recall" in frozen.user_attrs
        assert "disliked_recall" in frozen.user_attrs

    def test_separable_data_high_auc(self, rng):
        X_liked = rng.normal(loc=0.0, scale=0.5, size=(100, 5))
        X_disliked = rng.normal(loc=20.0, scale=0.5, size=(100, 5))
        study = optuna.create_study()
        trial = study.ask()
        trial.suggest_int("knn_k_min", 3, 8)
        trial.suggest_int("knn_k_max", 8, 25)
        trial.suggest_float("knn_k_scale", 0.3, 1.0)
        trial.suggest_int("gmm_components_max", 8, 32)
        trial.suggest_int("gmm_min_points_per_component", 20, 80)
        trial.suggest_float("outlier_threshold", 0.01, 0.10)
        result = objective(trial, X_liked, X_disliked, 0.5, 0.5)
        assert result > 0.7

    def test_too_few_samples_for_cv_returns_zero(self, rng):
        X_liked = rng.normal(size=(1, 5))
        X_disliked = rng.normal(size=(1, 5))
        study = optuna.create_study()
        trial = study.ask()
        trial.suggest_int("knn_k_min", 3, 8)
        trial.suggest_int("knn_k_max", 8, 25)
        trial.suggest_float("knn_k_scale", 0.3, 1.0)
        trial.suggest_int("gmm_components_max", 8, 32)
        trial.suggest_int("gmm_min_points_per_component", 20, 80)
        trial.suggest_float("outlier_threshold", 0.01, 0.10)
        result = objective(trial, X_liked, X_disliked, 0.5, 0.5)
        assert result == 0.0


class TestOptimizeEmbedding:
    def test_returns_expected_keys(self, liked_data, disliked_data):
        result = optimize_embedding(liked_data, disliked_data, 0.5, 0.5, n_iterations=3)
        assert "objective" in result
        assert "objective_top5_median" in result
        assert "best_params" in result
        assert "metrics_best" in result
        assert "metrics_top5_median" in result
        assert "n_trials" in result
        assert "trial_history" in result

    def test_best_params_keys(self, liked_data, disliked_data):
        result = optimize_embedding(liked_data, disliked_data, 0.5, 0.5, n_iterations=3)
        assert "knn_k_min" in result["best_params"]
        assert "knn_k_max" in result["best_params"]
        assert "knn_k_scale" in result["best_params"]
        assert "gmm_components_max" in result["best_params"]
        assert "gmm_min_points_per_component" in result["best_params"]
        assert "outlier_threshold" in result["best_params"]

    def test_best_params_types(self, liked_data, disliked_data):
        result = optimize_embedding(liked_data, disliked_data, 0.5, 0.5, n_iterations=3)
        assert isinstance(result["best_params"]["knn_k_min"], int)
        assert isinstance(result["best_params"]["knn_k_max"], int)
        assert isinstance(result["best_params"]["knn_k_scale"], float)
        assert isinstance(result["best_params"]["gmm_components_max"], int)
        assert isinstance(result["best_params"]["gmm_min_points_per_component"], int)
        assert isinstance(result["best_params"]["outlier_threshold"], float)

    def test_best_params_in_range(self, liked_data, disliked_data):
        result = optimize_embedding(liked_data, disliked_data, 0.5, 0.5, n_iterations=3)
        assert 3 <= result["best_params"]["knn_k_min"] <= 8
        assert 8 <= result["best_params"]["knn_k_max"] <= 25
        assert 0.3 <= result["best_params"]["knn_k_scale"] <= 1.0
        assert 8 <= result["best_params"]["gmm_components_max"] <= 32
        assert 20 <= result["best_params"]["gmm_min_points_per_component"] <= 80
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
            assert "auc_include" in trial_entry
            assert "auc_exclude" in trial_entry
            assert "disliked_false_accept" in trial_entry
            assert "liked_false_reject" in trial_entry
            assert "liked_recall" in trial_entry
            assert "disliked_recall" in trial_entry

    def test_objective_is_float(self, liked_data, disliked_data):
        result = optimize_embedding(liked_data, disliked_data, 0.5, 0.5, n_iterations=3)
        assert isinstance(result["objective"], float)

    def test_metrics_best_has_all_keys(self, liked_data, disliked_data):
        result = optimize_embedding(liked_data, disliked_data, 0.5, 0.5, n_iterations=3)
        assert "auc_include" in result["metrics_best"]
        assert "auc_exclude" in result["metrics_best"]
        assert "disliked_false_accept" in result["metrics_best"]
        assert "liked_false_reject" in result["metrics_best"]
        assert "liked_recall" in result["metrics_best"]
        assert "disliked_recall" in result["metrics_best"]

    def test_metrics_top5_median_has_all_keys(self, liked_data, disliked_data):
        result = optimize_embedding(liked_data, disliked_data, 0.5, 0.5, n_iterations=3)
        assert "auc_include" in result["metrics_top5_median"]
        assert "auc_exclude" in result["metrics_top5_median"]
        assert "disliked_false_accept" in result["metrics_top5_median"]
        assert "liked_false_reject" in result["metrics_top5_median"]
        assert "liked_recall" in result["metrics_top5_median"]
        assert "disliked_recall" in result["metrics_top5_median"]

    def test_deterministic_with_same_seed(self, liked_data, disliked_data):
        result1 = optimize_embedding(
            liked_data, disliked_data, 0.5, 0.5, n_iterations=3
        )
        result2 = optimize_embedding(
            liked_data, disliked_data, 0.5, 0.5, n_iterations=3
        )
        assert result1["best_params"] == result2["best_params"]
        assert result1["objective"] == pytest.approx(result2["objective"])

    def test_objective_with_preprocessor(self, liked_data, disliked_data):
        def make_preprocessor():
            return StandardizeSelectPreprocessor(n_features=5)

        study = optuna.create_study()
        trial = study.ask()
        trial.suggest_int("knn_k_min", 3, 8)
        trial.suggest_int("knn_k_max", 8, 25)
        trial.suggest_float("knn_k_scale", 0.3, 1.0)
        trial.suggest_int("gmm_components_max", 8, 32)
        trial.suggest_int("gmm_min_points_per_component", 20, 80)
        trial.suggest_float("outlier_threshold", 0.01, 0.10)
        result = objective(
            trial, liked_data, disliked_data, 0.5, 0.5, make_preprocessor
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
