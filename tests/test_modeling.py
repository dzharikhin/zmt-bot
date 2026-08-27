import pickle

import numpy as np
import pytest

from core.modeling import DualOneClassModel, ModelLoadError, OneClassSetModel
from core.preprocessing import StandardizeSelectPreprocessor
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


class TestOneClassSetModelAdaptiveCapacity:
    def test_knn_k_adapts_to_n_large_set(self):
        model = OneClassSetModel(knn_k_min=5, knn_k_max=15, knn_k_scale=0.5)
        rng = np.random.default_rng(42)
        X = rng.normal(size=(1300, 5))
        model.fit(X)
        assert model.knn_k_used == 15

    def test_knn_k_adapts_to_n_medium_set(self):
        model = OneClassSetModel(knn_k_min=5, knn_k_max=15, knn_k_scale=0.5)
        assert model._effective_knn_k(600) == 12

    def test_knn_k_adapts_to_n_small_set(self):
        model = OneClassSetModel(knn_k_min=5, knn_k_max=15, knn_k_scale=0.5)
        assert model._effective_knn_k(50) == 5

    def test_knn_k_cannot_exceed_n_minus_1(self):
        model = OneClassSetModel(knn_k_min=5, knn_k_max=15, knn_k_scale=0.5)
        rng = np.random.default_rng(42)
        X = rng.normal(size=(6, 5))
        model.fit(X)
        assert model.knn_k_used == 5

    def test_knn_k_capped_at_n_minus_1_when_scale_demands_more(self):
        model = OneClassSetModel(knn_k_min=5, knn_k_max=15, knn_k_scale=0.5)
        rng = np.random.default_rng(42)
        X = rng.normal(size=(4, 5))
        model.fit(X)
        assert model.knn_k_used == 3

    def test_gmm_components_adapts_to_n_large_set(self):
        model = OneClassSetModel(gmm_components_max=16, gmm_min_points_per_component=40)
        assert model._effective_gmm_components(1300) == 16

    def test_gmm_components_adapts_to_n_medium_set(self):
        model = OneClassSetModel(gmm_components_max=16, gmm_min_points_per_component=40)
        assert model._effective_gmm_components(600) == 15

    def test_gmm_components_adapts_to_n_small_set(self):
        model = OneClassSetModel(gmm_components_max=16, gmm_min_points_per_component=40)
        assert model._effective_gmm_components(80) == 2

    def test_gmm_components_minimum_is_2(self):
        model = OneClassSetModel(gmm_components_max=16, gmm_min_points_per_component=40)
        assert model._effective_gmm_components(30) == 2

    def test_fit_sets_effective_params_on_instance(self, rng):
        X = rng.normal(size=(100, 5))
        model = OneClassSetModel(
            knn_k_min=5,
            knn_k_max=15,
            knn_k_scale=0.5,
            gmm_components_max=16,
            gmm_min_points_per_component=40,
        )
        model.fit(X)
        assert model.knn_k_used is not None
        assert model.gmm_components_used is not None
        assert model.knn_k_used == model._effective_knn_k(100)
        assert model.gmm_components_used == model._effective_gmm_components(100)


class TestOneClassSetModelBasic:
    def test_fit_sets_attributes(self, rng):
        X = rng.normal(size=(30, 4))
        model = OneClassSetModel(
            knn_k_min=3,
            knn_k_max=3,
            knn_k_scale=0.5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
        )
        model.fit(X)

        assert model.knn is not None
        assert model.gmm is not None
        assert model.knn_calibrator is not None
        assert model.gmm_calibrator is not None
        assert model.X_fit is not None
        np.testing.assert_array_equal(model.X_fit, X)

    def test_fit_raises_on_empty(self):
        model = OneClassSetModel()
        with pytest.raises(ValueError, match="Cannot fit on empty set"):
            model.fit(np.array([]).reshape(0, 5))

    def test_score_keys(self, rng):
        X = rng.normal(size=(30, 4))
        model = OneClassSetModel(
            knn_k_min=3,
            knn_k_max=3,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
        )
        model.fit(X)

        sample = X[0].reshape(1, -1)
        result = model.score(sample)

        assert "calibrated" in result
        assert "raw_knn" in result
        assert "raw_gmm_loglik" in result

    def test_score_calibrated_range(self, rng):
        X = rng.normal(size=(30, 4))
        model = OneClassSetModel(
            knn_k_min=3,
            knn_k_max=3,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
        )
        model.fit(X)

        scores = [model.score(X[i].reshape(1, -1))["calibrated"] for i in range(len(X))]
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_save_load(self, tmp_path, rng):
        X = rng.normal(size=(30, 4))
        model = OneClassSetModel(
            knn_k_min=3,
            knn_k_max=3,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
        )
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


class TestDualOneClassModelCVThresholds:
    def test_cv_thresholds_differ_from_in_sample(self, rng):
        X_liked = rng.normal(loc=0.0, scale=1.0, size=(100, 5))
        X_disliked = rng.normal(loc=5.0, scale=1.0, size=(100, 5))

        model_in_sample = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
            exclude_disliked_recall_target=0.90,
            include_liked_recall_target=0.80,
        )
        model_in_sample.fit(X_liked, X_disliked)

        model_cv = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=5,
            exclude_disliked_recall_target=0.90,
            include_liked_recall_target=0.80,
        )
        model_cv.fit(X_liked, X_disliked)

        assert model_in_sample.thresholds != model_cv.thresholds

    def test_cv_folds_none_uses_in_sample(self, rng):
        X_liked = rng.normal(loc=0.0, scale=1.0, size=(60, 5))
        X_disliked = rng.normal(loc=5.0, scale=1.0, size=(60, 5))

        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
        )
        model.fit(X_liked, X_disliked)

        assert model.stats["cv_folds_used"] is None

    def test_cv_folds_recorded_in_stats(self, rng):
        X_liked = rng.normal(loc=0.0, scale=1.0, size=(60, 5))
        X_disliked = rng.normal(loc=5.0, scale=1.0, size=(60, 5))

        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=5,
        )
        model.fit(X_liked, X_disliked)

        assert model.stats["cv_folds_used"] == 5

    def test_recall_targets_recorded_in_stats(self, rng):
        X_liked = rng.normal(loc=0.0, scale=1.0, size=(60, 5))
        X_disliked = rng.normal(loc=5.0, scale=1.0, size=(60, 5))

        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
            exclude_disliked_recall_target=0.85,
            include_liked_recall_target=0.75,
        )
        model.fit(X_liked, X_disliked)

        assert model.stats["exclude_disliked_recall_target"] == 0.85
        assert model.stats["include_liked_recall_target"] == 0.75

    def test_imbalance_ratio_in_stats(self, rng):
        X_liked = rng.normal(loc=0.0, scale=1.0, size=(130, 5))
        X_disliked = rng.normal(loc=5.0, scale=1.0, size=(60, 5))

        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
        )
        model.fit(X_liked, X_disliked)

        assert model.stats["imbalance_ratio"] == pytest.approx(2.17, abs=0.01)

    def test_per_side_effective_params_in_stats(self, rng):
        X_liked = rng.normal(loc=0.0, scale=1.0, size=(130, 5))
        X_disliked = rng.normal(loc=5.0, scale=1.0, size=(60, 5))

        model = DualOneClassModel(
            knn_k_min=5,
            knn_k_max=15,
            knn_k_scale=0.5,
            gmm_components_max=16,
            gmm_min_points_per_component=40,
            cv_folds=None,
        )
        model.fit(X_liked, X_disliked)

        assert model.stats["liked_knn_k_used"] == model.liked_model.knn_k_used
        assert model.stats["disliked_knn_k_used"] == model.dislike_model.knn_k_used
        assert (
            model.stats["liked_gmm_components_used"]
            == model.liked_model.gmm_components_used
        )
        assert (
            model.stats["disliked_gmm_components_used"]
            == model.dislike_model.gmm_components_used
        )

    def test_cv_folds_1_treated_as_in_sample(self, rng):
        X_liked = rng.normal(loc=0.0, scale=1.0, size=(60, 5))
        X_disliked = rng.normal(loc=5.0, scale=1.0, size=(60, 5))

        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=1,
        )
        model.fit(X_liked, X_disliked)

        assert model.stats["cv_folds_used"] is None


class TestDualOneClassModelBasic:
    def test_fit_computes_thresholds(self, liked_data, disliked_data):
        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=3,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
        )
        model.fit(liked_data, disliked_data)

        assert "exclude_disliked" in model.thresholds
        assert "include_liked" in model.thresholds
        assert isinstance(model.thresholds["exclude_disliked"], float)
        assert isinstance(model.thresholds["include_liked"], float)

    def test_predict_keys(self, liked_data, disliked_data):
        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=3,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
        )
        model.fit(liked_data, disliked_data)

        sample = liked_data[0].reshape(1, -1)
        result = model.predict(sample)

        assert "like" in result
        assert "dislike" in result
        assert "thresholds_at_build" in result
        assert result["like"]["calibrated"] is not None
        assert result["dislike"]["calibrated"] is not None

    def test_decide_exclude_disliked(self, liked_data, disliked_data):
        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=3,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
        )
        model.fit(liked_data, disliked_data)

        sample = liked_data[0].reshape(1, -1)
        scores = model.predict(sample)
        result = model.decide(scores, ModelType.EXCLUDE_DISLIKED)

        assert isinstance(result, bool)

    def test_decide_include_liked(self, liked_data, disliked_data):
        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=3,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
        )
        model.fit(liked_data, disliked_data)

        sample = liked_data[0].reshape(1, -1)
        scores = model.predict(sample)
        result = model.decide(scores, ModelType.INCLUDE_LIKED)

        assert isinstance(result, bool)

    def test_decide_raises_on_unknown_type(self, liked_data, disliked_data):
        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=3,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
        )
        model.fit(liked_data, disliked_data)

        sample = liked_data[0].reshape(1, -1)
        scores = model.predict(sample)

        with pytest.raises(ValueError, match="Unknown model_type"):
            model.decide(scores, 999)

    def test_decide_include_liked_positive_on_liked(self, liked_data, disliked_data):
        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=3,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
        )
        model.fit(liked_data, disliked_data)

        like_sample = liked_data[0].reshape(1, -1)
        like_scores = model.predict(like_sample)
        assert model.decide(like_scores, ModelType.INCLUDE_LIKED) is True

    def test_decide_include_liked_negative_on_disliked(self, liked_data, disliked_data):
        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=3,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
        )
        model.fit(liked_data, disliked_data)

        dislike_sample = disliked_data[0].reshape(1, -1)
        dislike_scores = model.predict(dislike_sample)
        assert model.decide(dislike_scores, ModelType.INCLUDE_LIKED) is False

    def test_decide_exclude_disliked_positive_on_disliked(
        self, liked_data, disliked_data
    ):
        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=3,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
        )
        model.fit(liked_data, disliked_data)

        dislike_sample = disliked_data[0].reshape(1, -1)
        dislike_scores = model.predict(dislike_sample)
        assert model.decide(dislike_scores, ModelType.EXCLUDE_DISLIKED) is True


class TestDualOneClassModelSchemaVersion:
    def test_save_load_round_trip(self, tmp_path, liked_data, disliked_data):
        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=3,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
        )
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

    def test_load_raises_on_missing_schema_version(self, tmp_path):
        bad_path = tmp_path / "bad_model"
        bad_path.mkdir()
        with open(bad_path / "model.pkl", "wb") as f:
            pickle.dump({"model": DualOneClassModel()}, f)

        with pytest.raises(ModelLoadError, match="too old"):
            DualOneClassModel.load(bad_path)

    def test_load_raises_on_wrong_schema_version(self, tmp_path):
        bad_path = tmp_path / "bad_model"
        bad_path.mkdir()
        with open(bad_path / "model.pkl", "wb") as f:
            pickle.dump({"schema_version": 1, "model": DualOneClassModel()}, f)

        with pytest.raises(ModelLoadError, match="no longer supported"):
            DualOneClassModel.load(bad_path)

    def test_load_raises_on_incompatible_format(self, tmp_path):
        bad_path = tmp_path / "bad_model"
        bad_path.mkdir()
        with open(bad_path / "model.pkl", "wb") as f:
            pickle.dump({"not_model": True}, f)

        with pytest.raises(ModelLoadError, match="Incompatible model format"):
            DualOneClassModel.load(bad_path)

    def test_save_load_with_preprocessor(self, tmp_path, liked_data, disliked_data):
        prep = StandardizeSelectPreprocessor(n_features=5)
        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=3,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
            preprocessor=prep,
        )
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


class TestOperatingMetrics:
    def test_separated_clusters_zero_cross_error(self, rng):
        X_liked = rng.normal(loc=0.0, scale=0.5, size=(150, 5))
        X_disliked = rng.normal(loc=20.0, scale=0.5, size=(150, 5))

        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=5,
        )
        model.fit(X_liked, X_disliked)

        assert model.operating_metrics["metrics_source"] == "cross_validated"
        assert model.operating_metrics["disliked_false_accept"] == pytest.approx(0.0)
        assert model.operating_metrics["liked_false_reject"] == pytest.approx(0.0)

    def test_identical_clusters_high_cross_error(self, rng):
        X_liked = rng.normal(loc=0.0, scale=1.0, size=(150, 5))
        X_disliked = rng.normal(loc=0.0, scale=1.0, size=(150, 5))

        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=5,
        )
        model.fit(X_liked, X_disliked)

        assert model.operating_metrics["disliked_false_accept"] > 0.4
        assert model.operating_metrics["liked_false_reject"] > 0.4

    def test_in_sample_fallback_sets_metrics_source(self, rng):
        X_liked = rng.normal(loc=0.0, scale=1.0, size=(60, 5))
        X_disliked = rng.normal(loc=5.0, scale=1.0, size=(60, 5))

        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
        )
        model.fit(X_liked, X_disliked)

        assert model.operating_metrics["metrics_source"] == "in_sample"
        assert model.operating_metrics["disliked_false_accept"] == pytest.approx(0.0)
        assert model.operating_metrics["liked_false_reject"] == pytest.approx(0.0)

    def test_operating_metrics_in_stats(self, rng):
        X_liked = rng.normal(loc=0.0, scale=1.0, size=(60, 5))
        X_disliked = rng.normal(loc=5.0, scale=1.0, size=(60, 5))

        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=5,
        )
        model.fit(X_liked, X_disliked)

        for key in ("disliked_false_accept", "liked_false_reject", "metrics_source"):
            assert model.stats[key] == model.operating_metrics[key]

    def test_false_accept_grows_with_overlap(self, rng):
        X_liked = rng.normal(loc=0.0, scale=1.0, size=(150, 5))
        X_disliked_far = rng.normal(loc=8.0, scale=1.0, size=(150, 5))
        X_disliked_near = rng.normal(loc=2.0, scale=1.0, size=(150, 5))

        def fit_with(X_disliked):
            model = DualOneClassModel(
                knn_k_min=3,
                knn_k_max=5,
                gmm_components_max=4,
                gmm_min_points_per_component=10,
                cv_folds=5,
            )
            model.fit(X_liked, X_disliked)
            return model

        far = fit_with(X_disliked_far)
        near = fit_with(X_disliked_near)

        assert (
            near.operating_metrics["disliked_false_accept"]
            > far.operating_metrics["disliked_false_accept"]
        )


class TestPerModelParams:
    def test_liked_override_reaches_liked_model_only(self, rng):
        X_liked = rng.normal(loc=0.0, scale=1.0, size=(130, 5))
        X_disliked = rng.normal(loc=5.0, scale=1.0, size=(60, 5))

        model = DualOneClassModel(
            knn_k_min=5,
            knn_k_max=15,
            knn_k_scale=0.5,
            gmm_components_max=16,
            gmm_min_points_per_component=40,
            cv_folds=None,
            liked_params={"knn_k_max": 8},
        )
        model.fit(X_liked, X_disliked)

        assert model.liked_model.knn_k_max == 8
        assert model.dislike_model.knn_k_max == 15
        assert model.liked_params["knn_k_max"] == 8
        assert model.disliked_params["knn_k_max"] == 15

    def test_no_overrides_equals_shared(self, rng):
        X_liked = rng.normal(loc=0.0, scale=1.0, size=(60, 5))
        X_disliked = rng.normal(loc=5.0, scale=1.0, size=(60, 5))

        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
        )
        model.fit(X_liked, X_disliked)

        assert model.liked_params == model.disliked_params
        assert model.liked_params["knn_k_min"] == 3

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError, match="Unknown per-model params"):
            DualOneClassModel(liked_params={"not_a_param": 1})

    def test_unknown_disliked_param_raises(self):
        with pytest.raises(ValueError, match="Unknown per-model params"):
            DualOneClassModel(disliked_params={"knn_k_tiny": 1})

    def test_cv_fold_models_use_owning_set_params(self, rng):
        X_liked = rng.normal(loc=0.0, scale=1.0, size=(130, 5))
        X_disliked = rng.normal(loc=5.0, scale=1.0, size=(130, 5))

        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=6,
            gmm_min_points_per_component=10,
            cv_folds=5,
            liked_params={"gmm_components_max": 2, "gmm_min_points_per_component": 65},
            disliked_params={"gmm_components_max": 10},
        )
        model.fit(X_liked, X_disliked)

        # full-data fit (n=130): liked 130//65=2 -> floor 2; disliked 130//10=13
        # capped at gmm_components_max=10
        assert model.stats["liked_gmm_components_used"] == 2
        assert model.stats["disliked_gmm_components_used"] == 10

    def test_save_records_per_model_params(self, tmp_path, rng):
        X_liked = rng.normal(loc=0.0, scale=1.0, size=(60, 5))
        X_disliked = rng.normal(loc=5.0, scale=1.0, size=(60, 5))

        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
            liked_params={"knn_k_max": 8},
            disliked_params={"knn_k_scale": 0.9},
        )
        model.fit(X_liked, X_disliked)
        model.save(tmp_path / "dual")

        loaded = DualOneClassModel.load(tmp_path / "dual")
        assert loaded.liked_params["knn_k_max"] == 8
        assert loaded.disliked_params["knn_k_scale"] == 0.9
        assert loaded.liked_params["knn_k_min"] == 3


class TestIsotonicCalibration:
    def test_in_distribution_higher_than_outliers(self, rng):
        in_dist = rng.normal(loc=0.0, scale=1.0, size=(80, 5))
        model = OneClassSetModel(
            knn_k_min=5,
            knn_k_max=5,
            gmm_components_max=8,
            gmm_min_points_per_component=10,
        )
        model.fit(in_dist)

        in_dist_scores = [
            model.score(x.reshape(1, -1))["calibrated"] for x in in_dist[:20]
        ]
        outlier_points = rng.normal(loc=10.0, scale=1.0, size=(20, 5))
        outlier_scores = [
            model.score(x.reshape(1, -1))["calibrated"] for x in outlier_points
        ]

        mean_in_dist = np.mean(in_dist_scores)
        mean_outlier = np.mean(outlier_scores)
        assert mean_in_dist > mean_outlier

    def test_in_distribution_vs_far_point_both_signals(self, rng):
        in_dist = rng.normal(loc=0.0, scale=1.0, size=(60, 5))
        far_point = rng.normal(loc=15.0, scale=1.0, size=(1, 5))
        model = OneClassSetModel(
            knn_k_min=5,
            knn_k_max=5,
            gmm_components_max=8,
            gmm_min_points_per_component=10,
        )
        model.fit(in_dist)

        in_dist_result = model.score(in_dist[0].reshape(1, -1))
        far_result = model.score(far_point)

        combined_in = in_dist_result["calibrated"]
        combined_far = far_result["calibrated"]

        assert combined_in > combined_far

        raw_knn_in = in_dist_result["raw_knn"]
        raw_knn_far = far_result["raw_knn"]
        assert raw_knn_in < raw_knn_far

        raw_gmm_in = in_dist_result["raw_gmm_loglik"]
        raw_gmm_far = far_result["raw_gmm_loglik"]
        assert raw_gmm_in > raw_gmm_far


class TestDualOneClassModelPerModelPreprocessors:
    @pytest.fixture
    def wide_data(self, rng):
        liked = rng.normal(loc=0.0, scale=1.0, size=(80, 30))
        disliked = rng.normal(loc=4.0, scale=1.0, size=(80, 30))
        return liked, disliked

    def test_per_model_fit_predict(self, wide_data):
        liked, disliked = wide_data
        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
            liked_preprocessor=StandardizeSelectPreprocessor(n_features=4),
            disliked_preprocessor=StandardizeSelectPreprocessor(n_features=6),
        )
        model.fit(liked, disliked)
        assert model.liked_model.X_fit.shape[1] == 4
        assert model.dislike_model.X_fit.shape[1] == 6
        result = model.predict(liked[0].reshape(1, -1))
        assert result["like"]["calibrated"] is not None
        assert result["dislike"]["calibrated"] is not None
        assert "liked_preprocessor" in model.stats
        assert "disliked_preprocessor" in model.stats

    def test_shared_preprocessor_untouched_when_both_overridden(self, wide_data):
        liked, disliked = wide_data
        shared = StandardizeSelectPreprocessor(n_features=5)
        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
            preprocessor=shared,
            liked_preprocessor=StandardizeSelectPreprocessor(n_features=4),
            disliked_preprocessor=StandardizeSelectPreprocessor(n_features=4),
        )
        model.fit(liked, disliked)
        assert not hasattr(shared, "selected_")

    def test_single_side_override_uses_shared_for_other(self, wide_data):
        liked, disliked = wide_data
        shared = StandardizeSelectPreprocessor(n_features=5)
        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
            preprocessor=shared,
            liked_preprocessor=StandardizeSelectPreprocessor(n_features=4),
        )
        model.fit(liked, disliked)
        assert hasattr(shared, "selected_")
        assert model.liked_model.X_fit.shape[1] == 4
        assert model.dislike_model.X_fit.shape[1] == 5

    def test_no_override_backward_compatible(self, wide_data):
        liked, disliked = wide_data
        shared = StandardizeSelectPreprocessor(n_features=5)
        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
            preprocessor=shared,
        )
        model.fit(liked, disliked)
        assert model.liked_model.X_fit.shape[1] == 5
        assert model.dislike_model.X_fit.shape[1] == 5

    def test_old_pickle_without_per_model_attrs(self, wide_data):
        liked, disliked = wide_data
        shared = StandardizeSelectPreprocessor(n_features=5)
        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
            preprocessor=shared,
        )
        model.fit(liked, disliked)
        # simulate a model pickled before per-model preprocessors existed
        del model.liked_preprocessor
        del model.disliked_preprocessor
        result = model.predict(liked[0].reshape(1, -1))
        assert result["like"]["calibrated"] is not None
        assert result["dislike"]["calibrated"] is not None

    def test_save_load_round_trip_per_model(self, tmp_path, wide_data):
        liked, disliked = wide_data
        model = DualOneClassModel(
            knn_k_min=3,
            knn_k_max=5,
            gmm_components_max=4,
            gmm_min_points_per_component=10,
            cv_folds=None,
            liked_preprocessor=StandardizeSelectPreprocessor(n_features=4),
            disliked_preprocessor=StandardizeSelectPreprocessor(n_features=6),
        )
        model.fit(liked, disliked)
        model.save(tmp_path / "dual")
        loaded = DualOneClassModel.load(tmp_path / "dual")
        sample = liked[0].reshape(1, -1)
        np.testing.assert_allclose(
            model.predict(sample)["like"]["calibrated"],
            loaded.predict(sample)["like"]["calibrated"],
        )
        np.testing.assert_allclose(
            model.predict(sample)["dislike"]["calibrated"],
            loaded.predict(sample)["dislike"]["calibrated"],
        )
