import json
import pickle

import pytest

import config
from core.modeling import DualOneClassModel
from models import ModelType


def test_subscription_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "data_path", tmp_path)

    sub = config.Subscription(
        estimate_from_channel_id=123,
        model_id=5,
        model_type=ModelType.EXCLUDE_DISLIKED,
    )
    config.add_subscription(1, sub)

    loaded = config.get_subscriptions(1)
    assert len(loaded) == 1
    assert loaded[0].estimate_from_channel_id == 123
    assert loaded[0].model_id == 5
    assert loaded[0].model_type == ModelType.EXCLUDE_DISLIKED


def test_subscription_default_model_type(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "data_path", tmp_path)

    sub = config.Subscription(
        estimate_from_channel_id=456,
        model_id=1,
    )
    assert sub.model_type == ModelType.INCLUDE_LIKED

    config.add_subscription(1, sub)
    loaded = config.get_subscriptions(1)
    assert loaded[0].model_type == ModelType.INCLUDE_LIKED


def test_subscription_deserialize_legacy_no_model_type(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "data_path", tmp_path)

    sub_file = tmp_path / "1" / "subscriptions.json"
    sub_file.parent.mkdir(parents=True, exist_ok=True)
    sub_file.write_text(json.dumps([{"estimate_from_channel_id": 789, "model_id": 3}]))

    loaded = config.get_subscriptions(1)
    assert len(loaded) == 1
    assert loaded[0].model_type == ModelType.INCLUDE_LIKED


def test_subscription_deserialize_int_model_type(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "data_path", tmp_path)

    sub_file = tmp_path / "1" / "subscriptions.json"
    sub_file.parent.mkdir(parents=True, exist_ok=True)
    sub_file.write_text(
        json.dumps([{"estimate_from_channel_id": 100, "model_id": 2, "model_type": 0}])
    )

    loaded = config.get_subscriptions(1)
    assert loaded[0].model_type == ModelType.EXCLUDE_DISLIKED


def test_update_subscription_model_type(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "data_path", tmp_path)

    sub = config.Subscription(
        estimate_from_channel_id=111,
        model_id=1,
        model_type=ModelType.INCLUDE_LIKED,
    )
    config.add_subscription(1, sub)

    config.update_subscription_model_type(1, 111, ModelType.EXCLUDE_DISLIKED)

    loaded = config.get_subscriptions(1)
    assert loaded[0].model_type == ModelType.EXCLUDE_DISLIKED


def test_remove_subscription(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "data_path", tmp_path)

    sub = config.Subscription(
        estimate_from_channel_id=111,
        model_id=1,
        model_type=ModelType.INCLUDE_LIKED,
    )
    config.add_subscription(1, sub)

    config.remove_subscription(1, 111)
    assert config.get_subscriptions(1) == []


def test_model_stats_ignores_legacy_model_type(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "data_path", tmp_path)

    model_dir = tmp_path / "1" / "models" / "42"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pkl").write_bytes(b"")

    model = DualOneClassModel(
        knn_k_min=3,
        knn_k_max=3,
        gmm_components_max=4,
        gmm_min_points_per_component=10,
        cv_folds=None,
    )
    artifact = {
        "model": model,
        "built_at": "2026-01-01",
        "embed_version": "v1",
        "segment_policy": "full",
        "stats": {},
        "thresholds": {"exclude_disliked": 0.5, "include_liked": 0.6},
        "config": {"knn_k": 3, "gmm_components": 4},
    }
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    (model_dir / "stats.json").write_text(
        json.dumps(
            {
                "model_type": "INCLUDE_LIKED",
                "liked_tracks_count": 10,
                "disliked_tracks_count": 5,
                "accuracy": 0.85,
                "thresholds": {"exclude_disliked": 0.5, "include_liked": 0.6},
                "embed_version": "v1",
            }
        )
    )

    model_obj = config.get_model(1, 42)
    assert model_obj is not None
    assert model_obj.model_id == 42
    assert model_obj.liked_tracks_count == 10
    assert not hasattr(model_obj, "model_type")


def test_model_stats_with_real_train_shape(tmp_path, monkeypatch):
    """Regression: stats.json as written by train.py:_build_profile contains
    many keys beyond the Model dataclass fields (liked_n, imbalance_ratio,
    recall targets, outliers_removed_*, ...). get_model must filter instead
    of crashing with TypeError."""
    monkeypatch.setattr(config, "data_path", tmp_path)

    model_dir = tmp_path / "1" / "models" / "7"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pkl").write_bytes(b"")

    stats = {
        "liked_n": 1376,
        "disliked_n": 603,
        "imbalance_ratio": 2.28,
        "liked_knn_k_used": 19,
        "liked_gmm_components_used": 17,
        "disliked_knn_k_used": 15,
        "disliked_gmm_components_used": 7,
        "cv_folds_used": 5,
        "exclude_disliked_recall_target": 0.9,
        "include_liked_recall_target": 0.8,
        "metrics_source": "cross_validated",
        "liked_tracks_count": 1376,
        "disliked_tracks_count": 603,
        "thresholds": {"exclude_disliked": 0.55, "include_liked": 0.65},
        "embed_version": "essentia-2.1b6.dev1438+profile-9d999a32d40d3078",
        "outliers_removed_liked": 12,
        "outliers_removed_disliked": 5,
        "include_liked_tp": 0.8,
        "include_liked_tn": 0.85,
        "include_liked_fp": 0.15,
        "include_liked_fn": 0.2,
        "exclude_disliked_tp": 0.9,
        "exclude_disliked_tn": 0.32,
        "exclude_disliked_fp": 0.68,
        "exclude_disliked_fn": 0.1,
    }
    (model_dir / "stats.json").write_text(json.dumps(stats))

    model_obj = config.get_model(1, 7)
    assert model_obj is not None
    assert model_obj.model_id == 7
    assert model_obj.liked_tracks_count == 1376
    assert model_obj.disliked_tracks_count == 603
    assert model_obj.metrics_source == "cross_validated"
    assert model_obj.thresholds == {"exclude_disliked": 0.55, "include_liked": 0.65}
    assert model_obj.embed_version == "essentia-2.1b6.dev1438+profile-9d999a32d40d3078"
    assert model_obj.outliers_removed_liked == 12
    assert model_obj.outliers_removed_disliked == 5
    assert model_obj.include_liked_tp == pytest.approx(0.8)
    assert model_obj.include_liked_tn == pytest.approx(0.85)
    assert model_obj.include_liked_fp == pytest.approx(0.15)
    assert model_obj.include_liked_fn == pytest.approx(0.2)
    assert model_obj.exclude_disliked_tp == pytest.approx(0.9)
    assert model_obj.exclude_disliked_tn == pytest.approx(0.32)
    assert model_obj.exclude_disliked_fp == pytest.approx(0.68)
    assert model_obj.exclude_disliked_fn == pytest.approx(0.1)
    assert not hasattr(model_obj, "liked_n")


def test_model_stats_legacy_accuracy_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "data_path", tmp_path)

    model_dir = tmp_path / "1" / "models" / "9"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pkl").write_bytes(b"")

    (model_dir / "stats.json").write_text(
        json.dumps(
            {
                "liked_tracks_count": 10,
                "disliked_tracks_count": 5,
                "accuracy": 0.85,
            }
        )
    )

    model_obj = config.get_model(1, 9)
    assert model_obj is not None
    assert model_obj.liked_tracks_count == 10
    assert not hasattr(model_obj, "accuracy")
    assert model_obj.include_liked_fp is None
    assert model_obj.exclude_disliked_fp is None
    assert model_obj.metrics_source is None


def test_get_model_none_when_model_pickle_missing(tmp_path, monkeypatch):
    """Regression: a model dir from the legacy {id}.pickle layout (stats.json
    present, model.pkl absent) must be reported as nonexistent instead of
    passing validation and exploding later at estimation time."""
    monkeypatch.setattr(config, "data_path", tmp_path)

    model_dir = tmp_path / "1" / "models" / "6177"
    model_dir.mkdir(parents=True)
    (model_dir / "6177.pickle").write_bytes(b"legacy layout")
    (model_dir / "stats.json").write_text(
        json.dumps({"liked_tracks_count": 10, "disliked_tracks_count": 5})
    )

    assert config.get_model(1, 6177) is None


def test_get_model_none_when_model_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "data_path", tmp_path)

    assert config.get_model(1, 12345) is None


def test_get_model_none_when_stats_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "data_path", tmp_path)

    model_dir = tmp_path / "1" / "models" / "42"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pkl").write_bytes(b"")

    assert config.get_model(1, 42) is None


def test_get_models_skips_models_without_stats(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "data_path", tmp_path)

    models_dir = tmp_path / "1" / "models"
    broken = models_dir / "7"
    broken.mkdir(parents=True)
    (broken / "model.pkl").write_bytes(b"")

    valid = models_dir / "8"
    valid.mkdir()
    (valid / "model.pkl").write_bytes(b"")
    (valid / "stats.json").write_text(
        json.dumps({"liked_tracks_count": 3, "disliked_tracks_count": 4})
    )

    models = config.get_models(1)
    assert [model.model_id for model in models] == [8]


def test_reset_training_executor_clears_global(monkeypatch):
    shutdowns = []

    class FakeExecutor:
        def shutdown(self, **kwargs):
            shutdowns.append(kwargs)

    monkeypatch.setattr(config, "_training_executor", FakeExecutor())
    config.reset_training_executor()
    assert config._training_executor is None
    assert len(shutdowns) == 1


def test_reset_estimation_executor_clears_global(monkeypatch):
    shutdowns = []

    class FakeExecutor:
        def shutdown(self, **kwargs):
            shutdowns.append(kwargs)

    monkeypatch.setattr(config, "_estimation_executor", FakeExecutor())
    config.reset_estimation_executor()
    assert config._estimation_executor is None
    assert len(shutdowns) == 1


def test_reset_executors_tolerate_none(monkeypatch):
    monkeypatch.setattr(config, "_training_executor", None)
    monkeypatch.setattr(config, "_estimation_executor", None)
    config.reset_training_executor()
    config.reset_estimation_executor()
    assert config._training_executor is None
    assert config._estimation_executor is None
