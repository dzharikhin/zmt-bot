import json
import pickle

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
        "thresholds": {"mode_a": 0.5, "mode_b": 0.6},
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
                "thresholds": {"mode_a": 0.5, "mode_b": 0.6},
                "embed_version": "v1",
            }
        )
    )

    model_obj = config.get_model(1, 42)
    assert model_obj is not None
    assert model_obj.model_id == 42
    assert model_obj.liked_tracks_count == 10
    assert not hasattr(model_obj, "model_type")
