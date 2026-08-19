import numpy as np
import optuna
import pytest

from benchmark.dfa_gate_study import (
    DFA_TARGET,
    RECALL_FLOOR,
    compute_metrics,
    decide_shipment,
    dfa_at_recall,
    dfa_cap_curve,
    evaluate_gate,
    exclude_curve,
    include_curve,
    objective,
    recall_at_dfa,
    run_cv,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _separable_scores(rng, n_liked=200, n_disliked=100):
    return {
        "like_on_liked": rng.uniform(0.8, 1.0, size=n_liked),
        "like_on_disliked": rng.uniform(0.0, 0.2, size=n_disliked),
        "dislike_on_disliked": rng.uniform(0.8, 1.0, size=n_disliked),
        "dislike_on_liked": rng.uniform(0.0, 0.2, size=n_liked),
    }


class TestIncludeCurve:
    def test_rows_and_keys(self, rng):
        scores = _separable_scores(rng)
        rows = include_curve(
            scores["like_on_liked"], scores["like_on_disliked"], (0.80, 0.75)
        )
        assert len(rows) == 2
        for row in rows:
            for key in (
                "recall_target",
                "threshold",
                "liked_recall",
                "disliked_false_accept",
                "disliked_false_accept_se",
            ):
                assert key in row

    def test_lower_recall_target_lowers_dfa(self, rng):
        rng2 = np.random.default_rng(7)
        s_ll = rng2.uniform(0.0, 1.0, size=1376)
        s_ld = 0.7 * s_ll[rng2.permutation(len(s_ll))][:603] + 0.15 * rng2.uniform(
            0.0, 1.0, size=603
        )
        rows = include_curve(s_ll, s_ld, (0.80, 0.75, 0.70))
        dfas = [r["disliked_false_accept"] for r in rows]
        assert dfas[0] >= dfas[1] >= dfas[2]

    def test_separable_data_zero_dfa(self, rng):
        scores = _separable_scores(rng)
        rows = include_curve(
            scores["like_on_liked"], scores["like_on_disliked"], (0.80,)
        )
        assert rows[0]["disliked_false_accept"] == pytest.approx(0.0)

    def test_se_formula(self, rng):
        scores = _separable_scores(rng, n_liked=400, n_disliked=100)
        rows = include_curve(
            scores["like_on_liked"], scores["like_on_disliked"], (0.80,)
        )
        dfa = rows[0]["disliked_false_accept"]
        assert rows[0]["disliked_false_accept_se"] == pytest.approx(
            np.sqrt(dfa * (1 - dfa) / 100)
        )


class TestDfaCapCurve:
    def test_rows_and_keys(self, rng):
        scores = _separable_scores(rng)
        rows = dfa_cap_curve(
            scores["like_on_liked"], scores["like_on_disliked"], (0.16, 0.15, 0.14)
        )
        assert len(rows) == 3
        for row in rows:
            for key in (
                "dfa_cap",
                "threshold",
                "liked_recall",
                "disliked_false_accept",
            ):
                assert key in row

    def test_tighter_cap_keeps_threshold_monotone(self, rng):
        rng2 = np.random.default_rng(11)
        s_ll = rng2.uniform(0.0, 1.0, size=1376)
        s_ld = rng2.uniform(0.0, 0.9, size=603)
        rows = dfa_cap_curve(s_ll, s_ld, (0.16, 0.15, 0.14))
        thresholds = [r["threshold"] for r in rows]
        assert thresholds[0] <= thresholds[1] <= thresholds[2]

    def test_cap_bounds_measured_dfa(self, rng):
        rng2 = np.random.default_rng(13)
        s_ll = rng2.uniform(0.0, 1.0, size=1376)
        s_ld = rng2.uniform(0.0, 0.9, size=603)
        rows = dfa_cap_curve(s_ll, s_ld, (0.15,))
        # linear percentile interpolation allows one sample of slack
        # (91/603 = .1509 at cap .15) — the reason DFA_TARGET buffers
        # below the 0.15 gate
        assert rows[0]["disliked_false_accept"] <= 0.15 + 1 / 603 + 1e-9


class TestExcludeCurve:
    def test_rows_and_keys(self, rng):
        scores = _separable_scores(rng)
        rows = exclude_curve(
            scores["dislike_on_disliked"], scores["dislike_on_liked"], (0.90, 0.85)
        )
        assert len(rows) == 2
        for row in rows:
            for key in (
                "recall_target",
                "threshold",
                "disliked_recall",
                "liked_false_reject",
            ):
                assert key in row

    def test_separable_data_zero_lfr(self, rng):
        scores = _separable_scores(rng)
        rows = exclude_curve(
            scores["dislike_on_disliked"], scores["dislike_on_liked"], (0.90,)
        )
        assert rows[0]["liked_false_reject"] == pytest.approx(0.0)


class TestOperatingPoints:
    def test_recall_at_dfa_separable_full_recall(self, rng):
        scores = _separable_scores(rng)
        point = recall_at_dfa(
            scores["like_on_liked"], scores["like_on_disliked"], DFA_TARGET
        )
        # recall is the free quantity; dFA is pinned to the cap by
        # construction (threshold anchored on the disliked scores)
        assert point["liked_recall"] == pytest.approx(1.0)
        assert point["disliked_false_accept"] == pytest.approx(
            DFA_TARGET, abs=1 / len(scores["like_on_disliked"])
        )

    def test_dfa_at_recall_separable_zero_dfa(self, rng):
        scores = _separable_scores(rng)
        point = dfa_at_recall(
            scores["like_on_liked"], scores["like_on_disliked"], RECALL_FLOOR
        )
        assert point["disliked_false_accept"] == pytest.approx(0.0)


class TestComputeMetrics:
    def test_separable_scores(self, rng):
        metrics = compute_metrics(_separable_scores(rng))
        assert metrics["auc_include"] > 0.99
        assert metrics["auc_exclude"] > 0.99
        assert metrics["recall_at_dfa_target"] > 0.99
        # B-mode operating point pins dFA at the target by construction
        # (one-sample quantization slack: 15/100 = .150 at cap .145)
        assert metrics["dfa_at_dfa_target"] == pytest.approx(DFA_TARGET, abs=1 / 100)
        assert metrics["dfa_at_recall_floor"] < 0.01
        assert metrics["dfa_at_recall_80"] < 0.01

    def test_overlapping_scores_degenerate(self, rng):
        scores = {
            "like_on_liked": rng.uniform(0.0, 1.0, size=100),
            "like_on_disliked": rng.uniform(0.0, 1.0, size=100),
            "dislike_on_disliked": rng.uniform(0.0, 1.0, size=100),
            "dislike_on_liked": rng.uniform(0.0, 1.0, size=100),
        }
        metrics = compute_metrics(scores)
        assert 0.3 < metrics["auc_include"] < 0.7
        assert 0.3 < metrics["auc_exclude"] < 0.7

    def test_keys(self, rng):
        metrics = compute_metrics(_separable_scores(rng))
        for key in (
            "auc_include",
            "auc_exclude",
            "recall_at_dfa_target",
            "dfa_at_dfa_target",
            "dfa_at_recall_floor",
            "dfa_at_recall_80",
        ):
            assert key in metrics


class TestEvaluateGate:
    def test_all_pass(self):
        gate = evaluate_gate(0.76, 0.86, 0.79)
        assert gate["passed"] is True
        assert all(gate["checks"].values())

    def test_recall_floor_fails(self):
        gate = evaluate_gate(0.74, 0.86, 0.79)
        assert gate["passed"] is False
        assert gate["checks"]["recall_floor"] is False

    def test_auc_floors_fail(self):
        assert evaluate_gate(0.76, 0.84, 0.79)["passed"] is False
        assert evaluate_gate(0.76, 0.86, 0.78)["passed"] is False


class TestDecideShipment:
    def test_ship_c_on_clear_improvement(self):
        winner_gate = {"passed": True, "checks": {}}
        baseline_gate = {"passed": True, "checks": {}}
        verdict, _ = decide_shipment(0.80, winner_gate, 0.75, baseline_gate, 0.005)
        assert verdict == "ship_C"

    def test_ship_baseline_when_within_margin(self):
        winner_gate = {"passed": True, "checks": {}}
        baseline_gate = {"passed": True, "checks": {}}
        verdict, _ = decide_shipment(0.752, winner_gate, 0.75, baseline_gate, 0.005)
        assert verdict == "ship_baseline_A"

    def test_ship_c_when_only_winner_passes(self):
        winner_gate = {"passed": True, "checks": {}}
        baseline_gate = {"passed": False, "checks": {}}
        verdict, _ = decide_shipment(0.75, winner_gate, 0.70, baseline_gate, 0.005)
        assert verdict == "ship_C"

    def test_no_pass(self):
        winner_gate = {"passed": False, "checks": {}}
        baseline_gate = {"passed": False, "checks": {}}
        verdict, _ = decide_shipment(0.80, winner_gate, 0.70, baseline_gate, 0.005)
        assert verdict == "no_pass"


class TestRunCv:
    def test_separable_data_shape_and_quality(self, rng):
        X_liked = rng.normal(loc=0.0, scale=0.5, size=(80, 10))
        X_disliked = rng.normal(loc=10.0, scale=0.5, size=(80, 10))
        scores = run_cv(
            X_liked,
            X_disliked,
            liked_model_params={
                "knn_k_min": 3,
                "knn_k_max": 5,
                "knn_k_scale": 0.5,
                "gmm_components_max": 4,
                "gmm_min_points_per_component": 10,
            },
            disliked_model_params={
                "knn_k_min": 3,
                "knn_k_max": 5,
                "knn_k_scale": 0.5,
                "gmm_components_max": 4,
                "gmm_min_points_per_component": 10,
            },
            liked_outlier_threshold=0.05,
            disliked_outlier_threshold=0.05,
        )
        assert scores is not None
        for key in (
            "like_on_liked",
            "like_on_disliked",
            "dislike_on_disliked",
            "dislike_on_liked",
        ):
            assert len(scores[key]) == 80
        metrics = compute_metrics(scores)
        assert metrics["auc_include"] > 0.9
        assert metrics["auc_exclude"] > 0.9

    def test_too_few_samples_returns_none(self, rng):
        result = run_cv(
            rng.normal(size=(1, 5)),
            rng.normal(size=(1, 5)),
            liked_model_params={},
            disliked_model_params={},
            liked_outlier_threshold=0.05,
            disliked_outlier_threshold=0.05,
        )
        assert result is None


class TestObjective:
    def _make_trial(self):
        study = optuna.create_study()
        trial = study.ask()
        trial.suggest_int("liked_knn_k_min", 3, 8)
        trial.suggest_int("liked_knn_k_max", 8, 25)
        trial.suggest_float("liked_knn_k_scale", 0.3, 1.0)
        trial.suggest_int("liked_gmm_components_max", 8, 32)
        trial.suggest_int("liked_gmm_min_points_per_component", 20, 80)
        trial.suggest_float("liked_outlier_threshold", 0.01, 0.10)
        return trial

    def test_returns_float_in_range(self, rng):
        X_liked = rng.normal(loc=0.0, scale=0.5, size=(80, 10))
        X_disliked = rng.normal(loc=10.0, scale=0.5, size=(80, 10))
        trial = self._make_trial()
        result = objective(trial, X_liked, X_disliked)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_sets_user_attrs(self, rng):
        X_liked = rng.normal(loc=0.0, scale=0.5, size=(80, 10))
        X_disliked = rng.normal(loc=10.0, scale=0.5, size=(80, 10))
        trial = self._make_trial()
        objective(trial, X_liked, X_disliked)
        frozen = trial.study.trials[0]
        for key in (
            "auc_include",
            "auc_exclude",
            "recall_at_dfa_target",
            "dfa_at_dfa_target",
            "dfa_at_recall_floor",
            "dfa_at_recall_80",
        ):
            assert key in frozen.user_attrs
