import numpy as np
import pytest

from benchmark.fused_rule_study import (
    CELLS,
    PARITY_ANCHORS,
    and_rule_metrics,
    fused_metrics,
    placebo_metrics,
    probe_metrics,
    rule_verdict,
    score_diagnostics,
)
from benchmark.gates_study import compute_metrics, run_cv

TINY_PARAMS = {
    "knn_k_min": 2,
    "knn_k_max": 4,
    "knn_k_scale": 0.5,
    "gmm_components_max": 2,
    "gmm_min_points_per_component": 5,
}


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def make_scores(rng, n=100):
    return {
        "like_on_liked": rng.uniform(0.6, 1.0, size=n),
        "like_on_disliked": rng.uniform(0.0, 0.4, size=n),
        "dislike_on_disliked": rng.uniform(0.6, 1.0, size=n),
        "dislike_on_liked": rng.uniform(0.0, 0.4, size=n),
    }


def test_fused_metrics_w0_matches_single_rule(rng):
    scores = {
        "like_on_liked": rng.uniform(size=80),
        "like_on_disliked": rng.uniform(size=40),
        "dislike_on_disliked": rng.uniform(size=40),
        "dislike_on_liked": rng.uniform(size=80),
    }
    fused = fused_metrics(scores, 0.0)
    single = compute_metrics(scores)
    for key, ref in single.items():
        assert fused[key] == pytest.approx(ref), key


def test_fused_metrics_extends_recall_targets(rng):
    scores = make_scores(rng)
    fused = fused_metrics(scores, 0.5)
    for rt in (0.7, 0.8, 0.85, 0.9):
        assert 0.0 <= fused[f"lfr_at_{rt}"] <= 1.0
    for rt in (0.775, 0.8):
        assert 0.0 <= fused[f"dfa_at_{rt}"] <= 1.0


def test_fusion_rescues_liked_periphery():
    # Liked "victims" have a high dislike score AND a high like score;
    # the single rule falsely excludes them, diff fusion must rescue them.
    scores = {
        "like_on_liked": np.full(50, 0.9),
        "like_on_disliked": np.full(50, 0.1),
        "dislike_on_disliked": np.full(50, 0.9),
        # half the liked set scores 0.95 on the dislike model (leakers,
        # above the 0.9 single-rule threshold)
        "dislike_on_liked": np.concatenate([np.full(25, 0.95), np.full(25, 0.05)]),
    }
    single = fused_metrics(scores, 0.0)
    fused = fused_metrics(scores, 1.0)
    assert single["lfr_at_0.8"] == pytest.approx(0.5)
    assert fused["lfr_at_0.8"] == pytest.approx(0.0)
    assert fused["auc_exclude"] == pytest.approx(1.0)


def test_and_rule_reaches_recall_and_beats_single():
    scores = {
        "like_on_liked": np.concatenate([np.full(50, 0.9), np.full(50, 0.5)]),
        "like_on_disliked": np.full(100, 0.1),
        "dislike_on_disliked": np.linspace(0.5, 1.0, 100),
        "dislike_on_liked": np.concatenate([np.full(50, 0.95), np.full(50, 0.1)]),
    }
    single = fused_metrics(scores, 0.0)
    and_m = and_rule_metrics(scores, recall_targets=(0.8,))
    assert and_m["recall_at_0.8"] >= 0.8
    assert and_m["lfr_at_0.8"] < single["lfr_at_0.8"]


def test_and_rule_fallback_when_target_unreachable():
    # tight grid: t1 at the disliked median, t2 at the 10th like-percentile
    # -> recall far below 0.9, fallback reports the loosest corner honestly
    scores = {
        "like_on_liked": np.full(20, 0.9),
        "like_on_disliked": np.linspace(0.0, 0.4, 20),
        "dislike_on_disliked": np.linspace(0.6, 1.0, 20),
        "dislike_on_liked": np.linspace(0.5, 0.7, 20),
    }
    and_m = and_rule_metrics(
        scores,
        recall_targets=(0.9,),
        t1_percentiles=(50.0,),
        t2_percentiles=(10.0,),
    )
    assert and_m["t1_pct_at_0.9"] == 50.0
    assert and_m["t2_pct_at_0.9"] == 10.0
    assert and_m["recall_at_0.9"] < 0.9


def test_probe_metrics_direction(rng):
    p_disliked = rng.uniform(0.7, 1.0, size=60)
    p_liked = rng.uniform(0.0, 0.3, size=60)
    m = probe_metrics(p_disliked, p_liked)
    assert m["auc_exclude"] == pytest.approx(1.0)
    assert m["lfr_at_0.85"] == pytest.approx(0.0)


def test_placebo_destroys_pairing_rescue():
    # victims pair a high dislike score with a high like rank; after
    # shuffling, roughly half the victims draw the unprotective 0.1 rank
    # and get excluded again
    scores = {
        "like_on_liked": np.array([0.95] * 25 + [0.1] * 25),
        "like_on_disliked": np.full(50, 0.05),
        "dislike_on_disliked": np.full(50, 0.9),
        "dislike_on_liked": np.array([0.95] * 25 + [0.05] * 25),
    }
    rng = np.random.default_rng(0)
    real = fused_metrics(scores, 1.0)
    placebo = placebo_metrics(scores, 1.0, rng)
    assert real["lfr_at_0.8"] == pytest.approx(0.0)
    assert placebo["lfr_at_0.8"] > 0.1


def test_score_diagnostics_clamp_fraction():
    scores = {
        "like_on_liked": np.array([0.2, 0.5, 0.9]),
        "like_on_disliked": np.array([0.0, 0.0, 0.4]),
        "dislike_on_disliked": np.array([0.3, 0.7, 0.8]),
        "dislike_on_liked": np.array([0.0, 0.1, 0.6]),
    }
    d = score_diagnostics(scores)
    assert d["disliked_like_score_clamped_frac"] == pytest.approx(2 / 3)
    assert d["liked_dislike_score_clamped_frac"] == pytest.approx(1 / 3)
    assert -1.0 <= d["liked_pairing_corr"] <= 1.0


def test_rule_verdict_labels():
    stretch = {"lfr_at_0.85": 0.10, "lfr_at_0.9": 0.18, "dfa_at_0.775": 0.08}
    passed = {"lfr_at_0.85": 0.18, "lfr_at_0.9": 0.30, "dfa_at_0.775": 0.15}
    fail_lfr = {"lfr_at_0.85": 0.25, "lfr_at_0.9": 0.40, "dfa_at_0.775": 0.10}
    fail_dfa = {"lfr_at_0.85": 0.10, "lfr_at_0.9": 0.15, "dfa_at_0.775": 0.25}
    assert rule_verdict(stretch) == "stretch"
    assert rule_verdict(passed) == "pass"
    assert rule_verdict(fail_lfr) == "fail"
    assert rule_verdict(fail_dfa) == "fail"


def test_cells_match_gates_study_extra_cells():
    from benchmark.gates_study import EXTRA_CELLS

    for cell in EXTRA_CELLS:
        assert CELLS[cell["name"]] == {
            k: cell[k] for k in ("outlier_method", "outlier_budget", "selection")
        }
    # anchored cells are a proper subset: the final-config study adds
    # selection-variant cells that have no parity reference
    assert set(PARITY_ANCHORS) < set(CELLS)


def test_final_config_cells_registered():
    from benchmark.gates_study import SELECTION_VARIANTS, _parse_selection

    for expected in ("per_welch64_ridge64", "per_quota32_ridge64", "quota64_shared"):
        assert expected in CELLS
    for name, cell in CELLS.items():
        like_sel, dis_sel = _parse_selection(cell["selection"])
        assert like_sel in SELECTION_VARIANTS
        assert dis_sel in SELECTION_VARIANTS


def test_check_parity_without_anchor_annotates_only():
    from benchmark.fused_rule_study import check_parity

    parity = check_parity("per_welch64_ridge64", {"lfr_at_0.8": 0.5})
    assert parity["ok"] is None
    assert parity["keys"] == {}


def test_run_cv_return_scores_roundtrip(rng):
    X_liked = rng.normal(loc=0.0, scale=1.0, size=(60, 80))
    X_disliked = rng.normal(loc=3.0, scale=1.0, size=(60, 80))
    kwargs = {
        "outlier_method": "knn",
        "outlier_budget": 0.05,
        "selection": "per:welch64/ridge_select64",
        "model_params": TINY_PARAMS,
        "seeds": (42,),
    }
    plain = run_cv(X_liked, X_disliked, **kwargs)
    with_scores = run_cv(X_liked, X_disliked, return_scores=True, **kwargs)
    assert plain is not None and with_scores is not None
    for key, ref in plain.items():
        assert with_scores["metrics"][key] == pytest.approx(ref), key

    (scores,) = with_scores["scores"]
    assert len(scores["like_on_liked"]) == 60
    assert len(scores["dislike_on_liked"]) == 60
    assert len(scores["like_on_disliked"]) == 60
    assert len(scores["dislike_on_disliked"]) == 60
    fused_w0 = fused_metrics(scores, 0.0)
    for key in plain:
        if key.endswith("_std"):
            continue
        assert fused_w0[key] == pytest.approx(plain[key]), key
