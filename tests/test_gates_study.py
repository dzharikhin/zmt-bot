import numpy as np
import pytest

from benchmark.gates_study import (
    ESSENTIA_DIMS,
    EXTRA_CELLS,
    FOCUS_OUTLIER_METHODS,
    OUTLIER_METHODS,
    PER_MODEL_FOCUS,
    SELECTION_VARIANTS,
    _parse_selection,
    compute_metrics,
    family_layout,
    load_features,
    make_preprocessor,
    run_cv,
    verdict,
    verdict_at_0_9,
)

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


def make_sets(rng, n=60, d=80):
    X_liked = rng.normal(loc=0.0, scale=1.0, size=(n, d))
    X_disliked = rng.normal(loc=3.0, scale=1.0, size=(n, d))
    return X_liked, X_disliked


def test_essentia_dims_constant():
    assert ESSENTIA_DIMS == 4380


@pytest.mark.parametrize("method", sorted(OUTLIER_METHODS))
def test_outlier_methods_valid_mask(rng, method):
    X = rng.normal(size=(120, 30))
    mask = OUTLIER_METHODS[method](X, 0.10)
    assert mask.dtype == bool
    assert mask.sum() > 0
    assert (~mask).sum() > 0
    assert mask.shape == (120,)


def test_outlier_budget_monotone(rng):
    X = rng.normal(size=(120, 30))
    removed_low = int((~OUTLIER_METHODS["knn"](X, 0.05)).sum())
    removed_high = int((~OUTLIER_METHODS["knn"](X, 0.12)).sum())
    assert removed_high >= removed_low


@pytest.mark.parametrize("selection", sorted(SELECTION_VARIANTS))
def test_preprocessors_shapes(rng, selection):
    d = ESSENTIA_DIMS + 16
    X = rng.normal(size=(80, d))
    y = np.concatenate([np.ones(40), np.zeros(40)])
    prep = make_preprocessor(selection)
    prep.fit(X, y)
    out = prep.transform(X)
    assert out.shape[0] == 80
    assert 0 < out.shape[1] <= 128


def test_make_preprocessor_unknown_name():
    with pytest.raises(ValueError):
        make_preprocessor("nope")


def test_parse_selection_shared_and_per_model():
    assert _parse_selection("welch64") == ("welch64", "welch64")
    assert _parse_selection("per:welch64/ridge_select64") == (
        "welch64",
        "ridge_select64",
    )
    assert _parse_selection("per:quota64/ridge_select128") == (
        "quota64",
        "ridge_select128",
    )


def test_focus_space_registered():
    for selection in PER_MODEL_FOCUS:
        like_sel, dis_sel = _parse_selection(selection)
        assert like_sel in SELECTION_VARIANTS
        assert dis_sel in SELECTION_VARIANTS
    assert set(FOCUS_OUTLIER_METHODS) <= set(OUTLIER_METHODS)


def test_quota_scales_with_n_features(rng):
    d = ESSENTIA_DIMS + 16
    X = rng.normal(size=(80, d))
    y = np.concatenate([np.ones(40), np.zeros(40)])
    for name, expected in (("quota32", 32), ("quota64", 64), ("quota128", 128)):
        prep = make_preprocessor(name)
        prep.fit(X, y)
        assert prep.transform(X).shape[1] == expected


def test_run_cv_per_model_selection(rng):
    X_liked = rng.normal(loc=0.0, scale=1.0, size=(60, 80))
    X_disliked = rng.normal(loc=3.0, scale=1.0, size=(60, 80))
    metrics = run_cv(
        X_liked,
        X_disliked,
        outlier_method="knn",
        outlier_budget=0.05,
        selection="per:welch64/ridge_select64",
        model_params=TINY_PARAMS,
        seeds=(42,),
    )
    assert metrics is not None
    assert metrics["lfr_at_0.8"] <= 0.20
    assert metrics["dfa_at_0.775"] <= 0.20


def test_compute_metrics_perfect_scores():
    scores = {
        "like_on_liked": np.linspace(0.9, 1.0, 50),
        "like_on_disliked": np.linspace(0.0, 0.1, 50),
        "dislike_on_disliked": np.linspace(0.9, 1.0, 50),
        "dislike_on_liked": np.linspace(0.0, 0.1, 50),
    }
    m = compute_metrics(scores)
    assert m["auc_include"] == pytest.approx(1.0)
    assert m["auc_exclude"] == pytest.approx(1.0)
    assert m["lfr_at_0.8"] == pytest.approx(0.0)
    assert m["dfa_at_0.775"] == pytest.approx(0.0)


def test_compute_metrics_chance_scores():
    rng = np.random.default_rng(0)
    s = rng.uniform(size=100)
    scores = {
        "like_on_liked": s[:50],
        "like_on_disliked": s[50:],
        "dislike_on_disliked": s[:50],
        "dislike_on_liked": s[50:],
    }
    m = compute_metrics(scores)
    assert m["auc_include"] == pytest.approx(0.5, abs=0.2)
    assert m["auc_exclude"] == pytest.approx(0.5, abs=0.2)


def test_verdict_labels():
    stretch = {"lfr_at_0.8": 0.10, "dfa_at_0.775": 0.07}
    guideline = {"lfr_at_0.8": 0.18, "dfa_at_0.775": 0.15}
    fail = {"lfr_at_0.8": 0.30, "dfa_at_0.775": 0.10}
    fail2 = {"lfr_at_0.8": 0.10, "dfa_at_0.775": 0.30}
    assert verdict(stretch) == "stretch"
    assert verdict(guideline) == "guideline"
    assert verdict(fail) == "fail"
    assert verdict(fail2) == "fail"


def test_run_cv_returns_metrics(rng):
    X_liked, X_disliked = make_sets(rng)
    metrics = run_cv(
        X_liked,
        X_disliked,
        outlier_method="knn",
        outlier_budget=0.05,
        selection="welch64",
        model_params=TINY_PARAMS,
        seeds=(42,),
    )
    assert metrics is not None
    for key in ("lfr_at_0.8", "dfa_at_0.775", "auc_include", "auc_exclude"):
        assert 0.0 <= metrics[key] <= 1.0


def test_run_cv_separable_sets_pass_guideline(rng):
    X_liked, X_disliked = make_sets(rng, n=80, d=40)
    metrics = run_cv(
        X_liked,
        X_disliked,
        outlier_method="knn",
        outlier_budget=0.05,
        selection="welch64",
        model_params=TINY_PARAMS,
        seeds=(42,),
    )
    assert metrics["lfr_at_0.8"] <= 0.20
    assert metrics["dfa_at_0.775"] <= 0.20


def test_load_features_missing_dir():
    with pytest.raises(Exception):
        load_features("/nonexistent-features-dir")


def test_verdict_at_0_9_labels():
    stretch = {"lfr_at_0.8": 0.3, "lfr_at_0.9": 0.10, "dfa_at_0.775": 0.07}
    guideline = {"lfr_at_0.8": 0.3, "lfr_at_0.9": 0.18, "dfa_at_0.775": 0.15}
    fail = {"lfr_at_0.8": 0.2, "lfr_at_0.9": 0.30, "dfa_at_0.775": 0.10}
    assert verdict_at_0_9(stretch) == "stretch"
    assert verdict_at_0_9(guideline) == "guideline"
    assert verdict_at_0_9(fail) == "fail"
    assert verdict(stretch) == "fail"


def test_extra_cells_registered():
    assert [cell["name"] for cell in EXTRA_CELLS] == [
        "prod_baseline",
        "ship_candidate",
    ]
    for cell in EXTRA_CELLS:
        assert cell["outlier_method"] in OUTLIER_METHODS
        like_sel, dis_sel = _parse_selection(cell["selection"])
        assert like_sel in SELECTION_VARIANTS
        assert dis_sel in SELECTION_VARIANTS


def test_family_layout_full_and_arm():
    full = family_layout()
    assert full[-1] == ("panns", ESSENTIA_DIMS, -1)
    assert sum(end - start for _, start, end in full[:-1]) == ESSENTIA_DIMS

    arm = family_layout(4368)
    assert arm[-1] == ("panns", 4368, -1)
    families = [name for name, _, _ in arm]
    assert "frames" not in families
    assert families[-2] == "rhythm"
    essentia_covered = sum(min(end, 4368) - start for _, start, end in arm[:-1])
    assert essentia_covered == 4368


def test_quota_preprocessor_with_arm_dims(rng):
    arm = 4368
    X = rng.normal(size=(80, arm + 16))
    y = np.concatenate([np.ones(40), np.zeros(40)])
    prep = make_preprocessor("quota64", arm)
    prep.fit(X, y)
    assert prep.transform(X).shape == (80, 64)
