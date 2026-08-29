# Benchmark Tool

The benchmark compares embedding variants against the dual one-class model pipeline (`includeLiked` and `excludeDisliked`). It uses Optuna Bayesian optimization to search over k-NN and GMM hyperparameters and measures cross-set separation via ROC-AUC and operating-point error rates.

This answers the key question: **"does the includeLiked model work? does the excludeDisliked model work?"** — the report clearly distinguishes a good model from a broken one by measuring discrimination between liked and disliked tracks.

## Prerequisites & Data

- **Liked/disliked tracks**: `data/{user_id}/liked/*.mp3` and `data/{user_id}/disliked/*.mp3` (`config.py:313-322`). At least 10 liked AND at least 10 disliked tracks must be extracted after feature computation, otherwise the variant is skipped (`benchmark/compare.py:395`).
- **PANNs weights**: Must exist at the path specified by `PANNS_WEIGHTS_PATH` env var or the default `data/panns_data/panns_cnn14.pth` (`config.py:69-71`).
- **Recall targets**: Configured via environment variables: `MODEL_INCLUDE_LIKED_RECALL` (default `0.775`) and `MODEL_EXCLUDE_DISLIKED_RECALL` (default `0.80`) (`config.py:55-60`). These define the operating points for reporting.
- **Essentia profile**: Used to compute feature cache keys via `get_embed_version()` (`core/paths.py:9-26`).

## Running in a Container

Benchmark tools must run in a container when operating on real data (per AGENTS.md).

### Build the image

```bash
VER=$(poetry version --short) docker buildx bake --progress=plain tg-zmt-bot
```

The image's `ENTRYPOINT` is `python` and `CMD` is `client.py`. To run the benchmark, override `CMD` by appending module arguments after the image name:

```bash
docker run ... tg-zmt-bot:$(poetry version --short) -m benchmark.compare ...
```

### Mounts and environment

- **`./data:/app/data`** — feature cache (`data/{user_id}/features/`) and PANNs weights (`data/panns_data/`). The container creates a symlink `/root/panns_data → /app/data/panns_data` at runtime.
- **`./local_data:/app/local_data`** — training scratch space (merged parquet, job state).
- **No Telegram env vars needed** — `API_ID`, `API_HASH`, `BOT_TOKEN`, `OWNER_USER_ID` are imported but unused by benchmark tools. Model config env vars use safe defaults (`MODEL_MIN_SET_SIZE=50`, `MODEL_EXCLUDE_DISLIKED_RECALL=0.80`, `MODEL_INCLUDE_LIKED_RECALL=0.775`).

### Smoke test (seconds, no data required)

Verify imports work in the image:

```bash
docker run --rm \
  -v "./data:/app/data" \
  -v "./local_data:/app/local_data" \
  tg-zmt-bot:$(poetry version --short) \
  -m benchmark.compare --help
```

### Running `compare.py` in a container

Example (variant is re-extracted only if not cached; output on host via mount):

```bash
docker run --rm \
  -v "./data:/app/data" \
  -v "./local_data:/app/local_data" \
  --cpus=4 --memory=4G \
  tg-zmt-bot:$(poetry version --short) \
  -m benchmark.compare \
    --config data/benchmark/embedding_variants.yaml \
    --objective-weights 0.5 0.5 \
    --output data/benchmark/report_v2.json \
    --user-id 123456789 \
    --n-iterations 50
```

### Resume behavior

The tool skips variants whose `name` field already appears in the `results` list of the specified `--output` JSON. Re-running the same command picks up where it left off — variants already present are skipped, new variants run.

### Env var overrides

Override model config defaults to explore different threshold regimes:

```bash
docker run ... \
  -e MODEL_EXCLUDE_DISLIKED_RECALL=0.85 \
  -e MODEL_INCLUDE_LIKED_RECALL=0.75 \
  tg-zmt-bot:$(poetry version --short) \
  -m benchmark.compare ...
```

Note the env var names have no `_TARGET` suffix (`MODEL_EXCLUDE_DISLIKED_RECALL` vs the Python `config.model_exclude_disliked_recall_target`).

---

## Configuration

The benchmark uses a YAML config file to define embedding variants. Each variant specifies the Essentia profile, PANNs weights, aggregation strategy, and segment policy.

### YAML Schema

```yaml
embeddings:
  - name: full_mean
    essentia_profile: profiles/default.yaml
    panns_weights: /app/data/panns_data/panns_cnn14.pth
    aggregation: mean
    segment_policy:
      type: full
      # window_s and k are not required for type=full

  - name: topk_energy_meanstd
    essentia_profile: profiles/default.yaml
    panns_weights: /app/data/panns_data/panns_cnn14.pth
    aggregation: meanstd
    segment_policy:
      type: topk_energy
      window_s: 30.0
      k: 3

  - name: uniform_max
    essentia_profile: profiles/default.yaml
    panns_weights: /app/data/panns_data/panns_cnn14.pth
    aggregation: max
    segment_policy:
      type: uniform
      window_s: 30.0
      k: 3

  - name: topk_spectral_flux_mean
    essentia_profile: profiles/default.yaml
    panns_weights: /app/data/panns_data/panns_cnn14.pth
    aggregation: mean
    segment_policy:
      type: topk_spectral_flux
      window_s: 30.0
      k: 3
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique variant identifier; used for `--variants` filter and skip logic (variant name must be absent from output file to re-run). |
| `essentia_profile` | path | Yes | Path to Essentia extractor YAML profile. Can be relative to `data/benchmark/` or absolute. |
| `panns_weights` | path | Yes | Absolute path to `panns_cnn14.pth` weights file. |
| `aggregation` | string | No | Aggregation strategy for segment features. Options: `mean`, `meanstd`, `max`. Default: `mean`. |
| `segment_policy.type` | string | Yes | Segment extraction policy. Options: `full`, `topk_energy`, `uniform`, `topk_spectral_flux`. |
| `segment_policy.window_s` | float | Conditional | Required for non-`full` types; specifies segment window size in seconds. |
| `segment_policy.k` | int | Conditional | Required for non-`full` types; specifies number of segments to extract. |

### Embed Version

The feature cache key is `embed_version`, computed by `get_embed_version()` (`core/paths.py:9-26`):

```
essentia-{version}+profile-{first_16_chars_of_hash}+panns-{first_16_chars_of_hash}+schema-{first_16_chars_of_hash}
```

Two variants with identical profile + PANNs weights + schema fingerprint share the same `embed_version` and thus reuse cached features.

## CLI Usage

The benchmark tool is invoked via `poetry run python -m benchmark.compare`.

### Arguments

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--config` | string | Yes | — | Path to embedding variants YAML config. |
| `--objective-weights` | float[2] | No | `0.5 0.5` | Weights for `auc_exclude` and `auc_include`, respectively. The objective is `w_a * auc_exclude + w_b * auc_include` (`benchmark/compare.py:160`). |
| `--output` | string | Yes | — | Output JSON report path. **Variant names already present in this file are skipped** (`benchmark/compare.py:298,315`). Use a fresh path to re-run all variants. |
| `--user-id` | int | Yes | — | Telegram user ID; read liked/disliked tracks from `data/{user_id}/liked/*.mp3` and `disliked/*.mp3`. |
| `--n-iterations` | int | No | `50` | Number of Optuna trials per variant. |
| `--n-workers` | int | No | `4` | Number of extraction worker processes per variant. |
| `--variants` | string | No | — | Regex to filter variant names (partial match). If set, only variants whose names contain this string are processed. |

### Example Invocations

Run a single variant with a fresh output file (re-runs everything, hits feature cache):

```bash
poetry run python -m benchmark.compare \
  --config data/benchmark/embedding_variants.yaml \
  --objective-weights 0.5 0.5 \
  --output data/benchmark/report_v2.json \
  --user-id 123456789 \
  --n-iterations 50 \
  --variants full_mean
```

Restart from a previous run (only processes new variants):

```bash
poetry run python -m benchmark.compare \
  --config data/benchmark/embedding_variants.yaml \
  --objective-weights 0.5 0.5 \
  --output data/benchmark/report_v2.json \
  --user-id 123456789 \
  --n-iterations 50
```

### Restart Behavior

- **Skip logic**: Variants whose `name` field already appears in the `results` list of the specified `--output` JSON are skipped (`benchmark/compare.py:298,315`).
- **Feature cache**: Cached features are reused when `embed_version` is unchanged (`core/paths.py:9-26`). The calibrator fix does not change the feature extraction, so cached features remain valid.
- **Restart cleanly**: Point `--output` at a new file (e.g., `report_v2.json`) to force a full re-run without modifying the YAML config.

## Output Schema

The tool outputs a JSON report with the following structure:

```json
{
  "timestamp": "2026-07-08T12:34:56.789Z",
  "objective_weights": {
    "auc_exclude": 0.5,
    "auc_include": 0.5
  },
  "threshold_regime": {
    "exclude_disliked_recall_target": 0.8,
    "include_liked_recall_target": 0.775,
    "cv_folds_in_benchmark": null
  },
  "n_iterations": 50,
  "results": [
    {
      "name": "full_mean",
      "embedding": "essentia-2.1+profile-abc12345+panns-def67890+schema-ghi12345",
      "segment_policy": "full",
      "config": { ... },
      "feature_dim": 6416,
      "extraction": {
        "time_s": 120.5,
        "liked": { "total": 100, "ok": 100 },
        "disliked": { "total": 100, "ok": 100 },
        "cached": 50,
        "newly_extracted": 50
      },
      "optimization": {
        "objective": 0.684,
        "objective_top5_median": 0.672,
        "best_params": {
          "knn_k_min": 3,
          "knn_k_max": 15,
          "knn_k_scale": 0.7,
          "gmm_components_max": 20,
          "gmm_min_points_per_component": 40,
          "outlier_threshold": 0.05
        },
        "metrics_best": {
          "auc_include": 0.91,
          "auc_exclude": 0.74,
          "disliked_false_accept": 0.18,
          "liked_false_reject": 0.28,
          "liked_recall": 0.775,
          "disliked_recall": 0.80
        },
        "metrics_top5_median": {
          "auc_include": 0.89,
          "auc_exclude": 0.71,
          "disliked_false_accept": 0.21,
          "liked_false_reject": 0.32,
          "liked_recall": 0.775,
          "disliked_recall": 0.80
        },
        "n_trials": 50,
        "trial_history": [
          { "params": { ... }, "value": 0.684, "auc_include": 0.91, "auc_exclude": 0.74, ... }
        ]
      }
    }
  ],
  "best_variant": { ... }
}
```

### Report-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | ISO 8601 string | Report generation timestamp. |
| `objective_weights` | object | `{auc_exclude: w_a, auc_include: w_b}` — weights used in the objective function. |
| `threshold_regime` | object | Recall target settings (`include_liked_recall_target`, `exclude_disliked_recall_target`) and whether cross-validation folds were used in this run (`cv_folds_in_benchmark`). |
| `n_iterations` | int | Total number of Optuna trials per variant. |
| `results` | array[object] | Per-variant results (see below). |
| `best_variant` | object or null | The variant with the highest `optimization.objective` value. |

### Per-Variant `optimization` Block

| Field | Type | Description |
|-------|------|-------------|
| `objective` | float | Objective value (`w_a * auc_exclude + w_b * auc_include`) from the best trial. |
| `objective_top5_median` | float | Median objective value across the top 5 trials (by objective value). Helps detect selection bias from a single lucky trial. |
| `best_params` | object | Hyperparameters of the best trial: `knn_k_min`, `knn_k_max`, `knn_k_scale`, `gmm_components_max`, `gmm_min_points_per_component`, `outlier_threshold`. |
| `metrics_best` | object | Six metrics from the **best** trial. |
| `metrics_top5_median` | object | Six metrics, **median** over the top 5 trials. |
| `n_trials` | int | Number of trials for this variant (same as `n_iterations` if not filtered). |
| `trial_history` | array[object] | Per-trial values for all six metrics; useful for debugging or post-hoc analysis. |

### The Six Metrics

| Metric | Description | Good Range | Useless Range |
|--------|-------------|------------|---------------|
| `auc_include` | ROC-AUC of the like-score separating liked (positive) from disliked (negative). Indicates how well the model can recommend tracks similar to liked ones. | ≥ 0.85 | ≈ 0.50 |
| `auc_exclude` | ROC-AUC of the dislike-score separating disliked (positive) from liked (negative). Indicates how well the model can reject tracks similar to disliked ones. | ≥ 0.85 | ≈ 0.50 |
| `liked_recall` | At the include operating point (target 0.775), fraction of liked tracks that are correctly accepted (recommended). Pinned near 0.775 by the percentile threshold definition (`benchmark/compare.py:141-144`). | ~0.775 by construction | — |
| `disliked_false_accept` | At the include operating point, fraction of **disliked** tracks that wrongly pass the "post it" gate. Lower is better. | < ~0.20 (strong) / < 0.40 (decent) | ≈ 0.775 (model does nothing) |
| `disliked_recall` | At the exclude operating point (target 0.80), fraction of disliked tracks that are correctly rejected. Pinned near 0.80 by the percentile threshold definition (`benchmark/compare.py:147-150`). | ~0.80 by construction | — |
| `liked_false_reject` | At the exclude operating point, fraction of **liked** tracks that are wrongly rejected (collateral damage). Lower is better. | < ~0.15 (strong) / < 0.35 (decent) | ≈ 0.80 (model nukes good tracks) |

**Key point:** `liked_recall` and `disliked_recall` are pinned near their targets by the threshold definition. They do **not** signal model quality; the quality signals are the other set's error rates at the same operating point (`disliked_false_accept` / `liked_false_reject`).

### Extraction Statistics

| Field | Type | Description |
|-------|------|-------------|
| `time_s` | float | Total feature extraction time in seconds. |
| `liked.total`, `liked.ok` | int | Total liked tracks and successful extractions. |
| `disliked.total`, `disliked.ok` | int | Total disliked tracks and successful extractions. |
| `cached` | int | Number of tracks whose features were reused from cache. |
| `newly_extracted` | int | Number of tracks whose features were newly computed. |

## Interpreting Results

### Quick Start

1. **Check AUCs first**: Look at `optimization.metrics_top5_median.auc_include` and `optimization.metrics_top5_median.auc_exclude`. If both ≈ 0.5, the entire pipeline cannot discriminate — check feature extraction or calibration. If one is ≈ 0.5 but the other is high, only that mode works.

2. **Cross-check best vs median**: A large gap on any metric means the best trial was an outlier. Trust the top-5 median for variant comparison.

3. **Compare variants**: A variant is "better" only if its AUC gap exceeds the best-vs-median noise band. Higher objective with a stable top-5 median is a true win.

### includeLiked Model

The includeLiked model answers "post tracks similar to liked ones." The like-score should be HIGH for liked tracks and LOW for disliked tracks.

| Field | Meaning | Good | Useless |
|-------|---------|------|---------|
| `auc_include` | Can the like-score separate liked from disliked? | ≥ 0.85 | ≈ 0.50 |
| `disliked_false_accept` | At `liked_recall`≈0.775, fraction of **disliked** tracks wrongly passed to "post" | < ~0.20 (strong) / < 0.40 (decent) | ≈ 0.775 (no discrimination) |

**Anchor**: If the model cannot discriminate (AUC 0.5), to accept 77.5% of liked it must also accept ~77.5% of disliked, so `disliked_false_accept ≈ 0.775` means the model is doing nothing useful.

#### Verdict Buckets

- **strong** (≥ 0.90 AUC AND < 0.20 `disliked_false_accept`): posts most liked tracks, leaks few disliked.
- **decent** (0.80–0.90 AUC AND < 0.40 `disliked_false_accept`): usable, some leakage.
- **weak** (0.70–0.80 AUC): meaningful separation but high leakage; likely needs better features/scaling.
- **broken** (≈ 0.50 AUC): cannot tell liked from disliked at the chosen recall.

### excludeDisliked Model

The excludeDisliked model answers "reject tracks similar to disliked ones." The dislike-score should be HIGH for disliked tracks and LOW for liked tracks. `decide()` rejects (returns False) when `dislike_calibrated >= threshold_exclude` (`core/modeling.py:348-357`).

| Field | Meaning | Good | Useless |
|-------|---------|------|---------|
| `auc_exclude` | Can the dislike-score separate disliked from liked? | ≥ 0.85 | ≈ 0.50 |
| `liked_false_reject` | At `disliked_recall`≈0.80, fraction of **liked** tracks wrongly rejected | < ~0.15 (strong) / < 0.35 (decent) | ≈ 0.80 (model nukes good tracks) |

**Anchor**: If the model cannot discriminate (AUC 0.5), to reject 80% of disliked it must also reject ~80% of liked, so `liked_false_reject ≈ 0.80` means the model is doing nothing useful.

#### Verdict Buckets

- **strong** (≥ 0.90 AUC AND < 0.15 `liked_false_reject`): catches most disliked, spares most liked.
- **decent** (0.80–0.90 AUC AND < 0.35 `liked_false_reject`): usable, some collateral damage.
- **weak** (0.70–0.80 AUC): rejects too much good music alongside the bad.
- **broken** (≈ 0.50 AUC): cannot distinguish; at 80% dislike recall it rejects ~80% of liked too — effectively unusable.

### Comparing Variants

When comparing two variants:

1. **Use AUCs, not constructed recalls**: The quality signals are `auc_include`, `auc_exclude`, `disliked_false_accept`, `liked_false_reject`. The recall metrics are pinned by the percentile threshold.

2. **Trust top-5 medians**: Compare `metrics_top5_median` between variants, not just `metrics_best`. A large gap between best and median indicates a lucky outlier.

3. **Objective as tiebreaker**: If two variants have similar AUCs, prefer the one with higher `objective_top5_median` and/or fewer hyperparameters.

4. **Consider practical cost**: A slightly lower AUC variant may be faster to extract if it uses a cheaper segment policy or aggregation.

### Worked Example

From a report with these values:

- `optimization.metrics_best.auc_include = 0.91`
- `optimization.metrics_best.disliked_false_accept = 0.18`
- `optimization.metrics_best.liked_recall = 0.775`

**Interpretation**: includeLiked works well. It posts ~77.5% of liked tracks while letting only ~18% of disliked slip through — strong discrimination with minimal leakage.

Another variant:

- `optimization.metrics_best.auc_exclude = 0.55`
- `optimization.metrics_best.liked_false_reject = 0.82`
- `optimization.metrics_best.disliked_recall = 0.80`

**Interpretation**: excludeDisliked is effectively broken. To catch 80% of disliked it rejects 82% of liked too, confirming no discrimination (AUC 0.55). This variant should be ignored or fixed before deployment.

## Implementation Details

### Model Architecture

The benchmark fits **two independent one-class models per user** (liked set + disliked set), not a binary classifier:

- `INCLUDE_LIKED`: recommend a track if `like_calibrated > threshold_include`.
- `EXCLUDE_DISLIKED`: recommend a track if `dislike_calibrated < threshold_exclude`.

### Optuna Objective

The objective function (`benchmark/compare.py:26-160`) performs:

- **Hyperparameter search**: k-NN `k_min`, `k_max`, `k_scale`; GMM `components_max`, `min_points_per_component`; outlier threshold.
- **Cross-validation**: 5-fold KFold on each set, with outliers removed per fold (`detect_outliers`).
- **Cross-set scoring**: For each fold, scores are computed for both sub-models on both held-out sets. This produces pooled arrays used to compute ROC-AUCs and operating-point error rates.
- **Weighted objective**: `w_a * auc_exclude + w_b * auc_include`, where weights are passed via `--objective-weights`.

### Trial Selection Bias

The reported per-variant metrics use:
- **Best trial**: `metrics_best` from `optimization.metrics_best`.
- **Top-5 median**: `metrics_top5_median` from the median of the top 5 trials by objective value.

This guards against a single lucky trial dominating the reported values.

### Feature Cache

Features are cached per `embed_version` (`core/paths.py:9-26`). When running multiple variants with identical profile, PANNs weights, and schema fingerprint, the cached features are reused across variants, making the benchmark more efficient.

---

## Gates Study (`gates_study.py`)

Isolates data-cleaning and preprocessing levers on a **fixed** feature set (loads parquet shards directly from a features dir; no extraction): outlier method × budget × selection variant. Multi-objective Optuna (NSGA-II) minimizing `(lfr@0.80, dfa@0.775)` from 5-fold OOF calibrated scores averaged over 3 seeds; per-model kNN/GMM params pinned at shipped values.

```bash
./benchmark/run_gates_study.sh [FEATURES_DIR] [N_ITERATIONS]
# defaults: features dir "features", 60 iterations, output data/benchmark/gates_study.json
```

Extra CLI flags:

- `--extra-cells` — also evaluate the pinned parity cells: `prod_baseline` (welch64 + prod_fused @ shipped budget) and `ship_candidate` (per:quota64/ridge_select64 + prod_fused @ 0.07) with full metric dicts (all recall points) and verdicts at BOTH anchors (0.8 guideline and 0.9 prod operating point).
- `--essentia-dims K` — essentia block width of the features dir (default 4380 = current schema). Used when running on a column-sliced arm so quota family layout adapts to the narrower essentia block.

Search space:

| Dimension | Values |
|-----------|--------|
| `outlier_method` | `prod_fused` (shipped kNN+IF rank fusion, raw space) / `knn` / `iforest` / `std_fused` (fusion on standardized space) / `lof_std` |
| `outlier_budget` | float 0.02–0.12 (prod_fused removes less than nominal: fusion behaves as a consensus rule) |
| `selection` | `welch64` (shipped) / `ridge_select64` (\|logistic coef\| top-64) / `fused_welch_ridge64` (rank fusion) / `quota64` (family quota: 16 lowlevel, 12 tonal, 12 rhythm, 24 panns — quotas scale with n_features; Welch within family) / `pls_project64` (PLS projection) |

Output JSON: `essentia_dims`, `baseline` (shipped config metrics + verdict at 0.8 and 0.9), `pareto_front` (params + full metric dicts + verdicts), `trial_history` (full metrics per trial, for plateau checks), `extra_cells` (when `--extra-cells`). Verdict labels vs owner guideline: `stretch` (lfr@0.8 ≤ 0.12 and dfa@0.775 ≤ 0.08), `guideline` (≤ 0.20 both), `fail`.

## Fused decision-rule study

`benchmark/fused_rule_study.py` evaluates score-level fusion of the two one-class gates on OOF scores from the gates_study CV protocol (5-fold × seeds, `run_cv(return_scores=True)`):

```bash
poetry run python -m benchmark.fused_rule_study --features-dir FEATURES_DIR \
    --essentia-dims 4380 --cells ship_candidate prod_baseline --skip-probe \
    --output data/benchmark/fused_rule_study.json
```

- Diff fusion: exclude iff `dislike_cal − w·like_cal ≥ t(w)` (t = percentile on OOF fused disliked scores, mirroring prod calibration); include side symmetric. w-grid 0–2; w=0 ≡ the shipped single-score rule.
- Diagnostics: AND-rule (2D percentile grid, grid-selected per seed → OPTIMISTIC, reference only), supervised logistic probe ceiling (offline only, never shipped), shuffle placebo (destroys per-point pairing), clamp/pairing score diagnostics, verdicts vs the hard gate (lfr@0.85 ≤ 0.20 AND dfa@0.775 ≤ 0.20; stretch lfr@0.9 ≤ 0.20).
- Cells: `prod_baseline` / `ship_candidate` carry single-rule PARITY_ANCHORS (arm4380) that must reproduce before any fused row is trusted; other cells (`per_welch64_ridge64`, `per_quota32_ridge64`, `quota64_shared`) are annotate-only.
- Production counterpart: `DualOneClassModel(decision_mode="fused_diff", fusion_weight=w)` in `core/modeling.py` — same fusion math, thresholds calibrated on fused OOF scores at fit time.

## Files

- `benchmark/compare.py` — Optuna benchmark tool and CLI.
- `benchmark/dfa_gate_study.py` — include-gate operating-point study (mechanism A/B + liked-side search).
- `benchmark/fused_rule_study.py` — Fused decision-rule study (diff fusion w-grid + diagnostics).
- `benchmark/gates_study.py` — Gates study (outlier × selection, NSGA-II).
- `benchmark/preprocessor.md` — Preprocessor implementation plan (executed).
- `benchmark/segment_sweep.yaml` — Segment policy sweep configuration for benchmark.
- `benchmark/README.md` — This file.
- `audio/segments.py` — Segment extraction policies and canonical string generation.
- `core/paths.py` — `get_embed_version()` and feature cache key computation.
- `config.py` — Recall targets, config helpers, and data paths.

## See Also

- `AGENTS.md` — Agent guide for building and running the benchmark.
- `train.py` — Production training pipeline using the same dual one-class models.
- `tests/test_modeling.py` — Unit tests for one-class models and scoring (for developers).
