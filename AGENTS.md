# tg-zmt-bot — Agent Guide

## Quick start
- **Entrypoint**: `client.py` — Telegram bot using Telethon (asyncio)
- **Build**: `VER=$(poetry version --short) docker buildx bake --progress=plain tg-zmt-bot`
- **Run**: Docker with env vars `API_HASH`, `API_ID`, `BOT_TOKEN`, `OWNER_USER_ID`; mounts `./data:/app/data` and `./local_data:/app/local_data`
- **Format**: `poetry run black . && poetry run ruff check --fix .`
- **Python**: 3.14, Poetry; PyTorch CPU-only via explicit `pytorch_cpu` source in pyproject.toml

## Real data runs MUST be dockerized
No raw audio in the dev checkout — real user data lives only on the remote host. Any script touching real data (extraction, benchmarks, audits) runs there inside the built image, exactly like prior benchmark runs (full manual: `benchmark/README.md`):
```bash
docker run --rm -v ./data:/app/data -v ./local_data:/app/local_data \
  --cpus 4 --memory 4G tg-zmt-bot:$(poetry version --short) -m benchmark.compare ...
```
- Image `ENTRYPOINT` is `python`, `CMD` is `client.py` — override by appending `-m <module> <args>` after the image tag.
- Code is baked via `COPY . /app` — every code change requires git pull + rebuild before the next docker run.
- `data/` (NAS mount) is gitignored; `local_data/` is runtime scratch.

## Architecture
- `client.py` — asyncio Telegram client, command handlers
- `train.py` — ML train/estimate pipeline; `build_preprocessor()` registry: `noop`, `standardize_select`, `welch<N>`, `ridge_select<N>`, `quota<N>` (+ `PANNS_FAMILY_QUOTA`)
- `core/` — `modeling.py` (`DualOneClassModel`, `OneClassSetModel`), `preprocessing.py` (`welch_scores`, `StandardizeSelect`/`RidgeSelect`/`QuotaSelect` preprocessor classes), `outliers.py`, `paths.py` (embed versioning), `storage.py` (FeatureStore + JobStore), `jobs.py`, `writer.py` (bulk extraction entry `start_extraction_job()`)
- `audio/` — `features.py` (essentia + PANNs CNN14 engine, `_DESCRIPTOR_SCHEMA`, `descriptor_family_layout()`, `schema_fingerprint()`), `extractor.py` (`CombinedExtractor`, DI pattern), `segments.py`, `aggregation.py`
- `benchmark/` — Optuna study tools (`compare.py`, `gates_study.py`, `dfa_gate_study.py`, `analyze_pareto.py`) — see `benchmark/README.md`
- `audit/descriptor_shapes.py` — discover/verify `_DESCRIPTOR_SCHEMA` against a real corpus
- `config.py` — env-based config with runtime override in `data/config.py` (bot_token/owner_user_id/data_path/local_data_path are locked)
- `distinction_improvement.md` (untracked) — active plan + results log for the gate-quality project

## Key conventions
- **snake case for bot commands** using argparse
- **Two one-class models** per user (liked + disliked), NOT a binary classifier; binary probes are offline diagnostics only, never shipped
- **Audio data only** — no artist, genre, or other metadata
- **Per-model preprocessing**: liked and disliked models each get their own preprocessor (`MODEL_LIKE_PREPROCESSOR` default `quota64`, `MODEL_DISLIKE_PREPROCESSOR` default `ridge_select64`, `MODEL_OUTLIER_THRESHOLD` default `0.07`). Old pickles without per-model attrs must keep loading (getattr-safe helpers in `DualOneClassModel`)
- **Static module-level imports only** — no function-level imports, no dynamic imports (`importlib.import_module()`, `exec()`, `eval()`)
- **Process pools** use `multiprocessing.get_context("spawn")` via lazy accessors `get_training_executor()`/`get_estimation_executor()` in config.py
- **PANNs CNN14 assets** user-provided under `data/panns_data/`: weights `panns_cnn14.pth` + labels `class_labels_indices.csv`; container symlinks `/root/panns_data` → `/app/data/panns_data`
- **Feature cache** per-user: `data/{user_id}/features/{embed_version}/{segment_policy}/{set_name}/{file_hash}.parquet`; one parquet shard per track, atomic via tmp+rename; probe = directory listing + set membership. Training reads via a short-lived merged parquet on `local_data/{user_id}/tmp/`. Job state in `local_data/{user_id}/jobs.duckdb`. Cache probe makes bulk extraction resumable.
- **embed_version** = essentia version + profile hash + panns hash + schema fingerprint — any profile/schema change re-keys the whole cache. Models bundle the profile YAML at `model_workdir/essentia_profile.yaml`; estimation refuses to run on mismatch.

## Essentia gotchas
- Runtime profile is `data/essentia_extractor_profile.yaml`; repo-root `essentia_profile.yaml` is an untracked dev copy.
- `_DESCRIPTOR_SCHEMA` families INTERLEAVE (rhythm/tonal runs repeat) — never assume family contiguity; use `descriptor_family_layout()` from `audio/features.py`.
- String descriptors (e.g. `chords_scale`, `key`) cannot enter the numeric vector.
- Descriptor-name validation needs no audio: construct `es.MusicExtractor(profile=...)` with a tmp profile — unsupported names fail at construction.

## Model code gotchas
- `OneClassSetModel.score()` is single-row only (indexes `[0]` internally) — loop per point for batches.

## Benchmark tool
Compare embedding variants with Optuna (config YAML schema documented in `benchmark/README.md`):
```bash
poetry run python -m benchmark.compare --config ... --objective-weights 0.5 0.5 \
    --output data/benchmark/report.json --user-id 123456789 --n-iterations 50
```
Gates study (outlier method × budget × selection, NSGA-II on lfr@0.8 + dfa@0.775) over pre-extracted parquet shards — `benchmark/run_gates_study.sh [FEATURES_DIR] [N_ITERATIONS]`.

## Audit tool
```bash
poetry run python -m audit.descriptor_shapes discover \
    --profile data/essentia_extractor_profile.yaml \
    --tracks /path/to/music/directory \
    --output data/audit/descriptor_schema_literal.txt   # emit _DESCRIPTOR_SCHEMA literal
poetry run python -m audit.descriptor_shapes verify \
    --profile data/essentia_extractor_profile.yaml --tracks /path/to/music/directory
```
`discover` stratified-samples `--k` tracks (default 10) from the corpus.

## Tests
`poetry run pytest` — full suite ≈5 min; focus with `poetry run pytest tests/test_modeling.py -q`.
