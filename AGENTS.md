# tg-zmt-bot — Agent Guide

## Quick start
- **Entrypoint**: `client.py` — Telegram bot using Telethon
- **Build**: `VER=$(poetry version --short) docker buildx bake --progress=plain tg-zmt-bot`
- **Run**: Docker with env vars `API_HASH`, `API_ID`, `BOT_TOKEN`, `OWNER_USER_ID`; mounts `./data:/app/data` and `./local_data:/app/local_data`
- **Format**: `poetry run black . && poetry run ruff check --fix .`
- **Format changed files**: `poetry run black client.py config.py train.py && poetry run ruff check --fix client.py config.py train.py`
- **Python**: 3.14, Poetry

## Architecture
- `client.py` — asyncio Telegram client, command handlers
- `core/` — shared infrastructure: `paths.py` (embed versioning), `storage.py` (FeatureStore + JobStore), `jobs.py` (JobManager), `writer.py` (extraction job coordinator)
- `train.py` — ML pipeline: k-NN + GMM dual one-class models (Phase 2+3 implemented)
- `audio/features.py` — Heavy ML engine: essentia, PANNs CNN14, feature extraction functions; `prepare_extractor()` factory; `_DESCRIPTOR_SCHEMA` constant (name, length, normalizer_key) + `_NORMALIZERS` dispatch; `schema_fingerprint()` feeds into embed_version
- `audio/extractor.py` — Lightweight: `CombinedExtractor` class (DI pattern), `extract_features_for_mp3` proxy
- `audio/segments.py` — `SegmentSpec` dataclass, `get_segments()` (full, topk_energy, uniform, topk_spectral_flux)
- `audio/aggregation.py` — `aggregate()` (mean, meanstd, max strategies)
- `benchmark/compare.py` — Optuna-based benchmark tool comparing embedding variants with 5-fold CV
- `audit/descriptor_shapes.py` — offline tool: discover Essentia descriptor shapes across a corpus and emit a `_DESCRIPTOR_SCHEMA` literal; verify schema against corpus for regression testing
- `config.py` — env-based config with runtime override in `data/config.py` (bot_token/owner_user_id/data_path/local_data_path are locked)

## Key conventions
- **snake case for bot commands** using argparse
- **Two one-class models** per user (liked + disliked), NOT a binary classifier
- **Process pools** use `multiprocessing.get_context("spawn")` via lazy accessors `get_training_executor()`/`get_estimation_executor()` in config.py
- **PyTorch CPU-only** via explicit `pytorch_cpu` source in pyproject.toml
- **PANNs CNN14 assets** user-provided under `data/panns_data/`: weights `panns_cnn14.pth` and labels `class_labels_indices.csv`; container symlinks `/root/panns_data` → `/app/data/panns_data`
- **Feature cache** per-user on NAS at `data/{user_id}/features/{embed_version}/{segment_policy}/{set_name}/{file_hash}.parquet`; one parquet shard per track, atomic via tmp+rename. Probe = directory listing + set membership. Training reads via a short-lived merged parquet on `local_data/{user_id}/tmp/`. Job state in `local_data/{user_id}/jobs.duckdb`.
- **Static module-level imports only** — no function-level imports, no dynamic imports (e.g., `importlib.import_module()`, `exec()`, `eval()`). All imports must be at the top of the module file.
- **Models bundle the profile YAML** used at training at `model_workdir/essentia_profile.yaml`; estimation uses the bundled profile and refuses to run if embed_version (essentia version + profile hash + panns hash + schema fingerprint) does not match.
- **Tests** in `tests/` — run with `poetry run pytest`

## Benchmark tool
Compare embedding variants (segment policies, aggregation strategies, profiles) with Optuna Bayesian optimization:
```bash
poetry run python -m benchmark.compare \
    --config data/benchmark/embedding_variants.yaml \
    --objective-weights 0.5 0.5 \
    --output data/benchmark/report.json \
    --user-id 123456789 \
    --n-iterations 50
```
See `data/benchmark/embedding_variants.example.yaml` for config format.

## Audit tool
Discover Essentia descriptor shapes and generate the `_DESCRIPTOR_SCHEMA` literal for `audio/features.py`:
```bash
poetry run python -m audit.descriptor_shapes discover \
    --profile data/essentia_extractor_profile.yaml \
    --tracks /path/to/music/directory \
    --output data/audit/descriptor_schema_literal.txt
```
Verify the committed schema against a fresh corpus (regression guard):
```bash
poetry run python -m audit.descriptor_shapes verify \
    --profile data/essentia_extractor_profile.yaml \
    --tracks /path/to/music/directory
```