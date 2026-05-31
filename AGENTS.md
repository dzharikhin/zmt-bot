# tg-zmt-bot — Agent Guide

## Quick start
- **Entrypoint**: `client.py` — Telegram bot using Telethon
- **Build**: `VER=$(poetry version --short) docker buildx bake --progress=plain tg-zmt-bot`
- **Run**: Docker with env vars `API_HASH`, `API_ID`, `BOT_TOKEN`, `OWNER_USER_ID`; mounts `./data:/app/data` and `./local_data:/app/local_data`
- **Format**: `poetry run black .`
- **Format changed files**: `poetry run black client.py config.py train.py`
- **Python**: 3.14, Poetry

## Architecture
- `client.py` — asyncio Telegram client, command handlers
- `core/` — shared infrastructure: `paths.py` (embed versioning), `storage.py` (DuckDB feature cache), `jobs.py` (extraction job tracking), `writer.py` (queue-based extraction workers)
- `train.py` — ML pipeline: k-NN + GMM dual one-class models (Phase 2+3 implemented)
- `audio/features.py` — Essentia + PANNs CNN14 audio feature extraction (~2248-d vector), with segment+aggregate support
- `audio/segments.py` — `SegmentSpec` dataclass, `get_segments()` (full, topk_energy, uniform, topk_spectral_flux)
- `audio/aggregation.py` — `aggregate()` (mean, meanstd, max strategies)
- `benchmark/compare.py` — Optuna-based benchmark tool comparing embedding variants with 5-fold CV
- `config.py` — env-based config with runtime override in `data/config.py` (bot_token/owner_user_id/data_path/local_data_path are locked)

## Key conventions
- **snake case for bot commands** using argparse
- **Two one-class models** per user (liked + disliked), NOT a binary classifier
- **Process pools** use `multiprocessing.get_context("spawn")` via lazy accessors `get_training_executor()`/`get_estimation_executor()` in config.py
- **PyTorch CPU-only** via explicit `pytorch_cpu` source in pyproject.toml
- **PANNs CNN14 assets** user-provided under `data/panns_data/`: weights `panns_cnn14.pth` and labels `class_labels_indices.csv`; container symlinks `/root/panns_data` → `/app/data/panns_data`
- **DuckDB** per-user feature cache at `data/{user_id}/features.duckdb`; composite PK `(file_hash, embed_version, segment_policy)`
- **No local imports** — all imports at module top level; never `import` inside functions/methods
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