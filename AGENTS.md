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
- `train.py` — ML pipeline: k-NN + GMM one-class models (Phase 2 — `train_model`/`execute_estimation` raise `NotImplementedError`)
- `audio/features.py` — Essentia + PANNs CNN14 audio feature extraction (~2248-d vector)
- `config.py` — env-based config with runtime override in `data/config.py` (bot_token/owner_user_id/data_path/local_data_path are locked)

## Key conventions
- **snake case for bot commands** using argparse
- **Two one-class models** per user (liked + disliked), NOT a binary classifier
- **Process pools** use `multiprocessing.get_context("spawn")` via lazy accessors `get_training_executor()`/`get_estimation_executor()` in config.py
- **PyTorch CPU-only** via explicit `pytorch_cpu` source in pyproject.toml
- **PANNs CNN14 weights** downloaded at Docker build time to `/app/models/panns_cnn14.pth`
- **DuckDB** per-user feature cache at `data/{user_id}/features.duckdb`
- **No local imports** — all imports at module top level; never `import` inside functions/methods
- **Tests** in `tests/` — run with `poetry run pytest`