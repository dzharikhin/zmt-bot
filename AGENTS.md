# tg-zmt-bot — Agent Guide

## Quick start
- **Entrypoint**: `client.py` — Telegram bot using Telethon
- **Build**: `VER=$(poetry version --short) docker buildx bake --progress=plain tg-zmt-bot`
- **Run**: Docker with env vars `API_HASH`, `API_ID`, `BOT_TOKEN`, `OWNER_USER_ID`; mounts `./data:/app/data` and `./local_data:/app/local_data`
- **Format**: `poetry run black .`
- **Format changed files**: `poetry run black client.py config.py train.py`
- **Python**: 3.12, Poetry

## Architecture
- `client.py` — asyncio Telegram client, command handlers
- `train.py` — ML pipeline: feature extraction → BayesSearchCV over sklearn pipelines (GMeans, OneClassSVM, etc.)
- `audio/features.py` — Essentia-based audio feature extraction (~200+ dim vector)
- `dataset/persistent_dataset_processor.py` — Polars DataFrame builder with atomic file locking
- `config.py` — env-based config with runtime override in `data/config.py` (bot_token/owner_user_id/data_path/local_data_path are locked)

## Key conventions
- **snake case for bot commands** using argparse
- **Two one-class models** per user (liked + disliked), NOT a binary classifier
- **Process pools** use `multiprocessing.get_context("spawn")` in config.py
- **PyTorch CPU-only** via explicit `pytorch_cpu` source in pyproject.toml
- **Essentia wheel** at `essentia/essentia-*-manylinux*.whl`; build from source: `docker buildx bake essentia-builder`
- **No tests, no CI, no typechecker** — only `black` for formatting