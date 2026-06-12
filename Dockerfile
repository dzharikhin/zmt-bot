# syntax=docker/dockerfile-upstream:master
FROM busybox:latest AS deps
COPY pyproject.toml /deps.toml
RUN sed -i 's/^version = ".*"/version = "0.0.0"/' /deps.toml



FROM python:3.14-slim AS builder
ARG POETRY_VERSION=2.4.1
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=1
ENV POETRY_VIRTUALENVS_CREATE=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_CACHE_DIR=/opt/.cache

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    libsndfile1 \
    libgomp1 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install "poetry==${POETRY_VERSION}"

WORKDIR /app
COPY --from=deps /deps.toml /app/pyproject.toml
COPY poetry.lock /app/
ARG POETRY_INSTALLER_MAX_WORKERS=4
ENV POETRY_INSTALLER_MAX_WORKERS=$POETRY_INSTALLER_MAX_WORKERS
RUN poetry env use 3.14 && . /app/.venv/bin/activate && cd /app \
    && poetry install --no-root \
    && rm -rf $POETRY_CACHE_DIR



FROM python:3.14-slim AS runtime
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    libsndfile1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app/.venv /app/.venv
COPY . /app

RUN ln -s /app/data/panns_data /root/panns_data

WORKDIR /app
ENTRYPOINT ["python"]
CMD ["client.py"]
