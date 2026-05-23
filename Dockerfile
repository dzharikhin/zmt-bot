# syntax=docker/dockerfile-upstream:master
FROM ubuntu:jammy-20250404 AS builder
ARG POETRY_VERSION=2.2.1
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=1
ENV POETRY_VIRTUALENVS_CREATE=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_CACHE_DIR=/opt/.cache

RUN add-apt-repository ppa:deadsnakes/ppa
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    build-essential \
    gcc \
    g++ \
    libopenblas0 \
    liblapack3 \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED \
    && curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python get-pip.py setuptools wheel \
    && python -m pip config set global.break-system-packages true \
    && pip install "poetry==${POETRY_VERSION}"

ARG CPU_COUNT=4
ENV CPU_COUNT=$CPU_COUNT

WORKDIR /app
COPY pyproject.toml poetry.lock /app/
RUN sed -i 's/^version = .*/version = "0.0.0"/' /app/pyproject.toml
COPY essentia /app/essentia
RUN . /app/.venv/bin/activate && pip install -U pip setuptools

ARG POETRY_INSTALLER_MAX_WORKERS=4
ENV POETRY_INSTALLER_MAX_WORKERS=$POETRY_INSTALLER_MAX_WORKERS
RUN . /app/.venv/bin/activate && cd /app \
    && poetry add essentia/essentia-2.1b6.dev0-cp312-cp312-manylinux_2_35_x86_64.whl \
    && poetry install --no-root \
    && rm -rf $POETRY_CACHE_DIR


FROM ubuntu:jammy-20250404 AS runtime
RUN add-apt-repository ppa:deadsnakes/ppa
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python-is-python3 \
    libopenblas0 \
    liblapack3 \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app/.venv /app/.venv
COPY . /app

WORKDIR /app
ENTRYPOINT ["python"]
CMD ["client.py"]
