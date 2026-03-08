# syntax=docker/dockerfile-upstream:master
FROM ubuntu:jammy-20250404 AS builder
ARG POETRY_VERSION=2.2.1
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=1
ENV POETRY_VIRTUALENVS_CREATE=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Tell Poetry where to place its cache and virtual environment
ENV POETRY_CACHE_DIR=/opt/.cache

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y software-properties-common curl git build-essential gcc g++ libopenblas0 liblapack3 libsndfile1 libgomp1
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update && apt install -y python3.12 python3.12-dev python-is-python3
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py setuptools wheel && python -m pip config set global.break-system-packages true
RUN add-apt-repository -s "$(cat /etc/apt/sources.list | grep -E '^deb(.+)$' | head -1 )" && apt update
RUN pip install "poetry==${POETRY_VERSION}"

# parallel compilation
ARG CPU_COUNT=4
ENV CPU_COUNT=$CPU_COUNT

WORKDIR /app
COPY pyproject.toml poetry.lock /app/
COPY essentia /app/essentia
RUN poetry env use 3.12 && . /app/.venv/bin/activate && pip install -U pip setuptools

ARG POETRY_INSTALLER_MAX_WORKERS=4
ENV POETRY_INSTALLER_MAX_WORKERS=$POETRY_INSTALLER_MAX_WORKERS
RUN . /app/.venv/bin/activate && cd /app \
    && poetry add --editable essentia/essentia-2.1b6.dev0-cp312-cp312-manylinux_2_35_x86_64.whl \
    && poetry -vv install --no-root \
    && rm -rf $POETRY_CACHE_DIR


FROM ubuntu:jammy-20250404 AS runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y software-properties-common curl
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update && apt install -y python3.12 python-is-python3 libopenblas0 liblapack3 libsndfile1 libgomp1
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app/.venv /app/.venv
COPY . /app

# ENV API_HASH
# ENV API_ID
# ENV BOT_TOKEN
# ENV OWNER_USER_ID
WORKDIR /app
ENTRYPOINT ["python"]
CMD ["client.py"]