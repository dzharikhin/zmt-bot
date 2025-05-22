# syntax=docker/dockerfile-upstream:master
FROM ubuntu:jammy-20250404 AS builder
ARG POETRY_VERSION=1.8.5
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=1
ENV POETRY_VIRTUALENVS_CREATE=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Tell Poetry where to place its cache and virtual environment
ENV POETRY_CACHE_DIR=/opt/.cache

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y software-properties-common curl git build-essential gcc g++
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update && apt install -y python3.12 python-is-python3
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py setuptools wheel && python -m pip config set global.break-system-packages true
RUN add-apt-repository -s "$(cat /etc/apt/sources.list | grep -E '^deb(.+)$' | head -1 )" && apt update
# https://llvmlite.readthedocs.io/en/latest/admin-guide/install.html#building-manually
RUN pip install cmake==3.31.2 --upgrade && cmake --version && apt install -y ninja-build \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && . $HOME/.cargo/env \
    && pip install "poetry==${POETRY_VERSION}"

WORKDIR /app
COPY pyproject.toml poetry.lock /app/
RUN poetry env use 3.12 && . /app/.venv/bin/activate && pip install -U pip setuptools

# parallel compilation
ARG CPU_COUNT=4
ENV CPU_COUNT=$CPU_COUNT

WORKDIR /llvm
ENV TARGET_LLVM_NAME=llvm-project-15.0.7.src
ENV TARGET_LLVMLITE_TAG=0.44.0
RUN curl -L "https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.7/${TARGET_LLVM_NAME}.tar.xz" | tar --absolute-names -xJf - && mv ${TARGET_LLVM_NAME} llvm \
    && curl -L "https://github.com/numba/llvmlite/archive/refs/tags/v${TARGET_LLVMLITE_TAG}.tar.gz" | tar --absolute-names -xzf - && mv llvmlite-${TARGET_LLVMLITE_TAG} llvmlite \
    && cd llvm && ls ../llvmlite/conda-recipes/llvm15* | xargs -I{} patch -p1 -i {}
# set = 1 to disable tests
ARG SKIP_LLVM_TESTS=0
ENV CONDA_BUILD_CROSS_COMPILATION=$SKIP_LLVM_TESTS
ENV PREFIX=/usr/local
RUN cd /llvm/llvm && bash ../llvmlite/conda-recipes/llvmdev/build.sh

WORKDIR /app
RUN . /app/.venv/bin/activate && cd /llvm/llvmlite && python setup.py build && python runtests.py && python setup.py install
ARG POETRY_INSTALLER_MAX_WORKERS=4
ENV POETRY_INSTALLER_MAX_WORKERS=$POETRY_INSTALLER_MAX_WORKERS
RUN . /app/.venv/bin/activate && cd /app && . $HOME/.cargo/env && poetry -vv install --no-root && rm -rf $POETRY_CACHE_DIR


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