# syntax=docker/dockerfile-upstream:master
FROM --platform=$BUILDPLATFORM python:3.12.9-bullseye AS armv7-polars_builder
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TARGET_ARCH=armv7-unknown-linux
ENV TARGET_ARCH_POSTFIX=gnueabihf
RUN apt update && apt install -y gcc-arm-linux-${TARGET_ARCH_POSTFIX} \
    && pip install -U pip setuptools maturin patchelf ziglang \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ARG CPU_COUNT=4
ENV CPU_COUNT=$CPU_COUNT

WORKDIR /polars
ARG CARGO_BUILD_JOBS=4
ENV CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS
ENV TARGET_POLARS_TAG=py-1.24.0
RUN curl -L https://github.com/pola-rs/polars/archive/refs/tags/${TARGET_POLARS_TAG}.tar.gz | tar --absolute-names -xzf - && mv polars-${TARGET_POLARS_TAG} polars \
    && . $HOME/.cargo/env \
    && sed -i "s/--upgrade --compile-bytecode --no-build/--upgrade --compile-bytecode/" /polars/polars/Makefile \
    && sed -i "s/kuzu//" /polars/polars/py-polars/requirements-dev.txt \
    && sed -i "s/deltalake>=0.15.0//" /polars/polars/py-polars/requirements-dev.txt \
    && cd /polars/polars/py-polars && make --debug=b -j${CPU_COUNT} requirements
RUN . $HOME/.cargo/env && cd /polars/polars && rustup target add ${TARGET_ARCH}-${TARGET_ARCH_POSTFIX} \
    && maturin build -j${CPU_COUNT} --profile dist-release --manifest-path py-polars/Cargo.toml --target ${TARGET_ARCH}-${TARGET_ARCH_POSTFIX} --zig




FROM python:3.12.9-bullseye AS linux.arm.v7-builder

ARG POETRY_VERSION=1.8.5

ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=1
ENV POETRY_VIRTUALENVS_CREATE=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Tell Poetry where to place its cache and virtual environment
ENV POETRY_CACHE_DIR=/opt/.cache

RUN apt update && apt install -y software-properties-common && add-apt-repository -s "$(cat /etc/apt/sources.list | grep -E '^deb(.+)$' | head -1 )" && apt update
# https://llvmlite.readthedocs.io/en/latest/admin-guide/install.html#building-manually
RUN apt -y build-dep scipy && pip install cmake --upgrade && cmake --version && apt install -y ninja-build \
# scipy https://docs.scipy.org/doc/scipy/building/index.html
    gcc g++ gfortran libopenblas-dev liblapack-dev pkg-config python3-dev \
# pyarrow https://arrow.apache.org/install/
    ca-certificates lsb-release \
    && curl -LO https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb \
    && apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb \
    && apt update \
    && apt install -y -V  build-essential \
# other
    && pip install -U pip && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && . $HOME/.cargo/env && pip install "poetry==${POETRY_VERSION}"
# parallel compilation
ARG CPU_COUNT=4
ENV CPU_COUNT=$CPU_COUNT

#create virtualenv - poetry works with venv, so no system python
WORKDIR /app
COPY pyproject.toml poetry.lock /app/
RUN poetry env use 3.12 && poetry env list && which python
# https://llvmlite.readthedocs.io/en/latest/admin-guide/install.html#building-manually
# https://github.com/pola-rs/polars#python-compile-polars-from-source
RUN . /app/.venv/bin/activate && which python && pip install -U pip setuptools maturin patchelf

WORKDIR /llvm
ENV TARGET_LLVM_NAME=llvm-project-15.0.7.src
ENV TARGET_LLVMLITE_TAG=0.44.0
RUN curl -L https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.7/${TARGET_LLVM_NAME}.tar.xz | tar --absolute-names -xJf - && mv ${TARGET_LLVM_NAME} llvm \
    && curl -L https://github.com/numba/llvmlite/archive/refs/tags/v${TARGET_LLVMLITE_TAG}.tar.gz | tar --absolute-names -xzf - && mv llvmlite-${TARGET_LLVMLITE_TAG} llvmlite \
    && cd llvm && ls ../llvmlite/conda-recipes/llvm15* | xargs -I{} patch -p1 -i {}
# set = 1 to disable tests
ARG SKIP_LLVM_TESTS=0
ENV CONDA_BUILD_CROSS_COMPILATION=$SKIP_LLVM_TESTS
ENV PREFIX=/usr/local
RUN cd /llvm/llvm && bash ../llvmlite/conda-recipes/llvmdev/build.sh

WORKDIR /arrow
ENV TARGET_ARROW_TAG=apache-arrow-19.0.1
RUN curl -L https://github.com/apache/arrow/releases/download/${TARGET_ARROW_TAG}/${TARGET_ARROW_TAG}.tar.gz | tar --absolute-names -xzf - && mv ${TARGET_ARROW_TAG} arrow \
    && cd arrow/cpp && cmake --preset -N ninja-release-python && mkdir build && cd build && cmake .. --preset ninja-release-python -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build . && cmake --install .

# https://github.com/apache/arrow-adbc/tree/main/python/adbc_driver_sqlite
WORKDIR /adbc
ENV TARGET_ADBC_TAG=apache-arrow-adbc-17
RUN curl -L https://github.com/apache/arrow-adbc/releases/download/${TARGET_ADBC_TAG}/${TARGET_ADBC_TAG}.tar.gz | tar --absolute-names -xzf - && mv ${TARGET_ADBC_TAG} adbc \
    && cd adbc && . /app/.venv/bin/activate && cmake -S c -B build -DADBC_DRIVER_SQLITE=ON -DADBC_BUILD_SHARED=1 && cmake --build build && cmake --install build
ENV ADBC_SQLITE_LIBRARY=/usr/local/lib/libadbc_driver_sqlite.so

WORKDIR /crates-typos
ENV TARGET_TYPOS_TAG=1.29.0
RUN curl -L https://github.com/crate-ci/typos/archive/refs/tags/v${TARGET_TYPOS_TAG}.tar.gz | tar --absolute-names -xzf - && mv typos-${TARGET_TYPOS_TAG} crates-typos \
    && . $HOME/.cargo/env && . /app/.venv/bin/activate \
    && cd /crates-typos/crates-typos && maturin build -j${CPU_COUNT} --manifest-path crates/typos-cli/Cargo.toml --release \
    && cd /crates-typos/crates-typos/target/wheels && ls | xargs -I{} pip install {} && pip list --format=freeze

WORKDIR /polars
COPY --from=armv7-polars_builder /polars/polars/target/wheels /polars/polars/target/wheels
RUN . /app/.venv/bin/activate && cd /polars/polars/target/wheels && ls | xargs -I{} pip install {} \
    && pip list --format=freeze

WORKDIR /app
RUN . /app/.venv/bin/activate && cd /llvm/llvmlite && python setup.py build && python runtests.py && python setup.py install
ARG POETRY_INSTALLER_MAX_WORKERS=4
ENV POETRY_INSTALLER_MAX_WORKERS=$POETRY_INSTALLER_MAX_WORKERS
RUN . /app/.venv/bin/activate && cd /app && . $HOME/.cargo/env \
    && export WHL=/polars/polars/target/wheels/$(ls /polars/polars/target/wheels) && sed -i -E "s:polars.+:polars = { path=\"$WHL\" }:" pyproject.toml && poetry lock --no-update \
    && poetry -vv install --no-root && rm -rf $POETRY_CACHE_DIR




FROM python:3.12.9-bullseye AS linux.x86_64-builder
ARG POETRY_VERSION=1.8.5
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=1
ENV POETRY_VIRTUALENVS_CREATE=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Tell Poetry where to place its cache and virtual environment
ENV POETRY_CACHE_DIR=/opt/.cache

RUN apt update && apt install -y software-properties-common && add-apt-repository -s "$(cat /etc/apt/sources.list | grep -E '^deb(.+)$' | head -1 )" && apt update
# https://llvmlite.readthedocs.io/en/latest/admin-guide/install.html#building-manually
RUN pip install cmake --upgrade && cmake --version && apt install -y ninja-build \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && . $HOME/.cargo/env \
    && pip install -U pip setuptools && pip install "poetry==${POETRY_VERSION}"
# parallel compilation
ARG CPU_COUNT=4
ENV CPU_COUNT=$CPU_COUNT

WORKDIR /llvm
ENV TARGET_LLVM_NAME=llvm-project-15.0.7.src
ENV TARGET_LLVMLITE_TAG=0.44.0
RUN curl -L https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.7/${TARGET_LLVM_NAME}.tar.xz | tar --absolute-names -xJf - && mv ${TARGET_LLVM_NAME} llvm \
    && curl -L https://github.com/numba/llvmlite/archive/refs/tags/v${TARGET_LLVMLITE_TAG}.tar.gz | tar --absolute-names -xzf - && mv llvmlite-${TARGET_LLVMLITE_TAG} llvmlite \
    && cd llvm && ls ../llvmlite/conda-recipes/llvm15* | xargs -I{} patch -p1 -i {}
# set = 1 to disable tests
ARG SKIP_LLVM_TESTS=0
ENV CONDA_BUILD_CROSS_COMPILATION=$SKIP_LLVM_TESTS
ENV PREFIX=/usr/local
RUN cd /llvm/llvm && bash ../llvmlite/conda-recipes/llvmdev/build.sh

WORKDIR /app
COPY pyproject.toml poetry.lock /app/
RUN poetry env use 3.12
RUN . /app/.venv/bin/activate && cd /llvm/llvmlite && python setup.py build && python runtests.py && python setup.py install
ARG POETRY_INSTALLER_MAX_WORKERS=4
ENV POETRY_INSTALLER_MAX_WORKERS=$POETRY_INSTALLER_MAX_WORKERS
RUN . /app/.venv/bin/activate && cd /app && . $HOME/.cargo/env && rm poetry.lock && poetry -vv install --no-root && rm -rf $POETRY_CACHE_DIR




FROM ${TARGETPLATFORM//\//.}-builder AS target_builder




FROM python:3.12.9-bullseye AS runtime
RUN apt update && apt install -y libopenblas0 liblapack3
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=target_builder /usr/local /usr/local
COPY --from=target_builder /app/.venv /app/.venv
COPY . /app

# ENV API_HASH
# ENV API_ID
# ENV BOT_TOKEN
# ENV OWNER_USER_ID
WORKDIR /app
ENTRYPOINT ["python"]
CMD ["client.py"]