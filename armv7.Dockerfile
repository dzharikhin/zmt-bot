# syntax=docker/dockerfile-upstream:master
FROM python:3.12.9-bullseye AS linux.arm.v7-builder
RUN getconf PAGE_SIZE

ARG POETRY_VERSION=1.8.5

ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=1
ENV POETRY_VIRTUALENVS_CREATE=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Tell Poetry where to place its cache and virtual environment
ENV POETRY_CACHE_DIR=/opt/.cache

RUN apt update && apt install -y software-properties-common
RUN add-apt-repository -s "$(cat /etc/apt/sources.list | grep -E '^deb(.+)$' | head -1 )" && apt update
# https://llvmlite.readthedocs.io/en/latest/admin-guide/install.html#building-manually
RUN pip install cmake==3.31.2 --upgrade && cmake --version && apt install -y ninja-build \
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

#create virtualenv - poetry works with venv, so no system python
WORKDIR /app
COPY pyproject.toml poetry.lock /app/
RUN poetry env use 3.12 && poetry env list && which python
# https://llvmlite.readthedocs.io/en/latest/admin-guide/install.html#building-manually
# https://github.com/pola-rs/polars#python-compile-polars-from-source
RUN . /app/.venv/bin/activate && which python && pip install -U pip setuptools maturin[patchelf]

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

WORKDIR /app
RUN . /app/.venv/bin/activate && cd /llvm/llvmlite && python setup.py build && python runtests.py && python setup.py install
RUN sed -i "s/polars = /polars-lts-cpu = /" /app/pyproject.toml && cat pyproject.toml
ENV CARGO_BUILD_JOBS=1
ENV RUSTFLAGS="--codegen opt-level=1 --codegen debuginfo=0 --codegen linker-plugin-lto=off"
ENV POETRY_INSTALLER_MAX_WORKERS=1
ENV POETRY_INSTALLER_PARALLEL=false
RUN . /app/.venv/bin/activate && cd /app && . $HOME/.cargo/env \
    && poetry lock && poetry -vv install --no-root && rm -rf $POETRY_CACHE_DIR




FROM python:3.12.9-bullseye AS runtime
RUN apt update && apt install -y libopenblas0 liblapack3 libsndfile1
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=linux.arm.v7-builder /usr/local /usr/local
COPY --from=linux.arm.v7-builder /app/.venv /app/.venv
COPY . /app

# ENV API_HASH
# ENV API_ID
# ENV BOT_TOKEN
# ENV OWNER_USER_ID
WORKDIR /app
ENTRYPOINT ["python"]
CMD ["client.py"]