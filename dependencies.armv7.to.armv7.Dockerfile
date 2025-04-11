# syntax=docker/dockerfile-upstream:master
FROM ghcr.io/rust-cross/manylinux_2_28-cross:armv7 AS dependencies-builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN getconf PAGE_SIZE
RUN apt update && \
    && pip install pip setuptools maturin[patchelf] \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ARG CPU_COUNT=1
ENV CPU_COUNT=$CPU_COUNT

ENV CARGO_BUILD_JOBS=1
ENV RUSTFLAGS="--cfg allocator=\"mimalloc\" --codegen opt-level=1 --codegen debuginfo=0 --codegen linker-plugin-lto=off"

WORKDIR /crates-typos
ENV TARGET_TYPOS_TAG=1.29.0
RUN curl -L https://github.com/crate-ci/typos/archive/refs/tags/v${TARGET_TYPOS_TAG}.tar.gz | tar --absolute-names -xzf - && mv typos-${TARGET_TYPOS_TAG} crates-typos
WORKDIR /crates-typos/crates-typos
RUN . $HOME/.cargo/env && maturin build --compatibility manylinux_2_28 -j${CPU_COUNT} --manifest-path crates/typos-cli/Cargo.toml --profile dev

WORKDIR /polars
ENV TARGET_POLARS_TAG=py-1.27.0
RUN curl -L https://github.com/pola-rs/polars/archive/refs/tags/${TARGET_POLARS_TAG}.tar.gz | tar --absolute-names -xzf - && mv polars-${TARGET_POLARS_TAG} polars
WORKDIR /polars/polars
RUN . $HOME/.cargo/env \
    && sed -i "s/--upgrade --compile-bytecode --no-build/--upgrade --compile-bytecode/" /polars/polars/Makefile \
    && sed -i "s/kuzu//" /polars/polars/py-polars/requirements-dev.txt \
    && sed -i "s/deltalake>=0.15.0//" /polars/polars/py-polars/requirements-dev.txt \
    && cd py-polars && make --debug=b -j${CPU_COUNT} requirements \
    && rustup component add llvm-tools-preview
ENV RUSTFLAGS="${RUSTFLAGS} --cfg allocator=\"mimalloc\""
RUN . $HOME/.cargo/env && maturin build --compatibility manylinux_2_28 -j${CPU_COUNT} --manifest-path py-polars/Cargo.toml




FROM python:3.12.9-bullseye AS tg-zmt-bot_dependencies_armv7
COPY --from=dependencies-builder /polars/polars/target/wheels /wheels
COPY --from=dependencies-builder /crates-typos/crates-typos/target/wheels /wheels
RUN ls /wheels