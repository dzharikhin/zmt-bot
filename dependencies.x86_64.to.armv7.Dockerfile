# syntax=docker/dockerfile-upstream:master
FROM ghcr.io/rust-cross/manylinux_2_28-cross:armv7 AS dependencies-builder
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TARGET_ARCH=armv7-unknown-linux
ENV TARGET_ARCH_POSTFIX=gnueabihf

RUN getconf PAGE_SIZE
RUN apt update && apt install -y build-essential
# RUN dpkg --add-architecture armhf && apt update
RUN apt install -y libstdc++-10-dev-armhf-cross clang libclang-dev libunwind-dev \
    && pip install pip setuptools maturin[patchelf] \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ARG CPU_COUNT=4
ENV CPU_COUNT=$CPU_COUNT

# ENV RUSTFLAGS="-C linker=arm-linux-${TARGET_ARCH_POSTFIX}-gcc"

WORKDIR /crates-typos
ENV TARGET_TYPOS_TAG=1.29.0
RUN curl -L https://github.com/crate-ci/typos/archive/refs/tags/v${TARGET_TYPOS_TAG}.tar.gz | tar --absolute-names -xzf - && mv typos-${TARGET_TYPOS_TAG} crates-typos
WORKDIR /crates-typos/crates-typos
RUN . $HOME/.cargo/env && rustup target add ${TARGET_ARCH}-${TARGET_ARCH_POSTFIX}
RUN . $HOME/.cargo/env && maturin build --compatibility manylinux_2_28 --auditwheel=skip -j${CPU_COUNT} --release --manifest-path crates/typos-cli/Cargo.toml --target ${TARGET_ARCH}-${TARGET_ARCH_POSTFIX}

ENV RUST_BACKTRACE=1
ENV CC=arm-linux-${TARGET_ARCH_POSTFIX}-gcc
ENV CXX=""
WORKDIR /polars
ARG CARGO_BUILD_JOBS=4
ENV CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS
ENV TARGET_POLARS_TAG=py-1.27.0
RUN curl -L https://github.com/pola-rs/polars/archive/refs/tags/${TARGET_POLARS_TAG}.tar.gz | tar --absolute-names -xzf - && mv polars-${TARGET_POLARS_TAG} polars
WORKDIR /polars/polars
RUN . $HOME/.cargo/env \
    && sed -i "s/--upgrade --compile-bytecode --no-build/--upgrade --compile-bytecode/" Makefile \
    && sed -i "s/kuzu//" py-polars/requirements-dev.txt \
    && sed -i "s/deltalake>=0.15.0//" py-polars/requirements-dev.txt \
    && cd py-polars && make --debug=b -j${CPU_COUNT} requirements
RUN . $HOME/.cargo/env \
#     && sed -i -E "s/tikv-jemallocator = .+/tcmalloc2 = { version = \"0.2.2\" }/" py-polars/Cargo.toml && cat py-polars/Cargo.toml \
#     && sed -i -E "s/static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;/static GLOBAL: tcmalloc2::TcMalloc = tcmalloc2::TcMalloc;/" py-polars/src/allocator.rs && cat py-polars/src/allocator.rs \
    && rustup target add ${TARGET_ARCH}-${TARGET_ARCH_POSTFIX} \
    && rustup component add llvm-tools-preview
ARG ALLOCATOR=
ENV ALLOCATOR=$ALLOCATOR
ARG JEMALLOC_SYS_WITH_LG_PAGE=15
ENV JEMALLOC_SYS_WITH_LG_PAGE=$JEMALLOC_SYS_WITH_LG_PAGE
ARG JEMALLOC_SYS_WITH_LG_HUGEPAGE=16
ENV JEMALLOC_SYS_WITH_LG_HUGEPAGE=$JEMALLOC_SYS_WITH_LG_HUGEPAGE
RUN . $HOME/.cargo/env && gcc --version && export RUSTFLAGS=$([ -n "${ALLOCATOR}" ] && echo "${RUSTFLAGS} --cfg allocator=\"${ALLOCATOR}\"") && maturin build -v --compatibility manylinux_2_28 --auditwheel=skip -j${CPU_COUNT} --profile debug-release --manifest-path py-polars/Cargo.toml --target ${TARGET_ARCH}-${TARGET_ARCH_POSTFIX}




FROM python:3.12.9-bullseye AS tg-zmt-bot_dependencies_armv7
COPY --from=dependencies-builder /polars/polars/target/wheels /wheels
COPY --from=dependencies-builder /crates-typos/crates-typos/target/wheels /wheels
RUN ls /wheels

