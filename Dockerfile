FROM python:3.12.9-bullseye AS builder

# --- Install Poetry ---
ARG POETRY_VERSION=1.8.5

ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=1
ENV POETRY_VIRTUALENVS_CREATE=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Tell Poetry where to place its cache and virtual environment
ENV POETRY_CACHE_DIR=/opt/.cache

RUN apt update && apt install -y cmake ninja-build && pip install -U pip && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && . $HOME/.cargo/env && pip install "poetry==${POETRY_VERSION}"
WORKDIR /llvm
RUN mkdir -p llvm/ && curl -L https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.7/llvm-project-15.0.7.src.tar.xz | tar -xJf - -C llvm/ --strip-components 1 \
    && mkdir -p llvmlite/ && curl -L https://github.com/numba/llvmlite/archive/refs/tags/v0.44.0.tar.gz | tar -xzf - -C llvmlite/ --strip-components 1 \
    && cd llvm && ls ../llvmlite/conda-recipes/llvm15* | xargs -I{} patch -p1 -i {}
ENV CPU_COUNT=1
RUN cd /llvm/llvm && export PREFIX=/usr/local && bash ../llvmlite/conda-recipes/llvmdev/build.sh
WORKDIR /app

# --- Reproduce the environment ---
# You can comment the following two lines if you prefer to manually install
#   the dependencies from inside the container.
COPY pyproject.toml poetry.lock /app/

# Install the dependencies and clear the cache afterwards.
#   This may save some MBs.
# RUN export POETRY_VIRTUALENVS_OPTIONS_SYSTEM_SITE_PACKAGES=true &&
RUN poetry env use 3.12 && poetry env list
RUN which python && . /app/.venv/bin/activate && which python && pip install -U pip setuptools \
  && cd /llvm/llvmlite && python setup.py build && python runtests.py && python setup.py install
RUN cd /app && poetry -vv install --no-root && rm -rf $POETRY_CACHE_DIR

# Now let's build the runtime image from the builder.
#   We'll just copy the env and the PATH reference.
FROM python:3.12.9-bullseye AS runtime

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