FROM python:3.12-bookworm AS os
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list \
    && curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN apt update
ARG BAZEL_VERSION=3.7.2
RUN apt-get -yq install cmake patchelf curl wget yasm bazel-${BAZEL_VERSION} \
    build-essential \
    libeigen3-dev libyaml-dev libfftw3-dev \
    libsamplerate0-dev libtag1-dev libchromaprint-dev \
    python3-pip python-is-python3 python3-dev python3-numpy-dev python3-numpy python3-yaml python3-six
RUN ln -s /usr/bin/bazel-${BAZEL_VERSION} /usr/bin/bazel
RUN pip install --upgrade pip && pip install auditwheel setuptools requests


FROM os AS sources
ARG TARGET_ESSENTIA_COMMIT=f0f6c358abd133e675710ce2d5f77cc935a75eb9
RUN curl -L "https://github.com/MTG/essentia/archive/${TARGET_ESSENTIA_COMMIT}.tar.gz" | tar xvzf - --transform s/essentia-${TARGET_ESSENTIA_COMMIT}/essentia/


FROM sources AS sources-with-deps
WORKDIR /essentia
# original link is not available from Russia. version is hardcoded. it would be better to update dependency to be able to use github
RUN sed -i 's|SLO http://www.mega-nerd.com/SRC/$LIBSAMPLERATE_VERSION.tar.gz|SL --output $LIBSAMPLERATE_VERSION.tar.gz https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/libsamplerate/0.1.8-7/libsamplerate_0.1.8.orig.tar.gz|' packaging/debian_3rdparty/build_libsamplerate.sh # && cat packaging/debian_3rdparty/build_libsamplerate.sh && exit 1
RUN TF_NEED_CUDA=0 ./packaging/build_3rdparty_static_debian.sh # --with-tensorflow
RUN pip install tensorflow-cpu

FROM sources-with-deps AS builder
WORKDIR /essentia
RUN src/3rdparty/tensorflow/setup_from_libtensorflow.sh
RUN sed -i "/\s*if version.count('-dev'):/,/\s*version += dev_commits/d" setup.py \
    && sed -i "s/macos_arm64_flags,/macos_arm64_flags + ['--with-tensorflow', '--verbose'],/" setup.py # && cat setup.py # && exit 1
WORKDIR /essentia-wheels
RUN ESSENTIA_WHEEL_SKIP_3RDPARTY=1 pip wheel --verbose /essentia/
RUN auditwheel repair $(ls essentia*) -w .
RUN ls -lR

FROM os AS models
COPY . /app
WORKDIR /app/essentia
RUN PYTHONPATH=.. python download_models.py


FROM scratch
COPY --from=builder /essentia-wheels/ /essentia/
COPY --from=models /app/essentia/models/ /essentia/models/
