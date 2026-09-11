FROM tensorflow/tensorflow:2.17.0-gpu AS ants-builder

WORKDIR /opt
RUN apt-get update && apt-get install -y --no-install-recommends wget unzip ca-certificates && \
    wget -q https://github.com/ANTsX/ANTs/releases/download/v2.4.3/ants-2.4.3-ubuntu-20.04-X64-gcc.zip && \
    unzip -q ants-2.4.3-ubuntu-20.04-X64-gcc.zip && \
    rm ants-2.4.3-ubuntu-20.04-X64-gcc.zip && \
    mkdir -p /opt/ants-runtime/bin && \
    cp -a /opt/ants-2.4.3/bin/antsRegistration /opt/ants-runtime/bin/ && \
    cp -a /opt/ants-2.4.3/bin/antsApplyTransforms /opt/ants-runtime/bin/ && \
    if [ -d /opt/ants-2.4.3/lib ]; then cp -a /opt/ants-2.4.3/lib /opt/ants-runtime/lib; fi && \
    ! ldd /opt/ants-runtime/bin/antsRegistration | grep -q 'not found' && \
    ! ldd /opt/ants-runtime/bin/antsApplyTransforms | grep -q 'not found' && \
    rm -rf /var/lib/apt/lists/*


FROM tensorflow/tensorflow:2.17.0-gpu AS tools-builder

WORKDIR /tmp/tools
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip ca-certificates && \
    curl -fsSLO https://github.com/rordenlab/niimath/releases/latest/download/niimath_lnx.zip && \
    unzip -q niimath_lnx.zip && \
    install -Dm755 niimath /opt/tools/bin/niimath && \
    rm -rf /tmp/tools /var/lib/apt/lists/*

WORKDIR /tmp/tools
RUN curl -fsSLO https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip && \
    unzip -q dcm2niix_lnx.zip && \
    install -Dm755 dcm2niix /opt/tools/bin/dcm2niix && \
    rm -rf /tmp/tools


FROM tensorflow/tensorflow:2.17.0-gpu AS app-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libcairo2 libgdk-pixbuf2.0-0 graphviz dos2unix weasyprint \
    libpango-1.0-0 libpangoft2-1.0-0 libffi-dev libjpeg-dev libopenjp2-7-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local/src/shivai
# Separate layers between installing dependencies and copying source code
# to avoid unnecessary rebuilds of the dependencies layer.
COPY requirements.txt ./
RUN dos2unix requirements.txt && \
    python -m venv --system-site-packages /opt/shivai-venv && \
    /opt/shivai-venv/bin/python -m pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY pyproject.toml ./
RUN find . -type f -print0 | xargs -0 dos2unix && \
    /opt/shivai-venv/bin/python -m pip install --no-cache-dir --no-deps .


FROM tensorflow/tensorflow:2.17.0-gpu

ENV PATH=/opt/shivai-venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/ants-2.4.3/bin \
    LD_LIBRARY_PATH=/opt/ants-2.4.3/lib

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 libgdk-pixbuf2.0-0 graphviz fontconfig shared-mime-info weasyprint \
    libpango-1.0-0 libpangoft2-1.0-0 libffi8 libopenjp2-7 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ants-builder /opt/ants-runtime/ /opt/ants-2.4.3/
COPY --from=tools-builder /opt/tools/bin/ /usr/local/bin/
COPY --from=app-builder /opt/shivai-venv/ /opt/shivai-venv/

RUN command -v antsRegistration antsApplyTransforms dcm2niix shiva quickshear && \
    ! ldd "$(command -v antsRegistration)" | grep -q 'not found' && \
    ! ldd "$(command -v antsApplyTransforms)" | grep -q 'not found' && \
    python -c "import tensorflow, keras; from weasyprint import HTML"

WORKDIR /root
RUN mkdir -p /mnt/model

CMD ["shiva", "--help"]