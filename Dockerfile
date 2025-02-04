FROM tensorflow/tensorflow:2.17.0-gpu

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/ants-2.4.3/bin
RUN apt update && apt-get install -y build-essential libcairo2 libgdk-pixbuf2.0-0 graphviz wget unzip dos2unix weasyprint libpango-1.0-0 libpangoft2-1.0-0 libffi-dev libjpeg-dev libopenjp2-7-dev
# RUN apt update && apt install libharfbuzz-subset0 libharfbuzz0b

# Installing ANTs
WORKDIR /opt
RUN wget https://github.com/ANTsX/ANTs/releases/download/v2.4.3/ants-2.4.3-ubuntu-20.04-X64-gcc.zip && \
    unzip ants-2.4.3-ubuntu-20.04-X64-gcc.zip && \
    rm ants-2.4.3-ubuntu-20.04-X64-gcc.zip

# Installing niimath
WORKDIR /usr/local/bin
RUN curl -fLO https://github.com/rordenlab/niimath/releases/latest/download/niimath_lnx.zip && \
    unzip niimath_lnx.zip && \
    rm niimath_lnx.zip

# Installing dcm2niix
RUN curl -fLO https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip && \
    unzip dcm2niix_lnx.zip && \
    rm dcm2niix_lnx.zip

# Installing shivai
COPY src /usr/local/src/shivai/src
COPY tests /usr/local/src/shivai/tests
COPY pyproject.toml /usr/local/src/shivai/
COPY requirements.txt /usr/local/src/shivai/

WORKDIR  /usr/local/src/shivai
RUN find . -type f -print0 | xargs -0 dos2unix
RUN python -m pip install build && \
    python -m pip install -r requirements.txt && \
    python -m pip cache purge
RUN python -m pip install .

WORKDIR /root
# RUN apt clean
RUN mkdir -p /mnt/model

# ENTRYPOINT ["shiva", "--containerized_all"]
CMD ["shiva", "--help"]