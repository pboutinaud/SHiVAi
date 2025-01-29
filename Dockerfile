FROM tensorflow/tensorflow:2.17.0-gpu

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/ants-2.4.3/bin
RUN apt update && apt-get install -y build-essential weasyprint libpangocairo-1.0-0 graphviz wget unzip dos2unix

# Installing ANTs
WORKDIR /opt
RUN wget https://github.com/ANTsX/ANTs/releases/download/v2.4.3/ants-2.4.3-ubuntu-20.04-X64-gcc.zip 
RUN unzip ants-2.4.3-ubuntu-20.04-X64-gcc.zip
RUN rm ants-2.4.3-ubuntu-20.04-X64-gcc.zip

# Installing niimath
WORKDIR /usr/local/bin
RUN curl -fLO https://github.com/rordenlab/niimath/releases/latest/download/niimath_lnx.zip
RUN unzip niimath_lnx.zip
RUN rm niimath_lnx.zip

# Installing dcm2niix
RUN curl -fLO https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip
RUN unzip dcm2niix_lnx.zip
RUN rm dcm2niix_lnx.zip

# Installing shivai
COPY src /usr/local/src/shivai/src
COPY tests /usr/local/src/shivai/tests
COPY pyproject.toml /usr/local/src/shivai/

WORKDIR  /usr/local/src/shivai
RUN find . -type f -print0 | xargs -0 dos2unix
RUN chmod 755 src/shivai/scripts/*
RUN python -m pip install build
RUN python -m pip install .
RUN pip cache purge

WORKDIR /root
RUN apt clean
RUN rm -rf /usr/local/src/shivai
RUN mkdir -p /mnt/model

ENTRYPOINT ["shiva", "--containerized_all"]
CMD ["--help"]