# nvidia/cuda base with CUDA 11.2 + cuDNN 8.1 for Ampere (A100, CC 8.0) GPU support.
# The tensorflow/tensorflow:2.4.0-gpu image had a cuDNN mismatch (8.0.4 shipped vs
# 8.1.0 required by the TF 2.4.0 binary), causing "Failed to get convolution algorithm".
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu18.04
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
# Prevent TF from pre-allocating all GPU memory;
# ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV DEBIAN_FRONTEND=noninteractive
# Remove stale NVIDIA apt sources, install system deps and Python.
# Ubuntu 18.04 (Bionic) is fully EOL; use Canonical's snapshot archive (frozen
# at a pre-EOL date). Check-Valid-Until is disabled because the Release files
# are now expired by design.
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia-ml.list && \
    printf 'deb http://snapshot.ubuntu.com/ubuntu/20230401T000000Z bionic main restricted universe multiverse\n\
deb http://snapshot.ubuntu.com/ubuntu/20230401T000000Z bionic-updates main restricted universe multiverse\n\
deb http://snapshot.ubuntu.com/ubuntu/20230401T000000Z bionic-security main restricted universe multiverse\n' \
        > /etc/apt/sources.list && \
    apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get -y --no-install-recommends install python3 python3-pip wget unzip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/local/bin/python

RUN pip3 install --no-cache-dir --upgrade 'pip<22'
# Install TF 2.4.0 and standalone keras 2.4.0 (SynthSeg uses 'import keras' directly)
RUN pip install --no-cache-dir tensorflow==2.4.0 keras==2.4.0
# TF 2.4.0 was compiled for CUDA 11.0 (libcusolver.so.10) but CUDA 11.2 ships
# libcusolver.so.11; they are ABI-compatible, so a symlink fixes the lookup.
RUN ln -s /usr/local/cuda/lib64/libcusolver.so.11 /usr/local/cuda/lib64/libcusolver.so.10

WORKDIR /usr/local
COPY SynthSeg_models.zip ./
RUN unzip SynthSeg_models.zip && rm SynthSeg_models.zip

RUN wget https://github.com/BBillot/SynthSeg/archive/refs/heads/master.zip && \
    unzip master.zip && rm master.zip && \
    mv SynthSeg_models/* SynthSeg-master/models/

WORKDIR SynthSeg-master
RUN pip install --no-cache-dir nibabel==3.2.2 matplotlib==3.3.4 && \
    pip install --no-cache-dir . --no-dependencies && \
    # Copy the predict script before pruning the source tree
    cp scripts/commands/SynthSeg_predict.py /usr/local/bin/mri_synthseg && \
    # Only the models directory is needed at runtime (mri_synthseg uses the
    # hardcoded path /usr/local/SynthSeg-master); remove everything else.
    find /usr/local/SynthSeg-master -mindepth 1 -maxdepth 1 \
        ! -name models -exec rm -rf {} +

# Patch and activate the mri_synthseg entry point
WORKDIR /usr/local/bin
# Replace relative path by absolute path (to the original directory)
RUN sed -i 's+os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv\[0\]))))+"/usr/local/SynthSeg-master"+g' mri_synthseg && \
    # Make it executable as a python script with a shebang
    sed -i '1s/^/#!\/usr\/local\/bin\/python\n# -*- coding: utf-8 -*-\n/' mri_synthseg && \
    chmod +x mri_synthseg

# Making standalone Nipype workflow executable (for fully-contained processing)
COPY precomp_synthseg.py .
RUN chmod +x precomp_synthseg.py && \
    pip install --no-cache-dir nipype pyyaml

CMD ["precomp_synthseg.py", "--help"]