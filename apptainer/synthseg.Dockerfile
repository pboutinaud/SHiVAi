# nvidia/cuda base with CUDA 11.2 + cuDNN 8.1 for Ampere (A100, CC 8.0) GPU support.
# The tensorflow/tensorflow:2.4.0-gpu image had a cuDNN mismatch (8.0.4 shipped vs
# 8.1.0 required by the TF 2.4.0 binary), causing "Failed to get convolution algorithm".
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu18.04
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
# Prevent TF from pre-allocating all GPU memory;
# ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV DEBIAN_FRONTEND=noninteractive
# Remove stale NVIDIA apt sources, install system deps and Python
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get -y install python3 python3-pip wget unzip
RUN ln -sf /usr/bin/python3 /usr/local/bin/python
RUN pip3 install --upgrade 'pip<22'
# Install TF 2.4.0 and standalone keras 2.4.0 (SynthSeg uses 'import keras' directly)
RUN pip install tensorflow==2.4.0 keras==2.4.0
# TF 2.4.0 was compiled for CUDA 11.0 (libcusolver.so.10) but CUDA 11.2 ships
# libcusolver.so.11; they are ABI-compatible, so a symlink fixes the lookup.
RUN ln -s /usr/local/cuda/lib64/libcusolver.so.11 /usr/local/cuda/lib64/libcusolver.so.10

WORKDIR /usr/local
COPY SynthSeg_models.zip ./
RUN unzip SynthSeg_models.zip && rm SynthSeg_models.zip

RUN wget https://github.com/BBillot/SynthSeg/archive/refs/heads/master.zip
RUN unzip master.zip && rm master.zip
RUN mv 'SynthSeg_models'/* SynthSeg-master/models/

WORKDIR SynthSeg-master
RUN pip install nibabel==3.2.2 matplotlib==3.3.4
RUN pip install . --no-dependencies

# Creating an executable (python) file to run the command "mri_synthseg"
WORKDIR /usr/local/bin
RUN cp /usr/local/SynthSeg-master/scripts/commands/SynthSeg_predict.py mri_synthseg
# Replace relative path by absolute path (to the original directory)
RUN sed -i 's+os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv\[0\]))))+"/usr/local/SynthSeg-master"+g' mri_synthseg 
# Make it executable as a python script with a shebang
RUN sed -i '1s/^/#!\/usr\/local\/bin\/python\n# -*- coding: utf-8 -*-\n/' mri_synthseg
RUN chmod +x mri_synthseg

# Making standalone Nypipe workflow executable (for fully-contained processing)
COPY precomp_synthseg.py .
RUN chmod +x precomp_synthseg.py
RUN pip install nipype pyyaml

CMD ["precomp_synthseg.py", "--help"]