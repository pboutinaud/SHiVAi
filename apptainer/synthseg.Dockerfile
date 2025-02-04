FROM tensorflow/tensorflow:2.0.4-gpu-py3
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get -y install wget unzip

WORKDIR /usr/local
COPY synthseg_models.zip ./
RUN unzip synthseg_models.zip && rm synthseg_models.zip

RUN wget https://github.com/BBillot/SynthSeg/archive/refs/heads/master.zip
RUN unzip master.zip && rm master.zip
RUN mv 'synthseg models'/* SynthSeg-master/models/

WORKDIR SynthSeg-master
RUN pip install keras==2.3.1 nibabel==3.2.2 matplotlib==3.3.4
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