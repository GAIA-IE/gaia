FROM ubuntu:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# entity_api env
ADD ./entity_api /entity_api
RUN /opt/conda/bin/conda env create -f /entity_api/aida_entity.yml

# relation env
ADD ./relation /relation
# relation coarse env
RUN /opt/conda/bin/conda create -n aida_relation_coarse python=3.6 && \
    /opt/conda/envs/aida_relation_coarse/bin/pip install torch==0.4.1 && \
    /opt/conda/envs/aida_relation_coarse/bin/pip install gensim networkx spacy flask && \
    /opt/conda/envs/aida_relation_coarse/bin/python -m spacy download en_core_web_sm
# relation udp env
RUN /opt/conda/bin/conda create -n py27 python=2.7 && \
    /opt/conda/envs/py27/bin/pip install torch==0.4.1 && \
    /opt/conda/envs/py27/bin/pip install -r /relation/udp_requirements.txt
# relation fine-grained
RUN /opt/conda/bin/conda create -n py36 python=3.6 && \
    /opt/conda/envs/py36/bin/pip install torch==1.0.1 && \
    /opt/conda/envs/py36/bin/pip install -r /relation/aida_requirements.txt
#ADD ./download_nltk.py /relation/download_nltk.py
#RUN /opt/conda/envs/py36/bin/python /relation/downloadall.py
#RUN /opt/conda/envs/py36/bin/python -c "import nltk; nltk.download('all')"
RUN /opt/conda/envs/py36/bin/python -m nltk.downloader popular -d /nltk_data

RUN /opt/conda/bin/conda clean -tipsy

LABEL maintainer="hengji@illinois.edu"

CMD ["/bin/bash"]