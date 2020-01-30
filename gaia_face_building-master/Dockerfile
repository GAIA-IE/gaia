FROM nvidia/cuda:10.1-runtime-ubuntu16.04
#FROM python:3.6.5

MAINTAINER Dan Napierski (ISI) <dan.napierski@toptal.com>

# Create app directory
WORKDIR /root/src/

# Update
RUN apt-get update
RUN apt-get install wget
RUN apt-get -y install apt-utils
RUN apt-get install bzip2
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-4.5.1-Linux-x86_64.sh
RUN chmod +x ./Miniconda3-4.5.1-Linux-x86_64.sh
RUN ./Miniconda3-4.5.1-Linux-x86_64.sh -b -p ~/conda
RUN echo ". /root/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate" >> ~/.bashrc
ENV PATH="/root/conda/bin:${PATH}"
RUN echo $PATH
RUN conda update -n base -c defaults conda
RUN conda -V
RUN conda install setuptools
RUN conda install -c anaconda scikit-learn
RUN conda install numpy
RUN conda install cffi
RUN conda install wrapt
RUN conda install absl-py
RUN conda install urllib3
RUN conda install six
#RUN conda install corenlp-python
RUN conda install -c conda-forge spacy-model-en_core_web_sm
#RUN conda update -n base -c defaults conda
#RUN conda config --add channels https://conda.anaconda.org
#RUN conda config --add channels conda-forge
#RUN conda config --add channels https://repo.anaconda.com/pkgs
#RUN conda config --add channels https://repo.anaconda.com/pkgs/r
#RUN conda config --add channels https://anaconda.org/anaconda/repo
#RUN conda config --add channels https://repo.continuum.io
#RUN conda config --add channels https://goodstuff.parc.xerox.com/conda
# copy env definition
COPY brian-env.txt ./
RUN conda create --name brian-env --file ./brian-env.txt 
RUN echo "conda activate brian-env" >> ~/.bashrc

# Bundle app source
COPY . .
# Open port
EXPOSE 8082
CMD [ "/bin/bash", "" ]
