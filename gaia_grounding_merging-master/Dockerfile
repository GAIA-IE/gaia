FROM python:3.6.3

MAINTAINER Dan Napierski (ISI) <dan.napierski@toptal.com>

# Create app directory
WORKDIR /root/src/

# Install app dependencies
COPY environment.txt ./

RUN apt-get update
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-4.5.1-Linux-x86_64.sh
RUN chmod +x ./Miniconda3-4.5.1-Linux-x86_64.sh
RUN ./Miniconda3-4.5.1-Linux-x86_64.sh -b -p ~/conda

RUN echo ". /root/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate" >> ~/.bashrc

ENV PATH="/root/conda/bin:${PATH}"
RUN echo $PATH
RUN conda -V
RUN conda config --add channels conda-forge
RUN conda create --name aida-env --file ./environment.txt 
RUN echo "conda activate aida-env" >> ~/.bashrc

# Bundle app source
COPY . .
# Open port
EXPOSE 8082
CMD [ "/bin/bash", "" ]
